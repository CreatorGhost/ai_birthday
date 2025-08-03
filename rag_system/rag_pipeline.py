import os
import time
from datetime import datetime, timezone, timedelta
from typing import List, TypedDict, Annotated
from dotenv import load_dotenv

from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.tools import tool

from .config import ModelConfig

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    from pinecone import Pinecone
    from pinecone.core.client.models import ServerlessSpec

load_dotenv()

class RAGState(TypedDict):
    """
    Represents the state of our enhanced RAG graph.
    
    Attributes:
        question: User's question
        chat_history: Previous conversation messages
        documents: Retrieved documents
        generation: LLM generation/answer
        search_needed: Whether web search is needed
        loop_step: Current iteration step
        source_documents: Source documents for the answer
        location: Detected park location (Dalma Mall, Yas Mall, Festival City)
        needs_location_clarification: Whether location clarification is needed
        confidence_score: Confidence score for answer completeness (0.0-1.0)
        needs_human_escalation: Whether human escalation is needed
        enhanced_context: Location-specific enhanced context
    """
    question: str
    chat_history: List[dict]
    documents: List[Document]
    generation: str
    search_needed: str
    loop_step: int
    source_documents: List[Document]
    location: str
    needs_location_clarification: bool
    confidence_score: float
    needs_human_escalation: bool
    enhanced_context: str

class RAGPipeline:
    """Advanced RAG pipeline using LangGraph for complex AI agent workflows"""
    
    def __init__(self):
        # Initialize model configuration
        self.model_config = ModelConfig()
        
        # Get API keys and configuration
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'faq-embeddings')
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize embeddings and LLM using model configuration
        self.embeddings = self.model_config.create_embedding_model()
        self.llm = self.model_config.create_chat_llm()
        
        # LLM for document grading (using same model with temperature 0)
        chat_config = self.model_config.get_chat_model_config().copy()
        chat_config['temperature'] = 0
        
        if self.model_config.llm_provider.value == 'openai':
            from langchain_openai import ChatOpenAI
            self.llm_json_mode = ChatOpenAI(
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                model=chat_config["model_name"],
                temperature=chat_config["temperature"],
                max_tokens=chat_config.get("max_tokens")
            )
        else:  # Google Gemini
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm_json_mode = ChatGoogleGenerativeAI(
                    google_api_key=os.getenv('GOOGLE_API_KEY'),
                    model=chat_config["model_name"],
                    temperature=chat_config["temperature"],
                    max_output_tokens=chat_config.get("max_output_tokens")
                )
            except ImportError:
                raise ImportError(
                    "langchain-google-genai is required for Google Gemini models. "
                    "Install it with: pip install langchain-google-genai"
                )
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.vector_store = None
        self.retriever = None
        self.graph = None
        
        # Setup prompts
        self._setup_prompts()
        
        # Print configuration info
        provider_info = self.model_config.get_provider_info()
        print(f"Initialized RAG Pipeline with:")
        print(f"  Provider: {provider_info['provider']}")
        print(f"  Chat Model: {provider_info['chat_model']}")
        print(f"  Embedding Model: {provider_info['embedding_model']}")
        
    def _setup_prompts(self):
        """Setup all prompts used in the enhanced RAG pipeline"""
        
        # Location detection prompt (enhanced for context awareness)
        self.location_detection_prompt = PromptTemplate(
            template="""Analyze this question and identify which Leo & Loona park location the user is asking about:
            
            Available locations:
            - Dalma Mall (Abu Dhabi) → respond with "Dalma Mall"
            - Yas Mall (Abu Dhabi) → respond with "Yas Mall"
            - Festival City (Dubai) → respond with "Festival City"
            
            Current Question: {question}
            Chat History: {chat_history}
            
            Instructions:
            1. If the current question mentions a specific location (Dalma, Yas, Festival), respond with that location name
            2. If the current question is just a location name (like "Festival City"), respond with that location
            3. If the chat history shows I previously asked for location and user is now providing it, use that location
            4. Look for keywords: "Dalma", "Yas", "Festival", "City", "Mall"
            5. If no location can be determined, respond with "NEEDS_CLARIFICATION"
            
            Examples:
            - "What are prices at Dalma Mall?" → "Dalma Mall"
            - "Festival City" → "Festival City"
            - "Yas" → "Yas Mall"
            - "What are hours?" → "NEEDS_CLARIFICATION"
            
            Response (location name only or NEEDS_CLARIFICATION):""",
            input_variables=["question", "chat_history"]
        )
        
        # Enhanced RAG generation prompt with location, date/time awareness and contextual safety messaging
        self.rag_prompt = PromptTemplate(
            template="""You are Leo & Loona's official FAQ assistant for {location}.
            
            {datetime_context}
            
            Location Context: {location}
            Enhanced Context: {enhanced_context}
            Retrieved Documents: {context}
            Chat History: {chat_history}
            
            Question: {question}
            
            INSTRUCTIONS:
            1. USE THE INFORMATION FROM THE RETRIEVED DOCUMENTS to answer the question
            2. UNDERSTAND DATE/TIME CONTEXT:
               - "Today" refers to the current date and day shown above
               - "Tonight" refers to this evening/night
               - "This weekend" refers to the upcoming Friday-Sunday
               - Use current day type (weekday/friday/weekend) to provide accurate hours
            3. Include safety supervision requirements ONLY when the question is about:
               - Safety rules, park rules, or safety procedures
               - Age restrictions or child activities
               - Play area access or activities
               - When safety information is directly relevant to the question
            4. Provide location-specific information for {location} when available
            5. If some specific details are missing, provide what information IS available from the documents
            6. Be helpful and use ALL relevant information from the context
            7. Keep responses informative but friendly
            8. Don't say you don't know if the documents contain relevant information
            9. For safety-related questions, emphasize: "Adult supervision is required at all times for children's safety"
            10. When providing hours, specify if they apply to "today" or the specific day type
            
            Answer:""",
            input_variables=["context", "question", "chat_history", "location", "enhanced_context", "datetime_context"]
        )
        
        # Location clarification prompt (date-aware)
        self.location_clarification_prompt = PromptTemplate(
            template="""I'd be happy to help you with information about Leo & Loona! 
            
            {datetime_context}
            
            To provide you with accurate details, could you please let me know which location you're asking about?
            
            🏢 Our locations:
            • **Dalma Mall** (Abu Dhabi)
            • **Yas Mall** (Abu Dhabi)
            • **Festival City** (Dubai)
            
            Each location may have different pricing, hours, and specific services available.
            
            Your question: {question}""",
            input_variables=["question", "datetime_context"]
        )
        
        # Document grader instructions (very liberal)
        self.doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question about Leo & Loona play areas.
        
        BE EXTREMELY LIBERAL - mark 'yes' if the document contains:
        - ANY mention of the topic being asked about (hours, prices, safety, etc.)
        - ANY information about Leo & Loona or play areas
        - ANY general FAQ information that might be helpful
        - Location information (even if different location)
        
        Only mark 'no' if the document is completely unrelated to Leo & Loona or play areas.
        
        Respond with only 'yes' or 'no' to indicate whether the document is relevant."""
        
        # Document grader prompt (enhanced)
        self.doc_grader_prompt = """Here is the retrieved document: 
        
        {document} 
        
        Here is the user question: 
        
        {question}
        
        Location context: {location}
        
        Carefully assess whether the document contains information relevant to the question and location.
        
        Respond with only 'yes' or 'no'."""
        
        # Confidence scoring prompt (more lenient)
        self.confidence_scoring_prompt = PromptTemplate(
            template="""Evaluate how completely the retrieved documents can answer this question about Leo & Loona.
            
            Question: {question}
            Location: {location}
            Documents: {documents_summary}
            
            Rate completeness on a scale of 0.0 to 1.0:
            - 1.0: Complete information available (prices, hours, specific details)
            - 0.8: Mostly complete, minor gaps
            - 0.6: Partial information that can provide a helpful response
            - 0.4: Some relevant information available
            - 0.2: Minimal relevant information but still useful
            - 0.0: No relevant information whatsoever
            
            Be GENEROUS in scoring - if documents contain ANY useful information that addresses the question, score at least 0.4.
            Focus on whether we can provide ANY helpful response, not just complete information.
            
            Respond with only the numerical score (e.g., 0.8):""",
            input_variables=["question", "location", "documents_summary"]
        )
        
        # Human escalation prompt
        self.human_escalation_prompt = PromptTemplate(
            template="""I don't have complete information to fully answer your question about {location}.
            
            For accurate and detailed information, I'd recommend contacting our {location} team directly:
            
            📞 **Contact Information:**
            - For bookings and detailed pricing
            - For group events and parties
            - For specific availability questions
            
            🕐 **Our team can help with:**
            - Exact current pricing (including weekday rates)
            - Workshop schedules and costs
            - Group booking arrangements
            - Special dietary accommodations
            - Real-time availability
            
            Your question: "{question}"
            
            Is there anything else I can help you with using the information I do have available?""",
            input_variables=["question", "location"]
        )
    
    def setup_pinecone_index(self, dimension=None):
        """Create or connect to Pinecone index"""
        try:
            # Use model-specific embedding dimensions if not provided
            if dimension is None:
                dimension = self.model_config.get_embedding_dimensions()
            
            # Check if index exists
            if self.index_name not in [index.name for index in self.pc.list_indexes()]:
                print(f"Creating new Pinecone index: {self.index_name} with dimension {dimension}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.pinecone_environment
                    )
                )
                # Wait for index to be ready
                time.sleep(10)
            else:
                print(f"Using existing Pinecone index: {self.index_name}")
            
            return True
        except Exception as e:
            print(f"Error setting up Pinecone index: {str(e)}")
            return False
    
    def create_vector_store(self, documents):
        """Create vector store from documents using Pinecone"""
        try:
            self.vector_store = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                index_name=self.index_name
            )
            
            # Setup retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            print(f"Created vector store with {len(documents)} documents")
            return True
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return False
    
    def load_existing_vector_store(self):
        """Load existing vector store from Pinecone"""
        try:
            self.vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings
            )
            
            # Setup retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            print("Loaded existing vector store")
            return True
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            return False
    
    def _create_retriever_tool(self):
        """Create a retriever tool for the graph"""
        @tool
        def retriever_tool(query: str) -> List[Document]:
            """Retrieve documents from the vector store based on the query."""
            if not self.retriever:
                raise ValueError("Retriever not initialized")
            return self.retriever.invoke(query)
        
        return retriever_tool
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Format documents for context"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def _format_chat_history(self, chat_history: List[dict]) -> str:
        """Format chat history for context"""
        if not chat_history:
            return "No previous conversation."
        
        formatted_history = []
        for msg in chat_history[-5:]:  # Keep last 5 messages for context
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            formatted_history.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted_history)
    
    def _get_current_datetime_info(self) -> dict:
        """Get comprehensive current date/time information"""
        # Get current time in UAE timezone (UTC+4)
        uae_tz = timezone(timedelta(hours=4))
        now_uae = datetime.now(uae_tz)
        
        # Also get UTC time for reference
        now_utc = datetime.now(timezone.utc)
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        weekday = now_uae.weekday()  # 0=Monday, 6=Sunday
        is_weekend = weekday >= 5  # Saturday=5, Sunday=6
        is_friday = weekday == 4   # Friday=4
        
        return {
            'current_datetime_uae': now_uae,
            'current_datetime_utc': now_utc,
            'date_str': now_uae.strftime('%Y-%m-%d'),
            'time_str': now_uae.strftime('%H:%M'),
            'day_name': day_names[weekday],
            'day_number': weekday + 1,  # 1=Monday, 7=Sunday
            'month_name': month_names[now_uae.month - 1],
            'month_number': now_uae.month,
            'year': now_uae.year,
            'is_weekend': is_weekend,
            'is_friday': is_friday,
            'is_weekday': not is_weekend,
            'day_type': 'weekend' if is_weekend else ('friday' if is_friday else 'weekday'),
            'formatted_date': now_uae.strftime('%A, %B %d, %Y'),
            'formatted_time': now_uae.strftime('%I:%M %p'),
        }
    
    def _format_datetime_context(self) -> str:
        """Format current date/time information for LLM context"""
        dt_info = self._get_current_datetime_info()
        
        context = f"""CURRENT DATE & TIME INFORMATION:
📅 Date: {dt_info['formatted_date']}
🕐 Time: {dt_info['formatted_time']} (UAE Time)
📝 Day Type: {dt_info['day_type'].title()}
🗓️ Today is: {dt_info['day_name']}

DAY CLASSIFICATION:
- Weekdays: Monday to Thursday
- Friday: Special day (part of weekend in UAE)
- Weekend: Friday, Saturday, Sunday
- Today is a {dt_info['day_type']}
"""
        return context
    
    def _is_location_followup(self, question: str, chat_history: List[dict]) -> bool:
        """Check if current question is just a location name following a clarification request"""
        if not chat_history:
            return False
        
        # Check if question is just a location name
        question_lower = question.lower().strip()
        location_keywords = ['dalma', 'yas', 'festival', 'city', 'mall']
        
        # Must be short and contain location keywords
        is_short_location = (
            len(question.split()) <= 3 and  # Short response
            any(keyword in question_lower for keyword in location_keywords)
        )
        
        if not is_short_location:
            return False
        
        # Check if last assistant message asked for location
        last_assistant_msg = None
        for msg in reversed(chat_history):
            if msg.get('role') == 'assistant':
                last_assistant_msg = msg.get('content', '').lower()
                break
        
        if last_assistant_msg:
            clarification_phrases = [
                'which location', 'which park', 'please let me know which',
                'could you please specify', 'which of the following locations'
            ]
            return any(phrase in last_assistant_msg for phrase in clarification_phrases)
        
        return False
    
    def _extract_original_question(self, chat_history: List[dict]) -> str:
        """Extract the original question from chat history"""
        # Look for the last user question before clarification
        for msg in reversed(chat_history):
            if msg.get('role') == 'user':
                content = msg.get('content', '').strip()
                # Skip if it's just a location name
                if len(content.split()) > 2:  # More than just location
                    return content
        return ""
    
    # Enhanced Graph Nodes
    
    def detect_location(self, state: RAGState) -> RAGState:
        """
        Detect which Leo & Loona location the user is asking about
        
        Args:
            state: The current graph state
            
        Returns:
            Updated state with location information
        """
        print("---DETECT LOCATION---")
        question = state["question"]
        chat_history = state.get("chat_history", [])
        
        # Format chat history for context
        chat_history_str = self._format_chat_history(chat_history)
        
        # Check if this is a follow-up location response
        is_location_only = self._is_location_followup(question, chat_history)
        
        if is_location_only:
            # User provided location as follow-up, extract the original question
            original_question = self._extract_original_question(chat_history)
            if original_question:
                # Combine original question with location for better processing
                combined_question = f"{original_question} {question}"
                print(f"---COMBINING: '{original_question}' + '{question}'---")
                question = combined_question
        
        # Use LLM to detect location
        location_chain = self.location_detection_prompt | self.llm_json_mode | StrOutputParser()
        location_result = location_chain.invoke({
            "question": question,
            "chat_history": chat_history_str
        })
        
        location_result = location_result.strip()
        
        if location_result == "NEEDS_CLARIFICATION":
            print("---LOCATION: NEEDS CLARIFICATION---")
            return {
                "question": question,
                "chat_history": chat_history,
                "documents": state.get("documents", []),
                "location": "unknown",
                "needs_location_clarification": True,
                "loop_step": state.get("loop_step", 0)
            }
        else:
            print(f"---LOCATION DETECTED: {location_result}---")
            return {
                "question": question,
                "chat_history": chat_history,
                "documents": state.get("documents", []),
                "location": location_result,
                "needs_location_clarification": False,
                "loop_step": state.get("loop_step", 0)
            }
    
    def enhance_context(self, state: RAGState) -> RAGState:
        """
        Enhance context with location-specific information and safety messaging
        
        Args:
            state: The current graph state
            
        Returns:
            Updated state with enhanced context
        """
        print("---ENHANCE CONTEXT---")
        location = state.get("location", "unknown")
        question = state["question"]
        
        # Get current date/time information
        dt_info = self._get_current_datetime_info()
        
        # Determine if question is safety-related
        safety_keywords = ['safety', 'rule', 'supervision', 'age', 'child', 'kid', 'play', 'activity', 'secure', 'safe']
        is_safety_related = any(keyword in question.lower() for keyword in safety_keywords)
        
        # Determine if question is time/date related
        time_keywords = ['today', 'tonight', 'hours', 'open', 'close', 'time', 'schedule', 'when', 'weekend', 'weekday']
        is_time_related = any(keyword in question.lower() for keyword in time_keywords)
        
        # Create enhanced context with location and date/time info
        enhanced_context = f"LOCATION: {location}\n"
        
        # Add date/time context if relevant
        if is_time_related:
            enhanced_context += f"""
CURRENT DATE/TIME CONTEXT:
- Today is {dt_info['day_name']}, {dt_info['formatted_date']}
- Current time: {dt_info['formatted_time']} (UAE)
- Day type: {dt_info['day_type']} 
- Weekend in UAE: Friday, Saturday, Sunday

"""
        
        # Only add safety context if question is safety-related
        if is_safety_related:
            enhanced_context += """
SAFETY REQUIREMENTS:
- Adult supervision is MANDATORY at all times
- Children must be accompanied by adults throughout their visit
- Safety rules apply to all activities and play areas

"""
        
        enhanced_context += "LOCATION-SPECIFIC NOTES:\n"
        
        if "Dalma" in location:
            enhanced_context += "- Dalma Mall location in Abu Dhabi\n- Check for Dalma-specific pricing and hours\n"
        elif "Yas" in location:
            enhanced_context += "- Yas Mall location in Abu Dhabi\n- Special POD holder discounts available\n- 50% discount for kids under 1 year\n"
        elif "Festival" in location:
            enhanced_context += "- Festival City location in Dubai\n- Check for Dubai-specific promotions\n"
        
        return {
            "question": question,
            "chat_history": state.get("chat_history", []),
            "documents": state.get("documents", []),
            "location": location,
            "needs_location_clarification": state.get("needs_location_clarification", False),
            "enhanced_context": enhanced_context,
            "loop_step": state.get("loop_step", 0)
        }
    
    def score_confidence(self, state: RAGState) -> RAGState:
        """
        Score confidence in our ability to provide complete information
        
        Args:
            state: The current graph state
            
        Returns:
            Updated state with confidence score
        """
        print("---SCORE CONFIDENCE---")
        question = state["question"]
        location = state.get("location", "unknown")
        documents = state.get("documents", [])
        
        # If we have any documents, default to reasonable confidence
        if documents:
            # Quick check if documents contain relevant keywords
            doc_text = " ".join([doc.page_content.lower() for doc in documents])
            
            # Keywords that suggest we can answer the question
            answer_keywords = {
                'hours': ['hours', 'open', 'close', 'time', 'monday', 'friday', 'weekend'],
                'price': ['price', 'cost', 'fee', 'aed', 'dirham', 'admission', 'ticket'],
                'safety': ['safety', 'rule', 'supervision', 'adult', 'child'],
                'location': ['mall', 'dalma', 'yas', 'festival', 'city', 'dubai', 'abu dhabi']
            }
            
            # Check if documents contain relevant keywords for the question
            question_lower = question.lower()
            relevance_score = 0.0
            
            for category, keywords in answer_keywords.items():
                if any(kw in question_lower for kw in keywords):
                    # Question is about this category, check if docs have related info
                    if any(kw in doc_text for kw in keywords):
                        relevance_score += 0.3
            
            # If we have relevant content, give decent confidence
            if relevance_score >= 0.3:
                confidence_score = min(0.8, 0.4 + relevance_score)  # At least 0.4, up to 0.8
                print(f"---QUICK CONFIDENCE: {confidence_score} (relevance: {relevance_score})---")
            else:
                # Fall back to LLM scoring
                documents_summary = "\n\n".join([
                    f"Doc {i+1}: {doc.page_content[:200]}..." 
                    for i, doc in enumerate(documents[:3])
                ]) if documents else "No documents retrieved"
        
                # Use LLM to score confidence only if quick check didn't work
                confidence_chain = self.confidence_scoring_prompt | self.llm_json_mode | StrOutputParser()
                confidence_result = confidence_chain.invoke({
                    "question": question,
                    "location": location,
                    "documents_summary": documents_summary
                })
                
                try:
                    confidence_score = float(confidence_result.strip())
                    confidence_score = max(0.4, min(1.0, confidence_score))  # At least 0.4
                except ValueError:
                    print(f"---CONFIDENCE SCORING ERROR: {confidence_result}---")
                    confidence_score = 0.5  # Default to medium confidence
        else:
            # No documents at all
            confidence_score = 0.1
        
        print(f"---CONFIDENCE SCORE: {confidence_score}---")
        
        return {
            "question": question,
            "chat_history": state.get("chat_history", []),
            "documents": documents,
            "location": location,
            "needs_location_clarification": state.get("needs_location_clarification", False),
            "enhanced_context": state.get("enhanced_context", ""),
            "confidence_score": confidence_score,
            "needs_human_escalation": confidence_score < 0.3,
            "loop_step": state.get("loop_step", 0)
        }
    
    def generate_clarification(self, state: RAGState) -> RAGState:
        """
        Generate location clarification question
        
        Args:
            state: The current graph state
            
        Returns:
            Updated state with clarification message
        """
        print("---GENERATE CLARIFICATION---")
        question = state["question"]
        
        # Get current date/time context
        datetime_context = self._format_datetime_context()
        
        # Generate clarification using prompt
        clarification_chain = self.location_clarification_prompt | self.llm | StrOutputParser()
        clarification = clarification_chain.invoke({
            "question": question,
            "datetime_context": datetime_context
        })
        
        return {
            "question": question,
            "chat_history": state.get("chat_history", []),
            "documents": state.get("documents", []),
            "location": state.get("location", "unknown"),
            "needs_location_clarification": True,
            "generation": clarification,
            "source_documents": [],
            "loop_step": state.get("loop_step", 0) + 1
        }
    
    def generate_escalation(self, state: RAGState) -> RAGState:
        """
        Generate human escalation message
        
        Args:
            state: The current graph state
            
        Returns:
            Updated state with escalation message
        """
        print("---GENERATE ESCALATION---")
        question = state["question"]
        location = state.get("location", "Leo & Loona")
        
        # Generate escalation using prompt
        escalation_chain = self.human_escalation_prompt | self.llm | StrOutputParser()
        escalation = escalation_chain.invoke({
            "question": question,
            "location": location
        })
        
        return {
            "question": question,
            "chat_history": state.get("chat_history", []),
            "documents": state.get("documents", []),
            "location": location,
            "needs_human_escalation": True,
            "generation": escalation,
            "source_documents": state.get("documents", []),
            "loop_step": state.get("loop_step", 0) + 1
        }
    
    def _detect_location_from_query(self, question: str) -> str:
        """
        Detect location from user question
        
        Args:
            question: User's question
            
        Returns:
            Detected location or 'unknown'
        """
        question_lower = question.lower()
        
        # Location keywords mapping
        location_keywords = {
            'DALMA_MALL': ['dalma', 'dalma mall'],
            'YAS_MALL': ['yas', 'yas mall', 'yas island'],
            'FESTIVAL_CITY': ['festival', 'festival city', 'dubai festival city'],
            'HELLO_PARK': ['hello', 'hello park']
        }
        
        # Check for exact location mentions
        for location, keywords in location_keywords.items():
            for keyword in keywords:
                if keyword in question_lower:
                    print(f"🎯 Location detected: {location} (keyword: '{keyword}')")
                    return location
        
        return 'unknown'
    
    def _filter_documents_by_location(self, documents: List[Document], target_location: str) -> List[Document]:
        """
        Filter documents by location metadata
        
        Args:
            documents: List of retrieved documents
            target_location: Target location to filter for
            
        Returns:
            Filtered documents from the specified location
        """
        if target_location == 'unknown':
            return documents
        
        location_docs = []
        other_docs = []
        
        for doc in documents:
            doc_location = doc.metadata.get('location', 'UNKNOWN_LOCATION')
            if doc_location == target_location:
                location_docs.append(doc)
            else:
                other_docs.append(doc)
        
        print(f"📍 Location filtering: Found {len(location_docs)} docs from {target_location}, {len(other_docs)} from other locations")
        
        # If we have location-specific docs, prioritize them
        if location_docs:
            # Return location docs + some other docs for context (max 2 other docs)
            return location_docs + other_docs[:2]
        else:
            print(f"⚠️ No documents found for {target_location}, returning all documents")
            return documents
    
    def retrieve_documents(self, state: RAGState) -> RAGState:
        """
        Retrieve documents from vectorstore with location-aware filtering
        
        Args:
            state: The current graph state
            
        Returns:
            Updated state with retrieved documents
        """
        print("---RETRIEVE DOCUMENTS---")
        question = state["question"]
        
        # Detect location from question
        detected_location = self._detect_location_from_query(question)
        
        # Retrieve documents from vector store
        raw_documents = self.retriever.invoke(question)
        print(f"📚 Retrieved {len(raw_documents)} raw documents from vector store")
        
        # Apply location-aware filtering
        filtered_documents = self._filter_documents_by_location(raw_documents, detected_location)
        print(f"🔍 After location filtering: {len(filtered_documents)} documents")
        
        # Update state location if detected
        current_location = state.get("location", "unknown")
        if detected_location != 'unknown':
            current_location = detected_location
        
        return {
            "documents": filtered_documents,
            "question": question,
            "chat_history": state.get("chat_history", []),
            "location": current_location,
            "needs_location_clarification": state.get("needs_location_clarification", False),
            "enhanced_context": state.get("enhanced_context", ""),
            "loop_step": state.get("loop_step", 0)
        }
    
    def grade_documents(self, state: RAGState) -> RAGState:
        """
        Determines whether the retrieved documents are relevant to the question
        
        Args:
            state: The current graph state
            
        Returns:
            Updated state with filtered documents and search decision
        """
        print("---GRADE DOCUMENTS---")
        question = state["question"]
        documents = state["documents"]
        
        # Score each document
        filtered_docs = []
        search_needed = "No"
        
        location = state.get("location", "unknown")
        
        for doc in documents:
            doc_grader_prompt_formatted = self.doc_grader_prompt.format(
                document=doc.page_content, 
                question=question,
                location=location
            )
            
            result = self.llm_json_mode.invoke([
                SystemMessage(content=self.doc_grader_instructions),
                HumanMessage(content=doc_grader_prompt_formatted)
            ])
            
            grade = result.content.strip().lower()
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            elif grade == "no":
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                search_needed = "Yes"
            else:
                print(f"---GRADE: UNEXPECTED RESPONSE '{grade}', KEEPING DOCUMENT---")
                filtered_docs.append(doc)
        
        return {
            "documents": filtered_docs,
            "question": question,
            "chat_history": state.get("chat_history", []),
            "location": location,
            "needs_location_clarification": state.get("needs_location_clarification", False),
            "enhanced_context": state.get("enhanced_context", ""),
            "search_needed": search_needed,
            "loop_step": state.get("loop_step", 0)
        }
    
    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer using RAG on retrieved documents
        
        Args:
            state: The current graph state
            
        Returns:
            Updated state with generated answer
        """
        print("---GENERATE ANSWER---")
        question = state["question"]
        documents = state["documents"]
        chat_history = state.get("chat_history", [])
        location = state.get("location", "Leo & Loona")
        enhanced_context = state.get("enhanced_context", "")
        loop_step = state.get("loop_step", 0)
        
        # Format documents for context
        context = self._format_docs(documents)
        
        # Format chat history for context
        chat_history_str = self._format_chat_history(chat_history)
        
        # Get current date/time context
        datetime_context = self._format_datetime_context()
        
        # Generate answer using enhanced RAG chain
        rag_chain = self.rag_prompt | self.llm | StrOutputParser()
        generation = rag_chain.invoke({
            "context": context,
            "question": question,
            "chat_history": chat_history_str,
            "location": location,
            "enhanced_context": enhanced_context,
            "datetime_context": datetime_context
        })
        
        return {
            "documents": documents,
            "question": question,
            "chat_history": chat_history,
            "location": location,
            "enhanced_context": enhanced_context,
            "generation": generation,
            "source_documents": documents,
            "loop_step": loop_step + 1
        }
    
    # Enhanced Routing Methods
    
    def route_after_location_detection(self, state: RAGState) -> str:
        """
        Route after location detection
        
        Args:
            state: The current graph state
            
        Returns:
            Next node to call
        """
        needs_clarification = state.get("needs_location_clarification", False)
        
        if needs_clarification:
            print("---ROUTE: NEEDS LOCATION CLARIFICATION---")
            return "clarification"
        else:
            print("---ROUTE: PROCEED TO RETRIEVAL---")
            return "retrieve"
    
    def route_after_confidence_scoring(self, state: RAGState) -> str:
        """
        Route after confidence scoring
        
        Args:
            state: The current graph state
            
        Returns:
            Next node to call
        """
        needs_escalation = state.get("needs_human_escalation", False)
        confidence_score = state.get("confidence_score", 0.5)
        
        # Lower threshold for escalation - be more willing to try answering
        if needs_escalation or confidence_score < 0.3:
            print(f"---ROUTE: ESCALATE TO HUMAN (confidence: {confidence_score})---")
            return "escalation"
        else:
            print(f"---ROUTE: GENERATE ANSWER (confidence: {confidence_score})---")
            return "generate"
    
    def decide_to_generate(self, state: RAGState) -> str:
        """
        Determines whether to generate an answer or need more search
        
        Args:
            state: The current graph state
            
        Returns:
            Next node to call
        """
        search_needed = state.get("search_needed", "No")
        
        if search_needed == "Yes":
            print("---DECISION: DOCUMENTS NOT RELEVANT, PROCEED TO CONTEXT ENHANCEMENT---")
            return "enhance_context"
        else:
            print("---DECISION: PROCEED TO CONTEXT ENHANCEMENT---")
            return "enhance_context"
    
    def setup_graph(self):
        """Setup the enhanced LangGraph workflow"""
        if not self.retriever:
            raise ValueError("Vector store and retriever must be initialized first")
        
        # Create the enhanced graph
        workflow = StateGraph(RAGState)
        
        # Add all nodes
        workflow.add_node("detect_location", self.detect_location)
        workflow.add_node("retrieve", self.retrieve_documents)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("enhance_context", self.enhance_context)
        workflow.add_node("score_confidence", self.score_confidence)
        workflow.add_node("generate", self.generate_answer)
        workflow.add_node("clarification", self.generate_clarification)
        workflow.add_node("escalation", self.generate_escalation)
        
        # Define the enhanced workflow
        # Start with location detection
        workflow.add_edge(START, "detect_location")
        
        # Route after location detection
        workflow.add_conditional_edges(
            "detect_location",
            self.route_after_location_detection,
            {
                "clarification": "clarification",
                "retrieve": "retrieve"
            }
        )
        
        # Clarification ends the conversation
        workflow.add_edge("clarification", END)
        
        # Normal flow: retrieve -> grade -> enhance context
        workflow.add_edge("retrieve", "grade_documents")
        
        # After grading, enhance context
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "enhance_context": "enhance_context"
            }
        )
        
        # After enhancing context, score confidence
        workflow.add_edge("enhance_context", "score_confidence")
        
        # Route based on confidence score
        workflow.add_conditional_edges(
            "score_confidence",
            self.route_after_confidence_scoring,
            {
                "generate": "generate",
                "escalation": "escalation"
            }
        )
        
        # Both generate and escalation end the conversation
        workflow.add_edge("generate", END)
        workflow.add_edge("escalation", END)
        
        # Compile the graph
        self.graph = workflow.compile()
        
        print("Enhanced LangGraph RAG workflow setup complete")
        print("Flow: detect_location -> [clarification | retrieve] -> grade -> enhance_context -> score_confidence -> [generate | escalation]")
    
    def answer_question(self, question: str, chat_history: List[dict] = None) -> dict:
        """
        Answer a question using the LangGraph RAG pipeline
        
        Args:
            question: User's question
            chat_history: Previous conversation messages for context
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.graph:
            raise ValueError("Graph not initialized. Call setup_graph() first.")
        
        if chat_history is None:
            chat_history = []
        
        try:
            # Enhanced initial state
            initial_state = {
                "question": question,
                "chat_history": chat_history,
                "documents": [],
                "generation": "",
                "search_needed": "No",
                "loop_step": 0,
                "source_documents": [],
                "location": "unknown",
                "needs_location_clarification": False,
                "confidence_score": 0.0,
                "needs_human_escalation": False,
                "enhanced_context": ""
            }
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            return {
                "answer": result.get("generation", "No answer generated"),
                "source_documents": result.get("source_documents", [])
            }
            
        except Exception as e:
            return {
                "answer": f"Error processing question: {str(e)}",
                "source_documents": []
            }
    
    def get_graph_visualization(self):
        """Get a visualization of the graph structure"""
        if not self.graph:
            raise ValueError("Graph not initialized")
        
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as e:
            return f"Error generating graph visualization: {str(e)}"