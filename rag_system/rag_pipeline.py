import os
import time
import random
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
from .user_tracker import UserTracker

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
        
        # User Information Tracking
        user_phone: str                   # Simulated phone number for testing
        user_name: str                    # Extracted or provided user name
        user_profile: dict                # Complete user profile from storage
        name_extraction_result: dict      # Result of name extraction attempt
        should_request_name: bool         # Whether to ask for name
        name_request_message: str         # Generated name request message
        conversation_logged: bool         # Whether conversation was logged
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
    
    # User Information Tracking
    user_phone: str
    user_name: str
    user_profile: dict
    name_extraction_result: dict
    should_request_name: bool
    name_request_message: str
    conversation_logged: bool

class RAGPipeline:
    """Advanced RAG pipeline using LangGraph for complex AI agent workflows"""
    
    def __init__(self):
        # Initialize model configuration
        self.model_config = ModelConfig()
        
        # Available locations - configured here for consistency
        self.available_locations = [
            {"name": "Dalma Mall", "city": "Abu Dhabi"},
            {"name": "Yas Mall", "city": "Abu Dhabi"},
            {"name": "Festival City Mall", "city": "Dubai"}
        ]
        
        # Get API keys and configuration
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.pinecone_environment = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'faq-embeddings')
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize embeddings and LLM using model configuration
        self.embeddings = self.model_config.create_embedding_model()
        self.llm = self.model_config.create_chat_llm()
        
        # Initialize user tracking for name collection and storage
        self.user_tracker = UserTracker(self.llm)
        
        # Initialize Bitrix lead manager
        try:
            from bitrix_integration.lead_manager import LeadManager
            self.lead_manager = LeadManager()
        except Exception as e:
            print(f"Warning: Bitrix integration not available - {e}")
            self.lead_manager = None
        
        # LLM for document grading (using same model)
        chat_config = self.model_config.get_chat_model_config().copy()
        
        if self.model_config.llm_provider.value == 'openai':
            from langchain_openai import ChatOpenAI
            self.llm_json_mode = ChatOpenAI(
                openai_api_key=os.getenv('OPENAI_API_KEY'),
                model=chat_config["model_name"],
                max_tokens=chat_config.get("max_tokens")
            )
        else:  # Google Gemini
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm_json_mode = ChatGoogleGenerativeAI(
                    google_api_key=os.getenv('GOOGLE_API_KEY'),
                    model=chat_config["model_name"],
                    temperature=0,
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
            5. For GENERAL questions (hours, prices, safety rules, booking info, etc.) that don't specify location, respond with "GENERAL" - the system can provide general information or information from all locations
            6. Only respond with "NEEDS_CLARIFICATION" if the question specifically requires location-specific information that varies significantly between locations AND no location is mentioned
            
            Examples:
            - "What are prices at Dalma Mall?" → "Dalma Mall"
            - "Festival City" → "Festival City"
            - "Yas" → "Yas Mall"
            - "What are your opening hours?" → "GENERAL"
            - "What are your prices?" → "GENERAL"
            - "What are your safety rules?" → "GENERAL"
            - "Can I book a birthday party?" → "GENERAL"
            - "Do you have parking specific to your mall location?" → "NEEDS_CLARIFICATION"
            
            Response (location name, "GENERAL", or "NEEDS_CLARIFICATION"):""",
            input_variables=["question", "chat_history"]
        )
        
        # Enhanced RAG generation prompt with Leo & Loona personality (matching exact tone from markdown)
        self.rag_prompt = PromptTemplate(
            template="""You are a warm, friendly, and knowledgeable virtual host of a magical family amusement park{location_context}. You speak with genuine enthusiasm and a caring tone, like a host who greets guests at the park entrance. You understand both the excitement of children and the practical questions of parents.

            ## Role & Personality
            You are a warm, friendly, and knowledgeable virtual host of a magical family amusement park. You speak with genuine enthusiasm and a caring tone, like a host who greets guests at the park entrance. You understand both the excitement of children and the practical questions of parents.

            ## Tone & Style
            - Warm, joyful, and professional.  
            - Use simple, friendly language without overcomplicating explanations.  
            - Include light, natural compliments when appropriate (e.g., "Oh, Anna – what a beautiful name!").  
            - Use a maximum of 2 emojis per message, and only when they enhance the warmth or excitement.  
            - Create a sense of anticipation and joy ("Oh! That means you're planning to visit us — wonderful choice, we can't wait to see you!").  
            - Be informative first, but wrap the information in a pleasant and engaging tone.

            ## Special Behavior — Promotions & News
            - When answering, if the topic or timing is relevant, *naturally* mention current offers, events, or news.
            - Example: If someone asks about visiting now, you can say:  
              "Actually, right now we're running our **Educational Months** project — if you're planning to visit soon, you might enjoy it! Would you like me to send you the daily schedule so you can plan ahead?"  
            - The offer should feel helpful and inviting, never pushy.

            ## Goals
            - Give accurate, clear, and helpful answers.  
            - Make guests feel welcome and valued.  
            - Encourage engagement with current offers/events.

            ## Restrictions
            - Never overuse emojis or exclamation marks.  
            - Avoid generic or robotic responses — always add a touch of personality.  
            - Do not pressure the guest into offers; only suggest them if relevant.
            - ONLY answer questions about Leo & Loona amusement park (locations, pricing, hours, safety, activities, bookings, etc.)
            - If asked about anything NOT related to Leo & Loona, politely redirect: "I'm here to help with questions about Leo & Loona! Is there anything you'd like to know about our magical play areas?"
            
            {datetime_context}
            
            Location Context: {location}
            Enhanced Context: {enhanced_context}
            Retrieved Documents: {context}
            Chat History: {chat_history}
            
            Question: {question}
            
            INSTRUCTIONS:
            1. FIRST check if the question is about Leo & Loona - if not, politely redirect
            2. USE THE INFORMATION FROM THE RETRIEVED DOCUMENTS to answer Leo & Loona questions
            3. ALWAYS reply as a warm, welcoming, and joyful park host
            4. Provide clear, useful information, sprinkle in warmth and compliments where natural
            5. Gently inform guests about current events or offers when relevant (like Educational Months)
            6. UNDERSTAND DATE/TIME CONTEXT:
               - "Today" refers to the current date and day shown above
               - "Tonight" refers to this evening/night
               - "This weekend" refers to the upcoming Saturday-Sunday
               - Use current day type (weekday/weekend) to provide accurate hours
            7. Include safety supervision requirements ONLY when the question is about:
               - Safety rules, park rules, or safety procedures
               - Age restrictions or child activities
               - Play area access or activities
               - When safety information is directly relevant to the question
            8. If location is "General", provide comprehensive information from all available locations, clearly noting when information varies by location
            9. If location is specific (Dalma Mall, Yas Mall, Festival City), focus on that location's information
            10. Express genuine excitement about their visit plans: "Oh! That means you're planning to visit us — wonderful choice, we can't wait to see you!"
            11. For safety-related questions, emphasize with warmth: "Adult supervision is required at all times for children's safety"
            12. Never use more than 2 emojis or excessive exclamation marks
            13. When you detect names in conversation, compliment them naturally: "Oh, [Name] – what a beautiful name!"
            
            Answer as Leo & Loona's warm, welcoming park host following the exact tone and style described above:""",
            input_variables=["context", "question", "chat_history", "location", "enhanced_context", "datetime_context", "location_context"]
        )
        
        # Location clarification prompt with Leo & Loona personality (matching exact tone from markdown)
        self.location_clarification_prompt = PromptTemplate(
            template="""Oh! So you're planning to visit us — that's wonderful! 🎉 I'd be delighted to help you with information about Leo & Loona!
            
            {datetime_context}
            
            To give you the most accurate details, could you please let me know which of our magical locations you're asking about?
            
            🏢 Our Leo & Loona locations:
            • **Dalma Mall** (Abu Dhabi)
            • **Yas Mall** (Abu Dhabi) 
            • **Festival City** (Dubai)
            
            Each location is special in its own way, and some details like pricing and hours may vary.
            
            Your question: "{question}"
            
            By the way, we're currently celebrating **Educational Months** — would you like me to send you the daily schedule so you can plan ahead?
            
            We can't wait to welcome you to Leo & Loona! ✨""",
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
        
        # Human escalation prompt with Leo & Loona warmth (matching exact tone from markdown)
        self.human_escalation_prompt = PromptTemplate(
            template="""Oh! I wish I had more detailed information to fully answer your wonderful question about {location}! 🎉
            
            For the most accurate and up-to-date details, I'd love to connect you with our amazing {location} team directly:
            
            📞 **Our friendly team can help with:**
            - Exact current pricing and special rates
            - Booking arrangements and availability
            - Group events and magical birthday parties
            - Workshop schedules and special activities
            - Any specific accommodations you might need
            
            Your question: "{question}"
            
            Actually, right now we're running our **Educational Months** project — if you're planning to visit soon, you might enjoy it! Would you like me to send you the daily schedule so you can plan ahead?
            
            Is there anything else about Leo & Loona that I can help you with using the information I do have? We're always here to make your visit as magical as possible! ✨""",
            input_variables=["question", "location"]
        )
    
    def setup_pinecone_index(self, dimension=None):
        """Create or connect to Pinecone index with retry logic"""
        max_retries = 3
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"Setting up Pinecone index (attempt {attempt + 1}/{max_retries})...")
                
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
                    # Wait for index to be ready with incremental checks
                    print("⏳ Waiting for index to be ready...")
                    for i in range(6):  # Check 6 times over 30 seconds
                        time.sleep(5)
                        try:
                            index_status = self.pc.describe_index(self.index_name)
                            if index_status.status.ready:
                                print("✅ Index is ready")
                                break
                        except:
                            continue
                    else:
                        print("⚠️ Index creation taking longer than expected, but continuing...")
                else:
                    print(f"Using existing Pinecone index: {self.index_name}")
                
                return True
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a connection/SSL error that we should retry
                if any(keyword in error_str for keyword in ['ssl', 'connection', 'timeout', 'eof', 'retry']):
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"⚠️ Connection error setting up index: {str(e)}")
                        print(f"🔄 Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"❌ Max retries exceeded setting up index: {str(e)}")
                        return False
                else:
                    # Non-retryable error
                    print(f"❌ Error setting up Pinecone index: {str(e)}")
                    return False
        
        return False
    
    def create_vector_store(self, documents):
        """Create vector store from documents using Pinecone with retry logic and batch processing"""
        max_retries = 5
        base_delay = 2
        
        # Try smaller batches if we have many documents
        batch_size = min(50, len(documents))  # Start with smaller batches
        
        for attempt in range(max_retries):
            try:
                print(f"Creating vector store (attempt {attempt + 1}/{max_retries})...")
                print(f"Processing {len(documents)} documents in batches of {batch_size}...")
                
                # For large document sets, try batch processing
                if len(documents) > batch_size and attempt > 0:
                    print(f"🔄 Trying batch processing with batch size {batch_size}...")
                    
                    # Create vector store with first batch
                    first_batch = documents[:batch_size]
                    self.vector_store = PineconeVectorStore.from_documents(
                        documents=first_batch,
                        embedding=self.embeddings,
                        index_name=self.index_name
                    )
                    
                    # Add remaining documents in batches
                    for i in range(batch_size, len(documents), batch_size):
                        batch = documents[i:i + batch_size]
                        print(f"Adding batch {i//batch_size + 1} ({len(batch)} documents)...")
                        self.vector_store.add_documents(batch)
                        time.sleep(1)  # Small delay between batches
                else:
                    # Try to process all documents at once
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
                
                print(f"✅ Created vector store with {len(documents)} documents")
                return True
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a connection/SSL error that we should retry
                if any(keyword in error_str for keyword in ['ssl', 'connection', 'timeout', 'eof', 'retry', 'rate limit']):
                    if attempt < max_retries - 1:
                        # Exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        print(f"⚠️ Connection error (attempt {attempt + 1}): {str(e)}")
                        print(f"🔄 Retrying in {delay:.1f} seconds...")
                        
                        # Reduce batch size for next attempt
                        if len(documents) > 10:
                            batch_size = max(10, batch_size // 2)
                            print(f"📉 Reducing batch size to {batch_size} for next attempt")
                        
                        time.sleep(delay)
                        continue
                    else:
                        print(f"❌ Max retries exceeded. Final error: {str(e)}")
                        print("💡 Suggestion: Try running the ingestion again, or check your internet connection")
                        return False
                else:
                    # Non-retryable error
                    print(f"❌ Error creating vector store: {str(e)}")
                    return False
        
        return False
    
    def load_existing_vector_store(self):
        """Load existing vector store from Pinecone with retry logic"""
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries):
            try:
                print(f"Loading existing vector store (attempt {attempt + 1}/{max_retries})...")
                
                self.vector_store = PineconeVectorStore(
                    index_name=self.index_name,
                    embedding=self.embeddings
                )
                
                # Setup fast retriever with fewer documents for speed
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 3}  # Reduced from 5 to 3 for speed
                )
                
                print("✅ Loaded existing vector store")
                return True
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check if it's a connection/SSL error that we should retry
                if any(keyword in error_str for keyword in ['ssl', 'connection', 'timeout', 'eof', 'retry']):
                    if attempt < max_retries - 1:
                        # Shorter delay for loading existing store
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 0.5)
                        print(f"⚠️ Connection error loading vector store: {str(e)}")
                        print(f"🔄 Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        print(f"❌ Max retries exceeded loading vector store: {str(e)}")
                        return False
                else:
                    # Non-retryable error
                    print(f"❌ Error loading vector store: {str(e)}")
                    return False
        
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
    
    def _detect_names_in_conversation(self, question: str, chat_history: List[dict]) -> str:
        """Detect names mentioned in the conversation for natural compliments"""
        import re
        
        # Common name patterns - look for "my [relation] is/name is [Name]" or "daughter [Name]" etc.
        name_patterns = [
            r"my\s+(?:daughter|son|child|kid)\s+(?:is\s+)?(?:called\s+|named\s+)?([A-Z][a-z]+)",
            r"(?:daughter|son|child|kid)\s+(?:is\s+)?(?:called\s+|named\s+)?([A-Z][a-z]+)",
            r"(?:name\s+is|called|named)\s+([A-Z][a-z]+)",
            r"my\s+name\s+is\s+([A-Z][a-z]+)",
            r"I'm\s+([A-Z][a-z]+)",
            r"this\s+is\s+([A-Z][a-z]+)"
        ]
        
        # Check current question and recent chat history
        all_text = question + " "
        for msg in chat_history[-3:]:  # Check last 3 messages
            if msg.get("role") == "user":
                all_text += msg.get("content", "") + " "
        
        detected_names = []
        for pattern in name_patterns:
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            for match in matches:
                if match and len(match) > 1 and match.isalpha():  # Valid name
                    detected_names.append(match.title())
        
        # Return the most recent unique name
        unique_names = list(dict.fromkeys(detected_names))  # Remove duplicates while preserving order
        return unique_names[-1] if unique_names else None
    
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
        is_weekend = weekday >= 6  # Saturday=5, Sunday=6 - Changed to only include Sat/Sun
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
            'day_type': 'weekend' if is_weekend else 'weekday',
            'formatted_date': now_uae.strftime('%A, %B %d, %Y'),
            'formatted_time': now_uae.strftime('%I:%M %p'),
        }
    
    def _format_datetime_context(self) -> str:
        """Format current date/time information for LLM context"""
        dt_info = self._get_current_datetime_info()
        
        context = f"""CURRENT DATE & TIME INFORMATION (UAE TIMEZONE):
📅 Date: {dt_info['formatted_date']} 
🕐 Time: {dt_info['formatted_time']} (UAE Time - UTC+4)
📝 Day Type: {dt_info['day_type'].title()}
🗓️ Today is: {dt_info['day_name']}

UAE DAY CLASSIFICATION:
- Weekdays: Monday to Friday
- Weekend: Saturday, Sunday
- Today is a {dt_info['day_type']}

TIMEZONE NOTE: All times are in UAE Standard Time (UTC+4)
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
        elif location_result == "GENERAL":
            print("---LOCATION: GENERAL QUESTION---")
            return {
                "question": question,
                "chat_history": chat_history,
                "documents": state.get("documents", []),
                "location": "General",
                "needs_location_clarification": False,
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
- Weekend in UAE: Saturday, Sunday

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
        Generate intelligent clarification message using LLM
        """
        print("---GENERATE CLARIFICATION---")
        question = state["question"]
        chat_history = state.get("chat_history", [])
        
        # Format chat history for context
        chat_history_str = self._format_chat_history(chat_history)
        
        # Get current date/time context for more natural responses
        datetime_context = self._format_datetime_context()
        
        # Format available locations for the prompt
        locations_list = "\n".join([
            f"{i+1}️⃣ {loc['name']} ({loc['city']})" 
            for i, loc in enumerate(self.available_locations)
        ])
        
        # Use LLM to generate contextual clarification with predefined locations
        clarification_prompt = PromptTemplate(
            template="""You are Leo & Loona's FAQ assistant. The user asked a question that requires location-specific information.

{datetime_context}

User's Question: "{question}"
Chat History: {chat_history}

Available Leo & Loona Locations:
{locations_list}

Instructions:
1. Start with a warm greeting: "Thanks for your interest in Leo & Loona! 🎉"
2. Acknowledge their specific question topic (extract the main topic from "{question}")
3. Explain briefly that you need location information because details may vary by location
4. Present ALL the available locations in this exact format:
   "📍 Available Locations:
   {locations_list}"
5. Ask them to specify which location they're interested in
6. Keep it friendly, concise, and well-formatted
7. Always show ALL locations, not just some

Example format:
"Thanks for your interest in Leo & Loona! 🎉
To help you with [question topic], could you please let us know which location you're referring to?

📍 Available Locations:
{locations_list}

Please let us know which location you're interested in!"

Generate the clarification message following this exact format:""",
            input_variables=["question", "chat_history", "datetime_context", "locations_list"]
        )
        
        # Get documents context for location information
        documents = state.get("documents", [])
        documents_context = self._format_docs(documents) if documents else "No specific documents retrieved yet."
        
        try:
            clarification_chain = clarification_prompt | self.llm | StrOutputParser()
            clarification = clarification_chain.invoke({
                "question": question,
                "chat_history": chat_history_str,
                "datetime_context": datetime_context,
                "locations_list": locations_list
            })
        except Exception as e:
            print(f"---CLARIFICATION GENERATION ERROR: {e}---")
            # Static fallback with predefined locations
            print(f"---USING STATIC FALLBACK WITH PREDEFINED LOCATIONS---")
            clarification = f"""Thanks for your interest in Leo & Loona! 🎉
To help you with '{question}', could you please let us know which location you're referring to?

📍 Available Locations:
{locations_list}

Please let us know which location you're interested in!"""
        
        return {
            "question": question,
            "chat_history": chat_history,
            "documents": state.get("documents", []),
            "generation": clarification,
            "source_documents": [],
            "loop_step": 1
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
        Simple document retrieval from vectorstore
        """
        print("---RETRIEVE DOCUMENTS---")
        question = state["question"]
        
        # Retrieve documents from vector store
        documents = self.retriever.invoke(question)
        print(f"📚 Retrieved {len(documents)} documents from vector store")
        
        return {
            "documents": documents,
            "question": question,
            "chat_history": state.get("chat_history", []),
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
        
        # Detect names for natural compliments
        detected_name = self._detect_names_in_conversation(question, chat_history)
        
        # Format documents for context
        context = self._format_docs(documents)
        
        # Format chat history for context
        chat_history_str = self._format_chat_history(chat_history)
        
        # Get current date/time context
        datetime_context = self._format_datetime_context()
        
        # Create location context string for the prompt
        if location == "General":
            location_context = " across all locations"
        else:
            location_context = f" for {location}"
        
        # Add name context if detected
        name_context = ""
        if detected_name:
            name_context = f"\n\nNAME DETECTED: {detected_name} - Remember to compliment this name naturally if appropriate (e.g., 'Oh, {detected_name} – what a beautiful name!')."
        
        # Generate answer using enhanced RAG chain
        rag_chain = self.rag_prompt | self.llm | StrOutputParser()
        generation = rag_chain.invoke({
            "context": context + name_context,
            "question": question,
            "chat_history": chat_history_str,
            "location": location,
            "enhanced_context": enhanced_context,
            "datetime_context": datetime_context,
            "location_context": location_context
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
    
    def generate_simplified_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer with simplified logic combining location detection, context enhancement, and answer generation
        """
        print("---GENERATE SIMPLIFIED ANSWER---")
        question = state["question"]
        documents = state["documents"]
        chat_history = state.get("chat_history", [])
        
        # Simple location detection from question
        location = self._detect_location_simple(question)
        
        # Detect names for natural compliments
        detected_name = self._detect_names_in_conversation(question, chat_history)
        
        # Format documents for context
        context = self._format_docs(documents)
        
        # Format chat history for context
        chat_history_str = self._format_chat_history(chat_history)
        
        # Get current date/time context
        datetime_context = self._format_datetime_context()
        
        # Create location context string
        if location == "General":
            location_context = " across all locations"
        else:
            location_context = f" for {location}"
        
        # Simple enhanced context
        enhanced_context = f"LOCATION: {location}\n"
        if any(keyword in question.lower() for keyword in ['safety', 'rule', 'supervision', 'age', 'child']):
            enhanced_context += "SAFETY: Adult supervision is required at all times\n"
        
        # Add name context if detected
        name_context = ""
        if detected_name:
            name_context = f"\n\nNAME DETECTED: {detected_name} - Remember to compliment this name naturally if appropriate (e.g., 'Oh, {detected_name} – what a beautiful name!')."
        
        # Generate answer using simplified prompt with Leo & Loona personality (matching exact tone from markdown)
        simplified_prompt = PromptTemplate(
            template="""You are a warm, friendly, and knowledgeable virtual host of a magical family amusement park{location_context}. You speak with genuine enthusiasm and a caring tone, like a host who greets guests at the park entrance. You understand both the excitement of children and the practical questions of parents.

            ## Tone & Style
            - Warm, joyful, and professional.  
            - Use simple, friendly language without overcomplicating explanations.  
            - Include light, natural compliments when appropriate (e.g., "Oh, Anna – what a beautiful name!").  
            - Use a maximum of 2 emojis per message, and only when they enhance the warmth or excitement.  
            - Create a sense of anticipation and joy ("Oh! That means you're planning to visit us — wonderful choice, we can't wait to see you!").  
            - Be informative first, but wrap the information in a pleasant and engaging tone.

            ## Special Behavior — Promotions & News
            - When answering, if the topic or timing is relevant, *naturally* mention current offers, events, or news.
            - Example: If someone asks about visiting now, you can say:  
              "Actually, right now we're running our **Educational Months** project — if you're planning to visit soon, you might enjoy it! Would you like me to send you the daily schedule so you can plan ahead?"  
            - The offer should feel helpful and inviting, never pushy.
            
            ## Restrictions
            - Never overuse emojis or exclamation marks.  
            - Avoid generic or robotic responses — always add a touch of personality.  
            - Do not pressure the guest into offers; only suggest them if relevant.
            - ONLY answer questions about Leo & Loona. For off-topic questions, politely redirect: "I'm here to help with questions about Leo & Loona! Is there anything you'd like to know about our magical play areas?"
            
            {datetime_context}
            
            Retrieved Information: {context}
            Chat History: {chat_history}
            Question: {question}
            
            Instructions:
            1. FIRST check if question is about Leo & Loona - if not, politely redirect
            2. Answer using retrieved information with warm, enthusiastic tone exactly as described above
            3. Always reply as a warm, welcoming, and joyful park host
            4. Provide clear, useful information, sprinkle in warmth and compliments where natural
            5. Gently inform guests about current events or offers when relevant (like Educational Months)
            6. If location is "General", provide comprehensive information from all locations
            7. Include safety reminders only when relevant: "Adult supervision is required at all times for children's safety"
            8. Use current date/time context when relevant
            9. Express excitement about their visit plans: "Oh! That means you're planning to visit us — wonderful choice, we can't wait to see you!"
            10. When you detect names, compliment them naturally: "Oh, [Name] – what a beautiful name!"
            
            Answer as Leo & Loona's welcoming park host following the exact tone and personality described above:""",
            input_variables=["context", "question", "chat_history", "datetime_context", "location_context"]
        )
        
        rag_chain = simplified_prompt | self.llm | StrOutputParser()
        generation = rag_chain.invoke({
            "context": context + name_context,
            "question": question,
            "chat_history": chat_history_str,
            "datetime_context": datetime_context,
            "location_context": location_context
        })
        
        return {
            "documents": documents,
            "question": question,
            "chat_history": chat_history,
            "location": location,
            "enhanced_context": enhanced_context,
            "generation": generation,
            "source_documents": documents,
            "loop_step": 1
        }
    
    def _detect_location_simple(self, question: str) -> str:
        """Simple location detection without LLM"""
        question_lower = question.lower()
        
        if any(keyword in question_lower for keyword in ['dalma']):
            return "Dalma Mall"
        elif any(keyword in question_lower for keyword in ['yas']):
            return "Yas Mall"
        elif any(keyword in question_lower for keyword in ['festival', 'dubai']):
            return "Festival City"
        else:
            return "General"
    
    def _requires_location_clarification(self, question: str, documents: list) -> bool:
        """
        Use LLM to intelligently determine if question requires location clarification
        """
        question_lower = question.lower().strip()
        
        # Quick check: if location is already specified, no clarification needed
        location_keywords = []
        for loc in self.available_locations:
            # Add location name and city keywords
            location_keywords.extend([
                loc['name'].lower(),
                loc['city'].lower(),
                # Add common variations
                loc['name'].lower().replace(' mall', '').replace(' city', ''),
            ])
        
        has_location = any(keyword in question_lower for keyword in location_keywords)
        
        if has_location:
            return False  # Location already specified
        
        # Use LLM to determine if location clarification is needed
        location_analysis_prompt = PromptTemplate(
            template="""You are analyzing a user question to determine if it requires location-specific information.

Context: This is for a business with multiple physical locations. Some information varies by location (like prices, hours, contact details, specific services, parking, addresses), while other information is general (like safety policies, general service descriptions, company information).

Question: "{question}"

Instructions:
1. If the question already mentions a specific location or place name, respond "NO"
2. If the question asks for information that typically varies by physical location (prices, hours, contact info, booking, address, parking, specific services, etc.), respond "YES"
3. If the question asks for general information that's typically the same across all locations (safety rules, general policies, what the business does, age requirements, etc.), respond "NO"
4. When in doubt, prefer "NO" to avoid over-asking for clarification
5. Consider the context of a multi-location business when making your decision

Examples for context:
- "What are the prices?" → "YES" (prices may vary by location)
- "What are your opening hours?" → "YES" (hours typically vary by location)
- "What are your safety rules?" → "NO" (safety rules are typically standardized)
- "What activities do you have?" → "NO" (general service offerings are typically the same)
- "How do I contact you?" → "YES" (contact info varies by location)
- "What age groups do you serve?" → "NO" (service parameters are typically the same)

Respond with only "YES" or "NO":""",
            input_variables=["question"]
        )
        
        try:
            analysis_chain = location_analysis_prompt | self.llm_json_mode | StrOutputParser()
            result = analysis_chain.invoke({"question": question})
            
            needs_clarification = result.strip().upper() == "YES"
            print(f"---LLM LOCATION ANALYSIS: {result.strip()} for '{question}'---")
            return needs_clarification
            
        except Exception as e:
            print(f"---LOCATION ANALYSIS ERROR: {e}, defaulting to False---")
            return False  # Default to not asking for clarification if LLM fails
    
    def route_after_retrieval(self, state: RAGState) -> str:
        """
        Smart routing after retrieval - check if location clarification is needed
        """
        documents = state.get("documents", [])
        question = state["question"]
        
        # Ask for location clarification for location-dependent questions
        if self._requires_location_clarification(question, documents):
            print("---ROUTE: NEEDS LOCATION CLARIFICATION---")
            return "clarification"
        
        # If we have documents, generate answer
        if documents:
            print("---ROUTE: GENERATE ANSWER---")
            return "generate"
        
        # If no documents and question is very vague, ask for clarification
        question_lower = question.lower().strip()
        vague_questions = ['hi', 'hello', 'help', 'info', 'what', 'tell me']
        
        if any(vague in question_lower for vague in vague_questions) and len(question.split()) <= 2:
            print("---ROUTE: NEEDS CLARIFICATION---")
            return "clarification"
        
        # Otherwise still try to generate an answer
        print("---ROUTE: GENERATE ANSWER (NO DOCS)---")
        return "generate"
    
    # Legacy routing methods (keeping for backward compatibility)
    
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
        """Setup the ultra-fast RAG workflow with minimal steps"""
        if not self.retriever:
            raise ValueError("Vector store and retriever must be initialized first")
        
        # Create the fastest possible graph with minimal nodes
        workflow = StateGraph(RAGState)
        
        # Add only essential nodes - removed grading and complex routing
        workflow.add_node("retrieve_and_generate", self.fast_retrieve_and_generate)
        
        # Define the ultra-simple workflow
        # Single step: retrieve and generate immediately
        workflow.add_edge(START, "retrieve_and_generate")
        workflow.add_edge("retrieve_and_generate", END)
        
        # Compile the graph
        self.graph = workflow.compile()
        
        print("Ultra-fast RAG workflow setup complete")
        print("Flow: retrieve_and_generate -> END")
    
    def fast_retrieve_and_generate(self, state: RAGState) -> RAGState:
        """
        Ultra-fast combined retrieval and generation in one step
        Eliminates document grading and complex processing for speed
        Now includes user information tracking and name collection
        """

        question = state["question"]
        chat_history = state.get("chat_history", [])
        
        # Initialize user tracking state if not present
        user_phone = state.get("user_phone")
        if not user_phone:
            user_phone = self.user_tracker.generate_test_phone_number()
            # Store phone in state for session consistency
            state["user_phone"] = user_phone
        
        # Get or create user profile
        user_profile = self.user_tracker.get_user_profile(user_phone)
        
        # Extract name from current message if possible
        name_extraction = self.user_tracker.extract_name_from_message(question, chat_history)
        
        # Update user profile if name was found
        if name_extraction.get("name_found") and name_extraction.get("confidence", 0) > 0.7:
            extracted_name = name_extraction.get("extracted_name")
            self.user_tracker.update_user_profile(user_phone, extracted_name)
            user_profile["name"] = extracted_name
        
        # NEW: Handle first interaction immediately - ask for name
        is_first_interaction = self.user_tracker.is_first_interaction(chat_history)
        is_name_response = self.user_tracker.is_name_response(question, chat_history)
        
        # Check if we should request name or handle name response
        should_request_name = False
        name_request_message = ""
        personalized_intro = ""
        
        if is_first_interaction and not name_extraction.get("name_found"):
            # First interaction without name - store question and ask for name politely
            should_request_name = True
            name_request_message = self.user_tracker.generate_name_request("polite_request")
            # Store the original question for later
            self.user_tracker.store_original_question(user_phone, question)
            
        elif name_extraction.get("name_found") and name_extraction.get("confidence", 0) > 0.7:
            # Name was just provided - check if we need to answer stored question
            extracted_name = name_extraction.get("extracted_name")
            
            # Always try to get stored question when name is provided
            stored_question = self.user_tracker.get_stored_question(user_phone)
            
            if stored_question:
                # Generate greeting + answer the original question
                personalized_greeting = self.user_tracker.generate_personalized_greeting(extracted_name)
                
                # Answer the stored question with personalization
                return self._answer_with_personalization(stored_question, personalized_greeting, user_phone, user_profile, chat_history, name_extraction)
            else:
                # No stored question, just greet
                personalized_intro = f"Thank you, {extracted_name}! 😊 How can I help you with Leo & Loona today?"
                
                # Log the personalized intro response
                self.user_tracker.log_conversation(user_phone, extracted_name, personalized_intro, is_user=False)
                
                return {
                    "generation": personalized_intro,
                    "source_documents": [],
                    "documents": [],
                    "question": question,
                    "chat_history": chat_history,
                    # User information state
                    "user_phone": user_phone,
                    "user_name": extracted_name,
                    "user_profile": user_profile,
                    "name_extraction_result": name_extraction,
                    "should_request_name": False,
                    "name_request_message": "",
                    "conversation_logged": True
                }
            
        elif not user_profile.get("name"):
            # No name yet, check if we should ask (fallback for edge cases)
            name_request_analysis = self.user_tracker.should_request_name(chat_history, question, user_profile)
            should_request_name = name_request_analysis.get("should_ask", False)
            if should_request_name:
                approach = name_request_analysis.get("approach", "personalization")
                name_request_message = self.user_tracker.generate_name_request(approach)
        
        # Log the conversation
        self.user_tracker.log_conversation(user_phone, user_profile.get("name"), question, is_user=True)
        self.user_tracker.update_user_profile(user_phone)  # Update message count and last seen
        
        # Analyze conversation for Bitrix lead categorization
        category_analysis = self.user_tracker.analyze_conversation_category(chat_history, question)
        
        # Check if we should create a lead in Bitrix
        if (self.lead_manager and 
            user_profile.get("name") and 
            self.lead_manager.should_create_lead(user_profile, chat_history + [{"role": "user", "content": question}])):
            
            # Create lead in Bitrix with proper categorization
            lead_result = self.lead_manager.create_chatbot_lead(
                user_info=user_profile,
                conversation_history=chat_history + [{"role": "user", "content": question}],
                category_analysis=category_analysis
            )
            
            # Store lead result for UI display
            if lead_result:
                state["lead_created"] = lead_result
        
        # If we should request name for first interaction, return name request
        if should_request_name and name_request_message and is_first_interaction:
            # Return polite name request
            self.user_tracker.log_conversation(user_phone, user_profile.get("name"), name_request_message, is_user=False)
            
            return {
                "generation": name_request_message,
                "source_documents": [],
                "documents": [],
                "question": question,
                "chat_history": chat_history,
                # User information state
                "user_phone": user_phone,
                "user_name": user_profile.get("name", ""),
                "user_profile": user_profile,
                "name_extraction_result": name_extraction,
                "should_request_name": True,
                "name_request_message": name_request_message,
                "conversation_logged": True
            }
        
        # Fast retrieval with fewer documents for speed
        documents = self.retriever.invoke(question)

        
        # Take only top 3 most relevant documents for speed
        top_documents = documents[:3]
        
        # Quick format documents
        context = "\n\n".join(doc.page_content for doc in top_documents)
        
        # Simple chat history formatting (last 3 messages only)
        chat_context = ""
        if chat_history:
            recent_messages = chat_history[-3:]
            chat_context = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in recent_messages])
        
        # Check if question is about Leo & Loona and detect location
        location_needed = self._check_location_clarification_needed(question, context)
        is_leo_loona_question = self._is_leo_loona_question(question, context)
        
        # Add standard opening hours if not in context (temporary solution)
        if "opening" in question.lower() or "hours" in question.lower():
            context += "\n\nSTANDARD OPENING HOURS:\n- Yas Mall: 10:00 AM - 10:00 PM (Daily)\n- Dalma Mall: 10:00 AM - 10:00 PM (Daily)\n- Festival City Mall: 10:00 AM - 10:00 PM (Daily)"
        
        if not is_leo_loona_question:
            # Politely redirect non-Leo & Loona questions
            answer = """I'm sorry, but I'm specifically here to help with questions about Leo & Loona amusement park! 🎠 I'd love to tell you about our magical attractions, ticket prices, opening hours, birthday parties, or anything else related to Leo & Loona. What would you like to know about our wonderful park? ✨"""
            
            # Add name request if it's the first interaction
            if should_request_name and name_request_message:
                answer += f"\n\n{name_request_message}"
                
            # Log the bot response for non-Leo & Loona questions too
            self.user_tracker.log_conversation(user_phone, user_profile.get("name"), answer, is_user=False)
            
            return {
                "generation": answer,
                "source_documents": [],
                "documents": [],
                "question": question,
                "chat_history": chat_history,
                # User information state
                "user_phone": user_phone,
                "user_name": user_profile.get("name", ""),
                "user_profile": user_profile,
                "name_extraction_result": name_extraction,
                "should_request_name": False,  # Don't ask for name on redirections
                "name_request_message": "",
                "conversation_logged": True
            }
        
        if location_needed:
            # Ask for location clarification
            answer = """I'd be happy to help you with that! 😊 Leo & Loona has magical locations at different malls. Could you please let me know which location you're asking about?

Our Leo & Loona parks are located at:
🎪 **Dalma Mall** (Abu Dhabi)
🎪 **Yas Mall** (Abu Dhabi)  
🎪 **Festival City Mall** (Dubai)

Which one would you like to know about? ✨"""
            
            # Add name request if it's needed
            if should_request_name and name_request_message:
                answer += f"\n\n{name_request_message}"
                
            # Log the bot response for location clarification
            self.user_tracker.log_conversation(user_phone, user_profile.get("name"), answer, is_user=False)
            
            return {
                "generation": answer,
                "source_documents": top_documents,
                "documents": top_documents,
                "question": question,
                "chat_history": chat_history,
                # User information state
                "user_phone": user_phone,
                "user_name": user_profile.get("name", ""),
                "user_profile": user_profile,
                "name_extraction_result": name_extraction,
                "should_request_name": False,  # Don't ask for name during location clarification
                "name_request_message": "",
                "conversation_logged": True
            }

        # Get current date/time context for accurate responses
        datetime_context = self._format_datetime_context()
        
        # Fast prompt with Leo & Loona personality and date/time awareness
        fast_prompt = f"""You are a warm, friendly, and knowledgeable virtual host of Leo & Loona magical family amusement park. You speak with genuine enthusiasm and a caring tone, like a host who greets guests at the park entrance.

{datetime_context}

IMPORTANT RESTRICTIONS:
- ONLY answer questions about Leo & Loona amusement park
- DO NOT answer questions about Hello Park, other parks, or unrelated topics
- If asked about non-Leo & Loona topics, politely redirect to Leo & Loona

PERSONALITY & TONE:
- Warm, joyful, and professional
- Use simple, friendly language  
- Include light, natural compliments when appropriate (e.g., "what a beautiful name!")
- Use maximum 2 emojis per message, only when they enhance warmth
- Create anticipation and joy about visiting
- Be informative first, wrapped in pleasant and engaging tone
- Never pressure guests, only suggest offers if relevant

IMPORTANT: Use the current date/time information above INTERNALLY to provide accurate answers. DO NOT mention the specific date/time to users unless they specifically ask "what time is it" or "what date is it". Instead, naturally reference:
- "Today's hours" or "we're open until" (without stating the current time)
- "Today's pricing" or "weekday/weekend rates" (without stating the day)
- "We're currently open" or "we're closed right now" (without stating exact time)
- Use phrases like "today", "right now", "current" instead of specific dates/times

Context from Leo & Loona FAQ:
{context}

Recent conversation:
{chat_context}

Question: {question}

Answer as Leo & Loona's warm, welcoming park host (Leo & Loona topics ONLY):"""
        
        # Generate response quickly
        response = self.llm.invoke([HumanMessage(content=fast_prompt)])
        answer = response.content
        
        # Add name request to answer if needed
        if should_request_name and name_request_message:
            answer += f"\n\n{name_request_message}"
        
        # Log the bot response
        self.user_tracker.log_conversation(user_phone, user_profile.get("name"), answer, is_user=False)
        
        return {
            "generation": answer,
            "source_documents": top_documents,
            "documents": top_documents,
            "question": question,
            "chat_history": chat_history,
            # User information state
            "user_phone": user_phone,
            "user_name": user_profile.get("name", ""),
            "user_profile": user_profile,
            "name_extraction_result": name_extraction,
            "should_request_name": should_request_name,
            "name_request_message": name_request_message,
            "conversation_logged": True
        }
    
    def _answer_with_personalization(self, stored_question: str, greeting: str, user_phone: str, user_profile: dict, chat_history: list, name_extraction: dict) -> dict:
        """Answer the stored question with personalization"""
        
        # Get documents for the stored question
        documents = self.retriever.invoke(stored_question)
        top_documents = documents[:3]
        context = "\n\n".join(doc.page_content for doc in top_documents)
        
        # Check location and Leo & Loona relevance for stored question
        location_needed = self._check_location_clarification_needed(stored_question, context)
        is_leo_loona_question = self._is_leo_loona_question(stored_question, context)
        
        # Add standard opening hours if needed
        if "opening" in stored_question.lower() or "hours" in stored_question.lower():
            context += "\n\nSTANDARD OPENING HOURS:\n- Yas Mall: 10:00 AM - 10:00 PM (Daily)\n- Dalma Mall: 10:00 AM - 10:00 PM (Daily)\n- Festival City Mall: 10:00 AM - 10:00 PM (Daily)"
        
        if not is_leo_loona_question:
            answer = f"{greeting} I'm specifically here to help with questions about Leo & Loona amusement park! 🎠 I'd love to tell you about our magical attractions, ticket prices, opening hours, birthday parties, or anything else related to Leo & Loona. What would you like to know about our wonderful park?"
        elif location_needed:
            answer = f"{greeting} Leo & Loona has magical locations at different malls. Could you please let me know which location you're asking about?\n\nOur Leo & Loona parks are located at:\n🎪 **Dalma Mall** (Abu Dhabi)\n🎪 **Yas Mall** (Abu Dhabi)\n🎪 **Festival City Mall** (Dubai)\n\nWhich one would you like to know about? ✨"
        else:
            # Generate normal answer with personalization
            datetime_context = self._format_datetime_context()
            
            fast_prompt = f"""You are a warm, friendly, and knowledgeable virtual host of Leo & Loona magical family amusement park.

{datetime_context}

IMPORTANT: Start your response with this exact greeting: "{greeting}"

Context information:
{context}

Question: {stored_question}

Answer as Leo & Loona's warm, welcoming park host with the personalized greeting first:"""
            
            response = self.llm.invoke([HumanMessage(content=fast_prompt)])
            answer = response.content
        
        # Log the personalized response
        self.user_tracker.log_conversation(user_phone, user_profile.get("name"), answer, is_user=False)
        
        return {
            "generation": answer,
            "source_documents": top_documents,
            "documents": top_documents,
            "question": stored_question,
            "chat_history": chat_history,
            # User information state
            "user_phone": user_phone,
            "user_name": user_profile.get("name", ""),
            "user_profile": user_profile,
            "name_extraction_result": name_extraction,
            "should_request_name": False,
            "name_request_message": "",
            "conversation_logged": True
        }
    
    def _is_leo_loona_question(self, question: str, context: str) -> bool:
        """
        Fast check if question is about Leo & Loona
        """
        question_lower = question.lower()
        context_lower = context.lower()
        
        # Leo & Loona indicators
        leo_loona_keywords = [
            "leo", "loona", "leo & loona", "leo and loona",
            "amusement park", "theme park", "family park"
        ]
        
        # Non-Leo & Loona indicators (Hello Park, etc.)
        other_park_keywords = [
            "hello park", "hello world", "other park", "different park"
        ]
        
        # General non-park topics
        general_topics = [
            "weather", "politics", "news", "programming", "cooking",
            "sports", "movies", "music", "school", "work", "health",
            "travel", "hotel", "restaurant", "shopping mall"
        ]
        
        # Check for non-Leo & Loona content
        for keyword in other_park_keywords + general_topics:
            if keyword in question_lower:
                return False
        
        # Check for Leo & Loona content in question or context
        for keyword in leo_loona_keywords:
            if keyword in question_lower or keyword in context_lower:
                return True
        
        # If context mentions Leo & Loona, assume it's related
        if "leo" in context_lower or "loona" in context_lower:
            return True
            
        # Default to True for ambiguous cases (let the LLM handle it)
        return True
    
    def _check_location_clarification_needed(self, question: str, context: str) -> bool:
        """
        Fast check if location clarification is needed
        """
        question_lower = question.lower()
        
        # Location-specific keywords that need clarification
        location_keywords = [
            "hours", "opening", "closing", "address", "location", "phone",
            "contact", "directions", "parking", "nearby", "mall"
        ]
        
        # Check if question mentions location keywords
        needs_location = any(keyword in question_lower for keyword in location_keywords)
        
        if not needs_location:
            return False
        
        # Check if location is already specified
        location_names = ["dalma", "yas", "festival", "abu dhabi", "dubai"]
        has_location = any(location in question_lower for location in location_names)
        
        # Need clarification if asking about location-specific info without specifying location
        return needs_location and not has_location
    
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
            # Enhanced initial state with user tracking
            # Use session-consistent phone number
            session_phone = getattr(self, '_session_phone', None)
            if not session_phone:
                session_phone = self.user_tracker.generate_test_phone_number()
                self._session_phone = session_phone
            
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
                "enhanced_context": "",
                # User tracking initialization with consistent phone
                "user_phone": session_phone,
                "user_name": "",
                "user_profile": {},
                "name_extraction_result": {},
                "should_request_name": False,
                "name_request_message": "",
                "conversation_logged": False
            }
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            return {
                "answer": result.get("generation", "No answer generated"),
                "source_documents": result.get("source_documents", []),
                # Include user information for Streamlit display
                "user_info": {
                    "phone": result.get("user_phone", ""),
                    "name": result.get("user_name", ""),
                    "profile": result.get("user_profile", {}),
                    "name_extraction": result.get("name_extraction_result", {}),
                    "should_request_name": result.get("should_request_name", False),
                    "conversation_logged": result.get("conversation_logged", False)
                },
                # Include lead creation info if available
                "lead_created": result.get("lead_created", None)
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