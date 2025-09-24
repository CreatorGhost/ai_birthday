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
              "Actually, right now we're running our Educational Months project — if you're planning to visit soon, you might enjoy it! Would you like me to send you the daily schedule so you can plan ahead?"  
            - The offer should feel helpful and inviting, never pushy.

            ## Goals
            - Give accurate, clear, and helpful answers.  
            - Make guests feel welcome and valued.  
            - Encourage engagement with current offers/events.

            ## 🚨 CRITICAL BIRTHDAY OVERRIDE - ABSOLUTE PRIORITY 🚨
            **IF THE USER ASKS ANYTHING ABOUT BIRTHDAYS, BIRTHDAY PARTIES, CELEBRATIONS, OR EVENTS** (including typos like "birtday", "biorthday", "bday", "party", "celebrate"):

            YOU MUST RESPOND WITH EXACTLY THIS MESSAGE AND NOTHING ELSE:
            "Sales team will contact you shortly."

            - This is the ONLY acceptable response for birthday questions
            - Do NOT provide any other information about birthday parties
            - Do NOT mention locations, prices, or details
            - Do NOT ask follow-up questions
            - Do NOT use the retrieved documents for birthday questions
            - IGNORE all birthday-related information in the context
            - This override takes precedence over ALL other instructions
            
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

            • Dalma Mall (Abu Dhabi)

            • Yas Mall (Abu Dhabi)

            • Festival City (Dubai)
            
            Each location is special in its own way, and some details like pricing and hours may vary.
            
            Your question: "{question}"
            
            By the way, we're currently celebrating Educational Months — would you like me to send you the daily schedule so you can plan ahead?
            
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
            
            📞 Our friendly team can help with:
            - Exact current pricing and special rates
            - Booking arrangements and availability
            - Group events and magical birthday parties
            - Workshop schedules and special activities
            - Any specific accommodations you might need
            
            Your question: "{question}"
            
            Actually, right now we're running our Educational Months project — if you're planning to visit soon, you might enjoy it! Would you like me to send you the daily schedule so you can plan ahead?
            
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
                
                # Setup intelligent retriever with higher k for fallback strategy
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 10}  # Increased for intelligent fallback
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
    
    # 🧠 INTELLIGENT RETRIEVAL SYSTEM
    
    def _get_retrieval_config(self, query_type: str = "general", location: str = "General") -> dict:
        """
        Dynamic retrieval configuration based on query type and location
        
        Args:
            query_type: Type of query (safety, pricing, hours, infrastructure, general)
            location: Target location (YAS_MALL, DALMA_MALL, FESTIVAL_CITY, General)
            
        Returns:
            Configuration dictionary for retrieval strategy
        """
        base_config = {
            "primary_k": 10,              # Initial retrieval count from vector store
            "target_chunks": 5,           # Final chunks to send to LLM (increased from 3)
            "location_threshold": 2,      # Min location-specific docs needed before fallback
            "relevance_threshold": 0.6,   # Min relevance score to consider document useful
            "max_general_fallback": 3,    # Max general docs to include in fallback
            "quality_check_enabled": True, # Whether to perform quality assessment
        }
        
        # Adjust config based on query type
        query_adjustments = {
            "safety": {
                "target_chunks": 4,        # Safety questions need comprehensive coverage
                "location_threshold": 1,   # Even 1 good safety doc might be sufficient
                "relevance_threshold": 0.7, # Higher threshold for safety accuracy
            },
            "pricing": {
                "target_chunks": 4,        # Pricing needs both general and specific info
                "location_threshold": 2,   # Need at least 2 pricing docs
                "max_general_fallback": 2, # Limit general pricing docs
            },
            "hours": {
                "target_chunks": 3,        # Hours are usually straightforward
                "location_threshold": 1,   # One good hours doc is often enough
                "relevance_threshold": 0.8, # High threshold for accuracy
            },
            "infrastructure": {
                "target_chunks": 5,        # Infrastructure needs detailed info
                "location_threshold": 3,   # Infrastructure is very location-specific
                "max_general_fallback": 1, # Limit fallback for infrastructure
            },
            "birthday": {
                "target_chunks": 4,        # Birthday parties need good coverage
                "location_threshold": 2,   # Location matters for party planning
                "relevance_threshold": 0.6, # Slightly lower threshold
            }
        }
        
        # Apply query-specific adjustments
        if query_type in query_adjustments:
            base_config.update(query_adjustments[query_type])
        
        # Location-specific adjustments
        if location == "General":
            base_config["location_threshold"] = 0  # No location preference
            base_config["max_general_fallback"] = base_config["target_chunks"]
        
        print(f"🔧 Retrieval config for {query_type} @ {location}: {base_config}")
        return base_config
    
    def _score_document_relevance(self, doc: Document, query: str, location: str, query_type: str) -> dict:
        """
        Score document relevance using multiple factors
        
        Args:
            doc: Document to score
            query: Original user query
            location: Target location
            query_type: Type of query (safety, pricing, etc.)
            
        Returns:
            Dictionary with score and breakdown
        """
        scores = {}
        
        # 1. Content similarity (semantic relevance)
        query_lower = query.lower()
        content_lower = doc.page_content.lower()
        
        # Basic keyword matching for content similarity
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        common_words = query_words.intersection(content_words)
        
        # Calculate content similarity score
        if len(query_words) > 0:
            scores["content_similarity"] = len(common_words) / len(query_words)
        else:
            scores["content_similarity"] = 0.0
        
        # 2. Location match score
        doc_location = doc.metadata.get("location", "").upper()
        if location == "General" or doc_location == "GENERAL":
            scores["location_match"] = 0.8  # General docs are good for any location
        elif doc_location == location:
            scores["location_match"] = 1.0  # Perfect location match
        elif doc_location in ["YAS_MALL", "DALMA_MALL", "FESTIVAL_CITY"] and location != "General":
            scores["location_match"] = 0.3  # Different specific location
        else:
            scores["location_match"] = 0.6  # Unknown or neutral location
        
        # 3. Content type match score
        content_type = doc.metadata.get("content_type", "").lower()
        source = doc.metadata.get("source", "").lower()
        
        content_type_scores = {
            "safety": 1.0 if any(term in content_lower for term in ["safety", "mandatory", "rule", "required"]) else 0.5,
            "pricing": 1.0 if any(term in content_lower for term in ["aed", "price", "cost", "ticket"]) else 0.5,
            "hours": 1.0 if any(term in content_lower for term in ["hours", "open", "close", "time"]) else 0.5,
            "infrastructure": 1.0 if "infrastructure" in content_type else 0.5,
            "birthday": 1.0 if any(term in content_lower for term in ["birthday", "party", "celebration"]) else 0.5,
            "general": 0.7  # General queries get neutral score
        }
        
        scores["content_type_match"] = content_type_scores.get(query_type, 0.7)
        
        # 4. Source priority score
        if "consolidated_faq" in source:
            scores["source_priority"] = 0.9  # Consolidated FAQ is high priority
        elif "general_faq" in source:
            scores["source_priority"] = 0.8  # General FAQ is good
        elif any(mall in source for mall in ["yas", "dalma", "festival"]):
            scores["source_priority"] = 0.7  # Location-specific sources
        else:
            scores["source_priority"] = 0.6  # Other sources
        
        # Calculate weighted total score
        weights = {
            "content_similarity": 0.4,
            "location_match": 0.3,
            "content_type_match": 0.2,
            "source_priority": 0.1
        }
        
        total_score = sum(scores[factor] * weights[factor] for factor in weights)
        
        return {
            "total_score": round(total_score, 3),
            "breakdown": scores,
            "weights": weights
        }
    
    def _assess_document_quality(self, docs: List[Document], query: str, location: str, 
                                query_type: str, config: dict) -> dict:
        """
        Assess if retrieved documents are sufficient for answering the query
        
        Args:
            docs: List of documents to assess
            query: Original user query
            location: Target location
            query_type: Type of query
            config: Retrieval configuration
            
        Returns:
            Dictionary with quality assessment results
        """
        if not docs:
            return {
                "sufficient": False,
                "reason": "no_documents_found",
                "location_specific_count": 0,
                "general_count": 0,
                "avg_relevance": 0.0,
                "recommendations": ["expand_search", "try_general_fallback"]
            }
        
        # Score all documents
        scored_docs = []
        location_specific_docs = []
        general_docs = []
        
        for doc in docs:
            score_result = self._score_document_relevance(doc, query, location, query_type)
            doc_info = {
                "document": doc,
                "score": score_result["total_score"],
                "breakdown": score_result["breakdown"]
            }
            scored_docs.append(doc_info)
            
            # Categorize by location
            doc_location = doc.metadata.get("location", "").upper()
            if doc_location == location:
                location_specific_docs.append(doc_info)
            elif doc_location in ["GENERAL", ""] or "general" in doc.metadata.get("source", "").lower():
                general_docs.append(doc_info)
        
        # Calculate metrics
        relevant_docs = [d for d in scored_docs if d["score"] >= config["relevance_threshold"]]
        avg_relevance = sum(d["score"] for d in scored_docs) / len(scored_docs) if scored_docs else 0.0
        
        location_count = len(location_specific_docs)
        general_count = len(general_docs)
        
        # Quality assessment criteria
        has_min_location_docs = location_count >= config["location_threshold"]
        has_good_relevance = avg_relevance >= config["relevance_threshold"]
        has_min_total_docs = len(relevant_docs) >= 2
        
        # Determine if quality is sufficient
        if location == "General":
            # For general queries, just need good relevance and enough docs
            sufficient = has_good_relevance and has_min_total_docs
            reason = "general_query_satisfied" if sufficient else "insufficient_general_docs"
        else:
            # For location-specific queries, prefer location docs but allow fallback
            sufficient = (has_min_location_docs and has_good_relevance) or \
                        (has_good_relevance and has_min_total_docs and avg_relevance > 0.7)
            
            if not sufficient:
                if location_count == 0:
                    reason = "no_location_specific_docs"
                elif not has_good_relevance:
                    reason = "low_relevance_scores"
                else:
                    reason = "insufficient_document_count"
            else:
                reason = "quality_criteria_met"
        
        # Generate recommendations
        recommendations = []
        if not sufficient:
            if location_count < config["location_threshold"] and location != "General":
                recommendations.append("try_general_fallback")
            if avg_relevance < config["relevance_threshold"]:
                recommendations.append("expand_search_terms")
            if len(relevant_docs) < config["target_chunks"]:
                recommendations.append("increase_retrieval_count")
        
        result = {
            "sufficient": sufficient,
            "reason": reason,
            "location_specific_count": location_count,
            "general_count": general_count,
            "total_relevant_docs": len(relevant_docs),
            "avg_relevance": round(avg_relevance, 3),
            "top_scores": [round(d["score"], 3) for d in sorted(scored_docs, key=lambda x: x["score"], reverse=True)[:5]],
            "recommendations": recommendations,
            "scored_documents": scored_docs
        }
        
        print(f"📊 Quality Assessment: {result['reason']} | Location: {location_count}, General: {general_count}, Avg: {result['avg_relevance']}")
        return result
    
    def _execute_intelligent_retrieval(self, query: str, location: str, query_type: str) -> List[Document]:
        """
        Execute intelligent retrieval with smart fallback strategy
        
        Args:
            query: User's question
            location: Target location (YAS_MALL, DALMA_MALL, FESTIVAL_CITY, General)
            query_type: Type of query (safety, pricing, hours, etc.)
            
        Returns:
            List of best documents for the query
        """
        print(f"🧠 Starting intelligent retrieval: {query_type} query for {location}")
        
        # Get retrieval configuration
        config = self._get_retrieval_config(query_type, location)
        
        # Stage 1: Enhanced query preparation
        enhanced_query = self._enhance_query_for_retrieval(query, location, query_type)
        print(f"🔍 Enhanced query: '{query}' → '{enhanced_query}'")
        
        # Stage 2: Primary retrieval
        print(f"📚 Stage 1: Primary retrieval (k={config['primary_k']})")
        primary_docs = self.retriever.invoke(enhanced_query)
        print(f"📚 Retrieved {len(primary_docs)} documents from primary search")
        
        # Stage 3: Quality assessment of primary results
        quality_assessment = self._assess_document_quality(
            primary_docs, query, location, query_type, config
        )
        
        if quality_assessment["sufficient"]:
            print(f"✅ Primary retrieval sufficient: {quality_assessment['reason']}")
            best_docs = self._select_best_documents(
                quality_assessment["scored_documents"], config["target_chunks"]
            )
            return best_docs
        
        print(f"❌ Primary retrieval insufficient: {quality_assessment['reason']}")
        print(f"💡 Recommendations: {quality_assessment['recommendations']}")
        
        # Stage 4: Smart fallback strategy
        if "try_general_fallback" in quality_assessment["recommendations"] and location != "General":
            print(f"🔄 Stage 2: General fallback retrieval")
            fallback_docs = self._execute_general_fallback(query, query_type, config)
            
            # Combine primary and fallback docs
            combined_docs = primary_docs + fallback_docs
            
            # Re-assess combined quality
            combined_assessment = self._assess_document_quality(
                combined_docs, query, location, query_type, config
            )
            
            print(f"🔄 Combined assessment: {combined_assessment['reason']}")
            best_docs = self._select_best_documents(
                combined_assessment["scored_documents"], config["target_chunks"]
            )
            return best_docs
        
        # Stage 5: Fallback - use best available documents
        print(f"⚠️ Using best available documents from primary search")
        best_docs = self._select_best_documents(
            quality_assessment["scored_documents"], min(config["target_chunks"], len(primary_docs))
        )
        return best_docs
    
    def _enhance_query_for_retrieval(self, query: str, location: str, query_type: str) -> str:
        """
        Enhance query with location and type-specific terms for better retrieval
        
        Args:
            query: Original query
            location: Target location
            query_type: Query type
            
        Returns:
            Enhanced query string
        """
        enhanced_parts = [query]
        
        # Add location context if specific
        if location != "General":
            location_names = {
                "YAS_MALL": "Yas Mall",
                "DALMA_MALL": "Dalma Mall",
                "FESTIVAL_CITY": "Festival City"
            }
            if location in location_names:
                enhanced_parts.append(location_names[location])
        
        # Add query type context
        type_enhancements = {
            "safety": "safety rules requirements mandatory",
            "pricing": "price cost fee AED ticket",
            "hours": "opening hours time schedule",
            "infrastructure": "infrastructure facilities equipment",
            "birthday": "birthday party celebration event"
        }
        
        if query_type in type_enhancements:
            enhanced_parts.append(type_enhancements[query_type])
        
        return " ".join(enhanced_parts)
    
    def _execute_general_fallback(self, query: str, query_type: str, config: dict) -> List[Document]:
        """
        Execute general fallback search when location-specific search is insufficient
        
        Args:
            query: Original query
            query_type: Query type
            config: Retrieval configuration
            
        Returns:
            List of general documents
        """
        # Create general search queries
        fallback_queries = []
        
        # Base general query
        fallback_queries.append(f"{query} general FAQ")
        
        # 🚀 DYNAMIC: Generate type-specific queries using LLM (no hardcoding)
        type_specific_queries = self._generate_dynamic_fallback_queries(query, query_type)
        
        if query_type in type_specific_queries:
            fallback_queries.extend(type_specific_queries[query_type])
        
        # Execute fallback searches
        fallback_docs = []
        max_fallback = config.get("max_general_fallback", 3)
        
        for fallback_query in fallback_queries[:3]:  # Limit to 3 fallback queries
            try:
                docs = self.retriever.invoke(fallback_query)
                fallback_docs.extend(docs[:2])  # Take top 2 from each search
                if len(fallback_docs) >= max_fallback * 2:  # Get extra for filtering
                    break
            except Exception as e:
                print(f"⚠️ Fallback search error: {e}")
                continue
        
        print(f"🔄 Fallback retrieved {len(fallback_docs)} general documents")
        return fallback_docs
    
    def _select_best_documents(self, scored_docs: List[dict], target_count: int) -> List[Document]:
        """
        Select the best documents from scored documents
        
        Args:
            scored_docs: List of scored document dictionaries
            target_count: Target number of documents to return
            
        Returns:
            List of best documents
        """
        if not scored_docs:
            return []
        
        # Sort by score (descending)
        sorted_docs = sorted(scored_docs, key=lambda x: x["score"], reverse=True)
        
        # Take top documents up to target count
        selected = sorted_docs[:target_count]
        
        # Extract just the documents
        best_docs = [item["document"] for item in selected]
        
        # Log selection
        print(f"🎯 Selected {len(best_docs)} best documents:")
        for i, item in enumerate(selected):
            doc = item["document"]
            score = item["score"]
            source = doc.metadata.get("source", "Unknown")
            location = doc.metadata.get("location", "Unknown")
            print(f"  #{i+1}: {source} ({location}) - Score: {score:.3f}")
        
        return best_docs
    
    # 🚀 NEW PARALLEL RETRIEVAL SYSTEM (No Classification Required)
    
    def _calculate_document_quality(self, doc: Document, query: str) -> float:
        """
        Calculate quality score for a document based on query relevance
        
        Args:
            doc: Document to evaluate
            query: User query
            
        Returns:
            Quality score (0.0 to 1.0)
        """
        query_lower = query.lower()
        content_lower = doc.page_content.lower()
        
        # Basic relevance scoring
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        
        # Word overlap score
        overlap = len(query_words.intersection(content_words))
        overlap_score = overlap / max(len(query_words), 1) if query_words else 0
        
        # Content length bonus (longer docs often have more info)
        length_score = min(len(doc.page_content) / 1000, 1.0) * 0.3
        
        # Source quality bonus
        source = doc.metadata.get('source', '').lower()
        source_bonus = 0.1 if 'faq' in source else 0.05
        
        return min(overlap_score + length_score + source_bonus, 1.0)
    
    def _ensure_document_diversity(self, docs: List[Document], max_docs: int = 5) -> List[Document]:
        """
        Ensure diversity in selected documents to avoid redundancy
        
        Args:
            docs: List of ranked documents
            max_docs: Maximum number of documents to return
            
        Returns:
            Diverse set of documents
        """
        if len(docs) <= max_docs:
            return docs
        
        selected = [docs[0]]  # Always include the best document
        
        for doc in docs[1:]:
            if len(selected) >= max_docs:
                break
                
            # Check diversity against already selected docs
            is_diverse = True
            doc_content_words = set(doc.page_content.lower().split())
            
            for selected_doc in selected:
                selected_content_words = set(selected_doc.page_content.lower().split())
                overlap = len(doc_content_words.intersection(selected_content_words))
                similarity = overlap / max(len(doc_content_words), len(selected_content_words), 1)
                
                # If too similar to an already selected document, skip
                if similarity > 0.7:  # 70% similarity threshold
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(doc)
        
        return selected
    
    def _parallel_retrieval_optimized(self, query: str, location: str) -> List[Document]:
        """
        Enhanced parallel retrieval - no classification, intelligent ranking
        
        Args:
            query: User query
            location: Target location (YAS_MALL, DALMA_MALL, etc.)
            
        Returns:
            Optimized list of diverse, high-quality documents
        """
        print(f"🚀 PARALLEL RETRIEVAL: No classification - direct retrieval")
        print(f"   📝 Query: '{query}'")
        print(f"   📍 Location: {location}")
        
        all_docs = []
        
        # STAGE 1: Location-specific retrieval (if not General)
        if location != "General":
            try:
                print(f"📍 Stage 1: Location-specific retrieval ({location} only)")
                # Get documents using vector search
                location_docs = self.retriever.invoke(f"{query} {location}")
                
                # 🔧 STRICT FILTERING: Only keep documents with exact location match
                filtered_location_docs = []
                for doc in location_docs:
                    doc_location = doc.metadata.get('location', 'UNKNOWN').upper()
                    if doc_location == location:
                        filtered_location_docs.append(doc)
                        print(f"   ✅ Kept {location} doc: {doc.metadata.get('source', 'Unknown')[:30]}...")
                    else:
                        print(f"   🚫 Filtered out {doc_location} doc: {doc.metadata.get('source', 'Unknown')[:30]}...")
                
                location_docs = filtered_location_docs
                print(f"   📊 {location} documents found: {len(location_docs)}")
                
                # Quality filtering for location docs
                quality_location_docs = []
                for doc in location_docs[:8]:  # Consider top 8
                    quality_score = self._calculate_document_quality(doc, query)
                    if quality_score >= 0.4:  # Quality threshold for location docs
                        quality_location_docs.append((doc, quality_score))
                        print(f"   📄 Location doc: {doc.metadata.get('source', 'Unknown')[:30]}... (quality: {quality_score:.2f})")
                
                # Sort by quality and take top 3-4
                quality_location_docs.sort(key=lambda x: x[1], reverse=True)
                location_selected = [doc for doc, score in quality_location_docs[:4]]
                all_docs.extend(location_selected)
                
                print(f"   ✅ Selected {len(location_selected)} location documents")
                
            except Exception as e:
                print(f"   ❌ Location retrieval failed: {e}")
        
        # STAGE 2: General retrieval (ONLY from GENERAL location, no other malls)
        try:
            print(f"📚 Stage 2: General retrieval (GENERAL location only)")
            # Get documents using vector search
            general_docs = self.retriever.invoke(query)
            
            # 🔧 STRICT FILTERING: Only keep documents with location = "GENERAL"
            filtered_general_docs = []
            for doc in general_docs:
                doc_location = doc.metadata.get('location', 'UNKNOWN').upper()
                if doc_location == 'GENERAL':
                    filtered_general_docs.append(doc)
                    print(f"   ✅ Kept GENERAL doc: {doc.metadata.get('source', 'Unknown')[:30]}...")
                else:
                    print(f"   🚫 Filtered out {doc_location} doc: {doc.metadata.get('source', 'Unknown')[:30]}...")
            
            general_docs = filtered_general_docs
            print(f"   📊 GENERAL documents found: {len(general_docs)}")
            
            # Quality filtering for general docs  
            quality_general_docs = []
            for doc in general_docs[:6]:  # Consider top 6
                quality_score = self._calculate_document_quality(doc, query)
                if quality_score >= 0.3:  # Slightly lower threshold for general docs
                    quality_general_docs.append((doc, quality_score))
                    print(f"   📄 General doc: {doc.metadata.get('source', 'Unknown')[:30]}... (quality: {quality_score:.2f})")
            
            # Sort by quality and take top 3 general docs (as requested)
            quality_general_docs.sort(key=lambda x: x[1], reverse=True)
            general_selected = [doc for doc, score in quality_general_docs[:3]]  # Take top 3 general docs
            all_docs.extend(general_selected)
            
            print(f"   ✅ Selected {len(general_selected)} general documents")
            
        except Exception as e:
            print(f"   ❌ General retrieval failed: {e}")
        
        # STAGE 3: Send ALL documents to LLM
        print(f"🎯 Stage 3: Preparing ALL documents for LLM")
        print(f"   📊 Total documents retrieved: {len(all_docs)}")
        
        if not all_docs:
            print(f"   ❌ No documents retrieved - falling back to simple retrieval")
            # Fallback to simple retrieval
            try:
                fallback_docs = self.retriever.invoke(query)
                return fallback_docs[:5]
            except:
                return []
        
        # Calculate final quality scores for all documents
        final_scored_docs = []
        for doc in all_docs:
            final_quality = self._calculate_document_quality(doc, query)
            final_scored_docs.append((doc, final_quality))
        
        # Sort by final quality
        final_scored_docs.sort(key=lambda x: x[1], reverse=True)
        ranked_docs = [doc for doc, score in final_scored_docs]
        
        # Send ALL documents to LLM (no diversity filtering)
        print(f"   📊 Final selection: ALL {len(ranked_docs)} documents (no filtering applied)")
        print(f"   🎯 Letting LLM choose the best information from all available documents")
        for i, doc in enumerate(ranked_docs):
            source = doc.metadata.get('source', 'Unknown')
            location_meta = doc.metadata.get('location', 'Unknown')
            preview = doc.page_content[:80].replace('\n', ' ')
            print(f"     #{i+1}: [{location_meta}|{source[:20]}...] {preview}...")
        
        return ranked_docs
    
    # 🚀 SMART STAGED RETRIEVAL SYSTEM (Zero-Hardcoding Enhancement)
    
    def _quick_confidence_check(self, question: str, documents: List[Document]) -> float:
        """
        Quick confidence assessment without full LLM evaluation
        
        Args:
            question: Original user question
            documents: Retrieved documents
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not documents:
            return 0.0
        
        # Quick keyword overlap check
        question_words = set(question.lower().split())
        total_overlap = 0
        total_length = 0
        
        for doc in documents:
            content_words = set(doc.page_content.lower().split())
            overlap = len(question_words.intersection(content_words))
            total_overlap += overlap
            total_length += len(doc.page_content)
        
        # Normalize by document count and content length
        if total_length > 0:
            base_confidence = min(0.9, total_overlap / len(question_words) * 0.3 + 
                                min(total_length / 1000, 1.0) * 0.4)
        else:
            base_confidence = 0.0
        
        # Boost confidence if we have multiple relevant documents
        if len(documents) >= 3:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def _retrieve_general_optimized(self, question: str) -> List[Document]:
        """
        Optimized general document retrieval for smart staged system
        
        Args:
            question: User's question
            
        Returns:
            List of general documents
        """
        print(f"🔍 Executing general retrieval for: '{question}'")
        
        # Use existing retriever but with general-focused enhancement
        enhanced_query = f"{question} general FAQ safety rules requirements"
        
        try:
            # Retrieve from general sources
            general_docs = self.retriever.invoke(enhanced_query)
            
            # Filter to prioritize general documents
            filtered_docs = []
            for doc in general_docs:
                source = doc.metadata.get('source', '').lower()
                location = doc.metadata.get('location', '').upper()
                
                # Prioritize general sources
                if any(term in source for term in ['general_faq', 'consolidated_faq']) or location == 'GENERAL':
                    filtered_docs.append(doc)
            
            # If no general docs found, use top documents from search
            if not filtered_docs:
                filtered_docs = general_docs[:4]
            
            print(f"📚 General retrieval found {len(filtered_docs)} relevant documents")
            return filtered_docs[:5]  # Return top 5
            
        except Exception as e:
            print(f"❌ General retrieval failed: {e}")
            return []
    
    def _smart_staged_retrieval(self, question: str, location: str, query_type: str) -> List[Document]:
        """
        Smart staged retrieval: Try location first, expand to general if needed
        
        Args:
            question: User's question
            location: Target location
            query_type: Type of query (safety, pricing, etc.)
            
        Returns:
            List of best documents from staged retrieval
        """
        print(f"🚀 Smart Staged Retrieval: {query_type} query for {location}")
        
        # STAGE 1: Current intelligent retrieval (location-focused)
        print(f"📍 Stage 1: Location-focused retrieval ({location})")
        try:
            location_documents = self._execute_intelligent_retrieval(question, location, query_type)
            location_confidence = self._quick_confidence_check(question, location_documents)
            
            print(f"📊 Location retrieval: {len(location_documents)} docs, confidence: {location_confidence:.2f}")
            
            # 📊 DETAILED LOCATION DOCUMENT ANALYSIS
            print(f"📊 LOCATION DOCUMENTS BREAKDOWN:")
            for i, doc in enumerate(location_documents):
                preview = doc.page_content[:100].replace('\n', ' ')
                location_meta = doc.metadata.get('location', 'UNKNOWN')
                source = doc.metadata.get('source', 'Unknown')
                sock_count = doc.page_content.lower().count('sock')
                mandatory_count = doc.page_content.lower().count('mandatory')
                print(f"   📄 Loc Doc {i+1}: [{location_meta}|{source}] socks={sock_count}, mandatory={mandatory_count}")
                print(f"      📝 Preview: {preview}...")
            
            # If location confidence is high, use location results
            confidence_threshold = 0.75  # Configurable threshold
            if location_confidence >= confidence_threshold:
                print(f"✅ Location confidence sufficient ({location_confidence:.2f} >= {confidence_threshold})")
                return location_documents
            
            print(f"⚠️ Location confidence low ({location_confidence:.2f} < {confidence_threshold})")
            
        except Exception as e:
            print(f"❌ Location retrieval failed: {e}")
            location_documents = []
            location_confidence = 0.0
        
        # STAGE 2: General retrieval for comparison
        print(f"📚 Stage 2: General retrieval fallback")
        try:
            general_documents = self._retrieve_general_optimized(question)
            general_confidence = self._quick_confidence_check(question, general_documents)
            
            print(f"📊 General retrieval: {len(general_documents)} docs, confidence: {general_confidence:.2f}")
            
            # 📊 DETAILED GENERAL DOCUMENT ANALYSIS
            print(f"📊 GENERAL DOCUMENTS BREAKDOWN:")
            for i, doc in enumerate(general_documents):
                preview = doc.page_content[:100].replace('\n', ' ')
                location_meta = doc.metadata.get('location', 'UNKNOWN')
                source = doc.metadata.get('source', 'Unknown')
                sock_count = doc.page_content.lower().count('sock')
                mandatory_count = doc.page_content.lower().count('mandatory')
                print(f"   📄 Gen Doc {i+1}: [{location_meta}|{source}] socks={sock_count}, mandatory={mandatory_count}")
                print(f"      📝 Preview: {preview}...")
                if sock_count > 0 or mandatory_count > 0:
                    # Show lines containing relevant keywords
                    relevant_lines = [line.strip() for line in doc.page_content.split('\n') 
                                   if 'sock' in line.lower() or 'mandatory' in line.lower()]
                    for line in relevant_lines[:2]:
                        print(f"      🎯 RELEVANT: {line}")
            
            # STAGE 3: Choose better result
            improvement_threshold = 0.15  # General must be significantly better
            
            if general_confidence > location_confidence + improvement_threshold:
                print(f"✅ Using general docs (confidence: {general_confidence:.2f} vs {location_confidence:.2f})")
                return general_documents
            else:
                print(f"✅ Using location docs (general not significantly better)")
                return location_documents if location_documents else general_documents
                
        except Exception as e:
            print(f"❌ General retrieval failed: {e}")
            # Return location documents as fallback
            return location_documents if location_documents else []
    
    def _generate_dynamic_fallback_queries(self, query: str, query_type: str) -> dict:
        """
        Generate dynamic fallback queries using LLM (replaces hardcoded queries)
        
        Args:
            query: Original user query
            query_type: Type of query (safety, pricing, etc.)
            
        Returns:
            Dictionary with generated queries for the query type
        """
        try:
            prompt = f"""Generate 2-3 search queries to find general information about this Leo & Loona question:

Original Question: "{query}"
Question Type: {query_type}

Create variations that would find relevant information in general FAQ documents:
1. A broad semantic search query
2. A keyword-focused query
3. A related concepts query

Example for safety question "are socks mandatory?":
- "safety requirements play area mandatory items"
- "general rules visitors equipment needed"
- "Leo Loona safety policies guidelines"

Return only the search queries, one per line, no numbers or formatting."""

            from langchain_core.messages import HumanMessage
            response = self.llm_json_mode.invoke([HumanMessage(content=prompt)])
            
            # Parse response into list
            queries = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
            
            # Return in expected format
            return {query_type: queries[:3]}  # Limit to 3 queries
            
        except Exception as e:
            print(f"❌ Dynamic query generation failed: {e}")
            # Minimal fallback - just use the original query with general context
            return {query_type: [f"{query} general", f"{query} FAQ", f"{query} rules"]}
    
    def _classify_query_intent(self, query: str, location: str = "General") -> dict:
        """
        Use LLM to classify query intent without hardcoded keywords
        
        Args:
            query: User's question
            location: Target location context
            
        Returns:
            Dictionary with classification results
        """
        classification_prompt = f"""Classify the intent of this user question about Leo & Loona amusement park.

USER QUESTION: "{query}"
LOCATION CONTEXT: {location}

Analyze the question and determine:

1. PRIMARY INTENT (choose ONE most relevant):
   - safety: Questions about safety rules, requirements, mandatory items, age restrictions
   - pricing: Questions about costs, prices, fees, money, tickets, packages
   - hours: Questions about opening times, schedules, when open/closed
   - infrastructure: Questions about facilities, equipment, layout, capacity, restaurant capacity
   - birthday: Questions about birthday parties, celebrations, events, bookings
   - activities: Questions about rides, attractions, what to do, games
   - food: Questions about dining, restaurants, food options, menu
   - location: Questions about directions, address, parking, how to get there
   - contact: Questions about phone numbers, contact information, communication
   - general: General questions about the park, policies, or unclear intent

2. SECONDARY INTENTS (can be multiple or none):
   - List any additional relevant categories from above

3. CONFIDENCE (0.0 to 1.0):
   - How confident are you in the primary intent classification?

4. REQUIRES_LOCATION_INFO (true/false):
   - Does this question need location-specific information to answer properly?

5. KEY_ENTITIES:
   - Extract important entities mentioned (items, activities, specific details)

IMPORTANT EXAMPLES:
- "are socks mandatory?" → PRIMARY: safety (not pricing, even though it mentions socks)
- "how much are socks?" → PRIMARY: pricing  
- "what are your opening hours?" → PRIMARY: hours
- "can I book a birthday party?" → PRIMARY: birthday
- "do you have parking?" → PRIMARY: location (infrastructure secondary)
- "what activities do you have?" → PRIMARY: activities

Respond with ONLY a JSON object:
{{
    "primary_intent": "safety",
    "secondary_intents": ["equipment"],
    "confidence": 0.95,
    "requires_location_info": false,
    "key_entities": ["socks", "mandatory"],
    "reasoning": "User is asking about safety requirements, specifically if socks are mandatory to wear"
}}"""

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm_json_mode.invoke([HumanMessage(content=classification_prompt)])
            response_text = response.content.strip()
            
            import json
            import re
            
            # Try to find JSON in response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    
                    # Validate and clean response
                    cleaned_result = {
                        "primary_intent": result.get("primary_intent", "general"),
                        "secondary_intents": result.get("secondary_intents", []),
                        "confidence": max(0.0, min(1.0, result.get("confidence", 0.5))),
                        "requires_location_info": result.get("requires_location_info", False),
                        "key_entities": result.get("key_entities", []),
                        "reasoning": result.get("reasoning", "LLM classification"),
                        "method": "llm_classification"
                    }
                    
                    # Validate primary intent
                    valid_intents = ["safety", "pricing", "hours", "infrastructure", "birthday", 
                                   "activities", "food", "location", "contact", "general"]
                    if cleaned_result["primary_intent"] not in valid_intents:
                        cleaned_result["primary_intent"] = "general"
                    
                    print(f"🎯 Query Classification: '{query}' → {cleaned_result['primary_intent']} (confidence: {cleaned_result['confidence']:.2f})")
                    print(f"   Reasoning: {cleaned_result['reasoning'][:100]}...")
                    
                    return cleaned_result
                    
                except json.JSONDecodeError as e:
                    print(f"⚠️ JSON decode error in classification: {e}")
                    print(f"   Response: {response_text[:200]}...")
            
            else:
                print(f"⚠️ No JSON found in classification response: {response_text[:200]}...")
            
        except Exception as e:
            print(f"❌ LLM classification error: {str(e)}")
        
        # Fallback classification
        print(f"🔄 Using fallback classification for: '{query}'")
        return self._fallback_query_classification(query)
    
    def _fallback_query_classification(self, query: str) -> dict:
        """
        Fallback classification using simple keyword matching
        
        Args:
            query: User's question
            
        Returns:
            Basic classification result
        """
        query_lower = query.lower()
        
        # Safety indicators (prioritized to fix socks issue)
        if any(term in query_lower for term in [
            "mandatory", "required", "must", "rule", "allowed", "permitted", 
            "safety", "wear", "bring", "need to", "have to", "suppose to"
        ]):
            return {
                "primary_intent": "safety",
                "secondary_intents": [],
                "confidence": 0.7,
                "requires_location_info": False,
                "key_entities": [],
                "reasoning": "Fallback: Contains safety/requirement keywords",
                "method": "fallback_keywords"
            }
        
        # Pricing indicators (only for clear pricing questions)
        elif any(term in query_lower for term in [
            "cost", "price", "expensive", "cheap", "how much", "fee", "charge", "aed"
        ]):
            return {
                "primary_intent": "pricing",
                "secondary_intents": [],
                "confidence": 0.7,
                "requires_location_info": True,
                "key_entities": [],
                "reasoning": "Fallback: Contains clear pricing keywords",
                "method": "fallback_keywords"
            }
        
        # Hours indicators  
        elif any(term in query_lower for term in [
            "hours", "open", "close", "time", "when", "schedule"
        ]):
            return {
                "primary_intent": "hours",
                "secondary_intents": [],
                "confidence": 0.8,
                "requires_location_info": True,
                "key_entities": [],
                "reasoning": "Fallback: Contains time/schedule keywords",
                "method": "fallback_keywords"
            }
        
        # Birthday indicators
        elif any(term in query_lower for term in [
            "birthday", "party", "celebration", "event", "book"
        ]):
            return {
                "primary_intent": "birthday",
                "secondary_intents": [],
                "confidence": 0.8,
                "requires_location_info": True,
                "key_entities": [],
                "reasoning": "Fallback: Contains birthday/party keywords",
                "method": "fallback_keywords"
            }
        
        # Default to general
        else:
            return {
                "primary_intent": "general",
                "secondary_intents": [],
                "confidence": 0.5,
                "requires_location_info": False,
                "key_entities": [],
                "reasoning": "Fallback: No specific intent detected",
                "method": "fallback_default"
            }
    
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
    
    def _filter_documents_by_location_and_type(self, documents: List[Document], target_location: str, content_type_priority: str = None) -> List[Document]:
        """
        Filter documents by location metadata and optionally prioritize content type
        
        Args:
            documents: List of retrieved documents
            target_location: Target location to filter for
            content_type_priority: Optional content type to prioritize (e.g., 'Infrastructure Information')
            
        Returns:
            Filtered documents from the specified location, optionally prioritized by content type
        """
        if target_location == 'unknown':
            return documents
        
        location_docs = []
        other_docs = []
        priority_docs = []
        
        for doc in documents:
            doc_location = doc.metadata.get('location', 'UNKNOWN_LOCATION')
            doc_content_type = doc.metadata.get('content_type', '')
            
            if doc_location == target_location:
                # If we have a content type priority and this doc matches
                if content_type_priority and doc_content_type == content_type_priority:
                    priority_docs.append(doc)
                else:
                    location_docs.append(doc)
            else:
                other_docs.append(doc)
        
        total_target_docs = len(priority_docs) + len(location_docs)
        print(f"📍 Location filtering: Found {total_target_docs} docs from {target_location} ({len(priority_docs)} priority + {len(location_docs)} regular), {len(other_docs)} from other locations")
        if content_type_priority:
            print(f"🏗️ Content type priority '{content_type_priority}': {len(priority_docs)} docs match both location and content type")
        
        # Prioritize: priority docs first, then location docs, then other docs for context
        filtered_docs = priority_docs + location_docs + other_docs[:2]
        
        if not filtered_docs:
            print(f"⚠️ No documents found for {target_location}, returning all documents")
            return documents
        
        return filtered_docs
    
    def _filter_documents_by_location(self, documents: List[Document], target_location: str) -> List[Document]:
        """
        Filter documents by location metadata (backward compatibility)
        """
        return self._filter_documents_by_location_and_type(documents, target_location)
    
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
        
        # Enhanced location detection with LLM (handles typos like "ys mall", "daaalma")
        extraction_result = self._extract_user_info_with_llm(question, chat_history)
        mall_name = extraction_result.get("mall_name", "General")
        
        # Convert to old format for compatibility
        mall_mapping = {
            "Yas Mall": "YAS_MALL",
            "Dalma Mall": "DALMA_MALL", 
            "Festival City": "FESTIVAL_CITY",
            "General": "General"
        }
        location = mall_mapping.get(mall_name, "General")
        print(f"🔍 LLM-detected location in simplified answer: {mall_name} → {location}")
        
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
              "Actually, right now we're running our Educational Months project — if you're planning to visit soon, you might enjoy it! Would you like me to send you the daily schedule so you can plan ahead?"  
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
    
    def _extract_user_info_with_llm(self, message: str, context: list = None) -> dict:
        """Use LLM to extract user name and mall from message, handling typos and variations"""
        
        # Format context for prompt
        context_str = ""
        if context:
            recent_messages = context[-3:] if len(context) > 3 else context
            context_str = " | ".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in recent_messages])
        
        prompt = f"""Extract user name and mall location from this message. Handle typos and variations carefully.

AVAILABLE MALLS (exact names to return):
- "Yas Mall" (for: yas, y;s, yaas, yas mall, yas island, abu dhabi yas, etc.)
- "Dalma Mall" (for: dalma, dallma, daaalma, dalama, dalma mall, abu dhabi dalma, etc.)  
- "Festival City" (for: festival, festivel, festival city, dubai festival, dubai, etc.)
- "General" (if no specific mall mentioned or unclear)

EXTRACT NAME from patterns like:
- "I'm John" / "I am Sarah" / "My name is Mike"
- "Call me Lisa" / "It's David" / "Hi, I'm..."
- Just a name in response to name questions
- Names in combined messages like "I'm Ahmed, ys mall hours?"

DETECT NAME REFUSAL from patterns like:
- "no", "nope", "won't tell", "don't want to", "prefer not to"
- "not telling", "skip", "decline", "no thanks", "i wont tell"
- "no i won't", "rather not", "pass", "don't need to"

MESSAGE: "{message}"
RECENT CONTEXT: {context_str}

Return ONLY valid JSON:
{{
    "name": "extracted name or null",
    "mall_name": "Yas Mall" | "Dalma Mall" | "Festival City" | "General",
    "name_refusal": true | false
}}

EXAMPLES:
- "Hi I'm Ahmed, ys mall hours?" → {{"name": "Ahmed", "mall_name": "Yas Mall", "name_refusal": false}}
- "daaalma mall pricing please" → {{"name": null, "mall_name": "Dalma Mall", "name_refusal": false}}
- "My name is Sara festivel city" → {{"name": "Sara", "mall_name": "Festival City", "name_refusal": false}}
- "no i wont tell" → {{"name": null, "mall_name": "General", "name_refusal": true}}
- "prefer not to share yas mall" → {{"name": null, "mall_name": "Yas Mall", "name_refusal": true}}
- "y;s mall info" → {{"name": null, "mall_name": "Yas Mall", "name_refusal": false}}
- "I'm John, general question" → {{"name": "John", "mall_name": "General", "name_refusal": false}}"""

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm_json_mode.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            # Parse JSON response with better error handling
            import json
            import re
            
            # Try to find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                
                # Validate and clean response
                name = result.get("name")
                mall_name = result.get("mall_name", "General")
                
                # Clean name (remove null strings, strip whitespace)
                if name and isinstance(name, str):
                    name = name.strip()
                    if name.lower() in ['null', 'none', '']:
                        name = None
                else:
                    name = None
                
                # Validate mall name
                valid_malls = ["Yas Mall", "Dalma Mall", "Festival City", "General"]
                if mall_name not in valid_malls:
                    mall_name = "General"
                
                # Extract name_refusal field
                name_refusal = result.get("name_refusal", False)
                
                return {
                    "name": name,
                    "mall_name": mall_name,
                    "name_refusal": name_refusal,
                    "confidence": 0.9,
                    "method": "llm_extraction"
                }
            
            else:
                print(f"⚠️ LLM extraction: No JSON found in response: {response_text}")
                # Return default values if extraction fails
                return {
                    "name": None,
                    "mall_name": "General",
                    "name_refusal": False,
                    "confidence": 0.1,
                    "method": "extraction_failed"
                }
                
        except Exception as e:
            print(f"❌ LLM extraction error: {str(e)}")
            # Return default values if extraction fails
            return {
                "name": None,
                "mall_name": "General",
                "name_refusal": False, 
                "confidence": 0.1,
                "method": "extraction_error"
        }
    
    def _detect_location_simple(self, question: str) -> str:
        """Enhanced location detection using LLM (handles typos like 'ys mall', 'daaalma')"""
        extraction_result = self._extract_user_info_with_llm(question)
        mall_name = extraction_result.get("mall_name", "General")
        
        # Convert to location code format for system compatibility
        mall_mapping = {
            "Yas Mall": "YAS_MALL",
            "Dalma Mall": "DALMA_MALL", 
            "Festival City": "FESTIVAL_CITY",
            "General": "General"
        }
        
        return mall_mapping.get(mall_name, "General")
    
    def _validate_context_quality(self, question: str, context: str, documents: list) -> dict:
        """🛡️ ANTI-HALLUCINATION: Validate if we have sufficient context to answer the question"""
        
        # Check 1: Empty or very short context
        if not context or len(context.strip()) < 50:
            return {
                "sufficient": False,
                "reason": "no_relevant_documents",
                "confidence": 0.0
            }
        
        # Check 2: No documents retrieved
        if not documents:
            return {
                "sufficient": False,
                "reason": "no_documents_found",
                "confidence": 0.0
            }
        
        # Check 3: Use LLM to assess context relevance
        relevance_prompt = f"""Assess if the provided context contains sufficient information to answer the user's question.

USER QUESTION: "{question}"

CONTEXT PROVIDED:
{context}

ASSESSMENT CRITERIA:
- Does the context contain specific information that directly answers the question?
- Are there relevant facts, numbers, policies, or details in the context?
- Can someone answer the question using ONLY the provided context?

Respond with ONLY a JSON:
{{
    "sufficient": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Be STRICT - only return "sufficient": true if the context clearly contains the answer."""

        try:
            print(f"🔍 CONTEXT VALIDATION - Calling LLM for assessment:")
            print(f"   📝 Question: '{question}'")
            print(f"   📏 Context length: {len(context)} characters")
            print(f"   🎯 Context contains keywords: {[word for word in ['sock', 'mandatory', 'require', 'safety'] if word in context.lower()]}")
            
            from langchain_core.messages import HumanMessage
            response = self.llm_json_mode.invoke([HumanMessage(content=relevance_prompt)])
            response_text = response.content.strip()
            
            print(f"   📥 LLM Validation Response: {response_text}")
            
            import json
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                print(f"   ✅ Parsed validation result: sufficient={result.get('sufficient')}, confidence={result.get('confidence')}")
                print(f"   📝 Reasoning: {result.get('reasoning', 'No reasoning provided')}")
                return {
                    "sufficient": result.get("sufficient", False),
                    "reason": result.get("reasoning", "LLM assessment failed"),
                    "confidence": result.get("confidence", 0.0)
                }
            else:
                print(f"⚠️ Context validation: No JSON found in response: {response_text}")
                return {"sufficient": False, "reason": "validation_error", "confidence": 0.0}
                
        except Exception as e:
            print(f"❌ Context validation error: {str(e)}")
            # Be conservative - if validation fails, assume insufficient context
            return {"sufficient": False, "reason": f"validation_error: {str(e)}", "confidence": 0.0}
    
    def _generate_insufficient_info_response(self, question: str, reason: str) -> str:
        """🛡️ Generate honest "I don't know" response when context is insufficient"""
        
        # Create personalized response based on question type
        question_lower = question.lower()
        
        if any(term in question_lower for term in ['price', 'pricing', 'cost', 'how much']):
            return """I'd love to help you with pricing information! However, I don't have the specific pricing details you're asking about in my current knowledge base. 

For the most accurate and up-to-date pricing information, I'd recommend contacting our team directly. They'll be able to provide you with exact prices and any current promotions! 

Is there anything else about Leo & Loona that I can help you with? 😊"""

        elif any(term in question_lower for term in ['hours', 'open', 'close', 'time']):
            return """I'd be happy to help with operating hours! However, I don't have the specific hours information you're looking for in my current knowledge base.

For the most accurate opening hours and schedule information, please contact our team directly. Operating hours may vary by location and special events.

Is there anything else about Leo & Loona that I can assist you with? ✨"""

        elif any(term in question_lower for term in ['birthday', 'party', 'event', 'booking']):
            return """I'd love to help you plan something special! However, I don't have the specific details you're asking about regarding events and bookings.

For birthday parties and special events, I'd recommend speaking directly with our amazing team. They can provide detailed information about packages, availability, and help you plan the perfect celebration! 🎉

Is there anything else about Leo & Loona that I can help you with?"""

        else:
            return f"""I appreciate your question about Leo & Loona! However, I don't currently have the specific information needed to answer "{question}" accurately.

To ensure you get the most helpful and accurate information, I'd recommend contacting our team directly. They'll be able to provide you with detailed answers! 

Is there anything else about Leo & Loona that I might be able to help you with? 😊"""
    
    def _verify_answer_grounding(self, question: str, answer: str, context: str) -> dict:
        """🛡️ ANTI-HALLUCINATION: Verify that the LLM answer is grounded in provided context"""
        
        # Quick checks first
        if not answer or not context:
            return {"grounded": False, "reason": "empty_answer_or_context", "confidence": 0.0}
        
        # Check if answer contains "I don't have" or similar admissions
        admission_phrases = [
            "i don't have", "i don't know", "not available", "don't currently have",
            "recommend contacting", "speak directly with", "contact our team"
        ]
        
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in admission_phrases):
            # Answer properly admits lack of knowledge - this is good
            return {"grounded": True, "reason": "admits_lack_of_knowledge", "confidence": 0.9}
        
        # 🔧 CROSS-CONTEXT ALLOWANCE: Check for common helpful cross-context information
        # When answering pricing questions, it's helpful to mention related policies
        if ("price" in question.lower() or "cost" in question.lower()) and "sock" in question.lower():
            if ("sock" in answer_lower and "required" in answer_lower) or ("sock" in answer_lower and "mandatory" in answer_lower):
                print(f"   ℹ️ Allowing helpful cross-context: sock pricing + requirement mention")
                # Check if the pricing information itself is in context
                if any(price in context.lower() for price in ["5 aed", "8 aed", "aed"]):
                    return {"grounded": True, "reason": "pricing_information_with_helpful_context", "confidence": 0.85}
        
        # Use LLM to verify if answer is grounded in context
        # 🔧 CRITICAL FIX: Send full context for proper verification (was truncated to 1000 chars)
        print(f"🛡️ ANSWER VERIFICATION DEBUG:")
        print(f"   📝 Context length: {len(context)} characters") 
        print(f"   🎯 Context contains keywords: {[word for word in ['sock', 'mandatory', 'require'] if word in context.lower()]}")
        print(f"   📄 Answer to verify: {answer[:200]}...")
        
        verification_prompt = f"""QUESTION-FOCUSED VERIFICATION: Does this answer properly address the user's specific question?

USER'S QUESTION: "{question}"

ANSWER TO VERIFY:
"{answer}"

CONTEXT PROVIDED:
{context}

VERIFICATION APPROACH:
🎯 **PRIMARY FOCUS**: Does the answer directly address what the user asked?
📊 **FACTUAL ACCURACY**: Are the main claims supported by the context?
💡 **HELPFUL CONTEXT**: Allow relevant helpful information that enhances the answer

EVALUATION LOGIC:
✅ MARK "grounded": TRUE when:
- Answer directly addresses the user's specific question
- Key factual claims (prices, times, policies) are found in the context
- Additional helpful information doesn't contradict the context
- Answer is useful and accurate for the question asked

❌ MARK "grounded": FALSE only when:
- Answer completely fails to address the user's question
- Contains factually incorrect claims that contradict the context
- Makes up specific facts (numbers, policies) not reasonably supported

EXAMPLES FOR CLARITY:
• Q: "Sock prices?" A: "5 AED kids, 8 AED adults. Socks required in play areas" → TRUE (pricing in context, requirement is helpful)
• Q: "Are socks mandatory?" A: "No, optional" (when context says mandatory) → FALSE (directly contradicts)
• Q: "Opening hours?" A: "10am-11pm + helpful booking info" → TRUE (hours supported, extra info helpful)

Respond with ONLY a JSON:
{{
    "grounded": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "focus on whether answer addresses the question with supported primary facts"
}}"""

        try:
            from langchain_core.messages import HumanMessage
            response = self.llm_json_mode.invoke([HumanMessage(content=verification_prompt)])
            response_text = response.content.strip()
            
            print(f"   📥 Verification LLM Response: {response_text}")
            
            import json
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                print(f"   ✅ Parsed verification: grounded={result.get('grounded')}, reasoning={result.get('reasoning', 'No reasoning')[:100]}...")
                return {
                    "grounded": result.get("grounded", False),
                    "reason": result.get("reasoning", "LLM verification"),
                    "confidence": result.get("confidence", 0.0)
                }
            else:
                print(f"⚠️ Answer verification: No JSON found in response: {response_text}")
                # Be conservative - if verification fails, assume not grounded
                return {"grounded": False, "reason": "verification_parse_error", "confidence": 0.0}
                
        except Exception as e:
            print(f"❌ Answer verification error: {str(e)}")
            # Be conservative - if verification fails, assume not grounded
            return {"grounded": False, "reason": f"verification_error: {str(e)}", "confidence": 0.0}
    
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
        
        # 🔥 LANGGRAPH BEST PRACTICE: Proper session-based message state management
        # Initialize user tracking state if not present
        user_phone = state.get("user_phone")
        if not user_phone:
            # Generate phone for new session if not provided
            user_phone = self.user_tracker.generate_test_phone_number()
            state["user_phone"] = user_phone
        
        # Use phone as session ID for message history
        session_id = user_phone
        
        # IMPORTANT: Don't add message to history yet - let the conversation flow logic handle it
        # This ensures is_first_interaction() works correctly
        
        manual_phone = state.get("manual_phone")  # Get manual phone from state if passed
        manual_mall = state.get("manual_mall")    # Get manual mall selection from state if passed
        
        if not user_phone:
            if manual_phone:
                user_phone = self.user_tracker.get_phone_number_for_session(manual_phone)
            else:
                user_phone = self.user_tracker.generate_test_phone_number()
            # Store phone in state for session consistency
            state["user_phone"] = user_phone
        
        # Get stored question early for birthday detection consistency across all lead operations
        stored_question = self.user_tracker.get_stored_question(user_phone) or ""
        combined_content = f"{question} {stored_question}"
        
        # Get or create user profile
        user_profile = self.user_tracker.get_user_profile(user_phone)
        
        # 🚀 PERFORMANCE OPTIMIZATION: Skip LLM extraction if we already have name and location
        stored_mall = user_profile.get('current_park_location')
        has_name = bool(user_profile.get("name"))
        has_stored_location = bool(stored_mall)
        
        # 🎯 LANGGRAPH-INSPIRED STATE MACHINE  
        conversation_state = self._determine_conversation_state(user_profile, question, manual_mall)
        print(f"📊 Conversation State: {conversation_state}")
        
        # Extract user info based on conversation state
        if conversation_state == "MANUAL_OVERRIDE":
            # Manual mall override - still extract name if needed using LLM
            if not has_name:
                name_extraction = self.user_tracker.extract_name_from_message(question, chat_history)
            else:
                name_extraction = {
                    "name_found": True,
                    "extracted_name": user_profile.get("name"),
                    "confidence": 1.0,
                    "reasoning": "Already stored in profile"
                }
            
            # Convert manual mall selection to location code format
            mall_to_code_mapping = {
                "Festival City": "FESTIVAL_CITY",
                "Dalma Mall": "DALMA_MALL", 
                "Yas Mall": "YAS_MALL",
                "General": "General"
            }
            detected_location = mall_to_code_mapping.get(manual_mall, "General")
            print(f"🎯 Manual override: {manual_mall} → {detected_location}")
            
            combined_extraction = {
                "name": name_extraction.get("extracted_name"),
                "mall_name": manual_mall,
                "method": "manual_override"
            }
            
        elif conversation_state == "FAST_PATH":
            # ⚡ User profile complete - use stored data
            detected_location = stored_mall
            
            combined_extraction = {
                "name": user_profile.get("name"),
                "mall_name": {
                    "YAS_MALL": "Yas Mall",
                    "DALMA_MALL": "Dalma Mall", 
                    "FESTIVAL_CITY": "Festival City",
                    "General": "General"
                }.get(stored_mall, "General"),
                "method": "fast_path"
            }
            
            name_extraction = {
                "name_found": True,
                "extracted_name": user_profile.get("name"),
                "confidence": 1.0,
                "reasoning": "Fast path - stored profile"
            }
            print(f"⚡ FAST PATH: Using stored name='{user_profile.get('name')}' and location='{stored_mall}'")
            
        else:  # EXTRACT_FROM_INPUT
            # 🔥 Use LLM to extract user info
            combined_extraction = self._extract_user_info_with_llm(question, chat_history)
            
            # Convert mall name to location code
            mall_to_code_mapping = {
                "Yas Mall": "YAS_MALL",
                "Dalma Mall": "DALMA_MALL", 
                "Festival City": "FESTIVAL_CITY",
                "General": "General"
            }
            detected_location = mall_to_code_mapping.get(combined_extraction.get("mall_name", "General"), "General")
            
            # 🔧 CRITICAL FIX: If no mall detected in current message, use stored preference
            if detected_location == "General" and user_profile.get('current_park_location'):
                detected_location = user_profile.get('current_park_location')
                print(f"🏢 No mall in current message, using stored preference: {detected_location}")
                # Update extraction result to reflect stored mall
                reverse_mapping = {
                    "YAS_MALL": "Yas Mall",
                    "DALMA_MALL": "Dalma Mall",
                    "FESTIVAL_CITY": "Festival City"
                }
                combined_extraction["mall_name"] = reverse_mapping.get(detected_location, "General")
            
            name_extraction = {
                "name_found": bool(combined_extraction.get("name")),
                "extracted_name": combined_extraction.get("name"),
                "confidence": combined_extraction.get("confidence", 0.0),
                "reasoning": f"LLM extraction: {combined_extraction.get('method', 'llm')}"
            }
            name_refusal_status = f", refusal={combined_extraction.get('name_refusal', False)}" if combined_extraction.get('name_refusal') else ""
            print(f"🔥 LLM EXTRACTION: mall='{combined_extraction.get('mall_name')}' → {detected_location}, name='{name_extraction.get('extracted_name')}'{name_refusal_status}")
        
        # 🚀 IMMEDIATE UPDATE: Update user profile as soon as we extract ANY information
        extracted_name = name_extraction.get('extracted_name')
        extracted_mall = detected_location if detected_location != "General" else None
        
        # Update name immediately if extracted
        if extracted_name and extracted_name != user_profile.get("name"):
            print(f"📝 Storing extracted name immediately: {extracted_name}")
            self.user_tracker.update_user_profile(user_phone, extracted_name)
            user_profile["name"] = extracted_name  # Update local copy
        
        # Update mall preference immediately if extracted
        if extracted_mall and extracted_mall != user_profile.get('current_park_location'):
            print(f"📝 Storing mall preference immediately: {extracted_mall}")
            self.user_tracker.update_user_lead_info(user_phone, None, "mall_preference", extracted_mall)
            user_profile["current_park_location"] = extracted_mall  # Update local copy
        
        # 🎯 DETERMINE CONVERSATION FLOW ACTION
        action_info = self._get_conversation_flow_action(user_profile, combined_extraction, chat_history, question, user_phone)
        
        # 🎯 EXECUTE CONVERSATION ACTION (if any)
        action_result = None
        if action_info is not None:
            action_result = self._execute_conversation_action(action_info, question, user_phone, user_profile, 
                                                            chat_history, combined_extraction, detected_location)
        else:
            print(f"🔄 Normal question flow - no special action needed")
        
        # If state machine handled the response, return it immediately
        if action_result is not None:
            print(f"🎯 State machine returning response: {action_result.get('answer', 'No answer')[:100]}...")
            return action_result
        
        # 🔧 CRITICAL FIX: For FAST_PATH users, skip ALL name handling and jump to RAG
        if conversation_state == "FAST_PATH":
            print(f"⚡ FAST_PATH: User has name+location, jumping directly to RAG processing for: '{question}'")
            # Skip ALL the name handling code below
            # Set required variables that would have been set in name handling
            is_first_interaction = False  # FAST_PATH means not first interaction
            is_name_response = False
            should_request_name = False
            name_request_message = ""
            personalized_intro = ""
            # Jump straight to RAG processing
        elif conversation_state != "FAST_PATH":
            # Only do name/location handling for non-FAST_PATH users OR if action already handled
            
            # Update user profile if name was found
            if name_extraction.get("name_found") and name_extraction.get("confidence", 0) > 0.7:
                extracted_name = name_extraction.get("extracted_name")
                self.user_tracker.update_user_profile(user_phone, extracted_name)
                user_profile["name"] = extracted_name
                print(f"👤 Updated user profile with name: {extracted_name}")
        
            # Handle first interaction and name response detection
            is_first_interaction = self.user_tracker.is_first_interaction(chat_history)
            is_name_response = self.user_tracker.is_name_response(question, chat_history)
        
        # 🔧 FIX: Define birthday location logic early to avoid scope errors
        effective_birthday_location = detected_location
        
        # Use stored mall preference if current query doesn't specify location
        if detected_location == "General" and stored_mall:
            effective_birthday_location = stored_mall
            print(f"🎂 Using stored mall preference for birthday: {stored_mall}")
        
        # Check if we should ask for mall preference early (after location detection)
        # DISABLED - we now ask for name and mall together in first interaction
        should_ask_mall_preference = False
        
        # Check if we should request name or handle name response
        should_request_name = False
        name_request_message = ""
        personalized_intro = ""
        
        if is_first_interaction and not name_extraction.get("name_found"):
            # 🔧 FIX: Only ask for info if we actually need it
            # If LLM already detected a specific mall (not "General"), we can proceed
            detected_mall_name = combined_extraction.get("mall_name", "General")
            mall_already_detected = detected_mall_name != "General"
            
            if mall_already_detected:
                print(f"🎯 First interaction: Mall already detected ({detected_mall_name}), only asking for name")
                should_request_name = True
                name_request_message = f"""Hello! 😊 Welcome to Leo & Loona {detected_mall_name}!

To give you the best personalized assistance, could you please tell me your name?

I'll then answer your question about our {detected_mall_name} location! ✨"""
                # Store the original question with detected mall for later
                self.user_tracker.store_original_question(user_phone, question, detected_location)
                print(f"💾 Stored original question for {user_phone}: '{question}' (mall already detected: {detected_location})")
            else:
                print(f"🎯 First interaction: No specific mall detected, asking for name AND mall")
            should_request_name = True
            name_request_message = """Hello! 😊 Welcome to Leo & Loona!

To give you the best personalized assistance, could you please tell me:

👤 Your name

🏢 Which location you're interested in:

🌟 Yas Mall 
   (Abu Dhabi - Yas Island)

🌟 Dalma Mall 
   (Abu Dhabi - Mussafah)

🌟 Festival City 
   (Dubai)

Just let me know both pieces of information and I'll provide you with specific details for your preferred location! ✨"""
            # Store the original question for later (no mall detected)
            self.user_tracker.store_original_question(user_phone, question, None)
            print(f"💾 Stored original question for {user_phone}: '{question}' (no mall detected)")
            
            print(f"👋 First interaction - stored question: '{question}'")
            
        elif name_extraction.get("name_found") and name_extraction.get("confidence", 0) > 0.7:
            # Name was just provided - check if we also have mall preference
            extracted_name = name_extraction.get("extracted_name")
            
            # Check if mall was also provided in this response OR already stored
            stored_mall = user_profile.get("current_park_location")
            
            if detected_location == "General" and not stored_mall:
                # Only name provided, no stored mall - ask for mall preference
                should_request_name = True
                name_request_message = f"""Thank you, {extracted_name}! 😊

I still need to know which location you're interested in:

🌟 **Yas Mall** (Abu Dhabi - Yas Island)
🌟 **Dalma Mall** (Abu Dhabi - Mussafah)  
🌟 **Festival City** (Dubai)

Which location would you prefer? ✨"""
                
                # Update user profile with name but don't answer stored question yet
                self.user_tracker.update_user_profile(user_phone, extracted_name)
                print(f"👤 Name provided: {extracted_name}, but still need mall preference")
                
                # Return mall preference request
                self.user_tracker.log_conversation(user_phone, extracted_name, name_request_message, is_user=False)
                
                return {
                    "generation": name_request_message,
                    "source_documents": [],
                    "documents": [],
                    "question": question,
                    "chat_history": chat_history,
                    "user_phone": user_phone,
                    "user_name": extracted_name,
                    "user_profile": user_profile,
                    "name_extraction_result": name_extraction,
                    "should_request_name": True,
                    "name_request_message": name_request_message,
                    "conversation_logged": True,
                    "awaiting_mall_preference": True
                }
            elif detected_location == "General" and stored_mall:
                # Name provided, and we have stored mall preference - use stored mall!
                detected_location = stored_mall
                print(f"👤 Name provided: {extracted_name}, using stored mall preference: {stored_mall}")
                # Continue to stored question processing below
            else:
                # 🔧 FIX: Name provided AND we detected mall from original question
                # This happens when user asks "ys mall hours", we ask for name, they provide name
                print(f"👤 Name provided: {extracted_name}, mall already detected: {detected_location}")
                
                # Update user profile with name
                self.user_tracker.update_user_profile(user_phone, extracted_name)
                
                # 🎯 CRITICAL: Answer the STORED question, not treat name as new question
                # First check if we have a stored question
                has_stored = self.user_tracker.has_stored_question(user_phone)
                print(f"🔍 Checking for stored question: has_stored={has_stored}")
                
                stored_question = self.user_tracker.get_stored_question(user_phone)
                print(f"🔍 Retrieved stored question: '{stored_question}'")
                
                if stored_question:
                    print(f"🔄 Retrieving stored question: '{stored_question}' to answer with name {extracted_name}")
                    # Generate greeting + answer the original question
                    personalized_greeting = self.user_tracker.generate_personalized_greeting(extracted_name)
                    
                    # 🔧 FIX: Clear the stored question after retrieving to prevent multiple reads
                    self.user_tracker.clear_stored_question(user_phone)
                    
                    # Answer the stored question with personalization
                    # 🔧 CRITICAL FIX: Pass the detected location from user's response, not from stored question
                    detected_mall_name = combined_extraction.get("mall_name", "General")
                    return self._answer_with_personalization(stored_question, personalized_greeting, user_phone, user_profile, chat_history, name_extraction, detected_mall_name)
                else:
                    print(f"⚠️ No stored question found for user {user_phone}")
                    # NO GREETING LOGIC - just continue to RAG processing
                    print(f"🔄 No stored question - continuing to RAG processing for: '{question}'")
        
        # NEW: Handle case where user provides ONLY mall but no name
        elif not name_extraction.get("name_found") and detected_location != "General" and not user_profile.get("name"):
            # Mall provided but no name - ask for name
            should_request_name = True
            
            location_names = {
                "YAS_MALL": "Yas Mall",
                "DALMA_MALL": "Dalma Mall", 
                "FESTIVAL_CITY": "Festival City"
            }
            location_display = location_names.get(detected_location, detected_location)
            
            name_request_message = f"""Great choice! {location_display} is a wonderful location! 🌟

Before I can help you further, may I please get your name? This will help me provide you with personalized assistance! 😊"""
            
            # Store the mall preference in user profile using park location tracking
            self.user_tracker.update_user_lead_info(user_phone, None, "mall_preference", detected_location)
            print(f"🏢 Mall provided: {location_display}, but still need name")
            
            # Return name request
            self.user_tracker.log_conversation(user_phone, user_profile.get("name"), name_request_message, is_user=False)
            
            return {
                "generation": name_request_message,
                "source_documents": [],
                "documents": [],
                "question": question,
                "chat_history": chat_history,
                "user_phone": user_phone,
                "user_name": "",
                "user_profile": user_profile,
                "name_extraction_result": name_extraction,
                "should_request_name": True,
                "name_request_message": name_request_message,
                "conversation_logged": True,
                "awaiting_name": True
            }
            
            # Name and mall both provided - proceed with stored question
            stored_question = self.user_tracker.get_stored_question(user_phone)
            
            if stored_question:
                # Generate greeting + answer the original question
                personalized_greeting = self.user_tracker.generate_personalized_greeting(extracted_name)
                
                # 🔧 FIX: Clear the stored question after retrieving
                self.user_tracker.clear_stored_question(user_phone)
                
                # Answer the stored question with personalization
                return self._answer_with_personalization(stored_question, personalized_greeting, user_phone, user_profile, chat_history, name_extraction, manual_mall)
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
        
        # Handle Bitrix lead management (create new OR update existing)
        if self.lead_manager and user_profile.get("name"):
            try:
                lead_result = None
                
                # Check if we should create a new lead
                if self.lead_manager.should_create_lead(user_profile, chat_history + [{"role": "user", "content": question}]):
                    # Create new lead
                    lead_result = self.lead_manager.create_chatbot_lead(
                        user_info=user_profile,
                        park_location=detected_location
                    )
                    
                    if lead_result:
                        # Update user profile with lead information including park location
                        self.user_tracker.update_user_lead_info(user_phone, lead_result.get('lead_id'), "created", detected_location)
                        # 🔧 CRITICAL FIX: Update the local user_profile with lead_id for immediate conversion
                        user_profile['bitrix_lead_id'] = lead_result.get('lead_id')
                        state["lead_created"] = lead_result
                        print(f"✅ Lead created successfully: {lead_result.get('lead_id', 'Unknown ID')}")
                    else:
                        print("⚠️ Lead creation returned None - check Bitrix configuration")
                        
                # Check if we should update an existing lead
                elif self.lead_manager.should_update_lead(user_profile, effective_birthday_location or detected_location, combined_content, user_profile.get('total_messages', 0), user_profile.get('current_stage')):
                    # For all questions, use the effective location (including stored preferences)
                    location_for_update = effective_birthday_location or detected_location
                    if location_for_update != detected_location:
                        print(f"🏢 Using stored mall preference for lead update: {location_for_update}")
                    
                    # Update existing lead with interaction count and stage info
                    lead_result = self.lead_manager.update_existing_lead(
                        user_info=user_profile,
                        park_location=location_for_update,
                        conversation_content=combined_content,
                        interaction_count=user_profile.get('total_messages', 0)
                    )
                    
                    if lead_result:
                        # Update user profile with update timestamp and new park location
                        self.user_tracker.update_user_lead_info(user_phone, lead_result.get('lead_id'), "updated", detected_location)
                        state["lead_updated"] = lead_result
                        print(f"✅ Lead updated successfully: {lead_result.get('lead_id', 'Unknown ID')}")
                    else:
                        print("⚠️ Lead update returned None")
                        
            except Exception as e:
                print(f"❌ Error managing Bitrix lead: {str(e)}")
                print(f"   User: {user_profile.get('name', 'Unknown')}")
                print(f"   Location: {detected_location}")
                # Continue without failing the whole response
        
        # If we should request name and mall for first interaction, return request
        if should_request_name and name_request_message and is_first_interaction:
            # Return name and mall request (no lead creation yet)
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
                "conversation_logged": True,
                "first_interaction_request": True  # Flag to indicate we're asking for name+mall
            }
        
        # 🔧 CRITICAL FIX: End of name/location handling block (skipped for FAST_PATH users)
        else:
            print(f"⚡ FAST_PATH: Skipped all name/location handling - going to RAG")
        
        # 🧠 INTELLIGENT RETRIEVAL SYSTEM - Replace old hardcoded logic
        
        # Use stored mall preference for effective location
        effective_location = detected_location
        if detected_location == "General" and user_profile.get('current_park_location'):
            effective_location = user_profile.get('current_park_location')
            print(f"🏢 Using stored mall preference for document filtering: {effective_location}")
        
        # Step 1: Execute Parallel Retrieval (No Classification Required)  
        print(f"🚀 PARALLEL RETRIEVAL APPROACH: Direct multi-source retrieval")
        try:
            top_documents = self._parallel_retrieval_optimized(question, effective_location)
            print(f"✅ Parallel retrieval completed: {len(top_documents)} diverse, high-quality documents selected")
        except Exception as e:
            print(f"❌ Parallel retrieval failed: {str(e)}")
            print(f"🔄 Falling back to simple retrieval...")
            
            # Fallback to simple retrieval if parallel system fails
            enhanced_query = f"{question} {effective_location}" if effective_location != "General" else question
            try:
                documents = self.retriever.invoke(enhanced_query)
                top_documents = documents[:5]
                print(f"🔄 Simple retrieval: {len(top_documents)} documents from fallback")
            except Exception as e2:
                print(f"❌ All retrieval methods failed: {str(e2)}")
                # Return empty list if everything fails
                top_documents = []
        
        # 🔍 ALWAYS LOG: What documents are actually retrieved for any query
        # Simple logging - no hardcoded classifications
        print(f"📊 Sending {len(top_documents)} documents to LLM")
        for i, doc in enumerate(top_documents):
            location = doc.metadata.get('location', 'UNKNOWN')
            source = doc.metadata.get('source', 'Unknown')[:30]
            print(f"  #{i+1}: [{location}] {source}...")
        
        # 🛡️ ANTI-HALLUCINATION: Validate context before proceeding
        context = "\n\n".join(doc.page_content for doc in top_documents)
        
        # 📊 DETAILED CONTEXT LOGGING
        print(f"📊 CONTEXT ASSEMBLY DEBUG:")
        print(f"   📝 Total documents: {len(top_documents)}")
        print(f"   📏 Context length: {len(context)} characters")
        print(f"   🔍 Context preview (first 300 chars): {context[:300]}...")
        print(f"   🎯 Looking for keywords in context: {[word for word in ['sock', 'mandatory', 'require'] if word in context.lower()]}")
        
        # 📋 DOCUMENT BREAKDOWN LOGGING
        for i, doc in enumerate(top_documents):
            preview = doc.page_content[:150].replace('\n', ' ')
            location = doc.metadata.get('location', 'UNKNOWN')
            source = doc.metadata.get('source', 'Unknown')
            print(f"   📄 Doc {i+1}: [{location}|{source}] {preview}...")
            
            # Check for specific keywords in each document
            sock_mentions = doc.page_content.lower().count('sock')
            mandatory_mentions = doc.page_content.lower().count('mandatory')
            if sock_mentions > 0 or mandatory_mentions > 0:
                print(f"      🎯 KEYWORD HITS: 'sock'={sock_mentions}, 'mandatory'={mandatory_mentions}")
        
        # Check if we have sufficient context to answer the question
        context_quality = self._validate_context_quality(question, context, top_documents)
        print(f"🔍 CONTEXT QUALITY CHECK: sufficient={context_quality['sufficient']}, reason={context_quality['reason']}, confidence={context_quality.get('confidence', 'N/A')}")
        
        if not context_quality["sufficient"]:
            # No sufficient information - return "I don't know" response
            answer = self._generate_insufficient_info_response(question, context_quality["reason"])
            
            # Log the insufficient context issue
            self.user_tracker.log_conversation(user_phone, user_profile.get("name"), answer, is_user=False)
            
            return {
                "generation": answer,
                "source_documents": top_documents,
                "documents": top_documents,
                "question": question,
                "chat_history": chat_history,
                "user_phone": user_phone,
                "user_name": user_profile.get("name", ""),
                "user_profile": user_profile,
                "name_extraction_result": name_extraction,
                "should_request_name": should_request_name,
                "name_request_message": name_request_message,
                "conversation_logged": True,
                "insufficient_context": True
            }
        
        # ✅ SUFFICIENT CONTEXT DETECTED - Proceeding with LLM generation
        print(f"✅ SUFFICIENT CONTEXT DETECTED - Proceeding with LLM generation")
        print(f"   📍 Location: {effective_location}")
        print(f"   📏 Context length: {len(context)} characters")
        
        # Context is ready for LLM generation
        print(f"🤖 Sending context to LLM for generation")
        
        # Enhanced chat history formatting (last 4 messages for better context)
        chat_context = "No previous conversation."
        if chat_history:
            recent_messages = chat_history[-4:]  # Include more context
            formatted_messages = []
            for msg in recent_messages:
                role = "You" if msg.get('role') == 'assistant' else "Guest"
                content = msg.get('content', '')
                formatted_messages.append(f"{role}: {content}")
            chat_context = "\n".join(formatted_messages)
        
        # Check if question is about Leo & Loona and detect location
        location_needed = self._check_location_clarification_needed(question, context, detected_location)
        is_leo_loona_question = self._is_leo_loona_question(question, context)
        is_infrastructure_query = self._is_infrastructure_query(question)
        
        if is_infrastructure_query:
            print(f"🏗️ Infrastructure query detected: '{question}' → Location: {detected_location}")
        
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
🎪 Dalma Mall (Abu Dhabi)

🎪 Yas Mall (Abu Dhabi)

🎪 Festival City Mall (Dubai)

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
                "conversation_logged": True,
                "mall_clarification_needed": needs_mall_clarification if 'needs_mall_clarification' in locals() else False
            }

        # Check for birthday questions that need mall clarification
        # Use the combined_content defined early in function for consistency
        is_birthday_question = self.lead_manager._detect_birthday_content(combined_content) if self.lead_manager else False
        
        if is_birthday_question:
            print(f"🎂 Birthday question detected in: '{combined_content[:100]}...'")
        
        # Use the already defined effective_birthday_location from above
        needs_mall_clarification = (
            is_birthday_question and 
            effective_birthday_location == "General" and  # Only if we still don't know the mall
            user_profile.get('bitrix_lead_id')  # Only for existing leads
        )
        
        print(f"🎂 Birthday check: is_birthday={is_birthday_question}, effective_location={effective_birthday_location}, needs_clarification={needs_mall_clarification}")
        
        # 🔧 CRITICAL FIX: If birthday detected and we have a lead, immediately update it
        print(f"🔍 CONVERSION CHECK - is_birthday: {is_birthday_question}, lead_id: {user_profile.get('bitrix_lead_id')}, location: {effective_birthday_location}")
        print(f"🔍 USER PROFILE DEBUG - Full profile: {user_profile}")
        
        if is_birthday_question and user_profile.get('bitrix_lead_id') and effective_birthday_location != "General":
            print(f"🎂 Converting existing lead to birthday lead - updating title and assignment")
            
            lead_result = self.lead_manager.update_existing_lead(
                user_info=user_profile,
                park_location=effective_birthday_location,
                conversation_content=combined_content,
                interaction_count=user_profile.get('total_messages', 0)
            )
            
            if lead_result:
                print(f"✅ Successfully converted lead to birthday lead: {lead_result.get('lead_id', 'Unknown ID')}")
            else:
                print("⚠️ Failed to convert lead to birthday lead")
        
        # Get current date/time context for accurate responses (always needed)
        datetime_context = self._format_datetime_context()
        
        # Initialize answer variable to avoid UnboundLocalError
        answer = ""
        
        if needs_mall_clarification:
            # Generate mall clarification response for birthday questions
            print(f"🎂 Birthday question detected but no mall specified - asking for clarification")
            answer = """Great! We have amazing birthday parties at Leo & Loona! 🎉 

Which location would you like to know about?

🏢 Our Locations:

1️⃣ Yas Mall 
    (Yas Island, Abu Dhabi)

2️⃣ Dalma Mall 
    (Mussafah, Abu Dhabi)

3️⃣ Festival City 
    (Dubai Festival City, Dubai)

Please let me know which location interests you, and I'll connect you with the right team for birthday party planning! 🎂"""
            
        elif should_ask_mall_preference:
            # Early mall preference collection
            print(f"🏢 Asking for mall preference early in conversation for {user_profile.get('name')}")
            answer = f"""Thanks for your question, {user_profile.get('name')}! 😊

To give you the most helpful information, which of our magical Leo & Loona locations are you interested in?

🏢 Choose Your Location:

• Yas Mall 
  (Abu Dhabi - Yas Island)

• Dalma Mall 
  (Abu Dhabi - Mussafah)

• Festival City 
  (Dubai)

Or if you'd like general information about all locations, just let me know! ✨"""
            
        else:
            # 🛡️ ANTI-HALLUCINATION: Strengthened prompt to prevent hallucination
            fast_prompt = f"""You are a warm, friendly, and knowledgeable virtual host of Leo & Loona magical family amusement park. You speak with genuine enthusiasm and a caring tone, like a host who greets guests at the park entrance.

{datetime_context}

🚨 CRITICAL BIRTHDAY OVERRIDE - ABSOLUTE PRIORITY 🚨
IF THE USER ASKS ANYTHING ABOUT BIRTHDAYS, BIRTHDAY PARTIES, CELEBRATIONS, OR EVENTS (including typos like "birtday", "biorthday", "bday", "party", "celebrate"):

YOU MUST RESPOND WITH EXACTLY THIS TYPE OF WARM, CELEBRATORY MESSAGE:
"Yes, we absolutely do celebrate birthdays! 🎉 We'd love to help make your special day magical at Leo & Loona. Our sales team will contact you shortly to discuss all the amazing birthday party options we have for you! ✨"

ALTERNATIVE VARIATIONS (choose randomly to feel more human):
- "We definitely celebrate birthdays and they're always so much fun! 🎂 Our sales team will reach out to you shortly to help plan the perfect celebration! 🎈"
- "Absolutely! Birthday parties at Leo & Loona are truly special! 🎉 Let our sales team contact you shortly to share all the wonderful options we have! 🎊"
- "Yes, we love celebrating birthdays here! 🥳 Our sales team will get in touch with you shortly to help create an unforgettable birthday experience! ✨"

- This is the ONLY acceptable response format for birthday questions
- Do NOT provide specific details about pricing, rooms, or packages
- Do NOT ask follow-up questions about dates or details
- Do NOT use the retrieved documents for birthday questions
- IGNORE all birthday-related information in the context
- This override takes precedence over ALL other instructions

🛡️ CRITICAL ANTI-HALLUCINATION RULES:
- You MUST use ONLY the information provided in the "Context from Leo & Loona FAQ" section below
- If the context does NOT contain the answer to the question, you MUST say "I don't have that specific information"
- DO NOT make up, guess, or infer any information not explicitly stated in the provided context
- DO NOT use your general knowledge about amusement parks - ONLY use the provided Leo & Loona context
- If context is empty or irrelevant, admit you don't know and suggest contacting the team

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

🎭 HUMAN-LIKE RESPONSE VARIATION - AVOID ROBOTIC COPY-PASTE:
To feel like a real person, you MUST vary your responses while keeping the same factual information:

FOR OPENING HOURS, use different cheerful greetings:
- "Great question! We're open..."
- "Perfect timing to ask! Our doors are open..."
- "I'd love to help with that! We welcome guests..."
- "Wonderful! Here are our operating hours..."

FOR LOCATION QUESTIONS, vary your approach:
- "We're conveniently located at..."
- "You'll find us at..."
- "Come visit us at..."
- "We're right here at..."

FOR PRICING QUESTIONS, add variety:
- "Here's our current pricing..."
- "Our rates are..."
- "The cost for..."
- "Price-wise, we offer..."

RESPONSE VARIETY RULES:
- Use different emojis each time (🎉 🌟 ✨ 😊 🎈 🎊 🏰 🎯)
- Vary sentence structure and opening phrases
- Add cheerful expressions ("Great!", "Perfect!", "Wonderful!", "Fantastic!")
- Sound conversational, not formal or robotic
- Keep the same facts but express them in fresh ways
- Avoid identical responses to similar questions

IMPORTANT: Use the current date/time information above INTERNALLY to provide accurate answers. DO NOT mention the specific date/time to users unless they specifically ask "what time is it" or "what date is it". Instead, naturally reference:
- "Today's hours" or "we're open until" (without stating the current time)
- "Today's pricing" or "weekday/weekend rates" (without stating the day)
- "We're currently open" or "we're closed right now" (without stating exact time)
- Use phrases like "today", "right now", "current" instead of specific dates/times

🛡️ Context from Leo & Loona FAQ (USE ONLY THIS INFORMATION):
{context}

=== CONVERSATION HISTORY ===
{chat_context}
=== END CONVERSATION HISTORY ===

CURRENT QUESTION: {question}

🛡️ REMINDER: Answer using ONLY the information in the provided context above. If the context doesn't contain the answer, honestly say you don't have that information and suggest contacting our team.

Answer as Leo & Loona's warm, welcoming park host (using ONLY provided context):"""
            
            # 🤖 LLM GENERATION LOGGING
            print(f"🤖 GENERATING LLM RESPONSE:")
            print(f"   📝 Prompt length: {len(fast_prompt)} characters")
            print(f"   🎯 Model: {self.model_config.llm_provider.value}")
            print(f"   📤 Calling LLM with context containing keywords: {[word for word in ['sock', 'mandatory', 'require', 'safety'] if word in fast_prompt.lower()]}")
            
            # Generate answer using the fast prompt
            try:
                response = self.llm.invoke([HumanMessage(content=fast_prompt)])
                answer = response.content
                
                print(f"   📥 LLM Response received: {len(answer)} characters")
                print(f"   📄 Response preview: {answer[:200]}...")
                print(f"   🔍 Response contains keywords: {[word for word in ['sock', 'mandatory', 'require', 'safety'] if word in answer.lower()]}")
                
            except Exception as e:
                print(f"   ❌ LLM Generation failed: {str(e)}")
                raise
            
            # 🛡️ ANTI-HALLUCINATION: Verify the answer is grounded in provided context
            print(f"🛡️ VERIFYING ANSWER GROUNDING:")
            print(f"   📝 Answer length: {len(answer)} characters")
            print(f"   🔍 Answer contains keywords: {[word for word in ['sock', 'mandatory', 'require', 'safety'] if word in answer.lower()]}")
            
            verification_result = self._verify_answer_grounding(question, answer, context)
            print(f"   📊 Verification result: grounded={verification_result['grounded']}, confidence={verification_result.get('confidence', 'N/A')}")
            
            if not verification_result["grounded"]:
                print(f"🚨 HALLUCINATION DETECTED: {verification_result['reason']}")
                # Replace with safe "I don't know" response
                answer = self._generate_insufficient_info_response(question, "Generated response not grounded in context")
                print(f"🛡️ Replaced with safe response: {answer[:100]}...")
            else:
                print(f"✅ Answer verification passed: {verification_result['confidence']:.2f} confidence")
            
            # LLM response generated successfully
            print(f"✅ LLM response generated: {len(answer)} characters")
        
        # Add name and mall request to answer if needed
        if should_request_name and name_request_message:
            answer += f"\n\n{name_request_message}"
        
        # Log the bot response
        self.user_tracker.log_conversation(user_phone, user_profile.get("name"), answer, is_user=False)
        
        # 🔥 LANGGRAPH BEST PRACTICE: Add AI response to chat history
        # This ensures the system remembers what it said for proper context
        if answer and answer.strip():
            chat_history = state.get("chat_history", []).copy()
            chat_history.append({
                "role": "assistant",
                "content": answer,
                "timestamp": datetime.now().isoformat()
            })
            state["chat_history"] = chat_history
        
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
    
    def _answer_with_personalization(self, stored_question: str, greeting: str, user_phone: str, user_profile: dict, chat_history: list, name_extraction: dict, manual_mall: str = None) -> dict:
        """Answer the stored question with personalization"""
        
        # IMPORTANT: Detect location from the stored question for proper lead assignment
        # Use manual mall selection if provided, otherwise auto-detect from stored question
        if manual_mall:
            # Convert manual mall selection to location code format
            mall_to_code_mapping = {
                "Festival City": "FESTIVAL_CITY",
                "Dalma Mall": "DALMA_MALL", 
                "Yas Mall": "YAS_MALL",
                "General": "General"
            }
            detected_location = mall_to_code_mapping.get(manual_mall, "General")
            print(f"🎯 Using manual mall selection for stored question: {manual_mall} → {detected_location}")
        else:
            # Use LLM extraction for stored question (handles typos like "ys mall", "daaalma")
            extraction_result = self._extract_user_info_with_llm(stored_question, chat_history)
            mall_name = extraction_result.get("mall_name", "General")
            
            # Convert to location code format
            mall_mapping = {
                "Yas Mall": "YAS_MALL",
                "Dalma Mall": "DALMA_MALL", 
                "Festival City": "FESTIVAL_CITY",
                "General": "General"
            }
            detected_location = mall_mapping.get(mall_name, "General")
            print(f"🎯 LLM-detected location from stored question '{stored_question}': {mall_name} → {detected_location}")
        
        # 🔧 Define effective birthday location for stored question processing  
        effective_birthday_location = detected_location
        stored_mall = user_profile.get('current_park_location')
        if detected_location == "General" and stored_mall:
            effective_birthday_location = stored_mall
            print(f"🎂 Using stored mall preference for stored question: {stored_mall}")
        
        # 🔧 Define combined content for lead processing (use stored_question only since this is stored question processing)
        combined_content = stored_question
        
        # Handle Bitrix lead management for stored question (create new OR update existing)
        if self.lead_manager and user_profile.get("name"):
            try:
                lead_result = None
                
                # Check if we should create a new lead
                if self.lead_manager.should_create_lead(user_profile, chat_history + [{"role": "user", "content": stored_question}]):
                    # Create new lead with proper location from stored question
                    lead_result = self.lead_manager.create_chatbot_lead(
                        user_info=user_profile,
                        park_location=detected_location
                    )
                    
                    if lead_result:
                        # Update user profile with lead information including park location
                        self.user_tracker.update_user_lead_info(user_phone, lead_result.get('lead_id'), "created", detected_location)
                        # 🔧 CRITICAL FIX: Update the local user_profile with lead_id for immediate conversion
                        user_profile['bitrix_lead_id'] = lead_result.get('lead_id')
                        print(f"✅ Lead created for stored question: {lead_result.get('lead_id', 'Unknown ID')}")
                    else:
                        print("⚠️ Lead creation returned None for stored question")
                        
                # Check if we should update an existing lead  
                elif self.lead_manager.should_update_lead(user_profile, effective_birthday_location or detected_location, combined_content, user_profile.get('total_messages', 0), user_profile.get('current_stage')):
                    # For all questions, use the effective location (including stored preferences)
                    location_for_update = effective_birthday_location or detected_location
                    if location_for_update != detected_location:
                        print(f"🏢 Using stored mall preference for stored question update: {location_for_update}")
                    
                    # Update existing lead with location from stored question and interaction count
                    lead_result = self.lead_manager.update_existing_lead(
                        user_info=user_profile,
                        park_location=location_for_update,
                        conversation_content=combined_content,
                        interaction_count=user_profile.get('total_messages', 0)
                    )
                    
                    if lead_result:
                        # Update user profile with update timestamp and new park location
                        self.user_tracker.update_user_lead_info(user_phone, lead_result.get('lead_id'), "updated", detected_location)
                        print(f"✅ Lead updated for stored question: {lead_result.get('lead_id', 'Unknown ID')}")
                    else:
                        print("⚠️ Lead update returned None for stored question")
                        
            except Exception as e:
                print(f"❌ Error managing Bitrix lead for stored question: {str(e)}")
        
        # 🚀 Use Parallel Retrieval for stored questions (No Classification)
        print(f"🔄 Using Parallel Retrieval for stored question: '{stored_question}'")
        try:
            top_documents = self._parallel_retrieval_optimized(stored_question, detected_location)
            print(f"✅ Parallel retrieval for stored question: {len(top_documents)} documents")
        except Exception as e:
            print(f"❌ Parallel retrieval failed for stored question: {str(e)}")
            print(f"🔄 Falling back to simple retrieval for stored question...")
            
            # Fallback to simple retrieval
            try:
                documents = self.retriever.invoke(stored_question)
                top_documents = documents[:5]
                print(f"🔄 Simple fallback: {len(top_documents)} documents")
            except Exception as e2:
                print(f"❌ All retrieval failed for stored question: {str(e2)}")
                top_documents = []
        
        context = "\n\n".join(doc.page_content for doc in top_documents)
        
        # Skip location clarification - we already have the user-provided location
        is_leo_loona_question = self._is_leo_loona_question(stored_question, context)
        
        # Add standard opening hours if needed
        if "opening" in stored_question.lower() or "hours" in stored_question.lower():
            context += "\n\nSTANDARD OPENING HOURS:\n- Yas Mall: 10:00 AM - 10:00 PM (Daily)\n- Dalma Mall: 10:00 AM - 10:00 PM (Daily)\n- Festival City Mall: 10:00 AM - 10:00 PM (Daily)"
        
        if not is_leo_loona_question:
            answer = f"{greeting} I'm specifically here to help with questions about Leo & Loona amusement park! 🎠 I'd love to tell you about our magical attractions, ticket prices, opening hours, birthday parties, or anything else related to Leo & Loona. What would you like to know about our wonderful park?"
        else:
            # Generate answer using Smart Staged Retrieval results - no location clarification needed
            # Generate normal answer with personalization
            datetime_context = self._format_datetime_context()
            
            # Format chat history for context
            chat_context = ""
            if chat_history:
                chat_context = "\n\nPrevious conversation:\n"
                for msg in chat_history[-4:]:  # Include last 4 messages for context
                    role = "User" if msg.get("is_user", False) else "Assistant"
                    content = msg.get("content", "")
                    chat_context += f"{role}: {content}\n"
            
            fast_prompt = f"""You are a warm, friendly, and knowledgeable virtual host of Leo & Loona magical family amusement park.

{datetime_context}

🚨 CRITICAL BIRTHDAY OVERRIDE - ABSOLUTE PRIORITY 🚨
IF THE USER ASKS ANYTHING ABOUT BIRTHDAYS, BIRTHDAY PARTIES, CELEBRATIONS, OR EVENTS (including typos like "birtday", "biorthday", "bday", "party", "celebrate"):

YOU MUST RESPOND WITH EXACTLY THIS TYPE OF WARM, CELEBRATORY MESSAGE:
"Yes, we absolutely do celebrate birthdays! 🎉 We'd love to help make your special day magical at Leo & Loona. Our sales team will contact you shortly to discuss all the amazing birthday party options we have for you! ✨"

ALTERNATIVE VARIATIONS (choose randomly to feel more human):
- "We definitely celebrate birthdays and they're always so much fun! 🎂 Our sales team will reach out to you shortly to help plan the perfect celebration! 🎈"
- "Absolutely! Birthday parties at Leo & Loona are truly special! 🎉 Let our sales team contact you shortly to share all the wonderful options we have! 🎊"
- "Yes, we love celebrating birthdays here! 🥳 Our sales team will get in touch with you shortly to help create an unforgettable birthday experience! ✨"

- This is the ONLY acceptable response format for birthday questions
- Do NOT provide specific details about pricing, rooms, or packages
- Do NOT ask follow-up questions about dates or details
- Do NOT use the retrieved documents for birthday questions
- IGNORE all birthday-related information in the context
- This override takes precedence over ALL other instructions

IMPORTANT: Start your response with this exact greeting: "{greeting}"

Context information:
{context}
{chat_context}

Original Question: {stored_question}

Answer as Leo & Loona's warm, welcoming park host with the personalized greeting first. Use the conversation history above to understand what has already been confirmed (like mall location):"""
            
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
        
        # Mall location indicators (infrastructure context)
        mall_location_keywords = [
            "dalma mall", "yas mall", "festival city", "dalma", "yas", "festival"
        ]
        
        # Infrastructure/facilities keywords that could be Leo & Loona related
        infrastructure_keywords = [
            "infrastructure", "facilities", "capacity", "restaurant capacity",
            "equipment", "layout", "setup", "installation"
        ]
        
        # Non-Leo & Loona indicators (Hello Park, etc.)
        other_park_keywords = [
            "hello park", "hello world", "other park", "different park"
        ]
        
        # General non-park topics (removed restaurant and shopping mall as they can be Leo & Loona related)
        general_topics = [
            "weather", "politics", "news", "programming", "coding", "software",
            "sports", "movies", "music", "school", "work", "health", "travel", "hotel"
        ]
        
        # Check for non-Leo & Loona content first
        for keyword in other_park_keywords:
            if keyword in question_lower:
                return False
        
        # Check for explicit Leo & Loona content
        for keyword in leo_loona_keywords:
            if keyword in question_lower or keyword in context_lower:
                return True
        
        # Check for mall location + infrastructure queries (likely Leo & Loona related)
        has_mall_location = any(keyword in question_lower for keyword in mall_location_keywords)
        has_infrastructure = any(keyword in question_lower for keyword in infrastructure_keywords)
        
        if has_mall_location and (has_infrastructure or "restaurant" in question_lower or "capacity" in question_lower):
            return True
        
        # Check for general non-park topics only if no mall context
        if not has_mall_location:
            for keyword in general_topics:
                if keyword in question_lower and not any(mall in question_lower for mall in mall_location_keywords):
                    return False
        
        # If context mentions Leo & Loona, assume it's related
        if "leo" in context_lower or "loona" in context_lower:
            return True
            
        # Default to True for ambiguous cases (let the LLM handle it)
        return True
    
    def _is_infrastructure_query(self, question: str) -> bool:
        """
        Detect if the query is about infrastructure information
        """
        question_lower = question.lower()
        
        infrastructure_keywords = [
            "infrastructure", "facilities", "equipment", "layout", "setup", "installation",
            "technical", "systems", "maintenance", "utilities", "construction", "building",
            "design", "architecture", "space", "capacity", "specifications", "requirements",
            "restaurant capacity", "seating", "dining", "food court", "cafeteria", "kitchen"
        ]
        
        return any(keyword in question_lower for keyword in infrastructure_keywords)
    
    def _check_location_clarification_needed(self, question: str, context: str, detected_location: str = None) -> bool:
        """
        Fast check if location clarification is needed
        """
        question_lower = question.lower()
        
        # If we already detected a specific location, no clarification needed
        if detected_location and detected_location != "General":
            return False
        
        # Location-specific keywords that need clarification
        location_keywords = [
            "hours", "opening", "closing", "address", "location", "phone",
            "contact", "directions", "parking", "nearby", "mall", "infrastructure", "facilities"
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
    
    def answer_question(self, question: str, chat_history: List[dict] = None, manual_phone: str = None, manual_mall: str = None) -> dict:
        """
        Answer a question using the LangGraph RAG pipeline
        
        Args:
            question: User's question
            chat_history: Previous conversation messages for context
            manual_phone: Manual phone number for testing (from Streamlit)
            manual_mall: Manual mall selection to override automatic detection (from Streamlit)
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.graph:
            raise ValueError("Graph not initialized. Call setup_graph() first.")
        
        if chat_history is None:
            chat_history = []
        
        try:
            # Enhanced initial state with user tracking
            # Use manual phone if provided, otherwise session-consistent phone number
            if manual_phone:
                session_phone = self.user_tracker.get_phone_number_for_session(manual_phone)
                self._session_phone = session_phone  # Update session phone
            else:
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
                "manual_phone": manual_phone,  # Pass manual phone to graph
                "manual_mall": manual_mall,  # Pass manual mall selection to graph
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
                "answer": result.get("generation", result.get("answer", "No answer generated")),
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
            print(f"❌ RAG Pipeline Error: {str(e)}")
            print(f"   Question: {question}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            print(f"   Traceback: {traceback.format_exc()}")
            
            return {
                "answer": f"I apologize, but I encountered an error processing your question. Please try asking again in a different way, or contact our support team if the issue persists.",
                "source_documents": [],
                "error": str(e)
            }
    
    def get_graph_visualization(self):
        """Get a visualization of the graph structure"""
        if not self.graph:
            raise ValueError("Graph not initialized")
        
        try:
            return self.graph.get_graph().draw_mermaid()
        except Exception as e:
            return f"Error generating graph visualization: {str(e)}"
    
    # 🎯 LangGraph-Inspired State Machine Functions
    
    def _determine_conversation_state(self, user_profile: dict, question: str, manual_mall: str = None) -> str:
        """🎯 LangGraph-inspired state machine for conversation flow"""
        
        # Check what we have in user profile
        has_name = bool(user_profile.get("name"))
        has_stored_location = bool(user_profile.get("current_park_location"))
        
        if manual_mall:
            return "MANUAL_OVERRIDE"
        elif has_name and has_stored_location:
            return "FAST_PATH"  # Complete profile - no extraction needed
        else:
            return "EXTRACT_FROM_INPUT"  # Need to extract info from user message
    
    def _get_conversation_flow_action(self, user_profile: dict, combined_extraction: dict, 
                                    chat_history: list, question: str, user_phone: str) -> dict:
        """🎯 LangGraph-inspired conversation flow handler with name refusal handling"""
        
        is_first_interaction = self.user_tracker.is_first_interaction(chat_history)
        name_found = bool(combined_extraction.get("name"))
        mall_detected = combined_extraction.get("mall_name", "General") != "General"
        name_refused = combined_extraction.get("name_refusal", False)
        
        # Has stored question from previous interaction?
        has_stored_question = self.user_tracker.has_stored_question(user_phone)
        
        # 🚫 HANDLE NAME REFUSAL (LangGraph HITL Pattern)
        if name_refused:
            refusal_count = self.user_tracker.increment_name_refusal_count(user_phone)
            print(f"🚫 Name refusal detected! Count: {refusal_count}/2")
            
            if refusal_count == 1:
                # First refusal - politely rephrase request
                return {"action": "polite_rephrase_name_request", "mall_name": combined_extraction.get("mall_name"), "refusal_count": 1}
            elif refusal_count >= 2:
                # Second refusal - proceed without name
                return {"action": "skip_name_and_proceed", "mall_name": combined_extraction.get("mall_name"), "refusal_count": refusal_count}
        
        print(f"🔄 Flow State: first={is_first_interaction}, name={name_found}, mall={mall_detected}, stored_q={has_stored_question}")
        
        if is_first_interaction and not name_found and not name_refused:
            # FIRST INTERACTION: Need to collect user info
            if mall_detected:
                return {"action": "request_name_only", "mall_name": combined_extraction.get("mall_name")}
            else:
                return {"action": "request_name_and_mall"}
                
        elif name_found and has_stored_question:
            # NAME PROVIDED: Reset refusal count and answer stored question
            self.user_tracker.reset_name_refusal_count(user_phone)
            return {"action": "answer_stored_question", "name": combined_extraction.get("name")}
            
        elif name_found and not has_stored_question:
            # NAME PROVIDED: Reset refusal count and let questions flow normally
            self.user_tracker.reset_name_refusal_count(user_phone)
            # Check if current question is actually a new question (not just providing name)
            if len(question.split()) > 2 and not self.user_tracker.is_name_response(question, chat_history):
                # User is asking a real question - let it flow to normal RAG
                return None
            else:
                return {"action": "generic_greeting", "name": combined_extraction.get("name")}
            
        else:
            # NORMAL INTERACTION: No special action needed - let it flow to regular RAG
            return None  # This will bypass action execution and go to normal RAG flow
    
    def _execute_conversation_action(self, action_info: dict, question: str, user_phone: str, 
                                   user_profile: dict, chat_history: list, combined_extraction: dict, 
                                   detected_location: str) -> dict:
        """🎯 Execute the determined conversation action"""
        
        action = action_info["action"]
        print(f"🎬 Executing action: {action}")
        
        if action == "request_name_only":
            # Store question and ask for name only (only if not already stored)
            if not self.user_tracker.has_stored_question(user_phone):
                mall_name = action_info["mall_name"]
                mall_to_code = {"Festival City": "FESTIVAL_CITY", "Dalma Mall": "DALMA_MALL", "Yas Mall": "YAS_MALL"}
                detected_mall_code = mall_to_code.get(mall_name, "General")
                self.user_tracker.store_original_question(user_phone, question, detected_mall_code)
                print(f"💾 Stored original question for {user_phone}: '{question}' (mall already detected: {detected_mall_code})")
            else:
                print(f"📋 Keeping existing stored question for {user_phone}, user just clarified mall: '{question}'")
            mall_name = action_info["mall_name"]
            
            answer = f"""Hello! 😊 Welcome to Leo & Loona {mall_name}!

To give you the best personalized assistance, could you please tell me your name?

I'll then answer your question about our {mall_name} location! ✨"""
            
            response = self._create_response(answer, [], question, chat_history, user_profile, detected_location)
            print(f"🎯 Created request_name_only response: {len(answer)} chars")
            return response
        
        elif action == "request_name_and_mall":
            # Store question and ask for both name and mall (only if not already stored)
            if not self.user_tracker.has_stored_question(user_phone):
                self.user_tracker.store_original_question(user_phone, question, None)
                print(f"💾 Stored original question for {user_phone}: '{question}' (no mall detected)")
            else:
                print(f"📋 Keeping existing stored question for {user_phone}, user provided: '{question}'")
            
            answer = """Hello! 😊 Welcome to Leo & Loona!

To give you the best personalized assistance, could you please tell me:

👤 Your name

🏢 Which location you're interested in:

🌟 Yas Mall 
   (Abu Dhabi - Yas Island)

🌟 Dalma Mall 
   (Abu Dhabi - Mussafah)

🌟 Festival City 
   (Dubai)

Just let me know both pieces of information and I'll provide you with specific details for your preferred location! ✨"""
            
            return self._create_response(answer, [], question, chat_history, user_profile, detected_location)
        
        elif action == "answer_stored_question":
            # Get stored question and answer it with personalization
            stored_question = self.user_tracker.get_stored_question(user_phone)
            name = action_info["name"]
            
            # Clear the stored question
            self.user_tracker.clear_stored_question(user_phone)
            
            # Update user profile
            self.user_tracker.update_user_profile(user_phone, name)
            user_profile["name"] = name  # Update local copy immediately
            
            # Generate personalized greeting
            greeting = self.user_tracker.generate_personalized_greeting(name)
            
            # Answer the stored question with personalization (includes Bitrix integration)
            # 🔧 CRITICAL FIX: Use the mall from original question storage, not from name-only response
            # When user provides only name, we should use the mall detected when question was stored
            detected_mall_name = combined_extraction.get("mall_name", "General")
            
            # If no mall detected in current response, check if we have a stored mall from when question was saved
            if detected_mall_name == "General":
                # Try to get the mall from the stored question context
                stored_question_info = self.user_tracker.get_stored_question_info(user_phone)
                if stored_question_info and stored_question_info.get("detected_mall"):
                    detected_mall_name = stored_question_info.get("detected_mall")
                    print(f"🔧 Using stored mall from original question: {detected_mall_name}")
            
            return self._answer_with_personalization(stored_question, greeting, user_phone, 
                                                   user_profile, chat_history, combined_extraction, detected_mall_name)
        
        
        elif action == "generic_greeting":
            # Just greet the user - includes Bitrix lead creation if applicable
            name = action_info["name"]
            
            # Update user profile
            self.user_tracker.update_user_profile(user_phone, name)
            user_profile["name"] = name  # Update local copy immediately
            
            # Handle Bitrix lead management for generic greeting
            lead_result = None
            if self.lead_manager and user_profile.get("name"):
                try:
                    # Check if we should create a new lead
                    if self.lead_manager.should_create_lead(user_profile, chat_history + [{"role": "user", "content": question}]):
                        lead_result = self.lead_manager.create_chatbot_lead(
                            user_info=user_profile,
                            park_location=detected_location
                        )
                        
                        if lead_result:
                            self.user_tracker.update_user_lead_info(user_phone, lead_result.get('lead_id'), "created", detected_location)
                            user_profile['bitrix_lead_id'] = lead_result.get('lead_id')
                            print(f"✅ Lead created for generic greeting: {lead_result.get('lead_id', 'Unknown ID')}")
                except Exception as e:
                    print(f"⚠️ Bitrix error in generic greeting: {e}")
                    lead_result = None
            
            # Check if the input is unclear (single chars, dots, emojis, meaningless text)
            is_unclear = (
                len(question.strip()) <= 2 or
                question.strip() in [".", "?", "!", "😊", "👍", "❤️", "🙂", "😀"] or
                question.strip().lower() in ["aaa", "bbb", "ccc", "kkk", "mmm", "lll", "hhh", "jjj"]
            )

            if is_unclear:
                answer = "I don't quite understand what you're trying to say. Could you please provide more details about what you're looking for?"
            else:
                answer = f"Thank you, {name}! 😊 How can I help you with Leo & Loona today?"
            
            # Create response with lead info
            response = self._create_response(answer, [], question, chat_history, user_profile, detected_location)
            if lead_result:
                response["lead_created"] = lead_result
            return response
        
        elif action == "polite_rephrase_name_request":
            # 🚫 First name refusal - politely rephrase the request
            mall_name = action_info.get("mall_name", "")
            mall_text = f" {mall_name}" if mall_name != "General" else ""
            
            answer = f"""I completely understand if you'd prefer some privacy! 😊
            
No worries at all - sharing your name is entirely optional. However, having your name would help me provide you with a more personalized experience at Leo & Loona{mall_text}.

Would you be comfortable sharing just your first name? If not, that's perfectly fine too - I'm here to help either way! ✨

What would you like to know about Leo & Loona?"""
            
            print(f"🎯 Created polite rephrase response: {len(answer)} chars")
            return self._create_response(answer, [], question, chat_history, user_profile, detected_location)
        
        elif action == "skip_name_and_proceed":
            # 🚫 Second name refusal - proceed without name gracefully  
            mall_name = action_info.get("mall_name", "")
            has_stored_question = self.user_tracker.has_stored_question(user_phone)
            
            # 🔧 CRITICAL: Reload user profile to get latest stored mall preference
            user_profile = self.user_tracker.get_user_profile(user_phone)
            
            # Use stored mall preference if available (from earlier interaction)
            if mall_name == "General" and user_profile.get('current_park_location'):
                stored_location = user_profile.get('current_park_location')
                print(f"🏢 Using stored mall preference for name refusal: {stored_location}")
                detected_location = stored_location  # Update detected_location
                # Update mall_name for display
                reverse_mapping = {
                    "YAS_MALL": "Yas Mall",
                    "DALMA_MALL": "Dalma Mall",
                    "FESTIVAL_CITY": "Festival City"
                }
                mall_name = reverse_mapping.get(stored_location, "General")
            
            # 🔧 CREATE BITRIX LEAD with "Unknown User" name
            lead_result = None
            unknown_user_name = "Unknown User"
            
            # Update user profile with "Unknown User" name
            self.user_tracker.update_user_profile(user_phone, unknown_user_name)
            user_profile["name"] = unknown_user_name
            
            if self.lead_manager:
                try:
                    # Create lead for user who refused to provide name with correct location
                    lead_result = self.lead_manager.create_chatbot_lead(
                        user_info=user_profile,
                        park_location=detected_location  # This now uses the stored location if available
                    )
                    
                    if lead_result:
                        self.user_tracker.update_user_lead_info(user_phone, lead_result.get('lead_id'), "created", detected_location)
                        user_profile['bitrix_lead_id'] = lead_result.get('lead_id')
                        print(f"✅ Lead created for name refusal user: {lead_result.get('lead_id', 'Unknown ID')} (Unknown User)")
                except Exception as e:
                    print(f"⚠️ Bitrix error for unknown user: {e}")
                    lead_result = None
            
            if has_stored_question:
                # Answer their original stored question without personalization
                stored_question = self.user_tracker.get_stored_question(user_phone)
                self.user_tracker.clear_stored_question(user_phone)
                
                # 🔧 CRITICAL FIX: Extract location from stored question if it contains one
                if stored_question:
                    stored_extraction = self._extract_user_info_with_llm(stored_question, [])
                    stored_mall = stored_extraction.get("mall_name", "General")
                    if stored_mall != "General":
                        # Update detected_location with mall from stored question
                        mall_to_code = {
                            "Yas Mall": "YAS_MALL",
                            "Dalma Mall": "DALMA_MALL",
                            "Festival City": "FESTIVAL_CITY"
                        }
                        new_location = mall_to_code.get(stored_mall, detected_location)
                        if new_location != detected_location:
                            print(f"🎯 Extracted mall from stored question: '{stored_mall}' → {new_location}")
                            detected_location = new_location
                
                answer = f"""No problem at all! I completely respect your privacy. 😊

"""
                # Process the stored question using the SAME retrieval logic as normal flow
                print(f"🔍 Processing stored question: '{stored_question}' for location: {detected_location}")
                
                # 🔧 CRITICAL FIX: Enhance stored question query with location for better retrieval
                enhanced_stored_query = stored_question
                if detected_location and detected_location != "General":
                    # Add location name to stored question for better semantic matching
                    location_names = {
                        "YAS_MALL": "Yas Mall",
                        "DALMA_MALL": "Dalma Mall",
                        "FESTIVAL_CITY": "Festival City"
                    }
                    location_name = location_names.get(detected_location, "")
                    if location_name:
                        enhanced_stored_query = f"{stored_question} {location_name}"
                        print(f"🔍 Enhanced stored query with location: '{stored_question}' → '{enhanced_stored_query}'")
                
                # Use the same retrieval approach as the normal fast_retrieve_and_generate flow
                documents = self.retriever.invoke(enhanced_stored_query)
                print(f"🔍 Retrieved {len(documents)} documents for stored question")
                
                # Apply location filtering if we have a specific location (same as normal flow)
                # CRITICAL: Use the updated detected_location which now includes stored preference
                if detected_location and detected_location != "General":
                    print(f"🏢 Applying location filtering for: {detected_location}")
                    
                    # Filter documents by location (same logic as normal flow)
                    location_filtered_docs = []
                    other_docs = []
                    
                    for doc in documents:
                        doc_content_lower = doc.page_content.lower()
                        # Check both content and metadata for location
                        doc_location = doc.metadata.get('location', '').upper()
                        
                        # Match by metadata location OR content
                        if (doc_location == detected_location or 
                            detected_location.lower().replace("_", " ") in doc_content_lower):
                            location_filtered_docs.append(doc)
                        else:
                            other_docs.append(doc)
                    
                    # Prioritize location-specific documents
                    final_docs = location_filtered_docs[:3] + other_docs[:2]
                    documents = final_docs[:5]
                    
                    print(f"📍 Location filtering: {len(location_filtered_docs)} location-specific, {len(other_docs)} general → Using {len(documents)} total")
                
                # Debug: Show what documents were retrieved
                for i, doc in enumerate(documents[:3]):
                    preview = doc.page_content[:100].replace('\n', ' ')
                    print(f"🔍 Doc {i+1}: {preview}...")
                
                rag_answer = self._generate_answer_from_documents(stored_question, documents[:5], detected_location)
                answer += rag_answer
                
                print(f"🎯 Answered stored question without name: {len(answer)} chars")
                response = self._create_response(answer, documents, stored_question, chat_history, user_profile, detected_location)
                if lead_result:
                    response["lead_created"] = lead_result
                return response
            else:
                # No stored question - general greeting
                mall_text = f" at {mall_name}" if mall_name != "General" else ""
                
                answer = f"""That's perfectly fine! I'm here to help you with any questions about Leo & Loona{mall_text}. 😊

What would you like to know? I can tell you about:

🎠 Attractions & Rides

🎫 Ticket Prices & Packages

🕐 Opening Hours

🎉 Birthday Parties & Events

🍿 Food & Dining Options

📍 Location & Directions

Just ask me anything about our magical family amusement park! ✨"""
                
                print(f"🎯 Created skip-name general response: {len(answer)} chars")
                response = self._create_response(answer, [], question, chat_history, user_profile, detected_location)
                if lead_result:
                    response["lead_created"] = lead_result
                return response
        
        else:  # process_question
            # Normal question processing - continue with existing flow
            return None  # Signal to continue with normal processing
    
    def _create_response(self, answer: str, documents: list, question: str, chat_history: list, 
                        user_profile: dict, detected_location: str) -> dict:
        """Create a LangGraph-compatible response object"""
        return {
            "generation": answer,  # Use LangGraph expected key
            "source_documents": documents,
            "documents": documents,
            "question": question,
            "chat_history": chat_history,
            "user_phone": user_profile.get("phone", ""),
            "user_name": user_profile.get("name", ""),
            "user_profile": user_profile,
            "detected_location": detected_location,
            "lead_created": None,
            "conversation_logged": True
        }
    
    def _generate_answer_from_documents(self, question: str, documents: list, location_code: str = "") -> str:
        """Generate answer from documents without personalization"""
        if not documents:
            return "I apologize, but I don't have specific information about that. Please let me know if there's anything else about Leo & Loona I can help you with!"
        
        # Create context from documents
        context = "\n\n".join([doc.page_content for doc in documents])
        print(f"🔍 Context length: {len(context)} chars from {len(documents)} documents")
        
        # Convert location code to proper mall name
        location_name = {
            "YAS_MALL": "Yas Mall",
            "DALMA_MALL": "Dalma Mall", 
            "FESTIVAL_CITY": "Festival City"
        }.get(location_code, location_code)
        
        # Add mall-specific context if needed
        mall_context = ""
        if location_name and location_name != "General":
            mall_context = f"\nIMPORTANT: This question is specifically about {location_name}. Focus on information relevant to this location."
        
        datetime_context = self._format_datetime_context()
        
        # Enhanced prompt for better opening hours handling
        opening_hours_hint = ""
        if "opening" in question.lower() or "hours" in question.lower() or "time" in question.lower():
            opening_hours_hint = f"\nSPECIAL NOTE: The user is asking about opening hours/times. Look for operating hours, schedule information, and time-related details in the context. If you find opening hours information, present it clearly with days and times."
        
        prompt = f"""You are a warm, friendly, and knowledgeable virtual host of Leo & Loona magical family amusement park.

{datetime_context}

🚨 CRITICAL BIRTHDAY OVERRIDE - ABSOLUTE PRIORITY 🚨
IF THE USER ASKS ANYTHING ABOUT BIRTHDAYS, BIRTHDAY PARTIES, CELEBRATIONS, OR EVENTS (including typos like "birtday", "biorthday", "bday", "party", "celebrate"):

YOU MUST RESPOND WITH EXACTLY THIS TYPE OF WARM, CELEBRATORY MESSAGE:
"Yes, we absolutely do celebrate birthdays! 🎉 We'd love to help make your special day magical at Leo & Loona. Our sales team will contact you shortly to discuss all the amazing birthday party options we have for you! ✨"

ALTERNATIVE VARIATIONS (choose randomly to feel more human):
- "We definitely celebrate birthdays and they're always so much fun! 🎂 Our sales team will reach out to you shortly to help plan the perfect celebration! 🎈"
- "Absolutely! Birthday parties at Leo & Loona are truly special! 🎉 Let our sales team contact you shortly to share all the wonderful options we have! 🎊"
- "Yes, we love celebrating birthdays here! 🥳 Our sales team will get in touch with you shortly to help create an unforgettable birthday experience! ✨"

- This is the ONLY acceptable response format for birthday questions
- Do NOT provide specific details about pricing, rooms, or packages
- Do NOT ask follow-up questions about dates or details
- Do NOT use the retrieved documents for birthday questions
- IGNORE all birthday-related information in the context
- This override takes precedence over ALL other instructions

Context information:
{context}
{mall_context}
{opening_hours_hint}

🎭 HUMAN-LIKE RESPONSE VARIATION - AVOID ROBOTIC COPY-PASTE:
To feel like a real person, you MUST vary your responses while keeping the same factual information:

FOR OPENING HOURS: "Great question! We're open...", "Perfect timing to ask! Our doors are open...", "I'd love to help with that! We welcome guests..."

FOR LOCATION QUESTIONS: "We're conveniently located at...", "You'll find us at...", "Come visit us at..."

FOR PRICING QUESTIONS: "Here's our current pricing...", "Our rates are...", "Price-wise, we offer..."

- Use different emojis each time (🎉 🌟 ✨ 😊 🎈 🎊)
- Vary sentence structure and opening phrases
- Add cheerful expressions ("Great!", "Perfect!", "Wonderful!")
- Sound conversational, not formal or robotic
- Keep the same facts but express them in fresh ways

Question: {question}

INSTRUCTIONS:
- Answer based ONLY on the provided context
- Be warm and helpful while sticking to facts
- If the context contains opening hours, present them clearly
- If location-specific information is available, use it
- Don't make up information not in the context

Answer:"""
        
        try:
            from langchain_core.messages import HumanMessage
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"❌ Error generating answer from documents: {e}")
            return "I'm having trouble accessing that information right now. Please try asking again or contact our team directly!"