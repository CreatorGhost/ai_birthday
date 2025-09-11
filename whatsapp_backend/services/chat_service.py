"""
Chat service for WhatsApp conversations
Handles message processing, RAG pipeline integration, and response generation
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any

from ..models.user import ChatResponse
from ..managers.session_manager import SessionManager
from ..core.config import config
from .lead_service import WhatsAppLeadService

logger = logging.getLogger(__name__)

class ChatService:
    """Service for handling chat conversations"""
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.rag_pipeline = None
        self.user_tracker = None
        self.lead_service = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize RAG pipeline and user tracker"""
        if self._initialized:
            return True
        
        try:
            logger.info("🚀 Initializing RAG pipeline...")
            
            # Import here to avoid circular imports
            from rag_system import RAGPipeline
            from rag_system.user_tracker import UserTracker
            from bitrix_integration.lead_manager import LeadManager
            
            # Initialize RAG pipeline
            self.rag_pipeline = RAGPipeline()
            if self.rag_pipeline.load_existing_vector_store():
                self.rag_pipeline.setup_graph()
                logger.info("✅ RAG pipeline initialized successfully")
                
                # Initialize user tracker
                self.user_tracker = UserTracker(llm=self.rag_pipeline.llm)
                logger.info("✅ User tracker initialized")
                
                # Initialize lead manager
                lead_manager = LeadManager()
                logger.info("✅ Lead manager initialized")
                
                # Initialize WhatsApp lead service
                self.lead_service = WhatsAppLeadService(
                    lead_manager=lead_manager,
                    user_tracker=self.user_tracker
                )
                logger.info("✅ WhatsApp lead service initialized")
                
                self._initialized = True
                return True
            else:
                logger.error("❌ Failed to load vector store")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize chat service: {e}")
            return False
    
    async def process_message(self, phone: str, message: str) -> ChatResponse:
        """Process incoming message and generate response"""
        start_time = time.time()
        
        try:
            # Ensure initialization
            if not self._initialized:
                await self.initialize()
            
            if not self._initialized:
                raise Exception("Chat service not properly initialized")
            
            # 🚀 WHATSAPP PHASE 1: Immediate Lead Creation
            # Ensure lead exists immediately on first message
            lead_created_now = None
            if self.lead_service:
                lead_created_now = await self.lead_service.ensure_lead_exists(phone, message)
                if lead_created_now:
                    logger.info(f"🎯 Immediate lead created: {lead_created_now}")
            
            # Get chat history BEFORE adding current message (critical for first detection)
            chat_history = self.session_manager.get_chat_history(phone)
            
            # DEBUG: Log chat history to compare with Streamlit
            logger.info(f"🔍 WebSocket chat history for {phone}: {chat_history}")
            
            # Log user message
            if self.user_tracker:
                session = self.session_manager.get_session(phone)
                user_name = session.user_info.name if session else None
                
                self.user_tracker.log_conversation(
                    phone=phone,
                    name=user_name,
                    message=message,
                    is_user=True
                )
            
            # 🧠 RAG PROCESSING: Process with RAG pipeline (unchanged)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.rag_pipeline.answer_question(
                    message,
                    chat_history=chat_history,  # Already excludes current message
                    manual_phone=phone,
                    manual_mall=None  # Same as Streamlit default (no manual override)
                )
            )
            
            # Extract response using correct key (same as Streamlit)
            response = result.get("answer")
            if not response:
                raise Exception("No answer generated from RAG pipeline")
            
            # Add both messages to chat history AFTER processing
            self.session_manager.add_message_to_session(phone, message, is_human=True)
            self.session_manager.add_message_to_session(phone, response, is_human=False)
            
            # 📈 WHATSAPP PHASE 2: Progressive Lead Updates
            # Update lead information as we learn more about the user
            lead_updated = False
            if self.lead_service:
                lead_updated = await self.lead_service.update_lead_progress(
                    phone=phone,
                    extracted_info=result.get("user_info", {}),
                    message=message
                )
                if lead_updated:
                    logger.info(f"📈 Lead progressively updated for {phone}")
            
            # 🎂 WHATSAPP PHASE 3: Continuous Birthday Monitoring (DISABLED)
            # Birthday conversion now handled entirely by RAG pipeline's fast detection system
            # This prevents conflicts between two birthday detection systems
            birthday_detected = False
            # DISABLED: Monitor system causes conflicts with RAG pipeline birthday detection
            # if self.lead_service:
            #     birthday_detected = await self.lead_service.monitor_birthday_keywords(
            #         phone=phone,
            #         message=message,
            #         chat_history=chat_history
            #     )
            #     if birthday_detected:
            #         logger.info(f"🎂 Birthday conversion completed for {phone}")
            
            # Update user info if extracted (legacy support)
            user_info_updated = False
            lead_created = lead_created_now  # Use immediate creation result
            
            if result.get("user_info") and result["user_info"].get("name"):
                self.session_manager.update_user_info(
                    phone=phone,
                    name=result["user_info"]["name"]
                )
                user_info_updated = True
            
            # Handle legacy lead creation from RAG pipeline
            if result.get("lead_created") and not lead_created_now:
                lead_id = result["lead_created"]
                self.session_manager.update_user_info(phone=phone, lead_id=str(lead_id))
                lead_created = str(lead_id)
            
            # Log bot response
            if self.user_tracker:
                session = self.session_manager.get_session(phone)
                user_name = session.user_info.name if session else None
                
                self.user_tracker.log_conversation(
                    phone=phone,
                    name=user_name,
                    message=response,
                    is_user=False
                )
            
            processing_time = (time.time() - start_time) * 1000
            
            return ChatResponse(
                phone=phone,
                message=message,
                response=response,
                user_info_updated=user_info_updated,
                lead_created=lead_created,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ Error processing message for {phone}: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return ChatResponse(
                phone=phone,
                message=message,
                response=f"ERROR: {str(e)}",
                error=str(e),
                processing_time_ms=processing_time
            )
    
    async def send_typing_indicator(self, phone: str) -> bool:
        """Send typing indicator to user"""
        return await self.session_manager.send_message(phone, "", "typing")
    
    async def send_response(self, phone: str, response: str) -> bool:
        """Send response message to user"""
        return await self.session_manager.send_message(phone, response, "message")
    
    def get_status(self) -> Dict[str, Any]:
        """Get chat service status"""
        status = {
            "initialized": self._initialized,
            "rag_pipeline": "ready" if self.rag_pipeline else "not initialized",
            "user_tracker": "ready" if self.user_tracker else "not initialized",
            "lead_service": "ready" if self.lead_service else "not initialized",
            "session_manager": self.session_manager.get_status()
        }
        
        # Add lead service details if available
        if self.lead_service:
            status["whatsapp_lead_flow"] = self.lead_service.get_status()
            
        return status
