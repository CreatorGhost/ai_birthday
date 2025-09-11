"""
User and session data models
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel

class ChatMessage(BaseModel):
    """Individual chat message"""
    type: str  # "human" or "ai"
    content: str
    timestamp: datetime = datetime.now()

class UserInfo(BaseModel):
    """User information tracking"""
    phone: str
    name: Optional[str] = None
    total_messages: int = 0
    last_seen: Optional[datetime] = None
    bitrix_lead_id: Optional[str] = None
    preferred_location: Optional[str] = None  # FESTIVAL_CITY, YAS_MALL, etc.

class UserSession(BaseModel):
    """Complete user session data"""
    phone: str
    connected_at: datetime
    message_count: int = 0
    chat_history: List[ChatMessage] = []
    user_info: UserInfo
    is_active: bool = True
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_message(self, content: str, is_human: bool = True) -> None:
        """Add message to chat history"""
        message_type = "human" if is_human else "ai"
        message = ChatMessage(type=message_type, content=content)
        self.chat_history.append(message)
        
        # Keep only recent messages to prevent memory bloat
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
        
        # Update counters
        if is_human:
            self.message_count += 1
            self.user_info.total_messages += 1
            self.user_info.last_seen = datetime.now()
    
    def get_chat_history_dict(self) -> List[Dict]:
        """Get chat history in RAG pipeline format (Streamlit compatible)"""
        return [
            {
                "role": "user" if msg.type == "human" else "assistant",
                "content": msg.content
            }
            for msg in self.chat_history
        ]
    
    def update_user_info(self, name: Optional[str] = None, 
                        lead_id: Optional[str] = None,
                        location: Optional[str] = None) -> None:
        """Update user information"""
        if name:
            self.user_info.name = name
        if lead_id:
            self.user_info.bitrix_lead_id = lead_id
        if location:
            self.user_info.preferred_location = location

class WhatsAppMessage(BaseModel):
    """Incoming WhatsApp message structure"""
    phone: Optional[str] = None
    name: Optional[str] = None
    message: Optional[str] = None
    line_id: Optional[str] = None
    chat_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = datetime.now()

class ChatResponse(BaseModel):
    """Response from chat processing"""
    phone: str
    message: str
    response: str
    user_info_updated: bool = False
    lead_created: Optional[str] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None
