"""
Session management for WhatsApp users
Handles user sessions, chat history, and connection state
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from fastapi import WebSocket

from ..models.user import UserSession, UserInfo, ChatMessage
from ..core.config import config

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages user sessions and WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, UserSession] = {}
    
    async def connect_websocket(self, websocket: WebSocket, phone: str) -> bool:
        """Connect WebSocket for a user"""
        try:
            await websocket.accept()
            self.active_connections[phone] = websocket
            
            # Create or update session
            if phone not in self.user_sessions:
                self.create_session(phone)
            else:
                # Reactivate existing session
                self.user_sessions[phone].is_active = True
            
            logger.info(f"✅ WebSocket connected: {phone}")
            return True
            
        except Exception as e:
            logger.error(f"❌ WebSocket connection failed for {phone}: {e}")
            return False
    
    def disconnect_websocket(self, phone: str) -> None:
        """Disconnect WebSocket for a user"""
        if phone in self.active_connections:
            del self.active_connections[phone]
        
        if phone in self.user_sessions:
            session = self.user_sessions[phone]
            session.is_active = False
            logger.info(f"❌ WebSocket disconnected: {phone} (sent {session.message_count} messages)")
    
    def create_session(self, phone: str) -> UserSession:
        """Create new user session"""
        user_info = UserInfo(phone=phone)
        session = UserSession(
            phone=phone,
            connected_at=datetime.now(),
            user_info=user_info
        )
        
        self.user_sessions[phone] = session
        logger.info(f"📱 Created session for: {phone}")
        return session
    
    def get_session(self, phone: str) -> Optional[UserSession]:
        """Get user session"""
        return self.user_sessions.get(phone)
    
    def add_message_to_session(self, phone: str, content: str, is_human: bool = True) -> bool:
        """Add message to user's chat history"""
        session = self.get_session(phone)
        if not session:
            logger.warning(f"⚠️ No session found for {phone}")
            return False
        
        session.add_message(content, is_human)
        return True
    
    def get_chat_history(self, phone: str) -> List[Dict]:
        """Get chat history for RAG pipeline"""
        session = self.get_session(phone)
        if not session:
            return []
        
        return session.get_chat_history_dict()
    
    def update_user_info(self, phone: str, name: Optional[str] = None, 
                         lead_id: Optional[str] = None,
                         location: Optional[str] = None) -> bool:
        """Update user information"""
        session = self.get_session(phone)
        if not session:
            return False
        
        session.update_user_info(name, lead_id, location)
        return True
    
    async def send_message(self, phone: str, message: str, msg_type: str = "message") -> bool:
        """Send message via WebSocket"""
        if phone not in self.active_connections:
            logger.warning(f"⚠️ No active connection for {phone}")
            return False
        
        try:
            await self.active_connections[phone].send_json({
                "type": msg_type,
                "content": message,
                "timestamp": datetime.now().isoformat()
            })
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send message to {phone}: {e}")
            self.disconnect_websocket(phone)
            return False
    
    async def broadcast_message(self, message: str) -> int:
        """Broadcast message to all connected users"""
        sent_count = 0
        for phone in list(self.active_connections.keys()):
            if await self.send_message(phone, message, "broadcast"):
                sent_count += 1
        
        return sent_count
    
    def cleanup_inactive_sessions(self) -> int:
        """Remove old inactive sessions"""
        cutoff_time = datetime.now() - timedelta(minutes=config.SESSION_TIMEOUT_MINUTES)
        removed_count = 0
        
        phones_to_remove = []
        for phone, session in self.user_sessions.items():
            if (not session.is_active and 
                session.user_info.last_seen and 
                session.user_info.last_seen < cutoff_time):
                phones_to_remove.append(phone)
        
        for phone in phones_to_remove:
            del self.user_sessions[phone]
            removed_count += 1
            logger.info(f"🧹 Cleaned up inactive session: {phone}")
        
        return removed_count
    
    def get_status(self) -> Dict:
        """Get session manager status"""
        active_sessions = [s for s in self.user_sessions.values() if s.is_active]
        
        return {
            "active_connections": len(self.active_connections),
            "total_sessions": len(self.user_sessions),
            "active_sessions": len(active_sessions),
            "users": {
                phone: {
                    "name": session.user_info.name,
                    "message_count": session.message_count,
                    "chat_history_length": len(session.chat_history),
                    "lead_id": session.user_info.bitrix_lead_id,
                    "connected_at": session.connected_at.isoformat(),
                    "last_seen": session.user_info.last_seen.isoformat() if session.user_info.last_seen else None,
                    "is_active": session.is_active
                }
                for phone, session in self.user_sessions.items()
            }
        }
