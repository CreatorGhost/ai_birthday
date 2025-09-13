"""
FastAPI routes for WhatsApp backend
Handles WebSocket connections, status endpoints, and webhook processing
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse

from ..services.chat_service import ChatService
from ..managers.session_manager import SessionManager
from ..models.user import WhatsAppMessage
from ..core.config import config

logger = logging.getLogger(__name__)

# Initialize managers and services
session_manager = SessionManager()
chat_service = ChatService(session_manager)

# Create API router
router = APIRouter()

@router.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await chat_service.initialize()

@router.get("/")
async def root():
    """Root endpoint - serves WhatsApp-like chat simulator"""
    try:
        with open("whatsapp_chat_simulator.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        try:
            # Fallback to old test interface
            with open("whatsapp_simple_test.html", "r") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content)
        except FileNotFoundError:
            return {
                "service": "WhatsApp Backend System",
                "status": "running",
                "mode": "TEST" if config.TEST_MODE else "PRODUCTION",
                "endpoints": {
                    "status": "/status",
                    "webhook": "/webhook/whatsapp",
                    "websocket": "/ws/{phone}",
                    "chat": "/",
                    "docs": "/docs"
                }
            }

@router.get("/test")
async def test_interface():
    """Simple test interface"""
    try:
        with open("whatsapp_simple_test.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return {"error": "Test interface not found"}

@router.get("/status")
async def get_status():
    """Get system status"""
    config_status = config.get_status()
    chat_status = chat_service.get_status()
    
    return {
        "system": "WhatsApp Backend",
        "status": "running",
        "config": config_status,
        "chat_service": chat_status,
        "active_connections": len(session_manager.active_connections),
        "total_sessions": len(session_manager.user_sessions)
    }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    config_valid, missing_keys = config.validate_required_keys()
    
    return {
        "status": "healthy" if config_valid else "unhealthy",
        "config_valid": config_valid,
        "missing_keys": missing_keys,
        "chat_service_ready": chat_service._initialized
    }

@router.post("/webhook/whatsapp")
async def handle_whatsapp_webhook(payload: Dict[str, Any]):
    """
    Handle incoming WhatsApp webhook from Bitrix24
    Production endpoint for real WhatsApp integration
    """
    try:
        logger.info(f"📨 WhatsApp webhook received: {payload}")
        
        # Extract message info
        message_info = WhatsAppMessage(**payload)
        
        if not message_info.phone or not message_info.message:
            raise HTTPException(status_code=400, detail="Missing phone or message")
        
        # Process message
        response = await chat_service.process_message(
            phone=message_info.phone,
            message=message_info.message
        )
        
        if response.error:
            logger.error(f"❌ Processing error: {response.error}")
            return {"status": "error", "error": response.error}
        
        # In production, this would send the response back to WhatsApp via Bitrix
        # For now, just log it
        logger.info(f"🤖 Generated response for {message_info.phone}: {response.response[:100]}...")
        
        return {
            "status": "success",
            "phone": response.phone,
            "response": response.response,
            "processing_time_ms": response.processing_time_ms,
            "user_info_updated": response.user_info_updated,
            "lead_created": response.lead_created
        }
        
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/{phone}")
async def websocket_endpoint(websocket: WebSocket, phone: str):
    """WebSocket endpoint for testing interface"""
    connected = await session_manager.connect_websocket(websocket, phone)
    
    if not connected:
        return
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data.get("type") == "message":
                message = data.get("content", "")
                logger.info(f"📨 WebSocket message from {phone}: {message}")
                
                # Send typing indicator
                await chat_service.send_typing_indicator(phone)
                
                # Process message
                response = await chat_service.process_message(phone, message)
                
                # Send response
                await chat_service.send_response(phone, response.response)
                
                logger.info(f"🤖 Response to {phone}: {response.response[:50]}...")
                
    except WebSocketDisconnect:
        session_manager.disconnect_websocket(phone)
    except Exception as e:
        logger.error(f"❌ WebSocket error for {phone}: {e}")
        session_manager.disconnect_websocket(phone)

@router.get("/sessions")
async def get_sessions():
    """Get all user sessions for monitoring"""
    return session_manager.get_status()

@router.post("/sessions/{phone}/clear")
async def clear_session(phone: str):
    """Clear a specific user's session"""
    session = session_manager.get_session(phone)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Clear chat history but keep user info
    session.chat_history = []
    session.message_count = 0
    
    return {"status": "cleared", "phone": phone}

@router.post("/sessions/cleanup")
async def cleanup_sessions():
    """Clean up inactive sessions"""
    removed_count = session_manager.cleanup_inactive_sessions()
    return {"status": "cleaned", "removed_sessions": removed_count}

@router.get("/logs")
async def get_logs(lines: int = 100):
    """Get recent logs for debugging"""
    try:
        log_file = f"{config.LOGS_DIR}/websocket_test.log"
        with open(log_file, "r") as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "logs": recent_lines,
            "total_lines": len(all_lines),
            "showing_lines": len(recent_lines),
            "sessions": session_manager.get_status()
        }
    except FileNotFoundError:
        return {"error": "Log file not found", "logs": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
