#!/usr/bin/env python3
"""
Simple WhatsApp Test Server
FastAPI server for testing WhatsApp integration with separate test number
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
from whatsapp_test_integration import WhatsAppTestIntegration

# Pydantic model for webhook data
class WhatsAppMessage(BaseModel):
    phone: Optional[str] = None
    name: Optional[str] = None
    message: Optional[str] = None
    line_id: Optional[str] = None
    chat_id: Optional[str] = None
    session_id: Optional[str] = None
    # Add other fields as needed based on actual Bitrix24 webhook structure

# Initialize FastAPI app
app = FastAPI(
    title="WhatsApp Test Integration",
    description="Simple WhatsApp testing with separate test number",
    version="1.0.0"
)

# Initialize integration in test mode
integration = WhatsAppTestIntegration(test_mode=True)

@app.post("/webhook/whatsapp-test")
async def handle_whatsapp_webhook(payload: Dict[str, Any]):
    """
    Handle incoming WhatsApp webhook from Bitrix24 test line
    
    This endpoint receives WhatsApp messages from your test number,
    processes them with your chatbot, and generates responses.
    """
    try:
        print(f"📨 Received WhatsApp message: {payload}")
        
        # Process the message
        result = integration.process_whatsapp_message(payload)
        
        print(f"✅ Processing result: {result}")
        return result
        
    except Exception as e:
        print(f"❌ Error processing WhatsApp message: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "WhatsApp Test Integration",
        "mode": "test" if integration.test_mode else "production",
        "target_line": f"{integration.target_line_id} ({integration.line_name})",
        "status": "ready",
        "endpoints": {
            "webhook": "/webhook/whatsapp-test",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "WhatsApp Test Integration",
        "mode": "test" if integration.test_mode else "production",
        "line": f"{integration.target_line_id} ({integration.line_name})"
    }

@app.post("/test-message")
async def test_message(message: WhatsAppMessage):
    """
    Test endpoint - send a message to test your chatbot without WhatsApp
    
    Use this to test your chatbot responses before connecting to WhatsApp
    """
    try:
        # Convert to dict format
        test_payload = {
            "phone": message.phone or "917704090366",  # Your personal number
            "name": message.name or "Test User",
            "message": message.message or "Hello",
            "line_id": integration.target_line_id,  # Use test line
            "chat_id": "test_chat",
            "session_id": "test_session"
        }
        
        print(f"🧪 Testing message: {test_payload}")
        
        # Process test message
        result = integration.process_whatsapp_message(test_payload)
        
        return {
            "test_input": test_payload,
            "chatbot_result": result,
            "message": "Test completed successfully"
        }
        
    except Exception as e:
        print(f"❌ Test error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

@app.post("/switch-to-production")
async def switch_to_production():
    """
    Switch to production mode (DANGER: Will send real messages!)
    Only use this when testing is complete and you're ready to go live
    """
    global integration
    
    if not integration.test_mode:
        return {"message": "Already in production mode"}
    
    print("🚨 SWITCHING TO PRODUCTION MODE")
    integration = WhatsAppTestIntegration(test_mode=False)
    
    return {
        "message": "SWITCHED TO PRODUCTION MODE",
        "warning": "Will now send REAL messages to WhatsApp!",
        "mode": "production",
        "line": f"{integration.target_line_id} ({integration.line_name})"
    }

@app.post("/switch-to-test")
async def switch_to_test():
    """Switch back to test mode (safe)"""
    global integration
    
    if integration.test_mode:
        return {"message": "Already in test mode"}
    
    print("🧪 Switching back to test mode")
    integration = WhatsAppTestIntegration(test_mode=True)
    
    return {
        "message": "Switched to test mode",
        "mode": "test",
        "line": f"{integration.target_line_id} ({integration.line_name})"
    }

if __name__ == "__main__":
    print("🚀 Starting WhatsApp Test Integration Server")
    print("=" * 50)
    print(f"🧪 Mode: {'TEST' if integration.test_mode else 'PRODUCTION'}")
    print(f"🎯 Target Line: {integration.target_line_id} ({integration.line_name})")
    print(f"🔗 Webhook URL: http://localhost:8000/webhook/whatsapp-test")
    print(f"📚 API Docs: http://localhost:8000/docs")
    print(f"🧪 Test Endpoint: http://localhost:8000/test-message")
    print("=" * 50)
    
    if integration.test_mode:
        print("✅ SAFE TEST MODE - No real messages will be sent")
    else:
        print("🚨 PRODUCTION MODE - REAL messages will be sent!")
    
    print("=" * 50)
    
    uvicorn.run(
        "whatsapp_test_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
