#!/usr/bin/env python3
"""
Simple WhatsApp Test Integration
Connect your existing chatbot to a test WhatsApp number for testing
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from bitrix_integration.bitrix_client import BitrixClient

class WhatsAppTestIntegration:
    """Simple WhatsApp integration for testing with separate number"""
    
    def __init__(self, test_mode=True):
        """Initialize the test integration"""
        self.client = BitrixClient()
        self.test_mode = test_mode
        
        # Test configuration - separate from production
        if test_mode:
            self.target_line_id = "99"  # Test line ID (configure in Bitrix24)
            self.line_name = "LeoLoona-Test"
            print(f"🧪 TEST MODE: Using test line {self.target_line_id}")
        else:
            self.target_line_id = "1"   # Production line
            self.line_name = "LeoLoona"
            print(f"🚨 PRODUCTION MODE: Using live line {self.target_line_id}")
    
    def process_whatsapp_message(self, webhook_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process incoming WhatsApp message from Bitrix24
        
        Args:
            webhook_data: Message data from Bitrix24 webhook
            
        Returns:
            Response data with chatbot reply
        """
        try:
            # Extract message information
            message_info = self.extract_message_info(webhook_data)
            
            if not message_info:
                return {"error": "Could not process message"}
            
            # Generate chatbot response using your existing logic
            response = self.generate_chatbot_response(message_info)
            
            # Log the interaction
            self.log_interaction(message_info, response)
            
            if self.test_mode:
                # Test mode: Only log what would happen
                print(f"🧪 TEST MODE - Response generated:")
                print(f"   📱 Customer: {message_info.get('customer_name')} ({message_info.get('customer_phone')})")
                print(f"   💬 Message: {message_info.get('message_text')}")
                print(f"   🤖 Response: {response}")
                print(f"   ✅ In production, this would be sent to WhatsApp")
                
                return {
                    "status": "test_success",
                    "customer": message_info.get('customer_name'),
                    "message": message_info.get('message_text'),
                    "response": response,
                    "note": "Test mode - response not sent"
                }
            else:
                # Production mode: Actually send response
                sent = self.send_to_whatsapp(message_info, response)
                
                return {
                    "status": "sent" if sent else "failed",
                    "response": response
                }
                
        except Exception as e:
            print(f"❌ Error processing message: {str(e)}")
            return {"error": str(e)}
    
    def extract_message_info(self, webhook_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract message information from webhook data"""
        try:
            # Adjust these field names based on actual Bitrix24 webhook structure
            message_info = {
                'customer_phone': webhook_data.get('phone', webhook_data.get('from', '')),
                'customer_name': webhook_data.get('name', webhook_data.get('contact_name', 'WhatsApp User')),
                'message_text': webhook_data.get('message', webhook_data.get('text', '')),
                'line_id': webhook_data.get('line_id', webhook_data.get('line', '')),
                'chat_id': webhook_data.get('chat_id', ''),
                'session_id': webhook_data.get('session_id', ''),
                'timestamp': datetime.now().isoformat()
            }
            
            # Only process messages from our test line (or production line)
            if message_info['line_id'] != self.target_line_id:
                print(f"🚫 Ignoring message from line {message_info['line_id']} (not {self.target_line_id})")
                return None
            
            print(f"📨 Processing message from {message_info['customer_name']}: {message_info['message_text']}")
            return message_info
            
        except Exception as e:
            print(f"❌ Error extracting message info: {str(e)}")
            return None
    
    def generate_chatbot_response(self, message_info: Dict[str, Any]) -> str:
        """
        Generate chatbot response - integrate with your existing Streamlit chatbot logic here
        
        Replace this with your actual chatbot logic from Streamlit app
        """
        message_text = message_info.get('message_text', '').lower()
        customer_name = message_info.get('customer_name', 'there')
        
        # TODO: Replace this simple logic with your actual chatbot from Streamlit
        if any(keyword in message_text for keyword in ['birthday', 'party', 'celebration', 'bday']):
            return f"Hi {customer_name}! 🎂 I'd love to help you plan a birthday party! Which Leo & Loona location interests you - Festival City, Dalma Mall, or Yas Mall?"
        
        elif any(keyword in message_text for keyword in ['hours', 'open', 'time', 'when']):
            return f"Hi {customer_name}! Our Leo & Loona parks are open daily. Which location would you like hours for - Festival City, Dalma Mall, or Yas Mall?"
        
        elif any(keyword in message_text for keyword in ['price', 'cost', 'ticket', 'how much']):
            return f"Hi {customer_name}! Our pricing varies by location and activities. Which Leo & Loona park are you interested in visiting?"
        
        elif any(keyword in message_text for keyword in ['hello', 'hi', 'hey', 'start']):
            return f"Hi {customer_name}! Welcome to Leo & Loona! 🦁🐰 How can I help you today? Are you looking for information about our parks, birthday parties, or something else?"
        
        else:
            return f"Hi {customer_name}! Thanks for your message. I'm here to help with information about Leo & Loona parks, birthday parties, pricing, and hours. What would you like to know?"
    
    def send_to_whatsapp(self, message_info: Dict[str, Any], response: str) -> bool:
        """Send response back to WhatsApp (production mode only)"""
        if self.test_mode:
            return True  # Don't actually send in test mode
        
        try:
            # Try different Bitrix24 API methods to send WhatsApp message
            chat_id = message_info.get('chat_id')
            session_id = message_info.get('session_id')
            
            # Method 1: imconnector.send
            try:
                params = {
                    'CONNECTOR': 'olchat_wa_connector_2',
                    'LINE': self.target_line_id,
                    'CHAT_ID': chat_id,
                    'MESSAGE': response
                }
                result = self.client._make_request('imconnector.send', params)
                if 'result' in result:
                    print(f"✅ Message sent via imconnector.send")
                    return True
            except Exception as e:
                print(f"⚠️ imconnector.send failed: {e}")
            
            # Method 2: imopenlines.message.send
            try:
                if session_id:
                    params = {
                        'SESSION_ID': session_id,
                        'MESSAGE': response
                    }
                    result = self.client._make_request('imopenlines.message.send', params)
                    if 'result' in result:
                        print(f"✅ Message sent via imopenlines.message.send")
                        return True
            except Exception as e:
                print(f"⚠️ imopenlines.message.send failed: {e}")
            
            print(f"⚠️ Could not send message via API, check Bitrix24 configuration")
            return False
            
        except Exception as e:
            print(f"❌ Error sending to WhatsApp: {str(e)}")
            return False
    
    def log_interaction(self, message_info: Dict[str, Any], response: str):
        """Log the interaction for monitoring"""
        try:
            log_entry = {
                'timestamp': message_info.get('timestamp'),
                'customer_phone': message_info.get('customer_phone'),
                'customer_name': message_info.get('customer_name'),
                'message': message_info.get('message_text'),
                'response': response,
                'line_id': message_info.get('line_id'),
                'test_mode': self.test_mode
            }
            
            # Save to log file
            os.makedirs('logs', exist_ok=True)
            log_file = f"logs/whatsapp_test_{datetime.now().strftime('%Y%m%d')}.log"
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            print(f"⚠️ Could not log interaction: {e}")

# Integration with your existing chatbot logic
def integrate_with_your_chatbot(message_text: str, customer_name: str) -> str:
    """
    TODO: Replace this function with your actual chatbot logic from Streamlit
    
    This is where you would call your existing chatbot functions that you
    developed and tested in Streamlit.
    
    Args:
        message_text: The customer's WhatsApp message
        customer_name: The customer's name
        
    Returns:
        The chatbot's response
    """
    
    # Example integration - replace with your actual chatbot logic:
    
    # from your_chatbot_module import generate_response
    # return generate_response(message_text, customer_name)
    
    # For now, using simple responses
    return f"Hi {customer_name}! Thanks for your message: '{message_text}'. This is where your actual chatbot logic would go."

if __name__ == "__main__":
    print("🧪 WhatsApp Test Integration")
    print("=" * 40)
    
    # Test the integration
    integration = WhatsAppTestIntegration(test_mode=True)
    
    # Simulate a test message with your actual numbers
    test_webhook = {
        'phone': '917704090366',  # Your personal number (without +)
        'name': 'Test Customer',
        'message': 'Do you do birthday parties?',
        'line_id': '99',  # Test line
        'chat_id': 'test_chat_123',
        'session_id': 'test_session_456'
    }
    
    print("📨 Testing with sample message...")
    result = integration.process_whatsapp_message(test_webhook)
    print(f"✅ Test result: {result}")
    
    print(f"\n🎯 Next steps:")
    print(f"1. Get a test WhatsApp number")
    print(f"2. Set up Line 99 in Bitrix24 with test number")  
    print(f"3. Configure webhook URL to point to your server")
    print(f"4. Start server: python whatsapp_test_server.py")
    print(f"5. Send WhatsApp messages to test number")
