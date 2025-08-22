import os
import json
from datetime import datetime
from typing import Dict, List, Optional
from .bitrix_client import BitrixClient
from .config import BitrixConfig

class LeadManager:
    """Manages lead creation in Bitrix with proper categorization for chatbot conversations"""
    
    def __init__(self):
        """Initialize Bitrix lead manager"""
        try:
            self.config = BitrixConfig()
            self.client = BitrixClient(self.config)
        except Exception as e:
            print(f"Warning: Bitrix not configured - {e}")
            self.client = None
    
    def create_chatbot_lead(self, user_info: Dict, conversation_history: List[Dict], category_analysis: Dict) -> Optional[Dict]:
        """Create a lead in Bitrix from chatbot conversation with proper categorization"""
        
        if not self.client:
            print("Bitrix client not available - lead not created")
            return None
        
        try:
            # Determine Bitrix status based on category
            bitrix_status = self._get_bitrix_status(category_analysis)
            
            # Generate lead title and description
            lead_title = self._generate_lead_title(user_info, category_analysis)
            lead_description = self._generate_lead_description(conversation_history, category_analysis)
            
            # Prepare lead data for Bitrix
            lead_data = {
                'TITLE': lead_title,
                'NAME': user_info.get('name', ''),
                'STATUS_ID': bitrix_status,
                'SOURCE_ID': 'WEB',
                'PHONE': [{'VALUE': user_info.get('phone', ''), 'VALUE_TYPE': 'WORK'}] if user_info.get('phone') else [],
                'COMMENTS': lead_description,
                'ASSIGNED_BY_ID': 1,  # Default assignment
                # Custom fields for tracking
                'UF_CHATBOT_SOURCE': 'Leo & Loona AI Assistant',
                'UF_CONVERSATION_CATEGORY': category_analysis.get('category', 'general'),
                'UF_CATEGORY_CONFIDENCE': str(category_analysis.get('confidence', 0.0)),
                'UF_KEYWORDS': ', '.join(category_analysis.get('keywords_found', [])),
                'UF_TOTAL_MESSAGES': str(len(conversation_history))
            }
            
            # Create lead in Bitrix
            response = self.client.create_lead(lead_data)
            
            if response and 'result' in response:
                lead_id = response['result']
                print(f"✅ Lead created in Bitrix: ID {lead_id}, Category: {bitrix_status}")
                
                # Log the lead creation
                self._log_lead_creation(lead_id, user_info, category_analysis)
                
                return {
                    'lead_id': lead_id,
                    'status': bitrix_status,
                    'category': category_analysis.get('category'),
                    'success': True
                }
            else:
                print(f"❌ Failed to create lead in Bitrix: {response}")
                return None
                
        except Exception as e:
            print(f"Error creating lead in Bitrix: {e}")
            return None
    
    def _get_bitrix_status(self, category_analysis: Dict) -> str:
        """Map conversation category to Bitrix lead status"""
        
        category = category_analysis.get('category', 'general')
        confidence = category_analysis.get('confidence', 0.0)
        
        # High confidence birthday party inquiries go to NEW APPROACH
        if category == 'birthday_party' and confidence > 0.7:
            return 'NEW'  # This maps to "NEW APPROACH" in your Bitrix
        else:
            return 'GENERAL_QUESTIONS'  # Default for general inquiries
    
    def _generate_lead_title(self, user_info: Dict, category_analysis: Dict) -> str:
        """Generate appropriate lead title based on category"""
        
        name = user_info.get('name', 'Web Visitor')
        category = category_analysis.get('category', 'general')
        
        if category == 'birthday_party':
            return f"🎂 Birthday Party Inquiry - {name}"
        else:
            return f"💬 General Inquiry - {name}"
    
    def _generate_lead_description(self, conversation_history: List[Dict], category_analysis: Dict) -> str:
        """Generate detailed lead description from conversation"""
        
        # Extract key conversation points
        user_messages = [msg.get('content', '') for msg in conversation_history if msg.get('role') == 'user']
        bot_messages = [msg.get('content', '') for msg in conversation_history if msg.get('role') == 'assistant']
        
        # Build description
        description_parts = []
        
        # Category and confidence
        category = category_analysis.get('category', 'general')
        confidence = category_analysis.get('confidence', 0.0)
        reasoning = category_analysis.get('reasoning', '')
        
        description_parts.append(f"🤖 **Chatbot Conversation Analysis**")
        description_parts.append(f"Category: {category.title().replace('_', ' ')}")
        description_parts.append(f"Confidence: {confidence:.1%}")
        description_parts.append(f"Analysis: {reasoning}")
        description_parts.append("")
        
        # Keywords found
        keywords = category_analysis.get('keywords_found', [])
        if keywords:
            description_parts.append(f"🔍 **Keywords Identified**: {', '.join(keywords)}")
            description_parts.append("")
        
        # Conversation summary
        description_parts.append(f"💬 **Conversation Summary**")
        description_parts.append(f"Total Messages: {len(conversation_history)}")
        description_parts.append(f"User Questions: {len(user_messages)}")
        description_parts.append("")
        
        # First user message (original inquiry)
        if user_messages:
            description_parts.append(f"**Original Inquiry:**")
            description_parts.append(f'"{user_messages[0]}"')
            description_parts.append("")
        
        # All user messages for context
        if len(user_messages) > 1:
            description_parts.append(f"**All User Messages:**")
            for i, msg in enumerate(user_messages, 1):
                description_parts.append(f"{i}. {msg}")
            description_parts.append("")
        
        # Timestamp
        description_parts.append(f"📅 **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        description_parts.append(f"🔗 **Source**: Leo & Loona AI Assistant")
        
        return "\n".join(description_parts)
    
    def _log_lead_creation(self, lead_id: int, user_info: Dict, category_analysis: Dict):
        """Log lead creation for tracking"""
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'lead_id': lead_id,
            'user_name': user_info.get('name'),
            'user_phone': user_info.get('phone'),
            'category': category_analysis.get('category'),
            'confidence': category_analysis.get('confidence'),
            'keywords': category_analysis.get('keywords_found', [])
        }
        
        try:
            # Ensure user_data directory exists
            os.makedirs('user_data', exist_ok=True)
            
            # Append to lead creation log
            with open('user_data/bitrix_leads_created.log', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Could not log lead creation - {e}")
    
    def should_create_lead(self, user_info: Dict, conversation_history: List[Dict]) -> bool:
        """Determine if a lead should be created based on conversation"""
        
        # Create lead if:
        # 1. We have user's name
        # 2. User has engaged in meaningful conversation (2+ messages)
        # 3. Not already created for this user recently
        
        has_name = bool(user_info.get('name'))
        meaningful_conversation = len(conversation_history) >= 2
        
        return has_name and meaningful_conversation
    
    def test_connection(self) -> bool:
        """Test Bitrix connection"""
        if not self.client:
            return False
        return self.client.test_connection()
