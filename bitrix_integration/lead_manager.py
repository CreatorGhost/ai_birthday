"""Bitrix lead management module."""
import os
import json
import random
from datetime import datetime
from typing import Dict, List, Optional
from .bitrix_client import BitrixClient
from .config import BitrixConfig

class LeadManager:
    """Manages lead creation in Bitrix with proper categorization for chatbot conversations"""

    # Mall-based sales team assignment configuration
    # Based on the sales team structure provided
    MALL_ASSIGNMENT_CONFIG = {
        'Festival City': {
            'team_name': 'DFCM LL (Dubai Festival City Mall - Leo & Loona)',
            'user_ids': [16711, 35, 33],  # Zariaah, Kizeleen, Jamaima
            'user_names': ['Zariaah Nankinga', 'Kizeleen Fernandez', 'Jamaima Ampoloquio']
        },
        'Dalma Mall': {
            'team_name': 'DALMA LL (Dalma Mall - Leo & Loona)', 
            'user_ids': [20547],  # Mack
            'user_names': ['Mack Ofiana']
        },
        'Yas Mall': {
            'team_name': 'YAS MALL LL (Yas Mall - Leo & Loona)',
            'user_ids': [19663, 29759],  # Erra, Abdalaziz
            'user_names': ['Erra Montenegro', 'Abdalaziz Said']
        },
        'General': {
            'team_name': 'General Inquiries',
            'user_ids': [1],  # Default admin user
            'user_names': ['Admin']
        }
    }

    def __init__(self):
        """Initialize Bitrix lead manager"""
        try:
            self.config = BitrixConfig()
            self.client = BitrixClient(self.config)
        except ValueError as e:
            print(f"Warning: Bitrix not configured - {e}")
            self.client = None

    def get_assigned_user_for_mall(self, park_location: str) -> Dict[str, any]:
        """
        Get the assigned user ID and name for a specific mall location.
        If multiple users are available for a mall, randomly selects one.
        
        Args:
            park_location: Mall location (e.g., 'Festival City', 'Dalma Mall', 'Yas Mall')
            
        Returns:
            Dictionary with user_id, user_name, and team_name
        """
        # Normalize park location names to match configuration keys
        location_mapping = {
            'FESTIVAL_CITY': 'Festival City',
            'Festival City': 'Festival City', 
            'festival city': 'Festival City',
            'DALMA_MALL': 'Dalma Mall',
            'Dalma Mall': 'Dalma Mall',
            'dalma mall': 'Dalma Mall',
            'dalma': 'Dalma Mall',
            'YAS_MALL': 'Yas Mall',
            'Yas Mall': 'Yas Mall', 
            'yas mall': 'Yas Mall',
            'yas': 'Yas Mall',
            'General': 'General',
            'general': 'General'
        }
        
        # Get normalized location
        normalized_location = location_mapping.get(park_location, 'General')
        
        # Get assignment configuration for this location
        assignment_config = self.MALL_ASSIGNMENT_CONFIG.get(normalized_location, self.MALL_ASSIGNMENT_CONFIG['General'])
        
        # If multiple users available, randomly select one
        user_ids = assignment_config['user_ids']
        user_names = assignment_config['user_names']
        team_name = assignment_config['team_name']
        
        if len(user_ids) > 1:
            # Random selection for load balancing
            selected_index = random.randint(0, len(user_ids) - 1)
            selected_user_id = user_ids[selected_index]
            selected_user_name = user_names[selected_index]
            
            print(f"🎲 Randomly assigned {park_location} lead to: {selected_user_name} (ID: {selected_user_id})")
        else:
            # Single user assignment
            selected_user_id = user_ids[0]
            selected_user_name = user_names[0]
            
            print(f"👤 Assigned {park_location} lead to: {selected_user_name} (ID: {selected_user_id})")
        
        return {
            'user_id': selected_user_id,
            'user_name': selected_user_name, 
            'team_name': team_name,
            'location': normalized_location
        }

    def create_simple_lead(
        self, name: str, phone: str, park_location: str = "General"
    ) -> Optional[Dict]:
        """Create a simple lead in Bitrix with just name, phone, and park location"""

        if not self.client:
            print("Bitrix client not available - lead not created")
            return None

        try:
            # Get assigned user for this mall location
            assignment_info = self.get_assigned_user_for_mall(park_location)
            assigned_user_id = assignment_info['user_id']
            assigned_user_name = assignment_info['user_name']
            team_name = assignment_info['team_name']
            
            # Working lead data with verified field values and dynamic assignment
            lead_data = {
                'TITLE': f"{park_location} - {name}",
                'NAME': name,
                'LAST_NAME': 'Customer',
                'STATUS_ID': 'UC_0MD91B',  # Correct GENERAL QUESTIONS status ID
                'SOURCE_ID': '1|OLCHAT_WA_CONNECTOR_2',  # Match working leads
                'PHONE': [{'VALUE': phone, 'VALUE_TYPE': 'WORK'}] if phone else [],
                'COMMENTS': f"AI Chatbot Lead - {park_location} inquiry for {name}\nAssigned to: {assigned_user_name} ({team_name})",
                'ASSIGNED_BY_ID': assigned_user_id,  # Dynamic assignment based on mall
                'OPENED': 'Y',
                'IS_MANUAL_OPPORTUNITY': 'Y',
                'OPPORTUNITY': 1.00
            }

            # Add working park field values
            park_fields = self._get_park_field_values(park_location)
            if park_fields:
                lead_data.update(park_fields)
                # Add dynamic date field
                current_date = datetime.now().strftime('%Y-%m-%dT03:00:00+03:00')
                lead_data['UF_CRM_1711693248818'] = [current_date]

            # Debug: Print the lead data being sent
            print(f"🔧 Creating SIMPLE lead with data: {lead_data}")

            # Create lead in Bitrix
            response = self.client.create_lead(lead_data)

            if response and 'result' in response:
                lead_id = response['result']
                print(
                    f"✅ Lead created in Bitrix: ID {lead_id}"
                    f"\n   Customer: {name} ({phone})"
                    f"\n   Location: {park_location}"
                    f"\n   Assigned to: {assigned_user_name} ({team_name})"
                )

                return {
                    'lead_id': lead_id,
                    'name': name,
                    'phone': phone,
                    'park_location': park_location,
                    'assigned_user_id': assigned_user_id,
                    'assigned_user_name': assigned_user_name,
                    'team_name': team_name,
                    'success': True
                }

            print(f"❌ Failed to create simple lead in Bitrix: {response}")
            return None

        except ValueError as e:
            print(f"Error creating simple lead in Bitrix: {e}")
            return None

    def create_chatbot_lead(
        self,
        user_info: Dict,
        park_location: str = "General"
    ) -> Optional[Dict]:
        """Create a lead in Bitrix from chatbot conversation - now uses simple lead creation"""

        name = user_info.get('name', 'Web Visitor')
        phone = user_info.get('phone', '')

        # Use the simple lead creation method
        return self.create_simple_lead(name, phone, park_location)
    def _get_bitrix_status(self, category_analysis: Dict) -> str:
        """Map conversation category to Bitrix lead status"""

        category = category_analysis.get('category', 'general')
        confidence = category_analysis.get('confidence', 0.0)

        # High confidence birthday party inquiries go to NEW APPROACH
        if category == 'birthday_party' and confidence > 0.8:  # Increased threshold
            return 'NEW'  # This maps to "NEW APPROACH" in your Bitrix

        # ALL other inquiries go to GENERAL QUESTIONS
        return 'GENERAL_QUESTIONS'
    def _get_park_field_values(self, park_location: str) -> dict:
        """Get the correct field values for park location - with location normalization"""

        # Normalize park location names to match configuration keys (same logic as get_assigned_user_for_mall)
        location_mapping = {
            'FESTIVAL_CITY': 'Festival City',
            'Festival City': 'Festival City', 
            'festival city': 'Festival City',
            'DALMA_MALL': 'Dalma Mall',
            'Dalma Mall': 'Dalma Mall',
            'dalma mall': 'Dalma Mall',
            'dalma': 'Dalma Mall',
            'YAS_MALL': 'Yas Mall',
            'Yas Mall': 'Yas Mall', 
            'yas mall': 'Yas Mall',
            'yas': 'Yas Mall',
            'General': 'General',
            'general': 'General'
        }
        
        # Get normalized location
        normalized_location = location_mapping.get(park_location, 'General')

        # VERIFIED working park field mappings from analysis of working leads
        park_field_mappings = {
            'Dalma Mall': {
                'UF_CRM_1686684042431': '1',
                'UF_CRM_1704787907109': '745',  # Key field - unique for Dalma
                'UF_CRM_1705999481152': '303',
                'UF_CRM_1716288129220': 'Not mention',
                'UF_CRM_1716288147600': 'not mention',
                'UF_CRM_65D475BE833D2': ['not mention'],
            },
            'Yas Mall': {
                'UF_CRM_1686684042431': '-',
                'UF_CRM_1704787907109': '733',  # Key field - unique for Yas
                'UF_CRM_1705999481152': '303',
                'UF_CRM_1716288129220': '-',
                'UF_CRM_1716288147600': '-',
                'UF_CRM_65D475BE833D2': ['-'],
            },
            'Festival City': {
                'UF_CRM_1686684042431': '1',
                'UF_CRM_1704787907109': '207',  # Key field - unique for Festival City
                'UF_CRM_1705999481152': '303',
                'UF_CRM_1716288129220': 'Not mention',
                'UF_CRM_1716288147600': 'not mention',
                'UF_CRM_65D475BE833D2': ['not mention'],
            },
            'General': {}  # No park fields for general inquiries
        }

        park_fields = park_field_mappings.get(normalized_location, {})
        
        # Debug logging to verify field mapping
        if park_fields:
            print(f"🏢 Park field mapping for '{park_location}' → '{normalized_location}': Found {len(park_fields)} fields")
            print(f"   Key field UF_CRM_1704787907109 = '{park_fields.get('UF_CRM_1704787907109', 'NOT SET')}'")
        else:
            print(f"⚠️ No park fields found for '{park_location}' → '{normalized_location}'")
        
        return park_fields
    def _generate_lead_title(
        self, user_info: Dict, category_analysis: Dict, park_location: str = "General"
    ) -> str:
        """Generate lead title with park name as primary identifier"""

        name = user_info.get('name', 'Web Visitor')
        category = category_analysis.get('category', 'general')

        # Use park location as the primary title element
        if park_location and park_location != "General":
            if category == 'birthday_party':
                return f"🎂 {park_location} - Birthday Party ({name})"

            return f"💬 {park_location} - General Inquiry ({name})"

        # When no specific park is detected
        if category == 'birthday_party':
            return f"🎂 Leo & Loona - Birthday Party ({name})"

        return f"💬 Leo & Loona - General Inquiry ({name})"
    def _generate_lead_description(
        self, conversation_history: List[Dict], category_analysis: Dict, park_location: str = "General"
    ) -> str:
        """Generate detailed lead description from conversation with location info"""

        # Extract key conversation points
        user_messages = [
            msg.get('content', '') for msg in conversation_history
            if msg.get('role') == 'user'
        ]

        # Build description
        description_parts = []

        # Category and confidence
        category = category_analysis.get('category', 'general')
        confidence = category_analysis.get('confidence', 0.0)
        reasoning = category_analysis.get('reasoning', '')

        description_parts.append("🤖 **Chatbot Conversation Analysis**")
        description_parts.append(f"Category: {category.title().replace('_', ' ')}")
        description_parts.append(f"Park Location: {park_location}")
        description_parts.append(f"Confidence: {confidence:.1%}")
        description_parts.append(f"Analysis: {reasoning}")
        description_parts.append("")

        # Keywords found
        keywords = category_analysis.get('keywords_found', [])
        if keywords:
            description_parts.append(f"🔍 **Keywords Identified**: {', '.join(keywords)}")
            description_parts.append("")

        # Conversation summary
        description_parts.append("💬 **Conversation Summary**")
        description_parts.append(f"Total Messages: {len(conversation_history)}")
        description_parts.append(f"User Questions: {len(user_messages)}")
        description_parts.append("")

        # First user message (original inquiry)
        if user_messages:
            description_parts.append("**Original Inquiry:**")
            description_parts.append(f'"{user_messages[0]}"')
            description_parts.append("")

        # All user messages for context
        if len(user_messages) > 1:
            description_parts.append("**All User Messages:**")
            for i, msg in enumerate(user_messages, 1):
                description_parts.append(f"{i}. {msg}")
            description_parts.append("")

        # Timestamp
        description_parts.append(
            f"📅 **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        description_parts.append("🔗 **Source**: Leo & Loona AI Assistant")

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
            with open('user_data/bitrix_leads_created.log', 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except (OSError, IOError) as e:
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

