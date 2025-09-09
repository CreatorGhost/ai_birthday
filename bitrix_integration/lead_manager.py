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
        },
        'Chatbot': {
            'team_name': 'AI Chatbot Assistant',
            'user_ids': [38005],  # Chatbot user - Finance.dfcm@leoloona.ae
            'user_names': ['AI Chatbot Assistant']
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
            'general': 'General',
            'Chatbot': 'Chatbot',
            'chatbot': 'Chatbot'
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
        self, name: str, phone: str, park_location: str = "General", conversation_content: str = ""
    ) -> Optional[Dict]:
        """Create a simple lead in Bitrix with content-based routing (birthday vs general)"""

        if not self.client:
            print("Bitrix client not available - lead not created")
            return None

        try:
            # Analyze conversation content for birthday detection
            is_birthday_query = self._detect_birthday_content(conversation_content)
            
            if is_birthday_query:
                # Birthday queries: Assign to mall-specific teams, go to NEW APPROACH
                print(f"🎂 Birthday party query detected - routing to NEW APPROACH stage")
                assignment_info = self.get_assigned_user_for_mall(park_location)
                status_id = 'UC_NJ6R1M'  # NEW APPROACH for birthday parties
                title_prefix = "🎂 BIRTHDAY PARTY"
            else:
                # General queries: Assign to chatbot, stay in Inquiry
                print(f"💬 General query detected - routing to Inquiry stage for chatbot")
                assignment_info = self.get_assigned_user_for_mall('Chatbot')
                status_id = 'NEW'  # Inquiry stage for general questions
                title_prefix = "🤖 CHATBOT"
            
            assigned_user_id = assignment_info['user_id']
            assigned_user_name = assignment_info['user_name']
            team_name = assignment_info['team_name']
            
            # Create lead data based on content type (birthday vs general)
            lead_data = {
                'TITLE': f"New Lead - {name}",  # FIXED: Clean professional title
                'NAME': name,  # FIXED: Clean name without brackets
                'LAST_NAME': '',  # Empty like successful leads
                'STATUS_ID': status_id,  # NEW (Inquiry) for general, UC_NJ6R1M (NEW APPROACH) for birthday
                'SOURCE_ID': '1|OLCHAT_WA_CONNECTOR_2',
                'PHONE': [{'VALUE': phone, 'VALUE_TYPE': 'WORK'}] if phone else [],
                'COMMENTS': f"{title_prefix} LEAD - {'BIRTHDAY PARTY' if is_birthday_query else 'GENERAL INQUIRY'}\n"
                           f"📍 Park Interest: {park_location}\n"
                           f"👤 Customer: {name}\n"
                           f"📞 Phone: {phone}\n"
                           f"🕒 Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                           f"🎯 Assigned to: {assigned_user_name} ({team_name})\n"
                           f"{'🎂 Birthday party inquiry - ready for sales team' if is_birthday_query else '🤖 General inquiry - ready for chatbot processing'}\n"
                           f"💬 Content: {conversation_content[:100]}..." if conversation_content else "",
                'ASSIGNED_BY_ID': assigned_user_id,  # Dynamic assignment based on content
                'OPENED': 'Y',
                'IS_MANUAL_OPPORTUNITY': 'N',  # Keep same as successful leads
                'OPPORTUNITY': 0.00  # Keep same as successful leads
            }

            # 🔧 CRITICAL FIX: ALWAYS add park fields when location is known (not just for birthdays)
            if park_location and park_location != "General":
                park_fields = self._get_park_field_values(park_location)
                if park_fields:
                    lead_data.update(park_fields)
                    print(f"🏢 Added park fields to lead creation: {park_location} → {len(park_fields)} fields")
                    
                    # Add dynamic date field for birthday queries
                    if is_birthday_query:
                        current_date = datetime.now().strftime('%Y-%m-%dT03:00:00+03:00')
                        lead_data['UF_CRM_1711693248818'] = [current_date]
                else:
                    print(f"⚠️ No park fields found for location: {park_location}")

            # Debug: Print the lead data being sent
            print(f"🔧 Creating SIMPLE lead with data: {lead_data}")

            # Create lead in Bitrix
            response = self.client.create_lead(lead_data)

            if response and 'result' in response:
                lead_id = response['result']
                stage_name = "NEW APPROACH (Birthday Party)" if is_birthday_query else "Inquiry (General Question)"
                print(
                    f"✅ Lead created in Bitrix: ID {lead_id}"
                    f"\n   Customer: {name} ({phone})"
                    f"\n   Type: {'🎂 Birthday Party' if is_birthday_query else '💬 General Question'}"
                    f"\n   Stage: {stage_name}"
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

    def _detect_birthday_content(self, content: str) -> bool:
        """Detect if conversation content is about birthday parties using intelligent LLM detection"""
        
        if not content:
            return False
        
        content_lower = content.lower()
        
        # Quick keyword check first (for obvious cases to save API calls)
        obvious_keywords = ['birthday', 'birthdays', 'bday', 'party', 'parties', 'celebration',
                           'biortdays', 'bithday', 'birhtday', 'birtday']  # Common typos
        
        for keyword in obvious_keywords:
            if keyword in content_lower:
                print(f"🎂 Quick birthday detection: '{keyword}' found")
                return True
        
        # For ambiguous cases, use LLM for intelligent detection
        try:
            from openai import OpenAI
            import os
            
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            # Use a fast, cheap model for classification
            response = client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a classifier that detects if a message is asking about birthday parties, celebrations, or party-related services.
                        
                        IMPORTANT: Pay special attention to TYPOS and variations of birthday-related words.
                        
                        Respond with ONLY 'YES' or 'NO'.
                        
                        Examples that should be YES:
                        - "Do you do birthdays?"
                        - "do you do biortdays?" (typo for birthdays)
                        - "hey do you do biortdays?" (typo for birthdays)  
                        - "Can we celebrate there?"
                        - "My son is turning 5"
                        - "Looking for a party venue"
                        - "Do you have party packages?"
                        - "Can you host events?"
                        - "Planning something special for my daughter"
                        - "Need a place for celebration"
                        - "do u do birthdays"
                        - "can i book for a party"
                        - "bithday party" (typo)
                        - "birhtday" (typo)
                        - "patrues" (typo for parties)
                        
                        Examples that should be NO:
                        - "What are your opening hours?"
                        - "How much are tickets?"
                        - "Where are you located?"
                        - "Do you have parking?"
                        - "What activities do you have?"
                        - "How much for socks?"
                        """
                    },
                    {
                        "role": "user",
                        "content": f"Is this message asking about birthdays/parties/celebrations? Message: '{content}'"
                    }
                ],
                max_completion_tokens=10
            )
            
            result = response.choices[0].message.content.strip().upper()
            is_birthday = result == "YES"
            
            if is_birthday:
                print(f"🎂 LLM birthday detection: YES for '{content[:100]}...'")
            else:
                print(f"📝 LLM birthday detection: NO for '{content[:100]}...'")
                
            return is_birthday
            
        except Exception as e:
            print(f"⚠️ LLM detection failed, falling back to keyword matching: {e}")
            
            # Fallback to enhanced keyword detection if LLM fails
            extended_keywords = [
                'party', 'parties', 'celebration', 'celebrate', 'event',
                'turning', 'years old', 'special day', 'venue', 'host',
                'package', 'packages', 'booking', 'reserve',
                # Common inquiry patterns
                'do u do', 'do you do', 'can you do', 'can u do',
                'do u have', 'do you have', 'offer birthday', 'host birthday'
            ]
            
            for keyword in extended_keywords:
                if keyword in content_lower:
                    print(f"🎂 Fallback birthday keyword: '{keyword}' found")
                    return True
                    
        return False

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
            # Show all park fields being set
            for field_name, field_value in park_fields.items():
                print(f"   🔧 {field_name} = {field_value}")
        else:
            print(f"⚠️ No park fields found for '{park_location}' → '{normalized_location}'")
            print(f"   Available locations in mapping: {list(park_field_mappings.keys())}")
        
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
        # 3. NOT already created for this user (prevents duplicates)

        has_name = bool(user_info.get('name'))
        meaningful_conversation = len(conversation_history) >= 2
        no_existing_lead = user_info.get('bitrix_lead_id') is None
        
        should_create = has_name and meaningful_conversation and no_existing_lead
        
        if not should_create:
            if not has_name:
                print(f"🚫 Not creating lead: No name yet")
            elif not meaningful_conversation:
                print(f"🚫 Not creating lead: Only {len(conversation_history)} messages")
            elif not no_existing_lead:
                existing_lead_id = user_info.get('bitrix_lead_id')
                print(f"🚫 Not creating lead: User already has lead {existing_lead_id}")

        return should_create

    def update_existing_lead(self, user_info: Dict, park_location: str = None, conversation_content: str = "", interaction_count: int = 0) -> Optional[Dict]:
        """Update an existing lead with new information - convert to birthday lead if needed or update location/stage"""
        
        if not self.client:
            print("Bitrix client not available - lead not updated")
            return None
            
        lead_id = user_info.get('bitrix_lead_id')
        if not lead_id:
            print("❌ No existing lead ID found for user")
            return None
            
        try:
            name = user_info.get('name', 'Web Visitor')
            phone = user_info.get('phone', '')
            
            # Check if this is a birthday question
            is_birthday_question = self._detect_birthday_content(conversation_content)
            
            print(f"🔍 UPDATE DEBUG - Birthday check result: {is_birthday_question}")
            print(f"🔍 UPDATE DEBUG - Content being checked: '{conversation_content[:100]}...'")
            print(f"🔍 UPDATE DEBUG - Park location: {park_location}")
            
            # Prepare update data
            update_data = {}
            
            if is_birthday_question:
                # Birthday question → Stay in Inquiry but assign to sales team
                print(f"🎂 Birthday question detected - staying in Inquiry but assigning to sales team for {park_location}")
                
                # Keep in Inquiry stage (STATUS_ID = 'NEW') but assign to sales team
                update_data['TITLE'] = f"Birthday - {name}"
                update_data['NAME'] = name  # FIXED: Clean name without brackets
                
                # Get mall-specific assignment for birthday
                if park_location and park_location != "General":
                    assignment_info = self.get_assigned_user_for_mall(park_location)
                    if assignment_info:
                        update_data['ASSIGNED_BY_ID'] = assignment_info['user_id']
                        update_data['COMMENTS'] = f"🎂 BIRTHDAY PARTY INQUIRY\n" \
                                               f"📍 Location: {park_location}\n" \
                                               f"👤 Customer: {name}\n" \
                                               f"📞 Phone: {phone}\n" \
                                               f"🔄 Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" \
                                               f"🎯 Assigned to: {assignment_info['user_name']} ({assignment_info['team_name']})\n" \
                                               f"📋 Status: Inquiry (Birthday)\n" \
                                               f"🎂 Birthday inquiry - ready for sales team\n" \
                                               f"💬 Question: {conversation_content[:100]}..."
                    
                    # Add park fields for birthday leads
                    park_fields = self._get_park_field_values(park_location)
                    if park_fields:
                        update_data.update(park_fields)
                        
                    print(f"🎂 Birthday lead: Inquiry → Inquiry + {assignment_info['user_name']}")
                
            else:
                # General question → Move to General Questions stage
                print(f"💬 General question detected - moving to General Questions stage")
                
                update_data['STATUS_ID'] = 'UC_0MD91B'  # General Questions
                update_data['TITLE'] = f"General Inquiry - {name}"
                update_data['NAME'] = name  # FIXED: Clean name without brackets
                
                # CRITICAL FIX: Always add park location fields when location is known
                if park_location and park_location != "General":
                    park_fields = self._get_park_field_values(park_location)
                    if park_fields:
                        update_data.update(park_fields)
                        print(f"🏢 Added/updated park location fields: {park_location} → {len(park_fields)} fields")
                    else:
                        print(f"⚠️ No park fields found for location: {park_location}")
                update_data['COMMENTS'] = f"💬 GENERAL QUESTIONS LEAD\n" \
                                       f"👤 Customer: {name}\n" \
                                       f"📞 Phone: {phone}\n" \
                                       f"🔄 Moved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" \
                                       f"🎯 Assigned to: AI Chatbot Assistant (38005)\n" \
                                       f"📋 Status: General Questions\n" \
                                       f"💬 General inquiry - ready for review\n" \
                                       f"📝 Content: {conversation_content[:100]}..."
                
                print(f"💬 Lead progression: Inquiry → General Questions")
            
            # Only update if we have something to update
            if not update_data:
                print(f"ℹ️ No updates needed for lead {lead_id}")
                return None
                
            print(f"🔄 Updating lead {lead_id} with data: {update_data}")
            
            # Update lead in Bitrix  
            response = self.client.update_lead(lead_id, update_data)
            
            if response and response.get('result'):
                action_type = "converted to birthday lead" if is_birthday_question else "updated"
                print(f"✅ Lead {lead_id} {action_type} successfully")
                return {
                    'lead_id': lead_id,
                    'action': 'birthday_conversion' if is_birthday_question else 'updated',
                    'name': name,
                    'phone': phone,
                    'park_location': park_location,
                    'updated_fields': list(update_data.keys()),
                    'is_birthday': is_birthday_question
                }
            else:
                print(f"❌ Failed to update lead {lead_id}: {response}")
                return None
                
        except Exception as e:
            print(f"❌ Error updating lead {lead_id}: {str(e)}")
            return None

    def should_update_lead(self, user_info: Dict, new_park_location: str = None, conversation_content: str = "", interaction_count: int = 0, current_stage: str = None) -> bool:
        """Determine if existing lead should be updated with new information"""
        
        # ENHANCED LOGIC with mall clarification:
        # 1. Birthday question + mall known → Stay in Inquiry + assign to sales team
        # 2. Birthday question + mall unknown → Ask for mall clarification (no update yet)
        # 3. General question → Move to General Questions stage (if not already there)
        
        has_lead = user_info.get('bitrix_lead_id') is not None
        
        if not has_lead:
            return False
        
        # Check if this is a birthday-related question
        is_birthday_question = self._detect_birthday_content(conversation_content)
        
        if is_birthday_question:
            # Birthday question → Check if we have mall information
            if new_park_location and new_park_location != "General":
                lead_id = user_info.get('bitrix_lead_id')
                print(f"🎂 Should update lead {lead_id}: Birthday question for '{new_park_location}' - assigning to sales team (staying in Inquiry)")
                return True
            else:
                # Birthday question but no mall specified → Will trigger clarification (no lead update)
                lead_id = user_info.get('bitrix_lead_id')
                print(f"🎂 Birthday question detected for lead {lead_id} - need mall clarification first (no update)")
                return False
        else:
            # General question → Move to General Questions stage if not already there
            if current_stage and current_stage == 'UC_0MD91B':
                # Already in General Questions stage, no update needed
                lead_id = user_info.get('bitrix_lead_id')
                print(f"💬 Lead {lead_id} already in General Questions stage - no update needed")
                return False
            else:
                # Move to General Questions stage
                lead_id = user_info.get('bitrix_lead_id')
                print(f"💬 Should update lead {lead_id}: General question - moving to General Questions stage")
                return True

    def test_connection(self) -> bool:
        """Test Bitrix connection"""
        if not self.client:
            return False
        return self.client.test_connection()

