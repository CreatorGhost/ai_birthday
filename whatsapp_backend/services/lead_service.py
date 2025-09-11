"""
WhatsApp Lead Service
Handles immediate lead creation, progressive updates, and birthday monitoring
for WhatsApp integration without modifying the core RAG pipeline
"""

import asyncio
import logging
import re
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class WhatsAppLeadService:
    """Service for managing WhatsApp leads with immediate creation and progressive updates"""
    
    def __init__(self, lead_manager, user_tracker):
        self.lead_manager = lead_manager
        self.user_tracker = user_tracker
        
        # Birthday keywords for monitoring
        self.birthday_keywords = [
            'birthday', 'b-day', 'bday', 'party', 'celebration', 'celebrate',
            'anniversary', 'special day', 'event planning', 'event', 'booking'
        ]
        
        # Mall mapping for quick lookup
        self.mall_mapping = {
            "FESTIVAL_CITY": "Festival City",
            "DALMA_MALL": "Dalma Mall", 
            "YAS_MALL": "Yas Mall",
            "General": "General"
        }
    
    async def ensure_lead_exists(self, phone: str, first_message: str) -> Optional[str]:
        """
        PHASE 1: Immediate Lead Creation
        Create lead immediately on first message if doesn't exist
        
        Args:
            phone: User phone number
            first_message: The first message content
            
        Returns:
            lead_id if created, None if already exists or error
        """
        try:
            logger.info(f"🔍 Checking if lead exists for {phone}")
            
            # Check if user already has a lead
            user_profile = self.user_tracker.get_user_profile(phone)
            
            if user_profile and user_profile.get('bitrix_lead_id'):
                logger.info(f"✅ Lead already exists for {phone}: {user_profile.get('bitrix_lead_id')}")
                return None
            
            # Create immediate lead with WhatsApp flow
            logger.info(f"🚀 Creating immediate lead for {phone} (WhatsApp flow)")
            
            lead_data = {
                'name': 'Unknown User',  # Temporary name
                'phone': phone,
                'park_location': 'General',  # Default location
                'conversation_content': 'WhatsApp General Inquiry',  # Use generic content to avoid birthday detection
                'source': 'WhatsApp Chatbot',
                'stage': 'Inquiry',  # Start in Inquiry
                'assigned_to': 'AI Chatbot Assistant'  # AI Bot (ID: 38005)
            }
            
            # Create lead using existing lead manager
            lead_info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.lead_manager.create_simple_lead(
                    name=lead_data['name'],
                    phone=lead_data['phone'],
                    park_location=lead_data['park_location'],
                    conversation_content=lead_data['conversation_content']
                )
            )
            
            if lead_info and lead_info.get('success'):
                lead_id = str(lead_info['lead_id'])
                # Update user tracker with lead ID using correct method
                self.user_tracker.update_user_lead_info(
                    phone=phone,
                    lead_id=lead_id,
                    action="created",
                    park_location=lead_info.get('park_location', 'General')
                )
                
                logger.info(f"✅ Immediate lead created for {phone}: {lead_info}")
                return lead_id
            else:
                logger.error(f"❌ Failed to create lead for {phone}: {lead_info}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error creating immediate lead for {phone}: {e}")
            return None
    
    async def update_lead_progress(self, phone: str, extracted_info: Dict, message: str) -> bool:
        """
        PHASE 2: Progressive Lead Updates
        Update lead information as we learn more about the user
        
        Args:
            phone: User phone number
            extracted_info: Information extracted from RAG pipeline
            message: Current message content
            
        Returns:
            True if updated, False otherwise
        """
        try:
            user_profile = self.user_tracker.get_user_profile(phone)
            if not user_profile or not user_profile.get('bitrix_lead_id'):
                logger.warning(f"⚠️ No lead found for {phone} to update")
                return False
            
            lead_id = user_profile.get('bitrix_lead_id')
            updates_made = False
            
            # Check for name update
            if extracted_info and extracted_info.get('name'):
                new_name = extracted_info.get('name')
                current_name = user_profile.get('name', 'Unknown User')
                
                if new_name != current_name and new_name != 'Unknown User':
                    logger.info(f"📝 Name detected for {phone}: {current_name} → {new_name}")
                    
                    # Update user profile
                    self.user_tracker.update_user_profile(phone, {'name': new_name})
                    
                    # Update lead title (but keep AI Bot assignment)
                    await self._update_lead_title_and_info(lead_id, phone, new_name, keep_ai_assignment=True)
                    updates_made = True
            
            # Check for mall/location update
            if extracted_info and extracted_info.get('mall_location'):
                new_mall = extracted_info.get('mall_location')
                current_mall = user_profile.get('current_park_location', 'General')
                
                if new_mall != current_mall and new_mall in self.mall_mapping:
                    logger.info(f"🏢 Mall detected for {phone}: {current_mall} → {new_mall}")
                    
                    # Update user profile
                    self.user_tracker.update_user_profile(phone, {
                        'current_park_location': new_mall,
                        'original_park_location': user_profile.get('original_park_location', new_mall)
                    })
                    
                    # Update lead mall fields (but keep AI Bot assignment)
                    await self._update_lead_mall_info(lead_id, new_mall, keep_ai_assignment=True)
                    updates_made = True
            
            if updates_made:
                logger.info(f"✅ Lead {lead_id} updated for {phone}")
                
            return updates_made
            
        except Exception as e:
            logger.error(f"❌ Error updating lead progress for {phone}: {e}")
            return False
    
    async def monitor_birthday_keywords(self, phone: str, message: str, chat_history: List[Dict]) -> bool:
        """
        PHASE 3: Continuous Birthday Monitoring
        Monitor every message for birthday keywords and convert lead to sales team
        
        Args:
            phone: User phone number
            message: Current message content
            chat_history: Conversation history
            
        Returns:
            True if birthday detected and lead moved, False otherwise
        """
        try:
            user_profile = self.user_tracker.get_user_profile(phone)
            if not user_profile or not user_profile.get('bitrix_lead_id'):
                logger.warning(f"⚠️ No lead found for {phone} to monitor")
                return False
            
            lead_id = user_profile.get('bitrix_lead_id')
            current_stage = user_profile.get('lead_stage', 'Inquiry')
            
            logger.info(f"🎂 Checking birthday conversion for lead {lead_id}, current stage: '{current_stage}'")
            
            # Skip if already moved to birthday flow (check both user profile and Bitrix status)
            if current_stage in ['New Approach', 'UC_NJ6R1M', 'NEW APPROACH']:
                logger.info(f"🎂 Skipping conversion - already in birthday stage: {current_stage}")
                return False
            
            # Check for birthday keywords in current message
            message_lower = message.lower()
            birthday_detected = any(keyword in message_lower for keyword in self.birthday_keywords)
            
            if birthday_detected:
                logger.info(f"🎂 Birthday keywords detected for {phone}: '{message}'")
                
                # Move lead to New Approach (Birthday) stage
                success = await self._convert_to_birthday_lead(lead_id, phone)
                
                if success:
                    logger.info(f"✅ Lead {lead_id} converted to birthday flow and assigned to sales team")
                    return True
                else:
                    logger.error(f"❌ Failed to convert lead {lead_id} to birthday")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error monitoring birthday keywords for {phone}: {e}")
            return False
    
    async def _update_lead_title_and_info(self, lead_id: str, phone: str, name: str, keep_ai_assignment: bool = True) -> bool:
        """Update lead title with new name while preserving AI assignment"""
        try:
            user_profile = self.user_tracker.get_user_profile(phone)
            mall = user_profile.get('current_park_location', 'General')
            mall_name = self.mall_mapping.get(mall, 'General')
            current_stage = user_profile.get('lead_stage', 'Inquiry')
            
            # Generate new title (General inquiry format)
            new_title = f"💬 {mall_name} - General Inquiry ({name})"
            
            # Update lead using Bitrix client
            logger.info(f"📝 Updating lead {lead_id} title: {new_title}")
            
            update_data = {
                'TITLE': new_title,
                'NAME': name
            }
            
            # Only move to General Questions if not already in birthday stage
            if current_stage not in ['New Approach', 'UC_NJ6R1M', 'NEW APPROACH']:
                update_data['STATUS_ID'] = 'UC_0MD91B'  # Move to General Questions when name is known
            else:
                logger.info(f"🎂 Preserving birthday stage for lead {lead_id}: {current_stage}")
            
            # If we need to preserve AI assignment, don't change ASSIGNED_BY_ID
            if keep_ai_assignment:
                logger.info(f"🤖 Preserving AI Bot assignment for lead {lead_id}")
            
            # Update via lead manager's client
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.lead_manager.client.update_lead(int(lead_id), update_data)
            )
            
            if result and 'result' in result:
                logger.info(f"✅ Lead {lead_id} title and stage updated successfully")
                return True
            else:
                logger.error(f"❌ Failed to update lead {lead_id}: {result}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error updating lead title for {lead_id}: {e}")
            return False
    
    async def _update_lead_mall_info(self, lead_id: str, mall: str, keep_ai_assignment: bool = True) -> bool:
        """Update lead mall information while preserving AI assignment"""
        try:
            logger.info(f"🏢 Updating lead {lead_id} mall: {mall}")
            
            # Get park fields for the specific mall
            park_fields = self.lead_manager.get_park_fields(mall)
            
            if not park_fields:
                logger.warning(f"⚠️ No park fields found for mall: {mall}")
                return False
            
            # Update lead with mall-specific fields
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.lead_manager.client.update_lead(int(lead_id), park_fields)
            )
            
            if result and 'result' in result:
                logger.info(f"✅ Lead {lead_id} mall fields updated to {mall}")
                return True
            else:
                logger.error(f"❌ Failed to update lead {lead_id} mall fields: {result}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error updating lead mall for {lead_id}: {e}")
            return False
    
    async def _convert_to_birthday_lead(self, lead_id: str, phone: str) -> bool:
        """Convert lead to birthday flow and assign to sales team"""
        try:
            user_profile = self.user_tracker.get_user_profile(phone)
            name = user_profile.get('name', 'Unknown User')
            mall = user_profile.get('current_park_location', 'General')
            mall_name = self.mall_mapping.get(mall, 'General')
            
            # Generate birthday title
            birthday_title = f"🎂 {mall_name} - Birthday Party ({name})"
            
            logger.info(f"🎂 Converting lead {lead_id} to birthday: {birthday_title}")
            logger.info(f"🏢 Converting for mall: {mall} → {mall_name}")
            
            # Get assignment info for the mall
            assignment_info = self.lead_manager.get_assigned_user_for_mall(mall)
            assigned_user_id = assignment_info['user_id']
            assigned_user_name = assignment_info['user_name']
            team_name = assignment_info['team_name']
            
            logger.info(f"👤 Assigning to {team_name}: {assigned_user_name} (ID: {assigned_user_id})")
            
            # Prepare update data for birthday conversion
            update_data = {
                'TITLE': birthday_title,
                'NAME': name,
                'STATUS_ID': 'UC_NJ6R1M',  # NEW APPROACH stage for birthday
                'ASSIGNED_BY_ID': assigned_user_id  # Assign to sales team
            }
            
            logger.info(f"📋 Birthday conversion data: {update_data}")
            
            # Add park-specific fields if mall is specified
            if mall != 'General':
                park_fields = self.lead_manager._get_park_field_values(mall)
                if park_fields:
                    update_data.update(park_fields)
                    logger.info(f"🏢 Adding park fields for {mall}")
            
            # Update lead in Bitrix
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.lead_manager.client.update_lead(int(lead_id), update_data)
            )
            
            if result and 'result' in result:
                logger.info(f"✅ Lead {lead_id} converted to birthday and assigned to {team_name}")
                logger.info(f"   👤 Customer: {name} ({phone})")
                logger.info(f"   🏢 Location: {mall_name}")
                logger.info(f"   👨‍💼 Assigned to: {assigned_user_name}")
                return True
            else:
                logger.error(f"❌ Failed to convert lead {lead_id} to birthday: {result}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error converting lead {lead_id} to birthday: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get lead service status"""
        return {
            "service": "WhatsApp Lead Service",
            "birthday_keywords": len(self.birthday_keywords),
            "mall_mapping": list(self.mall_mapping.keys()),
            "ready": True
        }
