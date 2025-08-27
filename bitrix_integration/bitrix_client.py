import os
import requests
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class BitrixClient:
    """Simple Bitrix24 API client for connection testing."""
    
    def __init__(self, config=None, webhook_url: str = None):
        """
        Initialize Bitrix client.
        
        Args:
            config: BitrixConfig object (preferred)
            webhook_url: Bitrix24 webhook URL. If not provided, will use BITRIX_WEBHOOK_URL from environment
        """
        if config:
            self.webhook_url = config.webhook_url
        else:
            self.webhook_url = webhook_url or os.getenv('BITRIX_WEBHOOK_URL')
        
        if not self.webhook_url:
            raise ValueError("Bitrix webhook URL is required. Set BITRIX_WEBHOOK_URL in .env file")
        
        # Remove trailing slash if present
        self.webhook_url = self.webhook_url.rstrip('/')
    
    def _make_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Make API request to Bitrix24.
        
        Args:
            method: Bitrix API method (e.g., 'profile')
            params: Request parameters
            
        Returns:
            API response data
        """
        url = f"{self.webhook_url}/{method}"
        
        try:
            response = requests.post(url, json=params or {}, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'error' in data:
                raise Exception(f"Bitrix API error: {data['error']}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to connect to Bitrix: {str(e)}")
    
    def test_connection(self) -> bool:
        """
        Test connection to Bitrix24 API.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = self._make_request('profile')
            print(f"✅ Connection successful! User: {response.get('result', {}).get('NAME', 'Unknown')}")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            return False
    
    def get_user_info(self) -> Dict[str, Any]:
        """
        Get current user information.
        
        Returns:
            User information dictionary
        """
        return self._make_request('profile')
    
    def get_all_data(self, method: str, select_fields: list, filter_params: Dict[str, Any] = None) -> list:
        """
        Fetch all data with pagination for any list method (e.g., leads or deals).
        
        Args:
            method: API method (e.g., 'crm.lead.list')
            select_fields: Fields to retrieve (e.g., ['ID', 'TITLE'])
            filter_params: Optional filters (e.g., {'STATUS_ID': 'NEW'})
            
        Returns:
            List of all items
        """
        all_data = []
        start = 0
        batch_size = 50  # Bitrix max per call
        max_iterations = 100  # Prevent infinite loops
        iteration_count = 0
        
        while iteration_count < max_iterations:
            params = {
                'select': select_fields,
                'start': start
            }
            if filter_params:
                params['filter'] = filter_params
            
            response = self._make_request(method, params)
            items = response.get('result', [])
            
            if not items:  # No more data
                break
                
            all_data.extend(items)
            
            # Check if there are more pages
            if len(items) < batch_size or 'next' not in response:
                break
            
            start += batch_size
            iteration_count += 1
        
        return all_data
    
    def get_leads(self, select_fields: list = None, filter_params: Dict[str, Any] = None) -> list:
        """
        Get all leads.
        
        Args:
            select_fields: Fields to retrieve (default: key ones)
            filter_params: Optional filters
            
        Returns:
            List of leads
        """
        if not select_fields:
            select_fields = ['ID', 'TITLE', 'NAME', 'LAST_NAME', 'STATUS_ID', 'DATE_CREATE', 'COMMENTS']
        return self.get_all_data('crm.lead.list', select_fields, filter_params)
    
    def get_deals(self, select_fields: list = None, filter_params: Dict[str, Any] = None) -> list:
        """
        Get all deals.
        
        Args:
            select_fields: Fields to retrieve (default: key ones)
            filter_params: Optional filters
            
        Returns:
            List of deals
        """
        if not select_fields:
            select_fields = ['ID', 'TITLE', 'OPPORTUNITY', 'STAGE_ID', 'DATE_CREATE', 'COMMENTS']
        return self.get_all_data('crm.deal.list', select_fields, filter_params)
    
    def create_lead(self, lead_data: Dict[str, Any], register_event: bool = True) -> Dict[str, Any]:
        """
        Create a new lead in Bitrix24.
        
        Args:
            lead_data: Dictionary containing lead fields
            register_event: Whether to register the event of adding a lead
            
        Returns:
            API response with created lead ID
            
        Example:
            lead_data = {
                'TITLE': 'New Lead Title',
                'NAME': 'John',
                'LAST_NAME': 'Doe', 
                'STATUS_ID': 'NEW',
                'PHONE': [{'VALUE': '+1234567890', 'VALUE_TYPE': 'WORK'}],
                'EMAIL': [{'VALUE': 'john@example.com', 'VALUE_TYPE': 'WORK'}],
                'COMMENTS': 'Lead created via API'
            }
        """
        params = {
            'fields': lead_data,
            'params': {
                'REGISTER_SONET_EVENT': 'Y' if register_event else 'N'
            }
        }
        
        response = self._make_request('crm.lead.add', params)
        return response
    
    def update_lead(self, lead_id: int, lead_data: Dict[str, Any], register_event: bool = True) -> Dict[str, Any]:
        """
        Update an existing lead in Bitrix24.
        
        Args:
            lead_id: ID of the lead to update
            lead_data: Dictionary containing fields to update
            register_event: Whether to register the event of updating a lead
            
        Returns:
            API response confirming update
            
        Example:
            lead_data = {
                'STATUS_ID': 'IN_PROCESS',
                'COMMENTS': 'Updated lead status',
                'OPPORTUNITY': 5000
            }
        """
        params = {
            'id': lead_id,
            'fields': lead_data,
            'params': {
                'REGISTER_SONET_EVENT': 'Y' if register_event else 'N'
            }
        }
        
        response = self._make_request('crm.lead.update', params)
        return response
    
    def get_lead_by_id(self, lead_id: int) -> Dict[str, Any]:
        """
        Get a specific lead by its ID.
        
        Args:
            lead_id: ID of the lead to retrieve
            
        Returns:
            Lead data dictionary
        """
        params = {'id': lead_id}
        response = self._make_request('crm.lead.get', params)
        return response
    
    def get_lead_fields(self) -> Dict[str, Any]:
        """
        Get available lead fields and their descriptions.
        
        Returns:
            Dictionary of available lead fields with their properties
        """
        response = self._make_request('crm.lead.fields')
        return response
    
    def get_lead_activities(self, lead_id, activity_types=None):
        """
        Get activities (calls, emails, meetings, etc.) for a specific lead
        
        Args:
            lead_id (int): Lead ID
            activity_types (list, optional): Filter by activity types (1=call, 2=meeting, 4=email, etc.)
        
        Returns:
            dict: Response containing lead activities
        """
        filter_params = {
            'OWNER_TYPE_ID': 1,  # 1 = Lead
            'OWNER_ID': lead_id
        }
        
        if activity_types:
            filter_params['TYPE_ID'] = activity_types
        
        params = {
            'filter': filter_params,
            'select': ['*', 'COMMUNICATIONS'],
            'order': {'ID': 'DESC'}
        }
        
        return self._make_request('crm.activity.list', params)
    
    def get_lead_timeline_comments(self, lead_id):
        """
        Get timeline comments for a specific lead
        
        Args:
            lead_id (int): Lead ID
        
        Returns:
            dict: Response containing timeline comments
        """
        params = {
            'filter': {
                'ENTITY_ID': lead_id,
                'ENTITY_TYPE': 'lead'
            },
            'select': ['ID', 'COMMENT', 'CREATED', 'AUTHOR_ID', 'FILES'],
            'order': {'ID': 'DESC'}
        }
        
        return self._make_request('crm.timeline.comment.list', params)
    
    def get_open_channel_messages(self, communication_value):
        """
        Get Open Channel (WhatsApp/Telegram/etc.) messages from ImConnector
        
        Args:
            communication_value (str): ImConnector communication value like "imol|olchat_wa_connector_2|1|971558505328|47"
        
        Returns:
            dict: Open Channel messages response
        """
        try:
            # Parse the ImConnector communication value
            # Format: imol|connector_id|line_id|external_id|session_id
            parts = communication_value.split('|')
            if len(parts) >= 5 and parts[0] == 'imol':
                connector_id = parts[1]
                line_id = parts[2]
                external_id = parts[3]
                session_id = parts[4]
                
                # Try different API methods to get Open Channel messages
                params = {
                    'CONNECTOR': connector_id,
                    'LINE': line_id,
                    'CHAT_ID': session_id
                }
                
                try:
                    # Method 1: Try imconnector.send.list (if available)
                    response = self._make_request('imconnector.send.list', params)
                    if 'result' in response:
                        return response
                except:
                    pass
                
                try:
                    # Method 2: Try imopenlines.session.get
                    params = {'SESSION_ID': session_id}
                    response = self._make_request('imopenlines.session.get', params)
                    if 'result' in response:
                        return response
                except:
                    pass
                
                try:
                    # Method 3: Try im.chat.get for chat messages
                    params = {'CHAT_ID': f"chat{session_id}"}
                    response = self._make_request('im.chat.get', params)
                    if 'result' in response:
                        return response
                except:
                    pass
                
                # Method 4: Try im.message.list for specific chat
                try:
                    params = {'CHAT_ID': session_id, 'LIMIT': 100}
                    response = self._make_request('im.message.list', params)
                    if 'result' in response:
                        return response
                except:
                    pass
                
        except Exception as e:
            print(f"    Warning: Could not retrieve Open Channel messages: {e}")
        
        return None
    
    def get_lead_conversation_history(self, lead_id):
        """
        Get complete conversation history for a lead (activities + timeline comments + Open Channel messages)
        
        Args:
            lead_id (int): Lead ID
        
        Returns:
            dict: Combined conversation history with activities, comments, and Open Channel messages
        """
        # Get activities (calls, emails, meetings, etc.)
        activities_response = self.get_lead_activities(lead_id)
        
        # Get timeline comments
        comments_response = self.get_lead_timeline_comments(lead_id)
        
        # Extract Open Channel messages from activities
        open_channel_messages = []
        activities = activities_response.get('result', [])
        
        for activity in activities:
            # Look for Open Channel activities
            if 'Open Channel' in str(activity.get('SUBJECT', '')):
                communications = activity.get('COMMUNICATIONS', [])
                for comm in communications:
                    if comm.get('TYPE') == 'IM' and 'imol|' in comm.get('VALUE', ''):
                        # This is an Open Channel conversation reference
                        messages = self.get_open_channel_messages(comm['VALUE'])
                        if messages:
                            open_channel_messages.append({
                                'activity_id': activity.get('ID'),
                                'activity_subject': activity.get('SUBJECT'),
                                'activity_date': activity.get('CREATED'),
                                'communication_value': comm['VALUE'],
                                'messages': messages,
                                'customer_info': comm.get('ENTITY_SETTINGS', {})
                            })
        
        conversation_history = {
            'lead_id': lead_id,
            'activities': activities,
            'timeline_comments': comments_response.get('result', []),
            'open_channel_messages': open_channel_messages,
            'total_activities': len(activities),
            'total_comments': len(comments_response.get('result', [])),
            'total_open_channel_conversations': len(open_channel_messages)
        }
        
        return conversation_history


if __name__ == "__main__":
    """Run Bitrix connection test when executed directly."""
    print("🔧 Testing Bitrix Integration...\n")
    
    try:
        # Initialize client
        client = BitrixClient()
        
        print("1. Testing connection...")
        connection_result = client.test_connection()
        
        if connection_result:
            print("\n2. Getting user details...")
            user_info = client.get_user_info()
            result = user_info.get('result', {})
            
            print(f"   User ID: {result.get('ID', 'Unknown')}")
            print(f"   Name: {result.get('NAME', 'Unknown')}")
            print(f"   Last Name: {result.get('LAST_NAME', 'Unknown')}")
            print(f"   Admin: {result.get('ADMIN', False)}")
            print(f"   Time Zone: {result.get('TIME_ZONE', 'Unknown')}")
            
            print("\n3. Testing CRM data retrieval...")
            
            # Test leads
            print("   Testing leads access...")
            try:
                leads = client.get_leads()
                print(f"   ✅ Leads: Found {len(leads)} leads")
                if leads:
                    print(f"      Sample lead: ID={leads[0].get('ID')}, Title={leads[0].get('TITLE', 'N/A')}")
            except Exception as e:
                print(f"   ❌ Leads access failed: {str(e)}")
            
            # Test deals
            print("   Testing deals access...")
            try:
                deals = client.get_deals()
                print(f"   ✅ Deals: Found {len(deals)} deals")
                if deals:
                    print(f"      Sample deal: ID={deals[0].get('ID')}, Title={deals[0].get('TITLE', 'N/A')}")
            except Exception as e:
                print(f"   ❌ Deals access failed: {str(e)}")
            
            print("\n4. Testing lead creation and update...")
            
            # Test lead creation
            print("   Testing lead creation...")
            try:
                test_lead_data = {
                    'TITLE': 'Test Lead - API Integration',
                    'NAME': 'Test',
                    'LAST_NAME': 'User',
                    'STATUS_ID': 'NEW',
                    'PHONE': [{'VALUE': '+1234567890', 'VALUE_TYPE': 'WORK'}],
                    'EMAIL': [{'VALUE': 'test@example.com', 'VALUE_TYPE': 'WORK'}],
                    'COMMENTS': 'Test lead created via API integration test',
                    'SOURCE_ID': 'OTHER'
                }
                
                create_response = client.create_lead(test_lead_data)
                if 'result' in create_response:
                    new_lead_id = create_response['result']
                    print(f"   ✅ Lead created successfully! ID: {new_lead_id}")
                    
                    # Test lead update
                    print("   Testing lead update...")
                    update_data = {
                        'STATUS_ID': 'IN_PROCESS',
                        'COMMENTS': 'Lead updated via API - test successful',
                        'OPPORTUNITY': 1000
                    }
                    
                    update_response = client.update_lead(new_lead_id, update_data)
                    if 'result' in update_response and update_response['result']:
                        print(f"   ✅ Lead updated successfully!")
                        
                        # Get the updated lead to verify changes
                        updated_lead = client.get_lead_by_id(new_lead_id)
                        if 'result' in updated_lead:
                            lead_info = updated_lead['result']
                            print(f"      Updated lead status: {lead_info.get('STATUS_ID')}")
                            print(f"      Updated opportunity: {lead_info.get('OPPORTUNITY')}")
                    else:
                        print(f"   ❌ Lead update failed: {update_response}")
                else:
                    print(f"   ❌ Lead creation failed: {create_response}")
                    
            except Exception as e:
                print(f"   ❌ Lead creation/update test failed: {str(e)}")
            
            print("\n5. Testing conversation history functionality...")
            
            # Test conversation history with existing leads
            print("   Testing conversation history retrieval...")
            try:
                if leads:  # Use leads from earlier test
                    test_lead = leads[0]
                    test_lead_id = test_lead['ID']
                    print(f"   Testing with Lead ID: {test_lead_id} - {test_lead.get('TITLE', 'N/A')}")
                    
                    # Get conversation history
                    conversation = client.get_lead_conversation_history(test_lead_id)
                    print(f"   ✅ Conversation history retrieved:")
                    print(f"      Total Activities: {conversation['total_activities']}")
                    print(f"      Total Timeline Comments: {conversation['total_comments']}")
                    
                    # Show some activity details if available
                    if conversation['activities']:
                        print("      Recent Activities:")
                        for i, activity in enumerate(conversation['activities'][:2]):  # Show first 2
                            activity_type = activity.get('TYPE_ID', 'Unknown')
                            subject = activity.get('SUBJECT', 'No subject')
                            created = activity.get('CREATED', 'Unknown date')
                            print(f"        {i+1}. Type: {activity_type}, Subject: {subject[:50]}...")
                    
                    # Show some timeline comments if available
                    if conversation['timeline_comments']:
                        print("      Recent Timeline Comments:")
                        for i, comment in enumerate(conversation['timeline_comments'][:2]):  # Show first 2
                            comment_text = comment.get('COMMENT', 'No comment text')
                            created = comment.get('CREATED', 'Unknown date')
                            print(f"        {i+1}. Created: {created}")
                            print(f"           Comment: {comment_text[:60]}...")
                else:
                    print("   ⚠️ No existing leads found to test conversation history")
                    
            except Exception as e:
                print(f"   ❌ Conversation history test failed: {str(e)}")
            
            print("\n✅ Bitrix integration testing complete!")
        else:
            print("\n❌ Connection test failed. Please check your webhook URL and permissions.")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        print("Please check your .env file and ensure BITRIX_WEBHOOK_URL is set correctly.")