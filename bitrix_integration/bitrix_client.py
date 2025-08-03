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
            
            print("\n✅ Bitrix integration testing complete!")
        else:
            print("\n❌ Connection test failed. Please check your webhook URL and permissions.")
            
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        print("Please check your .env file and ensure BITRIX_WEBHOOK_URL is set correctly.")