import sys
from datetime import datetime, timedelta
from bitrix_integration.bitrix_client import BitrixClient
from bitrix_integration.config import BitrixConfig

# Your Bitrix user details
YOUR_USER_ID = '36207'
YOUR_EMAIL = 'adityapratap2307@gmail.com'

def find_my_leads(days_back=7):
    """Find all leads created by you in the last N days"""
    print(f"🔍 Finding Your Leads (Last {days_back} days)")
    print("=" * 40)
    
    try:
        config = BitrixConfig()
        client = BitrixClient(config)
        
        # Calculate date filter (last N days)
        cutoff_date = datetime.now() - timedelta(days=days_back)
        date_filter = cutoff_date.strftime('%Y-%m-%d')
        
        print(f"Searching for leads created by User ID: {YOUR_USER_ID}")
        print(f"Email: {YOUR_EMAIL}")
        print(f"Created after: {date_filter}")
        
        # Get leads with filter for your user ID and recent date
        filter_params = {
            'CREATED_BY_ID': YOUR_USER_ID,
            '>=DATE_CREATE': date_filter
        }
        
        print(f"\nFetching leads...")
        my_leads = client.get_leads([
            'ID', 'TITLE', 'NAME', 'LAST_NAME', 'DATE_CREATE', 
            'CREATED_BY_ID', 'COMMENTS', 'STATUS_ID'
        ], filter_params)
        
        print(f"Found {len(my_leads)} leads created by you")
        
        return my_leads, client
        
    except Exception as e:
        print(f"❌ Error finding your leads: {e}")
        return [], None

def show_my_leads(my_leads):
    """Display the leads that will be deleted"""
    print(f"\n📋 Your Leads to be Deleted:")
    print("-" * 60)
    
    for lead in my_leads:
        lead_id = lead['ID']
        title = lead.get('TITLE', 'No Title')
        name = lead.get('NAME', '')
        last_name = lead.get('LAST_NAME', '')
        date_created = lead.get('DATE_CREATE', '')
        status = lead.get('STATUS_ID', '')
        
        full_name = f"{name} {last_name}".strip() or "No Name"
        
        print(f"  • ID {lead_id}: {title}")
        print(f"    Name: {full_name}")
        print(f"    Created: {date_created}")
        print(f"    Status: {status}")
        print()

def delete_my_leads(my_leads, client):
    """Delete all your leads with simplified confirmation"""
    if not my_leads:
        print("✅ No leads found to delete!")
        return
    
    print(f"🗑️ Preparing to Delete {len(my_leads)} Leads")
    print("=" * 50)
    
    # Show what will be deleted
    show_my_leads(my_leads)
    
    # Single simplified confirmation
    print(f"⚠️  This will delete {len(my_leads)} leads created by {YOUR_EMAIL}")
    response = input(f"\nProceed with deletion? (y/N): ")
    
    if response.lower() not in ['y', 'yes']:
        print("❌ Deletion cancelled")
        return
    
    print(f"\n🗑️ Deleting Your Leads...")
    print("-" * 30)
    
    deleted_count = 0
    failed_count = 0
    
    for lead in my_leads:
        lead_id = lead['ID']
        title = lead.get('TITLE', 'No Title')[:50] + "..."
        
        try:
            print(f"Deleting {lead_id}: {title}", end=" ")
            
            response = client._make_request('crm.lead.delete', {
                'id': int(lead_id)
            })
            
            if response and 'result' in response:
                print("✅")
                deleted_count += 1
            else:
                print(f"❌ Failed: {response}")
                failed_count += 1
                
        except Exception as e:
            print(f"❌ Error: {e}")
            failed_count += 1
    
    print(f"\n📊 Cleanup Summary:")
    print(f"✅ Successfully deleted: {deleted_count}")
    print(f"❌ Failed to delete: {failed_count}")
    print(f"🧹 Your leads cleanup complete!")

def main():
    """Main function with simplified options"""
    print("🧹 Your Leads Cleanup Script")
    print(f"Safely delete leads created by: {YOUR_EMAIL}")
    print("=" * 60)
    
    # Simplified time range selection
    print("Select time range:")
    print("1. Last 1 day")
    print("2. Last 7 days (default)")
    print("3. Last 30 days")
    print("4. All time")
    
    choice = input("\nEnter choice (1-4) or press Enter for default: ").strip()
    
    days_map = {'1': 1, '2': 7, '3': 30, '4': 365}
    days_back = days_map.get(choice, 7)  # Default 7 days
    
    # Find your leads
    my_leads, client = find_my_leads(days_back)
    
    if not client:
        print("❌ Could not connect to Bitrix")
        return
    
    # Delete your leads
    delete_my_leads(my_leads, client)

if __name__ == "__main__":
    main()