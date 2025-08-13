#!/usr/bin/env python3
"""
Bitrix24 Lead Conversation History Example

This script demonstrates how to retrieve and display conversation history
for existing leads in Bitrix24, including activities and timeline comments.
"""

import os
import sys
from datetime import datetime

# Add the current directory to Python path to import bitrix_client
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from bitrix_client import BitrixClient
except ImportError:
    # Alternative import method if the above fails
    import importlib.util
    spec = importlib.util.spec_from_file_location("bitrix_client", 
                                                  os.path.join(os.path.dirname(__file__), "bitrix_client.py"))
    bitrix_client_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bitrix_client_module)
    BitrixClient = bitrix_client_module.BitrixClient

def format_date(date_string):
    """
    Format Bitrix24 date string to readable format
    """
    try:
        if date_string:
            # Bitrix24 typically returns dates in ISO format
            dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        pass
    return date_string or 'Unknown date'

def get_activity_type_name(type_id):
    """
    Convert activity type ID to human-readable name
    """
    activity_types = {
        '1': 'Call',
        '2': 'Meeting', 
        '3': 'Task',
        '4': 'Email',
        '5': 'SMS',
        '6': 'Request',
        '7': 'Provider',
        '8': 'Rest App'
    }
    return activity_types.get(str(type_id), f'Type {type_id}')

def display_lead_activities(activities):
    """
    Display lead activities in a formatted way
    """
    if not activities:
        print("    No activities found.")
        return
    
    print(f"    Found {len(activities)} activities:")
    print("    " + "="*80)
    
    for i, activity in enumerate(activities[:10], 1):  # Show up to 10 activities
        activity_type = get_activity_type_name(activity.get('TYPE_ID'))
        subject = activity.get('SUBJECT', 'No subject')
        description = activity.get('DESCRIPTION', '')
        created = format_date(activity.get('CREATED'))
        completed = activity.get('COMPLETED', 'N') == 'Y'
        direction = 'Incoming' if activity.get('DIRECTION') == '1' else 'Outgoing'
        
        print(f"    {i}. [{activity_type}] {subject}")
        print(f"       Created: {created} | Direction: {direction} | Completed: {'Yes' if completed else 'No'}")
        
        if description:
            # Truncate long descriptions
            desc_preview = description[:100] + '...' if len(description) > 100 else description
            print(f"       Description: {desc_preview}")
        
        # Show communications if available
        communications = activity.get('COMMUNICATIONS', [])
        if communications:
            print(f"       Communications: {len(communications)} entries")
            for comm in communications[:2]:  # Show first 2 communications
                comm_type = comm.get('TYPE', 'Unknown')
                comm_value = comm.get('VALUE', 'No value')
                print(f"         - {comm_type}: {comm_value}")
        
        print()

def display_timeline_comments(comments):
    """
    Display timeline comments in a formatted way
    """
    if not comments:
        print("    No timeline comments found.")
        return
    
    print(f"    Found {len(comments)} timeline comments:")
    print("    " + "="*80)
    
    for i, comment in enumerate(comments[:10], 1):  # Show up to 10 comments
        comment_text = comment.get('COMMENT', 'No comment text')
        created = format_date(comment.get('CREATED'))
        author_id = comment.get('AUTHOR_ID', 'Unknown')
        
        print(f"    {i}. Author ID: {author_id} | Created: {created}")
        
        # Format comment text
        if comment_text:
            # Remove HTML tags if present and truncate
            import re
            clean_text = re.sub(r'<[^>]+>', '', comment_text)
            text_preview = clean_text[:200] + '...' if len(clean_text) > 200 else clean_text
            print(f"       Comment: {text_preview}")
        
        # Show files if available
        files = comment.get('FILES', [])
        if files:
            print(f"       Attachments: {len(files)} files")
        
        print()

def test_conversation_history_for_lead(client, lead_id, lead_title):
    """
    Test conversation history retrieval for a specific lead
    """
    print(f"\n{'='*100}")
    print(f"CONVERSATION HISTORY FOR LEAD ID: {lead_id}")
    print(f"Lead Title: {lead_title}")
    print(f"{'='*100}")
    
    try:
        # Get complete conversation history
        conversation = client.get_lead_conversation_history(lead_id)
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total Activities: {conversation['total_activities']}")
        print(f"   Total Timeline Comments: {conversation['total_comments']}")
        
        # Display activities
        print(f"\n📞 ACTIVITIES:")
        display_lead_activities(conversation['activities'])
        
        # Display timeline comments
        print(f"\n💬 TIMELINE COMMENTS:")
        display_timeline_comments(conversation['timeline_comments'])
        
        return True
        
    except Exception as e:
        print(f"❌ Error retrieving conversation history: {str(e)}")
        return False

def main():
    """
    Main function to demonstrate conversation history functionality
    """
    print("🚀 Bitrix24 Lead Conversation History Demo")
    print("="*60)
    
    # Initialize client
    webhook_url = os.getenv('BITRIX_WEBHOOK_URL')
    if not webhook_url:
        print("❌ Error: BITRIX_WEBHOOK_URL environment variable not set")
        print("Please set it using: export BITRIX_WEBHOOK_URL='your_webhook_url'")
        return
    
    # Create a simple config object
    class Config:
        def __init__(self, webhook_url):
            self.webhook_url = webhook_url
    
    config = Config(webhook_url)
    client = BitrixClient(config)
    
    # Test connection
    print("\n🔗 Testing connection...")
    if not client.test_connection():
        print("❌ Failed to connect to Bitrix24")
        return
    
    print("✅ Connected to Bitrix24 successfully")
    
    # Get existing leads
    print("\n📋 Retrieving existing leads...")
    try:
        leads = client.get_leads()
        
        if not leads:
            print("⚠️ No leads found in your Bitrix24 account")
            return
        
        print(f"✅ Found {len(leads)} leads")
        
        # Test conversation history for multiple leads
        test_leads = leads[:3]  # Test with first 3 leads
        
        for lead in test_leads:
            lead_id = lead['ID']
            lead_title = lead.get('TITLE', 'Untitled Lead')
            
            success = test_conversation_history_for_lead(client, lead_id, lead_title)
            if not success:
                continue
        
        print(f"\n✅ Conversation history demo completed!")
        print(f"\n💡 TIP: Older leads typically have more conversation history.")
        print(f"   New leads may not have activities or comments yet.")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")

if __name__ == "__main__":
    main()