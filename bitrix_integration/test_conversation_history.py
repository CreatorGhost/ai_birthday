#!/usr/bin/env python3
"""
Test script to find leads with conversation history and save chat data to file.
This will help verify if we can properly fetch conversation history from Bitrix24.
"""

import os
import sys
import json
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

class Config:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

def format_date(date_string):
    """Format date string to readable format"""
    if not date_string:
        return "N/A"
    try:
        # Parse the date and format it
        date_obj = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return date_obj.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return date_string

def save_conversation_data(leads_with_history, filename="leads_conversation_history.json"):
    """Save conversation history data to JSON file"""
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "total_leads_with_history": len(leads_with_history),
        "leads": leads_with_history
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Conversation history saved to: {filename}")
    return filename

def main():
    # Use environment variable for webhook URL
    webhook_url = os.getenv('BITRIX_WEBHOOK_URL')
    if not webhook_url:
        print("❌ Error: BITRIX_WEBHOOK_URL environment variable not set")
        print("Please set it in the .env file")
        return
    
    # Create config and client
    config = Config(webhook_url)
    client = BitrixClient(config)
    
    print("🔍 Testing Bitrix24 connection...")
    
    # Test connection
    if client.test_connection():
        print("✅ Connected to Bitrix24 successfully!")
        
        # Try to get user info for additional details
        try:
            user_info = client.get_user_info()
            if user_info and 'result' in user_info:
                result = user_info['result']
                print(f"   User: {result.get('NAME', 'Unknown')} {result.get('LAST_NAME', '')}")
        except:
            print("   (User details not available, but connection works)")
    else:
        print("❌ Failed to connect to Bitrix24")
        return
    
    print("\n🔍 Searching for leads with conversation history...")
    
    # Get all leads
    leads = client.get_leads()
    if not leads:
        print("❌ No leads found")
        return
    
    print(f"📊 Found {len(leads)} total leads. Checking for conversation history...")
    
    leads_with_history = []
    checked_count = 0
    
    # Check each lead for conversation history (limit to first 50 for performance)
    for lead in leads[:50]:
        checked_count += 1
        lead_id = lead.get('ID')
        lead_title = lead.get('TITLE', 'Untitled')
        
        print(f"\r🔍 Checking lead {checked_count}/50: ID {lead_id} - {lead_title[:30]}...", end="", flush=True)
        
        # Get conversation history for this lead
        conversation = client.get_lead_conversation_history(lead_id)
        
        if conversation and (conversation['total_activities'] > 0 or conversation['total_comments'] > 0):
            # This lead has conversation history
            lead_data = {
                "lead_info": {
                    "id": lead_id,
                    "title": lead_title,
                    "name": lead.get('NAME', ''),
                    "status": lead.get('STATUS_ID', ''),
                    "created_date": format_date(lead.get('DATE_CREATE', ''))
                },
                "conversation_summary": {
                    "total_activities": conversation['total_activities'],
                    "total_comments": conversation['total_comments']
                },
                "activities": [],
                "timeline_comments": []
            }
            
            # Process activities (focus on communication)
            for activity in conversation['activities']:
                activity_data = {
                    "id": activity.get('ID'),
                    "type_id": activity.get('TYPE_ID'),
                    "type_name": {
                        '1': 'Call',
                        '2': 'Meeting', 
                        '3': 'Task',
                        '4': 'Email',
                        '5': 'SMS'
                    }.get(str(activity.get('TYPE_ID', '')), 'Other'),
                    "subject": activity.get('SUBJECT', ''),
                    "description": activity.get('DESCRIPTION', ''),
                    "created": format_date(activity.get('CREATED', '')),
                    "author_id": activity.get('AUTHOR_ID', ''),
                    "communications": activity.get('COMMUNICATIONS', [])
                }
                lead_data["activities"].append(activity_data)
            
            # Process timeline comments
            for comment in conversation['timeline_comments']:
                comment_data = {
                    "id": comment.get('ID'),
                    "comment": comment.get('COMMENT', ''),
                    "created": format_date(comment.get('CREATED', '')),
                    "author_id": comment.get('AUTHOR_ID', ''),
                    "files": comment.get('FILES', [])
                }
                lead_data["timeline_comments"].append(comment_data)
            
            leads_with_history.append(lead_data)
            
            # Stop after finding 3 leads with history
            if len(leads_with_history) >= 3:
                break
    
    print(f"\n\n📋 Results Summary:")
    print(f"   • Checked: {checked_count} leads")
    print(f"   • Found with history: {len(leads_with_history)} leads")
    
    if leads_with_history:
        print("\n🎯 Leads with conversation history:")
        for i, lead_data in enumerate(leads_with_history, 1):
            lead_info = lead_data['lead_info']
            summary = lead_data['conversation_summary']
            print(f"\n   {i}. Lead ID {lead_info['id']}: {lead_info['title']}")
            print(f"      📞 Activities: {summary['total_activities']}")
            print(f"      💬 Comments: {summary['total_comments']}")
            
            # Show sample chat content
            if lead_data['activities']:
                print(f"      📋 Sample Activity: {lead_data['activities'][0].get('subject', 'No subject')[:50]}...")
            if lead_data['timeline_comments']:
                print(f"      💭 Sample Comment: {lead_data['timeline_comments'][0].get('comment', 'No comment')[:50]}...")
        
        # Save to file
        filename = save_conversation_data(leads_with_history)
        
        print(f"\n✅ Successfully found and saved conversation history for {len(leads_with_history)} leads!")
        print(f"📁 Data saved to: {filename}")
        print("\n🔍 You can now examine the JSON file to see the complete chat/conversation data.")
        
    else:
        print("\n⚠️  No leads with conversation history found in the checked leads.")
        print("   This could mean:")
        print("   • The leads are new and haven't had activities yet")
        print("   • The conversation history is stored differently")
        print("   • We need to check more leads (currently limited to first 50)")

if __name__ == "__main__":
    main()