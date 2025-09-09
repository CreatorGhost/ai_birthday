#!/usr/bin/env python3
"""
Cleanup My Leads - Simple utility to delete AI Chatbot leads from Bitrix CRM

This script fetches all leads assigned to the AI Chatbot and asks for confirmation to delete them all.
Use with caution - this will permanently delete leads from Bitrix!

Usage:
    python cleanup_my_leads.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from bitrix_integration.lead_manager import LeadManager
from bitrix_integration.bitrix_client import BitrixClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LeadCleanup:
    def __init__(self):
        self.lead_manager = LeadManager()
        self.bitrix_client = BitrixClient()
    
    def get_test_leads(self):
        """Get all AI Chatbot leads"""
        
        try:
            # Get all leads assigned to AI Chatbot
            filter_params = {
                'ASSIGNED_BY_ID': '38005'  # AI Chatbot Assistant
            }
            
            response = self.bitrix_client._make_request(
                'crm.lead.list',
                {
                    'select': ['ID', 'TITLE', 'NAME', 'PHONE', 'DATE_CREATE', 'ASSIGNED_BY_ID'],
                    'filter': filter_params,
                    'order': {'DATE_CREATE': 'DESC'}
                }
            )
            
            if not response or 'result' not in response:
                print("❌ Failed to fetch leads from Bitrix")
                return []
            
            all_leads = response['result']
            test_leads = []
            
            for lead in all_leads:
                lead_id = lead.get('ID')
                title = lead.get('TITLE', '')
                name = lead.get('NAME', '')
                phone_data = lead.get('PHONE', [])
                date_create = lead.get('DATE_CREATE', '')
                assigned_by = lead.get('ASSIGNED_BY_ID', '')
                
                # Extract phone number
                phone = ''
                if phone_data and isinstance(phone_data, list) and len(phone_data) > 0 and phone_data[0]:
                    phone = phone_data[0].get('VALUE', '')
                
                # All leads returned are AI Chatbot leads, so add them all
                test_leads.append({
                    'id': lead_id,
                    'title': title,
                    'name': name,
                    'phone': phone,
                    'date_create': date_create,
                    'assigned_by': assigned_by,
                    'reasons': ['AI Chatbot Lead']
                })
            
            return test_leads
            
        except Exception as e:
            print(f"❌ Error fetching leads: {e}")
            return []
    
    def delete_leads(self, lead_ids):
        """Delete multiple leads from Bitrix"""
        deleted_count = 0
        failed_count = 0
        
        for lead_id in lead_ids:
            try:
                response = self.bitrix_client._make_request(
                    'crm.lead.delete',
                    {'id': lead_id}
                )
                
                if response and response.get('result'):
                    print(f"✅ Deleted lead {lead_id}")
                    deleted_count += 1
                else:
                    print(f"❌ Failed to delete lead {lead_id}")
                    failed_count += 1
                    
            except Exception as e:
                print(f"❌ Error deleting lead {lead_id}: {e}")
                failed_count += 1
        
        return deleted_count, failed_count
    
    def show_leads(self, leads, title="Found Leads"):
        """Display leads in a formatted table"""
        if not leads:
            print(f"\n📋 {title}: None found")
            return
        
        print(f"\n📋 {title}: {len(leads)} leads")
        print("-" * 120)
        print(f"{'ID':<8} {'Title':<25} {'Name':<15} {'Phone':<15} {'Date':<20} {'Reasons'}")
        print("-" * 120)
        
        for lead in leads:
            title = lead.get('title') or ''
            name = lead.get('name') or ''
            phone = lead.get('phone') or ''
            date_create = lead.get('date_create') or ''
            reasons = lead.get('reasons') or []
            
            print(f"{lead['id']:<8} {title[:24]:<25} {name[:14]:<15} "
                  f"{phone:<15} {date_create[:19]:<20} {', '.join(reasons[:2])}")
    
    def run_cleanup(self):
        """Main cleanup function"""
        
        print("🧹 Lead Cleanup Utility")
        print("=" * 30)
        
        # Get test leads
        print("🔍 Fetching AI Chatbot leads...")
        test_leads = self.get_test_leads()
        
        if not test_leads:
            print("✅ No leads found to clean up!")
            return
        
        # Show what we found
        self.show_leads(test_leads, "AI Chatbot Leads Found")
        
        # Simple confirmation
        print(f"\n⚠️  Found {len(test_leads)} leads to delete!")
        confirm = input("Delete all these leads? (y/n): ").lower().strip()
        
        if confirm != 'y':
            print("❌ Deletion cancelled")
            return
        
        # Delete the leads
        print("\n🗑️  Deleting leads...")
        lead_ids = [lead['id'] for lead in test_leads]
        deleted_count, failed_count = self.delete_leads(lead_ids)
        
        print(f"\n📊 Results:")
        print(f"✅ Deleted: {deleted_count} leads")
        if failed_count > 0:
            print(f"❌ Failed: {failed_count} leads")
        
        print("\n🎉 Done!")

def main():
    try:
        cleanup = LeadCleanup()
        cleanup.run_cleanup()
        
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

