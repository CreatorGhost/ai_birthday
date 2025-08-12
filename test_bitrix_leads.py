#!/usr/bin/env python3
"""
Test script to fetch latest leads from Bitrix CRM sorted by creation date
"""

import json
from datetime import datetime, timedelta
from bitrix_integration.bitrix_client import BitrixClient


def main():
    """Fetch and display the latest leads from Bitrix CRM."""
    print("🔄 Testing Bitrix latest leads retrieval...\n")

    try:
        # Initialize client
        client = BitrixClient()

        # Test connection first
        print("1. Testing connection...")
        if not client.test_connection():
            print("❌ Connection failed. Please check your webhook URL.")
            return

        print("\n2. Fetching latest leads (sorted by creation date)...")

        # Get recent leads with date filter (last 30 days) and sorting
        recent_date = datetime.now() - timedelta(days=30)
        date_filter = recent_date.strftime("%Y-%m-%d")

        # Fetch leads with order by date (newest first)
        params = {
            'select': ['ID', 'TITLE', 'NAME', 'LAST_NAME', 'STATUS_ID', 
                      'DATE_CREATE', 'DATE_MODIFY', 'PHONE', 'EMAIL'],
            'order': {'DATE_CREATE': 'DESC'},  # Sort by creation date, newest first
            'filter': {'>DATE_CREATE': date_filter},  # Only leads from last 30 days
            'start': 0
        }

        print(f"   Filtering leads created after: {date_filter}")
        response = client._make_request('crm.lead.list', params)
        leads = response.get('result', [])
        print(f"✅ Successfully fetched {len(leads)} recent leads\n")

        if leads:
            print("📋 Latest Lead Details (sorted by newest first):")
            print("-" * 80)

            # Show latest leads
            for i, lead in enumerate(leads[:10]):  # Show first 10 leads
                print(f"Lead #{i+1}:")
                print(f"  ID: {lead.get('ID', 'N/A')}")
                print(f"  Title: {lead.get('TITLE', 'N/A')}")
                print(f"  Name: {lead.get('NAME', 'N/A')} {lead.get('LAST_NAME', '')}")
                print(f"  Status: {lead.get('STATUS_ID', 'N/A')}")
                print(f"  Created: {lead.get('DATE_CREATE', 'N/A')}")
                print(f"  Modified: {lead.get('DATE_MODIFY', 'N/A')}")
                phone_val = 'N/A'
                if lead.get('PHONE') and isinstance(lead['PHONE'], list) and lead['PHONE']:
                    phone_val = lead['PHONE'][0].get('VALUE', 'N/A')
                print(f"  Phone: {phone_val}")
                email_val = 'N/A'
                if lead.get('EMAIL') and isinstance(lead['EMAIL'], list) and lead['EMAIL']:
                    email_val = lead['EMAIL'][0].get('VALUE', 'N/A')
                print(f"  Email: {email_val}")
                print()

            if len(leads) > 10:
                print(f"... and {len(leads) - 10} more recent leads")

            # Save to JSON for further inspection
            print("💾 Saving recent leads to 'bitrix_recent_leads.json'...")
            with open('bitrix_recent_leads.json', 'w', encoding='utf-8') as f:
                json.dump(leads, f, indent=2, ensure_ascii=False, default=str)
            print("✅ Recent leads saved successfully!")

            # Also try to get the very latest leads from 2025
            print("\n3. Fetching leads from 2025 specifically...")
            params_2025 = {
                'select': ['ID', 'TITLE', 'NAME', 'LAST_NAME', 'STATUS_ID',
                          'DATE_CREATE', 'DATE_MODIFY', 'PHONE', 'EMAIL'],
                'order': {'DATE_CREATE': 'DESC'},
                'filter': {'>DATE_CREATE': '2025-01-01'},
                'start': 0
            }

            response_2025 = client._make_request('crm.lead.list', params_2025)
            leads_2025 = response_2025.get('result', [])

            if leads_2025:
                print(f"✅ Found {len(leads_2025)} leads from 2025:")
                for i, lead in enumerate(leads_2025[:5]):
                    print(f"  {i+1}. ID: {lead.get('ID')}, Title: {lead.get('TITLE', 'N/A')}, Created: {lead.get('DATE_CREATE')}")
            else:
                print("ℹ️ No leads found from 2025")

        else:
            print(f"ℹ️ No recent leads found (last 30 days)")
            print("Let's try fetching all leads and sort them by date...")

            # Fallback: get all leads and sort by date
            all_leads = client.get_leads(
                select_fields=['ID', 'TITLE', 'NAME', 'LAST_NAME', 'STATUS_ID',
                              'DATE_CREATE', 'PHONE', 'EMAIL']
            )
            if all_leads:
                # Sort by date (newest first)
                sorted_leads = sorted(all_leads, 
                                    key=lambda x: x.get('DATE_CREATE', ''), 
                                    reverse=True)
                print(f"✅ Found {len(sorted_leads)} total leads")
                print("📋 Latest 5 leads from all data:")
                for i, lead in enumerate(sorted_leads[:5]):
                    print(f"  {i+1}. ID: {lead.get('ID')}, Created: {lead.get('DATE_CREATE')}")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("\nTroubleshooting tips:")
        print("- Check if BITRIX_WEBHOOK_URL is correct in .env file")
        print("- Verify webhook has CRM read permissions")
        print("- Make sure your Bitrix24 account is active")


if __name__ == "__main__":
    main()