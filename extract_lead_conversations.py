#!/usr/bin/env python3
"""
Extract all lead conversations from Bitrix for RAG analysis.
Gets conversations from all lead statuses for the last year and formats for RAG ingestion.
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
from bitrix_integration.bitrix_client import BitrixClient
from bitrix_integration.config import BitrixConfig

def get_all_leads_last_year(client):
    """Get all leads from the last year across all statuses"""
    print("📅 Fetching all leads from last year...")
    
    # Calculate date filter (last year)
    one_year_ago = datetime.now() - timedelta(days=365)
    date_filter = one_year_ago.strftime('%Y-%m-%d')
    
    # Get all leads with comprehensive field selection
    filter_params = {
        '>=DATE_CREATE': date_filter
    }
    
    select_fields = [
        'ID', 'TITLE', 'NAME', 'LAST_NAME', 'STATUS_ID', 'DATE_CREATE', 
        'DATE_MODIFY', 'CREATED_BY_ID', 'ASSIGNED_BY_ID', 'COMMENTS',
        'PHONE', 'EMAIL', 'SOURCE_ID', 'OPPORTUNITY', 'CURRENCY_ID'
    ]
    
    leads = client.get_leads(select_fields, filter_params)
    print(f"Found {len(leads)} leads from last year")
    
    return leads

def extract_lead_conversation(client, lead_id, lead_info):
    """Extract complete conversation history for a lead"""
    print(f"  Getting conversation for lead {lead_id}...")
    
    try:
        # Get conversation history (activities + timeline comments)
        conversation = client.get_lead_conversation_history(lead_id)
        
        # Format conversation for RAG
        formatted_conversation = {
            'lead_metadata': {
                'lead_id': lead_id,
                'title': lead_info.get('TITLE', ''),
                'customer_name': f"{lead_info.get('NAME', '')} {lead_info.get('LAST_NAME', '')}".strip(),
                'status_id': lead_info.get('STATUS_ID', ''),
                'created_date': lead_info.get('DATE_CREATE', ''),
                'last_modified': lead_info.get('DATE_MODIFY', ''),
                'opportunity_value': lead_info.get('OPPORTUNITY', 0),
                'source': lead_info.get('SOURCE_ID', ''),
                'phone': extract_phone_number(lead_info.get('PHONE', [])),
                'email': extract_email(lead_info.get('EMAIL', [])),
                'initial_comments': lead_info.get('COMMENTS', '')
            },
            'conversation_timeline': [],
            'activities_summary': {
                'total_activities': conversation.get('total_activities', 0),
                'total_timeline_comments': conversation.get('total_comments', 0),
                'total_open_channel_conversations': conversation.get('total_open_channel_conversations', 0)
            }
        }
        
        # Process activities (calls, emails, meetings, etc.)
        activities = conversation.get('activities', [])
        timeline_comments = conversation.get('timeline_comments', [])
        
        # Combine and sort all interactions by date
        all_interactions = []
        
        # Add activities
        for activity in activities:
            all_interactions.append({
                'type': 'activity',
                'date': activity.get('CREATED', ''),
                'activity_type': get_activity_type_name(activity.get('TYPE_ID')),
                'subject': activity.get('SUBJECT', ''),
                'description': activity.get('DESCRIPTION', ''),
                'author_id': activity.get('RESPONSIBLE_ID', ''),
                'direction': activity.get('DIRECTION', ''),
                'result': activity.get('RESULT', ''),
                'raw_data': activity
            })
        
        # Add timeline comments
        for comment in timeline_comments:
            all_interactions.append({
                'type': 'timeline_comment',
                'date': comment.get('CREATED', ''),
                'comment': comment.get('COMMENT', ''),
                'author_id': comment.get('AUTHOR_ID', ''),
                'files': comment.get('FILES', []),
                'raw_data': comment
            })
        
        # Sort by date
        all_interactions.sort(key=lambda x: x.get('date', ''))
        
        # Format for RAG consumption
        conversation_text = format_conversation_for_rag(formatted_conversation['lead_metadata'], all_interactions)
        
        formatted_conversation['conversation_timeline'] = all_interactions
        formatted_conversation['rag_formatted_text'] = conversation_text
        
        return formatted_conversation
        
    except Exception as e:
        print(f"    ❌ Error extracting conversation for lead {lead_id}: {e}")
        return None

def extract_phone_number(phone_array):
    """Extract phone number from Bitrix phone array format"""
    if isinstance(phone_array, list) and phone_array:
        return phone_array[0].get('VALUE', '') if isinstance(phone_array[0], dict) else str(phone_array[0])
    return ''

def extract_email(email_array):
    """Extract email from Bitrix email array format"""
    if isinstance(email_array, list) and email_array:
        return email_array[0].get('VALUE', '') if isinstance(email_array[0], dict) else str(email_array[0])
    return ''

def get_activity_type_name(type_id):
    """Convert activity type ID to readable name"""
    activity_types = {
        '1': 'Call',
        '2': 'Meeting', 
        '3': 'Task',
        '4': 'Email',
        '5': 'Note',
        '6': 'SMS',
        '7': 'File',
        '8': 'Request',
        '9': 'Webform',
        '10': 'Invoice',
        '11': 'Quote'
    }
    return activity_types.get(str(type_id), f'Activity_{type_id}')

def format_conversation_for_rag(lead_metadata, interactions):
    """Format conversation in a RAG-friendly text format"""
    
    # Build comprehensive text for RAG embedding and search
    rag_text_parts = []
    
    # Lead header with metadata
    rag_text_parts.append("=== LEAD CONVERSATION RECORD ===")
    rag_text_parts.append(f"Lead ID: {lead_metadata['lead_id']}")
    rag_text_parts.append(f"Customer: {lead_metadata['customer_name']}")
    rag_text_parts.append(f"Status: {lead_metadata['status_id']}")
    rag_text_parts.append(f"Opportunity Value: ${lead_metadata['opportunity_value']}")
    rag_text_parts.append(f"Created: {lead_metadata['created_date']}")
    rag_text_parts.append(f"Phone: {lead_metadata['phone']}")
    rag_text_parts.append(f"Email: {lead_metadata['email']}")
    rag_text_parts.append(f"Source: {lead_metadata['source']}")
    
    if lead_metadata['initial_comments']:
        rag_text_parts.append(f"Initial Comments: {lead_metadata['initial_comments']}")
    
    rag_text_parts.append("\n=== CONVERSATION TIMELINE ===")
    
    # Format each interaction
    if not interactions:
        rag_text_parts.append("No conversation history available.")
    else:
        for i, interaction in enumerate(interactions, 1):
            interaction_date = interaction.get('date', 'Unknown date')
            
            if interaction['type'] == 'activity':
                activity_type = interaction.get('activity_type', 'Unknown')
                subject = interaction.get('subject', '')
                description = interaction.get('description', '')
                direction = interaction.get('direction', '')
                result = interaction.get('result', '')
                
                rag_text_parts.append(f"\n[{i}] {interaction_date} - {activity_type}")
                if direction:
                    rag_text_parts.append(f"Direction: {direction}")
                if subject:
                    rag_text_parts.append(f"Subject: {subject}")
                if description:
                    rag_text_parts.append(f"Content: {description}")
                if result:
                    rag_text_parts.append(f"Result: {result}")
                    
            elif interaction['type'] == 'timeline_comment':
                comment = interaction.get('comment', '')
                files = interaction.get('files', [])
                
                rag_text_parts.append(f"\n[{i}] {interaction_date} - Timeline Comment")
                if comment:
                    rag_text_parts.append(f"Comment: {comment}")
                if files:
                    rag_text_parts.append(f"Files: {len(files)} attached")
                    
            elif interaction['type'] == 'open_channel_message':
                message_text = interaction.get('message_text', '')
                sender = interaction.get('sender', 'Unknown')
                channel_type = interaction.get('channel_type', 'Open Channel')
                customer_name = interaction.get('customer_name', '')
                
                rag_text_parts.append(f"\n[{i}] {interaction_date} - {channel_type} Message")
                sender_name = customer_name if customer_name else f"User {sender}"
                rag_text_parts.append(f"From: {sender_name}")
                if message_text:
                    rag_text_parts.append(f"Message: {message_text}")
                    
            elif interaction['type'] == 'open_channel_reference':
                subject = interaction.get('subject', '')
                channel_type = interaction.get('channel_type', 'Open Channel')
                customer_name = interaction.get('customer_name', '')
                communication_ref = interaction.get('communication_ref', '')
                
                rag_text_parts.append(f"\n[{i}] {interaction_date} - {channel_type} Conversation")
                if customer_name:
                    rag_text_parts.append(f"Customer: {customer_name}")
                if subject:
                    rag_text_parts.append(f"Subject: {subject}")
                rag_text_parts.append(f"Note: Full conversation details could not be retrieved (Ref: {communication_ref})")
                rag_text_parts.append(f"This indicates a {channel_type} conversation occurred but message content is not accessible via current API methods.")
    
    rag_text_parts.append("\n=== END CONVERSATION RECORD ===\n")
    
    return "\n".join(rag_text_parts)

def save_conversations_for_rag(all_conversations):
    """Save all conversations in multiple RAG-friendly formats"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directory
    output_dir = 'conversation_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save as individual RAG text files (for vector embedding)
    rag_texts_dir = f"{output_dir}/rag_texts_{timestamp}"
    os.makedirs(rag_texts_dir, exist_ok=True)
    
    valid_conversations = 0
    
    for lead_id, conversation in all_conversations.items():
        if conversation and conversation.get('rag_formatted_text'):
            # Save individual RAG text file
            filename = f"lead_{lead_id}_conversation.txt"
            filepath = f"{rag_texts_dir}/{filename}"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(conversation['rag_formatted_text'])
            
            valid_conversations += 1
    
    # 2. Save as single combined RAG corpus
    combined_rag_file = f"{output_dir}/all_conversations_rag_corpus_{timestamp}.txt"
    with open(combined_rag_file, 'w', encoding='utf-8') as f:
        f.write("LEO & LOONA LEAD CONVERSATIONS CORPUS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Conversations: {valid_conversations}\n")
        f.write("=" * 50 + "\n\n")
        
        for lead_id, conversation in all_conversations.items():
            if conversation and conversation.get('rag_formatted_text'):
                f.write(conversation['rag_formatted_text'])
                f.write("\n" + "="*100 + "\n\n")
    
    # 3. Save as structured JSON for analysis
    json_file = f"{output_dir}/conversations_structured_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, indent=2, ensure_ascii=False, default=str)
    
    # 4. Save summary report
    summary_file = f"{output_dir}/extraction_summary_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("CONVERSATION EXTRACTION SUMMARY\n")
        f.write("=" * 40 + "\n")
        f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Leads Processed: {len(all_conversations)}\n")
        f.write(f"Leads with Conversations: {valid_conversations}\n")
        f.write(f"Leads without Conversations: {len(all_conversations) - valid_conversations}\n\n")
        
        # Status breakdown
        status_counts = {}
        for conversation in all_conversations.values():
            if conversation:
                status = conversation.get('lead_metadata', {}).get('status_id', 'Unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
        
        f.write("LEADS BY STATUS:\n")
        for status, count in sorted(status_counts.items()):
            f.write(f"  {status}: {count} leads\n")
    
    return {
        'rag_texts_directory': rag_texts_dir,
        'combined_corpus_file': combined_rag_file,
        'structured_json_file': json_file,
        'summary_file': summary_file,
        'total_conversations': valid_conversations
    }

def main():
    """Main extraction function"""
    print("🗂️ Leo & Loona Lead Conversation Extractor")
    print("=" * 60)
    print("Extracting all lead conversations for RAG analysis\n")
    
    try:
        # Initialize Bitrix client
        print("1. Connecting to Bitrix...")
        config = BitrixConfig()
        client = BitrixClient(config)
        
        if not client.test_connection():
            print("❌ Bitrix connection failed")
            return
        
        print("✅ Connected to Bitrix successfully")
        
        # Get all leads from last year
        print("\n2. Fetching leads from last year...")
        all_leads = get_all_leads_last_year(client)
        
        if not all_leads:
            print("❌ No leads found for the last year")
            return
        
        # Group leads by status for reporting
        status_groups = {}
        for lead in all_leads:
            status = lead.get('STATUS_ID', 'Unknown')
            if status not in status_groups:
                status_groups[status] = []
            status_groups[status].append(lead)
        
        print(f"✅ Found {len(all_leads)} leads across {len(status_groups)} different statuses")
        print("\nLead distribution by status:")
        for status, leads in status_groups.items():
            print(f"  • {status}: {len(leads)} leads")
        
        # Extract conversations
        print(f"\n3. Extracting conversations from {len(all_leads)} leads...")
        print("This may take a while...")
        
        all_conversations = {}
        processed_count = 0
        successful_extractions = 0
        
        for i, lead in enumerate(all_leads, 1):
            lead_id = lead['ID']
            
            if i % 10 == 0 or i == len(all_leads):
                print(f"  Progress: {i}/{len(all_leads)} leads processed")
            
            # Extract conversation for this lead
            conversation = extract_lead_conversation(client, lead_id, lead)
            
            if conversation:
                all_conversations[lead_id] = conversation
                successful_extractions += 1
            else:
                all_conversations[lead_id] = None
            
            processed_count += 1
        
        print(f"\n✅ Extraction complete!")
        print(f"  Total leads processed: {processed_count}")
        print(f"  Successful extractions: {successful_extractions}")
        print(f"  Leads without conversations: {processed_count - successful_extractions}")
        
        # Save in RAG-friendly formats
        print(f"\n4. Saving conversations for RAG analysis...")
        output_info = save_conversations_for_rag(all_conversations)
        
        print(f"\n🎉 Conversation extraction complete!")
        print(f"\n📁 Output files created:")
        print(f"  • RAG texts directory: {output_info['rag_texts_directory']}")
        print(f"  • Combined corpus: {output_info['combined_corpus_file']}")
        print(f"  • Structured JSON: {output_info['structured_json_file']}")
        print(f"  • Summary report: {output_info['summary_file']}")
        print(f"\n💾 Total conversations ready for RAG: {output_info['total_conversations']}")
        
        print(f"\n🔍 Next Steps for RAG Integration:")
        print(f"1. Use the combined corpus file for RAG vector embedding")
        print(f"2. Individual text files can be used for fine-grained analysis")
        print(f"3. JSON file contains full structured data for advanced processing")
        print(f"4. Query the RAG system to analyze conversion patterns")
        
        return output_info
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return None

if __name__ == "__main__":
    main()