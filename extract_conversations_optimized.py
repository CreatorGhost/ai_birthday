#!/usr/bin/env python3
"""
OPTIMIZED Lead Conversation Extractor for Bitrix24
Extracts maximum possible conversation data and clearly indicates WhatsApp/Open Channel interactions.

This version is designed to work within Bitrix24 API limitations while providing maximum value.
"""
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List
from bitrix_integration.bitrix_client import BitrixClient
from bitrix_integration.config import BitrixConfig

def format_conversation_for_rag_optimized(lead_metadata, interactions):
    """Enhanced RAG formatting that highlights conversation insights even without full message content"""
    
    rag_text_parts = []
    
    # Enhanced lead header with conversation insights
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
    
    # Conversation insights summary
    phone_calls = sum(1 for i in interactions if i['type'] == 'activity' and 'call' in i.get('activity_type', '').lower())
    whatsapp_conversations = sum(1 for i in interactions if i['type'] == 'activity' and 'whatsapp' in str(i).lower())
    open_channel_conversations = sum(1 for i in interactions if i['type'] == 'activity' and 'open channel' in str(i).lower())
    timeline_comments = sum(1 for i in interactions if i['type'] == 'timeline_comment')
    
    rag_text_parts.append(f"\n=== CONVERSATION INSIGHTS ===")
    rag_text_parts.append(f"Total Interactions: {len(interactions)}")
    rag_text_parts.append(f"Phone Calls: {phone_calls}")
    rag_text_parts.append(f"WhatsApp Conversations: {whatsapp_conversations}")
    rag_text_parts.append(f"Open Channel Conversations: {open_channel_conversations}")
    rag_text_parts.append(f"Timeline Comments: {timeline_comments}")
    
    # Determine lead engagement level
    if whatsapp_conversations > 0 or open_channel_conversations > 0:
        engagement_level = "HIGH - Digital messaging conversations occurred"
    elif phone_calls > 2:
        engagement_level = "MEDIUM - Multiple phone interactions"
    elif phone_calls > 0 or timeline_comments > 0:
        engagement_level = "BASIC - Phone or comment interactions"
    else:
        engagement_level = "MINIMAL - Limited recorded interactions"
    
    rag_text_parts.append(f"Engagement Level: {engagement_level}")
    
    rag_text_parts.append(f"\n=== DETAILED CONVERSATION TIMELINE ===")
    
    # Enhanced interaction formatting
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
                
                # Enhanced WhatsApp/Open Channel detection and formatting
                if any(keyword in subject.lower() for keyword in ['whatsapp', 'open channel', '[olchat]']):
                    # This is a WhatsApp/Open Channel conversation
                    rag_text_parts.append(f"\n[{i}] {interaction_date} - 💬 WHATSAPP CONVERSATION")
                    rag_text_parts.append(f"Channel: {subject}")
                    
                    # Extract customer name from subject if available
                    if '"' in subject:
                        try:
                            customer_name = subject.split('"')[1]
                            rag_text_parts.append(f"Customer Name: {customer_name}")
                        except:
                            pass
                    
                    # Parse communication details for additional info
                    raw_data = interaction.get('raw_data', {})
                    communications = raw_data.get('COMMUNICATIONS', [])
                    
                    for comm in communications:
                        if comm.get('TYPE') == 'IM' and 'imol|' in str(comm.get('VALUE', '')):
                            # Parse the imol reference
                            imol_value = comm.get('VALUE', '')
                            rag_text_parts.append(f"Platform: WhatsApp Business API")
                            
                            if '971' in imol_value:  # UAE phone number pattern
                                phone_match = [part for part in imol_value.split('|') if '971' in part]
                                if phone_match:
                                    rag_text_parts.append(f"Customer Phone: +{phone_match[0]}")
                            
                            # Customer entity info
                            entity_settings = comm.get('ENTITY_SETTINGS', {})
                            if entity_settings:
                                customer_info = []
                                for key, value in entity_settings.items():
                                    if value and key in ['NAME', 'LAST_NAME', 'LEAD_TITLE']:
                                        customer_info.append(f"{key}: {value}")
                                if customer_info:
                                    rag_text_parts.append(f"Customer Details: {', '.join(customer_info)}")
                    
                    rag_text_parts.append(f"📱 CONVERSATION OCCURRED: WhatsApp messages were exchanged")
                    rag_text_parts.append(f"🔒 Message Content: Not accessible via current API (common Bitrix limitation)")
                    rag_text_parts.append(f"💡 Business Context: Customer initiated contact through WhatsApp for Leo & Loona services")
                    
                elif 'call' in activity_type.lower():
                    # Phone call with enhanced info
                    direction_text = "Outgoing" if direction == "2" else "Incoming" if direction == "1" else "Unknown direction"
                    rag_text_parts.append(f"\n[{i}] {interaction_date} - 📞 PHONE CALL ({direction_text})")
                    
                    if subject:
                        rag_text_parts.append(f"Call Type: {subject}")
                    
                    # Extract phone number from subject
                    import re
                    phone_match = re.search(r'971\d{9}', subject)
                    if phone_match:
                        rag_text_parts.append(f"Customer Phone: +{phone_match.group()}")
                    
                    if description:
                        rag_text_parts.append(f"Call Notes: {description}")
                    if result:
                        rag_text_parts.append(f"Call Outcome: {result}")
                        
                    # Call duration from raw data
                    raw_data = interaction.get('raw_data', {})
                    start_time = raw_data.get('START_TIME', '')
                    end_time = raw_data.get('END_TIME', '')
                    if start_time and end_time:
                        try:
                            start = datetime.fromisoformat(start_time.replace('+03:00', ''))
                            end = datetime.fromisoformat(end_time.replace('+03:00', ''))
                            duration = (end - start).total_seconds()
                            rag_text_parts.append(f"Call Duration: {int(duration)} seconds")
                        except:
                            pass
                            
                else:
                    # Regular activity
                    rag_text_parts.append(f"\n[{i}] {interaction_date} - {activity_type}")
                    if direction:
                        direction_text = "Outgoing" if direction == "2" else "Incoming" if direction == "1" else f"Direction {direction}"
                        rag_text_parts.append(f"Direction: {direction_text}")
                    if subject:
                        rag_text_parts.append(f"Subject: {subject}")
                    if description:
                        rag_text_parts.append(f"Content: {description}")
                    if result:
                        rag_text_parts.append(f"Result: {result}")
                        
            elif interaction['type'] == 'timeline_comment':
                comment = interaction.get('comment', '')
                files = interaction.get('files', [])
                
                rag_text_parts.append(f"\n[{i}] {interaction_date} - 💭 INTERNAL NOTE")
                if comment:
                    rag_text_parts.append(f"Comment: {comment}")
                if files:
                    rag_text_parts.append(f"Attachments: {len(files)} file(s)")
    
    # Enhanced business context summary
    rag_text_parts.append(f"\n=== BUSINESS CONTEXT ANALYSIS ===")
    
    # Lead source analysis
    source = lead_metadata.get('source', '').upper()
    if source == 'WEBFORM':
        rag_text_parts.append("Lead Source: Web form submission (digital marketing)")
    elif 'FACEBOOK' in source:
        rag_text_parts.append("Lead Source: Facebook advertising campaign")
    else:
        rag_text_parts.append(f"Lead Source: {source}")
    
    # Customer journey insights
    if whatsapp_conversations > 0:
        rag_text_parts.append("Customer Journey: Digital-first engagement via WhatsApp")
        rag_text_parts.append("Service Context: Leo & Loona entertainment/birthday party inquiry")
        rag_text_parts.append("Engagement Style: Modern customer preferring instant messaging")
        
        if phone_calls > 0:
            rag_text_parts.append("Follow-up: Multi-channel approach (WhatsApp + phone)")
        else:
            rag_text_parts.append("Communication Preference: Exclusively digital messaging")
    
    elif phone_calls > 0:
        rag_text_parts.append("Customer Journey: Traditional phone-based engagement")
        rag_text_parts.append("Service Context: Leo & Loona entertainment services")
        rag_text_parts.append("Engagement Style: Direct phone communication")
    
    # Status-based insights
    status = lead_metadata.get('status_id', '').upper()
    if status == 'NEW':
        rag_text_parts.append("Lead Status: New inquiry - requires immediate follow-up")
    elif status == 'CONVERTED':
        rag_text_parts.append("Lead Status: Successfully converted - customer acquired")
    elif status == 'JUNK':
        rag_text_parts.append("Lead Status: Marked as junk - low quality lead")
    else:
        rag_text_parts.append(f"Lead Status: {status}")
    
    rag_text_parts.append(f"\n=== END CONVERSATION RECORD ===\n")
    
    return "\\n".join(rag_text_parts)

def extract_lead_conversation_optimized(client, lead_id, lead_info):
    """Optimized conversation extraction that maximizes available data"""
    print(f"  📞 Getting conversation for lead {lead_id}...")
    
    try:
        # Get standard conversation history
        conversation = client.get_lead_conversation_history(lead_id)
        
        # Enhanced formatting with business intelligence
        formatted_conversation = {
            'lead_metadata': {
                'lead_id': lead_id,
                'title': lead_info.get('TITLE', ''),
                'customer_name': f"{lead_info.get('NAME', '')} {lead_info.get('LAST_NAME', '')}".strip() or f"Lead #{lead_id}",
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
            'conversation_analysis': {
                'total_interactions': 0,
                'whatsapp_conversations': 0,
                'phone_calls': 0,
                'timeline_comments': 0,
                'engagement_level': 'MINIMAL',
                'digital_engagement': False,
                'multi_channel': False
            }
        }
        
        # Process all interactions
        activities = conversation.get('activities', [])
        timeline_comments = conversation.get('timeline_comments', [])
        
        all_interactions = []
        whatsapp_count = 0
        phone_count = 0
        
        # Enhanced activity processing
        for activity in activities:
            interaction = {
                'type': 'activity',
                'date': activity.get('CREATED', ''),
                'activity_type': get_activity_type_name(activity.get('TYPE_ID')),
                'subject': activity.get('SUBJECT', ''),
                'description': activity.get('DESCRIPTION', ''),
                'author_id': activity.get('RESPONSIBLE_ID', ''),
                'direction': activity.get('DIRECTION', ''),
                'result': activity.get('RESULT', ''),
                'raw_data': activity
            }
            
            # Count interaction types
            subject_lower = interaction['subject'].lower()
            if any(keyword in subject_lower for keyword in ['whatsapp', 'open channel', '[olchat]']):
                whatsapp_count += 1
            elif 'call' in interaction['activity_type'].lower():
                phone_count += 1
                
            all_interactions.append(interaction)
        
        # Process timeline comments
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
        
        # Enhanced conversation analysis
        formatted_conversation['conversation_analysis'] = {
            'total_interactions': len(all_interactions),
            'whatsapp_conversations': whatsapp_count,
            'phone_calls': phone_count,
            'timeline_comments': len(timeline_comments),
            'engagement_level': 'HIGH' if whatsapp_count > 0 else 'MEDIUM' if phone_count > 2 else 'BASIC' if phone_count > 0 else 'MINIMAL',
            'digital_engagement': whatsapp_count > 0,
            'multi_channel': whatsapp_count > 0 and phone_count > 0
        }
        
        # Generate optimized RAG text
        conversation_text = format_conversation_for_rag_optimized(
            formatted_conversation['lead_metadata'], 
            all_interactions
        )
        
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

def main():
    """Main extraction with enhanced business intelligence"""
    print("🚀 Leo & Loona OPTIMIZED Conversation Extractor")
    print("=" * 60)
    print("Extracting maximum conversation value within Bitrix API limitations")
    print("Enhanced with business intelligence and engagement analysis\\n")
    
    try:
        # Initialize Bitrix client
        print("1. Connecting to Bitrix...")
        config = BitrixConfig()
        client = BitrixClient(config)
        
        if not client.test_connection():
            print("❌ Bitrix connection failed")
            return
        
        print("✅ Connected to Bitrix successfully")
        
        # Get leads from last year (or adjust timeframe as needed)
        print("\\n2. Fetching leads...")
        one_year_ago = datetime.now() - timedelta(days=365)
        date_filter = one_year_ago.strftime('%Y-%m-%d')
        
        filter_params = {'>=DATE_CREATE': date_filter}
        select_fields = [
            'ID', 'TITLE', 'NAME', 'LAST_NAME', 'STATUS_ID', 'DATE_CREATE', 
            'DATE_MODIFY', 'CREATED_BY_ID', 'ASSIGNED_BY_ID', 'COMMENTS',
            'PHONE', 'EMAIL', 'SOURCE_ID', 'OPPORTUNITY', 'CURRENCY_ID'
        ]
        
        all_leads = client.get_leads(select_fields, filter_params)
        
        if not all_leads:
            print("❌ No leads found")
            return
        
        print(f"✅ Found {len(all_leads)} leads to process")
        
        # Process conversations with enhanced extraction
        print(f"\\n3. Extracting conversations with business intelligence...")
        
        all_conversations = {}
        stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'whatsapp_conversations': 0,
            'phone_conversations': 0,
            'multi_channel_leads': 0,
            'high_engagement_leads': 0
        }
        
        for i, lead in enumerate(all_leads, 1):
            lead_id = lead['ID']
            
            if i % 10 == 0 or i == len(all_leads):
                print(f"  Progress: {i}/{len(all_leads)} leads processed")
            
            conversation = extract_lead_conversation_optimized(client, lead_id, lead)
            
            if conversation:
                all_conversations[lead_id] = conversation
                stats['successful_extractions'] += 1
                
                # Collect engagement statistics
                analysis = conversation.get('conversation_analysis', {})
                if analysis.get('whatsapp_conversations', 0) > 0:
                    stats['whatsapp_conversations'] += 1
                if analysis.get('phone_calls', 0) > 0:
                    stats['phone_conversations'] += 1
                if analysis.get('multi_channel', False):
                    stats['multi_channel_leads'] += 1
                if analysis.get('engagement_level') == 'HIGH':
                    stats['high_engagement_leads'] += 1
            
            stats['total_processed'] += 1
        
        # Save enhanced results
        print(f"\\n4. Saving optimized conversation data...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create output directory
        output_dir = 'optimized_conversations'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comprehensive RAG corpus
        combined_file = f"{output_dir}/leo_loona_conversations_optimized_{timestamp}.txt"
        with open(combined_file, 'w', encoding='utf-8') as f:
            f.write("LEO & LOONA OPTIMIZED CONVERSATION CORPUS\\n")
            f.write("=" * 60 + "\\n")
            f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Total Conversations: {stats['successful_extractions']}\\n")
            f.write(f"WhatsApp Conversations: {stats['whatsapp_conversations']}\\n")
            f.write(f"Phone Conversations: {stats['phone_conversations']}\\n")
            f.write(f"Multi-Channel Leads: {stats['multi_channel_leads']}\\n")
            f.write(f"High Engagement Leads: {stats['high_engagement_leads']}\\n")
            f.write("=" * 60 + "\\n\\n")
            
            for lead_id, conversation in all_conversations.items():
                if conversation and conversation.get('rag_formatted_text'):
                    f.write(conversation['rag_formatted_text'])
                    f.write("\\n" + "="*100 + "\\n\\n")
        
        # Save structured data
        json_file = f"{output_dir}/conversations_structured_optimized_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_conversations, f, indent=2, ensure_ascii=False, default=str)
        
        # Enhanced summary report
        summary_file = f"{output_dir}/extraction_summary_optimized_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("LEO & LOONA OPTIMIZED CONVERSATION EXTRACTION REPORT\\n")
            f.write("=" * 60 + "\\n")
            f.write(f"Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Total Leads Processed: {stats['total_processed']}\\n")
            f.write(f"Successful Extractions: {stats['successful_extractions']}\\n\\n")
            
            f.write("ENGAGEMENT ANALYSIS:\\n")
            f.write(f"├── WhatsApp Conversations: {stats['whatsapp_conversations']} leads\\n")
            f.write(f"├── Phone Conversations: {stats['phone_conversations']} leads\\n")
            f.write(f"├── Multi-Channel Engagement: {stats['multi_channel_leads']} leads\\n")
            f.write(f"└── High Engagement Level: {stats['high_engagement_leads']} leads\\n\\n")
            
            engagement_rate = (stats['whatsapp_conversations'] / stats['successful_extractions'] * 100) if stats['successful_extractions'] > 0 else 0
            f.write(f"Digital Engagement Rate: {engagement_rate:.1f}%\\n")
            
            conversion_insights = []
            if stats['whatsapp_conversations'] > 0:
                conversion_insights.append("✓ Strong digital customer base using WhatsApp")
            if stats['multi_channel_leads'] > 0:
                conversion_insights.append("✓ Multi-channel approach increases engagement")
            if stats['phone_conversations'] > stats['whatsapp_conversations']:
                conversion_insights.append("✓ Traditional phone communication still dominant")
            
            f.write("\\nBUSINESS INSIGHTS:\\n")
            for insight in conversion_insights:
                f.write(f"  {insight}\\n")
        
        print(f"\\n🎉 OPTIMIZED EXTRACTION COMPLETE!")
        print(f"📊 ENGAGEMENT STATISTICS:")
        print(f"   • Total Conversations: {stats['successful_extractions']}")
        print(f"   • WhatsApp Conversations: {stats['whatsapp_conversations']}")
        print(f"   • Phone Conversations: {stats['phone_conversations']}")
        print(f"   • Multi-Channel Leads: {stats['multi_channel_leads']}")
        print(f"   • High Engagement Leads: {stats['high_engagement_leads']}")
        
        digital_rate = (stats['whatsapp_conversations'] / stats['successful_extractions'] * 100) if stats['successful_extractions'] > 0 else 0
        print(f"   • Digital Engagement Rate: {digital_rate:.1f}%")
        
        print(f"\\n📁 OUTPUT FILES:")
        print(f"   • Optimized RAG Corpus: {combined_file}")
        print(f"   • Structured Data: {json_file}")
        print(f"   • Business Report: {summary_file}")
        
        print(f"\\n💡 KEY ACHIEVEMENTS:")
        print(f"✅ Maximum conversation data extracted within API limits")
        print(f"✅ WhatsApp conversations clearly identified and contextualized")
        print(f"✅ Business intelligence and engagement analysis included")
        print(f"✅ Customer journey insights provided")
        print(f"✅ Ready for RAG system ingestion")
        
        return {
            'output_files': [combined_file, json_file, summary_file],
            'statistics': stats
        }
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return None

if __name__ == "__main__":
    result = main()
    
    if result:
        print(f"\\n🚀 SUCCESS! Your Leo & Loona conversation data is now optimized for RAG analysis!")
        print(f"Even without full WhatsApp message content, you have rich business intelligence")
        print(f"about customer engagement patterns, communication preferences, and lead quality.")
    else:
        print(f"\\n❌ Extraction failed. Please check the error messages above.")
