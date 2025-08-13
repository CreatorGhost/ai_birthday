# Bitrix24 Integration - Lead Management

This module provides a comprehensive Python client for interacting with Bitrix24 CRM, specifically focused on lead management operations.

## Features

- ✅ **Connection Testing**: Verify API connectivity and permissions
- ✅ **Lead Creation**: Create new leads with full field support
- ✅ **Lead Updates**: Update existing leads with partial or complete data
- ✅ **Lead Retrieval**: Get individual leads or lists with filtering
- ✅ **Field Discovery**: Retrieve available lead fields and their properties
- ✅ **Conversation History**: Retrieve activities and timeline comments for existing leads
- ✅ **Activity Tracking**: Access calls, emails, meetings, and other lead activities
- ✅ **Timeline Comments**: Get comments and notes from lead timelines
- ✅ **Error Handling**: Comprehensive error handling and logging
- ✅ **Pagination Support**: Handle large datasets automatically

## Setup

### 1. Environment Configuration

Create a `.env` file in your project root:

```env
BITRIX_WEBHOOK_URL=https://your-domain.bitrix24.com/rest/1/your-webhook-code/
```

### 2. Installation

Install required dependencies:

```bash
pip install requests python-dotenv
```

## Usage Examples

### Basic Connection Test

```python
from bitrix_integration.bitrix_client import BitrixClient

# Initialize client
client = BitrixClient()

# Test connection
if client.test_connection():
    print("Connected successfully!")
else:
    print("Connection failed!")
```

### Creating a New Lead

```python
# Define lead data
lead_data = {
    'TITLE': 'New Business Opportunity',
    'NAME': 'John',
    'LAST_NAME': 'Smith',
    'STATUS_ID': 'NEW',
    'PHONE': [{'VALUE': '+1234567890', 'VALUE_TYPE': 'WORK'}],
    'EMAIL': [{'VALUE': 'john@example.com', 'VALUE_TYPE': 'WORK'}],
    'COMMENTS': 'Interested in premium services',
    'OPPORTUNITY': 5000,
    'CURRENCY_ID': 'USD',
    'SOURCE_ID': 'WEB'
}

# Create the lead
response = client.create_lead(lead_data)
if 'result' in response:
    lead_id = response['result']
    print(f"Lead created with ID: {lead_id}")
```

### Updating an Existing Lead

```python
# Update lead status and add notes
update_data = {
    'STATUS_ID': 'IN_PROCESS',
    'COMMENTS': 'Customer showed strong interest during call',
    'OPPORTUNITY': 7500
}

response = client.update_lead(lead_id, update_data)
if response.get('result'):
    print("Lead updated successfully!")
```

### Retrieving Lead Information

```python
# Get specific lead
lead = client.get_lead_by_id(lead_id)
if 'result' in lead:
    lead_info = lead['result']
    print(f"Lead: {lead_info['TITLE']}")
    print(f"Status: {lead_info['STATUS_ID']}")
    print(f"Opportunity: {lead_info['OPPORTUNITY']}")

# Get all leads with filtering
leads = client.get_leads(
    select_fields=['ID', 'TITLE', 'STATUS_ID', 'OPPORTUNITY'],
    filter_params={'STATUS_ID': 'NEW'}
)
print(f"Found {len(leads)} new leads")
```

### Working with Lead Fields

```python
# Get all available lead fields
fields_response = client.get_lead_fields()
if 'result' in fields_response:
    fields = fields_response['result']
    for field_name, field_info in fields.items():
        print(f"Field: {field_name}")
        print(f"  Type: {field_info.get('type')}")
        print(f"  Title: {field_info.get('title')}")
        print(f"  Required: {field_info.get('isRequired', False)}")
```

### Retrieving Conversation History

```python
# Get complete conversation history for a lead
lead_id = 123
conversation = client.get_lead_conversation_history(lead_id)

print(f"Total Activities: {conversation['total_activities']}")
print(f"Total Comments: {conversation['total_comments']}")

# Access activities (calls, emails, meetings, etc.)
for activity in conversation['activities']:
    activity_type = activity.get('TYPE_ID')  # 1=Call, 2=Meeting, 4=Email, etc.
    subject = activity.get('SUBJECT')
    created = activity.get('CREATED')
    print(f"Activity: {subject} (Type: {activity_type}, Created: {created})")

# Access timeline comments
for comment in conversation['timeline_comments']:
    comment_text = comment.get('COMMENT')
    author_id = comment.get('AUTHOR_ID')
    created = comment.get('CREATED')
    print(f"Comment by {author_id}: {comment_text}")
```

### Getting Specific Activity Types

```python
# Get only calls and emails for a lead
lead_id = 123
activity_types = [1, 4]  # 1=Call, 4=Email
activities_response = client.get_lead_activities(lead_id, activity_types)

if 'result' in activities_response:
    activities = activities_response['result']
    for activity in activities:
        print(f"Activity: {activity.get('SUBJECT')}")
        
        # Access communication details
        communications = activity.get('COMMUNICATIONS', [])
        for comm in communications:
            comm_type = comm.get('TYPE')  # PHONE, EMAIL, etc.
            comm_value = comm.get('VALUE')
            print(f"  Communication: {comm_type} - {comm_value}")
```

### Getting Timeline Comments Only

```python
# Get only timeline comments for a lead
lead_id = 123
comments_response = client.get_lead_timeline_comments(lead_id)

if 'result' in comments_response:
    comments = comments_response['result']
    for comment in comments:
        comment_text = comment.get('COMMENT')
        created = comment.get('CREATED')
        files = comment.get('FILES', [])
        
        print(f"Comment: {comment_text}")
        print(f"Created: {created}")
        if files:
            print(f"Attachments: {len(files)} files")
```

## Available Lead Fields

### Basic Information
- `TITLE` - Lead title/name
- `NAME` - First name
- `SECOND_NAME` - Middle name
- `LAST_NAME` - Last name
- `COMPANY_TITLE` - Company name
- `POST` - Position/job title

### Contact Information
- `PHONE` - Phone numbers (array of objects)
- `EMAIL` - Email addresses (array of objects)
- `WEB` - Websites (array of objects)
- `IM` - Instant messengers (array of objects)

### Address Fields
- `ADDRESS` - Street address
- `ADDRESS_CITY` - City
- `ADDRESS_REGION` - Region/state
- `ADDRESS_COUNTRY` - Country
- `ADDRESS_POSTAL_CODE` - Postal/ZIP code

### Business Information
- `STATUS_ID` - Lead status (NEW, IN_PROCESS, PROCESSED, etc.)
- `SOURCE_ID` - Lead source (WEB, EMAIL, CALL, etc.)
- `OPPORTUNITY` - Expected deal value
- `CURRENCY_ID` - Currency code (USD, EUR, etc.)
- `ASSIGNED_BY_ID` - Responsible user ID

### Tracking & Analytics
- `UTM_SOURCE` - Traffic source
- `UTM_MEDIUM` - Marketing medium
- `UTM_CAMPAIGN` - Campaign name
- `UTM_CONTENT` - Ad content
- `UTM_TERM` - Search terms

### Additional Fields
- `COMMENTS` - Notes and comments
- `BIRTHDATE` - Date of birth
- `OPENED` - Visibility flag (Y/N)
- `SOURCE_DESCRIPTION` - Source description

## Activity Types

When retrieving conversation history, activities are categorized by type:

- **1**: Call
- **2**: Meeting
- **3**: Task
- **4**: Email
- **5**: SMS
- **6**: Request
- **7**: Provider
- **8**: Rest App

## Communication Types

Activities may contain communication details with these types:

- **PHONE**: Phone number
- **EMAIL**: Email address
- **WEB**: Website URL
- **IM**: Instant messenger
- **OTHER**: Other communication method

## Phone and Email Format

For multi-value fields like phone and email:

```python
# Phone numbers
'PHONE': [
    {'VALUE': '+1234567890', 'VALUE_TYPE': 'WORK'},
    {'VALUE': '+0987654321', 'VALUE_TYPE': 'MOBILE'},
    {'VALUE': '+1122334455', 'VALUE_TYPE': 'HOME'}
]

# Email addresses
'EMAIL': [
    {'VALUE': 'work@example.com', 'VALUE_TYPE': 'WORK'},
    {'VALUE': 'personal@example.com', 'VALUE_TYPE': 'HOME'}
]
```

## Lead Status Values

Common status values:
- `NEW` - Unprocessed
- `IN_PROCESS` - In progress
- `PROCESSED` - Processed
- `JUNK` - Low-quality lead
- `CONVERTED` - High-quality lead

## Source Values

Common source values:
- `CALL` - Phone call
- `EMAIL` - Email
- `WEB` - Website
- `ADVERTISING` - Advertising
- `PARTNER` - Partner referral
- `RECOMMENDATION` - Recommendation
- `TRADE_SHOW` - Trade show
- `WEBFORM` - Web form
- `OTHER` - Other sources

## Error Handling

The client includes comprehensive error handling:

```python
try:
    response = client.create_lead(lead_data)
    if 'result' in response:
        print(f"Success: {response['result']}")
    else:
        print(f"API Error: {response}")
except Exception as e:
    print(f"Connection Error: {str(e)}")
```

## Testing

Run the built-in test suite:

```bash
# Test basic functionality
python bitrix_integration/bitrix_client.py

# Run comprehensive lead management example
python example_lead_usage.py

# Run the conversation history demo
python example_conversation_history.py
```

**Note**: Conversation history will be most visible with older leads that have had activities (calls, emails, meetings) and timeline comments. Newly created leads may not have conversation history yet.

## API Methods Reference

### BitrixClient Methods

| Method | Description | Returns |
|--------|-------------|----------|
| `test_connection()` | Test API connectivity | `bool` |
| `get_user_info()` | Get current user info | `Dict` |
| `create_lead(lead_data, register_event=True)` | Create new lead | `Dict` |
| `update_lead(lead_id, lead_data, register_event=True)` | Update existing lead | `Dict` |
| `get_lead_by_id(lead_id)` | Get specific lead | `Dict` |
| `get_leads(select_fields=None, filter_params=None)` | Get leads list | `List` |
| `get_lead_fields()` | Get available fields | `Dict` |
| `get_lead_conversation_history(lead_id)` | Get complete conversation history | `Dict` |
| `get_lead_activities(lead_id, activity_types=None)` | Get lead activities | `Dict` |
| `get_lead_timeline_comments(lead_id)` | Get timeline comments | `Dict` |
| `get_deals(select_fields=None, filter_params=None)` | Get deals list | `List` |

## Best Practices

1. **Always test connection first** before performing operations
2. **Use specific field selection** to improve performance
3. **Handle errors gracefully** with try-catch blocks
4. **Set appropriate lead status** based on your workflow
5. **Include contact information** for better lead quality
6. **Use UTM parameters** for tracking lead sources
7. **Add meaningful comments** for lead context

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Check webhook URL format
   - Verify webhook permissions
   - Ensure network connectivity

2. **Field Errors**
   - Use `get_lead_fields()` to check available fields
   - Verify field value formats
   - Check required field constraints

3. **Permission Errors**
   - Ensure webhook has CRM permissions
   - Check user access rights
   - Verify lead creation permissions

### Debug Mode

Enable detailed logging by modifying the client:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## License

This integration module is part of the AI Birthday project and follows the same licensing terms.