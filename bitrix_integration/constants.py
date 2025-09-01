"""
Bitrix CRM Constants and Mappings
Generated from API on 2025-08-29
"""

# Bitrix Lead Status ID Mapping (fetched from crm.status.list API)
BITRIX_LEAD_STATUS_IDS = {
    'Inquiry': 'NEW',                    # Initial stage for chatbot (Sort: 10)
    'NEW APPROACH': 'UC_NJ6R1M',         # Birthday party leads (Sort: 20) - Note: typo in Bitrix "NEW APROACH"  
    'Follow up': 'UC_EIMYT2',            # Follow up stage (Sort: 30)
    'INTERESTED': 'UC_TKL10Z',           # Interested stage (Sort: 40)
    'GENERAL QUESTIONS': 'UC_0MD91B',    # General questions stage (Sort: 50)
    'Closed Deal': 'CONVERTED',          # Converted leads (Sort: 60)
    'Junk Lead': 'JUNK'                  # Junk leads (Sort: 70)
}

# Reverse mapping for easier lookup
BITRIX_STATUS_NAMES = {v: k for k, v in BITRIX_LEAD_STATUS_IDS.items()}

# Workflow stages in order
WORKFLOW_STAGES = [
    'Inquiry',           # NEW - Initial chatbot assignment
    'GENERAL QUESTIONS', # UC_0MD91B - General questions 
    'NEW APPROACH',      # UC_NJ6R1M - Birthday party leads
    'Follow up',         # UC_EIMYT2 - Follow up
    'INTERESTED',        # UC_TKL10Z - Interested
    'Closed Deal',       # CONVERTED - Converted
    'Junk Lead'          # JUNK - Junk
]