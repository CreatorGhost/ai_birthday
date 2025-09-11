"""
Phone number utilities for WhatsApp system
"""

import random
import re
from typing import Optional

def generate_test_phone() -> str:
    """Generate a random UAE phone number for testing"""
    uae_prefixes = ['50', '52', '54', '55', '56', '58']
    prefix = random.choice(uae_prefixes)
    number = ''.join([str(random.randint(0, 9)) for _ in range(7)])
    return f"+971{prefix}{number}"

def normalize_phone(phone: str) -> str:
    """Normalize phone number format"""
    # Remove all non-digit characters except +
    cleaned = re.sub(r'[^\d+]', '', phone)
    
    # Ensure it starts with + if it doesn't already
    if not cleaned.startswith('+'):
        cleaned = '+' + cleaned
    
    return cleaned

def validate_uae_phone(phone: str) -> bool:
    """Validate UAE phone number format"""
    normalized = normalize_phone(phone)
    
    # UAE phone number pattern: +971XXXXXXXXX (9 digits after country code)
    pattern = r'^\+971[0-9]{9}$'
    return bool(re.match(pattern, normalized))

def extract_phone_from_webhook(webhook_data: dict) -> Optional[str]:
    """Extract phone number from webhook data"""
    phone_fields = ['phone', 'from', 'number', 'contact_phone']
    
    for field in phone_fields:
        if field in webhook_data and webhook_data[field]:
            return normalize_phone(str(webhook_data[field]))
    
    return None
