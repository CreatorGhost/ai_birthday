"""
Core configuration for WhatsApp backend system
Handles environment variables, API keys, and system settings
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration management"""
    
    # API Keys
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    GOOGLE_API_KEY: Optional[str] = os.getenv('GOOGLE_API_KEY')
    PINECONE_API_KEY: Optional[str] = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX_NAME: str = os.getenv('PINECONE_INDEX_NAME', 'faq-embeddings')
    
    # Bitrix Integration
    BITRIX_WEBHOOK_URL: Optional[str] = os.getenv('BITRIX_WEBHOOK_URL')
    BITRIX_API_KEY: Optional[str] = os.getenv('BITRIX_API_KEY')
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = int(os.getenv('WHATSAPP_PORT', '8001'))
    DEBUG: bool = os.getenv('DEBUG', 'false').lower() == 'true'
    
    # Session Management
    MAX_CHAT_HISTORY_LENGTH: int = 20
    SESSION_TIMEOUT_MINUTES: int = 60
    
    # Storage
    STORAGE_DIR: str = "user_data"
    LOGS_DIR: str = "logs"
    
    # WhatsApp Settings
    TEST_MODE: bool = os.getenv('WHATSAPP_TEST_MODE', 'true').lower() == 'true'
    TEST_LINE_ID: str = "99"  # Test WhatsApp line in Bitrix
    PRODUCTION_LINE_ID: str = "1"  # Production WhatsApp line
    
    @classmethod
    def validate_required_keys(cls) -> tuple[bool, list[str]]:
        """Validate that required API keys are present"""
        missing_keys = []
        
        # At least one LLM provider required
        if not cls.OPENAI_API_KEY and not cls.GOOGLE_API_KEY:
            missing_keys.append("OPENAI_API_KEY or GOOGLE_API_KEY")
        
        # Vector store required
        if not cls.PINECONE_API_KEY:
            missing_keys.append("PINECONE_API_KEY")
        
        return len(missing_keys) == 0, missing_keys
    
    @classmethod
    def get_status(cls) -> dict:
        """Get configuration status for monitoring"""
        is_valid, missing = cls.validate_required_keys()
        
        return {
            "valid": is_valid,
            "missing_keys": missing,
            "api_keys": {
                "openai": bool(cls.OPENAI_API_KEY),
                "google": bool(cls.GOOGLE_API_KEY),
                "pinecone": bool(cls.PINECONE_API_KEY),
                "bitrix": bool(cls.BITRIX_WEBHOOK_URL)
            },
            "mode": "TEST" if cls.TEST_MODE else "PRODUCTION",
            "line_id": cls.TEST_LINE_ID if cls.TEST_MODE else cls.PRODUCTION_LINE_ID
        }

# Global config instance
config = Config()
