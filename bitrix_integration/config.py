import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class BitrixConfig:
    """Simple configuration class for Bitrix connection testing."""
    
    def __init__(self):
        """Initialize Bitrix configuration from environment variables."""
        self.webhook_url = os.getenv('BITRIX_WEBHOOK_URL')
        self.api_timeout = int(os.getenv('BITRIX_API_TIMEOUT', '30'))
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the configuration and return status."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required configurations
        if not self.webhook_url:
            validation_results['errors'].append("BITRIX_WEBHOOK_URL is required")
            validation_results['valid'] = False
        
        # Check optional configurations
        if self.api_timeout < 10:
            validation_results['warnings'].append("API timeout is very low, consider increasing it")
        
        return validation_results
    
    def get_webhook_url(self) -> str:
        """Get Bitrix webhook URL."""
        return self.webhook_url
    
    def is_configured(self) -> bool:
        """Check if basic configuration is available."""
        return bool(self.webhook_url)

# Helper functions
def get_bitrix_webhook_url() -> str:
    """Get Bitrix webhook URL from configuration."""
    config = BitrixConfig()
    return config.webhook_url

def is_bitrix_configured() -> bool:
    """Check if Bitrix integration is properly configured."""
    config = BitrixConfig()
    return config.is_configured()