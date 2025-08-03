"""Bitrix CRM Integration Package."""

# Import main classes for easy access
from .bitrix_client import BitrixClient
from .config import BitrixConfig

__all__ = [
    'BitrixClient',
    'BitrixConfig'
]