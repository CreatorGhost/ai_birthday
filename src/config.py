import os
from enum import Enum
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import logging
from .model_fetcher import ModelFetcher

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class LLMProvider(Enum):
    OPENAI = "openai"
    GOOGLE_GEMINI = "google_gemini"

class ModelConfig:
    """Configuration class for LLM models"""
    
    # Static fallback model configurations
    STATIC_MODEL_CONFIGS = {
        LLMProvider.OPENAI: {
            "chat_models": {
                "gpt-4o": {
                    "model_name": "gpt-4o",
                    "temperature": 0.1,
                    "max_tokens": 4096
                },
                "gpt-4o-mini": {
                    "model_name": "gpt-4o-mini",
                    "temperature": 0.1,
                    "max_tokens": 4096
                },
                "gpt-4-turbo": {
                    "model_name": "gpt-4-turbo",
                    "temperature": 0.1,
                    "max_tokens": 4096
                },
                "gpt-3.5-turbo": {
                    "model_name": "gpt-3.5-turbo",
                    "temperature": 0.1,
                    "max_tokens": 4096
                }
            },
            "embedding_models": {
                "text-embedding-3-large": {
                    "model_name": "text-embedding-3-large",
                    "dimensions": 3072
                },
                "text-embedding-3-small": {
                    "model_name": "text-embedding-3-small",
                    "dimensions": 1536
                },
                "text-embedding-ada-002": {
                    "model_name": "text-embedding-ada-002",
                    "dimensions": 1536
                }
            },
            "required_env_vars": ["OPENAI_API_KEY"]
        },
        LLMProvider.GOOGLE_GEMINI: {
            "chat_models": {
                "gemini-2.0-flash-exp": {
                    "model_name": "gemini-2.0-flash-exp",
                    "temperature": 0.1,
                    "max_output_tokens": 4096
                },
                "gemini-1.5-pro": {
                    "model_name": "gemini-1.5-pro",
                    "temperature": 0.1,
                    "max_output_tokens": 4096
                },
                "gemini-1.5-flash": {
                    "model_name": "gemini-1.5-flash",
                    "temperature": 0.1,
                    "max_output_tokens": 4096
                },
                "gemini-pro": {
                    "model_name": "gemini-pro",
                    "temperature": 0.1,
                    "max_output_tokens": 4096
                }
            },
            "embedding_models": {
                "gemini-embedding-001": {
                    "model_name": "gemini-embedding-001",
                    "dimensions": 3072
                },
                "text-embedding-004": {
                    "model_name": "text-embedding-004",
                    "dimensions": 768
                },
                "text-multilingual-embedding-002": {
                    "model_name": "text-multilingual-embedding-002",
                    "dimensions": 768
                }
            },
            "required_env_vars": ["GOOGLE_API_KEY"]
        }
    }
    

    
    @classmethod
    def get_dynamic_model_configs(cls, use_dynamic: bool = True) -> Dict[LLMProvider, Dict[str, Any]]:
        """Get model configurations with optional dynamic fetching"""
        if not use_dynamic:
            return cls.STATIC_MODEL_CONFIGS
            
        configs = {}
        
        # Fetch OpenAI models
        try:
            configs[LLMProvider.OPENAI] = ModelFetcher.fetch_openai_models()
        except Exception as e:
            logger.error(f"Error fetching OpenAI models: {e}")
            configs[LLMProvider.OPENAI] = cls.STATIC_MODEL_CONFIGS[LLMProvider.OPENAI]
        
        # Fetch Google models
        try:
            configs[LLMProvider.GOOGLE_GEMINI] = ModelFetcher.fetch_google_models()
        except Exception as e:
            logger.error(f"Error fetching Google models: {e}")
            configs[LLMProvider.GOOGLE_GEMINI] = cls.STATIC_MODEL_CONFIGS[LLMProvider.GOOGLE_GEMINI]
        
        return configs
    
    # Use dynamic configs by default, fallback to static
    MODEL_CONFIGS = property(lambda self: self.get_dynamic_model_configs())
    
    def __init__(self, use_dynamic_models: bool = True):
        # Load configuration from environment variables
        self.use_dynamic_models = use_dynamic_models
        self._model_configs = None
        self.llm_provider = self._get_llm_provider()
        self.chat_model = self._get_chat_model()
        self.embedding_model = self._get_embedding_model()
        self.validate_configuration()
    
    @property
    def model_configs(self):
        """Get model configurations (cached)"""
        if self._model_configs is None:
            self._model_configs = self.get_dynamic_model_configs(self.use_dynamic_models)
        return self._model_configs
    
    def _get_llm_provider(self) -> LLMProvider:
        """Get LLM provider from environment variable"""
        provider_str = os.getenv('LLM_PROVIDER', 'openai').lower()
        try:
            return LLMProvider(provider_str)
        except ValueError:
            print(f"Warning: Unknown LLM provider '{provider_str}', defaulting to OpenAI")
            return LLMProvider.OPENAI
    
    def _get_chat_model(self) -> str:
        """Get chat model from environment variable"""
        if self.llm_provider == LLMProvider.OPENAI:
            return os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o')
        elif self.llm_provider == LLMProvider.GOOGLE_GEMINI:
            return os.getenv('GOOGLE_CHAT_MODEL', 'gemini-1.5-pro')
        else:
            return 'gpt-4o'  # Default fallback
    
    def _get_embedding_model(self) -> str:
        """Get embedding model from environment variable"""
        if self.llm_provider == LLMProvider.OPENAI:
            return os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
        elif self.llm_provider == LLMProvider.GOOGLE_GEMINI:
            return os.getenv('GOOGLE_EMBEDDING_MODEL', 'gemini-embedding-001')
        else:
            return 'text-embedding-3-small'  # Default fallback
    
    def validate_configuration(self):
        """Validate the current configuration"""
        provider_config = self.model_configs.get(self.llm_provider, {})
        
        # Check if chat model exists
        chat_models = provider_config.get('chat_models', {})
        if self.chat_model not in chat_models:
            available_models = list(chat_models.keys())
            logger.warning(f"Chat model '{self.chat_model}' not found for {self.llm_provider.value}. Available: {available_models[:5]}...")
        
        # Check if embedding model exists
        embedding_models = provider_config.get('embedding_models', {})
        if self.embedding_model not in embedding_models:
            available_models = list(embedding_models.keys())
            logger.warning(f"Embedding model '{self.embedding_model}' not found for {self.llm_provider.value}. Available: {available_models}")
        
        # Check required environment variables
        required_vars = provider_config.get('required_env_vars', [])
        for var in required_vars:
            if not os.getenv(var):
                raise ValueError(f"Required environment variable '{var}' not found for {self.llm_provider.value}")
    
    def get_chat_model_config(self) -> Dict[str, Any]:
        """Get configuration for the selected chat model"""
        provider_config = self.model_configs.get(self.llm_provider, {})
        chat_models = provider_config.get('chat_models', {})
        return chat_models.get(self.chat_model, {})
    
    def get_embedding_model_config(self) -> Dict[str, Any]:
        """Get configuration for the selected embedding model"""
        provider_config = self.model_configs.get(self.llm_provider, {})
        embedding_models = provider_config.get('embedding_models', {})
        return embedding_models.get(self.embedding_model, {})
    
    def get_embedding_dimensions(self) -> int:
        """Get embedding dimensions for the selected model"""
        embedding_config = self.get_embedding_model_config()
        return embedding_config.get('dimensions', 1536)  # Default fallback
    
    def create_chat_llm(self):
        """Create a chat LLM instance based on the current configuration"""
        chat_config = self.get_chat_model_config()
        
        if self.llm_provider == LLMProvider.OPENAI:
            return ChatOpenAI(
                model=self.chat_model,
                temperature=chat_config.get('temperature', 0.1),
                max_tokens=chat_config.get('max_tokens', 4096),
                api_key=os.getenv('OPENAI_API_KEY')
            )
        elif self.llm_provider == LLMProvider.GOOGLE_GEMINI:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=self.chat_model,
                    temperature=chat_config.get('temperature', 0.1),
                    max_output_tokens=chat_config.get('max_output_tokens', 4096),
                    google_api_key=os.getenv('GOOGLE_API_KEY')
                )
            except ImportError:
                raise ImportError("langchain_google_genai is required for Google Gemini models. Install with: pip install langchain-google-genai")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def create_embedding_model(self):
        """Create an embedding model instance based on the current configuration"""
        if self.llm_provider == LLMProvider.OPENAI:
            return OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=os.getenv('OPENAI_API_KEY')
            )
        elif self.llm_provider == LLMProvider.GOOGLE_GEMINI:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                return GoogleGenerativeAIEmbeddings(
                    model=self.embedding_model,
                    google_api_key=os.getenv('GOOGLE_API_KEY')
                )
            except ImportError:
                raise ImportError("langchain_google_genai is required for Google Gemini models. Install with: pip install langchain-google-genai")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def get_provider_info(self) -> Dict[str, str]:
        """Get current provider information"""
        return {
            'provider': self.llm_provider.value.upper(),
            'chat_model': self.chat_model,
            'embedding_model': self.embedding_model
        }

# Global functions for backward compatibility
def get_available_providers() -> Dict[str, List[str]]:
    """Get available providers and their models"""
    config = ModelConfig()
    providers = {}
    
    for provider, provider_config in config.model_configs.items():
        chat_models = list(provider_config.get('chat_models', {}).keys())
        embedding_models = list(provider_config.get('embedding_models', {}).keys())
        
        providers[provider.value] = {
            'chat_models': chat_models,
            'embedding_models': embedding_models
        }
    
    return providers

def create_chat_llm(provider: str = None, model: str = None):
    """Create a chat LLM with optional provider and model override"""
    if provider:
        os.environ['LLM_PROVIDER'] = provider
    if model:
        if provider == 'openai' or (not provider and os.getenv('LLM_PROVIDER', 'openai') == 'openai'):
            os.environ['OPENAI_CHAT_MODEL'] = model
        elif provider == 'google_gemini' or (not provider and os.getenv('LLM_PROVIDER') == 'google_gemini'):
            os.environ['GOOGLE_CHAT_MODEL'] = model
    
    config = ModelConfig()
    return config.create_chat_llm()

def create_embedding_model(provider: str = None, model: str = None):
    """Create an embedding model with optional provider and model override"""
    if provider:
        os.environ['LLM_PROVIDER'] = provider
    if model:
        if provider == 'openai' or (not provider and os.getenv('LLM_PROVIDER', 'openai') == 'openai'):
            os.environ['OPENAI_EMBEDDING_MODEL'] = model
        elif provider == 'google_gemini' or (not provider and os.getenv('LLM_PROVIDER') == 'google_gemini'):
            os.environ['GOOGLE_EMBEDDING_MODEL'] = model
    
    config = ModelConfig()
    return config.create_embedding_model()