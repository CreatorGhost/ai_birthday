#!/usr/bin/env python3
"""
Dynamic model fetching utilities for LLM providers.

This module handles the dynamic retrieval of available models from
OpenAI and Google Gemini APIs, with fallback to static configurations.
"""

import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ModelFetcher:
    """Handles dynamic fetching of models from various LLM providers"""
    
    @staticmethod
    def fetch_openai_models() -> Dict[str, Dict[str, Any]]:
        """Dynamically fetch available OpenAI models"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            models = client.models.list()
            chat_models = {}
            embedding_models = {}
            
            for model in models.data:
                model_id = model.id
                
                # Filter chat models (GPT models)
                if any(prefix in model_id for prefix in ['gpt-', 'o1-', 'o3-']):
                    chat_models[model_id] = {
                        "model_name": model_id,
                        "temperature": 0.1,
                        "max_tokens": 4096
                    }
                
                # Filter embedding models
                elif 'embedding' in model_id:
                    # Default dimensions for known models
                    dimensions = 1536  # Default
                    if 'text-embedding-3-large' in model_id:
                        dimensions = 3072
                    elif 'text-embedding-3-small' in model_id:
                        dimensions = 1536
                    elif 'text-embedding-ada-002' in model_id:
                        dimensions = 1536
                    
                    embedding_models[model_id] = {
                        "model_name": model_id,
                        "dimensions": dimensions
                    }
            
            logger.info(f"Fetched {len(chat_models)} OpenAI chat models and {len(embedding_models)} embedding models")
            return {
                "chat_models": chat_models,
                "embedding_models": embedding_models,
                "required_env_vars": ["OPENAI_API_KEY"]
            }
            
        except Exception as e:
            logger.warning(f"Failed to fetch OpenAI models dynamically: {e}")
            raise
    
    @staticmethod
    def fetch_google_models() -> Dict[str, Dict[str, Any]]:
        """Dynamically fetch available Google Gemini models"""
        try:
            try:
                import google.generativeai as genai
            except ImportError:
                logger.warning("google-generativeai package not installed. Install with: pip install google-generativeai")
                raise ImportError("google-generativeai package required for dynamic Google model fetching")
            
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found")
                
            genai.configure(api_key=api_key)
            
            models = genai.list_models()
            chat_models = {}
            embedding_models = {}
            
            for model in models:
                model_name = model.name.replace('models/', '')
                
                # Filter chat models (Gemini models that support generateContent)
                if hasattr(model, 'supported_generation_methods') and 'generateContent' in model.supported_generation_methods:
                    chat_models[model_name] = {
                        "model_name": model_name,
                        "temperature": 0.1,
                        "max_output_tokens": 4096
                    }
                
                # Filter embedding models
                elif (hasattr(model, 'supported_generation_methods') and 'embedContent' in model.supported_generation_methods) or 'embedding' in model_name.lower():
                    # Set dimensions based on specific model
                    dimensions = 768  # Default for most Gemini embedding models
                    if 'gemini-embedding-001' in model_name:
                        dimensions = 3072  # Actual dimension for gemini-embedding-001
                    elif 'text-embedding-004' in model_name:
                        dimensions = 768
                    elif 'text-multilingual-embedding-002' in model_name:
                        dimensions = 768
                    
                    embedding_models[model_name] = {
                        "model_name": model_name,
                        "dimensions": dimensions
                    }
            
            logger.info(f"Fetched {len(chat_models)} Google chat models and {len(embedding_models)} embedding models")
            return {
                "chat_models": chat_models,
                "embedding_models": embedding_models,
                "required_env_vars": ["GOOGLE_API_KEY"]
            }
            
        except Exception as e:
            logger.warning(f"Failed to fetch Google models dynamically: {e}")
            raise