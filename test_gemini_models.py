#!/usr/bin/env python3
"""
Test script to demonstrate dynamic Google Gemini model fetching.

This script shows how to use the existing ModelFetcher class to
dynamically retrieve available Google Gemini models.
"""

import os
from src.model_fetcher import ModelFetcher

def test_gemini_models():
    """Test dynamic fetching of Google Gemini models"""
    print("=== Testing Dynamic Google Gemini Model Fetching ===")
    print()
    
    # Check if Google API key is available
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("❌ GOOGLE_API_KEY not found in environment variables")
        print("Please set your Google API key:")
        print("export GOOGLE_API_KEY='your-api-key-here'")
        return
    
    print("✅ Google API key found")
    print()
    
    try:
        # Use the existing fetch_google_models method
        print("🔄 Fetching Google Gemini models dynamically...")
        models_config = ModelFetcher.fetch_google_models()
        
        # Display results
        chat_models = models_config.get('chat_models', {})
        embedding_models = models_config.get('embedding_models', {})
        
        print(f"✅ Successfully fetched {len(chat_models)} chat models and {len(embedding_models)} embedding models")
        print()
        
        print("📝 Available Chat Models:")
        for model_name, config in chat_models.items():
            print(f"   - {model_name}")
            print(f"     Temperature: {config.get('temperature', 'N/A')}")
            print(f"     Max Tokens: {config.get('max_output_tokens', 'N/A')}")
        
        print()
        print("🔍 Available Embedding Models:")
        for model_name, config in embedding_models.items():
            print(f"   - {model_name}")
            print(f"     Dimensions: {config.get('dimensions', 'N/A')}")
        
        print()
        print("🎯 How this works:")
        print("1. Uses google.generativeai.list_models() to fetch all available models")
        print("2. Filters chat models by checking 'generateContent' in supported_generation_methods")
        print("3. Filters embedding models by checking 'embedContent' in supported_generation_methods")
        print("4. Returns structured configuration for each model type")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please install the Google Generative AI package:")
        print("pip install google-generativeai")
        
    except Exception as e:
        print(f"❌ Error fetching models: {e}")
        print("Please check your API key and internet connection")

if __name__ == "__main__":
    test_gemini_models()