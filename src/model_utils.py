#!/usr/bin/env python3
"""
Utility script for managing and testing LLM model configurations.

Usage:
    python -m src.model_utils --list-providers
    python -m src.model_utils --test-config
    python -m src.model_utils --show-current
"""

import argparse
import os
from typing import Dict, Any
from dotenv import load_dotenv

from .config import ModelConfig, LLMProvider, get_available_providers

def list_providers():
    """List all available providers and their models"""
    print("\n=== Available LLM Providers and Models ===")
    
    providers = get_available_providers()
    
    for provider_name, config in providers.items():
        print(f"\n🔹 Provider: {provider_name.upper()}")
        
        print("   📝 Chat Models:")
        for model in config['chat_models']:
            print(f"      - {model}")
        
        print("   🔍 Embedding Models:")
        for model in config['embedding_models']:
            print(f"      - {model}")
        print("-" * 50)

def show_current_config():
    """Show current configuration"""
    try:
        config = ModelConfig()
        
        print("\n=== Current Configuration ===")
        print(f"🔹 Provider: {config.llm_provider.value.upper()}")
        print(f"📝 Chat Model: {config.chat_model}")
        print(f"🔍 Embedding Model: {config.embedding_model}")
        print(f"📊 Embedding Dimensions: {config.get_embedding_dimensions()}")
        
        print("\n📋 Chat Model Config:")
        chat_config = config.get_chat_model_config()
        for key, value in chat_config.items():
            print(f"   {key}: {value}")
        
        print("\n📋 Embedding Model Config:")
        embedding_config = config.get_embedding_model_config()
        for key, value in embedding_config.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ Error loading configuration: {str(e)}")
        print("\n💡 Make sure you have set the required environment variables.")
        print("   Check your .env file or run with --list-providers to see requirements.")

def test_config():
    """Test the current configuration by creating model instances"""
    try:
        print("\n=== Testing Configuration ===")
        config = ModelConfig()
        
        print("✅ Configuration loaded successfully")
        
        # Test embeddings
        print("🔍 Testing embeddings model...")
        embeddings = config.create_embedding_model()
        print("✅ Embeddings model created successfully")
        
        # Test chat model
        print("📝 Testing chat model...")
        chat_llm = config.create_chat_llm()
        print("✅ Chat model created successfully")
        
        # Test a simple embedding
        print("🧪 Testing embedding generation...")
        test_embedding = embeddings.embed_query("Hello, world!")
        print(f"✅ Generated embedding with {len(test_embedding)} dimensions")
        
        # Test a simple chat completion
        print("🧪 Testing chat completion...")
        response = chat_llm.invoke("Say 'Hello, world!' in a friendly way.")
        print(f"✅ Chat response: {response.content[:100]}...")
        
        print("\n🎉 All tests passed! Your configuration is working correctly.")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check that all required API keys are set")
        print("   2. Verify your internet connection")
        print("   3. Ensure the selected models are available")
        print("   4. Run --show-current to see your current configuration")

def main():
    """Main CLI function"""
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="LLM Model Configuration Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.model_utils --list-providers    # Show all available providers
  python -m src.model_utils --show-current      # Show current configuration
  python -m src.model_utils --test-config       # Test current configuration

Environment Variables:
  Set LLM_PROVIDER to 'openai' or 'google_gemini'
  Set appropriate API keys and model names in your .env file
        """
    )
    
    parser.add_argument(
        '--list-providers', 
        action='store_true',
        help='List all available LLM providers and their models'
    )
    
    parser.add_argument(
        '--show-current',
        action='store_true', 
        help='Show current model configuration'
    )
    
    parser.add_argument(
        '--test-config',
        action='store_true',
        help='Test the current configuration by creating model instances'
    )
    
    args = parser.parse_args()
    
    if args.list_providers:
        list_providers()
    elif args.show_current:
        show_current_config()
    elif args.test_config:
        test_config()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()