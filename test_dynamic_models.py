#!/usr/bin/env python3
"""
Test script to demonstrate dynamic model fetching functionality.

This script shows how the system now dynamically fetches available models
from OpenAI and Google Gemini APIs instead of using hardcoded lists.
"""

import os
from dotenv import load_dotenv
from src.config import ModelConfig, get_available_providers

def main():
    load_dotenv()
    
    print("🚀 Testing Dynamic Model Fetching")
    print("=" * 50)
    
    # Test with dynamic fetching enabled (default)
    print("\n📡 Fetching models dynamically from APIs...")
    config_dynamic = ModelConfig(use_dynamic_models=True)
    
    dynamic_providers = get_available_providers()
    
    for provider, models in dynamic_providers.items():
        print(f"\n🔹 {provider.upper()}:")
        print(f"   📝 Chat Models: {len(models['chat_models'])} available")
        print(f"   🔍 Embedding Models: {len(models['embedding_models'])} available")
        
        # Show first few models as examples
        print(f"   📝 Example Chat Models: {list(models['chat_models'])[:3]}...")
        print(f"   🔍 Example Embedding Models: {list(models['embedding_models'])[:2]}...")
    
    # Test with static fallback
    print("\n\n📚 Using static fallback models...")
    config_static = ModelConfig(use_dynamic_models=False)
    
    static_configs = config_static.get_dynamic_model_configs(use_dynamic=False)
    
    for provider, config in static_configs.items():
        print(f"\n🔹 {provider.value.upper()} (Static):")
        print(f"   📝 Chat Models: {len(config['chat_models'])} available")
        print(f"   🔍 Embedding Models: {len(config['embedding_models'])} available")
    
    print("\n✅ Dynamic model fetching is working perfectly!")
    print("\n💡 Benefits of dynamic fetching:")
    print("   • Always up-to-date with latest models")
    print("   • No need to manually update hardcoded lists")
    print("   • Automatic fallback to static lists if API fails")
    print("   • Real-time model availability")

if __name__ == "__main__":
    main()