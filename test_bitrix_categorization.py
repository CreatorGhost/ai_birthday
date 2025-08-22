#!/usr/bin/env python3
"""
Test script to demonstrate Bitrix lead categorization
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system.user_tracker import UserTracker
from rag_system.model_fetcher import ModelFetcher

load_dotenv()

def test_categorization():
    """Test the conversation categorization logic"""
    
    print("🧪 Testing Bitrix Lead Categorization\n")
    
    try:
        # Initialize model and user tracker
        model_fetcher = ModelFetcher()
        llm = model_fetcher.get_llm()
        user_tracker = UserTracker(llm)
        
        # Test conversations
        test_cases = [
            {
                "name": "Birthday Party Inquiry",
                "conversation": [
                    {"role": "user", "content": "I want to plan a birthday party for my 5-year-old daughter"},
                    {"role": "assistant", "content": "Great! We have amazing birthday packages..."},
                    {"role": "user", "content": "What birthday decorations do you provide?"}
                ],
                "current_message": "How much does a birthday party package cost?",
                "expected_category": "birthday_party"
            },
            {
                "name": "General Park Inquiry", 
                "conversation": [
                    {"role": "user", "content": "What are your opening hours?"},
                    {"role": "assistant", "content": "We're open daily from 10 AM to 10 PM..."},
                    {"role": "user", "content": "How do I get to Yas Mall location?"}
                ],
                "current_message": "What activities do you have for kids?",
                "expected_category": "general"
            },
            {
                "name": "Mixed Conversation",
                "conversation": [
                    {"role": "user", "content": "Tell me about your park"},
                    {"role": "assistant", "content": "Leo & Loona is a magical family park..."},
                    {"role": "user", "content": "We're planning a birthday celebration"}
                ],
                "current_message": "Do you have party rooms available?",
                "expected_category": "birthday_party"
            }
        ]
        
        print("Testing conversation categorization:\n")
        
        for i, test in enumerate(test_cases, 1):
            print(f"🔍 Test {i}: {test['name']}")
            print(f"   Current message: '{test['current_message']}'")
            
            # Analyze the conversation
            result = user_tracker.analyze_conversation_category(
                test['conversation'], 
                test['current_message']
            )
            
            category = result.get('category', 'unknown')
            confidence = result.get('confidence', 0.0)
            reasoning = result.get('reasoning', '')
            keywords = result.get('keywords_found', [])
            
            print(f"   📊 Category: {category}")
            print(f"   🎯 Confidence: {confidence:.1%}")
            print(f"   💭 Reasoning: {reasoning}")
            print(f"   🔑 Keywords: {', '.join(keywords) if keywords else 'None'}")
            
            # Check if it matches expected
            if category == test['expected_category']:
                print(f"   ✅ CORRECT - Expected {test['expected_category']}")
            else:
                print(f"   ❌ INCORRECT - Expected {test['expected_category']}, got {category}")
            
            # Show Bitrix routing
            if category == 'birthday_party' and confidence > 0.7:
                bitrix_status = "NEW APPROACH"
                emoji = "🎂"
            else:
                bitrix_status = "GENERAL QUESTIONS"
                emoji = "💬"
            
            print(f"   🎯 Bitrix Route: {emoji} {bitrix_status}")
            print()
        
        print("✅ Categorization testing completed!")
        print("\n📋 Summary:")
        print("- Birthday party conversations → NEW APPROACH stage")
        print("- General inquiries → GENERAL QUESTIONS stage")  
        print("- High confidence (>70%) required for birthday categorization")
        print("- Full conversation context analyzed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    test_categorization()
