import os
import json
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class UserTracker:
    """Handles user information extraction, storage, and name collection for testing"""
    
    def __init__(self, llm, storage_dir: str = "user_data"):
        self.llm = llm
        self.storage_dir = storage_dir
        self.conversations_file = os.path.join(storage_dir, "user_conversations.txt")
        self.profiles_file = os.path.join(storage_dir, "user_profiles.txt")
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize files if they don't exist
        if not os.path.exists(self.conversations_file):
            with open(self.conversations_file, 'w') as f:
                f.write("# User Conversations Log\n")
                f.write("# Format: [Timestamp] | Phone: [phone] | Name: [name] | Message: [message]\n\n")
        
        if not os.path.exists(self.profiles_file):
            with open(self.profiles_file, 'w') as f:
                f.write("# User Profiles\n")
                f.write("# Format: Phone: [phone] | Name: [name] | Last_Seen: [timestamp] | Total_Messages: [count]\n\n")
    
    def generate_test_phone_number(self) -> str:
        """Generate a random phone number for WhatsApp simulation"""
        # UAE phone number format: +971XXXXXXXXX
        uae_prefixes = ['50', '52', '54', '55', '56', '58']
        prefix = random.choice(uae_prefixes)
        number = ''.join([str(random.randint(0, 9)) for _ in range(7)])
        return f"+971{prefix}{number}"
    
    def extract_name_from_message(self, message: str, conversation_context: List[Dict] = None) -> Dict:
        """Use LLM to extract user name from message"""
        
        # Check if this might be a response to a name request
        recent_bot_messages = []
        if conversation_context:
            recent_bot_messages = [msg.get('content', '') for msg in conversation_context[-3:] 
                                 if msg.get('role') == 'assistant']
        
        prompt = PromptTemplate(
            template="""You are extracting user names from chat messages. Look for clear name indicators.

Recent bot messages: {recent_messages}

User message: "{message}"

Extract the name if the user is clearly providing their name. Look for patterns like:
- "I'm John" / "I am Sarah"
- "My name is Mike"
- "Call me Lisa"
- "It's David"
- Just a name in response to a name question
- "Hi, I'm..." or "Hello, I'm..."

Respond with JSON:
{{
    "name_found": true/false,
    "extracted_name": "name or null",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Only extract if you're confident (>0.7) that the user is providing their name.""",
            input_variables=["message", "recent_messages"]
        )
        
        recent_context = " | ".join(recent_bot_messages[-2:]) if recent_bot_messages else "None"
        
        try:
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "message": message,
                "recent_messages": recent_context
            })
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                return {"name_found": False, "extracted_name": None, "confidence": 0.0, "reasoning": "Could not parse response"}
                
        except Exception as e:
            print(f"Error extracting name: {e}")
            return {"name_found": False, "extracted_name": None, "confidence": 0.0, "reasoning": f"Error: {str(e)}"}
    
    def should_request_name(self, conversation_history: List[Dict], current_message: str, user_profile: Dict) -> Dict:
        """Determine if and how to request user's name"""
        
        # Don't ask if we already have a name
        if user_profile.get('name'):
            return {"should_ask": False, "reason": "Name already known"}
        
        # NEW: Ask for name when user asks first question (more natural)
        if len(conversation_history) == 0:
            return {
                "should_ask": True, 
                "confidence": 1.0,
                "approach": "polite_request",
                "timing": "after_question",
                "reasoning": "First question asked - politely request name before answering"
            }
        
        # Check if we recently asked for name
        recent_messages = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history
        for msg in recent_messages:
            if msg.get('role') == 'assistant' and any(phrase in msg.get('content', '').lower() 
                for phrase in ['what\'s your name', 'what is your name', 'name so i can', 'your name?']):
                return {"should_ask": False, "reason": "Recently asked for name"}
        
        prompt = PromptTemplate(
            template="""Analyze this conversation to determine if now is a good time to ask for the user's name.

Conversation history:
{conversation}

Current user message: "{current_message}"

Good opportunities to ask for name:
- User shows serious interest (asks about booking, prices, specific details)
- User asks multiple questions (engaged conversation)
- User mentions planning a visit or event
- User asks about group activities or special events
- Conversation has been going for 2+ meaningful exchanges
- User seems ready for personalized help

Bad opportunities:
- User just started conversation
- User seems to be casually browsing
- User just asked a simple FAQ question
- User seems hesitant or unsure
- We just asked a question and should wait for response

Respond with JSON:
{{
    "should_ask": true/false,
    "confidence": 0.0-1.0,
    "approach": "personalization/follow_up/booking/assistance",
    "timing": "now/after_response/later",
    "reasoning": "brief explanation"
}}""",
            input_variables=["conversation", "current_message"]
        )
        
        # Format conversation history
        conversation_text = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
            for msg in conversation_history[-5:]  # Last 5 messages for context
        ])
        
        try:
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "conversation": conversation_text,
                "current_message": current_message
            })
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                return {"should_ask": False, "confidence": 0.0, "reasoning": "Could not parse response"}
                
        except Exception as e:
            print(f"Error in name request analysis: {e}")
            return {"should_ask": False, "confidence": 0.0, "reasoning": f"Error: {str(e)}"}
    
    def generate_name_request(self, approach: str = "personalization") -> str:
        """Generate natural name collection message based on approach"""
        
        name_requests = {
            "polite_request": [
                "Hey, happy to see you! 😊 Can you tell me your name so that I can assist you better?",
                "Hello there! 👋 I'd love to help you with that. May I have your name so I can give you personalized assistance?",
                "Hi! 😊 I'm excited to help you. Could you please tell me your name so I can assist you better?"
            ],
            "welcome": [
                "Hello! 👋 Welcome to Leo & Loona!\n\nI'm your friendly virtual host for our magical family amusement parks in Abu Dhabi and Dubai! ✨\n\nTo give you the best personalized assistance, may I please know your name?",
                "Welcome to Leo & Loona! 🎪\n\nI'm here to help you discover our magical world of fun and adventure!\n\nMay I have your name so I can provide you with personalized assistance?",
                "Hi there! 👋 Welcome to Leo & Loona!\n\nI'm your virtual assistant, ready to help you with all things Leo & Loona! ✨\n\nWhat's your name so I can assist you better?"
            ],
            "personalization": [
                "I'd love to help you better! What's your name so I can personalize my recommendations?",
                "What's your name? I'd like to make this conversation more personal!",
                "By the way, what's your name so I can give you more tailored suggestions?"
            ],
            "follow_up": [
                "What's your name so our team can follow up with you?",
                "I'd like to connect you with our specialists. What's your name?",
                "What's your name? I can have someone from our team reach out with more details!"
            ],
            "booking": [
                "Great! What's your name so I can help you get started with booking?",
                "Wonderful! What's your name? I'll help you with the next steps!",
                "Perfect! Let me get your name to start preparing your booking details."
            ],
            "assistance": [
                "What's your name? I'd love to provide you with better assistance!",
                "May I have your name so I can help you more effectively?",
                "What should I call you? I want to make sure I'm giving you the best help possible!"
            ]
        }
        
        return random.choice(name_requests.get(approach, name_requests["personalization"]))
    
    def generate_personalized_greeting(self, name: str) -> str:
        """Generate simple personalized greeting after name is provided"""
        
        greetings = [
            f"Thank you, {name}! 😊",
            f"Nice to meet you, {name}! ✨", 
            f"Wonderful, {name}! 😊",
            f"Great to meet you, {name}! 🎠"
        ]
        
        return random.choice(greetings)
    
    def analyze_conversation_category(self, conversation_history: List[Dict], current_message: str) -> Dict:
        """Analyze conversation to determine Bitrix category (GENERAL QUESTIONS vs NEW APPROACH)"""
        
        # Combine all conversation messages for analysis
        all_messages = []
        for msg in conversation_history:
            if msg.get('content'):
                all_messages.append(msg['content'])
        all_messages.append(current_message)
        
        conversation_text = " ".join(all_messages)
        
        prompt = f"""Analyze this conversation to determine if it's related to birthday parties or general park inquiries.

Conversation: "{conversation_text}"

Birthday Party Indicators:
- Mentions "birthday", "party", "celebration", "celebrate"
- Asks about birthday packages, party rooms, decorations
- Mentions age, kids birthday, party planning
- Asks about group bookings for celebrations
- Special event planning

General Inquiries:
- General park information, hours, prices
- Location questions, directions
- Regular family visits
- General activities and attractions
- Safety questions, policies

Respond with JSON:
{{
    "category": "birthday_party" or "general",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "keywords_found": ["list", "of", "relevant", "keywords"]
}}

Only choose "birthday_party" if clearly related to birthday celebrations."""

        try:
            chain = PromptTemplate.from_template(prompt) | self.llm | StrOutputParser()
            response = chain.invoke({})
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result
            else:
                return {"category": "general", "confidence": 0.5, "reasoning": "Could not parse response", "keywords_found": []}
                
        except Exception as e:
            print(f"Error analyzing conversation category: {e}")
            return {"category": "general", "confidence": 0.5, "reasoning": f"Error: {str(e)}", "keywords_found": []}
    
    def store_original_question(self, phone: str, question: str):
        """Store the original question until name is provided"""
        try:
            # Store in a simple format for now
            filename = f'user_data/pending_question_{phone.replace("+", "").replace(" ", "")}.txt'
            with open(filename, 'w') as f:
                f.write(question)

        except Exception as e:
            print(f"Error storing original question: {e}")
    
    def get_stored_question(self, phone: str) -> str:
        """Retrieve the stored original question"""
        try:
            filename = f'user_data/pending_question_{phone.replace("+", "").replace(" ", "")}.txt'
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    question = f.read().strip()
                # Clean up the file after reading
                os.remove(filename)
                return question
        except Exception as e:
            print(f"Error retrieving stored question: {e}")
        return ""
    
    def is_first_interaction(self, conversation_history: List[Dict]) -> bool:
        """Check if this is the very first interaction"""
        return len(conversation_history) == 0
    
    def is_name_response(self, message: str, conversation_history: List[Dict]) -> bool:
        """Check if this message is likely a response to a name request"""
        
        # Check if the last bot message was asking for name
        if conversation_history:
            last_bot_message = None
            for msg in reversed(conversation_history):
                if msg.get('role') == 'assistant':
                    last_bot_message = msg.get('content', '').lower()
                    break
            
            if last_bot_message:
                name_request_indicators = [
                    'what\'s your name', 'may i know your name', 'may i have your name',
                    'your name so i can', 'name so i can assist', 'may i please know your name',
                    'can you tell me your name', 'tell me your name'  # Added more patterns
                ]
                
                for indicator in name_request_indicators:
                    if indicator in last_bot_message:
                        return True
        
        return False
    
    def log_conversation(self, phone: str, name: Optional[str], message: str, is_user: bool = True):
        """Log conversation to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        role = "USER" if is_user else "BOT"
        name_display = name or "Unknown"
        
        log_entry = f"[{timestamp}] | Phone: {phone} | Name: {name_display} | {role}: {message}\n"
        
        with open(self.conversations_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def get_user_profile(self, phone: str) -> Dict:
        """Get user profile by phone number"""
        try:
            with open(self.profiles_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.startswith('#') or not line.strip():
                    continue
                
                if f"Phone: {phone}" in line:
                    # Parse existing profile
                    parts = line.strip().split(' | ')
                    profile = {"phone": phone, "name": None, "last_seen": None, "total_messages": 0}
                    
                    for part in parts:
                        if part.startswith("Name: "):
                            name = part.replace("Name: ", "").strip()
                            profile["name"] = name if name != "Unknown" else None
                        elif part.startswith("Last_Seen: "):
                            profile["last_seen"] = part.replace("Last_Seen: ", "").strip()
                        elif part.startswith("Total_Messages: "):
                            profile["total_messages"] = int(part.replace("Total_Messages: ", "").strip())
                    
                    return profile
            
            # New user
            return {"phone": phone, "name": None, "last_seen": None, "total_messages": 0}
            
        except Exception as e:
            print(f"Error reading user profile: {e}")
            return {"phone": phone, "name": None, "last_seen": None, "total_messages": 0}
    
    def update_user_profile(self, phone: str, name: Optional[str] = None):
        """Update user profile with new information"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Read existing profiles
            profiles = []
            if os.path.exists(self.profiles_file):
                with open(self.profiles_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    if line.startswith('#') or not line.strip():
                        profiles.append(line)
                        continue
                    
                    if f"Phone: {phone}" in line:
                        # Update existing profile
                        current_profile = self.get_user_profile(phone)
                        current_profile["last_seen"] = timestamp
                        current_profile["total_messages"] += 1
                        if name:
                            current_profile["name"] = name
                        
                        profile_line = f"Phone: {phone} | Name: {current_profile['name'] or 'Unknown'} | Last_Seen: {timestamp} | Total_Messages: {current_profile['total_messages']}\n"
                        profiles.append(profile_line)
                    else:
                        profiles.append(line)
            
            # If user not found, add new profile
            user_found = any(f"Phone: {phone}" in line for line in profiles if not line.startswith('#'))
            if not user_found:
                profile_line = f"Phone: {phone} | Name: {name or 'Unknown'} | Last_Seen: {timestamp} | Total_Messages: 1\n"
                profiles.append(profile_line)
            
            # Write back to file
            with open(self.profiles_file, 'w', encoding='utf-8') as f:
                f.writelines(profiles)
                
        except Exception as e:
            print(f"Error updating user profile: {e}")
    
    def get_conversation_summary(self, phone: str) -> str:
        """Get a summary of conversations for this user"""
        try:
            profile = self.get_user_profile(phone)
            return f"User: {profile['name'] or 'Unknown'} | Phone: {phone} | Messages: {profile['total_messages']} | Last seen: {profile['last_seen'] or 'Never'}"
        except Exception as e:
            return f"Phone: {phone} | Error loading profile: {str(e)}"
