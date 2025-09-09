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
        
        # 🔥 LANGGRAPH BEST PRACTICE: Session-based message history
        self.session_messages = {}  # In-memory storage for session messages
        
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
    
    def get_phone_number_for_session(self, manual_phone: str = None) -> str:
        """Get phone number for current session - manual for testing or generated"""
        if manual_phone and manual_phone.strip():
            # Use manual phone number provided by user (e.g., from Streamlit input)
            phone = manual_phone.strip()
            # Ensure it starts with + if it doesn't already
            if not phone.startswith('+'):
                phone = '+' + phone
            return phone
        else:
            # Generate random phone number for testing
            return self.generate_test_phone_number()
    
    def extract_name_from_message(self, message: str, conversation_context: List[Dict] = None) -> Dict:
        """Enhanced LLM-based name extraction with better typo handling"""
        
        # Check if this might be a response to a name request
        recent_bot_messages = []
        if conversation_context:
            recent_bot_messages = [msg.get('content', '') for msg in conversation_context[-3:] 
                                 if msg.get('role') == 'assistant']
        
        recent_context = " | ".join(recent_bot_messages[-2:]) if recent_bot_messages else "None"
        
        prompt = f"""Extract user name from this message. Handle typos, cultural names, and variations.

RECENT BOT MESSAGES: {recent_context}
USER MESSAGE: "{message}"

EXTRACT NAME from patterns like:
- "I'm John" / "I am Sarah" / "My name is Mike"
- "Call me Lisa" / "It's David" / "Hi, I'm..."
- Just a name in response to name questions
- Cultural names (Arabic, Indian, etc.)
- Names with typos or unusual spellings
- Combined messages like "I'm Ahmed, ys mall hours?"

IMPORTANT: Only extract if user is clearly providing their name, not mentioning someone else.

Return ONLY valid JSON:
{{
    "name_found": true/false,
    "extracted_name": "name or null",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

EXAMPLES:
- "I'm Ahmed" → {{"name_found": true, "extracted_name": "Ahmed", "confidence": 0.9, "reasoning": "Clear name introduction"}}
- "My daughter Sarah wants to know" → {{"name_found": false, "extracted_name": null, "confidence": 0.0, "reasoning": "Referring to someone else"}}
- "Call me Mike please" → {{"name_found": true, "extracted_name": "Mike", "confidence": 0.8, "reasoning": "Clear name request"}}

Only extract if confidence > 0.7 that user is providing THEIR OWN name."""

        try:
            from langchain_core.messages import HumanMessage
            from langchain_core.output_parsers import StrOutputParser
            
            # Use JSON mode LLM if available
            llm_to_use = getattr(self, 'llm_json_mode', self.llm)
            
            response = llm_to_use.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()
            
            # Parse JSON response with better error handling
            import json
            import re
            
            # Try to find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                
                # Validate and clean response
                name_found = result.get("name_found", False)
                extracted_name = result.get("extracted_name")
                confidence = result.get("confidence", 0.0)
                reasoning = result.get("reasoning", "LLM extraction")
                
                # Clean extracted name
                if extracted_name and isinstance(extracted_name, str):
                    extracted_name = extracted_name.strip().title()
                    if extracted_name.lower() in ['null', 'none', '']:
                        extracted_name = None
                        name_found = False
                else:
                    extracted_name = None
                    name_found = False
                
                return {
                    "name_found": name_found,
                    "extracted_name": extracted_name,
                    "confidence": float(confidence),
                    "reasoning": reasoning
                }
            
            else:
                print(f"⚠️ Name extraction: No JSON found in response: {response_text}")
                return {
                    "name_found": False,
                    "extracted_name": None,
                    "confidence": 0.0,
                    "reasoning": "LLM extraction failed - no JSON found"
                }
                
        except Exception as e:
            print(f"❌ Name extraction error: {str(e)}")
            return {
                "name_found": False,
                "extracted_name": None,
                "confidence": 0.0,
                "reasoning": f"LLM extraction error: {str(e)}"
            }
    
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
        
        prompt_template = """Analyze this conversation to determine if it's related to birthday parties or general park inquiries.

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
{{{{
    "category": "birthday_party" or "general",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "keywords_found": ["list", "of", "relevant", "keywords"]
}}}}

Only choose "birthday_party" if clearly related to birthday celebrations."""
        
        prompt = prompt_template.format(conversation_text=conversation_text)

        try:
            # Use the LLM directly since we already formatted the prompt
            response = self.llm.invoke([{"role": "user", "content": prompt}]).content
            
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
    
    def store_original_question(self, phone: str, question: str, detected_mall: str = None):
        """Store the original question and detected mall until name is provided"""
        try:
            # Store question and mall information in JSON format
            import json
            filename = f'user_data/pending_question_{phone.replace("+", "").replace(" ", "")}.txt'
            
            import datetime
            question_data = {
                "question": question,
                "detected_mall": detected_mall,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(question_data, f)

        except Exception as e:
            print(f"Error storing original question: {e}")
    
    def get_stored_question(self, phone: str) -> str:
        """Retrieve the stored original question (non-destructive read)"""
        try:
            filename = f'user_data/pending_question_{phone.replace("+", "").replace(" ", "")}.txt'
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    content = f.read().strip()
                    
                # Try to parse as JSON (new format)
                try:
                    import json
                    question_data = json.loads(content)
                    return question_data.get("question", "")
                except json.JSONDecodeError:
                    # Fallback for old format (plain text)
                    return content
        except Exception as e:
            print(f"Error retrieving stored question: {e}")
        return ""
    
    def get_stored_question_info(self, phone: str) -> dict:
        """Retrieve the full stored question information including detected mall"""
        try:
            filename = f'user_data/pending_question_{phone.replace("+", "").replace(" ", "")}.txt'
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    content = f.read().strip()
                    
                # Try to parse as JSON (new format)
                try:
                    import json
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Fallback for old format (plain text)
                    return {"question": content, "detected_mall": None}
            return {}
        except Exception as e:
            print(f"Error retrieving stored question info: {e}")
            return {}
    
    def clear_stored_question(self, phone: str):
        """Clear the stored question after it has been answered"""
        try:
            filename = f'user_data/pending_question_{phone.replace("+", "").replace(" ", "")}.txt'
            if os.path.exists(filename):
                os.remove(filename)
                print(f"✅ Cleared stored question for {phone}")
        except Exception as e:
            print(f"Error clearing stored question: {e}")
    
    def has_stored_question(self, phone: str) -> bool:
        """Check if there's a stored question without reading it"""
        try:
            filename = f'user_data/pending_question_{phone.replace("+", "").replace(" ", "")}.txt'
            return os.path.exists(filename)
        except Exception as e:
            print(f"Error checking stored question: {e}")
            return False
    
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
                    profile = {
                        "phone": phone, 
                        "name": None, 
                        "last_seen": None, 
                        "total_messages": 0,
                        "bitrix_lead_id": None,
                        "lead_created_at": None,
                        "lead_updated_at": None,
                        "original_park_location": None,
                        "current_park_location": None,
                        "name_refusal_count": 0,
                        "greeted": False
                    }
                    
                    for part in parts:
                        if part.startswith("Name: "):
                            name = part.replace("Name: ", "").strip()
                            profile["name"] = name if name != "Unknown" else None
                        elif part.startswith("Last_Seen: "):
                            profile["last_seen"] = part.replace("Last_Seen: ", "").strip()
                        elif part.startswith("Total_Messages: "):
                            profile["total_messages"] = int(part.replace("Total_Messages: ", "").strip())
                        elif part.startswith("Lead_ID: "):
                            lead_id = part.replace("Lead_ID: ", "").strip()
                            profile["bitrix_lead_id"] = lead_id if lead_id != "None" else None
                        elif part.startswith("Lead_Created: "):
                            profile["lead_created_at"] = part.replace("Lead_Created: ", "").strip()
                        elif part.startswith("Lead_Updated: "):
                            profile["lead_updated_at"] = part.replace("Lead_Updated: ", "").strip()
                        elif part.startswith("Original_Park: "):
                            park = part.replace("Original_Park: ", "").strip()
                            profile["original_park_location"] = park if park != "None" else None
                        elif part.startswith("Current_Park: "):
                            park = part.replace("Current_Park: ", "").strip()
                            profile["current_park_location"] = park if park != "None" else None
                        elif part.startswith("Name_Refusal_Count: "):
                            count = part.replace("Name_Refusal_Count: ", "").strip()
                            profile["name_refusal_count"] = int(count) if count.isdigit() else 0
                        elif part.startswith("Greeted: "):
                            greeted = part.replace("Greeted: ", "").strip()
                            profile["greeted"] = greeted.lower() == "true"
                    
                    return profile
            
            # New user
            return {
                "phone": phone, 
                "name": None, 
                "last_seen": None, 
                "total_messages": 0,
                "bitrix_lead_id": None,
                "lead_created_at": None,
                "lead_updated_at": None,
                "original_park_location": None,  # Track the first park they were assigned to
                "current_park_location": None,    # Track current park (can be updated)
                "name_refusal_count": 0,
                "greeted": False
            }
            
        except Exception as e:
            print(f"Error reading user profile: {e}")
            return {"phone": phone, "name": None, "last_seen": None, "total_messages": 0}
    
    def increment_name_refusal_count(self, phone: str) -> int:
        """Increment name refusal count and return current count"""
        profile = self.get_user_profile(phone)
        current_count = profile.get("name_refusal_count", 0)
        new_count = current_count + 1
        
        # Update profile with new refusal count
        self.update_user_profile(phone, name_refusal_count=new_count)
        return new_count
    
    def get_name_refusal_count(self, phone: str) -> int:
        """Get current name refusal count for user"""
        profile = self.get_user_profile(phone)
        return profile.get("name_refusal_count", 0)
    
    def reset_name_refusal_count(self, phone: str):
        """Reset name refusal count (when user finally provides name)"""
        self.update_user_profile(phone, name_refusal_count=0)
    
    # 🔥 LANGGRAPH BEST PRACTICE: Session-based message history management
    def add_user_message(self, session_id: str, message: str):
        """Add a user message to the session history"""
        if session_id not in self.session_messages:
            self.session_messages[session_id] = []
        
        self.session_messages[session_id].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_ai_message(self, session_id: str, message: str):
        """Add an AI message to the session history"""
        if session_id not in self.session_messages:
            self.session_messages[session_id] = []
        
        self.session_messages[session_id].append({
            "role": "assistant",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_session_history(self, session_id: str, last_n: int = 10) -> List[dict]:
        """Get the message history for a session"""
        if session_id not in self.session_messages:
            return []
        
        # Return the last N messages
        return self.session_messages[session_id][-last_n:] if last_n else self.session_messages[session_id]
    
    def clear_session_history(self, session_id: str):
        """Clear the message history for a session"""
        if session_id in self.session_messages:
            del self.session_messages[session_id]

    def update_user_profile(self, phone: str, name: Optional[str] = None, name_refusal_count: Optional[int] = None, greeted: Optional[bool] = None):
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
                        if name_refusal_count is not None:
                            current_profile["name_refusal_count"] = name_refusal_count
                        if greeted is not None:
                            current_profile["greeted"] = greeted
                        
                        profile_line = f"Phone: {phone} | Name: {current_profile['name'] or 'Unknown'} | Last_Seen: {timestamp} | Total_Messages: {current_profile['total_messages']} | Lead_ID: {current_profile.get('bitrix_lead_id', 'None')} | Lead_Created: {current_profile.get('lead_created_at', 'None')} | Lead_Updated: {current_profile.get('lead_updated_at', 'None')} | Original_Park: {current_profile.get('original_park_location', 'None')} | Current_Park: {current_profile.get('current_park_location', 'None')} | Name_Refusal_Count: {current_profile.get('name_refusal_count', 0)} | Greeted: {current_profile.get('greeted', False)}\n"
                        profiles.append(profile_line)
                    else:
                        profiles.append(line)
            
            # If user not found, add new profile
            user_found = any(f"Phone: {phone}" in line for line in profiles if not line.startswith('#'))
            if not user_found:
                profile_line = f"Phone: {phone} | Name: {name or 'Unknown'} | Last_Seen: {timestamp} | Total_Messages: 1 | Lead_ID: None | Lead_Created: None | Lead_Updated: None | Original_Park: None | Current_Park: None | Name_Refusal_Count: {name_refusal_count or 0} | Greeted: {greeted or False}\n"
                profiles.append(profile_line)
            
            # Write back to file
            with open(self.profiles_file, 'w', encoding='utf-8') as f:
                f.writelines(profiles)
                
        except Exception as e:
            print(f"Error updating user profile: {e}")
    
    def update_user_lead_info(self, phone: str, lead_id: str, action: str = "created", park_location: str = None):
        """Update user profile with Bitrix lead information and track park location"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Get current profile
            current_profile = self.get_user_profile(phone)
            
            # Update lead information
            if action == "created":
                current_profile["bitrix_lead_id"] = lead_id
                current_profile["lead_created_at"] = timestamp
                current_profile["lead_updated_at"] = timestamp
                
                # Track park location for first-time lead creation
                if park_location and park_location != "General":
                    current_profile["original_park_location"] = park_location
                    current_profile["current_park_location"] = park_location
                    print(f"📝 Recorded lead creation: User {phone} → Lead {lead_id} → Park: {park_location}")
                else:
                    print(f"📝 Recorded lead creation: User {phone} → Lead {lead_id}")
                    
            elif action == "updated":
                current_profile["lead_updated_at"] = timestamp
                
                # Update current park location if provided (but keep original)
                if park_location and park_location != "General":
                    current_profile["current_park_location"] = park_location
                    print(f"📝 Recorded lead update: User {phone} → Lead {lead_id} → New Park: {park_location}")
                else:
                    print(f"📝 Recorded lead update: User {phone} → Lead {lead_id}")
                    
            elif action == "mall_preference":
                # Store mall preference without lead_id
                if park_location and park_location != "General":
                    current_profile["current_park_location"] = park_location
                    print(f"📝 Stored mall preference: User {phone} → {park_location}")
                else:
                    print(f"📝 Stored general mall preference: User {phone}")
            
            # Read existing profiles
            profiles = []
            user_found = False
            
            if os.path.exists(self.profiles_file):
                with open(self.profiles_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    if line.startswith('#') or not line.strip():
                        profiles.append(line)
                        continue
                    
                    if f"Phone: {phone}" in line:
                        # Update the profile line with new lead info
                        profile_line = f"Phone: {phone} | Name: {current_profile['name'] or 'Unknown'} | Last_Seen: {current_profile.get('last_seen', 'None')} | Total_Messages: {current_profile.get('total_messages', 0)} | Lead_ID: {current_profile.get('bitrix_lead_id', 'None')} | Lead_Created: {current_profile.get('lead_created_at', 'None')} | Lead_Updated: {current_profile.get('lead_updated_at', 'None')} | Original_Park: {current_profile.get('original_park_location', 'None')} | Current_Park: {current_profile.get('current_park_location', 'None')}\n"
                        profiles.append(profile_line)
                        user_found = True
                    else:
                        profiles.append(line)
            
            # If user not found in file, create new entry
            if not user_found:
                profile_line = f"Phone: {phone} | Name: {current_profile['name'] or 'Unknown'} | Last_Seen: {current_profile.get('last_seen', timestamp)} | Total_Messages: {current_profile.get('total_messages', 0)} | Lead_ID: {current_profile.get('bitrix_lead_id', 'None')} | Lead_Created: {current_profile.get('lead_created_at', 'None')} | Lead_Updated: {current_profile.get('lead_updated_at', 'None')} | Original_Park: {current_profile.get('original_park_location', 'None')} | Current_Park: {current_profile.get('current_park_location', 'None')}\n"
                profiles.append(profile_line)
                print(f"📝 Created new profile entry for {phone} with lead {lead_id}")
            
            # Write back to file
            with open(self.profiles_file, 'w', encoding='utf-8') as f:
                f.writelines(profiles)
                
        except Exception as e:
            print(f"Error updating user lead info: {e}")
    
    def has_existing_lead(self, phone: str) -> bool:
        """Check if user already has a Bitrix lead created"""
        try:
            profile = self.get_user_profile(phone)
            return profile.get("bitrix_lead_id") is not None
        except Exception as e:
            print(f"Error checking existing lead: {e}")
            return False
    
    def get_conversation_summary(self, phone: str) -> str:
        """Get a summary of conversations for this user"""
        try:
            profile = self.get_user_profile(phone)
            return f"User: {profile['name'] or 'Unknown'} | Phone: {phone} | Messages: {profile['total_messages']} | Last seen: {profile['last_seen'] or 'Never'}"
        except Exception as e:
            return f"Phone: {phone} | Error loading profile: {str(e)}"
