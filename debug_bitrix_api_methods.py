#!/usr/bin/env python3
"""
Debug script to discover the correct Bitrix24 API methods for Open Channel messages.
Tests multiple API endpoints to find the right one for WhatsApp/Open Channel content.
"""
import json
from bitrix_integration.bitrix_client import BitrixClient
from bitrix_integration.config import BitrixConfig

def test_all_possible_api_methods():
    """Test various API methods that might contain Open Channel messages"""
    print("🔍 Discovering Bitrix24 API Methods for Open Channel Messages")
    print("=" * 60)
    
    try:
        # Initialize Bitrix client
        config = BitrixConfig()
        client = BitrixClient(config)
        
        if not client.test_connection():
            print("❌ Bitrix connection failed")
            return
        
        # Communication details from our test
        # "imol|olchat_wa_connector_2|1|971558505328|47"
        session_id = "47"
        connector_id = "olchat_wa_connector_2"
        line_id = "1"
        external_id = "971558505328"
        
        print(f"Testing with session_id: {session_id}")
        print(f"Testing with connector_id: {connector_id}")
        
        # List of API methods to test
        api_methods_to_test = [
            # ImConnector methods
            ("imconnector.send.messages", {}),
            ("imconnector.connector.list", {}),
            ("imconnector.status.get", {}),
            
            # Open Lines methods
            ("imopenlines.session.list", {}),
            ("imopenlines.session.get", {"SESSION_ID": session_id}),
            ("imopenlines.session.get", {"ID": session_id}),
            ("imopenlines.message.list", {"SESSION_ID": session_id}),
            ("imopenlines.message.list", {"CHAT_ID": session_id}),
            
            # IM (Instant Messaging) methods
            ("im.message.list", {"CHAT_ID": session_id}),
            ("im.message.list", {"CHAT_ID": f"chat{session_id}"}),
            ("im.chat.get", {"CHAT_ID": session_id}),
            ("im.chat.get", {"ID": session_id}),
            ("im.dialog.messages.get", {"DIALOG_ID": session_id}),
            
            # Live Chat methods
            ("livechat.operator.list", {}),
            ("livechat.session.list", {}),
            
            # Try with different parameter combinations
            ("imopenlines.session.get", {"CHAT_ID": session_id}),
            ("im.message.list", {"DIALOG_ID": f"imol|{session_id}"}),
            ("im.message.list", {"DIALOG_ID": f"imol|{connector_id}|{session_id}"}),
        ]
        
        successful_methods = []
        
        print(f"\n🧪 Testing {len(api_methods_to_test)} different API methods...")
        
        for i, (method, params) in enumerate(api_methods_to_test, 1):
            print(f"\n[{i:2d}] Testing: {method}")
            if params:
                print(f"    Params: {params}")
            
            try:
                response = client._make_request(method, params)
                
                if 'result' in response and response['result'] is not None:
                    print(f"    ✅ SUCCESS! Got data:")
                    result = response['result']
                    
                    if isinstance(result, list):
                        print(f"       Result type: List with {len(result)} items")
                        if result:  # Show first item structure
                            print(f"       First item keys: {list(result[0].keys()) if isinstance(result[0], dict) else 'Not a dict'}")
                    elif isinstance(result, dict):
                        print(f"       Result type: Dict with keys: {list(result.keys())}")
                    else:
                        print(f"       Result type: {type(result)} - {str(result)[:100]}...")
                    
                    successful_methods.append({
                        'method': method,
                        'params': params,
                        'result_type': type(result).__name__,
                        'response': response
                    })
                else:
                    print(f"    ⚠️ Empty result (but method exists)")
                    
            except Exception as e:
                error_msg = str(e)
                if "METHOD_NOT_FOUND" in error_msg:
                    print(f"    ❌ Method not available in your Bitrix")
                elif "ACCESS_DENIED" in error_msg:
                    print(f"    ❌ Access denied - need different permissions")
                elif "WRONG_REQUEST" in error_msg:
                    print(f"    ❌ Wrong request parameters")
                else:
                    print(f"    ❌ Error: {error_msg}")
        
        # Test some general discovery methods
        print(f"\n🔍 Testing Discovery Methods...")
        
        discovery_methods = [
            ("methods", {}),  # List all available methods
            ("scope", {}),    # List available scopes
            ("app.info", {}), # App information
        ]
        
        for method, params in discovery_methods:
            print(f"\n🔍 {method}:")
            try:
                response = client._make_request(method, params)
                if 'result' in response:
                    result = response['result']
                    if isinstance(result, list) and 'methods' in method:
                        # Filter methods related to messaging
                        messaging_methods = [m for m in result if any(keyword in m.lower() for keyword in ['im', 'chat', 'message', 'openlines', 'connector'])]
                        print(f"    Found {len(messaging_methods)} messaging-related methods:")
                        for mm in sorted(messaging_methods)[:10]:  # Show first 10
                            print(f"      • {mm}")
                        if len(messaging_methods) > 10:
                            print(f"      ... and {len(messaging_methods) - 10} more")
                    else:
                        print(f"    Result: {str(result)[:200]}...")
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        # Results summary
        print(f"\n📊 DISCOVERY RESULTS:")
        print(f"=" * 40)
        
        if successful_methods:
            print(f"✅ Found {len(successful_methods)} working API methods!")
            
            for method_info in successful_methods:
                print(f"\n🎯 {method_info['method']}")
                print(f"   Params: {method_info['params']}")
                print(f"   Result Type: {method_info['result_type']}")
                
                # Save detailed response for analysis
                filename = f"api_response_{method_info['method'].replace('.', '_')}.json"
                with open(filename, 'w') as f:
                    json.dump(method_info['response'], f, indent=2, default=str)
                print(f"   📁 Full response saved: {filename}")
        else:
            print(f"❌ No working methods found for Open Channel messages")
            print(f"\nPossible reasons:")
            print(f"1. 🔒 Your Bitrix account doesn't have Open Channel permissions")
            print(f"2. 📡 Open Channel messages use different API architecture")  
            print(f"3. 🏗️ WhatsApp messages may only be visible in Bitrix web interface")
            print(f"4. 🔌 May require special connector API keys or different authentication")
        
        return successful_methods
        
    except Exception as e:
        print(f"❌ Error during API discovery: {e}")
        return []

def provide_solutions_based_on_findings(successful_methods):
    """Provide next steps based on what we discovered"""
    print(f"\n🎯 RECOMMENDED NEXT STEPS:")
    print(f"=" * 40)
    
    if successful_methods:
        print(f"✅ GREAT NEWS! We found working API methods.")
        print(f"   You can now modify the conversation extraction to use:")
        for method_info in successful_methods:
            print(f"   • {method_info['method']} - {method_info['result_type']}")
        
        print(f"\n📝 Next actions:")
        print(f"1. Update extract_lead_conversations.py to use the working API method")
        print(f"2. Parse the response structure to extract actual message content")
        print(f"3. Run the full extraction to get real chat conversations")
        
    else:
        print(f"❌ Open Channel messages not accessible via standard API")
        print(f"\n🔄 Alternative approaches:")
        print(f"1. 📞 Contact Bitrix24 support for Open Channel API documentation")
        print(f"2. 🔍 Check if your plan includes Open Channel API access")
        print(f"3. 🌐 Use Bitrix24 web interface export features") 
        print(f"4. 🔌 Look into webhook/push notifications for real-time message capture")
        
        print(f"\n⚡ IMMEDIATE WORKAROUND:")
        print(f"Your current script DOES extract valuable information:")
        print(f"• Customer names and contact details")
        print(f"• Conversation timestamps and activity types")
        print(f"• Phone call records and durations")
        print(f"• Lead progression and status changes")
        print(f"This is still very useful for RAG analysis!")

if __name__ == "__main__":
    print("🚀 Bitrix24 Open Channel API Discovery")
    print("This script systematically tests API methods to find")
    print("the correct way to retrieve WhatsApp/Open Channel messages.\n")
    
    successful_methods = test_all_possible_api_methods()
    provide_solutions_based_on_findings(successful_methods)
