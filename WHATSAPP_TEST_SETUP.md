# WhatsApp Test Setup Guide

Simple guide to test your chatbot with a separate WhatsApp number.

## 🎯 Goal

Test your chatbot by:

1. Getting a separate WhatsApp test number
2. Sending messages to this test number from your phone
3. Chatting with your chatbot within WhatsApp
4. Confirming everything works before moving to production

## 📱 Step 1: Get Test WhatsApp Number

### Option A: New UAE Number

- Get a new UAE number (+971 xxx xxx xxx)
- Set up WhatsApp Business on this number
- This will be your test number (separate from production)

### Option B: Secondary Business Line

- Use an existing secondary number
- Must be different from your main Leo & Loona number

## 🔧 Step 2: Set Up Test Line in Bitrix24

1. **Go to Bitrix24**:

   - Contact Center → Open Channels
   - Click "Add Channel" → WhatsApp

2. **Configure Test Line**:

   - Enter your test WhatsApp number
   - Set Line ID as "99" (test line)
   - Name it "LeoLoona-Test"
   - Complete the setup process

3. **Configure Webhook**:
   - In the WhatsApp channel settings
   - Set webhook URL: `https://your-server.com/webhook/whatsapp-test`
   - Enable webhook for incoming messages

## 🚀 Step 3: Start Your Test Server

```bash
# 1. Start the test server
python whatsapp_test_server.py

# Server will start on http://localhost:8000
# Webhook endpoint: http://localhost:8000/webhook/whatsapp-test
```

## 🧪 Step 4: Test Your Setup

### Test 1: API Test (Before WhatsApp)

```bash
# Test your chatbot logic first
curl -X POST "http://localhost:8000/test-message" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test User",
    "message": "Do you do birthday parties?",
    "phone": "971501234567"
  }'
```

### Test 2: WhatsApp Test

1. **Send message to test number**: Use your personal phone to send WhatsApp message to test number
2. **Check server logs**: See the message processing in your server console
3. **Verify response**: Check that appropriate response is generated

## 📋 Test Scenarios

Send these messages to your test WhatsApp number:

### Birthday Party Test

- "Do you do birthday parties?"
- "I want to book a party for my daughter"
- Expected: Birthday party information and location options

### General Information Test

- "What are your opening hours?"
- "How much does it cost?"
- Expected: Appropriate information requests

### Greeting Test

- "Hello"
- "Hi there"
- Expected: Welcome message with options

## 🔍 Monitoring Your Tests

### Check Server Console

- All messages will be logged
- See exactly what responses are generated
- Monitor for any errors

### Check Bitrix24

- Go to Contact Center → Open Channels
- Check the test line for message history
- Verify messages are coming through

## ⚙️ Configuration

### Current Settings (Test Mode)

- **Line ID**: 99 (test line)
- **Mode**: Test (responses generated but can be controlled)
- **Target**: Test WhatsApp number only

### File Structure

```
whatsapp_test_integration.py  # Core integration logic
whatsapp_test_server.py       # FastAPI webhook server
logs/                         # Test interaction logs
```

## 🎯 Expected Flow

1. **You send WhatsApp message** → Test number
2. **Bitrix24 receives message** → Test line (ID: 99)
3. **Webhook triggers** → Your server
4. **Server processes message** → Your chatbot logic
5. **Response generated** → Logged in console
6. **In production mode** → Response sent back to WhatsApp

## 🔄 Switch Modes

### Stay in Test Mode (Safe)

- Responses are generated but you control when they're sent
- Perfect for testing logic and flow

### Switch to Production Mode (When Ready)

```bash
# Via API call
curl -X POST "http://localhost:8000/switch-to-production"

# Will now send REAL responses to WhatsApp
```

## 🆘 Troubleshooting

### Server Won't Start

- Check if port 8000 is available
- Verify BITRIX_WEBHOOK_URL in .env file
- Check Python dependencies installed

### No Messages Received

- Verify webhook URL is correct in Bitrix24
- Check that test line ID is "99"
- Confirm test WhatsApp number is connected

### No Responses Generated

- Check server console for errors
- Test with `/test-message` endpoint first
- Verify chatbot logic is working

### Messages to Wrong Line

- Confirm you're messaging the test number
- Check line filtering in logs
- Verify Line ID configuration

## ✅ Success Criteria

You'll know it's working when:

- [ ] Test messages reach your server
- [ ] Appropriate responses are generated
- [ ] Only test line messages are processed
- [ ] Logs show correct message flow
- [ ] Chatbot responses are relevant and helpful

## 🚀 Next Steps

Once testing is successful:

1. **Integrate your actual Streamlit chatbot logic** into `generate_chatbot_response()`
2. **Test all scenarios** thoroughly
3. **Switch to production mode** when ready
4. **Configure production webhook** to use Line 1 (main LeoLoona line)

## 🔗 Quick Commands

```bash
# Start server
python whatsapp_test_server.py

# Test message via API
curl -X POST "http://localhost:8000/test-message" \
  -H "Content-Type: application/json" \
  -d '{"name":"Test","message":"Hello","phone":"971501234567"}'

# Check health
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

Your test setup is ready! 🎉

Get your test WhatsApp number and start testing!
