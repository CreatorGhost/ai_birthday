# 🚀 Simple WhatsApp Chatbot Testing

Clean, simple setup to test your chatbot with a separate WhatsApp number.

## 📁 What You Have

### Core Files (Only 3!)

1. **`whatsapp_test_integration.py`** - Main integration logic
2. **`whatsapp_test_server.py`** - FastAPI server
3. **`WHATSAPP_TEST_SETUP.md`** - Complete setup guide

### Key Features

✅ **Test with separate WhatsApp number** (won't affect production)  
✅ **Send real WhatsApp messages** to test number  
✅ **Chat with your chatbot** within WhatsApp  
✅ **Safe test mode** (control when responses are sent)  
✅ **Easy integration** with your existing Streamlit chatbot

## ⚡ Quick Start (10 minutes)

### 1. Get Test WhatsApp Number

- Get a new UAE number (+971 xxx xxx xxx)
- Set up WhatsApp Business on it
- This becomes your test number

### 2. Configure Bitrix24

- Contact Center → Open Channels → Add WhatsApp
- Use test number, set Line ID as "99"
- Set webhook: `https://your-server.com/webhook/whatsapp-test`

### 3. Start Testing

```bash
# Start the server
python whatsapp_test_server.py

# Send WhatsApp messages to your test number
# Watch server console for processing logs
```

## 🧪 Test Your Chatbot

### Via API (Quick Test)

```bash
curl -X POST "http://localhost:8000/test-message" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test User",
    "message": "Do you do birthday parties?",
    "phone": "971501234567"
  }'
```

### Via WhatsApp (Real Test)

1. Send message to your test WhatsApp number
2. Watch server console logs
3. See chatbot response generated
4. Test different message types

## 🔧 Integration with Your Streamlit Chatbot

In `whatsapp_test_integration.py`, find the `generate_chatbot_response()` function:

```python
def generate_chatbot_response(self, message_info: Dict[str, Any]) -> str:
    """
    TODO: Replace this with your actual Streamlit chatbot logic
    """

    # Replace this simple logic with your actual chatbot:
    # from your_chatbot_module import generate_response
    # return generate_response(message_text, customer_name)

    message_text = message_info.get('message_text', '')
    customer_name = message_info.get('customer_name', 'there')

    # Your actual chatbot logic goes here
    return "Your chatbot response"
```

## 🎯 What Happens

1. **You send WhatsApp message** → Test number
2. **Bitrix24 receives it** → Line 99 (test line)
3. **Webhook calls your server** → `/webhook/whatsapp-test`
4. **Your chatbot processes it** → Generates response
5. **Server logs everything** → You see what would happen
6. **When ready** → Switch to production mode to send real responses

## 🔄 Test vs Production Mode

### Test Mode (Default)

- ✅ Processes messages
- ✅ Generates responses
- ✅ Logs everything
- ❌ Doesn't send responses to WhatsApp
- **Perfect for testing logic**

### Production Mode (When Ready)

```bash
curl -X POST "http://localhost:8000/switch-to-production"
```

- ✅ Sends REAL responses to WhatsApp
- **Use only when testing is complete**

## 📊 Monitor Your Tests

### Server Console

```
📨 Processing message from Ahmed: Do you do birthday parties?
🧪 TEST MODE - Response generated:
   📱 Customer: Ahmed (971501234567)
   💬 Message: Do you do birthday parties?
   🤖 Response: Hi Ahmed! 🎂 I'd love to help...
   ✅ In production, this would be sent to WhatsApp
```

### Log Files

- Saved in `logs/whatsapp_test_YYYYMMDD.log`
- JSON format for easy analysis

## ✅ Success Checklist

- [ ] Test WhatsApp number obtained
- [ ] Line 99 configured in Bitrix24
- [ ] Webhook URL set correctly
- [ ] Server starts without errors
- [ ] Test messages reach server
- [ ] Appropriate responses generated
- [ ] Your chatbot logic integrated
- [ ] Ready for production switch

## 🚀 Next Steps

1. **Test thoroughly** with different message types
2. **Integrate your Streamlit chatbot** logic
3. **Test edge cases** (typos, different languages)
4. **When satisfied** → Switch to production mode
5. **Configure main line** → Switch from test Line 99 to production Line 1

## 🆘 Need Help?

### Common Issues

- **Server won't start**: Check port 8000 is available
- **No messages received**: Verify webhook URL in Bitrix24
- **Wrong responses**: Check your chatbot logic integration

### Check Health

```bash
curl http://localhost:8000/health
```

### API Documentation

Open http://localhost:8000/docs in browser

---

## 🎉 You're Ready!

Your simple WhatsApp test setup is complete. Get your test number and start chatting with your chatbot! 🤖💬
