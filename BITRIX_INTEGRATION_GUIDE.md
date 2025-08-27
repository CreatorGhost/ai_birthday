# 🎯 **Bitrix Lead Categorization System**

## **📋 Overview**

The chatbot now automatically creates leads in Bitrix with intelligent categorization based on conversation content.

---

## **🎂 Lead Categories**

### **1. NEW APPROACH (Birthday Party Leads)**

**When conversations contain:**

- Keywords: "birthday", "party", "celebration", "celebrate"
- Birthday packages, party room inquiries
- Age mentions, kids birthday planning
- Group bookings for celebrations
- Special event planning

**Bitrix Status:** `NEW` (maps to your "NEW APPROACH" stage)

### **2. GENERAL QUESTIONS (Regular Inquiries)**

**When conversations contain:**

- General park information, hours, prices
- Location questions, directions
- Regular family visits
- General activities and attractions
- Safety questions, policies

**Bitrix Status:** `GENERAL_QUESTIONS` (maps to your "GENERAL QUESTIONS" stage)

---

## **🤖 How It Works**

### **Step 1: Conversation Analysis**

- LLM analyzes entire conversation context
- Identifies keywords and intent
- Assigns confidence score (0-100%)
- Determines appropriate category

### **Step 2: Lead Creation**

- Triggered when user provides name + meaningful conversation
- Creates lead with rich conversation context
- Includes analysis, keywords, and full chat history
- Assigns to correct Bitrix stage automatically

### **Step 3: Lead Data Structure**

```
Lead Title: 🎂 Birthday Party Inquiry - John
           💬 General Inquiry - Sarah

Lead Description:
🤖 Chatbot Conversation Analysis
Category: Birthday Party
Confidence: 85%
Analysis: User mentioned birthday celebration for child

🔍 Keywords Identified: birthday, party, kids, celebration

💬 Conversation Summary
Total Messages: 4
User Questions: 2

Original Inquiry: "I want to plan a birthday party for my 5-year-old"

📅 Generated: 2024-01-15 14:30:22
🔗 Source: Leo & Loona AI Assistant
```

---

## **📊 Bitrix Field Mapping**

| **Chatbot Data**      | **Bitrix Field**         | **Purpose**                   |
| --------------------- | ------------------------ | ----------------------------- |
| User Name             | `NAME`                   | Lead identification           |
| Phone Number          | `PHONE`                  | Contact information           |
| Conversation Category | `STATUS_ID`              | Auto-routing to correct stage |
| Full Analysis         | `COMMENTS`               | Complete conversation context |
| Keywords              | `UF_KEYWORDS`            | Quick topic identification    |
| Confidence Score      | `UF_CATEGORY_CONFIDENCE` | AI analysis reliability       |
| Message Count         | `UF_TOTAL_MESSAGES`      | Engagement level              |

---

## **🔧 Configuration**

### **Environment Variables**

Add to your `.env` file:

```
BITRIX_WEBHOOK_URL=your_bitrix_webhook_url_here
```

### **Bitrix Status IDs**

Update in `bitrix_integration/lead_manager.py` if your Bitrix uses different status IDs:

```python
def _get_bitrix_status(self, category_analysis: Dict) -> str:
    if category == 'birthday_party' and confidence > 0.7:
        return 'NEW'  # Change to your "NEW APPROACH" status ID
    else:
        return 'GENERAL_QUESTIONS'  # Change to your "GENERAL QUESTIONS" status ID
```

---

## **📈 Benefits**

1. **Automatic Categorization**: No manual sorting needed
2. **Rich Context**: Full conversation history in Bitrix
3. **High Accuracy**: LLM-powered analysis with confidence scoring
4. **Immediate Action**: Leads created in real-time during chat
5. **WhatsApp Ready**: Works with current Streamlit and future WhatsApp integration

---

## **🧪 Testing**

### **Birthday Party Test**

1. Ask: "I want to plan a birthday party for my daughter"
2. Provide name when asked
3. Check Bitrix - lead should appear in "NEW APPROACH" with 🎂 icon

### **General Inquiry Test**

1. Ask: "What are your opening hours?"
2. Provide name when asked
3. Check Bitrix - lead should appear in "GENERAL QUESTIONS" with 💬 icon

---

## **📁 Files Added/Modified**

- `bitrix_integration/lead_manager.py` - **NEW** - Lead creation and categorization logic
- `rag_system/user_tracker.py` - Added conversation analysis function
- `rag_system/rag_pipeline.py` - Integrated Bitrix lead creation
- `app.py` - Added lead creation notifications

**Ready for production use!** 🚀

