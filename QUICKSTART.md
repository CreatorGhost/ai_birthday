# Quick Start Guide

Get your FAQ RAG pipeline running in 5 minutes!

## 🚀 Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
# Run the setup script
./setup.sh

# Edit your API keys
nano .env

# Test the setup
python3 test_setup.py

# Start asking questions!
python3 simple_rag.py
```

### Option 2: Manual Setup

```bash
# Install dependencies
pip3 install -r requirements.txt

# Create environment file
cp .env.example .env

# Edit with your API keys
nano .env
```

## 🔑 Get Your API Keys

### OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create account or sign in
3. Click "Create new secret key"
4. Copy the key to your `.env` file

### Pinecone API Key

1. Visit [Pinecone](https://app.pinecone.io/)
2. Sign up for free account
3. Create a new project
4. Copy API key from dashboard
5. Add to your `.env` file

## 📝 Your .env File Should Look Like:

```
OPENAI_API_KEY=sk-your-openai-key-here
PINECONE_API_KEY=your-pinecone-key-here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=faq-embeddings
```

## 🧪 Test Everything Works

```bash
python3 test_setup.py
```

You should see all tests pass ✅

## 🎯 Start Using the RAG Pipeline

### Command Line (Simple)

```bash
python3 simple_rag.py
```

### Web Interface (Advanced)

```bash
streamlit run rag_pipeline.py
```

## 💡 Example Questions to Try

- "What are the park rules?"
- "What is the waiver policy?"
- "What are the festival guidelines?"
- "What should I know about YAS events?"
- "Tell me about DALMA requirements"

## 🔧 Troubleshooting

### "No documents found"

- Check that your FAQ folder contains .docx files
- Ensure files are not corrupted

### "API key not found"

- Verify your .env file exists
- Check API keys are correct
- Ensure no extra spaces in .env file

### "Import errors"

- Run: `pip3 install -r requirements.txt`
- Check Python version: `python3 --version` (need 3.8+)

### "Pinecone connection error"

- Verify Pinecone API key
- Check internet connection
- Try different environment (us-west-1, eu-west-1)

## 📊 What Happens on First Run

1. **Document Loading**: Reads all .docx files from FAQ folder
2. **Text Processing**: Splits documents into chunks
3. **Embedding Creation**: Converts text to vectors (takes 1-2 minutes)
4. **Vector Storage**: Saves to Pinecone (persistent)
5. **Ready**: Start asking questions!

**Note**: First run takes longer due to embedding creation. Subsequent runs are much faster!

## 💰 Cost Estimates

- **OpenAI**: ~$0.10-0.50 for initial embedding creation
- **Pinecone**: Free tier supports 1M vectors (plenty for most FAQ sets)
- **Ongoing**: ~$0.01-0.05 per question

## 🎉 You're Ready!

Once setup is complete, you can ask natural language questions about your FAQ documents and get AI-powered answers with source references.

Happy questioning! 🤖✨
