# FAQ RAG Pipeline with Pinecone

A Retrieval-Augmented Generation (RAG) pipeline that processes FAQ documents and provides AI-powered question answering using LangChain, OpenAI embeddings, and Pinecone vector database.

## Features

### Core RAG System
- **Document Processing**: Automatically loads and processes .docx FAQ documents
- **Vector Embeddings**: Uses OpenAI embeddings for semantic search
- **Pinecone Integration**: Stores and retrieves vectors using Pinecone vector database
- **Question Answering**: Provides accurate answers with source document references
- **Dual Interface**: Both command-line and web-based Streamlit interface
- **Dynamic Model Support**: Automatically fetches and uses latest OpenAI models
- **Advanced RAG**: Implements sophisticated retrieval-augmented generation with LangGraph

### Bitrix CRM Integration 🆕
- **CRM Data Access**: Retrieve deals, contacts, companies, leads, and activities from Bitrix24
- **Semantic CRM Search**: Convert CRM data into embeddings for natural language queries
- **Automatic Synchronization**: Keep your vector store updated with latest CRM data
- **Flexible Configuration**: Customize data processing and synchronization settings
- **Command-line Tools**: Utilities for testing, syncing, and searching CRM data
- **Unified Search**: Search across both FAQ documents and CRM data simultaneously

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Pinecone API key
- FAQ documents in .docx format

## Installation

1. **Clone or download this project**

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:

   - Copy `.env.example` to `.env`
   - Fill in your API keys:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` file:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_ENVIRONMENT=us-east-1
   PINECONE_INDEX_NAME=faq-embeddings
   ```

## Getting API Keys

### OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key

### Pinecone API Key

1. Go to [Pinecone](https://www.pinecone.io/)
2. Sign up for a free account
3. Create a new project
4. Get your API key from the dashboard
5. Note your environment (usually `us-east-1` for free tier)

## Usage

### Command Line Interface (Recommended)

Run the simple command-line version:

```bash
python simple_rag.py
```

This will:

1. Load all .docx files from the FAQ folder
2. Create embeddings and store them in Pinecone
3. Start an interactive Q&A session

**Example interaction**:

```
============================================================
FAQ RAG Pipeline with Pinecone Vector Database
============================================================

Loading documents from: ./FAQ
✓ Loaded: TASK 5.1.1 - LL. FESTIVAL - FAQ.docx (2543 characters)
✓ Loaded: TASK 5.1.1 - LL. YAS- FAQ.docx (1876 characters)
...
Split 13 documents into 45 chunks

Initializing RAG pipeline...
Initializing OpenAI embeddings and LLM...
Connecting to Pinecone...
Using existing Pinecone index: faq-embeddings
✓ Existing vector store loaded
Setting up QA chain...
✓ QA chain setup complete

✓ RAG pipeline ready! Loaded 13 documents with 45 chunks.

============================================================
You can now ask questions about the FAQ documents.
Type 'quit' or 'exit' to stop.
============================================================

Enter your question: What are the park rules?

Processing question: What are the park rules?

--------------------------------------------------
ANSWER:
--------------------------------------------------
Based on the FAQ documents, here are the key park rules:

1. No outside food or beverages allowed
2. All bags subject to security inspection
3. No smoking in designated areas
4. Follow all posted signs and staff instructions
...

--------------------------------------------------
SOURE DOCUMENTS:
--------------------------------------------------

1. Source: TASK 5.1.3 - LL. FESTIVAL- PARK RULES.docx
   Preview: Park Rules and Regulations...
```

### Streamlit Web Interface

Run the web interface:

```bash
streamlit run rag_pipeline.py
```

This provides a user-friendly web interface where you can:

- Configure API keys through the sidebar
- Load FAQ documents
- Ask questions and get answers
- View source documents

## Project Structure

```
FAQ-RAG-Pipeline/
├── rag_system/             # Core RAG system
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration management
│   ├── document_processor.py # Document loading and processing
│   ├── rag_pipeline.py     # RAG pipeline implementation
│   ├── model_fetcher.py    # Dynamic model fetching
│   └── model_utils.py      # Model utilities
├── bitrix_integration/     # Bitrix CRM integration
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Bitrix-specific configuration
│   ├── bitrix_client.py    # Bitrix24 API client
│   ├── bitrix_data_processor.py # CRM data processing
│   ├── bitrix_pipeline.py  # CRM pipeline orchestration
│   ├── bitrix_utils.py     # Command-line utilities
│   ├── example_usage.py    # Usage examples
│   └── README.md           # Bitrix integration docs
├── FAQ/                    # FAQ documents (.docx files)
│   ├── TASK 5.1.1 - LL. FESTIVAL - FAQ.docx
│   ├── TASK 5.1.1 - LL. YAS- FAQ.docx
│   └── ...
├── tests/                  # Test scripts
├── app.py                  # Streamlit web interface
├── simple_rag.py          # Command-line interface
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .env                    # Your environment variables (create this)
└── README.md             # This file
```

## How It Works

1. **Document Loading**: The system scans the FAQ folder for .docx files and extracts text content

2. **Text Splitting**: Large documents are split into smaller chunks (1000 characters with 200 character overlap) for better retrieval

3. **Embedding Generation**: Each text chunk is converted to a vector embedding using OpenAI's text-embedding-ada-002 model

4. **Vector Storage**: Embeddings are stored in Pinecone vector database with metadata about source documents

5. **Question Processing**: When you ask a question:
   - Your question is converted to an embedding
   - Pinecone finds the most similar document chunks
   - The relevant chunks are sent to GPT-3.5-turbo along with your question
   - The AI generates a contextual answer based on the retrieved information

## Configuration Options

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `PINECONE_API_KEY`: Your Pinecone API key (required)
- `PINECONE_ENVIRONMENT`: Pinecone environment (default: us-east-1)
- `PINECONE_INDEX_NAME`: Name for your Pinecone index (default: faq-embeddings)

### Customization

You can modify these parameters in the code:

- **Chunk size**: Change `chunk_size` in `split_documents()` method
- **Chunk overlap**: Change `chunk_overlap` in `split_documents()` method
- **Number of retrieved documents**: Change `k` value in retriever setup
- **LLM model**: Change the model in `ChatOpenAI` initialization
- **Embedding model**: Change the model in `OpenAIEmbeddings` initialization

## Troubleshooting

### Common Issues

1. **"No documents found"**:

   - Ensure .docx files are in the FAQ folder
   - Check that files are not corrupted
   - Verify files contain readable text

2. **"API key not found"**:

   - Check your .env file exists and has correct keys
   - Verify API keys are valid and have sufficient credits

3. **"Pinecone connection error"**:

   - Verify your Pinecone API key and environment
   - Check your internet connection
   - Ensure you haven't exceeded Pinecone limits

4. **"Import errors"**:
   - Run `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

### Performance Tips

- **First run**: Creating embeddings takes time. Subsequent runs will be faster as they reuse the existing vector store
- **Large documents**: Consider reducing chunk size for very large documents
- **Many questions**: The vector store persists in Pinecone, so you don't need to recreate it each time

## Cost Considerations

- **OpenAI**: Charges for embedding generation and LLM usage
- **Pinecone**: Free tier includes 1 index with 1M vectors
- **Tip**: Use existing vector store when possible to avoid re-embedding documents

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License.
