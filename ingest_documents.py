#!/usr/bin/env python3
"""
Document Ingestion Script for Leo & Loona FAQ System

This script processes FAQ documents and creates vector embeddings in Pinecone.
Run this script whenever you want to update the knowledge base with new documents.

Usage:
    python ingest_documents.py
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import DocumentProcessor, RAGPipeline

def check_environment():
    """Check if all required environment variables are set"""
    print("🔍 Checking environment configuration...")
    
    load_dotenv()
    
    # Check API keys
    google_key = os.getenv('GOOGLE_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    pinecone_key = os.getenv('PINECONE_API_KEY')
    
    if not pinecone_key:
        print("❌ PINECONE_API_KEY is required")
        return False
    
    if not (google_key or openai_key):
        print("❌ Either GOOGLE_API_KEY or OPENAI_API_KEY is required")
        return False
    
    # Show configuration
    provider = os.getenv('LLM_PROVIDER', 'openai')
    index_name = os.getenv('PINECONE_INDEX_NAME', 'faq-embeddings')
    
    print(f"✅ Provider: {provider.upper()}")
    print(f"✅ Pinecone Index: {index_name}")
    print(f"✅ API Keys: {'Google' if google_key else 'OpenAI'} + Pinecone")
    
    return True

def check_documents(faq_folder="./FAQ"):
    """Check if FAQ documents exist"""
    print(f"📁 Checking documents in {faq_folder}...")
    
    if not os.path.exists(faq_folder):
        print(f"❌ FAQ folder not found: {faq_folder}")
        return False, []
    
    # Get all .docx files
    docx_files = list(Path(faq_folder).glob("*.docx"))
    
    if not docx_files:
        print(f"❌ No .docx files found in {faq_folder}")
        return False, []
    
    print(f"✅ Found {len(docx_files)} document(s):")
    for doc_file in docx_files:
        file_size = doc_file.stat().st_size
        print(f"   • {doc_file.name} ({file_size:,} bytes)")
    
    return True, docx_files

def ingest_documents():
    """Main ingestion function"""
    print("🦁 Leo & Loona Document Ingestion")
    print("=" * 50)
    
    start_time = time.time()
    
    # Step 1: Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please check your .env file.")
        return False
    
    # Step 2: Check documents
    docs_exist, doc_files = check_documents()
    if not docs_exist:
        print("\n❌ Document check failed. Please ensure FAQ documents exist.")
        return False
    
    try:
        # Step 3: Load and process documents
        print(f"\n📋 Step 1: Loading {len(doc_files)} documents...")
        
        doc_processor = DocumentProcessor("./FAQ")
        documents = doc_processor.load_docx_files()
        
        if not documents:
            print("❌ Failed to load documents")
            return False
        
        print(f"✅ Loaded {len(documents)} documents")
        
        # Show document details
        total_chars = sum(len(doc.page_content) for doc in documents)
        print(f"📊 Total content: {total_chars:,} characters")
        
        # Step 4: Split documents into chunks with improved settings
        print("\n🔪 Step 2: Splitting documents into chunks...")
        print("   Using improved chunk size: 1800 chars (was 1000)")
        
        split_docs = doc_processor.split_documents(documents, chunk_size=1800, chunk_overlap=300)
        print(f"✅ Created {len(split_docs)} chunks")
        
        avg_chunk_size = sum(len(doc.page_content) for doc in split_docs) / len(split_docs)
        print(f"📊 Average chunk size: {avg_chunk_size:.0f} characters")
        
        # Step 5: Initialize RAG pipeline
        print("\n🤖 Step 3: Initializing enhanced RAG pipeline...")
        
        rag_pipeline = RAGPipeline()
        provider_info = rag_pipeline.model_config.get_provider_info()
        print(f"✅ Using {provider_info['provider']} - {provider_info['chat_model']}")
        
        # Step 6: Setup Pinecone index
        print("\n🔧 Step 4: Setting up Pinecone index...")
        
        if not rag_pipeline.setup_pinecone_index():
            print("❌ Failed to setup Pinecone index")
            return False
        
        print("✅ Pinecone index ready")
        
        # Step 7: Create vector embeddings
        print("\n🧠 Step 5: Creating vector embeddings...")
        print("⏳ This may take a few minutes depending on document size...")
        
        embedding_start = time.time()
        
        if not rag_pipeline.create_vector_store(split_docs):
            print("❌ Failed to create vector store")
            return False
        
        embedding_time = time.time() - embedding_start
        print(f"✅ Vector embeddings created in {embedding_time:.1f} seconds")
        
        # Step 8: Test the ingestion
        print("\n🧪 Step 6: Testing ingestion...")
        
        # Setup the enhanced graph
        rag_pipeline.setup_graph()
        
        # Test with location-specific questions
        test_questions = [
            "What are the prices at Dalma Mall?",
            "What are your opening hours?",
            "Tell me about safety rules"
        ]
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n   Test {i}: {question}")
            result = rag_pipeline.answer_question(question)
            
            # Show first 100 characters of response
            response_preview = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
            print(f"   Response: {response_preview}")
            
            # Check if documents were retrieved
            doc_count = len(result.get('source_documents', []))
            print(f"   Sources: {doc_count} document(s)")
        
        # Final summary
        total_time = time.time() - start_time
        
        print(f"\n🎉 INGESTION COMPLETE!")
        print("=" * 50)
        print(f"📊 Summary:")
        print(f"   • Documents processed: {len(documents)}")
        print(f"   • Chunks created: {len(split_docs)}")
        print(f"   • Total time: {total_time:.1f} seconds")
        print(f"   • Average time per document: {total_time/len(documents):.1f}s")
        
        print(f"\n✅ Your enhanced FAQ system is ready!")
        print(f"🚀 Run: streamlit run app.py")
        print(f"   The app will auto-load the vectors and be ready for questions.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during ingestion: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your .env file has correct API keys")
        print("2. Ensure FAQ folder contains .docx files")
        print("3. Verify internet connectivity for API calls")
        print("4. Check Pinecone API key permissions")
        return False

def main():
    """Main entry point"""
    success = ingest_documents()
    
    if success:
        print(f"\n🎊 SUCCESS! Your Leo & Loona FAQ system is ready.")
        print(f"   You can now run your Streamlit app to start answering questions.")
    else:
        print(f"\n💥 FAILED! Please check the errors above and try again.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)