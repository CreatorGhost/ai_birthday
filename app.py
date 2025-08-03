import os
import sys
import streamlit as st
from dotenv import load_dotenv
import nest_asyncio

# Import from rag_system package
from rag_system import RAGPipeline

# Load environment variables
load_dotenv()

# Enable nested event loops for Streamlit compatibility
nest_asyncio.apply()

@st.cache_resource
def initialize_rag_pipeline():
    """Initialize and cache the RAG pipeline with existing vector store"""
    try:
        print("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline()
        
        # Try to load existing vector store
        if rag_pipeline.load_existing_vector_store():
            rag_pipeline.setup_graph()
            print("Successfully loaded existing vector store")
            return rag_pipeline, True, None
        else:
            print("No existing vector store found")
            return None, False, "No existing vector store found. Please load documents first."
            
    except Exception as e:
        error_msg = f"Error initializing RAG pipeline: {str(e)}"
        print(error_msg)
        return None, False, error_msg

def main():
    st.set_page_config(
        page_title="Leo & Loona FAQ Assistant",
        page_icon="🦁",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🦁 Leo & Loona FAQ Assistant")
    st.write("Ask questions about our play areas and get instant answers with enhanced location awareness!")
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'initialization_attempted' not in st.session_state:
        st.session_state.initialization_attempted = False

    # Auto-initialize RAG pipeline on startup
    if not st.session_state.initialization_attempted:
        st.session_state.initialization_attempted = True
        
        with st.spinner("🚀 Loading Leo & Loona FAQ system..."):
            rag_pipeline, success, error_msg = initialize_rag_pipeline()
            
            if success:
                st.session_state.rag_pipeline = rag_pipeline
                st.session_state.documents_loaded = True
                st.success("✅ FAQ system ready! You can now ask questions about Leo & Loona.")
            else:
                st.session_state.rag_pipeline = None
                st.session_state.documents_loaded = False
                st.error(f"❌ {error_msg}")
                st.info("💡 Please use the sidebar options to load documents first.")
    
    # Sidebar for configuration and manual controls
    st.sidebar.header("🔧 Configuration")
    
    # Load credentials from .env file
    openai_key = os.getenv('OPENAI_API_KEY')
    pinecone_key = os.getenv('PINECONE_API_KEY') 
    google_key = os.getenv('GOOGLE_API_KEY')
    index_name = os.getenv('PINECONE_INDEX_NAME', 'faq-embeddings')
    
    # Display configuration status
    st.sidebar.write(f"**API Keys Status:**")
    st.sidebar.write(f"OpenAI: {'✅' if openai_key else '❌'}")
    st.sidebar.write(f"Google: {'✅' if google_key else '❌'}")
    st.sidebar.write(f"Pinecone: {'✅' if pinecone_key else '❌'}")
    st.sidebar.write(f"Index: {index_name}")
    
    # Show system status
    if hasattr(st.session_state, 'documents_loaded') and st.session_state.documents_loaded:
        st.sidebar.success("✅ System Ready")
    else:
        st.sidebar.error("❌ System Not Ready")
        
    if not (openai_key or google_key) or not pinecone_key:
        st.sidebar.error("Missing API keys!")
        st.error("❌ Missing required API keys. Please check your .env file configuration.")
        st.info("Required: (OPENAI_API_KEY or GOOGLE_API_KEY) and PINECONE_API_KEY")
        return

    # Manual controls in sidebar
    st.sidebar.header("🔄 Manual Controls")
    
    # Force reload button
    if st.sidebar.button("🔄 Reload System"):
        # Clear cache and reinitialize
        initialize_rag_pipeline.clear()
        st.session_state.initialization_attempted = False
        st.rerun()
    
    # Document ingestion info
    st.sidebar.info(
        "📋 **Document Ingestion**\n\n"
        "To update the knowledge base with new documents:\n\n"
        "1. Add .docx files to the FAQ folder\n"
        "2. Run: `python ingest_documents.py`\n"
        "3. Restart this app\n\n"
        "Current system loads pre-built vectors automatically."
    )
    
    # Manual ingestion button (advanced users)
    with st.sidebar.expander("🔧 Advanced: Manual Ingestion"):
        st.warning("⚠️ Only use if ingest_documents.py fails")
        
        if st.button("📂 Force Re-ingest Documents"):
            with st.spinner("Re-ingesting documents (this may take a few minutes)..."):
                try:
                    import subprocess
                    import sys
                    
                    # Run the ingestion script
                    result = subprocess.run(
                        [sys.executable, "ingest_documents.py"],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    
                    if result.returncode == 0:
                        st.success("✅ Documents re-ingested successfully!")
                        st.info("🔄 Please restart the app to load the updated vectors.")
                        # Clear cache
                        initialize_rag_pipeline.clear()
                    else:
                        st.error(f"❌ Ingestion failed: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    st.error("⏰ Ingestion timed out. Please run `python ingest_documents.py` manually.")
                except Exception as e:
                    st.error(f"❌ Error running ingestion: {str(e)}")
                    st.info("💡 Try running `python ingest_documents.py` in terminal instead.")
    
    # Clear chat history button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
        
    # Show enhanced features info
    st.sidebar.header("✨ Enhanced Features")
    st.sidebar.info(
        "🎯 **Location Awareness**\n"
        "Automatically detects which park you're asking about\n\n"
        "🛡️ **Safety First**\n"
        "Always includes supervision requirements\n\n"
        "🤖 **Smart Escalation**\n"
        "Connects you to human agents when needed"
    )

    # Main chat interface
    if hasattr(st.session_state, 'documents_loaded') and st.session_state.documents_loaded and st.session_state.rag_pipeline:
        st.header("💬 Chat with Leo & Loona Assistant")
        
        # Show helpful tips for first-time users
        if not st.session_state.messages:
            st.info(
                "👋 **Welcome!** I'm your Leo & Loona assistant. Try asking me:\n\n"
                "• 'What are the prices at Dalma Mall?'\n"
                "• 'What are the opening hours?'\n"
                "• 'Tell me about your safety rules'\n"
                "• 'Can I book a birthday party?'"
            )
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about Leo & Loona:"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Show typing indicator and use enhanced pipeline
                    result = st.session_state.rag_pipeline.answer_question(
                        prompt, 
                        chat_history=st.session_state.messages[:-1]  # Exclude current message
                    )
                    
                    response = result["answer"]
                    st.markdown(response)
                    
                    # Show source documents in an expander
                    if result["source_documents"]:
                        with st.expander("📚 Source Documents"):
                            for i, doc in enumerate(result["source_documents"]):
                                st.write(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')}")
                                st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                st.divider()
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Show usage tips in expandable section
        with st.expander("💡 Tips for better answers"):
            st.write(
                "• **Be specific about location**: Mention Dalma Mall, Yas Mall, or Festival City\n"
                "• **Ask about specific topics**: Prices, hours, safety rules, bookings\n"
                "• **Follow up questions**: I remember our conversation context\n"
                "• **Need human help?**: I'll connect you when I can't fully answer"
            )
    else:
        # System not ready - show status and help
        st.warning("🔄 System is initializing or encountered an error...")
        
        if hasattr(st.session_state, 'rag_pipeline') and not st.session_state.rag_pipeline:
            st.error(
                "❌ **System Not Ready**\n\n"
                "The FAQ system could not initialize. This might be due to:\n"
                "• Missing API keys in .env file\n"
                "• No existing vector store found\n"
                "• Network connectivity issues\n\n"
                "**Solutions:**\n"
                "1. Check your .env file has the required API keys\n"
                "2. Run document ingestion: `python ingest_documents.py`\n"
                "3. Try the 'Reload System' button\n"
                "4. Check Pinecone index exists with correct name"
            )
        else:
            st.info("⏳ Please wait while the system initializes...")

if __name__ == "__main__":
    main()