import os
import sys
import streamlit as st
from dotenv import load_dotenv
import asyncio
import nest_asyncio

# Import from src package
from src import DocumentProcessor, RAGPipeline

# Load environment variables
load_dotenv()

# Enable nested event loops for Streamlit compatibility
nest_asyncio.apply()

def main():
    st.title("FAQ RAG Pipeline with Pinecone")
    st.write("Ask questions about the FAQ documents and get AI-powered answers!")
    
    # Initialize session state
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Load credentials from .env file
    openai_key = os.getenv('OPENAI_API_KEY')
    pinecone_key = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_ENVIRONMENT', 'us-east-1')
    index_name = os.getenv('PINECONE_INDEX_NAME', 'faq-embeddings')
    
    # Display configuration status
    st.sidebar.write(f"**Configuration Status:**")
    st.sidebar.write(f"OpenAI API Key: {'✓ Loaded' if openai_key else '✗ Missing'}")
    st.sidebar.write(f"Pinecone API Key: {'✓ Loaded' if pinecone_key else '✗ Missing'}")
    st.sidebar.write(f"Pinecone Environment: {pinecone_env}")
    st.sidebar.write(f"Index Name: {index_name}")
    
    if not openai_key or not pinecone_key:
        st.sidebar.error("Please set OPENAI_API_KEY and PINECONE_API_KEY in your .env file")
        st.error("Missing API keys. Please check your .env file configuration.")
        return
    
    # Load documents button
    if st.sidebar.button("Load FAQ Documents"):
        
        with st.spinner("Loading documents and setting up RAG pipeline..."):
            try:
                # Initialize document processor
                faq_folder = "./FAQ"
                doc_processor = DocumentProcessor(faq_folder)
                
                # Load and process documents
                documents = doc_processor.load_docx_files()
                if not documents:
                    st.error("No documents found in FAQ folder")
                    return
                
                split_docs = doc_processor.split_documents(documents)
                
                # Initialize RAG pipeline with event loop handling
                try:
                    # Ensure we have an event loop
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    
                    rag_pipeline = RAGPipeline()
                    
                    # Setup Pinecone index
                    if not rag_pipeline.setup_pinecone_index():
                        st.error("Failed to setup Pinecone index")
                        return
                    
                    # Create vector store
                    if not rag_pipeline.create_vector_store(split_docs):
                        st.error("Failed to create vector store")
                        return
                    
                    # Setup LangGraph workflow
                    rag_pipeline.setup_graph()
                    
                except RuntimeError as e:
                    if "event loop" in str(e).lower():
                        st.error("Event loop error. This is a known issue with Streamlit and async operations. Please try refreshing the page.")
                        st.info("If the problem persists, try using the command-line interface instead: `python -m src.model_utils`")
                        return
                    else:
                        raise e
                
                st.session_state.rag_pipeline = rag_pipeline
                st.session_state.documents_loaded = True
                
                st.success(f"Successfully loaded {len(documents)} documents and created {len(split_docs)} chunks!")
                
            except Exception as e:
                st.error(f"Error setting up RAG pipeline: {str(e)}")
    
    # Load existing vector store button
    if st.sidebar.button("Load Existing Vector Store"):
        
        with st.spinner("Loading existing vector store..."):
            try:
                # Ensure we have an event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                rag_pipeline = RAGPipeline()
                if rag_pipeline.load_existing_vector_store():
                    rag_pipeline.setup_graph()
                    st.session_state.rag_pipeline = rag_pipeline
                    st.session_state.documents_loaded = True
                    st.success("Successfully loaded existing vector store!")
                else:
                    st.error("Failed to load existing vector store")
            except RuntimeError as e:
                if "event loop" in str(e).lower():
                    st.error("Event loop error. Please try refreshing the page.")
                    st.info("If the problem persists, try using the command-line interface instead.")
                else:
                    st.error(f"Runtime error: {str(e)}")
            except Exception as e:
                st.error(f"Error loading vector store: {str(e)}")
    
    # Question answering interface with chat history
    if st.session_state.documents_loaded and st.session_state.rag_pipeline:
        st.header("Chat with FAQ Assistant")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the FAQ:"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
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
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("Please load the FAQ documents first using the sidebar.")

if __name__ == "__main__":
    main()