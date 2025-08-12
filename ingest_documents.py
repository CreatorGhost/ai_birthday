#!/usr/bin/env python3
"""
Document Ingestion Class for Leo & Loona FAQ System

This class processes FAQ documents and creates vector embeddings in Pinecone.
Configurable parameters for chunk size, overlap, and other settings.

Usage:
    from ingest_documents import DocumentIngestor
    ingestor = DocumentIngestor(chunk_size=2800, chunk_overlap=500)
    success = ingestor.ingest_markdown_file("./rag_ready_faq/consolidated_faq.md")
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import DocumentProcessor, RAGPipeline
from langchain_core.documents import Document

class DocumentIngestor:
    """
    Configurable document ingestion class for FAQ system
    """
    
    def __init__(
        self,
        chunk_size: int = 2800,
        chunk_overlap: int = 500,
        faq_folder: str = "./FAQ",
        markdown_file: str = "./rag_ready_faq/consolidated_faq.md",
        test_questions: Optional[List[str]] = None,
        run_tests: bool = True,
        verbose: bool = True
    ):
        """
        Initialize DocumentIngestor with configurable parameters
        
        Args:
            chunk_size: Size of text chunks for document splitting
            chunk_overlap: Overlap between chunks
            faq_folder: Path to FAQ documents folder
            markdown_file: Path to consolidated markdown file
            test_questions: List of questions for testing ingestion
            run_tests: Whether to run tests after ingestion
            verbose: Whether to print detailed output
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.faq_folder = faq_folder
        self.markdown_file = markdown_file
        self.run_tests = run_tests
        self.verbose = verbose
        
        # Default test questions
        self.test_questions = test_questions or [
            "What are the contact details for Leo & Loona?",
            "What are the pricing options?",
            "Tell me about the rules and regulations"
        ]
        
        # Initialize components
        self.doc_processor = None
        self.rag_pipeline = None
        
    def _print(self, message: str):
        """Print message if verbose mode is enabled"""
        if self.verbose:
            print(message)
    
    def check_environment(self) -> bool:
        """Check if all required environment variables are set"""
        self._print("🔍 Checking environment configuration...")
        
        load_dotenv()
        
        # Check API keys
        google_key = os.getenv('GOOGLE_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        pinecone_key = os.getenv('PINECONE_API_KEY')
        
        if not pinecone_key:
            self._print("❌ PINECONE_API_KEY is required")
            return False
        
        if not (google_key or openai_key):
            self._print("❌ Either GOOGLE_API_KEY or OPENAI_API_KEY is required")
            return False
        
        # Show configuration
        provider = os.getenv('LLM_PROVIDER', 'openai')
        index_name = os.getenv('PINECONE_INDEX_NAME', 'faq-embeddings')
        
        self._print(f"✅ Provider: {provider.upper()}")
        self._print(f"✅ Pinecone Index: {index_name}")
        self._print(f"✅ API Keys: {'Google' if google_key else 'OpenAI'} + Pinecone")
        
        return True
    
    def check_documents(self) -> tuple[bool, List[Path]]:
        """Check if FAQ documents exist"""
        self._print(f"📁 Checking documents in {self.faq_folder}...")
        
        if not os.path.exists(self.faq_folder):
            self._print(f"❌ FAQ folder not found: {self.faq_folder}")
            return False, []
        
        # Get all .docx files
        docx_files = list(Path(self.faq_folder).glob("*.docx"))
        
        if not docx_files:
            self._print(f"❌ No .docx files found in {self.faq_folder}")
            return False, []
        
        self._print(f"✅ Found {len(docx_files)} document(s):")
        for doc_file in docx_files:
            file_size = doc_file.stat().st_size
            self._print(f"   • {doc_file.name} ({file_size:,} bytes)")
        
        return True, docx_files
    
    def check_markdown_file(self, file_path: str = None) -> bool:
        """Check if consolidated FAQ markdown file exists"""
        file_path = file_path or self.markdown_file
        self._print(f"📁 Checking markdown file: {file_path}...")
        
        if not os.path.exists(file_path):
            self._print(f"❌ Markdown file not found: {file_path}")
            return False
        
        file_size = Path(file_path).stat().st_size
        self._print(f"✅ Found consolidated FAQ file ({file_size:,} bytes)")
        
        return True
    
    def initialize_components(self):
        """Initialize DocumentProcessor and RAGPipeline"""
        if not self.doc_processor:
            self.doc_processor = DocumentProcessor(self.faq_folder)
        
        if not self.rag_pipeline:
            self.rag_pipeline = RAGPipeline()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using configured parameters"""
        self._print(f"\n🔪 Splitting documents into chunks...")
        self._print(f"   Using chunk size: {self.chunk_size} chars")
        self._print(f"   Using chunk overlap: {self.chunk_overlap} chars")
        
        self.initialize_components()
        split_docs = self.doc_processor.split_documents(
            documents, 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        
        self._print(f"✅ Created {len(split_docs)} chunks")
        
        if split_docs:
            avg_chunk_size = sum(len(doc.page_content) for doc in split_docs) / len(split_docs)
            self._print(f"📊 Average chunk size: {avg_chunk_size:.0f} characters")
        
        return split_docs
    
    def setup_rag_pipeline(self) -> bool:
        """Setup RAG pipeline and Pinecone index"""
        self._print("\n🤖 Initializing RAG pipeline...")
        
        self.initialize_components()
        provider_info = self.rag_pipeline.model_config.get_provider_info()
        self._print(f"✅ Using {provider_info['provider']} - {provider_info['chat_model']}")
        
        # Setup Pinecone index
        self._print("\n🔧 Setting up Pinecone index...")
        
        if not self.rag_pipeline.setup_pinecone_index():
            self._print("❌ Failed to setup Pinecone index")
            return False
        
        self._print("✅ Pinecone index ready")
        return True
    
    def create_embeddings(self, split_docs: List[Document]) -> bool:
        """Create vector embeddings for documents"""
        self._print("\n🧠 Creating vector embeddings...")
        self._print("⏳ This may take a few minutes...")
        
        embedding_start = time.time()
        
        if not self.rag_pipeline.create_vector_store(split_docs):
            self._print("❌ Failed to create vector store")
            return False
        
        embedding_time = time.time() - embedding_start
        self._print(f"✅ Vector embeddings created in {embedding_time:.1f} seconds")
        
        return True
    
    def test_ingestion(self):
        """Test the ingestion with configured test questions"""
        if not self.run_tests:
            return
            
        self._print("\n🧪 Testing ingestion...")
        
        self.rag_pipeline.setup_graph()
        
        for i, question in enumerate(self.test_questions, 1):
            self._print(f"\n   Test {i}: {question}")
            result = self.rag_pipeline.answer_question(question)
            
            response_preview = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
            self._print(f"   Response: {response_preview}")
            
            doc_count = len(result.get('source_documents', []))
            self._print(f"   Sources: {doc_count} document(s)")
    
    def ingest_markdown_file(self, file_path: str = None) -> bool:
        """Ingest consolidated FAQ markdown file"""
        file_path = file_path or self.markdown_file
        
        self._print("🦁 Leo & Loona Markdown FAQ Ingestion")
        self._print("=" * 50)
        
        start_time = time.time()
        
        # Step 1: Check environment
        if not self.check_environment():
            self._print("\n❌ Environment check failed. Please check your .env file.")
            return False
        
        # Step 2: Check markdown file
        if not self.check_markdown_file(file_path):
            self._print(f"\n❌ Markdown file check failed. Please ensure {file_path} exists.")
            return False
        
        try:
            # Step 3: Load markdown file
            self._print(f"\n📋 Loading markdown file...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create document
            document = Document(
                page_content=content,
                metadata={
                    'filename': Path(file_path).name,
                    'file_path': file_path,
                    'source': 'consolidated_faq',
                    'document_type': 'markdown'
                }
            )
            
            documents = [document]
            self._print(f"✅ Loaded consolidated FAQ document")
            
            # Show document details
            total_chars = len(content)
            self._print(f"📊 Total content: {total_chars:,} characters")
            
            # Step 4: Split documents
            split_docs = self.split_documents(documents)
            
            # Step 5: Setup RAG pipeline
            if not self.setup_rag_pipeline():
                return False
            
            # Step 6: Create embeddings
            if not self.create_embeddings(split_docs):
                return False
            
            # Step 7: Test ingestion
            self.test_ingestion()
            
            # Final summary
            total_time = time.time() - start_time
            
            self._print(f"\n🎉 MARKDOWN INGESTION COMPLETE!")
            self._print("=" * 50)
            self._print(f"📊 Summary:")
            self._print(f"   • Markdown file processed: {Path(file_path).name}")
            self._print(f"   • Chunks created: {len(split_docs)}")
            self._print(f"   • Chunk size: {self.chunk_size}")
            self._print(f"   • Chunk overlap: {self.chunk_overlap}")
            self._print(f"   • Total time: {total_time:.1f} seconds")
            
            self._print(f"\n✅ Your FAQ system is ready with consolidated data!")
            self._print(f"🚀 Run: streamlit run app.py")
            
            return True
            
        except Exception as e:
            self._print(f"\n❌ Error during markdown ingestion: {str(e)}")
            return False
    
    def ingest_docx_documents(self) -> bool:
        """Ingest DOCX documents from FAQ folder"""
        self._print("🦁 Leo & Loona Document Ingestion")
        self._print("=" * 50)
        
        start_time = time.time()
        
        # Step 1: Check environment
        if not self.check_environment():
            self._print("\n❌ Environment check failed. Please check your .env file.")
            return False
        
        # Step 2: Check documents
        docs_exist, doc_files = self.check_documents()
        if not docs_exist:
            self._print("\n❌ Document check failed. Please ensure FAQ documents exist.")
            return False
        
        try:
            # Step 3: Load documents
            self._print(f"\n📋 Loading {len(doc_files)} documents...")
            
            self.initialize_components()
            documents = self.doc_processor.load_docx_files()
            
            if not documents:
                self._print("❌ Failed to load documents")
                return False
            
            self._print(f"✅ Loaded {len(documents)} documents")
            
            # Show document details
            total_chars = sum(len(doc.page_content) for doc in documents)
            self._print(f"📊 Total content: {total_chars:,} characters")
            
            # Step 4: Split documents
            split_docs = self.split_documents(documents)
            
            # Step 5: Setup RAG pipeline
            if not self.setup_rag_pipeline():
                return False
            
            # Step 6: Create embeddings
            if not self.create_embeddings(split_docs):
                return False
            
            # Step 7: Test ingestion
            self.test_ingestion()
            
            # Final summary
            total_time = time.time() - start_time
            
            self._print(f"\n🎉 INGESTION COMPLETE!")
            self._print("=" * 50)
            self._print(f"📊 Summary:")
            self._print(f"   • Documents processed: {len(documents)}")
            self._print(f"   • Chunks created: {len(split_docs)}")
            self._print(f"   • Chunk size: {self.chunk_size}")
            self._print(f"   • Chunk overlap: {self.chunk_overlap}")
            self._print(f"   • Total time: {total_time:.1f} seconds")
            self._print(f"   • Average time per document: {total_time/len(documents):.1f}s")
            
            self._print(f"\n✅ Your enhanced FAQ system is ready!")
            self._print(f"🚀 Run: streamlit run app.py")
            
            return True
            
        except Exception as e:
            self._print(f"\n❌ Error during ingestion: {str(e)}")
            self._print("\nTroubleshooting:")
            self._print("1. Check your .env file has correct API keys")
            self._print("2. Ensure FAQ folder contains .docx files")
            self._print("3. Verify internet connectivity for API calls")
            self._print("4. Check Pinecone API key permissions")
            return False
    
    def auto_ingest(self) -> bool:
        """
        Automatically choose best ingestion method based on available files
        """
        # Check for consolidated markdown first (recommended)
        if os.path.exists(self.markdown_file):
            self._print("📝 Found consolidated FAQ markdown file!")
            return self.ingest_markdown_file()
        else:
            self._print("📄 Using original .docx files (no consolidated markdown found)")
            return self.ingest_docx_documents()

def main():
    """Main entry point with interactive mode"""
    # Create ingestor with default parameters
    ingestor = DocumentIngestor()
    
    # Check if we should use markdown file instead
    markdown_file = "./rag_ready_faq/consolidated_faq.md"
    
    if os.path.exists(markdown_file):
        print("📝 Found consolidated FAQ markdown file!")
        print("Choose ingestion method:")
        print("1. Use consolidated FAQ markdown (recommended)")
        print("2. Use original .docx files")
        
        choice = input("\nEnter choice (1 or 2): ").strip()
        
        if choice == "1":
            success = ingestor.ingest_markdown_file(markdown_file)
        else:
            success = ingestor.ingest_docx_documents()
    else:
        print("📄 Using original .docx files (no consolidated markdown found)")
        success = ingestor.ingest_docx_documents()
    
    if success:
        print(f"\n🎊 SUCCESS! Your Leo & Loona FAQ system is ready.")
        print(f"   You can now run your Streamlit app to start answering questions.")
    else:
        print(f"\n💥 FAILED! Please check the errors above and try again.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)