#!/usr/bin/env python3
"""
Document Ingestion Class for Leo & Loona FAQ System

This class processes FAQ documents and creates vector embeddings in Pinecone.
Configurable parameters for chunk size, overlap, and other settings.

Usage:
    from ingest_documents import DocumentIngestor
    ingestor = DocumentIngestor(chunk_size=2800, chunk_overlap=500)
    success = ingestor.auto_ingest()
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Optional, Dict
from dotenv import load_dotenv
import pandas as pd

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
        infrastructure_folder: str = "./Park infrastructure",
        markdown_file: str = "./rag_ready_faq/consolidated_faq.md",
        test_questions: Optional[List[str]] = None,
        run_tests: bool = True,
        verbose: bool = True,
        force_method: Optional[str] = None
    ):
        """
        Initialize DocumentIngestor with configurable parameters
        
        Args:
            chunk_size: Size of text chunks for document splitting
            chunk_overlap: Overlap between chunks
            faq_folder: Path to FAQ documents folder
            infrastructure_folder: Path to Park Infrastructure documents folder
            markdown_file: Path to consolidated markdown file
            test_questions: List of questions for testing ingestion
            run_tests: Whether to run tests after ingestion
            verbose: Whether to print detailed output
            force_method: Force specific ingestion method ('json', 'excel', 'markdown', 'docx', None for auto)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.faq_folder = faq_folder
        self.infrastructure_folder = infrastructure_folder
        self.markdown_file = markdown_file
        self.run_tests = run_tests
        self.verbose = verbose
        self.force_method = force_method
        
        # Default test questions including infrastructure
        self.test_questions = test_questions or [
            "What are the contact details for Leo & Loona?",
            "What are the pricing options?",
            "Tell me about the rules and regulations",
            "What infrastructure does Yas Mall have?",
            "Tell me about the facilities at Dalma Mall"
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
    
    def check_infrastructure_documents(self) -> bool:
        """Check if Infrastructure documents exist"""
        self._print(f"🏗️ Checking Infrastructure documents in {self.infrastructure_folder}...")
        
        if not os.path.exists(self.infrastructure_folder):
            self._print(f"❌ Infrastructure folder not found: {self.infrastructure_folder}")
            return False
        
        # Get all .docx files
        docx_files = list(Path(self.infrastructure_folder).glob("*.docx"))
        
        if not docx_files:
            self._print(f"❌ No .docx files found in {self.infrastructure_folder}")
            return False
        
        self._print(f"✅ Found {len(docx_files)} Infrastructure document(s):")
        for doc_file in docx_files:
            file_size = doc_file.stat().st_size
            self._print(f"   • {doc_file.name} ({file_size:,} bytes)")
        
        return True
    
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
            self.doc_processor = DocumentProcessor(self.faq_folder, self.infrastructure_folder)
        
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
    
    def extract_excel_data(self) -> Dict[str, List[Dict]]:
        """Extract data from all Excel files and all sheets within each file"""
        self._print("🔍 Extracting Excel data...")
        
        extracted_data = {
            'contacts': [],
            'locations': [],
            'pricing': []
        }
        
        excel_files = list(Path(self.faq_folder).glob("*.xlsx"))
        
        if not excel_files:
            self._print("⚠️ No Excel files found in FAQ folder")
            return extracted_data
        
        for excel_file in excel_files:
            self._print(f"Processing {excel_file.name}")
            
            try:
                # Read all sheets from the Excel file
                excel_data = pd.read_excel(excel_file, sheet_name=None)  # None reads all sheets
                
                for sheet_name, df in excel_data.items():
                    self._print(f"  Processing sheet: {sheet_name}")
                    
                    if df.empty:
                        self._print(f"    Sheet {sheet_name} is empty, skipping")
                        continue
                    
                    # Clean column names
                    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
                    
                    data_list = []
                    for _, row in df.iterrows():
                        row_data = {}
                        for col in df.columns:
                            if pd.notna(row[col]) and str(row[col]).strip():
                                row_data[col] = str(row[col]).strip()
                        
                        if row_data:
                            row_data['source_file'] = excel_file.name
                            row_data['sheet_name'] = sheet_name
                            data_list.append(row_data)
                    
                    # Categorize by filename and sheet name
                    filename_lower = excel_file.name.lower()
                    sheet_name_lower = sheet_name.lower()
                    
                    # Check both filename and sheet name for categorization
                    if ('contact' in filename_lower or 'contact' in sheet_name_lower):
                        extracted_data['contacts'].extend(data_list)
                        self._print(f"    Added {len(data_list)} contact entries from sheet {sheet_name}")
                    elif ('location' in filename_lower or 'location' in sheet_name_lower):
                        extracted_data['locations'].extend(data_list)
                        self._print(f"    Added {len(data_list)} location entries from sheet {sheet_name}")
                    elif ('pricing' in filename_lower or 'park' in filename_lower or 
                          'pricing' in sheet_name_lower or 'park' in sheet_name_lower or
                          'price' in sheet_name_lower or 'ticket' in sheet_name_lower):
                        extracted_data['pricing'].extend(data_list)
                        self._print(f"    Added {len(data_list)} pricing entries from sheet {sheet_name}")
                    else:
                        # Default to pricing for uncategorized data
                        extracted_data['pricing'].extend(data_list)
                        self._print(f"    Added {len(data_list)} entries from sheet {sheet_name} (default categorization)")
                
            except Exception as e:
                self._print(f"❌ Error processing {excel_file.name}: {str(e)}")
        
        total_contacts = len(extracted_data['contacts'])
        total_locations = len(extracted_data['locations'])
        total_pricing = len(extracted_data['pricing'])
        self._print(f"✅ Extracted: {total_contacts} contacts, {total_locations} locations, {total_pricing} pricing entries")
        
        return extracted_data

    def convert_excel_to_documents(self, excel_data: Dict) -> List[Document]:
        """Convert Excel data to LangChain Document format"""
        documents = []
        
        # Convert contacts to document
        if excel_data.get('contacts'):
            contact_content = "# Contact Information\n\n"
            for contact in excel_data['contacts']:
                source_info = []
                if 'source_file' in contact:
                    source_info.append(f"File: {contact['source_file']}")
                if 'sheet_name' in contact:
                    source_info.append(f"Sheet: {contact['sheet_name']}")
                if source_info:
                    contact_content += f"*Source: {', '.join(source_info)}*\n"
                
                for key, value in contact.items():
                    if key not in ['source_file', 'sheet_name']:
                        readable_key = key.replace('_', ' ').title()
                        contact_content += f"**{readable_key}:** {value}\n"
                contact_content += "\n"
            
            documents.append(Document(
                page_content=contact_content,
                metadata={
                    'source': 'excel_contacts',
                    'document_type': 'contact_information',
                    'content_length': len(contact_content)
                }
            ))
        
        # Convert locations to document
        if excel_data.get('locations'):
            location_content = "# Location Information\n\n"
            for location in excel_data['locations']:
                source_info = []
                if 'source_file' in location:
                    source_info.append(f"File: {location['source_file']}")
                if 'sheet_name' in location:
                    source_info.append(f"Sheet: {location['sheet_name']}")
                if source_info:
                    location_content += f"*Source: {', '.join(source_info)}*\n"
                
                for key, value in location.items():
                    if key not in ['source_file', 'sheet_name']:
                        readable_key = key.replace('_', ' ').title()
                        location_content += f"**{readable_key}:** {value}\n"
                location_content += "\n"
            
            documents.append(Document(
                page_content=location_content,
                metadata={
                    'source': 'excel_locations',
                    'document_type': 'location_information',
                    'content_length': len(location_content)
                }
            ))
        
        # Convert pricing to document
        if excel_data.get('pricing'):
            pricing_content = "# Pricing Information\n\n"
            for price_item in excel_data['pricing']:
                source_info = []
                if 'source_file' in price_item:
                    source_info.append(f"File: {price_item['source_file']}")
                if 'sheet_name' in price_item:
                    source_info.append(f"Sheet: {price_item['sheet_name']}")
                if source_info:
                    pricing_content += f"*Source: {', '.join(source_info)}*\n"
                
                for key, value in price_item.items():
                    if key not in ['source_file', 'sheet_name']:
                        readable_key = key.replace('_', ' ').title()
                        pricing_content += f"**{readable_key}:** {value}\n"
                pricing_content += "\n"
            
            documents.append(Document(
                page_content=pricing_content,
                metadata={
                    'source': 'excel_pricing',
                    'document_type': 'pricing_information',
                    'content_length': len(pricing_content)
                }
            ))
        
        return documents

    def ingest_excel_files(self) -> bool:
        """Ingest Excel files directly from FAQ folder"""
        self._print("🦁 Leo & Loona Excel FAQ Ingestion")
        self._print("=" * 50)
        
        start_time = time.time()
        
        # Step 1: Check environment
        if not self.check_environment():
            self._print("\n❌ Environment check failed. Please check your .env file.")
            return False
        
        # Step 2: Check for Excel files
        excel_files = list(Path(self.faq_folder).glob("*.xlsx"))
        if not excel_files:
            self._print(f"\n❌ No Excel files found in {self.faq_folder}")
            return False
        
        self._print(f"✅ Found {len(excel_files)} Excel file(s)")
        for excel_file in excel_files:
            file_size = excel_file.stat().st_size
            self._print(f"   • {excel_file.name} ({file_size:,} bytes)")
        
        try:
            # Step 3: Extract Excel data
            excel_data = self.extract_excel_data()
            
            # Step 4: Convert to documents
            self._print("\n📋 Converting Excel data to documents...")
            documents = self.convert_excel_to_documents(excel_data)
            
            if not documents:
                self._print("❌ No documents created from Excel data")
                return False
            
            self._print(f"✅ Created {len(documents)} documents from Excel data")
            
            # Show document details
            total_chars = sum(len(doc.page_content) for doc in documents)
            self._print(f"📊 Total content: {total_chars:,} characters")
            
            # Step 5: Split documents
            split_docs = self.split_documents(documents)
            
            # Step 6: Setup RAG pipeline
            if not self.setup_rag_pipeline():
                return False
            
            # Step 7: Create embeddings
            if not self.create_embeddings(split_docs):
                return False
            
            # Step 8: Test ingestion
            self.test_ingestion()
            
            # Final summary
            total_time = time.time() - start_time
            
            self._print(f"\n🎉 EXCEL INGESTION COMPLETE!")
            self._print("=" * 50)
            self._print(f"📊 Summary:")
            self._print(f"   • Excel files processed: {len(excel_files)}")
            self._print(f"   • Documents created: {len(documents)}")
            self._print(f"   • Chunks created: {len(split_docs)}")
            self._print(f"   • Chunk size: {self.chunk_size}")
            self._print(f"   • Chunk overlap: {self.chunk_overlap}")
            self._print(f"   • Total time: {total_time:.1f} seconds")
            
            self._print(f"\n✅ Your FAQ system is ready with Excel data!")
            self._print(f"🚀 Run: streamlit run app.py")
            
            return True
            
        except Exception as e:
            self._print(f"\n❌ Error during Excel ingestion: {str(e)}")
            return False

    def ingest_json_file(self, file_path: str) -> bool:
        """Ingest FAQ data from a JSON file."""
        self._print("🦁 Leo & Loona JSON FAQ Ingestion")
        self._print("=" * 50)
        
        start_time = time.time()

        # Step 1: Check environment
        if not self.check_environment():
            self._print("\n❌ Environment check failed. Please check your .env file.")
            return False

        # Step 2: Check JSON file
        self._print(f"📁 Checking JSON file: {file_path}...")
        if not os.path.exists(file_path):
            self._print(f"❌ JSON file not found: {file_path}")
            return False
        
        file_size = Path(file_path).stat().st_size
        self._print(f"✅ Found JSON FAQ file ({file_size:,} bytes)")

        try:
            # Step 3: Load documents from JSON
            self._print(f"\n📋 Loading documents from JSON...")
            with open(file_path, 'r', encoding='utf-8') as f:
                docs_data = json.load(f)
            
            documents = [Document(**doc_data) for doc_data in docs_data]
            self._print(f"✅ Loaded {len(documents)} documents from JSON")

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
            
            self._print(f"\n🎉 JSON INGESTION COMPLETE!")
            self._print("=" * 50)
            self._print(f"📊 Summary:")
            self._print(f"   • JSON file processed: {Path(file_path).name}")
            self._print(f"   • Documents loaded: {len(documents)}")
            self._print(f"   • Chunks created: {len(split_docs)}")
            self._print(f"   • Total time: {total_time:.1f} seconds")
            
            self._print(f"\n✅ Your FAQ system is ready with structured data!")
            self._print(f"🚀 Run: streamlit run app.py")
            
            return True
            
        except Exception as e:
            self._print(f"\n❌ Error during JSON ingestion: {str(e)}")
            return False

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
        """Ingest DOCX documents from FAQ folder and Park Infrastructure folder"""
        self._print("🦁 Leo & Loona Document Ingestion (FAQ + Infrastructure)")
        self._print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Check environment
        if not self.check_environment():
            self._print("\n❌ Environment check failed. Please check your .env file.")
            return False
        
        # Step 2: Check FAQ documents
        docs_exist, doc_files = self.check_documents()
        
        # Step 3: Check Infrastructure documents
        infrastructure_docs_exist = self.check_infrastructure_documents()
        
        # We need at least one type of document
        if not docs_exist and not infrastructure_docs_exist:
            self._print("\n❌ No documents found in either FAQ or Infrastructure folders.")
            return False
        
        if not docs_exist:
            self._print("\n⚠️ Warning: No FAQ .docx documents found, using Infrastructure documents only")
        if not infrastructure_docs_exist:
            self._print("\n⚠️ Warning: No Infrastructure documents found, using FAQ documents only")
        
        if infrastructure_docs_exist:
            self._print(f"✅ Infrastructure documents found in {self.infrastructure_folder}")
        if docs_exist:
            self._print(f"✅ FAQ documents found in {self.faq_folder}")
        
        try:
            # Step 4: Load documents from both folders
            self._print(f"\n📋 Loading documents from FAQ and Infrastructure folders...")
            
            self.initialize_components()
            documents = self.doc_processor.load_all_docx_files()
            
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
            
            # Count FAQ vs Infrastructure documents
            faq_docs = [doc for doc in documents if doc.metadata.get('source_folder') == 'FAQ']
            infra_docs = [doc for doc in documents if doc.metadata.get('source_folder') == 'Park infrastructure']
            
            self._print(f"\n🎉 INGESTION COMPLETE!")
            self._print("=" * 60)
            self._print(f"📊 Summary:")
            self._print(f"   • Total documents processed: {len(documents)}")
            self._print(f"     - FAQ documents: {len(faq_docs)}")
            self._print(f"     - Infrastructure documents: {len(infra_docs)}")
            self._print(f"   • Chunks created: {len(split_docs)}")
            self._print(f"   • Chunk size: {self.chunk_size}")
            self._print(f"   • Chunk overlap: {self.chunk_overlap}")
            self._print(f"   • Total time: {total_time:.1f} seconds")
            self._print(f"   • Average time per document: {total_time/len(documents):.1f}s")
            
            self._print(f"\n✅ Your enhanced FAQ + Infrastructure system is ready!")
            self._print(f"🏗️ Mall-specific infrastructure queries are now supported!")
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
    
    def ingest_location_specific_files(self) -> bool:
        """Ingest location-specific markdown files with proper metadata"""
        self._print("🦁 Leo & Loona Location-Specific Files Ingestion")
        self._print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Check environment
        if not self.check_environment():
            self._print("\n❌ Environment check failed. Please check your .env file.")
            return False
        
        # Step 2: Define location-specific files with their metadata
        location_files = [
            {
                "file_path": "./rag_ready_faq/yas_mall_specific.md",
                "location": "YAS_MALL",
                "source": "yas_mall_faq",
                "description": "Yas Mall specific information"
            },
            {
                "file_path": "./rag_ready_faq/dalma_mall_specific.md", 
                "location": "DALMA_MALL",
                "source": "dalma_mall_faq",
                "description": "Dalma Mall specific information"
            },
            {
                "file_path": "./rag_ready_faq/festival_city_specific.md",
                "location": "FESTIVAL_CITY", 
                "source": "festival_city_faq",
                "description": "Festival City specific information"
            },
            {
                "file_path": "./rag_ready_faq/general_faq.md",
                "location": "GENERAL",
                "source": "general_faq", 
                "description": "General non-location-specific information"
            }
        ]
        
        # Step 3: Check if all files exist
        missing_files = []
        existing_files = []
        
        for file_info in location_files:
            file_path = file_info["file_path"]
            if os.path.exists(file_path):
                file_size = Path(file_path).stat().st_size
                self._print(f"✅ Found {file_info['description']}: {Path(file_path).name} ({file_size:,} bytes)")
                existing_files.append(file_info)
            else:
                self._print(f"❌ Missing {file_info['description']}: {file_path}")
                missing_files.append(file_path)
        
        if missing_files:
            self._print(f"\n❌ Missing {len(missing_files)} required files. Please ensure all location-specific files exist.")
            return False
        
        if not existing_files:
            self._print("\n❌ No location-specific files found.")
            return False
        
        try:
            # Step 4: Load all location-specific files
            self._print(f"\n📋 Loading {len(existing_files)} location-specific files...")
            
            documents = []
            total_chars = 0
            
            for file_info in existing_files:
                file_path = file_info["file_path"]
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create document with proper location metadata
                document = Document(
                    page_content=content,
                    metadata={
                        'filename': Path(file_path).name,
                        'file_path': file_path,
                        'source': file_info["source"],
                        'location': file_info["location"],  # 🎯 CRITICAL: Location metadata for filtering
                        'document_type': 'location_specific_faq',
                        'content_length': len(content),
                        'description': file_info["description"]
                    }
                )
                
                documents.append(document)
                total_chars += len(content)
                
                self._print(f"   ✅ Loaded {file_info['description']}: {len(content):,} chars")
            
            self._print(f"✅ Loaded {len(documents)} location-specific documents")
            self._print(f"📊 Total content: {total_chars:,} characters")
            
            # Step 5: Split documents
            split_docs = self.split_documents(documents)
            
            # Step 6: Setup RAG pipeline
            if not self.setup_rag_pipeline():
                return False
            
            # Step 7: Create embeddings
            if not self.create_embeddings(split_docs):
                return False
            
            # Step 8: Test ingestion with location-specific queries
            self._print("\n🧪 Testing location-specific ingestion...")
            
            # Location-specific test questions
            location_test_questions = [
                "What are the opening hours for Yas Mall?",
                "Tell me about Dalma Mall pricing",
                "Where is Festival City located?",
                "What are the general rules and regulations?"
            ]
            
            self.rag_pipeline.setup_graph()
            
            for i, question in enumerate(location_test_questions, 1):
                self._print(f"\n   Test {i}: {question}")
                result = self.rag_pipeline.answer_question(question)
                
                response_preview = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
                self._print(f"   Response: {response_preview}")
                
                doc_count = len(result.get('source_documents', []))
                self._print(f"   Sources: {doc_count} document(s)")
                
                # Show which locations were found in source documents
                if result.get('source_documents'):
                    locations_found = set()
                    for doc in result['source_documents']:
                        location = doc.metadata.get('location', 'UNKNOWN')
                        locations_found.add(location)
                    self._print(f"   Locations: {', '.join(sorted(locations_found))}")
            
            # Final summary
            total_time = time.time() - start_time
            
            # Count documents by location
            location_counts = {}
            for doc in documents:
                location = doc.metadata.get('location', 'UNKNOWN')
                location_counts[location] = location_counts.get(location, 0) + 1
            
            self._print(f"\n🎉 LOCATION-SPECIFIC INGESTION COMPLETE!")
            self._print("=" * 60)
            self._print(f"📊 Summary:")
            self._print(f"   • Total files processed: {len(documents)}")
            for location, count in location_counts.items():
                self._print(f"     - {location}: {count} file(s)")
            self._print(f"   • Chunks created: {len(split_docs)}")
            self._print(f"   • Chunk size: {self.chunk_size}")
            self._print(f"   • Chunk overlap: {self.chunk_overlap}")
            self._print(f"   • Total time: {total_time:.1f} seconds")
            
            self._print(f"\n✅ Your location-specific FAQ system is ready!")
            self._print(f"🎯 Mall-specific filtering is now properly configured!")
            self._print(f"🚀 Run: streamlit run app.py")
            
            return True
            
        except Exception as e:
            self._print(f"\n❌ Error during location-specific ingestion: {str(e)}")
            return False
    
    def auto_ingest(self) -> bool:
        """
        Automatically choose best ingestion method based on available files or force specific method
        """
        json_file = "./rag_ready_faq/langchain_documents.json"
        excel_files = list(Path(self.faq_folder).glob("*.xlsx"))
        
        # If force_method is specified, use that method
        if self.force_method:
            method = self.force_method.lower()
            self._print(f"🎯 Force method specified: {method}")
            
            if method == 'json':
                if os.path.exists(json_file):
                    self._print("✨ Using forced JSON ingestion.")
                    return self.ingest_json_file(json_file)
                else:
                    self._print(f"❌ JSON file not found: {json_file}")
                    return False
                    
            elif method == 'excel':
                if excel_files:
                    self._print("📊 Using forced Excel ingestion.")
                    return self.ingest_excel_files()
                else:
                    self._print(f"❌ No Excel files found in {self.faq_folder}")
                    return False
                    
            elif method == 'markdown':
                if os.path.exists(self.markdown_file):
                    self._print("📝 Using forced markdown ingestion.")
                    return self.ingest_markdown_file()
                else:
                    self._print(f"❌ Markdown file not found: {self.markdown_file}")
                    return False
                    
            elif method == 'docx':
                self._print("📄 Using forced DOCX ingestion.")
                return self.ingest_docx_documents()
                
            elif method == 'location-specific':
                self._print("🎯 Using forced location-specific ingestion.")
                return self.ingest_location_specific_files()
                
            else:
                self._print(f"❌ Invalid force_method: {method}. Valid options: json, excel, markdown, docx, location-specific")
                return False
        
        # Auto-selection based on available files (original logic)
        self._print("🔍 Auto-detecting best ingestion method...")
        
        # Check for JSON file first (best quality)
        if os.path.exists(json_file):
            self._print("✨ Found structured JSON file! Using this for best quality ingestion.")
            return self.ingest_json_file(json_file)
        # Check for Excel files next (structured data)
        elif excel_files:
            self._print("📊 Found Excel files! Using Excel data for structured ingestion.")
            return self.ingest_excel_files()
        # Check for consolidated markdown next
        elif os.path.exists(self.markdown_file):
            self._print("📝 Found consolidated FAQ markdown file!")
            return self.ingest_markdown_file()
        # Fallback to docx
        else:
            self._print("📄 Using original .docx files (no structured files found)")
            return self.ingest_docx_documents()

def main():
    """Main entry point for document ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Leo & Loona FAQ + Infrastructure Document Ingestion')
    parser.add_argument('--method', choices=['json', 'excel', 'markdown', 'docx', 'location-specific'], 
                       help='Force specific ingestion method (default: auto-detect)')
    parser.add_argument('--chunk-size', type=int, default=2800,
                       help='Chunk size for document splitting (default: 2800)')
    parser.add_argument('--chunk-overlap', type=int, default=500,
                       help='Chunk overlap for document splitting (default: 500)')
    parser.add_argument('--infrastructure-folder', type=str, default='./Park infrastructure',
                       help='Path to Park Infrastructure folder (default: ./Park infrastructure)')
    parser.add_argument('--no-tests', action='store_true',
                       help='Skip testing after ingestion')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    ingestor = DocumentIngestor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        infrastructure_folder=args.infrastructure_folder,
        force_method=args.method,
        run_tests=not args.no_tests,
        verbose=not args.quiet
    )
    
    if args.method:
        print(f"🎯 Forced ingestion method: {args.method}")
    else:
        print("🔍 Auto-detecting best ingestion method...")
    
    success = ingestor.auto_ingest()
    
    if success:
        print(f"\n🎊 SUCCESS! Your Leo & Loona FAQ system is ready.")
        print(f"   You can now run your Streamlit app to start answering questions.")
        if args.method:
            print(f"   Used method: {args.method}")
    else:
        print(f"\n💥 FAILED! Please check the errors above and try again.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)