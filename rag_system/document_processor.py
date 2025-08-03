import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Handles document loading and processing from FAQ folder using LangChain loaders"""
    
    def __init__(self, faq_folder_path):
        self.faq_folder_path = faq_folder_path
    
    def _extract_location_from_filename(self, filename):
        """Extract location information from filename"""
        filename_upper = filename.upper()
        
        if 'DALMA' in filename_upper:
            return 'DALMA_MALL'
        elif 'YAS' in filename_upper:
            return 'YAS_MALL'
        elif 'FESTIVAL' in filename_upper:
            return 'FESTIVAL_CITY'
        elif 'HELLO' in filename_upper:
            return 'HELLO_PARK'
        else:
            return 'UNKNOWN_LOCATION'
    
    def load_docx_files(self):
        """Load all .docx files from the FAQ folder using LangChain UnstructuredWordDocumentLoader"""
        documents = []
        
        for filename in os.listdir(self.faq_folder_path):
            if filename.endswith('.docx') and not filename.startswith('~'):
                file_path = os.path.join(self.faq_folder_path, filename)
                try:
                    # Use LangChain's UnstructuredWordDocumentLoader
                    loader = UnstructuredWordDocumentLoader(file_path)
                    docs = loader.load()
                    
                    # Extract location from filename
                    location = self._extract_location_from_filename(filename)
                    
                    # Process each document and enhance metadata
                    for doc in docs:
                        # Ensure we have the filename in metadata for future deletion
                        doc.metadata.update({
                            'filename': filename,
                            'file_path': file_path,
                            'source': filename,  # Keep source for backward compatibility
                            'document_type': 'docx',
                            'loader_type': 'UnstructuredWordDocumentLoader',
                            'location': location,  # Add location metadata
                            'location_display': location.replace('_', ' ').title()  # Human readable
                        })
                        
                        # Only add documents with content
                        if doc.page_content.strip():
                            documents.append(doc)
                            print(f"Loaded: {filename} [{location}] ({len(doc.page_content)} characters)")
                        
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        print(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def load_docx_files_with_elements(self):
        """Load .docx files with element-level separation for better structure preservation"""
        documents = []
        
        for filename in os.listdir(self.faq_folder_path):
            if filename.endswith('.docx') and not filename.startswith('~'):
                file_path = os.path.join(self.faq_folder_path, filename)
                try:
                    # Use mode="elements" to preserve document structure
                    loader = UnstructuredWordDocumentLoader(file_path, mode="elements")
                    docs = loader.load()
                    
                    # Process each element and enhance metadata
                    for i, doc in enumerate(docs):
                        doc.metadata.update({
                            'filename': filename,
                            'file_path': file_path,
                            'source': filename,
                            'document_type': 'docx',
                            'loader_type': 'UnstructuredWordDocumentLoader',
                            'element_index': i,
                            'total_elements': len(docs)
                        })
                        
                        # Only add elements with content
                        if doc.page_content.strip():
                            documents.append(doc)
                    
                    print(f"Loaded: {filename} ({len(docs)} elements)")
                        
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        print(f"Successfully loaded {len(documents)} document elements")
        return documents
    
    def split_documents(self, documents, chunk_size=1800, chunk_overlap=300):
        """Split documents into larger chunks for better context and retrieval"""
        # Increased chunk size from 1000 to 1800 chars for better context
        # Increased overlap from 200 to 300 chars to maintain continuity
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(documents)
        
        # Preserve filename and location in chunk metadata
        for chunk in split_docs:
            if 'filename' not in chunk.metadata and 'source' in chunk.metadata:
                chunk.metadata['filename'] = chunk.metadata['source']
            
            # Ensure location metadata is preserved in chunks
            if 'location' not in chunk.metadata and 'filename' in chunk.metadata:
                chunk.metadata['location'] = self._extract_location_from_filename(chunk.metadata['filename'])
                chunk.metadata['location_display'] = chunk.metadata['location'].replace('_', ' ').title()
        
        # Calculate average chunk size for reporting
        chunk_sizes = [len(doc.page_content) for doc in split_docs]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        print(f"Average chunk size: {avg_size:.0f} characters (target: {chunk_size})")
        return split_docs
    
    def get_documents_by_filename(self, documents, filename):
        """Get all document chunks by filename - useful for deletion"""
        return [doc for doc in documents if doc.metadata.get('filename') == filename]