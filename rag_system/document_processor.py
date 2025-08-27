import os
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Handles document loading and processing from FAQ folder using LangChain loaders"""
    
    def __init__(self, faq_folder_path, infrastructure_folder_path=None):
        self.faq_folder_path = faq_folder_path
        self.infrastructure_folder_path = infrastructure_folder_path
    
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
            return 'GENERAL'  # Changed from HELLO_PARK to GENERAL for better semantics
        else:
            return 'UNKNOWN_LOCATION'
    
    def _get_location_display_name(self, location_code):
        """Get human-readable location name"""
        location_mapping = {
            'DALMA_MALL': 'Dalma Mall',
            'YAS_MALL': 'Yas Mall', 
            'FESTIVAL_CITY': 'Festival City',
            'GENERAL': 'All Locations'
        }
        return location_mapping.get(location_code, location_code)
    
    def _get_content_type_from_filename(self, filename):
        """Extract content type from filename"""
        filename_upper = filename.upper()
        
        if 'FAQ' in filename_upper:
            return 'FAQ'
        elif 'WAIVER' in filename_upper:
            return 'Waiver Information'
        elif 'PARK RULES' in filename_upper or 'RULES' in filename_upper:
            return 'Park Rules'
        elif 'INFRASTRUCTURE' in filename_upper:
            return 'Infrastructure Information'
        else:
            return 'General Information'
    
    def _enhance_content_with_context(self, content, location_code, content_type, filename):
        """Add location and content context to document content for better retrieval"""
        location_display = self._get_location_display_name(location_code)
        
        # Create context header
        if location_code == 'GENERAL':
            context_header = f"LOCATION: {location_display} | TYPE: {content_type}\n"
            context_header += "This information applies to all Leo & Loona locations unless specified otherwise.\n\n"
        else:
            context_header = f"LOCATION: {location_display} | TYPE: {content_type}\n"
            context_header += f"This information is specific to Leo & Loona at {location_display}.\n\n"
        
        # Add location keywords for better search
        location_keywords = ""
        if location_code == 'DALMA_MALL':
            location_keywords = "Keywords: Dalma Mall, Abu Dhabi, Dalma"
            if content_type == 'Infrastructure Information':
                location_keywords += ", infrastructure, facilities, equipment, layout\n"
            else:
                location_keywords += "\n"
        elif location_code == 'YAS_MALL':
            location_keywords = "Keywords: Yas Mall, Abu Dhabi, Yas, POD holder discounts"
            if content_type == 'Infrastructure Information':
                location_keywords += ", infrastructure, facilities, equipment, layout\n"
            else:
                location_keywords += "\n"
        elif location_code == 'FESTIVAL_CITY':
            location_keywords = "Keywords: Festival City, Dubai, Festival"
            if content_type == 'Infrastructure Information':
                location_keywords += ", infrastructure, facilities, equipment, layout\n"
            else:
                location_keywords += "\n"
        elif location_code == 'GENERAL':
            location_keywords = "Keywords: All locations, General information, Leo Loona\n"
        
        # Combine context + keywords + original content
        enhanced_content = context_header + location_keywords + "\n" + content
        
        return enhanced_content
    
    def load_docx_files(self):
        """Load all .docx files from the FAQ folder with enhanced location context"""
        documents = []
        
        for filename in os.listdir(self.faq_folder_path):
            if filename.endswith('.docx') and not filename.startswith('~'):
                file_path = os.path.join(self.faq_folder_path, filename)
                try:
                    # Use LangChain's UnstructuredWordDocumentLoader
                    loader = UnstructuredWordDocumentLoader(file_path)
                    docs = loader.load()
                    
                    # Extract location and content type from filename
                    location_code = self._extract_location_from_filename(filename)
                    content_type = self._get_content_type_from_filename(filename)
                    location_display = self._get_location_display_name(location_code)
                    
                    # Process each document and enhance metadata
                    for doc in docs:
                        if doc.page_content.strip():
                            # Enhance content with location context
                            enhanced_content = self._enhance_content_with_context(
                                doc.page_content, location_code, content_type, filename
                            )
                            
                            # Update document content
                            doc.page_content = enhanced_content
                            
                            # Enhanced metadata
                            doc.metadata.update({
                                'filename': filename,
                                'file_path': file_path,
                                'source': filename,
                                'document_type': 'docx',
                                'loader_type': 'UnstructuredWordDocumentLoader',
                                'location': location_code,
                                'location_display': location_display,
                                'content_type': content_type,
                                'enhanced': True  # Flag to indicate enhanced processing
                            })
                            
                            documents.append(doc)
                            print(f"Enhanced: {filename} [{location_display}] - {content_type} ({len(enhanced_content)} chars)")
                        
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        print(f"Successfully loaded {len(documents)} enhanced documents")
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
                location_code = self._extract_location_from_filename(chunk.metadata['filename'])
                chunk.metadata['location'] = location_code
                chunk.metadata['location_display'] = self._get_location_display_name(location_code)
                
            # Ensure content type is preserved in chunks
            if 'content_type' not in chunk.metadata and 'filename' in chunk.metadata:
                chunk.metadata['content_type'] = self._get_content_type_from_filename(chunk.metadata['filename'])
        
        # Calculate average chunk size for reporting
        chunk_sizes = [len(doc.page_content) for doc in split_docs]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        print(f"Average chunk size: {avg_size:.0f} characters (target: {chunk_size})")
        return split_docs
    
    def get_documents_by_filename(self, documents, filename):
        """Get all document chunks by filename - useful for deletion"""
        return [doc for doc in documents if doc.metadata.get('filename') == filename]
    
    def load_docx_from_folder(self, folder_path):
        """Load all .docx files from a specific folder with enhanced location context"""
        documents = []
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            return documents
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.docx') and not filename.startswith('~'):
                file_path = os.path.join(folder_path, filename)
                try:
                    # Use LangChain's UnstructuredWordDocumentLoader
                    loader = UnstructuredWordDocumentLoader(file_path)
                    docs = loader.load()
                    
                    # Extract location and content type from filename
                    location_code = self._extract_location_from_filename(filename)
                    content_type = self._get_content_type_from_filename(filename)
                    location_display = self._get_location_display_name(location_code)
                    
                    # Process each document and enhance metadata
                    for doc in docs:
                        if doc.page_content.strip():
                            # Enhance content with location context
                            enhanced_content = self._enhance_content_with_context(
                                doc.page_content, location_code, content_type, filename
                            )
                            
                            # Update document content
                            doc.page_content = enhanced_content
                            
                            # Enhanced metadata
                            doc.metadata.update({
                                'filename': filename,
                                'file_path': file_path,
                                'source': filename,
                                'document_type': 'docx',
                                'loader_type': 'UnstructuredWordDocumentLoader',
                                'location': location_code,
                                'location_display': location_display,
                                'content_type': content_type,
                                'enhanced': True,  # Flag to indicate enhanced processing
                                'source_folder': os.path.basename(folder_path)  # Track source folder
                            })
                            
                            documents.append(doc)
                            print(f"Enhanced: {filename} [{location_display}] - {content_type} ({len(enhanced_content)} chars)")
                        
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
        
        return documents
    
    def load_all_docx_files(self):
        """Load all .docx files from FAQ folder and Infrastructure folder if specified"""
        all_documents = []
        
        # Load FAQ documents
        print(f"Loading FAQ documents from: {self.faq_folder_path}")
        faq_documents = self.load_docx_from_folder(self.faq_folder_path)
        all_documents.extend(faq_documents)
        
        # Load Infrastructure documents if path is specified
        if self.infrastructure_folder_path:
            print(f"Loading Infrastructure documents from: {self.infrastructure_folder_path}")
            infra_documents = self.load_docx_from_folder(self.infrastructure_folder_path)
            all_documents.extend(infra_documents)
        
        print(f"Successfully loaded {len(all_documents)} enhanced documents total")
        print(f"  - FAQ documents: {len(faq_documents)}")
        if self.infrastructure_folder_path:
            print(f"  - Infrastructure documents: {len(infra_documents)}")
        
        return all_documents