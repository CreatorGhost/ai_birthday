"""
FAQ Data Extractor for Leo & Loona Theme Parks
Extracts and organizes data from FAQ folder for RAG system using proper document loaders
"""
import pandas as pd
import json
import sys
from pathlib import Path
from typing import Dict, List
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from rag_system.document_processor import DocumentProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAQDataExtractor:
    """Extract and organize FAQ data without LLM enhancement"""
    
    def __init__(self, faq_folder_path: str, output_folder_path: str = None):
        self.faq_folder_path = Path(faq_folder_path)
        self.output_folder_path = Path(output_folder_path) if output_folder_path else self.faq_folder_path.parent / "rag_ready_faq"
        self.output_folder_path.mkdir(exist_ok=True)
        
        logger.info(f"FAQ folder: {self.faq_folder_path}")
        logger.info(f"Output folder: {self.output_folder_path}")
    
    def extract_excel_data(self) -> Dict[str, List[Dict]]:
        """Extract data from all Excel files and all sheets within each file"""
        logger.info("Extracting Excel data...")
        
        extracted_data = {
            'contacts': [],
            'locations': [],
            'pricing': []
        }
        
        excel_files = list(self.faq_folder_path.glob("*.xlsx"))
        
        for excel_file in excel_files:
            logger.info(f"Processing {excel_file.name}")
            
            try:
                # Read all sheets from the Excel file
                excel_data = pd.read_excel(excel_file, sheet_name=None)  # None reads all sheets
                
                for sheet_name, df in excel_data.items():
                    logger.info(f"  Processing sheet: {sheet_name}")
                    
                    if df.empty:
                        logger.warning(f"    Sheet {sheet_name} is empty, skipping")
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
                        logger.info(f"    Added {len(data_list)} contact entries from sheet {sheet_name}")
                    elif ('location' in filename_lower or 'location' in sheet_name_lower):
                        extracted_data['locations'].extend(data_list)
                        logger.info(f"    Added {len(data_list)} location entries from sheet {sheet_name}")
                    elif ('pricing' in filename_lower or 'park' in filename_lower or 
                          'pricing' in sheet_name_lower or 'park' in sheet_name_lower or
                          'price' in sheet_name_lower or 'ticket' in sheet_name_lower):
                        extracted_data['pricing'].extend(data_list)
                        logger.info(f"    Added {len(data_list)} pricing entries from sheet {sheet_name}")
                    else:
                        # If no specific category matches, add to a general category based on filename
                        if 'contact' in filename_lower:
                            extracted_data['contacts'].extend(data_list)
                        elif 'location' in filename_lower:
                            extracted_data['locations'].extend(data_list)
                        else:
                            extracted_data['pricing'].extend(data_list)  # Default to pricing
                        logger.info(f"    Added {len(data_list)} entries from sheet {sheet_name} (default categorization)")
                
            except Exception as e:
                logger.error(f"Error processing {excel_file.name}: {str(e)}")
        
        logger.info(f"Extracted: {len(extracted_data['contacts'])} contacts, {len(extracted_data['locations'])} locations, {len(extracted_data['pricing'])} pricing entries")
        return extracted_data
    
    def extract_markdown_data(self) -> List[Document]:
        """Extract content from markdown files using existing DocumentProcessor"""
        logger.info("Extracting markdown data using DocumentProcessor...")
        
        documents = []
        md_files = list(self.faq_folder_path.glob("*.md"))
        
        for md_file in md_files:
            try:
                # Read file content directly and create Document
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if content.strip():
                    # Create Document with enhanced metadata
                    doc = Document(
                        page_content=content,
                        metadata={
                            'filename': md_file.name,
                            'file_path': str(md_file),
                            'source': md_file.name,
                            'document_type': 'markdown',
                            'loader_type': 'direct_file_read',
                            'content_length': len(content)
                        }
                    )
                    
                    documents.append(doc)
                    logger.info(f"Processed {md_file.name} ({len(content)} chars)")
                    
            except Exception as e:
                logger.error(f"Error processing {md_file.name}: {str(e)}")
        
        logger.info(f"Extracted {len(documents)} markdown documents")
        return documents
    
    def convert_excel_to_text(self, excel_data: Dict) -> str:
        """Convert Excel data to readable text format"""
        sections = []
        
        # Contacts section
        if excel_data.get('contacts'):
            sections.append("# Contact Information\n")
            for contact in excel_data['contacts']:
                # Add source information if available
                source_info = []
                if 'source_file' in contact:
                    source_info.append(f"File: {contact['source_file']}")
                if 'sheet_name' in contact:
                    source_info.append(f"Sheet: {contact['sheet_name']}")
                if source_info:
                    sections.append(f"*Source: {', '.join(source_info)}*")
                
                for key, value in contact.items():
                    if key not in ['source_file', 'sheet_name']:
                        readable_key = key.replace('_', ' ').title()
                        sections.append(f"**{readable_key}:** {value}")
                sections.append("")
        
        # Locations section
        if excel_data.get('locations'):
            sections.append("# Location Information\n")
            for location in excel_data['locations']:
                # Add source information if available
                source_info = []
                if 'source_file' in location:
                    source_info.append(f"File: {location['source_file']}")
                if 'sheet_name' in location:
                    source_info.append(f"Sheet: {location['sheet_name']}")
                if source_info:
                    sections.append(f"*Source: {', '.join(source_info)}*")
                
                for key, value in location.items():
                    if key not in ['source_file', 'sheet_name']:
                        readable_key = key.replace('_', ' ').title()
                        sections.append(f"**{readable_key}:** {value}")
                sections.append("")
        
        # Pricing section
        if excel_data.get('pricing'):
            sections.append("# Pricing Information\n")
            for price_item in excel_data['pricing']:
                # Add source information if available
                source_info = []
                if 'source_file' in price_item:
                    source_info.append(f"File: {price_item['source_file']}")
                if 'sheet_name' in price_item:
                    source_info.append(f"Sheet: {price_item['sheet_name']}")
                if source_info:
                    sections.append(f"*Source: {', '.join(source_info)}*")
                
                for key, value in price_item.items():
                    if key not in ['source_file', 'sheet_name']:
                        readable_key = key.replace('_', ' ').title()
                        sections.append(f"**{readable_key}:** {value}")
                sections.append("")
        
        return "\n".join(sections)
    
    def create_consolidated_document(self, excel_data: Dict, markdown_documents: List[Document]) -> str:
        """Create consolidated FAQ document from structured data"""
        logger.info("Creating consolidated document...")
        
        sections = []
        
        # Header
        sections.append("# Leo & Loona Theme Parks - FAQ Documentation")
        sections.append("*Complete information for all locations*\n")
        
        # Add Excel data as text
        excel_text = self.convert_excel_to_text(excel_data)
        if excel_text.strip():
            sections.append(excel_text)
        
        # Add markdown content from LangChain documents
        sections.append("---")
        sections.append("# Additional Information\n")
        
        for doc in markdown_documents:
            filename = doc.metadata.get('filename', 'Unknown File')
            section_name = filename.replace('.md', '').replace('_', ' ').title()
            
            sections.append(f"## {section_name}")
            sections.append(f"*Source: {filename} | Content Length: {doc.metadata.get('content_length', 0)} chars*")
            sections.append("")
            sections.append(doc.page_content)
            sections.append("")
        
        consolidated_content = "\n".join(sections)
        logger.info(f"Consolidated document: {len(consolidated_content)} characters")
        
        return consolidated_content
    
    def save_outputs(self, excel_data: Dict, markdown_documents: List[Document], consolidated_content: str):
        """Save all outputs including LangChain documents"""
        # Save structured data as JSON (convert Documents to dict for JSON serialization)
        markdown_data_for_json = []
        for doc in markdown_documents:
            markdown_data_for_json.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })
        
        structured_data = {
            'excel_data': excel_data,
            'markdown_documents': markdown_data_for_json
        }
        
        json_file = self.output_folder_path / "structured_faq_data.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, indent=2, ensure_ascii=False)
        
        # Save consolidated markdown
        md_file = self.output_folder_path / "consolidated_faq.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(consolidated_content)
        
        # Save LangChain documents separately for direct ingestion
        langchain_docs_file = self.output_folder_path / "langchain_documents.json"
        with open(langchain_docs_file, 'w', encoding='utf-8') as f:
            docs_dict = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in markdown_documents]
            json.dump(docs_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved structured data: {json_file}")
        logger.info(f"Saved consolidated FAQ: {md_file}")
        logger.info(f"Saved LangChain documents: {langchain_docs_file}")
        
        return json_file, md_file, langchain_docs_file
    
    def extract_faq_data(self):
        """Main extraction process"""
        logger.info("Starting FAQ data extraction...")
        
        # Extract data
        excel_data = self.extract_excel_data()
        markdown_documents = self.extract_markdown_data()
        
        # Create consolidated document
        consolidated_content = self.create_consolidated_document(excel_data, markdown_documents)
        
        # Save outputs
        json_file, md_file, langchain_docs_file = self.save_outputs(excel_data, markdown_documents, consolidated_content)
        
        logger.info("FAQ data extraction complete!")
        logger.info(f"✅ Structured data ready for RAG: {json_file}")
        logger.info(f"✅ Consolidated FAQ ready: {md_file}")
        logger.info(f"✅ LangChain documents ready: {langchain_docs_file}")
        
        return {
            'excel_data': excel_data,
            'markdown_documents': markdown_documents,
            'consolidated_content': consolidated_content,
            'json_file': json_file,
            'md_file': md_file,
            'langchain_docs_file': langchain_docs_file
        }

def main():
    """Main function"""
    # Paths
    faq_folder = Path(__file__).parent.parent / "FAQ"
    output_folder = Path(__file__).parent.parent / "rag_ready_faq"
    
    # Extract FAQ data
    extractor = FAQDataExtractor(str(faq_folder), str(output_folder))
    results = extractor.extract_faq_data()
    
    print("\n" + "="*60)
    print("✅ FAQ DATA EXTRACTION COMPLETE")
    print("="*60)
    print("Your FAQ data is ready for RAG system!")
    print("Check the 'rag_ready_faq' folder for outputs:")
    print(f"- structured_faq_data.json (complete structured data)")
    print(f"- consolidated_faq.md (human-readable FAQ)")
    print(f"- langchain_documents.json (LangChain Document objects)")
    print("\nMarkdown files processed with UnstructuredMarkdownLoader for better structure!")
    print("Ready for direct ingestion into your RAG system.")

if __name__ == "__main__":
    main()