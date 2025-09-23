#!/usr/bin/env python3
"""
Delete and Re-ingest Specific Document Utility

This utility allows you to:
1. Delete all chunks from a specific source file in Pinecone
2. Re-ingest the updated version of that file

Usage:
    python delete_and_reingest.py festival_city_specific.md
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from pinecone import Pinecone
except ImportError:
    from pinecone import Pinecone

class DocumentManager:
    """Manage document deletion and re-ingestion in Pinecone"""

    def __init__(self):
        load_dotenv()

        # Initialize Pinecone
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')
        self.index_name = os.getenv('PINECONE_INDEX_NAME', 'leo-loona-faq')

        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment")

        # Initialize embeddings
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        else:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            google_key = os.getenv('GOOGLE_API_KEY')
            if not google_key:
                raise ValueError("Either OPENAI_API_KEY or GOOGLE_API_KEY required")
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=google_key
            )

        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(self.index_name)

        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )

        print(f"🔗 Connected to Pinecone index: {self.index_name}")

    def get_document_ids_by_source(self, source_filename: str) -> list:
        """
        Get all document IDs that match a specific source filename

        Args:
            source_filename: Name of the source file (e.g., 'festival_city_specific.md')

        Returns:
            List of document IDs to delete
        """
        print(f"🔍 Searching for chunks from source: {source_filename}")

        try:
            # Query Pinecone to find documents with matching metadata
            # Note: This is a workaround since Pinecone doesn't have direct metadata search
            # We'll need to fetch documents and filter by metadata

            # Get index stats first
            stats = self.index.describe_index_stats()
            total_vectors = stats['total_vector_count']
            print(f"📊 Total vectors in index: {total_vectors}")

            # We'll do a broad query and filter results
            # This is not the most efficient but works for the scale
            dummy_query_vector = [0.0] * 1536  # OpenAI embedding dimension

            # Query in batches to find matching documents
            matching_ids = []
            batch_size = 100

            for i in range(0, min(total_vectors, 1000), batch_size):  # Limit to first 1000 for safety
                try:
                    results = self.index.query(
                        vector=dummy_query_vector,
                        top_k=batch_size,
                        include_metadata=True,
                        filter=None  # We'll filter in code
                    )

                    for match in results.get('matches', []):
                        metadata = match.get('metadata', {})
                        doc_source = metadata.get('source', '')
                        doc_filename = metadata.get('filename', '')

                        # Check if this document is from our target source
                        if (source_filename in doc_source or
                            source_filename in doc_filename or
                            source_filename == os.path.basename(doc_source) or
                            source_filename == os.path.basename(doc_filename)):

                            matching_ids.append(match['id'])
                            print(f"   📄 Found chunk: {match['id']} from {doc_source}")

                except Exception as e:
                    print(f"⚠️ Error in batch query: {e}")
                    break

            print(f"✅ Found {len(matching_ids)} chunks to delete")
            return matching_ids

        except Exception as e:
            print(f"❌ Error searching for documents: {e}")
            return []

    def delete_documents_by_ids(self, document_ids: list) -> bool:
        """
        Delete documents from Pinecone by their IDs

        Args:
            document_ids: List of document IDs to delete

        Returns:
            True if successful, False otherwise
        """
        if not document_ids:
            print("ℹ️ No documents to delete")
            return True

        print(f"🗑️ Deleting {len(document_ids)} document chunks...")

        try:
            # Delete in batches (Pinecone has limits)
            batch_size = 100
            for i in range(0, len(document_ids), batch_size):
                batch = document_ids[i:i + batch_size]
                self.index.delete(ids=batch)
                print(f"   ✅ Deleted batch {i//batch_size + 1}: {len(batch)} chunks")

            print(f"✅ Successfully deleted all {len(document_ids)} chunks")
            return True

        except Exception as e:
            print(f"❌ Error deleting documents: {e}")
            return False

    def ingest_markdown_file(self, file_path: str, location: str = "FESTIVAL_CITY") -> bool:
        """
        Ingest a markdown file into the vector store

        Args:
            file_path: Path to the markdown file
            location: Location code for metadata

        Returns:
            True if successful, False otherwise
        """
        print(f"📥 Ingesting file: {file_path}")

        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            filename = os.path.basename(file_path)

            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    'source': file_path,
                    'filename': filename,
                    'location': location,
                    'content_type': 'FAQ',
                    'source_folder': 'rag_ready_faq'
                }
            )

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2800,
                chunk_overlap=500,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )

            chunks = text_splitter.split_documents([doc])

            # Ensure all chunks have proper metadata
            for chunk in chunks:
                chunk.metadata.update({
                    'source': file_path,
                    'filename': filename,
                    'location': location,
                    'content_type': 'FAQ',
                    'source_folder': 'rag_ready_faq'
                })

            print(f"📄 Split into {len(chunks)} chunks")

            # Add to vector store
            self.vector_store.add_documents(chunks)

            print(f"✅ Successfully ingested {len(chunks)} chunks from {filename}")
            return True

        except Exception as e:
            print(f"❌ Error ingesting file: {e}")
            return False

    def delete_and_reingest(self, source_filename: str, file_path: str, location: str = "FESTIVAL_CITY") -> bool:
        """
        Complete workflow: delete existing chunks and re-ingest updated file

        Args:
            source_filename: Name of source file to delete (e.g., 'festival_city_specific.md')
            file_path: Path to the updated file to ingest
            location: Location code for metadata

        Returns:
            True if successful, False otherwise
        """
        print(f"🔄 Starting delete and re-ingest for: {source_filename}")
        print("=" * 60)

        # Step 1: Find and delete existing chunks
        document_ids = self.get_document_ids_by_source(source_filename)

        if document_ids:
            if not self.delete_documents_by_ids(document_ids):
                print("❌ Failed to delete existing documents")
                return False
        else:
            print("ℹ️ No existing documents found to delete")

        print("\n" + "=" * 60)

        # Step 2: Re-ingest the updated file
        if not self.ingest_markdown_file(file_path, location):
            print("❌ Failed to ingest updated file")
            return False

        print("\n" + "=" * 60)
        print("✅ Delete and re-ingest completed successfully!")
        return True

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python delete_and_reingest.py <filename>")
        print("Example: python delete_and_reingest.py festival_city_specific.md")
        sys.exit(1)

    filename = sys.argv[1]
    file_path = f"./rag_ready_faq/{filename}"

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        sys.exit(1)

    try:
        manager = DocumentManager()
        success = manager.delete_and_reingest(filename, file_path)

        if success:
            print(f"\n🎉 Successfully updated {filename} in vector database!")
            sys.exit(0)
        else:
            print(f"\n💥 Failed to update {filename}")
            sys.exit(1)

    except Exception as e:
        print(f"💥 Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()