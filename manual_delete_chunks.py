#!/usr/bin/env python3
"""
Manual chunk deletion utility

Use this if you want to manually find and delete specific chunks
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

def manual_delete_festival_chunks():
    """Manually delete festival_city_specific.md chunks"""
    load_dotenv()

    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(os.getenv('PINECONE_INDEX_NAME', 'leo-loona-faq'))

    print("🔍 Manual deletion of festival_city_specific.md chunks")

    # Option 1: Delete by namespace (if you use namespaces)
    # index.delete(delete_all=True, namespace="festival_city")

    # Option 2: Delete specific IDs (if you know them)
    # chunk_ids = ["chunk_id_1", "chunk_id_2", ...]
    # index.delete(ids=chunk_ids)

    # Option 3: Delete by metadata filter (most accurate)
    try:
        # This requires that your chunks have filterable metadata
        result = index.delete(
            filter={
                "filename": {"$eq": "festival_city_specific.md"}
            }
        )
        print(f"✅ Deleted chunks with filename = festival_city_specific.md")
        print(f"Result: {result}")

    except Exception as e:
        print(f"❌ Metadata filtering failed: {e}")
        print("💡 Try the main delete_and_reingest.py script instead")

if __name__ == "__main__":
    manual_delete_festival_chunks()