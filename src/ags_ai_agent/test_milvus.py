from tools.vector_db import ingest_documents

if __name__ == "__main__":
    print("Starting document ingestion...")
    ingest_documents("knowledge")
    print("Document ingestion completed!")