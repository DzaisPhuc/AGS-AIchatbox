import os
import openai
from openai import OpenAI
import tiktoken
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from langchain.text_splitter import RecursiveCharacterTextSplitter
from uuid import uuid4
from PyPDF2 import PdfReader
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
dotenv_path = os.path.join(project_root, '.env')

load_dotenv(dotenv_path)

client = OpenAI()

# Establish connection to Milvus
try:
    connections.connect("default", host="localhost", port="19530")
    print("Connected to Milvus successfully")
except Exception as e:
    print(f"Failed to connect to Milvus: {e}")

collection_name = "knowledge_base"

def num_tokens_from_string(string: str, model_name="text-embedding-3-small") -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))

def embed_texts(texts: list[str]) -> list[list[float]]:
    return [client.embeddings.create(input=chunk, model="text-embedding-3-small").data[0].embedding
            for chunk in texts]

def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(path):
    try:
        text = ""
        reader = PdfReader(path)
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading PDF {path}: {e}")
        return ""

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

def create_collection():
    # Drop existing collection if it exists to recreate it properly
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        collection.drop()
        print(f"Dropped existing collection: {collection_name}")

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=36),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
    ]
    schema = CollectionSchema(fields=fields, description="AI Knowledge Base")
    collection = Collection(name=collection_name, schema=schema)
    
    # Create index after collection is created
    index_params = {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="vector", index_params=index_params)
    print(f"Created collection: {collection_name} with index")
    
    return collection

def ingest_documents(folder_path="knowledge"):
    if not os.path.exists(folder_path):
        print(f"Knowledge folder '{folder_path}' not found!")
        return
    # Check if the collection already exists
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        collection.drop()
        print(f"Dropped existing collection: {collection_name}")

    collection = create_collection()
    
    files_processed = 0
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        ext = os.path.splitext(file)[-1].lower()

        if ext == ".txt":
            text = load_txt(full_path)
        elif ext == ".pdf":
            text = load_pdf(full_path)
        else:
            print(f"Skipping unsupported file type: {file}")
            continue

        if not text.strip():
            print(f"No text extracted from {file}")
            continue

        chunks = split_text(text)
        if not chunks:
            print(f"No chunks created from {file}")
            continue
            
        vectors = embed_texts(chunks)
        ids = [str(uuid4()) for _ in chunks]

        collection.insert([ids, chunks, vectors])
        print(f"[+] Inserted {len(chunks)} chunks from {file}")
        files_processed += 1

    if files_processed > 0:
        collection.flush()
        print(f"Collection flushed successfully. Processed {files_processed} files.")
    else:
        print("No files were processed!")