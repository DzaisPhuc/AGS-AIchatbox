from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from pymilvus import Collection, utility, connections
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
dotenv_path = os.path.join(project_root, '.env')

load_dotenv(dotenv_path=dotenv_path)

client = OpenAI()

class MyCustomToolInput(BaseModel):
    argument: str = Field(..., description="The natural language query to search in the knowledge base.")

class MyCustomTool(BaseTool):
    name: str = "Milvus Knowledge Search"
    description: str = (
        "A tool that performs semantic search over documents embedded in Milvus."
        " Given a natural language query, it returns the most relevant information."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        try:
            # Establish connection to Milvus
            try:
                connections.connect("default", host="localhost", port="19530")
            except Exception as conn_error:
                # Connection might already exist, check if we can proceed
                pass
            
            # Check if collection exists
            if not utility.has_collection("knowledge_base"):
                return "Knowledge base is not available. Please ensure documents are ingested first by running: python ingest_docs.py"
            
            # Generate embedding for the query
            embedding = client.embeddings.create(
                input=argument,
                model="text-embedding-3-small"
            ).data[0].embedding

            # Load Milvus collection
            collection = Collection("knowledge_base")
            
            # Load collection into memory
            collection.load()
            
            # Perform semantic search
            results = collection.search(
                data=[embedding],
                anns_field="vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=5,
                output_fields=["content"]
            )

            if not results[0]:
                return "No relevant information found in the knowledge base."

            hits = results[0]
            answer = "\n---\n".join(hit.entity.get("content") for hit in hits)
            return f"Here are the relevant details from our knowledge base:\n\n{answer}"
            
        except Exception as e:
            return f"Error accessing knowledge base: {str(e)}. Please ensure Milvus is running and documents are ingested by running: python ingest_docs.py"

    # def cache_function(self, *args, **kwargs) -> bool:
    #     return False    