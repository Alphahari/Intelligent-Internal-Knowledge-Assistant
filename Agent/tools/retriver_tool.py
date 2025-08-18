from langchain_core.tools import Tool
from pydantic import BaseModel
from typing import List
import json
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
import os

class Metadata(BaseModel):
    source: str
    page: int | None = None

class StructuredOutput(BaseModel):
    content: str
    metadata: Metadata

pc = Pinecone(api_key=os.getenv("PINECONE_API"))
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5", 
    model_kwargs={'device': 'cpu'}, 
    encode_kwargs={'normalize_embeddings': True}
)
bm25 = BM25Encoder().default()

def create_retriever_tool(index_name: str, tool_name: str, description: str) -> Tool:
    """Create retrieval tool for existing Pinecone index"""
    index = pc.Index(index_name)
    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25,
        index=index,
        text_key="text"
    )
    
    def search(query: str) -> List[StructuredOutput]:
        results = retriever.invoke(query)
        return [
            StructuredOutput(
                content=res.page_content,
                metadata=Metadata(
                    source=res.metadata.get("source", "unknown"),
                    page=res.metadata.get("page")
                )
            ) for res in results
        ]
    
    def tool_wrapper(query: str) -> str:
        outputs = search(query)
        return "\n\n".join(
            f"{o.content}\n[Source: {o.metadata.source}, Page: {o.metadata.page}]" 
            for o in outputs
        )
    
    return Tool(name=tool_name, func=tool_wrapper, description=description)