import os
import time
import json
from uuid import uuid4
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.tools import Tool

class Metadata(BaseModel):
    source: str
    page: int | None = None

class StructuredOutput(BaseModel):
    content: str
    metadata: Metadata

def load_pdf(file_path: str = "attention.pdf", index_name: str = "db"):
    load_dotenv()
    pc = Pinecone(api_key=os.getenv("PINECONE_API"))

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    texts = [doc.page_content for doc in docs]

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", 
                                       model_kwargs={'device': 'cpu'}, 
                                       encode_kwargs={'normalize_embeddings': True})
    bm25 = BM25Encoder().default()
    bm25.fit(texts)

    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=768, metric="dotproduct", spec=ServerlessSpec(cloud="aws", region="us-east-1"))

    index = pc.Index(index_name)
    dense = embeddings.embed_documents(texts)
    sparse = [bm25.encode_documents(text) for text in texts]

    vectors = [
        {
            'id': str(uuid4()),
            'values': dense[i],
            'sparse_values': {
                'indices': sparse[i]['indices'],
                'values': sparse[i]['values']
            },
            'metadata': {
                'text': texts[i],
                'source': docs[i].metadata.get("source", "unknown"),
                'page': docs[i].metadata.get("page")
            }
        } for i in range(len(texts))
    ]
    index.upsert(vectors=vectors)
    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25, index=index, text_key="text")
    time.sleep(5)

    def search(query: str) -> List[StructuredOutput]:
        results = retriever.invoke(query)
        return [
            StructuredOutput(
                content=res.page_content,
                metadata=Metadata(source=res.metadata.get("source", "unknown"), page=res.metadata.get("page"))
            ) for res in results
        ]

    def tool_wrapper(query: str) -> str:
        outputs = search(query)
        return json.dumps([o.model_dump() for o in outputs], indent=2)

    return Tool(name="pdf_search", func=tool_wrapper, description="Search PDF content with metadata")