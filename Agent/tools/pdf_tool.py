import os
import time
from uuid import uuid4
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API"))

def upsert_pdf(file_path: str, index_name: str = "db"):
    """Process and upsert PDF document to Pinecone"""
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(30)
    
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)
    texts = [doc.page_content for doc in docs]

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    bm25 = BM25Encoder()
    bm25.fit(texts)

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
    
    index = pc.Index(index_name)
    index.upsert(vectors=vectors)
    print(f"Upserted {len(vectors)} chunks from {file_path}")