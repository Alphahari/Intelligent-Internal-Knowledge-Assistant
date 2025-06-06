from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.tools import Tool
from pydantic import BaseModel
import os, time, json

class Metadata(BaseModel):
    source: str

class StructuredOutput(BaseModel):
    content: str
    metadata: Metadata

def Text_Document(file_path: str, index_name: str = "db"):
    loader = TextLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    texts = [doc.page_content for doc in docs]

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    bm25 = BM25Encoder().default()
    bm25.fit(texts)

    pc = Pinecone(api_key=os.getenv("PINECONE_API"))
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name, dimension=768, metric="dotproduct", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
    index = pc.Index(index_name)

    dense = embeddings.embed_documents(texts)
    sparse = [bm25.encode_documents(text) for text in texts]
    vectors = [
        {
            'id': str(i),
            'values': dense[i],
            'sparse_values': {
                'indices': sparse[i]['indices'],
                'values': sparse[i]['values']
            },
            'metadata': {
                'text': texts[i],
                'source': docs[i].metadata.get("source", "textfile")
            }
        } for i in range(len(texts))
    ]
    index.upsert(vectors=vectors)
    retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25, index=index, text_key="text")
    time.sleep(5)

    def search(query: str) -> list[StructuredOutput]:
        results = retriever.invoke(query)
        return [
            StructuredOutput(
                content=res.page_content,
                metadata=Metadata(source=res.metadata.get("source", "textfile"))
            ) for res in results
        ]

    def tool_wrapper(query: str) -> str:
        return json.dumps([r.model_dump() for r in search(query)], indent=2)

    return Tool(name="Text Documents Search Tool", func=tool_wrapper, description="Search local text documents with metadata")