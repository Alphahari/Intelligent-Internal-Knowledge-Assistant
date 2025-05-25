from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.tools.retriever import create_retriever_tool
from langchain.schema import Document
import os
import time
from langchain_core.tools import Tool

def Text_Document(file_path: str, index_name: str = "txt"):
    """Load and index text documents in Pinecone with hybrid search"""
    # Initialize components
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    texts = [doc.page_content for doc in docs]
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Initialize BM25 encoder
    bm25_encoder = BM25Encoder().default()
    bm25_encoder.fit(texts)
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API"))
    
    # Create or connect to index
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,  # matches BAAI/bge-base-en-v1.5
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = pc.Index(index_name)
    
    # Generate and upsert embeddings
    # dense_embeddings = embeddings.embed_documents(texts)
    # sparse_embeddings = [bm25_encoder.encode_documents(text) for text in texts]
    
    # vectors = []
    # for i, text in enumerate(texts):
    #     vectors.append({
    #         'id': str(i),
    #         'values': dense_embeddings[i],
    #         'sparse_values': {
    #             'indices': sparse_embeddings[i]['indices'],
    #             'values': sparse_embeddings[i]['values']
    #         },
    #         'metadata': {
    #             'text': text,
    #             'source': docs[i].metadata["source"]
    #         }
    #     })
    
    # index.upsert(vectors=vectors)
    # Create retriever
    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25_encoder,
        index=index,
        text_key="text"
    )
    time.sleep(10)
    def search_raw(query: str) -> list[Document]:
        try:
            results = retriever.invoke(query)
            return results
        except Exception as e:
            return [Document(page_content=f"Error during search: {e}", metadata={})]    
    # Create tool
    return Tool(
        name="Text Documents Search Tool",
        func=search_raw,
        description="Return raw documents matching the query from Text file with full metadata"
    )
if __name__ == "__main__":
    try:
        tool = Text_Document("speech.txt")
        print(tool.name)
    except Exception as e:
        print("Text document tool failed:", e)