import os
from uuid import uuid4
from dotenv import load_dotenv
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever

def load_pdf(file_path: str = "attention.pdf", index_name: str = "nlp-project"):
    # Load environment variables
    load_dotenv()
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    PINECONE_API = os.getenv("PINECONE_API")

    try:
        # Load and split PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)
        texts = [doc.page_content for doc in docs]

        # Initialize embeddings
        huggingface_embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize and fit BM25 encoder
        bm25_encoder = BM25Encoder().default()
        bm25_encoder.fit(texts)
        bm25_encoder.dump("bm25_values.json")
        bm25_encoder = BM25Encoder().load("bm25_values.json")

        # Initialize Pinecone
        pc = Pinecone(api_key=PINECONE_API)

        index_name = "nlp-project"
        # Create or connect to index
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )

        index = pc.Index(index_name)

        # Generate embeddings
        dense_embeddings = huggingface_embeddings.embed_documents(texts)
        sparse_embeddings = [bm25_encoder.encode_documents(text) for text in texts]

        # Generate dense and sparse embeddings
        dense_embeddings = huggingface_embeddings.embed_documents(texts)
        sparse_embeddings = [bm25_encoder.encode_documents(text) for text in texts]

        # Prepare vectors for upsert
        vectors = []
        for i in range(len(texts)):
            vectors.append({
                'id': str(uuid4()),
                'values': dense_embeddings[i],
                'sparse_values': {
                    'indices': sparse_embeddings[i]['indices'],
                    'values': sparse_embeddings[i]['values']
                },
                'metadata': {
                    'source': docs[i].metadata["source"],
                    'page': docs[i].metadata["page"],
                    'page_label': docs[i].metadata["page_label"],
                    'creation_date': docs[i].metadata["creationdate"],
                    'text': texts[i]  # Optional - keep original if needed
                }
            })

        # Upsert vectors into Pinecone
        index.upsert(vectors=vectors)
        print(f"Upserted {len(vectors)} vectors")

        # Create retriever
        retriever = PineconeHybridSearchRetriever(
            embeddings=huggingface_embeddings,
            sparse_encoder=bm25_encoder,
            index=index,
            text_key="text" 
        )
        return create_retriever_tool(
            retriever,
            "Search Tool",
            "Search across all documents (including PDFs) for relevant information"
        )


    except Exception as e:
        raise RuntimeError(f"Failed to initialize PDF retriever: {str(e)}")

if __name__ == "__main__":
    try:
        tool_call = load_pdf("attention.pdf")
        print(tool_call.name)
    except Exception as e:
        print(f"Error: {str(e)}")