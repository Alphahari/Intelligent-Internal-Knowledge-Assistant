from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
import os
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
import time
from langchain_core.tools import Tool

load_dotenv()

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
ALLOWED_CHANNEL_ID = os.getenv("ALLOWED_CHANNEL_ID")

client = WebClient(token=SLACK_BOT_TOKEN)

def enforce_single_channel(channel_id):
    if channel_id != ALLOWED_CHANNEL_ID:
        raise PermissionError(f"Access denied: {channel_id} is not the allowed channel.")

def read_messages(channel_id, limit=20):
    enforce_single_channel(channel_id)
    try:
        response = client.conversations_history(channel=channel_id, limit=limit)
        return response["messages"]
    except SlackApiError as e:
        print("Slack API error:", e.response["error"])
        return []

def slack_message_retriever(channel_id=ALLOWED_CHANNEL_ID, limit=20, index_name="slack"):
    """Load and index Slack messages in Pinecone with hybrid search"""
    raw_messages = read_messages(channel_id, limit)
    documents = []
    
    for msg in raw_messages:
        if msg.get("subtype") != "channel_join":
            user = msg.get("user", "unknown")
            text = msg.get("text", "")
            documents.append(Document(page_content=text, metadata={"user": user, "channel": channel_id}))
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    docs = splitter.split_documents(documents)
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
            dimension=768,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = pc.Index(index_name)
    
    # Generate and upsert embeddings
    # dense_embeddings = embeddings.embed_documents(texts)
    # sparse_embeddings = [bm25_encoder.encode_documents(text) for text in texts]
    
    # vectors = []
    # for i, doc in enumerate(docs):
    #     vectors.append({
    #         'id': str(i),
    #         'values': dense_embeddings[i],
    #         'sparse_values': {
    #             'indices': sparse_embeddings[i]['indices'],
    #             'values': sparse_embeddings[i]['values']
    #         },
    #         'metadata': {
    #             'text': texts[i],
    #             'user': doc.metadata["user"],
    #             'channel': doc.metadata["channel"],
    #             'source': "Slack Channel"
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
        name="Slack Search Tool",
        func=search_raw,
        description="Return raw documents matching the query from Slack Channel with full metadata"
    )

if __name__ == "__main__":
    try:
        tool = slack_message_retriever()
        print(tool.name)
    except Exception as e:
        print("Slack tool failed:", e)