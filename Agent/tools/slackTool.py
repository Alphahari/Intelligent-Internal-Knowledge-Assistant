# tools/slack.py
import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from langchain_core.documents import Document
from pydantic import BaseModel
import time
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import ServerlessSpec
from pinecone_text.sparse import BM25Encoder


class SlackConfig(BaseModel):
    channel_id: str
    limit: int = 20

def slack_message_retriever(index_name, config: SlackConfig) -> Tool:
    """Create a search tool for Slack messages"""
    
    SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
    ALLOWED_CHANNEL_ID = os.getenv("ALLOWED_CHANNEL_ID")
    
    if config.channel_id != ALLOWED_CHANNEL_ID:
        raise PermissionError("Unauthorized channel access")
    
    client = WebClient(token=SLACK_BOT_TOKEN)
    
    try:
        response = client.conversations_history(
            channel=config.channel_id, 
            limit=config.limit
        )
        messages = response["messages"]
        
        documents = [
            Document(
                page_content=msg.get("text", ""),
                metadata={
                    "user": msg.get("user", "anon"),
                    "channel": config.channel_id
                }
            )
            for msg in messages if msg.get("subtype") != "channel_join"
        ]
        
        def metadata_extractor(doc):
            return {
                "source": "Slack",
                "user": doc.metadata.get("user"),
                "channel": doc.metadata.get("channel")
            }
        
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="dotproduct",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(30)
        
        loader = TextLoader(file_path)
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
        
        index = pc.Index(index_name)
        index.upsert(vectors=vectors)
        print(f"Upserted {len(vectors)} chunks from {file_path}")
    except SlackApiError as e:
        print("SlackApiError ", e)
    except Exception as e:
        print("Slack Exception Error ", e)