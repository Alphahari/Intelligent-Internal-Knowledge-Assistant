from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.tools import Tool
from dotenv import load_dotenv
from pydantic import BaseModel
import os, json, time

load_dotenv()

class Metadata(BaseModel):
    source: str
    user: str | None = None
    channel: str | None = None

class StructuredOutput(BaseModel):
    content: str
    metadata: Metadata

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
ALLOWED_CHANNEL_ID = os.getenv("ALLOWED_CHANNEL_ID")
client = WebClient(token=SLACK_BOT_TOKEN)

def read_messages(channel_id, limit=20):
    if channel_id != ALLOWED_CHANNEL_ID:
        raise PermissionError("Unauthorized channel.")
    try:
        return client.conversations_history(channel=channel_id, limit=limit)["messages"]
    except SlackApiError as e:
        print("Slack API Error:", e)
        return []

def slack_message_retriever(channel_id=ALLOWED_CHANNEL_ID, limit=20, index_name="db"):
    raw = read_messages(channel_id, limit)
    documents = [
        Document(page_content=msg.get("text", ""), metadata={"user": msg.get("user", "anon"), "channel": channel_id, "source": "Slack"})
        for msg in raw if msg.get("subtype") != "channel_join"
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    texts = [d.page_content for d in docs]

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
            'metadata': docs[i].metadata | {"text": texts[i]}
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
                metadata=Metadata(source=res.metadata.get("source", "Slack"), user=res.metadata.get("user"), channel=res.metadata.get("channel"))
            ) for res in results
        ]

    def tool_wrapper(query: str) -> str:
        return json.dumps([r.model_dump() for r in search(query)], indent=2)

    return Tool(name="Slack Search Tool", func=tool_wrapper, description="Search Slack messages with metadata")