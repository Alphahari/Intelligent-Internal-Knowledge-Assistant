from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
import os
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools.retriever import create_retriever_tool

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

def slack_message_retriever(channel_id=ALLOWED_CHANNEL_ID, limit=20):
    """Load and index Slack messages for retrieval"""
    raw_messages = read_messages(channel_id, limit)
    documents = []

    for msg in raw_messages:
        if msg.get("subtype") != "channel_join":
            user = msg.get("user", "unknown")
            text = msg.get("text", "")
            documents.append(Document(page_content=text, metadata={"user": user}))
    # print(documents)

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    # print(docs)
    vectordb = FAISS.from_documents(docs, OllamaEmbeddings(model="deepseek-r1"))
    retriever = vectordb.as_retriever()

    slack_tool = create_retriever_tool(
        retriever,
        name="SlackMessageSearch",
        description="Search Slack messages for relevant responses or discussions."
    )
    return slack_tool

if __name__ == "__main__":
    try:
        tool = slack_message_retriever()
        print(tool.name)
    except Exception as e:
        print("Slack tool failed:", e)
