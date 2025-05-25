from tools.document_loader import Text_Document
from tools.pdf_tool import load_pdf
from tools.slack import slack_message_retriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from pinecone import Pinecone, ServerlessSpec
import os

load_dotenv()

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def make_tools_calling_llm(llm_with_tools):
    def tools_calling_llm(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"][-1].content)]}
    return tools_calling_llm

def main():
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API")
    os.environ["PINECONE_API"] = os.getenv("PINECONE_API")
    os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
    
    # Initialize Pinecone index (ensure it exists)
    pc = Pinecone(api_key=os.getenv("PINECONE_API"))
    if "nlp-project" not in pc.list_indexes().names():
        pc.create_index(
            name="nlp-project",
            dimension=768,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    llm = ChatGroq(model="qwen-qwq-32b")

    # Initialize all tools with the unified Pinecone index
    pdf_tool = load_pdf("attention.pdf")
    text_tool = Text_Document("speech.txt")
    slack_tool = slack_message_retriever(limit=10)

    tools = [pdf_tool, text_tool, slack_tool]
    llm_with_tools = llm.bind_tools(tools=tools)

    # Build the retrieval graph
    builder = StateGraph(State)
    builder.add_node("tools_calling_llm", make_tools_calling_llm(llm_with_tools))
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "tools_calling_llm")
    builder.add_conditional_edges("tools_calling_llm", tools_condition)
    builder.add_edge("tools", END)

    graph = builder.compile()

    # graph.get_graph().draw_mermaid_png()

    # Example query
    response = graph.invoke({
        "messages": "Authors of Attention is all you need"
    })

    print(response)

if __name__ == "__main__":
    main()