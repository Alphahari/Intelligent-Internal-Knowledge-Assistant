from tools.document_loader import Text_Document
from tools.pdf_tool import load_pdf
from tools.slack import slack_message_retriever
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from IPython.display import display, Image
from typing_extensions import TypedDict
from typing import Annotated
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

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
    llm = ChatGroq(model="qwen-qwq-32b")

    pdf_tool = load_pdf("attention.pdf")
    text_tool = Text_Document("speech.txt")
    slack_tool = slack_message_retriever(limit=10)

    tools = [pdf_tool, text_tool, slack_tool]
    llm_with_tools = llm.bind_tools(tools=tools)

    builder = StateGraph(State)
    builder.add_node("tools_calling_llm", make_tools_calling_llm(llm_with_tools))
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "tools_calling_llm")
    builder.add_conditional_edges("tools_calling_llm", tools_condition)
    builder.add_edge("tools", END)

    graph = builder.compile()

    display(Image(graph.get_graph().draw_mermaid_png()))

    response = graph.invoke({
        "messages": [HumanMessage(content="Can you find anything relate to AI in slack channel?")]
    })

    for m in response["messages"]:
        print(m.content)

if __name__ == "__main__":
    main()
