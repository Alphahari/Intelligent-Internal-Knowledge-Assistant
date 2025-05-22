from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from ..utils.embedding import get_embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.tools.retriever import create_retriever_tool


def Text_Document(file_path):
    """Load and split PDF documents"""
    loader = TextLoader(file_path)
    documents = loader.load()
    docs = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100).split_documents(documents)
    vectordb = FAISS.from_documents(docs,OllamaEmbeddings(model="deepseek-r1"))
    retriver = vectordb.as_retriever()
    textdocument_retriver_tool = create_retriever_tool(retriver,"textDocs_Search",
                      "Search text document to get any relavent information to the question asked")
    # print(textdocument_retriver_tool.name)
    return textdocument_retriver_tool

if __name__ == "__main__":
    try:
        tool =Text_Document("speech.txt")
        tool.name
    except Exception as e:
        print("pdf tool failed",e)