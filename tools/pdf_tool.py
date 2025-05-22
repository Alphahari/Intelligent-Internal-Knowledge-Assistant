from typing import List
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from ..utils.embedding import get_embeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.tools.retriever import create_retriever_tool


def load_pdf(file_path):
    """Load and split PDF documents"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    docs = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100).split_documents(documents)
    # print(docs)
    vectordb = FAISS.from_documents(docs[:15],OllamaEmbeddings(model="deepseek-r1"))
    retriver = vectordb.as_retriever()
    pdf_retriver_tool = create_retriever_tool(retriver,"PDF_Search",
                      "Search pdf to get any relavent information to the question asked")
    # print(pdf_retriver_tool.name)
    return pdf_retriver_tool

if __name__ == "__main__":
    try:
        r1=load_pdf("attention.pdf")
        print(r1.name)
    except Exception as e:
        print("pdf tool failed",e)