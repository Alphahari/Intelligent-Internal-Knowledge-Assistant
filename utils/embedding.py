from langchain_ollama import OllamaEmbeddings
from config.settings import settings

def get_embeddings():
    return OllamaEmbeddings(model=settings.OLLAMA_MODEL)