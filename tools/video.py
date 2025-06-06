import moviepy as mp
import speech_recognition as sr
import os, time, json
from uuid import uuid4
from typing import List
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_core.tools import Tool

class Metadata(BaseModel):
    source: str

class StructuredOutput(BaseModel):
    content: str
    metadata: Metadata

def transcribe_video(video_path: str, index_name: str = "db", duration: int = 200):
    # Extract audio and transcribe
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio_file_name = f'{os.path.splitext(video_path)[0]}.wav'
    audio.write_audiofile(audio_file_name, logger=None)
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_name) as source:
        audio_data = recognizer.record(source, duration=duration)
    try:
        text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = "[Unintelligible speech]"
    except sr.RequestError as e:
        text = f"[Request error: {e}]"
    
    # Process text for Pinecone
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    texts = splitter.split_text(text)
    
    # Initialize embeddings and encoder
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    bm25 = BM25Encoder().default()
    bm25.fit(texts)
    
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API"))
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name, 
            dimension=768, 
            metric="dotproduct", 
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    index = pc.Index(index_name)
    
    # Create and upsert vectors
    dense = embeddings.embed_documents(texts)
    sparse = [bm25.encode_documents(t) for t in texts]
    
    vectors = [
        {
            'id': str(uuid4()),
            'values': dense[i],
            'sparse_values': {
                'indices': sparse[i]['indices'],
                'values': sparse[i]['values']
            },
            'metadata': {
                'text': texts[i],
                'source': os.path.basename(video_path)
            }
        } for i in range(len(texts))
    ]
    index.upsert(vectors=vectors)
    
    retriever = PineconeHybridSearchRetriever(
        embeddings=embeddings,
        sparse_encoder=bm25,
        index=index,
        text_key="text"
    )
    time.sleep(5)
    
    def search(query: str) -> List[StructuredOutput]:
        results = retriever.invoke(query)
        return [
            StructuredOutput(
                content=res.page_content,
                metadata=Metadata(source=res.metadata.get("source", "video"))
            ) for res in results
        ]
    
    def tool_wrapper(query: str) -> str:
        outputs = search(query)
        return json.dumps([o.model_dump() for o in outputs], indent=2)
    
    return Tool(
        name="video_search",
        func=tool_wrapper,
        description="Search transcribed video content with metadata"
    )

if __name__ == "__main__":
    video_path = 'footage.mp4'
    tool = transcribe_video(video_path)
    print("Video processing complete. Tool ready for searching.")