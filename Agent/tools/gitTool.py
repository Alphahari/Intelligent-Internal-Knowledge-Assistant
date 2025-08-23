import os
import time
from langchain_community.document_loaders import GithubFileLoader, GitHubIssuesLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

pc = Pinecone(api_key=os.getenv("PINECONE_API"))

def upsert_text(github_repo: str, index_name: str = "db"):
    """Process and upsert text document to Pinecone"""
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        time.sleep(30)
    issueLoader = GitHubIssuesLoader(
            repo=github_repo,
            # assignee="",
            # labels="",
            # creator="",
            # include_prs=False,
        )
    fileLoader = GithubFileLoader(
                repo=github_repo,
                branch="master",
                github_api_url="https://api.github.com",
                file_filter=lambda file_path: file_path.endswith(
                    ".md"
                ),
            )
    documents = issueLoader.load()
    documents = fileLoader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
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
                'source': docs[i].metadata
            }
        } for i in range(len(texts))
    ]
    
    index = pc.Index(index_name)
    index.upsert(vectors=vectors)
    print(f"Upserted {len(vectors)} chunks from {github_repo}")