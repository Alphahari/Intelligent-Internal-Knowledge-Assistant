import os
import hashlib
from sqlalchemy.orm import Session
from Agent.tools.document_loader import upsert_text
from Agent.tools.pdf_tool import upsert_pdf
from database import SessionLocal
from model import Document, DocumentStatus

def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def scan_directory(directory: str):
    """Scan directory and register new/changed files in DB"""
    db: Session = SessionLocal()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if not os.path.isfile(file_path):
            continue
            
        file_hash = compute_file_hash(file_path)
        existing_doc = db.query(Document).filter(
            Document.filehash == file_hash
        ).first()
        
        if existing_doc:
            if existing_doc.status == DocumentStatus.PROCESSED:
                print(f"Skipping processed file: {filename}")
                continue
        else:
            if file_path:
                print(f"Processing document: {file_path}")
                if file_path.endswith('.pdf'):
                    upsert_pdf(file_path)
                elif file_path.endswith('.txt'):
                    upsert_text(file_path)
                print("Document processing complete. Index updated.")
            new_doc = Document(
                filename=filename,
                filehash=file_hash,
                filepath=file_path,
                status=DocumentStatus.NEW
            )
            db.add(new_doc)
            print(f"Registered new file: {filename}")
    
    db.commit()

