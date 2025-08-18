import os
import hashlib
from sqlalchemy.orm import Session
from database import SessionLocal
from model import Document, DocumentStatus
from tasks import process_document

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
    documents_to_process = []
    
    try:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            print("File path: ", file_path)
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
                existing_doc.status = DocumentStatus.NEW
                documents_to_process.append(existing_doc)
            else:
                new_doc = Document(
                    filename=filename,
                    filehash=file_hash,
                    filepath=file_path,
                    status=DocumentStatus.NEW
                )
                db.add(new_doc)
                documents_to_process.append(new_doc)
                print(f"Registered new file: {filename}")
        
        db.commit()
        for doc in documents_to_process:
            process_document.delay(str(doc.id))
            print(f"Queued processing for: {doc.filename}")
    
    except Exception as e:
        print(f"Error in scan_directory: {e}")
        db.rollback()
    finally:
        db.close()