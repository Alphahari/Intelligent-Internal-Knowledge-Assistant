from database import SessionLocal
from model import Document, DocumentStatus
from tasks import process_document

def process_new_documents():
    """Find and queue NEW documents for processing"""
    db = SessionLocal()
    try:
        new_docs = db.query(Document).filter(
            Document.status == DocumentStatus.NEW
        ).all()
        
        for doc in new_docs:
            print(f"Queueing document for processing: {doc.filename}")
            process_document.delay(str(doc.id))
            
    finally:
        db.close()