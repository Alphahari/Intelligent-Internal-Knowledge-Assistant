from celery import Celery
from sqlalchemy.orm import Session
from database import SessionLocal
from model import Document, DocumentStatus
from .tools.pdf_tool import upsert_pdf
from .tools.document_loader import upsert_text
import os

app = Celery('document_processing', 
             broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"))

@app.task
def process_document(document_id: str):
    """Celery task to process and upsert a document"""
    db: Session = SessionLocal()
    try:
        doc = db.query(Document).get(document_id)
        if not doc or doc.status != DocumentStatus.NEW:
            return

        print(f"Processing document: {doc.filename}")
        
        if doc.filename.endswith('.pdf'):
            upsert_pdf(doc.filepath, index_name="db")
        elif doc.filename.endswith('.txt'):
            upsert_text(doc.filepath, index_name="db")

        doc.status = DocumentStatus.PROCESSED
        db.commit()
        print(f"Successfully processed: {doc.filename}")
        
    except Exception as e:
        if doc:
            doc.status = DocumentStatus.ERROR
            doc.error_message = str(e)
            db.commit()
            print(f"Error processing {doc.filename}: {str(e)}")
    finally:
        db.close()