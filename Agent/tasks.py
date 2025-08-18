from celery import Celery
from sqlalchemy.orm import Session
from database import SessionLocal
from model import Document, DocumentStatus
from tools.pdf_tool import upsert_pdf
from tools.document_loader import upsert_text
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Celery('document_processing', 
             broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6380/0"))

@app.task
def process_document(document_id: str):
    db: Session = SessionLocal()
    try:
        doc = db.query(Document).get(document_id)
        if not doc or doc.status != DocumentStatus.NEW:
            return

        print(f"Processing document: {doc.filename}")
    
        file_path = os.path.abspath(doc.filepath)
        
        if doc.filename.endswith('.pdf'):
            upsert_pdf(file_path, index_name="db")
        elif doc.filename.endswith('.txt'):
            upsert_text(file_path, index_name="db")

        doc.status = DocumentStatus.PROCESSED
        db.commit()
        print(f"Successfully processed: {doc.filename}")
        
    except Exception as e:
        print("Error from task queuing...",e)
    finally:
        db.close()