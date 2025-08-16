import uuid
from sqlalchemy import (
    Column, String, Enum, Text, TIMESTAMP, func
)
from sqlalchemy.dialects.postgresql import UUID
from database import Base
import enum

class DocumentStatus(str, enum.Enum):
    NEW = "NEW"
    PROCESSED = "PROCESSED"
    ERROR = "ERROR"

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, unique=True, nullable=False)
    filename = Column(Text, nullable=False)
    filehash = Column(Text, nullable=False, unique=True)
    filepath = Column(Text, nullable=False)
    status = Column(Enum(DocumentStatus, name="status_enum"), default=DocumentStatus.NEW, nullable=False)
    last_updated = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())
    error_message = Column(Text, nullable=True)
