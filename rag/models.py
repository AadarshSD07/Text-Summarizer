from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime, timezone
from rag.database import Base

class Document(Base):
    __tablename__ = "documents"

    id          = Column(Integer, primary_key=True)
    filename    = Column(String(255), nullable=False, unique=True)
    title       = Column(Text)           # from Block A summary
    sentiment   = Column(String(20))     # from Block A summary
    ingested_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    chunks      = relationship("Chunk", back_populates="document",
                               cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = "chunks"

    id          = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)   # position in document
    text        = Column(Text, nullable=False)
    embedding   = Column(Vector(768))   # dimension matches your model
    document    = relationship("Document", back_populates="chunks")