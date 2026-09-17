import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from rag.models import Document, Chunk
from rag.embedder import embed_batch
from rag.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping word-window chunks.

    Why overlap? The answer to a question might straddle a chunk
    boundary. Overlap ensures that boundary content appears in at
    least one full chunk on each side.

    chunk_size=400 words ≈ 500 tokens ≈ leaves room in context window.
    overlap=50 words = ~12% overlap — enough for boundary coverage,
    not so much that you store huge duplicate content.
    """
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap   # slide forward with overlap
    return chunks


async def ingest_document(
    db: AsyncSession,
    filename: str,
    content: str,
    title: str = "",
    sentiment: str = "",
) -> Document:
    """
    Chunk a document, embed each chunk, store in DB.
    Idempotent: re-ingesting same filename replaces old chunks.
    """
    # Delete existing record if re-ingesting
    existing = await db.execute(
        select(Document).where(Document.filename == filename)
    )
    existing_doc = existing.scalar_one_or_none()
    if existing_doc:
        await db.delete(existing_doc)
        await db.flush()
        logger.info(f"Re-ingesting {filename} — old chunks deleted")

    # Create document record
    doc = Document(filename=filename, title=title, sentiment=sentiment)
    db.add(doc)
    await db.flush()   # get doc.id without committing

    # Chunk the text
    chunks = chunk_text(content)
    logger.info(f"{filename}: {len(chunks)} chunks created")

    # Embed all chunks
    embeddings = await embed_batch(chunks)

    # Store chunks with embeddings
    for i, (text, vec) in enumerate(zip(chunks, embeddings)):
        chunk = Chunk(
            document_id=doc.id,
            chunk_index=i,
            text=text,
            embedding=vec,
        )
        db.add(chunk)

    await db.commit()
    logger.info(f"Ingested {filename}: {len(chunks)} chunks stored")
    return doc