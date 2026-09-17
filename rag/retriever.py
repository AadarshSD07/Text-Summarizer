import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from rag.models import Chunk, Document
from rag.embedder import embed_text
from rag.config import TOP_K

logger = logging.getLogger(__name__)


async def retrieve_relevant_chunks(
    db: AsyncSession,
    query: str,
    top_k: int = TOP_K,
) -> list[dict]:
    """
    Find the top-k most semantically similar chunks to the query.

    Steps:
      1. Embed the query into the same vector space as the chunks
      2. Use pgvector cosine distance operator (<=>) to rank chunks
      3. Return top_k chunks with their source document info

    Cosine distance: 0 = identical direction, 2 = opposite direction.
    We ORDER BY distance ASC — smallest distance = most similar.
    """
    query_embedding = await embed_text(query)

    # pgvector cosine distance: embedding <=> '[...]'::vector
    # Cast the Python list to a pgvector literal
    vec_literal = "[" + ",".join(str(v) for v in query_embedding) + "]"

    sql = text("""
        SELECT
            c.id,
            c.text,
            c.chunk_index,
            d.filename,
            d.title,
            d.sentiment,
            c.embedding <=> :vec ::vector AS distance
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        ORDER BY distance ASC
        LIMIT :top_k
    """)

    result = await db.execute(sql, {"vec": vec_literal, "top_k": top_k})
    rows = result.fetchall()

    chunks = []
    for row in rows:
        chunks.append({
            "chunk_id":    row.id,
            "text":        row.text,
            "chunk_index": row.chunk_index,
            "filename":    row.filename,
            "title":       row.title,
            "sentiment":   row.sentiment,
            "distance":    round(float(row.distance), 4),
        })

    logger.debug(f"Query '{query[:50]}' → {len(chunks)} chunks, "
                 f"best distance: {chunks[0]['distance'] if chunks else 'n/a'}")
    return chunks