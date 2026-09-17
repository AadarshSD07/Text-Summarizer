import httpx
import logging
from rag.config import EMBED_MODEL, OLLAMA_URL

logger = logging.getLogger(__name__)

async def embed_text(text: str) -> list[float]:
    """
    Get embedding vector for a text string.
    Calls Ollama's /api/embeddings endpoint.
    Returns a list of floats (length = model dimension).
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
        )
        response.raise_for_status()
        data = response.json()

    embedding = data["embedding"]
    logger.debug(f"Embedded {len(text)} chars → {len(embedding)}-dim vector")
    return embedding


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts. Sequential — Ollama has no batch endpoint."""
    results = []
    for i, text in enumerate(texts):
        vec = await embed_text(text)
        results.append(vec)
        logger.debug(f"Embedded chunk {i+1}/{len(texts)}")
    return results