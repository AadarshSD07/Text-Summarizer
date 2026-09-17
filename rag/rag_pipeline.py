import logging
from sqlalchemy.ext.asyncio import AsyncSession
from openai import AsyncOpenAI
from rag.retriever import retrieve_relevant_chunks
from rag.config import LLM_MODEL, TOP_K
import os

logger = logging.getLogger(__name__)

# Distance threshold: chunks above this score are not relevant enough
RELEVANCE_THRESHOLD = 0.5   # cosine distance; lower = more similar


def build_rag_prompt(query: str, chunks: list[dict]) -> tuple[str, str]:
    """Build system and user prompts with retrieved context injected."""
    system = (
        "You are a precise document analyst. "
        "Answer questions using ONLY the provided document excerpts. "
        "If the excerpts do not contain enough information to answer, "
        "say so explicitly. Never invent facts. "
        "After your answer, list the source filenames you used."
    )

    context_block = "\n\n".join(
        f"[Source: {c['filename']}, chunk {c['chunk_index']}]\n{c['text']}"
        for c in chunks
    )

    user = (
        f"DOCUMENT EXCERPTS:\n"
        f"{'='*50}\n"
        f"{context_block}\n"
        f"{'='*50}\n\n"
        f"QUESTION: {query}"
    )
    return system, user


async def answer_query(
    db: AsyncSession,
    query: str,
    top_k: int = TOP_K,
) -> dict:
    """
    Full RAG pipeline: retrieve → build prompt → generate → return.

    Returns dict with: answer, sources, chunks_used, no_relevant_found flag.
    """
    # Step 1: Retrieve
    chunks = await retrieve_relevant_chunks(db, query, top_k=top_k)

    # Step 2: Filter by relevance threshold
    # Handles "no relevant chunks" case — don't hallucinate
    relevant = [c for c in chunks if c["distance"] <= RELEVANCE_THRESHOLD]

    if not relevant:
        logger.warning(f"No relevant chunks found for query: '{query[:60]}'")
        return {
            "answer": (
                "I could not find relevant information in the documents "
                "to answer this question. The query may be outside the "
                "scope of the ingested documents."
            ),
            "sources": [],
            "chunks_used": 0,
            "no_relevant_found": True,
        }

    # Step 3: Build grounded prompt
    system, user = build_rag_prompt(query, relevant)

    # Step 4: Call LLM
    client = AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1"),
    )
    response = await client.chat.completions.create(
        model=os.getenv("LLM_MODEL", LLM_MODEL),
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.2,   # low — factual retrieval task
        max_tokens=600,
    )

    answer  = response.choices[0].message.content.strip()
    sources = list({c["filename"] for c in relevant})

    logger.info(f"Query answered using {len(relevant)} chunks from {sources}")

    return {
        "answer":            answer,
        "sources":           sources,
        "chunks_used":       len(relevant),
        "no_relevant_found": False,
        "top_chunk_distance": relevant[0]["distance"],
    }