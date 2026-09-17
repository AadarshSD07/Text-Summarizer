import os

EMBED_MODEL   = os.getenv("EMBED_MODEL",   "nomic-embed-text")
OLLAMA_URL    = os.getenv("OLLAMA_URL",    "http://localhost:11434")
LLM_MODEL     = os.getenv("LLM_MODEL",    "mistral")
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "400"))   # words per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))    # word overlap
TOP_K         = int(os.getenv("TOP_K",         "4"))     # chunks to retrieve