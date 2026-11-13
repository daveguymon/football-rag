"""Shared config + local embedding model + Chroma collection."""
import os
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "football"

# Small, fast, good-quality local embedding model (~130 MB, runs fine on CPU).
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge models work best when *queries* (not documents) get this prefix.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Local LLM served by Ollama (https://ollama.com). Change via env var.
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")
LLM_CONTEXT_TOKENS = int(os.getenv("LLM_CONTEXT_TOKENS", "2048"))
LLM_OUTPUT_TOKENS = int(os.getenv("LLM_OUTPUT_TOKENS", "256"))
if LLM_CONTEXT_TOKENS <= 0 or not 0 < LLM_OUTPUT_TOKENS < LLM_CONTEXT_TOKENS:
    raise ValueError(
        "LLM_CONTEXT_TOKENS must be positive and LLM_OUTPUT_TOKENS must be "
        "positive and smaller than LLM_CONTEXT_TOKENS."
    )
TOP_K = int(os.getenv("TOP_K", "6"))


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def embed_documents(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    return get_embedder().encode(QUERY_PREFIX + text, normalize_embeddings=True).tolist()
