"""Retrieve relevant chunks and stream a grounded answer from a local LLM (Ollama)."""
from typing import Iterator

import ollama

from store import LLM_MODEL, TOP_K, embed_query, get_collection

SYSTEM_PROMPT = """You are an American football strategy assistant. Answer the \
user's question using the numbered SOURCES provided.

Rules:
- Ground your answer in the sources and cite them inline like [1] or [2][3].
- If the sources only partly cover the question, answer the covered part, then \
clearly say what the sources don't address. You may add general football \
knowledge, but label it "General knowledge (not from sources):".
- If the sources don't cover the question at all, say so plainly. Do not invent \
what a coverage or formation "always" does.
- Football terms have synonyms (e.g. Cover 2 / Tampa 2 / two-high shell). Treat \
related terms sensibly, and note when a source is about a variant.
- Be concise and organized. Use short paragraphs or bullets."""


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    col = get_collection()
    if col.count() == 0:
        return []
    res = col.query(
        query_embeddings=[embed_query(query)],
        n_results=min(k, col.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": d, "meta": m, "distance": dist}
        for d, m, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def format_sources(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        blocks.append(f"[{i}] {m['title']} ({m['site']}) - {m['url']}\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def answer_stream(
    question: str, history: list[dict], k: int = TOP_K
) -> tuple[list[dict], Iterator[str]]:
    """Returns (retrieved hits, generator of answer text chunks)."""
    # Follow-ups like "what about against the run?" retrieve poorly alone,
    # so fold in the previous user question.
    prev_user = [m["content"] for m in history if m["role"] == "user"]
    search_query = f"{prev_user[-1]} {question}" if prev_user else question
    hits = retrieve(search_query, k)

    if not hits:
        def empty() -> Iterator[str]:
            yield "No sources have been ingested yet. Add some URLs first."
        return hits, empty()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history[-6:]  # recent turns only; small local models have limited context
    messages.append(
        {
            "role": "user",
            "content": f"SOURCES:\n\n{format_sources(hits)}\n\nQUESTION: {question}",
        }
    )

    def gen() -> Iterator[str]:
        for part in ollama.chat(model=LLM_MODEL, messages=messages, stream=True):
            yield part["message"]["content"]

    return hits, gen()
