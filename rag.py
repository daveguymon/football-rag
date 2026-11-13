"""Retrieve relevant chunks and stream a grounded answer from a local LLM (Ollama)."""
import json
from typing import Iterator

import ollama

from store import (
    LLM_CONTEXT_TOKENS,
    LLM_MODEL,
    LLM_OUTPUT_TOKENS,
    TOP_K,
    embed_query,
    get_collection,
)

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

TEMPLATE_ALLOWANCE = 128
MESSAGE_ALLOWANCE = 32
DECOMPOSITION_PROMPT = (
    "For a comparison question, return JSON with a queries array containing "
    "two or three short, standalone search questions, one for each option. "
    "Preserve the shared opponent or situation in each question. For example, "
    "'Is 3-4 better than 4-3 against wing-t?' becomes "
    '{"queries":["3-4 defense against wing-t offense",'
    '"4-3 defense against wing-t offense"]}. '
    'If the current question is not a comparison, return {"queries":[]}. '
    "Use the previous question only to resolve follow-up context. "
    "Do not answer the question."
)
DECOMPOSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "queries": {"type": "array", "items": {"type": "string"}, "maxItems": 3}
    },
    "required": ["queries"],
}


class ContextBudgetError(ValueError):
    """The question or source metadata cannot fit in the context budget."""


class DecompositionError(ValueError):
    """The model did not return usable search queries."""


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
        {"id": source_id, "text": d, "meta": m, "distance": dist}
        for source_id, d, m, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def decompose_question(question: str, previous_question: str = "") -> list[str]:
    messages = [
        {"role": "system", "content": DECOMPOSITION_PROMPT},
        {"role": "user", "content": f"Current question: {question}"},
    ]
    output_limit = min(LLM_OUTPUT_TOKENS, 256)
    if prompt_size(messages) > LLM_CONTEXT_TOKENS - output_limit:
        raise ContextBudgetError(
            "The question is too long to decompose within the context budget. "
            "Shorten it or increase LLM_CONTEXT_TOKENS."
        )
    if previous_question:
        content = f"Previous question: {previous_question}\n\nCurrent question: {question}"
        with_previous = [messages[0], {"role": "user", "content": content}]
        if prompt_size(with_previous) <= LLM_CONTEXT_TOKENS - output_limit:
            messages = with_previous

    response = ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        format=DECOMPOSITION_SCHEMA,
        options={
            "num_ctx": LLM_CONTEXT_TOKENS,
            "num_predict": output_limit,
            "temperature": 0,
        },
    )
    try:
        data = json.loads(response["message"]["content"])
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise DecompositionError(
            "The model returned invalid query decomposition. Try again."
        ) from exc
    queries = data.get("queries") if isinstance(data, dict) else None
    if (
        not isinstance(queries, list)
        or len(queries) not in (0, 2, 3)
        or any(not isinstance(query, str) or not query.strip() for query in queries)
    ):
        raise DecompositionError(
            "The model returned invalid query decomposition. Try again."
        )
    queries = [query.strip() for query in queries]
    if len({query.casefold() for query in queries}) != len(queries):
        raise DecompositionError("The model returned duplicate sub-queries. Try again.")
    return queries


def merge_results(result_lists: list[list[dict]], k: int) -> list[dict]:
    """Interleave rankings so each sub-query can contribute distinct sources."""
    merged = []
    seen = set()
    for rank in range(max((len(results) for results in result_lists), default=0)):
        for results in result_lists:
            if rank >= len(results):
                continue
            hit = results[rank]
            if hit["id"] not in seen:
                merged.append(hit)
                seen.add(hit["id"])
                if len(merged) >= k:
                    return merged
    return merged


def format_sources(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h["meta"]
        blocks.append(f"[{i}] {m['title']} ({m['site']}) - {m['url']}\n{h['text']}")
    return "\n\n---\n\n".join(blocks)


def prompt_size(messages: list[dict]) -> int:
    """Conservative byte-based estimate with allowances for chat formatting."""
    return TEMPLATE_ALLOWANCE + sum(
        len(m["role"].encode("utf-8"))
        + len(m["content"].encode("utf-8"))
        + MESSAGE_ALLOWANCE
        for m in messages
    )


def build_messages(
    question: str, history: list[dict], hits: list[dict], balanced: bool = False
) -> tuple[list[dict], list[dict]]:
    input_budget = LLM_CONTEXT_TOKENS - LLM_OUTPUT_TOKENS

    def messages_for(sources: list[dict], recent: list[dict]) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *recent,
            {
                "role": "user",
                "content": f"SOURCES:\n\n{format_sources(sources)}\n\nQUESTION: {question}",
            },
        ]

    if prompt_size(messages_for([], [])) > input_budget:
        raise ContextBudgetError(
            "The question is too long for the context budget. Shorten it or "
            "increase LLM_CONTEXT_TOKENS."
        )

    if balanced and len(hits) > 1:
        # Reserve space for each source before filling it with an excerpt.
        candidates = hits[:]
        while candidates:
            empty_sources = [{**hit, "text": ""} for hit in candidates]
            remaining = input_budget - prompt_size(messages_for(empty_sources, []))
            minimums = [
                len(hit["text"][:1].encode("utf-8"))
                + (3 if len(hit["text"]) > 1 else 0)
                for hit in candidates
            ]
            if remaining >= sum(minimums):
                break
            candidates.pop()
        if not candidates:
            raise ContextBudgetError(
                "The question and source metadata leave no room for a source excerpt. "
                "Shorten the question or increase LLM_CONTEXT_TOKENS."
            )
        selected = []
        for index, hit in enumerate(candidates):
            future_minimum = sum(minimums[index + 1:])
            allowance = max(
                minimums[index],
                min(remaining - future_minimum, remaining // (len(candidates) - index)),
            )
            text = hit["text"]
            if len(text.encode("utf-8")) <= allowance:
                excerpt = text
            else:
                low, high = 0, len(text)
                while low < high:
                    mid = (low + high + 1) // 2
                    if len((text[:mid] + "...").encode("utf-8")) <= allowance:
                        low = mid
                    else:
                        high = mid - 1
                excerpt = text[:low] + "..."
            selected.append({**hit, "text": excerpt})
            remaining -= len(excerpt.encode("utf-8"))
    else:
        selected = []
        for hit in hits:
            if prompt_size(messages_for([*selected, hit], [])) <= input_budget:
                selected.append(hit)
                continue

            # Keep as much as possible of the highest-ranked source that does not fit.
            text = hit["text"]
            low, high = 0, len(text)
            while low < high:
                mid = (low + high + 1) // 2
                excerpt = {**hit, "text": text[:mid] + "..."}
                if prompt_size(messages_for([*selected, excerpt], [])) <= input_budget:
                    low = mid
                else:
                    high = mid - 1
            if low:
                selected.append({**hit, "text": text[:low] + "..."})
            break

    if hits and not selected:
        raise ContextBudgetError(
            "The question and source metadata leave no room for a source excerpt. "
            "Shorten the question or increase LLM_CONTEXT_TOKENS."
        )

    recent = []
    for message in reversed(history[-6:]):
        if prompt_size(messages_for(selected, [message, *recent])) > input_budget:
            break
        recent.insert(0, message)
    return selected, messages_for(selected, recent)


def answer_stream(
    question: str, history: list[dict], k: int = TOP_K
) -> tuple[list[dict], Iterator[str]]:
    """Returns (retrieved hits, generator of answer text chunks)."""
    build_messages(question, [], [])
    # Follow-ups like "what about against the run?" retrieve poorly alone,
    # so fold in the previous user question.
    prev_user = [m["content"] for m in history if m["role"] == "user"]
    previous_question = prev_user[-1] if prev_user else ""
    search_query = f"{previous_question} {question}" if previous_question else question
    hits = retrieve(search_query, k)

    if not hits:
        def empty() -> Iterator[str]:
            yield "No sources have been ingested yet. Add some URLs first."
        return hits, empty()

    queries = decompose_question(question, previous_question)
    if queries:
        hits = merge_results([*(retrieve(query, k) for query in queries), hits], k)

    hits, messages = build_messages(question, history, hits, balanced=bool(queries))

    def gen() -> Iterator[str]:
        for part in ollama.chat(
            model=LLM_MODEL,
            messages=messages,
            stream=True,
            options={"num_ctx": LLM_CONTEXT_TOKENS, "num_predict": LLM_OUTPUT_TOKENS},
        ):
            yield part["message"]["content"]

    return hits, gen()
