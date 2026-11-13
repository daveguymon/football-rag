# Football Strategy RAG (local-only)

Everything runs on your machine: local embeddings (sentence-transformers),
local vector store (Chroma), and a local LLM (Ollama). No API keys.

## Setup

1. Install Ollama from https://ollama.com, then pull the default model:
   ```
   ollama pull llama3.2:3b
   ```
   This smaller model is a better starting point for an 8 GB Mac, but can
   still cause memory pressure when other apps are running.
2. Install Python deps (Python 3.10+; a virtualenv is recommended):
   ```
   pip install -r requirements.txt
   ```
   The first run downloads the embedding model (~130 MB).

## Use

Ingest sources, either from the command line:
```
python ingest.py https://example.com/some-article
python ingest.py --urls-file urls.txt
```
or by pasting URLs into the app sidebar. Then start the chat:
```
streamlit run app.py
```

To switch models, pull the model first, then set the environment variable:
```
ollama pull llama3.1:8b
LLM_MODEL=llama3.1:8b streamlit run app.py
```
Larger models such as `llama3.1:8b` may require more memory than an 8 GB Mac
can comfortably provide.

The app defaults to a 2,048-token Ollama context window and at most 256 output
tokens. Override them with `LLM_CONTEXT_TOKENS` and `LLM_OUTPUT_TOKENS`, for
example:
```
LLM_CONTEXT_TOKENS=4096 LLM_OUTPUT_TOKENS=512 streamlit run app.py
```
Both must be positive, and the output limit must be smaller than the context
window. Larger limits can increase memory use. The app reserves the output
budget and estimates input size conservatively using UTF-8 bytes plus chat
formatting allowances; this is not exact model-specific tokenization. It keeps
the question and highest-ranked sources first, shortening excerpts if needed,
then includes as much recent history as fits. A question or source
metadata that leaves no room for an excerpt produces an error. The displayed
sources reflect the excerpts actually sent.

## How it works

- `ingest.py`: checks robots.txt, downloads each page, extracts main content
  with trafilatura, splits by heading then paragraph (~450 tokens), prepends
  "Title > Heading" to each chunk, embeds, and stores in Chroma. Unchanged
  pages are skipped on re-ingest (content hash); changed pages are replaced.
- `rag.py`: searches the original question and uses the local LLM to identify
  comparisons and generate up to three focused sub-queries. It interleaves and
  deduplicates results, shares the input budget between comparison excerpts, and
  asks the LLM for an answer with `[n]` citations.
- `app.py`: chat UI with expandable sources under each answer.
- `store.py`: config and shared embedder/collection.

## Known limits (good next steps)

- JavaScript-heavy sites may return no content (add a Playwright fetcher).
- Decomposition adds a local LLM call and may miss or misinterpret comparisons;
  retrieved evidence may still be insufficient to answer both sides.
- No hybrid (BM25) search or reranking yet.
- Diagrams in articles are not captured.
- Small local models follow citation rules less reliably than hosted ones.
