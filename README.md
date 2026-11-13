# Football Strategy RAG (local-only)

Everything runs on your machine: local embeddings (sentence-transformers),
local vector store (Chroma), and a local LLM (Ollama). No API keys.

## Setup

1. Install Ollama from https://ollama.com, then pull a model:
   ```
   ollama pull llama3.1:8b
   ```
   (Smaller machine? Try `llama3.2:3b`. More RAM/GPU? Try `qwen2.5:14b`.)
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

Switch models with an env var: `LLM_MODEL=llama3.2:3b streamlit run app.py`

## How it works

- `ingest.py`: checks robots.txt, downloads each page, extracts main content
  with trafilatura, splits by heading then paragraph (~450 tokens), prepends
  "Title > Heading" to each chunk, embeds, and stores in Chroma. Unchanged
  pages are skipped on re-ingest (content hash); changed pages are replaced.
- `rag.py`: embeds the question, retrieves top-k chunks, and asks the LLM to
  answer with `[n]` citations and to say when sources don't cover something.
- `app.py`: chat UI with expandable sources under each answer.
- `store.py`: config and shared embedder/collection.

## Known limits (good next steps)

- JavaScript-heavy sites may return no content (add a Playwright fetcher).
- Single retrieval per question: comparison questions like "4-3 vs 3-4 against
  pistol" benefit from sub-query decomposition.
- No hybrid (BM25) search or reranking yet.
- Diagrams in articles are not captured.
- Small local models follow citation rules less reliably than hosted ones.
