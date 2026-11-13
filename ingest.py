"""Fetch URLs, extract article text, chunk by heading/paragraph, embed, store.

Usage:
    python ingest.py https://example.com/cover-2-explained
    python ingest.py --urls-file urls.txt
"""
import argparse
import hashlib
import re
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import trafilatura

from store import embed_documents, get_collection

USER_AGENT = "football-rag-bot/0.1 (personal research)"
MAX_CHARS = 1800   # ~450 tokens
MIN_CHARS = 100    # drop tiny fragments
CRAWL_DELAY = 2.0  # seconds between requests


def allowed_by_robots(url: str) -> bool:
    parts = urlparse(url)
    rp = RobotFileParser()
    rp.set_url(f"{parts.scheme}://{parts.netloc}/robots.txt")
    try:
        rp.read()
    except Exception:
        return True  # robots.txt unreachable; proceed
    return rp.can_fetch(USER_AGENT, url)


def split_long(paragraph: str) -> list[str]:
    """Split an oversized paragraph on sentence boundaries."""
    if len(paragraph) <= MAX_CHARS:
        return [paragraph]
    out, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", paragraph):
        if cur and len(cur) + len(sent) > MAX_CHARS:
            out.append(cur)
            cur = ""
        cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur)
    return out


def chunk_markdown(text: str) -> list[tuple[str, str]]:
    """Return [(section_heading, chunk_text)], splitting on headings then paragraphs."""
    sections, heading, buf = [], "", []
    for line in text.splitlines():
        if re.match(r"^#{1,4}\s", line):
            sections.append((heading, "\n".join(buf)))
            heading, buf = line.lstrip("# ").strip(), []
        else:
            buf.append(line)
    sections.append((heading, "\n".join(buf)))

    chunks = []
    for heading, body in sections:
        paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        paras = [piece for p in paras for piece in split_long(p)]
        cur = ""
        for p in paras:
            if cur and len(cur) + len(p) > MAX_CHARS:
                chunks.append((heading, cur))
                cur = ""
            cur = f"{cur}\n\n{p}" if cur else p
        if cur:
            chunks.append((heading, cur))
    return [(h, c) for h, c in chunks if len(c) >= MIN_CHARS]


def ingest_url(url: str) -> str:
    """Ingest one URL. Returns a short status message."""
    if not allowed_by_robots(url):
        return "skipped (blocked by robots.txt)"

    html = trafilatura.fetch_url(url)
    if not html:
        return "failed (could not download)"

    text = trafilatura.extract(
        html,
        output_format="markdown",
        include_comments=False,
        include_tables=True,
        favor_precision=True,
    )
    if not text:
        return "failed (no main content found; page may need JavaScript)"

    meta = trafilatura.extract_metadata(html)
    title = (meta.title if meta and meta.title else url).strip()
    date = (meta.date if meta and meta.date else "") or ""
    site = (meta.sitename if meta and meta.sitename else urlparse(url).netloc) or ""

    content_hash = hashlib.sha256(text.encode()).hexdigest()
    col = get_collection()
    existing = col.get(where={"url": url}, limit=1)
    if existing["ids"] and existing["metadatas"][0].get("content_hash") == content_hash:
        return "unchanged (skipped)"
    if existing["ids"]:
        col.delete(where={"url": url})  # page changed: replace old chunks

    chunks = chunk_markdown(text)
    if not chunks:
        return "failed (no usable chunks)"

    # Prepend title + heading so every chunk carries its own context.
    docs = [
        f"{title} > {h}\n\n{c}" if h else f"{title}\n\n{c}" for h, c in chunks
    ]
    col.add(
        ids=[hashlib.sha1(f"{url}#{i}".encode()).hexdigest() for i in range(len(docs))],
        documents=docs,
        embeddings=embed_documents(docs),
        metadatas=[
            {
                "url": url,
                "title": title,
                "site": site,
                "date": date,
                "heading": h,
                "content_hash": content_hash,
            }
            for h, _ in chunks
        ],
    )
    return f"ok ({len(docs)} chunks)"


def ingest_urls(urls: list[str], progress=print) -> None:
    for i, url in enumerate(urls):
        progress(f"[{i + 1}/{len(urls)}] {url} ...")
        try:
            progress(f"    {ingest_url(url)}")
        except Exception as e:  # keep going on errors
            progress(f"    error: {e}")
        if i < len(urls) - 1:
            time.sleep(CRAWL_DELAY)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="*")
    ap.add_argument("--urls-file", help="text file with one URL per line")
    args = ap.parse_args()

    urls = list(args.urls)
    if args.urls_file:
        with open(args.urls_file) as f:
            urls += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not urls:
        ap.error("provide at least one URL")
    ingest_urls(urls)
    print(f"Done. Collection now holds {get_collection().count()} chunks.")
