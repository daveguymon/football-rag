"""Streamlit chat UI. Run with: streamlit run app.py"""
import streamlit as st

from ingest import ingest_urls
from rag import ContextBudgetError, DecompositionError, answer_stream
from store import LLM_MODEL, TOP_K, get_collection

st.set_page_config(page_title="Football Strategy RAG", page_icon="🏈")
st.title("🏈 Football Strategy Chat")

# ---------- Sidebar: add sources ----------
with st.sidebar:
    st.header("Sources")
    st.caption(f"{get_collection().count()} chunks indexed · LLM: {LLM_MODEL}")

    url_text = st.text_area("Add URLs (one per line)", height=120)
    if st.button("Ingest URLs", use_container_width=True):
        urls = [u.strip() for u in url_text.splitlines() if u.strip()]
        if urls:
            log = st.empty()
            lines: list[str] = []

            def progress(msg: str) -> None:
                lines.append(msg)
                log.code("\n".join(lines))

            with st.spinner("Fetching and embedding..."):
                ingest_urls(urls, progress=progress)
            st.success("Done. The chunk count above updates on your next interaction.")
        else:
            st.warning("Paste at least one URL.")

    k = st.slider("Chunks to retrieve", 2, 12, TOP_K)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                st.markdown(msg["sources"])

if question := st.chat_input("Ask about football strategy..."):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching sources..."):
                hits, stream = answer_stream(question, history, k)
        except (ContextBudgetError, DecompositionError) as e:
            st.session_state.messages.pop()
            st.error(str(e))
        else:
            answer = st.write_stream(stream)

            sources_md = "\n\n".join(
                f"**[{i}] [{h['meta']['title']}]({h['meta']['url']})** "
                f"· {h['meta']['site']} · similarity {1 - h['distance']:.2f}\n\n"
                f"> {h['text'][:300].replace(chr(10), ' ')}..."
                for i, h in enumerate(hits, 1)
            )
            if sources_md:
                with st.expander("Sources"):
                    st.markdown(sources_md)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources_md}
            )
