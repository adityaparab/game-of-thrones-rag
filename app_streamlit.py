"""Browser chat UI.  streamlit run app_streamlit.py

Streams the answer and shows numbered, clickable sources under each reply.
"""
import streamlit as st

import rag
import agent

st.set_page_config(page_title="GoT Wiki Assistant", page_icon="🐉")
st.title("🐉 Game of Thrones Wiki Assistant")

mode = st.sidebar.radio("Mode", ["agentic", "simple"],
                        help="agentic = query rewriting + multi-hop re-retrieval")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about Westeros..."):
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching the wiki..."):
            stream, sources = (agent.answer(prompt) if mode == "agentic"
                               else rag.answer(prompt))
        answer_text = st.write_stream(stream)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    label = f"[{s['n']}] {s['title']} — {s['section']}"
                    st.markdown(f"- [{label}]({s['url']})" if s["url"] else f"- {label}")
    st.session_state.history.append({"role": "assistant", "content": answer_text})
