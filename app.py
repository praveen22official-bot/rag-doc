"""Streamlit front end. Run the API first:  uvicorn api:app --port 8000"""
import os

import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="Document Q&A", page_icon="📄")
st.title("📄 Ask your document")

up = st.file_uploader("Upload a PDF or TXT", type=["pdf", "txt"])
if up and st.session_state.get("loaded") != up.name:
    r = requests.post(f"{API}/upload", files={"file": (up.name, up.getvalue())}, timeout=120)
    if r.ok:
        st.session_state["loaded"] = up.name
        st.success(f"Indexed {r.json()['chunks']} chunks from {up.name}")
    else:
        st.error(r.text)

q = st.text_input("Your question")
if q and st.session_state.get("loaded"):
    r = requests.post(f"{API}/ask", json={"question": q}, timeout=120)
    if r.ok:
        data = r.json()
        st.markdown(data["answer"])
        with st.expander("Sources"):
            for s in data["sources"]:
                st.caption(f"page {s['page']} · score {s['score']}")
                st.write(s["text"])
    else:
        st.error(r.text)
