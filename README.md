# Document Q&A (RAG pipeline)

Upload a PDF/TXT and ask questions about it. Answers come with page-numbered sources.

- **Retrieval:** TF-IDF (offline, default) or OpenAI embeddings + FAISS (`RAG_BACKEND=openai`).
- **Answering:** if `OPENAI_API_KEY` is set, an LLM answers from the retrieved chunks and cites pages; otherwise the best matching passage is returned.
- **Stack:** Python, FastAPI, Streamlit, scikit-learn, Docker.

## Run
```bash
pip install -r requirements.txt
uvicorn api:app --port 8000 &
streamlit run app.py
```
Or `docker build -t doc-qa . && docker run -p 8000:8000 -p 8501:8501 doc-qa`.

## Test
```bash
pip install pytest httpx && pytest
```

> No accuracy or speed figures are claimed here. Measure them on your own documents before quoting numbers.
