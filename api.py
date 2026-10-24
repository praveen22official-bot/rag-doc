"""FastAPI backend: upload a PDF/TXT, then ask questions about it."""
import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from rag import DocumentIndex, answer, chunk_pages, load_text

app = FastAPI(title="Document Q&A (RAG)")
STATE: dict[str, DocumentIndex] = {}
BACKEND = os.getenv("RAG_BACKEND", "tfidf")


class Question(BaseModel):
    question: str
    k: int = 4


@app.get("/health")
def health():
    return {"status": "ok", "indexed": "doc" in STATE, "backend": BACKEND}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in {".pdf", ".txt"}:
        raise HTTPException(400, "Only .pdf and .txt files are supported.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
    try:
        chunks = chunk_pages(load_text(tmp.name))
        STATE["doc"] = DocumentIndex(backend=BACKEND).build(chunks)
    except ValueError as e:
        raise HTTPException(422, str(e))
    finally:
        os.unlink(tmp.name)
    return {"filename": file.filename, "chunks": len(chunks)}


@app.post("/ask")
def ask(q: Question):
    if "doc" not in STATE:
        raise HTTPException(409, "Upload a document first.")
    return answer(STATE["doc"], q.question, q.k)
