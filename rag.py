"""Retrieval-augmented document Q&A.

Retrieval backends:
  * "tfidf"  - default, fully offline (scikit-learn).
  * "openai" - OpenAI embeddings + FAISS (needs OPENAI_API_KEY and `pip install faiss-cpu openai`).

Answer generation: if OPENAI_API_KEY is set, an LLM answers using the retrieved
chunks as context; otherwise the top passages are returned as an extractive answer.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_text(path: str) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for a .pdf or .txt file."""
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(path)
        return [(i + 1, p.extract_text() or "") for i, p in enumerate(reader.pages)]
    with open(path, encoding="utf-8") as f:
        return [(1, f.read())]


def chunk_pages(pages: list[tuple[int, str]], size: int = 120, overlap: int = 30) -> list[dict]:
    """Split pages into overlapping word-window chunks, keeping the page number."""
    chunks = []
    step = max(1, size - overlap)
    for page, text in pages:
        words = re.sub(r"\s+", " ", text).strip().split(" ")
        if not words or words == [""]:
            continue
        for start in range(0, len(words), step):
            piece = " ".join(words[start : start + size])
            if piece:
                chunks.append({"page": page, "text": piece})
            if start + size >= len(words):
                break
    return chunks


@dataclass
class DocumentIndex:
    backend: str = "tfidf"
    chunks: list[dict] = field(default_factory=list)
    _vec: TfidfVectorizer | None = None
    _matrix: object = None
    _faiss: object = None

    def build(self, chunks: list[dict]) -> "DocumentIndex":
        if not chunks:
            raise ValueError("No text could be extracted from the document.")
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        if self.backend == "openai":
            import faiss  # type: ignore

            emb = _openai_embed(texts)
            faiss.normalize_L2(emb)
            self._faiss = faiss.IndexFlatIP(emb.shape[1])
            self._faiss.add(emb)
        else:
            self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
            self._matrix = self._vec.fit_transform(texts)
        return self

    def search(self, question: str, k: int = 4) -> list[dict]:
        if self.backend == "openai":
            import faiss  # type: ignore

            q = _openai_embed([question])
            faiss.normalize_L2(q)
            scores, idx = self._faiss.search(q, k)
            pairs = zip(idx[0], scores[0])
        else:
            q = self._vec.transform([question])
            sims = (self._matrix @ q.T).toarray().ravel()
            top = np.argsort(-sims)[:k]
            pairs = [(i, sims[i]) for i in top]
        return [
            {**self.chunks[int(i)], "score": float(s)} for i, s in pairs if int(i) >= 0 and float(s) > 0
        ]


def _openai_embed(texts: list[str]) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()
    out = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in out.data], dtype="float32")


def answer(index: DocumentIndex, question: str, k: int = 4) -> dict:
    hits = index.search(question, k)
    if not hits:
        return {"answer": "I couldn't find anything relevant in the document.", "sources": []}
    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        context = "\n\n".join(f"[page {h['page']}] {h['text']}" for h in hits)
        resp = OpenAI().chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "Answer only from the context. Cite page numbers. If the answer is not in the context, say so."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        text = resp.choices[0].message.content
    else:
        text = hits[0]["text"]
    return {"answer": text, "sources": [{"page": h["page"], "score": round(h["score"], 3), "text": h["text"]} for h in hits]}
