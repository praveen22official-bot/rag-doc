import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient

from api import app
from rag import DocumentIndex, chunk_pages

DOC = (
    "The warranty covers manufacturing defects for twenty-four months from purchase. "
    "Batteries are covered for twelve months only. " * 3
    + "To request a refund, contact support with your order number within thirty days. "
    "Refunds are issued to the original payment method within five business days. " * 3
    + "Shipping to remote islands takes up to three weeks and requires customs paperwork. " * 3
)


def test_chunking_keeps_page_numbers():
    chunks = chunk_pages([(1, DOC), (2, "second page text here")])
    assert chunks and {c["page"] for c in chunks} == {1, 2}


def test_retrieval_finds_relevant_chunk():
    idx = DocumentIndex().build(chunk_pages([(1, DOC)], size=30, overlap=5))
    hits = idx.search("how do I get a refund?", k=1)
    assert "refund" in hits[0]["text"].lower()


def test_api_flow():
    c = TestClient(app)
    assert c.post("/ask", json={"question": "x"}).status_code in (200, 409)
    r = c.post("/upload", files={"file": ("doc.txt", DOC.encode())})
    assert r.status_code == 200
    out = c.post("/ask", json={"question": "how long is the warranty?"}).json()
    assert out["sources"] and "warranty" in out["answer"].lower()
    assert c.post("/upload", files={"file": ("x.exe", b"1")}).status_code == 400
