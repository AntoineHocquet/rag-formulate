# tests/test_retrieval.py

import numpy as np
from src.retrieval import Retriever

def test_retriever():
    dim = 384
    texts = ["This is test sentence one.", "Another sentence.", "More text to test."]
    embeddings = np.random.rand(len(texts), dim)

    retriever = Retriever(dim=dim)
    retriever.add(embeddings, texts)

    query = embeddings[0]  # test with known vector
    results = retriever.query(query, k=2)

    assert len(results) == 2
    assert results[0][0] in texts
    assert isinstance(results[0][1], float)
