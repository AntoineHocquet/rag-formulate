# tests/test_embed.py

from src.embed import Embedder
import numpy as np

def test_embedder():
    model = Embedder()
    sentences = ["Hello world.", "This is a test sentence."]
    embeddings = model.embed(sentences)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0  # should be 384 for MiniLM
