# src/embed.py

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class Embedder:
    """
    Wrapper for SentenceTransformer embedding model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts (sentences or chunks) into dense vectors.
        Returns a 2D NumPy array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)
