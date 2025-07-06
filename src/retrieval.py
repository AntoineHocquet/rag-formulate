# src/retrieval.py

from sklearn.preprocessing import normalize
import faiss
import numpy as np
from typing import List, Tuple

class Retriever:
    def __init__(self, dim: int):
        """
        Initializes the FAISS index and stores text alongside vectors.
        :param dim: dimensionality of the embedding vectors
        """
        self.index = faiss.IndexFlatIP(dim)
        self.normalize = True # whether to normalize vectors
        self.text_chunks = []  # store associated texts

    def add(self, embeddings: np.ndarray, texts: List[str]):
        """
        Add vectors and corresponding texts to the index.
        :param embeddings: numpy array of shape (n_samples, dim)
        :param texts: list of n_samples strings
        """
        if embeddings.shape[0] != len(texts):
            raise ValueError("Number of embeddings and texts must match.")
        if self.normalize:
            embeddings = normalize(embeddings, axis=1)
        self.index.add(embeddings.astype(np.float32))
        self.text_chunks.extend(texts)

    def query(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve top-k nearest chunks for a given query vector.
        :param query_vector: shape (dim,) or (1, dim)
        :return: List of (text, distance)
        """
        if query_vector.ndim == 1:
            query_vector = query_vector[np.newaxis, :]
        if self.normalize:
            query_vector = normalize(query_vector, axis=1)
        distances, indices = self.index.search(query_vector.astype(np.float32), k)
        return [(self.text_chunks[i], float(dist)) for i, dist in zip(indices[0], distances[0])]
