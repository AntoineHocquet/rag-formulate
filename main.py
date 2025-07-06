# main.py
"""
Playground / prototype CLI for testing chunks + retrieval
"""

from src.chunking import SentenceChunker
from src.embed import Embedder
from src.retrieval import Retriever

import numpy as np
import argparse
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main(args):
    print("🔹 Loading and chunking reference text...")
    raw_text = read_text_file(args.input)
    chunker = SentenceChunker()
    chunks = chunker.chunk(raw_text)
    print(f"✅ Found {len(chunks)} sentences.")

    print("🔹 Embedding sentences...")
    embedder = Embedder()
    embeddings = embedder.embed(chunks)
    dim = embeddings.shape[1]
    print(f"✅ Embedding shape: {embeddings.shape}")

    print("🔹 Building FAISS index...")
    retriever = Retriever(dim)
    retriever.add(embeddings, chunks)

    print(f"🔹 Retrieving matches for user input:\n\"{args.query}\"")
    query_vec = embedder.embed([args.query])
    results = retriever.query(query_vec, k=args.top_k)

    print("\n🔎 Top matches:")
    for i, (text, dist) in enumerate(results, 1):
        print(f"{i}. ({dist:.4f}) {text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rag-formulate prototype CLI")
    parser.add_argument("--input", type=str, required=True, help="Path to reference .txt file")
    parser.add_argument("--query", type=str, required=True, help="Query sentence")
    parser.add_argument("--top_k", type=int, default=3, help="Number of retrieved chunks")
    args = parser.parse_args()

    main(args)
