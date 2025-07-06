# reformulate.py

"""
Full end-to-end pipeline for rag-formulate
"""

from src.chunking import SentenceChunker
from src.embed import Embedder
from src.retrieval import Retriever
from src.generation import Reformulator

import numpy as np
import argparse
import os

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def save_output(sentences, path):
    with open(path, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")

def main(args):
    print("🔹 Loading and chunking reference text...")
    reference_text = read_text_file(args.reference)
    chunker = SentenceChunker()
    reference_chunks = chunker.chunk(reference_text)

    print(f"✅ Reference chunks: {len(reference_chunks)}")

    embedder = Embedder()
    print("🔹 Embedding reference chunks...")
    reference_embeddings = embedder.embed(reference_chunks)
    dim = reference_embeddings.shape[1]

    retriever = Retriever(dim)
    retriever.add(reference_embeddings, reference_chunks)

    print("🔹 Chunking input text...")
    input_text = read_text_file(args.input)
    input_chunks = chunker.chunk(input_text)
    print(f"✅ Input chunks: {len(input_chunks)}")

    reformulator = Reformulator(raw_mode=args.raw)

    reformulated_sentences = []
    for i, sentence in enumerate(input_chunks):
        print(f"\n✏️ Reformulating [{i+1}/{len(input_chunks)}]: {sentence}")
        input_vec = embedder.embed([sentence])
        retrieved = retriever.query(input_vec, k=args.top_k)
        retrieved_texts = [t for t, _ in retrieved]

        new_sentence = reformulator.reformulate(sentence, retrieved_texts)
        print(f"➡️  {new_sentence}")
        reformulated_sentences.append(new_sentence)

    if args.output:
        save_output(reformulated_sentences, args.output)
        print(f"\n📁 Reformulated text saved to: {args.output}")
    else:
        print("\n📝 Final reformulated text:\n")
        print("\n".join(reformulated_sentences))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rag-formulate full pipeline")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference text (e.g., Proust corpus)")
    parser.add_argument("--input", type=str, required=True, help="Path to user input text to reformulate")
    parser.add_argument("--output", type=str, help="Path to save reformulated text")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top chunks to retrieve")
    parser.add_argument("--raw", action="store_true", help="Use raw mode (no API calls, just return first retrieved chunk)")

    args = parser.parse_args()
    main(args)
