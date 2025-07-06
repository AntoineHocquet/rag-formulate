# tests/test_chunking.py

from src.chunking import SentenceChunker

def test_chunker():
    chunker = SentenceChunker()
    text = "This is sentence one. This is sentence two! Is this sentence three? Yes."
    chunks = chunker.chunk(text)
    assert len(chunks) == 4
    assert chunks[0] == "This is sentence one."
    assert chunks[-1] == "Yes."
