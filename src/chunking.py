# src/chunking.py

import spacy
from typing import List
import subprocess
import importlib.util


def ensure_spacy_model(model: str = "en_core_web_sm"):
    if not importlib.util.find_spec(model):
        subprocess.run(["python", "-m", "spacy", "download", model], check=True)


class SentenceChunker:
    """
    Uses spaCy's built-in sentence segmentation (faster, cleaner than NLTK).
    """
    def __init__(self, method= "sentence", model: str = "en_core_web_sm", window_size=20):
        self.method = method
        self.window_size = window_size
        if method == "sentence":
            ensure_spacy_model(model)
            self.nlp = spacy.load(model)

    def chunk(self, text: str) -> List[str]:
        """
        Chunk text into sentences (using spacy) or sliding windows of tokens.
        Args:
            text (str): The input text to be chunked.
        Returns:
            List[str]: A list of text chunks.
        """
        text = text.strip().replace('\n', ' ')
        if self.method == "sentence":
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 3]
        elif self.method == "sliding":
            tokens = text.split()
            return [
                " ".join(tokens[i:i + self.window_size])
                for i in range(0, len(tokens), self.window_size)
                if len(tokens[i:i + self.window_size]) > 3
            ]
        else:
            raise ValueError(f"Unknown chunking method: {self.method}")
