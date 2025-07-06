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
    def __init__(self, model: str = "en_core_web_sm"):
        ensure_spacy_model(model)
        self.nlp = spacy.load(model)

    def chunk(self, text: str) -> List[str]:
        """
        Splits text into sentences using spaCy's sentence segmentation.
        """
        doc = self.nlp(text.strip().replace("\n", " "))
        return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 3]
