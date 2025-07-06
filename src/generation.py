# src/generation.py

import os
from typing import List
from mistralai.client import MistralClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Reformulator:
    """
    Reformulates a sentence using retrieved reference text chunks and the official Mistral API.
    """

    def __init__(self, model="mistral-small", api_key=None, raw_mode=False):
        self.raw_mode = raw_mode
        if not self.raw_mode:
            self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
            self.client = MistralClient(api_key=self.api_key)
            self.model = model

    def build_prompt(self, input_sentence: str, retrieved_chunks: List[str]) -> str:
        retrieved_text = "\n".join(f"- {chunk}" for chunk in retrieved_chunks)
        return (
            f"You are a stylistic editor. Your task is to rewrite the following sentence:\n\n"
            f"\"{input_sentence}\"\n\n"
            f"using only the phrasing and style from the following reference sentences:\n\n"
            f"{retrieved_text}\n\n"
            f"Make sure the meaning stays close to the original sentence, "
            f"but the wording resembles the style of the references as much as possible."
        )

    def reformulate(self, input_sentence: str, retrieved_chunks: List[str]) -> str:
        if self.raw_mode:
            # Return the first retrieved chunk (or random or joined, if you want)
            return retrieved_chunks[0].strip()

        prompt = self.build_prompt(input_sentence, retrieved_chunks)

        messages = [
            {"role": "system", "content": "You are a careful stylistic assistant."},
            {"role": "user", "content": prompt}
        ]

        response = self.client.chat(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=150,
        )
        return response.choices[0].message.content.strip()


