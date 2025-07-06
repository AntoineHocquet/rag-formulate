# src/utils.py

def is_too_large_for_embedding(filepath: str, limit: int = 100_000, count: bool = False):
    """
    Check if a file exceeds the character limit for sentence embedding.

    Args:
        filepath (str): Path to the text file.
        limit (int): Character limit for sentence-transformers (default: 100,000).
        count (bool): If True, also return the character count.

    Returns:
        bool or (bool, int): Whether the file is too large, and optionally the count.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    char_count = len(text)
    too_large = char_count > limit
    return (too_large, char_count) if count else too_large
