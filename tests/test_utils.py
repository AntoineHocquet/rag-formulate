# tests/test_utils.py
from src.utils import is_too_large_for_embedding
import tempfile

def test_is_too_large_for_embedding():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write("x" * 100_001)
        f.flush()
        assert is_too_large_for_embedding(f.name) is True

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as f:
        f.write("x" * 99_999)
        f.flush()
        assert is_too_large_for_embedding(f.name) is False
