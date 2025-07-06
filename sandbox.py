# sandbox.py

# from src.generation import Reformulator

# ref = Reformulator()
# output = ref.reformulate("I love LLMs.", [
#     "Language models are quite interesting.",
#     "You should try using one."
# ])
# print("✨ Reformulated:", output)

import argparse
from src.utils import is_too_large_for_embedding

def main():
    parser = argparse.ArgumentParser(description="Check if a file exceeds the embedding character limit.")
    parser.add_argument("filepath", type=str, help="Path to the text file to check.")
    parser.add_argument("--limit", type=int, default=100_000, help="Character limit for embedding.")
    parser.add_argument("--count", action="store_true", help="Print character count.")
    args = parser.parse_args()

    result = is_too_large_for_embedding(args.filepath, limit=args.limit, count=args.count)

    if args.count:
        too_large, char_count = result
        print(f"Character count: {char_count}")
    else:
        too_large = result

    if too_large:
        print("⚠️  File too large for embedding.")
    else:
        print("✅ File is OK for embedding.")

if __name__ == "__main__":
    main()

