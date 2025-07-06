# 📚 rag-formulate

**RAG-based reformulation using reference texts**  
This project allows you to reformulate an input document using exact fragments from a reference corpus, like Proust or the Bible. It combines embedding-based retrieval and LLM-guided composition.

---

## 🔧 Project Structure

```
.
├── data
│   ├── embeddings/               # FAISS indexes
│   ├── input/                    # Raw user input texts
│   │   ├── companyPolicies.txt
│   │   └── to_rewrite.txt
│   ├── output/                   # Reformulated outputs
│   │   └── reformulated.txt
│   ├── processed_chunks/         # Chunked reference data
│   └── reference_texts/          # Full reference corpora
│       ├── bible_catholic_public.txt
│       ├── dummy_text.txt
│       ├── genesis.txt
│       ├── proust_full.txt
│       └── proust.txt
├── main.py
├── Makefile
├── README.md
├── reformulate.py                # Main script
├── requirements.txt
├── sandbox.py
├── src/
│   ├── chunking.py
│   ├── embed.py
│   ├── generation.py
│   ├── __init__.py
│   ├── retrieval.py
│   └── utils.py
└── tests/
    ├── test_chunking.py
    ├── test_embed.py
    ├── test_generation.py
    ├── test_reformulate.py
    ├── test_retrieval.py
    └── test_utils.py
```

---

## 🚀 Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### 1. Reformulate with sentence splitting (default)

```bash
python reformulate.py \
    --input data/input/companyPolicies.txt \
    --ref data/reference_texts/proust.txt \
    --output data/output/reformulated.txt
```

📌 Each sentence is treated independently and matched to reference fragments before being reformulated.

---

### 2. Reformulate raw paragraph (no sentence splitting)

```bash
python reformulate.py \
    --input data/input/companyPolicies.txt \
    --ref data/reference_texts/proust.txt \
    --output data/output/reformulated.txt \
    --raw
```

📌 The full input is used as one block. Useful for poetic or short-form texts.

---

## 🧠 How It Works

1. Input is chunked (or not, with `--raw`)
2. Embeddings are created and compared with pre-embedded reference chunks
3. Best matches are selected and passed to an LLM (e.g., via Mistral API)
4. Output is generated using only original reference text fragments

---

## ✅ Tests

Run all unit tests:

```bash
pytest tests/
```