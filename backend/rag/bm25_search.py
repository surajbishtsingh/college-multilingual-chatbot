# bm25_search.py — BM25 index over the QA database
import re
import os
import json
import glob
from rank_bm25 import BM25Okapi

_bm25_index  = None
_bm25_corpus = []   # list of dicts: {tokens, question, answer, source}


# ── Tokenizer (works for English + Hindi) ─────────────────────────────
def tokenize(text: str) -> list[str]:
    """
    Split text into lowercase tokens.
    Handles both Latin (English) and Devanagari (Hindi) scripts.
    """
    text   = text.lower().strip()
    tokens = re.findall(r'[\u0900-\u097F]+|[a-zA-Z0-9]+', text)
    return [t for t in tokens if len(t) > 1]


def build_bm25_index(data_folder: str | None = None) -> BM25Okapi:
    """
    Build BM25 index from all JSON files in data folder.
    Indexes question + answer together for better recall.
    Cached after first build.
    """
    global _bm25_index, _bm25_corpus

    if _bm25_index is not None:
        return _bm25_index

    if data_folder is None:
        # Primary: backend/data/ (one level up from rag/)
        data_folder = os.path.join(os.path.dirname(__file__), "..", "data")
        data_folder = os.path.normpath(data_folder)

    # Guard: folder doesn't exist
    if not os.path.exists(data_folder):
        print(f"[BM25] Data folder not found: {data_folder} — building empty index")
        _bm25_index = BM25Okapi([[]])
        return _bm25_index

    corpus_tokens = []

    for filepath in sorted(glob.glob(os.path.join(data_folder, "*.json"))):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                q = item.get("question", "")
                a = item.get("answer",   "")
                if not isinstance(q, str) or not isinstance(a, str):
                    continue
                if not q.strip() or not a.strip():
                    continue

                # Index question + answer combined
                combined = q + " " + a
                tokens   = tokenize(combined)

                if tokens:
                    _bm25_corpus.append({
                        "tokens":   tokens,
                        "question": q.strip(),
                        "answer":   a.strip(),
                        "source":   os.path.basename(filepath),
                    })
                    corpus_tokens.append(tokens)

        except Exception as e:
            print(f"[BM25] Error loading {filepath}: {e}")

    # Guard: no data found
    if not corpus_tokens:
        print(f"[BM25] No JSON files found in {data_folder} — building empty index")
        _bm25_index = BM25Okapi([[]])
        return _bm25_index

    _bm25_index = BM25Okapi(corpus_tokens)
    print(f"[BM25] Index built — {len(_bm25_corpus)} documents from {data_folder}")
    return _bm25_index


def bm25_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Search the BM25 index.

    Returns list of:
    {
        "question":   str,
        "answer":     str,
        "source":     str,
        "bm25_score": float,
        "rank":       int,
        "text":       str,
    }
    """
    index = build_bm25_index()
    if not _bm25_corpus:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = index.get_scores(query_tokens)

    # Pair each doc with its score and sort
    ranked = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for rank, (idx, score) in enumerate(ranked[:top_k], start=1):
        if score <= 0:
            break
        doc = _bm25_corpus[idx]
        results.append({
            "question":   doc["question"],
            "answer":     doc["answer"],
            "source":     doc["source"],
            "bm25_score": float(score),
            "rank":       rank,
            "text":       f"Question: {doc['question']}\nAnswer: {doc['answer']}",
        })

    if results:
        print(f"[BM25] '{query[:40]}' → {len(results)} results "
              f"(top score: {results[0]['bm25_score']:.3f})")
    return results