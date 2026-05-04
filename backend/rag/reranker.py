# backend/rag/reranker.py
# Reranks search results for better relevance before sending to LLM

import re
import os
from dotenv import load_dotenv

load_dotenv()

# ── Boost keywords by topic ────────────────────────────────────────────
# If query contains these words, boost results that also contain them
TOPIC_BOOST_KEYWORDS = {
    "admission":    ["admission", "jee", "gate", "counselling", "eligibility",
                     "seat", "apply", "utuee", "cutoff", "merit"],
    "fees":         ["fee", "fees", "tuition", "payment", "sbi", "collect",
                     "scholarship", "hostel fee", "mess fee"],
    "hostel":       ["hostel", "warden", "mess", "accommodation", "room",
                     "boys", "girls", "seats", "allotment"],
    "placement":    ["placement", "package", "lpa", "recruiter", "campus",
                     "amazon", "microsoft", "job", "offer"],
    "faculty":      ["faculty", "professor", "hod", "head", "department",
                     "dr", "prof", "teacher"],
    "courses":      ["btech", "mtech", "mca", "phd", "branch", "program",
                     "undergraduate", "postgraduate"],
    "contact":      ["contact", "phone", "email", "address", "website",
                     "helpline", "number"],
    "location":     ["reach", "route", "direction", "distance", "km",
                     "rishikesh", "haridwar", "dehradun", "bus"],
    "result":       ["result", "exam", "semester", "erp", "portal",
                     "timetable", "calendar"],
    "library":      ["library", "book", "digital", "reading", "resource"],
    "sports":       ["sport", "cricket", "football", "basketball", "gym",
                     "complex", "ground"],
    "bank":         ["bank", "sbi", "atm", "campus", "branch"],
    "ragging":      ["ragging", "anti", "helpline", "complaint", "zero"],
    "scholarship":  ["scholarship", "merit", "sc", "st", "obc", "stipend",
                     "financial", "aid"],
}

# ── Penalty keywords — results with these are less useful ─────────────
PENALTY_KEYWORDS = [
    "sitemap", "login", "wp-content", "javascript",
    "cookie", "privacy policy", "terms of service",
    "404", "page not found", "error",
]

# ── Source trust scores ───────────────────────────────────────────────
SOURCE_TRUST = {
    "website":    1.2,   # scraped from real college website — high trust
    "faq":        1.1,   # curated FAQ data — high trust
    "admissions": 1.1,
    "fees":       1.1,
    "hostel":     1.1,
    "kb_en":      1.0,
    "kb_hi":      1.0,
    "internet":   0.8,   # internet search — lower trust
    "duckduckgo": 0.7,
}


def detect_query_topic(query: str) -> str | None:
    """Detect which topic the query is about."""
    q = query.lower()
    best_topic = None
    best_count = 0

    for topic, keywords in TOPIC_BOOST_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in q)
        if count > best_count:
            best_count = count
            best_topic = topic

    return best_topic if best_count > 0 else None


def score_result(
    result:      dict,
    query:       str,
    query_topic: str | None = None,
) -> float:
    """
    Calculate a reranking score for a single result.

    Factors:
    1. Base RRF/hybrid score
    2. Topic keyword boost
    3. Query word overlap with result text
    4. Source trust multiplier
    5. Penalty for useless content
    6. Recency boost for website content
    """
    text   = result.get("text", "").lower()
    score  = result.get("rrf_score", result.get("score", 0.0))
    source = result.get("metadata", {}).get("source", "") or \
             result.get("source", "")

    # ── 1. Topic keyword boost ─────────────────────────────────────
    if query_topic and query_topic in TOPIC_BOOST_KEYWORDS:
        topic_kws = TOPIC_BOOST_KEYWORDS[query_topic]
        hits      = sum(1 for kw in topic_kws if kw in text)
        if hits > 0:
            boost = min(hits * 0.05, 0.25)   # max 0.25 boost
            score += boost

    # ── 2. Direct query word overlap ──────────────────────────────
    query_words = set(re.findall(r'[a-zA-Z\u0900-\u097F]{3,}', query.lower()))
    text_words  = set(re.findall(r'[a-zA-Z\u0900-\u097F]{3,}', text))
    overlap     = len(query_words & text_words)
    if overlap > 0:
        score += min(overlap * 0.02, 0.10)   # max 0.10 boost

    # ── 3. Source trust multiplier ─────────────────────────────────
    # Determine source category
    source_lower = source.lower()
    trust        = 1.0
    for src_key, src_trust in SOURCE_TRUST.items():
        if src_key in source_lower:
            trust = src_trust
            break

    score *= trust

    # ── 4. Penalty for useless content ────────────────────────────
    if any(p in text for p in PENALTY_KEYWORDS):
        score *= 0.3   # heavy penalty

    # ── 5. Length penalty — very short chunks are less useful ──────
    text_len = len(text.strip())
    if text_len < 50:
        score *= 0.5
    elif text_len < 100:
        score *= 0.8

    # ── 6. URL boost — results with URLs are more trustworthy ──────
    url = result.get("url", "") or result.get("metadata", {}).get("source", "")
    if url and url.startswith("https://gbpiet.ac.in"):
        score *= 1.15   # college website URL — high trust

    return score


def rerank(
    results:   list[dict],
    query:     str,
    top_k:     int = 3,
    min_score: float = 0.001,
) -> list[dict]:
    """
    Rerank a list of search results by relevance to the query.

    Args:
        results:   List of result dicts from hybrid search / BM25 / RAG
        query:     Original user query string
        top_k:     Number of results to return
        min_score: Minimum score threshold — results below this are dropped

    Returns:
        Top-k reranked results with updated scores
    """
    if not results:
        return []

    query_topic = detect_query_topic(query)
    if query_topic:
        print(f"[Reranker] Detected topic: {query_topic}")

    # Score each result
    scored = []
    for r in results:
        new_score = score_result(r, query, query_topic)
        r_copy    = dict(r)
        r_copy["rerank_score"]    = new_score
        r_copy["original_score"]  = r.get("rrf_score", r.get("score", 0.0))
        scored.append(r_copy)

    # Sort by rerank score
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Filter by minimum score
    filtered = [r for r in scored if r["rerank_score"] >= min_score]

    # Take top-k
    top = filtered[:top_k]

    if top:
        print(
            f"[Reranker] {len(results)} → {len(top)} results | "
            f"top score: {top[0]['rerank_score']:.4f} "
            f"(was: {top[0]['original_score']:.4f})"
        )

    return top


def rerank_with_diversity(
    results: list[dict],
    query:   str,
    top_k:   int = 3,
) -> list[dict]:
    """
    Rerank with diversity — avoids returning multiple chunks
    from the same source URL.

    Useful when website scraping returns many chunks from one page.
    """
    if not results:
        return []

    query_topic = detect_query_topic(query)
    scored      = []

    for r in results:
        new_score = score_result(r, query, query_topic)
        r_copy    = dict(r)
        r_copy["rerank_score"] = new_score
        scored.append(r_copy)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    # ── Select with diversity ──────────────────────────────────────
    seen_urls = set()
    diverse   = []

    for r in scored:
        url = (
            r.get("url")
            or r.get("metadata", {}).get("source", "")
            or ""
        )
        # Normalize URL — remove query params
        url_base = url.split("?")[0].rstrip("/")

        # Allow max 2 chunks from same URL
        url_count = sum(1 for u in seen_urls if u == url_base)
        if url_count < 2:
            seen_urls.add(url_base)
            diverse.append(r)

        if len(diverse) >= top_k:
            break

    if diverse:
        print(
            f"[Reranker+Diversity] {len(results)} → {len(diverse)} results | "
            f"top: {diverse[0]['rerank_score']:.4f}"
        )

    return diverse


# ══════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Simulate some search results
    test_results = [
        {
            "text": "GBPIET has a State Bank of India SBI branch and ATM on campus.",
            "rrf_score": 0.008,
            "url": "https://gbpiet.ac.in/bank-atms",
            "source": "website",
            "metadata": {"source": "website"},
        },
        {
            "text": "Students can use the SBI bank ATM inside the campus premises.",
            "rrf_score": 0.006,
            "url": "https://gbpiet.ac.in/facilities",
            "source": "faq",
            "metadata": {"source": "faq"},
        },
        {
            "text": "Login to ERP portal for results.",
            "rrf_score": 0.009,
            "url": "https://gbpiet.ac.in/erp",
            "source": "website",
            "metadata": {"source": "website"},
        },
        {
            "text": "Sitemap page with navigation links.",
            "rrf_score": 0.010,
            "url": "https://gbpiet.ac.in/sitemap",
            "source": "website",
            "metadata": {"source": "website"},
        },
    ]

    query = "which bank is available on campus"
    print(f"Query: {query}")
    print(f"Topic: {detect_query_topic(query)}")
    print()

    reranked = rerank(test_results, query, top_k=3)
    for i, r in enumerate(reranked, 1):
        print(f"{i}. Score: {r['rerank_score']:.4f} | {r['text'][:60]}")
