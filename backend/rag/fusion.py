# fusion.py — Reciprocal Rank Fusion (RRF) to merge BM25 + Qdrant results
#
# RRF formula: score(d) = Σ 1 / (k + rank(d))
# where k=60 is a smoothing constant.
# Documents appearing high in BOTH lists get the highest scores.


def reciprocal_rank_fusion(
    bm25_results:   list[dict],
    vector_results: list[dict],
    k: int = 60,
    bm25_weight:   float = 0.4,
    vector_weight: float = 0.6,
) -> list[dict]:
    """
    Merge BM25 and Qdrant vector results using Reciprocal Rank Fusion.

    Args:
        bm25_results   : from bm25_search()   — must have 'text', 'answer', 'rank'
        vector_results : from hybrid_search()  — must have 'text', 'score'
        k              : RRF smoothing constant (default 60)
        bm25_weight    : contribution of BM25 ranks  (default 0.4)
        vector_weight  : contribution of vector ranks (default 0.6)

    Returns:
        Merged list sorted by RRF score, highest first.
        Each item has: text, answer, source, rrf_score, origin
    """
    # Use first 100 chars of text as dedup key
    scores: dict[str, dict] = {}

    # ── Score BM25 results ─────────────────────────────────────────
    for rank, doc in enumerate(bm25_results, start=1):
        key       = doc["text"][:100]
        rrf_score = bm25_weight * (1.0 / (k + rank))
        if key not in scores:
            scores[key] = {
                "text":      doc["text"],
                "answer":    doc.get("answer", ""),
                "source":    doc.get("source", ""),
                "rrf_score": 0.0,
                "origin":    set(),
            }
        scores[key]["rrf_score"] += rrf_score
        scores[key]["origin"].add("bm25")

    # ── Score vector results ───────────────────────────────────────
    for rank, doc in enumerate(vector_results, start=1):
        key       = doc["text"][:100]
        rrf_score = vector_weight * (1.0 / (k + rank))
        if key not in scores:
            scores[key] = {
                "text":      doc["text"],
                "answer":    doc.get("answer", ""),
                "source":    doc.get("metadata", {}).get("source", ""),
                "rrf_score": 0.0,
                "origin":    set(),
            }
        scores[key]["rrf_score"] += rrf_score
        scores[key]["origin"].add("vector")

    # ── Sort and label ─────────────────────────────────────────────
    merged = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    # Convert origin set to string for logging
    for item in merged:
        item["origin"] = "+".join(sorted(item["origin"]))

    if merged:
        print(
            f"[RRF] Merged {len(bm25_results)} BM25 + "
            f"{len(vector_results)} vector → "
            f"{len(merged)} unique | "
            f"top RRF: {merged[0]['rrf_score']:.4f} "
            f"({merged[0]['origin']})"
        )

    return merged