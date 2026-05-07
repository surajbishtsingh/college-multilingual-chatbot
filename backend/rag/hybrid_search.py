# hybrid_search.py — Fixed: removed language filter (no index on Qdrant Cloud)
import re
from qdrant_client import QdrantClient

STOP = {
    'what','who','is','are','the','at','in','of','a','an','and','or',
    'for','to','how','does','do','has','have','many','which','tell',
    'me','about','please','can','you','i','my','their','kya','hai',
    'hain','ka','ki','ke','mein','se','per','ek','gbpiet',
    'क्या','कौन','का','की','के','में','से','है','हैं','एक',
    'और','या','को','ने','था','थी','थे','कि','जो','तो','भी',
    'मैं','हम','आप','वे','इस','उस','यह','वह','पर','बारे',
    'कैसे','कहाँ','कहां','तक',
}

def extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r'[\u0900-\u097F]+|[a-zA-Z0-9]+', text.lower())
    return [t for t in tokens if t not in STOP and len(t) > 1]


def keyword_score(query_keywords: list[str], payload_text: str) -> float:
    if not query_keywords or not payload_text:
        return 0.0
    text_lower = payload_text.lower()
    hits = sum(1 for kw in query_keywords if kw in text_lower)
    return hits / len(query_keywords)


def hybrid_search(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    query_text: str,
    limit: int = 5,
    lang_filter: str | None = None,   # kept for API compatibility but NOT used
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
) -> list[dict]:
    try:
        # ── No language filter — Qdrant Cloud has no index on 'language' ──
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit * 2,
            with_payload=True,
            with_vectors=False,
        )

        raw_results = response.points
        if not raw_results:
            return []

        # ── Re-rank with keyword overlap ──────────────────────────────
        q_keywords = extract_keywords(query_text)
        scored = []

        for hit in raw_results:
            text         = hit.payload.get("text", "")
            vector_sim   = hit.score
            kw_sim       = keyword_score(q_keywords, text)
            hybrid_score = (vector_weight * vector_sim) + (keyword_weight * kw_sim)

            scored.append({
                "text":         text,
                "score":        hybrid_score,
                "vector_score": vector_sim,
                "kw_score":     kw_sim,
                "url":          hit.payload.get("url", ""),
                "metadata": {
                    "source":   hit.payload.get("source",   ""),
                    "category": hit.payload.get("category", ""),
                    "language": hit.payload.get("language", ""),
                }
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        top = scored[:limit]

        if top:
            print(
                f"[Hybrid] {collection_name} → {len(top)} results | "
                f"top: {top[0]['score']:.3f} "
                f"(vec={top[0]['vector_score']:.3f}, "
                f"kw={top[0]['kw_score']:.3f})"
            )
        return top

    except Exception as e:
        print(f"[Hybrid] Error searching {collection_name}: {e}")
        return []


def multi_collection_search(
    client: QdrantClient,
    collections: list[str],
    query_vector: list[float],
    query_text: str,
    limit: int = 3,
    lang_filter: str | None = None,   # kept for API compatibility but NOT used
) -> list[dict]:
    all_results = []

    for collection in collections:
        try:
            results = hybrid_search(
                client=client,
                collection_name=collection,
                query_vector=query_vector,
                query_text=query_text,
                limit=limit,
            )
            all_results.extend(results)
        except Exception as e:
            print(f"[MultiSearch] Skipping {collection}: {e}")
            continue

    if not all_results:
        return []

    # Deduplicate by first 100 chars of text
    seen, unique = set(), []
    for r in all_results:
        key = r["text"][:100]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    unique.sort(key=lambda x: x["score"], reverse=True)
    return unique[:limit]
