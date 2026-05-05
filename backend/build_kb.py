# ingest_to_qdrant.py
# ─────────────────────────────────────────────────────────────
# Run this ONCE locally to upload all JSON data to Qdrant Cloud
#
# Usage:
#   cd backend
#   pip install qdrant-client sentence-transformers python-dotenv
#   python ingest_to_qdrant.py
# ─────────────────────────────────────────────────────────────

import os
import json
import glob
import uuid
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter,
    FieldCondition, MatchValue,
)
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────
QDRANT_URL     = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
DATA_FOLDER    = os.path.join(os.path.dirname(__file__), "data")
EMBED_MODEL    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_SIZE    = 384
BATCH_SIZE     = 50

# ── Collection routing by category/lang ───────────────────────
COLLECTION_MAP = {
    ("admission",   "en"): "gbpiet_admissions",
    ("eligibility", "en"): "gbpiet_admissions",
    ("fees",        "en"): "gbpiet_fees",
    ("fee",         "en"): "gbpiet_fees",
    ("hostel",      "en"): "gbpiet_hostel",
    ("admission",   "hi"): "gbpiet_kb_hi",
    ("fees",        "hi"): "gbpiet_kb_hi",
    ("hostel",      "hi"): "gbpiet_kb_hi",
}
DEFAULT_EN = "gbpiet_kb_en"
DEFAULT_HI = "gbpiet_kb_hi"
FAQ_COLLECTION = "gbpiet_faq"

ALL_COLLECTIONS = [
    "gbpiet_faq",
    "gbpiet_kb_en",
    "gbpiet_kb_hi",
    "gbpiet_web",
    "gbpiet_hostel",
    "gbpiet_fees",
    "gbpiet_admissions",
]


def get_collection(item: dict) -> str:
    category = item.get("category", "").lower().strip()
    lang     = item.get("lang", "en").lower().strip()

    # FAQ items go to FAQ collection
    if category in ("faq", "general", ""):
        return FAQ_COLLECTION if lang == "en" else DEFAULT_HI

    key = (category, lang)
    if key in COLLECTION_MAP:
        return COLLECTION_MAP[key]

    return DEFAULT_EN if lang == "en" else DEFAULT_HI


def load_all_json(data_folder: str) -> list[dict]:
    all_items = []
    files = sorted(glob.glob(os.path.join(data_folder, "*.json")))

    if not files:
        print(f"❌ No JSON files found in {data_folder}")
        return []

    for filepath in files:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data if isinstance(data, list) else [data]
            count = 0

            for item in items:
                if not isinstance(item, dict):
                    continue

                answer = item.get("answer", "")
                if not answer or not answer.strip():
                    continue

                q_field = item.get("question", "")
                if isinstance(q_field, str):
                    questions = [q_field] if q_field.strip() else []
                elif isinstance(q_field, list):
                    questions = [q for q in q_field if isinstance(q, str) and q.strip()]
                else:
                    questions = []

                # Build text for embedding = question + answer
                primary_q = questions[0] if questions else ""
                text_for_embed = f"{primary_q}\n{answer}".strip()

                all_items.append({
                    "text":       text_for_embed,
                    "question":   primary_q,
                    "answer":     answer.strip(),
                    "category":   item.get("category", "general"),
                    "tags":       item.get("tags", []),
                    "lang":       item.get("lang", "en"),
                    "source":     filename,
                    "item_id":    item.get("id", None),
                })
                count += 1

            print(f"  📄 {filename}: {count} items loaded")

        except Exception as e:
            print(f"  ❌ Error loading {filename}: {e}")

    print(f"\n✅ Total items loaded: {len(all_items)}")
    return all_items


def ensure_collections(client: QdrantClient):
    existing = {c.name for c in client.get_collections().collections}
    for name in ALL_COLLECTIONS:
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            print(f"  ✅ Created collection: {name}")
        else:
            count = client.get_collection(name).points_count
            print(f"  📦 {name}: already exists ({count} points)")


def embed_in_batches(model: SentenceTransformer, texts: list[str]) -> list:
    all_vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        vectors = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vectors.extend(vectors.tolist())
        print(f"  🔢 Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)}")
    return all_vectors


def upload_to_qdrant(client: QdrantClient, items: list[dict], vectors: list):
    # Group by collection
    groups: dict[str, list] = {}
    for item, vector in zip(items, vectors):
        col = get_collection(item)
        if col not in groups:
            groups[col] = []
        groups[col].append((item, vector))

    # Upload each group
    for collection, pairs in groups.items():
        points = []
        for item, vector in pairs:
            point_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "text":     item["text"],
                    "question": item["question"],
                    "answer":   item["answer"],
                    "category": item["category"],
                    "tags":     item["tags"],
                    "lang":     item["lang"],
                    "source":   item["source"],
                },
            ))

        # Upload in batches
        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]
            client.upsert(collection_name=collection, points=batch)

        count = client.get_collection(collection).points_count
        print(f"  ✅ {collection}: {len(pairs)} items uploaded → {count} total points")


def main():
    print("=" * 55)
    print("  GBPIET Qdrant Ingestion Script")
    print("=" * 55)

    # ── Validate env ──────────────────────────────────────
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("❌ Missing QDRANT_URL or QDRANT_API_KEY in .env")
        print("   Add them to backend/.env and retry.")
        return

    print(f"\n🔗 Connecting to Qdrant: {QDRANT_URL[:40]}...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    print("✅ Connected\n")

    # ── Ensure collections exist ──────────────────────────
    print("📦 Checking collections...")
    ensure_collections(client)

    # ── Load JSON data ────────────────────────────────────
    print(f"\n📂 Loading JSON from: {DATA_FOLDER}")
    items = load_all_json(DATA_FOLDER)
    if not items:
        print("❌ No data found. Check your data/ folder.")
        return

    # ── Embed ─────────────────────────────────────────────
    print(f"\n🤖 Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    print("✅ Model loaded\n")

    print(f"🔢 Embedding {len(items)} items...")
    texts   = [item["text"] for item in items]
    vectors = embed_in_batches(model, texts)
    print("✅ Embedding complete\n")

    # ── Upload ────────────────────────────────────────────
    print("⬆️  Uploading to Qdrant Cloud...")
    upload_to_qdrant(client, items, vectors)

    # ── Final summary ─────────────────────────────────────
    print("\n" + "=" * 55)
    print("✅ INGESTION COMPLETE — Collection summary:")
    print("=" * 55)
    for name in ALL_COLLECTIONS:
        count = client.get_collection(name).points_count
        status = "✅" if count > 0 else "⚠️  empty"
        print(f"  {status}  {name}: {count} points")
    print("=" * 55)
    print("\nYour Qdrant is now populated!")
    print("Redeploy on Railway — vector search will now work.")


if __name__ == "__main__":
    main()
