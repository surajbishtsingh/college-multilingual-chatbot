# backend/qdrant_setup.py
import os
import atexit
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, OptimizersConfigDiff
from dotenv import load_dotenv

load_dotenv()

# ── Connection settings ────────────────────────────────────────────────
QDRANT_URL     = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_PATH    = os.path.join(os.path.dirname(__file__), "qdrant_storage")

USE_CLOUD = bool(QDRANT_URL and "qdrant.io" in QDRANT_URL)

EMBED_DIM = 384

# ── Collection names ───────────────────────────────────────────────────
# Use prefix for cloud to avoid conflicts
PREFIX = "gbpiet_" if USE_CLOUD else ""

COLLECTIONS = {
    f"{PREFIX}kb_en":      "General English knowledge base",
    f"{PREFIX}kb_hi":      "General Hindi knowledge base",
    f"{PREFIX}faq":        "Frequently asked questions",
    f"{PREFIX}admissions": "Admission process documents",
    f"{PREFIX}fees":       "Fee structure documents",
    f"{PREFIX}hostel":     "Hostel information documents",
    f"{PREFIX}website":    "Live scraped website content",
}

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if USE_CLOUD:
            _client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                timeout=30,
            )
            print(f"[Qdrant] Connected to Cloud → {QDRANT_URL[:50]}...")
        else:
            os.makedirs(QDRANT_PATH, exist_ok=True)
            _client = QdrantClient(path=QDRANT_PATH)
            print(f"[Qdrant] Connected to Local → {QDRANT_PATH}")

        atexit.register(_close_client)
    return _client


def _close_client():
    global _client
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
        _client = None


def ensure_collections(recreate: bool = False):
    client   = get_client()
    existing = {c.name for c in client.get_collections().collections}

    for name in COLLECTIONS:
        if name in existing:
            if recreate:
                client.delete_collection(name)
                print(f"[Qdrant] Deleted: {name}")
            else:
                print(f"[Qdrant] Exists: {name}")
                continue

        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=EMBED_DIM,
                distance=Distance.COSINE,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=5000
            ),
        )
        print(f"[Qdrant] ✅ Created: {name}")


def collection_info():
    client = get_client()
    for name in COLLECTIONS:
        try:
            info = client.get_collection(name)
            print(f"  [Qdrant] {name}: {info.points_count} points")
        except Exception:
            print(f"  [Qdrant] {name}: NOT FOUND")
