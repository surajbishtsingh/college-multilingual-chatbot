# qdrant_setup.py — Fixed with clean shutdown
import os
import atexit
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, OptimizersConfigDiff
)
from dotenv import load_dotenv

load_dotenv()

QDRANT_PATH = os.path.join(os.path.dirname(__file__), "qdrant_storage")
EMBED_DIM   = 384

COLLECTIONS = {
    "kb_en":      "General English knowledge base",
    "kb_hi":      "General Hindi knowledge base",
    "faq":        "Frequently asked questions (all languages)",
    "admissions": "Admission process documents",
    "fees":       "Fee structure documents",
    "hostel":     "Hostel information documents",
    "website":    "Live scraped website content",
}

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
        print(f"[Qdrant] Client connected → {QDRANT_PATH}")
        # ✅ Register clean shutdown — fixes ImportError on exit
        atexit.register(_close_client)
    return _client


def _close_client():
    """Called automatically on Python shutdown — prevents ImportError."""
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

    for name, description in COLLECTIONS.items():
        if name in existing:
            if recreate:
                client.delete_collection(name)
                print(f"[Qdrant] Deleted: {name}")
            else:
                print(f"[Qdrant] Exists, skipping: {name}")
                continue

        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=EMBED_DIM,
                distance=Distance.COSINE,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=10_000
            ),
        )
        print(f"[Qdrant] ✅ Created: {name} — {description}")


def collection_info():
    client = get_client()
    for name in COLLECTIONS:
        try:
            info = client.get_collection(name)
            print(f"  {name:15s}: {info.points_count} points")
        except Exception:
            print(f"  {name:15s}: NOT FOUND")