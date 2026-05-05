# qdrant_setup.py — Qdrant Cloud + Local fallback
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
QDRANT_URL    = os.getenv("QDRANT_URL", "")       # Cloud URL
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")  # Cloud API key
QDRANT_LOCAL  = os.path.join(os.path.dirname(__file__), "qdrant_storage")

VECTOR_SIZE   = 384   # paraphrase-multilingual-MiniLM-L12-v2

# ── Collection names ──────────────────────────────────────────────────
COLLECTIONS = {
    "faq":      "gbpiet_faq",
    "kb_en":    "gbpiet_kb_en",
    "kb_hi":    "gbpiet_kb_hi",
    "website":  "gbpiet_web",
    "hostel":   "gbpiet_hostel",
    "fees":     "gbpiet_fees",
    "admissions": "gbpiet_admissions",
}

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """
    Get Qdrant client.
    WHY LOCAL FAILS ON RAILWAY:
      Railway filesystem is ephemeral — qdrant_storage/ gets deleted on restart.
      Use Qdrant Cloud (free tier: 1GB, 1 collection) for production.
    """
    global _client
    if _client is not None:
        return _client

    if QDRANT_URL and QDRANT_API_KEY:
        # ── Cloud mode ───────────────────────────────────────────
        _client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30,
        )
        print(f"[Qdrant] Connected to Cloud → {QDRANT_URL[:40]}...")
    elif QDRANT_URL:
        # ── Self-hosted without API key ───────────────────────────
        _client = QdrantClient(url=QDRANT_URL, timeout=30)
        print(f"[Qdrant] Connected to self-hosted → {QDRANT_URL}")
    else:
        # ── Local mode (dev only — NOT for Railway) ───────────────
        os.makedirs(QDRANT_LOCAL, exist_ok=True)
        _client = QdrantClient(path=QDRANT_LOCAL)
        print(f"[Qdrant] Local mode → {QDRANT_LOCAL}")
        print("[Qdrant] ⚠️  WARNING: Local mode not suitable for Railway/production!")

    # Ensure all collections exist
    _ensure_collections(_client)
    return _client


def _ensure_collections(client: QdrantClient):
    """Create collections if they don't exist."""
    existing = {c.name for c in client.get_collections().collections}

    for key, name in COLLECTIONS.items():
        if name not in existing:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[Qdrant] Created collection: {name}")
        else:
            count = client.get_collection(name).points_count
            print(f"[Qdrant] {name}: {count} points")
