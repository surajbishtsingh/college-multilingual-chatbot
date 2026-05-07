# qdrant_setup.py — Qdrant Cloud + payload indexes
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams,
    PayloadSchemaType
)

load_dotenv()

QDRANT_URL     = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_LOCAL   = os.path.join(os.path.dirname(__file__), "qdrant_storage")

VECTOR_SIZE = 384

COLLECTIONS = {
    "faq":        "gbpiet_faq",
    "kb_en":      "gbpiet_kb_en",
    "kb_hi":      "gbpiet_kb_hi",
    "website":    "gbpiet_web",
    "hostel":     "gbpiet_hostel",
    "fees":       "gbpiet_fees",
    "admissions": "gbpiet_admissions",
}

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is not None:
        return _client

    if QDRANT_URL and QDRANT_API_KEY:
        _client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=30,
        )
        print(f"[Qdrant] Cloud → {QDRANT_URL[:40]}...")
    elif QDRANT_URL:
        _client = QdrantClient(url=QDRANT_URL, timeout=30)
        print(f"[Qdrant] Self-hosted → {QDRANT_URL}")
    else:
        os.makedirs(QDRANT_LOCAL, exist_ok=True)
        _client = QdrantClient(path=QDRANT_LOCAL)
        print(f"[Qdrant] Local → {QDRANT_LOCAL}")

    _ensure_collections(_client)
    return _client


def _ensure_collections(client: QdrantClient):
    existing = {c.name for c in client.get_collections().collections}

    for key, name in COLLECTIONS.items():
        if name not in existing:
            # Create collection
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            print(f"[Qdrant] Created: {name}")

        # ── CREATE PAYLOAD INDEXES ─────────────────────────────
        # Fix: "Index required but not found for language"
        _ensure_indexes(client, name)

        count = client.get_collection(name).points_count
        print(f"[Qdrant] {name}: {count} points")


def _ensure_indexes(client: QdrantClient, collection_name: str):
    """
    Create payload indexes for filterable fields.
    WHY: Qdrant requires index before filtering on a field.
    Without index → 400 Bad Request error.
    """
    indexes_needed = [
        ("language", PayloadSchemaType.KEYWORD),
        ("category", PayloadSchemaType.KEYWORD),
        ("source",   PayloadSchemaType.KEYWORD),
        ("lang",     PayloadSchemaType.KEYWORD),
    ]

    for field_name, field_type in indexes_needed:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_type,
            )
            print(f"[Qdrant] Index created: {collection_name}.{field_name}")
        except Exception as e:
            err = str(e).lower()
            # Index already exists — ignore
            if "already exists" in err or "conflict" in err or "409" in err:
                pass
            else:
                print(f"[Qdrant] Index warning ({field_name}): {e}")
