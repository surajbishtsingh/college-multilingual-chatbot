# build_kb.py — Multi-collection Qdrant index builder
import json
import os
import uuid
import shutil
import glob
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client.models import PointStruct
try:
    from backend.qdrant_setup import get_client, ensure_collections, QDRANT_PATH
    import backend.qdrant_setup as qdrant_setup
except ImportError:
    from qdrant_setup import get_client, ensure_collections, QDRANT_PATH
    import qdrant_setup

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── Map filename patterns → collection name ────────────────────────────
# Files matching a pattern go into a specific collection.
FILE_TO_COLLECTION = {
    "admission":   "admissions",
    "fees":        "fees",
    "fee":         "fees",
    "hostel":      "hostel",
    "hods":        "faq",
    "hodshi":      "faq",
    "faculty":     "faq",
    "facultyhi":   "faq",
    "placement":   "faq",
    "placementhi": "faq",
    "campus":      "faq",
    "grievance":   "faq",
    "student":     "faq",
    "general":     "kb_en",
    "generalhi":   "kb_hi",
    "courses":     "kb_en",
    "courseshi":   "kb_hi",
    "administration":   "kb_en",
    "administrationhi": "kb_hi",
}

HINDI_SUFFIXES = ("hi", "hindi", "hin")


def detect_course_type(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ["btech", "b.tech", "undergraduate", "ug", "12th"]):
        return "btech"
    if any(k in t for k in ["mca"]):
        return "mca"
    if any(k in t for k in ["mtech", "m.tech", "pg"]):
        return "mtech"
    return "general"


def filename_to_collection(filename: str) -> tuple[str, str]:
    """
    Returns (collection_name, language) for a given JSON filename.
    Falls back to kb_en / kb_hi based on 'hi' suffix.
    """
    base = filename.replace("faqs_", "").replace("faqs-", "").replace(".json", "").lower()

    # Check explicit mapping first
    for pattern, collection in FILE_TO_COLLECTION.items():
        if pattern in base:
            lang = "hi" if any(base.endswith(s) for s in HINDI_SUFFIXES) else "en"
            return collection, lang

    # Fallback: Hindi suffix → kb_hi, else → kb_en
    if any(base.endswith(s) for s in HINDI_SUFFIXES):
        return "kb_hi", "hi"
    return "kb_en", "en"

def safe_text(value):
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    if value is None:
        return ""
    return str(value)
def load_documents(data_folder: str) -> dict[str, list[Document]]:
    """
    Load all JSON files and bucket Documents by target collection.
    Returns { collection_name: [Document, ...] }
    """
    buckets: dict[str, list[Document]] = {name: [] for name in [
        "kb_en", "kb_hi", "faq", "admissions", "fees", "hostel"
    ]}

    total = 0
    for filepath in sorted(glob.glob(os.path.join(data_folder, "*.json"))):
        filename   = os.path.basename(filepath)
        collection, lang = filename_to_collection(filename)
        cat        = filename.replace("faqs_", "").replace("faqs-", "").replace(".json", "")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data if isinstance(data, list) else [data]
            count = 0

            for item in items:
                if not isinstance(item, dict):
                    continue
                if not item.get("question") or not item.get("answer"):
                    continue

                q = safe_text(item.get("question", "")).strip()
                a = safe_text(item.get("answer", "")).strip()

                # Detect Hindi by character ratio
                hindi_chars = sum(1 for c in q if '\u0900' <= c <= '\u097F')
                is_hindi    = hindi_chars > len(q) * 0.3
                lang_tag    = "hi" if is_hindi else lang
                course_type = detect_course_type(q + " " + a)

                text = (
                    f"Language: {lang_tag}\n"
                    f"Course: {course_type}\n"
                    f"Category: {cat}\n\n"
                    f"Question: {q}\n"
                    f"Answer: {a}"
                )

                doc = Document(
                    page_content=text,
                    metadata={
                        "source":   filename,
                        "category": cat,
                        "language": lang_tag,
                        "course":   course_type,
                        "question": q,
                    }
                )
                buckets[collection].append(doc)
                count += 1

            total += count
            print(f"  {count:3d} entries → {collection:12s} ← {filename}")

        except Exception as e:
            print(f"  ERROR {filename}: {e}")

    print(f"\n  Total: {total} entries across {len(buckets)} collections")
    return buckets


def build_knowledge_base(recreate: bool = True):
    print("=" * 60)
    print("Diksha KB Builder — Multi-Collection Qdrant")
    print("=" * 60)

    data_folder = os.path.join(os.path.dirname(__file__), "data")

    # ── Step 1: Load and bucket documents ─────────────────────────────
    print("\n[1/5] Loading JSON files...")
    buckets = load_documents(data_folder)

    # ── Step 2: Chunk documents ────────────────────────────────────────
    print("\n[2/5] Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunked: dict[str, list] = {}
    for collection, docs in buckets.items():
        if not docs:
            continue
        chunks = splitter.split_documents(docs)
        chunked[collection] = chunks
        print(f"  {collection}: {len(docs)} docs → {len(chunks)} chunks")

    # ── Step 3: Embed all chunks ───────────────────────────────────────
    print("\n[3/5] Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    all_texts = []
    for chunks in chunked.values():
        all_texts.extend([c.page_content for c in chunks])

    print(f"  Embedding {len(all_texts)} total chunks...")
    all_vectors = embeddings.embed_documents(all_texts)
    dim         = len(all_vectors[0])
    print(f"  Embedding dim: {dim}")

    # ── Step 4: Write to temp Qdrant, then swap ────────────────────────
    print("\n[4/5] Building Qdrant collections...")

    temp_path  = os.path.join(os.path.dirname(__file__), "qdrant_storage_temp")
    final_path = QDRANT_PATH

    # Clean any leftover temp
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(temp_path, exist_ok=True)

    # Temporarily point client to temp path
    import qdrant_setup
    qdrant_setup.QDRANT_PATH = temp_path
    qdrant_setup._client     = None   # reset singleton

    ensure_collections(recreate=True)
    client = get_client()

    # Upsert each collection
    vector_idx = 0
    for collection, chunks in chunked.items():
        if not chunks:
            continue

        points = []
        for chunk in chunks:
            vec = all_vectors[vector_idx]
            vector_idx += 1
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    **chunk.metadata,
                    "text": chunk.page_content,
                }
            ))

        # Upsert in batches of 256
        batch_size = 256
        for start in range(0, len(points), batch_size):
            batch = points[start:start + batch_size]
            client.upsert(collection_name=collection, points=batch)

        print(f"  ✅ {collection}: {len(points)} points upserted")

    # Close temp client before swapping
    client.close()
    qdrant_setup._client = None

    # Atomic swap temp → final
    print("\n[5/5] Swapping index into place...")
    if os.path.exists(final_path):
        shutil.rmtree(final_path)
    shutil.move(temp_path, final_path)

    # Restore final path in qdrant_setup
    qdrant_setup.QDRANT_PATH = final_path

    print("\n" + "=" * 60)
    print("✅ Multi-Collection Qdrant KB Built Successfully!")
    print(f"📁 Saved at: {final_path}")
    print("=" * 60)

    # Print collection stats
    qdrant_setup._client = None
    client = get_client()
    print("\nCollection stats:")
    for name in chunked:
        try:
            info = client.get_collection(name)
            print(f"  {name:15s}: {info.points_count} points")
        except Exception:
            print(f"  {name:15s}: not found")


if __name__ == "__main__":
    build_knowledge_base(recreate=True)