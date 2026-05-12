# backend/run_scrape.py — Railway compatible (no Chrome needed)
import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR  = os.path.join(THIS_DIR, "rag")

sys.path.insert(0, THIS_DIR)
sys.path.insert(0, RAG_DIR)

from dotenv import load_dotenv
load_dotenv()

import time
import uuid
import hashlib
import requests
import re
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup

print("\n" + "=" * 55)
print("  Diksha Website Scraper — Railway Edition")
print("=" * 55)

# CONFIG
BASE_URL  = os.getenv("COLLEGE_WEBSITE_URL", "https://gbpiet.ac.in")
MAX_PAGES = 50

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
}

SKIP_PATTERNS = [
    "/wp-content/", "/wp-admin/", "/wp-includes/", "/wp-login",
    "sitemap", ".jpeg", ".jpg", ".png", ".gif", ".svg", ".webp",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".zip",
    ".css", ".js", ".xml", "javascript:", "mailto:", "tel:",
    "/feed/", "/tag/", "/author/", "/tender", "/revised-order",
    "/provisional-result", "?share=", "?replytocom", "#",
]

PRIORITY_PAGES = [
    "",
    "/admission",
    "/prospective-students/courses-offered",
    "/academic-programmes/undergraduate",
    "/academic-programmes/postgraduate",
    "/academic-programmes/doctoral",
    "/fee-structure",
    "/training-and-placement-centre",
    "/placement-records",
    "/recruitment-process",
    "/campus-drives",
    "/departments",
    "/departments/computer-science-engineering",
    "/departments/electronics-and-communication-engineering",
    "/departments/electrical-engineering",
    "/departments/mechanical-engineering",
    "/departments/civil-engineering",
    "/departments/biotechnology",
    "/departments/computer-science-applications",
    "/departments/applied-sciences-and-humanities",
    "/about",
    "/about/directors-message",
    "/about/vision-and-mission",
    "/about/how-to-reach",
    "/administration/governing-council",
    "/administration/board-of-governors",
    "/administration/office-of-the-registrar",
    "/administration/deans-associate-deans",
    "/facilities",
    "/facilities/computer-centre",
    "/facilities/central-workshop",
    "/health-centre",
    "/sports-complex",
    "/bank-atms",
    "/transport-service",
    "/student-life",
    "/student-life/student-activity-cell",
    "/academic-calendar",
    "/academic-information",
    "/rules-and-regulations",
    "/conduct-rules",
    "/result",
    "/nirf",
    "/contact-us",
    "/mous",
    "/rti-gbpiet",
    "/guidelines-for-anti-ragging-undertaken",
]


def should_skip(url: str) -> bool:
    url_lower = url.lower()
    for pattern in SKIP_PATTERNS:
        if pattern in url_lower:
            return True
    return False


def detect_category(url: str) -> str:
    url_lower = url.lower()
    if any(x in url_lower for x in ["/admission", "/courses", "/undergraduate", "/postgraduate", "/doctoral"]):
        return "admissions"
    elif any(x in url_lower for x in ["/fee"]):
        return "fees"
    elif any(x in url_lower for x in ["/placement", "/training", "/recruitment", "/campus-drives"]):
        return "placement"
    elif any(x in url_lower for x in ["/department"]):
        return "departments"
    elif any(x in url_lower for x in ["/facility", "/facilities", "/health", "/sports", "/transport", "/bank"]):
        return "facilities"
    elif any(x in url_lower for x in ["/about", "/vision", "/mission", "/director"]):
        return "about"
    elif any(x in url_lower for x in ["/administration", "/governing", "/board", "/registrar", "/deans"]):
        return "administration"
    elif any(x in url_lower for x in ["/student", "/activity"]):
        return "student_life"
    elif any(x in url_lower for x in ["/academic", "/calendar", "/result", "/rules"]):
        return "academics"
    elif any(x in url_lower for x in ["/contact", "/reach"]):
        return "contact"
    else:
        return "general"


def fetch_page(url: str) -> tuple[str, str]:
    """Fetch page using requests — no Chrome needed."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            print(f"  [Fetch] HTTP {r.status_code} for {url}")
            return "", ""

        html = r.text
        soup = BeautifulSoup(html, "html.parser")

        # Remove useless tags
        for tag in ["script", "style", "noscript", "iframe", "svg", "head", "nav", "footer"]:
            for el in soup.find_all(tag):
                el.decompose()

        body = soup.find("body")
        if not body:
            return html, ""

        # Clean text
        raw = body.get_text(separator="\n", strip=True)
        lines = [l.strip() for l in raw.split("\n") if len(l.strip()) > 5]
        text = "\n".join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        print(f"  [Fetch] {len(html)} chars HTML, {len(text)} chars text from {url}")
        return html, text.strip()

    except Exception as e:
        print(f"  [Fetch] Error: {e}")
        return "", ""


# ── Setup ────────────────────────────────────────────────────────────
print("\n[Setup] Loading embeddings model...")
from rag.embeddings import get_embed_model
embed_model = get_embed_model()
test_vec    = embed_model.embed_query("test")
VECTOR_SIZE = len(test_vec)
print(f"[Setup] Vector size: {VECTOR_SIZE}")

print("[Setup] Connecting to Qdrant...")
from qdrant_setup import get_client
from qdrant_client.models import Distance, VectorParams, PointStruct

client     = get_client()
COLLECTION = "website"

existing_collections = {c.name for c in client.get_collections().collections}
if COLLECTION not in existing_collections:
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"  [Qdrant] Created '{COLLECTION}' collection")
else:
    info = client.get_collection(COLLECTION)
    print(f"  [Qdrant] Collection exists with {info.points_count} points")

# ── Get existing hashes ───────────────────────────────────────────────
print("[Setup] Loading existing content hashes...")
existing_hashes = set()
try:
    offset = None
    while True:
        result, offset = client.scroll(
            collection_name=COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=["content_hash"],
            with_vectors=False,
        )
        for point in result:
            h = point.payload.get("content_hash", "")
            if h:
                existing_hashes.add(h)
        if offset is None:
            break
    print(f"  [Qdrant] {len(existing_hashes)} chunks already indexed (will skip)")
except Exception:
    pass

# ── Crawl ─────────────────────────────────────────────────────────────
print(f"\n[1/4] Crawling (max {MAX_PAGES} pages)...")
print("-" * 55)

visited    = set()
all_chunks = []
pages_done = 0

for path in PRIORITY_PAGES:
    if pages_done >= MAX_PAGES:
        break

    url = BASE_URL.rstrip("/") + path

    if should_skip(url):
        continue

    if url in visited:
        continue
    visited.add(url)

    pages_done += 1
    print(f"\n[{pages_done}/{MAX_PAGES}] {url}")

    html, text = fetch_page(url)

    if len(text) < 100:
        print(f"  [Skip] Too short: {len(text)} chars")
        continue

    print(f"  [OK] {len(text)} chars extracted")

    # Title
    try:
        soup  = BeautifulSoup(html, "html.parser")
        title = ""
        if soup.title:
            title = soup.title.get_text(strip=True).split("|")[0].strip()
        if not title and soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        if not title:
            title = path.strip("/").replace("-", " ").title() or "GBPIET"
    except Exception:
        title = "GBPIET Page"

    # Hash + chunk
    content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks   = splitter.split_text(text)
    category = detect_category(url)

    chunk_count = 0
    for i, chunk_text in enumerate(chunks):
        chunk_text = chunk_text.strip()
        if len(chunk_text) < 50:
            continue

        chunk_hash = content_hash + f"_{i}"
        if chunk_hash in existing_hashes:
            continue

        all_chunks.append({
            "text":         chunk_text,
            "url":          url,
            "title":        title,
            "category":     category,
            "content_hash": chunk_hash,
            "source":       "website",
            "language":     "en",
        })
        chunk_count += 1

    print(f"  [Chunk] {chunk_count} new chunks")
    time.sleep(0.5)

# ── Summary ───────────────────────────────────────────────────────────
print(f"\n[2/4] Summary:")
print(f"  Pages crawled : {pages_done}")
print(f"  Total chunks  : {len(all_chunks)}")

# ── Index ─────────────────────────────────────────────────────────────
print(f"\n[3/4] Indexing into Qdrant...")

if not all_chunks:
    print("  Nothing new to index")
else:
    texts       = [c["text"] for c in all_chunks]
    all_vectors = []
    batch_size  = 16

    for i in range(0, len(texts), batch_size):
        batch   = texts[i:i + batch_size]
        vectors = embed_model.embed_documents(batch)
        all_vectors.extend(vectors)
        print(f"  [Embed] {min(i+batch_size, len(texts))}/{len(texts)} done")

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=all_vectors[i],
            payload={
                "text":         all_chunks[i]["text"],
                "url":          all_chunks[i]["url"],
                "title":        all_chunks[i]["title"],
                "category":     all_chunks[i]["category"],
                "content_hash": all_chunks[i]["content_hash"],
                "source":       "website",
                "language":     "en",
            }
        )
        for i in range(len(all_chunks))
    ]

    total_done = 0
    for start in range(0, len(points), 50):
        batch = points[start:start + 50]
        client.upsert(collection_name=COLLECTION, points=batch)
        total_done += len(batch)
        print(f"  [Qdrant] Upserted {total_done}/{len(points)}")

    print(f"  [Qdrant] Done — {total_done} chunks indexed")

# ── Final count ───────────────────────────────────────────────────────
final_info = client.get_collection(COLLECTION)
print(f"\n[4/4] Final Qdrant count: {final_info.points_count} points")

print("\n" + "=" * 55)
print(f"Done! {len(all_chunks)} new chunks from {pages_done} pages")
print("=" * 55)
