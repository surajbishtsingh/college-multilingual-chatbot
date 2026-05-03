# backend/run_scrape.py
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
from urllib.parse import urlparse
from bs4 import BeautifulSoup

print("\n" + "=" * 55)
print("  Diksha Website Scraper — FIXED VERSION")
print("=" * 55)

# CONFIG
BASE_URL  = os.getenv("COLLEGE_WEBSITE_URL", "https://gbpiet.ac.in")
MAX_PAGES = 50

# ✅ Skip these URL patterns — images, wp-content, tenders, sitemaps
SKIP_PATTERNS = [
    "/wp-content/",       # images, uploads
    "/wp-admin/",
    "/wp-includes/",
    "/wp-login",
    "sitemap",
    ".jpeg", ".jpg", ".png", ".gif", ".svg", ".webp",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".zip",
    ".css", ".js", ".xml",
    "javascript:", "mailto:", "tel:",
    "/feed/", "/tag/", "/author/",
    "/tender",            # tender notices — not useful for chatbot
    "/revised-order",     # procurement notices
    "/provisional-result",# result notices (change often)
    "?share=", "?replytocom",
    "#",
]

# ✅ Priority pages — most important content for students
PRIORITY_PAGES = [
    "",                                           # home page
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
    "/facilities",
    "/facilities/computer-centre",
    "/facilities/central-workshop",
    "/health-centre",
    "/sports-complex",
    "/bank-atms",
    "/transport-service",
    "/student-life",
    "/student-life/student-activity-cell",
    "/student-life/flim-and-music",
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
    "/key-documents/minutes-of-board-of-governors",
]


def should_skip(url: str) -> bool:
    """Return True if URL should be skipped."""
    url_lower = url.lower()
    for pattern in SKIP_PATTERNS:
        if pattern in url_lower:
            return True
    return False

def detect_category(url: str) -> str:
    """Categorize page based on URL path."""
    url_lower = url.lower()
    if any(x in url_lower for x in ["/admission", "/courses", "/undergraduate", "/postgraduate", "/doctoral"]):
        return "admissions"
    elif any(x in url_lower for x in ["/fee", "/fees"]):
        return "fees"
    elif any(x in url_lower for x in ["/placement", "/training", "/recruitment", "/campus-drives"]):
        return "placement"
    elif any(x in url_lower for x in ["/department"]):
        return "departments"
    elif any(x in url_lower for x in ["/facility", "/facilities", "/health", "/sports", "/transport", "/bank"]):
        return "facilities"
    elif any(x in url_lower for x in ["/about", "/vision", "/mission", "/director", "/history"]):
        return "about"
    elif any(x in url_lower for x in ["/administration", "/governing", "/board", "/registrar"]):
        return "administration"
    elif any(x in url_lower for x in ["/student", "/activity", "/club"]):
        return "student_life"
    elif any(x in url_lower for x in ["/academic", "/calendar", "/result", "/syllabus", "/rules"]):
        return "academics"
    elif any(x in url_lower for x in ["/contact", "/reach", "/location"]):
        return "contact"
    else:
        return "general"


def is_same_domain(url: str) -> bool:
    base_domain = urlparse(BASE_URL).netloc
    parsed      = urlparse(url)
    return not parsed.netloc or parsed.netloc == base_domain


# ── Setup ───────────────────────────────────────────────────────────────
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

# ── Get already-indexed hashes (to skip duplicates) ────────────────────
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

# ── Start Chrome/Selenium ───────────────────────────────────────────────
print("[Setup] Starting Chrome driver...")
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

options = Options()
options.add_argument("--headless=new")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0")
options.add_argument("--disable-blink-features=AutomationControlled")
# Block images and fonts — faster loading
prefs = {
    "profile.managed_default_content_settings.images": 2,
    "profile.managed_default_content_settings.fonts":  2,
}
options.add_experimental_option("prefs", prefs)
options.add_experimental_option("excludeSwitches", ["enable-automation"])

service = Service(ChromeDriverManager().install())
driver  = webdriver.Chrome(service=service, options=options)
driver.set_page_load_timeout(30)


def fetch_page(url: str) -> tuple[str, str]:
    """
    Returns (html, body_text) tuple.
    body_text is the rendered text from Selenium directly.
    """
    try:
        driver.get(url)

        # Wait for React to render
        import time
        time.sleep(4)

        # ✅ Get both the HTML and the already-rendered text
        html      = driver.page_source
        body_text = driver.find_element(By.TAG_NAME, "body").text

        print(f"  [Fetch] Got {len(html)} chars HTML, "
              f"{len(body_text)} chars text from {url}")

        return html, body_text

    except Exception as e:
        print(f"  [Fetch] Error: {e}")
        return "", ""

def parse_html(html: str, url: str) -> str:
    """
    Extract clean text from HTML.
    Uses Selenium's body.text directly — most reliable for JS sites.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")

        # ── Only remove clearly useless tags ─────────────────────
        # DO NOT remove by class — too aggressive for React sites
        for tag in ["script", "style", "noscript",
                    "iframe", "svg", "canvas", "head"]:
            for el in soup.find_all(tag):
                el.decompose()

        # ── Get all text from body ────────────────────────────────
        body = soup.find("body")
        if not body:
            return ""

        # Get raw text
        raw = body.get_text(separator="\n", strip=True)

        # ── Clean up ──────────────────────────────────────────────
        import re
        lines = []
        for line in raw.split("\n"):
            line = line.strip()
            # Skip very short lines (menu items, single words)
            if len(line) > 5:
                lines.append(line)

        text = "\n".join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)

        return text.strip()

    except Exception as e:
        print(f"  [Parse] Error: {e}")
        return ""

# ══════════════════════════════════════════════════════════════════════
print(f"\n[1/4] Crawling + Parsing (max {MAX_PAGES} pages)...")
print("-" * 55)

visited      = set()
all_chunks   = []
pages_done   = 0

for path in PRIORITY_PAGES:
    if pages_done >= MAX_PAGES:
        break

    url = BASE_URL.rstrip("/") + path

    # ✅ Skip if matches any bad pattern
    if should_skip(url):
        print(f"\n[SKIP] {url}")
        continue

    if url in visited:
        continue
    visited.add(url)

    pages_done += 1
    print(f"\n[{pages_done}/{MAX_PAGES}] {url}")

    # Fetch
    # Fetch
    result = fetch_page(url)
    if not result or not result[0]:
        print(f"  [Skip] No content returned")
        continue

    html, body_text = result

# Use body_text directly if it has content (already rendered by Selenium)
    if len(body_text) > 100:
        text = body_text
    else:
        text = parse_html(html, url)
    print(f"  [Parse] Extracted {len(text)} chars of clean text")

    if len(text) < 100:
        print(f"  [Skip] Too short after parsing: {len(text)} chars")
        continue

    print(f"  [OK] {len(text)} chars extracted")

    # Get page title
    # Get page title  
    try:
        soup  = BeautifulSoup(html, "html.parser")   # ← html is now a string ✅
        title = ""
        if soup.title:
            title = soup.title.get_text(strip=True)
            title = title.split("|")[0].strip()
        if not title and soup.find("h1"):
            title = soup.find("h1").get_text(strip=True)
        if not title:
            title = url.split("/")[-1].replace("-", " ").title()
    except Exception:
        title = "GBPIET Page"

    # Content hash for dedup
    content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()

    # Chunk the text
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

        # Skip if already indexed
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

    print(f"  [Chunk] {chunk_count} new chunks created")
    time.sleep(1)

# Close browser
driver.quit()
print("\n[Crawler] Chrome closed")

# ══════════════════════════════════════════════════════════════════════
print(f"\n[2/4] Summary:")
print(f"  Pages crawled:  {pages_done}")
print(f"  Total chunks:   {len(all_chunks)}")

# ══════════════════════════════════════════════════════════════════════
print(f"\n[3/4] Indexing into Qdrant...")

if not all_chunks:
    print("  Nothing new to index")
else:
    # Embed in small batches
    texts       = [c["text"] for c in all_chunks]
    all_vectors = []
    batch_size  = 16

    for i in range(0, len(texts), batch_size):
        batch   = texts[i:i + batch_size]
        vectors = embed_model.embed_documents(batch)
        all_vectors.extend(vectors)

    # Build Qdrant points
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

    # Upsert in batches
    upsert_batch = 50
    total_done   = 0
    for start in range(0, len(points), upsert_batch):
        batch = points[start:start + upsert_batch]
        client.upsert(collection_name=COLLECTION, points=batch)
        total_done += len(batch)

    print(f"  [Qdrant] Indexed {total_done} new chunks")

# ══════════════════════════════════════════════════════════════════════
final_info = client.get_collection(COLLECTION)
print(f"\n[4/4] Final Qdrant count: {final_info.points_count} points")

print("\n" + "=" * 55)
print(f"✅ Done! {len(all_chunks)} new chunks indexed from {pages_done} pages")
print("=" * 55)