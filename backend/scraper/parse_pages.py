# backend/scraper/parse_pages.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import hashlib
from datetime import datetime
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Tags to remove completely
REMOVE_TAGS = [
    "script", "style", "noscript", "nav", "footer",
    "header", "aside", "iframe", "form", "button",
    "meta", "link", "head", "svg", "canvas",
]

# CSS class/id keywords that indicate non-content areas
REMOVE_CLASS_KEYWORDS = [
    "nav", "navigation", "menu", "sidebar", "footer",
    "header", "breadcrumb", "pagination", "social",
    "share", "widget", "advertisement", "cookie",
    "popup", "modal", "banner", "search-bar",
]

# URL → category mapping
URL_CATEGORIES = {
    "/admission":   "admissions",
    "/fee":         "fees",
    "/hostel":      "hostel",
    "/placement":   "placement",
    "/faculty":     "faculty",
    "/department":  "academic",
    "/course":      "courses",
    "/exam":        "examination",
    "/result":      "results",
    "/notice":      "notices",
    "/event":       "events",
    "/research":    "research",
    "/contact":     "contact",
    "/about":       "about",
    "/library":     "library",
    "/sport":       "sports",
    "/scholarship": "scholarship",
    "/academic":    "academic",
    "/administrat": "administration",
}


def detect_category(url: str) -> str:
    url_lower = url.lower()
    for pattern, category in URL_CATEGORIES.items():
        if pattern in url_lower:
            return category
    return "general"


def safe_get_title(soup: BeautifulSoup) -> str:
    """Extract title safely — never crashes."""
    try:
        # Try <title> tag
        if soup.title:
            title_text = soup.title.get_text(strip=True)
            if title_text:
                # Remove site suffix like "| GBPIET"
                title_text = re.sub(r'\s*[|\-–]\s*.{0,30}$', '', title_text).strip()
                if title_text:
                    return title_text

        # Try H1
        h1 = soup.find("h1")
        if h1:
            text = h1.get_text(strip=True)
            if text:
                return text

        # Try og:title meta tag
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"].strip()

    except Exception:
        pass

    return "GBPIET Page"


def remove_clutter(soup: BeautifulSoup) -> BeautifulSoup:
    """Remove navigation, scripts, ads etc."""
    # Remove by tag
    for tag_name in REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Remove by class/id keywords
    for tag in soup.find_all(True):
        try:
            classes = " ".join(tag.get("class") or []).lower()
            tag_id  = (tag.get("id") or "").lower()
            combined = classes + " " + tag_id
            if any(kw in combined for kw in REMOVE_CLASS_KEYWORDS):
                tag.decompose()
        except Exception:
            continue

    return soup


def extract_main_content(soup: BeautifulSoup) -> str:
    """
    Try multiple strategies to find main content.
    Always returns a string (never None).
    """
    # Strategy 1: Semantic HTML5 tags
    for selector in ["main", "article"]:
        el = soup.find(selector)
        if el:
            text = el.get_text(separator="\n", strip=True)
            if len(text) > 150:
                return text

    # Strategy 2: Common content div IDs
    for id_name in ["content", "main-content", "page-content",
                     "main", "primary", "site-content"]:
        el = soup.find(id=id_name)
        if el:
            text = el.get_text(separator="\n", strip=True)
            if len(text) > 150:
                return text

    # Strategy 3: Common content div classes
    for class_name in ["entry-content", "post-content", "page-content",
                        "content-area", "site-content", "main-content",
                        "container", "wrapper", "inner-content"]:
        el = soup.find(class_=class_name)
        if el:
            text = el.get_text(separator="\n", strip=True)
            if len(text) > 150:
                return text

    # Strategy 4: Largest <div> by text length
    divs = soup.find_all("div")
    if divs:
        best_div  = max(divs, key=lambda d: len(d.get_text(strip=True)))
        best_text = best_div.get_text(separator="\n", strip=True)
        if len(best_text) > 150:
            return best_text

    # Strategy 5: Full body text as last resort
    body = soup.find("body")
    if body:
        return body.get_text(separator="\n", strip=True)

    # Strategy 6: Everything
    return soup.get_text(separator="\n", strip=True)


def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""

    # Remove lines that are just whitespace or single chars
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if len(line) > 2:          # skip very short lines
            lines.append(line)

    text = "\n".join(lines)

    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    # Remove weird characters but keep:
    # - Hindi/Devanagari: \u0900-\u097F
    # - Common punctuation
    # - URLs and emails
    text = re.sub(
        r'[^\w\s\u0900-\u097F\.\,\:\;\!\?\-\(\)\[\]\/\%\₹\@\#\+\=]',
        ' ',
        text
    )

    # Final cleanup
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def parse_page(page: dict) -> dict | None:
    """
    Parse a single raw page dict into a clean content dict.

    Input:  { url, html, status_code, content_type }
    Output: { url, title, text, category, timestamp, content_hash, source }
    Returns None if page has no useful content.
    """
    # ── Validate input ────────────────────────────────────────────
    if not isinstance(page, dict):
        return None

    url  = page.get("url", "")
    html = page.get("html", "")

    if not url or not html:
        return None

    # Skip non-HTML content types
    content_type = page.get("content_type", "")
    if content_type and "html" not in content_type.lower():
        return None

    try:
        # ── Parse HTML ────────────────────────────────────────────
        soup = BeautifulSoup(html, "html.parser")

        # ── Extract title BEFORE removing elements ────────────────
        title = safe_get_title(soup)

        # ── Remove clutter ────────────────────────────────────────
        soup = remove_clutter(soup)

        # ── Extract main content ──────────────────────────────────
        raw_text = extract_main_content(soup)

        # ── Clean text ────────────────────────────────────────────
        clean = clean_text(raw_text)

        # ── Skip pages with too little content ────────────────────
        if len(clean) < 100:
            print(f"[Parser] Skipping (too short: {len(clean)} chars): {url}")
            return None

        # ── Build content hash ────────────────────────────────────
        content_hash = hashlib.md5(clean.encode("utf-8")).hexdigest()
        category     = detect_category(url)

        return {
            "url":          url,
            "title":        title,
            "text":         clean,
            "category":     category,
            "timestamp":    datetime.utcnow().isoformat(),
            "content_hash": content_hash,
            "source":       "website",
        }

    except Exception as e:
        print(f"[Parser] Error parsing {url}: {e}")
        return None


def parse_pages(raw_pages: list[dict]) -> list[dict]:
    """Parse a list of raw page dicts → list of clean content dicts."""
    parsed  = []
    skipped = 0

    for page in raw_pages:
        result = parse_page(page)
        if result:
            parsed.append(result)
            print(f"  [Parser] ✅ {result['category']:15s} | "
                  f"{len(result['text']):5d} chars | {result['url']}")
        else:
            skipped += 1

    print(f"[Parser] ✅ Parsed {len(parsed)} pages, skipped {skipped}")
    return parsed


def chunk_pages(
    parsed_pages: list[dict],
    chunk_size:    int = 500,
    chunk_overlap: int = 80,
) -> list[dict]:
    """
    Split long page texts into smaller overlapping chunks for Qdrant.
    Each chunk keeps its parent page metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "। ", " "],
    )

    all_chunks = []

    for page in parsed_pages:
        text   = page.get("text", "")
        if not text:
            continue

        chunks = splitter.split_text(text)

        for i, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 50:
                continue

            all_chunks.append({
                "text":         chunk_text,
                "url":          page["url"],
                "title":        page["title"],
                "category":     page["category"],
                "timestamp":    page["timestamp"],
                "content_hash": page["content_hash"] + f"_{i}",
                "source":       "website",
                "chunk_index":  i,
                "chunk_total":  len(chunks),
            })

    print(f"[Parser] ✅ Created {len(all_chunks)} chunks "
          f"from {len(parsed_pages)} pages")
    return all_chunks


# ── Quick test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import requests

    print("Testing parser on GBPIET admission page...")
    url = "https://gbpiet.ac.in/admission"

    try:
        r    = requests.get(url, timeout=15,
                            headers={"User-Agent": "DikshaChatbot/1.0"})
        page = {
            "url":          url,
            "html":         r.text,
            "status_code":  r.status_code,
            "content_type": r.headers.get("content-type", ""),
        }

        result = parse_page(page)
        if result:
            print(f"✅ Title:    {result['title']}")
            print(f"   Category: {result['category']}")
            print(f"   Length:   {len(result['text'])} chars")
            print(f"   Preview:  {result['text'][:300]}")
        else:
            print("❌ Parse returned None")
            print("   Raw HTML preview:")
            print(r.text[:500])

    except Exception as e:
        print(f"❌ Error: {e}")