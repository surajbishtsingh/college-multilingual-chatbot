# backend/scraper/crawl_site.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import hashlib
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

BASE_URL      = os.getenv("COLLEGE_WEBSITE_URL", "https://gbpiet.ac.in")
MAX_PAGES     = int(os.getenv("MAX_PAGES_TO_CRAWL", "100"))
REQUEST_DELAY = 2.0

SKIP_PATTERNS = [
    # ✅ File types to skip
    "/wp-content/",    # WordPress uploads — images, docs
    "/wp-admin/",
    "/wp-includes/",
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
    ".css", ".js", ".xml", ".zip", ".doc", ".docx",
    # ✅ URL patterns to skip
    "/feed/", "/tag/", "/author/",
    "javascript:", "mailto:", "tel:",
    "#", "?replytocom", "?share=",
    "sitemap",
    # ✅ Specific pages with no useful content
    "/wp-login",
]
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,hi;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
}


def should_crawl(url: str) -> bool:
    if not url.startswith("http"):
        return False
    parsed      = urlparse(url)
    base_domain = urlparse(BASE_URL).netloc
    if parsed.netloc and parsed.netloc != base_domain:
        return False
    for pattern in SKIP_PATTERNS:
        if pattern in url.lower():
            return False
    return True


# ── Check if site needs JavaScript ────────────────────────────────────
def needs_javascript(url: str) -> bool:
    return False  # Railway pe Selenium nahi chalta — hamesha requests use karo

# ── Selenium fetcher ───────────────────────────────────────────────────
def fetch_with_selenium(url: str) -> dict | None:
    """
    Fetch JavaScript-rendered page using Selenium.
    Railway ke liye chromium system path se use karta hai.
    """
    try:
        import shutil
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        options.add_argument(f"--user-agent={HEADERS['User-Agent']}")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-infobars")
        options.add_argument("--remote-debugging-port=9222")

        # ✅ Railway ke liye — system chromium use karo (webdriver-manager nahi)
        chrome_path = (
            shutil.which("chromium") or
            shutil.which("chromium-browser") or
            shutil.which("google-chrome") or
            "/usr/bin/chromium"
        )
        chromedriver_path = (
            shutil.which("chromedriver") or
            "/usr/bin/chromedriver"
        )

        print(f"  [Selenium] Chrome: {chrome_path}")
        print(f"  [Selenium] Driver: {chromedriver_path}")

        options.binary_location = chrome_path
        service = Service(chromedriver_path)

        # Disable images — faster loading
        prefs = {"profile.managed_default_content_settings.images": 2}
        options.add_experimental_option("prefs", prefs)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])

        driver = webdriver.Chrome(service=service, options=options)
        driver.set_page_load_timeout(30)

        try:
            driver.get(url)

            WebDriverWait(driver, 15).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            time.sleep(3)

            driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            time.sleep(1)
            driver.execute_script("window.scrollTo(0, 0);")
            time.sleep(1)

            html      = driver.page_source
            body_text = driver.find_element(By.TAG_NAME, "body").text

            print(f"  [Selenium] Got {len(body_text)} chars from {url[:50]}")

            if len(body_text) < 200:
                time.sleep(4)
                html      = driver.page_source
                body_text = driver.find_element(By.TAG_NAME, "body").text
                print(f"  [Selenium] After extra wait: {len(body_text)} chars")

        finally:
            driver.quit()

        return {
            "url":          url,
            "html":         html,
            "status_code":  200,
            "content_type": "text/html",
        }

    except Exception as e:
        print(f"  [Selenium] Error for {url}: {e}")
        return None

# ── Regular requests fetcher ───────────────────────────────────────────
def fetch_with_requests(url: str) -> dict | None:
    """Fetch a page using regular requests (faster, no JS)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        return {
            "url":          url,
            "html":         r.text,
            "status_code":  r.status_code,
            "content_type": r.headers.get("content-type", ""),
        }
    except Exception as e:
        print(f"[Crawler] Failed to fetch {url}: {e}")
        return None


# ── Smart fetcher — tries requests first, Selenium if empty ───────────
def fetch_page(url: str, use_selenium: bool = False) -> dict | None:
    """
    Fetch a page — uses Selenium if site needs JavaScript rendering.
    """
    if use_selenium:
        print(f"  [JS] Using Selenium for: {url[:60]}")
        return fetch_with_selenium(url)
    else:
        return fetch_with_requests(url)


def extract_links(html: str, base_url: str) -> list[str]:
    soup  = BeautifulSoup(html, "html.parser")
    links = []
    for tag in soup.find_all("a", href=True):
        href     = tag["href"].strip()
        full_url = urljoin(base_url, href)
        if should_crawl(full_url):
            full_url = full_url.split("#")[0].rstrip("/")
            links.append(full_url)
    return list(set(links))


def get_urls_from_sitemap(base_url: str) -> list[str]:
    """Read sitemap.xml to get all page URLs."""
    sitemap_urls = [
        f"{base_url}/sitemap.xml",
        f"{base_url}/sitemap_index.xml",
        f"{base_url}/page-sitemap.xml",
    ]
    urls    = []
    headers = {"User-Agent": HEADERS["User-Agent"]}

    for sitemap_url in sitemap_urls:
        try:
            r = requests.get(sitemap_url, headers=headers, timeout=10)
            if r.status_code != 200:
                continue

            soup = BeautifulSoup(r.content, "lxml-xml")

            # Sitemap index → multiple sitemaps
            for sm in soup.find_all("sitemap"):
                loc = sm.find("loc")
                if loc:
                    sub = get_urls_from_sitemap(loc.text.strip())
                    urls.extend(sub)

            # Regular sitemap URLs
            for loc in soup.find_all("loc"):
                url = loc.text.strip()
                if should_crawl(url):
                    urls.append(url)

            if urls:
                print(f"[Sitemap] Found {len(urls)} URLs")
                break

        except Exception as e:
            print(f"[Sitemap] {sitemap_url}: {e}")
            continue

    return list(set(urls))


# ── Priority URLs — always crawl these ────────────────────────────────
PRIORITY_PAGES = [
    # Admissions
    "/admission",
    "/prospective-students/courses-offered",
    "/academic-programmes/undergraduate",
    "/academic-programmes/postgraduate",
    "/academic-programmes/doctoral",

    # Fees
    "/fee-structure",
    "/prospective-students/fee-structure",

    # Placement
    "/training-and-placement-centre",
    "/placement-records",
    "/recruitment-process",
    "/placement-team",
    "/campus-drives",
    "/gate-records",

    # Departments
    "/departments",
    "/departments/computer-science-engineering",
    "/departments/electronics-and-communication-engineering",
    "/departments/electrical-engineering",
    "/departments/mechanical-engineering",
    "/departments/civil-engineering",
    "/departments/biotechnology",
    "/departments/applied-sciences-and-humanities",
    "/departments/computer-science-applications",

    # About
    "/about",
    "/about/directors-message",
    "/about/vision-and-mission",
    "/about/how-to-reach",
    "/about/history",

    # Administration
    "/administration/governing-council",
    "/administration/board-of-governors",
    "/administration/office-of-the-registrar",
    "/administration/deans",

    # Facilities
    "/facilities",
    "/facilities/computer-centre",
    "/facilities/central-workshop",
    "/facilities/library",
    "/health-centre",
    "/sports-complex",
    "/bank-atms",
    "/transport-service",

    # Academic
    "/academic-calendar",
    "/academic-information",
    "/curricula-and-syllabi",
    "/rules-and-regulations",
    "/conduct-rules",
    "/result",
    "/nirf",

    # Student Life
    "/student-life",
    "/student-life/student-activity-cell",
    "/student-life/techno-management",

    # Contact & Info
    "/contact-us",
    "/mous",
    "/rti-gbpiet",
]

def crawl_website(
    start_url: str = BASE_URL,
    max_pages: int = MAX_PAGES,
) -> list[dict]:
    """
    gbpiet.ac.in JS-rendered hai — SerpAPI se data fetch karo.
    """
    print("[Crawler] Using SerpAPI for gbpiet.ac.in (JS site)")
    
    QUERIES = [
        "GBPIET fees BTech MTech MCA",
        "GBPIET admission process JEE",
        "GBPIET hostel facility boys girls",
        "GBPIET placement record companies",
        "GBPIET director faculty staff",
        "GBPIET departments CSE ECE ME Civil",
        "GBPIET contact address phone",
        "GBPIET library transport sports",
        "GBPIET scholarship result exam",
        "GBPIET anti ragging rules regulations",
    ]
    
    pages = []
    serpapi_key = os.getenv("SERPAPI_KEY", "").strip()
    
    if not serpapi_key:
        print("[Crawler] ❌ SERPAPI_KEY not set — skipping")
        return []
    
    for query in QUERIES:
        try:
            params = {
                "q":       f"site:gbpiet.ac.in {query}",
                "api_key": serpapi_key,
                "num":     10,
            }
            r = requests.get(
                "https://serpapi.com/search",
                params=params,
                timeout=15,
            )
            if r.status_code != 200:
                print(f"[Crawler] SerpAPI error {r.status_code} for: {query}")
                continue
                
            results = r.json().get("organic_results", [])
            print(f"[Crawler] '{query}' → {len(results)} results")
            
            for result in results:
                url     = result.get("link", "")
                snippet = result.get("snippet", "")
                title   = result.get("title", "")
                
                if not snippet or len(snippet) < 30:
                    continue
                    
                # HTML format mein wrap karo
                fake_html = f"<html><body><h1>{title}</h1><p>{snippet}</p></body></html>"
                pages.append({
                    "url":          url,
                    "html":         fake_html,
                    "status_code":  200,
                    "content_type": "text/html",
                })
            
            time.sleep(1)  # Rate limit
            
        except Exception as e:
            print(f"[Crawler] SerpAPI failed for '{query}': {e}")
            continue
    
    print(f"[Crawler] ✅ SerpAPI: {len(pages)} snippets collected")
    return pages

if __name__ == "__main__":
    pages = crawl_website(max_pages=5)
    for p in pages:
        soup = BeautifulSoup(p["html"], "html.parser")
        body = soup.find("body")
        text = body.get_text(strip=True)[:200] if body else "EMPTY"
        print(f"\n✅ {p['url']}")
        print(f"   Content preview: {text[:150]}")
