# backend/rag/internet_search.py
import os
import httpx
from dotenv import load_dotenv

load_dotenv()

SERPAPI_KEY    = os.getenv("SERPAPI_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID  = os.getenv("GOOGLE_CSE_ID", "")
ENABLE_SEARCH  = os.getenv("ENABLE_INTERNET_SEARCH", "true").lower() == "true"
COLLEGE_NAME   = "GBPIET Pauri Garhwal Uttarakhand"
COLLEGE_SITE   = "gbpiet.ac.in"


# ══════════════════════════════════════════════════════
# METHOD 1 — Google Custom Search (100/day free)
# ══════════════════════════════════════════════════════
def search_google_cse(query: str, num: int = 5) -> list[dict]:
    if not GOOGLE_API_KEY or not GOOGLE_CSE_ID:
        return []
    try:
        params = {
            "key": GOOGLE_API_KEY,
            "cx":  GOOGLE_CSE_ID,
            "q":   f"{query} {COLLEGE_NAME}",
            "num": min(num, 10),
            "gl":  "in",
            "hl":  "en",
        }
        with httpx.Client(timeout=10) as client:
            r = client.get(
                "https://www.googleapis.com/customsearch/v1",
                params=params
            )
            r.raise_for_status()
            data = r.json()

        results = []
        for item in data.get("items", [])[:num]:
            snippet = item.get("snippet", "")
            title   = item.get("title", "")
            url     = item.get("link", "")
            if snippet and url:
                results.append({
                    "title":   title,
                    "snippet": snippet,
                    "url":     url,
                    "source":  "google",
                })
        print(f"[Google CSE] {len(results)} results for: {query[:40]}")
        return results
    except Exception as e:
        print(f"[Google CSE] Failed: {e}")
        return []


# ══════════════════════════════════════════════════════
# METHOD 2 — SerpApi (100/month free)
# ══════════════════════════════════════════════════════
def search_serpapi(query: str, num: int = 5) -> list[dict]:
    if not SERPAPI_KEY:
        return []
    try:
        params = {
            "api_key": SERPAPI_KEY,
            "engine":  "google",
            "q":       f"{query} {COLLEGE_NAME}",
            "num":     num,
            "hl":      "en",
            "gl":      "in",
        }
        with httpx.Client(timeout=10) as client:
            r = client.get("https://serpapi.com/search", params=params)
            r.raise_for_status()
            data = r.json()

        results = []
        for item in data.get("organic_results", [])[:num]:
            snippet = item.get("snippet", "")
            title   = item.get("title", "")
            url     = item.get("link", "")
            if snippet and url:
                results.append({
                    "title":   title,
                    "snippet": snippet,
                    "url":     url,
                    "source":  "serpapi",
                })
        print(f"[SerpApi] {len(results)} results for: {query[:40]}")
        return results
    except Exception as e:
        print(f"[SerpApi] Failed: {e}")
        return []


# ══════════════════════════════════════════════════════
# METHOD 3 — DuckDuckGo via duckduckgo-search library
# (already in requirements.txt — no API key needed)
# ══════════════════════════════════════════════════════
def search_duckduckgo(query: str, num: int = 5) -> list[dict]:
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            search_results = list(ddgs.text(
                keywords=f"{query} {COLLEGE_NAME}",
                region="in-en",
                safesearch="off",
                max_results=num,
            ))

        for item in search_results:
            title   = item.get("title", "")
            snippet = item.get("body", "")
            url     = item.get("href", "")
            if snippet and url:
                results.append({
                    "title":   title,
                    "snippet": snippet,
                    "url":     url,
                    "source":  "duckduckgo",
                })

        print(f"[DuckDuckGo] {len(results)} results for: {query[:40]}")
        return results

    except Exception as e:
        print(f"[DuckDuckGo] Failed: {e}")
        return []


def search_duckduckgo_site(query: str, num: int = 5) -> list[dict]:
    """Search specifically within gbpiet.ac.in using DDG."""
    try:
        from duckduckgo_search import DDGS

        results = []
        with DDGS() as ddgs:
            search_results = list(ddgs.text(
                keywords=f"site:{COLLEGE_SITE} {query}",
                region="in-en",
                safesearch="off",
                max_results=num,
            ))

        for item in search_results:
            title   = item.get("title", "")
            snippet = item.get("body", "")
            url     = item.get("href", "")
            if snippet and url:
                results.append({
                    "title":   title,
                    "snippet": snippet,
                    "url":     url,
                    "source":  "college_website",
                })

        print(f"[DuckDuckGo Site] {len(results)} results for: {query[:40]}")
        return results

    except Exception as e:
        print(f"[DuckDuckGo Site] Failed: {e}")
        return []


# ══════════════════════════════════════════════════════
# MAIN FUNCTIONS — used by kb_query.py
# ══════════════════════════════════════════════════════
def search_internet(query: str, num_results: int = 5) -> list[dict]:
    """
    Auto-selects best available search method.
    Priority: Google CSE → SerpApi → DuckDuckGo
    """
    if not ENABLE_SEARCH:
        return []

    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        results = search_google_cse(query, num_results)
        if results:
            return results

    if SERPAPI_KEY:
        results = search_serpapi(query, num_results)
        if results:
            return results

    # DuckDuckGo always works as fallback
    return search_duckduckgo(query, num_results)


def search_college_website(query: str) -> list[dict]:
    """
    Search specifically within gbpiet.ac.in
    Imported by kb_query.py
    Priority: Google CSE → SerpApi → DuckDuckGo site search
    """
    if not ENABLE_SEARCH:
        return []

    # ── Google CSE with site restriction ─────────────────
    if GOOGLE_API_KEY and GOOGLE_CSE_ID:
        try:
            params = {
                "key": GOOGLE_API_KEY,
                "cx":  GOOGLE_CSE_ID,
                "q":   f"site:{COLLEGE_SITE} {query}",
                "num": 5,
                "gl":  "in",
                "hl":  "en",
            }
            with httpx.Client(timeout=10) as client:
                r = client.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params=params
                )
                r.raise_for_status()
                data = r.json()

            results = []
            for item in data.get("items", [])[:5]:
                snippet = item.get("snippet", "")
                title   = item.get("title", "")
                url     = item.get("link", "")
                if snippet and url:
                    results.append({
                        "title":   title,
                        "snippet": snippet,
                        "url":     url,
                        "source":  "college_website",
                    })
            if results:
                print(f"[CollegeSite] {len(results)} results via Google CSE")
                return results
        except Exception as e:
            print(f"[CollegeSite] Google CSE failed: {e}")

    # ── SerpApi with site: operator ───────────────────────
    if SERPAPI_KEY:
        try:
            params = {
                "api_key": SERPAPI_KEY,
                "engine":  "google",
                "q":       f"site:{COLLEGE_SITE} {query}",
                "num":     5,
                "hl":      "en",
                "gl":      "in",
            }
            with httpx.Client(timeout=10) as client:
                r = client.get("https://serpapi.com/search", params=params)
                r.raise_for_status()
                data = r.json()

            results = []
            for item in data.get("organic_results", [])[:5]:
                snippet = item.get("snippet", "")
                url     = item.get("link", "")
                title   = item.get("title", "")
                if snippet:
                    results.append({
                        "title":   title,
                        "snippet": snippet,
                        "url":     url,
                        "source":  "college_website",
                    })
            if results:
                print(f"[CollegeSite] {len(results)} results via SerpApi")
                return results
        except Exception as e:
            print(f"[CollegeSite] SerpApi failed: {e}")

    # ── DuckDuckGo site search fallback ───────────────────
    results = search_duckduckgo_site(query, num=5)
    if results:
        return results

    # ── General DuckDuckGo with college name ──────────────
    return search_duckduckgo(query, num=5)


def format_internet_context(results: list[dict]) -> str:
    """Format search results as context string for LLM."""
    if not results:
        return ""
    parts = ["[Web Search Results]"]
    for i, r in enumerate(results, 1):
        parts.append(
            f"\n{i}. {r['title']}\n"
            f"   {r['snippet']}\n"
            f"   Source: {r['url']}"
        )
    return "\n".join(parts)


def get_search_status() -> dict:
    return {
        "google_cse":    "✅ active" if (GOOGLE_API_KEY and GOOGLE_CSE_ID) else "❌ no key",
        "serpapi":       "✅ active" if SERPAPI_KEY else "❌ no key",
        "duckduckgo":    "✅ always available (duckduckgo-search library)",
        "active_method": (
            "Google CSE" if (GOOGLE_API_KEY and GOOGLE_CSE_ID) else
            "SerpApi"    if SERPAPI_KEY else
            "DuckDuckGo"
        ),
    }


# ── Test ───────────────────────────────────────────────
if __name__ == "__main__":
    print("Search status:")
    for k, v in get_search_status().items():
        print(f"  {k}: {v}")

    print("\nTesting search_college_website('MCA admission')...")
    results = search_college_website("MCA admission")
    for r in results:
        print(f"\n  Title:   {r['title']}")
        print(f"  Snippet: {r['snippet'][:120]}")
        print(f"  URL:     {r['url']}")
