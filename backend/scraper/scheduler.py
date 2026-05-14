# scraper/scheduler.py
# Auto-scrapes website daily and updates Qdrant

import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

load_dotenv()

SCRAPE_INTERVAL_HOURS = int(os.getenv("SCRAPE_INTERVAL_HOURS", "24"))
ENABLE_AUTO_SCRAPE    = os.getenv("ENABLE_AUTO_SCRAPE", "true").lower() == "true"

# Track last scrape results
scrape_status = {
    "last_run":      None,
    "pages_scraped": 0,
    "chunks_added":  0,
    "status":        "never_run",
    "errors":        [],
}

_scheduler = None


def index_website_chunks(chunks: list[dict]) -> int:
    try:
        # ✅ Use try/except for both import styles
        try:
            from qdrant_db.qdrant_setup import get_client
        except ImportError:
            from backend.qdrant_db.qdrant_setup import get_client

        try:
            from rag.embeddings import embed_texts
        except ImportError:
            from backend.rag.embeddings import embed_texts

        from qdrant_client.models import PointStruct
        import uuid

        client     = get_client()
        collection = "gbpiet_web"

        # ── Ensure website collection exists ──────────────────────────
        existing = {c.name for c in client.get_collections().collections}
        if collection not in existing:
            from qdrant_client.models import Distance, VectorParams
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print(f"[Scheduler] Created '{collection}' collection")

        # ── Get existing hashes to skip duplicates ────────────────────
        existing_hashes = set()
        try:
            offset = None
            while True:
                result, offset = client.scroll(
                    collection_name=collection,
                    limit=500,
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
        except Exception:
            pass

        # ── Filter new chunks only ────────────────────────────────────
        new_chunks = [
            c for c in chunks
            if c.get("content_hash", "") not in existing_hashes
        ]

        if not new_chunks:
            print("[Scheduler] No new content to index")
            return 0

        print(f"[Scheduler] Embedding {len(new_chunks)} new chunks...")
        texts   = [c["text"] for c in new_chunks]
        vectors = embed_texts(texts)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[i],
                payload={
                    "text":         new_chunks[i]["text"],
                    "url":          new_chunks[i]["url"],
                    "title":        new_chunks[i]["title"],
                    "category":     new_chunks[i]["category"],
                    "timestamp":    new_chunks[i]["timestamp"],
                    "content_hash": new_chunks[i]["content_hash"],
                    "source":       "website",
                    "language":     "en",
                }
            )
            for i in range(len(new_chunks))
        ]

        # Upsert in batches
        batch_size = 50
        for start in range(0, len(points), batch_size):
            client.upsert(
                collection_name=collection,
                points=points[start:start + batch_size],
            )
            print(f"  Upserted {min(start+batch_size, len(points))}/{len(points)}")

        print(f"[Scheduler] ✅ Indexed {len(new_chunks)} chunks")
        return len(new_chunks)

    except Exception as e:
        print(f"[Scheduler] ❌ Indexing error: {e}")
        import traceback
        traceback.print_exc()
        return 0


def run_scrape_job():
    """
    Main scrape job — called by scheduler.
    Scrapes website → parses → chunks → indexes into Qdrant.
    """
    global scrape_status

    print(f"\n[Scheduler] 🔄 Starting scrape job at {datetime.now()}")
    scrape_status["status"] = "running"
    scrape_status["errors"] = []

    try:
        from scraper.crawl_site import crawl_website
        from scraper.parse_pages import parse_pages, chunk_pages

        # Step 1: Crawl
        raw_pages = crawl_website()
        scrape_status["pages_scraped"] = len(raw_pages)

        # Step 2: Parse
        parsed = parse_pages(raw_pages)

        # Step 3: Chunk
        chunks = chunk_pages(parsed)

        # Step 4: Index into Qdrant
        added = index_website_chunks(chunks)
        scrape_status["chunks_added"] = added

        scrape_status["last_run"] = datetime.now().isoformat()
        scrape_status["status"]   = "success"
        print(f"[Scheduler] ✅ Scrape complete — {added} new chunks added")

    except Exception as e:
        error_msg = str(e)
        scrape_status["status"] = "error"
        scrape_status["errors"].append(error_msg)
        print(f"[Scheduler] ❌ Scrape failed: {error_msg}")


def start_scheduler():
    global _scheduler

    if not ENABLE_AUTO_SCRAPE:
        print("[Scheduler] Auto-scrape disabled")
        return

    if _scheduler and _scheduler.running:
        return

    _scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

    # Website scrape — every 24 hours
    _scheduler.add_job(
        func=run_scrape_job,
        trigger=IntervalTrigger(hours=SCRAPE_INTERVAL_HOURS),
        id="website_scrape",
        name="Website Scraper",
        replace_existing=True,
        misfire_grace_time=3600,
    )

    # ✅ DB cleanup — every 7 days, keep last 30 days of chats
    _scheduler.add_job(
        func=cleanup_db_job,
        trigger=IntervalTrigger(days=7),
        id="db_cleanup",
        name="DB Cleanup",
        replace_existing=True,
    )

    _scheduler.start()
    print(f"[Scheduler] ✅ Started — scrape every {SCRAPE_INTERVAL_HOURS}hrs, "
          f"cleanup every 7 days")


def cleanup_db_job():
    """Runs every 7 days — removes old conversations."""
    import asyncio
    try:
        from memory.database import cleanup_old_conversations
        asyncio.run(cleanup_old_conversations(days_to_keep=30))
        print("[Scheduler] ✅ DB cleanup done")
    except Exception as e:
        print(f"[Scheduler] DB cleanup error: {e}")    


def stop_scheduler():
    """Stop the scheduler gracefully."""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
        print("[Scheduler] Stopped")


def get_scrape_status() -> dict:
    return scrape_status.copy()
