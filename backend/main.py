import sys
import traceback

print("=" * 60)
print("[BOOT] Starting import sequence...")
sys.stdout.flush()

try:
    print("[BOOT] importing os, uuid, asyncio...")
    sys.stdout.flush()
    import os
    import uuid
    import asyncio
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from datetime import datetime
    from typing import List, Optional
    from dotenv import load_dotenv
    load_dotenv()
    print("[BOOT] ✅ standard libs OK")
    sys.stdout.flush()

    print("[BOOT] importing FastAPI...")
    sys.stdout.flush()
    from fastapi import FastAPI, Request, BackgroundTasks, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.concurrency import run_in_threadpool
    from pydantic import BaseModel
    print("[BOOT] ✅ FastAPI OK")
    sys.stdout.flush()

    print("[BOOT] importing language_detector...")
    sys.stdout.flush()
    from language_detector import detect_language
    print("[BOOT] ✅ language_detector OK")
    sys.stdout.flush()

    print("[BOOT] importing rag.kb_query...")
    sys.stdout.flush()
    from rag.kb_query import get_answer, get_qdrant, get_embed_model
    print("[BOOT] ✅ rag.kb_query OK")
    sys.stdout.flush()

    print("[BOOT] importing voice...")
    sys.stdout.flush()
    from voice import generate_voice
    print("[BOOT] ✅ voice OK")
    sys.stdout.flush()

    print("[BOOT] importing memory.database...")
    sys.stdout.flush()
    from memory.database import init_db, close_pg_pool
    print("[BOOT] ✅ memory.database OK")
    sys.stdout.flush()

    print("[BOOT] importing memory.memory_manager...")
    sys.stdout.flush()
    from memory.memory_manager import (
        process_user_message,
        process_bot_message,
        build_memory_context,
    )
    print("[BOOT] ✅ memory.memory_manager OK")
    sys.stdout.flush()

    print("[BOOT] importing scraper.scheduler...")
    sys.stdout.flush()
    from scraper.scheduler import (
        start_scheduler,
        stop_scheduler,
        get_scrape_status,
        run_scrape_job,
    )
    print("[BOOT] ✅ scraper.scheduler OK")
    sys.stdout.flush()

    print("[BOOT] ✅ ALL IMPORTS SUCCESSFUL")
    sys.stdout.flush()

except Exception as e:
    print("=" * 60)
    print(f"[BOOT] ❌ IMPORT CRASHED: {e}")
    traceback.print_exc()
    print("=" * 60)
    sys.stdout.flush()
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# HF CACHE
# ─────────────────────────────────────────────────────────────

#os.environ.setdefault("HF_HOME", "/app/.cache")
#os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/app/.cache")
#os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ─────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────

DATABASE_URL = os.getenv("DATABASE_URL", "")

USE_POSTGRES = (
    bool(DATABASE_URL)
    and "postgresql" in DATABASE_URL
)

print(
    f"[DB] Using "
    f"{'PostgreSQL' if USE_POSTGRES else 'SQLite (dev only)'}"
)

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────

from language_detector import detect_language

from rag.kb_query import (
    get_answer,
    get_qdrant,
    get_embed_model,
)

from voice import generate_voice

from memory.database import (
    init_db,
    close_pg_pool,
)

from memory.memory_manager import (
    process_user_message,
    process_bot_message,
    build_memory_context,
)

from scraper.scheduler import (
    start_scheduler,
    stop_scheduler,
    get_scrape_status,
    run_scrape_job,
)

# ═════════════════════════════════════════════════════════════
# MODELS
# ═════════════════════════════════════════════════════════════

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    is_first_message: bool = False
    language: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    language: str
    session_id: str
    chatbot_name: str = "Diksha"


app = FastAPI(
    title="Diksha - GBPIET Chatbot",
    version="2.0.0"
)

# ═════════════════════════════════════════════════════════════
# STARTUP
# ═════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════
# STARTUP
# ═════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():

    from rag.bm25_search import build_bm25_index

    print("=" * 60)
    print("Starting Diksha Dynamic Edition...")

    # DATABASE

    try:
        print("[Startup] Initialising database...")

        await asyncio.wait_for(
            init_db(),
            timeout=30
        )

        print("[Startup] ✅ Database ready")

    except Exception as e:
        print(f"[Startup] Database failed: {e}")

    # BM25

    try:
        print("[Startup] Building BM25 index...")

        await asyncio.wait_for(
            run_in_threadpool(build_bm25_index),
            timeout=60
        )

        print("[Startup] ✅ BM25 ready")

    except Exception as e:
        print(f"[Startup] BM25 failed: {e}")

    # EMBEDDING MODEL
    # Removed timeout because HuggingFace model loading
    # can take longer on Railway cold start

    try:
        print("[Startup] Loading embedding model...")

        await run_in_threadpool(get_embed_model)

        print("[Startup] ✅ Embedding model ready")

    except Exception as e:
        print(f"[Startup] Embedding model failed: {e}")

    # QDRANT

    try:
        print("[Startup] Connecting Qdrant...")

        await asyncio.wait_for(
            run_in_threadpool(get_qdrant),
            timeout=120
        )

        print("[Startup] ✅ Qdrant connected")

    except Exception as e:
        print(f"[Startup] Qdrant failed: {e}")

    # SCHEDULER

    try:
        start_scheduler()
        print("[Startup] ✅ Scheduler started")

    except Exception as e:
        print(f"[Startup] Scheduler failed: {e}")

    # ENV STATUS

    groq_key = os.getenv("GROQ_API_KEY")
    groq_key2 = os.getenv("GROQ_API_KEY_2")
    serpapi = os.getenv("SERPAPI_KEY")

    print(f"\n  Groq Key 1 : {'✅' if groq_key else '❌'}")
    print(f"  Groq Key 2 : {'✅' if groq_key2 else '⚠️'}")
    print(f"  SerpAPI    : {'✅' if serpapi else '⚠️ DDG only'}")

    print(
        f"  DB         : "
        f"{'PostgreSQL' if USE_POSTGRES else 'SQLite'}"
    )

    print(
        f"  Qdrant     : "
        f"{'Cloud' if os.getenv('QDRANT_URL') else 'Local'}"
    )

    print("\n✅ Diksha Ready!")
    print("=" * 60)
