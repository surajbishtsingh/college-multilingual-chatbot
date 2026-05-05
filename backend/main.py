import sys
import traceback

# Global DB vars
DATABASE_URL = ""
USE_POSTGRES = False

print("=" * 60)
print("[BOOT] Starting import sequence...")
sys.stdout.flush()

try:
    print("[BOOT] importing os, uuid, asyncio...")
    sys.stdout.flush()
    import os
    import uuid
    import asyncio
    import base64
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
    from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
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

    # DATABASE URL — read after load_dotenv()
    DATABASE_URL = os.getenv("DATABASE_URL", "")
    USE_POSTGRES = bool(DATABASE_URL) and "postgresql" in DATABASE_URL
    print(f"[DB] Using {'PostgreSQL' if USE_POSTGRES else 'SQLite (dev only)'}")
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


# ═════════════════════════════════════════════════════════════
# APP
# ═════════════════════════════════════════════════════════════

app = FastAPI(
    title="Diksha - GBPIET Chatbot",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ═════════════════════════════════════════════════════════════
# STARTUP
# ═════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    from rag.bm25_search import build_bm25_index

    print("=" * 60)
    print("[Startup] BEGIN")
    sys.stdout.flush()

    # ── DATABASE ──────────────────────────────────────────
    print("[Startup] Step 1: Database...")
    sys.stdout.flush()
    try:
        await asyncio.wait_for(init_db(), timeout=15)
        print("[Startup] ✅ Database ready")
    except asyncio.TimeoutError:
        print("[Startup] ❌ Database TIMED OUT — check DATABASE_URL")
    except Exception as e:
        print(f"[Startup] ❌ Database ERROR: {e}")
        traceback.print_exc()
    sys.stdout.flush()

    # ── BM25 ──────────────────────────────────────────────
    print("[Startup] Step 2: BM25 index...")
    sys.stdout.flush()
    try:
        await asyncio.wait_for(
            run_in_threadpool(build_bm25_index), timeout=30
        )
        print("[Startup] ✅ BM25 ready")
    except asyncio.TimeoutError:
        print("[Startup] ❌ BM25 TIMED OUT")
    except Exception as e:
        print(f"[Startup] ❌ BM25 ERROR: {e}")
        traceback.print_exc()
    sys.stdout.flush()

    # ── EMBEDDING MODEL ───────────────────────────────────
    print("[Startup] Step 3: Embedding model — skipped at startup, loads on first request ⚡")
    sys.stdout.flush()

    # ── QDRANT ────────────────────────────────────────────
    print("[Startup] Step 4: Qdrant...")
    sys.stdout.flush()
    try:
        await asyncio.wait_for(
            run_in_threadpool(get_qdrant), timeout=15
        )
        print("[Startup] ✅ Qdrant connected")
    except asyncio.TimeoutError:
        print("[Startup] ❌ Qdrant TIMED OUT — check QDRANT_URL and QDRANT_API_KEY")
    except Exception as e:
        print(f"[Startup] ❌ Qdrant ERROR: {e}")
        traceback.print_exc()
    sys.stdout.flush()

    # ── SCHEDULER ─────────────────────────────────────────
    print("[Startup] Step 5: Scheduler...")
    sys.stdout.flush()
    try:
        start_scheduler()
        print("[Startup] ✅ Scheduler started")
    except Exception as e:
        print(f"[Startup] ❌ Scheduler ERROR: {e}")
        traceback.print_exc()
    sys.stdout.flush()

    # ── ENV STATUS ────────────────────────────────────────
    print(f"  Groq Key 1 : {'✅' if os.getenv('GROQ_API_KEY') else '❌'}")
    print(f"  Groq Key 2 : {'✅' if os.getenv('GROQ_API_KEY_2') else '⚠️'}")
    print(f"  SerpAPI    : {'✅' if os.getenv('SERPAPI_KEY') else '⚠️'}")
    print(f"  DB         : {'PostgreSQL' if USE_POSTGRES else 'SQLite'}")
    print(f"  Qdrant     : {'Cloud' if os.getenv('QDRANT_URL') else 'Local'}")

    print("[Startup] ✅ Diksha Ready!")
    print("=" * 60)
    sys.stdout.flush()

# ← startup_event ends here (no more indentation below)


# ═════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    return {"status": "ok", "chatbot": "Diksha", "version": "2.0.0"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    # Detect language
    lang = request.language or detect_language(request.question)

    try:
        # Build memory context from past messages
        history = await build_memory_context(session_id)

        # Save user message
        await process_user_message(session_id, request.question, lang)

        # Get answer
        answer = await run_in_threadpool(
            get_answer, request.question, lang, history
        )

        # Save bot response
        await process_bot_message(session_id, answer, lang)

    except Exception as e:
        print(f"[Chat] ERROR: {e}")
        traceback.print_exc()
        answer = "I'm sorry, something went wrong. Please try again."

    return ChatResponse(
        answer=answer,
        language=lang,
        session_id=session_id,
    )


# ── FIX: TTS now returns base64 JSON (frontend expects this) ──
@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        audio_bytes = await run_in_threadpool(generate_voice, request.text, request.lang)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {"audio_base64": audio_b64}
    except Exception as e:
        print(f"[TTS] ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="TTS generation failed")


@app.get("/scrape-status")
async def scrape_status():
    try:
        status = get_scrape_status()
        return {"status": status}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/scrape-now")
async def scrape_now(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_scrape_job)
    return {"message": "Scrape job started"}


# ═════════════════════════════════════════════════════════════
# SHUTDOWN
# ═════════════════════════════════════════════════════════════

@app.on_event("shutdown")
async def shutdown_event():
    print("[Shutdown] Stopping scheduler...")
    try:
        stop_scheduler()
        print("[Shutdown] ✅ Scheduler stopped")
    except Exception as e:
        print(f"[Shutdown] Scheduler error: {e}")

    print("[Shutdown] Closing DB pool...")
    try:
        await close_pg_pool()
        print("[Shutdown] ✅ DB pool closed")
    except Exception as e:
        print(f"[Shutdown] DB pool error: {e}")

    print("[Shutdown] 👋 Diksha stopped")
    sys.stdout.flush()
