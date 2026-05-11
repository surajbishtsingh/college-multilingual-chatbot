import sys
import traceback

DATABASE_URL = ""
USE_POSTGRES = False

print("=" * 60)
print("[BOOT] Starting import sequence...")
sys.stdout.flush()

try:
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

    from fastapi import FastAPI, Request, BackgroundTasks, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.concurrency import run_in_threadpool
    from pydantic import BaseModel
    print("[BOOT] ✅ FastAPI OK")
    sys.stdout.flush()

    from language_detector import detect_language
    print("[BOOT] ✅ language_detector OK")

    from rag.kb_query import get_answer, get_qdrant, get_embed_model
    print("[BOOT] ✅ rag.kb_query OK")

    from voice import generate_voice
    print("[BOOT] ✅ voice OK")

    from memory.database import init_db, close_pg_pool
    print("[BOOT] ✅ memory.database OK")

    from memory.memory_manager import (
        process_user_message,
        process_bot_message,
        build_memory_context,
    )
    print("[BOOT] ✅ memory.memory_manager OK")

    from scraper.scheduler import (
        start_scheduler,
        stop_scheduler,
        get_scrape_status,
        run_scrape_job,
    )
    print("[BOOT] ✅ scraper.scheduler OK")

    DATABASE_URL = os.getenv("DATABASE_URL", "")
    USE_POSTGRES = bool(DATABASE_URL) and "postgresql" in DATABASE_URL
    print(f"[DB] Using {'PostgreSQL' if USE_POSTGRES else 'SQLite'}")
    print("[BOOT] ✅ ALL IMPORTS SUCCESSFUL")
    sys.stdout.flush()

except Exception as e:
    print(f"[BOOT] ❌ IMPORT CRASHED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ═══════════════════════════════════════════════
# SECTION → LANG MAPPING
# ═══════════════════════════════════════════════

SECTION_TO_LANG = {
    "garhwali": "ga",
    "kumauni":  "ku",
    "hindi":    "hi",
    "english":  "en",
}


# ═══════════════════════════════════════════════
# MODELS
# ═══════════════════════════════════════════════

class TTSRequest(BaseModel):
    text: str
    lang: str = "en"


class ChatRequest(BaseModel):
    question:         str
    session_id:       Optional[str] = None
    is_first_message: bool          = False
    language:         Optional[str] = None
    section:          Optional[str] = None   # "garhwali"|"kumauni"|"hindi"|"english"


class ChatResponse(BaseModel):
    answer:       str
    language:     str
    session_id:   str
    chatbot_name: str = "Diksha"


# ═══════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════

app = FastAPI(title="Diksha - GBPIET Chatbot", version="2.0.0")

# ── CORS — allow ALL origins, ALL networks ────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://college-multilingual-chatbot-lzrp.vercel.app/"],        # every domain/IP allowed
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)


# ═══════════════════════════════════════════════
# STARTUP
# ═══════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    from rag.bm25_search import build_bm25_index
    from rag.kb_query import get_embed_model, load_qa_database

    print("=" * 60)
    print("[Startup] BEGIN")
    sys.stdout.flush()

    # 1. Database
    print("[Startup] Step 1: Database...")
    try:
        await asyncio.wait_for(init_db(), timeout=15)
        print("[Startup] ✅ Database ready")
    except Exception as e:
        print(f"[Startup] ⚠️ Database: {e}")
    sys.stdout.flush()

    # 2. BM25
    print("[Startup] Step 2: BM25 index...")
    try:
        await asyncio.wait_for(run_in_threadpool(build_bm25_index), timeout=30)
        print("[Startup] ✅ BM25 ready")
    except Exception as e:
        print(f"[Startup] ⚠️ BM25: {e}")
    sys.stdout.flush()

    # 3. QA Database — embedding model loads lazily (saves ~500MB RAM)
    print("[Startup] Step 3: Loading QA database...")
    try:
        await run_in_threadpool(load_qa_database)
        print("[Startup] ✅ QA database ready")
    except Exception as e:
        print(f"[Startup] ⚠️ QA database: {e}")
    sys.stdout.flush()

    # 4. Qdrant
    print("[Startup] Step 4: Qdrant...")
    try:
        await asyncio.wait_for(run_in_threadpool(get_qdrant), timeout=15)
        print("[Startup] ✅ Qdrant connected")
    except Exception as e:
        print(f"[Startup] ⚠️ Qdrant: {e}")
    sys.stdout.flush()

    # 5. Scheduler
    print("[Startup] Step 5: Scheduler...")
    try:
        start_scheduler()
        print("[Startup] ✅ Scheduler started")
    except Exception as e:
        print(f"[Startup] ⚠️ Scheduler: {e}")
    sys.stdout.flush()

    # ── API Keys & Environment Summary ────────────────────────────────
    print("-" * 60)
    print("[Startup] Environment:")
    print(f"  Groq Key 1 : {'✅' if os.getenv('GROQ_API_KEY')   else '❌ NOT SET'}")
    print(f"  Groq Key 2 : {'✅' if os.getenv('GROQ_API_KEY_2') else '⚠️  not set'}")
    print(f"  Groq Key 3 : {'✅' if os.getenv('GROQ_API_KEY_3') else '⚠️  not set'}")
    print(f"  Groq Key 4 : {'✅' if os.getenv('GROQ_API_KEY_4') else '⚠️  not set'}")
    print(f"  SerpAPI    : {'✅' if os.getenv('SERPAPI_KEY')     else '⚠️  not set'}")
    print(f"  DB         : {'PostgreSQL' if USE_POSTGRES else 'SQLite'}")
    print(f"  Qdrant     : {'Cloud' if os.getenv('QDRANT_URL') else 'Local'}")
    print(f"  Env        : {os.getenv('ENVIRONMENT', 'development')}")
    print("-" * 60)
    print("[Startup] ✅ Diksha Ready!")
    print("=" * 60)
    sys.stdout.flush()


# ═══════════════════════════════════════════════
# SHUTDOWN
# ═══════════════════════════════════════════════

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


# ═══════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════

@app.get("/")
def home():
    return {
        "chatbot": "Diksha",
        "college": "GBPIET, Pauri Garhwal",
        "status":  "running",
        "version": "2.0.0",
        "db":      "PostgreSQL" if USE_POSTGRES else "SQLite",
        "qdrant":  "Cloud" if os.getenv("QDRANT_URL") else "Local",
    }


@app.get("/health")
async def health_check():
    """Detailed health check — shows all 4 Groq key statuses."""
    return {
        "status":  "ok",
        "chatbot": "Diksha",
        "version": "2.0.0",
        "groq_keys": {
            "key1": bool(os.getenv("GROQ_API_KEY")),
            "key2": bool(os.getenv("GROQ_API_KEY_2")),
            "key3": bool(os.getenv("GROQ_API_KEY_3")),
            "key4": bool(os.getenv("GROQ_API_KEY_4")),
        },
        "serpapi": bool(os.getenv("SERPAPI_KEY")),
        "db":      "PostgreSQL" if USE_POSTGRES else "SQLite",
        "qdrant":  "Cloud" if os.getenv("QDRANT_URL") else "Local",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    session_id = request.session_id or str(uuid.uuid4())

    # ── Language resolution — 3 level priority ────────────────────────
    # Priority 1: section field  → user ne language choose ki
    # Priority 2: language field → explicit lang code
    # Priority 3: auto-detect   → question text se detect
    if request.section and request.section.lower().strip() in SECTION_TO_LANG:
        lang = SECTION_TO_LANG[request.section.lower().strip()]
        print(f"[Chat] Lang from section='{request.section}' → '{lang}'")
    elif request.language in ("en", "hi", "ga", "ku"):
        lang = request.language
        print(f"[Chat] Lang from request.language='{lang}'")
    else:
        lang = detect_language(request.question)
        print(f"[Chat] Lang auto-detected='{lang}'")

    try:
        history = await build_memory_context(session_id)
        await process_user_message(session_id, request.question, lang)
        answer  = await run_in_threadpool(get_answer, request.question, lang, history)
        await process_bot_message(session_id, answer, lang)
    except Exception as e:
        print(f"[Chat] ERROR: {e}")
        traceback.print_exc()
        answer = "I'm sorry, something went wrong. Please try again."

    return ChatResponse(answer=answer, language=lang, session_id=session_id)


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Returns base64 encoded audio — matches frontend expectation."""
    try:
        audio_bytes = await run_in_threadpool(generate_voice, request.text, request.lang)
        if audio_bytes:
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            return {"audio_base64": audio_b64}
        return {"audio_base64": None}
    except Exception as e:
        print(f"[TTS] ERROR: {e}")
        traceback.print_exc()
        return {"audio_base64": None}


@app.get("/scrape-status")
async def scrape_status():
    try:
        return {"status": get_scrape_status()}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.post("/scrape-now")
async def scrape_now(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_scrape_job)
    return {"message": "Scrape job started in background"}


@app.get("/admin/visits")
def get_visit_stats():
    return {"message": "Visit stats not configured in this version"}


@app.get("/admin/groq-stats")
def groq_stats():
    """Check Groq key usage and cache stats."""
    try:
        from rag.groq_manager import get_stats
        return get_stats()
    except Exception as e:
        return {"error": str(e)}


@app.post("/admin/clear-cache")
def clear_groq_cache():
    """Clear Groq response cache — useful after KB updates."""
    try:
        from rag.groq_manager import clear_cache
        clear_cache()
        return {"status": "Cache cleared ✅"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/admin/rebuild-kb")
async def rebuild_kb(background_tasks: BackgroundTasks):
    import subprocess

    def run():
        result = subprocess.run(
            [sys.executable, "build_kb.py"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            from rag import kb_query
            kb_query._qa_database = []
            kb_query._embed_model = None
            print("[Admin] ✅ KB rebuilt")
        else:
            print(f"[Admin] ❌ Build failed:\n{result.stderr}")

    background_tasks.add_task(run)
    return {"status": "KB rebuild started — check logs"}


# ═══════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
