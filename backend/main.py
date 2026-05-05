# main.py — Diksha GBPIET Chatbot — Production Ready

from fastapi import FastAPI, Request, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

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

# ─────────────────────────────────────────────────────────────
# HF CACHE
# ─────────────────────────────────────────────────────────────

os.environ.setdefault("HF_HOME", "/app/.cache")
os.environ.setdefault("TRANSFORMERS_CACHE", "/app/.cache")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", "/app/.cache")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

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


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[dict]

# ═════════════════════════════════════════════════════════════
# FASTAPI
# ═════════════════════════════════════════════════════════════

app = FastAPI(
    title="Diksha - GBPIET Chatbot",
    version="2.0.0"
)

# ═════════════════════════════════════════════════════════════
# CORS
# ═════════════════════════════════════════════════════════════

FRONTEND_URL = os.getenv("FRONTEND_URL", "")

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

if FRONTEND_URL:
    ALLOWED_ORIGINS.append(FRONTEND_URL)
    ALLOWED_ORIGINS.append(FRONTEND_URL.rstrip("/"))

EXTRA_ORIGINS = os.getenv("EXTRA_ORIGINS", "")

if EXTRA_ORIGINS:
    for origin in EXTRA_ORIGINS.split(","):
        origin = origin.strip()

        if origin:
            ALLOWED_ORIGINS.append(origin)

print(f"[CORS] Allowed origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "Accept",
        "Origin",
        "X-Requested-With",
        "Access-Control-Request-Method",
        "Access-Control-Request-Headers",
    ],
    expose_headers=["Content-Length"],
    max_age=3600,
)

# ═════════════════════════════════════════════════════════════
# OPTIONS FIX
# ═════════════════════════════════════════════════════════════

@app.options("/{full_path:path}")
async def options_handler(request: Request, full_path: str):

    origin = request.headers.get("origin", "*")

    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Methods":
                "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers":
                "Content-Type, Authorization, Accept, Origin",
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "3600",
        },
    )

# ═════════════════════════════════════════════════════════════
# CHAT SESSIONS
# ═════════════════════════════════════════════════════════════

chat_sessions = {}

# ═════════════════════════════════════════════════════════════
# VISIT DATA
# ═════════════════════════════════════════════════════════════

visit_data = {
    "total_visits": 0,
    "unique_ips": set(),
    "chatbot_usage": 0,
    "daily_counts": {},
    "first_visit": None,
    "last_visit": None,
    "unique_chatbot_users": set(),
    "user_count": 0,
}

REPORT_EMAIL = os.getenv(
    "REPORT_EMAIL",
    "bishtsuraj0311@gmail.com"
)

# ═════════════════════════════════════════════════════════════
# EMAIL REPORT
# ═════════════════════════════════════════════════════════════

def send_visit_report():

    sender_email = os.getenv("SMTP_EMAIL")
    sender_pass = os.getenv("SMTP_PASSWORD")

    smtp_host = os.getenv(
        "SMTP_HOST",
        "smtp.gmail.com"
    )

    smtp_port = int(
        os.getenv("SMTP_PORT", "587")
    )

    if not sender_email or not sender_pass:
        print("[VisitCounter] SMTP credentials missing")
        return

    today = datetime.now().strftime("%Y-%m-%d")

    daily_table = "\n".join(
        f"  {d}: {c} visits"
        for d, c in sorted(
            visit_data["daily_counts"].items()
        )[-7:]
    )

    body = f"""
Diksha Chatbot Visit Report

Total Visits    : {visit_data["total_visits"]}
Chatbot Usage   : {visit_data["chatbot_usage"]}
Unique Visitors : {len(visit_data["unique_ips"])}
New Users       : {visit_data["user_count"]}
Today Visits    : {visit_data["daily_counts"].get(today, 0)}
First Visit     : {visit_data["first_visit"]}
Last Visit      : {visit_data["last_visit"]}

Last 7 Days:
{daily_table}

Report Time:
{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    msg = MIMEMultipart()

    msg["From"] = sender_email
    msg["To"] = REPORT_EMAIL

    msg["Subject"] = (
        f"Diksha Report — "
        f"{visit_data['user_count']} Users"
    )

    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:

            server.ehlo()
            server.starttls()

            server.login(sender_email, sender_pass)

            server.sendmail(
                sender_email,
                REPORT_EMAIL,
                msg.as_string(),
            )

        print(f"[VisitCounter] Email sent")

    except Exception as e:
        print(f"[VisitCounter] Email failed: {e}")

# ═════════════════════════════════════════════════════════════
# RECORD VISIT
# ═════════════════════════════════════════════════════════════

def record_visit(ip: str):

    now = datetime.now()

    today = now.strftime("%Y-%m-%d")

    visit_data["total_visits"] += 1

    visit_data["last_visit"] = now.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    visit_data["unique_ips"].add(ip)

    visit_data["daily_counts"][today] = (
        visit_data["daily_counts"].get(today, 0) + 1
    )

    if visit_data["first_visit"] is None:
        visit_data["first_visit"] = visit_data["last_visit"]

    if ip not in visit_data["unique_chatbot_users"]:

        visit_data["unique_chatbot_users"].add(ip)

        visit_data["user_count"] += 1

        print(f"[User] New user: {ip}")

        if visit_data["user_count"] % 10 == 0:
            send_visit_report()

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

    try:
        print("[Startup] Loading embedding model...")

        await asyncio.wait_for(
            run_in_threadpool(get_embed_model),
            timeout=120
        )

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
    gemini_key = os.getenv("GEMINI_API_KEY")
    serpapi = os.getenv("SERPAPI_KEY")

    print(f"\n  Groq Key 1 : {'✅' if groq_key else '❌'}")
    print(f"  Groq Key 2 : {'✅' if groq_key2 else '⚠️'}")
    print(f"  Gemini     : {'✅' if gemini_key else '❌'}")
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

# ═════════════════════════════════════════════════════════════
# SHUTDOWN
# ═════════════════════════════════════════════════════════════

@app.on_event("shutdown")
async def shutdown_event():

    try:
        stop_scheduler()

    except Exception as e:
        print(f"[Shutdown] Scheduler stop failed: {e}")

    if USE_POSTGRES:

        try:
            await close_pg_pool()

        except Exception as e:
            print(f"[Shutdown] PG pool close failed: {e}")

    print("[Shutdown] Clean shutdown complete")

# ═════════════════════════════════════════════════════════════
# HOME
# ═════════════════════════════════════════════════════════════

@app.get("/")
def home():

    return {
        "chatbot": "Diksha",
        "status": "running",
        "db": (
            "PostgreSQL"
            if USE_POSTGRES
            else "SQLite"
        ),
        "qdrant": (
            "Cloud"
            if os.getenv("QDRANT_URL")
            else "Local"
        ),
    }

# ═════════════════════════════════════════════════════════════
# HEALTH
# ═════════════════════════════════════════════════════════════

@app.get("/health")
async def health():

    from memory.database import get_db_stats

    db_stats = await get_db_stats()

    return {
        "status": "ok",

        "db_type": (
            "PostgreSQL"
            if USE_POSTGRES
            else "SQLite"
        ),

        "groq": (
            "present"
            if os.getenv("GROQ_API_KEY")
            else "missing"
        ),

        "gemini": (
            "present"
            if os.getenv("GEMINI_API_KEY")
            else "missing"
        ),

        "qdrant": (
            "cloud"
            if os.getenv("QDRANT_URL")
            else "local"
        ),

        "db_stats": db_stats,
    }

# ═════════════════════════════════════════════════════════════
# TTS
# ═════════════════════════════════════════════════════════════

@app.post("/tts")
async def tts_endpoint(request: TTSRequest):

    audio = await run_in_threadpool(
        generate_voice,
        request.text,
        request.lang
    )

    return {
        "audio_base64": audio
    }

# ═════════════════════════════════════════════════════════════
# CHAT
# ═════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):

    try:
        question = request.question.strip()

        session_id = (
            request.session_id
            or str(uuid.uuid4())
        )

        lang = (
            request.language
            if request.language in ["en", "hi", "ga", "ku"]
            else detect_language(question)
        )

        print(f"[Chat] lang={lang}")

        # VISIT TRACKING

        forwarded = req.headers.get("x-forwarded-for")

        ip = (
            forwarded.split(",")[0].strip()
            if forwarded
            else (
                req.client.host
                if req.client
                else "unknown"
            )
        )

        record_visit(ip)

        visit_data["chatbot_usage"] += 1

        # MEMORY

        await process_user_message(
            session_id,
            question,
            lang
        )

        memory_context = await build_memory_context(
            session_id
        )

        # ANSWER GENERATION

        try:
            answer = await asyncio.wait_for(
                run_in_threadpool(
                    get_answer,
                    question,
                    lang,
                    memory_context
                ),
                timeout=25.0
            )

            if not answer or len(answer.strip()) == 0:
                raise ValueError("Empty answer")

        except asyncio.TimeoutError:

            print("[TIMEOUT] Answer generation timeout")

            answer = (
                "I'm thinking... please try again."
                if lang == "en"
                else "कृपया फिर से पूछें।"
            )

        except Exception as e:

            print(f"[ERROR] get_answer failed: {e}")

            answer = (
                "Sorry, I don't have information on that yet."
                if lang == "en"
                else "माफ़ कीजिए, अभी जानकारी उपलब्ध नहीं है।"
            )

        # SAVE BOT MESSAGE

        await process_bot_message(
            session_id,
            answer,
            lang
        )

        return ChatResponse(
            answer=answer,
            language=lang,
            session_id=session_id,
        )

    except Exception as e:

        print(f"[FATAL ERROR] Chat failed: {e}")

        return ChatResponse(
            answer="Sorry, something went wrong.",
            language="en",
            session_id=request.session_id or "unknown"
        )

# ═════════════════════════════════════════════════════════════
# HISTORY
# ═════════════════════════════════════════════════════════════

@app.get(
    "/history/{session_id}",
    response_model=HistoryResponse
)
def get_history(session_id: str):

    return HistoryResponse(
        session_id=session_id,
        messages=chat_sessions.get(session_id, []),
    )

# ═════════════════════════════════════════════════════════════
# SESSIONS
# ═════════════════════════════════════════════════════════════

@app.get("/sessions")
def get_sessions():

    return {
        "total_sessions": len(chat_sessions),
        "session_ids": list(chat_sessions.keys()),
    }

# ═════════════════════════════════════════════════════════════
# ADMIN — VISITS
# ═════════════════════════════════════════════════════════════

@app.get("/admin/visits")
def get_visit_stats():

    return {
        "total_visits": visit_data["total_visits"],
        "chatbot_usage": visit_data["chatbot_usage"],
        "unique_users": len(
            visit_data["unique_chatbot_users"]
        ),
        "daily_counts": visit_data["daily_counts"],
    }

# ═════════════════════════════════════════════════════════════
# ADMIN — SEND REPORT
# ═════════════════════════════════════════════════════════════

@app.get("/admin/send-report")
def send_report_now():

    send_visit_report()

    return {
        "status": "Report sent",
        "to": REPORT_EMAIL,
    }

# ═════════════════════════════════════════════════════════════
# ADMIN — SCRAPE NOW
# ═════════════════════════════════════════════════════════════

@app.post("/admin/scrape-now")
async def scrape_now(background_tasks: BackgroundTasks):

    background_tasks.add_task(run_scrape_job)

    return {
        "status": "Scrape started"
    }

# ═════════════════════════════════════════════════════════════
# ADMIN — SCRAPE STATUS
# ═════════════════════════════════════════════════════════════

@app.get("/admin/scrape-status")
def scrape_status_endpoint():

    return get_scrape_status()

# ═════════════════════════════════════════════════════════════
# ADMIN — USER MEMORY
# ═════════════════════════════════════════════════════════════

@app.get("/admin/user-memory/{session_id}")
async def get_user_memory(session_id: str):

    from memory.database import (
        get_user_facts,
        get_recent_history,
    )

    facts = await get_user_facts(session_id)

    history = await get_recent_history(
        session_id,
        limit=10
    )

    return {
        "facts": facts,
        "recent_history": history,
    }

# ═════════════════════════════════════════════════════════════
# ADMIN — REBUILD KB
# ═════════════════════════════════════════════════════════════

@app.post("/admin/rebuild-kb")
async def rebuild_kb(background_tasks: BackgroundTasks):

    import subprocess
    import sys

    def run():

        result = subprocess.run(
            [sys.executable, "build_kb.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(
                os.path.abspath(__file__)
            ),
        )

        if result.returncode == 0:

            import rag.kb_query as kb

            kb._qa_database = []
            kb._embed_model = None

            print("[Admin] KB rebuilt successfully")

        else:
            print(
                f"[Admin] Build failed:\n"
                f"{result.stderr}"
            )

    background_tasks.add_task(run)

    return {
        "status": "KB rebuild started"
    }

# ═════════════════════════════════════════════════════════════
# RUN
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port
    )
