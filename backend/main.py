# main.py — Diksha GBPIET Chatbot
from fastapi import FastAPI, Request, BackgroundTasks
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

# ── Cache HuggingFace model to disk (prevents re-download on restart) ─
os.environ.setdefault("HF_HOME",                     "/app/.cache")
os.environ.setdefault("TRANSFORMERS_CACHE",           "/app/.cache")
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME",   "/app/.cache")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# ── Detect Database Type ──────────────────────────────────────────────
USE_POSTGRES = (
    bool(os.getenv("DATABASE_URL")) and
    "postgresql" in os.getenv("DATABASE_URL", "")
)
print(f"[DB] Using {'PostgreSQL' if USE_POSTGRES else 'SQLite'}")

# ── Local imports ─────────────────────────────────────────────────────
from language_detector import detect_language
from rag.kb_query import get_answer, get_qdrant, get_embed_model
from voice import generate_voice
from memory.database import init_db, close_pg_pool
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

# ══════════════════════════════════════════════════════════════════════
#   MODELS
# ══════════════════════════════════════════════════════════════════════
class TTSRequest(BaseModel):
    text: str
    lang: str = "en"


class ChatRequest(BaseModel):
    question:         str
    session_id:       Optional[str] = None
    is_first_message: bool          = False
    language:         Optional[str] = None


class ChatResponse(BaseModel):
    answer:       str
    language:     str
    session_id:   str
    chatbot_name: str = "Diksha"


class HistoryResponse(BaseModel):
    session_id: str
    messages:   List[dict]


# ══════════════════════════════════════════════════════════════════════
#   APP
# ══════════════════════════════════════════════════════════════════════
app = FastAPI(title="Diksha - GBPIET Chatbot", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://gbpiet.ac.in",
        "https://www.gbpiet.ac.in",
        # ✅ Add your exact Vercel URL
        "https://college-multilingual-chatbot-lzrp.vercel.app",
        "https://*.vercel.app",
        "https://*.railway.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════════
#   VISIT COUNTER
# ══════════════════════════════════════════════════════════════════════
visit_data = {
    "total_visits":         0,
    "unique_ips":           set(),
    "chatbot_usage":        0,
    "daily_counts":         {},
    "first_visit":          None,
    "last_visit":           None,
    "unique_chatbot_users": set(),
    "user_count":           0,
}

REPORT_EMAIL = "bishtsuraj0311@gmail.com"


def send_visit_report():
    sender_email = os.getenv("SMTP_EMAIL")
    sender_pass  = os.getenv("SMTP_PASSWORD")
    smtp_host    = os.getenv("SMTP_HOST", "sandbox.smtp.mailtrap.io")
    smtp_port    = int(os.getenv("SMTP_PORT", "2525"))

    if not sender_email or not sender_pass:
        print("[VisitCounter] SMTP credentials missing in .env")
        return

    today       = datetime.now().strftime("%Y-%m-%d")
    daily_table = "\n".join(
        f"  {d}: {c} visits"
        for d, c in sorted(visit_data["daily_counts"].items())[-7:]
    )

    body = f"""
नमस्ते,

Diksha Chatbot की Latest Visit Report:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  कुल Visits         : {visit_data["total_visits"]}
  Chatbot Usage      : {visit_data["chatbot_usage"]}
  Unique Visitors    : {len(visit_data["unique_ips"])}
  नए Chatbot Users  : {visit_data["user_count"]}
  आज की Visits      : {visit_data["daily_counts"].get(today, 0)}
  पहली Visit        : {visit_data["first_visit"]}
  आखिरी Visit       : {visit_data["last_visit"]}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

पिछले 7 दिन:
{daily_table}

Report Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

— Diksha Bot | GBPIET
"""

    msg            = MIMEMultipart()
    msg["From"]    = sender_email
    msg["To"]      = REPORT_EMAIL
    msg["Subject"] = (
        f"Diksha Report — {visit_data['user_count']} Users, "
        f"{visit_data['total_visits']} Visits"
    )
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(sender_email, sender_pass)
            server.sendmail(sender_email, REPORT_EMAIL, msg.as_string())
        print(f"[VisitCounter] Email sent to {REPORT_EMAIL}")
    except Exception as e:
        print(f"[VisitCounter] Email failed: {e}")


def record_visit(ip: str):
    now   = datetime.now()
    today = now.strftime("%Y-%m-%d")

    visit_data["total_visits"] += 1
    visit_data["last_visit"]    = now.strftime("%Y-%m-%d %H:%M:%S")
    visit_data["unique_ips"].add(ip)
    visit_data["daily_counts"][today] = (
        visit_data["daily_counts"].get(today, 0) + 1
    )

    if visit_data["first_visit"] is None:
        visit_data["first_visit"] = visit_data["last_visit"]

    if ip not in visit_data["unique_chatbot_users"]:
        visit_data["unique_chatbot_users"].add(ip)
        visit_data["user_count"] += 1
        print(f"[User] New chatbot user: {ip}")

        if visit_data["user_count"] % 10 == 0:
            send_visit_report()


# ══════════════════════════════════════════════════════════════════════
#   STARTUP — lightweight only, NO heavy model loading
# ══════════════════════════════════════════════════════════════════════
@app.on_event("startup")
async def startup_event():
    from rag.bm25_search import build_bm25_index

    print("=" * 55)
    print("Starting Diksha Dynamic Edition...")

    # ── 1. Database ───────────────────────────────────────────────────
    try:
        print("[Startup] Initialising database...")
        await asyncio.wait_for(init_db(), timeout=30)
        print("[Startup] ✅ Database ready")
    except asyncio.TimeoutError:
        print("[Startup] ⚠️  Database init timed out — continuing anyway")
    except Exception as e:
        print(f"[Startup] ⚠️  Database init failed ({e}) — continuing anyway")

    # ── 2. BM25 index (light — just reads JSON files) ─────────────────
    try:
        print("[Startup] Building BM25 index...")
        await asyncio.wait_for(
            run_in_threadpool(build_bm25_index), timeout=60
        )
        print("[Startup] ✅ BM25 ready")
    except asyncio.TimeoutError:
        print("[Startup] ⚠️  BM25 timed out — will build on first request")
    except Exception as e:
        print(f"[Startup] ⚠️  BM25 failed ({e}) — will build on first request")

    # ── 3. Scheduler ──────────────────────────────────────────────────
    try:
        start_scheduler()
        print("[Startup] ✅ Scheduler started")
    except Exception as e:
        print(f"[Startup] ⚠️  Scheduler failed ({e}) — continuing without it")

    # ── NOTE ──────────────────────────────────────────────────────────
    # Embedding model + Qdrant are NOT loaded here.
    # They load lazily on the first /chat request.
    # This prevents OOM crash-loop on low-memory containers.
    # ─────────────────────────────────────────────────────────────────

    groq_key   = os.getenv("GROQ_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    serpapi    = os.getenv("SERPAPI_KEY")
    print(f"\n  Groq    : {'✅' if groq_key   else '❌ MISSING — chat will fail'}")
    print(f"  Gemini  : {'✅' if gemini_key else '❌ MISSING — fallback unavailable'}")
    print(f"  SerpAPI : {'✅' if serpapi    else '⚠️  missing — no internet search'}")
    print(f"  DB Type : {'PostgreSQL' if USE_POSTGRES else 'SQLite'}")
    print("\n✅ Diksha is ready! (embedding model loads on first /chat request)")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════════════
#   SHUTDOWN
# ══════════════════════════════════════════════════════════════════════
@app.on_event("shutdown")
async def shutdown_event():
    try:
        stop_scheduler()
    except Exception as e:
        print(f"[Shutdown] Scheduler stop error: {e}")

    if USE_POSTGRES:
        try:
            await close_pg_pool()
        except Exception as e:
            print(f"[Shutdown] PG pool close error: {e}")

    print("[Shutdown] Clean shutdown complete")


# ══════════════════════════════════════════════════════════════════════
#   ROUTES
# ══════════════════════════════════════════════════════════════════════
@app.get("/")
def home():
    return {
        "chatbot": "Diksha",
        "status":  "running",
        "db":      "PostgreSQL" if USE_POSTGRES else "SQLite",
    }


@app.get("/health")
async def health():
    from memory.database import get_db_stats
    db_stats = await get_db_stats()
    return {
        "status":     "ok",
        "db_type":    "PostgreSQL" if USE_POSTGRES else "SQLite",
        "groq_key":   "present" if os.getenv("GROQ_API_KEY")   else "missing",
        "gemini_key": "present" if os.getenv("GEMINI_API_KEY") else "missing",
        "db_stats":   db_stats,
    }


@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    audio = await run_in_threadpool(generate_voice, request.text, request.lang)
    return {"audio_base64": audio}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    try:
        question   = request.question.strip()
        session_id = request.session_id or str(uuid.uuid4())

        # Language detection
        lang = (
            request.language
            if request.language in ["en", "hi", "ga", "ku"]
            else detect_language(question)
        )

        # Visit tracking
        forwarded = req.headers.get("x-forwarded-for")
        ip        = forwarded.split(",")[0].strip() if forwarded else req.client.host
        record_visit(ip)
        visit_data["chatbot_usage"] += 1

        # Memory
        await process_user_message(session_id, question, lang)
        memory_context = await build_memory_context(session_id)

        # 🔥 SAFE ANSWER GENERATION
        try:
            answer = await run_in_threadpool(get_answer, question, lang, memory_context)

            # ✅ fallback if empty
            if not answer or len(answer.strip()) == 0:
                raise ValueError("Empty answer")

        except Exception as e:
            print(f"[ERROR] get_answer failed: {e}")

            # ✅ fallback response
            answer = (
                "Sorry, I don’t have information on that yet. "
                "Please ask about admissions, fees, courses, or contact details."
            )

        await process_bot_message(session_id, answer, lang)

        return ChatResponse(
            answer=answer,
            language=lang,
            session_id=session_id
        )

    except Exception as e:
        print(f"[FATAL ERROR] Chat failed: {e}")

        # 🚨 NEVER CRASH → ALWAYS RETURN RESPONSE
        return ChatResponse(
            answer="Sorry, something went wrong. Please try again.",
            language="en",
            session_id=request.session_id or "unknown"
        )

@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str):
    return HistoryResponse(
        session_id=session_id,
        messages=chat_sessions.get(session_id, []),
    )


@app.get("/sessions")
def get_sessions():
    return {
        "total_sessions": len(chat_sessions),
        "session_ids":    list(chat_sessions.keys()),
    }


# ══════════════════════════════════════════════════════════════════════
#   ADMIN ROUTES
# ══════════════════════════════════════════════════════════════════════
@app.get("/admin/visits")
def get_visit_stats():
    return {
        "total_visits":         visit_data["total_visits"],
        "chatbot_usage":        visit_data["chatbot_usage"],
        "unique_chatbot_users": len(visit_data["unique_chatbot_users"]),
        "unique_visitors":      len(visit_data["unique_ips"]),
        "daily_counts":         visit_data["daily_counts"],
        "first_visit":          visit_data["first_visit"],
        "last_visit":           visit_data["last_visit"],
    }


@app.get("/admin/send-report")
def send_report_now():
    send_visit_report()
    return {
        "status":       "Report sent",
        "to":           REPORT_EMAIL,
        "unique_users": len(visit_data["unique_ips"]),
    }


@app.post("/admin/scrape-now")
async def scrape_now(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_scrape_job)
    return {"status": "Scrape started in background"}


@app.get("/admin/scrape-status")
def scrape_status_endpoint():
    return get_scrape_status()


@app.get("/admin/user-memory/{session_id}")
async def get_user_memory(session_id: str):
    from memory.database import get_user_facts, get_recent_history
    facts   = await get_user_facts(session_id)
    history = await get_recent_history(session_id, limit=10)
    return {"facts": facts, "recent_history": history}


@app.post("/admin/rebuild-kb")
async def rebuild_kb(background_tasks: BackgroundTasks):
    import subprocess, sys

    def run():
        result = subprocess.run(
            [sys.executable, "build_kb.py"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if result.returncode == 0:
            import rag.kb_query as kb
            kb._qa_database = []
            kb._embed_model = None
            print("[Admin] KB rebuilt successfully")
        else:
            print(f"[Admin] Build failed:\n{result.stderr}")

    background_tasks.add_task(run)
    return {"status": "KB rebuild started — check server logs"}


# ══════════════════════════════════════════════════════════════════════
#   RUN (Railway / local)
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
