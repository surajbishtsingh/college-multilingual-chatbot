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
    from fastapi.responses import PlainTextResponse
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

# ── CORS — allow all origins (mobile + all networks) ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # ✅ Sab allow — "not connected" fix
    allow_credentials=False,   # ✅ False zaroori hai jab origins=["*"] ho
    allow_methods=["*"],
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

    # 3. QA Database
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

    # 6. Auto-scrape if gbpiet_web collection is empty
    print("[Startup] Step 6: Checking Qdrant data...")
    try:
        from qdrant_setup import get_client as _get_qdrant
        _qc   = _get_qdrant()
        _info = _qc.get_collection("gbpiet_web")
        if _info.points_count < 10:
            print(f"[Startup] ⚠️ Qdrant website has only {_info.points_count} points — auto-scraping...")
            import threading
            def _auto_scrape():
                try:
                    run_scrape_job()
                    print("[Startup] ✅ Auto-scrape complete")
                except Exception as e:
                    print(f"[Startup] ❌ Auto-scrape failed: {e}")
            threading.Thread(target=_auto_scrape, daemon=True).start()
        else:
            print(f"[Startup] ✅ Qdrant has {_info.points_count} points — no scrape needed")
    except Exception as e:
        print(f"[Startup] ⚠️ Auto-scrape check: {e}")
    sys.stdout.flush()

    # ── Environment Summary ───────────────────────────────────────────
    print("-" * 60)
    print("[Startup] Environment:")
    print(f"  Groq Key 1 : {'✅' if os.getenv('GROQ_API_KEY')   else '❌ NOT SET'}")
    print(f"  Groq Key 2 : {'✅' if os.getenv('GROQ_API_KEY_2') else '⚠️  not set'}")
    print(f"  Groq Key 3 : {'✅' if os.getenv('GROQ_API_KEY_3') else '⚠️  not set'}")
    print(f"  Groq Key 4 : {'✅' if os.getenv('GROQ_API_KEY_4') else '⚠️  not set'}")
    print(f"  SerpAPI    : {'✅' if os.getenv('SERPAPI_KEY')     else '⚠️  not set'}")
    print(f"  Gemini     : {'✅' if os.getenv('GEMINI_API_KEY')  else '⚠️  not set'}")
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

# ── OPTIONS preflight — mobile browsers ke liye ZAROORI ─────────────
@app.options("/{full_path:path}")
async def options_handler(full_path: str, request: Request):
    origin = request.headers.get("origin", "*")
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin":  origin,
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age":       "3600",
        }
    )


# ── robots.txt — search crawlers ke liye ─────────────────────────────
@app.get("/robots.txt", response_class=PlainTextResponse)
def robots_txt():
    return "User-agent: *\nDisallow: /admin/\nAllow: /"


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
        "gemini":  bool(os.getenv("GEMINI_API_KEY")),
        "serpapi": bool(os.getenv("SERPAPI_KEY")),
        "db":      "PostgreSQL" if USE_POSTGRES else "SQLite",
        "qdrant":  "Cloud" if os.getenv("QDRANT_URL") else "Local",
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    session_id = request.session_id or str(uuid.uuid4())

    # ── Language resolution — 3 level priority ────────────────────────
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


# ── Evaluation endpoint — Confusion Matrix ───────────────────────────
@app.get("/evaluate")
async def evaluate_chatbot(background_tasks: BackgroundTasks):
    """Run confusion matrix evaluation — results appear in Railway logs."""

    def _run():
        try:
            from rag.kb_query import get_answer, is_out_of_scope
            from language_detector import detect_language

            # ── Test Data ─────────────────────────────────────────────
            LANG_TESTS = [
                ("What are the fees for BTech?",        "en"),
                ("Who is the director of GBPIET?",      "en"),
                ("Tell me about hostel facilities",      "en"),
                ("What courses are offered?",            "en"),
                ("जीबीपीआईईटी की फीस कितनी है?",        "hi"),
                ("निदेशक कौन हैं?",                      "hi"),
                ("हॉस्टल की सुविधाएं क्या हैं?",         "hi"),
                ("CSE के HOD कौन हैं?",                  "hi"),
                ("फीस कति छ?",                           "ga"),
                ("एडमिशन कनकै होंद?",                    "ga"),
                ("हॉस्टल माँ कि सुविधा छ?",              "ga"),
                ("निदेशक को छ?",                          "ga"),
                ("फीस कतु छु?",                          "ku"),
                ("एडमिशन कसि होंछ?",                     "ku"),
                ("हॉस्टल मा कतु सुबिद छ?",               "ku"),
                ("निदेशक को छु?",                         "ku"),
            ]

            SCOPE_TESTS = [
                ("What are the fees for BTech?",         True),
                ("Who is the HOD of CSE?",               True),
                ("admission process kya hai",            True),
                ("placement record GBPIET",              True),
                ("director kaun hai",                    True),
                ("library timing kya hai",               True),
                ("scholarship kaise milti hai",          True),
                ("hostel facility kya hai",              True),
                ("What is the IPL score today?",         False),
                ("Salman Khan ki film kaunsi hai?",      False),
                ("Aaj ka mausam kaisa hai?",             False),
                ("Bitcoin price kya hai?",               False),
                ("Modi ji ke baare mein batao",          False),
                ("Taj Mahal kahan hai?",                 False),
                ("Recipe of biryani",                    False),
                ("Stock market update",                  False),
            ]

            ANS_TESTS = [
                ("What are the fees?",   "en", ["fee", "tuition", "per", "semester", "year"]),
                ("Who is the director?", "en", ["director", "gbpiet", "dr", "prof"]),
                ("hostel kya hai",       "en", ["hostel", "accommodation", "facility"]),
                ("fees kitni hai",       "hi", ["फीस", "शुल्क", "हजार", "प्रति", "लाख"]),
                ("director kaun hai",    "hi", ["निदेशक", "gbpiet", "डॉ", "प्रो"]),
                ("hostel ki jankari",    "hi", ["हॉस्टल", "छात्रावास", "सुविधा"]),
                ("placement record",     "en", ["placement", "company", "package", "lpa"]),
                ("admission process",    "en", ["admission", "jee", "application", "rank"]),
            ]

            # ── Test 1: Language Detection ────────────────────────────
            print("\n[EVAL] ══════════════════════════════════")
            print("[EVAL]   TEST 1: LANGUAGE DETECTION")
            print("[EVAL] ══════════════════════════════════")

            lang_actual    = []
            lang_predicted = []
            lang_correct   = 0

            for q, expected in LANG_TESTS:
                detected = detect_language(q)
                lang_actual.append(expected)
                lang_predicted.append(detected)
                ok = detected == expected
                if ok: lang_correct += 1
                print(f"[EVAL] {'✅' if ok else '❌'} [{expected}→{detected}] {q[:45]}")

            lang_acc = lang_correct / len(LANG_TESTS) * 100
            print(f"[EVAL] Language Accuracy: {lang_acc:.1f}% ({lang_correct}/{len(LANG_TESTS)})")

            # Confusion matrix (text)
            labels = ["en", "hi", "ga", "ku"]
            print("\n[EVAL] Confusion Matrix (Language):")
            print(f"[EVAL] {'':8}", end="")
            for l in labels: print(f"{l:>6}", end="")
            print()
            for actual_l in labels:
                print(f"[EVAL] {actual_l:8}", end="")
                for pred_l in labels:
                    count = sum(1 for a, p in zip(lang_actual, lang_predicted) if a == actual_l and p == pred_l)
                    print(f"{count:>6}", end="")
                print()

            # ── Test 2: Scope Detection ───────────────────────────────
            print("\n[EVAL] ══════════════════════════════════")
            print("[EVAL]   TEST 2: SCOPE DETECTION")
            print("[EVAL] ══════════════════════════════════")

            scope_actual    = []
            scope_predicted = []
            scope_correct   = 0
            tp = fp = tn = fn = 0

            for q, should_in_scope in SCOPE_TESTS:
                is_oos          = is_out_of_scope(q)
                detected_in     = not is_oos
                ok              = detected_in == should_in_scope
                if ok: scope_correct += 1

                exp_label = "in_scope"  if should_in_scope else "out_scope"
                det_label = "in_scope"  if detected_in     else "out_scope"
                scope_actual.append(exp_label)
                scope_predicted.append(det_label)

                if should_in_scope and detected_in:     tp += 1
                elif not should_in_scope and not detected_in: tn += 1
                elif not should_in_scope and detected_in:     fp += 1
                else:                                         fn += 1

                print(f"[EVAL] {'✅' if ok else '❌'} [{exp_label}→{det_label}] {q[:45]}")

            scope_acc = scope_correct / len(SCOPE_TESTS) * 100
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print(f"[EVAL] Scope Accuracy : {scope_acc:.1f}% ({scope_correct}/{len(SCOPE_TESTS)})")
            print(f"[EVAL] Precision      : {precision*100:.1f}%")
            print(f"[EVAL] Recall         : {recall*100:.1f}%")
            print(f"[EVAL] F1 Score       : {f1*100:.1f}%")
            print(f"\n[EVAL] Confusion Matrix (Scope):")
            print(f"[EVAL]              in_scope  out_scope")
            print(f"[EVAL] in_scope  :  {tp:>6}    {fn:>6}")
            print(f"[EVAL] out_scope :  {fp:>6}    {tn:>6}")

            # ── Test 3: Answer Quality ────────────────────────────────
            print("\n[EVAL] ══════════════════════════════════")
            print("[EVAL]   TEST 3: ANSWER QUALITY")
            print("[EVAL] ══════════════════════════════════")

            ans_correct  = 0
            ans_wrong    = 0
            ans_no_answer = 0

            for q, lang, keywords in ANS_TESTS:
                try:
                    answer = get_answer(q, lang)
                    if not answer or len(answer.strip()) < 10:
                        ans_no_answer += 1
                        print(f"[EVAL] ⚠️  [{lang}] {q[:40]} → NO ANSWER")
                        continue
                    found = any(kw.lower() in answer.lower() for kw in keywords)
                    if found:
                        ans_correct += 1
                        print(f"[EVAL] ✅ [{lang}] {q[:40]}")
                    else:
                        ans_wrong += 1
                        print(f"[EVAL] ❌ [{lang}] {q[:40]}")
                        print(f"[EVAL]    Expected: {keywords}")
                        print(f"[EVAL]    Got: {answer[:80]}...")
                except Exception as e:
                    ans_no_answer += 1
                    print(f"[EVAL] ❌ [{lang}] {q[:40]} → ERROR: {e}")

            total_ans = len(ANS_TESTS)
            ans_acc   = ans_correct / total_ans * 100
            print(f"\n[EVAL] Answer Quality : {ans_acc:.1f}% ({ans_correct}/{total_ans})")
            print(f"[EVAL] Correct        : {ans_correct}")
            print(f"[EVAL] Wrong          : {ans_wrong}")
            print(f"[EVAL] No Answer      : {ans_no_answer}")
            print(f"\n[EVAL] Confusion Matrix (Answer):")
            print(f"[EVAL]            correct  wrong  no_answer")
            print(f"[EVAL] correct :  {ans_correct:>6}  {ans_wrong:>5}  {ans_no_answer:>8}")

            # ── Final Summary ─────────────────────────────────────────
            overall = (lang_acc + scope_acc + ans_acc) / 3
            print("\n[EVAL] ══════════════════════════════════")
            print("[EVAL]   FINAL SUMMARY")
            print("[EVAL] ══════════════════════════════════")
            print(f"[EVAL] Language Detection : {lang_acc:.1f}%")
            print(f"[EVAL] Scope Detection    : {scope_acc:.1f}%  (P:{precision*100:.0f}% R:{recall*100:.0f}% F1:{f1*100:.0f}%)")
            print(f"[EVAL] Answer Quality     : {ans_acc:.1f}%")
            print(f"[EVAL] ─────────────────────────────────")
            print(f"[EVAL] Overall Score      : {overall:.1f}%")
            print("[EVAL] ✅ Evaluation complete!")

        except Exception as e:
            print(f"[EVAL] ❌ Evaluation failed: {e}")
            traceback.print_exc()

    background_tasks.add_task(_run)
    return {
        "message": "✅ Evaluation started — check Railway logs for results",
        "tip":     "Results will appear in logs within 2-3 minutes",
        "tests": {
            "language_detection": "16 questions (en/hi/ga/ku)",
            "scope_detection":    "16 questions (in_scope/out_scope)",
            "answer_quality":     "8 questions (correct/wrong/no_answer)",
        }
    }


@app.get("/admin/visits")
def get_visit_stats():
    return {"message": "Visit stats not configured in this version"}


@app.get("/admin/groq-stats")
def groq_stats():
    try:
        from rag.groq_manager import get_stats
        return get_stats()
    except Exception as e:
        return {"error": str(e)}


@app.post("/admin/clear-cache")
def clear_groq_cache():
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
