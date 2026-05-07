# rag/kb_query.py — Complete RAG pipeline
import os
import json
import glob
import re
import asyncio
import concurrent.futures

from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

from qdrant_setup import get_client, COLLECTIONS
from intent_detector import get_collection_for_query
from rag.hybrid_search import multi_collection_search
from rag.bm25_search import bm25_search, build_bm25_index
from rag.fusion import reciprocal_rank_fusion
from rag.internet_search import search_college_website

_groq1 = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
_groq2_key = os.getenv("GROQ_API_KEY_2", "")
_groq2 = Groq(api_key=_groq2_key) if _groq2_key else None

_embed_model = None
_qa_database = []

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Greeting / Identity responses ─────────────────────────────────────
GREETINGS = {"hello", "hi", "hlo", "hey", "hii", "helo", "namaste", "नमस्ते", "हेलो", "हाय"}
IDENTITY_Q = {"who are you", "what are you", "who r u", "tum kaun ho", "aap kaun hain", "aap kaun ho", "kaun ho tum"}

GREETING_RESPONSE = {
    "en": "Hello! I'm Diksha 👋, the official AI assistant for GBPIET. You can ask me about admissions, fees, hostel, placements, faculty, courses and more!",
    "hi": "नमस्ते! मैं दीक्षा हूँ 👋 — GBPIET की आधिकारिक AI सहायक। आप मुझसे admission, fees, hostel, placement के बारे में पूछ सकते हैं।",
    "ga": "नमस्ते! मैं दीक्षा छू — GBPIET की AI सहायक।",
    "ku": "नमस्ते! मैं दीक्षा छु — GBPIET की AI सहायक।",
}

IDENTITY_RESPONSE = {
    "en": "I'm Diksha 🎓, the official AI chatbot for GBPIET (Govind Ballabh Pant Institute of Engineering & Technology), Pauri Garhwal, Uttarakhand. I can help you with college information in English, Hindi, Garhwali and Kumauni!",
    "hi": "मैं दीक्षा हूँ 🎓 — GBPIET (गोविंद बल्लभ पंत इंजीनियरिंग कॉलेज), पौड़ी गढ़वाल की आधिकारिक AI chatbot। मैं आपकी मदद हिंदी, अंग्रेज़ी, गढ़वाली और कुमाउनी में कर सकती हूँ!",
    "ga": "मैं दीक्षा छू — GBPIET की AI chatbot।",
    "ku": "मैं दीक्षा छु — GBPIET की AI chatbot।",
}


def groq_call(messages, max_tokens=500, temperature=0.3):
    for label, client in [("Key1", _groq1), ("Key2", _groq2)]:
        if client is None:
            continue
        try:
            r = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            print(f"[LLM] ✅ Groq {label} answered")
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"[LLM] Groq {label} failed: {e}")
    return ""


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("[Embed] Loading model...")
        try:
            _embed_model = HuggingFaceEmbeddings(
                model_name=EMBED_MODEL_NAME,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            print("[Embed] ✅ Model loaded")
        except Exception as e:
            print(f"[Embed] ❌ Failed: {e}")
            raise
    return _embed_model


def get_qdrant():
    return get_client()


def load_qa_database():
    global _qa_database
    if _qa_database:
        return _qa_database
    data_folder = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )
    for filepath in sorted(glob.glob(os.path.join(data_folder, "*.json"))):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                answer = item.get("answer", "")
                if not isinstance(answer, str) or not answer.strip():
                    continue
                q_field = item.get("question", "")
                if isinstance(q_field, str):
                    questions = [q_field]
                elif isinstance(q_field, list):
                    questions = [q for q in q_field if isinstance(q, str) and q.strip()]
                else:
                    continue
                for q in questions:
                    if q.strip():
                        _qa_database.append({
                            "question": q.strip(),
                            "answer": answer.strip(),
                            "source": os.path.basename(filepath),
                        })
        except Exception as e:
            print(f"[DB] Error loading {filepath}: {e}")
    print(f"[DB] Loaded {len(_qa_database)} QA pairs")
    return _qa_database


def is_hindi_text(text):
    if not text:
        return False
    devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total = len(text.replace(" ", ""))
    return total > 0 and (devanagari / total) > 0.2


def translate_answer_if_needed(answer, lang, question):
    answer_is_hindi = is_hindi_text(answer)
    if lang == "en" and not answer_is_hindi:
        return answer
    if lang in ("hi", "ga", "ku") and answer_is_hindi:
        return answer
    if lang == "en":
        prompt = f"Translate to English. Return ONLY translated text.\n\n{answer}"
        system = "You are a translator. Translate Hindi to English accurately."
    else:
        prompt = f"Translate to Hindi (Devanagari). Return ONLY translated text.\n\n{answer}"
        system = "You are a translator. Translate English to Hindi accurately."
    print(f"[TRANSLATE] Translating for lang={lang}...")
    result = groq_call(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        max_tokens=400, temperature=0.1,
    )
    if result:
        return result
    if lang == "en" and is_hindi_text(answer):
        return "I found information but couldn't translate it. Please try again."
    return answer


HINDI_MAP = {
    'जीबीपीआईईटी': 'gbpiet', 'संस्थान': 'institute', 'कॉलेज': 'college',
    'पहुँचें': 'reach', 'निदेशक': 'director', 'विभागाध्यक्ष': 'head department hod',
    'अध्यक्ष': 'chairman', 'डीन': 'dean', 'संकाय': 'faculty', 'वार्डन': 'warden',
    'प्रवेश': 'admission', 'फीस': 'fees', 'शुल्क': 'fees', 'छात्रवृत्ति': 'scholarship',
    'हॉस्टल': 'hostel', 'प्लेसमेंट': 'placement', 'पुस्तकालय': 'library',
    'परिवहन': 'transport', 'परिणाम': 'result', 'रैगिंग': 'ragging',
    'संपर्क': 'contact', 'रजिस्ट्रार': 'registrar', 'कुलसचिव': 'registrar',
}


def hi_to_en(text):
    t = text.lower()
    for h, e in HINDI_MAP.items():
        t = t.replace(h, ' ' + e + ' ')
    return re.sub(r'\s+', ' ', t).strip()


SPECIFIC_ROLE_MAP = {
    "dean academic affairs": "dean academic", "dean of academic": "dean academic",
    "dean academics": "dean academic", "dean academic": "dean academic",
    "dean accadmic": "dean academic", "dean student welfare": "dean student welfare",
    "dean of student": "dean student welfare", "dean student": "dean student welfare",
    "dean research": "dean research", "dean planning": "dean planning",
    "dean faculty welfare": "dean faculty welfare", "dean faculty": "dean faculty welfare",
    "hod of cse": "hod cse", "hod cse": "hod cse",
    "hod of ece": "hod ece", "hod ece": "hod ece",
    "hod of me": "hod mechanical", "hod me": "hod mechanical",
    "hod of mechanical": "hod mechanical", "hod mechanical": "hod mechanical",
    "hod of civil": "hod civil", "hod civil": "hod civil",
    "hod of ee": "hod electrical", "hod ee": "hod electrical",
    "hod of electrical": "hod electrical", "hod electrical": "hod electrical",
    "hod of mca": "hod mca", "hod mca": "hod mca",
    "hod of csa": "hod mca", "hod csa": "hod mca",
    "hod of biotech": "hod biotechnology", "hod biotech": "hod biotechnology",
    "hod of biotechnology": "hod biotechnology", "hod biotechnology": "hod biotechnology",
    "warden of kailash": "warden kailash", "warden kailash": "warden kailash",
    "warden of trishul": "warden trishul", "warden trishul": "warden trishul",
    "warden of neelkanth": "warden neelkanth", "warden neelkanth": "warden neelkanth",
    "warden of vh": "warden viswerwarya", "warden vh": "warden viswerwarya",
    "warden of viswerwarya": "warden viswerwarya", "warden viswerwarya": "warden viswerwarya",
    "warden of raman": "warden raman", "warden raman": "warden raman",
    "warden of bhagirathi": "warden bhagirathi", "warden bhagirathi": "warden bhagirathi",
    "warden of rudra": "warden rudra", "warden rudra": "warden rudra",
    "warden of badri": "warden badri", "warden badri": "warden badri",
    "warden of kedar": "warden kedar", "warden kedar": "warden kedar",
    "warden of alaknanda": "warden alaknanda", "warden alaknanda": "warden alaknanda",
    "warden of shivalik": "warden shivalik", "warden shivalik": "warden shivalik",
    "priti dimri": "hod mca", "prof priti dimri": "hod mca",
}


def specific_role_answer(question):
    q_clean = re.sub(r'[^\w\s]', '', question.strip().lower()).strip()
    mapped = None
    for phrase, topic in sorted(SPECIFIC_ROLE_MAP.items(), key=lambda x: -len(x[0])):
        if phrase in q_clean:
            mapped = topic
            print(f"[ROLE] '{q_clean}' → topic '{mapped}'")
            break
    if not mapped:
        return None
    topic_words = mapped.lower().split()
    best_score, best_ans = 0, None
    for item in load_qa_database():
        q_lower = item["question"].lower()
        a_lower = item["answer"].lower()
        score = 0
        if all(w in q_lower for w in topic_words): score += 3
        elif mapped.lower() in q_lower: score += 2
        elif any(w in q_lower for w in topic_words): score += 1
        if any(w in a_lower for w in topic_words): score += 1
        if score > best_score:
            best_score = score
            best_ans = item["answer"]
    if best_ans:
        print(f"[ROLE] ✅ score={best_score}")
    return best_ans


DIRECT_KEYWORD_MAP = {
    "registrar": "registrar", "director": "director", "dean": "dean",
    "chairman": "chairman", "warden": "warden", "placement": "placement",
    "placements": "placement", "hostel": "hostel", "hostels": "hostel",
    "fees": "fees", "fee": "fees", "admission": "admission", "admissions": "admission",
    "contact": "contact", "courses": "courses", "course": "courses",
    "library": "library", "transport": "transport", "scholarship": "scholarship",
    "result": "result", "ragging": "ragging", "sports": "sports",
    "faculty": "faculty", "hod": "head of department", "about": "about gbpiet",
    "रजिस्ट्रार": "registrar", "निदेशक": "director", "डीन": "dean",
    "प्लेसमेंट": "placement", "हॉस्टल": "hostel", "फीस": "fees",
    "प्रवेश": "admission", "संपर्क": "contact", "पुस्तकालय": "library",
    "परिवहन": "transport", "रैगिंग": "ragging", "संकाय": "faculty",
}


def direct_keyword_answer(question):
    q_clean = question.strip().lower()
    if len(q_clean.split()) > 2:
        return None
    first_word = q_clean.split()[0] if q_clean.split() else ""
    mapped = DIRECT_KEYWORD_MAP.get(q_clean) or DIRECT_KEYWORD_MAP.get(first_word)
    if not mapped:
        qt = hi_to_en(q_clean)
        mapped = DIRECT_KEYWORD_MAP.get(qt.strip()) or DIRECT_KEYWORD_MAP.get(qt.split()[0] if qt.split() else "")
    if not mapped:
        return None
    print(f"[DIRECT_KW] '{q_clean}' → '{mapped}'")
    mapped_lower = mapped.lower()
    best_score, best_ans = 0, None
    for item in load_qa_database():
        score = 0
        if mapped_lower in item["question"].lower(): score += 2
        if mapped_lower in item["answer"].lower(): score += 1
        if score > best_score:
            best_score = score
            best_ans = item["answer"]
    return best_ans


def exact_match(question):
    q = question.strip().lower()
    for item in load_qa_database():
        if q == item["question"].strip().lower():
            print(f"[EXACT] {item['question'][:60]}")
            return item["answer"]
    return None


STOP = {
    'what','who','is','are','the','at','in','of','a','an','and','or','for','to',
    'how','does','do','has','have','many','which','tell','me','about','please',
    'can','you','i','my','their','kya','hai','hain','ka','ki','ke','mein','se',
    'क्या','कौन','का','की','के','में','से','है','हैं','और','या','को','ने',
    'मैं','हम','आप','वे','इस','उस','यह','वह','पर','कैसे','कहाँ',
}
HOSTEL_NAMES = {
    'kailash','neelkanth','kedar','rudra','badri','alaknanda',
    'shivalik','trishul','raman','bhagirathi','viswerwarya','vh'
}


def get_keywords(text):
    words = set(re.findall(r'[\u0900-\u097F]+|[a-zA-Z0-9]+', text.lower()))
    translated = set(re.findall(r'[a-zA-Z0-9]+', hi_to_en(text)))
    return (words | translated) - STOP


def keyword_match(question, threshold=2):
    q_kw = get_keywords(question.lower())
    specific_hostel = q_kw & HOSTEL_NAMES
    if not q_kw:
        return None
    best_score, best_ans = 0.0, None
    for item in load_qa_database():
        s_kw = get_keywords(item["question"].lower())
        matches = len(q_kw & s_kw)
        score = matches / max(len(q_kw), len(s_kw), 1)
        if specific_hostel:
            if not (specific_hostel & (s_kw & HOSTEL_NAMES)):
                continue
        if matches >= threshold and score > best_score:
            best_score = score
            best_ans = item["answer"]
    return best_ans


async def rag_search_async(question, lang="en"):
    """Full RAG pipeline: BM25 + vector search + internet fallback."""
    sources = []
    used_internet = False
    try:
        # ── BM25 search ───────────────────────────────────
        bm25_results = bm25_search(query=question, top_k=5)

        # ── Vector search ─────────────────────────────────
        collections = get_collection_for_query(question, lang)
        if "website" not in collections:
            collections.append("website")

        vector = get_embed_model().embed_query(question)
        lang_filter = lang if lang in ("en", "hi") else None
        vector_results = multi_collection_search(
            client=get_client(), collections=collections,
            query_vector=vector, query_text=question,
            limit=5, lang_filter=lang_filter,
        )

        # ── Fusion ────────────────────────────────────────
        merged = reciprocal_rank_fusion(
            bm25_results=bm25_results, vector_results=vector_results,
            bm25_weight=0.4, vector_weight=0.6,
        )

        ctx_parts = []
        for r in merged[:3]:
            url = r.get("url") or r.get("metadata", {}).get("source", "")
            if url and url.startswith("http"):
                sources.append(url)
            ctx_parts.append(f"[Score: {r['rrf_score']:.3f}]\n{r['text']}")

        # ── Internet fallback if low confidence ───────────
        top_score = merged[0]["rrf_score"] if merged else 0
        if top_score < 0.05 or not merged:
            print(f"[RAG] Low score ({top_score:.3f}) — trying internet search...")
            internet_results = search_college_website(question)
            if internet_results:
                used_internet = True
                print(f"[RAG] ✅ Internet search returned {len(internet_results)} results")
                for r in internet_results[:2]:
                    ctx_parts.append(f"[Web]\n{r['snippet']}\nSource: {r['url']}")
                    sources.append(r["url"])

        if not ctx_parts:
            return {"context": None, "sources": [], "used_internet": False}

        return {
            "context": "\n\n---\n\n".join(ctx_parts),
            "sources": list(dict.fromkeys(sources)),
            "used_internet": used_internet,
        }

    except Exception as e:
        print(f"[RAG] Error: {e}")
        return {"context": None, "sources": [], "used_internet": False}


def rag_search(question, lang="en"):
    """
    Run rag_search_async safely from a sync context.
    Fixes 'There is no current event loop in thread' error on Railway.
    """
    try:
        # Always create a fresh event loop — never reuse FastAPI's loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(rag_search_async(question, lang))
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    except Exception as e:
        print(f"[RAG] error: {e}")
        result = {}
    return result.get("context")


def build_prompt(question, context, lang, history=""):
    if lang == "hi":
        return f"Aap दीक्षा (Diksha) hain — GBPIET chatbot.\nHindi mein jawab dein.\n{history}\nContext:\n{context}\nSawaal: {question}\nJawab:"
    elif lang == "ga":
        return f"Tu दीक्षा (Diksha) chhe — GBPIET chatbot. Garhwali mein jawab de.\n{history}\nContext: {context}\nSawaal: {question}\nJawab:"
    elif lang == "ku":
        return f"Tu दीक्षा (Diksha) chhu — GBPIET chatbot. Kumauni mein jawab de.\n{history}\nContext: {context}\nSawaal: {question}\nJawab:"
    else:
        return f"""You are दीक्षा (Diksha) — AI assistant for GBPIET.
Answer in ENGLISH ONLY. Use ONLY the context below.
If not found: "I'm sorry, I couldn't find that information."

{history}
Context:
{context}

Question: {question}
Answer:"""


def llm_answer(question, context, lang, history=""):
    prompt = build_prompt(question, context, lang, history)
    result = groq_call(
        messages=[
            {"role": "system", "content": "You are Diksha, helpful AI assistant for GBPIET."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=500, temperature=0.3,
    )
    return result if result else "I'm sorry, I couldn't generate a response right now."


def get_answer(question, lang="en", history=""):
    question = question.strip()

    # ── Greeting handler ──────────────────────────────────
    if question.lower().strip() in GREETINGS:
        print("[RESULT] Greeting")
        return GREETING_RESPONSE.get(lang, GREETING_RESPONSE["en"])

    # ── Identity handler ──────────────────────────────────
    if question.lower().strip() in IDENTITY_Q:
        print("[RESULT] Identity")
        return IDENTITY_RESPONSE.get(lang, IDENTITY_RESPONSE["en"])

    print(f"\n{'='*55}\n[Q/{lang}] {question}\n{'='*55}")

    # ── Role-specific match ───────────────────────────────
    ans = specific_role_answer(question)
    if ans:
        print("[RESULT] Specific role match")
        return translate_answer_if_needed(ans, lang, question)

    # ── Direct keyword match ──────────────────────────────
    ans = direct_keyword_answer(question)
    if ans:
        print("[RESULT] Direct keyword")
        return translate_answer_if_needed(ans, lang, question)

    # ── Exact match ───────────────────────────────────────
    ans = exact_match(question)
    if ans:
        print("[RESULT] Exact match")
        return translate_answer_if_needed(ans, lang, question)

    # ── Keyword match ─────────────────────────────────────
    word_count = len(question.split())
    thresh = 1 if word_count <= 2 else (2 if word_count <= 5 else 3)
    ans = keyword_match(question, thresh)
    if ans:
        print("[RESULT] Keyword match")
        return translate_answer_if_needed(ans, lang, question)

    # ── RAG + LLM (with internet fallback inside) ─────────
    ctx = rag_search(question, lang)
    if ctx:
        print("[RESULT] RAG + LLM")
        return llm_answer(question, ctx, lang, history)

    # ── No match ──────────────────────────────────────────
    print("[RESULT] No match")
    fb = {
        "hi": "माफ़ करें, मैं आपकी क्वेरी समझ नहीं पाई। कृपया अधिक जानकारी के लिए GBPIET की वेबसाइट देखें: https://gbpiet.ac.in",
        "ga": "माफ करा, मी तैं त्वे सवाल समझ नि ऐ।",
        "ku": "माफ करिया! म्यर पास तस के जानकारी न्है़ंं!",
        "en": "I'm sorry, I couldn't find information about that. Please visit https://gbpiet.ac.in or call 01368-228030 for more details.",
    }
    return fb.get(lang, fb["en"])
