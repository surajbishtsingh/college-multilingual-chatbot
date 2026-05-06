# kb_query.py — Groq-only (4 API keys, 8-attempt fallback chain)
import os
import json
import glob
import re
import asyncio
import unicodedata
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Imports ───────────────────────────────────────────────────────────
from qdrant_setup import get_client, COLLECTIONS
from intent_detector import get_collection_for_query
from rag.hybrid_search import multi_collection_search
from rag.bm25_search import bm25_search, build_bm25_index
from rag.fusion import reciprocal_rank_fusion
from rag.internet_search import search_college_website
from rag.reranker import rerank_with_diversity

# ══════════════════════════════════════════════════════════════════════
# GROQ CLIENTS — 4 API keys
# Add keys to .env:
#   GROQ_API_KEY   = "gsk_..."   ← Key 1 (primary)
#   GROQ_API_KEY_2 = "gsk_..."   ← Key 2
#   GROQ_API_KEY_3 = "gsk_..."   ← Key 3
#   GROQ_API_KEY_4 = "gsk_..."   ← Key 4
# ══════════════════════════════════════════════════════════════════════
def _make_client(env_var: str) -> Groq | None:
    key = os.getenv(env_var)
    if key:
        print(f"[GROQ] {env_var} ✅ loaded")
        return Groq(api_key=key)
    print(f"[GROQ] {env_var} ❌ not set")
    return None

groq_client_1 = _make_client("GROQ_API_KEY")
groq_client_2 = _make_client("GROQ_API_KEY_2")
groq_client_3 = _make_client("GROQ_API_KEY_3")
groq_client_4 = _make_client("GROQ_API_KEY_4")

# Models — two models available on Groq
GROQ_PRIMARY  = "llama-3.3-70b-versatile"   # best quality
GROQ_FALLBACK = "llama3-70b-8192"           # backup model

# ── Fallback chain: 8 attempts across 4 keys × 2 models ──────────────
# Logic: try primary model on each key first,
#        then fallback model on each key.
# This maximises rate-limit resilience.
GROQ_ATTEMPTS = [
    # (client,        model,         label)
    (groq_client_1, GROQ_PRIMARY,  "Key1 / Primary model"),
    (groq_client_2, GROQ_PRIMARY,  "Key2 / Primary model"),
    (groq_client_3, GROQ_PRIMARY,  "Key3 / Primary model"),
    (groq_client_4, GROQ_PRIMARY,  "Key4 / Primary model"),
    (groq_client_1, GROQ_FALLBACK, "Key1 / Fallback model"),
    (groq_client_2, GROQ_FALLBACK, "Key2 / Fallback model"),
    (groq_client_3, GROQ_FALLBACK, "Key3 / Fallback model"),
    (groq_client_4, GROQ_FALLBACK, "Key4 / Fallback model"),
]

_embed_model = None
_qa_database = []

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


# ══════════════════════════════════════════════════════════════════════
# EMBEDDING MODEL
# ══════════════════════════════════════════════════════════════════════
def get_embed_model() -> HuggingFaceEmbeddings:
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("[Embed] Model loaded")
    return _embed_model


def get_qdrant():
    return get_client()


# ══════════════════════════════════════════════════════════════════════
# QA DATABASE — loads all JSON files into memory
# ══════════════════════════════════════════════════════════════════════
def load_qa_database() -> list[dict]:
    global _qa_database
    if _qa_database:
        return _qa_database

    data_folder = os.path.join(os.path.dirname(__file__), "..", "data")
    data_folder = os.path.normpath(data_folder)

    for filepath in sorted(glob.glob(os.path.join(data_folder, "*.json"))):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue

                a = item.get("answer", "")
                if not isinstance(a, str) or not a.strip():
                    continue

                q_field = item.get("question", "")

                if isinstance(q_field, str):
                    questions = [q_field]
                elif isinstance(q_field, list):
                    questions = [q for q in q_field if isinstance(q, str) and q.strip()]
                else:
                    continue

                for q in questions:
                    if not q.strip():
                        continue
                    _qa_database.append({
                        "question": q.strip(),
                        "answer":   a.strip(),
                        "source":   os.path.basename(filepath),
                        "lang":     item.get("lang", "").strip().lower(),
                    })

        except Exception as e:
            print(f"[DB] Error loading {filepath}: {e}")

    print(f"[DB] Loaded {len(_qa_database)} QA pairs")
    return _qa_database


# ══════════════════════════════════════════════════════════════════════
# GARHWALI MARKERS — module-level constant reused everywhere
# ══════════════════════════════════════════════════════════════════════
GA_MARKERS = [
    'छन', 'छ।', 'हूँद', 'कुण', 'मिलद', 'पैलू', 'अर',
    'कनकै', 'कख',    'बटि',    'त्वै',   'छौ',    'छी',
    'छ्यायी', 'थ्यायी', 'तुमुं', 'यैसैं', 'वैसें',
    'माँ',    'मी',   'जु',    'यु',    'वु',
]


# ══════════════════════════════════════════════════════════════════════
# LANGUAGE HELPERS
# ══════════════════════════════════════════════════════════════════════
def is_hindi_text(text: str) -> bool:
    if not text:
        return False
    devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total      = len(text.replace(" ", ""))
    return total > 0 and (devanagari / total) > 0.2


def detect_answer_lang(answer_text: str, item_lang: str = "") -> str:
    """Detect language of an answer — item lang tag takes priority."""
    if item_lang in ("ga", "garhwali"):
        return "ga"
    if item_lang in ("hi", "hindi"):
        return "hi"
    if item_lang == "en":
        return "en"
    if any(m in answer_text for m in GA_MARKERS):
        return "ga"
    latin_chars = sum(1 for c in answer_text if c.isascii() and c.isalpha())
    total_chars = len(answer_text.replace(" ", ""))
    if total_chars > 0 and (latin_chars / total_chars) > 0.5:
        return "en"
    devanagari = sum(1 for c in answer_text if '\u0900' <= c <= '\u097F')
    return "hi" if devanagari > 5 else "en"


# ══════════════════════════════════════════════════════════════════════
# GROQ CALL HELPERS
# ══════════════════════════════════════════════════════════════════════
def _groq_call(
    client: Groq,
    model:  str,
    system: str,
    prompt: str,
    max_tokens:  int,
    temperature: float,
) -> str | None:
    """Single Groq API call. Returns text or None on failure."""
    if client is None:
        return None
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GROQ] {model} failed: {e}")
        return None


def _groq_with_fallback(
    system:      str,
    prompt:      str,
    max_tokens:  int   = 400,
    temperature: float = 0.1,
) -> str | None:
    """
    Try all 8 attempts in order (4 keys × 2 models).
    Returns first successful response, or None if all fail.
    """
    for client, model, label in GROQ_ATTEMPTS:
        result = _groq_call(client, model, system, prompt, max_tokens, temperature)
        if result:
            print(f"[GROQ] ✅ {label}")
            return result
    print("[GROQ] ❌ All 8 attempts failed")
    return None


# ══════════════════════════════════════════════════════════════════════
# TRANSLATE
# ══════════════════════════════════════════════════════════════════════
def translate_answer_if_needed(answer: str, lang: str, question: str) -> str:
    answer_is_hindi = is_hindi_text(answer)

    if lang == "en" and not answer_is_hindi:
        return answer
    if lang in ("hi", "ga", "ku") and answer_is_hindi:
        return answer

    if lang == "en":
        prompt = (
            f"Translate the following text to English. "
            f"Keep names, numbers, and URLs unchanged. "
            f"Return ONLY the translated text.\n\nText: {answer}"
        )
        system = (
            "You are a translator. Translate Hindi/Devanagari text to English accurately. "
            "Keep proper nouns, numbers, and URLs unchanged."
        )
    else:
        prompt = (
            f"Translate the following text to Hindi (Devanagari script). "
            f"Keep names, numbers, and URLs unchanged. "
            f"Return ONLY the translated text.\n\nText: {answer}"
        )
        system = (
            "You are a translator. Translate English text to Hindi accurately. "
            "Keep proper nouns, numbers, and URLs unchanged."
        )

    print(f"[TRANSLATE] Lang mismatch — translating...")
    translated = _groq_with_fallback(system, prompt, max_tokens=400, temperature=0.1)
    return translated if translated else answer


# ══════════════════════════════════════════════════════════════════════
# HINDI → ENGLISH MAP
# ══════════════════════════════════════════════════════════════════════
HINDI_MAP = {
    # Hindi
    'जीबीपीआईईटी': 'gbpiet',    'जीबीपीईटी': 'gbpiet',
    'संस्थान': 'institute',      'कॉलेज': 'college',
    'पहुँचें': 'reach',           'पहुंचें': 'reach',
    'कैसे': 'how',                'रास्ता': 'route direction',
    'पता': 'address',             'कहाँ': 'where',
    'निदेशक': 'director',         'विभागाध्यक्ष': 'head department hod',
    'अध्यक्ष': 'chairman',        'डीन': 'dean',
    'शिक्षक': 'faculty',          'प्राध्यापक': 'professor faculty',
    'संकाय': 'faculty',           'वार्डन': 'warden',
    'प्रवेश': 'admission',        'दाखिला': 'admission',
    'आवेदन': 'apply',             'पात्रता': 'eligibility',
    'कोर्स': 'courses',           'शाखा': 'branch',
    'फीस': 'fees',                'शुल्क': 'fees',
    'छात्रवृत्ति': 'scholarship', 'हॉस्टल': 'hostel',
    'छात्रावास': 'hostel',        'लड़कियों': 'girls',
    'लड़कों': 'boys',             'प्रथम वर्ष': 'first year',
    'प्लेसमेंट': 'placement',     'पैकेज': 'package',
    'पुस्तकालय': 'library',       'खेल': 'sports',
    'परिवहन': 'transport',         'परिणाम': 'result',
    'परीक्षा': 'exam',             'रैगिंग': 'ragging',
    'संपर्क': 'contact',           'फोन': 'phone',
    'रजिस्ट्रार': 'registrar',     'कुलसचिव': 'registrar',
    # Garhwali
    'कु':       'who what of',   'कू':       'who',
    'कन':       'how',           'कनकै':     'how',
    'कख':       'where',         'कनै':      'where',
    'बटि':      'from',          'कुण':      'to for',
    'अर':       'and',           'त्वै':     'then',
    'माँ':      'in',            'मा':       'in',
    'बारे माँ': 'about',
    'मी':       'i me',          'आम':       'we us',
    'तुमुं':    'you',           'जु':       'who which',
    'यु':       'this',          'वु':       'that they',
    'यैसैं':    'this',          'वैसें':    'that',
    'वैन':      'that',
    'च':        'is',            'छ':        'is',
    'छन':       'are is',        'छौ':       'was',
    'थौ':       'was',           'छी':       'was',
    'छ्यायी':  'were',          'थ्यायी':  'were',
}


def hi_to_en(text: str) -> str:
    t = text.lower()
    for h, e in HINDI_MAP.items():
        t = t.replace(h, ' ' + e + ' ')
    return re.sub(r'\s+', ' ', t).strip()


# ══════════════════════════════════════════════════════════════════════
# STEP 0a — SPECIFIC ROLE MAP
# ══════════════════════════════════════════════════════════════════════
SPECIFIC_ROLE_MAP = {
    "dean academic affairs":  "dean academic",
    "dean of academic":       "dean academic",
    "dean academics":         "dean academic",
    "dean academic":          "dean academic",
    "dean accadmic":          "dean academic",
    "dean acadmic":           "dean academic",
    "dean student welfare":   "dean student welfare",
    "dean of student":        "dean student welfare",
    "dean student":           "dean student welfare",
    "dean welfare":           "dean student welfare",
    "dean research":          "dean research",
    "dean of research":       "dean research",
    "dean planning":          "dean planning",
    "dean faculty welfare":   "dean faculty welfare",
    "dean faculty":           "dean faculty welfare",
    "hod of cse":             "hod cse",
    "hod cse":                "hod cse",
    "hod of ece":             "hod ece",
    "hod ece":                "hod ece",
    "hod of me":              "hod mechanical",
    "hod me":                 "hod mechanical",
    "hod of mechanical":      "hod mechanical",
    "hod mechanical":         "hod mechanical",
    "hod of civil":           "hod civil",
    "hod civil":              "hod civil",
    "hod of ee":              "hod electrical",
    "hod ee":                 "hod electrical",
    "hod of electrical":      "hod electrical",
    "hod electrical":         "hod electrical",
    "hod of mca":             "hod mca",
    "hod mca":                "hod mca",
    "hod of csa":             "hod mca",
    "hod csa":                "hod mca",
    "hod of biotech":         "hod biotechnology",
    "hod biotech":            "hod biotechnology",
    "hod of biotechnology":   "hod biotechnology",
    "hod biotechnology":      "hod biotechnology",
    "warden of kailash":      "warden kailash",
    "warden kailash":         "warden kailash",
    "warden of trishul":      "warden trishul",
    "warden trishul":         "warden trishul",
    "warden of neelkanth":    "warden neelkanth",
    "warden neelkanth":       "warden neelkanth",
    "warden of vh":           "warden viswerwarya",
    "warden vh":              "warden viswerwarya",
    "warden of viswerwarya":  "warden viswerwarya",
    "warden viswerwarya":     "warden viswerwarya",
    "warden of raman":        "warden raman",
    "warden raman":           "warden raman",
    "warden of bhagirathi":   "warden bhagirathi",
    "warden bhagirathi":      "warden bhagirathi",
    "warden of rudra":        "warden rudra",
    "warden rudra":           "warden rudra",
    "warden of badri":        "warden badri",
    "warden badri":           "warden badri",
    "warden of kedar":        "warden kedar",
    "warden kedar":           "warden kedar",
    "warden of alaknanda":    "warden alaknanda",
    "warden alaknanda":       "warden alaknanda",
    "warden of shivalik":     "warden shivalik",
    "warden shivalik":        "warden shivalik",
}

TYPO_MAP = {
    "mechenical":    "mechanical",   "mechnical":     "mechanical",
    "mechincal":     "mechanical",   "mechanicle":    "mechanical",
    "mechinical":    "mechanical",   "electical":     "electrical",
    "electrcal":     "electrical",   "biotechonlogy": "biotechnology",
    "bitoech":       "biotechnology","admision":      "admission",
    "admisson":      "admission",    "palcement":     "placement",
    "hostle":        "hostel",       "dircetor":      "director",
    "registar":      "registrar",
}

def fix_typos(text: str) -> str:
    words = text.lower().split()
    return " ".join(TYPO_MAP.get(w, w) for w in words)


def specific_role_answer(question: str, preferred_lang: str = "en") -> str | None:
    q_fixed = fix_typos(question)
    q_clean = re.sub(r'[^\w\s]', '', q_fixed.strip().lower()).strip()

    mapped = None
    for phrase, topic in sorted(SPECIFIC_ROLE_MAP.items(), key=lambda x: -len(x[0])):
        if phrase in q_clean:
            mapped = topic
            print(f"[ROLE] '{q_clean}' → topic '{mapped}'")
            break

    if not mapped:
        return None

    topic_words = mapped.lower().split()
    BLACKLIST   = [
        "list all", "all hod", "all department", "all heads",
        "departments at gbpiet", "list of hod", "all hods",
        "सभी विभाग", "सभी hod", "सभी hods"
    ]

    best_score, best_ans = 0, None
    for item in load_qa_database():
        q_lower = item["question"].lower()
        a_lower = item["answer"].lower()
        if any(p in q_lower for p in BLACKLIST):
            continue
        score = 0
        if all(w in q_lower for w in topic_words):      score += 3
        elif mapped.lower() in q_lower:                  score += 2
        elif any(w in q_lower for w in topic_words):     score += 1
        if any(w in a_lower for w in topic_words):       score += 1
        if score > best_score:
            best_score = score
            best_ans   = item["answer"]
            print(f"[ROLE] ✅ score={best_score} q={item['question'][:60]}")

    return best_ans


# ══════════════════════════════════════════════════════════════════════
# STEP 0b — DIRECT KEYWORD MAP
# ══════════════════════════════════════════════════════════════════════
DIRECT_KEYWORD_MAP = {
    "registrar":    "registrar",   "registrar?":   "registrar",
    "director":     "director",    "dean":         "dean",
    "chairman":     "chairman",    "warden":       "warden",
    "placement":    "placement",   "placements":   "placement",
    "hostel":       "hostel",      "hostels":      "hostel",
    "fees":         "fees",        "fee":          "fees",
    "admission":    "admission",   "admissions":   "admission",
    "contact":      "contact",     "courses":      "courses",
    "course":       "courses",     "library":      "library",
    "transport":    "transport",   "scholarship":  "scholarship",
    "result":       "result",      "ragging":      "ragging",
    "sports":       "sports",      "faculty":      "faculty",
    "hod":          "head of department",
    "about":        "about gbpiet",
    "रजिस्ट्रार":  "registrar",   "कुलसचिव":     "registrar",
    "निदेशक":       "director",   "डीन":          "dean",
    "प्लेसमेंट":    "placement",  "हॉस्टल":       "hostel",
    "फीस":          "fees",       "प्रवेश":       "admission",
    "संपर्क":       "contact",    "पुस्तकालय":    "library",
    "परिवहन":       "transport",  "छात्रवृत्ति":  "scholarship",
    "रैगिंग":       "ragging",    "संकाय":        "faculty",
}


def direct_keyword_answer(question: str, preferred_lang: str = "en") -> str | None:
    q_clean    = question.strip().lower()
    word_count = len(q_clean.split())

    if word_count > 2:
        return None

    first_word = q_clean.split()[0] if q_clean.split() else ""
    mapped     = (
        DIRECT_KEYWORD_MAP.get(q_clean)
        or DIRECT_KEYWORD_MAP.get(first_word)
    )
    if not mapped:
        qt     = hi_to_en(q_clean)
        mapped = (
            DIRECT_KEYWORD_MAP.get(qt.strip())
            or DIRECT_KEYWORD_MAP.get(qt.split()[0] if qt.split() else "")
        )
    if not mapped:
        return None

    print(f"[DIRECT_KW] '{q_clean}' → topic '{mapped}'")
    mapped_lower = mapped.lower()
    candidates   = []

    for item in load_qa_database():
        score = 0
        if mapped_lower in item["question"].lower(): score += 2
        if mapped_lower in item["answer"].lower():   score += 1
        if score > 0:
            candidates.append({
                "answer": item["answer"],
                "score":  score,
                "lang":   detect_answer_lang(item["answer"], item.get("lang", "")),
            })

    if not candidates:
        return None

    candidates.sort(
        key=lambda c: (1 if c["lang"] == preferred_lang else 0, c["score"]),
        reverse=True
    )
    best = candidates[0]
    print(f"[DIRECT_KW] ✅ score={best['score']} lang={best['lang']}")
    return best["answer"]


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — EXACT MATCH
# ══════════════════════════════════════════════════════════════════════
def exact_match(question: str) -> str | None:
    q = question.strip().lower()
    for item in load_qa_database():
        if q == item["question"].strip().lower():
            print(f"[EXACT] {item['question'][:60]}")
            return item["answer"]
    return None


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — KEYWORD MATCH
# ══════════════════════════════════════════════════════════════════════
STOP = {
    # English
    'what','who','is','are','the','at','in','of','a','an','and','or',
    'for','to','how','does','do','has','have','many','which','tell',
    'me','about','please','can','you','i','my','their','kya','hai',
    'hain','ka','ki','ke','mein','se','per','ek',
    # Hindi
    'क्या','कौन','का','की','के','में','से','है','हैं','एक',
    'और','या','को','ने','था','थी','थे','कि','जो','तो','भी',
    'मैं','हम','आप','वे','इस','उस','यह','वह','पर','बारे',
    'कैसे','कहाँ','कहां','तक',
    # Garhwali
    'कु','कू','कि','कन','कनकै','कख','कनै',
    'माँ','मा','बटि','च','छ','छन','एक',
    'अर','त','त्वै','भी','न',
    'थौ','छौ','छी','छ्यायी','थ्यायी',
    'मी','आम','तुम','तुमुं','वु','वे',
    'यैसैं','यें','वैसें','वैन','यु',
    'पर','बारे','तक','जु',
}

HOSTEL_NAMES = {
    'kailash','neelkanth','kedar','rudra','badri','alaknanda',
    'shivalik','trishul','raman','bhagirathi','viswerwarya','vh',
}

def get_keywords(text: str) -> set:
    words      = set(re.findall(r'[\u0900-\u097F]+|[a-zA-Z0-9]+', text.lower()))
    translated = set(re.findall(r'[a-zA-Z0-9]+', hi_to_en(text)))
    return (words | translated) - STOP


def keyword_match(question: str, threshold: int = 2, preferred_lang: str = "en") -> str | None:
    q_kw            = get_keywords(question.lower())
    specific_hostel = q_kw & HOSTEL_NAMES

    if not q_kw:
        return None

    candidates = []
    for item in load_qa_database():
        s_kw    = get_keywords(item["question"].lower())
        matches = len(q_kw & s_kw)
        score   = matches / max(len(q_kw), len(s_kw), 1)

        if specific_hostel:
            if not (specific_hostel & (s_kw & HOSTEL_NAMES)):
                continue

        if matches >= threshold and score > 0:
            candidates.append({
                "answer": item["answer"],
                "score":  score,
                "lang":   detect_answer_lang(item["answer"], item.get("lang", "")),
            })
            print(f"[KW] {score:.2f} m={matches}: {item['question'][:45]}")

    if not candidates:
        return None

    candidates.sort(
        key=lambda c: (1 if c["lang"] == preferred_lang else 0, c["score"]),
        reverse=True
    )
    return candidates[0]["answer"]


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — FULL RAG PIPELINE (BM25 + Qdrant + RRF Fusion)
# ══════════════════════════════════════════════════════════════════════
async def rag_search_async(question: str, lang: str = "en") -> dict:
    sources       = []
    used_internet = False

    try:
        bm25_results = bm25_search(query=question, top_k=5)

        collections = get_collection_for_query(question, lang)
        if "website" not in collections:
            collections.append("website")

        vector      = get_embed_model().embed_query(question)
        lang_filter = lang if lang in ("en", "hi", "ga") else None

        vector_results = multi_collection_search(
            client=get_client(),
            collections=collections,
            query_vector=vector,
            query_text=question,
            limit=5,
            lang_filter=lang_filter,
        )

        merged = reciprocal_rank_fusion(
            bm25_results=bm25_results,
            vector_results=vector_results,
            bm25_weight=0.4,
            vector_weight=0.6,
        )

        if merged:
            merged = rerank_with_diversity(results=merged, query=question, top_k=3)

        ctx_parts = []
        for r in merged[:3]:
            url = r.get("url") or r.get("metadata", {}).get("source", "")
            if url and url.startswith("http"):
                sources.append(url)
            ctx_parts.append(f"[Score: {r['rerank_score']:.3f}]\n{r['text']}")

        top_score = merged[0]["rerank_score"] if merged else 0
        if top_score < 0.05 or not merged:
            print("[RAG] Low confidence — trying internet search")
            internet_results = search_college_website(question)
            if internet_results:
                used_internet = True
                for r in internet_results[:2]:
                    ctx_parts.append(f"[Web Result]\n{r['snippet']}\nSource: {r['url']}")
                    sources.append(r["url"])

        if not ctx_parts:
            return {"context": None, "sources": [], "used_internet": False}

        context = "\n\n---\n\n".join(ctx_parts)
        sources = list(dict.fromkeys(sources))
        print(f"[RAG] Context: {len(ctx_parts)} chunks | Internet: {used_internet}")
        return {"context": context, "sources": sources, "used_internet": used_internet}

    except Exception as e:
        print(f"[RAG] Error: {e}")
        return {"context": None, "sources": [], "used_internet": False}


def rag_search(question: str, lang: str = "en") -> str | None:
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        result = loop.run_until_complete(rag_search_async(question, lang))
    except RuntimeError:
        try:
            result = asyncio.run(rag_search_async(question, lang))
        except Exception as e:
            print(f"[RAG] rag_search error: {e}")
            return None
    except Exception as e:
        print(f"[RAG] rag_search error: {e}")
        return None
    return result.get("context") if result else None


# ══════════════════════════════════════════════════════════════════════
# LLM PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════
def build_prompt(
    question: str,
    context:  str,
    lang:     str,
    history:  str = "",
    sources:  list[str] | None = None,
) -> str:
    source_text = ""
    if sources:
        source_text = "\nSources available: " + ", ".join(sources[:3])

    if lang == "hi":
        return f"""Aap दीक्षा hain — GBPIET ke liye helpful AI chatbot.
RULES:
- HAMESHA Hindi mein jawab dein.
- Aap ek ladki hain — hamesha feminine forms use karein:
  सकती हूँ (na ki सकता हूँ), करूँगी (na ki करूँगा),
  जानती हूँ (na ki जानता हूँ).
- Sirf context use karein.
{history}
Context:
{context}
{source_text}
Sawaal: {question}
Jawab (Hindi mein, feminine):"""

    elif lang == "ga":
        return f"""तू दीक्षा छुं — जीबीपीआईईटी कु चैटबॉट।
RULES:
- हमेशा गढ़वाली मा जवाब दे।
- तू एक लड़की छुं — feminine forms use कर।
- अपणु नाम हमेशा "दीक्षा" लिख — कभी "डिक्षा" या "Diksha" नि लिखण।
- Sirf context use kar।
{history}
Context: {context}
{source_text}
Sawaal: {question}
Jawab (गढ़वाली मा):"""

    elif lang == "ku":
        return f"""तू दीक्षा छु — GBPIET री chatbot।
RULES:
- हमेशा कुमाउनी मा जवाब दे।
- तू एक लड़की छु — feminine forms use कर।
- अपणु नाम हमेशा "दीक्षा" लिख।
- Sirf context use kar।
{history}
Context: {context}
{source_text}
Sawaal: {question}
Jawab (कुमाउनी मा):"""

    else:
        return f"""You are दीक्षा (Diksha) — AI assistant for GBPIET.

LANGUAGE RULE: ENGLISH ONLY. Translate Hindi context to English.

RULES:
- Use proper titles (Prof., Dr.)
- Use ONLY the context below
- If not found: "I'm sorry, I couldn't find that information."
- If sources available, mention them naturally in answer
{source_text}

{history}
Context:
{context}

Question: {question}
Answer (ENGLISH ONLY):"""


# ══════════════════════════════════════════════════════════════════════
# LLM ANSWER — 4 keys × 2 models = 8 attempts
# ══════════════════════════════════════════════════════════════════════
def llm_answer(question: str, context: str, lang: str, history: str = "") -> str:
    prompt = build_prompt(question, context, lang, history)
    system = (
        "You are दीक्षा (Diksha), a helpful female AI assistant for GBPIET college students. "
        "You are female — always use feminine Hindi grammar: "
        "सकती हूँ (not सकता हूँ), करूँगी (not करूँगा), जानती हूँ (not जानता हूँ). "
        "When writing in Hindi, Garhwali, or Kumauni, always spell your name as दीक्षा."
    )

    result = _groq_with_fallback(system, prompt, max_tokens=500, temperature=0.3)
    if result:
        return result

    return "I'm sorry, I couldn't generate a response right now. Please try again."


# ══════════════════════════════════════════════════════════════════════
# MAIN ENTRY — complete pipeline
# ══════════════════════════════════════════════════════════════════════
def get_answer(question: str, lang: str = "en", history: str = "") -> str:
    question = question.strip()
    print(f"\n{'='*55}")
    print(f"[Q/{lang}] {question}")
    print(f"{'='*55}")

    build_bm25_index()

    ans = specific_role_answer(question, preferred_lang=lang)
    if ans:
        print("[RESULT] Specific role match")
        return translate_answer_if_needed(ans, lang, question)

    ans = direct_keyword_answer(question, preferred_lang=lang)
    if ans:
        print("[RESULT] Direct keyword")
        return translate_answer_if_needed(ans, lang, question)

    ans = exact_match(question)
    if ans:
        print("[RESULT] Exact match")
        return translate_answer_if_needed(ans, lang, question)

    word_count = len(question.split())
    thresh     = 1 if word_count <= 2 else (2 if word_count <= 5 else 3)
    ans        = keyword_match(question, thresh, preferred_lang=lang)
    if ans:
        print("[RESULT] Keyword match")
        return translate_answer_if_needed(ans, lang, question)

    ctx = rag_search(question, lang)
    if ctx:
        print("[RESULT] Hybrid RAG + LLM")
        return llm_answer(question, ctx, lang, history)

    print("[RESULT] No match found")
    fb = {
        "hi": "माफ़ करें, मैं आपकी क्वेरी समझ नहीं पाई।",
        "ga": "माफ करा, मीथे ये जानकारी नी मिली।",
        "ku": "माफ करिया! म्यर पास तस के जानकारी नैं च।",
        "en": "I'm sorry, I'm unable to understand your query."
    }
    return fb.get(lang, fb["en"])


# ══════════════════════════════════════════════════════════════════════
# TEST
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    load_qa_database()
    print("\nTESTING\n" + "=" * 55)
    tests = [
        ("registrar",                              "en"),
        ("dean academic",                          "en"),
        ("What is the placement record?",          "en"),
        ("MCA की फीस कितनी है?",                   "hi"),
        ("जीबीपीआईईटी कख च?",                     "ga"),
        ("को cha",                                 "ga"),
    ]
    for q, l in tests:
        print(f"\nQ[{l}]: {q}")
        print(f"A: {get_answer(q, l)[:250]}")
        print("-" * 55)
