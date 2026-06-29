import os
import json
import glob
import re
import asyncio
import unicodedata
from dotenv import load_dotenv

load_dotenv()

# Gemini optional
try:
    from google import genai
    from google.genai import types as genai_types
    _gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    print("[GEMINI] ✅ client ready")
except Exception:
    _gemini_client = None
    print("[GEMINI] ❌ not available")

from qdrant_setup import get_client, COLLECTIONS
from intent_detector import get_collection_for_query
from rag.hybrid_search import multi_collection_search
from rag.bm25_search import bm25_search, build_bm25_index
from rag.fusion import reciprocal_rank_fusion
from rag.internet_search import search_college_website

try:
    from rag.reranker import rerank_with_diversity
    _HAS_RERANKER = True
except ImportError:
    _HAS_RERANKER = False


# ══════════════════════════════════════════════════════════════════════
# ✅ LAZY GROQ CLIENTS — startup pe load nahi honge
# ══════════════════════════════════════════════════════════════════════
_groq1 = None
_groq2 = None
_groq3 = None
_groq4 = None
_groq_initialized = False

def _init_groq_clients():
    """Groq clients sirf pehli baar call hone pe load honge."""
    global _groq1, _groq2, _groq3, _groq4, _groq_initialized
    if _groq_initialized:
        return
    from groq import Groq

    def _make_groq(env_var: str):
        key = os.getenv(env_var, "").strip()
        if key:
            print(f"[GROQ] {env_var} ✅")
            return Groq(api_key=key)
        print(f"[GROQ] {env_var} ❌ not set")
        return None

    _groq1 = _make_groq("GROQ_API_KEY")
    _groq2 = _make_groq("GROQ_API_KEY_2")
    _groq3 = _make_groq("GROQ_API_KEY_3")
    _groq4 = _make_groq("GROQ_API_KEY_4")
    _groq_initialized = True
    print("[GROQ] ✅ All clients initialized")


GROQ_PRIMARY  = "llama-3.3-70b-versatile"
GROQ_FALLBACK = "llama3-70b-8192"

def _get_groq_attempts():
    """Attempts list dynamically banao taaki lazy init kaam kare."""
    _init_groq_clients()
    return [
        (_groq1, GROQ_PRIMARY,  "Key1/Primary"),
        (_groq2, GROQ_PRIMARY,  "Key2/Primary"),
        (_groq3, GROQ_PRIMARY,  "Key3/Primary"),
        (_groq4, GROQ_PRIMARY,  "Key4/Primary"),
        (_groq1, GROQ_FALLBACK, "Key1/Fallback"),
        (_groq2, GROQ_FALLBACK, "Key2/Fallback"),
        (_groq3, GROQ_FALLBACK, "Key3/Fallback"),
        (_groq4, GROQ_FALLBACK, "Key4/Fallback"),
    ]


# ✅ LAZY EMBED MODEL — pehli request pe load hoga
_embed_model = None
_qa_database = []

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GBPIET_URL       = "https://gbpiet.ac.in"


# ══════════════════════════════════════════════════════════════════════
# ROMAN ↔ DEVANAGARI BRIDGE
# ══════════════════════════════════════════════════════════════════════
ROMAN_TO_DEVANAGARI = {
    "hostel":      "हॉस्टल",
    "hostels":     "हॉस्टल",
    "hostle":      "हॉस्टल",
    "fees":        "फीस",
    "fee":         "फीस",
    "admission":   "प्रवेश",
    "admissions":  "प्रवेश",
    "director":    "निदेशक",
    "hod":         "विभागाध्यक्ष",
    "warden":      "वार्डन",
    "placement":   "प्लेसमेंट",
    "placements":  "प्लेसमेंट",
    "library":     "पुस्तकालय",
    "canteen":     "कैंटीन",
    "result":      "परिणाम",
    "results":     "परिणाम",
    "faculty":     "संकाय",
    "exam":        "परीक्षा",
    "exams":       "परीक्षा",
    "sports":      "खेल",
    "transport":   "परिवहन",
    "ragging":     "रैगिंग",
    "bank":        "बैंक",
    "mess":        "मेस",
    "contact":     "संपर्क",
    "scholarship": "छात्रवृत्ति",
    "scholarships":"छात्रवृत्ति",
    "course":      "कोर्स",
    "courses":     "कोर्स",
    "branch":      "शाखा",
    "branches":    "शाखा",
    "department":  "विभाग",
    "departments": "विभाग",
    "btech":       "बी.टेक",
    "b.tech":      "बी.टेक",
    "mtech":       "एम.टेक",
    "m.tech":      "एम.टेक",
    "mca":         "एमसीए",
    "phd":         "पीएचडी",
    "ph.d":        "पीएचडी",
    "cse":         "सीएसई",
    "ece":         "ईसीई",
    "civil":       "सिविल",
    "electrical":  "इलेक्ट्रिकल",
    "mechanical":  "मैकेनिकल",
    "biotech":     "बायोटेक्नोलॉजी",
    "biotechnology":"बायोटेक्नोलॉजी",
    "dean":        "डीन",
    "registrar":   "रजिस्ट्रार",
    "chairman":    "अध्यक्ष",
    "boys":        "नौना",
    "girls":       "नौन्यिँ",
    "first":       "पैलि",
    "year":        "साल",
    "semester":    "सेमेस्टर",
    "wifi":        "वाई-फाई",
    "internet":    "इंटरनेट",
    "bus":         "बस",
    "health":      "स्वास्थ्य",
    "atm":         "एटीएम",
    "auditorium":  "ऑडिटोरियम",
    "gate":        "गेट",
    "jee":         "जेईई",
    "total":       "कुल",
    "how many":    "कति",
    "kitne":       "कति",
    "kitna":       "कति",
    "kaun":        "कु",
    "kon":         "कु",
    "kahan":       "कख",
    "kya":         "क्या",
    "bca":         "बीसीए",
    "mba":         "एमबीए",
    "nptel":       "एनपीटीईएल",
    "ieee":        "आईईईई",
}

DEVANAGARI_TO_ROMAN = {v: k for k, v in ROMAN_TO_DEVANAGARI.items()}


# ══════════════════════════════════════════════════════════════════════
# GROQ CALL — lazy init + 8 attempts + Gemini fallback
# ══════════════════════════════════════════════════════════════════════
def _groq_call(client, model, messages, max_tokens, temperature):
    if client is None:
        return None
    try:
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GROQ] {model} failed: {e}")
        return None


def groq_call(messages, max_tokens=500, temperature=0.3) -> str:
    for client, model, label in _get_groq_attempts():
        result = _groq_call(client, model, messages, max_tokens, temperature)
        if result:
            print(f"[LLM] ✅ {label}")
            return result

    # Gemini fallback
    if _gemini_client:
        try:
            from google.genai import types as genai_types
            prompt_text = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            )
            r = _gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt_text,
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
            print("[LLM] ✅ Gemini fallback")
            return r.text.strip()
        except Exception as e:
            print(f"[LLM] Gemini also failed: {e}")

    print("[LLM] ❌ All attempts failed")
    return ""


# ══════════════════════════════════════════════════════════════════════
# EMOJI STRIPPER + CLEAN RESPONSE
# ══════════════════════════════════════════════════════════════════════
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U00002600-\U000026FF"
    "\U0000200D"
    "\U0000FE0F"
    "]+",
    flags=re.UNICODE,
)

_SELF_INTRO_PATTERN = re.compile(
    r'^(Respected\s+\w+[\s,]*'
    r'|Dear\s+\w+[\s,]*'
    r'|प्रिय\s+\w+[\s,]*'
    r'|दीक्षा\s+(छु[ंँ]?|छन|हूँ|हूं)\s*[—\-,।]?\s*'
    r'|मैं\s+दीक्षा\s+(छु[ंँ]?|छन|हूँ|हूं)\s*[—\-,।]?\s*'
    r'|I\s+am\s+Diksha[,.]?\s*'
    r'|नमस्ते[!,।]?\s+मैं\s+दीक्षा\s*'
    r')',
    flags=re.IGNORECASE,
)

def clean_response(text: str) -> str:
    text = _EMOJI_PATTERN.sub("", text)
    for _ in range(3):
        text = _SELF_INTRO_PATTERN.sub("", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return re.sub(r'\s+', ' ', text).strip()


# ══════════════════════════════════════════════════════════════════════
# LANGUAGE MARKERS
# ══════════════════════════════════════════════════════════════════════
GA_MARKERS = [
    'छन', 'छ।', 'हूँद', 'कुण', 'मिलद', 'पैलू', 'अर',
    'कनकै', 'कख', 'बटि', 'त्वै', 'छौ', 'छी',
    'छ्यायी', 'थ्यायी', 'तुमुं', 'यैसैं', 'वैसें',
    'माँ', 'मी', 'जु', 'यु', 'वु',
]

KU_MARKERS = [
    'छौ', 'छन', 'छौँ', 'छा', 'लै', 'बटी', 'हैबर',
    'कसि', 'कै', 'म्यूँ', 'त्यूँ', 'यो', 'वो', 'भयो',
    'ज्यू', 'भल', 'नानी', 'ठुली', 'हिटा', 'तल्लि', 'मल्लि',
]


# ══════════════════════════════════════════════════════════════════════
# GREETING / IDENTITY
# ══════════════════════════════════════════════════════════════════════
GREETINGS = {
    "hello", "hi", "hlo", "hey", "hii", "helo", "namaste",
    "नमस्ते", "हेलो", "हाय", "good morning", "good afternoon", "good evening",
    "नमस्कार", "राम राम", "जय हो", "समन्या", "समन्या जी",
    "प्रणाम", "प्रणाम जी",
}

IDENTITY_Q = {
    "who are you", "what are you", "who r u", "tum kaun ho", "kon ho",
    "aap kaun hain", "aap kaun ho", "kaun ho tum", "who made you",
    "who created you", "who made diksha", "diksha kaun hai",
    "who build you", "who built you", "who developed you", "who designed you",
    "who build diksha", "who built diksha", "who created diksha",
    "tumhe kisne banaya", "kisne banaya", "aapko kisne banaya",
    "tumhe kisne bnaya", "kon banaya", "kisne banayi", "kisne build kiya",
    "को च", "कु च", "कू च", "को छ", "कु छ", "को cha", "ko cha",
    "तू को छ", "तू कु छ", "तू को च",
    "को छै", "के छै", "तू को छै", "तुम को छौ", "तुमार नाम के छ",
    "ko chai", "tumar naam ke cha", "tu ko chai",
}

GREETING_RESPONSE = {
    "en": "Hello! I'm Diksha, the official AI assistant for GBPIET, Pauri Garhwal. Ask me about admissions, fees, hostel, placements, faculty, courses and more!",
    "hi": "नमस्ते! GBPIET की आधिकारिक AI सहायिका दीक्षा आपकी सेवा में हूँ। admission, fees, hostel, placement के बारे में पूछ सकते हैं।",
    "ga": "समन्या जी! जीबीपीआईईटी की AI दगड़िया छुं। कुछ भी पुछि सकदन।",
    "ku": "नमस्कार जी! जीबीपीआईईटी की AI दगड़िया छु। कुछ भी पूछ सकदन।",
}

IDENTITY_RESPONSE = {
    "en": f"I'm Diksha, the official AI chatbot for GBPIET (Govind Ballabh Pant Institute of Engineering and Technology), Pauri Garhwal. Visit: {GBPIET_URL}",
    "hi": f"GBPIET (गोविंद बल्लभ पंत इंजीनियरिंग कॉलेज), पौड़ी गढ़वाल की आधिकारिक AI chatbot दीक्षा हूँ। वेबसाइट: {GBPIET_URL}",
    "ga": f"जीबीपीआईईटी, पौड़ी गढ़वाल की official AI chatbot छुं — नाम दीक्षा छ। वेबसाइट: {GBPIET_URL}",
    "ku": f"जीबीपीआईईटी, पौड़ी गढ़वाल की official AI chatbot छु — नाम दीक्षा छ। वेबसाइट: {GBPIET_URL}",
}

OUT_OF_SCOPE_RESPONSE = {
    "en": "Sorry, this question is out of my syllabus! I can only answer questions related to GBPIET — admissions, fees, hostel, placements, faculty, courses and more.",
    "hi": "माफ़ करें, यह सवाल मेरे syllabus से बाहर है! मैं केवल GBPIET से जुड़े सवालों का जवाब दे सकती हूँ।",
    "ga": "माफ करा, यु सवाल मेरे syllabus बटि बाहर छ! मी सिर्फ GBPIET बारे माँ जानकारी दे सकदुं।",
    "ku": "माफ करिया, यु सवाल मेरे syllabus बटा भ्यार छ! मी सिर्फ GBPIET बारे मा ज्याणी दे सकदु।",
}


# ══════════════════════════════════════════════════════════════════════
# LLM-BASED OUT-OF-SCOPE DETECTOR
# ══════════════════════════════════════════════════════════════════════
_scope_cache: dict = {}

def is_out_of_scope(question: str) -> bool:
    q = question.strip().lower()

    if len(q.split()) <= 4:
        return False

    if q in GREETINGS or q in IDENTITY_Q:
        return False

    _intro_words = {"i am", "i'm", "my name", "mera naam", "main hoon", "naam hai"}
    if any(iw in q for iw in _intro_words):
        print(f"[SCOPE] User intro detected → IN SCOPE")
        return False

    q_clean = re.sub(r'[^\w\s]', '', q).strip()
    for phrase in SPECIFIC_ROLE_MAP:
        if phrase in q_clean:
            print(f"[SCOPE] Role map match '{phrase}' → IN SCOPE")
            return False

    STAFF_KEYWORDS = {
        "priti", "dimri", "rawat", "nautiyal", "negi", "bisht", "kunwar",
        "narayan", "siddharth", "ghansela", "professor", "prof", "dr",
        "doctor", "faculty", "teacher", "assistant professor",
        "associate professor", "hod", "dean", "warden", "registrar",
        "director", "chairman",
    }
    for kw in STAFF_KEYWORDS:
        if kw in q:
            print(f"[SCOPE] Staff keyword '{kw}' → IN SCOPE")
            return False

    if q in _scope_cache:
        return _scope_cache[q]

    system = (
        "You are a strict classifier for a college chatbot. "
        "Decide if the question is related to GBPIET college. "
        "Answer with ONLY one word: YES (college-related) or NO (not college-related). "
        "YES: fees, hod, admission, hostel, placement, GBPIET, pauri, director, faculty, "
        "courses, library, transport, ragging, result, exam, scholarship, department, dean, "
        "registrar, chairman, contact, sports, canteen, bus, mess, mca, btech, mtech, "
        "how to reach, college location, campus, back paper, pyq, previous year question. "
        "NO: salman khan, taj mahal, ipl score, weather, modi, cooking, bitcoin, "
        "bollywood, ram mandir ayodhya, politics, news, other city location, stock market."
    )
    user_msg = f"Question: {question}\nAnswer YES or NO only."

    result = groq_call(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=5,
        temperature=0.0,
    )

    is_oos = "YES" not in result.upper()
    _scope_cache[q] = is_oos

    label = "OUT OF SCOPE" if is_oos else "IN SCOPE"
    print(f"[SCOPE] LLM says {label}: '{question}' → '{result}'")
    return is_oos


# ══════════════════════════════════════════════════════════════════════
# ✅ LAZY EMBED MODEL + QDRANT
# ══════════════════════════════════════════════════════════════════════
def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("[Embed] Loading model... (first request)")
        from langchain_huggingface import HuggingFaceEmbeddings
        _embed_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[Embed] ✅ Model loaded")
    return _embed_model


def get_qdrant():
    return get_client()


# ══════════════════════════════════════════════════════════════════════
# ✅ LAZY QA DATABASE
# ══════════════════════════════════════════════════════════════════════
def load_qa_database() -> list:
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
                            "answer":   answer.strip(),
                            "source":   os.path.basename(filepath),
                            "lang":     item.get("lang", "").strip().lower(),
                        })
        except Exception as e:
            print(f"[DB] Error loading {filepath}: {e}")

    print(f"[DB] ✅ Loaded {len(_qa_database)} QA pairs")
    return _qa_database


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
    if item_lang in ("ga", "garhwali"): return "ga"
    if item_lang in ("ku", "kumauni"):  return "ku"
    if item_lang in ("hi", "hindi"):    return "hi"
    if item_lang == "en":               return "en"
    if any(m in answer_text for m in GA_MARKERS): return "ga"
    if any(m in answer_text for m in KU_MARKERS): return "ku"
    latin = sum(1 for c in answer_text if c.isascii() and c.isalpha())
    total = len(answer_text.replace(" ", ""))
    if total > 0 and (latin / total) > 0.5: return "en"
    dev = sum(1 for c in answer_text if '\u0900' <= c <= '\u097F')
    return "hi" if dev > 5 else "en"


def translate_answer_if_needed(answer: str, lang: str, question: str) -> str:
    answer_lang = detect_answer_lang(answer)

    if answer_lang == lang:
        return answer

    if lang == "ga":
        prompt = (
            f"Translate this college information into Garhwali language (गढ़वाली).\n"
            f"Use Garhwali words: छन, छ, अर, कुण, बटि, मिलद, हूँद, कनकै, यु, वु।\n"
            f"Keep names, numbers, URLs unchanged.\n"
            f"Return ONLY the Garhwali translation.\n\n{answer}"
        )
        system = (
            "You are a Garhwali translator. Translate to Garhwali ONLY. "
            "Use words like छन, छ, अर, कुण, बटि, मिलद। "
            "Keep proper nouns, numbers, URLs unchanged."
        )
        print(f"[TRANSLATE] → Garhwali...")
        result = groq_call(
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=500, temperature=0.1,
        )
        return result if result else answer

    if lang == "ku":
        prompt = (
            f"Translate this college information into Kumauni language (कुमाउनी).\n"
            f"Use Kumauni words: छु, छन, राछ, हुनी, कनाँ, लै, बटा, ज्याणी, कैं।\n"
            f"Keep names, numbers, URLs unchanged.\n"
            f"Return ONLY the Kumauni translation.\n\n{answer}"
        )
        system = (
            "You are a Kumauni translator. Translate to Kumauni ONLY. "
            "Use words like छु, छन, लै, बटा, ज्याणी। "
            "Keep proper nouns, numbers, URLs unchanged."
        )
        print(f"[TRANSLATE] → Kumauni...")
        result = groq_call(
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=500, temperature=0.1,
        )
        return result if result else answer

    if lang in ("hi", "ga", "ku") and answer_lang == "en":
        prompt = f"Translate to Hindi (Devanagari). Return ONLY translated text.\n\n{answer}"
        system = "Translator. English→Hindi. Keep names, numbers, URLs unchanged."
        print(f"[TRANSLATE] → Hindi...")
        result = groq_call(
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=400, temperature=0.1,
        )
        return result if result else answer

    if lang == "en" and answer_lang in ("hi", "ga", "ku"):
        prompt = f"Translate to English. Return ONLY translated text.\n\n{answer}"
        system = "Translator. Hindi→English. Keep names, numbers, URLs unchanged."
        print(f"[TRANSLATE] → English...")
        result = groq_call(
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=400, temperature=0.1,
        )
        return result if result else answer

    return answer


# ══════════════════════════════════════════════════════════════════════
# HINDI → ENGLISH MAP
# ══════════════════════════════════════════════════════════════════════
HINDI_MAP = {
    'जीबीपीआईईटी': 'gbpiet',    'जीबीपीईटी': 'gbpiet',
    'संस्थान': 'institute',      'कॉलेज': 'college',
    'पहुँचें': 'reach',           'कैसे': 'how',
    'रास्ता': 'route direction',  'पता': 'address',
    'कहाँ': 'where',              'निदेशक': 'director',
    'विभागाध्यक्ष': 'head department hod',
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
    'कितने': 'how many',           'कौन से': 'which',
    'कु': 'who', 'कू': 'who', 'कन': 'how', 'कनकै': 'how',
    'कख': 'where', 'बटि': 'from', 'कुण': 'for',
    'अर': 'and', 'माँ': 'in', 'मा': 'in',
    'मी': 'i me', 'जु': 'who', 'यु': 'this', 'वु': 'that',
    'को': 'who', 'के': 'what', 'कते': 'where',
    'कसि': 'how', 'कतु': 'how much', 'बटी': 'from',
}

def hi_to_en(text: str) -> str:
    t = text.lower()
    for h, e in HINDI_MAP.items():
        t = t.replace(h, ' ' + e + ' ')
    return re.sub(r'\s+', ' ', t).strip()


# ══════════════════════════════════════════════════════════════════════
# GARHWALI SYNONYM MAP
# ══════════════════════════════════════════════════════════════════════
GARHWALI_SYNONYM_MAP = {
    'खुणि': 'ke liye for', 'वास्ति': 'ke liye',
    'कूण': 'kaun who',     'कोण': 'kaun who',
    'कखे': 'kahan where',  'कख': 'kahan where',
    'कसे': 'kaise how',    'कनूँ': 'kaise how',
    'कति': 'kitna',        'कितणु': 'kitna',
    'एडमिशन': 'admission', 'दाखिला': 'admission', 'भर्ती': 'admission',
    'दाम': 'fees',         'पैसा': 'fees',
    'आवास': 'hostel',      'निवास': 'hostel',
    'मेस': 'mess',         'नौकरी': 'job placement',
    'तनख्वाह': 'salary',   'पगार': 'salary',
    'मी': 'main I',        'मेरो': 'mera my',
    'अर': 'aur and',       'बटे': 'se from',
}

def ga_ku_to_hi_en(text: str) -> str:
    t = text.lower()
    for word, tr in sorted(GARHWALI_SYNONYM_MAP.items(), key=lambda x: -len(x[0])):
        if word in t:
            t = t.replace(word, ' ' + tr + ' ')
    return re.sub(r'\s+', ' ', t).strip()


# ══════════════════════════════════════════════════════════════════════
# KUMAUNI SYNONYM MAP
# ══════════════════════════════════════════════════════════════════════
KUMAUNI_SYNONYM_MAP = {
    'नाइ':    'नहीं',
    'छु नाइ': 'न्हां छ',
    'नी छ':   'न्हां छ',
    'न छ':    'न्हां छ',
    'र':      'और',
    'लै हुनी':  'लिजी',
    'लै':       'लिजी',
    'कनाँ':     'लिजी',
    'जावा':   'जाया',
    'जा':     'जाया',
    'बटा':    'from / se',
    'छु':     'है / छ',
    'छन':     'हैं / छन',
    'ज्याणी': 'जानकारी',
    'लिजीये': 'लिजी',
    'कैं':    'को',
    'हैबेर':  'के बाद',
    'कसि':    'कैसे',
    'कतु':    'कितना',
    'किलै':   'क्यों',
    'को':     'कौन',
    'के':     'क्या',
}

def ku_to_hi_en(text: str) -> str:
    t = text.lower()
    for word, tr in sorted(KUMAUNI_SYNONYM_MAP.items(), key=lambda x: -len(x[0])):
        if word in t:
            t = t.replace(word, ' ' + tr + ' ')
    return re.sub(r'\s+', ' ', t).strip()


# ══════════════════════════════════════════════════════════════════════
# TYPO MAP
# ══════════════════════════════════════════════════════════════════════
TYPO_MAP = {
    "h0d": "hod",               "f33s": "fees",
    "mechenical": "mechanical", "mechnical": "mechanical",
    "mechincal": "mechanical",  "mechanicle": "mechanical",
    "mechinical": "mechanical", "electical": "electrical",
    "electrcal": "electrical",  "biotechonlogy": "biotechnology",
    "bitoech": "biotechnology", "admision": "admission",
    "admisson": "admission",    "palcement": "placement",
    "hostle": "hostel",         "dircetor": "director",
    "registar": "registrar",    "collage": "college",
    "colege": "college",        "faculity": "faculty",
    "placment": "placement",    "semster": "semester",
}

def fix_typos(text: str) -> str:
    words = text.lower().split()
    fixed = []
    for w in words:
        w_clean = w.replace('0', 'o').replace('1', 'i').replace('3', 'e')
        fixed.append(TYPO_MAP.get(w) or TYPO_MAP.get(w_clean) or w)
    result = " ".join(fixed)
    result = ga_ku_to_hi_en(result)
    result = ku_to_hi_en(result)
    return result


# ══════════════════════════════════════════════════════════════════════
# SPECIFIC ROLE MAP
# ══════════════════════════════════════════════════════════════════════
SPECIFIC_ROLE_MAP = {
    "dean academic affairs": "dean academic", "dean of academic": "dean academic",
    "dean academics": "dean academic",        "dean academic": "dean academic",
    "dean accadmic": "dean academic",         "dean acadmic": "dean academic",
    "dean student welfare": "dean student welfare",
    "dean of student": "dean student welfare",
    "dean student": "dean student welfare",   "dean welfare": "dean student welfare",
    "dean research": "dean research",         "dean planning": "dean planning",
    "dean faculty welfare": "dean faculty welfare",
    "dean faculty": "dean faculty welfare",
    "hod of cse": "hod cse",   "hod cse": "hod cse",
    "hod of ece": "hod ece",   "hod ece": "hod ece",
    "hod of me": "hod mechanical",     "hod me": "hod mechanical",
    "hod of mechanical": "hod mechanical", "hod mechanical": "hod mechanical",
    "hod of civil": "hod civil",       "hod civil": "hod civil",
    "hod of ee": "hod electrical",     "hod ee": "hod electrical",
    "hod of electrical": "hod electrical", "hod electrical": "hod electrical",
    "hod of mca": "hod mca",           "hod mca": "hod mca",
    "hod of csa": "hod mca",           "hod csa": "hod mca",
    "hod of biotech": "hod biotechnology", "hod biotech": "hod biotechnology",
    "hod of biotechnology": "hod biotechnology", "hod biotechnology": "hod biotechnology",
    "hod of applied": "hod applied sciences", "hod applied": "hod applied sciences",
    "warden of kailash": "warden kailash",   "warden kailash": "warden kailash",
    "warden of trishul": "warden trishul",   "warden trishul": "warden trishul",
    "warden of neelkanth": "warden neelkanth", "warden neelkanth": "warden neelkanth",
    "warden of vh": "warden viswerwarya",    "warden vh": "warden viswerwarya",
    "warden of viswerwarya": "warden viswerwarya", "warden viswerwarya": "warden viswerwarya",
    "warden of raman": "warden raman",       "warden raman": "warden raman",
    "warden of bhagirathi": "warden bhagirathi", "warden bhagirathi": "warden bhagirathi",
    "warden of rudra": "warden rudra",       "warden rudra": "warden rudra",
    "warden of badri": "warden badri",       "warden badri": "warden badri",
    "warden of kedar": "warden kedar",       "warden kedar": "warden kedar",
    "warden of alaknanda": "warden alaknanda", "warden alaknanda": "warden alaknanda",
    "warden of shivalik": "warden shivalik", "warden shivalik": "warden shivalik",
    "priti dimri": "hod mca",               "prof priti dimri": "hod mca",
    "kunwar deep narayan": "faculty mca",   "kunwar narayan": "faculty mca",
    "siddharth ghansela": "faculty mca",    "dr siddharth": "faculty mca",
}

ROLE_BLACKLIST = [
    "list all", "all hod", "all department", "all heads",
    "departments at gbpiet", "list of hod", "all hods",
    "सभी विभाग", "सभी hod",
]

def specific_role_answer(question: str, preferred_lang: str = "en"):
    q_fixed = fix_typos(question)
    q_clean = re.sub(r'[^\w\s]', '', q_fixed.strip().lower()).strip()

    mapped = None
    for phrase, topic in sorted(SPECIFIC_ROLE_MAP.items(), key=lambda x: -len(x[0])):
        if phrase in q_clean:
            mapped = topic
            print(f"[ROLE] '{q_clean}' → '{mapped}'")
            break
    if not mapped:
        return None

    topic_words = mapped.lower().split()
    candidates  = []

    for item in load_qa_database():
        q_lower = item["question"].lower()
        a_lower = item["answer"].lower()
        if any(p in q_lower for p in ROLE_BLACKLIST):
            continue
        score = 0
        if all(w in q_lower for w in topic_words):                                       score += 3
        elif mapped.lower() in q_lower:                                                   score += 2
        elif len(topic_words) >= 2 and sum(1 for w in topic_words if w in q_lower) >= 2: score += 1
        if any(w in a_lower for w in topic_words):                                        score += 1
        if score >= 2:
            candidates.append({
                "answer": item["answer"],
                "score":  score,
                "lang":   detect_answer_lang(item["answer"], item.get("lang", "")),
            })

    if not candidates:
        return None

    candidates.sort(
        key=lambda c: (1 if c["lang"] == preferred_lang else 0, c["score"]),
        reverse=True,
    )
    best = candidates[0]
    print(f"[ROLE] ✅ score={best['score']} lang={best['lang']}")
    return best["answer"]


# ══════════════════════════════════════════════════════════════════════
# DIRECT KEYWORD MAP
# ══════════════════════════════════════════════════════════════════════
DIRECT_KEYWORD_MAP = {
    "registrar": "registrar",   "director":   "director",
    "dean":      "dean",        "chairman":   "chairman",
    "warden":    "warden",      "placement":  "placement",
    "placements":"placement",   "hostel":     "hostel",
    "hostels":   "hostel",      "fees":       "fees",
    "fee":       "fees",        "admission":  "admission",
    "admissions":"admission",   "contact":    "contact",
    "courses":   "courses",     "course":     "courses",
    "library":   "library",     "transport":  "transport",
    "scholarship":"scholarship","result":     "result",
    "ragging":   "ragging",     "sports":     "sports",
    "faculty":   "faculty",     "hod":        "head of department",
    "about":     "about gbpiet","website":    "gbpiet website",
    "h0d":       "head of department", "hods": "head of department",
    "रजिस्ट्रार": "registrar",  "निदेशक":   "director",
    "डीन":         "dean",       "प्लेसमेंट": "placement",
    "हॉस्टल":      "hostel",     "फीस":       "fees",
    "प्रवेश":      "admission",  "संपर्क":    "contact",
    "पुस्तकालय":  "library",    "परिवहन":    "transport",
    "रैगिंग":      "ragging",    "संकाय":     "faculty",
    "एडमिशन":      "admission",  "भर्ती":    "admission",
    "नौकरी":        "placement",  "सुविधा":   "facility",
    "भर्ति":        "admission",  "सुबिद":    "facility",
    "ज्याणी":       "information","दाम":      "fees",
}

def direct_keyword_answer(question: str, preferred_lang: str = "en"):
    q_fixed    = fix_typos(question)
    q_clean    = q_fixed.strip().lower()
    word_count = len(q_clean.split())

    if word_count > 2:
        return None

    first_word = q_clean.split()[0] if q_clean.split() else ""
    mapped = (
        DIRECT_KEYWORD_MAP.get(q_clean)
        or DIRECT_KEYWORD_MAP.get(first_word)
    )
    if not mapped:
        qt = hi_to_en(q_clean)
        mapped = (
            DIRECT_KEYWORD_MAP.get(qt.strip())
            or DIRECT_KEYWORD_MAP.get(qt.split()[0] if qt.split() else "")
        )
    if not mapped:
        return None

    print(f"[DIRECT_KW] '{q_clean}' → '{mapped}'")
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
        reverse=True,
    )
    best = candidates[0]
    print(f"[DIRECT_KW] ✅ score={best['score']} lang={best['lang']}")
    return best["answer"]


# ══════════════════════════════════════════════════════════════════════
# EXACT MATCH
# ══════════════════════════════════════════════════════════════════════
def exact_match(question: str):
    q = question.strip().lower()
    for item in load_qa_database():
        if q == item["question"].strip().lower():
            print(f"[EXACT] {item['question'][:60]}")
            return item["answer"]
    return None


# ══════════════════════════════════════════════════════════════════════
# KEYWORD MATCH
# ══════════════════════════════════════════════════════════════════════
STOP = {
    'what','who','is','are','the','at','in','of','a','an','and','or',
    'for','to','how','does','do','has','have','many','which','tell',
    'me','about','please','can','you','i','my','their','kya','hai',
    'hain','ka','ki','ke','mein','se','per','ek',
    'क्या','कौन','का','की','के','में','से','है','हैं','एक',
    'और','या','को','ने','था','थी','थे','कि','जो','तो','भी',
    'मैं','हम','आप','वे','इस','उस','यह','वह','पर','बारे',
    'कैसे','कहाँ','कहां','तक',
    'कु','कू','कि','कन','कनकै','कख','कनै',
    'माँ','मा','बटि','च','छ','छन',
    'अर','त','त्वै','भी','न','थौ','छौ','छी',
    'मी','आम','तुम','तुमुं','वु','वे',
    'यैसैं','यें','वैसें','वैन','यु','जु',
    'के','कन','कसि','कै','छन','छा',
    'लै','बटा','हैबर','यो','वो','भयो',
    'म्यूँ','त्यूँ','ज्यू','भल',
}

HOSTEL_NAMES = {
    'kailash','neelkanth','kedar','rudra','badri','alaknanda',
    'shivalik','trishul','raman','bhagirathi','viswerwarya','vh',
}

LATERAL_KEYWORDS = {"lateral", "लेटरल", "second year", "द्वितीय वर्ष", "2nd year"}

def _is_lateral_query(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in LATERAL_KEYWORDS)

def _is_lateral_item(item: dict) -> bool:
    combined = (item["question"] + " " + item["answer"]).lower()
    return any(k in combined for k in LATERAL_KEYWORDS)

def get_keywords(text: str) -> set:
    words         = set(re.findall(r'[\u0900-\u097F]+|[a-zA-Z0-9]+', text.lower()))
    translated    = set(re.findall(r'[a-zA-Z0-9]+', hi_to_en(text)))
    ga_translated = set(re.findall(r'[a-zA-Z0-9]+', ga_ku_to_hi_en(text)))
    ku_translated = set(re.findall(r'[a-zA-Z0-9]+', ku_to_hi_en(text)))
    return (words | translated | ga_translated | ku_translated) - STOP


def keyword_match(question: str, threshold: int = 2, preferred_lang: str = "en"):
    q_fixed         = fix_typos(question)
    q_kw            = get_keywords(q_fixed.lower())
    specific_hostel = q_kw & HOSTEL_NAMES

    if not q_kw:
        return None

    candidates = []
    for item in load_qa_database():
        if _is_lateral_item(item) and not _is_lateral_query(question):
            continue
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

    if not candidates:
        return None

    candidates.sort(
        key=lambda c: (1 if c["lang"] == preferred_lang else 0, c["score"]),
        reverse=True,
    )
    best = candidates[0]
    print(f"[KW] ✅ score={best['score']:.2f} lang={best['lang']}")
    return best["answer"]


# ══════════════════════════════════════════════════════════════════════
# DATASET-ONLY SEARCH
# ══════════════════════════════════════════════════════════════════════
def _enrich_keywords_roman(kw_set: set, original_text: str) -> set:
    enriched = set(kw_set)
    words = original_text.lower().split()
    for word in words:
        if word in ROMAN_TO_DEVANAGARI:
            enriched.add(ROMAN_TO_DEVANAGARI[word])
    for word in list(enriched):
        if word in DEVANAGARI_TO_ROMAN:
            enriched.add(DEVANAGARI_TO_ROMAN[word])
    return enriched - STOP


def dataset_only_search(question: str, lang: str) -> str | None:
    q_fixed = fix_typos(question)
    q_lower = q_fixed.lower()

    q_kw_base = get_keywords(q_lower)
    q_kw      = _enrich_keywords_roman(q_kw_base, q_lower)
    q_en      = hi_to_en(q_lower)
    q_en_kw   = set(re.findall(r'[a-zA-Z0-9]+', q_en)) - STOP
    q_all_kw  = q_kw | q_en_kw

    if not q_all_kw:
        return None

    candidates = []

    for item in load_qa_database():
        item_lang = item.get("lang", "").strip().lower()
        if item_lang != lang:
            continue
        if _is_lateral_item(item) and not _is_lateral_query(question):
            continue

        s_lower   = item["question"].lower()
        s_kw_base = get_keywords(s_lower)
        s_kw      = _enrich_keywords_roman(s_kw_base, s_lower)
        s_en_kw   = set(re.findall(r'[a-zA-Z0-9]+', hi_to_en(s_lower))) - STOP
        s_all_kw  = s_kw | s_en_kw

        matches = len(q_all_kw & s_all_kw)
        if matches == 0:
            continue

        denom = max(len(q_all_kw), len(s_all_kw), 1)
        score = matches / denom

        candidates.append({
            "answer":  item["answer"],
            "score":   score,
            "matches": matches,
        })

    if not candidates:
        print(f"[DATASET_ONLY] No match for lang={lang} q='{question}'")
        return None

    candidates.sort(key=lambda c: (c["matches"], c["score"]), reverse=True)
    best = candidates[0]
    print(f"[DATASET_ONLY] ✅ lang={lang} matches={best['matches']} score={best['score']:.2f}")
    return best["answer"]


# ══════════════════════════════════════════════════════════════════════
# RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════
async def rag_search_async(question: str, lang: str = "en") -> dict:
    sources       = []
    used_internet = False
    try:
        bm25_results = bm25_search(query=question, top_k=5)
        collections  = get_collection_for_query(question, lang)

        if lang == "ga":
            if "kb_ga" in COLLECTIONS and "kb_ga" not in collections:
                collections = ["kb_ga"] + [c for c in collections if c not in ("kb_hi", "kb_ku")]
        elif lang == "ku":
            if "kb_ku" in COLLECTIONS and "kb_ku" not in collections:
                collections = ["kb_ku"] + [c for c in collections if c not in ("kb_hi", "kb_ga")]
        elif lang == "hi":
            if "kb_hi" not in collections:
                collections.insert(0, "kb_hi")
        elif lang == "en":
            if "kb_en" not in collections and "kb_en" in COLLECTIONS:
                collections.insert(0, "kb_en")

        if "website" not in collections:
            collections.append("website")

        vector      = get_embed_model().embed_query(question)
        lang_filter = lang if lang in ("en", "hi", "ga", "ku") else None

        vector_results = multi_collection_search(
            client=get_client(), collections=collections,
            query_vector=vector, query_text=question,
            limit=5, lang_filter=lang_filter,
        )
        merged = reciprocal_rank_fusion(
            bm25_results=bm25_results,
            vector_results=vector_results,
            bm25_weight=0.4, vector_weight=0.6,
        )

        if _HAS_RERANKER and merged:
            merged    = rerank_with_diversity(results=merged, query=question, top_k=3)
            score_key = "rerank_score"
        else:
            score_key = "rrf_score"

        ctx_parts = []
        for r in merged[:3]:
            url = r.get("url") or r.get("metadata", {}).get("source", "")
            if url and url.startswith("http"):
                sources.append(url)
            ctx_parts.append(f"[Score: {r.get(score_key, 0):.3f}]\n{r['text']}")

        top_score = merged[0].get(score_key, 0) if merged else 0
        if top_score < 0.05 or not merged:
            print(f"[RAG] Low score ({top_score:.3f}) — trying internet...")
            internet_results = search_college_website(question)
            if internet_results:
                used_internet = True
                for r in internet_results[:2]:
                    ctx_parts.append(f"[Web]\n{r['snippet']}\nSource: {r['url']}")
                    sources.append(r["url"])

        if not ctx_parts:
            return {"context": None, "sources": [], "used_internet": False}

        return {
            "context":       "\n\n---\n\n".join(ctx_parts),
            "sources":       list(dict.fromkeys(sources)),
            "used_internet": used_internet,
        }
    except Exception as e:
        print(f"[RAG] Error: {e}")
        return {"context": None, "sources": [], "used_internet": False}


def rag_search(question: str, lang: str = "en"):
    try:
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


# ══════════════════════════════════════════════════════════════════════
# STRICT LANGUAGE SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════
LANG_SYSTEM_PROMPTS = {
    "en": (
        f"You are Diksha, the official AI assistant for GBPIET ({GBPIET_URL}). "
        "STRICT RULES: "
        "1. Respond in ENGLISH ONLY. "
        "2. NEVER start with 'Respected', 'Dear', or any salutation. "
        "3. NEVER start answer with your own name like 'I am Diksha' — start directly with the answer. "
        "4. NEVER repeat the question. Start the answer directly. "
        "5. Be concise, accurate and helpful."
    ),
    "hi": (
        f"तुम दीक्षा हो — GBPIET ({GBPIET_URL}) की official AI chatbot। "
        "सख्त नियम: "
        "1. केवल और केवल हिंदी में जवाब दो। "
        "2. 'Respected', 'Dear', 'प्रिय' जैसे शब्दों से शुरू मत करो। "
        "3. अपना नाम लेकर शुरू मत करो जैसे 'मैं दीक्षा हूँ' — सीधे जवाब दो। "
        "4. सवाल दोबारा मत लिखो — सीधे जवाब दो। "
        "5. Feminine grammar: सकती हूँ, करूँगी, जानती हूँ। "
        "6. User को 'आप' कहो।"
    ),
    "ga": (
        f"You are दीक्षा, official AI of GBPIET ({GBPIET_URL}). "
        "STRICT RULES: "
        "1. Always respond in Garhwali ONLY. Never Hindi or English. "
        "2. NEVER start answer with your own name like 'दीक्षा छुं' — start directly with the answer. "
        "3. NEVER start with 'Respected', 'Dear' or any salutation. "
        "4. NEVER repeat the question. "
        "5. Use Garhwali words: छन, छ, अर, कुण, बटि, मिलद, हूँद, कनकै। "
        "6. Address user as 'आप'. "
        "7. Use feminine Garhwali grammar. "
        f"8. If answer not found: माफ़ करया जी, मीथे यु जानकारी नी च। {GBPIET_URL} पर जावा।"
    ),
    "ku": (
        f"You are दीक्षा, official AI of GBPIET ({GBPIET_URL}). "
        "STRICT RULES: "
        "1. Always respond in Kumauni ONLY. Never Hindi or English. "
        "2. NEVER start answer with your own name like 'दीक्षा छु' — start directly with the answer. "
        "3. NEVER start with 'Respected', 'Dear' or any salutation. "
        "4. NEVER repeat the question. "
        "5. Use Kumauni words: छु, छन, राछ, हुनी, कनाँ, कैं, बेर, लै, बटा, ज्याणी। "
        "6. Address user as 'आप' or 'ज्यू'. "
        "7. Use feminine Kumauni grammar. "
        f"8. If answer not found: माफ़ करिया जी, मीकें यु जानकारी नैं च। {GBPIET_URL} पर जाया।"
    ),
}


# ══════════════════════════════════════════════════════════════════════
# LLM PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════
def build_prompt(question: str, context: str, lang: str, history: str = "") -> str:
    no_answer = {
        "en": f"Sorry, I couldn't find that. Please visit {GBPIET_URL} or call 01368-228030.",
        "hi": f"माफ़ करें, यह जानकारी नहीं मिली। कृपया {GBPIET_URL} देखें।",
        "ga": f"माफ़ करया जी, मीथे यु जानकारी नी च। {GBPIET_URL} पर जावा।",
        "ku": f"माफ़ करिया जी, मीकें यु जानकारी नैं च। {GBPIET_URL} पर जाया।",
    }
    lang_instruction = {
        "en": "Answer in ENGLISH ONLY. NEVER start with your name.",
        "hi": "केवल हिंदी में जवाब दो। अपने नाम से शुरू मत करो।",
        "ga": "केवल गढ़वाली भाषा मा जवाब दे। अपणे नाम से शुरू नि करण। गढ़वाली शब्द: छन, छ, अर, कुण, बटि, कनकै।",
        "ku": "केवल कुमाउनी भाषा मा जवाब दे। अपणे नाम से शुरू नि करण। कुमाउनी शब्द: छु, छन, लै, बटा, कनाँ, ज्याणी।",
    }
    return f"""{lang_instruction.get(lang, lang_instruction['en'])}
NEVER repeat the question. Start answer directly.
NEVER use 'Respected', 'Dear', or introduce yourself.
If not in context: {no_answer.get(lang, no_answer['en'])}
Website: {GBPIET_URL}
{history}
Context:
{context}

Question: {question}
Answer ({lang.upper()} ONLY — direct, no self-intro, no salutation):"""


# ══════════════════════════════════════════════════════════════════════
# LANGUAGE DRIFT DETECTOR + ENFORCER
# ══════════════════════════════════════════════════════════════════════
def detect_response_lang(text: str) -> str:
    if any(m in text for m in GA_MARKERS): return "ga"
    if any(m in text for m in KU_MARKERS): return "ku"
    devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    latin      = sum(1 for c in text if c.isascii() and c.isalpha())
    total      = devanagari + latin
    if total == 0: return "en"
    return "en" if (latin / total) > 0.6 else "hi"


def enforce_language(response: str, expected_lang: str) -> str:
    if expected_lang not in ("ga", "ku"):
        return response

    actual = detect_response_lang(response)

    if expected_lang == "ga" and actual != "ga":
        print(f"[LANG GUARD] Garhwali drift (got '{actual}'), correcting...")
        correction = (
            f"Translate this answer into Garhwali language only. "
            f"Use: छन, छ, अर, कुण, बटि, मिलद, हूँद, कनकै, यु। "
            f"Do NOT write in Hindi. ONLY Garhwali.\n\nAnswer:\n{response}"
        )
        retry = groq_call(
            messages=[
                {"role": "system", "content": LANG_SYSTEM_PROMPTS["ga"]},
                {"role": "user",   "content": correction},
            ],
            max_tokens=400, temperature=0.2,
        )
        if retry:
            return clean_response(retry)

    elif expected_lang == "ku" and actual in ("hi", "en"):
        print(f"[LANG GUARD] Kumauni drift (got '{actual}'), correcting...")
        correction = (
            f"Translate this answer into Kumauni language only. "
            f"Use: छु, छन, राछ, हुनी, कनाँ, कैं, लै, बटा, ज्याणी। "
            f"Do NOT write in Hindi. ONLY Kumauni.\n\nAnswer:\n{response}"
        )
        retry = groq_call(
            messages=[
                {"role": "system", "content": LANG_SYSTEM_PROMPTS["ku"]},
                {"role": "user",   "content": correction},
            ],
            max_tokens=400, temperature=0.2,
        )
        if retry:
            return clean_response(retry)

    return response


# ══════════════════════════════════════════════════════════════════════
# LLM ANSWER
# ══════════════════════════════════════════════════════════════════════
def llm_answer(question: str, context: str, lang: str, history: str = "") -> str:
    prompt = build_prompt(question, context, lang, history)
    system = LANG_SYSTEM_PROMPTS.get(lang, LANG_SYSTEM_PROMPTS["en"])

    result = groq_call(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=500, temperature=0.3,
    )

    if result:
        result = clean_response(result)
        result = enforce_language(result, lang)
        return result

    if context:
        lines = [l.strip() for l in context.split('\n') if len(l.strip()) > 30]
        if lines:
            return clean_response(lines[0]) + f"\n\nFor more info: {GBPIET_URL}"

    return f"Sorry, I couldn't generate a response. Please visit {GBPIET_URL} or call 01368-228030."


# ══════════════════════════════════════════════════════════════════════
# MAIN ENTRY
# ══════════════════════════════════════════════════════════════════════
def get_answer(question: str, lang: str = "en", history: str = "") -> str:
    question = question.strip()

    q_lower   = question.lower().strip()
    q_no_name = re.sub(r'\b(diksha|disha|dixa|दीक्षा)\b', '', q_lower).strip()

    # ── Greeting ──────────────────────────────────────────────────────
    if q_lower in GREETINGS or q_no_name in GREETINGS:
        print("[RESULT] Greeting")
        return clean_response(GREETING_RESPONSE.get(lang, GREETING_RESPONSE["en"]))

    # ── Identity ──────────────────────────────────────────────────────
    if q_lower in IDENTITY_Q or q_no_name in IDENTITY_Q:
        print("[RESULT] Identity")
        return clean_response(IDENTITY_RESPONSE.get(lang, IDENTITY_RESPONSE["en"]))

    # ── User introducing themselves ───────────────────────────────────
    _intro_words = {"i am", "my name is", "mera naam", "naam hai", "naam h"}
    if any(iw in q_lower for iw in _intro_words):
        _name = q_lower.split()[-1].capitalize()
        _intro_resp = {
            "en": f"Hello {_name}! How can I help you? Ask me anything about GBPIET.",
            "hi": f"नमस्ते {_name} जी! GBPIET के बारे में कुछ भी पूछ सकते हैं।",
            "ga": f"समन्या {_name} जी! GBPIET बारे माँ कुछ भी पुछि सकदन।",
            "ku": f"नमस्कार {_name} जी! GBPIET बारे मा कुछ भी पूछ सकदन।",
        }
        print(f"[RESULT] User intro — name: {_name}")
        return _intro_resp.get(lang, _intro_resp["en"])

    # ── Out-of-scope ──────────────────────────────────────────────────
    if is_out_of_scope(question):
        print("[RESULT] Out of scope")
        return OUT_OF_SCOPE_RESPONSE.get(lang, OUT_OF_SCOPE_RESPONSE["en"])

    print(f"\n{'='*55}\n[Q/{lang}] {question}\n{'='*55}")

    build_bm25_index()

    # ── Step 0a: Specific role ────────────────────────────────────────
    ans = specific_role_answer(question, preferred_lang=lang)
    if ans:
        print("[RESULT] Specific role match")
        raw = clean_response(translate_answer_if_needed(ans, lang, question))
        return enforce_language(raw, lang)

    # ── Step 0b: Direct keyword ───────────────────────────────────────
    ans = direct_keyword_answer(question, preferred_lang=lang)
    if ans:
        print("[RESULT] Direct keyword")
        raw = clean_response(translate_answer_if_needed(ans, lang, question))
        return enforce_language(raw, lang)

    # ── Step 1: Exact match ───────────────────────────────────────────
    ans = exact_match(question)
    if ans:
        print("[RESULT] Exact match")
        raw = clean_response(translate_answer_if_needed(ans, lang, question))
        return enforce_language(raw, lang)

    # ── Step 2: Keyword match ─────────────────────────────────────────
    word_count = len(question.split())
    thresh     = 1 if word_count <= 2 else (2 if word_count <= 5 else 3)
    ans        = keyword_match(question, thresh, preferred_lang=lang)
    if ans:
        print("[RESULT] Keyword match")
        raw = clean_response(translate_answer_if_needed(ans, lang, question))
        return enforce_language(raw, lang)

    # ── Step 3: ga/ku → Dataset ONLY, no LLM ─────────────────────────
    if lang in ("ga", "ku"):
        ans = dataset_only_search(question, lang)
        if ans:
            print(f"[RESULT] Dataset-only match ({lang})")
            return clean_response(ans)
        print(f"[RESULT] No dataset match for lang={lang}")
        fb_ga_ku = {
            "ga": f"माफ़ करया जी, मीथे यु जानकारी नी च। {GBPIET_URL} पर जावा या 01368-228030 पर फोन कर्या।",
            "ku": f"माफ़ करिया जी, मीकें यु जानकारी नैं च। {GBPIET_URL} पर जाया या 01368-228030 पर फोन करिया।",
        }
        return fb_ga_ku[lang]

    # ── Step 3: hi/en → RAG + LLM ────────────────────────────────────
    ctx = rag_search(question, lang)
    if ctx:
        print("[RESULT] RAG + LLM")
        return llm_answer(question, ctx, lang, history)

    # ── No match ──────────────────────────────────────────────────────
    print("[RESULT] No match")
    fb = {
        "hi": f"माफ़ करें, यह जानकारी नहीं मिली। कृपया {GBPIET_URL} देखें या 01368-228030 पर कॉल करें।",
        "ga": f"माफ़ करया जी, मीथे यु जानकारी नी च। {GBPIET_URL} पर जावा या 01368-228030 पर फोन कर्या।",
        "ku": f"माफ़ करिया जी, मीकें यु जानकारी नैं च। {GBPIET_URL} पर जाया या 01368-228030 पर फोन करिया।",
        "en": f"Sorry, I couldn't find that information. Please visit {GBPIET_URL} or call 01368-228030.",
    }
    return fb.get(lang, fb["en"])
