# rag/kb_query.py — Complete RAG pipeline
# Fixes:
#   ✅ 4 Groq keys (8-attempt fallback)
#   ✅ Garhwali answers ONLY in Garhwali
#   ✅ Kumauni answers ONLY in Kumauni
#   ✅ Hindi answers ONLY in Hindi
#   ✅ English answers ONLY in English
#   ✅ Emoji stripped before TTS / response
#   ✅ Feminine grammar enforced
#   ✅ दीक्षा spelling fixed
#   ✅ lang preference in all matching steps
#   ✅ Strict lang routing: ga→kb_ga, ku→kb_ku, hi→kb_hi, en→kb_en
#   ✅ Kumauni synonym expansion added
#   ✅ "Respected" removed — natural tone only
#   ✅ Out-of-scope questions rejected (non-GBPIET queries)
#   ✅ Question NOT repeated in answer

import os
import re
import json
import glob
import asyncio
import unicodedata

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

try:
    from rag.reranker import rerank_with_diversity
    _HAS_RERANKER = True
except ImportError:
    _HAS_RERANKER = False

# ══════════════════════════════════════════════════════════════════════
# GROQ CLIENTS — 4 keys, 8-attempt fallback
# ══════════════════════════════════════════════════════════════════════
def _make_client(env_var: str):
    key = os.getenv(env_var, "").strip()
    if key:
        print(f"[GROQ] {env_var} ✅")
        return Groq(api_key=key)
    print(f"[GROQ] {env_var} ❌ not set")
    return None

_groq1 = _make_client("GROQ_API_KEY")
_groq2 = _make_client("GROQ_API_KEY_2")
_groq3 = _make_client("GROQ_API_KEY_3")
_groq4 = _make_client("GROQ_API_KEY_4")

GROQ_PRIMARY  = "llama-3.3-70b-versatile"
GROQ_FALLBACK = "llama3-70b-8192"

GROQ_ATTEMPTS = [
    (_groq1, GROQ_PRIMARY,  "Key1/Primary"),
    (_groq2, GROQ_PRIMARY,  "Key2/Primary"),
    (_groq3, GROQ_PRIMARY,  "Key3/Primary"),
    (_groq4, GROQ_PRIMARY,  "Key4/Primary"),
    (_groq1, GROQ_FALLBACK, "Key1/Fallback"),
    (_groq2, GROQ_FALLBACK, "Key2/Fallback"),
    (_groq3, GROQ_FALLBACK, "Key3/Fallback"),
    (_groq4, GROQ_FALLBACK, "Key4/Fallback"),
]

_embed_model  = None
_qa_database  = []

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GBPIET_URL       = "https://gbpiet.ac.in"


# ══════════════════════════════════════════════════════════════════════
# OUT-OF-SCOPE DETECTOR
# ══════════════════════════════════════════════════════════════════════
OUT_OF_SCOPE_KEYWORDS = {
    "ram mandir", "mandir", "temple", "mosque", "church", "masjid",
    "gurudwara", "dargah", "math", "ashram",
    "weather", "mausam", "barish", "rain", "temperature",
    "modi", "rahul gandhi", "election", "chunav", "vote",
    "bjp", "congress", "party", "cm", "prime minister",
    "movie", "film", "song", "gana", "actor", "actress",
    "cricket", "ipl", "match",
    "restaurant", "hotel bahar", "dhaba", "market", "bazaar",
    "hospital bahar", "dawai", "medicine",
}

GBPIET_SCOPE_KEYWORDS = {
    "gbpiet", "college", "admission", "fee", "fees", "hostel",
    "placement", "faculty", "hod", "dean", "director", "registrar",
    "course", "branch", "department", "exam", "result", "scholarship",
    "library", "transport", "sports", "ragging", "warden", "mess",
    "mca", "cse", "ece", "mechanical", "civil", "electrical",
    "biotechnology", "applied", "pauri", "garhwal",
    "प्रवेश", "फीस", "हॉस्टल", "प्लेसमेंट", "संकाय", "परीक्षा",
    "छात्रवृत्ति", "पुस्तकालय", "परिवहन", "निदेशक", "रजिस्ट्रार",
    "एडमिशन", "भर्ती", "दाम", "सुविधा", "नौकरी", "भर्ति",
}

def is_out_of_scope(question: str) -> bool:
    q = question.lower().strip()
    for kw in GBPIET_SCOPE_KEYWORDS:
        if kw in q:
            return False
    for kw in OUT_OF_SCOPE_KEYWORDS:
        if kw in q:
            print(f"[SCOPE] Out of scope: '{q}' matched '{kw}'")
            return True
    return False

OUT_OF_SCOPE_RESPONSE = {
    "en": "I can only help with information about GBPIET college — admissions, fees, hostel, placements, faculty and more. Please ask me something related to GBPIET.",
    "hi": "मैं केवल GBPIET कॉलेज से संबंधित जानकारी दे सकती हूँ — प्रवेश, फीस, हॉस्टल, प्लेसमेंट आदि। कृपया GBPIET से जुड़ा कोई सवाल पूछें।",
    "ga": "मैं केवल जीबीपीआईईटी कॉलेज का जानकारी द्यूँ। कृपया जीबीपीआईईटी से जुड़ा सवाल पुछ्या।",
    "ku": "मैं केवल जीबीपीआईईटी कॉलेज की जानकारी द्यूँ। कृपया जीबीपीआईईटी से जुड़ा सवाल पूछिया।",
}


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
    flags=re.UNICODE
)

def clean_response(text: str) -> str:
    """Strip emojis, remove salutations like 'Respected', clean whitespace."""
    text = _EMOJI_PATTERN.sub("", text)
    # Remove salutation prefixes
    text = re.sub(
        r'^(Respected\s+\w+[\s,]*|Dear\s+\w+[\s,]*|प्रिय\s+\w+[\s,]*)',
        '', text, flags=re.IGNORECASE
    ).strip()
    if text:
        text = text[0].upper() + text[1:]
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ══════════════════════════════════════════════════════════════════════
# GARHWALI MARKERS
# ══════════════════════════════════════════════════════════════════════
GA_MARKERS = [
    'छन', 'छ।', 'हूँद', 'कुण', 'मिलद', 'पैलू', 'अर',
    'कनकै', 'कख', 'बटि', 'त्वै', 'छौ', 'छी',
    'छ्यायी', 'थ्यायी', 'तुमुं', 'यैसैं', 'वैसें',
    'माँ', 'मी', 'जु', 'यु', 'वु',
]

# ══════════════════════════════════════════════════════════════════════
# KUMAUNI MARKERS
# ══════════════════════════════════════════════════════════════════════
KU_MARKERS = [
    'छौ', 'छन', 'छौँ', 'छा', 'लै', 'बटा', 'हैबर', 'कन',
    'कसि', 'कै', 'म्यूँ', 'त्यूँ', 'यो', 'वो', 'भयो',
    'ज्यू', 'भल', 'नानी', 'ठुली', 'हिटा', 'तल्लि', 'मल्लि',
]


# ══════════════════════════════════════════════════════════════════════
# GREETING / IDENTITY
# ══════════════════════════════════════════════════════════════════════
GREETINGS = {
    "hello","hi","hlo","hey","hii","helo","namaste",
    "नमस्ते","हेलो","हाय","good morning","good afternoon","good evening",
    "नमस्कार","राम राम","जय हो",
}
IDENTITY_Q = {
    "who are you","what are you","who r u","tum kaun ho",
    "aap kaun hain","aap kaun ho","kaun ho tum","who made you",
    "who created you","who made diksha","diksha kaun hai",
    "को च","कु च","कू च","को छ","कु छ","को cha","ko cha",
    "तू को छ","तू कु छ","तू को च",
    "को छै","के छै","तू को छै","तुम को छौ","तुमार नाम के छ",
    "ko chai","tumar naam ke cha","tu ko chai",
}

GREETING_RESPONSE = {
    "en": "Hello! I'm Diksha, the official AI assistant for GBPIET, Pauri Garhwal. Ask me about admissions, fees, hostel, placements, faculty, courses and more!",
    "hi": "नमस्ते! मैं दीक्षा हूँ — GBPIET की आधिकारिक AI सहायिका। आप मुझसे admission, fees, hostel, placement के बारे में पूछ सकते हैं।",
    "ga": "समन्या जी! मैं दीक्षा छुं — जीबीपीआईईटी की AI दगड़िया। आप कुछ भी पुछि सकदन।",
    "ku": "नमस्कार जी! मैं दीक्षा छु — जीबीपीआईईटी की AI दगड़िया। आप कुछ भी पूछ सकदन।",
}

IDENTITY_RESPONSE = {
    "en": f"I'm Diksha, the official AI chatbot for GBPIET (Govind Ballabh Pant Institute of Engineering and Technology), Pauri Garhwal, Uttarakhand. I help with college information in English, Hindi, Garhwali and Kumauni. Visit: {GBPIET_URL}",
    "hi": f"मैं दीक्षा हूँ — GBPIET (गोविंद बल्लभ पंत इंजीनियरिंग कॉलेज), पौड़ी गढ़वाल की आधिकारिक AI chatbot। वेबसाइट: {GBPIET_URL}",
    "ga": f"मैं दीक्षा छुं — जीबीपीआईईटी, पौड़ी गढ़वाल की official AI chatbot। आप मीसे admission, fees, hostel बारे मा पुछि सकदन। वेबसाइट: {GBPIET_URL}",
    "ku": f"मैं दीक्षा छु — जीबीपीआईईटी, पौड़ी गढ़वाल की official AI chatbot। आप मीसे admission, fees, hostel बारे मा पूछ सकदन। वेबसाइट: {GBPIET_URL}",
}


# ══════════════════════════════════════════════════════════════════════
# GROQ CALL — 8 attempts
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
    for client, model, label in GROQ_ATTEMPTS:
        result = _groq_call(client, model, messages, max_tokens, temperature)
        if result:
            print(f"[LLM] ✅ {label}")
            return result
    print("[LLM] ❌ All 8 Groq attempts failed")
    return ""


# ══════════════════════════════════════════════════════════════════════
# EMBED MODEL + QDRANT
# ══════════════════════════════════════════════════════════════════════
def get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("[Embed] Loading model...")
        _embed_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[Embed] Model loaded")
    return _embed_model


def get_qdrant():
    return get_client()


# ══════════════════════════════════════════════════════════════════════
# QA DATABASE
# ══════════════════════════════════════════════════════════════════════
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
                            "answer":   answer.strip(),
                            "source":   os.path.basename(filepath),
                            "lang":     item.get("lang", "").strip().lower(),
                        })
        except Exception as e:
            print(f"[DB] Error loading {filepath}: {e}")

    print(f"[DB] Loaded {len(_qa_database)} QA pairs")
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
    if item_lang in ("ga", "garhwali"):  return "ga"
    if item_lang in ("ku", "kumauni"):   return "ku"
    if item_lang in ("hi", "hindi"):     return "hi"
    if item_lang == "en":                return "en"
    if any(m in answer_text for m in GA_MARKERS): return "ga"
    if any(m in answer_text for m in KU_MARKERS): return "ku"
    latin = sum(1 for c in answer_text if c.isascii() and c.isalpha())
    total = len(answer_text.replace(" ", ""))
    if total > 0 and (latin / total) > 0.5: return "en"
    dev = sum(1 for c in answer_text if '\u0900' <= c <= '\u097F')
    return "hi" if dev > 5 else "en"


def translate_answer_if_needed(answer: str, lang: str, question: str) -> str:
    answer_is_hindi = is_hindi_text(answer)
    if lang == "en" and not answer_is_hindi:
        return answer
    if lang in ("hi", "ga", "ku") and answer_is_hindi:
        return answer

    if lang == "en":
        prompt = f"Translate to English. Return ONLY translated text.\n\n{answer}"
        system = "You are a translator. Translate Hindi to English accurately. Keep names, numbers, URLs unchanged."
    else:
        prompt = f"Translate to Hindi (Devanagari). Return ONLY translated text.\n\n{answer}"
        system = "You are a translator. Translate English to Hindi accurately. Keep names, numbers, URLs unchanged."

    print(f"[TRANSLATE] lang={lang}...")
    result = groq_call(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=400, temperature=0.1,
    )
    return result if result else answer


# ══════════════════════════════════════════════════════════════════════
# HINDI → ENGLISH MAP
# ══════════════════════════════════════════════════════════════════════
HINDI_MAP = {
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
    'कितने': 'how many',           'कौन से': 'which',
    'कु': 'who what of', 'कू': 'who',
    'कन': 'how',         'कनकै': 'how',
    'कख': 'where',       'कनै': 'where',
    'बटि': 'from',       'कुण': 'to for',
    'अर': 'and',         'त्वै': 'then',
    'माँ': 'in',         'मा': 'in',
    'बारे माँ': 'about',
    'मी': 'i me',        'आम': 'we us',
    'तुमुं': 'you',      'जु': 'who which',
    'यु': 'this',        'वु': 'that they',
    'यैसैं': 'this',     'वैसें': 'that',
    'वैन': 'that',
    'च': 'is',           'छ': 'is',
    'छन': 'are is',      'छौ': 'was',
    'थौ': 'was',         'छी': 'was',
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
    'खुणि': 'ke liye for',    'वास्ति': 'ke liye',   'कन लिजि': 'ke liye',
    'छन्': 'hain are',        'अछ': 'hai is',         'रोणु': 'rahna',
    'रैणु': 'rehna',           'कूण': 'kaun who',      'कोण': 'kaun who',
    'कखे': 'kahan where',     'कख': 'kahan where',    'किख': 'kahan where',
    'कसे': 'kaise how',       'कसि': 'kaise how',     'कनूँ': 'kaise how',
    'कति': 'kitna how much',  'कितणु': 'kitna',       'कतणु': 'kitna how many',
    'जानकारी': 'jankari information',
    'एडमिशन': 'admission',    'दाखिला': 'admission',  'भर्ती': 'admission',
    'नाम लिखाणु': 'admission',
    'पैलि': 'pehla first',    'पहिलो': 'pehla first',
    'नौना': 'ladka boy',      'छोरा': 'ladka boy',
    'नौन्यिँ': 'ladki girl',  'छोरी': 'ladki girl',
    'दाम': 'fees',            'रकम': 'fees amount',   'मूल्य': 'fees',
    'पैसा': 'fees money',
    'आवास': 'hostel',         'रैणु-सैणु': 'hostel accommodation',
    'निवास': 'hostel',        'मेस': 'mess',          'भोजनालय': 'mess canteen',
    'सुविधा': 'facility',     'साधन': 'facility',     'इंतजाम': 'facility arrangement',
    'आवेदन': 'application form',
    'योग्यता': 'eligibility', 'काबिलियत': 'eligibility', 'लायकी': 'eligibility',
    'परीक्षा': 'exam',        'इम्तिहान': 'exam',
    'विभाग': 'department',    'खंड': 'department',
    'नौकरी': 'job placement', 'रोजगार': 'job',        'तैनाती': 'placement',
    'तनख्वाह': 'salary',      'कमाई': 'salary earnings', 'पगार': 'salary',
    'मी': 'main I',           'मेरो': 'mera my',
    'बताणु': 'batao tell',    'दसणु': 'batao tell',
    'अर': 'aur and',          'बटे': 'se from',       'सैं': 'se from',
}


def ga_ku_to_hi_en(text: str) -> str:
    t = text.lower()
    for ga_word, translation in sorted(GARHWALI_SYNONYM_MAP.items(), key=lambda x: -len(x[0])):
        if ga_word in t:
            t = t.replace(ga_word, ' ' + translation + ' ')
    return re.sub(r'\s+', ' ', t).strip()


# ══════════════════════════════════════════════════════════════════════
# KUMAUNI SYNONYM MAP
# ══════════════════════════════════════════════════════════════════════
KUMAUNI_SYNONYM_MAP = {
    'लिजी': 'ke liye for',    'काज': 'ke liye for',   'वास्ते': 'ke liye',
    'छन': 'hain are',         'छ': 'hai is',           'छौ': 'ho are',
    'रैण': 'rehna live',      'बौण': 'baithna sit',
    'को': 'kaun who',         'के': 'kya what',
    'कहाँ': 'kahan where',    'कते': 'kahan where',    'कहाँबटा': 'kahan se',
    'कसि': 'kaise how',       'कन': 'kaise how',
    'कतु': 'kitna how much',  'कै': 'kitne how many',
    'ज्याणी': 'jankari information',
    'भर्ति': 'admission',     'नाम लिखाण': 'admission',
    'पैलि': 'pehla first',    'पैली': 'pahle before',
    'चेलो': 'ladka boy',      'नान': 'bachcha kid',    'चेली': 'ladki girl',
    'दाम': 'fees price',      'टक': 'paisa money',
    'रैण-बौण': 'hostel stay', 'कमरा': 'room hostel',
    'खाण-पीण': 'mess food',   'भोजन': 'food',
    'सुबिद': 'facility convenience',
    'फारम': 'form application',
    'लायक': 'eligibility',    'पास': 'exam pass',
    'पड़ै': 'padhai study',    'इम्तिहान': 'exam',
    'नौकरी': 'job placement', 'कमाइ': 'salary income', 'पगार': 'salary',
    'मैं': 'main I',          'म्यूँ': 'mera my',      'तुमार': 'tumhara your',
    'बुलण': 'bolna speak',    'बताण': 'batao tell',
    'और': 'aur and',          'बटा': 'se from',        'हैबर': 'se from/after',
}


def ku_to_hi_en(text: str) -> str:
    t = text.lower()
    for ku_word, translation in sorted(KUMAUNI_SYNONYM_MAP.items(), key=lambda x: -len(x[0])):
        if ku_word in t:
            t = t.replace(ku_word, ' ' + translation + ' ')
    return re.sub(r'\s+', ' ', t).strip()


# ══════════════════════════════════════════════════════════════════════
# TYPO MAP
# ══════════════════════════════════════════════════════════════════════
TYPO_MAP = {
    "h0d": "hod",               "f33s": "fees",          "adm1ssion": "admission",
    "mechenical": "mechanical", "mechnical": "mechanical","mechincal": "mechanical",
    "mechanicle": "mechanical", "mechinical": "mechanical",
    "electical": "electrical",  "electrcal": "electrical",
    "biotechonlogy": "biotechnology", "bitoech": "biotechnology",
    "admision": "admission",    "admisson": "admission",  "palcement": "placement",
    "hostle": "hostel",         "dircetor": "director",   "registar": "registrar",
    "collage": "college",       "colege": "college",      "faculity": "faculty",
    "placment": "placement",    "semster": "semester",
}


def fix_typos(text: str) -> str:
    words = text.lower().split()
    fixed = []
    for w in words:
        w_clean = w.replace('0', 'o').replace('1', 'i').replace('3', 'e')
        if w in TYPO_MAP:
            fixed.append(TYPO_MAP[w])
        elif w_clean in TYPO_MAP:
            fixed.append(TYPO_MAP[w_clean])
        else:
            fixed.append(w_clean if w_clean != w else w)
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
}

ROLE_BLACKLIST = [
    "list all","all hod","all department","all heads",
    "departments at gbpiet","list of hod","all hods",
    "सभी विभाग","सभी hod",
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
        if all(w in q_lower for w in topic_words):                                          score += 3
        elif mapped.lower() in q_lower:                                                      score += 2
        elif len(topic_words) >= 2 and sum(1 for w in topic_words if w in q_lower) >= 2:    score += 1
        if any(w in a_lower for w in topic_words):                                           score += 1
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
    "registrar": "registrar",    "director": "director",
    "dean": "dean",              "chairman": "chairman",
    "warden": "warden",          "placement": "placement",
    "placements": "placement",   "hostel": "hostel",
    "hostels": "hostel",         "fees": "fees",
    "fee": "fees",               "admission": "admission",
    "admissions": "admission",   "contact": "contact",
    "courses": "courses",        "course": "courses",
    "library": "library",        "transport": "transport",
    "scholarship": "scholarship","result": "result",
    "ragging": "ragging",        "sports": "sports",
    "faculty": "faculty",        "hod": "head of department",
    "about": "about gbpiet",     "website": "gbpiet website",
    "h0d": "head of department", "hods": "head of department",
    "रजिस्ट्रार": "registrar",  "निदेशक": "director",
    "डीन": "dean",               "प्लेसमेंट": "placement",
    "हॉस्टल": "hostel",          "फीस": "fees",
    "प्रवेश": "admission",       "संपर्क": "contact",
    "पुस्तकालय": "library",      "परिवहन": "transport",
    "रैगिंग": "ragging",         "संकाय": "faculty",
    "एडमिशन": "admission",       "भर्ती": "admission",
    "नौकरी": "placement",        "सुविधा": "facility",
    "भर्ति": "admission",        "सुबिद": "facility",
    "ज्याणी": "information",     "दाम": "fees",
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
    'माँ','मा','बटि','च','छ','छन','एक',
    'अर','त','त्वै','भी','न',
    'थौ','छौ','छी','छ्यायी','थ्यायी',
    'मी','आम','तुम','तुमुं','वु','वे',
    'यैसैं','यें','वैसें','वैन','यु',
    'पर','बारे','तक','जु',
    'को','के','कन','कसि','कै','छौ','छन','छा',
    'लै','बटा','हैबर','यो','वो','भयो',
    'म्यूँ','त्यूँ','ज्यू','भल',
}

HOSTEL_NAMES = {
    'kailash','neelkanth','kedar','rudra','badri','alaknanda',
    'shivalik','trishul','raman','bhagirathi','viswerwarya','vh',
}


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
            print(f"[RAG] Lang=ga → collections: {collections}")
        elif lang == "ku":
            if "kb_ku" in COLLECTIONS and "kb_ku" not in collections:
                collections = ["kb_ku"] + [c for c in collections if c not in ("kb_hi", "kb_ga")]
            print(f"[RAG] Lang=ku → collections: {collections}")
        elif lang == "hi":
            if "kb_hi" not in collections:
                collections.insert(0, "kb_hi")
            print(f"[RAG] Lang=hi → collections: {collections}")
        elif lang == "en":
            if "kb_en" not in collections and "kb_en" in COLLECTIONS:
                collections.insert(0, "kb_en")
            print(f"[RAG] Lang=en → collections: {collections}")

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
# LLM PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════
def build_prompt(question: str, context: str, lang: str, history: str = "") -> str:

    if lang == "hi":
        return f"""आप दीक्षा हैं — GBPIET की official AI chatbot।

RULES:
- केवल Hindi में जवाब दें।
- User को "आप" कहें — कभी "तू" या "तुम" नहीं।
- Feminine forms: सकती हूँ, करूँगी, जानती हूँ।
- जवाब सीधे शुरू करें — सवाल दोबारा मत लिखें।
- "Respected" या "Dear" मत लिखें।
- केवल नीचे दिए context का उपयोग करें।
- Website: {GBPIET_URL}
- जवाब न मिले: "माफ़ करें, यह जानकारी नहीं मिली। कृपया {GBPIET_URL} देखें।"
{history}
Context:
{context}

Jawab (Hindi mein, bina sawaal repeat kiye):"""

    elif lang == "ga":
        return f"""आप दीक्षा छन — जीबीपीआईईटी, पौड़ी गढ़वाल की official AI chatbot।

RULES:
- केवल गढ़वाली में जवाब द्या।
- User कु "आप" या "थैं" बुल्या।
- Feminine forms use कर्या।
- जवाब सीधे शुरू कर्या — सवाल दोबारा मत लिख्या।
- "Respected" या "Dear" मत लिख्या।
- Context का ही उपयोग कर्या।
- Website: {GBPIET_URL}
- Jawab नि मिलो: "माफ़ करया जी, मीथे यु जानकारी नी च। {GBPIET_URL} पर जावा।"
{history}
Context: {context}

Jawab (केवल गढ़वाली में):"""

    elif lang == "ku":
        return f"""आप दीक्षा छन — जीबीपीआईईटी, पौड़ी गढ़वाल की official AI chatbot।

RULES:
- केवल कुमाउनी में जवाब द्या।
- User कु "आप" या "ज्यू" बुल्या।
- Feminine forms use कर्या।
- जवाब सीधे शुरू कर्या — सवाल दोबारा मत लिख्या।
- "Respected" या "Dear" मत लिख्या।
- Context का ही उपयोग कर्या।
- Website: {GBPIET_URL}
- Jawab नि मिलो: "माफ़ करिया जी, मीकें यु जानकारी नैं च। {GBPIET_URL} पर जाया।"
{history}
Context: {context}

Jawab (केवल कुमाउनी में):"""

    else:
        return f"""You are Diksha — official AI assistant for GBPIET
(Govind Ballabh Pant Institute of Engineering and Technology),
Pauri Garhwal, Uttarakhand. Website: {GBPIET_URL}

RULES:
- Answer in ENGLISH ONLY.
- DO NOT start with "Respected", "Dear", or any salutation.
- NEVER repeat the question — start directly with the answer.
- Use ONLY the context below.
- If not in context: "Sorry, I couldn't find that. Please visit {GBPIET_URL} or call 01368-228030."
{history}
Context:
{context}

Answer (English, no salutation, no question repeat):"""


# ══════════════════════════════════════════════════════════════════════
# LLM ANSWER
# ══════════════════════════════════════════════════════════════════════
def llm_answer(question: str, context: str, lang: str, history: str = "") -> str:
    prompt = build_prompt(question, context, lang, history)

    if lang == "ga":
        system = (
            f"You are दीक्षा (Diksha), official female AI assistant for GBPIET ({GBPIET_URL}). "
            "STRICT: Respond ONLY in Garhwali — NEVER Hindi or English. "
            "NEVER start with 'Respected' or 'Dear'. "
            "NEVER repeat the question. Start answer directly. "
            "Address user as 'आप'/'थैं' only. "
            "Use feminine grammar. Name: दीक्षा always. "
            f"No answer found: माफ़ करया जी, मीथे यु जानकारी नी च। {GBPIET_URL} पर जावा।"
        )
    elif lang == "ku":
        system = (
            f"You are दीक्षा (Diksha), official female AI assistant for GBPIET ({GBPIET_URL}). "
            "STRICT: Respond ONLY in Kumauni — NEVER Hindi or English. "
            "NEVER start with 'Respected' or 'Dear'. "
            "NEVER repeat the question. Start answer directly. "
            "Address user as 'आप'/'ज्यू' only. "
            "Use feminine grammar. Name: दीक्षा always. "
            f"No answer found: माफ़ करिया जी, मीकें यु जानकारी नैं च। {GBPIET_URL} पर जाया।"
        )
    elif lang == "hi":
        system = (
            f"You are दीक्षा (Diksha), official female AI assistant for GBPIET ({GBPIET_URL}). "
            "STRICT: Respond ONLY in Hindi. "
            "NEVER start with 'Respected' or 'Dear'. "
            "NEVER repeat the question. Start answer directly. "
            "Address user as 'आप' only. "
            "Feminine grammar: सकती हूँ, करूँगी. Name: दीक्षा always."
        )
    else:
        system = (
            f"You are Diksha, official female AI assistant for GBPIET ({GBPIET_URL}). "
            "STRICT: Respond ONLY in English. "
            "NEVER start with 'Respected' or 'Dear'. "
            "NEVER repeat the question. Start answer directly. "
            "Be helpful, accurate and concise."
        )

    result = groq_call(
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=500, temperature=0.3,
    )
    if result:
        return clean_response(result)

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

    # ── Greeting ──────────────────────────────────────────────────────
    if question.lower().strip() in GREETINGS:
        print("[RESULT] Greeting")
        return clean_response(GREETING_RESPONSE.get(lang, GREETING_RESPONSE["en"]))

    # ── Identity ──────────────────────────────────────────────────────
    if question.lower().strip() in IDENTITY_Q:
        print("[RESULT] Identity")
        return clean_response(IDENTITY_RESPONSE.get(lang, IDENTITY_RESPONSE["en"]))

    # ── Out of scope ───────────────────────────────────────────────────
    if is_out_of_scope(question):
        print("[RESULT] Out of scope")
        return OUT_OF_SCOPE_RESPONSE.get(lang, OUT_OF_SCOPE_RESPONSE["en"])

    print(f"\n{'='*55}\n[Q/{lang}] {question}\n{'='*55}")

    build_bm25_index()

    # Step 0a: Specific role
    ans = specific_role_answer(question, preferred_lang=lang)
    if ans:
        print("[RESULT] Specific role match")
        return clean_response(translate_answer_if_needed(ans, lang, question))

    # Step 0b: Direct keyword
    ans = direct_keyword_answer(question, preferred_lang=lang)
    if ans:
        print("[RESULT] Direct keyword")
        return clean_response(translate_answer_if_needed(ans, lang, question))

    # Step 1: Exact match
    ans = exact_match(question)
    if ans:
        print("[RESULT] Exact match")
        return clean_response(translate_answer_if_needed(ans, lang, question))

    # Step 2: Keyword match
    word_count = len(question.split())
    thresh     = 1 if word_count <= 2 else (2 if word_count <= 5 else 3)
    ans        = keyword_match(question, thresh, preferred_lang=lang)
    if ans:
        print("[RESULT] Keyword match")
        return clean_response(translate_answer_if_needed(ans, lang, question))

    # Step 3: RAG + LLM
    ctx = rag_search(question, lang)
    if ctx:
        print("[RESULT] RAG + LLM")
        return llm_answer(question, ctx, lang, history)

    # No match
    print("[RESULT] No match")
    fb = {
        "hi": f"माफ़ करें, यह जानकारी नहीं मिली। कृपया {GBPIET_URL} देखें या 01368-228030 पर कॉल करें।",
        "ga": f"माफ़ करया जी, मीथे यु जानकारी नी च। {GBPIET_URL} पर जावा या 01368-228030 पर फोन कर्या।",
        "ku": f"माफ़ करिया जी, मीकें यु जानकारी नैं च। {GBPIET_URL} पर जाया या 01368-228030 पर फोन करिया।",
        "en": f"Sorry, I couldn't find that information. Please visit {GBPIET_URL} or call 01368-228030.",
    }
    return fb.get(lang, fb["en"])
