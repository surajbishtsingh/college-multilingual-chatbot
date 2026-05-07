# rag/kb_query.py — Complete RAG pipeline
import os
import json
import glob
import re
import asyncio

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

# ── Groq clients (dual key fallback) ─────────────────────────────────
_groq1     = Groq(api_key=os.getenv("GROQ_API_KEY", ""))
_groq2_key = os.getenv("GROQ_API_KEY_2", "")
_groq2     = Groq(api_key=_groq2_key) if _groq2_key else None

_embed_model = None
_qa_database = []

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── GBPIET website base URL ───────────────────────────────────────────
GBPIET_URL = "https://gbpiet.ac.in"

# ══════════════════════════════════════════════════════════════════════
# GREETING / IDENTITY
# ══════════════════════════════════════════════════════════════════════
GREETINGS = {
    "hello","hi","hlo","hey","hii","helo","namaste",
    "नमस्ते","हेलो","हाय","good morning","good afternoon","good evening"
}
IDENTITY_Q = {
    "who are you","what are you","who r u","tum kaun ho",
    "aap kaun hain","aap kaun ho","kaun ho tum","who made you",
    "who created you","who made diksha","diksha kaun hai"
}

GREETING_RESPONSE = {
    "en": "Hello! I'm Diksha, the official AI assistant for GBPIET, Pauri Garhwal. Ask me about admissions, fees, hostel, placements, faculty, courses and more!",
    "hi": "नमस्ते! मैं दीक्षा हूँ — GBPIET की आधिकारिक AI सहायक। आप मुझसे admission, fees, hostel, placement के बारे में पूछ सकते हैं।",
    "ga": "नमस्ते! मैं दीक्षा छू — GBPIET की AI सहायक। कुछ भी पूछो!",
    "ku": "नमस्ते! मैं दीक्षा छु — GBPIET की AI सहायक। कुछ भी पूछो!",
}

IDENTITY_RESPONSE = {
    "en": "I'm Diksha, the official AI chatbot for GBPIET (Govind Ballabh Pant Institute of Engineering and Technology), Pauri Garhwal, Uttarakhand. I can help you with college information in English, Hindi, Garhwali and Kumauni! Visit: https://gbpiet.ac.in",
    "hi": "मैं दीक्षा हूँ — GBPIET (गोविंद बल्लभ पंत इंजीनियरिंग कॉलेज), पौड़ी गढ़वाल की आधिकारिक AI chatbot। वेबसाइट: https://gbpiet.ac.in",
    "ga": "मैं दीक्षा छू — GBPIET की AI chatbot। वेबसाइट: https://gbpiet.ac.in",
    "ku": "मैं दीक्षा छु — GBPIET की AI chatbot। वेबसाइट: https://gbpiet.ac.in",
}

# ══════════════════════════════════════════════════════════════════════
# GROQ CALL — dual key fallback
# ══════════════════════════════════════════════════════════════════════
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
                        })
        except Exception as e:
            print(f"[DB] Error loading {filepath}: {e}")

    print(f"[DB] Loaded {len(_qa_database)} QA pairs")
    return _qa_database


# ══════════════════════════════════════════════════════════════════════
# LANGUAGE HELPERS
# ══════════════════════════════════════════════════════════════════════
def is_hindi_text(text):
    if not text:
        return False
    devanagari = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total      = len(text.replace(" ", ""))
    return total > 0 and (devanagari / total) > 0.2


def translate_answer_if_needed(answer, lang, question):
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
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        max_tokens=400, temperature=0.1,
    )
    return result if result else answer


# ══════════════════════════════════════════════════════════════════════
# HINDI → ENGLISH MAP
# ══════════════════════════════════════════════════════════════════════
HINDI_MAP = {
    'जीबीपीआईईटी': 'gbpiet',    'संस्थान': 'institute',
    'कॉलेज': 'college',          'पहुँचें': 'reach',
    'निदेशक': 'director',         'विभागाध्यक्ष': 'head department hod',
    'अध्यक्ष': 'chairman',        'डीन': 'dean',
    'संकाय': 'faculty',           'वार्डन': 'warden',
    'प्रवेश': 'admission',        'दाखिला': 'admission',
    'फीस': 'fees',                'शुल्क': 'fees',
    'छात्रवृत्ति': 'scholarship', 'हॉस्टल': 'hostel',
    'छात्रावास': 'hostel',        'प्लेसमेंट': 'placement',
    'पुस्तकालय': 'library',       'परिवहन': 'transport',
    'परिणाम': 'result',            'रैगिंग': 'ragging',
    'संपर्क': 'contact',           'रजिस्ट्रार': 'registrar',
    'कुलसचिव': 'registrar',        'लड़कों': 'boys',
    'लड़कियों': 'girls',           'कितने': 'how many',
    'कौन से': 'which',             'कहाँ': 'where',
    'कैसे': 'how',
}


def hi_to_en(text):
    t = text.lower()
    for h, e in HINDI_MAP.items():
        t = t.replace(h, ' ' + e + ' ')
    return re.sub(r'\s+', ' ', t).strip()


# ══════════════════════════════════════════════════════════════════════
# GARHWALI / KUMAUNI → HINDI/ENGLISH MAP
# ══════════════════════════════════════════════════════════════════════
GARHWALI_SYNONYM_MAP = {
    # Purpose / For
    'खुणि':     'ke liye for',
    'वास्ति':   'ke liye',
    'कन लिजि':  'ke liye',
    'खातिर':    'ke liye',
    'निमित':    'ke liye',
    'हेतु':     'ke liye',

    # Is/Are/Exist
    'छन्':      'hain are',
    'अछ':       'hai is',
    'रोणु':     'rahna',
    'रैणु':     'rehna',
    'होणु':     'hona',
    'विद्यमान': 'hai is',

    # Who/Whose
    'कूण':      'kaun who',
    'किसको':    'whose',
    'कसको':     'whose',
    'कोण':      'kaun who',

    # Where
    'कखे':      'kahan where',
    'कख':       'kahan where',
    'किख':      'kahan where',
    'कतरफ':     'kahan direction',
    'किस ठौर':  'kahan where',

    # How
    'कसे':      'kaise how',
    'कसि':      'kaise how',
    'कनूँ':     'kaise how',
    'कसी भाँत': 'kaise how',

    # How much/many
    'कति':      'kitna how much',
    'कितणु':    'kitna',
    'कतणु':     'kitna how many',
    'कति सो':   'kitna',
    'किणा':     'kitna',

    # Information
    'जानकारी':  'jankari information',
    'विवरण':    'details information',
    'खबर':      'information',
    'बात':      'information',
    'सूचना':    'information',

    # Admission
    'एडमिशन':      'admission',
    'दाखिला':       'admission',
    'भर्ती':        'admission',
    'नाम लिखाणु':   'admission',
    'दाखल होणु':    'admission',

    # First/Previous
    'पैलि':     'pehla first',
    'पहिलो':    'pehla first',
    'अगेतु':    'pehle first',
    'पैल्यां':  'pehle first',

    # Boy/Girl
    'नौना':     'ladka boy',
    'छोरा':     'ladka boy',
    'बालक':     'ladka boy',
    'भ्वाइ':    'bhai brother',
    'नौन्यिँ':  'ladki girl',
    'छोरी':     'ladki girl',
    'बालिका':   'ladki girl',
    'दुलारी':   'ladki girl',

    # Fees
    'दाम':      'fees',
    'रकम':      'fees amount',
    'मूल्य':    'fees',
    'पैसा':     'fees money',
    'धन':       'fees',

    # Hostel
    'आवास':         'hostel',
    'रैणु-सैणु':    'hostel accommodation',
    'निवास':        'hostel',
    'रहाण को जगह':  'hostel',

    # Mess/Food
    'मेस':          'mess',
    'भोजनालय':      'mess canteen',
    'खाण को जगह':   'mess',
    'रसोई':         'mess kitchen',
    'थाळी':         'mess food',

    # Facility
    'सुविधा':   'facility',
    'साधन':     'facility',
    'इंतजाम':   'facility arrangement',
    'व्यवस्था': 'facility arrangement',
    'जुगाड़':   'arrangement',

    # Contact
    'संपर्क करा':  'contact karo',
    'मिला':         'contact',
    'फोन करा':      'phone contact',
    'जोड़ा':        'contact',

    # Application
    'आवेदन':        'application form',
    'फॉर्म भरा':    'form fill apply',
    'दरखास्त दया':  'apply',
    'अर्जी दया':    'apply',

    # Eligibility
    'योग्यता':      'eligibility',
    'काबिलियत':     'eligibility',
    'लायकी':        'eligibility',
    'हैसियत':       'eligibility',

    # Exam
    'परीक्षा':      'exam',
    'इम्तिहान':     'exam',
    'जाँच':         'exam test',
    'टेस्ट':        'test exam',
    'परख':          'test exam',

    # Department
    'विभाग':        'department',
    'खंड':          'department',
    'अनुभाग':       'department section',
    'टोली':         'department group',

    # Research
    'शोध':          'research',
    'अनुसंधान':     'research',
    'खोज':          'research search',
    'तलाश':         'search',

    # Documents
    'कागज-पत्तर':   'documents',
    'कागज':         'documents',
    'प्रमाण':       'proof documents',
    'चिट्ठी':       'letter document',

    # Placement
    'नौकरी':    'job placement',
    'रोजगार':   'job',
    'तैनाती':   'placement',
    'धंधा':     'job work',

    # Salary/Package
    'तनख्वाह':  'salary',
    'कमाई':     'salary earnings',
    'पगार':     'salary',
    'मेहनताना': 'salary',
    'आमदनी':    'earnings salary',

    # Pronouns
    'मी':       'main I',
    'मेरो':     'mera my',
    'खुद':      'khud self',
    'स्वयं':    'khud self',
    'तेरो':     'tera your',
    'थ्वारो':   'tumhara your',
    'तुमरो':    'tumhara your',

    # Verbs
    'देखा':     'dekha see',
    'निहारा':   'dekha look',
    'बताणु':    'batao tell',
    'दसणु':     'batao tell',
    'कहणु':     'kehna say',
    'सुणाणु':   'sunao tell',
    'जाणु':     'jana go',
    'जावा':     'jana go',
    'चलणु':     'chalna go',
    'पहुँचणु':  'pahunchna reach',
    'औणु':      'aana come',
    'आणु':      'aana come',

    # Yes/No
    'हाँ':      'haan yes',
    'ठीकई':    'yes okay',
    'नी':       'nahi no',
    'नाही':     'nahi no',
    'नैं':      'nahi no',

    # Conjunctions
    'अर':       'aur and',
    'औ':        'aur and',
    'बटे':      'se from',
    'सैं':      'se from',
    'सें':      'dwara by',

    # Prepositions
    'पै':       'par on',
    # Note: 'म','मा','कु','कू' are single chars — handle carefully
}


def ga_ku_to_hi_en(text: str) -> str:
    """Translate Garhwali/Kumauni words → Hindi/English for keyword matching."""
    t = text.lower()
    # Sort by phrase length descending — match longer phrases first
    for ga_word, translation in sorted(
        GARHWALI_SYNONYM_MAP.items(), key=lambda x: -len(x[0])
    ):
        if ga_word in t:
            t = t.replace(ga_word, ' ' + translation + ' ')
    return re.sub(r'\s+', ' ', t).strip()


# ══════════════════════════════════════════════════════════════════════
# TYPO MAP + fix_typos()
# ══════════════════════════════════════════════════════════════════════
TYPO_MAP = {
    # Number-letter confusion
    "h0d":          "hod",
    "f33s":         "fees",
    "adm1ssion":    "admission",
    # Common spelling mistakes
    "mechenical":   "mechanical",
    "mechnical":    "mechanical",
    "mechincal":    "mechanical",
    "mechanicle":   "mechanical",
    "mechinical":   "mechanical",
    "electical":    "electrical",
    "electrcal":    "electrical",
    "biotechonlogy":"biotechnology",
    "bitoech":      "biotechnology",
    "admision":     "admission",
    "admisson":     "admission",
    "palcement":    "placement",
    "hostle":       "hostel",
    "dircetor":     "director",
    "registar":     "registrar",
    "collage":      "college",
    "colege":       "college",
    "univeristy":   "university",
    "faculity":     "faculty",
    "placment":     "placement",
    "semster":      "semester",
    "semister":     "semester",
}


def fix_typos(text: str) -> str:
    """Fix common typos + number-letter substitution + Garhwali words."""
    words = text.lower().split()
    fixed = []
    for w in words:
        # Number → letter fix (l33t speak: 0→o, 1→i, 3→e)
        w_clean = w.replace('0', 'o').replace('1', 'i').replace('3', 'e')
        if w in TYPO_MAP:
            fixed.append(TYPO_MAP[w])
        elif w_clean in TYPO_MAP:
            fixed.append(TYPO_MAP[w_clean])
        else:
            fixed.append(w_clean if w_clean != w else w)

    result = " ".join(fixed)
    # Apply Garhwali/Kumauni translation
    result = ga_ku_to_hi_en(result)
    return result


# ══════════════════════════════════════════════════════════════════════
# SPECIFIC ROLE MAP
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
    "hod of applied":         "hod applied sciences",
    "hod applied":            "hod applied sciences",
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
    "priti dimri":            "hod mca",
    "prof priti dimri":       "hod mca",
}

# Blacklist — skip "List all HODs" type entries
ROLE_BLACKLIST = [
    "list all", "all hod", "all department", "all heads",
    "departments at gbpiet", "list of hod", "all hods",
    "सभी विभाग", "सभी hod",
]


def specific_role_answer(question: str) -> str | None:
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
    best_score, best_ans = 0, None

    for item in load_qa_database():
        q_lower = item["question"].lower()
        a_lower = item["answer"].lower()

        # Skip list-type entries
        if any(p in q_lower for p in ROLE_BLACKLIST):
            continue

        score = 0
        if all(w in q_lower for w in topic_words):
            score += 3
        elif mapped.lower() in q_lower:
            score += 2
        elif len(topic_words) >= 2 and sum(1 for w in topic_words if w in q_lower) >= 2:
            score += 1
        if any(w in a_lower for w in topic_words):
            score += 1

        # Minimum score 2 required
        if score >= 2 and score > best_score:
            best_score = score
            best_ans   = item["answer"]
            print(f"[ROLE] ✅ score={best_score} q={item['question'][:55]}")

    return best_ans


# ══════════════════════════════════════════════════════════════════════
# DIRECT KEYWORD MAP
# ══════════════════════════════════════════════════════════════════════
DIRECT_KEYWORD_MAP = {
    # English
    "registrar":    "registrar",
    "director":     "director",
    "dean":         "dean",
    "chairman":     "chairman",
    "warden":       "warden",
    "placement":    "placement",
    "placements":   "placement",
    "hostel":       "hostel",
    "hostels":      "hostel",
    "fees":         "fees",
    "fee":          "fees",
    "admission":    "admission",
    "admissions":   "admission",
    "contact":      "contact",
    "courses":      "courses",
    "course":       "courses",
    "library":      "library",
    "transport":    "transport",
    "scholarship":  "scholarship",
    "result":       "result",
    "ragging":      "ragging",
    "sports":       "sports",
    "faculty":      "faculty",
    "hod":          "head of department",
    "about":        "about gbpiet",
    "website":      "gbpiet website",
    # Typo variants
    "h0d":          "head of department",
    "hods":         "head of department",
    # Hindi
    "रजिस्ट्रार":   "registrar",
    "निदेशक":        "director",
    "डीन":           "dean",
    "प्लेसमेंट":    "placement",
    "हॉस्टल":       "hostel",
    "फीस":           "fees",
    "प्रवेश":        "admission",
    "संपर्क":        "contact",
    "पुस्तकालय":    "library",
    "परिवहन":        "transport",
    "रैगिंग":        "ragging",
    "संकाय":         "faculty",
    # Garhwali/Kumauni (after ga_ku_to_hi_en translation)
    "एडमिशन":       "admission",
    "भर्ती":         "admission",
    "नौकरी":         "placement",
    "सुविधा":        "facility",
}


def direct_keyword_answer(question: str) -> str | None:
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
    mapped_lower         = mapped.lower()
    best_score, best_ans = 0, None

    for item in load_qa_database():
        score = 0
        if mapped_lower in item["question"].lower(): score += 2
        if mapped_lower in item["answer"].lower():   score += 1
        if score > best_score:
            best_score = score
            best_ans   = item["answer"]

    return best_ans


# ══════════════════════════════════════════════════════════════════════
# EXACT MATCH
# ══════════════════════════════════════════════════════════════════════
def exact_match(question: str) -> str | None:
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
    'क्या','कौन','का','की','के','में','से','है','हैं',
    'और','या','को','ने','मैं','हम','आप','वे','इस',
    'उस','यह','वह','पर','कैसे','कहाँ','कहां',
}
HOSTEL_NAMES = {
    'kailash','neelkanth','kedar','rudra','badri','alaknanda',
    'shivalik','trishul','raman','bhagirathi','viswerwarya','vh',
}


def get_keywords(text: str) -> set:
    words         = set(re.findall(r'[\u0900-\u097F]+|[a-zA-Z0-9]+', text.lower()))
    translated    = set(re.findall(r'[a-zA-Z0-9]+', hi_to_en(text)))
    # ── NEW: Garhwali/Kumauni translation ─────────────────────────
    ga_translated = set(re.findall(r'[a-zA-Z0-9]+', ga_ku_to_hi_en(text)))
    return (words | translated | ga_translated) - STOP


def keyword_match(question: str, threshold: int = 2) -> str | None:
    q_fixed         = fix_typos(question)
    q_kw            = get_keywords(q_fixed.lower())
    specific_hostel = q_kw & HOSTEL_NAMES

    if not q_kw:
        return None

    best_score, best_ans = 0.0, None

    for item in load_qa_database():
        s_kw    = get_keywords(item["question"].lower())
        matches = len(q_kw & s_kw)
        score   = matches / max(len(q_kw), len(s_kw), 1)

        if specific_hostel:
            if not (specific_hostel & (s_kw & HOSTEL_NAMES)):
                continue

        if matches >= threshold and score > best_score:
            best_score = score
            best_ans   = item["answer"]
            print(f"[KW] {score:.2f} m={matches}: {item['question'][:45]}")

    return best_ans


# ══════════════════════════════════════════════════════════════════════
# RAG PIPELINE — BM25 + Vector + Rerank + Internet fallback
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
        lang_filter = lang if lang in ("en", "hi") else None

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

        ctx_parts = []
        for r in merged[:3]:
            url = r.get("url") or r.get("metadata", {}).get("source", "")
            if url and url.startswith("http"):
                sources.append(url)
            ctx_parts.append(f"[Score: {r['rrf_score']:.3f}]\n{r['text']}")

        top_score = merged[0]["rrf_score"] if merged else 0
        if top_score < 0.05 or not merged:
            print(f"[RAG] Low score ({top_score:.3f}) — trying internet search...")
            internet_results = search_college_website(question)
            if internet_results:
                used_internet = True
                print(f"[RAG] Internet: {len(internet_results)} results")
                for r in internet_results[:2]:
                    ctx_parts.append(f"[Web]\n{r['snippet']}\nSource: {r['url']}")
                    sources.append(r["url"])

        if not ctx_parts:
            return {"context": None, "sources": [], "used_internet": False}

        return {
            "context":      "\n\n---\n\n".join(ctx_parts),
            "sources":      list(dict.fromkeys(sources)),
            "used_internet": used_internet,
        }

    except Exception as e:
        print(f"[RAG] Error: {e}")
        return {"context": None, "sources": [], "used_internet": False}


def rag_search(question: str, lang: str = "en") -> str | None:
    """Run async RAG safely from sync context."""
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
# LLM PROMPT — college website focused
# ══════════════════════════════════════════════════════════════════════
def build_prompt(question: str, context: str, lang: str, history: str = "") -> str:

    website_note = f"\nCollege Website: {GBPIET_URL}"

    if lang == "hi":
        return f"""Aap दीक्षा (Diksha) hain — GBPIET ke liye helpful AI chatbot.

RULES:
- HAMESHA shuddh Hindi mein jawab dein
- Sirf context use karein
- Answer mein college website ka URL include karein jahan relevant ho: {GBPIET_URL}
- Agar context mein answer nahi: "माफ़ करें, यह जानकारी नहीं मिली। कृपया {GBPIET_URL} देखें।"

{history}
Context:
{context}
{website_note}

Sawaal: {question}
Jawab (Hindi mein):"""

    elif lang == "ga":
        return f"""Tu दीक्षा (Diksha) chhe — GBPIET chatbot. Garhwali mein jawab de.
Sirf context use kar. Website: {GBPIET_URL}
Nahi mila: "माफ करा, {GBPIET_URL} देखो।"
{history}
Context: {context}
Sawaal: {question}
Jawab:"""

    elif lang == "ku":
        return f"""Tu दीक्षा (Diksha) chhu — GBPIET chatbot. Kumauni mein jawab de.
Sirf context use kar. Website: {GBPIET_URL}
Nahi mila: "माफ करो, {GBPIET_URL} देखो।"
{history}
Context: {context}
Sawaal: {question}
Jawab:"""

    else:
        return f"""You are (Diksha) — official AI assistant for GBPIET
(Govind Ballabh Pant Institute of Engineering and Technology),
Pauri Garhwal, Uttarakhand. Website: {GBPIET_URL}

RULES:
- Answer in ENGLISH ONLY
- Use ONLY the context below — do NOT hallucinate
- Include relevant URLs from context in your answer
- If answer mentions specific pages, add: "More info: {GBPIET_URL}/relevant-page"
- If not found: "I'm sorry, I couldn't find that information. Please visit {GBPIET_URL} or call 01368-228030."
- Keep answers concise and helpful

{history}
Context:
{context}

Question: {question}
Answer:"""


def llm_answer(question: str, context: str, lang: str, history: str = "") -> str:
    prompt = build_prompt(question, context, lang, history)
    result = groq_call(
        messages=[
            {"role": "system", "content": f"You are Diksha, official AI assistant for GBPIET ({GBPIET_URL}). Be helpful, accurate and concise."},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=500, temperature=0.3,
    )
    if result:
        return result

    # If LLM fails — return best context chunk directly
    if context:
        lines = [l.strip() for l in context.split('\n') if len(l.strip()) > 30]
        if lines:
            return lines[0] + f"\n\nFor more info: {GBPIET_URL}"

    return f"I'm sorry, I couldn't generate a response. Please visit {GBPIET_URL} or call 01368-228030."


# ══════════════════════════════════════════════════════════════════════
# MAIN ENTRY — get_answer()
# ══════════════════════════════════════════════════════════════════════
def get_answer(question: str, lang: str = "en", history: str = "") -> str:
    question = question.strip()

    # ── Greeting ─────────────────────────────────────────────────
    if question.lower().strip() in GREETINGS:
        print("[RESULT] Greeting")
        return GREETING_RESPONSE.get(lang, GREETING_RESPONSE["en"])

    # ── Identity ─────────────────────────────────────────────────
    if question.lower().strip() in IDENTITY_Q:
        print("[RESULT] Identity")
        return IDENTITY_RESPONSE.get(lang, IDENTITY_RESPONSE["en"])

    print(f"\n{'='*55}\n[Q/{lang}] {question}\n{'='*55}")

    # ── Step 0a: Specific role ────────────────────────────────────
    ans = specific_role_answer(question)
    if ans:
        print("[RESULT] Specific role match")
        return translate_answer_if_needed(ans, lang, question)

    # ── Step 0b: Direct keyword ───────────────────────────────────
    ans = direct_keyword_answer(question)
    if ans:
        print("[RESULT] Direct keyword")
        return translate_answer_if_needed(ans, lang, question)

    # ── Step 1: Exact match ───────────────────────────────────────
    ans = exact_match(question)
    if ans:
        print("[RESULT] Exact match")
        return translate_answer_if_needed(ans, lang, question)

    # ── Step 2: Keyword match ─────────────────────────────────────
    word_count = len(question.split())
    thresh     = 1 if word_count <= 2 else (2 if word_count <= 5 else 3)
    ans        = keyword_match(question, thresh)
    if ans:
        print("[RESULT] Keyword match")
        return translate_answer_if_needed(ans, lang, question)

    # ── Step 3: RAG + LLM ────────────────────────────────────────
    ctx = rag_search(question, lang)
    if ctx:
        print("[RESULT] RAG + LLM")
        return llm_answer(question, ctx, lang, history)

    # ── No match ─────────────────────────────────────────────────
    print("[RESULT] No match")
    fb = {
        "hi": f"माफ़ करें, यह जानकारी नहीं मिली। कृपया {GBPIET_URL} देखें या 01368-228030 पर कॉल करें।",
        "ga": f"माफ करा, जानकारी नि मिलि। {GBPIET_URL} देखो।",
        "ku": f"माफ करिया! जानकारी नैं च। {GBPIET_URL} देखो।",
        "en": f"I'm sorry, I couldn't find that information. Please visit {GBPIET_URL} or call 01368-228030.",
    }
    return fb.get(lang, fb["en"])
