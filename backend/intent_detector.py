# intent_detector.py — Fixed hostel + how many routing
import re

INTENT_RULES = {
    "admissions": {
        "en": [
            "admission", "apply", "application", "jee", "gate", "utuee",
            "eligibility", "counselling", "counseling", "merit", "rank",
            "cutoff", "cut off", "seat", "intake", "enroll", "registration",
            "document", "certificate", "noc", "btech", "b.tech", "mtech",
            "m.tech", "mca", "phd", "undergraduate", "postgraduate"
        ],
        "hi": [
            "प्रवेश", "दाखिला", "आवेदन", "पात्रता", "काउंसलिंग",
            "सीट", "रजिस्ट्रेशन", "दस्तावेज़", "नामांकन"
        ],
    },
    "fees": {
        "en": [
            "fee", "fees", "tuition", "payment", "scholarship", "stipend",
            "refund", "sbi", "collect", "challan", "receipt", "installment",
            "semester fee", "annual fee", "hostel fee", "mess fee"
        ],
        "hi": [
            "फीस", "शुल्क", "भुगतान", "छात्रवृत्ति", "वजीफा",
            "रिफंड", "किस्त", "मेस शुल्क"
        ],
    },
    "hostel": {
        "en": [
            "hostel", "accommodation", "room", "warden", "mess", "canteen",
            "dormitory", "boys hostel", "girls hostel", "residential",
            "boarding", "bed", "allotment",
            # ✅ Added — catches "how many hostels"
            "how many hostel", "number of hostel", "hostel available",
            "hostel list", "hostel name", "all hostel", "hostel count",
            "hostel facility", "hostel detail", "hostel information",
        ],
        "hi": [
            "हॉस्टल", "छात्रावास", "वार्डन", "कमरा", "मेस",
            "आवास", "रहना", "बिस्तर",
            "कितने हॉस्टल", "हॉस्टल की सूची", "सभी हॉस्टल",
        ],
    },
    "faculty": {
        "en": [
            "faculty", "professor", "teacher", "lecturer", "hod",
            "head of department", "staff", "department head", "dr.", "prof."
        ],
        "hi": [
            "शिक्षक", "प्राध्यापक", "संकाय", "विभागाध्यक्ष",
            "प्रोफेसर", "डॉक्टर"
        ],
    },
    "placement": {
        "en": [
            "placement", "job", "recruit", "recruiter", "package", "lpa",
            "salary", "company", "campus", "drive", "offer", "hired",
            "amazon", "microsoft", "tcs", "infosys", "wipro"
        ],
        "hi": [
            "प्लेसमेंट", "नौकरी", "वेतन", "कंपनी", "भर्ती", "पैकेज"
        ],
    },
    "general": {
        "en": [
            "gbpiet", "college", "institute", "university", "campus",
            "history", "about", "location", "address", "contact", "phone",
            "email", "website", "reach", "route", "direction", "distance",
            "ranking", "naac", "nba", "accreditation", "autonomous"
        ],
        "hi": [
            "संस्थान", "कॉलेज", "पता", "संपर्क", "फोन",
            "वेबसाइट", "इतिहास", "रैंकिंग"
        ],
    },
}

# intent → Qdrant collection
INTENT_TO_COLLECTION = {
    "admissions": "admissions",
    "fees":       "fees",
    "hostel":     "hostel",
    "faculty":    "faq",
    "placement":  "faq",
    "general":    "faq",
}

# language → fallback collection
LANG_TO_COLLECTION = {
    "en": "kb_en",
    "hi": "kb_hi",
    "ga": "kb_hi",
    "ku": "kb_hi",
}


def detect_intent(question: str, lang: str = "en") -> str:
    q = question.lower().strip()

    # ✅ Check longer phrases first to avoid partial mismatches
    for intent, keyword_sets in INTENT_RULES.items():
        lang_keywords = keyword_sets.get(lang, [])
        # Sort by length descending so longer phrases match first
        for kw in sorted(lang_keywords, key=len, reverse=True):
            if kw in q:
                return intent

        en_keywords = keyword_sets.get("en", [])
        for kw in sorted(en_keywords, key=len, reverse=True):
            if kw in q:
                return intent

    return "general"


def get_collection_for_query(question: str, lang: str = "en") -> list[str]:
    intent   = detect_intent(question, lang)
    primary  = INTENT_TO_COLLECTION.get(intent, "faq")
    fallback = LANG_TO_COLLECTION.get(lang, "kb_en")

    collections = [primary]
    if fallback != primary:
        collections.append(fallback)

    print(f"[Intent] '{intent}' → collections: {collections}")
    return collections