# language_detector.py
import re

def detect_language(text: str) -> str:
    """
    Detects language from text.
    Priority: Devanagari script detection first,
    then English fallback for short/ambiguous inputs.
    """
    if not text or not text.strip():
        return "en"

    text = text.strip()

    # Count Devanagari characters
    devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    total_chars      = len(text.replace(" ", ""))

    if total_chars == 0:
        return "en"

    ratio = devanagari_chars / total_chars

    # ✅ Only classify as Hindi if strongly Devanagari
    # Short English phrases like "hod of ece" should stay English
    if ratio > 0.4:
        # Check for Garhwali/Kumauni specific words
        ga_words = ['छां', 'लग्यां', 'द्वीसर', 'मीथे', 'तैं', 'त्वे']
        ku_words = ['करौ', 'सकूँ', 'करिया', 'म्यर', 'तस']
        text_lower = text.lower()
        if any(w in text_lower for w in ga_words):
            return "ga"
        if any(w in text_lower for w in ku_words):
            return "ku"
        return "hi"

    # ✅ Default to English for Latin-script text
    return "en"
