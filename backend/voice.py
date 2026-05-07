import io
import asyncio
import tempfile
import os
import re

# ── Voice config ──────────────────────────────────────────────────────
EDGE_VOICES = {
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "ga": "hi-IN-SwaraNeural",
    "ku": "hi-IN-SwaraNeural",
}

GTTS_LANGS = {
    "en": "en",
    "hi": "hi",
    "ga": "hi",
    "ku": "hi",
}


# ═════════════════════════════════════════════════════════════
# CLEAN TEXT FOR TTS
# ═════════════════════════════════════════════════════════════
def clean_tts_text(text: str, lang: str = "en") -> str:
    """Clean and expand text before TTS."""

    if not text:
        return ""

    # ── Remove emojis ───────────────────────────────────────
    text = re.sub(
        r'[\U0001F300-\U0001FAFF\u2600-\u27BF]+',
        ' ',
        text
    )

    # ── Common English abbreviations ───────────────────────
    text = re.sub(r'\bDr\.', 'Doctor', text, flags=re.IGNORECASE)
    text = re.sub(r'\bDr\b', 'Doctor', text, flags=re.IGNORECASE)

    text = re.sub(r'\bProf\.', 'Professor', text, flags=re.IGNORECASE)
    text = re.sub(r'\bProf\b', 'Professor', text, flags=re.IGNORECASE)

    text = re.sub(r'\bHOD\b', 'H O D', text, flags=re.IGNORECASE)

    # ── GBPIET pronunciation ───────────────────────────────
    if lang in ["hi", "ga", "ku"]:
        text = re.sub(
            r'\bGBPIET\b',
            'जी बी पी आई ई टी',
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r'जीबीपीआईईटी',
            'जी बी पी आई ई टी',
            text
        )
    else:
        text = re.sub(
            r'\bGBPIET\b',
            'G B P I E T',
            text,
            flags=re.IGNORECASE
        )

    # ── Hindi / Garhwali title fixes ───────────────────────
    if lang in ["hi", "ga", "ku"]:
        text = text.replace("डॉ.", "डॉक्टर")
        text = text.replace("प्रो.", "प्रोफेसर")

    # ── Improve pronunciation spacing ──────────────────────
    if lang == "hi":
        text = text.replace("है", "है ")
        text = text.replace("हैं", "हैं ")

    if lang == "ga":
        text = text.replace("छ", "छ ")
        text = text.replace("च", "च ")
        text = text.replace("एआई", "A I")
        text = text.replace("AI", "A I")

    # ── Remove extra spaces ────────────────────────────────
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ═════════════════════════════════════════════════════════════
# EDGE TTS
# ═════════════════════════════════════════════════════════════
async def _edge_tts_generate(text: str, lang: str) -> bytes:
    """Try edge-tts with Indian female voice."""

    import edge_tts

    voice = EDGE_VOICES.get(lang, EDGE_VOICES["en"])

    with tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False
    ) as tmp:
        tmp_path = tmp.name

    try:
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice
        )

        await communicate.save(tmp_path)

        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        if len(audio_bytes) < 100:
            raise ValueError(
                "Audio too small — likely empty response"
            )

        print(
            f"[Voice] ✅ edge-tts ({voice}) "
            f"— {len(audio_bytes)} bytes"
        )

        return audio_bytes

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ═════════════════════════════════════════════════════════════
# GTTS FALLBACK
# ═════════════════════════════════════════════════════════════
def _gtts_generate(text: str, lang: str) -> bytes:
    """Fallback: gTTS Indian accent."""

    from gtts import gTTS

    tts_lang = GTTS_LANGS.get(lang, "en")

    if tts_lang == "en":
        tts = gTTS(
            text=text,
            lang="en",
            tld="co.in",
            slow=False
        )
    else:
        tts = gTTS(
            text=text,
            lang=tts_lang,
            slow=False
        )

    buf = io.BytesIO()

    tts.write_to_fp(buf)

    audio_bytes = buf.getvalue()

    print(
        f"[Voice] ✅ gTTS fallback "
        f"(lang={tts_lang}) "
        f"— {len(audio_bytes)} bytes"
    )

    return audio_bytes


# ═════════════════════════════════════════════════════════════
# MAIN VOICE GENERATOR
# ═════════════════════════════════════════════════════════════
def generate_voice(text: str, lang: str = "en") -> bytes:
    """
    Generate Indian female voice audio.
    Tries edge-tts first, falls back to gTTS.
    """

    if not text or not text.strip():
        return b""

    # ── Clean text before TTS ──────────────────────────────
    text = clean_tts_text(text, lang)

    # ── Truncate very long text ────────────────────────────
    if len(text) > 500:
        text = text[:500] + "..."

    # ── Try edge-tts first ─────────────────────────────────
    try:
        loop = asyncio.new_event_loop()

        try:
            audio = loop.run_until_complete(
                _edge_tts_generate(text, lang)
            )
            return audio

        finally:
            loop.close()

    except Exception as e:
        print(
            f"[Voice] edge-tts failed ({e}), "
            f"trying gTTS fallback..."
        )

    # ── Fallback to gTTS ───────────────────────────────────
    try:
        return _gtts_generate(text, lang)

    except Exception as e:
        print(f"[Voice] gTTS also failed: {e}")
        return b""
