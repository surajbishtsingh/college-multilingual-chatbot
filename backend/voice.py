# voice.py
import io
import asyncio
import tempfile
import os

# ── Voice config ──────────────────────────────────────────────────────
# Primary: edge-tts Indian female voices (most natural)
EDGE_VOICES = {
    "en":  "en-IN-NeerjaNeural",      # Indian English female (natural)
    "hi":  "hi-IN-SwaraNeural",       # Hindi female (natural)
    "ga":  "hi-IN-SwaraNeural",       # Garhwali → Hindi voice
    "ku":  "hi-IN-SwaraNeural",       # Kumauni → Hindi voice
}

# Fallback: gTTS
GTTS_LANGS = {
    "en":  "en",
    "hi":  "hi",
    "ga":  "hi",
    "ku":  "hi",
}


async def _edge_tts_generate(text: str, lang: str) -> bytes:
    """Try edge-tts with Indian female voice."""
    import edge_tts

    voice = EDGE_VOICES.get(lang, EDGE_VOICES["en"])

    # Write to a temp file then read back as bytes
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(tmp_path)

        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        if len(audio_bytes) < 100:
            raise ValueError("Audio too small — likely empty response")

        print(f"[Voice] ✅ edge-tts ({voice}) — {len(audio_bytes)} bytes")
        return audio_bytes

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _gtts_generate(text: str, lang: str) -> bytes:
    """Fallback: gTTS Indian accent."""
    from gtts import gTTS

    tts_lang = GTTS_LANGS.get(lang, "en")

    # gTTS supports tld="co.in" for Indian accent on English
    if tts_lang == "en":
        tts = gTTS(text=text, lang="en", tld="co.in", slow=False)
    else:
        tts = gTTS(text=text, lang=tts_lang, slow=False)

    buf = io.BytesIO()
    tts.write_to_fp(buf)
    audio_bytes = buf.getvalue()

    print(f"[Voice] ✅ gTTS fallback (lang={tts_lang}, Indian accent) — {len(audio_bytes)} bytes")
    return audio_bytes


def generate_voice(text: str, lang: str = "en") -> bytes:
    """
    Generate Indian female voice audio.
    Tries edge-tts (NeerjaNeural/SwaraNeural) first,
    falls back to gTTS with Indian accent (tld=co.in).
    """
    if not text or not text.strip():
        return b""

    # Truncate very long text to avoid timeout
    if len(text) > 500:
        text = text[:500] + "..."

    # ── Try edge-tts first ────────────────────────────────
    try:
        loop = asyncio.new_event_loop()
        try:
            audio = loop.run_until_complete(_edge_tts_generate(text, lang))
            return audio
        finally:
            loop.close()

    except Exception as e:
        print(f"[Voice] edge-tts failed ({e}), trying gTTS fallback...")

    # ── Fallback to gTTS ──────────────────────────────────
    try:
        return _gtts_generate(text, lang)
    except Exception as e:
        print(f"[Voice] gTTS also failed: {e}")
        return b""
