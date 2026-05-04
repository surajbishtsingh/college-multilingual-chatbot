# voice.py — Railway-safe voice generation
# edge-tts first → gTTS fallback (Railway blocks WebSocket)

import base64
import asyncio
import io
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

# ── Voice settings per language ────────────────────────────────────────
# Female voices — natural human feel
EDGE_VOICES = {
    "en": "en-IN-NeerjaNeural",      # Indian English female — natural
    "hi": "hi-IN-SwaraNeural",       # Hindi female — clear
    "ga": "hi-IN-SwaraNeural",       # Garhwali → Hindi voice
    "ku": "hi-IN-SwaraNeural",       # Kumauni → Hindi voice
}

GTTS_LANGS = {
    "en": "en",
    "hi": "hi",
    "ga": "hi",   # fallback to Hindi
    "ku": "hi",   # fallback to Hindi
}

# Edge TTS settings — human-like pacing
EDGE_RATE   = "-5%"    # slightly slower — more natural
EDGE_PITCH  = "+2Hz"   # slight pitch up — female warmth
EDGE_VOLUME = "+10%"   # slightly louder


# ══════════════════════════════════════════════════════════════════════
# EDGE TTS — Primary (best quality, human-like)
# WHY IT FAILS ON RAILWAY: edge-tts uses WebSocket (wss://)
# Railway free tier sometimes blocks outbound WebSocket connections
# ══════════════════════════════════════════════════════════════════════
async def _edge_tts_generate(text: str, lang: str) -> bytes | None:
    """Generate audio using Microsoft Edge TTS (best quality)."""
    try:
        import edge_tts

        voice      = EDGE_VOICES.get(lang, EDGE_VOICES["en"])
        communicate = edge_tts.Communicate(
            text,
            voice,
            rate=EDGE_RATE,
            pitch=EDGE_PITCH,
            volume=EDGE_VOLUME,
        )

        audio_buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buf.write(chunk["data"])

        audio_bytes = audio_buf.getvalue()
        if len(audio_bytes) > 100:
            print(f"[Voice] edge-tts OK — {len(audio_bytes)} bytes — voice: {voice}")
            return audio_bytes
        return None

    except Exception as e:
        err = str(e).lower()
        if "websocket" in err or "403" in err or "connection" in err:
            print(f"[Voice] edge-tts blocked (Railway WebSocket): {e}")
        else:
            print(f"[Voice] edge-tts failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# gTTS FALLBACK — Works on Railway (uses HTTPS not WebSocket)
# WHY FALLBACK: Lower quality than edge-tts but Railway-safe
# ══════════════════════════════════════════════════════════════════════
def _gtts_generate(text: str, lang: str) -> bytes | None:
    """Generate audio using Google TTS (Railway-safe fallback)."""
    try:
        from gtts import gTTS

        gtts_lang = GTTS_LANGS.get(lang, "en")

        # Limit text length for gTTS (avoids timeout)
        text_trimmed = text[:500] if len(text) > 500 else text

        tts = gTTS(text=text_trimmed, lang=gtts_lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_bytes = buf.read()

        if len(audio_bytes) > 100:
            print(f"[Voice] gTTS OK — {len(audio_bytes)} bytes — lang: {gtts_lang}")
            return audio_bytes
        return None

    except Exception as e:
        print(f"[Voice] gTTS failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════
# MAIN — generate_voice() with automatic fallback
# ══════════════════════════════════════════════════════════════════════
def generate_voice(text: str, lang: str = "en") -> str | None:
    """
    Generate base64 audio with automatic fallback:
    1. edge-tts (best quality, human-like female voice)
    2. gTTS (Railway-safe fallback)
    3. None (silent — frontend handles gracefully)

    Returns base64 encoded MP3 string or None.
    """
    if not text or not text.strip():
        return None

    # Clean text for TTS — remove markdown symbols
    clean_text = text.replace("*", "").replace("#", "").replace("`", "")
    clean_text = clean_text.replace("→", " ").replace("•", " ").replace("—", " ")
    clean_text = clean_text.strip()

    # Limit to 1000 chars for performance
    if len(clean_text) > 1000:
        clean_text = clean_text[:1000] + "..."

    audio_bytes = None

    # ── Try edge-tts first ────────────────────────────────────────
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_bytes = loop.run_until_complete(_edge_tts_generate(clean_text, lang))
        loop.close()
    except Exception as e:
        print(f"[Voice] edge-tts loop error: {e}")

    # ── Fallback to gTTS ─────────────────────────────────────────
    if not audio_bytes:
        print("[Voice] Trying gTTS fallback...")
        audio_bytes = _gtts_generate(clean_text, lang)

    # ── Return base64 ─────────────────────────────────────────────
    if audio_bytes:
        return base64.b64encode(audio_bytes).decode("utf-8")

    print("[Voice] All TTS methods failed — returning None")
    return None
