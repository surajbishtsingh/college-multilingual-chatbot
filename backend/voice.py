import io
import os
import re
import base64
import requests

from gtts import gTTS

# ══════════════════════════════════════════════════════════════════════
# SARVAM AI CONFIG  ←  Primary TTS (best Indian female voice)
# ══════════════════════════════════════════════════════════════════════
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_URL     = "https://api.sarvam.ai/text-to-speech"

# Sarvam supported language codes
SARVAM_LANG_MAP = {
    "en": "en-IN",
    "hi": "hi-IN",
    "ga": "hi-IN",   # Garhwali → Hindi voice (closest available)
    "ku": "hi-IN",   # Kumauni  → Hindi voice (closest available)
}

# Sarvam Indian female speakers
# Options: meera, pavithra, maitreyi, arvind, amol, amartya
SARVAM_SPEAKER = "neha"    # Pleasant and soothing Indian female voice


# ══════════════════════════════════════════════════════════════════════
# GTTS FALLBACK CONFIG
# ══════════════════════════════════════════════════════════════════════
GTTS_LANGS = {
    "en": "en",
    "hi": "hi",
    "ga": "hi",
    "ku": "hi",
}


# ══════════════════════════════════════════════════════════════════════
# CLEAN TEXT FOR TTS
# ══════════════════════════════════════════════════════════════════════
def clean_tts_text(text: str, lang: str = "en") -> str:
    """Clean and expand text before sending to TTS."""

    if not text:
        return ""

    # Remove emojis
    text = re.sub(
        r'[\U0001F300-\U0001FAFF\u2600-\u27BF]+',
        ' ', text
    )

    # Remove markdown symbols
    text = re.sub(r'[*_`#~>]', '', text)

    # Common English abbreviations
    text = re.sub(r'\bDr\.', 'Doctor', text, flags=re.IGNORECASE)
    text = re.sub(r'\bDr\b',  'Doctor', text, flags=re.IGNORECASE)
    text = re.sub(r'\bProf\.','Professor', text, flags=re.IGNORECASE)
    text = re.sub(r'\bProf\b','Professor', text, flags=re.IGNORECASE)
    text = re.sub(r'\bHOD\b', 'H O D',    text, flags=re.IGNORECASE)
    text = re.sub(r'\bMCA\b', 'M C A',    text, flags=re.IGNORECASE)
    text = re.sub(r'\bCSE\b', 'C S E',    text, flags=re.IGNORECASE)
    text = re.sub(r'\bECE\b', 'E C E',    text, flags=re.IGNORECASE)

    # GBPIET pronunciation
    if lang in ("hi", "ga", "ku"):
        text = re.sub(
            r'\bGBPIET\b',
            'जी बी पी आई ई टी',
            text, flags=re.IGNORECASE
        )
        text = re.sub(r'जीबीपीआईईटी', 'जी बी पी आई ई टी', text)
    else:
        text = re.sub(
            r'\bGBPIET\b',
            'G B P I E T',
            text, flags=re.IGNORECASE
        )

    # Hindi title fixes
    if lang in ("hi", "ga", "ku"):
        text = text.replace("डॉ.", "डॉक्टर")
        text = text.replace("प्रो.", "प्रोफेसर")

    # Remove URLs (TTS should not read them)
    text = re.sub(r'https?://\S+', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ══════════════════════════════════════════════════════════════════════
# SARVAM AI TTS  ←  PRIMARY
# ══════════════════════════════════════════════════════════════════════
def sarvam_tts(text: str, lang: str = "en") -> bytes:
    """
    Sarvam AI TTS — best Indian female voice.
    Returns audio bytes (WAV format) or empty bytes on failure.
    """
    if not SARVAM_API_KEY:
        print("[Voice] ⚠️  SARVAM_API_KEY not set — skipping Sarvam")
        return b""

    target_lang = SARVAM_LANG_MAP.get(lang, "en-IN")

    try:
        response = requests.post(
            SARVAM_URL,
            headers={
                "api-subscription-key": SARVAM_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "inputs":               [text[:500]],
                "target_language_code": target_lang,
                "speaker":              SARVAM_SPEAKER,
                "pace":                 0.9,
                "speech_sample_rate":   22050,
                "enable_preprocessing": True,
                "model":                "bulbul:v3",
            },
            timeout=15,
        )

        if response.status_code == 200:
            data       = response.json()
            audio_b64  = data["audios"][0]
            audio_bytes = base64.b64decode(audio_b64)

            if len(audio_bytes) < 100:
                print("[Voice] Sarvam returned empty audio")
                return b""

            print(
                f"[Voice] ✅ Sarvam AI "
                f"(speaker={SARVAM_SPEAKER}, lang={target_lang}) "
                f"— {len(audio_bytes)} bytes"
            )
            return audio_bytes

        else:
            print(
                f"[Voice] Sarvam HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
            return b""

    except requests.exceptions.Timeout:
        print("[Voice] Sarvam timeout — falling back to gTTS")
        return b""
    except Exception as e:
        print(f"[Voice] Sarvam error: {e}")
        return b""


# ══════════════════════════════════════════════════════════════════════
# GTTS  ←  FALLBACK
# ══════════════════════════════════════════════════════════════════════
def gtts_generate(text: str, lang: str = "en") -> bytes:
    """gTTS fallback with Indian accent."""
    tts_lang = GTTS_LANGS.get(lang, "en")

    if tts_lang == "en":
        tts = gTTS(text=text, lang="en", tld="co.in", slow=False)
    else:
        tts = gTTS(text=text, lang=tts_lang, slow=False)

    buf = io.BytesIO()
    tts.write_to_fp(buf)
    audio_bytes = buf.getvalue()

    print(
        f"[Voice] ✅ gTTS fallback "
        f"(lang={tts_lang}) "
        f"— {len(audio_bytes)} bytes"
    )
    return audio_bytes


# ══════════════════════════════════════════════════════════════════════
# MAIN ENTRY
# ══════════════════════════════════════════════════════════════════════
def generate_voice(text: str, lang: str = "en") -> bytes:
    """
    Generate Indian female voice audio.

    Priority:
      1. Sarvam AI  — natural Indian female voice (meera)
      2. gTTS       — Indian accent fallback

    Edge-TTS is NOT used (blocked on Railway/cloud IPs).
    """
    if not text or not text.strip():
        return b""

    # Clean text before TTS
    text = clean_tts_text(text, lang)

    # Truncate very long text
    if len(text) > 500:
        text = text[:500] + "..."

    # ── 1. Try Sarvam AI ──────────────────────────────────────────────
    if SARVAM_API_KEY:
        audio = sarvam_tts(text, lang)
        if audio:
            return audio
        print("[Voice] Sarvam failed — trying gTTS fallback...")

    # ── 2. Fallback to gTTS ───────────────────────────────────────────
    try:
        return gtts_generate(text, lang)
    except Exception as e:
        print(f"[Voice] gTTS also failed: {e}")
        return b""
