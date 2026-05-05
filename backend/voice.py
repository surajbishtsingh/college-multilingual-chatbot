from gtts import gTTS
import io

def generate_voice(text: str, lang: str = "en") -> bytes:
    try:
        tts_lang = "hi" if lang in ("hi", "ga", "ku") else "en"
        tts = gTTS(text=text, lang=tts_lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        return buf.getvalue()
    except Exception as e:
        print(f"[Voice] gTTS error: {e}")
        return b""
