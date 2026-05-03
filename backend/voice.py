import edge_tts
import asyncio
import base64
import io

VOICES = {
    'en': 'en-IN-NeerjaNeural',
    'hi': 'hi-IN-SwaraNeural',
    'ga': 'hi-IN-SwaraNeural',
    'ku': 'hi-IN-SwaraNeural',
}

# Natural voice ke liye SSML use karo
async def _tts_async(text: str, lang: str) -> str:
    try:
        voice = VOICES.get(lang, 'en-IN-NeerjaNeural')

        # SSML se voice natural banao
        communicate = edge_tts.Communicate(
            text,
            voice,
            rate="-5%",     # Thoda slow — natural lagta hai
            volume="+10%",  # Clear volume
            pitch="+5Hz"    # Slight higher — female voice
        )

        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        buf.seek(0)
        data = buf.read()
        if not data:
            return ""
        return base64.b64encode(data).decode('utf-8')
    except Exception as e:
        print(f"Edge TTS error: {e}")
        return ""

def generate_voice(text: str, lang: str) -> str:
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_tts_async(text, lang))
        loop.close()
        return result
    except Exception as e:
        print(f"Voice error: {e}")
        return ""