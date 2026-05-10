# rag/groq_manager.py — Production-grade Groq manager
# Features:
#   ✅ Round-robin key rotation
#   ✅ Automatic retry with next key on failure
#   ✅ Timeout protection
#   ✅ In-memory response cache
#   ✅ Fallback models
#   ✅ Graceful failure messages
#   ✅ Token usage tracking

import os
import time
import hashlib
import threading
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
PRIMARY_MODEL  = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama3-70b-8192"
LIGHT_MODEL    = "llama3-8b-8192"        # fastest, used if others fail
TIMEOUT        = 30                       # seconds per request
CACHE_TTL      = 3600                     # cache responses for 1 hour (seconds)
CACHE_MAX_SIZE = 500                      # max cached responses

# ══════════════════════════════════════════════════════════════════════
# LOAD GROQ KEYS
# ══════════════════════════════════════════════════════════════════════
_raw_keys = [
    os.getenv("GROQ_API_KEY",   ""),
    os.getenv("GROQ_API_KEY_2", ""),
    os.getenv("GROQ_API_KEY_3", ""),
    os.getenv("GROQ_API_KEY_4", ""),
]
GROQ_KEYS    = [k.strip() for k in _raw_keys if k.strip()]
GROQ_CLIENTS = [Groq(api_key=k) for k in GROQ_KEYS]

print(f"[GroqManager] {len(GROQ_CLIENTS)} key(s) loaded")

# ══════════════════════════════════════════════════════════════════════
# ROUND-ROBIN STATE (thread-safe)
# ══════════════════════════════════════════════════════════════════════
_lock        = threading.Lock()
_current_key = 0
_stats = {
    "total_requests":  0,
    "cache_hits":      0,
    "groq_calls":      0,
    "failed_requests": 0,
    "key_usage":       [0] * max(len(GROQ_CLIENTS), 1),
}


def _next_client():
    """Get next Groq client in round-robin order."""
    global _current_key
    with _lock:
        if not GROQ_CLIENTS:
            return None, -1
        idx          = _current_key
        client       = GROQ_CLIENTS[idx]
        _current_key = (idx + 1) % len(GROQ_CLIENTS)
        return client, idx


# ══════════════════════════════════════════════════════════════════════
# IN-MEMORY CACHE
# ══════════════════════════════════════════════════════════════════════
_cache: dict = {}


def _cache_key(messages: list, model: str) -> str:
    """Generate cache key from messages + model."""
    content = str(messages) + model
    return hashlib.md5(content.encode()).hexdigest()


def _cache_get(key: str):
    entry = _cache.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > CACHE_TTL:
        del _cache[key]
        return None
    return entry["response"]


def _cache_set(key: str, response: str):
    if len(_cache) >= CACHE_MAX_SIZE:
        oldest = min(_cache.items(), key=lambda x: x[1]["ts"])
        del _cache[oldest[0]]
    _cache[key] = {"response": response, "ts": time.time()}


def _is_cacheable(messages: list) -> bool:
    """
    Only cache factual/FAQ queries — not conversational ones.
    """
    if not messages:
        return False
    user_msg = ""
    for m in messages:
        if m.get("role") == "user":
            user_msg = m.get("content", "").lower()
            break
    skip_words = [
        "you", "your", "my ", "i ", "me ", "we ",
        "what do you", "tell me about yourself",
    ]
    if any(w in user_msg for w in skip_words):
        return False
    return True


# ══════════════════════════════════════════════════════════════════════
# MAIN GROQ CALL — with rotation, retry, cache, timeout
# ══════════════════════════════════════════════════════════════════════
def groq_call(
    messages:    list,
    max_tokens:  int   = 500,
    temperature: float = 0.3,
    use_cache:   bool  = True,
) -> str:
    """
    Production Groq call with:
    - Round-robin key rotation
    - Auto retry on failure
    - Response caching
    - Timeout protection
    - Model fallback
    """
    _stats["total_requests"] += 1

    # ── Cache check ───────────────────────────────────────────────────
    cache_key = _cache_key(messages, PRIMARY_MODEL)
    if use_cache and _is_cacheable(messages):
        cached = _cache_get(cache_key)
        if cached:
            _stats["cache_hits"] += 1
            print(f"[GroqManager] ✅ Cache hit")
            return cached

    # ── Try all keys × all models ─────────────────────────────────────
    models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL, LIGHT_MODEL]
    tried_clients = set()

    for model in models_to_try:
        for _ in range(len(GROQ_CLIENTS)):
            client, idx = _next_client()
            if client is None:
                break

            client_id = id(client)
            if client_id in tried_clients and model == PRIMARY_MODEL:
                continue
            tried_clients.add(client_id)

            try:
                _stats["groq_calls"]      += 1
                _stats["key_usage"][idx]  += 1

                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=TIMEOUT,
                )
                result = response.choices[0].message.content.strip()

                print(f"[GroqManager] ✅ Key{idx+1}/{model} answered")

                # Cache successful response
                if use_cache and _is_cacheable(messages):
                    _cache_set(cache_key, result)

                return result

            except Exception as e:
                err = str(e).lower()
                print(f"[GroqManager] Key{idx+1}/{model} failed: {str(e)[:80]}")

                if "rate limit" in err or "429" in err:
                    continue
                if "timeout" in err or "timed out" in err:
                    continue
                if "401" in err or "invalid api key" in err:
                    print(f"[GroqManager] ⚠️ Key{idx+1} auth failed — skipping")
                    continue
                if "500" in err or "503" in err:
                    continue
                continue

    # ── All failed ────────────────────────────────────────────────────
    _stats["failed_requests"] += 1
    print("[GroqManager] ❌ All keys and models failed")
    return ""


# ══════════════════════════════════════════════════════════════════════
# TRANSLATION CALL — lightweight, uses smaller model
# ══════════════════════════════════════════════════════════════════════
def groq_translate(text: str, target_lang: str) -> str:
    """Lightweight translation using smaller/faster model."""
    if target_lang == "en":
        prompt = f"Translate to English. Return ONLY the translation, nothing else.\n\n{text}"
        system = "Translator. Hindi to English. Keep names, numbers, URLs unchanged."
    else:
        prompt = f"Translate to Hindi (Devanagari). Return ONLY the translation.\n\n{text}"
        system = "Translator. English to Hindi. Keep names, numbers, URLs unchanged."

    client, idx = _next_client()
    if not client:
        return text

    try:
        r = client.chat.completions.create(
            model=LIGHT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=400,
            temperature=0.1,
            timeout=20,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GroqManager] Translation failed: {e}")
        return text


# ══════════════════════════════════════════════════════════════════════
# STATS — for monitoring
# ══════════════════════════════════════════════════════════════════════
def get_stats() -> dict:
    total = _stats["total_requests"]
    cache = _stats["cache_hits"]
    return {
        "total_requests":   total,
        "cache_hits":       cache,
        "cache_hit_rate":   f"{(cache/total*100):.1f}%" if total > 0 else "0%",
        "groq_api_calls":   _stats["groq_calls"],
        "failed_requests":  _stats["failed_requests"],
        "cached_responses": len(_cache),
        "key_usage": {
            f"key_{i+1}": count
            for i, count in enumerate(_stats["key_usage"])
        },
        "active_keys": len(GROQ_CLIENTS),
    }


def clear_cache():
    """Clear response cache — useful after KB updates."""
    _cache.clear()
    print("[GroqManager] Cache cleared")
