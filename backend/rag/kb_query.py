# ── Imports ───────────────────────────────────────────────────────────
import os
import json
import glob
import re
import asyncio
import unicodedata

from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Internal Imports ──────────────────────────────────────────────────
from qdrant_setup import get_client, COLLECTIONS
from intent_detector import get_collection_for_query
from rag.hybrid_search import multi_collection_search
from rag.bm25_search import bm25_search
from rag.fusion import reciprocal_rank_fusion
from rag.internet_search import search_college_website
from rag.reranker import rerank_with_diversity

# ── LLM Client ────────────────────────────────────────────────────────
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ── Globals ───────────────────────────────────────────────────────────
_embed_model = None
_qa_database = []

EMBED_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

# ══════════════════════════════════════════════════════════════════════
# EMBEDDING MODEL
# ══════════════════════════════════════════════════════════════════════

def get_embed_model() -> HuggingFaceEmbeddings:

    global _embed_model

    if _embed_model is None:

        _embed_model = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL_NAME,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

        print("[Embed] Model loaded")

    return _embed_model


def get_qdrant():
    return get_client()

# ══════════════════════════════════════════════════════════════════════
# QA DATABASE
# ══════════════════════════════════════════════════════════════════════

def load_qa_database() -> list[dict]:

    global _qa_database

    if _qa_database:
        return _qa_database

    data_folder = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data"
    )

    data_folder = os.path.normpath(data_folder)

    for filepath in sorted(
        glob.glob(os.path.join(data_folder, "*.json"))
    ):

        try:

            with open(
                filepath,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            items = data if isinstance(data, list) else [data]

            for item in items:

                if not isinstance(item, dict):
                    continue

                answer = item.get("answer", "")

                if not isinstance(answer, str):
                    continue

                if not answer.strip():
                    continue

                q_field = item.get("question", "")

                if isinstance(q_field, str):

                    questions = [q_field]

                elif isinstance(q_field, list):

                    questions = [
                        q for q in q_field
                        if isinstance(q, str)
                        and q.strip()
                    ]

                else:
                    continue

                for q in questions:

                    if not q.strip():
                        continue

                    _qa_database.append({
                        "question": q.strip(),
                        "answer": answer.strip(),
                        "source": os.path.basename(filepath),
                    })

        except Exception as e:

            print(
                f"[DB] Error loading "
                f"{filepath}: {e}"
            )

    print(f"[DB] Loaded {len(_qa_database)} QA pairs")

    return _qa_database

# ══════════════════════════════════════════════════════════════════════
# LANGUAGE HELPER
# ══════════════════════════════════════════════════════════════════════

def is_hindi_text(text: str) -> bool:

    if not text:
        return False

    devanagari = sum(
        1 for c in text
        if '\u0900' <= c <= '\u097F'
    )

    total = len(text.replace(" ", ""))

    return (
        total > 0
        and (devanagari / total) > 0.2
    )


def translate_answer_if_needed(
    answer: str,
    lang: str,
    question: str
) -> str:

    answer_is_hindi = is_hindi_text(answer)

    if lang == "en" and not answer_is_hindi:
        return answer

    if lang in ("hi", "ga", "ku") and answer_is_hindi:
        return answer

    if lang == "en":

        prompt = (
            "Translate the following text to English.\n\n"
            f"{answer}"
        )

        system = (
            "You are a translator."
        )

    else:

        prompt = (
            "Translate the following text to Hindi.\n\n"
            f"{answer}"
        )

        system = (
            "You are a translator."
        )

    try:

        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            max_tokens=400,
            temperature=0.1,
        )

        translated = (
            r.choices[0]
            .message.content
            .strip()
        )

        print("[TRANSLATE] ✅ Groq translated")

        return translated

    except Exception as e:

        print(
            f"[TRANSLATE] Groq failed: {e}"
        )

        return answer

# ══════════════════════════════════════════════════════════════════════
# LLM ANSWER
# ══════════════════════════════════════════════════════════════════════

def llm_answer(
    question: str,
    context: str,
    lang: str,
    history: str = ""
) -> str:

    prompt = build_prompt(
        question,
        context,
        lang,
        history
    )

    try:

        r = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content":
                    (
                        "You are Diksha, "
                        "AI assistant for GBPIET."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                },
            ],
            max_tokens=500,
            temperature=0.3,
        )

        answer = (
            r.choices[0]
            .message.content
            .strip()
        )

        print("[LLM] ✅ Groq answered")

        return answer

    except Exception as e:

        print(f"[LLM] Groq failed: {e}")

        return (
            "I'm sorry, I couldn't "
            "generate a response right now."
        )

# ══════════════════════════════════════════════════════════════════════
# IMPORTANT FIX
# ══════════════════════════════════════════════════════════════════════

# REMOVE THIS LINE FROM get_answer():
#
# build_bm25_index()
#
# DO NOT rebuild BM25 on every request.
#
# It should only run during startup in main.py
