# kb_query.py — FINAL ALL-IN-ONE (Railway Safe)

import os
import json
import glob
import re
import asyncio
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
import google.generativeai as genai

load_dotenv()

# ═══════════════════════════════════════════════════════
# 🔥 LAZY CLIENTS (NO STARTUP CRASH)
# ═══════════════════════════════════════════════════════
groq_client = None
gemini_model = None

def get_groq_client():
    global groq_client
    if groq_client is None:
        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("[Groq] Initialized")
    return groq_client

def get_gemini_model():
    global gemini_model
    if gemini_model is None:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        print("[Gemini] Initialized")
    return gemini_model


# ═══════════════════════════════════════════════════════
# 🔥 EMBEDDING MODEL (LAZY)
# ═══════════════════════════════════════════════════════
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("[Embed] Loaded")
    return _embed_model


# ═══════════════════════════════════════════════════════
# 🔥 LOAD LOCAL QA DATABASE
# ═══════════════════════════════════════════════════════
_qa_database = []

def load_qa_database():
    global _qa_database
    if _qa_database:
        return _qa_database

    data_folder = os.path.join(os.path.dirname(__file__), "..", "data")

    for file in glob.glob(os.path.join(data_folder, "*.json")):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                data = [data]

            for item in data:
                if not isinstance(item, dict):
                    continue

                q = item.get("question")
                a = item.get("answer")

                if not q or not a:
                    continue

                if isinstance(q, str):
                    _qa_database.append({"question": q.lower(), "answer": a})
                elif isinstance(q, list):
                    for qq in q:
                        _qa_database.append({"question": qq.lower(), "answer": a})

        except Exception as e:
            print("[DB ERROR]", e)

    print(f"[DB] Loaded {_qa_database.__len__()} QA")
    return _qa_database


# ═══════════════════════════════════════════════════════
# 🔥 EXACT MATCH
# ═══════════════════════════════════════════════════════
def exact_match(q):
    q = q.lower().strip()
    for item in load_qa_database():
        if q == item["question"]:
            print("[MATCH] Exact")
            return item["answer"]
    return None


# ═══════════════════════════════════════════════════════
# 🔥 SIMPLE KEYWORD MATCH
# ═══════════════════════════════════════════════════════
def keyword_match(q):
    q_words = set(q.lower().split())

    best = None
    best_score = 0

    for item in load_qa_database():
        words = set(item["question"].split())
        score = len(q_words & words)

        if score > best_score:
            best_score = score
            best = item["answer"]

    if best_score >= 2:
        print("[MATCH] Keyword")
        return best

    return None


# ═══════════════════════════════════════════════════════
# 🔥 TRANSLATION (Groq → Gemini fallback)
# ═══════════════════════════════════════════════════════
def translate_answer(answer, lang):
    if lang == "en":
        return answer

    prompt = f"Translate to Hindi:\n{answer}"

    try:
        client = get_groq_client()
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )
        return r.choices[0].message.content.strip()

    except:
        pass

    try:
        model = get_gemini_model()
        r = model.generate_content(prompt)
        return r.text.strip()
    except:
        return answer


# ═══════════════════════════════════════════════════════
# 🔥 LLM ANSWER
# ═══════════════════════════════════════════════════════
def llm_answer(question, context):
    prompt = f"""
Context:
{context}

Question:
{question}

Answer clearly:
"""

    try:
        client = get_groq_client()
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )
        print("[LLM] Groq")
        return r.choices[0].message.content.strip()

    except:
        pass

    try:
        model = get_gemini_model()
        r = model.generate_content(prompt)
        print("[LLM] Gemini")
        return r.text.strip()
    except:
        return "Sorry, I couldn't answer."


# ═══════════════════════════════════════════════════════
# 🔥 MAIN FUNCTION
# ═══════════════════════════════════════════════════════
def get_answer(question, lang="en"):
    print("\n==============================")
    print("Q:", question)

    # Step 1: Exact
    ans = exact_match(question)
    if ans:
        return translate_answer(ans, lang)

    # Step 2: Keyword
    ans = keyword_match(question)
    if ans:
        return translate_answer(ans, lang)

    # Step 3: LLM fallback
    context = "No structured data found."
    return llm_answer(question, context)


# ═══════════════════════════════════════════════════════
# 🔥 TEST
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    print(get_answer("fees"))
    print(get_answer("hod of cse"))
