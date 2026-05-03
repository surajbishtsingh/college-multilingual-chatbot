# 🤖 Diksha — GBPIET College AI Chatbot

<div align="center">

![Diksha Chatbot](https://img.shields.io/badge/Diksha-GBPIET%20Chatbot-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-green?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal?style=for-the-badge&logo=fastapi)
![Railway](https://img.shields.io/badge/Deploy-Railway-purple?style=for-the-badge)

**Diksha** is an AI-powered multilingual chatbot for GBPIET (Govind Ballabh Pant Institute of Engineering & Technology), Pauri Garhwal. It answers student queries about admissions, fees, placements, faculty, hostel, and more — in English, Hindi, Garhwali, and Kumauni.

</div>

---

## ✨ Features

- 🌐 **Multilingual** — English, Hindi (हिंदी), Garhwali (गढ़वाली), Kumauni (कुमाऊँनी)
- 🔍 **Hybrid RAG Search** — BM25 + Qdrant vector search + RRF fusion
- 🧠 **Persistent Memory** — Remembers user context across the session
- 🌍 **Live Website Scraping** — Auto-scrapes GBPIET website every 24 hours
- 🔊 **Text-to-Speech** — Voice responses for all languages
- 🤖 **Dual LLM** — Groq (LLaMA 3.3) with Gemini 2.0 Flash fallback
- 📊 **Admin Dashboard** — Visit stats, scrape triggers, user memory viewer
- 📧 **Email Reports** — Auto-sends visit reports via SMTP

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Language Detection (auto / frontend selection)
    │
    ▼
Step 0a: Specific Role Match  (HOD, Dean, Warden queries)
    │
Step 0b: Direct Keyword Match (fees, hostel, placement)
    │
Step 1:  Exact Match          (FAQ database)
    │
Step 2:  Keyword Match        (overlap scoring)
    │
Step 3:  Hybrid RAG Pipeline
         ├── BM25 Search      (local JSON data)
         ├── Qdrant Vector    (scraped website chunks)
         └── RRF Fusion       → Groq / Gemini LLM
    │
    ▼
Answer (translated if needed) + TTS
```

---

## 📁 Project Structure

```
backend/
├── main.py                  # FastAPI app — all routes
├── requirements.txt
├── Procfile                 # Railway deploy config
├── railway.json
│
├── rag/
│   ├── kb_query.py          # Main RAG pipeline
│   ├── hybrid_search.py     # Qdrant multi-collection search
│   ├── bm25_search.py       # BM25 keyword search
│   ├── fusion.py            # Reciprocal Rank Fusion
│   ├── embeddings.py        # HuggingFace embeddings
│   ├── internet_search.py   # SerpAPI fallback
│   └── reranker.py
│
├── scraper/
│   ├── crawl_site.py        # Selenium website crawler
│   ├── parse_pages.py       # HTML → text parser
│   └── scheduler.py        # APScheduler 24hr auto-scrape
│
├── memory/
│   ├── database.py          # SQLite async DB
│   └── memory_manager.py    # Session memory manager
│
├── data/                    # FAQ JSON knowledge base
├── qdrant_storage/          # Local Qdrant vector store
├── intent_detector.py       # Query → collection router
├── language_detector.py     # Auto language detection
└── voice.py                 # TTS engine
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.11+
- Google Chrome (for Selenium scraper)
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/diksha-chatbot.git
cd diksha-chatbot/backend

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
cp .env.example .env
# Fill in your API keys

# 5. Run the scraper to populate Qdrant
python run_scrape.py

# 6. Start the server
uvicorn main:app --reload --port 8000
```


## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Uvicorn |
| LLM | Groq (LLaMA 3.3 70B) + Gemini 2.0 Flash |
| Vector DB | Qdrant (local / cloud) |
| Embeddings | paraphrase-multilingual-MiniLM-L12-v2 |
| Keyword Search | BM25 (rank-bm25) |
| Web Scraping | Selenium + BeautifulSoup |
| Memory | SQLite (aiosqlite) |
| Scheduler | APScheduler |
| TTS | gTTS / edge-tts |
| Deployment | Railway |

---

## 👨‍💻 Author

Built with ❤️ for GBPIET students.

---

## 📄 License

MIT License — free to use and modify.
