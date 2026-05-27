# 🤖 Diksha — GBPIET College AI Chatbot

<div align="center">

![Diksha Chatbot](https://img.shields.io/badge/Diksha-GBPIET%20Chatbot-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-green?style=for-the-badge&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal?style=for-the-badge&logo=fastapi)
![Railway](https://img.shields.io/badge/Deploy-Railway-purple?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-2.0.0-orange?style=for-the-badge)

**Diksha** is an AI-powered multilingual chatbot for GBPIET (Govind Ballabh Pant Institute of Engineering & Technology), Pauri Garhwal. It answers student queries about admissions, fees, placements, faculty, hostel, and more — in English, Hindi, Garhwali, and Kumauni.

**Live Backend:** `https://motivated-forgiveness-production-1681.up.railway.app`

</div>

---

## ✨ Features

- 🌐 **Multilingual** — English, Hindi (हिंदी), Garhwali (गढ़वाली), Kumauni (कुमाऊँनी)
- 🔍 **Hybrid RAG Search** — BM25 + Qdrant vector search + RRF fusion
- 🧠 **Persistent Memory** — Remembers user context across the session (PostgreSQL / SQLite)
- 🌍 **Live Website Scraping** — Auto-scrapes GBPIET website (no Chrome needed on Railway)
- 🔊 **Text-to-Speech** — Sarvam AI Indian female voice with gTTS fallback
- 🤖 **Multi-Key LLM** — 4 Groq API keys (8 attempts) → Gemini 2.0 Flash fallback
- 📊 **Evaluation System** — Confusion matrix for language, scope, and answer quality
- 🔄 **Roman↔Devanagari Bridge** — Mixed-script queries work (e.g. `"hostel kitne h"`)
- 🛡️ **Language Drift Guard** — Auto-corrects if LLM responds in wrong language
- ⚡ **Startup Auto-check** — Auto-scrapes if Qdrant data is empty on boot

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Language Detection (auto-detect / frontend selection / section override)
    │
    ▼
Out-of-Scope Check  (LLM classifier — IPL, weather, Bollywood → blocked)
    │
    ▼
Step 0a: Specific Role Match    (HOD, Dean, Warden, Director → fast dataset answer)
    │
Step 0b: Direct Keyword Match   (≤2 words: "fees", "hostel" → fast lookup)
    │
Step 1:  Exact FAQ Match        (exact string match in QA database)
    │
Step 2:  Keyword Overlap Match  (BM25-style scoring with multilingual bridge)
    │
Step 3a: [ga/ku] Dataset-Only   (NO LLM — dataset search with Roman↔Devanagari bridge)
    │
Step 3b: [hi/en] Hybrid RAG + LLM
         ├── BM25 Search        (rank-bm25 on local JSON)
         ├── Qdrant Vector      (multi-collection semantic search)
         ├── RRF Fusion         (Reciprocal Rank Fusion, weights: BM25=0.4 / Vector=0.6)
         ├── SerpAPI fallback   (if score < 0.05)
         └── Groq LLaMA 3.3 70B → LLaMA3 70B → Gemini 2.0 Flash
    │
    ▼
Language Drift Guard (corrects wrong-language LLM responses)
    │
    ▼
Answer + TTS (Sarvam AI → gTTS fallback)
```

---

## 📁 Project Structure

```
college-multilingual-chatbot/
│
├── backend/
│   ├── main.py                  # FastAPI app — all API routes (v2.0.0)
│   ├── language_detector.py     # Auto-detects en / hi / ga / ku
│   ├── intent_detector.py       # Routes query to correct Qdrant collection
│   ├── voice.py                 # TTS: Sarvam AI (primary) → gTTS (fallback)
│   ├── qdrant_setup.py          # Qdrant client + collection definitions
│   ├── build_kb.py              # One-time script: uploads JSON → Qdrant Cloud
│   ├── run_scrape.py            # Railway-compatible scraper (requests + BS4, no Chrome)
│   ├── evaluate_chatbot.py      # Confusion matrix evaluation (sklearn + matplotlib)
│   │
│   ├── rag/
│   │   ├── kb_query.py          # Main 4-step query pipeline + RAG + LLM
│   │   ├── hybrid_search.py     # Qdrant multi-collection vector search
│   │   ├── bm25_search.py       # BM25 keyword search
│   │   ├── fusion.py            # Reciprocal Rank Fusion (RRF)
│   │   ├── embeddings.py        # HuggingFace embedding model loader
│   │   ├── internet_search.py   # SerpAPI fallback for low-confidence queries
│   │   ├── groq_manager.py      # Groq key stats + response cache
│   │   └── reranker.py          # Optional diversity reranker
│   │
│   ├── memory/
│   │   ├── database.py          # Dual DB: PostgreSQL (asyncpg) / SQLite (aiosqlite)
│   │   └── memory_manager.py    # Per-session conversation history + user facts
│   │
│   ├── scraper/
│   │   └── scheduler.py         # APScheduler — periodic auto-scrape
│   │
│   └── data/
│       └── *.json               # FAQ & KB files (en / hi / ga / ku)
│
├── frontend/
│   └── diksha-chat/
│       ├── src/
│       │   ├── App.jsx          # React chat widget
│       │   └── App.css
│       └── assets/
│           ├── logo.png         # Diksha avatar
│           └── image.png        # Counselling 2026 notice
│
├── requirements.txt
├── railway.json                 # Railway deployment config
└── .env                         # API keys (not committed)
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Backend** | FastAPI + Uvicorn + Gunicorn |
| **Primary LLM** | Groq — LLaMA 3.3 70B Versatile (4 API keys, 8 attempts) |
| **Fallback LLM** | Groq — LLaMA3 70B 8192, then Google Gemini 2.0 Flash |
| **Embeddings** | `paraphrase-multilingual-MiniLM-L12-v2` (384-dim, HuggingFace) |
| **Vector DB** | Qdrant (local or cloud via `QDRANT_URL`) |
| **Keyword Search** | BM25 (`rank-bm25`) |
| **Fusion** | Reciprocal Rank Fusion (RRF) |
| **Database** | PostgreSQL (`asyncpg`) in production / SQLite (`aiosqlite`) in dev |
| **Web Scraping** | `requests` + BeautifulSoup4 (Railway-compatible, no Chrome) |
| **Scheduler** | APScheduler |
| **TTS Primary** | Sarvam AI — `bulbul:v3`, speaker: `neha` (Indian female voice) |
| **TTS Fallback** | gTTS (Indian accent, `tld=co.in`) |
| **LangChain** | `langchain-huggingface`, `langchain-community` (embeddings + text splitting) |
| **Frontend** | React (Vite) + CSS |
| **Deployment** | Railway (`railway.json`) |

---

## 🗂️ Qdrant Collections

| Collection | Language | Contents |
|---|---|---|
| `gbpiet_faq` | English | General FAQ |
| `gbpiet_kb_en` | English | Full English knowledge base |
| `gbpiet_kb_hi` | Hindi | Hindi knowledge base |
| `gbpiet_hostel` | English | Hostel-specific data |
| `gbpiet_fees` | English | Fee structure data |
| `gbpiet_admissions` | English | Admission process data |
| `gbpiet_web` / `website` | English | Scraped GBPIET website chunks |

---

## 🔧 Environment Variables

Create a `.env` file in the `backend/` folder:

```env
# LLM — Groq (up to 4 keys for fallback)
GROQ_API_KEY=your_key_1
GROQ_API_KEY_2=your_key_2
GROQ_API_KEY_3=your_key_3
GROQ_API_KEY_4=your_key_4

# LLM — Gemini fallback
GEMINI_API_KEY=your_gemini_key

# TTS — Sarvam AI (Indian female voice)
SARVAM_API_KEY=your_sarvam_key

# Vector DB — leave empty to use local Qdrant
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your_qdrant_key

# Database — leave empty to use SQLite
DATABASE_URL=postgresql://user:pass@host/dbname

# Internet fallback search (optional)
SERPAPI_KEY=your_serpapi_key

# Misc
ENVIRONMENT=production
COLLEGE_WEBSITE_URL=https://gbpiet.ac.in
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.11+
- Node.js 18+ (for frontend)
- Groq API key — free at [console.groq.com](https://console.groq.com)
- Gemini API key — free at [aistudio.google.com](https://aistudio.google.com)

> ⚠️ **No Chrome needed** — the scraper now uses `requests` + BeautifulSoup and works on any server.

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/surajbishtsingh/college-multilingual-chatbot.git
cd college-multilingual-chatbot/backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file and fill in your API keys
cp .env.example .env

# 4. Build the knowledge base (first time only — uploads JSON to Qdrant)
python build_kb.py

# 5. Run the scraper to populate website collection
python run_scrape.py

# 6. Start the backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd ../frontend/diksha-chat
npm install
npm run dev
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Status + version info |
| `GET` | `/health` | Health check (all keys, DB type, Qdrant mode) |
| `POST` | `/chat` | Main chat endpoint |
| `POST` | `/tts` | Text-to-speech → returns base64 audio |
| `GET` | `/scrape-status` | Last scrape job status |
| `POST` | `/scrape-now` | Trigger manual scrape in background |
| `GET` | `/evaluate` | Run confusion matrix evaluation (results in logs) |
| `GET` | `/admin/groq-stats` | Groq key usage stats |
| `POST` | `/admin/clear-cache` | Clear Groq response cache |
| `POST` | `/admin/rebuild-kb` | Rebuild knowledge base from JSON |

### Chat Request Example

```json
POST /chat
{
  "question": "What are the fees for BTech?",
  "session_id": "abc-123",
  "language": "en",
  "section": "english"
}
```

---

## 📊 Evaluation

Run a full confusion matrix evaluation:

```bash
cd backend
python evaluate_chatbot.py
```

**Generates:**
- `cm_language_detection.png` — Language detection confusion matrix
- `cm_scope_detection.png` — Out-of-scope classifier matrix
- `cm_answer_quality.png` — Answer quality matrix
- `evaluation_report.json` — Full JSON report

**Test coverage:**
- 24 language detection tests (en / hi / ga / ku)
- 20 scope detection tests (in-scope vs out-of-scope)
- 8 answer quality tests (keyword presence check)

---

## 🗣️ Multilingual Notes

| Language | Code | Script | Pipeline |
|---|---|---|---|
| English | `en` | Latin | RAG + LLM |
| Hindi | `hi` | Devanagari | RAG + LLM |
| Garhwali | `ga` | Devanagari | Dataset-only (no LLM) |
| Kumauni | `ku` | Devanagari | Dataset-only (no LLM) |

- Mixed-script queries like `"hostel kitne h"` work via a **Roman↔Devanagari bridge** (50+ word mappings)
- Garhwali and Kumauni answers use dataset-only search to avoid translation errors
- A **language drift guard** detects if LLM responds in wrong language and auto-corrects it

---

## 👨‍💻 Team

Built with ❤️ for GBPIET students by **Diksha Phartyal, Anjali Gusain, Suraj Bisht, Priyanshu Dhyani** (MCA Team, GBPIET).

---

## 📄 License

MIT License — free to use and modify.
