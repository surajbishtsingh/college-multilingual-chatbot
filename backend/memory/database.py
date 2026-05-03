# memory/database.py
# SQLite database for user memory and conversation history

import os
import aiosqlite
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "diksha_memory.db")

async def cleanup_old_conversations(days_to_keep: int = 30):
    """
    Delete conversations older than N days.
    Call this from scheduler — runs automatically.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """DELETE FROM conversations
               WHERE timestamp < datetime('now', ?)""",
            (f"-{days_to_keep} days",)
        )
        await db.commit()

    print(f"[DB] Cleaned conversations older than {days_to_keep} days")


async def get_db_stats() -> dict:
    """Show database size and record counts."""
    async with aiosqlite.connect(DB_PATH) as db:
        stats = {}
        for table in ["users", "conversations", "user_facts", "scrape_history"]:
            try:
                cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
                count  = (await cursor.fetchone())[0]
                stats[table] = count
            except Exception:
                stats[table] = 0

    # File size
    try:
        size_bytes = os.path.getsize(DB_PATH)
        stats["db_size_kb"] = round(size_bytes / 1024, 1)
    except Exception:
        stats["db_size_kb"] = 0

    return stats


async def init_db():
    """Create all tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:

        # User profiles
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                session_id   TEXT PRIMARY KEY,
                name         TEXT,
                branch       TEXT,
                semester     TEXT,
                course       TEXT,
                language     TEXT DEFAULT 'en',
                created_at   TEXT,
                updated_at   TEXT
            )
        """)

        # Conversation history
        await db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT,
                role         TEXT,
                message      TEXT,
                language     TEXT,
                timestamp    TEXT,
                FOREIGN KEY (session_id) REFERENCES users(session_id)
            )
        """)

        # Extracted facts about users
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_facts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT,
                fact_type    TEXT,
                fact_value   TEXT,
                confidence   REAL DEFAULT 1.0,
                created_at   TEXT,
                UNIQUE(session_id, fact_type)
            )
        """)

        # Scrape history — tracks which pages were scraped when
        await db.execute("""
            CREATE TABLE IF NOT EXISTS scrape_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                url          TEXT UNIQUE,
                content_hash TEXT,
                scraped_at   TEXT,
                chunk_count  INTEGER DEFAULT 0
            )
        """)

        await db.commit()

    print("[DB] ✅ Database initialized")


# ── User operations ────────────────────────────────────────────────────
async def get_or_create_user(session_id: str) -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM users WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        if row:
            return dict(row)

        now = datetime.utcnow().isoformat()
        await db.execute(
            "INSERT INTO users (session_id, created_at, updated_at) VALUES (?, ?, ?)",
            (session_id, now, now)
        )
        await db.commit()
        return {"session_id": session_id, "created_at": now}


async def update_user_profile(session_id: str, **kwargs):
    """Update user profile fields."""
    if not kwargs:
        return
    valid_fields = {"name", "branch", "semester", "course", "language"}
    updates = {k: v for k, v in kwargs.items() if k in valid_fields}
    if not updates:
        return

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values     = list(updates.values()) + [datetime.utcnow().isoformat(), session_id]

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            f"UPDATE users SET {set_clause}, updated_at = ? WHERE session_id = ?",
            values
        )
        await db.commit()


# ── Conversation operations ────────────────────────────────────────────
async def save_message(session_id: str, role: str, message: str, language: str = "en"):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO conversations
               (session_id, role, message, language, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, role, message, language, datetime.utcnow().isoformat())
        )
        await db.commit()


async def get_recent_history(session_id: str, limit: int = 6) -> list[dict]:
    """Get last N messages for context."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT role, message, language, timestamp
               FROM conversations
               WHERE session_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (session_id, limit)
        )
        rows = await cursor.fetchall()
        # Reverse to get chronological order
        return [dict(r) for r in reversed(rows)]


# ── Fact operations ────────────────────────────────────────────────────
async def save_user_fact(
    session_id: str,
    fact_type: str,
    fact_value: str,
    confidence: float = 1.0,
):
    """Save or update a user fact (name, branch, semester etc.)"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO user_facts
               (session_id, fact_type, fact_value, confidence, created_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(session_id, fact_type)
               DO UPDATE SET fact_value=excluded.fact_value,
                             confidence=excluded.confidence""",
            (session_id, fact_type, fact_value,
             confidence, datetime.utcnow().isoformat())
        )
        await db.commit()


async def get_user_facts(session_id: str) -> dict:
    """Get all known facts about a user."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT fact_type, fact_value FROM user_facts WHERE session_id = ?",
            (session_id,)
        )
        rows = await cursor.fetchall()
        return {r["fact_type"]: r["fact_value"] for r in rows}


# ── Scrape history ─────────────────────────────────────────────────────
async def get_scraped_hashes() -> set[str]:
    """Get all content hashes of already-scraped pages."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT content_hash FROM scrape_history")
        rows   = await cursor.fetchall()
        return {r[0] for r in rows}


async def save_scrape_record(url: str, content_hash: str, chunk_count: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO scrape_history (url, content_hash, scraped_at, chunk_count)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(url)
               DO UPDATE SET content_hash=excluded.content_hash,
                             scraped_at=excluded.scraped_at,
                             chunk_count=excluded.chunk_count""",
            (url, content_hash, datetime.utcnow().isoformat(), chunk_count)
        )
        await db.commit()