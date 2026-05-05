# memory/database.py
# Supports both PostgreSQL (production) and SQLite (development)

import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")
USE_POSTGRES = bool(DATABASE_URL) and "postgresql" in DATABASE_URL

# ── SQLite fallback path ───────────────────────────────────────────────
SQLITE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "diksha_memory.db"
)

print(f"[DB] Using {'PostgreSQL' if USE_POSTGRES else 'SQLite'}")


# ══════════════════════════════════════════════════════════════════════
# PostgreSQL SETUP (using asyncpg)
# ══════════════════════════════════════════════════════════════════════
_pg_pool = None


async def get_pg_pool():
    """Get or create PostgreSQL connection pool."""
    global _pg_pool
    if _pg_pool is None:
        import asyncpg

        db_url = DATABASE_URL
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)

        _pg_pool = await asyncpg.create_pool(
            dsn=db_url,
            min_size=1,
            max_size=10,
            command_timeout=30,
        )
        print("[DB] PostgreSQL pool created")
    return _pg_pool


async def close_pg_pool():
    """Close PostgreSQL pool on shutdown."""
    global _pg_pool
    if _pg_pool:
        await _pg_pool.close()
        _pg_pool = None
        print("[DB] PostgreSQL pool closed")


# ══════════════════════════════════════════════════════════════════════
# INIT — Create tables
# ══════════════════════════════════════════════════════════════════════
async def init_db():
    if USE_POSTGRES:
        await _init_postgres()
    else:
        await _init_sqlite()


async def _init_postgres():
    pool = await get_pg_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
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
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id           SERIAL PRIMARY KEY,
                session_id   TEXT,
                role         TEXT,
                message      TEXT,
                language     TEXT,
                timestamp    TEXT
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_session
            ON conversations(session_id)
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_facts (
                id           SERIAL PRIMARY KEY,
                session_id   TEXT,
                fact_type    TEXT,
                fact_value   TEXT,
                confidence   REAL DEFAULT 1.0,
                created_at   TEXT,
                UNIQUE(session_id, fact_type)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS scrape_history (
                id           SERIAL PRIMARY KEY,
                url          TEXT UNIQUE,
                content_hash TEXT,
                scraped_at   TEXT,
                chunk_count  INTEGER DEFAULT 0
            )
        """)
    print("[DB] ✅ PostgreSQL tables ready")


async def _init_sqlite():
    import aiosqlite
    async with aiosqlite.connect(SQLITE_PATH) as db:
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
    print("[DB] ✅ SQLite tables ready")


# ══════════════════════════════════════════════════════════════════════
# USER OPERATIONS
# ══════════════════════════════════════════════════════════════════════
async def get_or_create_user(session_id: str) -> dict:
    now = datetime.utcnow().isoformat()

    if USE_POSTGRES:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE session_id = $1", session_id
            )
            if row:
                return dict(row)
            await conn.execute(
                """INSERT INTO users (session_id, created_at, updated_at)
                   VALUES ($1, $2, $3)
                   ON CONFLICT (session_id) DO NOTHING""",
                session_id, now, now
            )
            return {"session_id": session_id, "created_at": now}
    else:
        import aiosqlite
        async with aiosqlite.connect(SQLITE_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM users WHERE session_id = ?", (session_id,)
            )
            row = await cursor.fetchone()
            if row:
                return dict(row)
            await db.execute(
                "INSERT INTO users (session_id, created_at, updated_at) VALUES (?, ?, ?)",
                (session_id, now, now)
            )
            await db.commit()
            return {"session_id": session_id, "created_at": now}


async def update_user_profile(session_id: str, **kwargs):
    """
    Update user profile fields.
    
    BUG FIX: Previous version had wrong parameter index in UPDATE query.
    Now each column is updated individually with explicit ::text cast
    to avoid 'could not determine data type of parameter $1' error.
    """
    valid_fields = {"name", "branch", "semester", "course", "language", "year"}
    updates = {k: v for k, v in kwargs.items() if k in valid_fields}
    if not updates:
        return

    now = datetime.utcnow().isoformat()

    if USE_POSTGRES:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            # Ensure user row exists first
            await conn.execute(
                """INSERT INTO users (session_id, created_at, updated_at)
                   VALUES ($1, $2, $2)
                   ON CONFLICT (session_id) DO NOTHING""",
                session_id, now
            )
            # ✅ FIX: Update each column separately with explicit ::text cast
            # This avoids asyncpg "could not determine data type of parameter $1"
            for col, val in updates.items():
                await conn.execute(
                    f"UPDATE users SET {col} = $1::text, updated_at = $2::text WHERE session_id = $3::text",
                    str(val), now, str(session_id)
                )
    else:
        import aiosqlite
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values     = list(updates.values()) + [now, session_id]
        async with aiosqlite.connect(SQLITE_PATH) as db:
            await db.execute(
                f"UPDATE users SET {set_clause}, updated_at = ? WHERE session_id = ?",
                values
            )
            await db.commit()


# ══════════════════════════════════════════════════════════════════════
# CONVERSATION OPERATIONS
# ══════════════════════════════════════════════════════════════════════
async def save_message(
    session_id: str,
    role:       str,
    message:    str,
    language:   str = "en",
):
    now = datetime.utcnow().isoformat()

    if USE_POSTGRES:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO users (session_id, created_at, updated_at)
                   VALUES ($1, $2, $2)
                   ON CONFLICT (session_id) DO NOTHING""",
                session_id, now
            )
            await conn.execute(
                """INSERT INTO conversations
                   (session_id, role, message, language, timestamp)
                   VALUES ($1, $2, $3, $4, $5)""",
                session_id, role, message, language, now
            )
    else:
        import aiosqlite
        async with aiosqlite.connect(SQLITE_PATH) as db:
            await db.execute(
                """INSERT INTO conversations
                   (session_id, role, message, language, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, role, message, language, now)
            )
            await db.commit()


async def get_recent_history(session_id: str, limit: int = 6) -> list[dict]:
    if USE_POSTGRES:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT role, message, language, timestamp
                   FROM conversations
                   WHERE session_id = $1
                   ORDER BY timestamp DESC
                   LIMIT $2""",
                session_id, limit
            )
            return [dict(r) for r in reversed(rows)]
    else:
        import aiosqlite
        async with aiosqlite.connect(SQLITE_PATH) as db:
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
            return [dict(r) for r in reversed(rows)]


# ══════════════════════════════════════════════════════════════════════
# FACT OPERATIONS
# ══════════════════════════════════════════════════════════════════════
async def save_user_fact(
    session_id: str,
    fact_type:  str,
    fact_value: str,
    confidence: float = 1.0,
):
    now = datetime.utcnow().isoformat()

    if USE_POSTGRES:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO user_facts
                   (session_id, fact_type, fact_value, confidence, created_at)
                   VALUES ($1, $2, $3, $4, $5)
                   ON CONFLICT (session_id, fact_type)
                   DO UPDATE SET
                       fact_value = EXCLUDED.fact_value,
                       confidence = EXCLUDED.confidence""",
                session_id, fact_type, fact_value, confidence, now
            )
    else:
        import aiosqlite
        async with aiosqlite.connect(SQLITE_PATH) as db:
            await db.execute(
                """INSERT INTO user_facts
                   (session_id, fact_type, fact_value, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(session_id, fact_type)
                   DO UPDATE SET
                       fact_value = excluded.fact_value,
                       confidence = excluded.confidence""",
                (session_id, fact_type, fact_value, confidence, now)
            )
            await db.commit()


async def get_user_facts(session_id: str) -> dict:
    if USE_POSTGRES:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT fact_type, fact_value FROM user_facts WHERE session_id = $1",
                session_id
            )
            return {r["fact_type"]: r["fact_value"] for r in rows}
    else:
        import aiosqlite
        async with aiosqlite.connect(SQLITE_PATH) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT fact_type, fact_value FROM user_facts WHERE session_id = ?",
                (session_id,)
            )
            rows = await cursor.fetchall()
            return {r["fact_type"]: r["fact_value"] for r in rows}


# ══════════════════════════════════════════════════════════════════════
# SCRAPE HISTORY
# ══════════════════════════════════════════════════════════════════════
async def get_scraped_hashes() -> set[str]:
    if USE_POSTGRES:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT content_hash FROM scrape_history")
            return {r["content_hash"] for r in rows}
    else:
        import aiosqlite
        async with aiosqlite.connect(SQLITE_PATH) as db:
            cursor = await db.execute("SELECT content_hash FROM scrape_history")
            rows   = await cursor.fetchall()
            return {r[0] for r in rows}


async def save_scrape_record(url: str, content_hash: str, chunk_count: int):
    now = datetime.utcnow().isoformat()

    if USE_POSTGRES:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO scrape_history (url, content_hash, scraped_at, chunk_count)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (url)
                   DO UPDATE SET
                       content_hash = EXCLUDED.content_hash,
                       scraped_at   = EXCLUDED.scraped_at,
                       chunk_count  = EXCLUDED.chunk_count""",
                url, content_hash, now, chunk_count
            )
    else:
        import aiosqlite
        async with aiosqlite.connect(SQLITE_PATH) as db:
            await db.execute(
                """INSERT INTO scrape_history (url, content_hash, scraped_at, chunk_count)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(url)
                   DO UPDATE SET
                       content_hash = excluded.content_hash,
                       scraped_at   = excluded.scraped_at,
                       chunk_count  = excluded.chunk_count""",
                (url, content_hash, now, chunk_count)
            )
            await db.commit()


# ══════════════════════════════════════════════════════════════════════
# CLEANUP & STATS
# ══════════════════════════════════════════════════════════════════════
async def cleanup_old_conversations(days_to_keep: int = 30):
    if USE_POSTGRES:
        pool = await get_pg_pool()
        async with pool.acquire() as conn:
            # ✅ FIX: INTERVAL does not accept $1 parameter — use string format
            await conn.execute(
                f"DELETE FROM conversations WHERE timestamp < NOW() - INTERVAL '{days_to_keep} days'"
            )
            print(f"[DB] Cleaned conversations older than {days_to_keep} days (PostgreSQL)")
    else:
        import aiosqlite
        async with aiosqlite.connect(SQLITE_PATH) as db:
            await db.execute(
                "DELETE FROM conversations WHERE timestamp < datetime('now', ?)",
                (f"-{days_to_keep} days",)
            )
            await db.commit()
        print(f"[DB] Cleaned conversations older than {days_to_keep} days (SQLite)")


async def get_db_stats() -> dict:
    stats  = {}
    tables = ["users", "conversations", "user_facts", "scrape_history"]

    if USE_POSTGRES:
        try:
            pool = await get_pg_pool()
            async with pool.acquire() as conn:
                for table in tables:
                    row = await conn.fetchrow(f"SELECT COUNT(*) as c FROM {table}")
                    stats[table] = row["c"]
            stats["db_type"]    = "PostgreSQL"
            stats["db_size_kb"] = "N/A (cloud)"
        except Exception as e:
            stats["error"] = str(e)
    else:
        import aiosqlite
        async with aiosqlite.connect(SQLITE_PATH) as db:
            for table in tables:
                try:
                    cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
                    count  = (await cursor.fetchone())[0]
                    stats[table] = count
                except Exception:
                    stats[table] = 0
        try:
            size_bytes          = os.path.getsize(SQLITE_PATH)
            stats["db_size_kb"] = round(size_bytes / 1024, 1)
        except Exception:
            stats["db_size_kb"] = 0
        stats["db_type"] = "SQLite"

    return stats
