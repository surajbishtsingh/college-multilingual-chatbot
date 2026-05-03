# memory/memory_manager.py
# Extracts facts from conversations and manages user memory

import re
from memory.database import (
    save_user_fact, get_user_facts,
    update_user_profile, get_recent_history,
    save_message,
)

# ── Fact extraction patterns ───────────────────────────────────────────
FACT_PATTERNS = [
    # Name
    (r"my name is ([A-Za-z\s]+)",           "name"),
    (r"i am ([A-Za-z\s]+), a student",      "name"),
    (r"mera naam ([A-Za-z\u0900-\u097F\s]+) hai", "name"),

    # Branch/Department
    (r"i am (?:from|in) (cse|ece|ee|me|civil|biotech|mca|mtech)",  "branch"),
    (r"i study (cse|ece|ee|me|civil|biotech|mca|mtech)",           "branch"),
    (r"(cse|ece|ee|me|civil|biotech|mca|mtech) (?:student|branch|department)", "branch"),
    (r"mera branch ([a-z]+) hai",           "branch"),

    # Semester/Year
    (r"i am in (\d+)(?:st|nd|rd|th) (?:semester|sem)",  "semester"),
    (r"(\d+)(?:st|nd|rd|th) year student",               "year"),
    (r"semester (\d+)",                                   "semester"),

    # Course
    (r"i am (?:a|an) (btech|b\.tech|mca|mtech|m\.tech|phd) student", "course"),
    (r"doing (btech|b\.tech|mca|mtech|m\.tech|phd)",                  "course"),
]


def extract_facts(text: str) -> dict[str, str]:
    """
    Extract user facts from a message.
    Returns dict of { fact_type: fact_value }.
    """
    facts    = {}
    text_low = text.lower().strip()

    for pattern, fact_type in FACT_PATTERNS:
        match = re.search(pattern, text_low, re.IGNORECASE)
        if match:
            value = match.group(1).strip().upper()
            facts[fact_type] = value

    return facts


async def process_user_message(session_id: str, message: str, lang: str = "en"):
    """
    Called every time a user sends a message.
    1. Save message to history
    2. Extract and save any facts mentioned
    3. Update user profile if facts found
    """
    # Save message
    await save_message(session_id, "user", message, lang)

    # Extract facts
    facts = extract_facts(message)
    for fact_type, fact_value in facts.items():
        await save_user_fact(session_id, fact_type, fact_value)
        print(f"[Memory] Extracted: {fact_type} = {fact_value}")

    # Update profile for key facts
    profile_updates = {}
    if "name"     in facts: profile_updates["name"]     = facts["name"]
    if "branch"   in facts: profile_updates["branch"]   = facts["branch"]
    if "semester" in facts: profile_updates["semester"]  = facts["semester"]
    if "course"   in facts: profile_updates["course"]   = facts["course"]
    if lang:                profile_updates["language"]  = lang

    if profile_updates:
        await update_user_profile(session_id, **profile_updates)


async def process_bot_message(session_id: str, message: str, lang: str = "en"):
    """Save bot response to history."""
    await save_message(session_id, "diksha", message, lang)


async def build_memory_context(session_id: str) -> str:
    facts   = await get_user_facts(session_id)
    history = await get_recent_history(session_id, limit=6)

    parts = []

    # ── User profile ──────────────────────────────────────────
    if facts:
        profile_parts = []
        if "name"     in facts: profile_parts.append(f"Name: {facts['name']}")
        if "branch"   in facts: profile_parts.append(f"Branch: {facts['branch']}")
        if "semester" in facts: profile_parts.append(f"Semester: {facts['semester']}")
        if "course"   in facts: profile_parts.append(f"Course: {facts['course']}")
        if profile_parts:
            parts.append("Student Profile: " + ", ".join(profile_parts))

    # ── Recent chat history ────────────────────────────────────
    if history:
        parts.append("\nRecent Conversation:")
        for msg in history:
            role = "Student" if msg["role"] == "user" else "Diksha"
            parts.append(f"{role}: {msg['message']}")

    return "\n".join(parts) if parts else ""