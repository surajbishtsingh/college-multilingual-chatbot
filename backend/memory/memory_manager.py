# memory/memory_manager.py
import re
from memory.database import (
    save_user_fact, get_user_facts,
    update_user_profile, get_recent_history,
    save_message,
)

# ── Fact extraction patterns ───────────────────────────────────────────
FACT_PATTERNS = [
    # Name — English
    (r"my name is ([A-Za-z\s]+)",                "name"),
    (r"i am ([A-Za-z\s]+),?\s+a student",        "name"),
    (r"call me ([A-Za-z\s]+)",                   "name"),
    # Name — Hindi
    (r"mera naam ([A-Za-z\u0900-\u097F\s]+) hai","name"),
    (r"main ([A-Za-z\u0900-\u097F\s]+) hun",     "name"),
    # Name — Garhwali
    (r"मेरु नाम ([A-Za-z\u0900-\u097F\s]+) छ",  "name"),
    # Name — Kumauni
    (r"मेरो नाम ([A-Za-z\u0900-\u097F\s]+) च",  "name"),

    # Branch — English
    (r"i am (?:from|in) (cse|ece|ee|me|civil|biotech|mca|mtech)", "branch"),
    (r"i study (cse|ece|ee|me|civil|biotech|mca|mtech)",          "branch"),
    (r"(cse|ece|ee|me|civil|biotech|mca|mtech) (?:student|branch|department)", "branch"),
    (r"mera branch ([a-z]+) hai",                "branch"),
    # Branch — Hindi/Garhwali/Kumauni
    (r"मैं (cse|ece|ee|me|civil|biotech|mca|mtech) (?:में|मा|से)",  "branch"),

    # Semester
    (r"i am in (\d+)(?:st|nd|rd|th) (?:semester|sem)", "semester"),
    (r"(\d+)(?:st|nd|rd|th) year student",              "year"),
    (r"semester (\d+)",                                  "semester"),
    (r"(\d+)(?:st|nd|rd|th) sem(?:ester)?",             "semester"),

    # Course
    (r"i am (?:a|an) (btech|b\.tech|mca|mtech|m\.tech|phd) student", "course"),
    (r"doing (btech|b\.tech|mca|mtech|m\.tech|phd)",                  "course"),
    (r"pursuing (btech|b\.tech|mca|mtech|m\.tech|phd)",               "course"),
]


def extract_facts(text: str) -> dict[str, str]:
    """Extract user facts from a message."""
    facts    = {}
    text_low = text.lower().strip()

    for pattern, fact_type in FACT_PATTERNS:
        match = re.search(pattern, text_low, re.IGNORECASE)
        if match:
            value = match.group(1).strip().upper()
            if fact_type == "name":
                value = re.sub(
                    r'\b(AND|THE|IS|ARE|FROM|IN|A|AN)\b', '', value
                ).strip()
                if len(value) < 2 or len(value) > 50:
                    continue
            facts[fact_type] = value

    return facts


async def process_user_message(
    session_id: str,
    message:    str,
    lang:       str = "en",
):
    """
    Called every time a user sends a message.
    1. Save to DB
    2. Extract facts
    3. Update profile — uses 'users' table (not 'user_profiles')
    """
    await save_message(session_id, "user", message, lang)

    facts = extract_facts(message)
    for fact_type, fact_value in facts.items():
        await save_user_fact(session_id, fact_type, fact_value)
        print(f"[Memory] Extracted: {fact_type} = {fact_value}")

    profile_updates = {}
    if "name"     in facts: profile_updates["name"]     = facts["name"]
    if "branch"   in facts: profile_updates["branch"]   = facts["branch"]
    if "semester" in facts: profile_updates["semester"] = facts["semester"]
    if "course"   in facts: profile_updates["course"]   = facts["course"]
    if "year"     in facts: profile_updates["year"]     = facts["year"]
    if lang:                profile_updates["language"]  = lang

    if profile_updates:
        try:
            # update_user_profile in database.py uses 'users' table — correct!
            await update_user_profile(session_id, **profile_updates)
        except Exception as e:
            print(f"[Memory] Profile update failed (non-fatal): {e}")


async def process_bot_message(
    session_id: str,
    message:    str,
    lang:       str = "en",
):
    """Save bot response to history."""
    await save_message(session_id, "diksha", message, lang)


async def build_memory_context(session_id: str) -> str:
    """Build context string from user memory for LLM prompt."""
    try:
        facts   = await get_user_facts(session_id)
        history = await get_recent_history(session_id, limit=6)
    except Exception as e:
        print(f"[Memory] Context build failed (non-fatal): {e}")
        return ""

    parts = []

    if facts:
        profile_parts = []
        if "name"     in facts: profile_parts.append(f"Name: {facts['name']}")
        if "branch"   in facts: profile_parts.append(f"Branch: {facts['branch']}")
        if "semester" in facts: profile_parts.append(f"Semester: {facts['semester']}")
        if "year"     in facts: profile_parts.append(f"Year: {facts['year']}")
        if "course"   in facts: profile_parts.append(f"Course: {facts['course']}")
        if profile_parts:
            parts.append("Student Profile: " + ", ".join(profile_parts))

    if history:
        parts.append("\nRecent Conversation:")
        for msg in history:
            role = "Student" if msg["role"] == "user" else "Diksha"
            parts.append(f"{role}: {msg['message']}")

    return "\n".join(parts) if parts else ""
