"""
Session-scoped conversation memory. Resolves referents like "it", "there",
"that region" against the previous turn's result — this is what turns
independent Q&A into a spatial conversation ("highlight the water body" ->
"how large is it?" -> "what changed there?").

Kept as an in-process dict for the 4-day build. Swap for Redis if you need
multi-worker deployment; the interface (get/update) won't need to change.
"""
import re
import time
import uuid

_SESSIONS: dict[str, dict] = {}
_SESSION_TTL_SECONDS = 60 * 60  # 1 hour of inactivity expires a session

_REFERENT_PATTERNS = [r"\bit\b", r"\bthere\b", r"\bthat region\b", r"\bthis area\b", r"\bthe same\b"]


def new_session() -> str:
    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = {"turns": [], "last_active": time.time()}
    return session_id


def _prune_expired() -> None:
    now = time.time()
    expired = [sid for sid, s in _SESSIONS.items() if now - s["last_active"] > _SESSION_TTL_SECONDS]
    for sid in expired:
        del _SESSIONS[sid]


def has_referent(query: str) -> bool:
    text = query.lower()
    return any(re.search(p, text) for p in _REFERENT_PATTERNS)


def get_last_entity(session_id: str | None) -> dict | None:
    """Returns the last grounded region / task result the user can refer back to."""
    if not session_id or session_id not in _SESSIONS:
        return None
    turns = _SESSIONS[session_id]["turns"]
    return turns[-1]["entity"] if turns else None


def record_turn(session_id: str, query: str, task: str, entity: dict | None) -> None:
    _prune_expired()
    if session_id not in _SESSIONS:
        _SESSIONS[session_id] = {"turns": [], "last_active": time.time()}
    _SESSIONS[session_id]["turns"].append({"query": query, "task": task, "entity": entity})
    _SESSIONS[session_id]["last_active"] = time.time()


def turn_count(session_id: str | None) -> int:
    if not session_id or session_id not in _SESSIONS:
        return 0
    return len(_SESSIONS[session_id]["turns"])
