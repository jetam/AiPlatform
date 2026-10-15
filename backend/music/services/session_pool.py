import asyncio
import time
import uuid

from .composer import Composer

SESSION_COOKIE_NAME = "session_id"


def new_session_id() -> str:
    return uuid.uuid4().hex


class SessionPool:

    def __init__(self, max_sessions: int = 3):
        self.max_sessions = max_sessions
        self._sessions: dict[str, dict] = {}
        self._pool_lock = asyncio.Lock()

    async def get(self, session_id: str) -> dict:
        async with self._pool_lock:
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry["last_used"] = time.monotonic()
                return entry

            if len(self._sessions) >= self.max_sessions:
                self._evict_one_locked()

            entry = {
                "composer": Composer(),
                "lock": asyncio.Lock(),
                "last_used": time.monotonic(),
            }
            self._sessions[session_id] = entry
            return entry

    def _evict_one_locked(self):
        #Must be called while holding self._pool_lock
        oldest_first = sorted(self._sessions.items(), key=lambda kv: kv[1]["last_used"])
        for session_id, entry in oldest_first:
            if not entry["lock"].locked():
                del self._sessions[session_id]
                return

        del self._sessions[oldest_first[0][0]]

    def session_count(self) -> int:
        return len(self._sessions)


session_pool = SessionPool(max_sessions=3)
