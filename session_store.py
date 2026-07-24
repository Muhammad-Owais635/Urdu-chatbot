"""
In-memory, per-session conversation history with TTL-based expiry.

For a single-process deployment this is sufficient. For multi-worker/
multi-server production deployments, swap this for Redis (interface is
kept intentionally simple to make that swap easy).
"""

import time
import threading
from collections import defaultdict


class SessionStore:
    def __init__(self, max_turns: int, ttl_seconds: int):
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._sessions = defaultdict(list)
        self._last_active = {}
        self._lock = threading.Lock()

    def add_turn(self, session_id: str, role: str, content: str):
        with self._lock:
            self._cleanup_expired()
            history = self._sessions[session_id]
            history.append({"role": role, "content": content})
            if len(history) > self.max_turns * 2:  # *2 for user+assistant pairs
                self._sessions[session_id] = history[-self.max_turns * 2:]
            self._last_active[session_id] = time.time()

    def get_history(self, session_id: str) -> list:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def clear(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)
            self._last_active.pop(session_id, None)

    def _cleanup_expired(self):
        now = time.time()
        expired = [
            sid for sid, last in self._last_active.items()
            if now - last > self.ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_active.pop(sid, None)
