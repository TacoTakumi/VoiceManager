"""Session registry for per-connection state management.

SessionEntry holds per-session metadata and TTSPipeline reference.
SessionRegistry provides async-safe CRUD operations for session lifecycle.
"""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class SessionEntry:
    """Per-session metadata and pipeline reference.

    Fields:
        session_id: Unique identifier for the WebSocket session (UUID)
        connected_at: Timestamp when the session was established
        last_activity: Timestamp of the most recent activity on the session
        has_voice_profile: Whether a voice sample has been uploaded for this session
        tts_pipeline: TTSPipeline instance for this session (typed as object to avoid circular import)
        owner: Name of the user who owns the bearer token that opened the session.
            None for an entry registered without one, which is not scoped to a user.
    """
    session_id: str
    connected_at: datetime
    last_activity: datetime
    has_voice_profile: bool = False
    tts_pipeline: object | None = None
    owner: str | None = None

    def visible_to(self, user: str) -> bool:
        """Whether ``user`` may see and use this session: only its owner may.

        A session with no owner is visible to nobody, so a missing owner can
        never widen access.
        """
        return self.owner is not None and self.owner == user


class SessionRegistry:
    """Async-safe registry for active WebSocket sessions.

    Thread safety is achieved via asyncio.Lock — all operations acquire the lock
    before mutating or reading session state (single event loop assumption).
    """

    def __init__(self):
        self._sessions: dict[str, SessionEntry] = {}
        self._lock = asyncio.Lock()

    async def register(self, entry: SessionEntry) -> None:
        """Add a SessionEntry to the registry keyed by session_id."""
        async with self._lock:
            self._sessions[entry.session_id] = entry

    async def remove(self, session_id: str) -> None:
        """Remove a session by ID. No-op if session_id does not exist."""
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def get(self, session_id: str) -> SessionEntry | None:
        """Retrieve a session by ID. Returns None if not found."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def update_activity(self, session_id: str) -> None:
        """Update last_activity timestamp on an existing session. No-op if not found."""
        async with self._lock:
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry.last_activity = datetime.now(timezone.utc)

    async def set_voice_profile(self, session_id: str, has_profile: bool) -> None:
        """Set has_voice_profile flag on an existing session. No-op if not found."""
        async with self._lock:
            entry = self._sessions.get(session_id)
            if entry is not None:
                entry.has_voice_profile = has_profile

    async def snapshot(self, owner: str | None = None) -> list[dict]:
        """Return a list of session metadata dicts (no TTSPipeline references).

        Each dict contains: session_id, connected_at (ISO 8601), last_activity (ISO 8601),
        has_voice_profile. With ``owner``, only the sessions visible to that user
        (see SessionEntry.visible_to); without it, every session.
        """
        async with self._lock:
            return [
                {
                    "session_id": entry.session_id,
                    "connected_at": entry.connected_at.isoformat(),
                    "last_activity": entry.last_activity.isoformat(),
                    "has_voice_profile": entry.has_voice_profile,
                }
                for entry in self._sessions.values()
                if owner is None or entry.visible_to(owner)
            ]
