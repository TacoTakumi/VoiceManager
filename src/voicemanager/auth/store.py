"""The auth store: users, tokens, sessions and sign-in failures.

All of it lives in one SQLite file, by default ``state.db`` in the XDG state
dir (VM_DB_PATH overrides it; see :func:`voicemanager.config.resolve_db_path`).
No SQL on these tables lives anywhere but in this module.

Every token belongs to a user and may name a preset voice, empty for the
server default. Removing a user drops their tokens, sessions and failed
sign-ins in the same transaction as the user row.

Standard library only. A store is opened per unit of work and closed after
it: a SQLite connection belongs to the thread that opened it, and the server
answers requests on several.
"""
from __future__ import annotations

import dataclasses
import logging
import sqlite3
from pathlib import Path

from voicemanager.config import Settings, get_settings, resolve_db_path

logger = logging.getLogger(__name__)

STORE_FILENAME = "state.db"

# The ways a session may have been opened.
SESSION_METHODS = ("password",)


@dataclasses.dataclass(frozen=True)
class User:
    """One named user; a None ``password_hash`` means no password is set."""

    name: str
    password_hash: str | None
    created: str


@dataclasses.dataclass(frozen=True)
class Token:
    """One bearer token as the store holds it: never the secret, only its hash.

    ``name`` is the token's label, unique per owner; ``last4`` is the tail of
    the secret, enough to tell two tokens apart in a listing; ``voice`` is
    the preset voice the token speaks with, empty for the server default.
    """

    owner: str
    name: str
    hash: str
    last4: str | None
    created: str
    voice: str = ""
    last_used: str | None = None


@dataclasses.dataclass(frozen=True)
class Session:
    """One browser session, keyed by the hash of its cookie secret.

    ``expires`` and ``written`` are epoch seconds; ``written`` is when the
    row was last written, so a caller can skip a write that would change
    little.
    """

    hash: str
    user: str
    method: str
    expires: float
    written: float


class Conflict(Exception):
    """A write that would duplicate a key the store keeps unique."""


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    name          TEXT PRIMARY KEY,
    password_hash TEXT,
    created       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tokens (
    hash      TEXT PRIMARY KEY,
    owner     TEXT NOT NULL REFERENCES users(name) ON DELETE CASCADE,
    name      TEXT NOT NULL,
    last4     TEXT,
    voice     TEXT NOT NULL DEFAULT '',
    created   TEXT NOT NULL,
    last_used TEXT,
    UNIQUE (owner, name)
);
CREATE TABLE IF NOT EXISTS sessions (
    hash    TEXT PRIMARY KEY,
    user    TEXT NOT NULL REFERENCES users(name) ON DELETE CASCADE,
    method  TEXT NOT NULL,
    expires REAL NOT NULL,
    written REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_by_user ON sessions (user);
CREATE TABLE IF NOT EXISTS signin_failures (
    user TEXT NOT NULL REFERENCES users(name) ON DELETE CASCADE,
    at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS signin_failures_by_user ON signin_failures (user, at);
"""

# Columns added after their table's first shape, as (table, column, type):
# CREATE TABLE IF NOT EXISTS leaves an existing table alone, so a store
# opened from before the column gains it here, with its rows kept. A NOT NULL
# column needs a default so the rows already there have a value.
ADDED_COLUMNS = (("tokens", "voice", "TEXT NOT NULL DEFAULT ''"),)

_USER_COLUMNS = "name, password_hash, created"
_TOKEN_COLUMNS = "owner, name, hash, last4, voice, created, last_used"
_SESSION_COLUMNS = "hash, user, method, expires, written"


def store_path(settings: Settings | None = None) -> Path:
    """Where the store file lives, from VM_DB_PATH or the XDG state dir."""
    if settings is None:
        settings = get_settings()
    return resolve_db_path(settings.db_path)


class SqliteStore:
    """The auth store on a SQLite file; the schema is created on open."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        with self.connection:
            self.connection.executescript(SCHEMA)
            for table, column, kind in ADDED_COLUMNS:
                present = {
                    row["name"]
                    for row in self.connection.execute(f"PRAGMA table_info({table})")
                }
                if column not in present:
                    logger.info("Adding column %s.%s to the auth store", table, column)
                    self.connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {kind}"
                    )

    # --- users -------------------------------------------------------------

    def add_user(self, user: User) -> None:
        """Insert ``user``; raises :class:`Conflict` on a taken name."""
        try:
            with self.connection:
                self.connection.execute(
                    f"INSERT INTO users ({_USER_COLUMNS}) VALUES (?, ?, ?)",
                    (user.name, user.password_hash, user.created),
                )
        except sqlite3.IntegrityError as exc:
            raise Conflict(user.name) from exc

    def get_user(self, name: str) -> User | None:
        """The user called ``name``, or None."""
        row = self.connection.execute(
            f"SELECT {_USER_COLUMNS} FROM users WHERE name = ?", (name,)
        ).fetchone()
        return _user(row) if row is not None else None

    def list_users(self) -> list[User]:
        """Every user, by name."""
        rows = self.connection.execute(
            f"SELECT {_USER_COLUMNS} FROM users ORDER BY name"
        ).fetchall()
        return [_user(row) for row in rows]

    def set_user_password(self, name: str, password_hash: str | None) -> User | None:
        """Set or clear a user's password hash; None when there is no such user."""
        with self.connection:
            changed = self.connection.execute(
                "UPDATE users SET password_hash = ? WHERE name = ?", (password_hash, name)
            ).rowcount
        return self.get_user(name) if changed else None

    def remove_user(self, name: str) -> bool:
        """Remove a user and everything they hold; False when there was none."""
        # One transaction: a user is never left half removed, with their
        # row gone and a token still able to act for them, or the reverse.
        with self.connection:
            self.connection.execute("DELETE FROM tokens WHERE owner = ?", (name,))
            self.connection.execute("DELETE FROM sessions WHERE user = ?", (name,))
            self.connection.execute("DELETE FROM signin_failures WHERE user = ?", (name,))
            changed = self.connection.execute(
                "DELETE FROM users WHERE name = ?", (name,)
            ).rowcount
        return bool(changed)

    # --- tokens ------------------------------------------------------------

    def add_token(self, token: Token) -> None:
        """Insert ``token``; :class:`Conflict` on a taken (owner, name) or hash."""
        try:
            with self.connection:
                self.connection.execute(
                    f"INSERT INTO tokens ({_TOKEN_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        token.owner, token.name, token.hash, token.last4,
                        token.voice, token.created, token.last_used,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            if "FOREIGN KEY" in str(exc):
                raise ValueError(f"No user {token.owner!r} to own a token.") from exc
            raise Conflict(f"{token.owner}/{token.name}") from exc

    def list_tokens(self, owner: str | None = None) -> list[Token]:
        """Every token, or only ``owner``'s, by owner then name."""
        where, values = (" WHERE owner = ?", (owner,)) if owner is not None else ("", ())
        rows = self.connection.execute(
            f"SELECT {_TOKEN_COLUMNS} FROM tokens{where} ORDER BY owner, name", values
        ).fetchall()
        return [_token(row) for row in rows]

    def find_token(self, token_hash: str) -> Token | None:
        """The token whose secret hashes to ``token_hash``, or None."""
        row = self.connection.execute(
            f"SELECT {_TOKEN_COLUMNS} FROM tokens WHERE hash = ?", (token_hash,)
        ).fetchone()
        return _token(row) if row is not None else None

    def revoke_token(self, owner: str, name: str) -> bool:
        """Delete ``owner``'s token ``name``; False when there was none."""
        with self.connection:
            changed = self.connection.execute(
                "DELETE FROM tokens WHERE owner = ? AND name = ?", (owner, name)
            ).rowcount
        return bool(changed)

    def set_token_voice(self, owner: str, name: str, voice: str) -> bool:
        """Set ``owner``'s token ``name`` to speak with ``voice`` (empty: the
        server default); False when there is no such token."""
        with self.connection:
            changed = self.connection.execute(
                "UPDATE tokens SET voice = ? WHERE owner = ? AND name = ?",
                (voice, owner, name),
            ).rowcount
        return bool(changed)

    def touch_token(self, token_hash: str, when: str) -> bool:
        """Record a token's last use; False when there is no such token."""
        with self.connection:
            changed = self.connection.execute(
                "UPDATE tokens SET last_used = ? WHERE hash = ?", (when, token_hash)
            ).rowcount
        return bool(changed)

    # --- sessions ----------------------------------------------------------

    def add_session(self, session: Session) -> None:
        """Insert ``session``; ValueError on a bad method or no such user."""
        if session.method not in SESSION_METHODS:
            raise ValueError(
                f"Unknown session method {session.method!r}: expected one of "
                + ", ".join(SESSION_METHODS) + "."
            )
        try:
            with self.connection:
                self.connection.execute(
                    f"INSERT INTO sessions ({_SESSION_COLUMNS}) VALUES (?, ?, ?, ?, ?)",
                    (session.hash, session.user, session.method, session.expires,
                     session.written),
                )
        except sqlite3.IntegrityError as exc:
            # The user column refers to a user row; a session for no user is
            # a caller's mistake, not a duplicate.
            if "FOREIGN KEY" in str(exc):
                raise ValueError(f"No user {session.user!r} to open a session for.") from exc
            raise Conflict(session.hash) from exc

    def get_session(self, session_hash: str) -> Session | None:
        """The session keyed by ``session_hash``, or None."""
        row = self.connection.execute(
            f"SELECT {_SESSION_COLUMNS} FROM sessions WHERE hash = ?", (session_hash,)
        ).fetchone()
        return _session(row) if row is not None else None

    def update_session(
        self, session_hash: str, *, expires: float, written: float
    ) -> Session | None:
        """Move a session's expiry; None when there is no such session."""
        with self.connection:
            changed = self.connection.execute(
                "UPDATE sessions SET expires = ?, written = ? WHERE hash = ?",
                (expires, written, session_hash),
            ).rowcount
        return self.get_session(session_hash) if changed else None

    def remove_session(self, session_hash: str) -> bool:
        """Delete one session; False when there was none."""
        with self.connection:
            changed = self.connection.execute(
                "DELETE FROM sessions WHERE hash = ?", (session_hash,)
            ).rowcount
        return bool(changed)

    def remove_user_sessions(self, user: str, *, keep: str | None = None) -> int:
        """Delete ``user``'s sessions except ``keep``; returns how many went."""
        with self.connection:
            return self.connection.execute(
                "DELETE FROM sessions WHERE user = ? AND hash IS NOT ?", (user, keep)
            ).rowcount

    def remove_expired_sessions(self, now: float) -> int:
        """Delete every session expired at ``now``; returns how many went."""
        with self.connection:
            return self.connection.execute(
                "DELETE FROM sessions WHERE expires <= ?", (now,)
            ).rowcount

    # --- sign-in failures --------------------------------------------------

    def add_signin_failure(self, user: str, at: float) -> None:
        """Record a failed sign-in for ``user`` at epoch ``at``."""
        try:
            with self.connection:
                self.connection.execute(
                    "INSERT INTO signin_failures (user, at) VALUES (?, ?)", (user, at)
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"No user {user!r} to record a failed sign-in for.") from exc

    def list_signin_failures(self, user: str) -> list[float]:
        """``user``'s failed sign-in times, oldest first."""
        rows = self.connection.execute(
            "SELECT at FROM signin_failures WHERE user = ? ORDER BY at", (user,)
        ).fetchall()
        return [row["at"] for row in rows]

    def clear_signin_failures(self, user: str, *, before: float | None = None) -> int:
        """Delete ``user``'s failures, or only those before ``before``; returns the count."""
        with self.connection:
            if before is None:
                return self.connection.execute(
                    "DELETE FROM signin_failures WHERE user = ?", (user,)
                ).rowcount
            return self.connection.execute(
                "DELETE FROM signin_failures WHERE user = ? AND at < ?", (user, before)
            ).rowcount

    # --- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Close the connection."""
        self.connection.close()

    def __enter__(self) -> SqliteStore:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _user(row: sqlite3.Row) -> User:
    return User(**{column: row[column] for column in _USER_COLUMNS.split(", ")})


def _token(row: sqlite3.Row) -> Token:
    return Token(**{column: row[column] for column in _TOKEN_COLUMNS.split(", ")})


def _session(row: sqlite3.Row) -> Session:
    return Session(**{column: row[column] for column in _SESSION_COLUMNS.split(", ")})


def open_store(path: Path | None = None, *, settings: Settings | None = None) -> SqliteStore:
    """Open the store for one unit of work.

    ``path`` defaults to :func:`store_path` for ``settings`` (the loaded
    settings when that is None too).
    """
    return SqliteStore(Path(path) if path is not None else store_path(settings))
