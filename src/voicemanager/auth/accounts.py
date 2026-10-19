"""Account rules for VoiceManager: users, passwords, sessions and tokens.

The functions that change accounts take an open store (see
:mod:`voicemanager.auth.store`), so the CLI and the server share one set of
rules: a user name is checked and lowercased here, a password change ends
the user's sessions here, and a token's secret is made and hashed here. A
refusal is an :class:`AccountError` whose message is fit to show the
operator.

A password is stored as ``scrypt$N$r$p$salt$hash`` (salt and hash in hex),
so each stored string names the parameters it was made with: verifying uses
those parameters, and :func:`needs_rehash` tells when they are older than
today's. Verification compares in constant time, and a stored string that
asks scrypt for more than :data:`SCRYPT_MAX_MEMORY` verifies false instead
of allocating it.

A browser session and a bearer token are both random secrets of which the
store keeps only the SHA-256 (:func:`hash_secret`), so a copy of the store
hands out neither. A session expires a while after its last use; a token
lives until it is revoked, belongs to one user, and may name the preset
voice it speaks with. Both write their last use at most once a minute.

Every function that reads a clock takes ``now``, so a test can drive time;
left out, it is the wall clock. Secrets and passwords are never logged.
Stdlib only.
"""
from __future__ import annotations

import contextlib
import dataclasses
import datetime
import functools
import hashlib
import hmac
import logging
import re
import secrets
import threading
import time
from collections.abc import Iterator

from voicemanager.auth.store import Conflict, Session, SqliteStore, Token, User

logger = logging.getLogger(__name__)

SCRYPT_N = 16384
SCRYPT_R = 8
SCRYPT_P = 5
SCRYPT_SALT_BYTES = 16
SCRYPT_KEY_BYTES = 32

# The most memory a stored hash may make scrypt use. A stored string is
# data; one that names a huge N must not be able to exhaust the server.
SCRYPT_MAX_MEMORY = 256 * 1024 * 1024

# A user name: 1 to 64 characters from a-z 0-9 . _ - @, starting with a
# letter or digit. Short and plain, so it is safe in a URL, a log line and a
# listing without quoting.
USER_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._@-]{0,63}")
USER_NAME_RULE = (
    "a user name is 1 to 64 characters from a-z 0-9 . _ - @,"
    " starting with a letter or digit"
)


class AccountError(ValueError):
    """An account change refused: a bad name, a taken or unknown user or token."""


# --- users -------------------------------------------------------------------

def normalise_user_name(name: str) -> str:
    """``name`` lowercased, or an :class:`AccountError` naming the rule it breaks.

    The check runs on the lowercased name, so 'Ann' is kept as 'ann'.
    """
    lowered = name.lower() if isinstance(name, str) else ""
    if not USER_NAME_PATTERN.fullmatch(lowered):
        raise AccountError(f"Bad user name {name!r}: {USER_NAME_RULE}.")
    return lowered


def check_new_user(store: SqliteStore, name: str) -> str:
    """The name a new user would be stored under, or a refusal.

    Refuses a bad name and a name already taken, so a caller can check
    before it asks for a password.
    """
    name = normalise_user_name(name)
    if store.get_user(name) is not None:
        raise AccountError(f"A user named {name!r} already exists.")
    return name


def add_user(
    store: SqliteStore, name: str, password: str | None, *, now: str | None = None
) -> User:
    """Add the user ``name``; the user as stored.

    The name is stored lowercase. With ``password`` only its scrypt hash is
    kept; with None the user has no password, so cannot sign in to the web
    pages, but can still own tokens. An empty password is refused, as is
    anything :func:`check_new_user` refuses.
    """
    name = check_new_user(store, name)
    _refuse_empty(password)
    user = User(
        name=name,
        password_hash=hash_password(password) if password is not None else None,
        created=now or _now(),
    )
    try:
        store.add_user(user)
    except Conflict:
        raise AccountError(f"A user named {name!r} already exists.") from None
    logger.info("Added user %s", name)
    return user


def require_user(store: SqliteStore, name: str) -> User:
    """The user called ``name``, in any case, or an :class:`AccountError`."""
    user = store.get_user(name.lower())
    if user is None:
        raise AccountError(f"No user named {name.lower()!r}.")
    return user


def set_password(store: SqliteStore, name: str, password: str | None) -> User:
    """Replace the password of ``name``, or clear it with None; the updated user.

    Every session of the user ends with it, so a browser signed in under
    the old password must sign in again; the user's failed sign-ins go too,
    so a lockout ends at once. An empty password and an unknown user are
    refused.
    """
    _refuse_empty(password)
    name = require_user(store, name).name
    password_hash = hash_password(password) if password is not None else None
    # The password first, then the sessions: a session opened under the old
    # password just before the change is still among those dropped.
    user = store.set_user_password(name, password_hash)
    store.remove_user_sessions(name)
    store.clear_signin_failures(name)
    if user is None:
        raise AccountError(f"No user named {name!r}.")
    logger.info("%s the password of user %s", "Set" if password else "Cleared", name)
    return user


# What a change of one's own password says when it is refused.
WRONG_PASSWORD = "That is not your current password."
PASSWORD_EMPTY = "The new password may not be empty."


def change_password(
    store: SqliteStore,
    name: str,
    current: str,
    new: str,
    *,
    keep_session_hash: str | None,
) -> str | None:
    """Replace the password of ``name`` when ``current`` is theirs; the refusal, or None.

    The current password is checked against the stored hash directly, not
    through :func:`check_password`: a wrong one here is no failed sign-in,
    so it neither counts toward the lockout nor is blocked by one. A user
    with no password has no current password to give. On a change, the
    session whose hash is ``keep_session_hash`` (the one that made it) stays
    and every other session of the user ends (all of them when it is None);
    their failed sign-ins go too, as with a password the operator sets. A
    refusal changes nothing.
    """
    if not new:
        return PASSWORD_EMPTY
    user = store.get_user(name.lower()) if isinstance(name, str) else None
    if (user is None or user.password_hash is None
            or not verify_password(current, user.password_hash)):
        return WRONG_PASSWORD
    new_hash = hash_password(new)
    # The password first, then the sessions: a session opened under the old
    # password just before the change is still among those dropped.
    store.set_user_password(user.name, new_hash)
    ended = store.remove_user_sessions(user.name, keep=keep_session_hash)
    store.clear_signin_failures(user.name)
    logger.info("User %s changed their password; ended %d other session(s)", user.name, ended)
    return None


def remove_user(store: SqliteStore, name: str) -> str:
    """Remove ``name`` with their tokens, sessions and failures; the name removed."""
    name = name.lower()
    if not store.remove_user(name):
        raise AccountError(f"No user named {name!r}.")
    logger.info("Removed user %s", name)
    return name


def _refuse_empty(password: str | None) -> None:
    if password is not None and not password:
        raise AccountError("A password may not be empty.")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(_TIME_FORMAT)


# --- passwords ---------------------------------------------------------------

def hash_password(
    password: str, *, n: int = SCRYPT_N, r: int = SCRYPT_R, p: int = SCRYPT_P
) -> str:
    """The stored form of ``password``: a scrypt hash under a fresh random salt.

    ``n``, ``r`` and ``p`` default to today's parameters; tests pass other
    values to make hashes that :func:`needs_rehash` reports as stale.
    """
    salt = secrets.token_bytes(SCRYPT_SALT_BYTES)
    digest = _scrypt(password, salt, n, r, p, SCRYPT_KEY_BYTES)
    return f"scrypt${n}${r}${p}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    """Whether ``password`` hashes to ``stored``, under the parameters ``stored`` names.

    A malformed stored string, or one asking for more memory than
    :data:`SCRYPT_MAX_MEMORY`, verifies false rather than raising.
    """
    parsed = _parse(stored)
    if parsed is None:
        return False
    n, r, p, salt, expected = parsed
    try:
        digest = _scrypt(password, salt, n, r, p, len(expected))
    except (ValueError, OverflowError, MemoryError):
        return False
    return hmac.compare_digest(digest, expected)


def needs_rehash(stored: str | None) -> bool:
    """Whether ``stored`` was made under other parameters than today's, or is unreadable."""
    parsed = _parse(stored)
    if parsed is None:
        return True
    n, r, p, _salt, expected = parsed
    return (n, r, p, len(expected)) != (SCRYPT_N, SCRYPT_R, SCRYPT_P, SCRYPT_KEY_BYTES)


def authenticate_password(store: SqliteStore, name: str, password: str) -> User | None:
    """The user ``name`` (in any case) when ``password`` is theirs, else None.

    An unknown name, a user with no password and a wrong password all
    answer None, and each costs one scrypt run, so neither the answer nor
    how long it takes tells which. A right password over a hash made under
    older parameters replaces that hash with one under today's; the user's
    sessions stay open, as the password itself did not change.
    """
    user = store.get_user(name.lower()) if isinstance(name, str) and name else None
    if user is None or user.password_hash is None:
        verify_password(password, _decoy_hash())
        return None
    if not verify_password(password, user.password_hash):
        return None
    if needs_rehash(user.password_hash):
        user = store.set_user_password(user.name, hash_password(password)) or user
        logger.info("Rehashed the password of user %s under today's parameters", user.name)
    return user


@functools.cache
def _decoy_hash() -> str:
    # What a sign-in with no stored hash checks against, so it takes as long
    # as one with a hash; no password is ever its own.
    return hash_password(secrets.token_urlsafe(32))


def _scrypt(password: str, salt: bytes, n: int, r: int, p: int, dklen: int) -> bytes:
    # scrypt needs about 128 * r * (N + p) bytes; past the cap, refuse
    # before asking the library for it.
    needed = 128 * r * (n + p + 2)
    if needed > SCRYPT_MAX_MEMORY:
        raise ValueError("scrypt parameters ask for too much memory")
    return hashlib.scrypt(
        password.encode(), salt=salt, n=n, r=r, p=p, dklen=dklen, maxmem=needed + 1024 * 1024
    )


def _parse(stored: str | None) -> tuple[int, int, int, bytes, bytes] | None:
    """The parameters, salt and hash a stored string names, or None if it is malformed."""
    parts = stored.split("$") if isinstance(stored, str) else []
    if len(parts) != 6 or parts[0] != "scrypt":
        return None
    try:
        n, r, p = (int(part) for part in parts[1:4])
        salt, digest = bytes.fromhex(parts[4]), bytes.fromhex(parts[5])
    except ValueError:
        return None
    if n < 2 or n & (n - 1) or r < 1 or p < 1 or not salt or not digest:
        return None
    return n, r, p, salt, digest


# --- password sign-in lockout ------------------------------------------------
#
# SIGNIN_MAX_FAILURES wrong passwords for one user within SIGNIN_WINDOW
# seconds lock that user's password sign-in for SIGNIN_LOCKOUT seconds after
# the last of them, even for the right password. The failures live in the
# store, so a restart does not lift a lockout. A right password, or a new
# one set by the operator or by the user, clears them. Only check_password
# reads the lockout: a token never does.

SIGNIN_MAX_FAILURES = 5
SIGNIN_WINDOW = 15 * 60
SIGNIN_LOCKOUT = 15 * 60
SIGNIN_LOCKED = (
    "Too many failed sign-ins for this user. Password sign-in is locked for"
    f" {SIGNIN_LOCKOUT // 60} minutes. Try again later."
)


def check_password(
    store: SqliteStore, name: str, password: str, *, now: float | None = None
) -> tuple[User | None, bool]:
    """``(user, locked)`` for a password sign-in as ``name`` (in any case).

    ``user`` is as :func:`authenticate_password` answers. ``locked`` is True
    while the user is locked out: the password is then not checked and
    nothing is recorded, so the right password is refused too. A wrong
    password for a user who has one is recorded at ``now`` (epoch seconds),
    and the failure that closes a run of :data:`SIGNIN_MAX_FAILURES` answers
    locked as well. A right password clears the user's failures. An unknown
    name records nothing, as it has no row to record against.
    """
    now = time.time() if now is None else now
    # One name's attempts run one at a time, from the lockout check to the
    # failure record, so a burst of wrong passwords cannot all be checked
    # before the first is counted. Other names do not wait.
    with _signin_lock(name if isinstance(name, str) else ""):
        user = store.get_user(name.lower()) if isinstance(name, str) and name else None
        if user is not None and _locked(store, user.name, now):
            return None, True
        signed_in = authenticate_password(store, name, password)
        if signed_in is not None:
            store.clear_signin_failures(signed_in.name)
            return signed_in, False
        if user is None or user.password_hash is None:
            return None, False
        # Failures older than the window can never count again: each check
        # measures back from a failure no older than this one.
        store.clear_signin_failures(user.name, before=now - SIGNIN_WINDOW)
        try:
            store.add_signin_failure(user.name, now)
        except ValueError:
            # Removed between the read and the write.
            return None, False
        locked = _locked(store, user.name, now)
        if locked:
            logger.warning(
                "Password sign-in for user %s is locked after %d failures",
                user.name, SIGNIN_MAX_FAILURES,
            )
        return None, locked


def _locked(store: SqliteStore, user: str, now: float) -> bool:
    """Whether password sign-in for ``user`` is refused at ``now``.

    It is while the last failure is under :data:`SIGNIN_LOCKOUT` old and
    closes a run of :data:`SIGNIN_MAX_FAILURES` within :data:`SIGNIN_WINDOW`.
    Nothing is recorded while a lockout holds, so the last failure is the
    one that locked.
    """
    failures = store.list_signin_failures(user)
    if not failures or now >= failures[-1] + SIGNIN_LOCKOUT:
        return False
    last = failures[-1]
    recent = sum(1 for at in failures if at > last - SIGNIN_WINDOW)
    return recent >= SIGNIN_MAX_FAILURES


# Each name with a sign-in running or waiting: its lock and how many hold or
# wait on it. The guard covers the dict and the counts.
_signin_locks: dict[str, list] = {}
_signin_locks_guard = threading.Lock()


@contextlib.contextmanager
def _signin_lock(name: str) -> Iterator[None]:
    """Run the password sign-ins for ``name`` (in any case) one at a time.

    The server is one process, so a lock per name is enough. A name's lock
    lives only while a sign-in for it holds or waits on it, so names tried
    once, known or not, leave nothing behind.
    """
    key = name.lower()
    with _signin_locks_guard:
        entry = _signin_locks.setdefault(key, [threading.Lock(), 0])
        entry[1] += 1
    try:
        with entry[0]:
            yield
    finally:
        with _signin_locks_guard:
            entry[1] -= 1
            if not entry[1]:
                del _signin_locks[key]


# --- browser sessions --------------------------------------------------------
#
# A session is a random cookie secret. The store keeps only its SHA-256, so a
# copy of the store does not hand out live sessions. A session expires
# SESSION_TTL after its last use; a use moves the expiry, but the row (and
# so the cookie) is rewritten at most once per SESSION_WRITE_EVERY, so a
# page that fires many requests a second writes once.

SESSION_TTL = 12 * 3600
SESSION_WRITE_EVERY = 60


def hash_secret(secret: str) -> str:
    """The stored form of a session or token secret: its SHA-256 hex digest."""
    return hashlib.sha256(secret.encode()).hexdigest()


def open_session(
    store: SqliteStore, user: str, method: str = "password", *, now: float | None = None
) -> str:
    """Start a session for ``user`` and return its cookie secret; the only copy.

    The store keeps the secret's hash, the user, how the session was opened
    (one of the store's ``SESSION_METHODS``) and an expiry
    :data:`SESSION_TTL` after ``now``. Sessions already expired are swept
    on the way. ValueError on an unknown method or user.
    """
    now = time.time() if now is None else now
    secret = secrets.token_urlsafe(32)
    store.remove_expired_sessions(now)
    store.add_session(Session(
        hash=hash_secret(secret), user=user, method=method,
        expires=now + SESSION_TTL, written=now,
    ))
    logger.info("Opened a %s session for user %s", method, user)
    return secret


def resume_session(
    store: SqliteStore, secret: str, *, now: float | None = None
) -> tuple[Session | None, bool]:
    """The live session ``secret`` opens, and whether its expiry was written.

    ``(None, False)`` for an empty or unknown secret, or a session whose
    expiry is at or before ``now``; an expired session's row is deleted on
    sight. A live session's expiry moves to :data:`SESSION_TTL` after
    ``now``, but only once :data:`SESSION_WRITE_EVERY` has passed since its
    row was last written; the bool is True when that write happened, which
    tells the caller to refresh the cookie.
    """
    if not secret:
        return None, False
    now = time.time() if now is None else now
    digest = hash_secret(secret)
    # The lookup is by hash: equal hashes mean equal secrets, and the
    # secret itself is never compared, so there is no timing to leak.
    session = store.get_session(digest)
    if session is None:
        return None, False
    if session.expires <= now:
        store.remove_session(digest)
        return None, False
    if now - session.written < SESSION_WRITE_EVERY:
        return session, False
    updated = store.update_session(digest, expires=now + SESSION_TTL, written=now)
    if updated is None:
        # Ended between the read and the write.
        return None, False
    return updated, True


def end_session(store: SqliteStore, secret: str) -> bool:
    """End the session ``secret`` opens; False when there was none."""
    if not secret:
        return False
    return store.remove_session(hash_secret(secret))


# --- bearer tokens -----------------------------------------------------------
#
# A token is a random secret a client sends as a bearer token. It acts for
# its owner and may name the preset voice it speaks with (empty: the server
# default). The store keeps the secret's SHA-256 and its last 4 characters
# (to tell tokens apart in a listing), never the secret. Tokens do not
# expire. A use is written as the token's last use at most once per
# TOKEN_WRITE_EVERY, so a busy client does not write on every request.

TOKEN_PREFIX = "vmp_"
TOKEN_WRITE_EVERY = 60

_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def mint_token(
    store: SqliteStore,
    owner: str,
    name: str,
    *,
    voice: str | None = None,
    now: str | None = None,
) -> str:
    """Make the token ``name`` for ``owner`` and return its secret; the only copy.

    The secret is :data:`TOKEN_PREFIX` and 43 URL-safe characters. The store
    keeps its SHA-256, its last 4 characters, ``voice`` (None or empty: the
    server default) and the time made. An unknown owner, an empty name and
    a name the owner already uses are refused. Whether ``voice`` names a
    voice the server has is for the caller to check: the store does not
    know the engine.
    """
    owner = require_user(store, owner).name
    if not name or not name.strip():
        raise AccountError("A token needs a name.")
    secret = TOKEN_PREFIX + secrets.token_urlsafe(32)
    token = Token(
        owner=owner,
        name=name,
        hash=hash_secret(secret),
        last4=secret[-4:],
        voice=(voice or "").strip(),
        created=now or _now(),
    )
    try:
        store.add_token(token)
    except Conflict:
        raise AccountError(f"{owner} already has a token named {name!r}.") from None
    logger.info("Made token %r for user %s", name, owner)
    return secret


def list_tokens(store: SqliteStore, owner: str) -> list[Token]:
    """The tokens of ``owner`` (any case), by name; an unknown user is refused."""
    owner = require_user(store, owner).name
    return store.list_tokens(owner)


def revoke_token(store: SqliteStore, owner: str, name: str) -> bool:
    """Delete ``owner``'s token ``name``; True, or an :class:`AccountError` when missing."""
    owner = require_user(store, owner).name
    if not store.revoke_token(owner, name):
        raise AccountError(f"{owner} has no token named {name!r}.")
    logger.info("Revoked token %r of user %s", name, owner)
    return True


def set_token_voice(store: SqliteStore, owner: str, name: str, voice: str | None) -> str:
    """Set the voice of ``owner``'s token ``name`` and return it as stored.

    None or empty clears it, so the token speaks with the server default. An
    unknown owner or token is refused. As with :func:`mint_token`, whether
    ``voice`` names a voice the server has is for the caller to check.
    """
    owner = require_user(store, owner).name
    voice = (voice or "").strip()
    if not store.set_token_voice(owner, name, voice):
        raise AccountError(f"{owner} has no token named {name!r}.")
    logger.info("Set the voice of token %r of user %s to %r", name, owner, voice)
    return voice


def token_for(
    store: SqliteStore, secret: str, *, now: float | None = None
) -> Token | None:
    """The token ``secret`` is, or None for an empty or unknown secret.

    The lookup is by the secret's SHA-256, so the secret itself is never
    compared. The token's last use is written as ``now`` (epoch seconds)
    when it has none yet or the one stored is at least
    :data:`TOKEN_WRITE_EVERY` old; the token returned carries what is
    stored after that.
    """
    if not secret:
        return None
    now = time.time() if now is None else now
    digest = hash_secret(secret)
    token = store.find_token(digest)
    if token is None:
        return None
    if _token_due(token.last_used, now):
        when = _format_time(now)
        if store.touch_token(digest, when):
            token = dataclasses.replace(token, last_used=when)
    return token


def _token_due(last_used: str | None, now: float) -> bool:
    """Whether a token last used at ``last_used`` should have ``now`` written."""
    if last_used is None:
        return True
    try:
        then = datetime.datetime.strptime(last_used, _TIME_FORMAT).replace(
            tzinfo=datetime.timezone.utc
        ).timestamp()
    except ValueError:
        return True
    return now - then >= TOKEN_WRITE_EVERY


def _format_time(epoch: float) -> str:
    """``epoch`` in the store's UTC time format, to the second."""
    return datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).strftime(_TIME_FORMAT)
