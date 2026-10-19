"""Tests for the account rules: users, passwords, tokens, sessions and lockout.

Model-free and server-free: every store is a SQLite file under pytest's
tmp_path, and every clock is passed in as ``now`` (or patched on the
module's ``time``), so no test waits on the wall clock.
"""
import hashlib
import logging
import re
import secrets
import sqlite3
import threading

import pytest

from voicemanager.auth import accounts
from voicemanager.auth.accounts import (
    PASSWORD_EMPTY,
    SESSION_TTL,
    SESSION_WRITE_EVERY,
    SIGNIN_LOCKOUT,
    SIGNIN_MAX_FAILURES,
    SIGNIN_WINDOW,
    TOKEN_PREFIX,
    TOKEN_WRITE_EVERY,
    WRONG_PASSWORD,
    AccountError,
    add_user,
    authenticate_password,
    change_password,
    check_new_user,
    check_password,
    end_session,
    hash_password,
    hash_secret,
    list_tokens,
    mint_token,
    needs_rehash,
    normalise_user_name,
    open_session,
    remove_user,
    require_user,
    resume_session,
    revoke_token,
    set_password,
    token_for,
    verify_password,
)
from voicemanager.auth.store import Session, User, open_store

T0 = 1_800_000_000.0
MINUTE = 60.0
HOUR = 3600.0
PASSWORD = "correct horse battery"
WRONG = "not the password"
NEW_PASSWORD = "a brand new password"


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "state.db"


@pytest.fixture
def store(db_path):
    """A store on a temp file, holding the user 'ann' with no password."""
    with open_store(db_path) as opened:
        opened.add_user(User(name="ann", password_hash=None, created="2026-01-01T00:00:00Z"))
        yield opened


@pytest.fixture
def pw_store(db_path):
    """A store on a temp file, holding the user 'ann' with PASSWORD."""
    with open_store(db_path) as opened:
        add_user(opened, "ann", PASSWORD)
        yield opened


def _rows(store, table):
    return store.connection.execute(f"SELECT * FROM {table}").fetchall()


def _every_value(db_path):
    """Every value of every column of every table in the store file, as text."""
    connection = sqlite3.connect(db_path)
    try:
        tables = [row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )]
        values = []
        for table in tables:
            for row in connection.execute(f"SELECT * FROM {table}"):
                values.extend(str(value) for value in row)
        return values
    finally:
        connection.close()


def _raw_bytes(db_path):
    raw = db_path.read_bytes()
    for suffix in ("-wal", "-journal"):
        side = db_path.with_name(db_path.name + suffix)
        if side.exists():
            raw += side.read_bytes()
    return raw


# --- passwords ---------------------------------------------------------------

def test_hash_names_current_parameters():
    stored = hash_password("correct horse")
    parts = stored.split("$")
    assert len(parts) == 6
    assert parts[:4] == ["scrypt", "16384", "8", "5"]
    assert len(bytes.fromhex(parts[4])) == 16
    assert len(bytes.fromhex(parts[5])) == 32


def test_salt_is_fresh_each_time():
    assert hash_password("same") != hash_password("same")


def test_right_password_verifies():
    assert verify_password("correct horse", hash_password("correct horse")) is True


def test_wrong_password_does_not_verify():
    assert verify_password("battery staple", hash_password("correct horse")) is False


def test_current_hash_needs_no_rehash():
    assert needs_rehash(hash_password("pw")) is False


def test_weaker_hash_needs_rehash_but_still_verifies():
    stored = hash_password("pw", n=1024)
    assert stored.split("$")[1] == "1024"
    assert needs_rehash(stored) is True
    assert verify_password("pw", stored) is True


def test_hash_over_memory_cap_verifies_false():
    # N=2^20 with r=8 asks for about 1 GiB, over the 256 MiB cap.
    salt = secrets.token_bytes(16).hex()
    digest = secrets.token_bytes(32).hex()
    stored = f"scrypt${2 ** 20}$8$5${salt}${digest}"
    assert verify_password("pw", stored) is False


def test_malformed_stored_strings_verify_false_and_need_rehash():
    for stored in ["", "plain", "scrypt$abc$8$5$00$00", "scrypt$1000$8$5$00$00",
                   "bcrypt$16384$8$5$00$00", "scrypt$16384$8$5$zz$00", None]:
        assert verify_password("pw", stored) is False
        assert needs_rehash(stored) is True


# --- user names --------------------------------------------------------------

@pytest.mark.parametrize("name", ["ann", "a", "rob.s", "r_b-1", "ann@example.org",
                                  "0day", "a" * 64])
def test_name_rule_accepts(name):
    assert normalise_user_name(name) == name


def test_name_rule_lowercases():
    assert normalise_user_name("Ann") == "ann"


@pytest.mark.parametrize("name", ["", ".ann", "-ann", "an n", "ann/x", "a" * 65,
                                  "ann!", "*", None])
def test_name_rule_refuses(name):
    with pytest.raises(AccountError):
        normalise_user_name(name)


# --- users -------------------------------------------------------------------

def test_add_user_stores_only_a_scrypt_hash(db_path):
    with open_store(db_path) as store:
        user = add_user(store, "Ann", PASSWORD)
        stored = store.get_user("ann")
    assert user == stored
    assert stored.name == "ann"
    assert stored.password_hash != PASSWORD
    assert stored.password_hash.startswith("scrypt$")
    assert verify_password(PASSWORD, stored.password_hash) is True
    assert verify_password(WRONG, stored.password_hash) is False
    assert PASSWORD not in _every_value(db_path)
    assert PASSWORD.encode() not in _raw_bytes(db_path)


def test_add_user_without_password(db_path):
    with open_store(db_path) as store:
        user = add_user(store, "bob", None, now="2026-01-02T03:04:05Z")
    assert user == User(name="bob", password_hash=None, created="2026-01-02T03:04:05Z")


def test_add_user_refuses_taken_bad_and_empty(pw_store):
    with pytest.raises(AccountError, match="already exists"):
        add_user(pw_store, "ANN", "other")
    with pytest.raises(AccountError, match="Bad user name"):
        add_user(pw_store, "an n", "pw")
    with pytest.raises(AccountError, match="empty"):
        add_user(pw_store, "bob", "")
    assert [user.name for user in pw_store.list_users()] == ["ann"]


def test_check_new_user(pw_store):
    assert check_new_user(pw_store, "Bob") == "bob"
    with pytest.raises(AccountError):
        check_new_user(pw_store, "ann")


def test_require_user_any_case(pw_store):
    assert require_user(pw_store, "ANN").name == "ann"
    with pytest.raises(AccountError, match="No user"):
        require_user(pw_store, "nobody")


def test_set_password_ends_sessions_and_clears_failures(pw_store):
    add_user(pw_store, "bob", PASSWORD)
    open_session(pw_store, "ann", "password", now=T0)
    bob = open_session(pw_store, "bob", "password", now=T0)
    pw_store.add_signin_failure("ann", T0)
    user = set_password(pw_store, "Ann", NEW_PASSWORD)
    assert verify_password(NEW_PASSWORD, user.password_hash)
    assert pw_store.list_signin_failures("ann") == []
    assert [row["user"] for row in _rows(pw_store, "sessions")] == ["bob"]
    assert resume_session(pw_store, bob, now=T0 + 1)[0] is not None


def test_set_password_none_clears_it(pw_store):
    assert set_password(pw_store, "ann", None).password_hash is None
    assert authenticate_password(pw_store, "ann", PASSWORD) is None


def test_set_password_refuses_empty_and_unknown(pw_store):
    before = pw_store.get_user("ann").password_hash
    with pytest.raises(AccountError):
        set_password(pw_store, "ann", "")
    with pytest.raises(AccountError):
        set_password(pw_store, "nobody", "pw")
    assert pw_store.get_user("ann").password_hash == before


def test_remove_user(pw_store):
    mint_token(pw_store, "ann", "laptop")
    open_session(pw_store, "ann", "password", now=T0)
    pw_store.add_signin_failure("ann", T0)
    assert remove_user(pw_store, "ANN") == "ann"
    for table in ("users", "tokens", "sessions", "signin_failures"):
        assert _rows(pw_store, table) == [], table
    with pytest.raises(AccountError, match="No user"):
        remove_user(pw_store, "ann")


# --- password sign-in --------------------------------------------------------

def test_authenticate_password(pw_store):
    assert authenticate_password(pw_store, "ANN", PASSWORD).name == "ann"
    assert authenticate_password(pw_store, "ann", WRONG) is None
    assert authenticate_password(pw_store, "nobody", PASSWORD) is None
    assert authenticate_password(pw_store, "", PASSWORD) is None


def test_user_with_no_password_cannot_sign_in(store):
    assert authenticate_password(store, "ann", "") is None
    assert authenticate_password(store, "ann", PASSWORD) is None


def test_right_password_over_a_stale_hash_rehashes(pw_store):
    pw_store.set_user_password("ann", hash_password(PASSWORD, n=1024))
    assert needs_rehash(pw_store.get_user("ann").password_hash)
    user = authenticate_password(pw_store, "ann", PASSWORD)
    assert user is not None
    stored = pw_store.get_user("ann").password_hash
    assert not needs_rehash(stored)
    assert verify_password(PASSWORD, stored)


# --- lockout -----------------------------------------------------------------

def test_lockout_constants():
    assert SIGNIN_MAX_FAILURES == 5
    assert SIGNIN_WINDOW == 15 * 60
    assert SIGNIN_LOCKOUT == 15 * 60


def test_five_failures_lock_until_the_window_passes(pw_store):
    user, locked = check_password(pw_store, "ann", PASSWORD, now=T0)
    assert (user.name, locked) == ("ann", False)
    for i in range(SIGNIN_MAX_FAILURES - 1):
        assert check_password(pw_store, "ann", WRONG, now=T0 + i) == (None, False)
    # The fifth failure closes the run and answers locked.
    assert check_password(pw_store, "ann", WRONG, now=T0 + 10) == (None, True)
    # The right password is refused while the lock holds.
    assert check_password(pw_store, "ann", PASSWORD, now=T0 + 11) == (None, True)
    assert check_password(
        pw_store, "ann", PASSWORD, now=T0 + 10 + SIGNIN_LOCKOUT - 1
    ) == (None, True)
    user, locked = check_password(pw_store, "ann", PASSWORD, now=T0 + 10 + SIGNIN_LOCKOUT)
    assert (user.name, locked) == ("ann", False)


def test_four_failures_still_let_the_right_password_in(pw_store):
    for i in range(SIGNIN_MAX_FAILURES - 1):
        check_password(pw_store, "ann", WRONG, now=T0 + i)
    user, locked = check_password(pw_store, "ann", PASSWORD, now=T0 + 10)
    assert user is not None and not locked


def test_a_locked_try_records_nothing(pw_store):
    for i in range(SIGNIN_MAX_FAILURES):
        check_password(pw_store, "ann", WRONG, now=T0 + i)
    before = pw_store.list_signin_failures("ann")
    assert check_password(pw_store, "ann", WRONG, now=T0 + 100) == (None, True)
    assert pw_store.list_signin_failures("ann") == before


def test_name_case_does_not_dodge_the_lock(pw_store):
    add_user(pw_store, "bob", PASSWORD)
    for i in range(SIGNIN_MAX_FAILURES):
        check_password(pw_store, "Ann", WRONG, now=T0 + i)
    assert check_password(pw_store, "ANN", PASSWORD, now=T0 + 10) == (None, True)
    # The lock is per user.
    user, locked = check_password(pw_store, "bob", PASSWORD, now=T0 + 10)
    assert user.name == "bob" and not locked


def test_failures_spread_past_the_window_do_not_lock(pw_store):
    step = SIGNIN_WINDOW / (SIGNIN_MAX_FAILURES - 1) + 1
    for i in range(SIGNIN_MAX_FAILURES):
        assert check_password(pw_store, "ann", WRONG, now=T0 + i * step) == (None, False)
    user, locked = check_password(
        pw_store, "ann", PASSWORD, now=T0 + SIGNIN_MAX_FAILURES * step
    )
    assert user is not None and not locked


def test_a_right_password_clears_the_failures(pw_store):
    for i in range(SIGNIN_MAX_FAILURES - 1):
        check_password(pw_store, "ann", WRONG, now=T0 + i)
    check_password(pw_store, "ann", PASSWORD, now=T0 + 10)
    assert pw_store.list_signin_failures("ann") == []


def test_a_new_password_ends_the_lockout(pw_store):
    for i in range(SIGNIN_MAX_FAILURES):
        check_password(pw_store, "ann", WRONG, now=T0 + i)
    set_password(pw_store, "ann", NEW_PASSWORD)
    user, locked = check_password(pw_store, "ann", NEW_PASSWORD, now=T0 + 10)
    assert user is not None and not locked


def test_an_unknown_name_records_nothing(pw_store):
    assert check_password(pw_store, "nobody", WRONG, now=T0) == (None, False)
    assert _rows(pw_store, "signin_failures") == []


def test_the_lock_survives_reopening_the_store(pw_store, db_path):
    for i in range(SIGNIN_MAX_FAILURES):
        check_password(pw_store, "ann", WRONG, now=T0 + i)
    with open_store(db_path) as again:
        assert check_password(again, "ann", PASSWORD, now=T0 + 10) == (None, True)


def test_check_password_defaults_now_to_the_clock(pw_store, monkeypatch):
    monkeypatch.setattr(accounts.time, "time", lambda: T0)
    check_password(pw_store, "ann", WRONG)
    assert pw_store.list_signin_failures("ann") == [T0]


def test_concurrent_wrong_tries_record_no_more_than_the_limit(pw_store, db_path):
    tries = 12
    start = threading.Barrier(tries)
    answers = []
    answers_guard = threading.Lock()

    def attempt():
        with open_store(db_path) as store:
            start.wait()
            answer = check_password(store, "ann", WRONG, now=T0)
        with answers_guard:
            answers.append(answer)

    threads = [threading.Thread(target=attempt) for _ in range(tries)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(answers) == tries
    assert len(pw_store.list_signin_failures("ann")) == SIGNIN_MAX_FAILURES
    locked = sum(1 for _user, is_locked in answers if is_locked)
    # The failure that locks answers locked, and so does every try after it.
    assert locked == tries - SIGNIN_MAX_FAILURES + 1
    assert all(user is None for user, _locked in answers)


# --- change password ---------------------------------------------------------

def test_wrong_current_password_changes_nothing(pw_store):
    mine = open_session(pw_store, "ann", "password", now=T0)
    other = open_session(pw_store, "ann", "password", now=T0)
    pw_store.add_signin_failure("ann", T0)
    before = pw_store.get_user("ann").password_hash
    refusal = change_password(
        pw_store, "ann", WRONG, NEW_PASSWORD, keep_session_hash=hash_secret(mine)
    )
    assert refusal == WRONG_PASSWORD
    assert pw_store.get_user("ann").password_hash == before
    assert authenticate_password(pw_store, "ann", PASSWORD) is not None
    # A wrong current password is no failed sign-in, and no session ends.
    assert pw_store.list_signin_failures("ann") == [T0]
    assert resume_session(pw_store, other, now=T0 + 1)[0] is not None


def test_change_keeps_this_session_and_ends_the_others(pw_store):
    add_user(pw_store, "bob", PASSWORD)
    mine = open_session(pw_store, "ann", "password", now=T0)
    other = open_session(pw_store, "ann", "password", now=T0)
    bob = open_session(pw_store, "bob", "password", now=T0)
    pw_store.add_signin_failure("ann", T0)
    assert change_password(
        pw_store, "ann", PASSWORD, NEW_PASSWORD, keep_session_hash=hash_secret(mine)
    ) is None
    assert authenticate_password(pw_store, "ann", NEW_PASSWORD) is not None
    assert authenticate_password(pw_store, "ann", PASSWORD) is None
    assert pw_store.list_signin_failures("ann") == []
    assert resume_session(pw_store, mine, now=T0 + 1)[0] is not None
    assert resume_session(pw_store, other, now=T0 + 1) == (None, False)
    assert resume_session(pw_store, bob, now=T0 + 1)[0] is not None


def test_change_with_no_session_to_keep_ends_them_all(pw_store):
    open_session(pw_store, "ann", "password", now=T0)
    assert change_password(
        pw_store, "ann", PASSWORD, NEW_PASSWORD, keep_session_hash=None
    ) is None
    assert _rows(pw_store, "sessions") == []


def test_change_refuses_an_empty_new_password(pw_store):
    before = pw_store.get_user("ann").password_hash
    assert change_password(
        pw_store, "ann", PASSWORD, "", keep_session_hash=None
    ) == PASSWORD_EMPTY
    assert pw_store.get_user("ann").password_hash == before


def test_change_refuses_a_user_with_no_password_or_no_user(store):
    assert change_password(store, "ann", "", "x", keep_session_hash=None) == WRONG_PASSWORD
    assert change_password(store, "nobody", "x", "y", keep_session_hash=None) == WRONG_PASSWORD


def test_change_while_locked_out_works_and_clears_the_lockout(pw_store):
    for i in range(SIGNIN_MAX_FAILURES):
        check_password(pw_store, "ann", WRONG, now=T0 + i)
    assert change_password(
        pw_store, "ann", PASSWORD, NEW_PASSWORD, keep_session_hash=None
    ) is None
    user, locked = check_password(pw_store, "ann", NEW_PASSWORD, now=T0 + 10)
    assert user is not None and not locked


# --- sessions ----------------------------------------------------------------

def test_session_constants():
    assert SESSION_TTL == 12 * 3600
    assert SESSION_WRITE_EVERY == 60


def test_hash_secret_is_sha256_hex():
    assert hash_secret("abc") == hashlib.sha256(b"abc").hexdigest()
    assert len(hash_secret("x")) == 64


def test_open_returns_secret_and_stores_only_its_hash(store):
    secret = open_session(store, "ann", "password", now=T0)
    assert isinstance(secret, str) and len(secret) >= 40
    assert store.get_session(hash_secret(secret)) == Session(
        hash=hash_secret(secret), user="ann", method="password",
        expires=T0 + SESSION_TTL, written=T0,
    )
    assert store.get_session(secret) is None


def test_open_session_method_defaults_to_password(store):
    secret = open_session(store, "ann", now=T0)
    assert store.get_session(hash_secret(secret)).method == "password"


def test_store_never_holds_the_session_secret(store, db_path):
    secret = open_session(store, "ann", "password", now=T0)
    resume_session(store, secret, now=T0 + 2 * MINUTE)
    assert secret not in _every_value(db_path)
    assert secret.encode() not in _raw_bytes(db_path)


def test_open_gives_a_fresh_secret_each_time(store):
    first = open_session(store, "ann", "password", now=T0)
    second = open_session(store, "ann", "password", now=T0)
    assert first != second
    assert len(_rows(store, "sessions")) == 2


def test_open_sweeps_expired_sessions(store):
    old = open_session(store, "ann", "password", now=T0)
    live = open_session(store, "ann", "password", now=T0 + 6 * HOUR)
    open_session(store, "ann", "password", now=T0 + 12 * HOUR + MINUTE)
    assert store.get_session(hash_secret(old)) is None
    assert store.get_session(hash_secret(live)) is not None
    assert len(_rows(store, "sessions")) == 2


def test_open_rejects_unknown_method_and_user(store):
    with pytest.raises(ValueError):
        open_session(store, "ann", "carrier-pigeon", now=T0)
    with pytest.raises(ValueError):
        open_session(store, "nobody", "password", now=T0)
    assert _rows(store, "sessions") == []


def test_open_defaults_now_to_the_clock(store, monkeypatch):
    monkeypatch.setattr(accounts.time, "time", lambda: T0)
    secret = open_session(store, "ann", "password")
    assert store.get_session(hash_secret(secret)).expires == T0 + SESSION_TTL


def test_resume_at_11h59m_keeps_and_extends(store):
    secret = open_session(store, "ann", "password", now=T0)
    later = T0 + 11 * HOUR + 59 * MINUTE
    session, refreshed = resume_session(store, secret, now=later)
    assert refreshed is True
    assert session.user == "ann"
    assert session.expires == later + SESSION_TTL
    assert session.written == later
    assert store.get_session(hash_secret(secret)) == session


def test_resume_at_12h1m_expires_and_deletes_the_row(store):
    secret = open_session(store, "ann", "password", now=T0)
    assert resume_session(store, secret, now=T0 + 12 * HOUR + MINUTE) == (None, False)
    assert _rows(store, "sessions") == []


def test_resume_at_exactly_the_expiry_is_expired(store):
    secret = open_session(store, "ann", "password", now=T0)
    assert resume_session(store, secret, now=T0 + SESSION_TTL) == (None, False)
    assert _rows(store, "sessions") == []


def test_expiry_slides_from_the_last_use(store):
    secret = open_session(store, "ann", "password", now=T0)
    session, _ = resume_session(store, secret, now=T0 + 11 * HOUR)
    assert session is not None
    # 12 h 1 m after the open, but only 1 h 1 m after the last use.
    session, _ = resume_session(store, secret, now=T0 + 12 * HOUR + MINUTE)
    assert session is not None
    # 12 h 1 m after the last write: gone.
    assert resume_session(
        store, secret, now=session.written + 12 * HOUR + MINUTE
    ) == (None, False)


def test_two_uses_ten_seconds_apart_write_once(store):
    secret = open_session(store, "ann", "password", now=T0)
    first_use = T0 + 5 * MINUTE
    session, refreshed = resume_session(store, secret, now=first_use)
    assert refreshed is True
    again, refreshed = resume_session(store, secret, now=first_use + 10)
    assert refreshed is False
    assert again == session
    assert store.get_session(hash_secret(secret)).written == first_use


def test_writes_again_once_the_interval_has_passed(store):
    secret = open_session(store, "ann", "password", now=T0)
    _, refreshed = resume_session(store, secret, now=T0 + SESSION_WRITE_EVERY - 1)
    assert refreshed is False
    session, refreshed = resume_session(store, secret, now=T0 + SESSION_WRITE_EVERY)
    assert refreshed is True
    assert session.expires == T0 + SESSION_WRITE_EVERY + SESSION_TTL


def test_resume_unknown_empty_or_hash_secret(store):
    secret = open_session(store, "ann", "password", now=T0)
    assert resume_session(store, "not-a-secret", now=T0 + 1) == (None, False)
    assert resume_session(store, "", now=T0 + 1) == (None, False)
    # A leaked row's hash is not a cookie.
    assert resume_session(store, hash_secret(secret), now=T0 + 1) == (None, False)
    assert len(_rows(store, "sessions")) == 1


def test_session_goes_with_its_user(store):
    secret = open_session(store, "ann", "password", now=T0)
    store.remove_user("ann")
    assert resume_session(store, secret, now=T0 + 1) == (None, False)


def test_end_session(store):
    secret = open_session(store, "ann", "password", now=T0)
    other = open_session(store, "ann", "password", now=T0)
    assert end_session(store, secret) is True
    assert resume_session(store, secret, now=T0 + 1) == (None, False)
    assert store.get_session(hash_secret(other)) is not None
    assert end_session(store, "nope") is False
    assert end_session(store, "") is False


def test_sessions_persist_across_store_opens(store, db_path):
    secret = open_session(store, "ann", "password", now=T0)
    with open_store(db_path) as again:
        session, refreshed = resume_session(again, secret, now=T0 + 2 * MINUTE)
    assert session is not None and refreshed is True


# --- tokens ------------------------------------------------------------------

def test_token_constants():
    assert TOKEN_PREFIX == "vmp_"
    assert TOKEN_WRITE_EVERY == 60


def test_mint_returns_a_vmp_secret(store):
    secret = mint_token(store, "ann", "laptop")
    assert re.fullmatch(r"vmp_[A-Za-z0-9_-]{43}", secret)
    assert mint_token(store, "ann", "phone") != secret


def test_store_holds_only_the_hash_and_last4_of_a_token(store, db_path):
    secret = mint_token(store, "ann", "laptop")
    token_for(store, secret, now=T0)
    values = _every_value(db_path)
    assert hash_secret(secret) in values
    assert secret[-4:] in values
    assert not any(secret in value for value in values)
    assert not any(secret[len(TOKEN_PREFIX):] in value for value in values)
    raw = _raw_bytes(db_path)
    assert secret[len(TOKEN_PREFIX):].encode() not in raw
    [token] = store.list_tokens("ann")
    assert token.hash == hash_secret(secret)
    assert token.last4 == secret[-4:]


def test_minting_and_using_a_token_never_logs_the_secret(store, caplog):
    caplog.set_level(logging.DEBUG)
    secret = mint_token(store, "ann", "laptop", voice="alice")
    token_for(store, secret, now=T0)
    token_for(store, secret, now=T0 + 2 * MINUTE)
    revoke_token(store, "ann", "laptop")
    body = secret[len(TOKEN_PREFIX):]
    for record in caplog.records:
        assert body not in record.getMessage()


def test_signing_in_never_logs_the_password(pw_store, caplog):
    caplog.set_level(logging.DEBUG)
    for i in range(SIGNIN_MAX_FAILURES):
        check_password(pw_store, "ann", WRONG, now=T0 + i)
    check_password(pw_store, "ann", PASSWORD, now=T0 + 10)
    change_password(pw_store, "ann", PASSWORD, NEW_PASSWORD, keep_session_hash=None)
    set_password(pw_store, "ann", PASSWORD)
    for record in caplog.records:
        message = record.getMessage()
        for secret in (PASSWORD, WRONG, NEW_PASSWORD):
            assert secret not in message


def test_mint_stores_the_voice(store):
    mint_token(store, "ann", "plain")
    mint_token(store, "ann", "speaker", voice="alice")
    mint_token(store, "ann", "none", voice=None)
    voices = {token.name: token.voice for token in store.list_tokens("ann")}
    assert voices == {"plain": "", "speaker": "alice", "none": ""}


def test_mint_accepts_owner_in_any_case(store):
    mint_token(store, "ANN", "laptop")
    assert [token.owner for token in store.list_tokens()] == ["ann"]


def test_mint_refuses_bad_input(store):
    mint_token(store, "ann", "laptop")
    with pytest.raises(AccountError, match="already has"):
        mint_token(store, "ann", "laptop")
    with pytest.raises(AccountError, match="name"):
        mint_token(store, "ann", "")
    with pytest.raises(AccountError, match="name"):
        mint_token(store, "ann", "   ")
    with pytest.raises(AccountError, match="No user"):
        mint_token(store, "nobody", "x")
    assert [token.name for token in store.list_tokens()] == ["laptop"]


def test_same_token_name_for_another_user_allowed(store):
    add_user(store, "bob", None)
    mint_token(store, "ann", "laptop")
    mint_token(store, "bob", "laptop")
    assert [(t.owner, t.name) for t in store.list_tokens()] == [
        ("ann", "laptop"), ("bob", "laptop"),
    ]


def test_mint_uses_injected_time(store):
    mint_token(store, "ann", "x", now="2026-01-02T03:04:05Z")
    [token] = store.list_tokens("ann")
    assert token.created == "2026-01-02T03:04:05Z"
    assert token.last_used is None


def test_token_for_unknown_or_empty_secret(store):
    assert token_for(store, "", now=T0) is None
    assert token_for(store, "vmp_" + "x" * 43, now=T0) is None


def test_token_for_returns_owner_name_and_voice(store):
    secret = mint_token(store, "ann", "laptop", voice="alice")
    token = token_for(store, secret, now=T0)
    assert (token.owner, token.name, token.voice) == ("ann", "laptop", "alice")


def test_token_last_used_is_written_at_most_once_a_minute(store, monkeypatch):
    start = 1_700_000_000
    secret = mint_token(store, "ann", "laptop")
    writes = []
    touch = store.touch_token

    def counting_touch(token_hash, when):
        writes.append(when)
        return touch(token_hash, when)

    monkeypatch.setattr(store, "touch_token", counting_touch)

    first = token_for(store, secret, now=start)
    assert first.last_used == "2023-11-14T22:13:20Z"
    assert store.find_token(hash_secret(secret)).last_used == first.last_used

    # Within the minute: found, nothing written.
    again = token_for(store, secret, now=start + 30)
    assert again.last_used == first.last_used
    again = token_for(store, secret, now=start + TOKEN_WRITE_EVERY - 1)
    assert again.last_used == first.last_used
    assert writes == [first.last_used]

    # 61 s on: written, and the value advanced.
    later = token_for(store, secret, now=start + 61)
    assert later.last_used == "2023-11-14T22:14:21Z"
    assert store.find_token(hash_secret(secret)).last_used == later.last_used
    assert len(writes) == 2


def test_token_for_defaults_now_to_the_clock(store, monkeypatch):
    secret = mint_token(store, "ann", "laptop")
    monkeypatch.setattr(accounts.time, "time", lambda: 1_700_000_000)
    assert token_for(store, secret).last_used == "2023-11-14T22:13:20Z"


def test_list_and_revoke_tokens(store):
    mint_token(store, "ann", "b")
    secret = mint_token(store, "ann", "a")
    assert [t.name for t in list_tokens(store, "ANN")] == ["a", "b"]
    assert revoke_token(store, "ann", "a") is True
    assert token_for(store, secret, now=T0) is None
    with pytest.raises(AccountError, match="no token"):
        revoke_token(store, "ann", "a")
    assert [t.name for t in list_tokens(store, "ann")] == ["b"]
    with pytest.raises(AccountError, match="No user"):
        list_tokens(store, "nobody")
    with pytest.raises(AccountError, match="No user"):
        revoke_token(store, "nobody", "b")


def test_a_locked_users_token_still_works(pw_store):
    secret = mint_token(pw_store, "ann", "laptop")
    for i in range(SIGNIN_MAX_FAILURES):
        check_password(pw_store, "ann", WRONG, now=T0 + i)
    assert token_for(pw_store, secret, now=T0 + 10) is not None
