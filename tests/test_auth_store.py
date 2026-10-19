"""Tests for the SQLite auth store: location, schema, migration and primitives.

Model-free and server-free: every store lives under pytest's tmp_path, and the
environment that decides the default location (HOME, XDG_STATE_HOME,
VM_DB_PATH) is set per test with monkeypatch.
"""
import sqlite3

import pytest

from voicemanager.auth.store import (
    ADDED_COLUMNS,
    SESSION_METHODS,
    STORE_FILENAME,
    Conflict,
    Session,
    SqliteStore,
    Token,
    User,
    open_store,
    store_path,
)
from voicemanager.config import Settings, resolve_db_path


def _user(name="ann", password_hash="h1"):
    return User(name=name, password_hash=password_hash, created="2026-01-01T00:00:00Z")


def _token(owner="ann", name="laptop", hash_="tokhash", last4="abcd", voice=""):
    return Token(owner=owner, name=name, hash=hash_, last4=last4, voice=voice,
                 created="2026-01-02T00:00:00Z")


def _session(hash_="sess", user="ann", method="password", expires=200.0, written=100.0):
    return Session(hash=hash_, user=user, method=method, expires=expires, written=written)


def _settings(**overrides):
    # No .env file: a developer's local .env must not steer the test.
    return Settings(_env_file=None, **overrides)


@pytest.fixture
def clean_env(tmp_path, monkeypatch):
    """A temporary HOME with no XDG_STATE_HOME and no VM_DB_PATH."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("VM_DB_PATH", raising=False)
    return home


# --- location ---------------------------------------------------------------

def test_store_filename():
    assert STORE_FILENAME == "state.db"


def test_default_path_is_under_home_local_state(clean_env):
    expected = clean_env / ".local" / "state" / "voicemanager" / "state.db"
    assert resolve_db_path("") == expected
    assert store_path(_settings()) == expected
    assert expected.parent.is_dir()


def test_xdg_state_home_wins_over_home(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    expected = tmp_path / "xdg" / "voicemanager" / "state.db"
    assert store_path(_settings()) == expected
    assert expected.parent.is_dir()
    assert not (clean_env / ".local").exists()


def test_db_path_setting_wins(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    target = tmp_path / "custom" / "deep" / "auth.db"
    assert store_path(_settings(db_path=str(target))) == target
    assert target.parent.is_dir()
    assert not (tmp_path / "xdg").exists()


def test_db_path_read_from_env(clean_env, tmp_path, monkeypatch):
    target = tmp_path / "from-env.db"
    monkeypatch.setenv("VM_DB_PATH", str(target))
    assert _settings().db_path == str(target)
    assert store_path(_settings()) == target


def test_db_path_expands_user(clean_env):
    assert resolve_db_path("~/x/state.db") == clean_env / "x" / "state.db"


def test_open_store_default_creates_file_at_resolved_path(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    with open_store(settings=_settings()) as store:
        assert isinstance(store, SqliteStore)
        store.add_user(_user())
    assert (tmp_path / "xdg" / "voicemanager" / STORE_FILENAME).is_file()
    assert not clean_env.exists()


def test_open_store_explicit_path(tmp_path):
    path = tmp_path / "a" / "b" / "x.db"
    with open_store(path) as store:
        store.add_user(_user())
    assert path.is_file()
    with open_store(path) as store:
        assert store.get_user("ann") == _user()


# --- schema and migration ---------------------------------------------------

def _columns(path, table):
    conn = sqlite3.connect(path)
    try:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


def test_schema_has_auth_tables_only(tmp_path):
    path = tmp_path / "s.db"
    open_store(path).close()
    conn = sqlite3.connect(path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert names == {"users", "tokens", "sessions", "signin_failures"}


def test_tokens_table_has_voice_and_no_scope(tmp_path):
    path = tmp_path / "s.db"
    open_store(path).close()
    columns = _columns(path, "tokens")
    assert "voice" in columns
    assert "scope" not in columns


def test_voice_is_a_registered_added_column():
    assert any(table == "tokens" and column == "voice"
               for table, column, _kind in ADDED_COLUMNS)


def test_reopen_is_idempotent(tmp_path):
    path = tmp_path / "s.db"
    with open_store(path) as store:
        store.add_user(_user())
    with open_store(path) as store:
        assert [u.name for u in store.list_users()] == ["ann"]
    assert _columns(path, "tokens").count("voice") == 1


def test_old_tokens_table_gains_voice_and_keeps_rows(tmp_path):
    """A store made before the voice column existed is upgraded in place."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE users (
            name          TEXT PRIMARY KEY,
            password_hash TEXT,
            created       TEXT NOT NULL
        );
        CREATE TABLE tokens (
            hash      TEXT PRIMARY KEY,
            owner     TEXT NOT NULL REFERENCES users(name) ON DELETE CASCADE,
            name      TEXT NOT NULL,
            last4     TEXT,
            created   TEXT NOT NULL,
            last_used TEXT,
            UNIQUE (owner, name)
        );
        INSERT INTO users VALUES ('ann', 'h1', '2026-01-01T00:00:00Z');
        INSERT INTO tokens VALUES
            ('tokhash', 'ann', 'laptop', 'abcd', '2026-01-02T00:00:00Z', NULL);
    """)
    conn.commit()
    conn.close()
    assert "voice" not in _columns(path, "tokens")

    with open_store(path) as store:
        token = store.find_token("tokhash")
        assert token == _token()
        assert token.voice == ""
        assert store.get_user("ann") == _user()

    assert "voice" in _columns(path, "tokens")
    conn = sqlite3.connect(path)
    row = conn.execute("SELECT owner, name, last4, voice FROM tokens").fetchone()
    conn.close()
    assert row == ("ann", "laptop", "abcd", "")


# --- users ------------------------------------------------------------------

def test_user_round_trip(tmp_path):
    with open_store(tmp_path / "s.db") as store:
        store.add_user(_user("ann", "h1"))
        store.add_user(_user("bob", None))
        assert store.get_user("ann") == _user("ann", "h1")
        assert store.get_user("bob").password_hash is None
        assert store.get_user("nobody") is None
        assert [u.name for u in store.list_users()] == ["ann", "bob"]


def test_add_user_duplicate_conflicts(tmp_path):
    with open_store(tmp_path / "s.db") as store:
        store.add_user(_user())
        with pytest.raises(Conflict):
            store.add_user(_user())


def test_set_user_password(tmp_path):
    with open_store(tmp_path / "s.db") as store:
        store.add_user(_user("ann", "h1"))
        updated = store.set_user_password("ann", "h2")
        assert updated.password_hash == "h2"
        assert store.get_user("ann").password_hash == "h2"
        assert store.set_user_password("ann", None).password_hash is None
        assert store.set_user_password("nobody", "x") is None


# --- tokens -----------------------------------------------------------------

def test_token_round_trip_with_and_without_voice(tmp_path):
    with open_store(tmp_path / "s.db") as store:
        store.add_user(_user())
        plain = _token(name="plain", hash_="h-plain")
        voiced = _token(name="voiced", hash_="h-voiced", voice="designed_female_s101",
                        last4=None)
        store.add_token(plain)
        store.add_token(voiced)
        assert store.find_token("h-plain") == plain
        assert store.find_token("h-plain").voice == ""
        assert store.find_token("h-voiced") == voiced
        assert store.find_token("h-voiced").voice == "designed_female_s101"
        assert store.find_token("missing") is None
        assert [t.name for t in store.list_tokens()] == ["plain", "voiced"]


def test_token_voice_defaults_to_empty():
    token = Token(owner="ann", name="x", hash="h", last4=None, created="c")
    assert token.voice == ""
    assert token.last_used is None


def test_add_token_duplicate_owner_name_conflicts(tmp_path):
    with open_store(tmp_path / "s.db") as store:
        store.add_user(_user())
        store.add_token(_token(hash_="h1"))
        with pytest.raises(Conflict):
            store.add_token(_token(hash_="h2"))


def test_add_token_requires_user(tmp_path):
    with open_store(tmp_path / "s.db") as store:
        with pytest.raises(ValueError):
            store.add_token(_token(owner="nobody"))


def test_list_tokens_by_owner(tmp_path):
    with open_store(tmp_path / "s.db") as store:
        store.add_user(_user("ann"))
        store.add_user(_user("bob"))
        store.add_token(_token(owner="ann", name="x", hash_="h1"))
        store.add_token(_token(owner="bob", name="y", hash_="h2"))
        assert [t.name for t in store.list_tokens("ann")] == ["x"]
        assert [t.owner for t in store.list_tokens()] == ["ann", "bob"]


def test_revoke_and_touch_token(tmp_path):
    with open_store(tmp_path / "s.db") as store:
        store.add_user(_user())
        store.add_token(_token(hash_="h1"))
        assert store.touch_token("h1", "2026-02-02T00:00:00Z") is True
        assert store.find_token("h1").last_used == "2026-02-02T00:00:00Z"
        assert store.touch_token("missing", "x") is False
        assert store.revoke_token("ann", "laptop") is True
        assert store.revoke_token("ann", "laptop") is False
        assert store.find_token("h1") is None


# --- sessions ---------------------------------------------------------------

def test_session_methods():
    assert SESSION_METHODS == ("password",)


def test_session_round_trip_and_update(tmp_path):
    with open_store(tmp_path / "s.db") as store:
        store.add_user(_user())
        store.add_session(_session())
        assert store.get_session("sess") == _session()
        updated = store.update_session("sess", expires=500.0, written=300.0)
        assert updated.expires == 500.0 and updated.written == 300.0
        assert store.update_session("missing", expires=1.0, written=1.0) is None
        assert store.remove_session("sess") is True
        assert store.remove_session("sess") is False
        assert store.get_session("sess") is None


def test_add_session_rejects_bad_method_and_missing_user(tmp_path):
    with open_store(tmp_path / "s.db") as store:
        store.add_user(_user())
        with pytest.raises(ValueError):
            store.add_session(_session(method="magic"))
        with pytest.raises(ValueError):
            store.add_session(_session(user="nobody"))
        store.add_session(_session())
        with pytest.raises(Conflict):
            store.add_session(_session())


def test_remove_user_sessions_keep_and_expired(tmp_path):
    with open_store(tmp_path / "s.db") as store:
        store.add_user(_user())
        store.add_session(_session("s1", expires=50.0))
        store.add_session(_session("s2", expires=500.0))
        store.add_session(_session("s3", expires=500.0))
        assert store.remove_user_sessions("ann", keep="s3") == 2
        assert store.get_session("s3") is not None
        assert store.remove_expired_sessions(1000.0) == 1
        assert store.get_session("s3") is None


# --- sign-in failures -------------------------------------------------------

def test_signin_failures(tmp_path):
    with open_store(tmp_path / "s.db") as store:
        store.add_user(_user())
        for at in (30.0, 10.0, 20.0):
            store.add_signin_failure("ann", at)
        assert store.list_signin_failures("ann") == [10.0, 20.0, 30.0]
        assert store.clear_signin_failures("ann", before=25.0) == 2
        assert store.list_signin_failures("ann") == [30.0]
        assert store.clear_signin_failures("ann") == 1
        with pytest.raises(ValueError):
            store.add_signin_failure("nobody", 1.0)


# --- remove_user ------------------------------------------------------------

def _populate(store, name):
    store.add_user(_user(name))
    store.add_token(_token(owner=name, hash_=f"tok-{name}"))
    store.add_session(_session(hash_=f"sess-{name}", user=name))
    store.add_signin_failure(name, 1.0)


def test_remove_user_deletes_everything_it_owns(tmp_path):
    path = tmp_path / "s.db"
    with open_store(path) as store:
        _populate(store, "ann")
        _populate(store, "bob")
        assert store.remove_user("ann") is True
        assert store.remove_user("ann") is False
        assert store.get_user("ann") is None
        assert store.list_tokens("ann") == []
        assert store.get_session("sess-ann") is None
        assert store.list_signin_failures("ann") == []
        # The other user is untouched.
        assert store.find_token("tok-bob") is not None
        assert store.get_session("sess-bob") is not None
        assert store.list_signin_failures("bob") == [1.0]
    conn = sqlite3.connect(path)
    for table, column in (("tokens", "owner"), ("sessions", "user"),
                          ("signin_failures", "user")):
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {column} = 'ann'").fetchone()[0]
        assert count == 0, table
    conn.close()


def test_remove_user_is_one_transaction(tmp_path):
    """A failure partway through leaves every row in place."""
    path = tmp_path / "s.db"
    with open_store(path) as store:
        _populate(store, "ann")
    # A trigger that aborts the final DELETE of the user row.
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TRIGGER block_user_delete BEFORE DELETE ON users "
        "BEGIN SELECT RAISE(ABORT, 'blocked'); END")
    conn.commit()
    conn.close()
    with open_store(path) as store:
        with pytest.raises(sqlite3.DatabaseError):
            store.remove_user("ann")
    with open_store(path) as store:
        assert store.get_user("ann") is not None
        assert store.find_token("tok-ann") is not None
        assert store.get_session("sess-ann") is not None
        assert store.list_signin_failures("ann") == [1.0]
