"""Tests for the tokens page: the signed-in user lists, makes and revokes
their own tokens and sets the voice each one speaks with.

Model-free and server-free: the app comes from create_app() without its
lifespan (no model loads), the auth store is a SQLite file under pytest's
tmp_path (VM_DB_PATH), sessions are opened with the account rules directly,
and the running TTS engine is a fake on app.state that only names voices.
"""
import authkit  # noqa: F401  (sets up the auth env before voicemanager is imported)

import re

import pytest
from fastapi.testclient import TestClient

import voicemanager.config as config
from voicemanager.app import create_app
from voicemanager.auth import accounts
from voicemanager.auth.store import open_store
from voicemanager.web.session import SESSION_COOKIE, SESSION_FIELD

TOKENS = "/tokens"
CREATE = "/tokens/create"
REVOKE = "/tokens/revoke"
VOICE = "/tokens/voice"

VOICES = ["Alice", "Carter", "Emma"]
SECRET_PATTERN = re.compile(r"vmp_[A-Za-z0-9_-]{43}")


class PresetEngine:
    """A stand-in for a running engine with preset voices (VibeVoice, audiocpp)."""

    available_voices = VOICES
    default_voice = "Carter"


class CloningEngine:
    """A stand-in for OpenVoice: it clones voices and has no presets."""


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """A fresh store file the app under test reads, via VM_DB_PATH."""
    path = tmp_path / "state.db"
    monkeypatch.setenv("VM_DB_PATH", str(path))
    # Settings are read once and cached; drop the cache so this test's env counts.
    monkeypatch.setattr(config, "_settings", None)
    return path


@pytest.fixture
def app(db_path):
    """The app with users ann and bob, bob holding the token "laptop", and a
    running engine with preset voices."""
    with open_store(db_path) as store:
        accounts.add_user(store, "ann", "ann password")
        accounts.add_user(store, "bob", "bob password")
        accounts.mint_token(store, "bob", "laptop", voice="Emma")
    app = create_app()
    app.state.tts_engine = PresetEngine()
    return app


def session_client(app, db_path, user: str = "ann") -> tuple[TestClient, str]:
    """A client holding the cookie of a new session of ``user``, and that
    session's secret."""
    with open_store(db_path) as store:
        secret = accounts.open_session(store, user)
    # No `with`: the lifespan (model loading) never runs.
    client = TestClient(app, follow_redirects=False)
    client.cookies.set(SESSION_COOKIE, secret)
    return client, secret


@pytest.fixture
def ann(app, db_path):
    return session_client(app, db_path)


def post(client: TestClient, path: str, secret: str | None, **data):
    if secret is not None:
        data[SESSION_FIELD] = secret
    return client.post(path, data=data)


def tokens(db_path, owner: str = "ann") -> dict:
    """``owner``'s tokens in the store, by name."""
    with open_store(db_path) as store:
        return {t.name: t for t in accounts.list_tokens(store, owner)}


def mint(db_path, name: str, owner: str = "ann", voice: str | None = None) -> str:
    with open_store(db_path) as store:
        return accounts.mint_token(store, owner, name, voice=voice)


def row(html: str, name: str) -> str | None:
    found = re.search(rf'<tr data-token="{re.escape(name)}">(.*?)</tr>', html, re.S)
    return found.group(1) if found else None


# --- the page -------------------------------------------------------------------


def test_tokens_page_needs_a_session(app):
    resp = TestClient(app, follow_redirects=False).get(TOKENS)
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/signin")


def test_page_lists_only_the_users_own_tokens(db_path, ann):
    client, _ = ann
    mint(db_path, "phone", voice="Alice")
    resp = client.get(TOKENS)
    assert resp.status_code == 200
    phone = row(resp.text, "phone")
    assert phone is not None
    assert tokens(db_path)["phone"].last4 in phone
    assert "Alice" in phone
    assert "never" in phone
    assert "laptop" not in resp.text
    assert "<script" not in resp.text
    assert resp.headers["cache-control"] == "private, no-store"


def test_page_without_tokens_says_so(ann):
    client, _ = ann
    resp = client.get(TOKENS)
    assert "You have no tokens." in resp.text
    assert "<tr data-token=" not in resp.text


def test_every_form_carries_the_session(db_path, ann):
    client, secret = ann
    mint(db_path, "phone")
    html = client.get(TOKENS).text
    forms = re.findall(r'<form[^>]*method="post"[^>]*>(.*?)</form>', html, re.S)
    assert len(forms) == 4  # sign-out, create, one voice, one revoke
    for body in forms:
        assert f'<input type="hidden" name="{SESSION_FIELD}" value="{secret}">' in body


# --- create -----------------------------------------------------------------------


def test_create_shows_the_secret_once(db_path, ann):
    client, secret = ann
    resp = post(client, CREATE, secret, name="phone")
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "private, no-store"
    shown = SECRET_PATTERN.findall(resp.text)
    assert len(set(shown)) == 1
    minted = shown[0]
    stored = tokens(db_path)["phone"]
    assert stored.hash == accounts.hash_secret(minted)
    assert stored.last4 == minted[-4:]
    assert stored.voice == ""

    again = client.get(TOKENS)
    assert SECRET_PATTERN.findall(again.text) == []
    assert minted not in again.text
    assert minted[-4:] in row(again.text, "phone")


def test_create_with_a_voice_stores_it(db_path, ann):
    client, secret = ann
    resp = post(client, CREATE, secret, name="phone", voice="Alice")
    assert resp.status_code == 200
    assert tokens(db_path)["phone"].voice == "Alice"


def test_create_with_the_server_default_stores_no_voice(db_path, ann):
    client, secret = ann
    assert post(client, CREATE, secret, name="phone", voice="").status_code == 200
    assert tokens(db_path)["phone"].voice == ""


def test_create_with_an_unknown_voice_is_refused(db_path, ann):
    client, secret = ann
    resp = post(client, CREATE, secret, name="phone", voice="Nobody")
    assert resp.status_code == 400
    assert SECRET_PATTERN.findall(resp.text) == []
    assert tokens(db_path) == {}


@pytest.mark.parametrize("name", ["", "   "])
def test_create_without_a_name_is_refused(db_path, ann, name):
    client, secret = ann
    resp = post(client, CREATE, secret, name=name)
    assert resp.status_code == 400
    assert "The token needs a name." in resp.text
    assert tokens(db_path) == {}


def test_create_with_a_name_in_use_is_refused(db_path, ann):
    client, secret = ann
    mint(db_path, "phone")
    before = tokens(db_path)
    resp = post(client, CREATE, secret, name="phone")
    assert resp.status_code == 409
    assert SECRET_PATTERN.findall(resp.text) == []
    assert tokens(db_path) == before


def test_create_may_reuse_another_users_token_name(db_path, ann):
    client, secret = ann
    before = tokens(db_path, "bob")
    assert post(client, CREATE, secret, name="laptop").status_code == 200
    assert set(tokens(db_path)) == {"laptop"}
    assert tokens(db_path, "bob") == before


# --- revoke -----------------------------------------------------------------------


def test_revoke_own_token(db_path, ann):
    client, secret = ann
    mint(db_path, "phone")
    mint(db_path, "desk")
    resp = post(client, REVOKE, secret, name="phone")
    assert resp.status_code == 303
    assert resp.headers["location"] == TOKENS
    assert set(tokens(db_path)) == {"desk"}
    assert row(client.get(TOKENS).text, "phone") is None


def test_revoke_another_users_token_name_is_a_404(db_path, ann):
    client, secret = ann
    before = tokens(db_path, "bob")
    resp = post(client, REVOKE, secret, name="laptop")
    assert resp.status_code == 404
    assert tokens(db_path, "bob") == before


def test_revoke_an_unknown_name_is_a_404_and_keeps_own_tokens(db_path, ann):
    client, secret = ann
    mint(db_path, "phone")
    resp = post(client, REVOKE, secret, name="nope")
    assert resp.status_code == 404
    assert set(tokens(db_path)) == {"phone"}


# --- voice ------------------------------------------------------------------------


def test_voice_control_lists_the_running_engines_voices(db_path, ann):
    client, _ = ann
    mint(db_path, "phone", voice="Alice")
    phone = row(client.get(TOKENS).text, "phone")
    assert 'class="token-voice"' in phone
    options = re.findall(r'<option value="([^"]*)"', phone)
    assert options == [""] + VOICES
    assert re.search(r'<option value="">server default', phone)
    assert '<option value="Alice" selected>' in phone


def test_create_form_offers_the_running_engines_voices(ann):
    client, _ = ann
    form = re.search(r'<form class="token-create"(.*?)</form>',
                     client.get(TOKENS).text, re.S).group(1)
    assert re.findall(r'<option value="([^"]*)"', form) == [""] + VOICES


def test_set_voice(db_path, ann):
    client, secret = ann
    mint(db_path, "phone")
    resp = post(client, VOICE, secret, name="phone", voice="Emma")
    assert resp.status_code == 303
    assert resp.headers["location"] == TOKENS
    assert tokens(db_path)["phone"].voice == "Emma"


def test_set_voice_to_the_server_default_clears_it(db_path, ann):
    client, secret = ann
    mint(db_path, "phone", voice="Alice")
    assert post(client, VOICE, secret, name="phone", voice="").status_code == 303
    assert tokens(db_path)["phone"].voice == ""


@pytest.mark.parametrize("voice", ["Nobody", "alice", "Alice Emma"])
def test_set_voice_not_offered_is_refused(db_path, ann, voice):
    client, secret = ann
    mint(db_path, "phone", voice="Alice")
    resp = post(client, VOICE, secret, name="phone", voice=voice)
    assert resp.status_code == 400
    assert tokens(db_path)["phone"].voice == "Alice"


def test_set_voice_on_another_users_token_name_is_a_404(db_path, ann):
    client, secret = ann
    before = tokens(db_path, "bob")
    resp = post(client, VOICE, secret, name="laptop", voice="Alice")
    assert resp.status_code == 404
    assert tokens(db_path, "bob") == before


def test_set_voice_on_an_unknown_name_is_a_404(db_path, ann):
    client, secret = ann
    resp = post(client, VOICE, secret, name="nope", voice="Alice")
    assert resp.status_code == 404
    assert tokens(db_path) == {}


# --- an engine without presets (OpenVoice) ----------------------------------------


@pytest.fixture
def cloning(app, ann):
    app.state.tts_engine = CloningEngine()
    return ann


def test_no_voice_control_without_presets(db_path, cloning):
    client, _ = cloning
    mint(db_path, "phone")
    html = client.get(TOKENS).text
    assert row(html, "phone") is not None
    assert 'class="token-voice"' not in html
    assert 'name="voice"' not in html
    assert "<select" not in html


def test_voice_posts_are_refused_without_presets(db_path, cloning):
    client, secret = cloning
    mint(db_path, "phone")
    for voice in ("Alice", ""):
        resp = post(client, VOICE, secret, name="phone", voice=voice)
        assert resp.status_code == 400
        assert tokens(db_path)["phone"].voice == ""


def test_create_with_a_voice_is_refused_without_presets(db_path, cloning):
    client, secret = cloning
    resp = post(client, CREATE, secret, name="phone", voice="Alice")
    assert resp.status_code == 400
    assert tokens(db_path) == {}
    assert post(client, CREATE, secret, name="phone").status_code == 200
    assert tokens(db_path)["phone"].voice == ""


# --- the session field ------------------------------------------------------------


FORMS = [
    (CREATE, {"name": "new"}),
    (REVOKE, {"name": "phone"}),
    (VOICE, {"name": "phone", "voice": "Emma"}),
]


@pytest.mark.parametrize("path, data", FORMS)
@pytest.mark.parametrize("field", [None, "", "wrong"])
def test_missing_or_wrong_session_field_is_refused_and_changes_nothing(
        db_path, ann, path, data, field):
    client, _ = ann
    mint(db_path, "phone", voice="Alice")
    before = tokens(db_path)
    resp = post(client, path, field, **data)
    assert resp.status_code == 403
    assert SECRET_PATTERN.findall(resp.text) == []
    assert tokens(db_path) == before


@pytest.mark.parametrize("path, data", FORMS)
def test_another_sessions_secret_is_refused(app, db_path, ann, path, data):
    client, _ = ann
    _, other_secret = session_client(app, db_path)
    mint(db_path, "phone", voice="Alice")
    before = tokens(db_path)
    resp = post(client, path, other_secret, **data)
    assert resp.status_code == 403
    assert tokens(db_path) == before


@pytest.mark.parametrize("path, data", FORMS)
def test_post_without_a_session_changes_nothing(app, db_path, path, data):
    mint(db_path, "phone", voice="Alice")
    before = tokens(db_path)
    resp = post(TestClient(app, follow_redirects=False), path, "anything", **data)
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/signin")
    assert tokens(db_path) == before


def test_token_secrets_are_not_logged(ann, caplog):
    client, secret = ann
    with caplog.at_level("DEBUG"):
        resp = post(client, CREATE, secret, name="phone")
    minted = SECRET_PATTERN.findall(resp.text)[0]
    assert minted not in caplog.text
    assert secret not in caplog.text
