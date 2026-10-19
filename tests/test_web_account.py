"""Tests for the account page: the signed-in user changes their password.

Model-free and server-free: the app comes from create_app() without its
lifespan (no model loads), the auth store is a SQLite file under pytest's
tmp_path (VM_DB_PATH), and sessions are opened with the account rules
directly, so these tests do not go through the sign-in form.
"""
import authkit  # noqa: F401  (sets up the auth env before voicemanager is imported)

import pytest
from fastapi.testclient import TestClient

import voicemanager.config as config
from voicemanager.app import create_app
from voicemanager.auth import accounts
from voicemanager.auth.store import open_store
from voicemanager.web.session import SESSION_COOKIE, SESSION_FIELD

OLD_PASSWORD = "correct horse battery"
NEW_PASSWORD = "staple tray gravel"
ACCOUNT = "/account"
CHANGE = "/account/password"


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
    """The app, with the user ann (who has a password) in its store."""
    with open_store(db_path) as store:
        accounts.add_user(store, "ann", OLD_PASSWORD)
    return create_app()


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
def ann_client(app, db_path):
    return session_client(app, db_path)


def change(client: TestClient, secret: str | None, current: str, new: str,
           confirm: str | None = None):
    data = {"current": current, "new": new,
            "confirm": new if confirm is None else confirm}
    if secret is not None:
        data[SESSION_FIELD] = secret
    return client.post(CHANGE, data=data)


def signs_in(db_path, password: str, user: str = "ann") -> bool:
    with open_store(db_path) as store:
        found, locked = accounts.check_password(store, user, password)
    return found is not None and not locked


def stored_hash(db_path, user: str = "ann") -> str:
    with open_store(db_path) as store:
        return store.get_user(user).password_hash


def still_signed_in(client: TestClient) -> bool:
    return client.get(ACCOUNT).status_code == 200


# --- the page -------------------------------------------------------------------


def test_account_page_needs_a_session(app):
    resp = TestClient(app, follow_redirects=False).get(ACCOUNT)
    assert resp.status_code == 303
    assert resp.headers["location"].startswith("/signin")


def test_account_page_shows_the_user_and_the_password_form(ann_client):
    client, secret = ann_client
    resp = client.get(ACCOUNT)
    assert resp.status_code == 200
    assert "<strong>ann</strong>" in resp.text
    assert f'action="{CHANGE}"' in resp.text
    for field in ("current", "new", "confirm"):
        assert f'type="password" name="{field}"' in resp.text
    assert f'name="{SESSION_FIELD}" value="{secret}"' in resp.text
    assert "<script" not in resp.text
    assert resp.headers["cache-control"] == "private, no-store"


# --- changing the password ---------------------------------------------------------


def test_wrong_current_password_is_refused_and_changes_nothing(app, db_path,
                                                               ann_client):
    client, secret = ann_client
    other, _ = session_client(app, db_path)
    before = stored_hash(db_path)
    resp = change(client, secret, "not it", NEW_PASSWORD)
    assert resp.status_code == 400
    assert accounts.WRONG_PASSWORD in resp.text
    assert stored_hash(db_path) == before
    assert signs_in(db_path, OLD_PASSWORD)
    assert not signs_in(db_path, NEW_PASSWORD)
    assert still_signed_in(client)
    assert still_signed_in(other)


def test_right_current_password_changes_it_and_ends_other_sessions(app, db_path,
                                                                   ann_client):
    client, secret = ann_client
    other, _ = session_client(app, db_path)
    resp = change(client, secret, OLD_PASSWORD, NEW_PASSWORD)
    assert resp.status_code == 200
    assert "Your password is changed." in resp.text
    assert signs_in(db_path, NEW_PASSWORD)
    assert not signs_in(db_path, OLD_PASSWORD)
    # This session stays; the user's other session is sent to sign in.
    assert still_signed_in(client)
    gone = other.get(ACCOUNT)
    assert gone.status_code == 303
    assert gone.headers["location"].startswith("/signin")


def test_change_leaves_another_users_sessions_alone(app, db_path, ann_client):
    with open_store(db_path) as store:
        accounts.add_user(store, "bob", "bob password")
    bob, _ = session_client(app, db_path, "bob")
    client, secret = ann_client
    assert change(client, secret, OLD_PASSWORD, NEW_PASSWORD).status_code == 200
    assert still_signed_in(bob)
    assert signs_in(db_path, "bob password", "bob")


@pytest.mark.parametrize("new, confirm, message", [
    (NEW_PASSWORD, NEW_PASSWORD + "x", "The two new passwords do not match."),
    ("", "", accounts.PASSWORD_EMPTY),
])
def test_bad_new_password_is_refused_and_changes_nothing(app, db_path, ann_client,
                                                         new, confirm, message):
    client, secret = ann_client
    other, _ = session_client(app, db_path)
    before = stored_hash(db_path)
    resp = change(client, secret, OLD_PASSWORD, new, confirm)
    assert resp.status_code == 400
    assert message in resp.text
    assert stored_hash(db_path) == before
    assert signs_in(db_path, OLD_PASSWORD)
    assert still_signed_in(other)


@pytest.mark.parametrize("field", [None, "", "wrong"])
def test_missing_or_wrong_session_field_is_refused_and_changes_nothing(
        app, db_path, ann_client, field):
    client, _ = ann_client
    other, _ = session_client(app, db_path)
    before = stored_hash(db_path)
    resp = change(client, field, OLD_PASSWORD, NEW_PASSWORD)
    assert resp.status_code == 403
    assert stored_hash(db_path) == before
    assert signs_in(db_path, OLD_PASSWORD)
    assert not signs_in(db_path, NEW_PASSWORD)
    assert still_signed_in(client)
    assert still_signed_in(other)


def test_another_sessions_secret_is_refused(app, db_path, ann_client):
    client, _ = ann_client
    _, other_secret = session_client(app, db_path)
    before = stored_hash(db_path)
    resp = change(client, other_secret, OLD_PASSWORD, NEW_PASSWORD)
    assert resp.status_code == 403
    assert stored_hash(db_path) == before


def test_change_without_a_session_changes_nothing(app, db_path):
    before = stored_hash(db_path)
    resp = change(TestClient(app, follow_redirects=False), "anything",
                  OLD_PASSWORD, NEW_PASSWORD)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/signin"
    assert stored_hash(db_path) == before


def test_passwords_are_not_logged(db_path, ann_client, caplog):
    client, secret = ann_client
    with caplog.at_level("DEBUG"):
        change(client, secret, "not it", NEW_PASSWORD)
        change(client, secret, OLD_PASSWORD, NEW_PASSWORD)
    for text in (OLD_PASSWORD, NEW_PASSWORD, "not it", secret):
        assert text not in caplog.text
