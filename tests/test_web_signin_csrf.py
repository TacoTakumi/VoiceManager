"""Tests for the sign-in form's pre-session token and the sign-in lockout
through the web form.

GET /signin leaves a short-lived cookie and puts the same value in a hidden
field; POST /signin is refused with 403 unless the two match, before any
password work, so a refused post opens no session and records no failed
sign-in. Five wrong passwords lock the user's password sign-in, even for the
right password, until the lockout ends.

Model-free and server-free: the app comes from create_app() without its
lifespan, the auth store is a SQLite file under tmp_path (VM_DB_PATH), and the
clock is accounts' patched ``time``.
"""
import authkit  # noqa: F401  (sets up the auth env before voicemanager is imported)

import re
import time
import types

import pytest
from fastapi.testclient import TestClient

import voicemanager.config as config
from voicemanager.app import create_app
from voicemanager.auth import accounts
from voicemanager.auth.store import open_store
from voicemanager.web.session import SESSION_COOKIE
from voicemanager.web.signin_routes import SIGNIN_COOKIE, SIGNIN_FIELD, SIGNIN_FORM_STALE

PASSWORD = "correct horse battery"
WRONG = "not the password"
USER = "ann"
SIGNIN = "/signin"
TOKENS = "/tokens"


@pytest.fixture
def clock(monkeypatch):
    """The epoch seconds accounts reads as now; move it with clock[0] += s."""
    now = [time.time()]
    monkeypatch.setattr(accounts, "time", types.SimpleNamespace(time=lambda: now[0]))
    return now


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """A fresh store file holding the user ann, read by the app via VM_DB_PATH."""
    path = tmp_path / "state.db"
    monkeypatch.setenv("VM_DB_PATH", str(path))
    # Settings are read once and cached; drop the cache so this test's env counts.
    monkeypatch.setattr(config, "_settings", None)
    with open_store(path) as store:
        accounts.add_user(store, USER, PASSWORD)
    return path


def make_client(base_url: str = "http://testserver") -> TestClient:
    # No `with`: the lifespan (model loading) never runs.
    return TestClient(create_app(), base_url=base_url, follow_redirects=False)


def signin_form(client: TestClient) -> dict[str, str]:
    """The sign-in form's hidden token as a browser would post it: GET /signin
    leaves the token cookie on ``client`` and the field carries the same value."""
    page = client.get(SIGNIN).text
    found = re.search(rf'name="{SIGNIN_FIELD}" value="([^"]+)"', page)
    assert found, page
    return {SIGNIN_FIELD: found.group(1)}


def post_sign_in(client: TestClient, name: str = USER, password: str = PASSWORD):
    data = signin_form(client)
    data.update(name=name, password=password)
    return client.post(SIGNIN, data=data)


def cookie_header(response, name: str) -> str:
    """The Set-Cookie line that sets ``name``, or ''."""
    for value in response.headers.get_list("set-cookie"):
        if value.startswith(f"{name}="):
            return value
    return ""


def cookie_attrs(header: str) -> set[str]:
    """The lowercased attribute names and name=value pairs of a Set-Cookie line."""
    return {part.strip().lower() for part in header.split(";")[1:]}


def stored_state(db_path) -> tuple[int, list[float]]:
    """(sessions in the store, ann's recorded sign-in failures)."""
    with open_store(db_path) as store:
        sessions = store.connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        return sessions, store.list_signin_failures(USER)


# --- the token cookie and field -----------------------------------------------

def test_the_signin_page_sets_the_cookie_and_the_matching_field(db_path):
    response = make_client().get(SIGNIN)
    header = cookie_header(response, SIGNIN_COOKIE)
    assert header, response.headers.get_list("set-cookie")
    value = header.split(";")[0].split("=", 1)[1]
    assert value
    attrs = cookie_attrs(header)
    assert "httponly" in attrs
    assert any(attr.startswith("samesite=") and attr != "samesite=none" for attr in attrs)
    assert "secure" not in attrs
    assert f"path={SIGNIN}" in attrs
    assert f'name="{SIGNIN_FIELD}" value="{value}"' in response.text


def test_the_cookie_is_secure_over_https(db_path):
    attrs = cookie_attrs(cookie_header(make_client("https://testserver").get(SIGNIN),
                                       SIGNIN_COOKIE))
    assert "secure" in attrs
    assert "httponly" in attrs


def test_each_browser_gets_its_own_token(db_path):
    first = signin_form(make_client())[SIGNIN_FIELD]
    second = signin_form(make_client())[SIGNIN_FIELD]
    assert first != second


def test_a_browser_holding_the_cookie_keeps_its_token(db_path):
    client = make_client()
    token = signin_form(client)[SIGNIN_FIELD]
    again = client.get(SIGNIN)
    assert not cookie_header(again, SIGNIN_COOKIE)
    assert f'name="{SIGNIN_FIELD}" value="{token}"' in again.text


# --- sign-in with and without a matching token ---------------------------------

def test_a_matching_token_signs_in(db_path):
    client = make_client()
    response = post_sign_in(client)
    assert response.status_code == 303
    assert cookie_header(response, SESSION_COOKIE)
    assert client.get(TOKENS).status_code == 200


@pytest.mark.parametrize("case", ["no cookie", "no field", "empty field", "mismatch",
                                  "nothing"])
def test_a_post_without_a_matching_token_is_refused_before_any_check(
        db_path, monkeypatch, case):
    client = make_client()
    data = signin_form(client)
    data.update(name=USER, password=PASSWORD)
    if case == "no cookie":
        client.cookies.delete(SIGNIN_COOKIE, path=SIGNIN)
    elif case == "no field":
        del data[SIGNIN_FIELD]
    elif case == "empty field":
        data[SIGNIN_FIELD] = ""
    elif case == "mismatch":
        data[SIGNIN_FIELD] = "not-the-token"
    else:
        client.cookies.delete(SIGNIN_COOKIE, path=SIGNIN)
        del data[SIGNIN_FIELD]
    calls = []
    monkeypatch.setattr(accounts, "check_password", lambda *a, **k: calls.append(a))

    response = client.post(SIGNIN, data=data)
    assert response.status_code == 403
    assert SIGNIN_FORM_STALE in response.text
    assert not cookie_header(response, SESSION_COOKIE)
    assert calls == []
    assert stored_state(db_path) == (0, [])
    assert client.get(TOKENS).status_code == 303


def test_a_refused_post_never_echoes_the_password(db_path, caplog):
    caplog.set_level("DEBUG")
    client = make_client()
    response = client.post(SIGNIN, data={"name": USER, "password": "hunter2-secret",
                                         SIGNIN_FIELD: "wrong"})
    assert response.status_code == 403
    assert "hunter2-secret" not in response.text
    assert "hunter2-secret" not in caplog.text


def test_the_refusal_page_carries_a_fresh_form_that_signs_in(db_path):
    client = make_client()
    refused = client.post(SIGNIN, data={"name": USER, "password": PASSWORD})
    assert refused.status_code == 403
    found = re.search(rf'name="{SIGNIN_FIELD}" value="([^"]+)"', refused.text)
    assert found
    response = client.post(SIGNIN, data={"name": USER, "password": PASSWORD,
                                         SIGNIN_FIELD: found.group(1)})
    assert response.status_code == 303


def test_a_wrong_password_with_a_matching_token_still_counts(db_path):
    response = post_sign_in(make_client(), password=WRONG)
    assert response.status_code == 400
    sessions, failures = stored_state(db_path)
    assert sessions == 0
    assert len(failures) == 1


# --- the lockout through the web form -------------------------------------------

def test_five_failures_lock_the_right_password_out_until_the_window_passes(
        db_path, clock):
    client = make_client()
    for _ in range(accounts.SIGNIN_MAX_FAILURES - 1):
        assert post_sign_in(client, password=WRONG).status_code == 400
    # The failure that closes the run answers locked.
    assert post_sign_in(client, password=WRONG).status_code == 429

    locked = post_sign_in(client)
    assert locked.status_code == 429
    assert accounts.SIGNIN_LOCKED in locked.text
    assert not cookie_header(locked, SESSION_COOKIE)
    assert stored_state(db_path)[0] == 0
    assert client.get(TOKENS).status_code == 303

    clock[0] += accounts.SIGNIN_LOCKOUT - 1
    assert post_sign_in(client).status_code == 429

    clock[0] += 2
    response = post_sign_in(client)
    assert response.status_code == 303
    assert cookie_header(response, SESSION_COOKIE)
    assert client.get(TOKENS).status_code == 200
    # A right password clears the failures.
    assert stored_state(db_path)[1] == []


def test_four_failures_still_let_the_right_password_in(db_path, clock):
    client = make_client()
    for _ in range(accounts.SIGNIN_MAX_FAILURES - 1):
        assert post_sign_in(client, password=WRONG).status_code == 400
    assert post_sign_in(client).status_code == 303


def test_refused_tokens_do_not_count_toward_the_lockout(db_path, clock):
    client = make_client()
    for _ in range(accounts.SIGNIN_MAX_FAILURES + 1):
        refused = client.post(SIGNIN, data={"name": USER, "password": WRONG,
                                            SIGNIN_FIELD: "wrong"})
        assert refused.status_code == 403
    assert stored_state(db_path)[1] == []
    assert post_sign_in(client).status_code == 303
