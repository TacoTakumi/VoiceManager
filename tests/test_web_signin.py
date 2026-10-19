"""Tests for the web shell: sign-in, sign-out, the session cookie and the page CSP.

Model-free and server-free: the app comes from create_app() without its
lifespan (no model loads), the auth store is a SQLite file under pytest's
tmp_path (VM_DB_PATH), and users are seeded with the account rules directly.
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
from voicemanager.web.signin_routes import SIGNIN_COOKIE, SIGNIN_FIELD

PASSWORD = "correct horse battery"
SIGNIN = "/signin"
SIGNOUT = "/signout"
TOKENS = "/tokens"


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """A fresh store file the app under test reads, via VM_DB_PATH."""
    path = tmp_path / "state.db"
    monkeypatch.setenv("VM_DB_PATH", str(path))
    # Settings are read once and cached; drop the cache so this test's env counts.
    monkeypatch.setattr(config, "_settings", None)
    return path


@pytest.fixture
def ann(db_path):
    """A store holding the user ann with a password."""
    with open_store(db_path) as store:
        accounts.add_user(store, "ann", PASSWORD)
    return "ann"


def make_client(base_url: str = "http://testserver") -> TestClient:
    # No `with`: the lifespan (model loading) never runs.
    return TestClient(create_app(), base_url=base_url, follow_redirects=False)


def signin_form(client: TestClient) -> dict[str, str]:
    """The sign-in form's token as a browser would post it: GET /signin leaves
    the token cookie on ``client`` and the hidden field carries the same value.
    A signed-in client is sent on from /signin, so it posts the cookie it holds."""
    page = client.get(SIGNIN).text
    found = re.search(rf'name="{SIGNIN_FIELD}" value="([^"]+)"', page)
    return {SIGNIN_FIELD: found.group(1) if found else client.cookies.get(SIGNIN_COOKIE)}


def sign_in(client: TestClient, name: str = "ann", password: str = PASSWORD, **extra):
    data = {**signin_form(client), "name": name, "password": password, **extra}
    return client.post(SIGNIN, data=data)


def session_cookie_header(response) -> str:
    """The Set-Cookie line that carries the session cookie, or ''."""
    for value in response.headers.get_list("set-cookie"):
        if value.startswith(f"{SESSION_COOKIE}="):
            return value
    return ""


def cookie_attrs(header: str) -> set[str]:
    """The lowercased attribute names and name=value pairs of a Set-Cookie line."""
    return {part.strip().lower() for part in header.split(";")[1:]}


def csp_directives(response) -> dict[str, str]:
    policy = response.headers["content-security-policy"]
    directives = {}
    for part in policy.split(";"):
        part = part.strip()
        if part:
            name, _, value = part.partition(" ")
            directives[name] = value
    return directives


def style_nonce(response) -> str:
    match = re.search(r"'nonce-([^']+)'", csp_directives(response)["style-src"])
    assert match, response.headers["content-security-policy"]
    return match.group(1)


# --- redirects without a session ------------------------------------------------

def test_home_without_session_redirects_to_signin(ann):
    response = make_client().get("/")
    assert response.status_code == 303
    assert response.headers["location"] == SIGNIN


def test_tokens_without_session_redirects_to_signin_with_next(ann):
    response = make_client().get(TOKENS)
    assert response.status_code == 303
    assert response.headers["location"] == f"{SIGNIN}?next=%2Ftokens"


def test_unknown_cookie_counts_as_no_session(ann):
    client = make_client()
    client.cookies.set(SESSION_COOKIE, "not-a-session")
    response = client.get(TOKENS)
    assert response.status_code == 303
    assert response.headers["location"].startswith(SIGNIN)


# --- sign-in ----------------------------------------------------------------------

def test_signin_over_http_sets_httponly_samesite_cookie_without_secure(ann):
    client = make_client()
    response = sign_in(client)
    assert response.status_code == 303
    header = session_cookie_header(response)
    assert header, response.headers.get_list("set-cookie")
    attrs = cookie_attrs(header)
    assert "httponly" in attrs
    assert any(attr.startswith("samesite=") and attr != "samesite=none" for attr in attrs)
    assert "secure" not in attrs
    assert "private" in response.headers["cache-control"]
    assert "no-store" in response.headers["cache-control"]

    page = client.get(TOKENS)
    assert page.status_code == 200
    assert "ann" in page.text


def test_signin_over_https_sets_secure_cookie(ann):
    client = make_client("https://testserver")
    response = sign_in(client)
    assert response.status_code == 303
    attrs = cookie_attrs(session_cookie_header(response))
    assert "secure" in attrs
    assert "httponly" in attrs
    assert any(attr.startswith("samesite=") and attr != "samesite=none" for attr in attrs)
    assert client.get(TOKENS).status_code == 200


def test_signin_name_is_case_insensitive(ann):
    client = make_client()
    assert sign_in(client, name="  ANN ").status_code == 303
    assert client.get(TOKENS).status_code == 200


def test_signin_goes_to_next_and_home_goes_to_tokens(ann):
    client = make_client()
    response = sign_in(client, next=TOKENS)
    assert response.headers["location"] == TOKENS
    home = client.get("/")
    assert home.status_code == 303
    assert home.headers["location"] == TOKENS


@pytest.mark.parametrize("target", ["//evil.example/x", "https://evil.example/", "\\evil", "x"])
def test_signin_next_off_site_goes_home(ann, target):
    response = sign_in(make_client(), next=target)
    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_signin_page_with_session_redirects(ann):
    client = make_client()
    sign_in(client)
    response = client.get(SIGNIN)
    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_wrong_password_and_unknown_user_are_refused_alike(ann):
    client = make_client()
    wrong = sign_in(client, password="not it")
    unknown = sign_in(client, name="bob")
    assert wrong.status_code == unknown.status_code == 400
    assert not session_cookie_header(wrong)
    assert not session_cookie_header(unknown)
    # Byte for byte the same page, but for the per-response style nonce.
    assert _without_nonce(wrong.text) == _without_nonce(unknown.text)
    assert "not it" not in wrong.text
    # The random sign-in form token and nonce could contain "bob" by chance.
    shown = re.sub(rf'name="{SIGNIN_FIELD}" value="[^"]*"', "", _without_nonce(unknown.text))
    assert "bob" not in shown
    assert client.get(TOKENS).status_code == 303


def _without_nonce(html: str) -> str:
    return re.sub(r'nonce="[^"]*"', "", html)


def test_wrong_password_opens_no_session(ann, db_path):
    sign_in(make_client(), password="not it")
    with open_store(db_path) as store:
        count = store.connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    assert count == 0


def test_signin_failure_never_logs_the_password(ann, caplog):
    caplog.set_level("DEBUG")
    sign_in(make_client(), password="hunter2-secret")
    assert "hunter2-secret" not in caplog.text


# --- the store with no users --------------------------------------------------

def test_empty_store_signin_page_says_to_add_a_user(db_path):
    response = make_client().get(SIGNIN)
    assert response.status_code == 200
    assert "voicemanager user add" in response.text


def test_signin_page_with_users_has_no_add_user_hint(ann):
    response = make_client().get(SIGNIN)
    assert response.status_code == 200
    assert "voicemanager user add" not in response.text
    assert 'action="/signin"' in response.text
    assert 'method="post"' in response.text


# --- sign-out ---------------------------------------------------------------------

def test_signout_ends_the_session_and_clears_the_cookie(ann):
    client = make_client()
    sign_in(client)
    secret = client.cookies.get(SESSION_COOKIE)
    assert secret

    response = client.post(SIGNOUT, data={SESSION_FIELD: secret})
    assert response.status_code == 303
    assert response.headers["location"] == SIGNIN
    cleared = session_cookie_header(response)
    assert cleared
    assert "max-age=0" in cookie_attrs(cleared)
    assert client.get(TOKENS).status_code == 303

    # The old cookie, sent again by hand, no longer opens anything.
    replay = make_client()
    replay.cookies.set(SESSION_COOKIE, secret)
    response = replay.get(TOKENS)
    assert response.status_code == 303
    assert response.headers["location"].startswith(SIGNIN)


def test_signout_without_session_field_ends_nothing(ann):
    client = make_client()
    sign_in(client)
    refused = client.post(SIGNOUT)
    assert refused.status_code == 403
    wrong = client.post(SIGNOUT, data={SESSION_FIELD: "wrong"})
    assert wrong.status_code == 403
    assert client.get(TOKENS).status_code == 200


def test_signout_without_session_redirects_to_signin(ann):
    response = make_client().post(SIGNOUT, data={SESSION_FIELD: "anything"})
    assert response.status_code == 303
    assert response.headers["location"] == SIGNIN


def test_signed_in_page_has_a_signout_form_echoing_the_session(ann):
    client = make_client()
    sign_in(client)
    secret = client.cookies.get(SESSION_COOKIE)
    page = client.get(TOKENS)
    assert f'action="{SIGNOUT}"' in page.text
    assert f'name="{SESSION_FIELD}" value="{secret}"' in page.text
    assert "private" in page.headers["cache-control"]
    assert "no-store" in page.headers["cache-control"]


# --- the sliding session --------------------------------------------------------

def test_a_use_that_moves_the_expiry_sets_the_cookie_again(ann, db_path):
    client = make_client()
    sign_in(client)
    secret = client.cookies.get(SESSION_COOKIE)
    # Make the row old enough that the next use writes a new expiry.
    with open_store(db_path) as store:
        digest = accounts.hash_secret(secret)
        session = store.get_session(digest)
        store.update_session(digest, expires=session.expires,
                             written=session.written - accounts.SESSION_WRITE_EVERY - 1)

    response = client.get(TOKENS)
    assert response.status_code == 200
    header = session_cookie_header(response)
    assert header.startswith(f"{SESSION_COOKIE}={secret}")
    assert "httponly" in cookie_attrs(header)

    # A use inside the write interval leaves the cookie alone.
    again = client.get(TOKENS)
    assert not session_cookie_header(again)


# --- the content security policy ----------------------------------------------

def _pages(client_with_session: TestClient, client_without: TestClient):
    """Every page the shell serves, as (label, response)."""
    return [
        ("signin", client_without.get(SIGNIN)),
        ("signin-failed", sign_in(client_without, password="not it")),
        ("tokens", client_with_session.get(TOKENS)),
    ]


def test_every_page_has_a_strict_csp_and_no_script(ann):
    signed_in = make_client()
    sign_in(signed_in)
    for label, response in _pages(signed_in, make_client()):
        assert response.headers["content-type"].startswith("text/html"), label
        directives = csp_directives(response)
        policy = response.headers["content-security-policy"]
        assert directives.get("default-src") == "'none'", label
        assert directives.get("img-src") == "'self'", label
        assert "style-src 'nonce-" in policy, label
        assert directives.get("form-action") == "'self'", label
        assert directives.get("frame-ancestors") == "'none'", label
        assert "script-src" not in policy, label
        assert "<script" not in response.text.lower(), label
        # The one style block carries this response's nonce.
        assert f'<style nonce="{style_nonce(response)}">' in response.text, label


def test_empty_store_signin_page_has_a_strict_csp(db_path):
    response = make_client().get(SIGNIN)
    policy = response.headers["content-security-policy"]
    assert "default-src 'none'" in policy
    assert "script-src" not in policy
    assert "<script" not in response.text.lower()


def test_the_style_nonce_differs_per_response(ann):
    client = make_client()
    first, second = client.get(SIGNIN), client.get(SIGNIN)
    assert style_nonce(first) != style_nonce(second)

    sign_in(client)
    third, fourth = client.get(TOKENS), client.get(TOKENS)
    assert style_nonce(third) != style_nonce(fourth)


# --- the favicons and the header mark ---------------------------------------------

FAVICONS = {
    "/favicon.svg": "image/svg+xml",
    "/favicon.ico": "image/vnd.microsoft.icon",
    "/apple-touch-icon.png": "image/png",
}
# The five bars of the mark, as the header draws them.
MARK_RECTS = (
    '<rect x="2" y="14" width="4" height="4" rx="1"/>',
    '<rect x="8" y="10" width="4" height="12" rx="1"/>',
    '<rect x="14" y="6" width="4" height="20" rx="1"/>',
    '<rect x="20" y="10" width="4" height="12" rx="1"/>',
    '<rect x="26" y="14" width="4" height="4" rx="1"/>',
)


@pytest.mark.parametrize("url", FAVICONS)
def test_each_favicon_is_served_without_a_session(ann, url):
    response = make_client().get(url)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith(FAVICONS[url])


def test_the_favicon_files_hold_the_sizes_browsers_ask_for(ann):
    client = make_client()
    ico = client.get("/favicon.ico").content
    count = int.from_bytes(ico[4:6], "little")
    png = client.get("/apple-touch-icon.png").content
    assert sorted(ico[6 + 16 * i] for i in range(count)) == [16, 32, 48]
    assert int.from_bytes(png[16:20], "big") == int.from_bytes(png[20:24], "big") == 180


def test_every_page_links_the_favicons_and_draws_the_mark(ann):
    signed_in = make_client()
    sign_in(signed_in)
    for label, response in _pages(signed_in, make_client()):
        html = response.text
        assert '<link rel="icon" href="/favicon.svg" type="image/svg+xml">' in html, label
        assert '<link rel="icon" href="/favicon.ico" sizes="16x16 32x32 48x48">' in html, label
        assert '<link rel="apple-touch-icon" href="/apple-touch-icon.png">' in html, label
        brand = re.search(r'<a class="brand" href="/">(.*?)</a>', html, re.S).group(1)
        for rect in MARK_RECTS:
            assert rect in brand, label


def test_the_favicon_svg_is_the_tile_with_the_header_mark(ann):
    svg = make_client().get("/favicon.svg").text
    for rect in MARK_RECTS:
        assert rect in svg


def test_the_api_routes_are_still_served(ann):
    # The web shell sits beside the API, not in front of it.
    response = make_client().get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
