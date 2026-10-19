"""Bearer-token auth on /ws/stream and the REST endpoints.

Model-free and server-free: the app is driven through TestClient (no lifespan)
and the tokens live in the suite's temporary auth store (see authkit).

A WebSocket rejection is a close before accept. TestClient reports it as
WebSocketDisconnect with the close code; an ASGI server (uvicorn) answers the
same close-before-accept with HTTP 403 on the upgrade request, so the ASGI-level
tests below assert that the app sends nothing but that close.
"""
import ast
import asyncio
import logging
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.websockets import WebSocketDisconnect

import authkit

from voicemanager.app import create_app
from voicemanager.auth import accounts, deps
from voicemanager.session import SessionRegistry

SRC = Path(deps.__file__).resolve().parents[1]

WAV = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 100


@pytest.fixture
def client():
    app = create_app()
    app.state.session_registry = SessionRegistry()
    return TestClient(app)


@pytest.fixture
def revoked_secret():
    secret = authkit.mint("revoked-for-test")
    authkit.revoke("revoked-for-test")
    return secret


def _rejections(revoked: str) -> dict[str, tuple[dict, dict]]:
    """Each rejected credential shape: (headers, query params)."""
    return {
        "missing header": ({}, {}),
        "wrong scheme": ({"Authorization": f"Token {authkit.TOKEN}"}, {}),
        "basic scheme": ({"Authorization": "Basic dGVzdGVyOnB3"}, {}),
        "empty bearer": ({"Authorization": "Bearer "}, {}),
        "unknown token": (authkit.bearer_headers(authkit.UNKNOWN_TOKEN), {}),
        "revoked token": (authkit.bearer_headers(revoked), {}),
        "legacy query only": ({}, authkit.legacy_query(authkit.TOKEN)),
        "token query only": ({}, {"token": authkit.TOKEN}),
    }


def _asgi_ws_attempt(app, headers: dict, params: dict) -> list[dict]:
    """Drive one /ws/stream connect at the ASGI layer; the messages the app sent."""
    from urllib.parse import urlencode

    sent: list[dict] = []
    connected = False

    async def receive():
        nonlocal connected
        if not connected:
            connected = True
            return {"type": "websocket.connect"}
        return {"type": "websocket.disconnect", "code": 1000}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": "ws",
        "path": "/ws/stream",
        "raw_path": b"/ws/stream",
        "query_string": urlencode(params).encode(),
        "root_path": "",
        "headers": [(b"host", b"testserver")]
                   + [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "subprotocols": [],
    }
    asyncio.run(app(scope, receive, send))
    return sent


# --- /ws/stream -----------------------------------------------------------------

def test_ws_valid_bearer_gets_the_ready_handshake(client):
    with client.websocket_connect("/ws/stream", headers=authkit.auth_headers()) as ws:
        assert ws.receive_json()["type"] == "ready"


def test_ws_freshly_minted_token_is_accepted(client):
    secret = authkit.mint("fresh-for-test")
    with client.websocket_connect("/ws/stream", headers=authkit.bearer_headers(secret)) as ws:
        assert ws.receive_json()["type"] == "ready"


@pytest.mark.parametrize("case", list(_rejections("x")))
def test_ws_rejects_before_accept(client, revoked_secret, case):
    headers, params = _rejections(revoked_secret)[case]

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/stream", headers=headers, params=params) as ws:
            ws.receive_json()
    assert exc.value.code == 1008

    # Nothing but the close: no accept (so the upgrade becomes HTTP 403), no frame.
    sent = _asgi_ws_attempt(client.app, headers, params)
    assert [m["type"] for m in sent] == ["websocket.close"]
    assert sent[0]["code"] == 1008


def test_ws_valid_bearer_is_accepted_at_the_asgi_layer(client):
    sent = _asgi_ws_attempt(client.app, authkit.auth_headers(), {})
    assert sent[0]["type"] == "websocket.accept"
    assert sent[1]["type"] == "websocket.send"


# --- REST ------------------------------------------------------------------------

def test_health_is_open(client):
    assert client.get("/health").status_code == 200


def test_info_succeeds_with_bearer(client):
    resp = client.get("/info", headers=authkit.auth_headers())
    assert resp.status_code == 200
    assert "sessions" in resp.json()


def test_voice_upload_passes_auth_with_bearer(client):
    resp = client.post(
        "/voice/upload",
        params={"session_id": "nonexistent"},
        headers=authkit.auth_headers(),
        files={"file": ("test.wav", WAV, "audio/wav")},
    )
    # Past auth: the unknown session is what stops it.
    assert resp.status_code == 404


@pytest.mark.parametrize("case", list(_rejections("x")))
def test_info_returns_401(client, revoked_secret, case):
    headers, params = _rejections(revoked_secret)[case]
    resp = client.get("/info", headers=headers, params=params)
    assert resp.status_code == 401
    assert resp.headers.get("www-authenticate") == "Bearer"


@pytest.mark.parametrize("case", list(_rejections("x")))
def test_voice_upload_returns_401(client, revoked_secret, case):
    headers, params = _rejections(revoked_secret)[case]
    resp = client.post(
        "/voice/upload",
        params={**params, "session_id": "nonexistent"},
        headers=headers,
        files={"file": ("test.wav", WAV, "audio/wav")},
    )
    assert resp.status_code == 401


# --- bearer parsing ----------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (None, None),
    ("", None),
    ("Bearer ", None),
    ("Bearer    ", None),
    ("Token abc", None),
    ("bearerabc", None),
    ("Bearer abc", "abc"),
    ("Bearer  abc ", "abc"),
])
def test_bearer_secret(value, expected):
    assert deps.bearer_secret(value) == expected


# --- the lookup runs off the event loop -----------------------------------------

def _record_lookup_threads(monkeypatch) -> list[tuple[int, bool]]:
    """Patch the store lookup to record (thread id, has a running loop) per call."""
    calls: list[tuple[int, bool]] = []
    real = accounts.token_for

    def recording(store, secret, **kwargs):
        try:
            asyncio.get_running_loop()
            has_loop = True
        except RuntimeError:
            has_loop = False
        calls.append((threading.get_ident(), has_loop))
        return real(store, secret, **kwargs)

    monkeypatch.setattr(accounts, "token_for", recording)
    return calls


def test_rest_token_lookup_runs_off_the_event_loop_thread(monkeypatch):
    calls = _record_lookup_threads(monkeypatch)
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/info",
        "headers": [(b"authorization", f"Bearer {authkit.TOKEN}".encode())],
        "query_string": b"",
    }

    async def run():
        return threading.get_ident(), await deps.require_token(Request(scope))

    loop_thread, token = asyncio.run(run())

    assert token.owner == authkit.USER
    assert token.name == authkit.TOKEN_NAME
    assert len(calls) == 1
    assert calls[0][0] != loop_thread
    assert calls[0][1] is False


def test_ws_token_lookup_runs_off_the_event_loop(client, monkeypatch):
    calls = _record_lookup_threads(monkeypatch)

    with client.websocket_connect("/ws/stream", headers=authkit.auth_headers()) as ws:
        assert ws.receive_json()["type"] == "ready"

    assert len(calls) == 1
    assert calls[0][1] is False, "the lookup ran on a thread with a running event loop"


def _calls_in(tree: ast.AST) -> list[tuple[ast.Call, str]]:
    """Each call in ``tree`` with the name it calls (the last dotted part)."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            found.append((node, name))
    return found


def test_store_lookup_only_runs_through_run_in_threadpool():
    tree = ast.parse((SRC / "auth" / "deps.py").read_text())
    functions = {n.name: n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}

    # The blocking work (open the store, look the token up) lives in lookup_token.
    blocking = {"token_for", "open_store"}
    lookup_calls = {name for _, name in _calls_in(functions["lookup_token"])}
    assert blocking <= lookup_calls
    for name, func in functions.items():
        if name == "lookup_token":
            continue
        assert not blocking & {n for _, n in _calls_in(func)}, name

    # lookup_token is never called directly, only handed to run_in_threadpool.
    calls = _calls_in(tree)
    assert not [c for c, name in calls if name == "lookup_token"]
    handed = [c for c, name in calls if name == "run_in_threadpool"]
    assert handed
    for call in handed:
        assert isinstance(call.args[0], ast.Name) and call.args[0].id == "lookup_token"

    # The transports go through deps and never touch the store themselves.
    for rel in ("transport/http.py", "transport/websocket.py"):
        names = {name for _, name in _calls_in(ast.parse((SRC / rel).read_text()))}
        assert not names & (blocking | {"lookup_token"}), rel


# --- secrets stay out of the log ----------------------------------------------------

def test_token_secrets_are_never_logged(client, revoked_secret, caplog):
    caplog.set_level(logging.DEBUG)
    with client.websocket_connect("/ws/stream", headers=authkit.auth_headers()) as ws:
        ws.receive_json()
    for headers, params in _rejections(revoked_secret).values():
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/stream", headers=headers, params=params):
                pass
        client.get("/info", headers=headers, params=params)

    # The server's own log only: the test client (httpx) logs the URLs it requests.
    server_log = "\n".join(r.getMessage() for r in caplog.records
                           if r.name.startswith("voicemanager"))
    assert server_log, "expected the server to log the rejections"
    assert authkit.TOKEN not in server_log
    assert revoked_secret not in server_log
