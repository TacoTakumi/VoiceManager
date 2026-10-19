"""Session ownership: each live session records the user whose token opened it.

/info lists only the caller's own sessions, and POST /voice/upload for another
user's session answers exactly like an unknown session id (404).

Model-free and server-free: the app is driven through TestClient (no lifespan),
and both users and their tokens live in the suite's temporary auth store (see
authkit).
"""
import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

import authkit

from voicemanager.app import create_app
from voicemanager.auth import accounts
from voicemanager.auth.store import open_store
from voicemanager.session import SessionRegistry

OTHER_USER = "ownership-other"
OTHER_TOKEN_NAME = "ownership"

WAV = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 100


@pytest.fixture(scope="module")
def other_secret() -> str:
    """A bearer token secret owned by a second user (not authkit.USER)."""
    with open_store(authkit.STORE_PATH) as store:
        if store.get_user(OTHER_USER) is None:
            accounts.add_user(store, OTHER_USER, None)
        return accounts.mint_token(store, OTHER_USER, f"{OTHER_TOKEN_NAME}-{uuid.uuid4().hex}")


@pytest.fixture
def app_and_registry():
    app = create_app()
    registry = SessionRegistry()
    app.state.session_registry = registry
    return app, registry


def _open(client: TestClient, secret: str):
    """Open /ws/stream as the owner of ``secret``."""
    return client.websocket_connect("/ws/stream", headers=authkit.bearer_headers(secret))


def _ready(ws) -> str:
    """Read the handshake; return the session id once the session is registered.

    A turn_start round trip is the barrier: the handler only reads frames after it
    has registered the session.
    """
    session_id = ws.receive_json()["session_id"]
    ws.send_json({"type": "turn_start"})
    assert ws.receive_json()["type"] == "turn_started"
    return session_id


def _info_ids(client: TestClient, secret: str) -> list[str]:
    resp = client.get("/info", headers=authkit.bearer_headers(secret))
    assert resp.status_code == 200
    return [s["session_id"] for s in resp.json()["sessions"]]


def _upload(client: TestClient, secret: str, session_id: str):
    return client.post(
        "/voice/upload",
        params={"session_id": session_id},
        headers=authkit.bearer_headers(secret),
        files={"file": ("test.wav", WAV, "audio/wav")},
    )


def test_each_session_records_its_token_owner(app_and_registry, other_secret):
    app, registry = app_and_registry
    client = TestClient(app)
    with _open(client, authkit.TOKEN) as ws_a, _open(client, other_secret) as ws_b:
        a_id = _ready(ws_a)
        b_id = _ready(ws_b)
        entry_a = asyncio.run(registry.get(a_id))
        entry_b = asyncio.run(registry.get(b_id))
    assert entry_a.owner == authkit.USER
    assert entry_b.owner == OTHER_USER


def test_info_lists_only_the_callers_sessions(app_and_registry, other_secret):
    app, _ = app_and_registry
    client = TestClient(app)
    with _open(client, authkit.TOKEN) as ws_a, _open(client, other_secret) as ws_b:
        a_id = _ready(ws_a)
        b_id = _ready(ws_b)
        assert _info_ids(client, authkit.TOKEN) == [a_id]
        assert _info_ids(client, other_secret) == [b_id]


def test_upload_to_another_users_session_is_an_unknown_session(app_and_registry, other_secret):
    app, _ = app_and_registry
    client = TestClient(app)
    with _open(client, authkit.TOKEN) as ws_a, _open(client, other_secret) as ws_b:
        _ready(ws_a)
        b_id = _ready(ws_b)
        cross = _upload(client, authkit.TOKEN, b_id)
        unknown = _upload(client, authkit.TOKEN, str(uuid.uuid4()))
    assert cross.status_code == 404
    assert unknown.status_code == 404
    assert cross.content == unknown.content
    assert cross.headers.get("content-type") == unknown.headers.get("content-type")


def test_upload_to_own_session_gets_past_the_session_check(app_and_registry, other_secret):
    """The owner's own session is found: the upload fails later, on the unloaded engine."""
    app, _ = app_and_registry
    client = TestClient(app)
    with _open(client, other_secret) as ws_b:
        b_id = _ready(ws_b)
        own = _upload(client, other_secret, b_id)
    assert own.status_code == 503
    assert own.json()["detail"] == "TTS engine not loaded"
