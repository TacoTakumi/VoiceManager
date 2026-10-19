"""WebSocket protocol tests for auth, handshake, and audio validation."""
import asyncio
import json
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# Set env before importing app
import authkit

from voicemanager.app import create_app

@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)

def test_ws_rejects_missing_bearer_header(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws/stream"):
            pass
    assert exc.value.code == 1008

def test_ws_rejects_unknown_bearer_token(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect(
                "/ws/stream", headers=authkit.bearer_headers(authkit.UNKNOWN_TOKEN)):
            pass
    assert exc.value.code == 1008

def test_ws_bad_token_closes_before_accept(client):
    """A bad token is rejected during the handshake, before accept().

    The app never sends websocket.accept, so no WebSocket ever exists and the
    close code is not observable by the client: an ASGI server (uvicorn) turns
    a close-before-accept into an HTTP 403 on the handshake response. This test
    asserts at the ASGI layer because TestClient does not model that mapping.
    """
    sent = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "websocket",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": "ws",
        "path": "/ws/stream",
        "raw_path": b"/ws/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"authorization", f"Bearer {authkit.UNKNOWN_TOKEN}".encode()),
        ],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "subprotocols": [],
    }
    asyncio.run(client.app(scope, receive, send))

    assert [m["type"] for m in sent] == ["websocket.close"]


def test_ws_accepts_valid_bearer_token_and_sends_handshake(client):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        data = ws.receive_json()
        assert data["type"] == "ready"
        assert "session_id" in data
        assert len(data["session_id"]) == 36  # UUID format
        assert data["audio_format"]["sample_rate"] == 16000
        assert data["audio_format"]["channels"] == 1
        assert data["audio_format"]["bit_depth"] == 16
        assert data["audio_format"]["encoding"] == "pcm"

def test_ws_rejects_odd_byte_audio(client):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        ws.receive_json()  # consume handshake
        ws.send_bytes(b"\x01\x02\x03")  # odd = invalid PCM
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error_type"] == "invalid_audio_format"

def test_ws_accepts_valid_pcm_bytes(client):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        ws.receive_json()  # consume handshake
        ws.send_bytes(b"\x00\x01\x00\x02")  # even bytes = valid PCM
        # Should not raise — connection stays open

def test_ws_accepts_wav_header(client):
    # Minimal WAV-like header: RIFF....WAVE
    wav_header = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 20
    with authkit.ws_connect(client, "/ws/stream") as ws:
        ws.receive_json()  # consume handshake
        ws.send_bytes(wav_header)
        # Should not raise — connection stays open
