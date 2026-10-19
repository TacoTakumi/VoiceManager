"""A token's stored voice is the connection's default voice for TTS.

The voice for a TTS request is, in order: the request's 'voice' field; else the
token's voice when the running engine has it; else the engine's configured
default. A stale token voice falls back to the default with one warning per
connection, and the handshake reports the resolved default as 'default_voice'
(null under OpenVoice, which has no presets).

Model-free and server-free: a fake preset engine on app.state records the voice
each synthesis was called with, the STT pipeline is a stand-in, and TestClient
runs without its lifespan, so no model loads.
"""
import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

import authkit  # test auth env; must be imported before voicemanager

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voicemanager.app import create_app
from voicemanager.session import SessionRegistry
from voicemanager.transport import websocket as ws_module

VOICES = ["alpha", "beta", "gamma"]
DEFAULT = "alpha"
WS_LOGGER = "voicemanager.transport.websocket"


class _FakePresetEngine:
    """Streaming preset engine: records the voice each synthesis resolved to.

    Like the real preset engines, no voice means the configured default and an
    unknown voice raises before any audio is produced.
    """

    sample_rate = 24000

    def __init__(self) -> None:
        self.available_voices = list(VOICES)
        self.default_voice = DEFAULT
        self.voices_used: list[str] = []

    async def stream_synthesize(self, text: str, **kwargs):
        voice = kwargs.get("voice") or self.default_voice
        if voice not in self.available_voices:
            raise ValueError(f"unknown voice {voice!r}")
        self.voices_used.append(voice)
        yield np.zeros(240, dtype=np.float32)


class _OpenVoiceLikeEngine:
    """No presets: no available_voices and no default_voice attribute."""

    sample_rate = 22050


class _InstantSTT:
    """Minimal STT stand-in: the receive loop needs an STT pipeline."""

    def __init__(self, asr_model=None, **kwargs):
        pass

    async def feed(self, data: bytes) -> bool:
        return False

    def get_current_text(self):
        return ("", "en", 0)

    async def finalize(self):
        return ("", "en", 0)

    def reset(self) -> None:
        pass

    def dispose(self) -> None:
        pass


def _app(monkeypatch, engine, engine_type: str):
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    app = create_app()
    app.state.stt_engine_type = "qwen"  # swapped for _InstantSTT above
    app.state.qwen_model = object()
    app.state.tts_engine_type = engine_type
    app.state.tts_engine = engine
    app.state.tts_executor = ThreadPoolExecutor(max_workers=1)
    app.state.session_registry = SessionRegistry()
    return app


@pytest.fixture
def engine() -> _FakePresetEngine:
    return _FakePresetEngine()


@pytest.fixture
def client(monkeypatch, engine) -> TestClient:
    return TestClient(_app(monkeypatch, engine, "vibevoice"))


def _token(voice: str | None) -> tuple[str, str]:
    """Mint a fresh token with ``voice``; its (name, secret)."""
    name = f"voice-{uuid.uuid4().hex[:8]}"
    return name, authkit.mint(name, voice=voice)


def _connect(client: TestClient, secret: str | None = None):
    headers = authkit.bearer_headers(secret) if secret else {}
    return authkit.ws_connect(client, "/ws/stream", headers=headers)


def _until_audio_end(ws) -> list[dict]:
    """The JSON frames of one TTS request, through its audio_end."""
    frames = []
    while True:
        msg = ws.receive()
        if msg.get("text") is None:
            continue  # PCM
        frame = json.loads(msg["text"])
        frames.append(frame)
        if frame.get("type") == "audio_end":
            return frames


def _tts(ws, request_id: str, voice: str | None = None) -> list[dict]:
    req = {"type": "tts", "text": "Hello there.", "request_id": request_id}
    if voice is not None:
        req["voice"] = voice
    ws.send_json(req)
    return _until_audio_end(ws)


def _tts_stream(ws, request_id: str, voice: str | None = None) -> list[dict]:
    req = {"type": "tts_stream", "text": "Hello there.", "request_id": request_id,
           "done": True}
    if voice is not None:
        req["voice"] = voice
    ws.send_json(req)
    return _until_audio_end(ws)


def _voice_warnings(caplog, name: str) -> list[logging.LogRecord]:
    return [r for r in caplog.records
            if r.name == WS_LOGGER and r.levelno == logging.WARNING
            and name in r.getMessage()]


# --- voice order ----------------------------------------------------------------

def test_request_voice_wins_over_token_voice(client, engine) -> None:
    _, secret = _token("beta")
    with _connect(client, secret) as ws:
        ws.receive_json()
        _tts(ws, "r1", voice="gamma")
        _tts_stream(ws, "s1", voice="gamma")
    assert engine.voices_used == ["gamma", "gamma"]


def test_token_voice_used_when_request_names_none(client, engine) -> None:
    _, secret = _token("beta")
    with _connect(client, secret) as ws:
        ws.receive_json()
        _tts(ws, "r1")
        _tts_stream(ws, "s1")
    assert engine.voices_used == ["beta", "beta"]


def test_engine_default_used_without_request_or_token_voice(client, engine) -> None:
    with _connect(client) as ws:  # the suite token has no voice
        ws.receive_json()
        _tts(ws, "r1")
        _tts_stream(ws, "s1")
    assert engine.voices_used == [DEFAULT, DEFAULT]


def test_unknown_request_voice_gives_the_existing_error_frame(client, engine) -> None:
    with _connect(client) as ws:
        ws.receive_json()
        baseline = _tts(ws, "r1", voice="nope")
    _, secret = _token("beta")
    with _connect(client, secret) as ws:
        ws.receive_json()
        frames = _tts(ws, "r1", voice="nope")
    assert frames == baseline
    assert frames[-1]["type"] == "audio_end"
    assert frames[-1]["reason"] == "error"
    assert engine.voices_used == []


# --- stale token voice ------------------------------------------------------------

def test_stale_token_voice_falls_back_with_one_warning(client, engine, caplog) -> None:
    name, secret = _token("gone")
    caplog.set_level(logging.WARNING, logger=WS_LOGGER)
    with _connect(client, secret) as ws:
        handshake = ws.receive_json()
        _tts(ws, "r1")
        _tts(ws, "r2")
        _tts_stream(ws, "s1")
    assert handshake["default_voice"] == DEFAULT
    assert engine.voices_used == [DEFAULT, DEFAULT, DEFAULT]
    warnings = _voice_warnings(caplog, name)
    assert len(warnings) == 1
    assert "gone" in warnings[0].getMessage()
    assert all(secret not in r.getMessage() for r in caplog.records)


def test_stale_token_voice_warns_once_per_connection(client, caplog) -> None:
    name, secret = _token("gone")
    caplog.set_level(logging.WARNING, logger=WS_LOGGER)
    for _ in range(2):
        with _connect(client, secret) as ws:
            ws.receive_json()
    assert len(_voice_warnings(caplog, name)) == 2


def test_valid_token_voice_logs_no_warning(client, caplog) -> None:
    name, secret = _token("beta")
    caplog.set_level(logging.WARNING, logger=WS_LOGGER)
    with _connect(client, secret) as ws:
        ws.receive_json()
    assert _voice_warnings(caplog, name) == []


# --- handshake default_voice --------------------------------------------------------

def test_handshake_default_voice_is_the_token_voice(client) -> None:
    _, secret = _token("beta")
    with _connect(client, secret) as ws:
        assert ws.receive_json()["default_voice"] == "beta"


def test_handshake_default_voice_is_the_engine_default_without_token_voice(client) -> None:
    with _connect(client) as ws:
        assert ws.receive_json()["default_voice"] == DEFAULT


def test_handshake_default_voice_is_null_under_openvoice(monkeypatch, caplog) -> None:
    client = TestClient(_app(monkeypatch, _OpenVoiceLikeEngine(), "openvoice"))
    name, secret = _token("beta")
    caplog.set_level(logging.WARNING, logger=WS_LOGGER)
    with _connect(client, secret) as ws:
        handshake = ws.receive_json()
    assert "default_voice" in handshake
    assert handshake["default_voice"] is None
    assert _voice_warnings(caplog, name) == []
