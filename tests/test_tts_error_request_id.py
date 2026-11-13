"""TTS request error frames name the request they end; other error frames do not change.

A tts_stream request that fails ends with an error frame (tts_unavailable or
unknown_voice) and then audio_end reason "error". The error frame carries the
request_id of that request, so a client ties the frame to that request's audio_end
by id and not by wire order. The audio.cpp frames are covered in
tests/test_audiocpp_tts_stream_failure.py; this file covers the VoXtream
unknown_voice frame, over the pipeline and over the WebSocket.

Every other error frame the server sends stays the same on the wire: it has no
request_id key, not even a null one.

Model-free and server-free: the VoXtream engine talks to the fake sidecar
(tests/voxtream_fake_sidecar.py) on a loopback port, an unknown voice fails before
any connection to it, and the WS cases run the real receive loop over a TestClient.
"""
import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import authkit  # test auth env; must be imported before voicemanager
import pytest
from fastapi.testclient import TestClient

from voicemanager import protocol
from voicemanager.app import create_app
from voicemanager.pipeline.tts import TTSPipeline
from voicemanager.pipeline.voxtream_tts import (
    VoxtreamSidecar, VoxtreamTTSEngine, VoxtreamTTSStreamPipeline)
from voicemanager.session import SessionRegistry
from voicemanager.transport import websocket as ws_module
from voxtream_fake_sidecar import FakeVoxtreamSidecar

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_REFS = REPO_ROOT / "voices" / "designed" / "refs.json"


@pytest.fixture
def fake():
    sidecar = FakeVoxtreamSidecar().start()
    try:
        yield sidecar
    finally:
        sidecar.stop()


def _engine(fake: FakeVoxtreamSidecar) -> VoxtreamTTSEngine:
    engine = VoxtreamTTSEngine(VoxtreamSidecar(sys.executable, fake.port),
                               default_voice="designed_female_s101")
    engine.load(voices_path=REAL_REFS, start_sidecar=False)
    return engine


class _WS:
    """Captures the JSON frames a synthesis loop sends; PCM is recorded as 'pcm'."""

    def __init__(self) -> None:
        self.frames: list = []

    async def send_text(self, data: str) -> None:
        self.frames.append(json.loads(data))

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append("pcm")


class _InstantSTT:
    """Minimal QwenSTTPipeline stand-in: the receive loop needs an STT pipeline."""

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


def _tts_app(engine, executor):
    app = create_app()
    app.state.stt_engine_type = "qwen"  # swapped for _InstantSTT via monkeypatch
    app.state.qwen_model = object()
    app.state.tts_engine_type = "voxtream"
    app.state.tts_engine = engine
    app.state.tts_executor = executor
    app.state.session_registry = SessionRegistry()
    return app


# --- the VoXtream unknown_voice frame names its request ----------------------------

def test_voxtream_unknown_voice_frame_carries_the_request_id(fake) -> None:
    ws = _WS()

    async def run() -> None:
        pipe = VoxtreamTTSStreamPipeline(
            TTSPipeline(engine=_engine(fake), executor=None))
        await pipe.feed_text("hello", True, ws, "r-bad", voice="nope")
        await asyncio.wait_for(pipe.synthesis_task, 3.0)

    asyncio.run(run())

    assert [f["type"] for f in ws.frames] == ["error", "audio_start", "audio_end"]
    error = ws.frames[0]
    assert error["error_type"] == "unknown_voice"
    assert error["request_id"] == "r-bad"
    assert ws.frames[2]["request_id"] == "r-bad"
    assert ws.frames[2]["reason"] == "error"
    assert fake.connections == 0


def test_voxtream_unknown_voice_over_the_endpoint_carries_the_request_id(
        fake, monkeypatch) -> None:
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    engine = _engine(fake)
    with ThreadPoolExecutor(max_workers=1) as executor:
        client = TestClient(_tts_app(engine, executor))
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "tts_stream", "text": "hello", "done": True,
                          "request_id": "bad", "voice": "nope"})
            frames = [ws.receive_json() for _ in range(3)]
    assert [f["type"] for f in frames] == ["error", "audio_start", "audio_end"]
    assert frames[0]["error_type"] == "unknown_voice"
    assert frames[0]["request_id"] == "bad"
    assert frames[2]["request_id"] == "bad"


# --- every other error frame is unchanged on the wire ------------------------------

def test_a_plain_error_frame_has_no_request_id_key() -> None:
    frame = json.loads(protocol.ErrorMessage(
        error_type="timeout", message="Session closed due to inactivity.").model_dump_json())
    assert frame == {"type": "error", "error_type": "timeout",
                     "message": "Session closed due to inactivity."}
    assert "request_id" not in protocol.ErrorMessage.model_fields


def test_the_request_error_frame_requires_a_request_id_and_keeps_the_error_shape() -> None:
    frame = json.loads(protocol.RequestErrorMessage(
        error_type="unknown_voice", message="m", request_id="r1").model_dump_json())
    assert frame == {"type": "error", "error_type": "unknown_voice", "message": "m",
                     "request_id": "r1"}
    assert protocol.RequestErrorMessage.model_fields["request_id"].is_required()
    assert issubclass(protocol.RequestErrorMessage, protocol.ErrorMessage)


def test_an_invalid_tts_stream_frame_error_has_no_request_id_key(fake, monkeypatch) -> None:
    """The bad-chunking error names no request, even though the dropped frame had one."""
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    engine = _engine(fake)
    with ThreadPoolExecutor(max_workers=1) as executor:
        client = TestClient(_tts_app(engine, executor))
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "tts_stream", "text": "hello", "done": True,
                          "request_id": "r1", "chunking": "bogus"})
            error = ws.receive_json()
    assert error["type"] == "error"
    assert error["error_type"] == "invalid_request"
    assert set(error) == {"type", "error_type", "message"}


def test_only_the_tts_request_failures_build_a_request_error_frame() -> None:
    """The request error frame is built in the two TTS stream pipelines and nowhere else."""
    src = REPO_ROOT / "src" / "voicemanager"
    users = sorted(p.relative_to(src).as_posix() for p in src.rglob("*.py")
                   if any("RequestErrorMessage(" in line
                          and not line.lstrip().startswith("class ")
                          for line in p.read_text().splitlines()))
    assert users == ["pipeline/audiocpp_tts.py", "pipeline/voxtream_tts.py"]
