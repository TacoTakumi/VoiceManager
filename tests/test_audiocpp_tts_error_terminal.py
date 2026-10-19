"""A one-shot tts failing AFTER audio_start still terminates on the wire.

TTSPipeline.synthesize_and_stream sends audio_start BEFORE it consumes the
engine, so any synthesis failure after that point used to leave the client with
an opened audio stream and eternal silence - no audio_end, no error, nothing.
Now the pipeline guarantees exactly ONE terminal frame for the failed
request_id: audio_end with reason="error", sent engine-agnostically from
pipeline/tts.py, after which the original exception still propagates (the
session's done-callback logs it; a SidecarFault's session-close path is
unchanged - the terminal send is best-effort on a socket that may be gone).

Model-free and server-free (the CLAUDE.md invariant): the real AudioCppTTSEngine
speaking the real wire protocol to the fake TTS sidecar
(tests/audiocpp_tts_fake.py) over a genuine AF_UNIX socket. Both failure
classes the review named are pinned:

  * a refusal-class SidecarError - the sidecar answers TEXT with an ERROR frame
    (the fake's text_error script) - which lands mid-request by definition;
  * UnknownVoiceError - raised inside the engine's generator on first pull,
    which is AFTER the pipeline already sent audio_start.

The happy path stays byte-compatible: exactly one audio_start, the PCM, and one
audio_end reason="complete" - covered by the existing tts WS tests; the WS-level
test here additionally proves the failed request's terminal frame reaches a real
client and the SESSION survives to serve the next request.
"""
from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import authkit  # test auth env; must be imported before voicemanager

import pytest
from fastapi.testclient import TestClient

from voicemanager.app import create_app
from voicemanager.pipeline.audiocpp_tts import (
    AudioCppTTSEngine,
    SidecarError,
    UnknownVoiceError,
)
from voicemanager.pipeline.tts import TTSPipeline
from voicemanager.transport import websocket as ws_module

from audiocpp_tts_fake import FakeTtsSidecar

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_REFS = REPO_ROOT / "voices" / "designed" / "refs.json"


class _RecordingSocket:
    """Duck-typed WebSocket: records the JSON frames and PCM sends in order."""

    def __init__(self) -> None:
        self.frames: list = []

    async def send_text(self, text: str) -> None:
        self.frames.append(json.loads(text))

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append("pcm")


def _external_engine(fake: FakeTtsSidecar) -> AudioCppTTSEngine:
    engine = AudioCppTTSEngine({"socket": fake.socket_path})
    engine.load(voices_path=str(REAL_REFS))
    return engine


def _pipeline_for(engine: AudioCppTTSEngine) -> TTSPipeline:
    # executor=None is deliberate: the audiocpp engine is non-batch, so the
    # pipeline consumes its generator on the loop and never touches the executor.
    return TTSPipeline(engine=engine, executor=None)


def _assert_exactly_one_error_terminal(ws: _RecordingSocket, request_id: str) -> None:
    assert ws.frames, "audio_start must have been sent before the failure"
    assert ws.frames[0]["type"] == "audio_start"
    assert ws.frames[0]["request_id"] == request_id
    terminals = [f for f in ws.frames[1:] if f != "pcm"]
    assert len(terminals) == 1, (
        f"exactly ONE terminal frame must follow audio_start, got {terminals!r} "
        "- silence (none) is the old defect; two is a double terminal")
    terminal = terminals[0]
    assert terminal["type"] == "audio_end", terminal
    assert terminal["request_id"] == request_id
    assert terminal["reason"] == "error", terminal


# --- pipeline level: both failure classes terminate on the wire ----------------------

def test_refusal_error_after_audio_start_sends_one_error_terminal() -> None:
    with FakeTtsSidecar(text_error="refused") as fake:
        engine = _external_engine(fake)
        ws = _RecordingSocket()
        try:
            with pytest.raises(SidecarError):
                asyncio.run(_pipeline_for(engine).synthesize_and_stream(
                    ws, "doomed by refusal", "r-refused"))
        finally:
            engine.dispose()
        _assert_exactly_one_error_terminal(ws, "r-refused")
        assert "pcm" not in ws.frames, "the refused request must deliver no audio"


def test_unknown_voice_after_audio_start_sends_one_error_terminal() -> None:
    with FakeTtsSidecar() as fake:
        engine = _external_engine(fake)
        ws = _RecordingSocket()
        try:
            with pytest.raises(UnknownVoiceError):
                asyncio.run(_pipeline_for(engine).synthesize_and_stream(
                    ws, "doomed by voice", "r-voice", voice="no-such-voice"))
        finally:
            engine.dispose()
        _assert_exactly_one_error_terminal(ws, "r-voice")
        assert fake.requests == [], "an unknown voice must never reach the sidecar"


# --- WS level: the terminal reaches a real client and the session survives ----------

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
    app.state.tts_engine_type = "audiocpp"
    app.state.tts_engine = engine
    app.state.tts_executor = executor
    return app


def test_ws_refused_request_gets_error_terminal_and_session_serves_the_next(
        monkeypatch) -> None:
    """The fake refuses only the FIRST TEXT, so r1 fails mid-request and r2 on
    the SAME session must then synthesize normally (refusals are request-scoped,
    not session-fatal)."""
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    with FakeTtsSidecar(text_error="refused") as fake:
        engine = _external_engine(fake)
        with ThreadPoolExecutor(max_workers=1) as executor:
            client = TestClient(_tts_app(engine, executor))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"

                ws.send_json({"type": "tts", "text": "doomed", "request_id": "r1"})
                start = ws.receive_json()
                assert start["type"] == "audio_start" and start["request_id"] == "r1"
                terminal = ws.receive_json()
                assert terminal["type"] == "audio_end", terminal
                assert terminal["request_id"] == "r1"
                assert terminal["reason"] == "error"

                ws.send_json({"type": "tts", "text": "clean follow-up",
                              "request_id": "r2"})
                start2 = ws.receive_json()
                assert start2["type"] == "audio_start" and start2["request_id"] == "r2"
                saw_pcm = False
                while True:
                    frame = ws.receive()
                    if frame.get("bytes") is not None:
                        saw_pcm = True
                        continue
                    end2 = json.loads(frame["text"])
                    break
                assert end2["type"] == "audio_end" and end2["request_id"] == "r2"
                assert end2["reason"] == "complete"
                assert saw_pcm, "the follow-up request must actually deliver audio"

        assert [r["text"] for r in fake.requests] == ["doomed", "clean follow-up"]
