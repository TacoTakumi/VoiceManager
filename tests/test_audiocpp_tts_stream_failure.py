"""A tts_stream request whose audiocpp engine raises ends with an error the client can see.

When the TTS sidecar cannot load (for example a CUDA buffer that does not fit on a
card another session fills), every request on the session fails in the engine. The
client must not see audio_start, no audio and a normal end: it gets an error frame
of error_type tts_unavailable, then audio_end with reason "error", and the
connection stays open, because a later load can succeed.

The error message names the sidecar error code (load_failed) and never carries the
sidecar's free-form detail text (an allocator line, a path). A raise that has no
code gets a fixed message.

Model-free and server-free: the real engine and stream pipeline speak the real
wire protocol to the fake TTS sidecar (tests/audiocpp_tts_fake.py) over an
AF_UNIX socket, and the WS case runs the real receive loop over a TestClient.
"""
from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import authkit  # test auth env; must be imported before voicemanager

import pytest
from fastapi.testclient import TestClient

from voicemanager import protocol
from voicemanager.app import create_app
from voicemanager.pipeline.audiocpp_tts import (
    AudioCppTTSEngine,
    AudioCppTTSStreamPipeline,
    SidecarError,
)
from voicemanager.pipeline.tts import TTSPipeline
from voicemanager.transport import websocket as ws_module

from audiocpp_tts_fake import FakeTtsSidecar

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_REFS = REPO_ROOT / "voices" / "designed" / "refs.json"

#: The free-form detail a failed load sends: it must never reach the client.
ALLOCATOR_DETAIL = ("alloc_tensor_range: failed to allocate CUDA4 buffer of size "
                    "317411456")
SENTENCE = "The quick brown fox jumps over the lazy dog near the river bank."


class _FakeWS:
    """Captures the frames a synthesis loop sends: JSON dicts and the string 'pcm'."""

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


class _CodelessRaisingEngine:
    """An engine whose synthesis raises an error that carries no sidecar code."""

    sample_rate = 48000

    async def ensure_ready(self) -> None:
        return None

    async def stream_synthesize(self, text: str, **kwargs):
        raise RuntimeError(ALLOCATOR_DETAIL)
        yield  # pragma: no cover - makes this an async generator


def _external_engine(fake: FakeTtsSidecar) -> AudioCppTTSEngine:
    engine = AudioCppTTSEngine({"socket": fake.socket_path})
    engine.load(voices_path=str(REAL_REFS))
    return engine


def _stream_pipeline(engine) -> AudioCppTTSStreamPipeline:
    # executor=None: the audiocpp engine streams on the loop, never on an executor.
    return AudioCppTTSStreamPipeline(tts_pipeline=TTSPipeline(engine=engine, executor=None))


async def _run_request(stream, ws, request_id: str) -> BaseException | None:
    await stream.feed_text(chunk=SENTENCE, done=True, websocket=ws, request_id=request_id)
    task = stream.synthesis_task
    assert task is not None
    try:
        await asyncio.wait_for(task, timeout=10.0)
    except Exception as exc:  # the task re-raises the engine's error
        return exc
    return None


def _assert_failed_request(frames: list, request_id: str) -> dict:
    """audio_start, then the tts_unavailable error, then audio_end reason error."""
    assert "pcm" not in frames, "a failed load must deliver no audio"
    types = [f["type"] for f in frames]
    assert types == ["audio_start", "error", "audio_end"], frames
    assert frames[0]["request_id"] == request_id
    error = frames[1]
    assert error["error_type"] == protocol.TTS_UNAVAILABLE == "tts_unavailable"
    assert ALLOCATOR_DETAIL not in error["message"]
    assert "alloc_tensor_range" not in error["message"]
    assert frames[2]["request_id"] == request_id
    assert frames[2]["reason"] == "error", frames[2]
    return error


# --- pipeline level ----------------------------------------------------------------

def test_load_failure_sends_tts_unavailable_then_error_end_and_next_request_completes(
) -> None:
    with FakeTtsSidecar(hello_error="load_failed",
                        hello_error_message=ALLOCATOR_DETAIL) as fake:
        engine = _external_engine(fake)
        ws_failed = _FakeWS()
        ws_next = _FakeWS()

        async def scenario() -> tuple:
            stream = _stream_pipeline(engine)
            failed = await _run_request(stream, ws_failed, "r1")
            fake.hello_error = None  # the card has room again: the load works now
            ok = await _run_request(stream, ws_next, "r2")
            return failed, ok

        try:
            failed, ok = asyncio.run(scenario())
        finally:
            engine.dispose()

    assert isinstance(failed, SidecarError) and failed.code == "load_failed"
    error = _assert_failed_request(ws_failed.frames, "r1")
    assert "load_failed" in error["message"]

    assert ok is None
    assert ws_next.frames[0]["type"] == "audio_start"
    assert "pcm" in ws_next.frames
    assert not any(isinstance(f, dict) and f["type"] == "error" for f in ws_next.frames)
    assert ws_next.frames[-1]["type"] == "audio_end"
    assert ws_next.frames[-1]["reason"] == "complete"
    assert [r["text"] for r in fake.requests] == [SENTENCE]


def test_raise_without_a_code_gets_the_fixed_message() -> None:
    ws = _FakeWS()

    async def scenario():
        return await _run_request(_stream_pipeline(_CodelessRaisingEngine()), ws, "r1")

    failed = asyncio.run(scenario())

    assert isinstance(failed, RuntimeError)
    error = _assert_failed_request(ws.frames, "r1")
    assert error["message"] == "The TTS engine is unavailable. A later request may succeed."


@pytest.mark.parametrize("code", [ALLOCATOR_DETAIL, "", "x" * 65])
def test_a_code_that_is_not_a_short_token_is_not_named(code: str) -> None:
    """A sidecar that puts free text in the code field gets the fixed message."""

    class _Engine(_CodelessRaisingEngine):
        async def stream_synthesize(self, text: str, **kwargs):
            raise SidecarError(code, ALLOCATOR_DETAIL)
            yield  # pragma: no cover

    ws = _FakeWS()

    async def scenario():
        return await _run_request(_stream_pipeline(_Engine()), ws, "r1")

    asyncio.run(scenario())
    error = _assert_failed_request(ws.frames, "r1")
    assert error["message"] == "The TTS engine is unavailable. A later request may succeed."


# --- WS level: the connection stays open and serves the next request ---------------

def _tts_app(engine, executor):
    app = create_app()
    app.state.stt_engine_type = "qwen"  # swapped for _InstantSTT via monkeypatch
    app.state.qwen_model = object()
    app.state.tts_engine_type = "audiocpp"
    app.state.tts_engine = engine
    app.state.tts_executor = executor
    return app


def _read_request(ws) -> list:
    """Every frame of one tts_stream reply, up to and including its audio_end."""
    frames: list = []
    while True:
        frame = ws.receive()
        if frame.get("bytes") is not None:
            frames.append("pcm")
            continue
        payload = json.loads(frame["text"])
        frames.append(payload)
        if payload["type"] == "audio_end":
            return frames


def test_ws_failed_load_keeps_the_connection_open_for_the_next_request(
        monkeypatch) -> None:
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    with FakeTtsSidecar(hello_error="load_failed",
                        hello_error_message=ALLOCATOR_DETAIL) as fake:
        engine = _external_engine(fake)
        with ThreadPoolExecutor(max_workers=1) as executor:
            client = TestClient(_tts_app(engine, executor))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"

                ws.send_json({"type": "tts_stream", "text": SENTENCE,
                              "request_id": "r1", "done": True})
                error = _assert_failed_request(_read_request(ws), "r1")
                assert "load_failed" in error["message"]

                fake.hello_error = None
                ws.send_json({"type": "tts_stream", "text": SENTENCE,
                              "request_id": "r2", "done": True})
                frames = _read_request(ws)
                assert frames[0]["type"] == "audio_start"
                assert frames[0]["request_id"] == "r2"
                assert "pcm" in frames
                assert frames[-1]["request_id"] == "r2"
                assert frames[-1]["reason"] == "complete"

    assert [r["text"] for r in fake.requests] == [SENTENCE]
