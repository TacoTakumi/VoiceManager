"""The tts_stream idle flush is armed by tts_stream frames, not by any frame.

A flushed-but-never-closed ``tts_stream`` request is ended by the
VM_TTS_STREAM_FLUSH_TIMEOUT safety path. That path used to be a cap on the
receive wait, measured from the last frame of ANY kind. A full-duplex client
sends microphone audio every ~100 ms, so the wait never expired and the open
request stayed open for the whole session. The idle clock now runs from the last
``tts_stream`` frame of the open request: audio frames and other messages still
reset the session inactivity timer, but they do not re-arm the flush.

Model-free and server-free: ``TestClient(app)`` without ``with client:`` skips the
lifespan, the STT pipeline and the TTS engine are fakes, so no model loads, no
GPU, no network and no audio hardware. The timeouts are integers in seconds, so
the flush timeout is 1 s and every test is bounded by a few seconds.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import authkit  # test auth env; must be imported before voicemanager

import anyio
import numpy as np
import pytest
from fastapi.testclient import TestClient

import voicemanager.config as cfg
from voicemanager.app import create_app
from voicemanager.session import SessionRegistry

# 512 bytes: a valid raw 16-bit PCM frame for the first-frame check.
_PCM = b"\x00\x00" * 256

FLUSH_TIMEOUT = 1          # VM_TTS_STREAM_FLUSH_TIMEOUT (seconds, an int)
FRAME_INTERVAL = FLUSH_TIMEOUT / 10
# Scheduling slack on top of the timing bounds: the receive loop, the synthesis
# task and the test client all hand off through threads.
SLACK = 0.5


# --- test doubles -----------------------------------------------------------

class _FakeWhisper:
    """Drop-in for STTPipeline: swallows audio, never transcribes."""

    def __init__(self, **kwargs):
        pass

    def feed(self, data: bytes) -> bool:
        return False

    async def run_inference(self, loop):
        return [], None, 0.0

    def dispose(self):
        pass


class _FakeStreamEngine:
    """Streaming TTS engine stub: one 8-sample chunk per sentence, no executor work."""

    is_batch = False

    @property
    def sample_rate(self) -> int:
        return 24000

    async def stream_synthesize(self, text: str, **kwargs):
        yield np.zeros(8, dtype=np.float32)


def _app(executor):
    app = create_app()
    app.state.stt_engine_type = "whisper"
    app.state.session_registry = SessionRegistry()
    app.state.tts_engine_type = "openvoice"
    app.state.tts_engine = _FakeStreamEngine()
    app.state.tts_executor = executor
    return app


@pytest.fixture
def fast_flush(monkeypatch):
    monkeypatch.setattr("voicemanager.transport.websocket.STTPipeline", _FakeWhisper)
    monkeypatch.setenv("VM_TTS_STREAM_FLUSH_TIMEOUT", str(FLUSH_TIMEOUT))
    cfg._settings = None
    yield
    cfg._settings = None


# --- timed reads on the test socket ----------------------------------------

def _poll(ws) -> dict | None:
    """One queued inbound message, or None when nothing is waiting."""
    try:
        return ws.portal.call(ws._send_rx.receive_nowait)
    except anyio.WouldBlock:
        return None


def _drain(ws, seen: list) -> None:
    """Move every queued message into ``seen`` (text frames decoded)."""
    while True:
        msg = _poll(ws)
        if msg is None:
            return
        if msg.get("text") is not None:
            seen.append(json.loads(msg["text"]))
        else:
            seen.append(msg)


def _audio_ends(seen: list, request_id: str) -> list[dict]:
    return [f for f in seen if isinstance(f, dict) and f.get("type") == "audio_end"
            and f.get("request_id") == request_id]


def _closed(seen: list) -> bool:
    return any(isinstance(f, dict) and f.get("type") == "websocket.close" for f in seen)


def _stream_audio_until(ws, seen: list, stop, interval: float = FRAME_INTERVAL) -> None:
    """Send one audio frame per ``interval`` until ``stop(seen)`` is true or the deadline."""
    deadline = time.monotonic() + 3 * FLUSH_TIMEOUT
    while time.monotonic() < deadline:
        ws.send_bytes(_PCM)
        time.sleep(interval)
        _drain(ws, seen)
        if stop(seen):
            return


def _open_request(ws, request_id: str) -> float:
    """Flush a fragment on ``request_id`` without done; the time it was sent."""
    ws.send_json({"type": "tts_stream", "request_id": request_id,
                  "text": "Hello there", "flush": True})
    return time.monotonic()


def _wait_for_audio_end(ws, seen: list, request_id: str, timeout: float) -> float:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _drain(ws, seen)
        if _audio_ends(seen, request_id):
            return time.monotonic()
        time.sleep(0.01)
    raise AssertionError(f"no audio_end for {request_id!r} within {timeout}s: {seen}")


# --- the tests ---------------------------------------------------------------

def test_steady_audio_does_not_keep_a_flushed_request_open(fast_flush):
    # Open r1 with a flush and no done, then stream mic audio every tenth of the
    # flush timeout. The audio must not re-arm the flush: audio_end arrives within
    # one timeout plus one frame interval, and the socket still works afterwards.
    with ThreadPoolExecutor(max_workers=1) as executor:
        client = TestClient(_app(executor))
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            seen: list = []
            opened = _open_request(ws, "r1")
            _stream_audio_until(ws, seen, lambda s: bool(_audio_ends(s, "r1")))
            ends = _audio_ends(seen, "r1")
            assert len(ends) == 1, f"expected one audio_end for r1, saw {seen}"
            assert time.monotonic() - opened <= FLUSH_TIMEOUT + FRAME_INTERVAL + SLACK
            assert not _closed(seen)

            # The connection is still open and still serves TTS.
            ws.send_json({"type": "tts_stream", "request_id": "r2", "text": "Bye.",
                          "done": True})
            _wait_for_audio_end(ws, seen, "r2", timeout=2.0)
            assert not _closed(seen)


def test_tts_stream_frames_keep_the_request_open(fast_flush):
    # tts_stream frames for the open request every half timeout, with audio
    # between them, for three timeouts: the idle clock restarts on each, so no
    # audio_end is sent. Closing with done afterwards still ends the SAME request.
    with ThreadPoolExecutor(max_workers=1) as executor:
        client = TestClient(_app(executor))
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            seen: list = []
            _open_request(ws, "r1")
            start = time.monotonic()
            next_chunk = start + FLUSH_TIMEOUT / 2
            while time.monotonic() - start < 3 * FLUSH_TIMEOUT:
                ws.send_bytes(_PCM)
                time.sleep(FRAME_INTERVAL)
                if time.monotonic() >= next_chunk:
                    ws.send_json({"type": "tts_stream", "request_id": "r1",
                                  "text": " and"})
                    next_chunk += FLUSH_TIMEOUT / 2
                _drain(ws, seen)
                assert not _audio_ends(seen, "r1"), f"request closed early: {seen}"

            ws.send_json({"type": "tts_stream", "request_id": "r1", "text": " more.",
                          "done": True})
            _wait_for_audio_end(ws, seen, "r1", timeout=2.0)
            texts = [f for f in seen if isinstance(f, dict)]
            assert [f["type"] for f in texts if f["type"] == "audio_start"] == ["audio_start"]
            assert len(_audio_ends(seen, "r1")) == 1
            assert _audio_ends(seen, "r1")[0]["reason"] == "complete"


def test_idle_flush_fires_after_about_one_timeout(fast_flush):
    # Only audio after the open: audio_end comes no earlier than the timeout
    # (within tolerance) and no later than the timeout plus one frame interval.
    with ThreadPoolExecutor(max_workers=1) as executor:
        client = TestClient(_app(executor))
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            seen: list = []
            opened = _open_request(ws, "r1")
            _stream_audio_until(ws, seen, lambda s: bool(_audio_ends(s, "r1")))
            ended = time.monotonic()
            assert len(_audio_ends(seen, "r1")) == 1, f"saw {seen}"
            elapsed = ended - opened
            # The drain runs once per frame interval, so the lower bound allows one
            # interval of lateness in the measurement, not in the flush itself.
            assert elapsed >= FLUSH_TIMEOUT - FRAME_INTERVAL, elapsed
            assert elapsed <= FLUSH_TIMEOUT + FRAME_INTERVAL + SLACK, elapsed


def test_open_request_with_steady_audio_does_not_shorten_the_session(
        fast_flush, monkeypatch):
    # The session inactivity timeout is still reset by audio frames: an open
    # request must not turn the shorter flush cap into a session close.
    session_timeout = 3
    monkeypatch.setenv("VM_SESSION_TIMEOUT", str(session_timeout))
    cfg._settings = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        client = TestClient(_app(executor))
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            seen: list = []
            _open_request(ws, "r1")
            start = time.monotonic()
            # Past the flush, and past where a session timeout measured from the
            # open would have fired, with audio flowing the whole time.
            while time.monotonic() - start < session_timeout + 0.5:
                ws.send_bytes(_PCM)
                time.sleep(FRAME_INTERVAL)
                _drain(ws, seen)
                assert not _closed(seen), f"session closed early: {seen}"
                assert not any(isinstance(f, dict) and f.get("error_type") == "timeout"
                               for f in seen), seen
            assert len(_audio_ends(seen, "r1")) == 1

            # Still open: a new request is served.
            ws.send_json({"type": "tts_stream", "request_id": "r2", "text": "Bye.",
                          "done": True})
            _wait_for_audio_end(ws, seen, "r2", timeout=2.0)
            assert not _closed(seen)
