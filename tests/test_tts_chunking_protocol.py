"""The ``chunking`` field of a ``tts_stream`` frame: its contract, its error path, its storage.

A ``tts_stream`` frame may carry ``chunking`` set to "sentence" or "clause". Without
it the request is "sentence", which is today's behaviour. Like ``voice``, the value is
fixed by the first frame of a request: a later frame's value is ignored.

A frame whose only fault is a bad ``chunking`` value gets an error frame with
``error_type`` "invalid_request" and is dropped. The connection stays open, a valid
request after it is served, and STT keeps running.

Model-free and server-free: ``TestClient(app)`` without ``with client:`` skips the
lifespan, the STT pipeline and the TTS engine are fakes, so no model loads, no GPU,
no network and no audio hardware.
"""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import authkit

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from voicemanager.app import create_app
from voicemanager.pipeline.tts import TTSPipeline, TTSStreamPipeline
from voicemanager.pipeline.tts_session import TTSSession
from voicemanager.protocol import TTSStreamMessage
from voicemanager.session import SessionRegistry

# 512 bytes: a valid raw 16-bit PCM frame, enough to trigger one STT feed().
_ONE_CHUNK = b"\x00\x00" * 256


# --- test doubles -----------------------------------------------------------

class _FakeWhisperSeg:
    def __init__(self, text: str):
        self.text = text
        self.avg_logprob = -0.1
        self.start = 0.0
        self.end = 1.0


class _FakeWhisperInfo:
    language = "en"


def _fake_whisper_factory(inferences):
    """Drop-in for STTPipeline: each run_inference() pops one inference's texts."""

    class _FakeWhisper:
        def __init__(self, **kwargs):
            self._inferences = list(inferences)

        def feed(self, data: bytes) -> bool:
            return bool(self._inferences)

        async def run_inference(self, loop):
            texts = self._inferences.pop(0) if self._inferences else []
            return [_FakeWhisperSeg(t) for t in texts], _FakeWhisperInfo(), 0.01

        def dispose(self):
            pass

    return _FakeWhisper


class _FakeStreamEngine:
    """Streaming TTS engine stub: one 8-sample chunk per sentence, no executor work."""

    is_batch = False

    @property
    def sample_rate(self) -> int:
        return 24000

    async def stream_synthesize(self, text: str, **kwargs):
        yield np.zeros(8, dtype=np.float32)


class _FakeWebSocket:
    def __init__(self):
        self.frames: list[tuple[str, object]] = []

    async def send_text(self, data: str) -> None:
        self.frames.append(("text", json.loads(data)))

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append(("bytes", data))


def _app(executor):
    app = create_app()
    app.state.stt_engine_type = "whisper"
    app.state.session_registry = SessionRegistry()
    app.state.tts_engine_type = "openvoice"
    app.state.tts_engine = _FakeStreamEngine()
    app.state.tts_executor = executor
    return app


def _receive_until(ws, frame_type: str, limit: int = 20) -> list[object]:
    """Read frames until a text frame of ``frame_type``. Returns everything read."""
    seen: list[object] = []
    for _ in range(limit):
        msg = ws.receive()
        if msg.get("text") is not None:
            frame = json.loads(msg["text"])
            seen.append(frame)
            if frame.get("type") == frame_type:
                return seen
        elif msg.get("bytes") is not None:
            seen.append(msg["bytes"])
        else:
            raise AssertionError(f"socket ended before a {frame_type!r} frame: {msg!r}")
    raise AssertionError(f"no {frame_type!r} frame in {limit} frames: {seen!r}")


@pytest.fixture
def fresh_settings():
    import voicemanager.config as cfg
    cfg._settings = None
    yield
    cfg._settings = None


# --- the message contract ---------------------------------------------------

def test_chunking_defaults_to_sentence_and_is_optional():
    msg = TTSStreamMessage(request_id="r", text="t")
    assert msg.chunking == "sentence"
    assert TTSStreamMessage.model_fields["chunking"].is_required() is False
    assert json.loads(msg.model_dump_json())["chunking"] == "sentence"


def test_chunking_accepts_sentence_and_clause():
    for value in ("sentence", "clause"):
        msg = TTSStreamMessage.model_validate(
            {"type": "tts_stream", "request_id": "r", "text": "t", "chunking": value})
        assert msg.chunking == value


def test_chunking_rejects_any_other_value():
    for value in ("word", "", "Clause", None, 3):
        with pytest.raises(ValidationError):
            TTSStreamMessage.model_validate(
                {"type": "tts_stream", "request_id": "r", "text": "t", "chunking": value})


# --- the error path over the real WebSocket handler -------------------------

def test_bad_chunking_gets_invalid_request_and_the_connection_keeps_working(
        monkeypatch, fresh_settings):
    monkeypatch.setattr(
        "voicemanager.transport.websocket.STTPipeline",
        _fake_whisper_factory([["all stop"]]),
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        client = TestClient(_app(executor))
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"

            ws.send_json({"type": "tts_stream", "request_id": "bad", "text": "Hello.",
                          "done": True, "chunking": "word"})
            error = ws.receive_json()
            assert error["type"] == "error"
            assert error["error_type"] == "invalid_request"
            assert "chunking" in error["message"]

            # The same connection still serves a valid request...
            ws.send_json({"type": "tts_stream", "request_id": "good", "text": "Hello.",
                          "done": True, "chunking": "clause"})
            frames = _receive_until(ws, "audio_end")
            texts = [f for f in frames if isinstance(f, dict)]
            assert texts[0]["type"] == "audio_start"
            assert texts[0]["request_id"] == "good"
            assert texts[-1]["request_id"] == "good"
            assert any(isinstance(f, bytes) for f in frames), "no PCM audio flowed"
            # ...and the dropped frame opened nothing: no frame names its request_id.
            assert all(f.get("request_id") != "bad" for f in texts)

            # ...and STT keeps running.
            ws.send_bytes(_ONE_CHUNK)
            frames = _receive_until(ws, "transcription")
            assert frames[-1]["text"] == "all stop"


def test_bad_chunking_on_an_open_request_leaves_it_alone(monkeypatch, fresh_settings):
    monkeypatch.setattr(
        "voicemanager.transport.websocket.STTPipeline", _fake_whisper_factory([]))
    with ThreadPoolExecutor(max_workers=1) as executor:
        client = TestClient(_app(executor))
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "tts_stream", "request_id": "r1", "text": "Hello "})
            ws.send_json({"type": "tts_stream", "request_id": "r2", "text": "x",
                          "chunking": "word"})
            assert ws.receive_json()["error_type"] == "invalid_request"
            # r1 is still the open request: closing it speaks its text and ends it.
            ws.send_json({"type": "tts_stream", "request_id": "r1", "text": "there.",
                          "done": True})
            frames = _receive_until(ws, "audio_end")
            texts = [f for f in frames if isinstance(f, dict)]
            assert [f["type"] for f in texts] == ["audio_start", "audio_end"]
            assert {f["request_id"] for f in texts} == {"r1"}
            assert texts[-1]["reason"] == "complete"


def test_other_malformed_tts_stream_frames_keep_the_server_error_path(
        monkeypatch, fresh_settings):
    # A frame with a missing request_id is not a chunking fault: it keeps today's path,
    # even when it also carries a bad chunking value.
    monkeypatch.setattr(
        "voicemanager.transport.websocket.STTPipeline", _fake_whisper_factory([]))
    for frame in (
        {"type": "tts_stream", "text": "Hello."},
        {"type": "tts_stream", "text": "Hello.", "chunking": "word"},
    ):
        with ThreadPoolExecutor(max_workers=1) as executor:
            client = TestClient(_app(executor))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"
                ws.send_json(frame)
                error = ws.receive_json()
                assert error["type"] == "error"
                assert error["error_type"] == "server_error"


def test_the_handler_threads_chunking_through_to_the_stream_pipeline(
        monkeypatch, fresh_settings):
    monkeypatch.setattr(
        "voicemanager.transport.websocket.STTPipeline", _fake_whisper_factory([]))
    seen: list[str | None] = []
    original = TTSStreamPipeline.feed_text

    async def recording_feed_text(self, *args, **kwargs):
        seen.append(kwargs.get("chunking"))
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(TTSStreamPipeline, "feed_text", recording_feed_text)
    with ThreadPoolExecutor(max_workers=1) as executor:
        client = TestClient(_app(executor))
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "tts_stream", "request_id": "r1", "text": "Hi.",
                          "done": True, "chunking": "clause"})
            _receive_until(ws, "audio_end")
            ws.send_json({"type": "tts_stream", "request_id": "r2", "text": "Hi.",
                          "done": True})
            _receive_until(ws, "audio_end")
    assert seen == ["clause", "sentence"]


# --- storage: fixed by the request's first frame ----------------------------

def test_the_stored_chunking_value_is_the_first_frames_value():
    async def run():
        ws = _FakeWebSocket()
        tts = TTSPipeline(engine=_FakeStreamEngine(), executor=None)
        stream = TTSStreamPipeline(tts)
        session = TTSSession(websocket=ws, tts_pipeline=tts, tts_stream_pipeline=stream)
        assert stream.chunking == "sentence"

        # Opened with clause; later frames say sentence, or nothing, and are ignored.
        await session.feed_stream(chunk="One, ", done=False, request_id="r1",
                                  chunking="clause")
        assert stream.chunking == "clause"
        await session.feed_stream(chunk="two, ", done=False, request_id="r1",
                                  chunking="sentence")
        assert stream.chunking == "clause"
        await session.feed_stream(chunk="three.", done=True, request_id="r1")
        assert stream.chunking == "clause"
        await asyncio.wait_for(session.wait_idle(), 2.0)

        # A new request opened without the field is a sentence request.
        await session.feed_stream(chunk="Four, ", done=False, request_id="r2")
        assert stream.chunking == "sentence"
        await session.feed_stream(chunk="five.", done=True, request_id="r2",
                                  chunking="clause")
        assert stream.chunking == "sentence"
        await asyncio.wait_for(session.wait_idle(), 2.0)

    asyncio.run(run())


def test_feed_text_stores_chunking_at_request_start_like_voice():
    async def run():
        ws = _FakeWebSocket()
        stream = TTSStreamPipeline(TTSPipeline(engine=_FakeStreamEngine(), executor=None))
        await stream.feed_text("a", False, ws, "r1", voice="v1", chunking="clause")
        await stream.feed_text("b", False, ws, "r1", voice="v2", chunking="sentence")
        assert stream.chunking == "clause"
        await stream.feed_text("c", True, ws, "r1")
        assert stream.chunking == "clause"
        await asyncio.wait_for(stream.synthesis_task, 2.0)

    asyncio.run(run())
