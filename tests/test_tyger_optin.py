"""The turn_engine opt-in frame, model-free.

A connection opts in to the Tyger turn engine with {"type": "turn_engine",
"engine": "tyger"} before its first audio frame. On the audiocpp STT engine, with
the setting on and the sidecar up, the server opens one sidecar WebSocket for the
connection and answers turn_engine_ready. Every refusal is one error frame, after
which the session goes on as one that never opted in. A connection that never
sends the frame gets exactly the frames it gets with the setting off.

The STT side is the real audiocpp pipeline against tests/audiocpp_fake.py, the TTS
side a fake engine, and the sidecar the in-process FakeTygerSidecar behind a small
stand-in for app.state.turn_engine_sidecar. No GPU, no model, no live server.
"""
import asyncio
import json
import socket
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)
from fastapi.testclient import TestClient

from voicemanager.app import create_app
from voicemanager.pipeline import audiocpp_stt
from voicemanager.pipeline.tyger import wire
from voicemanager.protocol import (
    INVALID_REQUEST, TURN_ENGINE_UNAVAILABLE, TURN_ENGINE_UNSUPPORTED,
)
from voicemanager.session import SessionRegistry
from voicemanager.transport import websocket as ws_module
from voicemanager.transport.turn_engine import TurnEngineSession

from audiocpp_fake import FakeSidecar
from tyger_fake_sidecar import FakeTygerSidecar

# 512 samples of nonzero 16-bit PCM: even length, so it passes first-frame validation.
_PCM = b"\x11\x22" * 512
_PARTIAL = "scanning"
_FINAL = "scan complete"
_OPT_IN = {"type": "turn_engine", "engine": "tyger"}


# --- doubles --------------------------------------------------------------------

class _FakeTTSEngine:
    """Yields one short chunk per synthesis call."""

    @property
    def sample_rate(self) -> int:
        return 24000

    async def stream_synthesize(self, text: str, **kwargs):
        yield np.zeros(8, dtype=np.float32)


class _StandInSidecar:
    """What the handler reads from app.state.turn_engine_sidecar: up and session_url."""

    def __init__(self, url: str, up: bool = True) -> None:
        self.url = url
        self.up = up

    def session_url(self, label: str) -> str:
        return f"{self.url}?{urllib.parse.urlencode({'session': label})}"


class _InstantSTT:
    """A qwen stand-in that transcribes every frame at once."""

    def __init__(self, asr_model=None, **kwargs):
        pass

    async def feed(self, data: bytes) -> bool:
        return True

    def get_current_text(self):
        return ("qwen words", "en", 0)

    async def finalize(self):
        return ("", "en", 0)

    def reset(self) -> None:
        pass

    def dispose(self) -> None:
        pass


class _ScriptedAudioCppSTT:
    """A stand-in for AudioCppSTTPipeline whose frames do not depend on timing.

    The fake audio.cpp sidecar answers over a socket, so whether an interim lands in
    the same feed as its final varies from run to run. Two runs compared frame by
    frame need the same answer every time: each fed frame gives one interim and one
    final on its own segment.
    """

    def __init__(self, config=None, **kwargs):
        self._segment = 0
        self._final_pending = False

    async def feed(self, data: bytes) -> bool:
        self._final_pending = True
        return True

    def get_current_text(self):
        return (_PARTIAL, "en", self._segment)

    def take_final(self):
        if not self._final_pending:
            return None
        self._final_pending = False
        return (_FINAL, "en", self._segment)

    async def finalize(self):
        return ("", "en", self._segment)

    def reset(self) -> None:
        self._segment += 1

    def dispose(self) -> None:
        pass


# --- fixtures and helpers -------------------------------------------------------

@pytest.fixture
def tyger():
    fake = FakeTygerSidecar().start()
    try:
        yield fake
    finally:
        fake.stop()


@pytest.fixture
def stt():
    with FakeSidecar(audio_events=[_PARTIAL], audio_final=_FINAL) as fake:
        yield fake


@pytest.fixture
def tts_executor():
    with ThreadPoolExecutor(max_workers=1) as executor:
        yield executor


def _audiocpp_config(fake: FakeSidecar) -> dict:
    return {
        "sidecar_bin": "", "socket": fake.socket_path, "model": "",
        "model_task": "nemotron-3.5-asr-streaming", "backend": "cpu", "device": 0,
        "threads": 4, "assets_root": "",
    }


def _app(stt_fake=None, *, sidecar=None, stt_engine: str = "audiocpp", executor=None,
         set_sidecar: bool = True):
    """The real app on a fake STT and TTS. set_sidecar=False leaves the attribute unset,
    as on a TestClient that skips the lifespan with the setting off."""
    app = create_app()
    app.state.stt_engine_type = stt_engine
    if stt_engine == "audiocpp":
        app.state.audiocpp_config = _audiocpp_config(stt_fake)
    else:
        app.state.qwen_model = object()
    app.state.session_registry = SessionRegistry()
    if executor is not None:
        app.state.tts_engine_type = "openvoice"
        app.state.tts_engine = _FakeTTSEngine()
        app.state.tts_executor = executor
    if set_sidecar:
        app.state.turn_engine_sidecar = sidecar
    return app


def _frame(raw: dict):
    """A received frame as a comparable value: the JSON of a text frame, else
    ('pcm', length) for a binary one."""
    if raw.get("bytes") is not None:
        return ("pcm", len(raw["bytes"]))
    return json.loads(raw["text"])


def _recv_until(ws, done, limit: int = 50) -> list:
    """Receive frames until done(frame) is true for one; all of them, in order."""
    frames = []
    for _ in range(limit):
        frame = _frame(ws.receive())
        frames.append(frame)
        if done(frame):
            return frames
    raise AssertionError(f"no closing frame in {frames}")


def _is_final(frame) -> bool:
    return (isinstance(frame, dict) and frame.get("type") == "transcription"
            and frame["is_final"])


def _ready(ws) -> dict:
    ready = ws.receive_json()
    assert ready["type"] == "ready"
    return ready


def _assert_audio_still_served(ws) -> None:
    """One audio frame still gives transcription frames up to a final."""
    ws.send_bytes(_PCM)
    frames = _recv_until(ws, _is_final)
    assert all(f["type"] == "transcription" for f in frames), frames


def _assert_error(ws, error_type: str) -> dict:
    frame = ws.receive_json()
    assert frame["type"] == "error", frame
    assert frame["error_type"] == error_type, frame
    return frame


def _sidecar_for(tyger: FakeTygerSidecar, up: bool = True) -> _StandInSidecar:
    return _StandInSidecar(tyger.url, up=up)


# --- opting in ------------------------------------------------------------------

def test_opt_in_before_audio_on_audiocpp_answers_ready_and_opens_one_session(tyger, stt):
    client = TestClient(_app(stt, sidecar=_sidecar_for(tyger)))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        ready = _ready(ws)
        ws.send_json(_OPT_IN)
        assert ws.receive_json() == {"type": "turn_engine_ready", "engine": "tyger"}
        assert tyger.wait_for(lambda: len(tyger.sessions) == 1)
        # The connection's session id labels its sidecar session.
        assert tyger.sessions[0].key == ready["session_id"]
        _assert_audio_still_served(ws)
    # The client went away: the sidecar session is closed with it.
    assert tyger.wait_for(lambda: tyger.sessions[0].closed)
    assert len(tyger.sessions) == 1


def test_two_connections_get_two_sidecar_sessions(tyger, stt):
    client = TestClient(_app(stt, sidecar=_sidecar_for(tyger)))
    labels = []
    for _ in range(2):
        with authkit.ws_connect(client, "/ws/stream") as ws:
            labels.append(_ready(ws)["session_id"])
            ws.send_json(_OPT_IN)
            assert ws.receive_json()["type"] == "turn_engine_ready"
    assert tyger.wait_for(lambda: len(tyger.sessions) == 2 and
                          all(s.closed for s in tyger.sessions))
    assert [s.key for s in tyger.sessions] == labels


# --- refusals -------------------------------------------------------------------

def test_setting_off_answers_unavailable_and_the_session_goes_on(stt):
    client = TestClient(_app(stt, set_sidecar=False))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        ws.send_json(_OPT_IN)
        _assert_error(ws, TURN_ENGINE_UNAVAILABLE)
        _assert_audio_still_served(ws)


def test_setting_off_explicit_none_answers_unavailable(stt):
    client = TestClient(_app(stt, sidecar=None))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        ws.send_json(_OPT_IN)
        _assert_error(ws, TURN_ENGINE_UNAVAILABLE)
        _assert_audio_still_served(ws)


def test_sidecar_not_up_answers_unavailable_and_opens_nothing(tyger, stt):
    client = TestClient(_app(stt, sidecar=_sidecar_for(tyger, up=False)))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        ws.send_json(_OPT_IN)
        _assert_error(ws, TURN_ENGINE_UNAVAILABLE)
        _assert_audio_still_served(ws)
    assert tyger.sessions == []


def test_sidecar_connect_failure_answers_unavailable(stt):
    # Up by its own account, but nothing listens on the address it gives.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    client = TestClient(_app(stt, sidecar=_StandInSidecar(f"ws://127.0.0.1:{port}/tyger")))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        ws.send_json(_OPT_IN)
        _assert_error(ws, TURN_ENGINE_UNAVAILABLE)
        _assert_audio_still_served(ws)


def test_other_stt_engine_answers_unsupported_and_the_session_goes_on(
        tyger, monkeypatch):
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    client = TestClient(_app(sidecar=_sidecar_for(tyger), stt_engine="qwen"))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        ws.send_json(_OPT_IN)
        _assert_error(ws, TURN_ENGINE_UNSUPPORTED)
        ws.send_bytes(_PCM)
        frame = ws.receive_json()
        assert frame["type"] == "transcription" and frame["text"] == "qwen words"
    assert tyger.sessions == []


def test_opt_in_after_the_first_audio_frame_is_invalid(tyger, stt):
    client = TestClient(_app(stt, sidecar=_sidecar_for(tyger)))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        _assert_audio_still_served(ws)
        ws.send_json(_OPT_IN)
        _assert_error(ws, INVALID_REQUEST)
        _assert_audio_still_served(ws)
    assert tyger.sessions == []


def test_a_second_opt_in_is_invalid_and_keeps_the_first(tyger, stt):
    client = TestClient(_app(stt, sidecar=_sidecar_for(tyger)))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        ws.send_json(_OPT_IN)
        assert ws.receive_json()["type"] == "turn_engine_ready"
        ws.send_json(_OPT_IN)
        _assert_error(ws, INVALID_REQUEST)
        _assert_audio_still_served(ws)
        assert len(tyger.sessions) == 1
        assert not tyger.sessions[0].closed
    assert tyger.wait_for(lambda: tyger.sessions[0].closed)


def test_an_unknown_engine_is_invalid(tyger, stt):
    client = TestClient(_app(stt, sidecar=_sidecar_for(tyger)))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        ws.send_json({"type": "turn_engine", "engine": "other"})
        _assert_error(ws, INVALID_REQUEST)
        _assert_audio_still_served(ws)
    assert tyger.sessions == []


@pytest.mark.parametrize("frame", [
    {"type": "turn_engine"},
    {"type": "turn_engine", "engine": 5},
    {"type": "turn_engine", "engine": None},
])
def test_a_malformed_opt_in_is_invalid(tyger, stt, frame):
    client = TestClient(_app(stt, sidecar=_sidecar_for(tyger)))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        ws.send_json(frame)
        _assert_error(ws, INVALID_REQUEST)
        _assert_audio_still_served(ws)
    assert tyger.sessions == []


def test_a_refused_opt_in_does_not_count_as_the_first(tyger, stt):
    """Only an opt-in that got turn_engine_ready makes a later one a second opt-in."""
    client = TestClient(_app(stt, sidecar=_sidecar_for(tyger)))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        ws.send_json({"type": "turn_engine", "engine": "other"})
        _assert_error(ws, INVALID_REQUEST)
        ws.send_json(_OPT_IN)
        assert ws.receive_json()["type"] == "turn_engine_ready"
    assert tyger.wait_for(lambda: len(tyger.sessions) == 1 and tyger.sessions[0].closed)


# --- precedence when more than one refusal applies -------------------------------

def test_setting_off_wins_over_a_late_or_unknown_opt_in(stt):
    client = TestClient(_app(stt, sidecar=None))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        _assert_audio_still_served(ws)
        ws.send_json({"type": "turn_engine", "engine": "other"})
        _assert_error(ws, TURN_ENGINE_UNAVAILABLE)
        _assert_audio_still_served(ws)


def test_unsupported_engine_wins_over_a_late_or_unknown_opt_in(tyger, monkeypatch):
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    client = TestClient(_app(sidecar=_sidecar_for(tyger), stt_engine="qwen"))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        ws.send_bytes(_PCM)
        assert ws.receive_json()["type"] == "transcription"
        ws.send_json({"type": "turn_engine"})
        _assert_error(ws, TURN_ENGINE_UNSUPPORTED)


def test_a_late_opt_in_is_invalid_even_when_the_sidecar_is_down(tyger, stt):
    client = TestClient(_app(stt, sidecar=_sidecar_for(tyger, up=False)))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        _assert_audio_still_served(ws)
        ws.send_json(_OPT_IN)
        _assert_error(ws, INVALID_REQUEST)


# --- a connection that never opts in --------------------------------------------

def _scripted_session(client) -> list:
    """Audio, tts_stream, tts_cancel, turn_start and audio again; every frame received."""
    frames = []
    with authkit.ws_connect(client, "/ws/stream") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        ready.pop("session_id")  # a fresh uuid per connection
        frames.append(ready)

        ws.send_bytes(_PCM)
        frames += _recv_until(ws, _is_final)

        ws.send_json({"type": "tts_stream", "request_id": "r1", "text": "Hello there."})
        ws.send_json({"type": "tts_stream", "request_id": "r1", "text": " Bye now.",
                      "done": True})
        frames += _recv_until(ws, lambda f: isinstance(f, dict)
                              and f.get("type") == "audio_end")

        ws.send_json({"type": "tts_stream", "request_id": "r2", "text": "Never said"})
        ws.send_json({"type": "tts_cancel", "request_id": "r2"})
        frames += _recv_until(ws, lambda f: isinstance(f, dict)
                              and f.get("type") == "audio_end")

        ws.send_json({"type": "turn_start"})
        frames += _recv_until(ws, lambda f: isinstance(f, dict)
                              and f.get("type") == "turn_started")

        ws.send_bytes(_PCM)
        frames += _recv_until(ws, _is_final)
    return frames


def test_a_connection_that_never_opts_in_gets_the_same_frames_with_the_setting_on(
        tyger, stt, tts_executor, monkeypatch):
    monkeypatch.setattr(audiocpp_stt, "AudioCppSTTPipeline", _ScriptedAudioCppSTT)
    off = _scripted_session(TestClient(_app(stt, sidecar=None, executor=tts_executor)))
    on = _scripted_session(TestClient(_app(stt, sidecar=_sidecar_for(tyger),
                                           executor=tts_executor)))
    # The script reached every step, so the comparison covers each frame kind.
    kinds = {f["type"] if isinstance(f, dict) else f[0] for f in off}
    assert {"ready", "transcription", "audio_start", "pcm", "audio_end",
            "turn_started"} <= kinds
    assert on == off
    # The sidecar never heard from the connection that did not opt in.
    assert tyger.sessions == []


# --- the session object on its own ----------------------------------------------

def test_the_reader_parses_event_lines_and_close_leaves_no_task(tyger):
    async def scenario():
        sent: list[str] = []
        events: list = []

        async def send_text(text: str) -> None:
            sent.append(text)

        async def on_event(event) -> None:
            events.append(event)

        session = TurnEngineSession(
            sidecar=_sidecar_for(tyger), stt_engine_type="audiocpp", label="unit",
            send_text=send_text, on_event=on_event)
        assert not session.active
        await session.handle_opt_in(dict(_OPT_IN), audio_started=False)
        assert session.active
        assert [json.loads(t) for t in sent] == [
            {"type": "turn_engine_ready", "engine": "tyger"}]

        assert await asyncio.to_thread(tyger.wait_for, lambda: len(tyger.sessions) == 1)
        endpoint = wire.EndpointEvent(by="model", mic_sample=5120)
        barge = wire.BargeEvent(onset_mic_sample=100, mic_sample=5120)
        await asyncio.to_thread(tyger.send, 0, endpoint, barge)
        for _ in range(200):
            if len(events) == 2:
                break
            await asyncio.sleep(0.01)
        assert events == [endpoint, barge]

        # A line the wire format rejects is skipped; the reader keeps going.
        await asyncio.to_thread(
            asyncio.run_coroutine_threadsafe(
                tyger.sessions[0].ws.send('{"event": "nope"}'), tyger._loop).result, 5)
        await asyncio.to_thread(tyger.send, 0, endpoint)
        for _ in range(200):
            if len(events) == 3:
                break
            await asyncio.sleep(0.01)
        assert events == [endpoint, barge, endpoint]

        reader = session._reader
        await session.aclose()
        assert reader is not None and reader.done()
        assert not session.active
        await session.aclose()  # idempotent

    asyncio.run(scenario())
    assert tyger.wait_for(lambda: tyger.sessions and tyger.sessions[0].closed)


def test_close_without_opt_in_is_a_no_op():
    async def scenario():
        async def send_text(text: str) -> None:
            raise AssertionError("nothing to send")

        session = TurnEngineSession(sidecar=None, stt_engine_type="audiocpp",
                                    label="unit", send_text=send_text)
        await session.aclose()
        assert not session.active

    asyncio.run(scenario())
