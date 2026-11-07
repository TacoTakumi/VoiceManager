"""The listening boundary over the real WebSocket handler, model-free.

A client that stops sending mic audio sends {"type": "listen_stop"} and later
{"type": "listen_start"} on the same connection. These tests prove, through the
real /ws/stream handler:

* listen_stop is answered at once with {"type": "listen_stopped", "mic_sample": N},
  N the mic samples the server accepted on the connection (a WAV header on the
  first frame is not samples);
* while stopped, binary mic frames are dropped: no error, no transcription, not
  fed to the STT or the turn engine, not counted;
* listen_start resumes listening with no answer, and the count goes on from where
  it was; audio sent right behind a listen_stop, before its answer arrives, works;
* both frames are idempotent and never send an error frame, and many stop/start
  pairs work on one connection;
* listen_stop before the first audio frame does not count as that frame, so a
  turn_engine opt-in after it still succeeds, and the turn engine's own mic count
  agrees with mic_sample;
* a tts request while stopped is synthesized and sent as usual;
* an unknown client frame type still gets no error frame;
* a stopped connection is exempt from the session inactivity timeout: it gets no
  timeout_warning and is not closed, while the other receive caps (the tts_stream
  idle flush) still run; after listen_start the timeout runs again from that frame.

The STT is a whisper stand-in that answers every fed frame with one final, the TTS
a fake engine, and the turn-engine sidecar the in-process FakeTygerSidecar.
TestClient without ``with client:`` skips the lifespan. No GPU, no model, no live
server, no audio hardware.
"""
import asyncio
import io
import time
import wave
from concurrent.futures import ThreadPoolExecutor

import pytest

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)
from fastapi.testclient import TestClient

import voicemanager.config as cfg
from voicemanager.app import create_app
from voicemanager.pipeline import audiocpp_stt
from voicemanager.session import SessionRegistry
from voicemanager.transport import websocket as ws_module

from tyger_fake_sidecar import FakeTygerSidecar
from tyger_ws_kit import (
    FakeTTSEngine, ScriptedAudioCppSTT, StandInSidecar, barrier, errors, frame, is_type,
    make_app, opt_in, recv_until, speak,
)

STOP = {"type": "listen_stop"}
START = {"type": "listen_start"}


def _pcm(samples: int, value: int = 0x0101) -> bytes:
    """``samples`` samples of 16-bit PCM."""
    return value.to_bytes(2, "little") * samples


def _wav(samples: int) -> bytes:
    """A WAV container holding ``samples`` samples of 16 kHz mono 16-bit PCM."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(_pcm(samples))
    return buf.getvalue()


def _stopped(n: int) -> dict:
    return {"type": "listen_stopped", "mic_sample": n}


# --- doubles --------------------------------------------------------------------

class _CountingWhisper:
    """A whisper stand-in: every fed frame is recorded and gives one final."""

    def __init__(self, **kwargs) -> None:
        self.fed: list[bytes] = []

    def feed(self, data: bytes) -> bool:
        self.fed.append(bytes(data))
        return True

    async def run_inference(self, loop):
        return [_Seg(f"frame {len(self.fed)}")], _Info(), 0.01

    def dispose(self) -> None:
        pass


class _Seg:
    def __init__(self, text: str) -> None:
        self.text = text
        self.avg_logprob = -0.1
        self.start = 0.0
        self.end = 1.0


class _Info:
    language = "en"


# --- fixtures -------------------------------------------------------------------

@pytest.fixture
def settings(monkeypatch):
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "60")
    cfg._settings = None
    yield
    cfg._settings = None


@pytest.fixture
def whispers(monkeypatch, settings) -> list:
    """The whisper stand-ins the connections made."""
    made: list[_CountingWhisper] = []

    def factory(**kwargs):
        made.append(_CountingWhisper(**kwargs))
        return made[-1]

    monkeypatch.setattr(ws_module, "STTPipeline", factory)
    return made


@pytest.fixture
def executor():
    with ThreadPoolExecutor(max_workers=1) as pool:
        yield pool


def _classic_client(executor=None) -> TestClient:
    app = create_app()
    app.state.stt_engine_type = "whisper"
    app.state.session_registry = SessionRegistry()
    if executor is not None:
        app.state.tts_engine_type = "openvoice"
        app.state.tts_engine = FakeTTSEngine()
        app.state.tts_executor = executor
    return TestClient(app)


@pytest.fixture
def classic(whispers):
    return _classic_client()


def _types(frames: list) -> list:
    return [f.get("type") if isinstance(f, dict) else f[0] for f in frames]


def _ready(ws) -> None:
    assert ws.receive_json()["type"] == "ready"


def _send_and_read_final(ws, data: bytes) -> dict:
    ws.send_bytes(data)
    msg = ws.receive_json()
    assert msg["type"] == "transcription" and msg["is_final"], msg
    return msg


def _stop(ws) -> list:
    """Send listen_stop and read up to its answer: every frame, the answer last."""
    ws.send_json(STOP)
    return recv_until(ws, is_type("listen_stopped"))


# --- the answer and the count -----------------------------------------------------

def test_listen_stop_answers_with_the_accepted_sample_count(classic, whispers):
    with authkit.ws_connect(classic, "/ws/stream") as ws:
        _ready(ws)
        _send_and_read_final(ws, _pcm(256))
        _send_and_read_final(ws, _pcm(300))
        frames = _stop(ws)
    assert frames == [_stopped(556)]
    assert whispers[0].fed == [_pcm(256), _pcm(300)]


def test_a_wav_header_on_the_first_frame_is_not_counted(classic):
    with authkit.ws_connect(classic, "/ws/stream") as ws:
        _ready(ws)
        _send_and_read_final(ws, _wav(400))
        _send_and_read_final(ws, _pcm(100))
        assert _stop(ws) == [_stopped(500)]


def test_listen_stop_before_any_audio_answers_zero(classic):
    with authkit.ws_connect(classic, "/ws/stream") as ws:
        _ready(ws)
        assert _stop(ws) == [_stopped(0)]


# --- audio while stopped ----------------------------------------------------------

def test_audio_while_stopped_is_dropped_and_not_counted(classic, whispers):
    with authkit.ws_connect(classic, "/ws/stream") as ws:
        _ready(ws)
        _send_and_read_final(ws, _pcm(256))
        assert _stop(ws) == [_stopped(256)]
        for _ in range(3):
            ws.send_bytes(_pcm(512))
        # Nothing at all comes back for the dropped frames: no error, no transcription.
        assert barrier(ws) == []
        assert _stop(ws) == [_stopped(256)]
    assert whispers[0].fed == [_pcm(256)]


def test_odd_or_empty_frames_while_stopped_are_dropped_without_an_error(classic, whispers):
    """A dropped frame is not validated, so not even a malformed one gets an error."""
    with authkit.ws_connect(classic, "/ws/stream") as ws:
        _ready(ws)
        assert _stop(ws) == [_stopped(0)]
        ws.send_bytes(b"\x01\x02\x03")
        assert barrier(ws) == []
        ws.send_json(START)
        _send_and_read_final(ws, _pcm(64))
        assert _stop(ws) == [_stopped(64)]
    assert whispers[0].fed == [_pcm(64)]


def test_audio_resumes_after_listen_start_and_the_count_goes_on(classic, whispers):
    with authkit.ws_connect(classic, "/ws/stream") as ws:
        _ready(ws)
        _send_and_read_final(ws, _pcm(256))
        assert _stop(ws) == [_stopped(256)]
        ws.send_bytes(_pcm(1000))                 # dropped
        ws.send_json(START)                       # no answer
        _send_and_read_final(ws, _pcm(128, 0x0202))
        assert _stop(ws) == [_stopped(384)]
    assert whispers[0].fed == [_pcm(256), _pcm(128, 0x0202)]


# --- idempotent frames ------------------------------------------------------------

def test_repeated_frames_are_idempotent_and_never_send_an_error(classic, whispers):
    with authkit.ws_connect(classic, "/ws/stream") as ws:
        _ready(ws)
        ws.send_json(START)                       # already listening: nothing
        ws.send_json(START)
        assert barrier(ws) == []
        _send_and_read_final(ws, _pcm(256))
        assert _stop(ws) == [_stopped(256)]
        assert _stop(ws) == [_stopped(256)]       # stopped already: same answer
        ws.send_bytes(_pcm(256))
        assert _stop(ws) == [_stopped(256)]
        ws.send_json(START)
        ws.send_json(START)
        assert barrier(ws) == []
        _send_and_read_final(ws, _pcm(256))
        assert _stop(ws) == [_stopped(512)]
    assert len(whispers[0].fed) == 2


def test_listen_start_and_audio_right_behind_listen_stop(classic, whispers):
    """The client does not wait for listen_stopped: the answer still carries the count
    at the boundary, and the audio after listen_start is accepted."""
    with authkit.ws_connect(classic, "/ws/stream") as ws:
        _ready(ws)
        _send_and_read_final(ws, _pcm(256))
        ws.send_json(STOP)
        ws.send_json(START)
        ws.send_bytes(_pcm(100))
        first = frame(ws.receive())
        second = frame(ws.receive())
        assert first == _stopped(256)
        assert second["type"] == "transcription" and second["is_final"]
        assert _stop(ws) == [_stopped(356)]
    assert whispers[0].fed == [_pcm(256), _pcm(100)]


def test_many_stop_start_pairs_on_one_classic_connection(classic, whispers):
    total = 0
    with authkit.ws_connect(classic, "/ws/stream") as ws:
        _ready(ws)
        for i in range(25):
            n = 10 + i
            _send_and_read_final(ws, _pcm(n))
            total += n
            assert _stop(ws) == [_stopped(total)]
            ws.send_bytes(_pcm(77))               # dropped every time
            ws.send_json(START)
        _send_and_read_final(ws, _pcm(5))
        assert _stop(ws) == [_stopped(total + 5)]
    assert len(whispers[0].fed) == 26
    assert sum(len(b) for b in whispers[0].fed) // 2 == total + 5


def test_an_unknown_frame_type_gets_no_error_frame(classic):
    with authkit.ws_connect(classic, "/ws/stream") as ws:
        _ready(ws)
        ws.send_json({"type": "listen_pause"})
        assert barrier(ws) == []
        assert _stop(ws) == [_stopped(0)]
        ws.send_json({"type": "no_such_frame"})   # also while stopped
        assert barrier(ws) == []


def test_extra_fields_on_the_listen_frames_get_no_error(classic):
    with authkit.ws_connect(classic, "/ws/stream") as ws:
        _ready(ws)
        ws.send_json({"type": "listen_stop", "reason": "user"})
        assert recv_until(ws, is_type("listen_stopped")) == [_stopped(0)]
        ws.send_json({"type": "listen_start", "x": 1})
        assert barrier(ws) == []
        _send_and_read_final(ws, _pcm(10))


# --- TTS while stopped --------------------------------------------------------------

def test_a_tts_request_while_stopped_is_synthesized(whispers, executor):
    client = _classic_client(executor)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        _send_and_read_final(ws, _pcm(256))
        assert _stop(ws) == [_stopped(256)]
        start, pcm = speak(ws, "r1")
        assert start["request_id"] == "r1"
        assert len(pcm) > 0
        assert _stop(ws) == [_stopped(256)]


# --- the turn engine ----------------------------------------------------------------

@pytest.fixture
def tyger():
    fake = FakeTygerSidecar().start()
    try:
        yield fake
    finally:
        fake.stop()


@pytest.fixture
def turn_engines(monkeypatch, settings) -> list:
    """The connections' turn-engine sessions, so a test can read their mic count."""
    made = []
    base = ws_module.TurnEngineSession

    class Recording(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(ws_module, "TurnEngineSession", Recording)
    monkeypatch.setattr(audiocpp_stt, "AudioCppSTTPipeline", ScriptedAudioCppSTT)
    return made


def test_listen_stop_before_audio_still_lets_the_turn_engine_opt_in(
        tyger, turn_engines, executor):
    client = TestClient(make_app(sidecar=StandInSidecar(tyger.url), executor=executor))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        assert _stop(ws) == [_stopped(0)]
        ws.send_json({"type": "turn_engine", "engine": "tyger"})
        assert ws.receive_json() == {"type": "turn_engine_ready", "engine": "tyger"}
        assert tyger.wait_for(lambda: len(tyger.sessions) == 1)
        ws.send_json(START)
        ws.send_bytes(_wav(2000))
        ws.send_bytes(_pcm(1000))
        assert errors(barrier(ws)) == []
        assert _stop(ws)[-1] == _stopped(3000)
        feed = turn_engines[0]._feed
        assert feed.mic_samples == 3000
        ws.send_bytes(_pcm(4000))                 # dropped: not fed to the turn engine
        assert barrier(ws) == []
        assert feed.mic_samples == 3000
        ws.send_json(START)
        ws.send_bytes(_pcm(500))
        assert errors(barrier(ws)) == []
        assert _stop(ws)[-1] == _stopped(3500)
        assert feed.mic_samples == 3500


def test_a_turn_engine_session_that_never_stops_counts_the_same(
        tyger, turn_engines, executor):
    """Opted in with no listen frames until the end: the count matches the engine's."""
    client = TestClient(make_app(sidecar=StandInSidecar(tyger.url), executor=executor))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        assert tyger.wait_for(lambda: len(tyger.sessions) == 1)
        ws.send_bytes(_pcm(1500))
        ws.send_bytes(_pcm(700))
        frames = barrier(ws)
        assert errors(frames) == []
        assert "listen_stopped" not in _types(frames)
        assert _stop(ws)[-1] == _stopped(turn_engines[0]._feed.mic_samples) == _stopped(2200)


# --- the session inactivity timeout ---------------------------------------------------

# A short session timeout and warning lead, so the warning comes 1 s after the last
# frame and the close 1 s later. The lead is a module constant of the handler.
TIMEOUT = 2
LEAD = 1
# Scheduling slack on the timing bounds: the receive loop and the test client hand
# off through threads.
SLACK = 0.5


@pytest.fixture
def short_timeout(monkeypatch, whispers):
    monkeypatch.setenv("VM_SESSION_TIMEOUT", str(TIMEOUT))
    monkeypatch.setenv("VM_TTS_STREAM_FLUSH_TIMEOUT", "1")
    monkeypatch.setattr(ws_module, "WARNING_LEAD", LEAD, raising=False)
    cfg._settings = None
    yield
    cfg._settings = None


def _is_warning(f) -> bool:
    return isinstance(f, dict) and f.get("error_type") == "timeout_warning"


def _read_timeout_close(ws, since: float) -> None:
    """Read the warning, the timeout error and the 1001 close, each not early."""
    warning = ws.receive_json()
    assert _is_warning(warning), warning
    assert time.monotonic() - since >= TIMEOUT - LEAD - 0.1
    closing = ws.receive_json()
    assert closing.get("error_type") == "timeout", closing
    assert time.monotonic() - since >= TIMEOUT - 0.1
    closed = ws.receive()
    assert closed["type"] == "websocket.close" and closed["code"] == 1001, closed
    assert time.monotonic() - since <= TIMEOUT + SLACK


def test_session_timeout_still_warns_and_closes_a_listening_connection(
        classic, short_timeout):
    """A connection that never sends listen_stop times out as before."""
    with authkit.ws_connect(classic, "/ws/stream") as ws:
        _ready(ws)
        _send_and_read_final(ws, _pcm(64))
        _read_timeout_close(ws, time.monotonic())


def test_session_timeout_does_not_fire_while_stopped(classic, short_timeout):
    with authkit.ws_connect(classic, "/ws/stream") as ws:
        _ready(ws)
        _send_and_read_final(ws, _pcm(64))
        assert _stop(ws) == [_stopped(64)]
        time.sleep(TIMEOUT + 0.3)
        # Nothing came while stopped: no warning, no timeout error, no close.
        assert barrier(ws) == []
        assert _stop(ws) == [_stopped(64)]


def test_session_timeout_runs_again_from_listen_start(classic, whispers, short_timeout):
    """A warning sent before the stop does not carry over; the timer starts at
    listen_start, so the next warning and the close are measured from it."""
    with authkit.ws_connect(classic, "/ws/stream") as ws:
        _ready(ws)
        assert _is_warning(ws.receive_json())     # warned while listening
        assert _stop(ws) == [_stopped(0)]
        time.sleep(TIMEOUT + 0.3)
        ws.send_json(START)
        _read_timeout_close(ws, time.monotonic())


def test_session_timeout_exemption_keeps_the_tts_stream_idle_flush(
        whispers, executor, short_timeout):
    """While stopped, an open tts_stream request is still flushed by its idle cap."""
    client = _classic_client(executor)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        assert _stop(ws) == [_stopped(0)]
        ws.send_json({"type": "tts_stream", "request_id": "r1", "text": "Hello there",
                      "flush": True})
        sent = time.monotonic()
        frames = recv_until(ws, is_type("audio_end"))
        assert time.monotonic() - sent <= 1 + SLACK
        assert not any(_is_warning(f) for f in frames), frames
        assert errors(frames) == []
        time.sleep(TIMEOUT + 0.3 - (time.monotonic() - sent))
        assert barrier(ws) == []


class _RecordingAsyncio:
    """The handler's asyncio, recording the timeout of every receive wait."""

    def __init__(self) -> None:
        self.timeouts: list = []

    def __getattr__(self, name):
        return getattr(asyncio, name)

    async def wait_for(self, awaitable, timeout):
        self.timeouts.append(timeout)
        return await asyncio.wait_for(awaitable, timeout)


def test_session_timeout_leg_is_off_while_stopped_so_the_wait_blocks(
        classic, short_timeout, monkeypatch):
    """While stopped and with no other cap armed, the receive waits with no deadline
    and the loop does not spin."""
    shim = _RecordingAsyncio()
    monkeypatch.setattr(ws_module, "asyncio", shim)
    with authkit.ws_connect(classic, "/ws/stream") as ws:
        _ready(ws)
        assert _stop(ws) == [_stopped(0)]
        time.sleep(0.3)
        before = len(shim.timeouts)
        time.sleep(TIMEOUT)
        assert len(shim.timeouts) == before
        assert shim.timeouts[-1] is None
        assert barrier(ws) == []
