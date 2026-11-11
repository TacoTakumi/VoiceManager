"""The server-side wake word over the real WebSocket handler, model-free.

A client with no keyword spotter of its own sends {"type": "wake_word"}, with an
optional phrase list. These tests prove, through the real /ws/stream handler:

* the answer is {"type": "wake_word_ready", "phrases": [...]}, with the server's
  default list or the client's own list, and the list is per connection;
* the connection is then dormant: its mic frames go to its spotter stream and to
  nothing else (no transcription, not fed to the STT or the turn engine, not counted);
* when the spotter hears a phrase the server sends exactly one {"type": "wake",
  "phrase": ..., "mic_sample": N} and listens: the frames after it reach the STT, and
  the frame the phrase was heard in does not;
* a listen_stop gets its listen_stopped answer and makes the connection dormant
  again on a new spotter stream; a listen_start wakes a dormant connection with no
  wake frame;
* a wake_word the server cannot use (no wake engine, a malformed frame, a bad phrase
  list, a phrase the model has no tokens for) gets one invalid_request error frame
  and changes nothing;
* a connection that does not opt in sends nothing to the spotter, on a server that
  has one;
* a dormant connection is exempt from the session inactivity timeout, and the
  timeout runs again from the wake.

The spotter is a stand-in that reports a phrase for a marked frame, the STT a whisper
stand-in that answers every fed frame with one final, and the turn-engine sidecar the
in-process FakeTygerSidecar. TestClient without ``with client:`` skips the lifespan.
No GPU, no model, no live server, no audio hardware.
"""
import time

import pytest

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)
from fastapi.testclient import TestClient

import voicemanager.config as cfg
from voicemanager.app import create_app
from voicemanager.pipeline import audiocpp_stt
from voicemanager.session import SessionRegistry
from voicemanager.transport import websocket as ws_module
from voicemanager.wake import WakePhraseError

from tyger_fake_sidecar import FakeTygerSidecar
from tyger_ws_kit import (
    ScriptedAudioCppSTT, StandInSidecar, barrier, errors, is_type, make_app, recv_until,
)

WAKE_WORD = {"type": "wake_word"}
STOP = {"type": "listen_stop"}
START = {"type": "listen_start"}

DEFAULT_PHRASES = ["hey computer"]
# The first sample of a frame in which the stand-in spotter hears a phrase: the
# stream's first phrase, or its last.
HEAR_FIRST = 0x7001
HEAR_LAST = 0x7002


def _pcm(samples: int, value: int = 0x0101) -> bytes:
    """``samples`` samples of 16-bit PCM."""
    return value.to_bytes(2, "little") * samples


def _ready_frame(phrases=DEFAULT_PHRASES) -> dict:
    return {"type": "wake_word_ready", "phrases": list(phrases)}


def _wake(phrase: str, n: int, dormant: int) -> dict:
    return {"type": "wake", "phrase": phrase, "mic_sample": n, "dormant_sample": dormant}


def _stopped(n: int) -> dict:
    return {"type": "listen_stopped", "mic_sample": n}


# --- doubles --------------------------------------------------------------------

class _FakeWakeStream:
    def __init__(self, phrases: list[str]) -> None:
        self.phrases = phrases
        self.fed: list[bytes] = []

    def feed(self, pcm: bytes) -> str | None:
        self.fed.append(bytes(pcm))
        first = int.from_bytes(pcm[:2], "little") if len(pcm) >= 2 else 0
        if first == HEAR_FIRST:
            return self.phrases[0]
        if first == HEAR_LAST:
            return self.phrases[-1]
        return None


class _FakeWakeSpotter:
    """A stand-in wake engine. It has no tokens for a phrase with the word 'zzz'."""

    executor = None

    def __init__(self) -> None:
        self.default_phrases = list(DEFAULT_PHRASES)
        self.streams: list[_FakeWakeStream] = []

    def open_stream(self, phrases: list[str]) -> _FakeWakeStream:
        for phrase in phrases:
            if "zzz" in phrase.split():
                raise WakePhraseError(f"the wake model has no tokens for the phrase {phrase!r}")
        self.streams.append(_FakeWakeStream(list(phrases)))
        return self.streams[-1]

    def close(self) -> None:
        pass


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
def spotter() -> _FakeWakeSpotter:
    return _FakeWakeSpotter()


def _client(wake_spotter=None, *, set_state: bool = True) -> TestClient:
    app = create_app()
    app.state.stt_engine_type = "whisper"
    app.state.session_registry = SessionRegistry()
    if set_state:
        app.state.wake_spotter = wake_spotter
    return TestClient(app)


@pytest.fixture
def client(whispers, spotter) -> TestClient:
    """A server with a wake engine."""
    return _client(spotter)


def _ready(ws) -> None:
    assert ws.receive_json()["type"] == "ready"


def _send_and_read_final(ws, data: bytes) -> dict:
    ws.send_bytes(data)
    msg = ws.receive_json()
    assert msg["type"] == "transcription" and msg["is_final"], msg
    return msg


def _opt_in(ws, phrases=None) -> dict:
    """Read the ready frame, opt in to the wake word and return the answer."""
    _ready(ws)
    ws.send_json(WAKE_WORD if phrases is None else {**WAKE_WORD, "phrases": phrases})
    return ws.receive_json()


def _stop(ws) -> list:
    """Send listen_stop and read up to its answer: every frame, the answer last."""
    ws.send_json(STOP)
    return recv_until(ws, is_type("listen_stopped"))


# --- a server with no wake engine --------------------------------------------------

@pytest.mark.parametrize("set_state", [True, False])
def test_wake_word_on_a_server_with_no_wake_engine_is_refused(whispers, set_state):
    """Engine none: the app's state holds no spotter, or (a test app) no such entry."""
    with authkit.ws_connect(_client(None, set_state=set_state), "/ws/stream") as ws:
        _ready(ws)
        ws.send_json(WAKE_WORD)
        answer = ws.receive_json()
        assert answer["type"] == "error" and answer["error_type"] == "invalid_request", answer
        assert "no wake word engine" in answer["message"]
        # No wake_word_ready follows, and the connection is open and listening.
        assert barrier(ws) == []
        _send_and_read_final(ws, _pcm(64))
        assert _stop(ws) == [_stopped(64)]
    assert whispers[0].fed == [_pcm(64)]


# --- the opt-in and the dormant state ---------------------------------------------

def test_wake_word_answers_with_the_default_phrases(client, spotter):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws) == _ready_frame()
        assert barrier(ws) == []
    assert [s.phrases for s in spotter.streams] == [DEFAULT_PHRASES]


def test_dormant_mic_frames_go_to_the_spotter_and_not_to_the_stt(client, spotter, whispers):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws) == _ready_frame()
        frames = [_pcm(160, 0x0101), _pcm(320, 0x0202), _pcm(160, 0x0303)]
        for f in frames:
            ws.send_bytes(f)
        # Nothing comes back for them: no transcription, no error, no wake.
        assert barrier(ws) == []
        # A listen_stop on a dormant connection answers with the count, and the
        # dormant audio is not in it.
        assert _stop(ws) == [_stopped(0)]
    assert spotter.streams[0].fed == frames
    assert whispers[0].fed == []


def test_an_odd_or_empty_dormant_frame_is_given_to_the_spotter_with_no_error(client, spotter):
    """A dormant frame is not validated as the connection's first audio frame."""
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws) == _ready_frame()
        ws.send_bytes(b"\x01\x02\x03")
        assert barrier(ws) == []
    assert spotter.streams[0].fed == [b"\x01\x02\x03"]


def test_opting_in_after_audio_drops_what_was_open_and_keeps_the_count(
        client, spotter, whispers):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        _send_and_read_final(ws, _pcm(256))
        ws.send_json(WAKE_WORD)
        assert ws.receive_json() == _ready_frame()
        ws.send_bytes(_pcm(512))
        assert barrier(ws) == []
        assert _stop(ws) == [_stopped(256)]
    assert whispers[0].fed == [_pcm(256)]
    assert spotter.streams[0].fed == [_pcm(512)]


# --- the wake ----------------------------------------------------------------------

def test_a_heard_phrase_sends_one_wake_frame_and_the_connection_listens(
        client, spotter, whispers):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws) == _ready_frame()
        ws.send_bytes(_pcm(160))
        ws.send_bytes(_pcm(160, HEAR_FIRST))
        assert ws.receive_json() == _wake("hey computer", 0, 320)
        after = [_pcm(200, 0x0404), _pcm(300, 0x0505)]
        for f in after:
            _send_and_read_final(ws, f)
        # One wake frame only, and no more frames for the spotter after it.
        assert barrier(ws) == []
        assert _stop(ws) == [_stopped(500)]
    assert spotter.streams[0].fed == [_pcm(160), _pcm(160, HEAR_FIRST)]
    # The frames before the wake, and the frame of the wake, never reach the STT.
    assert whispers[0].fed == after


def test_the_wake_frame_names_the_phrase_that_was_heard(client, spotter):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws, ["Hey Hermes", "hey computer"]) == _ready_frame(
            ["hey hermes", "hey computer"])
        ws.send_bytes(_pcm(160, HEAR_LAST))
        assert ws.receive_json() == _wake("hey computer", 0, 160)


def test_the_wake_count_is_the_audio_accepted_before_the_connection_was_dormant(
        client, whispers):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        _send_and_read_final(ws, _pcm(256))
        _send_and_read_final(ws, _pcm(300))
        ws.send_json(WAKE_WORD)
        assert ws.receive_json() == _ready_frame()
        ws.send_bytes(_pcm(4000))
        ws.send_bytes(_pcm(160, HEAR_FIRST))
        # Dormant audio is not counted: the next accepted frame starts at 556.
        assert ws.receive_json() == _wake("hey computer", 556, 4160)
        _send_and_read_final(ws, _pcm(100))
        assert _stop(ws) == [_stopped(656)]


def test_the_dormant_count_goes_on_over_a_listen_stop_while_dormant(client, spotter):
    """A listen_stop on a dormant connection opens no new stream, so the count of
    dormant audio is not started again."""
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws) == _ready_frame()
        ws.send_bytes(_pcm(1000))
        assert _stop(ws) == [_stopped(0)]
        ws.send_bytes(_pcm(160, HEAR_FIRST))
        assert ws.receive_json() == _wake("hey computer", 0, 1160)


def test_the_dormant_count_starts_again_at_a_second_wake_word(client, spotter):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws) == _ready_frame()
        ws.send_bytes(_pcm(1000))
        ws.send_json(WAKE_WORD)
        assert ws.receive_json() == _ready_frame()
        ws.send_bytes(_pcm(160, HEAR_FIRST))
        assert ws.receive_json() == _wake("hey computer", 0, 160)


def test_the_dormant_count_is_in_whole_samples(client, spotter):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws) == _ready_frame()
        ws.send_bytes(b"\x01\x02\x03")
        ws.send_bytes(_pcm(160, HEAR_FIRST))
        assert ws.receive_json() == _wake("hey computer", 0, 161)


# --- back to dormant, and a wake by hand --------------------------------------------

def test_listen_stop_after_a_wake_makes_the_connection_dormant_on_a_new_stream(
        client, spotter, whispers):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws, ["hey hermes"]) == _ready_frame(["hey hermes"])
        ws.send_bytes(_pcm(160, HEAR_FIRST))
        assert ws.receive_json() == _wake("hey hermes", 0, 160)
        _send_and_read_final(ws, _pcm(256))
        assert _stop(ws) == [_stopped(256)]
        # Dormant again: the frames go to a new stream with the same phrases.
        ws.send_bytes(_pcm(160, 0x0606))
        assert barrier(ws) == []
        ws.send_bytes(_pcm(160, HEAR_FIRST))
        assert ws.receive_json() == _wake("hey hermes", 256, 320)
        _send_and_read_final(ws, _pcm(64))
        assert _stop(ws) == [_stopped(320)]
    first, second, third = spotter.streams
    assert first.phrases == second.phrases == third.phrases == ["hey hermes"]
    assert first.fed == [_pcm(160, HEAR_FIRST)]
    assert second.fed == [_pcm(160, 0x0606), _pcm(160, HEAR_FIRST)]
    assert third.fed == []
    assert whispers[0].fed == [_pcm(256), _pcm(64)]


def test_listen_stop_while_dormant_answers_again_and_keeps_the_stream(client, spotter):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws) == _ready_frame()
        ws.send_bytes(_pcm(160))
        assert _stop(ws) == [_stopped(0)]
        assert _stop(ws) == [_stopped(0)]
        ws.send_bytes(_pcm(160, 0x0707))
        assert barrier(ws) == []
    assert len(spotter.streams) == 1
    assert spotter.streams[0].fed == [_pcm(160), _pcm(160, 0x0707)]


def test_listen_start_wakes_a_dormant_connection_with_no_wake_frame(
        client, spotter, whispers):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws) == _ready_frame()
        ws.send_bytes(_pcm(160))
        ws.send_json(START)
        after = [_pcm(200, 0x0404), _pcm(160, HEAR_FIRST)]
        for f in after:
            _send_and_read_final(ws, f)
        # No wake frame: not for the listen_start, and not for the marked frame,
        # which went to the STT and not to the spotter.
        assert barrier(ws) == []
        # The connection is still opted in: a stop makes it dormant again.
        assert _stop(ws) == [_stopped(360)]
        ws.send_bytes(_pcm(160, HEAR_FIRST))
        assert ws.receive_json() == _wake("hey computer", 360, 160)
    assert whispers[0].fed == after
    assert spotter.streams[0].fed == [_pcm(160)]
    assert spotter.streams[1].fed == [_pcm(160, HEAR_FIRST)]


# --- the phrase list ----------------------------------------------------------------

def test_a_phrase_list_is_for_its_connection_only(client, spotter):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws, ["hey hermes"]) == _ready_frame(["hey hermes"])
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws) == _ready_frame(DEFAULT_PHRASES)
    assert [s.phrases for s in spotter.streams] == [["hey hermes"], DEFAULT_PHRASES]
    assert spotter.default_phrases == DEFAULT_PHRASES


def test_a_null_phrase_list_means_the_default(client):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        ws.send_json({"type": "wake_word", "phrases": None})
        assert ws.receive_json() == _ready_frame()


def test_a_second_wake_word_replaces_the_phrases(client, spotter):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws) == _ready_frame()
        ws.send_json({"type": "wake_word", "phrases": ["hey hermes"]})
        assert ws.receive_json() == _ready_frame(["hey hermes"])
        ws.send_bytes(_pcm(160, HEAR_FIRST))
        assert ws.receive_json() == _wake("hey hermes", 0, 160)
    assert spotter.streams[0].fed == []


@pytest.mark.parametrize("phrases, reason", [
    ([], "empty"),
    (["hey computer", "  "], "empty"),
    (["hey computer 9"], "not a letter"),
    (["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"], "too many"),
    (["hey zzz"], "no tokens"),
    ("hey computer", "list of strings"),
    ([7], "list of strings"),
])
def test_a_phrase_list_the_server_cannot_use_is_refused(
        client, spotter, whispers, phrases, reason):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        answer = _opt_in(ws, phrases)
        assert answer["type"] == "error" and answer["error_type"] == "invalid_request", answer
        assert reason in answer["message"]
        # One error frame, no wake_word_ready, and the connection still listens.
        assert barrier(ws) == []
        _send_and_read_final(ws, _pcm(64))
        assert _stop(ws) == [_stopped(64)]
        # It did not opt in: stopped frames are dropped, not given to a spotter.
        ws.send_bytes(_pcm(160, HEAR_FIRST))
        assert barrier(ws) == []
    assert spotter.streams == []
    assert whispers[0].fed == [_pcm(64)]


def test_a_refused_second_wake_word_keeps_the_first_phrases_and_the_dormant_state(
        client, spotter):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws, ["hey hermes"]) == _ready_frame(["hey hermes"])
        ws.send_json({"type": "wake_word", "phrases": ["hey computer 9"]})
        answer = ws.receive_json()
        assert answer["error_type"] == "invalid_request", answer
        ws.send_bytes(_pcm(160, HEAR_FIRST))
        assert ws.receive_json() == _wake("hey hermes", 0, 160)
    assert len(spotter.streams) == 1


# --- a connection that does not opt in ----------------------------------------------

def test_a_connection_with_no_opt_in_gives_the_spotter_nothing(client, spotter, whispers):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _ready(ws)
        _send_and_read_final(ws, _pcm(256))
        assert _stop(ws) == [_stopped(256)]
        ws.send_bytes(_pcm(512))
        ws.send_bytes(_pcm(160, HEAR_FIRST))
        # The stopped frames are dropped as before: only listen_stopped came.
        assert barrier(ws) == []
        ws.send_json(START)
        _send_and_read_final(ws, _pcm(64))
        assert _stop(ws) == [_stopped(320)]
    assert spotter.streams == []
    assert whispers[0].fed == [_pcm(256), _pcm(64)]


# --- with the turn engine -------------------------------------------------------------

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


def test_dormant_audio_is_not_fed_to_the_turn_engine(tyger, turn_engines, spotter):
    app = make_app(sidecar=StandInSidecar(tyger.url))
    app.state.wake_spotter = spotter
    with authkit.ws_connect(TestClient(app), "/ws/stream") as ws:
        # The wake word first, then the turn engine: dormant audio is not the
        # connection's first audio frame, so the turn-engine opt-in still succeeds.
        assert _opt_in(ws) == _ready_frame()
        ws.send_bytes(_pcm(4000))
        ws.send_json({"type": "turn_engine", "engine": "tyger"})
        assert recv_until(ws, is_type("turn_engine_ready"))[-1] == {
            "type": "turn_engine_ready", "engine": "tyger"}
        assert tyger.wait_for(lambda: len(tyger.sessions) == 1)
        ws.send_bytes(_pcm(2000))
        ws.send_bytes(_pcm(160, HEAR_FIRST))
        assert ws.receive_json() == _wake("hey computer", 0, 6160)
        ws.send_bytes(_pcm(1500))
        ws.send_bytes(_pcm(700))
        assert errors(barrier(ws)) == []
        assert _stop(ws)[-1] == _stopped(2200)
        # The engine was fed the audio after the wake and none of the dormant audio.
        feed = turn_engines[0]._feed
        assert feed.mic_samples == 2200
        # Dormant again: the turn engine's count stays with mic_sample.
        ws.send_bytes(_pcm(3000))
        assert barrier(ws) == []
        assert feed.mic_samples == 2200
    assert [len(f) for f in spotter.streams[0].fed] == [8000, 4000, 320]
    assert [len(f) for f in spotter.streams[1].fed] == [6000]


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
    monkeypatch.setattr(ws_module, "WARNING_LEAD", LEAD, raising=False)
    cfg._settings = None
    yield
    cfg._settings = None


def test_a_dormant_connection_does_not_time_out(client, short_timeout):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws) == _ready_frame()
        time.sleep(TIMEOUT + 0.3)
        # Nothing came while dormant: no warning, no timeout error, no close.
        assert barrier(ws) == []
        assert _stop(ws) == [_stopped(0)]


def test_the_session_timeout_runs_again_from_the_wake(client, short_timeout):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert _opt_in(ws) == _ready_frame()
        time.sleep(TIMEOUT + 0.3)
        ws.send_bytes(_pcm(160, HEAR_FIRST))
        since = time.monotonic()
        assert ws.receive_json() == _wake("hey computer", 0, 160)
        warning = ws.receive_json()
        assert warning.get("error_type") == "timeout_warning", warning
        assert time.monotonic() - since >= TIMEOUT - LEAD - 0.1
        closing = ws.receive_json()
        assert closing.get("error_type") == "timeout", closing
        assert time.monotonic() - since >= TIMEOUT - 0.1
        closed = ws.receive()
        assert closed["type"] == "websocket.close" and closed["code"] == 1001, closed
        assert time.monotonic() - since <= TIMEOUT + SLACK
