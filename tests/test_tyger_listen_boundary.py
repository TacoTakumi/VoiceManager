"""The turn engine at a listening boundary, over the real WebSocket handler, model-free.

A client that stops listening sends {"type": "listen_stop"} and later
{"type": "listen_start"}. For an opted-in session nothing from the audio before the
stop crosses that boundary:

* the first turn_end after listen_start holds only finals from audio after it;
* the sidecar gets the boundary control after every hop before the stop, and no hop
  with pre-stop mic audio after it (the partial hop at the stop is dropped); the
  hops after listen_start go on with the connection's mic count;
* an endpoint or barge event at or before the boundary sends nothing to the client,
  also when it comes in after listen_stopped;
* playback frames while stopped get no error and track no reply, a reply spoken
  while stopped holds no mic audio, and after listen_start playback and barge work as
  usual;
* mic audio the hold kept from before the stop never reaches the STT: not by a late
  playback start, not by the start wait running out while stopped;
* the sidecar link stays the same one (its voice gate and enrollment live there),
  and a session whose sidecar was lost answers listen_stop as usual.

The STT is a stand-in for the audiocpp pipeline that names the audio it was fed, the
TTS a fake engine, the sidecar the in-process FakeTygerSidecar. No GPU, no model, no
live server.
"""
import contextlib
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)
from fastapi.testclient import TestClient

import voicemanager.config as cfg
from voicemanager.pipeline import audiocpp_stt
from voicemanager.pipeline.tyger import wire
from voicemanager.transport import turn_engine as te

from tyger_fake_sidecar import FakeTygerSidecar
from tyger_ws_kit import (
    FakeTTSEngine, StandInSidecar, barrier, errors, is_type, make_app, opt_in,
    recv_until, speak,
)

STOP = {"type": "listen_stop"}
START = {"type": "listen_start"}
#: Samples per mic frame the tests send.
FRAME = 1024
HOP = wire.HOP_SAMPLES
_PRE_VALUE = 0x1111
_POST_VALUE = 0x2222


def _audio(value: int, samples: int) -> bytes:
    return value.to_bytes(2, "little") * samples


def _name(data: bytes) -> str:
    """The name of the audio in data: "pre", "post" or "other"."""
    first = int.from_bytes(data[:2], "little")
    return {_PRE_VALUE: "pre", _POST_VALUE: "post"}.get(first, "other")


class _NamingSTT:
    """An audiocpp stand-in: every fed frame gives one final naming its audio."""

    def __init__(self) -> None:
        self.feeds: list[bytes] = []
        self._pending: str | None = None
        self._segment = 0

    async def feed(self, data: bytes) -> bool:
        self.feeds.append(bytes(data))
        self._pending = _name(data)
        return True

    def get_current_text(self):
        return ("", "en", self._segment)   # no interims: only finals go out

    def take_final(self):
        name, self._pending = self._pending, None
        return None if name is None else (name, "en", self._segment)

    async def finalize(self):
        return ("", "en", self._segment)

    def reset(self) -> None:
        self._segment += 1

    def dispose(self) -> None:
        pass

    @property
    def fed(self) -> bytes:
        return b"".join(self.feeds)


class _Mic:
    """Sends the client's mic audio in FRAME-sample frames and counts the samples."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.at = 0

    def send(self, value: int, n: int) -> bytes:
        sent = b""
        while n > 0:
            step = min(FRAME, n)
            data = _audio(value, step)
            self.ws.send_bytes(data)
            sent += data
            self.at += step
            n -= step
        return sent


def _start(request_id: str, mic_sample: int) -> dict:
    return {"type": "playback", "request_id": request_id, "event": "start",
            "mic_sample": mic_sample}


def _stop(request_id: str, mic_sample: int, reply_sample: int = 4800) -> dict:
    return {"type": "playback", "request_id": request_id, "event": "stop",
            "mic_sample": mic_sample, "reply_sample": reply_sample}


def _of_type(frames: list, kind: str) -> list:
    return [f for f in frames if isinstance(f, dict) and f.get("type") == kind]


def _finals(frames: list) -> list[str]:
    return [f["text"] for f in _of_type(frames, "transcription") if f["is_final"]]


# --- fixtures -------------------------------------------------------------------

@pytest.fixture
def tyger():
    fake = FakeTygerSidecar().start()
    try:
        yield fake
    finally:
        fake.stop()


@pytest.fixture
def executor():
    with ThreadPoolExecutor(max_workers=1) as pool:
        yield pool


@pytest.fixture
def env(monkeypatch):
    """Set VM_* settings for the connection under test."""
    def setenv(**values) -> None:
        for key, value in values.items():
            monkeypatch.setenv(key, str(value))
        cfg._settings = None
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "60")
    monkeypatch.setenv("VM_AUDIOCPP_SILENCE_MS", "30000")   # no frame-gap flush
    cfg._settings = None
    yield setenv
    cfg._settings = None


@pytest.fixture
def dispatched(monkeypatch) -> list:
    """Every sidecar event the server has finished handling, in order."""
    done = []
    original = te.TurnEngineSession._dispatch

    async def recording(self, event):
        try:
            await original(self, event)
        finally:
            done.append(event)
    monkeypatch.setattr(te.TurnEngineSession, "_dispatch", recording)
    return done


@pytest.fixture
def made(monkeypatch) -> list:
    """The naming STT pipelines the connections made."""
    pipelines: list[_NamingSTT] = []

    def factory(config=None, **kwargs):
        pipelines.append(_NamingSTT())
        return pipelines[-1]

    monkeypatch.setattr(audiocpp_stt, "AudioCppSTTPipeline", factory)
    return pipelines


@pytest.fixture
def client(tyger, executor, made, env, dispatched):
    # One second of reply at 24 kHz: 16000 samples on the mic stream once played.
    chunks = [np.full(2400, 0.05 * (i + 1), dtype=np.float32) for i in range(10)]
    # Not entered as a context manager: that would run the real lifespan.
    return contextlib.nullcontext(TestClient(make_app(
        sidecar=StandInSidecar(tyger.url), executor=executor,
        tts_engine=FakeTTSEngine(chunks))))


def _opted_in(ws, tyger) -> None:
    opt_in(ws)
    assert tyger.wait_for(lambda: len(tyger.sessions) == 1)


def _events(ws, tyger, dispatched: list, *events) -> list:
    """Have the sidecar send events; every frame the server sent once it handled them."""
    want = len(dispatched) + len(events)
    tyger.send(0, *events)
    assert tyger.wait_for(lambda: len(dispatched) >= want, timeout=5), dispatched
    return barrier(ws)


def _listen_stop(ws, mic: _Mic) -> list:
    """Send listen_stop and read up to its answer; the frames before it."""
    ws.send_json(STOP)
    frames = recv_until(ws, is_type("listen_stopped"))
    assert frames[-1] == {"type": "listen_stopped", "mic_sample": mic.at}
    return frames[:-1]


# --- the turn_end ------------------------------------------------------------------

def test_the_first_turn_end_after_listen_start_holds_only_post_start_finals(
        tyger, client, made, dispatched):
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        mic.send(_PRE_VALUE, 3 * FRAME)
        assert _finals(barrier(ws)) == ["pre"] * 3     # sent, and pending a turn_end
        _listen_stop(ws, mic)
        ws.send_json(START)
        mic.send(_POST_VALUE, 2 * FRAME)
        assert _finals(barrier(ws)) == ["post"] * 2
        frames = _events(ws, tyger, dispatched,
                         wire.EndpointEvent(by="model", mic_sample=mic.at))
    turn_ends = _of_type(frames, "turn_end")
    assert len(turn_ends) == 1, frames
    assert turn_ends[0]["text"] == "post post"
    assert turn_ends[0]["mic_sample"] == 5 * FRAME
    assert errors(frames) == []


# --- the sidecar side -------------------------------------------------------------

@pytest.mark.parametrize("awaiting_start", [False, True])
def test_the_sidecar_gets_the_boundary_after_every_whole_hop_and_no_pre_stop_hop_after(
        tyger, client, made, awaiting_start):
    """Two whole hops and a partial one before the stop. With a reply that has no
    playback start yet, the second whole hop still waits for a late start at the
    stop; it is cut and sent before the boundary all the same."""
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        if awaiting_start:
            speak(ws, "r1")
        mic.send(_PRE_VALUE, 2 * HOP + 1000)
        barrier(ws)
        boundary = mic.at
        _listen_stop(ws, mic)
        session = tyger.sessions[0]
        assert tyger.wait_for(lambda: session.boundaries == [2]), session.boundaries
        assert len(session.hops) == 2
        ws.send_json(START)
        post = mic.send(_POST_VALUE, HOP + 100)
        assert tyger.wait_for(lambda: len(session.hops) == 3), len(session.hops)
        barrier(ws)
        time.sleep(0.05)
        # The same link all along: the voice gate and enrollment it holds are kept.
        assert len(tyger.sessions) == 1 and not session.closed
    hops = session.hops
    assert len(hops) == 3                                # the 100 left is a partial hop
    assert [h.mic_sample for h in hops] == [HOP, 2 * HOP, boundary + HOP]
    assert hops[0].user == hops[1].user == _audio(_PRE_VALUE, HOP)
    assert hops[2].user == post[:2 * HOP]                # starts fresh at the boundary
    assert session.bad == []


def test_events_at_or_before_the_boundary_send_nothing_even_after_listen_stopped(
        tyger, client, made, dispatched):
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        mic.send(_PRE_VALUE, FRAME)
        speak(ws, "r1")                                  # done; plays for 16000 samples
        ws.send_json(_start("r1", FRAME))
        mic.send(_PRE_VALUE, 4 * FRAME)
        assert errors(barrier(ws)) == []
        boundary = mic.at
        _listen_stop(ws, mic)
        # After the answer: a barge over r1 and an endpoint, both before the boundary.
        frames = _events(ws, tyger, dispatched,
                         wire.BargeEvent(onset_mic_sample=2 * FRAME, mic_sample=boundary),
                         wire.EndpointEvent(by="model", mic_sample=boundary))
        assert _of_type(frames, "barge") == [] and _of_type(frames, "turn_end") == []
        ws.send_json(START)
        mic.send(_POST_VALUE, FRAME)
        assert _finals(barrier(ws)) == ["post"]
        # Still from before the boundary, now with a final pending: still nothing.
        frames = _events(ws, tyger, dispatched,
                         wire.EndpointEvent(by="silence", mic_sample=boundary - 10),
                         wire.BargeEvent(onset_mic_sample=0, mic_sample=boundary))
        assert _of_type(frames, "turn_end") == [] and _of_type(frames, "barge") == []
        # The next endpoint after the boundary gets the post-start final.
        frames = _events(ws, tyger, dispatched,
                         wire.EndpointEvent(by="model", mic_sample=boundary + FRAME))
    assert [f["text"] for f in _of_type(frames, "turn_end")] == ["post"]
    assert errors(frames) == []


# --- playback and TTS while stopped ---------------------------------------------

def test_playback_while_stopped_is_quiet_and_tracks_no_reply_and_barge_works_after(
        tyger, client, made, dispatched):
    """Pinned: every playback frame while stopped is dropped with no answer, and a
    playback frame after listen_start for a reply forgotten at the boundary or sent
    while stopped is dropped with no answer too."""
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        mic.send(_PRE_VALUE, FRAME)
        speak(ws, "r1")
        ws.send_json(_start("r1", FRAME))
        mic.send(_PRE_VALUE, FRAME)
        assert errors(barrier(ws)) == []
        frames = _listen_stop(ws, mic)
        boundary = mic.at
        # The client stops r1 late, and speaks a goodbye while stopped.
        ws.send_json(_stop("r1", boundary))
        speak(ws, "r2")
        ws.send_json(_start("r2", boundary))
        ws.send_json(_stop("r2", boundary, 2400))
        ws.send_json(_start("r2", boundary))             # a repeat: still no answer
        frames += _events(ws, tyger, dispatched,
                          wire.BargeEvent(onset_mic_sample=boundary, mic_sample=boundary))
        assert errors(frames) == [] and _of_type(frames, "barge") == []

        ws.send_json(START)
        ws.send_json(_start("r2", boundary))             # late, for the goodbye
        ws.send_json(_stop("r2", boundary + 100, 2400))
        ws.send_json(_stop("r1", boundary + 100))
        fed = len(made[0].feeds)
        post = mic.send(_POST_VALUE, 2 * FRAME)
        frames = barrier(ws)
        assert errors(frames) == []
        # The goodbye holds no mic audio: the post-start frames go straight on.
        assert b"".join(made[0].feeds[fed:]) == post
        frames = _events(ws, tyger, dispatched,
                         wire.BargeEvent(onset_mic_sample=boundary + 10,
                                         mic_sample=boundary + FRAME))
        assert _of_type(frames, "barge") == []           # no reply is in play

        # A reply after listen_start is tracked as usual: held, then barged.
        speak(ws, "r3")
        at = mic.at
        ws.send_json(_start("r3", at))
        mic.send(_POST_VALUE, 2 * FRAME)
        assert errors(barrier(ws)) == []
        assert b"".join(made[0].feeds[fed:]) == post     # r3 holds the mic
        frames = _events(ws, tyger, dispatched,
                         wire.BargeEvent(onset_mic_sample=at + 100, mic_sample=at + FRAME))
        assert [(f["request_id"], f["mic_sample"]) for f in _of_type(frames, "barge")] \
            == [("r3", at + FRAME)]
        # Unknown request ids still get the usual answer.
        ws.send_json(_start("nope", mic.at))
        assert len(errors(barrier(ws))) == 1


# --- opting in while stopped ------------------------------------------------------

_OPT_IN = {"type": "turn_engine", "engine": "tyger"}
_STOPPED_AT_ZERO = {"type": "listen_stopped", "mic_sample": 0}


def _stopped_and_opted_in(ws, c, tyger, order: str) -> None:
    """Get the connection stopped at mic_sample 0 and opted in, in the given order.

    opt_in_first:   opt in, then listen_stop (the boundary goes to the sidecar)
    stop_first:     listen_stop before any audio, then opt in while stopped
    stop_again:     as stop_first, then a second listen_stop while stopped
    refused_first:  listen_stop, a refused opt-in, then a good one while stopped
    lost_first:     listen_stop, an opt-in whose sidecar cannot be reached, then a
                    good one while stopped
    """
    assert ws.receive_json()["type"] == "ready"
    if order == "opt_in_first":
        ws.send_json(_OPT_IN)
        assert ws.receive_json() == {"type": "turn_engine_ready", "engine": "tyger"}
        ws.send_json(STOP)
        assert ws.receive_json() == _STOPPED_AT_ZERO
        assert tyger.wait_for(lambda: len(tyger.sessions) == 1)
        return
    ws.send_json(STOP)
    assert ws.receive_json() == _STOPPED_AT_ZERO
    if order == "refused_first":
        ws.send_json({"type": "turn_engine", "engine": "nope"})
        (error,) = errors(barrier(ws))
        assert error["error_type"] == "invalid_request"
    elif order == "lost_first":
        sidecar = c.app.state.turn_engine_sidecar
        url, sidecar.url = sidecar.url, "ws://127.0.0.1:1/"
        ws.send_json(_OPT_IN)
        (error,) = errors(barrier(ws))
        assert error["error_type"] == "turn_engine_unavailable"
        sidecar.url = url
    ws.send_json(_OPT_IN)
    assert ws.receive_json() == {"type": "turn_engine_ready", "engine": "tyger"}
    assert tyger.wait_for(lambda: len(tyger.sessions) == 1)
    if order == "stop_again":
        ws.send_json(STOP)
        assert ws.receive_json() == _STOPPED_AT_ZERO


@pytest.mark.parametrize("order", ["opt_in_first", "stop_first", "stop_again",
                                   "refused_first", "lost_first"])
def test_a_reply_sent_while_stopped_is_not_tracked_whatever_the_order_of_stop_and_opt_in(
        tyger, client, made, dispatched, order):
    """A wall console connects, stops listening, opts in and waits for its wake word.
    The goodbye it speaks while stopped is not tracked, whether the stop came before
    or after the opt-in: its playback frames are dropped quietly, it holds no mic
    audio after listen_start and is no barge candidate. A reply after listen_start
    holds and barges as usual."""
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _stopped_and_opted_in(ws, c, tyger, order)
        mic = _Mic(ws)
        speak(ws, "r2")                                  # a goodbye while stopped
        ws.send_json(_start("r2", 0))
        ws.send_json(_stop("r2", 0, 2400))
        ws.send_json(_start("r2", 0))                    # a repeat: still no answer
        frames = _events(ws, tyger, dispatched,
                         wire.BargeEvent(onset_mic_sample=0, mic_sample=0))
        assert errors(frames) == [] and _of_type(frames, "barge") == []

        ws.send_json(START)
        ws.send_json(_start("r2", 0))                    # late, for the goodbye
        ws.send_json(_stop("r2", 100, 2400))
        fed = len(made[0].feeds)
        post = mic.send(_POST_VALUE, 2 * FRAME)
        frames = barrier(ws)
        assert errors(frames) == []
        # The goodbye holds no mic audio: the post-start frames go straight on.
        assert b"".join(made[0].feeds[fed:]) == post
        frames = _events(ws, tyger, dispatched,
                         wire.BargeEvent(onset_mic_sample=10, mic_sample=FRAME))
        assert _of_type(frames, "barge") == []           # no reply is in play

        # A reply after listen_start is tracked as usual: held, then barged.
        speak(ws, "r3")
        at = mic.at
        ws.send_json(_start("r3", at))
        mic.send(_POST_VALUE, 2 * FRAME)
        assert errors(barrier(ws)) == []
        assert b"".join(made[0].feeds[fed:]) == post     # r3 holds the mic
        frames = _events(ws, tyger, dispatched,
                         wire.BargeEvent(onset_mic_sample=at + 100, mic_sample=at + FRAME))
        assert [(f["request_id"], f["mic_sample"]) for f in _of_type(frames, "barge")] \
            == [("r3", at + FRAME)]
        assert errors(frames) == []
        session = tyger.sessions[0]
        # Only a stop on an opted-in session sends the sidecar a boundary; a session
        # opened while stopped has nothing from before the stop to fence off.
        want = [0] if order == "opt_in_first" else []
        assert tyger.wait_for(lambda: session.boundaries == want), session.boundaries
        assert len(tyger.sessions) == 1


def test_a_fresh_session_follows_a_stop_and_start_before_its_opt_in(
        tyger, client, made, dispatched):
    """A stop and a start before the opt-in leave the connection listening: a reply
    after the opt-in is tracked and holds the mic as usual."""
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_json(STOP)
        assert ws.receive_json() == _STOPPED_AT_ZERO
        ws.send_json(START)
        ws.send_json(_OPT_IN)
        assert ws.receive_json() == {"type": "turn_engine_ready", "engine": "tyger"}
        assert tyger.wait_for(lambda: len(tyger.sessions) == 1)
        mic = _Mic(ws)
        first = mic.send(_POST_VALUE, FRAME)
        speak(ws, "r1")
        ws.send_json(_start("r1", FRAME))
        mic.send(_POST_VALUE, 2 * FRAME)
        assert errors(barrier(ws)) == []
        assert made[0].fed == first                      # r1 holds the mic
        frames = _events(ws, tyger, dispatched,
                         wire.BargeEvent(onset_mic_sample=FRAME + 100,
                                         mic_sample=2 * FRAME))
    assert [(f["request_id"], f["mic_sample"]) for f in _of_type(frames, "barge")] \
        == [("r1", 2 * FRAME)]
    assert errors(frames) == []


# --- the mic hold -----------------------------------------------------------------

def test_held_pre_stop_audio_is_not_released_by_a_playback_start_after_listen_start(
        tyger, client, made):
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        first = mic.send(_PRE_VALUE, FRAME)
        speak(ws, "r1")                                  # holds the mic from FRAME on
        mic.send(_PRE_VALUE, 3 * FRAME)
        barrier(ws)
        assert made[0].fed == first                      # held
        _listen_stop(ws, mic)
        ws.send_json(START)
        post = mic.send(_POST_VALUE, 2 * FRAME)
        ws.send_json(_start("r1", 2 * FRAME))            # would free [FRAME, 2 * FRAME)
        post += mic.send(_POST_VALUE, FRAME)
        frames = barrier(ws)
    assert errors(frames) == []
    assert made[0].fed == first + post
    assert "pre" not in _finals(frames)


def test_the_start_wait_running_out_while_stopped_releases_no_pre_stop_audio(
        tyger, client, made, env):
    env(VM_TYGER_START_WAIT_MS=200)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        first = mic.send(_PRE_VALUE, FRAME)
        speak(ws, "r1")                                  # audio_end out: the wait runs
        mic.send(_PRE_VALUE, 3 * FRAME)
        barrier(ws)
        assert made[0].fed == first
        _listen_stop(ws, mic)
        time.sleep(0.5)                                  # past the start wait
        frames = barrier(ws)
        ws.send_json(START)
        post = mic.send(_POST_VALUE, 2 * FRAME)
        frames += barrier(ws)
    assert errors(frames) == []
    assert made[0].fed == first + post
    assert _finals(frames) == ["post", "post"]


# --- no turn engine, a lost one ---------------------------------------------------

def test_a_lost_turn_engine_answers_listen_stop_as_usual(tyger, client, made):
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        mic.send(_PRE_VALUE, FRAME)
        speak(ws, "r1")                                  # holds the mic
        mic.send(_PRE_VALUE, FRAME)
        barrier(ws)
        tyger.drop(0)
        lost = recv_until(ws, lambda f: isinstance(f, dict) and f.get("type") == "error")
        assert errors(lost)[-1]["error_type"] == "turn_engine_lost"
        frames = barrier(ws)
        _listen_stop(ws, mic)
        ws.send_json(START)
        post = mic.send(_POST_VALUE, FRAME)
        frames += barrier(ws)
    assert errors(frames) == []
    assert made[0].fed.endswith(post)


# --- the hop feed and the hold on their own ---------------------------------------

def test_the_hop_feed_cuts_whole_hops_at_the_boundary_and_starts_fresh_after():
    feed = te.HopFeed()
    feed.feed_mic(_audio(_PRE_VALUE, 2 * HOP + 300))
    assert [j.mic_sample for j in feed.due_hops()] == [HOP, 2 * HOP]
    feed.feed_mic(_audio(_PRE_VALUE, HOP))               # one whole hop and 300 more
    jobs = feed.listen_boundary()
    assert [j.mic_sample for j in jobs] == [3 * HOP]
    assert not feed.listening
    feed.listen_start()
    feed.feed_mic(_audio(_POST_VALUE, HOP))
    (job,) = feed.due_hops()
    assert job.mic_sample == 3 * HOP + 300 + HOP
    assert job.user == _audio(_POST_VALUE, HOP)
    assert feed.mic_samples == 4 * HOP + 300             # the count goes on


def test_the_hold_drops_everything_it_keeps_on_clear():
    feed = te.HopFeed()
    hold = te.MicHold(feed, tail_samples=0, start_wait_samples=32000)
    feed.on_tts_text("r1", '{"type": "audio_start", "request_id": "r1", '
                           '"sample_rate": 16000}')
    data = _audio(_PRE_VALUE, FRAME)
    hold.push(0, data, feed.feed_mic(data))
    assert hold.take() == [] and hold.held_samples == FRAME
    assert hold.clear() == FRAME
    assert hold.empty and hold.take() == []
