"""The mic hold of an opted-in session: what reaches the STT while a reply plays.

While a reply plays at the client, the mic picks it up, so the server holds mic
audio back from the STT from the moment it sends the reply's audio_start:

* the playback start frees the held audio before its mic_sample, fed in order;
* audio from the start to the stop's mic_sample plus the hold tail stays held and,
  with no barge, is dropped;
* a barge feeds the held audio from its onset (at most 4 s back) before any later
  audio;
* with no playback start by the start wait after the reply's audio_end, the held
  audio is fed in order, as if the reply never played;
* a connection that never opted in is never held.

The turn engine's hop feed still sees every mic sample; only the STT is held. The STT
is a recording stand-in for the audiocpp pipeline, the sidecar the in-process
FakeTygerSidecar. No GPU, no model, no live server.
"""
import contextlib
import json
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
    FakeTTSEngine, StandInSidecar, barrier, errors, make_app, opt_in, speak,
)

#: Samples per mic frame the tests send.
FRAME = 1024
#: The default hold tail, 150 ms at 16 kHz.
TAIL = 150 * 16
#: The client's whole mic stream: a ramp, so any slice shows where it came from.
_STREAM = ((np.arange(40 * 16000) % 30000) + 1).astype("<i2").tobytes()


def _s(a: int, b: int) -> bytes:
    """Mic samples [a, b) of the client's stream."""
    return _STREAM[2 * a:2 * b]


class _Mic:
    """Sends the client's mic stream in FRAME-sample frames and counts what went."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.at = 0
        self.frames: list[bytes] = []

    def send(self, n: int) -> None:
        end = self.at + n
        while self.at < end:
            step = min(FRAME, end - self.at)
            data = _s(self.at, self.at + step)
            self.ws.send_bytes(data)
            self.frames.append(data)
            self.at += step


class _RecordingSTT:
    """A stand-in for AudioCppSTTPipeline that records every call, in order."""

    def __init__(self) -> None:
        self.events: list = []

    async def feed(self, data: bytes) -> bool:
        self.events.append(("feed", bytes(data)))
        return False

    def get_current_text(self):
        return ("", "en", 0)

    def take_final(self):
        return None

    async def finalize(self):
        self.events.append(("finalize",))
        return ("", "en", 0)

    def reset(self) -> None:
        pass

    def dispose(self) -> None:
        pass

    @property
    def feeds(self) -> list[bytes]:
        return [e[1] for e in self.events if e[0] == "feed"]

    @property
    def fed(self) -> bytes:
        return b"".join(self.feeds)


def _start(request_id: str, mic_sample: int) -> dict:
    return {"type": "playback", "request_id": request_id, "event": "start",
            "mic_sample": mic_sample}


def _stop(request_id: str, mic_sample: int, reply_sample: int = 4800) -> dict:
    return {"type": "playback", "request_id": request_id, "event": "stop",
            "mic_sample": mic_sample, "reply_sample": reply_sample}


def _chunks(n: int = 2) -> list[np.ndarray]:
    return [np.full(2400, 0.1 * (i + 1), dtype=np.float32) for i in range(n)]


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
    # These tests pin the mic audio the hold lets go; the noise lead-in a barge
    # puts in front of it has its own tests.
    monkeypatch.setenv("VM_TYGER_BARGE_LEAD_MS", "0")
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
    """The recording STT pipelines the connections made."""
    pipelines: list[_RecordingSTT] = []

    def factory(config=None, **kwargs):
        pipelines.append(_RecordingSTT())
        return pipelines[-1]

    monkeypatch.setattr(audiocpp_stt, "AudioCppSTTPipeline", factory)
    return pipelines


@pytest.fixture
def client(tyger, executor, made, env):
    # Not entered as a context manager: that would run the real lifespan.
    return contextlib.nullcontext(TestClient(make_app(
        sidecar=StandInSidecar(tyger.url), executor=executor,
        tts_engine=FakeTTSEngine(_chunks()))))


def _opted_in(ws, tyger) -> None:
    opt_in(ws)
    assert tyger.wait_for(lambda: len(tyger.sessions) == 1)


def _events(ws, tyger, dispatched: list, *events) -> list:
    want = len(dispatched) + len(events)
    tyger.send(0, *events)
    assert tyger.wait_for(lambda: len(dispatched) >= want, timeout=5), dispatched
    return barrier(ws)


# --- the start frame, the stop frame and the tail ----------------------------------

def test_the_start_releases_audio_before_it_and_the_rest_is_dropped_to_the_tail(
        tyger, client, made):
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        mic.send(2048)                       # before any reply: straight to the STT
        assert errors(barrier(ws)) == []
        stt = made[0]
        assert stt.feeds == mic.frames       # frame for frame, as without a turn engine

        speak(ws, "r1")                      # its audio_start went out at sample 2048
        mic.send(8192)
        barrier(ws)
        assert stt.fed == _s(0, 2048)        # held

        ws.send_json(_start("r1", 4000))
        assert errors(barrier(ws)) == []
        assert stt.fed == _s(0, 4000)        # released up to the start, in order

        mic.send(8192)                       # to 18432, the reply still plays
        barrier(ws)
        assert stt.fed == _s(0, 4000)

        ws.send_json(_stop("r1", 12000))
        assert errors(barrier(ws)) == []
        # [4000, 12000 + tail) played over the mic: dropped. The rest goes on.
        assert stt.fed == _s(0, 4000) + _s(12000 + TAIL, 18432)

        mic.send(2048)
        barrier(ws)
        assert stt.fed == _s(0, 4000) + _s(12000 + TAIL, 20480)
    hops = tyger.sessions[0].hops
    assert b"".join(h.user for h in hops) == _s(0, len(hops) * wire.HOP_SAMPLES)


def test_the_tail_follows_the_hold_tail_setting(tyger, client, made, env):
    env(VM_TYGER_HOLD_TAIL_MS=500)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        speak(ws, "r1")                      # holding from sample 0
        mic.send(20480)
        ws.send_json(_start("r1", 1000))
        ws.send_json(_stop("r1", 6000))
        assert errors(barrier(ws)) == []
        assert made[0].fed == _s(0, 1000) + _s(6000 + 500 * 16, 20480)


def test_audio_still_in_the_tail_stays_held_until_the_tail_passes(tyger, client, made):
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        speak(ws, "r1")
        mic.send(4096)
        ws.send_json(_start("r1", 0))
        ws.send_json(_stop("r1", 4000))      # the tail runs to 4000 + TAIL, past 4096
        barrier(ws)
        assert made[0].fed == b""
        mic.send(4096)
        barrier(ws)
        assert made[0].fed == _s(4000 + TAIL, 8192)


# --- a barge -------------------------------------------------------------------------

def test_a_barge_feeds_the_held_audio_from_its_onset_before_later_audio(
        tyger, client, made, dispatched):
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        mic.send(2048)
        speak(ws, "r1")
        mic.send(16000)                      # to 18048
        ws.send_json(_start("r1", 3000))
        assert errors(barrier(ws)) == []
        assert made[0].fed == _s(0, 3000)

        frames = _events(ws, tyger, dispatched,
                         wire.BargeEvent(onset_mic_sample=10000, mic_sample=15360))
        assert [f["type"] for f in frames if isinstance(f, dict)] == ["barge"]
        assert made[0].fed == _s(0, 3000) + _s(10000, 18048)

        mic.send(2048)                       # after the barge: no hold
        ws.send_json(_stop("r1", 16000))     # the client's answer changes nothing fed
        barrier(ws)
        mic.send(1024)
        barrier(ws)
        assert made[0].fed == _s(0, 3000) + _s(10000, 21120)


def test_a_barge_feeds_at_most_four_seconds_back(tyger, client, made, dispatched, env):
    # The reply is short and the client reports no stop: its open end (the start plus
    # its length plus the start wait) must be past the barge for it to be in play.
    env(VM_TYGER_START_WAIT_MS=10000)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        speak(ws, "r1")
        mic.send(3072)
        ws.send_json(_start("r1", 3000))
        mic.send(7 * 16000)                  # to 115072
        barrier(ws)
        assert made[0].fed == _s(0, 3000)
        _events(ws, tyger, dispatched,
                wire.BargeEvent(onset_mic_sample=4000, mic_sample=112640))
        assert made[0].fed == _s(0, 3000) + _s(112640 - 4 * 16000, 115072)


def test_a_barge_then_a_turn_end_feeds_the_held_audio_before_the_flush(
        tyger, client, made, dispatched, env):
    env(VM_AUDIOCPP_SILENCE_MS=30000)       # no frame-gap flush of its own
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        speak(ws, "r1")
        mic.send(16384)
        ws.send_json(_start("r1", 0))
        barrier(ws)
        assert made[0].events == []
        _events(ws, tyger, dispatched,
                wire.BargeEvent(onset_mic_sample=8000, mic_sample=15360),
                wire.EndpointEvent(by="model", mic_sample=16384))
    events = made[0].events
    flush = events.index(("finalize",))
    assert b"".join(e[1] for e in events[:flush]) == _s(8000, 16384)


# --- no start frame ----------------------------------------------------------------

def test_with_no_start_by_the_start_wait_the_held_audio_is_fed_in_order(
        tyger, client, made, env):
    env(VM_TYGER_START_WAIT_MS=700)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        mic.send(1024)
        before_end = time.monotonic()
        speak(ws, "r1")                      # audio_end went out; no start follows
        mic.send(6144)
        barrier(ws)
        assert made[0].fed == _s(0, 1024)    # not before the wait
        assert tyger.wait_for(lambda: made[0].fed == _s(0, 7168), timeout=5), \
            len(made[0].fed)
        assert time.monotonic() - before_end >= 0.7
        # The hold for it is over: a late start holds nothing.
        ws.send_json(_start("r1", 2000))
        mic.send(1024)
        barrier(ws)
        assert made[0].fed == _s(0, 8192)


# --- no opt-in ---------------------------------------------------------------------

def test_a_connection_that_did_not_opt_in_is_never_held(tyger, client, made):
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        mic = _Mic(ws)
        mic.send(2048)
        speak(ws, "r1")
        mic.send(8192)
        ws.send_json(_start("r1", 4000))     # dropped unanswered without an opt-in
        mic.send(2048)
        frames = barrier(ws)
    assert errors(frames) == []
    assert made[0].feeds == mic.frames
    assert tyger.sessions == []


# --- the hold on its own: the memory bounds ----------------------------------------

def _audio_start(rate: int = 24000) -> str:
    return json.dumps({"type": "audio_start", "request_id": "r1", "sample_rate": rate})


def _drive(n_frames: int, *, start: int | None, start_wait_samples: int = 32000):
    """Feed n_frames mic frames into a hold with r1 sent at sample 0; per frame, the
    samples held and everything fed."""
    feed = te.HopFeed()
    hold = te.MicHold(feed, tail_samples=TAIL, start_wait_samples=start_wait_samples)
    feed.on_tts_text("r1", _audio_start())
    feed.on_tts_pcm("r1", b"\x01\x00" * 24000)
    if start is not None:
        assert feed.playback(_start("r1", start)) is None
    held, fed = [], []
    for i in range(n_frames):
        data = _s(i * FRAME, (i + 1) * FRAME)
        at = feed.mic_samples
        hold.push(at, data, feed.feed_mic(data))
        fed += hold.take()
        held.append(hold.held_samples)
    return held, b"".join(fed)


def test_audio_in_play_is_kept_only_for_the_barge_lookback():
    n_frames = 30 * 16000 // FRAME
    held, fed = _drive(n_frames, start=0)
    bound = te.BARGE_LOOKBACK_SAMPLES + te.HOLD_MARGIN_SAMPLES
    assert max(held) <= bound + FRAME
    assert held[-1] >= te.BARGE_LOOKBACK_SAMPLES
    assert fed == b""                        # a reply with no stop: all of it is echo


def test_audio_before_a_start_is_kept_only_for_the_start_wait_and_then_fed():
    n_frames = 30 * 16000 // FRAME
    held, fed = _drive(n_frames, start=None, start_wait_samples=32000)
    bound = 32000 + te.HOLD_MARGIN_SAMPLES
    assert max(held) <= bound + FRAME
    total = n_frames * FRAME
    assert fed == _s(0, total - held[-1])    # the oldest goes on, in order


# --- a reply that ended with no stop: its hold ends at its open end ----------------

#: r1 is 1 s long and plays from mic sample 3000: its open end is the start plus the
#: played length plus the start wait.
_NO_STOP_START = 3000
_OPEN_END = _NO_STOP_START + 16000 + 32000


def _no_stop_hold(*, progress: bool):
    """r1 sent at sample 0, 1 s at 16 kHz, done, started at 3000 and never stopped."""
    feed = te.HopFeed()
    hold = te.MicHold(feed, tail_samples=TAIL, start_wait_samples=32000)
    feed.on_tts_text("r1", _audio_start(16000))
    feed.on_tts_pcm("r1", b"\x01\x00" * 16000)
    feed.on_tts_text("r1", json.dumps({"type": "audio_end", "request_id": "r1"}))
    assert feed.playback(_start("r1", _NO_STOP_START)) is None
    if progress:
        hold.on_scored(0)
    return feed, hold


def _push_to(feed, hold, end: int) -> list[tuple[int, bytes]]:
    """Push mic frames up to sample ``end``; (mic count, audio fed) per frame."""
    out = []
    while feed.mic_samples < end:
        at = feed.mic_samples
        data = _s(at, min(at + FRAME, end))
        hold.push(at, data, feed.feed_mic(data))
        out.append((feed.mic_samples, b"".join(hold.take())))
    return out


def test_a_reply_with_no_stop_frees_the_audio_past_its_open_end_on_progress():
    feed, hold = _no_stop_hold(progress=True)
    steps = _push_to(feed, hold, 60 * FRAME)
    assert b"".join(c for _, c in steps) == _s(0, _NO_STOP_START)
    assert hold.on_scored(_OPEN_END + TAIL - 1) is False
    assert hold.take() == []
    assert hold.on_scored(_OPEN_END + TAIL) is True
    assert b"".join(hold.take()) == _s(_OPEN_END, 60 * FRAME)
    # From the open end on, the STT gets the mic stream as sent, in order.
    later = _push_to(feed, hold, 70 * FRAME)
    assert b"".join(c for _, c in later) == _s(60 * FRAME, 70 * FRAME)


def test_a_reply_with_no_stop_and_a_slow_sidecar_frees_the_audio_by_the_start_wait():
    feed, hold = _no_stop_hold(progress=True)
    steps = _push_to(feed, hold, 140 * FRAME)
    fed = b"".join(c for _, c in steps)
    # Nothing past the reply's start until the mic count is the open end plus the
    # tail plus the start wait: the first frame boundary at or past it.
    bound = _OPEN_END + TAIL + 32000
    first = next(count for count, chunk in steps if count > FRAME * 3 and chunk)
    assert first == -(-bound // FRAME) * FRAME
    assert first != 131072
    assert fed == _s(0, _NO_STOP_START) + _s(_OPEN_END, 140 * FRAME)


def test_a_reply_with_no_stop_and_no_progress_reported_frees_the_audio_at_its_open_end():
    feed, hold = _no_stop_hold(progress=False)
    steps = _push_to(feed, hold, 60 * FRAME)
    first = next(count for count, chunk in steps if count > FRAME * 3 and chunk)
    assert first == -(-_OPEN_END // FRAME) * FRAME
    assert b"".join(c for _, c in steps) == _s(0, _NO_STOP_START) + _s(_OPEN_END, 60 * FRAME)
