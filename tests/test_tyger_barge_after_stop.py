"""A barge that the sidecar scores after the reply's playback stop, model-free.

The reference client reports a stop by DAC time, so the stop frame can reach the
server before the mic audio it points to has been scored. A barge the user started
over the end of a reply then comes in after the stop. The server must still act on
it:

* a barge whose speech began by a stopped reply's stop picks that reply while the
  reply's mic hold still waits for the sidecar's progress: the barge frame goes
  out, then audio_end 'interrupted' if the request is still open, or alone if its
  audio_end already went out. That covers a barge decided late in the hop that
  holds the stop, and one from a hop sent in speaking mode before a late stop;
* a barge whose speech began after the stop, or one that comes once the hold for
  the reply has ended (dropped, or released by an earlier barge), does not pick it;
* the in-play mic audio of a stopped reply stays held until the sidecar's progress
  (its scored lines) passes the stop plus the tail, or the end of the last hop sent
  in speaking mode over the reply when that is later: then, with no barge, it is
  dropped, and with a barge it goes to the STT from the onset, before later audio;
* a sidecar that does not report progress, or one too slow to, cannot hold it for
  longer than the start wait of mic audio past the stop plus the tail.

The sidecar is the in-process FakeTygerSidecar with progress lines on, the STT a
recording stand-in, the TTS engine a fake. No GPU, no model, no live server.
"""
import contextlib
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as StarletteClient

import voicemanager.config as cfg
from voicemanager.pipeline import audiocpp_stt
from voicemanager.pipeline.tyger import wire
from voicemanager.transport import turn_engine as te

from tyger_fake_sidecar import FakeTygerSidecar
from tyger_ws_kit import (
    TTS_RATE, FakeTTSEngine, GatedTTSEngine, StandInSidecar, barrier, errors, frame,
    make_app, opt_in, speak,
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
    """Sends the client's mic stream in FRAME-sample frames."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.at = 0

    def send(self, n: int) -> None:
        end = self.at + n
        while self.at < end:
            step = min(FRAME, end - self.at)
            self.ws.send_bytes(_s(self.at, self.at + step))
            self.at += step


class _RecordingSTT:
    """A stand-in for AudioCppSTTPipeline that keeps everything it is fed."""

    def __init__(self) -> None:
        self.feeds: list[bytes] = []

    async def feed(self, data: bytes) -> bool:
        self.feeds.append(bytes(data))
        return False

    def get_current_text(self):
        return ("", "en", 0)

    def take_final(self):
        return None

    async def finalize(self):
        return ("", "en", 0)

    def reset(self) -> None:
        pass

    def dispose(self) -> None:
        pass

    @property
    def fed(self) -> bytes:
        return b"".join(self.feeds)


def _start(request_id: str, mic_sample: int) -> dict:
    return {"type": "playback", "request_id": request_id, "event": "start",
            "mic_sample": mic_sample}


def _stop(request_id: str, mic_sample: int, reply_sample: int = 4800) -> dict:
    return {"type": "playback", "request_id": request_id, "event": "stop",
            "mic_sample": mic_sample, "reply_sample": reply_sample}


def _barge(onset: int, mic_sample: int) -> wire.BargeEvent:
    return wire.BargeEvent(onset_mic_sample=onset, mic_sample=mic_sample)


def _chunks(n: int = 2) -> list[np.ndarray]:
    return [np.full(2400, 0.1 * (i + 1), dtype=np.float32) for i in range(n)]


def _of_type(frames: list, kind: str) -> list:
    return [f for f in frames if isinstance(f, dict) and f.get("type") == kind]


# --- fixtures -------------------------------------------------------------------

@pytest.fixture
def tyger():
    fake = FakeTygerSidecar(progress=True)
    fake.withhold_progress = True            # the tests say how far the sidecar got
    fake.start()
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
    def setenv(**values) -> None:
        for key, value in values.items():
            monkeypatch.setenv(key, str(value))
        cfg._settings = None
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "60")
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
    pipelines: list[_RecordingSTT] = []

    def factory(config=None, **kwargs):
        pipelines.append(_RecordingSTT())
        return pipelines[-1]

    monkeypatch.setattr(audiocpp_stt, "AudioCppSTTPipeline", factory)
    return pipelines


@pytest.fixture
def make_client(tyger, executor, made, env):
    def build(engine):
        # Not entered as a context manager: that would run the real lifespan.
        return contextlib.nullcontext(TestClient(make_app(
            sidecar=StandInSidecar(tyger.url), executor=executor, tts_engine=engine)))
    return build


def _opted_in(ws, tyger, dispatched) -> None:
    """Opt in, and wait until the server has read the sidecar's opening progress line."""
    opt_in(ws)
    assert tyger.wait_for(lambda: len(tyger.sessions) == 1)
    assert tyger.wait_for(
        lambda: any(isinstance(e, wire.ScoredEvent) for e in dispatched), timeout=5)


def _events(ws, tyger, dispatched: list, *events) -> list:
    """Have the sidecar send events; every frame the server sent once it handled them."""
    want = len(dispatched) + len(events)
    tyger.send(0, *events)
    assert tyger.wait_for(lambda: len(dispatched) >= want, timeout=5), dispatched
    return barrier(ws)


# --- (1) the reply a barge after the stop picks ---------------------------------

def test_a_barge_after_the_stop_ends_a_still_open_reply_interrupted(
        tyger, make_client, dispatched):
    engine = GatedTTSEngine(_chunks(5), open_chunks=2)
    with make_client(engine) as client, authkit.ws_connect(client, "/ws/stream") as ws:
        _opted_in(ws, tyger, dispatched)
        ws.send_json({"type": "tts", "request_id": "r1", "text": "Hello there."})
        first = [frame(ws.receive()) for _ in range(3)]
        assert first[0]["type"] == "audio_start"
        assert engine.held.wait(5)
        ws.send_json(_start("r1", 0))
        ws.send_json(_stop("r1", 4000))      # the client stopped before the barge
        assert errors(barrier(ws)) == []

        frames = _events(ws, tyger, dispatched, _barge(3000, 4000 + TAIL))
        assert frames == [
            {"type": "barge", "request_id": "r1", "onset_mic_sample": 3000,
             "mic_sample": 4000 + TAIL},
            {"type": "audio_end", "request_id": "r1", "reason": "interrupted"},
        ]
        engine.release.set()
        later = barrier(ws)
        assert [f for f in later if isinstance(f, tuple)] == []
        assert _of_type(later, "audio_end") == []


def test_a_barge_after_the_stop_of_a_finished_reply_sends_the_frame_alone(
        tyger, make_client, dispatched):
    with make_client(FakeTTSEngine(_chunks())) as client, \
            authkit.ws_connect(client, "/ws/stream") as ws:
        _opted_in(ws, tyger, dispatched)
        speak(ws, "r1")                      # audio_end(complete) went out
        ws.send_json(_start("r1", 1000))
        ws.send_json(_stop("r1", 4200))
        assert errors(barrier(ws)) == []
        frames = _events(ws, tyger, dispatched, _barge(3500, 4300))
        assert frames == [{"type": "barge", "request_id": "r1", "onset_mic_sample": 3500,
                           "mic_sample": 4300}]


def test_a_barge_past_the_stop_plus_the_tail_after_the_hold_ended_does_not_pick_it(
        tyger, make_client, dispatched):
    with make_client(FakeTTSEngine(_chunks())) as client, \
            authkit.ws_connect(client, "/ws/stream") as ws:
        _opted_in(ws, tyger, dispatched)
        speak(ws, "r1")
        ws.send_json(_start("r1", 1000))
        ws.send_json(_stop("r1", 4200))
        assert errors(barrier(ws)) == []
        # The sidecar has scored past the stop plus the tail: the hold for r1 is over.
        _events(ws, tyger, dispatched, wire.ScoredEvent(mic_sample=4200 + TAIL))
        assert _events(ws, tyger, dispatched, _barge(4000, 4200 + TAIL + 1)) == []


# --- (2) the hold waits for the sidecar's progress -------------------------------

def _stopped_reply(ws, tyger, dispatched, stt) -> _Mic:
    """r1 sent at mic 0, started at 1000 and stopped at 6000, with mic up to 10240."""
    _opted_in(ws, tyger, dispatched)
    mic = _Mic(ws)
    speak(ws, "r1")
    mic.send(10240)
    ws.send_json(_start("r1", 1000))
    ws.send_json(_stop("r1", 6000))
    assert errors(barrier(ws)) == []
    return mic


def test_in_play_audio_stays_held_until_progress_passes_the_stop_then_is_dropped(
        tyger, make_client, made, dispatched):
    with make_client(FakeTTSEngine(_chunks())) as client, \
            authkit.ws_connect(client, "/ws/stream") as ws:
        mic = _stopped_reply(ws, tyger, dispatched, made)
        stt = made[0]
        # Nothing scored past the stop yet: a barge may still come for it.
        assert stt.fed == _s(0, 1000)
        _events(ws, tyger, dispatched, wire.ScoredEvent(mic_sample=5120))
        assert stt.fed == _s(0, 1000)
        _events(ws, tyger, dispatched, wire.ScoredEvent(mic_sample=6000 + TAIL - 1))
        assert stt.fed == _s(0, 1000)
        # Every hop through the stop plus the tail scored, with no barge: dropped.
        _events(ws, tyger, dispatched, wire.ScoredEvent(mic_sample=10240))
        assert stt.fed == _s(0, 1000) + _s(6000 + TAIL, 10240)
        mic.send(1024)
        barrier(ws)
        assert stt.fed == _s(0, 1000) + _s(6000 + TAIL, 11264)


def test_a_barge_after_the_stop_feeds_the_held_audio_from_its_onset_in_order(
        tyger, make_client, made, dispatched):
    with make_client(FakeTTSEngine(_chunks())) as client, \
            authkit.ws_connect(client, "/ws/stream") as ws:
        mic = _stopped_reply(ws, tyger, dispatched, made)
        stt = made[0]
        assert stt.fed == _s(0, 1000)
        frames = _events(ws, tyger, dispatched, _barge(5000, 7680),
                         wire.ScoredEvent(mic_sample=10240))
        assert [f["type"] for f in _of_type(frames, "barge")] == ["barge"]
        assert stt.fed == _s(0, 1000) + _s(5000, 10240)
        mic.send(1024)
        barrier(ws)
        assert stt.fed == _s(0, 1000) + _s(5000, 11264)


def test_a_slow_sidecar_holds_the_audio_only_for_the_start_wait_past_the_stop(
        tyger, make_client, made, dispatched, env):
    env(VM_TYGER_START_WAIT_MS=700)          # 11200 mic samples
    bound = 6000 + TAIL + 700 * 16
    with make_client(FakeTTSEngine(_chunks())) as client, \
            authkit.ws_connect(client, "/ws/stream") as ws:
        mic = _stopped_reply(ws, tyger, dispatched, made)
        stt = made[0]
        mic.send(bound - FRAME - mic.at)     # no progress line comes at all
        barrier(ws)
        assert stt.fed == _s(0, 1000)
        mic.send(2 * FRAME)
        barrier(ws)
        assert stt.fed == _s(0, 1000) + _s(6000 + TAIL, mic.at)


# --- the hold and the reply choice on their own ---------------------------------

def _audio_start(request_id: str, rate: int = 16000) -> str:
    return json.dumps({"type": "audio_start", "request_id": request_id, "sample_rate": rate})


def _feed_with(*replies: tuple[str, int, int | None]) -> te.HopFeed:
    """A HopFeed with each (request_id, start, stop) reply sent, 1 s long at 16 kHz."""
    feed = te.HopFeed()
    for request_id, start, stop in replies:
        feed.on_tts_text(request_id, _audio_start(request_id))
        feed.on_tts_pcm(request_id, b"\x01\x00" * 16000)
        assert feed.playback(_start(request_id, start)) is None
        if stop is not None:
            assert feed.playback(_stop(request_id, stop, stop - start)) is None
    return feed


def test_a_stopped_reply_is_picked_through_its_stop_plus_the_tail_and_no_further():
    feed = _feed_with(("r1", 1000, 4000))
    assert feed.played_reply(3999, 3000, TAIL).request_id == "r1"
    assert feed.played_reply(4000 + TAIL, 3000, TAIL).request_id == "r1"
    assert feed.played_reply(4000 + TAIL + 1, 3000, TAIL) is None
    assert feed.played_reply(999, 500, TAIL) is None


def test_a_barge_whose_speech_began_after_the_stop_does_not_pick_the_reply():
    feed = _feed_with(("r1", 1000, 4000))
    assert feed.played_reply(4100, 4000, TAIL).request_id == "r1"
    assert feed.played_reply(4100, 4001, TAIL) is None


def test_a_stopped_reply_whose_span_holds_the_barge_wins_over_a_later_playing_one():
    feed = _feed_with(("r1", 1000, 4000), ("r2", 8000, None))
    assert feed.played_reply(4100, 3000, TAIL).request_id == "r1"
    # Outside every span: the latest-started reply with no stop, as before.
    assert feed.played_reply(7000, 6000, TAIL).request_id == "r2"


def test_a_hold_with_no_progress_reported_drops_at_the_stop_as_before():
    feed = _feed_with(("r1", 0, 4000))
    hold = te.MicHold(feed, tail_samples=TAIL, start_wait_samples=32000)
    data = _s(0, 8192)
    hold.push(0, data, feed.feed_mic(data))
    assert hold.take() == [_s(4000 + TAIL, 8192)]


def test_a_hold_with_progress_waits_for_it_past_the_stop_plus_the_tail():
    feed = _feed_with(("r1", 0, 4000))
    hold = te.MicHold(feed, tail_samples=TAIL, start_wait_samples=32000)
    hold.on_scored(0)
    data = _s(0, 8192)
    hold.push(0, data, feed.feed_mic(data))
    assert hold.take() == []
    hold.on_scored(4000 + TAIL - 1)
    assert hold.take() == []
    hold.on_scored(4000 + TAIL)
    assert hold.take() == [_s(4000 + TAIL, 8192)]


# --- (3) a barge from a hop sent in speaking mode past the stop plus the tail ----

def _speaking_hops(tyger) -> list[tuple[str, int]]:
    """(mode, mic_sample) of every hop the sidecar got, once they are all in."""
    return [(h.mode, h.mic_sample) for h in tyger.sessions[0].hops]


def _stopped_early(ws, tyger, dispatched) -> _Mic:
    """r1 started at 3000 and stopped at 6000 before the mic got there; mic to 10240.

    Both hops overlap the played span [3000, 6000), so both go out in speaking mode,
    and every frame of the second one, [5120, 10240), may give a barge up to 10240.
    """
    _opted_in(ws, tyger, dispatched)
    mic = _Mic(ws)
    speak(ws, "r1")
    ws.send_json(_start("r1", 3000))
    mic.send(4096)
    ws.send_json(_stop("r1", 6000, reply_sample=3000 * TTS_RATE // 16000))
    mic.send(10240 - mic.at)
    assert errors(barrier(ws)) == []
    assert tyger.wait_for(lambda: len(tyger.sessions[0].hops) >= 2, timeout=5)
    assert _speaking_hops(tyger) == [(wire.MODE_SPEAKING, 5120), (wire.MODE_SPEAKING, 10240)]
    return mic


def test_a_barge_late_in_the_speaking_hop_that_holds_the_stop_is_acted_on(
        tyger, make_client, made, dispatched):
    with make_client(FakeTTSEngine(_chunks(5))) as client, \
            authkit.ws_connect(client, "/ws/stream") as ws:
        mic = _stopped_early(ws, tyger, dispatched)
        stt = made[0]
        assert stt.fed == _s(0, 3000)
        # Decided 8960 - 6000 samples (360 ms) past the stop, before the hop's scored line.
        frames = _events(ws, tyger, dispatched, _barge(5000, 8960),
                         wire.ScoredEvent(mic_sample=10240))
        assert _of_type(frames, "barge") == [
            {"type": "barge", "request_id": "r1", "onset_mic_sample": 5000,
             "mic_sample": 8960}]
        assert stt.fed == _s(0, 3000) + _s(5000, 10240)
        mic.send(1024)
        barrier(ws)
        assert stt.fed == _s(0, 3000) + _s(5000, 11264)


def test_a_late_stop_keeps_the_hold_and_takes_a_barge_from_hops_sent_speaking(
        tyger, make_client, made, dispatched):
    # r1 is 1 s long at 16 kHz and starts at 0: hops to 15360 go out with the whole
    # reply in them, all speaking. The stop at 3000 comes only after that.
    with make_client(FakeTTSEngine(_chunks(10))) as client, \
            authkit.ws_connect(client, "/ws/stream") as ws:
        _opted_in(ws, tyger, dispatched)
        mic = _Mic(ws)
        speak(ws, "r1")
        ws.send_json(_start("r1", 0))
        mic.send(15360)
        assert tyger.wait_for(lambda: len(tyger.sessions[0].hops) >= 3, timeout=5)
        ws.send_json(_stop("r1", 3000, reply_sample=3000 * TTS_RATE // 16000))
        assert errors(barrier(ws)) == []
        assert [m for m, _ in _speaking_hops(tyger)] == [wire.MODE_SPEAKING] * 3
        stt = made[0]
        # Scored past the stop plus the tail, but not past the hops sent speaking.
        _events(ws, tyger, dispatched, wire.ScoredEvent(mic_sample=5120),
                wire.ScoredEvent(mic_sample=10240))
        assert stt.fed == b""
        frames = _events(ws, tyger, dispatched, _barge(2500, 12000),
                         wire.ScoredEvent(mic_sample=15360))
        assert [f["request_id"] for f in _of_type(frames, "barge")] == ["r1"]
        assert stt.fed == _s(2500, 15360)
        mic.send(1024)
        barrier(ws)
        assert stt.fed == _s(2500, 16384)


def test_a_late_stop_with_no_barge_drops_the_echo_once_the_speaking_hops_are_scored(
        tyger, make_client, made, dispatched):
    with make_client(FakeTTSEngine(_chunks(10))) as client, \
            authkit.ws_connect(client, "/ws/stream") as ws:
        _opted_in(ws, tyger, dispatched)
        mic = _Mic(ws)
        speak(ws, "r1")
        ws.send_json(_start("r1", 0))
        mic.send(15360)
        assert tyger.wait_for(lambda: len(tyger.sessions[0].hops) >= 3, timeout=5)
        ws.send_json(_stop("r1", 3000, reply_sample=3000 * TTS_RATE // 16000))
        assert errors(barrier(ws)) == []
        stt = made[0]
        _events(ws, tyger, dispatched, wire.ScoredEvent(mic_sample=10240))
        assert stt.fed == b""
        _events(ws, tyger, dispatched, wire.ScoredEvent(mic_sample=15360))
        assert stt.fed == _s(3000 + TAIL, 15360)
        mic.send(1024)
        barrier(ws)
        assert stt.fed == _s(3000 + TAIL, 16384)


def test_a_barge_whose_speech_began_after_the_stop_is_ignored_in_a_speaking_hop(
        tyger, make_client, made, dispatched):
    with make_client(FakeTTSEngine(_chunks(5))) as client, \
            authkit.ws_connect(client, "/ws/stream") as ws:
        _stopped_early(ws, tyger, dispatched)
        stt = made[0]
        frames = _events(ws, tyger, dispatched, _barge(6001, 8960),
                         wire.ScoredEvent(mic_sample=10240))
        assert _of_type(frames, "barge") == []
        assert stt.fed == _s(0, 3000) + _s(6000 + TAIL, 10240)


def test_a_barge_after_the_hold_was_dropped_is_ignored(
        tyger, make_client, made, dispatched):
    with make_client(FakeTTSEngine(_chunks(5))) as client, \
            authkit.ws_connect(client, "/ws/stream") as ws:
        _stopped_early(ws, tyger, dispatched)
        stt = made[0]
        _events(ws, tyger, dispatched, wire.ScoredEvent(mic_sample=10240))
        assert stt.fed == _s(0, 3000) + _s(6000 + TAIL, 10240)
        assert _events(ws, tyger, dispatched, _barge(5000, 8960)) == []
        assert stt.fed == _s(0, 3000) + _s(6000 + TAIL, 10240)


def test_a_barge_after_the_slow_sidecar_bound_dropped_the_hold_is_ignored(
        tyger, make_client, made, dispatched, env):
    env(VM_TYGER_START_WAIT_MS=200)          # 3200 mic samples past the stop plus the tail
    with make_client(FakeTTSEngine(_chunks(5))) as client, \
            authkit.ws_connect(client, "/ws/stream") as ws:
        mic = _stopped_early(ws, tyger, dispatched)
        stt = made[0]
        mic.send(6000 + TAIL + 3200 + FRAME - mic.at)
        barrier(ws)
        assert stt.fed == _s(0, 3000) + _s(6000 + TAIL, mic.at)
        assert _events(ws, tyger, dispatched, _barge(5000, 8960)) == []
        assert stt.fed == _s(0, 3000) + _s(6000 + TAIL, mic.at)


def test_a_second_barge_after_the_first_released_the_hold_is_ignored(
        tyger, make_client, made, dispatched):
    with make_client(FakeTTSEngine(_chunks(5))) as client, \
            authkit.ws_connect(client, "/ws/stream") as ws:
        _stopped_early(ws, tyger, dispatched)
        frames = _events(ws, tyger, dispatched, _barge(5000, 8000))
        assert len(_of_type(frames, "barge")) == 1
        assert _events(ws, tyger, dispatched, _barge(5500, 8960)) == []


def _cut_feed(start: int, stop: int, mic: int) -> te.HopFeed:
    """r1, 1 s long, started at ``start``; hops cut to ``mic``, then stopped at ``stop``."""
    feed = te.HopFeed()
    feed.on_tts_text("r1", _audio_start("r1"))
    feed.on_tts_pcm("r1", b"\x01\x00" * 16000)
    assert feed.playback(_start("r1", start)) is None
    feed.feed_mic(b"\x00\x00" * mic)
    feed.due_hops()
    assert feed.playback(_stop("r1", stop, stop - start)) is None
    return feed


def test_a_stopped_reply_takes_a_barge_through_its_last_hop_sent_speaking():
    feed = _cut_feed(0, 3000, 15360)        # three hops went out speaking
    assert feed.played_reply(15360, 2500, TAIL).request_id == "r1"
    assert feed.played_reply(15361, 2500, TAIL) is None
    assert feed.played_reply(12000, 3001, TAIL) is None


def test_a_hold_state_decides_a_stopped_reply_over_the_span():
    feed = _cut_feed(0, 3000, 15360)
    assert feed.played_reply(20000, 2500, TAIL, lambda r: True).request_id == "r1"
    assert feed.played_reply(4000, 2500, TAIL, lambda r: False) is None
    assert feed.played_reply(20000, 3001, TAIL, lambda r: True) is None


def test_a_hold_with_progress_waits_for_the_last_hop_sent_speaking():
    feed = _cut_feed(0, 3000, 15360)
    hold = te.MicHold(feed, tail_samples=TAIL, start_wait_samples=32000)
    hold.on_scored(0)
    data = _s(0, 15360)
    hold.push(0, data, data)
    reply = feed.reply("r1")
    assert hold.on_scored(10240) is False
    assert hold.take() == [] and hold.hold_state(reply) is True
    assert hold.on_scored(15360) is True
    assert hold.take() == [_s(3000 + TAIL, 15360)]
    assert hold.hold_state(reply) is False


# --- the sidecar's progress lines -----------------------------------------------

def test_scored_event_lines_round_trip_and_bad_ones_are_rejected():
    event = wire.ScoredEvent(mic_sample=10240)
    line = wire.format_event(event)
    assert json.loads(line) == {"event": "scored", "mic_sample": 10240}
    assert wire.parse_event_line(line) == event
    for bad in ('{"event": "scored"}', '{"event": "scored", "mic_sample": -1}',
                '{"event": "scored", "mic_sample": true}'):
        with pytest.raises(ValueError):
            wire.parse_event_line(bad)


def _sidecar_app():
    from voicemanager.pipeline.tyger import scorer, server
    # Session "b": one hop of quiet, then a reply plays and the yield head fires.
    frames = ([[0.0, 0.0, -80.0]] * 4 + [[0.0, 0.9, -30.0]] * 16)
    return server.create_app(turn_scorer=scorer.FakeScorer({"b": frames}))


def _hop(mode: str, mic_sample: int) -> bytes:
    silence = bytes(wire.CHANNEL_BYTES)
    return wire.pack_hop(mode, mic_sample, silence, silence)


def test_the_sidecar_sends_progress_lines_only_when_asked():
    with StarletteClient(_sidecar_app()) as c:
        with c.websocket_connect(f"/tyger?session=b&{wire.PROGRESS_PARAM}=1") as ws:
            ws.send_bytes(b"not a hop")      # answered after the opening line
            assert wire.parse_events(ws.receive_text()) == [wire.ScoredEvent(mic_sample=0)]
            assert [type(e) for e in wire.parse_events(ws.receive_text())] == [wire.ErrorEvent]
            ws.send_bytes(_hop(wire.MODE_LISTENING, 5120))
            assert wire.parse_events(ws.receive_text()) == [wire.ScoredEvent(5120)]
            events = []
            for k in range(2, 6):
                ws.send_bytes(_hop(wire.MODE_SPEAKING, k * 5120))
                events.append(wire.parse_events(ws.receive_text()))
            # Each hop's message ends with its progress line, after its events.
            assert [e[-1] for e in events] == [wire.ScoredEvent(k * 5120) for k in range(2, 6)]
            assert any(isinstance(e, wire.BargeEvent) for batch in events for e in batch[:-1])
        with c.websocket_connect("/tyger?session=quiet") as ws:
            ws.send_bytes(_hop(wire.MODE_LISTENING, 5120))
            ws.send_bytes(b"not a hop")
            # Not asked: the quiet hop sends nothing, so the error line comes first.
            assert [type(e) for e in wire.parse_events(ws.receive_text())] == [wire.ErrorEvent]
