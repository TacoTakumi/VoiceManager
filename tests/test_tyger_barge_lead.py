"""The noise lead-in the STT decoder gets in front of a barge's audio, model-free.

A short utterance spoken over a playing reply reaches the decoder with nothing
usable in front of it: the onset pre-roll is from before the reply, and a client
with an echo canceller sends digital silence while the reply plays. The streaming
decoder then gives no text for some short answers. So at a barge the session acts
on, the decoder first gets VM_TYGER_BARGE_LEAD_MS of low noise, and the mic audio
from the barge onset follows it with nothing between. 0 turns the lead-in off.

The STT is a recording stand-in for the audiocpp pipeline, the sidecar the
in-process FakeTygerSidecar. No GPU, no model, no live server.
"""
import contextlib
import json
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from pydantic import ValidationError

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)
from fastapi.testclient import TestClient

import voicemanager.config as cfg
from voicemanager.pipeline import audiocpp_stt
from voicemanager.pipeline.tyger import wire
from voicemanager.transport import turn_engine as te

from test_audiocpp_idle_gate import FakeVadModel
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
    """Sends a mic stream (the client's ramp, if no other is given) in FRAME-sample
    frames and counts what went."""

    def __init__(self, ws, stream: bytes = _STREAM) -> None:
        self.ws = ws
        self.stream = stream
        self.at = 0
        self.frames: list[bytes] = []

    def send(self, n: int) -> None:
        end = self.at + n
        while self.at < end:
            step = min(FRAME, end - self.at)
            data = self.stream[2 * self.at:2 * (self.at + step)]
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


#: Bytes per millisecond of 16 kHz mono 16-bit audio.
MS = 32


def _level_dbfs(pcm: bytes) -> float:
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float64) / 32768.0
    return float(20.0 * np.log10(np.sqrt(np.mean(x ** 2))))


def _play_and_barge(ws, tyger, dispatched, mic: _Mic) -> None:
    """A reply that plays from sample 3000 and a barge with its onset at 10000,
    while the mic is at 18048."""
    mic.send(2048)
    speak(ws, "r1")
    mic.send(16000)                          # to 18048
    ws.send_json(_start("r1", 3000))
    assert errors(barrier(ws)) == []
    frames = _events(ws, tyger, dispatched,
                     wire.BargeEvent(onset_mic_sample=10000, mic_sample=15360))
    assert [f["type"] for f in frames if isinstance(f, dict)] == ["barge"]


# --- no VAD armed: every frame goes straight to the decoder -------------------------

def test_a_barge_feeds_a_noise_lead_in_then_the_audio_from_its_onset(
        tyger, client, made, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=600)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        _play_and_barge(ws, tyger, dispatched, mic)
        stt = made[0]
        before = len(_s(0, 3000))
        assert stt.fed[:before] == _s(0, 3000)           # released by the start, no lead-in
        lead = stt.fed[before:before + 600 * MS]
        assert lead in stt.feeds                         # one chunk of its own
        assert len(lead) == 600 * MS == 19200
        assert -62.5 <= _level_dbfs(lead) <= -59.5
        assert stt.fed[before + 600 * MS:] == _s(10000, 18048)

        mic.send(2048)                       # later audio follows with no second lead-in
        barrier(ws)
        assert stt.fed[before + 600 * MS:] == _s(10000, 20096)


def test_the_lead_in_length_follows_the_setting(tyger, client, made, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=250)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        _play_and_barge(ws, tyger, dispatched, mic)
        stt = made[0]
        before = len(_s(0, 3000))
        assert len(stt.fed) == before + 250 * MS + len(_s(10000, 18048))
        assert stt.fed[before + 250 * MS:] == _s(10000, 18048)


def test_zero_turns_the_lead_in_off(tyger, client, made, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=0)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        _play_and_barge(ws, tyger, dispatched, mic)
        assert made[0].fed == _s(0, 3000) + _s(10000, 18048)


# --- the idle gate armed: the lead-in waits for the speech onset ---------------------
#
# With a VAD the decoder gets nothing outside an utterance: quiet frames go to a
# rolling pre-roll, and the speech onset feeds the pre-roll and then the frame. So a
# barge only arms the lead-in. The onset feeds it, then the audio kept since the barge.

#: The onset pre-roll of these tests, the default: 200 ms.
PREROLL = 200 * MS
#: The most audio the pre-roll keeps while a lead-in is armed: 2 s.
BARGE_PREROLL = 2000 * MS
#: Quiet mic audio: the scripted VAD reads |x| <= 327 as no speech. Random, so any
#: slice shows where it came from.
_QUIET = np.random.default_rng(7).integers(-90, 91, 6 * 16000).astype("<i2")


def _speech_from(sample: int) -> bytes:
    """A mic stream that is quiet up to ``sample`` and a loud ramp from there on."""
    stream = _QUIET.copy()
    stream[sample:] = (np.arange(len(stream) - sample) % 30000) + 1000
    return stream.tobytes()


def _is_lead_in(chunk: bytes, ms: int = 600) -> bool:
    return len(chunk) == ms * MS and -62.5 <= _level_dbfs(chunk) <= -59.5


@pytest.fixture
def gated_client(tyger, executor, made, env):
    """The client of an app whose idle gate is armed by the scripted VAD."""
    env(VM_AUDIOCPP_IDLE_GATE=1, VM_AUDIOCPP_ONSET_PREROLL_MS=200,
        VM_AUDIOCPP_VAD_THRESHOLD=0.5, VM_AUDIOCPP_SILENCE_MS=800)
    app = make_app(sidecar=StandInSidecar(tyger.url), executor=executor,
                   tts_engine=FakeTTSEngine(_chunks()))
    app.state.audiocpp_config = {"vad_model_factory": FakeVadModel}
    return contextlib.nullcontext(TestClient(app))


def test_under_the_idle_gate_the_lead_in_is_fed_at_the_speech_onset(
        tyger, gated_client, made, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=600)
    stream = _speech_from(12000)             # the words start 125 ms after the onset
    with gated_client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws, stream)
        _play_and_barge(ws, tyger, dispatched, mic)
        stt = made[0]
        # Quiet audio is gated, so the lead-in is the first thing the decoder gets.
        lead = stt.feeds[0]
        assert len(lead) == 600 * MS == 19200
        assert -62.5 <= _level_dbfs(lead) <= -59.5
        # Then the audio from the barge onset on: the frames before the VAD's start
        # (one chunk, the pre-roll kept since the barge), then the frame of the start.
        assert stt.feeds[1] == stream[2 * 10000:2 * 11264]
        assert stt.fed == lead + stream[2 * 10000:2 * 18048]
        # Nothing from before the reply played (the mic up to sample 3000), which the
        # pre-roll held when the barge came.
        assert all(stream[2 * a:2 * (a + 32)] not in stt.fed
                   for a in range(0, 3000 - 32, 16))
        assert sum(_is_lead_in(f) for f in stt.feeds) == 1

        mic.send(2048)                       # later audio follows with no second lead-in
        barrier(ws)
        assert stt.fed == lead + stream[2 * 10000:2 * 20096]


def test_zero_gives_no_lead_in_under_the_idle_gate(
        tyger, gated_client, made, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=0)
    stream = _speech_from(12000)
    with gated_client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws, stream)
        _play_and_barge(ws, tyger, dispatched, mic)
        stt = made[0]
        assert not any(_is_lead_in(f) for f in stt.feeds)
        # As with no lead-in setting at all: the onset feeds the usual pre-roll, the
        # last 200 ms the gate kept, and that still reaches back before the reply.
        gated = stream[:2 * 3000] + stream[2 * 10000:2 * 11264]
        assert stt.feeds[0] == gated[-PREROLL:]
        assert stt.fed == gated[-PREROLL:] + stream[2 * 11264:2 * 18048]


def test_the_idle_gate_keeps_the_audio_from_the_onset_past_the_usual_pre_roll(
        tyger, gated_client, made, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=600)
    stream = _speech_from(18000)             # the words start 500 ms after the onset
    with gated_client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws, stream)
        _play_and_barge(ws, tyger, dispatched, mic)
        stt = made[0]
        # The VAD has not heard the start yet (its chunk is not whole): the lead-in
        # is armed, and the decoder has nothing.
        assert stt.feeds == []

        mic.send(2048)                       # to 20096: the start is in
        barrier(ws)
        assert _is_lead_in(stt.feeds[0])
        assert len(stt.feeds[1]) > PREROLL   # more than the usual pre-roll keeps
        assert stt.fed[600 * MS:] == stream[2 * 10000:2 * 20096]
        assert sum(_is_lead_in(f) for f in stt.feeds) == 1


def test_under_the_idle_gate_speech_long_after_the_barge_gets_no_lead_in(
        tyger, gated_client, made, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=600)
    stream = _speech_from(50000)             # the words start 2.5 s after the onset
    with gated_client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws, stream)
        _play_and_barge(ws, tyger, dispatched, mic)
        mic.send(34 * FRAME)                 # to 52864
        barrier(ws)
        stt = made[0]
        # More than 2 s of quiet audio followed the barge, so the armed lead-in was
        # dropped: these words are an utterance of their own, not the barge's.
        assert not any(_is_lead_in(f) for f in stt.feeds)
        # They get the usual pre-roll: the last 200 ms the gate kept before the frame
        # the VAD started on (the one from sample 49792), then that frame and the rest.
        assert stt.feeds[0] == stream[2 * 49792 - PREROLL:2 * 49792]
        assert stt.fed == stream[2 * 49792 - PREROLL:2 * 52864]


# --- no lead-in without a barge, after a listen_stop, or past the time limit ---------
#
# Only a barge the session acts on puts the lead-in in front of its audio. Every other
# way the mic hold lets audio go feeds that audio alone, byte for byte. A lead-in that
# waits for its speech onset under the idle gate is dropped by a listen_stop, and by
# 2 s of quiet audio after the barge. Each test sets a 600 ms lead-in, so one would
# show if the server fed it.

LISTEN_STOP = {"type": "listen_stop"}
LISTEN_START = {"type": "listen_start"}


def _no_lead_in(stt: _RecordingSTT) -> bool:
    """True when no chunk the decoder got is a lead-in."""
    return not any(_is_lead_in(chunk) for chunk in stt.feeds)


def test_a_playback_start_feeds_the_held_audio_with_no_lead_in(tyger, client, made, env):
    env(VM_TYGER_BARGE_LEAD_MS=600)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        mic.send(2048)
        speak(ws, "r1")                      # holding from sample 2048
        mic.send(8192)                       # to 10240
        barrier(ws)
        stt = made[0]
        assert stt.fed == _s(0, 2048)        # held
        ws.send_json(_start("r1", 4000))
        assert errors(barrier(ws)) == []
        assert stt.fed == _s(0, 4000)        # released up to the start, and no more
        assert _no_lead_in(stt)


def test_a_playback_stop_with_no_barge_feeds_no_lead_in(tyger, client, made, env):
    env(VM_TYGER_BARGE_LEAD_MS=600)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        mic.send(2048)
        speak(ws, "r1")
        mic.send(16384)                      # to 18432
        ws.send_json(_start("r1", 4000))
        assert errors(barrier(ws)) == []
        stt = made[0]
        assert stt.fed == _s(0, 4000)
        ws.send_json(_stop("r1", 12000))
        assert errors(barrier(ws)) == []
        # The reply's echo is dropped to the stop plus the tail; the rest goes on.
        assert stt.fed == _s(0, 4000) + _s(12000 + TAIL, 18432)
        mic.send(2048)
        barrier(ws)
        assert stt.fed == _s(0, 4000) + _s(12000 + TAIL, 20480)
        assert _no_lead_in(stt)


def test_the_start_wait_feeds_the_held_audio_with_no_lead_in(tyger, client, made, env):
    env(VM_TYGER_BARGE_LEAD_MS=600, VM_TYGER_START_WAIT_MS=700)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        mic.send(1024)
        speak(ws, "r1")                      # audio_end went out; no start follows
        mic.send(6144)
        barrier(ws)
        stt = made[0]
        assert stt.fed == _s(0, 1024)        # not before the wait
        assert tyger.wait_for(lambda: len(stt.fed) >= len(_s(0, 7168)), timeout=5), \
            len(stt.fed)
        mic.send(1024)
        barrier(ws)
        assert stt.fed == _s(0, 8192)
        assert _no_lead_in(stt)


def test_the_loss_of_the_sidecar_feeds_the_held_audio_with_no_lead_in(
        tyger, client, made, env):
    env(VM_TYGER_BARGE_LEAD_MS=600)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        mic.send(2048)
        speak(ws, "r1")                      # holding from sample 2048
        mic.send(4096)                       # to 6144
        ws.send_json(_start("r1", 3000))     # in play from 3000: held for a barge
        assert errors(barrier(ws)) == []
        stt = made[0]
        assert stt.fed == _s(0, 3000)

        tyger.drop(0)
        # The hold is over: everything it kept goes to the STT, in order.
        assert tyger.wait_for(lambda: len(stt.fed) >= len(_s(0, 6144)), timeout=5), \
            len(stt.fed)
        mic.send(2048)
        frames = barrier(ws)
        assert [f["error_type"] for f in errors(frames)] == ["turn_engine_lost"]
        assert stt.fed == _s(0, 8192)
        assert _no_lead_in(stt)


def test_a_barge_the_session_ignores_feeds_no_lead_in(
        tyger, client, made, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=600)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        mic.send(4096)
        # No reply at all: nothing for the barge to stop.
        assert _events(ws, tyger, dispatched,
                       wire.BargeEvent(onset_mic_sample=1000, mic_sample=3072)) == []
        mic.send(2048)                       # to 6144
        assert errors(barrier(ws)) == []
        stt = made[0]
        assert stt.feeds == mic.frames       # frame for frame, nothing in front

        # A reply that was sent but has no playback start: still no played reply.
        speak(ws, "r1")                      # holding from sample 6144
        mic.send(4096)                       # to 10240
        frames = _events(ws, tyger, dispatched,
                         wire.BargeEvent(onset_mic_sample=7000, mic_sample=9216))
        assert [f for f in frames if isinstance(f, dict) and f["type"] == "barge"] == []
        assert stt.fed == _s(0, 6144)        # the ignored barge did not end the hold
        ws.send_json(_start("r1", 8000))
        assert errors(barrier(ws)) == []
        assert stt.fed == _s(0, 8000)
        assert _no_lead_in(stt)


def test_a_connection_that_never_opted_in_gets_no_lead_in(tyger, client, made, env):
    env(VM_TYGER_BARGE_LEAD_MS=600)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        mic = _Mic(ws)
        mic.send(2048)
        speak(ws, "r1")
        mic.send(8192)
        ws.send_json(_start("r1", 4000))     # dropped unanswered without an opt-in
        mic.send(2048)
        ws.send_json(_stop("r1", 9000))
        mic.send(2048)
        assert errors(barrier(ws)) == []
        stt = made[0]
    assert stt.feeds == mic.frames           # frame for frame, nothing in front
    assert _no_lead_in(stt)
    assert tyger.sessions == []


def test_a_listen_stop_drops_a_lead_in_that_waits_under_the_idle_gate(
        tyger, gated_client, made, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=600)
    stream = _speech_from(25216)             # quiet until 320 ms after the listen_start
    with gated_client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws, stream)
        _play_and_barge(ws, tyger, dispatched, mic)
        mic.send(2048)                       # to 20096, all quiet: no speech onset
        barrier(ws)
        stt = made[0]
        assert stt.feeds == []               # the lead-in is armed, nothing is fed

        ws.send_json(LISTEN_STOP)
        assert {"type": "listen_stopped", "mic_sample": 20096} in barrier(ws)
        ws.send_json(LISTEN_START)
        mic.send(7 * FRAME)                  # to 27264; the words start at 25216
        assert errors(barrier(ws)) == []
        # An utterance of its own: no lead-in, and the usual pre-roll in front of the
        # frame the VAD started on. That is the last 200 ms of quiet audio, all of it
        # from after the listen_start (sample 20096): the 2 s of audio a waiting
        # lead-in would keep is not kept, and nothing from before the stop is fed.
        assert _no_lead_in(stt)
        assert stt.feeds[0] == stream[2 * 25216 - PREROLL:2 * 25216]
        assert stt.fed == stream[2 * 25216 - PREROLL:2 * 27264]


def test_a_listen_stop_after_a_barge_leaves_no_lead_in_behind(
        tyger, client, made, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=600)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        _play_and_barge(ws, tyger, dispatched, mic)      # the barge got its lead-in
        stt = made[0]
        fed = len(stt.feeds)
        assert sum(_is_lead_in(f) for f in stt.feeds) == 1

        ws.send_json(LISTEN_STOP)
        assert {"type": "listen_stopped", "mic_sample": 18048} in barrier(ws)
        ws.send_json(LISTEN_START)
        mic.send(4096)
        assert errors(barrier(ws)) == []
        assert stt.feeds[fed:] == mic.frames[-4:]        # frame for frame, no lead-in


@pytest.mark.parametrize("speech_at, in_time", [(42000, True), (43000, False)])
def test_the_idle_gate_time_limit_is_the_line_between_a_lead_in_and_no_lead_in(
        tyger, gated_client, made, dispatched, env, speech_at, in_time):
    env(VM_TYGER_BARGE_LEAD_MS=600)
    stream = _speech_from(speech_at)
    with gated_client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws, stream)
        _play_and_barge(ws, tyger, dispatched, mic)
        mic.send(26 * FRAME)                 # to 44672
        barrier(ws)
        stt = made[0]
        # The gate kept the audio from the barge onset (sample 10000) up to the frame
        # the VAD started on: the one from 41600 for the first case (1.975 s of it),
        # the one from 42624 for the second (2.039 s, past the limit).
        started = 41600 if in_time else 42624
        assert (len(stream[2 * 10000:2 * started]) <= BARGE_PREROLL) is in_time
        if in_time:
            assert _is_lead_in(stt.feeds[0])
            assert stt.fed[600 * MS:] == stream[2 * 10000:2 * 44672]
            assert sum(_is_lead_in(f) for f in stt.feeds) == 1
        else:
            assert _no_lead_in(stt)
            assert stt.feeds[0] == stream[2 * started - PREROLL:2 * started]
            assert stt.fed == stream[2 * started - PREROLL:2 * 44672]


def test_two_lead_ins_are_not_the_same_noise():
    a, b = te.barge_lead_in(600), te.barge_lead_in(600)
    assert len(a) == len(b) == 19200 and a != b
    assert te.barge_lead_in(0) == b""


def test_the_default_is_600_ms_and_a_negative_length_is_refused():
    assert cfg.Settings(_env_file=None).tyger_barge_lead_ms == 600
    with pytest.raises(ValidationError, match="tyger_barge_lead_ms"):
        cfg.Settings(_env_file=None, tyger_barge_lead_ms=-1)


# --- the lead-in goes where the fed audio breaks --------------------------------------
#
# The mic hold says where the audio it gives the STT does not continue the audio before
# it, and the barge's flush and lead-in go at that place and nowhere else. A barge whose
# audio continues what the STT already has (the user talked through the start of the
# reply) is no boundary at all. Audio the hold lets go from before the played reply is
# fed as it is, in front of the lead-in.

def test_a_barge_whose_audio_continues_what_the_stt_has_gets_no_flush_and_no_lead_in(
        tyger, client, made, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=600, VM_AUDIOCPP_SILENCE_MS=30000)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        stt = made[0]
        mic = _Mic(ws)
        mic.send(2048)
        speak(ws, "r1")
        mic.send(16000)                      # to 18048
        ws.send_json(_start("r1", 3000))
        assert tyger.wait_for(lambda: stt.fed == _s(0, 3000), timeout=5), len(stt.fed)
        # The user began 200 samples before the reply played: the hold drops nothing,
        # so the audio from 3000 on goes on right behind the audio the STT has.
        frames = _events(ws, tyger, dispatched,
                         wire.BargeEvent(onset_mic_sample=2800, mic_sample=15360))
        assert "barge" in [f["type"] for f in frames if isinstance(f, dict)]
        assert ("finalize",) not in stt.events
        assert _no_lead_in(stt)
        assert stt.fed == _s(0, 18048)       # one stretch of speech, byte for byte

        mic.send(2048)
        barrier(ws)
        assert stt.fed == _s(0, 20096)
        assert _no_lead_in(stt)


def test_under_the_idle_gate_a_barge_whose_audio_continues_stays_in_the_open_utterance(
        tyger, gated_client, worded, endpointers, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=600, VM_AUDIOCPP_SILENCE_MS=30000)
    stream = _speech_from(0)                 # the user talks all the way through
    with gated_client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        stt = worded[0]
        mic = _Mic(ws, stream)
        mic.send(2048)
        speak(ws, "r1")
        mic.send(16000)                      # to 18048
        ws.send_json(_start("r1", 3000))
        assert tyger.wait_for(lambda: stt.fed == stream[:2 * 3000], timeout=5), \
            len(stt.fed)
        frames = _events(ws, tyger, dispatched,
                         wire.BargeEvent(onset_mic_sample=2800, mic_sample=15360))
        assert "barge" in [f["type"] for f in frames if isinstance(f, dict)]
        assert ("finalize",) not in stt.events
        assert _no_lead_in(stt)
        assert stt.fed == stream[:2 * 18048]
        # The barge's audio is in the segment of the audio before it, and the VAD that
        # heard the start of the speech is the one that hears the rest.
        assert set(stt.feed_segments) == {0}
        assert len(endpointers) == 1
        assert [f for f in _transcriptions(frames) if f["is_final"]] == []


def test_audio_from_before_the_played_reply_is_fed_before_the_lead_in(
        tyger, client, made, dispatched, env):
    # A long start wait: the reply with no playback start holds for the whole test.
    env(VM_TYGER_BARGE_LEAD_MS=600, VM_AUDIOCPP_SILENCE_MS=30000,
        VM_TYGER_START_WAIT_MS=20000)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        stt = made[0]
        mic = _Mic(ws)
        mic.send(2048)
        speak(ws, "r1")                      # holding from sample 2048
        mic.send(2048)                       # to 4096
        speak(ws, "r2")                      # a later reply, holding from 4096
        mic.send(13952)                      # to 18048
        ws.send_json(_start("r1", 6000))     # r1 plays; no start comes for r2
        # The start frees the audio before 6000, but r2 still holds it from 4096 on.
        assert tyger.wait_for(lambda: stt.fed == _s(0, 4096), timeout=5), len(stt.fed)
        frames = _events(ws, tyger, dispatched,
                         wire.BargeEvent(onset_mic_sample=10000, mic_sample=15360))
        assert "barge" in [f["type"] for f in frames if isinstance(f, dict)]

        # The barge ends both holds: the audio up to r1's start goes on behind the
        # audio the decoder has, then the flush and the lead-in, then the barge's audio.
        events = stt.events
        assert events.count(("finalize",)) == 1
        flush = events.index(("finalize",))
        assert all(e[0] == "feed" for e in events[:flush] + events[flush + 1:])
        assert b"".join(e[1] for e in events[:flush]) == _s(0, 6000)
        assert _is_lead_in(events[flush + 1][1])
        assert b"".join(e[1] for e in events[flush + 2:]) == _s(10000, 18048)
        assert sum(_is_lead_in(f) for f in stt.feeds) == 1


def test_the_loss_of_the_sidecar_after_a_barge_leaves_no_barge_break_behind(
        tyger, client, made, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=600, VM_AUDIOCPP_SILENCE_MS=30000,
        VM_TYGER_START_WAIT_MS=20000)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        stt = made[0]
        mic = _Mic(ws)
        mic.send(2048)
        speak(ws, "r1")
        mic.send(16000)                      # to 18048
        ws.send_json(_start("r1", 3000))
        assert errors(barrier(ws)) == []
        # A barge at the very end of the mic audio in: the hold has none of its audio
        # to give yet, so its break still waits.
        frames = _events(ws, tyger, dispatched,
                         wire.BargeEvent(onset_mic_sample=18048, mic_sample=18048))
        assert "barge" in [f["type"] for f in frames if isinstance(f, dict)]
        speak(ws, "r2")                      # sent after the barge: it holds from 18048
        mic.send(2048)                       # to 20096
        barrier(ws)
        assert stt.fed == _s(0, 3000)

        tyger.drop(0)
        # The hold is over: what it kept goes to the STT as it is, and so does every
        # later frame. The session goes on as one that never opted in.
        want = len(_s(0, 3000)) + len(_s(18048, 20096))
        assert tyger.wait_for(lambda: len(stt.fed) >= want, timeout=5), len(stt.fed)
        mic.send(2048)
        frames = barrier(ws)
        assert [f["error_type"] for f in errors(frames)] == ["turn_engine_lost"]
        assert stt.fed == _s(0, 3000) + _s(18048, 22144)
        assert _no_lead_in(stt)


# The hold on its own: a HopFeed and its MicHold, driven as the session drives them.

GAP, BARGE = te.StreamBreak.GAP, te.StreamBreak.BARGE


class _HoldRig:
    """A HopFeed and its MicHold with no session around them."""

    def __init__(self) -> None:
        self.feed = te.HopFeed()
        self.hold = te.MicHold(self.feed, tail_samples=TAIL, start_wait_samples=32000)

    def mic(self, to: int) -> None:
        """Mic frames up to sample ``to``, pushed into the hold."""
        while self.feed.mic_samples < to:
            at = self.feed.mic_samples
            data = _s(at, min(at + FRAME, to))
            self.hold.push(at, data, self.feed.feed_mic(data))

    def reply(self, request_id: str) -> None:
        """A reply of 1 s sent now: its audio_start, its PCM and its audio_end."""
        for kind in ("audio_start", "audio_end"):
            self.feed.on_tts_text(request_id, json.dumps(
                {"type": kind, "request_id": request_id, "sample_rate": 16000}))
            if kind == "audio_start":
                self.feed.on_tts_pcm(request_id, b"\x01\x00" * 16000)

    def playback(self, frame: dict) -> None:
        assert self.feed.playback(frame) is None

    def barge(self, onset: int, mic_sample: int) -> None:
        """What the session does to the hold for a barge it acts on."""
        feed_from = max(onset, mic_sample - te.BARGE_LOOKBACK_SAMPLES)
        for kept in self.feed.replies():
            if kept.seen_at <= mic_sample and kept.barge_at is None:
                kept.barge_at = feed_from
        self.hold.mark_barge(feed_from)

    def marked(self) -> list:
        """What ``take_marked`` gives now: its breaks, and each stretch of audio
        between them as one piece."""
        out: list = []
        for item in self.hold.take_marked():
            if isinstance(item, bytes) and out and isinstance(out[-1], bytes):
                out[-1] += item
            else:
                out.append(item)
        return out


def _rig_with_a_reply_in_play(start: int = 3000) -> _HoldRig:
    """The mic at 18048, r1 sent at 2048 and in play from ``start``, and the audio
    before the start taken."""
    rig = _HoldRig()
    rig.mic(2048)
    assert rig.marked() == [_s(0, 2048)]
    rig.reply("r1")
    rig.mic(18048)
    rig.playback(_start("r1", start))
    assert rig.marked() == [_s(2048, start)]
    return rig


def test_a_reply_that_plays_to_its_stop_gives_a_gap_break_in_front_of_the_next_audio():
    rig = _rig_with_a_reply_in_play()
    rig.playback(_stop("r1", 12000))
    assert rig.marked() == [GAP, _s(12000 + TAIL, 18048)]
    rig.mic(20096)
    assert rig.marked() == [_s(18048, 20096)]            # no second break


def test_a_barge_gives_a_barge_break_in_front_of_the_audio_from_its_onset():
    rig = _rig_with_a_reply_in_play()
    rig.barge(10000, 15360)
    assert rig.marked() == [BARGE, _s(10000, 18048)]
    rig.mic(20096)
    assert rig.marked() == [_s(18048, 20096)]            # no second break


def test_audio_let_go_in_order_gets_no_break():
    rig = _HoldRig()
    rig.mic(2048)
    assert rig.marked() == [_s(0, 2048)]                 # the first audio: no break
    rig.reply("r1")
    rig.mic(10240)
    assert rig.marked() == []                            # held for the start
    rig.playback(_start("r1", 4000))
    assert rig.marked() == [_s(2048, 4000)]              # nothing was dropped before it


def test_a_barge_with_its_onset_before_the_playback_start_gives_no_break():
    rig = _rig_with_a_reply_in_play()
    rig.barge(2800, 15360)
    assert rig.marked() == [_s(3000, 18048)]             # it continues the audio fed
    # The barge is done with: the next place the audio breaks is a gap, not a barge.
    rig.reply("r2")
    rig.mic(30000)
    rig.playback(_start("r2", 20000))
    assert rig.marked() == [_s(18048, 20000)]
    rig.playback(_stop("r2", 24000))
    assert rig.marked() == [GAP, _s(24000 + TAIL, 30000)]


def test_the_barge_break_comes_after_the_audio_from_before_the_played_reply():
    rig = _HoldRig()
    rig.mic(2048)
    assert rig.marked() == [_s(0, 2048)]
    rig.reply("r1")                          # holding from 2048
    rig.mic(4096)
    rig.reply("r2")                          # holding from 4096, and it gets no start
    rig.mic(18048)
    rig.playback(_start("r1", 6000))
    assert rig.marked() == [_s(2048, 4096)]
    rig.barge(10000, 15360)
    assert rig.marked() == [_s(4096, 6000), BARGE, _s(10000, 18048)]


def test_a_barge_over_a_reply_that_played_from_the_first_mic_sample_gets_its_break():
    rig = _HoldRig()
    rig.reply("r1")                          # sent before any mic audio
    rig.mic(18048)
    rig.playback(_start("r1", 0))
    assert rig.marked() == []
    rig.barge(10000, 15360)
    # The first audio the STT gets at all, and it is the barge's.
    assert rig.marked() == [BARGE, _s(10000, 18048)]


def test_two_barges_with_none_of_their_audio_given_yet_give_one_barge_break():
    rig = _rig_with_a_reply_in_play()
    rig.barge(18048, 18048)                  # none of its audio is in yet
    assert rig.marked() == []
    rig.mic(20096)
    rig.barge(19000, 20096)                  # a second one before any taker ran
    # The break is in front of the first audio of the two, and there is only one.
    assert rig.marked() == [BARGE, _s(18048, 20096)]
    rig.mic(22144)
    assert rig.marked() == [_s(20096, 22144)]


def test_a_listening_boundary_leaves_no_barge_break_for_the_audio_after_it():
    rig = _rig_with_a_reply_in_play()
    rig.barge(18048, 18048)                  # none of its audio is in yet
    assert rig.marked() == []
    assert rig.hold.clear() == 0             # the listening boundary
    rig.feed.listen_boundary()
    rig.feed.listen_start()
    rig.mic(20096)
    # The STT starts over at the boundary, so this audio breaks from nothing.
    assert rig.marked() == [_s(18048, 20096)]

    # A barge after the boundary is marked as usual, also over a reply that plays
    # from the boundary's own sample.
    rig.reply("r2")
    rig.mic(30000)
    rig.playback(_start("r2", 20096))
    rig.barge(26000, 28000)
    assert rig.marked() == [BARGE, _s(26000, 30000)]


def test_take_all_after_a_barge_leaves_no_barge_break_behind():
    rig = _rig_with_a_reply_in_play()
    rig.barge(18048, 18048)                  # none of its audio is in yet
    assert rig.marked() == []
    rig.reply("r2")                          # sent after the barge: it holds from 18048
    rig.mic(20096)
    assert rig.marked() == []
    # The sidecar is lost: everything goes out in order, as plain audio.
    assert rig.hold.take_all() == [_s(18048, 19072), _s(19072, 20096)]
    rig.feed.reply("r2").start_wait_over = True          # nothing holds any more
    rig.mic(22144)
    assert rig.marked() == [_s(20096, 22144)]


def test_take_gives_the_audio_of_take_marked_without_the_breaks():
    def drive(take) -> list:
        """One reply with a barge, one that plays to its stop; every take's result."""
        rig = _HoldRig()
        taken = []

        def step() -> None:
            taken.append(take(rig.hold))

        rig.mic(2048)
        step()
        rig.reply("r1")
        rig.mic(18048)
        rig.playback(_start("r1", 3000))
        step()
        rig.barge(10000, 15360)
        step()
        rig.reply("r2")
        rig.mic(30000)
        rig.playback(_start("r2", 20000))
        step()
        rig.playback(_stop("r2", 24000))
        step()
        return taken

    marked = drive(lambda hold: hold.take_marked())
    plain = drive(lambda hold: hold.take())
    assert [item for step in marked for item in step if not isinstance(item, bytes)] \
        == [BARGE, GAP]
    assert all(isinstance(item, bytes) for step in plain for item in step)
    assert plain == [[item for item in step if isinstance(item, bytes)] for step in marked]


def test_take_moves_the_break_marks_as_take_marked_does():
    rig = _rig_with_a_reply_in_play()
    rig.barge(10000, 15360)
    assert b"".join(rig.hold.take()) == _s(10000, 18048)  # the barge's audio, unmarked
    rig.mic(20096)
    assert rig.marked() == [_s(18048, 20096)]            # so no break is left for later


# --- an utterance still open at the barge --------------------------------------------
#
# The reply can start while an utterance is still open: the mic hold keeps the audio
# under the reply from the VAD, so the VAD never hears the silence that ends the
# utterance, and with no VAD the audio fed before the reply waits for its flush. A
# barge is an utterance boundary for the STT: the open utterance is flushed first, and
# the lead-in and the barge's audio start a new one.

class _WordedSTT(_RecordingSTT):
    """A recording decoder that has a text for every utterance it got audio for, and
    moves to the next segment at a reset, as the real pipeline does."""

    def __init__(self) -> None:
        super().__init__()
        self.segment = 0
        #: The segment each feed went to, in the order of ``feeds``.
        self.feed_segments: list[int] = []
        self._heard = False

    async def feed(self, data: bytes) -> bool:
        self.events.append(("feed", bytes(data)))
        self.feed_segments.append(self.segment)
        self._heard = True
        return True

    def get_current_text(self):
        return (f"partial {self.segment}" if self._heard else "", "en", self.segment)

    async def finalize(self):
        self.events.append(("finalize",))
        return (f"final {self.segment}" if self._heard else "", "en", self.segment)

    def reset(self) -> None:
        self.segment += 1
        self._heard = False


@pytest.fixture
def worded(made, monkeypatch) -> list:
    """The worded STT pipelines the connections made, in place of the plain ones."""
    pipelines: list[_WordedSTT] = []

    def factory(config=None, **kwargs):
        pipelines.append(_WordedSTT())
        return pipelines[-1]

    monkeypatch.setattr(audiocpp_stt, "AudioCppSTTPipeline", factory)
    return pipelines


@pytest.fixture
def endpointers(monkeypatch) -> list:
    """Every acoustic endpointer the connections built, in order."""
    from voicemanager.pipeline import AcousticEndpointer

    built: list = []

    class Counted(AcousticEndpointer):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            built.append(self)

    monkeypatch.setattr("voicemanager.transport.websocket.AcousticEndpointer", Counted)
    return built


#: Where the reply of these tests starts to play, and the onset of the barge over it.
REPLY_AT = 3000
ONSET = 10000


def _speech_then_barge_speech() -> bytes:
    """A mic stream that is loud up to the reply, quiet under it, and loud again from
    the barge onset on. The two loud parts are different ramps."""
    stream = _QUIET.copy()
    stream[:REPLY_AT] = (np.arange(REPLY_AT) % 20000) + 1000
    stream[ONSET:] = (np.arange(len(stream) - ONSET) % 20000) + 9000
    return stream.tobytes()


def _barge_over_open_utterance(ws, tyger, dispatched, mic: _Mic, stt) -> list:
    """Mic audio up to REPLY_AT reaches the STT, a reply plays from there, and a barge
    with its onset at ONSET comes while the mic is at 18048. The frames the server
    sent up to the end of the barge.

    No barrier before the barge: its turn_start resets the decoder on this engine. The
    audio the playback start lets go shows that the start was handled.
    """
    mic.send(2048)
    speak(ws, "r1")
    mic.send(16000)                          # to 18048
    ws.send_json(_start("r1", REPLY_AT))
    assert tyger.wait_for(lambda: stt.fed == mic.stream[:2 * REPLY_AT], timeout=5), \
        len(stt.fed)
    return _events(ws, tyger, dispatched,
                   wire.BargeEvent(onset_mic_sample=ONSET, mic_sample=15360))


def _transcriptions(frames: list) -> list[dict]:
    return [f for f in frames if isinstance(f, dict) and f.get("type") == "transcription"]


def test_an_open_utterance_is_flushed_before_the_lead_in_under_the_idle_gate(
        tyger, gated_client, worded, endpointers, dispatched, env):
    # 30 s: no frame-gap flush of its own. The VAD's silence window is as long, but the
    # VAD hears no silence at all here: the hold keeps the quiet audio under the reply.
    env(VM_TYGER_BARGE_LEAD_MS=600, VM_AUDIOCPP_SILENCE_MS=30000)
    stream = _speech_then_barge_speech()
    with gated_client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        stt = worded[0]
        frames = _barge_over_open_utterance(ws, tyger, dispatched, _Mic(ws, stream), stt)

        # The decoder: the old utterance, its flush, the lead-in, the barge's audio.
        events = stt.events
        assert events.count(("finalize",)) == 1
        flush = events.index(("finalize",))
        assert all(e[0] == "feed" for e in events[:flush] + events[flush + 1:])
        assert b"".join(e[1] for e in events[:flush]) == stream[:2 * REPLY_AT]
        assert _is_lead_in(events[flush + 1][1])
        assert b"".join(e[1] for e in events[flush + 2:]) == stream[2 * ONSET:2 * 18048]
        assert sum(_is_lead_in(f) for f in stt.feeds) == 1
        # The flush reset the decoder: the lead-in and the barge's audio are in the
        # next segment, the old audio is not.
        assert stt.feed_segments == [0] * flush + [1] * (len(events) - flush - 1)
        # The VAD started fresh for the barge's audio, on a new endpointer.
        assert len(endpointers) == 2

        # The client: the old utterance's final, on its own segment, before any frame
        # of the barge's utterance.
        sent = _transcriptions(frames)
        finals = [f for f in sent if f["is_final"]]
        assert [(f["text"], f["segment_index"]) for f in finals] == [("final 0", 0)]
        at = sent.index(finals[0])
        assert all(f["segment_index"] == 0 for f in sent[:at])
        assert sent[at + 1:] and all(f["segment_index"] == 1 for f in sent[at + 1:])


def test_an_open_utterance_is_flushed_before_the_lead_in_with_no_vad(
        tyger, client, made, endpointers, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=600, VM_AUDIOCPP_SILENCE_MS=30000)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        stt = made[0]
        _barge_over_open_utterance(ws, tyger, dispatched, _Mic(ws), stt)
        events = stt.events
        assert events.count(("finalize",)) == 1
        flush = events.index(("finalize",))
        assert all(e[0] == "feed" for e in events[:flush] + events[flush + 1:])
        assert b"".join(e[1] for e in events[:flush]) == _s(0, REPLY_AT)
        assert _is_lead_in(events[flush + 1][1])
        assert b"".join(e[1] for e in events[flush + 2:]) == _s(ONSET, 18048)
        assert len(endpointers) == 1         # with no VAD there is nothing to start fresh


def test_with_no_open_utterance_a_barge_flushes_nothing(
        tyger, gated_client, made, endpointers, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=600)
    stream = _speech_from(12000)             # quiet up to the barge: no utterance yet
    with gated_client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        _play_and_barge(ws, tyger, dispatched, _Mic(ws, stream))
        stt = made[0]
        assert ("finalize",) not in stt.events
        assert _is_lead_in(stt.feeds[0])
        assert stt.fed[600 * MS:] == stream[2 * 10000:2 * 18048]
        assert len(endpointers) == 1


def test_with_the_lead_in_off_an_open_utterance_is_not_flushed_under_the_idle_gate(
        tyger, gated_client, made, endpointers, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=0, VM_AUDIOCPP_SILENCE_MS=30000)
    stream = _speech_then_barge_speech()
    with gated_client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        stt = made[0]
        _barge_over_open_utterance(ws, tyger, dispatched, _Mic(ws, stream), stt)
        # As before: the barge's audio goes on in the utterance the VAD is still in.
        assert ("finalize",) not in stt.events
        assert stt.fed == stream[:2 * REPLY_AT] + stream[2 * ONSET:2 * 18048]
        assert len(endpointers) == 1


def test_with_the_lead_in_off_an_open_utterance_is_not_flushed_with_no_vad(
        tyger, client, made, endpointers, dispatched, env):
    env(VM_TYGER_BARGE_LEAD_MS=0, VM_AUDIOCPP_SILENCE_MS=30000)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        stt = made[0]
        _barge_over_open_utterance(ws, tyger, dispatched, _Mic(ws), stt)
        assert ("finalize",) not in stt.events
        assert stt.fed == _s(0, REPLY_AT) + _s(ONSET, 18048)
        assert len(endpointers) == 1


@pytest.fixture
def playbacks(monkeypatch) -> list:
    """Every playback frame the server has finished handling, in order."""
    done = []
    original = te.TurnEngineSession.handle_playback

    async def recording(self, frame):
        try:
            await original(self, frame)
        finally:
            done.append(frame)
    monkeypatch.setattr(te.TurnEngineSession, "handle_playback", recording)
    return done


def test_a_flushed_open_utterance_the_vad_is_still_in_gets_no_second_flush(
        tyger, gated_client, worded, endpointers, dispatched, playbacks, env):
    # The usual order: the turn end flushes the utterance, the reply starts, and the
    # hold keeps the VAD from the silence. At the barge the VAD is inside speech and
    # the decoder has nothing un-flushed: an empty flush would only skip a segment.
    env(VM_TYGER_BARGE_LEAD_MS=600, VM_AUDIOCPP_SILENCE_MS=30000)
    stream = _speech_then_barge_speech()
    with gated_client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        stt = worded[0]
        mic = _Mic(ws, stream)
        mic.send(2048)                       # speech: the VAD is inside an utterance
        assert tyger.wait_for(lambda: stt.fed == stream[:2 * 2048], timeout=5)
        # The turn end flushes it. No barrier from here to the barge: its turn_start
        # would reset the decoder and move the segment.
        tyger.send(0, wire.EndpointEvent(by="model", mic_sample=2048))
        assert tyger.wait_for(lambda: len(dispatched) == 1, timeout=5), dispatched
        assert stt.events.count(("finalize",)) == 1
        left = stt.segment                   # the segment the turn end left
        assert left == 1
        # The reply plays from the sample its hold began at, so the start lets no
        # audio go: the VAD hears nothing more and the decoder gets nothing.
        speak(ws, "r1")
        mic.send(16000)                      # to 18048
        ws.send_json(_start("r1", 2048))
        assert tyger.wait_for(lambda: len(playbacks) == 1, timeout=5), playbacks
        assert stt.fed == stream[:2 * 2048]
        frames = _events(ws, tyger, dispatched,
                         wire.BargeEvent(onset_mic_sample=ONSET, mic_sample=15360))
        assert "barge" in [f["type"] for f in frames if isinstance(f, dict)]

        events = stt.events
        assert events.count(("finalize",)) == 1          # the turn end's; none at the barge
        flush = events.index(("finalize",))
        assert all(e[0] == "feed" for e in events[:flush] + events[flush + 1:])
        assert b"".join(e[1] for e in events[:flush]) == stream[:2 * 2048]
        assert _is_lead_in(events[flush + 1][1])
        assert b"".join(e[1] for e in events[flush + 2:]) == stream[2 * ONSET:2 * 18048]
        # No segment was skipped: the lead-in and the barge's audio are in the segment
        # the turn end left.
        assert stt.feed_segments == [0] * flush + [left] * (len(events) - flush - 1)
        # The VAD started fresh for the barge's audio, on a new endpointer.
        assert len(endpointers) == 2
