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


def test_an_idle_gate_lead_in_stays_armed_while_old_quiet_audio_is_trimmed(
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
        assert _is_lead_in(stt.feeds[0])
        # The pre-roll kept its cap of the newest audio, all of it from after the onset.
        assert len(stt.feeds[1]) == BARGE_PREROLL
        rest = stt.fed[600 * MS:]
        assert len(rest) > BARGE_PREROLL
        assert rest == stream[2 * 10000:2 * 52864][-len(rest):]
        assert sum(_is_lead_in(f) for f in stt.feeds) == 1


def test_two_lead_ins_are_not_the_same_noise():
    a, b = te.barge_lead_in(600), te.barge_lead_in(600)
    assert len(a) == len(b) == 19200 and a != b
    assert te.barge_lead_in(0) == b""


def test_the_default_is_600_ms_and_a_negative_length_is_refused():
    assert cfg.Settings(_env_file=None).tyger_barge_lead_ms == 600
    with pytest.raises(ValidationError, match="tyger_barge_lead_ms"):
        cfg.Settings(_env_file=None, tyger_barge_lead_ms=-1)
