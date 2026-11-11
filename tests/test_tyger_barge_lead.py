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


def test_two_lead_ins_are_not_the_same_noise():
    a, b = te.barge_lead_in(600), te.barge_lead_in(600)
    assert len(a) == len(b) == 19200 and a != b
    assert te.barge_lead_in(0) == b""


def test_the_default_is_600_ms_and_a_negative_length_is_refused():
    assert cfg.Settings(_env_file=None).tyger_barge_lead_ms == 600
    with pytest.raises(ValidationError, match="tyger_barge_lead_ms"):
        cfg.Settings(_env_file=None, tyger_barge_lead_ms=-1)
