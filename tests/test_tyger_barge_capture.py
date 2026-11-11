"""The debug capture of the mic around a barge (VM_TYGER_BARGE_CAPTURE_DIR), model-free.

With the setting on, a barge the session acts on writes the client's mic stream
around it as a WAV, from the audio the hold kept before the onset to the turn end,
and a JSON note with the positions on the mic stream, the finals and the turn end.
A capture with no turn end is written at the cap and when the connection closes.
With the setting off nothing is written.

The sidecar is the in-process FakeTygerSidecar, the STT the scripted stand-in (one
final per fed frame). No GPU, no model, no live server.
"""
import contextlib
import json
import time
import wave
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
    FINAL, FakeTTSEngine, ScriptedAudioCppSTT, StandInSidecar, barrier, errors, make_app, opt_in,
    speak,
)

#: Samples per mic frame the tests send.
FRAME = 1024
#: The client's whole mic stream: a ramp, so any slice shows where it came from.
_STREAM = ((np.arange(40 * 16000) % 30000) + 1).astype("<i2").tobytes()


def _s(a: int, b: int) -> bytes:
    return _STREAM[2 * a:2 * b]


class _Mic:
    def __init__(self, ws) -> None:
        self.ws = ws
        self.at = 0

    def send(self, n: int) -> None:
        end = self.at + n
        while self.at < end:
            step = min(FRAME, end - self.at)
            self.ws.send_bytes(_s(self.at, self.at + step))
            self.at += step


def _start(request_id: str, mic_sample: int) -> dict:
    return {"type": "playback", "request_id": request_id, "event": "start",
            "mic_sample": mic_sample}


def _stop(request_id: str, mic_sample: int, reply_sample: int = 4800) -> dict:
    return {"type": "playback", "request_id": request_id, "event": "stop",
            "mic_sample": mic_sample, "reply_sample": reply_sample}


def _chunks(n: int = 2) -> list[np.ndarray]:
    return [np.full(2400, 0.1 * (i + 1), dtype=np.float32) for i in range(n)]


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
def client(tyger, executor, env, monkeypatch):
    monkeypatch.setattr(audiocpp_stt, "AudioCppSTTPipeline", ScriptedAudioCppSTT)
    # Not entered as a context manager: that would run the real lifespan.
    return contextlib.nullcontext(TestClient(make_app(
        sidecar=StandInSidecar(tyger.url), executor=executor,
        tts_engine=FakeTTSEngine(_chunks()))))


def _opted_in(ws, tyger) -> dict:
    ready = opt_in(ws)
    assert tyger.wait_for(lambda: len(tyger.sessions) == 1)
    return ready


def _events(ws, tyger, dispatched: list, *events) -> list:
    want = len(dispatched) + len(events)
    tyger.send(0, *events)
    assert tyger.wait_for(lambda: len(dispatched) >= want, timeout=5), dispatched
    return barrier(ws)


def _wait_for_files(folder, count: int, timeout: float = 5.0) -> list:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        notes = sorted(folder.glob("*.json"))
        if len(notes) >= count and all(p.with_suffix(".wav").exists() for p in notes):
            return notes
        time.sleep(0.02)
    raise AssertionError(f"{count} capture(s) expected in {folder}: {list(folder.iterdir())}")


def _read(note_path) -> tuple[dict, bytes]:
    with wave.open(str(note_path.with_suffix(".wav")), "rb") as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
        pcm = wav.readframes(wav.getnframes())
    return json.loads(note_path.read_text()), pcm


def _play_and_barge(ws, tyger, dispatched, mic: _Mic) -> list:
    """A reply that plays from sample 4000, and a barge at 16000 with its onset at
    14000, while the mic is at 20480. The frames the barge made."""
    mic.send(2048)
    speak(ws, "r1")                          # its audio_start went out at sample 2048
    mic.send(18432)                          # to 20480
    ws.send_json(_start("r1", 4000))
    assert errors(barrier(ws)) == []
    return _events(ws, tyger, dispatched,
                   wire.BargeEvent(onset_mic_sample=14000, mic_sample=16000))


def test_a_barge_and_its_turn_end_write_the_mic_around_it(tyger, client, env, dispatched,
                                                          tmp_path):
    env(VM_TYGER_BARGE_CAPTURE_DIR=tmp_path / "captures")
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        ready = _opted_in(ws, tyger)
        mic = _Mic(ws)
        frames = _play_and_barge(ws, tyger, dispatched, mic)
        assert [f["type"] for f in frames if isinstance(f, dict)][:1] == ["barge"]
        assert list((tmp_path / "captures").glob("*")) == []   # still collecting

        ws.send_json(_stop("r1", 16500))
        mic.send(4096)                       # to 24576: the user's words after the barge
        barrier(ws)
        frames = _events(ws, tyger, dispatched,
                         wire.EndpointEvent(by="model", mic_sample=22000))
        turn_ends = [f for f in frames if isinstance(f, dict) and f.get("type") == "turn_end"]
        assert len(turn_ends) == 1

        (note_path,) = _wait_for_files(tmp_path / "captures", 1)
        note, pcm = _read(note_path)
        assert note_path.name == f"barge-{ready['session_id']}-16000.json"
        # The hold kept the reply's in-play audio from its start at 4000.
        assert note["start_mic_sample"] == 4000
        assert note["end_mic_sample"] == 24576
        assert pcm == _s(4000, 24576)
        assert note["request_id"] == "r1"
        assert note["onset_mic_sample"] == 14000
        assert note["barge_mic_sample"] == 16000
        assert note["stt_fed_from_mic_sample"] == 14000
        assert note["playback_stop_mic_sample"] == 16500
        assert note["onset_offset_s"] == pytest.approx(10000 / 16000, abs=0.001)
        assert note["barge_offset_s"] == pytest.approx(12000 / 16000, abs=0.001)
        assert note["reason"] == "turn_end"
        assert note["turn_end"]["by"] == "model"
        assert note["turn_end"]["mic_sample"] == 22000
        assert note["turn_end"]["text"] == turn_ends[0]["text"]
        # The finals since the barge; the turn_end also joins the ones before it.
        assert note["finals"]
        assert turn_ends[0]["text"].endswith(" ".join(note["finals"]))
        assert len(note["finals"]) < turn_ends[0]["text"].count(FINAL)

        # The capture is over: later audio and a later turn end write nothing more.
        mic.send(4096)
        _events(ws, tyger, dispatched, wire.EndpointEvent(by="silence", mic_sample=26000))
    assert len(list((tmp_path / "captures").glob("*.json"))) == 1


def test_a_capture_with_no_turn_end_is_written_at_the_cap(tyger, client, env, dispatched,
                                                          tmp_path, monkeypatch):
    monkeypatch.setattr(te, "CAPTURE_MAX_SAMPLES", 8192)
    env(VM_TYGER_BARGE_CAPTURE_DIR=tmp_path)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        _play_and_barge(ws, tyger, dispatched, mic)
        mic.send(8192)                       # to 28672: 12672 past the barge
        barrier(ws)
        (note_path,) = _wait_for_files(tmp_path, 1)
        note, pcm = _read(note_path)
        assert note["reason"] == "cap"
        assert note["turn_end"] is None
        # The first frame that takes it to 8192 samples past the barge ends it.
        assert note["end_mic_sample"] == 24576
        assert pcm == _s(4000, 24576)


def test_a_capture_still_open_is_written_when_the_connection_closes(
        tyger, client, env, dispatched, tmp_path):
    env(VM_TYGER_BARGE_CAPTURE_DIR=tmp_path)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        _play_and_barge(ws, tyger, dispatched, mic)
        mic.send(1024)
        barrier(ws)
    (note_path,) = _wait_for_files(tmp_path, 1)
    note, pcm = _read(note_path)
    assert note["reason"] == "closed"
    assert pcm == _s(4000, 21504)


def test_an_endpoint_with_no_text_still_ends_the_capture(tyger, client, env, dispatched,
                                                         tmp_path, monkeypatch):
    # The STT gives no final at all: the case the capture exists for.
    class Silent(ScriptedAudioCppSTT):
        async def feed(self, data: bytes) -> bool:
            return False
    monkeypatch.setattr(audiocpp_stt, "AudioCppSTTPipeline", Silent)
    env(VM_TYGER_BARGE_CAPTURE_DIR=tmp_path)
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        _play_and_barge(ws, tyger, dispatched, mic)
        frames = _events(ws, tyger, dispatched,
                         wire.EndpointEvent(by="silence", mic_sample=19000))
        assert [f for f in frames if isinstance(f, dict) and f.get("type") == "turn_end"] == []
        (note_path,) = _wait_for_files(tmp_path, 1)
        note, _ = _read(note_path)
        assert note["reason"] == "turn_end"
        assert note["finals"] == []
        assert note["turn_end"] == {"by": "silence", "mic_sample": 19000,
                                    "offset_s": pytest.approx(15000 / 16000, abs=0.001),
                                    "text": None}


def test_with_the_setting_off_nothing_is_written(tyger, client, dispatched, tmp_path,
                                                 monkeypatch):
    writes = []
    monkeypatch.setattr(te, "_write_capture", lambda *args: writes.append(args))
    with client as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        mic = _Mic(ws)
        frames = _play_and_barge(ws, tyger, dispatched, mic)
        assert [f["type"] for f in frames if isinstance(f, dict)][:1] == ["barge"]
        _events(ws, tyger, dispatched, wire.EndpointEvent(by="model", mic_sample=19000))
    assert writes == []
