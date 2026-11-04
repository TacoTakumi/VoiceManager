"""An opted-in session whose turn engine sidecar is lost, model-free.

When the sidecar goes away (its process exits, or the session's own link to it
closes) the server keeps serving. Each opted-in session gets one error frame of
error_type turn_engine_lost and goes on as a session that never opted in:

* the mic audio held at the moment of loss goes to the STT in order, before any
  later audio, and later audio goes straight to the STT (no hold, also for a
  reply sent after the loss);
* silero still ends utterances and their finals go out;
* no barge or turn_end frame follows, even for events the sidecar still sends;
* no hop goes to the sidecar any more, and a playback frame is dropped unanswered;
* a second opt-in on the same connection is refused with invalid_request.

The watcher starts the sidecar again, and a new connection's opt-in then gets
turn_engine_ready. The sidecar is the in-process FakeTygerSidecar, or the fake
process from tests/tyger_fake_sidecar.py under the real TygerSidecar and its
watcher. No GPU, no model, no live server.
"""
import concurrent.futures
import contextlib
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)
from fastapi.testclient import TestClient

import voicemanager.app as app_module
import voicemanager.config as cfg
import voicemanager.pipeline.tyger.sidecar as sidecar_module
from voicemanager.pipeline import audiocpp_stt
from voicemanager.pipeline.tyger import wire
from voicemanager.protocol import INVALID_REQUEST, TURN_ENGINE_LOST
from voicemanager.transport import turn_engine as te

from test_tyger_hold import _Mic, _RecordingSTT, _chunks, _s, _start, _stop
from test_tyger_startup import _settings, fake_sidecar  # noqa: F401 (fixture)
from test_tyger_turn_end import _AmplitudeVad, _FlushOnlySTT, _SILENCE_MS, _say_closed
from tyger_fake_sidecar import FakeTygerSidecar
from tyger_ws_kit import (
    OPT_IN, FakeTTSEngine, ListeningSidecar, ScriptedAudioCppSTT, StandInSidecar, barrier,
    errors, is_type, make_app, opt_in, recv_until, speak,
)


def _lost(frames: list) -> list:
    return [f for f in errors(frames) if f.get("error_type") == TURN_ENGINE_LOST]


def _of_type(frames: list, kind: str) -> list:
    return [f for f in frames if is_type(kind)(f)]


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
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "60")
    cfg._settings = None
    yield monkeypatch
    cfg._settings = None


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


def _hold_client(tyger, executor, sidecar=None):
    # Not entered as a context manager: that would run the real lifespan.
    return contextlib.nullcontext(TestClient(make_app(
        sidecar=sidecar or StandInSidecar(tyger.url), executor=executor,
        tts_engine=FakeTTSEngine(_chunks()))))


def _opted_in(ws, tyger) -> None:
    opt_in(ws)
    assert tyger.wait_for(lambda: len(tyger.sessions) == 1)


# --- the link closes: held audio, then no hold -----------------------------------

def test_a_dropped_link_sends_one_lost_frame_and_feeds_the_held_audio_in_order(
        tyger, executor, made, env):
    with _hold_client(tyger, executor) as c, authkit.ws_connect(c, "/ws/stream") as ws:
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
        frames = recv_until(ws, is_type("error"))
        lost = _lost(frames)
        assert len(lost) == 1
        assert "turn engine" in lost[0]["message"].lower()
        # The hold is over: everything it kept goes to the STT, in order.
        assert tyger.wait_for(lambda: stt.fed == _s(0, 6144), timeout=5), len(stt.fed)

        n_before = len(stt.feeds)
        mic.send(2048)                       # straight to the STT, frame for frame
        frames += barrier(ws)
        assert stt.feeds[n_before:] == mic.frames[-2:]

        # A reply sent now holds nothing, and its playback frames go unanswered.
        speak(ws, "r2")
        mic.send(2048)
        ws.send_json(_start("r2", 9000))
        ws.send_json(_stop("r1", 9000))
        mic.send(1024)
        frames += barrier(ws)
        assert stt.fed == _s(0, mic.at)
    assert len(_lost(frames)) == 1
    assert [f for f in errors(frames) if f not in lost] == []
    assert _of_type(frames, "barge") == [] and _of_type(frames, "turn_end") == []


def test_an_opt_in_after_the_loss_is_refused(tyger, executor, made, env):
    with _hold_client(tyger, executor) as c, authkit.ws_connect(c, "/ws/stream") as ws:
        _opted_in(ws, tyger)
        tyger.drop(0)
        assert len(_lost(recv_until(ws, is_type("error")))) == 1
        ws.send_json(OPT_IN)                 # before any audio: refused all the same
        reply = ws.receive_json()
        assert reply["type"] == "error" and reply["error_type"] == INVALID_REQUEST
        assert "lost" in reply["message"]
        mic = _Mic(ws)
        mic.send(2048)
        assert errors(barrier(ws)) == []
        assert made[0].feeds == mic.frames
    assert len(tyger.sessions) == 1          # no second link was opened


# --- the watcher reports the loss: later events are ignored ----------------------

@pytest.fixture
def silence_window(env):
    env.setenv("VM_AUDIOCPP_SILENCE_MS", _SILENCE_MS)
    cfg._settings = None


def test_a_reported_loss_ends_barges_and_turn_ends_and_silero_still_ends_utterances(
        tyger, silence_window, monkeypatch, dispatched):
    made: list[_FlushOnlySTT] = []

    def factory(config=None, **kwargs):
        made.append(_FlushOnlySTT(["one", "two"]))
        return made[-1]

    monkeypatch.setattr(audiocpp_stt, "AudioCppSTTPipeline", factory)
    sidecar = ListeningSidecar(tyger.url)
    app = make_app(sidecar=sidecar)
    app.state.audiocpp_config = {"vad_model_factory": _AmplitudeVad}
    with authkit.ws_connect(TestClient(app), "/ws/stream") as ws:
        _opted_in(ws, tyger)
        assert len(sidecar.listeners) == 1
        sidecar.lose()                       # from this thread, as from a watcher's
        frames = recv_until(ws, is_type("error"))
        assert len(_lost(frames)) == 1
        assert sidecar.listeners == []       # heard once, then let go
        frames += _say_closed(ws)            # silero ends it; its final goes out
        assert [f["text"] for f in frames if is_type("transcription")(f)
                and f["is_final"]] == ["one"]
        # The link is still open here: the events reach the session, which ignores them.
        want = len(dispatched) + 2
        tyger.send(0, wire.BargeEvent(onset_mic_sample=1000, mic_sample=2000),
                   wire.EndpointEvent(by="model", mic_sample=20480))
        assert tyger.wait_for(lambda: len(dispatched) >= want, timeout=5), dispatched
        frames += barrier(ws)
    assert len(_lost(frames)) == 1
    assert _of_type(frames, "barge") == [] and _of_type(frames, "turn_end") == []
    assert tyger.sessions[0].hops == []      # no hop went out after the loss


# --- the real sidecar class: the process dies and comes back ---------------------

def _until(predicate, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition not met in time")
        time.sleep(0.02)


def test_a_killed_sidecar_degrades_the_session_and_a_new_opt_in_works_once_it_is_back(
        fake_sidecar, monkeypatch, env):  # noqa: F811 (fixture)
    settings = _settings()
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(app_module, "load_models", lambda s: {
        "stt_engine_type": "audiocpp", "device": "cpu", "audiocpp_config": {}})
    monkeypatch.setattr(app_module, "load_tts_models", lambda s: {
        "tts_engine": FakeTTSEngine(),
        "tts_executor": concurrent.futures.ThreadPoolExecutor(max_workers=1),
        "tts_engine_type": "openvoice"})
    monkeypatch.setattr(audiocpp_stt, "AudioCppSTTPipeline", ScriptedAudioCppSTT)
    monkeypatch.setattr(sidecar_module, "_WATCH_POLL_S", 0.05)
    monkeypatch.setattr(sidecar_module, "_WATCH_RETRY_S", 0.05)
    with TestClient(app_module.create_app()) as client:
        sidecar = client.app.state.turn_engine_sidecar
        assert sidecar.up and sidecar.generation == 1
        with authkit.ws_connect(client, "/ws/stream") as ws:
            opt_in(ws)
            first = sidecar.process
            first.kill()
            first.wait(timeout=5)
            frames = recv_until(ws, is_type("error"))
            assert len(_lost(frames)) == 1
            # The session goes on: its audio still gives finals.
            ws.send_bytes(b"\x01\x00" * 1024)
            frames += recv_until(ws, lambda f: is_type("transcription")(f) and f["is_final"])
            _until(lambda: sidecar.up and sidecar.generation == 2, timeout=10)
            assert sidecar.process.pid != first.pid
            with authkit.ws_connect(client, "/ws/stream") as ws2:
                opt_in(ws2)                  # turn_engine_ready from the new sidecar
            frames += barrier(ws)
        assert len(_lost(frames)) == 1
