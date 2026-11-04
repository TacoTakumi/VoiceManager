"""A barge from the turn engine stops the played reply, model-free.

When the sidecar reports a barge while a reply plays at the client (its playback
start came and no stop yet), the server sends a barge frame for that request, then
cancels it: no more PCM for it goes out, and it ends with audio_end reason
'interrupted' right after the barge frame. A barge with no played reply is ignored.
A reply whose audio_end already went out gets the barge frame alone. A plain
tts_cancel still ends with reason 'cancelled'.

The sidecar is the in-process FakeTygerSidecar, the TTS engine a gated fake that
keeps the reply in flight until the test lets it go.
"""
import asyncio
import contextlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)
from fastapi.testclient import TestClient

from voicemanager.pipeline import audiocpp_stt
from voicemanager.pipeline.tts_session import TTSSession
from voicemanager.pipeline.tyger import wire
from voicemanager.transport import turn_engine as te

from tyger_fake_sidecar import FakeTygerSidecar
from tyger_ws_kit import (
    FakeTTSEngine, GatedTTSEngine, ScriptedAudioCppSTT, StandInSidecar, barrier, errors,
    frame, is_type, make_app, opt_in, recv_until, speak,
)


def _chunks(n: int = 5) -> list[np.ndarray]:
    return [np.full(2400, 0.1 * (i + 1), dtype=np.float32) for i in range(n)]


def _start(request_id: str, mic_sample: int) -> dict:
    return {"type": "playback", "request_id": request_id, "event": "start",
            "mic_sample": mic_sample}


def _stop(request_id: str, mic_sample: int, reply_sample: int) -> dict:
    return {"type": "playback", "request_id": request_id, "event": "stop",
            "mic_sample": mic_sample, "reply_sample": reply_sample}


def _barge(onset: int, mic_sample: int) -> wire.BargeEvent:
    return wire.BargeEvent(onset_mic_sample=onset, mic_sample=mic_sample)


def _of_type(frames: list, kind: str) -> list:
    return [f for f in frames if isinstance(f, dict) and f.get("type") == kind]


def _pcm(frames: list) -> list:
    return [f for f in frames if isinstance(f, tuple)]


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
def dispatched(monkeypatch) -> list:
    """Every sidecar event the server has finished handling, in order.

    The reader task handles events beside the receive loop, so a test waits on this
    before its barrier: whatever the handling sent then precedes turn_started.
    """
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
def make_client(tyger, monkeypatch, executor, dispatched):
    monkeypatch.setattr(audiocpp_stt, "AudioCppSTTPipeline", ScriptedAudioCppSTT)

    def build(engine):
        # Not entered as a context manager: that would run the real lifespan.
        return contextlib.nullcontext(TestClient(make_app(
            sidecar=StandInSidecar(tyger.url), executor=executor, tts_engine=engine)))
    return build


def _opted_in(ws, tyger) -> None:
    opt_in(ws)
    assert tyger.wait_for(lambda: len(tyger.sessions) == 1)


def _events(ws, tyger, dispatched: list, *events) -> list:
    """Have the sidecar send events; every frame the server sent once it handled them."""
    want = len(dispatched) + len(events)
    tyger.send(0, *events)
    assert tyger.wait_for(lambda: len(dispatched) >= want, timeout=5), dispatched
    return barrier(ws)


def _start_streaming(ws, engine: GatedTTSEngine, request_id: str = "r1") -> list:
    """Send a tts request and read its audio_start and the PCM before the gate."""
    ws.send_json({"type": "tts", "request_id": request_id, "text": "Hello there."})
    frames = [frame(ws.receive()) for _ in range(1 + engine.open_chunks)]
    assert frames[0]["type"] == "audio_start"
    assert len(_pcm(frames)) == engine.open_chunks
    assert engine.held.wait(5)
    return frames


# --- a barge during a streaming reply -------------------------------------------

def test_a_barge_during_a_played_reply_sends_the_frame_then_ends_it_interrupted(
        tyger, make_client, dispatched):
    engine = GatedTTSEngine(_chunks(), open_chunks=2)
    with make_client(engine) as client:
        with authkit.ws_connect(client, "/ws/stream") as ws:
            _opted_in(ws, tyger)
            _start_streaming(ws, engine)
            ws.send_json(_start("r1", 0))
            assert errors(barrier(ws)) == []

            frames = _events(ws, tyger, dispatched, _barge(1000, 4000))
            assert frames == [
                {"type": "barge", "request_id": "r1", "onset_mic_sample": 1000,
                 "mic_sample": 4000},
                {"type": "audio_end", "request_id": "r1", "reason": "interrupted"},
            ]

            # The rest of the reply never reaches the client, nor a second audio_end.
            engine.release.set()
            later = barrier(ws)
            assert _pcm(later) == []
            assert _of_type(later, "audio_end") == []

            # The reference client answers with a stop carrying what played: accepted.
            ws.send_json(_stop("r1", 3000, 4800))
            assert errors(barrier(ws)) == []

            # The connection goes on: the next reply speaks in full.
            engine.chunks = _chunks(2)
            engine.open_chunks = 99
            _, pcm = speak(ws, "r2")
            assert len(pcm) == 2 * 2 * 2400


def test_a_plain_tts_cancel_still_ends_cancelled(tyger, make_client):
    engine = GatedTTSEngine(_chunks(), open_chunks=1)
    with make_client(engine) as client:
        with authkit.ws_connect(client, "/ws/stream") as ws:
            _opted_in(ws, tyger)
            _start_streaming(ws, engine)
            ws.send_json(_start("r1", 0))
            ws.send_json({"type": "tts_cancel", "request_id": "r1"})
            frames = recv_until(ws, is_type("audio_end"))
            assert frames[-1] == {"type": "audio_end", "request_id": "r1",
                                  "reason": "cancelled"}
            assert _of_type(frames, "barge") == []
            engine.release.set()


# --- barges with nothing to stop ------------------------------------------------

def test_a_barge_with_no_played_reply_is_ignored(tyger, make_client, dispatched):
    engine = GatedTTSEngine(_chunks(), open_chunks=1)
    with make_client(engine) as client:
        with authkit.ws_connect(client, "/ws/stream") as ws:
            _opted_in(ws, tyger)
            assert _events(ws, tyger, dispatched, _barge(0, 100)) == []  # no reply at all
            _start_streaming(ws, engine)          # sent, but no playback start yet
            assert _of_type(_events(ws, tyger, dispatched, _barge(100, 200)), "barge") == []
            ws.send_json(_start("r1", 0))
            ws.send_json(_stop("r1", 2400, 2400))  # started and stopped: not playing
            assert errors(barrier(ws)) == []
            later = _events(ws, tyger, dispatched, _barge(3000, 3200))
            assert _of_type(later, "barge") == [] and _of_type(later, "audio_end") == []
            engine.release.set()
            frames = recv_until(ws, is_type("audio_end"))
            assert frames[-1]["reason"] == "complete"


def test_a_barge_after_the_reply_ended_sends_the_frame_alone(tyger, make_client, dispatched):
    with make_client(FakeTTSEngine(_chunks(2))) as client:
        with authkit.ws_connect(client, "/ws/stream") as ws:
            _opted_in(ws, tyger)
            speak(ws, "r1")                       # audio_end(complete) went out
            ws.send_json(_start("r1", 0))         # still playing at the client
            assert errors(barrier(ws)) == []
            # The barge frame, and no second audio_end.
            assert _events(ws, tyger, dispatched, _barge(500, 1500)) == [
                {"type": "barge", "request_id": "r1", "onset_mic_sample": 500,
                 "mic_sample": 1500}]
            ws.send_json(_stop("r1", 1600, 2400))
            assert errors(barrier(ws)) == []


def test_the_barge_goes_to_the_reply_whose_span_holds_its_mic_sample(tyger, make_client, dispatched):
    # Each reply is 0.2 s at 24 kHz: 3200 samples at 16 kHz once placed.
    with make_client(FakeTTSEngine(_chunks(2))) as client:
        with authkit.ws_connect(client, "/ws/stream") as ws:
            _opted_in(ws, tyger)
            speak(ws, "r1")
            speak(ws, "r2")
            ws.send_json(_start("r1", 1000))      # plays over [1000, 4200)
            ws.send_json(_start("r2", 20000))     # plays over [20000, 23200), no stop
            assert errors(barrier(ws)) == []
            frames = _events(ws, tyger, dispatched, _barge(1500, 2000))
            assert [f["request_id"] for f in _of_type(frames, "barge")] == ["r1"]
            # Outside every span: the latest started reply with no stop.
            frames = _events(ws, tyger, dispatched, _barge(25000, 25500))
            assert [f["request_id"] for f in _of_type(frames, "barge")] == ["r2"]
            assert _of_type(frames, "audio_end") == []


# --- the cancel path the barge reuses -------------------------------------------

class _RecordingSocket:
    def __init__(self) -> None:
        self.sent: list = []

    async def send_text(self, data: str) -> None:
        self.sent.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)


def test_cancel_sends_its_notice_before_the_audio_end_with_the_given_reason():
    async def run():
        sock = _RecordingSocket()
        session = TTSSession(sock, tts_pipeline=None)
        await session.cancel("r1", reason="interrupted", notice='{"type":"barge"}')
        await session.cancel("r1", reason="interrupted", notice='{"type":"again"}')
        session._mark_completed("r2")
        await session.cancel("r2", reason="interrupted", notice='{"type":"late"}')
        await session.cancel("r3")
        return sock.sent

    sent = asyncio.run(run())
    assert sent == [
        '{"type":"barge"}',
        '{"type":"audio_end","request_id":"r1","reason":"interrupted"}',
        '{"type":"again"}',                  # already ended: the notice alone
        '{"type":"late"}',
        '{"type":"audio_end","request_id":"r3","reason":"cancelled"}',
    ]
