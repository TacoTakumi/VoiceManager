"""Tests for `reason` on audio_end and the TTSSession seam.

`audio_end` gains an additive `reason: str = "complete"`. This project emits two
values: 'complete' (the reply finished normally, INCLUDING the flush-timeout
close) and, in later work, 'cancelled' (a tts_cancel or a supersede-by-new-
request_id stopped it). 'interrupted' is reserved for the future server-side
barge-in project and is never emitted here.

This file locks the 'complete' default across every close path that exists today:
a one-shot `tts` request, a `tts_stream` request closed by done=true, and a
request closed by the VM_TTS_STREAM_FLUSH_TIMEOUT safety path. Being additive,
the field must not change the meaning of any existing frame -- a client
that ignores `reason` must see exactly today's behaviour.

It also holds the TTSSession tests: the session is the single owner of a
connection's in-flight TTS request, and it is *callable* -- every test here drives
it with a fake WebSocket and no receive loop present, which is the property a
future server-side barge-in needs from the STT path.

Model-free: fake engine, fake WebSocket.
"""
import asyncio
import inspect
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import authkit

import numpy as np
from fastapi.testclient import TestClient

from voicemanager.app import create_app
from voicemanager.pipeline.tts import TTSPipeline, TTSStreamPipeline
from voicemanager.pipeline.tts_session import DEAD_ID_CAPACITY, TTSSession
from voicemanager.protocol import AudioEndMessage
from voicemanager.transport import websocket as ws_module


# --- test doubles ----------------------------------------------------------

class _FakeEngine:
    """Yields one 8-sample chunk per synthesis call."""

    def __init__(self):
        self.calls: list[str] = []

    @property
    def sample_rate(self) -> int:
        return 24000

    async def stream_synthesize(self, text: str, **kwargs):
        self.calls.append(text)
        yield np.zeros(8, dtype=np.float32)


class _GatedEngine:
    """Synthesis parks mid-flight until the test opens the gate.

    Lets a test hold a synthesis task observably *live* while it dispatches a
    second request — the only way to check that the session never ends up with
    two of them (the single-flight tripwire).
    """

    def __init__(self):
        self.calls: list[str] = []
        self.started = asyncio.Event()
        self.gate = asyncio.Event()

    @property
    def sample_rate(self) -> int:
        return 24000

    async def stream_synthesize(self, text: str, **kwargs):
        self.calls.append(text)
        self.started.set()
        await self.gate.wait()
        yield np.zeros(8, dtype=np.float32)


class _BlockingEngine:
    """A batch engine whose synthesis genuinely blocks a thread inside run_in_executor.

    This is the harness the ack tests need. An executor thread cannot be interrupted, so the
    only way to observe "the ack arrived BEFORE the synthesis unwound" is to have a
    synthesis that is provably still running when the ack is asserted — today's server
    parks on it for the whole duration (measured 858-1089ms).

    Blocks for `block_s` (>= 2s by default, per the plan's harness note) unless a test
    releases it early. Releasing early only ever *helps* the server finish sooner, so
    doing it after the timing assertions keeps the suite fast without weakening them —
    and `entered` lets a test confirm the thread really is parked inside synthesis at
    the moment it cancels.
    """

    is_batch = True  # routes through run_in_executor, exactly like OpenVoice

    def __init__(self, block_s: float = 2.0):
        self.block_s = block_s
        self.calls: list[str] = []
        self.entered = threading.Event()   # set from inside the executor thread
        self.released = threading.Event()

    @property
    def sample_rate(self) -> int:
        return 24000

    def synthesize(self, text: str, speaker_key: str = "EN-US", speed: float = 1.0):
        self.calls.append(text)
        self.entered.set()
        self.released.wait(timeout=self.block_s)
        return np.zeros(8, dtype=np.float32), 24000

    async def stream_synthesize(self, text: str, **kwargs):
        audio, _ = self.synthesize(text)
        yield audio


class _FakeWebSocket:
    def __init__(self):
        self.frames: list[tuple[str, object]] = []

    async def send_text(self, data: str) -> None:
        self.frames.append(("text", json.loads(data)))

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append(("bytes", data))

    def texts(self) -> list[dict]:
        return [p for kind, p in self.frames if kind == "text"]

    def types(self) -> list[str]:
        """The ordered text-frame type sequence — the client's view of the stream."""
        return [p["type"] for p in self.texts()]

    def binaries(self) -> list[bytes]:
        return [p for kind, p in self.frames if kind == "bytes"]

    def audio_ends(self) -> list[dict]:
        return [p for p in self.texts() if p["type"] == "audio_end"]

    def frames_after_last_audio_end(self) -> list[tuple[str, object]]:
        """Everything the client would see AFTER the stream was declared over."""
        last = max(
            (i for i, (kind, p) in enumerate(self.frames)
             if kind == "text" and p["type"] == "audio_end"),
            default=None,
        )
        return [] if last is None else self.frames[last + 1:]


def _assert_last_audio_start_wins(ws: _FakeWebSocket) -> None:
    """Prove the frame-ordering invariant against a connection's whole recorded stream.

    This is the invariant that lets a binary PCM frame stay bare. Walking the
    recorded frames in order, two things must hold:

      * audio_start / audio_end strictly alternate -- an audio_start NEVER lands while a
        previous request is still open (no `audio_start(rN+1)` before `audio_end(rN)`), so at
        most one request is ever open at a time; and
      * every binary PCM frame falls inside an open request, so "the last audio_start wins"
        names that frame's request unambiguously and "discard binary until audio_end" discards
        nothing real.

    An audio_end with no request open is allowed: it is the bare cancel-ack of an id that
    never started. An audio_end whose id is NOT the open request is likewise a
    cancel-ack for an unrelated id (a client cancelling some rX while r1 is still in
    flight) -- the real client is id-scoped and discards it, so it
    does not close the open request either. Text frames that are neither audio_start nor
    audio_end (a transcription, say) are irrelevant to the invariant and skipped.
    """
    open_id: str | None = None
    for kind, payload in ws.frames:
        if kind == "bytes":
            assert open_id is not None, (
                "a PCM frame arrived with no request open -- no audio_start preceded it, so "
                "'last audio_start wins' cannot attribute it"
            )
            continue
        ptype = payload["type"]
        if ptype == "audio_start":
            assert open_id is None, (
                f"audio_start({payload['request_id']}) landed before audio_end({open_id}): two "
                "requests would be open at once, and a bare PCM frame would become ambiguous"
            )
            open_id = payload["request_id"]
        elif ptype == "audio_end":
            if open_id is not None and payload["request_id"] != open_id:
                # A cancel-ack for an id that is not the open one; the id-scoped client
                # discards it, so it does not close the in-flight request.
                continue
            open_id = None
    assert open_id is None, f"request {open_id} was opened but never closed by an audio_end"


def _stream_pipeline(engine) -> TTSStreamPipeline:
    return TTSStreamPipeline(TTSPipeline(engine=engine, executor=None))


def _session(engine, ws: _FakeWebSocket, executor=None) -> TTSSession:
    """A session wired exactly as the WS handler wires it, minus the receive loop."""
    tts = TTSPipeline(engine=engine, executor=executor)
    return TTSSession(
        websocket=ws,
        tts_pipeline=tts,
        tts_stream_pipeline=TTSStreamPipeline(tts),
    )


async def _await_flag(event: threading.Event, timeout: float = 3.0) -> None:
    """Wait for a threading.Event from async code without blocking the event loop."""
    deadline = time.monotonic() + timeout
    while not event.is_set():
        assert time.monotonic() < deadline, "engine never entered synthesis"
        await asyncio.sleep(0.005)


# --- the field itself is additive -----------------------------------------

def test_audio_end_reason_defaults_to_complete():
    msg = AudioEndMessage(request_id="r1")
    assert msg.reason == "complete"
    assert json.loads(msg.model_dump_json())["reason"] == "complete"


def test_audio_end_reason_is_additive_and_changes_no_existing_field():
    # A client reading only today's fields must see today's frame.
    payload = json.loads(AudioEndMessage(request_id="r1").model_dump_json())
    assert payload["type"] == "audio_end"
    assert payload["request_id"] == "r1"
    # reason is the ONLY new key
    assert set(payload) == {"type", "request_id", "reason"}


# --- every close path that exists today says 'complete' --------------------

def test_a_completed_tts_request_ends_with_reason_complete():
    engine = _FakeEngine()
    pipe = TTSPipeline(engine=engine, executor=None)
    ws = _FakeWebSocket()

    async def run():
        await pipe.synthesize_and_stream(
            websocket=ws, text="Hello there.", request_id="r1",
        )
        ends = ws.audio_ends()
        assert len(ends) == 1
        assert ends[0]["reason"] == "complete"
        assert ends[0]["request_id"] == "r1"

    asyncio.run(run())


def test_a_tts_stream_closed_by_done_ends_with_reason_complete():
    engine = _FakeEngine()
    pipe = _stream_pipeline(engine)
    ws = _FakeWebSocket()

    async def run():
        await pipe.feed_text(
            chunk="Hello there.", done=True,
            websocket=ws, request_id="r1",
        )
        await asyncio.wait_for(pipe._synthesis_task, 2.0)

        ends = ws.audio_ends()
        assert len(ends) == 1
        assert ends[0]["reason"] == "complete"

    asyncio.run(run())


def test_a_request_closed_by_the_flush_timeout_ends_with_reason_complete():
    # The flush-timeout close is a NORMAL completion, not
    # an interruption: the server flushed what it had and finished the reply.
    engine = _FakeEngine()
    pipe = _stream_pipeline(engine)
    ws = _FakeWebSocket()

    async def run():
        await pipe.feed_text(
            chunk="Hello there", done=False, flush=True,
            websocket=ws, request_id="r1",
        )
        await pipe.flush_timeout(ws, "r1")
        await asyncio.wait_for(pipe._synthesis_task, 2.0)

        ends = ws.audio_ends()
        assert len(ends) == 1
        assert ends[0]["reason"] == "complete"

    asyncio.run(run())


def test_an_empty_tts_request_still_ends_with_reason_complete():
    # The degenerate close path: nothing to say, so audio_start/audio_end with no
    # PCM between them. It is still a normal completion.
    engine = _FakeEngine()
    pipe = TTSPipeline(engine=engine, executor=None)
    ws = _FakeWebSocket()

    async def run():
        await pipe.synthesize_and_stream(websocket=ws, text="   ", request_id="r1")
        ends = ws.audio_ends()
        assert len(ends) == 1
        assert ends[0]["reason"] == "complete"

    asyncio.run(run())


def test_no_close_path_emits_interrupted_yet():
    # 'interrupted' is reserved for the future server-side barge-in project. If it
    # ever shows up in this project's frames, a scope line has been crossed.
    engine = _FakeEngine()
    pipe = _stream_pipeline(engine)
    ws = _FakeWebSocket()

    async def run():
        await pipe.feed_text(
            chunk="Hello there.", done=True,
            websocket=ws, request_id="r1",
        )
        await asyncio.wait_for(pipe._synthesis_task, 2.0)
        assert all(e["reason"] != "interrupted" for e in ws.audio_ends())

    asyncio.run(run())


# --- the session is callable with no receive loop present ---

def test_the_session_drives_a_one_shot_request_with_no_receive_loop():
    # A fake websocket and a direct method call are enough to drive TTS.
    # Nothing here goes through transport/websocket.py.
    engine = _FakeEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)
        await session.speak(text="Hello there.", request_id="r1")
        await asyncio.wait_for(session.wait_idle(), 2.0)

        assert engine.calls == ["Hello there."]
        assert ws.types() == ["audio_start", "audio_end"]
        assert ws.audio_ends()[0] == {
            "type": "audio_end", "request_id": "r1", "reason": "complete"}

    asyncio.run(run())


def test_the_session_drives_a_stream_request_with_no_receive_loop():
    engine = _FakeEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)
        await session.feed_stream(chunk="Hello there.", done=True, request_id="r1")
        await asyncio.wait_for(session.wait_idle(), 2.0)

        assert engine.calls == ["Hello there."]
        assert ws.types() == ["audio_start", "audio_end"]
        assert ws.audio_ends()[0]["reason"] == "complete"

    asyncio.run(run())


# --- at most ONE live synthesis task per connection (the single-flight tripwire) ---

def test_a_one_shot_request_supersedes_a_live_stream_request():
    # Today the receive loop holds the one-shot task and TTSStreamPipeline holds the
    # stream task, and neither stops the other: `tts_stream` then `tts` leaves BOTH
    # live. That is exactly the two-in-flight state the single-flight rule forbids.
    engine = _GatedEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)

        await session.feed_stream(chunk="First reply.", done=True, request_id="r1")
        await asyncio.wait_for(engine.started.wait(), 2.0)  # r1 is parked mid-synthesis
        assert len(session.live_tasks) == 1

        await session.speak(text="Second reply.", request_id="r2")
        assert len(session.live_tasks) == 1  # r1's task is gone, not merely ignored

        engine.gate.set()
        await asyncio.wait_for(session.wait_idle(), 2.0)
        assert len(session.live_tasks) == 0

    asyncio.run(run())


def test_a_stream_request_supersedes_a_live_one_shot():
    engine = _GatedEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)

        await session.speak(text="First reply.", request_id="r1")
        await asyncio.wait_for(engine.started.wait(), 2.0)
        assert len(session.live_tasks) == 1

        await session.feed_stream(chunk="Second reply.", done=True, request_id="r2")
        assert len(session.live_tasks) == 1

        engine.gate.set()
        await asyncio.wait_for(session.wait_idle(), 2.0)
        assert len(session.live_tasks) == 0

    asyncio.run(run())


def test_the_session_never_holds_two_live_synthesis_tasks():
    # The structural invariant, hammered: interleave both dispatch paths and assert
    # after every single dispatch. This is what makes "last audio_start wins, discard
    # binary until audio_end" a complete client rule.
    engine = _GatedEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)

        await session.speak(text="One.", request_id="r1")
        assert len(session.live_tasks) <= 1
        await session.feed_stream(chunk="Two.", done=False, request_id="r2")
        assert len(session.live_tasks) <= 1
        await session.feed_stream(chunk=" More.", done=True, request_id="r2")
        assert len(session.live_tasks) <= 1
        await session.speak(text="Three.", request_id="r3")
        assert len(session.live_tasks) <= 1

        engine.gate.set()
        await asyncio.wait_for(session.wait_idle(), 2.0)
        assert len(session.live_tasks) == 0

    asyncio.run(run())


def test_closing_the_session_stops_every_live_task():
    # The WS finally-block path: teardown leaves nothing running.
    engine = _GatedEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)
        await session.speak(text="Hello there.", request_id="r1")
        await asyncio.wait_for(engine.started.wait(), 2.0)
        assert len(session.live_tasks) == 1

        await session.aclose()
        assert len(session.live_tasks) == 0

    asyncio.run(run())


# --- the flush-timeout safety path still routes through the session ---

def test_the_session_tracks_the_active_stream_request_for_the_flush_timeout():
    # The receive loop arms its flush timeout off this id, so the session has to
    # carry it now that the loop holds no TTS state of its own. An open request keeps
    # it armed; done=true disarms it.
    engine = _FakeEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)
        assert session.active_stream_request_id is None

        await session.feed_stream(chunk="Hello", done=False, request_id="r1")
        assert session.active_stream_request_id == "r1"

        await session.feed_stream(chunk=" there.", done=True, request_id="r1")
        assert session.active_stream_request_id is None

        await asyncio.wait_for(session.wait_idle(), 2.0)

    asyncio.run(run())


def test_the_session_flush_timeout_closes_the_open_stream_request():
    engine = _FakeEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)
        await session.feed_stream(chunk="Hello there", done=False, request_id="r1")
        assert session.active_stream_request_id == "r1"

        await session.flush_timeout()
        await asyncio.wait_for(session.wait_idle(), 2.0)

        assert session.active_stream_request_id is None
        assert engine.calls == ["Hello there"]  # the buffer was spoken, not dropped
        ends = ws.audio_ends()
        assert len(ends) == 1
        assert ends[0]["reason"] == "complete"  # a flush-timeout close is normal

    asyncio.run(run())


# --- the ack is immediate ----------------------------------
#
# Every test below cancels DIRECTLY -- session.cancel(), a fake websocket, no receive
# loop and no tts_cancel frame -- which is the property server-side barge-in needs from
# the STT path.

ACK_BUDGET_S = 0.1  # audio_end must land within 100ms of the cancel call


def test_cancelling_a_one_shot_acks_immediately_while_the_executor_is_still_blocked():
    engine = _BlockingEngine(block_s=2.0)
    ws = _FakeWebSocket()

    async def run():
        with ThreadPoolExecutor(max_workers=1) as executor:
            session = _session(engine, ws, executor=executor)
            await session.speak(text="A long reply.", request_id="r1")
            await _await_flag(engine.entered)  # the thread is now parked INSIDE synthesis

            t0 = time.monotonic()
            await session.cancel("r1")
            elapsed = time.monotonic() - t0

            # The synthesis thread has NOT finished — that is the whole point. Today the
            # server would still be sitting on `await task` here for ~1s.
            assert not engine.released.is_set()
            assert elapsed < ACK_BUDGET_S, f"cancel took {elapsed*1000:.0f}ms to ack"

            ends = ws.audio_ends()
            assert len(ends) == 1
            assert ends[0]["request_id"] == "r1"
            assert ends[0]["reason"] == "cancelled"

            engine.released.set()
            await asyncio.wait_for(session.wait_idle(), 5.0)

    asyncio.run(run())


def test_cancelling_a_tts_stream_request_acks_immediately_too():
    # The immediate ack applies to BOTH paths. The streaming path is the one that used to answer a
    # cancel with its own trailing audio_end(reason='complete') as it unwound.
    engine = _BlockingEngine(block_s=2.0)
    ws = _FakeWebSocket()

    async def run():
        with ThreadPoolExecutor(max_workers=1) as executor:
            session = _session(engine, ws, executor=executor)
            await session.feed_stream(chunk="A long reply.", done=True, request_id="r1")
            await _await_flag(engine.entered)

            t0 = time.monotonic()
            await session.cancel("r1")
            elapsed = time.monotonic() - t0

            assert not engine.released.is_set()
            assert elapsed < ACK_BUDGET_S, f"cancel took {elapsed*1000:.0f}ms to ack"

            ends = ws.audio_ends()
            assert len(ends) == 1
            assert ends[0]["reason"] == "cancelled"

            engine.released.set()
            await asyncio.wait_for(session.wait_idle(), 5.0)

    asyncio.run(run())


# --- nothing follows the audio_end (exactly one audio_end, no PCM) ---


def test_no_pcm_and_exactly_one_audio_end_follow_a_cancelled_one_shot():
    engine = _BlockingEngine(block_s=2.0)
    ws = _FakeWebSocket()

    async def run():
        with ThreadPoolExecutor(max_workers=1) as executor:
            session = _session(engine, ws, executor=executor)
            await session.speak(text="A long reply.", request_id="r1")
            await _await_flag(engine.entered)

            await session.cancel("r1")

            # Let the blocked thread finish and the cancelled task unwind COMPLETELY --
            # the window in which it would otherwise emit its PCM and its own audio_end.
            engine.released.set()
            await asyncio.wait_for(session.wait_idle(), 5.0)

            assert len(ws.audio_ends()) == 1              # exactly one
            assert ws.audio_ends()[0]["reason"] == "cancelled"
            assert ws.binaries() == []                    # no PCM at all
            assert ws.frames_after_last_audio_end() == []  # nothing after the end
            _assert_last_audio_start_wins(ws)

    asyncio.run(run())


def test_no_pcm_and_exactly_one_audio_end_follow_a_cancelled_stream():
    engine = _BlockingEngine(block_s=2.0)
    ws = _FakeWebSocket()

    async def run():
        with ThreadPoolExecutor(max_workers=1) as executor:
            session = _session(engine, ws, executor=executor)
            await session.feed_stream(chunk="A long reply.", done=True, request_id="r1")
            await _await_flag(engine.entered)

            await session.cancel("r1")

            engine.released.set()
            await asyncio.wait_for(session.wait_idle(), 5.0)

            # The unwinding _synthesis_loop's finally-block still tries to send its own
            # audio_end. The client must never see it.
            assert len(ws.audio_ends()) == 1
            assert ws.audio_ends()[0]["reason"] == "cancelled"
            assert ws.binaries() == []
            assert ws.frames_after_last_audio_end() == []
            _assert_last_audio_start_wins(ws)

    asyncio.run(run())


# --- a cancelled request_id is permanently dead -----
#
# The hazard these close, server-side: a client cancels r1 and a token for r1 is still
# in flight on the wire. It lands after the cancel, re-opens r1, and the client hears
# audio it already killed. HermesConsole built a @Synchronized monitor to work around
# exactly this. A dead id stays dead, so the server can't be re-opened.


def test_a_cancelled_request_id_is_permanently_dead_for_the_life_of_the_connection():
    # Cancel r1, run r2 to completion, then let two LATE r1 frames land.
    # They must produce nothing at all — no synthesis, no audio_start, no audio_end,
    # no reopen — and must not touch r2 or raise.
    engine = _GatedEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)

        await session.feed_stream(chunk="First reply.", done=True, request_id="r1")
        await asyncio.wait_for(engine.started.wait(), 2.0)
        await session.cancel("r1")

        engine.gate.set()  # let the doomed r1 synthesis unwind completely
        await asyncio.wait_for(session.wait_idle(), 2.0)

        await session.speak(text="Second reply.", request_id="r2")
        await asyncio.wait_for(session.wait_idle(), 2.0)
        # r1 opened and was cut off; r2 then ran start-to-finish.
        assert [(p["type"], p["request_id"]) for p in ws.texts()] == [
            ("audio_start", "r1"), ("audio_end", "r1"),
            ("audio_start", "r2"), ("audio_end", "r2"),
        ]

        settled = list(ws.frames)

        # The late frames the client already cancelled. Neither may reach the engine.
        await session.feed_stream(chunk="stale token", done=False, request_id="r1")
        await session.feed_stream(chunk="", done=True, request_id="r1")
        await asyncio.wait_for(session.wait_idle(), 2.0)

        assert ws.frames == settled          # no frames for r1, none further for r2
        assert engine.calls == ["First reply.", "Second reply."]  # 'stale token' never spoken
        assert session.live_tasks == []      # r1 was not reopened
        _assert_last_audio_start_wins(ws)

    asyncio.run(run())


def test_a_late_frame_for_a_dead_id_does_not_disturb_the_current_request():
    # The dead-id rule's other half: the stale r1 frame must not cancel or disturb whatever is
    # current now. r2 is mid-synthesis when it lands, and must finish normally.
    engine = _GatedEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)

        await session.speak(text="First reply.", request_id="r1")
        await asyncio.wait_for(engine.started.wait(), 2.0)
        await session.cancel("r1")

        await session.speak(text="Second reply.", request_id="r2")  # r2 is now current
        assert len(session.live_tasks) == 1

        await session.feed_stream(chunk="stale token", done=True, request_id="r1")
        assert len(session.live_tasks) == 1  # r2 still live: the dead id disturbed nothing

        engine.gate.set()
        await asyncio.wait_for(session.wait_idle(), 2.0)

        ends = ws.audio_ends()
        assert [(e["request_id"], e["reason"]) for e in ends] == [
            ("r1", "cancelled"), ("r2", "complete")]  # r2 completed, it was not cancelled
        _assert_last_audio_start_wins(ws)

    asyncio.run(run())


def test_cancelling_a_request_that_never_started_is_idempotent_and_kills_the_id():
    # The cancel of an id with no prior request still acks exactly once and
    # still marks the id dead — so the token racing behind it cannot open the request.
    # (An earlier rule said "nothing in flight -> no audio_end". This one acks anyway,
    # because the ack is what makes the cancel unconditionally safe for the client.)
    engine = _FakeEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)

        await session.cancel("rX")
        assert ws.types() == ["audio_end"]
        assert ws.audio_ends()[0] == {
            "type": "audio_end", "request_id": "rX", "reason": "cancelled"}

        await session.feed_stream(chunk="text", done=True, request_id="rX")
        await asyncio.wait_for(session.wait_idle(), 2.0)

        assert ws.types() == ["audio_end"]  # no audio_start, no second audio_end
        assert ws.binaries() == []
        assert engine.calls == []
        _assert_last_audio_start_wins(ws)  # a bare cancel-ack is a legal stream

    asyncio.run(run())


def test_cancelling_twice_is_idempotent_and_emits_exactly_one_audio_end():
    # The global invariant: at most ONE audio_end per request_id over
    # the life of a connection. A duplicate tts_cancel is a no-op, not a second ack.
    engine = _GatedEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)

        await session.speak(text="A long reply.", request_id="r1")
        await asyncio.wait_for(engine.started.wait(), 2.0)

        await session.cancel("r1")
        await session.cancel("r1")

        engine.gate.set()
        await asyncio.wait_for(session.wait_idle(), 2.0)

        assert len(ws.audio_ends()) == 1
        assert ws.audio_ends()[0]["reason"] == "cancelled"
        _assert_last_audio_start_wins(ws)

    asyncio.run(run())


def test_cancelling_an_already_completed_one_shot_emits_no_second_audio_end():
    # The global invariant: a request that finished normally already sent its
    # ONE audio_end(reason='complete'). The barge-in race this project exists for -- the
    # last word's audio_end(complete) going out just as the client's cancel_tts(last_id)
    # fires -- must NOT produce a second audio_end. A completed id is terminal; a late
    # cancel of it is a no-op, exactly like a double-cancel.
    engine = _FakeEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)

        await session.speak(text="Hello there.", request_id="r1")
        await asyncio.wait_for(session.wait_idle(), 2.0)
        # r1 finished on its own: exactly one audio_end(complete) is on the wire.
        assert [(e["request_id"], e["reason"]) for e in ws.audio_ends()] == [
            ("r1", "complete")]

        await session.cancel("r1")  # the late barge-in cancel of the just-completed r1

        # Still exactly one audio_end for r1 -- no second, 'cancelled' one.
        assert [(e["request_id"], e["reason"]) for e in ws.audio_ends()] == [
            ("r1", "complete")]
        _assert_last_audio_start_wins(ws)

    asyncio.run(run())


def test_cancelling_an_already_completed_stream_emits_no_second_audio_end():
    # The same for the tts_stream done=true completion path (the other half of the fix:
    # the mechanism must mark BOTH the one-shot task and the feed_stream done=true path).
    engine = _FakeEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)

        await session.feed_stream(chunk="Hello there.", done=True, request_id="r1")
        await asyncio.wait_for(session.wait_idle(), 2.0)
        assert [(e["request_id"], e["reason"]) for e in ws.audio_ends()] == [
            ("r1", "complete")]

        await session.cancel("r1")

        assert [(e["request_id"], e["reason"]) for e in ws.audio_ends()] == [
            ("r1", "complete")]
        _assert_last_audio_start_wins(ws)

    asyncio.run(run())


def test_cancelling_a_completed_request_is_a_no_op_even_after_a_later_request_ran():
    # The broader invariant: "at most ONE audio_end per request_id over the LIFE of
    # the connection." r1 completes, r2 then runs to completion, and only THEN does the
    # stale cancel(r1) land. The completion must be remembered past r2, or the late cancel
    # emits r1's second audio_end.
    engine = _FakeEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)

        await session.speak(text="First.", request_id="r1")
        await asyncio.wait_for(session.wait_idle(), 2.0)
        await session.speak(text="Second.", request_id="r2")
        await asyncio.wait_for(session.wait_idle(), 2.0)

        await session.cancel("r1")  # stale, arriving two requests late

        assert [(e["request_id"], e["reason"]) for e in ws.audio_ends()] == [
            ("r1", "complete"), ("r2", "complete")]  # no third, 'cancelled' end for r1
        _assert_last_audio_start_wins(ws)

    asyncio.run(run())


def test_cancelling_a_never_started_id_still_acks_after_the_completion_fix():
    # Regression guard: marking completed ids terminal must NOT swallow the ack
    # for a genuinely never-seen id. A completed id is terminal (no-op); a never-seen id
    # is not -- it still acks, which is what makes an unconditional client cancel safe.
    engine = _FakeEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)
        await session.cancel("rX")
        assert ws.audio_ends() == [
            {"type": "audio_end", "request_id": "rX", "reason": "cancelled"}]

    asyncio.run(run())


def test_the_dead_id_store_is_bounded_and_keeps_the_most_recent_ids():
    # Structural: a long-lived socket cancels indefinitely, so the store cannot be
    # an unbounded set. It is a fixed-capacity ring holding at least the 32 most recent
    # ids — the window in which a stale token can still be in flight on the wire.
    engine = _FakeEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)
        store = session._dead_ids

        assert store.maxlen is not None and store.maxlen >= 32  # bounded, not a set

        for i in range(500):
            await session.cancel(f"r{i}")

        assert len(store) == store.maxlen  # memory does not grow without bound
        for i in range(500 - 32, 500):
            assert session._is_cancelled(f"r{i}")  # the 32 most recent are still dead

    asyncio.run(run())


def test_unwinding_does_not_leak_completed_tasks_on_a_long_lived_socket():
    # `_unwinding` is the list of doomed synthesis tasks a cancel cuts
    # loose. It was pruned ONLY in wait_idle(), which the production receive loop never
    # calls — so on a socket held for hours, every barge-in leaked one completed Task.
    # The bound on the sibling `_dead_ids` ring must hold here too.
    #
    # Drive ONLY production methods (feed_stream + cancel), never wait_idle, and let the
    # doomed FakeEngine tasks complete on their own between cycles — exactly what a
    # long-lived socket does. `_unwinding` must not grow with the cancel count.
    engine = _FakeEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)

        for i in range(200):
            await session.feed_stream(chunk="reply", done=True, request_id=f"r{i}")
            await session.cancel(f"r{i}")
            await asyncio.sleep(0)  # let the doomed task finish, as real wall-clock would

        assert len(session._unwinding) <= DEAD_ID_CAPACITY, (
            f"_unwinding grew to {len(session._unwinding)} tasks over 200 cancels — one "
            "completed Task leaked per barge-in (nothing prunes it outside wait_idle)"
        )
        assert len(session._dead_ids) == DEAD_ID_CAPACITY  # the sibling ring stayed capped

    asyncio.run(run())


# --- the receive loop keeps reading while the cancel unwinds ---------
#
# Everything above drives the TTSSession directly. These two close the loop
# at the layer the client actually talks to: the real WS endpoint,
# its real receive loop, and a blocking engine that is provably still parked in an
# executor thread at the moment the assertions run. The old code awaited the
# cancelled task right here, so the server read NO frames for the ~1s unwind — the
# barging-in human's audio sat unread in the socket buffer. That is the regression
# these guard.

BARGE_IN_PCM = b"\x00\x00" * 1600  # 100ms of 16kHz mono 16-bit PCM — a valid first frame

TRANSCRIPT_BUDGET_S = 0.5  # the barge-in audio is transcribed WHILE the executor unwinds


class _InstantSTT:
    """An STT pipeline that transcribes every frame at once — the human barging in.

    Stands in for QwenSTTPipeline in the receive loop: `feed` reporting an update is
    what makes the loop emit a transcription frame, so a transcription arriving is
    proof the loop read the frame rather than sitting on the cancelled synthesis.
    """

    def __init__(self, asr_model=None, **kwargs):
        self.frames_fed = 0

    async def feed(self, data: bytes) -> bool:
        self.frames_fed += 1
        return True

    def get_current_text(self):
        return ("stop talking", "en", 0)

    async def finalize(self):
        return ("", "en", 0)

    def reset(self) -> None:
        pass

    def dispose(self) -> None:
        pass


def _app_with_blocking_tts(engine, executor):
    """The real app, wired to the blocking engine and the instant STT — no models."""
    app = create_app()
    app.state.stt_engine_type = "qwen"  # the fake below is swapped in for the pipeline class
    app.state.qwen_model = object()
    app.state.tts_engine_type = "openvoice"
    app.state.tts_engine = engine
    app.state.tts_executor = executor
    return app


def test_tts_cancel_does_not_block_the_receive_loop(monkeypatch):
    """The barged-in-on server keeps reading while the executor unwinds.

    The client cancels and immediately speaks. Both must land promptly: the ack, and
    a transcription of audio that arrived while the doomed synthesis was still running.
    """
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    engine = _BlockingEngine(block_s=2.0)

    with ThreadPoolExecutor(max_workers=1) as executor:
        client = TestClient(_app_with_blocking_tts(engine, executor))
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"

            ws.send_json({"type": "tts", "text": "A long reply.", "request_id": "r1"})
            assert engine.entered.wait(timeout=3.0), "engine never entered synthesis"

            # The barge-in: cancel, then speak — without waiting for anything in between,
            # exactly as a client that just heard itself talked over would.
            t0 = time.monotonic()
            ws.send_json({"type": "tts_cancel", "request_id": "r1"})
            ws.send_bytes(BARGE_IN_PCM)

            # Collect the whole picture before judging it, so a stalled loop reports the
            # stall rather than tripping on whichever symptom happened to surface first.
            ack_at = None
            ack = None
            transcript_at = None
            pcm_after_cancel = 0
            while transcript_at is None:
                frame = ws.receive()
                if frame.get("bytes") is not None:
                    pcm_after_cancel += 1  # audio the client already told us to stop
                    continue
                payload = json.loads(frame["text"])
                if payload["type"] == "audio_end":
                    ack_at = time.monotonic() - t0
                    ack = payload
                elif payload["type"] == "transcription":
                    transcript_at = time.monotonic() - t0
                    assert payload["text"] == "stop talking"

            # The thread is STILL parked inside synthesis: the server did all of the above
            # while the cancelled request was unwinding, not after it finished.
            assert not engine.released.is_set()

            assert ack is not None, "the cancel was never acked"
            assert ack["request_id"] == "r1"
            assert ack["reason"] == "cancelled"
            assert ack_at < ACK_BUDGET_S, f"ack took {ack_at*1000:.0f}ms"
            assert transcript_at < TRANSCRIPT_BUDGET_S, (
                f"the barge-in audio went unread for {transcript_at*1000:.0f}ms — "
                "the receive loop was blocked on the cancelled synthesis"
            )
            assert pcm_after_cancel == 0, (
                f"{pcm_after_cancel} PCM frame(s) reached the client after its cancel"
            )

            engine.released.set()


def _tts_cancel_branch_source() -> str:
    """The body of the `elif msg_type == 'tts_cancel'` branch of the real receive loop."""
    lines = inspect.getsource(ws_module.websocket_endpoint).splitlines()
    start = next(i for i, ln in enumerate(lines) if 'msg_type == "tts_cancel"' in ln)
    indent = len(lines[start]) - len(lines[start].lstrip())
    body = []
    for line in lines[start + 1:]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break  # dedented back out — the branch is over
        body.append(line)
    assert body, "could not locate the tts_cancel branch body"
    return "\n".join(body)


def test_the_tts_cancel_branch_is_a_thin_delegation():
    """Structural: the branch calls the session and owns no cancellation logic.

    The logic must live on the session so the STT path can reach it when server-side
    barge-in lands — and the branch must never await the cancelled task, which
    is what would stall the receive loop all over again.
    """
    branch = _tts_cancel_branch_source()

    assert "tts_session.cancel(" in branch  # it delegates

    forbidden = {
        ".cancel()": "cancels a task itself instead of delegating",
        "AudioEndMessage": "sends the audio_end itself instead of delegating",
        "await task": "awaits the cancelled task — this is the stall",
        "wait_idle": "awaits the cancelled task — this is the stall",
        "create_task": "owns task lifecycle that belongs to the session",
        "_dead_ids": "reaches into the session's cancellation state",
    }
    for token, why in forbidden.items():
        assert token not in branch, f"tts_cancel branch {why}: {token!r}"


# --- superseding a request takes the SAME non-blocking cancel path ----
#
# A client that starts r2 while r1 is still speaking has cancelled r1 — it just did not
# say so in a `tts_cancel` frame. That supersede must behave exactly like the
# cancel above, and for the same reason. The old path did the opposite: feed_text() saw
# a new request_id and AWAITED the previous synthesis task, parking the receive loop
# inside an executor thread nobody can interrupt — the very stall the cancel path above
# exists to kill, reached through the door a barging-in client is most likely to use.


def test_a_supersede_acks_the_superseded_request_immediately():
    # audio_end(r1, reason='cancelled') within 100ms of the r2 chunk landing,
    # while the r1 synthesis is provably still parked in its executor thread.
    engine = _BlockingEngine(block_s=2.0)
    ws = _FakeWebSocket()

    async def run():
        with ThreadPoolExecutor(max_workers=1) as executor:
            session = _session(engine, ws, executor=executor)
            await session.feed_stream(chunk="A long reply.", done=True, request_id="r1")
            await _await_flag(engine.entered)

            t0 = time.monotonic()
            await session.feed_stream(chunk="New reply.", done=True, request_id="r2")
            elapsed = time.monotonic() - t0

            # The r1 thread has NOT finished — the supersede did not wait for it.
            assert not engine.released.is_set()
            assert elapsed < ACK_BUDGET_S, f"supersede took {elapsed*1000:.0f}ms to ack"

            ends = ws.audio_ends()
            assert len(ends) == 1
            assert ends[0]["request_id"] == "r1"
            assert ends[0]["reason"] == "cancelled"

            # And the single-flight invariant still holds across the supersede.
            assert len(session.live_tasks) == 1

            engine.released.set()
            await asyncio.wait_for(session.wait_idle(), 5.0)

    asyncio.run(run())


def test_a_superseded_request_sends_no_pcm_and_exactly_one_audio_end():
    # No PCM and one audio_end, for the supersede door: r1's doomed synthesis wakes up later and
    # tries to send its PCM and its own audio_end. The client must see neither — only
    # r2's audio follows r1's cancelled end.
    engine = _BlockingEngine(block_s=2.0)
    ws = _FakeWebSocket()

    async def run():
        with ThreadPoolExecutor(max_workers=1) as executor:
            session = _session(engine, ws, executor=executor)
            await session.feed_stream(chunk="A long reply.", done=True, request_id="r1")
            await _await_flag(engine.entered)

            await session.feed_stream(chunk="New reply.", done=True, request_id="r2")

            # Give the doomed r1 task its full chance to misbehave.
            engine.released.set()
            await asyncio.wait_for(session.wait_idle(), 5.0)

            assert [(p["type"], p["request_id"]) for p in ws.texts()] == [
                ("audio_start", "r1"), ("audio_end", "r1"),
                ("audio_start", "r2"), ("audio_end", "r2"),
            ]
            assert ws.audio_ends()[0]["reason"] == "cancelled"
            assert ws.audio_ends()[1]["reason"] == "complete"

            # Every binary frame lands after r2's audio_start: none is attributable to r1.
            r2_start = next(i for i, (kind, p) in enumerate(ws.frames)
                            if kind == "text" and p["type"] == "audio_start"
                            and p["request_id"] == "r2")
            binary_at = [i for i, (kind, _) in enumerate(ws.frames) if kind == "bytes"]
            assert binary_at, "r2 never spoke"
            assert min(binary_at) > r2_start
            assert engine.calls == ["A long reply.", "New reply."]
            _assert_last_audio_start_wins(ws)

    asyncio.run(run())


def test_a_superseded_request_id_is_permanently_dead():
    # The dead-id rule through the supersede door: r1 was killed by r2 starting, so a late r1
    # token still on the wire must be ignored outright — it cannot reopen r1, and it
    # must not disturb r2.
    engine = _GatedEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)

        await session.feed_stream(chunk="First reply.", done=True, request_id="r1")
        await asyncio.wait_for(engine.started.wait(), 2.0)

        await session.feed_stream(chunk="New reply.", done=True, request_id="r2")
        assert session._is_cancelled("r1")  # dead the moment it was superseded

        settled_ends = len(ws.audio_ends())

        # The stale r1 token, arriving after r2 took over.
        await session.feed_stream(chunk="stale token", done=False, request_id="r1")
        await session.speak(text="stale one-shot", request_id="r1")
        assert len(session.live_tasks) == 1  # r2 still live, r1 not reopened

        engine.gate.set()
        await asyncio.wait_for(session.wait_idle(), 2.0)

        assert engine.calls == ["First reply.", "New reply."]  # the stale text never spoke
        ends = ws.audio_ends()
        assert len(ends) == settled_ends + 1  # only r2's own end was added
        assert [(e["request_id"], e["reason"]) for e in ends] == [
            ("r1", "cancelled"), ("r2", "complete")]
        _assert_last_audio_start_wins(ws)

    asyncio.run(run())


def test_supersede_does_not_block_the_receive_loop(monkeypatch):
    """The supersede at the real endpoint: the barge-in audio is read during the unwind.

    The client speaks over the reply by starting a new one — a `tts_stream` for r2 while
    r1 is synthesizing — and immediately sends the audio it is barging in with. Both must
    land promptly while the r1 executor thread is still parked.
    """
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    engine = _BlockingEngine(block_s=2.0)

    with ThreadPoolExecutor(max_workers=1) as executor:
        client = TestClient(_app_with_blocking_tts(engine, executor))
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"

            ws.send_json({"type": "tts_stream", "text": "A long reply.",
                          "request_id": "r1", "done": True})
            assert engine.entered.wait(timeout=3.0), "engine never entered synthesis"

            t0 = time.monotonic()
            ws.send_json({"type": "tts_stream", "text": "New reply.",
                          "request_id": "r2", "done": True})
            ws.send_bytes(BARGE_IN_PCM)

            ack_at = None
            ack = None
            transcript_at = None
            pcm_frames = 0
            while transcript_at is None:
                frame = ws.receive()
                if frame.get("bytes") is not None:
                    pcm_frames += 1
                    continue
                payload = json.loads(frame["text"])
                if payload["type"] == "audio_end":
                    ack_at = time.monotonic() - t0
                    ack = payload
                elif payload["type"] == "transcription":
                    transcript_at = time.monotonic() - t0
                    assert payload["text"] == "stop talking"

            # Still parked in r1's synthesis: everything above happened DURING the unwind.
            assert not engine.released.is_set()

            assert ack is not None, "the superseded request was never ended"
            assert ack["request_id"] == "r1"
            assert ack["reason"] == "cancelled"
            assert ack_at < ACK_BUDGET_S, f"supersede ack took {ack_at*1000:.0f}ms"
            assert transcript_at < TRANSCRIPT_BUDGET_S, (
                f"the barge-in audio went unread for {transcript_at*1000:.0f}ms — "
                "the receive loop was blocked on the superseded synthesis"
            )
            # The one executor thread is still parked in r1, so no PCM can have been
            # produced by anyone: any frame here would be r1's, after its own end.
            assert pcm_frames == 0, (
                f"{pcm_frames} PCM frame(s) followed the superseded request's audio_end"
            )

            engine.released.set()


def test_the_supersede_path_in_feed_text_never_awaits_the_cancelled_task():
    """Structural: the stall is gone at its source, not just routed around.

    feed_text() used to `await self.cancel()` when it saw a new request_id — and cancel()
    awaits a task blocked in an executor thread. Any caller on the receive path (the WS
    loop today, server-side barge-in tomorrow) inherited that stall. The supersede must
    cut the doomed task loose instead, exactly as TTSSession.cancel() does.
    """
    source = inspect.getsource(TTSStreamPipeline.feed_text)

    assert "await self.cancel()" not in source, (
        "feed_text awaits the cancelled synthesis — this is the receive-loop stall"
    )
    assert "self.abandon()" in source, "feed_text must cut the superseded task loose"


# --- the frame-ordering invariant that lets a binary PCM frame stay bare -------
#
# Every test above checks one teardown in isolation; this one drives
# a single connection through BOTH doors at once -- an explicit cancel and a supersede -- plus
# a dead-id frame that must produce nothing, and proves the two halves of the invariant that
# together make an id on binary PCM frames unnecessary:
#
#   * Structural (asserted, not assumed): after EVERY dispatch the connection has at most one
#     live synthesis task. This is the single-flight tripwire; it is what keeps
#     audio_start/audio_end from ever interleaving in the first place.
#   * Observable: the whole recorded text-frame sequence keeps audio_start/audio_end strictly
#     alternating -- audio_start(rN+1) never precedes audio_end(rN) -- so "last audio_start
#     wins, discard binary until audio_end" is a complete and sufficient client rule.


def test_last_audio_start_wins_across_a_mixed_cancel_and_supersede_run():
    engine = _GatedEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)

        # r1: a one-shot, torn down by an explicit cancel while it is parked mid-synthesis.
        engine.started.clear()
        await session.speak(text="One.", request_id="r1")
        assert len(session.live_tasks) <= 1
        await asyncio.wait_for(engine.started.wait(), 2.0)  # audio_start(r1) is on the wire
        await session.cancel("r1")
        assert len(session.live_tasks) == 0  # the cancelled one-shot unwinds; it is not live

        # r2: a stream, torn down by the OTHER door -- r3 supersedes it while it is parked.
        engine.started.clear()
        await session.feed_stream(chunk="Two.", done=True, request_id="r2")
        assert len(session.live_tasks) <= 1
        await asyncio.wait_for(engine.started.wait(), 2.0)  # audio_start(r2) is on the wire
        engine.started.clear()
        await session.feed_stream(chunk="Three.", done=True, request_id="r3")  # supersedes r2
        assert len(session.live_tasks) <= 1  # r2 cut loose, r3 live -- never both at once

        # A stale token for the already-dead r1 lands mid-run: no frame, no task, no reopen.
        await session.feed_stream(chunk="stale", done=True, request_id="r1")
        assert len(session.live_tasks) <= 1

        engine.gate.set()  # let r1's and r2's doomed tasks unwind (silently) and r3 finish
        await asyncio.wait_for(session.wait_idle(), 3.0)
        assert len(session.live_tasks) == 0

        # The payoff: the client's whole view keeps exactly one request open at a time.
        _assert_last_audio_start_wins(ws)

        # Concretely -- each request closed once, by the reason its own teardown door implies.
        assert [(e["request_id"], e["reason"]) for e in ws.audio_ends()] == [
            ("r1", "cancelled"), ("r2", "cancelled"), ("r3", "complete")]

    asyncio.run(run())


# --- cancel() is scoped to the in-flight request ----------
#
# cancel(request_id) used to grab self.live_tasks regardless of the id and cancel it, so a
# client cancelling an id that is NOT the one in flight -- misuse the documented protocol
# says clients avoid, but defense-in-depth all the same -- killed the CURRENT reply: the
# stream became audio_start(r1)...audio_end(rX) with NO audio_end(r1), breaking the
# single-flight ordering invariant. The fix scopes the teardown to
# request_id == _active_request_id; an unrelated cancel is still acked and marked dead
# but leaves the in-flight task, and all its bookkeeping, untouched. The supersede
# path calls cancel(_active_request_id), so it still tears down -- proven by the supersede tests
# above re-running green.


def test_cancelling_an_unrelated_id_leaves_the_in_flight_request_running():
    engine = _GatedEngine()
    ws = _FakeWebSocket()

    async def run():
        session = _session(engine, ws)

        await session.speak(text="A long reply.", request_id="r1")
        await asyncio.wait_for(engine.started.wait(), 2.0)  # audio_start(r1) is on the wire
        assert len(session.live_tasks) == 1

        # Client misuse: cancel an id that is NOT the one synthesizing.
        await session.cancel("rX")

        # rX is acked and permanently dead, but r1's task was never the target -- it survives.
        assert session._is_cancelled("rX")
        assert not session._is_cancelled("r1")
        assert len(session.live_tasks) == 1, "cancel(rX) killed the in-flight r1 task"
        assert ws.audio_ends() == [
            {"type": "audio_end", "request_id": "rX", "reason": "cancelled"}]

        # r1 runs to its own normal completion and emits its OWN audio_end(complete).
        engine.gate.set()
        await asyncio.wait_for(session.wait_idle(), 2.0)

        assert engine.calls == ["A long reply."]  # rX triggered no synthesis; r1 ran once
        assert [(e["request_id"], e["reason"]) for e in ws.audio_ends()] == [
            ("rX", "cancelled"), ("r1", "complete")]
        _assert_last_audio_start_wins(ws)  # single-flight ordering held throughout

    asyncio.run(run())
