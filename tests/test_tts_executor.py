"""Speaking must never block the receive loop.

The cloned-voice path already runs its blocking inference in the TTS executor.
The NON-cloned path -- which is the default: OpenVoice with no uploaded voice
profile -- did not. It consumed the engine's `stream_synthesize()` directly on the
event loop, and for a batch engine that async generator does its inference in the
CALLING context. So for the whole synthesis the loop was dead: the server read no
frames at all, including the barging-in human's STT audio and their tts_cancel.

These tests pin that down the only way that means anything: a heartbeat coroutine
ticks every 10ms alongside a synthesis, and we assert its worst-case gap. Before
the fix the gap is the full synthesis duration (the heartbeat ticks once); after
it, the loop stays live and the gap stays in the tens of milliseconds.

The streaming-engine path (VibeVoice) must NOT be offloaded -- it yields
incremental chunks from its own background thread, and consuming it on the loop
is what delivers them as they arrive. The last test guards that.

Model-free: fake engines, fake WebSocket, a real one-worker executor.
"""
import asyncio
import concurrent.futures
import json
import time

import authkit  # noqa: F401  (test auth env; import before voicemanager)

import numpy as np
import pytest

from voicemanager.pipeline.tts import TTSPipeline, TTSStreamPipeline


# How long the batch engine blocks its thread. Comfortably above the 250ms gap
# budget, so an un-offloaded synthesis fails the assertion by a wide margin
# rather than flirting with it.
BLOCKING_SYNTHESIS_SECONDS = 1.0

# The receive loop must never be stalled longer than this by a synthesis.
MAX_ACCEPTABLE_GAP_SECONDS = 0.25

HEARTBEAT_INTERVAL_SECONDS = 0.01


# --- test doubles ----------------------------------------------------------

class _BlockingBatchEngine:
    """A batch engine, like OpenVoice: synthesis blocks the thread it runs on.

    `is_batch` is the flag the pipeline reads to decide it must offload. The
    engine makes no attempt to be async itself -- that is exactly the point: the
    pipeline is what has to keep the loop alive, so the stub must not do it.
    """

    is_batch = True

    def __init__(self):
        self.calls: list[str] = []

    @property
    def sample_rate(self) -> int:
        return 24000

    def synthesize(self, text: str, speaker_key: str = "EN-US", speed: float = 1.0):
        self.calls.append(text)
        time.sleep(BLOCKING_SYNTHESIS_SECONDS)
        return np.zeros(8, dtype=np.float32), 24000

    async def stream_synthesize(self, text: str, **kwargs):
        # The batch engine's async wrapper: it does the blocking work in the
        # CALLING context, which is precisely why the pipeline must not consume
        # it on the event loop.
        audio_np, _ = self.synthesize(
            text,
            kwargs.get("speaker_key", "EN-US"),
            kwargs.get("speed", 1.0),
        )
        yield audio_np


class _StreamingEngine:
    """A genuinely-async engine, like VibeVoice: yields chunks without blocking."""

    def __init__(self):
        self.stream_calls: list[str] = []
        self.synthesize_calls: list[str] = []

    @property
    def sample_rate(self) -> int:
        return 24000

    def synthesize(self, text: str, speaker_key: str = "EN-US", speed: float = 1.0):
        self.synthesize_calls.append(text)
        return np.zeros(8, dtype=np.float32), 24000

    async def stream_synthesize(self, text: str, **kwargs):
        self.stream_calls.append(text)
        for _ in range(3):
            await asyncio.sleep(0)
            yield np.zeros(4, dtype=np.float32)


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
        return [p["type"] for p in self.texts()]

    def audio_bytes(self) -> list[bytes]:
        return [p for kind, p in self.frames if kind == "bytes"]


# --- the heartbeat -----------------------------------------------------------

async def _worst_gap_during(coro) -> tuple[float, object]:
    """Run `coro`, ticking every 10ms alongside it. Returns (worst gap, result).

    A gap is wall-clock time between two consecutive ticks. While the event loop
    is blocked, `asyncio.sleep()` cannot return -- so the worst gap IS the length
    of the longest stall the receive loop would have suffered.
    """
    gaps: list[float] = []
    stop = asyncio.Event()
    ticking = asyncio.Event()

    async def heartbeat():
        ticking.set()
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    # The heartbeat must be running BEFORE the work starts. Without this the work
    # can run to completion without ever yielding, no tick is ever recorded, and
    # the measurement passes while proving nothing.
    await ticking.wait()
    try:
        result = await coro
    finally:
        stop.set()
        await beat

    assert gaps, "heartbeat never ticked -- this measurement is vacuous"
    return max(gaps), result


@pytest.fixture
def executor():
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    yield ex
    ex.shutdown(wait=True)


# --- the loop stays live through a non-cloned synthesis ----------------------

def test_tts_with_no_voice_profile_does_not_stall_the_event_loop(executor):
    # The default configuration: batch engine, target_se is None. Today this
    # blocks the loop for the entire synthesis.
    engine = _BlockingBatchEngine()
    pipe = TTSPipeline(engine=engine, executor=executor)
    ws = _FakeWebSocket()
    assert pipe.target_se is None

    async def run():
        worst_gap, _ = await _worst_gap_during(
            pipe.synthesize_and_stream(
                websocket=ws, text="Hello there.", request_id="r1",
            )
        )
        assert worst_gap < MAX_ACCEPTABLE_GAP_SECONDS, (
            f"event loop stalled {worst_gap * 1000:.0f}ms during synthesis; "
            f"the receive loop was deaf to STT audio and tts_cancel that whole time"
        )
        # and the synthesis really did happen, on a thread that is not the loop
        assert engine.calls == ["Hello there."]

    asyncio.run(run())


def test_tts_stream_with_no_voice_profile_does_not_stall_the_event_loop(executor):
    engine = _BlockingBatchEngine()
    pipe = TTSStreamPipeline(TTSPipeline(engine=engine, executor=executor))
    ws = _FakeWebSocket()

    async def run():
        async def feed_and_drain():
            await pipe.feed_text(
                chunk="Hello there.", done=True,
                websocket=ws, request_id="r1",
            )
            await asyncio.wait_for(pipe._synthesis_task, 10.0)

        worst_gap, _ = await _worst_gap_during(feed_and_drain())
        assert worst_gap < MAX_ACCEPTABLE_GAP_SECONDS, (
            f"event loop stalled {worst_gap * 1000:.0f}ms during streaming synthesis"
        )
        assert engine.calls == ["Hello there."]

    asyncio.run(run())


# --- behaviour is otherwise unchanged ----------------------------------------

def test_the_offloaded_synthesis_still_sends_audio_start_pcm_audio_end(executor):
    engine = _BlockingBatchEngine()
    pipe = TTSPipeline(engine=engine, executor=executor)
    ws = _FakeWebSocket()

    async def run():
        await pipe.synthesize_and_stream(
            websocket=ws, text="Hello there.", request_id="r1",
        )

        assert ws.types() == ["audio_start", "audio_end"]
        start = ws.texts()[0]
        assert start["request_id"] == "r1"
        assert start["sample_rate"] == 24000
        # 8 float32 samples -> 8 int16 samples -> 16 bytes
        assert ws.audio_bytes() == [b"\x00" * 16]

    asyncio.run(run())


# --- the streaming engine must NOT be offloaded ------------------------------

def test_a_streaming_engine_is_still_consumed_on_the_event_loop(executor):
    # VibeVoice yields incremental chunks from its own background thread. Pushing
    # that through the executor would collapse it into one batch and throw away
    # the low first-audio latency that is the whole point of it.
    engine = _StreamingEngine()
    pipe = TTSPipeline(engine=engine, executor=executor)
    ws = _FakeWebSocket()

    async def run():
        await pipe.synthesize_and_stream(
            websocket=ws, text="Hello there.", request_id="r1",
        )

        assert engine.stream_calls == ["Hello there."]
        assert engine.synthesize_calls == []
        # all three incremental chunks reached the wire, separately
        assert len(ws.audio_bytes()) == 3

    asyncio.run(run())
