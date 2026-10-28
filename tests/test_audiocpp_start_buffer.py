"""Start buffer for the audiocpp tts_stream pipeline.

The first frames of a synthesis arrive from the sidecar at about real-time speed,
so a client that plays on arrival runs dry between them. The audiocpp stream
pipeline therefore holds the first VM_AUDIOCPP_TTS_START_BUFFER_MS of audio of
each engine request and sends it as one chunk; later chunks of that request go
out as they arrive.

What must hold:

  * with the buffer on, the first send of an engine request is one block of at
    least the buffer length, and later chunks pass through one by one;
  * a request that ends before the buffer fills sends what it made, once;
  * with the buffer at 0 every chunk is sent on arrival;
  * each engine request (each clause piece) holds again;
  * the unplayed-audio report counts every sample once, at the moment it is sent.

Model-free and server-free: a fake engine yields float32 chunks and a fake
WebSocket records the frames.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from voicemanager.pipeline.audiocpp_tts import AudioCppTTSStreamPipeline
from voicemanager.pipeline.tts import TTSPipeline, float32_to_pcm16

RATE = 24000
CHUNK = 1000  # samples per engine chunk, about 42 ms at 24 kHz
BUFFER_MS = 150
BUFFER_SAMPLES = RATE * BUFFER_MS // 1000  # 3600

_NO_TIMER = {"clause_first_words": 2, "clause_first_deadline_ms": 60_000}


class _Engine:
    """Yields `n_chunks` distinct float32 chunks of CHUNK samples per request."""

    is_batch = False
    sample_rate = RATE

    def __init__(self, n_chunks: int):
        self.n_chunks = n_chunks
        self.texts: list[str] = []
        self.chunks: list[np.ndarray] = []
        self.finished = 0

    async def ensure_ready(self) -> None:
        return None

    async def stream_synthesize(self, text: str, **kwargs):
        self.texts.append(text)
        for _ in range(self.n_chunks):
            i = len(self.chunks)
            chunk = ((np.arange(CHUNK) + i * CHUNK) % 200 - 100).astype(np.float32) / 200
            self.chunks.append(chunk)
            yield chunk
            await asyncio.sleep(0)
        self.finished += 1


class _WS:
    def __init__(self, events: list):
        self.events = events
        self.audio: list[bytes] = []

    async def send_text(self, data: str) -> None:
        json.loads(data)

    async def send_bytes(self, data: bytes) -> None:
        self.audio.append(data)
        self.events.append(("send", len(data) // 2))


def _pipe(engine: _Engine, events: list, **kwargs) -> AudioCppTTSStreamPipeline:
    pipe = AudioCppTTSStreamPipeline(TTSPipeline(engine=engine, executor=None),
                                     clause_continue=False, **{**_NO_TIMER, **kwargs})
    original = pipe._report_audio

    def spy(request_id: str, n_samples: int, sample_rate: int) -> None:
        events.append(("report", n_samples))
        original(request_id, n_samples, sample_rate)

    pipe._report_audio = spy
    return pipe


async def _until(predicate, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting on the condition")
        await asyncio.sleep(0.005)


def _one_request(n_chunks: int, buffer_ms: int) -> tuple[_Engine, _WS, list]:
    engine, events = _Engine(n_chunks), []
    ws = _WS(events)

    async def run():
        pipe = _pipe(engine, events, start_buffer_ms=buffer_ms)
        await pipe.feed_text("One two", True, ws, "r1", chunking="clause")
        await asyncio.wait_for(pipe.synthesis_task, 5.0)

    asyncio.run(run())
    return engine, ws, events


def _sends(events: list) -> list[int]:
    return [n for kind, n in events if kind == "send"]


def _assert_reported_on_send(engine: _Engine, ws: _WS, events: list) -> None:
    """Each send is followed by a report of its size; every sample counts once."""
    assert [kind for kind, _ in events] == ["send", "report"] * (len(events) // 2)
    for (_, sent), (_, reported) in zip(events[::2], events[1::2]):
        assert sent == reported
    total = sum(len(c) for c in engine.chunks)
    assert sum(n for kind, n in events if kind == "report") == total
    expected = float32_to_pcm16(np.concatenate(engine.chunks))
    assert b"".join(ws.audio) == expected


# --- one engine request ---------------------------------------------------------------

def test_the_first_send_holds_the_buffer_then_chunks_pass_through():
    engine, ws, events = _one_request(n_chunks=8, buffer_ms=BUFFER_MS)
    sends = _sends(events)
    # 4 chunks of 1000 samples are the first to reach 3600: one block.
    assert sends[0] >= BUFFER_SAMPLES
    assert sends == [4 * CHUNK] + [CHUNK] * 4
    _assert_reported_on_send(engine, ws, events)


def test_a_request_shorter_than_the_buffer_sends_once_at_its_end():
    engine, ws, events = _one_request(n_chunks=2, buffer_ms=BUFFER_MS)
    assert _sends(events) == [2 * CHUNK]
    _assert_reported_on_send(engine, ws, events)


def test_a_zero_buffer_sends_every_chunk_on_arrival():
    engine, ws, events = _one_request(n_chunks=8, buffer_ms=0)
    assert _sends(events) == [CHUNK] * 8
    _assert_reported_on_send(engine, ws, events)


# --- each clause piece holds again ------------------------------------------------------

def test_the_second_clause_piece_holds_again():
    engine, events = _Engine(6), []
    ws = _WS(events)

    async def run():
        pipe = _pipe(engine, events, start_buffer_ms=BUFFER_MS)
        await pipe.feed_text("One two", False, ws, "r1", chunking="clause")
        await _until(lambda: engine.finished == 1)
        await pipe.feed_text(" three four", True, ws, "r1")
        await asyncio.wait_for(pipe.synthesis_task, 5.0)

    asyncio.run(run())
    assert engine.texts == ["One two", "three four"]
    piece = [4 * CHUNK, CHUNK, CHUNK]
    assert _sends(events) == piece + piece
    _assert_reported_on_send(engine, ws, events)


# --- the setting -------------------------------------------------------------------------

def test_the_buffer_defaults_to_the_setting(monkeypatch):
    import voicemanager.config as cfg

    monkeypatch.setenv("VM_AUDIOCPP_TTS_START_BUFFER_MS", "275")
    cfg._settings = None
    try:
        pipe = AudioCppTTSStreamPipeline(TTSPipeline(engine=object(), executor=None))
        assert pipe.start_buffer_ms == 275
        explicit = AudioCppTTSStreamPipeline(TTSPipeline(engine=object(), executor=None),
                                             start_buffer_ms=0)
        assert explicit.start_buffer_ms == 0
    finally:
        cfg._settings = None


def test_the_setting_has_its_default_and_env_name():
    from voicemanager.config import Settings

    assert Settings(_env_file=None).audiocpp_tts_start_buffer_ms == 150
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    assert "# VM_AUDIOCPP_TTS_START_BUFFER_MS=150" in text
