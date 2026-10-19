"""tts_stream batching under the audiocpp TTS engine.

Model-free and server-free (the CLAUDE.md invariant): the real app and WS
receive loop over a `TestClient` for the end-to-end cases, the real stream
pipeline driven directly for the exact-boundary case, and the fake TTS sidecar
(tests/audiocpp_tts_fake.py) whose gate makes synthesis duration controllable.
No GPU, no binary, no model.

Busy-period batching, the mapping under test:

  * the FIRST completed sentence dispatches as a sidecar request IMMEDIATELY -
    never gated on further text arriving;
  * ALL text that accumulates while a request synthesizes coalesces into
    exactly ONE next request (one request per busy period - dots' internal
    chunker then handles the text with continuous vocoder state, the smooth
    seam path);
  * trailing text with no sentence boundary flushes after
    VM_TTS_STREAM_FLUSH_TIMEOUT with its existing meaning;
  * other engines' tts_stream behavior is byte-identical (their pipeline class
    is untouched; the batching variant is selected only under audiocpp).

The sentences are long (>10 words) because the shared split_sentences() only
closes a group past that length - the exact per-feed splits asserted here were
verified against the real splitter.
"""
from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import authkit  # test auth env; must be imported before voicemanager

import numpy as np
import pytest
from fastapi.testclient import TestClient

import voicemanager.config as cfg
from voicemanager.app import create_app
from voicemanager.pipeline.audiocpp_tts import AudioCppTTSEngine
from voicemanager.pipeline.tts import TTSPipeline, float32_to_pcm16
from voicemanager.transport import websocket as ws_module

from audiocpp_tts_fake import DEFAULT_CHUNKS, DEFAULT_FLUSH, FakeTtsSidecar

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_REFS = REPO_ROOT / "voices" / "designed" / "refs.json"

# Four >10-word sentences; the feed chunks split them mid-word so sentence
# boundaries only ever appear inside an accumulated buffer.
S1 = "The quick brown fox jumps over the lazy dog near the river bank every single morning."
S2 = "Meanwhile the careful gray owl watches quietly from the tallest branch of the old oak tree."
S3 = "Down in the valley the patient badger digs a new tunnel beneath the crumbling stone wall."
S4 = "Far above them all a lonely hawk traces slow circles through the cold and cloudless sky."

#: Verified against the real splitter: c1 dispatches exactly [S1]; c2 dispatches
#: exactly [S2, S3]; c3 leaves S4's tail as the buffer for the done/timeout flush.
C1 = S1 + " Meanwhile the careful gray owl watches"
C2 = " quietly from the tallest branch of the old oak tree. " + S3 + " Far above them all a lonely"
C3 = " hawk traces slow circles through the cold and cloudless sky."

#: Everything that accumulated while S1 synthesized - S2, S3 and the
#: done-flushed S4 - coalesces into exactly ONE next request.
COALESCED = f"{S2} {S3} {S4}"


def _delivered_pcm(*payloads: bytes) -> bytes:
    """The exact PCM the pipeline delivers for these sidecar payloads (s16le ->
    float32 /32768 -> float32_to_pcm16), so assertions stay byte-exact."""
    floats = np.concatenate([
        np.frombuffer(p, dtype="<i2").astype(np.float32) / 32768.0 for p in payloads])
    return float32_to_pcm16(floats)


ONE_REQUEST_PCM = _delivered_pcm(*DEFAULT_CHUNKS, DEFAULT_FLUSH)


class _InstantSTT:
    """Minimal QwenSTTPipeline stand-in: the receive loop needs an STT pipeline."""

    def __init__(self, asr_model=None, **kwargs):
        pass

    async def feed(self, data: bytes) -> bool:
        return False

    def get_current_text(self):
        return ("", "en", 0)

    async def finalize(self):
        return ("", "en", 0)

    def reset(self) -> None:
        pass

    def dispose(self) -> None:
        pass


class _FakeWS:
    """Captures the frames a synthesis loop sends: JSON dicts and PCM bytes."""

    def __init__(self):
        self.frames: list[dict | bytes] = []

    async def send_text(self, data: str) -> None:
        self.frames.append(json.loads(data))

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append(data)


def _tts_app(engine, executor):
    app = create_app()
    app.state.stt_engine_type = "qwen"  # swapped for _InstantSTT via monkeypatch
    app.state.qwen_model = object()
    app.state.tts_engine_type = "audiocpp"
    app.state.tts_engine = engine
    app.state.tts_executor = executor
    return app


def _external_engine(fake: FakeTtsSidecar) -> AudioCppTTSEngine:
    engine = AudioCppTTSEngine({"socket": fake.socket_path})
    engine.load(voices_path=str(REAL_REFS))
    return engine


async def _await_requests(fake: FakeTtsSidecar, count: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while len(fake.requests) < count:
        if time.monotonic() > deadline:
            raise AssertionError(
                f"expected {count} sidecar requests within {timeout}s, "
                f"have {[r['text'][:30] for r in fake.requests]}")
        await asyncio.sleep(0.01)


# --- the exact request boundaries, driven at the pipeline seam -----------------------

def test_first_sentence_alone_then_one_coalesced_request() -> None:
    """The batching boundary claim, exactly: [S1 alone, ONE coalesced busy-period
    request] - a first dispatch delayed for more text, or S2/S3/S4 arriving as
    separate queued requests, both fail."""
    from voicemanager.pipeline.audiocpp_tts import AudioCppTTSStreamPipeline

    with FakeTtsSidecar(gate_after_first=True) as fake:
        engine = _external_engine(fake)
        ws = _FakeWS()

        async def scenario() -> None:
            with ThreadPoolExecutor(max_workers=1) as executor:
                pipeline = TTSPipeline(engine=engine, executor=executor)
                stream = AudioCppTTSStreamPipeline(tts_pipeline=pipeline)

                # c1 completes S1: it must dispatch NOW, while nothing else has
                # arrived - the fake holds it in flight (gate) once dispatched.
                await stream.feed_text(chunk=C1, done=False, websocket=ws,
                                       request_id="s1")
                await _await_requests(fake, 1)
                assert fake.requests[0]["text"] == S1, \
                    "the first completed sentence must dispatch immediately, alone"

                # The busy period: S2 and S3 complete, S4 arrives with done=True -
                # all while the fake still holds request 1 in flight.
                await stream.feed_text(chunk=C2, done=False, websocket=ws,
                                       request_id="s1")
                await stream.feed_text(chunk=C3, done=True, websocket=ws,
                                       request_id="s1")
                assert len(fake.requests) == 1, \
                    "nothing new may dispatch while a request is synthesizing"

                # Request 1 finishes: everything that accumulated coalesces into
                # exactly ONE next request.
                fake.proceed.set()
                task = stream.synthesis_task
                assert task is not None
                await asyncio.wait_for(task, timeout=10.0)

        try:
            asyncio.run(scenario())
        finally:
            engine.dispose()

        assert [r["text"] for r in fake.requests] == [S1, COALESCED], (
            "busy-period text must coalesce into exactly ONE next request; got "
            f"{[r['text'][:40] for r in fake.requests]}")
        # One audio_start (48 kHz), the PCM of both requests, one audio_end.
        starts = [f for f in ws.frames if isinstance(f, dict) and f["type"] == "audio_start"]
        ends = [f for f in ws.frames if isinstance(f, dict) and f["type"] == "audio_end"]
        assert len(starts) == 1 and starts[0]["sample_rate"] == 48000
        assert len(ends) == 1
        pcm = b"".join(f for f in ws.frames if isinstance(f, bytes))
        assert pcm == ONE_REQUEST_PCM * 2, \
            "both requests' audio must be delivered completely, in order"


# --- end to end over the real WS receive loop ----------------------------------------

def test_ws_tts_stream_batches_busy_period_into_one_request(monkeypatch) -> None:
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    with FakeTtsSidecar(gate_after_first=True) as fake:
        engine = _external_engine(fake)
        with ThreadPoolExecutor(max_workers=1) as executor:
            client = TestClient(_tts_app(engine, executor))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"

                ws.send_json({"type": "tts_stream", "text": C1, "request_id": "s1"})
                # The first sentence is in flight (held by the fake's gate)...
                deadline = time.monotonic() + 5.0
                while len(fake.requests) < 1 and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert [r["text"] for r in fake.requests] == [S1]

                # ...while the rest of the text arrives.
                ws.send_json({"type": "tts_stream", "text": C2, "request_id": "s1"})
                ws.send_json({"type": "tts_stream", "text": C3, "request_id": "s1",
                              "done": True})
                time.sleep(0.5)  # let the receive loop ingest both feeds
                fake.proceed.set()

                start = ws.receive_json()
                assert start["type"] == "audio_start"
                assert start["sample_rate"] == 48000
                pcm = b""
                while True:
                    frame = ws.receive()
                    if frame.get("bytes") is not None:
                        pcm += frame["bytes"]
                        continue
                    end = json.loads(frame["text"])
                    assert end["type"] == "audio_end", end
                    break

        assert [r["text"] for r in fake.requests] == [S1, COALESCED]
        assert pcm == ONE_REQUEST_PCM * 2


# --- the flush timeout keeps its meaning for a trailing fragment ---------------------

def test_ws_tts_stream_audio_start_carries_the_ready_stated_24k_rate(monkeypatch) -> None:
    """Under a 24000 READY (breeze) the
    tts_stream reply's audio_start says 24000 while the 48 kHz cases above hold."""
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    with FakeTtsSidecar(ready_sample_rate=24000) as fake:
        engine = AudioCppTTSEngine({"socket": fake.socket_path,
                                    "model_task": "breeze-tts-streaming"})
        engine.load(voices_path=str(REAL_REFS))
        with ThreadPoolExecutor(max_workers=1) as executor:
            client = TestClient(_tts_app(engine, executor))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"
                ws.send_json({"type": "tts_stream", "text": S1, "request_id": "b1",
                              "done": True})
                start = ws.receive_json()
                assert start["type"] == "audio_start"
                assert start["sample_rate"] == 24000
                pcm = b""
                while True:
                    frame = ws.receive()
                    if frame.get("bytes") is not None:
                        pcm += frame["bytes"]
                        continue
                    end = json.loads(frame["text"])
                    assert end["type"] == "audio_end", end
                    break
        assert [r["text"] for r in fake.requests] == [S1]
        assert pcm == ONE_REQUEST_PCM


def test_ws_flush_timeout_dispatches_trailing_fragment(monkeypatch) -> None:
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    monkeypatch.setenv("VM_TTS_STREAM_FLUSH_TIMEOUT", "1")  # int field; 1 s minimum
    cfg._settings = None
    fragment = "a trailing fragment that never gets a sentence boundary"
    try:
        with FakeTtsSidecar() as fake:
            engine = _external_engine(fake)
            with ThreadPoolExecutor(max_workers=1) as executor:
                client = TestClient(_tts_app(engine, executor))
                with authkit.ws_connect(client, "/ws/stream") as ws:
                    assert ws.receive_json()["type"] == "ready"

                    ws.send_json({"type": "tts_stream", "text": fragment,
                                  "request_id": "f1"})
                    # No boundary, no done: only VM_TTS_STREAM_FLUSH_TIMEOUT may
                    # dispatch it.
                    start = ws.receive_json()
                    assert start["type"] == "audio_start"
                    pcm = b""
                    while True:
                        frame = ws.receive()
                        if frame.get("bytes") is not None:
                            pcm += frame["bytes"]
                            continue
                        end = json.loads(frame["text"])
                        assert end["type"] == "audio_end", end
                        break

            assert [r["text"] for r in fake.requests] == [fragment], \
                "the flush timeout must dispatch exactly the trailing fragment"
            assert pcm == ONE_REQUEST_PCM
    finally:
        cfg._settings = None


# --- the bounded coalescing window ---------------------------------------------------

#: A slow producer: each chunk completes exactly one sentence, and the next only
#: arrives after the previous request has already finished synthesizing. This is
#: HermesConsole's shape - it forwards every Hermes delta verbatim with no
#: client-side assembly - and it is the case the busy-period get_nowait drain cannot help.
#: Verified against the real splitter (it merges short trailing fragments, so each
#: chunk must leave a substantial one): d1 dispatches [S1], d2 dispatches [S2],
#: d3 flushes S3 on done.
D1 = S1 + " Meanwhile the careful gray owl watches"
D2 = " quietly from the tallest branch of the old oak tree. Down in the valley the patient badger"
D3 = " digs a new tunnel beneath the crumbling stone wall."


def _slow_producer_requests(coalesce_ms: int) -> list[str]:
    """Drive the slow-producer sequence and return the sidecar request texts."""
    from voicemanager.pipeline.audiocpp_tts import AudioCppTTSStreamPipeline

    with FakeTtsSidecar() as fake:
        engine = _external_engine(fake)
        ws = _FakeWS()

        async def scenario() -> None:
            with ThreadPoolExecutor(max_workers=1) as executor:
                pipeline = TTSPipeline(engine=engine, executor=executor)
                stream = AudioCppTTSStreamPipeline(
                    tts_pipeline=pipeline, coalesce_ms=coalesce_ms)

                # S1 dispatches immediately and alone - never gated.
                await stream.feed_text(chunk=D1, done=False, websocket=ws,
                                       request_id="r1")
                await _await_requests(fake, 1)

                # S2 arrives only after request 1 has drained, so the get_nowait
                # drain finds an empty queue - the whole point of this scenario.
                await stream.feed_text(chunk=D2, done=False, websocket=ws,
                                       request_id="r1")
                if coalesce_ms == 0:
                    # Without a window S2 goes out on its own; wait for it so the
                    # comparison is against settled state, not a race.
                    await _await_requests(fake, 2)
                else:
                    await asyncio.sleep(0.05)  # well inside the window

                await stream.feed_text(chunk=D3, done=True, websocket=ws,
                                       request_id="r1")
                task = stream.synthesis_task
                assert task is not None
                await asyncio.wait_for(task, timeout=10.0)

        try:
            asyncio.run(scenario())
        finally:
            engine.dispose()
        return [r["text"] for r in fake.requests]


def test_coalesce_window_merges_a_slow_producers_sentences() -> None:
    """With the window on, sentences that arrive one at a time still reach
    the sidecar as ONE request, so Breeze conditions the speaker once instead of
    re-sampling it per sentence."""
    assert _slow_producer_requests(coalesce_ms=400) == [S1, f"{S2} {S3}"]


def test_coalesce_window_of_zero_keeps_the_shipped_d09_behavior() -> None:
    """0 is an exact escape hatch to busy-period batching: each sentence its own request."""
    assert _slow_producer_requests(coalesce_ms=0) == [S1, S2, S3]


def test_coalesce_window_never_delays_the_first_sentence() -> None:
    """The first-sentence rule still holds: the first completed sentence dispatches without waiting
    out the window, even with a long one configured."""
    from voicemanager.pipeline.audiocpp_tts import AudioCppTTSStreamPipeline

    with FakeTtsSidecar() as fake:
        engine = _external_engine(fake)
        ws = _FakeWS()

        async def scenario() -> float:
            with ThreadPoolExecutor(max_workers=1) as executor:
                pipeline = TTSPipeline(engine=engine, executor=executor)
                stream = AudioCppTTSStreamPipeline(
                    tts_pipeline=pipeline, coalesce_ms=5000)
                t0 = time.monotonic()
                await stream.feed_text(chunk=D1, done=False, websocket=ws,
                                       request_id="r1")
                await _await_requests(fake, 1)
                elapsed = time.monotonic() - t0
                await stream.feed_text(chunk="", done=True, websocket=ws,
                                       request_id="r1")
                task = stream.synthesis_task
                if task is not None:
                    await asyncio.wait_for(task, timeout=10.0)
                return elapsed

        try:
            elapsed = asyncio.run(scenario())
        finally:
            engine.dispose()

    assert fake.requests[0]["text"] == S1
    assert elapsed < 2.0, (
        f"first sentence waited {elapsed:.2f}s - the window must never gate it")
