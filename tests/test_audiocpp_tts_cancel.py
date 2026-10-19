"""Cancel semantics under the audiocpp TTS engine.

Model-free and server-free (the CLAUDE.md invariant): the real app and WS
receive loop over a `TestClient`, and the fake TTS sidecar
(tests/audiocpp_tts_fake.py) speaking the real wire protocol - including the
CANCEL frame (0x0D) - over a genuine AF_UNIX socket. No GPU, no binary,
no model.

The mechanism under test is the chosen one: SINK-THROW SESSION REUSE.
On cancel the engine sends a wire CANCEL; the sidecar's sink throws the
sentinel, generation stops, and the request closes with TTS_DONE
{cancelled:true} on the SAME still-warm connection. The engine's wire hygiene
(ensure_wire_clean) then guarantees the next dispatch never reads the
cancelled request's leftover frames as its own audio. What must hold:

  * after a tts_cancel, the client receives ZERO further audio from the
    cancelled request - the cancel ack (audio_end reason=cancelled) is the
    request's last frame;
  * a new tts request SUPERSEDING an in-flight one gets the same guarantee;
  * the NEXT request on the same session synthesizes correctly and its
    delivered PCM is byte-exact - no leaked frames from the cancelled request
    can prefix it (the defect this task exists to prevent);
  * the sidecar actually receives the CANCEL (generation stops server-side,
    not just client-side silence).

The ~1 s audible-stop bound is the live half, timed in a live run.
"""
from __future__ import annotations

import asyncio
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import authkit  # test auth env; must be imported before voicemanager

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voicemanager.app import create_app
from voicemanager.pipeline.audiocpp_tts import AudioCppTTSEngine
from voicemanager.pipeline.tts import float32_to_pcm16
from voicemanager.transport import websocket as ws_module

from audiocpp_tts_fake import DEFAULT_CHUNKS, DEFAULT_FLUSH, FakeTtsSidecar

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_REFS = REPO_ROOT / "voices" / "designed" / "refs.json"


def _delivered_pcm(*payloads: bytes) -> bytes:
    """The exact bytes the WS pipeline delivers for these sidecar payloads:
    the engine's s16le -> float32 (/32768) then the pipeline's shared
    float32_to_pcm16 back out. Deterministic, so the leak-proof assertions can
    stay byte-exact without re-litigating the pre-existing PCM conversion."""
    floats = np.concatenate([
        np.frombuffer(p, dtype="<i2").astype(np.float32) / 32768.0 for p in payloads])
    return float32_to_pcm16(floats)


#: What one complete default request delivers end to end.
FULL_REQUEST_PCM = _delivered_pcm(*DEFAULT_CHUNKS, DEFAULT_FLUSH)


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


def _drain_request(ws) -> tuple[dict, bytes, dict]:
    """Read one complete TTS reply: audio_start, concatenated PCM, audio_end."""
    start = ws.receive_json()
    assert start["type"] == "audio_start", start
    pcm = b""
    while True:
        frame = ws.receive()
        if frame.get("bytes") is not None:
            pcm += frame["bytes"]
            continue
        payload = json.loads(frame["text"])
        assert payload["type"] == "audio_end", payload
        return start, pcm, payload


# --- engine level: cancelled leftovers never leak into the next request --------------

def test_cancelled_request_leftovers_never_leak_into_the_next_request() -> None:
    """Cancel the consuming task mid-request, then synthesize again on the same
    engine: the second request's chunks must be EXACTLY its own - the cancelled
    request's undelivered frames (chunk B, the flush tail, its TTS_DONE) must
    have been drained off the wire, never surfaced as the new request's audio."""
    with FakeTtsSidecar(gate_after_first=True) as fake:
        engine = _external_engine(fake)

        async def scenario() -> list[np.ndarray]:
            got_first = asyncio.Event()

            async def consume() -> None:
                async for _chunk in engine.stream_synthesize("doomed request"):
                    got_first.set()

            task = asyncio.create_task(consume())
            await asyncio.wait_for(got_first.wait(), timeout=5.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            # The cancel is done with the gate; release it so the follow-up
            # request streams without waiting out the fake's 5 s gate deadline.
            fake.proceed.set()
            second: list[np.ndarray] = []
            async for chunk in engine.stream_synthesize("clean follow-up"):
                second.append(chunk)
            return second

        try:
            second = asyncio.run(scenario())
        finally:
            engine.dispose()

        expected = [np.frombuffer(p, dtype="<i2").astype(np.float32) / 32768.0
                    for p in (*DEFAULT_CHUNKS, DEFAULT_FLUSH)]
        assert len(second) == len(expected), (
            f"the follow-up must deliver exactly its own {len(expected)} chunks, "
            f"got {len(second)} - leaked frames from the cancelled request?")
        for got, want in zip(second, expected):
            assert np.array_equal(got, want), \
                "follow-up audio must be byte-exact - no leaked cancelled-request frames"
        assert [r["text"] for r in fake.requests] == ["doomed request", "clean follow-up"]
        assert len(fake.cancels) == 1, \
            "the sidecar must receive the wire CANCEL (generation stops server-side)"


# --- engine level: a STALE unwind cannot cancel the successor ---------

def test_stale_unwind_after_supersede_sends_no_cancel_and_next_request_is_whole() -> None:
    """The late-unwind edge: a doomed request's generator finalizes LATE - its
    GeneratorExit lands only AFTER the superseding request's send_text has taken
    the wire (e.g. a GC-scheduled finalization). The unwind's cancel is scoped
    to the generation its own dispatch returned, so it must send NOTHING - an
    unscoped cancel here would truncate the successor mid-stream. The common
    supersede path itself still sends exactly one CANCEL (the wire hygiene that
    stops the doomed request server-side)."""
    with FakeTtsSidecar(gate_after_first=True) as fake:
        engine = _external_engine(fake)

        async def scenario() -> list[np.ndarray]:
            # r1 mid-flight: dispatched, first chunk out, fake gated.
            agen1 = engine.stream_synthesize("doomed request")
            await agen1.__anext__()

            # r2 supersedes while r1's unwind has NOT run yet: its send_text's
            # wire hygiene sends the one legitimate CANCEL, drains r1, and bumps
            # the wire generation. r2 is then itself gated after its first chunk.
            agen2 = engine.stream_synthesize("the new request")
            chunks = [await agen2.__anext__()]

            # THE EDGE: r1 finalizes late, with r2 mid-flight on the wire. The
            # scoped cancel must recognize the wire moved on and send nothing.
            await agen1.aclose()

            fake.proceed.set()  # release r2's gate; it must stream to completion
            async for chunk in agen2:
                chunks.append(chunk)
            return chunks

        try:
            second = asyncio.run(scenario())
        finally:
            engine.dispose()

        expected = [np.frombuffer(p, dtype="<i2").astype(np.float32) / 32768.0
                    for p in (*DEFAULT_CHUNKS, DEFAULT_FLUSH)]
        assert len(second) == len(expected), (
            f"the superseding request must deliver all {len(expected)} of its own "
            f"chunks, got {len(second)} - did the stale unwind truncate it?")
        for got, want in zip(second, expected):
            assert np.array_equal(got, want), \
                "the superseding request's audio must be byte-exact"
        assert [r["text"] for r in fake.requests] == \
            ["doomed request", "the new request"]
        assert len(fake.cancels) == 1, (
            "exactly ONE cancel may reach the sidecar (the supersede's wire "
            f"hygiene); got {len(fake.cancels)} - the stale unwind must send none")


# --- WS level: tts_cancel, then a clean next request ---------------------------------

def test_ws_tts_cancel_stops_audio_and_next_request_is_clean(monkeypatch) -> None:
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    with FakeTtsSidecar(gate_after_first=True) as fake:
        engine = _external_engine(fake)
        with ThreadPoolExecutor(max_workers=1) as executor:
            client = TestClient(_tts_app(engine, executor))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"

                # r1 is mid-request: audio_start plus its first PCM frame arrived.
                ws.send_json({"type": "tts", "text": "doomed", "request_id": "r1"})
                assert ws.receive_json()["type"] == "audio_start"
                first = ws.receive()
                assert first.get("bytes"), "r1 must be streaming before the cancel"

                # Cancel: the ack must be the LAST frame of r1 - zero further audio.
                ws.send_json({"type": "tts_cancel", "request_id": "r1"})
                ack = ws.receive_json()
                assert ack["type"] == "audio_end", ack
                assert ack["request_id"] == "r1"
                assert ack["reason"] == "cancelled"

                # The next request is served completely and byte-exactly: nothing
                # from r1 - not a PCM frame, not its terminal - may leak into r2.
                fake.proceed.set()  # r1's gate served its purpose; don't gate r2
                ws.send_json({"type": "tts", "text": "clean follow-up",
                              "request_id": "r2"})
                start2, pcm2, end2 = _drain_request(ws)
                assert start2["request_id"] == "r2"
                assert end2["request_id"] == "r2"
                assert pcm2 == FULL_REQUEST_PCM, \
                    "r2's audio must be exactly its own bytes (no r1 leftovers)"

        assert [r["text"] for r in fake.requests] == ["doomed", "clean follow-up"]
        assert len(fake.cancels) == 1, "the sidecar must see the CANCEL for r1"


# --- WS level: a superseding tts request gets the same guarantee ---------------------

def test_ws_supersede_cancels_in_flight_and_new_request_is_clean(monkeypatch) -> None:
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    with FakeTtsSidecar(gate_after_first=True) as fake:
        engine = _external_engine(fake)
        with ThreadPoolExecutor(max_workers=1) as executor:
            client = TestClient(_tts_app(engine, executor))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"

                ws.send_json({"type": "tts", "text": "superseded", "request_id": "r1"})
                assert ws.receive_json()["type"] == "audio_start"
                assert ws.receive().get("bytes"), "r1 must be streaming first"

                # A new tts request IS the cancel (the supersede path).
                ws.send_json({"type": "tts", "text": "the new request",
                              "request_id": "r2"})
                ack = ws.receive_json()
                assert ack["type"] == "audio_end" and ack["request_id"] == "r1"
                assert ack["reason"] == "cancelled"

                fake.proceed.set()  # r1's gate served its purpose; don't gate r2
                start2, pcm2, end2 = _drain_request(ws)
                assert start2["request_id"] == "r2"
                assert end2["request_id"] == "r2"
                assert pcm2 == FULL_REQUEST_PCM, \
                    "r2's audio must be exactly its own bytes (no r1 leftovers)"

        assert [r["text"] for r in fake.requests] == ["superseded", "the new request"]
        assert len(fake.cancels) == 1, "the sidecar must see the CANCEL for r1"
