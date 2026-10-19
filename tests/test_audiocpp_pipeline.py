"""Contract tests for AudioCppSTTPipeline over a fake sidecar.

Model-free and server-free (CLAUDE.md invariant): the pipeline drives a real AF_UNIX
socket speaking sidecar/PROTOCOL.md via tests/audiocpp_fake.py -- no GPU, no native
sidecar binary, no model. These assert each of the six methods' contract and that
feed() is offloaded off the event loop (a slow fake does not stall a concurrent task).
The end-to-end /ws/stream frame pattern and the crash path have their own tests; a real
GGML crash is a live check.
"""
import asyncio
import time

import authkit  # noqa: F401  (test auth env; must precede voicemanager imports)

import pytest

from voicemanager.pipeline.audiocpp_stt import (
    AudioCppSTTPipeline,
    SidecarError,
    SidecarFault,
)

from audiocpp_fake import FakeSidecar

# 512 samples of nonzero 16-bit PCM (content is irrelevant to the fake; forwarded raw).
_PCM = b"\x11\x22" * 512


def _config(fake: FakeSidecar) -> dict:
    """Point the pipeline at the fake's socket with NO binary -> never spawn (connect only)."""
    return {
        "sidecar_bin": "",                 # empty -> connect to the already-listening fake
        "socket": fake.socket_path,
        "model": "",
        "model_task": "nemotron-3.5-asr-streaming",
        "backend": "cpu",
        "device": 0,
        "threads": 4,
        "assets_root": "",
    }


def _feed_until(pipeline, predicate, *, tries: int = 40, gap: float = 0.02):
    """Feed frames until `predicate()` is true (cross-thread delivery is not instant).

    The sidecar emits EVENT/FINAL frames from its own thread, and feed()'s reply drain
    is non-blocking, so a frame sent for AUDIO #N is typically picked up on drain #N+1.
    Polling a bounded number of frames makes that deterministic without flakiness.
    """
    for _ in range(tries):
        asyncio.run(pipeline.feed(_PCM))
        if predicate():
            return True
        time.sleep(gap)
    return False


def test_feed_forwards_audio_without_partial_for_flush_only_model():
    # A flush-only model (the Nemotron-3.5 of the earlier pins) emits nothing on AUDIO
    # (deltas arrive at FLUSH), so feed() reports no interim and the sidecar still
    # receives the raw PCM frames.
    with FakeSidecar(audio_events=[], flush_final=None) as fake:
        p = AudioCppSTTPipeline(config=_config(fake))
        assert asyncio.run(p.feed(_PCM)) is False
        assert asyncio.run(p.feed(_PCM)) is False
        assert p.get_current_text() == ("", "", 0)
        # give the fake thread a beat to count the frames it received
        time.sleep(0.05)
        assert fake.connections and fake.connections[0].audio_frames >= 2
        assert fake.connections[0].hello["model_task"] == "nemotron-3.5-asr-streaming"
        p.dispose()


def test_feed_accumulates_delta_events_into_growing_partial():
    # audio.cpp emits partial_text as token/char DELTAS (PROTOCOL.md); the pipeline
    # accumulates them into the utterance transcript-so-far, so every partial carries
    # the whole accumulation (replace semantics), never the last fragment alone
    # (the last-fragment 'te' bug).
    with FakeSidecar(audio_delta_script=["long ", "range ", "scan"],
                     flush_final=None) as fake:
        p = AudioCppSTTPipeline(config=_config(fake))
        observed: list[str] = []

        def accumulated() -> bool:
            text = p.get_current_text()[0]
            if text and (not observed or text != observed[-1]):
                observed.append(text)
            return text == "long range scan"

        assert _feed_until(p, accumulated), f"accumulation never completed: {observed}"
        # Every observed partial is the transcript-so-far: a strictly growing prefix
        # of the full accumulation.
        assert all("long range scan".startswith(t) for t in observed), observed
        for earlier, later in zip(observed, observed[1:]):
            assert later.startswith(earlier) and len(later) > len(earlier), observed
        assert p.get_current_text() == ("long range scan", "en", 0)
        # reset() clears the accumulator for the next utterance.
        p.reset()
        assert p.get_current_text() == ("", "en", 1)
        p.dispose()


def test_final_marker_event_yields_no_partial_and_preserves_accumulation():
    # The sidecar forwards the model's real is_final on EVENT frames; the
    # client side must treat a final-flagged event as a MARKER: it never surfaces as
    # a partial and never clobbers the accumulated transcript. Both
    # marker shapes are covered -- a text-carrying final event and a contentless
    # end-of-stream marker.
    script = [
        "long ",
        "range",
        {"is_final": True, "partial_text": {"text": " scan", "language": "en"}},
        {"is_final": True},
    ]
    with FakeSidecar(audio_delta_script=script, flush_final=None) as fake:
        p = AudioCppSTTPipeline(config=_config(fake))
        assert _feed_until(p, lambda: p.get_current_text()[0] == "long range")
        # Drain well past both marker events: no feed() may report a new partial,
        # and the live accumulation must survive untouched.
        for _ in range(10):
            assert asyncio.run(p.feed(_PCM)) is False
            time.sleep(0.02)
        assert p.get_current_text() == ("long range", "en", 0)
        p.dispose()


def test_volunteered_final_clears_the_accumulator():
    # A FINAL commits the utterance, so the accumulator clears the moment it is
    # absorbed -- deltas of the NEXT utterance must never append to committed text,
    # even before the drive loop gets around to reset().
    with FakeSidecar(audio_delta_script=["long range scan"],
                     audio_final="long range scan", flush_final=None) as fake:
        p = AudioCppSTTPipeline(config=_config(fake))
        finals = []
        for _ in range(40):
            asyncio.run(p.feed(_PCM))
            if (f := p.take_final()) is not None:
                finals.append(f)
                break
            time.sleep(0.02)
        assert finals == [("long range scan", "en", 0)]
        assert p.get_current_text()[0] == ""
        p.dispose()


def test_finalize_burst_honors_is_final_and_accumulates_deltas():
    # The FLUSH burst is read by finalize(), not feed(): its deltas must accumulate
    # identically, and a final-flagged event inside the burst must not clobber the
    # accumulation -- the is_final mapping holds on both receive paths.
    burst = ["long ", "range", {"is_final": True, "partial_text": {"text": " NOPE"}}]
    with FakeSidecar(flush_events=burst, flush_final="long range complete") as fake:
        p = AudioCppSTTPipeline(config=_config(fake))
        asyncio.run(p.feed(_PCM))
        assert asyncio.run(p.finalize()) == ("long range complete", "en", 0)
        # The accumulated burst stays live for the WS layer's pre-final partial
        # (cleared at the reset() every finalize caller runs at the commit boundary).
        assert p.get_current_text()[0] == "long range"
        p.dispose()


def test_finalize_flushes_and_returns_the_final():
    with FakeSidecar(flush_events=["long range scan"],
                     flush_final="long range scan of planet complete") as fake:
        p = AudioCppSTTPipeline(config=_config(fake))
        asyncio.run(p.feed(_PCM))
        final = asyncio.run(p.finalize())
        assert final == ("long range scan of planet complete", "en", 0)
        # exactly one FLUSH produced exactly one FINAL; burst EVENT updated the partial
        time.sleep(0.05)
        assert fake.connections[0].flush_count == 1
        assert p.get_current_text()[0] == "long range scan"
        p.dispose()


def test_finalize_is_idempotent():
    with FakeSidecar(flush_final="done") as fake:
        p = AudioCppSTTPipeline(config=_config(fake))
        asyncio.run(p.feed(_PCM))
        assert asyncio.run(p.finalize()) == ("done", "en", 0)
        # A second finalize (teardown may run twice) must not re-FLUSH.
        assert asyncio.run(p.finalize()) == ("", "en", 0)
        time.sleep(0.05)
        assert fake.connections[0].flush_count == 1
        p.dispose()


def test_finalize_without_audio_returns_empty_without_connecting():
    # Teardown after zero audio: never connected, so nothing to flush and no socket op.
    with FakeSidecar() as fake:
        p = AudioCppSTTPipeline(config=_config(fake))
        assert asyncio.run(p.finalize()) == ("", "", 0)
        assert fake.connections == []  # never dialed
        p.dispose()


def test_take_final_returns_one_final_and_then_none():
    with FakeSidecar(audio_final="done", flush_final=None) as fake:
        p = AudioCppSTTPipeline(config=_config(fake))
        finals = []
        for _ in range(40):
            asyncio.run(p.feed(_PCM))
            f = p.take_final()
            if f is not None:
                finals.append(f)
            time.sleep(0.02)
            if finals:
                break
        assert finals == [("done", "en", 0)]
        assert p.take_final() is None  # the volunteered final fired once -> drained
        p.dispose()


def test_reset_advances_segment_and_defers_the_reset_frame():
    with FakeSidecar(flush_final="second") as fake:
        p = AudioCppSTTPipeline(config=_config(fake))
        asyncio.run(p.feed(_PCM))                 # connect + AUDIO #1
        time.sleep(0.05)
        assert fake.connections[0].reset_count == 0   # reset() not called yet

        p.reset()                                 # sync: advances segment, DEFERS the RESET
        assert p.get_current_text()[2] == 1       # segment advanced immediately
        assert fake.connections[0].reset_count == 0   # no socket round-trip on the loop

        asyncio.run(p.feed(_PCM))                 # next off-loop op flushes the deferred RESET
        time.sleep(0.05)
        assert fake.connections[0].reset_count == 1
        assert fake.connections[0].audio_frames == 2

        # A final after the reset carries the advanced segment index.
        assert asyncio.run(p.finalize()) == ("second", "en", 1)
        p.dispose()


def test_dispose_closes_the_connection():
    with FakeSidecar(flush_final="x") as fake:
        p = AudioCppSTTPipeline(config=_config(fake))
        asyncio.run(p.feed(_PCM))
        assert p._client.is_connected is True     # noqa: SLF001
        p.dispose()
        assert p._client.is_connected is False     # noqa: SLF001
        p.dispose()                                # idempotent, no raise


def test_feed_offloads_and_does_not_stall_the_event_loop():
    # A slow sidecar (READY delayed 0.3s) blocks feed()'s worker thread, NOT the loop:
    # a concurrent ticker must keep advancing while the first feed's connect is in flight.
    with FakeSidecar(ready_delay=0.3, flush_final=None) as fake:
        p = AudioCppSTTPipeline(config=_config(fake))
        ticks = asyncio.run(_offload_probe(p))
        assert ticks >= 3, f"loop appears stalled during feed(): only {ticks} ticks"
        p.dispose()


async def _offload_probe(pipeline) -> int:
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        for _ in range(100):
            ticks += 1
            await asyncio.sleep(0.01)

    tick_task = asyncio.create_task(ticker())
    await pipeline.feed(_PCM)     # ~0.3s in a worker thread; the loop runs ticker meanwhile
    tick_task.cancel()
    return ticks


def test_handshake_error_raises_sidecar_error():
    with FakeSidecar(hello_error="unknown_model_task") as fake:
        p = AudioCppSTTPipeline(config=_config(fake))
        with pytest.raises(SidecarError) as exc:
            asyncio.run(p.feed(_PCM))
        assert exc.value.code == "unknown_model_task"
        p.dispose()


def test_unexpected_sidecar_death_raises_fault_not_a_hang():
    # Abrupt EOF (no BYE) is the process-fatal-fault shape; the client must
    # surface a defined SidecarFault within a bound rather than hang. Full WS handling:
    # test_audiocpp_crash.py.
    with FakeSidecar(die_on_audio=True) as fake:
        p = AudioCppSTTPipeline(config=_config(fake))
        with pytest.raises(SidecarFault):
            for _ in range(40):
                asyncio.run(p.feed(_PCM))
                time.sleep(0.02)
        p.dispose()
