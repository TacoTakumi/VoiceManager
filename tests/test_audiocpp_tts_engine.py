"""AudioCppTTSEngine core tests against the fake TTS sidecar.

Model-free and server-free: the engine dials the in-process FakeTtsSidecar
(tests/audiocpp_tts_fake.py) over a real AF_UNIX socket, so these exercise the
actual TTS wire framing end to end. What must hold:

  * every dispatched TEXT request carries the resolved preset's WAV path AND its
    non-empty ref_text
  * the voice kwarg routes presets via the preset loader; no voice -> the default;
    an unknown voice raises the defined error and dispatches NOTHING
  * chunks are yielded incrementally - the first chunk arrives before the fake
    finishes generating
  * sample count and values are preserved end to end at 48 kHz - no resample
    - and the flush tail is included before the stream ends
"""
from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from voicemanager.pipeline.audiocpp_tts import (
    BREEZE_TTS_MODEL_TASK,
    DEFAULT_VOICE,
    AudioCppTTSEngine,
    SidecarError,
    UnknownVoiceError,
)
from audiocpp_tts_fake import (
    DEFAULT_CHUNKS,
    DEFAULT_FLUSH,
    FakeTtsSidecar,
    s16le,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REFS_JSON = REPO_ROOT / "voices" / "designed" / "refs.json"


def _engine_for(fake: FakeTtsSidecar) -> AudioCppTTSEngine:
    engine = AudioCppTTSEngine({"socket": fake.socket_path})
    engine.load(voices_path=str(REFS_JSON))
    return engine


def _collect(engine: AudioCppTTSEngine, text: str, **kwargs) -> list[np.ndarray]:
    async def run() -> list[np.ndarray]:
        chunks = []
        async for chunk in engine.stream_synthesize(text, **kwargs):
            chunks.append(chunk)
        return chunks

    return asyncio.run(run())


def _expected_floats(payload: bytes) -> np.ndarray:
    values = struct.unpack(f"<{len(payload) // 2}h", payload)
    return np.asarray(values, dtype=np.float32) / 32768.0


# --- dispatch carries ref_text and the preset routing ---------------

def test_every_dispatch_carries_the_default_presets_wav_and_ref_text() -> None:
    with FakeTtsSidecar() as fake:
        engine = _engine_for(fake)
        try:
            _collect(engine, "Warp core is stable.")
        finally:
            engine.dispose()
        assert len(fake.requests) == 1
        request = fake.requests[0]
        assert request["text"] == "Warp core is stable."
        assert request["voice_wav"].endswith("designed_female_s101.wav")
        assert Path(request["voice_wav"]).is_file()
        assert request["ref_text"].strip(), "ref_text must be non-empty on EVERY dispatch"


def test_voice_kwarg_routes_each_preset() -> None:
    with FakeTtsSidecar() as fake:
        engine = _engine_for(fake)
        try:
            for voice in ("designed_male_s202", "lcars_demo", "designed_female_s101"):
                _collect(engine, f"as {voice}", voice=voice)
        finally:
            engine.dispose()
        assert [r["voice_wav"].rsplit("/", 1)[-1] for r in fake.requests] == [
            "designed_male_s202.wav", "lcars_demo.wav", "designed_female_s101.wav"]
        for request in fake.requests:
            assert request["ref_text"].strip()


def test_unknown_voice_raises_and_dispatches_nothing() -> None:
    with FakeTtsSidecar() as fake:
        engine = _engine_for(fake)
        try:
            with pytest.raises(UnknownVoiceError) as exc_info:
                _collect(engine, "hello", voice="no-such-voice")
        finally:
            engine.dispose()
        assert "designed_female_s101" in str(exc_info.value)
        assert fake.requests == [], "an unknown voice must never reach the sidecar"


# --- incremental streaming --------------------------------------------------

def test_first_chunk_yields_before_the_fake_finishes_generating() -> None:
    with FakeTtsSidecar(gate_after_first=True) as fake:
        engine = _engine_for(fake)

        async def run() -> list[np.ndarray]:
            chunks = []
            stream = engine.stream_synthesize("incremental check")
            first = await anext(stream)
            chunks.append(first)
            # The fake is still gated: its request frame sequence is not finished,
            # so this chunk was yielded strictly before generation completed.
            assert not fake.request_done.is_set(), \
                "first chunk must be yielded while the request is still in flight"
            fake.proceed.set()
            async for chunk in stream:
                chunks.append(chunk)
            return chunks

        try:
            chunks = asyncio.run(run())
        finally:
            engine.dispose()
        assert not fake.gate_timed_out
        assert len(chunks) == len(DEFAULT_CHUNKS) + 1  # + the flush tail


# --- 48 kHz pass-through, sample count preserved, flush tail included -------

def test_sample_count_and_values_preserved_no_resample() -> None:
    chunks = [s16le(*range(0, 400)), s16le(*range(-400, 0))]
    flush = s16le(9, -9, 9, -9)
    with FakeTtsSidecar(chunks=chunks, flush_chunk=flush) as fake:
        engine = _engine_for(fake)
        try:
            yielded = _collect(engine, "sample count check")
        finally:
            engine.dispose()
    sent = b"".join(chunks) + flush
    got = np.concatenate(yielded)
    assert got.dtype == np.float32
    assert len(got) == len(sent) // 2, "sample count must be preserved end to end"
    np.testing.assert_array_equal(got, _expected_floats(sent))
    assert engine.sample_rate == 48000


def test_engine_sample_rate_follows_a_24k_ready(monkeypatch) -> None:
    """The engine's declared rate is the
    READY-stated one, not the 48 kHz constant - a breeze sidecar says 24000 and
    every reply follows."""
    with FakeTtsSidecar(ready_sample_rate=24000) as fake:
        engine = AudioCppTTSEngine({"socket": fake.socket_path,
                                    "model_task": BREEZE_TTS_MODEL_TASK})
        engine.load(voices_path=str(REFS_JSON))
        try:
            assert engine.sample_rate == 24000, "before connect: the model_task default"
            engine.connect()
            assert engine.sample_rate == 24000
            audio, sample_rate = engine.synthesize("rate check")
        finally:
            engine.dispose()
    assert sample_rate == 24000
    sent = b"".join(DEFAULT_CHUNKS) + DEFAULT_FLUSH
    np.testing.assert_array_equal(audio, _expected_floats(sent))


def test_flush_tail_is_the_last_yielded_chunk() -> None:
    with FakeTtsSidecar() as fake:
        engine = _engine_for(fake)
        try:
            yielded = _collect(engine, "flush tail check")
        finally:
            engine.dispose()
    assert len(yielded) == len(DEFAULT_CHUNKS) + 1
    np.testing.assert_array_equal(yielded[-1], _expected_floats(DEFAULT_FLUSH))


def test_batch_synthesize_returns_the_full_array_at_48k() -> None:
    with FakeTtsSidecar() as fake:
        engine = _engine_for(fake)
        try:
            audio, sample_rate = engine.synthesize("batch check")
        finally:
            engine.dispose()
    sent = b"".join(DEFAULT_CHUNKS) + DEFAULT_FLUSH
    assert sample_rate == 48000
    np.testing.assert_array_equal(audio, _expected_floats(sent))


# --- sequential requests and error surfaces ------------------------------------------

def test_two_sequential_requests_on_one_connection() -> None:
    with FakeTtsSidecar() as fake:
        engine = _engine_for(fake)
        try:
            first = _collect(engine, "request one")
            second = _collect(engine, "request two", voice="designed_male_s202")
        finally:
            engine.dispose()
        assert len(fake.requests) == 2
        assert [r["text"] for r in fake.requests] == ["request one", "request two"]
        assert len(first) == len(second) == len(DEFAULT_CHUNKS) + 1


def test_sidecar_error_reply_raises_sidecar_error() -> None:
    with FakeTtsSidecar(text_error="internal") as fake:
        engine = _engine_for(fake)
        try:
            with pytest.raises(SidecarError):
                _collect(engine, "will fail")
        finally:
            engine.dispose()
        assert len(fake.requests) == 1


def test_voxcpm2_preset_without_ref_text_omits_the_ref_text_key(
        tmp_path: Path) -> None:
    # A voxcpm2 engine (model_task selects the per-model ref_text rule) loads
    # a refs.json whose default preset omits ref_text, and its dispatch omits the
    # ref_text key - the dots default requires it (pinned by the preset tests).
    wav = tmp_path / "v.wav"
    wav.write_bytes(b"RIFFfake")
    refs = tmp_path / "refs.json"
    refs.write_text(json.dumps({DEFAULT_VOICE: {"file": "v.wav"}}), encoding="utf-8")

    with FakeTtsSidecar() as fake:
        engine = AudioCppTTSEngine({"socket": fake.socket_path,
                                    "model_task": "voxcpm2-tts-streaming"})
        engine.load(voices_path=str(refs))
        try:
            _collect(engine, "Warp core is stable.")
        finally:
            engine.dispose()
        assert len(fake.requests) == 1
        request = fake.requests[0]
        assert request["text"] == "Warp core is stable."
        assert request["voice_wav"].endswith("v.wav")
        assert "ref_text" not in request, (
            "a ref_text-less voxcpm2 preset must OMIT the ref_text key")
