"""Composed Breeze reference for the later clause pieces of a tts_stream request.

Breeze re-conditions the speaker from the reference clip on every sidecar
request, and audio.cpp continues a voice only inside one request, so the seam
between the first clause piece of a reply and the next one is audible. With
VM_AUDIOCPP_TTS_CLAUSE_CONTINUE on, the audiocpp stream pipeline composes the
reference itself for every engine request after the first one of a clause
request: a temporary WAV holding the preset clip followed by the previous
piece's own audio (its tail, capped at VM_AUDIOCPP_TTS_CLAUSE_TAIL_MS), and a
ref_text of the preset transcript followed by the previous piece's text.

What must hold:

  * the first engine request of a clause request clones from the preset
    unchanged; each later one clones from the composed reference;
  * the previous piece's audio is capped at the tail length (keep the end);
  * a preset at another sample rate is resampled to the engine rate first;
  * with the setting off, or on a sentence request, every engine request
    carries the preset path and transcript exactly as before;
  * the composed files belong to the request and are gone once it ends,
    whichever way it ends: done, cancel, the flush timeout, or disconnect;
  * each clause engine request logs its reference kind and its first-audio
    latency at INFO; a sentence request logs neither.

Model-free and server-free: the fake TTS sidecar (tests/audiocpp_tts_fake.py)
answers over a real AF_UNIX socket, a fake WebSocket records the frames, and
the preset is a small WAV written to tmp_path. No GPU, no binary, no model.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from pathlib import Path

import authkit  # noqa: F401  test auth env; must be imported before voicemanager

import numpy as np
import pytest
import soundfile as sf

from voicemanager.pipeline import audiocpp_tts
from voicemanager.pipeline.audiocpp_tts import (
    AudioCppTTSEngine,
    AudioCppTTSStreamPipeline,
)
from voicemanager.pipeline.tts import TTSPipeline

from audiocpp_tts_fake import FakeTtsSidecar, s16le

ENGINE_RATE = 24000
TRANSCRIPT = "This is the reference transcript."
FIRST = "One two"
SECOND = "three four"

# Two long sentences for the sentence-mode case; the first feed completes S1
# and leaves a fragment, the second completes the rest on done.
S1 = "The quick brown fox jumps over the lazy dog near the river bank every single morning."
FEED1 = S1 + " Meanwhile the careful gray owl watches"
FEED2 = " quietly from the tallest branch of the old oak tree."

_NO_TIMER = {"clause_first_words": 2, "clause_first_deadline_ms": 60_000}


# --- helpers ------------------------------------------------------------------------

def _piece_samples(n: int) -> np.ndarray:
    """A deterministic, non-repeating int16 pattern of n samples."""
    return ((np.arange(n) * 37) % 20001 - 10000).astype(np.int16)


def _preset_samples(n: int) -> np.ndarray:
    return ((np.arange(n) * 11) % 8001 - 4000).astype(np.int16)


def _chunks_for(samples: np.ndarray, size: int = 1000) -> list[bytes]:
    return [s16le(*samples[i:i + size].tolist()) for i in range(0, len(samples), size)]


def _write_refs(tmp_path: Path, rate: int, n_samples: int) -> tuple[Path, np.ndarray]:
    """Write a one-preset refs.json and its WAV; return the manifest path and samples."""
    samples = _preset_samples(n_samples)
    sf.write(str(tmp_path / "ref.wav"), samples, rate, subtype="PCM_16")
    refs = tmp_path / "refs.json"
    refs.write_text(json.dumps({
        audiocpp_tts.DEFAULT_VOICE: {"file": "ref.wav", "ref_text": TRANSCRIPT},
    }), encoding="utf-8")
    return refs, samples


class _SnapshotSidecar(FakeTtsSidecar):
    """The fake sidecar, plus a copy of each request's voice_wav taken on arrival.

    The pipeline removes a composed WAV once its request ends, so the test must
    read it while the request is open: at the moment the sidecar would.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_wavs: list[tuple[np.ndarray, int] | None] = []

    def _answer_request(self, conn) -> None:
        path = self.requests[-1].get("voice_wav", "")
        try:
            data, rate = sf.read(path, dtype="int16", always_2d=True)
            self.ref_wavs.append((data[:, 0].copy(), int(rate)))
        except (OSError, RuntimeError):
            self.ref_wavs.append(None)
        super()._answer_request(conn)


class _WS:
    def __init__(self):
        self.frames: list[tuple[str, object]] = []

    async def send_text(self, data: str) -> None:
        self.frames.append(("text", json.loads(data)))

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append(("bytes", data))


def _engine(fake: FakeTtsSidecar, refs: Path) -> AudioCppTTSEngine:
    engine = AudioCppTTSEngine({"socket": fake.socket_path,
                                "model_task": audiocpp_tts.BREEZE_TTS_MODEL_TASK})
    engine.load(voices_path=str(refs))
    return engine


def _pipe(engine: AudioCppTTSEngine, **kwargs) -> AudioCppTTSStreamPipeline:
    kwargs = {**_NO_TIMER, "clause_continue": True, **kwargs}
    return AudioCppTTSStreamPipeline(TTSPipeline(engine=engine, executor=None), **kwargs)


async def _until(predicate, timeout: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting on the condition")
        await asyncio.sleep(0.005)


async def _first_piece(fake: FakeTtsSidecar, pipe, ws, chunking: str = "clause") -> None:
    """Open request r1 and wait until its first engine request has been answered."""
    text = FIRST if chunking == "clause" else FEED1
    await pipe.feed_text(text, False, ws, "r1", chunking=chunking)
    await _until(lambda: len(fake.requests) == 1)
    await _until(lambda: fake.request_done.is_set())


async def _second_piece(fake: FakeTtsSidecar, pipe, ws, *, done: bool,
                        chunking: str = "clause") -> None:
    """Send the second piece; with done=False it goes as a flush, request open."""
    text = SECOND if chunking == "clause" else FEED2
    if done:
        await pipe.feed_text(" " + text, True, ws, "r1")
        await asyncio.wait_for(pipe.synthesis_task, 5.0)
    else:
        await pipe.feed_text(" " + text, False, ws, "r1", flush=True)
        await _until(lambda: len(fake.requests) == 2)
        await _until(lambda: fake.request_done.is_set())


def _composed_files() -> list[str]:
    folder = audiocpp_tts.clause_ref_dir()
    return sorted(os.listdir(folder)) if os.path.isdir(folder) else []


# --- the composed reference ---------------------------------------------------------

def test_the_second_piece_clones_from_the_preset_plus_the_first_piece(tmp_path):
    refs, preset = _write_refs(tmp_path, ENGINE_RATE, 2400)
    piece = _piece_samples(3000)
    with _SnapshotSidecar(chunks=_chunks_for(piece), flush_chunk=None,
                          ready_sample_rate=ENGINE_RATE) as fake:
        engine = _engine(fake, refs)

        async def run():
            pipe, ws = _pipe(engine), _WS()
            await _first_piece(fake, pipe, ws)
            await _second_piece(fake, pipe, ws, done=True)

        try:
            asyncio.run(run())
        finally:
            engine.dispose()

    assert [r["text"] for r in fake.requests] == [FIRST, SECOND]
    first, second = fake.requests
    assert first["voice_wav"] == engine.resolve_voice(None).wav_path
    assert first["ref_text"] == TRANSCRIPT

    assert second["voice_wav"] != first["voice_wav"]
    assert second["voice_wav"].endswith(".wav")
    assert second["ref_text"] == TRANSCRIPT + " " + FIRST
    composed, rate = fake.ref_wavs[1]
    assert rate == ENGINE_RATE
    expected = np.concatenate([preset, piece])
    assert len(composed) == len(expected)
    assert np.max(np.abs(composed.astype(np.int32) - expected.astype(np.int32))) <= 1


def test_the_previous_piece_is_capped_at_its_tail(tmp_path):
    refs, preset = _write_refs(tmp_path, ENGINE_RATE, 2400)
    piece = _piece_samples(3000)
    tail_ms = 50  # 1200 samples at 24 kHz, shorter than the 3000-sample piece
    with _SnapshotSidecar(chunks=_chunks_for(piece), flush_chunk=None,
                          ready_sample_rate=ENGINE_RATE) as fake:
        engine = _engine(fake, refs)

        async def run():
            pipe, ws = _pipe(engine, clause_tail_ms=tail_ms), _WS()
            await _first_piece(fake, pipe, ws)
            await _second_piece(fake, pipe, ws, done=True)

        try:
            asyncio.run(run())
        finally:
            engine.dispose()

    composed, rate = fake.ref_wavs[1]
    cap = ENGINE_RATE * tail_ms // 1000
    expected = np.concatenate([preset, piece[-cap:]])
    assert rate == ENGINE_RATE
    assert len(composed) == len(expected)
    assert np.max(np.abs(composed.astype(np.int32) - expected.astype(np.int32))) <= 1


def test_a_preset_at_half_the_engine_rate_is_resampled_first(tmp_path):
    refs, preset = _write_refs(tmp_path, ENGINE_RATE // 2, 1200)
    piece = _piece_samples(3000)
    with _SnapshotSidecar(chunks=_chunks_for(piece), flush_chunk=None,
                          ready_sample_rate=ENGINE_RATE) as fake:
        engine = _engine(fake, refs)

        async def run():
            pipe, ws = _pipe(engine), _WS()
            await _first_piece(fake, pipe, ws)
            await _second_piece(fake, pipe, ws, done=True)

        try:
            asyncio.run(run())
        finally:
            engine.dispose()

    composed, rate = fake.ref_wavs[1]
    assert rate == ENGINE_RATE
    assert len(composed) == 2 * len(preset) + len(piece)
    # The piece rides through untouched after the resampled preset.
    tail = composed[-len(piece):]
    assert np.max(np.abs(tail.astype(np.int32) - piece.astype(np.int32))) <= 1


# --- the preset stays when the feature is off, or on a sentence request --------------

def test_continue_off_keeps_the_preset_on_every_clause_request(tmp_path):
    refs, _ = _write_refs(tmp_path, ENGINE_RATE, 2400)
    with _SnapshotSidecar(ready_sample_rate=ENGINE_RATE) as fake:
        engine = _engine(fake, refs)

        async def run():
            pipe, ws = _pipe(engine, clause_continue=False), _WS()
            await _first_piece(fake, pipe, ws)
            await _second_piece(fake, pipe, ws, done=True)

        try:
            asyncio.run(run())
        finally:
            engine.dispose()

    preset_path = engine.resolve_voice(None).wav_path
    assert [r["text"] for r in fake.requests] == [FIRST, SECOND]
    assert [r["voice_wav"] for r in fake.requests] == [preset_path, preset_path]
    assert [r["ref_text"] for r in fake.requests] == [TRANSCRIPT, TRANSCRIPT]
    assert _composed_files() == []


def test_a_sentence_request_keeps_the_preset_with_continue_on(tmp_path):
    refs, _ = _write_refs(tmp_path, ENGINE_RATE, 2400)
    with _SnapshotSidecar(ready_sample_rate=ENGINE_RATE) as fake:
        engine = _engine(fake, refs)

        async def run():
            pipe, ws = _pipe(engine), _WS()
            await _first_piece(fake, pipe, ws, chunking="sentence")
            await _second_piece(fake, pipe, ws, done=True, chunking="sentence")

        try:
            asyncio.run(run())
        finally:
            engine.dispose()

    preset_path = engine.resolve_voice(None).wav_path
    assert len(fake.requests) == 2
    assert fake.requests[0]["text"] == S1
    assert [r["voice_wav"] for r in fake.requests] == [preset_path, preset_path]
    assert [r["ref_text"] for r in fake.requests] == [TRANSCRIPT, TRANSCRIPT]
    assert _composed_files() == []


# --- the composed files go with the request ------------------------------------------

@pytest.mark.parametrize("ending", ["done", "cancel", "flush_timeout", "dispose", "abandon"])
def test_the_composed_files_are_removed_when_the_request_ends(tmp_path, ending):
    refs, _ = _write_refs(tmp_path, ENGINE_RATE, 2400)
    with _SnapshotSidecar(ready_sample_rate=ENGINE_RATE) as fake:
        engine = _engine(fake, refs)

        async def run():
            pipe, ws = _pipe(engine), _WS()
            await _first_piece(fake, pipe, ws)
            if ending == "done":
                await _second_piece(fake, pipe, ws, done=True)
                return
            await _second_piece(fake, pipe, ws, done=False)
            composed_path = fake.requests[1]["voice_wav"]
            assert os.path.exists(composed_path), \
                "the composed WAV must live while its request is open"
            task = pipe.synthesis_task
            if ending == "cancel":
                await pipe.cancel()
            elif ending == "flush_timeout":
                await pipe.flush_timeout(ws, "r1")
                await asyncio.wait_for(task, 5.0)
            elif ending == "dispose":
                pipe.dispose()
            else:
                pipe.abandon()
            try:
                await task
            except asyncio.CancelledError:
                pass

        try:
            asyncio.run(run())
        finally:
            engine.dispose()

    assert fake.ref_wavs[1] is not None, "the sidecar must have been able to read it"
    composed_path = fake.requests[1]["voice_wav"]
    assert composed_path != fake.requests[0]["voice_wav"]
    assert not os.path.exists(composed_path), f"{ending}: the composed WAV was left behind"
    assert _composed_files() == [], f"{ending}: {_composed_files()}"


def test_a_cancel_during_the_write_leaves_no_composed_file(tmp_path, monkeypatch):
    """Cancel lands while the worker thread is still writing the composed WAV.

    The task unwinds at once and the request's cleanup unlinks the paths, but the
    thread goes on and creates the file after that. The file must still be gone
    once the thread has finished.
    """
    writing = threading.Event()
    release = threading.Event()
    real_write = audiocpp_tts.write_composed_reference

    def held_write(*args):
        writing.set()
        assert release.wait(5.0), "the test never released the write"
        real_write(*args)

    monkeypatch.setattr(audiocpp_tts, "write_composed_reference", held_write)

    preset = audiocpp_tts.PresetVoice(name="p", wav_path=str(tmp_path / "unused.wav"),
                                      ref_text=TRANSCRIPT)
    state = audiocpp_tts._ClauseReference(preset)
    state.preset_clip = _preset_samples(240)
    state.previous_tail = [_piece_samples(240)]
    state.cap = 120
    pipe = _pipe(object())
    minted: list[str] = []

    async def run():
        loop = asyncio.get_running_loop()
        task = asyncio.create_task(pipe._write_clause_reference(state, loop, ENGINE_RATE))
        await _until(writing.is_set)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # What the request's own cleanup does on the cancellation unwind.
        minted.extend(state.paths)
        for path in state.paths:
            audiocpp_tts._remove_quietly(path)
        state.paths.clear()
        release.set()

    # asyncio.run shuts the default executor down, so the worker thread has
    # finished by the time it returns.
    asyncio.run(run())

    assert minted, "the call must have registered its path"
    for path in minted:
        assert not os.path.exists(path), f"the composed WAV was left behind: {path}"


# --- logging --------------------------------------------------------------------------

def _info_lines(caplog, marker: str) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.INFO and marker in r.getMessage()]


def test_each_clause_engine_request_logs_its_reference_and_first_audio(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    refs, _ = _write_refs(tmp_path, ENGINE_RATE, 2400)
    with _SnapshotSidecar(ready_sample_rate=ENGINE_RATE) as fake:
        engine = _engine(fake, refs)

        async def run():
            pipe, ws = _pipe(engine), _WS()
            await _first_piece(fake, pipe, ws)
            await _second_piece(fake, pipe, ws, done=True)

        try:
            asyncio.run(run())
        finally:
            engine.dispose()

    request_lines = _info_lines(caplog, "Clause TTS engine request")
    assert len(request_lines) == 2
    assert "request_id=r1" in request_lines[0] and "ref=preset" in request_lines[0]
    assert "request_id=r1" in request_lines[1] and "ref=composed" in request_lines[1]
    first_audio = _info_lines(caplog, "first_audio_ms=")
    assert len(first_audio) == 2
    for line in first_audio:
        assert "request_id=r1" in line
        value = line.split("first_audio_ms=", 1)[1].split()[0]
        assert value.isdigit()


def test_a_sentence_request_logs_neither_line(tmp_path, caplog):
    caplog.set_level(logging.INFO)
    refs, _ = _write_refs(tmp_path, ENGINE_RATE, 2400)
    with _SnapshotSidecar(ready_sample_rate=ENGINE_RATE) as fake:
        engine = _engine(fake, refs)

        async def run():
            pipe, ws = _pipe(engine), _WS()
            await _first_piece(fake, pipe, ws, chunking="sentence")
            await _second_piece(fake, pipe, ws, done=True, chunking="sentence")

        try:
            asyncio.run(run())
        finally:
            engine.dispose()

    assert len(fake.requests) == 2
    assert _info_lines(caplog, "Clause TTS engine request") == []
    assert _info_lines(caplog, "first_audio_ms=") == []


# --- the settings reach the pipeline -------------------------------------------------

def test_the_pipeline_defaults_come_from_settings(monkeypatch):
    import voicemanager.config as cfg

    monkeypatch.setenv("VM_AUDIOCPP_TTS_CLAUSE_CONTINUE", "1")
    monkeypatch.setenv("VM_AUDIOCPP_TTS_CLAUSE_TAIL_MS", "1234")
    cfg._settings = None
    try:
        pipe = AudioCppTTSStreamPipeline(TTSPipeline(engine=object(), executor=None))
        assert pipe.clause_continue is True
        assert pipe.clause_tail_ms == 1234
        explicit = AudioCppTTSStreamPipeline(TTSPipeline(engine=object(), executor=None),
                                             clause_continue=False, clause_tail_ms=10)
        assert explicit.clause_continue is False
        assert explicit.clause_tail_ms == 10
    finally:
        cfg._settings = None


def test_the_settings_have_their_defaults_and_env_names():
    from voicemanager.config import Settings

    settings = Settings(_env_file=None)
    assert settings.audiocpp_tts_clause_continue is False
    assert settings.audiocpp_tts_clause_tail_ms == 4000
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    assert "# VM_AUDIOCPP_TTS_CLAUSE_CONTINUE=0" in text
    assert "# VM_AUDIOCPP_TTS_CLAUSE_TAIL_MS=4000" in text
