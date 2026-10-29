"""The voxtream engine's streaming: one sidecar generation per request, model-free.

A fake sidecar (tests/voxtream_fake_sidecar.py, FakeVoxtreamSidecar) serves the wire
contract on a loopback WebSocket and records what it was sent. Covers: a tts_stream
request's chunks pass straight through in order, flush sends nothing, done sends eot,
the client gets one audio_start (24000, mono, 16 bit), the frames as int16 PCM and one
audio_end; text fed after audio has started is still spoken in the same generation;
the one-shot tts request is one generation with the whole text; voice selection reaches
the init frame and an unknown voice fails before any connection; and the pipeline is
wired per connection for the voxtream engine type.
"""
import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import authkit  # test auth env; must be imported before voicemanager
import numpy as np
import pytest
from fastapi.testclient import TestClient

from voicemanager.app import create_app
from voicemanager.pipeline.audiocpp_tts import UnknownVoiceError
from voicemanager.pipeline.tts import TTSPipeline, float32_to_pcm16
from voicemanager.pipeline.voxtream_tts import (
    VoxtreamSidecar, VoxtreamTTSEngine, VoxtreamTTSStreamPipeline)
from voicemanager.transport import websocket as ws_module
from voxtream_fake_sidecar import FRAME_SAMPLES, FakeVoxtreamSidecar

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_REFS = REPO_ROOT / "voices" / "designed" / "refs.json"
VOICES_DIR = REAL_REFS.parent
CHUNKS = ["Okay, so ", "for dark stains, ", "you can try ", "baking soda ", "and water."]


def _fake_frame(n: int) -> bytes:
    return float32_to_pcm16(np.full(FRAME_SAMPLES, n / 100.0, dtype=np.float32))


@pytest.fixture
def fake():
    sidecar = FakeVoxtreamSidecar().start()
    try:
        yield sidecar
    finally:
        sidecar.stop()


def _engine(fake: FakeVoxtreamSidecar, default_voice: str = "designed_female_s101"):
    engine = VoxtreamTTSEngine(VoxtreamSidecar(sys.executable, fake.port),
                               default_voice=default_voice)
    engine.load(voices_path=REAL_REFS, start_sidecar=False)
    return engine


def _pipe(engine, voice: str | None = None) -> VoxtreamTTSStreamPipeline:
    return VoxtreamTTSStreamPipeline(TTSPipeline(engine=engine, executor=None, voice=voice))


class _WS:
    def __init__(self):
        self.frames: list[tuple[str, object]] = []

    async def send_text(self, data: str) -> None:
        self.frames.append(("text", json.loads(data)))

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append(("bytes", data))

    def kinds(self) -> list[str]:
        return [kind if kind != "text" else p["type"] for kind, p in self.frames]

    def pcm(self) -> list[bytes]:
        return [p for kind, p in self.frames if kind == "bytes"]

    def texts(self, kind: str) -> list[dict]:
        return [p for k, p in self.frames if k == "text" and p["type"] == kind]


async def _until(predicate, timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting on the condition")
        await asyncio.sleep(0.005)


# --- pass-through ----------------------------------------------------------------

@pytest.mark.parametrize("chunking", ["sentence", "clause"])
def test_chunks_pass_straight_through_flush_is_silent_done_ends(fake, chunking):
    async def run():
        engine, ws = _engine(fake), _WS()
        pipe = _pipe(engine)
        rid = "r1"
        await pipe.feed_text(CHUNKS[0], False, ws, rid, chunking=chunking)
        await pipe.feed_text(CHUNKS[1], False, ws, rid, chunking=chunking)
        await pipe.feed_text("", False, ws, rid, flush=True, chunking=chunking)
        await pipe.feed_text(CHUNKS[2], False, ws, rid, chunking=chunking)
        await pipe.feed_text(CHUNKS[3], False, ws, rid, chunking=chunking)
        await pipe.feed_text(CHUNKS[4], True, ws, rid, chunking=chunking)
        await asyncio.wait_for(pipe.synthesis_task, 3.0)
        assert pipe.chunking == chunking
    asyncio.run(run())

    assert fake.connections == 1
    assert fake.inits[0]["full_stream"] is True
    assert fake.inits[0]["prompt_audio_path"] == str(VOICES_DIR / "designed_female_s101.wav")
    assert fake.chunks[0] == CHUNKS  # five text frames, in order, nothing for the flush
    assert fake.eots == [0]


def test_client_sees_one_start_the_frames_as_pcm_and_one_end(fake):
    ws = _WS()

    async def run():
        pipe = _pipe(_engine(fake))
        for i, chunk in enumerate(CHUNKS):
            await pipe.feed_text(chunk, i == len(CHUNKS) - 1, ws, "r1")
        await asyncio.wait_for(pipe.synthesis_task, 3.0)
    asyncio.run(run())

    starts = ws.texts("audio_start")
    assert len(starts) == 1
    assert (starts[0]["sample_rate"], starts[0]["channels"], starts[0]["bit_depth"],
            starts[0]["encoding"]) == (24000, 1, 16, "pcm")
    assert ws.pcm() == [_fake_frame(n) for n in range(1, len(CHUNKS) + 1)]
    assert len(ws.texts("audio_end")) == 1
    assert ws.kinds()[0] == "audio_start" and ws.kinds()[-1] == "audio_end"


def test_text_fed_after_audio_started_is_spoken_in_the_same_generation(fake):
    ws = _WS()

    async def run():
        pipe = _pipe(_engine(fake))
        await pipe.feed_text("first words ", False, ws, "r1")
        await pipe.feed_text("second words ", False, ws, "r1")
        await _until(lambda: len(ws.pcm()) == 2)  # the sidecar is already speaking
        assert ws.texts("audio_end") == []
        await pipe.feed_text("third words", False, ws, "r1")
        await _until(lambda: len(ws.pcm()) == 3)
        await pipe.feed_text("", True, ws, "r1")
        await asyncio.wait_for(pipe.synthesis_task, 3.0)
    asyncio.run(run())

    assert fake.connections == 1
    assert fake.chunks[0] == ["first words ", "second words ", "third words"]
    assert ws.kinds() == ["audio_start", "bytes", "bytes", "bytes", "audio_end"]


def test_late_frames_after_done_are_ignored(fake):
    async def run():
        ws = _WS()
        pipe = _pipe(_engine(fake))
        await pipe.feed_text("all of it", True, ws, "r1")
        await pipe.feed_text("too late", False, ws, "r1")
        await asyncio.wait_for(pipe.synthesis_task, 3.0)
        assert len(ws.texts("audio_end")) == 1
    asyncio.run(run())
    assert fake.chunks[0] == ["all of it"]


# --- one-shot --------------------------------------------------------------------

def test_one_shot_tts_is_one_generation_with_the_whole_text(fake):
    ws = _WS()
    text = "Two sentences. Here they are."

    async def run():
        engine = _engine(fake)
        frames = [f async for f in engine.stream_synthesize(text)]
        assert len(frames) == 1 and frames[0].dtype == np.float32
        pipeline = TTSPipeline(engine=engine, executor=None)
        await pipeline.synthesize_and_stream(ws, text, "one")
    asyncio.run(run())

    assert fake.connections == 2
    assert fake.chunks[0] == [text] and fake.chunks[1] == [text]
    assert fake.eots == [0, 1]
    assert ws.kinds() == ["audio_start", "bytes", "audio_end"]
    assert ws.texts("audio_start")[0]["sample_rate"] == 24000


def test_synthesize_collects_one_generation(fake):
    engine = _engine(fake)
    with ThreadPoolExecutor(max_workers=1) as pool:
        audio, sr = pool.submit(engine.synthesize, "Shields up.").result(timeout=5)
    assert sr == 24000
    assert len(audio) == FRAME_SAMPLES
    assert fake.chunks[0] == ["Shields up."]


# --- voice -----------------------------------------------------------------------

def test_voice_selection_reaches_init_and_defaults(fake):
    async def run():
        engine = _engine(fake)
        pipe = _pipe(engine)
        ws = _WS()
        await pipe.feed_text("one", True, ws, "a", voice="designed_gm_02")
        await asyncio.wait_for(pipe.synthesis_task, 3.0)
        await pipe.feed_text("two", True, ws, "b")
        await asyncio.wait_for(pipe.synthesis_task, 3.0)
        # The connection default (the token's voice) applies when the request names none.
        pipe_default = _pipe(engine, voice="designed_male_s202")
        await pipe_default.feed_text("three", True, ws, "c")
        await asyncio.wait_for(pipe_default.synthesis_task, 3.0)
    asyncio.run(run())

    paths = [init["prompt_audio_path"] for init in fake.inits]
    assert paths == [str(VOICES_DIR / "designed_gm_02.wav"),
                     str(VOICES_DIR / "designed_female_s101.wav"),
                     str(VOICES_DIR / "designed_male_s202.wav")]


def test_unknown_voice_fails_before_any_connection(fake):
    async def run():
        pipe = _pipe(_engine(fake))
        with pytest.raises(UnknownVoiceError) as exc:
            await pipe.feed_text("hello", False, _WS(), "r1", voice="nope")
        assert "designed_female_s101" in str(exc.value)
        assert pipe.synthesis_task is None
    asyncio.run(run())
    assert fake.connections == 0


# --- wired per connection ------------------------------------------------------------

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
    app.state.tts_engine_type = "voxtream"
    app.state.tts_engine = engine
    app.state.tts_executor = executor
    from voicemanager.session import SessionRegistry
    app.state.session_registry = SessionRegistry()
    return app


def test_tts_stream_over_the_websocket_endpoint_uses_the_pass_through(fake, monkeypatch):
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    engine = _engine(fake)
    with ThreadPoolExecutor(max_workers=1) as executor:
        client = TestClient(_tts_app(engine, executor))
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            for i, chunk in enumerate(CHUNKS):
                ws.send_json({"type": "tts_stream", "text": chunk,
                              "done": i == len(CHUNKS) - 1, "request_id": "s1",
                              "chunking": "clause"})
            start = ws.receive_json()
            assert start["type"] == "audio_start" and start["sample_rate"] == 24000
            pcm = []
            while True:
                frame = ws.receive()
                if frame.get("bytes") is not None:
                    pcm.append(frame["bytes"])
                    continue
                payload = json.loads(frame["text"])
                assert payload["type"] == "audio_end", payload
                break
    assert fake.connections == 1
    assert fake.chunks[0] == CHUNKS
    assert pcm == [_fake_frame(n) for n in range(1, len(CHUNKS) + 1)]
