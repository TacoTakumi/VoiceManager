"""Shared doubles for the turn-engine WebSocket tests, model-free.

The STT side is a scripted stand-in for the audiocpp pipeline (its frames do not
depend on timing), the TTS side a fake engine that yields known float32 chunks,
and the sidecar the in-process FakeTygerSidecar behind a small stand-in for
app.state.turn_engine_sidecar. No GPU, no model, no live server.

Not a test module: pytest collects nothing here.
"""
import json
import urllib.parse

import numpy as np

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)

from voicemanager.app import create_app
from voicemanager.session import SessionRegistry

#: The rate the fake TTS engine speaks at, as today's engines do.
TTS_RATE = 24000
OPT_IN = {"type": "turn_engine", "engine": "tyger"}
PARTIAL = "scanning"
FINAL = "scan complete"


class FakeTTSEngine:
    """Yields the given float32 chunks, in order, for every synthesis call."""

    def __init__(self, chunks: list[np.ndarray] | None = None, rate: int = TTS_RATE) -> None:
        self.chunks = chunks if chunks is not None else [np.zeros(8, dtype=np.float32)]
        self._rate = rate

    @property
    def sample_rate(self) -> int:
        return self._rate

    async def stream_synthesize(self, text: str, **kwargs):
        for chunk in self.chunks:
            yield chunk


class StandInSidecar:
    """What the handler reads from app.state.turn_engine_sidecar: up and session_url."""

    def __init__(self, url: str, up: bool = True) -> None:
        self.url = url
        self.up = up

    def session_url(self, label: str) -> str:
        return f"{self.url}?{urllib.parse.urlencode({'session': label})}"


class ScriptedAudioCppSTT:
    """A stand-in for AudioCppSTTPipeline: each fed frame gives one interim and one
    final on its own segment, whatever the timing."""

    def __init__(self, config=None, **kwargs):
        self._segment = 0
        self._final_pending = False

    async def feed(self, data: bytes) -> bool:
        self._final_pending = True
        return True

    def get_current_text(self):
        return (PARTIAL, "en", self._segment)

    def take_final(self):
        if not self._final_pending:
            return None
        self._final_pending = False
        return (FINAL, "en", self._segment)

    async def finalize(self):
        return ("", "en", self._segment)

    def reset(self) -> None:
        self._segment += 1

    def dispose(self) -> None:
        pass


def make_app(*, sidecar=None, tts_engine=None, executor=None):
    """The real app on the audiocpp engine with a fake TTS engine.

    Patch voicemanager.pipeline.audiocpp_stt.AudioCppSTTPipeline with
    ScriptedAudioCppSTT before connecting; the audiocpp config is never read then.
    """
    app = create_app()
    app.state.stt_engine_type = "audiocpp"
    app.state.audiocpp_config = {}
    app.state.session_registry = SessionRegistry()
    if executor is not None:
        app.state.tts_engine_type = "openvoice"
        app.state.tts_engine = tts_engine if tts_engine is not None else FakeTTSEngine()
        app.state.tts_executor = executor
    app.state.turn_engine_sidecar = sidecar
    return app


def frame(raw: dict):
    """A received frame: the JSON of a text frame, else ('pcm', bytes)."""
    if raw.get("bytes") is not None:
        return ("pcm", raw["bytes"])
    return json.loads(raw["text"])


def recv_until(ws, done, limit: int = 200) -> list:
    """Receive frames until done(frame) is true for one; all of them, in order."""
    frames = []
    for _ in range(limit):
        f = frame(ws.receive())
        frames.append(f)
        if done(f):
            return frames
    raise AssertionError(f"no closing frame in {frames}")


def is_type(kind: str):
    return lambda f: isinstance(f, dict) and f.get("type") == kind


def opt_in(ws) -> dict:
    """Read the ready frame, opt in and read turn_engine_ready; the ready frame."""
    ready = ws.receive_json()
    assert ready["type"] == "ready"
    ws.send_json(OPT_IN)
    assert ws.receive_json() == {"type": "turn_engine_ready", "engine": "tyger"}
    return ready


def speak(ws, request_id: str, text: str = "Hello there.") -> tuple[dict, bytes]:
    """Send one tts request and read it to its audio_end: (audio_start, the PCM sent)."""
    ws.send_json({"type": "tts", "request_id": request_id, "text": text})
    frames = recv_until(ws, is_type("audio_end"))
    starts = [f for f in frames if isinstance(f, dict) and f.get("type") == "audio_start"]
    assert len(starts) == 1, frames
    pcm = b"".join(f[1] for f in frames if isinstance(f, tuple))
    return starts[0], pcm


def barrier(ws) -> list:
    """Send turn_start and read up to its turn_started: every frame before it.

    The handler answers frames in order, so everything sent before the barrier has
    been handled once this returns.
    """
    ws.send_json({"type": "turn_start"})
    frames = recv_until(ws, is_type("turn_started"))
    return frames[:-1]


def errors(frames: list) -> list:
    return [f for f in frames if isinstance(f, dict) and f.get("type") == "error"]
