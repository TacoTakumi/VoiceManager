"""WS end-to-end under the audiocpp TTS engine.

Model-free and server-free (the CLAUDE.md invariant): the real app and its real
WS receive loop, the fake TTS sidecar (tests/audiocpp_tts_fake.py) over a real
AF_UNIX socket, and a spawn stub for the ownership tests. What must hold:

  * a tts request end to end delivers audio_start with sample_rate 48000 and
    binary audio whose sample count matches what the sidecar synthesized - no
    resample, schemas untouched
  * two sequential tts requests on one connection both complete
  * a /ws/stream connection under audiocpp owns exactly ONE TTS sidecar,
    spawned on a private socket path at connect and reaped on disconnect,
    mirroring the STT ownership pattern
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import authkit  # test auth env; must be imported before voicemanager

import pytest
from fastapi.testclient import TestClient

import voicemanager.pipeline.audiocpp_tts as audiocpp_tts
from voicemanager.app import create_app
from voicemanager.pipeline.audiocpp_tts import BREEZE_TTS_MODEL_TASK, AudioCppTTSEngine
from voicemanager.pipeline.audiocpp_tts import QWEN3_TTS_MODEL_TASK
from voicemanager.transport import websocket as ws_module

from audiocpp_tts_fake import DEFAULT_CHUNKS, DEFAULT_FLUSH, FakeTtsSidecar

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_REFS = REPO_ROOT / "voices" / "designed" / "refs.json"


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
    """Read one TTS reply: audio_start, concatenated binary audio, audio_end."""
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


# --- audio_start follows READY -----------------------------------------------------------

def test_tts_audio_start_carries_the_ready_stated_24k_rate(monkeypatch):
    """A sidecar whose READY says 24000 (breeze) yields audio_start 24000 for a
    one-shot tts request; the sample count still matches, no resample."""
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    with FakeTtsSidecar(ready_sample_rate=24000) as fake:
        engine = AudioCppTTSEngine({"socket": fake.socket_path,
                                    "model_task": BREEZE_TTS_MODEL_TASK})
        engine.load(voices_path=str(REAL_REFS))
        with ThreadPoolExecutor(max_workers=1) as executor:
            client = TestClient(_tts_app(engine, executor))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"
                ws.send_json({"type": "tts", "text": "Shields up.", "request_id": "b1"})
                start, pcm, end = _drain_request(ws)

        assert start["sample_rate"] == 24000
        assert end["request_id"] == "b1"
        sent = b"".join(DEFAULT_CHUNKS) + DEFAULT_FLUSH
        assert len(pcm) // 2 == len(sent) // 2


def test_qwen3_tts_audio_start_carries_24k(monkeypatch):
    """A qwen3 sidecar's READY says 24000; audio_start carries it for a one-shot tts
    request and the sample count matches what the sidecar sent."""
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    with FakeTtsSidecar(ready_sample_rate=24000) as fake:
        engine = AudioCppTTSEngine({"socket": fake.socket_path,
                                    "model_task": QWEN3_TTS_MODEL_TASK})
        engine.load(voices_path=str(REAL_REFS))
        with ThreadPoolExecutor(max_workers=1) as executor:
            client = TestClient(_tts_app(engine, executor))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"
                ws.send_json({"type": "tts", "text": "Shields up.", "request_id": "q1"})
                start, pcm, end = _drain_request(ws)

    assert start["sample_rate"] == 24000
    assert end["request_id"] == "q1"
    sent = b"".join(DEFAULT_CHUNKS) + DEFAULT_FLUSH
    assert len(pcm) // 2 == len(sent) // 2


# --- the unchanged client contract at 48 kHz --------------------------------

def test_tts_delivers_48k_audio_start_and_matching_sample_count(monkeypatch):
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    with FakeTtsSidecar() as fake:
        engine = _external_engine(fake)
        with ThreadPoolExecutor(max_workers=1) as executor:
            client = TestClient(_tts_app(engine, executor))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"
                ws.send_json({"type": "tts", "text": "Shields up.", "request_id": "r1"})
                start, pcm, end = _drain_request(ws)

        assert start["sample_rate"] == 48000
        assert start["channels"] == 1
        assert start["bit_depth"] == 16
        assert start["encoding"] == "pcm"
        assert end["request_id"] == "r1"
        sent = b"".join(DEFAULT_CHUNKS) + DEFAULT_FLUSH
        assert len(pcm) // 2 == len(sent) // 2, \
            "delivered sample count must match the synthesized duration at 48 kHz"
        assert len(fake.requests) == 1
        assert fake.requests[0]["ref_text"].strip()


def test_two_sequential_tts_requests_both_complete(monkeypatch):
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    with FakeTtsSidecar() as fake:
        engine = _external_engine(fake)
        with ThreadPoolExecutor(max_workers=1) as executor:
            client = TestClient(_tts_app(engine, executor))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"
                for rid, text in (("r1", "First request."), ("r2", "Second request.")):
                    ws.send_json({"type": "tts", "text": text, "request_id": rid})
                    start, pcm, end = _drain_request(ws)
                    assert end["request_id"] == rid
                    assert len(pcm) > 0
        assert [r["text"] for r in fake.requests] == ["First request.", "Second request."]


# --- per-session sidecar ownership ------------------------------------

class _StubProc:
    """Stands in for subprocess.Popen: records terminate()/wait()."""

    def __init__(self):
        self.terminated = False
        self.waited = False
        self.pid = -1

    def poll(self):
        return None

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True
        return 0

    def kill(self) -> None:
        self.terminated = True


def test_ws_session_owns_one_private_sidecar_spawned_on_connect_reaped_on_disconnect(
        monkeypatch):
    """Owned mode: the WS session spawns its OWN TTS sidecar on a private path at
    connect (not lazily at the first request), serves requests from that one
    process, and reaps it on disconnect."""
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)

    spawns: list[dict] = []
    fakes: list[FakeTtsSidecar] = []

    def _fake_spawn(self, bin_path, socket_path):
        fake = FakeTtsSidecar(socket_path=socket_path)
        fake.__enter__()
        fakes.append(fake)
        stub = _StubProc()
        spawns.append({"bin": bin_path, "socket": socket_path, "proc": stub,
                       "fake": fake})
        return stub

    monkeypatch.setattr(audiocpp_tts._TtsSidecarClient, "_spawn", _fake_spawn)

    engine = AudioCppTTSEngine({
        "sidecar_bin": "/fake/bin/audiocpp_sidecar",
        "model": "/fake/dots.gguf",
    })
    engine.load(voices_path=str(REAL_REFS))

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            client = TestClient(_tts_app(engine, executor))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"

                # Spawned ON CONNECT: the session's sidecar handshakes without any
                # tts request having been sent.
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline and (
                        not spawns or spawns[0]["fake"].hello is None):
                    time.sleep(0.02)
                assert len(spawns) == 1, "exactly one sidecar per session"
                assert spawns[0]["fake"].hello is not None, \
                    "the session's sidecar must be handshaken at connect"
                assert spawns[0]["fake"].hello["model_task"] == "dots-tts-streaming"
                assert "vm-audiocpp-tts-" in spawns[0]["socket"], \
                    "the socket path must be private to this session"

                # A request is served by that same sidecar - still exactly one spawn.
                ws.send_json({"type": "tts", "text": "One.", "request_id": "r1"})
                _drain_request(ws)
                assert len(spawns) == 1
                assert len(spawns[0]["fake"].requests) == 1

        # Reaped on disconnect.
        deadline = time.monotonic() + 5.0
        proc = spawns[0]["proc"]
        while time.monotonic() < deadline and not (proc.terminated and proc.waited):
            time.sleep(0.02)
        assert proc.terminated and proc.waited, \
            "the session's sidecar must be reaped on disconnect"
    finally:
        for fake in fakes:
            fake.stop()


def test_session_engine_is_a_separate_instance_sharing_validated_presets():
    """The startup engine only validates; each WS session gets its own engine
    (own sidecar client) sharing the already-validated presets."""
    engine = AudioCppTTSEngine({"sidecar_bin": "/fake/bin", "model": "/fake.gguf"})
    engine.load(voices_path=str(REAL_REFS))
    session = engine.session_engine()
    assert isinstance(session, AudioCppTTSEngine)
    assert session is not engine
    assert session._presets is engine._presets
    assert session._client is not engine._client
