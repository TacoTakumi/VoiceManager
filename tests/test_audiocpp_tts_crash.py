"""A dying TTS sidecar fails only its own session.

Model-free and server-free (the CLAUDE.md invariant): the real app and its real
WS receive loop over a `TestClient`, the fake TTS sidecar
(tests/audiocpp_tts_fake.py) on a genuine AF_UNIX socket, and a real `python -c`
child for the reap half. No GPU, no native binary, no model. Mirrors the STT
crash/reap suites (test_audiocpp_crash.py / test_audiocpp_reap.py).

One TTS sidecar process per WS session on a private socket path, so a
sidecar death is CONTAINED - it fails its own session with a defined error and
every other live session continues. What these tests assert:

  * a sidecar death mid-request (abrupt EOF, no TTS_DONE - the crash shape a
    kill -9 leaves on the wire) raises `SidecarFault`, the process-fatal class,
    not the refusal-class `SidecarError` - so the WS layer can tell "the backend
    WAS serving and broke" from "the backend refused to serve" (the same split
    the STT side makes);
  * on the WS path that fault surfaces as the DEFINED session error - one error
    frame (error_type="tts_backend_fault") plus close 1011 - with no hang and no
    catch-all "Unexpected error" traceback in the server log path;
  * a PEER session on the same server keeps synthesizing through and after the
    fault, and the VoiceManager process stays alive (/health answers ok);
  * a sidecar process that died on its own is still reaped on close() - no
    `<defunct>` zombie survives (the STT zombie lesson, applied to the TTS client).

CONCURRENCY NOTE. Unlike the ASR fake (single-session by design), the TTS fake
serves one thread per connection - which is a FAIR stand-in here, because the
production topology under test is one sidecar PROCESS per session: the
peer's fake connection models the peer's own separate sidecar, exactly the
isolation the requirement claims. The live half of the claim (kill -9 one
session's real sidecar, the other keeps talking) is a live spot check.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import authkit  # test auth env; must be imported before voicemanager

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import voicemanager.config as cfg
from voicemanager.app import create_app
from voicemanager.pipeline.audiocpp_stt import SidecarFault
from voicemanager.pipeline.audiocpp_tts import AudioCppTTSEngine, _TtsSidecarClient
from voicemanager.transport import websocket as ws_module

from audiocpp_tts_fake import FakeTtsSidecar

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_REFS = REPO_ROOT / "voices" / "designed" / "refs.json"

#: The marker the fake kills on - only the session that speaks it dies.
DIE = "%%DIE%%"


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


def _drain_until_close(ws, *, budget: int = 200) -> tuple[list[dict], int | None]:
    """Collect JSON frames (skipping binary PCM) until the server closes."""
    frames: list[dict] = []
    close_code = None
    try:
        for _ in range(budget):
            frame = ws.receive()
            if frame.get("type") == "websocket.close":
                close_code = frame.get("code")
                break
            if frame.get("bytes") is not None:
                continue
            if frame.get("text") is not None:
                frames.append(json.loads(frame["text"]))
    except WebSocketDisconnect as disc:
        close_code = disc.code
    return frames, close_code


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


# --- the fault class: death mid-request is a FAULT, not a refusal (engine level) -----

def test_mid_request_eof_raises_sidecar_fault() -> None:
    """An abrupt EOF mid-request (no TTS_DONE) must raise the process-fatal
    SidecarFault - the class the WS layer maps to the defined session error -
    not the refusal-class SidecarError the 4001/4002 paths are for."""
    async def run(engine: AudioCppTTSEngine) -> None:
        async for _ in engine.stream_synthesize("this request dies"):
            pass

    with FakeTtsSidecar(die_on_text=True) as fake:
        engine = _external_engine(fake)
        try:
            with pytest.raises(SidecarFault):
                asyncio.run(run(engine))
        finally:
            engine.dispose()
        assert len(fake.requests) == 1


# --- the WS surface: defined error on that session only, peer unaffected -------------

def test_tts_sidecar_death_error_closes_only_that_session_peer_continues(
        monkeypatch, caplog):
    """The crash acceptance, end to end on the real WS path:

    - ws_dies: its request hits the sidecar-death EOF; the session must get ONE
      defined error frame (tts_backend_fault) plus close 1011 - no hang.
    - ws_peer: open on the same server throughout, its own fake connection (its
      own per-session sidecar) - synthesizes before AND after the fault.
    - the server process stays alive (/health ok) and the fault never reaches
      the catch-all server_error path (no "Unexpected error" traceback).
    """
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "30")
    cfg._settings = None
    try:
        with FakeTtsSidecar(die_on_text_containing=DIE) as fake:
            engine = _external_engine(fake)
            with ThreadPoolExecutor(max_workers=2) as executor:
                client = TestClient(_tts_app(engine, executor))
                with caplog.at_level(logging.WARNING, logger="voicemanager"), \
                     authkit.ws_connect(client, "/ws/stream") as ws_dies, \
                     authkit.ws_connect(client, "/ws/stream") as ws_peer:
                    assert ws_dies.receive_json()["type"] == "ready"
                    assert ws_peer.receive_json()["type"] == "ready"

                    # The peer synthesizes normally BEFORE the fault.
                    ws_peer.send_json({"type": "tts", "text": "Before the fault.",
                                       "request_id": "p1"})
                    _, pcm, end = _drain_request(ws_peer)
                    assert end["request_id"] == "p1" and len(pcm) > 0

                    # Fault the first session: its request carries the die marker,
                    # so ONLY its sidecar connection dies (abrupt EOF, no TTS_DONE).
                    ws_dies.send_json({"type": "tts", "text": f"Please {DIE} now.",
                                       "request_id": "d1"})
                    dead_frames, close_code = _drain_until_close(ws_dies)

                    # The DEFINED session error: one error frame + close 1011.
                    errors = [f for f in dead_frames if f.get("type") == "error"]
                    assert len(errors) == 1, f"expected one error frame, got {dead_frames}"
                    assert errors[0]["error_type"] == "tts_backend_fault"
                    assert close_code == 1011, f"expected close 1011, got {close_code}"

                    # The peer was open through the fault and keeps synthesizing AFTER it.
                    ws_peer.send_json({"type": "tts", "text": "After the fault.",
                                       "request_id": "p2"})
                    _, pcm, end = _drain_request(ws_peer)
                    assert end["request_id"] == "p2" and len(pcm) > 0

            # The VoiceManager process survived the sidecar fault.
            assert client.get("/health").json()["status"] == "ok"

        # The fault took the defined path, not the catch-all bug path: no
        # "Unexpected error in WebSocket handler" traceback may be logged.
        assert not [r for r in caplog.records
                    if "Unexpected error in WebSocket handler" in r.getMessage()]
    finally:
        cfg._settings = None


# --- reap: a sidecar that died on its own leaves no zombie ---------------------------

_READY_LINE = "print('ready', flush=True); "
_IDLE_CHILD = "import time; " + _READY_LINE + "time.sleep(300)"


def _is_zombie_or_unreaped(pid: int) -> bool:
    """True while `pid` is still an unreaped child of ours (a `<defunct>` entry)."""
    try:
        reaped_pid, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return False        # not our child any more -> somebody wait()ed it: reaped
    return reaped_pid == pid


def test_close_reaps_a_tts_sidecar_that_already_died(monkeypatch) -> None:
    """The reap clause: the crashed sidecar's process is genuinely wait()ed on
    close() - no `<defunct>` zombie survives - and close() does not raise on a
    process that is already gone. Real OS child, fake on its socket path (the
    test_audiocpp_reap.py split: the process half is real, the wire half fake)."""
    fakes: list[FakeTtsSidecar] = []
    procs: list[subprocess.Popen] = []

    def _spawn_real_child(self, bin_path: str, socket_path: str) -> subprocess.Popen:
        fake = FakeTtsSidecar(socket_path=socket_path)
        fake.__enter__()
        fakes.append(fake)
        proc = subprocess.Popen([sys.executable, "-c", _IDLE_CHILD],
                                stdout=subprocess.PIPE, text=True)
        procs.append(proc)
        assert proc.stdout.readline().strip() == "ready", "child sidecar failed to start"
        return proc

    monkeypatch.setattr(_TtsSidecarClient, "_spawn", _spawn_real_child)
    client = _TtsSidecarClient({
        "sidecar_bin": "/nonexistent/audiocpp_sidecar",  # never executed; _spawn stubbed
        "model": "/nonexistent/dots.gguf",
    })
    try:
        client.connect()
        proc = procs[0]
        proc.kill()  # the sidecar dies on its own - the crash, not our teardown
        time.sleep(0.2)

        client.close()  # must not raise on a process that is already gone
        reaper = client._reaper
        assert reaper is not None, "close() must hand the dead child to a reaper"
        reaper.join(10.0)
        assert not reaper.is_alive(), "TTS sidecar reaper did not finish"

        assert proc.returncode is not None, "close() must wait() the sidecar it spawned"
        assert not _is_zombie_or_unreaped(proc.pid), \
            "crashed TTS sidecar left as a <defunct> zombie after close()"
    finally:
        for fake in fakes:
            fake.stop()
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            proc.stdout.close()
