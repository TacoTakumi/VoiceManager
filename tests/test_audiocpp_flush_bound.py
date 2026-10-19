"""The FLUSH -> FINAL exchange is bounded: no FINAL cannot mean no exit.

Model-free and server-free (the CLAUDE.md invariant): the real AudioCppSTTPipeline over
a genuine AF_UNIX socket to tests/audiocpp_fake.py. No GPU, no native binary, no model.

_finalize_sync reads frames until it sees the FLUSH's FINAL. Its only per-read backstop
is the 120 s socket timeout, and every EVENT frame RESETS that clock -- so a sidecar
dripping EVENTs and never a FINAL used to keep the loop alive forever, hanging the
session instead of failing it. PROTOCOL.md guarantees exactly one FINAL per FLUSH, so
this is sidecar MISBEHAVIOUR (no real path to it is known); the pipeline's contract for sidecar
misbehaviour is "a defined exception rather than a hang", and this closes the last
hang with a wall-clock deadline on the whole FLUSH -> FINAL exchange.

Asserted here, with the deadline monkeypatched small so the suite stays fast:

  * an EVENT flood after FLUSH raises SidecarFault within the bound - never a hang;
  * a WS session whose finalize hits the flood error-closes through the existing fault
    path (error_type="stt_backend_fault", close 1011) and the server stays up;
  * a legitimately SLOW single decode (one long pause, then the FINAL) still succeeds:
    the bound must not break the case the generous per-recv timeout exists for.
"""
import asyncio
import time

import authkit  # test auth env; must be imported before voicemanager

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import voicemanager.config as cfg
import voicemanager.pipeline.audiocpp_stt as audiocpp_stt
from voicemanager.app import create_app
from voicemanager.pipeline.audiocpp_stt import AudioCppSTTPipeline, SidecarFault
from voicemanager.session import SessionRegistry

from audiocpp_fake import FakeSidecar

# 512 samples of nonzero 16-bit PCM -- even length passes first-frame validation.
_PCM = b"\x11\x22" * 512


def _config(fake: FakeSidecar) -> dict:
    """Point the pipeline at the fake's socket with NO binary -> connect, never spawn."""
    return {
        "sidecar_bin": "",
        "socket": fake.socket_path,
        "model": "",
        "model_task": "nemotron-3.5-asr-streaming",
        "backend": "cpu",
        "device": 0,
        "threads": 4,
        "assets_root": "",
    }


def test_event_flood_after_flush_raises_within_the_bound(monkeypatch):
    """A sidecar dripping EVENTs forever after FLUSH: finalize() raises, promptly.

    Each EVENT resets the per-recv socket timeout, so before the deadline this exact shape
    looped forever. The whole-exchange deadline must cut it off with SidecarFault --
    the sidecar is misbehaving, not misconfigured -- well inside the asserted ceiling.
    """
    monkeypatch.setattr(audiocpp_stt, "_FLUSH_FINAL_DEADLINE_S", 0.5)
    with FakeSidecar(flush_event_flood="still going") as fake:
        p = AudioCppSTTPipeline(config=_config(fake))
        asyncio.run(p.feed(_PCM))
        started = time.monotonic()
        with pytest.raises(SidecarFault):
            asyncio.run(p.finalize())
        assert time.monotonic() - started < 5.0, "bound fired far too late"
        p.dispose()  # frees the fake's serve thread from its drip promptly


def test_slow_single_decode_is_not_broken_by_the_bound(monkeypatch):
    """One long pause then a FINAL -- the legit slow-GPU-decode shape -- still succeeds.

    The deadline bounds the whole exchange, not any single pause; a FINAL that simply
    takes a while (here 0.8 s against a 3 s deadline) must arrive intact.
    """
    monkeypatch.setattr(audiocpp_stt, "_FLUSH_FINAL_DEADLINE_S", 3.0)
    with FakeSidecar(flush_events=["slow partial"], flush_final="slow ok",
                     flush_final_delay=0.8) as fake:
        p = AudioCppSTTPipeline(config=_config(fake))
        asyncio.run(p.feed(_PCM))
        assert asyncio.run(p.finalize()) == ("slow ok", "en", 0)


def _drain_until_close(ws, *, budget: int = 200):
    """Collect frames until the server closes the socket; return (frames, close_code)."""
    frames: list[dict] = []
    close_code = None
    try:
        for _ in range(budget):
            frames.append(ws.receive_json())
    except WebSocketDisconnect as disc:
        close_code = disc.code
    return frames, close_code


def test_ws_session_error_closes_on_flood_instead_of_hanging(monkeypatch):
    """The WS handler surfaces the tripped bound through the EXISTING fault path.

    The mid-session silence trigger runs finalize(); the flood trips the deadline; the
    session must get the defined stt_backend_fault error frame plus close 1011 (the
    same path a sidecar crash takes), and the server must stay alive -- not hang.
    """
    monkeypatch.setenv("VM_AUDIOCPP_SILENCE_MS", "200")  # fire fast so the test is quick
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "30")       # far above the silence window
    monkeypatch.setattr(audiocpp_stt, "_FLUSH_FINAL_DEADLINE_S", 0.5)
    cfg._settings = None
    try:
        with FakeSidecar(flush_event_flood="never final") as fake:
            app = create_app()
            app.state.stt_engine_type = "audiocpp"
            app.state.audiocpp_config = _config(fake)
            app.state.session_registry = SessionRegistry()
            client = TestClient(app)
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"
                for _ in range(8):
                    ws.send_bytes(_PCM)
                    time.sleep(0.02)
                # Stop sending: the silence gap triggers the mid-session finalize, whose
                # flood must trip the bound -> defined error frame + defined close code.
                frames, close_code = _drain_until_close(ws)
                errors = [f for f in frames if f.get("type") == "error"]
                assert len(errors) == 1, f"expected one error frame, got {frames}"
                assert errors[0]["error_type"] == "stt_backend_fault"
                assert close_code == 1011, f"expected close 1011, got {close_code}"
            # The VoiceManager process survived the misbehaving sidecar.
            assert client.get("/health").json()["status"] == "ok"
    finally:
        cfg._settings = None
