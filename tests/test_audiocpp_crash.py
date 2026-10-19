"""A sidecar crash mid-session error-closes only that session, server stays up.

Model-free and server-free (the CLAUDE.md invariant): a real FastAPI `/ws/stream`
driven by a `TestClient`, with the real `AudioCppSTTPipeline` speaking sidecar/
PROTOCOL.md over a genuine AF_UNIX socket to `tests/audiocpp_fake.py`. No GPU, no
native binary, no model -- the fake simulates a process-fatal audio.cpp fault by
closing the socket abruptly mid-stream (no BYE), the exact fault shape the pipeline
surfaces as a bounded `SidecarFault` (rather than a hang).

A process-fatal fault in the audio.cpp runtime must NOT crash the
VoiceManager server. Because the runtime lives in a separate sidecar process reached
over a socket, and each WS connection is its own asyncio task, the fault is contained.
What this model-free test asserts -- and ONLY this:

  * the affected session gets a DEFINED error frame (error_type="stt_backend_fault")
    plus a defined close code (1011), rather than a hang or a silent drop;
  * the VoiceManager process stays alive (its HTTP /health still answers ok);
  * a PEER WS session that was already open when the fault struck is not torn down by
    it -- the fault does not propagate across sessions, and once the sidecar is free
    that peer transcribes normally on the same server + same sidecar.

WHAT THIS TEST DELIBERATELY DOES NOT ASSERT. The fault-isolation "leaves concurrent
sessions unaffected" clause is NOT verifiable model-free. One sidecar process serves one
session at a time (sidecar/PROTOCOL.md; a concurrent second connection is refused with
ERROR `busy`), and real concurrency comes from each /ws/stream session spawning its OWN
sidecar process -- which needs the real binary and a GPU. Both sessions here share one
external fake, so the peer below is served only AFTER the faulted connection is gone.
That is survival of a peer, not two sessions running at once.

An earlier version of this test DID assert concurrency, and passed only because the fake
spawned a thread per connection and so out-capabilitied the runtime it stands in for. The
fake is single-session by default now (audiocpp_fake.py). The concurrency claim stays a
LIVE one: scripts/audiocpp_two_session_repro.py exercised the gap against the real
sidecar and is the acceptance evidence for per-session sidecars.

The fake is armed with die_on_audio_conn=0 so ONLY the first sidecar connection (the
faulted session) dies; the next connection is healthy.
"""
import time

import authkit  # test auth env; must be imported before voicemanager

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import voicemanager.config as cfg
from voicemanager.app import create_app
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


def _app_for(fake: FakeSidecar):
    app = create_app()
    app.state.stt_engine_type = "audiocpp"
    app.state.audiocpp_config = _config(fake)
    app.state.session_registry = SessionRegistry()
    return app


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


def test_sidecar_crash_error_closes_only_that_session_and_server_survives(monkeypatch):
    """A mid-session sidecar death: the affected session error-closes, a peer survives.

    - ws_dies: its sidecar connection (ordinal 0) dies on the first AUDIO frame; the
      handler must deliver a stt_backend_fault error frame + close 1011.
    - ws_peer: a second WS session, open on the same server throughout the fault. The
      fault does not tear it down; it transcribes on the shared fake's next connection,
      ordinal 1. This is survival, NOT concurrency -- see the module docstring; the
      concurrency half of fault isolation is a live-only claim.
    - /health still answers ok -> the VoiceManager process is alive, not crashed.
    """
    # Generous timeout: the test drives the fault directly, never via inactivity.
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "30")
    cfg._settings = None
    try:
        with FakeSidecar(die_on_audio_conn=0, audio_events=["healthy partial"],
                         flush_final="done") as fake:
            client = TestClient(_app_for(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws_dies, \
                 authkit.ws_connect(client, "/ws/stream") as ws_peer:
                assert ws_dies.receive_json()["type"] == "ready"
                assert ws_peer.receive_json()["type"] == "ready"

                # --- fault the first session. Its sidecar connection is ordinal 0
                # (its audio arrives first), so it dies on AUDIO. Send a burst: the
                # abrupt EOF surfaces as a SidecarFault within a frame or two.
                try:
                    for _ in range(60):
                        ws_dies.send_bytes(_PCM)
                        time.sleep(0.01)
                except Exception:
                    # Once the server error-closes the faulted session, further sends
                    # on it fail -- expected; the close is the point.
                    pass

                dead_frames, close_code = _drain_until_close(ws_dies)

                # DEFINED error frame + close code for the affected session.
                errors = [f for f in dead_frames if f.get("type") == "error"]
                assert len(errors) == 1, f"expected one error frame, got {dead_frames}"
                assert errors[0]["error_type"] == "stt_backend_fault"
                assert close_code == 1011, f"expected close 1011, got {close_code}"

                # --- the peer session was open throughout the fault and was NOT torn
                # down by it: on the SAME still-alive server it now transcribes over the
                # sidecar's next connection (ordinal 1, healthy). The single-session fake
                # only admits it now that the faulted connection is gone -- so this is
                # "the fault did not propagate", not "the two ran concurrently".
                for _ in range(16):
                    ws_peer.send_bytes(_PCM)
                    time.sleep(0.02)
                peer_interim = None
                for _ in range(50):
                    frame = ws_peer.receive_json()
                    if frame.get("type") == "transcription" and frame["is_final"] is False:
                        peer_interim = frame
                        break
                assert peer_interim is not None, "peer session got no interim partial"
                # The fake repeats its scripted event on EVERY audio frame while the
                # client ACCUMULATES deltas, so the captured interim may carry
                # N repetitions -- how many is a cross-thread race. The claim here is
                # survival (the healthy sidecar's output surfaced), not text shape.
                assert peer_interim["text"].startswith("healthy partial")

            # The VoiceManager process survived the sidecar fault.
            assert client.get("/health").json()["status"] == "ok"
    finally:
        cfg._settings = None
