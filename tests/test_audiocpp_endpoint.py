"""Mid-session silence endpoint trigger for the audio.cpp STT backend.

Model-free / server-free (the CLAUDE.md invariant): a real FastAPI `/ws/stream` driven by
a `TestClient`, the real `AudioCppSTTPipeline` speaking sidecar/PROTOCOL.md over a genuine
AF_UNIX socket to `tests/audiocpp_fake.py`. No GPU, no native sidecar, no model download.

The fake here is scripted to model the finalize-only Nemotron-3.5 behaviour of the earlier
audio.cpp pins, which a live run uncovered: it emits **nothing during AUDIO frames**
(`audio_events=[]`, `audio_final=None`) and produces its growing interim deltas + the FINAL
**only on FLUSH** (`flush_events=[...]`, `flush_final=...`). So while the client streams, no
transcription frame is emitted -- exactly the live bug shape (0 mid-session frames). The
ONLY way a per-utterance partial+final can appear DURING the session is a VoiceManager-side
silence timer that FLUSHes per utterance (mirroring qwen's qwen_silence_ms path). The
current pin streams its deltas during AUDIO (covered by test_audiocpp_nemotron_cadence.py);
this file still covers the per-utterance FLUSH path.

This test proves that mid-session endpoint trigger:
  * a silence GAP between utterances (the client stops sending audio) triggers a FLUSH;
  * the flush burst surfaces >=1 is_final=False partial and exactly one is_final=True final;
  * the trigger does NOT close the socket -- a second utterance is transcribed on the same
    connection, at the next (advanced, monotonic) segment_index.

The speculative endpoint-hold is nemotron-only, so every frame must carry endpoint=False.
"""
import time

import authkit  # test auth env; must be imported before voicemanager

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import voicemanager.config as cfg
from voicemanager.app import create_app
from voicemanager.session import SessionRegistry

from audiocpp_fake import FakeSidecar

# 512 samples of nonzero 16-bit PCM -- even length -> passes first-frame validation.
_PCM = b"\x11\x22" * 512

# Growing interim deltas emitted on FLUSH (the finalize burst of the earlier pins), then the
# committed final -- what a finalize-only Nemotron-3.5 sidecar produces per utterance.
_FLUSH_DELTAS = ["long", "long range", "long range scan"]
_FLUSH_FINAL = "long range scan complete"


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


def _stream_utterance(ws, *, frames: int = 8, gap: float = 0.02) -> None:
    """Send a multi-word utterance as a burst of PCM frames, each gap << the silence window.

    No inter-frame gap approaches the silence threshold, so the mid-session trigger must
    NOT fire during speech -- only once the burst stops (the deliberate silence after).
    """
    for _ in range(frames):
        ws.send_bytes(_PCM)
        time.sleep(gap)


def _recv_utterance(ws):
    """Block until the silence-triggered FLUSH pushes this utterance's frames.

    Collects is_final=False partials until the single is_final=True final arrives, then
    returns (partials, final). The connection is left OPEN (no close), so a second call
    after a second utterance proves the trigger did not tear the session down.
    """
    partials: list[dict] = []
    for _ in range(50):
        f = ws.receive_json()
        assert f.get("type") == "transcription", f
        if f["is_final"]:
            return partials, f
        partials.append(f)
    raise AssertionError("no is_final=True final arrived within bound")


def test_silence_gap_triggers_mid_session_finalize_without_closing(monkeypatch):
    """A silence gap flushes the utterance mid-session; the socket stays open for the next.

    Two utterances are streamed on ONE connection, each followed by silence. Each yields
    >=1 is_final=False partial + exactly one is_final=True final DURING the session, at a
    monotonically advancing segment_index -- proving both the per-utterance FLUSH trigger
    and that it never closes the connection.
    """
    monkeypatch.setenv("VM_AUDIOCPP_SILENCE_MS", "200")  # fire fast so the test is quick
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "30")       # far above the silence window
    cfg._settings = None
    try:
        # Model the finalize-only Nemotron-3.5 of the earlier pins: NOTHING during AUDIO,
        # the burst only on FLUSH.
        with FakeSidecar(audio_events=[], audio_final=None,
                         flush_events=_FLUSH_DELTAS, flush_final=_FLUSH_FINAL) as fake:
            client = TestClient(_app_for(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                ready = ws.receive_json()
                assert ready["type"] == "ready" and ready["stt_engine"] == "audiocpp"

                # --- utterance 1: stream, then go silent -> mid-session FLUSH -----------
                _stream_utterance(ws)
                partials1, final1 = _recv_utterance(ws)

                # --- utterance 2: same connection is still open (the key claim) ----------
                _stream_utterance(ws)
                partials2, final2 = _recv_utterance(ws)

            # >=1 is_final=False partial per utterance (surfaced from the finalize burst).
            assert len(partials1) >= 1, f"no mid-session partial for utterance 1: {partials1}"
            assert len(partials2) >= 1, f"no mid-session partial for utterance 2: {partials2}"

            # Exactly one is_final=True final per utterance, with the committed text.
            assert final1["text"] == _FLUSH_FINAL
            assert final2["text"] == _FLUSH_FINAL

            # The trigger did not close the socket: utterance 2 arrived on the same
            # connection, at the next (advanced, never-rewound) segment_index.
            assert final1["segment_index"] == 0
            assert final2["segment_index"] == 1

            # The speculative endpoint-hold is nemotron-only and must not leak here.
            for f in partials1 + partials2 + [final1, final2]:
                assert f.get("endpoint", False) is False, f
    finally:
        cfg._settings = None
