"""A recoverable FLUSH error drops its utterance WHOLLY - no dangling partial.

Model-free / server-free (the CLAUDE.md invariant): a real FastAPI `/ws/stream` driven by
a `TestClient`, the real `AudioCppSTTPipeline` speaking sidecar/PROTOCOL.md over a genuine
AF_UNIX socket to `tests/audiocpp_fake.py`. No GPU, no native sidecar, no model download.

The shape under test is the one drive_loop.cpp really produces: it emits its EVENT
deltas through the sink FIRST, then `session.finalize()` throws, so its catch block
writes ERROR `internal` INSTEAD of the FINAL and keeps serving. The
pipeline calls that utterance dropped and streams on - but the EVENT burst has
already landed in the live partial, so the WS flush handler used to surface that stale
text as an `is_final=False` frame with NO final ever following it. "Drop this utterance"
only half-held, and a client tracking partial->final pairs saw one orphaned forever.

The dropped utterance must emit NEITHER a partial NOR a final, and the session must stay
usable: the next utterance on the SAME connection transcribes normally at the next
segment_index. `tests/test_audiocpp_handshake_error.py` covers the same recovery at the
pipeline level; this file pins the observable WS frames, which is where the defect showed.
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

# The pre-error EVENT burst: text the sidecar HAS produced and the client HAS seen as a
# partial before finalize() throws. This is the text that used to dangle.
_BURST = "burst text"
_SECOND_FINAL = "second utterance"


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
    """Send an utterance as a burst of frames, each gap << the silence window."""
    for _ in range(frames):
        ws.send_bytes(_PCM)
        time.sleep(gap)


def _drain_to_close(ws) -> list[dict]:
    """Collect every transcription frame until the inactivity close (bounded)."""
    frames: list[dict] = []
    try:
        for _ in range(200):
            f = ws.receive_json()
            if f.get("type") == "transcription":
                frames.append(f)
    except WebSocketDisconnect:
        pass
    return frames


def test_recoverable_flush_error_emits_no_orphan_partial(monkeypatch):
    """The errored utterance emits nothing at all; the next one is unaffected.

    Two utterances on one connection, each endpointed by the frame-gap trigger.
    The fake errors only the FIRST FLUSH (after emitting its EVENT burst), then serves
    normally - drive_loop.cpp's real recoverable shape. Segment 0 must therefore carry
    ZERO frames (not a partial with no final), and segment 1 must carry the normal
    >=1 partial + exactly one final.
    """
    monkeypatch.setenv("VM_AUDIOCPP_SILENCE_MS", "200")  # fire the gap trigger fast
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "2")        # then close so the drain ends
    cfg._settings = None
    try:
        # Finalize-only Nemotron-3.5 shape (earlier pins): nothing during AUDIO, the burst
        # only on FLUSH --
        # and the first FLUSH answers ERROR `internal` in place of its FINAL.
        with FakeSidecar(audio_events=[], audio_final=None,
                         flush_events=[_BURST], flush_error="internal",
                         flush_final=_SECOND_FINAL) as fake:
            client = TestClient(_app_for(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                ready = ws.receive_json()
                assert ready["type"] == "ready" and ready["stt_engine"] == "audiocpp"

                _stream_utterance(ws)   # utterance 1 -> FLUSH -> EVENT burst, then ERROR
                time.sleep(0.5)         # > the 200 ms window: the gap trigger flushes it
                _stream_utterance(ws)   # utterance 2 -> FLUSH -> EVENT burst, then FINAL
                # Stop sending: the gap trigger flushes #2, then the timeout closes.
                frames = _drain_to_close(ws)

            assert len(fake.connections) == 1, (
                "the pipeline reconnected instead of continuing on the live connection")

        # The dropped utterance is dropped WHOLLY: segment 0 emits no frame of any kind.
        seg0 = [f for f in frames if f["segment_index"] == 0]
        assert seg0 == [], (
            f"the errored utterance must emit neither a partial nor a final, got {seg0}")

        # No orphan anywhere: every segment that produced a partial also produced a final.
        partial_segs = {f["segment_index"] for f in frames if not f["is_final"]}
        final_segs = {f["segment_index"] for f in frames if f["is_final"]}
        assert partial_segs <= final_segs, (
            f"is_final=False frames with no final for segment(s) "
            f"{sorted(partial_segs - final_segs)}: {frames}")

        # The session stayed usable: utterance 2 transcribed normally at the next segment.
        finals = [f for f in frames if f["is_final"]]
        assert [(f["text"], f["segment_index"]) for f in finals] == [(_SECOND_FINAL, 1)], (
            f"expected exactly one final, for utterance 2 at segment 1, got {finals}")
        assert len([f for f in frames if not f["is_final"]]) >= 1, (
            f"utterance 2's own flush burst partial went missing too: {frames}")

        # The speculative endpoint-hold is nemotron-only and must not leak here.
        for f in frames:
            assert f.get("endpoint", False) is False, f
    finally:
        cfg._settings = None
