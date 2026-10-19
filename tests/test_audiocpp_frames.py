"""End-to-end /ws/stream frame pattern for the audio.cpp STT backend.

Model-free and server-free (the CLAUDE.md invariant): a real FastAPI `/ws/stream`
driven by a `TestClient`, with the real `AudioCppSTTPipeline` speaking sidecar/
PROTOCOL.md over a genuine AF_UNIX socket to `tests/audiocpp_fake.py`. No GPU, no
native sidecar binary, no model download -- the fake scripts the EVENT deltas and
the FINAL, so the assertion is purely about the *observable frame mapping*:

    streaming a multi-word utterance yields >=1 TranscriptionMessage is_final=False
    partial during speech and exactly one is_final=True final at end-of-utterance,

matching the NeMo (nemotron) engine's disable-path frame pattern for the same input
(interim partials via get_current_text(), one committed final via take_final()/
finalize()). The audio.cpp branch runs the SAME generic drive path with the
speculative endpoint-hold OFF (hold_machine=None), so every frame carries
endpoint=False -- the hold is a nemotron-only feature and must not leak here.

The two tests cover both commit mechanisms:
  * final via finalize()  -- the shipping Nemotron-3.5 model (deltas at speech,
    the FINAL arrives on FLUSH at end-of-utterance/teardown).
  * final via take_final() -- a model that volunteers the FINAL mid-stream (generic
    forwarding); finalize()'s trailing flush is then empty and drops nothing.

Each test streams the utterance then goes silent, letting the inactivity timeout
flush the trailing final and close the socket, so the drain loop terminates on
WebSocketDisconnect and can prove *exactly one* final across the whole session.
"""
import time

import authkit  # test auth env; must be imported before voicemanager

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import voicemanager.config as cfg
from voicemanager.app import create_app
from voicemanager.session import SessionRegistry

from audiocpp_fake import FakeSidecar

# 512 samples of nonzero 16-bit PCM -- even length -> passes first-frame validation
# (raw PCM), content irrelevant to the fake (forwarded byte-for-byte).
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


def _stream_utterance(ws, *, frames: int = 16, gap: float = 0.02) -> None:
    """Send a multi-word utterance as a burst of PCM frames.

    A small gap lets the fake's serve thread emit its scripted EVENT/FINAL and the
    pipeline's non-blocking reply drain pick them up across the thread hop -- the same
    cross-thread delivery the pipeline tests poll for (feed #N's EVENT lands on
    drain #N+1). Each send resets the server's inactivity clock, so the timeout only
    starts counting from the last frame (the deliberate silence below).
    """
    for _ in range(frames):
        ws.send_bytes(_PCM)
        time.sleep(gap)


def _drain_to_close(ws) -> list[dict]:
    """Collect every frame until the server closes the socket (bounded)."""
    frames: list[dict] = []
    try:
        for _ in range(200):
            frames.append(ws.receive_json())
    except WebSocketDisconnect:
        pass
    return frames


def _split(frames: list[dict]):
    transcriptions = [f for f in frames if f.get("type") == "transcription"]
    interims = [f for f in transcriptions if f["is_final"] is False]
    finals = [f for f in transcriptions if f["is_final"] is True]
    return transcriptions, interims, finals


def test_ws_stream_maps_interim_deltas_and_final_via_finalize(monkeypatch):
    """Delta-streaming pattern: EVENT deltas during speech, the FINAL on flush at end.

    >=1 is_final=False partial while streaming, each carrying the ACCUMULATED
    transcript-so-far (never the last fragment alone), then exactly
    one is_final=True final (via finalize() at the inactivity-timeout teardown),
    every frame endpoint=False. The scripted is_final=true marker event must produce
    no partial frame and leave the accumulation untouched.
    """
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "1")
    cfg._settings = None
    try:
        with FakeSidecar(audio_delta_script=[
                            "long ", "range",
                            {"is_final": True,
                             "partial_text": {"text": " scan", "language": "en"}},
                         ],
                         audio_final=None,
                         flush_final="long range scan complete") as fake:
            client = TestClient(_app_for(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                ready = ws.receive_json()
                assert ready["type"] == "ready" and ready["stt_engine"] == "audiocpp"

                _stream_utterance(ws)
                # Go silent: the inactivity timeout flushes the trailing final and closes.
                frames = _drain_to_close(ws)

            _, interims, finals = _split(frames)

            # >=1 is_final=False partial during speech, every one the accumulated
            # transcript-so-far: a prefix chain ending at the full accumulation, with
            # the marker event's text (" scan") never surfacing as a partial.
            assert len(interims) >= 1, f"no interim partial emitted; got {frames}"
            texts = [f["text"] for f in interims]
            assert all("long range".startswith(t) for t in texts), texts
            assert texts[-1] == "long range", texts
            for earlier, later in zip(texts, texts[1:]):
                assert later.startswith(earlier), texts

            # Exactly one is_final=True final at end-of-utterance.
            assert len(finals) == 1, f"expected exactly one final, got {finals}"
            assert finals[0]["text"] == "long range scan complete"
            assert finals[0]["segment_index"] == 0

            # The speculative endpoint-hold is nemotron-only: it must not leak here.
            assert all(f.get("endpoint", False) is False for f in interims + finals)
            # Ordering: the committed final follows at least one interim.
            transcriptions = interims + finals
            assert transcriptions.index(finals[0]) >= 1
    finally:
        cfg._settings = None


def test_ws_stream_maps_final_via_take_final(monkeypatch):
    """Generic-model pattern: the sidecar volunteers the FINAL mid-stream.

    The volunteered FINAL is drained via take_final() and mapped to the one
    is_final=True; finalize()'s trailing flush is empty and drops nothing, so the
    session still shows exactly one final. Proves the take_final() half.
    """
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "1")
    cfg._settings = None
    try:
        with FakeSidecar(audio_events=["scanning"], audio_final="scan complete",
                         flush_final="") as fake:
            client = TestClient(_app_for(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"

                _stream_utterance(ws)
                frames = _drain_to_close(ws)

            _, interims, finals = _split(frames)

            assert len(interims) >= 1, f"no interim partial emitted; got {frames}"
            # Exactly one is_final=True -- the volunteered final via take_final(); the
            # empty trailing finalize() flush emits nothing.
            assert len(finals) == 1, f"expected exactly one final, got {finals}"
            assert finals[0]["text"] == "scan complete"
            assert finals[0]["segment_index"] == 0
            assert all(f.get("endpoint", False) is False for f in interims + finals)
    finally:
        cfg._settings = None
