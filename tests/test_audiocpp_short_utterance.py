"""Sub-threshold utterances yield NO frames and leave the session healthy.

Model-free and server-free (the CLAUDE.md invariant): a real FastAPI `/ws/stream`
driven by a `TestClient`, the real `AudioCppSTTPipeline` speaking sidecar/PROTOCOL.md
over a genuine AF_UNIX socket to `tests/audiocpp_fake.py`. No GPU, no native sidecar,
no model download.

Voxtral produces no output at all for roughly under 0.6 s of speech: no deltas during
AUDIO and an EMPTY finalize at FLUSH. The fake scripts exactly that shape for the
first utterance (`None` delta entries -- nothing per frame -- and a flush_script pair
of no events + an empty FINAL). What must hold (matching how the other
engines VAD-gate non-speech):

  * the sub-threshold utterance produces ZERO transcription frames over the WS --
    no dangling is_final=False partial, no empty is_final=True final;
  * the session stays healthy: the FOLLOWING utterance on the SAME connection
    transcribes normally through the re-armed session (growing partials + final);
  * the suppression is content-based, not utterance-counted: the second utterance's
    NON-empty final must not be dropped;
  * the empty flush still advances the (monotonic, never-rewound) segment_index, so
    the following utterance's final lands at seg 1 -- proof the first FLUSH really
    happened and was suppressed, not skipped.
"""
import time

import authkit  # test auth env; must be imported before voicemanager

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import voicemanager.config as cfg
from voicemanager.app import create_app
from voicemanager.session import SessionRegistry

from audiocpp_fake import FakeSidecar

VOXTRAL_ID = "voxtral-mini-4b-realtime-asr-streaming"

# 512 samples of nonzero 16-bit PCM -- even length -> passes first-frame validation.
_PCM = b"\x11\x22" * 512

# Utterance 1 is sub-threshold: three AUDIO frames, no delta for any of them.
_UTT1_FRAMES = 3
# Utterance 2 is normal voxtral streaming: one delta per frame, marker + final at FLUSH.
_UTT2_DELTAS = ["all ", "systems ", "nominal"]
_UTT2_FINAL = "all systems nominal"
_END_MARKER = {"is_final": True, "partial_text": {"text": "", "language": "en"}}


def _config(fake: FakeSidecar) -> dict:
    """Point the pipeline at the fake's socket with NO binary -> connect, never spawn."""
    return {
        "sidecar_bin": "",
        "socket": fake.socket_path,
        "model": "",
        "model_task": VOXTRAL_ID,
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


def _stream(ws, *, frames: int, gap: float = 0.02) -> None:
    for _ in range(frames):
        ws.send_bytes(_PCM)
        time.sleep(gap)


def _wait_for_flush(fake: FakeSidecar, count: int, *, timeout: float = 5.0) -> None:
    """Block until the fake has served `count` FLUSHes (the silence trigger fired)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fake.connections and fake.connections[0].flush_count >= count:
            return
        time.sleep(0.01)
    raise AssertionError(f"the sidecar never saw FLUSH #{count} -- endpointer did not fire")


def test_sub_threshold_utterance_emits_no_frames_and_next_utterance_transcribes(monkeypatch):
    """Zero frames for the empty utterance; the re-armed session serves the next one."""
    monkeypatch.setenv("VM_AUDIOCPP_SILENCE_MS", "200")  # fire fast so the test is quick
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "30")       # far above the silence window
    cfg._settings = None
    try:
        with FakeSidecar(audio_delta_script=[None] * _UTT1_FRAMES + list(_UTT2_DELTAS),
                         audio_final=None,
                         flush_script=[([], ""),                    # sub-threshold: EMPTY
                                       ([_END_MARKER], _UTT2_FINAL)]) as fake:
            client = TestClient(_app_for(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"

                # --- utterance 1: sub-threshold; silence -> FLUSH #1 (empty) ----------
                _stream(ws, frames=_UTT1_FRAMES)
                # Wait on the FAKE's state, not the WS: no frame may arrive, so the only
                # observable proof the empty flush happened is the sidecar's FLUSH count.
                _wait_for_flush(fake, 1)

                # --- utterance 2: normal; must transcribe through the re-armed session -
                _stream(ws, frames=len(_UTT2_DELTAS))
                frames: list[dict] = []
                for _ in range(50):
                    f = ws.receive_json()
                    assert f.get("type") == "transcription", f
                    frames.append(f)
                    if f["is_final"]:
                        break

            # The FIRST frame the client ever received belongs to utterance 2: the
            # sub-threshold utterance emitted ZERO frames -- had a dangling partial or
            # an empty final been emitted, it would sit at the head of this list.
            assert frames, "utterance 2 produced no frames -- session not healthy"
            assert frames[0]["is_final"] is False, (
                f"expected utterance 2's first partial as the first-ever frame, got "
                f"{frames[0]}")
            assert _UTT2_FINAL.startswith(frames[0]["text"].rstrip()), (
                f"first frame carries foreign text (a leaked utterance-1 frame?): "
                f"{frames[0]['text']!r}")
            for f in frames:
                assert f["text"].strip(), f"an empty frame leaked: {f}"

            # Exactly one final -- utterance 2's, with its NON-empty text intact (the
            # suppression is content-based and must not eat a real final).
            finals = [f for f in frames if f["is_final"]]
            assert len(finals) == 1, f"expected exactly one final, got {finals}"
            assert finals[0]["text"] == _UTT2_FINAL

            # Seg 0 was consumed by the suppressed empty flush; utterance 2 lands at 1.
            # (Monotonic, never rewound -- a skip is allowed, a reuse is not.)
            assert finals[0]["segment_index"] == 1, finals

            # Both utterances' audio and both FLUSHes reached the one sidecar session.
            conn = fake.connections[0]
            assert conn.audio_frames == _UTT1_FRAMES + len(_UTT2_DELTAS)
            assert conn.flush_count >= 2
    finally:
        cfg._settings = None
