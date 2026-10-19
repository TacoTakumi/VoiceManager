"""First-frame audio-format validation is shared and unchanged under audio.cpp.

Model-free and server-free (the CLAUDE.md invariant): a real FastAPI `/ws/stream`
driven by a `TestClient`. Reject cases close before any sidecar contact (the pipeline
defers connect to the first feed()), so they need no fake at all; the accept cases use
`tests/audiocpp_fake.py` over a genuine AF_UNIX socket. No GPU, no native binary, no
model download.

Standard audio contract: the audio.cpp backend consumes the SAME 16 kHz mono
16-bit PCM as every other engine, and the first-frame audio-format validation path is
UNCHANGED for it -- shared, not forked. That is structurally true because
`_is_valid_audio` runs in transport/websocket.py BEFORE the per-engine pipeline branch,
so no engine can fork it. These tests prove it behaviorally:

  * A bad first frame (odd byte count) under VM_STT_ENGINE=audiocpp is rejected
    IDENTICALLY to the default engine -- same error_type ("invalid_audio_format") and
    the same close code (1003) -- asserted by running the exact same probe through both
    engines and comparing.
  * A valid first frame under audiocpp is accepted and forwarded to the sidecar: raw
    16-bit PCM and a WAV container -- the two shapes `_is_valid_audio` accepts -- both
    pass validation and drive an interim partial out of the fake, proving the frame
    reached feed() rather than being dropped at the boundary.
"""
import time

import authkit  # test auth env; must be imported before voicemanager

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import voicemanager.config as cfg
from voicemanager.app import create_app
from voicemanager.session import SessionRegistry

from audiocpp_fake import FakeSidecar


# Even length -> valid raw 16-bit PCM (2-byte sample alignment); content is irrelevant
# to the fake (forwarded byte-for-byte). Odd length -> an incomplete sample == invalid.
_PCM = b"\x11\x22" * 512
_ODD = b"\x01\x02\x03"  # 3 bytes: not 2-byte aligned -> rejected, same as test_websocket
# Minimal WAV container: RIFF....WAVE magic -> the other branch `_is_valid_audio` accepts.
_WAV = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 20


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


def _audiocpp_app(fake: FakeSidecar):
    app = create_app()
    app.state.stt_engine_type = "audiocpp"
    app.state.audiocpp_config = _config(fake)
    app.state.session_registry = SessionRegistry()
    return app


def _default_app():
    """The default (whisper) engine app -- the 'other engines' baseline for parity.

    An invalid first frame is rejected before any model is touched (the validation runs
    ahead of the STT path), so no lifespan / model load is needed here either.
    """
    app = create_app()
    app.state.session_registry = SessionRegistry()
    return app


def _reject_first_frame(app, bad_frame: bytes) -> tuple[dict, int | None]:
    """Send `bad_frame` as the first binary frame; return (error_frame, close_code)."""
    client = TestClient(app)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"  # consume handshake
        ws.send_bytes(bad_frame)
        error = ws.receive_json()
        close_code = None
        try:
            ws.receive_json()  # the close follows the error frame
        except WebSocketDisconnect as disc:
            close_code = disc.code
    return error, close_code


def _first_interim(ws, first_frame: bytes, *, frames: int = 16, gap: float = 0.02) -> dict | None:
    """Stream an utterance starting with `first_frame`; return the first interim partial.

    The first send is the frame under test (raw PCM or a WAV container); the rest are
    plain PCM so the fake's scripted EVENT lands on a later drain (feed #N's EVENT is
    picked up on drain #N+1, the same cross-thread timing the frames test relies on).
    """
    ws.send_bytes(first_frame)
    time.sleep(gap)
    for _ in range(frames - 1):
        ws.send_bytes(_PCM)
        time.sleep(gap)
    for _ in range(50):
        frame = ws.receive_json()
        if frame.get("type") == "transcription" and frame["is_final"] is False:
            return frame
    return None


def test_bad_first_frame_rejected_identically_to_other_engines():
    """An invalid first frame closes the same way under audiocpp as under the default engine.

    Running the exact same odd-byte probe through both engines and comparing proves the
    validation path is SHARED, not forked: same error_type, same close code (1003).
    """
    with FakeSidecar(audio_events=["ignored"], flush_final="ignored") as fake:
        audiocpp_error, audiocpp_close = _reject_first_frame(_audiocpp_app(fake), _ODD)
    default_error, default_close = _reject_first_frame(_default_app(), _ODD)

    # The audio.cpp engine's reject behavior matches the other engines' byte-for-byte.
    assert audiocpp_error["type"] == "error"
    assert audiocpp_error["error_type"] == "invalid_audio_format"
    assert audiocpp_close == 1003
    assert audiocpp_error["error_type"] == default_error["error_type"]
    assert audiocpp_close == default_close


def test_valid_raw_pcm_first_frame_accepted_under_audiocpp(monkeypatch):
    """A valid raw 16-bit PCM first frame passes validation and reaches the sidecar.

    An interim partial coming back proves the frame was accepted at the boundary and
    forwarded to feed() (not dropped) -- i.e. the backend consumes the standard PCM the
    shared validation already accepts.
    """
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "30")  # drive directly, never via timeout
    cfg._settings = None
    try:
        with FakeSidecar(audio_events=["long range"], flush_final="long range scan") as fake:
            client = TestClient(_audiocpp_app(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"
                interim = _first_interim(ws, _PCM)
        assert interim is not None, "valid PCM first frame produced no interim partial"
        # startswith, not ==: the fake repeats its event per AUDIO frame and the
        # client accumulates deltas, so the interim may carry N repetitions (a race).
        # The claim is acceptance at the boundary, not text shape.
        assert interim["text"].startswith("long range")
    finally:
        cfg._settings = None


def test_valid_wav_first_frame_accepted_under_audiocpp(monkeypatch):
    """A WAV-container first frame -- the other shape `_is_valid_audio` accepts -- passes too.

    Mirrors test_websocket.py::test_ws_accepts_wav_header for the audiocpp engine: the
    WAV magic is accepted at the boundary and the session proceeds to transcribe.
    """
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "30")
    cfg._settings = None
    try:
        with FakeSidecar(audio_events=["scanning"], flush_final="scan complete") as fake:
            client = TestClient(_audiocpp_app(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"
                interim = _first_interim(ws, _WAV)
        assert interim is not None, "valid WAV first frame produced no interim partial"
        assert interim["text"].startswith("scanning")  # accumulation-tolerant, see above
    finally:
        cfg._settings = None
