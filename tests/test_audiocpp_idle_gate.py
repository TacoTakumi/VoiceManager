"""Idle-feed gating for the audio.cpp backend.

Model-free / server-free (the CLAUDE.md invariant): a real FastAPI `/ws/stream` driven
by a `TestClient`, the real `AudioCppSTTPipeline` speaking sidecar/PROTOCOL.md over a
genuine AF_UNIX socket to `tests/audiocpp_fake.py`. The silero NEURAL NET is replaced
by the amplitude fake from the acoustic-endpoint tests, running under the REAL `silero_vad.
VADIterator` -- so the classification the gate keys on is the production machinery.

The point: a continuously-streaming mic client (HermesConsole) sends silence as
FRAMES of quiet audio between utterances. Without gating every such frame is billed to
the 4B Voxtral decoder. With the gate:

  * frames the existing acoustic endpointer classifies as inter-utterance silence
    (outside silero's start..end span) are NOT fed to the sidecar session -- and,
    fed nothing, the session cannot produce empty inter-utterance flushes either,
    so consecutive utterances get consecutive segment indexes;
  * a rolling pre-roll (VM_AUDIOCPP_ONSET_PREROLL_MS) is replayed into the session
    at speech onset, so the attack silero needed a chunk to confirm is not clipped:
    a leading-silence clip transcribes exactly like a feed-everything one;
  * classification runs BEFORE the feed decision;
  * the gate is engine-scoped and toggleable: VM_AUDIOCPP_IDLE_GATE=0 restores
    feed-everything, and with no VAD armed the gate is inert (frame-gap-only
    configs keep their exact pre-gate behavior -- every other audiocpp test file
    runs without a vad_model_factory and pins that).
"""
import time

import authkit  # test auth env; must be imported before voicemanager

import torch
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import voicemanager.config as cfg
from voicemanager.app import create_app
from voicemanager.session import SessionRegistry

from audiocpp_fake import FakeSidecar

VOXTRAL_ID = "voxtral-mini-4b-realtime-asr-streaming"

# One silero-sized chunk (512 samples at 16 kHz = 32 ms) per frame, aligning VAD
# windows 1:1 with frames. Loud alternating bytes read as speech to the amplitude
# fake; zeros read as silence.
_SPEECH = b"\x11\x22" * 512
_SILENCE = b"\x00\x00" * 512

_SILENCE_MS = "200"       # test-fast endpoint window (6.25 chunks)
_PREROLL_MS = "192"       # 6 chunks of pre-roll = 6144 bytes
_N_SPEECH = 8
_N_SILENCE = 20           # 640 ms of silent CONTENT, comfortably past the window


class FakeVadModel:
    """The amplitude stand-in for the silero net under the real VADIterator."""

    def __call__(self, x, sr):
        return torch.tensor(1.0 if float(x.abs().max()) > 0.01 else 0.0)

    def reset_states(self):
        pass


def _config(fake: FakeSidecar) -> dict:
    return {
        "sidecar_bin": "",
        "socket": fake.socket_path,
        "model": "",
        "model_task": VOXTRAL_ID,
        "backend": "cpu",
        "device": 0,
        "threads": 4,
        "assets_root": "",
        "vad_model_factory": FakeVadModel,
    }


def _app_for(fake: FakeSidecar):
    app = create_app()
    app.state.stt_engine_type = "audiocpp"
    app.state.audiocpp_config = _config(fake)
    app.state.session_registry = SessionRegistry()
    return app


def _collect_until_close(ws) -> list[dict]:
    """Drain transcription frames until the session-timeout close (set low per test)."""
    frames: list[dict] = []
    try:
        while True:
            f = ws.receive_json()
            if f.get("type") == "error":
                break
            if f.get("type") == "transcription":
                frames.append(f)
    except WebSocketDisconnect:
        pass
    return frames


def _env(monkeypatch, **extra):
    monkeypatch.setenv("VM_AUDIOCPP_SILENCE_MS", _SILENCE_MS)
    monkeypatch.setenv("VM_AUDIOCPP_ONSET_PREROLL_MS", _PREROLL_MS)
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "2")
    for key, value in extra.items():
        monkeypatch.setenv(key, value)
    cfg._settings = None


def test_inter_utterance_silence_reaches_the_session_as_zero_feeds(monkeypatch):
    """Silence outside an utterance is never fed; speech and its trailing window are.

    Leading silence: the pipeline connects lazily on the first feed, so gated leading
    silence means NO sidecar connection at all until speech starts. Inter-utterance
    silence: after the acoustic endpoint closes utterance one, the remaining silence
    frames (and a whole extra burst of them) must not grow the fed-bytes count by one
    byte -- zero feed calls during scripted inter-utterance silence.
    """
    _env(monkeypatch)
    try:
        with FakeSidecar(audio_events=[], audio_final=None,
                         flush_script=[([], "utterance one")]) as fake:
            client = TestClient(_app_for(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"

                # --- leading silence: gated, so not even a lazy connect ------------
                for _ in range(10):
                    ws.send_bytes(_SILENCE)
                    time.sleep(0.01)
                time.sleep(0.2)
                assert not fake.connections, (
                    "leading silence was fed to the sidecar (it even dialed the "
                    "session) -- the idle gate is not gating")

                # --- utterance one, then its endpointing silence -------------------
                for _ in range(_N_SPEECH):
                    ws.send_bytes(_SPEECH)
                    time.sleep(0.01)
                for _ in range(_N_SILENCE):
                    ws.send_bytes(_SILENCE)
                    time.sleep(0.01)

                # Wait until the acoustic trigger has flushed utterance one.
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    if fake.connections and fake.connections[0].flush_count >= 1:
                        break
                    time.sleep(0.02)
                assert fake.connections, "speech never reached the sidecar"
                conn = fake.connections[0]
                assert conn.flush_count >= 1, "the acoustic endpoint never flushed"
                fed_after_utterance = len(conn.audio_bytes)

                # --- scripted inter-utterance silence: ZERO further feeds ----------
                for _ in range(_N_SILENCE):
                    ws.send_bytes(_SILENCE)
                    time.sleep(0.01)
                time.sleep(0.3)
                assert len(conn.audio_bytes) == fed_after_utterance, (
                    f"inter-utterance silence was fed to the decoder: "
                    f"{len(conn.audio_bytes) - fed_after_utterance} extra bytes")

                frames = _collect_until_close(ws)

            # The utterance itself transcribed normally through the gate.
            finals = [f for f in frames if f["is_final"]]
            assert [f["text"] for f in finals] == ["utterance one"], frames
            # Every speech byte reached the session, contiguously (no mid-speech gating).
            assert _SPEECH * _N_SPEECH in bytes(conn.audio_bytes), (
                "speech bytes were clipped or reordered on their way to the session")
    finally:
        cfg._settings = None


def test_onset_preroll_preserves_leading_speech_and_segments_stay_consecutive(monkeypatch):
    """A leading-silence clip transcribes like a feed-everything clip, twice over.

    Two utterances separated by a gated silence gap: each utterance's speech bytes
    arrive intact (the pre-roll covers silero's onset-confirmation latency, so nothing
    leading is clipped), each yields its scripted final, and the segment indexes are
    CONSECUTIVE -- the gate feeds no inter-utterance silence, so no empty flush can
    consume an index between them (the live seg-skip the pre-gate run showed).
    """
    _env(monkeypatch)
    try:
        with FakeSidecar(audio_events=[], audio_final=None,
                         flush_script=[([], "utterance one"),
                                       ([], "utterance two")]) as fake:
            client = TestClient(_app_for(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"

                for _ in range(10):                     # leading silence (gated)
                    ws.send_bytes(_SILENCE)
                    time.sleep(0.01)
                for _ in range(_N_SPEECH):              # utterance one
                    ws.send_bytes(_SPEECH)
                    time.sleep(0.01)
                for _ in range(_N_SILENCE):             # endpoint + inter-utterance gap
                    ws.send_bytes(_SILENCE)
                    time.sleep(0.01)
                for _ in range(_N_SPEECH):              # utterance two
                    ws.send_bytes(_SPEECH)
                    time.sleep(0.01)
                for _ in range(_N_SILENCE):             # endpoint again
                    ws.send_bytes(_SILENCE)
                    time.sleep(0.01)

                frames = _collect_until_close(ws)

            finals = [f for f in frames if f["is_final"]]
            assert [f["text"] for f in finals] == ["utterance one", "utterance two"], frames
            assert [f["segment_index"] for f in finals] == [0, 1], (
                "segment indexes skipped -- an inter-utterance empty flush got through "
                f"the gate: {[(f['text'], f['segment_index']) for f in finals]}")

            # Both utterances' speech reached the session intact and in order.
            fed = bytes(fake.connections[0].audio_bytes)
            first = fed.find(_SPEECH * _N_SPEECH)
            second = fed.find(_SPEECH * _N_SPEECH, first + len(_SPEECH * _N_SPEECH))
            assert first != -1, "utterance one's speech was clipped by the gate"
            assert second != -1, "utterance two's speech was clipped by the gate"
    finally:
        cfg._settings = None


def test_gate_off_restores_feed_everything(monkeypatch):
    """VM_AUDIOCPP_IDLE_GATE=0: every frame is fed, exactly the pre-gate behavior."""
    _env(monkeypatch, VM_AUDIOCPP_IDLE_GATE="0")
    try:
        with FakeSidecar(audio_events=[], audio_final=None,
                         flush_script=[([], "utterance one")]) as fake:
            client = TestClient(_app_for(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"
                sent = 0
                for _ in range(10):
                    ws.send_bytes(_SILENCE)
                    sent += 1
                    time.sleep(0.01)
                for _ in range(_N_SPEECH):
                    ws.send_bytes(_SPEECH)
                    sent += 1
                    time.sleep(0.01)
                for _ in range(_N_SILENCE):
                    ws.send_bytes(_SILENCE)
                    sent += 1
                    time.sleep(0.01)
                frames = _collect_until_close(ws)

            conn = fake.connections[0]
            assert conn.audio_frames == sent, (
                f"gate off must feed everything: sent {sent}, fed {conn.audio_frames}")
            finals = [f for f in frames if f["is_final"]]
            assert [f["text"] for f in finals] == ["utterance one"], frames
    finally:
        cfg._settings = None
