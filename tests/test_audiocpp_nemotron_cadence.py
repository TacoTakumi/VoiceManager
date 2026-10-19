"""Mid-speech partial cadence over /ws/stream under Nemotron when deltas arrive during AUDIO.

Model-free and server-free (the CLAUDE.md invariant): a real FastAPI `/ws/stream`
driven by a `TestClient`, the real `AudioCppSTTPipeline` speaking sidecar/PROTOCOL.md
over a genuine AF_UNIX socket to `tests/audiocpp_fake.py`. No GPU, no native sidecar,
no model download.

The shipping Nemotron-3.5 build streams partial_text deltas while audio is being fed,
and decodes only the utterance tail at FLUSH. Those mid-AUDIO deltas must surface
exactly the way Voxtral's do: as cumulative is_final=False interim transcripts for
the current segment, before the utterance is flushed. The client is
model-agnostic, so nothing here is Voxtral-specific except that the wire model_task is
the Nemotron id.

The fake is scripted with one text DELTA per AUDIO frame (`audio_delta_script`) and a
FLUSH that carries NO partial text, only the FINAL (`flush_script`). The FINAL is
deliberately not the raw concatenation of the deltas (casing and punctuation differ), so
the committed text provably comes from the FINAL frame and not from the accumulator.

Why the pre-final assertion can only pass on AUDIO-time deltas: the FLUSH path emits at
most ONE is_final=False frame per utterance (the accumulated burst, just ahead of the
final). So >=2 partials, at least one of them shorter than the full accumulation, can
only have come from deltas absorbed while the audio was still streaming.
"""
import time

import authkit  # test auth env; must be imported before voicemanager

from fastapi.testclient import TestClient

import voicemanager.config as cfg
from voicemanager.app import create_app
from voicemanager.session import SessionRegistry

from audiocpp_fake import FakeSidecar

NEMOTRON_ID = "nemotron-3.5-asr-streaming"

# 512 samples of nonzero 16-bit PCM -- even length -> passes first-frame validation.
_PCM = b"\x11\x22" * 512

# One delta per AUDIO frame; the FLUSH contributes only the FINAL below.
_DELTAS = ["computer ", "raise ", "the ", "lights ", "in ", "engineering"]
_ACCUMULATED = "".join(_DELTAS)
_FINAL = "Computer, raise the lights in engineering."


def _config(fake: FakeSidecar) -> dict:
    """Point the pipeline at the fake's socket with NO binary -> connect, never spawn."""
    return {
        "sidecar_bin": "",
        "socket": fake.socket_path,
        "model": "",
        "model_task": NEMOTRON_ID,
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


def _stream_utterance(ws, *, frames: int, gap: float = 0.02) -> None:
    """Send an utterance as a burst of PCM frames, each gap << the silence window."""
    for _ in range(frames):
        ws.send_bytes(_PCM)
        time.sleep(gap)


def _recv_utterance(ws):
    """Collect is_final=False partials until this utterance's single final arrives."""
    partials: list[dict] = []
    for _ in range(50):
        f = ws.receive_json()
        assert f.get("type") == "transcription", f
        if f["is_final"]:
            return partials, f
        partials.append(f)
    raise AssertionError("no is_final=True final arrived within bound")


def test_nemotron_audio_time_deltas_become_cumulative_partials_before_final(monkeypatch):
    """Deltas scripted on AUDIO frames surface as growing cumulative interims, then one final."""
    monkeypatch.setenv("VM_AUDIOCPP_SILENCE_MS", "200")  # fire fast so the test is quick
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "30")       # far above the silence window
    cfg._settings = None
    try:
        with FakeSidecar(audio_delta_script=list(_DELTAS),
                         audio_final=None,
                         flush_script=[([], _FINAL)]) as fake:
            client = TestClient(_app_for(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                ready = ws.receive_json()
                assert ready["type"] == "ready" and ready["stt_engine"] == "audiocpp"

                # Deltas stream DURING the feed; the silence gap then triggers FLUSH.
                _stream_utterance(ws, frames=len(_DELTAS))
                partials, final = _recv_utterance(ws)

            # The sidecar really was driven as Nemotron, and flushed exactly once.
            conn = fake.connections[0]
            assert conn.hello["model_task"] == NEMOTRON_ID
            assert conn.flush_count == 1

            texts = [f["text"] for f in partials]
            cumulative = ["".join(_DELTAS[:k]) for k in range(1, len(_DELTAS) + 1)]

            # At least two growing interims ahead of the final.
            assert len(texts) >= 2, f"expected >=2 interim partials before the final, got {texts}"
            assert len(set(texts)) >= 2, f"partials never grew: {texts}"

            # Each interim is the concatenation of the deltas so far -- replace semantics
            # over the whole transcript, never a lone delta or foreign text.
            for t in texts:
                assert t in cumulative, (
                    f"partial {t!r} is not a concatenation of the deltas so far {cumulative}")
            for earlier, later in zip(texts, texts[1:]):
                assert later.startswith(earlier), f"partials must grow monotonically: {texts}"

            # At least one interim is shorter than the full accumulation, which the
            # FLUSH-time partial never is: it was surfaced while audio was streaming.
            assert any(len(t) < len(_ACCUMULATED) for t in texts), (
                f"no partial arrived before the flush: {texts}")

            # Exactly one is_final=True final, carrying the FLUSH's FINAL text.
            assert final["text"] == _FINAL
            assert final["segment_index"] == 0
            for f in partials:
                assert f["segment_index"] == final["segment_index"], f
    finally:
        cfg._settings = None
