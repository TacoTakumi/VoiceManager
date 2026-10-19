"""Mid-speech partial cadence over /ws/stream under Voxtral.

Model-free and server-free (the CLAUDE.md invariant): a real FastAPI `/ws/stream`
driven by a `TestClient`, the real `AudioCppSTTPipeline` speaking sidecar/PROTOCOL.md
over a genuine AF_UNIX socket to `tests/audiocpp_fake.py`. No GPU, no native sidecar,
no model download.

The fake is scripted to the measured VOXTRAL shape: genuine token-incremental
streaming -- one text DELTA per AUDIO frame while speech is being fed
(`audio_delta_script`) -- and at FLUSH only a contentless is_final=true end-marker
event ahead of the FINAL (`flush_script`). This is the exact opposite of the
finalize-only nemotron shape of the earlier pins (nothing during AUDIO, the whole
burst at FLUSH), and it is what makes the
"partials-only-at-finalize fails" clause real: the FLUSH here contributes NO partial
text, so the >=2 growing partials asserted below can only have come from the deltas
emitted DURING the AUDIO feed.

What must hold per utterance, with VM_AUDIOCPP_MODEL_TASK=<voxtral id> and
end-of-utterance driven by the EXISTING endpointer trigger (the silence gap --
no Voxtral-specific machinery, since the model has no internal endpointing):

  * at least 2 growing is_final=False partials, each the cumulative
    transcript-so-far (replace semantics);
  * then exactly one is_final=True final carrying a monotonic segment_index;
  * the end-marker event produces no partial frame (honest is_final);
  * a second utterance on the same connection starts a FRESH accumulation
    (no utterance-1 text bleeding into utterance-2 partials).
"""
import time

import authkit  # test auth env; must be imported before voicemanager

from fastapi.testclient import TestClient

import voicemanager.config as cfg
from voicemanager.app import create_app
from voicemanager.session import SessionRegistry

from audiocpp_fake import FakeSidecar

VOXTRAL_ID = "voxtral-mini-4b-realtime-asr-streaming"

# 512 samples of nonzero 16-bit PCM -- even length -> passes first-frame validation.
_PCM = b"\x11\x22" * 512

# One delta per AUDIO frame, the voxtral streaming shape: utterance 1 pops the first
# eight entries, utterance 2 (same connection, re-armed session) pops the rest.
_UTT1_DELTAS = ["uss ", "enterprise ", "to ", "any ", "federation ", "ship ", "in ", "range"]
_UTT2_DELTAS = ["please ", "respond ", "on ", "this ", "frequency ", "priority ", "one ", "now"]
_UTT1_FINAL = "uss enterprise to any federation ship in range"
_UTT2_FINAL = "please respond on this frequency priority one now"

# The voxtral finalize shape on FLUSH: a contentless is_final=true end-marker event
# (suppressed), then the FINAL. No text deltas here -- see the module docstring.
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


def _stream_utterance(ws, *, frames: int, gap: float = 0.02) -> None:
    """Send an utterance as a burst of PCM frames, each gap << the silence window."""
    for _ in range(frames):
        ws.send_bytes(_PCM)
        time.sleep(gap)


def _recv_utterance(ws):
    """Collect is_final=False partials until this utterance's single final arrives.

    Leaves the connection OPEN, so a second utterance can follow on the same socket.
    """
    partials: list[dict] = []
    for _ in range(50):
        f = ws.receive_json()
        assert f.get("type") == "transcription", f
        if f["is_final"]:
            return partials, f
        partials.append(f)
    raise AssertionError("no is_final=True final arrived within bound")


def _assert_growing_cumulative(partials: list[dict], full_text: str) -> None:
    """>=2 growing partials, each the transcript-so-far: a strictly-extending prefix
    chain of the utterance's accumulated text (replace semantics, never a lone delta)."""
    texts = [f["text"] for f in partials]
    assert len(texts) >= 2, f"expected >=2 mid-speech partials, got {texts}"
    for t in texts:
        assert full_text.startswith(t.rstrip()) or full_text.startswith(t), (
            f"partial {t!r} is not a prefix of the accumulated transcript {full_text!r} "
            "-- a lone delta or foreign text leaked in")
    for earlier, later in zip(texts, texts[1:]):
        assert later.startswith(earlier), f"partials must grow monotonically: {texts}"
    assert len(texts[-1]) > len(texts[0]), f"partials never grew: {texts}"


def test_voxtral_cadence_growing_partials_then_one_final_per_utterance(monkeypatch):
    """The cadence frame pattern over /ws/stream, two utterances on one connection."""
    monkeypatch.setenv("VM_AUDIOCPP_SILENCE_MS", "200")  # fire fast so the test is quick
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "30")       # far above the silence window
    cfg._settings = None
    try:
        with FakeSidecar(audio_delta_script=list(_UTT1_DELTAS + _UTT2_DELTAS),
                         audio_final=None,
                         flush_script=[([_END_MARKER], _UTT1_FINAL),
                                       ([_END_MARKER], _UTT2_FINAL)]) as fake:
            client = TestClient(_app_for(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                ready = ws.receive_json()
                assert ready["type"] == "ready" and ready["stt_engine"] == "audiocpp"

                # --- utterance 1: deltas stream DURING the feed; silence -> FLUSH ------
                _stream_utterance(ws, frames=len(_UTT1_DELTAS))
                partials1, final1 = _recv_utterance(ws)

                # --- utterance 2: same connection, re-armed session, fresh text -------
                _stream_utterance(ws, frames=len(_UTT2_DELTAS))
                partials2, final2 = _recv_utterance(ws)

            # >=2 growing cumulative partials per utterance, from the AUDIO deltas alone
            # (the FLUSH script carries no partial text -- partials-only-at-finalize
            # would leave these lists empty and fail here).
            _assert_growing_cumulative(partials1, _UTT1_FINAL)
            _assert_growing_cumulative(partials2, _UTT2_FINAL)

            # Exactly one is_final=True final per utterance with the committed text.
            assert final1["text"] == _UTT1_FINAL
            assert final2["text"] == _UTT2_FINAL

            # Monotonic segment_index across utterances on the one connection.
            assert final1["segment_index"] == 0
            assert final2["segment_index"] == 1

            # Fresh accumulation per utterance: no utterance-1 text in utterance-2
            # partials (the re-armed session starts a new transcript).
            for f in partials2:
                assert "enterprise" not in f["text"], (
                    f"utterance-1 text bled into utterance 2: {f['text']!r}")

            # The contentless end-marker produced no partial frame: every partial has
            # non-empty text, and none arrived AFTER its utterance's final.
            for f in partials1 + partials2:
                assert f["text"].strip(), f"an empty/marker partial surfaced: {f}"

            # The speculative endpoint-hold is nemotron-only; it must not leak.
            for f in partials1 + partials2 + [final1, final2]:
                assert f.get("endpoint", False) is False, f
    finally:
        cfg._settings = None
