"""Acoustic-silence endpointing for the audio.cpp STT backend.

Model-free / server-free (the CLAUDE.md invariant): a real FastAPI `/ws/stream` driven
by a `TestClient`, the real `AudioCppSTTPipeline` speaking sidecar/PROTOCOL.md over a
genuine AF_UNIX socket to `tests/audiocpp_fake.py`. The silero NEURAL NET is replaced by
a tiny amplitude fake, but it runs under the REAL `silero_vad.VADIterator`, so the
threshold/hysteresis/min-silence machinery a live server uses is the machinery under test.

The defect this file pins down (live-confirmed 2026-08-04 against HermesConsole): the
frame-gap endpoint trigger fires only on a GAP in inbound frames, but a real mic client
streams CONTINUOUSLY -- end-of-speech arrives as frames of near-silent audio, never as
an absence of frames -- so the FLUSH never fired and a listening client got nothing.
These tests stream HermesConsole-style: frames queued back-to-back with no wall-clock
gap the frame-gap trigger could ever see.

  * test_acoustic_silence_splits_back_to_back_utterances -- the acoustic trigger: two
    utterances (speech frames, then silence-CONTENT frames) queued with NO gap must
    yield two separate partial+final bursts at advancing segment_index. Only a
    content-based (acoustic) trigger can split them: the receive loop never idles
    between them, so the frame-gap path cannot fire in between.
  * test_frame_gap_fallback_still_endpoints_when_client_stops -- the frame-gap behaviour
    must NOT regress with the endpointer active: a clip-sending client that simply
    stops transmitting still gets its final via the frame-gap trigger.
  * test_endpointer_inference_runs_off_the_event_loop -- the engine's off-loop constraint:
    the per-frame VAD inference must not block the loop the WS handler runs on.
"""
import asyncio
import time

import authkit  # test auth env; must be imported before voicemanager

import torch
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import voicemanager.config as cfg
from voicemanager.app import create_app
from voicemanager.session import SessionRegistry

from audiocpp_fake import FakeSidecar

# One silero-sized chunk (512 samples at 16 kHz = 32 ms) per frame, so the VAD windows
# align 1:1 with frames. Loud alternating bytes read as speech to the amplitude fake;
# zeros read as silence.
_SPEECH = b"\x11\x22" * 512
_SILENCE = b"\x00\x00" * 512

# 200 ms silence window (test-fast) = 6.25 chunks; 20 silence frames = 640 ms of
# silent AUDIO CONTENT, comfortably past the window while frames keep arriving.
_SILENCE_MS = "200"
_N_SPEECH = 8
_N_SILENCE = 20


class FakeVadModel:
    """Stands in for the silero net under the REAL VADIterator: loud chunk -> speech.

    VADIterator needs exactly two things of the model -- `model(x, sr)` returning a
    0-dim probability tensor, and `reset_states()`. Amplitude is a perfect stand-in
    here because the test's frames are constructed loud-vs-zero; every downstream
    decision (threshold, hysteresis, min_silence_duration_ms) is the real iterator's.
    """

    def __call__(self, x, sr):
        return torch.tensor(1.0 if float(x.abs().max()) > 0.01 else 0.0)

    def reset_states(self):
        pass


class LoopProbeVadModel(FakeVadModel):
    """FakeVadModel that also records WHICH kind of thread each inference ran on.

    `asyncio.get_running_loop()` succeeds only on a thread with a loop actually running
    on it. So calling it from inside the VAD inference answers the off-loop question exactly:
    if it returns a loop, this inference is executing ON the event loop and is blocking
    every other session on that loop for its duration; if it raises RuntimeError, the
    inference has been handed to a worker thread. No timing, no sleeping -- a direct
    observation of the property, so it cannot pass by being fast.
    """

    on_loop = 0
    off_loop = 0

    @classmethod
    def reset_counts(cls) -> None:
        cls.on_loop = 0
        cls.off_loop = 0

    def __call__(self, x, sr):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            type(self).off_loop += 1
        else:
            type(self).on_loop += 1
        return super().__call__(x, sr)


def _config(fake: FakeSidecar, vad_factory=FakeVadModel) -> dict:
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
        # The production contract: a zero-arg factory, called once per connection so
        # concurrent sessions never share VAD state (the class itself is that factory).
        "vad_model_factory": vad_factory,
    }


def _app_for(fake: FakeSidecar, vad_factory=FakeVadModel):
    app = create_app()
    app.state.stt_engine_type = "audiocpp"
    app.state.audiocpp_config = _config(fake, vad_factory)
    app.state.session_registry = SessionRegistry()
    return app


def _collect_until_close(ws) -> list[dict]:
    """Drain every transcription frame until the session's timeout close.

    Bounded by VM_SESSION_TIMEOUT (set low by each test): the server closes idle
    sessions with an error frame, so a missing final fails an assertion rather than
    hanging the test on a blocking receive.
    """
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


def test_acoustic_silence_splits_back_to_back_utterances(monkeypatch):
    """Silence CONTENT while frames keep flowing must flush each utterance.

    Two utterances are queued back-to-back with no wall-clock gap anywhere: speech
    frames, silence frames, and immediately the next utterance's speech frames. The
    receive loop drains them as one uninterrupted backlog, so the frame-gap trigger
    can never fire between the utterances -- if two separate finals come back, at
    advancing segment_index and with each utterance's own text, the acoustic
    (content-based) trigger split them. This is the frame pattern HermesConsole
    produces and the exact shape that emitted NOTHING before the acoustic trigger.
    """
    monkeypatch.setenv("VM_AUDIOCPP_SILENCE_MS", _SILENCE_MS)
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "2")
    cfg._settings = None
    try:
        with FakeSidecar(audio_events=[], audio_final=None, flush_script=[
            (["utterance", "utterance one"], "utterance one"),
            (["utterance two"], "utterance two"),
        ]) as fake:
            client = TestClient(_app_for(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                ready = ws.receive_json()
                assert ready["type"] == "ready" and ready["stt_engine"] == "audiocpp"

                # Both utterances up front, zero inter-frame wall-clock gap.
                for _ in range(_N_SPEECH):
                    ws.send_bytes(_SPEECH)
                for _ in range(_N_SILENCE):
                    ws.send_bytes(_SILENCE)
                for _ in range(_N_SPEECH):
                    ws.send_bytes(_SPEECH)
                for _ in range(_N_SILENCE):
                    ws.send_bytes(_SILENCE)

                frames = _collect_until_close(ws)

        finals = [f for f in frames if f["is_final"]]
        assert [f["text"] for f in finals] == ["utterance one", "utterance two"], (
            f"expected one final per utterance from the acoustic trigger, got "
            f"{[(f['text'], f['segment_index']) for f in finals]}")
        assert [f["segment_index"] for f in finals] == [0, 1]

        # The finalize burst surfaces >=1 is_final=False partial per utterance, each at
        # its own utterance's (pre-reset) segment_index.
        for seg in (0, 1):
            partials = [f for f in frames if not f["is_final"] and f["segment_index"] == seg]
            assert len(partials) >= 1, f"no partial for utterance at segment {seg}: {frames}"

        # The speculative endpoint-hold is nemotron-only and must not leak here.
        for f in frames:
            assert f.get("endpoint", False) is False, f
    finally:
        cfg._settings = None


def test_frame_gap_fallback_still_endpoints_when_client_stops(monkeypatch):
    """A client that stops sending entirely still gets its final (frame-gap must not regress).

    Speech frames only -- no silent-content tail -- then transmission stops. The
    acoustic trigger has no silence to see, so only the frame-gap path can flush; the
    final arriving proves the fallback survived the endpointer's arrival.
    """
    monkeypatch.setenv("VM_AUDIOCPP_SILENCE_MS", _SILENCE_MS)
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "2")
    cfg._settings = None
    try:
        with FakeSidecar(audio_events=[], audio_final=None,
                         flush_events=["long range"],
                         flush_final="long range scan complete") as fake:
            client = TestClient(_app_for(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                ready = ws.receive_json()
                assert ready["type"] == "ready"

                for _ in range(_N_SPEECH):
                    ws.send_bytes(_SPEECH)
                    time.sleep(0.02)  # << the silence window: no mid-utterance flush
                # Stop transmitting -- the frame gap is the only endpoint signal left.

                frames = _collect_until_close(ws)

        finals = [f for f in frames if f["is_final"]]
        assert [f["text"] for f in finals] == ["long range scan complete"], frames
        assert len([f for f in frames if not f["is_final"]]) >= 1, frames
    finally:
        cfg._settings = None


def test_endpointer_inference_runs_off_the_event_loop(monkeypatch):
    """The per-frame VAD inference must not execute on the event loop.

    Off-loop inference is this engine's explicit constraint, and the endpointer is real
    torch inference on EVERY 32 ms chunk: on the loop it serializes against every other
    session's I/O, and the cost multiplies with concurrency on a slow CPU. The probe
    model asks `asyncio.get_running_loop()` from inside the inference, so this asserts
    the property directly rather than inferring it from a duration.

    The utterance is still endpointed acoustically, proving the move off-loop preserved
    per-frame ordering: each frame's decision is applied before the next frame is read,
    so the trailing-silence trigger fires exactly as it did on the loop.
    """
    monkeypatch.setenv("VM_AUDIOCPP_SILENCE_MS", _SILENCE_MS)
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "2")
    cfg._settings = None
    LoopProbeVadModel.reset_counts()
    try:
        with FakeSidecar(audio_events=[], audio_final=None, flush_script=[
            (["utterance", "utterance one"], "utterance one"),
        ]) as fake:
            client = TestClient(_app_for(fake, LoopProbeVadModel))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"

                for _ in range(_N_SPEECH):
                    ws.send_bytes(_SPEECH)
                for _ in range(_N_SILENCE):
                    ws.send_bytes(_SILENCE)

                frames = _collect_until_close(ws)

        assert LoopProbeVadModel.off_loop + LoopProbeVadModel.on_loop > 0, (
            "the endpointer never ran -- the probe model was not wired in")
        assert LoopProbeVadModel.on_loop == 0, (
            f"{LoopProbeVadModel.on_loop} of "
            f"{LoopProbeVadModel.on_loop + LoopProbeVadModel.off_loop} VAD inferences ran "
            "ON the event loop; the engine requires them off it")

        # Ordering preserved: the acoustic trigger still endpointed the utterance.
        finals = [f for f in frames if f["is_final"]]
        assert [f["text"] for f in finals] == ["utterance one"], (
            f"moving the endpointer off-loop broke the acoustic trigger: {frames}")
    finally:
        cfg._settings = None
