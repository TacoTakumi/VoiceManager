"""Prove the whole turn-taking surface is additive and opt-in.

A *legacy* client -- one that streams STT and drives ``tts_stream`` to ``done``, and
that never sends ``flush``, ``turn_start`` or ``tts_cancel`` -- must see exactly
today's behaviour: no new frame TYPE is emitted unsolicited, no existing field
changes meaning, no field becomes required, and every additive field carries a
default that reproduces the pre-feature wire (``flush=false``, ``turn_id=0``,
``reason='complete'``, ``endpoint=false``). There is no negotiated mode, capability
flag, or client-driven state machine.

Two complementary proofs:

  * ``test_legacy_session_trace_is_additive_and_opt_in`` records the end-to-end frame
    trace of such a session. The streaming-STT leg runs through the REAL WebSocket
    handler (a whisper fake stands in for the model); the ``tts_stream``-to-done leg
    runs through the REAL ``TTSSession.feed_stream`` the handler itself calls (a
    streaming-TTS engine fake stands in for the model). The combined trace must
    contain no ``turn_started`` frame, no type outside the pre-project message set,
    and every new field at its default.

  * the structural checks pin that the project added ONLY the two turn message types,
    that none of the additive fields became required, and that ``turn_started`` is
    constructed in exactly one place -- the ``turn_start`` branch -- so it can never
    be emitted unsolicited.

Model-free and server-free: ``TestClient(app)`` WITHOUT ``with client:`` skips the
lifespan, and both the STT pipeline and the TTS engine are fakes, so no model loads,
no GPU, no network, no audio hardware (the automated-suite contract in conftest).
"""
import asyncio
import json
import pathlib

import authkit

import numpy as np
from fastapi.testclient import TestClient

from voicemanager.app import create_app
from voicemanager.pipeline.tts import TTSPipeline, TTSStreamPipeline
from voicemanager.pipeline.tts_session import TTSSession
from voicemanager.protocol import (
    MessageType, TranscriptionMessage, AudioEndMessage, TTSStreamMessage,
)
from voicemanager.session import SessionRegistry

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "voicemanager"

# The WebSocket message types that existed BEFORE this project -- captured from the
# enum at git 29e24e6^, the commit immediately before turn_start/turn_started were
# added. A legacy session may only ever see these; they are today's behaviour.
_LEGACY_MESSAGE_TYPES = frozenset({
    "ready", "transcription", "audio_chunk", "audio_end",
    "error", "tts", "audio_start", "tts_cancel", "tts_stream",
})
# The only additions this project makes to the enum. turn_start is client->server;
# turn_started is the sole new SERVER-emitted type, and only as a turn_start ack.
_TURN_MESSAGE_TYPES = frozenset({"turn_start", "turn_started"})

# 512 bytes: a valid raw 16-bit PCM frame, enough to trigger one STT feed().
_ONE_CHUNK = b"\x00\x00" * 256


# --- test doubles (model-free, mirroring the project's proven fakes) --------

class _FakeWhisperSeg:
    """A faster-whisper-like segment carrying the fields _send_transcription reads."""

    def __init__(self, text: str):
        self.text = text
        self.avg_logprob = -0.1
        self.start = 0.0
        self.end = 1.0


class _FakeWhisperInfo:
    language = "en"


def _fake_whisper_factory(inferences):
    """Drop-in for STTPipeline (whisper): feed() reports work while the script remains;
    each run_inference() pops one inference's worth of segment texts. No VAD/model."""

    class _FakeWhisper:
        def __init__(self, **kwargs):
            self._inferences = list(inferences)

        def feed(self, data: bytes) -> bool:
            return bool(self._inferences)

        async def run_inference(self, loop):
            texts = self._inferences.pop(0) if self._inferences else []
            return [_FakeWhisperSeg(t) for t in texts], _FakeWhisperInfo(), 0.01

        def dispose(self):
            pass

    return _FakeWhisper


class _FakeStreamEngine:
    """Streaming TTS engine stub: yields one 8-sample chunk per sentence, no executor."""

    is_batch = False

    @property
    def sample_rate(self) -> int:
        return 24000

    async def stream_synthesize(self, text: str, **kwargs):
        yield np.zeros(8, dtype=np.float32)


class _FakeWebSocket:
    """Records the outbound frame sequence as ("text", dict) | ("bytes", bytes) tuples."""

    def __init__(self):
        self.frames: list[tuple[str, object]] = []

    async def send_text(self, data: str) -> None:
        self.frames.append(("text", json.loads(data)))

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append(("bytes", data))


# --- recording a legacy session --------------------------------------------

def _record_stt_leg(monkeypatch) -> list[tuple[str, object]]:
    """Streaming-STT leg through the REAL WS handler: handshake + one transcription.

    Returns the server->client frames as ("text", dict) tuples, exactly as the client
    saw them on the wire.
    """
    import voicemanager.config as cfg
    cfg._settings = None
    monkeypatch.setattr(
        "voicemanager.transport.websocket.STTPipeline",
        _fake_whisper_factory([["all stop"]]),
    )
    try:
        app = create_app()
        app.state.stt_engine_type = "whisper"      # the default, legacy STT engine
        app.state.session_registry = SessionRegistry()
        client = TestClient(app)                    # no `with client:` -> no lifespan

        frames: list[tuple[str, object]] = []
        with authkit.ws_connect(client, "/ws/stream") as ws:
            frames.append(("text", ws.receive_json()))   # ready handshake
            ws.send_bytes(_ONE_CHUNK)
            frames.append(("text", ws.receive_json()))   # the one transcription final
        return frames
    finally:
        cfg._settings = None


def _record_tts_stream_leg() -> tuple[list[tuple[str, object]], TTSStreamMessage]:
    """tts_stream-to-done leg through the REAL TTSSession the handler drives.

    A legacy client builds a plain ``tts_stream`` frame (no flush field), sends the
    whole reply and closes it with done=true. Returns the recorded server frames and
    the exact frame the legacy client put on the wire (so flush=false is observable).
    """
    engine = _FakeStreamEngine()
    ws = _FakeWebSocket()
    # The frame a legacy client sends: request_id + text + done, nothing else. flush
    # is never set, so it defaults false -- asserted by the caller.
    client_frame = TTSStreamMessage(request_id="r1", text="Hello there.", done=True)

    async def run():
        tts = TTSPipeline(engine=engine, executor=None)
        session = TTSSession(
            websocket=ws, tts_pipeline=tts, tts_stream_pipeline=TTSStreamPipeline(tts),
        )
        # Exactly what the handler's tts_stream branch calls (transport/websocket.py).
        await session.feed_stream(
            chunk=client_frame.text,
            done=client_frame.done,
            request_id=client_frame.request_id,
            voice=client_frame.voice,
            flush=client_frame.flush,
        )
        await asyncio.wait_for(session.wait_idle(), 2.0)

    asyncio.run(run())
    return list(ws.frames), client_frame


def _text_frames(trace) -> list[dict]:
    return [p for kind, p in trace if kind == "text"]


def _types(trace) -> list[str]:
    return [p["type"] for kind, p in trace if kind == "text"]


# --- the headline proof: a recorded legacy-session frame trace --------------

def test_legacy_session_trace_is_additive_and_opt_in(monkeypatch):
    stt_frames = _record_stt_leg(monkeypatch)
    tts_frames, client_frame = _record_tts_stream_leg()
    # One ordered trace of the whole legacy session: streaming STT, then tts_stream
    # to done. Both legs came out of production emit paths.
    trace = stt_frames + tts_frames
    types = _types(trace)

    # Sanity: we actually exercised streaming STT + tts_stream to done.
    assert "ready" in types and "transcription" in types
    assert types.count("audio_start") == 1 and types.count("audio_end") == 1
    assert any(kind == "bytes" for kind, _ in trace), "no PCM audio flowed"

    # No new frame TYPE is emitted unsolicited -- the turn ack never appears,
    # and neither does the (client-only) turn_start type.
    assert "turn_started" not in types
    assert "turn_start" not in types

    # Every server frame type is a pre-project type -- nothing outside today's
    # set, and every one is a known enum member.
    known = {m.value for m in MessageType}
    for t in types:
        assert t in _LEGACY_MESSAGE_TYPES, f"legacy session saw non-legacy frame type {t!r}"
        assert t in known

    # Every transcription frame carries the additive fields at their
    # behaviour-preserving defaults (a connection that never opened a turn stays at 0).
    for f in _text_frames(trace):
        if f["type"] == "transcription":
            assert f["turn_id"] == 0, f
            assert f["endpoint"] is False, f

    # Every audio_end carries reason='complete' -- a normal, uncancelled close.
    for f in _text_frames(trace):
        if f["type"] == "audio_end":
            assert f["reason"] == "complete", f

    # The legacy client never set flush, so its tts_stream frame defaults false.
    assert client_frame.flush is False
    assert json.loads(client_frame.model_dump_json())["flush"] is False


# --- structural proofs backing the trace ------------------------------------

def test_project_added_only_the_two_turn_message_types():
    # The enum grew by EXACTLY the two turn control frames and nothing else,
    # so a future non-additive frame type (a retraction, a mode negotiation) trips here.
    current = {m.value for m in MessageType}
    assert current - _LEGACY_MESSAGE_TYPES == set(_TURN_MESSAGE_TYPES)
    # ...and nothing pre-existing was removed or renamed either.
    assert _LEGACY_MESSAGE_TYPES <= current


def test_additive_fields_are_optional_with_behaviour_preserving_defaults():
    # No field became required, and each new field's default reproduces the
    # pre-feature wire. Construct each model with ONLY its pre-feature required args.
    t = TranscriptionMessage(text="x")
    assert t.turn_id == 0
    assert t.endpoint is False
    assert AudioEndMessage().reason == "complete"
    assert TTSStreamMessage(request_id="r", text="t").flush is False

    for model, field in [
        (TranscriptionMessage, "turn_id"),
        (TranscriptionMessage, "endpoint"),
        (AudioEndMessage, "reason"),
        (TTSStreamMessage, "flush"),
    ]:
        assert model.model_fields[field].is_required() is False, (
            f"{model.__name__}.{field} became required -- the additive rule forbids it")


def test_turn_started_is_constructed_only_in_the_turn_start_branch():
    # turn_started is the sole new server-emitted type; it must be impossible
    # to emit unsolicited. Structurally: TurnStartedMessage is constructed in exactly
    # one place in the transport, and that line sits INSIDE the `turn_start` branch --
    # so no code path emits it without a client turn_start frame.
    lines = (_SRC / "transport" / "websocket.py").read_text().splitlines()
    ctor_lines = [i for i, l in enumerate(lines) if "TurnStartedMessage(" in l]
    assert len(ctor_lines) == 1, (
        f"TurnStartedMessage constructed in {len(ctor_lines)} places; expected 1")
    ctor = ctor_lines[0]

    branch_line = next(i for i, l in enumerate(lines) if 'msg_type == "turn_start"' in l)
    branch_indent = len(lines[branch_line]) - len(lines[branch_line].lstrip())
    # The branch body runs until a line dedents back to (or past) the elif's own indent.
    branch_end = len(lines)
    for i in range(branch_line + 1, len(lines)):
        line = lines[i]
        if line.strip() and (len(line) - len(line.lstrip())) <= branch_indent:
            branch_end = i
            break
    assert branch_line < ctor < branch_end, (
        "TurnStartedMessage is constructed outside the turn_start branch")
