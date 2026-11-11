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
from voicemanager import protocol
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
# The turn-engine frames, added later and also opt-in: turn_engine and playback are
# client->server; turn_engine_ready answers a turn_engine opt-in, and barge and
# turn_end are sent only to a session that opted in.
_TURN_ENGINE_MESSAGE_TYPES = frozenset({
    "turn_engine", "turn_engine_ready", "playback", "barge", "turn_end",
})
# The listening-boundary frames, added later still: listen_stop and listen_start are
# client->server; listen_stopped answers a listen_stop. A client that never sends
# listen_stop sees none of them.
_LISTEN_MESSAGE_TYPES = frozenset({"listen_stop", "listen_start", "listen_stopped"})
# The wake-word frames, added after those: wake_word is client->server;
# wake_word_ready answers it and wake reports a heard phrase. A client that never
# sends wake_word sees none of them.
_WAKE_MESSAGE_TYPES = frozenset({"wake_word", "wake_word_ready", "wake"})

# Every error_type the server sent before the turn engine. The turn engine reuses
# invalid_request for bad frames and adds only its three own types.
_LEGACY_ERROR_TYPES = frozenset({
    "invalid_audio_format", "invalid_request", "server_error", "stt_backend_fault",
    "stt_backend_unavailable", "stt_backend_unready", "timeout", "timeout_warning",
    "tts_backend_fault", "unknown_voice",
})
_TURN_ENGINE_ERROR_TYPES = (
    "turn_engine_unavailable", "turn_engine_unsupported", "turn_engine_lost",
)

# Marks a required field in the snapshot below.
_REQUIRED = object()
# The frame models as they stood before the turn engine: every field, in order,
# with its default or _REQUIRED. The turn engine must leave each one untouched.
_LEGACY_MODEL_FIELDS = {
    "HandshakeMessage": {
        "type": "ready", "session_id": _REQUIRED, "audio_format": _REQUIRED,
        "stt_engine": "whisper", "tts_engine": "openvoice", "available_voices": [],
        "default_voice": None},
    "ErrorMessage": {"type": "error", "error_type": _REQUIRED, "message": _REQUIRED},
    "TranscriptionMessage": {
        "type": "transcription", "text": _REQUIRED, "is_final": True, "endpoint": False,
        "turn_id": 0, "segment_index": 0, "language": "", "inference_duration": 0.0,
        "confidence": 0.0, "start": 0.0, "end": 0.0},
    "AudioEndMessage": {"type": "audio_end", "request_id": None, "reason": "complete"},
    "TTSRequestMessage": {
        "type": "tts", "text": _REQUIRED, "request_id": _REQUIRED, "voice": None},
    "AudioStartMessage": {
        "type": "audio_start", "request_id": _REQUIRED, "sample_rate": _REQUIRED,
        "channels": 1, "bit_depth": 16, "encoding": "pcm"},
    "TTSCancelMessage": {"type": "tts_cancel", "request_id": _REQUIRED},
    "TTSStreamMessage": {
        "type": "tts_stream", "request_id": _REQUIRED, "text": _REQUIRED, "done": False,
        "flush": False, "voice": None, "chunking": "sentence"},
    "TurnStartedMessage": {"type": "turn_started", "turn_id": _REQUIRED},
}

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
    # Nor does any listening-boundary frame: listen_stopped answers only a
    # listen_stop, which a legacy client never sends.
    assert not _LISTEN_MESSAGE_TYPES & set(types)
    # Nor any wake-word frame: a legacy client never opts in.
    assert not _WAKE_MESSAGE_TYPES & set(types)

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
    # The enum grew by EXACTLY the two turn control frames, the five opt-in
    # turn-engine frames and the three listening-boundary frames and nothing else, so
    # a future non-additive frame type (a retraction, a mode negotiation) trips here.
    current = {m.value for m in MessageType}
    assert current - _LEGACY_MESSAGE_TYPES == set(
        _TURN_MESSAGE_TYPES | _TURN_ENGINE_MESSAGE_TYPES | _LISTEN_MESSAGE_TYPES
        | _WAKE_MESSAGE_TYPES)
    # The additions are separate sets: no type is counted twice.
    assert not _TURN_MESSAGE_TYPES & _TURN_ENGINE_MESSAGE_TYPES
    assert not _LISTEN_MESSAGE_TYPES & (_TURN_MESSAGE_TYPES | _TURN_ENGINE_MESSAGE_TYPES)
    assert not _WAKE_MESSAGE_TYPES & (
        _TURN_MESSAGE_TYPES | _TURN_ENGINE_MESSAGE_TYPES | _LISTEN_MESSAGE_TYPES)
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


# --- the turn-engine additions leave the existing contract alone ------------

def _field_snapshot(model) -> dict:
    return {name: (_REQUIRED if field.is_required() else field.default)
            for name, field in model.model_fields.items()}


def test_existing_frame_models_are_unchanged():
    # Same fields, same order, same defaults, same required set for every frame
    # model that existed before the turn engine.
    for name, expected in _LEGACY_MODEL_FIELDS.items():
        actual = _field_snapshot(getattr(protocol, name))
        assert list(actual) == list(expected), f"{name} fields changed"
        for field, default in expected.items():
            if default is _REQUIRED:
                assert actual[field] is _REQUIRED, f"{name}.{field} is no longer required"
            else:
                assert actual[field] == default, f"{name}.{field} default changed"


def test_new_error_types_are_distinct_from_the_existing_ones():
    assert protocol.TURN_ENGINE_ERROR_TYPES == _TURN_ENGINE_ERROR_TYPES
    assert len(set(_TURN_ENGINE_ERROR_TYPES)) == 3
    assert not set(_TURN_ENGINE_ERROR_TYPES) & _LEGACY_ERROR_TYPES
    # Bad turn-engine frames reuse the existing invalid_request type.
    assert protocol.INVALID_REQUEST == "invalid_request"
    assert protocol.INVALID_REQUEST in _LEGACY_ERROR_TYPES


def test_existing_error_types_are_still_sent_by_name():
    # The turn engine renames none of today's error types: each is still in the
    # source the server sends it from.
    source = "\n".join(p.read_text() for p in _SRC.rglob("*.py"))
    for error_type in _LEGACY_ERROR_TYPES:
        assert f'"{error_type}"' in source, error_type


def test_turn_engine_fields_are_opt_in_only():
    # The new frames are new models: no existing model gained a turn-engine field,
    # and none of the five frame types is in the set a legacy session may see.
    for fields in _LEGACY_MODEL_FIELDS.values():
        assert not {"engine", "onset_mic_sample", "mic_sample", "by", "reply_sample"} & set(
            fields)
    assert not _TURN_ENGINE_MESSAGE_TYPES & _LEGACY_MESSAGE_TYPES
    for model in (protocol.TurnEngineMessage, protocol.TurnEngineReadyMessage,
                  protocol.PlaybackMessage, protocol.BargeMessage,
                  protocol.TurnEndMessage):
        assert model.model_fields["type"].default in _TURN_ENGINE_MESSAGE_TYPES


def test_listen_frames_are_opt_in_only():
    # The listening-boundary frames are new models: no existing model gained one of
    # their types, and none is in the set a legacy session may see.
    assert not _LISTEN_MESSAGE_TYPES & _LEGACY_MESSAGE_TYPES
    for model in (protocol.ListenStopMessage, protocol.ListenStartMessage,
                  protocol.ListenStoppedMessage):
        assert model.model_fields["type"].default in _LISTEN_MESSAGE_TYPES
    # The two client frames carry nothing but their type, so no field is required.
    for model in (protocol.ListenStopMessage, protocol.ListenStartMessage):
        assert list(model.model_fields) == ["type"]
