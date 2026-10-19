"""WebSocket protocol message types and Pydantic models.

All WebSocket text frames are JSON with a 'type' field for routing.
This module defines the canonical message types and their schemas.
"""
import enum
from typing import Literal
from pydantic import BaseModel


class MessageType(str, enum.Enum):
    """All valid WebSocket text message types."""
    ready = "ready"
    transcription = "transcription"
    audio_chunk = "audio_chunk"
    audio_end = "audio_end"
    error = "error"
    tts = "tts"
    audio_start = "audio_start"
    tts_cancel = "tts_cancel"
    tts_stream = "tts_stream"
    # Turn-boundary control frames: a client asks to open a turn with
    # turn_start; the server acks with turn_started carrying the new id.
    turn_start = "turn_start"
    turn_started = "turn_started"


class HandshakeMessage(BaseModel):
    """Sent on WebSocket connect after accept.

    Minimal: session_id + audio_format only, no capabilities field.
    """
    type: Literal["ready"] = "ready"
    session_id: str
    audio_format: dict
    stt_engine: str = "whisper"  # "whisper" or "qwen" — informational only
    tts_engine: str = "openvoice"         # "openvoice" or "vibevoice"
    available_voices: list[str] = []      # populated for vibevoice, empty for openvoice
    # Additive: the voice a TTS request without 'voice' gets on this connection --
    # the token's voice when the engine has it, else the engine's default. None
    # for an engine without presets (OpenVoice).
    default_voice: str | None = None


class ErrorMessage(BaseModel):
    """Sent before disconnect on protocol violations.

    Fields: error_type (machine-readable category) and message (human-readable detail).
    """
    type: Literal["error"] = "error"
    error_type: str
    message: str


class TranscriptionMessage(BaseModel):
    """STT result sent back to client."""
    type: Literal["transcription"] = "transcription"
    text: str
    is_final: bool = True
    # Additive speculative-endpoint flag: true only on the
    # nemotron path's provisional endpoint frame; whisper/qwen carry the default.
    endpoint: bool = False
    # Additive turn field: the turn this utterance BELONGS TO -- the
    # turn that was open when its audio was received, not the turn current at send
    # time. A connection that never opens a turn stays on turn_id=0 forever.
    turn_id: int = 0
    segment_index: int = 0
    language: str = ""
    inference_duration: float = 0.0
    confidence: float = 0.0
    start: float = 0.0
    end: float = 0.0


class AudioEndMessage(BaseModel):
    """Signals end of TTS audio stream."""
    type: Literal["audio_end"] = "audio_end"
    request_id: str | None = None
    # Additive: why the audio stream ended. Defaults to "complete",
    # which reproduces today's behaviour for every existing close path -- including
    # the VM_TTS_STREAM_FLUSH_TIMEOUT close, which is a normal completion, not an
    # interruption. "cancelled" is set when a tts_cancel or a supersede-by-new-
    # request_id stops a reply. "interrupted" is reserved for the future
    # server-side barge-in work and is never emitted here.
    reason: str = "complete"


class TTSRequestMessage(BaseModel):
    """TTS synthesis request from client. Client sends text + request_id."""
    type: Literal["tts"] = "tts"
    text: str
    request_id: str
    voice: str | None = None


class AudioStartMessage(BaseModel):
    """Sent before first TTS audio chunk. Contains format info for client."""
    type: Literal["audio_start"] = "audio_start"
    request_id: str
    sample_rate: int
    channels: int = 1
    bit_depth: int = 16
    encoding: str = "pcm"


class TTSCancelMessage(BaseModel):
    """Cancel in-progress TTS synthesis. Server stops and sends audio_end."""
    type: Literal["tts_cancel"] = "tts_cancel"
    request_id: str


class TTSStreamMessage(BaseModel):
    """Streaming TTS text chunk. Server accumulates by request_id."""
    type: Literal["tts_stream"] = "tts_stream"
    request_id: str
    text: str
    done: bool = False
    # Additive: dispatch the whole accumulated buffer for synthesis
    # as-is — complete sentence or not — and leave the request open. Distinct from
    # `done`, which closes the request; flush never queues the end-of-sentence
    # sentinel and never yields an audio_end.
    flush: bool = False
    voice: str | None = None


class TurnStartedMessage(BaseModel):
    """Ack for a client turn_start control frame.

    The server OWNS the turn counter; turn_start increments it and this frame reports
    the newly opened turn's id so the client learns which turn_id to filter on rather
    than inferring it from the next stamped transcription frame. The incoming request
    is the bare {"type": "turn_start"} frame (no payload), so it needs no model here.
    """
    type: Literal["turn_started"] = "turn_started"
    turn_id: int


# Audio format specification for handshake
AUDIO_FORMAT_SPEC = {
    "sample_rate": 16000,
    "channels": 1,
    "bit_depth": 16,
    "encoding": "pcm",
}
