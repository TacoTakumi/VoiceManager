"""WebSocket protocol message types and Pydantic models.

All WebSocket text frames are JSON with a 'type' field for routing.
This module defines the canonical message types and their schemas.
"""
import enum
from typing import Literal
from pydantic import BaseModel, Field, model_serializer, model_validator


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
    # Turn-engine frames, all opt-in: a client opts in with turn_engine and the
    # server answers turn_engine_ready; the client reports playback edges with
    # playback; the server reports barge and turn_end only to an opted-in session.
    turn_engine = "turn_engine"
    turn_engine_ready = "turn_engine_ready"
    playback = "playback"
    barge = "barge"
    turn_end = "turn_end"
    # Listening-boundary frames: a client that stops sending mic audio sends
    # listen_stop and the server acks with listen_stopped; listen_start resumes
    # listening on the same connection and gets no ack. A client that never sends
    # listen_stop sees none of them.
    listen_stop = "listen_stop"
    listen_start = "listen_start"
    listen_stopped = "listen_stopped"
    # Wake-word frames, all opt-in: a client asks for a server-side wake word with
    # wake_word and the server answers wake_word_ready; the server reports wake when
    # it hears one of the connection's phrases. A client that never sends wake_word
    # sees none of them.
    wake_word = "wake_word"
    wake_word_ready = "wake_word_ready"
    wake = "wake"


class HandshakeMessage(BaseModel):
    """Sent on WebSocket connect after accept.

    Minimal: session_id + audio_format only, no capabilities field.
    """
    type: Literal["ready"] = "ready"
    session_id: str
    audio_format: dict
    stt_engine: str = "whisper"  # "whisper" or "qwen" - informational only
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
    # request_id stops a reply. "interrupted" is set when the turn engine reports
    # that the user cut in over a played reply; the barge frame comes before it.
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
    # as-is - complete sentence or not - and leave the request open. Distinct from
    # `done`, which closes the request; flush never queues the end-of-sentence
    # sentinel and never yields an audio_end.
    flush: bool = False
    voice: str | None = None
    # Additive: where the server may cut the accumulated text into synthesis units.
    # "sentence" (the default) is today's behaviour; "clause" also allows cuts at
    # clause boundaries. Fixed by the request's first frame, like voice: a later
    # frame's value is ignored. Any other value is rejected with an error frame of
    # error_type "invalid_request" and the frame is dropped.
    chunking: Literal["sentence", "clause"] = "sentence"


#: The tts_stream chunking modes a client may ask for.
CHUNKING_MODES: tuple[str, ...] = ("sentence", "clause")


class TurnStartedMessage(BaseModel):
    """Ack for a client turn_start control frame.

    The server OWNS the turn counter; turn_start increments it and this frame reports
    the newly opened turn's id so the client learns which turn_id to filter on rather
    than inferring it from the next stamped transcription frame. The incoming request
    is the bare {"type": "turn_start"} frame (no payload), so it needs no model here.
    """
    type: Literal["turn_started"] = "turn_started"
    turn_id: int


# --- turn engine ---------------------------------------------------------------
#
# A session opts in to a turn engine, which decides from the mic audio and the
# agent's own playback when the user's turn ended and when the user cut in over a
# reply. A session that never sends turn_engine sees none of these frames.

#: The turn engines a client may name in a turn_engine opt-in.
TURN_ENGINES: tuple[str, ...] = ("tyger",)
#: The edges a playback frame reports.
PLAYBACK_EVENTS: tuple[str, ...] = ("start", "stop")
#: What ended a turn: the turn model's endpoint head, or its silence fallback.
TURN_END_BY: tuple[str, ...] = ("model", "silence")

#: error_type for a bad frame. Turn-engine frames reuse it: a malformed frame, an
#: opt-in after the first audio frame, a second opt-in, an unknown engine name, a
#: playback stop without a start, or an unknown request_id.
INVALID_REQUEST = "invalid_request"
#: error_type for an opt-in while the server runs with the turn engine off.
TURN_ENGINE_UNAVAILABLE = "turn_engine_unavailable"
#: error_type for an opt-in on an STT engine the turn engine does not support.
TURN_ENGINE_UNSUPPORTED = "turn_engine_unsupported"
#: error_type sent once to each opted-in session when the turn engine's process
#: exits; the session goes on as one that never opted in.
TURN_ENGINE_LOST = "turn_engine_lost"
#: The error types the turn engine adds, in the order above.
TURN_ENGINE_ERROR_TYPES: tuple[str, ...] = (
    TURN_ENGINE_UNAVAILABLE, TURN_ENGINE_UNSUPPORTED, TURN_ENGINE_LOST,
)


class TurnEngineMessage(BaseModel):
    """Client opt-in to a turn engine, sent before the connection's first audio frame.

    ``engine`` is kept as any string so the server can answer an unknown name with
    an invalid_request error frame; the known names are ``TURN_ENGINES``.
    """
    type: Literal["turn_engine"] = "turn_engine"
    engine: str


class TurnEngineReadyMessage(BaseModel):
    """Server answer to a successful turn_engine opt-in."""
    type: Literal["turn_engine_ready"] = "turn_engine_ready"
    engine: str


class PlaybackMessage(BaseModel):
    """Client report that a reply's playback started or stopped at the speaker.

    ``mic_sample`` is the index on the client's own outgoing 16 kHz mic stream
    (samples sent since its first audio frame) at which the edge reached the
    speaker. ``reply_sample`` is sent only on stop: how many of the reply's samples
    played. The frame carries no audio; the server builds the agent channel from the
    PCM it sent for ``request_id``. Integers are strict: a string, float or bool is a
    malformed frame.
    """
    type: Literal["playback"] = "playback"
    request_id: str
    event: Literal["start", "stop"]
    mic_sample: int = Field(ge=0, strict=True)
    reply_sample: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def _reply_sample_only_on_stop(self) -> "PlaybackMessage":
        if self.event == "stop" and self.reply_sample is None:
            raise ValueError("a playback stop needs reply_sample")
        if self.event == "start" and self.reply_sample is not None:
            raise ValueError("reply_sample is sent only on a playback stop")
        return self

    @model_serializer(mode="wrap")
    def _omit_reply_sample_on_start(self, handler):
        data = handler(self)
        if self.reply_sample is None:
            data.pop("reply_sample", None)
        return data


class BargeMessage(BaseModel):
    """Server report that the user cut in over a played reply.

    ``onset_mic_sample`` is where on the client's mic stream the user's speech began,
    and ``mic_sample`` is where the turn engine decided. The server cancels the
    request, sends no more PCM for it, and ends it with audio_end reason
    'interrupted' after this frame.
    """
    type: Literal["barge"] = "barge"
    request_id: str
    onset_mic_sample: int
    mic_sample: int


class TurnEndMessage(BaseModel):
    """Server report that the user's turn ended, for an opted-in session.

    ``text`` joins, in order, every final sent since the previous turn_end.
    ``mic_sample`` is where on the client's mic stream the turn engine decided, and
    ``by`` says whether the model's endpoint head or its silence fallback ended it.
    """
    type: Literal["turn_end"] = "turn_end"
    turn_id: int
    text: str
    mic_sample: int
    by: Literal["model", "silence"]


# --- listening boundary ----------------------------------------------------------
#
# A client may stop listening (it stops sending mic audio, for example when the user
# ends a conversation) and later start again on the same connection. At the boundary
# the server drops the transcription and turn state built from the audio before it,
# so nothing from before the stop reaches the client after it. The frames add no
# error types.


class ListenStopMessage(BaseModel):
    """Client report that it stopped sending mic audio on this connection.

    The bare {"type": "listen_stop"} frame; the server answers listen_stopped.
    """
    type: Literal["listen_stop"] = "listen_stop"


class ListenStartMessage(BaseModel):
    """Client report that it starts sending mic audio again on this connection.

    The bare {"type": "listen_start"} frame. The server sends no answer.
    """
    type: Literal["listen_start"] = "listen_start"


class ListenStoppedMessage(BaseModel):
    """Server answer to a listen_stop.

    ``mic_sample`` is how many mic samples the server accepted on the connection at
    the boundary. Integers are strict: a string, float or bool is a malformed frame.
    """
    type: Literal["listen_stopped"] = "listen_stopped"
    mic_sample: int = Field(ge=0, strict=True)


# --- server-side wake word -------------------------------------------------------
#
# A client with no keyword spotter of its own can ask the server to listen for a wake
# phrase. An opted-in connection is dormant while it is stopped: its mic audio goes to
# a keyword spotter and to nothing else. When the spotter hears a phrase the server
# sends wake and listens, as for a client listen_start. A listen_stop makes the
# connection dormant again. The frames add no error types: a refused wake_word gets
# an invalid_request error frame.


class WakeWordMessage(BaseModel):
    """Client opt-in to the server-side wake word.

    ``phrases`` is the list of wake phrases for this connection, as plain text. Left
    out (or null), the server's default list is used. The list is kept as sent so the
    server can answer a list it cannot use with an invalid_request error frame.
    """
    type: Literal["wake_word"] = "wake_word"
    phrases: list[str] | None = None


class WakeWordReadyMessage(BaseModel):
    """Server answer to a successful wake_word opt-in.

    ``phrases`` is the list the connection's spotter listens for. The connection is
    dormant from this frame on.
    """
    type: Literal["wake_word_ready"] = "wake_word_ready"
    phrases: list[str]


class WakeMessage(BaseModel):
    """Server report that it heard a wake phrase on a dormant connection.

    ``phrase`` is the phrase that matched, as the connection's list has it.
    ``mic_sample`` is how many mic samples the server accepted on the connection at
    the wake: the first mic frame after the dormant audio starts at that count. Audio
    sent while dormant is not counted in it.

    ``dormant_sample`` is how many samples of dormant audio the server gave the
    spotter on the connection since it last became dormant, up to and including the
    frame the phrase was heard in. A client keeps sending while this frame is on its
    way: the audio it sent after those ``dormant_sample`` samples was not dormant
    audio any more, and the server counted it as mic audio. A client that keeps its
    own mic count (for playback frames) adds that audio to it.

    Integers are strict, as in listen_stopped.
    """
    type: Literal["wake"] = "wake"
    phrase: str
    mic_sample: int = Field(ge=0, strict=True)
    dormant_sample: int = Field(ge=0, strict=True)


# Audio format specification for handshake
AUDIO_FORMAT_SPEC = {
    "sample_rate": 16000,
    "channels": 1,
    "bit_depth": 16,
    "encoding": "pcm",
}
