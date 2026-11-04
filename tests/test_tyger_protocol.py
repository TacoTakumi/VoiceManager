"""Schema tests for the turn-engine frames and error types in protocol.py.

The turn engine adds five public frames: the client's opt-in (turn_engine) and the
server's answer (turn_engine_ready), the client's playback report (playback), and the
server's barge and turn_end. It also adds three error types. These tests pin each
frame's fields, which fields are required, and how each frame parses and serializes.

Model-free and server-free: they build Pydantic models and nothing else.
"""
import json

import pytest
from pydantic import ValidationError

from voicemanager.protocol import (
    PLAYBACK_EVENTS,
    TURN_END_BY,
    TURN_ENGINE_ERROR_TYPES,
    TURN_ENGINE_LOST,
    TURN_ENGINE_UNAVAILABLE,
    TURN_ENGINE_UNSUPPORTED,
    TURN_ENGINES,
    BargeMessage,
    MessageType,
    PlaybackMessage,
    TurnEndMessage,
    TurnEngineMessage,
    TurnEngineReadyMessage,
)


def _wire(model) -> dict:
    """The JSON object the model puts on the wire."""
    return json.loads(model.model_dump_json())


# --- message types and constants --------------------------------------------

def test_the_five_frame_types_are_message_types():
    for name in ("turn_engine", "turn_engine_ready", "playback", "barge", "turn_end"):
        assert MessageType(name).value == name


def test_error_types_are_named_constants_with_their_wire_values():
    assert TURN_ENGINE_UNAVAILABLE == "turn_engine_unavailable"
    assert TURN_ENGINE_UNSUPPORTED == "turn_engine_unsupported"
    assert TURN_ENGINE_LOST == "turn_engine_lost"
    assert TURN_ENGINE_ERROR_TYPES == (
        "turn_engine_unavailable", "turn_engine_unsupported", "turn_engine_lost")


def test_engine_and_enum_constants():
    assert TURN_ENGINES == ("tyger",)
    assert PLAYBACK_EVENTS == ("start", "stop")
    assert TURN_END_BY == ("model", "silence")


# --- turn_engine / turn_engine_ready -----------------------------------------

def test_turn_engine_opt_in_parses_and_serializes():
    frame = TurnEngineMessage.model_validate({"type": "turn_engine", "engine": "tyger"})
    assert frame.engine == "tyger"
    assert _wire(frame) == {"type": "turn_engine", "engine": "tyger"}


def test_turn_engine_needs_an_engine():
    with pytest.raises(ValidationError):
        TurnEngineMessage.model_validate({"type": "turn_engine"})


def test_turn_engine_keeps_an_unknown_engine_name_for_the_server_to_answer():
    # The model takes any engine name; the server answers an unknown one with
    # invalid_request, so the name has to survive parsing.
    assert TurnEngineMessage(engine="other").engine == "other"


def test_turn_engine_ready_serializes():
    assert _wire(TurnEngineReadyMessage(engine="tyger")) == {
        "type": "turn_engine_ready", "engine": "tyger"}


# --- playback -----------------------------------------------------------------

def test_playback_start_without_reply_sample_is_valid():
    frame = PlaybackMessage.model_validate(
        {"type": "playback", "request_id": "r1", "event": "start", "mic_sample": 48000})
    assert frame.event == "start"
    assert frame.mic_sample == 48000
    assert frame.reply_sample is None


def test_playback_start_serializes_without_reply_sample():
    frame = PlaybackMessage(request_id="r1", event="start", mic_sample=48000)
    assert _wire(frame) == {
        "type": "playback", "request_id": "r1", "event": "start", "mic_sample": 48000}


def test_playback_stop_carries_reply_sample():
    frame = PlaybackMessage.model_validate({
        "type": "playback", "request_id": "r1", "event": "stop",
        "mic_sample": 96000, "reply_sample": 72000})
    assert frame.reply_sample == 72000
    assert _wire(frame) == {
        "type": "playback", "request_id": "r1", "event": "stop",
        "mic_sample": 96000, "reply_sample": 72000}


def test_playback_stop_needs_reply_sample():
    with pytest.raises(ValidationError):
        PlaybackMessage.model_validate(
            {"type": "playback", "request_id": "r1", "event": "stop", "mic_sample": 1})


def test_playback_start_rejects_reply_sample():
    with pytest.raises(ValidationError):
        PlaybackMessage.model_validate({
            "type": "playback", "request_id": "r1", "event": "start",
            "mic_sample": 1, "reply_sample": 0})


def test_playback_has_no_audio_field():
    assert "audio" not in PlaybackMessage.model_fields
    schema = PlaybackMessage.model_json_schema()
    assert "audio" not in schema["properties"]
    assert set(schema["properties"]) == {
        "type", "request_id", "event", "mic_sample", "reply_sample"}


@pytest.mark.parametrize("bad", [
    {"request_id": "r1", "mic_sample": 1},                                # no event
    {"request_id": "r1", "event": "pause", "mic_sample": 1},              # unknown event
    {"event": "start", "mic_sample": 1},                                  # no request_id
    {"request_id": "r1", "event": "start"},                               # no mic_sample
    {"request_id": "r1", "event": "start", "mic_sample": -1},             # negative
    {"request_id": "r1", "event": "start", "mic_sample": "10"},           # string
    {"request_id": "r1", "event": "start", "mic_sample": 1.5},            # float
    {"request_id": "r1", "event": "start", "mic_sample": True},           # bool
    {"request_id": 7, "event": "start", "mic_sample": 1},                 # id not a str
    {"request_id": "r1", "event": "stop", "mic_sample": 1, "reply_sample": -1},
    {"request_id": "r1", "event": "stop", "mic_sample": 1, "reply_sample": "5"},
])
def test_malformed_playback_frames_are_rejected(bad):
    with pytest.raises(ValidationError):
        PlaybackMessage.model_validate({"type": "playback", **bad})


def test_playback_rejects_another_frame_type():
    with pytest.raises(ValidationError):
        PlaybackMessage.model_validate(
            {"type": "tts", "request_id": "r1", "event": "start", "mic_sample": 1})


# --- barge --------------------------------------------------------------------

def test_barge_serializes_with_all_fields():
    frame = BargeMessage(request_id="r1", onset_mic_sample=80000, mic_sample=84480)
    assert _wire(frame) == {
        "type": "barge", "request_id": "r1",
        "onset_mic_sample": 80000, "mic_sample": 84480}


def test_barge_fields_are_required():
    for field in ("request_id", "onset_mic_sample", "mic_sample"):
        assert BargeMessage.model_fields[field].is_required(), field


# --- turn_end -----------------------------------------------------------------

@pytest.mark.parametrize("by", ["model", "silence"])
def test_turn_end_serializes(by):
    frame = TurnEndMessage(turn_id=3, text="all stop. full impulse.", mic_sample=160000, by=by)
    assert _wire(frame) == {
        "type": "turn_end", "turn_id": 3, "text": "all stop. full impulse.",
        "mic_sample": 160000, "by": by}


def test_turn_end_rejects_an_unknown_reason():
    with pytest.raises(ValidationError):
        TurnEndMessage(turn_id=0, text="x", mic_sample=0, by="vad")


def test_turn_end_fields_are_required():
    for field in ("turn_id", "text", "mic_sample", "by"):
        assert TurnEndMessage.model_fields[field].is_required(), field


def test_turn_end_round_trips():
    frame = TurnEndMessage(turn_id=1, text="engage", mic_sample=5120, by="model")
    assert TurnEndMessage.model_validate_json(frame.model_dump_json()) == frame
