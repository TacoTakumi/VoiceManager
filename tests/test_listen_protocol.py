"""Schema tests for the listening-boundary frames in protocol.py.

A client that stops sending mic audio (it stops listening, or ends a conversation)
sends listen_stop; the server answers listen_stopped with the number of mic samples
it accepted on the connection at that point. When the client resumes on the same
connection it sends listen_start, which gets no answer. These tests pin each frame's
fields, how each one parses and serializes, and that mic_sample is a strict int >= 0.

Model-free and server-free: they build Pydantic models and nothing else.
"""
import json

import pytest
from pydantic import ValidationError

from voicemanager.protocol import (
    ListenStartMessage,
    ListenStopMessage,
    ListenStoppedMessage,
    MessageType,
)


def _wire(model) -> dict:
    """The JSON object the model puts on the wire."""
    return json.loads(model.model_dump_json())


# --- message types -----------------------------------------------------------

def test_the_three_frame_types_are_message_types():
    for name in ("listen_stop", "listen_start", "listen_stopped"):
        assert MessageType(name).value == name
        assert getattr(MessageType, name).value == name


def test_each_model_carries_its_own_type():
    assert ListenStopMessage().type == MessageType.listen_stop.value
    assert ListenStartMessage().type == MessageType.listen_start.value
    assert ListenStoppedMessage(mic_sample=0).type == MessageType.listen_stopped.value


# --- listen_stop ---------------------------------------------------------------

def test_listen_stop_parses_the_bare_frame():
    msg = ListenStopMessage.model_validate_json('{"type": "listen_stop"}')
    assert msg.type == "listen_stop"


def test_listen_stop_serializes_to_the_bare_frame():
    assert _wire(ListenStopMessage()) == {"type": "listen_stop"}


def test_listen_stop_rejects_another_type():
    with pytest.raises(ValidationError):
        ListenStopMessage.model_validate({"type": "listen_start"})


# --- listen_start --------------------------------------------------------------

def test_listen_start_parses_the_bare_frame():
    msg = ListenStartMessage.model_validate_json('{"type": "listen_start"}')
    assert msg.type == "listen_start"


def test_listen_start_serializes_to_the_bare_frame():
    assert _wire(ListenStartMessage()) == {"type": "listen_start"}


def test_listen_start_rejects_another_type():
    with pytest.raises(ValidationError):
        ListenStartMessage.model_validate({"type": "listen_stop"})


# --- listen_stopped ------------------------------------------------------------

@pytest.mark.parametrize("mic_sample", [0, 1, 5120, 16000 * 3600])
def test_listen_stopped_round_trips(mic_sample):
    text = json.dumps({"type": "listen_stopped", "mic_sample": mic_sample})
    msg = ListenStoppedMessage.model_validate_json(text)
    assert msg.mic_sample == mic_sample
    assert _wire(msg) == {"type": "listen_stopped", "mic_sample": mic_sample}


def test_listen_stopped_needs_mic_sample():
    with pytest.raises(ValidationError):
        ListenStoppedMessage.model_validate({"type": "listen_stopped"})


@pytest.mark.parametrize("mic_sample", [-1, "5120", 5120.0, 1.5, True, False, None, [1]])
def test_listen_stopped_rejects_a_bad_mic_sample(mic_sample):
    with pytest.raises(ValidationError):
        ListenStoppedMessage.model_validate(
            {"type": "listen_stopped", "mic_sample": mic_sample})
    with pytest.raises(ValidationError):
        ListenStoppedMessage.model_validate_json(
            json.dumps({"type": "listen_stopped", "mic_sample": mic_sample}))


def test_listen_stopped_rejects_another_type():
    with pytest.raises(ValidationError):
        ListenStoppedMessage.model_validate({"type": "listen_stop", "mic_sample": 0})
