"""Schema tests for the server-side wake-word frames in protocol.py.

A client with no keyword spotter of its own sends wake_word, with an optional list of
phrases; the server answers wake_word_ready with the list it listens for. When the
server hears one of them on a dormant connection it sends wake, with the phrase and
the number of mic samples it accepted on the connection at that point. These tests
pin each frame's fields, how each one parses and serializes, and that mic_sample is a
strict int >= 0.

Model-free and server-free: they build Pydantic models and nothing else.
"""
import json

import pytest
from pydantic import ValidationError

from voicemanager.protocol import (
    MessageType,
    WakeMessage,
    WakeWordMessage,
    WakeWordReadyMessage,
)


def _wire(model) -> dict:
    """The JSON object the model puts on the wire."""
    return json.loads(model.model_dump_json())


# --- message types -----------------------------------------------------------

def test_the_three_frame_types_are_message_types():
    for name in ("wake_word", "wake_word_ready", "wake"):
        assert MessageType(name).value == name
        assert getattr(MessageType, name).value == name


def test_each_model_carries_its_own_type():
    assert WakeWordMessage().type == MessageType.wake_word.value
    assert WakeWordReadyMessage(phrases=["hey computer"]).type == MessageType.wake_word_ready.value
    assert WakeMessage(phrase="hey computer", mic_sample=0).type == MessageType.wake.value


def test_the_models_have_exactly_their_documented_fields():
    assert list(WakeWordMessage.model_fields) == ["type", "phrases"]
    assert list(WakeWordReadyMessage.model_fields) == ["type", "phrases"]
    assert list(WakeMessage.model_fields) == ["type", "phrase", "mic_sample"]


# --- wake_word -----------------------------------------------------------------

def test_wake_word_parses_the_bare_frame_as_the_server_default():
    msg = WakeWordMessage.model_validate_json('{"type": "wake_word"}')
    assert msg.phrases is None


def test_wake_word_parses_a_null_phrase_list_as_the_server_default():
    msg = WakeWordMessage.model_validate_json('{"type": "wake_word", "phrases": null}')
    assert msg.phrases is None


def test_wake_word_parses_a_phrase_list():
    msg = WakeWordMessage.model_validate_json(
        '{"type": "wake_word", "phrases": ["hey hermes", "hey computer"]}')
    assert msg.phrases == ["hey hermes", "hey computer"]


def test_wake_word_keeps_an_empty_list_for_the_server_to_refuse():
    # The model does not judge the list: the server answers a list it cannot use
    # with an error frame, so an empty one must reach it as sent.
    assert WakeWordMessage.model_validate({"type": "wake_word", "phrases": []}).phrases == []


def test_wake_word_round_trips():
    msg = WakeWordMessage(phrases=["hey hermes"])
    assert _wire(msg) == {"type": "wake_word", "phrases": ["hey hermes"]}
    assert WakeWordMessage.model_validate_json(msg.model_dump_json()) == msg


@pytest.mark.parametrize("phrases", ["hey hermes", [1, 2], [None], {"a": "b"}, 7])
def test_wake_word_rejects_a_phrase_list_that_is_not_a_list_of_strings(phrases):
    with pytest.raises(ValidationError):
        WakeWordMessage.model_validate({"type": "wake_word", "phrases": phrases})


def test_wake_word_rejects_another_type():
    with pytest.raises(ValidationError):
        WakeWordMessage.model_validate({"type": "wake"})


# --- wake_word_ready -----------------------------------------------------------

def test_wake_word_ready_round_trips():
    msg = WakeWordReadyMessage(phrases=["hey computer"])
    assert _wire(msg) == {"type": "wake_word_ready", "phrases": ["hey computer"]}
    assert WakeWordReadyMessage.model_validate_json(msg.model_dump_json()) == msg


def test_wake_word_ready_requires_the_phrase_list():
    with pytest.raises(ValidationError):
        WakeWordReadyMessage.model_validate({"type": "wake_word_ready"})


# --- wake ------------------------------------------------------------------------

def test_wake_round_trips():
    msg = WakeMessage(phrase="hey hermes", mic_sample=48000)
    assert _wire(msg) == {"type": "wake", "phrase": "hey hermes", "mic_sample": 48000}
    assert WakeMessage.model_validate_json(msg.model_dump_json()) == msg


@pytest.mark.parametrize("missing", ["phrase", "mic_sample"])
def test_wake_requires_each_field(missing):
    frame = {"type": "wake", "phrase": "hey hermes", "mic_sample": 0}
    del frame[missing]
    with pytest.raises(ValidationError):
        WakeMessage.model_validate(frame)


@pytest.mark.parametrize("mic_sample", [-1, "16000", 1.5, True, None])
def test_wake_mic_sample_is_a_strict_int_not_below_zero(mic_sample):
    with pytest.raises(ValidationError):
        WakeMessage.model_validate(
            {"type": "wake", "phrase": "hey hermes", "mic_sample": mic_sample})
