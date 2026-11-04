"""The playback frame of an opted-in session, model-free.

An opted-in client reports when a reply's playback started and stopped at its
speaker, on its own mic clock. The server accepts a well-formed start and stop for
a reply it sent. A malformed frame, a stop without a start, or a request_id the
server sent no reply for gets one invalid_request error frame; the frame is
dropped and the session goes on. A connection that did not opt in gets nothing
for a playback frame, as for any frame type it does not handle.
"""
from concurrent.futures import ThreadPoolExecutor

import pytest

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)
from fastapi.testclient import TestClient

from voicemanager.pipeline import audiocpp_stt
from voicemanager.protocol import INVALID_REQUEST, PlaybackMessage

from tyger_fake_sidecar import FakeTygerSidecar
from tyger_ws_kit import (
    ScriptedAudioCppSTT, StandInSidecar, barrier, errors, make_app, opt_in, speak,
)

_PCM = b"\x11\x22" * 512


@pytest.fixture
def tyger():
    fake = FakeTygerSidecar().start()
    try:
        yield fake
    finally:
        fake.stop()


@pytest.fixture
def client(tyger, monkeypatch):
    monkeypatch.setattr(audiocpp_stt, "AudioCppSTTPipeline", ScriptedAudioCppSTT)
    with ThreadPoolExecutor(max_workers=1) as executor:
        yield TestClient(make_app(sidecar=StandInSidecar(tyger.url), executor=executor))


def _start(request_id="r1", mic_sample=100):
    return {"type": "playback", "request_id": request_id, "event": "start",
            "mic_sample": mic_sample}


def _stop(request_id="r1", mic_sample=900, reply_sample=1200):
    return {"type": "playback", "request_id": request_id, "event": "stop",
            "mic_sample": mic_sample, "reply_sample": reply_sample}


def _assert_one_invalid(frames) -> None:
    errs = errors(frames)
    assert len(errs) == 1, frames
    assert errs[0]["error_type"] == INVALID_REQUEST


def _assert_session_goes_on(ws) -> None:
    """Audio is still transcribed, and a fresh reply's start and stop are accepted."""
    ws.send_bytes(_PCM)
    frames = barrier(ws)
    assert errors(frames) == [] and any(
        f.get("type") == "transcription" for f in frames), frames
    speak(ws, "again")
    ws.send_json(_start("again", 2000))
    ws.send_json(_stop("again", 2500, 300))
    assert errors(barrier(ws)) == []


def test_the_frame_model_carries_no_audio():
    assert set(PlaybackMessage.model_fields) == {
        "type", "request_id", "event", "mic_sample", "reply_sample"}


def test_start_and_stop_for_a_played_reply_are_accepted(client):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        speak(ws, "r1")
        ws.send_json(_start())
        ws.send_json(_stop())
        assert errors(barrier(ws)) == []
        _assert_session_goes_on(ws)


def test_a_start_alone_is_accepted(client):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        speak(ws, "r1")
        ws.send_json(_start())
        assert errors(barrier(ws)) == []


@pytest.mark.parametrize("bad", [
    {"type": "playback", "event": "start", "mic_sample": 1},               # no request_id
    {"type": "playback", "request_id": 7, "event": "start", "mic_sample": 1},
    {"type": "playback", "request_id": "r1", "event": "pause", "mic_sample": 1},
    {"type": "playback", "request_id": "r1", "event": "start"},            # no mic_sample
    {"type": "playback", "request_id": "r1", "event": "start", "mic_sample": -1},
    {"type": "playback", "request_id": "r1", "event": "start", "mic_sample": "5"},
    {"type": "playback", "request_id": "r1", "event": "start", "mic_sample": 1.5},
    {"type": "playback", "request_id": "r1", "event": "start", "mic_sample": True},
    {"type": "playback", "request_id": "r1", "event": "stop", "mic_sample": 1},
    {"type": "playback", "request_id": "r1", "event": "start", "mic_sample": 1,
     "reply_sample": 3},
    {"type": "playback", "request_id": "r1", "event": "stop", "mic_sample": 1,
     "reply_sample": -2},
])
def test_a_malformed_frame_is_invalid_and_dropped(client, bad):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        speak(ws, "r1")
        ws.send_json(bad)
        _assert_one_invalid(barrier(ws))
        # Dropped: a start for r1 is still the first one, so it is accepted.
        ws.send_json(_start())
        assert errors(barrier(ws)) == []
        _assert_session_goes_on(ws)


def test_a_stop_without_a_start_is_invalid(client):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        speak(ws, "r1")
        ws.send_json(_stop())
        _assert_one_invalid(barrier(ws))
        # The stop was dropped, so a start and a stop now are a normal pair.
        ws.send_json(_start())
        ws.send_json(_stop())
        assert errors(barrier(ws)) == []
        _assert_session_goes_on(ws)


@pytest.mark.parametrize("event", ["start", "stop"])
def test_an_unknown_request_id_is_invalid(client, event):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        speak(ws, "r1")
        ws.send_json(_start("nope") if event == "start" else _stop("nope"))
        _assert_one_invalid(barrier(ws))
        _assert_session_goes_on(ws)


def test_a_second_start_or_stop_is_invalid(client):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        speak(ws, "r1")
        ws.send_json(_start())
        ws.send_json(_start())
        _assert_one_invalid(barrier(ws))
        ws.send_json(_stop())
        ws.send_json(_stop())
        _assert_one_invalid(barrier(ws))
        _assert_session_goes_on(ws)


def test_a_stop_before_its_start_on_the_mic_clock_is_invalid(client):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        speak(ws, "r1")
        ws.send_json(_start(mic_sample=5000))
        ws.send_json(_stop(mic_sample=4000))
        _assert_one_invalid(barrier(ws))
        _assert_session_goes_on(ws)


def test_a_connection_that_did_not_opt_in_gets_nothing_for_playback(client, tyger):
    """Today a frame type the handler does not know is dropped without an answer;
    a playback frame on a connection that did not opt in stays that way."""
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        speak(ws, "r1")
        for frame in (_start(), _stop(), _stop("nope"), {"type": "playback"}):
            ws.send_json(frame)
        assert barrier(ws) == []
        ws.send_bytes(_PCM)
        assert errors(barrier(ws)) == []
    assert tyger.sessions == []
