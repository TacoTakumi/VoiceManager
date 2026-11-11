"""The turn_empty frame: a turn-engine client learns that a turn ended with no words.

At a turn end with no final text since the previous one, the server sends turn_empty
in place of turn_end. Its ``speech`` is true when the STT's speech detector found a
speech start in that time (the user spoke and the decoder gave no words) and false
when the turn engine alone ended a turn. A turn end with text still gives turn_end
only. Nothing goes out once the sidecar is lost or for audio from before a
listening boundary.

Covers the frame model, the real WebSocket handler with the in-process fake sidecar
and a scripted STT, the turn engine session alone, the Python client, and the client
guide. Model-free and server-free.
"""
import asyncio
import json
import re
from pathlib import Path

import pytest
import torch

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)
from fastapi.testclient import TestClient

import voicemanager.config as cfg
from voicemanager.client import VoiceManagerClient
from voicemanager.pipeline import audiocpp_stt
from voicemanager.pipeline.tyger import wire
from voicemanager.protocol import MessageType, TurnEmptyMessage
from voicemanager.transport.turn_engine import TurnEngineSession

from tyger_fake_sidecar import FakeTygerSidecar
from tyger_ws_kit import StandInSidecar, barrier, is_type, make_app, opt_in, recv_until

REPO_ROOT = Path(__file__).resolve().parents[1]

# One silero-sized chunk (512 samples, 32 ms) per frame. Loud reads as speech to the
# amplitude stand-in, zeros as silence.
_SPEECH = b"\x11\x22" * 512
_SILENCE = b"\x00\x00" * 512
_N_SPEECH = 8
_N_SILENCE = 45   # past the 1 s silence window, so the speech detector ends the utterance
_LIMIT = 600


class _AmplitudeVad:
    """The silero net's place under the real VADIterator: a loud chunk is speech."""

    def __call__(self, x, sr):
        return torch.tensor(1.0 if float(x.abs().max()) > 0.01 else 0.0)

    def reset_states(self):
        pass


class _ScriptedSTT:
    """Gives one scripted final per flush of an utterance that was fed audio; "" when
    the script is used up, as a decoder that heard speech and found no words."""

    def __init__(self, finals: list[str]) -> None:
        self._finals = list(finals)
        self._segment = 0
        self._pending = False

    async def feed(self, data: bytes) -> bool:
        self._pending = True
        return False

    def get_current_text(self):
        return ("", "en", self._segment)

    def take_final(self):
        return None

    async def finalize(self):
        if not self._pending:
            return ("", "en", self._segment)
        self._pending = False
        return (self._finals.pop(0) if self._finals else "", "en", self._segment)

    def reset(self) -> None:
        self._segment += 1

    def dispose(self) -> None:
        pass


@pytest.fixture
def tyger():
    fake = FakeTygerSidecar().start()
    try:
        yield fake
    finally:
        fake.stop()


@pytest.fixture
def connect(tyger, monkeypatch):
    """The real handler on the audiocpp engine, with an STT that says ``finals``."""
    monkeypatch.setenv("VM_AUDIOCPP_SILENCE_MS", "1000")
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "60")
    cfg._settings = None

    def open_session(finals: list[str]):
        monkeypatch.setattr(audiocpp_stt, "AudioCppSTTPipeline",
                            lambda config=None, **kwargs: _ScriptedSTT(finals))
        app = make_app(sidecar=StandInSidecar(tyger.url))
        app.state.audiocpp_config = {"vad_model_factory": _AmplitudeVad}
        # Not entered as a context manager: that would run the real lifespan.
        return authkit.ws_connect(TestClient(app), "/ws/stream")

    yield open_session
    cfg._settings = None


def _opt_in(ws, tyger) -> None:
    opt_in(ws)
    assert tyger.wait_for(lambda: len(tyger.sessions) == 1)


def _say(ws, *, close: bool = True) -> list:
    """One utterance of loud frames, then with close the silence that ends it. The
    barrier returns once the handler has taken every frame: the frames it sent."""
    for _ in range(_N_SPEECH):
        ws.send_bytes(_SPEECH)
    if close:
        for _ in range(_N_SILENCE):
            ws.send_bytes(_SILENCE)
    return barrier(ws)


def _end_turn(ws, tyger, by: str, mic_sample: int) -> dict:
    """Report a turn end and read up to the frame it gives: that frame."""
    tyger.send(0, wire.EndpointEvent(by=by, mic_sample=mic_sample))
    done = lambda f: is_type("turn_empty")(f) or is_type("turn_end")(f)
    return recv_until(ws, done, limit=_LIMIT)[-1]


def _empty(mic_sample: int, by: str, speech: bool) -> dict:
    return {"type": "turn_empty", "mic_sample": mic_sample, "by": by, "speech": speech}


# --- the frame model ----------------------------------------------------------------

def test_the_frame_model_needs_every_field():
    assert MessageType.turn_empty.value == "turn_empty"
    frame = _empty(488960, "silence", True)
    assert json.loads(TurnEmptyMessage.model_validate(frame).model_dump_json()) == frame
    for missing in ("mic_sample", "by", "speech"):
        with pytest.raises(ValueError):
            TurnEmptyMessage.model_validate({k: v for k, v in frame.items() if k != missing})
    with pytest.raises(ValueError):
        TurnEmptyMessage(mic_sample=1, by="timeout", speech=True)


# --- the server's WebSocket handler -------------------------------------------------

@pytest.mark.parametrize("by", ["model", "silence"])
def test_speech_that_gave_no_words_sends_turn_empty_with_speech(connect, tyger, by):
    with connect([]) as ws:
        _opt_in(ws, tyger)
        assert _say(ws) == []                    # the utterance ended, with no final
        assert _end_turn(ws, tyger, by, 40960) == _empty(40960, by, True)


def test_an_open_utterance_flushed_empty_at_the_turn_end_sends_turn_empty_with_speech(
        connect, tyger):
    with connect([]) as ws:
        _opt_in(ws, tyger)
        assert _say(ws, close=False) == []       # the speech detector still hears speech
        assert _end_turn(ws, tyger, "model", 4096) == _empty(4096, "model", True)


def test_a_turn_end_with_no_speech_sends_turn_empty_without_speech(connect, tyger):
    with connect([]) as ws:
        _opt_in(ws, tyger)
        assert _end_turn(ws, tyger, "silence", 5120) == _empty(5120, "silence", False)
        for _ in range(10):
            ws.send_bytes(_SILENCE)              # audio, but no speech
        assert barrier(ws) == []
        assert _end_turn(ws, tyger, "silence", 10240) == _empty(10240, "silence", False)


def test_a_turn_end_clears_the_speech_heard_before_it(connect, tyger):
    with connect([]) as ws:
        _opt_in(ws, tyger)
        _say(ws)
        assert _end_turn(ws, tyger, "model", 40960)["speech"] is True
        assert _end_turn(ws, tyger, "silence", 51200) == _empty(51200, "silence", False)


def test_a_turn_end_with_text_sends_turn_end_only(connect, tyger):
    with connect(["one"]) as ws:
        _opt_in(ws, tyger)
        said = _say(ws)
        assert [f["text"] for f in said if f.get("is_final")] == ["one"]
        end = _end_turn(ws, tyger, "model", 40960)
        assert end["type"] == "turn_end" and end["text"] == "one"
        assert barrier(ws) == []                 # no turn_empty beside it
        # The speech of that turn does not count for the next one.
        assert _end_turn(ws, tyger, "silence", 51200) == _empty(51200, "silence", False)


def test_a_connection_that_does_not_opt_in_never_gets_turn_empty(connect, tyger):
    with connect([]) as ws:
        assert ws.receive_json()["type"] == "ready"
        assert _say(ws) == []
        assert tyger.sessions == []


# --- the turn engine session alone --------------------------------------------------

def _session() -> tuple[TurnEngineSession, list]:
    sent: list = []

    async def send_text(text: str) -> None:
        sent.append(json.loads(text))

    session = TurnEngineSession(sidecar=StandInSidecar("ws://127.0.0.1:1/tyger"),
                                stt_engine_type="audiocpp", label="s", send_text=send_text)
    session._ws = object()   # an open sidecar link: the session is active
    return session, sent


def _endpoint(mic_sample: int) -> wire.EndpointEvent:
    return wire.EndpointEvent(by="model", mic_sample=mic_sample)


def test_nothing_goes_out_once_the_sidecar_is_lost():
    session, sent = _session()
    session.note_speech()
    session._lost = True
    assert asyncio.run(session.send_turn_end(_endpoint(100))) is False
    assert sent == []


def test_nothing_goes_out_for_a_turn_end_from_before_a_listening_boundary():
    session, sent = _session()
    session.note_speech()
    session._boundary = 1000
    assert asyncio.run(session.send_turn_end(_endpoint(1000))) is False
    assert sent == []
    assert asyncio.run(session.send_turn_end(_endpoint(1001))) is False
    assert sent == [_empty(1001, "model", True)]


def test_a_listening_boundary_drops_the_speech_heard_before_it():
    session, sent = _session()
    session.note_speech()
    session.listen_boundary()
    asyncio.run(session.send_turn_end(_endpoint(100)))
    assert sent == [_empty(100, "model", False)]


def test_speech_does_not_count_before_the_opt_in():
    session, sent = _session()
    session._ws = None       # not opted in
    session.note_speech()
    session._ws = object()
    asyncio.run(session.send_turn_end(_endpoint(100)))
    assert sent == [_empty(100, "model", False)]


# --- the Python client --------------------------------------------------------------

class _FakeServerWS:
    """A socket whose incoming frames come from a queue."""

    def __init__(self) -> None:
        self._incoming: asyncio.Queue = asyncio.Queue()

    async def send(self, data) -> None:
        pass

    async def close(self) -> None:
        pass

    def push(self, frame) -> None:
        self._incoming.put_nowait(json.dumps(frame))

    def end(self) -> None:
        self._incoming.put_nowait(None)

    def __aiter__(self):
        return self._frames()

    async def _frames(self):
        while (frame := await self._incoming.get()) is not None:
            yield frame


def test_the_python_client_gives_turn_empty_frames_apart_from_turn_ends():
    ws = _FakeServerWS()
    client = VoiceManagerClient(token="vmp_test")
    client._ws = ws
    client.connected = True
    empty = _empty(488960, "silence", True)
    turn_end = {"type": "turn_end", "turn_id": 1, "text": "one", "mic_sample": 600000,
                "by": "model"}

    async def run():
        client._reader = asyncio.create_task(client._reader_loop())
        async with client.duplex() as session:
            ws.push(empty)
            ws.push(turn_end)
            ws.end()
            empties = [frame async for frame in session.turn_empties()]
            ends = [frame async for frame in session.turn_ends()]
            return empties, ends
    empties, ends = asyncio.run(run())
    assert empties == [empty]
    assert ends == [turn_end]


# --- the guide ----------------------------------------------------------------------

def test_the_guide_documents_the_frame_and_the_changelog_records_it():
    guide = (REPO_ROOT / "docs" / "client-integration.md").read_text(encoding="utf-8")
    heading = "#### `turn_empty` - `TurnEmptyMessage`"
    assert heading in guide
    body = guide.split(heading, 1)[1].split("\n### ", 1)[0]
    for name in TurnEmptyMessage.model_fields:
        assert re.search(rf"^\| `{name}` \|", body, re.M), name
    examples = [json.loads(block) for block in re.findall(r"```json\n(.*?)\n```", body, re.S)]
    assert examples and all(e["type"] == "turn_empty" for e in examples)
    for example in examples:
        TurnEmptyMessage.model_validate(example)
    assert "(#turn_empty---turnemptymessage)" in guide
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "`turn_empty`" in changelog
