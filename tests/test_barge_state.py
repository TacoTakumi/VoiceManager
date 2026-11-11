"""The barge_state frame: a client learns when the pre-filter can give a barge.

With the pre-filter a session gets no barge until the filter has learned the user's
voice. The sidecar reports that state on its event stream (``learning`` when the
session opens, ``ready`` at the hop the filter enrolled on) and the server sends
each report on to the client as a barge_state frame, never before
turn_engine_ready. With no pre-filter neither end sends anything.

Covers the sidecar's event line, the sidecar app with and without a pre-filter, the
server's WebSocket handler with the in-process fake sidecar, the Python client, and
the client guide. Model-free and server-free.
"""
import asyncio
import contextlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)
from fastapi.testclient import TestClient

from voicemanager.client import VoiceManagerClient
from voicemanager.pipeline import audiocpp_stt
from voicemanager.pipeline.tyger import server, wire
from voicemanager.pipeline.tyger.scorer import FrameScore
from voicemanager.protocol import TURN_ENGINE_LOST, BargeStateMessage, MessageType
from voicemanager.transport.turn_engine import TurnEngineSession

from tyger_fake_sidecar import FakeTygerSidecar
from tyger_ws_kit import (
    OPT_IN, ScriptedAudioCppSTT, StandInSidecar, barrier, make_app, opt_in,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LEARNING = {"type": "barge_state", "state": "learning"}
READY = {"type": "barge_state", "state": "ready"}


# --- the sidecar's event line ------------------------------------------------------

@pytest.mark.parametrize("state", wire.BARGE_STATES)
def test_the_event_line_round_trips(state):
    line = wire.format_event(wire.BargeStateEvent(state))
    assert json.loads(line) == {"event": "barge_state", "state": state}
    assert wire.parse_event_line(line) == wire.BargeStateEvent(state)


@pytest.mark.parametrize("line", [
    '{"event":"barge_state","state":"done"}',
    '{"event":"barge_state","state":1}',
    '{"event":"barge_state"}',
])
def test_a_bad_event_line_is_refused(line):
    with pytest.raises(ValueError):
        wire.parse_event_line(line)


def test_an_unknown_state_is_not_formatted():
    with pytest.raises(ValueError):
        wire.format_event(wire.BargeStateEvent("done"))


# --- the sidecar app ---------------------------------------------------------------

class _QuietSession:
    def reset(self):
        pass

    def score_hop(self, user, agent, mode):
        return iter([FrameScore(0.0, 0.0, -90.0)] * 4)


class _QuietScorer:
    name = "quiet"

    def new_session(self, key, index):
        return _QuietSession()


class _SilentFactory:
    """Enrolls at once into a filter that outputs silence."""

    def enroll(self, sample):
        class _Filter:
            delay = 384

            def push(self, samples):
                return np.zeros(len(samples), np.float32)
        return _Filter()


LISTEN_HOPS = 20   # 6.4 s of voice: 2 s calibrate the gate, then 3 s enroll the filter


def _sidecar_messages(prefilter_factory, progress: bool = True) -> list[list]:
    """The events of each message a session gets: the opening one, then one per hop."""
    app = server.create_app(turn_scorer=_QuietScorer(), prefilter_factory=prefilter_factory)
    rng = np.random.default_rng(5)
    silent = bytes(wire.HOP_SAMPLES * 2)
    query = f"session=p&{wire.PROGRESS_PARAM}=1" if progress else "session=p"
    messages = []
    with TestClient(app) as client, client.websocket_connect(f"/tyger?{query}") as ws:
        if progress or prefilter_factory is not None:
            messages.append(wire.parse_events(ws.receive_text()))
        for k in range(LISTEN_HOPS + 2):
            voiced = (rng.uniform(-0.1, 0.1, wire.HOP_SAMPLES) * 32768).astype("<i2").tobytes()
            ws.send_bytes(wire.pack_hop(wire.MODE_LISTENING, (k + 1) * wire.HOP_SAMPLES,
                                        voiced, silent))
            if progress:
                messages.append(wire.parse_events(ws.receive_text()))
        if not progress:
            # Without progress lines only the enrollment gives a message.
            messages.append(wire.parse_events(ws.receive_text()))
    return messages


def _states(messages: list[list]) -> list[str]:
    return [e.state for m in messages for e in m if isinstance(e, wire.BargeStateEvent)]


def test_the_sidecar_reports_learning_at_the_open_and_ready_at_enrollment():
    messages = _sidecar_messages(_SilentFactory())
    assert messages[0] == [wire.BargeStateEvent("learning"), wire.ScoredEvent(mic_sample=0)]
    assert _states(messages) == ["learning", "ready"]
    enrolled = [m for m in messages[1:] if wire.BargeStateEvent("ready") in m]
    assert len(enrolled) == 1
    # In the message of the hop it enrolled on, before that hop's scored line. 2 s
    # of calibration and 3 s of voice put that hop past 5 s of mic audio.
    ready, scored = enrolled[0]
    assert ready == wire.BargeStateEvent("ready")
    assert isinstance(scored, wire.ScoredEvent)
    assert 5 * wire.SAMPLE_RATE < scored.mic_sample <= LISTEN_HOPS * wire.HOP_SAMPLES


def test_the_sidecar_reports_the_state_to_a_session_that_asked_for_no_progress():
    messages = _sidecar_messages(_SilentFactory(), progress=False)
    assert messages == [[wire.BargeStateEvent("learning")], [wire.BargeStateEvent("ready")]]


def test_the_sidecar_reports_no_state_without_a_pre_filter():
    messages = _sidecar_messages(None)
    assert _states(messages) == []
    assert messages[0] == [wire.ScoredEvent(mic_sample=0)]


# --- the server's WebSocket handler ------------------------------------------------

@pytest.fixture
def connect(monkeypatch):
    """Open the real handler against a fake sidecar that opens with ``opening``."""
    monkeypatch.setattr(audiocpp_stt, "AudioCppSTTPipeline", ScriptedAudioCppSTT)
    fakes = []

    @contextlib.contextmanager
    def open_session(opening=None):
        fake = FakeTygerSidecar(opening=opening).start()
        fakes.append(fake)
        # Not entered as a context manager: that would run the real lifespan.
        client = TestClient(make_app(sidecar=StandInSidecar(fake.url)))
        with authkit.ws_connect(client, "/ws/stream") as ws:
            yield ws, fake

    yield open_session
    for fake in fakes:
        fake.stop()


def test_the_client_gets_learning_after_turn_engine_ready_and_ready_at_enrollment(connect):
    with connect(opening=[wire.BargeStateEvent("learning")]) as (ws, fake):
        opt_in(ws)   # reads turn_engine_ready as the first frame after the opt-in
        assert ws.receive_json() == LEARNING
        fake.send(0, wire.BargeStateEvent("ready"))
        assert ws.receive_json() == READY
        assert barrier(ws) == []


def test_the_client_gets_no_barge_state_without_a_pre_filter(connect):
    with connect() as (ws, fake):
        opt_in(ws)
        assert fake.wait_for(lambda: len(fake.sessions) == 1)
        assert barrier(ws) == []


def test_a_state_reported_before_turn_engine_ready_waits_for_it():
    sent = []

    async def send_text(text: str) -> None:
        sent.append(json.loads(text))

    session = TurnEngineSession(sidecar=StandInSidecar("ws://127.0.0.1:1/tyger"),
                                stt_engine_type="audiocpp", label="s", send_text=send_text)

    async def connect_and_report() -> bool:
        # The sidecar's opening line reaches the reader while the opt-in is answered.
        session._ws = object()
        await session._dispatch(wire.BargeStateEvent("learning"))
        assert sent == []
        return True

    session._connect = connect_and_report
    asyncio.run(session.handle_opt_in(dict(OPT_IN), audio_started=False))
    assert sent == [{"type": "turn_engine_ready", "engine": "tyger"}, LEARNING]


# --- the Python client -------------------------------------------------------------

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


def _client(ws) -> VoiceManagerClient:
    client = VoiceManagerClient(token="vmp_test")
    client._ws = ws
    client.connected = True
    return client


def test_the_python_client_gives_the_frames_and_keeps_the_last_state():
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        client._reader = asyncio.create_task(client._reader_loop())
        async with client.duplex() as session:
            before = session.barge_state
            ws.push(LEARNING)
            ws.push(READY)
            ws.end()
            frames = [frame async for frame in session.barge_states()]
            return before, frames, session.barge_state
    before, frames, after = asyncio.run(run())
    assert before is None
    assert frames == [LEARNING, READY]
    assert after == "ready"


def test_turn_engine_lost_clears_the_python_clients_barge_state():
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        client._reader = asyncio.create_task(client._reader_loop())
        async with client.duplex() as session:
            ws.push(LEARNING)
            ws.push({"type": "error", "error_type": TURN_ENGINE_LOST, "message": "gone"})
            ws.end()
            [frame async for frame in session.errors()]
            return session.barge_state
    assert asyncio.run(run()) is None


# --- the frame model and the guide -------------------------------------------------

def test_the_frame_model_takes_the_two_states_only():
    assert MessageType.barge_state.value == "barge_state"
    for frame in (LEARNING, READY):
        assert json.loads(BargeStateMessage.model_validate(frame).model_dump_json()) == frame
    with pytest.raises(ValueError):
        BargeStateMessage(state="done")


def test_the_guide_documents_the_frame_and_the_changelog_records_it():
    guide = (REPO_ROOT / "docs" / "client-integration.md").read_text(encoding="utf-8")
    heading = "#### `barge_state` - `BargeStateMessage`"
    assert heading in guide
    body = guide.split(heading, 1)[1].split("\n#### ", 1)[0]
    for name in BargeStateMessage.model_fields:
        assert re.search(rf"^\| `{name}` \|", body, re.M), name
    examples = [json.loads(block) for block in re.findall(r"```json\n(.*?)\n```", body, re.S)]
    assert examples == [LEARNING, READY]
    for example in examples:
        BargeStateMessage.model_validate(example)
    assert "(#barge_state---bargestatemessage)" in guide
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "`barge_state`" in changelog
