"""The turn_end frame of an opted-in session, model-free.

When the turn engine reports a turn end (its endpoint head, or its silence
fallback), the server FLUSHes the open STT utterance, if there is one, sends that
final as today, and then sends {"type": "turn_end", "turn_id", "text",
"mic_sample", "by"}. text joins, in order, every final sent since the previous
turn_end. A turn end with no final text since the previous one sends no turn_end
(it sends a turn_empty: see tests/test_turn_empty.py).
Silero still ends utterances on its own, so finals keep flowing between turn ends.

The STT side is a scripted stand-in for the audiocpp pipeline that commits only at
a flush (as the shipping model does), behind the real silero endpointer with an
amplitude stand-in for the net. The sidecar is the in-process FakeTygerSidecar, so
a test pushes the turn end itself. No GPU, no model, no live server.
"""
import asyncio
import json

import pytest
import torch

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)
from fastapi.testclient import TestClient

import voicemanager.config as cfg
from voicemanager.pipeline import audiocpp_stt
from voicemanager.pipeline.tyger import wire
from voicemanager.transport.turn_engine import TurnEngineSession

from tyger_fake_sidecar import FakeTygerSidecar
from tyger_ws_kit import OPT_IN, StandInSidecar, barrier, is_type, make_app, opt_in, recv_until

# One silero-sized chunk (512 samples, 32 ms) per frame. Loud reads as speech to the
# amplitude stand-in, zeros as silence.
_SPEECH = b"\x11\x22" * 512
_SILENCE = b"\x00\x00" * 512
_N_SPEECH = 8
# 45 frames of 32 ms of silent audio: past the 1 s silence window while frames keep
# coming, so silero ends the utterance. The 1 s window also keeps the frame-gap trigger
# from flushing an open utterance before the test reports its turn end.
_N_SILENCE = 45
_SILENCE_MS = "1000"
_LIMIT = 600


class _AmplitudeVad:
    """The silero net's place under the real VADIterator: a loud chunk is speech."""

    def __call__(self, x, sr):
        return torch.tensor(1.0 if float(x.abs().max()) > 0.01 else 0.0)

    def reset_states(self):
        pass


class _FlushOnlySTT:
    """Commits one scripted final per flush of an utterance that was fed audio.

    Each loud frame fed shows as the interim "heard N". ``overlaps`` counts calls that
    started while another feed or finalize was still running: the pipeline must never
    be driven from two tasks at once.
    """

    def __init__(self, finals: list[str], delay: float = 0.0) -> None:
        self._finals = list(finals)
        self._delay = delay
        self._segment = 0
        self._loud = 0
        self._pending = False
        self._busy = 0
        self.overlaps = 0
        self.flushes = 0

    async def _work(self) -> None:
        if self._busy:
            self.overlaps += 1
        self._busy += 1
        try:
            await asyncio.sleep(self._delay)
        finally:
            self._busy -= 1

    async def feed(self, data: bytes) -> bool:
        await self._work()
        self._pending = True
        if any(data):
            self._loud += 1
        return True

    def get_current_text(self):
        return (f"heard {self._loud}" if self._loud else "", "en", self._segment)

    def take_final(self):
        return None

    async def finalize(self):
        await self._work()
        if not self._pending:
            return ("", "en", self._segment)
        self._pending = False
        self.flushes += 1
        return (self._finals.pop(0) if self._finals else "", "en", self._segment)

    def reset(self) -> None:
        self._segment += 1
        self._loud = 0

    def dispose(self) -> None:
        pass


# --- fixtures -------------------------------------------------------------------

@pytest.fixture
def tyger():
    fake = FakeTygerSidecar().start()
    try:
        yield fake
    finally:
        fake.stop()


@pytest.fixture
def silence_window(monkeypatch):
    monkeypatch.setenv("VM_AUDIOCPP_SILENCE_MS", _SILENCE_MS)
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "60")
    cfg._settings = None
    yield
    cfg._settings = None


def _client(tyger, monkeypatch, finals: list[str], delay: float = 0.0):
    """A client on the audiocpp engine whose pipelines say ``finals``; the pipelines."""
    made: list[_FlushOnlySTT] = []

    def factory(config=None, **kwargs):
        made.append(_FlushOnlySTT(finals, delay))
        return made[-1]

    monkeypatch.setattr(audiocpp_stt, "AudioCppSTTPipeline", factory)
    app = make_app(sidecar=StandInSidecar(tyger.url))
    app.state.audiocpp_config = {"vad_model_factory": _AmplitudeVad}
    return TestClient(app), made


# --- helpers --------------------------------------------------------------------

def _transcription(final: bool):
    return lambda f: (isinstance(f, dict) and f.get("type") == "transcription"
                      and f["is_final"] is final)


def _heard(n: int):
    return lambda f: _transcription(False)(f) and f["text"] == f"heard {n}"


def _say(ws, *, close: bool = True) -> None:
    """One utterance of loud frames; with close, then enough silence for silero to end it."""
    for _ in range(_N_SPEECH):
        ws.send_bytes(_SPEECH)
    if close:
        for _ in range(_N_SILENCE):
            ws.send_bytes(_SILENCE)


def _say_closed(ws) -> list:
    """Say one utterance that silero ends; every frame up to its final."""
    _say(ws)
    return recv_until(ws, _transcription(True), limit=_LIMIT)


def _say_open(ws) -> list:
    """Say one utterance and leave it open; every frame up to its last interim."""
    _say(ws, close=False)
    return recv_until(ws, _heard(_N_SPEECH), limit=_LIMIT)


def _opt_in(ws, tyger) -> None:
    opt_in(ws)
    assert tyger.wait_for(lambda: len(tyger.sessions) == 1)


def _end_turn(tyger, by: str, mic_sample: int) -> None:
    tyger.send(0, wire.EndpointEvent(by=by, mic_sample=mic_sample))


def _finals(frames: list) -> list[str]:
    return [f["text"] for f in frames if _transcription(True)(f)]


def _turn_ends(frames: list) -> list[dict]:
    return [f for f in frames if is_type("turn_end")(f)]


# --- the spec case --------------------------------------------------------------

@pytest.mark.parametrize("by", ["model", "silence"])
def test_turn_end_flushes_the_open_utterance_and_joins_every_final(
        tyger, silence_window, monkeypatch, by):
    client, made = _client(tyger, monkeypatch, [" one ", "two", "three  "])
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _opt_in(ws, tyger)
        frames = _say_closed(ws) + _say_closed(ws)
        assert _finals(frames) == [" one ", "two"]
        frames += _say_open(ws)
        assert made[0].flushes == 2  # the third utterance is still open

        _end_turn(tyger, by, 40960)
        tail = recv_until(ws, is_type("turn_end"), limit=_LIMIT)

    # The flush of the open utterance sent its final, then the turn_end came.
    assert _finals(tail) == ["three  "]
    assert made[0].flushes == 3
    assert tail[-1] == {"type": "turn_end", "turn_id": 0, "text": "one two three",
                        "mic_sample": 40960, "by": by}
    assert _turn_ends(frames + tail[:-1]) == []
    assert made[0].overlaps == 0


def test_a_turn_end_with_no_utterance_open_sends_the_finals_already_sent(
        tyger, silence_window, monkeypatch):
    client, made = _client(tyger, monkeypatch, ["alpha"])
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _opt_in(ws, tyger)
        _say_closed(ws)
        _end_turn(tyger, "model", 5120)
        tail = recv_until(ws, is_type("turn_end"), limit=_LIMIT)

    assert _finals(tail) == []
    assert tail[-1]["text"] == "alpha"
    assert made[0].flushes == 1  # nothing was open, so nothing more was flushed


# --- no final text, no turn_end -------------------------------------------------

def test_a_turn_end_with_no_speech_fed_sends_no_turn_end(tyger, silence_window, monkeypatch):
    client, made = _client(tyger, monkeypatch, ["later words"])
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _opt_in(ws, tyger)
        _end_turn(tyger, "model", 5120)          # nothing said yet
        ws.send_bytes(_SILENCE)                  # audio, but no speech
        _end_turn(tyger, "silence", 10240)       # still nothing said
        frames = barrier(ws)
        _say_closed(ws)
        _end_turn(tyger, "model", 99999)
        frames += recv_until(ws, is_type("turn_end"), limit=_LIMIT)

    # The first turn_end the client gets is the one after the speech.
    turn_ends = _turn_ends(frames)
    assert len(turn_ends) == 1
    assert turn_ends[0]["mic_sample"] == 99999
    assert turn_ends[0]["text"] == "later words"
    assert made[0].flushes == 1


def test_a_second_turn_end_with_nothing_new_sends_no_turn_end(
        tyger, silence_window, monkeypatch):
    client, _ = _client(tyger, monkeypatch, ["first", "second"])
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _opt_in(ws, tyger)
        _say_closed(ws)
        _end_turn(tyger, "model", 1000)
        _end_turn(tyger, "silence", 2000)        # nothing new since the first
        frames = recv_until(ws, is_type("turn_end"), limit=_LIMIT)
        _say_closed(ws)
        _end_turn(tyger, "model", 3000)
        frames += recv_until(ws, is_type("turn_end"), limit=_LIMIT)

    assert [(t["text"], t["mic_sample"]) for t in _turn_ends(frames)] == [
        ("first", 1000), ("second", 3000)]


# --- turn ids and finals between turn ends ---------------------------------------

def test_turn_end_carries_the_turn_of_its_finals(tyger, silence_window, monkeypatch):
    client, _ = _client(tyger, monkeypatch, ["one", "two"])
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _opt_in(ws, tyger)
        barrier(ws)
        barrier(ws)                              # turn 2 is open
        frames = _say_open(ws)
        _end_turn(tyger, "model", 5120)
        frames += recv_until(ws, is_type("turn_end"), limit=_LIMIT)
        barrier(ws)                              # turn 3 is open
        frames += _say_open(ws)
        _end_turn(tyger, "silence", 10240)
        frames += recv_until(ws, is_type("turn_end"), limit=_LIMIT)

    finals = [f for f in frames if _transcription(True)(f)]
    assert [(f["text"], f["turn_id"]) for f in finals] == [("one", 2), ("two", 3)]
    assert [(t["text"], t["turn_id"]) for t in _turn_ends(frames)] == [
        ("one", 2), ("two", 3)]


def test_finals_keep_flowing_between_turn_ends(tyger, silence_window, monkeypatch):
    client, _ = _client(tyger, monkeypatch, ["one", "two", "three"])
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _opt_in(ws, tyger)
        frames = _say_closed(ws)
        _end_turn(tyger, "model", 5120)
        frames += recv_until(ws, is_type("turn_end"), limit=_LIMIT)
        frames += _say_closed(ws)
        frames += _say_closed(ws)
        _end_turn(tyger, "silence", 30720)
        frames += recv_until(ws, is_type("turn_end"), limit=_LIMIT)

    kinds = [("final", f["text"]) if _transcription(True)(f) else ("turn_end", f["text"])
             for f in frames if _transcription(True)(f) or is_type("turn_end")(f)]
    assert kinds == [("final", "one"), ("turn_end", "one"), ("final", "two"),
                     ("final", "three"), ("turn_end", "two three")]


def test_the_flush_is_serialized_with_the_handler_stt_work(
        tyger, silence_window, monkeypatch):
    # Each feed takes a while, so the turn end lands while frames are still being fed.
    client, made = _client(tyger, monkeypatch, ["early", "late"], delay=0.01)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        _opt_in(ws, tyger)
        for _ in range(3 * _N_SPEECH):
            ws.send_bytes(_SPEECH)
        frames = recv_until(ws, _heard(1), limit=_LIMIT)  # the handler is feeding now
        _end_turn(tyger, "model", 5120)
        frames += recv_until(ws, is_type("turn_end"), limit=_LIMIT)
        for _ in range(_N_SILENCE):
            ws.send_bytes(_SILENCE)
        frames += recv_until(ws, _transcription(True), limit=_LIMIT)

    assert made[0].overlaps == 0
    turn_end = _turn_ends(frames)[0]
    before = frames[:frames.index(turn_end)]
    assert _finals(before) == ["early"] and turn_end["text"] == "early"
    assert _finals(frames) == ["early", "late"]


# --- a connection that does not opt in ------------------------------------------

def test_a_connection_that_does_not_opt_in_never_gets_turn_end(
        tyger, silence_window, monkeypatch):
    client, _ = _client(tyger, monkeypatch, ["one", "two"])
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        frames = _say_closed(ws) + _say_closed(ws)
        frames += barrier(ws)

    assert _finals(frames) == ["one", "two"]
    assert _turn_ends(frames) == []
    assert tyger.sessions == []


# --- the session object on its own ----------------------------------------------

def test_the_session_joins_stripped_finals_and_skips_empty_ones(tyger):
    async def scenario():
        sent: list[dict] = []

        async def send_text(text: str) -> None:
            sent.append(json.loads(text))

        session = TurnEngineSession(sidecar=StandInSidecar(tyger.url),
                                    stt_engine_type="audiocpp", label="unit",
                                    send_text=send_text)
        session.note_final("before the opt-in", 0)   # not opted in: not collected
        await session.handle_opt_in(dict(OPT_IN), audio_started=False)
        sent.clear()
        session.note_final("  alpha ", 1)
        session.note_final("   ", 1)
        session.note_final("beta", 2)
        assert await session.send_turn_end(wire.EndpointEvent(by="silence", mic_sample=7))
        assert not await session.send_turn_end(wire.EndpointEvent(by="model", mic_sample=8))
        await session.aclose()
        return sent

    sent = asyncio.run(scenario())
    # The second turn end has nothing new: it gives a turn_empty, not a turn_end.
    assert sent == [{"type": "turn_end", "turn_id": 2, "text": "alpha beta",
                     "mic_sample": 7, "by": "silence"},
                    {"type": "turn_empty", "mic_sample": 8, "by": "model", "speech": False}]
