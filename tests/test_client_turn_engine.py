"""The client's turn engine on a DuplexSession: opt-in, playback reports, barge and turn_end.

Model-free and server-free. The socket is the same AsyncMock fake that
tests/test_client_duplex.py uses, and the frames the server would send go through the
client's real _reader_loop from a queue-fed fake socket, so routing is tested on what
the reader does. Covers: the opt-in sends turn_engine and returns on turn_engine_ready;
an error frame raises with its error_type; the opt-in refuses after audio and a second
time; the playback frames are well formed (checked against the protocol models); the
session counts the mic samples it sent; barge reaches audio() in order with the pcm
and turn_end reaches turn_ends(); turn_engine_lost clears the opt-in; and a session
that never opts in sends none of these frames.
"""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from voicemanager.client import TurnEngineError, VoiceManagerClient, VoiceManagerError
from voicemanager.protocol import (
    INVALID_REQUEST,
    TURN_ENGINE_LOST,
    TURN_ENGINE_UNAVAILABLE,
    TURN_ENGINE_UNSUPPORTED,
    BargeMessage,
    PlaybackMessage,
    TurnEngineMessage,
    TurnEngineReadyMessage,
    TurnEndMessage,
)


class _FakeServerWS:
    """A socket whose incoming frames come from a queue; ``send`` is an AsyncMock.

    ``replies`` maps a sent text frame's type to the frames the fake server answers
    with, so an awaited request sees its reply arrive through the reader loop.
    """

    def __init__(self, replies: dict[str, list[dict]] | None = None) -> None:
        self._incoming: asyncio.Queue = asyncio.Queue()
        self.replies = replies or {}
        self.send = AsyncMock(side_effect=self._on_send)
        self.close = AsyncMock()

    async def _on_send(self, data) -> None:
        if isinstance(data, str):
            for frame in self.replies.get(json.loads(data).get("type"), []):
                self.push(frame)

    def push(self, frame) -> None:
        self._incoming.put_nowait(frame if isinstance(frame, bytes) else json.dumps(frame))

    def end(self) -> None:
        self._incoming.put_nowait(None)

    def __aiter__(self):
        return self._frames()

    async def _frames(self):
        while (frame := await self._incoming.get()) is not None:
            yield frame


def _client(ws=None) -> VoiceManagerClient:
    client = VoiceManagerClient(token="vmp_test")
    client._ws = ws or AsyncMock()
    client.connected = True
    return client


def _start_reader(client: VoiceManagerClient) -> None:
    client._reader = asyncio.create_task(client._reader_loop())


def _sent(client: VoiceManagerClient) -> list:
    """Every frame sent on the socket: dicts for text frames, bytes as they are."""
    out = []
    for call in client._ws.send.call_args_list:
        data = call.args[0]
        out.append(json.loads(data) if isinstance(data, str) else data)
    return out


def _ready(engine: str = "tyger") -> dict:
    return TurnEngineReadyMessage(engine=engine).model_dump()


def _error(error_type: str, message: str = "no") -> dict:
    return {"type": "error", "error_type": error_type, "message": message}


# --- the opt-in ----------------------------------------------------------------------

def test_enable_turn_engine_sends_the_frame_and_returns_on_ready():
    ws = _FakeServerWS({"turn_engine": [_ready()]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            engine = await session.enable_turn_engine("tyger")
            assert session.turn_engine == "tyger"
            ws.end()
            return engine
    assert asyncio.run(run()) == "tyger"
    sent = _sent(client)
    assert sent == [{"type": "turn_engine", "engine": "tyger"}]
    TurnEngineMessage.model_validate(sent[0])


def test_enable_turn_engine_waits_for_a_late_ready():
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            task = asyncio.create_task(session.enable_turn_engine())
            await asyncio.sleep(0.01)
            assert not task.done()
            ws.push(_ready())
            engine = await asyncio.wait_for(task, 2.0)
            ws.end()
            return engine
    assert asyncio.run(run()) == "tyger"


@pytest.mark.parametrize("error_type", [
    TURN_ENGINE_UNAVAILABLE, TURN_ENGINE_UNSUPPORTED, INVALID_REQUEST])
def test_enable_turn_engine_raises_on_an_error_frame(error_type):
    ws = _FakeServerWS({"turn_engine": [_error(error_type, "refused")]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            with pytest.raises(TurnEngineError) as info:
                await session.enable_turn_engine("tyger")
            assert session.turn_engine is None
            ws.end()
            return info.value
    exc = asyncio.run(run())
    assert isinstance(exc, VoiceManagerError)
    assert exc.error_type == error_type
    assert "refused" in str(exc)


def test_the_opt_ins_error_does_not_reach_errors():
    ws = _FakeServerWS({"turn_engine": [_error(TURN_ENGINE_UNAVAILABLE)]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            with pytest.raises(TurnEngineError):
                await session.enable_turn_engine()
            ws.push(_error("timeout_warning", "closing soon"))
            ws.end()
            return [frame async for frame in session.errors()]
    errors = asyncio.run(run())
    assert [e["error_type"] for e in errors] == ["timeout_warning"]


def test_enable_turn_engine_times_out():
    client = _client(_FakeServerWS())

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            with pytest.raises(TurnEngineError, match="Timed out") as info:
                await session.enable_turn_engine(timeout=0.01)
            client._ws.end()
            return info.value
    assert asyncio.run(run()).error_type is None


def test_enable_turn_engine_raises_when_the_connection_closes_while_waiting():
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            task = asyncio.create_task(session.enable_turn_engine())
            await asyncio.sleep(0.01)
            ws.end()
            with pytest.raises(TurnEngineError, match="Connection lost"):
                await asyncio.wait_for(task, 2.0)
    asyncio.run(run())


def test_enable_turn_engine_refuses_after_audio_was_sent():
    client = _client()

    async def run():
        async with client.duplex() as session:
            await session.send_audio(b"\x00\x00" * 160)
            with pytest.raises(VoiceManagerError, match="first audio frame"):
                await session.enable_turn_engine("tyger")
    asyncio.run(run())
    assert all(isinstance(f, bytes) for f in _sent(client))


def test_enable_turn_engine_refuses_a_second_opt_in():
    ws = _FakeServerWS({"turn_engine": [_ready()]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.enable_turn_engine()
            with pytest.raises(VoiceManagerError, match="already"):
                await session.enable_turn_engine()
            ws.end()
    asyncio.run(run())
    assert [f["type"] for f in _sent(client)] == ["turn_engine"]


def test_enable_turn_engine_refuses_on_a_closed_session():
    client = _client()

    async def run():
        session = client.duplex()
        await session.close()
        with pytest.raises(VoiceManagerError, match="closed"):
            await session.enable_turn_engine()
    asyncio.run(run())
    assert _sent(client) == []


# --- mic samples and playback reports -------------------------------------------------

def test_mic_sample_counts_the_samples_sent():
    client = _client()

    async def run():
        async with client.duplex() as session:
            assert session.mic_sample == 0
            await session.send_audio(b"\x00\x00" * 320)
            await session.send_audio(b"\x00\x00" * 160)
            return session.mic_sample
    assert asyncio.run(run()) == 480


def test_playback_start_and_stop_send_well_formed_frames():
    client = _client()

    async def run():
        async with client.duplex() as session:
            req = session.tts_request()
            await session.playback_start(req.request_id, mic_sample=16000)
            await session.playback_stop(req.request_id, mic_sample=40000, reply_sample=24000)
            return req.request_id
    rid = asyncio.run(run())
    start, stop = _sent(client)
    assert start == {"type": "playback", "request_id": rid, "event": "start",
                     "mic_sample": 16000}
    assert stop == {"type": "playback", "request_id": rid, "event": "stop",
                    "mic_sample": 40000, "reply_sample": 24000}
    for frame in (start, stop):
        assert PlaybackMessage.model_validate(frame).model_dump() == frame


def test_playback_defaults_mic_sample_to_the_samples_sent():
    client = _client()

    async def run():
        async with client.duplex() as session:
            await session.send_audio(b"\x00\x00" * 800)
            await session.playback_start("r1")
            await session.send_audio(b"\x00\x00" * 200)
            await session.playback_stop("r1", reply_sample=5)
    asyncio.run(run())
    start, stop = [f for f in _sent(client) if isinstance(f, dict)]
    assert start["mic_sample"] == 800
    assert stop["mic_sample"] == 1000 and stop["reply_sample"] == 5


def test_playback_takes_a_tts_request_for_its_id():
    client = _client()

    async def run():
        async with client.duplex() as session:
            req = session.tts_request(request_id="reply-1")
            await session.playback_start(req, mic_sample=0)
    asyncio.run(run())
    assert _sent(client)[0]["request_id"] == "reply-1"


@pytest.mark.parametrize("call, kwargs", [
    ("playback_stop", {"mic_sample": 10}),                    # stop needs reply_sample
    ("playback_stop", {"mic_sample": 10, "reply_sample": -1}),
    ("playback_stop", {"mic_sample": 10, "reply_sample": 1.5}),
    ("playback_stop", {"mic_sample": 10, "reply_sample": True}),
    ("playback_start", {"mic_sample": -1}),
    ("playback_start", {"mic_sample": "10"}),
    ("playback_start", {"mic_sample": 2.0}),
    ("playback_start", {"mic_sample": False}),
])
def test_playback_refuses_bad_arguments_before_any_frame(call, kwargs):
    client = _client()

    async def run():
        async with client.duplex() as session:
            with pytest.raises((ValueError, TypeError)):
                await getattr(session, call)("r1", **kwargs)
    asyncio.run(run())
    assert _sent(client) == []


def test_playback_refuses_an_empty_request_id():
    client = _client()

    async def run():
        async with client.duplex() as session:
            with pytest.raises(ValueError):
                await session.playback_start("", mic_sample=0)
    asyncio.run(run())
    assert _sent(client) == []


# --- barge and turn_end reach the caller ------------------------------------------------

def test_barge_reaches_audio_in_order_and_the_requests_late_pcm_is_dropped():
    ws = _FakeServerWS()
    client = _client(ws)
    barge = BargeMessage(request_id="r1", onset_mic_sample=32000,
                         mic_sample=36000).model_dump()

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            ws.push({"type": "audio_start", "request_id": "r1", "sample_rate": 24000})
            ws.push(b"\x01\x01")
            ws.push(barge)
            ws.push(b"\x01\x02")  # pcm that was in flight: dropped like a cancel
            ws.push({"type": "audio_end", "request_id": "r1", "reason": "interrupted"})
            ws.end()
            return [event async for event in session.audio()]
    events = asyncio.run(run())
    assert events == [
        ("audio_start", {"type": "audio_start", "request_id": "r1", "sample_rate": 24000}),
        ("pcm", b"\x01\x01", "r1"),
        ("barge", barge),
        ("audio_end", {"type": "audio_end", "request_id": "r1", "reason": "interrupted"}),
    ]
    assert "r1" not in client._cancelled_tts


def test_turn_end_reaches_turn_ends_and_not_transcripts():
    ws = _FakeServerWS()
    client = _client(ws)
    turn_end = TurnEndMessage(turn_id=3, text="hello there", mic_sample=48000,
                              by="model").model_dump()

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            ws.push({"type": "transcription", "text": "hello there", "is_final": True,
                     "segment_index": 0, "turn_id": 3})
            ws.push(turn_end)
            ws.end()
            turns = [frame async for frame in session.turn_ends()]
            transcripts = [frame async for frame in session.transcripts()]
            return turns, transcripts
    turns, transcripts = asyncio.run(run())
    assert turns == [turn_end]
    assert [t["type"] for t in transcripts] == ["transcription"]


def test_turn_engine_lost_reaches_errors_and_clears_the_opt_in():
    ws = _FakeServerWS({"turn_engine": [_ready()]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.enable_turn_engine()
            ws.push(_error(TURN_ENGINE_LOST, "the turn engine exited"))
            ws.end()
            errors = [frame async for frame in session.errors()]
            return errors, session.turn_engine
    errors, engine = asyncio.run(run())
    assert [e["error_type"] for e in errors] == [TURN_ENGINE_LOST]
    assert engine is None


def test_an_unsolicited_ready_is_ignored():
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            ws.push(_ready())
            ws.end()
            await asyncio.wait_for(client._reader, 2.0)
            return session.turn_engine
    assert asyncio.run(run()) is None


# --- a session that never opts in ------------------------------------------------------

def test_a_session_that_never_opts_in_sends_no_turn_engine_frames():
    client = _client()

    async def run():
        async with client.duplex() as session:
            await session.send_audio(b"\x00\x00" * 160)
            req = session.tts_request()
            await req.send("hello", flush=True)
            await req.done()
    asyncio.run(run())
    types = [f["type"] for f in _sent(client) if isinstance(f, dict)]
    assert types == ["tts_stream", "tts_stream"]


def test_a_reconnect_resets_the_opt_in_and_the_mic_count(monkeypatch):
    client = _client()
    client.turn_engine = "tyger"
    client._mic_bytes = 640
    real_sleep = asyncio.sleep

    async def _no_wait(_delay):
        await real_sleep(0)

    async def _fail():
        raise OSError("down")

    # _reconnect waits between attempts; skip the waits.
    monkeypatch.setattr("voicemanager.client.asyncio.sleep", _no_wait)
    monkeypatch.setattr(client, "_connect", _fail)

    async def run():
        with pytest.raises(VoiceManagerError):
            await client._reconnect()
    asyncio.run(run())
    assert client.turn_engine is None
    assert client._mic_bytes == 0
