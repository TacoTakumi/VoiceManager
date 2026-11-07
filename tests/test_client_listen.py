"""The listening boundary helpers on a DuplexSession: listen_stop, listen_start, listening.

Model-free and server-free. The frames the server would send go through the client's
real _reader_loop from a queue-fed fake socket, as in tests/test_client_turn_engine.py.
Covers: listen_stop sends the frame and returns the ack's mic_sample; a count that
differs from the session's own logs a warning; no ack in time (an older server) or a
closed connection returns None and the session stays stopped; a late ack after a
timeout does not answer the next listen_stop; while stopped send_audio sends and counts
nothing; listen_start sends the frame and audio flows and counts again; both refuse on
a closed session; a reconnect starts listening again; closing a stopped session sends
listen_start, so the one-sided methods and the next session listen again, while closing
a listening session sends nothing and closing on a gone connection raises nothing; and
a session that never calls them is unchanged.
"""
import asyncio
import json
import logging
from unittest.mock import AsyncMock

import pytest
import websockets

from voicemanager.client import VoiceManagerClient, VoiceManagerError
from voicemanager.protocol import (
    ListenStartMessage,
    ListenStoppedMessage,
    ListenStopMessage,
)

CHUNK = b"\x00\x00" * 160  # 160 samples


class _FakeServerWS:
    """A socket whose incoming frames come from a queue; ``send`` is an AsyncMock.

    ``replies`` maps a sent text frame's type to a callable that returns the frames
    the fake server answers with, so an awaited request sees its reply arrive through
    the reader loop.
    """

    def __init__(self, replies: dict | None = None) -> None:
        self._incoming: asyncio.Queue = asyncio.Queue()
        self.replies = replies or {}
        self.send = AsyncMock(side_effect=self._on_send)
        self.close = AsyncMock()

    async def _on_send(self, data) -> None:
        if isinstance(data, str):
            reply = self.replies.get(json.loads(data).get("type"))
            for frame in (reply() if reply else []):
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


def _server_mic_sample(sent: list) -> int:
    """The mic samples a server counts from these frames: audio sent while listening."""
    listening, count = True, 0
    for frame in sent:
        if isinstance(frame, bytes):
            if listening:
                count += len(frame) // 2
        elif frame.get("type") == "listen_stop":
            listening = False
        elif frame.get("type") == "listen_start":
            listening = True
    return count


def _stopped(mic_sample: int) -> dict:
    return ListenStoppedMessage(mic_sample=mic_sample).model_dump()


# --- listen_stop -----------------------------------------------------------------------

def test_listen_stop_sends_the_frame_and_returns_the_acks_mic_sample(caplog):
    client = _client()
    ws = _FakeServerWS({"listen_stop": lambda: [_stopped(client._mic_bytes // 2)]})
    client._ws = ws

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            assert session.listening is True
            await session.send_audio(CHUNK)
            count = await session.listen_stop()
            assert session.listening is False
            ws.end()
            return count
    with caplog.at_level(logging.WARNING, logger="voicemanager.client"):
        assert asyncio.run(run()) == 160
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    sent = _sent(client)
    assert sent == [CHUNK, {"type": "listen_stop"}, {"type": "listen_start"}]  # closing the stopped session
    ListenStopMessage.model_validate(sent[1])


def test_listen_stop_warns_when_the_servers_count_differs(caplog):
    ws = _FakeServerWS({"listen_stop": lambda: [_stopped(100)]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.send_audio(CHUNK)
            count = await session.listen_stop()
            ws.end()
            return count
    with caplog.at_level(logging.WARNING, logger="voicemanager.client"):
        assert asyncio.run(run()) == 100
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "100" in warnings[0] and "160" in warnings[0]


def test_listen_stop_returns_none_and_stays_stopped_when_no_ack_comes():
    client = _client(_FakeServerWS())

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            count = await session.listen_stop(timeout=0.01)
            listening = session.listening
            await session.send_audio(CHUNK)
            client._ws.end()
            return count, listening
    assert asyncio.run(run()) == (None, False)
    assert _sent(client) == [{"type": "listen_stop"}, {"type": "listen_start"}]  # closing the stopped session


def test_listen_stop_returns_none_when_the_connection_closes_while_waiting():
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            task = asyncio.create_task(session.listen_stop())
            await asyncio.sleep(0.01)
            ws.end()
            count = await asyncio.wait_for(task, 2.0)
            return count, session.listening
    assert asyncio.run(run()) == (None, False)


def test_a_late_ack_after_a_timeout_does_not_answer_the_next_listen_stop():
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.send_audio(CHUNK)
            assert await session.listen_stop(timeout=0.01) is None
            await session.listen_start()
            await session.send_audio(CHUNK)
            task = asyncio.create_task(session.listen_stop())
            await asyncio.sleep(0.01)
            ws.push(_stopped(160))  # the late answer to the first listen_stop
            await asyncio.sleep(0.01)
            assert not task.done()
            ws.push(_stopped(320))
            count = await asyncio.wait_for(task, 2.0)
            ws.end()
            return count
    assert asyncio.run(run()) == 320


def test_listen_stop_while_stopped_still_sends_the_frame():
    ws = _FakeServerWS({"listen_stop": lambda: [_stopped(0)]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            first = await session.listen_stop()
            second = await session.listen_stop()
            ws.end()
            return first, second
    assert asyncio.run(run()) == (0, 0)
    assert _sent(client) == [
        {"type": "listen_stop"}, {"type": "listen_stop"}, {"type": "listen_start"}]  # closing the stopped session


def test_listen_stop_refuses_on_a_closed_session():
    client = _client()

    async def run():
        session = client.duplex()
        await session.close()
        with pytest.raises(VoiceManagerError, match="closed"):
            await session.listen_stop()
    asyncio.run(run())
    assert _sent(client) == []
    assert client._duplex is None


# --- the stopped state -----------------------------------------------------------------

def test_send_audio_while_stopped_sends_and_counts_nothing():
    ws = _FakeServerWS({"listen_stop": lambda: [_stopped(160)]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.send_audio(CHUNK)
            await session.listen_stop()
            await session.send_audio(CHUNK)
            await session.send_audio(CHUNK)
            ws.end()
            return session.mic_sample
    assert asyncio.run(run()) == 160
    assert _sent(client) == [CHUNK, {"type": "listen_stop"}, {"type": "listen_start"}]  # closing the stopped session


# --- listen_start ----------------------------------------------------------------------

def test_listen_start_sends_the_frame_and_audio_flows_and_counts_again():
    ws = _FakeServerWS({"listen_stop": lambda: [_stopped(160)]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.send_audio(CHUNK)
            await session.listen_stop()
            await session.send_audio(CHUNK)
            await session.listen_start()
            assert session.listening is True
            await session.send_audio(CHUNK)
            ws.end()
            return session.mic_sample
    assert asyncio.run(run()) == 320
    sent = _sent(client)
    assert sent == [CHUNK, {"type": "listen_stop"}, {"type": "listen_start"}, CHUNK]
    ListenStartMessage.model_validate(sent[2])


def test_listen_start_while_listening_still_sends_the_frame():
    client = _client()

    async def run():
        async with client.duplex() as session:
            await session.listen_start()
            return session.listening
    assert asyncio.run(run()) is True
    assert _sent(client) == [{"type": "listen_start"}]


def test_listen_start_refuses_on_a_closed_session():
    client = _client()

    async def run():
        session = client.duplex()
        await session.close()
        with pytest.raises(VoiceManagerError, match="closed"):
            await session.listen_start()
    asyncio.run(run())
    assert _sent(client) == []


# --- a reconnect and a session that never calls them -------------------------------------

def test_a_reconnect_starts_the_new_connection_listening(monkeypatch):
    client = _client()
    real_sleep = asyncio.sleep

    async def _no_wait(_delay):
        await real_sleep(0)

    async def _fail():
        raise OSError("down")

    # _reconnect waits between attempts; skip the waits.
    monkeypatch.setattr("voicemanager.client.asyncio.sleep", _no_wait)
    monkeypatch.setattr(client, "_connect", _fail)

    async def run():
        session = client.duplex()
        assert await session.listen_stop(timeout=0.01) is None
        assert session.listening is False
        with pytest.raises(VoiceManagerError):
            await client._reconnect()
        return session.listening
    assert asyncio.run(run()) is True


def test_a_session_that_never_calls_them_is_unchanged():
    client = _client()

    async def run():
        async with client.duplex() as session:
            assert session.listening is True
            await session.send_audio(CHUNK)
            await session.send_audio(CHUNK)
            return session.mic_sample
    assert asyncio.run(run()) == 320
    sent = _sent(client)
    assert sent == [CHUNK, CHUNK]


# --- closing a stopped session ---------------------------------------------------------

def _close_stopped_session(client: VoiceManagerClient, ws: _FakeServerWS):
    """Send audio, stop listening, and close the session."""
    async def run():
        session = client.duplex()
        await session.send_audio(CHUNK)
        assert await session.listen_stop() == 160
        await session.close()
    return run()


def test_stream_stt_after_closing_a_stopped_session_sends_audio_the_server_counts():
    ws = _FakeServerWS({"listen_stop": lambda: [_stopped(160)]})
    client = _client(ws)

    async def chunks():
        yield CHUNK
        ws.end()  # the reader then ends the results, so stream_stt returns at once

    async def run():
        _start_reader(client)
        await _close_stopped_session(client, ws)
        await asyncio.wait_for(client.stream_stt(chunks()), 2.0)
    asyncio.run(run())
    sent = _sent(client)
    assert sent == [CHUNK, {"type": "listen_stop"}, {"type": "listen_start"}, CHUNK]
    assert client._mic_bytes // 2 == _server_mic_sample(sent) == 320


def test_stream_stt_live_after_closing_a_stopped_session_sends_audio_the_server_counts():
    ws = _FakeServerWS({"listen_stop": lambda: [_stopped(160)]})
    client = _client(ws)

    async def chunks():
        yield CHUNK
        ws.end()

    async def run():
        _start_reader(client)
        await _close_stopped_session(client, ws)

        async def drain():
            return [frame async for frame in client.stream_stt_live(chunks())]
        await asyncio.wait_for(drain(), 2.0)
    asyncio.run(run())
    sent = _sent(client)
    assert sent == [CHUNK, {"type": "listen_stop"}, {"type": "listen_start"}, CHUNK]
    assert client._mic_bytes // 2 == _server_mic_sample(sent) == 320


def test_a_new_session_after_closing_a_stopped_one_starts_listening():
    ws = _FakeServerWS({"listen_stop": lambda: [_stopped(160)]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        await _close_stopped_session(client, ws)
        async with client.duplex() as session:
            listening = session.listening
            await session.send_audio(CHUNK)
            mic_sample = session.mic_sample
        ws.end()
        return listening, mic_sample
    assert asyncio.run(run()) == (True, 320)
    sent = _sent(client)
    assert sent == [CHUNK, {"type": "listen_stop"}, {"type": "listen_start"}, CHUNK]
    assert _server_mic_sample(sent) == 320


def test_closing_a_listening_session_sends_no_extra_frame():
    ws = _FakeServerWS({"listen_stop": lambda: [_stopped(160)]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.send_audio(CHUNK)
        async with client.duplex() as session:
            await session.listen_stop()
            await session.listen_start()
        ws.end()
    asyncio.run(run())
    assert _sent(client) == [CHUNK, {"type": "listen_stop"}, {"type": "listen_start"}]


def test_closing_a_closed_session_again_sends_nothing_and_leaves_a_new_session_stopped():
    ws = _FakeServerWS({"listen_stop": lambda: [_stopped(0)]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        first = client.duplex()
        await first.close()
        second = client.duplex()
        await second.listen_stop()
        await first.close()
        listening = second.listening
        ws.end()
        return listening
    assert asyncio.run(run()) is False
    assert _sent(client) == [{"type": "listen_stop"}]


@pytest.mark.parametrize("gone", ["send_fails", "disconnected"])
def test_closing_a_stopped_session_on_a_gone_connection_raises_nothing(gone, monkeypatch):
    client = _client()
    connects = []

    async def _record_connect():
        connects.append(1)
        raise OSError("down")
    monkeypatch.setattr(client, "_connect", _record_connect)

    async def run():
        session = client.duplex()
        assert await session.listen_stop(timeout=0.01) is None
        if gone == "send_fails":
            client._ws.send.side_effect = websockets.exceptions.ConnectionClosed(None, None)
        else:
            await client._disconnect()
        await session.close()
        return session.listening, session.closed, client._duplex
    assert asyncio.run(run()) == (True, True, None)
    assert connects == []  # closing does not reconnect; the next send does
