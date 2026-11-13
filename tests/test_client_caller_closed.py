"""A ConnectionClosed from the caller's own iterator does not reconnect the client.

Model-free and server-free: the socket is a fake and ``_connect`` is replaced by a
fake that records each call. The iterator passed to stream_stt_live or
stream_tts_live raises ConnectionClosed after one item. That exception is not about
the client's connection, so it comes out unchanged, no new connection is made and
the client keeps its socket.
"""
import asyncio

import pytest
import websockets

from voicemanager.client import VoiceManagerClient

CHUNK = b"\x00\x00" * 160  # 160 samples


class _FakeWS:
    """A healthy socket that records what is sent on it."""

    def __init__(self) -> None:
        self.sent: list = []
        self.closed = False

    async def send(self, data) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True


def _client(monkeypatch) -> tuple[VoiceManagerClient, _FakeWS, list]:
    client = VoiceManagerClient(token="vmp_test")
    ws = _FakeWS()
    client._ws = ws
    client.connected = True
    connects: list = []

    async def connect():
        connects.append(True)
        client._ws = _FakeWS()
        client.connected = True

    async def no_wait(delay):
        pass

    monkeypatch.setattr(client, "_connect", connect)
    monkeypatch.setattr(asyncio, "sleep", no_wait)
    return client, ws, connects


def _closed() -> websockets.exceptions.ConnectionClosed:
    return websockets.exceptions.ConnectionClosed(None, None)


def test_stream_stt_live_lets_the_callers_connection_closed_through(monkeypatch):
    client, ws, connects = _client(monkeypatch)
    raised = _closed()

    async def chunks():
        yield CHUNK
        raise raised

    async def run():
        async for _ in client.stream_stt_live(chunks()):
            pass

    with pytest.raises(websockets.exceptions.ConnectionClosed) as info:
        asyncio.run(run())
    assert info.value is raised
    assert connects == []
    assert client._ws is ws
    assert not ws.closed
    assert ws.sent == [CHUNK]


def test_stream_tts_live_lets_the_callers_connection_closed_through(monkeypatch):
    client, ws, connects = _client(monkeypatch)
    raised = _closed()

    async def chunks():
        yield "Hello."
        raise raised

    async def run():
        async for _ in client.stream_tts_live(chunks(), request_id="rid-1"):
            pass

    with pytest.raises(websockets.exceptions.ConnectionClosed) as info:
        asyncio.run(run())
    assert info.value is raised
    assert connects == []
    assert client._ws is ws
    assert not ws.closed
    assert len(ws.sent) == 1
