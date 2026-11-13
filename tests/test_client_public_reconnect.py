"""The public reconnect() of VoiceManagerClient.

Model-free and server-free: the sockets are fakes, ``_connect`` is replaced by a
fake, and the backoff sleeps are patched so no test waits the real 1, 2 and 4 s.
Covers: after a reconnect whose attempts all fail there is no socket, and
reconnect() opens a new one, so a send works again on it and reconnected is True;
reconnect() while another reconnect runs waits for that one and opens no
connection of its own; reconnect() whose attempts all fail raises ConnectionError.
"""
import asyncio

import pytest
import websockets

from voicemanager.client import ConnectionError as ClientConnectionError, VoiceManagerClient

CHUNK = b"\x00\x00" * 160  # 160 samples


class _FakeWS:
    """A socket that records what is sent on it; its incoming frames come from a queue."""

    def __init__(self) -> None:
        self.sent: list = []
        self.closed = False
        self._incoming: asyncio.Queue = asyncio.Queue()

    async def send(self, data) -> None:
        if self.closed:
            raise websockets.exceptions.ConnectionClosed(None, None)
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True
        self._incoming.put_nowait(None)

    def __aiter__(self):
        return self._frames()

    async def _frames(self):
        while (frame := await self._incoming.get()) is not None:
            yield frame


class _GatedConnect:
    """Stands in for ``VoiceManagerClient._connect``.

    Each call opens a new fake socket once its gate is set (at once when
    ``open_at_once`` is True), then makes it the client's socket and starts a
    reader on it, as the real ``_connect`` does.
    """

    def __init__(self, client: VoiceManagerClient, open_at_once: bool = False) -> None:
        self.client = client
        self.open_at_once = open_at_once
        self.gates: list[asyncio.Event] = []
        self.sockets: list[_FakeWS] = []

    async def __call__(self) -> None:
        gate = asyncio.Event()
        self.gates.append(gate)
        if self.open_at_once:
            gate.set()
        await gate.wait()
        ws = _FakeWS()
        self.sockets.append(ws)
        self.client._ws = ws
        self.client._reader = asyncio.create_task(self.client._reader_loop())
        self.client.connected = True


def _client(monkeypatch) -> tuple[VoiceManagerClient, _FakeWS]:
    client = VoiceManagerClient(token="vmp_test")
    old = _FakeWS()
    client._ws = old
    client.connected = True
    return client, old


def _no_backoff(monkeypatch) -> list:
    """Patch asyncio.sleep so a backoff delay does not wait; return the delays asked."""
    real_sleep = asyncio.sleep
    delays: list = []

    async def no_wait(delay, *args, **kwargs):
        if delay:
            delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    return delays


def _refusing_connect(connects: list):
    async def refuse():
        connects.append(True)
        raise OSError("refused")
    return refuse


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


def test_reconnect_after_a_failed_reconnect_opens_a_new_connection(monkeypatch):
    client, old = _client(monkeypatch)
    connects: list = []
    monkeypatch.setattr(client, "_connect", _refusing_connect(connects))
    _no_backoff(monkeypatch)

    async def run():
        with pytest.raises(ClientConnectionError):
            await client._reconnect()
        after_failure = (client._ws, client.reconnected)
        connect = _GatedConnect(client, open_at_once=True)
        monkeypatch.setattr(client, "_connect", connect)
        await client.reconnect()
        async with client.duplex() as session:
            await session.send_audio(CHUNK)
            sample = session.mic_sample
        state = (client._ws, client.reconnected, client.connected, sample)
        await client._disconnect()
        return after_failure, connect, state

    after_failure, connect, (current, reconnected, connected, sample) = asyncio.run(run())
    assert len(connects) == 3
    assert after_failure == (None, False)
    assert old.closed
    assert len(connect.sockets) == 1
    assert current is connect.sockets[0]
    assert current.sent == [CHUNK]
    assert sample == len(CHUNK) // 2
    assert (reconnected, connected) == (True, True)


def test_reconnect_while_a_reconnect_runs_opens_no_connection_of_its_own(monkeypatch):
    client, old = _client(monkeypatch)
    connect = _GatedConnect(client)
    monkeypatch.setattr(client, "_connect", connect)
    _no_backoff(monkeypatch)

    async def run():
        running = asyncio.create_task(client._reconnect())
        for _ in range(100):
            if connect.gates:
                break
            await asyncio.sleep(0)
        assert len(connect.gates) == 1              # the first reconnect waits for its socket
        joined = asyncio.create_task(client.reconnect())
        await _settle()
        for gate in connect.gates:                  # a second connect, if one started
            gate.set()
        await _settle()
        await asyncio.gather(running, joined)
        await _settle()
        state = (client._ws, client.reconnected)
        await client._disconnect()
        return state

    current, reconnected = asyncio.run(run())
    assert len(connect.gates) == 1                  # one connect in all
    assert current is connect.sockets[0]
    assert old.closed
    assert reconnected is True


def test_reconnect_whose_attempts_all_fail_raises_connection_error(monkeypatch):
    client, old = _client(monkeypatch)
    connects: list = []
    monkeypatch.setattr(client, "_connect", _refusing_connect(connects))
    delays = _no_backoff(monkeypatch)

    async def run():
        with pytest.raises(ClientConnectionError):
            await client.reconnect()
        return client._ws, client.reconnected, client._reconnecting

    current, reconnected, running = asyncio.run(run())
    assert len(connects) == 3
    assert delays == [1, 2]
    assert (current, reconnected, running) == (None, False, None)
    assert old.closed
