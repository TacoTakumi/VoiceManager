"""The public reconnect() of VoiceManagerClient.

Model-free and server-free: the sockets are fakes, ``_connect`` is replaced by a
fake, and the backoff sleeps are patched so no test waits the real 1, 2 and 4 s.
Covers: after a reconnect whose attempts all fail there is no socket, and
reconnect() opens a new one, so a send works again on it and reconnected is True;
reconnect() while another reconnect runs waits for that one and opens no
connection of its own; reconnect() whose attempts all fail raises ConnectionError; a cancel of reconnect()
while it waits for the old reader reaches the caller after the old socket is closed,
and a cancel that the task swallowed earlier does not make a later reconnect() raise;
a cancel or an error during the old socket's close resets the client and reaches the
caller.
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


def test_cancelled_reconnect_closes_the_socket_and_raises_the_cancel(monkeypatch):
    """A cancel of reconnect() while it waits for the old reader to end reaches the
    caller. The old socket is closed first, so no socket stays open with no reader,
    and a later reconnect() opens a new connection."""
    client, old = _client(monkeypatch)
    _no_backoff(monkeypatch)
    connect = _GatedConnect(client, open_at_once=True)
    monkeypatch.setattr(client, "_connect", connect)
    ending = asyncio.Event()

    async def slow_reader():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            ending.set()
            await asyncio.Event().wait()    # slow to end: only a second cancel ends it
            raise

    async def run():
        client._reader = asyncio.create_task(slow_reader())
        await _settle()
        task = asyncio.create_task(client.reconnect())
        await ending.wait()                 # the disconnect cancelled the reader and waits
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        after_cancel = (client._ws, client.connected, client._reconnecting,
                        len(connect.sockets))
        await client.reconnect()
        state = (client._ws, client.connected, client.reconnected)
        await client._disconnect()
        return after_cancel, state

    after_cancel, (current, connected, reconnected) = asyncio.run(run())
    assert old.closed
    assert after_cancel == (None, False, None, 0)
    assert len(connect.sockets) == 1
    assert current is connect.sockets[0]
    assert (connected, reconnected) == (True, True)


def test_reconnect_in_a_task_with_an_old_swallowed_cancel_succeeds(monkeypatch):
    """A task that once swallowed a cancel keeps a pending cancel count of 1. A
    reconnect() in it that nobody cancels must still open the new connection."""
    client, old = _client(monkeypatch)
    _no_backoff(monkeypatch)
    connect = _GatedConnect(client, open_at_once=True)
    monkeypatch.setattr(client, "_connect", connect)

    async def run():
        client._reader = asyncio.create_task(asyncio.Event().wait())
        await _settle()
        me = asyncio.current_task()
        me.cancel()
        try:
            await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass                            # swallowed, with no uncancel()
        stale = me.cancelling()
        await client.reconnect()
        state = (client._ws, client.connected, client.reconnected)
        await client._disconnect()
        return stale, state

    stale, (current, connected, reconnected) = asyncio.run(run())
    assert stale == 1
    assert old.closed
    assert current is connect.sockets[0]
    assert (connected, reconnected) == (True, True)


class _SlowCloseWS(_FakeWS):
    """A socket whose close waits until the task that runs it is cancelled."""

    def __init__(self) -> None:
        super().__init__()
        self.closing = asyncio.Event()

    async def close(self) -> None:
        self.closing.set()
        await asyncio.Event().wait()


class _FailingCloseWS(_FakeWS):
    async def close(self) -> None:
        raise RuntimeError("close failed")


def _queues(client) -> tuple:
    return (client._stt_queue, client._tts_queue, client._error_queue, client._turn_queue,
            client._turn_end_queue, client._turn_empty_queue, client._barge_state_queue)


def test_cancel_during_the_socket_close_resets_the_client(monkeypatch):
    """A cancel of reconnect() while the old socket's close waits reaches the caller,
    and the client keeps no socket, so no later call sends on a socket nothing reads."""
    client = VoiceManagerClient(token="vmp_test")
    old = _SlowCloseWS()
    client._ws = old
    client.connected = True
    _no_backoff(monkeypatch)
    connect = _GatedConnect(client, open_at_once=True)
    monkeypatch.setattr(client, "_connect", connect)

    async def run():
        client._reader = asyncio.create_task(asyncio.Event().wait())
        await _settle()
        task = asyncio.create_task(client.reconnect())
        await old.closing.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        after_cancel = (client._ws, client._reader, client.connected,
                        [q.get_nowait() for q in _queues(client)])
        await client.reconnect()
        state = (client._ws, client.connected)
        await client._disconnect()
        return after_cancel, state

    after_cancel, (current, connected) = asyncio.run(run())
    assert after_cancel == (None, None, False, [None] * 7)
    assert current is connect.sockets[0]
    assert connected is True


def test_failing_socket_close_resets_the_client_and_raises(monkeypatch):
    client = VoiceManagerClient(token="vmp_test")
    client._ws = _FailingCloseWS()
    client.connected = True

    async def run():
        client._reader = asyncio.create_task(asyncio.Event().wait())
        await _settle()
        with pytest.raises(RuntimeError, match="close failed"):
            await client._disconnect()
        return (client._ws, client._reader, client.connected,
                [q.get_nowait() for q in _queues(client)])

    assert asyncio.run(run()) == (None, None, False, [None] * 7)
