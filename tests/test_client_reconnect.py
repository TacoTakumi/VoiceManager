"""One reconnect serves every task of a VoiceManagerClient.

Model-free and server-free: the sockets are fakes and ``_connect`` is replaced by a
fake that waits at a gate, so a test decides when each new connection comes up.
Covers: a send from another task while a reconnect runs (there is no socket) raises
the documented error type of that call, sends nothing and starts no reconnect, and
leaves the state of the new connection as it was; two tasks whose sends both find
the same closed socket cause one new connection, whose mic counts count only the
frames sent on it, and no other socket or reader task stays open. A cancel with no
socket raises TTSError and leaves the cancelled set of the new connection empty; a
cancel with a socket marks the id before the cancel frame is sent.
"""
import asyncio
import json

import pytest
import websockets

from voicemanager.client import (
    ConnectionError as ClientConnectionError, STTError, TTSError, TurnEngineError,
    VoiceManagerClient, VoiceManagerError, WakeWordError)

CHUNK = b"\x00\x00" * 160  # 160 samples


class _FakeWS:
    """A socket that records what is sent on it; its incoming frames come from a queue.

    Set ``fail_after`` to an Event to make every send wait for it and then raise
    ConnectionClosed, as a send does that waits in the drain when the connection
    is lost.
    """

    def __init__(self) -> None:
        self.sent: list = []
        self.closed = False
        self.fail_after: asyncio.Event | None = None
        self._incoming: asyncio.Queue = asyncio.Queue()

    async def send(self, data) -> None:
        if self.closed:
            raise websockets.exceptions.ConnectionClosed(None, None)
        if self.fail_after is not None:
            await self.fail_after.wait()
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

    Each call opens a new fake socket once its gate is set, then makes it the
    client's socket and starts a reader on it, as the real ``_connect`` does.
    """

    def __init__(self, client: VoiceManagerClient) -> None:
        self.client = client
        self.gates: list[asyncio.Event] = []
        self.sockets: list[_FakeWS] = []
        self.readers: list[asyncio.Task] = []

    async def __call__(self) -> None:
        gate = asyncio.Event()
        self.gates.append(gate)
        await gate.wait()
        ws = _FakeWS()
        self.sockets.append(ws)
        self.client._ws = ws
        self.client._reader = asyncio.create_task(self.client._reader_loop())
        self.readers.append(self.client._reader)
        self.client.connected = True


def _client(monkeypatch) -> tuple[VoiceManagerClient, _FakeWS, _GatedConnect]:
    client = VoiceManagerClient(token="vmp_test")
    old = _FakeWS()
    client._ws = old
    client.connected = True
    connect = _GatedConnect(client)
    monkeypatch.setattr(client, "_connect", connect)
    return client, old, connect


async def _settle() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


async def _wait_for_gate(connect: _GatedConnect, count: int) -> None:
    for _ in range(100):
        if len(connect.gates) >= count:
            return
        await asyncio.sleep(0)
    raise AssertionError(f"no connect number {count}")


# Each call that sends a frame, with the error type its docstring documents. Every
# one of them runs while another task reconnects, so there is no socket.
_CALLS = {
    "send_audio": (lambda s: s.send_audio(CHUNK), STTError),
    "tts_request_send": (lambda s: s.tts_request().send("hello"), TTSError),
    "playback_start": (lambda s: s.playback_start("rid-1", mic_sample=0), VoiceManagerError),
    "listen_stop": (lambda s: s.listen_stop(timeout=0.1), VoiceManagerError),
    "listen_start": (lambda s: s.listen_start(), VoiceManagerError),
    "enable_wake_word": (lambda s: s.enable_wake_word(timeout=0.1), WakeWordError),
    "enable_turn_engine": (lambda s: s.enable_turn_engine(timeout=0.1), TurnEngineError),
    "turn_start": (lambda s: s.turn_start(timeout=0.1), STTError),
}


@pytest.mark.parametrize("name", sorted(_CALLS))
def test_a_send_while_another_task_reconnects_raises_the_documented_error(
        monkeypatch, name):
    call, error = _CALLS[name]
    client, old, connect = _client(monkeypatch)
    old.closed = True

    async def run():
        async with client.duplex() as session:
            first = asyncio.create_task(session.send_audio(CHUNK))
            await _wait_for_gate(connect, 1)       # the reconnect waits for its socket
            with pytest.raises(error) as info:
                await call(session)
            assert not isinstance(info.value, AttributeError)
            await _settle()
            connect.gates[0].set()
            with pytest.raises(STTError):
                await first
            await _settle()
            for gate in connect.gates:              # a second reconnect, if one started
                gate.set()
            await _settle()
            state = (session.mic_sample, client._mic_frames, session.listening,
                     client._turn_engine_sent, client._listen_stops_sent,
                     client._wake_sent)
            new = client._ws
        for reader in connect.readers:
            reader.cancel()
        return state, new

    state, new = asyncio.run(run())
    assert len(connect.gates) == 1                  # one connect, and no second one
    assert new is connect.sockets[0]
    assert new.sent == []                           # the refused frame went nowhere
    # The new connection starts clean: nothing the refused call did stays on it.
    assert state == (0, 0, True, None, 0, None)


def test_a_send_after_a_failed_reconnect_raises_and_does_not_reconnect(monkeypatch):
    client, old, _ = _client(monkeypatch)
    connects = []

    async def refuse():
        connects.append(True)
        raise OSError("refused")

    async def no_wait(delay):
        pass

    monkeypatch.setattr(client, "_connect", refuse)
    old.closed = True

    async def run():
        monkeypatch.setattr(asyncio, "sleep", no_wait)
        async with client.duplex() as session:
            with pytest.raises(ClientConnectionError):
                await session.send_audio(CHUNK)
            tried = len(connects)
            with pytest.raises(STTError):
                await session.send_audio(CHUNK)
            with pytest.raises(TTSError):
                await client.flush_tts("rid-1")
            return tried, session.mic_sample, client._mic_frames

    tried, sample, frames = asyncio.run(run())
    assert tried == 3
    assert len(connects) == 3                       # the later sends started none
    assert (sample, frames) == (0, 0)


def test_two_sends_that_find_the_same_closed_socket_cause_one_new_connection(monkeypatch):
    """Both sends wait in the drain when the connection is lost. Four frames go out
    on the new connection between the first connect and any second one."""
    client, old, connect = _client(monkeypatch)

    async def run():
        old.fail_after = asyncio.Event()
        old_reader = asyncio.create_task(client._reader_loop())
        client._reader = old_reader
        async with client.duplex() as session:
            a = asyncio.create_task(session.send_audio(CHUNK))
            b = asyncio.create_task(session.send_audio(CHUNK))
            await _settle()
            old.fail_after.set()                    # the connection is lost
            await _wait_for_gate(connect, 1)
            await _settle()
            connect.gates[0].set()
            await _settle()
            for _ in range(4):
                await session.send_audio(CHUNK)
            for gate in connect.gates:              # a second connect, if one started
                gate.set()
            await _settle()
            results = await asyncio.gather(a, b, return_exceptions=True)
            state = (session.mic_sample, client._mic_frames)
            current = client._ws
            readers = [r for r in connect.readers if not r.done()]
            current_reader = client._reader
        for reader in connect.readers:
            reader.cancel()
        await _settle()
        return results, state, current, readers, current_reader, old_reader.done()

    results, (sample, frames), current, readers, current_reader, old_done = asyncio.run(run())
    assert [type(r) for r in results] == [STTError, STTError]
    assert len(connect.gates) == 1                  # one new connection
    assert current is connect.sockets[0]
    received = [f for f in current.sent if isinstance(f, bytes)]
    assert frames == len(received) == 4
    assert sample == sum(len(f) for f in received) // 2 == 640
    # No other new socket is open, and only the current one has a reader running.
    assert [ws for ws in connect.sockets if not ws.closed] == [current]
    assert readers == [current_reader]
    assert old.closed and old_done


def test_a_send_that_finds_its_socket_already_replaced_does_not_reconnect(monkeypatch):
    """A reconnect that finished before the second send gave up: that send raises
    the documented error and leaves the new connection alone."""
    client, old, connect = _client(monkeypatch)

    async def run():
        old.fail_after = asyncio.Event()
        async with client.duplex() as session:
            a = asyncio.create_task(session.send_audio(CHUNK))
            await _settle()
            # The second send waits in the drain of the old socket longer: it gives
            # up only after the first one's reconnect is done.
            late = asyncio.Event()

            async def late_send(data):
                await late.wait()
                raise websockets.exceptions.ConnectionClosed(None, None)

            old.send = late_send
            b = asyncio.create_task(client._send_or_raise(
                json.dumps({"type": "listen_start"}), VoiceManagerError))
            await _settle()
            old.fail_after.set()
            await _wait_for_gate(connect, 1)
            connect.gates[0].set()
            with pytest.raises(STTError):
                await a
            await session.send_audio(CHUNK)
            late.set()
            await _settle()
            for gate in connect.gates:
                gate.set()
            with pytest.raises(VoiceManagerError):
                await b
            state = (session.mic_sample, client._mic_frames)
        for reader in connect.readers:
            reader.cancel()
        return state

    assert asyncio.run(run()) == (160, 1)
    assert len(connect.gates) == 1


# A cancel with no socket raises TTSError and marks nothing: a mark made then would
# land in the cancelled set of the new connection and stay there for good.
_CANCELS = {
    "cancel_tts": lambda c, s: c.cancel_tts("rid-1"),
    "tts_request_cancel": lambda c, s: s.tts_request(request_id="rid-1").cancel(),
}


@pytest.mark.parametrize("name", sorted(_CANCELS))
def test_a_cancel_while_another_task_reconnects_marks_nothing(monkeypatch, name):
    call = _CANCELS[name]
    client, old, connect = _client(monkeypatch)
    old.closed = True

    async def run():
        async with client.duplex() as session:
            first = asyncio.create_task(session.send_audio(CHUNK))
            await _wait_for_gate(connect, 1)       # the reconnect waits for its socket
            with pytest.raises(TTSError):
                await call(client, session)
            await _settle()
            connect.gates[0].set()
            with pytest.raises(STTError):
                await first
            await _settle()
            for gate in connect.gates:              # a second reconnect, if one started
                gate.set()
            await _settle()
            cancelled = set(client._cancelled_tts)
            new = client._ws
        for reader in connect.readers:
            reader.cancel()
        return cancelled, new

    cancelled, new = asyncio.run(run())
    assert len(connect.gates) == 1                  # one connect, and no second one
    assert new is connect.sockets[0]
    assert new.sent == []                           # the refused cancel went nowhere
    assert cancelled == set()                       # the new connection starts clean


@pytest.mark.parametrize("name", sorted(_CANCELS))
def test_a_cancel_with_a_connection_marks_the_id_before_the_frame_goes_out(
        monkeypatch, name):
    call = _CANCELS[name]
    client, ws, _ = _client(monkeypatch)
    marked_at_send = []

    async def send(data):
        marked_at_send.append("rid-1" in client._cancelled_tts)
        ws.sent.append(data)

    ws.send = send

    async def run():
        async with client.duplex() as session:
            await call(client, session)
        return set(client._cancelled_tts)

    cancelled = asyncio.run(run())
    assert marked_at_send == [True]
    assert [json.loads(f) for f in ws.sent] == [
        {"type": "tts_cancel", "request_id": "rid-1"}]
    assert cancelled == {"rid-1"}
