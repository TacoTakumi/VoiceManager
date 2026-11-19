"""The client gives the frames of the user's turns on one stream, in the server's order.

A final that the server sent before a turn_end is a part of that turn, and a frame
that it sent after the turn_end belongs to the next one. The order of arrival is the
only thing that tells them apart on a connection where every frame has turn_id 0, so
the client must keep it. ``DuplexSession.turn_frames()`` does: after the call the
reader puts transcription, turn_end and turn_empty frames on one queue.

Model-free and server-free. The frames go through the client's real _reader_loop from
a queue-fed fake socket.
"""
import asyncio
import json
from unittest.mock import AsyncMock

from voicemanager.client import VoiceManagerClient


class _FakeServerWS:
    """A socket whose incoming frames come from a queue."""

    def __init__(self) -> None:
        self._incoming: asyncio.Queue = asyncio.Queue()
        self.send = AsyncMock()
        self.close = AsyncMock(side_effect=self._on_close)

    async def _on_close(self) -> None:
        self.end()

    def push(self, frame: dict) -> None:
        self._incoming.put_nowait(json.dumps(frame))

    def end(self) -> None:
        self._incoming.put_nowait(None)

    def __aiter__(self):
        return self._frames()

    async def _frames(self):
        while (frame := await self._incoming.get()) is not None:
            yield frame


def _client(ws: _FakeServerWS) -> VoiceManagerClient:
    client = VoiceManagerClient(token="vmp_test")
    client._ws = ws
    client.connected = True
    return client


def _start_reader(client: VoiceManagerClient) -> None:
    client._reader = asyncio.create_task(client._reader_loop())


def _final(segment: int, text: str) -> dict:
    # No turn_id on the wire here: the stream must add the key, as transcripts() does.
    return {"type": "transcription", "segment_index": segment, "text": text,
            "is_final": True}


def _partial(segment: int, text: str) -> dict:
    return {"type": "transcription", "segment_index": segment, "text": text,
            "is_final": False}


def _turn_end(text: str) -> dict:
    return {"type": "turn_end", "turn_id": 0, "text": text, "mic_sample": 1,
            "by": "model"}


TURN_EMPTY = {"type": "turn_empty", "mic_sample": 2, "by": "silence", "speech": True}


def _kinds(frames: list[dict]) -> list[tuple]:
    return [(f["type"], f.get("text")) for f in frames]


def _old_queues_empty(client: VoiceManagerClient) -> bool:
    return (client._stt_queue.empty() and client._turn_end_queue.empty()
            and client._turn_empty_queue.empty())


async def _take(frames, count: int) -> list[dict]:
    async with asyncio.timeout(3):
        return [await anext(frames) for _ in range(count)]


def test_the_frames_of_a_turn_come_out_of_one_stream_in_the_servers_order():
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            frames = session.turn_frames()
            sent = []
            for n in range(20):
                sent += [_final(3 * n, "end session"), _final(3 * n + 1, "no wait"),
                         _turn_end("end session no wait"),
                         _partial(3 * n + 2, "I didn't"), TURN_EMPTY]
            for frame in sent:  # all queued before the reader runs once
                ws.push(frame)
            got = await _take(frames, len(sent))
            empty = _old_queues_empty(client)
            ws.end()
            return sent, got, empty
    sent, got, old_queues_empty = asyncio.run(run())
    assert _kinds(got) == _kinds(sent)
    assert all("turn_id" in f for f in got if f["type"] == "transcription")
    assert old_queues_empty


def test_the_one_stream_ends_when_the_connection_closes():
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            frames = session.turn_frames()
            ws.push(_final(0, "hello"))
            first = await _take(frames, 1)
            ws.end()
            async with asyncio.timeout(3):
                rest = [f async for f in frames]
            return first, rest
    first, rest = asyncio.run(run())
    assert _kinds(first) == [("transcription", "hello")]
    assert rest == []


def test_a_session_that_does_not_ask_for_the_one_stream_keeps_its_three_streams():
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            ws.push(_final(0, "hello"))
            ws.push(_turn_end("hello"))
            ws.push(TURN_EMPTY)
            ws.end()
            async with asyncio.timeout(3):
                return ([f async for f in session.transcripts()],
                        [f async for f in session.turn_ends()],
                        [f async for f in session.turn_empties()])
    transcripts, ends, empties = asyncio.run(run())
    assert _kinds(transcripts) == [("transcription", "hello")]
    assert _kinds(ends) == [("turn_end", "hello")]
    assert empties == [TURN_EMPTY]


def test_the_choice_of_the_one_stream_holds_on_the_next_connection(monkeypatch):
    old = _FakeServerWS()
    new = _FakeServerWS()
    client = _client(old)

    async def connect() -> None:
        client._ws = new
        client.connected = True
        _start_reader(client)

    monkeypatch.setattr(client, "_connect", connect)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            frames = session.turn_frames()
            old.push(_final(0, "before"))
            before = await _take(frames, 1)
            await client._reconnect_now()
            async with asyncio.timeout(3):
                ended = [f async for f in frames]
            sent = [_final(0, "after"), _turn_end("after"), TURN_EMPTY]
            for frame in sent:
                new.push(frame)
            got = await _take(session.turn_frames(), len(sent))
            empty = _old_queues_empty(client)
            new.end()
            return before, ended, sent, got, empty
    before, ended, sent, got, old_queues_empty = asyncio.run(run())
    assert _kinds(before) == [("transcription", "before")]
    assert ended == []
    assert _kinds(got) == _kinds(sent)
    assert old_queues_empty


def test_the_choice_ends_with_its_session_so_one_sided_stt_reads_transcriptions_again():
    ws = _FakeServerWS()
    client = _client(ws)

    async def chunks():
        yield b"\x00" * 320
        ws.push(_final(0, "hello"))
        ws.end()

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            session.turn_frames()
        async with asyncio.timeout(3):
            return [f async for f in client.stream_stt_live(chunks())]
    assert _kinds(asyncio.run(run())) == [("transcription", "hello")]
    assert client._turn_frame_queue.qsize() == 1  # the end sentinel only


def test_a_later_session_that_does_not_ask_for_the_one_stream_gets_its_three_streams():
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            session.turn_frames()
        async with client.duplex() as session:
            ws.push(_final(0, "hello"))
            ws.push(_turn_end("hello"))
            ws.push(TURN_EMPTY)
            ws.end()
            async with asyncio.timeout(3):
                return ([f async for f in session.transcripts()],
                        [f async for f in session.turn_ends()],
                        [f async for f in session.turn_empties()])
    transcripts, ends, empties = asyncio.run(run())
    assert _kinds(transcripts) == [("transcription", "hello")]
    assert _kinds(ends) == [("turn_end", "hello")]
    assert empties == [TURN_EMPTY]
