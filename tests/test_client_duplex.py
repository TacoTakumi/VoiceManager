"""The client's full-duplex session: STT and TTS at the same time on one connection.

Model-free and server-free. The socket is an AsyncMock as in tests/test_client.py and
the frames the server would send are put on the client's reader-loop queues the way
_reader_loop does, so the session is tested on exactly what it reads. Covers: audio
goes out while transcripts come in; the TTS request frames (voice and chunking on the
first frame only, flush, done, cancel); the audio event stream with a cancelled
request's pcm dropped until its audio_end; a barge that marks only the request still
playing; the one-sided live methods refusing while a
session is open; and the errors() iterator.
"""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from voicemanager.client import DuplexSession, VoiceManagerClient, VoiceManagerError


def _client() -> VoiceManagerClient:
    client = VoiceManagerClient(token="vmp_test")
    client._ws = AsyncMock()
    client.connected = True
    return client


def _sent(client: VoiceManagerClient) -> list:
    """Every frame sent on the socket: dicts for text frames, bytes as they are."""
    out = []
    for call in client._ws.send.call_args_list:
        data = call.args[0]
        out.append(json.loads(data) if isinstance(data, str) else data)
    return out


def _transcript(seg: int, text: str, final: bool = False, endpoint: bool = False,
                turn_id: int | None = None) -> dict:
    frame = {"type": "transcription", "segment_index": seg, "text": text,
             "is_final": final, "endpoint": endpoint}
    if turn_id is not None:
        frame["turn_id"] = turn_id
    return frame


# --- audio out while transcripts come in --------------------------------------------

def test_send_audio_and_transcripts_run_at_the_same_time():
    client = _client()

    async def run():
        async with client.duplex() as session:
            got = []

            async def reader():
                async for frame in session.transcripts():
                    got.append(frame)
                    if frame["is_final"]:
                        return

            async def sender():
                for i in range(3):
                    await session.send_audio(bytes([i]) * 4)
                    await client._stt_queue.put(_transcript(0, "hello" + " x" * i,
                                                            final=(i == 2)))
                    await asyncio.sleep(0)

            await asyncio.gather(reader(), sender())
            return got
    got = asyncio.run(run())
    assert [f["text"] for f in got] == ["hello", "hello x", "hello x x"]
    assert all("turn_id" in f for f in got)
    binary = [f for f in _sent(client) if isinstance(f, bytes)]
    assert binary == [b"\x00" * 4, b"\x01" * 4, b"\x02" * 4]


def test_turn_start_returns_the_servers_turn_id():
    client = _client()

    async def run():
        async with client.duplex() as session:
            await client._turn_queue.put({"type": "turn_started", "turn_id": 7})
            return await session.turn_start()
    assert asyncio.run(run()) == 7
    assert {"type": "turn_start"} in _sent(client)


# --- the TTS request frames ----------------------------------------------------------

def test_tts_request_frames_in_order():
    client = _client()

    async def run():
        async with client.duplex() as session:
            req = session.tts_request(voice="designed_male_s202", chunking="clause")
            await req.send("Sure, I can")
            await req.send(" help with that.")
            await req.flush()
            await req.send(" One moment.")
            await req.done()
            with pytest.raises(VoiceManagerError):
                await req.send("too late")
            return req.request_id
    rid = asyncio.run(run())
    frames = [f for f in _sent(client) if isinstance(f, dict)]
    assert [f["type"] for f in frames] == ["tts_stream"] * 5
    assert all(f["request_id"] == rid for f in frames)
    assert frames[0]["voice"] == "designed_male_s202" and frames[0]["chunking"] == "clause"
    assert all("voice" not in f and "chunking" not in f for f in frames[1:])
    assert [f["text"] for f in frames] == ["Sure, I can", " help with that.", "", " One moment.", ""]
    assert [f.get("flush", False) for f in frames] == [False, False, True, False, False]
    assert [f["done"] for f in frames] == [False, False, False, False, True]
    assert client.last_tts_request_id == rid


def test_tts_request_cancel_sends_tts_cancel_and_marks_the_id():
    client = _client()

    async def run():
        async with client.duplex() as session:
            req = session.tts_request()
            await req.send("hello")
            await req.cancel()
            assert req.cancelled
            return req.request_id
    rid = asyncio.run(run())
    assert _sent(client)[-1] == {"type": "tts_cancel", "request_id": rid}


def test_tts_request_refuses_an_unknown_chunking_before_any_frame():
    client = _client()

    async def run():
        async with client.duplex() as session:
            with pytest.raises(ValueError):
                session.tts_request(chunking="paragraph")
    asyncio.run(run())
    assert _sent(client) == []


# --- the audio event stream ----------------------------------------------------------

async def _until(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting on the condition")
        await asyncio.sleep(0.001)


def test_audio_events_drop_a_cancelled_requests_pcm_until_its_audio_end():
    client = _client()
    events = []

    async def run():
        async with client.duplex() as session:
            first = session.tts_request()
            second = session.tts_request()
            q = client._tts_queue

            async def collect():
                async for event in session.audio():
                    events.append(event)
            task = asyncio.create_task(collect())
            await q.put({"type": "audio_start", "request_id": first.request_id,
                         "sample_rate": 24000})
            await q.put(b"\x01\x01")
            await q.put({"type": "audio_end", "request_id": first.request_id,
                         "reason": "complete"})
            await q.put({"type": "audio_start", "request_id": second.request_id,
                         "sample_rate": 24000})
            await q.put(b"\x02\x02")
            await _until(lambda: len(events) == 5)  # the second request's pcm was played
            await second.cancel()
            await q.put(b"\x02\x03")  # still in flight after the cancel: dropped
            await q.put({"type": "audio_end", "request_id": second.request_id,
                         "reason": "cancelled"})
            await q.put(None)  # the reader loop's end-of-connection sentinel
            await asyncio.wait_for(task, 2.0)
            return first.request_id, second.request_id
    first, second = asyncio.run(run())
    assert events == [
        ("audio_start", {"type": "audio_start", "request_id": first, "sample_rate": 24000}),
        ("pcm", b"\x01\x01", first),
        ("audio_end", {"type": "audio_end", "request_id": first, "reason": "complete"}),
        ("audio_start", {"type": "audio_start", "request_id": second, "sample_rate": 24000}),
        ("pcm", b"\x02\x02", second),
        ("audio_end", {"type": "audio_end", "request_id": second, "reason": "cancelled"}),
    ]
    assert client.last_tts_sample_rate == 24000
    assert second not in client._cancelled_tts  # cleared by its audio_end


async def _collect_audio(client: VoiceManagerClient, session, frames: list) -> list:
    """Feed the frames to the audio stream, end it, and return every event it yielded."""
    events = []

    async def collect():
        async for event in session.audio():
            events.append(event)
    task = asyncio.create_task(collect())
    for frame in frames:
        await client._tts_queue.put(frame)
    await client._tts_queue.put(None)  # the reader loop's end-of-connection sentinel
    await asyncio.wait_for(task, 2.0)
    return events


def test_a_barge_after_the_requests_audio_end_leaves_the_cancelled_set_unchanged():
    # The audio_end was already read, so no pcm of that request can follow and no
    # second audio_end will come to clear the id: marking it would leak it forever.
    client = _client()

    async def run():
        async with client.duplex() as session:
            rid = session.tts_request().request_id
            events = await _collect_audio(client, session, [
                {"type": "audio_start", "request_id": rid, "sample_rate": 24000},
                b"\x01\x01",
                {"type": "audio_end", "request_id": rid, "reason": "complete"},
                {"type": "barge", "request_id": rid},
            ])
            return rid, events
    rid, events = asyncio.run(run())
    assert [e[0] for e in events] == ["audio_start", "pcm", "audio_end", "barge"]
    assert events[-1] == ("barge", {"type": "barge", "request_id": rid})
    assert client._cancelled_tts == set()


def test_a_barge_before_the_requests_audio_end_drops_its_pcm_until_the_audio_end():
    client = _client()

    async def run():
        async with client.duplex() as session:
            rid = session.tts_request().request_id
            events = await _collect_audio(client, session, [
                {"type": "audio_start", "request_id": rid, "sample_rate": 24000},
                b"\x01\x01",
                {"type": "barge", "request_id": rid},
                b"\x01\x02",  # still in flight after the barge: dropped
                b"\x01\x03",
                {"type": "audio_end", "request_id": rid, "reason": "interrupted"},
            ])
            return rid, events
    rid, events = asyncio.run(run())
    assert events == [
        ("audio_start", {"type": "audio_start", "request_id": rid, "sample_rate": 24000}),
        ("pcm", b"\x01\x01", rid),
        ("barge", {"type": "barge", "request_id": rid}),
        ("audio_end", {"type": "audio_end", "request_id": rid, "reason": "interrupted"}),
    ]
    assert client._cancelled_tts == set()  # cleared by its audio_end


def test_errors_yields_error_frames():
    client = _client()

    async def run():
        async with client.duplex() as session:
            await client._error_queue.put({"type": "error", "error_type": "invalid_request",
                                           "message": "bad chunking"})
            async for frame in session.errors():
                return frame
    assert asyncio.run(run())["error_type"] == "invalid_request"


# --- one session at a time, and the one-sided methods refuse while it is open --------

def test_one_sided_methods_refuse_while_a_session_is_open_and_work_after_close():
    client = _client()

    async def chunks():
        yield b"\x00\x00"

    async def words():
        yield "hi"

    async def run():
        session = client.duplex()
        with pytest.raises(VoiceManagerError, match="duplex"):
            client.duplex()
        for coro in (client.stream_stt(chunks()), client.stream_tts(words()),
                     client.synthesize_tts("hi")):
            with pytest.raises(VoiceManagerError, match="duplex"):
                await coro
        for gen in (client.stream_stt_live(chunks()), client.stream_tts_live(words())):
            with pytest.raises(VoiceManagerError, match="duplex"):
                async for _ in gen:
                    pass
        await session.close()
        # After close the one-sided path runs: the sentinel ends the live iterator.
        await client._stt_queue.put(None)
        got = [f async for f in client.stream_stt_live(chunks())]
        assert got == []
        assert isinstance(client.duplex(), DuplexSession)
    asyncio.run(run())


class _FeedWS:
    """A socket whose incoming frames the test puts on a queue; None ends the stream."""

    def __init__(self) -> None:
        self.incoming: asyncio.Queue = asyncio.Queue()

    async def send(self, data) -> None:
        pass

    async def close(self) -> None:
        self.incoming.put_nowait(None)

    def __aiter__(self):
        return self._frames()

    async def _frames(self):
        while (frame := await self.incoming.get()) is not None:
            yield frame


def test_error_queue_keeps_the_64_newest_frames_when_nobody_reads_it():
    """One-sided use never reads the error queue, so the reader keeps only the newest
    frames there and the queue cannot grow for the life of the client."""
    client = VoiceManagerClient(token="vmp_test")
    ws = _FeedWS()

    async def run():
        client._ws = ws
        client.connected = True
        client._reader = asyncio.create_task(client._reader_loop())
        for n in range(100):
            ws.incoming.put_nowait(json.dumps(
                {"type": "error", "error_type": "invalid_request", "message": f"m{n}"}))
        await _until(lambda: ws.incoming.empty())
        for _ in range(5):
            await asyncio.sleep(0)
        kept = []
        while not client._error_queue.empty():
            kept.append(client._error_queue.get_nowait()["message"])
        await ws.close()
        await asyncio.wait_for(client._reader, timeout=1.0)
        return kept

    assert asyncio.run(run()) == [f"m{n}" for n in range(36, 100)]
