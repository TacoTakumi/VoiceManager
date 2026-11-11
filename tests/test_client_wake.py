"""The wake word helpers on a DuplexSession: enable_wake_word, wakes, dormant.

Model-free and server-free. The frames the server would send go through the client's
real _reader_loop from a queue-fed fake socket, as in tests/test_client_listen.py.
Covers: enable_wake_word sends the frame, with the phrase list or without, and returns
the phrases of wake_word_ready; a refusal or no answer raises WakeWordError and leaves
the session as it was; while dormant send_audio sends its frames and does not count
them; a wake frame makes the session listen and comes out of wakes(); the session's
mic count follows the server's over a wake, also for audio that was in flight when the
server woke, for a listen_start by hand that crossed the wake, and for a listen_stop
sent while dormant that crossed it; listen_stop makes the session dormant again on a
new count; a reconnect starts with no opt-in; and a session that never opts in is
unchanged.
"""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest
import websockets

from voicemanager.client import VoiceManagerClient, VoiceManagerError, WakeWordError
from voicemanager.protocol import (
    ErrorMessage,
    ListenStoppedMessage,
    WakeMessage,
    WakeWordMessage,
    WakeWordReadyMessage,
)

CHUNK = b"\x00\x00" * 160  # 160 samples


class _FakeServerWS:
    """A socket whose incoming frames come from a queue; ``send`` is an AsyncMock.

    ``replies`` maps a sent text frame's type to a callable that takes the frame and
    returns the frames the fake server answers with, so an awaited request sees its
    reply arrive through the reader loop.
    """

    def __init__(self, replies: dict | None = None) -> None:
        self._incoming: asyncio.Queue = asyncio.Queue()
        self.replies = replies or {}
        self.send = AsyncMock(side_effect=self._on_send)
        self.close = AsyncMock()
        # Set to an Event to make the send of each audio frame wait for it, as a
        # send waits when the transport's write buffer is full. The frame is on the
        # wire by then: only the return of the call waits.
        self.pause: asyncio.Event | None = None

    async def _on_send(self, data) -> None:
        if isinstance(data, bytes) and self.pause is not None:
            await self.pause.wait()
        if isinstance(data, str):
            sent = json.loads(data)
            reply = self.replies.get(sent.get("type"))
            for frame in (reply(sent) if reply else []):
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


def _ready(phrases=("hey computer",)) -> dict:
    return WakeWordReadyMessage(phrases=list(phrases)).model_dump()


def _wake(mic_sample: int, dormant_sample: int, phrase: str = "hey computer") -> dict:
    return WakeMessage(
        phrase=phrase, mic_sample=mic_sample, dormant_sample=dormant_sample).model_dump()


def _refusal(reason: str = "the phrase list is empty") -> dict:
    return ErrorMessage(
        error_type="invalid_request",
        message=f"wake_word refused: {reason}; the frame was dropped.").model_dump()


def _accepting() -> dict:
    """Replies of a server that accepts the opt-in with the list sent, or its default."""
    return {"wake_word": lambda sent: [_ready(sent.get("phrases") or ["hey computer"])]}


def _client(ws) -> VoiceManagerClient:
    client = VoiceManagerClient(token="vmp_test")
    client._ws = ws
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


async def _settle() -> None:
    """Let the reader loop handle the frames pushed so far."""
    for _ in range(5):
        await asyncio.sleep(0)


# --- the opt-in --------------------------------------------------------------------

def test_enable_wake_word_sends_the_bare_frame_and_returns_the_servers_phrases():
    ws = _FakeServerWS(_accepting())
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            assert session.wake_phrases is None and session.dormant is False
            phrases = await session.enable_wake_word()
            state = (session.wake_phrases, session.dormant, session.listening)
            ws.end()
            return phrases, state
    phrases, state = asyncio.run(run())
    assert phrases == ["hey computer"]
    assert state == (["hey computer"], True, False)
    sent = _sent(client)
    assert sent[0] == {"type": "wake_word"}
    assert WakeWordMessage.model_validate(sent[0]).phrases is None


def test_enable_wake_word_sends_the_phrase_list():
    ws = _FakeServerWS({"wake_word": lambda sent: [_ready(["hey hermes"])]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            phrases = await session.enable_wake_word(["Hey Hermes"])
            ws.end()
            return phrases
    # The list the server answers with is returned, not the list that was sent.
    assert asyncio.run(run()) == ["hey hermes"]
    assert _sent(client)[0] == {"type": "wake_word", "phrases": ["Hey Hermes"]}


def test_a_refused_opt_in_raises_with_the_servers_error_and_changes_nothing():
    ws = _FakeServerWS({"wake_word": lambda sent: [_refusal()]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.send_audio(CHUNK)
            with pytest.raises(WakeWordError) as err:
                await session.enable_wake_word([])
            await session.send_audio(CHUNK)
            state = (session.wake_phrases, session.dormant, session.listening,
                     session.mic_sample)
            # The refusal went to the call, not to errors().
            ws.end()
            leftover = [e async for e in session.errors()]
            return err.value, state, leftover
    error, state, leftover = asyncio.run(run())
    assert error.error_type == "invalid_request"
    assert "the phrase list is empty" in str(error)
    assert state == (None, False, True, 320)
    assert leftover == []


def test_another_invalid_request_does_not_answer_the_opt_in():
    """Only an error frame that names the wake_word refusal answers the call."""
    other = ErrorMessage(error_type="invalid_request",
                         message="tts_stream chunking must be one of: ...").model_dump()
    ws = _FakeServerWS({"wake_word": lambda sent: [other, _ready()]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            phrases = await session.enable_wake_word()
            ws.end()
            return phrases, [e async for e in session.errors()]
    phrases, leftover = asyncio.run(run())
    assert phrases == ["hey computer"]
    assert leftover == [other]


def test_no_answer_raises_and_leaves_the_session_listening():
    ws = _FakeServerWS()  # an older server drops the frame unanswered
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            with pytest.raises(WakeWordError) as err:
                await session.enable_wake_word(timeout=0.05)
            await session.send_audio(CHUNK)
            state = (session.wake_phrases, session.dormant, session.mic_sample)
            ws.end()
            return err.value, state
    error, state = asyncio.run(run())
    assert error.error_type is None
    assert state == (None, False, 160)
    assert _sent(client) == [{"type": "wake_word"}, CHUNK]


def test_a_closed_connection_while_waiting_raises():
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            call = asyncio.create_task(session.enable_wake_word())
            await _settle()
            ws.end()
            with pytest.raises(WakeWordError, match="Connection lost"):
                await call
    asyncio.run(run())


def test_enable_wake_word_refuses_on_a_closed_session():
    ws = _FakeServerWS(_accepting())
    client = _client(ws)

    async def run():
        session = client.duplex()
        await session.close()
        with pytest.raises(VoiceManagerError, match="closed"):
            await session.enable_wake_word()
    asyncio.run(run())
    assert _sent(client) == []


def test_a_ready_after_the_timeout_still_makes_the_session_dormant():
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            with pytest.raises(WakeWordError):
                await session.enable_wake_word(timeout=0.05)
            await session.send_audio(CHUNK)      # counted as mic audio for now
            ws.push(_ready())
            await _settle()
            state = (session.wake_phrases, session.dormant, session.mic_sample)
            # The server took that frame as dormant audio: its wake counts it.
            ws.push(_wake(0, 160))
            await _settle()
            ws.end()
            return state, session.mic_sample, session.listening
    state, mic_sample, listening = asyncio.run(run())
    assert state == (["hey computer"], True, 0)
    assert (mic_sample, listening) == (0, True)


# --- dormant audio -----------------------------------------------------------------

def test_send_audio_while_dormant_sends_and_does_not_count():
    ws = _FakeServerWS(_accepting())
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.send_audio(CHUNK)
            await session.enable_wake_word()
            await session.send_audio(CHUNK)
            await session.send_audio(CHUNK)
            ws.end()
            return session.mic_sample
    assert asyncio.run(run()) == 160
    assert _sent(client)[:4] == [CHUNK, {"type": "wake_word"}, CHUNK, CHUNK]


def test_dormant_audio_does_not_bar_a_turn_engine_opt_in():
    """The server takes dormant audio as no first audio frame, and so does the client."""
    ws = _FakeServerWS({
        **_accepting(),
        "turn_engine": lambda sent: [{"type": "turn_engine_ready", "engine": "tyger"}]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.enable_wake_word()
            await session.send_audio(CHUNK)
            engine = await session.enable_turn_engine("tyger")
            ws.end()
            return engine
    assert asyncio.run(run()) == "tyger"


def test_a_stopped_session_with_no_opt_in_still_drops_its_audio():
    ws = _FakeServerWS(
        {"listen_stop": lambda sent: [ListenStoppedMessage(mic_sample=0).model_dump()]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.listen_stop()
            await session.send_audio(CHUNK)
            ws.end()
            return session.dormant
    assert asyncio.run(run()) is False
    assert CHUNK not in _sent(client)


# --- the wake ----------------------------------------------------------------------

def test_a_wake_frame_makes_the_session_listen_and_comes_out_of_wakes():
    ws = _FakeServerWS(_accepting())
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.enable_wake_word(["hey hermes"])
            await session.send_audio(CHUNK)
            await session.send_audio(CHUNK)
            ws.push(_wake(0, 320, "hey hermes"))
            await _settle()
            state = (session.listening, session.dormant, session.mic_sample)
            await session.send_audio(CHUNK)
            after = session.mic_sample
            ws.end()
            return state, after, [w async for w in session.wakes()]
    state, after, wakes = asyncio.run(run())
    assert state == (True, False, 0)
    assert after == 160
    assert wakes == [_wake(0, 320, "hey hermes")]
    assert wakes[0]["phrase"] == "hey hermes"


def test_audio_in_flight_at_the_wake_is_counted_as_the_server_counts_it():
    """Three frames were sent before the wake frame was read; the server took the
    first as dormant audio and woke, so it counted the other two as mic audio."""
    ws = _FakeServerWS(_accepting())
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.enable_wake_word()
            for _ in range(3):
                await session.send_audio(CHUNK)
            ws.push(_wake(0, 160))
            await _settle()
            first = session.mic_sample
            await session.send_audio(CHUNK)
            ws.end()
            return first, session.mic_sample
    assert asyncio.run(run()) == (320, 480)


def test_audio_sent_between_the_opt_in_and_its_answer_is_dormant_audio():
    """The server is dormant from the wake_word frame on, not from its answer."""
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.send_audio(CHUNK)
            call = asyncio.create_task(session.enable_wake_word())
            await _settle()
            await session.send_audio(CHUNK)      # after the frame, before the answer
            ws.push(_ready())
            await call
            counted = session.mic_sample
            await session.send_audio(CHUNK)
            # The server heard the phrase in the first of the two dormant frames.
            ws.push(_wake(160, 160))
            await _settle()
            ws.end()
            return counted, session.mic_sample
    assert asyncio.run(run()) == (160, 320)


def test_a_listen_start_by_hand_that_crossed_the_wake_keeps_the_count():
    """The session woke itself after two dormant frames; the server had heard the
    phrase in the first, so it counted the second as mic audio."""
    ws = _FakeServerWS(_accepting())
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.enable_wake_word()
            await session.send_audio(CHUNK)
            await session.send_audio(CHUNK)
            await session.listen_start()
            await session.send_audio(CHUNK)
            ws.push(_wake(0, 160))
            await _settle()
            ws.end()
            return session.mic_sample, session.listening
    assert asyncio.run(run()) == (320, True)


# A send can wait: the frame is written to the socket and the call returns only when
# the write buffer has room. Another task, or the reader, runs in that wait. The frame
# is in the session's count from the moment it is on the wire, so each reader of the
# count below sees it.

def test_a_wake_read_while_a_dormant_frames_send_waits_counts_that_frame_as_mic_audio():
    """The server heard the phrase in the first frame, so it counted the second,
    whose send had not returned yet, as mic audio."""
    ws = _FakeServerWS(_accepting())
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.enable_wake_word()
            await session.send_audio(CHUNK)
            ws.pause = asyncio.Event()
            sending = asyncio.create_task(session.send_audio(CHUNK))
            await _settle()
            ws.push(_wake(0, 160))
            await _settle()
            ws.pause.set()
            await sending
            ws.end()
            return session.mic_sample, session.listening
    assert asyncio.run(run()) == (160, True)


def test_an_opt_in_sent_while_a_mic_frames_send_waits_leaves_that_frame_mic_audio():
    """The mic frame was on the wire before the wake_word frame, so the server took
    it as mic audio."""
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            ws.pause = asyncio.Event()
            sending = asyncio.create_task(session.send_audio(CHUNK))
            await _settle()
            call = asyncio.create_task(session.enable_wake_word())
            await _settle()
            ws.pause.set()
            await sending
            ws.push(_ready())
            await call
            ws.end()
            return session.mic_sample, session.dormant
    assert asyncio.run(run()) == (160, True)


def test_a_listen_stop_sent_while_a_dormant_frames_send_waits_has_that_frame_in_its_mark():
    """The server heard the phrase in the second dormant frame, whose send had not
    returned when the listen_stop went out. It woke and then got the listen_stop, so
    it is dormant again, and so is the session."""
    ws = _FakeServerWS(_accepting())
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.enable_wake_word()
            await session.send_audio(CHUNK)
            ws.pause = asyncio.Event()
            sending = asyncio.create_task(session.send_audio(CHUNK))
            await _settle()
            stop = asyncio.create_task(session.listen_stop())
            await _settle()
            ws.push(_wake(0, 320))
            ws.push(ListenStoppedMessage(mic_sample=0).model_dump())
            ws.pause.set()
            await sending
            count = await stop
            state = (session.dormant, session.mic_sample)
            ws.end()
            return count, state, [w async for w in session.wakes()]
    assert asyncio.run(run()) == (0, (True, 0), [])


def test_a_wake_frame_with_no_opt_in_is_ignored():
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.send_audio(CHUNK)
            ws.push(_wake(0, 0))
            await _settle()
            ws.end()
            return session.mic_sample, session.listening, [w async for w in session.wakes()]
    assert asyncio.run(run()) == (160, True, [])


# --- dormant again -----------------------------------------------------------------

def test_listen_stop_after_a_wake_makes_the_session_dormant_on_a_new_count():
    stops = iter([160])
    ws = _FakeServerWS({
        **_accepting(),
        "listen_stop": lambda sent: [ListenStoppedMessage(mic_sample=next(stops)).model_dump()]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.enable_wake_word()
            await session.send_audio(CHUNK)
            ws.push(_wake(0, 160))
            await _settle()
            await session.send_audio(CHUNK)
            assert await session.listen_stop() == 160
            dormant = session.dormant
            # Two frames in the new dormant period; the server hears the phrase in
            # the first, so the second is mic audio.
            await session.send_audio(CHUNK)
            await session.send_audio(CHUNK)
            ws.push(_wake(160, 160))
            await _settle()
            ws.end()
            return dormant, session.mic_sample, session.listening
    assert asyncio.run(run()) == (True, 320, True)


def test_a_listen_stop_sent_while_dormant_that_crossed_the_wake_keeps_the_session_stopped():
    """The server heard the phrase in the first of two dormant frames, woke, counted
    the second as mic audio, and then got the listen_stop: it is dormant again. The
    wake is not reported, and the counts agree."""
    ws = _FakeServerWS(_accepting())
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.enable_wake_word()
            await session.send_audio(CHUNK)
            await session.send_audio(CHUNK)
            stop = asyncio.create_task(session.listen_stop())
            await _settle()
            await session.send_audio(CHUNK)      # dormant audio of the new period
            ws.push(_wake(0, 160))
            ws.push(ListenStoppedMessage(mic_sample=160).model_dump())
            count = await stop
            state = (session.dormant, session.mic_sample)
            # The new period counts from the listen_stop: one frame so far.
            ws.push(_wake(160, 160))
            await _settle()
            ws.end()
            return count, state, session.listening, [w async for w in session.wakes()]
    count, state, listening, wakes = asyncio.run(run())
    assert count == 160
    assert state == (True, 160)
    assert listening is True
    assert wakes == [_wake(160, 160)]


def test_a_listen_stop_while_dormant_that_the_server_got_first_changes_nothing():
    """The stop reached the server while it was dormant: no new count, and the wake
    that follows is a real one."""
    ws = _FakeServerWS({
        **_accepting(),
        "listen_stop": lambda sent: [ListenStoppedMessage(mic_sample=0).model_dump()]})
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.enable_wake_word()
            await session.send_audio(CHUNK)
            assert await session.listen_stop() == 0
            await session.send_audio(CHUNK)
            await session.send_audio(CHUNK)
            ws.push(_wake(0, 320))               # heard in the second frame
            await _settle()
            ws.end()
            return session.mic_sample, session.listening, len(
                [w async for w in session.wakes()])
    assert asyncio.run(run()) == (160, True, 1)


def test_a_second_opt_in_starts_a_new_count_of_dormant_audio():
    ws = _FakeServerWS(_accepting())
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.enable_wake_word()
            await session.send_audio(CHUNK)
            assert await session.enable_wake_word(["hey hermes"]) == ["hey hermes"]
            await session.send_audio(CHUNK)
            await session.send_audio(CHUNK)
            ws.push(_wake(0, 160, "hey hermes"))
            await _settle()
            ws.end()
            return session.mic_sample, session.wake_phrases
    assert asyncio.run(run()) == (160, ["hey hermes"])


# --- a reconnect, and a session that never opts in -----------------------------------

def test_a_reconnect_starts_the_new_connection_with_no_opt_in(monkeypatch):
    ws = _FakeServerWS(_accepting())
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.enable_wake_word()
            await session.send_audio(CHUNK)
            monkeypatch.setattr(client, "_disconnect", AsyncMock())
            monkeypatch.setattr(client, "_connect", AsyncMock())
            await client._reconnect()
            return session.wake_phrases, session.dormant, session.listening, client._dormant_bytes
    assert asyncio.run(run()) == (None, False, True, 0)


def test_closing_a_dormant_session_sends_listen_start():
    ws = _FakeServerWS(_accepting())
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.enable_wake_word()
        ws.end()
    asyncio.run(run())
    assert _sent(client) == [{"type": "wake_word"}, {"type": "listen_start"}]
    assert client._listen_stopped is False


def test_a_session_that_never_opts_in_is_unchanged():
    ws = _FakeServerWS()
    client = _client(ws)

    async def run():
        _start_reader(client)
        async with client.duplex() as session:
            await session.send_audio(CHUNK)
            await session.send_audio(CHUNK)
            ws.end()
            return (session.mic_sample, session.listening, session.dormant,
                    session.wake_phrases, [w async for w in session.wakes()])
    assert asyncio.run(run()) == (320, True, False, None, [])
    assert _sent(client) == [CHUNK, CHUNK]


def test_a_lost_connection_while_sending_the_opt_in_raises(monkeypatch):
    ws = _FakeServerWS()
    client = _client(ws)
    ws.send = AsyncMock(side_effect=websockets.exceptions.ConnectionClosed(None, None))

    async def run():
        monkeypatch.setattr(client, "_reconnect", AsyncMock())
        async with client.duplex() as session:
            with pytest.raises(WakeWordError):
                await session.enable_wake_word()
    asyncio.run(run())
