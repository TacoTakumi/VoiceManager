"""A failed TTS request raises TTSError in the one-sided client methods.

Model-free and server-free: the socket is a fake that answers each TTS request
with a scripted list of frames, and the client's real reader task reads them in
order. Covers: synthesize_tts, stream_tts and stream_tts_live raise TTSError at
their request's audio_end with reason error, with the error_type and message of
the TTS error frame read for that request in the audiocpp order (audio_start,
error, audio_end) and the VoXtream order (error, audio_start, audio_end), and with
error_type None when no error frame came; a stored error frame of an earlier
request does not mark a later request, also when that earlier request's error
frame and audio_end come after the later request started, and also when the
reader reads them and the later request's own frames in one batch; the reasons
complete, cancelled and interrupted return as before; stream_tts_live yields the
audio that came before the audio_end and then raises, in its send loop and after
it; DuplexSession.audio() yields the audio_end with no client-private key. With
error frames that carry a request_id: a cancel ack of another id between the
frame and its audio_end keeps the frame, an earlier request's named frame never
marks a later request, and no stored frame outlives its request's audio_end, a
request start or a reconnect, nor grows past a cap.
"""
import asyncio
import json

import pytest

from voicemanager.client import TTSError, VoiceManagerClient

PCM_A = b"\x01\x02" * 8
PCM_B = b"\x03\x04" * 8

UNAVAILABLE_MSG = "The TTS engine is unavailable (load_failed). A later request may succeed."
UNKNOWN_VOICE_MSG = "unknown voice 'nope'; valid presets: a, b"


def _start(rid: str) -> str:
    return json.dumps({"type": "audio_start", "request_id": rid, "sample_rate": 24000})


def _end(rid: str, reason: str) -> str:
    return json.dumps({"type": "audio_end", "request_id": rid, "reason": reason})


def _err(error_type: str, message: str) -> str:
    return json.dumps({"type": "error", "error_type": error_type, "message": message})


def _audiocpp_order(rid: str) -> list:
    return [_start(rid), PCM_A, _err("tts_unavailable", UNAVAILABLE_MSG), _end(rid, "error")]


def _voxtream_order(rid: str) -> list:
    return [_err("unknown_voice", UNKNOWN_VOICE_MSG), _start(rid), _end(rid, "error")]


def _no_error_frame(rid: str) -> list:
    return [_start(rid), PCM_A, _end(rid, "error")]


def _complete(rid: str) -> list:
    return [_start(rid), PCM_A, PCM_B, _end(rid, "complete")]


class _ScriptedWS:
    """A socket that answers each TTS request with the next scripted frame list.

    A request is answered when its last frame is sent: a ``tts`` frame, or a
    ``tts_stream`` frame with ``done`` true. Each script is a function of the
    request_id that returns the frames to deliver, in wire order.
    """

    def __init__(self, scripts: list) -> None:
        self.sent: list = []
        self._scripts = list(scripts)
        self._incoming: asyncio.Queue = asyncio.Queue()

    async def send(self, data) -> None:
        self.sent.append(data)
        frame = json.loads(data)
        ends_request = (frame.get("type") == "tts"
                        or (frame.get("type") == "tts_stream" and frame.get("done")))
        if ends_request and self._scripts:
            for item in self._scripts.pop(0)(frame["request_id"]):
                self._incoming.put_nowait(item)

    def push(self, item) -> None:
        """Deliver a frame that answers no request (read while the client is idle)."""
        self._incoming.put_nowait(item)

    async def close(self) -> None:
        self._incoming.put_nowait(None)

    def __aiter__(self):
        return self._frames()

    async def _frames(self):
        while (frame := await self._incoming.get()) is not None:
            yield frame


async def _text():
    yield "Hello there."


async def _run_synthesize(client):
    return await client.synthesize_tts("Hello there.")


async def _run_stream(client):
    return await client.stream_tts(_text())


async def _run_live(client, sink: list | None = None):
    got = [] if sink is None else sink
    async for pcm in client.stream_tts_live(_text()):
        got.append(pcm)
    return b"".join(got), client.last_tts_sample_rate


METHODS = {
    "synthesize_tts": _run_synthesize,
    "stream_tts": _run_stream,
    "stream_tts_live": _run_live,
}


@pytest.fixture
async def make_client():
    """Build a client on a scripted socket with its real reader task running."""
    made: list = []

    def _make(scripts: list) -> tuple[VoiceManagerClient, _ScriptedWS]:
        client = VoiceManagerClient(token="t")
        ws = _ScriptedWS(scripts)
        client._ws = ws
        client.connected = True
        client._reader = asyncio.get_running_loop().create_task(client._reader_loop())
        made.append((client, ws))
        return client, ws

    yield _make
    for client, ws in made:
        await ws.close()
        await asyncio.wait_for(client._reader, timeout=1.0)


async def _idle(client) -> None:
    """Let the reader task read every frame delivered so far."""
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("order, error_type, message", [
    (_audiocpp_order, "tts_unavailable", UNAVAILABLE_MSG),
    (_voxtream_order, "unknown_voice", UNKNOWN_VOICE_MSG),
], ids=["audiocpp_order", "voxtream_order"])
async def test_error_frame_gives_its_error_type(make_client, method, order, error_type,
                                                message):
    """The request's error frame, before or after audio_start, sets the TTSError."""
    client, _ = make_client([order])
    with pytest.raises(TTSError) as excinfo:
        await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    assert excinfo.value.error_type == error_type
    assert str(excinfo.value) == message


@pytest.mark.parametrize("method", list(METHODS))
async def test_no_error_frame_gives_error_type_none(make_client, method):
    """The one-shot path sends audio_end reason error with no error frame."""
    client, _ = make_client([_no_error_frame])
    with pytest.raises(TTSError) as excinfo:
        await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    assert excinfo.value.error_type is None
    assert str(excinfo.value)


@pytest.mark.parametrize("method", list(METHODS))
async def test_error_frame_still_goes_to_error_queue(make_client, method):
    """Keeping the frame for the request does not take it off the error queue."""
    client, _ = make_client([_voxtream_order])
    with pytest.raises(TTSError):
        await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    frame = client._error_queue.get_nowait()
    assert frame["error_type"] == "unknown_voice"


@pytest.mark.parametrize("method", list(METHODS))
async def test_earlier_error_does_not_mark_a_later_complete_request(make_client, method):
    """A request that ends with reason complete after a failed one returns its audio."""
    client, _ = make_client([_voxtream_order, _complete])
    with pytest.raises(TTSError) as excinfo:
        await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    assert excinfo.value.error_type == "unknown_voice"

    pcm, rate = await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    assert pcm == PCM_A + PCM_B
    assert rate == 24000


@pytest.mark.parametrize("method", list(METHODS))
async def test_earlier_error_does_not_name_a_later_failed_request(make_client, method):
    """A later failed request with no error frame of its own gets error_type None."""
    client, _ = make_client([_voxtream_order, _no_error_frame])
    with pytest.raises(TTSError) as first:
        await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    assert first.value.error_type == "unknown_voice"

    with pytest.raises(TTSError) as second:
        await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    assert second.value.error_type is None


@pytest.mark.parametrize("method", list(METHODS))
async def test_error_read_while_idle_is_cleared_at_request_start(make_client, method):
    """An error frame read before the request starts does not name its failure."""
    client, ws = make_client([_no_error_frame])
    ws.push(_err("tts_unavailable", UNAVAILABLE_MSG))
    await _idle(client)

    with pytest.raises(TTSError) as excinfo:
        await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    assert excinfo.value.error_type is None


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("reason", ["complete", "cancelled", "interrupted"])
async def test_other_reasons_return_audio(make_client, method, reason):
    """Reasons other than error return the audio, even after a TTS error frame."""
    def _script(rid: str) -> list:
        return [_start(rid), PCM_A, _err("unknown_voice", UNKNOWN_VOICE_MSG),
                _end(rid, reason)]

    client, _ = make_client([_script])
    pcm, rate = await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    assert pcm == PCM_A
    assert rate == 24000


@pytest.mark.parametrize("method", list(METHODS))
async def test_other_error_types_are_not_kept(make_client, method):
    """Only tts_unavailable and unknown_voice name a failed TTS request."""
    def _script(rid: str) -> list:
        return [_start(rid), _err("invalid_request", "bad frame"), _end(rid, "error")]

    client, _ = make_client([_script])
    with pytest.raises(TTSError) as excinfo:
        await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    assert excinfo.value.error_type is None


async def test_stream_tts_live_yields_audio_before_the_error(make_client):
    """Both pcm frames reach the caller, then the iteration raises."""
    def _script(rid: str) -> list:
        return [_start(rid), PCM_A, PCM_B, _err("tts_unavailable", UNAVAILABLE_MSG),
                _end(rid, "error")]

    client, _ = make_client([_script])
    got: list = []
    with pytest.raises(TTSError) as excinfo:
        await asyncio.wait_for(_run_live(client, got), timeout=2.0)
    assert got == [PCM_A, PCM_B]
    assert excinfo.value.error_type == "tts_unavailable"
    assert str(excinfo.value) == UNAVAILABLE_MSG


def test_tts_error_keeps_the_one_argument_form():
    """A TTSError built with a message alone has error_type None."""
    err = TTSError("Connection lost during TTS synthesis")
    assert str(err) == "Connection lost during TTS synthesis"
    assert err.error_type is None
    assert TTSError("x", "unknown_voice").error_type == "unknown_voice"


# The request_id of an earlier request the caller gave up on (task cancelled,
# timeout, break) that the server still finishes.
EARLIER_RID = "earlier-request"


def _earlier_failed() -> list:
    """The late error frame and audio_end of the earlier request."""
    return [_err("unknown_voice", UNKNOWN_VOICE_MSG), _start(EARLIER_RID),
            _end(EARLIER_RID, "error")]


@pytest.mark.parametrize("method", list(METHODS))
async def test_late_error_of_an_earlier_request_does_not_name_a_later_one(make_client,
                                                                          method):
    """The earlier request's frame and audio_end come after the later request started.

    For stream_tts_live this is the audio_end read after the send loop.
    """
    def _script(rid: str) -> list:
        return _earlier_failed() + _no_error_frame(rid)

    client, _ = make_client([_script])
    with pytest.raises(TTSError) as excinfo:
        await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    assert excinfo.value.error_type is None
    assert str(excinfo.value) == "The server ended the TTS request with an error"


async def _text_with_frames(ws, frames: list):
    """Deliver frames while text is still sent, so the send loop reads them."""
    yield "Hello there."
    for item in frames:
        ws.push(item)
    for _ in range(5):
        await asyncio.sleep(0)
    yield "How are you?"
    yield "Fine."


async def test_live_late_error_of_an_earlier_request_while_text_is_sent(make_client):
    """The same, with the audio_end read while text chunks are still sent."""
    client, ws = make_client([])
    rid = "later-request"
    frames = _earlier_failed() + _no_error_frame(rid)
    with pytest.raises(TTSError) as excinfo:
        async for _ in client.stream_tts_live(_text_with_frames(ws, frames),
                                              request_id=rid):
            pass
    assert excinfo.value.error_type is None
    assert str(excinfo.value) == "The server ended the TTS request with an error"
    # The error came in the send loop: the done frame was never sent.
    assert not any(json.loads(d).get("done") for d in ws.sent)


async def test_live_own_error_while_text_is_sent(make_client):
    """The request's own error frame and audio_end, read while text is still sent."""
    client, ws = make_client([])
    rid = "live-request"
    frames = _audiocpp_order(rid)
    got: list = []
    with pytest.raises(TTSError) as excinfo:
        async for pcm in client.stream_tts_live(_text_with_frames(ws, frames),
                                                request_id=rid):
            got.append(pcm)
    assert got == [PCM_A]
    assert excinfo.value.error_type == "tts_unavailable"
    assert str(excinfo.value) == UNAVAILABLE_MSG
    assert not any(json.loads(d).get("done") for d in ws.sent)


def _earlier_failed_then_own(rid: str) -> list:
    """The earlier request's late failure, then the later request's own failure."""
    return _earlier_failed() + [_err("tts_unavailable", UNAVAILABLE_MSG), _start(rid),
                                _end(rid, "error")]


@pytest.mark.parametrize("method", list(METHODS))
async def test_own_error_after_an_earlier_late_error_read_in_one_batch(make_client,
                                                                        method):
    """The reader reads all the frames before the method reads its queue.

    The later request still gets its own error frame, not the earlier one and not
    none. For stream_tts_live this is the audio_end read after the send loop.
    """
    client, _ = make_client([_earlier_failed_then_own])
    with pytest.raises(TTSError) as excinfo:
        await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    assert excinfo.value.error_type == "tts_unavailable"
    assert str(excinfo.value) == UNAVAILABLE_MSG


async def test_live_own_error_after_an_earlier_late_error_while_text_is_sent(make_client):
    """The same, with the audio_end read while text chunks are still sent."""
    client, ws = make_client([])
    rid = "later-request"
    with pytest.raises(TTSError) as excinfo:
        async for _ in client.stream_tts_live(
                _text_with_frames(ws, _earlier_failed_then_own(rid)), request_id=rid):
            pass
    assert excinfo.value.error_type == "tts_unavailable"
    assert str(excinfo.value) == UNAVAILABLE_MSG
    assert not any(json.loads(d).get("done") for d in ws.sent)


async def test_duplex_audio_end_has_only_wire_fields(make_client):
    """The audio_end of a failed request in DuplexSession.audio() is the wire frame."""
    client, ws = make_client([])
    rid = "duplex-request"
    for item in _audiocpp_order(rid):
        ws.push(item)
    events: list = []
    async with client.duplex() as session:
        async for event in session.audio():
            events.append(event)
            if event[0] == "audio_end":
                break
    assert events[-1] == ("audio_end",
                          {"type": "audio_end", "request_id": rid, "reason": "error"})
    assert client._error_queue.get_nowait()["error_type"] == "tts_unavailable"


# A server that names the request on its TTS error frames. The client ties such a
# frame to the audio_end of that request_id, whatever audio_end comes between.
FOREIGN_RID = "other-request"


def _err_for(rid: str, error_type: str, message: str) -> str:
    return json.dumps({"type": "error", "error_type": error_type, "message": message,
                       "request_id": rid})


def _audiocpp_with_ack(rid: str) -> list:
    """The audiocpp order with a cancel ack of another id after the error frame."""
    return [_start(rid), PCM_A, _err_for(rid, "tts_unavailable", UNAVAILABLE_MSG),
            _end(FOREIGN_RID, "cancelled"), _end(rid, "error")]


def _voxtream_with_ack(rid: str) -> list:
    """The VoXtream order with a cancel ack of another id after the error frame."""
    return [_err_for(rid, "unknown_voice", UNKNOWN_VOICE_MSG),
            _end(FOREIGN_RID, "cancelled"), _start(rid), _end(rid, "error")]


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("order, error_type, message", [
    (_audiocpp_with_ack, "tts_unavailable", UNAVAILABLE_MSG),
    (_voxtream_with_ack, "unknown_voice", UNKNOWN_VOICE_MSG),
], ids=["audiocpp_order", "voxtream_order"])
async def test_cancel_ack_of_another_id_keeps_the_error_frame(make_client, method, order,
                                                              error_type, message):
    """An audio_end of another request_id does not take the request's frame.

    For stream_tts_live this is the audio_end read after the send loop.
    """
    client, _ = make_client([order])
    with pytest.raises(TTSError) as excinfo:
        await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    assert excinfo.value.error_type == error_type
    assert str(excinfo.value) == message
    assert client._tts_error_frames == {}


@pytest.mark.parametrize("order, error_type", [
    (_audiocpp_with_ack, "tts_unavailable"),
    (_voxtream_with_ack, "unknown_voice"),
], ids=["audiocpp_order", "voxtream_order"])
async def test_live_cancel_ack_of_another_id_while_text_is_sent(make_client, order,
                                                                error_type):
    """The same, with the audio_end read while text chunks are still sent."""
    client, ws = make_client([])
    rid = "live-request"
    with pytest.raises(TTSError) as excinfo:
        async for _ in client.stream_tts_live(_text_with_frames(ws, order(rid)),
                                              request_id=rid):
            pass
    assert excinfo.value.error_type == error_type
    assert not any(json.loads(d).get("done") for d in ws.sent)
    assert client._tts_error_frames == {}


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("earlier_end", [True, False], ids=["earlier_end", "no_earlier_end"])
async def test_earlier_frame_with_its_id_does_not_mark_a_later_request(make_client, method,
                                                                       earlier_end):
    """An earlier request's frame, read after the later request started, is not its.

    The earlier request's audio_end comes after the later one's, or never.
    """
    def _script(rid: str) -> list:
        frames = [_err_for(EARLIER_RID, "unknown_voice", UNKNOWN_VOICE_MSG),
                  _start(rid), _end(rid, "error")]
        if earlier_end:
            frames.append(_end(EARLIER_RID, "error"))
        return frames

    client, _ = make_client([_script])
    with pytest.raises(TTSError) as excinfo:
        await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    assert excinfo.value.error_type is None
    assert str(excinfo.value) == "The server ended the TTS request with an error"


def _earlier_then_own_with_ids(rid: str) -> list:
    """The earlier request's late failure, then the later request's own, both named."""
    return [_err_for(EARLIER_RID, "unknown_voice", UNKNOWN_VOICE_MSG), _start(EARLIER_RID),
            _end(EARLIER_RID, "error"),
            _err_for(rid, "tts_unavailable", UNAVAILABLE_MSG), _start(rid),
            _end(rid, "error")]


@pytest.mark.parametrize("method", list(METHODS))
async def test_own_named_error_after_an_earlier_one_read_in_one_batch(make_client, method):
    """The reader reads all the frames first; the later request gets its own frame."""
    client, _ = make_client([_earlier_then_own_with_ids])
    with pytest.raises(TTSError) as excinfo:
        await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    assert excinfo.value.error_type == "tts_unavailable"
    assert str(excinfo.value) == UNAVAILABLE_MSG
    assert client._tts_error_frames == {}


@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("reason", ["complete", "cancelled", "interrupted"])
async def test_named_frame_is_removed_at_an_audio_end_of_another_reason(make_client, method,
                                                                        reason):
    """An audio_end of the request with any reason removes its stored frame."""
    def _script(rid: str) -> list:
        return [_start(rid), PCM_A, _err_for(rid, "unknown_voice", UNKNOWN_VOICE_MSG),
                _end(rid, reason)]

    client, _ = make_client([_script])
    pcm, _ = await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    assert pcm == PCM_A
    assert client._tts_error_frames == {}


@pytest.mark.parametrize("method", list(METHODS))
async def test_named_frame_read_while_idle_is_cleared_at_request_start(make_client, method):
    """A stored frame of a request that got no audio_end goes when a request starts."""
    client, ws = make_client([_complete])
    ws.push(_err_for(EARLIER_RID, "unknown_voice", UNKNOWN_VOICE_MSG))
    await _idle(client)
    assert EARLIER_RID in client._tts_error_frames

    await asyncio.wait_for(METHODS[method](client), timeout=2.0)
    assert client._tts_error_frames == {}


async def test_stored_named_frames_are_capped(make_client):
    """Frames of requests that never get an audio_end do not pile up without limit."""
    client, ws = make_client([])
    for n in range(200):
        ws.push(_err_for(f"lost-{n}", "tts_unavailable", UNAVAILABLE_MSG))
    for _ in range(50):
        await asyncio.sleep(0)
    assert 0 < len(client._tts_error_frames) <= 32
    # The newest frames are the ones kept.
    assert "lost-199" in client._tts_error_frames


async def test_reconnect_clears_stored_error_frames(monkeypatch):
    """A new connection starts with no stored TTS error frame."""
    from unittest.mock import AsyncMock

    client = VoiceManagerClient(token="t")
    client._tts_error_frames = {"a": {"type": "error", "error_type": "unknown_voice"}}
    client._tts_error_frame = {"type": "error", "error_type": "tts_unavailable"}
    monkeypatch.setattr(client, "_disconnect", AsyncMock())
    monkeypatch.setattr(client, "_connect", AsyncMock())
    await client._reconnect_now()
    assert client._tts_error_frames == {}
    assert client._tts_error_frame is None


async def test_duplex_named_frame_audio_end_has_only_wire_fields(make_client):
    """DuplexSession.audio() yields the wire audio_end; a cancel ack between is kept."""
    client, ws = make_client([])
    rid = "duplex-request"
    for item in _audiocpp_with_ack(rid):
        ws.push(item)
    events: list = []
    async with client.duplex() as session:
        async for event in session.audio():
            events.append(event)
            if event[0] == "audio_end" and event[1]["request_id"] == rid:
                break
    ends = [e[1] for e in events if e[0] == "audio_end"]
    assert ends == [{"type": "audio_end", "request_id": FOREIGN_RID, "reason": "cancelled"},
                    {"type": "audio_end", "request_id": rid, "reason": "error"}]
    assert client._tts_error_frames == {}
