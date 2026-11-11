"""Async Python client for VoiceManager STT/TTS server.

Standalone module - requires only ``websockets`` and ``httpx``.
Copy this file into any Python 3.12+ project.

The client authenticates with a bearer token: it sends
``Authorization: Bearer <token>`` on the WebSocket handshake and on every
REST call, and never puts the token in a URL.

Usage::

    async with VoiceManagerClient(host="localhost", port=8255, token="secret") as client:
        results = await client.transcribe_file("audio.wav")
        pcm, rate = await client.synthesize_tts("Hello world")
        client.save_tts_to_file(pcm, "output.wav", rate)
"""
import asyncio
import io
import json
import logging
import uuid
import wave
from typing import AsyncIterator

import httpx
import websockets.asyncio.client
import websockets.exceptions


logger = logging.getLogger("voicemanager.client")

# The tts_stream chunking modes the server accepts.
CHUNKING_MODES = ("sentence", "clause")

# The error_types the server answers a turn_engine opt-in with when it refuses it:
# a bad or late opt-in, the turn engine off on the server, or an STT engine the turn
# engine does not support. This module stands alone, so these mirror the server's
# protocol constants rather than import them.
_TURN_ENGINE_REFUSALS = ("invalid_request", "turn_engine_unavailable",
                         "turn_engine_unsupported")
# The error_type the server sends once when the turn engine's process exits; the
# session then goes on as one that never opted in.
_TURN_ENGINE_LOST = "turn_engine_lost"
# How the message of the server's invalid_request error frame starts when it refuses
# a wake_word opt-in. The refusal has no error_type of its own, so the message tells
# it from the other invalid_request frames.
_WAKE_REFUSAL = "wake_word refused"


def _wav_payload_offset(data: bytes) -> int | None:
    """Where the PCM starts in a WAV container, or None when data is not one.

    The offset is past the 'data' chunk header; a container with no 'data' chunk
    in this frame holds no samples (the whole frame is header). It matches the
    server's rule, so the client counts the mic samples the server counts.
    """
    if not (len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"):
        return None
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos:pos + 4]
        size = int.from_bytes(data[pos + 4:pos + 8], "little")
        if chunk_id == b"data":
            return pos + 8
        pos += 8 + size + (size & 1)
    return len(data)


def _check_chunking(chunking: str | None) -> None:
    """Refuse an unknown chunking mode before a request opens.

    The server drops a first frame with a bad value (an invalid_request error that the
    streaming calls do not read) and the next frame then opens the request in sentence
    mode, so the first chunk's words would be lost without an error.
    """
    if chunking is not None and chunking not in CHUNKING_MODES:
        raise ValueError(
            f"chunking must be one of {CHUNKING_MODES} or None, got {chunking!r}")


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class VoiceManagerError(Exception):
    """Base exception for all VoiceManager client errors."""


class AuthenticationError(VoiceManagerError):
    """Raised when the server rejects the token (HTTP 401 or 403)."""


class ConnectionError(VoiceManagerError):  # noqa: A001 - shadows builtin intentionally
    """Raised when the WebSocket connection cannot be established or is lost."""


class STTError(VoiceManagerError):
    """Raised when an error occurs during STT streaming."""


class TTSError(VoiceManagerError):
    """Raised when an error occurs during TTS synthesis."""


class TurnEngineError(VoiceManagerError):
    """Raised when a turn_engine opt-in fails.

    ``error_type`` is the server's error_type when it refused the opt-in (for
    example ``turn_engine_unavailable``), and None when no answer came (a timeout or
    a lost connection).
    """

    def __init__(self, message: str, error_type: str | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type


class WakeWordError(VoiceManagerError):
    """Raised when a wake_word opt-in fails.

    ``error_type`` is the server's error_type when it refused the opt-in
    (``invalid_request``: no wake engine on the server, or a phrase list it cannot
    use), and None when no answer came (a timeout, for example from a server older
    than 0.9.0, or a lost connection).
    """

    def __init__(self, message: str, error_type: str | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type


def _check_sample(name: str, value) -> None:
    """Refuse a sample index the server would reject as a malformed playback frame."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")


# ---------------------------------------------------------------------------
# VoiceManagerClient
# ---------------------------------------------------------------------------

class TTSRequest:
    """One streaming TTS reply on a DuplexSession.

    Made by ``DuplexSession.tts_request``. ``send`` appends text to the request (the
    first frame carries the voice and the chunking the request was made with),
    ``flush`` speaks the held text now, ``done`` ends the text and ``cancel`` stops
    the reply. The audio comes back on ``DuplexSession.audio`` under this
    ``request_id``.
    """

    def __init__(self, client: "VoiceManagerClient", request_id: str,
                 voice: str | None, chunking: str | None) -> None:
        self._client = client
        self.request_id = request_id
        self._voice = voice
        self._chunking = chunking
        self._first = True
        self.done_sent = False
        self.cancelled = False

    async def send(self, text: str, flush: bool = False) -> None:
        """Append ``text`` to the request; with ``flush`` the server speaks it now."""
        if self.done_sent:
            raise VoiceManagerError(
                f"TTS request {self.request_id} is closed: done() was already sent")
        payload = {"type": "tts_stream", "request_id": self.request_id,
                   "text": text, "done": False}
        if flush:
            payload["flush"] = True
        if self._first:
            if self._voice is not None:
                payload["voice"] = self._voice
            if self._chunking is not None:
                payload["chunking"] = self._chunking
            self._first = False
        await self._client._send_or_raise(json.dumps(payload), TTSError)

    async def flush(self) -> None:
        """Speak the text the server holds for this request now; keep it open."""
        await self.send("", flush=True)

    async def done(self) -> None:
        """End the request's text; the server answers with one audio_end."""
        if self.done_sent:
            return
        self.done_sent = True
        payload = {"type": "tts_stream", "request_id": self.request_id, "text": "",
                   "done": True}
        if self._first:  # a reply with no text at all still names its voice
            if self._voice is not None:
                payload["voice"] = self._voice
            if self._chunking is not None:
                payload["chunking"] = self._chunking
            self._first = False
        await self._client._send_or_raise(json.dumps(payload), TTSError)

    async def cancel(self) -> None:
        """Stop the reply: the server answers with one audio_end, reason cancelled."""
        self.cancelled = True
        self.done_sent = True
        await self._client.cancel_tts(self.request_id)


class DuplexSession:
    """STT and TTS at the same time on one connection.

    Made by ``VoiceManagerClient.duplex()``. The send methods and the two read
    iterators are independent, so one task can stream the microphone with
    ``send_audio`` while another reads ``transcripts()``, a third feeds a
    ``TTSRequest`` and a fourth plays what ``audio()`` yields. While a session is
    open the client's one-sided methods (``stream_stt_live``, ``stream_tts_live``
    and the batch ones) refuse, since they read the same reply queues.

    ``audio()`` yields ``("audio_start", frame)``, ``("pcm", bytes, request_id)`` and
    ``("audio_end", frame)`` in arrival order. The pcm of a cancelled request is
    dropped until its audio_end, which is still yielded. Each iterator ends when
    the connection closes.

    Turn engine: ``await session.enable_turn_engine("tyger")`` before the first
    ``send_audio`` opts the connection in. The session then reports each reply's
    playback edges with ``playback_start`` and ``playback_stop``, reads the turn_end
    frames from ``turn_ends()``, and sees ``("barge", frame)`` in ``audio()`` when
    the user cut in over a reply; that request's audio_end (reason ``interrupted``)
    follows. A turn_engine_lost error frame on ``errors()`` means the opt-in is
    gone and the session goes on as one that never opted in. A reconnect starts a
    new connection that has not opted in, with ``mic_sample`` back at 0.

    Listening boundary: ``await session.listen_stop()`` when the mic stops (the user
    stops listening, a conversation ends) keeps the connection open but stopped. It
    returns the server's count of mic samples accepted, or None from a server older
    than 0.8.0. While stopped, ``send_audio`` drops its frames without sending or
    counting them, and TTS goes on as usual. ``await session.listen_start()`` before
    the next audio listens again; ``listening`` tells which state the session is in.
    A connection starts listening, and so does the new connection of a reconnect.
    ``close()`` on a stopped session sends listen_start, so the connection listens
    again for the one-sided methods and the next session.

    Wake word: ``await session.enable_wake_word()`` asks the server to listen for a
    wake phrase (its default list, or the list passed in), for a consumer with no
    keyword spotter of its own. The session is then dormant: stopped, but
    ``send_audio`` keeps sending for the server's spotter, and nothing is
    transcribed. When the server hears a phrase the session listens and ``wakes()``
    yields the wake frame. ``listen_stop`` makes it dormant again, ``listen_start``
    wakes it by hand, and ``dormant`` tells whether it waits for a phrase. A
    reconnect starts a new connection that has not opted in.
    """

    def __init__(self, client: "VoiceManagerClient") -> None:
        self._client = client
        self.closed = False

    async def __aenter__(self) -> "DuplexSession":
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def close(self) -> None:
        """End the session; the client's one-sided methods work again.

        Closing a stopped session (see ``listen_stop``) sends listen_start, so the
        connection listens again: audio from ``stream_stt``, ``stream_stt_live`` or a
        new session reaches STT and is counted as the server counts it. Closing a
        session that is listening sends nothing. If the connection is already gone
        the frame is not sent and nothing is raised; the client still counts as
        listening, as the new connection of a reconnect does. Closing a session
        again does nothing.
        """
        if self.closed:
            return
        client = self._client
        self.closed = True
        if client._duplex is self:
            client._duplex = None
        if not client._listen_stopped:
            return
        client._listen_stopped = False
        if client._ws is None:
            return
        try:
            await client._ws.send(json.dumps({"type": "listen_start"}))
        except websockets.exceptions.ConnectionClosed:
            # The next send finds the closed connection and reconnects, and a new
            # connection listens; closing the session does not reconnect itself.
            client._logger.debug("Connection gone; listen_start not sent on close")

    async def send_audio(self, pcm: bytes) -> None:
        """Send one binary frame of 16 kHz mono 16-bit PCM to STT.

        While the session is stopped (see ``listen_stop``) the frame is dropped
        here: it is not sent and not counted in ``mic_sample``, as the server
        would not count it either. A session that opted in to the wake word (see
        ``enable_wake_word``) is dormant while stopped: the frame is sent, for the
        server's spotter, and still not counted in ``mic_sample``.
        """
        client = self._client
        if client._listen_stopped:
            if client._wake_phrases is None:
                return
            # Dormant: the server gives the frame to its wake word spotter and
            # counts it as dormant audio, not as mic audio.
            dormant = True
        else:
            dormant = False
        # The frame is counted before the send is awaited. A send can wait for room
        # in the write buffer with the frame already on the wire, and the reader
        # (a wake frame) or another task (an opt-in, a listen_stop) that runs in
        # that wait must find the frame in the count, as the server will.
        ws = client._ws
        if dormant:
            client._dormant_bytes += len(pcm)
            counted = len(pcm)
        else:
            before = client._mic_bytes
            client._count_mic(pcm)
            counted = client._mic_bytes - before
        try:
            await client._send_or_raise(pcm, STTError)
        except STTError:
            # The connection was lost: the client reconnected, and the counts of
            # the new connection start at 0.
            raise
        except Exception:
            # Nothing was sent: there is no socket while another task reconnects,
            # or the socket refused the data. The frame comes out of the count
            # again, unless a reconnect started the counts over in the meantime.
            # A cancelled send is not caught here: its frame is on the wire.
            if client._ws is ws:
                if dormant:
                    client._dormant_bytes -= counted
                else:
                    client._mic_bytes -= counted
                    client._mic_frames -= 1
            raise

    @property
    def listening(self) -> bool:
        """True unless ``listen_stop`` stopped this connection and no ``listen_start`` followed."""
        return not self._client._listen_stopped

    async def listen_stop(self, timeout: float = 5.0) -> int | None:
        """Stop listening on this connection and return the server's mic sample count.

        The session is stopped first, so ``send_audio`` drops every frame from now
        on. Then the listen_stop frame is sent and the call waits for listen_stopped,
        whose ``mic_sample`` (the mic samples the server accepted on the connection)
        it returns. If that differs from ``self.mic_sample`` a warning names both
        counts. Calling it while already stopped still sends the frame; the server
        answers again with the same count.

        Returns None when no answer comes within ``timeout`` (a server older than
        0.8.0 drops the frame) or the connection closes first; the session stays
        stopped either way. If the send itself fails the client reconnects, and the
        new connection listens.

        Raises:
            VoiceManagerError: The session is closed (nothing is sent), or the
                connection was lost while sending the frame.
        """
        client = self._client
        if self.closed:
            raise VoiceManagerError("the duplex session is closed")
        if client._wake_phrases is not None:
            if client._listen_stopped:
                # Sent while dormant. If the server hears a phrase before this frame
                # reaches it, the frame stops a connection that had just woken; the
                # wake frame's count then tells which of the two came first.
                client._dormant_stop_marks.append(client._dormant_bytes)
            else:
                # The connection becomes dormant: a new count of dormant audio.
                client._dormant_bytes = 0
                client._dormant_stop_marks = []
        client._listen_stopped = True
        client._listen_stops_sent += 1
        number = client._listen_stops_sent
        waiter = asyncio.get_running_loop().create_future()
        # Registered before the send: the answer may land before the send returns.
        client._listen_waiters[number] = waiter
        try:
            await client._send_or_raise(
                json.dumps({"type": "listen_stop"}), VoiceManagerError)
            try:
                async with asyncio.timeout(timeout):
                    server_count = await waiter
            except TimeoutError:
                client._logger.debug("No listen_stopped within %ss", timeout)
                return None
        finally:
            if client._listen_waiters.get(number) is waiter:
                del client._listen_waiters[number]
        if server_count is not None and server_count != self.mic_sample:
            client._logger.warning(
                "listen_stopped mic_sample %s differs from the %d samples this "
                "session sent", server_count, self.mic_sample)
        return server_count

    async def listen_start(self) -> None:
        """Listen again on this connection: send listen_start and return at once.

        The server sends no answer. ``send_audio`` sends and counts frames again
        from now on. Calling it while already listening still sends the frame,
        which the server ignores.

        Raises:
            VoiceManagerError: The session is closed (nothing is sent), or the
                connection was lost while sending the frame.
        """
        client = self._client
        if self.closed:
            raise VoiceManagerError("the duplex session is closed")
        client._listen_stopped = False
        await client._send_or_raise(json.dumps({"type": "listen_start"}), VoiceManagerError)

    @property
    def wake_phrases(self) -> list[str] | None:
        """The wake phrases this connection listens for, or None with no opt-in."""
        phrases = self._client._wake_phrases
        return None if phrases is None else list(phrases)

    @property
    def dormant(self) -> bool:
        """True while the session is stopped and the server listens for a wake phrase."""
        return self._client._listen_stopped and self._client._wake_phrases is not None

    async def enable_wake_word(self, phrases: list[str] | None = None,
                               timeout: float = 10.0) -> list[str]:
        """Opt this connection in to the server's wake word and return its phrases.

        Sends the wake_word frame and waits for wake_word_ready. ``phrases`` is the
        list of wake phrases for this connection, as plain text ("hey hermes"); None
        takes the server's default list. The returned list is what the server
        listens for (lowercased, repeats left out).

        The session is dormant from the answer on: ``listening`` is False,
        ``send_audio`` keeps sending frames for the server's spotter without counting
        them in ``mic_sample``, and nothing is transcribed. When the server hears a
        phrase the session listens again and ``wakes()`` yields the wake frame.
        ``listen_stop`` makes the session dormant again, and ``listen_start`` wakes
        it by hand. Calling this again replaces the phrases. Wait for the answer
        before the next ``listen_stop`` or ``listen_start``.

        Raises:
            VoiceManagerError: The session is closed; nothing is sent.
            WakeWordError: The server refused the opt-in (``error_type`` holds its
                error_type), or no answer came within ``timeout`` or before the
                connection closed (``error_type`` is None). The session is then as
                it was before the call.
        """
        client = self._client
        if self.closed:
            raise VoiceManagerError("the duplex session is closed")
        frame: dict = {"type": "wake_word"}
        if phrases is not None:
            frame["phrases"] = list(phrases)
        waiter = asyncio.get_running_loop().create_future()
        client._wake_waiter = waiter
        # What the session was when the frame went out. The server handles frames in
        # order, so the audio sent after this point is dormant audio once it accepts.
        client._wake_sent = (client._listen_stopped, client._wake_phrases is not None,
                             client._mic_bytes, client._mic_frames, client._dormant_bytes)
        try:
            await client._send_or_raise(json.dumps(frame), WakeWordError)
            try:
                async with asyncio.timeout(timeout):
                    return await waiter
            except TimeoutError:
                raise WakeWordError(
                    "Timed out waiting for wake_word_ready (a server older than 0.9.0 "
                    "does not answer)") from None
        finally:
            if client._wake_waiter is waiter:
                client._wake_waiter = None

    async def wakes(self) -> AsyncIterator[dict]:
        """Yield every wake frame as it arrives (opted-in connections only).

        A frame has ``phrase``, the phrase the server heard, and ``mic_sample``, the
        server's mic sample count at the wake. The session is listening again by the
        time the frame is yielded.
        """
        while (frame := await self._client._wake_queue.get()) is not None:
            yield frame

    @property
    def mic_sample(self) -> int:
        """Samples of mic audio sent on this connection since its first audio frame.

        Read it when a playback edge reaches the speaker to get that edge's
        ``mic_sample``, the index on this outgoing 16 kHz mic stream.
        """
        return self._client._mic_bytes // 2

    @property
    def turn_engine(self) -> str | None:
        """The turn engine this connection opted in to, or None."""
        return self._client.turn_engine

    async def enable_turn_engine(self, engine: str = "tyger", timeout: float = 10.0) -> str:
        """Opt this connection in to a turn engine and return the engine the server named.

        Sends the turn_engine frame and waits for turn_engine_ready. It must come
        before the first audio frame, and a connection opts in once.

        Raises:
            VoiceManagerError: The session is closed, audio was already sent, or an
                opt-in was already sent on this connection; nothing is sent.
            TurnEngineError: The server refused the opt-in (``error_type`` holds its
                error_type), or no answer came within ``timeout`` or before the
                connection closed (``error_type`` is None).
        """
        client = self._client
        if self.closed:
            raise VoiceManagerError("the duplex session is closed")
        if client._mic_frames:
            raise VoiceManagerError(
                "enable_turn_engine must come before the first audio frame")
        if client._turn_engine_sent is not None:
            raise VoiceManagerError(
                "a turn_engine opt-in was already sent on this connection")
        waiter = asyncio.get_running_loop().create_future()
        client._turn_engine_waiter = waiter
        client._turn_engine_sent = engine
        try:
            await client._send_or_raise(
                json.dumps({"type": "turn_engine", "engine": engine}), TurnEngineError)
            try:
                async with asyncio.timeout(timeout):
                    return await waiter
            except TimeoutError:
                raise TurnEngineError("Timed out waiting for turn_engine_ready") from None
        finally:
            if client._turn_engine_waiter is waiter:
                client._turn_engine_waiter = None

    async def playback_start(self, request, mic_sample: int | None = None) -> None:
        """Report that a reply's playback started at the speaker.

        ``request`` is a TTSRequest or its request_id. ``mic_sample`` is where on
        the outgoing mic stream the first sample reached the speaker; it defaults
        to ``self.mic_sample`` now.
        """
        await self._playback(request, "start", mic_sample, None)

    async def playback_stop(self, request, mic_sample: int | None = None, *,
                            reply_sample: int) -> None:
        """Report that a reply's playback stopped at the speaker.

        ``reply_sample`` is how many of the reply's samples played. ``request`` and
        ``mic_sample`` are as in ``playback_start``.
        """
        await self._playback(request, "stop", mic_sample, reply_sample)

    async def _playback(self, request, event: str, mic_sample: int | None,
                        reply_sample: int | None) -> None:
        request_id = request.request_id if isinstance(request, TTSRequest) else request
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("a playback frame needs a non-empty request_id")
        if mic_sample is None:
            mic_sample = self.mic_sample
        _check_sample("mic_sample", mic_sample)
        payload = {"type": "playback", "request_id": request_id, "event": event,
                   "mic_sample": mic_sample}
        if event == "stop":
            _check_sample("reply_sample", reply_sample)
            payload["reply_sample"] = reply_sample
        await self._client._send_or_raise(json.dumps(payload), VoiceManagerError)

    async def turn_start(self, timeout: float = 5.0) -> int:
        """Open a conversation turn and return the server's turn_id."""
        return await self._client.turn_start(timeout=timeout)

    def tts_request(self, voice: str | None = None, chunking: str | None = None,
                    request_id: str | None = None) -> TTSRequest:
        """Make a streaming TTS request; nothing is sent until its first ``send``."""
        _check_chunking(chunking)
        request_id = request_id or str(uuid.uuid4())
        self._client.last_tts_request_id = request_id
        return TTSRequest(self._client, request_id, voice, chunking)

    async def transcripts(self) -> AsyncIterator[dict]:
        """Yield every transcription dict as it arrives, turn_id stamped."""
        while (frame := await self._client._stt_queue.get()) is not None:
            yield self._client._stamp_turn_id(frame)

    async def audio(self) -> AsyncIterator[tuple]:
        """Yield audio_start, pcm, barge and audio_end events in arrival order."""
        client = self._client
        current: str | None = None
        while (item := await client._tts_queue.get()) is not None:
            if isinstance(item, bytes):
                if current is not None and current not in client._cancelled_tts:
                    yield ("pcm", item, current)
                continue
            kind, rid = item.get("type"), item.get("request_id")
            if kind == "audio_start":
                current = rid
                client.last_tts_sample_rate = item.get("sample_rate", client.last_tts_sample_rate)
                yield ("audio_start", item)
            elif kind == "audio_end":
                client._cancelled_tts.discard(rid)
                if rid == current:
                    current = None
                yield ("audio_end", item)
            elif kind == "barge":
                # The server sends no more pcm for a barged request; drop any that
                # was already in flight, as for a cancel, until its audio_end.
                client._cancelled_tts.add(rid)
                yield ("barge", item)

    async def turn_ends(self) -> AsyncIterator[dict]:
        """Yield every turn_end frame as it arrives (opted-in connections only)."""
        while (frame := await self._client._turn_end_queue.get()) is not None:
            yield frame

    async def errors(self) -> AsyncIterator[dict]:
        """Yield the server's error frames (timeout warnings, invalid requests)."""
        while (frame := await self._client._error_queue.get()) is not None:
            yield frame


class VoiceManagerClient:
    """Async client for the VoiceManager STT/TTS server.

    Maintains a single persistent WebSocket connection per context manager
    scope.  All STT and TTS calls share the connection.

    Usage::

        async with VoiceManagerClient(token="secret") as client:
            transcriptions = await client.transcribe_file("input.wav")
            pcm, rate = await client.synthesize_tts("Hello!")
            client.save_tts_to_file(pcm, "output.wav", rate)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8255,
        token: str = "",
        max_retries: int = 3,
        use_ssl: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.token = token
        self.max_retries = max_retries
        self.use_ssl = use_ssl

        self._ws = None
        self._reader: asyncio.Task | None = None

        # Per-operation dispatch queues (single reader, multiple waiters)
        self._stt_queue: asyncio.Queue = asyncio.Queue()
        self._tts_queue: asyncio.Queue = asyncio.Queue()
        self._error_queue: asyncio.Queue = asyncio.Queue()
        # turn_started acks. turn_start() sends the control frame then
        # awaits its server-owned turn_id here, off the single reader.
        self._turn_queue: asyncio.Queue = asyncio.Queue()
        # turn_end frames, for a connection that opted in to a turn engine.
        self._turn_end_queue: asyncio.Queue = asyncio.Queue()

        # Turn engine state for this connection: the engine named in the opt-in
        # sent (at most one per connection), the waiter for its answer, and the
        # engine the server confirmed (None until turn_engine_ready, and again
        # after turn_engine_lost).
        self._turn_engine_sent: str | None = None
        self._turn_engine_waiter: asyncio.Future | None = None
        self.turn_engine: str | None = None
        # Bytes of mic audio sent on this connection, past a WAV header on the
        # first frame. Half of it is the mic_sample a playback frame reports.
        self._mic_bytes: int = 0
        # Binary mic frames sent on this connection; any of them bars a later opt-in.
        self._mic_frames: int = 0
        # Listening boundary state for this connection: whether a DuplexSession
        # stopped listening (send_audio then drops its frames), how many listen_stop
        # frames were sent and how many listen_stopped answers came in. The server
        # answers every listen_stop once and in order, so the Nth answer belongs to
        # the Nth frame; _listen_waiters maps a pending call's number to its future,
        # and an answer that finds no waiter (its call timed out) is dropped.
        self._listen_stopped: bool = False
        self._listen_stops_sent: int = 0
        self._listen_acks_seen: int = 0
        self._listen_waiters: dict[int, asyncio.Future] = {}
        # Wake word state for this connection. _wake_phrases is the list the server
        # confirmed (None until wake_word_ready); an opted-in connection is dormant
        # while it is stopped. _wake_sent is what the session was when the pending
        # opt-in went out, and _wake_waiter the future of the call that waits for
        # its answer. _dormant_bytes counts the dormant audio sent since the
        # connection last became dormant, as the server counts it for the wake
        # frame's dormant_sample; _dormant_stop_marks holds that count at each
        # listen_stop sent while dormant.
        self._wake_queue: asyncio.Queue = asyncio.Queue()
        self._wake_phrases: list[str] | None = None
        self._wake_sent: tuple | None = None
        self._wake_waiter: asyncio.Future | None = None
        self._dormant_bytes: int = 0
        self._dormant_stop_marks: list[int] = []

        self.session_id: str | None = None
        self._audio_format: dict = {}
        self.stt_engine: str = "whisper"
        self.tts_engine: str = "openvoice"
        self.available_voices: list[str] = []
        # The voice a TTS request without 'voice' gets on this connection (the
        # token's voice, else the server default); None when the engine has no presets.
        self.default_voice: str | None = None
        self.connected: bool = False
        self.reconnected: bool = False  # consumer checks this after reconnection
        self.last_tts_sample_rate: int = 22050  # Updated by audio_start in stream_tts_live()
        # request_id of the most recently started TTS request. Exposed
        # so a caller can reach the id of the request it just started and call
        # cancel_tts() on it - barge-in is impossible if the id stays internal.
        self.last_tts_request_id: str | None = None
        # request_ids the caller has cancelled. cancel_tts() adds one here
        # and returns immediately without awaiting audio_end; the live-yield path
        # discards binary frames for a cancelled request until its audio_end lands.
        self._cancelled_tts: set[str] = set()
        # The open DuplexSession, if any; the one-sided methods refuse while it is set.
        self._duplex: DuplexSession | None = None

        self._logger = logger

    def _auth_headers(self) -> dict[str, str]:
        """Return the Authorization header for the WebSocket handshake and REST calls.

        The token is sent only as ``Authorization: Bearer <token>``; it never
        appears in a URL or query string.
        """
        return {"Authorization": f"Bearer {self.token}"}

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "VoiceManagerClient":
        await self._connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self._disconnect()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def _connect(self) -> None:
        """Open WebSocket, read handshake, start reader task."""
        ws_scheme = "wss" if self.use_ssl else "ws"
        # The token travels only in the Authorization header, never in the URL.
        url = f"{ws_scheme}://{self.host}:{self.port}/ws/stream"

        try:
            self._ws = await websockets.asyncio.client.connect(
                url, additional_headers=self._auth_headers()
            )
        except websockets.exceptions.InvalidStatus as exc:
            if exc.response.status_code in (401, 403):
                raise AuthenticationError("Server rejected the token") from exc
            raise ConnectionError(
                f"Server rejected connection: HTTP {exc.response.status_code}"
            ) from exc

        # Read handshake: {"type": "ready", "session_id": "...", "audio_format": {...}}
        raw = await self._ws.recv()
        msg = json.loads(raw)
        if msg.get("type") != "ready":
            await self._ws.close()
            raise ConnectionError(
                f"Unexpected first frame type: {msg.get('type')!r}"
            )

        self.session_id = msg["session_id"]
        self._audio_format = msg.get("audio_format", {})
        self.stt_engine = msg.get("stt_engine", "whisper")
        self.tts_engine = msg.get("tts_engine", "openvoice")
        self.available_voices = msg.get("available_voices", [])
        self.default_voice = msg.get("default_voice")

        # Start the single reader task (mandatory)
        self._reader = asyncio.create_task(self._reader_loop())
        self.connected = True
        self._logger.info("Connected session=%s", self.session_id)

    async def _disconnect(self) -> None:
        """Cancel reader task, close WebSocket, drain queues."""
        if self._reader is not None:
            self._reader.cancel()
            try:
                await self._reader
            except asyncio.CancelledError:
                pass

        if self._ws is not None:
            await self._ws.close()

        self.connected = False
        self._ws = None
        self._reader = None

        # Drain all queues - put sentinel None to unblock any waiters
        for q in (self._stt_queue, self._tts_queue, self._error_queue, self._turn_queue,
                  self._turn_end_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            await q.put(None)

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff.

        Raises ConnectionError after max_retries attempts.
        Sets self.reconnected = True on success - consumer must re-upload
        their voice profile.
        """
        await self._disconnect()

        # Reset queues for the new connection
        self._stt_queue = asyncio.Queue()
        self._tts_queue = asyncio.Queue()
        self._error_queue = asyncio.Queue()
        self._turn_queue = asyncio.Queue()
        self._turn_end_queue = asyncio.Queue()
        self._wake_queue = asyncio.Queue()
        # A fresh connection carries no pending cancellations, has not opted in to
        # a turn engine or the wake word, has sent no mic audio, and listens.
        self._wake_phrases = None
        self._wake_sent = None
        self._dormant_bytes = 0
        self._dormant_stop_marks = []
        self._cancelled_tts = set()
        self._turn_engine_sent = None
        self.turn_engine = None
        self._mic_bytes = 0
        self._mic_frames = 0
        self._listen_stopped = False
        self._listen_stops_sent = 0
        self._listen_acks_seen = 0
        self._listen_waiters = {}

        delays = [1, 2, 4]
        for attempt, delay in enumerate(delays):
            try:
                await self._connect()
                self.reconnected = True
                self._logger.warning(
                    "Reconnected - voice profile must be re-uploaded (session=%s)",
                    self.session_id,
                )
                return
            except Exception:
                if attempt == len(delays) - 1:
                    raise ConnectionError(
                        f"Failed to reconnect after {self.max_retries} attempts"
                    ) from None
                self._logger.warning(
                    "Reconnect attempt %d/%d failed, retrying in %ds",
                    attempt + 1,
                    len(delays),
                    delay,
                )
                await asyncio.sleep(delay)

    # ------------------------------------------------------------------
    # Reader task (mandatory single reader)
    # ------------------------------------------------------------------

    async def _reader_loop(self) -> None:
        """Background task - dispatches all received WebSocket frames to queues.

        This is the ONLY place that calls recv().  Callers read from asyncio.Queue
        instances to avoid ConcurrencyError.
        """
        try:
            async for msg in self._ws:
                if isinstance(msg, bytes):
                    # TTS audio binary frame
                    await self._tts_queue.put(msg)
                else:
                    frame = json.loads(msg)
                    ftype = frame.get("type")
                    if ftype == "transcription":
                        await self._stt_queue.put(frame)
                    elif ftype in ("audio_start", "audio_end"):
                        await self._tts_queue.put(frame)
                    elif ftype == "turn_started":
                        # Server-owned ack for a turn_start - routed to
                        # its own queue so turn_start() awaits it independently.
                        await self._turn_queue.put(frame)
                    elif ftype == "turn_end":
                        await self._turn_end_queue.put(frame)
                    elif ftype == "barge":
                        # In the audio stream, so it lands in order with the pcm
                        # and before the request's audio_end.
                        await self._tts_queue.put(frame)
                    elif ftype == "turn_engine_ready":
                        self._on_turn_engine_ready(frame)
                    elif ftype == "listen_stopped":
                        self._on_listen_stopped(frame)
                    elif ftype == "wake_word_ready":
                        self._on_wake_word_ready(frame)
                    elif ftype == "wake":
                        if self._on_wake(frame):
                            await self._wake_queue.put(frame)
                    elif ftype == "error":
                        error_type = frame.get("error_type", "")
                        if error_type == "timeout_warning":
                            self._logger.warning(
                                "Server timeout warning: %s", frame.get("message", "")
                            )
                        if (error_type == "invalid_request" and self._wake_sent is not None
                                and frame.get("message", "").startswith(_WAKE_REFUSAL)):
                            # The server's answer to the pending wake_word opt-in: it
                            # goes to enable_wake_word, not to errors(). The session
                            # is as it was before the opt-in.
                            self._wake_sent = None
                            wake_waiter = self._wake_waiter
                            if wake_waiter is not None and not wake_waiter.done():
                                wake_waiter.set_exception(WakeWordError(
                                    frame.get("message", "") or error_type, error_type))
                            continue
                        waiter = self._turn_engine_waiter
                        if (error_type in _TURN_ENGINE_REFUSALS and waiter is not None
                                and not waiter.done()):
                            # The server's answer to the pending opt-in: it goes to
                            # enable_turn_engine, not to errors().
                            waiter.set_exception(TurnEngineError(
                                frame.get("message", "") or error_type, error_type))
                            continue
                        if error_type == _TURN_ENGINE_LOST:
                            self._logger.warning(
                                "Turn engine lost: %s", frame.get("message", ""))
                            self.turn_engine = None
                        await self._error_queue.put(frame)
                    else:
                        self._logger.debug("Unhandled message type: %s", ftype)
        except websockets.exceptions.ConnectionClosed:
            self._logger.info("WebSocket connection closed")
            self.connected = False
        finally:
            # Signal all waiters that the connection is gone
            waiter = self._turn_engine_waiter
            if waiter is not None and not waiter.done():
                waiter.set_exception(
                    TurnEngineError("Connection lost waiting for turn_engine_ready"))
            for listen_waiter in self._listen_waiters.values():
                if not listen_waiter.done():
                    listen_waiter.set_result(None)
            wake_waiter = self._wake_waiter
            if wake_waiter is not None and not wake_waiter.done():
                wake_waiter.set_exception(
                    WakeWordError("Connection lost waiting for wake_word_ready"))
            await self._wake_queue.put(None)
            await self._stt_queue.put(None)
            await self._tts_queue.put(None)
            await self._turn_queue.put(None)
            await self._turn_end_queue.put(None)
            await self._error_queue.put(None)

    def _on_turn_engine_ready(self, frame: dict) -> None:
        """Record the server's turn_engine_ready and wake a pending opt-in."""
        if self._turn_engine_sent is None:
            self._logger.debug("Ignoring a turn_engine_ready that answers no opt-in")
            return
        engine = frame.get("engine", self._turn_engine_sent)
        # A ready that lands after the opt-in timed out still means the server
        # opted the connection in.
        self.turn_engine = engine
        waiter = self._turn_engine_waiter
        if waiter is not None and not waiter.done():
            waiter.set_result(engine)

    def _on_listen_stopped(self, frame: dict) -> None:
        """Hand a listen_stopped answer to the listen_stop call it answers, if it waits."""
        self._listen_acks_seen += 1
        waiter = self._listen_waiters.get(self._listen_acks_seen)
        if waiter is None or waiter.done():
            # The answer to a listen_stop whose call already timed out.
            self._logger.debug("Ignoring a listen_stopped that answers no waiting call")
            return
        waiter.set_result(frame.get("mic_sample"))

    def _on_wake_word_ready(self, frame: dict) -> None:
        """Make the connection dormant as the server did, and answer a pending opt-in.

        The server made the connection dormant when it handled the wake_word frame,
        so every frame sent after that frame was dormant audio. What that audio was
        counted as here depends on what the session was when the frame went out.
        """
        sent = self._wake_sent
        if sent is None:
            self._logger.debug("Ignoring a wake_word_ready that answers no opt-in")
            return
        self._wake_sent = None
        was_stopped, was_opted_in, mic_bytes, mic_frames, dormant_bytes = sent
        if not was_stopped:
            # Listening: the audio sent since then was counted as mic audio.
            self._dormant_bytes = self._mic_bytes - mic_bytes
            self._mic_bytes, self._mic_frames = mic_bytes, mic_frames
            self._dormant_stop_marks = []
        elif was_opted_in:
            # Dormant already: the server started a new count of dormant audio.
            self._dormant_bytes -= dormant_bytes
            self._dormant_stop_marks = [
                mark - dormant_bytes for mark in self._dormant_stop_marks
                if mark >= dormant_bytes]
        else:
            # Stopped with no opt-in: its audio was dropped here and never sent.
            self._dormant_bytes = 0
            self._dormant_stop_marks = []
        self._listen_stopped = True
        # A ready that lands after the opt-in timed out still means the server
        # opted the connection in.
        self._wake_phrases = list(frame.get("phrases") or [])
        waiter = self._wake_waiter
        if waiter is not None and not waiter.done():
            waiter.set_result(list(self._wake_phrases))

    def _on_wake(self, frame: dict) -> bool:
        """Follow the server's wake; True when the frame is one for the application.

        The server took ``dormant_sample`` samples as dormant audio in this dormant
        period; the audio sent after them reached a connection that was listening,
        and the server counted it as mic audio, so it is counted here too.

        A listen_stop that was sent while dormant, and that reached the server after
        it heard the phrase, stopped the connection again right away. The server
        counted the audio between the two as mic audio and then became dormant on a
        new count. The session stays stopped then and the wake is not reported.
        """
        if self._wake_phrases is None:
            self._logger.debug("Ignoring a wake frame on a connection with no opt-in")
            return False
        taken = 2 * int(frame.get("dormant_sample") or 0)
        crossed = [mark for mark in self._dormant_stop_marks if mark >= taken]
        if crossed:
            mark = min(crossed)
            self._add_mic(mark - taken)
            self._dormant_bytes -= mark
            self._dormant_stop_marks = [m - mark for m in self._dormant_stop_marks if m > mark]
            return False
        self._add_mic(self._dormant_bytes - taken)
        self._dormant_bytes = taken
        self._dormant_stop_marks = []
        self._listen_stopped = False
        return True

    def _add_mic(self, count: int) -> None:
        """Count ``count`` bytes of sent audio as mic audio the server accepted."""
        if count > 0:
            self._mic_bytes += count
            self._mic_frames += 1

    # ------------------------------------------------------------------
    # Full duplex
    # ------------------------------------------------------------------

    def duplex(self) -> DuplexSession:
        """Open a DuplexSession: STT and TTS at the same time on this connection.

        Use it as ``async with client.duplex() as session:`` or call ``close()``
        when done. Only one session is open at a time, and while it is open the
        one-sided methods (``stream_stt_live``, ``stream_tts_live``, ``stream_stt``,
        ``stream_tts``, ``synthesize_tts``) raise VoiceManagerError.
        """
        if self._duplex is not None and not self._duplex.closed:
            raise VoiceManagerError("a duplex session is already open on this client")
        self._duplex = DuplexSession(self)
        return self._duplex

    def _require_no_duplex(self, method: str) -> None:
        if self._duplex is not None and not self._duplex.closed:
            raise VoiceManagerError(
                f"{method} cannot run while a duplex session is open; use the session")

    async def _send_or_raise(self, data, error: type) -> None:
        """Send one frame; a closed connection reconnects and raises ``error``."""
        try:
            await self._ws.send(data)
        except websockets.exceptions.ConnectionClosed:
            await self._reconnect()
            raise error("Connection lost during duplex session")

    def _count_mic(self, data: bytes) -> None:
        """Count one sent binary mic frame the way the server counts it.

        The server skips a WAV header on the connection's first frame, so the
        client does too, or mic_sample would be off by the header.
        """
        if self._mic_frames == 0:
            offset = _wav_payload_offset(data)
            if offset is not None:
                data = data[offset:]
        self._mic_frames += 1
        self._mic_bytes += len(data)

    # ------------------------------------------------------------------
    # STT methods
    # ------------------------------------------------------------------

    @staticmethod
    def _stamp_turn_id(frame: dict) -> dict:
        """Guarantee a transcription dict carries a turn_id key.

        The server stamps turn_id on every transcription frame (default 0), so
        this is normally a no-op - but the client's public contract is that every
        dict it yields/returns has the key (0 when no turn was ever opened), so a
        Python consumer can filter by turn without a KeyError guard.
        """
        frame.setdefault("turn_id", 0)
        return frame

    async def stream_stt(self, audio_chunks: AsyncIterator[bytes]) -> list[dict]:
        """Send audio chunks for STT; return all transcription result dicts.

        Sends all chunks to the server, then waits for transcriptions triggered
        by VAD.  Returns a list of transcription dicts with keys:
        text, is_final, endpoint, segment_index, language, confidence, start, end.

        ``endpoint`` is the speculative-endpoint flag: true only on a
        provisional end-of-utterance frame (is_final=False), false on ordinary
        interim and confirmed-final frames. Only the nemotron STT path ever sets
        it true; other engines always carry endpoint=false.

        Args:
            audio_chunks: Async iterator of raw PCM bytes (16-bit, 16 kHz, mono).

        Returns:
            List of transcription dicts from the server.

        Raises:
            STTError: If the connection is lost during streaming.
        """
        self._require_no_duplex("stream_stt")
        results: list[dict] = []
        try:
            async for chunk in audio_chunks:
                await self._ws.send(chunk)
                self._count_mic(chunk)
                # Drain any transcriptions that arrived while sending
                while not self._stt_queue.empty():
                    item = self._stt_queue.get_nowait()
                    if item is None:
                        return results
                    results.append(self._stamp_turn_id(item))
        except websockets.exceptions.ConnectionClosed:
            await self._reconnect()
            raise STTError("Connection lost during STT streaming")

        # Wait for remaining transcriptions (VAD may still be processing)
        try:
            async with asyncio.timeout(5.0):
                while True:
                    item = await self._stt_queue.get()
                    if item is None:
                        break
                    results.append(self._stamp_turn_id(item))
                    # Check briefly for additional results
                    try:
                        async with asyncio.timeout(2.0):
                            continue
                    except TimeoutError:
                        break
        except TimeoutError:
            pass

        return results

    async def stream_stt_live(
        self, audio_chunks: AsyncIterator[bytes]
    ) -> AsyncIterator[dict]:
        """Send audio and yield transcription dicts as they arrive.

        Unlike stream_stt() which collects all results, this yields each
        transcription dict immediately - both interim (is_final=False) and
        final (is_final=True) - so consumers can react to partials in
        real-time.

        Args:
            audio_chunks: Async iterator of raw PCM bytes (16-bit, 16 kHz, mono).

        Yields:
            Transcription dicts with keys: text, is_final, endpoint,
            segment_index, language, confidence, start, end.

            ``endpoint`` is the speculative-endpoint flag: true only on
            a provisional end-of-utterance frame (is_final=False), false on
            ordinary interim and confirmed-final frames. Only the nemotron STT
            path ever sets it true; other engines always carry endpoint=false.

        Raises:
            STTError: If the connection is lost during streaming.
        """
        self._require_no_duplex("stream_stt_live")
        send_done = False
        try:
            send_iter = audio_chunks.__aiter__()
            while not send_done:
                # Send next audio chunk
                try:
                    chunk = await send_iter.__anext__()
                    await self._ws.send(chunk)
                    self._count_mic(chunk)
                except StopAsyncIteration:
                    send_done = True
                except websockets.exceptions.ConnectionClosed:
                    await self._reconnect()
                    raise STTError("Connection lost during STT streaming")

                # Drain any transcriptions that arrived
                while not self._stt_queue.empty():
                    item = self._stt_queue.get_nowait()
                    if item is None:
                        return
                    yield self._stamp_turn_id(item)

            # Audio exhausted - wait for remaining transcriptions
            try:
                async with asyncio.timeout(5.0):
                    while True:
                        item = await self._stt_queue.get()
                        if item is None:
                            return
                        yield self._stamp_turn_id(item)
                        try:
                            async with asyncio.timeout(2.0):
                                continue
                        except TimeoutError:
                            return
            except TimeoutError:
                pass
        except websockets.exceptions.ConnectionClosed:
            await self._reconnect()
            raise STTError("Connection lost during STT streaming")

    async def turn_start(self, timeout: float = 5.0) -> int:
        """Open a new conversation turn and return the server-assigned turn_id.

        Sends the bare ``turn_start`` control frame and waits for the server's
        ``turn_started`` ack, returning the newly opened turn's id. The server
        OWNS the turn counter: ``turn_start`` increments it and the ack
        reports the new id, so a consumer learns which turn_id to filter on rather
        than inferring it from the next stamped transcription frame.

        Opening a turn also clears the STT live partial server-side, so
        words spoken before this call cannot stitch into the new turn's first
        transcription.

        Args:
            timeout: Seconds to wait for the ``turn_started`` ack (default 5.0).

        Returns:
            The newly opened turn's id - a positive, monotonically increasing int.

        Raises:
            STTError: If the connection is lost or the ack does not arrive in time.
        """
        try:
            await self._ws.send(json.dumps({"type": "turn_start"}))
        except websockets.exceptions.ConnectionClosed:
            await self._reconnect()
            raise STTError("Connection lost sending turn_start")

        try:
            async with asyncio.timeout(timeout):
                ack = await self._turn_queue.get()
        except TimeoutError:
            raise STTError("Timed out waiting for turn_started ack")

        if ack is None:
            raise STTError("Connection lost waiting for turn_started ack")
        return ack.get("turn_id", 0)

    # ------------------------------------------------------------------
    # TTS methods
    # ------------------------------------------------------------------

    async def synthesize_tts(self, text: str, voice: str | None = None) -> tuple[bytes, int]:
        """Request TTS synthesis and collect all audio as PCM bytes.

        Args:
            text: Text to synthesize.
            voice: Voice preset name (VibeVoice only, e.g. "Carter", "Emma").
                Ignored when the server is using OpenVoice.

        Returns:
            Tuple of (pcm_bytes, sample_rate).  Use sample_rate when writing
            WAV files - TTS output rate depends on engine (22050 Hz for
            OpenVoice, 24000 Hz for VibeVoice).

        Raises:
            TTSError: If the connection is lost or times out.
        """
        self._require_no_duplex("synthesize_tts")
        request_id = str(uuid.uuid4())
        self.last_tts_request_id = request_id  # reachable so a caller can cancel_tts()
        payload = {"type": "tts", "text": text, "request_id": request_id}
        if voice is not None:
            payload["voice"] = voice
        msg = json.dumps(payload)
        await self._ws.send(msg)

        audio_buf = bytearray()
        sample_rate = 22050  # fallback; overwritten by audio_start

        try:
            async with asyncio.timeout(60.0):
                while True:
                    item = await self._tts_queue.get()
                    if item is None:
                        raise TTSError("Connection lost during TTS synthesis")

                    if isinstance(item, bytes):
                        audio_buf.extend(item)

                    elif isinstance(item, dict):
                        ftype = item.get("type")
                        item_rid = item.get("request_id")

                        if ftype == "audio_start":
                            if item_rid == request_id:
                                sample_rate = item.get("sample_rate", 22050)
                            # else: stale audio_start from cancelled request - discard

                        elif ftype == "audio_end":
                            if item_rid == request_id:
                                break
                            # else: stale audio_end - discard

        except TimeoutError:
            raise TTSError("Timed out waiting for TTS synthesis to complete")

        return bytes(audio_buf), sample_rate

    async def stream_tts(
        self,
        text_chunks: AsyncIterator[str],
        voice: str | None = None,
        chunking: str | None = None,
    ) -> tuple[bytes, int]:
        """Stream text chunks for TTS and collect all audio as PCM bytes.

        Sends text as tts_stream messages. The server synthesizes sentences
        progressively as complete sentences are detected. Audio is collected
        until audio_end signals completion.

        Primary use case: pipe LLM output directly into TTS synthesis.

        Args:
            text_chunks: Async iterator of text strings (e.g., LLM token stream).
            voice: Voice preset name (VibeVoice only). Sent on the first
                tts_stream message; the server fixes voice at request start.
            chunking: Text cut mode, "sentence" or "clause". Sent on the first
                tts_stream message; the server fixes it at request start. When
                omitted, no chunking field is sent and the server default applies.
                Any other value raises ValueError before anything is sent.

        Returns:
            Tuple of (pcm_bytes, sample_rate). Use sample_rate when writing
            WAV files - TTS output rate depends on engine.

        Raises:
            TTSError: If the connection is lost or times out.
            ValueError: If chunking is not None, "sentence" or "clause".
        """
        self._require_no_duplex("stream_tts")
        _check_chunking(chunking)
        request_id = str(uuid.uuid4())
        self.last_tts_request_id = request_id  # reachable so a caller can cancel_tts()
        audio_buf = bytearray()
        sample_rate = 22050  # fallback; overwritten by audio_start

        # Send all text chunks as tts_stream messages (done=True on final)
        first_chunk = True
        try:
            async for chunk in text_chunks:
                payload = {
                    "type": "tts_stream",
                    "request_id": request_id,
                    "text": chunk,
                    "done": False,
                }
                if first_chunk:
                    if voice is not None:
                        payload["voice"] = voice
                    if chunking is not None:
                        payload["chunking"] = chunking
                    first_chunk = False
                msg = json.dumps(payload)
                await self._ws.send(msg)
        except websockets.exceptions.ConnectionClosed:
            await self._reconnect()
            raise TTSError("Connection lost during streaming TTS")

        # Send done=True to flush server buffer
        try:
            await self._ws.send(json.dumps({
                "type": "tts_stream",
                "request_id": request_id,
                "text": "",
                "done": True,
            }))
        except websockets.exceptions.ConnectionClosed:
            await self._reconnect()
            raise TTSError("Connection lost sending TTS stream done signal")

        # Collect audio until audio_end (same pattern as synthesize_tts: filter by request_id)
        try:
            async with asyncio.timeout(120.0):
                while True:
                    item = await self._tts_queue.get()
                    if item is None:
                        raise TTSError("Connection lost during streaming TTS")

                    if isinstance(item, bytes):
                        audio_buf.extend(item)

                    elif isinstance(item, dict):
                        ftype = item.get("type")
                        item_rid = item.get("request_id")

                        if ftype == "audio_start":
                            if item_rid == request_id:
                                sample_rate = item.get("sample_rate", 22050)
                            # else: stale audio_start - discard

                        elif ftype == "audio_end":
                            if item_rid == request_id:
                                break
                            # else: stale audio_end - discard

        except TimeoutError:
            raise TTSError("Timed out waiting for streaming TTS to complete")

        self.last_tts_sample_rate = sample_rate
        return bytes(audio_buf), sample_rate

    async def stream_tts_live(
        self,
        text_chunks: AsyncIterator[str],
        voice: str | None = None,
        request_id: str | None = None,
        chunking: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream text for TTS and yield audio bytes in real-time.

        Unlike stream_tts() which collects all audio, this yields each PCM
        audio chunk as it arrives from the server - enabling real-time playback
        while text is still being sent.

        Sample rate is available via client.last_tts_sample_rate after the
        first audio chunk is yielded (updated when audio_start arrives).

        Primary use case: real-time LLM-to-speech with immediate playback.

        Barge-in: to cancel this request mid-stream, you need its
        request_id. Either supply one via the ``request_id`` argument (generate
        it yourself, e.g. ``str(uuid.uuid4())``, so you hold it before iteration
        begins), or read ``client.last_tts_request_id`` once iteration has
        started. Pass that id to ``cancel_tts()`` - typically from a separate
        task, since this generator is what you are iterating. Cancellation ends
        the iteration once the server's audio_end for the id arrives.

        Args:
            text_chunks: Async iterator of text strings (e.g., LLM token stream).
            voice: Voice preset name (VibeVoice only). Sent on the first
                tts_stream message; the server fixes voice at request start.
            request_id: Optional caller-supplied request id for this request. When
                omitted (default), one is generated internally and exposed via
                ``client.last_tts_request_id``. Supplying one lets the caller hold
                the id up front so it can call ``cancel_tts()`` on the in-flight
                request without waiting for iteration to start.
            chunking: Text cut mode, "sentence" or "clause". Sent on the first
                tts_stream message; the server fixes it at request start. When
                omitted, no chunking field is sent and the server default applies.
                Any other value raises ValueError before anything is sent.

        Yields:
            Raw PCM bytes per audio chunk from the server.

        Raises:
            TTSError: If the connection is lost or times out.
            ValueError: If chunking is not None, "sentence" or "clause".
        """
        self._require_no_duplex("stream_tts_live")
        _check_chunking(chunking)
        if request_id is None:
            request_id = str(uuid.uuid4())
        # Expose the id so the caller can reach the request it just started.
        self.last_tts_request_id = request_id
        send_done = False
        audio_started = False
        first_chunk = True

        try:
            send_iter = text_chunks.__aiter__()

            while not send_done:
                # Send next text chunk
                try:
                    chunk = await send_iter.__anext__()
                    payload = {
                        "type": "tts_stream",
                        "request_id": request_id,
                        "text": chunk,
                        "done": False,
                    }
                    if first_chunk:
                        if voice is not None:
                            payload["voice"] = voice
                        if chunking is not None:
                            payload["chunking"] = chunking
                        first_chunk = False
                    msg = json.dumps(payload)
                    await self._ws.send(msg)
                except StopAsyncIteration:
                    # All text sent - send done=True
                    await self._ws.send(json.dumps({
                        "type": "tts_stream",
                        "request_id": request_id,
                        "text": "",
                        "done": True,
                    }))
                    send_done = True
                except websockets.exceptions.ConnectionClosed:
                    await self._reconnect()
                    raise TTSError("Connection lost during streaming TTS")

                # Drain any audio that arrived while sending
                while not self._tts_queue.empty():
                    item = self._tts_queue.get_nowait()
                    if item is None:
                        raise TTSError("Connection lost during streaming TTS")
                    if isinstance(item, bytes):
                        # Discard PCM for a cancelled request until its audio_end.
                        if audio_started and request_id not in self._cancelled_tts:
                            yield item
                    elif isinstance(item, dict):
                        ftype = item.get("type")
                        item_rid = item.get("request_id")
                        if ftype == "audio_start" and item_rid == request_id:
                            self.last_tts_sample_rate = item.get("sample_rate", 22050)
                            audio_started = True
                        elif ftype == "audio_end" and item_rid == request_id:
                            self._cancelled_tts.discard(request_id)
                            return

            # Text exhausted - collect remaining audio until audio_end
            try:
                async with asyncio.timeout(120.0):
                    while True:
                        item = await self._tts_queue.get()
                        if item is None:
                            raise TTSError("Connection lost during streaming TTS")
                        if isinstance(item, bytes):
                            # Discard PCM for a cancelled request until its audio_end.
                            if audio_started and request_id not in self._cancelled_tts:
                                yield item
                        elif isinstance(item, dict):
                            ftype = item.get("type")
                            item_rid = item.get("request_id")
                            if ftype == "audio_start" and item_rid == request_id:
                                self.last_tts_sample_rate = item.get("sample_rate", 22050)
                                audio_started = True
                            elif ftype == "audio_end" and item_rid == request_id:
                                self._cancelled_tts.discard(request_id)
                                return
            except TimeoutError:
                raise TTSError("Timed out waiting for streaming TTS audio")

        except websockets.exceptions.ConnectionClosed:
            await self._reconnect()
            raise TTSError("Connection lost during streaming TTS")

    async def cancel_tts(self, request_id: str) -> None:
        """Cancel an in-progress TTS synthesis request.

        Returns as soon as the tts_cancel frame is sent - it does NOT wait for
        the server's audio_end, which may lag while the executor unwinds a long
        synthesis. Any binary frames still in flight for the cancelled request
        are discarded by the live-yield path (stream_tts_live) until that
        request's own audio_end arrives and ends the iteration.

        Args:
            request_id: The request_id of the in-flight TTS request - either the
                one supplied to stream_tts_live(), or read from
                client.last_tts_request_id after starting any TTS request.
        """
        # Mark first, then send: once marked, the live-yield path drops binary
        # frames for this id even if the cancel frame and the next PCM chunk race.
        self._cancelled_tts.add(request_id)
        msg = json.dumps({"type": "tts_cancel", "request_id": request_id})
        await self._send_or_raise(msg, TTSError)

    async def flush_tts(self, request_id: str) -> None:
        """Flush the server's buffered text for an in-flight streaming request.

        Puts exactly ONE tts_stream frame with flush=true and done=false on the
        wire for the given request_id. The server synthesizes everything it has
        buffered for that request so far - exactly as-is - and leaves the request
        OPEN for more text: no end-of-sentence sentinel is queued and no audio_end
        is emitted (that is what done=true does). This is what lets a one-sentence
        reply be spoken before the caller ever sends done=true.

        WARNING - flush only at a boundary YOU have detected to be the end of a
        sentence. Flushing MID-sentence hands the server a partial sentence and
        will audibly CUT that sentence in HALF: the buffered fragment is spoken on
        its own and the remainder arrives as a separate, disjoint utterance. The
        correct use is a sentence boundary the client itself detected (e.g. after a
        '.', '?' or '!' emerges from an LLM token stream) - never once per token.

        Args:
            request_id: The request_id of the in-flight streaming TTS request -
                either the one supplied to stream_tts_live(), or read from
                client.last_tts_request_id after starting a streaming request.
        """
        msg = json.dumps({
            "type": "tts_stream",
            "request_id": request_id,
            "text": "",
            "flush": True,
            "done": False,
        })
        await self._send_or_raise(msg, TTSError)

    # ------------------------------------------------------------------
    # Voice upload (REST)
    # ------------------------------------------------------------------

    async def upload_voice(
        self, audio_data: bytes, filename: str = "voice.wav"
    ) -> dict:
        """Upload a voice sample for cloning via REST.

        After upload, subsequent TTS synthesis on this session will use the
        cloned voice.  Re-upload is required after reconnection.

        Args:
            audio_data: WAV file bytes of the voice sample.
            filename: Filename hint (used for content-type detection server-side).

        Returns:
            Server response dict, e.g. ``{"status": "voice profile updated"}``.

        Raises:
            AuthenticationError: If the server rejects the token.
            VoiceManagerError: If the session is not found or server error.
        """
        http_scheme = "https" if self.use_ssl else "http"
        url = f"{http_scheme}://{self.host}:{self.port}/voice/upload"

        async with httpx.AsyncClient(timeout=120.0, headers=self._auth_headers()) as http:
            response = await http.post(
                url,
                params={"session_id": self.session_id},
                files={"file": (filename, audio_data)},
            )

        if response.status_code in (401, 403):
            raise AuthenticationError("Server rejected the token")
        if response.status_code == 404:
            raise VoiceManagerError("Session not found on server")
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise VoiceManagerError(f"Voice upload failed ({response.status_code}): {detail}")

        # Voice profile re-established - clear the reconnected flag
        self.reconnected = False
        return response.json()

    # ------------------------------------------------------------------
    # REST helpers
    # ------------------------------------------------------------------

    async def health(self) -> dict:
        """GET /health - model readiness status.

        Returns:
            ``{"status": "ok", "models": {"stt": ..., "vad": ..., "tts": ...}}``
        """
        http_scheme = "https" if self.use_ssl else "http"
        async with httpx.AsyncClient(headers=self._auth_headers()) as http:
            response = await http.get(
                f"{http_scheme}://{self.host}:{self.port}/health"
            )
        response.raise_for_status()
        return response.json()

    async def info(self) -> dict:
        """GET /info - loaded models, device, and active sessions.

        Returns:
            Dict with keys: models, device, whisper_model, tts_sample_rate, sessions.

        Raises:
            AuthenticationError: If the server rejects the token.
        """
        http_scheme = "https" if self.use_ssl else "http"
        async with httpx.AsyncClient(headers=self._auth_headers()) as http:
            response = await http.get(
                f"{http_scheme}://{self.host}:{self.port}/info"
            )
        if response.status_code in (401, 403):
            raise AuthenticationError("Server rejected the token")
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    async def transcribe_file(
        self, path: str, chunk_frames: int = 4000
    ) -> list[dict]:
        """Transcribe a WAV file using STT streaming.

        Reads the file as raw PCM frames (no WAV header) and
        streams them to the server in chunks.

        Args:
            path: Path to a 16-bit mono 16 kHz WAV file.
            chunk_frames: Number of PCM frames per send (default 4000 = 250ms).

        Returns:
            List of transcription result dicts.
        """
        async def _chunks() -> AsyncIterator[bytes]:
            with wave.open(path, "rb") as wf:
                # Warn if format deviates from expected
                if wf.getnchannels() != 1:
                    self._logger.warning(
                        "transcribe_file: expected mono (1 channel), got %d",
                        wf.getnchannels(),
                    )
                if wf.getsampwidth() != 2:
                    self._logger.warning(
                        "transcribe_file: expected 16-bit (sampwidth=2), got %d",
                        wf.getsampwidth(),
                    )
                if wf.getframerate() != 16000:
                    self._logger.warning(
                        "transcribe_file: expected 16000 Hz, got %d Hz",
                        wf.getframerate(),
                    )
                while True:
                    data = wf.readframes(chunk_frames)
                    if not data:
                        break
                    yield data

        return await self.stream_stt(_chunks())

    @staticmethod
    def save_tts_to_file(
        pcm_data: bytes, path: str, sample_rate: int, channels: int = 1
    ) -> None:
        """Write raw PCM bytes to a WAV file.

        Args:
            pcm_data: Raw 16-bit PCM bytes from synthesize_tts().
            path: Output file path (e.g. "output.wav").
            sample_rate: Sample rate from synthesize_tts() return value (e.g. 22050).
            channels: Number of audio channels (default 1 = mono).
        """
        with wave.open(path, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit = 2 bytes per sample
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)

    @staticmethod
    def pcm_to_wav_bytes(
        pcm_data: bytes, sample_rate: int, channels: int = 1
    ) -> bytes:
        """Wrap raw PCM bytes in a WAV container without writing to disk.

        Args:
            pcm_data: Raw 16-bit PCM bytes.
            sample_rate: Sample rate (e.g. 22050 for TTS, 16000 for STT).
            channels: Number of audio channels (default 1 = mono).

        Returns:
            WAV-formatted bytes (RIFF container with PCM payload).
        """
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_data)
        return buf.getvalue()
