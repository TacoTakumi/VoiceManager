"""Async Python client for VoiceManager STT/TTS server.

Standalone module — requires only ``websockets`` and ``httpx``.
Copy this file into any Python 3.12+ project.

The client authenticates with a bearer token: it sends
``Authorization: Bearer <token>`` on the WebSocket handshake and on every
REST call, and never puts the token in a URL.

Usage::

    async with VoiceManagerClient(host="localhost", port=8000, token="secret") as client:
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


# ---------------------------------------------------------------------------
# VoiceManagerClient
# ---------------------------------------------------------------------------

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
        port: int = 8000,
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
        # cancel_tts() on it — barge-in is impossible if the id stays internal.
        self.last_tts_request_id: str | None = None
        # request_ids the caller has cancelled. cancel_tts() adds one here
        # and returns immediately without awaiting audio_end; the live-yield path
        # discards binary frames for a cancelled request until its audio_end lands.
        self._cancelled_tts: set[str] = set()

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

        # Drain all queues — put sentinel None to unblock any waiters
        for q in (self._stt_queue, self._tts_queue, self._error_queue, self._turn_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break
            await q.put(None)

    async def _reconnect(self) -> None:
        """Reconnect with exponential backoff.

        Raises ConnectionError after max_retries attempts.
        Sets self.reconnected = True on success — consumer must re-upload
        their voice profile.
        """
        await self._disconnect()

        # Reset queues for the new connection
        self._stt_queue = asyncio.Queue()
        self._tts_queue = asyncio.Queue()
        self._error_queue = asyncio.Queue()
        self._turn_queue = asyncio.Queue()
        # A fresh connection carries no pending cancellations.
        self._cancelled_tts = set()

        delays = [1, 2, 4]
        for attempt, delay in enumerate(delays):
            try:
                await self._connect()
                self.reconnected = True
                self._logger.warning(
                    "Reconnected — voice profile must be re-uploaded (session=%s)",
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
        """Background task — dispatches all received WebSocket frames to queues.

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
                        # Server-owned ack for a turn_start — routed to
                        # its own queue so turn_start() awaits it independently.
                        await self._turn_queue.put(frame)
                    elif ftype == "error":
                        error_type = frame.get("error_type", "")
                        if error_type == "timeout_warning":
                            self._logger.warning(
                                "Server timeout warning: %s", frame.get("message", "")
                            )
                        await self._error_queue.put(frame)
                    else:
                        self._logger.debug("Unhandled message type: %s", ftype)
        except websockets.exceptions.ConnectionClosed:
            self._logger.info("WebSocket connection closed")
            self.connected = False
        finally:
            # Signal all waiters that the connection is gone
            await self._stt_queue.put(None)
            await self._tts_queue.put(None)
            await self._turn_queue.put(None)

    # ------------------------------------------------------------------
    # STT methods
    # ------------------------------------------------------------------

    @staticmethod
    def _stamp_turn_id(frame: dict) -> dict:
        """Guarantee a transcription dict carries a turn_id key.

        The server stamps turn_id on every transcription frame (default 0), so
        this is normally a no-op — but the client's public contract is that every
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
        results: list[dict] = []
        try:
            async for chunk in audio_chunks:
                await self._ws.send(chunk)
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
        transcription dict immediately — both interim (is_final=False) and
        final (is_final=True) — so consumers can react to partials in
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
        send_done = False
        try:
            send_iter = audio_chunks.__aiter__()
            while not send_done:
                # Send next audio chunk
                try:
                    chunk = await send_iter.__anext__()
                    await self._ws.send(chunk)
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

            # Audio exhausted — wait for remaining transcriptions
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
            The newly opened turn's id — a positive, monotonically increasing int.

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
            WAV files — TTS output rate depends on engine (22050 Hz for
            OpenVoice, 24000 Hz for VibeVoice).

        Raises:
            TTSError: If the connection is lost or times out.
        """
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
        self, text_chunks: AsyncIterator[str], voice: str | None = None
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

        Returns:
            Tuple of (pcm_bytes, sample_rate). Use sample_rate when writing
            WAV files — TTS output rate depends on engine.

        Raises:
            TTSError: If the connection is lost or times out.
        """
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
                if first_chunk and voice is not None:
                    payload["voice"] = voice
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
    ) -> AsyncIterator[bytes]:
        """Stream text for TTS and yield audio bytes in real-time.

        Unlike stream_tts() which collects all audio, this yields each PCM
        audio chunk as it arrives from the server — enabling real-time playback
        while text is still being sent.

        Sample rate is available via client.last_tts_sample_rate after the
        first audio chunk is yielded (updated when audio_start arrives).

        Primary use case: real-time LLM-to-speech with immediate playback.

        Barge-in: to cancel this request mid-stream, you need its
        request_id. Either supply one via the ``request_id`` argument (generate
        it yourself, e.g. ``str(uuid.uuid4())``, so you hold it before iteration
        begins), or read ``client.last_tts_request_id`` once iteration has
        started. Pass that id to ``cancel_tts()`` — typically from a separate
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

        Yields:
            Raw PCM bytes per audio chunk from the server.

        Raises:
            TTSError: If the connection is lost or times out.
        """
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
                    if first_chunk and voice is not None:
                        payload["voice"] = voice
                        first_chunk = False
                    msg = json.dumps(payload)
                    await self._ws.send(msg)
                except StopAsyncIteration:
                    # All text sent — send done=True
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

            # Text exhausted — collect remaining audio until audio_end
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

        Returns as soon as the tts_cancel frame is sent — it does NOT wait for
        the server's audio_end, which may lag while the executor unwinds a long
        synthesis. Any binary frames still in flight for the cancelled request
        are discarded by the live-yield path (stream_tts_live) until that
        request's own audio_end arrives and ends the iteration.

        Args:
            request_id: The request_id of the in-flight TTS request — either the
                one supplied to stream_tts_live(), or read from
                client.last_tts_request_id after starting any TTS request.
        """
        # Mark first, then send: once marked, the live-yield path drops binary
        # frames for this id even if the cancel frame and the next PCM chunk race.
        self._cancelled_tts.add(request_id)
        msg = json.dumps({"type": "tts_cancel", "request_id": request_id})
        await self._ws.send(msg)

    async def flush_tts(self, request_id: str) -> None:
        """Flush the server's buffered text for an in-flight streaming request.

        Puts exactly ONE tts_stream frame with flush=true and done=false on the
        wire for the given request_id. The server synthesizes everything it has
        buffered for that request so far — exactly as-is — and leaves the request
        OPEN for more text: no end-of-sentence sentinel is queued and no audio_end
        is emitted (that is what done=true does). This is what lets a one-sentence
        reply be spoken before the caller ever sends done=true.

        WARNING — flush only at a boundary YOU have detected to be the end of a
        sentence. Flushing MID-sentence hands the server a partial sentence and
        will audibly CUT that sentence in HALF: the buffered fragment is spoken on
        its own and the remainder arrives as a separate, disjoint utterance. The
        correct use is a sentence boundary the client itself detected (e.g. after a
        '.', '?' or '!' emerges from an LLM token stream) — never once per token.

        Args:
            request_id: The request_id of the in-flight streaming TTS request —
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
        await self._ws.send(msg)

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

        # Voice profile re-established — clear the reconnected flag
        self.reconnected = False
        return response.json()

    # ------------------------------------------------------------------
    # REST helpers
    # ------------------------------------------------------------------

    async def health(self) -> dict:
        """GET /health — model readiness status.

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
        """GET /info — loaded models, device, and active sessions.

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
