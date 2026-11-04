"""Per-connection turn-engine session: the opt-in frame and the sidecar link.

A connection opts in to a turn engine with {"type": "turn_engine", "engine":
"tyger"} sent before its first audio frame. ``TurnEngineSession`` is the one
per-connection object behind that: the WebSocket handler creates it for every
connection (no I/O, no task), routes the opt-in frame to ``handle_opt_in`` and
calls ``aclose`` when the connection ends. Until an opt-in succeeds the object
does nothing, so a connection that never opts in runs exactly the handler's
code path without a turn engine.

An opt-in is checked in this order, and the first check that fails answers it:

1. The turn engine is off on this server (no sidecar configured):
   ``turn_engine_unavailable``.
2. The connection's STT engine is not audiocpp: ``turn_engine_unsupported``.
3. The frame itself: malformed (no string ``engine``), an engine name not in
   ``TURN_ENGINES``, sent after the first audio frame, or sent after an earlier
   opt-in that already got ``turn_engine_ready``: ``invalid_request``.
4. The sidecar is configured but not up (starting or restarting), or the
   connection to it fails: ``turn_engine_unavailable``.

The two server-wide answers come first: they hold for the life of the server
(or of the connection's engine), so a client learns them whatever its frame
holds. A refused opt-in changes nothing, so the client may send a corrected one
before its first audio; only a successful opt-in makes a later one a second
opt-in. A second opt-in leaves the first one in place.

On success the session holds one WebSocket to the sidecar
(``sidecar.session_url(label)``) and a reader task that parses the sidecar's
event lines (``wire.parse_events``) and hands each event to ``on_event``. The
server imports only the wire format of the sidecar, never its scorer.
"""
import asyncio
import logging
from typing import Awaitable, Callable

import websockets.asyncio.client
import websockets.exceptions
from pydantic import ValidationError

from voicemanager.pipeline.tyger import wire
from voicemanager.protocol import (
    ErrorMessage, TurnEngineMessage, TurnEngineReadyMessage, INVALID_REQUEST,
    TURN_ENGINES, TURN_ENGINE_UNAVAILABLE, TURN_ENGINE_UNSUPPORTED,
)

logger = logging.getLogger(__name__)

#: The STT engines a connection may opt in on.
SUPPORTED_STT_ENGINES: tuple[str, ...] = ("audiocpp",)
#: How long the connection to the sidecar may take before the opt-in is refused.
CONNECT_TIMEOUT_S = 5.0
#: How long closing the sidecar connection may take at teardown.
CLOSE_TIMEOUT_S = 2.0

SendText = Callable[[str], Awaitable[None]]
EventHandler = Callable[[wire.Event], Awaitable[None]]


class TurnEngineSession:
    """One connection's turn engine: opt-in checks, sidecar link and event reader."""

    def __init__(self, *, sidecar, stt_engine_type: str, label: str,
                 send_text: SendText, on_event: EventHandler | None = None) -> None:
        """``sidecar`` is app.state.turn_engine_sidecar (None when the setting is off):
        anything with ``up`` and ``session_url(label)``. ``label`` names this
        connection's session on the sidecar (the connection's session id).
        ``send_text`` sends one text frame to the client. ``on_event`` gets each
        event the sidecar sends; without one, events are only logged."""
        self._sidecar = sidecar
        self._stt_engine_type = stt_engine_type
        self._label = label
        self._send_text = send_text
        self._on_event = on_event
        self.engine: str | None = None
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._closed = False

    @property
    def active(self) -> bool:
        """The connection opted in and its sidecar link is open."""
        return self._ws is not None and not self._closed

    # --- the opt-in frame -----------------------------------------------------------

    async def handle_opt_in(self, frame: dict, *, audio_started: bool) -> None:
        """Answer one turn_engine frame: turn_engine_ready, or one error frame.

        ``audio_started`` is True once the connection has sent its first audio frame.
        """
        refusal = self._refusal(frame, audio_started)
        if refusal is not None:
            error_type, message = refusal
            logger.info("Turn engine opt-in refused: session=%s error=%s (%s)",
                        self._label, error_type, message)
            await self._send_error(error_type, message)
            return
        engine = frame["engine"]
        if not await self._connect():
            await self._send_error(
                TURN_ENGINE_UNAVAILABLE,
                "The turn engine could not be reached; the session goes on without it.")
            return
        self.engine = engine
        logger.info("Turn engine ready: session=%s engine=%s", self._label, engine)
        await self._send_text(TurnEngineReadyMessage(engine=engine).model_dump_json())

    def _refusal(self, frame: dict, audio_started: bool) -> tuple[str, str] | None:
        """The (error_type, message) that refuses this opt-in, or None to go ahead."""
        if self._sidecar is None:
            return (TURN_ENGINE_UNAVAILABLE,
                    "The turn engine is off on this server.")
        if self._stt_engine_type not in SUPPORTED_STT_ENGINES:
            return (TURN_ENGINE_UNSUPPORTED,
                    f"The turn engine needs the {', '.join(SUPPORTED_STT_ENGINES)} STT "
                    f"engine; this server runs {self._stt_engine_type}.")
        try:
            msg = TurnEngineMessage.model_validate(frame)
        except ValidationError:
            return (INVALID_REQUEST,
                    "A turn_engine frame needs a string 'engine'; the frame was dropped.")
        if msg.engine not in TURN_ENGINES:
            return (INVALID_REQUEST,
                    f"Unknown turn engine {msg.engine!r}; known: "
                    f"{', '.join(TURN_ENGINES)}.")
        if audio_started:
            return (INVALID_REQUEST,
                    "A turn_engine frame must come before the first audio frame.")
        if self.engine is not None:
            return (INVALID_REQUEST,
                    "This connection already opted in to a turn engine.")
        if not self._sidecar.up:
            return (TURN_ENGINE_UNAVAILABLE,
                    "The turn engine is not running right now; try again later.")
        return None

    async def _send_error(self, error_type: str, message: str) -> None:
        await self._send_text(
            ErrorMessage(error_type=error_type, message=message).model_dump_json())

    # --- the sidecar link -----------------------------------------------------------

    async def _connect(self) -> bool:
        """Open this connection's sidecar WebSocket and start its reader."""
        url = self._sidecar.session_url(self._label)
        try:
            self._ws = await websockets.asyncio.client.connect(
                url, open_timeout=CONNECT_TIMEOUT_S, close_timeout=CLOSE_TIMEOUT_S)
        except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as exc:
            logger.warning("Turn engine sidecar connect failed: session=%s url=%s: %s",
                           self._label, url, exc)
            self._ws = None
            return False
        self._reader = asyncio.create_task(self._read(self._ws),
                                           name=f"turn-engine-reader-{self._label}")
        return True

    async def _read(self, ws) -> None:
        """Parse every event line the sidecar sends and hand each event on."""
        try:
            async for message in ws:
                if not isinstance(message, str):
                    logger.warning("Turn engine sidecar sent a binary message: session=%s",
                                   self._label)
                    continue
                try:
                    events = wire.parse_events(message)
                except ValueError as exc:
                    logger.warning("Bad event line from the turn engine sidecar: "
                                   "session=%s: %s", self._label, exc)
                    continue
                for event in events:
                    await self._dispatch(event)
        except websockets.exceptions.ConnectionClosed as exc:
            logger.info("Turn engine sidecar connection closed: session=%s: %s",
                        self._label, exc)
        else:
            logger.info("Turn engine sidecar connection ended: session=%s", self._label)

    async def _dispatch(self, event: wire.Event) -> None:
        logger.debug("Turn engine event: session=%s %s", self._label, event)
        if self._on_event is not None:
            await self._on_event(event)

    # --- teardown -------------------------------------------------------------------

    async def aclose(self) -> None:
        """Stop the reader and close the sidecar connection. Safe to call twice.

        A connection that never opted in returns at once, without suspending. When
        the caller is cancelled during the close, the connection is dropped at once
        and the cancellation goes on.
        """
        if self._closed:
            return
        self._closed = True
        reader, self._reader = self._reader, None
        ws, self._ws = self._ws, None
        if reader is not None:
            reader.cancel()
        if ws is None:
            return
        try:
            await ws.close()  # bounded by CLOSE_TIMEOUT_S
            if reader is not None:
                await asyncio.gather(reader, return_exceptions=True)
        except asyncio.CancelledError:
            ws.transport.abort()
            raise
        except Exception:
            logger.debug("Closing the turn engine sidecar connection failed: session=%s",
                         self._label, exc_info=True)
            ws.transport.abort()
