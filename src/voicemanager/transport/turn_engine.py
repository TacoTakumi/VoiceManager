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

The hop feed
------------

An opted-in session feeds the sidecar one hop per 5120 samples (320 ms) of the
client's mic stream, built by ``HopFeed``:

* The mic stream is the PCM of the client's binary frames, counted in samples from
  its first audio frame. A WAV header on the first frame is not counted.
* The TTS session's audio tap (``on_tts_text`` / ``on_tts_pcm``) keeps a copy of the
  PCM sent per request_id, at the rate its audio_start named.
* A playback start places reply sample 0 at its ``mic_sample``. The played span ends
  at the stop's ``reply_sample`` (counted at the reply's own rate, so scaled by
  16000 / rate), or at the end of the PCM sent so far when there is no stop yet.
* The agent channel of a hop is the reply resampled to 16 kHz over the part of the
  played span inside the hop, and silence elsewhere. The resampling is scipy's
  ``resample_poly`` over a slice with a margin wider than its filter, which gives
  the same samples as resampling the whole reply (as the offline replay harness
  does). A hop that overlaps a played span is in speaking mode, any other hop in
  listening mode with a silent agent channel.

A start frame reports an edge after the fact, so it can come in after the mic audio
it points into. While a reply the server sent has no start yet, each hop waits for
``START_LOOKAHEAD_SAMPLES`` more mic audio before it goes out, so a start that is
late by up to that much still places its hop. The wait ends when the start comes,
or ``START_WAIT_SAMPLES`` of mic audio after the reply's audio_start went out (a
client that never reports playback costs only that window). A hop already sent stays
as it was: a start later than the lookahead leaves the earlier hops in listening
mode, and a late stop leaves the hops already sent with the reply past the cut.

Memory is bounded per connection: at most ``MAX_KEPT_REPLIES`` replies are kept (the
oldest is forgotten first, and a playback frame for it is then for an unknown
request_id), each with at most ``MAX_REPLY_SECONDS`` of PCM. A reply's PCM is let go
as soon as its played span is fully fed and no more of it can play (its audio_end
went out, a stop came, or it reached the cap).

Hops are packed off the event loop and sent by one sender task, in order. When the
sidecar link is down they are dropped quietly; at most ``MAX_QUEUED_HOPS`` wait to
be sent, and a hop past that is dropped with a warning.
"""
import asyncio
import collections
import functools
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import numpy as np
import websockets.asyncio.client
import websockets.exceptions
from pydantic import ValidationError
from websockets.protocol import State

from voicemanager.pipeline.tyger import wire
from voicemanager.protocol import (
    ErrorMessage, PlaybackMessage, TurnEngineMessage, TurnEngineReadyMessage,
    INVALID_REQUEST, TURN_ENGINES, TURN_ENGINE_UNAVAILABLE, TURN_ENGINE_UNSUPPORTED,
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

#: How much more mic audio a hop waits for while a sent reply has no playback start.
START_LOOKAHEAD_SAMPLES = wire.HOP_SAMPLES
#: How long, in mic samples after its audio_start, a reply's start is waited for.
START_WAIT_SAMPLES = 2 * wire.SAMPLE_RATE
#: How many replies a connection keeps the sent PCM of.
MAX_KEPT_REPLIES = 4
#: The most PCM kept per reply, in seconds at the reply's rate.
MAX_REPLY_SECONDS = 120
#: How many packed hops may wait for the sidecar link.
MAX_QUEUED_HOPS = 32


# --- the agent channel ----------------------------------------------------------

@dataclass
class _Reply:
    """What one request sent, and where the client reports it played."""
    request_id: str
    rate: int
    seen_at: int                      # mic samples in when its audio_start went out
    pcm: bytearray | None = field(default_factory=bytearray)  # None once let go
    n_samples: int = 0                # samples sent (kept after the PCM is let go)
    done: bool = False                # its audio_end went out
    capped: bool = False              # reached MAX_REPLY_SECONDS; the rest is not kept
    start: int | None = None          # the start frame's mic_sample
    stop_mic: int | None = None       # the stop frame's mic_sample
    cut16: int | None = None          # the stop frame's reply_sample, at 16 kHz

    @property
    def length16(self) -> int:
        """The played length at 16 kHz: the reply so far, cut at the stop."""
        n = -(-self.n_samples * wire.SAMPLE_RATE // self.rate)
        return n if self.cut16 is None else min(n, self.cut16)

    @property
    def complete(self) -> bool:
        """No more of this reply can play."""
        return self.done or self.capped or self.cut16 is not None


@dataclass(frozen=True)
class _Piece:
    """The part of one reply that fills a hop's agent channel."""
    offset: int    # the first hop sample it fills
    count: int     # how many hop samples it fills
    pcm: bytes     # a slice of the reply, int16 LE at its own rate
    rate: int
    first: int     # index, in the slice resampled to 16 kHz, of the first sample used


@dataclass(frozen=True)
class _HopJob:
    """Everything one hop needs, cut on the event loop and packed off it."""
    mode: str
    mic_sample: int
    user: bytes
    pieces: tuple[_Piece, ...]


@functools.lru_cache(maxsize=None)
def _ratio(rate: int) -> tuple[int, int]:
    """(up, down) taking ``rate`` to 16 kHz, in lowest terms."""
    g = math.gcd(wire.SAMPLE_RATE, rate)
    return wire.SAMPLE_RATE // g, rate // g


def _margin(up: int, down: int) -> int:
    """Input samples beyond a slice's edges that resample_poly's filter reaches."""
    return -(-10 * max(up, down) // up) + 2


@functools.lru_cache(maxsize=None)
def _filter(up: int, down: int):
    """resample_poly's own default filter, designed once per ratio."""
    from scipy.signal import firwin
    max_rate = max(up, down)
    return firwin(2 * 10 * max_rate + 1, 1.0 / max_rate, window=("kaiser", 5.0))


def _resample16(pcm: bytes, rate: int) -> np.ndarray:
    """int16 LE PCM at ``rate`` as int16 samples at 16 kHz."""
    x = np.frombuffer(pcm, dtype="<i2")
    if rate == wire.SAMPLE_RATE:
        return x.astype(np.int16)
    from scipy.signal import resample_poly
    up, down = _ratio(rate)
    y = resample_poly(x.astype(np.float64), up, down, window=_filter(up, down))
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def _pack_hop(job: _HopJob) -> bytes:
    """Build one hop's binary message. Runs in a worker thread."""
    agent = np.zeros(wire.HOP_SAMPLES, dtype=np.int16)
    for piece in job.pieces:
        y = _resample16(piece.pcm, piece.rate)
        agent[piece.offset:piece.offset + piece.count] = y[piece.first:piece.first + piece.count]
    return wire.pack_hop(job.mode, job.mic_sample, job.user, agent.astype("<i2"))


def _wav_payload_offset(data: bytes) -> int | None:
    """Where the PCM starts in a WAV container, or None when data is not one.

    The offset is past the 'data' chunk header; a container with no 'data' chunk
    in this frame holds no samples (the whole frame is header).
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


class HopFeed:
    """One connection's mic stream, sent replies and playback edges, cut into hops.

    Plain state, no I/O: the session feeds it and sends what ``due_hops`` returns.
    """

    def __init__(self, label: str = "") -> None:
        self._label = label
        self._replies: collections.OrderedDict[str, _Reply] = collections.OrderedDict()
        self._mic = bytearray()       # mic PCM not yet in a hop
        self._mic_bytes = 0           # mic PCM bytes in, ever
        self._mic_frames = 0
        self._hops = 0                # hops cut so far

    @property
    def mic_samples(self) -> int:
        """Samples of the mic stream received so far."""
        return self._mic_bytes // 2

    def reply(self, request_id: str) -> _Reply | None:
        return self._replies.get(request_id)

    # --- the TTS side ---------------------------------------------------------------

    def on_tts_text(self, request_id: str, text: str) -> None:
        """A text frame a synthesis task sent: note audio_start and audio_end."""
        if '"audio_start"' not in text and '"audio_end"' not in text:
            return
        try:
            frame = json.loads(text)
        except json.JSONDecodeError:
            return
        kind = frame.get("type")
        if kind == "audio_start":
            rate = frame.get("sample_rate")
            if request_id in self._replies or type(rate) is not int or rate <= 0:
                return
            self._replies[request_id] = _Reply(request_id=request_id, rate=rate,
                                               seen_at=self.mic_samples)
            while len(self._replies) > MAX_KEPT_REPLIES:
                self._replies.popitem(last=False)
        elif kind == "audio_end":
            reply = self._replies.get(request_id)
            if reply is not None:
                reply.done = True

    def on_tts_pcm(self, request_id: str, pcm: bytes) -> None:
        """PCM a synthesis task sent for request_id: keep a copy, up to the cap."""
        reply = self._replies.get(request_id)
        if reply is None or reply.pcm is None or reply.capped:
            return
        room = reply.rate * MAX_REPLY_SECONDS * 2 - len(reply.pcm)
        if len(pcm) > room:
            pcm = pcm[:max(0, room)]
            reply.capped = True
            logger.warning("Turn engine keeps at most %d s of a reply: session=%s "
                           "request_id=%s", MAX_REPLY_SECONDS, self._label, request_id)
        reply.pcm += pcm
        reply.n_samples = len(reply.pcm) // 2

    # --- the playback frame ---------------------------------------------------------

    def playback(self, frame: dict) -> str | None:
        """Apply one playback frame. The reason it is refused, or None."""
        try:
            msg = PlaybackMessage.model_validate(frame)
        except ValidationError:
            return ("A playback frame needs a string request_id, an event of 'start' or "
                    "'stop', an integer mic_sample >= 0 and, on 'stop' only, an integer "
                    "reply_sample >= 0; the frame was dropped.")
        reply = self._replies.get(msg.request_id)
        if reply is None:
            return (f"No reply with request_id {msg.request_id!r} was sent on this "
                    "connection; the frame was dropped.")
        if msg.event == "start":
            if reply.start is not None:
                return f"Playback of {msg.request_id!r} already started; the frame was dropped."
            reply.start = msg.mic_sample
            return None
        if reply.start is None:
            return (f"A playback stop for {msg.request_id!r} came without a start; "
                    "the frame was dropped.")
        if reply.stop_mic is not None:
            return f"Playback of {msg.request_id!r} already stopped; the frame was dropped."
        if msg.mic_sample < reply.start:
            return (f"The playback stop for {msg.request_id!r} is before its start on the "
                    "mic stream; the frame was dropped.")
        reply.stop_mic = msg.mic_sample
        # reply_sample counts samples at the reply's own rate.
        reply.cut16 = (msg.reply_sample * wire.SAMPLE_RATE + reply.rate // 2) // reply.rate
        return None

    # --- the mic side ---------------------------------------------------------------

    def feed_mic(self, data: bytes) -> None:
        """One binary frame of the client's mic stream."""
        if self._mic_frames == 0:
            offset = _wav_payload_offset(data)
            if offset is not None:
                data = data[offset:]
        self._mic_frames += 1
        self._mic += data
        self._mic_bytes += len(data)

    def due_hops(self) -> list[_HopJob]:
        """Cut every hop whose mic audio is in and that need not wait any longer."""
        jobs = []
        while True:
            end = (self._hops + 1) * wire.HOP_SAMPLES
            wait = START_LOOKAHEAD_SAMPLES if self._awaiting_start() else 0
            if self.mic_samples < end + wait:
                return jobs
            jobs.append(self._cut(end))

    def _awaiting_start(self) -> bool:
        """A sent reply may still get a start that points into audio already in."""
        for reply in self._replies.values():
            if reply.start is not None or (reply.done and reply.n_samples == 0):
                continue
            if self.mic_samples - reply.seen_at < START_WAIT_SAMPLES:
                return True
        return False

    def _cut(self, end: int) -> _HopJob:
        start = end - wire.HOP_SAMPLES
        user = bytes(self._mic[:wire.CHANNEL_BYTES])
        del self._mic[:wire.CHANNEL_BYTES]
        self._hops += 1
        speaking = False
        pieces = []
        # Oldest reply first, so a later reply's samples win where two overlap.
        for reply in self._replies.values():
            if reply.start is None:
                continue
            s0, s1 = reply.start, reply.start + reply.length16
            if not (s1 > s0 and s0 < end and s1 > start):
                continue
            speaking = True
            piece = self._piece(reply, start, end)
            if piece is not None:
                pieces.append(piece)
        for reply in self._replies.values():
            # Played span fully fed and nothing more can play: let the PCM go.
            if (reply.pcm is not None and reply.start is not None and reply.complete
                    and end >= reply.start + reply.length16):
                reply.pcm = None
        mode = wire.MODE_SPEAKING if speaking else wire.MODE_LISTENING
        return _HopJob(mode=mode, mic_sample=end, user=user, pieces=tuple(pieces))

    @staticmethod
    def _piece(reply: _Reply, start: int, end: int) -> _Piece | None:
        """The slice of reply that fills mic samples [start, end), with its margins."""
        if reply.pcm is None:
            return None
        j0 = max(start - reply.start, 0)
        j1 = min(end - reply.start, reply.length16)
        if j1 <= j0:
            return None
        if reply.rate == wire.SAMPLE_RATE:
            s, e, first = j0, j1, 0
        else:
            up, down = _ratio(reply.rate)
            margin = _margin(up, down)
            # The slice starts on a multiple of `down` input samples, so its resampled
            # output lands on the whole reply's 16 kHz grid.
            s = max(0, j0 * down // up - margin) // down * down
            e = min(reply.n_samples, -(-j1 * down // up) + margin)
            first = j0 - s // down * up
        return _Piece(offset=reply.start + j0 - start, count=j1 - j0,
                      pcm=bytes(reply.pcm[2 * s:2 * e]), rate=reply.rate, first=first)


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
        self._feed = HopFeed(label)
        self._hop_queue: asyncio.Queue | None = None
        self._sender: asyncio.Task | None = None
        self._dropping = False        # hops are being dropped on a full queue

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

    # --- playback, the TTS audio tap and the mic stream ------------------------------

    async def handle_playback(self, frame: dict) -> None:
        """Apply one playback frame, or answer it with an invalid_request error frame."""
        refusal = self._feed.playback(frame)
        if refusal is not None:
            logger.info("Playback frame refused: session=%s (%s)", self._label, refusal)
            await self._send_error(INVALID_REQUEST, refusal)
            return
        logger.debug("Playback: session=%s %s %s at mic_sample=%s", self._label,
                     frame.get("request_id"), frame.get("event"), frame.get("mic_sample"))
        self._queue_due_hops()

    def on_tts_text(self, request_id: str, text: str) -> None:
        """The TTS session's audio tap: a text frame a reply sent."""
        if self.active:
            self._feed.on_tts_text(request_id, text)

    def on_tts_pcm(self, request_id: str, pcm: bytes) -> None:
        """The TTS session's audio tap: PCM a reply sent."""
        if self.active:
            self._feed.on_tts_pcm(request_id, pcm)

    def feed_mic(self, data: bytes) -> None:
        """One binary frame of the client's mic stream. Never waits on the sidecar."""
        self._feed.feed_mic(data)
        self._queue_due_hops()

    def _queue_due_hops(self) -> None:
        if self._hop_queue is None:
            return
        for job in self._feed.due_hops():
            try:
                self._hop_queue.put_nowait(job)
            except asyncio.QueueFull:
                if not self._dropping:
                    logger.warning("Turn engine sidecar is behind; dropping hops: session=%s",
                                   self._label)
                self._dropping = True
            else:
                self._dropping = False

    async def _send_hops(self, ws, queue: asyncio.Queue) -> None:
        """Pack each queued hop off the loop and send it, in order."""
        link_down = False
        while True:
            job = await queue.get()
            if ws.state is not State.OPEN:
                if not link_down:
                    logger.info("Turn engine sidecar link is down; dropping hops: session=%s",
                                self._label)
                link_down = True
                continue
            try:
                hop = await asyncio.to_thread(_pack_hop, job)
                await ws.send(hop)
            except websockets.exceptions.ConnectionClosed:
                link_down = True
            except Exception:
                logger.exception("Sending a hop to the turn engine sidecar failed: session=%s",
                                 self._label)

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
        self._hop_queue = asyncio.Queue(maxsize=MAX_QUEUED_HOPS)
        self._sender = asyncio.create_task(self._send_hops(self._ws, self._hop_queue),
                                           name=f"turn-engine-sender-{self._label}")
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
        sender, self._sender = self._sender, None
        self._hop_queue = None
        ws, self._ws = self._ws, None
        tasks = [t for t in (reader, sender) if t is not None]
        for task in tasks:
            task.cancel()
        if ws is None:
            return
        try:
            await ws.close()  # bounded by CLOSE_TIMEOUT_S
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            ws.transport.abort()
            raise
        except Exception:
            logger.debug("Closing the turn engine sidecar connection failed: session=%s",
                         self._label, exc_info=True)
            ws.transport.abort()
