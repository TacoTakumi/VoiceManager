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
   opt-in that already got ``turn_engine_ready`` (also once that turn engine is
   lost: see "Losing the sidecar"): ``invalid_request``.
4. The sidecar is configured but not up (starting or restarting), or the
   connection to it fails: ``turn_engine_unavailable``.

The two server-wide answers come first: they hold for the life of the server
(or of the connection's engine), so a client learns them whatever its frame
holds. A refused opt-in changes nothing, so the client may send a corrected one
before its first audio; only a successful opt-in makes a later one a second
opt-in. A second opt-in leaves the first one in place.

On success the session holds one WebSocket to the sidecar
(``sidecar.session_url(label)``, with ``progress=1`` added to ask for progress lines)
and a reader task that parses the sidecar's event lines (``wire.parse_events``) and
hands each event to ``on_event``. The server imports only the wire format of the
sidecar, never its scorer.

A progress line (``wire.ScoredEvent``) says the sidecar is done with every hop up to
its ``mic_sample``, and it comes after those hops' events, so once the reader has
handled one at X it has handled every barge up to X. The sidecar sends one at 0 when
the session opens; a sidecar that sends none reports no progress, and the mic hold
then works as if every hop were scored at once (see "The mic hold").

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
* A reply's sending ends at its audio_end. The tap does not see an audio_end sent from
  the TTS session's cancel path, so a client's tts_cancel ends it through
  ``on_tts_cancel``, and a new reply's audio_start ends every earlier one (one TTS
  request is in flight per connection).
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
or the start wait (VM_TYGER_START_WAIT_MS, ``START_WAIT_SAMPLES`` by default) of mic
audio after the reply's audio_start went out (a client that never reports playback
costs only that window). A hop already sent stays
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

A barge
-------

A barge event says the user cut in over the reply playing at the client. It is about
the reply that played at the event's ``mic_sample``, picked by position, not by the
order the frames came in, from the replies whose playback start came
(``HopFeed.played_reply``):

1. the latest-started one whose played span holds the barge. A reply with no stop
   holds ``mic_sample`` from its start on, and a reply that may still get PCM (no
   audio_end yet) has a span open at its end, since the client can play what is
   still to come. A stopped reply holds a barge whose ``onset_mic_sample`` is at or
   before its stop's ``mic_sample`` (the user began over the reply) while the mic
   hold still waits for the sidecar's progress on that reply (see "The mic hold"),
   and not once that hold has ended (its audio dropped, or fed after an earlier
   barge). With a sidecar that reports no progress, nothing waits, and the barge's
   ``mic_sample`` must be at or before the stop plus the hold tail, or the end of
   the last hop sent in speaking mode over the reply, whichever is later. The
   client reports a stop by DAC time, so the stop frame can come in before the
   sidecar has scored the mic audio up to it: a barge over the last moments of a
   reply comes after its stop. The hop that holds the stop overlaps the played span,
   so all of it goes out in speaking mode and the yield head may decide a barge on
   any of its frames, up to a hop past the stop; a late stop leaves the hops already
   sent with the reply in speaking mode further past it;
2. else the latest-started one with no stop, which covers a client whose playback
   runs behind the PCM count (a stall, a deep buffer) and so past the span the
   server works out.

A barge with no such reply is ignored (logged at debug). For the reply it picks, the
server sends the barge frame ({"type": "barge", "request_id", "onset_mic_sample",
"mic_sample"}) and cancels the request through the TTS session's cancel path with
reason 'interrupted' and the barge frame as its notice. That path silences the
request before the barge frame goes out and sends the frame and the audio_end from
one coroutine, so the client gets the barge frame, then audio_end reason
'interrupted', and no PCM for the request after the barge frame, whatever the
synthesis task is doing at the time. A reply whose audio_end already went out (it
finished, and still plays at the client) gets the barge frame alone, so the client
clears its speaker: there is nothing left to cancel and no second audio_end. A
stopped reply is handled the same way: its request is cancelled if it is still open
(the client stopped it early), else it gets the frame alone. The client's playback
stop for the reply is still taken after the cancel, as it carries
how much really played.

The mic hold
------------

While a reply plays at the client, the mic picks it up, and the STT would transcribe
the agent's own voice. So for an opted-in session the handler sends mic audio to the
STT through ``MicHold`` (the hop feed still gets every frame at once, unchanged, as
the sidecar needs the user channel during playback to hear a barge). The hold reads
each reply's playback state from the ``HopFeed``; positions are mic samples:

* A reply holds the mic from its ``seen_at`` (the mic samples in when its audio_start
  went out).
* Its playback start frees the audio before its ``mic_sample``, which goes to the STT
  in order. From the start on the reply is in play.
* A stop ends the play at its ``mic_sample`` plus the hold tail (VM_TYGER_HOLD_TAIL_MS).
  A barge for the reply can still come until the sidecar has scored the mic audio up
  to there and every hop sent in speaking mode over the reply, so the in-play audio
  stays held until its progress reaches the stop plus the tail or the end of the
  last such hop, whichever is later (the reply's wait end). Then, with no barge, it
  is dropped; with a barge, it is fed as below. A sidecar that reports no progress
  leaves nothing to wait for: the audio is dropped at the stop. A slow one is waited
  for until the mic stream is the start wait (in mic samples) past the stop plus the
  tail; the audio is then dropped as if no barge came.
* A barge the session acts on (it picked a played reply) feeds the held audio from
  ``max(onset_mic_sample, mic_sample - BARGE_LOOKBACK_SAMPLES)``: every reply whose
  audio_start went out by the barge's mic_sample stops holding there (one with no
  start yet holds nothing more), as the barge frame clears the client's speaker. A
  reply whose audio_start goes out after it holds as usual. A barge the session
  ignores changes nothing.
* A reply with no start by the start wait (VM_TYGER_START_WAIT_MS, wall clock) after
  its audio_end ends its hold: its held audio goes to the STT in order, as if it never
  played, and a later start for it holds nothing. A reply that ended with no PCM
  never plays and holds nothing.

Several replies may be kept at once. Each holds as above, and the hold is their
union: a sample is held while any reply still holds it (a reply with no start yet, one
in play with no stop, or a stopped one the sidecar has not scored past), else dropped
while any reply's played span covers it, else fed. The STT gets audio strictly in mic
order: the oldest held sample blocks everything after it, so audio released late
still goes before any later audio.

Memory is bounded. In-play audio is kept for ``BARGE_LOOKBACK_SAMPLES`` plus
``HOLD_MARGIN_SAMPLES`` (4 s and 1 s): older audio of a reply still in play is dropped,
as no barge can reach it. Audio waiting for a start is kept for the start wait plus
``HOLD_MARGIN_SAMPLES``; older audio is fed, as if that reply had not played. A reply
that ended with no stop yet is in play for at most its length plus the start wait, so
a client that never reports its stop costs only that. A stopped reply waits for the
sidecar's progress for at most the start wait of mic audio past its stop plus the
tail (and keeps only the last ``BARGE_LOOKBACK_SAMPLES`` plus ``HOLD_MARGIN_SAMPLES``
of its in-play audio meanwhile), so a slow sidecar cannot grow the hold without
limit; when the sidecar is lost, the hold ends at once (see "Losing the sidecar").

Every STT drive happens under the handler's STT lock. The handler routes each mic
frame through ``take_stt_audio`` under it; when the hold may let audio go without a
new frame (a playback frame, a barge, a progress line that passes a stopped reply's
wait end, the start wait), the session calls the handler's ``release_held``, which
takes the lock and feeds what ``take_stt_audio`` gives. As each taker removes audio
from the front of the hold under the lock, no release reorders with a live frame. A
barge's release runs before the reader handles the next event, so a turn end that
follows it flushes the user's words. A turn end feeds whatever the hold has let go
and then flushes; audio still held stays held and reaches the STT after that
turn_end. A held first frame is fed without its WAV header. When the connection
ends, audio still held is dropped.

The turn end
------------

An endpoint event (``wire.EndpointEvent``, by the model's endpoint head or by its
silence fallback) ends the user's turn. The session collects the text of every final
the connection sends (``note_final``, called by the handler's transcription sink once
the frame is out), and ``send_turn_end`` sends {"type": "turn_end", "turn_id", "text",
"mic_sample", "by"}: text is those finals since the previous turn_end, each stripped,
empty ones skipped, joined with one space, and turn_id is the turn the last of them
carries. With no final text since the previous turn_end, no frame goes out.

The event comes in on the reader task, concurrent with the handler. The handler owns
the STT pipeline, so it passes ``end_turn``: a callback that FLUSHes the open
utterance under the same lock the handler drives the pipeline with, and then calls
``send_turn_end`` under that lock too, so the flushed final goes out first and is in
the turn_end's text, and no other final can slip in between. Without ``end_turn`` the
session calls ``send_turn_end`` directly. Silero still ends utterances and their finals
still go out at once; for an opted-in session only turn_end ends the turn.

The listening boundary
----------------------

When the client stops listening (a listen_stop on a listening connection), the
handler calls ``listen_boundary`` before it resets the STT and before the
listen_stopped answer goes out. Nothing from the audio before the stop crosses it:

* the boundary is the hop feed's mic count at the stop. The whole hops still in the
  feed are cut at once (with no wait for a late start) and queued, then the boundary
  control (``wire.format_boundary_control``) is queued behind them on the same hop
  queue, so the sidecar gets it after every hop before the stop and clears that
  session's turn rules (its voice gate and VoxLock enrollment stay: the link is not
  reopened). The partial hop after them is dropped, not padded. The next hop starts
  at the boundary; the mic count goes on, so hop mic_samples stay on the
  connection's count;
* an endpoint or barge event at or before the boundary is ignored, whenever it comes
  in, and ``send_turn_end`` sends nothing for one, so no turn_end and no barge frame
  goes out for pre-stop audio after the answer;
* the finals collected for the next turn_end are dropped: the first turn_end after
  listen_start holds only finals from audio after it;
* the mic hold drops everything it keeps and the start-wait timers are cancelled;
  the handler makes a release or turn end queued before the stop do nothing (it
  carries the listen epoch it was started in), so held pre-stop audio cannot reach
  the reset STT;
* every kept reply is forgotten. While stopped, playback frames are dropped quietly
  (no barge can come with no mic audio) and a reply sent meanwhile (a spoken
  goodbye) is not kept, so it holds no mic audio and is no barge candidate. After
  listen_start a playback frame for a reply forgotten at the boundary or sent while
  stopped is dropped quietly too; replies sent from then on work as usual.

A connection that never stops listening is not touched. A lost session only drops
what its hold keeps.

Losing the sidecar
------------------

The session's sidecar is lost when its own link to the sidecar closes while the
session did not close it (the reader ends), or when the sidecar's watcher reports
that the generation it connected to is gone (``sidecar.add_listener``; the listener
may run on another thread or loop, so it hands over to the session's loop). Either
one, whichever comes first, degrades the session once:

* the client gets one error frame of error_type ``turn_engine_lost``;
* ``active`` turns False, so the handler routes everything as for a connection that
  never opted in: playback frames are dropped unanswered, and no final is collected;
* the hold ends: all the mic audio it keeps goes to the STT in order, through
  ``release_held``, and the handler routes each later frame through the hold for as
  long as it still keeps audio (``holds_mic``), so no later frame overtakes it; once
  it is empty, frames go straight to the STT;
* the TTS audio tap is detached, start-wait timers are cancelled, the hop sender
  stops and queued hops are dropped;
* events the sidecar still sends are ignored, and a turn end already under way sends
  no turn_end frame.

The session stays lost for the life of the connection: a later opt-in on it is
refused with ``invalid_request``. Once the sidecar is back, a new connection may opt
in again.
"""
import asyncio
import collections
import functools
import json
import logging
import math
import urllib.parse
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import numpy as np
import websockets.asyncio.client
import websockets.exceptions
from pydantic import ValidationError
from websockets.protocol import State

from voicemanager.pipeline.tyger import wire
from voicemanager.protocol import (
    BargeMessage, ErrorMessage, PlaybackMessage, TurnEndMessage, TurnEngineMessage,
    TurnEngineReadyMessage,
    INVALID_REQUEST, TURN_ENGINES, TURN_ENGINE_LOST, TURN_ENGINE_UNAVAILABLE,
    TURN_ENGINE_UNSUPPORTED,
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
EndTurn = Callable[[wire.EndpointEvent], Awaitable[None]]
ReleaseHeld = Callable[[], Awaitable[None]]

#: How much more mic audio a hop waits for while a sent reply has no playback start.
START_LOOKAHEAD_SAMPLES = wire.HOP_SAMPLES
#: How long, in mic samples after its audio_start, a reply's start is waited for.
START_WAIT_SAMPLES = 2 * wire.SAMPLE_RATE
#: How many replies a connection keeps the sent PCM of.
MAX_KEPT_REPLIES = 4
#: How many request ids not tracked across a listening boundary are remembered.
MAX_QUIET_REPLIES = 64
#: The most PCM kept per reply, in seconds at the reply's rate.
MAX_REPLY_SECONDS = 120
#: How many packed hops may wait for the sidecar link.
MAX_QUEUED_HOPS = 32
#: How far back from a barge's mic_sample the held audio is fed from, at most.
BARGE_LOOKBACK_SAMPLES = 4 * wire.SAMPLE_RATE
#: Held audio kept past the barge lookback and the start wait, for events that lag.
HOLD_MARGIN_SAMPLES = wire.SAMPLE_RATE


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
    speaking_end: int | None = None   # the end of the last hop cut speaking over it
    # The mic hold's own marks (see "The mic hold" in the module docstring).
    start_wait_over: bool = False     # no start came by the start wait: never held again
    barge_at: int | None = None       # a barge ended its hold here

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

    def __init__(self, label: str = "", start_wait_samples: int = START_WAIT_SAMPLES) -> None:
        self._label = label
        self._start_wait = start_wait_samples
        self._replies: collections.OrderedDict[str, _Reply] = collections.OrderedDict()
        self._mic = bytearray()       # mic PCM not yet in a hop
        self._mic_bytes = 0           # mic PCM bytes in, ever
        self._mic_frames = 0
        self._cut_to = 0              # where the last hop ended, or the last boundary
        self._listening = True        # False between a listening boundary and listen_start
        # Request ids whose playback frames are dropped quietly: replies forgotten at a
        # listening boundary and replies sent while stopped (oldest first, bounded).
        self._quiet: collections.OrderedDict[str, None] = collections.OrderedDict()

    @property
    def mic_samples(self) -> int:
        """Samples of the mic stream received so far."""
        return self._mic_bytes // 2

    @property
    def listening(self) -> bool:
        """False from a listening boundary until listen_start."""
        return self._listening

    def reply(self, request_id: str) -> _Reply | None:
        return self._replies.get(request_id)

    def replies(self) -> list[_Reply]:
        """The kept replies, oldest first."""
        return list(self._replies.values())

    def played_reply(self, mic_sample: int, onset_mic_sample: int | None = None,
                     tail_samples: int = 0,
                     hold_state: Callable[[_Reply], bool | None] | None = None,
                     ) -> _Reply | None:
        """The reply a barge at ``mic_sample`` cut in over: see "A barge" in the module
        docstring.

        A stopped reply takes a barge whose speech began by its stop
        (``onset_mic_sample``, else ``mic_sample``). ``hold_state`` gives, for a
        stopped reply, True while the mic hold still waits for the sidecar's progress
        on it, False once that hold has ended, and None when there is no such wait
        (the sidecar reports no progress). While it waits, the barge is taken; once
        it has ended, it is not; with no wait (or no ``hold_state``), the barge is
        taken when its ``mic_sample`` is by the stop plus ``tail_samples`` or by the
        end of the last hop sent in speaking mode over the reply."""
        started = [r for r in self._replies.values() if r.start is not None]
        onset = mic_sample if onset_mic_sample is None else onset_mic_sample

        def holds(reply: _Reply) -> bool:
            if mic_sample < reply.start:
                return False
            if reply.stop_mic is not None:
                if onset > reply.stop_mic:
                    return False  # the user began after the reply stopped
                state = None if hold_state is None else hold_state(reply)
                if state is not None:
                    return state
                end = max(reply.stop_mic + tail_samples, reply.speaking_end or 0)
                return mic_sample <= end
            return not reply.complete or mic_sample < reply.start + reply.length16

        holding = [r for r in started if holds(r)]
        playing = [r for r in started if r.stop_mic is None]
        if not (holding or playing):
            return None
        # max keeps the first of equal starts; the later-kept reply wins a tie.
        return max(reversed(holding or playing), key=lambda r: r.start)

    # --- the TTS side ---------------------------------------------------------------

    def on_tts_text(self, request_id: str, text: str) -> list[_Reply]:
        """A text frame a synthesis task sent: note audio_start and audio_end.

        Returns the replies whose sending this frame ended. A connection has one TTS
        request in flight at a time, so a new audio_start also ends every earlier
        reply: one that was superseded got its audio_end from the cancel path, which
        the tap does not see.
        """
        if '"audio_start"' not in text and '"audio_end"' not in text:
            return []
        try:
            frame = json.loads(text)
        except json.JSONDecodeError:
            return []
        kind = frame.get("type")
        if kind == "audio_start":
            rate = frame.get("sample_rate")
            if request_id in self._replies or type(rate) is not int or rate <= 0:
                return []
            if not self._listening:
                # Sent while the mic is stopped (a spoken goodbye): no barge can come
                # for it and it holds no mic audio, so it is not kept.
                self._keep_quiet(request_id)
                return []
            self._quiet.pop(request_id, None)
            ended = [r for r in self._replies.values() if not r.done]
            for reply in ended:
                reply.done = True
            self._replies[request_id] = _Reply(request_id=request_id, rate=rate,
                                               seen_at=self.mic_samples)
            while len(self._replies) > MAX_KEPT_REPLIES:
                self._replies.popitem(last=False)
            return ended
        if kind == "audio_end":
            reply = self.end_reply(request_id)
            return [reply] if reply is not None else []
        return []

    def end_reply(self, request_id: str) -> _Reply | None:
        """The reply's audio_end went out; the reply, when this ended its sending."""
        reply = self._replies.get(request_id)
        if reply is None or reply.done:
            return None
        reply.done = True
        return reply

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
        if reply is None and msg.request_id in self._quiet:
            logger.debug("Playback frame for a reply not tracked across a listening "
                         "boundary dropped: session=%s request_id=%s", self._label,
                         msg.request_id)
            return None
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

    def feed_mic(self, data: bytes) -> bytes:
        """One binary frame of the client's mic stream; its PCM (past any WAV header)."""
        if self._mic_frames == 0:
            offset = _wav_payload_offset(data)
            if offset is not None:
                data = data[offset:]
        self._mic_frames += 1
        self._mic += data
        self._mic_bytes += len(data)
        return data

    def due_hops(self, *, now: bool = False) -> list[_HopJob]:
        """Cut every hop whose mic audio is in and that need not wait any longer.

        ``now`` cuts every whole hop in, without the wait for a late start."""
        jobs = []
        while True:
            end = self._cut_to + wire.HOP_SAMPLES
            wait = START_LOOKAHEAD_SAMPLES if not now and self._awaiting_start() else 0
            if self.mic_samples < end + wait:
                return jobs
            jobs.append(self._cut(end))

    def _awaiting_start(self) -> bool:
        """A sent reply may still get a start that points into audio already in."""
        for reply in self._replies.values():
            if reply.start is not None or (reply.done and reply.n_samples == 0):
                continue
            if self.mic_samples - reply.seen_at < self._start_wait:
                return True
        return False

    def _cut(self, end: int) -> _HopJob:
        start = end - wire.HOP_SAMPLES
        user = bytes(self._mic[:wire.CHANNEL_BYTES])
        del self._mic[:wire.CHANNEL_BYTES]
        self._cut_to = end
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
            reply.speaking_end = end
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

    # --- the listening boundary -----------------------------------------------------

    def listen_boundary(self) -> list[_HopJob]:
        """The mic stops at ``mic_samples``: the whole hops still to go, cut now.

        Every whole hop in is cut at once, with the replies as they are; the partial
        hop after them is dropped, and the next hop starts at the boundary (the mic
        count goes on, so hop mic_samples stay on the connection's count). Every kept
        reply is forgotten: a playback frame for it is dropped quietly from now on,
        as is one for a reply sent while stopped."""
        jobs = self.due_hops(now=True)
        dropped = len(self._mic) // 2
        self._mic.clear()
        self._cut_to = self.mic_samples
        for request_id in self._replies:
            self._keep_quiet(request_id)
        self._replies.clear()
        self._listening = False
        logger.debug("Listening boundary at mic_sample=%d: session=%s whole hops=%d "
                     "partial hop samples dropped=%d", self.mic_samples, self._label,
                     len(jobs), dropped)
        return jobs

    def listen_start(self) -> None:
        """The mic is back: replies sent from now on are kept as usual."""
        self._listening = True

    def _keep_quiet(self, request_id: str) -> None:
        self._quiet[request_id] = None
        self._quiet.move_to_end(request_id)
        while len(self._quiet) > MAX_QUIET_REPLIES:
            self._quiet.popitem(last=False)

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


# --- the mic hold -----------------------------------------------------------------

_FEED, _DROP, _HOLD = "feed", "drop", "hold"


@dataclass
class _Held:
    """A run of mic audio waiting in the hold."""
    start: int            # the mic sample of its first sample
    pcm: bytes
    raw: bytes | None     # the frame as received, while the run is still a whole frame


class MicHold:
    """The mic audio between the client and the STT of one opted-in connection.

    Plain state, no I/O, read from the ``HopFeed`` it is given: ``push`` takes each
    mic frame in order, and ``take`` gives back, in order, the audio the STT may have
    now. See "The mic hold" in the module docstring for what is fed, dropped or held.
    """

    def __init__(self, feed: HopFeed, *, tail_samples: int, start_wait_samples: int) -> None:
        self._feed = feed
        self._tail = tail_samples
        self._start_wait = start_wait_samples
        self._pending_keep = start_wait_samples + HOLD_MARGIN_SAMPLES
        self._play_keep = BARGE_LOOKBACK_SAMPLES + HOLD_MARGIN_SAMPLES
        self._runs: collections.deque[_Held] = collections.deque()
        # How far the sidecar has scored the mic stream; None while it reports no
        # progress, and then a stop drops its in-play audio at once.
        self._scored: int | None = None

    def on_scored(self, mic_sample: int) -> bool:
        """The sidecar's progress: it is done with the hops up to ``mic_sample``.

        True when this passed the end of a stopped reply's play, so ``take`` may now
        let audio go that it held for it."""
        before = self._scored
        if before is not None and mic_sample <= before:
            return False
        self._scored = mic_sample
        if before is None:
            return False  # nothing waited for progress before the first line
        return any(before < self._wait_end(reply) <= mic_sample
                   for reply in self._feed.replies()
                   if reply.start is not None and reply.stop_mic is not None
                   and reply.barge_at is None)

    def hold_state(self, reply: _Reply) -> bool | None:
        """For a stopped reply: whether its hold still waits for the sidecar's progress.

        True while it waits (a barge for it may still come and be fed), False once it
        has ended (dropped, or released by a barge), None when the sidecar reports no
        progress, or when the hold never held the reply (its start came after the start
        wait), so nothing waits."""
        if self._scored is None or reply.start_wait_over:
            return None
        if reply.barge_at is not None:
            return False
        return self._waits(reply, self._feed.mic_samples)

    def _wait_end(self, reply: _Reply) -> int:
        """How far the sidecar must score before no barge can come for a stopped reply.

        The stop plus the tail, or the end of the last hop sent in speaking mode over
        the reply when that is later: the yield head may decide a barge on any frame
        of a speaking hop, and a late stop leaves hops already sent with the reply
        past the cut."""
        return max(reply.stop_mic + self._tail, reply.speaking_end or 0)

    def _waits(self, reply: _Reply, now: int) -> bool:
        """The stopped reply's in-play audio still waits for the sidecar's progress.

        False when the sidecar reports no progress (as before it could), when it has
        scored through the reply's wait end, or when the mic stream is the start wait
        past the stop plus the tail with no such progress: a slow sidecar holds the
        audio no longer than that."""
        end = reply.stop_mic + self._tail
        return (self._scored is not None and self._scored < self._wait_end(reply)
                and now < end + self._start_wait)

    @property
    def held_samples(self) -> int:
        """Mic samples in the hold now."""
        return sum(len(run.pcm) for run in self._runs) // 2

    @property
    def empty(self) -> bool:
        """Nothing is in the hold (not even a frame with no samples)."""
        return not self._runs

    def push(self, start: int, raw: bytes, pcm: bytes) -> None:
        """One mic frame: ``raw`` as received, ``pcm`` its samples from mic sample start."""
        self._runs.append(_Held(start=start, pcm=pcm, raw=raw))

    def take(self) -> list[bytes]:
        """The audio the STT gets now, in order: whole frames where they were not cut."""
        rules = self._rules(self._feed.mic_samples)
        out = []
        while self._runs:
            run = self._runs[0]
            kind, until = self._kind(run.start, rules)
            if kind == _HOLD:
                break
            n = len(run.pcm) // 2
            cut = n if until - run.start >= n else int(until - run.start)
            if cut >= n:
                self._runs.popleft()
                if kind == _FEED:
                    out.append(run.raw if run.raw is not None else run.pcm)
                continue
            head = run.pcm[:2 * cut]
            self._runs[0] = _Held(start=run.start + cut, pcm=run.pcm[2 * cut:], raw=None)
            if kind == _FEED:
                out.append(head)
        return out

    def take_all(self) -> list[bytes]:
        """Everything in the hold, in order, whatever the replies hold: the hold ends."""
        out = [run.raw if run.raw is not None else run.pcm for run in self._runs]
        self._runs.clear()
        return out

    def clear(self) -> int:
        """Drop everything in the hold; the samples dropped."""
        n = self.held_samples
        self._runs.clear()
        return n

    def _rules(self, now: int) -> list[tuple[int, float, str]]:
        """(from, to, kind) spans of the mic stream every kept reply holds or drops."""
        rules: list[tuple[int, float, str]] = []
        for reply in self._feed.replies():
            if reply.start_wait_over:
                continue
            if reply.start is None:
                if reply.barge_at is not None or (reply.done and reply.n_samples == 0):
                    continue  # it never plays
                rules.append((max(reply.seen_at, now - self._pending_keep), math.inf, _HOLD))
                continue
            start = reply.start
            end = None if reply.stop_mic is None else reply.stop_mic + self._tail
            if reply.barge_at is not None:
                end = reply.barge_at if end is None else min(end, reply.barge_at)
            elif end is not None and self._waits(reply, now):
                # Stopped, but the sidecar may still report a barge for it.
                keep_from = max(start, now - self._play_keep)
                if keep_from > start:
                    rules.append((start, min(keep_from, end), _DROP))
                if end > keep_from:
                    rules.append((keep_from, end, _HOLD))
                continue
            if end is not None:
                if end > start:
                    rules.append((start, end, _DROP))
                continue
            # Still playing: a barge may yet come for the last BARGE_LOOKBACK of it.
            # A reply that ended with no stop yet plays at most its length plus the
            # start wait, so a client that never reports its stop costs only that.
            open_end = (start + reply.length16 + self._start_wait) if reply.done else math.inf
            keep_from = max(start, now - self._play_keep)
            if keep_from > start:
                rules.append((start, min(keep_from, open_end), _DROP))
            if open_end > keep_from:
                rules.append((keep_from, open_end, _HOLD))
        return rules

    @staticmethod
    def _kind(at: int, rules: list[tuple[int, float, str]]) -> tuple[str, float]:
        """What mic sample ``at`` is (hold over drop over feed), and up to where."""
        kind, until = _FEED, math.inf
        for lo, hi, what in rules:
            if lo <= at < hi:
                if what == _HOLD or (what == _DROP and kind == _FEED):
                    kind = what
                until = min(until, hi)
            elif lo > at:
                until = min(until, lo)
        return kind, until


def _asking_progress(url: str) -> str:
    """The session URL with the query parameter that asks for progress lines."""
    parts = urllib.parse.urlsplit(url)
    query = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query.append((wire.PROGRESS_PARAM, "1"))
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


class TurnEngineSession:
    """One connection's turn engine: opt-in checks, sidecar link and event reader."""

    def __init__(self, *, sidecar, stt_engine_type: str, label: str,
                 send_text: SendText, on_event: EventHandler | None = None,
                 end_turn: EndTurn | None = None, release_held: ReleaseHeld | None = None,
                 hold_tail_ms: int = 150, start_wait_ms: int = 2000) -> None:
        """``sidecar`` is app.state.turn_engine_sidecar (None when the setting is off):
        anything with ``up`` and ``session_url(label)``. ``label`` names this
        connection's session on the sidecar (the connection's session id).
        ``send_text`` sends one text frame to the client. ``on_event`` gets each
        event the sidecar sends; without one, events are only logged. ``end_turn``
        acts on an endpoint event: it flushes the open STT utterance and then calls
        ``send_turn_end``, both serialized with the handler's STT work.
        ``release_held`` feeds the STT what ``take_stt_audio`` gives, under the
        handler's STT lock; it is called when the hold may have let audio go
        without a new mic frame (a playback frame, a barge, the start wait). The
        hold keeps ``hold_tail_ms`` past a stop and waits ``start_wait_ms`` after
        a reply's audio_end for its playback start."""
        self._sidecar = sidecar
        self._stt_engine_type = stt_engine_type
        self._label = label
        self._send_text = send_text
        self._on_event = on_event
        self._end_turn = end_turn
        self._release_held = release_held
        self._start_wait_s = start_wait_ms / 1000.0
        self._finals: list[tuple[str, int]] = []  # (text, turn_id) since the last turn_end
        self.engine: str | None = None
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._closed = False
        start_wait_samples = start_wait_ms * wire.SAMPLE_RATE // 1000
        self._feed = HopFeed(label, start_wait_samples=start_wait_samples)
        self._tail_samples = hold_tail_ms * wire.SAMPLE_RATE // 1000
        self._hold = MicHold(self._feed, tail_samples=self._tail_samples,
                             start_wait_samples=start_wait_samples)
        self._timers: set[asyncio.TimerHandle] = set()
        self._releases: set[asyncio.Task] = set()
        self._hop_queue: asyncio.Queue | None = None
        self._sender: asyncio.Task | None = None
        self._dropping = False        # hops are being dropped on a full queue
        self._tts_session = None      # the connection's TTS session, once attached
        self._lost = False            # the sidecar was lost: see "Losing the sidecar"
        self._generation = None       # the sidecar generation the link was opened on
        self._remove_listener: Callable[[], None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        # The mic_sample of the last listening boundary: no event at or before it is
        # acted on. None while the connection never stopped listening.
        self._boundary: int | None = None

    @property
    def active(self) -> bool:
        """The connection opted in, its sidecar link is open and it was not lost."""
        return self._ws is not None and not self._closed and not self._lost

    @property
    def lost(self) -> bool:
        """The connection opted in and then lost its sidecar."""
        return self._lost

    @property
    def holds_mic(self) -> bool:
        """The handler routes mic frames through the hold (``feed_mic`` and
        ``take_stt_audio``): while active, and after a loss until the hold is empty."""
        if self.active:
            return True
        return self._lost and not self._closed and not self._hold.empty

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
        if self._lost:
            return (INVALID_REQUEST,
                    "This connection's turn engine was lost; open a new connection to "
                    "opt in again.")
        if self.engine is not None:
            return (INVALID_REQUEST,
                    "This connection already opted in to a turn engine.")
        if not self._sidecar.up:
            return (TURN_ENGINE_UNAVAILABLE,
                    "The turn engine is not running right now; try again later.")
        return None

    # --- playback, the TTS audio tap and the mic stream ------------------------------

    async def handle_playback(self, frame: dict) -> None:
        """Apply one playback frame, or answer it with an invalid_request error frame.

        While the mic is stopped every playback frame is dropped quietly: no barge
        can come with no mic audio, so no reply is tracked."""
        if not self._feed.listening:
            logger.debug("Playback frame while stopped dropped: session=%s %s %s",
                         self._label, frame.get("request_id"), frame.get("event"))
            return
        refusal = self._feed.playback(frame)
        if refusal is not None:
            logger.info("Playback frame refused: session=%s (%s)", self._label, refusal)
            await self._send_error(INVALID_REQUEST, refusal)
            return
        logger.debug("Playback: session=%s %s %s at mic_sample=%s", self._label,
                     frame.get("request_id"), frame.get("event"), frame.get("mic_sample"))
        self._queue_due_hops()
        await self._release()  # a start frees audio before it; a stop, audio past its tail

    def attach_tts_session(self, session) -> None:
        """The TTS session this object taps; called when it is set as the audio tap."""
        self._tts_session = session

    def on_tts_text(self, request_id: str, text: str) -> None:
        """The TTS session's audio tap: a text frame a reply sent."""
        if self.active:
            for reply in self._feed.on_tts_text(request_id, text):
                self._reply_ended(reply)

    def on_tts_cancel(self, request_id: str) -> None:
        """The client cancelled request_id: its audio_end went out from the cancel path."""
        if self.active:
            reply = self._feed.end_reply(request_id)
            if reply is not None:
                self._reply_ended(reply)

    def _reply_ended(self, reply: _Reply) -> None:
        """No more of reply goes out: start its start wait, if it has no start yet."""
        if reply.start is not None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if reply.n_samples == 0:
            self._schedule_release()  # it never plays, so it holds nothing now
            return
        handle = None

        def over() -> None:
            self._timers.discard(handle)
            self._start_wait_over(reply)

        handle = loop.call_later(self._start_wait_s, over)
        self._timers.add(handle)

    def _start_wait_over(self, reply: _Reply) -> None:
        if self._closed or reply.start is not None or reply.start_wait_over:
            return
        reply.start_wait_over = True
        logger.info("No playback start for request_id=%s within the start wait; its held "
                    "audio goes to the STT: session=%s", reply.request_id, self._label)
        self._schedule_release()

    def _schedule_release(self) -> None:
        task = asyncio.get_running_loop().create_task(self._release())
        self._releases.add(task)
        task.add_done_callback(self._releases.discard)

    async def _release(self) -> None:
        """Have the handler feed the STT whatever the hold lets go now."""
        if self._release_held is None or not self.active:
            return
        try:
            await self._release_held()
        except Exception:
            logger.exception("Feeding held mic audio to the STT failed: session=%s",
                             self._label)

    def take_stt_audio(self) -> list[bytes]:
        """The mic audio the STT gets now, in order. Call it under the STT lock.

        After a loss it is everything still in the hold, which ends the hold."""
        if self._lost:
            return self._hold.take_all()
        return self._hold.take()

    def on_tts_pcm(self, request_id: str, pcm: bytes) -> None:
        """The TTS session's audio tap: PCM a reply sent."""
        if self.active:
            self._feed.on_tts_pcm(request_id, pcm)

    def feed_mic(self, data: bytes) -> None:
        """One binary frame of the client's mic stream. Never waits on the sidecar.

        The hop feed gets it at once; the STT gets it through ``take_stt_audio``.
        After a loss (while ``holds_mic``) it only queues behind the held audio.
        """
        if self._lost:
            self._hold.push(self._feed.mic_samples, data, data)
            return
        at = self._feed.mic_samples
        self._hold.push(at, data, self._feed.feed_mic(data))
        self._queue_due_hops()

    def _queue_due_hops(self) -> None:
        if self._hop_queue is None:
            return
        self._queue_hops(self._feed.due_hops())

    def _queue_hops(self, jobs: list[_HopJob]) -> None:
        for job in jobs:
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
                if isinstance(job, str):
                    await ws.send(job)  # a control message, in order with the hops
                    continue
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

    # --- the listening boundary -----------------------------------------------------

    def listen_boundary(self) -> None:
        """The client stopped listening: see "The listening boundary" in the module
        docstring. Call it on a real listening -> stopped change, before the answer
        goes out and before the STT is reset; it never waits."""
        dropped = self._hold.clear()  # pre-stop audio never reaches the STT after it
        for handle in self._timers:
            handle.cancel()
        self._timers.clear()
        self._finals.clear()
        if not self.active:
            return  # never opted in, or lost: no sidecar to tell
        boundary = self._feed.mic_samples
        self._boundary = boundary
        if self._hop_queue is not None:
            self._queue_hops(self._feed.listen_boundary())
            self._queue_control(wire.format_boundary_control())
        else:
            self._feed.listen_boundary()
        logger.info("Turn engine listening boundary: session=%s mic_sample=%d "
                    "held samples dropped=%d", self._label, boundary, dropped)

    def listen_start(self) -> None:
        """The client listens again; replies sent from now on are tracked as usual."""
        self._feed.listen_start()

    def _queue_control(self, message: str) -> None:
        """Queue a control message behind every hop already queued; it is never
        dropped: on a full queue the oldest hop goes instead."""
        queue = self._hop_queue
        if queue.full():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            logger.warning("Turn engine hop queue full at a control message; dropped "
                           "the oldest hop: session=%s", self._label)
        queue.put_nowait(message)

    def _before_boundary(self, mic_sample: int) -> bool:
        """An event at mic_sample is about audio from before a listening boundary."""
        return self._boundary is not None and mic_sample <= self._boundary

    # --- the turn end ---------------------------------------------------------------

    def note_final(self, text: str, turn_id: int) -> None:
        """A final transcription the connection sent, with its turn_id."""
        text = text.strip()
        if self.active and text:
            self._finals.append((text, turn_id))

    async def send_turn_end(self, event: wire.EndpointEvent) -> bool:
        """Send the turn_end for every final since the previous one; False when none.

        Nothing goes out once the sidecar is lost, nor for an event at or before a
        listening boundary (and the finals since the boundary stay for the next)."""
        if self._before_boundary(event.mic_sample):
            logger.debug("Turn end from before a listening boundary, nothing sent: "
                         "session=%s mic_sample=%s", self._label, event.mic_sample)
            return False
        finals, self._finals = self._finals, []
        if self._lost:
            logger.debug("Turn end after the turn engine was lost, nothing sent: "
                         "session=%s", self._label)
            return False
        if not finals:
            logger.debug("Turn end with no final text, nothing sent: session=%s by=%s "
                         "mic_sample=%s", self._label, event.by, event.mic_sample)
            return False
        msg = TurnEndMessage(turn_id=finals[-1][1], text=" ".join(t for t, _ in finals),
                             mic_sample=event.mic_sample, by=event.by)
        logger.info("Turn end: session=%s turn_id=%s by=%s mic_sample=%s finals=%d",
                    self._label, msg.turn_id, msg.by, msg.mic_sample, len(finals))
        await self._send_text(msg.model_dump_json())
        return True

    async def _handle_endpoint(self, event: wire.EndpointEvent) -> None:
        """End the turn. A failure is logged and the reader goes on."""
        try:
            if self._end_turn is not None:
                await self._end_turn(event)
            else:
                await self.send_turn_end(event)
        except Exception:
            logger.exception("Handling a turn end failed: session=%s", self._label)

    # --- the sidecar link -----------------------------------------------------------

    async def _connect(self) -> bool:
        """Open this connection's sidecar WebSocket and start its reader."""
        url = _asking_progress(self._sidecar.session_url(self._label))
        try:
            self._ws = await websockets.asyncio.client.connect(
                url, open_timeout=CONNECT_TIMEOUT_S, close_timeout=CLOSE_TIMEOUT_S)
        except (OSError, asyncio.TimeoutError, websockets.exceptions.WebSocketException) as exc:
            logger.warning("Turn engine sidecar connect failed: session=%s url=%s: %s",
                           self._label, url, exc)
            self._ws = None
            return False
        self._loop = asyncio.get_running_loop()
        self._generation = getattr(self._sidecar, "generation", None)
        add_listener = getattr(self._sidecar, "add_listener", None)
        if callable(add_listener):
            self._remove_listener = add_listener(self._on_sidecar_change)
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
        # The link closed and this session did not close it (aclose cancels the reader).
        self._lose("its connection to the turn engine sidecar closed")

    def _on_sidecar_change(self, up: bool, generation: int) -> None:
        """The sidecar's listener: a loss of the generation this link was opened on."""
        if up or (self._generation is not None and generation != self._generation):
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        why = "the turn engine sidecar exited"
        if running is loop:
            self._lose(why)
        else:
            loop.call_soon_threadsafe(self._lose, why)

    def _lose(self, why: str) -> None:
        """Degrade to a session that never opted in: see "Losing the sidecar"."""
        if self._lost or self._closed or self._ws is None:
            return
        self._lost = True
        logger.warning("Turn engine lost: session=%s (%s); the session goes on without it",
                       self._label, why)
        if self._remove_listener is not None:
            self._remove_listener()
            self._remove_listener = None
        for handle in self._timers:
            handle.cancel()
        self._timers.clear()
        self._finals.clear()
        tts = self._tts_session
        if tts is not None and tts.audio_tap is self:
            tts.audio_tap = None
        sender, self._sender = self._sender, None
        self._hop_queue = None
        if sender is not None:
            sender.cancel()
        task = asyncio.get_running_loop().create_task(self._after_loss())
        self._releases.add(task)
        task.add_done_callback(self._releases.discard)

    async def _after_loss(self) -> None:
        """Tell the client, then feed the STT everything the hold keeps, in order."""
        try:
            await self._send_error(
                TURN_ENGINE_LOST,
                "The turn engine stopped; the session goes on without it (silero "
                "endpointing, no barge or turn_end frames).")
        except Exception:
            logger.debug("Sending turn_engine_lost failed: session=%s", self._label,
                         exc_info=True)
        if self._release_held is None or self._hold.empty:
            return
        try:
            await self._release_held()
        except Exception:
            logger.exception("Feeding held mic audio to the STT failed: session=%s",
                             self._label)

    async def _dispatch(self, event: wire.Event) -> None:
        logger.debug("Turn engine event: session=%s %s", self._label, event)
        if self._lost:
            return  # the session goes on as one that never opted in
        if (isinstance(event, (wire.BargeEvent, wire.EndpointEvent))
                and self._before_boundary(event.mic_sample)):
            logger.debug("Turn engine event from before a listening boundary ignored: "
                         "session=%s %s", self._label, event)
            return
        if isinstance(event, wire.BargeEvent):
            await self._on_barge(event)
        elif isinstance(event, wire.ScoredEvent):
            # Every event up to here came before it: a stopped reply's hold may end.
            if self._hold.on_scored(event.mic_sample) and not self._hold.empty:
                await self._release()
        elif isinstance(event, wire.EndpointEvent):
            await self._handle_endpoint(event)
        if self._on_event is not None:
            await self._on_event(event)

    async def _on_barge(self, event: wire.BargeEvent) -> None:
        """Stop the reply the user cut in over: see "A barge" in the module docstring."""
        reply = self._feed.played_reply(event.mic_sample, event.onset_mic_sample,
                                        self._tail_samples, self._hold.hold_state)
        if reply is None:
            logger.debug("Barge with no played reply ignored: session=%s at mic_sample=%s",
                         self._label, event.mic_sample)
            return
        notice = BargeMessage(request_id=reply.request_id,
                              onset_mic_sample=event.onset_mic_sample,
                              mic_sample=event.mic_sample).model_dump_json()
        logger.info("Barge: session=%s request_id=%s onset_mic_sample=%s mic_sample=%s",
                    self._label, reply.request_id, event.onset_mic_sample, event.mic_sample)
        # The user talks from the onset: no reply sent by now holds the mic from there.
        # Marked before any await, so the next mic frame already sees it.
        feed_from = max(event.onset_mic_sample, event.mic_sample - BARGE_LOOKBACK_SAMPLES)
        for kept in self._feed.replies():
            if kept.seen_at <= event.mic_sample and kept.barge_at is None:
                kept.barge_at = feed_from
        tts = self._tts_session
        if reply.done or tts is None:
            # Its audio_end already went out: the frame alone clears the speaker.
            await self._send_text(notice)
        else:
            # No more of it goes out; its audio_end goes on the real socket, unseen by
            # the tap.
            reply.done = True
            await tts.cancel(reply.request_id, reason="interrupted", notice=notice)
        await self._release()

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
        if self._remove_listener is not None:
            self._remove_listener()
            self._remove_listener = None
        for handle in self._timers:
            handle.cancel()
        self._timers.clear()
        for task in self._releases:
            task.cancel()
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
