"""Per-connection TTS session: the single owner of a connection's in-flight request.

`TTSSession` is the one place that owns TTS synthesis for a
connection. Both protocol paths dispatch through it -- the one-shot `tts` request
(`speak`) and the incremental `tts_stream` request (`feed_stream`) -- so "at most
ONE TTS request in flight per connection" is enforced structurally instead of by
convention scattered across the receive loop.

That tripwire is load-bearing. It is what makes the client rule "last `audio_start`
wins, discard binary until `audio_end`" complete and sufficient, and so what lets
binary PCM frames stay bare. Before this seam existed the receive loop
held the one-shot task while `TTSStreamPipeline` held the streaming one and neither
stopped the other, so a `tts` after a `tts_stream` left BOTH synthesizing onto the
same socket. `live_tasks` is that invariant in observable form.

The session is also deliberately *callable*: every method here can be driven with a
fake WebSocket and no receive loop present. A future server-side barge-in fires from
the STT path, not from a `tts_cancel` frame, and needs somewhere to call.

Cancellation rests on one hard fact: **an executor thread cannot be
interrupted.** `task.cancel()` does not stop a synthesis that is blocked inside
`run_in_executor` -- the task keeps sitting there until the thread finishes on its own,
and only then does it notice it was cancelled. So a cancel path may never *await* the
task it just cancelled, and it must assume the doomed task will wake up later and try
to finish speaking. Two mechanisms follow:

  * `cancel()` acks on the spot and cuts the task loose, unawaited.
  * every synthesis task writes to a `_GuardedSocket`, not to the real WebSocket, and
    that socket goes silent the moment its request is cancelled. Whatever the unwinding
    task still tries to send -- PCM, its own trailing `audio_end` -- is dropped.

A cancelled request_id is then **permanently dead**: a `tts`/`tts_stream`
frame carrying it is ignored outright, so a token that was already on the wire when the
client cancelled cannot re-open the request the client just killed. That hazard is real --
HermesConsole carries a `@Synchronized` monitor built solely to work around it. Dead ids
live in `_dead_ids`, a fixed-capacity ring: a long-lived socket cancels indefinitely, so
the store must not grow without bound, and the recent window is the only one where a stale
token can still be in flight.
"""
import asyncio
import collections
import functools
import logging

from voicemanager.pipeline.tts import TTSPipeline, TTSStreamPipeline
from voicemanager.protocol import AudioEndMessage

logger = logging.getLogger(__name__)

#: How many cancelled request_ids stay dead. It must cover at least the 32 most
#: recent; a stale token can only still be on the wire for the request that was just
#: cancelled, so this window is generous. Bounded because the socket is long-lived.
DEAD_ID_CAPACITY = 64


class _GuardedSocket:
    """The socket a synthesis task speaks through. Goes silent once its request is cancelled.

    A cancelled synthesis does NOT stop when told to — it is blocked in an executor
    thread and will come back, sometimes seconds later, still trying to send its audio
    and its own `audio_end`. Routing it through here makes "no PCM ever follows audio_end"
    and "exactly ONE audio_end per request_id" true by construction:
    the frames are dropped at the door rather than raced against.
    """

    def __init__(self, websocket, request_id: str, is_cancelled):
        self._websocket = websocket
        self._request_id = request_id
        self._is_cancelled = is_cancelled  # callable: (request_id) -> bool

    async def send_text(self, data: str) -> None:
        if self._is_cancelled(self._request_id):
            logger.debug("Dropped text frame from cancelled request_id=%s", self._request_id)
            return
        await self._websocket.send_text(data)

    async def send_bytes(self, data: bytes) -> None:
        if self._is_cancelled(self._request_id):
            logger.debug("Dropped %d PCM bytes from cancelled request_id=%s",
                         len(data), self._request_id)
            return
        await self._websocket.send_bytes(data)


class TTSSession:
    """Owns the one in-flight TTS request for a single WebSocket connection."""

    def __init__(
        self,
        websocket,
        tts_pipeline: TTSPipeline,
        tts_stream_pipeline: TTSStreamPipeline | None = None,
    ):
        self._websocket = websocket
        self._tts_pipeline = tts_pipeline
        self._stream_pipeline = tts_stream_pipeline
        self._task: asyncio.Task | None = None      # the one-shot `tts` task
        self._stream_request_id: str | None = None  # open `tts_stream` request, if any
        self._active_request_id: str | None = None  # the request the live task belongs to
        self._unwinding: list[asyncio.Task] = []    # cancelled, still finishing in a thread
        # The request_ids that are dead for the life of this connection. A ring, not
        # a set — this grows once per cancel on a socket that may live for hours.
        self._dead_ids: collections.deque[str] = collections.deque(maxlen=DEAD_ID_CAPACITY)
        # The request_ids that finished on their own and already sent their ONE
        # audio_end(reason='complete'). A late cancel of such an id must be a no-op, not a
        # second audio_end. Same bounded ring shape as _dead_ids, and for the same reason —
        # a long-lived socket completes indefinitely, so it must not grow without bound.
        self._completed_ids: collections.deque[str] = collections.deque(maxlen=DEAD_ID_CAPACITY)

    # --- introspection --------------------------------------------------

    @property
    def live_tasks(self) -> list[asyncio.Task]:
        """Every synthesis task currently in flight - the single-flight tripwire, observable.

        MUST never hold more than one entry: both dispatch paths stop the other
        before starting. Asserted directly in tests/test_tts_cancel.py.
        """
        tasks = []
        if self._task is not None and not self._task.done():
            tasks.append(self._task)
        if self._stream_pipeline is not None:
            stream_task = self._stream_pipeline.synthesis_task
            if stream_task is not None and not stream_task.done():
                tasks.append(stream_task)
        return tasks

    @property
    def active_stream_request_id(self) -> str | None:
        """The open `tts_stream` request, or None. Arms the receive loop's flush timeout."""
        return self._stream_request_id

    async def wait_idle(self) -> None:
        """Await everything in flight AND everything still unwinding after a cancel.

        The unwinding tasks matter: they are the ones blocked in an executor thread that
        will wake up and try to keep speaking. Anything asserting "nothing followed the
        audio_end" has to give them their chance to misbehave first.
        """
        for task in [*self.live_tasks, *self._unwinding]:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._unwinding = [t for t in self._unwinding if not t.done()]

    def _is_cancelled(self, request_id: str) -> bool:
        """True once this id has been cancelled — permanently, for this connection."""
        return request_id in self._dead_ids

    def _mark_dead(self, request_id: str) -> None:
        """Kill this request_id for good. Evicts the oldest id once the ring is full."""
        if request_id not in self._dead_ids:
            self._dead_ids.append(request_id)

    def _is_completed(self, request_id: str) -> bool:
        """True once this id finished on its own and sent its audio_end('complete')."""
        return request_id in self._completed_ids

    def _mark_completed(self, request_id: str) -> None:
        """Record that this id already reached its terminal audio_end via normal completion."""
        if request_id not in self._completed_ids:
            self._completed_ids.append(request_id)

    def _on_synthesis_done(self, request_id: str, task: asyncio.Task) -> None:
        """Done-callback fired when a synthesis task finishes.

        A task that COMPLETED normally has already sent its own audio_end('complete'), so
        the id is terminal: a later cancel of it must be a no-op, not a second audio_end.
        A task that was CANCELLED (a tts_cancel or a supersede) is skipped — cancel() has
        already acked and marked it dead — and so is one that errored out (no clean
        audio_end went out to be a duplicate of).

        Attached at dispatch, BEFORE anything awaits the task, so it runs ahead of any
        waiter (wait_idle) and the id is marked the instant the task settles.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            # No clean audio_end('complete') went out, so this id is NOT terminal. Log it
            # rather than let it vanish — reading exception() here retrieves it, which would
            # otherwise suppress asyncio's "Task exception was never retrieved" diagnostic.
            logger.warning("TTS synthesis for request_id=%s failed: %r", request_id, exc)
            return
        self._mark_completed(request_id)

    def _socket_for(self, request_id: str) -> _GuardedSocket:
        """The socket this request's synthesis task speaks through — silenced if cancelled."""
        return _GuardedSocket(self._websocket, request_id, self._is_cancelled)

    # --- dispatch -------------------------------------------------------

    async def speak(self, text: str, request_id: str, voice: str | None = None) -> None:
        """Dispatch a one-shot `tts` request.

        Supersedes anything already in flight — including a streaming request, which
        the old receive loop left running alongside it — down the cancel path
        (see `_supersede`).
        """
        # A dead id is dead. Checked BEFORE stopping anything, or a stale frame
        # would kill the request that is current now on its way to being ignored.
        if self._is_cancelled(request_id):
            logger.debug("Ignored tts for cancelled request_id=%s", request_id)
            return

        await self._supersede(request_id)
        await self._stop_all()  # a same-id restart, or a task that already finished
        self._active_request_id = request_id
        self._task = asyncio.create_task(
            self._tts_pipeline.synthesize_and_stream(
                websocket=self._socket_for(request_id),
                text=text,
                request_id=request_id,
                tgt_se=self._tts_pipeline.target_se,  # per-session voice profile
                voice=voice,                          # voice parameter routing
            )
        )
        # Mark the id completed the instant this task finishes normally, so a late
        # cancel of it is a no-op. Attached now, before any await, so it runs before wait_idle.
        self._task.add_done_callback(functools.partial(self._on_synthesis_done, request_id))

    async def feed_stream(
        self,
        chunk: str,
        done: bool,
        request_id: str,
        voice: str | None = None,
        flush: bool = False,
    ) -> None:
        """Feed one `tts_stream` chunk.

        A chunk opening a NEW request supersedes anything in flight — one-shot or stream
        — down the cancel path (see `_supersede`), which is also what keeps the
        single-flight invariant.
        """
        if self._stream_pipeline is None:
            return

        # The token that was already on the wire when the client cancelled. It
        # must not reopen the request, and — checked before we stop anything — it must
        # not disturb whatever request is current now either.
        if self._is_cancelled(request_id):
            logger.debug("Ignored tts_stream chunk for cancelled request_id=%s", request_id)
            return

        await self._supersede(request_id)
        if request_id != self._stream_request_id:
            await self._stop_oneshot()  # a one-shot that already finished on its own

        # A flush leaves the request OPEN, so the id stays set and the receive
        # loop's flush-timeout safety path stays armed. Only done=true clears it.
        self._stream_request_id = request_id
        self._active_request_id = request_id
        # feed_text starts a fresh synthesis task on the first chunk of a new
        # request. Watch for that new task and mark the id completed when it finishes
        # normally (via done=true or the flush timeout), so a late cancel of it is a no-op.
        previous_task = self._stream_pipeline.synthesis_task
        await self._stream_pipeline.feed_text(
            chunk=chunk,
            done=done,
            websocket=self._socket_for(request_id),
            request_id=request_id,
            voice=voice,
            flush=flush,
        )
        new_task = self._stream_pipeline.synthesis_task
        if new_task is not None and new_task is not previous_task:
            new_task.add_done_callback(functools.partial(self._on_synthesis_done, request_id))
        if done:
            self._stream_request_id = None

    async def flush_timeout(self) -> None:
        """Safety flush of the open streaming request when no chunk arrived in time."""
        request_id = self._stream_request_id
        if request_id is None or self._stream_pipeline is None:
            return
        await self._stream_pipeline.flush_timeout(self._socket_for(request_id), request_id)
        self._stream_request_id = None

    # --- cancellation ---------------------------------------------------

    async def cancel(self, request_id: str) -> None:
        """Stop the in-flight request and ack it at once.

        Callable by ANY caller — a `tts_cancel` frame today, the STT path when
        server-side barge-in arrives. It never awaits the cancelled synthesis:
        that task is blocked in an executor thread nobody can interrupt, so awaiting it
        would hold the caller for the rest of the synthesis (measured 858-1089ms) and,
        on the receive path, stop the server reading the very audio that barged in.

        So: silence the request, send its `audio_end(reason='cancelled')` immediately,
        and cut the doomed task loose. It will wake up later and try to send its audio
        and its own `audio_end` — and its guarded socket will drop every frame, which is
        what leaves the client with exactly one `audio_end` and no PCM after it.

        Scoped to the in-flight request: the task teardown fires only when
        `request_id` is the one actually synthesizing (`_active_request_id`). Cancelling any
        other id — client misuse, or a never-seen id — still acks and marks it dead, but must
        NOT touch the live request; killing the reply the client did not cancel would strand
        it with an `audio_start` and no `audio_end`. The supersede path always cancels
        `_active_request_id`, so it still tears down.

        Idempotent and safe for an id that never started. Cancelling an id the
        server has not seen yet still acks and still kills it, which is what lets a client
        cancel unconditionally without racing its own in-flight tokens. Cancelling an id
        that already reached a terminal `audio_end` is a no-op — whether it was cancelled
        before, or finished on its own and sent `audio_end('complete')` (the barge-in race:
        the last word completes just as the client's cancel fires). Either way the one ack
        already went out, and at most ONE `audio_end` is ever emitted per request_id over
        the life of a connection.
        """
        # Global invariant: at most ONE audio_end per request_id over the life of a
        # connection. An id that already reached a terminal audio_end — cancelled earlier,
        # or finished on its own and sent audio_end('complete') — must not get a second one.
        # (A genuinely never-seen id is NOT terminal, so it still acks below.)
        if self._is_cancelled(request_id):
            logger.debug("TTS cancel for already-cancelled request_id=%s (no-op)", request_id)
            return
        if self._is_completed(request_id):
            logger.debug("TTS cancel for already-completed request_id=%s (no-op)", request_id)
            return

        # Only the request whose synthesis is actually in flight may have its task
        # torn down. A cancel of an id that is NOT the in-flight one -- client misuse, or
        # a never-seen id -- is still acked and marked dead below, but must leave the
        # live task (and every field that tracks it) untouched: killing r1 because the client
        # cancelled rX would strand r1 with an audio_start and no audio_end, breaking
        # single-flight ordering. The supersede path always calls cancel(_active_request_id),
        # so it still tears down here. Decided BEFORE the ack await so a task settling during
        # the send cannot change which branch we take.
        is_active = request_id == self._active_request_id
        tasks = self.live_tasks if is_active else []
        self._mark_dead(request_id)  # from here on, that request cannot reach the client

        # The ack goes out on the REAL socket: the request is dead now, so a guarded
        # socket would drop the very frame that tells the client so.
        await self._websocket.send_text(
            AudioEndMessage(request_id=request_id, reason="cancelled").model_dump_json()
        )

        if not is_active:
            logger.info("TTS cancel for non-active request_id=%s (acked and marked dead; "
                        "in-flight request left untouched)", request_id)
            return

        for task in tasks:
            task.cancel()
        self._unwinding.extend(tasks)
        # `_unwinding` must not grow without bound on a long-lived socket. The
        # doomed tasks a barge-in cuts loose DO finish on their own (the executor thread
        # runs to completion), they are just never awaited here. Prune the finished ones
        # now — wait_idle() only prunes from tests, and the production receive loop never
        # calls it, so this cancel path is the only place a real socket ever prunes.
        self._unwinding = [t for t in self._unwinding if not t.done()]
        self._task = None
        if self._stream_pipeline is not None:
            self._stream_pipeline.abandon()  # cancels without awaiting
        self._stream_request_id = None
        self._active_request_id = None
        logger.info("TTS cancelled: request_id=%s (acked; executor unwinds in background)",
                    request_id)

    async def _supersede(self, request_id: str) -> None:
        """A new request_id arriving over a live one IS a cancel — take that path.

        A client that starts r2 while r1 is still speaking has cancelled r1; it just did
        not say so in a `tts_cancel` frame. So r1 gets exactly what the frame would have
        given it: an immediate `audio_end(reason='cancelled')`, a permanently dead id,
        and a doomed task cut loose rather than awaited.

        That last part is the point. This runs on the receive path, and the old supersede
        awaited the previous synthesis task — blocked in an executor thread nobody can
        interrupt — so the server read nothing at all for the rest of the synthesis,
        including the audio of the human who just barged in. Superseding is the
        door a barging-in client is most likely to come through, so it cannot be the one
        door that still stalls.
        """
        previous = self._active_request_id
        if previous is None or previous == request_id:
            return
        if not self.live_tasks:
            # It finished on its own and has already sent its own audio_end. Cancelling it
            # now would emit a second one for that id.
            self._active_request_id = None
            return
        await self.cancel(previous)

    async def aclose(self) -> None:
        """Teardown for the WS finally-block: nothing is left synthesizing."""
        await self._stop_all()
        for task in self._unwinding:
            if not task.done():
                task.cancel()
        self._unwinding = []

    # --- internals ------------------------------------------------------

    async def _stop_oneshot(self) -> None:
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def _stop_stream(self) -> None:
        if self._stream_pipeline is not None:
            await self._stream_pipeline.cancel()
        self._stream_request_id = None

    async def _stop_all(self) -> None:
        await self._stop_oneshot()
        await self._stop_stream()
