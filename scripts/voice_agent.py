"""Reference voice agent over VoiceManager: the classic turn pattern and the micro-turn one.

Three subcommands, one per mode:

- ``classic``: wait for the person's turn, send it to any OpenAI-compatible chat
  endpoint, stream the reply through VoiceManager's TTS, and stop the reply when the
  person speaks over it. The turn ends on the server's endpoint frame (or the final
  transcript on engines without one); if the final revises the text, the reply is
  cancelled and the revision is sent instead. With ``--turn-engine tyger`` the
  connection opts in to the server's turn engine: the turn ends on its turn_end
  frame, a barge frame stops the reply, and the client reports when each reply's
  first and last sample reached the speaker, placed on its own mic stream by the
  PortAudio output DAC time and input ADC time.
- ``microturn``: the experimental full-duplex loop. Every 0.6 s the LLM with the
  micro-turn LoRA gets the words heard in that window and answers with a few words to
  speak or ``listen``. Needs the LoRA model under llama-server or llama-swap.
- ``replay``: the micro-turn loop on a scripted call as text, no audio.

Both live modes stream the microphone and play the speaker through the default audio
device: use headphones, there is no echo cancellation. They write a JSON lines log and
a WAV of what played under ``runs/``; ``--capture`` also saves the mic as captured,
the output timeline and both start times beside the WAV. The bearer token comes from the
``VOICEMANAGER_TOKEN`` environment variable.

    python scripts/voice_agent.py classic [--vm-host HOST] [--vm-port PORT] [--voice NAME]
        [--url URL] [--model NAME] [--tts-mode {clause,sentence}] [--think]
        [--duration SECONDS] [--turn-engine tyger] [--capture]
    python scripts/voice_agent.py microturn [...the options of the micro-turn live mode]
    python scripts/voice_agent.py replay [--script FILE] [--url URL] [--model NAME]
"""
import argparse
import asyncio
import json
import logging
import os
import pathlib
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import microturn_client  # noqa: E402
from microturn_client import (  # noqa: E402
    AudioPlayer, MicCapture, _heard_words, write_capture, write_wav)

logger = logging.getLogger("voice_agent")

SYSTEM_PROMPT = ("You are a voice assistant talking with a person. Answer in one or two "
                 "short spoken sentences of plain text: no lists, no markdown, no headings.")
CLASSIC_RUN_DIR = pathlib.Path("runs") / "classic"
TURN_ENGINES = ("tyger",)
MIC_RATE = MicCapture.SAMPLE_RATE


# --- playback edges on the mic clock ---------------------------------------------------

class MicClock:
    """The first mic block's times, without keeping the audio (``MicCapture`` keeps it).

    ``_mic_task`` hands every block to ``add``; only the first one's PortAudio ADC time
    and ``time.monotonic()`` are kept. Sample 0 of the outgoing mic stream was captured
    at ``first_adc_time``.
    """

    def __init__(self) -> None:
        self.first_adc_time: float | None = None
        self.first_monotonic: float | None = None
        self._lock = threading.Lock()

    def add(self, block: bytes, time_info=None) -> None:
        """Note the first block's times; runs on the audio thread."""
        with self._lock:
            if self.first_monotonic is None:
                self.first_monotonic = time.monotonic()
                self.first_adc_time = getattr(time_info, "inputBufferAdcTime", None)


def mic_sample_at(t: float, first_t: float, rate: int = MIC_RATE) -> int:
    """The index on the outgoing mic stream of an instant ``t`` on the mic's clock.

    ``first_t`` is when the stream's sample 0 was captured. An instant before it maps
    to 0, since a playback frame's mic_sample cannot be negative.
    """
    return max(0, round((t - first_t) * rate))


def playback_mic_sample(dac_time: float | None, monotonic: float | None, mic) -> int | None:
    """Where a playback edge falls on the outgoing mic stream, or None with no mic clock.

    PortAudio's output DAC time and input ADC time share one stream clock, so the edge
    is placed by ``dac_time - mic.first_adc_time``. A host API that reports no stream
    time (None or zero) falls back to the looser ``time.monotonic()`` pair.
    """
    if mic is None or mic.first_monotonic is None:
        return None
    if dac_time and mic.first_adc_time:
        return mic_sample_at(dac_time, mic.first_adc_time)
    if monotonic is not None:
        return mic_sample_at(monotonic, mic.first_monotonic)
    return None


@dataclass
class PlaybackEdge:
    """A reply's first sample (``start``) or the end of its last played one (``stop``).

    ``dac_time`` is the PortAudio stream time the edge reached the DAC, ``monotonic``
    the same instant on ``time.monotonic()``. ``reply_sample`` (stop only) counts the
    reply's samples that played, in the reply's own PCM as the server sent it.
    """

    event: str
    request_id: str
    dac_time: float | None
    monotonic: float | None
    reply_sample: int | None = None


class _Span:
    """The bytes of one reply (or of no reply, ``request_id`` None) in the play buffer."""

    def __init__(self, request_id: str | None) -> None:
        self.request_id = request_id
        self.queued = 0  # bytes still in the buffer
        self.played = 0  # bytes pulled for the device
        self.finished = False  # audio_end seen: no more bytes will come
        self.started = False
        self.end: tuple[float | None, float | None] = (None, None)


class TrackedPlayer(AudioPlayer):
    """AudioPlayer that reports when each reply's first and last sample reached the DAC.

    ``begin_reply`` tags the audio enqueued after it with a request id, and
    ``end_reply`` says that reply is whole. The output callback times each block with
    its ``outputBufferDacTime``, and a sample ``k`` samples into the block is at that
    time plus ``k / sample_rate``. A reply's first played sample gives a ``start``
    edge; the end of its last sample gives a ``stop`` edge once it ended and drained,
    or at ``clear`` (a barge or a cancel), counting only what was pulled for the
    device. Audio running dry mid-reply is no stop: the reply is still open.

    ``on_edge`` gets each ``PlaybackEdge``. It is called with the player's lock held,
    on the audio thread or the caller's, so it must only hand the edge on (for example
    with ``loop.call_soon_threadsafe``); holding the lock keeps a reply's start ahead of
    its stop.
    """

    def __init__(self, timeline: bool = False) -> None:
        super().__init__(timeline=timeline)
        self.on_edge: Callable[[PlaybackEdge], None] | None = None
        self._spans: list[_Span] = []
        self._current: _Span | None = None
        self._block_times: tuple[float | None, float | None] = (None, None)

    def _callback(self, outdata, frames: int, time_info, status) -> None:
        dac = getattr(time_info, "outputBufferDacTime", None)
        now = getattr(time_info, "currentTime", None)
        mono = time.monotonic()
        if dac and now and dac > now:
            mono += dac - now  # the output latency: the block plays this much later
        self._block_times = (dac, mono)
        super()._callback(outdata, frames, time_info, status)

    def begin_reply(self, request_id: str) -> None:
        """Tag the audio enqueued from now on with ``request_id``."""
        with self._lock:
            if self._current is not None:
                self._current.finished = True
            self._current = _Span(request_id)
            self._spans.append(self._current)

    def end_reply(self, request_id: str) -> None:
        """The reply is whole; its stop edge comes once its last byte is pulled."""
        with self._lock:
            for span in self._spans:
                if span.request_id == request_id:
                    span.finished = True
            if self._current is not None and self._current.request_id == request_id:
                self._current = None
            self._pop_drained()

    def enqueue(self, pcm: bytes) -> None:
        """Queue PCM after what is queued, as part of the current reply if any."""
        with self._lock:
            self._buf += pcm
            span = self._current
            if span is None:
                if not self._spans or self._spans[-1].request_id is not None:
                    self._spans.append(_Span(None))
                span = self._spans[-1]
            span.queued += len(pcm)

    def clear(self) -> None:
        """Drop all queued audio; a reply that was playing stops at what played."""
        with self._lock:
            self._buf.clear()
            for span in self._spans:
                if span.started:
                    self._stop(span)
            self._spans.clear()
            self._current = None

    def _edge_at(self, offset: int) -> tuple[float | None, float | None]:
        """The block's DAC and monotonic times moved ``offset`` bytes into the block."""
        dt = (offset // 2) / (self.sample_rate or MIC_RATE)
        dac, mono = self._block_times
        return (None if dac is None else dac + dt, None if mono is None else mono + dt)

    def _emit(self, edge: PlaybackEdge) -> None:
        if self.on_edge is not None:
            self.on_edge(edge)

    def _stop(self, span: _Span) -> None:
        self._emit(PlaybackEdge("stop", span.request_id, *span.end,
                                reply_sample=span.played // 2))

    def _pop_drained(self) -> None:
        """Drop the head spans that are done: empty and closed to more audio."""
        while self._spans:
            span = self._spans[0]
            if span.queued or not (span.finished or span is not self._current):
                return
            self._spans.pop(0)
            if span.started:
                self._stop(span)

    def _on_pulled(self, n: int) -> None:
        offset = 0
        while self._spans and offset < n:
            span = self._spans[0]
            take = min(span.queued, n - offset)
            if take == 0:
                if span is self._current and not span.finished:
                    return  # the open reply ran dry: silence follows, not a stop
                self._pop_drained()
                if self._spans and self._spans[0] is span:
                    return
                continue
            if not span.started and span.request_id is not None:
                span.started = True
                self._emit(PlaybackEdge("start", span.request_id, *self._edge_at(offset)))
            span.queued -= take
            span.played += take
            offset += take
            if span.request_id is not None:
                span.end = self._edge_at(offset)
            self._pop_drained()
        self._pop_drained()


class ChatStream:
    """One streaming chat completion from an OpenAI-compatible endpoint, over urllib.

    The HTTP read runs on a thread and hands each content delta to the event loop;
    ``deltas`` yields them as they arrive and ``text`` holds what has arrived so far.
    ``close`` ends the read early (a cancelled reply) by closing the response.
    """

    def __init__(self, base_url: str, model: str | None, messages: list[dict],
                 timeout: float = 120.0, think: bool = False):
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/v1/chat/completions"
        self.body: dict = {"messages": messages, "stream": True}
        if model:
            self.body["model"] = model
        if not think:
            # A thinking model reasons for seconds before its first spoken word and
            # those deltas carry no content (gemma-4 12B: 14 s against 0.2 s without).
            self.body["chat_template_kwargs"] = {"enable_thinking": False}
        self.timeout = timeout
        self.text = ""
        self._queue: asyncio.Queue | None = None
        self._response = None
        self._closed = False
        self._thread: threading.Thread | None = None

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Send the request and start reading its stream on a thread."""
        self._queue = asyncio.Queue()
        self._thread = threading.Thread(target=self._read, args=(loop,),
                                        name="chat-stream", daemon=True)
        self._thread.start()

    def _read(self, loop: asyncio.AbstractEventLoop) -> None:
        def put(item) -> None:
            loop.call_soon_threadsafe(self._queue.put_nowait, item)
        try:
            request = urllib.request.Request(
                self.url, data=json.dumps(self.body).encode(),
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                self._response = response
                for raw in response:
                    if self._closed:
                        break
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = payload.get("choices") or []
                    delta = (choices[0].get("delta") or {}).get("content") if choices else None
                    if delta:
                        put(delta)
            put(None)
        except BaseException as exc:  # the loop decides; a closed stream is not an error
            put(None if self._closed else exc)

    async def deltas(self):
        """Yield each content delta as it arrives; raise what the HTTP read raised."""
        while True:
            item = await self._queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            self.text += item
            yield item

    def close(self) -> None:
        """End the read; a reply the person interrupted needs no more tokens."""
        self._closed = True
        response = self._response
        if response is not None:
            try:
                response.close()
            except Exception:
                pass


def chat_error_message(base_url: str, timeout: float, exc: BaseException) -> str | None:
    """One line for a failed chat request, or None when ``exc`` is not an HTTP failure."""
    if isinstance(exc, urllib.error.HTTPError):
        return f"error: {base_url} answered HTTP {exc.code} {exc.reason}"
    if isinstance(exc, TimeoutError) or (
            isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError)):
        return f"error: {base_url} timed out after {timeout:g} s"
    if isinstance(exc, (urllib.error.URLError, ConnectionError, OSError)):
        reason = getattr(exc, "reason", exc)
        return f"error: cannot reach {base_url}: {reason}"
    return None


class Reply:
    """One reply in flight: its TTS request, its chat stream and its timing."""

    def __init__(self, request, chat, turn_id: int, text: str, dispatched_at: float):
        self.request = request
        self.chat = chat
        self.turn_id = turn_id
        self.text = text  # what the person said
        self.dispatched_at = dispatched_at
        self.first_audio_ms: float | None = None
        self.task: asyncio.Task | None = None
        self.ended = False  # audio_end seen, or cancelled
        self.cancelled = False
        self.reason = "complete"
        self.failed = False
        self.recorded = False  # its assistant message is in the history


class ClassicSession:
    """The classic turn pattern over a duplex session, with the LLM and the player injected.

    One listening turn at a time, opened with ``turn_start``; frames of another turn
    are ignored. The latest text per segment is kept, and the turn dispatches on the
    first endpointed frame (``endpoint`` or ``is_final`` true) with words. While the
    reply is in flight, a frame of the turn whose words differ from the dispatched
    ones ends the reply: an endpointed one dispatches the revision at once (the
    endpoint hold revised what was heard), another one is the person talking over the
    reply, and the turn dispatches again at its next endpoint. A reply that ends on
    its own opens the next turn; if its last audio is still playing when the next
    turn's first words arrive, the speaker is cleared.

    With ``turn_engine`` set (the connection opted in to the server's turn engine) the
    transcripts are only shown: the turn ends on the server's turn_end frame, which
    dispatches its text. A turn_end while a reply is in flight cancels it; a reply that
    had no words yet is folded into the new text. A barge frame ends the reply it
    names and clears the player; the server cancelled the request itself, so no
    tts_cancel goes back, and its audio_end (interrupted) leaves the turn open for the
    next turn_end. The player (a ``TrackedPlayer``) is told which reply each audio
    chunk belongs to, and ``on_playback_edge`` reports its edges on the mic stream of
    ``mic_clock``. A turn_engine_lost error frame turns all of this off again.
    """

    def __init__(self, duplex, chat_factory: Callable[[list[dict]], ChatStream], player,
                 log, clock: Callable[[], float] = time.monotonic, voice: str | None = None,
                 chunking: str = "clause", system_prompt: str = SYSTEM_PROMPT,
                 on_error: Callable[[str], None] = print,
                 say: Callable[[str], None] = lambda line: None,
                 turn_engine: str | None = None, mic_clock=None):
        self.duplex = duplex
        self.turn_engine = turn_engine
        self.mic_clock = mic_clock  # MicClock or MicCapture: the mic's first ADC time
        self.say = say  # one terminal line per utterance heard and per reply
        self.chat_factory = chat_factory
        self.player = player
        self.log = log
        self.clock = clock
        self.voice = voice
        self.chunking = chunking
        self.system_prompt = system_prompt
        self.on_error = on_error
        self.history: list[dict] = []
        self.turn_id: int | None = None
        self.segments: dict[int, str] = {}
        self.dispatched_words: list[str] | None = None
        self.reply: Reply | None = None
        self.replies: list[Reply] = []
        self._user_index: int | None = None  # history index of this turn's user message
        self._dispatched_keys: set[int] = set()  # the segments the dispatched text held
        self._answered_keys: set[int] = set()  # segments answered earlier in this turn
        self._t0 = clock()

    # --- the listening turn ---------------------------------------------------------

    def _reset_turn_state(self) -> None:
        self.segments = {}
        self.dispatched_words = None
        self._user_index = None
        self._dispatched_keys = set()
        self._answered_keys = set()

    async def open_turn(self) -> int:
        """Open the next listening turn; frames of earlier turns are ignored from now.

        The turn state is cleared before the server's ack is awaited, so a frame
        that lands in that window starts a fresh utterance instead of touching the
        answered one; a reply that such a frame started is left as it is.
        """
        self._reset_turn_state()
        turn_id = await self.duplex.turn_start()
        self.turn_id = turn_id
        if self.reply is None or self.reply.ended:
            self._reset_turn_state()
        return turn_id

    def turn_text(self) -> str:
        return " ".join(self.segments[k] for k in sorted(self.segments)).strip()

    async def on_transcript(self, frame: dict, now: float) -> None:
        """Feed one transcription frame: upsert its segment, dispatch or supersede."""
        if self.turn_id is not None and frame.get("turn_id", 0) != self.turn_id:
            return
        seg = frame.get("segment_index", 0)
        if self.turn_engine:
            self._show_transcript(frame)
            return
        if seg in self._answered_keys:
            return  # the endpoint hold confirming a segment answered before a barge-in
        heard_before = bool(_heard_words(self.turn_text()))
        self.segments[seg] = frame.get("text", "")
        text = self.turn_text()
        words = _heard_words(text)
        endpointed = bool(frame.get("endpoint") or frame.get("is_final"))
        if not words:
            return
        if not heard_before:
            self.say(f"... {text}")  # the first words of this turn: the microphone works
            if (self.reply is None or self.reply.ended) and self.player.is_playing():
                # The last reply's audio_end has passed but its tail is still playing:
                # the person talking over it stops the speaker, as a barge-in would.
                self.player.clear()
                self.say("  (playback stopped)")
        if self.reply is not None and not self.reply.ended:
            if words == self.dispatched_words:
                return  # a confirmation (or a re-casing) of what was dispatched
            if seg in self._dispatched_keys:
                # The dispatched segment itself changed: the endpoint hold revised it,
                # or the person went on with the same sentence. The revision replaces
                # the question.
                await self._cancel_reply(now, "superseded")
            else:
                # A new segment: the person's next utterance, said over the reply. The
                # answered segments leave the text; the history keeps the exchange.
                await self._cancel_reply(now, "barge-in")
                for key in self._dispatched_keys:
                    self.segments.pop(key, None)
                self._answered_keys |= self._dispatched_keys
                text = self.turn_text()
                words = _heard_words(text)
                if not words:
                    return
        if endpointed and words != self.dispatched_words:
            await self._dispatch(text, words, now)

    def _show_transcript(self, frame: dict) -> None:
        """Turn-engine mode: keep the turn's text for the terminal; turn_end ends turns."""
        heard_before = bool(_heard_words(self.turn_text()))
        self.segments[frame.get("segment_index", 0)] = frame.get("text", "")
        text = self.turn_text()
        if not heard_before and _heard_words(text):
            self.say(f"... {text}")

    async def on_turn_end(self, frame: dict, now: float) -> None:
        """Turn-engine mode: the server ended the person's turn; dispatch its text."""
        if not self.turn_engine:
            return
        text = (frame.get("text") or "").strip()
        self._write_event("turn_end", now, turn_id=frame.get("turn_id"), text=text,
                          mic_sample=frame.get("mic_sample"), by=frame.get("by"))
        self.segments = {}  # the next words of the person show again
        words = _heard_words(text)
        if not words:
            return
        reply = self.reply
        if reply is not None and not reply.ended:
            if reply.chat.text:
                # The reply had begun to answer: the history keeps that exchange.
                await self._cancel_reply(now, "barge-in")
            else:
                # Nothing was answered yet: the person went on, so one question.
                await self._cancel_reply(now, "superseded")
                text = f"{reply.text} {text}"
                words = _heard_words(text)
        await self._dispatch(text, words, now)

    async def on_barge(self, frame: dict, now: float) -> None:
        """The server heard the person cut in over a reply and cancelled it."""
        rid = frame.get("request_id")
        self._write_event("barge", now, request_id=rid,
                          onset_mic_sample=frame.get("onset_mic_sample"),
                          mic_sample=frame.get("mic_sample"))
        reply = self.reply
        if reply is None or rid != reply.request.request_id:
            return
        if reply.ended:
            # Its audio_end has passed but its tail is still playing.
            self.player.clear()
        else:
            await self._cancel_reply(now, "barge", send_cancel=False)
        self.say("  (barge-in: playback stopped)")

    async def on_playback_edge(self, edge: "PlaybackEdge") -> None:
        """Report one playback edge of the player to the server, on the mic stream."""
        if not self.turn_engine:
            return
        mic_sample = playback_mic_sample(edge.dac_time, edge.monotonic, self.mic_clock)
        if edge.event == "start":
            await self.duplex.playback_start(edge.request_id, mic_sample)
        else:
            await self.duplex.playback_stop(edge.request_id, mic_sample,
                                            reply_sample=edge.reply_sample)
        self._write_event("playback", self.clock(), request_id=edge.request_id,
                          event=edge.event, mic_sample=mic_sample,
                          reply_sample=edge.reply_sample)

    def on_error_frame(self, frame: dict) -> None:
        """A turn_engine_lost error: the session goes on as one that never opted in."""
        if frame.get("error_type") == "turn_engine_lost" and self.turn_engine:
            self.turn_engine = None
            if self.reply is None or self.reply.ended:
                self._reset_turn_state()
            self.say("turn engine lost: turns end on the server's endpoint frames again")

    async def _dispatch(self, text: str, words: list[str], now: float) -> None:
        self.dispatched_words = words
        self._dispatched_keys = set(self.segments)
        if self._user_index is None:
            self._user_index = len(self.history)
            self.history.append({"role": "user", "content": text})
        else:
            # The same turn again (a revision, or the person went on): one user message.
            del self.history[self._user_index:]
            self.history.append({"role": "user", "content": text})
        messages = [{"role": "system", "content": self.system_prompt}] + list(self.history)
        request = self.duplex.tts_request(voice=self.voice, chunking=self.chunking)
        chat = self.chat_factory(messages)
        reply = Reply(request, chat, self.turn_id or 0, text, now)
        self.reply = reply
        self.replies.append(reply)
        self.say(f"> {text}")
        reply.task = asyncio.create_task(self._run_reply(reply))

    async def _run_reply(self, reply: Reply) -> None:
        chat = reply.chat
        try:
            chat.start(asyncio.get_running_loop())
            async for delta in chat.deltas():
                await reply.request.send(delta)  # every token, as it arrives
            await reply.request.done()
            self._settle_history(reply)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reply.failed = True
            message = chat_error_message(getattr(chat, "base_url", "the chat endpoint"),
                                         getattr(chat, "timeout", 0.0), exc)
            self.on_error(message or f"error: chat request failed: {exc!r}")
            logger.debug("reply failed", exc_info=exc)
            try:
                await reply.request.done()  # the server answers with one audio_end
            except Exception:
                reply.ended = True
                self._settle_history(reply)
                self._write_utterance(reply)
                await self.open_turn()

    async def _cancel_reply(self, now: float, reason: str, send_cancel: bool = True) -> None:
        reply = self.reply
        reply.ended = reply.cancelled = True
        reply.reason = reason
        reply.chat.close()
        if reply.task is not None and not reply.task.done():
            reply.task.cancel()
            try:
                await reply.task
            except (asyncio.CancelledError, Exception):
                pass
        if send_cancel:
            await reply.request.cancel()
        self.player.clear()
        if reason == "superseded" and self._user_index is not None:
            del self.history[self._user_index:]  # the revision replaces the message
        else:
            self._settle_history(reply)
        self._user_index = None
        self._write_utterance(reply)

    def _settle_history(self, reply: Reply) -> None:
        """Leave the history alternating after ``reply``: its answer, or no question.

        The assistant message goes in once, with whatever text arrived. A reply that
        produced nothing (it failed, or was cut before its first token) takes its
        question out with it, since a user message without an answer would make every
        later request two user messages in a row.
        """
        if reply.recorded:
            return
        reply.recorded = True
        if reply.chat.text:
            self.history.append({"role": "assistant", "content": reply.chat.text})
        elif self._user_index is not None:
            del self.history[self._user_index:]

    # --- the audio events -----------------------------------------------------------

    async def on_audio_start(self, frame: dict, now: float) -> None:
        if self.reply is not None and frame.get("request_id") == self.reply.request.request_id:
            self.player.start(frame.get("sample_rate", 22050))
            if self.turn_engine:
                self.player.begin_reply(self.reply.request.request_id)

    async def on_pcm(self, pcm: bytes, request_id: str, now: float) -> None:
        reply = self.reply
        if reply is None or request_id != reply.request.request_id or reply.cancelled:
            return
        if reply.first_audio_ms is None:
            reply.first_audio_ms = (now - reply.dispatched_at) * 1000.0
        self.player.enqueue(pcm)

    async def on_audio_end(self, frame: dict, now: float) -> None:
        reply = self.reply
        if reply is None or frame.get("request_id") != reply.request.request_id:
            return
        if self.turn_engine:
            self.player.end_reply(reply.request.request_id)
        if reply.ended:
            return  # a cancelled reply's ack: the turn goes on listening
        reply.ended = True
        # The server's verdict on a reply it ended itself: complete, or error (a
        # generation failure, an unknown voice). A cancelled reply keeps its own reason.
        if not reply.cancelled:
            reply.reason = frame.get("reason", "complete")
        if reply.task is not None:
            try:
                await reply.task
            except (asyncio.CancelledError, Exception):
                pass
        self._settle_history(reply)
        self._write_utterance(reply)
        await self.open_turn()

    async def on_event(self, event: tuple, now: float) -> None:
        """Dispatch one DuplexSession.audio() event."""
        if event[0] == "pcm":
            await self.on_pcm(event[1], event[2], now)
        elif event[0] == "audio_start":
            await self.on_audio_start(event[1], now)
        elif event[0] == "audio_end":
            await self.on_audio_end(event[1], now)
        elif event[0] == "barge":
            await self.on_barge(event[1], now)

    # --- the record -----------------------------------------------------------------

    @property
    def t0(self) -> float:
        """The clock time the log's ``t`` counts from."""
        return self._t0

    def _write_event(self, kind: str, now: float, **fields) -> None:
        self.log.write(json.dumps({"type": kind, "t": round(now - self._t0, 3),
                                   **fields}) + "\n")
        self.log.flush()

    def _write_utterance(self, reply: Reply) -> None:
        delay = "-" if reply.first_audio_ms is None else f"{reply.first_audio_ms:.0f} ms"
        if reply.cancelled:
            state = f"cancelled, {reply.reason}"
        elif reply.reason != "complete":
            state = f"{reply.reason}, first audio {delay}"
        else:
            state = f"first audio {delay}"
        self.say(f"< {reply.chat.text or '(no reply)'}  [{state}]")
        self.log.write(json.dumps({
            "type": "utterance", "turn_id": reply.turn_id, "text": reply.text,
            "t": round(reply.dispatched_at - self._t0, 3),
            "first_audio_ms": (None if reply.first_audio_ms is None
                               else round(reply.first_audio_ms, 1)),
            "cancelled": reply.cancelled, "reason": reply.reason,
            "reply": reply.chat.text, "failed": reply.failed,
        }) + "\n")
        self.log.flush()

    async def shutdown(self) -> None:
        """Stop a reply still in flight at the end of the call."""
        if self.reply is not None and not self.reply.ended:
            await self._cancel_reply(self.clock(), "shutdown")

    def summary_line(self) -> str:
        delays = [r.first_audio_ms for r in self.replies if r.first_audio_ms is not None]
        first = f"{statistics.median(delays):.0f} ms" if delays else "n/a"
        cancelled = sum(r.cancelled for r in self.replies)
        return (f"{len(self.replies)} replies, median first audio {first} after the "
                f"endpoint, {cancelled} cancelled, tts chunking {self.chunking}")


async def warm_up_chat(base_url: str, model: str | None, say: Callable[[str], None] = print,
                       timeout: float = 300.0, think: bool = False) -> bool:
    """Send one short chat request so llama-swap loads the model before the first turn."""
    say(f"loading the chat model{' ' + model if model else ''} at {base_url} ...")
    chat = ChatStream(base_url, model, [{"role": "user", "content": "Say ready."}],
                      timeout=timeout, think=think)
    try:
        chat.start(asyncio.get_running_loop())
        async for _ in chat.deltas():
            pass
    except Exception as exc:
        say(chat_error_message(base_url, timeout, exc) or f"error: chat request failed: {exc!r}")
        return False
    say("chat model ready")
    return True


async def drain_errors(duplex, say: Callable[[str], None] = print,
                       on_frame: Callable[[dict], None] | None = None) -> None:
    """Print the server's error frames (an unknown voice, a bad request, a timeout warning).

    Without a reader the frames sit in the session's queue and a reply that the
    server refused ends with no audio and no explanation on the terminal. Each frame
    also goes to ``on_frame`` when it is given.
    """
    async for frame in duplex.errors():
        say(f"server error: {frame.get('error_type', 'error')}: {frame.get('message', '')}")
        if on_frame is not None:
            on_frame(frame)


def classic_output_paths(args: argparse.Namespace) -> tuple[str, str]:
    """The log and WAV paths of a classic run; defaults go in ``runs/classic/``."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if not (args.log and args.wav):
        CLASSIC_RUN_DIR.mkdir(parents=True, exist_ok=True)
    return (args.log or str(CLASSIC_RUN_DIR / f"classic-{stamp}.jsonl"),
            args.wav or str(CLASSIC_RUN_DIR / f"classic-{stamp}.wav"))


def classic_audio(args: argparse.Namespace):
    """The mic capture, the mic clock and the player a classic run needs.

    ``--capture`` keeps the mic as captured and the output timeline, as the micro-turn
    mode does. ``--turn-engine`` needs the mic's first ADC time (the capture's, else a
    ``MicClock`` that keeps no audio) and a ``TrackedPlayer`` for the playback edges.
    Without either flag it is the plain player and no mic clock.
    """
    capture = MicCapture() if args.capture else None
    mic_clock = capture if capture is not None else (MicClock() if args.turn_engine else None)
    player_type = TrackedPlayer if args.turn_engine else AudioPlayer
    return capture, mic_clock, player_type(timeline=capture is not None)


async def classic(args: argparse.Namespace) -> None:
    """Run a classic call over VoiceManager until Ctrl-C or ``--duration`` seconds."""
    token = os.environ.get("VOICEMANAGER_TOKEN")
    if not token:
        raise SystemExit("error: set VOICEMANAGER_TOKEN to a VoiceManager bearer token")
    from voicemanager.client import TurnEngineError, VoiceManagerClient, VoiceManagerError

    log_path, wav_path = classic_output_paths(args)
    print("warning: use headphones; there is no echo cancellation", flush=True)
    if not await warm_up_chat(args.url, args.model, think=args.think):
        raise SystemExit(2)
    capture, mic_clock, player = classic_audio(args)
    if capture is not None:
        print("capture: the mic and the output timeline are saved beside the WAV")
    client = VoiceManagerClient(host=args.vm_host, port=args.vm_port, token=token)
    try:
        await client.__aenter__()
    except (VoiceManagerError, OSError) as exc:
        raise SystemExit(f"error: cannot connect to VoiceManager: {exc}") from exc
    session = None

    def say(line: str) -> None:
        print(line, flush=True)

    try:
        async with client.duplex() as duplex:
            if args.turn_engine:
                # The opt-in must come before the first audio frame.
                try:
                    await duplex.enable_turn_engine(args.turn_engine)
                except TurnEngineError as exc:
                    raise SystemExit(f"error: the server refused the turn engine "
                                     f"{args.turn_engine}: {exc.error_type or 'no answer'}: "
                                     f"{exc}") from exc
            with open(log_path, "w") as log:
                session = ClassicSession(
                    duplex, lambda messages: ChatStream(args.url, args.model, messages,
                                                        think=args.think),
                    player, log, voice=args.voice, chunking=args.tts_mode, say=say,
                    turn_engine=args.turn_engine, mic_clock=mic_clock)
                await session.open_turn()
                ends = ("the server's turn engine ends each turn" if args.turn_engine
                        else "speak, pause, and the reply follows")
                say(f"listening (stt {client.stt_engine}, tts {client.tts_engine}); "
                    f"{ends}; Ctrl-C to stop")

                async def transcripts() -> None:
                    async for frame in duplex.transcripts():
                        await session.on_transcript(frame, session.clock())
                    logger.warning("VoiceManager closed the connection")

                async def audio() -> None:
                    async for event in duplex.audio():
                        await session.on_event(event, session.clock())

                coros = [microturn_client._mic_task(duplex, mic_clock), transcripts(),
                         audio(), drain_errors(duplex, say, on_frame=session.on_error_frame)]
                if args.turn_engine:
                    coros += [_turn_end_task(duplex, session),
                              _playback_task(session, player, asyncio.get_running_loop())]
                tasks = [asyncio.create_task(c) for c in coros]
                try:
                    await asyncio.wait(tasks, timeout=args.duration,
                                       return_when=asyncio.FIRST_COMPLETED)
                except asyncio.CancelledError:
                    pass  # Ctrl-C: stop cleanly below
                for task in tasks:
                    task.cancel()
                for task, outcome in zip(tasks, await asyncio.gather(
                        *tasks, return_exceptions=True)):
                    if isinstance(outcome, Exception):
                        logger.error("%s stopped: %r", task.get_coro().__name__, outcome)
                try:
                    await session.shutdown()
                except Exception:
                    pass
    finally:
        player.close()
        await client.__aexit__(None, None, None)
    out_rate = player.sample_rate or client.last_tts_sample_rate
    write_wav(wav_path, bytes(player.played), out_rate)
    print(f"log: {log_path}  wav: {wav_path}")
    if capture is not None:
        mic_path, out_path, json_path = write_capture(
            wav_path, capture, player, out_rate,
            log_t0=session.t0 if session is not None else None)
        print(f"capture: {mic_path}  {out_path}  {json_path}")
    if session is not None:
        print(session.summary_line())


async def _turn_end_task(duplex, session: ClassicSession) -> None:
    """Feed the server's turn_end frames to the session until the connection closes."""
    async for frame in duplex.turn_ends():
        await session.on_turn_end(frame, session.clock())


async def _playback_task(session: ClassicSession, player: "TrackedPlayer",
                         loop: asyncio.AbstractEventLoop) -> None:
    """Send the player's playback edges as playback frames, in the order they happened.

    The player emits them on the audio thread under its lock, so they only cross to
    the event loop here.
    """
    edges: asyncio.Queue = asyncio.Queue()
    player.on_edge = lambda edge: loop.call_soon_threadsafe(edges.put_nowait, edge)
    try:
        while True:
            edge = await edges.get()
            try:
                await session.on_playback_edge(edge)
            except Exception as exc:  # a closed connection; the other tasks notice it
                logger.debug("playback frame not sent: %r", exc)
    finally:
        player.on_edge = None


def build_parser() -> argparse.ArgumentParser:
    """The command line: one subcommand per mode."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_subparsers(dest="mode", required=True)
    classic = modes.add_parser(
        "classic", help="turn by turn with any OpenAI-compatible chat model")
    add_classic_arguments(classic)
    microturn = modes.add_parser(
        "microturn", help="the micro-turn full-duplex loop (needs the LoRA model)")
    microturn_client.add_live_arguments(microturn)
    replay = modes.add_parser("replay", help="the micro-turn loop on a scripted call, no audio")
    microturn_client.add_replay_arguments(replay)
    return parser


def add_classic_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of the classic mode."""
    microturn_client.add_server_options(parser)
    parser.add_argument("--vm-host", default="localhost", help="VoiceManager host")
    parser.add_argument("--vm-port", type=int, default=8255, help="VoiceManager port")
    parser.add_argument("--voice", default=None, help="TTS voice (default: the token's voice)")
    parser.add_argument("--duration", type=float, default=None,
                        help="stop after this many seconds (default: at Ctrl-C)")
    parser.add_argument("--log", default=None,
                        help="JSON lines log (default: runs/classic/classic-<stamp>.jsonl)")
    parser.add_argument("--wav", default=None,
                        help="WAV of the audio played (default: runs/classic/classic-<stamp>.wav)")
    parser.add_argument("--tts-mode", choices=("clause", "sentence"), default="clause",
                        help="chunking of the reply's tts_stream request (default: clause)")
    parser.add_argument("--think", action="store_true",
                        help="let a thinking model reason before it answers (default: off, "
                             "since the reasoning delays the first spoken word by seconds)")
    parser.add_argument("--turn-engine", choices=TURN_ENGINES, default=None,
                        help="opt in to the server's turn engine: the turn ends on its "
                             "turn_end frame, a barge frame stops the reply, and the client "
                             "reports each reply's playback edges (default: off)")
    parser.add_argument("--capture", action="store_true",
                        help="also save the mic as captured (<stem>-mic.wav), the output with "
                             "its silence (<stem>-out.wav) and both start times "
                             "(<stem>-capture.json), as the micro-turn mode does")


def main(argv: list[str] | None = None) -> None:
    """Run the selected mode."""
    args = build_parser().parse_args(argv)
    if args.mode == "microturn":
        asyncio.run(microturn_client.live(args))
    elif args.mode == "replay":
        microturn_client.run_replay(args)
    else:
        asyncio.run(classic(args))


if __name__ == "__main__":
    main()
