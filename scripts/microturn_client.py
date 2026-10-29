"""Reference client for micro-turn full-duplex voice.

Every 0.6 s the client sends the language model the words heard in that window, or
<no voice>, and the model answers with up to three words to speak next, or <listen>.

The model is Qwen3.6-35B-A3B with the micro-turn duplex LoRA, served by llama.cpp's
llama-server started with the base GGUF and ``--lora microturn-duplex-lora.gguf`` (or a
llama-swap entry that serves it; pass ``--model`` so llama-swap can route the request).

The system prompt below is the exact text the adapter was trained on. Keep it byte for
byte, and keep the history append-only so llama-server's prompt cache resumes each tick.

Replay mode runs a scripted call through the model as text, with no audio and no
VoiceManager connection, and prints one line per tick. Without ``--script`` it plays the
adapter card's demo call; a script file has one heard string per line, empty for silence.

Live mode is a real call over VoiceManager: the microphone streams to its STT, each tick
sends the new words to the model, and the model's words go to its streaming TTS and play
on the speaker. ``--tts-mode clause`` (the default) opens each utterance with chunking
"clause", so the server cuts the held words into small pieces itself; ``--tts-mode flush``
flushes each speaking tick's words instead. It writes one JSON line per tick and per
utterance (turn time, delay from the last heard word to the first audio, and the TTS mode
on each tick line; an utterance that a cut-in stops gets its line again with cancelled
true) and a WAV of what played. Use headphones: there is no echo cancellation, so the
microphone would hear the assistant. The bearer token comes from the
``VOICEMANAGER_TOKEN`` environment variable.

The command line is scripts/voice_agent.py, which runs this module's two modes as
its ``replay`` and ``microturn`` subcommands beside the classic turn pattern:

    python scripts/voice_agent.py replay [--script FILE] [--url URL] [--model NAME]
    python scripts/voice_agent.py microturn [--vm-host HOST] [--vm-port PORT] [--voice NAME]
        [--duration SECONDS] [--log FILE] [--wav FILE] [--tts-mode {clause,flush}]
        [--url URL] [--model NAME]
"""
import argparse
import asyncio
import json
import logging
import math
import os
import pathlib
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

logger = logging.getLogger(__name__)

TICK = 0.6
LISTEN = "<listen>"
NOVOICE = "<no voice>"
# Listen only if "<" holds at least this much of the first token's probability.
LISTEN_P = 0.3
SYSTEM = (
    "You are a voice assistant in a live phone call; both sides can talk at the same time. "
    f"Every {TICK:g} seconds you get what the microphone heard in that moment, as a streaming "
    f"transcript fragment, or {NOVOICE}. You reply with up to three new words to say next, "
    f"or {LISTEN} to add nothing."
)
# The adapter card's scripted call.
DEMO = [
    "Hey I'm cleaning", "the bathroom", "right now and", "I got some", "bleach here", NOVOICE,
    "and I'm going", "to mix it", "with the ammonia", "cleaner so it", "gets extra strong",
    *[NOVOICE] * 7,
]


class TickSlicer:
    """Turns VoiceManager's streaming transcription frames into one string per tick.

    The server sends cumulative frames: each interim carries the whole transcript so far
    of the current utterance (its ``segment_index``), the final carries the committed
    text of that segment, and on silence nothing arrives at all. The model expects only
    the words new in each 0.6 s window, or ``NOVOICE``. So the client keeps the latest
    frame per segment, and each tick sends the words beyond those already sent.

    Sent words are never taken back, because the model's history only grows. When the
    STT revises words already sent (or shrinks the text), the tick counts one revision
    for that segment and sends only the words past the already-sent count.
    """

    def __init__(self) -> None:
        self.revisions = 0
        self.last_revised = False
        self._latest: dict[int, str] = {}
        # Per segment: how many words were sent, and the latest version of those words.
        # After a revision or a shrink the version follows the new text, so the same
        # change is not counted again on later frames that repeat or extend it.
        self._count: dict[int, int] = {}
        self._sent: dict[int, list[str]] = {}

    def feed(self, frame: dict) -> None:
        """Store one transcription frame; only the latest per segment counts per tick."""
        self._latest[frame.get("segment_index", 0)] = frame.get("text", "")

    def tick(self) -> str:
        """Return the new words heard since the last tick, or ``NOVOICE``, and reset."""
        emitted: list[str] = []
        self.last_revised = False
        for seg in sorted(self._latest):
            words = self._latest[seg].split()
            sent = self._sent.get(seg, [])
            n = self._count.get(seg, 0)
            if len(words) < len(sent) or words[:len(sent)] != sent:
                self.revisions += 1
                self.last_revised = True
                logger.debug("segment %d revised: sent=%r now=%r", seg, sent, words)
            if len(words) > n:
                emitted.extend(words[n:])
                self._count[seg] = len(words)
            self._sent[seg] = words[:self._count.get(seg, 0)]
        self._latest.clear()
        return " ".join(emitted) if emitted else NOVOICE


def build_prompt(history: list[tuple[str, str]], heard: str) -> str:
    """Raw ChatML for one tick: system, every past (role, content), the new user turn.

    Ends with an open assistant turn and no thinking block, as the adapter was trained.
    """
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
    for role, content in [*history, ("user", heard)]:
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return prompt + "<|im_start|>assistant\n"


def completion_body(prompt: str, model: str | None = None) -> dict:
    """The llama-server ``/completion`` request body for one tick.

    ``model`` is added only when given, since llama-swap routes ``/completion`` by it.
    """
    body = {
        "prompt": prompt,
        "n_predict": 64,
        "temperature": 0,
        "stop": ["<|im_end|>"],
        "cache_prompt": True,
        "n_probs": 8,
    }
    if model:
        body["model"] = model
    return body


class CompletionEngine(Protocol):
    """A text completion backend that answers in llama-server's ``/completion`` shape.

    The reply carries ``content`` and, when ``n_probs`` > 0,
    ``completion_probabilities[0]["top_logprobs"]``: a list of
    ``{"id", "token", "logprob"}`` for the first generated token.
    """

    def complete(self, body: dict) -> dict:
        """Send one completion request and return the parsed JSON reply."""
        ...


class LlamaServerEngine:
    """CompletionEngine over HTTP to llama-server (or llama-swap) with the stdlib only."""

    # llama-swap may take more than 30 s to cold-load the model on the first tick.
    def __init__(self, base_url: str, model: str | None = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, body: dict) -> dict:
        """POST the body to ``{base_url}/completion`` and return the JSON reply."""
        if self.model:
            body = {**body, "model": self.model}
        req = urllib.request.Request(
            f"{self.base_url}/completion",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.load(resp)


def http_error_message(engine: LlamaServerEngine, exc: BaseException) -> str | None:
    """One line for an HTTP failure of a tick, or None when ``exc`` is not one.

    An ``HTTPError`` names its status, a timeout (also one wrapped in a ``URLError``)
    the timeout, and any other ``URLError`` is an unreachable server.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return f"error: {engine.base_url} answered HTTP {exc.code} {exc.reason}"
    if isinstance(exc, TimeoutError) or (
            isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, TimeoutError)):
        return f"error: {engine.base_url} timed out after {engine.timeout:g} s"
    if isinstance(exc, urllib.error.URLError):
        return f"error: cannot reach {engine.base_url}: {exc.reason}"
    return None


@dataclass
class TickResult:
    """The outcome of one micro-turn decision."""

    said: str
    p_listen: float
    turn_ms: float
    requests: int


def _first_token_candidates(reply: dict) -> list[dict]:
    """The first token's top candidates, or an empty list when the reply has none."""
    probs = reply.get("completion_probabilities") or []
    if not probs:
        return []
    return probs[0].get("top_logprobs") or []


def decide(engine: CompletionEngine, history: list[tuple[str, str]], heard: str,
           model: str | None = None) -> TickResult:
    """Run one micro-turn with the speak-or-listen rule and return what to say.

    Synchronous on purpose: an async caller runs it with ``asyncio.to_thread``.
    """
    start = time.perf_counter()
    body = completion_body(build_prompt(history, heard), model=model)
    reply = engine.complete(body)
    requests = 1
    out = reply.get("content", "").strip()
    top = _first_token_candidates(reply)
    listen = [c for c in top if c["token"].startswith("<")]
    listen_ids = [c["id"] for c in listen]
    p_listen = sum(math.exp(c["logprob"]) for c in listen)
    # Greedy decoding picks "<" even at e.g. 27% when the rest of the mass is spread over
    # many possible next words, and the model then stops mid-sentence. So below the
    # threshold every "<" token is banned and the model has to speak.
    if out.startswith("<") and p_listen < LISTEN_P and listen_ids:
        retry = {**body, "logit_bias": [[i, False] for i in listen_ids], "n_probs": 0}
        out = engine.complete(retry).get("content", "").strip()
        requests = 2
    said = LISTEN if out.startswith("<") else out
    turn_ms = (time.perf_counter() - start) * 1000.0
    logger.debug("heard=%r said=%r p_listen=%.2f requests=%d %.0f ms",
                 heard, said, p_listen, requests, turn_ms)
    return TickResult(said=said, p_listen=p_listen, turn_ms=turn_ms, requests=requests)


class TTSSink(Protocol):
    """Where the orchestrator sends speech: VoiceManager's streaming TTS requests.

    ``send_text`` appends to the request (``tts_stream``), ``flush`` synthesizes what is
    buffered and keeps it open, ``done`` ends it and ``cancel`` stops it (``tts_cancel``).
    """

    async def send_text(self, request_id: str, text: str, chunking: str | None = None) -> None:
        """Append text to the request, opening it on first use.

        ``chunking`` is passed only on the first call of a request, and only in clause
        mode; the server fixes it when the request starts.
        """
        ...

    async def flush(self, request_id: str) -> None:
        """Synthesize the buffered text now and keep the request open."""
        ...

    async def done(self, request_id: str) -> None:
        """End the request; the server sends ``audio_end`` after the last audio."""
        ...

    async def cancel(self, request_id: str) -> None:
        """Stop the request and drop any audio not yet sent."""
        ...


class Player(Protocol):
    """The local audio output that plays the TTS audio."""

    def is_playing(self) -> bool:
        """True while audio is playing or queued."""
        ...

    def clear(self) -> None:
        """Stop playback and drop all queued audio."""
        ...


TTS_MODES = ("clause", "flush")


class TTSOrchestrator:
    """Turns each tick's decision into VoiceManager streaming TTS calls.

    One request per assistant utterance. In ``flush`` mode each speaking tick's words
    are flushed, so their audio starts within a tick. In ``clause`` mode the first text
    of a request carries chunking "clause" and no flush is sent: the server cuts the
    held words into small pieces itself. In both modes the first LISTEN after speech
    ends the request with ``done``. The model is never told what audio is playing, so the client
    stops playback itself when the user talks over it: user words, a LISTEN and audio
    still playing cancel the request and clear the player.
    """

    def __init__(self, sink: TTSSink, player: Player,
                 new_id: Callable[[], str] = lambda: str(uuid.uuid4()),
                 mode: str = "flush"):
        if mode not in TTS_MODES:
            raise ValueError(f"unknown TTS mode {mode!r}; expected one of {TTS_MODES}")
        self.mode = mode
        self.sink = sink
        self.player = player
        self.new_id = new_id
        self.request_id: str | None = None  # the current request, or the last one
        self.open = False  # request_id still takes text (no done or cancel sent)
        self.live = False  # request_id may still have audio to cancel
        self.cutins = 0

    async def on_tick(self, heard: str, said: str) -> None:
        """Send the TTS calls for one tick: what was heard and what the model said."""
        if said != LISTEN:
            if self.open:
                await self.sink.send_text(self.request_id, f" {said}")
            else:
                self.request_id, self.open, self.live = self.new_id(), True, True
                if self.mode == "clause":
                    await self.sink.send_text(self.request_id, said, chunking="clause")
                else:
                    await self.sink.send_text(self.request_id, said)
            if self.mode == "flush":
                await self.sink.flush(self.request_id)
            return
        user_spoke = bool(heard) and heard != NOVOICE
        if user_spoke and self.player.is_playing():
            # After done the server may still be sending audio for the request, so it
            # is cancelled too, not only an open one.
            if self.live:
                await self.sink.cancel(self.request_id)
            self.open = self.live = False
            self.player.clear()
            self.cutins += 1
            logger.debug("cut-in on %s: heard=%r", self.request_id, heard)
        elif self.open:
            await self.sink.done(self.request_id)
            self.open = False


def format_tick(heard: str, result: TickResult) -> str:
    """One aligned output line for a tick: heard, said, P(listen) and the LLM turn time."""
    return (f"  heard: {heard:<22} said: {result.said:<28} "
            f"P(listen) {result.p_listen:.2f} {result.turn_ms:>4.0f} ms")


def replay(engine: CompletionEngine, script: Iterable[str],
           out: Callable[[str], None] = print,
           results: list[TickResult] | None = None) -> list[tuple[str, str]]:
    """Run each heard string through the model as one tick and return the history.

    Ticks run back to back: the turn time is reported, not paced. Each tick's
    ``TickResult`` is appended to ``results`` when it is given.
    """
    history: list[tuple[str, str]] = []
    for heard in script:
        result = decide(engine, history, heard)
        history += [("user", heard), ("assistant", result.said)]
        out(format_tick(heard, result))
        if results is not None:
            results.append(result)
    return history


def read_script(path: str | pathlib.Path) -> list[str]:
    """Read a replay script: one heard string per line, an empty line means ``NOVOICE``."""
    lines = pathlib.Path(path).read_text().splitlines()
    return [line.strip() or NOVOICE for line in lines]


def summary(results: list[TickResult]) -> str:
    """One line for the end of a run: tick count, median turn time, ticks that spoke."""
    median = statistics.median(r.turn_ms for r in results) if results else 0.0
    spoke = sum(r.said != LISTEN for r in results)
    return f"{len(results)} ticks, median turn {median:.0f} ms, spoke on {spoke}"


class SessionTTSSink:
    """TTSSink over a ``DuplexSession`` of the VoiceManager client.

    One ``TTSRequest`` per request id, made on its first text with the voice and the
    chunking the orchestrator asks for; flush, done and cancel go to that request.
    ``is_cancelled`` tells the live session which request's late audio to ignore.
    """

    def __init__(self, session, voice: str | None = None):
        self.session = session
        self.voice = voice
        self._requests: dict = {}
        self._cancelled: set[str] = set()

    def _request(self, request_id: str, chunking: str | None = None):
        if request_id not in self._requests:
            self._requests[request_id] = self.session.tts_request(
                voice=self.voice, chunking=chunking, request_id=request_id)
        return self._requests[request_id]

    async def send_text(self, request_id: str, text: str, chunking: str | None = None) -> None:
        """Append text to the request; its first frame carries the voice and chunking."""
        await self._request(request_id, chunking).send(text)

    async def flush(self, request_id: str) -> None:
        """Synthesize the buffered text now and keep the request open."""
        await self._request(request_id).flush()

    async def done(self, request_id: str) -> None:
        """End the request."""
        await self._request(request_id).done()

    async def cancel(self, request_id: str) -> None:
        """Stop the request; the session drops its audio still in flight."""
        self._cancelled.add(request_id)
        await self._request(request_id).cancel()

    def is_cancelled(self, request_id: str) -> bool:
        return request_id in self._cancelled


class AudioPlayer:
    """Player that plays mono int16 PCM on the default output device.

    The TTS task feeds a byte buffer with ``enqueue``; the output stream's callback, on
    the audio thread, takes from it with ``pull``, which pads with silence when the
    buffer runs dry. Every real byte pulled is kept in ``played`` for the WAV.
    """

    def __init__(self) -> None:
        self.played = bytearray()
        self.sample_rate: int | None = None
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._stream = None

    def start(self, sample_rate: int) -> None:
        """Open the output stream at the first reply's rate; later calls do nothing."""
        if self._stream is not None:
            return
        import sounddevice as sd

        self.sample_rate = sample_rate
        self._stream = sd.RawOutputStream(samplerate=sample_rate, channels=1, dtype="int16",
                                          callback=self._callback)
        self._stream.start()

    def _callback(self, outdata, frames: int, time_info, status) -> None:
        if status:
            logger.debug("output status: %s", status)
        outdata[:] = self.pull(frames * 2)

    def enqueue(self, pcm: bytes) -> None:
        """Queue PCM to play after what is already queued."""
        with self._lock:
            self._buf += pcm

    def pull(self, n: int) -> bytes:
        """Take n bytes to play, padded with zeros (silence) when fewer are queued."""
        with self._lock:
            chunk = bytes(self._buf[:n])
            del self._buf[:n]
            self.played += chunk
        return chunk + b"\x00" * (n - len(chunk))

    def is_playing(self) -> bool:
        """True while queued audio remains."""
        with self._lock:
            return bool(self._buf)

    def clear(self) -> None:
        """Drop all queued audio."""
        with self._lock:
            self._buf.clear()

    def close(self) -> None:
        """Stop and close the output stream, if open."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def write_wav(path: str | pathlib.Path, pcm: bytes, sample_rate: int) -> None:
    """Write mono int16 PCM to a WAV file."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def _heard_words(text: str) -> list[str]:
    """The words of a transcript with case and punctuation removed."""
    return [w for w in (re.sub(r"\W+", "", word).lower() for word in text.split()) if w]


class LiveSession:
    """The state of one live call: history, tick log and first-audio timing.

    The parts are injected, so the loop runs the same over fakes as over llama-server
    and VoiceManager. Times are clock seconds; the log's ``t`` counts from the start.

    An utterance line is written at a request's first audio. When a cut-in cancels that
    request later, the line is written again with ``cancelled`` true and ``t`` the time
    of the cut-in, so the last utterance line per request id is its final state.
    """

    def __init__(self, engine: CompletionEngine, slicer: TickSlicer,
                 orchestrator: TTSOrchestrator, log,
                 clock: Callable[[], float] = time.monotonic,
                 cancelled: Callable[[str], bool] = lambda request_id: False):
        self.engine = engine
        self.slicer = slicer
        self.orchestrator = orchestrator
        self.log = log
        self.clock = clock
        self.cancelled = cancelled
        self.history: list[tuple[str, str]] = []
        self.results: list[TickResult] = []
        self.first_audio_ms: list[float] = []
        self.last_word_t: float | None = None
        self._t0 = clock()
        self._text: dict[int, str] = {}
        # Per request: the time of the last word heard before the tick that opened it.
        self._opened_at: dict[str, float | None] = {}
        self._logged: set[str] = set()
        # Per logged request: its first_audio_ms, written again when a cut-in cancels it.
        self._first_audio: dict[str, float | None] = {}
        self._audio_id: str | None = None  # the request the next binary frames belong to

    def _write(self, record: dict) -> None:
        self.log.write(json.dumps(record) + "\n")
        self.log.flush()

    def on_transcription(self, frame: dict, now: float) -> None:
        """Feed a transcription frame; a segment that gains words marks the last word's time.

        A final that only re-cases or re-punctuates the words already heard is not a
        new word, so it leaves the time alone and first_audio_ms is not understated.
        """
        self.slicer.feed(frame)
        seg, text = frame.get("segment_index", 0), frame.get("text", "")
        if len(_heard_words(text)) > len(_heard_words(self._text.get(seg, ""))):
            self.last_word_t = now
        self._text[seg] = text

    async def run_tick(self, now: float) -> TickResult:
        """Run one tick: decide, drive the TTS and write the tick line."""
        # Taken before the decision: words heard while the model runs did not cause it.
        word_t = self.last_word_t
        heard = self.slicer.tick()
        result = await asyncio.to_thread(decide, self.engine, self.history, heard)
        self.history += [("user", heard), ("assistant", result.said)]
        self.results.append(result)
        before, cutins = self.orchestrator.request_id, self.orchestrator.cutins
        await self.orchestrator.on_tick(heard, result.said)
        if self.orchestrator.request_id != before:
            self._opened_at[self.orchestrator.request_id] = word_t
        # The orchestrator counts the cut-in itself, so this does not depend on when the
        # client's cancel mark for the request is set or dropped.
        if self.orchestrator.cutins > cutins and before in self._logged:
            self._write_utterance(before, now, self._first_audio[before], cancelled=True)
        self._write({
            "type": "tick", "t": round(now - self._t0, 3), "heard": heard,
            "said": result.said, "p_listen": round(result.p_listen, 4),
            "turn_ms": round(result.turn_ms, 1), "requests": result.requests,
            "revised": self.slicer.last_revised, "revisions": self.slicer.revisions,
            "tts_mode": self.orchestrator.mode,
        })
        return result

    def on_audio_start(self, request_id: str, sample_rate: int, now: float) -> None:
        """Note the request the following binary frames belong to."""
        self._audio_id = request_id
        logger.debug("audio_start %s at %d Hz", request_id, sample_rate)

    def on_audio_end(self, request_id: str) -> None:
        """Forget the request whose audio ended."""
        if self._audio_id == request_id:
            self._audio_id = None

    def on_audio(self, pcm: bytes, now: float) -> bool:
        """Log the first audio of a request; False when the audio must not be played."""
        rid = self._audio_id
        if rid is None or self.cancelled(rid):
            return False
        if rid not in self._logged:
            self._logged.add(rid)
            word_t = self._opened_at.get(rid, self.last_word_t)
            delay = None if word_t is None else (now - word_t) * 1000.0
            if delay is not None:
                self.first_audio_ms.append(delay)
            self._first_audio[rid] = delay
            self._write_utterance(rid, now, delay, cancelled=False)
        return True

    def _write_utterance(self, rid: str, now: float, delay: float | None,
                         cancelled: bool) -> None:
        self._write({
            "type": "utterance", "request_id": rid, "t": round(now - self._t0, 3),
            "first_audio_ms": None if delay is None else round(delay, 1),
            "cancelled": cancelled,
        })

    def summary_line(self) -> str:
        """One line for the end of a call: turn time, first audio, budget, cut-ins, mode."""
        turns = [r.turn_ms for r in self.results]
        turn = f"{statistics.median(turns):.0f} ms" if turns else "n/a"
        first = (f"{statistics.median(self.first_audio_ms):.0f} ms"
                 if self.first_audio_ms else "n/a")
        over = sum(t > TICK * 1000 for t in turns)
        return (f"{len(turns)} ticks, median turn {turn}, median first audio {first}, "
                f"{over} over budget, {self.slicer.revisions} revisions, "
                f"{self.orchestrator.cutins} cut-ins, tts mode {self.orchestrator.mode}")


async def _mic_task(duplex) -> None:
    """Send the microphone to VoiceManager as 16 kHz mono int16 blocks of 100 ms."""
    import sounddevice as sd

    loop = asyncio.get_running_loop()
    blocks: asyncio.Queue = asyncio.Queue()

    def callback(indata, frames: int, time_info, status) -> None:
        # Runs on the audio thread; hand the block to the event loop.
        loop.call_soon_threadsafe(blocks.put_nowait, bytes(indata))

    with sd.RawInputStream(samplerate=16000, channels=1, dtype="int16", blocksize=1600,
                           callback=callback):
        while True:
            await duplex.send_audio(await blocks.get())


async def _stt_task(duplex, session: LiveSession) -> None:
    """Feed transcription frames to the session until the connection closes."""
    async for frame in duplex.transcripts():
        session.on_transcription(frame, session.clock())
    logger.warning("VoiceManager closed the connection")


async def _tts_task(duplex, session: LiveSession, player: AudioPlayer) -> None:
    """Play TTS audio and time its first bytes until the connection closes."""
    async for event in duplex.audio():
        if event[0] == "pcm":
            if session.on_audio(event[1], session.clock()):
                player.enqueue(event[1])
        elif event[0] == "audio_start":
            frame = event[1]
            rate = frame.get("sample_rate", 22050)
            session.on_audio_start(frame.get("request_id"), rate, session.clock())
            player.start(rate)
        elif event[0] == "audio_end":
            session.on_audio_end(event[1].get("request_id"))


async def _tick_task(session: LiveSession) -> None:
    """Run a tick every TICK seconds on a monotonic schedule."""
    next_at = session.clock() + TICK
    while True:
        await asyncio.sleep(max(0.0, next_at - session.clock()))
        await session.run_tick(session.clock())
        # After an overrun the next tick starts at once and the schedule restarts from
        # there: no tick is skipped and no burst of ticks follows to catch up.
        next_at = max(next_at + TICK, session.clock())


def check_stt_engine(name: str) -> None:
    """Exit unless the server's STT engine streams interim transcripts the slicer can use.

    Whisper sends only finals and numbers each utterance's segments from 0 again, so the
    slicer would take later utterances for revisions of earlier ones and drop their words.
    """
    if name == "whisper":
        raise SystemExit("error: the server's STT engine is whisper, which sends no interim "
                         "transcripts; run VoiceManager with a streaming engine "
                         "(VM_STT_ENGINE=audiocpp, qwen or nemotron)")


RUN_DIR = pathlib.Path("runs") / "microturn"


def live_output_paths(args: argparse.Namespace) -> tuple[str, str]:
    """The log and WAV paths for a live run; defaults go in ``runs/microturn/``.

    The directory is under the current directory, is git-ignored and is created when
    missing. ``--log`` and ``--wav`` override the defaults.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if not (args.log and args.wav):
        RUN_DIR.mkdir(parents=True, exist_ok=True)
    return (args.log or str(RUN_DIR / f"microturn-{stamp}.jsonl"),
            args.wav or str(RUN_DIR / f"microturn-{stamp}.wav"))


async def live(args: argparse.Namespace) -> None:
    """Run a live call over VoiceManager until Ctrl-C or ``--duration`` seconds."""
    token = os.environ.get("VOICEMANAGER_TOKEN")
    if not token:
        raise SystemExit("error: set VOICEMANAGER_TOKEN to a VoiceManager bearer token")
    from voicemanager.client import VoiceManagerClient, VoiceManagerError
    from websockets.exceptions import ConnectionClosed

    log_path, wav_path = live_output_paths(args)
    print("warning: use headphones; there is no echo cancellation")
    client = VoiceManagerClient(host=args.vm_host, port=args.vm_port, token=token)
    try:
        await client.__aenter__()
    except (VoiceManagerError, OSError) as exc:
        raise SystemExit(f"error: cannot connect to VoiceManager: {exc}") from exc
    player = AudioPlayer()
    engine = LlamaServerEngine(args.url, model=args.model)
    session = None
    try:
        check_stt_engine(client.stt_engine)
        async with client.duplex() as duplex:
            sink = SessionTTSSink(duplex, voice=args.voice)
            orchestrator = TTSOrchestrator(sink, player, mode=args.tts_mode)
            with open(log_path, "w") as log:
                session = LiveSession(engine, TickSlicer(), orchestrator, log,
                                      cancelled=sink.is_cancelled)
                tasks = [asyncio.create_task(c) for c in (
                    _mic_task(duplex), _stt_task(duplex, session),
                    _tts_task(duplex, session, player), _tick_task(session))]
                try:
                    await asyncio.wait(tasks, timeout=args.duration,
                                       return_when=asyncio.FIRST_COMPLETED)
                except asyncio.CancelledError:
                    pass  # Ctrl-C: asyncio.run cancels this task; stop cleanly below
                for task in tasks:
                    task.cancel()
                for task, outcome in zip(tasks, await asyncio.gather(
                        *tasks, return_exceptions=True)):
                    if isinstance(outcome, Exception):
                        logger.error("%s stopped: %s", task.get_coro().__name__,
                                     http_error_message(engine, outcome) or repr(outcome))
                if orchestrator.open:
                    try:
                        await sink.done(orchestrator.request_id)
                    except (ConnectionClosed, VoiceManagerError):
                        pass
    finally:
        player.close()
        await client.__aexit__(None, None, None)
    write_wav(wav_path, bytes(player.played), player.sample_rate or client.last_tts_sample_rate)
    print(f"log: {log_path}  wav: {wav_path}")
    if session is not None:
        print(session.summary_line())


def add_server_options(parser: argparse.ArgumentParser) -> None:
    """The llama-server options the replay and micro-turn modes share."""
    parser.add_argument("--url", default="http://127.0.0.1:8080",
                        help="llama-server or llama-swap base URL")
    parser.add_argument("--model", default=None,
                        help="model name for llama-swap routing (omit for plain llama-server)")


def add_replay_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of the replay mode: a scripted call as text, no audio."""
    add_server_options(parser)
    parser.add_argument("--script", type=pathlib.Path, default=None,
                        help="one heard string per line, empty line for silence (default: demo)")


def add_live_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of the micro-turn live mode."""
    add_server_options(parser)
    parser.add_argument("--vm-host", default="localhost", help="VoiceManager host")
    parser.add_argument("--vm-port", type=int, default=8255, help="VoiceManager port")
    parser.add_argument("--voice", default=None, help="TTS voice (default: the token's voice)")
    parser.add_argument("--duration", type=float, default=None,
                        help="stop after this many seconds (default: at Ctrl-C)")
    parser.add_argument("--log", default=None,
                        help="JSON lines log (default: runs/microturn/microturn-<stamp>.jsonl)")
    parser.add_argument("--wav", default=None,
                        help="WAV of the audio played (default: runs/microturn/microturn-<stamp>.wav)")
    parser.add_argument("--tts-mode", choices=TTS_MODES, default="clause",
                        help="clause: the server cuts each utterance into small pieces; "
                             "flush: flush each speaking tick's words (default: clause)")


def run_replay(args: argparse.Namespace) -> None:
    """Replay a script through llama-server and print each tick and a summary."""
    engine = LlamaServerEngine(args.url, model=args.model)
    script = read_script(args.script) if args.script else DEMO
    results: list[TickResult] = []
    try:
        replay(engine, script, results=results)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(http_error_message(engine, exc)) from exc
    print(summary(results))


if __name__ == "__main__":
    sys.stderr.write("the command line moved: run scripts/voice_agent.py "
                     "(replay | microturn | classic)\n")
    sys.exit(2)
