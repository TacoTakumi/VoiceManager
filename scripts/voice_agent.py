"""Reference voice agent over VoiceManager: the classic turn pattern and the micro-turn one.

Three subcommands, one per mode:

- ``classic``: wait for the person's turn, send it to any OpenAI-compatible chat
  endpoint, stream the reply through VoiceManager's TTS, and stop the reply when the
  person speaks over it. The turn ends on the server's endpoint frame (or the final
  transcript on engines without one); if the final revises the text, the reply is
  cancelled and the revision is sent instead.
- ``microturn``: the experimental full-duplex loop. Every 0.6 s the LLM with the
  micro-turn LoRA gets the words heard in that window and answers with a few words to
  speak or ``listen``. Needs the LoRA model under llama-server or llama-swap.
- ``replay``: the micro-turn loop on a scripted call as text, no audio.

Both live modes stream the microphone and play the speaker through the default audio
device: use headphones, there is no echo cancellation. They write a JSON lines log and
a WAV of what played under ``runs/``. The bearer token comes from the
``VOICEMANAGER_TOKEN`` environment variable.

    python scripts/voice_agent.py classic [--vm-host HOST] [--vm-port PORT] [--voice NAME]
        [--url URL] [--model NAME] [--tts-mode {clause,sentence}] [--think]
        [--duration SECONDS]
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
from typing import Callable

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import microturn_client  # noqa: E402
from microturn_client import AudioPlayer, _heard_words, write_wav  # noqa: E402

logger = logging.getLogger("voice_agent")

SYSTEM_PROMPT = ("You are a voice assistant talking with a person. Answer in one or two "
                 "short spoken sentences of plain text: no lists, no markdown, no headings.")
CLASSIC_RUN_DIR = pathlib.Path("runs") / "classic"


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
    """

    def __init__(self, duplex, chat_factory: Callable[[list[dict]], ChatStream], player,
                 log, clock: Callable[[], float] = time.monotonic, voice: str | None = None,
                 chunking: str = "clause", system_prompt: str = SYSTEM_PROMPT,
                 on_error: Callable[[str], None] = print,
                 say: Callable[[str], None] = lambda line: None):
        self.duplex = duplex
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
        self._t0 = clock()

    # --- the listening turn ---------------------------------------------------------

    async def open_turn(self) -> int:
        """Open the next listening turn; frames of earlier turns are ignored from now."""
        self.turn_id = await self.duplex.turn_start()
        self.segments = {}
        self.dispatched_words = None
        self._user_index = None
        self._dispatched_keys = set()
        return self.turn_id

    def turn_text(self) -> str:
        return " ".join(self.segments[k] for k in sorted(self.segments)).strip()

    async def on_transcript(self, frame: dict, now: float) -> None:
        """Feed one transcription frame: upsert its segment, dispatch or supersede."""
        if self.turn_id is not None and frame.get("turn_id", 0) != self.turn_id:
            return
        heard_before = bool(_heard_words(self.turn_text()))
        seg = frame.get("segment_index", 0)
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
                text = self.turn_text()
                words = _heard_words(text)
                if not words:
                    return
        if endpointed and words != self.dispatched_words:
            await self._dispatch(text, words, now)

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

    async def _cancel_reply(self, now: float, reason: str) -> None:
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
        await reply.request.cancel()
        self.player.clear()
        if reason == "superseded":
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
        if reply.ended:
            return  # a cancelled reply's ack: the turn goes on listening
        reply.ended = True
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

    # --- the record -----------------------------------------------------------------

    def _write_utterance(self, reply: Reply) -> None:
        delay = "-" if reply.first_audio_ms is None else f"{reply.first_audio_ms:.0f} ms"
        state = f"cancelled, {reply.reason}" if reply.cancelled else f"first audio {delay}"
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


def classic_output_paths(args: argparse.Namespace) -> tuple[str, str]:
    """The log and WAV paths of a classic run; defaults go in ``runs/classic/``."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    if not (args.log and args.wav):
        CLASSIC_RUN_DIR.mkdir(parents=True, exist_ok=True)
    return (args.log or str(CLASSIC_RUN_DIR / f"classic-{stamp}.jsonl"),
            args.wav or str(CLASSIC_RUN_DIR / f"classic-{stamp}.wav"))


async def classic(args: argparse.Namespace) -> None:
    """Run a classic call over VoiceManager until Ctrl-C or ``--duration`` seconds."""
    token = os.environ.get("VOICEMANAGER_TOKEN")
    if not token:
        raise SystemExit("error: set VOICEMANAGER_TOKEN to a VoiceManager bearer token")
    from voicemanager.client import VoiceManagerClient, VoiceManagerError

    log_path, wav_path = classic_output_paths(args)
    print("warning: use headphones; there is no echo cancellation", flush=True)
    if not await warm_up_chat(args.url, args.model, think=args.think):
        raise SystemExit(2)
    client = VoiceManagerClient(host=args.vm_host, port=args.vm_port, token=token)
    try:
        await client.__aenter__()
    except (VoiceManagerError, OSError) as exc:
        raise SystemExit(f"error: cannot connect to VoiceManager: {exc}") from exc
    player = AudioPlayer()
    session = None

    def say(line: str) -> None:
        print(line, flush=True)

    try:
        async with client.duplex() as duplex:
            with open(log_path, "w") as log:
                session = ClassicSession(
                    duplex, lambda messages: ChatStream(args.url, args.model, messages,
                                                        think=args.think),
                    player, log, voice=args.voice, chunking=args.tts_mode, say=say)
                await session.open_turn()
                say(f"listening (stt {client.stt_engine}, tts {client.tts_engine}); "
                    "speak, pause, and the reply follows; Ctrl-C to stop")

                async def transcripts() -> None:
                    async for frame in duplex.transcripts():
                        await session.on_transcript(frame, session.clock())
                    logger.warning("VoiceManager closed the connection")

                async def audio() -> None:
                    async for event in duplex.audio():
                        await session.on_event(event, session.clock())

                tasks = [asyncio.create_task(c) for c in (
                    microturn_client._mic_task(duplex), transcripts(), audio())]
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
    write_wav(wav_path, bytes(player.played), player.sample_rate or client.last_tts_sample_rate)
    print(f"log: {log_path}  wav: {wav_path}")
    if session is not None:
        print(session.summary_line())


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
