"""Reference client for micro-turn full-duplex voice.

Every 0.6 s the client sends the language model the words heard in that window, or
<no voice>, and the model answers with up to three words to speak next, or <listen>.

The model is Qwen3.6-35B-A3B with the micro-turn duplex LoRA, served by llama.cpp's
llama-server started with the base GGUF and ``--lora microturn-duplex-lora.gguf`` (or a
llama-swap entry that serves it; pass ``--model`` so llama-swap can route the request).

The system prompt below is the exact text the adapter was trained on. Keep it byte for
byte, and keep the history append-only so llama-server's prompt cache resumes each tick.

    python scripts/microturn_client.py --url http://127.0.0.1:8080 [--model NAME]
"""
import argparse
import json
import logging
import math
import time
import urllib.request
import uuid
from collections.abc import Callable
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
        # Per segment: the words treated as sent. After a revision this holds the new
        # text's prefix, so the same revision is not counted again on later frames.
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
            n = len(sent)
            if len(words) < n or words[:n] != sent:
                self.revisions += 1
                self.last_revised = True
                logger.debug("segment %d revised: sent=%r now=%r", seg, sent, words)
            if len(words) > n:
                emitted.extend(words[n:])
                self._sent[seg] = words
            elif len(words) == n:
                self._sent[seg] = words
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

    def __init__(self, base_url: str, model: str | None = None, timeout: float = 30.0):
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

    async def send_text(self, request_id: str, text: str) -> None:
        """Append text to the request, opening it on first use."""
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


class TTSOrchestrator:
    """Turns each tick's decision into VoiceManager streaming TTS calls.

    One request per assistant utterance, with a flush per speaking tick, so the audio
    of each tick's words starts within a tick. The first LISTEN after speech ends the
    request with ``done``. The model is never told what audio is playing, so the client
    stops playback itself when the user talks over it: user words, a LISTEN and audio
    still playing cancel the request and clear the player.
    """

    def __init__(self, sink: TTSSink, player: Player,
                 new_id: Callable[[], str] = lambda: str(uuid.uuid4())):
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
                text = f" {said}"
            else:
                self.request_id, self.open, self.live = self.new_id(), True, True
                text = said
            await self.sink.send_text(self.request_id, text)
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


def main() -> None:
    """Parse the server options. Modes that drive a call are added on top of this core."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8080",
                        help="llama-server or llama-swap base URL")
    parser.add_argument("--model", default=None,
                        help="model name for llama-swap routing (omit for plain llama-server)")
    args = parser.parse_args()
    print(f"usage: microturn_client.py --url {args.url}"
          f"{f' --model {args.model}' if args.model else ''} (no mode selected yet)")


if __name__ == "__main__":
    main()
