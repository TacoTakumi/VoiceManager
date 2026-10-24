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
