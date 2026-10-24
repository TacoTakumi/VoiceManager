"""Tests for the micro-turn reference client core: prompt text, request body, listen rule.

Model-free and server-free. A fake completion engine stands in for llama-server.
"""
import importlib.util
import math
import pathlib
import sys

import pytest

_PATH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "microturn_client.py"
_spec = importlib.util.spec_from_file_location("microturn_client", _PATH)
mc = importlib.util.module_from_spec(_spec)
# dataclasses look the module up in sys.modules while the class body runs.
sys.modules["microturn_client"] = mc
_spec.loader.exec_module(mc)

# Written out literally, not taken from the module, so a drift in the constant fails.
SYSTEM_TEXT = (
    "You are a voice assistant in a live phone call; both sides can talk at the same time. "
    "Every 0.6 seconds you get what the microphone heard in that moment, as a streaming "
    "transcript fragment, or <no voice>. You reply with up to three new words to say next, "
    "or <listen> to add nothing."
)

HISTORY = [
    ("user", "Hey I'm cleaning"), ("assistant", "<listen>"),
    ("user", "<no voice>"), ("assistant", "Do not mix"),
    ("user", "the bathroom"), ("assistant", "<listen>"),
]


class FakeEngine:
    """Returns canned /completion responses in order and records each request body."""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.bodies: list[dict] = []

    def complete(self, body: dict) -> dict:
        self.bodies.append(dict(body))
        return self.responses.pop(0)


def _response(content: str, top: list[tuple[int, str, float]] | None) -> dict:
    """A llama-server /completion JSON with first-token candidates (id, token, prob)."""
    resp: dict = {"content": content}
    if top is not None:
        resp["completion_probabilities"] = [{
            "top_logprobs": [{"id": i, "token": t, "logprob": math.log(p)} for i, t, p in top],
        }]
    return resp


def test_prompt_three_ticks_exact():
    expected = (
        "<|im_start|>system\n" + SYSTEM_TEXT + "<|im_end|>\n"
        "<|im_start|>user\nHey I'm cleaning<|im_end|>\n"
        "<|im_start|>assistant\n<listen><|im_end|>\n"
        "<|im_start|>user\n<no voice><|im_end|>\n"
        "<|im_start|>assistant\nDo not mix<|im_end|>\n"
        "<|im_start|>user\nthe bathroom<|im_end|>\n"
        "<|im_start|>assistant\n<listen><|im_end|>\n"
        "<|im_start|>user\nright now and<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert mc.build_prompt(HISTORY, "right now and") == expected
    assert mc.SYSTEM == SYSTEM_TEXT


def test_prompt_prefix_property():
    history: list[tuple[str, str]] = []
    ticks = [("Hey I'm cleaning", "<listen>"), ("<no voice>", "Do not mix"),
             ("the bathroom", "<listen>"), ("bleach here", "That is dangerous")]
    tail = "<|im_start|>assistant\n"
    for heard, said in ticks:
        prompt = mc.build_prompt(history, heard)
        assert prompt.endswith(tail)
        history = history + [("user", heard), ("assistant", said)]
        grown = prompt[:-len(tail)] + tail + said + "<|im_end|>\n"
        assert mc.build_prompt(history, "next").startswith(grown)


def test_completion_body_fields():
    body = mc.completion_body("PROMPT")
    assert body["prompt"] == "PROMPT"
    assert body["cache_prompt"] is True
    assert body["temperature"] == 0
    assert body["n_probs"] > 0
    assert body["n_predict"] == 64
    assert "<|im_end|>" in body["stop"]
    assert "model" not in body
    assert mc.completion_body("PROMPT", model="microturn")["model"] == "microturn"


def test_listen_rule_bans_below_threshold():
    first = _response("<listen>", [(27, "<", 0.20), (99, "<l", 0.07), (5, " Do", 0.15),
                                   (6, " Please", 0.12), (7, " Wait", 0.10)])
    engine = FakeEngine([first, _response(" Do not mix", None)])
    result = mc.decide(engine, HISTORY, "bleach here")
    assert result.requests == 2
    assert len(engine.bodies) == 2
    second = engine.bodies[1]
    assert sorted(second["logit_bias"]) == [[27, False], [99, False]]
    assert second["n_probs"] == 0
    assert result.said == "Do not mix"
    assert result.p_listen == pytest.approx(0.27)
    assert result.turn_ms >= 0


def test_listen_rule_listens_at_threshold_or_more():
    first = _response("<listen>", [(27, "<", 0.34), (5, " Do", 0.20)])
    engine = FakeEngine([first])
    result = mc.decide(engine, HISTORY, "<no voice>")
    assert result.requests == 1
    assert len(engine.bodies) == 1
    assert result.said == mc.LISTEN
    assert result.p_listen == pytest.approx(0.34)


def test_listen_rule_speaks_when_model_speaks():
    first = _response(" Stop right there", [(5, " Stop", 0.80), (27, "<", 0.05)])
    engine = FakeEngine([first])
    result = mc.decide(engine, HISTORY, "with the ammonia")
    assert result.requests == 1
    assert result.said == "Stop right there"


def test_listen_rule_without_probabilities():
    engine = FakeEngine([_response("<listen>", None)])
    result = mc.decide(engine, HISTORY, "<no voice>")
    assert result.p_listen == 0.0
    assert result.requests == 1
    assert result.said == mc.LISTEN


def test_listen_rule_sends_model_name():
    engine = FakeEngine([_response(" Hello", [(5, " Hello", 0.9)])])
    mc.decide(engine, [], "hi", model="microturn")
    assert engine.bodies[0]["model"] == "microturn"


def test_script_not_collected():
    assert not _PATH.name.startswith("test_")
    assert _PATH.parent.name == "scripts"
    assert "tests" not in _PATH.parent.parts[-1:]
