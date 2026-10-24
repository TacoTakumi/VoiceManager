"""Tests for the micro-turn client's replay mode: tick lines, history and script files.

Model-free and server-free. A fake completion engine stands in for llama-server.
"""
import importlib.util
import math
import pathlib
import sys

_PATH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "microturn_client.py"
if "microturn_client" in sys.modules:
    mc = sys.modules["microturn_client"]
else:
    _spec = importlib.util.spec_from_file_location("microturn_client", _PATH)
    mc = importlib.util.module_from_spec(_spec)
    # dataclasses look the module up in sys.modules while the class body runs.
    sys.modules["microturn_client"] = mc
    _spec.loader.exec_module(mc)

LISTEN = "<listen>"
NOVOICE = "<no voice>"
SCRIPT = ["Hey I'm cleaning", "the bathroom", NOVOICE, "bleach here"]


def _response(content: str, top: list[tuple[int, str, float]]) -> dict:
    """A llama-server /completion JSON with first-token candidates (id, token, prob)."""
    return {
        "content": content,
        "completion_probabilities": [{
            "top_logprobs": [{"id": i, "token": t, "logprob": math.log(p)} for i, t, p in top],
        }],
    }


class ReplayFakeEngine:
    """Listens to user words and speaks on silence; records each request prompt."""

    def __init__(self):
        self.prompts: list[str] = []

    def complete(self, body: dict) -> dict:
        prompt = body["prompt"]
        self.prompts.append(prompt)
        last_user = prompt.rsplit("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
        if last_user == NOVOICE:
            return _response(" Do not mix", [(11, " Do", 0.95), (10, "<", 0.05)])
        return _response(LISTEN, [(10, "<", 0.9), (11, " Do", 0.1)])


def test_replay_one_line_per_tick():
    lines: list[str] = []
    mc.replay(ReplayFakeEngine(), SCRIPT, out=lines.append)
    assert len(lines) == len(SCRIPT)
    for line, heard in zip(lines, SCRIPT):
        said = "Do not mix" if heard == NOVOICE else LISTEN
        assert heard in line
        assert said in line
        assert "P(listen)" in line
        assert line.endswith(" ms")


def test_replay_history_grows():
    engine = ReplayFakeEngine()
    history = mc.replay(engine, SCRIPT, out=lambda line: None)
    expected = []
    for heard in SCRIPT:
        expected += [("user", heard), ("assistant", "Do not mix" if heard == NOVOICE else LISTEN)]
    assert history == expected
    # One request per tick (P(listen) is never under the threshold), each extending the last.
    assert len(engine.prompts) == len(SCRIPT)
    for prev, cur in zip(engine.prompts, engine.prompts[1:]):
        assert cur.startswith(prev) and len(cur) > len(prev)


def test_replay_script_file(tmp_path):
    path = tmp_path / "call.txt"
    path.write_text("Hey I'm cleaning\nthe bathroom\n\n  bleach here  \n")
    assert mc.read_script(path) == ["Hey I'm cleaning", "the bathroom", NOVOICE, "bleach here"]


def test_replay_demo_is_the_card_call():
    assert mc.DEMO[:6] == ["Hey I'm cleaning", "the bathroom", "right now and", "I got some",
                           "bleach here", NOVOICE]
    assert mc.DEMO[6:11] == ["and I'm going", "to mix it", "with the ammonia",
                             "cleaner so it", "gets extra strong"]
    assert mc.DEMO[11:] == [NOVOICE] * 7


def test_format_tick():
    result = mc.TickResult(said="Do not mix", p_listen=0.054, turn_ms=231.4, requests=1)
    expected = ("  heard: bleach here" + " " * 11 + " said: Do not mix" + " " * 18
                + " P(listen) 0.05  231 ms")
    assert mc.format_tick("bleach here", result) == expected


def test_replay_summary():
    results = [
        mc.TickResult(said=LISTEN, p_listen=0.9, turn_ms=100.0, requests=1),
        mc.TickResult(said="Do not mix", p_listen=0.05, turn_ms=300.0, requests=1),
        mc.TickResult(said=LISTEN, p_listen=0.8, turn_ms=200.0, requests=1),
    ]
    assert mc.summary(results) == "3 ticks, median turn 200 ms, spoke on 1"
