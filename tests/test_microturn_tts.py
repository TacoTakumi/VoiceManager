"""Tests for the micro-turn client's TTS orchestration: request lifecycle and cut-in.

Model-free and server-free. A fake sink stands in for the VoiceManager WebSocket and a
fake player stands in for the audio output.
"""
import asyncio
import importlib.util
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


class FakeSink:
    """Records every TTS call as a tuple, in order."""

    def __init__(self):
        self.calls: list[tuple] = []

    async def send_text(self, request_id: str, text: str) -> None:
        self.calls.append(("text", request_id, text))

    async def flush(self, request_id: str) -> None:
        self.calls.append(("flush", request_id))

    async def done(self, request_id: str) -> None:
        self.calls.append(("done", request_id))

    async def cancel(self, request_id: str) -> None:
        self.calls.append(("cancel", request_id))


class FakePlayer:
    """A player with a settable playing flag and a queue that clear() empties."""

    def __init__(self, playing: bool = False, queue: list | None = None):
        self.playing = playing
        self.queue = list(queue or [])
        self.clears = 0

    def is_playing(self) -> bool:
        return self.playing

    def clear(self) -> None:
        self.queue.clear()
        self.clears += 1


def _ids():
    """Deterministic request ids: req-1, req-2, ..."""
    n = 0

    def new_id() -> str:
        nonlocal n
        n += 1
        return f"req-{n}"

    return new_id


def _run(orch, ticks):
    async def go():
        for heard, said in ticks:
            await orch.on_tick(heard, said)

    asyncio.run(go())


def _make(playing: bool = False, queue: list | None = None):
    sink, player = FakeSink(), FakePlayer(playing, queue)
    return sink, player, mc.TTSOrchestrator(sink, player, new_id=_ids())


def test_tts_sequence():
    sink, player, orch = _make()
    _run(orch, [(NOVOICE, LISTEN), (NOVOICE, "Do not mix"), (NOVOICE, "those chemicals."),
                (NOVOICE, LISTEN), (NOVOICE, LISTEN)])
    assert sink.calls == [
        ("text", "req-1", "Do not mix"), ("flush", "req-1"),
        ("text", "req-1", " those chemicals."), ("flush", "req-1"),
        ("done", "req-1"),
    ]
    assert orch.request_id == "req-1"
    assert player.clears == 0


def test_cutin_cancels_and_clears():
    sink, player, orch = _make()
    _run(orch, [(NOVOICE, "Do not mix")])
    player.playing, player.queue = True, [b"chunk1", b"chunk2"]
    _run(orch, [("wait stop", LISTEN)])
    assert [c for c in sink.calls if c[0] == "cancel"] == [("cancel", "req-1")]
    assert not [c for c in sink.calls if c[0] == "done"]
    assert player.queue == []
    # Nothing more after the cut-in, even on later LISTEN ticks.
    before = list(sink.calls)
    _run(orch, [(NOVOICE, LISTEN)])
    assert sink.calls == before


def test_cutin_after_done_still_cancels():
    sink, player, orch = _make()
    _run(orch, [(NOVOICE, "Do not mix"), (NOVOICE, LISTEN)])
    assert sink.calls[-1] == ("done", "req-1")
    player.playing, player.queue = True, [b"chunk"]
    _run(orch, [("hold on", LISTEN)])
    assert sink.calls[-1] == ("cancel", "req-1")
    assert player.queue == []


def test_no_voice_listen_never_cancels():
    sink, player, orch = _make()
    _run(orch, [(NOVOICE, "Do not mix")])
    player.playing, player.queue = True, [b"chunk"]
    _run(orch, [(NOVOICE, LISTEN), (NOVOICE, LISTEN), ("", LISTEN)])
    assert not [c for c in sink.calls if c[0] == "cancel"]
    assert player.queue == [b"chunk"]


def test_cutin_needs_playing_audio():
    sink, player, orch = _make()
    _run(orch, [(NOVOICE, "Okay."), (NOVOICE, LISTEN), ("thanks", LISTEN)])
    assert not [c for c in sink.calls if c[0] == "cancel"]
    assert player.clears == 0


def test_tts_new_utterance_new_id():
    sink, player, orch = _make()
    _run(orch, [(NOVOICE, "Do not mix"), (NOVOICE, LISTEN), ("okay", LISTEN),
                (NOVOICE, "Open a window.")])
    assert sink.calls[-2:] == [("text", "req-2", "Open a window."), ("flush", "req-2")]
    assert orch.request_id == "req-2"
