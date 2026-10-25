"""Clause mode on the streaming TTS pipelines.

A ``tts_stream`` request opened with ``chunking="clause"`` sends its text to the
engine in clause-sized pieces cut by ClauseChunker (see
``voicemanager.pipeline.clause_chunker``), instead of whole sentences:

- a clause mark cuts only when the piece up to it holds the minimum word count;
- while nothing is playing, a piece goes at such a clause mark, at the first-piece
  word count, or at the first-piece deadline;
- while audio is playing, held words wait for such a clause mark, the later-piece
  word count, or the unplayed audio dropping below the margin;
- done sends all held text at once and ends the request with one audio_end;
- flush sends all held text at once and leaves the request open;
- the mode is fixed by the request's first frame.

A sentence request takes the old path untouched.

Model-free and server-free: fake engines record the text they get and when, a fake
WebSocket records the frames. Rule tests that need no clock run on word counts and
marks alone, a fake clock drives the deadline rule deterministically, and the few
timer tests use real time with thresholds of tens of milliseconds and wide windows.
"""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import authkit

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voicemanager.app import create_app
from voicemanager.pipeline.audiocpp_tts import AudioCppTTSStreamPipeline
from voicemanager.pipeline.tts import TTSPipeline, TTSStreamPipeline
from voicemanager.pipeline.tts_session import TTSSession
from voicemanager.session import SessionRegistry

# Thresholds large enough that no timer fires during a test that does not want one.
_NO_TIMER = {"clause_first_deadline_ms": 60_000}


# --- test doubles -----------------------------------------------------------

class _Engine:
    """Streaming engine stub: records (text, loop time) per call, yields one chunk.

    ``audio_s`` sets the duration of the chunk at ``sample_rate``; None yields an
    empty chunk, so the unplayed estimate stays 0 and only the first-piece rule
    applies.
    """

    is_batch = False

    def __init__(self, audio_s: float | None = None, sample_rate: int = 24000):
        self.calls: list[tuple[str, float]] = []
        self._audio_s = audio_s
        self._sample_rate = sample_rate

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def texts(self) -> list[str]:
        return [text for text, _ in self.calls]

    async def ensure_ready(self) -> None:  # the audiocpp pipeline awaits this
        return None

    async def stream_synthesize(self, text: str, **kwargs):
        self.calls.append((text, asyncio.get_running_loop().time()))
        n = 0 if self._audio_s is None else int(round(self._audio_s * self._sample_rate))
        yield np.zeros(n, dtype=np.float32)


class _WS:
    def __init__(self):
        self.frames: list[tuple[str, object]] = []

    async def send_text(self, data: str) -> None:
        self.frames.append(("text", json.loads(data)))

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append(("bytes", data))

    def kinds(self) -> list[str]:
        return [kind if kind != "text" else p["type"] for kind, p in self.frames]

    def audio_ends(self) -> list[dict]:
        return [p for kind, p in self.frames if kind == "text" and p["type"] == "audio_end"]


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _pipe(engine, cls=TTSStreamPipeline, **kwargs):
    return cls(TTSPipeline(engine=engine, executor=None), **kwargs)


async def _until(predicate, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting on the condition")
        await asyncio.sleep(0.005)


async def _finish(pipe, ws, request_id="r1") -> None:
    await pipe.feed_text("", True, ws, request_id)
    await asyncio.wait_for(pipe.synthesis_task, 2.0)


def _other_tasks() -> set:
    return {t for t in asyncio.all_tasks() if t is not asyncio.current_task()}


PIPELINES = [
    pytest.param(TTSStreamPipeline, id="shared"),
    pytest.param(AudioCppTTSStreamPipeline, id="audiocpp"),
]


# --- routing ----------------------------------------------------------------

def test_a_sentence_request_keeps_the_sentence_rule():
    async def run():
        engine, ws = _Engine(), _WS()
        pipe = _pipe(engine, **_NO_TIMER)
        await pipe.feed_text("Well, I think so", False, ws, "r1")
        await asyncio.sleep(0.05)
        assert engine.texts == []  # one sentence fragment: held, as before
        await _finish(pipe, ws)
        assert engine.texts == ["Well, I think so"]

    asyncio.run(run())


@pytest.mark.parametrize("cls", PIPELINES)
def test_a_clause_mark_sends_the_piece_at_once(cls):
    async def run():
        engine, ws = _Engine(), _WS()
        pipe = _pipe(engine, cls, **_NO_TIMER)
        await pipe.feed_text("Do not mix them. It creates", False, ws, "r1",
                             chunking="clause")
        await _until(lambda: engine.texts == ["Do not mix them."])
        assert pipe._clause_chunker.held_text == "It creates"
        await _finish(pipe, ws)
        assert engine.texts == ["Do not mix them.", "It creates"]

    asyncio.run(run())


@pytest.mark.parametrize("cls", PIPELINES)
def test_a_clause_mark_after_one_word_does_not_cut(cls):
    async def run():
        engine, ws = _Engine(), _WS()
        pipe = _pipe(engine, cls, **_NO_TIMER)
        await pipe.feed_text("Yes, I", False, ws, "r1", chunking="clause")
        await asyncio.sleep(0.05)
        assert engine.texts == []
        await pipe.feed_text(" can hear you fine. What", False, ws, "r1")
        await _until(lambda: engine.texts == ["Yes, I can hear you fine."])
        await _finish(pipe, ws)
        assert engine.texts == ["Yes, I can hear you fine.", "What"]

    asyncio.run(run())


@pytest.mark.parametrize("cls", PIPELINES)
def test_the_mark_minimum_is_passed_to_the_chunker(cls):
    async def run():
        engine, ws = _Engine(), _WS()
        pipe = _pipe(engine, cls, clause_mark_min_words=1, **_NO_TIMER)
        await pipe.feed_text("Yes, I", False, ws, "r1", chunking="clause")
        await _until(lambda: engine.texts == ["Yes,"])
        await _finish(pipe, ws)
        assert engine.texts == ["Yes,", "I"]

    asyncio.run(run())


@pytest.mark.parametrize("cls", PIPELINES)
def test_the_default_margin_and_mark_minimum(cls):
    async def run():
        engine, ws = _Engine(), _WS()
        pipe = _pipe(engine, cls, **_NO_TIMER)
        await pipe.feed_text("Hi", False, ws, "r1", chunking="clause")
        assert pipe._clause_chunker.unplayed_margin_ms == 1000
        assert pipe._clause_chunker.mark_min_words == 3
        await _finish(pipe, ws)

    asyncio.run(run())


@pytest.mark.parametrize("cls", PIPELINES)
def test_the_first_piece_goes_at_the_first_piece_word_count(cls):
    async def run():
        engine, ws = _Engine(), _WS()
        pipe = _pipe(engine, cls, clause_first_words=3, **_NO_TIMER)
        await pipe.feed_text("one two", False, ws, "r1", chunking="clause")
        await asyncio.sleep(0.05)
        assert engine.texts == []
        await pipe.feed_text(" three", False, ws, "r1")
        await _until(lambda: engine.texts == ["one two three"])
        await _finish(pipe, ws)

    asyncio.run(run())


@pytest.mark.parametrize("cls", PIPELINES)
def test_the_first_piece_deadline_follows_the_injected_clock(cls):
    async def run():
        engine, ws, clock = _Engine(), _WS(), _Clock()
        pipe = _pipe(engine, cls, clause_first_deadline_ms=600, clause_clock=clock)
        await pipe.feed_text("Hi", False, ws, "r1", chunking="clause")
        clock.t = 0.5
        await pipe.feed_text(" there", False, ws, "r1")
        await asyncio.sleep(0.05)
        assert engine.texts == []  # 0.5 s after the first word: not yet
        clock.t = 0.6
        await pipe.feed_text("", False, ws, "r1")
        await _until(lambda: engine.texts == ["Hi there"])
        await _finish(pipe, ws)

    asyncio.run(run())


@pytest.mark.parametrize("cls", PIPELINES)
def test_the_timer_sends_the_first_piece_at_the_deadline(cls):
    async def run():
        engine, ws = _Engine(), _WS()
        pipe = _pipe(engine, cls, clause_first_deadline_ms=80)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        await pipe.feed_text("Hi", False, ws, "r1", chunking="clause")
        await asyncio.sleep(0.03)
        assert engine.texts == []
        await _until(lambda: engine.texts == ["Hi"])
        assert engine.calls[0][1] - t0 >= 0.075
        await _finish(pipe, ws)

    asyncio.run(run())


@pytest.mark.parametrize("cls", PIPELINES)
def test_held_words_wait_for_the_margin_while_audio_plays(cls):
    async def run():
        # Each piece is 0.4 s of audio; the margin is 0.1 s, so text held while
        # the first piece plays goes about 0.3 s after its audio was sent.
        engine, ws = _Engine(audio_s=0.4, sample_rate=1000), _WS()
        pipe = _pipe(engine, cls, clause_first_words=2, clause_first_deadline_ms=40,
                     clause_later_words=12, clause_margin_ms=100)
        await pipe.feed_text("One two", False, ws, "r1", chunking="clause")
        await _until(lambda: len(engine.calls) == 1)
        first_at = engine.calls[0][1]
        await pipe.feed_text(" three four five", False, ws, "r1")
        # Past the first-piece deadline and word count: the later rule holds them.
        await asyncio.sleep(0.12)
        assert engine.texts == ["One two"]
        await _until(lambda: len(engine.calls) == 2, timeout=3.0)
        assert engine.texts == ["One two", "three four five"]
        assert engine.calls[1][1] - first_at >= 0.27
        await _finish(pipe, ws)

    asyncio.run(run())


@pytest.mark.parametrize("cls", PIPELINES)
def test_the_later_piece_word_count_sends_while_audio_plays(cls):
    async def run():
        engine, ws = _Engine(audio_s=5.0, sample_rate=1000), _WS()
        pipe = _pipe(engine, cls, clause_first_words=2, clause_later_words=4, **_NO_TIMER)
        await pipe.feed_text("One two", False, ws, "r1", chunking="clause")
        await _until(lambda: len(engine.calls) == 1)
        await pipe.feed_text(" three four five", False, ws, "r1")
        await asyncio.sleep(0.05)
        assert engine.texts == ["One two"]  # 3 words, 5 s still unplayed
        await pipe.feed_text(" six", False, ws, "r1")
        await _until(lambda: engine.texts == ["One two", "three four five six"])
        await _finish(pipe, ws)

    asyncio.run(run())


class _BusyEngine(_Engine):
    """An engine whose first request stays in flight until ``release`` is set."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.release = asyncio.Event()

    async def stream_synthesize(self, text: str, **kwargs):
        first = not self.calls
        self.calls.append((text, asyncio.get_running_loop().time()))
        if first:
            await self.release.wait()
        n = 0 if self._audio_s is None else int(round(self._audio_s * self._sample_rate))
        yield np.zeros(n, dtype=np.float32)


def test_audiocpp_merges_pieces_cut_while_busy_into_one_request():
    async def run():
        engine, ws = _BusyEngine(), _WS()
        pipe = _pipe(engine, AudioCppTTSStreamPipeline, clause_first_words=2, **_NO_TIMER)
        await pipe.feed_text("Hi there friend,", False, ws, "r1", chunking="clause")
        await _until(lambda: engine.texts == ["Hi there friend,"])
        # Two pieces become ready while the first request is still synthesizing:
        # one at a clause mark, one at the first-piece word count.
        await pipe.feed_text(" Well, I see,", False, ws, "r1")
        await pipe.feed_text(" one two", False, ws, "r1")
        assert pipe._clause_chunker.held_text == ""  # both were cut, none held
        await asyncio.sleep(0.05)
        assert engine.texts == ["Hi there friend,"]
        engine.release.set()
        # They arrive as ONE next request, joined by a space, punctuation kept.
        await _until(lambda: len(engine.calls) == 2)
        assert engine.texts == ["Hi there friend,", "Well, I see, one two"]
        await _finish(pipe, ws)
        assert engine.texts == ["Hi there friend,", "Well, I see, one two"]
        assert ws.kinds() == ["audio_start", "bytes", "bytes", "audio_end"]

    asyncio.run(run())


@pytest.mark.parametrize("cls", PIPELINES)
def test_audio_reports_are_ignored_on_a_sentence_request(cls):
    async def run():
        engine, ws = _Engine(), _WS()
        pipe = _pipe(engine, cls)
        await pipe.feed_text("Hello there", False, ws, "r1")
        pipe._report_audio("r1", 48000, 24000)  # no chunker: a no-op
        await _finish(pipe, ws)
        assert engine.texts == ["Hello there"]

    asyncio.run(run())


# --- the mode is fixed by the first frame -----------------------------------

def test_later_frames_cannot_change_the_mode():
    async def run():
        engine, ws = _Engine(), _WS()
        pipe = _pipe(engine, **_NO_TIMER)
        await pipe.feed_text("Yes, I agree, ", False, ws, "r1", chunking="clause")
        await pipe.feed_text("and no more, ", False, ws, "r1", chunking="sentence")
        await pipe.feed_text("or maybe not; ", False, ws, "r1")
        await pipe.feed_text("fine", False, ws, "r1", chunking="sentence")
        await _until(lambda: len(engine.calls) == 3)
        await _finish(pipe, ws)
        assert engine.texts == ["Yes, I agree,", "and no more,", "or maybe not;", "fine"]

        # Opened without the field: sentence rule, whatever later frames say.
        engine.calls.clear()
        await pipe.feed_text("Yes, and", False, ws, "r2")
        await pipe.feed_text(" no, ", False, ws, "r2", chunking="clause")
        await asyncio.sleep(0.05)
        assert engine.texts == []
        await _finish(pipe, ws, "r2")
        assert engine.texts == ["Yes, and no,"]

    asyncio.run(run())


# --- done and flush ---------------------------------------------------------

@pytest.mark.parametrize("cls", PIPELINES)
def test_done_sends_held_text_at_once_and_ends_with_one_audio_end(cls):
    async def run():
        engine, ws = _Engine(), _WS()
        pipe = _pipe(engine, cls, **_NO_TIMER)
        await pipe.feed_text("two words", False, ws, "r1", chunking="clause")
        await asyncio.sleep(0.05)
        assert engine.texts == []
        await _finish(pipe, ws)
        assert engine.texts == ["two words"]
        assert ws.kinds() == ["audio_start", "bytes", "audio_end"]
        assert ws.audio_ends()[0]["reason"] == "complete"
        await asyncio.sleep(0)
        assert _other_tasks() == set()  # the timer went with the request

    asyncio.run(run())


@pytest.mark.parametrize("cls", PIPELINES)
def test_flush_sends_held_text_and_leaves_the_request_open(cls):
    async def run():
        engine, ws = _Engine(), _WS()
        pipe = _pipe(engine, cls, **_NO_TIMER)
        await pipe.feed_text("two words", False, ws, "r1", chunking="clause")
        await pipe.feed_text("", False, ws, "r1", flush=True)
        await _until(lambda: engine.texts == ["two words"])
        await asyncio.sleep(0.02)
        assert ws.audio_ends() == []
        # Same request, still the clause rule.
        await pipe.feed_text(" more text again, here", False, ws, "r1")
        await _until(lambda: engine.texts == ["two words", "more text again,"])
        await _finish(pipe, ws)
        assert engine.texts == ["two words", "more text again,", "here"]
        assert len(ws.audio_ends()) == 1

    asyncio.run(run())


@pytest.mark.parametrize("cls", PIPELINES)
def test_flush_timeout_sends_held_text_and_ends_the_request(cls):
    async def run():
        engine, ws = _Engine(), _WS()
        pipe = _pipe(engine, cls, **_NO_TIMER)
        await pipe.feed_text("two words", False, ws, "r1", chunking="clause")
        await pipe.flush_timeout(ws, "r1")
        await asyncio.wait_for(pipe.synthesis_task, 2.0)
        assert engine.texts == ["two words"]
        assert len(ws.audio_ends()) == 1
        await asyncio.sleep(0)
        assert _other_tasks() == set()

    asyncio.run(run())


# --- the timer goes with the request ----------------------------------------

@pytest.mark.parametrize("cls", PIPELINES)
def test_abandon_stops_the_timer(cls):
    async def run():
        engine, ws = _Engine(), _WS()
        pipe = _pipe(engine, cls, clause_first_deadline_ms=40)
        await pipe.feed_text("Hi", False, ws, "r1", chunking="clause")
        task = pipe.abandon()
        await asyncio.sleep(0.12)
        assert engine.texts == []
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert _other_tasks() == set()

    asyncio.run(run())


@pytest.mark.parametrize("cls", PIPELINES)
def test_cancel_and_dispose_stop_the_timer(cls):
    async def run():
        for stop in ("cancel", "dispose"):
            engine, ws = _Engine(), _WS()
            pipe = _pipe(engine, cls, clause_first_deadline_ms=40)
            await pipe.feed_text("Hi", False, ws, "r1", chunking="clause")
            task = pipe.synthesis_task
            if stop == "cancel":
                await pipe.cancel()
            else:
                pipe.dispose()
            await asyncio.sleep(0.12)
            assert engine.texts == [], stop
            try:
                await task
            except asyncio.CancelledError:
                pass
            assert _other_tasks() == set(), stop

    asyncio.run(run())


@pytest.mark.parametrize("cls", PIPELINES)
def test_a_superseding_request_stops_the_old_timer(cls):
    async def run():
        engine, ws = _Engine(), _WS()
        tts = TTSPipeline(engine=engine, executor=None)
        pipe = cls(tts, clause_first_deadline_ms=40)
        session = TTSSession(websocket=ws, tts_pipeline=tts, tts_stream_pipeline=pipe)
        await session.feed_stream(chunk="Hi", done=False, request_id="r1",
                                  chunking="clause")
        await session.feed_stream(chunk="Hello there", done=False, request_id="r2")
        await asyncio.sleep(0.12)
        assert engine.texts == []  # r1's held word never spoke
        await session.feed_stream(chunk="", done=True, request_id="r2")
        await asyncio.wait_for(session.wait_idle(), 2.0)
        assert engine.texts == ["Hello there"]
        assert _other_tasks() == set()

    asyncio.run(run())


# --- the WS layer passes the settings ---------------------------------------

class _FakeSTT:
    def __init__(self, **kwargs):
        pass

    def feed(self, data: bytes) -> bool:
        return False

    def dispose(self):
        pass


@pytest.fixture
def fresh_settings():
    import voicemanager.config as cfg
    cfg._settings = None
    yield
    cfg._settings = None


def _receive_until_audio_end(ws) -> None:
    for _ in range(50):
        msg = ws.receive()
        if msg.get("text") is not None and json.loads(msg["text"])["type"] == "audio_end":
            return
    raise AssertionError("no audio_end")


def test_the_handler_builds_the_chunker_from_settings(monkeypatch, fresh_settings):
    monkeypatch.setattr("voicemanager.transport.websocket.STTPipeline", _FakeSTT)
    monkeypatch.setenv("VM_TTS_CLAUSE_FIRST_WORDS", "2")
    monkeypatch.setenv("VM_TTS_CLAUSE_FIRST_DEADLINE_MS", "60000")
    engine = _Engine()
    with ThreadPoolExecutor(max_workers=1) as executor:
        app = create_app()
        app.state.stt_engine_type = "whisper"
        app.state.session_registry = SessionRegistry()
        app.state.tts_engine_type = "openvoice"
        app.state.tts_engine = engine
        app.state.tts_executor = executor
        client = TestClient(app)
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "tts_stream", "request_id": "r1", "text": "one two",
                          "chunking": "clause"})
            ws.send_json({"type": "tts_stream", "request_id": "r1", "text": " three",
                          "done": True})
            _receive_until_audio_end(ws)
    # Two first words sends "one two" alone; the default of 4 would hold all three.
    assert engine.texts == ["one two", "three"]


def test_the_handler_passes_the_mark_minimum(monkeypatch, fresh_settings):
    monkeypatch.setattr("voicemanager.transport.websocket.STTPipeline", _FakeSTT)
    monkeypatch.setenv("VM_TTS_CLAUSE_MARK_MIN_WORDS", "1")
    monkeypatch.setenv("VM_TTS_CLAUSE_FIRST_DEADLINE_MS", "60000")
    engine = _Engine()
    with ThreadPoolExecutor(max_workers=1) as executor:
        app = create_app()
        app.state.stt_engine_type = "whisper"
        app.state.session_registry = SessionRegistry()
        app.state.tts_engine_type = "openvoice"
        app.state.tts_engine = engine
        app.state.tts_executor = executor
        client = TestClient(app)
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_json({"type": "tts_stream", "request_id": "r1", "text": "Yes, I",
                          "chunking": "clause"})
            ws.send_json({"type": "tts_stream", "request_id": "r1", "text": "",
                          "done": True})
            _receive_until_audio_end(ws)
    # A minimum of 1 cuts at "Yes,"; the default of 3 would send "Yes, I" whole.
    assert engine.texts == ["Yes,", "I"]
