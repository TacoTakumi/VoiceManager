"""The seam between two engine requests of a clause request keeps the client's reserve.

The audiocpp engine does one request at a time, so the audio of the second engine
request of a reply comes after the first has been generated. Two things in the
stream pipeline delayed it further without need: the start buffer, which held the
first audio of every engine request, and the coalescing window, which waited for
more text. In a clause request both now look at the audio the client has not
played yet.

What must hold for the start buffer:

  * with at least the buffer's worth of audio unplayed, the first audio of an
    engine request is sent at once;
  * with less unplayed, audio is held until the held audio plus the unplayed
    audio reaches the buffer;
  * with nothing unplayed, the full buffer is held, as before;
  * a sentence request holds the full buffer for every engine request, as before.

What must hold for the coalescing window:

  * with 500 ms or less unplayed, the next engine request starts at once;
  * with much more unplayed, the full window is used and text that arrives in it
    joins the engine request;
  * the wait ends when the unplayed audio falls to 500 ms;
  * a sentence request uses the full window, as before.

Each engine request after the first logs the reserve at its seam.

Model-free and server-free: a fake engine gives the audio and a fake WebSocket
takes the frames. The tests run on a virtual clock: an event loop whose time
moves to the next timer when nothing is ready, and the pipeline's estimate of
the unplayed audio reads the same clock. So a wait of 300 ms takes no real
time, a busy machine cannot change a result, and the times are exact.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

import numpy as np
import pytest

from voicemanager.pipeline.audiocpp_tts import AudioCppTTSStreamPipeline
from voicemanager.pipeline.tts import TTSPipeline

SR = 24000
CHUNK = SR // 20  # 50 ms
BUFFER_MS = 250
_NO_TIMER = {"clause_first_words": 2, "clause_first_deadline_ms": 60_000}
_SENTENCE = "The quick brown fox jumps over the lazy dog near the quiet riverbank today."


def ms(samples: int) -> int:
    return samples * 1000 // SR


class _VirtualLoop(asyncio.SelectorEventLoop):
    """An event loop on a virtual clock.

    When no callback is ready, the loop does not sleep until the next timer: its
    time moves there at once. Nothing in these tests waits on a socket or a
    thread, so a loop with nothing ready and no timer is a test that hangs, and
    that is raised as an error.
    """

    def __init__(self) -> None:
        super().__init__()
        self._now = 0.0
        poll = self._selector.select

        def select(timeout=None):
            if timeout is None:
                raise RuntimeError("virtual clock: nothing is ready and no timer is set")
            if timeout > 0:
                self._now += timeout
            return poll(0)

        self._selector.select = select

    def time(self) -> float:
        return self._now


def _run(coro):
    """Run the coroutine to its end on a virtual clock."""
    with asyncio.Runner(loop_factory=_VirtualLoop) as runner:
        return runner.run(coro)


def test_the_virtual_clock_gives_exact_times_and_takes_no_real_time():
    import time

    async def run() -> list[float]:
        loop = asyncio.get_running_loop()
        seen = [loop.time()]
        await asyncio.sleep(30.0)
        seen.append(loop.time())
        try:
            await asyncio.wait_for(asyncio.Event().wait(), timeout=12.5)
        except asyncio.TimeoutError:
            seen.append(loop.time())
        return seen

    started = time.monotonic()
    assert _run(run()) == [0.0, 30.0, 42.5]
    assert time.monotonic() - started < 5.0


class _Engine:
    """Gives a set number of 50 ms chunks for each engine request.

    The first request ends when `gate` is set, after its chunks. `delays` holds
    the seconds to wait before the first chunk of each request, and `gaps` the
    seconds to wait before every chunk of each request (0.05 is an engine at
    real-time speed).
    """

    is_batch = False
    sample_rate = SR

    def __init__(self, chunks: list[int], delays: list[float] | None = None,
                 gaps: list[float] | None = None):
        self.chunks = chunks
        self.delays = delays or [0.0] * len(chunks)
        self.gaps = gaps or [0.0] * len(chunks)
        self.gate = asyncio.Event()
        self.texts: list[str] = []
        self.started_at: list[float] = []
        self.ended_at: list[float] = []
        self.ws: _WS | None = None
        self.frames_before: list[int] = []  # frames sent when each request started

    async def ensure_ready(self) -> None:
        return None

    async def stream_synthesize(self, text: str, **kwargs):
        loop = asyncio.get_running_loop()
        index = len(self.texts)
        self.texts.append(text)
        self.started_at.append(loop.time())
        self.frames_before.append(len(self.ws.frames))
        if self.delays[index]:
            await asyncio.sleep(self.delays[index])
        for _ in range(self.chunks[index]):
            if self.gaps[index]:
                await asyncio.sleep(self.gaps[index])
            # Not silence: the pipeline caps a silent run in a clause request.
            yield np.full(CHUNK, 0.1, dtype=np.float32)
        if index == 0:
            await self.gate.wait()
        self.ended_at.append(loop.time())


class _WS:
    def __init__(self) -> None:
        self.frames: list[int] = []  # samples in each binary frame

    async def send_text(self, data: str) -> None:
        json.loads(data)

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append(len(data) // 2)


def _pipe(engine: _Engine, **kwargs) -> AudioCppTTSStreamPipeline:
    """A pipeline whose unplayed-audio estimate reads the running loop's clock."""
    options = {"clause_continue": False, "clause_end_mark": "", "clause_silence_ms": 0,
               "start_buffer_ms": BUFFER_MS,
               "clause_clock": asyncio.get_running_loop().time, **_NO_TIMER, **kwargs}
    return AudioCppTTSStreamPipeline(TTSPipeline(engine=engine, executor=None), **options)


async def _until(cond, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not cond():
        assert loop.time() < end, "condition not met in time"
        await asyncio.sleep(0.002)


def _two_requests(engine: _Engine, chunking: str = "clause", **kwargs) -> list[int]:
    """One request with two engine requests; returns the frames of the second."""
    ws = _WS()
    engine.ws = ws

    async def run() -> None:
        pipe = _pipe(engine, **kwargs)
        if chunking == "clause":
            await pipe.feed_text("One two three. ", False, ws, "r1", chunking=chunking,
                                 flush=True)
            second = "Four five six."
        else:
            # A sentence request sends a sentence when the next one has begun.
            await pipe.feed_text(_SENTENCE + " Meanwhile the careful gray owl watches",
                                 False, ws, "r1", chunking=chunking)
            second = " quietly from the tallest branch of the old oak tree."
        await _until(lambda: len(engine.texts) == 1)
        await pipe.feed_text(second, True, ws, "r1", chunking=chunking)
        # Let the first engine request give all its chunks before it ends.
        await asyncio.sleep(0.02)
        engine.gate.set()
        await asyncio.wait_for(pipe.synthesis_task, 5.0)

    _run(run())
    assert len(engine.texts) == 2
    return ws.frames[engine.frames_before[1]:]


# --- the start buffer -------------------------------------------------------------------

def test_with_a_full_reserve_the_first_audio_of_the_next_request_goes_at_once():
    """The first engine request left 1.0 s unplayed: nothing is held."""
    frames = _two_requests(_Engine([20, 8]))
    assert [ms(n) for n in frames] == [50] * 8


def test_with_a_small_reserve_only_the_missing_part_is_held():
    """150 ms unplayed and a buffer of 250 ms: the audio goes when 100 ms are held."""
    frames = _two_requests(_Engine([3, 8]))
    assert [ms(n) for n in frames] == [100] + [50] * 6


def test_a_reserve_a_little_below_the_missing_part_holds_one_chunk_more():
    """The second engine request gives its first audio 1 ms later: 149 ms unplayed,
    so 100 ms held are not enough and the audio goes when 150 ms are held."""
    frames = _two_requests(_Engine([3, 8], delays=[0.0, 0.001]))
    assert [ms(n) for n in frames] == [150] + [50] * 5


def test_a_reserve_that_falls_while_audio_is_held_is_read_at_every_chunk():
    """150 ms unplayed, and the second engine request gives 50 ms of audio each
    50 ms, as an engine at real-time speed does. The reserve falls as fast as the
    held audio grows: 100 + 50, then 50 + 100, then 0 + 150, 0 + 200. The audio
    goes only when the full buffer is held. An estimate read once, at the first
    chunk (100 ms), would send the audio before the buffer is full."""
    frames = _two_requests(_Engine([3, 8], gaps=[0.0, 0.05]))
    assert [ms(n) for n in frames] == [250] + [50] * 3


def test_with_no_reserve_the_full_buffer_is_held():
    """The second engine request gives its first audio after the 150 ms of the
    first have played out: the client is dry, so the full buffer is held."""
    frames = _two_requests(_Engine([3, 8], delays=[0.0, 0.3]))
    assert [ms(n) for n in frames] == [250] + [50] * 3


def test_the_first_engine_request_holds_the_full_buffer():
    engine = _Engine([20, 8])
    _two_requests(engine)
    first = engine.ws.frames[:engine.frames_before[1]]
    assert [ms(n) for n in first] == [250] + [50] * 15


def test_a_sentence_request_holds_the_full_buffer_for_every_engine_request():
    frames = _two_requests(_Engine([20, 8]), chunking="sentence")
    assert [ms(n) for n in frames] == [250] + [50] * 3


# --- the coalescing window --------------------------------------------------------------

WINDOW_MS = 350


def _later_text(engine: _Engine, third_after: float | None, chunking: str = "clause"):
    """One request: a first piece, a second piece queued while the first engine
    request runs, and (with `third_after`) a third piece that many seconds after
    the first engine request ended. Done comes 1 s after that end."""
    ws = _WS()
    engine.ws = ws

    async def run() -> None:
        pipe = _pipe(engine, coalesce_ms=WINDOW_MS)
        if chunking == "clause":
            texts = ["One two three. ", "Four five six. ", "Seven eight nine."]
            flush = True
        else:
            texts = [_SENTENCE + " Meanwhile the careful gray owl watches",
                     " quietly from the tallest branch of the old oak tree. Down in the"
                     " valley the patient badger",
                     " digs a new tunnel beneath the crumbling stone wall. Far above them"]
            flush = False
        await pipe.feed_text(texts[0], False, ws, "r1", chunking=chunking, flush=flush)
        await _until(lambda: len(engine.texts) == 1)
        await pipe.feed_text(texts[1], False, ws, "r1", chunking=chunking, flush=flush)
        await asyncio.sleep(0.02)
        engine.gate.set()
        await _until(lambda: len(engine.ended_at) >= 1)
        if third_after is not None:
            await asyncio.sleep(third_after)
            await pipe.feed_text(texts[2], False, ws, "r1", chunking=chunking, flush=flush)
            await asyncio.sleep(1.0 - third_after)
        else:
            await asyncio.sleep(1.0)
        await pipe.feed_text("", True, ws, "r1", chunking=chunking)
        await asyncio.wait_for(pipe.synthesis_task, 5.0)

    _run(run())
    return engine.started_at[1] - engine.ended_at[0]


def test_with_a_low_reserve_the_next_engine_request_starts_at_once():
    """0.3 s unplayed, below the floor of 0.5 s: no wait for more text."""
    engine = _Engine([6, 4, 4])
    wait = _later_text(engine, third_after=None)
    assert wait == 0.0
    assert engine.texts[1] == "Four five six."


def test_with_a_high_reserve_the_full_window_is_used():
    """3.0 s unplayed: text that comes 150 ms after the first engine request ended
    joins the second one, which starts when the window of 0.35 s has passed."""
    engine = _Engine([60, 4])
    wait = _later_text(engine, third_after=0.15)
    assert engine.texts[1:] == ["Four five six. Seven eight nine."]
    assert wait == pytest.approx(0.35)


def test_the_wait_ends_when_the_reserve_falls_to_the_floor():
    """0.65 s of audio went out 22 ms before the first engine request ends (the
    helper waits 20 ms and polls each 2 ms): the wait is the 0.15 s above the
    floor less those 22 ms, not the window of 0.35 s."""
    engine = _Engine([13, 4])
    wait = _later_text(engine, third_after=None)
    assert wait == pytest.approx(0.128)
    assert engine.texts[1] == "Four five six."


def test_a_sentence_request_uses_the_full_window_with_nothing_unplayed():
    """The first engine request left 0.15 s, played out before the third sentence
    comes: a sentence request still waits the full window and takes it in. The
    words left at done make a third engine request."""
    engine = _Engine([3, 4, 4])
    wait = _later_text(engine, third_after=0.2, chunking="sentence")
    assert len(engine.texts) == 3
    assert "owl" in engine.texts[1] and "badger" in engine.texts[1]
    assert wait == pytest.approx(0.35)


# --- the log line -----------------------------------------------------------------------

_SEAM = re.compile(r"Clause TTS seam: request_id=(\S+) reserve_ms=(\d+) held_ms=(\d+)")
_LOGGER = "voicemanager.pipeline.audiocpp_tts"


def _seam_lines(caplog) -> list[tuple[str, int, int]]:
    found = []
    for record in caplog.records:
        match = _SEAM.fullmatch(record.getMessage())
        if match:
            assert record.levelno == logging.INFO
            found.append((match.group(1), int(match.group(2)), int(match.group(3))))
    return found


def test_a_later_engine_request_logs_the_reserve_at_its_seam(caplog):
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        _two_requests(_Engine([20, 8]))
    lines = _seam_lines(caplog)
    assert len(lines) == 1
    request_id, reserve, held = lines[0]
    assert request_id == "r1"
    # 1.0 s went out 22 ms before the first engine request ended (the helper
    # waits 20 ms and polls each 2 ms).
    assert reserve == 978
    assert held == 0


def test_the_seam_line_gives_the_audio_that_waited(caplog):
    """149 ms unplayed, buffer 250 ms: 100 ms waited when the third chunk came."""
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        _two_requests(_Engine([3, 8], delays=[0.0, 0.001]))
    assert _seam_lines(caplog) == [("r1", 149, 100)]


def test_the_seam_line_of_a_falling_reserve_gives_the_reserve_at_the_send(caplog):
    """The reserve of 150 ms is gone when the full buffer goes out, 250 ms later:
    200 ms waited when the fifth chunk came."""
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        _two_requests(_Engine([3, 8], gaps=[0.0, 0.05]))
    assert _seam_lines(caplog) == [("r1", 0, 200)]


@pytest.mark.parametrize("delay, reserve", [(0.001, 149), (0.3, 0)])
def test_a_later_engine_request_that_ends_before_the_buffer_fills_logs_its_seam(
        caplog, delay, reserve):
    """The second engine request makes 100 ms of audio and ends. With 149 ms or
    with nothing unplayed that does not fill the buffer of 250 ms, so its audio
    goes out in one frame when it ends. That send is the seam: the line gives the
    reserve then and the 100 ms that waited."""
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        frames = _two_requests(_Engine([3, 2], delays=[0.0, delay]))
    assert [ms(n) for n in frames] == [100]
    assert _seam_lines(caplog) == [("r1", reserve, 100)]


def test_a_dry_client_shows_as_a_reserve_of_zero(caplog):
    with caplog.at_level(logging.INFO, logger=_LOGGER):
        _two_requests(_Engine([3, 8], delays=[0.0, 0.3]))
    assert _seam_lines(caplog) == [("r1", 0, 200)]


def test_one_engine_request_and_a_sentence_request_log_no_seam_line(caplog):
    engine = _Engine([8])
    ws = _WS()
    engine.ws = ws

    async def run() -> None:
        pipe = _pipe(engine)
        engine.gate.set()
        await pipe.feed_text("One two three.", True, ws, "r1", chunking="clause")
        await asyncio.wait_for(pipe.synthesis_task, 5.0)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        _run(run())
        _two_requests(_Engine([20, 8]), chunking="sentence")
    assert len(engine.texts) == 1
    assert _seam_lines(caplog) == []


# -- the client guide ---------------------------------------------------------

def test_the_guide_says_that_a_low_reserve_skips_the_coalescing_wait():
    """With nothing unplayed only the start buffer works as in a sentence
    request. The coalescing wait does not: a clause request does not wait at
    all, and a sentence request waits the full window. The guide must say so."""
    guide = (Path(__file__).resolve().parents[1] / "docs"
             / "client-integration.md").read_text(encoding="utf-8")
    heading = "**Reserve at a seam (audiocpp).**"
    paragraphs = [p for p in guide.split("\n\n") if p.startswith(heading)]
    assert len(paragraphs) == 1
    text = paragraphs[0]
    assert "both work as in a sentence-mode request" not in text
    assert ("With nothing unplayed the start buffer works as in a "
            "sentence-mode request.") in text
    assert ("with 500 ms or less unplayed a clause-mode request does not wait, "
            "where a sentence-mode request waits the full window") in text
