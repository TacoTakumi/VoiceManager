"""The audiocpp stream pipeline logs when it sent each frame of a request.

A client's playback queue can run dry in the middle of a reply. The server could
not show its side of that: it logged the first audio of an engine request and the
reserve at a seam between two engine requests, and nothing for the frames between.
The stream pipeline now writes one line at the end of each tts_stream request:

    TTS frames: request_id=<id> chunking=<mode> frames=<n> audio_ms=<n>
        span_ms=<n> min_reserve_ms=<n> at_ms=<n> sent=[<t>:<len> ... | <t>:<len> ...]

(one line in the log). Each entry of the list is one binary frame: the time of its
send in ms from the first frame, and its audio length in ms. A `|` stands before
the first frame of each later engine request.

`min_reserve_ms` is the lowest reserve of the simplest client: playback starts
when the first frame arrives and does not stop. The reserve at a send is the audio
sent before it minus the time since the first send, so a negative value means that
such a client ran dry by that amount. `at_ms` is the time of that send.

What must hold:

  * a clause request and a sentence request each log exactly one line;
  * the times and lengths are those of the frames as sent: after the silence cap,
    with the audio that the start buffer held as the one frame it is sent as, and
    with the rest that the silence limiter sends at the end of a clause request.

Model-free and server-free: a fake engine gives the audio and sets a fake clock,
and a fake WebSocket takes the frames.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

import numpy as np

from voicemanager.pipeline.audiocpp_tts import AudioCppTTSStreamPipeline
from voicemanager.pipeline.tts import TTSPipeline

SR = 24000
_NO_TIMER = {"clause_first_words": 2, "clause_first_deadline_ms": 60_000}
_LOGGER = "voicemanager.pipeline.audiocpp_tts"
_LINE = re.compile(
    r"TTS frames: request_id=(?P<request_id>\S+) chunking=(?P<chunking>\S+) "
    r"frames=(?P<frames>\d+) audio_ms=(?P<audio>\d+) span_ms=(?P<span>\d+) "
    r"min_reserve_ms=(?P<reserve>-?\d+|none) at_ms=(?P<at>\d+|none) "
    r"sent=\[(?P<sent>[^\]]*)\]"
)


class _Clock:
    """A clock that only the test moves."""

    def __init__(self) -> None:
        self.start = 500.0
        self.now = self.start

    def __call__(self) -> float:
        return self.now

    def set_ms(self, ms: int) -> None:
        self.now = self.start + ms / 1000.0


def _speech(ms: int) -> np.ndarray:
    return np.full(SR * ms // 1000, 0.1, dtype=np.float32)


def _silence(ms: int) -> np.ndarray:
    return np.zeros(SR * ms // 1000, dtype=np.float32)


class _Engine:
    """Gives scripted chunks for each engine request.

    `script` has one list for each engine request. Each entry is (time in ms,
    audio): the engine sets the clock to that time and then gives the audio, so
    the pipeline sends the frame at exactly that time. The first engine request
    ends when `gate` is set, after its chunks.
    """

    is_batch = False
    sample_rate = SR

    def __init__(self, clock: _Clock, script: list[list[tuple[int, np.ndarray]]]):
        self.clock = clock
        self.script = script
        self.gate = asyncio.Event()
        self.texts: list[str] = []

    async def ensure_ready(self) -> None:
        return None

    async def stream_synthesize(self, text: str, **kwargs):
        index = len(self.texts)
        self.texts.append(text)
        for at_ms, audio in self.script[index]:
            self.clock.set_ms(at_ms)
            yield audio
        if index == 0:
            await self.gate.wait()


class _WS:
    def __init__(self) -> None:
        self.frames: list[int] = []  # samples in each binary frame
        self.texts: list[dict] = []

    async def send_text(self, data: str) -> None:
        self.texts.append(json.loads(data))

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append(len(data) // 2)


def _pipe(engine: _Engine, **kwargs) -> AudioCppTTSStreamPipeline:
    options = {"clause_continue": False, "clause_end_mark": "", "clause_silence_ms": 0,
               "start_buffer_ms": 0, "frame_clock": engine.clock, **_NO_TIMER, **kwargs}
    return AudioCppTTSStreamPipeline(TTSPipeline(engine=engine, executor=None), **options)


async def _until(cond, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while not cond():
        assert loop.time() < end, "condition not met in time"
        await asyncio.sleep(0.002)


def _frame_lines(caplog) -> list[dict]:
    found = []
    for record in caplog.records:
        message = record.getMessage()
        if not message.startswith("TTS frames:"):
            continue
        match = _LINE.fullmatch(message)
        assert match, f"unexpected form of the frame line: {message}"
        assert record.levelno == logging.INFO
        assert record.name == _LOGGER
        found.append(match.groupdict())
    return found


def _lengths(sent: str) -> list[int]:
    """The audio lengths in a `sent` list, in order."""
    return [int(entry.split(":")[1]) for entry in sent.split() if entry != "|"]


def _one_request(caplog, script, chunking: str = "sentence", **kwargs):
    """One request with one engine request; returns (the frame line, the websocket)."""
    clock = _Clock()
    engine = _Engine(clock, [script])
    ws = _WS()

    async def run() -> None:
        pipe = _pipe(engine, **kwargs)
        engine.gate.set()
        await pipe.feed_text("One two three.", True, ws, "r1", chunking=chunking)
        await asyncio.wait_for(pipe.synthesis_task, 5.0)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        asyncio.run(run())
    lines = _frame_lines(caplog)
    assert len(lines) == 1
    return lines[0], ws


# --- the line ---------------------------------------------------------------------------

def test_a_clause_request_logs_one_line_with_each_frame_and_a_mark_at_the_seam(caplog):
    clock = _Clock()
    engine = _Engine(clock, [
        [(0, _speech(200)), (100, _speech(200)), (200, _speech(200))],
        [(500, _speech(200)), (600, _speech(200))],
    ])
    ws = _WS()

    async def run() -> None:
        pipe = _pipe(engine)
        await pipe.feed_text("One two three. ", False, ws, "r1", chunking="clause",
                             flush=True)
        await _until(lambda: len(ws.frames) == 3)
        await pipe.feed_text("Four five six.", True, ws, "r1", chunking="clause")
        engine.gate.set()
        await asyncio.wait_for(pipe.synthesis_task, 5.0)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        asyncio.run(run())
    assert len(engine.texts) == 2
    lines = _frame_lines(caplog)
    assert len(lines) == 1
    line = lines[0]
    assert line["request_id"] == "r1"
    assert line["chunking"] == "clause"
    assert line["frames"] == "5"
    assert line["audio"] == "1000"
    assert line["span"] == "600"
    assert line["sent"] == "0:200 100:200 200:200 | 500:200 600:200"
    # Reserve at each send after the first: 100, 200, 100, 200. The first lowest counts.
    assert (line["reserve"], line["at"]) == ("100", "100")


def test_a_sentence_request_logs_the_same_line(caplog):
    line, ws = _one_request(
        caplog, [(0, _speech(200)), (150, _speech(200)), (300, _speech(200))])
    assert line["request_id"] == "r1"
    assert line["chunking"] == "sentence"
    assert line["frames"] == "3"
    assert line["audio"] == "600"
    assert line["span"] == "300"
    assert line["sent"] == "0:200 150:200 300:200"
    assert (line["reserve"], line["at"]) == ("50", "150")
    assert len(ws.frames) == 3


# --- the lowest reserve -----------------------------------------------------------------

def test_the_lowest_reserve_is_the_audio_sent_before_less_the_time_passed(caplog):
    """500 ms sent at 0, 200 and 800 ms: 300 ms left at 200 ms, 200 ms left at 800 ms."""
    line, _ = _one_request(
        caplog, [(0, _speech(500)), (200, _speech(500)), (800, _speech(500))])
    assert (line["reserve"], line["at"]) == ("200", "800")


def test_a_frame_that_comes_after_the_audio_played_out_gives_a_negative_reserve(caplog):
    """500 ms sent at 0 and the next frame at 900 ms: the client was dry for 400 ms."""
    line, _ = _one_request(caplog, [(0, _speech(500)), (900, _speech(500))])
    assert (line["reserve"], line["at"]) == ("-400", "900")


def test_the_lowest_reserve_reads_the_same_in_a_clause_request(caplog):
    line, _ = _one_request(
        caplog, [(0, _speech(500)), (900, _speech(500))], chunking="clause")
    assert line["chunking"] == "clause"
    assert (line["reserve"], line["at"]) == ("-400", "900")


# --- the frames as sent -----------------------------------------------------------------

def test_the_line_gives_the_frames_as_the_client_got_them(caplog):
    """A clause request with the silence cap, a start buffer and an engine that
    makes a silent run longer than the cap. The first chunks wait in the start
    buffer and go as one frame, the cap drops part of the silent run, and the
    limiter sends its rest (the 5 ms after the last whole window) at the end."""
    script = [
        (0, _speech(100)),
        (40, _speech(100)),
        (80, _speech(100)),     # 300 ms held: the start buffer of 250 ms is full
        (200, _silence(300)),   # 100 ms pass the cap, 200 ms are dropped
        (320, _speech(105)),
        (400, _speech(100)),
    ]
    line, ws = _one_request(caplog, script, chunking="clause",
                            clause_silence_ms=100, start_buffer_ms=250)
    sent_ms = [round(n * 1000 / SR) for n in ws.frames]
    assert _lengths(line["sent"]) == sent_ms
    assert int(line["frames"]) == len(ws.frames)
    assert int(line["audio"]) == sum(sent_ms)
    assert sent_ms[0] == 300                      # the held block is one frame
    assert sum(sent_ms) == 100 * 3 + 100 + 105 + 100   # 200 ms of silence dropped
    assert sent_ms[-1] == 5                       # the limiter's rest at the end
    assert "|" not in line["sent"]
