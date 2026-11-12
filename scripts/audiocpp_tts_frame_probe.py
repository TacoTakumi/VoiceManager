"""Frame times of tts_stream requests on the real audiocpp TTS engine.

MANUAL live script - never collected by pytest (scripts/ is outside tests/).

It answers one question: when a reply is streamed, when does each audio frame
leave the server, and does a client that plays on arrival run dry? It builds the
real stream pipeline in THIS process, with its own TTS sidecar, and takes the
frames with a websocket stand-in. It does not connect to a running VoiceManager
server and sends it nothing, so a live service is not disturbed. It does need
GPU memory for one more TTS sidecar on the audio.cpp device (about 5.2 GiB for
Breeze q8_0). It never falls back to the CPU: a sidecar that cannot load is an
error.

The settings are the server's own: the VM_* variables in the shell, then the
.env file of the directory it is started from. Run it from the repository root
to measure what the service there would do. Set VM_AUDIOCPP_DEVICE in the shell
to use another card.

For each reply it prints the pipeline's own log lines (the first audio and the
seam of each engine request, the silence statistics, and the frame line), and at
the end one row per reply:

    first_ms    from the first text to the first frame
    reserve_ms  the lowest reserve of a client that starts playback when the
                first frame arrives; negative means that it ran dry
    at_ms       when that was, from the first frame
    wait_ms     the longest time between two frames, and when it ended

Sessions:
  warm   one sidecar session for all replies, with one reply spoken first and
         not counted, as in a call that is already going.
  cold   a new sidecar session for every reply, so each one is the first reply
         of its session.

Text is fed as an LLM gives it: one word each --feed-ms. With --feed-ms 0 the
whole text is fed at once.

Usage (from the repository root):
    python scripts/audiocpp_tts_frame_probe.py --chunking clause --session cold
    python scripts/audiocpp_tts_frame_probe.py --chunking sentence \
        --voice designed_gm_02 --runs 5
    python scripts/audiocpp_tts_frame_probe.py --text "Yes. Go ahead." --silence-cap-ms 0
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import time

TEXTS = [
    "Yes, I can hear you. The voice link is nominal. Go ahead.",
    "Understood. It sounds like the speech recognition is doing well today. I am "
    "glad that the demonstration is running clean, and I am ready for the next item.",
    "That is a well known question. The answer depends on which bird you mean, and "
    "on the wind that day. A small bird cruises at about eleven meters each second. "
    "A larger one is faster, but it tires sooner, so over a long distance the two "
    "arrive at almost the same time.",
]
WARM_UP_TEXT = "This is a warm up sentence. It is not counted."

_PIPELINE_LOGGER = "voicemanager.pipeline.audiocpp_tts"
_FRAME_LINE = re.compile(
    r"TTS frames: request_id=\S+ chunking=\S+ frames=(?P<frames>\d+) "
    r"audio_ms=(?P<audio>\d+) span_ms=(?P<span>\d+) "
    r"min_reserve_ms=(?P<reserve>-?\d+|none) at_ms=(?P<at>\d+|none) "
    r"sent=\[(?P<sent>[^\]]*)\]"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frame times of tts_stream requests on the real audiocpp TTS "
                    "engine, on a private sidecar. Needs a GPU; does not use a "
                    "running server.")
    parser.add_argument("--chunking", choices=["sentence", "clause"], default="clause",
                        help="the chunking mode of each request (default: clause)")
    parser.add_argument("--voice", default=None,
                        help="the voice preset (default: VM_AUDIOCPP_TTS_VOICE)")
    parser.add_argument("--text", action="append", default=None, metavar="TEXT",
                        help="a reply text; repeat for more (default: three built-in "
                             "replies of different lengths)")
    parser.add_argument("--feed-ms", type=int, default=30,
                        help="feed one word each this many ms; 0 feeds the whole "
                             "text at once (default: 30)")
    parser.add_argument("--session", choices=["warm", "cold"], default="warm",
                        help="warm: one session, one uncounted reply first; cold: a "
                             "new sidecar session for every reply (default: warm)")
    parser.add_argument("--silence-cap-ms", type=int, default=None,
                        help="the cap for a silent run in a clause request; 0 is off "
                             "(default: VM_AUDIOCPP_TTS_CLAUSE_SILENCE_MS)")
    parser.add_argument("--runs", type=int, default=3,
                        help="how many times each text is spoken (default: 3)")
    parser.add_argument("--pause-s", type=float, default=0.5,
                        help="pause between two replies of a warm session (default: 0.5)")
    return parser.parse_args(argv)


class _Frames:
    """Stands in for the websocket: takes what the pipeline sends."""

    def __init__(self) -> None:
        self.first_frame_at: float | None = None
        self.frames = 0

    async def send_text(self, data: str) -> None:
        del data

    async def send_bytes(self, data: bytes) -> None:
        if self.first_frame_at is None and data:
            self.first_frame_at = time.monotonic()
        self.frames += 1


class _Lines(logging.Handler):
    """Prints the pipeline's log lines and keeps the frame line of each request."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.frame_line: str | None = None

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if record.levelno < logging.INFO:
            return
        print(f"  {time.strftime('%H:%M:%S')} {message}", flush=True)
        if message.startswith("TTS frames:"):
            self.frame_line = message


def longest_wait(sent: str) -> tuple[int, int]:
    """(the longest time between two frames in ms, the time at its end)."""
    times = [int(entry.split(":")[0]) for entry in sent.split()
             if ":" in entry and entry[0].isdigit()]
    best, at = 0, 0
    for before, after in zip(times, times[1:]):
        if after - before > best:
            best, at = after - before, after
    return best, at


async def speak(pipe, text: str, request_id: str, args, lines: _Lines) -> dict:
    """Feed one reply and wait for its end; returns the row for the summary."""
    lines.frame_line = None
    ws = _Frames()
    started = time.monotonic()
    if args.feed_ms <= 0:
        await pipe.feed_text(text, True, ws, request_id, voice=args.voice,
                             chunking=args.chunking)
    else:
        for word in re.findall(r"\S+\s*", text):
            await pipe.feed_text(word, False, ws, request_id, voice=args.voice,
                                 chunking=args.chunking)
            await asyncio.sleep(args.feed_ms / 1000.0)
        await pipe.feed_text("", True, ws, request_id, voice=args.voice,
                             chunking=args.chunking)
    await pipe.synthesis_task
    row = {"chars": len(text), "first_ms": None, "frames": ws.frames}
    if ws.first_frame_at is not None:
        row["first_ms"] = round((ws.first_frame_at - started) * 1000)
    match = _FRAME_LINE.fullmatch(lines.frame_line or "")
    if match is None:
        raise RuntimeError(f"no frame line for {request_id}: {lines.frame_line!r}")
    row.update(match.groupdict())
    row["wait_ms"], row["wait_at"] = longest_wait(match["sent"])
    return row


async def open_session(base_engine, settings, args):
    """A new sidecar session and its stream pipeline, as the server makes them
    for one websocket connection. Returns (engine, pipeline)."""
    from voicemanager.pipeline.audiocpp_tts import AudioCppTTSStreamPipeline
    from voicemanager.pipeline.tts import TTSPipeline

    engine = base_engine.session_engine()
    # Spawn and load now, as the server's warm-up does at the connect. A load
    # that fails (no memory on the card) raises here and ends the probe.
    await asyncio.to_thread(engine.connect)
    tts_pipeline = TTSPipeline(
        engine=engine, executor=None, language=settings.tts_language,
        speaker_key=settings.tts_speaker_key, speed=settings.tts_speed,
        voice=args.voice)
    options = {}
    if args.silence_cap_ms is not None:
        options["clause_silence_ms"] = args.silence_cap_ms
    pipe = AudioCppTTSStreamPipeline(
        tts_pipeline=tts_pipeline,
        language=settings.tts_language,
        clause_first_words=settings.tts_clause_first_words,
        clause_first_deadline_ms=settings.tts_clause_first_deadline_ms,
        clause_later_words=settings.tts_clause_later_words,
        clause_margin_ms=settings.tts_clause_margin_ms,
        clause_mark_min_words=settings.tts_clause_mark_min_words,
        clause_word_grace_ms=settings.tts_clause_word_grace_ms,
        coalesce_ms=settings.audiocpp_tts_coalesce_ms,
        **options,
    )
    return engine, pipe


async def close_session(engine) -> None:
    """End the sidecar and wait until its process is gone, so its GPU memory is
    free before the next session loads. The server's own teardown does not wait;
    the probe must, because two sidecars may not fit on the card."""
    client = engine._client
    await asyncio.to_thread(client.close)
    reaper = client._reaper
    if reaper is not None:
        await asyncio.to_thread(reaper.join, 12.0)


async def run(args: argparse.Namespace) -> int:
    from voicemanager.config import get_settings
    from voicemanager.models import _load_audiocpp_tts

    settings = get_settings()
    if settings.audiocpp_backend == "cpu":
        print("FAIL the audio.cpp backend is cpu; this probe measures the GPU only")
        return 2
    lines = _Lines()
    pipeline_logger = logging.getLogger(_PIPELINE_LOGGER)
    pipeline_logger.setLevel(logging.INFO)
    pipeline_logger.addHandler(lines)
    pipeline_logger.propagate = False

    texts = args.text or TEXTS
    print(f"model={settings.audiocpp_tts_model}")
    print(f"backend={settings.audiocpp_backend} device={settings.audiocpp_device} "
          f"voice={args.voice or settings.audiocpp_tts_voice} chunking={args.chunking} "
          f"session={args.session} feed_ms={args.feed_ms} runs={args.runs} "
          f"silence_cap_ms="
          f"{settings.audiocpp_tts_clause_silence_ms if args.silence_cap_ms is None else args.silence_cap_ms} "
          f"coalesce_ms={settings.audiocpp_tts_coalesce_ms} "
          f"start_buffer_ms={settings.audiocpp_tts_start_buffer_ms}")

    base = _load_audiocpp_tts(settings)
    rows: list[tuple[str, dict]] = []
    engine = None
    try:
        if args.session == "warm":
            engine, pipe = await open_session(base["tts_engine"], settings, args)
            print("warm-up reply (not counted)")
            await speak(pipe, WARM_UP_TEXT, "warm-up", args, lines)
            await asyncio.sleep(args.pause_s)
        for run_index in range(1, args.runs + 1):
            for text_index, text in enumerate(texts, start=1):
                label = f"run{run_index}-text{text_index}"
                if args.session == "cold":
                    engine, pipe = await open_session(base["tts_engine"], settings, args)
                print(f"{label} ({len(text)} chars)")
                rows.append((label, await speak(pipe, text, label, args, lines)))
                if args.session == "cold":
                    await close_session(engine)
                    engine = None
                else:
                    await asyncio.sleep(args.pause_s)
    finally:
        if engine is not None:
            await close_session(engine)
        base["tts_executor"].shutdown(wait=False)

    print()
    print(f"{'reply':<12} {'chars':>5} {'frames':>6} {'audio_ms':>8} {'span_ms':>7} "
          f"{'first_ms':>8} {'reserve_ms':>10} {'at_ms':>6} {'wait_ms':>7} {'wait_at':>7}")
    dry = 0
    for label, row in rows:
        if row["reserve"] != "none" and int(row["reserve"]) < 0:
            dry += 1
        print(f"{label:<12} {row['chars']:>5} {row['frames']:>6} {row['audio']:>8} "
              f"{row['span']:>7} {str(row['first_ms']):>8} {row['reserve']:>10} "
              f"{row['at']:>6} {row['wait_ms']:>7} {row['wait_at']:>7}")
    print(f"=== {len(rows)} replies, {dry} with a negative reserve ===")
    return 0


def main() -> int:
    args = parse_args()
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
