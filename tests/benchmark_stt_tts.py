"""Benchmark and sanity-check the currently active STT and TTS engines.

Run against an already-running VoiceManager server. The script:

  1. Connects with VoiceManagerClient, reads the active engines from /info.
  2. Shows an interactive menu of clips in tests/benchmark_clips/.
     - empty dir => jumps straight to recording a new clip.
     - [1..N] = pick one, [a] = all, [r] = record new.
  3. Records new clips with a 3-2-1 countdown + live elapsed timer
     (8 seconds, 16 kHz mono int16, saved as YYYY-MM-DD_HH-MM-SS.wav).
  4. Warms up each (engine, method) pair, then runs 3 timed iterations.
  5. Prints stdout tables and appends a dated section to
     tests/benchmark_results.md.
  6. Exits non-zero if any TTS sanity check failed.

Usage::

    python tests/benchmark_stt_tts.py
"""
import asyncio
import datetime as dt
import os
import statistics
import sys
import time
import wave
from pathlib import Path
from typing import AsyncIterator

import sounddevice as sd
from dotenv import load_dotenv

from voicemanager.client import VoiceManagerClient

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECORD_SECONDS = 8
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

ITERATIONS = 3
STT_CHUNK_MS = 100  # paces stream_stt_live() at ~real-time

CLIPS_DIR = Path("tests/benchmark_clips")
RESULTS_PATH = Path("tests/benchmark_results.md")

# Reused from tests/test_stream_tts.py so TTS results stay comparable
TTS_TEXT = (
    "The stars are bright tonight. "
    "I can see the constellations clearly from here. "
    "It reminds me of when we used to go camping."
)
TTS_CHUNK_SIZE = 8  # chars per simulated LLM token


# ---------------------------------------------------------------------------
# WAV helpers
# ---------------------------------------------------------------------------

def read_wav_duration(path: Path) -> float:
    """Return audio duration in seconds for a WAV file."""
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


async def wav_to_chunks(path: Path, chunk_ms: int = STT_CHUNK_MS) -> AsyncIterator[bytes]:
    """Yield PCM frame chunks from a WAV at ~real-time pacing.

    Sleeps for chunk_ms between yields so streaming STT engines see audio
    arrive at roughly the rate a microphone would deliver it. TTFP would
    be meaningless otherwise.
    """
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        if rate != SAMPLE_RATE or wf.getnchannels() != CHANNELS or wf.getsampwidth() != 2:
            raise ValueError(
                f"Expected {SAMPLE_RATE} Hz mono int16 WAV, got "
                f"{rate} Hz / {wf.getnchannels()} ch / {wf.getsampwidth() * 8}-bit"
            )
        chunk_frames = int(rate * chunk_ms / 1000)
        while True:
            frames = wf.readframes(chunk_frames)
            if not frames:
                return
            yield frames
            await asyncio.sleep(chunk_ms / 1000)


# ---------------------------------------------------------------------------
# Microphone recording
# ---------------------------------------------------------------------------

def record_clip() -> Path:
    """Record RECORD_SECONDS of audio with a 3-2-1 countdown + live timer.

    Saves to tests/benchmark_clips/YYYY-MM-DD_HH-MM-SS.wav and returns the path.
    """
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nGet ready to speak — recording will be {RECORD_SECONDS} seconds.")
    for n in (3, 2, 1):
        print(f"  {n}...", flush=True)
        time.sleep(1)
    print("  RECORDING\n", flush=True)

    total_frames = RECORD_SECONDS * SAMPLE_RATE
    recording = sd.rec(
        total_frames,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocking=False,
    )

    # Live elapsed-seconds ticker (carriage return — no scroll)
    for elapsed in range(1, RECORD_SECONDS + 1):
        time.sleep(1)
        print(f"\r  {elapsed}s / {RECORD_SECONDS}s", end="", flush=True)
    sd.wait()
    print("\n  done.\n")

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = CLIPS_DIR / f"{stamp}.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(recording.tobytes())
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Clip selection menu
# ---------------------------------------------------------------------------

def select_clips() -> list[Path]:
    """Interactive menu. Returns the list of clip paths to benchmark.

    Empty dir -> skip menu and record one new clip.
    Otherwise show: [1..N] pick one, [a] all, [r] record new.
    """
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(CLIPS_DIR.glob("*.wav"))

    if not existing:
        print("No clips in tests/benchmark_clips/ yet — recording one now.")
        return [record_clip()]

    print("\nAvailable clips:")
    for i, p in enumerate(existing, 1):
        dur = read_wav_duration(p)
        print(f"  [{i}] {p.name}  ({dur:.1f}s)")
    print("  [a] use all")
    print("  [r] record a new clip")

    while True:
        choice = input("Choice: ").strip().lower()
        if choice == "a":
            return existing
        if choice == "r":
            return [record_clip()]
        if choice.isdigit() and 1 <= int(choice) <= len(existing):
            return [existing[int(choice) - 1]]
        print("  Invalid choice — try again.")


# ---------------------------------------------------------------------------
# STT benchmarks
# ---------------------------------------------------------------------------

async def bench_transcribe_file(client: VoiceManagerClient, wav: Path) -> dict:
    """ITERATIONS timed runs of client.transcribe_file(wav).

    Returns: {e2e: [..], rtf: [..], ttfp: [None, ...], text: <last>, audio_dur: float}
    """
    audio_dur = read_wav_duration(wav)
    e2e_list: list[float] = []
    rtf_list: list[float] = []
    last_text = ""

    for _ in range(ITERATIONS):
        start = time.perf_counter()
        results = await client.transcribe_file(str(wav))
        e2e = time.perf_counter() - start
        e2e_list.append(e2e)
        rtf_list.append(e2e / audio_dur if audio_dur > 0 else 0.0)
        last_text = " ".join(
            r.get("text", "") for r in results if r.get("is_final") is True
        ).strip()

    return {
        "e2e": e2e_list,
        "rtf": rtf_list,
        "ttfp": [None] * ITERATIONS,
        "text": last_text,
        "audio_dur": audio_dur,
    }


async def bench_stream_stt_live(client: VoiceManagerClient, wav: Path) -> dict:
    """ITERATIONS timed runs of client.stream_stt_live() with real-time pacing.

    Strategy per iteration:
      - chunks() yields wav frames at ~real-time; sets audio_done when the
        wav is exhausted.
      - consumer records first is_final=False (TTFP) and the last is_final=True
        time (E2E). Breaks once a final arrives after audio_done.
      - Whole thing is bounded by audio_duration + 15s safety timeout so
        whisper (which may emit no further finals after the buffered one)
        cannot hang the script.
    """
    audio_dur = read_wav_duration(wav)
    e2e_list: list[float] = []
    rtf_list: list[float] = []
    ttfp_list: list[float | None] = []
    last_text = ""

    for _ in range(ITERATIONS):
        audio_done = asyncio.Event()
        start = time.perf_counter()
        ttfp: float | None = None
        last_final: float | None = None
        iter_text = ""

        async def chunks():
            async for c in wav_to_chunks(wav):
                yield c
            audio_done.set()

        async def consume():
            nonlocal ttfp, last_final, iter_text
            # Strict identity (`is True` / `is False`) — a missing `is_final` key must
            # not be treated as a partial or a final.
            async for msg in client.stream_stt_live(chunks()):
                now = time.perf_counter() - start
                if msg.get("is_final") is False and ttfp is None:
                    ttfp = now
                if msg.get("is_final") is True:
                    last_final = now
                    text = msg.get("text", "")
                    if text:
                        iter_text = text
                    if audio_done.is_set():
                        break

        try:
            await asyncio.wait_for(consume(), timeout=audio_dur + 15)
        except asyncio.TimeoutError:
            pass  # use whatever we collected

        if last_final is None:
            last_final = time.perf_counter() - start  # no final emitted; record bound
        e2e_list.append(last_final)
        rtf_list.append(last_final / audio_dur if audio_dur > 0 else 0.0)
        ttfp_list.append(ttfp)
        if iter_text:
            last_text = iter_text

    return {
        "e2e": e2e_list,
        "rtf": rtf_list,
        "ttfp": ttfp_list,
        "text": last_text,
        "audio_dur": audio_dur,
    }


# ---------------------------------------------------------------------------
# TTS benchmarks
# ---------------------------------------------------------------------------

async def fake_llm_stream(text: str = TTS_TEXT, chunk_size: int = TTS_CHUNK_SIZE) -> AsyncIterator[str]:
    """Yield small text chunks to simulate streamed LLM tokens.

    Same shape as fake_llm_stream in tests/test_stream_tts.py.
    """
    for i in range(0, len(text), chunk_size):
        yield text[i : i + chunk_size]
        await asyncio.sleep(0.02)


async def bench_synthesize_tts(client: VoiceManagerClient) -> dict:
    """ITERATIONS timed runs of client.synthesize_tts(TTS_TEXT).

    synthesize_tts is non-streaming, so TTFA == E2E (no first-chunk event
    exists). We still report TTFA so the column lines up with stream_tts_live.
    """
    e2e_list: list[float] = []
    ttfa_list: list[float] = []
    rtf_list: list[float] = []
    rate = 0
    sanity_ok = True

    for _ in range(ITERATIONS):
        start = time.perf_counter()
        pcm, rate = await client.synthesize_tts(TTS_TEXT)
        e2e = time.perf_counter() - start
        duration = len(pcm) / (rate * 2) if rate > 0 else 0.0
        e2e_list.append(e2e)
        ttfa_list.append(e2e)  # non-streaming
        rtf_list.append(e2e / duration if duration > 0 else 0.0)
        if not (rate > 0 and len(pcm) > 0 and duration >= 0.3):
            sanity_ok = False

    return {
        "e2e": e2e_list,
        "ttfa": ttfa_list,
        "rtf": rtf_list,
        "rate": rate,
        "sanity": sanity_ok,
    }


async def bench_stream_tts_live(client: VoiceManagerClient) -> dict:
    """ITERATIONS timed runs of client.stream_tts_live() with fake LLM stream."""
    e2e_list: list[float] = []
    ttfa_list: list[float] = []
    rtf_list: list[float] = []
    rate = 0
    sanity_ok = True

    for _ in range(ITERATIONS):
        start = time.perf_counter()
        ttfa: float | None = None
        total = bytearray()
        async for chunk in client.stream_tts_live(fake_llm_stream()):
            if ttfa is None and chunk:
                ttfa = time.perf_counter() - start
            total.extend(chunk)
        e2e = time.perf_counter() - start

        rate = client.last_tts_sample_rate
        duration = len(total) / (rate * 2) if rate > 0 else 0.0

        e2e_list.append(e2e)
        ttfa_list.append(ttfa if ttfa is not None else e2e)
        rtf_list.append(e2e / duration if duration > 0 else 0.0)
        if not (rate > 0 and len(total) > 0 and duration >= 0.3):
            sanity_ok = False

    return {
        "e2e": e2e_list,
        "ttfa": ttfa_list,
        "rtf": rtf_list,
        "rate": rate,
        "sanity": sanity_ok,
    }


# ---------------------------------------------------------------------------
# Stats + formatting
# ---------------------------------------------------------------------------

def stats(samples: list[float]) -> tuple[float, float, float]:
    """Return (median, min, max). Samples must be non-empty."""
    return (statistics.median(samples), min(samples), max(samples))


def _fmt_trio(values: list[float]) -> str:
    med, lo, hi = stats(values)
    return f"{med:.2f} / {lo:.2f} / {hi:.2f}"


def _fmt_opt(values: list[float | None]) -> str:
    real = [v for v in values if v is not None]
    if not real:
        return "—"
    return f"{statistics.median(real):.2f}"


def format_stt_table_stdout(clip: Path, results: dict[str, dict]) -> str:
    """results keys: 'transcribe_file', 'stream_stt_live'."""
    audio_dur = results["transcribe_file"]["audio_dur"]
    lines = [
        f"STT  (clip: {clip.name}, {audio_dur:.1f}s audio)",
        "  method                  E2E (med/min/max)    RTF (med)    TTFP (med)",
    ]
    for name in ("transcribe_file", "stream_stt_live"):
        r = results[name]
        e2e = _fmt_trio(r["e2e"])
        rtf_med = f"{statistics.median(r['rtf']):.2f}"
        ttfp_med = _fmt_opt(r["ttfp"])
        lines.append(f"  {name:<22}  {e2e:<19}  {rtf_med:<9}    {ttfp_med}")
    return "\n".join(lines)


def format_tts_table_stdout(results: dict[str, dict]) -> str:
    """results keys: 'synthesize_tts', 'stream_tts_live'."""
    lines = [
        f"TTS  (text: {len(TTS_TEXT)} chars)",
        "  method                  E2E (med/min/max)    TTFA (med)   RTF (med)   sanity",
    ]
    for name in ("synthesize_tts", "stream_tts_live"):
        r = results[name]
        e2e = _fmt_trio(r["e2e"])
        ttfa_med = f"{statistics.median(r['ttfa']):.2f}"
        rtf_med = f"{statistics.median(r['rtf']):.2f}"
        sanity = "OK" if r["sanity"] else "FAIL"
        lines.append(f"  {name:<22}  {e2e:<19}  {ttfa_med:<10}   {rtf_med:<9}   {sanity}")
    return "\n".join(lines)


def format_markdown_section(
    info: dict,
    stt_results_by_clip: dict[Path, dict[str, dict]],
    tts_results: dict[str, dict],
) -> str:
    """Build the markdown section to append for this run."""
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    server_bits = [
        f"stt_engine={info.get('stt_engine')}",
        f"tts_engine={info.get('tts_engine')}",
    ]
    if info.get("device"):
        server_bits.append(f"device={info['device']}")
    out = [f"## {stamp}", "", "Server: " + ", ".join(server_bits), ""]

    # STT tables (one per clip)
    for clip, results in stt_results_by_clip.items():
        audio_dur = results["transcribe_file"]["audio_dur"]
        out.append(f"### STT — `{clip.name}` ({audio_dur:.1f}s audio)")
        out.append("")
        out.append("| method | E2E (med/min/max) | RTF (med) | TTFP (med) |")
        out.append("|---|---|---|---|")
        for name in ("transcribe_file", "stream_stt_live"):
            r = results[name]
            rtf_med = f"{statistics.median(r['rtf']):.2f}"
            out.append(f"| `{name}` | {_fmt_trio(r['e2e'])} | {rtf_med} | {_fmt_opt(r['ttfp'])} |")
        out.append("")

    # TTS table
    out.append(f"### TTS — {len(TTS_TEXT)} chars")
    out.append("")
    out.append("| method | E2E (med/min/max) | TTFA (med) | RTF (med) | rate | sanity |")
    out.append("|---|---|---|---|---|---|")
    for name in ("synthesize_tts", "stream_tts_live"):
        r = tts_results[name]
        ttfa_med = f"{statistics.median(r['ttfa']):.2f}"
        rtf_med = f"{statistics.median(r['rtf']):.2f}"
        sanity = "OK" if r["sanity"] else "FAIL"
        out.append(
            f"| `{name}` | {_fmt_trio(r['e2e'])} | {ttfa_med} | {rtf_med} | {r['rate']} | {sanity} |"
        )
    out.append("")

    # Transcriptions (last iteration text per clip+method)
    out.append("### Transcriptions (last iteration)")
    out.append("")
    for clip, results in stt_results_by_clip.items():
        out.append(f"- `{clip.name}`")
        for name in ("transcribe_file", "stream_stt_live"):
            text = results[name]["text"].strip() or "—"
            out.append(f"  - `{name}`: {text}")
    out.append("")
    return "\n".join(out)


def append_markdown_log(section: str) -> None:
    """Append (or create) tests/benchmark_results.md.

    First write seeds the file with a heading so it's recognisable.
    """
    header = "# VoiceManager Benchmark Results\n\n" if not RESULTS_PATH.exists() else ""
    with RESULTS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(header + section + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    token = os.environ["VOICEMANAGER_TOKEN"]

    async with VoiceManagerClient(token=token) as client:
        info = await client.info()
        device = info.get("device")
        device_bit = f", device={device}" if device else ""
        print(
            f"VoiceManager Benchmark — {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Server: stt_engine={info.get('stt_engine')}, "
            f"tts_engine={info.get('tts_engine')}{device_bit}\n"
        )

        clips = select_clips()

        # --- STT warm-up (one untimed call per method on first clip) ---
        print("Warming up STT ...", flush=True)
        await client.transcribe_file(str(clips[0]))

        async def _warmup_chunks():
            async for c in wav_to_chunks(clips[0]):
                yield c

        # consume one final to ensure the model is fully primed
        try:
            await asyncio.wait_for(
                _consume_one_final(client.stream_stt_live(_warmup_chunks())),
                timeout=read_wav_duration(clips[0]) + 15,
            )
        except asyncio.TimeoutError:
            pass

        # --- STT benchmarks ---
        stt_results_by_clip: dict[Path, dict[str, dict]] = {}
        for clip in clips:
            print(f"\nBenchmarking STT on {clip.name} ...", flush=True)
            r_tf = await bench_transcribe_file(client, clip)
            r_sl = await bench_stream_stt_live(client, clip)
            stt_results_by_clip[clip] = {
                "transcribe_file": r_tf,
                "stream_stt_live": r_sl,
            }
            print()
            print(format_stt_table_stdout(clip, stt_results_by_clip[clip]))

        # --- TTS warm-up ---
        print("\nWarming up TTS ...", flush=True)
        await client.synthesize_tts(TTS_TEXT)
        async for _ in client.stream_tts_live(fake_llm_stream()):
            pass

        # --- TTS benchmarks ---
        print("Benchmarking TTS ...", flush=True)
        tts_results = {
            "synthesize_tts": await bench_synthesize_tts(client),
            "stream_tts_live": await bench_stream_tts_live(client),
        }
        print()
        print(format_tts_table_stdout(tts_results))

        # --- Persist markdown log ---
        section = format_markdown_section(info, stt_results_by_clip, tts_results)
        append_markdown_log(section)
        print(f"\nWrote {RESULTS_PATH} (appended)")

        # --- Exit code: non-zero on any TTS sanity failure ---
        if not all(r["sanity"] for r in tts_results.values()):
            sys.exit(1)


async def _consume_one_final(it):
    """Drain an stream_stt_live iterator until one is_final=True arrives."""
    async for msg in it:
        if msg.get("is_final") is True:
            return


if __name__ == "__main__":
    asyncio.run(main())
