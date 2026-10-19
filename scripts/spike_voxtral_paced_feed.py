#!/usr/bin/env python3
"""Spike harness - Voxtral mid-speech partials from a paced live feed - MANUAL.

Feeds a multi-second reference clip to ``audiocpp_cli --mode streaming --audio -``
as raw s16le PCM on stdin, paced to real time (default 100 ms of audio per write,
scheduled against a monotonic clock) to simulate a live microphone. A reader
thread timestamps every ``partial_text`` delta the moment it arrives on the CLI's
stdout (the CLI flushes each partial explicitly, see app/cli/main.cpp).

Gate: at least 2 distinct, GROWING accumulated partial transcripts whose
arrival timestamps precede the end of the audio feed. A single compute-paced
burst at finalize - the nemotron sidecar pattern - FAILS.

Voxtral context: the model decodes one text token per 80 ms audio token with a
~480 ms architectural delay, so on a real-time feed the first partial is expected
roughly a second into the clip, then word-by-word partials THROUGHOUT the feed.

THROWAWAY / MANUAL: needs the built CUDA CLI ($AUDIOCPP_CLI with the
voxtral_realtime loader compiled in), the q4_k GGUF staged under
/home/user/audiocpp-models/, a GPU, and ffmpeg. It is a scripts/ helper (never
collected by the model-free pytest suite). Do not add it to the automated suite.

Run (from the VoiceManager repo root):
    ./venv/bin/python scripts/spike_voxtral_paced_feed.py                 # default: 16 s tng concat
    ./venv/bin/python scripts/spike_voxtral_paced_feed.py --audio /path/to/16k_mono.wav
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIOCPP_ROOT = REPO_ROOT / "3rdParty" / "audio.cpp"
CLI = Path(os.environ.get(
    "AUDIOCPP_CLI", AUDIOCPP_ROOT / "build" / "linux-cuda-release" / "bin" / "audiocpp_cli"))
GGUF = (Path(os.environ.get("AUDIOCPP_MODELS_DIR", "/home/user/audiocpp-models"))
        / "Voxtral-Mini-4B-Realtime-2602-GGUF" / "voxtral-mini-4b-realtime-2602-q4_k.gguf")
TNG_CLIPS = sorted((REPO_ROOT / "3rdParty" / "tng").glob("*_clean.mp3"))

SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2


def build_concat_clip(out_wav: Path) -> None:
    """Concatenate the six tng reference MP3s into one 16 kHz mono s16 WAV + 1 s pad."""
    inputs: list[str] = []
    for clip in TNG_CLIPS:
        inputs += ["-i", str(clip)]
    n = len(TNG_CLIPS)
    filtergraph = (
        "".join(f"[{i}:a]" for i in range(n))
        + f"concat=n={n}:v=0:a=1[cat];[cat]apad=pad_dur=1.0[out]"
    )
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", filtergraph,
         "-map", "[out]", "-ar", str(SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16",
         str(out_wav)],
        check=True,
    )


def read_pcm(wav_path: Path) -> bytes:
    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE, "harness expects a 16 kHz clip"
        assert wav.getnchannels() == 1, "harness expects a mono clip"
        assert wav.getsampwidth() == BYTES_PER_SAMPLE, "harness expects s16 samples"
        return wav.readframes(wav.getnframes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--audio", type=Path, default=None,
                        help="16 kHz mono s16 WAV to feed (default: tng concat)")
    parser.add_argument("--chunk-ms", type=int, default=100,
                        help="audio per stdin write, milliseconds (default 100)")
    parser.add_argument("--device", type=int, default=int(os.environ.get("VM_AUDIOCPP_DEVICE", "0")),
                        help="CUDA device index in the CLI's ordering "
                             "(default: $VM_AUDIOCPP_DEVICE, else 0)")
    parser.add_argument("--session-option", action="append", default=[],
                        metavar="KEY=VALUE",
                        help="extra --session-option passed through to the CLI "
                             "(repeatable; e.g. voxtral_realtime.audio_encoder_graph_arena_mb=256)")
    args = parser.parse_args()

    if not CLI.exists():
        sys.exit(f"CLI not built: {CLI}")
    if not GGUF.exists():
        sys.exit(f"GGUF not staged: {GGUF}")

    tmpdir = tempfile.TemporaryDirectory(prefix="voxtral_paced_")
    if args.audio is None:
        clip = Path(tmpdir.name) / "tng_concat_16k_pad1s.wav"
        build_concat_clip(clip)
    else:
        clip = args.audio
    pcm = read_pcm(clip)
    clip_seconds = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    chunk_bytes = SAMPLE_RATE * BYTES_PER_SAMPLE * args.chunk_ms // 1000
    print(f"clip={clip.name} duration={clip_seconds:.2f}s chunk={args.chunk_ms}ms")

    session_options: list[str] = []
    for option in args.session_option:
        session_options += ["--session-option", option]
    proc = subprocess.Popen(
        [str(CLI), "--task", "asr", "--family", "voxtral_realtime",
         "--model", str(GGUF), "--backend", "cuda", "--device", str(args.device),
         "--mode", "streaming", "--audio", "-", "--input-format", "s16le",
         "--input-rate", str(SAMPLE_RATE), "--input-channels", "1",
         *session_options],
        cwd=AUDIOCPP_ROOT,  # model_specs/ resolves relative to cwd
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, bufsize=0,
    )

    # (timestamp, accumulated_transcript) per partial; timestamps are offsets
    # from feed start on the same monotonic clock the feeder schedules against.
    partials: list[tuple[float, str]] = []
    final_text: list[str] = []
    t0 = time.monotonic()

    def reader() -> None:
        accumulated = ""
        for raw in iter(proc.stdout.readline, b""):
            now = time.monotonic() - t0
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if line.startswith("partial_text="):
                accumulated += line[len("partial_text="):]
                partials.append((now, accumulated))
                print(f"  [{now:7.3f}s] partial: {accumulated!r}")
            elif line.startswith("text_output="):
                final_text.append(line[len("text_output="):])
                print(f"  [{now:7.3f}s] final:   {final_text[0]!r}")

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    # Absolute-schedule pacing: chunk i goes out at t0 + i*chunk, immune to drift.
    n_chunks = (len(pcm) + chunk_bytes - 1) // chunk_bytes
    for i in range(n_chunks):
        target = t0 + i * args.chunk_ms / 1000.0
        delay = target - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        proc.stdin.write(pcm[i * chunk_bytes:(i + 1) * chunk_bytes])
    proc.stdin.close()  # audio source runs dry -> the session finalizes
    feed_end = time.monotonic() - t0
    print(f"  [{feed_end:7.3f}s] --- feed ended ({n_chunks} chunks) ---")

    proc.wait(timeout=120)
    reader_thread.join(timeout=10)

    # Gate: >= 2 distinct growing partials strictly before the feed ended.
    before_feed_end = [p for p in partials if p[0] < feed_end]
    distinct_growing = 0
    prev = ""
    for _, text in before_feed_end:
        if len(text) > len(prev):
            distinct_growing += 1
            prev = text
    print(f"\npartials total={len(partials)} before_feed_end={len(before_feed_end)} "
          f"distinct_growing_before_feed_end={distinct_growing}")
    if final_text:
        print(f"final: {final_text[0]!r}")
    if distinct_growing >= 2:
        print("PASS: genuine mid-speech partials while audio was still being fed")
        return 0
    print("FAIL: partials did not precede the end of the audio feed "
          "(nemotron-style finalize burst)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
