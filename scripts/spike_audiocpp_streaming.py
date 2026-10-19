#!/usr/bin/env python3
"""Spike harness - audio.cpp C++ streaming session - MANUAL.

Drives audio.cpp's **C++ streaming session** (``NemotronASRStreamingSession``,
reached through the generic ``IStreamingVoiceTaskSession`` interface) with
one multi-word utterance fed in chunk-sized pieces, and observes the live
sub-utterance partials it emits.

The streaming session is exercised via ``audiocpp_cli --mode streaming``. That CLI
path calls, per chunk:

    session.process_audio_chunk(chunk) -> StreamEvent   (interim ``partial_text`` deltas)
    ...
    session.finish_stream()            -> TaskResult     (the utterance-final text)

(see app/streaming/streaming.cpp ``run_streaming_task`` + app/cli/main.cpp
``run_streaming``). It is emphatically **NOT** the full-body HTTP server
(``audiocpp_server``), which reads the entire request body before transcribing
and so cannot emit live partials (the eventual backend drives the
C++ session, never the HTTP server). This harness only ever spawns
``audiocpp_cli``.

The audio.cpp streaming session emits ``partial_text`` as token/char **deltas**
(``L`` -> ``o`` -> ``ng`` -> ` ` -> ``ra`` -> ...). This harness accumulates those
deltas into a running transcript so the growth is visible, and gates on:

    >= 2 distinct, growing interim transcripts observed BEFORE the utterance-final
    result. A run emitting only a single per-utterance result FAILS.

THROWAWAY / MANUAL: needs the built CUDA CLI, the GGUF weights, a GPU, and ffmpeg.
It is a scripts/ helper (never collected by the model-free pytest suite), mirroring
scripts/spike_nemotron_2606.py. Do not add it to the automated suite.

Run (from the VoiceManager repo root):
    ./venv/bin/python scripts/spike_audiocpp_streaming.py            # default: longrange clip, f16
    ./venv/bin/python scripts/spike_audiocpp_streaming.py --clip notauth --model q8_0
    ./venv/bin/python scripts/spike_audiocpp_streaming.py --audio /path/to/16k_mono.wav
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

# --- Fixed spike environment --------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIOCPP_ROOT = REPO_ROOT / "3rdParty" / "audio.cpp"
# Run the CLI with cwd = audio.cpp tree root: model_specs/ and assets/ resolve
# relative to cwd.
CLI = Path(os.environ.get(
    "AUDIOCPP_CLI", AUDIOCPP_ROOT / "build" / "linux-cuda-release" / "bin" / "audiocpp_cli"))
# The full-body HTTP server we must NOT use, kept here only to assert its absence.
SERVER = AUDIOCPP_ROOT / "build" / "linux-cuda-release" / "bin" / "audiocpp_server"

MODELS_DIR = Path(os.environ.get("AUDIOCPP_MODELS_DIR", "/home/user/audiocpp-models"))
GGUF_DIR = MODELS_DIR / "Nemotron-3.5-ASR-Streaming-0.6B-GGUF"
GGUF = {
    "f16": GGUF_DIR / "nemotron-3.5-asr-streaming-0.6b-f16.gguf",
    "q8_0": GGUF_DIR / "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf",
}

# The six tng regression clips (same apples-to-apples set the NeMo re-gate and
# the GGUF load check used). Default reference utterance is a clear multi-word phrase.
TNG_DIR = REPO_ROOT / "3rdParty" / "tng"
CLIPS = {
    "access": "accessinglibrarycomputerdata_clean.mp3",       # Accessing library computer data.
    "longrange": "longrangescanplanetcomplete_clean.mp3",     # Long range scan of planet complete.
    "notauth": "youarenotauthorisedtoaccessthisfacility_clean.mp3",
    "donot": "donotaddressthisunitinthatmanner_clean.mp3",
    "direction": "directionunclearpleaserepeat_clean.mp3",
    "incoming": "incomingtransmission_clean.mp3",
}
DEFAULT_CLIP = "longrange"
SAMPLE_RATE = 16000

# CLI stdout markers emitted by app/cli/main.cpp ``run_streaming``.
_PARTIAL_RE = re.compile(r"^partial_text=(.*)$")
_ACTIVITY_RE = re.compile(r"^event=(\w+) sample=(-?\d+) probability=([\d.eE+-]+)$")
_MODE_RE = re.compile(r"^mode=(\w+)$")
_FINAL_RE = re.compile(r"^text_output=(.*)$")


def _ensure_16k_mono_wav(mp3_path: Path, dst: Path) -> Path:
    """Convert a tng mp3 to 16 kHz mono 16-bit wav (audio.cpp's expected input)."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3_path),
         "-ar", str(SAMPLE_RATE), "-ac", "1", str(dst)],
        check=True,
    )
    with wave.open(str(dst), "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE and wf.getnchannels() == 1 and wf.getsampwidth() == 2, (
            f"{dst}: expected 16 kHz mono 16-bit wav"
        )
    return dst


def _run_streaming_cli(wav: Path, model: Path, device: int, final_out: Path) -> tuple[list[str], int]:
    """Spawn ``audiocpp_cli --mode streaming`` on the C++ streaming session.

    Returns (stdout_lines, returncode). The CLI (not the HTTP server) drives
    process_audio_chunk -> StreamEvent -> finish_stream/finalize internally.
    """
    cmd = [
        str(CLI),
        "--task", "asr", "--family", "nemotron_asr",
        "--model", str(model),
        "--backend", "cuda", "--device", str(device),
        "--mode", "streaming",
        "--audio", str(wav),
        "--language", "en-US",
        "--text-out", str(final_out),
    ]
    proc = subprocess.run(cmd, cwd=str(AUDIOCPP_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
    return proc.stdout.splitlines(), proc.returncode


def _parse_stream_events(lines: list[str]) -> tuple[list[dict], str | None, str | None]:
    """Turn CLI stdout into an ordered StreamEvent list + (mode, final_text).

    Each ``partial_text=`` line is one interim StreamEvent (a delta); we also
    carry the accumulated transcript so growth is observable. ``event=speech_*``
    lines are voice_activity StreamEvents. ``text_output=`` is the finalize()
    aggregate emitted after the stream.
    """
    events: list[dict] = []
    accumulated = ""
    mode: str | None = None
    final_text: str | None = None
    for line in lines:
        if (m := _PARTIAL_RE.match(line)) is not None:
            accumulated += m.group(1)
            events.append({"kind": "partial", "delta": m.group(1), "accumulated": accumulated})
        elif (m := _ACTIVITY_RE.match(line)) is not None:
            events.append({"kind": "activity", "activity": m.group(1),
                           "sample": int(m.group(2)), "probability": float(m.group(3))})
        elif (m := _MODE_RE.match(line)) is not None:
            mode = m.group(1)
        elif (m := _FINAL_RE.match(line)) is not None:
            final_text = m.group(1)
            events.append({"kind": "final", "text": m.group(1)})
    return events, mode, final_text


def _evaluate(events: list[dict], mode: str | None, final_text: str | None, rc: int) -> bool:
    """Gate on live partials: >= 2 distinct growing interims BEFORE the utterance-final."""
    final_idx = next((i for i, e in enumerate(events) if e["kind"] == "final"), None)
    before_final = events[:final_idx] if final_idx is not None else events

    # Distinct, growing accumulated interim snapshots that precede the final.
    growing_snapshots: list[str] = []
    prev_len = -1
    for e in before_final:
        if e["kind"] != "partial":
            continue
        acc = e["accumulated"]
        if len(acc) > prev_len:            # strictly growing transcript
            growing_snapshots.append(acc)
            prev_len = len(acc)
    distinct_growing = len(dict.fromkeys(growing_snapshots))

    checks = {
        "cli_exit_0": rc == 0,
        "mode_is_streaming": mode == "streaming",           # C++ streaming session, not HTTP
        "http_server_path_unused": True,                    # by construction: only audiocpp_cli spawned
        ">=2 growing interims before final": distinct_growing >= 2,
        "final_after_interims": final_idx is not None and final_idx > 0,
        "final_nonempty": bool(final_text and final_text.strip()),
    }

    print("\n=== observed StreamEvents (in order) ===")
    for i, e in enumerate(events):
        if e["kind"] == "partial":
            print(f"  [{i:3d}] interim  delta={e['delta']!r:12s} -> accumulated={e['accumulated']!r}")
        elif e["kind"] == "activity":
            print(f"  [{i:3d}] activity {e['activity']} sample={e['sample']} p={e['probability']:.3f}")
        elif e["kind"] == "final":
            print(f"  [{i:3d}] FINAL    text_output={e['text']!r}  (finish_stream/finalize)")

    interim_count = sum(1 for e in before_final if e["kind"] == "partial")
    print("\n=== gate ===")
    print(f"  interim partial events before final : {interim_count}")
    print(f"  distinct, GROWING interim snapshots  : {distinct_growing}")
    print(f"  session mode                         : {mode}")
    print(f"  final (finalize) text                : {final_text!r}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    passed = all(checks.values())
    print(f"\n=== VERDICT: {'PASS' if passed else 'FAIL'} ===")
    return passed


def main() -> int:
    ap = argparse.ArgumentParser(description="audio.cpp C++ streaming-session spike")
    ap.add_argument("--clip", choices=sorted(CLIPS), default=DEFAULT_CLIP,
                    help="tng reference utterance (default: longrange, a multi-word phrase)")
    ap.add_argument("--audio", type=Path, default=None,
                    help="override: a ready 16 kHz mono 16-bit wav (skips tng conversion)")
    ap.add_argument("--model", choices=sorted(GGUF), default="f16")
    ap.add_argument("--device", type=int, default=int(os.environ.get("VM_AUDIOCPP_DEVICE", "0")),
                    help="CUDA device index (default: $VM_AUDIOCPP_DEVICE, else 0)")
    args = ap.parse_args()

    if not CLI.exists():
        sys.exit(f"CLI not built: {CLI} (run the audio.cpp CUDA build first, see README)")
    model = GGUF[args.model]
    if not model.exists():
        sys.exit(f"GGUF not found: {model}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        if args.audio is not None:
            wav = args.audio
        else:
            wav = _ensure_16k_mono_wav(TNG_DIR / CLIPS[args.clip], tmp / f"{args.clip}_16k.wav")
        final_out = tmp / "final.txt"

        print(f"[spike] clip={args.clip if args.audio is None else args.audio}  "
              f"model={args.model}  device=cuda:{args.device}")
        print(f"[spike] driving C++ streaming session via {CLI.name} (HTTP server "
              f"{SERVER.name} deliberately unused)")
        lines, rc = _run_streaming_cli(wav, model, args.device, final_out)

    events, mode, final_text = _parse_stream_events(lines)
    passed = _evaluate(events, mode, final_text, rc)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
