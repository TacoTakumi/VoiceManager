#!/usr/bin/env python3
"""Spike - Voxtral vs NeMo nemotron paced head-to-head - MANUAL.

Runs the SAME pinned reference clip set through both streaming paths on the
chosen device (DEVICE, from VM_AUDIOCPP_DEVICE) and records, per clip:

  - TTFP: wall-clock time-to-first-partial, audio-in to first partial, under a
    real-time paced feed (a simulated live mic). t0 is the moment speech
    capture starts on a READY engine (model loaded - the persistent sidecar
    amortizes load). Chunk i covers audio [i*c, (i+1)*c) and therefore
    only EXISTS at t0 + (i+1)*c; each chunk is delivered at its availability
    time, never earlier. Skipping that shift hands an engine its first chunk
    of audio before a microphone could have captured it and understates TTFP
    by one full chunk (1.12 s for NeMo's native cadence).
  - the final transcript, for the not-worse quality eyeball vs the reference.

Paths:
  - voxtral : audiocpp_cli (voxtral_realtime, q4_k GGUF) fed s16le
    PCM on stdin in 100 ms real-time-paced chunks. t0 = the CLI's
    ``audio_input=stdin`` marker (printed once the model is loaded and the
    session is about to consume). Each run is a fresh process, so chunk-1
    carries the per-session CUDA graph compile - a handicap voxtral absorbs.
  - nemo    : the SHIPPED VM path (nemotron-speech-streaming-en-0.6b,
    att_context [70,13], NeMo 26.06) via spike_nemotron_2606.build_pipeline,
    fed Frame chunks at the pipeline's native cadence on the same real-time
    schedule, WARM (one discarded warmup clip; the running VM server keeps
    NeMo warm the same way).

Clip set (pinned; reused verbatim by the later WS head-to-head): the six tng
reference MP3s as 16 kHz mono s16 WAV each padded with 1 s of trailing
silence, plus ``concat17`` - all six concatenated + 1 s pad (17.27 s, the
required >10 s utterance). Built deterministically into 3rdParty/tng/derived/ and
pinned by sha256 in the spike record.

THROWAWAY / MANUAL: needs the built CUDA CLI with the voxtral loader, the
q4_k GGUF, the nemo2606-spike-venv with the NeMo model cached, a GPU, and
ffmpeg. A scripts/ helper, never collected by the model-free pytest suite.

Run (from the VoiceManager repo root):
    ./venv/bin/python scripts/spike_voxtral_headtohead.py                # both engines
    ./venv/bin/python scripts/spike_voxtral_headtohead.py --engine voxtral
"""
from __future__ import annotations

import argparse
import os
import hashlib
import json
import statistics
import subprocess
import sys
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

# Shipped NeMo path (src/voicemanager/config.py defaults for VM_STT_ENGINE=nemotron).
NEMO_PY = Path(os.environ.get("NEMO_PY", "/home/user/nemo2606-spike-venv/bin/python"))
NEMO_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
NEMO_ATT_CONTEXT = "70,13"

DEVICE = int(os.environ.get("VM_AUDIOCPP_DEVICE", "0"))  # CUDA fastest-first order
SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 2
CHUNK_MS = 100  # voxtral stdin feed granularity; nemo uses its native cadence
N_RUNS = 3      # per clip; per-clip figure = median

TNG_DIR = REPO_ROOT / "3rdParty" / "tng"
DERIVED = TNG_DIR / "derived"
CLIPS = {
    "access": "accessinglibrarycomputerdata_clean.mp3",
    "direction": "directionunclearpleaserepeat_clean.mp3",
    "donot": "donotaddressthisunitinthatmanner_clean.mp3",
    "incoming": "incomingtransmission_clean.mp3",
    "longrange": "longrangescanplanetcomplete_clean.mp3",
    "notauth": "youarenotauthorisedtoaccessthisfacility_clean.mp3",
}
REFERENCE = {
    "access": "Accessing library computer data",
    "direction": "Direction unclear. Please repeat request.",
    "donot": "Please do not address this unit in that manner",
    "incoming": "Incoming transmission",
    "longrange": "Long-range scan of planet complete",
    "notauth": "You are not authorised to access this facility",
    "concat17": "(all six, in the order above)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_clip_set() -> dict[str, Path]:
    """Six padded singles + the 17.27 s concat, deterministic, into 3rdParty/tng/derived/."""
    DERIVED.mkdir(exist_ok=True)
    wavs: dict[str, Path] = {}
    for name, mp3 in CLIPS.items():
        out = DERIVED / f"{name}_16k_pad1s.wav"
        if not out.exists():
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-i", str(TNG_DIR / mp3),
                 "-af", "apad=pad_dur=1.0", "-ar", str(SAMPLE_RATE), "-ac", "1",
                 "-sample_fmt", "s16", str(out)],
                check=True)
        wavs[name] = out
    concat = DERIVED / "concat17_16k_pad1s.wav"
    if not concat.exists():
        inputs: list[str] = []
        for mp3 in CLIPS.values():
            inputs += ["-i", str(TNG_DIR / mp3)]
        n = len(CLIPS)
        graph = ("".join(f"[{i}:a]" for i in range(n))
                 + f"concat=n={n}:v=0:a=1[cat];[cat]apad=pad_dur=1.0[out]")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph,
             "-map", "[out]", "-ar", str(SAMPLE_RATE), "-ac", "1",
             "-sample_fmt", "s16", str(concat)],
            check=True)
    wavs["concat17"] = concat
    return wavs


def read_pcm(wav_path: Path) -> bytes:
    with wave.open(str(wav_path), "rb") as wav:
        assert wav.getframerate() == SAMPLE_RATE
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == BYTES_PER_SAMPLE
        return wav.readframes(wav.getnframes())


# --- voxtral path --------------------------------------------------------------
def voxtral_run(wav: Path) -> tuple[float, str, str]:
    """One cold-process paced run; returns (ttfp_ms, first_partial, final_text)."""
    pcm = read_pcm(wav)
    chunk_bytes = SAMPLE_RATE * BYTES_PER_SAMPLE * CHUNK_MS // 1000
    proc = subprocess.Popen(
        [str(CLI), "--task", "asr", "--family", "voxtral_realtime",
         "--model", str(GGUF), "--backend", "cuda", "--device", str(DEVICE),
         "--mode", "streaming", "--audio", "-", "--input-format", "s16le",
         "--input-rate", str(SAMPLE_RATE), "--input-channels", "1"],
        cwd=AUDIOCPP_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, bufsize=0)

    ready = threading.Event()
    state: dict[str, float | str | None] = {"t0": None, "ttfp": None,
                                            "first": None, "final": None}

    def reader() -> None:
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if line.startswith("audio_input=stdin"):
                ready.set()  # model loaded; the session is about to consume
            elif line.startswith("partial_text=") and state["ttfp"] is None \
                    and state["t0"] is not None:
                state["ttfp"] = (time.monotonic() - state["t0"]) * 1000.0
                state["first"] = line[len("partial_text="):]
            elif line.startswith("text_output="):
                state["final"] = line[len("text_output="):].strip()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    if not ready.wait(timeout=300):
        proc.kill()
        raise RuntimeError("voxtral CLI never became ready")

    t0 = time.monotonic()
    state["t0"] = t0
    n_chunks = (len(pcm) + chunk_bytes - 1) // chunk_bytes
    for i in range(n_chunks):
        # Chunk i is available once its audio has been captured: t0 + (i+1)*c.
        delay = (t0 + (i + 1) * CHUNK_MS / 1000.0) - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        proc.stdin.write(pcm[i * chunk_bytes:(i + 1) * chunk_bytes])
    proc.stdin.close()
    proc.wait(timeout=120)
    thread.join(timeout=10)
    if state["ttfp"] is None:
        raise RuntimeError(f"no partial observed for {wav.name}")
    return float(state["ttfp"]), str(state["first"]), str(state["final"] or "")


def measure_voxtral(wavs: dict[str, Path]) -> dict:
    print(f"[voxtral] q4_k GGUF, device {DEVICE}, {CHUNK_MS} ms paced chunks, "
          f"{N_RUNS} runs/clip")
    per_clip: dict[str, dict] = {}
    for name, wav in wavs.items():
        runs = [voxtral_run(wav) for _ in range(N_RUNS)]
        per_clip[name] = {
            "ttfp_ms": statistics.median(r[0] for r in runs),
            "first_partial_text": runs[0][1],
            "final_text": runs[0][2],
        }
        c = per_clip[name]
        print(f"[voxtral] {name:10s} ttfp={c['ttfp_ms']:7.1f}ms "
              f"first={c['first_partial_text']!r} final={c['final_text']!r}")
    return per_clip


# --- nemo path (runs under the nemo2606 venv) ----------------------------------
def measure_nemo(wavs: dict[str, Path]) -> dict:
    import os
    os.environ["VM_MODEL_NAME"] = NEMO_MODEL
    os.environ["VM_ATT_CONTEXT"] = NEMO_ATT_CONTEXT

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import numpy as np
    import spike_nemotron_2606 as nm
    import torch
    from nemo.collections.asr.inference.streaming.framing.request import Frame

    pipeline, _cfg = nm.build_pipeline("segment")
    chunk_secs = pipeline.chunk_size_in_secs
    chunk_samples = int(round(chunk_secs * SAMPLE_RATE))
    print(f"[nemo] model={NEMO_MODEL} att_context={NEMO_ATT_CONTEXT} "
          f"chunk={chunk_secs:.3f}s, paced, {N_RUNS} runs/clip")

    def paced_run(pcm: bytes) -> tuple[float, str, str]:
        audio = nm._pcm_to_float32(pcm)
        n_chunks = int(np.ceil(len(audio) / chunk_samples))
        options = nm._make_options()
        pipeline.open_session()
        ttfp_ms: float | None = None
        first = ""
        finals: list[str] = []
        last_partial = ""
        try:
            t0 = time.monotonic()
            for i in range(n_chunks):
                # Chunk i is available once its audio has been captured.
                delay = (t0 + (i + 1) * chunk_secs) - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                seg = audio[i * chunk_samples:(i + 1) * chunk_samples]
                valid = seg.shape[0]
                if valid < chunk_samples:
                    padded = np.zeros(chunk_samples, dtype=np.float32)
                    padded[:valid] = seg
                    seg = padded
                frame = Frame(samples=torch.from_numpy(seg.copy()), stream_id=0,
                              is_first=(i == 0), is_last=(i == n_chunks - 1),
                              length=valid, options=options if i == 0 else None)
                out = pipeline.transcribe_step([frame])[0]
                torch.cuda.synchronize(DEVICE)
                partial = out.partial_transcript.strip()
                final = out.final_transcript.strip()
                if final:
                    finals.append(final)
                if partial:
                    last_partial = partial
                if ttfp_ms is None and (partial or final):
                    ttfp_ms = (time.monotonic() - t0) * 1000.0
                    first = partial or final
        finally:
            pipeline.close_session()
        if ttfp_ms is None:
            raise RuntimeError("nemo produced no text")
        return ttfp_ms, first, " ".join(finals) if finals else last_partial

    paced_run(read_pcm(next(iter(wavs.values()))))  # discarded warmup pass

    per_clip: dict[str, dict] = {}
    for name, wav in wavs.items():
        pcm = read_pcm(wav)
        runs = [paced_run(pcm) for _ in range(N_RUNS)]
        per_clip[name] = {
            "ttfp_ms": statistics.median(r[0] for r in runs),
            "first_partial_text": runs[0][1],
            "final_text": runs[0][2],
        }
        c = per_clip[name]
        print(f"[nemo] {name:10s} ttfp={c['ttfp_ms']:7.1f}ms "
              f"first={c['first_partial_text']!r} final={c['final_text']!r}")
    return per_clip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", choices=["both", "voxtral", "nemo"],
                        default="both")
    args = parser.parse_args()

    wavs = build_clip_set()

    if args.engine == "nemo":
        result = measure_nemo(wavs)
        print("NEMO_JSON=" + json.dumps(result))
        return 0

    print("pinned clip set (sha256):")
    for name, wav in wavs.items():
        print(f"  {name:10s} {sha256(wav)}  {wav.relative_to(REPO_ROOT)}")

    voxtral = measure_voxtral(wavs)
    if args.engine == "voxtral":
        print("VOXTRAL_JSON=" + json.dumps(voxtral))
        return 0

    nemo_proc = subprocess.run(
        [str(NEMO_PY), str(Path(__file__).resolve()), "--engine", "nemo"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    sys.stderr.write(nemo_proc.stderr[-2000:])
    nemo: dict | None = None
    for line in nemo_proc.stdout.splitlines():
        print(line)
        if line.startswith("NEMO_JSON="):
            nemo = json.loads(line[len("NEMO_JSON="):])
    if nemo is None:
        sys.exit("nemo run produced no NEMO_JSON")

    print(f"\n{'clip':10s} {'voxtral ttfp':>13s} {'nemo ttfp':>10s}  verdict")
    for name in wavs:
        v, n = voxtral[name]["ttfp_ms"], nemo[name]["ttfp_ms"]
        verdict = "voxtral<=nemo PASS" if v <= n else "voxtral>nemo"
        print(f"{name:10s} {v:11.1f}ms {n:8.1f}ms  {verdict}")
    print("\nfinals (reference | voxtral | nemo):")
    for name in wavs:
        print(f"  {name}:\n    ref     : {REFERENCE[name]}\n"
              f"    voxtral : {voxtral[name]['final_text']}\n"
              f"    nemo    : {nemo[name]['final_text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
