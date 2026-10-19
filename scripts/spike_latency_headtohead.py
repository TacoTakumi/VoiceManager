#!/usr/bin/env python3
"""Spike - first-partial latency head-to-head - MANUAL.

Measures **warm first-partial latency** (compute time from the first audio chunk
submitted to the first non-empty interim partial, fed as-fast-as-possible) for two
streaming STT paths on the SAME six tng reference clips, and reports the median
ratio against the confirmed parity bar.

Confirmed metric + bar (2026-07-19):
  - metric : median first-partial latency over the six tng clips, compute-only.
  - basis  : WARM is the gate (model loaded + kernels autotuned; the persistent
             sidecar amortizes load + graph prebuild + first-chunk autotune
             once per session). COLD one-time cost recorded separately.
  - bar    : GO if median(audio.cpp) <= 1.5x median(NeMo); materially worse = NO-GO.

Two paths, measured apples-to-apples on one GPU (DEVICE):
  - audio.cpp : the audio.cpp CUDA build, nemotron-3.5 GGUF, driven via
                ``audiocpp_cli --mode streaming --log``. First-partial is read from
                the CLI's own per-chunk ``[TIMING]`` lines (frontend_ms +
                encoder.stream_ms), engine-internal compute. WARM = the K chunks to
                the first partial with chunk-1's cold autotune (~76 ms) replaced by
                the steady-state per-chunk median; COLD = the raw sum incl. autotune.
  - NeMo      : the SHIPPED VM path - nemotron-speech-streaming-en-0.6b,
                att_context=[70,13], NeMo 26.06. First-partial is the summed,
                cuda-synchronized ``transcribe_step`` wall-time to the first
                non-empty partial (the multilingual 3.5 is what NeMo cannot
                finalize, so en-0.6b is NeMo's best runnable streaming path).

THROWAWAY / MANUAL: needs the built CUDA CLI, the GGUF weights, both NeMo models
cached, a GPU, ffmpeg, and (for the NeMo path) the nemo2606-spike-venv. A scripts/
helper, never collected by the model-free pytest suite.

Run (from the VoiceManager repo root):
    # audio.cpp only (any python with numpy):
    ./venv/bin/python scripts/spike_latency_headtohead.py --engine audiocpp --model f16
    # NeMo only (MUST use the nemo2606 venv):
    /home/user/nemo2606-spike-venv/bin/python \
        scripts/spike_latency_headtohead.py --engine nemo --wav-dir <dir>
    # both (orchestrates audio.cpp in-process + shells out to the nemo venv):
    ./venv/bin/python scripts/spike_latency_headtohead.py --engine both
"""
from __future__ import annotations

import argparse
import os
import json
import re
import statistics
import subprocess
import sys
import wave
from pathlib import Path

# --- Fixed spike environment --------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIOCPP_ROOT = REPO_ROOT / "3rdParty" / "audio.cpp"
CLI = Path(os.environ.get(
    "AUDIOCPP_CLI", AUDIOCPP_ROOT / "build" / "linux-cuda-release" / "bin" / "audiocpp_cli"))

MODELS_DIR = Path(os.environ.get("AUDIOCPP_MODELS_DIR", "/home/user/audiocpp-models"))
GGUF_DIR = MODELS_DIR / "Nemotron-3.5-ASR-Streaming-0.6B-GGUF"
GGUF = {
    "f16": GGUF_DIR / "nemotron-3.5-asr-streaming-0.6b-f16.gguf",
    "q8_0": GGUF_DIR / "nemotron-3.5-asr-streaming-0.6b-q8_0.gguf",
}

# Shipped NeMo path (src/voicemanager/config.py defaults for VM_STT_ENGINE=nemotron).
NEMO_PY = Path(os.environ.get("NEMO_PY", "/home/user/nemo2606-spike-venv/bin/python"))
NEMO_MODEL = "nvidia/nemotron-speech-streaming-en-0.6b"
NEMO_ATT_CONTEXT = "70,13"

DEVICE = int(os.environ.get("VM_AUDIOCPP_DEVICE", "0"))  # one GPU for both paths
SAMPLE_RATE = 16000
N_RUNS = 5  # per clip, to smooth GPU jitter; per-clip figure = median of runs

TNG_DIR = REPO_ROOT / "3rdParty" / "tng"
CLIPS = {
    "access": "accessinglibrarycomputerdata_clean.mp3",
    "longrange": "longrangescanplanetcomplete_clean.mp3",
    "notauth": "youarenotauthorisedtoaccessthisfacility_clean.mp3",
    "donot": "donotaddressthisunitinthatmanner_clean.mp3",
    "direction": "directionunclearpleaserepeat_clean.mp3",
    "incoming": "incomingtransmission_clean.mp3",
}


# --- shared audio helpers -----------------------------------------------------
def ensure_wavs(wav_dir: Path) -> dict[str, Path]:
    """Convert each tng mp3 to 16 kHz mono 16-bit wav (once). Returns name -> wav."""
    wav_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for name, mp3 in CLIPS.items():
        wav = wav_dir / f"{name}_16k.wav"
        if not wav.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(TNG_DIR / mp3),
                 "-ar", str(SAMPLE_RATE), "-ac", "1", str(wav)],
                check=True,
            )
        with wave.open(str(wav), "rb") as wf:
            assert wf.getframerate() == SAMPLE_RATE and wf.getnchannels() == 1 and wf.getsampwidth() == 2
        out[name] = wav
    return out


def _median(xs: list[float]) -> float:
    return statistics.median(xs)


# --- audio.cpp path (CLI --log, engine-internal per-chunk compute) ------------
_FRONTEND_RE = re.compile(r"\] nemotron_asr\.frontend_ms ([\d.eE+-]+)")
_ENC_STREAM_RE = re.compile(r"\] nemotron_asr\.encoder\.stream_ms ([\d.eE+-]+)")
_PARTIAL_RE = re.compile(r"^partial_text=(.*)$")


def _run_audiocpp_cli(wav: Path, model: Path) -> list[str]:
    cmd = [
        str(CLI), "--task", "asr", "--family", "nemotron_asr",
        "--model", str(model), "--backend", "cuda", "--device", str(DEVICE),
        "--mode", "streaming", "--audio", str(wav), "--language", "en-US", "--log",
    ]
    proc = subprocess.run(cmd, cwd=str(AUDIOCPP_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise RuntimeError(f"audiocpp_cli exited {proc.returncode} on {wav.name}")
    return proc.stdout.splitlines()


def _parse_audiocpp_run(lines: list[str]) -> dict:
    """Per-chunk (frontend_ms + encoder.stream_ms) compute, and K = chunks to first partial.

    Chunks are delimited in stream order by their ``encoder.stream_ms`` line (one per
    streaming chunk). ``frontend_ms`` precedes it in the same chunk block. K = the
    number of encoder chunks completed when the first non-empty ``partial_text`` appears.
    """
    frontends: list[float] = []
    encoders: list[float] = []
    pending_frontend: float | None = None
    K: int | None = None
    first_partial_text: str | None = None
    for line in lines:
        if (m := _FRONTEND_RE.search(line)) is not None:
            pending_frontend = float(m.group(1))
        elif (m := _ENC_STREAM_RE.search(line)) is not None:
            encoders.append(float(m.group(1)))
            frontends.append(pending_frontend if pending_frontend is not None else 0.0)
            pending_frontend = None
        elif (m := _PARTIAL_RE.match(line)) is not None:
            if K is None and m.group(1).strip():
                K = len(encoders)  # chunks whose encoder step has completed
                first_partial_text = m.group(1)
    if K is None or K == 0:
        raise RuntimeError("no non-empty partial_text found in audiocpp --log output")

    per_chunk = [f + e for f, e in zip(frontends, encoders)]
    # steady-state per-chunk compute excludes chunk 1 (its encoder is cold-autotuned).
    steady = _median(per_chunk[1:]) if len(per_chunk) > 1 else per_chunk[0]
    cold_first = sum(per_chunk[:K])
    # warm: chunk 1's cold autotune replaced by the steady per-chunk median.
    warm_first = steady + sum(per_chunk[1:K]) if K >= 1 else steady
    return {
        "K": K,
        "warm_ms": warm_first,
        "cold_ms": cold_first,
        "steady_per_chunk_ms": steady,
        "n_chunks": len(per_chunk),
        "chunk1_cold_ms": per_chunk[0],
        "first_partial_text": first_partial_text,
    }


def measure_audiocpp(model_key: str, wavs: dict[str, Path]) -> dict:
    model = GGUF[model_key]
    if not CLI.exists():
        sys.exit(f"CLI not built: {CLI} (run the audio.cpp CUDA build first, see README)")
    if not model.exists():
        sys.exit(f"GGUF not found: {model}")
    per_clip: dict[str, dict] = {}
    for name, wav in wavs.items():
        runs = [_parse_audiocpp_run(_run_audiocpp_cli(wav, model)) for _ in range(N_RUNS)]
        per_clip[name] = {
            "warm_ms": _median([r["warm_ms"] for r in runs]),
            "cold_ms": _median([r["cold_ms"] for r in runs]),
            "K": runs[0]["K"],
            "steady_per_chunk_ms": _median([r["steady_per_chunk_ms"] for r in runs]),
            "first_partial_text": runs[0]["first_partial_text"],
        }
        c = per_clip[name]
        print(f"[audiocpp/{model_key}] {name:10s} K={c['K']} "
              f"warm={c['warm_ms']:6.1f}ms cold={c['cold_ms']:7.1f}ms "
              f"steady/chunk={c['steady_per_chunk_ms']:5.1f}ms first={c['first_partial_text']!r}")
    return {
        "engine": f"audiocpp-{model_key}",
        "per_clip": per_clip,
        "median_warm_ms": _median([c["warm_ms"] for c in per_clip.values()]),
        "median_cold_ms": _median([c["cold_ms"] for c in per_clip.values()]),
    }


# --- NeMo path (in-process; MUST run under the nemo2606 venv) ------------------
def measure_nemo(wavs: dict[str, Path]) -> dict:
    import os
    os.environ["VM_MODEL_NAME"] = NEMO_MODEL
    os.environ["VM_ATT_CONTEXT"] = NEMO_ATT_CONTEXT

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import spike_nemotron_2606 as nm  # vetted NeMo 26.06 config (build_cfg == VM shipped cfg)
    import time
    import numpy as np
    import torch
    from nemo.collections.asr.inference.streaming.framing.request import Frame

    pipeline, _cfg = nm.build_pipeline("segment")
    chunk_samples = int(round(pipeline.chunk_size_in_secs * SAMPLE_RATE))
    print(f"[nemo] model={NEMO_MODEL} att_context={NEMO_ATT_CONTEXT} "
          f"chunk_size_in_secs={pipeline.chunk_size_in_secs:.3f} ({chunk_samples} samples)")

    def _first_partial_ms(pcm: bytes) -> tuple[float, int, str]:
        audio = nm._pcm_to_float32(pcm)
        n_chunks = int(np.ceil(len(audio) / chunk_samples))
        options = nm._make_options()
        pipeline.open_session()
        cum = 0.0
        try:
            for i in range(n_chunks):
                seg = audio[i * chunk_samples:(i + 1) * chunk_samples]
                valid = seg.shape[0]
                if valid < chunk_samples:
                    padded = np.zeros(chunk_samples, dtype=np.float32)
                    padded[:valid] = seg
                    seg = padded
                frame = Frame(samples=torch.from_numpy(seg.copy()), stream_id=0,
                              is_first=(i == 0), is_last=False, length=valid,
                              options=options if i == 0 else None)
                torch.cuda.synchronize(DEVICE)
                t0 = time.perf_counter()
                out = pipeline.transcribe_step([frame])[0]
                torch.cuda.synchronize(DEVICE)
                cum += (time.perf_counter() - t0) * 1000.0
                text = out.partial_transcript.strip() or out.final_transcript.strip()
                if text:
                    return cum, i + 1, text
        finally:
            pipeline.close_session()
        return cum, n_chunks, ""

    # Warm cudnn/autotune with one discarded pass before measuring.
    _first_partial_ms(nm._read_wav_pcm(str(next(iter(wavs.values())))))

    per_clip: dict[str, dict] = {}
    for name, wav in wavs.items():
        pcm = nm._read_wav_pcm(str(wav))
        samples = [_first_partial_ms(pcm) for _ in range(N_RUNS)]
        per_clip[name] = {
            "warm_ms": _median([s[0] for s in samples]),
            "K": samples[0][1],
            "first_partial_text": samples[0][2],
        }
        c = per_clip[name]
        print(f"[nemo] {name:10s} K={c['K']} warm={c['warm_ms']:6.1f}ms "
              f"first={c['first_partial_text']!r}")
    return {
        "engine": "nemo-en-0.6b",
        "per_clip": per_clip,
        "median_warm_ms": _median([c["warm_ms"] for c in per_clip.values()]),
    }


# --- orchestration ------------------------------------------------------------
def _verdict(audiocpp: dict, nemo: dict, bar: float) -> dict:
    a = audiocpp["median_warm_ms"]
    n = nemo["median_warm_ms"]
    ratio = a / n
    return {"audiocpp_median_warm_ms": a, "nemo_median_warm_ms": n,
            "ratio": ratio, "bar": bar, "go": ratio <= bar}


def main() -> int:
    ap = argparse.ArgumentParser(description="first-partial latency head-to-head")
    ap.add_argument("--engine", choices=["audiocpp", "nemo", "both"], default="both")
    ap.add_argument("--model", choices=sorted(GGUF), default="f16",
                    help="audio.cpp GGUF (default f16, conservative; q8_0 is the faster shipped option)")
    ap.add_argument("--bar", type=float, default=1.5, help="parity bar: GO if ratio <= bar")
    ap.add_argument("--wav-dir", type=Path,
                    default=Path("/tmp/vm_latency_wavs"),
                    help="where the 16 kHz mono wavs live (converted once; stable path "
                         "shared by the audiocpp + nemo-venv passes)")
    ap.add_argument("--json", type=Path, default=None, help="dump combined results as JSON")
    args = ap.parse_args()

    wavs = ensure_wavs(args.wav_dir)

    if args.engine == "audiocpp":
        result = measure_audiocpp(args.model, wavs)
        print(json.dumps(result))
    elif args.engine == "nemo":
        result = measure_nemo(wavs)
        print(json.dumps(result))
    else:
        acpp = measure_audiocpp(args.model, wavs)
        print(f"[both] shelling out to the nemo venv for the NeMo path: {NEMO_PY}")
        proc = subprocess.run(
            [str(NEMO_PY), str(Path(__file__).resolve()),
             "--engine", "nemo", "--wav-dir", str(args.wav_dir)],
            capture_output=True, text=True,
        )
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            sys.exit(f"nemo measurement subprocess failed ({proc.returncode})")
        nemo = json.loads(proc.stdout.splitlines()[-1])
        verdict = _verdict(acpp, nemo, args.bar)
        result = {"audiocpp": acpp, "nemo": nemo, "verdict": verdict}
        print("\n=== first-partial latency head-to-head (warm, median over 6 clips) ===")
        print(f"  audio.cpp ({acpp['engine']}) : {verdict['audiocpp_median_warm_ms']:.1f} ms "
              f"(cold {acpp['median_cold_ms']:.1f} ms)")
        print(f"  NeMo      ({nemo['engine']}) : {verdict['nemo_median_warm_ms']:.1f} ms")
        print(f"  ratio audio.cpp/NeMo         : {verdict['ratio']:.3f}  (bar <= {args.bar})")
        print(f"  === VERDICT input: {'GO' if verdict['go'] else 'NO-GO'} ===")
        if args.json:
            args.json.write_text(json.dumps(result, indent=2))
            print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
