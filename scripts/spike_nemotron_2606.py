#!/usr/bin/env python3
"""RE-GATE de-risking SPIKE harness - THROWAWAY.

Re-tests the cache-aware FastConformer-RNNT finalization crash on NeMo 26.06
(nemo_toolkit 3.1.0 @ the 26.06 commit) with the NEW model
``nvidia/nemotron-3.5-asr-streaming-0.6b`` via the new
``nemo.collections.asr.inference`` Pipeline API.

The bug being re-tested (was a NO-GO on NeMo 2.7.2/2.7.3):
    ValueError: tokens, timesteps and confidences must have the same length
    raised in bpe_decoder.{group_tokens_into_words,group_tokens_into_segment}
    at FINALIZATION (end-of-utterance). Interim transcripts were CORRECT; the
    final crashed so no final was ever emitted.

DECISIVE QUESTION: does finalization now emit a CLEAN final transcript
containing the boundary words {data, complete, this facility, not authorized}
WITHOUT that ValueError?

This is a copy of scripts/spike_nemotron.py adapted to the NeMo 26.06 schema.
All NeMo API usage was verified against the INSTALLED source under
``/home/user/nemo2606-spike-venv/.../nemo/collections/asr/inference/``.

Run with:
    CUDA_DEVICE_ORDER=PCI_BUS_ID TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
        /home/user/nemo2606-spike-venv/bin/python \
        scripts/spike_nemotron_2606.py --check vram
    ... --check accuracy [--granularity segment|word]
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
import wave

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

# Re-gate target: the VM_AUDIOCPP_DEVICE GPU (default 0). Model = nemotron 3.5.
DEVICE_STR = "cuda"
DEVICE_ID = int(os.environ.get("VM_AUDIOCPP_DEVICE", "0"))
# Default to the 3.5 multilingual model; override to the English-only model
# (nvidia/nemotron-speech-streaming-en-0.6b) via VM_MODEL_NAME.
MODEL_NAME = os.environ.get("VM_MODEL_NAME", "nvidia/nemotron-3.5-asr-streaming-0.6b")
SAMPLE_RATE = 16000

CLIPS = {
    "access": "/tmp/accessinglibrarycomputerdata_clean.wav",                 # -> data
    "longrange": "/tmp/longrangescanplanetcomplete_clean.wav",               # -> complete
    "notauth": "/tmp/youarenotauthorisedtoaccessthisfacility_clean.wav",     # -> this facility, not authoris/zed
    "donot": "/tmp/donotaddressthisunitinthatmanner_clean.wav",              # boundary 'manner'
    "direction": "/tmp/directionunclearpleaserepeat_clean.wav",              # bonus coverage
    "incoming": "/tmp/incomingtransmission_clean.wav",                       # bonus coverage
}
REGRESSION_TOKENS = ["data", "complete", "this facility", "not authorized"]
# Spelling-tolerant regex patterns for the 4 gate words (audio uses British
# "authoriSED"; the token list uses American "authoriZED" — accept either).
REGRESSION_PATTERNS = {
    "data": r"\bdata\b",
    "complete": r"\bcomplete\b",
    "this facility": r"\bthis facility\b",
    "not authorized": r"\bnot authori[sz]ed\b",
}

FRAME_MS = 32
TRAIL_SILENCE_S = 4.0
N_RUNS = 3

# Filled by build_pipeline() once the model is loaded so feed loops can set a
# valid language_code on prompt-enabled (multilingual) models.
PROMPT_LANG_CODE: str | None = None


def _pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def _read_wav_pcm(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == SAMPLE_RATE and wf.getnchannels() == 1 and wf.getsampwidth() == 2, (
            f"{path}: expected 16 kHz mono 16-bit WAV"
        )
        return wf.readframes(wf.getnframes())


def build_cfg(granularity: str = "segment") -> "OmegaConf":
    """NeMo 26.06 cache-aware RNNT pipeline cfg.

    Mirrors examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml.
    Key 26.06 deltas vs the 2.7.2 harness:
      - NO top-level enable_pnc (removed upstream, PR #15591)
      - return_tail_result is top-level (was under streaming/confidence churn)
      - decoding.strategy="greedy_batch" with a structured greedy sub-block
        (use_cuda_graph_decoder, enable_per_stream_biasing, max_symbols,
        ngram_lm_model, boosting_tree, ...)
      - confidence block rewritten: exclude_blank=true, method_cfg.name=entropy
      - itn block: input_case/batch_size/n_jobs/left_padding_size/...
      - lang: en (3.5 is multilingual w/ language-ID prompt)
    """
    cfg = {
        "asr": {
            "model_name": MODEL_NAME,
            "device": DEVICE_STR,
            "device_id": DEVICE_ID,
            "compute_dtype": "bfloat16",
            "use_amp": False,
            "decoding": {
                "strategy": "greedy_batch",
                "preserve_alignments": False,
                "fused_batch_size": -1,
                "greedy": {
                    "use_cuda_graph_decoder": False,
                    "enable_per_stream_biasing": True,
                    "preserve_frame_confidence": False,
                    "max_symbols": 10,
                    "ngram_lm_model": None,
                    "ngram_lm_alpha": 0.0,
                    "boosting_tree": {
                        "model_path": None,
                        "key_phrases_file": None,
                        "key_phrases_list": None,
                        "key_phrase_items_list": None,
                        "source_lang": "en",
                    },
                    "boosting_tree_alpha": 0.0,
                },
            },
        },
        "itn": {
            "input_case": "lower_cased",
            "whitelist": None,
            "overwrite_cache": False,
            "max_number_of_permutations_per_split": 729,
            "left_padding_size": 4,
            "batch_size": 32,
            "n_jobs": 16,
        },
        "confidence": {
            "exclude_blank": True,
            "aggregation": "mean",
            "method_cfg": {
                "name": "entropy",
                "entropy_type": "tsallis",
                "alpha": 0.5,
                "entropy_norm": "exp",
            },
        },
        "endpointing": {
            "stop_history_eou": 800,
            "residue_tokens_at_end": 2,
        },
        "streaming": {
            "sample_rate": 16000,
            "batch_size": 1,
            "word_boundary_tolerance": 4,
            # 3.5 model's supported look-aheads are [[56,3],[56,0],[56,6],[56,13]]
            # (left context 56, not 70 as on the old 0.6b model). Use the max-lookahead
            # supported variant [56,13]. Override via VM_ATT_CONTEXT="56,13".
            "att_context_size": [
                int(x) for x in os.environ.get("VM_ATT_CONTEXT", "56,13").split(",")
            ],
            "use_cache": True,
            "use_feat_cache": True,
            "chunk_size_in_secs": None,
            "request_type": "frame",
            "num_slots": 1,
        },
        "matmul_precision": "high",
        "log_level": 30,
        "pipeline_type": "cache_aware",
        "asr_decoding_type": "rnnt",
        "audio_file": None,
        "output_filename": None,
        "output_dir": None,
        "enable_itn": False,
        "enable_nmt": False,
        "asr_output_granularity": granularity,
        "cache_dir": None,
        "lang": "en",
        "return_tail_result": False,
        "calculate_wer": False,
        "calculate_bleu": False,
        "warmup_steps": 0,
        "run_steps": 1,
    }
    return OmegaConf.create(cfg)


def _resolve_prompt_lang(pipeline) -> str | None:
    """If the model is prompt-enabled (multilingual), pick a valid language_code.

    create_state() raises ValueError if a prompt-enabled model gets no
    language_code. Default in NeMo is "en-US"; verify it exists in the model's
    prompt dictionary, else fall back to the first English-looking key.
    """
    if not getattr(pipeline, "prompt_enabled", False):
        return None
    prompt_dict = getattr(pipeline, "_prompt_config", {}).get("prompt_dict", {})
    keys = list(prompt_dict.keys()) if prompt_dict else []
    print(f"[prompt] prompt-enabled model. available language codes: {keys}")
    for cand in ("en-US", "en-us", "en", "en-GB"):
        if cand in prompt_dict:
            return cand
    for k in keys:
        if str(k).lower().startswith("en"):
            return k
    return keys[0] if keys else "en-US"


def build_pipeline(granularity: str = "segment"):
    from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

    global PROMPT_LANG_CODE
    torch.cuda.set_device(DEVICE_ID)
    cfg = build_cfg(granularity)
    pipeline = PipelineBuilder.build_pipeline(cfg)
    PROMPT_LANG_CODE = _resolve_prompt_lang(pipeline)
    print(f"[prompt] using language_code={PROMPT_LANG_CODE!r}")
    return pipeline, cfg


def _make_options():
    from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions

    if PROMPT_LANG_CODE is not None:
        return ASRRequestOptions(language_code=PROMPT_LANG_CODE)
    return ASRRequestOptions()


def _mib(n_bytes: int) -> float:
    return n_bytes / (1024 * 1024)


def check_vram() -> None:
    torch.cuda.set_device(DEVICE_ID)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE_ID)

    before = torch.cuda.memory_allocated(DEVICE_ID)
    print(f"[vram] before build: allocated={_mib(before):.1f} MiB")

    pipeline, cfg = build_pipeline()
    after_build = torch.cuda.memory_allocated(DEVICE_ID)
    print(f"[vram] after  build: allocated={_mib(after_build):.1f} MiB "
          f"(delta={_mib(after_build - before):.1f} MiB) — ONE model copy")

    from nemo.collections.asr.inference.streaming.framing.request import Frame

    pipeline.open_session()
    pcm = _read_wav_pcm(CLIPS["access"])
    chunk_secs = pipeline.chunk_size_in_secs
    chunk_samples = int(round(chunk_secs * SAMPLE_RATE))
    audio = _pcm_to_float32(pcm)
    first = torch.from_numpy(audio[:chunk_samples].copy())
    frame = Frame(samples=first, stream_id=0, is_first=True, is_last=False,
                  length=first.shape[0], options=_make_options())
    pipeline.transcribe_step([frame])
    pipeline.close_session()

    after_step = torch.cuda.memory_allocated(DEVICE_ID)
    peak = torch.cuda.max_memory_allocated(DEVICE_ID)
    print(f"[vram] after  step : allocated={_mib(after_step):.1f} MiB")
    print(f"[vram] peak        : {_mib(peak):.1f} MiB")
    print(f"[vram] chunk_size_in_secs={chunk_secs:.3f}  ({chunk_samples} samples/chunk)")
    print(f"[vram] att_context_size={pipeline.asr_model.get_att_context_size()}")
    print(f"[vram] subsampling_factor={pipeline.subsampling_factor}  "
          f"window_stride={pipeline.window_stride}  "
          f"model_stride_ms={pipeline.model_stride_in_milliseconds}")
    print("[vram] RESULT: single-model footprint built AND ran a transcribe step.")


def feed_clip(pipeline, pcm: bytes, *, verbose: bool = False) -> tuple[str, list[tuple]]:
    from nemo.collections.asr.inference.streaming.framing.request import Frame

    chunk_samples = int(round(pipeline.chunk_size_in_secs * SAMPLE_RATE))

    body = pcm + (b"\x00\x00" * int(SAMPLE_RATE * TRAIL_SILENCE_S))
    audio = _pcm_to_float32(body)
    n_chunks = int(np.ceil(len(audio) / chunk_samples))

    pipeline.open_session()
    accumulated = ""
    segment_index = 0
    events: list[tuple] = []
    options = _make_options()

    for i in range(n_chunks):
        start = i * chunk_samples
        end = min(start + chunk_samples, len(audio))
        seg = audio[start:end]
        valid = seg.shape[0]
        is_last = i == n_chunks - 1
        if valid < chunk_samples:
            padded = np.zeros(chunk_samples, dtype=np.float32)
            padded[:valid] = seg
            seg = padded
        samples = torch.from_numpy(seg.copy())
        frame = Frame(
            samples=samples,
            stream_id=0,
            is_first=(i == 0),
            is_last=is_last,
            length=valid,
            options=options if i == 0 else None,
        )
        outputs = pipeline.transcribe_step([frame])
        out = outputs[0]
        if out.final_transcript.strip():
            accumulated += out.final_transcript
            events.append(("final", segment_index, out.final_transcript.strip()))
            if verbose:
                print(f"  [chunk {i:3d}] FINAL  seg={segment_index} "
                      f"text={out.final_transcript.strip()!r}")
            segment_index += 1
        elif out.partial_transcript.strip():
            events.append(("interim", segment_index, out.partial_transcript.strip()))
            if verbose:
                print(f"  [chunk {i:3d}] interim seg={segment_index} "
                      f"text={out.partial_transcript.strip()!r}")

    pipeline.close_session()
    return accumulated.strip(), events


def check_accuracy(granularity: str = "segment") -> None:
    pipeline, cfg = build_pipeline(granularity)
    print(f"[accuracy] granularity={granularity}  "
          f"chunk_size_in_secs={pipeline.chunk_size_in_secs:.3f}  "
          f"att_context_size={pipeline.asr_model.get_att_context_size()}")
    print(f"[accuracy] stop_history_eou(ms)={pipeline.stop_history_eou_in_milliseconds}  "
          f"model_stride_ms={pipeline.model_stride_in_milliseconds}")

    all_runs: dict[str, list[str]] = {}
    combined_lower = ""
    seg_monotonic_ok = True

    for name, path in CLIPS.items():
        pcm = _read_wav_pcm(path)
        runs: list[str] = []
        print(f"\n=== clip '{name}' ({path}) ===")
        for r in range(N_RUNS):
            text, events = feed_clip(pipeline, pcm, verbose=(r == 0))
            runs.append(text)
            finals = [e for e in events if e[0] == "final"]
            segs = [e[1] for e in finals]
            mono = segs == sorted(segs) and segs == list(range(len(segs)))
            seg_monotonic_ok = seg_monotonic_ok and mono
            print(f"  run {r}: finals={len(finals)} segs={segs} monotonic={mono} "
                  f"text={text!r}")
        all_runs[name] = runs
        combined_lower += " " + (runs[0].lower() if runs else "")

    print("\n=== RAW per-clip final transcript (run 0, verbatim) ===")
    for name, runs in all_runs.items():
        print(f"  {name:10s}: {runs[0]!r}")

    print("\n=== determinism (Check 3) — REAL requires non-empty AND identical ===")
    determinism_ok = True
    for name, runs in all_runs.items():
        identical = len(set(runs)) == 1
        nonempty = bool(runs[0].strip())
        real = identical and nonempty
        determinism_ok = determinism_ok and real
        kind = "REAL" if real else ("TRIVIAL-EMPTY" if identical and not nonempty else "NONDETERMINISTIC")
        print(f"  clip '{name}': identical={identical} nonempty={nonempty} -> {kind}")

    print("\n=== regression tokens (Check 3) — spelling-tolerant, per expected clip ===")
    import re as _re
    # Which clip each gate word is expected to appear in.
    expected_clip = {
        "data": "access",
        "complete": "longrange",
        "this facility": "notauth",
        "not authorized": "notauth",
    }
    found = {}
    for tok in REGRESSION_TOKENS:
        pat = REGRESSION_PATTERNS[tok]
        clip = expected_clip.get(tok)
        clip_text = (all_runs.get(clip, [""])[0] or "").lower() if clip else combined_lower
        hit = bool(_re.search(pat, clip_text))
        found[tok] = hit
        print(f"  token {tok!r} (pat={pat}) in clip '{clip}': "
              f"{'FOUND' if hit else 'MISSING'}")
    words_ok = all(found.values())

    print("\n=== EoU -> protocol mapping (Check 4) ===")
    print(f"  segment_index monotonic & per-utterance: {seg_monotonic_ok}")

    print("\n=== SUMMARY ===")
    print(f"  granularity        : {granularity}")
    print(f"  Check 3 words      : {'PASS' if words_ok else 'FAIL'}")
    print(f"  Check 3 determinism: {'PASS' if determinism_ok else 'FAIL'}")
    print(f"  Check 4 EoU mapping: {'PASS' if seg_monotonic_ok else 'FAIL'}")
    print(f"  combined transcript: {combined_lower.strip()!r}")


def offline_probe() -> None:
    """Secondary probe: prove the checkpoint itself decodes via the high-level API."""
    from nemo.collections.asr.models import ASRModel

    torch.cuda.set_device(DEVICE_ID)
    m = ASRModel.from_pretrained(MODEL_NAME, map_location=f"cuda:{DEVICE_ID}")
    # Prompt-enabled multilingual model: must pass a language prompt.
    try:
        out = m.transcribe([CLIPS["access"], CLIPS["donot"]], target_lang="en-US")
    except TypeError:
        out = m.transcribe([CLIPS["access"], CLIPS["donot"]], prompt={"target_lang": "en-US"})
    print("[offline] transcribe output:")
    for clip, o in zip(CLIPS, out):
        txt = getattr(o, "text", o)
        print(f"  {clip}: {txt!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Nemotron 3.5 cache-aware re-gate spike (NeMo 26.06)")
    ap.add_argument("--check", choices=["vram", "accuracy", "offline"], required=True)
    ap.add_argument("--granularity", choices=["segment", "word"], default="segment")
    args = ap.parse_args()
    try:
        if args.check == "vram":
            check_vram()
        elif args.check == "accuracy":
            check_accuracy(args.granularity)
        else:
            offline_probe()
    except Exception:
        print("\n!!! SPIKE CAUGHT EXCEPTION — full traceback below !!!", file=sys.stderr)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    sys.exit(main())
