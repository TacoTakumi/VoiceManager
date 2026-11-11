#!/usr/bin/env python3
"""Wake word probe - runs WAV files through the server's wake spotter, with the real model.

It loads the sherpa-onnx keyword spotting model the way the server does (the
SherpaWakeSpotter class), opens one stream per file with the given phrases, feeds
the file in 100 ms frames as a WebSocket client would send it, and prints the phrase
the spotter heard and when, or that it heard none. No server runs, and no GPU is
used. It needs the wake engine's packages (requirements-wake.txt) and the model that
`python setup_models.py wake` downloads.

    venv/bin/python scripts/wake_probe.py --model-dir /home/you/models/kws \\
        --phrase "hey computer" --phrase "hey hermes" hey-computer.wav other-speech.wav

A WAV file of any sample rate and channel count is taken (16-bit PCM); it is mixed
to mono and resampled to 16 kHz. Each file gets one second of silence before and
after it. Use it after a change to the spotter code, a new model, or to try a phrase
and a threshold before they go into VM_WAKE_PHRASES and VM_WAKE_THRESHOLD.

Pass criteria for a check of the spotter code: a clip of a phrase prints that phrase,
and a clip of other speech prints no detection.
"""
import argparse
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from voicemanager.wake import (  # noqa: E402
    SAMPLE_RATE, SherpaWakeSpotter, WakeModelError, WakePhraseError, normalize_phrases,
)

FRAME_SAMPLES = SAMPLE_RATE // 10
PAD_S = 1.0


def load_pcm(path: str) -> bytes:
    """The file as 16 kHz mono 16-bit PCM, with silence before and after it."""
    with wave.open(path, "rb") as wf:
        if wf.getsampwidth() != 2:
            sys.exit(f"{path}: expected 16-bit PCM, got {8 * wf.getsampwidth()}-bit")
        rate, channels = wf.getframerate(), wf.getnchannels()
        samples = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float64)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != SAMPLE_RATE:
        at = np.arange(0, len(samples), rate / SAMPLE_RATE)
        samples = np.interp(at, np.arange(len(samples)), samples)
    pad = np.zeros(int(PAD_S * SAMPLE_RATE))
    return np.concatenate([pad, samples, pad]).round().astype("<i2").tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wavs", nargs="+", metavar="WAV", help="16-bit PCM WAV files")
    parser.add_argument("--model-dir", default=os.environ.get("VM_WAKE_MODEL_DIR", ""),
                        help="the keyword spotting model directory (default: VM_WAKE_MODEL_DIR)")
    parser.add_argument("--phrase", action="append", dest="phrases", metavar="TEXT",
                        help="a wake phrase; repeat for more (default: hey computer)")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="as VM_WAKE_THRESHOLD (default: 0.15)")
    parser.add_argument("--score", type=float, default=1.5,
                        help="as VM_WAKE_SCORE (default: 1.5)")
    args = parser.parse_args()
    if not args.model_dir:
        sys.exit("give --model-dir, or set VM_WAKE_MODEL_DIR")

    try:
        phrases = normalize_phrases(args.phrases or ["hey computer"])
        spotter = SherpaWakeSpotter(
            args.model_dir, default_phrases=phrases, threshold=args.threshold, score=args.score)
    except (WakeModelError, WakePhraseError) as exc:
        sys.exit(str(exc))
    print(f"phrases={phrases} threshold={args.threshold} score={args.score}")
    try:
        for path in args.wavs:
            pcm = load_pcm(path)
            stream = spotter.open_stream(phrases)
            hits = []
            step = FRAME_SAMPLES * 2
            for start in range(0, len(pcm), step):
                # On the spotter's own thread, as the server calls it.
                phrase = spotter.executor.submit(stream.feed, pcm[start:start + step]).result()
                if phrase is not None:
                    hits.append((phrase, start / 2 / SAMPLE_RATE - PAD_S))
            if hits:
                heard = ", ".join(f"{phrase!r} at {at:.1f} s" for phrase, at in hits)
                print(f"{path}: detected {heard}")
            else:
                print(f"{path}: no detection")
    finally:
        spotter.close()


if __name__ == "__main__":
    main()
