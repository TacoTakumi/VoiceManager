"""Build deterministic audio clips for the speculative-endpoint verification.

Splits a single real-speech utterance ("accessing library computer data") into
two halves at its natural mid-sentence gap, then reassembles three clips whose
baked-in silence exercises the nemotron endpoint-hold state machine when streamed
at real-time pace (see tests/test_mic_stt.py --file):

  clip_confirm.wav   full utterance + long trailing silence
                     -> EoU fires, no resume -> endpoint then confirmed final

  clip_stitch.wav    A + short gap (> stop_history_eou, < hold window) + B + tail
                     -> EoU fires on the gap, B resumes inside the hold window
                        -> one stitched final under A's segment_index

  clip_twofinal.wav  A + long gap (> hold window) + B + tail
                     -> A confirms, B opens a new segment -> two finals (negative
                        case)

Gaps are chosen relative to the server's defaults (VM_NEMOTRON_EOU_STOP_HISTORY_MS
=800, VM_STT_ENDPOINT_HOLD_MS=1000). Regenerate any time; output is deterministic.

Usage:
    venv/bin/python tests/build_endpoint_clips.py [OUT_DIR]
"""
import sys
import wave

import librosa
import numpy as np

SAMPLE_RATE = 16000
SRC = "accessinglibrarycomputerdata_clean.mp3"

# Gap durations (seconds). Server defaults: EoU stop-history 0.8s, hold 1.0s.
STITCH_GAP = 0.90      # > 0.8 (EoU fires) but < 1.0 (resume lands in hold window)
TWOFINAL_GAP = 1.60    # > 1.0 hold window -> A confirms before B arrives
TAIL = 2.00            # trailing silence long enough to confirm the final held frag
LEAD = 0.20            # small lead-in so the first frame isn't clipped


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.int16)


def _write_wav(path: str, pcm: np.ndarray) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    print(f"  wrote {path}  ({len(pcm) / SAMPLE_RATE:.2f}s)")


def _split_halves(pcm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split at the quietest short window in the utterance's middle third."""
    win = int(0.05 * SAMPLE_RATE)  # 50 ms
    lo, hi = int(0.30 * len(pcm)), int(0.70 * len(pcm))
    energy = np.abs(pcm.astype(np.float32))
    best_i, best_e = lo, float("inf")
    for i in range(lo, hi - win):
        e = energy[i:i + win].mean()
        if e < best_e:
            best_e, best_i = e, i
    split = best_i + win // 2
    return pcm[:split], pcm[split:]


def main() -> None:
    out_dir = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "."
    y, _ = librosa.load(SRC, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    full = (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)
    a, b = _split_halves(full)
    print(f"Split: A={len(a) / SAMPLE_RATE:.2f}s  B={len(b) / SAMPLE_RATE:.2f}s")

    lead = _silence(LEAD)
    tail = _silence(TAIL)

    _write_wav(f"{out_dir}/clip_confirm.wav",
               np.concatenate([lead, full, tail]))
    _write_wav(f"{out_dir}/clip_stitch.wav",
               np.concatenate([lead, a, _silence(STITCH_GAP), b, tail]))
    _write_wav(f"{out_dir}/clip_twofinal.wav",
               np.concatenate([lead, a, _silence(TWOFINAL_GAP), b, tail]))


if __name__ == "__main__":
    main()
