"""Design a fresh VoxCPM2 native voice (new seed) and speak the listen passage
through it, to test whether the crackling is seed-specific or model-wide.

MANUAL - not collected by pytest. Uses audiocpp_cli directly (no sidecar), the
same tool that designed the shipped seeds 101/202.
"""
import os
import subprocess
import time
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent / "3rdParty" / "audio.cpp"
CLI = Path(os.environ.get(
    "AUDIOCPP_CLI", ENGINE_DIR / "build" / "linux-cuda-release" / "bin" / "audiocpp_cli"))
MODELS_DIR = Path(os.environ.get("AUDIOCPP_MODELS_DIR", "/home/user/audiocpp-models"))
VOX = str(MODELS_DIR / "VoxCPM2-GGUF" / "voxcpm2-q8_0.gguf")
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("VM_AUDIOCPP_DEVICE", "0"))

DESC = ("A calm adult female voice, precise and even-paced, slightly synthetic, "
        "neutral affect, like a starship computer")
REF_PARAGRAPH = (
    "All primary systems are functioning within normal parameters. "
    "Long range communications are standing by. "
    "State the nature of your request, and I will route it to the appropriate subsystem."
)
LISTEN_TEXT = (
    "Captain's log, stardate four seven six three point nine. We have entered the "
    "Veridian system after a six day crossing at warp seven, and the ship has settled "
    "into the easy rhythm of survey work. The science teams are mapping the second "
    "planet's magnetosphere while engineering runs a full diagnostic on the port "
    "nacelle. For now the bridge is quiet, the stars are steady, and the ship feels "
    "very much like home."
)

SEEDS = [404, 505]


def run(args, log):
    t0 = time.perf_counter()
    with open(log, "w") as f:
        rc = subprocess.run([str(CLI), *args], cwd=ENGINE_DIR, env=ENV,
                            stdout=f, stderr=subprocess.STDOUT).returncode
    return rc, time.perf_counter() - t0


def main():
    for seed in SEEDS:
        design = f"/tmp/vox_new_s{seed}.wav"
        rc, wall = run([
            "--task", "tts", "--family", "voxcpm2", "--model", VOX,
            "--backend", "cuda", "--seed", str(seed),
            "--text", f"({DESC}){REF_PARAGRAPH}",
            "--out", design,
        ], f"/tmp/vox_new_s{seed}.design.log")
        print(f"design s{seed}: rc={rc} wall={wall:.1f}s -> {design}")
        if rc != 0 or not Path(design).exists():
            continue

        out = f"/tmp/voxcpm2_newvoice_s{seed}_test.wav"
        rc, wall = run([
            "--task", "tts", "--mode", "streaming", "--family", "voxcpm2",
            "--model", VOX, "--backend", "cuda",
            "--text", LISTEN_TEXT,
            "--voice-ref", design, "--reference-text", REF_PARAGRAPH,
            "--request-option", "retry_badcase=false",
            "--out", out,
        ], f"/tmp/vox_new_s{seed}.test.log")
        print(f"speak s{seed}: rc={rc} wall={wall:.1f}s -> {out}")


if __name__ == "__main__":
    main()
