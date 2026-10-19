#!/usr/bin/env python3
"""Design a new Breeze-TTS voice from a written prompt file, then audition it.

MANUAL script - never collected by pytest (scripts/ is outside tests/). It talks
to audiocpp_cli directly, so it needs no running VoiceManager server and spawns
no sidecar.

Two legs per seed, both offline on VM_AUDIOCPP_DEVICE:

  1. design  - `--task tts` with the prompt's voice description as `instruction`
               and no reference audio. The spoken "Reference line" becomes the
               new voice's reference WAV.
  2. audition - `--task clon` with that WAV plus its exact transcript, speaking
               the prompt's "Audition" text. This is the leg that sounds like
               what the server will produce, because the server clones the same
               way (audiocpp_tts.py, ref_text mandatory under breeze).

Design is stochastic: the same description at different seeds gives different
voices. Ask for a few seeds and pick by ear.

Usage:

    ./venv/bin/python scripts/design_voice.py voices/prompts/captain.md
    ./venv/bin/python scripts/design_voice.py voices/prompts/captain.md --seeds 311,312,313
    ./venv/bin/python scripts/design_voice.py voices/prompts/captain.md --register --seed 312

Nothing is added to voices/designed/refs.json until you pass --register, so a
run is always safe to repeat. See voices/prompts/README.md for the file format.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
ENGINE_DIR = REPO / "3rdParty" / "audio.cpp"
DEFAULT_CLI = Path(os.environ.get(
    "AUDIOCPP_CLI", ENGINE_DIR / "build" / "linux-cuda-release" / "bin" / "audiocpp_cli"))
REFS_JSON = REPO / "voices" / "designed" / "refs.json"

#: A reference clip well outside this range tends to clone poorly. The shipped
#: presets sit at 7.8 to 10.2 s. Advisory only - nothing here rejects a clip.
GOOD_REF_SECONDS = (6.0, 14.0)

DEFAULT_AUDITION = (
    "The long range sensors have finished their sweep of the outer belt and "
    "found nothing unusual. Everything else can wait for the morning briefing."
)


def read_env(key: str, default: str = "") -> str:
    """Read one VM_* value out of .env (the server's own config), else default."""
    env_path = REPO / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{key}=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip()
    return default


def parse_prompt(path: Path) -> dict:
    """Parse a prompt file: optional YAML front matter plus markdown sections.

    Sections are '# Heading' blocks; their headings are lowercased and used as
    keys, so '# Reference line' becomes 'reference line'. Front-matter keys win
    over nothing - the two namespaces do not overlap.
    """
    text = path.read_text(encoding="utf-8")
    meta: dict = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end == -1:
            raise SystemExit(f"{path}: front matter opened with --- but never closed")
        meta = yaml.safe_load(text[3:end]) or {}
        text = text[end + 4:]

    sections: dict[str, str] = {}
    current = None
    buf: list[str] = []
    for line in text.splitlines():
        heading = re.match(r"^#{1,6}\s+(.*\S)\s*$", line)
        if heading:
            if current:
                sections[current] = "\n".join(buf).strip()
            current = heading.group(1).strip().lower()
            buf = []
        elif current:
            buf.append(line)
    if current:
        sections[current] = "\n".join(buf).strip()

    def section(*names: str) -> str:
        for n in names:
            if sections.get(n):
                return " ".join(sections[n].split())
        return ""

    name = meta.get("name") or path.stem
    description = section("voice", "description", "voice description")
    reference = section("reference line", "reference", "reference text")
    audition = section("audition", "audition text", "test line")

    if not description:
        raise SystemExit(f"{path}: no '# Voice' section - the voice description is required")
    if not reference:
        raise SystemExit(
            f"{path}: no '# Reference line' section. Breeze clones from a clip PLUS its "
            "exact transcript, so the reference line is mandatory.")

    return {
        "name": str(name),
        "description": description,
        "reference": reference,
        "audition": audition or DEFAULT_AUDITION,
        "seeds": meta.get("seeds") or [],
        "options": meta.get("options") or {},
        "notes": section("notes"),
    }


def wav_info(path: Path) -> tuple[int, float]:
    with wave.open(str(path)) as w:
        return w.getframerate(), w.getnframes() / w.getframerate()


def run_cli(cli: Path, args: list[str], gpu: str, log: Path) -> tuple[int, float]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    t0 = time.perf_counter()
    with open(log, "w") as fh:
        rc = subprocess.run([str(cli), *args], cwd=ENGINE_DIR, env=env,
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    return rc, time.perf_counter() - t0


def design_one(cfg, seed, out_dir, cli, model, gpu) -> dict | None:
    stem = f"{cfg['name']}_s{seed}"
    ref_wav = out_dir / f"{stem}.wav"
    args = [
        "--task", "tts", "--family", "breeze_tts", "--model", model,
        "--backend", "cuda", "--text", cfg["reference"],
        "--request-option", f"instruction={cfg['description']}",
        "--request-option", f"seed={seed}",
        "--metrics", "--out", str(ref_wav),
    ]
    for k, v in cfg["options"].items():
        args += ["--request-option", f"{k}={v}"]
    rc, wall = run_cli(cli, args, gpu, out_dir / f"{stem}.design.log")
    if rc != 0 or not ref_wav.is_file():
        print(f"  seed {seed}: DESIGN FAILED (rc={rc}), see {stem}.design.log")
        return None
    rate, dur = wav_info(ref_wav)
    flag = "" if GOOD_REF_SECONDS[0] <= dur <= GOOD_REF_SECONDS[1] else "  <-- length outside 6-14 s"
    print(f"  seed {seed}: reference {dur:.2f}s @ {rate} Hz in {wall:.1f}s{flag}")

    aud_wav = out_dir / f"{stem}_audition.wav"
    args = [
        "--task", "clon", "--family", "breeze_tts", "--model", model,
        "--backend", "cuda", "--text", cfg["audition"],
        "--voice-ref", str(ref_wav), "--reference-text", cfg["reference"],
        "--metrics", "--out", str(aud_wav),
    ]
    rc, wall = run_cli(cli, args, gpu, out_dir / f"{stem}.audition.log")
    if rc != 0 or not aud_wav.is_file():
        print(f"  seed {seed}: AUDITION FAILED (rc={rc}), see {stem}.audition.log")
        return {"seed": seed, "ref_wav": ref_wav, "ref_seconds": dur, "audition": None}
    _, adur = wav_info(aud_wav)
    print(f"  seed {seed}: audition  {adur:.2f}s in {wall:.1f}s -> {aud_wav.name}")
    return {"seed": seed, "ref_wav": ref_wav, "ref_seconds": dur, "audition": aud_wav}


def register(cfg, seed, out_dir) -> None:
    """Copy the chosen reference into voices/designed/ and add its refs.json entry."""
    preset = cfg["name"]
    src = out_dir / f"{preset}_s{seed}.wav"
    if not src.is_file():
        raise SystemExit(f"no designed reference for seed {seed}: {src} is absent")

    refs = json.loads(REFS_JSON.read_text(encoding="utf-8"))
    if preset in refs:
        raise SystemExit(
            f"refs.json already has a preset named {preset!r}. Rename it in the prompt "
            "file, or remove the existing entry by hand if you mean to replace it.")

    dest = REFS_JSON.parent / f"{preset}.wav"
    if dest.exists():
        raise SystemExit(f"{dest} already exists - refusing to overwrite it")
    shutil.copy2(src, dest)
    rate, _ = wav_info(dest)

    refs[preset] = {
        "file": dest.name,
        "ref_text": cfg["reference"],
        "origin": f"Breeze-TTS 2 q8_0 voice design, seed {seed}, instruction: "
                  f"'{cfg['description']}'",
        "sample_rate": rate,
    }
    REFS_JSON.write_text(json.dumps(refs, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    print(f"\nRegistered {preset!r} (seed {seed}) -> {dest}")
    print(f"Updated {REFS_JSON}")
    print("Restart the server to load it; startup validates every preset and fails loudly.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", type=Path, help="voice prompt file (see voices/prompts/README.md)")
    ap.add_argument("--seeds", help="comma-separated seeds, overriding the prompt file")
    ap.add_argument("--out-dir", type=Path, help="default: voice_design/<name>/")
    ap.add_argument("--register", action="store_true",
                    help="add the chosen seed to voices/designed/refs.json (needs --seed)")
    ap.add_argument("--seed", type=int, help="which seed to register")
    ap.add_argument("--gpu", default=read_env("VM_AUDIOCPP_DEVICE", "0"))
    ap.add_argument("--cli", type=Path, default=DEFAULT_CLI)
    ap.add_argument("--model", default=read_env(
        "VM_AUDIOCPP_TTS_MODEL",
        str(Path(os.environ.get("AUDIOCPP_MODELS_DIR", "/home/user/audiocpp-models"))
            / "Breeze-TTS-2-GGUF" / "breeze-tts-2-q8_0.gguf")))
    args = ap.parse_args()

    if not args.prompt.is_file():
        raise SystemExit(f"prompt file not found: {args.prompt}")
    cfg = parse_prompt(args.prompt)
    out_dir = args.out_dir or (REPO / "voice_design" / cfg["name"])

    if args.register:
        if args.seed is None:
            raise SystemExit("--register needs --seed to say which take to keep")
        register(cfg, args.seed, out_dir)
        return 0

    if not args.cli.is_file():
        raise SystemExit(f"audiocpp_cli not found: {args.cli}")
    if not Path(args.model).is_file():
        raise SystemExit(f"breeze model not found: {args.model}")

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [int(s) for s in cfg["seeds"]] or [101])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"voice   : {cfg['name']}")
    print(f"describe: {cfg['description']}")
    print(f"ref line: {cfg['reference']}")
    print(f"seeds   : {', '.join(str(s) for s in seeds)}   gpu {args.gpu}")
    print(f"output  : {out_dir}\n")

    made = [r for r in (design_one(cfg, s, out_dir, args.cli, args.model, args.gpu)
                        for s in seeds) if r]
    if not made:
        print("\nNo takes were produced.")
        return 1

    print(f"\nListen to the *_audition.wav files in {out_dir}, then keep one with:\n")
    print(f"    ./venv/bin/python scripts/design_voice.py {args.prompt} "
          f"--register --seed <seed>\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
