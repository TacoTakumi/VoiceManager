"""Download and cache models for VoiceManager.

Usage::

    python setup_models.py              # only the engines configured in .env / the defaults
    python setup_models.py audiocpp-stt # only the audio.cpp STT GGUF (VM_AUDIOCPP_MODEL)
    python setup_models.py audiocpp-tts # only the audio.cpp TTS GGUF (VM_AUDIOCPP_TTS_MODEL)
    python setup_models.py whisper      # only Whisper (faster-whisper, VM_WHISPER_MODEL)
    python setup_models.py openvoice    # only OpenVoice v2 checkpoints + NLTK
    python setup_models.py vibevoice    # only VibeVoice model + voice presets
    python setup_models.py whisper vibevoice  # multiple targets

requirements.txt covers only the default audiocpp engines. When an engine needs
packages from its requirements-<engine>.txt that are missing, this script asks
before it installs them, and only prints the pip command when stdin is not a TTY.
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

from huggingface_hub import hf_hub_download, snapshot_download

from voicemanager.config import get_settings


VOICES_DIR = "voices/streaming_model"
AUDIOCPP_GGUF_REPO = "audio-cpp/audio.cpp-gguf"

# Setup target for each configured engine. None means the engine has no setup step.
STT_TARGETS = {"audiocpp": "audiocpp-stt", "whisper": "whisper", "nemotron": None, "qwen": None}
TTS_TARGETS = {"audiocpp": "audiocpp-tts", "openvoice": "openvoice", "vibevoice": "vibevoice"}
# Engine whose requirements a named target needs (the audiocpp targets need only the core).
TARGET_ENGINES = {"whisper": "whisper", "openvoice": "openvoice", "vibevoice": "vibevoice"}

# Modules each non-core engine imports, and the file that installs them.
ENGINE_REQUIREMENTS = {
    "whisper": (("faster_whisper",), "requirements-whisper.txt"),
    "openvoice": (("melo", "nltk"), "requirements-openvoice.txt"),
    "vibevoice": (("vibevoice", "diffusers"), "requirements-vibevoice.txt"),
    "nemotron": (("nemo",), "requirements-nemotron.txt"),
    "qwen": (("qwen_asr", "vllm"), "requirements-qwen.txt"),
}


def ensure_requirements(engine: str) -> bool:
    """Make sure an engine's packages are installed. Ask before installing them.

    Returns False when they are still missing (the user said no, or no TTY).
    """
    if engine not in ENGINE_REQUIREMENTS:
        return True
    modules, req_file = ENGINE_REQUIREMENTS[engine]
    if engine == "openvoice" and importlib.util.find_spec("openvoice") is None:
        print("openvoice: the OpenVoice package is not installed. Install it from a clone,\n"
              "  see the README section \"Whisper and OpenVoice\".")
    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if not missing:
        return True
    cmd = [sys.executable, "-m", "pip", "install", "-r", req_file]
    print(f"{engine}: missing Python packages: {', '.join(missing)}")
    if not sys.stdin.isatty():
        print(f"Install them with: {' '.join(cmd)}\n")
        return False
    answer = input(f"Install {req_file} now? This can be a large download. [y/N] ")
    if answer.strip().lower() not in ("y", "yes"):
        print(f"Skipped {engine}. Install later with: {' '.join(cmd)}\n")
        return False
    subprocess.run(cmd, check=True)
    importlib.invalidate_caches()
    return True


def _download_gguf(path: str):
    """Fetch one GGUF from the audio.cpp-gguf repo unless it is already on disk.

    The repo stores each file as <model dir>/<file>, so the last two parts of the
    configured path name the file in the repo and the rest is the download root.
    """
    if os.path.exists(path):
        print(f"{path} already exists.\n")
        return
    parent = os.path.dirname(path)
    filename = f"{os.path.basename(parent)}/{os.path.basename(path)}"
    local_dir = os.path.dirname(parent) or "."
    print(f"Downloading {filename} from {AUDIOCPP_GGUF_REPO} into {local_dir}/")
    hf_hub_download(repo_id=AUDIOCPP_GGUF_REPO, filename=filename, local_dir=local_dir)
    print(f"{path} ready.\n")


def setup_audiocpp_stt():
    print("--- Setting up audio.cpp STT model ---")
    _download_gguf(get_settings().audiocpp_model)


def setup_audiocpp_tts():
    print("--- Setting up audio.cpp TTS model ---")
    _download_gguf(get_settings().audiocpp_tts_model)


def setup_whisper():
    from faster_whisper import download_model

    name = get_settings().whisper_model
    print("--- Setting up Whisper ---")
    download_model(name)
    print(f"faster-whisper '{name}' model ready.\n")


def setup_openvoice():
    print("--- Setting up OpenVoice ---")
    settings = get_settings()
    # The v2 repo holds converter/ and base_speakers/ses/ at its root, so both
    # configured dirs sit under the converter dir's parent (checkpoints_v2/).
    root = os.path.dirname(settings.tts_checkpoint_dir)

    if not os.path.exists(os.path.join(settings.tts_checkpoint_dir, "checkpoint.pth")):
        print(f"Downloading OpenVoice v2 checkpoints from Hugging Face into {root}/")
        snapshot_download(
            repo_id="myshell-ai/OpenVoiceV2",
            allow_patterns=["converter/*", "base_speakers/ses/*"],
            local_dir=root,
        )
        print("OpenVoice checkpoints ready.")
    else:
        print(f"OpenVoice checkpoints already exist in {root}/.")

    import nltk

    print("\n--- Setting up NLTK data ---")
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    print("NLTK data ready.\n")


def setup_vibevoice(model_name: str = "microsoft/VibeVoice-Realtime-0.5B"):
    print("--- Setting up VibeVoice ---")

    # 1. Download model weights from HuggingFace
    print(f"Downloading VibeVoice model: {model_name}")
    snapshot_download(repo_id=model_name)
    print("VibeVoice model cached.\n")

    # 2. Verify processor loads
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )
    print("Loading processor (verifying cache)...")
    VibeVoiceStreamingProcessor.from_pretrained(model_name)
    print("VibeVoice processor ready.\n")

    # 3. Download voice preset .pt files from the VibeVoice GitHub repo
    _setup_voice_presets()


def _setup_voice_presets():
    """Download voice preset .pt files from the VibeVoice GitHub repo."""
    os.makedirs(VOICES_DIR, exist_ok=True)

    existing = [f for f in os.listdir(VOICES_DIR) if f.endswith(".pt")]
    if existing:
        print(f"Voice presets already exist ({len(existing)} files in {VOICES_DIR}).")
        for f in sorted(existing):
            print(f"  {f}")
        print()
        return

    print("Downloading voice presets from VibeVoice GitHub repo...")

    # Shallow clone just the demo/voices directory
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = os.path.join(tmpdir, "VibeVoice")
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                "https://github.com/microsoft/VibeVoice.git", repo_dir,
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "demo/voices/streaming_model"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )

        src_dir = os.path.join(repo_dir, "demo", "voices", "streaming_model")
        if not os.path.exists(src_dir):
            print(f"ERROR: Voice presets not found in cloned repo at {src_dir}")
            sys.exit(1)

        pt_files = [f for f in os.listdir(src_dir) if f.endswith(".pt")]
        for f in sorted(pt_files):
            shutil.copy2(os.path.join(src_dir, f), VOICES_DIR)

    copied = [f for f in os.listdir(VOICES_DIR) if f.endswith(".pt")]
    print(f"Copied {len(copied)} voice presets to {VOICES_DIR}/")
    for f in sorted(copied):
        print(f"  {f}")
    print()


TARGETS = {
    "audiocpp-stt": setup_audiocpp_stt,
    "audiocpp-tts": setup_audiocpp_tts,
    "whisper": setup_whisper,
    "openvoice": setup_openvoice,
    "vibevoice": setup_vibevoice,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and cache models for VoiceManager.",
        epilog="If no targets are specified, only the configured engines are set up.",
    )
    parser.add_argument(
        "targets",
        nargs="*",
        choices=list(TARGETS.keys()),
        metavar="TARGET",
        help=f"Models to set up: {', '.join(TARGETS.keys())} (default: configured engines)",
    )
    args = parser.parse_args()

    targets = []
    skipped = False
    if args.targets:
        for target in args.targets:
            if ensure_requirements(TARGET_ENGINES.get(target, "")):
                targets.append(target)
            else:
                skipped = True
    else:
        settings = get_settings()
        print(f"Configured engines: STT={settings.stt_engine}, TTS={settings.tts_engine}\n")
        for engine, table in ((settings.stt_engine, STT_TARGETS),
                              (settings.tts_engine, TTS_TARGETS)):
            if not ensure_requirements(engine):
                skipped = True
                continue
            target = table.get(engine)
            if target is None:
                print(f"{engine}: no setup step, it downloads its model on first start.\n")
            elif target not in targets:
                targets.append(target)

    for name in targets:
        TARGETS[name]()

    if skipped:
        sys.exit("Done, but some engines were skipped (see above).")
    print("Done.")
