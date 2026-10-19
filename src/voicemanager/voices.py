"""The preset voices the configured TTS engine offers, read without a model.

A token may name the preset voice it speaks with, and the CLI and the web
pages check that name before they store it. They run beside the server, not
inside it, so they cannot ask a loaded engine: they read the same config the
engine reads at startup instead.

- audiocpp: the preset names in the manifest at VM_AUDIOCPP_TTS_VOICES
  (refs.json); keys starting with "_" are metadata, as the engine skips them.
- vibevoice: the short names of the ``*.pt`` files in VM_VIBEVOICE_VOICES_DIR
  (``en-Carter_man.pt`` is "Carter"), as the engine names them.
- openvoice: None. It clones a voice from uploaded audio and has no presets.

Only file names and the manifest's keys are read: no reference audio, no
voice tensor and no model. This module must not import the pipeline or model
packages, which pull in torch; the engines' own loaders validate the assets
and load them at server startup, so the name rules are repeated here.
"""
from __future__ import annotations

import json
from pathlib import Path

from voicemanager.config import Settings

ENGINES_WITH_PRESETS = ("audiocpp", "vibevoice")
ENGINES_WITHOUT_PRESETS = ("openvoice",)


class VoiceError(ValueError):
    """A voice name that cannot be used, or a voice list that cannot be read.

    The message is one line, fit to show the operator.
    """


def configured_voices(settings: Settings) -> list[str] | None:
    """The sorted preset names of the configured TTS engine, or None (no presets).

    Raises VoiceError when the engine is unknown or its manifest or voices
    directory cannot be read.
    """
    engine = settings.tts_engine
    if engine == "audiocpp":
        return _manifest_presets(Path(settings.audiocpp_tts_voices))
    if engine == "vibevoice":
        return _vibevoice_voices(Path(settings.vibevoice_voices_dir))
    if engine in ENGINES_WITHOUT_PRESETS:
        return None
    valid = ", ".join(ENGINES_WITH_PRESETS + ENGINES_WITHOUT_PRESETS)
    raise VoiceError(f"Unknown TTS engine {engine!r} (VM_TTS_ENGINE); valid: {valid}.")


def check_voice(settings: Settings, voice: str) -> str:
    """``voice`` without surrounding blanks, if the configured engine offers it.

    Raises VoiceError when the name is empty, when the engine has no preset
    voices (OpenVoice), or when the name is not one of its presets; the last
    message names every valid preset.
    """
    name = voice.strip()
    if not name:
        raise VoiceError("A voice name may not be empty.")
    presets = configured_voices(settings)
    if presets is None:
        raise VoiceError(
            f"The {settings.tts_engine} TTS engine has no preset voices, "
            "so a token cannot name one.")
    if name not in presets:
        raise VoiceError(
            f"Unknown voice {name!r}; valid presets: {', '.join(presets) or 'none'}.")
    return name


def _manifest_presets(path: Path) -> list[str]:
    """The preset names in an audiocpp refs.json; the entries are not checked."""
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VoiceError(f"Cannot read the voice preset manifest {path}: {exc}") from exc
    if not isinstance(entries, dict):
        raise VoiceError(f"The voice preset manifest {path} is not a JSON object.")
    return sorted(
        name for name, entry in entries.items()
        if not name.startswith("_") and isinstance(entry, dict)
    )


def _vibevoice_voices(directory: Path) -> list[str]:
    """The short names of the VibeVoice preset files in ``directory``."""
    if not directory.is_dir():
        raise VoiceError(f"The VibeVoice voices directory {directory} does not exist.")
    return sorted({_vibevoice_short_name(path.stem) for path in directory.glob("*.pt")})


def _vibevoice_short_name(stem: str) -> str:
    """The voice name of a preset file stem: ``en-Carter_man`` is "Carter".

    A stem without a language prefix is kept whole. The same rule as the
    VibeVoice engine's preset loader.
    """
    parts = stem.split("-", 1)
    if len(parts) == 2:
        return parts[1].rsplit("_", 1)[0]
    return stem
