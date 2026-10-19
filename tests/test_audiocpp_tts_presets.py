"""Preset voice loader tests for the audiocpp TTS engine.

Model-free and server-free: the loader only reads refs.json and stats WAV files.
The real voices/designed/refs.json covers the happy paths; temp refs.json trees
cover the fail-fast cases (a missing/empty ref_text or an absent WAV must
fail AT LOAD TIME, never at request time).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from voicemanager.pipeline.audiocpp_tts import (
    BREEZE_TTS_MODEL_TASK,
    DEFAULT_VOICE,
    QWEN3_TTS_MODEL_TASK,
    VOXCPM2_TTS_MODEL_TASK,
    AudioCppTTSEngine,
    PresetLoadError,
    UnknownVoiceError,
    load_presets,
    resolve_preset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_REFS = REPO_ROOT / "voices" / "designed" / "refs.json"


def _write_refs(tmp_path: Path, entries: dict) -> Path:
    refs = tmp_path / "refs.json"
    refs.write_text(json.dumps(entries), encoding="utf-8")
    return refs


def _touch_wav(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_bytes(b"RIFFfake")


def _valid_entries() -> dict:
    # A minimal valid refs.json shape including the default preset.
    return {
        "_comment": "ignored metadata key",
        DEFAULT_VOICE: {"file": "a.wav", "ref_text": "transcript a"},
        "other_voice": {"file": "b.wav", "ref_text": "transcript b"},
    }


# --- happy paths against the real frozen assets -----------------------------

def test_real_refs_loads_the_required_presets() -> None:
    # Superset, not equality: registering a new voice is a normal workflow step
    # and must not break this test. What matters is that the presets other code
    # names by default are present and that every entry is a valid pair.
    presets = load_presets(REAL_REFS)
    assert {"designed_female_s101", "designed_male_s202", "lcars_demo"} <= set(presets)
    for preset in presets.values():
        assert Path(preset.wav_path).is_file(), preset.wav_path
        assert preset.ref_text.strip(), preset.name


def test_each_preset_resolves_its_own_pair() -> None:
    presets = load_presets(REAL_REFS)
    for name in ("designed_female_s101", "designed_male_s202", "lcars_demo"):
        preset = resolve_preset(presets, name)
        assert preset.name == name
        assert Path(preset.wav_path).name == f"{name}.wav"
        assert preset.ref_text == presets[name].ref_text


def test_no_voice_named_uses_the_default() -> None:
    presets = load_presets(REAL_REFS)
    assert resolve_preset(presets, None).name == DEFAULT_VOICE
    assert resolve_preset(presets, "").name == DEFAULT_VOICE
    assert DEFAULT_VOICE == "designed_female_s101"


def test_unknown_voice_raises_listing_valid_presets() -> None:
    presets = load_presets(REAL_REFS)
    with pytest.raises(UnknownVoiceError) as exc_info:
        resolve_preset(presets, "no-such-voice")
    message = str(exc_info.value)
    assert "no-such-voice" in message
    for name in ("designed_female_s101", "designed_male_s202", "lcars_demo"):
        assert name in message


# --- fail-fast at load time -------------------------------------------

def test_missing_ref_text_fails_at_load_time(tmp_path: Path) -> None:
    entries = _valid_entries()
    del entries["other_voice"]["ref_text"]
    refs = _write_refs(tmp_path, entries)
    _touch_wav(tmp_path, "a.wav")
    _touch_wav(tmp_path, "b.wav")
    with pytest.raises(PresetLoadError, match="other_voice"):
        load_presets(refs)


def test_empty_ref_text_fails_at_load_time(tmp_path: Path) -> None:
    entries = _valid_entries()
    entries["other_voice"]["ref_text"] = "   "
    refs = _write_refs(tmp_path, entries)
    _touch_wav(tmp_path, "a.wav")
    _touch_wav(tmp_path, "b.wav")
    with pytest.raises(PresetLoadError, match="other_voice"):
        load_presets(refs)


def test_absent_wav_fails_at_load_time(tmp_path: Path) -> None:
    refs = _write_refs(tmp_path, _valid_entries())
    _touch_wav(tmp_path, "a.wav")  # b.wav deliberately absent
    with pytest.raises(PresetLoadError, match="b.wav"):
        load_presets(refs)


def test_missing_default_preset_fails_at_load_time(tmp_path: Path) -> None:
    entries = _valid_entries()
    del entries[DEFAULT_VOICE]
    refs = _write_refs(tmp_path, entries)
    _touch_wav(tmp_path, "a.wav")
    _touch_wav(tmp_path, "b.wav")
    with pytest.raises(PresetLoadError, match=DEFAULT_VOICE):
        load_presets(refs)


def test_missing_refs_json_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(PresetLoadError, match="refs.json"):
        load_presets(tmp_path / "refs.json")


# --- per-model ref_text -------------------------------------------------------------

def test_ref_text_optional_when_not_required(tmp_path: Path) -> None:
    entries = _valid_entries()
    del entries["other_voice"]["ref_text"]
    refs = _write_refs(tmp_path, entries)
    _touch_wav(tmp_path, "a.wav")
    _touch_wav(tmp_path, "b.wav")
    presets = load_presets(refs, ref_text_required=False)
    assert presets["other_voice"].ref_text == ""


def test_ref_text_optional_allows_empty_when_not_required(tmp_path: Path) -> None:
    entries = _valid_entries()
    entries["other_voice"]["ref_text"] = ""
    refs = _write_refs(tmp_path, entries)
    _touch_wav(tmp_path, "a.wav")
    _touch_wav(tmp_path, "b.wav")
    presets = load_presets(refs, ref_text_required=False)
    assert presets["other_voice"].ref_text == ""


def test_ref_text_optional_still_requires_wav(tmp_path: Path) -> None:
    entries = _valid_entries()
    del entries["other_voice"]["ref_text"]
    refs = _write_refs(tmp_path, entries)
    _touch_wav(tmp_path, "a.wav")  # b.wav absent
    with pytest.raises(PresetLoadError, match="b.wav"):
        load_presets(refs, ref_text_required=False)


# --- breeze requires ref_text at engine load -----------------------------------------

def _refs_with_empty_ref_text(tmp_path: Path) -> Path:
    entries = _valid_entries()
    entries["other_voice"]["ref_text"] = ""
    refs = _write_refs(tmp_path, entries)
    _touch_wav(tmp_path, "a.wav")
    _touch_wav(tmp_path, "b.wav")
    return refs


def test_breeze_engine_rejects_empty_ref_text_at_load(tmp_path: Path) -> None:
    """Under breeze-tts-streaming a preset with an empty transcript fails at
    engine load, never at request time."""
    engine = AudioCppTTSEngine({"socket": str(tmp_path / "unused.sock"),
                                "model_task": BREEZE_TTS_MODEL_TASK})
    with pytest.raises(PresetLoadError, match="other_voice"):
        engine.load(voices_path=str(_refs_with_empty_ref_text(tmp_path)))


def test_qwen3_engine_rejects_empty_ref_text_at_load(tmp_path: Path) -> None:
    """Under qwen3-tts-streaming a preset with an empty transcript fails at engine
    load, never at request time."""
    engine = AudioCppTTSEngine({"socket": str(tmp_path / "unused.sock"),
                                "model_task": QWEN3_TTS_MODEL_TASK})
    with pytest.raises(PresetLoadError, match="other_voice"):
        engine.load(voices_path=str(_refs_with_empty_ref_text(tmp_path)))


def test_voxcpm2_engine_still_accepts_empty_ref_text_at_load(tmp_path: Path) -> None:
    """The contrast: the same manifest loads under voxcpm2 (ref_text optional)."""
    engine = AudioCppTTSEngine({"socket": str(tmp_path / "unused.sock"),
                                "model_task": VOXCPM2_TTS_MODEL_TASK})
    engine.load(voices_path=str(_refs_with_empty_ref_text(tmp_path)))
    assert engine._presets["other_voice"].ref_text == ""


def test_ref_text_required_is_the_default(tmp_path: Path) -> None:
    # The default flag (dots) still fails fast on a missing ref_text.
    entries = _valid_entries()
    del entries["other_voice"]["ref_text"]
    refs = _write_refs(tmp_path, entries)
    _touch_wav(tmp_path, "a.wav")
    _touch_wav(tmp_path, "b.wav")
    with pytest.raises(PresetLoadError, match="other_voice"):
        load_presets(refs)
