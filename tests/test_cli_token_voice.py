"""A token's preset voice: the configured voice list and the CLI that sets it.

`voicemanager user token add --voice` and `voicemanager user token voice`
check a name against the configured TTS engine's presets, read from its
config, manifest or voices directory alone. Every test patches the model
loaders to fail, so a test that passes proves no model was loaded; each runs
against a fresh store file (VM_DB_PATH) and freshly loaded settings.
"""
import authkit  # noqa: F401  (sets up the auth env before voicemanager is imported)

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from click.testing import CliRunner

import voicemanager.models as models
from voicemanager import config, voices
from voicemanager.auth import accounts
from voicemanager.auth.store import open_store
from voicemanager.cli import main
from voicemanager.pipeline import audiocpp_tts, vibevoice_tts

AUDIOCPP_PRESETS = ["calm_male", "designed_female_s101", "lcars_demo"]
VIBEVOICE_VOICES = ["Carter", "Emma", "Solo"]


def _model_loaded(*_args, **_kwargs):
    raise AssertionError("a model loader was called")


@pytest.fixture(autouse=True)
def no_models(monkeypatch):
    """Make every model and voice-asset loader fail the test if it is called."""
    for name in ("load_models", "load_tts_models", "_load_openvoice_tts",
                 "_load_vibevoice_tts", "_load_audiocpp_tts"):
        monkeypatch.setattr(models, name, _model_loaded)
    monkeypatch.setattr(audiocpp_tts, "load_presets", _model_loaded)
    monkeypatch.setattr(audiocpp_tts.AudioCppTTSEngine, "load", _model_loaded)
    monkeypatch.setattr(vibevoice_tts.VibeVoiceTTSEngine, "load", _model_loaded)
    monkeypatch.setattr(vibevoice_tts.VibeVoiceTTSEngine, "_load_voice_presets",
                        _model_loaded)
    monkeypatch.setattr(torch, "load", _model_loaded)


@pytest.fixture(autouse=True)
def db_path(tmp_path, monkeypatch):
    """A fresh store file per test, read through freshly loaded settings."""
    path = tmp_path / "state.db"
    monkeypatch.setenv("VM_DB_PATH", str(path))
    monkeypatch.setattr(config, "_settings", None)
    return path


def _use_engine(monkeypatch, engine, **env):
    monkeypatch.setenv("VM_TTS_ENGINE", engine)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setattr(config, "_settings", None)


@pytest.fixture
def manifest(tmp_path):
    """A preset manifest shaped like voices/designed/refs.json."""
    path = tmp_path / "designed" / "refs.json"
    path.parent.mkdir()
    entries = {"_comment": "metadata, not a preset"}
    for name in AUDIOCPP_PRESETS:
        entries[name] = {"file": f"{name}.wav", "ref_text": "Standing by."}
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


@pytest.fixture
def audiocpp(monkeypatch, manifest):
    _use_engine(monkeypatch, "audiocpp", VM_AUDIOCPP_TTS_VOICES=manifest)


@pytest.fixture
def voices_dir(tmp_path):
    """A VibeVoice voices dir: preset .pt files named like the upstream ones."""
    path = tmp_path / "streaming_model"
    path.mkdir()
    for name in ("en-Carter_man.pt", "en-Emma_woman.pt", "Solo.pt", "notes.txt"):
        (path / name).write_bytes(b"not a real preset")
    return path


@pytest.fixture
def vibevoice(monkeypatch, voices_dir):
    _use_engine(monkeypatch, "vibevoice", VM_VIBEVOICE_VOICES_DIR=voices_dir)


@pytest.fixture
def openvoice(monkeypatch):
    _use_engine(monkeypatch, "openvoice")


@pytest.fixture
def ann():
    with open_store() as store:
        accounts.add_user(store, "ann", None)
        accounts.mint_token(store, "ann", "console")
    return "ann"


def run(*args):
    return CliRunner().invoke(main, ["user", "token", *args])


def _one_line_error(result):
    """The error the command printed: exactly one line starting 'Error:'."""
    lines = [line for line in result.output.splitlines() if line.startswith("Error:")]
    assert len(lines) == 1, result.output
    return lines[0]


def _voice_of(owner, name):
    with open_store() as store:
        [token] = [t for t in store.list_tokens(owner) if t.name == name]
    return token.voice


def _names_every(names, line):
    return all(name in line for name in names)


# --- configured_voices --------------------------------------------------------

def test_audiocpp_voices_are_the_manifest_presets(audiocpp):
    assert voices.configured_voices(config.get_settings()) == AUDIOCPP_PRESETS


def test_vibevoice_voices_are_the_preset_file_names(vibevoice):
    assert voices.configured_voices(config.get_settings()) == VIBEVOICE_VOICES


def test_openvoice_has_no_preset_voices(openvoice):
    assert voices.configured_voices(config.get_settings()) is None


def test_missing_manifest_is_a_voice_error(monkeypatch, tmp_path):
    _use_engine(monkeypatch, "audiocpp", VM_AUDIOCPP_TTS_VOICES=tmp_path / "none.json")

    with pytest.raises(voices.VoiceError, match="none.json"):
        voices.configured_voices(config.get_settings())


def test_missing_voices_dir_is_a_voice_error(monkeypatch, tmp_path):
    _use_engine(monkeypatch, "vibevoice", VM_VIBEVOICE_VOICES_DIR=tmp_path / "gone")

    with pytest.raises(voices.VoiceError, match="gone"):
        voices.configured_voices(config.get_settings())


def test_importing_voices_imports_no_model_runtime():
    src = Path(__file__).resolve().parent.parent / "src"
    code = (
        "import sys, voicemanager.voices\n"
        "heavy = {'torch', 'faster_whisper', 'silero_vad', 'voicemanager.models',\n"
        "         'voicemanager.pipeline'} & set(sys.modules)\n"
        "assert not heavy, heavy\n"
    )
    env = dict(os.environ, PYTHONPATH=str(src))
    result = subprocess.run([sys.executable, "-c", code], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --- the store primitive and its account rule ---------------------------------

def test_store_sets_and_reports_a_missing_token(ann):
    with open_store() as store:
        assert store.set_token_voice("ann", "console", "lcars_demo") is True
        assert store.set_token_voice("ann", "nothing", "lcars_demo") is False
    assert _voice_of("ann", "console") == "lcars_demo"


def test_account_rule_refuses_unknown_user_and_token(ann):
    with open_store() as store:
        with pytest.raises(accounts.AccountError, match="No user named 'nobody'"):
            accounts.set_token_voice(store, "nobody", "console", "lcars_demo")
        with pytest.raises(accounts.AccountError, match="no token named 'nothing'"):
            accounts.set_token_voice(store, "ann", "nothing", "lcars_demo")
        accounts.set_token_voice(store, "ANN", "console", "lcars_demo")
        accounts.set_token_voice(store, "ann", "console", None)
    assert _voice_of("ann", "console") == ""


# --- token add --voice --------------------------------------------------------

def test_add_with_a_listed_voice_is_shown_by_list(audiocpp, ann):
    result = run("add", "ann", "laptop", "--voice", "lcars_demo")

    assert result.exit_code == 0, result.output
    assert _voice_of("ann", "laptop") == "lcars_demo"
    listing = run("list", "ann")
    assert "  voice: lcars_demo" in listing.output


def test_add_with_an_unlisted_voice_names_the_presets(audiocpp, ann):
    result = run("add", "ann", "laptop", "--voice", "Bogus")

    assert result.exit_code != 0
    line = _one_line_error(result)
    assert "'Bogus'" in line and _names_every(AUDIOCPP_PRESETS, line), line
    assert "vmp_" not in result.output
    with open_store() as store:
        assert [t.name for t in store.list_tokens("ann")] == ["console"]


def test_add_without_a_voice_needs_no_presets(openvoice, ann):
    result = run("add", "ann", "laptop")

    assert result.exit_code == 0, result.output
    assert _voice_of("ann", "laptop") == ""


def test_add_with_a_voice_refused_under_openvoice(openvoice, ann):
    result = run("add", "ann", "laptop", "--voice", "lcars_demo")

    assert result.exit_code != 0
    assert "openvoice" in _one_line_error(result).lower()
    with open_store() as store:
        assert [t.name for t in store.list_tokens("ann")] == ["console"]


# --- token voice --------------------------------------------------------------

def test_voice_is_a_verb_of_the_token_group():
    assert "voice" in main.commands["user"].commands["token"].commands


def test_voice_sets_a_listed_preset(audiocpp, ann):
    result = run("voice", "ANN", "console", "calm_male")

    assert result.exit_code == 0, result.output
    assert "calm_male" in result.output
    assert _voice_of("ann", "console") == "calm_male"
    assert "  voice: calm_male" in run("list", "ann").output


def test_voice_refuses_an_unlisted_name_and_keeps_the_old_one(audiocpp, ann):
    assert run("voice", "ann", "console", "calm_male").exit_code == 0

    result = run("voice", "ann", "console", "Bogus")

    assert result.exit_code != 0
    line = _one_line_error(result)
    assert "'Bogus'" in line and _names_every(AUDIOCPP_PRESETS, line), line
    assert _voice_of("ann", "console") == "calm_male"


def test_voice_clear_empties_it(audiocpp, ann):
    assert run("voice", "ann", "console", "calm_male").exit_code == 0

    result = run("voice", "ann", "console", "--clear")

    assert result.exit_code == 0, result.output
    assert _voice_of("ann", "console") == ""


def test_voice_needs_a_name_or_clear_but_not_both(audiocpp, ann):
    neither = run("voice", "ann", "console")
    both = run("voice", "ann", "console", "calm_male", "--clear")

    assert neither.exit_code != 0 and both.exit_code != 0
    _one_line_error(neither)
    _one_line_error(both)
    assert _voice_of("ann", "console") == ""


def test_voice_refused_under_openvoice(openvoice, ann):
    result = run("voice", "ann", "console", "lcars_demo")

    assert result.exit_code != 0
    assert "openvoice" in _one_line_error(result).lower()
    assert _voice_of("ann", "console") == ""


def test_voice_clear_allowed_under_openvoice(openvoice, ann):
    with open_store() as store:
        store.set_token_voice("ann", "console", "left_over")

    result = run("voice", "ann", "console", "--clear")

    assert result.exit_code == 0, result.output
    assert _voice_of("ann", "console") == ""


def test_voice_with_vibevoice_presets(vibevoice, ann):
    accepted = run("voice", "ann", "console", "Carter")
    refused = run("voice", "ann", "console", "Bob")

    assert accepted.exit_code == 0, accepted.output
    assert refused.exit_code != 0
    assert _names_every(VIBEVOICE_VOICES, _one_line_error(refused))
    assert _voice_of("ann", "console") == "Carter"


def test_voice_on_unknown_token_refused(audiocpp, ann):
    result = run("voice", "ann", "nothing", "calm_male")

    assert result.exit_code != 0
    assert "no token named 'nothing'" in _one_line_error(result)


def test_voice_for_unknown_user_refused(audiocpp):
    result = run("voice", "nobody", "console", "calm_male")

    assert result.exit_code != 0
    assert "No user named 'nobody'" in _one_line_error(result)


def test_unreadable_manifest_is_a_one_line_error(monkeypatch, tmp_path, ann):
    _use_engine(monkeypatch, "audiocpp", VM_AUDIOCPP_TTS_VOICES=tmp_path / "none.json")

    result = run("voice", "ann", "console", "calm_male")

    assert result.exit_code != 0
    assert "none.json" in _one_line_error(result)
