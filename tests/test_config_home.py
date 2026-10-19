"""One home directory holds .env and anchors relative path settings.

Model-free and server-free. Pins the resolution order of resolve_home(), that
Settings() reads <home>/.env and not ./.env, and that relative path settings
resolve against the home while absolute ones stay as given.
"""
from pathlib import Path

import pytest

import authkit  # test auth env; must be imported before voicemanager

import voicemanager.config as cfg
from voicemanager.config import Settings, resolve_home

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv(cfg.HOME_ENV, raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv(authkit.LEGACY_KEY_ENV, raising=False)


@pytest.fixture
def not_a_clone(monkeypatch, tmp_path):
    """Make config.py look installed outside a clone (no pyproject.toml above it)."""
    fake = tmp_path / "site-packages" / "lib" / "voicemanager" / "config.py"
    fake.parent.mkdir(parents=True)
    monkeypatch.setattr(cfg, "__file__", str(fake))


# --- resolve_home() order ------------------------------------------------------------

def test_override_env_var_wins(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv(cfg.HOME_ENV, str(tmp_path))
    assert resolve_home() == tmp_path


def test_override_expands_the_user_home(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv(cfg.HOME_ENV, "~/vm")
    assert resolve_home() == tmp_path / "vm"


def test_clone_root_without_an_override(clean_env):
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert resolve_home() == REPO_ROOT


def test_xdg_config_home_outside_a_clone(clean_env, not_a_clone, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert resolve_home() == tmp_path / "xdg" / "voicemanager"


def test_dot_config_outside_a_clone_without_xdg(clean_env, not_a_clone, monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert resolve_home() == tmp_path / ".config" / "voicemanager"


# --- the default .env is <home>/.env -------------------------------------------------

def test_settings_read_the_home_env_not_the_cwd_env(clean_env, monkeypatch, tmp_path):
    home, cwd = tmp_path / "home", tmp_path / "cwd"
    home.mkdir()
    cwd.mkdir()
    (home / ".env").write_text("VM_PORT=1234\n")
    (cwd / ".env").write_text("VM_PORT=4321\n")
    monkeypatch.setenv(cfg.HOME_ENV, str(home))
    monkeypatch.chdir(cwd)
    assert Settings().port == 1234


def test_an_explicit_env_file_still_wins(clean_env, monkeypatch, tmp_path):
    (tmp_path / "other.env").write_text("VM_PORT=4321\n")
    monkeypatch.setenv(cfg.HOME_ENV, str(tmp_path))
    assert Settings(_env_file=tmp_path / "other.env").port == 4321


# --- relative path settings resolve against the home --------------------------------

def test_relative_defaults_resolve_against_the_home(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv(cfg.HOME_ENV, str(tmp_path))
    s = Settings(_env_file=None)
    assert s.audiocpp_sidecar_bin == str(tmp_path / "sidecar/build/bin/audiocpp_sidecar")
    assert s.audiocpp_model == str(
        tmp_path / "models/audiocpp/Nemotron-3.5-ASR-Streaming-0.6B-GGUF/"
        "nemotron-3.5-asr-streaming-0.6b-f16.gguf")
    assert s.audiocpp_tts_model == str(
        tmp_path / "models/audiocpp/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf")
    assert s.audiocpp_tts_voices == str(tmp_path / "voices/designed/refs.json")
    assert s.vibevoice_voices_dir == str(tmp_path / "voices/streaming_model")
    assert s.tts_checkpoint_dir == str(tmp_path / "checkpoints_v2/converter")
    assert s.tts_base_speaker_dir == str(tmp_path / "checkpoints_v2/base_speakers/ses")


def test_relative_env_values_resolve_against_the_home(clean_env, monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("VM_AUDIOCPP_MODEL=gguf/stt.gguf\n")
    monkeypatch.setenv(cfg.HOME_ENV, str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)
    assert Settings().audiocpp_model == str(tmp_path / "gguf/stt.gguf")


def test_absolute_and_blank_paths_stay_as_given(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv(cfg.HOME_ENV, str(tmp_path))
    s = Settings(_env_file=None, audiocpp_tts_model="/opt/models/tts.gguf", audiocpp_model="")
    assert s.audiocpp_tts_model == "/opt/models/tts.gguf"
    assert s.audiocpp_model == ""
