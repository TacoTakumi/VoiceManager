"""The turn-engine settings: VM_TURN_ENGINE and the VM_TYGER_* values.

Model-free and server-free. Covers the defaults, an override from the environment
and from .env, the off|tyger choice, the range checks, and that the sidecar's
interpreter path resolves against the VoiceManager home like other path settings.
"""
from pathlib import Path

import pytest
from pydantic import ValidationError

import authkit  # test auth env; must be imported before voicemanager  # noqa: F401

from voicemanager.config import Settings, resolve_home

PINNED_REVISION = "a5d4d73bb74fa4a7082dcd4387e44a1e5ac151b6"


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


# --- defaults ------------------------------------------------------------------------

def test_turn_engine_is_off_by_default():
    assert _settings().turn_engine == "off"


def test_rule_defaults():
    s = _settings()
    assert s.tyger_endpoint_threshold == 0.915
    assert s.tyger_endpoint_debounce == 2
    assert s.tyger_barge_threshold == 0.6
    assert s.tyger_barge_debounce == 2
    assert s.tyger_warmup_frames == 6
    assert s.tyger_min_speech_ms == 350
    assert s.tyger_voice_gate_db == -50.0
    assert s.tyger_gate_margin_db == 8.0
    assert s.tyger_gate_max_db == -38.0
    assert s.tyger_silence_fallback_ms == 1200


def test_sidecar_defaults():
    s = _settings()
    assert s.tyger_py == str(resolve_home() / "venv-tyger" / "bin" / "python")
    assert s.tyger_port == 8257
    assert s.tyger_device == ""
    assert s.tyger_model == "nineninesix/tyger"
    assert s.tyger_revision == PINNED_REVISION
    assert s.tyger_hold_tail_ms == 150
    assert s.tyger_start_wait_ms == 2000
    assert s.tyger_ready_timeout == 180.0


def test_sidecar_port_differs_from_the_server_and_voxtream_ports():
    s = _settings()
    assert s.tyger_port not in (s.port, s.voxtream_port)


# --- overrides -----------------------------------------------------------------------

def test_turn_engine_tyger_from_the_environment(monkeypatch):
    monkeypatch.setenv("VM_TURN_ENGINE", "tyger")
    assert _settings().turn_engine == "tyger"


def test_every_tyger_value_reads_from_the_environment(monkeypatch):
    values = {
        "VM_TYGER_PY": "/home/you/venv-tyger/bin/python",
        "VM_TYGER_PORT": "9001",
        "VM_TYGER_DEVICE": "cuda:1",
        "VM_TYGER_MODEL": "/home/you/bundles/tyger",
        "VM_TYGER_REVISION": "main",
        "VM_TYGER_ENDPOINT_THRESHOLD": "0.8",
        "VM_TYGER_ENDPOINT_DEBOUNCE": "3",
        "VM_TYGER_BARGE_THRESHOLD": "0.55",
        "VM_TYGER_BARGE_DEBOUNCE": "4",
        "VM_TYGER_WARMUP_FRAMES": "0",
        "VM_TYGER_MIN_SPEECH_MS": "200",
        "VM_TYGER_VOICE_GATE_DB": "-45",
        "VM_TYGER_GATE_MARGIN_DB": "6",
        "VM_TYGER_GATE_MAX_DB": "-40",
        "VM_TYGER_SILENCE_FALLBACK_MS": "800",
        "VM_TYGER_HOLD_TAIL_MS": "0",
        "VM_TYGER_START_WAIT_MS": "500",
        "VM_TYGER_READY_TIMEOUT": "30",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    s = _settings()
    assert s.tyger_py == "/home/you/venv-tyger/bin/python"
    assert s.tyger_port == 9001
    assert s.tyger_device == "cuda:1"
    assert s.tyger_model == "/home/you/bundles/tyger"
    assert s.tyger_revision == "main"
    assert s.tyger_endpoint_threshold == 0.8
    assert s.tyger_endpoint_debounce == 3
    assert s.tyger_barge_threshold == 0.55
    assert s.tyger_barge_debounce == 4
    assert s.tyger_warmup_frames == 0
    assert s.tyger_min_speech_ms == 200
    assert s.tyger_voice_gate_db == -45.0
    assert s.tyger_gate_margin_db == 6.0
    assert s.tyger_gate_max_db == -40.0
    assert s.tyger_silence_fallback_ms == 800
    assert s.tyger_hold_tail_ms == 0
    assert s.tyger_start_wait_ms == 500
    assert s.tyger_ready_timeout == 30.0


def test_tyger_keys_in_a_dotenv_file_are_known(tmp_path):
    """An unknown key in .env stops startup, so each new key must be a field."""
    env = tmp_path / ".env"
    env.write_text(
        "VM_TURN_ENGINE=tyger\n"
        "VM_TYGER_BARGE_THRESHOLD=0.55\n"
        "VM_TYGER_HOLD_TAIL_MS=200\n"
        "VM_TYGER_START_WAIT_MS=1500\n"
    )
    s = Settings(_env_file=env)
    assert s.turn_engine == "tyger"
    assert s.tyger_barge_threshold == 0.55
    assert s.tyger_hold_tail_ms == 200
    assert s.tyger_start_wait_ms == 1500


def test_relative_interpreter_resolves_against_the_home():
    s = _settings(tyger_py="venv-tyger/bin/python")
    assert s.tyger_py == str(resolve_home() / "venv-tyger/bin/python")
    assert Path(s.tyger_py).is_absolute()


def test_model_repo_id_is_not_anchored_to_the_home():
    assert _settings(tyger_model="org/name").tyger_model == "org/name"


@pytest.mark.parametrize("device", ["", "auto", "cpu", "cuda", "cuda:0", "cuda:3"])
def test_device_accepts_the_vm_device_forms(device):
    assert _settings(tyger_device=device).tyger_device == device


# --- validation ----------------------------------------------------------------------

def test_turn_engine_rejects_an_unknown_value():
    with pytest.raises(ValidationError, match="turn_engine"):
        _settings(turn_engine="silero")


@pytest.mark.parametrize("field", ["tyger_endpoint_threshold", "tyger_barge_threshold"])
@pytest.mark.parametrize("value", [-0.1, 1.01])
def test_thresholds_stay_within_zero_and_one(field, value):
    with pytest.raises(ValidationError, match=field):
        _settings(**{field: value})


@pytest.mark.parametrize("field", [
    "tyger_endpoint_debounce", "tyger_barge_debounce", "tyger_silence_fallback_ms",
    "tyger_start_wait_ms", "tyger_ready_timeout", "tyger_port",
])
def test_values_that_must_be_positive(field):
    with pytest.raises(ValidationError, match=field):
        _settings(**{field: 0})


@pytest.mark.parametrize("field", [
    "tyger_warmup_frames", "tyger_min_speech_ms", "tyger_hold_tail_ms",
])
def test_values_that_must_not_be_negative(field):
    assert getattr(_settings(**{field: 0}), field) == 0
    with pytest.raises(ValidationError, match=field):
        _settings(**{field: -1})


def test_voice_gate_is_a_level_at_or_below_full_scale():
    assert _settings(tyger_voice_gate_db=0, tyger_gate_max_db=0).tyger_voice_gate_db == 0
    with pytest.raises(ValidationError, match="tyger_voice_gate_db"):
        _settings(tyger_voice_gate_db=3)


@pytest.mark.parametrize("device", ["gpu", "cuda:", "cuda:x", "cuda0", "CPU"])
def test_device_rejects_other_forms(device):
    with pytest.raises(ValidationError, match="tyger_device"):
        _settings(tyger_device=device)


@pytest.mark.parametrize("field", ["tyger_model", "tyger_revision", "tyger_py"])
def test_blank_bundle_and_interpreter_values_are_refused(field):
    with pytest.raises(ValidationError, match=field):
        _settings(**{field: ""})


def test_env_example_lists_every_turn_engine_key():
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    names = [name for name in Settings.model_fields
             if name == "turn_engine" or name.startswith("tyger_")]
    assert len(names) == 23
    for name in names:
        assert f"# VM_{name.upper()}=" in text, name


# --- the experimental pre-filter -------------------------------------------------------

def test_the_prefilter_is_off_by_default_with_a_floor_of_minus_52_db():
    s = _settings()
    assert s.tyger_prefilter == "off"
    assert s.tyger_prefilter_floor_db == -52.0


def test_the_prefilter_settings_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("VM_TYGER_PREFILTER", "voxlock")
    monkeypatch.setenv("VM_TYGER_PREFILTER_FLOOR_DB", "-48.5")
    s = _settings()
    assert s.tyger_prefilter == "voxlock"
    assert s.tyger_prefilter_floor_db == -48.5


@pytest.mark.parametrize("field, value, env_name", [
    ("tyger_prefilter", "speex", "VM_TYGER_PREFILTER"),
    ("tyger_prefilter_floor_db", 3.0, "VM_TYGER_PREFILTER_FLOOR_DB"),
])
def test_a_bad_prefilter_value_is_refused_naming_the_setting(field, value, env_name, monkeypatch):
    monkeypatch.setenv(env_name, str(value))
    with pytest.raises(ValidationError) as err:
        _settings()
    assert field in str(err.value)


def test_the_gate_max_may_not_sit_under_the_voice_gate():
    with pytest.raises(ValidationError) as err:
        _settings(tyger_voice_gate_db=-40.0, tyger_gate_max_db=-45.0)
    assert "VM_TYGER_GATE_MAX_DB" in str(err.value) and "VM_TYGER_VOICE_GATE_DB" in str(err.value)
    pinned = _settings(tyger_voice_gate_db=-45.0, tyger_gate_margin_db=0, tyger_gate_max_db=-45.0)
    assert pinned.tyger_gate_max_db == pinned.tyger_voice_gate_db


def test_the_gate_margin_is_not_negative():
    with pytest.raises(ValidationError, match="tyger_gate_margin_db"):
        _settings(tyger_gate_margin_db=-1)

