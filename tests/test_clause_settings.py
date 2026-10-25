"""Clause-mode tts_stream thresholds are settings with VM_ environment variables.

Model-free and server-free.
"""
import pytest
from pydantic import ValidationError

import authkit  # test auth env; must be imported before voicemanager

from voicemanager.config import Settings

CASES = [
    ("VM_TTS_CLAUSE_FIRST_WORDS", "tts_clause_first_words", 4, 7),
    ("VM_TTS_CLAUSE_FIRST_DEADLINE_MS", "tts_clause_first_deadline_ms", 600, 250),
    ("VM_TTS_CLAUSE_LATER_WORDS", "tts_clause_later_words", 12, 20),
    ("VM_TTS_CLAUSE_MARGIN_MS", "tts_clause_margin_ms", 1000, 900),
    ("VM_TTS_CLAUSE_MARK_MIN_WORDS", "tts_clause_mark_min_words", 3, 5),
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for env, _, _, _ in CASES:
        monkeypatch.delenv(env, raising=False)


@pytest.mark.parametrize("env,field,default,override", CASES)
def test_default_when_unset(env, field, default, override):
    assert getattr(Settings(_env_file=None), field) == default


@pytest.mark.parametrize("env,field,default,override", CASES)
def test_env_override(monkeypatch, env, field, default, override):
    monkeypatch.setenv(env, str(override))
    assert getattr(Settings(_env_file=None), field) == override


@pytest.mark.parametrize("env,field,default,override", CASES)
@pytest.mark.parametrize("bad", ["0", "-3"])
def test_non_positive_rejected(monkeypatch, env, field, default, override, bad):
    monkeypatch.setenv(env, bad)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_env_example_lists_all_names():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    for env, _, default, _ in CASES:
        assert f"# {env}={default}" in text
