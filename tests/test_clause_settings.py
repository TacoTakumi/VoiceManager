"""Clause-mode tts_stream thresholds are settings with VM_ environment variables.

Model-free and server-free.
"""
import pytest
from pydantic import ValidationError

import authkit  # test auth env; must be imported before voicemanager

from voicemanager.config import Settings

CASES = [
    ("VM_TTS_CLAUSE_FIRST_WORDS", "tts_clause_first_words", 3, 7),
    ("VM_TTS_CLAUSE_FIRST_DEADLINE_MS", "tts_clause_first_deadline_ms", 300, 250),
    ("VM_TTS_CLAUSE_LATER_WORDS", "tts_clause_later_words", 12, 20),
    ("VM_TTS_CLAUSE_MARGIN_MS", "tts_clause_margin_ms", 1000, 900),
    ("VM_TTS_CLAUSE_MARK_MIN_WORDS", "tts_clause_mark_min_words", 3, 5),
]

# The word grace may be 0 (every frame end is a word end), so it is not in CASES,
# whose settings all reject 0.
GRACE_ENV = "VM_TTS_CLAUSE_WORD_GRACE_MS"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for env, _, _, _ in CASES:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv(GRACE_ENV, raising=False)


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


def test_word_grace_default_when_unset():
    assert Settings(_env_file=None).tts_clause_word_grace_ms == 250


@pytest.mark.parametrize("value", [0, 400])
def test_word_grace_env_override(monkeypatch, value):
    monkeypatch.setenv(GRACE_ENV, str(value))
    assert Settings(_env_file=None).tts_clause_word_grace_ms == value


def test_word_grace_negative_rejected(monkeypatch):
    monkeypatch.setenv(GRACE_ENV, "-1")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_env_example_lists_the_word_grace():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    assert f"# {GRACE_ENV}=250" in text


def test_the_guide_names_the_engines_of_the_clause_debug_line():
    """The audiocpp loop and the base loop (openvoice, vibevoice) write the
    'Clause TTS engine request' line. The voxtream engine cuts no pieces and
    writes no such line, so the guide must not give the line to every engine."""
    from pathlib import Path

    guide = (Path(__file__).resolve().parents[1] / "docs"
             / "client-integration.md").read_text(encoding="utf-8")
    paragraph = next(p for p in guide.split("\n\n")
                     if "Clause TTS engine request: request_id" in p)
    assert "The other engines write the line" not in paragraph
    assert "The voxtream engine writes no such line" in paragraph
