"""VM_AUDIOCPP_{SESSION,REQUEST}_OPTIONS reach the sidecar spawn argv.

Model-free: `subprocess.Popen` is monkeypatched to capture argv, so no process is
spawned. The settings are comma-separated KEY=VALUE lists parsed by
`config.parse_option_list` (fail-fast on malformed input), carried through the
audiocpp_config dict as lists, and rendered by `_SidecarClient._spawn` as repeatable
`--session-option` / `--request-option` args; empty settings add no args.
"""
from __future__ import annotations

import pytest

from voicemanager.config import parse_option_list
from voicemanager.pipeline import audiocpp_stt
from voicemanager.pipeline.audiocpp_stt import _SidecarClient


class _DummyProc:
    pass


@pytest.fixture()
def captured_argv(monkeypatch):
    captured: list[list[str]] = []

    def fake_popen(argv, *args, **kwargs):
        captured.append(list(argv))
        return _DummyProc()

    monkeypatch.setattr(audiocpp_stt.subprocess, "Popen", fake_popen)
    return captured


def _flag_values(argv: list[str], flag: str) -> list[str]:
    return [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == flag]


# --- parse_option_list (config.py) ------------------------------------------------


def test_parse_option_list_splits_and_strips():
    raw = " a=1, voxtral_realtime.stream_batch_tokens=2 ,c=x=y "
    assert parse_option_list(raw) == [
        "a=1", "voxtral_realtime.stream_batch_tokens=2", "c=x=y"]


def test_parse_option_list_empty_and_blank():
    assert parse_option_list("") == []
    assert parse_option_list(" , ,") == []


def test_parse_option_list_rejects_malformed():
    with pytest.raises(ValueError):
        parse_option_list("novalue")
    with pytest.raises(ValueError):
        parse_option_list("=1")


# --- _spawn argv rendering ---------------------------------------------------------


def test_spawn_renders_repeatable_option_args(captured_argv):
    client = _SidecarClient({
        "model": "/models/x.gguf",
        "session_options": ["voxtral_realtime.stream_batch_tokens=2", "b=2"],
        "request_options": ["temperature=0.0"],
    })
    client._spawn("/bin/sidecar", "/tmp/s.sock")
    (argv,) = captured_argv
    assert _flag_values(argv, "--session-option") == [
        "voxtral_realtime.stream_batch_tokens=2", "b=2"]
    assert _flag_values(argv, "--request-option") == ["temperature=0.0"]


def test_spawn_with_default_empty_options_adds_no_args(captured_argv):
    client = _SidecarClient({"model": "/models/x.gguf"})
    client._spawn("/bin/sidecar", "/tmp/s.sock")
    (argv,) = captured_argv
    assert "--session-option" not in argv
    assert "--request-option" not in argv
