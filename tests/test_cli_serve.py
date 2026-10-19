"""The `voicemanager serve` command and `python -m voicemanager` start the same server.

uvicorn.run is patched, so no server starts and no model loads.
"""
import runpy

import pytest
from click.testing import CliRunner

from voicemanager import config
from voicemanager.cli import main


@pytest.fixture
def settings_env(monkeypatch):
    """Fresh settings from the env, with a known host and port."""
    monkeypatch.setenv("VM_HOST", "127.0.0.9")
    monkeypatch.setenv("VM_PORT", "8123")
    monkeypatch.setattr(config, "_settings", None)


@pytest.fixture
def run_calls(monkeypatch):
    """Record each uvicorn.run call instead of starting a server."""
    calls: list[tuple[tuple, dict]] = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr("uvicorn.run", fake_run)
    return calls


EXPECTED_CALL = (
    ("voicemanager.app:create_app",),
    {
        "host": "127.0.0.9",
        "port": 8123,
        "factory": True,
        "proxy_headers": False,
    },
)


def test_serve_command_runs_uvicorn_with_the_app_factory(settings_env, run_calls):
    result = CliRunner().invoke(main, ["serve"])

    assert result.exit_code == 0, result.output
    assert run_calls == [EXPECTED_CALL]


def test_module_entry_point_runs_the_same_serve_path(settings_env, run_calls):
    runpy.run_module("voicemanager", run_name="__main__")

    assert run_calls == [EXPECTED_CALL]


def test_main_is_a_group_that_lists_serve():
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    assert "serve" in result.output
