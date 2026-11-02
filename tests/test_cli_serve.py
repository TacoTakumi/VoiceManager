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


def test_serve_command_runs_uvicorn_with_the_app_factory(settings_env, run_calls,
                                                        restore_sighup):
    result = CliRunner().invoke(main, ["serve"])

    assert result.exit_code == 0, result.output
    assert run_calls == [EXPECTED_CALL]


def test_module_entry_point_runs_the_same_serve_path(settings_env, run_calls, restore_sighup):
    runpy.run_module("voicemanager", run_name="__main__")

    assert run_calls == [EXPECTED_CALL]


def test_main_is_a_group_that_lists_serve():
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0, result.output
    assert "serve" in result.output


# --- a terminal hangup runs the orderly shutdown ----------------------------------------
#
# uvicorn stops cleanly on SIGINT and SIGTERM, not on SIGHUP. The VoXtream sidecar is a
# session of its own (so a service stop reaches the server first), which means a dropped
# SSH session or a closed terminal pane no longer reaches the sidecar: if the hangup ended
# the server without its lifespan, the sidecar would live on holding the port and the GPU.

import os
import pty
import signal
import subprocess
import sys
import time

from voicemanager.cli import hangup_means_shutdown


@pytest.fixture
def restore_sighup():
    before = signal.getsignal(signal.SIGHUP)
    yield
    signal.signal(signal.SIGHUP, before)


def test_hangup_handler_installs_on_the_default_disposition_and_calls_shutdown(restore_sighup):
    signal.signal(signal.SIGHUP, signal.SIG_DFL)
    calls = []
    assert hangup_means_shutdown(shutdown=lambda: calls.append("shutdown")) is True
    handler = signal.getsignal(signal.SIGHUP)
    assert callable(handler) and handler is not signal.SIG_DFL
    handler(signal.SIGHUP, None)
    assert calls == ["shutdown"]


def test_hangup_handler_leaves_an_ignored_hangup_alone(restore_sighup):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)  # nohup
    assert hangup_means_shutdown(shutdown=lambda: None) is False
    assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN


def test_serve_installs_the_hangup_handler_before_uvicorn_runs(settings_env, run_calls,
                                                             monkeypatch, restore_sighup):
    signal.signal(signal.SIGHUP, signal.SIG_DFL)
    order = []
    monkeypatch.setattr("voicemanager.cli.hangup_means_shutdown",
                        lambda **kw: order.append("hangup") or True)
    run_calls.clear()
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: order.append("run"))
    CliRunner().invoke(main, ["serve"], catch_exceptions=False)
    assert order == ["hangup", "run"]


# A server stand-in: installs the handler, spawns a child in its own session like the
# sidecar, and on SIGTERM stops the child and exits, as the lifespan does.
_SERVER_STAND_IN = """
import os, signal, subprocess, sys, time
from voicemanager.cli import hangup_means_shutdown
child = subprocess.Popen([sys.executable, "-c", "import time\\nwhile True: time.sleep(0.1)"],
                         stdin=subprocess.DEVNULL, start_new_session=True)
with open(sys.argv[1], "w") as f:
    f.write(str(child.pid))
def on_term(signum, frame):
    child.terminate()
    child.wait(timeout=5)
    sys.exit(0)
signal.signal(signal.SIGTERM, on_term)
signal.signal(signal.SIGHUP, signal.SIG_DFL)  # as a server started from a terminal
if sys.argv[2] == "with-handler":
    hangup_means_shutdown()
while True:
    time.sleep(0.1)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


_reaped_pids: set[int] = set()


@pytest.fixture(autouse=True)
def _fresh_reaped_pids():
    """A pid the kernel reuses across tests must not read as already reaped."""
    _reaped_pids.clear()
    yield
    _reaped_pids.clear()


def _reaped(pid: int) -> bool:
    """True once the forked stand-in has exited (a zombie still answers kill 0)."""
    if pid in _reaped_pids:
        return True
    try:
        done, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        done = pid
    if done == pid:
        _reaped_pids.add(pid)
        return True
    return False


def _hangup_run(tmp_path, mode: str) -> tuple[bool, bool]:
    """Run the stand-in in a pty, hang the pty up, report (server alive, child alive)."""
    pidfile = tmp_path / f"child-{mode}.pid"
    pid, fd = pty.fork()
    if pid == 0:  # pragma: no cover - the forked server stand-in
        os.execv(sys.executable, [sys.executable, "-c", _SERVER_STAND_IN, str(pidfile), mode])
    try:
        deadline = time.monotonic() + 10
        while not pidfile.exists() or not pidfile.read_text():
            assert time.monotonic() < deadline, "the stand-in did not start"
            time.sleep(0.05)
        child = int(pidfile.read_text())
        os.close(fd)  # the hangup: SIGHUP to the terminal's foreground process group
        deadline = time.monotonic() + 5
        while not _reaped(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        server_alive = not _reaped(pid)
        time.sleep(0.3)  # time for the stand-in's shutdown to reach the child
        return server_alive, _alive(child)
    finally:
        for p in (pid, child if "child" in locals() else None):
            if p is not None:
                try:
                    os.kill(p, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass


@pytest.mark.skipif(sys.platform != "linux", reason="pty hangup semantics")
def test_hangup_with_the_handler_stops_the_child_in_its_own_session(tmp_path):
    # Without the handler the hangup ends the server and the child lives on.
    assert _hangup_run(tmp_path, "without") == (False, True)
    assert _hangup_run(tmp_path, "with-handler") == (False, False)
