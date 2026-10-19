"""Reaping the spawned audio.cpp sidecar on teardown.

Model-free and server-free (the CLAUDE.md invariant): no GPU, no native binary, no
model, no live server. The "sidecar" here is a plain `python -c` child process, which
is all that is needed to assert the thing under test -- that a sidecar VoiceManager
spawned is genuinely *reaped* on disconnect. Its socket side is the usual in-process
`tests/audiocpp_fake.py` standing on the private path the owned-mode spawn asked for
(same fiction the ownership tests use), so connect() takes the production path.

The defect this pins down: `close()` called
`self._proc.terminate()` with no `wait()`, so every spawn-and-own sidecar became a
`<defunct>` zombie for the life of the server -- observed live as
`2280741 [audiocpp_sideca] <defunct>` after a single WS connection closed. And with no
SIGKILL escalation, a sidecar stuck in a CUDA decode that does not act on SIGTERM
simply survived, holding GPU VRAM, while VoiceManager had already forgotten it. Owned
mode is the production DEFAULT (`audiocpp_sidecar_bin` has a non-empty default), so
this is the normal path, not an edge case.

The bound matters as much as the reap: `dispose()` -> `close()` runs on the event loop
(the WS handler calls it without await), so the wait/escalate must not stall the loop.
Hence close() signals synchronously and hands the wait -> SIGKILL ->
wait escalation to a short-lived reaper thread; `test_close_returns_immediately_...`
is what keeps that property honest.
"""
import os
import signal
import subprocess
import sys
import time

import authkit  # noqa: F401  (test auth env; must precede voicemanager imports)

import pytest

from voicemanager.pipeline import audiocpp_stt
from voicemanager.pipeline.audiocpp_stt import _SidecarClient

from audiocpp_fake import FakeSidecar

# Both children announce themselves on stdout once they are set up, and the spawner
# waits for that line: a SIGTERM racing the interpreter's startup would kill the
# stubborn child before it installs SIG_IGN, and the escalation assertion would then
# be measuring a startup race rather than the escalation.
_READY = "print('ready', flush=True); "
# A child that sits idle and dies on the default SIGTERM disposition, like a healthy
# sidecar between decodes.
_GRACEFUL_CHILD = "import time; " + _READY + "time.sleep(300)"
# A child that IGNORES SIGTERM, standing in for a sidecar wedged in a CUDA decode: only
# an escalation to SIGKILL gets the VRAM back.
_STUBBORN_CHILD = ("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                   + _READY + "time.sleep(300)")


def _is_zombie_or_unreaped(pid: int) -> bool:
    """True while `pid` is still an unreaped child of ours (a `<defunct>` entry)."""
    try:
        reaped_pid, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return False        # not our child any more -> somebody wait()ed it: reaped
    # A live child returns (0, 0); an exited-but-unwaited one returns (pid, status) --
    # and the only reason we could still collect it is that close() never did.
    return reaped_pid == pid


class _RealChildSpawner:
    """Replaces _SidecarClient._spawn: a REAL child process + a fake on its socket path.

    The socket side is the in-process FakeSidecar (so HELLO/READY works), while `_proc`
    is a genuine OS process -- the half this test is about. Splitting them keeps the
    suite model-free while still making terminate/wait/kill observable for real.
    """

    def __init__(self, child_source: str):
        self._child_source = child_source
        self.procs: list[subprocess.Popen] = []
        self.fakes: list[FakeSidecar] = []

    def __call__(self, bin_path: str, socket_path: str) -> subprocess.Popen:
        fake = FakeSidecar(socket_path=socket_path, flush_final="reap final")
        fake.__enter__()
        self.fakes.append(fake)
        proc = subprocess.Popen([sys.executable, "-c", self._child_source],
                                stdout=subprocess.PIPE, text=True)
        self.procs.append(proc)
        assert proc.stdout.readline().strip() == "ready", "child sidecar failed to start"
        return proc

    def cleanup(self) -> None:
        for fake in self.fakes:
            fake.stop()
        for proc in self.procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            proc.stdout.close()


@pytest.fixture
def owned_config():
    return {
        "sidecar_bin": "/nonexistent/audiocpp_sidecar",  # never executed; _spawn is stubbed
        "socket": "",                                    # empty -> owned mode (spawn our own)
        "model": "/nonexistent/model.gguf",
        "model_task": "nemotron-3.5-asr-streaming",
        "backend": "cpu",
        "device": 0,
        "threads": 4,
        "assets_root": "",
    }


@pytest.fixture
def spawn_real_child(monkeypatch):
    """Stub _spawn with a real OS child; short grace so the escalation test is quick."""
    made: list[_RealChildSpawner] = []

    def _install(child_source: str, *, grace_s: float = 0.3) -> _RealChildSpawner:
        spawner = _RealChildSpawner(child_source)
        made.append(spawner)
        monkeypatch.setattr(_SidecarClient, "_spawn",
                            lambda self, bin_path, socket_path: spawner(bin_path, socket_path))
        monkeypatch.setattr(audiocpp_stt, "_TERM_GRACE_S", grace_s)
        return spawner

    try:
        yield _install
    finally:
        for spawner in made:
            spawner.cleanup()


def _join_reap(client: _SidecarClient, timeout: float = 10.0) -> None:
    """Wait out the off-loop reaper close() started (the loop itself never waits)."""
    reaper = client._reaper
    assert reaper is not None, "close() must hand the spawned child to a reaper"
    reaper.join(timeout)
    assert not reaper.is_alive(), f"sidecar reaper did not finish within {timeout}s"


def test_close_reaps_a_spawned_sidecar_leaving_no_zombie(owned_config, spawn_real_child):
    """The spawned child is wait()ed, not just terminated -- no `<defunct>` survives."""
    spawner = spawn_real_child(_GRACEFUL_CHILD)
    client = _SidecarClient(owned_config)
    client.connect()
    proc = spawner.procs[0]
    assert proc.poll() is None, "the spawned sidecar should be running before teardown"

    client.close()
    _join_reap(client)

    assert proc.returncode is not None, "close() must wait() the sidecar it spawned"
    assert not _is_zombie_or_unreaped(proc.pid), \
        "spawned sidecar left as a <defunct> zombie after close()"


def test_close_escalates_to_sigkill_when_the_sidecar_ignores_sigterm(
        owned_config, spawn_real_child):
    """A sidecar wedged in a decode must not survive holding GPU VRAM."""
    spawner = spawn_real_child(_STUBBORN_CHILD)
    client = _SidecarClient(owned_config)
    client.connect()
    proc = spawner.procs[0]

    client.close()
    _join_reap(client)

    assert proc.returncode == -signal.SIGKILL, (
        "a sidecar that ignores SIGTERM must be SIGKILLed rather than left alive holding "
        f"VRAM (returncode was {proc.returncode})")
    assert not _is_zombie_or_unreaped(proc.pid), "the SIGKILLed sidecar was never reaped"


def test_close_returns_immediately_even_when_the_sidecar_ignores_sigterm(
        owned_config, spawn_real_child):
    """Off-loop: dispose() -> close() runs on the event loop, so it must not wait."""
    grace = 2.0
    spawner = spawn_real_child(_STUBBORN_CHILD, grace_s=grace)
    client = _SidecarClient(owned_config)
    client.connect()

    started = time.monotonic()
    client.close()
    elapsed = time.monotonic() - started

    assert elapsed < grace / 2, (
        f"close() blocked for {elapsed:.2f}s waiting on a stubborn sidecar; teardown must "
        "hand the wait off-loop")
    _join_reap(client)
    assert spawner.procs[0].returncode == -signal.SIGKILL


def test_close_is_safe_when_the_sidecar_already_exited(owned_config, spawn_real_child):
    """A sidecar that died on its own (a crash) is still reaped, without raising."""
    spawner = spawn_real_child(_GRACEFUL_CHILD)
    client = _SidecarClient(owned_config)
    client.connect()
    proc = spawner.procs[0]
    proc.kill()
    time.sleep(0.2)

    client.close()          # must not raise on a process that is already gone
    _join_reap(client)

    assert proc.returncode is not None
    assert not _is_zombie_or_unreaped(proc.pid), \
        "a self-terminated sidecar must still be reaped on close()"


def test_external_mode_close_reaps_nothing(owned_config, monkeypatch):
    """Dial-only sessions own no process, so there is nothing to reap and no thread to run."""
    def _forbidden(self, bin_path, socket_path):
        raise AssertionError("external mode must never spawn a sidecar")

    monkeypatch.setattr(_SidecarClient, "_spawn", _forbidden)
    with FakeSidecar(flush_final="external final") as fake:
        client = _SidecarClient(owned_config | {"socket": fake.socket_path})
        client.connect()
        client.close()
        assert client._reaper is None, "external mode must not reap a process it does not own"
