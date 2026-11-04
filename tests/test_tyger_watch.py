"""The Tyger sidecar watcher, model-free.

The sidecar is one shared child process. When it exits the server keeps serving:
the watcher sees the exit, tells its listeners the sidecar is lost, starts it again
with a backoff between failed tries, and tells them it is back. The sidecar is the
fake process from tests/tyger_fake_sidecar.py, started with the test interpreter.
"""
import asyncio
import inspect
import os
import socket
import time

import pytest

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)
from fastapi.testclient import TestClient

import voicemanager.app as app_module
import voicemanager.pipeline.tyger.sidecar as sidecar_module
from voicemanager.pipeline.tyger.sidecar import TygerSidecar, TygerStartupError
from test_tyger_startup import _settings, _stub_engines, fake_sidecar  # noqa: F401 (fixture)


async def _until(predicate, timeout: float, step: float = 0.02) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(step)


def _wait(predicate, timeout: float, step: float = 0.02) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition not met in time")
        time.sleep(step)


def _started() -> TygerSidecar:
    sidecar = TygerSidecar.from_settings(_settings(), device="cpu")
    sidecar.start()
    return sidecar


def _kill(proc) -> None:
    proc.kill()
    proc.wait(timeout=5)


async def _cancel(task) -> None:
    if task is not None:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_up_reads_the_exit_status_and_makes_no_request():
    for prop in (TygerSidecar.up, TygerSidecar.alive):
        source = inspect.getsource(prop.fget)
        assert "urllib" not in source and "socket" not in source


async def test_a_killed_sidecar_is_seen_lost_then_restarted_and_back(fake_sidecar):
    sidecar = _started()
    events = []
    remove = sidecar.add_listener(lambda up, generation: events.append((up, generation)))
    watcher = None
    try:
        assert sidecar.up and sidecar.generation == 1
        first = sidecar.process
        watcher = asyncio.create_task(sidecar.watch(poll_interval=0.05, retry_delay=0.1))
        _kill(first)
        assert not sidecar.up  # at once: read from the exit status
        await _until(lambda: (False, 1) in events, timeout=5)
        await _until(lambda: sidecar.up, timeout=10)
        assert sidecar.process.pid != first.pid
        assert sidecar.generation == 2
        await _until(lambda: (True, 2) in events, timeout=5)
        assert events == [(False, 1), (True, 2)]
        with socket.create_connection(("127.0.0.1", sidecar.port), timeout=1):
            pass
        assert sidecar.is_ready()  # /health answers 200 again
        assert not watcher.done()
        remove()
        _kill(sidecar.process)
        await _until(lambda: sidecar.generation == 3, timeout=10)
        assert events == [(False, 1), (True, 2)]  # removed listeners hear nothing more
    finally:
        await _cancel(watcher)
        sidecar.close()
    assert sidecar.process is None and not sidecar.up


async def test_wait_lost_and_wait_up_wake_on_the_change(fake_sidecar):
    sidecar = _started()
    watcher = None
    try:
        watcher = asyncio.create_task(sidecar.watch(poll_interval=0.05, retry_delay=0.1))
        lost = asyncio.create_task(sidecar.wait_lost())
        await asyncio.sleep(0.1)
        assert not lost.done()
        _kill(sidecar.process)
        assert await asyncio.wait_for(lost, 5) == 1  # the generation that was lost
        assert await asyncio.wait_for(sidecar.wait_up(), 10) == 2
        assert sidecar.up
    finally:
        await _cancel(watcher)
        sidecar.close()


async def test_a_failing_restart_backs_off_is_logged_and_retries_until_it_works(
        fake_sidecar, caplog):
    sidecar = _started()
    events = []
    sidecar.add_listener(lambda up, generation: events.append(up))
    watcher = None
    try:
        watcher = asyncio.create_task(
            sidecar.watch(poll_interval=0.05, retry_delay=0.05, max_retry_delay=0.2))
        fake_sidecar("exit")  # every respawn now exits with code 3 at once
        _kill(sidecar.process)

        def failures():
            return [r for r in caplog.records if "restart failed" in r.getMessage()]

        with caplog.at_level("WARNING", logger="voicemanager.pipeline.tyger.sidecar"):
            await _until(lambda: len(failures()) >= 4, timeout=10)
        messages = [r.getMessage() for r in failures()]
        assert "VM_TYGER_PY" in messages[0]
        # The wait between tries doubles up to the cap.
        assert "next try in 0.05 s" in messages[0]
        assert "next try in 0.1 s" in messages[1]
        assert "next try in 0.2 s" in messages[2]
        assert "next try in 0.2 s" in messages[3]
        gone = [r for r in caplog.records if "is gone" in r.getMessage()]
        assert len(gone) == 1  # one death; the failed tries are not new deaths
        assert events == [False]
        assert not watcher.done() and not sidecar.up
        fake_sidecar("ready")
        await _until(lambda: sidecar.up, timeout=10)
        await _until(lambda: events == [False, True], timeout=5)
    finally:
        await _cancel(watcher)
        sidecar.close()


async def test_an_unexpected_restart_error_is_logged_with_traceback(
        fake_sidecar, caplog, monkeypatch):
    sidecar = _started()
    real_start = sidecar.start
    calls = []

    def start_once_broken():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("spawn machinery broke")
        real_start()

    watcher = None
    try:
        first = sidecar.process
        monkeypatch.setattr(sidecar, "start", start_once_broken)
        watcher = asyncio.create_task(sidecar.watch(poll_interval=0.05, retry_delay=0.05))
        with caplog.at_level("ERROR", logger="voicemanager.pipeline.tyger.sidecar"):
            _kill(first)
            await _until(lambda: sidecar.up, timeout=10)
        failed = [r for r in caplog.records if "restart failed" in r.getMessage()]
        assert len(failed) == 1 and failed[0].exc_info is not None
        assert "spawn machinery broke" in caplog.text
        assert len(calls) == 2
    finally:
        await _cancel(watcher)
        sidecar.close()


async def test_a_listener_that_raises_does_not_stop_the_watcher(fake_sidecar, caplog):
    sidecar = _started()

    def broken(up, generation):
        raise RuntimeError("listener broke")

    heard = []
    sidecar.add_listener(broken)
    sidecar.add_listener(lambda up, generation: heard.append(up))
    watcher = None
    try:
        watcher = asyncio.create_task(sidecar.watch(poll_interval=0.05, retry_delay=0.05))
        _kill(sidecar.process)
        await _until(lambda: heard == [False, True], timeout=10)
        assert "listener broke" in caplog.text
        assert not watcher.done()
    finally:
        await _cancel(watcher)
        sidecar.close()


def test_close_leaves_no_process_and_refuses_a_later_start(fake_sidecar):
    sidecar = _started()
    proc = sidecar.process
    sidecar.close()
    _wait(lambda: proc.poll() is not None, timeout=5)
    assert sidecar.process is None and not sidecar.up
    with pytest.raises(TygerStartupError, match="closing"):
        sidecar.start()
    assert sidecar.process is None


def test_the_sidecar_runs_in_its_own_session(fake_sidecar):
    # A Ctrl-C or a service stop signals the server's process group; the sidecar
    # must not die ahead of the lifespan, or the watcher starts one the shutdown
    # then has to kill.
    sidecar = _started()
    try:
        pid = sidecar.process.pid
        assert os.getsid(pid) == pid
        assert os.getsid(pid) != os.getsid(os.getpid())
    finally:
        sidecar.close()


def test_shutdown_cancels_the_watcher_first_and_interrupts_a_restart_in_flight(
        fake_sidecar, monkeypatch, caplog):
    _stub_engines(monkeypatch, _settings())
    monkeypatch.setattr(sidecar_module, "_WATCH_POLL_S", 0.05)
    monkeypatch.setattr(sidecar_module, "_WATCH_RETRY_S", 0.05)
    caplog.set_level("INFO", logger="voicemanager.pipeline.tyger.sidecar")
    with TestClient(app_module.create_app()) as client:
        sidecar = client.app.state.turn_engine_sidecar
        watcher = client.app.state.turn_engine_watcher
        first = sidecar.process
        at_close = {}
        real_close = sidecar.close

        def close_recording():
            at_close["watcher_done"] = watcher.done()
            real_close()
        monkeypatch.setattr(sidecar, "close", close_recording)
        # The respawn takes 5 s to answer /health: the start waits inside it.
        fake_sidecar("slow", TYGER_FAKE_DELAY_S=5)
        _kill(first)
        _wait(lambda: sidecar.alive, timeout=10)
        second = sidecar.process
        assert second.pid != first.pid and not sidecar.up
        shutdown_began = time.time()
    assert time.time() - shutdown_began < 3  # not the 5 s the respawn needed
    _wait(lambda: second.poll() is not None, timeout=5)
    assert sidecar.process is None and not sidecar.up
    assert at_close["watcher_done"] is True
    assert watcher.cancelled()
    after = [r for r in caplog.records if r.created > shutdown_began]
    assert [r for r in after if "Starting Tyger sidecar" in r.getMessage()] == []
    time.sleep(0.3)  # a watcher still alive would have started one by now
    assert sidecar.process is None
