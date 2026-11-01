"""The VoXtream sidecar watcher, model-free.

Nothing used to watch the sidecar after startup: when it died, every TTS request
failed until a person restarted the server, and /health kept saying ok. The engine
now has a ``ready`` flag read from the child's exit status and a ``watch``
coroutine that starts the sidecar again when it dies. The sidecar is the fake
process from tests/voxtream_fake_sidecar.py, started with the test interpreter.
"""
import asyncio
import inspect
import socket
import time

import authkit  # test auth env; must be imported before voicemanager

import voicemanager.models as models
from voicemanager.pipeline.voxtream_tts import VoxtreamSidecar, VoxtreamTTSEngine
from test_voxtream_startup import _settings, fake_sidecar  # noqa: F401 (fixture)


async def _until(predicate, timeout: float, step: float = 0.02) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(step)


def _load(**overrides):
    result = models._load_voxtream_tts(_settings(**overrides))
    return result, result["tts_engine"]


def _release(result, engine) -> None:
    engine.dispose()
    result["tts_executor"].shutdown(wait=True)


def _kill(proc) -> None:
    proc.kill()
    proc.wait(timeout=5)


def test_ready_reads_the_exit_status_and_makes_no_request():
    for prop in (VoxtreamTTSEngine.ready, VoxtreamSidecar.ready, VoxtreamSidecar.alive):
        source = inspect.getsource(prop.fget)
        assert "urllib" not in source and "socket" not in source


async def test_watcher_respawns_a_killed_sidecar(fake_sidecar):
    result, engine = _load()
    watcher = None
    try:
        assert engine.ready
        first = engine.sidecar.process
        watcher = asyncio.create_task(engine.watch(poll_interval=0.05, retry_delay=0.1))
        _kill(first)
        assert not engine.ready
        # The flag stays false while the new process is spawned but not yet ready.
        await _until(lambda: engine.sidecar.alive, timeout=10)
        assert engine.sidecar.process.pid != first.pid
        await _until(lambda: engine.ready, timeout=10)
        with socket.create_connection(("127.0.0.1", engine.sidecar.port), timeout=1):
            pass
        assert not watcher.done()
    finally:
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        _release(result, engine)
    assert not engine.ready


async def test_a_failed_restart_is_logged_and_the_retry_runs_until_it_works(fake_sidecar, caplog):
    result, engine = _load()
    watcher = None
    try:
        first = engine.sidecar.process
        watcher = asyncio.create_task(engine.watch(poll_interval=0.05, retry_delay=0.1))
        fake_sidecar("exit")  # every respawn now exits with code 3 at once
        _kill(first)
        def failures():
            return [r for r in caplog.records if "restart failed" in r.getMessage()]

        with caplog.at_level("ERROR", logger="voicemanager.pipeline.voxtream_tts"):
            await _until(lambda: len(failures()) >= 2, timeout=10)  # a second try happened
        failures = failures()
        assert "VM_VOXTREAM_PY" in failures[0].getMessage()
        assert not watcher.done()
        assert not engine.ready
        fake_sidecar("ready")
        await _until(lambda: engine.ready, timeout=10)
        assert engine.sidecar.alive
    finally:
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        _release(result, engine)


async def test_an_unexpected_restart_error_is_logged_with_traceback_before_the_retry(
        fake_sidecar, caplog, monkeypatch):
    result, engine = _load()
    watcher = None
    real_start = engine.sidecar.start
    calls = []

    def start_once_broken():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("spawn machinery broke")
        real_start()

    try:
        first = engine.sidecar.process
        monkeypatch.setattr(engine.sidecar, "start", start_once_broken)
        watcher = asyncio.create_task(engine.watch(poll_interval=0.05, retry_delay=0.1))
        with caplog.at_level("ERROR", logger="voicemanager.pipeline.voxtream_tts"):
            _kill(first)
            await _until(lambda: engine.ready, timeout=10)
        failures = [r for r in caplog.records if "restart failed" in r.getMessage()]
        assert len(failures) == 1 and failures[0].exc_info is not None
        assert "spawn machinery broke" in caplog.text
        assert len(calls) == 2
        assert not watcher.done()
    finally:
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
        _release(result, engine)
