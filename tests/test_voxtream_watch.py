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
