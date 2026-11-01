"""The VoXtream sidecar watcher, model-free.

Nothing used to watch the sidecar after startup: when it died, every TTS request
failed until a person restarted the server, and /health kept saying ok. The engine
now has a ``ready`` flag read from the child's exit status and a ``watch``
coroutine that starts the sidecar again when it dies. The sidecar is the fake
process from tests/voxtream_fake_sidecar.py, started with the test interpreter.
"""
import asyncio
import inspect
import os
import socket
import time
from pathlib import Path

import pytest

import authkit  # test auth env; must be imported before voicemanager
from fastapi.testclient import TestClient

import voicemanager.app as app_module
import voicemanager.models as models
import voicemanager.pipeline.voxtream_tts as voxtream_tts
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


def _wait(predicate, timeout: float, step: float = 0.02) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition not met in time")
        time.sleep(step)


def test_shutdown_stops_the_watcher_first_and_interrupts_a_restart_in_flight(
        fake_sidecar, monkeypatch, caplog):
    settings = _settings()
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(app_module, "load_models", lambda s: {
        "stt_engine_type": "qwen", "device": "cpu", "qwen_model": object()})
    monkeypatch.setattr(voxtream_tts, "_WATCH_POLL_S", 0.05)
    monkeypatch.setattr(voxtream_tts, "_WATCH_RETRY_S", 0.1)
    caplog.set_level("INFO", logger="voicemanager.pipeline.voxtream_tts")
    with TestClient(app_module.create_app()) as client:
        engine = client.app.state.tts_engine
        first = engine.sidecar.process
        # The respawn will take 5 s to answer /health: the start waits inside it.
        fake_sidecar("slow", VOXTREAM_FAKE_DELAY_S=5)
        _kill(first)
        _wait(lambda: engine.sidecar.alive, timeout=10)
        second = engine.sidecar.process
        assert second.pid != first.pid and not engine.ready
        shutdown_began = time.time()
    shutdown_took = time.time() - shutdown_began
    assert shutdown_took < 3, shutdown_took  # not the 5 s the respawn needed
    _wait(lambda: second.poll() is not None, timeout=5)
    assert first.poll() is not None
    assert engine.sidecar.process is None and not engine.ready
    starts_after = [r for r in caplog.records
                    if "Starting VoXtream sidecar" in r.getMessage()
                    and r.created > shutdown_began]
    assert starts_after == []
    time.sleep(0.3)  # a watcher still alive would have started one by now
    assert engine.sidecar.process is None


def test_disposed_engine_refuses_to_start_again(fake_sidecar):
    result, engine = _load()
    _release(result, engine)
    with pytest.raises(voxtream_tts.VoxtreamStartupError, match="closing"):
        engine.sidecar.start()
    assert not engine.ready


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_the_sidecar_runs_in_its_own_session(fake_sidecar):
    # A service stop or a Ctrl-C signals the server's process group; the sidecar
    # must not be in it, or it dies ahead of the lifespan and the watcher starts a
    # new one that the shutdown then has to kill.
    result, engine = _load()
    try:
        pid = engine.sidecar.process.pid
        assert os.getsid(pid) == pid
        assert os.getsid(pid) != os.getsid(os.getpid())
    finally:
        _release(result, engine)


def test_the_systemd_units_use_killmode_mixed_so_the_server_stops_the_sidecar():
    for unit in ("system", "user"):
        text = (REPO_ROOT / "contrib" / "systemd" / unit / "voicemanager.service").read_text()
        assert "KillMode=mixed" in text, unit
