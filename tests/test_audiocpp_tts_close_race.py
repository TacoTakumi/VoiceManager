"""A disconnect that beats the TTS warm-up must not leak a sidecar.

The WS layer warms each session's TTS sidecar in a background worker
(asyncio.to_thread(engine.connect)) and tears the session down with a
non-awaited dispose(). When teardown wins the race - dispose() completes before
the warm-up worker thread ever starts - the late connect() used to find a
pristine client and spawn a fresh sidecar that no code path would ever reap: an
orphan process holding GPU memory. dispose() now marks the client closed
SYNCHRONOUSLY (before handing the socket/process teardown to its daemon
thread), and a post-close connect is a no-op: no process spawned, no socket
dialed, no private dir minted.

Model-free: the spawn/dial internals are monkeypatched to record any attempt,
so nothing here needs a binary, a GPU, or a listener.

The closed latch belongs ONLY to dispose()/close(): a connect() that raised
still tears its own debris down via the same path but stays reconnectable -
the WS layer's per-request connect() retry depends on that.
"""
from __future__ import annotations

import pytest

from voicemanager.pipeline.audiocpp_tts import (
    AudioCppTTSEngine,
    SidecarError,
    _TtsSidecarClient,
)


def _recording_engine(monkeypatch, attempts: list[str]) -> AudioCppTTSEngine:
    """An owned-mode engine whose spawn/dial record instead of doing real work."""
    engine = AudioCppTTSEngine(
        {"sidecar_bin": "/nonexistent/vm-tts-sidecar", "model": "fake.gguf"})
    monkeypatch.setattr(
        _TtsSidecarClient, "_spawn",
        lambda self, bin_path, socket_path: attempts.append("spawn"))
    monkeypatch.setattr(
        _TtsSidecarClient, "_dial",
        lambda self, socket_path, *, deadline_s: attempts.append("dial"))
    return engine


def test_dispose_then_late_connect_spawns_nothing(monkeypatch) -> None:
    """The pinned race: teardown first, warm-up connect after - a no-op."""
    attempts: list[str] = []
    engine = _recording_engine(monkeypatch, attempts)

    engine.dispose()   # WS finally-block, non-awaited
    engine.connect()   # the warm-up worker body, arriving late: must be a no-op

    assert attempts == [], "a post-dispose connect must not spawn or dial"
    client = engine._client
    assert client._proc is None
    assert client._sock is None
    assert client._owned_dir is None, "no private socket dir may be minted post-close"


def test_dispose_marks_closed_before_its_teardown_thread_runs(monkeypatch) -> None:
    """dispose() must latch the closed state on the CALLING thread: the daemon
    teardown thread it starts may be arbitrarily delayed, and the warm-up
    connect can run in that gap."""
    attempts: list[str] = []
    engine = _recording_engine(monkeypatch, attempts)
    started: list[object] = []

    class _InertThread:
        def __init__(self, *args, **kwargs) -> None:
            started.append(self)

        def start(self) -> None:
            pass  # never runs close() - the worst-case delayed teardown thread

    import voicemanager.pipeline.audiocpp_tts as mod
    monkeypatch.setattr(mod.threading, "Thread", _InertThread)

    engine.dispose()
    assert started, "dispose() must still hand teardown to a thread"
    engine.connect()  # teardown thread has NOT run yet; must still be a no-op

    assert attempts == [], "connect must see the closed latch set by dispose() itself"
    assert engine._client._sock is None


def test_a_failed_connect_stays_reconnectable() -> None:
    """Only dispose()/close() latch closed. A connect() that raised must leave
    the client able to try again (the WS layer retries connect per request)."""
    engine = AudioCppTTSEngine(
        {"sidecar_bin": "/nonexistent/vm-tts-sidecar", "model": "fake.gguf"})
    with pytest.raises(SidecarError):
        engine.connect()  # spawn of a nonexistent binary fails fast
    # A second attempt must RETRY (and fail the same way), not silently no-op.
    with pytest.raises(SidecarError):
        engine.connect()
