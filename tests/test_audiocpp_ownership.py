"""Sidecar ownership + per-session isolation for the audio.cpp engine.

Model-free and server-free (the CLAUDE.md invariant): no GPU, no native binary, no
model. The real spawn is replaced by a stub that stands `tests/audiocpp_fake.py` up on
exactly the socket path the pipeline asked for, so these assert the *ownership model*
the pipeline implements, not the sidecar's inference.

This pins down three old defects. Before the ownership model the pipeline
dialled ONE fixed `VM_AUDIOCPP_SOCKET` first and spawned only if that was refused, which
gave three failure modes at once: two sessions shared one sidecar that serves a single
session at a time; the session that happened to spawn owned the process and killed
it on disconnect out from under a session still using it; and two sessions that
both found no listener spawned on the SAME fixed path, the second unlinking and
rebinding over the first, orphaning a model-loaded sidecar on the GPU behind an
unreachable inode.

The ownership model has exactly two modes, chosen by whether `socket` is configured,
so none of those three states is reachable:

  * OWNED (production default, `socket` empty) -- the pipeline mints a PRIVATE socket
    path of its own, spawns its own sidecar there and owns it exclusively. Two sessions
    never share a process, never share a path, and each may terminate its own sidecar
    without touching anyone else's.
  * EXTERNAL (`socket` set) -- an operator-managed listener. Dial only: never spawn,
    never terminate. There is no ownership to get wrong.

Whether two concurrent /ws/stream sessions really do transcribe against the REAL
sidecar stays a live claim (scripts/audiocpp_two_session_repro.py) -- the fake must
never be the evidence for it.
"""
import os
import time

import authkit  # noqa: F401  (test auth env; must precede voicemanager imports)

import pytest

from voicemanager.pipeline.audiocpp_stt import SidecarError, _SidecarClient

from audiocpp_fake import FakeSidecar


class _StubProc:
    """Stands in for subprocess.Popen: records terminate()/wait() without a real process.

    close() reaps what it terminates, so the stub answers wait() too -- as a
    well-behaved sidecar would, exiting on SIGTERM. Whether the reap ITSELF works is a
    real-child claim, not a stub one: tests/test_audiocpp_reap.py.
    """

    def __init__(self):
        self.terminated = False
        self.waited = False
        self.pid = -1

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True
        return 0

    def kill(self) -> None:
        self.terminated = True

    def poll(self):
        return None if not self.terminated else 0


class _SpawnRecorder:
    """Replaces _SidecarClient._spawn: stands a fake sidecar up on the requested path."""

    def __init__(self):
        self.paths: list[str] = []
        self.procs: list[_StubProc] = []
        self.fakes: list[FakeSidecar] = []

    def __call__(self, bin_path: str, socket_path: str) -> _StubProc:
        self.paths.append(socket_path)
        fake = FakeSidecar(socket_path=socket_path, flush_final="spawned final")
        fake.__enter__()
        self.fakes.append(fake)
        proc = _StubProc()
        self.procs.append(proc)
        return proc

    def stop_all(self) -> None:
        for fake in self.fakes:
            fake.stop()


@pytest.fixture
def spawner(monkeypatch):
    rec = _SpawnRecorder()
    monkeypatch.setattr(_SidecarClient, "_spawn",
                        lambda self, bin_path, socket_path: rec(bin_path, socket_path))
    try:
        yield rec
    finally:
        rec.stop_all()


def _connect_when_free(client: _SidecarClient, *, tries: int = 50, gap: float = 0.02) -> None:
    """Connect, tolerating the brief ERROR `busy` window after a peer session closes."""
    for _ in range(tries):
        try:
            client.connect()
            return
        except SidecarError as exc:
            if exc.code != "busy":
                raise
            time.sleep(gap)
    raise AssertionError("sidecar stayed busy after the previous session closed")


def _owned_config() -> dict:
    """Production shape: a binary configured, NO socket -> the pipeline mints its own."""
    return {
        "sidecar_bin": "/nonexistent/audiocpp_sidecar",  # never executed; _spawn is stubbed
        "socket": "",
        "model": "/nonexistent/model.gguf",
        "model_task": "nemotron-3.5-asr-streaming",
        "backend": "cpu",
        "device": 0,
        "threads": 4,
        "assets_root": "",
    }


def test_owned_mode_spawns_on_a_private_socket_path(spawner):
    """A session with no configured socket spawns its own sidecar on a path it minted."""
    client = _SidecarClient(_owned_config())
    try:
        client.connect()
        assert client.is_connected
        assert len(spawner.paths) == 1, "owned mode must spawn exactly one sidecar"
        # The path is private to this client -- not the old fixed /tmp default.
        assert spawner.paths[0] != "/tmp/audiocpp-sidecar.sock"
        assert os.path.exists(spawner.paths[0])
    finally:
        client.close()


def test_two_owned_sessions_get_separate_sidecars_on_separate_paths(spawner):
    """Concurrent sessions never share a process and never race on one path."""
    a = _SidecarClient(_owned_config())
    b = _SidecarClient(_owned_config())
    try:
        a.connect()
        b.connect()   # must NOT reuse a's sidecar, and must NOT rebind a's path
        assert a.is_connected and b.is_connected
        assert len(spawner.paths) == 2, "each session spawns its own sidecar"
        assert spawner.paths[0] != spawner.paths[1], "sessions must not share a socket path"
        assert spawner.procs[0] is not spawner.procs[1]
    finally:
        a.close()
        b.close()


def test_closing_one_owned_session_leaves_the_other_alive(spawner):
    """A session terminates only the sidecar it spawned -- never a peer's."""
    a = _SidecarClient(_owned_config())
    b = _SidecarClient(_owned_config())
    try:
        a.connect()
        b.connect()
        a.close()

        assert spawner.procs[0].terminated, "a session must terminate its OWN sidecar"
        assert not spawner.procs[1].terminated, \
            "closing one session must not kill another session's sidecar"
        # b's connection is untouched: it still round-trips a FLUSH -> FINAL.
        b.send_flush()
        mtype, _ = b._recv_frame_blocking()
        assert mtype == 0x06, f"peer session broke after a sibling closed: got 0x{mtype:02x}"
    finally:
        a.close()
        b.close()


def test_owned_socket_path_is_cleaned_up_on_close(spawner):
    """No stale socket (or its private dir) survives the session that minted it."""
    client = _SidecarClient(_owned_config())
    client.connect()
    path = spawner.paths[0]
    assert os.path.exists(path)
    client.close()
    assert not os.path.exists(path), "owned socket path must not be left behind"
    assert not os.path.exists(os.path.dirname(path)), "private socket dir must be removed"


def test_external_mode_never_spawns_and_never_terminates(monkeypatch):
    """A configured socket means an operator-managed listener: dial only, no ownership."""
    def _forbidden(self, bin_path, socket_path):
        raise AssertionError("external mode must never spawn a sidecar")

    monkeypatch.setattr(_SidecarClient, "_spawn", _forbidden)
    with FakeSidecar(flush_final="external final") as fake:
        config = _owned_config() | {"socket": fake.socket_path}
        first = _SidecarClient(config)
        first.connect()
        assert first.is_connected
        first.close()

        # The listener we merely dialled is still there for the next session: closing a
        # session must not tear down a sidecar it does not own. An external sidecar is
        # still SINGLE-SESSION, so the next session is admitted once the previous one is
        # released -- ERROR `busy` in the moment right after a close is the sidecar's
        # defined answer, not a teardown, hence the short retry.
        second = _SidecarClient(config)
        try:
            _connect_when_free(second)
            assert second.is_connected
        finally:
            second.close()


def test_no_socket_and_no_binary_raises_a_defined_error():
    """A sidecar that can be neither dialled nor spawned fails fast and explicitly."""
    config = _owned_config() | {"socket": "", "sidecar_bin": ""}
    client = _SidecarClient(config)
    with pytest.raises(SidecarError) as excinfo:
        client.connect()
    assert excinfo.value.code == "no_sidecar"
