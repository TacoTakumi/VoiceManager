"""Bounded concurrent-sidecar admission for the audio.cpp engine.

Model-free and server-free (the CLAUDE.md invariant): no GPU, no native binary, no
model. The real spawn is replaced by a stub that stands `tests/audiocpp_fake.py` up on
the socket path the pipeline asked for, so these assert the *admission policy*, not the
sidecar's inference or its actual VRAM footprint.

The defect this pins down is unbounded sidecar capacity. One sidecar per
session is deliberate -- it is what makes the fault isolation real, since a
process-fatal audio.cpp fault can then only reach one session -- but nothing bounded how
many could exist at once, and every one of them loads an f16 GGUF plus a CUDA context
onto the SAME `VM_AUDIOCPP_DEVICE` (measured ~1.8 GiB each). So the Nth concurrent
session used to fail by exhausting the GPU: `open_streaming_session` throws, the sidecar
replies ERROR `load_failed`, and the client is told -- via close 4001, "do not retry,
an operator must fix your config" -- something that is flatly untrue. The deployment is
configured correctly and is merely full, which clears on its own the moment a session
ends. Capacity exhaustion was indistinguishable from a bad model path.

Now the cap is explicit and checked BEFORE the spawn, so the failure is named
(`at_capacity`) and lands on the retriable side of the close-code taxonomy (close 4002,
`stt_backend_unready`, "retry with backoff") instead of the permanent one.
"""
import threading

import authkit  # noqa: F401  (test auth env; must be imported before voicemanager)

import pytest

from voicemanager.pipeline.audiocpp_stt import SidecarError, SidecarSlots, _SidecarClient

from audiocpp_fake import FakeSidecar


class _StubProc:
    """Stands in for subprocess.Popen: records terminate()/wait() without a real process."""

    def __init__(self):
        self.terminated = False
        self.pid = -1

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout=None):
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


def _capped_config(slots: SidecarSlots) -> dict:
    """Production OWNED shape (a binary, no socket) sharing one admission limiter."""
    return {
        "sidecar_bin": "/nonexistent/audiocpp_sidecar",  # never executed; _spawn is stubbed
        "socket": "",
        "model": "/nonexistent/model.gguf",
        "model_task": "nemotron-3.5-asr-streaming",
        "backend": "cpu",
        "device": 0,
        "threads": 4,
        "assets_root": "",
        "slots": slots,
    }


def _close_and_settle(client: _SidecarClient) -> None:
    """close(), then wait out the reaper that owns the slot release (see close())."""
    client.close()
    reaper = client._reaper
    if reaper is not None:
        reaper.join(timeout=5.0)
        assert not reaper.is_alive(), "sidecar reaper thread did not finish"


def test_sessions_within_the_cap_are_admitted_and_unaffected(spawner):
    """The cap bounds only what is OVER it: every session up to the limit is normal."""
    slots = SidecarSlots(3)
    clients = [_SidecarClient(_capped_config(slots)) for _ in range(3)]
    try:
        for client in clients:
            client.connect()
        assert all(c.is_connected for c in clients)
        assert len(spawner.paths) == 3, "each admitted session spawns its own sidecar"
        assert slots.in_use == 3
        # Still fully functional -- admission control must not disturb the wire protocol.
        for client in clients:
            client.send_flush()
            mtype, _ = client._recv_frame_blocking()
            assert mtype == 0x06, f"admitted session broke: got 0x{mtype:02x}"
    finally:
        for client in clients:
            client.close()


def test_session_over_the_cap_is_refused_with_a_defined_retriable_code(spawner):
    """The whole point: full is reported as full, and as RETRIABLE -- not as load_failed.

    Before the cap the (N+1)th session was refused by the GPU running out of memory, which
    surfaced as `load_failed` -> close 4001 `stt_backend_unavailable`, i.e. "your server
    is misconfigured, do not retry". Both halves of that were wrong.
    """
    slots = SidecarSlots(2)
    admitted = [_SidecarClient(_capped_config(slots)) for _ in range(2)]
    try:
        for client in admitted:
            client.connect()

        over = _SidecarClient(_capped_config(slots))
        with pytest.raises(SidecarError) as excinfo:
            over.connect()

        assert excinfo.value.code == "at_capacity", \
            "capacity exhaustion must be named, not disguised as a bad model path"
        assert excinfo.value.code != "load_failed"
        assert excinfo.value.is_permanent is False, \
            "a full server is transient (close 4002, retry) -- not an operator config fault"
    finally:
        for client in admitted:
            client.close()


def test_refused_session_leaves_nothing_behind(spawner):
    """The cap is checked BEFORE the spawn: no process, no socket dir, no consumed slot."""
    slots = SidecarSlots(1)
    first = _SidecarClient(_capped_config(slots))
    try:
        first.connect()
        assert len(spawner.paths) == 1

        over = _SidecarClient(_capped_config(slots))
        with pytest.raises(SidecarError):
            over.connect()

        assert len(spawner.paths) == 1, "a refused session must not spawn a sidecar"
        assert not over.is_connected
        assert over._owned_dir is None, "a refused session must not leave a private dir"
        # And it must not have consumed the slot it was denied -- otherwise one refusal
        # would permanently shrink capacity.
        assert slots.in_use == 1
    finally:
        first.close()


def test_closing_a_session_frees_its_slot_for_the_next_one(spawner):
    """Capacity is a live count, not a high-water mark: a disconnect readmits."""
    slots = SidecarSlots(1)
    first = _SidecarClient(_capped_config(slots))
    first.connect()
    assert slots.in_use == 1

    blocked = _SidecarClient(_capped_config(slots))
    with pytest.raises(SidecarError) as excinfo:
        blocked.connect()
    assert excinfo.value.code == "at_capacity"

    _close_and_settle(first)
    assert slots.in_use == 0, "closing a session must release its capacity slot"

    after = _SidecarClient(_capped_config(slots))
    try:
        after.connect()   # the same config that was refused a moment ago now succeeds
        assert after.is_connected
    finally:
        after.close()


def test_slot_is_held_until_the_sidecar_is_actually_reaped(spawner):
    """The slot tracks the PROCESS, not the socket -- VRAM is not free until it exits.

    close() only signals; the wait/SIGKILL escalation runs in a reaper thread. If
    the slot were released at close() a replacement session could spawn while the old
    sidecar still held its GGUF on the GPU, which is exactly the over-subscription the
    cap exists to prevent.
    """
    slots = SidecarSlots(1)
    client = _SidecarClient(_capped_config(slots))
    client.connect()

    gate = threading.Event()
    original_wait = spawner.procs[0].wait
    spawner.procs[0].wait = lambda timeout=None: (gate.wait(5.0), original_wait())[1]

    client.close()
    assert slots.in_use == 1, "slot must stay held while the sidecar is still exiting"
    gate.set()
    client._reaper.join(timeout=5.0)
    assert slots.in_use == 0, "slot must be released once the sidecar is reaped"


def test_unbounded_when_the_cap_is_zero(spawner):
    """0 opts out -- for an operator who has measured their own ceiling."""
    slots = SidecarSlots(0)
    clients = [_SidecarClient(_capped_config(slots)) for _ in range(4)]
    try:
        for client in clients:
            client.connect()   # no refusal at any count
        assert all(c.is_connected for c in clients)
        assert slots.limit == 0
    finally:
        for client in clients:
            client.close()


def test_external_mode_is_not_capped(monkeypatch):
    """The cap bounds sidecars WE spawn; an operator-managed listener is not ours to count.

    Its cost is fixed and already paid, and it is single-session by construction (the
    sidecar's own ERROR `busy`), so applying our spawn budget to it would bound nothing
    and could refuse the one warm sidecar the live measurements depend on.
    """
    def _forbidden(self, bin_path, socket_path):
        raise AssertionError("external mode must never spawn a sidecar")

    monkeypatch.setattr(_SidecarClient, "_spawn", _forbidden)
    slots = SidecarSlots(1)
    with FakeSidecar(flush_final="external final") as fake:
        config = _capped_config(slots) | {"socket": fake.socket_path}
        client = _SidecarClient(config)
        try:
            client.connect()
            assert client.is_connected
            assert slots.in_use == 0, "a dialled external sidecar must not consume a slot"
        finally:
            client.close()


def test_no_limiter_configured_means_unbounded(spawner):
    """A config dict without a limiter (the shape before the cap) still works, uncapped.

    Production always gets one from _load_audiocpp_models; this keeps hand-built configs
    (the other audio.cpp tests) working rather than failing closed on a missing key.
    """
    config = _capped_config(SidecarSlots(1))
    config.pop("slots")
    clients = [_SidecarClient(dict(config)) for _ in range(3)]
    try:
        for client in clients:
            client.connect()
        assert all(c.is_connected for c in clients)
    finally:
        for client in clients:
            client.close()
