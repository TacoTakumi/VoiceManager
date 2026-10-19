"""A sidecar that cannot be SPAWNED is a config failure too, and leaks nothing.

Model-free and server-free (the CLAUDE.md invariant): no GPU, no native binary, no model.
Critically, the first tests here deliberately do NOT stub `_spawn` -- every existing spawn
test does (tests/test_audiocpp_ownership.py, tests/test_audiocpp_reap.py), which is exactly
why this defect shipped: the one line the stub replaced, `subprocess.Popen`, was the only
unguarded failure point left on the default path.

An earlier fix gave the DIAL path a defined code -- `_dial` giving up now raises
`SidecarError` rather than a bare `OSError`, so an unreachable socket closes 4001
`stt_backend_unavailable` instead of a generic `server_error` 1011 plus a bug traceback.
It left the SPAWN path unguarded, so a missing or unexecutable binary still raised
`FileNotFoundError`/`PermissionError` into the WS catch-all.

That is the MOST LIKELY misconfiguration, not an edge case: `VM_AUDIOCPP_SIDECAR_BIN`
defaults to `sidecar/build/bin/audiocpp_sidecar`, a RELATIVE path to a git-ignored build
artifact, so a fresh clone -- or a server started from any directory other than the repo
root -- hits it on the very first audio frame. It now raises `SidecarError("spawn_failed")`
naming the path it actually resolved, and reaches the client through the same defined
4001 path as `load_failed`.

The second half is a second connect() leak. The earlier teardown guard wrapped
ONLY `_handshake()`, so a spawn that succeeded followed by a dial that failed left `_proc`
and `_owned_dir` set with no teardown at all. The guard now covers the whole
spawn + dial + handshake sequence: no connect() failure path leaks a process, an fd, or a
temp dir. That was harmless in practice only because the WS `finally` always reaches
dispose() -- the no-leak invariant the guard exists for was half-held, which is not held.
"""
import os
import stat

import authkit  # test auth env; must be imported before voicemanager

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import voicemanager.config as cfg
from voicemanager.app import create_app
from voicemanager.pipeline.audiocpp_stt import SidecarError, _SidecarClient
from voicemanager.session import SessionRegistry


# 512 samples of nonzero 16-bit PCM -- even length passes first-frame validation.
_PCM = b"\x11\x22" * 512


def _owned_config(bin_path: str) -> dict:
    """Production shape: a binary configured, NO socket -> the pipeline spawns its own."""
    return {
        "sidecar_bin": bin_path,
        "socket": "",
        "model": "",
        "model_task": "nemotron-3.5-asr-streaming",
        "backend": "cpu",
        "device": 0,
        "threads": 4,
        "assets_root": "",
    }


# --- spawn failures are SidecarError, not a raw OSError -----------------------------

def test_missing_sidecar_binary_raises_sidecar_error(tmp_path):
    """The default-path defect: a nonexistent binary must not raise FileNotFoundError.

    `_spawn` is NOT stubbed here -- a real `subprocess.Popen` runs against a path that does
    not exist. Before the spawn guard this raised `FileNotFoundError`, which is not a
    `SidecarError`, so it fell through the WS handler's `except SidecarError` into the catch-all.
    """
    missing = str(tmp_path / "no-such-audiocpp_sidecar")
    client = _SidecarClient(_owned_config(missing))
    try:
        with pytest.raises(SidecarError) as excinfo:
            client.connect()
    finally:
        client.close()
    assert not isinstance(excinfo.value, OSError), \
        "a spawn failure must be a SidecarError, not a raw OSError subclass"
    assert excinfo.value.code == "spawn_failed", \
        f"expected a defined spawn_failed code, got {excinfo.value.code!r}"


def test_spawn_failure_message_names_the_resolved_path(tmp_path):
    """An operator has to be able to see WHICH path was tried, absolute, not CWD-relative.

    The configured default is relative, so "could not spawn sidecar/build/bin/..." is not
    actionable on its own -- the whole failure mode is that the server's CWD is not what
    the operator assumed.
    """
    missing = str(tmp_path / "no-such-audiocpp_sidecar")
    client = _SidecarClient(_owned_config(missing))
    try:
        with pytest.raises(SidecarError) as excinfo:
            client.connect()
    finally:
        client.close()
    assert os.path.abspath(missing) in str(excinfo.value), (
        f"the spawn failure must name the resolved binary path; got {excinfo.value}")


def test_unexecutable_sidecar_binary_raises_sidecar_error(tmp_path):
    """A present-but-not-executable binary (PermissionError) is the same config class.

    A build artifact copied without its mode bit, or a shell script without +x, is a real
    operator mistake and must not read as a VoiceManager bug either.
    """
    not_exec = tmp_path / "audiocpp_sidecar"
    not_exec.write_text("#!/bin/sh\nexit 0\n")
    not_exec.chmod(stat.S_IRUSR | stat.S_IWUSR)   # readable, deliberately NOT executable
    client = _SidecarClient(_owned_config(str(not_exec)))
    try:
        with pytest.raises(SidecarError) as excinfo:
            client.connect()
    finally:
        client.close()
    assert excinfo.value.code == "spawn_failed"


def test_spawn_failure_leaves_no_private_socket_dir_behind(tmp_path):
    """The private dir is minted BEFORE the spawn, so a failed spawn must still remove it."""
    missing = str(tmp_path / "no-such-audiocpp_sidecar")
    client = _SidecarClient(_owned_config(missing))

    minted: list[str] = []
    real_mint = client._make_private_socket_path

    def _spy_mint():
        path = real_mint()
        minted.append(path)
        return path

    client._make_private_socket_path = _spy_mint

    with pytest.raises(SidecarError):
        client.connect()

    assert minted, "the test never observed a mint; the spy did not take effect"
    assert client._owned_dir is None, "a failed spawn left _owned_dir set"
    assert not os.path.exists(os.path.dirname(minted[0])), \
        "a failed spawn leaked its private socket dir"
    assert client.is_connected is False


# --- a dial failure AFTER a successful spawn tears the spawn back down --------------

class _DeadProc:
    """A sidecar that exited during model load: spawned fine, never bound the socket.

    `_dial` watches the process it spawned, so a non-None poll() short-circuits the retry
    window immediately -- which is both the realistic failure (a bad GGUF) and what keeps
    this test fast rather than waiting out the 60 s warm-up deadline.
    """

    def __init__(self):
        self.terminated = False
        self.waited = False
        self.pid = -1
        self.returncode = 1

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode


def test_dial_failure_after_a_successful_spawn_leaks_nothing(monkeypatch):
    """The earlier guard wrapped only _handshake(), so spawn-then-dial-fail leaked both.

    The sidecar process and the private temp dir both belong to this client the moment the
    spawn returns, so the teardown guard has to start before the spawn -- not after the dial.
    """
    proc = _DeadProc()
    monkeypatch.setattr(_SidecarClient, "_spawn",
                        lambda self, bin_path, socket_path: proc)

    client = _SidecarClient(_owned_config("/nonexistent/audiocpp_sidecar"))

    minted: list[str] = []
    real_mint = client._make_private_socket_path

    def _spy_mint():
        path = real_mint()
        minted.append(path)
        return path

    client._make_private_socket_path = _spy_mint

    with pytest.raises(SidecarError) as excinfo:
        client.connect()
    assert excinfo.value.code == "dial_failed"

    assert client._proc is None, \
        "a dial failure after a successful spawn left the sidecar process attached"
    assert proc.terminated, \
        "the spawned sidecar was never signalled after the dial failed -- it leaks"
    assert client._owned_dir is None, "a dial failure left _owned_dir set"
    assert minted and not os.path.exists(os.path.dirname(minted[0])), \
        "a dial failure leaked the private socket dir"
    assert client.is_connected is False


# --- and it reaches the client as the DEFINED config-failure contract ----------------

def _app_for(bin_path: str):
    app = create_app()
    app.state.stt_engine_type = "audiocpp"
    app.state.audiocpp_config = _owned_config(bin_path)
    app.state.session_registry = SessionRegistry()
    return app


def _drain_until_close(ws, *, budget: int = 50):
    """Collect frames until the server closes the socket; return (frames, close_code)."""
    frames: list[dict] = []
    close_code = None
    try:
        for _ in range(budget):
            frames.append(ws.receive_json())
    except WebSocketDisconnect as disc:
        close_code = disc.code
    return frames, close_code


def test_missing_binary_gets_the_defined_frame_and_close_code(monkeypatch, tmp_path, caplog):
    """The end-to-end claim: a missing binary closes 4001, not 1011 with a bug traceback.

    Same defined contract as `load_failed` (tests/test_audiocpp_handshake_error.py): both
    mean "this server cannot serve this engine at all -- read the config, stop retrying".
    """
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "30")
    cfg._settings = None
    missing = str(tmp_path / "no-such-audiocpp_sidecar")
    client = TestClient(_app_for(missing))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        try:
            ws.send_bytes(_PCM)   # the first feed() triggers the lazy spawn
        except Exception:
            pass                  # the server may already have closed us
        frames, close_code = _drain_until_close(ws)

    errors = [f for f in frames if f.get("type") == "error"]
    assert len(errors) == 1, f"expected exactly one error frame, got {frames}"
    assert errors[0]["error_type"] == "stt_backend_unavailable", (
        f"expected the defined config-failure error_type, got {errors[0]!r} -- a generic "
        "server_error means the spawn failure fell through to the catch-all handler")
    assert close_code == 4001, f"expected the defined close code 4001, got {close_code}"

    tracebacks = [r for r in caplog.records if r.exc_info and r.levelname == "ERROR"]
    assert not tracebacks, (
        f"a missing sidecar binary logged an unhandled traceback: "
        f"{[r.getMessage() for r in tracebacks]}")
