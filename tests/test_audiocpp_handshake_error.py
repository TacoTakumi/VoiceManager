"""A sidecar handshake/config failure gets a DEFINED close code, not a bug traceback.

Model-free and server-free (the CLAUDE.md invariant): a real FastAPI `/ws/stream` driven
by a `TestClient`, with the real `AudioCppSTTPipeline` speaking sidecar/PROTOCOL.md over a
genuine AF_UNIX socket to `tests/audiocpp_fake.py`. No GPU, no native binary, no model.

Two distinct audio.cpp failure classes must look different to a client. The
crash path already gave the RUNTIME fault its own frame + code (`stt_backend_fault` / 1011:
the backend was working and broke, so a reconnect may well succeed). What had no defined
code was the CONFIGURATION/handshake class -- `load_failed` for a missing or bad GGUF,
`unknown_model_task`, `unsupported_protocol`, or a dial that never reaches a listener. Those
fell through to the generic `except Exception`, so an operator misconfiguration emitted a
full `logger.exception` traceback and closed with a generic `server_error` -- indistinguishable
from a VoiceManager bug, both in the log and to the client. Each class needs a DEFINED code.

They now close `4001` with `error_type="stt_backend_unavailable"`. The asymmetry with the
1011 fault path is deliberate and is the useful signal: 1011 means "it broke, retrying may
help"; 4001 means "this server cannot serve this engine at all", so a client should stop
retrying and an operator should read the config, not the stack trace.

Also covers a socket leak: a handshake that raises must not leave a live socket behind, or the
documented "idempotent once connected" guard in connect() would let a retried feed() skip
the handshake and send AUDIO on an un-handshaken socket.

A later change then split that one class in two, because `SidecarError`
is ALSO the type for TRANSIENT conditions and 4001 documents "do not retry". The close
code is now chosen from `exc.code`: a permanent operator-configuration failure still gets
4001 `stt_backend_unavailable`, while a transient one -- `busy` from a sidecar that
already has a live session -- gets 4002 `stt_backend_unready`, which asks the client to
retry with backoff. A recoverable MID-STREAM error does not end the session at all: the
sidecar keeps serving, so the pipeline drops that one utterance and streams on.
"""
import asyncio
import socket

import authkit  # test auth env; must be imported before voicemanager

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import voicemanager.config as cfg
from voicemanager.app import create_app
from voicemanager.pipeline.audiocpp_stt import (
    AudioCppSTTPipeline,
    SidecarError,
    _SidecarClient,
)
from voicemanager.session import SessionRegistry

from audiocpp_fake import FakeSidecar

# 512 samples of nonzero 16-bit PCM -- even length passes first-frame validation.
_PCM = b"\x11\x22" * 512


def _config(socket_path: str) -> dict:
    """Dial-only config (no binary): connect to `socket_path`, never spawn."""
    return {
        "sidecar_bin": "",
        "socket": socket_path,
        "model": "",
        "model_task": "nemotron-3.5-asr-streaming",
        "backend": "cpu",
        "device": 0,
        "threads": 4,
        "assets_root": "",
    }


def _app_for(socket_path: str):
    app = create_app()
    app.state.stt_engine_type = "audiocpp"
    app.state.audiocpp_config = _config(socket_path)
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


def _assert_backend_unavailable(frames, close_code):
    """The defined config-failure contract: one stt_backend_unavailable frame, close 4001."""
    errors = [f for f in frames if f.get("type") == "error"]
    assert len(errors) == 1, f"expected exactly one error frame, got {frames}"
    assert errors[0]["error_type"] == "stt_backend_unavailable", (
        f"expected the defined config-failure error_type, got {errors[0]!r} -- a generic "
        "server_error means the SidecarError fell through to the catch-all handler")
    assert close_code == 4001, f"expected the defined close code 4001, got {close_code}"


def _feed_until_closed(ws, *, budget: int = 50):
    """Push PCM until the server closes; return (frames, close_code) like _drain_until_close.

    The lazy connect happens on the FIRST audio frame, so one send is normally enough --
    but the send and the close race, so tolerate a socket that is already gone.
    """
    try:
        ws.send_bytes(_PCM)
    except Exception:
        pass
    return _drain_until_close(ws, budget=budget)


def test_handshake_error_gets_defined_frame_and_close_code(monkeypatch, caplog):
    """ERROR load_failed at HELLO -> stt_backend_unavailable + 4001, and no bug traceback.

    `load_failed` is what the real sidecar replies when the configured GGUF is missing or
    unreadable -- the single most likely operator misconfiguration, and the one the manual
    LIVE half of this task's verify induces with a bogus VM_AUDIOCPP_MODEL.
    """
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "30")
    cfg._settings = None
    with FakeSidecar(hello_error="load_failed") as fake:
        client = TestClient(_app_for(fake.socket_path))
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            try:
                ws.send_bytes(_PCM)   # first feed() triggers the lazy connect + handshake
            except Exception:
                pass                  # the server may already have closed us
            frames, close_code = _drain_until_close(ws)
    _assert_backend_unavailable(frames, close_code)

    # An operator misconfiguration is NOT a VoiceManager bug: it must not be logged as one.
    tracebacks = [r for r in caplog.records if r.exc_info and r.levelname == "ERROR"]
    assert not tracebacks, (
        f"a sidecar config failure logged an unhandled traceback: "
        f"{[r.getMessage() for r in tracebacks]}")


def test_unreachable_sidecar_gets_the_same_defined_close_code(monkeypatch, tmp_path):
    """A dial that reaches no listener is the same config-failure class, not a server bug.

    A bare `OSError` out of `_dial` used to reach the WS handler's catch-all unchanged.
    It is now a SidecarError like every other config failure, so "the socket path
    is wrong" and "the model path is wrong" look the same to a client: 4001.
    """
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "30")
    cfg._settings = None
    dead_path = str(tmp_path / "nothing-is-listening.sock")
    client = TestClient(_app_for(dead_path))
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        try:
            ws.send_bytes(_PCM)
        except Exception:
            pass
        frames, close_code = _drain_until_close(ws)
    _assert_backend_unavailable(frames, close_code)


def test_failed_handshake_leaves_no_live_socket(monkeypatch):
    """connect() must not keep a socket it never handshook.

    `_sock` was assigned before the unguarded `_handshake()`, so a handshake exception left
    a live, un-handshaken fd behind. connect() is documented "idempotent once connected" and
    returns early on `_sock is not None`, so a retried feed() would have skipped the
    handshake entirely and sent AUDIO on a socket the sidecar never READYed -- and the fd
    itself was only reclaimed at dispose().

    Asserted at the fd level (`fileno() == -1` after a real close), not just `_sock is None`:
    dropping the reference without closing would leak the descriptor just as badly.
    """
    with FakeSidecar(hello_error="unknown_model_task") as fake:
        client = _SidecarClient(_config(fake.socket_path))

        dialled: list[socket.socket] = []
        real_dial = client._dial

        def _spy_dial(*args, **kwargs):
            sock = real_dial(*args, **kwargs)
            dialled.append(sock)
            return sock

        client._dial = _spy_dial

        with pytest.raises(SidecarError) as excinfo:
            client.connect()
        assert excinfo.value.code == "unknown_model_task"

        assert client.is_connected is False, "a failed handshake left the client 'connected'"
        assert client._sock is None, "a failed handshake left _sock set"
        assert dialled, "the test never observed a dial; the spy did not take effect"
        assert dialled[0].fileno() == -1, (
            "the handshake failed but its socket was never closed -- the fd leaks until "
            "dispose() and the connect() idempotency guard becomes unsound")


def test_retry_after_failed_handshake_never_sends_audio_unhandshaken(monkeypatch):
    """The invariant behind the leak fix: a retried connect() re-handshakes, it does not proceed.

    This is what a leaked `_sock` would actually have cost: the second connect() would hit
    the idempotency early-return, believe itself handshaken, and send AUDIO on a socket the
    sidecar had already rejected. The fake counts AUDIO frames per connection, so a
    regression here is observable rather than merely structural.
    """
    with FakeSidecar(hello_error="unsupported_protocol") as fake:
        client = _SidecarClient(_config(fake.socket_path))
        for _ in range(2):
            with pytest.raises(SidecarError):
                client.connect()
            assert client.is_connected is False
    assert all(conn.audio_frames == 0 for conn in fake.connections), (
        "AUDIO was sent on a connection that never completed a handshake")


# --- the close code is chosen from the ERROR CODE, not from the exception class ---------

def test_busy_gets_a_RETRIABLE_close_code_not_the_do_not_retry_one(monkeypatch, caplog):
    """ERROR `busy` -> stt_backend_unready + 4002, because a retry WOULD succeed.

    `busy` is what a sidecar that already has a live session answers a second connection
    with (sidecar/src/main.cpp), and it clears the moment that session ends -- the
    codebase already treats it as transient (test_audiocpp_ownership.py retries it in
    `_connect_when_free`). Reporting it as 4001 told the client the exact opposite of the
    truth: "this is a server-side configuration problem, every reconnect will fail the
    same way". A transient fault must never carry the do-not-retry contract.
    """
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "30")
    cfg._settings = None
    with FakeSidecar(hello_error="busy") as fake:
        client = TestClient(_app_for(fake.socket_path))
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            frames, close_code = _feed_until_closed(ws)

    errors = [f for f in frames if f.get("type") == "error"]
    assert len(errors) == 1, f"expected exactly one error frame, got {frames}"
    assert close_code != 4001, (
        "a transient `busy` closed with the PERMANENT code, which documents 'do NOT retry "
        "in a loop' -- the busy-code regression")
    assert close_code == 4002, f"expected the retriable close code 4002, got {close_code}"
    assert errors[0]["error_type"] == "stt_backend_unready", (
        f"expected the retriable error_type, got {errors[0]!r}")
    assert "busy" in errors[0]["message"], (
        "the specific cause code must still reach the client, as it does on the 4001 path")

    # Transient and expected: not a bug traceback, and not even an ERROR-level log.
    assert not [r for r in caplog.records if r.exc_info and r.levelname == "ERROR"], (
        "a transient sidecar condition was logged as an unhandled VoiceManager bug")


@pytest.mark.parametrize("code", [
    "load_failed", "unknown_model_task", "unsupported_protocol",
    "dial_failed", "spawn_failed", "no_sidecar",
])
def test_permanent_config_codes_keep_the_do_not_retry_close_code(code):
    """The operator-configuration class is what 4001 exists for; it must not widen or shrink.

    Each of these means this server cannot serve this engine until a human changes the
    `VM_AUDIOCPP_*` config, so every reconnect really does fail identically -- the one
    situation where telling a client to stop retrying is correct.
    """
    assert SidecarError(code).is_permanent is True, (
        f"{code} is an operator-configuration failure and must close 4001")


@pytest.mark.parametrize("code", ["busy", "internal", "bad_frame", "bad_handshake",
                                  "not_implemented", "some_future_sidecar_code"])
def test_everything_else_is_treated_as_retriable(code):
    """Unclassified codes fail SAFE: retriable, the direction that cannot strand a client.

    The permanent set is an explicit allow-list rather than "everything except the
    transient ones" on purpose. Misjudging a permanent fault as retriable costs a client
    a few doomed reconnects, each of which still surfaces the cause code; misjudging a
    transient one as permanent tells a working deployment to give up and page an
    operator. A new sidecar ERROR code added upstream lands on the harmless side.
    """
    assert SidecarError(code).is_permanent is False, (
        f"{code} is not a known operator-configuration failure, so it must not inherit "
        "4001's do-not-retry contract")


def test_recoverable_mid_stream_error_does_not_end_the_session():
    """ERROR `internal` on FLUSH costs that ONE utterance; the session keeps streaming.

    drive_loop.cpp emits `internal` from the catch block around session.finalize() and
    then CONTINUES its loop -- the sidecar is alive and still serving. Ending the session
    (let alone with the do-not-retry code) over a single recoverable decode error is a
    protocol-conformance divergence: the C++ side calls it recoverable, so the Python
    side must not call it fatal.
    """
    with FakeSidecar(flush_error="internal", flush_final="second utterance") as fake:
        p = AudioCppSTTPipeline(config=_config(fake.socket_path))

        asyncio.run(p.feed(_PCM))
        # The failed utterance yields no text -- and, crucially, no exception.
        text, _lang, _seg = asyncio.run(p.finalize())
        assert text == "", f"a failed decode must not invent a final, got {text!r}"

        # The session is still usable: the very next utterance finalizes normally.
        p.reset()
        asyncio.run(p.feed(_PCM))
        text, _lang, _seg = asyncio.run(p.finalize())
        assert text == "second utterance", (
            "the sidecar stayed alive and served the next utterance, but the pipeline did "
            "not -- a recoverable mid-stream error tore the session down")

        assert len(fake.connections) == 1, (
            "the pipeline reconnected instead of continuing on the live connection")
        p.dispose()


def test_non_recoverable_mid_stream_error_still_raises_but_as_retriable():
    """A mid-stream code the drive loop does NOT recover from still ends the session.

    Only the codes the sidecar itself keeps serving through are absorbed; anything else
    still surfaces as a SidecarError. It just must not claim to be a permanent operator
    misconfiguration, since nothing about it says the config is wrong.
    """
    with FakeSidecar(flush_error="bad_frame") as fake:
        p = AudioCppSTTPipeline(config=_config(fake.socket_path))
        asyncio.run(p.feed(_PCM))
        with pytest.raises(SidecarError) as excinfo:
            asyncio.run(p.finalize())
        assert excinfo.value.code == "bad_frame"
        assert excinfo.value.is_permanent is False
        p.dispose()
