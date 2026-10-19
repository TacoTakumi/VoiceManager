"""A capped receive-timeout wake must not warn a healthy session.

Model-free and server-free (the CLAUDE.md invariant): a real FastAPI `/ws/stream`
driven by a `TestClient` with the real AudioCppSTTPipeline over tests/audiocpp_fake.py.

The defect (MAJOR, confirmed live): the inactivity-warning
branch fires on any receive timeout the earlier branches decline, guarded only by the
STALE loop-top `remaining > WARNING_LEAD`. The silence cap makes sub-second receive
timeouts routine on audiocpp sessions, so a marginally-early wake (uvloop's cached
clock vs a fresh time.monotonic()) told healthy clients "Session will close in 30s due
to inactivity" ~1.2 s into a session - and latched warning_sent, suppressing the REAL
warning later. Live timing jitter cannot be provoked deterministically model-free, so
the first test injects ONE spurious asyncio.TimeoutError at
the exact seam where the jitter bites - a wake whose elapsed-since-last-audio re-check
lands short - and asserts no warning results. The second test pins the legitimate
path: a genuinely idle session still gets exactly one warning, after its final.
"""
import asyncio
import time

import authkit  # test auth env; must be imported before voicemanager

from fastapi.testclient import TestClient

import voicemanager.config as cfg
import voicemanager.transport.websocket as ws_mod
from voicemanager.app import create_app
from voicemanager.session import SessionRegistry

from audiocpp_fake import FakeSidecar


# 512 samples of nonzero 16-bit PCM -- even length passes first-frame validation.
_PCM = b"\x11\x22" * 512


def _config(fake: FakeSidecar) -> dict:
    """Point the pipeline at the fake's socket with NO binary -> connect, never spawn."""
    return {
        "sidecar_bin": "",
        "socket": fake.socket_path,
        "model": "",
        "model_task": "nemotron-3.5-asr-streaming",
        "backend": "cpu",
        "device": 0,
        "threads": 4,
        "assets_root": "",
    }


def _app_for(fake: FakeSidecar):
    app = create_app()
    app.state.stt_engine_type = "audiocpp"
    app.state.audiocpp_config = _config(fake)
    app.state.session_registry = SessionRegistry()
    return app


class _SpuriousWakeAsyncio:
    """Proxy for the websocket module's `asyncio` that can fire one spurious wake.

    While armed, the NEXT `wait_for` call raises asyncio.TimeoutError immediately --
    without awaiting the receive -- which is byte-for-byte what the live marginal wake
    looks like to the handler: a timeout that fired although the silence window has not
    actually elapsed, so the silence re-check declines and control falls through to the
    warning branch. Everything else delegates to the real asyncio.
    """

    def __init__(self):
        self.armed = False
        self.fired = 0

    def __getattr__(self, name):
        return getattr(asyncio, name)

    async def wait_for(self, awaitable, timeout):
        if self.armed:
            self.armed = False
            self.fired += 1
            # The un-awaited receive() coroutine must be closed or asyncio warns.
            awaitable.close()
            raise asyncio.TimeoutError()
        return await asyncio.wait_for(awaitable, timeout)


def test_spurious_capped_wake_does_not_warn_a_healthy_session(monkeypatch):
    """One injected mid-stream timeout wake: no timeout_warning, session keeps working.

    VM_AUDIOCPP_SILENCE_MS is set high (1.5 s) so the injected wake's elapsed-since-audio
    re-check declines DETERMINISTICALLY (elapsed is a few ms), exercising exactly the
    fallthrough the false warning came from. Before the guard this emitted the
    "Session will close in 30s" warning ~immediately; after it, nothing is emitted and
    the session still finalizes normally on demand.
    """
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "60")        # remaining >> WARNING_LEAD
    # Large enough that the injected wake's elapsed-since-audio re-check (a few tens of
    # ms) declines deterministically; small enough that the utterance finalizes fast.
    monkeypatch.setenv("VM_AUDIOCPP_SILENCE_MS", "1500")
    shim = _SpuriousWakeAsyncio()
    monkeypatch.setattr(ws_mod, "asyncio", shim)
    cfg._settings = None
    try:
        # audio_events: one interim partial per AUDIO frame -- an ordered in-band probe.
        # WS frames are ordered, so a warning emitted at injection time MUST show up
        # before the partials of the frames sent after the injection.
        with FakeSidecar(audio_events=["probe partial"], flush_final="healthy final") as fake:
            client = TestClient(_app_for(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"
                # Arm audiocpp_last_audio, then let the handler settle into its wait.
                for _ in range(3):
                    ws.send_bytes(_PCM)
                    time.sleep(0.03)
                time.sleep(0.05)
                shim.armed = True
                # This frame wakes the current (real) wait; the NEXT wait_for is the
                # injected spurious timeout, whose re-check declines (elapsed << 1.5 s).
                ws.send_bytes(_PCM)
                deadline = time.monotonic() + 2.0
                while shim.fired == 0 and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert shim.fired == 1, "the spurious wake was never injected"

                # Stream past the injection point, then go quiet so the silence cap
                # finalizes the utterance. A false warning is emitted AT injection
                # time, so it is ordered strictly before this utterance's final --
                # draining to the final provably crosses the injection point, and
                # stopping there keeps the session far from the REAL warning window.
                for _ in range(5):
                    ws.send_bytes(_PCM)
                    time.sleep(0.05)
                seen: list[dict] = []
                while True:
                    frame = ws.receive_json()
                    seen.append(frame)
                    assert frame.get("error_type") != "timeout_warning", (
                        f"healthy session got a false warning: {seen}")
                    if frame.get("type") == "transcription" and frame.get("is_final"):
                        break
    finally:
        cfg._settings = None


def test_genuinely_idle_session_still_gets_one_warning_after_its_final(monkeypatch):
    """The legitimate path is unchanged: idle long enough -> exactly one warning.

    VM_SESSION_TIMEOUT=31 puts the warning ~1 s after the utterance finalizes (the
    pre-warning leg is session_timeout - 30). The warning must come AFTER the final
    (never before -- that ordering is the live symptom) and exactly once.
    """
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "31")
    monkeypatch.setenv("VM_AUDIOCPP_SILENCE_MS", "200")
    cfg._settings = None
    try:
        with FakeSidecar(flush_events=["idle partial"], flush_final="idle final") as fake:
            client = TestClient(_app_for(fake))
            with authkit.ws_connect(client, "/ws/stream") as ws:
                assert ws.receive_json()["type"] == "ready"
                for _ in range(4):
                    ws.send_bytes(_PCM)
                    time.sleep(0.03)
                # Silence: the 200 ms gap finalizes the utterance; ~1 s later the real
                # inactivity warning is due. Collect frames until the warning arrives.
                seen: list[dict] = []
                deadline = time.monotonic() + 6.0
                got_warning = False
                while time.monotonic() < deadline and not got_warning:
                    frame = ws.receive_json()
                    seen.append(frame)
                    got_warning = frame.get("error_type") == "timeout_warning"
                finals = [f for f in seen if f.get("type") == "transcription"
                          and f.get("is_final")]
                warnings = [f for f in seen if f.get("error_type") == "timeout_warning"]
                assert len(warnings) == 1, f"expected exactly one warning, saw {seen}"
                assert finals, "utterance final never arrived"
                # Ordering: the final precedes the warning (the bug inverted this live).
                assert seen.index(finals[0]) < seen.index(warnings[0])
    finally:
        cfg._settings = None
