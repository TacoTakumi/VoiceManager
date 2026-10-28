"""audio.cpp TTS engine (preset loader and engine core).

The engine talks to the SAME audio.cpp sidecar binary as the STT backend over
the additive TTS frame set (sidecar/PROTOCOL.md): HELLO selects the model_task
(dots-tts-streaming by default, or voxcpm2-tts-streaming), each synthesis request is
one TEXT frame, and the reply is a stream of TTS_AUDIO frames (raw s16le mono at the
READY-stated rate: 48 kHz for dots and voxcpm2, 24 kHz for breeze-tts-streaming)
closed by exactly one TTS_DONE. Output audio passes through untouched - no
server-side resampling, ever (only a preset clip composed into a later clause
piece's reference is brought to the engine rate first; see the composed-reference
section below).

v1 voice selection is named presets from the frozen voices/designed/ assets:
each refs.json entry pairs reference audio with its exact transcript
(ref_text), and the engine clones from the (audio, transcript) pair. ref_text
is per-model: dots requires a non-empty transcript - validation happens
at LOAD time, so an asset with a missing/empty ref_text or an absent WAV fails
engine startup with a clear error, never a request mid-flight - while voxcpm2
allows a preset to omit it and drops the ref_text key from the dispatched TEXT
frame when absent.

Blocking socket I/O stays off the event loop (asyncio.to_thread), matching the
STT engine.

Failure classes follow the STT split: a socket-level failure -
an abrupt EOF, a read timeout, a reset - means the per-session sidecar PROCESS
died and raises `SidecarFault`; an ERROR frame reply or a config/refusal
problem stays `SidecarError`. The fault matters beyond the exception class:
the synthesis runs in a detached task the WS receive loop never awaits, so the
engine also hands a fault to the session's `on_fault` callback (set by the WS
layer), which surfaces the defined session error and closes that one session.
"""
from __future__ import annotations

import asyncio
import atexit
import json
import logging
import math
import os
import socket
import struct
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

import numpy as np
import soundfile as sf

from voicemanager.config import get_settings
from voicemanager.pipeline.audiocpp_stt import SidecarError, SidecarFault
from voicemanager.pipeline.tts import TTSEngine, TTSStreamPipeline, float32_to_pcm16

logger = logging.getLogger(__name__)

# Message type tags - must match sidecar/src/protocol.h / PROTOCOL.md.
_HELLO = 0x01
_READY = 0x02
_ERROR = 0x08
_BYE = 0x09
_TEXT = 0x0A
_TTS_AUDIO = 0x0B
_TTS_DONE = 0x0C
_CANCEL = 0x0D

_HEADER = struct.Struct(">BI")
_PROTOCOL_VERSION = 1

#: The default TTS output rate (dots and voxcpm2 vocoders); the READY ack states
#: the selected model's rate authoritatively and the engine passes audio through
#: at exactly that rate. breeze's READY says 24000: `sample_rate` below follows READY
#: once connected.
TTS_SAMPLE_RATE = 48000

DEFAULT_TTS_MODEL_TASK = "dots-tts-streaming"

#: The voxcpm2 model_task. Its presence is the
#: per-model signal that ref_text is OPTIONAL, mirroring the sidecar
#: registry descriptor's ref_text_required flag.
VOXCPM2_TTS_MODEL_TASK = "voxcpm2-tts-streaming"

#: The breeze model_task. Selected the
#: same deployment-level way (VM_AUDIOCPP_TTS_MODEL_TASK); ref_text is REQUIRED
#: under it - breeze clones from the reference clip plus its exact transcript -
#: so a preset without one fails at load like dots. Its sidecar READY advertises
#: 24000 and `sample_rate` follows it. The default stays dots.
BREEZE_TTS_MODEL_TASK = "breeze-tts-streaming"

#: The Qwen3-TTS 1.7B Base model_task. Selected the same deployment-level way;
#: ref_text is REQUIRED under it - Base clones from the reference clip plus its
#: transcript - so a preset without one fails at load. Its READY advertises 24000.
QWEN3_TTS_MODEL_TASK = "qwen3-tts-streaming"

#: Output rate per model_task before any sidecar has answered READY, mirroring
#: the sidecar registry descriptors (task_registry.cpp sample_rate). The startup
#: engine never opens a sidecar, so /info reads its rate from here; a connected
#: session engine still reports what READY said. Tasks not listed use the
#: default constant.
MODEL_TASK_SAMPLE_RATES = {BREEZE_TTS_MODEL_TASK: 24000, QWEN3_TTS_MODEL_TASK: 24000}

#: The preset used when a request names no voice.
DEFAULT_VOICE = "designed_female_s101"


class PresetLoadError(RuntimeError):
    """A voice asset failed validation at load time (fail-fast)."""


class UnknownVoiceError(ValueError):
    """A request named a voice that is not a loaded preset."""


@dataclass(frozen=True)
class PresetVoice:
    """One named preset: the reference WAV and its exact transcript."""

    name: str
    wav_path: str  # absolute path, verified to exist at load time
    ref_text: str  # empty when the model does not require it and the asset omits it


def load_presets(refs_path: str | Path, ref_text_required: bool = True) -> dict[str, PresetVoice]:
    """Read refs.json into validated presets, failing fast on any bad asset.

    Keys starting with "_" are metadata (e.g. "_comment") and are skipped.
    Raises PresetLoadError on: a missing/unreadable refs.json, an entry with a
    missing or empty ref_text when `ref_text_required` is set, an entry whose WAV
    file is absent, or a refs.json that lacks the DEFAULT_VOICE preset (the
    no-voice-named path must resolve). When `ref_text_required` is False
    (voxcpm2), an omitted or empty ref_text is allowed and stored as "".
    """
    refs_path = Path(refs_path)
    if not refs_path.is_file():
        raise PresetLoadError(f"voice presets refs.json not found: {refs_path}")
    try:
        entries = json.loads(refs_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PresetLoadError(f"could not read voice presets {refs_path}: {exc}") from exc

    presets: dict[str, PresetVoice] = {}
    for name, entry in entries.items():
        if name.startswith("_"):
            continue  # metadata key, not a preset
        if not isinstance(entry, dict):
            raise PresetLoadError(f"preset {name!r} in {refs_path} is not an object")
        ref_text = entry.get("ref_text", "")
        if not isinstance(ref_text, str):
            raise PresetLoadError(
                f"preset {name!r} in {refs_path} has a non-string ref_text")
        if ref_text_required and not ref_text.strip():
            raise PresetLoadError(
                f"preset {name!r} in {refs_path} has a missing or empty ref_text - "
                "the reference transcript is mandatory for this model")
        wav_file = entry.get("file")
        if not isinstance(wav_file, str) or not wav_file:
            raise PresetLoadError(f"preset {name!r} in {refs_path} names no WAV file")
        wav_path = (refs_path.parent / wav_file).resolve()
        if not wav_path.is_file():
            raise PresetLoadError(
                f"preset {name!r} reference WAV is absent: {wav_path}")
        presets[name] = PresetVoice(name=name, wav_path=str(wav_path), ref_text=ref_text)

    if DEFAULT_VOICE not in presets:
        raise PresetLoadError(
            f"default preset {DEFAULT_VOICE!r} is missing from {refs_path} - "
            "a request naming no voice could not resolve")

    logger.info("loaded %d voice presets from %s: %s",
                len(presets), refs_path, sorted(presets))
    return presets


def resolve_preset(presets: dict[str, PresetVoice], name: str | None) -> PresetVoice:
    """Resolve a request's voice name to a preset.

    No name (None or empty) resolves to DEFAULT_VOICE; an unknown name raises
    UnknownVoiceError listing the valid presets - never a silent fallback.
    """
    if not name:
        return presets[DEFAULT_VOICE]
    preset = presets.get(name)
    if preset is None:
        raise UnknownVoiceError(
            f"unknown voice {name!r}; valid presets: {', '.join(sorted(presets))}")
    return preset


# --- composed references for the later clause pieces ---------------------------------
#
# Breeze re-conditions the speaker from the reference (clip + transcript) on every
# request, and audio.cpp continues a voice only inside one request. A clause-mode
# tts_stream request is many engine requests, so the voice would be re-sampled at
# every piece. The stream pipeline closes that seam itself: for each engine request
# after the first one it writes a temporary WAV of the preset clip followed by the
# previous piece's own audio, and extends the transcript with that piece's text.
# The files live in one per-process directory and belong to the request that made
# them; the directory itself goes at process exit.

_clause_ref_dir_path: str | None = None
_clause_ref_dir_lock = threading.Lock()


def clause_ref_dir() -> str:
    """The per-process directory that holds the composed reference WAVs."""
    global _clause_ref_dir_path
    with _clause_ref_dir_lock:
        if _clause_ref_dir_path is None:
            _clause_ref_dir_path = tempfile.mkdtemp(prefix="vm-audiocpp-clause-ref-")
            atexit.register(shutil.rmtree, _clause_ref_dir_path, ignore_errors=True)
        return _clause_ref_dir_path


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        logger.debug("could not remove composed reference %s", path, exc_info=True)


def _float32_to_int16(audio: np.ndarray) -> np.ndarray:
    # The engine made these floats from the sidecar's s16le by /32768, so *32768
    # and rounding gives the sidecar's samples back exactly.
    return np.clip(np.rint(audio * 32768.0), -32768, 32767).astype(np.int16)


def read_preset_clip(wav_path: str, sample_rate: int) -> np.ndarray:
    """The preset clip as int16 mono at `sample_rate`, resampled when it differs.

    Blocking (file read, maybe a polyphase resample): call from a worker thread.
    A stereo clip keeps its first channel.
    """
    data, clip_rate = sf.read(wav_path, dtype="int16", always_2d=True)
    mono = data[:, 0]
    if int(clip_rate) == sample_rate:
        return mono.copy()
    from scipy.signal import resample_poly

    divisor = math.gcd(sample_rate, int(clip_rate))
    resampled = resample_poly(mono.astype(np.float32), sample_rate // divisor,
                              int(clip_rate) // divisor)
    return np.clip(np.rint(resampled), -32768, 32767).astype(np.int16)


def write_composed_reference(path: str, preset_clip: np.ndarray, tail: np.ndarray,
                             sample_rate: int) -> None:
    """Write the preset clip followed by the previous piece's tail as 16-bit mono WAV.

    Blocking: call from a worker thread.
    """
    sf.write(path, np.concatenate([preset_clip, tail]), sample_rate, subtype="PCM_16")


class _TtsSidecarClient:
    """Owns one TTS sidecar connection: spawn (if needed), socket, wire framing.

    All methods block and are meant to be called from a worker thread (the engine
    wraps them in asyncio.to_thread). One instance drives exactly one TTS session,
    mirroring PROTOCOL.md's "one connection is one session" and the STT backend's
    _SidecarClient ownership rules (per-session sidecar):

      * OWNED (no `socket` configured): mint a PRIVATE socket path, spawn our own
        sidecar on it, own it exclusively (terminate + remove the path on close).
      * EXTERNAL (`socket` configured): dial an operator-managed listener (the
        fake-sidecar tests, or an out-of-band warm sidecar). Never spawn or kill.
    """

    def __init__(self, config: dict | None):
        self._config = dict(config or {})
        self._sock: socket.socket | None = None
        self._proc: subprocess.Popen | None = None  # set only if THIS client spawned it
        self._owned_dir: str | None = None
        self._ready: dict | None = None  # the READY ack (output-audio contract)
        self._reaper: threading.Thread | None = None  # close()'s off-loop reap, joinable
        # Serializes connect() against close() and against a concurrent connect (the
        # WS warm-up task racing the first request). RLock: connect's failure path
        # cleans up while already holding the lock.
        self._lifecycle_lock = threading.RLock()
        # Latched by mark_closed()/close(), never cleared. The WS
        # teardown can win the race against a warm-up connect whose worker thread
        # has not started yet; without the latch that late connect would spawn a
        # sidecar nobody reaps. Set OUTSIDE the lifecycle lock so a connect already
        # waiting on the lock still sees it. A connect() that merely FAILED does
        # not latch - per-request connect retries depend on that.
        self._closed = False
        # Cancel machinery. `_awaiting_terminal` is True from a TEXT dispatch
        # until its closing TTS_DONE/ERROR is READ - the wire-hygiene flag: a new
        # dispatch while it is set means the previous request was cancelled or
        # abandoned, and its leftover frames must be drained off the wire first or
        # they would surface as the new request's audio. `_read_lock` serializes
        # frame reads (a cancelled task's in-flight to_thread recv keeps running;
        # it must not interleave with the drain's reads), and recv_frame updates
        # the terminal state itself so a frame consumed by an abandoned reader
        # still counts. `_send_lock` keeps concurrent small sends whole.
        self._read_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._awaiting_terminal = False
        self._cancel_sent = False
        # Bumped on every TEXT dispatch. A cancel names the generation its task
        # dispatched, so a doomed task waking up late can never cancel the NEXT
        # request that has since taken the wire.
        self._generation = 0

    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    @property
    def ready(self) -> dict | None:
        return self._ready

    def connect(self) -> None:
        """Connect to this session's TTS sidecar and handshake. Idempotent, and safe
        against a concurrent connect/close (the lifecycle lock serializes them)."""
        with self._lifecycle_lock:
            self._connect_locked()

    def _connect_locked(self) -> None:
        if self._closed:
            return  # the session is gone; connecting now would leak an orphan
        if self._sock is not None:
            return
        socket_path = self._config.get("socket", "")
        bin_path = self._config.get("sidecar_bin", "")
        if not socket_path and not bin_path:
            raise SidecarError(
                "no_sidecar",
                "no audio.cpp TTS sidecar to use: set VM_AUDIOCPP_SIDECAR_BIN to spawn "
                "one per session, or configure a socket to dial an external one")
        try:
            # A connect() that raises must leave nothing behind (as in the STT client):
            # the private dir, the spawned process, and a half-handshaken socket all
            # tear down through close() on any failure below.
            if socket_path:
                self._sock = self._dial(socket_path, deadline_s=5.0)
            else:
                self._owned_dir = tempfile.mkdtemp(prefix="vm-audiocpp-tts-")
                socket_path = os.path.join(self._owned_dir, "sidecar.sock")
                self._proc = self._spawn(bin_path, socket_path)
                self._sock = self._dial(socket_path, deadline_s=60.0)
            # The model loads on HELLO and a long-form generation can take minutes;
            # per-frame gaps while streaming are sub-second, so 120 s per recv is
            # generous without masking a wedged sidecar forever.
            self._sock.settimeout(120.0)
            self._handshake()
        except BaseException:
            # Cleanup only - NOT close(): a failed connect must stay retryable,
            # so the closed latch is dispose()/close()'s alone.
            self._close_locked()
            raise

    def _spawn(self, bin_path: str, socket_path: str) -> subprocess.Popen:
        model = self._config.get("model", "")
        if not model:
            raise SidecarError(
                "no_model", "VM_AUDIOCPP_TTS_MODEL is not set: the TTS sidecar cannot "
                            "start without its GGUF")
        argv = [
            bin_path,
            "--socket", socket_path,
            "--model", model,
            "--backend", str(self._config.get("backend", "cuda")),
            "--device", str(self._config.get("device", 0)),
            "--threads", str(self._config.get("threads", 4)),
            # Seam-silence cap in ms; 0 disables the sidecar's trim.
            "--tts-seam-silence-ms", str(self._config.get("seam_silence_ms", 400)),
        ]
        # Operator overrides: already-parsed KEY=VALUE items from
        # VM_AUDIOCPP_TTS_REQUEST_OPTIONS, rendered as repeatable flags. Opaque: the
        # sidecar merges them over its registry defaults; nothing is interpreted here.
        for pair in self._config.get("request_options", []) or []:
            argv += ["--request-option", pair]
        try:
            return subprocess.Popen(argv)
        except OSError as exc:
            raise SidecarError("spawn_failed",
                               f"could not spawn TTS sidecar {bin_path!r}: {exc}") from exc

    def _dial(self, socket_path: str, *, deadline_s: float) -> socket.socket:
        deadline = time.monotonic() + deadline_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise SidecarError(
                    "load_failed",
                    f"TTS sidecar exited with status {self._proc.returncode} before "
                    "accepting a connection")
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            try:
                sock.connect(socket_path)
                return sock
            except OSError as exc:
                last_err = exc
                sock.close()
                time.sleep(0.1)
        raise SidecarError("connect_failed",
                           f"could not connect to TTS sidecar at {socket_path}: {last_err}")

    def _handshake(self) -> None:
        model_task = self._config.get("model_task", DEFAULT_TTS_MODEL_TASK)
        self._send(_HELLO, json.dumps({
            "protocol": _PROTOCOL_VERSION, "model_task": model_task}).encode("utf-8"))
        mtype, payload = self.recv_frame()
        if mtype == _ERROR:
            err = self._parse_json(payload)
            raise SidecarError(str(err.get("code", "error")), str(err.get("message", "")))
        if mtype != _READY:
            raise SidecarError("bad_handshake",
                               f"expected READY, got frame type 0x{mtype:02x}")
        ready = self._parse_json(payload)
        if ready.get("ok") is not True or ready.get("task") != "tts":
            raise SidecarError("bad_handshake", f"unusable TTS READY ack: {ready}")
        self._ready = ready

    def send_text(self, text: str, voice_wav: str, ref_text: str) -> int:
        """Dispatch one TEXT request; returns its wire generation (for a scoped
        cancel). Cleans a cancelled/abandoned predecessor off the wire first.

        The ref_text key is OMITTED when the transcript is empty - a
        voxcpm2 preset may carry no transcript, and the sidecar then omits the
        reference_text option key symmetrically. dots always has a non-empty
        transcript (the loader enforces it), so its frames are unchanged.
        """
        self.ensure_wire_clean()
        payload: dict[str, str] = {"text": text, "voice_wav": voice_wav}
        if ref_text and ref_text.strip():
            payload["ref_text"] = ref_text
        self._send(_TEXT, json.dumps(payload).encode("utf-8"))
        with self._state_lock:
            self._awaiting_terminal = True
            self._cancel_sent = False
            self._generation += 1
            return self._generation

    def ensure_wire_clean(self) -> None:
        """Drain a cancelled/abandoned request's leftover frames before dispatching.

        A cancelled synthesis task stops READING, but the sidecar's
        already-written frames (and the cancelled request's closing TTS_DONE) are
        still on the wire. If they were left there, the NEXT request would read
        them as its own audio. So: ask the sidecar to stop (idempotent), then read
        and discard until the cancelled request's terminal frame. Blocking - call
        from a worker thread; the sidecar answers a CANCEL within one vocoder
        merge group, so this drain is short.
        """
        if not self._awaiting_terminal:
            return
        self.request_cancel()
        while True:
            # The flag must be re-checked UNDER the read lock: a cancelled task's
            # abandoned recv (still blocked in its worker thread, holding the lock)
            # may consume the terminal frame itself while we wait - reading again
            # after that would block on a frame that never comes.
            with self._read_lock:
                if not self._awaiting_terminal:
                    return
                self._recv_frame_locked()  # discard a leftover frame

    def request_cancel(self, generation: int | None = None) -> None:
        """Ask the sidecar to stop the in-flight request. Best-effort,
        non-raising and at-most-once per request - safe to call inline from the
        unwind path of a cancelled synthesis task. When `generation` is given,
        the cancel applies ONLY if that request still owns the wire, so a doomed
        task waking up late cannot cancel its successor."""
        try:
            with self._state_lock:
                if generation is not None and generation != self._generation:
                    return
                if not self._awaiting_terminal or self._cancel_sent:
                    return
                self._cancel_sent = True
            self._send(_CANCEL, b"{}")
        except Exception:
            # A dead socket is the fault path's problem, not the cancel's.
            logger.debug("TTS cancel send failed (socket already down?)", exc_info=True)

    def _send(self, mtype: int, payload: bytes = b"") -> None:
        if self._sock is None:
            raise SidecarError("not_connected", "TTS sidecar is not connected")
        try:
            with self._send_lock:
                self._sock.sendall(_HEADER.pack(mtype, len(payload)) + payload)
        except OSError as exc:
            # A send failing at the socket level (EPIPE after a sidecar death) is
            # the same process-fatal shape as a read EOF.
            raise SidecarFault(f"TTS sidecar send failed: {exc}") from exc

    def recv_frame(self) -> tuple[int, bytes]:
        with self._read_lock:
            return self._recv_frame_locked()

    def _recv_frame_locked(self) -> tuple[int, bytes]:
        """Read one frame; caller holds _read_lock. The terminal-state update happens
        HERE, still under the lock, so a frame consumed by an abandoned (cancelled)
        reader updates the wire state before any other reader can check it."""
        header = self._recv_exactly(_HEADER.size)
        mtype, length = _HEADER.unpack(header)
        payload = self._recv_exactly(length) if length else b""
        if mtype in (_TTS_DONE, _ERROR):
            with self._state_lock:
                self._awaiting_terminal = False
                self._cancel_sent = False
        return mtype, payload

    def _recv_exactly(self, n: int) -> bytes:
        if self._sock is None:
            raise SidecarError("not_connected", "TTS sidecar is not connected")
        buf = bytearray()
        while len(buf) < n:
            try:
                data = self._sock.recv(n - len(buf))
            except socket.timeout as exc:
                # Socket-level failures mid-session mean the sidecar
                # PROCESS is dead or wedged - the process-fatal SidecarFault class,
                # like the STT client - not the refusal-class SidecarError.
                raise SidecarFault(f"TTS sidecar read timed out: {exc}") from exc
            except OSError as exc:
                raise SidecarFault(f"TTS sidecar read failed: {exc}") from exc
            if not data:
                raise SidecarFault(
                    "unexpected TTS sidecar EOF (treat as a fault)")
            buf.extend(data)
        return bytes(buf)

    @staticmethod
    def _parse_json(payload: bytes) -> dict:
        try:
            parsed = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def mark_closed(self) -> None:
        """Latch the client closed so any FUTURE connect is a no-op.

        Non-blocking and lock-free (a plain flag write), so dispose() can call it
        synchronously on the event loop BEFORE handing the real teardown to a
        thread - the warm-up connect racing that thread still sees the latch.
        """
        self._closed = True

    def close(self) -> None:
        """Best-effort, NON-BLOCKING teardown (safe to call on the event loop).

        Mirrors the STT client's close(): closing the socket gives
        the sidecar an EOF, which PROTOCOL.md treats exactly like a BYE (frees the
        session), so no blocking BYE write is needed. In owned mode the child is
        SIGTERMed here, synchronously and non-blockingly, and the wait -> SIGKILL ->
        wait escalation runs on a short-lived daemon thread - the WS finally-block
        calls dispose() without await, and a teardown that awaits can be cancelled
        out from under the reap when the endpoint task is torn down.
        """
        self.mark_closed()  # before the lock: a connect waiting on it must not spawn
        with self._lifecycle_lock:
            self._close_locked()

    def _close_locked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()  # EOF == BYE on this wire
            except OSError:
                pass
            self._sock = None
        self._ready = None
        if self._proc is not None:
            proc, self._proc = self._proc, None
            try:
                proc.terminate()
            except Exception:  # already exited, or an unsignalable state
                pass
            self._reaper = threading.Thread(target=self._reap, args=(proc,),
                                            name="audiocpp-tts-sidecar-reap", daemon=True)
            self._reaper.start()
        if self._owned_dir is not None:
            owned_dir, self._owned_dir = self._owned_dir, None
            shutil.rmtree(owned_dir, ignore_errors=True)

    @staticmethod
    def _reap(proc: subprocess.Popen) -> None:
        """wait() an already-SIGTERMed sidecar, escalating to SIGKILL. Off-loop."""
        try:
            proc.wait(timeout=5.0)
            return
        except subprocess.TimeoutExpired:
            pass
        logger.warning("TTS sidecar pid %s ignored SIGTERM; sending SIGKILL so it "
                       "cannot survive holding GPU memory", proc.pid)
        try:
            proc.kill()
        except Exception:
            pass
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.error("TTS sidecar pid %s survived SIGKILL; it may still hold "
                         "GPU memory", proc.pid)


class AudioCppTTSEngine(TTSEngine):
    """audio.cpp dots_tts engine behind the TTS sidecar.

    Genuinely streaming: stream_synthesize yields float32 chunks as TTS_AUDIO
    frames arrive over the socket, so is_batch stays False and the pipeline
    consumes the generator on the event loop while inference runs in the
    sidecar process. Requests on one engine instance are sequential - one
    connection is one session, and the WS layer already serializes TTS requests
    per session (a new request supersedes the in-flight one).
    """

    is_batch = False

    def __init__(self, config: dict | None = None):
        self._config = dict(config or {})
        self._presets: dict[str, PresetVoice] = {}
        self._default_voice: str = self._config.get("voice") or DEFAULT_VOICE
        # ref_text is per-model, derived from the selected model_task (a deployment-level
        # selection). dots and breeze (and any unknown task, conservatively) require the
        # transcript; voxcpm2 does not. This mirrors the sidecar registry descriptor's
        # ref_text_required flag without ever consulting the sidecar at load time.
        self._ref_text_required = (
            self._config.get("model_task", DEFAULT_TTS_MODEL_TASK) != VOXCPM2_TTS_MODEL_TASK
        )
        self._client = _TtsSidecarClient(self._config)
        # The WS layer's fault surface. The synthesis runs in a
        # detached task the receive loop never awaits, so a SidecarFault raised
        # there would otherwise die in a done-callback log while the client hangs
        # with an audio_start and no terminal frame. stream_synthesize awaits this
        # (in the task's own context, before re-raising) so the session can send
        # its defined error frame and close - this session only, per-session
        # sidecar means per-session blast radius.
        self.on_fault: Callable[[SidecarFault], Awaitable[None]] | None = None

    def load(self, device: str = "", voices_path: str | None = None) -> None:
        """Validate the preset voice assets, failing fast on any bad entry.

        `device` is accepted for TTSEngine surface compatibility and unused: the
        sidecar owns the GPU (VM_AUDIOCPP_DEVICE side), not this process.
        """
        del device
        refs = voices_path or self._config.get("voices", "voices/designed/refs.json")
        self._presets = load_presets(refs, ref_text_required=self._ref_text_required)
        # The configured default must itself resolve, or a no-voice request
        # would fail at request time - exactly what fail-fast exists to prevent.
        resolve_preset(self._presets, self._default_voice)
        logger.info("AudioCppTTSEngine loaded: %d presets, default=%s",
                    len(self._presets), self._default_voice)

    def session_engine(self) -> "AudioCppTTSEngine":
        """A per-WS-session engine: own sidecar client, shared validated presets.

        One TTS sidecar process per WebSocket session: the
        startup-loaded engine only validates config and assets; each session
        constructs its own instance from it, so sessions never share a sidecar
        and a crash kills exactly one session's process.
        """
        engine = AudioCppTTSEngine(self._config)
        engine._presets = self._presets  # validated once at startup
        return engine

    def connect(self) -> None:
        """Connect this engine's sidecar now (spawn + handshake in owned mode).

        Blocking - call from a worker thread. Idempotent, so the WS-connect
        warm-up and the first request's own connect can race safely.
        """
        self._client.connect()

    async def ensure_ready(self) -> None:
        """Connect (idempotent, off-loop) so `sample_rate` is the READY-stated one.

        audio_start must carry READY's rate, and the WS-connect warm-up is a
        background task a fast first request can beat. The reply paths await this
        BEFORE building audio_start.
        Connect failures propagate unchanged (SidecarError); callers that must still end
        the request with an error audio_end catch here and let the dispatch itself
        re-raise the same failure after audio_start.
        """
        await asyncio.to_thread(self._client.connect)

    @property
    def sample_rate(self) -> int:
        """The READY-stated output rate once connected (48000 for dots and voxcpm2,
        24000 for breeze); before that, the configured
        model_task's registry rate, so /info is right without a sidecar."""
        ready = self._client.ready
        if ready and isinstance(ready.get("audio"), dict):
            return int(ready["audio"].get("sample_rate", TTS_SAMPLE_RATE))
        model_task = self._config.get("model_task", DEFAULT_TTS_MODEL_TASK)
        return MODEL_TASK_SAMPLE_RATES.get(model_task, TTS_SAMPLE_RATE)

    @property
    def available_voices(self) -> list[str]:
        """Names of the loaded voice presets, sorted; empty before load().

        The WS handshake and /info report this list for the audiocpp engine, the
        same way they do for VibeVoice.
        """
        return sorted(self._presets)

    @property
    def default_voice(self) -> str:
        """The preset a request that names no voice gets (the configured default)."""
        return self._default_voice

    def _resolve(self, voice: str | None) -> PresetVoice:
        if not self._presets:
            raise SidecarError("not_loaded", "AudioCppTTSEngine.load() was never called")
        return resolve_preset(self._presets, voice or self._default_voice)

    def resolve_voice(self, voice: str | None) -> PresetVoice:
        """The preset a request with this voice name clones from (None: the default).

        The stream pipeline reads the preset's clip and transcript from it to
        compose the reference for a later clause piece. Raises UnknownVoiceError
        for a name that is not a loaded preset, as a dispatch would.
        """
        return self._resolve(voice)

    @staticmethod
    def _payload_to_float32(payload: bytes) -> np.ndarray:
        # s16le -> float32 by the engine's own /32768 convention. A dtype view plus
        # a scale - never a resample.
        return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0

    async def stream_synthesize(self, text: str, **kwargs) -> AsyncIterator[np.ndarray]:
        """Yield float32 chunks at the sidecar's 48 kHz as TTS_AUDIO frames arrive.

        kwargs:
            voice: preset name from voices/designed/refs.json (default: the
                   configured default preset). Unknown names raise
                   UnknownVoiceError BEFORE anything is dispatched.
            ref_wav: a reference WAV to clone from instead of the preset's clip,
                   with ref_text as its transcript. The stream pipeline passes a
                   composed reference for a later clause piece; without it the
                   preset's own clip and transcript ride the request.
        """
        preset = self._resolve(kwargs.get("voice"))
        ref_wav = kwargs.get("ref_wav") or preset.wav_path
        ref_text = kwargs.get("ref_text") if kwargs.get("ref_wav") else preset.ref_text
        client = self._client
        # The wire generation OUR dispatch owns, captured so the
        # unwind below can scope its cancel. None until send_text returns - an
        # unwind before dispatch has nothing of its own to cancel.
        generation: int | None = None
        try:
            await asyncio.to_thread(client.connect)
            # ref_text rides every dispatch the model requires it for:
            # dots always has a non-empty transcript (the loader enforces it); a
            # voxcpm2 preset may omit it and send_text then drops the key.
            generation = await asyncio.to_thread(client.send_text, text,
                                                 ref_wav, ref_text or "")
            while True:
                mtype, payload = await asyncio.to_thread(client.recv_frame)
                if mtype == _TTS_AUDIO:
                    if payload:
                        yield self._payload_to_float32(payload)
                elif mtype == _TTS_DONE:
                    return
                elif mtype == _ERROR:
                    err = _TtsSidecarClient._parse_json(payload)
                    raise SidecarError(str(err.get("code", "error")),
                                       str(err.get("message", "")))
                else:
                    raise SidecarError(
                        "bad_frame",
                        f"unexpected frame type 0x{mtype:02x} in a TTS reply")
        except (asyncio.CancelledError, GeneratorExit):
            # The consuming task was cancelled (a tts_cancel or a
            # superseding request), or the abandoned generator is being finalized.
            # Ask the sidecar to stop generating NOW - the sink-throw lands within
            # one vocoder merge group - and leave the leftover frames on the wire:
            # the next dispatch's ensure_wire_clean drains to the cancelled
            # request's TTS_DONE off-loop, so stale audio can never prefix a later
            # request. request_cancel is a tiny non-raising send, safe inline here
            # (no awaiting during GeneratorExit). Scoped to OUR generation:
            # a doomed task finalizing late - after a superseding send_text took
            # the wire - must not cancel the successor; nothing dispatched means
            # nothing to cancel.
            if generation is not None:
                client.request_cancel(generation)
            raise
        except SidecarFault as exc:
            # This session's sidecar process died mid-request. Let
            # the WS layer surface the defined session error from the task's own
            # context, then re-raise so the synthesis task still fails cleanly.
            # ERROR-frame replies and refusals (SidecarError) pass through above:
            # they are request- or config-scoped, not a process death.
            if self.on_fault is not None:
                await self.on_fault(exc)
            raise

    def synthesize(self, text: str, speaker_key: str = "EN-US", speed: float = 1.0,
                   voice: str | None = None) -> tuple[np.ndarray, int]:
        """Blocking batch synthesis: the full request collected into one array.

        MUST be called from a worker thread/executor - it blocks on socket I/O.
        `speaker_key` and `speed` are surface-compatibility parameters; voice
        selection is the `voice` preset name.
        """
        del speaker_key, speed
        preset = self._resolve(voice)
        client = self._client
        client.connect()
        client.send_text(text, preset.wav_path, preset.ref_text)
        chunks: list[np.ndarray] = []
        while True:
            mtype, payload = client.recv_frame()
            if mtype == _TTS_AUDIO:
                if payload:
                    chunks.append(self._payload_to_float32(payload))
            elif mtype == _TTS_DONE:
                break
            elif mtype == _ERROR:
                err = _TtsSidecarClient._parse_json(payload)
                raise SidecarError(str(err.get("code", "error")),
                                   str(err.get("message", "")))
            else:
                raise SidecarError("bad_frame",
                                   f"unexpected frame type 0x{mtype:02x} in a TTS reply")
        audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return audio, self.sample_rate

    def dispose(self) -> None:
        """End the sidecar session. Non-blocking, safe on the event loop.

        The teardown itself runs on a daemon thread: close() serializes against a
        warm-up connect via the lifecycle lock, and that connect can legitimately
        hold the lock for a model-load's duration - the event loop must never wait
        on it. close() then signals the child and hands the reap to its own daemon
        thread, so nothing here blocks and nothing is left unreaped.

        The closed latch is set HERE, synchronously, not on the teardown thread:
        a warm-up connect whose worker starts after dispose() returns must
        already see it, or it would spawn an orphan sidecar.
        """
        self._client.mark_closed()
        threading.Thread(target=self._client.close,
                         name="audiocpp-tts-dispose", daemon=True).start()


class _ClauseReference:
    """One clause request's composed-reference state.

    Collects the audio of the engine request in flight (int16, capped at the tail
    length as it arrives, so a long piece never holds more than the cap), promotes
    it to the previous piece when that request completes, and remembers every
    composed WAV it wrote so the request's end can remove them.
    """

    def __init__(self, preset: PresetVoice) -> None:
        self.preset = preset
        self.preset_clip: np.ndarray | None = None  # read once, at the engine rate
        self.cap = 1  # samples of the previous piece kept; set per engine request
        self.previous_text: str | None = None
        self.previous_tail: list[np.ndarray] = []
        self.paths: list[str] = []
        self._current: list[np.ndarray] = []
        self._current_samples = 0

    def start_piece(self) -> None:
        self._current = []
        self._current_samples = 0

    def add_audio(self, chunk: np.ndarray) -> None:
        if len(chunk) == 0:
            return
        self._current.append(_float32_to_int16(chunk))
        self._current_samples += len(chunk)
        # Keep the end: drop whole leading chunks while the rest still covers the cap.
        while self._current and self._current_samples - len(self._current[0]) >= self.cap:
            self._current_samples -= len(self._current.pop(0))

    def finish_piece(self, text: str) -> None:
        self.previous_text = text
        self.previous_tail = self._current
        self._current = []
        self._current_samples = 0


class AudioCppTTSStreamPipeline(TTSStreamPipeline):
    """tts_stream batching for the audiocpp engine.

    The base TTSStreamPipeline queues each detected sentence and its synthesis
    loop consumes them ONE per stream_synthesize call - under audiocpp that
    would mean one sidecar request per sentence, a seam at every boundary.
    This engine maps it differently: the first completed sentence dispatches immediately
    (that part is inherited - the consumer picks it up the moment it is
    queued), and everything that accumulates while a request synthesizes
    coalesces into exactly ONE next request. Text inside one request flows
    through dots' internal chunker with continuous vocoder state - the smooth
    seam path.

    RTF < 1 cuts the other way from what that mapping assumed, though: because
    each request outruns its own playback, the engine is idle and waiting every
    time a sentence completes, so a producer that forwards tokens as they arrive
    leaves nothing queued to coalesce and every sentence becomes its own request
    after all. `coalesce_ms` closes that hole by waiting a bounded moment for
    more sentences once the first batch is away - see the comment at the drain
    itself for the measurements.

    Only the synthesis loop is overridden, and only the drain step differs:
    after taking one sentence it drains everything else already queued into the
    same request, plus whatever arrives inside the coalescing window. feed_text, the flush timeout, cancel/abandon
    and the audio_start/audio_end framing all keep the base behavior, and the
    base class itself is untouched - other engines' tts_stream behavior stays
    byte-identical. This class is selected only when the audiocpp
    engine is (the websocket layer's engine selection branch).

    A chunking="clause" request needs nothing extra beyond the inherited
    chunker and timer: the loop reports each PCM chunk it sends, as the base
    loop does, and clause pieces that are cut while a request synthesizes are
    queued like sentences, so the same drain merges them into one next request.

    With `clause_continue` on, a clause request's later engine requests clone
    from a composed reference (see the module's composed-reference section)
    instead of the preset: the preset clip plus the previous piece's own audio
    tail, and the preset transcript plus that piece's text. The first piece,
    sentence requests and the setting-off case send the preset as before.
    """

    def __init__(self, *args, coalesce_ms: int = 0, clause_continue: bool | None = None,
                 clause_tail_ms: int | None = None, start_buffer_ms: int | None = None,
                 **kwargs) -> None:
        """Same surface as the base pipeline plus the coalescing window.

        `coalesce_ms` defaults to 0 - the drain-what-is-already-queued behavior
        described above - so constructing this class without it is byte-identical
        to what shipped. The websocket layer passes VM_AUDIOCPP_TTS_COALESCE_MS.

        `clause_continue` and `clause_tail_ms` default to the settings
        (VM_AUDIOCPP_TTS_CLAUSE_CONTINUE, VM_AUDIOCPP_TTS_CLAUSE_TAIL_MS) when
        not given, so the websocket layer need not know about them.
        `start_buffer_ms` likewise defaults to VM_AUDIOCPP_TTS_START_BUFFER_MS.
        """
        super().__init__(*args, **kwargs)
        self._coalesce_ms = max(0, int(coalesce_ms))
        if clause_continue is None or clause_tail_ms is None or start_buffer_ms is None:
            settings = get_settings()
            if clause_continue is None:
                clause_continue = settings.audiocpp_tts_clause_continue
            if clause_tail_ms is None:
                clause_tail_ms = settings.audiocpp_tts_clause_tail_ms
            if start_buffer_ms is None:
                start_buffer_ms = settings.audiocpp_tts_start_buffer_ms
        self._clause_continue = bool(clause_continue)
        self._clause_tail_ms = max(1, int(clause_tail_ms))
        self._start_buffer_ms = max(0, int(start_buffer_ms))

    @property
    def clause_continue(self) -> bool:
        """Whether later clause pieces clone from a composed reference."""
        return self._clause_continue

    @property
    def clause_tail_ms(self) -> int:
        """How much of the previous piece's audio (its end) joins that reference."""
        return self._clause_tail_ms

    @property
    def start_buffer_ms(self) -> int:
        """How much audio each engine request holds before its first send."""
        return self._start_buffer_ms

    async def _write_clause_reference(self, state: "_ClauseReference", loop,
                                      sample_rate: int) -> str:
        """Compose the reference for the next piece into a new temp WAV, off-loop.

        The path is minted and registered on the loop BEFORE the thread runs, so
        the request's cleanup already knows it. A cancel that lands while the
        thread is still writing leaves the removal to the write's own completion.
        """
        fd, path = tempfile.mkstemp(prefix="ref-", suffix=".wav", dir=clause_ref_dir())
        os.close(fd)
        state.paths.append(path)
        preset = state.preset
        tail_chunks = state.previous_tail
        cap = state.cap

        def work() -> None:
            if state.preset_clip is None:
                state.preset_clip = read_preset_clip(preset.wav_path, sample_rate)
            tail = np.concatenate(tail_chunks)[-cap:]
            write_composed_reference(path, state.preset_clip, tail, sample_rate)

        future = loop.run_in_executor(None, work)
        try:
            await future
        except asyncio.CancelledError:
            future.add_done_callback(lambda _f: _remove_quietly(path))
            raise
        return path

    async def _synthesis_loop(self, websocket, request_id: str) -> None:
        from voicemanager.protocol import AudioStartMessage, AudioEndMessage

        engine = self._tts_pipeline._engine
        loop = asyncio.get_running_loop()
        batches = 0
        ended = False
        clause = self._chunking == "clause"
        # Composed-reference state for this request; None when the feature is
        # off, on a sentence request, or when the engine has no presets to read.
        refs: _ClauseReference | None = None
        if clause and self._clause_continue and hasattr(engine, "resolve_voice"):
            try:
                preset = engine.resolve_voice(self._voice or self._tts_pipeline._voice)
            except Exception:
                # An unknown voice fails at the dispatch below with the defined
                # error; the reference composition just stays out of its way.
                preset = None
            if preset is not None:
                refs = _ClauseReference(preset=preset)
        try:
            while not ended:
                sentence = await self._sentence_queue.get()
                if sentence is None:
                    break  # no more sentences
                # Coalesce everything already queued into this ONE request.
                # get_nowait only - the drain must never WAIT for more text (that
                # would gate dispatch on future chunks, the exact anti-pattern
                # ruled out for the first sentence, which dispatches immediately).
                parts = [sentence]
                while True:
                    try:
                        queued = self._sentence_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if queued is None:
                        ended = True  # done arrived: this batch is the last
                        break
                    parts.append(queued)

                # Once the first batch is away, keep draining for a bounded
                # window instead of dispatching whatever happened to be queued at
                # this instant. A token-by-token producer
                # - HermesConsole forwards every Hermes delta verbatim, with no
                # client-side sentence assembly by design - completes one sentence
                # at a time, and at RTF ~0.6 synthesis always outruns arrival, so
                # the get_nowait drain above reliably finds the queue empty. Every
                # sentence then becomes its own sidecar request, and because Breeze
                # re-conditions the speaker from the reference per request, each one
                # is a fresh sample of the voice: the timbre audibly walks between
                # sentences. Measured mean MFCC spread over the same four sentences
                # was 29.5 in one request against 48.6 across four (29.5 being the
                # floor from differing phonetic content alone).
                #
                # The window is spent against the PREVIOUS batch's playback, not
                # against silence: at RTF ~0.6 a dispatched batch buys roughly 40%
                # of its own duration in slack before the client runs dry, so a few
                # hundred ms is inaudible. The first batch never waits, a
                # `done` sentinel short-circuits the window so the tail never
                # stalls, and cancellation still lands promptly because the wait is
                # an awaited wait_for, not a sleep.
                if self._coalesce_ms and batches and not ended:
                    deadline = loop.time() + self._coalesce_ms / 1000.0
                    while (remaining := deadline - loop.time()) > 0:
                        try:
                            queued = await asyncio.wait_for(
                                self._sentence_queue.get(), timeout=remaining)
                        except asyncio.TimeoutError:
                            break
                        if queued is None:
                            ended = True  # done arrived: this batch is the last
                            break
                        parts.append(queued)

                text = " ".join(p.strip() for p in parts if p and p.strip())
                if not text:
                    continue

                if not self._audio_started:
                    # The rate in audio_start is
                    # READY's, so connect first. A connect failure is left for
                    # stream_synthesize below to raise again - after audio_start,
                    # so the terminal framing in `finally` still goes out.
                    try:
                        await engine.ensure_ready()
                    except Exception:
                        logger.debug("TTS sidecar connect before audio_start failed; "
                                     "the dispatch will surface it", exc_info=True)
                    await websocket.send_text(
                        AudioStartMessage(
                            request_id=request_id,
                            sample_rate=engine.sample_rate,
                            channels=1,
                            bit_depth=16,
                            encoding="pcm",
                        ).model_dump_json()
                    )
                    self._audio_started = True

                kwargs = {
                    "speaker_key": self._tts_pipeline._speaker_key,
                    "speed": self._tts_pipeline._speed,
                }
                effective_voice = self._voice or self._tts_pipeline._voice
                if effective_voice is not None:
                    kwargs["voice"] = effective_voice
                ref_kind = "preset"
                if refs is not None:
                    sample_rate = engine.sample_rate
                    refs.cap = max(1, sample_rate * self._clause_tail_ms // 1000)
                    if refs.previous_text is not None and refs.previous_tail:
                        # A later piece: clone from the preset clip plus the previous
                        # piece's audio tail, and its transcript plus that piece's
                        # text, so Breeze re-conditions on the voice it just spoke.
                        try:
                            kwargs["ref_wav"] = await self._write_clause_reference(
                                refs, loop, sample_rate)
                            kwargs["ref_text"] = (
                                refs.preset.ref_text + " " + refs.previous_text)
                            ref_kind = "composed"
                        except Exception:
                            # The preset clip is unreadable or the temp dir is not
                            # writable: the piece still speaks, from the preset.
                            logger.warning(
                                "Composed reference failed for request_id=%s; "
                                "the piece clones from the preset", request_id,
                                exc_info=True)
                    refs.start_piece()
                if clause:
                    logger.info(
                        "Clause TTS engine request: request_id=%s parts=%d text=%r ref=%s",
                        request_id, len(parts), text, ref_kind,
                    )
                # Start buffer: the first frames of a synthesis arrive at about
                # real-time speed, so a client that plays on arrival runs dry
                # between them. Hold this request's audio until the buffer is full
                # and send it as one chunk; later chunks go out as they arrive.
                # Audio still held when a cancel lands is dropped with the task.
                hold_samples = engine.sample_rate * self._start_buffer_ms // 1000
                held: list[np.ndarray] = []
                held_samples = 0
                dispatched_at = time.monotonic()
                first_chunk = True
                async for chunk_np in engine.stream_synthesize(text, **kwargs):
                    if first_chunk and clause:
                        first_chunk = False
                        logger.info(
                            "Clause TTS first audio: request_id=%s first_audio_ms=%d",
                            request_id, int((time.monotonic() - dispatched_at) * 1000),
                        )
                    if refs is not None:
                        refs.add_audio(chunk_np)
                    if hold_samples > 0:
                        held.append(chunk_np)
                        held_samples += len(chunk_np)
                        if held_samples < hold_samples:
                            continue
                        chunk_np = np.concatenate(held)
                        held = []
                        hold_samples = 0  # buffer sent: the rest pass through
                    await websocket.send_bytes(float32_to_pcm16(chunk_np))
                    # Clause mode's unplayed-audio estimate needs every chunk sent;
                    # a no-op on a sentence request. Clause pieces cut while this
                    # request synthesizes queue up and the drain above merges them.
                    self._report_audio(request_id, len(chunk_np), engine.sample_rate)
                if held:
                    # The request ended before the buffer filled: send what it made.
                    block = np.concatenate(held)
                    await websocket.send_bytes(float32_to_pcm16(block))
                    self._report_audio(request_id, len(block), engine.sample_rate)
                if refs is not None:
                    refs.finish_piece(text)
                batches += 1
                logger.debug(
                    "Streamed coalesced TTS batch: request_id=%s parts=%d text=%s",
                    request_id, len(parts), text[:50],
                )
        finally:
            # The composed references belong to this request: gone on done,
            # cancel, the flush timeout and disconnect alike. Unlinks only, so
            # this stays safe inline on a cancellation unwind.
            if refs is not None:
                for path in refs.paths:
                    _remove_quietly(path)
                refs.paths.clear()
            # Same terminal framing as the base loop: a request that never
            # produced audio still gets its audio_start/audio_end pair.
            if not self._audio_started:
                await websocket.send_text(
                    AudioStartMessage(
                        request_id=request_id,
                        sample_rate=engine.sample_rate,
                    ).model_dump_json()
                )
            try:
                await websocket.send_text(
                    AudioEndMessage(request_id=request_id).model_dump_json()
                )
            except Exception:
                logger.debug("Could not send audio_end for request_id=%s", request_id)
            logger.debug(
                "TTS stream complete: request_id=%s coalesced_batches=%d",
                request_id, batches,
            )
