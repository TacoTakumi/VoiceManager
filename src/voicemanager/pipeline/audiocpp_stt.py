"""audio.cpp streaming STT pipeline over the sidecar socket protocol.

Per-connection wrapper that implements the SAME six-method streaming contract as
nemotron_stt.py / qwen_stt.py -- async feed(pcm)->bool, get_current_text()->(text,
lang,seg), take_final()->(...)|None, async finalize()->(...), reset(), dispose() --
so transport/websocket.py and the turn-taking machinery are reused unchanged.
The only edit to websocket.py is the per-connection pipeline-selection branch; this
module is the engine behind it.

The audio.cpp runtime is native CUDA code whose fatal faults would take the FastAPI
server down, so it runs in a separate **sidecar process** reached over an AF_UNIX
socket (crash isolation). This pipeline owns that sidecar client: it speaks
the length-prefixed wire protocol in sidecar/PROTOCOL.md (HELLO/READY handshake, then
AUDIO -> EVENT interim deltas, FLUSH -> exactly one FINAL, RESET between utterances,
BYE/EOF to close). The sidecar drives audio.cpp through its generic runtime interface
(IStreamingVoiceTaskSession) keyed by the wire `model_task` id, so this
client is model-agnostic: EVENT partial_text arrives as token/char deltas (PROTOCOL.md)
and is accumulated into the utterance transcript-so-far; is_final=true
marker events and FINAL frames bound the utterance.

Off-loop discipline: every blocking sidecar round-trip -- connect+handshake,
AUDIO send + reply drain, FLUSH + FINAL read, RESET, close -- is awaited via
asyncio.to_thread, never run on the event loop. The two sync methods the WS handler
calls without await (reset, dispose) must not do a blocking socket round-trip on the
loop, so reset() *defers* its RESET frame to the next off-loop op (feed/finalize) and
dispose() only does the fast, non-blocking close (socket close is EOF == BYE, PROTOCOL).

Standard audio contract: the raw 16 kHz mono 16-bit PCM the WS first-frame
validation already accepts is forwarded as the AUDIO payload byte-for-byte -- no
re-encode at the boundary. The first-frame validation path stays in websocket.py,
unchanged.

Endpointing (when to FLUSH to produce a final) stays on VoiceManager's side (audio.cpp
emits no native acoustic EoU); this module implements the FLUSH/FINAL *mechanics*.
Mapping interim deltas to is_final=False and finals to is_final=True over /ws/stream
happens in transport/websocket.py, driven through this same contract.
"""
import asyncio
import json
import logging
import os
import select
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000  # standard audio contract; PCM is forwarded, not re-encoded
PROTOCOL_VERSION = 1

# Teardown escalation windows. A sidecar mid-CUDA-decode
# may not act on SIGTERM at once, so give it a grace period before SIGKILL; the kill
# itself is uninterruptible, so its wait only guards against an unkillable-state hang.
_TERM_GRACE_S = 5.0
_KILL_GRACE_S = 5.0

# Wall-clock deadline for the whole FLUSH -> FINAL exchange. The per-recv socket
# timeout (120 s, set at connect) tolerates one long silent pause -- finalize() runs a
# real GPU decode -- but every EVENT frame RESETS it, so a misbehaving sidecar dripping
# EVENTs and never a FINAL would keep _finalize_sync alive forever. PROTOCOL.md
# guarantees exactly one FINAL per FLUSH, so this is a defensive bound, not a protocol
# fix: "a defined exception rather than a hang" is the contract for sidecar
# misbehaviour, and this closes the one remaining hang. 300 s = 2.5x the per-recv
# timeout: a legit single decode fits with room to spare (its pauses are already capped
# at 120 s each by the socket timeout), while any exchange still FINAL-less after five
# minutes has long since failed the session's real-time contract.
_FLUSH_FINAL_DEADLINE_S = 300.0

# Message type tags -- must match sidecar/src/protocol.h and sidecar/PROTOCOL.md.
HELLO = 0x01
READY = 0x02
AUDIO = 0x03
EVENT = 0x04
FLUSH = 0x05
FINAL = 0x06
RESET = 0x07
ERROR = 0x08
BYE = 0x09

_HEADER = struct.Struct(">BI")   # type:u8, length:u32 big-endian (network order)
_MAX_FRAME = 0x0100_0000         # 16 MiB cap (PROTOCOL.md framing); reject larger
_RECV_CHUNK = 65536


# The OPERATOR-CONFIGURATION error codes: this server cannot serve this engine until a
# human changes the VM_AUDIOCPP_* settings, so every reconnect really does fail the same
# way and the WS handler is right to tell the client to stop retrying (close 4001).
#
# An explicit ALLOW-LIST, not "everything except the transient ones", so a code we have
# not classified fails SAFE. `SidecarError` covers both classes and the two ask a client
# for opposite behaviour, so a misclassification is not symmetric: calling a permanent
# fault retriable costs a few doomed reconnects that each still surface the cause code,
# while calling a transient one permanent tells a working deployment to give up and page
# an operator (the regression this taxonomy exists to fix).
# A new ERROR code added upstream in sidecar/PROTOCOL.md therefore lands on the harmless
# side until someone deliberately classifies it here.
#
# `at_capacity` (raised by SidecarSlots rather than by the sidecar) is deliberately
# ABSENT: a full server is the textbook transient condition -- nothing is misconfigured
# and it clears the moment any session ends -- so it must reach the client as 4002
# "retry with backoff", never 4001 "stop retrying and page an operator".
PERMANENT_ERROR_CODES = frozenset({
    "no_sidecar",           # nothing configured to dial or spawn
    "spawn_failed",         # sidecar binary missing / not executable
    "dial_failed",          # no listener at the configured socket
    "load_failed",          # the GGUF is missing or unreadable
    "unknown_model_task",   # the configured model_task id is not registered
    "unsupported_protocol", # sidecar speaks a different wire version
})

# Mid-stream codes the SIDECAR ITSELF recovers from: drive_loop.cpp emits `internal` from
# the catch block around session.finalize() and then CONTINUES its loop, still serving.
# Treating those as fatal on this side would be a protocol-conformance divergence -- the
# C++ drive loop calls them recoverable -- and would cost a whole session over one bad
# utterance. Absorbed by _absorb()/_finalize_sync(), which drop that utterance and stream
# on; every other mid-stream code still raises.
RECOVERABLE_STREAM_ERROR_CODES = frozenset({"internal"})


class SidecarError(Exception):
    """The sidecar replied with a protocol/runtime ERROR frame, or a bad frame."""

    def __init__(self, code: str, message: str = ""):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}" if message else code)

    @property
    def is_permanent(self) -> bool:
        """True if only an operator config change can fix this -- retrying is futile.

        The WS handler picks the close code from this: permanent -> 4001
        `stt_backend_unavailable` ("do not retry"), transient -> 4002
        `stt_backend_unready` ("retry with backoff"). It lives on the exception rather
        than in the handler because the exception is what carries the code, and the
        pipeline is where the codes are known.
        """
        return self.code in PERMANENT_ERROR_CODES


class SidecarSlots:
    """Bounds how many OWNED sidecars may exist at once.

    One sidecar per session is deliberate and must not be reverted -- it is what makes
    per-session fault isolation real, since a process-fatal audio.cpp fault can then only
    reach one session. But that design has a price the operator has to be told about:
    each concurrent session is a process, a CUDA context and a resident f16 GGUF on the
    SAME `VM_AUDIOCPP_DEVICE` (measured ~1.8 GiB each at first, ~1.5 GiB on the current
    pin), so concurrency is bounded by that one GPU's VRAM whether or not anyone counts it.

    Without a cap the bound still existed, it was just discovered by hitting it: the Nth
    session exhausted the GPU, `open_streaming_session` threw, the sidecar answered ERROR
    `load_failed`, and the WS handler dutifully mapped that to close 4001
    `stt_backend_unavailable` -- telling the client "this server is misconfigured, do not
    retry". A correctly configured, merely-full deployment was reported as a permanent
    operator fault, and the condition it was told not to retry would have cleared by
    itself the moment any session ended. Counting admissions ourselves is what lets the
    refusal be honest: `at_capacity`, on the retriable side of the error-code taxonomy above.

    The count tracks the PROCESS, not the socket: released by the reaper once the sidecar
    has actually exited (see _SidecarClient.close()), because that is when the VRAM is
    genuinely free. A limit of 0 disables the cap (mirrors VM_STT_ENDPOINT_HOLD_MS=0).

    One instance is created per server in `_load_audiocpp_models` and shared through the
    `audiocpp_config` dict, so admissions are counted across all connections without a
    module-level mutable global. Thread-safe: acquire/release run in the pipeline's
    worker threads, release also from a reaper thread.
    """

    def __init__(self, limit: int):
        self._limit = max(0, int(limit))
        self._lock = threading.Lock()
        self._in_use = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_use(self) -> int:
        with self._lock:
            return self._in_use

    def acquire(self) -> None:
        """Reserve one sidecar slot, or raise SidecarError `at_capacity` if full."""
        with self._lock:
            if self._limit and self._in_use >= self._limit:
                raise SidecarError(
                    "at_capacity",
                    f"all {self._limit} audio.cpp sidecar slots are in use "
                    f"(VM_AUDIOCPP_MAX_SIDECARS); each concurrent session runs its own "
                    f"sidecar process holding a CUDA context and a resident model on "
                    f"VM_AUDIOCPP_DEVICE, so this is a capacity limit, not a "
                    f"misconfiguration -- retry when a session ends, or raise the cap if "
                    f"the device has the VRAM headroom")
            self._in_use += 1

    def release(self) -> None:
        with self._lock:
            if self._in_use > 0:
                self._in_use -= 1


class SidecarFault(Exception):
    """The sidecar connection failed unexpectedly (ERROR sidecar_fault or a bare EOF).

    Per PROTOCOL.md, a process-fatal audio.cpp fault best-effort sends ERROR
    `sidecar_fault` then dies, but the frame may be lost (fault mid-write / SIGKILL), so
    the client MUST treat an unexpected socket EOF the same as a fault. The WS layer maps
    this to its error-close; this module only guarantees a defined exception, not a hang.
    """


class _SidecarClient:
    """Owns one sidecar connection: spawn (if needed), socket, and wire framing.

    All methods block and are meant to be called from a worker thread (the pipeline
    wraps them in asyncio.to_thread). One instance drives exactly one streaming session
    (one WS connection), mirroring PROTOCOL.md's "one connection is one session".

    Ownership is one of two modes, decided by whether a socket path is configured, and
    a session is never in both. See connect() for why sharing is not an option.
    """

    def __init__(self, config: dict | None):
        self._config = config or {}
        self._sock: socket.socket | None = None
        self._proc: subprocess.Popen | None = None  # set only if THIS client spawned it
        self._owned_dir: str | None = None           # private dir holding our socket path
        self._reaper: threading.Thread | None = None  # off-loop wait()er started by close()
        self._rbuf = bytearray()                     # bytes read but not yet framed
        self._ready: dict | None = None              # the READY ack (audio contract, etc.)
        # Shared concurrent-sidecar budget. None means uncapped: only reachable for
        # a config dict not built by _load_audiocpp_models (hand-built test configs), so
        # tests written before the cap keep working rather than failing closed on a missing key.
        self._slots: SidecarSlots | None = self._config.get("slots")
        self._holds_slot = False                     # True between acquire and release

    @property
    def is_connected(self) -> bool:
        return self._sock is not None

    # --- connect / handshake ------------------------------------------------------
    def connect(self) -> None:
        """Connect to this session's sidecar and handshake. Idempotent once connected.

        ONE SIDECAR PROCESS PER SESSION. Sessions never share a sidecar.
        A sidecar process serves exactly one streaming session at a time -- it admits a
        connection under a process-wide lock and refuses a concurrent second one with
        ERROR `busy` (sidecar/src/main.cpp) -- and, more fundamentally, sharing would
        defeat the crash isolation the process boundary exists for: a process-fatal
        audio.cpp fault (GGML_ASSERT -> abort, SIGSEGV) kills every session on that
        process, so concurrent sessions stay unaffected only while a
        fault can reach exactly one session. Concurrency at the /ws/stream level comes
        from each session running its own sidecar, not from one sidecar multiplexing.

        Two modes, chosen by whether `socket` is configured -- never both, so there is
        no ownership to get wrong:

          * OWNED (production default, no `socket`): mint a PRIVATE socket path, spawn
            our own sidecar on it, own it exclusively (terminate + remove the path on
            close). Two sessions can neither collide on a path nor kill each other's
            process, because neither knows the other's path.
          * EXTERNAL (`socket` set): an operator-managed listener -- the fake-sidecar
            tests, and the out-of-band warm sidecar used for measurements. Dial
            only: never spawn, never terminate. It is single-session, so a second
            concurrent consumer of one external sidecar gets a defined ERROR `busy`
            rather than a silent 120 s starvation.
        """
        if self._sock is not None:
            return
        socket_path = self._config.get("socket", "")
        bin_path = self._config.get("sidecar_bin", "")
        if not socket_path and not bin_path:
            raise SidecarError(
                "no_sidecar",
                "no audio.cpp sidecar to use: set VM_AUDIOCPP_SIDECAR_BIN to spawn one "
                "per session, or VM_AUDIOCPP_SOCKET to dial an externally-managed one")
        try:
            # A connect() that raises must leave NOTHING behind.
            # The guard covers the WHOLE spawn + dial + handshake sequence, not just the
            # handshake: the private dir is ours from the moment it is
            # minted and the sidecar process is ours from the moment it spawns, so a dial
            # that fails after a successful spawn owns a live process, an un-reaped child
            # and a temp dir with no other teardown on that path. A leaked SOCKET is the
            # sharpest of the three, because this method's contract is "idempotent once
            # connected" -- it early-returns on `_sock is not None` -- so a retried feed()
            # would skip the handshake and send AUDIO on a socket the sidecar never
            # READYed. close() is the right teardown rather than dropping the references:
            # the child must be signalled and reaped and the private path removed, exactly
            # as at end-of-session, or a failing connect leaks a process per retry.
            if socket_path:
                self._sock = self._dial(socket_path)
            else:
                # Admission FIRST, before anything is spawned. Checking the budget
                # here rather than letting the GPU enforce it is the whole fix: over the
                # cap we refuse with a named, retriable `at_capacity` instead of paying a
                # process spawn and a model load only to have the device OOM and report a
                # config-fault `load_failed`. A raise here leaves _holds_slot False, so
                # the except-guard's close() cannot release a slot we never took.
                self._acquire_slot()
                socket_path = self._make_private_socket_path()
                self._proc = self._spawn(bin_path, socket_path)
                # The sidecar loads the model before it accepts, so allow a generous window;
                # with a sidecar per session, several may be loading at once.
                self._sock = self._dial(socket_path, deadline_s=60.0, retry_delay_s=0.1,
                                        proc=self._proc)
            # finalize() runs a GPU decode; keep a generous per-op timeout on blocking reads.
            self._sock.settimeout(120.0)
            self._handshake()
        except BaseException:
            self.close()
            raise

    def _acquire_slot(self) -> None:
        """Take one slot from the shared sidecar budget, or raise `at_capacity`."""
        if self._slots is None:
            return
        self._slots.acquire()          # raises SidecarError("at_capacity") when full
        self._holds_slot = True

    def _release_slot(self) -> None:
        """Give back this session's slot, exactly once. Safe to call repeatedly."""
        if not self._holds_slot:
            return
        self._holds_slot = False
        if self._slots is not None:
            self._slots.release()

    def _make_private_socket_path(self) -> str:
        """Mint a socket path private to THIS session.

        The earlier code spawned every sidecar on one fixed VM_AUDIOCPP_SOCKET, and the
        sidecar unconditionally unlinks and rebinds that path, so a second concurrent
        spawn orphaned the first sidecar -- alive, holding a loaded GGUF on the GPU,
        listening on an inode nothing could reach. A per-session directory makes that
        collision unrepresentable. Removed in close().
        """
        self._owned_dir = tempfile.mkdtemp(prefix="vm-audiocpp-")
        return os.path.join(self._owned_dir, "sidecar.sock")

    def _dial(self, socket_path: str, *, deadline_s: float = 0.0, retry_delay_s: float = 0.0,
              proc: subprocess.Popen | None = None) -> socket.socket:
        """Connect to `socket_path`, retrying until `deadline_s` while a spawn warms up.

        When we spawned the sidecar we also watch it: a sidecar that dies during model
        load (a missing/bad GGUF, an unusable device) never binds, and waiting out the
        full window for a process we already know is gone would stall the session for a
        minute per connection now that every session spawns its own.

        Giving up raises `SidecarError`, not the bare `OSError`.
        Failing to reach a sidecar is the same CONFIGURATION-failure class as a sidecar
        that answers HELLO with ERROR `load_failed`, so the two must reach the WS handler
        as one catchable type and one defined close code. A bare OSError would instead fall
        through to the catch-all `except Exception` and be reported to the client as a
        generic server_error -- an operator's wrong socket path logged as a VoiceManager bug.
        """
        deadline = time.monotonic() + deadline_s
        last_err: Exception | None = None
        while True:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(socket_path)
                return sock
            except OSError as exc:
                sock.close()
                last_err = exc
                if proc is not None and proc.poll() is not None:
                    raise SidecarError(
                        "dial_failed",
                        f"sidecar exited with status {proc.returncode} before binding "
                        f"{socket_path!r} (last error: {last_err})") from exc
                if time.monotonic() >= deadline:
                    raise SidecarError(
                        "dial_failed",
                        f"could not connect to sidecar at {socket_path!r}: {last_err}") from exc
                time.sleep(retry_delay_s)

    def _spawn(self, bin_path: str, socket_path: str) -> subprocess.Popen:
        """Spawn this session's own sidecar on `socket_path` from the VM_AUDIOCPP_* config.

        Args mirror sidecar/src/main.cpp usage (--socket/--model/--backend/--device/
        --threads/--assets-root). `socket_path` is the private path minted for this
        session, so concurrent spawns never contend for it. The model-free suite
        stubs this method (tests/test_audiocpp_ownership.py) or dials an external fake
        with no bin configured; a live spawn needs the real binary + GGUF.

        A spawn that cannot happen at all raises `SidecarError` rather than the bare
        OSError subclass Popen throws. That classification was fixed on the _dial path
        first and this one was left behind, but "the binary is missing or
        not executable" is the same OPERATOR CONFIGURATION class as `load_failed` or
        `dial_failed`, and it is the LIKELIEST of them: VM_AUDIOCPP_SIDECAR_BIN defaults
        to a relative path (`sidecar/build/bin/audiocpp_sidecar`) to a git-ignored build
        artifact, so a fresh clone or a server started from any CWD other than the repo
        root hits it on the first audio frame. A raw FileNotFoundError falls through to
        the WS catch-all and is reported as a generic server_error 1011 with a full bug
        traceback -- an operator's untouched default logged as a VoiceManager bug. The
        message names the RESOLVED absolute path, since a CWD mismatch is the whole
        failure mode and the configured relative string alone would not reveal it.
        """
        argv = [bin_path, "--socket", socket_path]
        model = self._config.get("model", "")
        if model:
            argv += ["--model", model]
        argv += [
            "--backend", str(self._config.get("backend", "cpu")),
            "--device", str(self._config.get("device", 0)),
            "--threads", str(self._config.get("threads", 4)),
        ]
        # Operator option overrides: already-parsed KEY=VALUE items from
        # VM_AUDIOCPP_{SESSION,REQUEST}_OPTIONS, rendered as repeatable flags. Opaque:
        # the sidecar merges them over its registry defaults; nothing is interpreted here.
        for pair in self._config.get("session_options", []) or []:
            argv += ["--session-option", pair]
        for pair in self._config.get("request_options", []) or []:
            argv += ["--request-option", pair]
        assets_root = self._config.get("assets_root", "")
        if assets_root:
            argv += ["--assets-root", assets_root]
        logger.info("Spawning audio.cpp sidecar: %s", " ".join(argv))
        try:
            return subprocess.Popen(argv)
        except OSError as exc:
            raise SidecarError(
                "spawn_failed",
                f"could not spawn the audio.cpp sidecar {os.path.abspath(bin_path)!r} "
                f"(VM_AUDIOCPP_SIDECAR_BIN={bin_path!r}, resolved against the server's "
                f"working directory {os.getcwd()!r}): {exc}") from exc

    def _handshake(self) -> None:
        model_task = self._config.get("model_task", "nemotron-3.5-asr-streaming")
        self._send(HELLO, json.dumps(
            {"protocol": PROTOCOL_VERSION, "model_task": model_task}).encode("utf-8"))
        mtype, payload = self._recv_frame_blocking()
        if mtype == ERROR:
            obj = self._parse_json(payload)
            raise SidecarError(obj.get("code", "internal"), obj.get("message", ""))
        if mtype != READY:
            raise SidecarError("bad_handshake", f"expected READY (0x02), got 0x{mtype:02x}")
        self._ready = self._parse_json(payload)
        if self._ready.get("ok") is not True:
            raise SidecarError("bad_handshake", f"READY.ok is not true: {self._ready}")
        logger.info("audio.cpp sidecar handshake ok: %s", json.dumps(self._ready))

    # --- send ---------------------------------------------------------------------
    def _send(self, mtype: int, payload: bytes = b"") -> None:
        assert self._sock is not None
        try:
            self._sock.sendall(_HEADER.pack(mtype, len(payload)) + payload)
        except OSError as exc:
            raise SidecarFault(f"sidecar send failed: {exc}") from exc

    def send_audio(self, pcm_bytes: bytes) -> None:
        self._send(AUDIO, pcm_bytes)

    def send_flush(self) -> None:
        self._send(FLUSH, b"{}")

    def send_reset(self) -> None:
        self._send(RESET, b"{}")

    # --- receive ------------------------------------------------------------------
    @staticmethod
    def _parse_json(payload: bytes) -> dict:
        try:
            obj = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SidecarError("bad_frame", f"malformed JSON payload: {exc}") from exc
        return obj if isinstance(obj, dict) else {}

    def _next_buffered_frame(self) -> tuple[int, bytes] | None:
        """Pop one complete frame from the read buffer, or None if incomplete."""
        if len(self._rbuf) < _HEADER.size:
            return None
        mtype, length = _HEADER.unpack_from(self._rbuf, 0)
        if length > _MAX_FRAME:
            raise SidecarError("bad_frame", f"frame length {length} exceeds 16 MiB cap")
        end = _HEADER.size + length
        if len(self._rbuf) < end:
            return None
        payload = bytes(self._rbuf[_HEADER.size:end])
        del self._rbuf[:end]
        return mtype, payload

    def _recv_frame_blocking(self) -> tuple[int, bytes]:
        """Block until one full frame is available (respects the socket timeout)."""
        assert self._sock is not None
        while True:
            frame = self._next_buffered_frame()
            if frame is not None:
                return frame
            try:
                chunk = self._sock.recv(_RECV_CHUNK)
            except socket.timeout as exc:
                raise SidecarFault(f"sidecar read timed out: {exc}") from exc
            except OSError as exc:
                raise SidecarFault(f"sidecar read failed: {exc}") from exc
            if not chunk:
                raise SidecarFault("unexpected sidecar EOF (treat as a fault)")
            self._rbuf.extend(chunk)

    def drain_ready_frames(self) -> list[tuple[int, bytes]]:
        """Return every frame already available without blocking (may be empty).

        Used by feed() to forward any interim EVENT (or a volunteered FINAL) the sidecar
        has emitted, without ever waiting. The shipping Nemotron-3.5 model streams its
        deltas from process_audio_chunk during AUDIO, and they are forwarded unchanged;
        an empty result only means no delta is ready yet.
        """
        assert self._sock is not None
        frames: list[tuple[int, bytes]] = []
        while True:
            frame = self._next_buffered_frame()
            if frame is not None:
                frames.append(frame)
                continue
            readable, _, _ = select.select([self._sock], [], [], 0)
            if not readable:
                return frames
            try:
                chunk = self._sock.recv(_RECV_CHUNK)
            except OSError as exc:
                raise SidecarFault(f"sidecar read failed: {exc}") from exc
            if not chunk:
                raise SidecarFault("unexpected sidecar EOF (treat as a fault)")
            self._rbuf.extend(chunk)

    # --- teardown -----------------------------------------------------------------
    def close(self) -> None:
        """Best-effort, non-blocking teardown (safe to call on the event loop).

        Closing the socket gives the sidecar an EOF, which PROTOCOL.md treats exactly
        like a BYE (frees the session), so no blocking BYE write is needed. In OWNED
        mode we then terminate AND REAP the sidecar we spawned and remove the private
        socket path we minted, so neither a live process, a zombie, nor a stale socket
        outlives the session. In EXTERNAL mode there is nothing of ours to tear down:
        the listener belongs to whoever started it and is left exactly as we found it,
        so a session can never kill a sidecar another session is using.

        Two constraints pull against each other here. The child must be
        wait()ed -- terminate-and-forget left a `<defunct>` zombie per connection for
        the life of the server -- and a sidecar that ignores SIGTERM must be SIGKILLed
        so it cannot survive holding GPU VRAM. But close() is reached from dispose(),
        which the WS handler calls WITHOUT await, i.e. on the event loop. So the signal
        is delivered here, synchronously and non-blockingly,
        and the wait -> SIGKILL -> wait escalation is handed to a short-lived daemon
        thread. The loop returns immediately; the reap still genuinely happens.
        """
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._proc is not None:
            proc, self._proc = self._proc, None
            try:
                proc.terminate()
            except Exception:   # already exited, or an unsignalable state
                pass
            # Hand the capacity slot to the reaper rather than releasing it here.
            # close() only SIGNALs; the process is still alive, still holding its CUDA
            # context and its GGUF, until the reaper's wait() returns. Freeing the slot
            # now would let a replacement session spawn against VRAM that is not yet
            # free -- re-creating, inside the SIGTERM grace window, exactly the
            # over-subscription the cap exists to prevent.
            slots = self._slots if self._holds_slot else None
            self._holds_slot = False
            self._reaper = threading.Thread(
                target=self._reap, args=(proc, slots), name="audiocpp-sidecar-reap",
                daemon=True)
            self._reaper.start()
        else:
            # No process to wait for -- external mode, or a spawn that never happened
            # (e.g. mkdtemp failed after admission). Release directly, or the slot leaks.
            self._release_slot()
        if self._owned_dir is not None:
            # Ours alone, so removing it cannot disturb another session. The sidecar
            # unlinks the path itself on a clean exit; this covers every other exit.
            shutil.rmtree(self._owned_dir, ignore_errors=True)
            self._owned_dir = None

    @staticmethod
    def _reap(proc: subprocess.Popen, slots: "SidecarSlots | None" = None) -> None:
        """wait() an already-SIGTERMed sidecar, escalating to SIGKILL.

        Runs in close()'s daemon thread, never on the event loop. Bounded end to end:
        at worst _TERM_GRACE_S + _KILL_GRACE_S, after which we log and give up rather
        than leak a thread on a process stuck in uninterruptible sleep (a wedged CUDA
        ioctl is the realistic case, and nothing in userspace can clear that).

        Also releases this session's capacity slot, which is why the release is
        here and not in close(): the slot represents a live sidecar's GPU footprint, so
        it is only genuinely free once the process is. The release is in a `finally` --
        on the give-up path the VRAM may indeed still be held, but holding the slot
        forever would let one unkillable process permanently shrink the server's
        capacity, which is the worse of the two failures (the log line above is how an
        operator learns the real cause).
        """
        try:
            try:
                proc.wait(timeout=_TERM_GRACE_S)
                return
            except subprocess.TimeoutExpired:
                pass
            logger.warning("audio.cpp sidecar pid %d ignored SIGTERM for %.1fs; sending SIGKILL "
                           "so it cannot survive holding GPU memory", proc.pid, _TERM_GRACE_S)
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=_KILL_GRACE_S)
            except subprocess.TimeoutExpired:
                logger.error("audio.cpp sidecar pid %d survived SIGKILL for %.1fs (likely stuck in "
                             "an uninterruptible driver call); it may still hold GPU memory",
                             proc.pid, _KILL_GRACE_S)
        finally:
            if slots is not None:
                slots.release()


class AudioCppSTTPipeline:
    """Per-connection audio.cpp streaming STT pipeline."""

    def __init__(self, config: dict | None = None):
        """Store config only -- connecting is deferred to the first feed(), off-loop.

        __init__ runs on the event loop inside the WS handler, so it must not block;
        the spawn + socket connect + HELLO/READY handshake happen lazily in feed()'s
        worker thread. `config` is the VM_AUDIOCPP_* dict from _load_audiocpp_models.
        """
        self._client = _SidecarClient(config)
        self._segment_index = 0
        # The utterance transcript-so-far: EVENT deltas accumulate here,
        # cleared on FINAL and reset(). get_current_text() serves it with replace
        # semantics -- always the whole accumulation, never the last fragment alone.
        self._last_partial = ""
        self._last_lang = ""
        self._pending_finals: list[tuple[str, str, int]] = []
        self._reset_pending = False   # a RESET deferred by reset(), flushed off-loop next op
        # True once audio has been fed but not yet FLUSHed. finalize() flushes ONLY when it
        # is set and clears it on flush, so finalize() is re-armable per utterance (a
        # mid-session silence trigger flushes utterance after utterance) yet stays
        # idempotent at teardown: with nothing buffered it returns empty and sends no FLUSH.
        self._pending_flush = False

    # --- feed (interim path) ------------------------------------------------------
    async def feed(self, pcm_bytes: bytes) -> bool:
        """Send one PCM chunk to the sidecar; True if a new interim partial is available.

        Off-loop: the connect (first call), the AUDIO send, and the non-blocking
        reply drain all run in a worker thread so the event loop is never stalled.
        """
        return await asyncio.to_thread(self._feed_sync, pcm_bytes)

    def _feed_sync(self, pcm_bytes: bytes) -> bool:
        self._client.connect()          # lazy, idempotent (spawn + socket + handshake)
        self._flush_pending_reset()
        self._client.send_audio(pcm_bytes)
        self._pending_flush = True      # audio buffered -> a FLUSH now would yield a final
        previous = self._last_partial
        for mtype, payload in self._client.drain_ready_frames():
            self._absorb(mtype, payload)
        return self._last_partial != previous

    def _absorb(self, mtype: int, payload: bytes) -> None:
        """Fold one server frame into pipeline state (EVENT -> partial, FINAL -> queue)."""
        if mtype == EVENT:
            obj = _SidecarClient._parse_json(payload)
            if obj.get("is_final"):
                # A final-flagged EVENT is the model's end-of-utterance MARKER, not an
                # interim: the committed text arrives as the FINAL frame,
                # so this must neither surface as a partial nor clobber the accumulated
                # transcript-so-far.
                return
            partial = obj.get("partial_text") or {}
            # audio.cpp emits partial_text as token/char DELTAS (PROTOCOL.md), so
            # ACCUMULATE into the utterance transcript-so-far; get_current_text() then
            # carries replace semantics -- the whole transcript, never the last fragment
            # alone (the last-fragment 'te' bug).
            self._last_partial += partial.get("text", "")
            self._last_lang = partial.get("language", self._last_lang)
        elif mtype == FINAL:
            obj = _SidecarClient._parse_json(payload)
            text_output = obj.get("text_output") or {}
            text = text_output.get("text", "")
            lang = text_output.get("language", self._last_lang)
            if text.strip():
                # Queue for take_final(); the drive loop pops it then calls reset(),
                # which advances the segment so back-to-back finals stay distinct.
                self._pending_finals.append((text, lang, self._segment_index))
            # The FINAL commits this utterance: clear the accumulator NOW, not at the
            # eventual reset(), so deltas of the next utterance arriving in this same
            # drain can never append to committed text.
            self._last_partial = ""
        elif mtype == ERROR:
            obj = _SidecarClient._parse_json(payload)
            code = obj.get("code", "internal")
            if code in RECOVERABLE_STREAM_ERROR_CODES:
                # The sidecar is still serving (see RECOVERABLE_STREAM_ERROR_CODES); the
                # cost is this utterance, not the session. Log it and keep streaming.
                logger.warning("audio.cpp sidecar reported a recoverable %s mid-stream, "
                               "dropping this utterance: %s", code, obj.get("message", ""))
                return
            raise SidecarError(code, obj.get("message", ""))
        # Any other frame type mid-stream is ignored (forward-compatible reader).

    def get_current_text(self) -> tuple[str, str, int]:
        """Current live partial: (text, language, segment_index)."""
        return self._last_partial, self._last_lang, self._segment_index

    def take_final(self) -> tuple[str, str, int] | None:
        """Pop the next queued utterance-final (text, lang, seg), or None."""
        if self._pending_finals:
            return self._pending_finals.pop(0)
        return None

    # --- finalize (utterance/teardown flush) --------------------------------------
    async def finalize(self) -> tuple[str, str, int]:
        """FLUSH the buffered utterance and return its final (text, language, seg).

        Off-loop. Two callers share this one method:
          * the mid-session silence trigger, flushing each utterance while the session
            stays open -- and since the flush clears the pending-audio flag and feed()
            re-arms it, the NEXT utterance flushes too;
          * the WS teardown, flushing the trailing utterance before the socket closes so
            text is never silently dropped.

        Gated by `_pending_flush`: with no audio buffered since the last flush it returns
        empty and sends NO FLUSH, so a double teardown -- or a teardown right after a
        mid-session finalize with no new audio -- never emits a spurious/duplicate final.
        """
        return await asyncio.to_thread(self._finalize_sync)

    def _finalize_sync(self) -> tuple[str, str, int]:
        if not self._pending_flush or not self._client.is_connected:
            # Never streamed, or nothing buffered since the last flush -- nothing to do.
            return "", self._last_lang, self._segment_index
        self._flush_pending_reset()
        self._client.send_flush()
        final_text, final_lang = "", self._last_lang
        # Exactly one FINAL per FLUSH; a burst of EVENT deltas may precede it (PROTOCOL).
        # The burst is read under a whole-exchange deadline: every EVENT resets
        # the per-recv socket timeout, so without this a sidecar dripping EVENTs and
        # never a FINAL would hang the session forever instead of failing it. Checked
        # after each frame, so the worst case is deadline + one 120 s recv -- bounded.
        deadline = time.monotonic() + _FLUSH_FINAL_DEADLINE_S
        events_seen = 0
        while True:
            if time.monotonic() > deadline:
                raise SidecarFault(
                    f"sidecar sent {events_seen} EVENTs but no FINAL within "
                    f"{_FLUSH_FINAL_DEADLINE_S:.0f}s of FLUSH (PROTOCOL.md guarantees "
                    "exactly one FINAL per FLUSH)")
            mtype, payload = self._client._recv_frame_blocking()
            if mtype == EVENT:
                events_seen += 1
                obj = _SidecarClient._parse_json(payload)
                if not obj.get("is_final"):
                    # Same delta accumulation + marker suppression as _absorb: burst deltas
                    # grow the transcript-so-far; a final-flagged
                    # marker event changes nothing. Unlike the volunteered-FINAL path,
                    # the FINAL below does NOT clear the accumulation -- the WS layer
                    # reads it after finalize() returns to emit the pre-final partial,
                    # then clears it via the reset() at the commit boundary.
                    partial = obj.get("partial_text") or {}
                    self._last_partial += partial.get("text", "")
                    self._last_lang = partial.get("language", self._last_lang)
            elif mtype == FINAL:
                obj = _SidecarClient._parse_json(payload)
                text_output = obj.get("text_output") or {}
                final_text = text_output.get("text", "")
                final_lang = text_output.get("language", self._last_lang)
                self._last_lang = final_lang  # session language is now known
                break
            elif mtype == ERROR:
                obj = _SidecarClient._parse_json(payload)
                code = obj.get("code", "internal")
                if code not in RECOVERABLE_STREAM_ERROR_CODES:
                    raise SidecarError(code, obj.get("message", ""))
                # The decode failed but the sidecar kept its loop: this ERROR
                # arrives INSTEAD of the FINAL, so no final is coming -- stop waiting and
                # report an empty utterance rather than tearing the session down. The
                # session stays usable and the next feed()/finalize() pair works.
                #
                # Drop the utterance WHOLLY: the EVENT burst above has already
                # folded its deltas into the live partial, and the caller reads that via
                # get_current_text() to emit the is_final=False frame that normally
                # precedes the final. With no final coming, leaving it set would surface a
                # partial nothing ever commits -- an orphan the client waits on forever.
                # Clearing it here (not in the caller) keeps "dropped" a property of the
                # pipeline's own state, so every caller of finalize() gets it.
                self._last_partial = ""
                logger.warning("audio.cpp sidecar reported a recoverable %s on FLUSH, "
                               "dropping this utterance: %s", code, obj.get("message", ""))
                break
        self._pending_flush = False  # this utterance's audio is flushed; re-armed by feed()
        return final_text, final_lang, self._segment_index

    # --- reset / dispose ----------------------------------------------------------
    def reset(self) -> None:
        """Start a fresh utterance: clear the live partial and advance the segment.

        Sync (the WS handler calls it without await), so it must not do a blocking
        socket round-trip on the event loop: the sidecar RESET frame is DEFERRED and
        sent off-loop by the next feed()/finalize() (_flush_pending_reset). Advancing
        segment_index here mirrors qwen (monotonic, never rewound).
        """
        self._segment_index += 1
        self._last_partial = ""
        self._reset_pending = True
        logger.debug("AudioCppSTTPipeline reset -> segment %d (RESET deferred)", self._segment_index)

    def _flush_pending_reset(self) -> None:
        """Send a RESET frame deferred by reset(), off-loop before the next op."""
        if self._reset_pending:
            self._client.send_reset()
            self._reset_pending = False

    def dispose(self) -> None:
        """Release the connection on disconnect.

        Sync and non-blocking: close the socket (EOF == BYE, frees the sidecar session)
        and terminate a sidecar this pipeline spawned. No blocking round-trip on the loop.
        """
        self._client.close()
        self._pending_finals.clear()
        self._last_partial = ""
        logger.debug("AudioCppSTTPipeline disposed")
