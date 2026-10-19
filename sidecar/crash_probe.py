#!/usr/bin/env python3
"""Crash-containment harness for the audio.cpp sidecar.

This is a MANUAL script - it needs the compiled `audiocpp_sidecar` binary, the real
Nemotron-3.5 streaming GGUF, and a CUDA GPU. It is NOT part of the model-free pytest
suite: it lives outside tests/ and is not test_*-prefixed, so `pytest tests/` never
collects it.

It proves fault containment: a process-fatal fault in the sidecar is contained to the
sidecar process, surfaces to the connected client as a defined signal (a best-effort
ERROR `sidecar_fault` frame and/or a socket EOF) within a time bound - never a hang -
and a freshly spawned sidecar on the same socket path accepts a new session.

For each induced fault it:

  1. spawns a sidecar (real model), connects, HELLO -> READY (session warm)
  2. streams a few AUDIO frames (mid-stream, no FLUSH)
  3. induces a process-fatal fault mid-stream by signalling the sidecar:
       - SIGABRT: the faithful "induced GGML_ASSERT" (ggml's GGML_ASSERT calls abort()).
                  The sidecar's fatal-signal handler makes a best-effort ERROR
                  `sidecar_fault` write, then re-raises -> the process dies from SIGABRT.
       - SIGKILL: the faithful "kill". Uncatchable; the client sees a bare EOF.
  4. asserts the client observes a disconnect (ERROR and/or EOF) within a tight bound
     (no socket.timeout == no hang), and that the sidecar process actually exited with
     that signal (containment: only the sidecar died)
  5. respawns a fresh sidecar on the SAME socket path (exercising the stale-socket
     cleanup at bind), HELLO -> READY, streams a clip + FLUSH -> asserts a FINAL:
     a fresh sidecar accepts a new full session

Usage:
  python sidecar/crash_probe.py \
      --spawn sidecar/build/bin/audiocpp_sidecar \
      --model /home/user/audiocpp-models/Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-f16.gguf \
      --backend cuda --device 3
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time

# Message type tags - must match sidecar/src/protocol.h.
HELLO = 0x01
READY = 0x02
AUDIO = 0x03
EVENT = 0x04
FLUSH = 0x05
FINAL = 0x06
RESET = 0x07
ERROR = 0x08
BYE = 0x09

PROTOCOL_VERSION = 1
DEFAULT_MODEL_TASK = "nemotron-3.5-asr-streaming"
HEADER = struct.Struct(">BI")  # type:u8, length:u32 big-endian

# Standard audio contract: 16 kHz mono 16-bit PCM.
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 5120  # ~0.32 s chunks
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLIP = os.path.join(REPO_ROOT, "voices", "designed", "designed_female_s101.wav")

# The client must see a disconnect this fast after the fault, or it is a hang.
DISCONNECT_BOUND_S = 5.0
# The fatal-fault ERROR code the sidecar emits best-effort on a catchable fatal signal.
FAULT_CODE = "sidecar_fault"


def send_frame(sock: socket.socket, mtype: int, payload: bytes = b"") -> None:
    sock.sendall(HEADER.pack(mtype, len(payload)) + payload)


def send_json(sock: socket.socket, mtype: int, obj: dict) -> None:
    send_frame(sock, mtype, json.dumps(obj).encode("utf-8"))


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise EOFError(f"connection closed after {len(buf)}/{n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock: socket.socket) -> tuple[int, bytes]:
    mtype, length = HEADER.unpack(_recv_exactly(sock, HEADER.size))
    payload = _recv_exactly(sock, length) if length else b""
    return mtype, payload


def connect(socket_path: str, timeout_s: float = 30.0) -> socket.socket:
    # The sidecar loads the model before it accepts, so allow a generous connect window.
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(120.0)  # finalize() runs a GPU decode
            sock.connect(socket_path)
            return sock
        except OSError as exc:
            last_err = exc
            time.sleep(0.1)
    raise TimeoutError(f"could not connect to {socket_path}: {last_err}")


class CheckError(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def load_pcm_s16le(clip_path: str) -> bytes:
    """Decode any audio file to raw 16 kHz mono s16le PCM via ffmpeg."""
    if not os.path.exists(clip_path):
        raise CheckError(f"reference clip not found: {clip_path}")
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", clip_path,
         "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise CheckError(f"ffmpeg failed on {clip_path}: {proc.stderr.decode(errors='replace')}")
    if not proc.stdout:
        raise CheckError(f"ffmpeg produced no PCM for {clip_path}")
    return proc.stdout


def handshake(sock: socket.socket, model_task: str) -> dict:
    send_json(sock, HELLO, {"protocol": PROTOCOL_VERSION, "model_task": model_task})
    mtype, payload = recv_frame(sock)
    expect(mtype == READY, f"expected READY (0x02), got 0x{mtype:02x}: {payload!r}")
    ack = json.loads(payload)
    expect(ack.get("ok") is True, f"READY.ok is not true: {ack}")
    return ack


def spawn_sidecar(binary: str, socket_path: str, args: argparse.Namespace) -> subprocess.Popen:
    return subprocess.Popen([
        binary, "--socket", socket_path, "--model", args.model,
        "--backend", args.backend, "--device", str(args.device), "--threads", str(args.threads),
    ])


def stream_some_audio(sock: socket.socket, pcm: bytes, n_frames: int) -> int:
    """Stream up to n_frames AUDIO chunks (mid-stream, no FLUSH). Returns frames sent."""
    sent = 0
    for offset in range(0, len(pcm), CHUNK_SAMPLES * 2):
        if sent >= n_frames:
            break
        send_frame(sock, AUDIO, pcm[offset:offset + CHUNK_SAMPLES * 2])
        sent += 1
    return sent


def stream_and_finalize(sock: socket.socket, pcm: bytes, label: str) -> str:
    """Stream `pcm` as AUDIO frames, FLUSH, and assert exactly one FINAL with text."""
    for offset in range(0, len(pcm), CHUNK_SAMPLES * 2):
        send_frame(sock, AUDIO, pcm[offset:offset + CHUNK_SAMPLES * 2])
    send_json(sock, FLUSH, {})
    finals: list[dict] = []
    while True:
        mtype, payload = recv_frame(sock)
        if mtype == EVENT:
            continue
        if mtype == FINAL:
            finals.append(json.loads(payload))
            break
        if mtype == ERROR:
            raise CheckError(f"{label}: sidecar returned ERROR: {payload.decode(errors='replace')}")
        raise CheckError(f"{label}: unexpected frame 0x{mtype:02x}: {payload!r}")
    expect(len(finals) == 1, f"{label}: expected exactly one FINAL, got {len(finals)}")
    text = (finals[0].get("text_output") or {}).get("text", "")
    expect(bool(text.strip()), f"{label}: FINAL text_output is empty: {finals[0]}")
    return text


def drain_until_disconnect(sock: socket.socket, bound_s: float) -> tuple[list[dict], float]:
    """Read frames until the peer disconnects (EOF/RST). Returns (error_frames, elapsed_s).

    Raises CheckError if no disconnect arrives within bound_s - that would be the hang the
    acceptance forbids.
    """
    sock.settimeout(bound_s)
    t0 = time.monotonic()
    errors: list[dict] = []
    while True:
        try:
            mtype, payload = recv_frame(sock)
        except EOFError:
            return errors, time.monotonic() - t0            # clean EOF at/within a frame
        except ConnectionResetError:
            return errors, time.monotonic() - t0            # RST is also a defined signal
        except socket.timeout:
            raise CheckError(
                f"no disconnect within {bound_s:.1f}s after the fault - client is hung")
        if mtype == ERROR:
            errors.append(json.loads(payload))
        # A late EVENT or any other in-flight frame is tolerated: we only require that a
        # defined disconnect follows within the bound.


def induce_and_check(binary: str, socket_path: str, signum: int, args: argparse.Namespace) -> None:
    signame = signal.Signals(signum).name
    print(f"[{signame}] fault-containment check")
    pcm = load_pcm_s16le(DEFAULT_CLIP)

    proc = spawn_sidecar(binary, socket_path, args)
    try:
        sock = connect(socket_path)
        try:
            handshake(sock, args.model_task)
            n = stream_some_audio(sock, pcm, n_frames=3)
            # Let the sidecar drain its receive buffer and settle back into a blocking read
            # so the fault lands mid-stream at a defined point.
            time.sleep(0.3)

            print(f"  streamed {n} AUDIO frames, sending {signame} to pid {proc.pid}")
            proc.send_signal(signum)

            errors, elapsed = drain_until_disconnect(sock, DISCONNECT_BOUND_S)
            print(f"  client saw disconnect in {elapsed * 1000:.0f} ms "
                  f"({len(errors)} ERROR frame(s): {[e.get('code') for e in errors]})")
            expect(elapsed < DISCONNECT_BOUND_S,
                   f"disconnect took {elapsed:.2f}s, over the {DISCONNECT_BOUND_S:.1f}s bound")
        finally:
            sock.close()

        # The sidecar process must actually have exited, from the signal we sent - proving
        # the fault stayed contained to this one process (nothing else was taken down).
        rc = proc.wait(timeout=DISCONNECT_BOUND_S)
        expect(rc == -signum,
               f"sidecar exited with {rc}, expected -{signum} ({signame}); fault not contained as expected")
        print(f"  sidecar process exited with {rc} (killed by {signame}) - contained")

        # A catchable fatal signal (SIGABRT ~= GGML_ASSERT) must surface a defined ERROR
        # `sidecar_fault` before the process dies. SIGKILL is uncatchable, so a bare EOF is
        # the only guarantee there.
        if signum == signal.SIGABRT:
            codes = [e.get("code") for e in errors]
            expect(FAULT_CODE in codes,
                   f"expected a best-effort ERROR '{FAULT_CODE}' before EOF, saw codes {codes}")
            print(f"  fault surfaced as ERROR '{FAULT_CODE}' (not just a bare EOF)")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=DISCONNECT_BOUND_S)

    # Respawn a fresh sidecar on the SAME socket path: the crashed process left its socket
    # file behind, so binding here exercises the stale-socket unlink. A fresh full session
    # must succeed.
    print(f"  respawning a fresh sidecar on the same socket path")
    proc2 = spawn_sidecar(binary, socket_path, args)
    try:
        sock2 = connect(socket_path)
        try:
            handshake(sock2, args.model_task)
            text = stream_and_finalize(sock2, pcm, "respawn")
            print(f"  fresh sidecar accepted a new session -> FINAL {text!r}")
            send_json(sock2, BYE, {})
        finally:
            sock2.close()
    finally:
        proc2.terminate()
        try:
            proc2.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc2.kill()
    print(f"[{signame}] PASSED\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spawn", metavar="BIN", required=True,
                        help="sidecar binary to spawn")
    parser.add_argument("--model", required=True, help="model GGUF path")
    parser.add_argument("--backend", default="cuda", help="inference backend (default cuda)")
    parser.add_argument("--device", type=int, default=0, help="backend device index (default 0)")
    parser.add_argument("--threads", type=int, default=4, help="inference threads (default 4)")
    parser.add_argument("--model-task", default=DEFAULT_MODEL_TASK, help="registered wire id")
    parser.add_argument("--signal", choices=["abort", "kill", "both"], default="both",
                        help="which fault(s) to induce: SIGABRT (abort), SIGKILL (kill), or both")
    args = parser.parse_args()

    binary = os.path.abspath(args.spawn)
    if not os.path.exists(binary):
        print(f"error: sidecar binary not found: {binary}", file=sys.stderr)
        return 2

    modes = {"abort": [signal.SIGABRT], "kill": [signal.SIGKILL],
             "both": [signal.SIGABRT, signal.SIGKILL]}[args.signal]

    tmpdir = tempfile.mkdtemp(prefix="audiocpp-crash-")
    socket_path = os.path.join(tmpdir, "sidecar.sock")
    try:
        for signum in modes:
            induce_and_check(binary, socket_path, signum, args)
        print("ALL CHECKS PASSED")
        return 0
    finally:
        try:
            os.path.exists(socket_path) and os.unlink(socket_path)
        except OSError:
            pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CheckError, EOFError, TimeoutError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
