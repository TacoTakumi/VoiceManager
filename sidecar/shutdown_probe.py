#!/usr/bin/env python3
"""Shutdown-determinism harness for the audio.cpp sidecar.

This is a MANUAL script - it needs the compiled `audiocpp_sidecar` binary. It is NOT
part of the model-free pytest suite: it lives outside tests/ and is not test_*-prefixed,
so `pytest tests/` never collects it. Unlike crash_probe.py it needs NO model and NO
GPU: the model only loads on HELLO, and this probe never sends one - the session thread
just sits blocked in read_frame(), which is exactly the state under test.

It proves deterministic shutdown: a SIGTERM produces a prompt, graceful exit no matter
which thread the kernel delivers it to. The kernel may hand a process-directed signal
to ANY thread, and formerly a delivery to a session thread was swallowed (read_exact
retries EINTR) while main stayed blocked in accept() - the sidecar then never exited,
and every VoiceManager teardown ate the full 5 s SIGTERM grace plus a SIGKILL and a
reaper warning. Delivery choice cannot be provoked by plain kill(2), so each case is
forced with tgkill(2) to a specific thread id:

  1. process-directed SIGTERM (the Python reaper's normal terminate())
  2. tgkill to the MAIN thread (the delivery the old code happened to rely on)
  3. tgkill to the SESSION thread (the defect: builds before the fix hang forever here)

Pass criteria per case: with a client connected and a session thread live, the sidecar
exits within 2 s of the signal, with exit code 0, logging 'shut down cleanly'.

Usage:
  python sidecar/shutdown_probe.py [--spawn sidecar/build/bin/audiocpp_sidecar]
"""
from __future__ import annotations

import argparse
import ctypes
import os
import socket
import subprocess
import sys
import tempfile
import time

SYS_tgkill = 234  # x86_64
SIGTERM = 15

libc = ctypes.CDLL("libc.so.6", use_errno=True)


def tgkill(tgid: int, tid: int, sig: int) -> None:
    if libc.syscall(SYS_tgkill, tgid, tid, sig) != 0:
        raise OSError(ctypes.get_errno(), "tgkill failed")


def tids(pid: int) -> list[int]:
    return sorted(int(t) for t in os.listdir(f"/proc/{pid}/task"))


def spawn(binary: str, sock_path: str) -> tuple[subprocess.Popen, socket.socket]:
    proc = subprocess.Popen(
        [binary, "--socket", sock_path, "--model", "/nonexistent.gguf"],
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 5.0
    while not os.path.exists(sock_path):
        if proc.poll() is not None or time.monotonic() > deadline:
            proc.kill()
            raise RuntimeError("sidecar never bound its socket")
        time.sleep(0.02)
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(sock_path)
    time.sleep(0.3)  # let the session thread spawn and block in read_frame()
    return proc, client


def run_case(binary: str, name: str, deliver) -> bool:
    # tempfile.mkdtemp in /tmp keeps the socket path well under sun_path's 108 bytes.
    workdir = tempfile.mkdtemp(prefix="t25-probe-")
    sock_path = os.path.join(workdir, "probe.sock")
    proc, client = spawn(binary, sock_path)
    try:
        deliver(proc)
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            print(f"FAIL {name}: sidecar still alive 2 s after SIGTERM (hang)")
            proc.kill()
            proc.wait()
            return False
        stderr = proc.stderr.read().decode()
        if proc.returncode != 0:
            print(f"FAIL {name}: exit code {proc.returncode}, stderr:\n{stderr}")
            return False
        if "shut down cleanly" not in stderr:
            print(f"FAIL {name}: no 'shut down cleanly' in stderr:\n{stderr}")
            return False
        print(f"PASS {name}: prompt graceful exit")
        return True
    finally:
        client.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        os.rmdir(workdir)


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spawn",
        default=os.path.join(repo_root, "sidecar/build/bin/audiocpp_sidecar"),
        help="path to the compiled sidecar binary",
    )
    args = parser.parse_args()
    if not os.path.exists(args.spawn):
        print(f"error: sidecar binary not found at {args.spawn}")
        return 2

    ok = True
    ok &= run_case(args.spawn, "process-directed SIGTERM", lambda p: p.terminate())
    ok &= run_case(
        args.spawn, "SIGTERM to main thread",
        lambda p: tgkill(p.pid, tids(p.pid)[0], SIGTERM),
    )

    # The session-thread case: deliver to every non-main thread present. With one
    # client connected there is exactly one session thread, blocked in read_frame().
    def to_session_threads(p: subprocess.Popen) -> None:
        others = [t for t in tids(p.pid) if t != p.pid]
        if not others:
            raise RuntimeError("no session thread found - probe setup broken")
        for t in others:
            tgkill(p.pid, t, SIGTERM)

    ok &= run_case(args.spawn, "SIGTERM to session thread", to_session_threads)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
