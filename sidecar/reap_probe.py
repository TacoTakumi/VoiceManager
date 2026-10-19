#!/usr/bin/env python3
"""Worker-reaping harness for the audio.cpp sidecar.

This is a MANUAL script - it needs the compiled `audiocpp_sidecar` binary. It is NOT
part of the model-free pytest suite: it lives outside tests/ and is not test_*-prefixed,
so `pytest tests/` never collects it. Like shutdown_probe.py it needs NO model and NO
GPU - the model only loads on HELLO, and this probe never sends one, so every connection
is served by a session thread that sits in read_frame() until the peer disconnects.

The defect: the deterministic-shutdown fix made session threads joinable and joined
them before main returns - correct for quiesce - but every accepted connection's thread
stayed in the `workers` vector until then, including short-lived busy-refusal threads.
Bounded in VoiceManager's OWNED mode (one session per process life), but a long-lived
EXTERNAL sidecar under sequential reconnects, or a client retrying into busy refusals,
accumulated finished-but-unjoined handles without bound.

WHAT TO MEASURE. `/proc/<pid>/task` is NOT the discriminating metric and never was: a
finished pthread's KERNEL task exits at pthread_exit whether or not anyone joins it, so
the task count reads 1 (plus the live session) on a leaking build too - measured on the
pre-fix binary. What glibc retains until join is the thread's STACK, so the leak shows up
in address space: ~9.3 MiB of VmSize per connection on the pre-fix build, plus ~145 KB of
RSS each for the pages actually touched. This probe reports the task count for
completeness but decides on VmSize.

Nor is "VmSize per connection" the right shape, because a REAPING build still shows a
one-time step: joining returns the stack to glibc's stack cache, which stays mapped (a
fixed ~80 MiB here, measured identical at N = 30, 60, 120 and 240). Dividing that fixed
step by N just makes the per-connection figure shrink with N and proves nothing.

So each leg runs TWO equal batches against the SAME process and compares them. If a build
retains anything per connection, batch 2 costs what batch 1 cost. If growth is bounded,
batch 1 pays the one-time cache and batch 2 is ~free. That is exactly the defect's claim
("accumulates ... without bound") turned into a measurement, and it is N-independent.

Both accumulation paths are exercised against ONE sidecar process:
  1. sequential connect/disconnect cycles (the EXTERNAL-sidecar reconnect case)
  2. busy-refusals while one session is held live (the retrying-client case)

Pass criteria per leg: the SECOND batch grows VmSize by less than BUDGET_KIB_PER_BATCH -
under one 8 MiB thread stack, so not even one connection's worth is being retained.

Usage:
  python sidecar/reap_probe.py [--spawn sidecar/build/bin/audiocpp_sidecar] [-n 60]
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time

# One retained thread stack measures ~9.3 MiB (9500 KiB) on this box, so a second batch
# that retains even ONE connection's worth breaks this budget, while a build whose growth
# has already plateaued sits near zero. Not scaled by N - the whole point is that a
# reaping build's second batch costs nothing regardless of how many connections it holds.
BUDGET_KIB_PER_BATCH = 8192


def vm_size_kib(pid: int) -> int:
    with open(f"/proc/{pid}/status") as fh:
        for line in fh:
            if line.startswith("VmSize:"):
                return int(line.split()[1])
    raise RuntimeError("VmSize missing from /proc status")


def task_count(pid: int) -> int:
    return len(os.listdir(f"/proc/{pid}/task"))


def spawn(binary: str, sock_path: str) -> subprocess.Popen:
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
    time.sleep(0.2)  # let startup settle so the baseline is stable
    return proc


def connect_once(sock_path: str, hold: float = 0.01) -> None:
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.connect(sock_path)
    time.sleep(hold)
    conn.close()


def main() -> int:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spawn",
        default=os.path.join(repo_root, "sidecar/build/bin/audiocpp_sidecar"),
        help="path to the compiled sidecar binary",
    )
    parser.add_argument("-n", "--connections", type=int, default=60,
                        help="connections per BATCH; each leg runs two batches (default 60)")
    args = parser.parse_args()
    if not os.path.exists(args.spawn):
        print(f"error: sidecar binary not found at {args.spawn}")
        return 2

    n = args.connections
    workdir = tempfile.mkdtemp(prefix="t32-probe-")
    sock_path = os.path.join(workdir, "probe.sock")
    proc = spawn(args.spawn, sock_path)
    live: socket.socket | None = None
    ok = True
    try:
        def batch(label: str, count: int, hold: float, settle: float) -> int:
            """Run `count` connections, then report and return the VmSize growth in KiB."""
            before = vm_size_kib(proc.pid)
            for _ in range(count):
                connect_once(sock_path, hold=hold)
                time.sleep(settle)
            time.sleep(0.3)
            after, tasks = vm_size_kib(proc.pid), task_count(proc.pid)
            print(f"  {label}: VmSize {before} -> {after} KiB "
                  f"(+{after - before} KiB over {count} connections), tasks {tasks}")
            return after - before

        def verdict(leg: str, first: int, second: int) -> bool:
            """A reaping build pays the one-time cache in batch 1 and ~nothing in batch 2."""
            if second > BUDGET_KIB_PER_BATCH:
                print(f"FAIL {leg}: batch 2 still grew {second} KiB after batch 1's "
                      f"{first} KiB (budget {BUDGET_KIB_PER_BATCH}) - growth scales with "
                      "connection count, so finished workers are not being reaped")
                return False
            print(f"PASS {leg}: batch 2 grew {second} KiB (budget "
                  f"{BUDGET_KIB_PER_BATCH}) - growth plateaued, nothing retained per "
                  "connection")
            return True

        print(f"baseline: VmSize {vm_size_kib(proc.pid)} KiB, "
              f"tasks {task_count(proc.pid)}")

        # --- leg 1: sequential reconnects (the EXTERNAL-sidecar case) -----------------
        print(f"leg 1 - sequential reconnects, 2 batches of {n}:")
        seq1 = batch("batch 1", n, hold=0.01, settle=0.01)
        seq2 = batch("batch 2", n, hold=0.01, settle=0.01)
        ok &= verdict("sequential reconnects", seq1, seq2)

        # --- leg 2: busy refusals while one session is held live ----------------------
        live = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        live.connect(sock_path)
        time.sleep(0.2)  # the live session owns g_session_lock; the rest get refused
        print(f"leg 2 - busy refusals against a held live session, 2 batches of {n}:")
        busy1 = batch("batch 1", n, hold=0.005, settle=0.0)
        busy2 = batch("batch 2", n, hold=0.005, settle=0.0)
        ok &= verdict("busy refusals", busy1, busy2)

        # The live session must have survived every refusal untouched.
        if task_count(proc.pid) < 2:
            print("FAIL: the held live session thread is gone - refusals disturbed it")
            ok = False
    finally:
        if live is not None:
            live.close()
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            print("FAIL: sidecar did not exit within 5 s of SIGTERM")
            proc.kill()
            proc.wait()
            ok = False
        stderr = proc.stderr.read().decode()
        if proc.returncode == 0 and "shut down cleanly" in stderr:
            print("PASS shutdown: still exits cleanly with the sweep in place")
        else:
            print(f"FAIL shutdown: exit {proc.returncode}, stderr:\n{stderr}")
            ok = False
        if os.path.exists(sock_path):
            os.unlink(sock_path)
        os.rmdir(workdir)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
