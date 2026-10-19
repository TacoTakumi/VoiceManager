#!/usr/bin/env python3
"""Socket test client for the audio.cpp sidecar v1 handshake.

This is a MANUAL script - it needs the compiled `audiocpp_sidecar` binary. It is NOT
part of the model-free pytest suite: it lives outside tests/ and is not test_*-prefixed,
so `pytest tests/` never collects it. It needs no GPU and loads no model (the handshake
skeleton does not touch audio.cpp), so it is cheap to run by hand.

It exercises the wire protocol in PROTOCOL.md: connects over AF_UNIX, sends a HELLO
carrying the model/task id, and asserts the sidecar's reply -

  1. a registered model_task            -> READY (ack) with the audio contract
  2. an unregistered model_task         -> ERROR unknown_model_task
  3. a mismatched protocol version      -> ERROR unsupported_protocol

Usage:
  # spawn the sidecar on a temp socket, run all checks, tear it down (one command):
  python sidecar/handshake_probe.py --spawn build/bin/audiocpp_sidecar

  # or point at an already-running sidecar:
  python sidecar/handshake_probe.py --socket /path/to/sidecar.sock
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time

# Message type tags - must match sidecar/src/protocol.h.
HELLO = 0x01
READY = 0x02
FLUSH = 0x05
ERROR = 0x08
BYE = 0x09

PROTOCOL_VERSION = 1
DEFAULT_MODEL_TASK = "nemotron-3.5-asr-streaming"
HEADER = struct.Struct(">BI")  # type:u8, length:u32 big-endian


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


def connect(socket_path: str, timeout_s: float = 5.0) -> socket.socket:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            sock.connect(socket_path)
            return sock
        except OSError as exc:  # socket not up yet
            last_err = exc
            time.sleep(0.05)
    raise TimeoutError(f"could not connect to {socket_path}: {last_err}")


class CheckError(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def check_ready(socket_path: str, model_task: str) -> None:
    sock = connect(socket_path)
    try:
        send_json(sock, HELLO, {"protocol": PROTOCOL_VERSION, "model_task": model_task})
        mtype, payload = recv_frame(sock)
        expect(mtype == READY, f"expected READY (0x02), got 0x{mtype:02x}: {payload!r}")
        ack = json.loads(payload)
        expect(ack.get("ok") is True, f"READY.ok is not true: {ack}")
        expect(ack.get("protocol") == PROTOCOL_VERSION, f"READY.protocol mismatch: {ack}")
        expect(ack.get("model_task") == model_task, f"READY.model_task mismatch: {ack}")
        expect(ack.get("task") == "asr", f"READY.task mismatch: {ack}")
        expect(ack.get("mode") == "streaming", f"READY.mode mismatch: {ack}")
        expect(ack.get("input") == "audio_chunks", f"READY.input mismatch: {ack}")
        audio = ack.get("audio", {})
        expect(audio.get("sample_rate") == 16000, f"READY.audio.sample_rate mismatch: {ack}")
        expect(audio.get("channels") == 1, f"READY.audio.channels mismatch: {ack}")
        expect(audio.get("format") == "s16le", f"READY.audio.format mismatch: {ack}")
        send_json(sock, BYE, {})
        print(f"  PASS  registered model_task '{model_task}' -> READY ack {json.dumps(ack)}")
    finally:
        sock.close()


def check_error(socket_path: str, hello: dict, expected_code: str, label: str) -> None:
    sock = connect(socket_path)
    try:
        send_json(sock, HELLO, hello)
        mtype, payload = recv_frame(sock)
        expect(mtype == ERROR, f"expected ERROR (0x08), got 0x{mtype:02x}: {payload!r}")
        err = json.loads(payload)
        expect(err.get("ok") is False, f"ERROR.ok is not false: {err}")
        expect(err.get("code") == expected_code, f"expected code '{expected_code}', got: {err}")
        print(f"  PASS  {label} -> ERROR {json.dumps(err)}")
    finally:
        sock.close()


def run_checks(socket_path: str, model_task: str) -> None:
    print(f"handshake probe against {socket_path}")
    check_ready(socket_path, model_task)
    check_error(
        socket_path,
        {"protocol": PROTOCOL_VERSION, "model_task": "definitely-not-registered"},
        "unknown_model_task",
        "unregistered model_task",
    )
    check_error(
        socket_path,
        {"protocol": 999, "model_task": model_task},
        "unsupported_protocol",
        "protocol 999",
    )
    print("ALL CHECKS PASSED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--socket", help="connect to an already-running sidecar at this path")
    group.add_argument("--spawn", metavar="BIN", help="spawn this sidecar binary on a temp socket")
    parser.add_argument("--model-task", default=DEFAULT_MODEL_TASK, help="registered wire id to test")
    args = parser.parse_args()

    if args.socket:
        run_checks(args.socket, args.model_task)
        return 0

    binary = os.path.abspath(args.spawn)
    if not os.path.exists(binary):
        print(f"error: sidecar binary not found: {binary}", file=sys.stderr)
        return 2
    tmpdir = tempfile.mkdtemp(prefix="audiocpp-sidecar-")
    socket_path = os.path.join(tmpdir, "sidecar.sock")
    proc = subprocess.Popen([binary, "--socket", socket_path])
    try:
        run_checks(socket_path, args.model_task)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        try:
            os.path.exists(socket_path) and os.unlink(socket_path)
        except OSError:
            pass
        os.rmdir(tmpdir)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (CheckError, EOFError, TimeoutError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
