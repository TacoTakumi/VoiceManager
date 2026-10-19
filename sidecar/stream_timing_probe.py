#!/usr/bin/env python3
"""Does the Nemotron-3.5 sidecar emit interim partials DURING streaming, or only at
finalize? MANUAL - needs the real sidecar + GPU.

Context: the audio.cpp `/ws/stream` integration was silent with the real model. This
probe isolates *why* at the sidecar layer: it spawns the real sidecar, streams one
clip's AUDIO frames one at a time, and after each frame drains the socket (non-blocking,
short wait) to record whether any interim EVENT delta arrives BEFORE any FLUSH. Then it
FLUSHes and records the EVENT + FINAL frames that arrive at finalize.

Observed on cuda:3 (RTX 4060 Ti, f16): 0 EVENTs during process_audio_chunk; all 18
deltas + the FINAL arrive only AFTER FLUSH (first delta ~113 ms post-FLUSH). So the
sidecar path - which is what /ws/stream drives - emits interims only at finalize, and
something on VoiceManager's side must FLUSH per utterance. That observation is
from the audio.cpp pin of the time. On the current pin, upstream Nemotron encodes each
chunk in process_audio_chunk and emits deltas during AUDIO, so this probe should now
record EVENTs before FLUSH; the FINAL still arrives only after FLUSH.

This is a MANUAL script (outside tests/, not test_*-prefixed) - `pytest tests/` never
collects it. It mirrors the sidecar probe family (handshake_probe.py, stream_probe.py,
crash_probe.py).

Usage:
  python sidecar/stream_timing_probe.py \
      --spawn sidecar/build/bin/audiocpp_sidecar \
      --model /home/user/audiocpp-models/Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-f16.gguf \
      --backend cuda --device 3
"""
from __future__ import annotations

import argparse
import json
import os
import select
import socket
import struct
import subprocess
import sys
import tempfile
import time

# Message type tags - must match sidecar/src/protocol.h.
HELLO, READY, AUDIO, EVENT, FLUSH, FINAL, RESET, ERROR, BYE = (
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09)
HEADER = struct.Struct(">BI")  # type:u8, length:u32 big-endian
PROTOCOL_VERSION = 1
DEFAULT_MODEL_TASK = "nemotron-3.5-asr-streaming"

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 5120  # ~0.32 s, matches the pipeline / stream_probe chunking
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLIP = os.path.join(REPO_ROOT, "voices", "designed", "designed_female_s101.wav")


def send_frame(sock: socket.socket, mtype: int, payload: bytes = b"") -> None:
    sock.sendall(HEADER.pack(mtype, len(payload)) + payload)


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
    return mtype, (_recv_exactly(sock, length) if length else b"")


def drain_nonblocking(sock: socket.socket, wait_s: float = 0.05) -> list[tuple[int, bytes]]:
    """Return every complete frame available within wait_s (does not block beyond it)."""
    frames: list[tuple[int, bytes]] = []
    deadline = time.monotonic() + wait_s
    buf = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        readable, _, _ = select.select([sock], [], [], remaining)
        if not readable:
            break
        chunk = sock.recv(65536)
        if not chunk:
            raise EOFError("sidecar EOF mid-drain")
        buf.extend(chunk)
        while len(buf) >= HEADER.size:
            mtype, length = HEADER.unpack_from(buf, 0)
            if len(buf) < HEADER.size + length:
                break
            payload = bytes(buf[HEADER.size:HEADER.size + length])
            del buf[:HEADER.size + length]
            frames.append((mtype, payload))
    return frames


def load_pcm_s16le(clip_path: str) -> bytes:
    if not os.path.exists(clip_path):
        sys.exit(f"reference clip not found: {clip_path}")
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", clip_path, "-f", "s16le", "-acodec", "pcm_s16le",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-"], capture_output=True)
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed on {clip_path}: {proc.stderr.decode(errors='replace')}")
    return proc.stdout


def connect(socket_path: str, timeout_s: float = 60.0) -> socket.socket:
    deadline = time.monotonic() + timeout_s  # sidecar loads the model before it accepts
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(120.0)  # finalize() runs a GPU decode
            sock.connect(socket_path)
            return sock
        except OSError as exc:
            sock.close()
            last_err = exc
            time.sleep(0.1)
    raise TimeoutError(f"could not connect to {socket_path}: {last_err}")


def probe(socket_path: str, model_task: str, clip: str) -> None:
    pcm = load_pcm_s16le(clip)
    dur_s = len(pcm) / 2 / SAMPLE_RATE
    n_frames = -(-len(pcm) // (CHUNK_SAMPLES * 2))
    sock = connect(socket_path)
    try:
        send_frame(sock, HELLO, json.dumps(
            {"protocol": PROTOCOL_VERSION, "model_task": model_task}).encode("utf-8"))
        mtype, payload = recv_frame(sock)
        if mtype != READY:
            sys.exit(f"expected READY (0x02), got 0x{mtype:02x}: {payload!r}")
        print(f"handshake -> READY {payload.decode(errors='replace')}")
        print(f"\nclip={os.path.basename(clip)} dur={dur_s:.2f}s frames={n_frames} "
              f"(feed as-fast-as-possible, drain 50 ms after each; NO flush yet)\n")

        n = 0
        events_during_stream = 0
        t0 = time.monotonic()
        for off in range(0, len(pcm), CHUNK_SAMPLES * 2):
            send_frame(sock, AUDIO, pcm[off:off + CHUNK_SAMPLES * 2])
            n += 1
            for mt, pl in drain_nonblocking(sock, 0.05):
                if mt == EVENT:
                    events_during_stream += 1
                    if events_during_stream == 1:
                        txt = (json.loads(pl).get("partial_text") or {}).get("text", "")
                        print(f"  *** EVENT during stream after frame {n} "
                              f"(t={(time.monotonic()-t0)*1000:.0f} ms): partial={txt!r}")
                else:
                    print(f"  frame {n}: unexpected frame 0x{mt:02x}")
        print(f"\nEVENTs seen DURING stream (pre-FLUSH): {events_during_stream}")

        t_flush = time.monotonic()
        send_frame(sock, FLUSH, b"{}")
        first_post_flush_ms: float | None = None
        n_events_post = 0
        final_txt: str | None = None
        while True:
            mt, pl = recv_frame(sock)
            if mt == EVENT:
                if first_post_flush_ms is None:
                    first_post_flush_ms = (time.monotonic() - t_flush) * 1000
                n_events_post += 1
            elif mt == FINAL:
                final_txt = (json.loads(pl).get("text_output") or {}).get("text", "")
                break
            elif mt == ERROR:
                sys.exit(f"sidecar ERROR: {pl.decode(errors='replace')}")
        print(f"After FLUSH: first EVENT @ {first_post_flush_ms:.0f} ms, "
              f"{n_events_post} EVENT delta(s), FINAL={final_txt!r}")
        print(f"FLUSH->FINAL total: {(time.monotonic()-t_flush)*1000:.0f} ms")
        print("\nCONCLUSION: interims arrive only at finalize when events_during_stream == 0 "
              "(the observed case on the earlier audio.cpp pins).")

        send_frame(sock, BYE, b"{}")
    finally:
        sock.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--socket", help="connect to an already-running sidecar at this path")
    group.add_argument("--spawn", metavar="BIN", help="spawn this sidecar binary on a temp socket")
    ap.add_argument("--model", help="model GGUF path (required with --spawn)")
    ap.add_argument("--backend", default="cuda", help="inference backend (default cuda)")
    ap.add_argument("--device", type=int, default=3, help="backend device index (default 3)")
    ap.add_argument("--threads", type=int, default=4, help="inference threads (default 4)")
    ap.add_argument("--model-task", default=DEFAULT_MODEL_TASK, help="registered wire id")
    ap.add_argument("--clip", default=DEFAULT_CLIP, help="reference audio clip")
    args = ap.parse_args()

    if args.socket:
        probe(args.socket, args.model_task, args.clip)
        return 0

    binary = os.path.abspath(args.spawn)
    if not os.path.exists(binary):
        sys.exit(f"sidecar binary not found: {binary}")
    if not args.model:
        sys.exit("--model is required with --spawn")

    tmpdir = tempfile.mkdtemp(prefix="audiocpp-timing-")
    socket_path = os.path.join(tmpdir, "sidecar.sock")
    proc = subprocess.Popen([
        binary, "--socket", socket_path, "--model", args.model,
        "--backend", args.backend, "--device", str(args.device), "--threads", str(args.threads),
    ])
    try:
        probe(socket_path, args.model_task, args.clip)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        for path in (socket_path,):
            try:
                os.path.exists(path) and os.unlink(path)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (EOFError, TimeoutError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
