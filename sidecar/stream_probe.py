#!/usr/bin/env python3
"""Socket harness for the audio.cpp sidecar streaming drive loop.

This is a MANUAL script - it needs the compiled `audiocpp_sidecar` binary, the real
Nemotron-3.5 streaming GGUF, and a CUDA GPU. It is NOT part of the model-free pytest
suite: it lives outside tests/ and is not test_*-prefixed, so `pytest tests/` never
collects it.

It exercises the streaming path in PROTOCOL.md end to end:

  1. connect, HELLO -> READY (handshake, session warm)
  2. stream a chunked reference utterance as AUDIO frames
  3. FLUSH -> assert >=1 interim EVENT delta + exactly one FINAL, with real text
  4. RESET + a second utterance -> assert the warm session decodes again (held warm
     across utterances on one connection)

The final assertion is the drive loop's acceptance bar: "feeding a chunked utterance
over the socket emits >=1 interim delta + exactly one final back."

Usage:
  python sidecar/stream_probe.py \
      --spawn sidecar/build/bin/audiocpp_sidecar \
      --model /home/user/audiocpp-models/Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-f16.gguf \
      --backend cuda --device 3

Add --report-deltas to measure whether partials arrive DURING speech: AUDIO frames are
paced at 1x real time while a reader thread timestamps every EVENT as it arrives, and
each utterance reports the events seen before FLUSH, the ms from the first AUDIO frame to
the first delta, the total delta count, and whether the concatenated deltas equal the
FINAL text. Without the flag the probe streams as fast as it can, as before.
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
import threading
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
CHUNK_SAMPLES = 5120  # ~0.32 s chunks (the client's choice; the sidecar accepts any size)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLIP = os.path.join(REPO_ROOT, "voices", "designed", "designed_female_s101.wav")
SECOND_CLIP = os.path.join(REPO_ROOT, "voices", "designed", "designed_male_s202.wav")


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
    print(f"  handshake -> READY (session warm) {json.dumps(ack)}")
    return ack


def stream_utterance(sock: socket.socket, pcm: bytes, label: str) -> str:
    """Stream `pcm` as AUDIO frames, FLUSH, and assert >=1 EVENT + exactly one FINAL.

    Returns the final transcript text.
    """
    n_frames = 0
    for offset in range(0, len(pcm), CHUNK_SAMPLES * 2):
        send_frame(sock, AUDIO, pcm[offset:offset + CHUNK_SAMPLES * 2])
        n_frames += 1
    t_flush = time.monotonic()
    send_json(sock, FLUSH, {})

    events: list[dict] = []
    finals: list[dict] = []
    first_event_ms: float | None = None
    while True:
        mtype, payload = recv_frame(sock)
        if mtype == EVENT:
            if first_event_ms is None:
                first_event_ms = (time.monotonic() - t_flush) * 1000.0
            events.append(json.loads(payload))
        elif mtype == FINAL:
            finals.append(json.loads(payload))
            break
        elif mtype == ERROR:
            raise CheckError(f"{label}: sidecar returned ERROR: {payload.decode(errors='replace')}")
        else:
            raise CheckError(f"{label}: unexpected frame 0x{mtype:02x}: {payload!r}")

    expect(len(finals) == 1, f"{label}: expected exactly one FINAL, got {len(finals)}")
    final = finals[0]
    expect(final.get("is_final") is True, f"{label}: FINAL.is_final is not true: {final}")
    text = (final.get("text_output") or {}).get("text", "")
    expect(bool(text.strip()), f"{label}: FINAL text_output is empty: {final}")
    expect(len(events) >= 1, f"{label}: expected >=1 interim EVENT delta, got {len(events)}")
    for ev in events:
        expect(ev.get("is_final") is False, f"{label}: an EVENT was flagged is_final: {ev}")

    last_partial = ""
    for ev in reversed(events):
        if ev.get("partial_text"):
            last_partial = ev["partial_text"].get("text", "")
            break
    print(f"  {label}: sent {n_frames} AUDIO frames -> {len(events)} EVENT delta(s) "
          f"(first @ {first_event_ms:.0f} ms) + 1 FINAL")
    print(f"    last partial : {last_partial!r}")
    print(f"    final text   : {text!r}")
    return text


def _read_until_final(sock: socket.socket, frames: list[tuple[float, int, bytes]],
                      errors: list[BaseException]) -> None:
    """Reader thread body: timestamp each frame on arrival until FINAL, ERROR or EOF."""
    try:
        while True:
            mtype, payload = recv_frame(sock)
            frames.append((time.monotonic(), mtype, payload))
            if mtype in (FINAL, ERROR):
                return
    except BaseException as exc:  # handed to the main thread, which re-raises it
        errors.append(exc)


def stream_utterance_report(sock: socket.socket, pcm: bytes, label: str,
                            final_timeout_s: float = 120.0) -> str:
    """Stream `pcm` at 1x real time while reading EVENTs, FLUSH, and report the deltas.

    Unlike stream_utterance this does not require any EVENT: it measures when they
    arrive. It still asserts exactly one non-empty FINAL. Returns the final text.
    """
    frames: list[tuple[float, int, bytes]] = []
    errors: list[BaseException] = []
    # The reader blocks until the next frame, which may be a whole utterance away while
    # audio is paced in real time, so it waits without a socket timeout; the join below
    # bounds the wait for the FINAL instead.
    old_timeout = sock.gettimeout()
    sock.settimeout(None)
    reader = threading.Thread(target=_read_until_final, args=(sock, frames, errors),
                              daemon=True)
    reader.start()
    try:
        chunk_bytes = CHUNK_SAMPLES * 2
        n_frames = 0
        t_first_audio = time.monotonic()
        for offset in range(0, len(pcm), chunk_bytes):
            # Chunk start time in the clip, relative to the first AUDIO frame (1x pacing).
            delay = t_first_audio + (offset // 2) / SAMPLE_RATE - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            send_frame(sock, AUDIO, pcm[offset:offset + chunk_bytes])
            n_frames += 1
        # Let the last chunk's audio "play out" so FLUSH lands at the end of speech.
        audio_s = (len(pcm) // 2) / SAMPLE_RATE
        delay = t_first_audio + audio_s - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        t_flush = time.monotonic()
        send_json(sock, FLUSH, {})

        reader.join(timeout=final_timeout_s)
        if reader.is_alive():
            raise TimeoutError(f"{label}: no FINAL within {final_timeout_s:.0f} s of FLUSH")
    finally:
        if not reader.is_alive():
            sock.settimeout(old_timeout)
    if errors:
        raise errors[0]

    events: list[tuple[float, dict]] = []
    finals: list[dict] = []
    for t_arrival, mtype, payload in frames:
        if mtype == EVENT:
            events.append((t_arrival, json.loads(payload)))
        elif mtype == FINAL:
            finals.append(json.loads(payload))
        elif mtype == ERROR:
            raise CheckError(f"{label}: sidecar returned ERROR: {payload.decode(errors='replace')}")
        else:
            raise CheckError(f"{label}: unexpected frame 0x{mtype:02x}: {payload!r}")

    expect(len(finals) == 1, f"{label}: expected exactly one FINAL, got {len(finals)}")
    final = finals[0]
    expect(final.get("is_final") is True, f"{label}: FINAL.is_final is not true: {final}")
    text = (final.get("text_output") or {}).get("text", "")
    expect(bool(text.strip()), f"{label}: FINAL text_output is empty: {final}")

    before_flush = [t for t, _ in events if t < t_flush]
    deltas = [(t, (ev.get("partial_text") or {}).get("text", "")) for t, ev in events]
    deltas = [(t, d) for t, d in deltas if d]
    concat = "".join(d for _, d in deltas)
    first_delta_ms = (deltas[0][0] - t_first_audio) * 1000.0 if deltas else None
    first_delta = f"{first_delta_ms:.0f} ms" if first_delta_ms is not None else "none"

    print(f"  {label}: sent {n_frames} AUDIO frames at 1x ({audio_s:.2f} s of audio), "
          f"FLUSH @ {(t_flush - t_first_audio) * 1000.0:.0f} ms")
    print(f"    events before FLUSH       : {len(before_flush)} of {len(events)} EVENT(s)")
    print(f"    first AUDIO -> first delta: {first_delta}")
    print(f"    total deltas              : {len(deltas)}")
    print(f"    concat deltas == FINAL    : {concat == text}"
          + ("" if concat == text else f" (stripped: {concat.strip() == text.strip()}, "
             f"last delta == FINAL: {bool(deltas) and deltas[-1][1] == text})"))
    print(f"    concat deltas: {concat!r}")
    print(f"    final text   : {text!r}")
    return text


def run_checks(socket_path: str, model_task: str, report_deltas: bool = False) -> None:
    print(f"stream probe against {socket_path}")
    stream = stream_utterance_report if report_deltas else stream_utterance
    pcm1 = load_pcm_s16le(DEFAULT_CLIP)
    sock = connect(socket_path)
    try:
        handshake(sock, model_task)
        stream(sock, pcm1, "utterance-1")

        # Warm reuse across utterances on the same connection: RESET, decode
        # a second clip without reloading the model.
        send_json(sock, RESET, {})
        pcm2 = load_pcm_s16le(SECOND_CLIP) if os.path.exists(SECOND_CLIP) else pcm1
        stream(sock, pcm2, "utterance-2 (after RESET)")

        send_json(sock, BYE, {})
        print("ALL CHECKS PASSED")
    finally:
        sock.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--socket", help="connect to an already-running sidecar at this path")
    group.add_argument("--spawn", metavar="BIN", help="spawn this sidecar binary on a temp socket")
    parser.add_argument("--model", help="model GGUF path (required with --spawn)")
    parser.add_argument("--backend", default="cuda", help="inference backend (default cuda)")
    parser.add_argument("--device", type=int, default=0, help="backend device index (default 0)")
    parser.add_argument("--threads", type=int, default=4, help="inference threads (default 4)")
    parser.add_argument("--model-task", default=DEFAULT_MODEL_TASK, help="registered wire id to test")
    parser.add_argument("--report-deltas", action="store_true",
                        help="pace AUDIO at 1x real time, read EVENTs while feeding, and "
                             "report the deltas received before FLUSH per utterance")
    args = parser.parse_args()

    if args.socket:
        run_checks(args.socket, args.model_task, args.report_deltas)
        return 0

    binary = os.path.abspath(args.spawn)
    if not os.path.exists(binary):
        print(f"error: sidecar binary not found: {binary}", file=sys.stderr)
        return 2
    if not args.model:
        print("error: --model is required with --spawn", file=sys.stderr)
        return 2

    tmpdir = tempfile.mkdtemp(prefix="audiocpp-sidecar-")
    socket_path = os.path.join(tmpdir, "sidecar.sock")
    proc = subprocess.Popen([
        binary, "--socket", socket_path, "--model", args.model,
        "--backend", args.backend, "--device", str(args.device), "--threads", str(args.threads),
    ])
    try:
        run_checks(socket_path, args.model_task, args.report_deltas)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
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
