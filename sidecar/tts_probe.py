#!/usr/bin/env python3
"""Live TTS smoke probe for the audio.cpp sidecar.

This is a MANUAL script - it needs the compiled `audiocpp_sidecar` binary, the real
dots_tts Q8 GGUF, and a CUDA GPU. It is NOT part of the model-free pytest suite: it
lives outside tests/ and is not test_*-prefixed, so `pytest tests/` never collects it.

It exercises the TTS path in PROTOCOL.md end to end against the REAL model:

  1. connect, HELLO dots-tts-streaming -> READY describing task=tts and the 48 kHz
     output contract
  2. TEXT (a designed preset's WAV + ref_text from voices/designed/refs.json, and a
     multi-sentence input) -> TTS_AUDIO frames with per-frame timestamps -> TTS_DONE
  3. asserts the first TTS_AUDIO frame arrived well before TTS_DONE (incremental
     streaming) and writes the 48 kHz WAV for a listening check
  4. prints a timing block

The companion ASR leg (same binary) is the existing stream_probe.py:

  python sidecar/stream_probe.py --spawn sidecar/build-0.6/bin/audiocpp_sidecar \
      --model .../nemotron-3.5-asr-streaming-0.6b-f16.gguf --backend cuda --device 3

Usage:
  python sidecar/tts_probe.py \
      --spawn sidecar/build-0.6/bin/audiocpp_sidecar \
      --model /home/user/audiocpp-models/DotTTS-SOAR-GGUF/dots-tts-soar-q8_0.gguf \
      --backend cuda --device 3 --voice designed_female_s101 \
      --out sidecar/tts_probe_output.wav
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
import wave

# Message type tags - must match sidecar/src/protocol.h.
HELLO = 0x01
READY = 0x02
ERROR = 0x08
BYE = 0x09
TEXT = 0x0A
TTS_AUDIO = 0x0B
TTS_DONE = 0x0C

PROTOCOL_VERSION = 1
DEFAULT_MODEL_TASK = "dots-tts-streaming"
TTS_SAMPLE_RATE = 48000

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFS_JSON = os.path.join(REPO_ROOT, "voices", "designed", "refs.json")

DEFAULT_TEXT = (
    "Long range sensors have completed their sweep of the outer system. "
    "Three vessels are holding station near the fifth planet. "
    "Shields are at full strength, and all decks report ready."
)

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


def connect(socket_path: str, timeout_s: float = 30.0) -> socket.socket:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(600.0)  # model load + long generations
            sock.connect(socket_path)
            return sock
        except OSError as exc:
            last_err = exc
            time.sleep(0.05)
    raise TimeoutError(f"could not connect to {socket_path}: {last_err}")


class CheckError(AssertionError):
    pass


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise CheckError(message)


def load_preset(name: str) -> tuple[str, str]:
    """Resolve a designed preset to (absolute wav path, ref_text)."""
    with open(REFS_JSON, encoding="utf-8") as fh:
        refs = json.load(fh)
    entries = {k: v for k, v in refs.items() if not k.startswith("_")}
    if name not in entries:
        raise CheckError(f"unknown preset {name!r}; known: {sorted(entries)}")
    entry = entries[name]
    wav_path = os.path.join(os.path.dirname(REFS_JSON), entry["file"])
    ref_text = entry["ref_text"]
    expect(os.path.exists(wav_path), f"preset WAV missing: {wav_path}")
    expect(bool(ref_text.strip()), f"preset {name!r} has an empty ref_text")
    return wav_path, ref_text


def run_probe(socket_path: str, model_task: str, voice: str, text: str, out_path: str) -> None:
    wav_path, ref_text = load_preset(voice)
    print(f"tts probe against {socket_path}")
    print(f"  voice   : {voice} ({wav_path})")
    print(f"  text    : {text!r}")

    sock = connect(socket_path)
    try:
        t_hello = time.monotonic()
        send_json(sock, HELLO, {"protocol": PROTOCOL_VERSION, "model_task": model_task})
        mtype, payload = recv_frame(sock)
        expect(mtype == READY, f"expected READY (0x02), got 0x{mtype:02x}: {payload!r}")
        ack = json.loads(payload)
        load_s = time.monotonic() - t_hello
        expect(ack.get("ok") is True, f"READY.ok is not true: {ack}")
        expect(ack.get("task") == "tts", f"READY.task must be tts: {ack}")
        expect(ack.get("mode") == "streaming", f"READY.mode must be streaming: {ack}")
        expect(ack.get("input") == "none", f"READY.input must be none: {ack}")
        audio = ack.get("audio", {})
        expect(audio.get("sample_rate") == TTS_SAMPLE_RATE,
               f"READY.audio.sample_rate must be {TTS_SAMPLE_RATE}: {ack}")
        expect(audio.get("channels") == 1 and audio.get("format") == "s16le",
               f"READY.audio must be mono s16le: {ack}")
        expect("preferred_chunk_samples" not in ack,
               f"a TTS READY must not carry the input-side chunk hint: {ack}")
        print(f"  handshake -> READY (model warm in {load_s:.1f} s) {json.dumps(ack)}")

        t_text = time.monotonic()
        send_json(sock, TEXT, {"text": text, "voice_wav": wav_path, "ref_text": ref_text})

        frames: list[tuple[float, int]] = []  # (seconds since TEXT, payload bytes)
        pcm = bytearray()
        while True:
            mtype, payload = recv_frame(sock)
            now = time.monotonic() - t_text
            if mtype == TTS_AUDIO:
                frames.append((now, len(payload)))
                pcm.extend(payload)
            elif mtype == TTS_DONE:
                t_done = now
                break
            elif mtype == ERROR:
                raise CheckError(f"sidecar returned ERROR: {payload.decode(errors='replace')}")
            else:
                raise CheckError(f"unexpected frame 0x{mtype:02x}: {payload!r}")

        send_json(sock, BYE, {})
    finally:
        sock.close()

    expect(len(frames) >= 2, f"expected a stream of TTS_AUDIO frames, got {len(frames)}")
    first_audio_s = frames[0][0]
    last_audio_s = frames[-1][0]
    audio_seconds = len(pcm) / 2 / TTS_SAMPLE_RATE
    # Incremental: the first audio frame must arrive well before the request
    # completes - a batch-at-completion server would put every frame at ~t_done.
    expect(first_audio_s < t_done * 0.5,
           f"first audio at {first_audio_s:.2f}s is not meaningfully before done "
           f"at {t_done:.2f}s - not streaming")

    with wave.open(out_path, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(TTS_SAMPLE_RATE)
        out.writeframes(bytes(pcm))

    rtf = t_done / audio_seconds if audio_seconds else float("inf")
    print("\n  --- timing ---")
    print(f"  model load -> READY : {load_s:.1f} s")
    print(f"  first TTS_AUDIO     : {first_audio_s * 1000:.0f} ms after TEXT")
    print(f"  last TTS_AUDIO      : {last_audio_s:.2f} s;  TTS_DONE: {t_done:.2f} s")
    print(f"  frames              : {len(frames)}  ({len(pcm)} bytes)")
    print(f"  audio produced      : {audio_seconds:.2f} s @ 48 kHz  (RTF {rtf:.2f})")
    print(f"  output WAV          : {out_path}  <-- LISTEN to check the tail is clean")
    ts = "  frame timeline (s): " + ", ".join(f"{t:.2f}" for t, _ in frames[:20])
    print(ts + (" ..." if len(frames) > 20 else ""))
    print("ALL CHECKS PASSED")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--socket", help="connect to an already-running sidecar at this path")
    group.add_argument("--spawn", metavar="BIN", help="spawn this sidecar binary on a temp socket")
    parser.add_argument("--model", help="dots_tts GGUF path (required with --spawn)")
    parser.add_argument("--backend", default="cuda", help="inference backend (default cuda)")
    parser.add_argument("--device", type=int, default=0, help="backend device index (default 0)")
    parser.add_argument("--threads", type=int, default=4, help="inference threads (default 4)")
    parser.add_argument("--model-task", default=DEFAULT_MODEL_TASK, help="registered wire id")
    parser.add_argument("--voice", default="designed_female_s101",
                        help="preset name from voices/designed/refs.json")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="text to synthesize")
    parser.add_argument("--out", default=os.path.join(REPO_ROOT, "sidecar", "tts_probe_output.wav"),
                        help="output WAV path (48 kHz mono)")
    args = parser.parse_args()

    if args.socket:
        run_probe(args.socket, args.model_task, args.voice, args.text, args.out)
        return 0

    if not args.model:
        print("error: --model is required with --spawn", file=sys.stderr)
        return 2
    binary = os.path.abspath(args.spawn)
    if not os.path.exists(binary):
        print(f"error: sidecar binary not found: {binary}", file=sys.stderr)
        return 2
    tmpdir = tempfile.mkdtemp(prefix="audiocpp-tts-")
    socket_path = os.path.join(tmpdir, "s.sock")
    proc = subprocess.Popen([
        binary, "--socket", socket_path, "--model", args.model,
        "--backend", args.backend, "--device", str(args.device),
        "--threads", str(args.threads),
    ])
    try:
        run_probe(socket_path, args.model_task, args.voice, args.text, args.out)
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
        os.rmdir(tmpdir)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
