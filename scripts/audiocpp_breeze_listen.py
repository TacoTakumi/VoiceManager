"""Breeze listen-clip generator for side-by-side sidecar comparisons.

MANUAL live script - never collected by pytest (scripts/ is outside tests/).
Needs a running VoiceManager server with VM_TTS_ENGINE=audiocpp on the Breeze
model task. Request options (for example seed=2) are a server-side setting,
VM_AUDIOCPP_TTS_REQUEST_OPTIONS, so set them on the server before you run this;
--overlay only labels the clips with the overlay that server was started with.

On one WS session it synthesizes the LONGFORM passage from
scripts/audiocpp_tts_evidence.py twice:
  * once as a single one-shot tts request;
  * once as per-sentence tts_stream chunks, sent 1.5 s apart, the last chunk
    carrying done=true;
and writes each reply to a 24 kHz mono 16-bit WAV. For each clip it prints the
audio length, the first-audio latency and the wall time.

Output files: <out-dir>/<prefix>[_<overlay>]_oneshot.wav and ..._stream.wav.

Usage:
    VOICEMANAGER_TOKEN=vmp_... ./venv/bin/python scripts/audiocpp_breeze_listen.py \
        --port 8077 --voice designed_female_s101 --overlay seed=2 \
        --out-dir tts_evidence/breeze --prefix v073
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
import wave
from pathlib import Path

import websockets

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8077
DEFAULT_VOICE = "designed_female_s101"
STREAM_GAP_S = 1.5
EXPECTED_RATE = 24000  # Breeze output rate; audio_start declares the real one
RECV_TIMEOUT_S = 180.0


def load_passage() -> str:
    """Import LONGFORM from the sibling evidence script.

    Deferred so --help works without VOICEMANAGER_TOKEN: that module reads the
    token at import time.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audiocpp_tts_evidence import LONGFORM
    return LONGFORM


def split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation; later chunks keep a leading space."""
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
    return [parts[0]] + [" " + p for p in parts[1:]] if parts else []


def overlay_slug(label: str) -> str:
    """seed=2 -> seed2; several KEY=VALUE pairs join with '-'."""
    return re.sub(r"[^A-Za-z0-9-]+", "", label.replace(",", "-"))


class Reply:
    """Collects one TTS reply: PCM, declared rate and first-audio time."""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.pcm = bytearray()
        self.rate: int | None = None
        self.first_audio: float | None = None


async def collect(ws, reply: Reply) -> None:
    """Read frames until audio_end for reply.request_id; raise on an error frame."""
    while True:
        frame = await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT_S)
        if isinstance(frame, bytes):
            if reply.first_audio is None:
                reply.first_audio = time.monotonic()
            reply.pcm += frame
            continue
        msg = json.loads(frame)
        mtype = msg.get("type")
        if mtype == "audio_start" and msg.get("request_id") == reply.request_id:
            reply.rate = int(msg["sample_rate"])
        elif mtype == "audio_end" and msg.get("request_id") == reply.request_id:
            if msg.get("reason", "complete") != "complete":
                raise RuntimeError(f"{reply.request_id} ended early: {msg}")
            return
        elif mtype == "error":
            raise RuntimeError(f"server error during {reply.request_id}: {msg}")


async def one_shot(ws, text: str, voice: str) -> tuple[Reply, float]:
    reply = Reply("listen-oneshot")
    t0 = time.monotonic()
    await ws.send(json.dumps({"type": "tts", "text": text,
                              "request_id": reply.request_id, "voice": voice}))
    await collect(ws, reply)
    return reply, t0


async def stream(ws, text: str, voice: str) -> tuple[Reply, float]:
    reply = Reply("listen-stream")
    chunks = split_sentences(text)
    receiver = asyncio.create_task(collect(ws, reply))
    t0 = time.monotonic()
    try:
        for i, chunk in enumerate(chunks):
            delay = t0 + i * STREAM_GAP_S - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            frame = {"type": "tts_stream", "text": chunk,
                     "request_id": reply.request_id, "voice": voice}
            if i == len(chunks) - 1:
                frame["done"] = True
            await ws.send(json.dumps(frame))
        await receiver
    finally:
        if not receiver.done():
            receiver.cancel()
    return reply, t0


def write_wav(path: Path, reply: Reply) -> tuple[int, float]:
    rate = reply.rate or EXPECTED_RATE
    if rate != EXPECTED_RATE:
        print(f"WARNING {path.name}: server declared {rate} Hz, not {EXPECTED_RATE} Hz; "
              "writing at the declared rate")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(bytes(reply.pcm))
    return rate, len(reply.pcm) / 2 / rate


def report(label: str, path: Path, reply: Reply, t0: float, t_end: float) -> None:
    rate, length = write_wav(path, reply)
    first_ms = (reply.first_audio - t0) * 1000 if reply.first_audio else float("nan")
    wall = t_end - t0
    print(f"CLIP {label}: length_s={length:.2f} first_audio_ms={first_ms:.0f} "
          f"wall_s={wall:.2f} rate={rate} -> {path}")


async def run(args) -> int:
    token = os.environ.get("VOICEMANAGER_TOKEN")
    if not token:
        print("VOICEMANAGER_TOKEN is not set", file=sys.stderr)
        return 2
    text = load_passage()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.prefix + (f"_{overlay_slug(args.overlay)}" if args.overlay else "")
    url = f"ws://{args.host}:{args.port}/ws/stream"
    headers = {"Authorization": f"Bearer {token}"}

    print(f"server {args.host}:{args.port} voice={args.voice} "
          f"overlay={args.overlay or '(none)'} passage_chars={len(text)}")
    async with websockets.connect(url, max_size=None, additional_headers=headers) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=30.0))
        if ready.get("type") != "ready":
            print(f"unexpected first frame: {ready}", file=sys.stderr)
            return 1

        reply, t0 = await one_shot(ws, text, args.voice)
        report("oneshot", out_dir / f"{stem}_oneshot.wav", reply, t0, time.monotonic())

        reply, t0 = await stream(ws, text, args.voice)
        report("stream", out_dir / f"{stem}_stream.wav", reply, t0, time.monotonic())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Write one-shot and per-sentence tts_stream Breeze listen clips "
                    "of the LONGFORM passage from one WS session.")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"server host (default {DEFAULT_HOST})")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"server port (default {DEFAULT_PORT})")
    ap.add_argument("--voice", default=DEFAULT_VOICE, help=f"voice (default {DEFAULT_VOICE})")
    ap.add_argument("--overlay", default="",
                    help="label for the request-option overlay the server runs with, "
                         "e.g. seed=2 (set on the server via "
                         "VM_AUDIOCPP_TTS_REQUEST_OPTIONS); added to the clip names")
    ap.add_argument("--out-dir", default="tts_evidence",
                    help="directory for the WAVs (default tts_evidence)")
    ap.add_argument("--prefix", required=True,
                    help="clip-name prefix, e.g. the build label or short hash")
    return asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
