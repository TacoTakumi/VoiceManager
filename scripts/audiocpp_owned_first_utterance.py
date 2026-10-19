#!/usr/bin/env python3
"""Measure the FIRST-utterance cost of an audio.cpp /ws/stream session on the OWNED
(spawn-per-session) default - MANUAL, needs a running VoiceManager server + the real
sidecar + GPU.

Why this exists. The earlier sidecar-routing latency numbers were all measured in
EXTERNAL mode against ONE shared, already-warm sidecar
(`--socket /tmp/audiocpp-sidecar.sock`), so their "cold-session FIRST utterance ~1097 ms"
row only ever covered a per-connection CUDA-graph compile. A later fix made OWNED the
production default (`VM_AUDIOCPP_SOCKET` defaults to empty, one sidecar process per
session), and on that path the first `feed()` ALSO pays process spawn, CUDA context
init and the GGUF load. This script measures what a client actually waits for on the
shipped default, so `docs/client-integration.md` can quote a true number.

Method (mirrors the earlier measurement so the figures are comparable):

  * BURST-feed the clip (not realtime). The endpoint trigger is a wall-clock silence
    gap since the last audio frame, so realtime pacing (~0.32 s between chunks) fires
    it mid-stream and truncates the utterance - burst is the correct feed here.
  * Measure last-audio-frame -> first `is_final=True`. Under burst the whole clip is
    in flight before the pipeline's first `feed()`, so this interval contains the
    entire cold path: spawn + CUDA init + GGUF load + handshake + CUDA-graph compile
    + finalize decode + routing.
  * Subtract the server's silence window (`--silence-ms`, must match the server's
    `VM_AUDIOCPP_SILENCE_MS`) to get the engine-side cost.
  * Utterance 2+ runs on the SAME connection and is warm - the delta between them is
    the one-time per-connection cost a client can avoid by holding the connection.

Start a server on the OWNED default (no VM_AUDIOCPP_SOCKET), e.g.:

  VM_STT_ENGINE=audiocpp VM_AUDIOCPP_SILENCE_MS=100 \
  VM_HOST=127.0.0.1 VM_PORT=8077 \
  ./venv/bin/python -m voicemanager

Then:

  export VOICEMANAGER_TOKEN=vmp_...   # from: voicemanager user token add USER NAME
  ./venv/bin/python scripts/audiocpp_owned_first_utterance.py --port 8077 --repeats 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time

import websockets

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 5120  # ~0.32 s; the frame size the other repros use
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLIPS = [
    os.path.join(REPO_ROOT, "voices", "designed", "designed_female_s101.wav"),
    os.path.join(REPO_ROOT, "voices", "designed", "designed_male_s202.wav"),
]


def load_pcm_s16le(clip: str) -> bytes:
    if not os.path.exists(clip):
        sys.exit(f"reference clip not found: {clip}")
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", clip, "-f", "s16le", "-acodec", "pcm_s16le",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-"], capture_output=True)
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed on {clip}: {proc.stderr.decode(errors='replace')}")
    return proc.stdout


def _token_from_env() -> str:
    return os.environ.get("VOICEMANAGER_TOKEN", "")


async def _burst_utterance(ws, pcm: bytes, timeout: float,
                           verbose: bool = False) -> tuple[float, str]:
    """Burst-send `pcm`, then wait for this utterance's final. Returns (ms, text).

    A `timeout_warning` error frame is the session inactivity warning, not a failure:
    a burst feed sends nothing for the rest of the utterance, so the server's
    keep-alive warning can legitimately land while we wait. Log it and keep reading.
    """
    for off in range(0, len(pcm), CHUNK_SAMPLES * 2):
        await ws.send(pcm[off:off + CHUNK_SAMPLES * 2])
    t_last_frame = time.monotonic()
    deadline = t_last_frame + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"no final within {timeout:.0f}s")
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=remaining))
        if verbose:
            print(f"      [{(time.monotonic() - t_last_frame) * 1000:8.0f} ms] "
                  f"{json.dumps(msg)[:150]}")
        if msg.get("type") == "error":
            if msg.get("error_type") == "timeout_warning":
                continue
            raise RuntimeError(f"server error frame: {msg}")
        if msg.get("type") == "transcription" and msg.get("is_final") is True:
            return (time.monotonic() - t_last_frame) * 1000.0, msg.get("text", "")


async def one_connection(uri: str, headers: dict, clips: list[bytes], timeout: float,
                         verbose: bool = False) -> list[tuple[float, str]]:
    """One FRESH connection: utterance 1 is cold (spawn+load), 2+ are warm."""
    out: list[tuple[float, str]] = []
    async with websockets.connect(uri, max_size=None, additional_headers=headers) as ws:
        await ws.recv()  # handshake frame
        for pcm in clips:
            out.append(await _burst_utterance(ws, pcm, timeout, verbose))
    return out


async def run(port: int, token: str, clip_paths: list[str], repeats: int,
              silence_ms: float, timeout: float, verbose: bool = False) -> int:
    clips = [load_pcm_s16le(c) for c in clip_paths]
    uri = f"ws://127.0.0.1:{port}/ws/stream"
    headers = {"Authorization": f"Bearer {token}"}
    print(f"server={uri.split('?')[0]} repeats={repeats} silence_ms={silence_ms:.0f} "
          f"clips={[os.path.basename(c) for c in clip_paths]}")
    print("Each repeat is a FRESH connection => a fresh sidecar spawn on the OWNED default.\n")

    cold: list[float] = []
    warm: list[float] = []
    for r in range(1, repeats + 1):
        try:
            results = await one_connection(uri, headers, clips, timeout, verbose)
        except Exception as exc:  # noqa: BLE001 - a manual probe: report and fail loudly
            print(f"run {r}: FAILED: {type(exc).__name__}: {exc}")
            return 1
        for i, (ms, text) in enumerate(results, start=1):
            tag = "COLD (utt1, new connection)" if i == 1 else f"warm (utt{i})"
            print(f"run {r} {tag:28s} last-frame->final = {ms:8.0f} ms "
                  f"(minus silence: {ms - silence_ms:8.0f} ms)  text={text!r}")
            (cold if i == 1 else warm).append(ms)
        print()

    print("=" * 78)

    def _row(label: str, xs: list[float]) -> None:
        if not xs:
            return
        med = statistics.median(xs)
        print(f"{label:34s} n={len(xs)} median={med:8.0f} ms  "
              f"(minus {silence_ms:.0f} ms silence: {med - silence_ms:8.0f} ms)  "
              f"min={min(xs):.0f} max={max(xs):.0f}")

    _row("COLD first utterance (OWNED)", cold)
    _row("warm same-connection utterance", warm)
    if cold and warm:
        delta = statistics.median(cold) - statistics.median(warm)
        print(f"{'one-time per-connection cost':34s} = {delta:8.0f} ms "
              f"(cold median - warm median)")
    print("=" * 78)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--token", default="", help="bearer token (default: $VOICEMANAGER_TOKEN)")
    ap.add_argument("--clip", action="append", dest="clips", default=None,
                    help="reference clip (repeatable; first is the cold one)")
    ap.add_argument("--repeats", type=int, default=3, help="fresh connections to measure")
    ap.add_argument("--silence-ms", type=float, default=100.0,
                    help="server's VM_AUDIOCPP_SILENCE_MS, subtracted from the raw figure")
    ap.add_argument("--verbose", action="store_true", help="print every frame as it arrives")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="per-utterance wait for the final (cold spawn budget is 60s dial "
                         "+ 120s handshake read)")
    args = ap.parse_args()
    token = args.token or _token_from_env()
    if not token:
        sys.exit("no token: pass --token or export VOICEMANAGER_TOKEN")
    clips = args.clips or DEFAULT_CLIPS
    return asyncio.run(run(args.port, token, clips, args.repeats, args.silence_ms,
                           args.timeout, args.verbose))


if __name__ == "__main__":
    sys.exit(main())
