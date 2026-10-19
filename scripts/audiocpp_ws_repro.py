#!/usr/bin/env python3
"""End-to-end /ws/stream repro for the audio.cpp engine (no mid-session frames)
- MANUAL, needs a running VoiceManager server on the audiocpp engine + the real sidecar.

What it shows: streaming a multi-word
utterance over the real /ws/stream under VM_STT_ENGINE=audiocpp currently yields
**0 transcription frames** during the session - no is_final=False partials, no
per-utterance is_final=True final - because the sidecar model emits interim deltas only
at finalize and nothing on VoiceManager's side FLUSHes per utterance (audiocpp has no
mid-session endpoint trigger; qwen has silence-finalize, nemotron has native EoU + hold).

This is the **regression target**: once a mid-session endpoint trigger is wired,
this repro must instead print >=1 is_final=False partial and exactly one is_final=True
final DURING the session (not only at teardown). A `pytest`-collected model-free test
must NOT depend on a live server; this stays a MANUAL scripts/ helper (never collected).

Start the server first (test port, does not touch your .env), e.g.:

  VM_STT_ENGINE=audiocpp \
  VM_AUDIOCPP_MODEL=/home/user/audiocpp-models/Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-f16.gguf \
  VM_AUDIOCPP_BACKEND=cuda VM_AUDIOCPP_DEVICE=0 VM_DEVICE=cuda:0 \
  VM_HOST=127.0.0.1 VM_PORT=8077 \
  ./venv/bin/python -m voicemanager

Then, in another shell:

  export VOICEMANAGER_TOKEN=vmp_...   # from: voicemanager user token add USER NAME
  ./venv/bin/python scripts/audiocpp_ws_repro.py --port 8077
  ./venv/bin/python scripts/audiocpp_ws_repro.py --port 8077 --token vmp_... --clip voices/designed/designed_male_s202.wav
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import websockets  # the Python client's transport; also used by the test suite

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 5120  # ~0.32 s
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLIP = os.path.join(REPO_ROOT, "voices", "designed", "designed_female_s101.wav")


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


async def run(port: int, token: str, clip: str, realtime: bool, wait_s: float) -> int:
    pcm = load_pcm_s16le(clip)
    dur = len(pcm) / 2 / SAMPLE_RATE
    uri = f"ws://127.0.0.1:{port}/ws/stream"
    headers = {"Authorization": f"Bearer {token}"}
    print(f"connecting {uri}\nclip={os.path.basename(clip)} dur={dur:.2f}s realtime={realtime}")

    async with websockets.connect(uri, max_size=None, additional_headers=headers) as ws:
        hs = json.loads(await ws.recv())
        print(f"handshake: stt_engine={hs.get('stt_engine')} session={hs.get('session_id')}")
        msgs: list[dict] = []
        stop = asyncio.Event()

        async def reader() -> None:
            try:
                while True:
                    m = json.loads(await ws.recv())
                    print(f"  [{(time.monotonic()-t0)*1000:7.0f} ms] {m.get('type')} "
                          f"is_final={m.get('is_final')} endpoint={m.get('endpoint')} "
                          f"seg={m.get('segment_index')} text={m.get('text')!r}")
                    msgs.append(m)
            except Exception as exc:
                print(f"  reader done: {type(exc).__name__}")
            finally:
                stop.set()

        t0 = time.monotonic()
        rtask = asyncio.create_task(reader())
        chunk_dur = CHUNK_SAMPLES / SAMPLE_RATE
        n = 0
        for off in range(0, len(pcm), CHUNK_SAMPLES * 2):
            await ws.send(pcm[off:off + CHUNK_SAMPLES * 2])
            n += 1
            if realtime:
                await asyncio.sleep(chunk_dur)
        print(f"sent {n} audio frames ({dur:.2f}s); waiting {wait_s:.0f}s for mid-stream frames...")
        await asyncio.sleep(wait_s)

        during_partials = sum(1 for m in msgs if m.get("is_final") is False)
        during_finals = sum(1 for m in msgs if m.get("is_final") is True)
        print(f"\n>>> DURING SESSION: {during_partials} is_final=False partial(s), "
              f"{during_finals} is_final=True final(s) [{len(msgs)} total frames]")

        print("closing WS (teardown)...")
        await ws.close()
        try:
            await asyncio.wait_for(stop.wait(), timeout=5)
        except asyncio.TimeoutError:
            pass
        rtask.cancel()

    total_partials = sum(1 for m in msgs if m.get("is_final") is False)
    total_finals = sum(1 for m in msgs if m.get("is_final") is True)
    print(f">>> TOTAL (incl. teardown): {total_partials} is_final=False, "
          f"{total_finals} is_final=True")

    # Exit non-zero while the bug is present (0 mid-session frames) so this doubles as a
    # regression gate: it should exit 0 once the endpoint trigger lands.
    ok = during_partials >= 1 and during_finals >= 1
    print("RESULT:", "PASS (mid-session frames arrive)" if ok
          else "REPRO CONFIRMED - 0 mid-session frames (not yet fixed)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8077, help="running server port (default 8077)")
    ap.add_argument("--token", default="", help="bearer token (default: $VOICEMANAGER_TOKEN)")
    ap.add_argument("--clip", default=DEFAULT_CLIP, help="reference audio clip")
    ap.add_argument("--no-realtime", action="store_true",
                    help="stream as-fast-as-possible instead of at 1x")
    ap.add_argument("--wait", type=float, default=3.0,
                    help="seconds to wait after the last frame for mid-stream output")
    args = ap.parse_args()
    token = args.token or _token_from_env()
    if not token:
        sys.exit("no token: pass --token or export VOICEMANAGER_TOKEN")
    return asyncio.run(run(args.port, token, args.clip,
                           realtime=not args.no_realtime, wait_s=args.wait))


if __name__ == "__main__":
    sys.exit(main())
