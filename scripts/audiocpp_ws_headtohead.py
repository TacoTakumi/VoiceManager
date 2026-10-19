#!/usr/bin/env python3
"""WS-level head-to-head measurement for one engine configuration - MANUAL, needs a
running VoiceManager server. Never collected by pytest.

Streams the pinned spike reference clip set (3rdParty/tng/derived/, regenerate with
scripts/spike_voxtral_headtohead.py) through the real /ws/stream at 1x and records,
per clip:

  * WS-observed TTFP -- wall clock from the clip's feed start to the arrival of its
    first non-empty is_final=False partial, under availability-time pacing: chunk i
    (covering audio [i*c,(i+1)*c)) is sent at t_start+(i+1)*c, never earlier -- the
    same live-mic model the spike harness used (and the same bug it fixed:
    send-then-sleep hands the engine future audio and flatters its TTFP);
  * every is_final=True final the clip produces (concat17 legitimately endpoints
    into several utterances over the WS -- both engines' acoustic endpointing splits
    at the intra-concat pauses the standalone spike harness decoded whole).

Between clips the feed continues with 1x-paced ZERO-PCM silence frames (the live-mic
model: a real client streams continuously) until the clip's final has arrived and the
line has been quiet for --settle seconds. This is load-bearing for the nemotron
engine: its native EoU fires from silence AUDIO, so with dead air between clips an
utterance stays uncommitted until the NEXT clip's speech arrives -- every final then
lands one window late, stitched to the following clip's text. The voxtral idle gate
classifies the same padding as inter-utterance silence and feeds none of it.

Run it once against each engine configuration (same port, one server at a time):

  # audiocpp + Voxtral (see scripts/audiocpp_voxtral_ws_cadence.py for the server env)
  ./venv/bin/python scripts/audiocpp_ws_headtohead.py --port 8077 \
      --label voxtral-ws --json /tmp/voxtral_ws.json

  # NeMo nemotron engine
  VM_STT_ENGINE=nemotron VM_DEVICE=cuda:0 VM_HOST=127.0.0.1 VM_PORT=8077 ... \
      ./venv/bin/python -m voicemanager   # then:
  ./venv/bin/python scripts/audiocpp_ws_headtohead.py --port 8077 \
      --label nemotron-ws --json /tmp/nemotron_ws.json

Per-clip figure = median TTFP across --runs (default 3) passes on ONE connection,
after a throwaway warm-up clip (so no engine pays session/sidecar warm-up in the
measured numbers).
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
CHUNK_SAMPLES = 5120  # ~0.32 s
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DERIVED = os.path.join(REPO_ROOT, "3rdParty", "tng", "derived")

# The pinned spike clip order.
CLIP_NAMES = ["access", "direction", "donot", "incoming", "longrange", "notauth",
              "concat17"]
WARMUP_CLIP = "donot"  # throwaway first feed; not recorded


def load_pcm(name: str) -> bytes:
    path = os.path.join(DERIVED, f"{name}_16k_pad1s.wav")
    if not os.path.exists(path):
        sys.exit(f"clip not found: {path} (regenerate 3rdParty/tng/derived with "
                 "scripts/spike_voxtral_headtohead.py)")
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-acodec", "pcm_s16le",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-"], capture_output=True)
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed on {path}: {proc.stderr.decode(errors='replace')}")
    return proc.stdout


def _token_from_env() -> str:
    return os.environ.get("VOICEMANAGER_TOKEN", "")


async def feed_paced(ws, pcm: bytes, t_start: float) -> None:
    """Availability-time pacing: chunk i goes out no earlier than t_start+(i+1)*c."""
    chunk_dur = CHUNK_SAMPLES / SAMPLE_RATE
    i = 0
    for off in range(0, len(pcm), CHUNK_SAMPLES * 2):
        due = t_start + (i + 1) * chunk_dur
        delay = due - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        await ws.send(pcm[off:off + CHUNK_SAMPLES * 2])
        i += 1


async def run(port: int, token: str, label: str, runs: int, settle_s: float,
              json_path: str | None) -> int:
    clips = {name: load_pcm(name) for name in CLIP_NAMES}
    warm = load_pcm(WARMUP_CLIP)
    uri = f"ws://127.0.0.1:{port}/ws/stream"
    headers = {"Authorization": f"Bearer {token}"}

    msgs: list[dict] = []
    results: dict[str, dict] = {name: {"ttfp_ms": [], "finals": []}
                                for name in CLIP_NAMES}

    async with websockets.connect(uri, max_size=None, additional_headers=headers) as ws:
        hs = json.loads(await ws.recv())
        engine = hs.get("stt_engine")
        print(f"connected: stt_engine={engine} label={label} runs={runs}")

        async def reader():
            try:
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("type") == "transcription":
                        m["_t"] = time.monotonic()
                        msgs.append(m)
            except Exception:
                pass

        rtask = asyncio.create_task(reader())

        silence_chunk = b"\x00" * CHUNK_SAMPLES * 2
        chunk_dur = CHUNK_SAMPLES / SAMPLE_RATE

        async def settle(n0: int, feed_end: float) -> None:
            """Stream 1x-paced silence until >=1 final since n0 and settle_s of quiet.

            Continuous silence FRAMES, not dead air -- see the module docstring: the
            nemotron engine's EoU only fires from silence audio.
            """
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                finals = [m for m in msgs[n0:] if m["is_final"]]
                last = max((m["_t"] for m in msgs[n0:]), default=feed_end)
                if finals and time.monotonic() - last >= settle_s:
                    return
                await ws.send(silence_chunk)
                await asyncio.sleep(chunk_dur)

        # Warm-up: stand the session (and any sidecar) up; not recorded.
        t0 = time.monotonic()
        await feed_paced(ws, warm, t0)
        await settle(0, time.monotonic())
        print(f"warm-up done ({sum(1 for m in msgs if m['is_final'])} final)")

        for run_idx in range(runs):
            for name in CLIP_NAMES:
                n0 = len(msgs)
                t_start = time.monotonic()
                await feed_paced(ws, clips[name], t_start)
                await settle(n0, time.monotonic())
                new = msgs[n0:]
                partials = [m for m in new if not m["is_final"] and m["text"].strip()]
                finals = [m["text"].strip() for m in new if m["is_final"]]
                ttfp = (partials[0]["_t"] - t_start) * 1000 if partials else None
                results[name]["ttfp_ms"].append(ttfp)
                if run_idx == 0:
                    results[name]["finals"] = finals
                print(f"  run {run_idx + 1} {name}: ttfp="
                      f"{f'{ttfp:.0f} ms' if ttfp is not None else 'NONE'} "
                      f"finals={finals}")
        rtask.cancel()
        await ws.close()

    print(f"\n## {label} (engine={engine}, median of {runs})\n")
    print("| clip | ttfp median ms | runs | finals (run 1) |")
    print("|---|---|---|---|")
    for name in CLIP_NAMES:
        r = results[name]
        vals = [v for v in r["ttfp_ms"] if v is not None]
        med = f"{statistics.median(vals):.0f}" if vals else "NONE"
        per = ", ".join(f"{v:.0f}" if v is not None else "-" for v in r["ttfp_ms"])
        print(f"| {name} | {med} | {per} | {' / '.join(r['finals'])} |")

    if json_path:
        with open(json_path, "w") as f:
            json.dump({"label": label, "engine": engine, "runs": runs,
                       "results": results}, f, indent=2)
        print(f"\nwrote {json_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--token", default="", help="bearer token (default: $VOICEMANAGER_TOKEN)")
    ap.add_argument("--label", default="ws", help="label for the output table/json")
    ap.add_argument("--runs", type=int, default=3, help="passes per clip (median)")
    ap.add_argument("--settle", type=float, default=1.6,
                    help="seconds of frame quiet that ends a clip's collection")
    ap.add_argument("--json", default=None, help="write results as JSON here")
    args = ap.parse_args()
    token = args.token or _token_from_env()
    if not token:
        sys.exit("no token: pass --token or export VOICEMANAGER_TOKEN")
    return asyncio.run(run(args.port, token, args.label, args.runs, args.settle,
                           args.json))


if __name__ == "__main__":
    sys.exit(main())
