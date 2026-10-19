#!/usr/bin/env python3
"""Live /ws/stream cadence check for the Voxtral model_task
- MANUAL, needs a running VoiceManager server on the audiocpp engine + the real sidecar
+ the voxtral q4_k GGUF. Never collected by pytest (lives in scripts/).

What it proves live, on one WS connection streaming TWO clips at 1x:

  * mid-speech partials: >=2 growing is_final=False partials whose ARRIVAL timestamps
    precede the end of that utterance's speech feed (the genuine token-incremental
    streaming the spike measured -- a partials-only-at-finalize pattern fails);
  * exactly one is_final=True final per utterance, driven by the EXISTING endpointer
    (Voxtral has no internal endpointing; the silence trigger flushes per utterance);
  * segment_index increments across the two utterances and both transcribe -- the
    live check of consecutive utterances on one re-armed session.

The first utterance after model load pays warm-up (the spike saw the first partial
~2.8 s in on a cold session), so by default the >=2-partials-before-feed-end bar is
enforced on the WARM second utterance and reported (not enforced) for the first;
pass --strict-first to enforce it on both.

Start the server first (test port, does not touch your .env), e.g.:

  VM_STT_ENGINE=audiocpp \
  VM_AUDIOCPP_MODEL=/home/user/audiocpp-models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q4_k.gguf \
  VM_AUDIOCPP_MODEL_TASK=voxtral-mini-4b-realtime-asr-streaming \
  VM_AUDIOCPP_BACKEND=cuda VM_AUDIOCPP_DEVICE=0 VM_DEVICE=cuda:0 \
  VM_HOST=127.0.0.1 VM_PORT=8077 \
  ./venv/bin/python -m voicemanager

Then, in another shell:

  export VOICEMANAGER_TOKEN=vmp_...   # from: voicemanager user token add USER NAME
  ./venv/bin/python scripts/audiocpp_voxtral_ws_cadence.py --port 8077
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import websockets

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 5120  # ~0.32 s, the repro scripts' pacing
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The pinned head-to-head reference clips (3rdParty/tng/derived, regenerate with
# scripts/spike_voxtral_headtohead.py): two different singles so "both transcribe"
# is content-checked, not just frame-counted.
DEFAULT_CLIP1 = os.path.join(REPO_ROOT, "3rdParty", "tng", "derived", "longrange_16k_pad1s.wav")
DEFAULT_CLIP2 = os.path.join(REPO_ROOT, "3rdParty", "tng", "derived", "incoming_16k_pad1s.wav")


def load_pcm_s16le(clip: str) -> bytes:
    if not os.path.exists(clip):
        sys.exit(f"reference clip not found: {clip} (regenerate 3rdParty/tng/derived with "
                 "scripts/spike_voxtral_headtohead.py)")
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", clip, "-f", "s16le", "-acodec", "pcm_s16le",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-"], capture_output=True)
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed on {clip}: {proc.stderr.decode(errors='replace')}")
    return proc.stdout


def _token_from_env() -> str:
    return os.environ.get("VOICEMANAGER_TOKEN", "")


def _check_utterance(label: str, partials: list[dict], finals: list[dict],
                     feed_end: float, *, enforce_partials: bool) -> list[str]:
    """The per-utterance cadence bars; returns a list of failure strings."""
    fails: list[str] = []

    early = [p for p in partials if p["_t"] < feed_end]
    texts = [p["text"] for p in early]
    print(f"  {label}: {len(partials)} partial(s), {len(early)} before feed end "
          f"({feed_end*1000:.0f} ms), {len(finals)} final(s)")
    for p in partials:
        marker = "<" if p["_t"] < feed_end else ">"
        print(f"    [{p['_t']*1000:7.0f} ms {marker} feed end] partial {p['text']!r}")
    for f in finals:
        print(f"    [{f['_t']*1000:7.0f} ms] FINAL seg={f['segment_index']} {f['text']!r}")

    grown = len(texts) >= 2 and all(b.startswith(a) for a, b in zip(texts, texts[1:])) \
        and len(texts[-1]) > len(texts[0])
    if enforce_partials and not grown:
        fails.append(f"{label}: expected >=2 GROWING partials before feed end, got {texts}")
    elif not enforce_partials and not grown:
        print(f"    (info) {label} did not show >=2 growing partials before feed end "
              "-- tolerated (cold-session warm-up)")

    if len(finals) != 1:
        fails.append(f"{label}: expected exactly one is_final=True final, got {len(finals)}")
    elif not finals[0]["text"].strip():
        fails.append(f"{label}: the final is empty -- the utterance did not transcribe")
    return fails


async def run(port: int, token: str, clip1: str, clip2: str, wait_s: float,
              strict_first: bool) -> int:
    pcm1, pcm2 = load_pcm_s16le(clip1), load_pcm_s16le(clip2)
    uri = f"ws://127.0.0.1:{port}/ws/stream"
    headers = {"Authorization": f"Bearer {token}"}
    print(f"connecting {uri}\n  clip1={os.path.basename(clip1)} "
          f"({len(pcm1)/2/SAMPLE_RATE:.2f}s)  clip2={os.path.basename(clip2)} "
          f"({len(pcm2)/2/SAMPLE_RATE:.2f}s)")

    msgs: list[dict] = []
    async with websockets.connect(uri, max_size=None, additional_headers=headers) as ws:
        hs = json.loads(await ws.recv())
        print(f"handshake: stt_engine={hs.get('stt_engine')} session={hs.get('session_id')}")
        t0 = time.monotonic()

        async def reader() -> None:
            try:
                while True:
                    m = json.loads(await ws.recv())
                    m["_t"] = time.monotonic() - t0  # arrival timestamp
                    if m.get("type") == "transcription":
                        msgs.append(m)
            except Exception:
                pass

        rtask = asyncio.create_task(reader())
        chunk_dur = CHUNK_SAMPLES / SAMPLE_RATE
        feed_ends: list[float] = []

        for pcm in (pcm1, pcm2):
            # Snapshot BEFORE feeding: with the clip's 1 s trailing pad streamed at 1x,
            # the endpointer can legitimately fire (and the final arrive) before the
            # last chunk is even sent, so waiting for a final "after the feed" can sit
            # out the whole timeout in silence -- and that idle gap triggers extra
            # empty flushes server-side. Wait for a NEW final relative to the snapshot
            # and move on promptly.
            n_finals = sum(1 for m in msgs if m["is_final"])
            for off in range(0, len(pcm), CHUNK_SAMPLES * 2):
                await ws.send(pcm[off:off + CHUNK_SAMPLES * 2])
                await asyncio.sleep(chunk_dur)  # 1x realtime -- the whole point
            feed_ends.append(time.monotonic() - t0)
            deadline = time.monotonic() + wait_s
            while (sum(1 for m in msgs if m["is_final"]) <= n_finals
                   and time.monotonic() < deadline):
                await asyncio.sleep(0.05)
            # A still-missing final is judged below and fails the utterance check.
        await asyncio.sleep(0.5)  # let any trailing frames land
        rtask.cancel()
        await ws.close()

    # Split frames into utterances at the finals.
    finals = [m for m in msgs if m["is_final"]]
    partials = [m for m in msgs if not m["is_final"]]
    if len(finals) < 1:
        print("\nRESULT: FAIL -- no is_final=True final arrived at all")
        return 1
    f1_t = finals[0]["_t"]
    utt1_p = [p for p in partials if p["_t"] < f1_t]
    utt2_p = [p for p in partials if p["_t"] >= f1_t]

    print("\ntimeline check:")
    fails = _check_utterance("utterance 1", utt1_p, finals[:1], feed_ends[0],
                             enforce_partials=strict_first)
    fails += _check_utterance("utterance 2", utt2_p, finals[1:2],
                              feed_ends[1] if len(feed_ends) > 1 else 0.0,
                              enforce_partials=True)

    if len(finals) > 2:
        fails.append(f"expected one final per utterance (2 total), got {len(finals)}")
    if len(finals) >= 2:
        s1, s2 = finals[0]["segment_index"], finals[1]["segment_index"]
        if s2 <= s1:
            fails.append(f"segment_index did not increase: {s1} -> {s2}")
        elif s2 != s1 + 1:
            # An idle gap between the utterances lets the silence endpointer fire an
            # empty flush, which consumes a segment index without emitting a frame.
            # Monotonicity holds; idle-feed gating removes the skips.
            print(f"  (warn) segment_index skipped ({s1} -> {s2}): an empty "
                  "inter-utterance flush consumed an index (idle gap)")

    if fails:
        print("\nRESULT: FAIL")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\nRESULT: PASS -- mid-speech growing partials, one final per utterance, "
          "monotonic segment_index across two utterances on one connection")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8077, help="running server port (default 8077)")
    ap.add_argument("--token", default="", help="bearer token (default: $VOICEMANAGER_TOKEN)")
    ap.add_argument("--clip1", default=DEFAULT_CLIP1, help="first utterance clip")
    ap.add_argument("--clip2", default=DEFAULT_CLIP2, help="second utterance clip")
    ap.add_argument("--wait", type=float, default=10.0,
                    help="seconds to wait for each utterance's final")
    ap.add_argument("--strict-first", action="store_true",
                    help="enforce the >=2-growing-partials bar on the cold first utterance too")
    args = ap.parse_args()
    token = args.token or _token_from_env()
    if not token:
        sys.exit("no token: pass --token or export VOICEMANAGER_TOKEN")
    return asyncio.run(run(args.port, token, args.clip1, args.clip2, args.wait,
                           args.strict_first))


if __name__ == "__main__":
    sys.exit(main())
