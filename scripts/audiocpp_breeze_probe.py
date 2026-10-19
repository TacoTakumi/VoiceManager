"""Multi-request probe on one live Breeze session.

MANUAL live script - never collected by pytest (scripts/ is outside tests/).
Needs the real server running on a sidecar build (v0.7.3 or later) with the breeze
task, i.e. the switched state in .env:

    VM_TTS_ENGINE=audiocpp
    VM_AUDIOCPP_SIDECAR_BIN=sidecar/build-<hash8>/bin/audiocpp_sidecar
    VM_AUDIOCPP_TTS_MODEL=/home/user/audiocpp-models/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf
    VM_AUDIOCPP_TTS_MODEL_TASK=breeze-tts-streaming

Opens ONE WS session and sends three one-shot tts requests of differing lengths
back to back. Each must yield audio_start, non-empty PCM, and audio_end with
reason "complete" (the WS face of the sidecar's TTS_DONE), with no error frame
and no WS close in between. The TTS sidecar process must still be alive after
the last request. Prints a paste-ready block; exit 0 on pass, 1 on fail.

Usage: VOICEMANAGER_TOKEN=vmp_... BZ_PORT=8255 BZ_GPU=3 python scripts/audiocpp_breeze_probe.py
"""
import asyncio
import json
import os
import subprocess
import sys
import time
import wave

import websockets

PORT = int(os.environ.get("BZ_PORT", "8255"))
TOKEN = os.environ["VOICEMANAGER_TOKEN"]
URL = f"ws://localhost:{PORT}/ws/stream"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
GPU_INDEX = os.environ.get("BZ_GPU", "3")
OUT_DIR = os.environ.get("BZ_OUT", "/tmp")

REQUESTS = [
    ("r1_short", "Shields are holding."),
    ("r2_medium", "Shields are holding at sixty percent and the enemy vessel is "
                  "coming about for another pass."),
    ("r3_long", "Captain's log, stardate four seven six three point nine. We have "
                "entered the Veridian system after a six day crossing at warp seven, "
                "and the ship has settled into the easy rhythm of survey work. The "
                "science teams are mapping the second planet's magnetosphere while "
                "engineering runs a full diagnostic on the port nacelle."),
]


def sidecar_procs() -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", f"--id={GPU_INDEX}",
         "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"],
        capture_output=True, text=True).stdout
    return [ln.strip() for ln in out.splitlines() if "sidecar" in ln]


async def recv(ws, timeout=300.0):
    return await asyncio.wait_for(ws.recv(), timeout=timeout)


async def run_request(ws, request_id, text):
    """Return dict(first_ms, total_ms, pcm, sample_rate, reason)."""
    t0 = time.monotonic()
    await ws.send(json.dumps({"type": "tts", "text": text, "request_id": request_id}))
    first = None
    pcm = b""
    sample_rate = None
    while True:
        frame = await recv(ws)
        if isinstance(frame, bytes):
            if first is None:
                first = time.monotonic() - t0
            pcm += frame
            continue
        msg = json.loads(frame)
        if msg.get("type") == "audio_start" and msg.get("request_id") == request_id:
            sample_rate = msg.get("sample_rate")
        elif msg.get("type") == "audio_end" and msg.get("request_id") == request_id:
            return {"first_ms": (first or 0) * 1000, "total_ms": (time.monotonic() - t0) * 1000,
                    "pcm": pcm, "sample_rate": sample_rate,
                    "reason": msg.get("reason", "complete")}
        elif msg.get("type") == "error":
            raise AssertionError(f"error frame during {request_id}: {msg}")


async def main() -> int:
    print(f"connecting to {URL}")
    ws = await websockets.connect(URL, max_size=None, additional_headers=HEADERS)
    handshake = json.loads(await recv(ws, 30.0))
    assert "session_id" in handshake, handshake
    print(f"session {handshake['session_id']}")

    ok = True
    for request_id, text in REQUESTS:
        try:
            r = await run_request(ws, request_id, text)
        except (AssertionError, websockets.ConnectionClosed) as exc:
            print(f"FAIL {request_id}: {exc}")
            ok = False
            break
        n = len(r["pcm"])
        sr = r["sample_rate"] or 0
        dur = n / 2 / sr if sr else 0.0
        passed = n > 0 and r["reason"] == "complete"
        ok = ok and passed
        print(f"{'PASS' if passed else 'FAIL'} {request_id} chars={len(text)} "
              f"pcm_bytes={n} audio_s={dur:.2f} sample_rate={sr} "
              f"first_audio_ms={r['first_ms']:.0f} total_ms={r['total_ms']:.0f} "
              f"reason={r['reason']}")
        if n:
            path = os.path.join(OUT_DIR, f"breeze_probe_{request_id}.wav")
            with wave.open(path, "wb") as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(sr or 24000)
                f.writeframes(r["pcm"])
            print(f"WROTE {path}")

    procs = sidecar_procs()
    for line in procs:
        print(f"VRAM {line}")
    if not procs:
        print("FAIL no TTS sidecar process alive on the GPU after the last request")
        ok = False

    await ws.close()
    print("=== BREEZE PROBE", "PASS" if ok else "FAIL", "===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
