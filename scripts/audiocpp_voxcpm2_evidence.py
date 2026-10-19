"""Live evidence measurements for audiocpp voxcpm2.

MANUAL live script - never collected by pytest (scripts/ is outside tests/).
Needs the real server running with audiocpp STT + audiocpp voxcpm2 TTS, e.g.:

    VM_STT_ENGINE=audiocpp VM_TTS_ENGINE=audiocpp \
    VM_AUDIOCPP_TTS_MODEL=/home/user/audiocpp-models/VoxCPM2-GGUF/voxcpm2-q8_0.gguf \
    VM_AUDIOCPP_TTS_MODEL_TASK=voxcpm2-tts-streaming \
    VM_PORT=8130 python -m voicemanager

Measures and prints, in a paste-ready block:
  * per-session sidecar VRAM on the serving GPU (voxcpm2 TTS sidecar plus the
    STT sidecar spawned by feeding one audio frame), via
    nvidia-smi --query-compute-apps;
  * WS-level first-audio latency for a one-shot tts request (first and warm);
  * WS-level first-audio latency for tts_stream, timed from the chunk that
    completes the first sentence;
  * a natural passage synthesized to a WAV for the pace/quality listen.

Usage: VOICEMANAGER_TOKEN=vmp_... VX_PORT=8130 VX_GPU=3 python scripts/audiocpp_voxcpm2_evidence.py
"""
import asyncio
import json
import os
import subprocess
import sys
import time
import wave

import websockets

PORT = int(os.environ.get("VX_PORT", "8130"))
TOKEN = os.environ["VOICEMANAGER_TOKEN"]
URL = f"ws://localhost:{PORT}/ws/stream"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
GPU_INDEX = os.environ.get("VX_GPU", "3")
OUT_WAV = os.environ.get("VX_OUT", "/tmp/voxcpm2_listen.wav")

TYPICAL = "Shields are holding at sixty percent and the enemy vessel is coming about."

STREAM_S1 = ("The long range sensors have finished their sweep of the outer asteroid "
             "belt and found nothing unusual at all.")
STREAM_TAIL = " Everything else can wait for the morning briefing."

# Natural multi-sentence passage for the pace/quality listen.
LISTEN_TEXT = (
    "Captain's log, stardate four seven six three point nine. We have entered the "
    "Veridian system after a six day crossing at warp seven, and the ship has settled "
    "into the easy rhythm of survey work. The science teams are mapping the second "
    "planet's magnetosphere while engineering runs a full diagnostic on the port "
    "nacelle. For now the bridge is quiet, the stars are steady, and the ship feels "
    "very much like home."
)


def sidecar_vram() -> list[str]:
    """Per-process VRAM for audiocpp sidecars on the serving GPU."""
    out = subprocess.run(
        ["nvidia-smi", f"--id={GPU_INDEX}",
         "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"],
        capture_output=True, text=True).stdout
    return [ln.strip() for ln in out.splitlines() if "sidecar" in ln]


async def recv(ws, timeout=180.0):
    return await asyncio.wait_for(ws.recv(), timeout=timeout)


async def run_request(ws, request_id, text):
    """Send one tts request; return (first_audio_s, total_s, pcm_bytes)."""
    t0 = time.monotonic()
    await ws.send(json.dumps({"type": "tts", "text": text, "request_id": request_id}))
    first = None
    pcm = b""
    while True:
        frame = await recv(ws)
        if isinstance(frame, bytes):
            if first is None:
                first = time.monotonic() - t0
            pcm += frame
            continue
        msg = json.loads(frame)
        if msg.get("type") == "audio_end" and msg.get("request_id") == request_id:
            return first, time.monotonic() - t0, pcm
        if msg.get("type") == "error":
            raise AssertionError(f"error during {request_id}: {msg}")


async def main() -> int:
    print(f"connecting to {URL}")
    ws = await websockets.connect(URL, max_size=None, additional_headers=HEADERS)
    handshake = json.loads(await recv(ws, 30.0))
    assert "session_id" in handshake, handshake

    # Feed one 16 kHz s16le audio frame to make this session spawn its STT
    # sidecar ("an STT sidecar is also loaded"), then give it time to load.
    silence = b"\x00\x00" * 8000  # 0.5 s of silence @ 16 kHz mono s16
    await ws.send(silence)
    print("fed one audio frame; waiting for the STT sidecar to spawn...")
    for _ in range(120):
        lines = sidecar_vram()
        if any("sidecar" in ln for ln in lines):
            break
        await asyncio.sleep(1.0)
    await asyncio.sleep(2.0)

    # One-shot tts: first request (includes residual warm-up) then warm.
    first_lat, _, _ = await run_request(ws, "m1", TYPICAL)
    warm_lat, _, _ = await run_request(ws, "m2", TYPICAL)
    print(f"MEASURE tts_first_audio_first_request_ms={first_lat*1000:.0f}")
    print(f"MEASURE tts_first_audio_warm_ms={warm_lat*1000:.0f}")

    # VRAM with both sidecars of THIS session loaded on the serving GPU.
    for line in sidecar_vram():
        print(f"VRAM {line}")

    # tts_stream: timed from the chunk that completes the first sentence.
    await ws.send(json.dumps({"type": "tts_stream", "text": STREAM_S1,
                              "request_id": "st1"}))
    t0 = time.monotonic()
    await ws.send(json.dumps({"type": "tts_stream", "text": STREAM_TAIL,
                              "request_id": "st1", "done": True}))
    stream_first = None
    while True:
        frame = await recv(ws)
        if isinstance(frame, bytes):
            if stream_first is None:
                stream_first = time.monotonic() - t0
            continue
        msg = json.loads(frame)
        if msg.get("type") == "audio_end" and msg.get("request_id") == "st1":
            break
    print(f"MEASURE tts_stream_first_audio_ms={stream_first*1000:.0f}")

    # Listen WAV for the pace/quality verdict.
    _, lf_total, pcm = await run_request(ws, "lf1", LISTEN_TEXT)
    duration = len(pcm) / 2 / 48000.0
    with wave.open(OUT_WAV, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(48000)
        f.writeframes(pcm)
    print(f"MEASURE listen_audio_s={duration:.2f} wall_s={lf_total:.2f} "
          f"rtf={lf_total/duration:.2f}")
    print(f"WROTE {OUT_WAV} ({len(pcm)} bytes, {duration:.1f} s at 48 kHz) - "
          "play this for the pace/quality verdict")

    await ws.close()
    print("=== VOXCPM2 MEASUREMENTS COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
