"""Live evidence measurements for the audiocpp TTS backend.

MANUAL live script - never collected by pytest (scripts/ is outside tests/).
Needs the real server running with BOTH engines on audiocpp, e.g.:

    VM_STT_ENGINE=audiocpp VM_TTS_ENGINE=audiocpp \
    VM_AUDIOCPP_TTS_MODEL=/home/user/audiocpp-models/DotTTS-SOAR-GGUF/dots-tts-soar-q8_0.gguf \
    VM_PORT=8123 python -m voicemanager

Measures and prints, in a paste-ready block:
  * per-session sidecar VRAM on the serving GPU (per-process, via
    nvidia-smi --query-compute-apps) with the STT sidecar loaded alongside;
  * WS-level first-audio latency for a one-shot tts request (first and warm);
  * WS-level first-audio latency for tts_stream, timed from the chunk that
    completes the first sentence;
  * cancel-to-ack and cancel-to-next-request-audio under the shipped
    sink-throw mechanism;
  * a long-form multi-sentence synthesis crossing multiple 200-char chunks,
    written to tts_evidence/tts_evidence_longform.wav for a seam ear test, with its
    audio duration and wall-clock (RTF).

Usage: VOICEMANAGER_TOKEN=vmp_... T14_PORT=8123 python scripts/audiocpp_tts_evidence.py
"""
import asyncio
import json
import os
import subprocess
import sys
import time
import wave

import websockets

PORT = int(os.environ.get("T14_PORT", "8123"))
TOKEN = os.environ["VOICEMANAGER_TOKEN"]
URL = f"ws://localhost:{PORT}/ws/stream"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
GPU_INDEX = os.environ.get("T14_GPU", "3")
# Listen clips stay in the ignored tts_evidence/ dir, never in the repo root.
OUT_WAV = os.path.join("tts_evidence", "tts_evidence_longform.wav")

TYPICAL = "Shields are holding at sixty percent and the enemy vessel is coming about."

STREAM_S1 = ("The long range sensors have finished their sweep of the outer asteroid "
             "belt and found nothing unusual at all.")
STREAM_TAIL = " Everything else can wait for the morning briefing."

CANCEL_TEXT = (
    "This request is deliberately long so that the cancellation arrives squarely in "
    "the middle of its generation. The warp core hums steadily as the ship glides "
    "through the quiet dark between systems, and every console on the bridge glows "
    "with unhurried telemetry while the crew goes about the ordinary business of a "
    "long patrol far from home."
)

# Long-form seam test: multiple sentences, ~1200 chars, crossing several of the
# sidecar's 200-char internal chunks (text_chunk_size=200) - the one unverified
# quality risk from the brainstorm.
LONGFORM = (
    "Captain's log, stardate four seven six three point nine. We have entered the "
    "Veridian system after a six day crossing at warp seven, and the ship has "
    "settled into the easy rhythm of survey work. The science teams are mapping "
    "the second planet's magnetosphere while engineering runs a full diagnostic "
    "on the port nacelle, which has been reading two percent warm since the "
    "nebula passage. Counselor Reyes reports that morale is high, helped in no "
    "small part by the hydroponics bay's first successful tomato harvest. "
    "Tomorrow we make orbit around the third planet, a cold world of shallow "
    "seas and long grey beaches, where the probes have detected geometric "
    "formations that may be the remains of an ancient shoreline settlement. If "
    "the away team confirms artificial origin, this system will be referred to "
    "the archaeology council for a full expedition. For now the bridge is quiet, "
    "the stars are steady, and the ship feels very much like home."
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
    assert json.loads(await recv(ws, 30.0))["type"] == "ready"

    # Give the per-session sidecars a moment to spawn, then a first request
    # (its latency includes any residual warm-up wait).
    await asyncio.sleep(2.0)
    first_lat, _, _ = await run_request(ws, "m1", TYPICAL)
    warm_lat, _, _ = await run_request(ws, "m2", TYPICAL)
    print(f"MEASURE tts_first_audio_first_request_ms={first_lat*1000:.0f}")
    print(f"MEASURE tts_first_audio_warm_ms={warm_lat*1000:.0f}")

    # VRAM with both sidecars of THIS session loaded (plus the server itself).
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

    # Cancel timings under the shipped sink-throw mechanism.
    await ws.send(json.dumps({"type": "tts", "text": CANCEL_TEXT,
                              "request_id": "cx1"}))
    saw_audio = False
    while not saw_audio:
        frame = await recv(ws)
        saw_audio = isinstance(frame, bytes)
    await asyncio.sleep(1.0)
    t_cancel = time.monotonic()
    await ws.send(json.dumps({"type": "tts_cancel", "request_id": "cx1"}))
    while True:
        frame = await recv(ws, 10.0)
        if isinstance(frame, bytes):
            continue
        msg = json.loads(frame)
        if msg.get("type") == "audio_end" and msg.get("request_id") == "cx1":
            assert msg.get("reason") == "cancelled", msg
            break
    cancel_ack = time.monotonic() - t_cancel
    next_first, _, _ = await run_request(ws, "cx2", TYPICAL)
    print(f"MEASURE cancel_to_silence_ms={cancel_ack*1000:.0f}")
    print(f"MEASURE cancel_to_next_request_audio_ms={next_first*1000:.0f}")

    # Long-form multi-chunk synthesis for the seam ear test.
    lf_first, lf_total, pcm = await run_request(ws, "lf1", LONGFORM)
    duration = len(pcm) / 2 / 48000.0
    os.makedirs(os.path.dirname(OUT_WAV), exist_ok=True)
    with wave.open(OUT_WAV, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(48000)
        f.writeframes(pcm)
    print(f"MEASURE longform_first_audio_ms={lf_first*1000:.0f}")
    print(f"MEASURE longform_audio_s={duration:.2f} wall_s={lf_total:.2f} "
          f"rtf={lf_total/duration:.2f}")
    print(f"WROTE {OUT_WAV} ({len(pcm)} bytes, {duration:.1f} s at 48 kHz) - "
          "play this for the seam verdict")

    await ws.close()
    print("=== AUDIOCPP TTS MEASUREMENTS COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
