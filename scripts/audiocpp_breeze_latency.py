"""Live latency and VRAM measurements for audiocpp breeze.

MANUAL live script - never collected by pytest (scripts/ is outside tests/).
Needs the real server in the switched state (dev sidecar, breeze q8_0,
VM_TTS_ENGINE=audiocpp, audiocpp STT) on the default port. Run it against a
server with NO live session so the first request is a true cold start.

Measures and prints, each line stamped with the wall-clock time:
  * first-audio cold (fresh session, includes sidecar spawn and model load)
    and warm for one-shot tts;
  * first-audio for tts_stream, timed from the chunk that completes the first
    sentence;
  * RTF for one long one-shot request (audio seconds / wall seconds);
  * per-process VRAM on the serving GPU with the STT sidecar co-loaded
    (a 16 kHz speech WAV, BZ_SPEECH_WAV, is fed at real-time pace to spawn it;
    silence would be idle-gated and never reach the sidecar);
  * cancel on a long one-shot request at three points into synthesis:
    cancel -> audio_end(reason=cancelled) ack, PCM bytes after the ack, and
    cancel -> next request first audio. The worst (max) of each is reported.

Usage: VOICEMANAGER_TOKEN=vmp_... BZ_PORT=8000 BZ_GPU=3 BZ_SPEECH_WAV=/tmp/speech16k.wav \
    python scripts/audiocpp_breeze_latency.py
"""
import asyncio
import json
import os
import subprocess
import sys
import time
import wave
from datetime import datetime

import websockets

PORT = int(os.environ.get("BZ_PORT", "8000"))
TOKEN = os.environ["VOICEMANAGER_TOKEN"]
URL = f"ws://localhost:{PORT}/ws/stream"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
GPU_INDEX = os.environ.get("BZ_GPU", "3")
OUT_WAV = os.environ.get("BZ_OUT", "/tmp/breeze_longform.wav")
SPEECH_WAV = os.environ.get("BZ_SPEECH_WAV", "/tmp/speech16k.wav")  # 16 kHz mono s16 speech

TYPICAL = "Shields are holding at sixty percent and the enemy vessel is coming about."

STREAM_S1 = ("The long range sensors have finished their sweep of the outer asteroid "
             "belt and found nothing unusual at all.")
STREAM_TAIL = " Everything else can wait for the morning briefing."

LONG_TEXT = (
    "Captain's log, stardate four seven six three point nine. We have entered the "
    "Veridian system after a six day crossing at warp seven, and the ship has settled "
    "into the easy rhythm of survey work. The science teams are mapping the second "
    "planet's magnetosphere while engineering runs a full diagnostic on the port "
    "nacelle. For now the bridge is quiet, the stars are steady, and the ship feels "
    "very much like home."
)

CANCEL_POINTS_S = [0.5, 2.0, 5.0]


def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def sidecar_vram() -> list[str]:
    out = subprocess.run(
        ["nvidia-smi", f"--id={GPU_INDEX}",
         "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"],
        capture_output=True, text=True).stdout
    return [ln.strip() for ln in out.splitlines() if "sidecar" in ln]


async def recv(ws, timeout=300.0):
    return await asyncio.wait_for(ws.recv(), timeout=timeout)


async def run_request(ws, request_id, text):
    """Send one tts request; return (first_audio_s, total_s, pcm_bytes, sample_rate)."""
    t0 = time.monotonic()
    await ws.send(json.dumps({"type": "tts", "text": text, "request_id": request_id}))
    first = None
    pcm = b""
    sr = None
    while True:
        frame = await recv(ws)
        if isinstance(frame, bytes):
            if first is None:
                first = time.monotonic() - t0
            pcm += frame
            continue
        msg = json.loads(frame)
        if msg.get("type") == "audio_start" and msg.get("request_id") == request_id:
            sr = msg.get("sample_rate")
        elif msg.get("type") == "audio_end" and msg.get("request_id") == request_id:
            return first, time.monotonic() - t0, pcm, sr
        elif msg.get("type") == "error":
            raise AssertionError(f"error during {request_id}: {msg}")


async def cancel_trial(ws, request_id, cancel_after_s):
    """Start LONG_TEXT, cancel after cancel_after_s; return
    (ack_ms, bytes_after_ack, bytes_before_ack, next_first_ms)."""
    await ws.send(json.dumps({"type": "tts", "text": LONG_TEXT, "request_id": request_id}))
    t0 = time.monotonic()
    before = 0
    # Drain until the cancel point, then send tts_cancel.
    while time.monotonic() - t0 < cancel_after_s:
        remaining = cancel_after_s - (time.monotonic() - t0)
        try:
            frame = await asyncio.wait_for(ws.recv(), timeout=max(remaining, 0.001))
        except asyncio.TimeoutError:
            break
        if isinstance(frame, bytes):
            before += len(frame)
        else:
            msg = json.loads(frame)
            if msg.get("type") == "audio_end" and msg.get("request_id") == request_id:
                raise AssertionError(f"{request_id} finished before the cancel point")
    tc = time.monotonic()
    await ws.send(json.dumps({"type": "tts_cancel", "request_id": request_id}))
    ack = None
    while True:
        frame = await recv(ws)
        if isinstance(frame, bytes):
            before += len(frame)
            continue
        msg = json.loads(frame)
        if msg.get("type") == "audio_end" and msg.get("request_id") == request_id:
            ack = time.monotonic() - tc
            assert msg.get("reason") == "cancelled", msg
            break
    # Next request on the same session: any stray bytes before its audio_start
    # belong to the cancelled request.
    nxt = f"{request_id}_next"
    tn = time.monotonic()
    await ws.send(json.dumps({"type": "tts", "text": TYPICAL, "request_id": nxt}))
    after = 0
    started = False
    next_first = None
    while True:
        frame = await recv(ws)
        if isinstance(frame, bytes):
            if not started:
                after += len(frame)
            elif next_first is None:
                next_first = time.monotonic() - tn
            continue
        msg = json.loads(frame)
        if msg.get("type") == "audio_start" and msg.get("request_id") == nxt:
            started = True
        elif msg.get("type") == "audio_end" and msg.get("request_id") == nxt:
            break
    return ack * 1000, after, before, (next_first or 0) * 1000


async def main() -> int:
    print(f"{stamp()} connecting to {URL}")
    ws = await websockets.connect(URL, max_size=None, additional_headers=HEADERS)
    handshake = json.loads(await recv(ws, 30.0))
    assert "session_id" in handshake, handshake

    # Cold: first request on a fresh session (sidecar spawn + model load).
    cold, _, _, sr = await run_request(ws, "c1", TYPICAL)
    print(f"{stamp()} MEASURE tts_first_audio_cold_ms={cold*1000:.0f} sample_rate={sr}")
    warm, _, _, _ = await run_request(ws, "w1", TYPICAL)
    warm2, _, _, _ = await run_request(ws, "w2", TYPICAL)
    print(f"{stamp()} MEASURE tts_first_audio_warm_ms={warm*1000:.0f} "
          f"(second warm {warm2*1000:.0f})")

    # tts_stream: timed from the chunk that completes the first sentence.
    await ws.send(json.dumps({"type": "tts_stream", "text": STREAM_S1, "request_id": "st1"}))
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
    print(f"{stamp()} MEASURE tts_stream_first_audio_ms={stream_first*1000:.0f}")

    # RTF on one long one-shot request; WAV kept for the listen.
    lf_first, lf_total, pcm, sr = await run_request(ws, "lf1", LONG_TEXT)
    duration = len(pcm) / 2 / sr
    with wave.open(OUT_WAV, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(pcm)
    print(f"{stamp()} MEASURE longform_first_audio_ms={lf_first*1000:.0f} "
          f"audio_s={duration:.2f} wall_s={lf_total:.2f} rtf={lf_total/duration:.2f}")
    print(f"{stamp()} WROTE {OUT_WAV} ({len(pcm)} bytes at {sr} Hz)")

    # Co-load the STT sidecar by feeding real speech at real-time pace (the
    # audiocpp STT path idle-gates silence, so a silent frame never spawns it),
    # then read per-process VRAM.
    with wave.open(SPEECH_WAV, "rb") as wf:
        assert wf.getframerate() == 16000 and wf.getnchannels() == 1, SPEECH_WAV
        speech = wf.readframes(wf.getnframes())
    frame_bytes = 16000 * 2 * 32 // 1000
    for i in range(0, len(speech), frame_bytes):
        await ws.send(speech[i:i + frame_bytes])
        await asyncio.sleep(0.032)
    for _ in range(120):
        if len(sidecar_vram()) >= 2:
            break
        await asyncio.sleep(1.0)
    await asyncio.sleep(2.0)
    for line in sidecar_vram():
        print(f"{stamp()} VRAM {line}")

    # Cancel on a long one-shot request at three points; report the worst.
    acks, afters, nexts = [], [], []
    for i, point in enumerate(CANCEL_POINTS_S):
        ack_ms, after, before, next_ms = await cancel_trial(ws, f"cx{i}", point)
        acks.append(ack_ms)
        afters.append(after)
        nexts.append(next_ms)
        print(f"{stamp()} CANCEL at={point:.1f}s ack_ms={ack_ms:.0f} "
              f"bytes_before_ack={before} bytes_after_ack={after} "
              f"next_first_audio_ms={next_ms:.0f}")
    print(f"{stamp()} MEASURE cancel_ack_worst_ms={max(acks):.0f} "
          f"cancel_bytes_after_ack_worst={max(afters)} "
          f"cancel_to_next_first_audio_worst_ms={max(nexts):.0f}")

    await ws.close()
    print(f"{stamp()} === BREEZE MEASUREMENTS COMPLETE ===")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
