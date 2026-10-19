"""Synthesize the same passage in each shipped voice under voxcpm2 for a listen test.

MANUAL - not collected by pytest. One tts request per voice, written to
/tmp/voxcpm2_<voice>.wav to A/B the generated voices.
"""
import asyncio
import json
import os
import sys
import time
import wave

import websockets

URL = f"ws://localhost:{os.environ.get('VX_PORT', '8130')}/ws/stream"
HEADERS = {"Authorization": f"Bearer {os.environ['VOICEMANAGER_TOKEN']}"}

# Natural multi-sentence passage, same for every voice.
TEXT = (
    "The long range sensors have finished their sweep of the outer asteroid belt "
    "and found nothing unusual. Incoming transmission from Starfleet Command on "
    "priority channel. Shields are holding at sixty percent and the enemy vessel "
    "is coming about."
)

VOICES = ["designed_female_s101", "designed_male_s202", "lcars_demo"]


async def synth(ws, voice: str, request_id: str) -> tuple[float, bytes]:
    t0 = time.monotonic()
    await ws.send(json.dumps({"type": "tts", "text": TEXT, "request_id": request_id,
                              "voice": voice}))
    first = None
    pcm = b""
    while True:
        frame = await asyncio.wait_for(ws.recv(), timeout=180.0)
        if isinstance(frame, bytes):
            if first is None:
                first = time.monotonic() - t0
            pcm += frame
            continue
        msg = json.loads(frame)
        if msg.get("type") == "audio_end" and msg.get("request_id") == request_id:
            return first, pcm
        if msg.get("type") == "error":
            raise AssertionError(msg)


async def main() -> int:
    ws = await websockets.connect(URL, max_size=None, additional_headers=HEADERS)
    await asyncio.wait_for(ws.recv(), timeout=30.0)
    # Warm the sidecar with the default voice so per-voice timing is not
    # dominated by the first-request sidecar spawn.
    await synth(ws, VOICES[0], "warmup")
    for i, voice in enumerate(VOICES):
        first, pcm = await synth(ws, voice, f"v{i}")
        out = f"/tmp/voxcpm2_{voice}.wav"
        duration = len(pcm) / 2 / 48000.0
        with wave.open(out, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(48000)
            f.writeframes(pcm)
        print(f"{voice}: first_audio_ms={first*1000:.0f} audio_s={duration:.2f} "
              f"-> {out} ({len(pcm)} bytes)")
    await ws.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
