"""One-shot listen-WAV regeneration for the pace/quality verdict.

MANUAL - not collected by pytest. Synthesizes the listen passage under voxcpm2
and writes the WAV, no STT feed, so the long-prompt prefill graph has room.
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
OUT_WAV = os.environ.get("VX_OUT", "/tmp/voxcpm2_listen.wav")

LISTEN_TEXT = (
    "Captain's log, stardate four seven six three point nine. We have entered the "
    "Veridian system after a six day crossing at warp seven, and the ship has settled "
    "into the easy rhythm of survey work. The science teams are mapping the second "
    "planet's magnetosphere while engineering runs a full diagnostic on the port "
    "nacelle. For now the bridge is quiet, the stars are steady, and the ship feels "
    "very much like home."
)


async def main() -> int:
    ws = await websockets.connect(URL, max_size=None, additional_headers=HEADERS)
    await asyncio.wait_for(ws.recv(), timeout=30.0)
    t0 = time.monotonic()
    await ws.send(json.dumps({"type": "tts", "text": LISTEN_TEXT, "request_id": "listen"}))
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
        if msg.get("type") == "audio_end" and msg.get("request_id") == "listen":
            break
        if msg.get("type") == "error":
            raise AssertionError(msg)
    duration = len(pcm) / 2 / 48000.0
    with wave.open(OUT_WAV, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(48000)
        f.writeframes(pcm)
    print(f"first_audio_ms={first*1000:.0f}")
    print(f"audio_s={duration:.2f} wall_s={time.monotonic()-t0:.2f} rtf={(time.monotonic()-t0)/duration:.2f}")
    print(f"WROTE {OUT_WAV} ({len(pcm)} bytes)")
    await ws.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
