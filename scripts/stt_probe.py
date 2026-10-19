#!/usr/bin/env python3
"""Real-time STT probe — streams a WAV at mic pace and prints interim/final
transcripts with timestamps.

Mirrors the TVAgent MicPipeline: 32 ms frames sent at real-time pace, then a
few seconds of trailing silence so the server finalizes (qwen via its silence
detector on incoming frames; nemotron via its acoustic end-of-utterance
endpointer). Use it to verify streaming STT accuracy after a fix:

    VOICEMANAGER_TOKEN=vmp_... venv/bin/python scripts/stt_probe.py /tmp/probe.wav

Pass criteria: finals are NOT emitted while speech is still streaming, boundary
tokens aren't duplicated/garbled, and the tail of each utterance is complete.
"""
import asyncio
import os
import sys
import time
import wave

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from voicemanager.client import VoiceManagerClient  # noqa: E402

SR = 16000
FRAME_MS = 32
FRAME_BYTES = int(SR * FRAME_MS / 1000) * 2  # 1024 bytes = 512 samples * 2
TRAIL_SILENCE_S = 4.0


async def main(path: str) -> None:
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == SR and wf.getnchannels() == 1 and wf.getsampwidth() == 2, \
            "probe expects 16 kHz mono 16-bit WAV"
        pcm = wf.readframes(wf.getnframes())

    data = pcm + (b"\x00\x00" * int(SR * TRAIL_SILENCE_S))
    frames = [data[i:i + FRAME_BYTES] for i in range(0, len(data), FRAME_BYTES)]
    speech_s = len(pcm) / 2 / SR
    print(f"file={path} speech={speech_s:.2f}s + {TRAIL_SILENCE_S:.1f}s trailing silence, "
          f"{len(frames)} frames @ {FRAME_MS}ms")

    token = os.environ["VOICEMANAGER_TOKEN"]
    async with VoiceManagerClient(host="localhost", port=8255, token=token) as client:
        print(f"connected session={client.session_id} stt_engine={client.stt_engine}")
        t0 = time.monotonic()

        async def gen():
            for fr in frames:
                yield fr
                await asyncio.sleep(FRAME_MS / 1000)

        async for msg in client.stream_stt_live(gen()):
            t = time.monotonic() - t0
            kind = "FINAL  " if msg.get("is_final") else "interim"
            print(f"[{t:6.2f}s] {kind} seg={msg.get('segment_index')} text={msg.get('text')!r}")

        print(f"[{time.monotonic() - t0:6.2f}s] --- audio exhausted ---")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/probe.wav"))
