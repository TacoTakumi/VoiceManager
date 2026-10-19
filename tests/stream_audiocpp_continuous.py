"""MANUAL live repro: audio.cpp endpointing under a CONTINUOUSLY-streaming client.

Not an automated test (no test_ prefix, pytest never collects it). Run by hand against a
LIVE server with VM_STT_ENGINE=audiocpp, mimicking how a real mic client (HermesConsole)
streams: frames flow nonstop while listening -- speech, then silence AS FRAMES of quiet
audio -- so the frame-gap trigger never sees a gap. Before the acoustic trigger this exact shape
produced NOTHING until the client stopped transmitting (the live 2026-08-04 failure);
with the acoustic trigger the final must arrive while silence frames are still flowing.

Usage:
    ffmpeg -i speech.mp3 -ar 16000 -ac 1 -f s16le /tmp/clip.pcm
    ./venv/bin/python tests/stream_audiocpp_continuous.py <token> /tmp/clip.pcm [port]

Expected (fixed server): the is_final=True transcription lands DURING the 5 s of silence
frames (~VM_AUDIOCPP_SILENCE_MS + decode after speech ends), and nothing arrives after
the sender stops (the trailing silence-only flush decodes to empty and emits no frame).
"""
import asyncio
import json
import sys
import time

import websockets

TOKEN = sys.argv[1]
CLIP = sys.argv[2]
PORT = sys.argv[3] if len(sys.argv) > 3 else "8000"
URL = f"ws://localhost:{PORT}/ws/stream"

CHUNK_MS = 100
CHUNK_BYTES = 16000 * 2 * CHUNK_MS // 1000  # 3200 bytes per 100 ms frame
SILENCE_S = 5.0
POST_STOP_WAIT_S = 6.0

t0 = time.monotonic()


def log(tag, text):
    print(f"[{time.monotonic() - t0:7.2f}s] {tag}: {text}", flush=True)


async def reader(ws, events):
    try:
        async for msg in ws:
            if isinstance(msg, bytes):
                log("RECV", f"binary frame ({len(msg)} bytes)")
                continue
            obj = json.loads(msg)
            events.append((time.monotonic() - t0, obj))
            if obj.get("type") == "transcription":
                log("RECV", f"transcription is_final={obj.get('is_final')} "
                    f"seg={obj.get('segment_index')} text={obj.get('text')!r}")
            else:
                log("RECV", f"{obj.get('type')}: {json.dumps(obj)[:160]}")
    except websockets.ConnectionClosed as exc:
        log("CLOSE", f"code={exc.code} reason={exc.reason!r}")


async def main():
    pcm = open(CLIP, "rb").read()
    events = []
    headers = {"Authorization": f"Bearer {TOKEN}"}
    async with websockets.connect(URL, max_size=None, additional_headers=headers) as ws:
        rtask = asyncio.create_task(reader(ws, events))

        speech_chunks = [pcm[i:i + CHUNK_BYTES] for i in range(0, len(pcm), CHUNK_BYTES)]
        n_silence = int(SILENCE_S * 1000 / CHUNK_MS)
        silence = b"\x00" * CHUNK_BYTES

        log("SEND", f"streaming {len(speech_chunks)} speech frames "
            f"({len(pcm)/32000:.1f}s) at real-time pace")
        for chunk in speech_chunks:
            await ws.send(chunk)
            await asyncio.sleep(CHUNK_MS / 1000)

        log("SEND", f"speech done; now {SILENCE_S:.0f}s of CONTINUOUS silence frames "
            "(no gap -- HermesConsole style)")
        for _ in range(n_silence):
            await ws.send(silence)
            await asyncio.sleep(CHUNK_MS / 1000)
        mark_silence_done = time.monotonic() - t0

        log("STOP", f"no more frames; keeping socket open {POST_STOP_WAIT_S:.0f}s")
        await asyncio.sleep(POST_STOP_WAIT_S)

        rtask.cancel()

    print("\n===== VERDICT =====")
    tx = [e for e in events if e[1].get("type") == "transcription"]
    during = [e for e in tx if e[0] <= mark_silence_done]
    after = [e for e in tx if e[0] > mark_silence_done]
    print(f"transcriptions while frames were flowing (speech + {SILENCE_S:.0f}s "
          f"silence frames): {len(during)}  <- must be >=2 (partial + final) on a fixed server")
    print(f"transcriptions after frames STOPPED: {len(after)}  <- must be 0 on a fixed server")
    for ts, obj in after:
        print(f"  +{ts - mark_silence_done:.2f}s after stop: is_final={obj.get('is_final')} "
              f"text={obj.get('text')!r}")


asyncio.run(main())
