"""Smoke test for streaming TTS — stream_tts() and stream_tts_live().

Requires a running VoiceManager server. Reads the bearer token from
VOICEMANAGER_TOKEN, as test_tts.py and test_stt.py do.

Usage::

    python tests/test_stream_tts.py                # canned test text
    python tests/test_stream_tts.py --interactive  # paste a block of text
    python tests/test_stream_tts.py --live         # type line-by-line, audio streams back live
"""
import asyncio
import os
import sys
import threading

from dotenv import load_dotenv
from voicemanager.client import VoiceManagerClient

load_dotenv()

# Multi-sentence text split into small token-like chunks to simulate LLM output.
TEST_TEXT = (
    "The stars are bright tonight. "
    "I can see the constellations clearly from here. "
    "It reminds me of when we used to go camping."
)
CHUNK_SIZE = 8  # characters per chunk — simulates token-sized fragments


async def fake_llm_stream(text: str | None = None, chunk_size: int = CHUNK_SIZE):
    """Async generator that yields small text chunks like an LLM token stream."""
    src = text or TEST_TEXT
    for i in range(0, len(src), chunk_size):
        yield src[i : i + chunk_size]
        await asyncio.sleep(0.02)  # simulate inter-token latency


async def test_stream_tts(client: VoiceManagerClient) -> None:
    """Test stream_tts() — collect all audio then save."""
    print("\n--- stream_tts() ---")
    pcm, rate = await client.stream_tts(fake_llm_stream())
    out_path = "tests/stream_tts_output.wav"
    client.save_tts_to_file(pcm, out_path, rate)
    print(f"Saved {out_path} ({len(pcm)} bytes, {rate} Hz)")
    print(f"last_tts_sample_rate: {client.last_tts_sample_rate}")


async def test_stream_tts_live(client: VoiceManagerClient) -> None:
    """Test stream_tts_live() — yield audio chunks in real-time."""
    print("\n--- stream_tts_live() ---")
    audio_buf = bytearray()
    chunk_count = 0
    async for audio_chunk in client.stream_tts_live(fake_llm_stream()):
        audio_buf.extend(audio_chunk)
        chunk_count += 1
        if chunk_count <= 3:
            print(f"  chunk {chunk_count}: {len(audio_chunk)} bytes")
        elif chunk_count == 4:
            print("  ...")

    out_path = "tests/stream_tts_live_output.wav"
    rate = client.last_tts_sample_rate
    client.save_tts_to_file(bytes(audio_buf), out_path, rate)
    print(f"Saved {out_path} ({len(audio_buf)} bytes, {rate} Hz, {chunk_count} chunks)")


async def test_interactive(client: VoiceManagerClient) -> None:
    """Paste text, stream it gradually through TTS, save the result."""
    print("\n--- interactive mode ---")
    print("Paste or type text, then press Enter (blank line to finish):\n")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line and lines:
            break
        lines.append(line)

    text = " ".join(lines).strip()
    if not text:
        print("No text entered.")
        return

    print(f"\nStreaming {len(text)} chars in ~{CHUNK_SIZE}-char chunks...")
    audio_buf = bytearray()
    chunk_count = 0
    async for audio_chunk in client.stream_tts_live(fake_llm_stream(text)):
        audio_buf.extend(audio_chunk)
        chunk_count += 1

    out_path = "tests/stream_tts_interactive_output.wav"
    rate = client.last_tts_sample_rate
    client.save_tts_to_file(bytes(audio_buf), out_path, rate)
    print(f"Saved {out_path} ({len(audio_buf)} bytes, {rate} Hz, {chunk_count} audio chunks)")


async def test_live(client: VoiceManagerClient) -> None:
    """Type line-by-line, each line streams to TTS immediately, audio arrives live."""
    print("\n--- live mode ---")
    print("Type text and press Enter to send each line to TTS.")
    print("Audio streams back as you go. Blank line or Ctrl+D to finish.\n")

    text_queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _stdin_reader():
        """Read lines from stdin in a thread, push them into the async queue."""
        while True:
            try:
                line = input("> ")
            except EOFError:
                loop.call_soon_threadsafe(text_queue.put_nowait, None)
                return
            if not line:
                loop.call_soon_threadsafe(text_queue.put_nowait, None)
                return
            # Send the line plus a space so sentences join naturally
            loop.call_soon_threadsafe(text_queue.put_nowait, line + " ")

    async def _live_text_stream():
        """Async generator that yields lines as the user types them."""
        while True:
            chunk = await text_queue.get()
            if chunk is None:
                return
            yield chunk

    # Start stdin reader in background thread
    reader_thread = threading.Thread(target=_stdin_reader, daemon=True)
    reader_thread.start()

    audio_buf = bytearray()
    chunk_count = 0
    async for audio_chunk in client.stream_tts_live(_live_text_stream()):
        audio_buf.extend(audio_chunk)
        chunk_count += 1
        # Show audio arriving while user may still be typing
        dur_ms = len(audio_chunk) / (client.last_tts_sample_rate * 2) * 1000
        print(f"  [audio] chunk {chunk_count}: {len(audio_chunk)} bytes (~{dur_ms:.0f}ms)")

    reader_thread.join(timeout=1.0)

    if audio_buf:
        out_path = "tests/stream_tts_live_typing_output.wav"
        rate = client.last_tts_sample_rate
        client.save_tts_to_file(bytes(audio_buf), out_path, rate)
        total_dur = len(audio_buf) / (rate * 2)
        print(f"\nSaved {out_path} ({len(audio_buf)} bytes, {rate} Hz, {total_dur:.1f}s, {chunk_count} chunks)")
    else:
        print("\nNo audio received.")


async def main():
    args = sys.argv[1:]
    interactive = "--interactive" in args or "-i" in args
    live = "--live" in args or "-l" in args
    token = os.environ["VOICEMANAGER_TOKEN"]
    async with VoiceManagerClient(token=token) as client:
        print(await client.health())
        if live:
            await test_live(client)
        elif interactive:
            await test_interactive(client)
        else:
            await test_stream_tts(client)
            await test_stream_tts_live(client)
            print("\nAll streaming TTS tests passed.")


asyncio.run(main())
