"""Smoke test for VibeVoice TTS engine via the VoiceManager client.

Requires a running VoiceManager server with VM_TTS_ENGINE=vibevoice.
Reads the bearer token from VOICEMANAGER_TOKEN, as test_tts.py and test_stream_tts.py do.

Usage::

    python tests/test_vibevoice_tts.py                # run all tests with default voice
    python tests/test_vibevoice_tts.py --voice Emma    # use a specific voice preset
    python tests/test_vibevoice_tts.py --all-voices    # test every available voice
    python tests/test_vibevoice_tts.py --live          # type line-by-line, audio streams back live
    python tests/test_vibevoice_tts.py --play          # play audio after each test
"""
import asyncio
import os
import shutil
import subprocess
import sys
import threading

from dotenv import load_dotenv
from voicemanager.client import VoiceManagerClient

load_dotenv()

TEST_TEXT = (
    "The stars are bright tonight. "
    "I can see the constellations clearly from here. "
    "It reminds me of when we used to go camping."
)
CHUNK_SIZE = 8  # characters per chunk — simulates token-sized fragments


def play_wav(path: str) -> None:
    """Play a WAV file using mplayer (if available)."""
    mplayer = shutil.which("mplayer")
    if not mplayer:
        print("  (mplayer not found — skipping playback)")
        return
    subprocess.run([mplayer, "-really-quiet", path], check=False)


async def fake_llm_stream(text: str | None = None, chunk_size: int = CHUNK_SIZE):
    """Async generator that yields small text chunks like an LLM token stream."""
    src = text or TEST_TEXT
    for i in range(0, len(src), chunk_size):
        yield src[i : i + chunk_size]
        await asyncio.sleep(0.02)


async def test_handshake(client: VoiceManagerClient) -> None:
    """Verify handshake reports VibeVoice engine and voice list."""
    print("\n--- handshake ---")
    print(f"  tts_engine: {client.tts_engine}")
    print(f"  available_voices: {client.available_voices}")
    assert client.tts_engine == "vibevoice", (
        f"Expected tts_engine='vibevoice', got '{client.tts_engine}'. "
        "Is the server running with VM_TTS_ENGINE=vibevoice?"
    )
    assert len(client.available_voices) > 0, "No available voices reported"
    print("  OK")


async def test_single_shot(client: VoiceManagerClient, voice: str | None = None, play: bool = False) -> None:
    """Test synthesize_tts() with a voice preset."""
    label = voice or "default"
    print(f"\n--- synthesize_tts(voice={label}) ---")
    pcm, rate = await client.synthesize_tts("Hello, this is a test of the voice manager.", voice=voice)
    assert rate == 24000, f"Expected 24000 Hz for VibeVoice, got {rate}"
    assert len(pcm) > 0, "No audio received"
    out_path = f"tests/vibevoice_tts_{label}.wav"
    client.save_tts_to_file(pcm, out_path, rate)
    dur = len(pcm) / (rate * 2)
    print(f"  Saved {out_path} ({len(pcm)} bytes, {rate} Hz, {dur:.1f}s)")
    if play:
        play_wav(out_path)


async def test_stream_tts(client: VoiceManagerClient, voice: str | None = None, play: bool = False) -> None:
    """Test stream_tts() — collect all streaming audio then save."""
    label = voice or "default"
    print(f"\n--- stream_tts(voice={label}) ---")
    pcm, rate = await client.stream_tts(fake_llm_stream(), voice=voice)
    assert rate == 24000, f"Expected 24000 Hz for VibeVoice, got {rate}"
    assert len(pcm) > 0, "No audio received"
    out_path = f"tests/vibevoice_stream_{label}.wav"
    client.save_tts_to_file(pcm, out_path, rate)
    dur = len(pcm) / (rate * 2)
    print(f"  Saved {out_path} ({len(pcm)} bytes, {rate} Hz, {dur:.1f}s)")
    print(f"  last_tts_sample_rate: {client.last_tts_sample_rate}")
    if play:
        play_wav(out_path)


async def test_stream_tts_live(client: VoiceManagerClient, voice: str | None = None, play: bool = False) -> None:
    """Test stream_tts_live() — yield audio chunks in real-time."""
    label = voice or "default"
    print(f"\n--- stream_tts_live(voice={label}) ---")
    audio_buf = bytearray()
    chunk_count = 0
    async for audio_chunk in client.stream_tts_live(fake_llm_stream(), voice=voice):
        audio_buf.extend(audio_chunk)
        chunk_count += 1
        if chunk_count <= 3:
            print(f"  chunk {chunk_count}: {len(audio_chunk)} bytes")
        elif chunk_count == 4:
            print("  ...")

    rate = client.last_tts_sample_rate
    assert rate == 24000, f"Expected 24000 Hz for VibeVoice, got {rate}"
    out_path = f"tests/vibevoice_live_{label}.wav"
    client.save_tts_to_file(bytes(audio_buf), out_path, rate)
    dur = len(audio_buf) / (rate * 2)
    print(f"  Saved {out_path} ({len(audio_buf)} bytes, {rate} Hz, {dur:.1f}s, {chunk_count} chunks)")
    if play:
        play_wav(out_path)


async def test_upload_blocked(client: VoiceManagerClient) -> None:
    """Verify /voice/upload returns 400 when VibeVoice is active."""
    print("\n--- upload_voice (expect 400) ---")
    # Minimal WAV header — doesn't need to be real audio for this test
    fake_wav = b"RIFF" + b"\x00" * 40
    try:
        await client.upload_voice(fake_wav, "test.wav")
        print("  FAIL: upload_voice did not raise")
        assert False, "Expected VoiceManagerError for VibeVoice upload guard"
    except Exception as exc:
        if "not supported" in str(exc).lower() or "400" in str(exc):
            print(f"  OK: {exc}")
        else:
            print(f"  UNEXPECTED ERROR: {exc}")
            raise


async def test_all_voices(client: VoiceManagerClient, play: bool = False) -> None:
    """Test synthesize_tts with each available voice preset."""
    print(f"\n--- all voices: {client.available_voices} ---")
    for voice in client.available_voices:
        pcm, rate = await client.synthesize_tts(
            f"Hello, my name is {voice}.", voice=voice
        )
        out_path = f"tests/vibevoice_voice_{voice.lower()}.wav"
        client.save_tts_to_file(pcm, out_path, rate)
        dur = len(pcm) / (rate * 2)
        print(f"  {voice}: {len(pcm)} bytes, {rate} Hz, {dur:.1f}s -> {out_path}")
        if play:
            play_wav(out_path)


async def test_live(client: VoiceManagerClient, voice: str | None = None) -> None:
    """Type line-by-line, each line streams to TTS immediately."""
    label = voice or "default"
    print(f"\n--- live mode (voice={label}) ---")
    print("Type text and press Enter. Blank line or Ctrl+D to finish.\n")

    text_queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _stdin_reader():
        while True:
            try:
                line = input("> ")
            except EOFError:
                loop.call_soon_threadsafe(text_queue.put_nowait, None)
                return
            if not line:
                loop.call_soon_threadsafe(text_queue.put_nowait, None)
                return
            loop.call_soon_threadsafe(text_queue.put_nowait, line + " ")

    async def _live_text_stream():
        while True:
            chunk = await text_queue.get()
            if chunk is None:
                return
            yield chunk

    reader_thread = threading.Thread(target=_stdin_reader, daemon=True)
    reader_thread.start()

    audio_buf = bytearray()
    chunk_count = 0
    async for audio_chunk in client.stream_tts_live(_live_text_stream(), voice=voice):
        audio_buf.extend(audio_chunk)
        chunk_count += 1
        dur_ms = len(audio_chunk) / (client.last_tts_sample_rate * 2) * 1000
        print(f"  [audio] chunk {chunk_count}: {len(audio_chunk)} bytes (~{dur_ms:.0f}ms)")

    reader_thread.join(timeout=1.0)

    if audio_buf:
        out_path = f"tests/vibevoice_live_typing_{label}.wav"
        rate = client.last_tts_sample_rate
        client.save_tts_to_file(bytes(audio_buf), out_path, rate)
        total_dur = len(audio_buf) / (rate * 2)
        print(f"\nSaved {out_path} ({len(audio_buf)} bytes, {rate} Hz, {total_dur:.1f}s, {chunk_count} chunks)")
    else:
        print("\nNo audio received.")


async def main():
    args = sys.argv[1:]
    voice = None
    if "--voice" in args:
        idx = args.index("--voice")
        voice = args[idx + 1] if idx + 1 < len(args) else None
    all_voices = "--all-voices" in args
    live = "--live" in args or "-l" in args
    play = "--play" in args or "-p" in args

    token = os.environ["VOICEMANAGER_TOKEN"]
    async with VoiceManagerClient(token=token) as client:
        info = await client.info()
        print(f"Server info: tts_engine={info.get('tts_engine')}, "
              f"tts_model={info.get('tts_model')}, "
              f"available_voices={info.get('available_voices')}")

        await test_handshake(client)

        if live:
            await test_live(client, voice=voice)
        elif all_voices:
            await test_all_voices(client, play=play)
        else:
            await test_single_shot(client, voice=voice, play=play)
            await test_stream_tts(client, voice=voice, play=play)
            await test_stream_tts_live(client, voice=voice, play=play)
            await test_upload_blocked(client)
            print("\nAll VibeVoice TTS tests passed.")


asyncio.run(main())
