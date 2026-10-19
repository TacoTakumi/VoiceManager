"""Live microphone STT test — speak into the mic and see dictation in real time.

Usage:
    python tests/test_mic_stt.py           # STT only
    python tests/test_mic_stt.py --tts     # STT + echo back finals via TTS
    python tests/test_mic_stt.py --verbose # STT with endpoint-aware frame log
    python tests/test_mic_stt.py --clone   # Record one sentence, clone voice, echo back, exit
    python tests/test_mic_stt.py --clone-file voice.mp3   # Clone voice from file, then type text to hear it
    python tests/test_mic_stt.py --clone-file voice.mp3 --text "Hello world"  # One-shot clone + speak

--verbose surfaces the per-frame `endpoint`, `is_final`, and `segment_index`
fields so the speculative endpoint→confirm and endpoint→reopen→stitch cycles on
the nemotron STT path are observable during a live run. Each frame
is timestamped (seconds since first frame) so perceived latency of the confirm/
stitch beats can be eyeballed.

Press Ctrl+C to stop.
"""
import argparse
import asyncio
import io
import os
import sys
import time
import wave

import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from voicemanager.client import VoiceManagerClient

load_dotenv()

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
BLOCKSIZE = 1024  # frames per callback (~64ms at 16kHz)


def _classify_frame(item: dict) -> str:
    """Label a transcription frame by its speculative-endpoint state.

    - ENDPOINT: provisional end-of-utterance (is_final=False, endpoint=True) —
      the held fragment awaiting confirm-or-reopen.
    - FINAL:    immutable confirmed final (is_final=True), whether a plain commit
      or the stitched turn after a reopen.
    - partial:  ordinary interim.
    """
    is_final = item.get("is_final", True)
    endpoint = item.get("endpoint", False)
    if is_final:
        return "FINAL   "
    if endpoint:
        return "ENDPOINT"
    return "partial "


def _format_frame(item: dict, t0: float) -> str:
    """Render one frame as a timestamped, endpoint-aware event line.

    ``t0`` is the monotonic time of the first frame; the leading column is
    seconds since then so the confirm/stitch latency is readable off the log.
    """
    elapsed = time.monotonic() - t0
    state = _classify_frame(item)
    seg = item.get("segment_index", "?")
    text = item.get("text", "")
    return f"[{elapsed:7.2f}s] seg={seg} {state} {text!r}"


async def _play_tts(client, text: str):
    """Request TTS for text and play back through speakers."""
    try:
        pcm_data, sample_rate = await client.synthesize_tts(text)
        if not pcm_data:
            return
        audio = np.frombuffer(pcm_data, dtype=np.int16)
        # Run blocking playback in a thread so the event loop stays alive
        await asyncio.to_thread(_play_blocking, audio, sample_rate)
    except Exception as e:
        print(f"  [TTS error: {e}]", file=sys.stderr)


def _play_blocking(audio: np.ndarray, sample_rate: int):
    """Play audio synchronously (called from thread)."""
    audio_f32 = audio.astype(np.float32) / 32768.0
    sd.play(audio_f32, samplerate=sample_rate)
    sd.wait()


def _chunks_to_wav(chunks: list[bytes], sample_rate: int, channels: int) -> bytes:
    """Pack raw int16 PCM chunks into a WAV file in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sample_rate)
        for chunk in chunks:
            wf.writeframes(chunk)
    return buf.getvalue()


def _load_16k_mono_pcm(path: str) -> np.ndarray:
    """Load any audio file as 16 kHz mono int16 PCM (file-fed verification).

    Uses librosa so mp3/wav/etc. all work; resamples to the 16 kHz the STT path
    requires. Returned as a 1-D int16 array ready to chunk onto the socket.
    """
    import librosa
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return (np.clip(y, -1.0, 1.0) * 32767.0).astype(np.int16)


async def _file_stream_mode(client, file_path: str, args):
    """Stream a prepared audio file at real-time pace through the live STT socket.

    Real-time pacing (one ~64 ms block, then sleep ~64 ms) is essential: the
    speculative endpoint hold is a wall-clock timer (VM_STT_ENDPOINT_HOLD_MS) and
    NeMo's EoU keys off silence *duration*, so a clip's baked-in pauses only
    reproduce the endpoint→confirm / endpoint→reopen→stitch cycles
    when the silence elapses in real time on the wire. Dumping the file at once
    would collapse those timers. Deterministic, repeatable evidence for the endpoint hold.
    """
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    pcm = _load_16k_mono_pcm(file_path)
    duration = len(pcm) / SAMPLE_RATE
    print(f"Streaming {file_path} — {duration:.2f}s @ 16 kHz mono, real-time pace\n")

    async def file_chunks():
        for start in range(0, len(pcm), BLOCKSIZE):
            block = pcm[start:start + BLOCKSIZE]
            yield block.tobytes()
            await asyncio.sleep(len(block) / SAMPLE_RATE)

    t0 = None
    async for item in client.stream_stt_live(file_chunks()):
        text = item.get("text", "")
        final = item.get("is_final", True)
        if args.verbose:
            if t0 is None:
                t0 = time.monotonic()
            print("  " + _format_frame(item, t0))
        else:
            prefix = "FINAL" if final else "partial"
            print(f"  [{prefix}] {text}")


async def _clone_mode(client):
    """Record one sentence, clone the voice, echo it back, then exit."""
    mic_queue: asyncio.Queue[bytes] = asyncio.Queue()
    recorded_chunks: list[bytes] = []
    loop = asyncio.get_running_loop()

    def mic_callback(indata, frames, time_info, status):
        if status:
            print(f"  [mic warning: {status}]", file=sys.stderr)
        raw = bytes(indata)
        recorded_chunks.append(raw)
        loop.call_soon_threadsafe(mic_queue.put_nowait, raw)

    async def mic_chunks():
        while True:
            chunk = await mic_queue.get()
            yield chunk

    print("Say one sentence. It will be transcribed, your voice cloned, and echoed back.\n")

    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCKSIZE,
        callback=mic_callback,
    )
    stream.start()

    final_text = None
    try:
        async for item in client.stream_stt_live(mic_chunks()):
            text = item.get("text", "")
            final = item.get("is_final", True)
            prefix = "FINAL" if final else "partial"
            print(f"  [{prefix}] {text}")

            if final and text.strip():
                final_text = text.strip()
                # Keep recording briefly so the voice sample has a clean tail
                await asyncio.sleep(0.5)
                break
    finally:
        stream.stop()
        stream.close()

    if not final_text:
        print("No speech detected.")
        return

    print(f'\nFinal text: "{final_text}"')

    # Build WAV from recorded mic audio and upload for voice cloning
    wav_data = _chunks_to_wav(recorded_chunks, SAMPLE_RATE, CHANNELS)
    duration = len(b"".join(recorded_chunks)) / (SAMPLE_RATE * 2 * CHANNELS)
    print(f"Recorded {duration:.1f}s of audio ({len(wav_data)} bytes WAV)")
    print("Uploading voice sample for cloning...")
    result = await client.upload_voice(wav_data, filename="mic_recording.wav")
    print(f"  Server: {result}")

    # Synthesize with the cloned voice
    print("Synthesizing with cloned voice...")
    pcm_data, sample_rate = await client.synthesize_tts(final_text)
    if pcm_data:
        audio = np.frombuffer(pcm_data, dtype=np.int16)
        await asyncio.to_thread(_play_blocking, audio, sample_rate)
        print("Done.")
    else:
        print("TTS returned no audio.")


async def _clone_file_mode(client, file_path: str, text: str | None):
    """Clone voice from an audio file, then speak text in that voice."""
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading voice sample from {file_path}...")
    with open(file_path, "rb") as f:
        audio_data = f.read()

    filename = os.path.basename(file_path)
    print("Uploading voice sample for cloning...")
    result = await client.upload_voice(audio_data, filename=filename)
    print(f"  Server: {result}")

    if text:
        # One-shot mode
        print(f'Synthesizing: "{text}"')
        pcm_data, sample_rate = await client.synthesize_tts(text)
        if pcm_data:
            audio = np.frombuffer(pcm_data, dtype=np.int16)
            await asyncio.to_thread(_play_blocking, audio, sample_rate)
        else:
            print("TTS returned no audio.")
        return

    # Interactive prompt loop
    print("Voice cloned. Type text to hear it spoken. Ctrl+C or empty line to quit.\n")
    while True:
        try:
            line = await asyncio.to_thread(input, "Text> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.strip()
        if not line:
            break
        print("Synthesizing...")
        pcm_data, sample_rate = await client.synthesize_tts(line)
        if pcm_data:
            audio = np.frombuffer(pcm_data, dtype=np.int16)
            await asyncio.to_thread(_play_blocking, audio, sample_rate)
        else:
            print("TTS returned no audio.")


async def main():
    parser = argparse.ArgumentParser(description="Live mic STT test")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tts", action="store_true", help="Echo finals back via TTS")
    group.add_argument("--clone", action="store_true",
                       help="Record one sentence, clone voice, echo back, exit")
    group.add_argument("--clone-file", metavar="FILE",
                       help="Clone voice from audio file (mp3/wav), then speak text")
    group.add_argument("--file", metavar="FILE", dest="stream_file",
                       help="Stream a prepared audio file at real-time pace instead of the mic")
    parser.add_argument("--text", help="Text to speak (with --clone-file; omit for interactive prompt)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Endpoint-aware frame log: show segment_index/endpoint/is_final per frame")
    parser.add_argument("--host", default="localhost", help="Server host (default localhost)")
    parser.add_argument("--port", type=int, default=8255, help="Server port (default 8255)")
    args = parser.parse_args()

    token = os.environ["VOICEMANAGER_TOKEN"]

    print("Connecting to VoiceManager...")
    async with VoiceManagerClient(host=args.host, port=args.port, token=token) as client:
        print(f"Connected (session={client.session_id}, stt_engine={client.stt_engine})")

        if args.clone:
            print("Mode: Voice clone (mic)\n")
            await _clone_mode(client)
            return

        if args.clone_file:
            print("Mode: Voice clone (file)\n")
            await _clone_file_mode(client, args.clone_file, args.text)
            return

        if args.stream_file:
            mode = "File stream (verbose/endpoint-aware)" if args.verbose else "File stream"
            print(f"Mode: {mode}\n")
            await _file_stream_mode(client, args.stream_file, args)
            return

        mode = "STT + TTS echo" if args.tts else "STT only"
        if args.verbose:
            mode += " (verbose/endpoint-aware)"
        print(f"Mode: {mode}")
        print("Speak into the microphone. Ctrl+C to stop.\n")

        mic_queue: asyncio.Queue[bytes] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def mic_callback(indata, frames, time_info, status):
            if status:
                print(f"  [mic warning: {status}]", file=sys.stderr)
            loop.call_soon_threadsafe(mic_queue.put_nowait, bytes(indata))

        async def mic_chunks():
            while True:
                chunk = await mic_queue.get()
                yield chunk

        stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            blocksize=BLOCKSIZE,
            callback=mic_callback,
        )
        stream.start()

        t0 = None
        try:
            async for item in client.stream_stt_live(mic_chunks()):
                text = item.get("text", "")
                final = item.get("is_final", True)

                if args.verbose:
                    if t0 is None:
                        t0 = time.monotonic()
                    print("  " + _format_frame(item, t0))
                else:
                    lang = item.get("language", "?")
                    conf = item.get("confidence", 0)
                    prefix = "FINAL" if final else "partial"
                    print(f"  [{lang}] ({conf:.2f}) [{prefix}] {text}")

                if final and args.tts and text.strip():
                    await _play_tts(client, text)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            stream.stop()
            stream.close()


if __name__ == "__main__":
    # Guard module-scope execution: without this, importing this module (e.g.
    # pytest collection under `-k 'not mic'`) runs main() and argparse exits the
    # interpreter, crashing collection.
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Ctrl+C during a live mic pass: asyncio.run cancels the task and
        # re-raises KeyboardInterrupt at this level (the in-loop handler can't
        # catch it), so swallow it here for a clean exit instead of a traceback.
        print("\nStopped.")
