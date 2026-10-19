"""Generate audio samples for every available VibeVoice voice preset.

Connects to a running VoiceManager server (VM_TTS_ENGINE=vibevoice),
discovers all loaded voices, and synthesizes a sample passage for each.
WAV files are saved to tests/voice_samples/.

Usage::

    python tests/sample_vibevoice_voices.py                    # default sample text
    python tests/sample_vibevoice_voices.py --text "Custom text to speak"
    python tests/sample_vibevoice_voices.py --voice Carter     # single voice only
    python tests/sample_vibevoice_voices.py --play             # play each sample after generating
"""
import asyncio
import os
import shutil
import subprocess
import sys

from dotenv import load_dotenv
from voicemanager.client import VoiceManagerClient

load_dotenv()

DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Every voice has its own character and warmth. "
    "Some are deep and resonant, others bright and clear. "
    "Listen carefully, and you will hear the difference."
)

OUTPUT_DIR = "tests/voice_samples"


def play_wav(path: str) -> None:
    """Play a WAV file using mplayer (if available)."""
    mplayer = shutil.which("mplayer")
    if not mplayer:
        print("  (mplayer not found — skipping playback)")
        return
    subprocess.run([mplayer, "-really-quiet", path], check=False)


async def sample_voice(client: VoiceManagerClient, voice: str, text: str, play: bool = False) -> None:
    """Synthesize text with the given voice and save to WAV."""
    pcm, rate = await client.synthesize_tts(text, voice=voice)
    assert len(pcm) > 0, f"No audio received for voice '{voice}'"

    out_path = os.path.join(OUTPUT_DIR, f"{voice.lower()}.wav")
    client.save_tts_to_file(pcm, out_path, rate)
    dur = len(pcm) / (rate * 2)
    print(f"  {voice:>8s}  {dur:5.1f}s  {len(pcm):>8,} bytes  -> {out_path}")
    if play:
        play_wav(out_path)


async def main():
    args = sys.argv[1:]

    # Parse --text
    text = DEFAULT_TEXT
    if "--text" in args:
        idx = args.index("--text")
        text = args[idx + 1] if idx + 1 < len(args) else DEFAULT_TEXT

    # Parse --voice (optional: limit to one voice)
    only_voice = None
    if "--voice" in args:
        idx = args.index("--voice")
        only_voice = args[idx + 1] if idx + 1 < len(args) else None

    play = "--play" in args or "-p" in args

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    token = os.environ["VOICEMANAGER_TOKEN"]
    async with VoiceManagerClient(token=token) as client:
        info = await client.info()
        voices = info.get("available_voices", [])
        engine = info.get("tts_engine", "unknown")

        print(f"Engine: {engine}")
        print(f"Voices: {', '.join(voices)}")
        print(f"Text:   {text[:80]}{'...' if len(text) > 80 else ''}")
        print()

        if engine != "vibevoice":
            print(f"ERROR: Server is running '{engine}', not vibevoice.")
            sys.exit(1)

        if not voices:
            print("ERROR: No voices available on server.")
            sys.exit(1)

        targets = [only_voice] if only_voice else voices
        for voice in targets:
            if voice not in voices:
                print(f"  WARNING: '{voice}' not in available voices, skipping")
                continue
            await sample_voice(client, voice, text, play=play)

        print(f"\nDone. {len(targets)} sample(s) in {OUTPUT_DIR}/")


asyncio.run(main())
