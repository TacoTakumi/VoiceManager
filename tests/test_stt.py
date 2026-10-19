"""Quick STT smoke test — feed a WAV file to VoiceManager and print transcriptions."""
import asyncio
import os
import sys
from dotenv import load_dotenv
from voicemanager.client import VoiceManagerClient

load_dotenv()


async def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path-to-wav>")
        sys.exit(1)

    wav_path = sys.argv[1]
    if not os.path.exists(wav_path):
        print(f"File not found: {wav_path}")
        sys.exit(1)

    token = os.environ["VOICEMANAGER_TOKEN"]
    async with VoiceManagerClient(token=token) as client:
        print(f"Transcribing {wav_path} ...")
        results = await client.transcribe_file(wav_path)
        if not results:
            print("No transcriptions returned.")
        for r in results:
            print(f"  [{r.get('language', '?')}] (confidence={r.get('confidence', '?')}) {r.get('text', '')}")


asyncio.run(main())
