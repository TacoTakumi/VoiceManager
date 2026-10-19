"""Quick TTS smoke test using the VoiceManager client."""
import asyncio
import os
from dotenv import load_dotenv
from voicemanager.client import VoiceManagerClient

load_dotenv()


async def main():
    token = os.environ["VOICEMANAGER_TOKEN"]
    async with VoiceManagerClient(token=token) as client:
        print(await client.health())
        pcm, rate = await client.synthesize_tts("Hi.")
        client.save_tts_to_file(pcm, "test_output.wav", rate)
        print(f"Saved test_output.wav ({len(pcm)} bytes, {rate} Hz)")


asyncio.run(main())
