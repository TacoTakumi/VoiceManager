"""Reference voice agent over VoiceManager: the classic turn pattern and the micro-turn one.

Three subcommands, one per mode:

- ``classic``: wait for the person's turn, send it to any OpenAI-compatible chat
  endpoint, stream the reply through VoiceManager's TTS, and stop the reply when the
  person speaks over it. The turn ends on the server's endpoint frame (or the final
  transcript on engines without one); if the final revises the text, the reply is
  cancelled and the revision is sent instead.
- ``microturn``: the experimental full-duplex loop. Every 0.6 s the LLM with the
  micro-turn LoRA gets the words heard in that window and answers with a few words to
  speak or ``listen``. Needs the LoRA model under llama-server or llama-swap.
- ``replay``: the micro-turn loop on a scripted call as text, no audio.

Both live modes stream the microphone and play the speaker through the default audio
device: use headphones, there is no echo cancellation. They write a JSON lines log and
a WAV of what played under ``runs/``. The bearer token comes from the
``VOICEMANAGER_TOKEN`` environment variable.

    python scripts/voice_agent.py classic [--vm-host HOST] [--vm-port PORT] [--voice NAME]
        [--url URL] [--model NAME] [--tts-mode {clause,sentence}] [--duration SECONDS]
    python scripts/voice_agent.py microturn [...the options of the micro-turn live mode]
    python scripts/voice_agent.py replay [--script FILE] [--url URL] [--model NAME]
"""
import argparse
import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import microturn_client  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """The command line: one subcommand per mode."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    modes = parser.add_subparsers(dest="mode", required=True)
    classic = modes.add_parser(
        "classic", help="turn by turn with any OpenAI-compatible chat model")
    add_classic_arguments(classic)
    microturn = modes.add_parser(
        "microturn", help="the micro-turn full-duplex loop (needs the LoRA model)")
    microturn_client.add_live_arguments(microturn)
    replay = modes.add_parser("replay", help="the micro-turn loop on a scripted call, no audio")
    microturn_client.add_replay_arguments(replay)
    return parser


def add_classic_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of the classic mode."""
    microturn_client.add_server_options(parser)
    parser.add_argument("--vm-host", default="localhost", help="VoiceManager host")
    parser.add_argument("--vm-port", type=int, default=8255, help="VoiceManager port")
    parser.add_argument("--voice", default=None, help="TTS voice (default: the token's voice)")
    parser.add_argument("--duration", type=float, default=None,
                        help="stop after this many seconds (default: at Ctrl-C)")
    parser.add_argument("--log", default=None,
                        help="JSON lines log (default: runs/classic/classic-<stamp>.jsonl)")
    parser.add_argument("--wav", default=None,
                        help="WAV of the audio played (default: runs/classic/classic-<stamp>.wav)")
    parser.add_argument("--tts-mode", choices=("clause", "sentence"), default="clause",
                        help="chunking of the reply's tts_stream request (default: clause)")


def main(argv: list[str] | None = None) -> None:
    """Run the selected mode."""
    args = build_parser().parse_args(argv)
    if args.mode == "microturn":
        asyncio.run(microturn_client.live(args))
    elif args.mode == "replay":
        microturn_client.run_replay(args)
    else:
        raise SystemExit("error: classic mode is not built yet")


if __name__ == "__main__":
    main()
