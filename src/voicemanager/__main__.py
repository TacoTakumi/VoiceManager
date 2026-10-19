"""Entry point: python -m voicemanager.

Runs the same serve path as `voicemanager serve`.
"""
from voicemanager.cli import configure_logging, run_server


def main() -> None:
    """Start the VoiceManager server, as `voicemanager serve` does."""
    configure_logging()
    run_server()


if __name__ == "__main__":
    main()
