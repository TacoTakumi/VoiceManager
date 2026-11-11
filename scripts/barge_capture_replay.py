#!/usr/bin/env python3
"""Replay barge captures through the audio.cpp STT, with and without the lead-in.

A manual script: it needs the GPU, the audio.cpp sidecar binary and its STT model
(the VM_AUDIOCPP_* settings), and a directory of barge captures, which the server
writes when VM_TYGER_BARGE_CAPTURE_DIR is set. It is not part of the automated
suite. It does not talk to a running server: it starts one STT sidecar of its own
and stops it at the end.

Run it with the server's python:

    venv/bin/python scripts/barge_capture_replay.py \\
        /home/you/.local/state/voicemanager/barge-captures \\
        [--session PREFIX] [--request-options "k=v,k=v"] [--lead-ms N]

The settings come from the ``.env`` of the VoiceManager home, as for the server:
the checkout this file is in, whatever the current directory is. To run the script
of one checkout with the settings of another one (a git worktree has no ``.env``),
set VOICEMANAGER_HOME to the directory that holds the ``.env``.

Each capture is a pair ``barge-<session>-<mic_sample>.wav`` and ``.json`` (see "The
barge capture" in voicemanager/transport/turn_engine.py). For each one the script
takes the audio from the sample the server fed the STT from, up to the end of the
WAV, and decodes it twice as one utterance, in 20 ms chunks: once with the noise
lead-in in front (see "The barge lead-in" in the same module), once with nothing in
front. One row per capture gives the barge's mic sample, the session, the finals
the live server sent, and the two finals of the replay. Two summary lines count the
non-empty finals of each kind.

The lead-in is random noise, so two runs can differ. The captures hold a person's
voice: the script only reads them.

Exit code: 0 when the replay ran, 1 when the sidecar failed or the time limit
passed, 2 when the directory has no capture that matches.
"""
import argparse
import asyncio
import json
import sys
import threading
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))        # the code of the checkout this file is in
from voicemanager.config import get_settings, parse_option_list  # noqa: E402
from voicemanager.models import _load_audiocpp_models  # noqa: E402
from voicemanager.pipeline.audiocpp_stt import (  # noqa: E402
    AudioCppSTTPipeline, SidecarError, SidecarFault,
)
from voicemanager.transport.turn_engine import barge_lead_in  # noqa: E402

SR = 16000
CHUNK_BYTES = 640                 # 20 ms of 16 kHz mono 16-bit PCM
WARMUP = b"\x00\x00" * SR         # one second of silence
TIME_LIMIT_S = 300.0              # for the whole replay, the model load included


def load_captures(directory: Path, session_prefix: str) -> list[tuple[dict, bytes]]:
    """Each capture's note and the PCM the server fed the STT, by session, then barge."""
    captures = []
    for note_path in directory.glob("barge-*.json"):
        note = json.loads(note_path.read_text())
        wav_path = note_path.with_suffix(".wav")
        if not note["session"].startswith(session_prefix) or not wav_path.is_file():
            continue
        with wave.open(str(wav_path), "rb") as wf:
            if (wf.getframerate(), wf.getnchannels(), wf.getsampwidth()) != (SR, 1, 2):
                sys.exit(f"error: {wav_path.name} is not 16 kHz mono 16-bit")
            pcm = wf.readframes(wf.getnframes())
        offset = max(0, note["stt_fed_from_mic_sample"] - note["start_mic_sample"])
        captures.append((note, pcm[offset * 2:]))
    captures.sort(key=lambda c: (c[0]["session"], c[0]["barge_mic_sample"]))
    return captures


async def decode(pipe: AudioCppSTTPipeline, lead: bytes, pcm: bytes) -> str:
    """One utterance: the lead-in as its own chunk, the audio in 20 ms chunks, a flush.

    A final the sidecar gives before the flush is kept and starts a new segment, as
    in the server, so the row holds every word the decoder gave for the audio.
    """
    texts = []
    chunks = [pcm[i:i + CHUNK_BYTES] for i in range(0, len(pcm), CHUNK_BYTES)]
    for chunk in ([lead] if lead else []) + chunks:
        await pipe.feed(chunk)
        while (final := pipe.take_final()) is not None:
            texts.append(final[0])
            pipe.reset()
    texts.append((await pipe.finalize())[0])
    pipe.reset()
    return " ".join(t.strip() for t in texts if t.strip())


def cell(text: str) -> str:
    """A table cell in plain ASCII; ``-`` for no text."""
    return text.encode("ascii", "backslashreplace").decode() or "-"


async def replay(pipe: AudioCppSTTPipeline, captures, lead_ms: int) -> None:
    live = [cell(" / ".join(note["finals"])) for note, _ in captures]
    width = max(len("live server"), *map(len, live))
    print(f"{'barge':>9}  {'session':8}  {'live server':{width}}  "
          f"{'with the lead-in':28}  without the lead-in")
    await decode(pipe, b"", WARMUP)           # the cold start, outside the rows
    with_lead = without_lead = 0
    for (note, pcm), live_text in zip(captures, live):
        led = await decode(pipe, barge_lead_in(lead_ms), pcm)
        bare = await decode(pipe, b"", pcm)
        with_lead += bool(led)
        without_lead += bool(bare)
        print(f"{note['barge_mic_sample']:>9}  {note['session'][:8]:8}  {live_text:{width}}  "
              f"{cell(led):28}  {cell(bare)}", flush=True)
    print(f"non-empty finals with the lead-in:    {with_lead} of {len(captures)}")
    print(f"non-empty finals without the lead-in: {without_lead} of {len(captures)}")


async def run(config: dict, captures, lead_ms: int) -> None:
    pipe = AudioCppSTTPipeline(config)
    try:
        await asyncio.wait_for(replay(pipe, captures, lead_ms), TIME_LIMIT_S)
    finally:
        # Stops the sidecar, also when a worker thread still waits on it. dispose()
        # only signals the process; its reaper thread is a daemon, so wait for it
        # here, or the exit of the script would end it before a kill it may owe.
        pipe.dispose()
        for thread in threading.enumerate():
            if thread.name == "audiocpp-sidecar-reap":
                thread.join(timeout=15.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("capture_dir", metavar="CAPTURE_DIR", type=Path,
                        help="a directory of barge captures (barge-*.wav and .json)")
    parser.add_argument("--session", default="", metavar="PREFIX",
                        help="only the captures whose session starts with PREFIX")
    parser.add_argument("--request-options", default=None, metavar="K=V,K=V",
                        help="STT request options in place of the ones of the settings "
                             "(the format of VM_AUDIOCPP_REQUEST_OPTIONS)")
    parser.add_argument("--lead-ms", type=int, default=None, metavar="N",
                        help="the length of the lead-in (default: VM_TYGER_BARGE_LEAD_MS)")
    args = parser.parse_args()

    captures = load_captures(args.capture_dir, args.session)
    if not captures:
        print(f"error: no barge capture in {args.capture_dir}"
              + (f" for a session that starts with {args.session!r}" if args.session else ""),
              file=sys.stderr)
        return 2

    settings = get_settings()
    config = _load_audiocpp_models(settings)["audiocpp_config"]
    if args.request_options is not None:
        config["request_options"] = parse_option_list(args.request_options)
    lead_ms = settings.tyger_barge_lead_ms if args.lead_ms is None else args.lead_ms
    print(f"request options: {','.join(config['request_options']) or '(none)'}")
    print(f"lead-in: {lead_ms} ms")
    print(f"captures: {len(captures)}")
    try:
        asyncio.run(run(config, captures, lead_ms))
    except TimeoutError:
        print(f"error: the replay did not end in {TIME_LIMIT_S:.0f} s", file=sys.stderr)
        return 1
    except (SidecarError, SidecarFault) as exc:
        print(f"error: the STT sidecar failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
