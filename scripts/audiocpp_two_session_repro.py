#!/usr/bin/env python3
"""Two-session /ws/stream repro for the audio.cpp engine's single-client defect
- MANUAL, needs a running VoiceManager server on the audiocpp engine + the real sidecar.

What it shows: the audio.cpp
engine is effectively SINGLE-CLIENT. `sidecar/src/main.cpp` accepts a connection and runs
`handle_connection()` INLINE, so the next accept() does not happen until the current
client disconnects - there are zero threading primitives anywhere in sidecar/src/. A
second concurrent consumer completes the kernel-level connect via the listen backlog,
sends HELLO, then blocks in `_handshake` -> `_recv_frame_blocking` for the pipeline's
120 s socket timeout, raises SidecarFault, and error-closes 1011 having transcribed
nothing. For a service other projects consume (at least one live consumer
is a wall console), that is a blocker.

This is the **regression target for the concurrency fix**, the same pattern
scripts/audiocpp_ws_repro.py used for mid-session finals: it exits NON-ZERO while the
bug is present, and 0 once BOTH sessions transcribe. A `pytest`-collected model-free
test must NOT depend on a live server, and - just as importantly - must not assert this
at all: the fake sidecar is deliberately serial (tests/audiocpp_fake.py), so the claim
that a sidecar fault "leaves concurrent sessions unaffected" is a LIVE claim only. This
stays a MANUAL scripts/ helper (never collected).

Start the server first (test port, does not touch your .env), e.g.:

  VM_STT_ENGINE=audiocpp \
  VM_AUDIOCPP_MODEL=/home/user/audiocpp-models/Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-f16.gguf \
  VM_AUDIOCPP_BACKEND=cuda VM_AUDIOCPP_DEVICE=0 VM_DEVICE=cuda:0 \
  VM_HOST=127.0.0.1 VM_PORT=8077 \
  ./venv/bin/python -m voicemanager

Then, in another shell:

  export VOICEMANAGER_TOKEN=vmp_...   # from: voicemanager user token add USER NAME
  ./venv/bin/python scripts/audiocpp_two_session_repro.py --port 8077
  ./venv/bin/python scripts/audiocpp_two_session_repro.py --port 8077 --no-wait-close  # quick verdict

Two mechanisms can end the starved session, and which one you see depends on how long
the holding session stays connected. Both are the same single-client defect:

  * the holder outlives B's read timeout -> B blocks the full 120 s in `_handshake`,
    raises SidecarFault and error-closes 1011 (pure serial accept);
  * the holder disconnects first -> it OWNS the spawned sidecar process and terminates
    it on close (no refcount), so B's pending read gets EOF and error-closes 1011
    early. Same verdict, different mechanism.

The default --wait-close (135 s) keeps BOTH sessions connected past the 120 s timeout,
so a default run observes the serial path with the holder still attached - at the cost of
a ~2.5 min run. The transcription verdict (the part that gates exit status) is reached
in seconds either way; --no-wait-close skips the long wait and typically shows the
early-EOF path instead.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time

import websockets  # the Python client's transport; also used by the test suite

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 5120  # ~0.32 s
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CLIP = os.path.join(REPO_ROOT, "voices", "designed", "designed_female_s101.wav")


def load_pcm_s16le(clip: str) -> bytes:
    if not os.path.exists(clip):
        sys.exit(f"reference clip not found: {clip}")
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", clip, "-f", "s16le", "-acodec", "pcm_s16le",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-"], capture_output=True)
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed on {clip}: {proc.stderr.decode(errors='replace')}")
    return proc.stdout


def _token_from_env() -> str:
    return os.environ.get("VOICEMANAGER_TOKEN", "")


class Session:
    """One /ws/stream session: streams the clip and records what came back."""

    def __init__(self, name: str, uri: str, token: str, pcm: bytes, t0: float):
        self.name = name
        self.uri = uri
        self.headers = {"Authorization": f"Bearer {token}"}
        self.pcm = pcm
        self.t0 = t0
        self.msgs: list[dict] = []
        self.close_code: int | None = None
        self.closed = asyncio.Event()

    # --- observations the verdict is built from ---------------------------------
    @property
    def transcriptions(self) -> list[dict]:
        return [m for m in self.msgs if m.get("type") == "transcription"]

    @property
    def errors(self) -> list[dict]:
        return [m for m in self.msgs if m.get("type") == "error"]

    def log(self, text: str) -> None:
        print(f"  [{(time.monotonic() - self.t0) * 1000:7.0f} ms] {self.name}: {text}")

    async def run(self, stagger_s: float, hold_s: float) -> None:
        """Connect, wait `stagger_s`, stream the clip at 1x, then hold the socket open.

        The stagger makes the roles deterministic: whoever sends audio first reaches the
        sidecar first (the pipeline connects lazily on the first feed), so A is the served
        session and B is the one the serial accept loop leaves in the backlog.
        """
        async with websockets.connect(self.uri, max_size=None, additional_headers=self.headers) as ws:
            hs = json.loads(await ws.recv())
            self.log(f"handshake stt_engine={hs.get('stt_engine')} session={hs.get('session_id')}")
            reader = asyncio.create_task(self._reader(ws))
            try:
                if stagger_s:
                    await asyncio.sleep(stagger_s)
                chunk_dur = CHUNK_SAMPLES / SAMPLE_RATE
                n = 0
                for off in range(0, len(self.pcm), CHUNK_SAMPLES * 2):
                    try:
                        await ws.send(self.pcm[off:off + CHUNK_SAMPLES * 2])
                    except websockets.exceptions.ConnectionClosed:
                        self.log("send failed: server already closed this session")
                        break
                    n += 1
                    await asyncio.sleep(chunk_dur)
                self.log(f"sent {n} audio frames; holding the socket open {hold_s:.0f}s")
                await asyncio.wait_for(self.closed.wait(), timeout=hold_s)
                self.log("server closed the session")
            except asyncio.TimeoutError:
                pass
            finally:
                reader.cancel()

    async def _reader(self, ws) -> None:
        try:
            while True:
                m = json.loads(await ws.recv())
                self.msgs.append(m)
                if m.get("type") == "transcription":
                    self.log(f"transcription is_final={m.get('is_final')} "
                             f"seg={m.get('segment_index')} text={m.get('text')!r}")
                else:
                    self.log(f"{m.get('type')} {json.dumps(m)[:160]}")
        except websockets.exceptions.ConnectionClosed as exc:
            self.close_code = exc.rcvd.code if exc.rcvd is not None else ws.close_code
            self.log(f"closed by server, code={self.close_code}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a manual probe; report and move on
            self.log(f"reader ended: {type(exc).__name__}: {exc}")
        finally:
            self.closed.set()


def _verdict(a: Session, b: Session) -> int:
    print("\n" + "=" * 78)
    for s in (a, b):
        finals = [m for m in s.transcriptions if m.get("is_final") is True]
        partials = [m for m in s.transcriptions if m.get("is_final") is False]
        print(f"{s.name}: {len(partials)} partial(s), {len(finals)} final(s), "
              f"{len(s.errors)} error frame(s), close_code="
              f"{s.close_code if s.close_code is not None else 'not observed'}")
        for e in s.errors:
            print(f"    error_type={e.get('error_type')!r} message={e.get('message')!r}")
        for f in finals:
            print(f"    final text={f.get('text')!r}")

    both_transcribed = bool(a.transcriptions) and bool(b.transcriptions)
    print("=" * 78)
    if both_transcribed:
        print("RESULT: PASS - both concurrent sessions transcribed")
        return 0
    starved = [s.name for s in (a, b) if not s.transcriptions]
    print(f"RESULT: REPRO CONFIRMED - {', '.join(starved)} transcribed NOTHING while a "
          f"concurrent session held the sidecar.\n"
          f"        The audio.cpp engine serves one session at a time; not yet fixed.")
    return 1


async def run(port: int, token: str, clip: str, stagger: float, wait_close: float) -> int:
    pcm = load_pcm_s16le(clip)
    dur = len(pcm) / 2 / SAMPLE_RATE
    uri = f"ws://127.0.0.1:{port}/ws/stream"
    print(f"connecting two concurrent sessions to {uri}")
    print(f"clip={os.path.basename(clip)} dur={dur:.2f}s stagger={stagger:.1f}s "
          f"wait_close={wait_close:.0f}s\n")

    t0 = time.monotonic()
    a = Session("A", uri, token, pcm, t0)
    b = Session("B", uri, token, pcm, t0)
    # A streams immediately and so owns the sidecar; B starts `stagger` seconds later and
    # is the one the serial accept loop strands. hold_s covers the clip plus the wait for
    # B's error close (the pipeline's 120 s sidecar read timeout).
    hold = dur + wait_close + 5.0
    await asyncio.gather(a.run(0.0, hold), b.run(stagger, hold))
    return _verdict(a, b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8077, help="running server port (default 8077)")
    ap.add_argument("--token", default="", help="bearer token (default: $VOICEMANAGER_TOKEN)")
    ap.add_argument("--clip", default=DEFAULT_CLIP, help="reference audio clip")
    ap.add_argument("--stagger", type=float, default=1.0,
                    help="seconds session B waits before its first audio frame (default 1)")
    ap.add_argument("--wait-close", type=float, default=135.0,
                    help="seconds to wait for the starved session's error close "
                         "(default 135: the pipeline's 120s sidecar read timeout plus slack)")
    ap.add_argument("--no-wait-close", action="store_true",
                    help="skip the long wait; report the transcription verdict only")
    args = ap.parse_args()
    token = args.token or _token_from_env()
    if not token:
        sys.exit("no token: pass --token or export VOICEMANAGER_TOKEN")
    wait_close = 5.0 if args.no_wait_close else args.wait_close
    return asyncio.run(run(args.port, token, args.clip, args.stagger, wait_close))


if __name__ == "__main__":
    sys.exit(main())
