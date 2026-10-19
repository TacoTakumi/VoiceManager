#!/usr/bin/env python3
"""Concurrent-sidecar capacity probe for the audio.cpp engine
- MANUAL, needs a running VoiceManager server on the audiocpp engine + the real sidecar.

What it verifies: the documented admission policy is the observed one. One sidecar per
session is deliberate (it isolates the server from a runtime crash), but each session
costs a process, a CUDA context and a resident f16 GGUF on the single VM_AUDIOCPP_DEVICE,
so concurrency is bounded by that GPU's VRAM whether or not anyone counts it. Before
the cap, nothing counted it: the Nth session exhausted the device, `open_streaming_session`
threw, the sidecar answered ERROR `load_failed`, and the WS handler mapped that to close
4001 `stt_backend_unavailable` - telling a correctly-configured, merely-full server's
client "stop retrying, an operator must fix this". Now the cap is checked before
the spawn and the refusal is `at_capacity` -> close 4002 `stt_backend_unready`, retriable.

Run with MORE sessions than the cap. Expected: the first `--cap` sessions transcribe
normally, every later one is refused with close 4002 / at_capacity, and nvidia-smi shows
VRAM rising by roughly one sidecar's worth per ADMITTED session and not at all for the
refused ones. Exits 0 when observed matches documented, non-zero otherwise.

Start the server first (test port, does not touch your .env) - note the deliberately
small cap, so the ceiling is reachable without filling the card:

  VM_STT_ENGINE=audiocpp \
  VM_AUDIOCPP_MODEL=/home/user/audiocpp-models/Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-f16.gguf \
  VM_AUDIOCPP_BACKEND=cuda VM_AUDIOCPP_DEVICE=0 VM_DEVICE=cuda:0 \
  VM_AUDIOCPP_MAX_SIDECARS=2 \
  VM_HOST=127.0.0.1 VM_PORT=8077 \
  ./venv/bin/python -m voicemanager

Then, in another shell:

  export VOICEMANAGER_TOKEN=vmp_...   # from: voicemanager user token add USER NAME
  ./venv/bin/python scripts/audiocpp_capacity_repro.py --port 8077 --cap 2 --sessions 4

This stays a MANUAL scripts/ helper (never collected by pytest): it needs a live server,
a GPU and the real sidecar. The model-free half of the cap's verification - that the cap
refuses with a defined retriable code and leaves nothing behind - is
tests/test_audiocpp_capacity.py, which asserts the policy against a fake sidecar.
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

AT_CAPACITY_CLOSE = 4002          # stt_backend_unready (error-code taxonomy: retriable)
PERMANENT_CLOSE = 4001            # stt_backend_unavailable (what the bug used to send)


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


def gpu_used_mib(device: int) -> int | None:
    """Used VRAM on `device`, or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={device}", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


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

    @property
    def transcriptions(self) -> list[dict]:
        return [m for m in self.msgs if m.get("type") == "transcription"]

    @property
    def errors(self) -> list[dict]:
        return [m for m in self.msgs if m.get("type") == "error"]

    @property
    def refusal_cause(self) -> str:
        """The cause code the server named in its error frame, e.g. 'at_capacity'."""
        for m in self.errors:
            msg = m.get("message", "")
            if "(" in msg and ")" in msg:
                return msg[msg.index("(") + 1:msg.index(")")]
        return ""

    def log(self, text: str) -> None:
        print(f"  [{(time.monotonic() - self.t0) * 1000:7.0f} ms] {self.name}: {text}")

    async def run(self, stagger_s: float, hold_s: float) -> None:
        """Connect, wait `stagger_s`, stream the clip at 1x, then hold the socket open.

        The stagger makes admission order deterministic: the pipeline connects to its
        sidecar lazily on the FIRST AUDIO FRAME, not at WS accept, so whoever sends audio
        first takes the first capacity slot. Without it, which sessions land inside the
        cap would be a race and the verdict would be meaningless.
        """
        try:
            async with websockets.connect(self.uri, max_size=None, additional_headers=self.headers) as ws:
                hs = json.loads(await ws.recv())
                self.log(f"handshake stt_engine={hs.get('stt_engine')} "
                         f"session={hs.get('session_id')}")
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
                            self.log(f"send stopped: server closed this session after {n} frames")
                            break
                        n += 1
                        await asyncio.sleep(chunk_dur)
                    else:
                        self.log(f"sent {n} audio frames; holding the socket open")
                    await asyncio.wait_for(self.closed.wait(), timeout=hold_s)
                except asyncio.TimeoutError:
                    pass
                finally:
                    reader.cancel()
        except Exception as exc:  # noqa: BLE001 - a manual probe; report and move on
            self.log(f"session ended: {type(exc).__name__}: {exc}")
        finally:
            self.closed.set()

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
        except Exception as exc:  # noqa: BLE001
            self.log(f"reader ended: {type(exc).__name__}: {exc}")
        finally:
            self.closed.set()


def _verdict(sessions: list[Session], cap: int, vram: list[tuple[str, int | None]]) -> int:
    print("\n" + "=" * 78)
    for s in sessions:
        finals = [m for m in s.transcriptions if m.get("is_final") is True]
        partials = [m for m in s.transcriptions if m.get("is_final") is False]
        print(f"{s.name}: {len(partials)} partial(s), {len(finals)} final(s), "
              f"{len(s.errors)} error frame(s), close_code="
              f"{s.close_code if s.close_code is not None else 'not observed'}")
        for e in s.errors:
            print(f"    error_type={e.get('error_type')!r} message={e.get('message')!r}")
        for f in finals:
            print(f"    final text={f.get('text')!r}")

    print("-" * 78)
    print("VRAM on the sidecar device (nvidia-smi memory.used, MiB):")
    for label, used in vram:
        print(f"    {label:<28} {used if used is not None else 'n/a'}")
    baseline = vram[0][1] if vram else None
    peak = max((u for _, u in vram if u is not None), default=None)
    admitted = [s for s in sessions if s.transcriptions]
    if baseline is not None and peak is not None and admitted:
        print(f"    -> +{peak - baseline} MiB across {len(admitted)} admitted session(s) "
              f"= ~{(peak - baseline) // len(admitted)} MiB per sidecar")

    print("=" * 78)
    problems: list[str] = []
    for s in sessions[:cap]:
        if not s.transcriptions:
            problems.append(f"{s.name} was within the cap of {cap} but transcribed nothing")
    for s in sessions[cap:]:
        if s.transcriptions:
            problems.append(f"{s.name} was over the cap of {cap} but was admitted anyway "
                            f"- the cap is not being enforced")
            continue
        if s.close_code != AT_CAPACITY_CLOSE:
            what = ("the PERMANENT code the out-of-VRAM path used before the cap"
                    if s.close_code == PERMANENT_CLOSE else "an undefined code")
            problems.append(f"{s.name} was refused with close {s.close_code} ({what}), "
                            f"expected {AT_CAPACITY_CLOSE} stt_backend_unready")
        if s.refusal_cause != "at_capacity":
            problems.append(f"{s.name} named cause {s.refusal_cause!r}, expected 'at_capacity' "
                            f"- capacity exhaustion must not be disguised as another fault")

    if problems:
        print("RESULT: FAIL - observed behaviour does not match the documented policy:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"RESULT: PASS - the first {cap} session(s) transcribed; the remaining "
          f"{len(sessions) - cap} were refused with close {AT_CAPACITY_CLOSE} "
          f"stt_backend_unready / at_capacity (retriable), as documented")
    return 0


async def run(port: int, token: str, clip: str, cap: int, count: int,
              stagger: float, device: int) -> int:
    pcm = load_pcm_s16le(clip)
    dur = len(pcm) / 2 / SAMPLE_RATE
    uri = f"ws://127.0.0.1:{port}/ws/stream"
    print(f"connecting {count} concurrent sessions to {uri}")
    print(f"clip={os.path.basename(clip)} dur={dur:.2f}s cap={cap} stagger={stagger:.1f}s "
          f"device=cuda:{device}\n")

    vram = [("baseline (no sessions)", gpu_used_mib(device))]
    t0 = time.monotonic()
    sessions = [Session(f"S{i + 1}", uri, token, pcm, t0) for i in range(count)]
    hold = dur + 20.0

    async def sample_peak() -> None:
        """Sample VRAM once every admitted sidecar has had time to load its model."""
        await asyncio.sleep(stagger * count + 12.0)
        vram.append(("all sessions streaming", gpu_used_mib(device)))

    # Each session starts `stagger` seconds after the previous one, so slot order is the
    # session order and the verdict can name which ones should have been refused.
    await asyncio.gather(
        *(s.run(stagger * i, hold) for i, s in enumerate(sessions)),
        sample_peak(),
    )
    vram.append(("after all disconnected", gpu_used_mib(device)))
    return _verdict(sessions, cap, vram)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8077, help="running server port (default 8077)")
    ap.add_argument("--token", default="", help="bearer token (default: $VOICEMANAGER_TOKEN)")
    ap.add_argument("--clip", default=DEFAULT_CLIP, help="reference audio clip")
    ap.add_argument("--cap", type=int, default=2,
                    help="the VM_AUDIOCPP_MAX_SIDECARS the server was started with (default 2)")
    ap.add_argument("--sessions", type=int, default=4,
                    help="how many concurrent sessions to open; must exceed --cap (default 4)")
    ap.add_argument("--stagger", type=float, default=2.0,
                    help="seconds between each session's first audio frame (default 2)")
    ap.add_argument("--device", type=int, default=int(os.environ.get("VM_AUDIOCPP_DEVICE", "0")),
                    help="GPU index to sample with nvidia-smi "
                         "(default: $VM_AUDIOCPP_DEVICE, else 0)")
    args = ap.parse_args()
    token = args.token or _token_from_env()
    if not token:
        sys.exit("no token: pass --token or export VOICEMANAGER_TOKEN")
    if args.sessions <= args.cap:
        sys.exit(f"--sessions ({args.sessions}) must exceed --cap ({args.cap}); the point "
                 f"is to observe what happens PAST the bound")
    return asyncio.run(run(args.port, token, args.clip, args.cap, args.sessions,
                           args.stagger, args.device))


if __name__ == "__main__":
    sys.exit(main())
