"""Fidelity and speed check of the Tyger turn-engine sidecar on captured calls.

A manual script: it needs a GPU, the sidecar environment (venv-tyger), the Tyger
bundle in the Hugging Face cache, and the spike checkout of the offline replay
harness (3rdParty/tyger-spike: its own .venv and the tyger-s2s source). It is not
part of the automated suite.

Each CALL is a capture prefix of ``scripts/voice_agent.py microturn --capture``, for
example runs/microturn/microturn-20261103-182047, with the files <prefix>-mic.wav (the
16 kHz mic stream), <prefix>-out.wav (the output timeline, as it played) and
<prefix>-capture.json (the first ADC and DAC times that place the output on the mic's
clock).

What it does:

1. The reference. The offline replay harness (replay_call.py of the spike) runs in
   the spike's own environment with the unchanged upstream turn rules of tyger-s2s,
   at the thresholds the sidecar runs (its defaults unless given here). The harness
   places the output at (first DAC time - first ADC time) on the mic's clock and calls
   a hop "speaking" while the placed output had a frame above -60 dB in the last
   0.8 s. The upstream rules have no silence fallback and fire a barge on every
   debounced yield, while the sidecar's rules latch a barge until the reply stops and
   seed the next turn with the voiced audio of the last 2 s of the reply. So the
   reference applies the same three documented differences on top of the upstream
   rules: the 1.2 s silence fallback of the harness's corpus replay (replay_tyger.py),
   the latch and the seed. The upstream events without them are printed as well.
2. The server path. The capture holds no playback edges, so they are taken from the
   reference's speaking hops: each run of speaking hops is one reply whose played
   span covers those hops, its PCM the output timeline over the span at the output's
   own rate. The script drives the server's own ``HopFeed`` with them as a client
   and a TTS session would (audio_start, the PCM and audio_end ahead of the play, a
   playback start and a stop each 100 ms after the edge, mic audio in 20 ms frames)
   and packs each hop with the server's ``_pack_hop`` (``wire.pack_hop``).
3. The sidecar. ``src/voicemanager/pipeline/tyger/server.py`` runs from venv-tyger
   with the real scorer; every hop goes over one WebSocket per call, followed by a
   text probe. The sidecar answers the probe with an error line once the hop before
   it is scored, so the time from the hop's send to that answer is the hop's
   processing time. Its GPU memory is read by polling nvidia-smi for its PID.
4. The checks, per call: the hop feed's modes equal the reference's on every hop;
   every endpoint (by model or by silence) and every barge of the reference is
   matched by one of the sidecar's of the same kind within one 320 ms hop, with none
   extra and none missing. Over all calls: the median hop time is under 40 ms and the
   sidecar's peak GPU memory is under 4 GiB. Any failure exits non-zero.

Run it on the 4060 Ti (nvidia-smi GPU 3), with the server's environment::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 ./venv/bin/python \\
        scripts/tyger_fidelity.py runs/microturn/microturn-20261103-182047 ...

Paths not given are looked up in this checkout, then in the main checkout of a git
worktree (venv-tyger and 3rdParty are local, not in git).
"""
import argparse
import asyncio
import json
import logging
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import websockets.asyncio.client  # noqa: E402
from scipy.signal import resample_poly  # noqa: E402

from voicemanager.pipeline.tyger import rules, scorer, wire  # noqa: E402
from voicemanager.pipeline.tyger.server import RULE_FLAGS  # noqa: E402
from voicemanager.transport.turn_engine import HopFeed, _pack_hop  # noqa: E402

logger = logging.getLogger("tyger_fidelity")

SERVER_SCRIPT = REPO / "src" / "voicemanager" / "pipeline" / "tyger" / "server.py"
HOP = wire.HOP_SAMPLES
SR = wire.SAMPLE_RATE
MIC_FRAME = 320                     # 20 ms of mic audio per client frame
EDGE_LAG = 1600                     # a playback frame comes 100 ms after its edge
SEND_AHEAD = SR                     # a reply's PCM goes out 1 s before it plays
MAX_HOP_MS = 40.0
MAX_GPU_MIB = 4096.0
PROBE = "probe"
PROBE_ANSWER = "expected a binary hop"
MATCH_HOPS = 1

# The reference: replay_call.py's recipe with the upstream rules, run in the
# spike's environment. Arguments: TYGER_REPO BUNDLE SETTINGS_JSON OUT_JSON STEM...
REFERENCE = r'''
import collections, json, os, sys
from pathlib import Path

TYGER_REPO, BUNDLE, SETTINGS, OUT = sys.argv[1], Path(sys.argv[2]), json.loads(sys.argv[3]), sys.argv[4]
STEMS = sys.argv[5:]
sys.path.insert(0, TYGER_REPO)

import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly
from app.plugins.ears.tyger import rules
from app.plugins.ears.tyger.bundle import load
from app.plugins.ears.tyger.constants import (FRAME_S, HOP_FRAMES, HOP_N, L15, MIN_SPEECH_MS, SR,
                                               VAD_DB, WARMUP_FRAMES, WINDOW_S, LISTEN)
from app.plugins.ears.tyger.session import EarsSession

# The upstream rules hard-code these; the sidecar takes them as settings.
fixed = {"voice_gate_db": VAD_DB, "min_speech_ms": MIN_SPEECH_MS, "warmup_frames": WARMUP_FRAMES}
bad = {k: (SETTINGS[k], v) for k, v in fixed.items() if float(SETTINGS[k]) != float(v)}
if bad:
    sys.exit(f"the upstream rules fix {bad} (sidecar, upstream); run the sidecar at those")
FB_FRAMES = -(-int(SETTINGS["silence_fallback_ms"]) // int(FRAME_S * 1000))
SEED_FRAMES = int(round(SETTINGS["barge_seed_ms"] / (FRAME_S * 1000)))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

nemotron, proc, tt, cfg = load(BUNDLE)
nemotron = nemotron.to(DEV).eval(); tt = tt.to(DEV).eval(); enc = nemotron.encoder


class Stop(Exception):
    pass


grab = {}


def hook(m, a, o_):
    if grab.get("on"):
        grab["h"] = o_[0] if isinstance(o_, tuple) else o_
        raise Stop


enc.layers[L15 - 1].register_forward_hook(hook)


@torch.no_grad()
def l15(wav):
    inp = proc(wav, sampling_rate=SR, return_tensors="pt")
    am = inp.get("attention_mask")
    grab["on"] = True
    try:
        enc(input_features=inp["input_features"].to(DEV), attention_mask=am.to(DEV) if am is not None else None,
            num_lookahead_tokens=0)
    except Stop:
        pass
    finally:
        grab["on"] = False
    return grab["h"][0].float()


def fallback_endpoint(sess):
    """replay_tyger.py: the state changes rules.on_frame makes when it ends a turn."""
    sess.turn_u = np.zeros(0, np.float32); sess.voiced_ms = 0.0; sess.in_turn = False
    sess.seg_s = 0.0; sess.segs = []; sess.prev_state = LISTEN; sess.warm = 0
    sess.bc_run = 0; sess.bc_cooldown = 0; sess.ep_run = 0
    sess.partial_asr = ""; sess.partial_at_s = 0.0; sess.turn_gen += 1
    sess.words = sess.words_sent = ""


def stem_reference(stem):
    meta = json.loads(Path(f"{stem}-capture.json").read_text())
    mic, sr = sf.read(f"{stem}-mic.wav", dtype="float32"); assert sr == SR
    out, osr = sf.read(f"{stem}-out.wav", dtype="float32")
    out16 = resample_poly(out, SR, osr).astype(np.float32) if osr != SR else out
    off_s = meta["out"]["first_dac_time"] - meta["mic"]["first_adc_time"]
    agent = np.zeros_like(mic)
    o = int(round(off_s * SR)); seg = out16[:max(0, len(mic) - o)]
    agent[o:o + len(seg)] = seg
    n_hops = len(mic) // HOP_N
    mic, agent = mic[:n_hops * HOP_N], agent[:n_hops * HOP_N]
    a_db = rules.frame_energy(agent)
    voiced_a = a_db > -60.0
    look = int(round(0.8 / FRAME_S))
    speaking_hop = [bool(voiced_a[max(0, (h + 1) * HOP_FRAMES - look):(h + 1) * HOP_FRAMES].any())
                    for h in range(n_hops)]
    win_n = int(WINDOW_S * SR)
    feats_u, feats_a = [], []
    with torch.no_grad():
        for h in range(n_hops):
            e = (h + 1) * HOP_N
            wu = mic[max(0, e - win_n):e]
            wa = agent[max(0, e - win_n):e].copy()
            for k in range(h + 1):
                if not speaking_hop[k]:
                    s0 = k * HOP_N - (e - len(wa))
                    if s0 + HOP_N > 0:
                        wa[max(0, s0):s0 + HOP_N] = 0.0
            feats_u.append(l15(wu)[-HOP_FRAMES:]); feats_a.append(l15(wa)[-HOP_FRAMES:])

    @torch.no_grad()
    def run(matched):
        sess = EarsSession(0, ep_thr=SETTINGS["endpoint_threshold"], ep_deb=SETTINGS["endpoint_debounce_frames"],
                           barge_thr=SETTINGS["barge_threshold"], barge_deb=SETTINGS["barge_debounce_frames"])
        cache, evs = None, []
        quiet, barged, seed = 0, False, collections.deque(maxlen=SEED_FRAMES)
        for h in range(n_hops):
            mode = "speaking" if speaking_hop[h] else "listen"
            uh = mic[h * HOP_N:(h + 1) * HOP_N]
            ah = agent[h * HOP_N:(h + 1) * HOP_N] if mode == "speaking" else np.zeros(HOP_N, np.float32)
            if rules.begin_hop(sess, uh, ah, mode):
                cache = None
            if rules.switch_mode(sess, mode):
                cache = None; quiet = 0
                if matched and mode == "listen" and barged:
                    # The seed: the turn after a barge starts with the voiced frames of the reply's last 2 s.
                    sess.voiced_ms = float(sum(seed)) * FRAME_S * 1e3
                    sess.in_turn = sess.voiced_ms > 0
                    seed.clear()
                if mode == "listen":
                    barged = False
            act = rules.frame_energy(sess.win_u)[-HOP_FRAMES:]
            for j in range(HOP_FRAMES):
                a = float(act[j])
                x = ((torch.tensor([[a]], device=DEV) + 50) / 25).clamp(-3, 3)
                prev = torch.tensor([[sess.prev_state]], dtype=torch.long, device=DEV)
                o_, cache = tt(feats_u[h][j][None, None], feats_a[h][j][None, None], x, prev, cache)
                events, final, reset = rules.on_frame(sess, mode, o_[0, 0].cpu().numpy(), a,
                                                      left=HOP_FRAMES - 1 - j)
                for ev in events:
                    if ev["kind"] == "ENDPOINT":
                        evs.append({"kind": "endpoint", "by": "model", "hop": h, "frame": j,
                                    "p": ev.get("p_end")})
                    elif ev["kind"] == "BARGE":
                        if matched and barged:
                            continue        # the latch: one barge per reply
                        barged = True
                        evs.append({"kind": "barge", "by": "", "hop": h, "frame": j,
                                    "p": ev.get("yield_prob"), "onset_s": ev.get("onset_s")})
                if mode == "speaking":
                    seed.append(a > VAD_DB)
                elif final is not None:
                    quiet = 0
                elif matched and sess.in_turn and sess.voiced_ms >= MIN_SPEECH_MS:
                    quiet = 0 if a > VAD_DB else quiet + 1
                    if quiet >= FB_FRAMES:
                        fallback_endpoint(sess); reset = True; quiet = 0
                        evs.append({"kind": "endpoint", "by": "silence", "hop": h, "frame": j, "p": None})
                if reset:
                    cache = None
        return evs

    return {"n_hops": n_hops, "offset": o, "speaking": speaking_hop,
            "matched": run(True), "upstream": run(False)}


result = {"device": DEV, "torch": torch.__version__, "stems": {}}
for stem in STEMS:
    result["stems"][stem] = stem_reference(stem)
    print(f"reference done: {stem}", flush=True)
Path(OUT).write_text(json.dumps(result))
'''


# --- paths ---------------------------------------------------------------------------

def _main_checkout() -> Path | None:
    """The main checkout when this one is a git worktree, else None."""
    try:
        common = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    main = Path(common).parent
    return None if main == REPO else main


def _local(rel: str) -> Path:
    """rel in this checkout, else in the main checkout of a worktree."""
    here = REPO / rel
    if here.exists():
        return here
    main = _main_checkout()
    if main is not None and (main / rel).exists():
        return main / rel
    return here


# --- the reference ---------------------------------------------------------------------

def run_reference(args, settings: dict, stems: list[str]) -> dict:
    with tempfile.TemporaryDirectory(prefix="tyger-fidelity-") as tmp:
        script = Path(tmp) / "reference.py"
        script.write_text(REFERENCE)
        out = Path(tmp) / "reference.json"
        cmd = [str(args.spike_py), "-I", str(script), str(args.tyger_repo), str(args.bundle),
               json.dumps(settings), str(out), *stems]
        print(f"reference: {args.spike_py} (upstream rules from {args.tyger_repo})", flush=True)
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            raise SystemExit(f"the reference run failed with exit code {proc.returncode}")
        return json.loads(out.read_text())


# --- the server path --------------------------------------------------------------------

def spans_from_modes(speaking: list[bool], offset: int) -> list[tuple[int, int]]:
    """Each run of speaking hops as a played span [start, end) on the mic stream.

    The start is the run's first hop start, moved by one sample when needed so that
    (start - offset) is even: the reply's 16 kHz grid then falls on the output's own
    (a 24 kHz reply resamples 3 to 2), and the span still covers the run's first hop.
    """
    spans, h = [], 0
    while h < len(speaking):
        if not speaking[h]:
            h += 1
            continue
        first = h
        while h < len(speaking) and speaking[h]:
            h += 1
        start = first * HOP
        if (start - offset) % 2:
            start += 1
        spans.append((start, h * HOP))
    return spans


def reply_pcm(out24: np.ndarray, rate: int, offset: int, start: int, end: int) -> tuple[bytes, int]:
    """The output timeline at its own rate over the span -> (int16 LE PCM, reply_sample
    of the stop, which cuts it to the span's length at 16 kHz)."""
    length16 = end - start
    up, down = rate, SR
    i0 = (start - offset) * up // down
    n = -(-length16 * up // down)
    pcm = np.zeros(n, np.int16)
    lo, hi = max(i0, 0), min(i0 + n, len(out24))
    if hi > lo:
        pcm[lo - i0:hi - i0] = out24[lo:hi]
    # reply_sample: the stop frame's count at the reply's rate whose 16 kHz cut is the span.
    reply_sample = n
    while (reply_sample * SR + rate // 2) // rate > length16:
        reply_sample -= 1
    return pcm.astype("<i2").tobytes(), reply_sample


def build_hops(stem: str, ref: dict) -> tuple[list, list[str], np.ndarray]:
    """Drive the server's HopFeed with the call -> (packed hops, their modes, the
    harness's agent channel as int16 for the comparison)."""
    meta = json.loads(Path(f"{stem}-capture.json").read_text())
    mic, sr = sf.read(f"{stem}-mic.wav", dtype="int16")
    if sr != SR:
        raise SystemExit(f"{stem}-mic.wav is {sr} Hz, not {SR}")
    out24, rate = sf.read(f"{stem}-out.wav", dtype="int16")
    if mic.ndim != 1 or out24.ndim != 1:
        raise SystemExit(f"{stem}: mono WAVs expected")
    offset = int(round((meta["out"]["first_dac_time"] - meta["mic"]["first_adc_time"]) * SR))
    if offset != ref["offset"]:
        raise SystemExit(f"{stem}: output offset {offset} differs from the reference's {ref['offset']}")
    n_hops = ref["n_hops"]
    spans = spans_from_modes(ref["speaking"], offset)

    # The client's script: (mic sample at which it happens, order, action).
    actions = []
    for k, (start, end) in enumerate(spans):
        rid = f"reply-{k}"
        pcm, reply_sample = reply_pcm(out24, rate, offset, start, end)
        actions.append((max(0, start - SEND_AHEAD), 0, ("send", rid, pcm, rate)))
        actions.append((start + EDGE_LAG, 1, ("playback", {
            "type": "playback", "request_id": rid, "event": "start", "mic_sample": start})))
        actions.append((end + EDGE_LAG, 2, ("playback", {
            "type": "playback", "request_id": rid, "event": "stop", "mic_sample": end,
            "reply_sample": reply_sample})))
    actions.sort(key=lambda a: (a[0], a[1]))

    feed = HopFeed(label=Path(stem).name)
    pcm_mic = mic.astype("<i2").tobytes()
    jobs, next_action = [], 0
    for pos in range(0, len(mic), MIC_FRAME):
        while next_action < len(actions) and actions[next_action][0] <= pos:
            act = actions[next_action][2]
            next_action += 1
            if act[0] == "send":
                _, rid, pcm, r = act
                feed.on_tts_text(rid, json.dumps({"type": "audio_start", "request_id": rid,
                                                  "sample_rate": r}))
                for i in range(0, len(pcm), 9600):
                    feed.on_tts_pcm(rid, pcm[i:i + 9600])
                feed.on_tts_text(rid, json.dumps({"type": "audio_end", "request_id": rid}))
            else:
                refused = feed.playback(act[1])
                if refused:
                    raise SystemExit(f"{stem}: the hop feed refused {act[1]}: {refused}")
        feed.feed_mic(pcm_mic[2 * pos:2 * (pos + MIC_FRAME)])
        jobs.extend(feed.due_hops())
    jobs = jobs[:n_hops]
    if len(jobs) != n_hops:
        raise SystemExit(f"{stem}: the hop feed cut {len(jobs)} hops, the reference {n_hops}")
    hops = [_pack_hop(job) for job in jobs]
    modes = [job.mode for job in jobs]

    # The harness's agent channel (the whole output resampled, placed, zero in
    # listening hops), as int16, to compare with the packed agent channel.
    out16 = resample_poly(out24.astype(np.float64) / 32768.0, SR, rate) if rate != SR \
        else out24.astype(np.float64) / 32768.0
    agent = np.zeros(n_hops * HOP)
    seg = out16[:max(0, len(agent) - offset)]
    agent[offset:offset + len(seg)] = seg
    for h, speaking in enumerate(ref["speaking"]):
        if not speaking:
            agent[h * HOP:(h + 1) * HOP] = 0.0
    agent16 = np.clip(np.rint(agent * 32768.0), -32768, 32767).astype(np.int16)
    return hops, modes, agent16


# --- the sidecar ------------------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class GpuPoll(threading.Thread):
    """Polls nvidia-smi for one PID's GPU memory; ``peak_mib`` is the most seen."""

    def __init__(self, pid: int, interval: float = 0.1) -> None:
        super().__init__(daemon=True)
        self.pid, self.interval = pid, interval
        self.peak_mib = 0.0
        self.samples = 0
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                text = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout
            except (OSError, subprocess.TimeoutExpired):
                text = ""
            for line in text.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0] == str(self.pid):
                    try:
                        self.peak_mib = max(self.peak_mib, float(parts[1]))
                        self.samples += 1
                    except ValueError:
                        pass
            self._halt.wait(self.interval)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=15)


def start_sidecar(args, port: int) -> subprocess.Popen:
    cmd = [str(args.tyger_py), str(SERVER_SCRIPT), "--port", str(port), "--device", args.device,
           "--model", args.model, "--log-level", "warning"]
    if args.revision:
        cmd += ["--revision", args.revision]
    for flag, field, _kind, _text in RULE_FLAGS:
        cmd += [flag, str(getattr(args, field))]
    print("sidecar: " + " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL)


def wait_ready(proc: subprocess.Popen, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"the sidecar exited with code {proc.returncode} before it was ready")
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    print(f"sidecar ready: {resp.read().decode()}", flush=True)
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    raise SystemExit(f"the sidecar was not ready within {timeout:.0f} s")


def stop_sidecar(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


async def play_call(port: int, label: str, hops: list[bytes]) -> tuple[list, list[float]]:
    """Send each hop and a probe; -> (events as (hop index, event), hop times in ms)."""
    events, times = [], []
    url = f"ws://127.0.0.1:{port}/tyger?session={label}"
    async with websockets.asyncio.client.connect(url, max_size=None) as ws:
        for h, hop in enumerate(hops):
            t0 = time.perf_counter()
            await ws.send(hop)
            await ws.send(PROBE)
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=60)
                parsed = wire.parse_events(msg)
                if (len(parsed) == 1 and isinstance(parsed[0], wire.ErrorEvent)
                        and parsed[0].message.startswith(PROBE_ANSWER)):
                    times.append((time.perf_counter() - t0) * 1000.0)
                    break
                events.extend((h, ev) for ev in parsed)
    return events, times


# --- the comparison ---------------------------------------------------------------------

def _kind(ev: dict) -> str:
    return ev["kind"] + (f"/{ev['by']}" if ev["by"] else "")


def sidecar_events(raw: list) -> tuple[list[dict], list[str]]:
    out, errors = [], []
    for h, ev in raw:
        if isinstance(ev, wire.EndpointEvent):
            out.append({"kind": "endpoint", "by": ev.by, "hop": (ev.mic_sample - 1) // HOP,
                        "mic_sample": ev.mic_sample, "sent_hop": h})
        elif isinstance(ev, wire.BargeEvent):
            out.append({"kind": "barge", "by": "", "hop": (ev.mic_sample - 1) // HOP,
                        "mic_sample": ev.mic_sample, "onset": ev.onset_mic_sample, "sent_hop": h})
        else:
            errors.append(f"hop {h}: {ev.message}")
    return out, errors


def match(ref: list[dict], got: list[dict]) -> tuple[list, list, list]:
    """One-to-one, in time order, same kind, within MATCH_HOPS hops."""
    used, pairs, missing = set(), [], []
    for r in ref:
        best = None
        for i, g in enumerate(got):
            if i in used or _kind(g) != _kind(r):
                continue
            d = abs(g["hop"] - r["hop"])
            if d <= MATCH_HOPS and (best is None or d < abs(got[best]["hop"] - r["hop"])):
                best = i
        if best is None:
            missing.append(r)
        else:
            used.add(best)
            pairs.append((r, got[best]))
    extra = [g for i, g in enumerate(got) if i not in used]
    return pairs, missing, extra


def _t(hop: int) -> str:
    return f"{(hop + 1) * HOP / SR:7.2f} s"


def report_call(stem: str, ref: dict, modes: list[str], hops: list[bytes], agent16: np.ndarray,
                got: list[dict], errors: list[str], times: list[float]) -> bool:
    name = Path(stem).name
    ok = True
    print(f"\n== {name}: {ref['n_hops']} hops, output at mic sample {ref['offset']}")
    want = [wire.MODE_SPEAKING if s else wire.MODE_LISTENING for s in ref["speaking"]]
    bad_modes = [h for h, (a, b) in enumerate(zip(modes, want)) if a != b]
    agent = np.concatenate([np.frombuffer(wire.unpack_hop(hop).agent, dtype="<i2") for hop in hops])
    diff = np.abs(agent.astype(np.int32) - agent16.astype(np.int32))
    worst = int(diff.argmax()) if diff.size else 0
    print(f"hop modes: {sum(m == wire.MODE_SPEAKING for m in modes)} speaking, "
          f"{len(bad_modes)} differ from the reference" + (f" (hops {bad_modes[:10]})" if bad_modes else ""))
    print(f"agent channel vs the harness's placement: max |diff| {int(diff.max()) if diff.size else 0} "
          f"LSB at mic sample {worst}, mean {float(diff.mean()) if diff.size else 0:.3f} LSB")
    if bad_modes:
        ok = False
    pairs, missing, extra = match(ref["matched"], got)
    for r, g in pairs:
        print(f"  match   {_kind(r):17s} ref {_t(r['hop'])} (p {r['p']})  sidecar {_t(g['hop'])}"
              + (f"  [hop {g['hop'] - r['hop']:+d}]" if g["hop"] != r["hop"] else ""))
    for r in missing:
        print(f"  MISSING {_kind(r):17s} ref {_t(r['hop'])} (p {r['p']})")
    for g in extra:
        print(f"  EXTRA   {_kind(g):17s} sidecar {_t(g['hop'])}")
    for e in errors:
        print(f"  ERROR   {e}")
    up = ref["upstream"]
    print(f"events: {len(pairs)} matched, {len(extra)} extra, {len(missing)} missing, "
          f"{len(errors)} errors (reference {len(ref['matched'])}; upstream rules alone "
          f"{sum(e['kind'] == 'endpoint' for e in up)} endpoints, "
          f"{sum(e['kind'] == 'barge' for e in up)} barges)")
    if missing or extra or errors:
        ok = False
    med = statistics.median(times) if times else float("nan")
    p90 = sorted(times)[int(0.9 * (len(times) - 1))] if times else float("nan")
    print(f"hop time: median {med:.1f} ms, p90 {p90:.1f} ms, max {max(times):.1f} ms")
    return ok


# --- main -------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    from voicemanager.config import Settings
    fields = Settings.model_fields
    defaults = rules.TurnRuleSettings()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("calls", nargs="+", help="capture prefixes (<prefix>-mic.wav and so on)")
    p.add_argument("--tyger-py", type=Path, default=_local("venv-tyger/bin/python"),
                   help="the sidecar environment's python (default venv-tyger/bin/python)")
    p.add_argument("--spike-py", type=Path, default=_local("3rdParty/tyger-spike/.venv/bin/python"),
                   help="the spike environment's python, for the reference")
    p.add_argument("--tyger-repo", type=Path, default=_local("3rdParty/tyger-spike/tyger-s2s"),
                   help="the tyger-s2s checkout whose turn rules the reference runs")
    p.add_argument("--model", default=fields["tyger_model"].default)
    p.add_argument("--revision", default=fields["tyger_revision"].default)
    p.add_argument("--device", default="cuda:0",
                   help="the sidecar's device, counted within CUDA_VISIBLE_DEVICES (default cuda:0)")
    p.add_argument("--ready-timeout", type=float, default=300.0)
    for flag, field, kind, text in RULE_FLAGS:
        p.add_argument(flag, dest=field, type=kind, default=getattr(defaults, field),
                       help=f"{text} (default {getattr(defaults, field)})")
    return p


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = build_parser().parse_args()
    for path, what in ((args.tyger_py, "the sidecar python"), (args.spike_py, "the spike python"),
                       (args.tyger_repo, "the tyger-s2s checkout")):
        if not path.exists():
            raise SystemExit(f"{what} is not at {path}")
    stems = [str(Path(c).resolve()) for c in args.calls]
    for stem in stems:
        for suffix in ("-mic.wav", "-out.wav", "-capture.json"):
            if not Path(stem + suffix).exists():
                raise SystemExit(f"missing {stem}{suffix}")
    args.bundle = scorer.resolve_bundle(args.model, args.revision)
    settings = {field: getattr(args, field) for _f, field, _k, _t in RULE_FLAGS}
    settings["barge_seed_ms"] = rules.TurnRuleSettings().barge_seed_ms
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
          f"CUDA_DEVICE_ORDER={os.environ.get('CUDA_DEVICE_ORDER', '')}")
    print(f"rule settings: {settings}")
    print(f"bundle: {args.bundle}")

    reference = run_reference(args, settings, stems)
    print(f"reference ran on {reference['device']} with torch {reference['torch']}")
    calls = []
    for stem in stems:
        ref = reference["stems"][stem]
        hops, modes, agent16 = build_hops(stem, ref)
        calls.append((stem, ref, hops, modes, agent16))

    port = free_port()
    proc = start_sidecar(args, port)
    poll = GpuPoll(proc.pid)
    poll.start()
    all_ok, all_times = True, []
    try:
        wait_ready(proc, port, args.ready_timeout)
        for stem, ref, hops, modes, agent16 in calls:
            raw, times = asyncio.run(play_call(port, Path(stem).name, hops))
            got, errors = sidecar_events(raw)
            all_times.extend(times)
            all_ok &= report_call(stem, ref, modes, hops, agent16, got, errors, times)
    finally:
        poll.stop()
        stop_sidecar(proc)

    med = statistics.median(all_times)
    peak_gib = poll.peak_mib / 1024.0
    print(f"\n== all calls: {len(all_times)} hops, median hop time {med:.1f} ms "
          f"(bound {MAX_HOP_MS:.0f} ms); sidecar peak GPU memory {poll.peak_mib:.0f} MiB = "
          f"{peak_gib:.2f} GiB (bound {MAX_GPU_MIB / 1024:.0f} GiB; nvidia-smi, PID {proc.pid}, "
          f"{poll.samples} samples)")
    if not med < MAX_HOP_MS:
        print("FAIL: the median hop time is over the bound")
        all_ok = False
    if poll.samples == 0:
        print("FAIL: nvidia-smi never showed the sidecar's PID")
        all_ok = False
    elif not poll.peak_mib < MAX_GPU_MIB:
        print("FAIL: the sidecar's peak GPU memory is over the bound")
        all_ok = False
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
