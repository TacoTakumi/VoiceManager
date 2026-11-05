"""Fidelity and speed check of the experimental pre-filter on captured calls.

A manual script: it needs the GPU, the sidecar environment (venv-tyger, with
onnxruntime-gpu), the Tyger bundle and the VoxLock models in the Hugging Face cache
(``python setup_models.py tyger`` with VM_TYGER_PREFILTER=voxlock). It is not part
of the automated suite. Run it with the sidecar's python:

    venv-tyger/bin/python scripts/tyger_prefilter_check.py [--device cuda:3] CALL...

Each CALL is a capture prefix of ``scripts/voice_agent.py ... --capture`` (the files
<prefix>-mic.wav, <prefix>-out.wav and <prefix>-capture.json).

For each call the mic is cut into 320 ms hops. The output is placed on the mic's
clock at (first DAC time - first ADC time), and a hop is in speaking mode while the
placed output had an 80 ms frame above -60 dB in the last 0.8 s, as in the offline
replay harness. Every hop goes through the sidecar's own session code (the turn
rules, the pre-filter wrapper with self-enrollment, the real Tyger scorer and the
real VoxLock filter), in order, on one thread, as the sidecar runs it.

The user's speech is marked on 80 ms frames in two ways: the raw mic more than 9 dB
over the echo the placed output predicts at the call's median echo return, or
VoxLock's own output over the frame above -35 dB. A mid-reply span is a run of at
least 4 such frames while the output still plays. Printed per call: each mid-reply
span with its highest debounced yield (the lower of two frames in a row) in the
span and the 0.4 s after it, each barge with whether user speech was marked in the
0.6 s before it, when the session enrolled, the per-hop time and the GPU memory of
this process. The checks: on calls with mid-reply spans every span reaches 0.6 or
more (the sidecar's barge threshold), and no barge falls outside user speech.
"""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[1]
TYGER_DIR = REPO / "src" / "voicemanager" / "pipeline" / "tyger"
SR, HOP, FRAME = 16000, 5120, 1280
BRIDGE_FRAMES = 10                  # 0.8 s of 80 ms frames
FLOOR_DB, GATE_DB = -52.0, -50.0


def load_sidecar_package():
    """The sidecar directory as the package "tyger", as server.py loads itself."""
    spec = importlib.util.spec_from_file_location(
        "tyger", TYGER_DIR / "__init__.py", submodule_search_locations=[str(TYGER_DIR)])
    sys.modules["tyger"] = pkg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pkg)
    import tyger.prefilter, tyger.rules, tyger.scorer, tyger.server, tyger.voxlock, tyger.wire  # noqa
    return sys.modules["tyger"]


def db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)) + 1e-12)


def frame_db(x):
    n = len(x) // FRAME
    f = x[:n * FRAME].reshape(n, FRAME).astype(np.float64)
    return 20 * np.log10(np.sqrt(np.mean(f ** 2, axis=1)) + 1e-12)


class RecordingScorer:
    """Keeps every frame's yield, in order, of the scorer it wraps."""

    def __init__(self, inner):
        self.inner, self.yields = inner, []

    def reset(self):
        self.inner.reset()

    def score_hop(self, user, agent, mode):
        for f in self.inner.score_hop(user, agent, mode):
            self.yields.append(f.yield_prob)
            yield f


class RecordingFactory:
    """Enrolls the session's filter and keeps the sample it enrolled on."""

    def __init__(self, factory):
        self.factory, self.sample = factory, None

    def enroll(self, sample):
        self.sample = np.array(sample, np.float32)
        return self.factory.enroll(sample)


def call(tg, tyger_scorer, factory, stem):
    name = os.path.basename(stem)
    meta = json.loads(Path(f"{stem}-capture.json").read_text())
    mic, sr = sf.read(f"{stem}-mic.wav", dtype="float32")
    assert sr == SR
    out, osr = sf.read(f"{stem}-out.wav", dtype="float32")
    out16 = resample_poly(out, SR, osr).astype(np.float32) if osr != SR else out
    off = int(round((meta["out"]["first_dac_time"] - meta["mic"]["first_adc_time"]) * SR))
    agent = np.zeros_like(mic)
    seg = out16[:max(0, len(mic) - off)]
    agent[off:off + len(seg)] = seg
    n_hops = len(mic) // HOP
    ad = frame_db(agent)
    voiced_a = ad > -60
    speaking = [bool(voiced_a[max(0, (h + 1) * 4 - BRIDGE_FRAMES):(h + 1) * 4].any())
                for h in range(n_hops)]

    rec_factory = RecordingFactory(factory)
    rec = RecordingScorer(tyger_scorer.new_session(name, 0))
    session = tg.server.Session(name, tg.rules.TurnRules(tg.rules.TurnRuleSettings()),
                                tg.prefilter.PrefilterSessionScorer(rec, rec_factory, FLOOR_DB,
                                                                    GATE_DB, key=name))
    to_pcm = lambda x: np.clip(np.round(x * 32768), -32768, 32767).astype("<i2").tobytes()
    barges, times, enrolled_at = [], [], None
    for h in range(n_hops):
        mode = tg.wire.MODE_SPEAKING if speaking[h] else tg.wire.MODE_LISTENING
        a = agent[h * HOP:(h + 1) * HOP] if speaking[h] else np.zeros(HOP, np.float32)
        hop = tg.wire.unpack_hop(tg.wire.pack_hop(mode, (h + 1) * HOP, to_pcm(mic[h * HOP:(h + 1) * HOP]),
                                                  to_pcm(a)))
        t0 = time.perf_counter()
        events = tg.server.process_hop(session, hop)
        times.append(time.perf_counter() - t0)
        if enrolled_at is None and session.scorer.enrolled:
            enrolled_at = (h + 1) * HOP / SR
        barges += [e.mic_sample / SR for e in events if isinstance(e, tg.wire.BargeEvent)]

    # User speech marking.
    md = frame_db(mic)
    nf = min(len(md), len(ad))
    md, ad = md[:nf], ad[:nf]
    ads = np.maximum.reduce([ad, np.r_[-240, ad[:-1]], np.r_[-240, -240, ad[:-2]]])
    playing = ads > -45
    erl = np.median(md[playing] - ads[playing]) if playing.any() else 0.0
    pred = np.where(ads > -60, ads + erl, -240)
    by_energy = md > np.maximum(pred + 9, -40)
    by_filter = np.zeros(nf, bool)
    if rec_factory.sample is not None:
        s = factory.enroll(rec_factory.sample)
        y = np.concatenate([s.push(mic[i:i + HOP]) for i in range(0, len(mic), HOP)])
        yd = frame_db(np.r_[y, np.zeros(max(0, len(mic) - len(y)), np.float32)])[:nf]
        by_filter = yd > -35
    user = by_energy | by_filter

    ys = np.array(rec.yields)
    deb = np.minimum(ys, np.r_[0, ys[:-1]])
    spans, i = [], 0
    while i < nf:
        if user[i]:
            j, last = i, i
            while j < nf and j - last <= 2:
                if user[j]:
                    last = j
                j += 1
            if last - i + 1 >= 4 and np.median(ads[i:last + 1]) > -45:
                spans.append((i, last + 1))
            i = j
        else:
            i += 1
    print(f"== {name}: {n_hops} hops, enrolled at "
          f"{'never' if enrolled_at is None else f'{enrolled_at:.2f} s'}")
    span_ok = True
    for a0, a1 in spans:
        w = deb[a0:min(len(deb), a1 + 5)]
        peak = float(w.max()) if w.size else float("nan")
        span_ok &= peak >= 0.6
        print(f"   mid-reply span {a0 * 0.08:6.2f}-{a1 * 0.08:6.2f} s  highest yield {peak:.3f}")
    outside = []
    for b in barges:
        k = int(b / 0.08)
        if not user[max(0, k - 8):k + 1].any():
            outside.append(b)
    print(f"   barges: {' '.join(f'{b:.2f}' for b in barges) or 'none'}")
    print(f"   barges outside user speech: {' '.join(f'{b:.2f}' for b in outside) or 'none'}")
    t = np.array(times[5:]) * 1000
    print(f"   per hop: median {np.median(t):.1f} ms, p95 {np.percentile(t, 95):.1f} ms")
    return spans, span_ok, outside, list(t)


def gpu_mib():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return sum(int(m) for p, m in (l.split(", ") for l in out.strip().splitlines() if l)
               if int(p) == os.getpid())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("calls", nargs="+")
    parser.add_argument("--device", default="cuda:3")
    args = parser.parse_args()
    tg = load_sidecar_package()
    tyger_scorer = tg.scorer.TygerScorer.load("nineninesix/tyger",
                                             "a5d4d73bb74fa4a7082dcd4387e44a1e5ac151b6", args.device)
    factory = tg.voxlock.VoxLockFactory.load(args.device)
    print(f"VoxLock providers: {factory.chunk_session.get_providers()[:1]} on {args.device}")
    all_t, failed = [], []
    for stem in args.calls:
        spans, ok, outside, t = call(tg, tyger_scorer, factory, stem)
        all_t += t
        if spans and not ok:
            failed.append(f"{os.path.basename(stem)}: a mid-reply span under 0.6")
        if outside:
            failed.append(f"{os.path.basename(stem)}: barge outside user speech")
    mem = gpu_mib()
    med = float(np.median(all_t))
    print(f"\nall calls: per-hop median {med:.1f} ms; GPU memory of this process {mem} MiB")
    if med >= 40:
        failed.append(f"per-hop median {med:.1f} ms is not under 40 ms")
    if mem >= 4096:
        failed.append(f"GPU memory {mem} MiB is not under 4 GiB")
    print("PASS" if not failed else "FAIL: " + "; ".join(failed))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
