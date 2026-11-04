"""Scorers for the Tyger sidecar: per-frame turn scores for one session's hops.

A scorer turns each 320 ms hop of a session (the user's mic and the agent's played
reply) into four 80 ms frames of scores: the endpoint head's ``p_end``, the yield
head's probability and the frame's energy in dB. The sidecar feeds them to that
session's turn rules (``rules.TurnRules``).

The interface, per session (``Scorer.new_session`` makes one)::

    reset()                          the model's context starts over before the
                                     next frame scored
    score_hop(user, agent, mode)     -> an iterator over the hop's 4 FrameScores

``user`` and ``agent`` are the hop's two channels as little-endian int16 PCM bytes
(5120 samples each) and ``mode`` is ``listening`` or ``speaking``. The frames are
scored lazily, one per step of the iterator, so a ``reset()`` the caller makes
between two frames (the turn rules ask for one when a turn ends) applies to the
frames after it.

Two scorers:

* ``FakeScorer``: scores read from a script, per session. Standard library only.
* ``TygerScorer``: the real model, nemotron-3.5-asr's encoder to layer 15 on 8 s
  rolling windows and the turn-taker GRU from the Tyger bundle. It imports torch,
  transformers and numpy only when it loads, so this module imports with the
  standard library alone.

This module is loaded through the tyger package (``rules`` and ``wire`` are
imported relatively); see the recipe in ``rules``.
"""
from __future__ import annotations

import array
import hashlib
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Protocol

from .rules import FRAME_SAMPLES, HOP_FRAMES
from .wire import HOP_SAMPLES, MODE_LISTENING, MODE_SPEAKING, SAMPLE_RATE

logger = logging.getLogger(__name__)


class FrameScore(NamedTuple):
    """One 80 ms frame's scores."""
    p_end: float
    yield_prob: float
    energy_db: float


class SessionScorer(Protocol):
    """One session's scorer state."""

    def reset(self) -> None:
        """The model's context starts over before the next frame is scored."""

    def score_hop(self, user: bytes, agent: bytes, mode: str) -> Iterator[FrameScore]:
        """The hop's four frames, scored one per step of the iterator."""


class Scorer(Protocol):
    """Makes one SessionScorer per session. ``name`` is shown by /health."""
    name: str

    def new_session(self, key: str, index: int) -> SessionScorer:
        """A scorer for the session labelled ``key``, the ``index``-th to connect."""


class ScorerError(RuntimeError):
    """A scorer failed on a hop."""


def frame_energy_db(pcm: bytes) -> list[float]:
    """Per-frame energy in dB of little-endian int16 PCM, as the turn rules read it:
    20 log10(rms + 1e-9) on samples scaled to -1..1 (the formula of the real
    scorer's ``_frame_energy``, in plain Python)."""
    values = array.array("h", pcm)
    if sys.byteorder != "little":
        values.byteswap()
    out = []
    for start in range(0, len(values) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        frame = values[start:start + FRAME_SAMPLES]
        mean_sq = sum(v * v for v in frame) / (FRAME_SAMPLES * 32768.0 * 32768.0)
        out.append(20.0 * math.log10(math.sqrt(mean_sq + 1e-12) + 1e-9))
    return out


# --- the fake scorer -------------------------------------------------------------

class FakeScorer:
    """Scores from a script, for tests: no model, standard library only.

    The script is a JSON object ``{"sessions": {KEY: [ENTRY, ...]}}``. A session's
    key is the label it connects with (``/tyger?session=KEY``), else its connection
    order as a string ("0" for the first). Its entries are consumed one per frame,
    from the first, across hops; resets do not rewind them. An entry is one of

    * ``[p_end, yield, energy_db]``, or ``{"p_end": .., "yield": .., "energy_db": ..}``;
      an ``energy_db`` of null takes the frame's energy from the hop's user audio;
    * ``{"error": "message"}``: the scorer raises ScorerError with the message when
      it reaches this entry (which it consumes).

    Past the end of its entries, or with no entries under its key, a session scores
    ``p_end`` 0, yield 0 and the audio's own energy.
    """
    name = "fake"

    def __init__(self, streams: dict[str, list]) -> None:
        if not isinstance(streams, dict):
            raise ValueError("the fake scorer's sessions must be a JSON object")
        self._streams = {str(key): [_entry(e, f"{key}[{i}]") for i, e in enumerate(entries)]
                         for key, entries in streams.items()}

    @classmethod
    def from_file(cls, path) -> "FakeScorer":
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read the fake scorer script {path}: {exc}") from None
        if not isinstance(data, dict) or not isinstance(data.get("sessions", {}), dict):
            raise ValueError(f"fake scorer script {path}: expected {{'sessions': {{...}}}}")
        return cls(data.get("sessions", {}))

    def new_session(self, key: str, index: int) -> "FakeSessionScorer":
        return FakeSessionScorer(self._streams.get(key, []))


def _entry(raw, where: str):
    """One script entry, checked: a FrameScore with energy_db None for "from the
    audio", or an error message string."""
    if isinstance(raw, dict) and "error" in raw:
        if not isinstance(raw["error"], str):
            raise ValueError(f"fake scorer entry {where}: error must be a string")
        return raw["error"]
    if isinstance(raw, dict):
        raw = [raw.get("p_end", 0.0), raw.get("yield", 0.0), raw.get("energy_db")]
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"fake scorer entry {where}: expected [p_end, yield, energy_db]")
    p_end, yld, db = raw
    for value in (p_end, yld) + ((db,) if db is not None else ()):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"fake scorer entry {where}: scores must be numbers, got {raw}")
    return FrameScore(float(p_end), float(yld), None if db is None else float(db))


class FakeSessionScorer:
    """One session's position in its scripted stream. ``resets`` counts reset()."""

    def __init__(self, entries: list) -> None:
        self._entries = entries
        self._pos = 0
        self.resets = 0
        self.hops = 0

    def reset(self) -> None:
        self.resets += 1

    def score_hop(self, user: bytes, agent: bytes, mode: str) -> Iterator[FrameScore]:
        self.hops += 1
        return self._frames(user)

    def _frames(self, user: bytes) -> Iterator[FrameScore]:
        energy: list[float] | None = None
        for j in range(HOP_FRAMES):
            entry = self._entries[self._pos] if self._pos < len(self._entries) else None
            self._pos += 1
            if isinstance(entry, str):
                raise ScorerError(entry)
            if entry is None:
                entry = FrameScore(0.0, 0.0, None)
            if entry.energy_db is None:
                if energy is None:
                    energy = frame_energy_db(user)
                entry = entry._replace(energy_db=energy[j])
            yield entry


# --- the real scorer -------------------------------------------------------------
#
# The computation follows the offline replay the turn model was checked with, so
# that the sidecar's events match its traces: per hop, the
# trailing 8 s of the user's audio and of the agent channel (zeros for hops sent in
# listening mode) are encoded by nemotron's processor and encoder to layer 15, the
# hop's newest 4 frames are kept, and the GRU steps once per frame on the user and
# agent features, the frame's energy normalised as clip((dB + 50) / 25, -3, 3) and
# the previous state, LISTEN while listening and SPEAK while speaking. The rolling
# windows run across resets; a reset zeroes only the GRU's state.

WINDOW_S = 8.0
WINDOW_SAMPLES = int(WINDOW_S * SAMPLE_RATE)
L15 = 15                 # the encoder layer the turn-taker reads (of 24)
ENC_HIDDEN = 1024
LISTEN, SPEAK = 0, 2     # the turn-taker's state ids for the previous-state input
N_STATE = 4
STATE_NAMES = ("LISTEN", "HOLD", "SPEAK", "BACKCHANNEL")
YIELD_INDEX, P_END_INDEX = 5, 7   # in the GRU's 8 outputs
BUNDLE_FILES = ["config.json", "model.safetensors", "nemotron/*.json"]
MODEL_TYPE = "tyger_nemotron_gru"
_DEVICE_RE = re.compile(r"cpu|cuda(:\d+)?")


def check_device(device: str) -> str:
    """``cpu``, ``cuda`` or ``cuda:N``; ValueError otherwise."""
    if not isinstance(device, str) or not _DEVICE_RE.fullmatch(device):
        raise ValueError(f"device must be cpu, cuda or cuda:N, got {device!r}")
    return device


def resolve_bundle(model: str, revision: str | None = None) -> Path:
    """The bundle's directory: ``model`` itself if it is one, else its copy in the
    Hugging Face cache at ``revision``. Nothing is downloaded: the bundle must be
    fetched first (``python setup_models.py tyger``)."""
    path = Path(model).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download
    try:
        return Path(snapshot_download(repo_id=model, revision=revision,
                                      allow_patterns=BUNDLE_FILES, local_files_only=True))
    except Exception as exc:
        raise RuntimeError(
            f"the Tyger bundle {model} at {revision or 'main'} is not in the Hugging Face "
            f"cache ({exc}); fetch it with 'python setup_models.py tyger'") from exc


# Adapted from tyger-s2s app/plugins/ears/tyger/bundle.py (check_config), commit
# 97d011d, Apache-2.0; see LICENSE and NOTICE. The turn-rule guards (vad_db,
# min_speech_ms, warmup_frames, bc_refractory_frames) are not compared: here the
# rules take their values from the sidecar's settings.
def check_config(config: dict) -> None:
    """The bundle must be the GRU format and the audio grid this scorer runs;
    ValueError naming what differs."""
    if config.get("model_type") != MODEL_TYPE or config.get("format_version") != 1:
        raise ValueError(f"not a {MODEL_TYPE} bundle of format 1 "
                         f"(model_type {config.get('model_type')!r})")
    try:
        arch = config["controller"]["architecture"]
        fe = config["frontend"]
        want = {
            "dtype": (config["dtype"], "float32"),
            "controller": (arch["kind"], "gru"),
            "state_names": (tuple(config["state_names"]), STATE_NAMES),
            "output_order": (tuple(config["output_order"]),
                             ("state_logits", "hazard_logits", "yield_logits", "bc_logits",
                              "endpoint_logits")),
            "live previous state": (config["streaming"]["live_previous_state"],
                                    {"listen": LISTEN, "speaking": SPEAK}),
            "layer": (fe["layer"], L15),
            "hidden_size": (fe["hidden_size"], ENC_HIDDEN),
            "sample_rate": (fe["sample_rate"], SAMPLE_RATE),
            "frame_period_s": (fe["frame_period_s"], FRAME_SAMPLES / SAMPLE_RATE),
            "live_window_s": (fe["live_window_s"], WINDOW_S),
            "live_hop_s": (fe["live_hop_s"], HOP_SAMPLES / SAMPLE_RATE),
            "num_lookahead_tokens": (fe["num_lookahead_tokens"], 0),
            "energy_norm": (fe["energy_norm"], "clip((dB + 50) / 25, -3, 3)"),
        }
    except (KeyError, TypeError) as exc:
        raise ValueError(f"the bundle's config.json lacks {exc}") from None
    bad = [f"{k}: bundle {got!r}, scorer {exp!r}" for k, (got, exp) in want.items() if got != exp]
    if bad:
        raise ValueError("the bundle needs another frontend than this scorer runs: "
                         + "; ".join(bad))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _turn_taker_class():
    """The TurnTaker module class, built on first use so that torch loads lazily.

    Adapted from tyger-s2s app/plugins/ears/tyger/controller.py, commit 97d011d,
    Apache-2.0; see LICENSE and NOTICE. Module names and the order of the sums are
    the bundle's, so its tensors load as they are.
    """
    import torch
    import torch.nn as nn

    def mlp_head(d):
        return nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))

    class TurnTaker(nn.Module):
        """A causal GRU over layer 15 of the user's and the agent's audio."""

        def __init__(self, width=768, layers=3):
            super().__init__()
            d = width
            self.core = nn.GRU(d, d, layers, batch_first=True)
            self.out_norm = nn.LayerNorm(d)
            self.proj_user = nn.Linear(ENC_HIDDEN, d)
            self.proj_agent = nn.Linear(ENC_HIDDEN, d)
            self.proj_energy = nn.Linear(1, d)
            self.state_emb = nn.Embedding(N_STATE + 1, d)
            self.in_norm = nn.LayerNorm(d)
            self.state_head = nn.Linear(d, N_STATE)
            self.hazard_head = nn.Linear(d, 1)
            self.yield_head = mlp_head(d)
            self.bc_head = mlp_head(d)
            self.ep_head = mlp_head(d)

        def fuse(self, h_user, h_agent, act, prev_state):
            return self.in_norm(self.proj_user(h_user) + self.proj_agent(h_agent)
                                + self.proj_energy(act.unsqueeze(-1)) + self.state_emb(prev_state))

        def outputs(self, z):
            """(B, T, d) -> (B, T, 8): state logits x4, then the hazard, yield,
            backchannel and endpoint probabilities."""
            z = self.out_norm(z)
            return torch.cat([self.state_head(z), torch.sigmoid(self.hazard_head(z)),
                              torch.sigmoid(self.yield_head(z)), torch.sigmoid(self.bc_head(z)),
                              torch.sigmoid(self.ep_head(z))], -1)

        def forward(self, h_user, h_agent, act, prev_state, state=None):
            z, state = self.core(self.fuse(h_user, h_agent, act, prev_state), state)
            return self.outputs(z), state

    return TurnTaker


# Adapted from tyger-s2s app/plugins/ears/tyger/bundle.py (load), commit 97d011d,
# Apache-2.0; see LICENSE and NOTICE.
def load_bundle(folder) -> tuple[Any, Any, Any, dict]:
    """-> (nemotron model, its processor, the TurnTaker, config), fp32 on the CPU.
    The weights and nemotron's files must match the checksums in config.json, and
    every tensor must load (strict). Sets TF32 for the process as the bundle's
    config says (it was validated with TF32 off)."""
    import torch
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoProcessor, GenerationConfig

    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text())
    check_config(config)
    for name, digest in config["frontend_file_sha256"].items():
        if _sha256(folder / "nemotron" / name) != digest:
            raise ValueError(f"bundle {folder}: nemotron/{name} does not match its checksum")
    if _sha256(folder / "model.safetensors") != config["weights_sha256"]:
        raise ValueError(f"bundle {folder}: model.safetensors does not match its checksum")
    weights = load_file(str(folder / "model.safetensors"))
    if len(weights) != config["tensor_count"] or any(
            t.dtype != torch.float32 for t in weights.values()):
        raise ValueError(f"bundle {folder}: {len(weights)} tensors "
                         f"(want {config['tensor_count']}), all fp32")
    parts: dict[str, dict] = {"nemotron": {}, "turn_taker": {}}
    for key, value in weights.items():
        prefix, _, name = key.partition(".")
        parts[prefix][name] = value
    del weights
    nemotron = AutoModel.from_config(AutoConfig.from_pretrained(folder / "nemotron"),
                                     dtype=torch.float32)
    nemotron.load_state_dict(parts["nemotron"], strict=True)
    nemotron.generation_config = GenerationConfig.from_pretrained(folder / "nemotron")
    proc = AutoProcessor.from_pretrained(folder / "nemotron")
    arch = config["controller"]["architecture"]
    turn_taker = _turn_taker_class()(arch["width"], arch["layers"])
    turn_taker.load_state_dict(parts["turn_taker"], strict=True)
    torch.backends.cuda.matmul.allow_tf32 = config["tf32"]
    torch.backends.cudnn.allow_tf32 = config["tf32"]
    return nemotron.eval(), proc, turn_taker.eval(), config


class _StopAtL15(Exception):
    """Raised by the layer-15 hook to end an encode there."""


class TygerModel:
    """The loaded model, shared by every session. Not thread-safe: the sidecar
    uses it from its one scoring thread only."""

    def __init__(self, folder, device: str) -> None:
        import numpy as np
        import torch

        self.np, self.torch = np, torch
        self.device = device
        nemotron, self.proc, turn_taker, config = load_bundle(folder)
        self.nemotron = nemotron.to(device).eval()
        self.turn_taker = turn_taker.to(device).eval()
        self.encoder = self.nemotron.encoder
        self._grab: dict[str, Any] = {"on": False, "h": None}
        self.encoder.layers[L15 - 1].register_forward_hook(self._hook)
        self._zero: dict[int, Any] = {}  # window length -> layer 15 of an all-zero window
        logger.info("Tyger bundle %s loaded on %s (GRU %dx%d)", config["weights_sha256"][:8],
                    device, self.turn_taker.core.num_layers, self.turn_taker.core.hidden_size)

    def _hook(self, module, args, out):
        if self._grab["on"]:
            self._grab["h"] = out[0] if isinstance(out, tuple) else out
            raise _StopAtL15

    def l15(self, wav):
        """A float32 16 kHz window -> its layer-15 features (T, 1024) on the device,
        by the processor's own features and the encoder stopped at layer 15."""
        torch = self.torch
        with torch.no_grad():
            inp = self.proc(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            am = inp.get("attention_mask")
            self._grab["on"] = True
            try:
                self.encoder(input_features=inp["input_features"].to(self.device),
                             attention_mask=am.to(self.device) if am is not None else None,
                             num_lookahead_tokens=0)
            except _StopAtL15:
                pass
            finally:
                self._grab["on"] = False
            return self._grab["h"][0].float()

    def l15_agent(self, wav):
        """Like l15; an all-zero window (the agent channel while listening) is
        encoded once per length."""
        if wav.any():
            return self.l15(wav)
        n = len(wav)
        if n not in self._zero:
            self._zero[n] = self.l15(wav)
        return self._zero[n]


# Adapted from tyger-s2s app/plugins/ears/tyger/rules.py (frame_energy), commit
# 97d011d, Apache-2.0; see LICENSE and NOTICE.
def _frame_energy(np, wav):
    """Per-80 ms-frame energy in dB of a float32 window."""
    n = len(wav) // FRAME_SAMPLES
    if n == 0:
        return np.zeros(0, np.float32)
    seg = wav[:n * FRAME_SAMPLES].reshape(n, FRAME_SAMPLES).astype(np.float32)
    return (20 * np.log10(np.sqrt((seg ** 2).mean(1) + 1e-12) + 1e-9)).astype(np.float32)


class TygerScorer:
    """The real scorer: one TygerModel, one TygerSessionScorer per session."""
    name = "tyger"

    def __init__(self, model: TygerModel) -> None:
        self.model = model

    @classmethod
    def load(cls, model: str, revision: str | None, device: str) -> "TygerScorer":
        """Find the bundle (no download), check it and load it onto ``device``.
        ``cuda:N`` counts GPUs in PCI bus order (nvidia-smi's) unless
        CUDA_DEVICE_ORDER says otherwise."""
        check_device(device)
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        folder = resolve_bundle(model, revision)
        logger.info("Loading the Tyger bundle from %s on %s", folder, device)
        return cls(TygerModel(folder, device))

    def new_session(self, key: str, index: int) -> "TygerSessionScorer":
        return TygerSessionScorer(self.model)


class TygerSessionScorer:
    """One session's rolling windows and GRU state."""

    def __init__(self, model: TygerModel) -> None:
        np = model.np
        self._m = model
        self._win_u = np.zeros(0, np.float32)
        self._win_a = np.zeros(0, np.float32)
        self._state = None          # the GRU's state; None starts a fresh context

    def reset(self) -> None:
        self._state = None

    def score_hop(self, user: bytes, agent: bytes, mode: str) -> Iterator[FrameScore]:
        """Slide the windows and encode them now; the GRU steps as the frames are
        taken."""
        m, np = self._m, self._m.np
        if mode not in (MODE_LISTENING, MODE_SPEAKING):
            raise ValueError(f"bad mode {mode!r}")
        # int16 to float32 as soundfile reads 16-bit PCM: x / 32768.
        u = np.frombuffer(user, dtype="<i2").astype(np.float32) / 32768.0
        if mode == MODE_SPEAKING:
            a = np.frombuffer(agent, dtype="<i2").astype(np.float32) / 32768.0
        else:
            a = np.zeros(HOP_SAMPLES, np.float32)   # the agent channel is not fed while listening
        if len(u) != HOP_SAMPLES or len(a) != HOP_SAMPLES:
            raise ValueError(f"a hop's channels hold {HOP_SAMPLES} samples")
        self._win_u = np.concatenate([self._win_u, u])[-WINDOW_SAMPLES:]
        self._win_a = np.concatenate([self._win_a, a])[-WINDOW_SAMPLES:]
        hu, ha = m.l15(self._win_u), m.l15_agent(self._win_a)
        if hu.shape[0] < HOP_FRAMES or ha.shape[0] < HOP_FRAMES:
            raise ScorerError(f"the encoder gave {hu.shape[0]} and {ha.shape[0]} frames for a hop")
        act = _frame_energy(np, self._win_u)[-HOP_FRAMES:]
        prev = SPEAK if mode == MODE_SPEAKING else LISTEN
        return self._frames(hu[-HOP_FRAMES:], ha[-HOP_FRAMES:], act, prev)

    def _frames(self, hu, ha, act, prev_state: int) -> Iterator[FrameScore]:
        torch, dev = self._m.torch, self._m.device
        for j in range(HOP_FRAMES):
            db = float(act[j])
            with torch.no_grad():
                x = ((torch.tensor([[db]], device=dev) + 50) / 25).clamp(-3, 3)
                prev = torch.tensor([[prev_state]], dtype=torch.long, device=dev)
                out, self._state = self._m.turn_taker(hu[j][None, None], ha[j][None, None],
                                                      x, prev, self._state)
                o = out[0, 0].cpu().numpy()
            yield FrameScore(float(o[P_END_INDEX]), float(o[YIELD_INDEX]), db)
