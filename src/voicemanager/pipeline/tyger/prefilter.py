"""Experimental pre-filter: give Tyger a user channel that holds only the user.

On speakers the mic carries the reply's echo as well as the user, and Tyger's agent
channel, which keeps the echo from counting as a barge, also pulls the user's own
speech down while it overlaps a reply. A target-speaker filter in front of the
scorer removes the echo instead, so Tyger can run with a silent agent channel and
hear the user's barge-in as it does on headphones.

``PrefilterSessionScorer`` wraps one session's scorer:

* Enrollment. Until the session has 3 s of the user's voice, every hop reaches the
  scorer unchanged. The voice is collected from listening-mode hops only (no reply
  plays, so no echo), frame by frame, keeping the 80 ms frames whose energy is
  above the voice gate, in arrival order. At 3 s the filter is built once from
  exactly those samples. Before enrollment a speaking-mode hop's frames come back
  with a yield of 0, so no barge fires: without the filter the scorer hears the
  reply's echo on the raw mic, and an agent that speaks first (before the user has
  said 3 s) would otherwise stop its own reply on that echo.
* After enrollment every hop's user audio goes through the filter, in order, so the
  filter's state follows the stream. A speaking-mode hop then reaches the scorer
  with the filter's output as its user channel, plus white noise at the floor
  level, and an all-zero agent channel. The filter's output sits far under a real
  room's noise floor, and Tyger with a silent agent channel scores that
  near-silence as a pause it may barge into; the noise keeps the channel at the
  level the model knows. A listening-mode hop reaches the scorer unchanged.

The filter's output comes ``delay`` samples late. The wrapper starts its output
queue with that many zeros, so each hop takes exactly one hop of output and the
filtered channel lags the mic by that fixed delay.

A filter factory has ``enroll(sample) -> filter``, with ``sample`` 48000 float32
samples in -1..1 at 16 kHz; a filter has a ``delay`` in samples and
``push(samples) -> samples``, float32 in and out. This module loads no model.
"""
import logging
from typing import Any, Iterator, Protocol

import numpy as np

from .rules import FRAME_SAMPLES
from .wire import HOP_SAMPLES, MODE_SPEAKING, SAMPLE_RATE

logger = logging.getLogger(__name__)

ENROLL_SAMPLES = 3 * SAMPLE_RATE      # the 3 s of voice the filter is built from
_SILENT_HOP = bytes(HOP_SAMPLES * 2)


class Filter(Protocol):
    delay: int

    def push(self, samples: np.ndarray) -> np.ndarray:
        ...


class FilterFactory(Protocol):
    def enroll(self, sample: np.ndarray) -> Filter:
        ...


def _to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def _to_pcm(x: np.ndarray) -> bytes:
    return np.clip(np.round(x * 32768.0), -32768, 32767).astype("<i2").tobytes()


def _frame_energy_db(x: np.ndarray) -> np.ndarray:
    """Per-frame energy in dB, the formula the turn rules read the scorer's frames with."""
    frames = x[:len(x) // FRAME_SAMPLES * FRAME_SAMPLES].reshape(-1, FRAME_SAMPLES)
    mean_sq = np.mean(frames.astype(np.float64) ** 2, axis=1)
    return 20.0 * np.log10(np.sqrt(mean_sq + 1e-12) + 1e-9)


class PrefilterSessionScorer:
    """One session's scorer behind the pre-filter (see the module docstring)."""

    def __init__(self, inner: Any, factory: FilterFactory, floor_db: float,
                 voice_gate_db: float, key: str = "") -> None:
        self._inner = inner
        self._factory = factory
        self._floor_rms = float(10.0 ** (floor_db / 20.0))
        self._gate_db = float(voice_gate_db)
        self._key = key
        self._voice: list[np.ndarray] = []
        self._voice_len = 0
        self._filter: Filter | None = None
        self._out = np.zeros(0, np.float32)
        self._rng = np.random.default_rng()

    @property
    def enrolled(self) -> bool:
        return self._filter is not None

    def reset(self) -> None:
        self._inner.reset()

    def score_hop(self, user: bytes, agent: bytes, mode: str) -> Iterator:
        if self._filter is None:
            if mode != MODE_SPEAKING:
                self._collect(_to_float(user))
                return self._inner.score_hop(user, agent, mode)
            # No voice sample yet, so the scorer hears the raw mic, echo and all. On
            # speakers that echo can score as a barge, so no barge until enrollment.
            return (frame._replace(yield_prob=0.0)
                    for frame in self._inner.score_hop(user, agent, mode))
        filtered = self._next_output(_to_float(user))
        if mode != MODE_SPEAKING:
            return self._inner.score_hop(user, agent, mode)
        noise = self._rng.standard_normal(HOP_SAMPLES).astype(np.float32) * self._floor_rms
        return self._inner.score_hop(_to_pcm(filtered + noise), _SILENT_HOP, mode)

    def _collect(self, x: np.ndarray) -> None:
        for i, level in enumerate(_frame_energy_db(x)):
            if level > self._gate_db:
                self._voice.append(x[i * FRAME_SAMPLES:(i + 1) * FRAME_SAMPLES])
                self._voice_len += FRAME_SAMPLES
                if self._voice_len >= ENROLL_SAMPLES:
                    self._enroll()
                    return

    def _enroll(self) -> None:
        sample = np.concatenate(self._voice)[:ENROLL_SAMPLES]
        self._voice, self._voice_len = [], 0
        self._filter = self._factory.enroll(sample)
        self._out = np.zeros(int(self._filter.delay), np.float32)
        logger.info("session %s enrolled the pre-filter on %.1f s of voice",
                    self._key, len(sample) / SAMPLE_RATE)

    def _next_output(self, x: np.ndarray) -> np.ndarray:
        """Push one hop through the filter and take one hop of its delayed output."""
        self._out = np.concatenate([self._out, np.asarray(self._filter.push(x), np.float32)])
        if len(self._out) < HOP_SAMPLES:   # a filter that fell behind: pad, keep the hop size
            self._out = np.concatenate([self._out, np.zeros(HOP_SAMPLES - len(self._out),
                                                            np.float32)])
        hop, self._out = self._out[:HOP_SAMPLES], self._out[HOP_SAMPLES:]
        return hop
