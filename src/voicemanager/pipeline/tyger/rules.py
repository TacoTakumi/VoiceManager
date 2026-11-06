"""The Tyger turn rules: what one session's per-frame head scores mean for the turn.

Adapted from tyger-s2s (``app/plugins/ears/tyger/rules.py``, ``session.py`` and
``constants.py``, Apache-2.0; see LICENSE and NOTICE next to this file). Plain
bookkeeping on one ``TurnRules`` instance per session: no model, no audio, no
threads, standard library only, so it runs in the server's tests and in the
sidecar alike.

The sidecar scores each 80 ms frame with the turn model and feeds the result in:
the endpoint head's ``p_end``, the yield head's probability, the frame's energy in
dB and the mode (``listening``: no reply plays, ``speaking``: a reply plays).

* listening: an endpoint by ``model`` when ``p_end`` stays at or above the endpoint
  threshold for the debounce frames, once the turn holds the minimum of voiced
  audio; else an endpoint by ``silence`` once that many frames in a row are at or
  below the voice gate for the silence fallback's length.
* speaking: a barge when the yield probability stays at or above the barge
  threshold for the debounce frames. Its onset is where the user's current stretch
  of speech began (by frame energy, through pauses of up to ``onset_gap_frames``),
  since the head takes a second or more to be sure.
* The voice gate is calibrated once per session. Until the session has heard
  ``CALIBRATION_FRAMES`` listening-mode frames (2 s), the configured gate applies;
  then the gate becomes the 20th percentile of those frames' energies plus
  ``gate_margin_db``, no lower than the configured gate and no higher than
  ``gate_max_db``. Speaking-mode frames carry the reply's echo and do not count.
  Turn ends and mode changes keep the calibrated gate.
* The head's decisions count only after ``warmup_frames`` frames of fresh context:
  after the first hop, every switch of mode and every endpoint. The silence
  fallback reads only energy, so it needs no warm-up.

When the heads' context must start over the engine says so (``begin_hop`` returns
True, or ``FrameResult.reset`` is True) and the caller resets the model's state.
The model's previous-state input is LISTEN while listening and SPEAK while
speaking; it follows from the mode, so the engine does not track it.

A barge latches: no second barge fires until the mode goes back to listening,
since hops already on their way still say speaking. The first listening hop after
a barge starts a new turn that already holds the user's speech: the voiced frames
of the last ``barge_seed_ms`` of speaking count toward the minimum speech.

Load it in a process without the voicemanager package by loading this directory
as a package by path, so the relative import of ``wire`` resolves::

    spec = importlib.util.spec_from_file_location(
        "tyger", tyger_dir / "__init__.py", submodule_search_locations=[str(tyger_dir)])
    sys.modules["tyger"] = pkg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pkg)
    import tyger.rules, tyger.wire
"""
from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

from .wire import (
    HOP_SAMPLES,
    MODE_LISTENING,
    MODE_SPEAKING,
    SAMPLE_RATE,
    BargeEvent,
    EndpointEvent,
    Event,
)

logger = logging.getLogger(__name__)

FRAME_MS = 80                                   # one encoder frame = one decision
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 1280
HOP_FRAMES = HOP_SAMPLES // FRAME_SAMPLES       # 4 frames per wire hop

_MODES = (MODE_LISTENING, MODE_SPEAKING)

CALIBRATION_FRAMES = 25                         # 2 s of listening frames set the gate
CALIBRATION_PERCENTILE = 20.0                   # the room level among those frames


@dataclass(frozen=True)
class TurnRuleSettings:
    """The rule engine's thresholds. The defaults are the tuned values."""
    endpoint_threshold: float = 0.915     # p_end at or above this counts toward an endpoint
    endpoint_debounce_frames: int = 2     # ... for this many frames in a row
    barge_threshold: float = 0.6          # yield at or above this counts toward a barge
    barge_debounce_frames: int = 2        # ... for this many frames in a row
    warmup_frames: int = 6                # frames of fresh context before the heads count
    min_speech_ms: float = 350.0          # voiced audio a turn needs before it can end
    voice_gate_db: float = -50.0          # the gate before calibration, and the lowest one
    gate_margin_db: float = 8.0           # the calibrated gate sits this far above the room
    gate_max_db: float = -38.0            # ... and never above this
    silence_fallback_ms: float = 1200.0   # quiet that ends a spoken turn the head has not
    onset_gap_frames: int = 8             # a longer pause ends a stretch of speech (640 ms)
    onset_max_ms: float = 4000.0          # a barge's onset reaches back at most this far
    barge_seed_ms: float = 2000.0         # speaking audio a barge's new turn starts with

    def __post_init__(self) -> None:
        for name in ("endpoint_threshold", "barge_threshold"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in 0..1, got {value}")
        for name in ("endpoint_debounce_frames", "barge_debounce_frames"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")
        for name in ("warmup_frames", "onset_gap_frames"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in ("min_speech_ms", "barge_seed_ms"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        for name in ("silence_fallback_ms", "onset_max_ms"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.gate_margin_db < 0:
            raise ValueError(f"gate_margin_db must be >= 0, got {self.gate_margin_db}")
        if self.gate_max_db < self.voice_gate_db:
            raise ValueError(f"gate_max_db ({self.gate_max_db}) must be at or above "
                             f"voice_gate_db ({self.voice_gate_db})")

    @property
    def silence_fallback_frames(self) -> int:
        """The silence fallback in whole frames, rounded up (1200 ms is 15 frames)."""
        return math.ceil(self.silence_fallback_ms / FRAME_MS)

    @property
    def onset_max_samples(self) -> int:
        return int(round(self.onset_max_ms * SAMPLE_RATE / 1000))

    @property
    def barge_seed_frames(self) -> int:
        return int(round(self.barge_seed_ms / FRAME_MS))


class FrameResult(NamedTuple):
    """One frame's decision: its events, and whether the heads' context must start
    over before the next frame is scored."""
    events: list[Event]
    reset: bool


class TurnRules:
    """One session's turn state and its rules. Instances share nothing."""

    def __init__(self, settings: TurnRuleSettings | None = None, key: str = "") -> None:
        self.settings = settings if settings is not None else TurnRuleSettings()
        self._key = key
        self._seed: deque[bool] = deque(maxlen=self.settings.barge_seed_frames)
        # The session's voice gate. It is set once and outlives every turn.
        self._gate = self.settings.voice_gate_db
        self._calibration: list[float] = []
        self._calibrated = False
        self._pending_reset = True      # the first hop starts the heads' context
        self._clear()

    @property
    def voice_gate_db(self) -> float:
        """The gate this session uses now: the configured one until calibration."""
        return self._gate

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def _calibrate(self, energy_db: float) -> None:
        self._calibration.append(float(energy_db))
        if len(self._calibration) < CALIBRATION_FRAMES:
            return
        s = self.settings
        room = _percentile(self._calibration, CALIBRATION_PERCENTILE)
        self._gate = min(max(room + s.gate_margin_db, s.voice_gate_db), s.gate_max_db)
        self._calibrated = True
        self._calibration = []
        logger.info("session %s calibrated the voice gate: room %.1f dB, gate %.1f dB",
                    self._key, room, self._gate)

    # --- state -----------------------------------------------------------------

    def _clear(self) -> None:
        """A fresh session: a new turn in listening, every run at zero."""
        self._mode = MODE_LISTENING
        self._warm = 0                  # frames since the heads' context started
        self._ep_run = 0                # frames in a row with p_end at the threshold
        self._yield_run = 0             # frames in a row with yield at the threshold
        self._voiced_ms = 0.0           # voiced audio in this turn
        self._in_turn = False           # the user has made a sound in this turn
        self._silent_run = 0            # listening: quiet frames in a row
        self._talk_frames = 0           # speaking: frames since the user's stretch began
        self._quiet_frames = 0          #   (0: none), and its trailing quiet frames
        self._barged = False            # a barge fired in this reply
        self._seed.clear()              # speaking: voiced flags of the last seed frames

    def _switch_mode(self, mode: str) -> None:
        """A change between listening and speaking: fresh decision runs. The turn
        so far (its voiced audio) carries over."""
        self._warm = 0
        self._ep_run = self._yield_run = 0
        self._silent_run = 0
        self._talk_frames = self._quiet_frames = 0
        self._mode = mode

    def _enter(self, mode: str) -> bool:
        """Apply a pending reset or a mode change. True if the heads' context must
        start over."""
        if mode not in _MODES:
            raise ValueError(f"mode must be {MODE_LISTENING!r} or {MODE_SPEAKING!r}, got {mode!r}")
        if self._pending_reset:
            self._pending_reset = False
            self._clear()
            if mode != self._mode:
                self._switch_mode(mode)
            return True
        if self._barged and mode == MODE_LISTENING:
            # The reply stopped after a barge: a new turn that starts with the
            # user's speech over the end of the reply.
            seed_frames = sum(self._seed)
            self._clear()
            self._voiced_ms = float(seed_frames * FRAME_MS)
            self._in_turn = self._voiced_ms > 0
            logger.debug("turn seeded after a barge with %d voiced frames", seed_frames)
            return True
        if mode != self._mode:
            self._switch_mode(mode)
            return True
        return False

    # --- driving -----------------------------------------------------------------

    def begin_hop(self, mode: str) -> bool:
        """Before a hop's frames are scored: apply a pending reset or a mode change.
        True if the heads' context must start over before scoring this hop."""
        return self._enter(mode)

    def on_frame(self, mode: str, p_end: float, yield_prob: float, energy_db: float,
                 mic_sample: int) -> FrameResult:
        """One frame's decision.

        ``mic_sample`` is the index on the mic stream of the sample just after this
        frame (the frame's end); events carry it as their ``mic_sample``. A caller
        that scores a model calls ``begin_hop`` first; a frame-level caller may skip
        it, as a change of mode here is applied the same way.
        """
        self._enter(mode)
        voiced = energy_db > self._gate
        if mode == MODE_LISTENING and not self._calibrated:
            self._calibrate(energy_db)
        if mode == MODE_SPEAKING:
            return self._speaking_frame(yield_prob, voiced, mic_sample)
        return self._listening_frame(p_end, voiced, mic_sample)

    def _speaking_frame(self, yield_prob: float, voiced: bool, mic_sample: int) -> FrameResult:
        s = self.settings
        self._warm += 1
        self._track_talk(voiced)
        self._seed.append(voiced)
        if self._barged or self._warm <= s.warmup_frames:
            return FrameResult([], False)
        self._yield_run = self._yield_run + 1 if yield_prob >= s.barge_threshold else 0
        if self._yield_run < s.barge_debounce_frames:
            return FrameResult([], False)
        self._yield_run = 0
        self._barged = True
        onset = mic_sample
        if self._talk_frames:
            onset = max(mic_sample - self._talk_frames * FRAME_SAMPLES,
                        mic_sample - s.onset_max_samples, 0)
        return FrameResult([BargeEvent(onset_mic_sample=onset, mic_sample=mic_sample)], False)

    def _track_talk(self, voiced: bool) -> None:
        """How long the user has been talking: a stretch starts at a voiced frame
        and lasts through pauses of up to onset_gap_frames."""
        if self._talk_frames:
            self._talk_frames += 1
            self._quiet_frames = 0 if voiced else self._quiet_frames + 1
            if self._quiet_frames > self.settings.onset_gap_frames:
                self._talk_frames = self._quiet_frames = 0
        elif voiced:
            self._talk_frames, self._quiet_frames = 1, 0

    def _listening_frame(self, p_end: float, voiced: bool, mic_sample: int) -> FrameResult:
        s = self.settings
        self._warm += 1
        if voiced:
            self._voiced_ms += FRAME_MS
            self._in_turn = True
            self._silent_run = 0
        else:
            self._silent_run += 1
        if self._warm > s.warmup_frames:
            self._ep_run = self._ep_run + 1 if p_end >= s.endpoint_threshold else 0
        else:
            self._ep_run = 0
        spoke = self._in_turn and self._voiced_ms >= s.min_speech_ms
        if not spoke:
            return FrameResult([], False)
        if self._ep_run >= s.endpoint_debounce_frames:
            by = "model"
        elif self._silent_run >= s.silence_fallback_frames:
            by = "silence"
        else:
            return FrameResult([], False)
        # The turn ended: the next starts from nothing, with fresh context.
        self._voiced_ms = 0.0
        self._in_turn = False
        self._warm = 0
        self._ep_run = 0
        self._silent_run = 0
        return FrameResult([EndpointEvent(by=by, mic_sample=mic_sample)], True)


def _percentile(values: list[float], q: float) -> float:
    """The q-th percentile, interpolated between the two nearest ranks (numpy's
    default), without numpy: the rules use the standard library only."""
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    low = math.floor(pos)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)
