"""A streaming limiter for silent runs in synthesized speech.

A TTS engine can make a long pause at a period: up to about a second of
silence inside one synthesis or at its end. When the next synthesis follows at
once, the listener hears a gap between two sentences. The limiter passes a
silent run up to a cap and drops the rest of the run until audio comes again.

It works on whole windows of 10 ms and has no look-ahead: each feed returns all
of its audio that is to be sent, except a rest shorter than one window, which
waits for the next feed or for flush. The result does not depend on how the
input is cut into chunks. Audio that is not silence is passed unchanged.
"""
from __future__ import annotations

import numpy as np

WINDOW_MS = 10
# A window is silence when its RMS is below this level (-45 dBFS).
SILENCE_RMS = 10.0 ** (-45.0 / 20.0)


class SilenceLimiter:
    """Caps each silent run of a float32 mono stream at `cap_ms`.

    The state lives as long as the object, so a run that crosses two feeds (or
    two engine requests fed to one limiter) counts as one run. With `cap_ms` 0
    the limiter is off: feed returns its input unchanged and only measures.
    """

    def __init__(self, sample_rate: int, cap_ms: int) -> None:
        self._sample_rate = int(sample_rate)
        self._window = max(1, self._sample_rate * WINDOW_MS // 1000)
        self._cap = max(0, int(cap_ms)) * self._sample_rate // 1000
        self._rest = np.zeros(0, dtype=np.float32)
        self._run = 0       # samples in the current silent run, as given
        self._longest = 0   # longest run seen since the last take_stats
        self._dropped = 0   # samples dropped since the last take_stats

    @property
    def enabled(self) -> bool:
        return self._cap > 0

    @property
    def held_samples(self) -> int:
        """Samples that wait for the next feed: always fewer than one window."""
        return len(self._rest)

    def feed(self, chunk: np.ndarray) -> np.ndarray:
        """Take the next audio and return what is to be sent now."""
        data = np.concatenate([self._rest, chunk]) if len(self._rest) else chunk
        whole = len(data) // self._window * self._window
        self._rest = np.array(data[whole:], dtype=np.float32)
        out = self._limit(data[:whole])
        return out if self.enabled else chunk

    def flush(self) -> np.ndarray:
        """End of the stream: return the rest that is to be sent."""
        rest, self._rest = self._rest, np.zeros(0, dtype=np.float32)
        out = self._limit(rest)
        return out if self.enabled else np.zeros(0, dtype=np.float32)

    def take_stats(self) -> tuple[int, int]:
        """(longest silent run, time dropped) in ms since the last call.

        A run that is still open counts with its length so far, and goes on
        counting into the next period.
        """
        longest = max(self._longest, self._run)
        stats = (longest * 1000 // self._sample_rate,
                 self._dropped * 1000 // self._sample_rate)
        self._longest = 0
        self._dropped = 0
        return stats

    def _limit(self, data: np.ndarray) -> np.ndarray:
        if not len(data):
            return data
        window = self._window
        count = -(-len(data) // window)  # the last one may be a short rest
        keep = np.ones(count, dtype=bool)
        for i in range(count):
            piece = data[i * window:(i + 1) * window]
            if float(np.sqrt(np.mean(np.square(piece, dtype=np.float64)))) >= SILENCE_RMS:
                self._longest = max(self._longest, self._run)
                self._run = 0
                continue
            if self._cap and self._run >= self._cap:
                keep[i] = False
                self._dropped += len(piece)
            self._run += len(piece)
        if keep.all():
            return data
        parts = [data[i * window:(i + 1) * window] for i in range(count) if keep[i]]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
