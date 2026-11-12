"""A record of when the frames of one TTS request were sent.

A client's playback queue can run dry in the middle of a reply, and the server
must be able to show its side: when each frame left and how much audio it held.
The record takes the time and the length of each frame as it is sent and gives
one log line for the request.

The line also gives the lowest reserve of the simplest client: playback starts
when the first frame arrives and does not stop. The reserve at a send is the
audio sent before it minus the time since the first send. A negative value means
that such a client ran dry by that amount. A client that buffers before it plays
has more reserve than this by the length of its buffer.

The record only reads. It does not change what is sent.
"""
from __future__ import annotations

import time
from typing import Callable

# The line lists this many frames. A longer request says how many it left out;
# its counts and its lowest reserve still cover every frame.
MAX_LISTED = 200


class FrameLog:
    """Times and lengths of the frames of one request, and the line for the log."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._entries: list[str] = []
        self._count = 0
        self._audio_s = 0.0
        self._first_at: float | None = None
        self._last_at = 0.0
        self._lowest: float | None = None   # lowest reserve, seconds
        self._lowest_at = 0.0               # time of it from the first frame
        self._mark_next = False

    def engine_request(self) -> None:
        """A new engine request starts: mark its first frame in the list."""
        self._mark_next = self._count > 0

    def sent(self, n_samples: int, sample_rate: int) -> None:
        """A frame of `n_samples` was just sent."""
        if n_samples <= 0 or sample_rate <= 0:
            return
        now = self._clock()
        if self._first_at is None:
            self._first_at = now
        since_first = now - self._first_at
        if self._count:
            reserve = self._audio_s - since_first
            if self._lowest is None or reserve < self._lowest:
                self._lowest = reserve
                self._lowest_at = since_first
        length_s = n_samples / sample_rate
        if self._count < MAX_LISTED:
            if self._mark_next:
                self._entries.append("|")
            self._entries.append(f"{round(since_first * 1000)}:{round(length_s * 1000)}")
        self._mark_next = False
        self._count += 1
        self._audio_s += length_s
        self._last_at = since_first

    def line(self, request_id: str, chunking: str) -> str:
        """The log line for the request, from the frames recorded so far."""
        if self._lowest is None:
            reserve, at = "none", "none"
        else:
            reserve, at = str(round(self._lowest * 1000)), str(round(self._lowest_at * 1000))
        listed = " ".join(self._entries)
        if self._count > MAX_LISTED:
            listed += f" +{self._count - MAX_LISTED} more"
        return (
            f"TTS frames: request_id={request_id} chunking={chunking} "
            f"frames={self._count} audio_ms={round(self._audio_s * 1000)} "
            f"span_ms={round(self._last_at * 1000)} "
            f"min_reserve_ms={reserve} at_ms={at} "
            f"sent=[{listed}]"
        )
