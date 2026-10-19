"""Speculative endpoint-hold state machine.

Per-connection *transport-only* state for the nemotron STT path. When NeMo's
acoustic end-of-utterance (EoU) fires, the transport begins a hold: it emits a
provisional endpoint frame (is_final=false, endpoint=true) and starts an
AUDIO-CLOCK timer. The hold is advanced by the duration of audio fed afterwards —
not wall-clock time — and confirms an immutable final once the fed
post-EoU audio reaches the hold window.

The NeMo pipeline's decode/cache handling is never touched and no audio is
re-decoded across an EoU boundary: this object only tracks held text and
a sample counter.

Reopen/stitch: if decoded speech resumes before the hold confirms, the
pending final is cancelled and its text becomes a *stitch prefix* space-joined onto
the resumed segment's partials and eventual final; the resumed NeMo segment is
remapped onto the held segment_index (a growing offset keeps later segments
contiguous). Once a hold *confirms*, it is the commit boundary: a later
turn cannot reopen it — the transport is no longer holding — so it simply takes the
next segment_index through the same offset.
"""

SAMPLE_RATE = 16000


class EndpointHoldMachine:
    """Audio-clock hold for one connection's nemotron STT stream."""

    def __init__(self, hold_ms: int, sample_rate: int = SAMPLE_RATE):
        # Hold window measured in fed audio samples (the audio clock).
        self._hold_samples = int(round(hold_ms / 1000 * sample_rate))
        self._held: tuple[str, str, int] | None = None
        self._samples_since_eou = 0
        # Reopen/stitch state: text to space-join onto the resumed segment,
        # and a monotonic pipeline->emitted segment_index offset (one per reopen).
        self._stitch_prefix = ""
        self._segment_offset = 0
        # Turn carried WITH the held text: the turn open when the held audio was
        # received. A hold that confirms after a later turn_start keeps THIS turn, not the
        # send-time one, so a client reads the confirmed final as a revision of the turn it
        # dispatched. _held_turn rides the pending hold; _stitch_turn preserves it across a
        # reopen so a stitched turn keeps its original turn; _last_final_turn surfaces the
        # turn of the just-popped final (advance/take_held) for the emit site to stamp.
        self._held_turn = 0
        self._stitch_turn = 0
        self._last_final_turn = 0

    @property
    def is_holding(self) -> bool:
        """True while a provisional endpoint is awaiting confirmation."""
        return self._held is not None

    @property
    def stitch_prefix(self) -> str:
        """Held-turn text currently prefixed onto the resumed segment ("" if none)."""
        return self._stitch_prefix

    @property
    def last_final_turn(self) -> int:
        """Turn of the final just popped by advance()/take_held(), for the emit site to stamp."""
        return self._last_final_turn

    @property
    def stitch_turn(self) -> int:
        """Turn owning the active stitch prefix — the mid-stitch flush's turn."""
        return self._stitch_turn

    def begin_hold(self, text: str, language: str, segment_index: int,
                   turn_id: int = 0) -> None:
        """Start (or restart) the hold for a freshly-endpointed segment.

        Resets the audio clock so the window is measured from this EoU point. `text`
        and `segment_index` are the emitted (already stitched + remapped) values, so
        the stitch prefix has been folded in and is cleared here. `turn_id` is
        the turn open when this endpoint's audio was received; the confirmed final carries
        it back. A hold begun on top of an active stitch inherits the stitched
        turn's ORIGINAL turn, so a turn that resumes across a turn_start stays readable as
        that original turn rather than adopting the resumed audio's turn.
        """
        self._held = (text, language, segment_index)
        self._held_turn = self._stitch_turn if self._stitch_prefix else turn_id
        self._samples_since_eou = 0
        self._stitch_prefix = ""
        self._stitch_turn = 0

    def reopen(self) -> tuple[str, str, int] | None:
        """Decoded speech resumed during the hold: cancel the pending confirm.

        Turns the held text into a stitch prefix and bumps the pipeline->emitted
        segment offset so the resumed NeMo segment collapses onto the held index and
        later segments stay contiguous. Returns the cancelled hold, or None if idle.
        """
        if self._held is None:
            return None
        cancelled = self._held
        self._stitch_prefix = cancelled[0]
        self._stitch_turn = self._held_turn  # preserve the original turn across the reopen
        self._segment_offset += 1
        self._held = None
        self._samples_since_eou = 0
        return cancelled

    def stitch(self, text: str) -> str:
        """Space-join the active stitch prefix onto `text` (or return it unchanged)."""
        if self._stitch_prefix:
            return f"{self._stitch_prefix} {text}"
        return text

    def clear_stitch(self) -> None:
        """Drop the active stitch prefix — the resumed turn has been committed.

        Used at a teardown flush so a second flush is truly inert: without this the
        mid-stitch prefix would survive and re-stitch onto a re-finalized tail.
        """
        self._stitch_prefix = ""
        self._stitch_turn = 0

    def remap(self, pipeline_segment_index: int) -> int:
        """Map a NeMo segment_index to the emitted one; reopens collapse segments."""
        return pipeline_segment_index - self._segment_offset

    def advance(self, num_samples: int) -> tuple[str, str, int] | None:
        """Advance the audio clock by num_samples of fed audio.

        Returns the confirmed (text, language, segment_index) once the accumulated
        post-EoU audio reaches the hold window, then clears the hold; otherwise
        None. A no-op (returns None) when not holding.
        """
        if self._held is None:
            return None
        self._samples_since_eou += num_samples
        if self._samples_since_eou >= self._hold_samples:
            confirmed = self._held
            self._last_final_turn = self._held_turn  # surface the held turn for the emit stamp
            self._held = None
            self._samples_since_eou = 0
            return confirmed
        return None

    def take_held(self) -> tuple[str, str, int] | None:
        """Pop the held text unconditionally, clearing the hold (teardown flush).

        Returns None when idle. Used to flush a still-pending endpoint as a final
        when the stream ends mid-hold (wired in a later task).
        """
        held = self._held
        if held is not None:
            self._last_final_turn = self._held_turn  # surface the held turn for the emit stamp
        self._held = None
        self._samples_since_eou = 0
        return held
