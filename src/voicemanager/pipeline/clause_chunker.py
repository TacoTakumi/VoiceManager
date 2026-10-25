"""Clause cut rule for streamed TTS text.

A client that streams its reply in small chunks (2-3 words at a time) gets choppy
speech if every chunk is synthesized on its own. ClauseChunker holds the incoming
text and decides when a piece is big enough to send to the engine, using how much
audio of the request is still unplayed:

- First-piece rule, used while the unplayed-audio estimate is 0 (nothing is
  playing yet, or the audio already sent has run out): send at a clause mark, when
  ``first_piece_words`` words are held, or ``first_piece_deadline_ms`` after the
  first held word arrived, whichever comes first.
- Later-piece rule, used while audio is still playing: send at a clause mark, when
  ``later_piece_words`` words are held, or when the unplayed estimate drops below
  ``unplayed_margin_ms``, whichever comes first.

A clause mark is a cut point only when the held text from its start through the mark
holds at least ``mark_min_words`` words (whitespace-separated tokens). A mark after
fewer words ("Yes," or "Stop!") is an ordinary character: it stays inside the piece,
and the piece waits for a later mark, the word count or the time trigger. A
very short piece plays for less time than the next piece takes to start, so cutting
it would leave a gap.

A clause-mark trigger sends the held text up to and including the last clause mark
that qualifies (plus any closing quotes or brackets right after it); the words after
it stay held. A word-count or time trigger sends all held text. A piece is the held
text exactly as received, with only surrounding whitespace trimmed.

The unplayed-audio estimate is a play cursor: the clock time at which the audio sent
so far ends. Each audio chunk extends playback from the later of the cursor and now
by its duration, so a gap in which nothing played is not counted as audio played.
The estimate is the cursor minus now, floored at 0, and 0 before any audio was sent.

The class is pure and synchronous: it never sleeps, and time comes from an injected
clock (seconds, ``time.monotonic`` by default) or an explicit ``now`` argument. A
caller feeds text with ``feed``, reports each audio chunk it sends with
``record_audio``, calls ``poll`` after each event and sends the pieces it returns,
and arms a timer for ``next_deadline`` so a time trigger is not missed. ``drain``
empties the buffer for done and flush. One chunker serves one request.
"""
from collections.abc import Callable
import time

# Marks that end a clause.
CLAUSE_MARKS = ",;:.?!"
# Characters that may follow a clause mark and still belong to the piece it ends.
_CLOSERS = "\"')]}" + "\u201d\u2019"  # plus right double and single curly quotes
# Tolerance for float comparisons of clock times, so a deadline that lands exactly
# on a fake-clock tick is not missed by rounding.
_EPS = 1e-9


def _last_clause_cut(text: str, min_words: int = 1) -> int | None:
    """Return the index just past the last qualifying clause mark in ``text``, or None.

    A mark counts only when it ends a token: it is followed (after any closing quotes
    or brackets) by whitespace or the end of the text. So "3.5" and "a,b" hold no
    clause mark. It qualifies only when ``text`` up to the cut holds at least
    ``min_words`` whitespace-separated words.
    """
    for i in range(len(text) - 1, -1, -1):
        if text[i] not in CLAUSE_MARKS:
            continue
        j = i + 1
        while j < len(text) and text[j] in _CLOSERS:
            j += 1
        if j == len(text) or text[j].isspace():
            # Words up to a mark only shrink further left, so the first mark that
            # fails the minimum ends the search.
            return j if len(text[:j].split()) >= min_words else None
    return None


class ClauseChunker:
    """Hold streamed text and cut it into clause-sized pieces for synthesis.

    Args:
        first_piece_words: Words held that send a piece while nothing is playing.
        first_piece_deadline_ms: Time after the first held word arrived that sends a
            piece while nothing is playing.
        later_piece_words: Words held that send a piece while audio is playing.
        unplayed_margin_ms: A piece is sent while audio is playing once the unplayed
            estimate drops below this.
        mark_min_words: Words the held text must hold up to a clause mark for the
            mark to be a cut point.
        clock: Returns the current time in seconds. Every method that takes ``now``
            uses it when ``now`` is None.
    """

    def __init__(
        self,
        first_piece_words: int = 4,
        first_piece_deadline_ms: int = 600,
        later_piece_words: int = 12,
        unplayed_margin_ms: int = 1000,
        mark_min_words: int = 3,
        clock: Callable[[], float] = time.monotonic,
    ):
        if first_piece_words < 1 or later_piece_words < 1:
            raise ValueError("piece word counts must be at least 1")
        if mark_min_words < 1:
            raise ValueError("mark_min_words must be at least 1")
        if first_piece_deadline_ms <= 0:
            raise ValueError("first_piece_deadline_ms must be positive")
        if unplayed_margin_ms < 0:
            raise ValueError("unplayed_margin_ms must not be negative")
        self.first_piece_words = first_piece_words
        self.first_piece_deadline_ms = first_piece_deadline_ms
        self.later_piece_words = later_piece_words
        self.unplayed_margin_ms = unplayed_margin_ms
        self.mark_min_words = mark_min_words
        self._clock = clock
        # Held text as the frames that brought it, each with its arrival time, so the
        # words left after a clause cut keep the arrival time of their own frame.
        self._frames: list[tuple[str, float]] = []
        # Play cursor: clock time at which the audio sent so far ends; None before
        # any audio was sent.
        self._play_cursor: float | None = None

    # --- inputs ---------------------------------------------------------------------

    def feed(self, text: str, now: float | None = None) -> None:
        """Hold ``text`` (one incoming frame), stamped with its arrival time."""
        if text:
            self._frames.append((text, self._now(now)))

    def record_audio(self, duration_s: float, now: float | None = None) -> None:
        """Record that ``duration_s`` seconds of audio were sent to the client now.

        The chunk plays from the later of the play cursor and now, so the cursor moves
        to that time plus ``duration_s``.
        """
        if duration_s < 0:
            raise ValueError("duration_s must not be negative")
        if duration_s == 0:
            return
        now = self._now(now)
        start = now if self._play_cursor is None else max(self._play_cursor, now)
        self._play_cursor = start + duration_s

    # --- outputs --------------------------------------------------------------------

    def poll(self, now: float | None = None) -> list[str]:
        """Return the pieces to send to the engine now, in order (may be empty).

        Call it after every ``feed``, every ``record_audio`` and when the time from
        ``next_deadline`` is reached. The returned text is removed from the buffer.
        """
        now = self._now(now)
        pieces: list[str] = []
        while True:
            piece = self._next_piece(now)
            if piece is None:
                return pieces
            pieces.append(piece)

    def next_deadline(self, now: float | None = None) -> float | None:
        """Return the clock time at which a time trigger fires, or None.

        None means no text is held, so only a new ``feed`` can produce a piece. A time
        at or before ``now`` means a trigger is already due: call ``poll``. The answer
        can change after ``feed`` or ``record_audio``, so ask again after each.
        """
        first_word_at = self._first_word_at()
        if first_word_at is None:
            return None
        first_deadline = first_word_at + self.first_piece_deadline_ms / 1000.0
        if self.unplayed_estimate(now) == 0.0:
            return first_deadline
        audio_end = self._play_cursor
        if self.unplayed_margin_ms > 0:
            return audio_end - self.unplayed_margin_ms / 1000.0
        # With no margin the later-piece rule has no time trigger; the first-piece
        # deadline takes over once the audio has run out.
        return max(audio_end, first_deadline)

    def drain(self) -> list[str]:
        """Remove and return all held text as one piece ([] if nothing is held).

        Use it for done and flush, which send everything held at once.
        """
        text = self.held_text
        self._frames.clear()
        return [text] if text else []

    def unplayed_estimate(self, now: float | None = None) -> float:
        """Return the estimated unplayed audio in seconds, floored at 0.

        It is the play cursor minus now; 0 before any audio was sent.
        """
        if self._play_cursor is None:
            return 0.0
        estimate = self._play_cursor - self._now(now)
        return estimate if estimate > _EPS else 0.0

    @property
    def held_text(self) -> str:
        """The held text with surrounding whitespace trimmed."""
        return "".join(text for text, _ in self._frames).strip()

    @property
    def has_held(self) -> bool:
        """True when any non-whitespace text is held."""
        return self._first_word_at() is not None

    # --- internals ------------------------------------------------------------------

    def _now(self, now: float | None) -> float:
        return self._clock() if now is None else now

    def _first_word_at(self) -> float | None:
        """Arrival time of the frame that brought the first held word."""
        for text, at in self._frames:
            if not text.isspace():
                return at
        return None

    def _next_piece(self, now: float) -> str | None:
        """Cut one piece off the buffer if a trigger fires, else return None."""
        held = "".join(text for text, _ in self._frames)
        if not held.strip():
            return None
        cut = _last_clause_cut(held, self.mark_min_words)
        if cut is not None:
            self._take(cut)
            return held[:cut].strip()
        if self._count_or_time_due(held, now):
            self._frames.clear()
            return held.strip()
        return None

    def _count_or_time_due(self, held: str, now: float) -> bool:
        words = len(held.split())
        estimate = self.unplayed_estimate(now)
        if estimate == 0.0:
            if words >= self.first_piece_words:
                return True
            first_word_at = self._first_word_at()
            deadline = first_word_at + self.first_piece_deadline_ms / 1000.0
            return now >= deadline - _EPS
        if words >= self.later_piece_words:
            return True
        return estimate < self.unplayed_margin_ms / 1000.0 - _EPS

    def _take(self, cut: int) -> None:
        """Drop the first ``cut`` characters of held text, keeping frame times."""
        kept: list[tuple[str, float]] = []
        offset = 0
        for text, at in self._frames:
            end = offset + len(text)
            if end > cut:
                kept.append((text[max(cut - offset, 0):], at))
            offset = end
        # Leading whitespace-only frames carry no word; drop them so the remainder's
        # deadline counts from the frame that brought its first word.
        while kept and kept[0][0].isspace():
            kept.pop(0)
        self._frames = kept
