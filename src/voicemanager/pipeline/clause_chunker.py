"""Clause cut rule for streamed TTS text.

A client that streams its reply in small chunks (2-3 words, or single LLM tokens,
at a time) gets choppy speech if every chunk is synthesized on its own. ClauseChunker
holds the incoming text and decides when a piece is big enough to send to the engine,
using how much audio of the request is still unplayed:

- First-piece rule, used while the unplayed-audio estimate is 0 (nothing is
  playing yet, or the audio already sent has run out): send at a clause mark, when
  ``first_piece_words`` words are held, or ``first_piece_deadline_ms`` after the
  first held word arrived, whichever comes first. When the deadline comes while
  the held text ends in a word still inside its word grace (see below), the cut
  waits until that grace ends, so the word goes with the piece, or for at most
  ``word_grace_ms`` past the deadline, when frames keep arriving; the cut is then
  at the last whitespace.
- Later-piece rule, used while audio is still playing: send at a clause mark, when
  ``later_piece_words`` words are held, or when the unplayed estimate drops below
  ``unplayed_margin_ms``, whichever comes first. The margin trigger does not wait
  for a word inside its grace: it cuts at the last whitespace at once.

A clause mark is a cut point only when the held text from its start through the mark
holds at least ``mark_min_words`` words (whitespace-separated tokens). A mark after
fewer words ("Yes," or "Stop!") is an ordinary character: it stays inside the piece,
and the piece waits for a later mark, the word count or the time trigger. A
very short piece plays for less time than the next piece takes to start, so cutting
it would leave a gap.

A clause-mark trigger sends the held text up to and including the last clause mark
that qualifies (plus any closing quotes or brackets right after it); the words after
it stay held. A word-count or time trigger sends the held text up to its last word
end, so a token stream never has a word cut in two ("subsystem" + "s." stays one
word). A word ends at a whitespace character. The end of the held text is also a
word end once the newest frame has been held for ``word_grace_ms`` with no new frame:
a client that streams whole words without trailing whitespace stops sending its
last word, while a token stream sends the rest of a word within a few milliseconds.
Text after the last word end may be a partial word and stays held. The word count
counts only the words before the last word end, and neither trigger fires while no
word end follows a word. With ``word_grace_ms`` of 0 every frame end is a word end.
``drain`` still sends all held text.

A piece is a slice of the held text exactly as received, whitespace included: a
cut never trims, drops or adds a character, so the pieces of a request joined end
to end equal its frames joined end to end. Whitespace after a clause mark starts
the next piece; whitespace at a word end closes the piece before it. Only text that
holds a letter or digit is spoken, so no piece is text without one (a lone "." or
"&"). A cut whose piece would hold no letter or digit does not happen: that text
stays held and starts the next piece. Held text with no letter or digit has no
deadline, and ``drain`` drops it at done and at the flush timeout, so the only
text a request can lose is a tail with nothing to speak in it. A flush leaves the
request open, so its ``drain(keep_unspoken=True)`` keeps such text held to start
the next piece.

The unplayed-audio estimate is a play cursor: the clock time at which the audio sent
so far ends. Each audio chunk extends playback from the later of the cursor and now
by its duration, so a gap in which nothing played is not counted as audio played.
The estimate is the cursor minus now, floored at 0, and 0 before any audio was sent.

The class is pure and synchronous: it never sleeps, and time comes from an injected
clock (seconds, ``time.monotonic`` by default) or an explicit ``now`` argument. A
caller feeds text with ``feed``, reports each audio chunk it sends with
``record_audio``, calls ``poll`` after each event and sends the pieces it returns,
and arms a timer for ``next_deadline`` so a time trigger is not missed. ``drain``
sends all held text for done and flush (a flush keeps text with nothing to speak
held). One chunker serves one request.
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


def _is_spoken(text: str) -> bool:
    """True when ``text`` holds a letter or digit, so an engine has something to say."""
    return any(c.isalnum() for c in text)


def _word_boundary(text: str) -> int | None:
    """Return the index just past the last whitespace in ``text``, or None.

    The text before that index holds only complete words; the text after it may be a
    word still arriving. None when the text before the last whitespace holds no
    letter or digit, so ``text`` holds no complete word worth a piece of its own.
    A shorter prefix holds no more than a longer one, so only the last whitespace
    needs the check.
    """
    for i in range(len(text) - 1, -1, -1):
        if text[i].isspace():
            return i + 1 if _is_spoken(text[:i]) else None
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
        word_grace_ms: Time the newest frame must be held with no new frame before
            the end of the held text counts as a word end. 0 makes every frame end a
            word end.
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
        word_grace_ms: int = 250,
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
        if word_grace_ms < 0:
            raise ValueError("word_grace_ms must not be negative")
        self.first_piece_words = first_piece_words
        self.first_piece_deadline_ms = first_piece_deadline_ms
        self.later_piece_words = later_piece_words
        self.unplayed_margin_ms = unplayed_margin_ms
        self.mark_min_words = mark_min_words
        self.word_grace_ms = word_grace_ms
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
        Each piece is the held text exactly as received, whitespace included, and
        holds a letter or digit.
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

        None means no text with a letter or digit is held (only whitespace or
        punctuation, or nothing), so only a new ``feed`` can produce a piece. A time
        at or before ``now`` means a trigger is already due: call ``poll``. While no
        whitespace follows a held word, the time is no earlier than the end of the
        word grace of the newest frame, because no trigger can cut before then. When
        the first-piece deadline governs and the held text ends in a partial word
        after fewer complete words than ``first_piece_words``, the time is the
        earlier of the end of its grace and the deadline plus ``word_grace_ms``,
        because the first-piece cut waits until then. When a partial last word would
        complete the active word count, the time is no later than the end of its
        grace, when the count trigger fires. With an ``unplayed_margin_ms`` of 0 and
        ``first_piece_words`` held while audio plays, the time is no later than the
        audio end, or the end of the last word's grace when that is later, because
        the first-piece count fires then. The answer can change after ``feed`` or
        ``record_audio``, so ask again after each.
        """
        held = "".join(text for text, _ in self._frames)
        if not _is_spoken(held):
            return None
        first_word_at = self._first_word_at()
        deadline = self._trigger_deadline(first_word_at, now)
        grace_end = self._frames[-1][1] + self.word_grace_ms / 1000.0
        boundary = _word_boundary(held)
        count_met = len(held.split()) >= self._count_threshold(now)
        if (self.unplayed_margin_ms == 0 and self.unplayed_estimate(now) > 0.0
                and len(held.split()) >= self.first_piece_words):
            # The first-piece count takes over when the audio runs out, which can
            # be before the first-piece deadline.
            deadline = self._play_cursor
        if boundary is None:
            # No trigger can cut before the grace ends (a cut before the last word
            # would hold no letter or digit); once it ends, a met word count cuts at
            # once.
            return grace_end if count_met else max(deadline, grace_end)
        first_count_met = len(held[:boundary].split()) >= self.first_piece_words
        if held[boundary:] and not first_count_met and (
                self.unplayed_estimate(now) == 0.0 or self.unplayed_margin_ms == 0):
            # The first-piece deadline waits for the partial last word: until its
            # grace ends (all held text goes) or for at most the grace (the cut is
            # at the last whitespace), whichever comes first. A met first-piece
            # count of complete words cuts as soon as that rule governs, so it
            # does not wait.
            first_deadline = first_word_at + self.first_piece_deadline_ms / 1000.0
            grace = self.word_grace_ms / 1000.0
            deadline = max(deadline, min(grace_end, first_deadline + grace))
        if held[boundary:] and count_met:
            # The partial last word completes the word count once its grace has
            # passed, so wake then rather than wait for the time trigger.
            return min(deadline, grace_end)
        return deadline

    def drain(self, keep_unspoken: bool = False) -> list[str]:
        """Remove and return all held text as one piece, exactly as received.

        Use it for done, flush and the flush timeout, which send everything held at
        once. When the held text holds no letter or digit (nothing, whitespace or
        only punctuation), it returns [] and:

        - with ``keep_unspoken`` False (done, the flush timeout: the request ends),
          drops that text;
        - with ``keep_unspoken`` True (a flush: the request stays open), keeps it
          held, so it starts the next piece and the words around it do not run
          together ("use." + "\\n\\n" + "One:").

        Held text with a letter or digit is always removed and returned whole,
        whitespace included.
        """
        text = "".join(text for text, _ in self._frames)
        if not _is_spoken(text):
            if not keep_unspoken:
                self._frames.clear()
            return []
        self._frames.clear()
        return [text]

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
        """The held text with surrounding whitespace trimmed, for inspection.

        Unlike the pieces from ``poll`` and ``drain``, which keep whitespace exactly
        as received, this is stripped.
        """
        return "".join(text for text, _ in self._frames).strip()

    @property
    def has_held(self) -> bool:
        """True when any non-whitespace text is held."""
        return self._first_word_at() is not None

    # --- internals ------------------------------------------------------------------

    def _now(self, now: float | None) -> float:
        return self._clock() if now is None else now

    def _trigger_deadline(self, first_word_at: float, now: float | None) -> float:
        """Clock time at which the first-piece deadline or the margin trigger fires."""
        first_deadline = first_word_at + self.first_piece_deadline_ms / 1000.0
        if self.unplayed_estimate(now) == 0.0:
            return first_deadline
        audio_end = self._play_cursor
        if self.unplayed_margin_ms > 0:
            return audio_end - self.unplayed_margin_ms / 1000.0
        # With no margin the later-piece rule has no time trigger; the first-piece
        # deadline takes over once the audio has run out.
        return max(audio_end, first_deadline)

    def _word_end(self, held: str, now: float) -> int | None:
        """Return the index just past the last word end in ``held``, or None.

        The end of ``held`` is a word end once the newest frame has been held for the
        word grace; before that, only whitespace after a word ends one.
        """
        newest_at = self._frames[-1][1]
        if now - newest_at >= self.word_grace_ms / 1000.0 - _EPS:
            return len(held)
        return _word_boundary(held)

    def _first_word_at(self) -> float | None:
        """Arrival time of the frame that brought the first held word.

        Whitespace-only frames, kept at the front after a cut, are skipped, so the
        deadline counts from the frame that brought the first non-whitespace text.
        """
        for text, at in self._frames:
            if not text.isspace():
                return at
        return None

    def _next_piece(self, now: float) -> str | None:
        """Cut one piece off the buffer if a trigger fires, else return None.

        A clause mark cuts after the mark; a word-count or time trigger cuts at the
        last word end, leaving a partial last word held. The piece is returned raw.
        A mark cut whose piece holds no letter or digit does not happen; a shorter
        cut at an earlier mark would hold none either, so the count and time
        triggers decide.
        """
        held = "".join(text for text, _ in self._frames)
        if not _is_spoken(held):
            return None
        cut = _last_clause_cut(held, self.mark_min_words)
        if cut is not None and _is_spoken(held[:cut]):
            self._take(cut)
            return held[:cut]
        end = self._word_end(held, now)
        if end is not None and self._count_or_time_due(held[:end], end < len(held), now):
            # A word end before the grace holds a letter or digit (``_word_boundary``);
            # after the grace it is the end of the held text, which holds one too.
            self._take(end)
            return held[:end]
        return None

    def _count_threshold(self, now: float | None) -> int:
        """Word count that sends a piece under the rule active at ``now``."""
        if self.unplayed_estimate(now) == 0.0:
            return self.first_piece_words
        return self.later_piece_words

    def _count_or_time_due(self, complete: str, partial: bool, now: float) -> bool:
        """True when a word-count or time trigger fires for the complete words held.

        ``partial`` is True when held text after ``complete`` is a word still inside
        its grace; the first-piece deadline then waits up to the grace for it.
        """
        if len(complete.split()) >= self._count_threshold(now):
            return True
        estimate = self.unplayed_estimate(now)
        if estimate == 0.0:
            first_word_at = self._first_word_at()
            deadline = first_word_at + self.first_piece_deadline_ms / 1000.0
            if partial:
                deadline += self.word_grace_ms / 1000.0
            return now >= deadline - _EPS
        return estimate < self.unplayed_margin_ms / 1000.0 - _EPS

    def _take(self, cut: int) -> None:
        """Drop the first ``cut`` characters of held text, keeping frame times.

        Whitespace-only frames left at the front are kept, so no whitespace is lost;
        ``_first_word_at`` skips them, so the deadline still counts from the frame
        that brought the first word.
        """
        kept: list[tuple[str, float]] = []
        offset = 0
        for text, at in self._frames:
            end = offset + len(text)
            if end > cut:
                kept.append((text[max(cut - offset, 0):], at))
            offset = end
        self._frames = kept
