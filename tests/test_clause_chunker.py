"""Unit tests for the clause cut rule in ClauseChunker.

Model-free and server-free: a fake clock drives every time-based rule.
"""
import pytest

from voicemanager.pipeline.clause_chunker import CLAUSE_MARKS, ClauseChunker


class FakeClock:
    """A settable clock in seconds."""

    def __init__(self, start: float = 100.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def chunker(clock: FakeClock) -> ClauseChunker:
    return ClauseChunker(clock=clock)


def play_earlier_piece(chunker: ClauseChunker, seconds: float) -> None:
    """Record audio of an earlier piece so the later-piece rule applies."""
    chunker.record_audio(seconds)


# --- defaults and the clause marks -----------------------------------------------


def test_defaults_match_the_documented_thresholds():
    c = ClauseChunker()
    assert c.first_piece_words == 4
    assert c.first_piece_deadline_ms == 600
    assert c.later_piece_words == 12
    assert c.unplayed_margin_ms == 500


def test_clause_marks_are_the_six_marks():
    assert set(CLAUSE_MARKS) == set(",;:.?!")


@pytest.mark.parametrize("kwargs", [
    {"first_piece_words": 0},
    {"later_piece_words": 0},
    {"first_piece_deadline_ms": 0},
    {"unplayed_margin_ms": -1},
])
def test_bad_thresholds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ClauseChunker(**kwargs)


# --- first piece: no audio playing -----------------------------------------------


def test_first_piece_clause_mark_with_two_words_dispatches(chunker):
    chunker.feed("Hello there,")
    assert chunker.poll() == ["Hello there,"]
    assert chunker.held_text == ""


@pytest.mark.parametrize("mark", list(",;:.?!"))
def test_first_piece_each_clause_mark_dispatches(chunker, mark):
    chunker.feed(f"Two words{mark}")
    assert chunker.poll() == [f"Two words{mark}"]


def test_first_piece_four_words_without_mark_dispatches(chunker):
    chunker.feed("one two three four")
    assert chunker.poll() == ["one two three four"]


def test_first_piece_three_words_without_mark_is_held(chunker):
    chunker.feed("one two three")
    assert chunker.poll() == []
    assert chunker.held_text == "one two three"


def test_first_piece_two_words_dispatch_after_600_ms(chunker, clock):
    chunker.feed("two words")
    clock.advance(0.6)
    assert chunker.poll() == ["two words"]


def test_first_piece_two_words_not_dispatched_at_599_ms(chunker, clock):
    chunker.feed("two words")
    clock.advance(0.599)
    assert chunker.poll() == []
    assert chunker.held_text == "two words"


def test_first_piece_deadline_counts_from_first_held_word(chunker, clock):
    chunker.feed("one")
    clock.advance(0.4)
    chunker.feed(" two")
    clock.advance(0.199)
    assert chunker.poll() == []
    clock.advance(0.001)
    assert chunker.poll() == ["one two"]


def test_whitespace_only_frame_does_not_start_the_deadline(chunker, clock):
    chunker.feed("   ")
    clock.advance(0.5)
    chunker.feed("word")
    clock.advance(0.5)
    assert chunker.poll() == []
    clock.advance(0.1)
    assert chunker.poll() == ["word"]


def test_poll_accepts_explicit_now(clock):
    chunker = ClauseChunker(clock=clock)
    chunker.feed("two words", now=10.0)
    assert chunker.poll(now=10.599) == []
    assert chunker.poll(now=10.6) == ["two words"]


# --- later pieces: audio playing --------------------------------------------------


def test_later_piece_clause_mark_dispatches(chunker, clock):
    play_earlier_piece(chunker, 3.0)
    chunker.feed("two words.")
    assert chunker.poll() == ["two words."]


def test_later_piece_twelve_words_dispatch(chunker, clock):
    play_earlier_piece(chunker, 5.0)
    words = " ".join(f"w{i}" for i in range(12))
    chunker.feed(words)
    assert chunker.poll() == [words]


def test_later_piece_eleven_words_are_held(chunker, clock):
    play_earlier_piece(chunker, 5.0)
    words = " ".join(f"w{i}" for i in range(11))
    chunker.feed(words)
    assert chunker.poll() == []


def test_later_piece_dispatches_when_unplayed_drops_below_margin(chunker, clock):
    play_earlier_piece(chunker, 3.0)
    chunker.feed("two words")
    clock.advance(2.5)  # unplayed is exactly 0.5 s: not below the margin
    assert chunker.poll() == []
    clock.advance(0.01)
    assert chunker.poll() == ["two words"]


def test_later_piece_five_words_with_two_seconds_unplayed_are_held(chunker, clock):
    play_earlier_piece(chunker, 3.0)
    clock.advance(1.0)
    assert chunker.unplayed_estimate() == pytest.approx(2.0)
    chunker.feed("one two three four five")
    assert chunker.poll() == []
    assert chunker.held_text == "one two three four five"


def test_later_piece_ignores_first_piece_deadline(chunker, clock):
    play_earlier_piece(chunker, 10.0)
    chunker.feed("two words")
    clock.advance(2.0)
    assert chunker.poll() == []


def test_first_piece_rule_returns_once_audio_has_run_out(chunker, clock):
    play_earlier_piece(chunker, 1.0)
    clock.advance(2.0)
    assert chunker.unplayed_estimate() == 0.0
    chunker.feed("one two three four")
    assert chunker.poll() == ["one two three four"]


# --- the unplayed-audio estimate --------------------------------------------------


def test_estimate_is_zero_before_any_audio(chunker):
    assert chunker.unplayed_estimate() == 0.0


def test_estimate_three_seconds_sent_one_elapsed_is_two(chunker, clock):
    chunker.record_audio(3.0)
    clock.advance(1.0)
    assert chunker.unplayed_estimate() == pytest.approx(2.0)


def test_estimate_three_seconds_sent_four_elapsed_is_zero(chunker, clock):
    chunker.record_audio(3.0)
    clock.advance(4.0)
    assert chunker.unplayed_estimate() == 0.0


def test_estimate_sums_chunks_and_counts_from_the_first(chunker, clock):
    chunker.record_audio(1.0)
    clock.advance(0.5)
    chunker.record_audio(2.0)
    clock.advance(0.5)
    # 3 s sent, 1 s since the first chunk
    assert chunker.unplayed_estimate() == pytest.approx(2.0)


def test_record_audio_rejects_negative_duration(chunker):
    with pytest.raises(ValueError):
        chunker.record_audio(-0.1)


# --- the cut at the last clause mark ----------------------------------------------


def test_cut_at_clause_mark_keeps_following_words_held(chunker, clock):
    chunker.feed("them. It creates")
    assert chunker.poll() == ["them."]
    assert chunker.held_text == "It creates"
    clock.advance(0.599)
    assert chunker.poll() == []
    clock.advance(0.001)
    assert chunker.poll() == ["It creates"]


def test_cut_is_at_the_last_clause_mark(chunker):
    chunker.feed("Yes, of course. And")
    assert chunker.poll() == ["Yes, of course."]
    assert chunker.held_text == "And"


def test_remainder_deadline_counts_from_its_own_arrival(chunker, clock):
    chunker.feed("one")
    clock.advance(0.3)
    chunker.feed(" two, three")
    assert chunker.poll() == ["one two,"]
    clock.advance(0.3)  # 0.6 s since "one", 0.3 s since "three"
    assert chunker.poll() == []
    clock.advance(0.3)
    assert chunker.poll() == ["three"]


def test_remainder_can_trigger_its_own_word_count(chunker):
    chunker.feed("Well, one two three four")
    assert chunker.poll() == ["Well,", "one two three four"]
    assert chunker.held_text == ""


def test_mark_inside_a_token_is_not_a_clause_mark(chunker):
    chunker.feed("costs 3.5 now")
    assert chunker.poll() == []
    assert chunker.held_text == "costs 3.5 now"


def test_closing_quote_after_a_mark_stays_with_the_piece(chunker):
    chunker.feed('He said "go." Then')
    assert chunker.poll() == ['He said "go."']
    assert chunker.held_text == "Then"


def test_repeated_marks_are_cut_after_the_run(chunker):
    chunker.feed("Wait... what?! So")
    assert chunker.poll() == ["Wait... what?!"]
    assert chunker.held_text == "So"


# --- punctuation is kept exactly --------------------------------------------------


def test_text_in_two_frames_is_sent_exactly(chunker):
    chunker.feed("Do not mix")
    assert chunker.poll() == []
    chunker.feed(" them.")
    assert chunker.poll() == ["Do not mix them."]


def test_only_surrounding_whitespace_is_trimmed(chunker):
    chunker.feed("  \"Hi\" -- it's  me;\n")
    assert chunker.poll() == ["\"Hi\" -- it's  me;"]


def test_word_split_across_frames_is_joined_as_received(chunker):
    chunker.feed("Hel")
    chunker.feed("lo,")
    assert chunker.poll() == ["Hello,"]


# --- next_deadline ----------------------------------------------------------------


def test_next_deadline_is_none_when_nothing_held(chunker):
    assert chunker.next_deadline() is None
    chunker.feed("  ")
    assert chunker.next_deadline() is None


def test_next_deadline_first_piece_is_600_ms_after_first_word(chunker, clock):
    start = clock.now
    chunker.feed("two words")
    assert chunker.next_deadline() == pytest.approx(start + 0.6)


def test_next_deadline_later_piece_is_when_unplayed_reaches_margin(chunker, clock):
    start = clock.now
    chunker.record_audio(3.0)
    clock.advance(1.0)
    chunker.feed("two words")
    assert chunker.next_deadline() == pytest.approx(start + 2.5)


def test_next_deadline_then_poll_dispatches(chunker, clock):
    chunker.feed("two words")
    clock.now = chunker.next_deadline()
    assert chunker.poll() == ["two words"]
    assert chunker.next_deadline() is None


# --- drain ------------------------------------------------------------------------


def test_drain_returns_all_held_text(chunker):
    chunker.feed(" two words ")
    assert chunker.drain() == ["two words"]
    assert chunker.held_text == ""
    assert chunker.next_deadline() is None


def test_drain_with_nothing_held_returns_empty(chunker):
    assert chunker.drain() == []
    chunker.feed(" \n ")
    assert chunker.drain() == []


def test_chunker_keeps_working_after_drain(chunker, clock):
    chunker.feed("two words")
    chunker.drain()
    clock.advance(5.0)
    chunker.feed("new")
    clock.advance(0.599)
    assert chunker.poll() == []
    clock.advance(0.001)
    assert chunker.poll() == ["new"]
