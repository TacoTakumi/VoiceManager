"""Tests for the tick slicer: cumulative STT frames in, new words per 0.6 s tick out.

Model-free and server-free. Frames are plain dicts in the shape VoiceManager's
streaming STT sends.
"""
import importlib.util
import pathlib
import sys

_NAME = "microturn_client_slicer"
_PATH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "microturn_client.py"
_spec = importlib.util.spec_from_file_location(_NAME, _PATH)
mc = importlib.util.module_from_spec(_spec)
# dataclasses look the module up in sys.modules while the class body runs.
sys.modules[_NAME] = mc
_spec.loader.exec_module(mc)

NOVOICE = "<no voice>"


def frame(text: str, segment: int = 0, final: bool = False) -> dict:
    """One transcription frame as the server sends it."""
    return {"type": "transcription", "text": text, "is_final": final,
            "segment_index": segment}


def test_slicer_growth_emits_only_new_words():
    s = mc.TickSlicer()
    s.feed(frame("hey i'm"))
    assert s.tick() == "hey i'm"
    s.feed(frame("hey i'm cleaning"))
    s.feed(frame("hey i'm cleaning the"))
    assert s.tick() == "cleaning the"
    assert s.revisions == 0


def test_slicer_silent_tick_is_novoice():
    s = mc.TickSlicer()
    assert s.tick() == NOVOICE
    s.feed(frame("hey i'm"))
    assert s.tick() == "hey i'm"
    s.feed(frame("hey i'm"))
    assert s.tick() == NOVOICE
    assert s.revisions == 0


def test_slicer_revision_does_not_resend():
    s = mc.TickSlicer()
    s.feed(frame("hey i'm cleaning the"))
    assert s.tick() == "hey i'm cleaning the"
    # The first four words now read "hey I am cleaning": a revision of sent words.
    s.feed(frame("hey I am cleaning the bathroom"))
    assert s.tick() == "the bathroom"
    assert s.revisions == 1
    assert s.last_revised is True
    # The same text again is neither new words nor a second revision; the final
    # ends the utterance, so the tick sends the late end mark alone.
    s.feed(frame("hey I am cleaning the bathroom", final=True))
    assert s.tick() == "."
    assert s.revisions == 1
    assert s.last_revised is False


def test_slicer_new_segment_starts_from_zero():
    s = mc.TickSlicer()
    s.feed(frame("hey i'm cleaning"))
    assert s.tick() == "hey i'm cleaning"
    s.feed(frame("hey i'm cleaning the bathroom", final=True))
    s.feed(frame("right now", segment=1))
    # The end mark of the finished segment comes before the words of the next one.
    assert s.tick() == "the bathroom. right now"
    s.feed(frame("right now with bleach", segment=1, final=True))
    assert s.tick() == "with bleach."
    # A segment that is new in the tick is emitted whole, even if it repeats words.
    s.feed(frame("right", segment=2))
    assert s.tick() == "right"
    assert s.revisions == 0


def test_slicer_shrink_emits_nothing_and_counts_revision():
    s = mc.TickSlicer()
    s.feed(frame("hey i'm cleaning the"))
    assert s.tick() == "hey i'm cleaning the"
    s.feed(frame("hey i'm cleaning"))
    assert s.tick() == NOVOICE
    assert s.revisions == 1
    # Growth past what was sent still yields only the words beyond it.
    s.feed(frame("hey i'm cleaning the bathroom"))
    assert s.tick() == "bathroom"
    assert s.revisions == 1


def test_slicer_scripted_call_sequence():
    """Growth, a silent tick, a revision of a sent word, then a new segment."""
    script = [
        [frame("hey")],
        [frame("hey i'm"), frame("hey i'm cleaning")],
        [],
        [frame("hey i'm cleaning the")],
        [frame("hey I'm cleaning the bathroom")],
        [frame("hey I'm cleaning the bathroom", final=True), frame("right", segment=1)],
        [frame("right now", segment=1)],
        [],
        [frame("right now", segment=1, final=True)],
    ]
    s = mc.TickSlicer()
    out = []
    for frames in script:
        for f in frames:
            s.feed(f)
        out.append(s.tick())
    assert out == [
        "hey",
        "i'm cleaning",
        NOVOICE,
        "the",
        "bathroom",
        ". right",
        "now",
        NOVOICE,
        ".",
    ]
    assert s.revisions == 1


def test_slicer_repeated_shrink_counts_one_revision():
    s = mc.TickSlicer()
    s.feed(frame("hey i'm cleaning the"))
    assert s.tick() == "hey i'm cleaning the"
    for _ in range(3):
        s.feed(frame("hey i'm"))
        assert s.tick() == NOVOICE
    assert s.revisions == 1


def test_slicer_changed_shrink_then_growth_counts_one_revision():
    s = mc.TickSlicer()
    s.feed(frame("hey i'm cleaning the"))
    assert s.tick() == "hey i'm cleaning the"
    s.feed(frame("hey I"))
    assert s.tick() == NOVOICE
    # Growth back over the sent region re-fills words already sent: only the
    # fifth word onward is new.
    s.feed(frame("hey I am cleaning the bathroom"))
    assert s.tick() == "the bathroom"
    assert s.revisions == 1


def test_slicer_completes_a_cut_word_before_new_words():
    s = mc.TickSlicer()
    s.feed(frame("tell me a ra"))
    assert s.tick() == "tell me a ra"
    # The STT finishes the word it cut at the tick boundary: the missing letters
    # lead the next fragment, as in the training data, and nothing is re-sent.
    s.feed(frame("tell me a random fact"))
    assert s.tick() == "ndom fact"
    assert s.revisions == 0
    assert s.last_revised is False
    s.feed(frame("tell me a random fact"))
    assert s.tick() == NOVOICE


def test_slicer_completes_a_cut_word_alone():
    s = mc.TickSlicer()
    s.feed(frame("tell me a ra"))
    assert s.tick() == "tell me a ra"
    s.feed(frame("tell me a random"))
    assert s.tick() == "ndom"
    assert s.revisions == 0
    assert s.last_revised is False
    s.feed(frame("tell me a random fact"))
    assert s.tick() == "fact"


def test_slicer_sends_a_late_mark_as_a_completion():
    s = mc.TickSlicer()
    s.feed(frame("a fact about honey"))
    assert s.tick() == "a fact about honey"
    s.feed(frame("a fact about honey?"))
    assert s.tick() == "?"
    assert s.revisions == 0
    assert s.last_revised is False


def test_slicer_final_without_mark_sends_a_full_stop_alone():
    s = mc.TickSlicer()
    s.feed(frame("tell me a fact about honey"))
    assert s.tick() == "tell me a fact about honey"
    s.feed(frame("tell me a fact about honey", final=True))
    assert s.tick() == "."
    assert s.revisions == 0
    assert s.last_revised is False


def test_slicer_final_of_a_question_sends_a_question_mark():
    s = mc.TickSlicer()
    s.feed(frame("could you tell me a fact"))
    assert s.tick() == "could you tell me a fact"
    s.feed(frame("could you tell me a fact", final=True))
    assert s.tick() == "?"


def test_slicer_final_with_new_words_marks_the_last_word():
    s = mc.TickSlicer()
    s.feed(frame("tell me a fact about"))
    assert s.tick() == "tell me a fact about"
    s.feed(frame("tell me a fact about honey", final=True))
    assert s.tick() == "honey."
    # A segment gets one mark: the same final again sends nothing.
    s.feed(frame("tell me a fact about honey", final=True))
    assert s.tick() == NOVOICE


def test_slicer_final_that_ends_in_a_mark_gets_no_second_mark():
    s = mc.TickSlicer()
    s.feed(frame("a fact about honey"))
    assert s.tick() == "a fact about honey"
    s.feed(frame("a fact about honey?", final=True))
    assert s.tick() == "?"
    s.feed(frame("a fact about honey?", final=True))
    assert s.tick() == NOVOICE
    assert s.revisions == 0


def test_slicer_empty_final_sends_nothing():
    s = mc.TickSlicer()
    s.feed(frame("", final=True))
    assert s.tick() == NOVOICE
    s.feed(frame("", segment=1, final=True))
    s.feed(frame("hey", segment=2))
    assert s.tick() == "hey"


def test_slicer_shrink_inside_a_sent_word_then_regrowth_resends_nothing():
    s = mc.TickSlicer()
    s.feed(frame("hey I am cleaning the"))
    assert s.tick() == "hey I am cleaning the"
    # The shrink ends inside a word the model received whole; regrowth to the
    # same words must not send the rest of that word again.
    s.feed(frame("hey I am cle"))
    assert s.tick() == NOVOICE
    s.feed(frame("hey I am cleaning the bathroom"))
    assert s.tick() == "bathroom"


def test_slicer_shrink_to_a_shorter_prefix_then_regrowth_resends_nothing():
    s = mc.TickSlicer()
    s.feed(frame("hey I am cleaning the"))
    assert s.tick() == "hey I am cleaning the"
    s.feed(frame("hey I a"))
    assert s.tick() == NOVOICE
    s.feed(frame("hey I am cleaning the bathroom"))
    assert s.tick() == "bathroom"


def test_slicer_revised_last_word_is_not_completed():
    s = mc.TickSlicer()
    s.feed(frame("hey I am cleaning the"))
    assert s.tick() == "hey I am cleaning the"
    s.feed(frame("hey I am cleaning tha"))
    assert s.tick() == NOVOICE
    assert s.revisions == 1
    # "that" extends the revised word, not the word the model received: it is
    # another revision of a sent word, and nothing is sent.
    s.feed(frame("hey I am cleaning that"))
    assert s.tick() == NOVOICE
    assert s.revisions == 2


def test_slicer_completes_after_a_revision_of_earlier_words():
    s = mc.TickSlicer()
    s.feed(frame("hey i'm cleaning the"))
    assert s.tick() == "hey i'm cleaning the"
    # Six words against four sent: a revision of the sent ones, two new words.
    s.feed(frame("hey I am cleaning the ra"))
    assert s.tick() == "the ra"
    assert s.revisions == 1
    s.feed(frame("hey I am cleaning the random"))
    assert s.tick() == "ndom"
    assert s.revisions == 1
