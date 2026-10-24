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
    # The same text again is neither new words nor a second revision.
    s.feed(frame("hey I am cleaning the bathroom", final=True))
    assert s.tick() == NOVOICE
    assert s.revisions == 1
    assert s.last_revised is False


def test_slicer_new_segment_starts_from_zero():
    s = mc.TickSlicer()
    s.feed(frame("hey i'm cleaning"))
    assert s.tick() == "hey i'm cleaning"
    s.feed(frame("hey i'm cleaning the bathroom", final=True))
    s.feed(frame("right now", segment=1))
    assert s.tick() == "the bathroom right now"
    s.feed(frame("right now with bleach", segment=1, final=True))
    assert s.tick() == "with bleach"
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
        "right",
        "now",
        NOVOICE,
        NOVOICE,
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
