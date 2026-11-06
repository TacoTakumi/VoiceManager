"""The Tyger turn rules, fed a fake score stream.

The rule engine turns per-frame head scores (p_end, yield), the frame's energy and
the mode into endpoint and barge events. These tests drive it frame by frame with
hand-made scores: no model, no audio, no sidecar.

Frames are 80 ms (1280 samples at 16 kHz). Each helper frame is
(mode, p_end, yield, energy_db); the engine gets the frame's end as its mic_sample.
"""
import subprocess
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import pytest

from voicemanager.pipeline.tyger import wire
from voicemanager.pipeline.tyger.rules import (
    FRAME_MS,
    FRAME_SAMPLES,
    TurnRules,
    TurnRuleSettings,
)
from voicemanager.pipeline.tyger.wire import BargeEvent, EndpointEvent

TYGER_DIR = Path(__file__).resolve().parents[1] / "src" / "voicemanager" / "pipeline" / "tyger"

L = wire.MODE_LISTENING
S = wire.MODE_SPEAKING
VOICED = -30.0          # well above the -50 dB voice gate
QUIET = -70.0           # well below it


def listen(n, p_end=0.1, db=VOICED):
    return [(L, p_end, 0.0, db)] * n


def speak(n, yld=0.0, db=QUIET):
    return [(S, 0.0, yld, db)] * n


def run(frames, settings=None, engine=None):
    """Feed frames to one engine; return [(frame_index, event)] for every event."""
    engine = engine or TurnRules(settings)
    out = []
    for i, (mode, p_end, yld, db) in enumerate(frames):
        result = engine.on_frame(mode, p_end, yld, db, (i + 1) * FRAME_SAMPLES)
        out.extend((i, event) for event in result.events)
    return out


def end_of(i):
    return (i + 1) * FRAME_SAMPLES


# --- defaults --------------------------------------------------------------------

def test_defaults_match_the_documented_values():
    s = TurnRuleSettings()
    assert s.endpoint_threshold == 0.915
    assert s.endpoint_debounce_frames == 2
    assert s.barge_threshold == 0.6
    assert s.barge_debounce_frames == 2
    assert s.warmup_frames == 6
    assert s.min_speech_ms == 350.0
    assert s.voice_gate_db == -50.0
    assert s.silence_fallback_ms == 1200.0
    assert FRAME_MS == 80 and FRAME_SAMPLES == 1280
    assert s.silence_fallback_frames == 15
    assert wire.HOP_SAMPLES == 4 * FRAME_SAMPLES


@pytest.mark.parametrize("field, value", [
    ("endpoint_threshold", 1.5),
    ("endpoint_threshold", -0.1),
    ("barge_threshold", 2.0),
    ("endpoint_debounce_frames", 0),
    ("barge_debounce_frames", 0),
    ("warmup_frames", -1),
    ("min_speech_ms", -1.0),
    ("silence_fallback_ms", 0.0),
    ("onset_gap_frames", -1),
    ("onset_max_ms", 0.0),
    ("barge_seed_ms", -1.0),
])
def test_settings_reject_out_of_range_values(field, value):
    with pytest.raises(ValueError):
        TurnRuleSettings(**{field: value})


def test_unknown_mode_is_rejected():
    engine = TurnRules()
    with pytest.raises(ValueError):
        engine.on_frame("listen", 0.1, 0.0, VOICED, FRAME_SAMPLES)
    with pytest.raises(ValueError):
        engine.begin_hop("talking")


# --- (a) endpoint by the model ------------------------------------------------------

# 5 voiced frames are 400 ms, the first count at or past 350 ms; one more quiet frame
# takes the engine through its 6 warm-up frames, so the next two frames count.
SPEECH_THEN_QUIET = listen(5) + listen(1, db=QUIET)


def test_p_end_092_on_two_frames_after_350ms_of_speech_ends_the_turn_by_model():
    frames = SPEECH_THEN_QUIET + listen(2, p_end=0.92, db=QUIET)
    events = run(frames)
    assert events == [(7, EndpointEvent(by="model", mic_sample=end_of(7)))]


def test_p_end_091_gives_no_endpoint():
    frames = SPEECH_THEN_QUIET + listen(4, p_end=0.91, db=QUIET)
    assert run(frames) == []


def test_one_frame_of_high_p_end_gives_no_endpoint():
    frames = SPEECH_THEN_QUIET + listen(1, p_end=0.95, db=QUIET) + listen(1, db=QUIET) \
        + listen(1, p_end=0.95, db=QUIET)
    assert run(frames) == []


def test_no_endpoint_before_350ms_of_voiced_audio():
    # 4 voiced frames are 320 ms: the head may say the turn ended, but it cannot yet.
    frames = listen(4) + listen(2, db=QUIET) + listen(3, p_end=0.99, db=QUIET)
    assert run(frames) == []


def test_no_endpoint_during_the_warmup_frames():
    # 5 voiced frames with the head already high: frames 0-5 are warm-up, so the run
    # starts at frame 6 and the endpoint lands on frame 7, not frame 4.
    frames = listen(8, p_end=0.99)
    assert run(frames) == [(7, EndpointEvent(by="model", mic_sample=end_of(7)))]


# --- (b) endpoint by the silence fallback ----------------------------------------------

def test_15_quiet_frames_after_speech_end_the_turn_by_silence_on_the_15th():
    frames = listen(6) + listen(15, p_end=0.1, db=QUIET)
    events = run(frames)
    assert events == [(20, EndpointEvent(by="silence", mic_sample=end_of(20)))]


def test_14_quiet_frames_are_not_enough_and_a_voiced_frame_restarts_the_count():
    frames = listen(6) + listen(14, db=QUIET) + listen(1) + listen(14, db=QUIET)
    assert run(frames) == []
    assert run(frames + listen(1, db=QUIET)) == [
        (35, EndpointEvent(by="silence", mic_sample=end_of(35)))]


def test_silence_without_enough_speech_does_not_end_a_turn():
    assert run(listen(4) + listen(40, db=QUIET)) == []
    assert run(listen(40, db=QUIET)) == []


def test_a_frame_at_the_voice_gate_is_quiet():
    frames = listen(6) + listen(15, db=-50.0)
    assert run(frames) == [(20, EndpointEvent(by="silence", mic_sample=end_of(20)))]


def test_the_model_wins_when_both_would_fire_on_the_same_frame():
    frames = listen(6) + listen(13, db=QUIET) + listen(2, p_end=0.95, db=QUIET)
    assert run(frames) == [(20, EndpointEvent(by="model", mic_sample=end_of(20)))]


# --- (c) barge ---------------------------------------------------------------------

def test_yield_06_on_two_frames_after_warmup_gives_a_barge():
    frames = speak(6) + speak(2, yld=0.6)
    events = run(frames)
    assert events == [(7, BargeEvent(onset_mic_sample=end_of(7), mic_sample=end_of(7)))]


def test_yield_059_gives_no_barge():
    assert run(speak(6) + speak(10, yld=0.59)) == []


def test_a_single_frame_of_yield_gives_no_barge():
    assert run(speak(6) + speak(1, yld=0.9) + speak(1) + speak(1, yld=0.9) + speak(3)) == []


def test_nothing_fires_during_the_six_warmup_frames():
    events = run(speak(6, yld=0.99) + speak(1, yld=0.99))
    assert events == []
    # The run starts at the 7th frame, so the barge lands on the 8th.
    events = run(speak(8, yld=0.99))
    assert [i for i, _ in events] == [7]


def test_warmup_restarts_on_every_switch_to_speaking():
    frames = speak(4) + listen(1, db=QUIET) + speak(6, yld=0.99) + speak(2, yld=0.99)
    events = run(frames)
    assert [i for i, _ in events] == [12]


def test_barge_onset_is_the_start_of_the_users_voiced_run():
    # The user starts talking at frame 5 and pauses for 3 frames (within the
    # 8-frame gap a stretch of speech allows); the head is sure at frame 11.
    frames = speak(5) + speak(3, db=VOICED) + speak(3) + speak(2, yld=0.8, db=VOICED)
    events = run(frames)
    assert events == [(12, BargeEvent(onset_mic_sample=5 * FRAME_SAMPLES, mic_sample=end_of(12)))]


def test_a_pause_longer_than_the_gap_starts_a_new_stretch():
    # Voiced at frames 0-2, then 9 quiet frames (more than 8): the stretch that
    # counts begins at frame 12.
    frames = speak(3, db=VOICED) + speak(9) + speak(3, db=VOICED) + speak(2, yld=0.8, db=VOICED)
    events = run(frames)
    assert events == [(16, BargeEvent(onset_mic_sample=12 * FRAME_SAMPLES, mic_sample=end_of(16)))]


def test_barge_onset_reaches_back_at_most_four_seconds():
    frames = speak(70, db=VOICED) + speak(2, yld=0.8, db=VOICED)
    (i, event), = run(frames)
    assert event.mic_sample == end_of(71)
    assert event.mic_sample - event.onset_mic_sample == 4 * wire.SAMPLE_RATE


def test_barge_with_no_voiced_frames_has_its_onset_at_the_decision():
    (i, event), = run(speak(6) + speak(2, yld=0.8))
    assert event.onset_mic_sample == event.mic_sample


# --- (d) every value changes the outcome ---------------------------------------------

def test_endpoint_threshold_setting():
    frames = SPEECH_THEN_QUIET + listen(2, p_end=0.91, db=QUIET)
    assert run(frames) == []
    assert run(frames, TurnRuleSettings(endpoint_threshold=0.9)) == [
        (7, EndpointEvent(by="model", mic_sample=end_of(7)))]
    high = SPEECH_THEN_QUIET + listen(2, p_end=0.92, db=QUIET)
    assert run(high, TurnRuleSettings(endpoint_threshold=0.95)) == []


def test_endpoint_debounce_setting():
    frames = SPEECH_THEN_QUIET + listen(2, p_end=0.95, db=QUIET)
    assert len(run(frames)) == 1
    three = TurnRuleSettings(endpoint_debounce_frames=3)
    assert run(frames, three) == []
    assert run(frames + listen(1, p_end=0.95, db=QUIET), three) == [
        (8, EndpointEvent(by="model", mic_sample=end_of(8)))]
    one = TurnRuleSettings(endpoint_debounce_frames=1)
    assert [i for i, _ in run(frames, one)] == [6]


def test_barge_threshold_setting():
    frames = speak(6) + speak(2, yld=0.55)
    assert run(frames) == []
    assert [i for i, _ in run(frames, TurnRuleSettings(barge_threshold=0.5))] == [7]
    assert run(speak(6) + speak(2, yld=0.6), TurnRuleSettings(barge_threshold=0.7)) == []


def test_barge_debounce_setting():
    frames = speak(6) + speak(2, yld=0.8)
    three = TurnRuleSettings(barge_debounce_frames=3)
    assert run(frames, three) == []
    assert [i for i, _ in run(frames + speak(1, yld=0.8), three)] == [8]
    assert [i for i, _ in run(speak(6) + speak(1, yld=0.8), TurnRuleSettings(
        barge_debounce_frames=1))] == [6]


def test_warmup_setting():
    assert [i for i, _ in run(speak(4, yld=0.8), TurnRuleSettings(warmup_frames=2))] == [3]
    assert [i for i, _ in run(speak(8, yld=0.8), TurnRuleSettings(warmup_frames=6))] == [7]
    assert [i for i, _ in run(speak(10, yld=0.8), TurnRuleSettings(warmup_frames=8))] == [9]
    # It is the same warm-up after a reset in listening.
    assert [i for i, _ in run(listen(5, p_end=0.99), TurnRuleSettings(warmup_frames=0))] == [4]


def test_min_speech_setting():
    no_warmup = dict(warmup_frames=0)
    frames = listen(4) + listen(2, p_end=0.95, db=QUIET)       # 320 ms of speech
    assert run(frames, TurnRuleSettings(**no_warmup)) == []
    assert [e for _, e in run(frames, TurnRuleSettings(min_speech_ms=300.0, **no_warmup))] == [
        EndpointEvent(by="model", mic_sample=end_of(5))]
    five = listen(5) + listen(2, p_end=0.95, db=QUIET)         # 400 ms
    assert len(run(five, TurnRuleSettings(**no_warmup))) == 1
    assert run(five, TurnRuleSettings(min_speech_ms=450.0, **no_warmup)) == []


def test_voice_gate_setting():
    # -45 dB is speech at the default -50 dB gate, quiet at a -40 dB gate.
    frames = listen(6, db=-45.0) + listen(15, db=-45.0)
    assert run(frames) == []
    assert run(frames, TurnRuleSettings(voice_gate_db=-40.0)) == []   # never spoke
    spoken = listen(6) + listen(15, db=-45.0)
    assert run(spoken) == []
    assert run(spoken, TurnRuleSettings(voice_gate_db=-40.0)) == [
        (20, EndpointEvent(by="silence", mic_sample=end_of(20)))]


def test_silence_fallback_setting():
    frames = listen(6) + listen(15, db=QUIET)
    short = TurnRuleSettings(silence_fallback_ms=800.0)
    assert short.silence_fallback_frames == 10
    assert run(frames, short) == [(15, EndpointEvent(by="silence", mic_sample=end_of(15)))]
    long = TurnRuleSettings(silence_fallback_ms=2000.0)
    assert long.silence_fallback_frames == 25
    assert run(frames, long) == []
    # A length between frames rounds up to the next whole frame.
    assert TurnRuleSettings(silence_fallback_ms=1210.0).silence_fallback_frames == 16


def test_onset_settings():
    frames = speak(3, db=VOICED) + speak(5) + speak(2, yld=0.8, db=VOICED)
    (_, event), = run(frames)
    assert event.onset_mic_sample == 0
    (_, event), = run(frames, TurnRuleSettings(onset_gap_frames=4))
    assert event.onset_mic_sample == 8 * FRAME_SAMPLES
    (_, event), = run(frames, TurnRuleSettings(onset_max_ms=320.0))
    assert event.mic_sample - event.onset_mic_sample == 4 * FRAME_SAMPLES


# --- what resets ---------------------------------------------------------------------

def test_after_an_endpoint_the_next_turn_needs_new_speech_and_a_new_warmup():
    engine = TurnRules()
    first = SPEECH_THEN_QUIET + listen(2, p_end=0.95, db=QUIET)
    assert len(run(first, engine=engine)) == 1
    # The head stays high, but the turn starts over: no speech yet.
    assert run(listen(20, p_end=0.95, db=QUIET), engine=engine) == []


def test_an_endpoint_asks_for_the_heads_context_to_start_over():
    engine = TurnRules()
    resets = []
    for i, (mode, p_end, yld, db) in enumerate(SPEECH_THEN_QUIET + listen(3, p_end=0.95)):
        resets.append(engine.on_frame(mode, p_end, yld, db, end_of(i)).reset)
    assert resets == [False] * 7 + [True, False]


def test_begin_hop_reports_when_the_heads_context_must_start_over():
    engine = TurnRules()
    assert engine.begin_hop(L) is True          # the first hop
    assert engine.begin_hop(L) is False
    assert engine.begin_hop(S) is True          # a switch
    assert engine.begin_hop(S) is False
    assert engine.begin_hop(L) is True


def test_one_barge_per_reply_then_the_turn_starts_with_the_users_speech():
    engine = TurnRules()
    # The user talks over the reply for 6 frames (480 ms); the head is sure at frame 7.
    frames = speak(2) + speak(6, yld=0.9, db=VOICED)
    events = run(frames, engine=engine)
    assert [type(e) for _, e in events] == [BargeEvent]
    # Hops already in flight still say speaking: no second barge.
    assert run(speak(12, yld=0.9, db=VOICED), engine=engine) == []
    # Back to listening: the turn already holds the user's speech, so after the
    # warm-up the head can end it without another 350 ms.
    assert engine.begin_hop(L) is True
    events = run(listen(6, db=QUIET) + listen(2, p_end=0.95, db=QUIET), engine=engine)
    assert [e.by for _, e in events] == ["model"]


def test_the_barge_seed_counts_only_the_last_two_seconds_of_speaking():
    engine = TurnRules()
    # 3 voiced frames, then 30 quiet ones (more than 2 s), then the barge on quiet
    # frames: the seed holds no speech, so the turn needs 350 ms afresh.
    run(speak(3, db=VOICED) + speak(30) + speak(2, yld=0.9), engine=engine)
    assert run(listen(6, db=QUIET) + listen(4, p_end=0.95, db=QUIET), engine=engine) == []


def test_a_new_reply_after_listening_can_be_barged_again():
    engine = TurnRules()
    assert len(run(speak(6) + speak(2, yld=0.9), engine=engine)) == 1
    run(listen(3, db=QUIET), engine=engine)
    assert len(run(speak(6) + speak(2, yld=0.9), engine=engine)) == 1


def test_the_turn_survives_a_switch_to_speaking_and_back():
    # Speech in listening still counts after a short reply plays (the switch resets
    # the decision runs and the warm-up, not the turn).
    engine = TurnRules()
    run(listen(5), engine=engine)
    run(speak(3), engine=engine)
    events = run(listen(6, db=QUIET) + listen(2, p_end=0.95, db=QUIET), engine=engine)
    assert [e.by for _, e in events] == ["model"]


def test_events_go_through_the_wire_format():
    events = [e for _, e in run(SPEECH_THEN_QUIET + listen(2, p_end=0.95, db=QUIET))]
    events += [e for _, e in run(speak(2) + speak(6, yld=0.9, db=VOICED))]
    for event in events:
        assert wire.parse_event_line(wire.format_event(event)) == event


def test_settings_are_frozen_and_shared_safely():
    s = TurnRuleSettings()
    with pytest.raises(Exception):
        s.barge_threshold = 0.1
    assert replace(s, barge_threshold=0.7).barge_threshold == 0.7


# --- stdlib-only import ----------------------------------------------------------------

def test_rules_import_with_the_standard_library_alone(tmp_path):
    # The sidecar's venv has no voicemanager package, and voicemanager.pipeline
    # imports torch. A fresh isolated interpreter loads the tyger directory as a
    # package by path, with numpy, torch, pydantic and voicemanager blocked.
    script = tmp_path / "load_rules.py"
    script.write_text(textwrap.dedent("""
        import importlib.abc
        import importlib.util
        import sys
        from pathlib import Path

        BLOCKED = {"pydantic", "fastapi", "numpy", "torch", "starlette", "voicemanager"}

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError("blocked: " + name)
                return None

        sys.meta_path.insert(0, Blocker())
        pkg_dir = Path(sys.argv[1])
        spec = importlib.util.spec_from_file_location(
            "tyger", pkg_dir / "__init__.py", submodule_search_locations=[str(pkg_dir)])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["tyger"] = pkg
        spec.loader.exec_module(pkg)
        import tyger.rules as rules
        import tyger.wire as wire
        engine = rules.TurnRules()
        events = []
        for i in range(8):
            events += engine.on_frame("speaking", 0.0, 0.9, -70.0, (i + 1) * 1280).events
        assert len(events) == 1 and isinstance(events[0], wire.BargeEvent)
        print(wire.format_event(events[0]))
    """))
    result = subprocess.run(
        [sys.executable, "-I", str(script), str(TYGER_DIR)],
        capture_output=True, text=True, cwd=tmp_path, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert '"event":"barge"' in result.stdout


def test_licence_and_notice_sit_next_to_the_vendored_rules():
    licence = (TYGER_DIR / "LICENSE").read_text()
    assert "Apache License" in licence and "Version 2.0" in licence
    notice = (TYGER_DIR / "NOTICE").read_text()
    assert "tyger-s2s" in notice and "97d011d" in notice
    assert "rules.py" in notice and "session.py" in notice
    assert "modified" in notice.lower()


# --- voice gate calibration -----------------------------------------------------------

def _room(levels):
    """Listening frames at the given energies, endpoint head quiet."""
    return [(L, 0.1, 0.0, db) for db in levels]


def _p20(levels):
    import numpy as np
    return float(np.percentile(levels, 20))


@pytest.mark.parametrize("levels, expected_gate", [
    ([-51.0 + 0.1 * (i % 7) for i in range(25)], None),   # a quiet room: p20 + 8
    ([-70.0] * 25, -50.0),                                  # under the configured gate
    ([-30.0 + i * 0.2 for i in range(25)], -38.0),          # over the cap
])
def test_the_gate_is_set_once_from_25_listening_frames(levels, expected_gate):
    engine = TurnRules()
    run(_room(levels[:24]), engine=engine)
    assert not engine.calibrated and engine.voice_gate_db == -50.0
    run(_room(levels[24:]), engine=engine)
    if expected_gate is None:
        expected_gate = _p20(levels) + 8.0
    assert engine.calibrated
    assert engine.voice_gate_db == pytest.approx(expected_gate)
    run(_room([-20.0] * 50), engine=engine)          # later frames never move it
    assert engine.voice_gate_db == pytest.approx(expected_gate)


def test_speaking_frames_do_not_count_toward_calibration():
    # Six echo frames per room frame: counted with the room, the echo would hold the
    # 20th percentile at -10 dB and the gate at the -38 dB cap.
    engine = TurnRules()
    frames = []
    for i in range(25):
        frames += speak(6, db=-10.0)                 # the reply's echo, loud
        frames += _room([-51.0])
    run(frames, engine=engine)
    assert engine.calibrated and engine.voice_gate_db == pytest.approx(-43.0)


def test_a_frame_between_the_two_gates_is_voice_before_calibration_only():
    # -45 dB: over the configured -50 dB gate, under the calibrated -43 dB one.
    # Before calibration it keeps a spoken turn open; after, it is silence.
    # 20 listening frames, all before calibration: 15 of them at -45 dB would end
    # the turn by silence if they were quiet.
    before = run(listen(5) + _room([-45.0] * 15))
    assert before == []
    engine = TurnRules()
    run(_room([-51.0] * 25), engine=engine)
    after = run(listen(10) + _room([-45.0] * 15), engine=engine)
    assert after == [(24, EndpointEvent(by="silence", mic_sample=end_of(24)))]


def test_a_room_over_the_configured_gate_ends_turns_by_silence_after_calibration():
    room = _room([-49.0] * 25)
    speech = listen(10)
    quiet = _room([-49.0] * 40)
    events = run(room + speech + quiet)
    ends = [i for i, e in events if e.by == "silence"]
    assert ends and ends[0] - (len(room) + len(speech) - 1) <= 16
    # With the gate pinned at -50 dB the room counts as voice and the turn never ends.
    pinned = TurnRuleSettings(gate_margin_db=0.0, gate_max_db=-50.0)
    assert run(room + speech + quiet + _room([-49.0] * 200), pinned) == []


def test_turn_ends_mode_changes_and_barges_keep_the_calibrated_gate():
    # Calibrated on a -49 dB room: gate -41 dB. Every later listening run is a
    # -60 dB room, which would calibrate to the -50 dB floor if calibration restarted.
    engine = TurnRules()
    later = _room([-60.0] * 30)
    run(_room([-49.0] * 25), engine=engine)
    assert engine.voice_gate_db == pytest.approx(-41.0)
    events = run(listen(10) + _room([-49.0] * 20) + later, engine=engine)  # a silence turn end
    assert any(e.by == "silence" for _i, e in events)
    assert engine.voice_gate_db == pytest.approx(-41.0)
    run(speak(10) + later, engine=engine)                                   # a reply, no barge
    assert engine.voice_gate_db == pytest.approx(-41.0)
    events = run(speak(6) + speak(4, yld=0.9, db=VOICED) + later, engine=engine)  # a barge
    assert any(isinstance(e, BargeEvent) for _i, e in events)
    assert engine.voice_gate_db == pytest.approx(-41.0)


def test_gate_settings_defaults_and_checks():
    s = TurnRuleSettings()
    assert s.gate_margin_db == 8.0 and s.gate_max_db == -38.0
    with pytest.raises(ValueError, match="gate_max_db"):
        TurnRuleSettings(voice_gate_db=-40.0, gate_max_db=-45.0)
    with pytest.raises(ValueError, match="gate_margin_db"):
        TurnRuleSettings(gate_margin_db=-1.0)
