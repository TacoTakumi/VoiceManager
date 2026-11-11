"""The Tyger sidecar applies a listening boundary (the client stopped listening).

Model-free, GPU-free and server-free: the turn rules are driven frame by frame, and
the sidecar app runs under the FastAPI TestClient with the fake scorer.

Covers: after ``boundary()`` the turn rules keep their voice gate and calibration
(and the calibration frames heard so far) but drop the turn, and the next hop is a
fresh first hop: its context starts over. Through the sidecar, the boundary
control message gets no answer, applies after the hops sent before it, makes the
session's scorer reset once at its next hop, keeps a pre-filter session enrolled
and leaves every other session alone; any other text message still gets an error
event.
"""
import pytest

from voicemanager.pipeline.tyger import rules as rules_mod
from voicemanager.pipeline.tyger import wire
from voicemanager.pipeline.tyger.rules import (
    CALIBRATION_FRAMES,
    FRAME_SAMPLES,
    HOP_FRAMES,
    TurnRules,
)

L, S = wire.MODE_LISTENING, wire.MODE_SPEAKING
VOICED, QUIET = -30.0, -80.0
# Everything a session's turn rules hold that is not the voice gate.
_GATE_STATE = {"_gate", "_calibrated", "_calibration", "settings", "_key"}


def _turn_state(engine: TurnRules) -> dict:
    return {k: (list(v) if k == "_seed" else v)
            for k, v in vars(engine).items() if k not in _GATE_STATE}


def _feed(engine: TurnRules, frames, start: int = 0) -> list:
    """(mode, p_end, yield, energy_db) frames -> their events, a hop at a time."""
    events = []
    for i, (mode, p_end, yld, db) in enumerate(frames):
        if (start + i) % HOP_FRAMES == 0:
            engine.begin_hop(mode)
        events += engine.on_frame(mode, p_end, yld, db, (start + i + 1) * FRAME_SAMPLES).events
    return events


def _listen(n, p_end=0.1, db=VOICED):
    return [(L, p_end, 0.0, db)] * n


def _speak(n, yld=0.0, db=VOICED):
    return [(S, 0.0, yld, db)] * n


# --- the turn rules ---------------------------------------------------------------

def _mid_turn() -> TurnRules:
    """A calibrated session in the middle of a spoken turn, one frame short of an
    endpoint by the head."""
    engine = TurnRules()
    _feed(engine, _listen(28) + _listen(3) + [(L, 0.95, 0.0, VOICED)])
    assert engine.calibrated
    assert engine._in_turn and engine._voiced_ms > 0 and engine._ep_run == 1
    return engine


def _barged_in_reply() -> TurnRules:
    """A calibrated session whose user cut into a reply that still plays."""
    engine = TurnRules()
    events = _feed(engine, _listen(28) + _speak(12) + _speak(4, yld=0.9))
    assert any(isinstance(e, wire.BargeEvent) for e in events)
    assert engine._barged and engine._talk_frames
    return engine


@pytest.mark.parametrize("make", [_mid_turn, _barged_in_reply])
def test_a_boundary_keeps_the_gate_and_drops_the_turn(make):
    engine = make()
    gate = engine.voice_gate_db
    assert gate != engine.settings.voice_gate_db       # calibration moved it
    engine.boundary()
    assert engine.voice_gate_db == gate
    assert engine.calibrated is True
    assert engine._in_turn is False
    assert engine._voiced_ms == 0.0
    assert (engine._warm, engine._ep_run, engine._yield_run, engine._silent_run) == (0, 0, 0, 0)
    assert (engine._talk_frames, engine._quiet_frames) == (0, 0)
    assert engine._barged is False
    assert list(engine._seed) == []
    assert engine._mode == L


@pytest.mark.parametrize("make", [_mid_turn, _barged_in_reply])
@pytest.mark.parametrize("mode", [L, S])
def test_the_next_hop_after_a_boundary_is_a_fresh_first_hop(make, mode):
    engine, fresh = make(), TurnRules()
    engine.boundary()
    assert engine.begin_hop(mode) is True              # the heads' context starts over
    assert fresh.begin_hop(mode) is True
    assert _turn_state(engine) == _turn_state(fresh)
    # The hop after it is an ordinary one: nothing more starts over.
    assert engine.begin_hop(mode) is False


def test_after_a_boundary_the_session_decides_like_a_fresh_calibrated_one():
    # The same frames after a boundary give the events a session gives that has only
    # calibrated: nothing from the dropped turn counts toward the new one.
    after = (_listen(2) + [(L, 0.95, 0.0, VOICED)] * 2 + _listen(4)
             + _listen(16, db=QUIET) + _speak(8) + _speak(4, yld=0.9))
    engine = _mid_turn()
    engine.boundary()
    reference = TurnRules()
    _feed(reference, _listen(28))
    reference._pending_reset = True                    # as a fresh first hop
    assert reference.voice_gate_db == engine.voice_gate_db
    events = _feed(engine, after)
    assert events and events == _feed(reference, after)


def test_a_boundary_keeps_the_calibration_frames_heard_so_far():
    engine = TurnRules()
    _feed(engine, _listen(12, db=-60.0))
    assert not engine.calibrated
    engine.boundary()
    _feed(engine, _listen(CALIBRATION_FRAMES - 12, db=-60.0))
    assert engine.calibrated                           # 12 + 13 frames: one calibration


def test_a_boundary_on_a_fresh_session_changes_nothing():
    engine, fresh = TurnRules(), TurnRules()
    engine.boundary()
    assert vars(engine).keys() == vars(fresh).keys()
    assert _turn_state(engine) == _turn_state(fresh)
    assert engine.begin_hop(L) is True


# --- through the sidecar app ------------------------------------------------------

def _recording_sessions(monkeypatch):
    """Keep every Session the sidecar app makes, in connection order."""
    from voicemanager.pipeline.tyger import server
    made = []

    class RecordingSession(server.Session):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(server, "Session", RecordingSession)
    return made


def _hop(mode: str, index: int, user: bytes | None = None) -> bytes:
    silence = bytes(wire.CHANNEL_BYTES)
    return wire.pack_hop(mode, (index + 1) * wire.HOP_SAMPLES,
                         user if user is not None else silence, silence)


# Eight voiced listening hops: the gate calibrates and a turn is under way, with
# the head one frame short of an endpoint.
TURN_HOPS = 8
TURN_FRAMES = [[0.1, 0.1, VOICED]] * (TURN_HOPS * HOP_FRAMES - 1) + [[0.95, 0.1, VOICED]]


def _app(prefilter_factory=None):
    from voicemanager.pipeline.tyger import scorer, server
    fake = scorer.FakeScorer({"a": TURN_FRAMES, "b": TURN_FRAMES})
    return server.create_app(turn_scorer=fake, prefilter_factory=prefilter_factory)


def _sync(ws) -> list:
    """Send a text message that is not a control and return its answer: once it
    is in, every message sent before it has been handled. A message that holds
    only the pre-filter's barge-state lines is not the answer and is skipped."""
    ws.send_text("not a control")
    while True:
        events = wire.parse_events(ws.receive_text())
        if not all(isinstance(e, wire.BargeStateEvent) for e in events):
            return events


def _assert_turn_dropped(engine: TurnRules, gate: float) -> None:
    assert engine.voice_gate_db == gate and engine.calibrated
    assert engine._in_turn is False and engine._voiced_ms == 0.0
    assert (engine._warm, engine._ep_run, engine._silent_run) == (0, 0, 0)


def test_a_boundary_message_gets_no_answer_and_resets_the_scorer_at_the_next_hop(monkeypatch):
    from fastapi.testclient import TestClient
    made = _recording_sessions(monkeypatch)
    with TestClient(_app()) as client, client.websocket_connect("/tyger?session=a") as ws:
        # The hops and the boundary go out at once: the boundary applies after
        # every hop sent before it has been scored.
        for k in range(TURN_HOPS):
            ws.send_bytes(_hop(L, k))
        ws.send_text(wire.format_boundary_control())
        answer = _sync(ws)
        # The only answer is the one to the non-control text: none to the hops
        # (no events) and none to the boundary.
        assert len(answer) == 1 and isinstance(answer[0], wire.ErrorEvent)
        assert "binary" in answer[0].message
        session = made[0]
        assert session.scorer.hops == TURN_HOPS
        engine = session.rules
        assert engine.calibrated
        _assert_turn_dropped(engine, engine.voice_gate_db)
        resets = session.scorer.resets
        ws.send_bytes(_hop(L, TURN_HOPS))
        _sync(ws)
        assert session.scorer.resets == resets + 1
        # Only the first hop after the boundary starts over.
        ws.send_bytes(_hop(L, TURN_HOPS + 1))
        _sync(ws)
        assert session.scorer.resets == resets + 1


def test_a_boundary_with_spaces_and_extra_keys_is_still_a_boundary(monkeypatch):
    from fastapi.testclient import TestClient
    made = _recording_sessions(monkeypatch)
    with TestClient(_app()) as client, client.websocket_connect("/tyger?session=a") as ws:
        for k in range(TURN_HOPS):
            ws.send_bytes(_hop(L, k))
        ws.send_text('  {"control": "boundary", "note": 1}\n')
        answer = _sync(ws)
        assert len(answer) == 1
        assert made[0].rules._in_turn is False


@pytest.mark.parametrize("text", [
    "a text message", '{"control": "pause"}', '{"control": true}', "[]",
    wire.BOUNDARY_CONTROL + "\n" + wire.BOUNDARY_CONTROL,
])
def test_any_other_text_message_still_gets_an_error_and_changes_nothing(monkeypatch, text):
    from fastapi.testclient import TestClient
    made = _recording_sessions(monkeypatch)
    with TestClient(_app()) as client, client.websocket_connect("/tyger?session=a") as ws:
        for k in range(TURN_HOPS):
            ws.send_bytes(_hop(L, k))
        ws.send_text(text)
        events = wire.parse_events(ws.receive_text())
        assert len(events) == 1 and isinstance(events[0], wire.ErrorEvent)
        assert "expected a binary hop, got a text message" == events[0].message
        engine = made[0].rules
        assert engine._in_turn is True and engine._voiced_ms > 0
        resets = made[0].scorer.resets
        ws.send_bytes(_hop(L, TURN_HOPS))
        _sync(ws)
        assert made[0].scorer.resets == resets


def test_a_boundary_leaves_every_other_session_alone(monkeypatch):
    from fastapi.testclient import TestClient
    made = _recording_sessions(monkeypatch)
    with TestClient(_app()) as client, \
            client.websocket_connect("/tyger?session=a") as ws_a, \
            client.websocket_connect("/tyger?session=b") as ws_b:
        for k in range(TURN_HOPS):
            ws_a.send_bytes(_hop(L, k))
            ws_b.send_bytes(_hop(L, k))
        _sync(ws_a)
        _sync(ws_b)
        a, b = made
        before = _turn_state(b.rules), b.rules.voice_gate_db, b.rules.calibrated
        resets_b = b.scorer.resets
        ws_a.send_text(wire.format_boundary_control())
        _sync(ws_a)
        assert a.rules._in_turn is False
        assert (_turn_state(b.rules), b.rules.voice_gate_db, b.rules.calibrated) == before
        assert b.rules._in_turn is True
        ws_b.send_bytes(_hop(L, TURN_HOPS))
        _sync(ws_b)
        assert b.scorer.resets == resets_b
        assert b.rules._in_turn is True


class _SilentFactory:
    """Enrolls at once into a filter that outputs silence, 384 samples late."""

    def __init__(self):
        self.enrolled = 0

    def enroll(self, sample):
        import numpy as np
        self.enrolled += 1

        class _F:
            delay = 384

            def push(self, samples):
                return np.zeros(len(samples), np.float32)
        return _F()


def test_a_pre_filter_session_stays_enrolled_after_a_boundary(monkeypatch):
    import numpy as np
    from fastapi.testclient import TestClient
    from voicemanager.pipeline.tyger import prefilter
    made = _recording_sessions(monkeypatch)
    factory = _SilentFactory()
    rng = np.random.default_rng(7)

    def voiced():
        return (rng.uniform(-0.1, 0.1, wire.HOP_SAMPLES) * 32768).astype("<i2").tobytes()

    with TestClient(_app(factory)) as client, \
            client.websocket_connect("/tyger?session=a") as ws:
        # 20 voiced listening hops (6.4 s): 2 s calibrate the gate, then the
        # pre-filter takes its 3 s voice sample.
        hops = 20
        for k in range(hops):
            ws.send_bytes(_hop(L, k, voiced()))
        _sync(ws)
        session = made[0]
        assert isinstance(session.scorer, prefilter.PrefilterSessionScorer)
        assert session.scorer.enrolled and factory.enrolled == 1
        inner = session.scorer._inner
        resets = inner.resets
        gate = session.rules.voice_gate_db
        ws.send_text(wire.format_boundary_control())
        _sync(ws)
        _assert_turn_dropped(session.rules, gate)
        ws.send_bytes(_hop(L, hops, voiced()))
        _sync(ws)
        assert inner.resets == resets + 1                 # the reset reaches the scorer
        assert session.scorer.enrolled and factory.enrolled == 1
        # A reply after the boundary is filtered with the enrollment it had.
        ws.send_bytes(_hop(S, hops + 1, voiced()))
        _sync(ws)
        assert session.scorer.enrolled and factory.enrolled == 1


def test_the_sidecar_docstring_names_the_boundary_control():
    from voicemanager.pipeline.tyger import server
    assert "boundary" in server.__doc__
    assert rules_mod.TurnRules.boundary.__doc__


def test_the_fake_sidecar_records_a_boundary_and_answers_nothing():
    from websockets.sync.client import connect
    from tyger_fake_sidecar import FakeTygerSidecar
    fake = FakeTygerSidecar().start()
    try:
        with connect(fake.url + "?session=a", open_timeout=5) as ws:
            ws.send(_hop(L, 0))
            ws.send(_hop(L, 1))
            ws.send(wire.format_boundary_control())
            ws.send(_hop(L, 2))
            ws.send("not a control")
            events = wire.parse_events(ws.recv(timeout=5))
        assert len(events) == 1 and isinstance(events[0], wire.ErrorEvent)
        session = fake.sessions[0]
        assert fake.wait_for(lambda: len(session.bad) == 1)
        assert session.boundaries == [2]
        assert len(session.hops) == 3
        assert session.bad == ["not a control"]
    finally:
        fake.stop()
