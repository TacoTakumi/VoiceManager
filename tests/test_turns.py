"""Turn-boundary tests: the server-owned turn counter stamps turn_id.

The first tests cover the additive `turn_id` field on TranscriptionMessage and the
server-owned TurnTracker whose advance() method opens a turn -- bumping the counter
AND clearing the STT live partial -- so every transcription frame carries the turn
that was open when ITS AUDIO WAS RECEIVED, not the turn current at send time.
No receive loop, no turn_start frame: the turn-advance
method is exercised directly. Later tasks add the turn_start control frame,
the through-the-loop partial clearing, the crossing-final tagging and
the every-engine coverage to this same module.
"""
import asyncio
import json

import authkit

from voicemanager.protocol import TranscriptionMessage
from voicemanager.transport.turns import TurnTracker


# --- additive turn_id field on TranscriptionMessage ----------------

def test_transcription_message_turn_id_defaults_zero():
    msg = TranscriptionMessage(text="hello")
    assert msg.turn_id == 0
    assert json.loads(msg.model_dump_json())["turn_id"] == 0


def test_transcription_message_other_fields_unchanged_by_turn_id():
    # turn_id is additive: existing whisper/qwen/nemotron emit sites construct
    # TranscriptionMessage without it, so those call sites need no changes.
    msg = TranscriptionMessage(text="hi", is_final=True, segment_index=2, endpoint=True)
    payload = json.loads(msg.model_dump_json())
    assert payload["text"] == "hi"
    assert payload["is_final"] is True
    assert payload["segment_index"] == 2
    assert payload["endpoint"] is True
    assert payload["turn_id"] == 0


# --- a connection that never opens a turn stays at turn_id=0 --------

def test_stamp_stays_zero_when_no_turn_opened():
    tracker = TurnTracker()
    tracker.mark_received()
    for is_final in (False, True):
        msg = tracker.stamp(TranscriptionMessage(text="x", is_final=is_final))
        assert msg.turn_id == 0


# --- advance() is the callable server-side turn-advance method ------

def test_advance_increments_the_server_owned_counter():
    tracker = TurnTracker()
    assert tracker.current == 0
    assert tracker.advance() == 1
    assert tracker.advance() == 2
    assert tracker.current == 2


def test_advance_clears_the_live_partial():
    # Opening a turn bumps the counter AND clears the STT live partial.
    # This proves the mechanism: advance() invokes the injected clear callback.
    # The real per-engine wiring is covered by the WS receive-loop tests below.
    cleared = []
    tracker = TurnTracker(clear_partial=lambda: cleared.append(True))
    tracker.advance()
    assert cleared == [True]


def test_advance_without_clear_callback_is_a_noop_clear():
    # An engine with no live partial simply has nothing to clear (whisper, tested below):
    # advance() must still bump the counter and never raise.
    tracker = TurnTracker()
    assert tracker.advance() == 1


# --- the id is the turn open at RECEIVE time, not send time --

def test_stamp_uses_receive_time_turn_not_send_time():
    tracker = TurnTracker()
    # Audio for utterance A is received while turn 0 is open.
    tracker.mark_received()
    frame_a = tracker.stamp(TranscriptionMessage(text="switch to red alert"))
    assert frame_a.turn_id == 0

    # A turn opens -- advance() called DIRECTLY (no turn_start frame, no loop).
    tracker.advance()
    # A late frame for utterance A (whose audio was received in turn 0) is SENT
    # after the boundary: it must keep turn 0, the turn open when its audio was
    # received, NOT the turn current now (1).
    late_a = tracker.stamp(TranscriptionMessage(text="switch to red alert", is_final=True))
    assert late_a.turn_id == 0

    # Newly-received audio (utterance B) is stamped with the incremented turn.
    tracker.mark_received()
    frame_b = tracker.stamp(TranscriptionMessage(text="all stop"))
    assert frame_b.turn_id == 1


def test_advance_does_not_retag_audio_already_received():
    # advance() bumps the open turn but must not disturb the receive-time snapshot
    # of audio already in flight, even across several advances.
    tracker = TurnTracker()
    tracker.mark_received()          # received while turn 0 is open
    tracker.advance()                # open turn 1
    tracker.advance()                # open turn 2
    assert tracker.stamp(TranscriptionMessage(text="held")).turn_id == 0
    tracker.mark_received()          # new audio received while turn 2 is open
    assert tracker.stamp(TranscriptionMessage(text="new")).turn_id == 2


# --- frames emitted by the real STT driver carry the stamp --
# Reuse the NeMo-free scripted-nemotron driver (no receive loop): a stamping sink
# mirrors the handler's _emit, and the turn is advanced DIRECTLY. This exercises
# the same _drive_nemotron_stt code path the WS handler uses.
from test_nemotron_pipeline import _FakeShared, _ScriptedNemotron, _ONE_CHUNK
from voicemanager.transport.websocket import _drive_nemotron_stt


def _stamping_sink(tracker, sink):
    """A send sink identical in spirit to the handler's _emit: stamp then collect."""
    async def send(msg):
        sink.append(tracker.stamp(msg))
    return send


def test_stamp_on_driver_frames_carries_incremented_turn_id_after_advance():
    tracker = TurnTracker()
    sent: list[TranscriptionMessage] = []
    send = _stamping_sink(tracker, sent)

    async def run():
        # Utterance in turn 0: partial then EoU final (hold disabled -> immediate).
        pipe0 = _ScriptedNemotron(
            [("this", None), ("this facility", "this facility")],
            shared_pipeline=_FakeShared(), stream_id=1, att_context_size=[70, 13],
        )
        tracker.mark_received()
        await _drive_nemotron_stt(pipe0, None, _ONE_CHUNK, send)
        await _drive_nemotron_stt(pipe0, None, _ONE_CHUNK, send)

        # Open a new turn DIRECTLY (no turn_start frame).
        tracker.advance()

        # New audio received while turn 1 is open.
        pipe1 = _ScriptedNemotron(
            [("red", None), ("red alert", "red alert")],
            shared_pipeline=_FakeShared(), stream_id=2, att_context_size=[70, 13],
        )
        tracker.mark_received()
        await _drive_nemotron_stt(pipe1, None, _ONE_CHUNK, send)
        await _drive_nemotron_stt(pipe1, None, _ONE_CHUNK, send)

    asyncio.run(run())

    assert len(sent) == 4, [m.text for m in sent]
    turn0, turn1 = sent[:2], sent[2:]
    assert all(m.turn_id == 0 for m in turn0), [(m.text, m.turn_id) for m in turn0]
    assert all(m.turn_id == 1 for m in turn1), [(m.text, m.turn_id) for m in turn1]
    # Sanity: the frames are the ones we scripted, in order.
    assert "this" in turn0[0].text and "facility" in turn0[1].text
    assert "red" in turn1[0].text and "alert" in turn1[1].text


# --- the turn_start control frame and its turn_started ack ------------
# turn_start is a client-callable control frame; the server owns the counter and the frame
# handler is a thin delegation to advance(). These exercise the frame end-to-end
# through the real receive loop with a TestClient, plus a structural check that the
# receive-loop branch owns no counter/partial logic of its own.
import inspect
import time

from fastapi.testclient import TestClient

from voicemanager.app import create_app
from voicemanager.protocol import MessageType, TurnStartedMessage
from voicemanager.transport import websocket as ws_module

ACK_BUDGET_S = 0.1  # the turn_started ack arrives within 100ms


def _connected_ws(client):
    """Open a stream, drain the ready handshake, return the websocket context manager.

    Default whisper engine, no models: TestClient(app) WITHOUT `with client:` skips the
    lifespan so nothing is loaded, and turn_start needs no STT/TTS model -- no audio is
    ever sent, so the None pipeline is never exercised.
    """
    return authkit.ws_connect(client, "/ws/stream")


def test_turn_start_and_turn_started_are_known_message_types():
    # both the request and its ack are first-class protocol message types.
    assert MessageType.turn_start.value == "turn_start"
    assert MessageType.turn_started.value == "turn_started"


def test_turn_started_ack_message_serializes_type_and_turn_id():
    # the ack is a dedicated {"type":"turn_started","turn_id":N} frame.
    payload = json.loads(TurnStartedMessage(turn_id=7).model_dump_json())
    assert payload == {"type": "turn_started", "turn_id": 7}


def test_turn_start_frame_acks_with_turn_started_id_one_within_budget():
    # client sends {"type":"turn_start"} and gets {"type":"turn_started",
    # "turn_id":1} promptly.
    client = TestClient(create_app())
    with _connected_ws(client) as ws:
        assert ws.receive_json()["type"] == "ready"

        t0 = time.monotonic()
        ws.send_json({"type": "turn_start"})
        ack = ws.receive_json()
        ack_at = time.monotonic() - t0

        assert ack == {"type": "turn_started", "turn_id": 1}
        assert ack_at < ACK_BUDGET_S, f"turn_started ack took {ack_at*1000:.0f}ms"


def test_second_turn_start_yields_turn_id_two_monotonic():
    # the server owns the counter; a second turn_start advances it to 2.
    client = TestClient(create_app())
    with _connected_ws(client) as ws:
        assert ws.receive_json()["type"] == "ready"

        ws.send_json({"type": "turn_start"})
        assert ws.receive_json()["turn_id"] == 1
        ws.send_json({"type": "turn_start"})
        assert ws.receive_json()["turn_id"] == 2


def test_turn_start_counter_is_monotonic_and_never_rewinds():
    # across many turn_start calls the ids are strictly increasing 1..N and
    # never repeat or rewind.
    client = TestClient(create_app())
    with _connected_ws(client) as ws:
        assert ws.receive_json()["type"] == "ready"

        ids = []
        for _ in range(5):
            ws.send_json({"type": "turn_start"})
            ack = ws.receive_json()
            assert ack["type"] == "turn_started"
            ids.append(ack["turn_id"])

        assert ids == [1, 2, 3, 4, 5], ids


def _turn_start_branch_source() -> str:
    """The body of the `elif msg_type == 'turn_start'` branch of the real receive loop."""
    lines = inspect.getsource(ws_module.websocket_endpoint).splitlines()
    start = next(i for i, ln in enumerate(lines) if 'msg_type == "turn_start"' in ln)
    indent = len(lines[start]) - len(lines[start].lstrip())
    body = []
    for line in lines[start + 1:]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break  # dedented back out -- the branch is over
        body.append(line)
    assert body, "could not locate the turn_start branch body"
    return "\n".join(body)


def test_the_turn_start_branch_is_a_thin_delegation():
    """Structural: the branch delegates to advance() and owns no counter/partial
    logic of its own, so a future server-side barge-in can open a turn from the STT path
    by calling the SAME method."""
    branch = _turn_start_branch_source()

    assert "turn_tracker.advance(" in branch  # it delegates

    forbidden = {
        "+= 1": "increments a counter itself instead of delegating to advance()",
        "_current": "reaches into the tracker's private counter state",
        "_received": "reaches into the tracker's receive-time snapshot",
        "reset(": "clears the STT pipeline itself instead of delegating",
        "clear_partial": "clears the live partial itself instead of delegating",
        "_last_partial": "reaches into the STT pipeline's live partial",
    }
    for token, why in forbidden.items():
        assert token not in branch, f"turn_start branch {why}: {token!r}"


# --- opening a turn clears the uncommitted live partial -------
# Through the REAL WS receive loop (not a hand-wired callback -- the advance() tests above pinned
# the callback mechanism), so this proves the handler builds the connection's
# TurnTracker with a clear_partial that reaches the live nemotron pipeline/hold state.
# turn_id fences whole FRAMES but cannot fence text INSIDE a frame: a stale uncommitted
# partial (here the endpoint-hold's reopen stitch prefix) would otherwise stitch the old
# turn's words into the new turn's first frame. Reuses the scripted-nemotron + TestClient
# pattern (no NeMo import, no model load).
from voicemanager.session import SessionRegistry


def _scripted_nemotron_factory(script):
    """Drop-in for NemotronSTTPipeline that ignores NeMo and replays `script`."""
    def _factory(**kwargs):
        return _ScriptedNemotron(list(script), **kwargs)
    return _factory


def test_turn_start_clears_live_partial_so_no_stale_stitch_into_new_turn(monkeypatch):
    # Generous hold window + session timeout so the exchange never confirms on the
    # audio clock or the wall-clock fallback -- state transitions are driven by frame,
    # not by time, keeping the frame sequence deterministic.
    monkeypatch.setenv("VM_STT_ENDPOINT_HOLD_MS", "10000")
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "30")
    # A: EoU "switch to red alert" -> provisional endpoint + hold (seg 0, turn 0).
    # Reopen "resume" -> mid-stitch: "switch to red alert" becomes the stitch prefix,
    # emitted as a stitched interim (still turn 0). After turn_start a fresh utterance
    # decodes "affirmative captain" -- it must carry NONE of the pre-turn_start words.
    monkeypatch.setattr(
        "voicemanager.pipeline.nemotron_stt.NemotronSTTPipeline",
        _scripted_nemotron_factory([
            ("switch to red alert", "switch to red alert"),
            ("resume", None),
            ("affirmative captain", None),
        ]),
    )
    import voicemanager.config as cfg
    cfg._settings = None
    try:
        app = create_app()
        app.state.stt_engine_type = "nemotron"
        app.state.nemotron_pipeline = _FakeShared()
        app.state.session_registry = SessionRegistry()
        client = TestClient(app)

        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"

            # A -> the held/endpointed final: a provisional endpoint frame, tagged turn 0.
            ws.send_bytes(_ONE_CHUNK)
            prov = ws.receive_json()
            assert prov["endpoint"] is True and prov["is_final"] is False
            assert prov["text"] == "switch to red alert"
            assert prov["turn_id"] == 0 and prov["segment_index"] == 0

            # Reopen -> a stitched interim carrying the held text forward (still turn 0).
            ws.send_bytes(_ONE_CHUNK)
            stitched = ws.receive_json()
            assert stitched["is_final"] is False and stitched["endpoint"] is False
            assert stitched["text"] == "switch to red alert resume"
            assert stitched["turn_id"] == 0

            # Open a new turn: the uncommitted live partial (the stitch prefix that would
            # carry "switch to red alert" into turn 1) is cleared. The pending hold's
            # endpointed final above is NOT dropped -- it already emitted (turn 0).
            ws.send_json({"type": "turn_start"})
            assert ws.receive_json() == {"type": "turn_started", "turn_id": 1}

            # The new turn's first utterance carries only post-turn_start words.
            ws.send_bytes(_ONE_CHUNK)
            fresh = ws.receive_json()
            assert fresh["type"] == "transcription"
            assert fresh["turn_id"] == 1
            assert fresh["text"] == "affirmative captain"      # NOT "switch to red alert affirmative captain"
            assert "red alert" not in fresh["text"]            # no stale stitch into the new turn
            assert "resume" not in fresh["text"]
    finally:
        cfg._settings = None


# --- a final crossing the turn boundary keeps the PREVIOUS turn_id + segment --
# The endpoint-hold carries the turn its held audio was received in, so a hold that
# CONFIRMS after a turn_start still tags the confirmed final with the OLD turn. The
# id on the frame is itself the race-free fence: the client reads the revision as
# 'a revision of the turn I dispatched', not a frame of the new turn. segment_index
# is never rewound by turn_start, so the confirmed final keeps its original index and
# the next turn's first utterance takes the NEXT index, never turn 0's.
from voicemanager.transport.endpoint_hold import EndpointHoldMachine


def test_hold_machine_carries_turn_id_through_confirm():
    # Data carry: begin_hold records the turn its audio was received in and the
    # confirmed final surfaces it via last_final_turn, independent of any later receive.
    m = EndpointHoldMachine(hold_ms=1000)  # 16000-sample window
    m.begin_hold("switch to red alert", "en", 0, turn_id=0)
    assert m.advance(15999) is None                              # still holding
    assert m.advance(1) == ("switch to red alert", "en", 0)      # confirmed (3-tuple unchanged)
    assert m.last_final_turn == 0                                 # the held turn, not the send-time turn


def test_hold_machine_take_held_surfaces_turn_id():
    # The wall-clock/teardown flush path (take_held) surfaces the same held turn.
    m = EndpointHoldMachine(hold_ms=1000)
    m.begin_hold("apple", "en", 3, turn_id=0)
    assert m.take_held() == ("apple", "en", 3)
    assert m.last_final_turn == 0


def test_hold_machine_reopen_preserves_original_turn_into_stitch():
    # A reopen folds the held turn into the stitch so a stitched turn keeps its ORIGINAL
    # turn even if the resumed speech is received after a turn_start.
    m = EndpointHoldMachine(hold_ms=1000)
    m.begin_hold("this facility", "en", 0, turn_id=0)
    m.reopen()
    assert m.stitch_turn == 0
    # The next endpoint (received in turn 1) re-holds the stitched turn under turn 0.
    stitched = m.stitch("is not authorized")
    m.begin_hold(stitched, "en", 0, turn_id=1)
    assert m.take_held() == ("this facility is not authorized", "en", 0)
    assert m.last_final_turn == 0                                 # original turn, not the resumed turn


def _record_scripted_nemotron_factory(script, created):
    """Scripted-nemotron factory that records each created pipeline (to append steps later)."""
    def _factory(**kwargs):
        pipe = _ScriptedNemotron(list(script), **kwargs)
        created.append(pipe)
        return pipe
    return _factory


_SILENCE_SAMPLE = b"\x00\x00"


def _silence(n_samples: int) -> bytes:
    return _SILENCE_SAMPLE * n_samples


def test_final_crossing_turn_boundary_keeps_previous_turn_id_and_segment(monkeypatch):
    # Generous hold window + session timeout: confirmation is driven by fed audio, and the
    # wall-clock fallback (capped to the hold window) has a full second of slack so the
    # in-process TestClient always delivers turn_start first -- the sequence stays by-frame.
    monkeypatch.setenv("VM_STT_ENDPOINT_HOLD_MS", "1000")   # 16000-sample audio-clock window
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "30")
    # A: EoU "switch to red alert" -> provisional endpoint (turn 0, seg 0), begins the hold.
    # B ("affirmative captain") is appended AFTER A confirms, so the confirming silence frame
    # can't decode it into a reopen -- B is a genuinely NEW post-turn_start utterance.
    created: list = []
    monkeypatch.setattr(
        "voicemanager.pipeline.nemotron_stt.NemotronSTTPipeline",
        _record_scripted_nemotron_factory(
            [("switch to red alert", "switch to red alert")], created),
    )
    import voicemanager.config as cfg
    cfg._settings = None
    try:
        app = create_app()
        app.state.stt_engine_type = "nemotron"
        app.state.nemotron_pipeline = _FakeShared()
        app.state.session_registry = SessionRegistry()
        client = TestClient(app)

        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"

            # A -> provisional endpoint, tagged with the turn open at receive time (0), seg 0.
            ws.send_bytes(_ONE_CHUNK)
            prov = ws.receive_json()
            assert prov["endpoint"] is True and prov["is_final"] is False
            assert prov["text"] == "switch to red alert"
            assert prov["turn_id"] == 0 and prov["segment_index"] == 0

            # Open turn 1 while A's hold is still pending (not yet confirmed).
            ws.send_json({"type": "turn_start"})
            assert ws.receive_json() == {"type": "turn_started", "turn_id": 1}

            # Post-turn_start SILENCE crosses the audio-clock window and confirms A. The
            # confirmed final is a REVISION of seg 0: it MUST keep turn_id=0 (not 1) and
            # seg 0 -- the race-free fence that makes it readable as 'the turn I dispatched'.
            ws.send_bytes(_silence(16000))
            confirmed = ws.receive_json()
            assert confirmed["type"] == "transcription"
            assert confirmed["is_final"] is True and confirmed["endpoint"] is False
            assert confirmed["text"] == "switch to red alert"
            assert confirmed["turn_id"] == 0            # crosses the boundary -> keeps the OLD turn
            assert confirmed["segment_index"] == 0      # never rewound by turn_start

            # A genuinely new post-turn_start utterance: frames carry turn_id=1, and its
            # first final takes the NEXT segment (1), never reusing turn 0's seg 0.
            created[0]._script.append(("affirmative captain", "affirmative captain"))
            ws.send_bytes(_ONE_CHUNK)
            new_prov = ws.receive_json()
            assert new_prov["endpoint"] is True and new_prov["is_final"] is False
            assert new_prov["text"] == "affirmative captain"
            assert new_prov["turn_id"] == 1             # post-turn_start audio is turn 1
            assert new_prov["segment_index"] == 1       # monotonic, not reused for a new utterance
    finally:
        cfg._settings = None


# --- the turn affordance works on EVERY STT engine, not just nemotron --
# turn_start is acked and turn_id is stamped on transcription frames regardless of
# VM_STT_ENGINE. nemotron already routes every frame through the handler's stamping sink
# (_emit); whisper (_send_transcription) and qwen (interim/final sends) go through
# the same sink so no engine emits an un-stamped frame. Clearing the uncommitted live
# partial applies to whichever engines hold one: qwen accumulates a streaming partial (so
# opening a turn drops it), whisper emits only VAD-gated finals and holds no partial text
# to stitch (nothing to clear). Driven model-free through the real WS loop with per-engine
# fakes, mirroring the scripted-nemotron + TestClient pattern above (no model, no GPU).
import pytest


class _FakeWhisperSeg:
    """A faster-whisper-like segment: text plus the fields _send_transcription reads."""
    def __init__(self, text: str):
        self.text = text
        self.avg_logprob = -0.1
        self.start = 0.0
        self.end = 1.0


class _FakeWhisperInfo:
    language = "en"


def _fake_whisper_factory(inferences):
    """Drop-in for STTPipeline (whisper): feed() triggers while script remains; each
    run_inference() pops one inference's worth of segment texts. No VAD/model/executor."""
    class _FakeWhisper:
        def __init__(self, **kwargs):
            self._inferences = list(inferences)

        def feed(self, data: bytes) -> bool:
            return bool(self._inferences)

        async def run_inference(self, loop):
            texts = self._inferences.pop(0) if self._inferences else []
            return [_FakeWhisperSeg(t) for t in texts], _FakeWhisperInfo(), 0.01

        def dispose(self):
            pass

    return _FakeWhisper


def _fake_qwen_factory(words):
    """Drop-in for QwenSTTPipeline: accumulates decoded words like the real streaming state,
    so reset() (the clear_partial) genuinely drops the uncommitted partial. Each feed pops
    the next scripted word(s) and appends; get_current_text() returns the running partial."""
    class _FakeQwen:
        def __init__(self, **kwargs):
            self._words = list(words)
            self._text = ""
            self._seg = 0

        async def feed(self, data: bytes) -> bool:
            if not self._words:
                return False
            self._text = f"{self._text} {self._words.pop(0)}".strip()
            return True

        def get_current_text(self):
            return self._text, "en", self._seg

        async def finalize(self):
            return "", "en", self._seg      # no teardown final in these tests

        def reset(self):
            self._text = ""                 # drop the uncommitted partial
            self._seg += 1                  # monotonic; never rewound

        def dispose(self):
            pass

    return _FakeQwen


def _install_engine(monkeypatch, engine: str):
    """Monkeypatch the given STT engine's pipeline with a model-free fake that yields one
    post-turn_start transcription frame, and return a configured app for it."""
    if engine == "whisper":
        monkeypatch.setattr(
            "voicemanager.transport.websocket.STTPipeline",
            _fake_whisper_factory([["affirmative captain"]]),
        )
    elif engine == "qwen":
        monkeypatch.setattr(
            "voicemanager.transport.websocket.QwenSTTPipeline",
            _fake_qwen_factory(["affirmative captain"]),
        )
    else:  # nemotron
        monkeypatch.setattr(
            "voicemanager.pipeline.nemotron_stt.NemotronSTTPipeline",
            _scripted_nemotron_factory([("affirmative captain", None)]),
        )
    app = create_app()
    app.state.stt_engine_type = engine
    app.state.session_registry = SessionRegistry()
    if engine == "nemotron":
        app.state.nemotron_pipeline = _FakeShared()
    return app


@pytest.mark.parametrize("engine", ["whisper", "qwen", "nemotron"])
def test_turn_start_stamps_turn_id_on_every_stt_engine(engine, monkeypatch):
    # with any VM_STT_ENGINE, turn_start returns turn_started(turn_id=1) and the
    # first transcription frame for post-turn_start audio carries turn_id=1 -- no engine
    # raises, and none emits a turn_id that lags the acked value.
    import voicemanager.config as cfg
    cfg._settings = None
    try:
        app = _install_engine(monkeypatch, engine)
        client = TestClient(app)
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"

            ws.send_json({"type": "turn_start"})
            assert ws.receive_json() == {"type": "turn_started", "turn_id": 1}

            ws.send_bytes(_ONE_CHUNK)
            frame = ws.receive_json()
            assert frame["type"] == "transcription", (engine, frame)
            assert frame["turn_id"] == 1, (engine, frame)
    finally:
        cfg._settings = None


def test_turn_start_clears_qwen_engine_live_partial_no_stale_stitch(monkeypatch):
    # qwen accumulates a live streaming partial, so opening a turn must
    # drop it -- the new turn's first frame carries NONE of the pre-turn_start words.
    import voicemanager.config as cfg
    cfg._settings = None
    monkeypatch.setattr(
        "voicemanager.transport.websocket.QwenSTTPipeline",
        _fake_qwen_factory(["switch to red alert", "affirmative captain"]),
    )
    try:
        app = create_app()
        app.state.stt_engine_type = "qwen"
        app.state.session_registry = SessionRegistry()
        client = TestClient(app)

        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"

            # A live partial accumulates in turn 0.
            ws.send_bytes(_ONE_CHUNK)
            interim0 = ws.receive_json()
            assert interim0["text"] == "switch to red alert"
            assert interim0["turn_id"] == 0

            # Open turn 1: qwen's uncommitted partial is dropped.
            ws.send_json({"type": "turn_start"})
            assert ws.receive_json() == {"type": "turn_started", "turn_id": 1}

            # New audio: the partial carries ONLY post-turn_start words.
            ws.send_bytes(_ONE_CHUNK)
            interim1 = ws.receive_json()
            assert interim1["type"] == "transcription"
            assert interim1["turn_id"] == 1
            assert interim1["text"] == "affirmative captain"   # NOT "switch to red alert affirmative captain"
            assert "red alert" not in interim1["text"]         # no stale stitch into the new turn
    finally:
        cfg._settings = None


def test_whisper_engine_has_no_live_partial_to_clear_on_turn_start(monkeypatch):
    # Every-engine complement: whisper emits only VAD-gated finals and holds no interim partial
    # text, so opening a turn is a counter-only bump (clear_partial stays None) -- it must
    # neither raise nor disturb the monotonic segment_index, and post-turn_start finals are
    # stamped with the new turn.
    import voicemanager.config as cfg
    cfg._settings = None
    monkeypatch.setattr(
        "voicemanager.transport.websocket.STTPipeline",
        _fake_whisper_factory([["all stop"]]),
    )
    try:
        app = create_app()
        app.state.stt_engine_type = "whisper"
        app.state.session_registry = SessionRegistry()
        client = TestClient(app)

        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"

            ws.send_json({"type": "turn_start"})
            assert ws.receive_json() == {"type": "turn_started", "turn_id": 1}

            ws.send_bytes(_ONE_CHUNK)
            final = ws.receive_json()
            assert final["type"] == "transcription"
            assert final["is_final"] is True
            assert final["text"] == "all stop"
            assert final["turn_id"] == 1
    finally:
        cfg._settings = None
