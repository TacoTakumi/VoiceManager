"""Protocol + config tests for the speculative endpoint-hold feature.

The first tests cover the additive `endpoint` field on TranscriptionMessage and
the VM_STT_ENDPOINT_HOLD_MS setting. Behavior tests for the hold state
machine (endpoint→confirm, reopen/stitch, teardown, fallback) land in later
tasks and are added to this same module.
"""
import json

import authkit

from voicemanager.config import Settings
from voicemanager.protocol import TranscriptionMessage


# --- additive `endpoint` field on TranscriptionMessage -------------

def test_transcription_message_endpoint_defaults_false():
    msg = TranscriptionMessage(text="hello")
    assert msg.endpoint is False
    payload = json.loads(msg.model_dump_json())
    assert payload["endpoint"] is False


def test_transcription_message_endpoint_can_be_true():
    msg = TranscriptionMessage(text="hello", endpoint=True)
    assert msg.endpoint is True
    payload = json.loads(msg.model_dump_json())
    assert payload["endpoint"] is True


def test_transcription_message_other_fields_unchanged():
    # Existing whisper/qwen emit sites construct TranscriptionMessage without
    # `endpoint`; the field is additive so those call sites need no changes.
    msg = TranscriptionMessage(text="hi", is_final=True, segment_index=2)
    payload = json.loads(msg.model_dump_json())
    assert payload["text"] == "hi"
    assert payload["is_final"] is True
    assert payload["segment_index"] == 2
    assert payload["endpoint"] is False


# --- VM_STT_ENDPOINT_HOLD_MS setting -------------------------------

def test_endpoint_hold_ms_default_is_1000():
    s = Settings()
    assert s.stt_endpoint_hold_ms == 1000


def test_endpoint_hold_ms_override_from_env(monkeypatch):
    monkeypatch.setenv("VM_STT_ENDPOINT_HOLD_MS", "0")
    s = Settings()
    assert s.stt_endpoint_hold_ms == 0


# --- endpoint-hold confirm path -------------
# Reuse the NeMo-free scripted fake-pipeline pattern from the pipeline tests: a
# _ScriptedNemotron yields (partial, final_or_None) per chunk; unscripted chunks
# (silence) fall through to ("", None). No test imports NeMo or loads a model.
import asyncio
import time

from test_nemotron_pipeline import _FakeShared, _ScriptedNemotron, _ONE_CHUNK

from voicemanager.transport.endpoint_hold import EndpointHoldMachine
from voicemanager.transport.websocket import _drive_nemotron_stt

_SILENCE_SAMPLE = b"\x00\x00"  # one int16 sample of silence


def _silence(n_samples: int) -> bytes:
    """n_samples of 16-bit PCM silence (drives the audio clock without EoU/interim)."""
    return _SILENCE_SAMPLE * n_samples


async def _collect(pipeline, machine, frames):
    """Drive inbound PCM frames through the nemotron+hold path; return sent messages."""
    sent = []

    async def send(msg):
        sent.append(msg)

    for data in frames:
        await _drive_nemotron_stt(pipeline, machine, data, send)
    assert_immutable_finals(sent)  # every driver-based scenario checks immutable finals
    return sent


def test_eou_emits_provisional_endpoint_then_confirmed_final():
    # Utterance: interim "this", then interim+EoU "this facility" (segment 0),
    # then >1s of silence to elapse the hold on the audio clock.
    p = _ScriptedNemotron(
        [("this", None), ("this facility", "this facility")],
        shared_pipeline=_FakeShared(), stream_id=1, att_context_size=[70, 13],
    )
    machine = EndpointHoldMachine(hold_ms=1000)  # 16000-sample window @ 16kHz
    sent = asyncio.run(_collect(p, machine, [_ONE_CHUNK, _ONE_CHUNK, _silence(16000)]))

    endpoints = [m for m in sent if m.endpoint]
    finals = [m for m in sent if m.is_final]
    assert len(endpoints) == 1
    assert endpoints[0].is_final is False and endpoints[0].endpoint is True
    assert endpoints[0].text == "this facility" and endpoints[0].segment_index == 0
    assert len(finals) == 1
    assert finals[0].is_final is True and finals[0].endpoint is False
    # confirmed final matches the provisional text and segment_index.
    assert finals[0].text == "this facility" and finals[0].segment_index == 0
    # provisional endpoint frame precedes the immutable confirmed final.
    assert sent.index(endpoints[0]) < sent.index(finals[0])


def test_confirm_runs_on_the_audio_clock_not_wallclock():
    # expiry is driven by fed-audio duration, so a faster-than-realtime
    # silence feed confirms in far less wall-clock time than hold_ms.
    p = _ScriptedNemotron(
        [("hello", "hello")],
        shared_pipeline=_FakeShared(), stream_id=2, att_context_size=[70, 13],
    )
    machine = EndpointHoldMachine(hold_ms=1000)  # 16000-sample window

    async def run():
        sent = []

        async def send(msg):
            sent.append(msg)

        await _drive_nemotron_stt(p, machine, _ONE_CHUNK, send)  # EoU -> endpoint + hold
        assert any(m.endpoint for m in sent)
        assert not any(m.is_final for m in sent)     # not yet confirmed
        await _drive_nemotron_stt(p, machine, _silence(15000), send)  # just under window
        assert not any(m.is_final for m in sent)     # still holding
        await _drive_nemotron_stt(p, machine, _silence(2000), send)   # crosses window
        return sent

    start = time.monotonic()
    sent = asyncio.run(run())
    elapsed = time.monotonic() - start

    assert any(m.is_final and m.text == "hello" for m in sent)
    # No wall-clock waiting: the exchange is far quicker than the 1000ms window.
    assert elapsed < 1.0
    assert_immutable_finals(sent)  # the inline-run scenario checks immutable finals too


def test_hold_machine_confirms_after_window_samples():
    # Direct state-machine unit: the audio clock accumulates fed samples and
    # confirms once it reaches hold_samples (1000ms @ 16kHz == 16000 samples).
    m = EndpointHoldMachine(hold_ms=1000)
    assert m.is_holding is False
    m.begin_hold("done", "en", 3)
    assert m.is_holding is True
    assert m.advance(8000) is None                 # halfway
    assert m.advance(7999) is None                 # 15999 < 16000
    assert m.advance(1) == ("done", "en", 3)       # window reached -> confirm
    assert m.is_holding is False
    assert m.advance(100) is None                  # idle after confirm


def test_hold_machine_none_uses_legacy_immediate_final():
    # Infra guard: hold_machine=None drives the legacy immediate-final path with no
    # endpoint frame. The disable path (VM_STT_ENDPOINT_HOLD_MS=0 wiring + byte-for-
    # byte parity) is pinned by the disable-path tests below.
    p = _ScriptedNemotron(
        [("go", "go")],
        shared_pipeline=_FakeShared(), stream_id=4, att_context_size=[70, 13],
    )
    sent = asyncio.run(_collect(p, None, [_ONE_CHUNK]))
    assert any(m.is_final and m.text == "go" and m.segment_index == 0 for m in sent)
    assert not any(m.endpoint for m in sent)


# --- disable path: VM_STT_ENDPOINT_HOLD_MS=0 bypasses the hold --

from voicemanager.transport.websocket import _make_hold_machine


def test_disable_setting_zero_builds_no_hold_machine(monkeypatch):
    # The handler gate: 0 -> no hold machine -> the nemotron branch runs the legacy
    # immediate-final path. >0 builds a real machine.
    monkeypatch.setenv("VM_STT_ENDPOINT_HOLD_MS", "0")
    assert _make_hold_machine(Settings()) is None


def test_enabled_setting_builds_hold_machine():
    assert isinstance(_make_hold_machine(Settings()), EndpointHoldMachine)  # default 1000


def test_disable_path_emits_final_without_endpoint_frame():
    # With speculation disabled, a scripted utterance + trailing silence yields an
    # is_final=true frame and NO endpoint=true frame is ever produced.
    p = _ScriptedNemotron(
        [("this", None), ("this facility", "this facility")],
        shared_pipeline=_FakeShared(), stream_id=10, att_context_size=[70, 13],
    )
    sent = asyncio.run(_collect(p, None, [_ONE_CHUNK, _ONE_CHUNK, _silence(16000)]))
    assert not any(m.endpoint for m in sent)
    finals = [m for m in sent if m.is_final]
    assert len(finals) == 1
    assert finals[0].text == "this facility" and finals[0].segment_index == 0


def test_disable_path_matches_prefeature_emit_ordering():
    # Byte-for-byte pre-feature emit behavior: interim then immediate final, same
    # text/segment_index/ordering, with endpoint additively false throughout.
    p = _ScriptedNemotron(
        [("this", None), ("this facility", "this facility")],
        shared_pipeline=_FakeShared(), stream_id=11, att_context_size=[70, 13],
    )
    sent = asyncio.run(_collect(p, None, [_ONE_CHUNK, _ONE_CHUNK]))
    assert [(m.text, m.is_final, m.endpoint, m.segment_index) for m in sent] == [
        ("this", False, False, 0),
        ("this facility", True, False, 0),
    ]


def test_disable_path_increments_segment_index_per_final():
    # Two utterances under the disable path keep the pre-feature monotonic segments.
    p = _ScriptedNemotron(
        [("a", "a"), ("b", "b")],
        shared_pipeline=_FakeShared(), stream_id=12, att_context_size=[70, 13],
    )
    sent = asyncio.run(_collect(p, None, [_ONE_CHUNK, _ONE_CHUNK]))
    finals = [(m.text, m.segment_index) for m in sent if m.is_final]
    assert finals == [("a", 0), ("b", 1)]


# --- reopen + stitch, and the confirmed-final commit boundary --
# Scripting note: _ScriptedNemotron pops one script step per *full chunk* (256 samples),
# regardless of audio content — so a sub-chunk silence frame (<256 samples) advances the
# audio clock without consuming a script step. That lets an in-window pause sit between a
# held utterance and the resumed one while leaving the resumed one's steps in the script.


def test_reopen_stitches_held_text_under_original_segment_index():
    # A endpoints (seg 0, held), a sub-chunk in-window pause passes, then B decodes
    # WITHIN the hold window. B reopens the turn: A's pending final is cancelled, "this
    # facility" is space-joined onto B's partials, the resumed NeMo segment is remapped onto
    # seg 0, and the eventual confirmed final carries the whole stitched turn under seg 0.
    p = _ScriptedNemotron(
        [("this", None), ("this facility", "this facility"),          # A: interim, EoU (seg 0)
         ("is", None), ("is not", None),                              # B: interims (reopen)
         ("is not authorized", "is not authorized")],                 # B: EoU (stitched)
        shared_pipeline=_FakeShared(), stream_id=20, att_context_size=[70, 13],
    )
    machine = EndpointHoldMachine(hold_ms=1000)  # 16000-sample window
    frames = [
        _ONE_CHUNK,          # "this" interim, seg 0
        _ONE_CHUNK,          # "this facility" EoU -> provisional endpoint seg 0, begin hold
        _silence(200),       # in-window pause: sub-chunk, advances clock, consumes no step
        _ONE_CHUNK,          # "is" -> REOPEN: cancel hold, stitch prefix "this facility"
        _ONE_CHUNK,          # "is not" interim (stitched)
        _ONE_CHUNK,          # "is not authorized" EoU -> provisional stitched final, seg 0
        _silence(16000),     # long silence -> confirm the stitched immutable final
    ]
    sent = asyncio.run(_collect(p, machine, frames))

    finals = [m for m in sent if m.is_final]
    assert len(finals) == 1
    # Exactly one is_final=true frame: A and B space-joined under A's segment_index.
    assert finals[0].text == "this facility is not authorized"
    assert finals[0].segment_index == 0
    # Zero is_final=true frames containing only A.
    assert all(m.text != "this facility" for m in finals)
    # Interim frames after reopen carry the stitched text under the held segment_index.
    stitched = [m for m in sent
                if not m.is_final and not m.endpoint and m.text.startswith("this facility ")]
    assert stitched, "expected stitched interim frames after reopen"
    assert all(m.segment_index == 0 for m in stitched)


def test_commit_boundary_starts_new_turn_at_next_segment():
    # an out-of-window silence commits A (immutable final seg 0); speech that
    # resumes AFTER that confirmed final is a new turn under seg 1, never merged into A.
    p = _ScriptedNemotron(
        [("apple", "apple")],  # A: single-chunk EoU
        shared_pipeline=_FakeShared(), stream_id=21, att_context_size=[70, 13],
    )
    machine = EndpointHoldMachine(hold_ms=1000)

    async def run():
        sent = []

        async def send(msg):
            sent.append(msg)

        await _drive_nemotron_stt(p, machine, _ONE_CHUNK, send)       # A EoU -> provisional seg 0
        await _drive_nemotron_stt(p, machine, _silence(16000), send)  # cross window -> confirm "apple" seg 0
        assert any(m.is_final and m.text == "apple" for m in sent)
        p._script.append(("banana", "banana"))                       # B decodes as a NEW turn
        await _drive_nemotron_stt(p, machine, _ONE_CHUNK, send)       # B EoU -> provisional seg 1
        await _drive_nemotron_stt(p, machine, _silence(16000), send)  # confirm "banana" seg 1
        return sent

    sent = asyncio.run(run())
    assert_immutable_finals(sent)  # finals stay immutable across the cross-turn commit boundary
    finals = [(m.text, m.segment_index) for m in sent if m.is_final]
    assert finals == [("apple", 0), ("banana", 1)]  # B under A+1, A unchanged


# --- repeated EoU within a stitched turn re-enters the speculative cycle --


def test_repeated_eou_reemits_endpoint_for_growing_stitched_turn():
    # each time the reopened speech itself hits EoU, a FRESH endpoint=true frame
    # carries the full stitched turn so far under the original segment_index, and the hold
    # restarts — so a further resumption stitches again. A, reopen->B(EoU), reopen->C(EoU),
    # then long silence confirms the whole turn.
    p = _ScriptedNemotron(
        [("this", None), ("this facility", "this facility"),          # A: interim, EoU (seg 0)
         ("is", None), ("is not authorized", "is not authorized"),    # B: interim, reopen + EoU
         ("and", None), ("and dangerous", "and dangerous")],          # C: interim, reopen + EoU
        shared_pipeline=_FakeShared(), stream_id=22, att_context_size=[70, 13],
    )
    machine = EndpointHoldMachine(hold_ms=1000)  # 16000-sample window
    frames = [
        _ONE_CHUNK,          # "this" interim seg 0
        _ONE_CHUNK,          # EoU -> 1st endpoint "this facility" seg 0, hold
        _silence(100),       # in-window pause (sub-chunk: no script step consumed)
        _ONE_CHUNK,          # "is" -> reopen 1
        _ONE_CHUNK,          # EoU -> 2nd endpoint "this facility is not authorized" seg 0, re-hold
        _silence(100),       # in-window pause
        _ONE_CHUNK,          # "and" -> reopen 2
        _ONE_CHUNK,          # EoU -> 3rd endpoint (full stitched turn) seg 0, re-hold
        _silence(16000),     # long silence -> confirm the whole stitched turn
    ]
    sent = asyncio.run(_collect(p, machine, frames))

    # A fresh endpoint=true frame per EoU, each carrying the stitched turn so far, all seg 0.
    endpoints = [m for m in sent if m.endpoint]
    assert [m.text for m in endpoints] == [
        "this facility",
        "this facility is not authorized",           # 2nd endpoint contains A and B
        "this facility is not authorized and dangerous",
    ]
    assert all(m.is_final is False and m.segment_index == 0 for m in endpoints)
    # One immutable final, carrying every fragment, under the original segment_index.
    finals = [m for m in sent if m.is_final]
    assert len(finals) == 1
    assert finals[0].text == "this facility is not authorized and dangerous"
    assert finals[0].segment_index == 0 and finals[0].endpoint is False


# --- wall-clock fallback via the WS receive-timeout path ---------------
# A WS-level test: a client that endpoints then goes silent must still get its confirmed
# final within the bounded fallback interval (the hold window in wall-clock), not only at
# session close. The per-connection NeMo pipeline is monkeypatched to a scripted fake so
# the handler's nemotron branch runs without loading a model.
from fastapi.testclient import TestClient

from voicemanager.app import create_app
from voicemanager.session import SessionRegistry


def _make_scripted_nemotron_class(script):
    """A drop-in for NemotronSTTPipeline that ignores NeMo and replays `script`."""
    def _factory(**kwargs):
        return _ScriptedNemotron(list(script), **kwargs)
    return _factory


def test_wallclock_fallback_confirms_held_final_before_session_close(monkeypatch):
    # Tight windows so the fallback (hold_ms) fires well before session close (session
    # timeout): held-final at ~0.3s, close at ~2s — the gap is what the test asserts.
    monkeypatch.setenv("VM_STT_ENDPOINT_HOLD_MS", "300")
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "2")
    # Scripted per-connection pipeline: utterance A (EoU), then utterance B (EoU) to prove
    # the session survived the fallback. Both are single-chunk EoU steps.
    monkeypatch.setattr(
        "voicemanager.pipeline.nemotron_stt.NemotronSTTPipeline",
        _make_scripted_nemotron_class([("hi there", "hi there"), ("again", "again")]),
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
            ready = ws.receive_json()
            assert ready["type"] == "ready" and ready["stt_engine"] == "nemotron"

            # A: send one chunk -> acoustic EoU -> provisional endpoint frame, then go silent.
            ws.send_bytes(_ONE_CHUNK)
            prov = ws.receive_json()
            assert prov["type"] == "transcription"
            assert prov["endpoint"] is True and prov["is_final"] is False
            assert prov["text"] == "hi there" and prov["segment_index"] == 0

            # Client sends nothing further. The confirmed final must still arrive via
            # the receive-timeout fallback, within the bounded interval (not at session close).
            t0 = time.monotonic()
            final = ws.receive_json()
            elapsed = time.monotonic() - t0
            assert final["type"] == "transcription"
            assert final["is_final"] is True and final["endpoint"] is False
            assert final["text"] == "hi there" and final["segment_index"] == 0
            # Arrived within the fallback interval, comfortably before the 2s session close.
            assert elapsed < 1.5

            # Session stays open afterwards: a second utterance is still served (new segment).
            ws.send_bytes(_ONE_CHUNK)
            prov2 = ws.receive_json()
            assert prov2["type"] == "transcription"
            assert prov2["endpoint"] is True and prov2["text"] == "again"
            assert prov2["segment_index"] == 1
            # the full WS frame stream (JSON dicts) keeps finals immutable across both turns.
            assert_immutable_finals([prov, final, prov2])
    finally:
        cfg._settings = None


def test_midstitch_final_flushed_before_session_close(monkeypatch):
    # the mid-stitch state (reopened, is_holding=False, stitch prefix
    # set, no resumed EoU yet) has NO wall-clock fallback — the receive-timeout cap
    # gates on is_holding. So a client that reopens then goes fully silent rides the
    # session-inactivity timeout to close. The held stitched turn must still be flushed
    # as the confirmed final BEFORE the socket closes, not silently dropped.
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setenv("VM_STT_ENDPOINT_HOLD_MS", "1000")
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "1")
    # A: single-chunk EoU -> hold. B: interim only (reopen) -> mid-stitch, no EoU.
    monkeypatch.setattr(
        "voicemanager.pipeline.nemotron_stt.NemotronSTTPipeline",
        _make_scripted_nemotron_class([("a", "a"), ("b", None)]),
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

            # A -> provisional endpoint, hold begins.
            ws.send_bytes(_ONE_CHUNK)
            prov = ws.receive_json()
            assert prov["endpoint"] is True and prov["is_final"] is False
            assert prov["text"] == "a" and prov["segment_index"] == 0

            # B arrives well within the hold window -> reopen; stitched interim, no EoU.
            ws.send_bytes(_ONE_CHUNK)
            interim = ws.receive_json()
            assert interim["is_final"] is False and interim["endpoint"] is False
            assert interim["text"] == "a b" and interim["segment_index"] == 0

            # Client now goes silent. No fallback covers mid-stitch, so the session
            # rides to its inactivity timeout. Collect frames until the stitched final
            # arrives or the socket closes.
            final = None
            frames = [prov, interim]
            try:
                for _ in range(10):
                    f = ws.receive_json()
                    frames.append(f)
                    if f.get("is_final"):
                        final = f
                        break
            except WebSocketDisconnect:
                pass

            assert final is not None, f"stitched final dropped on session-timeout close; got {frames}"
            # one confirmed final carrying the full stitched turn under
            # the original segment_index, delivered before close.
            assert final["is_final"] is True and final["endpoint"] is False
            assert final["text"] == "a b" and final["segment_index"] == 0
            assert_immutable_finals(frames)
    finally:
        cfg._settings = None


# --- teardown flushes held text as a confirmed final -------------------
# The teardown flush is a shared code path (like _drive_nemotron_stt): _teardown_nemotron_stt
# runs pipeline.finalize() and emits the held (possibly stitched) text merged with any tail,
# before the socket closes. The WS finally-block and (defensively) other end paths call it.
from voicemanager.transport.websocket import _teardown_nemotron_stt


async def _teardown(pipeline, machine):
    sent = []

    async def send(msg):
        sent.append(msg)

    await _teardown_nemotron_stt(pipeline, machine, send)
    assert_immutable_finals(sent)  # the teardown flush keeps finals immutable too
    return sent


def test_teardown_flushes_held_final():
    # a hold pending at teardown is emitted as an is_final=true frame (not dropped).
    p = _ScriptedNemotron(
        [("hi", "hi")], shared_pipeline=_FakeShared(), stream_id=30, att_context_size=[70, 13],
    )
    machine = EndpointHoldMachine(hold_ms=1000)
    asyncio.run(_collect(p, machine, [_ONE_CHUNK]))          # EoU -> provisional, begin hold
    assert machine.is_holding
    sent = asyncio.run(_teardown(p, machine))
    finals = [m for m in sent if m.is_final]
    assert len(finals) == 1
    assert finals[0].text == "hi" and finals[0].segment_index == 0
    assert finals[0].endpoint is False
    assert machine.is_holding is False


def test_teardown_flushes_stitched_held_final():
    # the same holds when the hold began inside a stitched turn — the held text is
    # the full space-joined turn under the original segment_index.
    p = _ScriptedNemotron(
        [("a", None), ("a", "a"),          # A: interim, EoU (seg 0) -> hold
         ("b", None), ("b done", "b done")],  # B: reopen, EoU -> re-hold stitched "a b done"
        shared_pipeline=_FakeShared(), stream_id=31, att_context_size=[70, 13],
    )
    machine = EndpointHoldMachine(hold_ms=1000)
    asyncio.run(_collect(p, machine, [_ONE_CHUNK, _ONE_CHUNK, _silence(100), _ONE_CHUNK, _ONE_CHUNK]))
    assert machine.is_holding
    sent = asyncio.run(_teardown(p, machine))
    finals = [m for m in sent if m.is_final]
    assert len(finals) == 1
    assert finals[0].text == "a b done" and finals[0].segment_index == 0


def test_teardown_merges_finalize_tail_into_held_final():
    # Task title: merge any tail from pipeline.finalize() (residue after the EoU) into the
    # held turn, one final under the held segment_index (never two finals for one turn).
    p = _ScriptedNemotron(
        [("hi", "hi")], final_tail="there",
        shared_pipeline=_FakeShared(), stream_id=32, att_context_size=[70, 13],
    )
    machine = EndpointHoldMachine(hold_ms=1000)
    asyncio.run(_collect(p, machine, [_ONE_CHUNK]))
    sent = asyncio.run(_teardown(p, machine))
    finals = [m for m in sent if m.is_final]
    assert len(finals) == 1
    assert finals[0].text == "hi there" and finals[0].segment_index == 0


def test_teardown_midstitch_prefix_not_dropped():
    # held text is never dropped where a send is possible — if the stream ends after
    # a reopen but before the resumed EoU, the stitch prefix + trailing partial still flush
    # as a final under the original segment_index.
    p = _ScriptedNemotron(
        [("a", None), ("a", "a"),   # A: interim, EoU (seg 0) -> hold
         ("b", None)],              # B: interim only (reopen) — no EoU before teardown
        shared_pipeline=_FakeShared(), stream_id=33, att_context_size=[70, 13],
    )
    machine = EndpointHoldMachine(hold_ms=1000)
    asyncio.run(_collect(p, machine, [_ONE_CHUNK, _ONE_CHUNK, _silence(100), _ONE_CHUNK]))
    assert machine.is_holding is False and machine.stitch_prefix == "a"  # mid-stitch
    sent = asyncio.run(_teardown(p, machine))
    finals = [m for m in sent if m.is_final]
    assert len(finals) == 1
    assert finals[0].text == "a b" and finals[0].segment_index == 0  # prefix not dropped


def test_teardown_idempotent_no_double_final():
    # at most one is_final per segment_index — a second teardown emits nothing.
    p = _ScriptedNemotron(
        [("hi", "hi")], shared_pipeline=_FakeShared(), stream_id=34, att_context_size=[70, 13],
    )
    machine = EndpointHoldMachine(hold_ms=1000)
    asyncio.run(_collect(p, machine, [_ONE_CHUNK]))
    first = asyncio.run(_teardown(p, machine))
    second = asyncio.run(_teardown(p, machine))
    assert len([m for m in first if m.is_final]) == 1
    assert [m for m in second if m.is_final] == []


def test_teardown_midstitch_idempotent_no_double_final():
    # a second teardown in the MID-STITCH state must also emit nothing.
    # The flush clears the stitch prefix and the pipeline partial, so it is inert by
    # state — not merely because the socket closed first (the WS timeout path flushes
    # pre-close, then the finally-block calls teardown again).
    p = _ScriptedNemotron(
        [("a", None), ("a", "a"), ("b", None)],   # A EoU -> hold; B interim -> reopen (mid-stitch)
        shared_pipeline=_FakeShared(), stream_id=36, att_context_size=[70, 13],
    )
    machine = EndpointHoldMachine(hold_ms=1000)
    asyncio.run(_collect(p, machine, [_ONE_CHUNK, _ONE_CHUNK, _silence(100), _ONE_CHUNK]))
    assert machine.is_holding is False and machine.stitch_prefix == "a"  # mid-stitch
    first = asyncio.run(_teardown(p, machine))
    second = asyncio.run(_teardown(p, machine))
    assert len([m for m in first if m.is_final]) == 1
    assert first[0].text == "a b" and first[0].segment_index == 0
    assert [m for m in second if m.is_final] == []   # inert second flush
    assert machine.stitch_prefix == ""


def test_teardown_disabled_path_emits_tail_only():
    # with the hold disabled (hold_machine None) teardown is the legacy finalize —
    # the finalize tail under the pipeline's segment_index, no held/stitch machinery.
    p = _ScriptedNemotron(
        [("go", "go")], final_tail="tail",
        shared_pipeline=_FakeShared(), stream_id=35, att_context_size=[70, 13],
    )
    asyncio.run(_collect(p, None, [_ONE_CHUNK]))   # disable path: immediate final "go" (seg 0)
    sent = asyncio.run(_teardown(p, None))
    finals = [m for m in sent if m.is_final]
    assert len(finals) == 1
    assert finals[0].text == "tail" and finals[0].segment_index == 1  # next pipeline segment


# --- cross-cutting invariants + structural regression --
# One shared helper, applied to every endpoint-hold scenario, pins the immutable-finals
# invariant: across a scenario's emitted frames each segment_index carries at most
# one is_final=true frame, and once a segment is finalized no later frame may revise it. It is
# wired into the shared drivers (_collect / _teardown) so it runs over every driver-based
# scenario, and is called directly by the inline-run and WS scenarios above. The structural
# tests below check at the source level: no retraction type, endpoint=true set only in the
# nemotron WS driver, hold/stitch state transport-only with no re-decode path.
import pathlib

import pytest

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "voicemanager"


def _frame_fields(frame) -> tuple[bool, bool, int]:
    """(is_final, endpoint, segment_index) from a TranscriptionMessage or its JSON dict."""
    if isinstance(frame, dict):
        return (bool(frame["is_final"]),
                bool(frame.get("endpoint", False)),
                int(frame["segment_index"]))
    return bool(frame.is_final), bool(frame.endpoint), int(frame.segment_index)


def assert_immutable_finals(frames) -> None:
    """The immutable-finals invariant over an ordered list of emitted transcription frames.

    Accepts TranscriptionMessage objects or their JSON dicts. Raises AssertionError if any
    segment_index gets more than one is_final=true frame, or if any frame carries a
    segment_index that an earlier frame already finalized (a committed segment being revised).
    An endpoint/interim frame *preceding* its own confirmed final is fine — only frames after
    the commit are revisions.
    """
    finalized: set[int] = set()
    for i, frame in enumerate(frames):
        is_final, _endpoint, seg = _frame_fields(frame)
        assert seg not in finalized, (
            f"frame {i} (seg {seg}, is_final={is_final}) revises segment {seg}, "
            f"already finalized")
        if is_final:
            finalized.add(seg)


def test_invariant_helper_flags_double_final():
    # Two is_final=true frames for one segment_index must be rejected.
    frames = [
        TranscriptionMessage(text="a", is_final=True, segment_index=0),
        TranscriptionMessage(text="a again", is_final=True, segment_index=0),
    ]
    with pytest.raises(AssertionError):
        assert_immutable_finals(frames)


def test_invariant_helper_flags_post_commit_revision():
    # A frame that touches a segment_index after it was finalized is a revision.
    frames = [
        TranscriptionMessage(text="a", is_final=True, segment_index=0),
        TranscriptionMessage(text="a revised", is_final=False, segment_index=0),
    ]
    with pytest.raises(AssertionError):
        assert_immutable_finals(frames)


def test_invariant_helper_accepts_endpoint_before_its_final():
    # The real emit shape: a provisional endpoint precedes the confirmed final for the same
    # segment — not a revision, must pass.
    frames = [
        TranscriptionMessage(text="hi", is_final=False, endpoint=True, segment_index=0),
        TranscriptionMessage(text="hi", is_final=True, segment_index=0),
    ]
    assert_immutable_finals(frames)  # no raise


def test_invariant_helper_accepts_ws_json_dicts():
    # WS scenarios collect JSON dicts (receive_json); the helper handles them too.
    frames = [
        {"is_final": False, "endpoint": True, "segment_index": 0},
        {"is_final": True, "endpoint": False, "segment_index": 0},
        {"is_final": True, "endpoint": False, "segment_index": 1},
    ]
    assert_immutable_finals(frames)  # no raise


# --- Structural: the protocol defines no retraction/revision frame ------------

def test_no_retraction_or_revision_message_type():
    from voicemanager.protocol import MessageType, TranscriptionMessage

    banned = ("retract", "revise", "revision", "amend", "correct", "supersede", "undo", "delete")
    for m in MessageType:
        tag = f"{m.name} {m.value}".lower()
        assert not any(b in tag for b in banned), f"MessageType.{m.name} looks like a retraction"
    # TranscriptionMessage gained only additive fields over the pre-feature model -- `endpoint`
    # and `turn_id` -- and no field that points at/replaces a prior frame:
    # pin the field set so a revision hook or a non-additive change to the transcription frame
    # fails this test. Both additions are additive and carry defaults that reproduce today's
    # behaviour (endpoint=false, turn_id=0), so they extend the pin rather than break it.
    fields = set(TranscriptionMessage.model_fields)
    for f in fields:
        assert not any(b in f.lower() for b in banned), f"field {f!r} looks like a revision hook"
    assert fields == {
        "type", "text", "is_final", "endpoint", "turn_id", "segment_index",
        "language", "inference_duration", "confidence", "start", "end",
    }
    ep = TranscriptionMessage.model_fields["endpoint"]
    assert ep.annotation is bool and ep.default is False  # additive, defaults false


# --- Structural: endpoint=true is set only in the nemotron WS driver ----------

def _py_files() -> list[pathlib.Path]:
    return sorted(_SRC.rglob("*.py"))


def test_endpoint_true_emitted_only_in_nemotron_ws_driver():
    hits = []
    for path in _py_files():
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if "endpoint=True" in line.replace(" ", ""):
                hits.append((path, lineno))
    assert len(hits) == 1, f"endpoint=True must be set exactly once (nemotron driver); got {hits}"
    ws = _SRC / "transport" / "websocket.py"
    path, lineno = hits[0]
    assert path == ws, f"endpoint=True set outside the WS transport ({path})"
    lines = ws.read_text().splitlines()
    drive = next(i for i, l in enumerate(lines, 1) if l.startswith("async def _drive_nemotron_stt"))
    teardown = next(i for i, l in enumerate(lines, 1)
                    if l.startswith("async def _teardown_nemotron_stt"))
    assert drive < lineno < teardown, "endpoint=True is outside the _drive_nemotron_stt driver"


# --- Structural: hold/stitch state is transport-only, no audio re-decode ------

def test_hold_stitch_state_is_transport_only_with_no_redecode():
    # The NeMo pipeline is not modified to support merging: no pipeline module references the
    # hold machine or its module.
    for path in (_SRC / "pipeline").rglob("*.py"):
        text = path.read_text()
        assert "EndpointHoldMachine" not in text, f"{path.name} couples the pipeline to the hold machine"
        assert "endpoint_hold" not in text, f"{path.name} imports transport hold state"
    hold_src = (_SRC / "transport" / "endpoint_hold.py").read_text()
    # The machine imports nothing and touches no audio/model: it holds text + a sample counter,
    # so no code path can re-feed pre-EoU audio back for re-decoding.
    imports = [ln for ln in hold_src.splitlines() if ln.startswith(("import ", "from "))]
    assert imports == [], f"endpoint_hold.py should import nothing; found {imports}"
    for banned in (".feed(", ".decode(", ".transcribe(", "np.", "torch", "librosa", "soundfile"):
        assert banned not in hold_src, f"endpoint_hold.py references {banned!r} — must hold no audio"
    # And the state IS wired into the per-connection WS transport handler.
    assert "EndpointHoldMachine" in (_SRC / "transport" / "websocket.py").read_text()
