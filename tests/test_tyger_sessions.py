"""Each session's turn rules keep their own state.

One sidecar serves every opted-in session, each with its own rule engine. Two
engines fed different score streams in interleaved hops must emit exactly what
each emits when fed alone: no state is shared between instances or kept at module
level. Model-free: the scores are a fake, seeded stream.
"""
import random

from voicemanager.pipeline.tyger import wire
from voicemanager.pipeline.tyger.rules import FRAME_SAMPLES, TurnRules, TurnRuleSettings

HOP_FRAMES = wire.HOP_SAMPLES // FRAME_SAMPLES


def fake_stream(seed: int, hops: int) -> list[tuple[str, list[tuple[float, float, float]]]]:
    """Hops of (mode, [(p_end, yield, energy_db)] per frame), with turns, replies,
    silences and confident heads mixed in so that both event kinds occur."""
    rng = random.Random(seed)
    out = []
    mode = wire.MODE_LISTENING
    for _ in range(hops):
        if rng.random() < 0.15:
            mode = wire.MODE_SPEAKING if mode == wire.MODE_LISTENING else wire.MODE_LISTENING
        frames = []
        for _ in range(HOP_FRAMES):
            p_end = rng.choice([0.05, 0.3, 0.92, 0.97])
            yld = rng.choice([0.1, 0.4, 0.65, 0.9])
            db = rng.choice([-70.0, -60.0, -40.0, -25.0])
            frames.append((p_end, yld, db))
        out.append((mode, frames))
    return out


def drive_hop(engine: TurnRules, hop_index: int, hop) -> list[tuple[int, object, bool]]:
    mode, frames = hop
    log = [(hop_index, "begin", engine.begin_hop(mode))]
    for j, (p_end, yld, db) in enumerate(frames):
        mic_sample = hop_index * wire.HOP_SAMPLES + (j + 1) * FRAME_SAMPLES
        result = engine.on_frame(mode, p_end, yld, db, mic_sample)
        log.extend((hop_index, event, result.reset) for event in result.events)
        if result.reset:
            log.append((hop_index, "reset", True))
    return log


def alone(stream, settings=None):
    engine = TurnRules(settings)
    log = []
    for i, hop in enumerate(stream):
        log += drive_hop(engine, i, hop)
    return log


def test_streams_produce_both_event_kinds():
    log = alone(fake_stream(1, 400))
    kinds = {type(entry[1]) for entry in log}
    assert wire.EndpointEvent in kinds and wire.BargeEvent in kinds


def test_two_sessions_interleaved_emit_what_each_emits_alone():
    a_stream, b_stream = fake_stream(1, 400), fake_stream(2, 400)
    expected_a, expected_b = alone(a_stream), alone(b_stream)
    assert expected_a != expected_b

    a, b = TurnRules(), TurnRules()
    got_a, got_b = [], []
    for i in range(400):
        # Alternate who goes first, as a sidecar serving both would.
        order = [(a, a_stream, got_a), (b, b_stream, got_b)]
        if i % 3 == 1:
            order.reverse()
        for engine, stream, got in order:
            got += drive_hop(engine, i, stream[i])
    assert got_a == expected_a
    assert got_b == expected_b


def test_frame_by_frame_interleaving_and_different_settings_stay_apart():
    a_stream, b_stream = fake_stream(3, 300), fake_stream(4, 300)
    tight = TurnRuleSettings(barge_threshold=0.85, warmup_frames=2, silence_fallback_ms=640.0)
    expected_a, expected_b = alone(a_stream), alone(b_stream, tight)

    a, b = TurnRules(), TurnRules(tight)
    got_a, got_b = [], []
    for i in range(300):
        for engine, stream, got in ((a, a_stream, got_a), (b, b_stream, got_b)):
            mode, _ = stream[i]
            got.append((i, "begin", engine.begin_hop(mode)))
        for j in range(HOP_FRAMES):
            for engine, stream, got in ((b, b_stream, got_b), (a, a_stream, got_a)):
                mode, frames = stream[i]
                p_end, yld, db = frames[j]
                mic_sample = i * wire.HOP_SAMPLES + (j + 1) * FRAME_SAMPLES
                result = engine.on_frame(mode, p_end, yld, db, mic_sample)
                got.extend((i, event, result.reset) for event in result.events)
                if result.reset:
                    got.append((i, "reset", True))
    assert got_a == expected_a
    assert got_b == expected_b


def test_default_settings_are_not_shared_mutable_state():
    a, b = TurnRules(), TurnRules()
    assert a.settings == b.settings
    # Frozen settings: one engine cannot change another's thresholds.
    try:
        a.settings.barge_threshold = 0.1
    except Exception:
        pass
    assert b.settings.barge_threshold == 0.6
