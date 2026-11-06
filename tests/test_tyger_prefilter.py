"""The sidecar's experimental pre-filter wrapper, against a fake filter.

Model-free, GPU-free and server-free. A recording fake scorer stands behind the
wrapper and keeps every hop it is given; a fake filter stands in for VoxLock. The
wrapper must pass every hop unchanged until it has 3 s of voiced listening-mode
audio, enroll once from exactly those samples, and from then on feed the filter
every hop in order, give speaking hops the delayed filter output plus floor noise
and a silent agent channel, and leave listening hops as they came.
"""
import logging
import math

import numpy as np
import pytest

from voicemanager.pipeline.tyger import prefilter
from voicemanager.pipeline.tyger.scorer import FrameScore
from voicemanager.pipeline.tyger.wire import HOP_SAMPLES, MODE_LISTENING, MODE_SPEAKING

FRAME = 1280
ENROLL = 48000
GATE_DB = -50.0
FLOOR_DB = -52.0


class RecordingScorer:
    """Keeps (user, agent, mode) of every hop and gives four flat frames."""

    def __init__(self):
        self.hops = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def __init__(self, yield_prob=0.9):
        self.hops = []
        self.resets = 0
        self.yield_prob = yield_prob

    def score_hop(self, user, agent, mode):
        self.hops.append((bytes(user), bytes(agent), mode))
        return iter([FrameScore(0.1 * i, self.yield_prob, -20.0) for i in range(4)])


class FakeFilter:
    """Outputs what a given function makes of its input, ``delay`` samples late."""

    def __init__(self, delay, fn):
        self.delay = delay
        self.fn = fn
        self.pushed = []
        self._held = np.zeros(0, np.float32)

    def push(self, samples):
        # The joined output lines up with the input sample for sample, but the last
        # `delay` samples are held back: the first push returns that many fewer than
        # it took, as VoxLock does.
        samples = np.asarray(samples, np.float32)
        self.pushed.append(samples.copy())
        out = np.concatenate([self._held, self.fn(samples)])
        cut = len(out) - self.delay
        self._held = out[cut:]
        return out[:cut]


class FakeFactory:
    def __init__(self, delay=1000, fn=lambda x: x * 0.5):
        self.delay = delay
        self.fn = fn
        self.enrolled = []
        self.filters = []

    def enroll(self, sample):
        self.enrolled.append(np.asarray(sample, np.float32).copy())
        f = FakeFilter(self.delay, self.fn)
        self.filters.append(f)
        return f


def _pcm(x):
    return np.clip(np.round(np.asarray(x) * 32768.0), -32768, 32767).astype("<i2").tobytes()


def _floats(pcm):
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def _voiced_hop(seed, level=0.1):
    rng = np.random.default_rng(seed)
    return _pcm(rng.uniform(-level, level, HOP_SAMPLES))


def _quiet_hop():
    return _pcm(np.zeros(HOP_SAMPLES))


def _agent_hop(seed=99):
    return _voiced_hop(seed, 0.3)


def _wrap(factory=None, floor_db=FLOOR_DB):
    inner = RecordingScorer()
    factory = factory or FakeFactory()
    wrapped = prefilter.PrefilterSessionScorer(inner, factory, floor_db=floor_db,
                                               voice_gate_db=GATE_DB, key="s1")
    return wrapped, inner, factory


def _run(wrapped, user, agent, mode):
    return list(wrapped.score_hop(user, agent, mode))


def _enroll(wrapped, start_seed=0):
    """Feed voiced listening hops until the wrapper enrolls; returns the hops fed."""
    fed = []
    for i in range(20):
        hop = _voiced_hop(start_seed + i)
        fed.append(hop)
        _run(wrapped, hop, _quiet_hop(), MODE_LISTENING)
        if wrapped.enrolled:
            return fed
    raise AssertionError("never enrolled")


# --- before enrollment ------------------------------------------------------------------

def test_hops_before_enrollment_reach_the_scorer_unchanged():
    wrapped, inner, factory = _wrap()
    user, agent = _voiced_hop(1), _agent_hop()
    _run(wrapped, user, agent, MODE_SPEAKING)
    _run(wrapped, _voiced_hop(2), _quiet_hop(), MODE_LISTENING)
    assert inner.hops[0] == (user, agent, MODE_SPEAKING)
    assert inner.hops[1] == (_voiced_hop(2), _quiet_hop(), MODE_LISTENING)
    assert factory.enrolled == []


def test_the_scorer_frames_pass_through():
    wrapped, _inner, _f = _wrap()
    assert _run(wrapped, _voiced_hop(1), _quiet_hop(), MODE_LISTENING) == [
        FrameScore(0.1 * i, 0.9, -20.0) for i in range(4)]


def test_no_barge_before_enrollment_and_a_barge_after_it():
    """Before the voice sample exists the scorer hears the raw mic, echo and all, so
    a speaking hop's yield is held at 0 and the turn rules see no barge. After
    enrollment the same scores reach the rules unchanged and the barge fires."""
    from voicemanager.pipeline.tyger import rules, server

    def barges(session):
        hop = server.wire.unpack_hop(server.wire.pack_hop(
            MODE_SPEAKING, (session.hops + 1) * HOP_SAMPLES, _voiced_hop(500), _agent_hop()))
        return [e for e in server.process_hop(session, hop)
                if isinstance(e, server.wire.BargeEvent)]

    wrapped, inner, _f = _wrap()
    session = server.Session("s1", rules.TurnRules(rules.TurnRuleSettings()), wrapped)
    assert inner.yield_prob > rules.TurnRuleSettings().barge_threshold
    before = [barges(session) for _ in range(4)]
    assert before == [[]] * 4
    assert [f.yield_prob for f in _run(wrapped, _voiced_hop(1), _agent_hop(), MODE_SPEAKING)] \
        == [0.0] * 4

    _enroll(wrapped)
    session = server.Session("s1", rules.TurnRules(rules.TurnRuleSettings()), wrapped)
    after = [e for _ in range(4) for e in barges(session)]
    assert len(after) == 1


def test_enrollment_takes_exactly_3_s_of_voiced_listening_audio_in_order():
    wrapped, _inner, factory = _wrap()
    fed = _enroll(wrapped)
    assert len(factory.enrolled) == 1
    sample = factory.enrolled[0]
    assert len(sample) == ENROLL
    expected = np.concatenate([_floats(h) for h in fed])[:ENROLL]
    assert np.array_equal(sample, expected)


def test_speaking_hops_and_frames_under_the_gate_are_not_collected():
    wrapped, _inner, factory = _wrap()
    # Speaking audio and silence come first; neither may enter the sample.
    for i in range(10):
        _run(wrapped, _voiced_hop(100 + i), _agent_hop(), MODE_SPEAKING)
        _run(wrapped, _quiet_hop(), _quiet_hop(), MODE_LISTENING)
    # A hop with only its first frame voiced adds one frame.
    half = np.zeros(HOP_SAMPLES, np.float32)
    half[:FRAME] = np.random.default_rng(7).uniform(-0.1, 0.1, FRAME)
    _run(wrapped, _pcm(half), _quiet_hop(), MODE_LISTENING)
    assert factory.enrolled == []
    fed = _enroll(wrapped, start_seed=200)
    expected = np.concatenate([_floats(_pcm(half))[:FRAME]] + [_floats(h) for h in fed])[:ENROLL]
    assert np.array_equal(factory.enrolled[0], expected)


def test_enrollment_happens_once_and_is_logged_with_the_session(caplog):
    wrapped, _inner, factory = _wrap()
    with caplog.at_level(logging.INFO):
        _enroll(wrapped)
        for i in range(20):
            _run(wrapped, _voiced_hop(300 + i), _quiet_hop(), MODE_LISTENING)
    assert len(factory.enrolled) == 1
    lines = [r.getMessage() for r in caplog.records if "enrolled" in r.getMessage()]
    assert len(lines) == 1 and "s1" in lines[0]


# --- after enrollment -------------------------------------------------------------------

def test_every_hop_after_enrollment_goes_through_the_filter_in_order():
    wrapped, _inner, factory = _wrap()
    _enroll(wrapped)
    hops = [(_voiced_hop(400 + i), MODE_SPEAKING if i % 2 else MODE_LISTENING) for i in range(6)]
    for user, mode in hops:
        _run(wrapped, user, _agent_hop(), mode)
    pushed = factory.filters[0].pushed
    assert len(pushed) == len(hops)
    for got, (user, _mode) in zip(pushed, hops):
        assert np.array_equal(got, _floats(user))


def test_a_speaking_hop_gets_the_delayed_filter_output_and_a_silent_agent_channel():
    delay = 1000
    wrapped, inner, factory = _wrap(FakeFactory(delay=delay, fn=lambda x: x * 0.5), floor_db=-120.0)
    _enroll(wrapped)
    users = [_voiced_hop(500 + i) for i in range(3)]
    for user in users:
        _run(wrapped, user, _agent_hop(), MODE_SPEAKING)
    stream = np.concatenate([np.zeros(delay, np.float32)] + [_floats(u) * 0.5 for u in users])
    for k in range(3):
        user, agent, mode = inner.hops[-3 + k]
        assert mode == MODE_SPEAKING
        assert len(user) == HOP_SAMPLES * 2 and agent == bytes(HOP_SAMPLES * 2)
        expected = stream[k * HOP_SAMPLES:(k + 1) * HOP_SAMPLES]
        assert np.max(np.abs(_floats(user) - expected)) <= 1.5 / 32768.0


def test_the_floor_noise_sits_at_the_set_level_when_the_filter_outputs_silence():
    wrapped, inner, _f = _wrap(FakeFactory(delay=384, fn=lambda x: x * 0.0), floor_db=FLOOR_DB)
    _enroll(wrapped)
    for i in range(4):
        _run(wrapped, _voiced_hop(600 + i), _agent_hop(), MODE_SPEAKING)
    noise = np.concatenate([_floats(u) for u, _a, m in inner.hops[-4:]])
    rms_db = 20 * math.log10(float(np.sqrt(np.mean(noise.astype(np.float64) ** 2))))
    assert abs(rms_db - FLOOR_DB) <= 1.0


def test_a_listening_hop_after_enrollment_reaches_the_scorer_unchanged():
    wrapped, inner, _f = _wrap()
    _enroll(wrapped)
    _run(wrapped, _voiced_hop(700), _agent_hop(), MODE_SPEAKING)
    user, agent = _voiced_hop(701), _quiet_hop()
    _run(wrapped, user, agent, MODE_LISTENING)
    assert inner.hops[-1] == (user, agent, MODE_LISTENING)


def test_reset_reaches_the_scorer():
    wrapped, inner, _f = _wrap()
    wrapped.reset()
    assert inner.resets == 1


# --- the sidecar server wraps sessions only when asked ---------------------------------

def test_the_server_wraps_a_session_only_with_a_prefilter():
    from voicemanager.pipeline.tyger import server
    inner = RecordingScorer()
    assert server.session_scorer(inner, key="k", prefilter_factory=None,
                                 floor_db=FLOOR_DB, voice_gate_db=GATE_DB) is inner
    wrapped = server.session_scorer(inner, key="k", prefilter_factory=FakeFactory(),
                                    floor_db=FLOOR_DB, voice_gate_db=GATE_DB)
    assert isinstance(wrapped, prefilter.PrefilterSessionScorer)


def test_the_wrapper_module_imports_no_onnx_runtime():
    import ast
    from pathlib import Path
    tree = ast.parse(Path(prefilter.__file__).read_text())
    names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not [n for n in names if "onnx" in n or "voxlock" in n]
