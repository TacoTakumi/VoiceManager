"""The streaming silence limiter caps each silent run and leaves speech alone.

What must hold:

  * a silent run longer than the cap comes out with the length of the cap, and
    the audio around it is unchanged;
  * a run at or below the cap is unchanged;
  * the output does not depend on how the input is cut into chunks;
  * between feeds the limiter holds less than one window;
  * with the cap at 0 the input comes out unchanged and the run is still measured.

Model-free and server-free.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from voicemanager.pipeline.silence_limiter import SilenceLimiter

SR = 24000
WINDOW = SR // 100


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(SR * seconds), dtype=np.float32) / SR
    return (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


def run(signal: np.ndarray, cap_ms: int, sizes=None) -> tuple[np.ndarray, SilenceLimiter]:
    """Feed `signal` in chunks of the given sizes (one chunk when None), then flush."""
    limiter = SilenceLimiter(SR, cap_ms)
    out = []
    pos = 0
    sizes = iter(sizes) if sizes is not None else iter([len(signal)])
    while pos < len(signal):
        size = next(sizes)
        out.append(limiter.feed(signal[pos:pos + size]))
        assert limiter.held_samples < WINDOW
        pos += size
    out.append(limiter.flush())
    return np.concatenate(out), limiter


def test_a_long_run_is_cut_to_the_cap():
    a, b = tone(0.5), tone(0.5)
    out, _ = run(np.concatenate([a, silence(1.0), b]), 400)
    assert abs(len(out) - int(SR * 1.4)) <= WINDOW
    assert np.array_equal(out[:len(a)], a)
    assert np.array_equal(out[-len(b):], b)
    assert not out[len(a):-len(b)].any()


def test_a_run_below_the_cap_is_unchanged():
    signal = np.concatenate([tone(0.5), silence(0.3), tone(0.5)])
    out, limiter = run(signal, 400)
    assert np.array_equal(out, signal)
    assert limiter.take_stats() == (300, 0)


def test_a_run_of_exactly_the_cap_is_unchanged():
    signal = np.concatenate([tone(0.2), silence(0.4), tone(0.2)])
    out, _ = run(signal, 400)
    assert np.array_equal(out, signal)


def test_low_noise_counts_as_silence_and_quiet_speech_does_not():
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(SR) * 0.001).astype(np.float32)   # about -60 dBFS
    quiet = (tone(1.0) * 0.1).astype(np.float32)                   # about -33 dBFS
    out, _ = run(np.concatenate([tone(0.2), noise, tone(0.2)]), 400)
    assert abs(len(out) - int(SR * 0.8)) <= WINDOW
    signal = np.concatenate([tone(0.2), quiet, tone(0.2)])
    out, _ = run(signal, 400)
    assert np.array_equal(out, signal)


def test_the_output_does_not_depend_on_the_chunk_sizes():
    signal = np.concatenate([
        tone(0.33), silence(1.0), tone(0.207), silence(0.45), tone(0.1), silence(0.7)])
    whole, _ = run(signal, 400)
    small = SR * 7 // 1000

    def sevens():
        while True:
            yield small

    def randoms():
        rnd = random.Random(7)
        while True:
            yield rnd.randint(1, 3000)

    by_seven, _ = run(signal, 400, sevens())
    by_random, _ = run(signal, 400, randoms())
    assert np.array_equal(whole, by_seven)
    assert np.array_equal(whole, by_random)


def test_a_run_across_two_feeds_is_one_run():
    limiter = SilenceLimiter(SR, 400)
    first = limiter.feed(np.concatenate([tone(0.5), silence(0.3)]))
    second = limiter.feed(np.concatenate([silence(0.3), tone(0.5)]))
    out = np.concatenate([first, second, limiter.flush()])
    assert abs(len(out) - int(SR * 1.4)) <= WINDOW


def test_the_tail_of_the_stream_is_capped():
    out, limiter = run(np.concatenate([tone(0.5), silence(0.777)]), 400)
    assert abs(len(out) - int(SR * 0.9)) <= WINDOW
    longest, dropped = limiter.take_stats()
    assert abs(longest - 777) <= 10
    assert abs(dropped - 377) <= 10


def test_stats_report_the_longest_run_and_the_time_dropped_then_reset():
    limiter = SilenceLimiter(SR, 400)
    limiter.feed(np.concatenate([tone(0.5), silence(1.0), tone(0.5)]))
    assert limiter.take_stats() == (1000, 600)
    limiter.feed(tone(0.5))
    assert limiter.take_stats() == (0, 0)


@pytest.mark.parametrize("sizes", [None, [1234] * 100])
def test_cap_zero_passes_the_input_unchanged_and_still_measures(sizes):
    signal = np.concatenate([tone(0.5), silence(1.0), tone(0.5)])
    out, limiter = run(signal, 0, sizes)
    assert np.array_equal(out, signal)
    assert not limiter.enabled
    assert limiter.take_stats() == (1000, 0)


def test_cap_zero_returns_each_chunk_as_it_was_given():
    limiter = SilenceLimiter(SR, 0)
    chunk = silence(0.0123)
    assert limiter.feed(chunk) is chunk
    assert len(limiter.flush()) == 0
