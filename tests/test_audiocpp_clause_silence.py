"""A clause request on the audiocpp engine sends no silent run longer than the cap.

Breeze makes a pause of up to about a second at a period, inside one engine
request and at its end. The audiocpp stream pipeline caps each silent run of a
chunking="clause" request at VM_AUDIOCPP_TTS_CLAUSE_SILENCE_MS.

What must hold:

  * a silent run longer than the cap is sent with the length of the cap, and the
    audio around it is sent unchanged;
  * a run that goes from the end of one engine request into the start of the
    next is one run;
  * a sentence request, and a clause request with the cap at 0, send the
    engine's audio unchanged;
  * the unplayed-audio estimate counts the samples sent;
  * the composed reference gets the engine's own audio;
  * audio that is not a whole number of 10 ms windows arrives in full: the
    rest that the limiter keeps between chunks goes out before the request ends;
  * each engine request logs its longest silent run and the time dropped;
  * the cap defaults to the setting.

Model-free and server-free: a fake engine gives the audio and a fake WebSocket
takes the frames.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

import authkit  # test auth env; must be imported before voicemanager

from voicemanager.config import Settings
from voicemanager.pipeline import audiocpp_tts
from voicemanager.pipeline.audiocpp_tts import AudioCppTTSStreamPipeline
from voicemanager.pipeline.silence_limiter import SilenceLimiter
from voicemanager.pipeline.tts import TTSPipeline, float32_to_pcm16

SR = 24000
WINDOW = SR // 100
CHUNK = 1920  # 80 ms, about the size of an engine chunk
_NO_TIMER = {"clause_first_words": 2, "clause_first_deadline_ms": 60_000}


def tone(seconds: float) -> np.ndarray:
    t = np.arange(int(SR * seconds), dtype=np.float32) / SR
    return (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


def pcm(audio: np.ndarray) -> bytes:
    return float32_to_pcm16(audio)


LONG_PAUSE = np.concatenate([tone(0.5), silence(1.0), tone(0.5)])


class _Engine:
    """Gives one prepared signal for each engine request, in chunks of 80 ms."""

    is_batch = False
    sample_rate = SR

    def __init__(self, signals: list[np.ndarray], gate: asyncio.Event | None = None,
                 chunk: int = CHUNK):
        self._signals = list(signals)
        self._gate = gate
        self._chunk = chunk
        self.texts: list[str] = []

    async def ensure_ready(self) -> None:
        return None

    def resolve_voice(self, voice):
        return SimpleNamespace(wav_path="unused.wav", ref_text="unused")

    async def stream_synthesize(self, text: str, **kwargs):
        self.texts.append(text)
        signal = self._signals[len(self.texts) - 1]
        for pos in range(0, len(signal), self._chunk):
            yield signal[pos:pos + self._chunk]
        if self._gate is not None and len(self.texts) == 1:
            await self._gate.wait()


class _WS:
    def __init__(self) -> None:
        self.audio = bytearray()

    async def send_text(self, data: str) -> None:
        json.loads(data)

    async def send_bytes(self, data: bytes) -> None:
        self.audio += data


def _pipe(engine, **kwargs) -> AudioCppTTSStreamPipeline:
    options = {"clause_continue": False, "start_buffer_ms": 0, **_NO_TIMER, **kwargs}
    return AudioCppTTSStreamPipeline(TTSPipeline(engine=engine, executor=None), **options)


def _speak(signal: np.ndarray, chunking: str = "clause", chunk: int = CHUNK, **kwargs):
    """One request with one engine request; returns (bytes sent, samples reported)."""
    engine = _Engine([signal], chunk=chunk)
    ws = _WS()
    reported: list[int] = []

    async def run() -> None:
        pipe = _pipe(engine, **kwargs)
        real = pipe._report_audio

        def report(request_id, n_samples, sample_rate):
            reported.append(n_samples)
            real(request_id, n_samples, sample_rate)

        pipe._report_audio = report
        await pipe.feed_text("Permissions are set.", True, ws, "r1", chunking=chunking)
        await asyncio.wait_for(pipe.synthesis_task, 5.0)

    asyncio.run(run())
    assert len(engine.texts) == 1
    return bytes(ws.audio), sum(reported)


# --- the cap ----------------------------------------------------------------------------

def test_a_long_pause_is_sent_with_the_length_of_the_cap():
    sent, _ = _speak(LONG_PAUSE, clause_silence_ms=400)
    samples = len(sent) // 2
    assert abs(samples - int(SR * 1.4)) <= WINDOW
    head = pcm(tone(0.5))
    assert sent[:len(head)] == head
    assert sent[-len(head):] == head
    assert not any(sent[len(head):-len(head)])


def test_a_pause_below_the_cap_is_sent_unchanged():
    signal = np.concatenate([tone(0.5), silence(0.3), tone(0.5)])
    sent, _ = _speak(signal, clause_silence_ms=400)
    assert sent == pcm(signal)


def test_the_silent_tail_of_the_last_engine_request_is_capped():
    sent, _ = _speak(np.concatenate([tone(0.5), silence(0.77)]), clause_silence_ms=400)
    assert abs(len(sent) // 2 - int(SR * 0.9)) <= WINDOW


def test_the_start_buffer_holds_audio_after_the_limiter():
    sent, reported = _speak(LONG_PAUSE, clause_silence_ms=400, start_buffer_ms=250)
    assert abs(len(sent) // 2 - int(SR * 1.4)) <= WINDOW
    assert reported == len(sent) // 2


def test_a_run_across_two_engine_requests_is_one_run():
    """0.3 s at the end of the first engine request and 0.3 s at the start of the
    second: the client gets 0.4 s between the two tones."""
    first = np.concatenate([tone(0.5), silence(0.3)])
    second = np.concatenate([silence(0.3), tone(0.5)])
    ws = _WS()

    async def run() -> list[str]:
        gate = asyncio.Event()
        engine = _Engine([first, second], gate)
        pipe = _pipe(engine, clause_silence_ms=400)
        await pipe.feed_text("Permissions are set.", False, ws, "r1", chunking="clause",
                             flush=True)
        for _ in range(200):
            if engine.texts:
                break
            await asyncio.sleep(0.005)
        await pipe.feed_text(" The file is writable.", True, ws, "r1", chunking="clause")
        gate.set()
        await asyncio.wait_for(pipe.synthesis_task, 5.0)
        return engine.texts

    assert len(asyncio.run(run())) == 2
    sent = bytes(ws.audio)
    head = pcm(tone(0.5))
    assert sent[:len(head)] == head
    assert sent[-len(head):] == head
    between = sent[len(head):-len(head)]
    assert not any(between)
    assert abs(len(between) // 2 - int(SR * 0.4)) <= WINDOW


# --- audio that does not end on a window ------------------------------------------------

def _limited(signal: np.ndarray, cap_ms: int) -> np.ndarray:
    """What a limiter of its own makes of the whole signal."""
    limiter = SilenceLimiter(SR, cap_ms)
    return np.concatenate([limiter.feed(signal), limiter.flush()])


def _ramp(samples: int) -> np.ndarray:
    """A short piece that is not silence and has no two equal samples."""
    return np.linspace(0.2, 0.6, samples, dtype=np.float32)


@pytest.mark.parametrize("chunk", [1000, 1920, 7])
@pytest.mark.parametrize("start_buffer_ms", [0, 250])
def test_the_rest_of_an_unaligned_request_is_sent_before_it_ends(chunk, start_buffer_ms):
    """The signal is 157 samples longer than a whole number of windows. Those
    samples wait in the limiter after the last engine chunk; the request must
    still send them."""
    signal = np.concatenate([LONG_PAUSE, _ramp(157)])
    assert len(signal) % WINDOW
    expected = _limited(signal, 400)
    sent, reported = _speak(signal, chunk=chunk, clause_silence_ms=400,
                            start_buffer_ms=start_buffer_ms)
    assert sent == pcm(expected)
    assert sent.endswith(pcm(_ramp(157)))
    assert reported == len(expected)
    assert abs(len(expected) - (len(signal) - int(SR * 0.6))) <= WINDOW


def test_the_rest_goes_on_into_the_next_engine_request_and_out_at_the_end():
    """Two engine requests, neither a whole number of windows: the rest of the
    first joins the second, and the rest of the second goes out at the end."""
    first = np.concatenate([tone(0.5), silence(0.3), np.zeros(100, dtype=np.float32)])
    second = np.concatenate([silence(0.3), tone(0.5), _ramp(57)])
    assert len(first) % WINDOW and len(second) % WINDOW
    expected = _limited(np.concatenate([first, second]), 400)
    ws = _WS()
    reported: list[int] = []

    async def run() -> int:
        gate = asyncio.Event()
        engine = _Engine([first, second], gate, chunk=1000)
        pipe = _pipe(engine, clause_silence_ms=400)
        real = pipe._report_audio

        def report(request_id, n_samples, sample_rate):
            reported.append(n_samples)
            real(request_id, n_samples, sample_rate)

        pipe._report_audio = report
        await pipe.feed_text("Permissions are set.", False, ws, "r1", chunking="clause",
                             flush=True)
        for _ in range(200):
            if engine.texts:
                break
            await asyncio.sleep(0.005)
        await pipe.feed_text(" The file is writable.", True, ws, "r1", chunking="clause")
        gate.set()
        await asyncio.wait_for(pipe.synthesis_task, 5.0)
        return len(engine.texts)

    assert asyncio.run(run()) == 2
    sent = bytes(ws.audio)
    assert sent == pcm(expected)
    assert sent.endswith(pcm(_ramp(57)))
    assert sum(reported) == len(expected)


# --- where the limiter is not used ------------------------------------------------------

def test_a_sentence_request_sends_the_engine_audio_unchanged():
    sent, _ = _speak(LONG_PAUSE, chunking="sentence", clause_silence_ms=400)
    assert sent == pcm(LONG_PAUSE)


def test_cap_zero_sends_the_engine_audio_unchanged():
    sent, reported = _speak(LONG_PAUSE, clause_silence_ms=0)
    assert sent == pcm(LONG_PAUSE)
    assert reported == len(LONG_PAUSE)


# --- what counts the audio --------------------------------------------------------------

def test_the_unplayed_audio_estimate_counts_the_samples_sent():
    sent, reported = _speak(LONG_PAUSE, clause_silence_ms=400)
    assert reported == len(sent) // 2
    assert abs(reported - int(SR * 1.4)) <= WINDOW


def test_the_composed_reference_gets_the_engine_audio(monkeypatch):
    given: list[np.ndarray] = []
    real = audiocpp_tts._ClauseReference.add_audio

    def add_audio(self, chunk):
        given.append(np.array(chunk))
        real(self, chunk)

    monkeypatch.setattr(audiocpp_tts._ClauseReference, "add_audio", add_audio)
    sent, _ = _speak(LONG_PAUSE, clause_silence_ms=400, clause_continue=True)
    assert np.array_equal(np.concatenate(given), LONG_PAUSE)
    assert abs(len(sent) // 2 - int(SR * 1.4)) <= WINDOW


# --- the log line -----------------------------------------------------------------------

_LINE = re.compile(
    r"Clause TTS silence: request_id=(\S+) longest_run_ms=(\d+) dropped_ms=(\d+)")


def _silence_lines(caplog) -> list[tuple[str, int, int]]:
    found = []
    for record in caplog.records:
        match = _LINE.fullmatch(record.getMessage())
        if match:
            assert record.levelno == logging.INFO
            found.append((match.group(1), int(match.group(2)), int(match.group(3))))
    return found


def test_each_engine_request_logs_its_longest_run_and_the_time_dropped(caplog):
    with caplog.at_level(logging.INFO, logger="voicemanager.pipeline.audiocpp_tts"):
        _speak(LONG_PAUSE, clause_silence_ms=400)
    lines = _silence_lines(caplog)
    assert len(lines) == 1
    request_id, longest, dropped = lines[0]
    assert request_id == "r1"
    assert abs(longest - 1000) <= 10
    assert abs(dropped - 600) <= 10


def test_cap_zero_logs_the_run_with_nothing_dropped(caplog):
    with caplog.at_level(logging.INFO, logger="voicemanager.pipeline.audiocpp_tts"):
        _speak(LONG_PAUSE, clause_silence_ms=0)
    lines = _silence_lines(caplog)
    assert len(lines) == 1
    assert abs(lines[0][1] - 1000) <= 10
    assert lines[0][2] == 0


def test_a_sentence_request_logs_no_silence_line(caplog):
    with caplog.at_level(logging.INFO, logger="voicemanager.pipeline.audiocpp_tts"):
        _speak(LONG_PAUSE, chunking="sentence", clause_silence_ms=400)
    assert _silence_lines(caplog) == []


# --- the setting ------------------------------------------------------------------------

ENV = "VM_AUDIOCPP_TTS_CLAUSE_SILENCE_MS"


def test_the_setting_has_its_default_and_env_name(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert Settings(_env_file=None).audiocpp_tts_clause_silence_ms == 400
    monkeypatch.setenv(ENV, "0")
    assert Settings(_env_file=None).audiocpp_tts_clause_silence_ms == 0
    monkeypatch.setenv(ENV, "250")
    assert Settings(_env_file=None).audiocpp_tts_clause_silence_ms == 250


def test_a_negative_cap_is_refused(monkeypatch):
    monkeypatch.setenv(ENV, "-1")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_the_cap_defaults_to_the_setting(monkeypatch):
    import voicemanager.config as cfg

    monkeypatch.setenv(ENV, "123")
    cfg._settings = None
    try:
        pipe = AudioCppTTSStreamPipeline(TTSPipeline(engine=object(), executor=None))
        assert pipe.clause_silence_ms == 123
        explicit = AudioCppTTSStreamPipeline(TTSPipeline(engine=object(), executor=None),
                                             clause_silence_ms=0)
        assert explicit.clause_silence_ms == 0
    finally:
        cfg._settings = None
