"""The hops an opted-in session feeds the turn engine, model-free.

For an opted-in session the server cuts the client's mic stream into hops of 5120
samples (320 ms at 16 kHz) and sends each to the sidecar with a second channel: the
reply the server sent, resampled to 16 kHz, with reply sample 0 at the playback
start's mic_sample, ended at the stop's reply_sample (counted at the reply's own
rate) or at the reply's end, and silence elsewhere. A hop that overlaps the played
span is in speaking mode, every other hop in listening mode.

A recording FakeTygerSidecar keeps every hop it gets. The reference resampling is
scipy's resample_poly over the whole reply, as in the offline replay harness, and
the placement must match it to within one 16 kHz sample.
"""
import asyncio
import math
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy.signal import resample_poly

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)
from fastapi.testclient import TestClient

from voicemanager.pipeline import audiocpp_stt
from voicemanager.pipeline.tts_session import _GuardedSocket
from voicemanager.pipeline.tyger import wire
from voicemanager.transport import turn_engine as te

from tyger_fake_sidecar import FakeTygerSidecar
from tyger_ws_kit import (
    TTS_RATE, FakeTTSEngine, ScriptedAudioCppSTT, StandInSidecar, barrier, errors,
    make_app, opt_in, speak,
)

HOP = wire.HOP_SAMPLES
_CHUNK = 1600  # mic samples per binary frame (100 ms)


def _reply_chunks() -> list[np.ndarray]:
    """One second of broadband noise at 24 kHz, in five chunks: a sharp alignment."""
    rng = np.random.default_rng(7)
    audio = (rng.standard_normal(TTS_RATE) * 0.25).clip(-0.9, 0.9).astype(np.float32)
    return np.split(audio, 5)


def _mic(n: int) -> np.ndarray:
    return ((np.arange(n) % 2000) - 1000).astype(np.int16)


def _resampled(pcm: bytes) -> np.ndarray:
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    g = math.gcd(wire.SAMPLE_RATE, TTS_RATE)
    y = resample_poly(x, wire.SAMPLE_RATE // g, TTS_RATE // g)
    return np.clip(np.rint(y), -32768, 32767).astype(np.int16)


def _expected_agent(total: int, start: int, ref16: np.ndarray, cut16: int | None = None):
    agent = np.zeros(total, dtype=np.int16)
    played = ref16 if cut16 is None else ref16[:cut16]
    end = min(total, start + len(played))
    agent[start:end] = played[:end - start]
    return agent, (start, start + len(played))


def _assert_placed(got: np.ndarray, want: np.ndarray) -> None:
    """got equals want shifted by at most one sample, to within rounding."""
    errs = {}
    for lag in range(-3, 4):
        shifted = np.roll(want.astype(np.int32), lag)
        errs[lag] = int(np.max(np.abs(got.astype(np.int32) - shifted)))
    best = min(errs, key=errs.get)
    assert abs(best) <= 1 and errs[best] <= 2, errs


def _modes_for(n_hops: int, span: tuple[int, int]) -> list[str]:
    s0, s1 = span
    return [wire.MODE_SPEAKING if (k * HOP < s1 and (k + 1) * HOP > s0 and s1 > s0)
            else wire.MODE_LISTENING for k in range(n_hops)]


def _send_mic(ws, samples: np.ndarray) -> None:
    for i in range(0, len(samples), _CHUNK):
        ws.send_bytes(samples[i:i + _CHUNK].astype("<i2").tobytes())


def _hops(tyger, n: int) -> list:
    assert tyger.wait_for(lambda: tyger.sessions and len(tyger.sessions[0].hops) >= n,
                          timeout=5), (len(tyger.sessions[0].hops) if tyger.sessions else None)
    return list(tyger.sessions[0].hops)


def _channel(hops, which: str) -> np.ndarray:
    return np.concatenate([np.frombuffer(getattr(h, which), dtype="<i2") for h in hops])


@pytest.fixture
def tyger():
    fake = FakeTygerSidecar().start()
    try:
        yield fake
    finally:
        fake.stop()


@pytest.fixture
def client(tyger, monkeypatch):
    monkeypatch.setattr(audiocpp_stt, "AudioCppSTTPipeline", ScriptedAudioCppSTT)
    with ThreadPoolExecutor(max_workers=1) as executor:
        yield TestClient(make_app(sidecar=StandInSidecar(tyger.url), executor=executor,
                                  tts_engine=FakeTTSEngine(_reply_chunks())))


def _start(mic_sample: int, request_id: str = "r1") -> dict:
    return {"type": "playback", "request_id": request_id, "event": "start",
            "mic_sample": mic_sample}


def _stop(mic_sample: int, reply_sample: int, request_id: str = "r1") -> dict:
    return {"type": "playback", "request_id": request_id, "event": "stop",
            "mic_sample": mic_sample, "reply_sample": reply_sample}


# --- the agent channel ----------------------------------------------------------

def test_the_reply_is_placed_at_the_start_and_the_modes_follow_the_span(client, tyger):
    n_hops, start = 6, 3000
    mic = _mic(n_hops * HOP)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        audio_start, pcm = speak(ws, "r1")
        assert audio_start["sample_rate"] == TTS_RATE
        ws.send_json(_start(start))
        _send_mic(ws, mic)
        assert errors(barrier(ws)) == []
        hops = _hops(tyger, n_hops)
    assert len(hops) == n_hops
    assert [h.mic_sample for h in hops] == [(k + 1) * HOP for k in range(n_hops)]
    assert np.array_equal(_channel(hops, "user"), mic)
    want, span = _expected_agent(n_hops * HOP, start, _resampled(pcm))
    assert span == (start, start + 16000)
    _assert_placed(_channel(hops, "agent"), want)
    assert [h.mode for h in hops] == _modes_for(n_hops, span)
    assert [h.mode for h in hops] == ["speaking"] * 4 + ["listening"] * 2


def test_the_stop_cuts_the_channel_at_its_reply_sample(client, tyger):
    n_hops, start, reply_sample = 6, 3000, 12000  # 12000 at 24 kHz is 8000 at 16 kHz
    mic = _mic(n_hops * HOP)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        _, pcm = speak(ws, "r1")
        ws.send_json(_start(start))
        ws.send_json(_stop(start + 8000, reply_sample))
        _send_mic(ws, mic)
        assert errors(barrier(ws)) == []
        hops = _hops(tyger, n_hops)
    want, span = _expected_agent(n_hops * HOP, start, _resampled(pcm), cut16=8000)
    assert span == (3000, 11000)
    agent = _channel(hops, "agent")
    _assert_placed(agent, want)
    assert not agent[11001:].any()
    assert [h.mode for h in hops] == ["speaking"] * 3 + ["listening"] * 3


def test_a_start_that_comes_a_little_late_still_places_its_hop(client, tyger):
    """The start edge refers to mic audio the server already has: the hop waits."""
    n_hops, start = 4, 4000
    mic = _mic(n_hops * HOP)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        _, pcm = speak(ws, "r1")
        _send_mic(ws, mic[:6400])            # hop 0 is complete, its start is not in
        assert errors(barrier(ws)) == []
        ws.send_json(_start(start))
        _send_mic(ws, mic[6400:])
        assert errors(barrier(ws)) == []
        hops = _hops(tyger, n_hops)
    want, span = _expected_agent(n_hops * HOP, start, _resampled(pcm))
    _assert_placed(_channel(hops, "agent"), want)
    assert [h.mode for h in hops] == _modes_for(n_hops, span)
    assert hops[0].mode == wire.MODE_SPEAKING


def test_a_start_later_than_the_lookahead_leaves_sent_hops_as_they_were(client, tyger):
    n_hops, start = 5, 1000
    mic = _mic(n_hops * HOP)
    early = 2 * HOP + 1600                    # past hop 0's end plus the lookahead
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        _, pcm = speak(ws, "r1")
        _send_mic(ws, mic[:early])
        _hops(tyger, 1)
        ws.send_json(_start(start))
        _send_mic(ws, mic[early:])
        assert errors(barrier(ws)) == []
        hops = _hops(tyger, n_hops)
    assert hops[0].mode == wire.MODE_LISTENING
    assert not np.frombuffer(hops[0].agent, dtype="<i2").any()
    want, span = _expected_agent(n_hops * HOP, start, _resampled(pcm))
    _assert_placed(_channel(hops[1:], "agent"), want[HOP:])
    assert [h.mode for h in hops[1:]] == _modes_for(n_hops, span)[1:]


def test_a_reply_never_started_holds_hops_only_for_a_while(client, tyger):
    n = te.START_WAIT_SAMPLES + 2 * HOP
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        speak(ws, "r1")
        _send_mic(ws, _mic(n))
        assert errors(barrier(ws)) == []
        hops = _hops(tyger, n // HOP)
    assert len(hops) == n // HOP
    assert all(h.mode == wire.MODE_LISTENING and not any(h.agent) for h in hops)


# --- hops without a reply -------------------------------------------------------

def test_hops_without_a_reply_are_listening_with_silence(client, tyger):
    mic = _mic(3 * HOP + 2000)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        _send_mic(ws, mic)
        assert errors(barrier(ws)) == []
        hops = _hops(tyger, 3)
        assert not tyger.wait_for(lambda: len(tyger.sessions[0].hops) > 3, timeout=0.3)
    assert [h.mode for h in hops] == [wire.MODE_LISTENING] * 3
    assert [h.mic_sample for h in hops] == [HOP, 2 * HOP, 3 * HOP]
    assert np.array_equal(_channel(hops, "user"), mic[:3 * HOP])
    assert not _channel(hops, "agent").any()


def _wav_header(n_samples: int) -> bytes:
    data_bytes = 2 * n_samples
    fmt = struct.pack("<HHIIHH", 1, 1, 16000, 32000, 2, 16)
    return (b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + 4 + 8 + data_bytes)
            + b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"LIST" + struct.pack("<I", 4) + b"INFO"
            + b"data" + struct.pack("<I", data_bytes))


def test_a_wav_header_on_the_first_frame_is_not_counted(client, tyger):
    mic = _mic(2 * HOP)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        ws.send_bytes(_wav_header(3000) + mic[:3000].astype("<i2").tobytes())
        _send_mic(ws, mic[3000:])
        assert errors(barrier(ws)) == []
        hops = _hops(tyger, 2)
    assert [h.mic_sample for h in hops] == [HOP, 2 * HOP]
    assert np.array_equal(_channel(hops, "user"), mic)


def test_a_connection_that_did_not_opt_in_sends_no_hops(client, tyger):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        speak(ws, "r1")
        ws.send_json(_start(0))
        _send_mic(ws, _mic(3 * HOP))
        assert errors(barrier(ws)) == []
    assert tyger.sessions == []


def test_hops_are_dropped_quietly_when_the_sidecar_link_is_down(client, tyger):
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        _send_mic(ws, _mic(HOP))
        _hops(tyger, 1)
        tyger.drop(0)
        _send_mic(ws, _mic(3 * HOP))
        frames = barrier(ws)
        assert errors(frames) == []
        assert any(f.get("type") == "transcription" for f in frames)


# --- the bounds on kept PCM, on the feed alone ------------------------------------

def _audio_start(request_id: str, rate: int = TTS_RATE) -> str:
    return ('{"type":"audio_start","request_id":"%s","sample_rate":%d,"channels":1,'
            '"bit_depth":16,"encoding":"pcm"}' % (request_id, rate))


def _audio_end(request_id: str) -> str:
    return '{"type":"audio_end","request_id":"%s","reason":"complete"}' % request_id


def test_a_replys_pcm_is_let_go_once_its_played_span_is_fed():
    feed = te.HopFeed()
    feed.on_tts_text("r1", _audio_start("r1"))
    feed.on_tts_pcm("r1", b"\x01\x00" * 2400)      # 0.1 s: 1600 samples at 16 kHz
    feed.on_tts_text("r1", _audio_end("r1"))
    assert feed.playback(_start(1000)) is None
    feed.feed_mic(_mic(HOP).tobytes())
    jobs = feed.due_hops()
    assert [j.mode for j in jobs] == [wire.MODE_SPEAKING]
    assert feed.reply("r1").pcm is None             # span [1000, 2600) is fed
    # The record stays, so the client's closing stop is still a valid frame.
    assert feed.playback(_stop(2600, 2400)) is None



def test_a_reply_still_sending_keeps_its_pcm():
    feed = te.HopFeed()
    feed.on_tts_text("r1", _audio_start("r1"))
    feed.on_tts_pcm("r1", b"\x01\x00" * 2400)
    assert feed.playback(_start(0)) is None
    feed.feed_mic(_mic(2 * HOP).tobytes())
    assert len(feed.due_hops()) == 2
    assert feed.reply("r1").pcm is not None         # no audio_end, no stop: more may come


def test_only_the_newest_replies_are_kept():
    feed = te.HopFeed()
    for i in range(te.MAX_KEPT_REPLIES + 2):
        feed.on_tts_text(f"r{i}", _audio_start(f"r{i}"))
    assert feed.reply("r0") is None and feed.reply("r1") is None
    assert feed.reply(f"r{te.MAX_KEPT_REPLIES + 1}") is not None
    assert feed.playback(_start(0, "r0")) is not None   # forgotten: unknown request_id


def test_a_replys_pcm_is_capped(monkeypatch):
    monkeypatch.setattr(te, "MAX_REPLY_SECONDS", 1)
    feed = te.HopFeed()
    feed.on_tts_text("r1", _audio_start("r1"))
    for _ in range(3):
        feed.on_tts_pcm("r1", b"\x00\x00" * 16000)
    reply = feed.reply("r1")
    assert reply.n_samples == TTS_RATE and reply.capped


def test_a_frame_from_a_cancelled_request_never_reaches_the_tap():
    class Socket:
        def __init__(self):
            self.sent = []

        async def send_text(self, data):
            self.sent.append(data)

        async def send_bytes(self, data):
            self.sent.append(data)

    class Tap:
        def __init__(self):
            self.seen = []

        def on_tts_text(self, request_id, text):
            self.seen.append((request_id, text))

        def on_tts_pcm(self, request_id, pcm):
            self.seen.append((request_id, pcm))

    async def scenario():
        tap, dead = Tap(), set()
        guarded = _GuardedSocket(Socket(), "r1", dead.__contains__, tap=lambda: tap)
        await guarded.send_text("t")
        await guarded.send_bytes(b"pp")
        dead.add("r1")
        await guarded.send_bytes(b"late")
        return tap.seen

    assert asyncio.run(scenario()) == [("r1", "t"), ("r1", b"pp")]
