"""The audiocpp TTS evidence script takes its sample rate from audio_start.

The script is a manual live tool (scripts/, never collected); this test loads its
helpers under a fake token and drives run_request with a scripted socket, so no
server, model or GPU is needed. The long-form WAV and its duration and RTF used
to assume 48 kHz, which mislabelled 24 kHz engines (breeze, qwen3).
"""
import asyncio
import importlib.util
import json
import pathlib
import wave

import pytest

_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "audiocpp_tts_evidence.py"


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.setenv("VOICEMANAGER_TOKEN", "vmp_test")
    spec = importlib.util.spec_from_file_location("audiocpp_tts_evidence", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ScriptedSocket:
    """Answers one tts request with the frames given, in order."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []

    async def send(self, data):
        self.sent.append(json.loads(data))

    async def recv(self):
        return self.frames.pop(0)


def test_run_request_reports_the_audio_start_rate(evidence):
    pcm = b"\x01\x00" * 2400
    ws = _ScriptedSocket([
        json.dumps({"type": "audio_start", "request_id": "lf1", "sample_rate": 24000}),
        pcm[:2000],
        pcm[2000:],
        json.dumps({"type": "audio_end", "request_id": "lf1", "reason": "complete"}),
    ])
    first, total, got, rate = asyncio.run(evidence.run_request(ws, "lf1", "Hello."))
    assert got == pcm and rate == 24000
    assert first is not None and total >= first
    assert ws.sent == [{"type": "tts", "text": "Hello.", "request_id": "lf1"}]


def test_run_request_without_audio_start_reports_no_rate(evidence):
    ws = _ScriptedSocket([json.dumps({"type": "audio_end", "request_id": "x", "reason": "error"})])
    first, total, pcm, rate = asyncio.run(evidence.run_request(ws, "x", "Hello."))
    assert (first, pcm, rate) == (None, b"", None)


def test_wav_and_duration_follow_the_rate(evidence, tmp_path):
    pcm = b"\x00\x00" * 24000  # one second at 24 kHz
    assert evidence.pcm_duration_s(pcm, 24000) == 1.0
    assert evidence.pcm_duration_s(pcm, 48000) == 0.5
    out = tmp_path / "longform.wav"
    evidence.write_wav(str(out), pcm, 24000)
    with wave.open(str(out), "rb") as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (24000, 1, 2)
        assert f.getnframes() == 24000


def test_no_fixed_rate_remains_in_the_script():
    source = _PATH.read_text(encoding="utf-8")
    assert "48000" not in source
