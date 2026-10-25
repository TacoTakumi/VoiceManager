"""Clause-mode requests log the exact text of each engine request at INFO.

A sentence request logs no such line. Model-free: fake engine, fake WebSocket.
"""
import asyncio
import json
import logging

import numpy as np
import pytest

from voicemanager.pipeline.audiocpp_tts import AudioCppTTSStreamPipeline
from voicemanager.pipeline.tts import TTSPipeline, TTSStreamPipeline

_MARKER = "Clause TTS engine request"


class _Engine:
    is_batch = False
    sample_rate = 24000

    def __init__(self):
        self.texts: list[str] = []

    async def ensure_ready(self) -> None:
        return None

    async def stream_synthesize(self, text: str, **kwargs):
        self.texts.append(text)
        yield np.zeros(10, dtype=np.float32)


class _WS:
    async def send_text(self, data: str) -> None:
        json.loads(data)

    async def send_bytes(self, data: bytes) -> None:
        pass


def _lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.INFO and _MARKER in r.getMessage()]


async def _run(cls, chunking: str, pieces: list[str]) -> _Engine:
    engine = _Engine()
    pipe = cls(TTSPipeline(engine=engine, executor=None),
               clause_first_deadline_ms=60_000)
    ws = _WS()
    for piece in pieces:
        await pipe.feed_text(piece, False, ws, "r1", chunking=chunking)
    await pipe.feed_text("", True, ws, "r1")
    await asyncio.wait_for(pipe.synthesis_task, 2.0)
    return engine


@pytest.mark.parametrize("cls", [TTSStreamPipeline, AudioCppTTSStreamPipeline])
def test_a_clause_request_logs_each_engine_request(cls, caplog):
    caplog.set_level(logging.INFO)
    engine = asyncio.run(_run(cls, "clause", ["Well, I think so, yes."]))
    lines = _lines(caplog)
    assert len(lines) == len(engine.texts) >= 1
    for line, text in zip(lines, engine.texts):
        assert "request_id=r1" in line
        assert repr(text) in line


@pytest.mark.parametrize("cls", [TTSStreamPipeline, AudioCppTTSStreamPipeline])
def test_a_sentence_request_logs_no_engine_request_line(cls, caplog):
    caplog.set_level(logging.INFO)
    engine = asyncio.run(_run(cls, "sentence", ["Well, I think so. Yes."]))
    assert engine.texts
    assert _lines(caplog) == []
