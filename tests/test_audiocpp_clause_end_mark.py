"""End mark for an audiocpp clause piece that ends without one.

Breeze on audio.cpp can cut or weaken the last word of an engine request whose
text ends on a letter or digit ("...cuts the richness of a"), and a clause-mode
piece often ends in the middle of a sentence. The audiocpp stream pipeline adds
VM_AUDIOCPP_TTS_CLAUSE_END_MARK to the text of such an engine request.

What must hold:

  * a clause engine request whose text ends on a letter or digit gets the mark;
  * a text that ends with any other character is sent as it is;
  * an empty setting sends every text as it is;
  * a sentence-mode request is sent as it is;
  * the log line of the engine request shows the text that the engine got.

Model-free and server-free: a fake engine records the texts and a fake
WebSocket takes the frames.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from voicemanager.pipeline.audiocpp_tts import AudioCppTTSStreamPipeline
from voicemanager.pipeline.tts import TTSPipeline

_NO_TIMER = {"clause_first_words": 2, "clause_first_deadline_ms": 60_000}


class _Engine:
    """Records the text of each engine request and yields one short chunk."""

    is_batch = False
    sample_rate = 24000

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def ensure_ready(self) -> None:
        return None

    async def stream_synthesize(self, text: str, **kwargs):
        self.texts.append(text)
        yield np.zeros(240, dtype=np.float32)


class _WS:
    async def send_text(self, data: str) -> None:
        json.loads(data)

    async def send_bytes(self, data: bytes) -> None:
        return None


def _speak(text: str, chunking: str = "clause", **kwargs) -> list[str]:
    """Send `text` as one done request and return the texts the engine got."""
    engine = _Engine()

    async def run() -> None:
        pipe = AudioCppTTSStreamPipeline(
            TTSPipeline(engine=engine, executor=None), clause_continue=False,
            start_buffer_ms=0, **{**_NO_TIMER, **kwargs})
        await pipe.feed_text(text, True, _WS(), "r1", chunking=chunking)
        await asyncio.wait_for(pipe.synthesis_task, 5.0)

    asyncio.run(run())
    return engine.texts


# --- clause requests ------------------------------------------------------------------

@pytest.mark.parametrize("mark", [",", "."])
def test_a_piece_that_ends_on_a_word_gets_the_mark(mark):
    assert _speak("cuts the richness of a", clause_end_mark=mark) == [
        "cuts the richness of a" + mark]


def test_a_piece_that_ends_on_a_digit_gets_the_mark():
    assert _speak("deck 12", clause_end_mark=",") == ["deck 12,"]


@pytest.mark.parametrize("text", [
    "Add them to the pot.",
    "Add them to the pot,",
    "Is it ready?",
    "Red alert!",
    'She said "engage"',
    "the second option (the fast one)",
    "and then...",
])
def test_a_piece_that_ends_with_another_character_is_unchanged(text):
    assert _speak(text, clause_end_mark=",") == [text]


def test_trailing_whitespace_does_not_hide_the_last_word():
    assert _speak("Frozen leaves in ", clause_end_mark=",") == ["Frozen leaves in,"]


def test_an_empty_mark_turns_it_off():
    assert _speak("Frozen leaves in", clause_end_mark="") == ["Frozen leaves in"]
    assert _speak("Frozen leaves in", clause_end_mark="  ") == ["Frozen leaves in"]


def test_each_engine_request_is_checked_alone():
    """Two engine requests of one clause request: only the one with no mark changes."""
    gate = asyncio.Event()

    class _Slow(_Engine):
        async def stream_synthesize(self, text: str, **kwargs):
            self.texts.append(text)
            yield np.zeros(240, dtype=np.float32)
            await gate.wait()

    engine = _Slow()

    async def run() -> None:
        pipe = AudioCppTTSStreamPipeline(
            TTSPipeline(engine=engine, executor=None), clause_continue=False,
            start_buffer_ms=0, clause_end_mark=",", **_NO_TIMER)
        ws = _WS()
        await pipe.feed_text("Frozen leaves in", False, ws, "r1", chunking="clause",
                             flush=True)
        for _ in range(200):
            if engine.texts:
                break
            await asyncio.sleep(0.005)
        await pipe.feed_text(" oil work too.", True, ws, "r1", chunking="clause")
        gate.set()
        await asyncio.wait_for(pipe.synthesis_task, 5.0)

    asyncio.run(run())
    assert engine.texts == ["Frozen leaves in,", "oil work too."]


# --- sentence requests ------------------------------------------------------------------

def test_a_sentence_request_is_unchanged():
    assert _speak("Frozen leaves in", chunking="sentence", clause_end_mark=",") == [
        "Frozen leaves in"]


# --- the log line ---------------------------------------------------------------------

def test_the_log_shows_the_text_the_engine_got(caplog):
    with caplog.at_level(logging.INFO, logger="voicemanager.pipeline.audiocpp_tts"):
        _speak("Frozen leaves in", clause_end_mark=",")
    lines = [r.getMessage() for r in caplog.records
             if r.getMessage().startswith("Clause TTS engine request")]
    assert len(lines) == 1
    assert "text='Frozen leaves in,'" in lines[0]


# --- the setting ------------------------------------------------------------------------

def test_the_mark_defaults_to_the_setting(monkeypatch):
    import voicemanager.config as cfg

    monkeypatch.setenv("VM_AUDIOCPP_TTS_CLAUSE_END_MARK", ";")
    cfg._settings = None
    try:
        pipe = AudioCppTTSStreamPipeline(TTSPipeline(engine=object(), executor=None))
        assert pipe.clause_end_mark == ";"
        explicit = AudioCppTTSStreamPipeline(TTSPipeline(engine=object(), executor=None),
                                             clause_end_mark="")
        assert explicit.clause_end_mark == ""
    finally:
        cfg._settings = None


def test_the_setting_has_its_default_and_env_name():
    from voicemanager.config import Settings

    assert Settings(_env_file=None).audiocpp_tts_clause_end_mark == ","
    text = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    assert "# VM_AUDIOCPP_TTS_CLAUSE_END_MARK=," in text
