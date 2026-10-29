"""Ending a VoXtream generation early: cancel, supersede, flush timeout, disconnect.

Driven through TTSSession with a fake WebSocket and the fake sidecar, the way the
receive loop drives it. Covers: tts_cancel closes the sidecar connection within
200 ms and the client gets exactly one audio_end with reason cancelled and no later
binary frame while the sidecar keeps emitting; a cancel before the first frame; a new
request_id closes the first connection and opens a second; the flush timeout sends
eot and ends with a normal audio_end; the disconnect teardown closes the connection.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

import authkit  # noqa: F401  test auth env; must be imported before voicemanager
import pytest

from voicemanager.pipeline.tts import TTSPipeline
from voicemanager.pipeline.tts_session import TTSSession
from voicemanager.pipeline.voxtream_tts import (
    VoxtreamSidecar, VoxtreamTTSEngine, VoxtreamTTSStreamPipeline)
from voxtream_fake_sidecar import FakeVoxtreamSidecar

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_REFS = REPO_ROOT / "voices" / "designed" / "refs.json"


class _WS:
    def __init__(self):
        self.frames: list[tuple[str, object]] = []

    async def send_text(self, data: str) -> None:
        self.frames.append(("text", json.loads(data)))

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append(("bytes", data))

    def types(self) -> list[str]:
        return [p["type"] for kind, p in self.frames if kind == "text"]

    def audio_ends(self) -> list[dict]:
        return [p for kind, p in self.frames if kind == "text" and p["type"] == "audio_end"]

    def binaries(self) -> list[bytes]:
        return [p for kind, p in self.frames if kind == "bytes"]

    def after_last_audio_end(self) -> list[tuple[str, object]]:
        last = max((i for i, (k, p) in enumerate(self.frames)
                    if k == "text" and p["type"] == "audio_end"), default=None)
        return [] if last is None else self.frames[last + 1:]


def _session(fake: FakeVoxtreamSidecar, ws: _WS) -> tuple[TTSSession, VoxtreamTTSStreamPipeline]:
    engine = VoxtreamTTSEngine(VoxtreamSidecar(sys.executable, fake.port),
                               default_voice="designed_female_s101")
    engine.load(voices_path=REAL_REFS, start_sidecar=False)
    tts = TTSPipeline(engine=engine, executor=None)
    stream = VoxtreamTTSStreamPipeline(tts)
    return TTSSession(websocket=ws, tts_pipeline=tts, tts_stream_pipeline=stream), stream


async def _until(predicate, timeout: float = 3.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("timed out waiting on the condition")
        await asyncio.sleep(0.005)


@pytest.fixture
def emitting_fake():
    fake = FakeVoxtreamSidecar(keep_emitting_after_eot=True).start()
    try:
        yield fake
    finally:
        fake.stop()


@pytest.fixture
def fake():
    sidecar = FakeVoxtreamSidecar().start()
    try:
        yield sidecar
    finally:
        sidecar.stop()


def test_cancel_closes_the_connection_and_acks_once_while_the_sidecar_keeps_emitting(emitting_fake):
    fake, ws = emitting_fake, _WS()

    async def run():
        session, _ = _session(fake, ws)
        await session.feed_stream(chunk="A long reply that keeps going.", done=True,
                                  request_id="r1")
        await _until(lambda: len(ws.binaries()) >= 3)  # audio is flowing
        t_cancel = time.monotonic()
        await session.cancel("r1")
        ends = ws.audio_ends()
        assert len(ends) == 1 and ends[0]["reason"] == "cancelled"
        await _until(lambda: len(fake.closes) == 1)
        assert fake.closes[0][1] - t_cancel < 0.2
        await asyncio.wait_for(session.wait_idle(), 3.0)
        await asyncio.sleep(0.1)  # anything the doomed task still tried to send
        assert ws.after_last_audio_end() == []
        assert len(ws.audio_ends()) == 1
    asyncio.run(run())
    assert fake.frames_sent[0] >= 3  # the sidecar was mid-stream when it was cut


def test_cancel_before_the_first_frame(fake):
    ws = _WS()

    async def run():
        session, _ = _session(fake, ws)
        await session.feed_stream(chunk="", done=False, request_id="r1")  # opens, no text
        await _until(lambda: fake.connections == 1)
        await session.cancel("r1")
        assert ws.types() == ["audio_end"]
        assert ws.audio_ends()[0]["reason"] == "cancelled"
        await _until(lambda: len(fake.closes) == 1)
        await asyncio.wait_for(session.wait_idle(), 3.0)
        await asyncio.sleep(0.05)
        assert ws.after_last_audio_end() == []
    asyncio.run(run())


def test_a_new_request_id_closes_the_first_connection_and_opens_a_second(fake):
    ws = _WS()

    async def run():
        session, _ = _session(fake, ws)
        await session.feed_stream(chunk="first reply ", done=False, request_id="r1")
        await _until(lambda: len(ws.binaries()) == 1)
        await session.feed_stream(chunk="second reply", done=True, request_id="r2")
        await _until(lambda: fake.connections == 2)
        await _until(lambda: 0 in [i for i, _ in fake.closes])  # r1's connection cut
        await asyncio.wait_for(session.wait_idle(), 3.0)
        await _until(lambda: sorted(i for i, _ in fake.closes) == [0, 1])
        ends = ws.audio_ends()
        assert [(e["request_id"], e["reason"]) for e in ends] == [
            ("r1", "cancelled"), ("r2", "complete")]
        assert fake.chunks == [["first reply "], ["second reply"]]
        assert fake.eots == [1]
    asyncio.run(run())


def test_the_flush_timeout_sends_eot_and_ends_normally(fake):
    ws = _WS()

    async def run():
        session, _ = _session(fake, ws)
        await session.feed_stream(chunk="words so far", done=False, request_id="r1")
        await _until(lambda: len(ws.binaries()) == 1)
        assert fake.eots == []
        await session.flush_timeout()
        await asyncio.wait_for(session.wait_idle(), 3.0)
        assert fake.eots == [0]
        ends = ws.audio_ends()
        assert len(ends) == 1 and ends[0]["reason"] == "complete"
        assert session.active_stream_request_id is None
    asyncio.run(run())


def test_disconnect_teardown_closes_the_connection(fake):
    ws = _WS()

    async def run():
        session, stream = _session(fake, ws)
        await session.feed_stream(chunk="still talking", done=False, request_id="r1")
        await _until(lambda: fake.connections == 1)
        await session.aclose()  # what the receive loop's finally does
        stream.dispose()
        await _until(lambda: len(fake.closes) == 1)
        assert stream.synthesis_task is None
    asyncio.run(run())
    assert fake.eots == []  # cut, not ended
