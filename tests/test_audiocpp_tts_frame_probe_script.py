"""The audiocpp TTS frame probe script: its rows and its session open.

The script is a manual live tool (scripts/, never collected). These tests load
it as a module without running it and drive its helpers with fakes, so no
server, sidecar, model or GPU is needed.
"""
import argparse
import asyncio
import importlib.util
import pathlib
import re
import threading

import pytest

_PATH = (pathlib.Path(__file__).resolve().parents[1] / "scripts"
         / "audiocpp_tts_frame_probe.py")


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location("audiocpp_tts_frame_probe", _PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame_line(sent: str, frames: int) -> str:
    return (f"TTS frames: request_id=r1 chunking=clause frames={frames} "
            f"audio_ms=9000 span_ms=8000 min_reserve_ms=120 at_ms=40 sent=[{sent}]")


class _FakePipe:
    """Stands in for the stream pipeline: logs one frame line at the final feed."""

    def __init__(self, lines, line: str) -> None:
        self._lines = lines
        self._line = line
        self.synthesis_task = None

    async def feed_text(self, text, final, ws, request_id, voice=None, chunking=None):
        del text, request_id, voice, chunking
        await ws.send_bytes(b"\x00\x00")
        if final:
            self._lines.frame_line = self._line
            self.synthesis_task = asyncio.get_running_loop().create_future()
            self.synthesis_task.set_result(None)


def _speak(probe, line: str) -> dict:
    lines = probe._Lines()
    args = argparse.Namespace(feed_ms=0, voice=None, chunking="clause")
    pipe = _FakePipe(lines, line)
    return asyncio.run(probe.speak(pipe, "Hello there.", "r1", args, lines))


def test_a_cut_frame_list_marks_the_row_with_the_count_left_out(probe):
    line = _frame_line("0:40 50:40 | 300:40 +37 more", frames=240)
    row = _speak(probe, line)
    assert row["cut"] == 37
    # The longest wait is still read from the frames that the line lists.
    assert (row["wait_ms"], row["wait_at"]) == (250, 300)
    printed = probe.format_row("run1-text1", row)
    assert "+37" in printed


def test_a_whole_frame_list_has_no_cut_mark(probe):
    row = _speak(probe, _frame_line("0:40 50:40 100:40", frames=3))
    assert row["cut"] == 0
    assert "+" not in probe.format_row("run1-text1", row)


def test_the_frame_column_comes_from_the_frame_line(probe):
    row = _speak(probe, _frame_line("0:40 50:40 100:40", frames=3))
    assert row["frames"] == "3"


def test_the_websocket_stand_in_keeps_no_unused_frame_count():
    source = _PATH.read_text()
    assert not re.search(r"""row\[["']frames["']\]\s*=""", source)
    assert not re.search(r"""["']frames["']\s*:\s*ws\.""", source)
    assert "self.frames" not in source


class _FakeClient:
    def __init__(self) -> None:
        self.closed = False
        self._reaper = None

    def close(self) -> None:
        self.closed = True


class _BlockedEngine:
    """A session engine whose connect blocks until the test lets it go."""

    def __init__(self) -> None:
        self._client = _FakeClient()
        self.connecting = threading.Event()
        self.release = threading.Event()
        self.connected = False

    def connect(self) -> None:
        self.connecting.set()
        assert self.release.wait(10.0), "the test never let the connect go"
        self.connected = True


class _BaseEngine:
    def __init__(self, engine) -> None:
        self._engine = engine

    def session_engine(self):
        return self._engine


def test_a_cancel_during_the_connect_closes_the_engine(probe):
    engine = _BlockedEngine()
    args = argparse.Namespace(voice=None, silence_cap_ms=None)

    async def scenario():
        task = asyncio.create_task(
            probe.open_session(_BaseEngine(engine), settings=None, args=args))
        assert await asyncio.to_thread(engine.connecting.wait, 10.0)
        task.cancel()
        # Let the cancel land while the connect thread is still blocked.
        await asyncio.sleep(0.05)
        engine.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert engine.connected
    assert engine._client.closed
