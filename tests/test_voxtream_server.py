"""The VoXtream sidecar server's wire contract, with a fake generator (no model).

Covers: one config frame, the generator's frames as float32 bytes in order, eos; an
error frame when generation fails; one generation at a time across connections; a
connection that closes early stops its generation, and the next one is served;
/health is 503 before the model is set and 200 after; the default bind is loopback.
"""
import asyncio
import json
import struct
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from voicemanager.pipeline import voxtream_server
from voicemanager.pipeline.voxtream_server import GenerationCancelled, TextFeed, create_app

FRAME = 1920  # 80 ms at 24 kHz


def _frame(i: int) -> np.ndarray:
    return np.full(FRAME, float(i) / 100.0, dtype=np.float32)


class FakeGenerator:
    """Stands in for voxtream's SpeechGenerator.

    ``generate_stream`` polls the text feed the way the real one does: a None poll
    means no text yet (it sleeps a little), a chunk gives one frame per word, and the
    end of the text gives one trailing frame. It records when each generation started,
    ended and whether it was closed early.
    """

    def __init__(self, fail: bool = False, frame_delay_s: float = 0.0):
        self.fail = fail
        self.frame_delay_s = frame_delay_s
        self.starts: list[float] = []
        self.ends: list[float] = []
        self.closed_at: list[float | None] = []
        self.frames_out: list[int] = []
        self.prompts: list[Path] = []
        self.lock = threading.Lock()

    def generate_stream(self, prompt_audio_path, text):
        idx = len(self.starts)
        self.starts.append(time.monotonic())
        self.ends.append(0.0)
        self.closed_at.append(None)
        self.frames_out.append(0)
        self.prompts.append(prompt_audio_path)
        n = 0
        try:
            if self.fail:
                raise RuntimeError("fake generation failure")
            if isinstance(text, str):
                words = text.split()
                for _ in words:
                    n += 1
                    self.frames_out[idx] = n
                    time.sleep(self.frame_delay_s)
                    yield _frame(n), {}
            else:
                while True:
                    try:
                        chunk = next(text)
                    except StopIteration:
                        break
                    if chunk is None:
                        time.sleep(0.005)
                        continue
                    for _ in chunk.split():
                        n += 1
                        self.frames_out[idx] = n
                        time.sleep(self.frame_delay_s)
                        yield _frame(n), {}
            n += 1
            self.frames_out[idx] = n
            yield _frame(n), {}
        except GeneratorExit:
            self.closed_at[idx] = time.monotonic()
            raise
        finally:
            self.ends[idx] = time.monotonic()


@pytest.fixture
def prompt_wav(tmp_path: Path) -> Path:
    path = tmp_path / "prompt.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(struct.pack("<h", 0) * 2400)
    return path


def _init(prompt: Path, **extra) -> str:
    payload = {"event": "init", "prompt_audio_path": str(prompt), "full_stream": True}
    payload.update(extra)
    return json.dumps(payload)


def _recv_all(ws) -> tuple[list[dict], list[bytes]]:
    """Read frames until eos or error; return (text frames, binary frames)."""
    texts, blobs = [], []
    while True:
        msg = ws.receive()
        if msg["type"] == "websocket.close":
            break
        if msg.get("bytes") is not None:
            blobs.append(msg["bytes"])
            continue
        payload = json.loads(msg["text"])
        texts.append(payload)
        if payload["type"] in ("eos", "error"):
            break
    return texts, blobs


def test_text_feed_polls_without_blocking():
    feed = TextFeed()
    assert next(feed) is None
    feed.push("hello ")
    feed.push("world")
    assert next(feed) == "hello "
    assert next(feed) == "world"
    assert next(feed) is None
    feed.end()
    with pytest.raises(StopIteration):
        next(feed)
    cancelled = TextFeed()
    cancelled.push("late")
    cancelled.cancel()
    with pytest.raises(GenerationCancelled):
        next(cancelled)


def test_one_config_frames_in_order_then_eos(prompt_wav):
    fake = FakeGenerator()
    with TestClient(create_app(generator=fake)) as client:
        with client.websocket_connect("/voxtream") as ws:
            ws.send_text(_init(prompt_wav))
            ws.send_text(json.dumps({"event": "text", "chunk": "one two "}))
            ws.send_text(json.dumps({"event": "text", "chunk": "three"}))
            ws.send_text(json.dumps({"event": "eot"}))
            texts, blobs = _recv_all(ws)
    configs = [t for t in texts if t["type"] == "config"]
    assert configs == [{"type": "config", "sample_rate": 24000, "dtype": "float32",
                        "channels": 1}]
    assert texts[-1] == {"type": "eos"}
    # three words plus the trailing frame, each the fake's frame, in order
    assert len(blobs) == 4
    for i, blob in enumerate(blobs, start=1):
        assert np.frombuffer(blob, dtype=np.float32).tolist() == _frame(i).tolist()
    assert fake.prompts == [prompt_wav]


def test_whole_text_in_init_without_full_stream(prompt_wav):
    fake = FakeGenerator()
    with TestClient(create_app(generator=fake)) as client:
        with client.websocket_connect("/voxtream") as ws:
            ws.send_text(_init(prompt_wav, full_stream=False, text="four words in here"))
            texts, blobs = _recv_all(ws)
    assert texts[-1] == {"type": "eos"}
    assert len(blobs) == 5


def test_generation_error_gives_error_frame(prompt_wav):
    fake = FakeGenerator(fail=True)
    with TestClient(create_app(generator=fake)) as client:
        with client.websocket_connect("/voxtream") as ws:
            ws.send_text(_init(prompt_wav))
            ws.send_text(json.dumps({"event": "eot"}))
            texts, blobs = _recv_all(ws)
    assert texts[0]["type"] == "config"
    assert texts[-1]["type"] == "error"
    assert "fake generation failure" in texts[-1]["message"]
    assert blobs == []


def test_bad_init_and_missing_prompt_are_errors(tmp_path):
    fake = FakeGenerator()
    with TestClient(create_app(generator=fake)) as client:
        with client.websocket_connect("/voxtream") as ws:
            ws.send_text(json.dumps({"event": "text", "chunk": "no init"}))
            texts, _ = _recv_all(ws)
        assert texts[0]["type"] == "error"
        with client.websocket_connect("/voxtream") as ws:
            ws.send_text(_init(tmp_path / "missing.wav"))
            texts, _ = _recv_all(ws)
        assert texts[0]["type"] == "error"
        assert "prompt audio" in texts[0]["message"]
    assert fake.starts == []


def test_generations_run_one_at_a_time(prompt_wav):
    fake = FakeGenerator(frame_delay_s=0.01)
    with TestClient(create_app(generator=fake)) as client:
        with client.websocket_connect("/voxtream") as first:
            first.send_text(_init(prompt_wav))
            first.send_text(json.dumps({"event": "text", "chunk": "a b c d e"}))
            assert json.loads(first.receive_text())["type"] == "config"
            with client.websocket_connect("/voxtream") as second:
                second.send_text(_init(prompt_wav))
                second.send_text(json.dumps({"event": "text", "chunk": "x y"}))
                second.send_text(json.dumps({"event": "eot"}))
                # The second generation must not start while the first is open.
                time.sleep(0.2)
                assert len(fake.starts) == 1
                first.send_text(json.dumps({"event": "eot"}))
                texts1, blobs1 = _recv_all(first)
                texts2, blobs2 = _recv_all(second)
    assert texts1[-1] == {"type": "eos"} and texts2[-1] == {"type": "eos"}
    assert len(blobs1) == 6 and len(blobs2) == 3
    assert len(fake.starts) == 2
    assert fake.starts[1] >= fake.ends[0]


def test_early_close_stops_generation_and_next_connection_is_served(prompt_wav):
    fake = FakeGenerator(frame_delay_s=0.02)
    with TestClient(create_app(generator=fake)) as client:
        with client.websocket_connect("/voxtream") as ws:
            ws.send_text(_init(prompt_wav))
            ws.send_text(json.dumps({"event": "text", "chunk": " ".join(["w"] * 200)}))
            assert json.loads(ws.receive_text())["type"] == "config"
            assert ws.receive()["bytes"] is not None
            assert ws.receive()["bytes"] is not None
            t_close = time.monotonic()
        deadline = time.monotonic() + 0.5
        while fake.closed_at[0] is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert fake.closed_at[0] is not None
        assert fake.closed_at[0] - t_close < 0.5
        frames_at_close = fake.frames_out[0]
        time.sleep(0.1)
        assert fake.frames_out[0] == frames_at_close  # nothing produced after the close

        with client.websocket_connect("/voxtream") as ws:
            ws.send_text(_init(prompt_wav))
            ws.send_text(json.dumps({"event": "text", "chunk": "again"}))
            ws.send_text(json.dumps({"event": "eot"}))
            texts, blobs = _recv_all(ws)
    assert texts[-1] == {"type": "eos"}
    assert len(blobs) == 2
    assert len(fake.starts) == 2


def test_a_client_gone_before_its_config_frame_ends_the_handler_quietly(prompt_wav):
    # A connection parked behind the generation lock whose client left meanwhile:
    # under uvicorn the config send raises WebSocketDisconnect. The handler must
    # swallow it and never start a generation (TestClient's queue accepts a send to
    # a closed client, so this is pinned on a stub websocket).
    from starlette.websockets import WebSocketDisconnect

    class _GoneWS:
        async def send_text(self, _data):
            raise WebSocketDisconnect(code=1006)

        async def send_bytes(self, _data):
            raise WebSocketDisconnect(code=1006)

        async def receive(self):
            return {"type": "websocket.disconnect"}

        async def close(self, code=1000):
            pass

    fake = FakeGenerator()
    asyncio.run(voxtream_server._serve_generation(_GoneWS(), fake, prompt_wav, "", True))
    assert fake.starts == []


def test_health_reports_loading_then_ready():
    app = create_app(generator=None, config_dir=Path("/nonexistent"))
    # No lifespan: the loader task is not started, so the state is observable.
    client = TestClient(app)
    assert client.get("/health").status_code == 503
    app.state.generator = FakeGenerator()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["sample_rate"] == 24000


def test_websocket_before_ready_is_refused(prompt_wav):
    app = create_app(generator=None, config_dir=Path("/nonexistent"))
    app.state.generation_lock = None
    with TestClient(app).websocket_connect("/voxtream") as ws:
        texts, _ = _recv_all(ws)
    assert texts[0]["type"] == "error"


def test_default_bind_is_loopback_and_configs_ship():
    args = voxtream_server.build_parser().parse_args([])
    assert args.host == "127.0.0.1"
    assert (args.config_dir / "generator.json").is_file()
    assert (args.config_dir / "speaking_rate.json").is_file()
    assert json.load(open(args.config_dir / "generator.json"))["mimi_sr"] == 24000
