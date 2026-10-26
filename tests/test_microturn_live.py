"""Tests for the micro-turn client's live mode: tick log, first-audio delay, player, WAV.

Model-free, server-free and device-free. A fake engine stands in for llama-server, a
fake client for the VoiceManager WebSocket, and the player is driven by ``pull`` alone,
so no audio stream is ever opened.
"""
import asyncio
import importlib.util
import io
import json
import math
import pathlib
import sys
import wave

import pytest

_PATH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "microturn_client.py"
if "microturn_client" in sys.modules:
    mc = sys.modules["microturn_client"]
else:
    _spec = importlib.util.spec_from_file_location("microturn_client", _PATH)
    mc = importlib.util.module_from_spec(_spec)
    # dataclasses look the module up in sys.modules while the class body runs.
    sys.modules["microturn_client"] = mc
    _spec.loader.exec_module(mc)

LISTEN = "<listen>"
NOVOICE = "<no voice>"
TICK_FIELDS = {"type", "t", "heard", "said", "p_listen", "turn_ms", "requests", "revised",
               "revisions"}


def _response(content: str, p_listen: float) -> dict:
    """A llama-server /completion JSON whose first token puts p_listen on "<"."""
    return {
        "content": content,
        "completion_probabilities": [{
            "top_logprobs": [
                {"id": 1, "token": "<", "logprob": math.log(p_listen)},
                {"id": 2, "token": "Do", "logprob": math.log(1.0 - p_listen)},
            ],
        }],
    }


class ScriptedEngine:
    """Answers each tick with the next scripted reply."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)

    def complete(self, body: dict) -> dict:
        said = self.replies.pop(0)
        if said == LISTEN:
            return _response(LISTEN, 0.9)
        return _response(said, 0.1)


class FakeSink:
    """Records every TTS call as a tuple, in order."""

    def __init__(self):
        self.calls: list[tuple] = []

    async def send_text(self, request_id: str, text: str) -> None:
        self.calls.append(("text", request_id, text))

    async def flush(self, request_id: str) -> None:
        self.calls.append(("flush", request_id))

    async def done(self, request_id: str) -> None:
        self.calls.append(("done", request_id))

    async def cancel(self, request_id: str) -> None:
        self.calls.append(("cancel", request_id))


class FakePlayer:
    """A player that is never playing."""

    def is_playing(self) -> bool:
        return False

    def clear(self) -> None:
        pass


def _session(replies: list[str], cancelled: set[str] | None = None):
    """A LiveSession over fakes, its log and its orchestrator; request ids are r1, r2..."""
    ids = iter(f"r{i}" for i in range(1, 100))
    orch = mc.TTSOrchestrator(FakeSink(), FakePlayer(), new_id=lambda: next(ids))
    log = io.StringIO()
    cancelled = cancelled if cancelled is not None else set()
    session = mc.LiveSession(ScriptedEngine(replies), mc.TickSlicer(), orch, log,
                             clock=lambda: 100.0, cancelled=lambda rid: rid in cancelled)
    return session, log, orch


def _lines(log: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in log.getvalue().splitlines()]


def _frame(text: str, seg: int = 0, final: bool = False) -> dict:
    return {"type": "transcription", "text": text, "segment_index": seg, "is_final": final}


def test_live_tick_jsonl_fields():
    session, log, _ = _session([LISTEN, "Do not", "mix those", LISTEN])

    async def run():
        session.on_transcription(_frame("Hey I'm"), 100.2)
        await session.run_tick(100.6)
        await session.run_tick(101.2)
        session.on_transcription(_frame("Hey I am cleaning"), 101.5)
        await session.run_tick(101.8)
        await session.run_tick(102.4)

    asyncio.run(run())
    lines = _lines(log)
    assert [line["type"] for line in lines] == ["tick"] * 4
    for line in lines:
        assert TICK_FIELDS <= set(line)
    assert [line["said"] for line in lines] == [LISTEN, "Do not", "mix those", LISTEN]
    assert [line["heard"] for line in lines] == ["Hey I'm", NOVOICE, "am cleaning", NOVOICE]
    assert lines[0]["t"] == pytest.approx(0.6)
    assert lines[2]["revised"] is True and lines[2]["revisions"] == 1
    assert all(line["requests"] == 1 for line in lines)


def test_live_utterance_first_audio_ms():
    cancelled: set[str] = set()
    session, log, orch = _session([LISTEN, "Do not", LISTEN, "Stop"], cancelled)

    async def run():
        session.on_transcription(_frame("bleach and"), 100.3)
        await session.run_tick(100.6)
        await session.run_tick(101.2)  # opens r1; last word heard at 100.3

    asyncio.run(run())
    assert orch.request_id == "r1"
    session.on_audio_start("r1", 24000, 101.5)
    assert session.on_audio(b"\x01\x00" * 10, 101.7) is True
    assert session.on_audio(b"\x02\x00" * 10, 101.9) is True
    utterances = [line for line in _lines(log) if line["type"] == "utterance"]
    assert len(utterances) == 1
    assert utterances[0]["request_id"] == "r1"
    assert utterances[0]["first_audio_ms"] == pytest.approx((101.7 - 100.3) * 1000)
    assert utterances[0]["cancelled"] is False
    assert session.first_audio_ms == [pytest.approx(1400.0)]

    session.on_audio_end("r1")

    async def run2():
        await session.run_tick(102.4)  # listen: done for r1
        await session.run_tick(103.0)  # opens r2

    asyncio.run(run2())
    cancelled.add("r2")
    session.on_audio_start("r2", 24000, 103.2)
    assert session.on_audio(b"\x03\x00" * 10, 103.3) is False
    utterances = [line for line in _lines(log) if line["type"] == "utterance"]
    assert len(utterances) == 1


def test_live_utterance_null_without_heard_word():
    session, log, _ = _session(["Hello there"])
    asyncio.run(session.run_tick(100.6))
    session.on_audio_start("r1", 24000, 100.8)
    session.on_audio(b"\x00\x00", 100.9)
    utterances = [line for line in _lines(log) if line["type"] == "utterance"]
    assert utterances[0]["first_audio_ms"] is None


def test_live_player_buffer():
    player = mc.AudioPlayer()
    assert player.is_playing() is False
    player.enqueue(b"\x01\x02\x03\x04")
    player.enqueue(b"\x05\x06")
    assert player.is_playing() is True
    assert player.pull(4) == b"\x01\x02\x03\x04"
    assert player.pull(6) == b"\x05\x06\x00\x00\x00\x00"
    assert player.is_playing() is False
    assert player.pull(2) == b"\x00\x00"
    player.enqueue(b"\x07\x08\x09\x0a")
    player.clear()
    assert player.is_playing() is False
    assert player.pull(2) == b"\x00\x00"
    assert bytes(player.played) == b"\x01\x02\x03\x04\x05\x06"


def test_live_wav_valid(tmp_path):
    path = tmp_path / "out.wav"
    pcm = b"\x01\x00\x02\x00\x03\x00"
    mc.write_wav(path, pcm, 24000)
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24000
        assert wf.getnframes() == 3
        assert wf.readframes(3) == pcm


def test_live_summary():
    session, _, orch = _session([])
    session.results = [
        mc.TickResult(said=LISTEN, p_listen=0.9, turn_ms=t, requests=1)
        for t in (100.0, 300.0, 700.0, 200.0)
    ]
    session.first_audio_ms = [900.0, 500.0, 1300.0]
    session.slicer.revisions = 2
    orch.cutins = 1
    line = session.summary_line()
    assert "4 ticks" in line
    assert "median turn 250 ms" in line
    assert "median first audio 900 ms" in line
    assert "1 over budget" in line
    assert "2 revisions" in line
    assert "1 cut-ins" in line


def test_live_summary_without_audio():
    session, _, _ = _session([])
    line = session.summary_line()
    assert "0 ticks" in line
    assert "median first audio n/a" in line


class FakeWs:
    def __init__(self):
        self.sent: list[str] = []

    async def send(self, msg: str) -> None:
        self.sent.append(msg)


class FakeClient:
    """Records raw WebSocket sends and the public flush/cancel calls."""

    def __init__(self):
        self._ws = FakeWs()
        self.calls: list[tuple] = []

    async def flush_tts(self, request_id: str) -> None:
        self.calls.append(("flush", request_id))

    async def cancel_tts(self, request_id: str) -> None:
        self.calls.append(("cancel", request_id))


def test_live_sink_frames():
    client = FakeClient()
    sink = mc.WsTTSSink(client, voice="alice")

    async def run():
        await sink.send_text("r1", "Do not")
        await sink.flush("r1")
        await sink.send_text("r1", " mix")
        await sink.done("r1")
        await sink.send_text("r2", "Stop")
        await sink.cancel("r2")

    asyncio.run(run())
    frames = [json.loads(m) for m in client._ws.sent]
    assert frames == [
        {"type": "tts_stream", "request_id": "r1", "text": "Do not", "done": False,
         "voice": "alice"},
        {"type": "tts_stream", "request_id": "r1", "text": " mix", "done": False},
        {"type": "tts_stream", "request_id": "r1", "text": "", "done": True},
        {"type": "tts_stream", "request_id": "r2", "text": "Stop", "done": False,
         "voice": "alice"},
    ]
    assert client.calls == [("flush", "r1"), ("cancel", "r2")]


def test_live_sink_without_voice():
    client = FakeClient()
    sink = mc.WsTTSSink(client)
    asyncio.run(sink.send_text("r1", "Hi"))
    assert "voice" not in json.loads(client._ws.sent[0])


def test_live_cli_options():
    args = mc._parser().parse_args(["live", "--duration", "5", "--voice", "alice"])
    assert args.mode == "live"
    assert args.vm_host == "localhost" and args.vm_port == 8255
    assert args.duration == 5.0 and args.voice == "alice"
    assert args.log is None and args.wav is None
    assert args.url == "http://127.0.0.1:8080"


def test_live_no_sounddevice_at_import():
    """The script must not import sounddevice at module scope.

    The functions that need it import it as ``sd``, so a module-scope
    ``import sounddevice as sd`` binds the name ``sd``, not ``sounddevice``: a check of
    the name alone would still pass. So every attribute of the module object is
    compared with the sounddevice module itself, under any name. When no earlier test
    has loaded sounddevice, a fresh load of the script must also leave it out of
    sys.modules.
    """
    loaded = sys.modules.get("sounddevice")
    for name, value in vars(mc).items():
        assert getattr(value, "__name__", None) != "sounddevice", name
        if loaded is not None:
            assert value is not loaded, name
    if loaded is None:
        spec = importlib.util.spec_from_file_location("microturn_client_fresh", _PATH)
        fresh = importlib.util.module_from_spec(spec)
        sys.modules["microturn_client_fresh"] = fresh
        try:
            spec.loader.exec_module(fresh)
        finally:
            del sys.modules["microturn_client_fresh"]
        assert "sounddevice" not in sys.modules


def test_live_engine_check_refuses_whisper():
    with pytest.raises(SystemExit) as exc:
        mc.check_stt_engine("whisper")
    assert "whisper" in str(exc.value)


def test_live_engine_check_allows_streaming_engines():
    for name in ("audiocpp", "qwen", "nemotron"):
        mc.check_stt_engine(name)


def test_live_default_paths_are_in_run_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = mc._parser().parse_args(["live"])
    log_path, wav_path = mc.live_output_paths(args)
    run_dir = tmp_path / "runs" / "microturn"
    assert run_dir.is_dir()
    assert pathlib.Path(log_path).parent == pathlib.Path("runs/microturn")
    assert pathlib.Path(wav_path).parent == pathlib.Path("runs/microturn")
    assert str(log_path).endswith(".jsonl") and str(wav_path).endswith(".wav")


def test_live_explicit_paths_override_run_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = mc._parser().parse_args(["live", "--log", "a.jsonl", "--wav", "b.wav"])
    assert mc.live_output_paths(args) == ("a.jsonl", "b.wav")
    assert not (tmp_path / "runs").exists()


class ChunkingSink(FakeSink):
    """A FakeSink whose text calls also take the chunking value."""

    async def send_text(self, request_id: str, text: str, chunking: str | None = None) -> None:
        self.calls.append(("text", request_id, text, chunking))


def _mode_session(mode: str, replies: list[str]):
    ids = iter(f"r{i}" for i in range(1, 100))
    orch = mc.TTSOrchestrator(ChunkingSink(), FakePlayer(), new_id=lambda: next(ids),
                              mode=mode)
    log = io.StringIO()
    session = mc.LiveSession(ScriptedEngine(replies), mc.TickSlicer(), orch, log,
                             clock=lambda: 100.0)
    return session, log


def test_live_tick_log_names_tts_mode():
    for mode in ("clause", "flush"):
        session, log = _mode_session(mode, ["Do not", LISTEN])

        async def run():
            await session.run_tick(100.6)
            await session.run_tick(101.2)

        asyncio.run(run())
        assert [line["tts_mode"] for line in _lines(log)] == [mode, mode]


def test_live_summary_names_tts_mode():
    for mode in ("clause", "flush"):
        session, _ = _mode_session(mode, [])
        assert f"tts mode {mode}" in session.summary_line()


def test_live_sink_clause_chunking_on_first_frame():
    client = FakeClient()
    sink = mc.WsTTSSink(client, voice="alice")

    async def run():
        await sink.send_text("r1", "Do not", chunking="clause")
        await sink.send_text("r1", " mix")
        await sink.done("r1")

    asyncio.run(run())
    frames = [json.loads(m) for m in client._ws.sent]
    assert frames == [
        {"type": "tts_stream", "request_id": "r1", "text": "Do not", "done": False,
         "voice": "alice", "chunking": "clause"},
        {"type": "tts_stream", "request_id": "r1", "text": " mix", "done": False},
        {"type": "tts_stream", "request_id": "r1", "text": "", "done": True},
    ]
    assert client.calls == []


def test_live_sink_chunking_only_on_first_frame():
    client = FakeClient()
    sink = mc.WsTTSSink(client)

    async def run():
        await sink.send_text("r1", "Do not")
        await sink.send_text("r1", " mix", chunking="clause")

    asyncio.run(run())
    assert all("chunking" not in json.loads(m) for m in client._ws.sent)


def test_live_cli_tts_mode():
    assert mc._parser().parse_args(["live"]).tts_mode == "clause"
    args = mc._parser().parse_args(["live", "--tts-mode", "flush"])
    assert args.tts_mode == "flush"
    with pytest.raises(SystemExit):
        mc._parser().parse_args(["live", "--tts-mode", "word"])


def test_live_last_word_time_ignores_recase_and_punctuation():
    session, _, _ = _session([])
    session.on_transcription(_frame("bleach and ammonia"), 100.3)
    assert session.last_word_t == 100.3
    session.on_transcription(_frame("Bleach, and ammonia.", final=True), 100.9)
    assert session.last_word_t == 100.3
    session.on_transcription(_frame("Bleach and ammonia"), 101.1)
    assert session.last_word_t == 100.3


def test_live_last_word_time_moves_when_words_are_added():
    session, _, _ = _session([])
    session.on_transcription(_frame("bleach and"), 100.3)
    session.on_transcription(_frame("Bleach, and ammonia"), 101.0)
    assert session.last_word_t == 101.0
    session.on_transcription(_frame("here", seg=1), 101.4)
    assert session.last_word_t == 101.4


class TogglePlayer:
    """A player whose playing state the test sets."""

    def __init__(self):
        self.playing = False
        self.cleared = 0

    def is_playing(self) -> bool:
        return self.playing

    def clear(self) -> None:
        self.playing = False
        self.cleared += 1


def test_live_utterance_cancelled_by_cut_in():
    ids = iter(f"r{i}" for i in range(1, 100))
    player = TogglePlayer()
    sink = FakeSink()
    orch = mc.TTSOrchestrator(sink, player, new_id=lambda: next(ids))
    log = io.StringIO()
    session = mc.LiveSession(ScriptedEngine(["Do not", LISTEN, LISTEN, "Stop", LISTEN]),
                             mc.TickSlicer(), orch, log, clock=lambda: 100.0)

    async def run():
        session.on_transcription(_frame("bleach"), 100.2)
        await session.run_tick(100.6)  # opens r1
        session.on_audio_start("r1", 24000, 100.8)
        assert session.on_audio(b"\x01\x00", 100.9) is True
        player.playing = True
        session.on_transcription(_frame("wait no", seg=1), 101.0)
        await session.run_tick(101.2)  # user talks over r1: cut-in cancels it
        session.on_audio_end("r1")
        await session.run_tick(101.8)
        await session.run_tick(102.4)  # opens r2
        session.on_audio_start("r2", 24000, 102.6)
        assert session.on_audio(b"\x02\x00", 102.7) is True
        session.on_audio_end("r2")
        await session.run_tick(103.0)  # done for r2

    asyncio.run(run())
    assert ("cancel", "r1") in sink.calls and orch.cutins == 1
    utterances = [line for line in _lines(log) if line["type"] == "utterance"]
    final = {line["request_id"]: line["cancelled"] for line in utterances}
    assert final == {"r1": True, "r2": False}
    cut = [line for line in utterances if line["cancelled"]]
    assert cut[0]["request_id"] == "r1" and cut[0]["t"] == pytest.approx(1.2)
    assert cut[0]["first_audio_ms"] == pytest.approx((100.9 - 100.2) * 1000)


class FakeLiveClient:
    """Stands in for VoiceManagerClient in live(): connects, reports whisper, records."""

    instances: list = []

    def __init__(self, host, port, token):
        self.host, self.port, self.token = host, port, token
        self.stt_engine = "whisper"
        self.last_tts_sample_rate = 22050
        self._cancelled_tts: set[str] = set()
        self.connected = False
        self.disconnected = False
        FakeLiveClient.instances.append(self)

    async def _connect(self) -> None:
        self.connected = True

    async def _disconnect(self) -> None:
        self.disconnected = True


def test_live_checks_stt_engine_before_starting_tasks(tmp_path, monkeypatch):
    """live() must refuse a whisper server after connecting and before any task runs."""
    import types

    import voicemanager.client as vmclient

    started: list[str] = []

    def task_stub(name):
        async def stub(*args):
            started.append(name)
        return stub

    for name in ("_mic_task", "_stt_task", "_tts_task", "_tick_task"):
        monkeypatch.setattr(mc, name, task_stub(name))

    def no_http(*args, **kwargs):
        raise AssertionError("HTTP request made")

    monkeypatch.setattr(mc.urllib.request, "urlopen", no_http)

    def no_stream(*args, **kwargs):
        raise AssertionError("audio stream opened")

    fake_sd = types.SimpleNamespace(RawInputStream=no_stream, RawOutputStream=no_stream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    monkeypatch.setattr(vmclient, "VoiceManagerClient", FakeLiveClient)
    monkeypatch.setenv("VOICEMANAGER_TOKEN", "vmp_test")
    FakeLiveClient.instances.clear()
    log_path, wav_path = tmp_path / "call.jsonl", tmp_path / "call.wav"
    args = mc._parser().parse_args(["live", "--duration", "0.1", "--log", str(log_path),
                                   "--wav", str(wav_path)])

    with pytest.raises(SystemExit) as exc:
        asyncio.run(mc.live(args))

    assert "whisper" in str(exc.value)
    assert len(FakeLiveClient.instances) == 1
    client = FakeLiveClient.instances[0]
    assert client.connected and client.disconnected
    assert started == []
    assert not log_path.exists() and not wav_path.exists()
