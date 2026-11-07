"""The STT fence at the listening boundary, over the real WebSocket handler, model-free.

No transcription frame, partial or final, made from audio fed before a listen_stop
reaches the client after its listen_stopped. That holds for every STT engine path,
also for a decode that is still running when the listen_stop comes in and when
listen_start and new audio follow right behind it (the client does not wait for
listen_stopped). Transcription of the audio after listen_start is delivered as usual.

Each engine path gets a fake that can hold a decode in progress on a
threading.Event, so the cross-boundary case is deterministic:

* whisper - the real STTPipeline (its buffer and silero VADIterator) with an
  amplitude stand-in for the VAD net and a whisper stand-in that names the audio it
  was given;
* qwen - the real QwenSTTPipeline over a streaming stand-in that names the audio
  it was fed;
* nemotron - the scripted NemotronSTTPipeline of the pipeline tests, with the
  speculative endpoint hold on;
* audio.cpp - the real AudioCppSTTPipeline over the fake sidecar (late deltas, the
  quiet flush), and a flush-only stand-in under a turn engine whose turn-end flush is
  still running at the boundary.

Speech frames carry a value that names them ("pre" before the stop, "post" after
it), so a leak shows as "pre" in a frame the client reads after listen_stopped.
No GPU, no model, no live server, no audio hardware.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)
from fastapi.testclient import TestClient

import voicemanager.config as cfg
from voicemanager.app import create_app
from voicemanager.pipeline import audiocpp_stt
from voicemanager.pipeline.tyger import wire
from voicemanager.session import SessionRegistry
from voicemanager.transport.endpoint_hold import EndpointHoldMachine

from audiocpp_fake import FakeSidecar
from test_nemotron_pipeline import (
    STATEFUL_POST, STATEFUL_PRE, StatefulNemotron, StatefulShared,
    _FakeShared, _ScriptedNemotron, _ONE_CHUNK,
)
from tyger_fake_sidecar import FakeTygerSidecar
from tyger_ws_kit import StandInSidecar, barrier, frame, is_type, make_app, opt_in, recv_until

STOP = {"type": "listen_stop"}
START = {"type": "listen_start"}

# One silero-sized chunk (512 samples, 32 ms) per frame. The sample value names the
# audio; zeros are silence.
_PRE_VALUE = 0x1111
_POST_VALUE = 0x2222
_PRE = _PRE_VALUE.to_bytes(2, "little") * 512
_POST = _POST_VALUE.to_bytes(2, "little") * 512
_SILENCE = b"\x00\x00" * 512
_LIMIT = 600
_GATE_S = 5.0


def _labels(audio_f32: np.ndarray) -> list[str]:
    """The names of the speech in float32 audio: "pre", "post", both or neither."""
    found = []
    for name, value in (("pre", _PRE_VALUE), ("post", _POST_VALUE)):
        if np.any(np.abs(audio_f32 - value / 32768.0) < 1e-6):
            found.append(name)
    return found


def _label_of(data: bytes) -> str | None:
    """The name of one PCM frame, or None for silence."""
    if data[:2] == _PRE[:2]:
        return "pre"
    if data[:2] == _POST[:2]:
        return "post"
    return None


class _AmplitudeVad:
    """The silero net's place under the real VADIterator: a loud chunk is speech."""

    def __call__(self, x, sr):
        return torch.tensor(1.0 if float(x.abs().max()) > 0.01 else 0.0)

    def reset_states(self):
        pass


def _transcriptions(frames: list) -> list[dict]:
    return [f for f in frames if is_type("transcription")(f)]


def _after_stopped(frames: list) -> list[dict]:
    """The transcription frames that came after the listen_stopped frame."""
    kinds = [f.get("type") if isinstance(f, dict) else None for f in frames]
    assert "listen_stopped" in kinds, frames
    return _transcriptions(frames[kinds.index("listen_stopped") + 1:])


def _assert_nothing_pre_after_stopped(frames: list) -> list[dict]:
    after = _after_stopped(frames)
    leaked = [f for f in after if "pre" in f["text"]]
    assert leaked == [], f"pre-stop text after listen_stopped: {leaked}"
    return after


def _is_final(f) -> bool:
    return is_type("transcription")(f) and f["is_final"]


def _to_first_final_after_stop(ws) -> list:
    """Every frame up to listen_stopped, then on to the first final after it.

    The test then checks what came: waiting for one exact text would hang on a leak.
    """
    frames = recv_until(ws, is_type("listen_stopped"), limit=_LIMIT)
    return frames + recv_until(ws, _is_final, limit=_LIMIT)


@pytest.fixture
def settings(monkeypatch):
    """A long session timeout; a test sets its engine's own windows."""
    monkeypatch.setenv("VM_SESSION_TIMEOUT", "60")
    cfg._settings = None
    yield monkeypatch
    cfg._settings = None


def _set(monkeypatch, **env) -> None:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    cfg._settings = None


# --- whisper ----------------------------------------------------------------------

class _NamingWhisper:
    """A faster-whisper stand-in: one segment naming the audio it was given.

    ``hold`` makes the first transcribe wait on ``gate``; ``started`` is set when it
    is running.
    """

    def __init__(self, hold: bool = False) -> None:
        self.gate = threading.Event()
        self.started = threading.Event()
        self._hold = hold

    def transcribe(self, audio_np):
        self.started.set()
        if self._hold:
            self._hold = False
            self.gate.wait(_GATE_S)
        return iter([_Seg(" ".join(_labels(audio_np)))]), _Info()


class _Seg:
    def __init__(self, text: str) -> None:
        self.text = text
        self.avg_logprob = -0.1
        self.start = 0.0
        self.end = 1.0


class _Info:
    language = "en"


@pytest.fixture
def whisper_client(settings):
    _set(settings, VM_VAD_SILENCE_MS="100", VM_VAD_THRESHOLD="0.5")

    def make(model: _NamingWhisper) -> TestClient:
        app = create_app()
        app.state.stt_engine_type = "whisper"
        app.state.vad_model = _AmplitudeVad()
        app.state.whisper_model = model
        app.state.executor = pool
        app.state.session_registry = SessionRegistry()
        return TestClient(app)

    with ThreadPoolExecutor(max_workers=1) as pool:
        yield make


def _utterance(ws, data: bytes, *, close: bool = True) -> None:
    for _ in range(4):
        ws.send_bytes(data)
    if close:
        for _ in range(10):
            ws.send_bytes(_SILENCE)


def test_whisper_open_utterance_does_not_continue_after_listen_start(whisper_client):
    """Speech with no endpoint yet, then listen_stop, listen_start and new speech at
    once: the next final holds only the new speech."""
    client = whisper_client(_NamingWhisper())
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        _utterance(ws, _PRE, close=False)
        ws.send_json(STOP)
        ws.send_json(START)
        _utterance(ws, _POST)
        frames = _to_first_final_after_stop(ws)
    after = _assert_nothing_pre_after_stopped(frames)
    assert [f["text"] for f in after if f["is_final"]] == ["post"]


def test_whisper_final_in_progress_is_not_delivered_after_listen_stopped(whisper_client):
    model = _NamingWhisper(hold=True)
    client = whisper_client(model)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        _utterance(ws, _PRE)
        assert model.started.wait(_GATE_S)    # the pre-stop decode is running
        ws.send_json(STOP)
        ws.send_json(START)
        _utterance(ws, _POST)
        model.gate.set()
        frames = _to_first_final_after_stop(ws)
    after = _assert_nothing_pre_after_stopped(frames)
    assert [f["text"] for f in after if f["is_final"]] == ["post"]


# --- qwen -------------------------------------------------------------------------

class _QwenState:
    def __init__(self) -> None:
        self.text = ""
        self.language = "en"


class _NamingQwen:
    """A qwen-asr stand-in: each speech frame adds the name of its audio to the text.

    ``hold`` is "feed" or "finish": that call waits on ``gate`` the first time;
    ``started`` is set when it is running.
    """

    def __init__(self, hold: str | None = None) -> None:
        self.gate = threading.Event()
        self.started = threading.Event()
        self._hold = hold

    def _maybe_hold(self, where: str) -> None:
        if self._hold == where:
            self._hold = None
            self.started.set()
            self.gate.wait(_GATE_S)

    def init_streaming_state(self, **kwargs):
        return _QwenState()

    def streaming_transcribe(self, audio_f32, state) -> None:
        names = _labels(audio_f32)
        if names:
            self._maybe_hold("feed")
            state.text = " ".join([state.text, *names]).strip()

    def finish_streaming_transcribe(self, state) -> None:
        self._maybe_hold("finish")


@pytest.fixture
def qwen_client(settings):
    _set(settings, VM_QWEN_SILENCE_MS="100")

    def make(model: _NamingQwen) -> TestClient:
        app = create_app()
        app.state.stt_engine_type = "qwen"
        app.state.qwen_model = model
        app.state.session_registry = SessionRegistry()
        return TestClient(app)

    return make


def _qwen_final(ws) -> None:
    """Let the qwen silence window pass, then a silent frame asks for the final."""
    time.sleep(0.25)
    ws.send_bytes(_SILENCE)


def test_qwen_partial_in_progress_and_open_utterance_do_not_leak(qwen_client):
    """A feed still decoding pre-stop speech at listen_stop: its partial may only come
    before listen_stopped, and the stream after listen_start starts fresh."""
    model = _NamingQwen(hold="feed")
    client = qwen_client(model)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(_PRE)
        assert model.started.wait(_GATE_S)
        ws.send_json(STOP)
        ws.send_json(START)
        ws.send_bytes(_POST)
        model.gate.set()
        frames = recv_until(ws, is_type("listen_stopped"), limit=_LIMIT)
        frames += recv_until(ws, is_type("transcription"), limit=_LIMIT)
        _qwen_final(ws)
        frames += recv_until(ws, _is_final, limit=_LIMIT)
    after = _assert_nothing_pre_after_stopped(frames)
    assert [f["text"] for f in after] == ["post", "post"]


def test_qwen_final_in_progress_is_not_delivered_after_listen_stopped(qwen_client):
    model = _NamingQwen(hold="finish")
    client = qwen_client(model)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(_PRE)
        assert ws.receive_json()["text"] == "pre"
        _qwen_final(ws)
        assert model.started.wait(_GATE_S)    # the pre-stop final is decoding
        ws.send_json(STOP)
        ws.send_json(START)
        ws.send_bytes(_POST)
        model.gate.set()
        frames = recv_until(ws, is_type("listen_stopped"), limit=_LIMIT)
        frames += recv_until(ws, is_type("transcription"), limit=_LIMIT)
        _qwen_final(ws)
        frames += recv_until(ws, _is_final, limit=_LIMIT)
    after = _assert_nothing_pre_after_stopped(frames)
    assert [f["text"] for f in after if f["is_final"]] == ["post"]


# --- nemotron ---------------------------------------------------------------------

class _HeldNemotron(_ScriptedNemotron):
    """The scripted nemotron pipeline; ``hold`` makes the first step wait on ``gate``."""

    def __init__(self, script, hold: bool = False, **kw):
        self.gate = threading.Event()
        self.started = threading.Event()
        self._hold = hold
        super().__init__(script, **kw)

    def _transcribe(self, audio_f32):
        if self._hold:
            self._hold = False
            self.started.set()
            self.gate.wait(_GATE_S)
        return super()._transcribe(audio_f32)


@pytest.fixture
def nemotron_client(settings):
    """Hold window 300 ms = 4800 samples, under 19 chunks of 256 samples."""
    _set(settings, VM_STT_ENDPOINT_HOLD_MS="300")
    made: list[_HeldNemotron] = []

    def make(script, hold: bool = False) -> TestClient:
        def factory(**kwargs):
            made.append(_HeldNemotron(list(script), hold=hold, **kwargs))
            return made[-1]

        settings.setattr("voicemanager.pipeline.nemotron_stt.NemotronSTTPipeline", factory)
        app = create_app()
        app.state.stt_engine_type = "nemotron"
        app.state.nemotron_pipeline = _FakeShared()
        app.state.session_registry = SessionRegistry()
        return TestClient(app)

    make.made = made
    return make


# Utterance one ends at once ("pre", held); utterance two is "post", then silence.
_NEMOTRON_SCRIPT = [("pre", "pre"), ("post", None), ("post", "post")]


def _nemotron_post(ws) -> list:
    """Speech after listen_start, then enough silence to confirm its held final."""
    for _ in range(2):
        ws.send_bytes(_ONE_CHUNK)
    for _ in range(25):
        ws.send_bytes(_ONE_CHUNK)
    return recv_until(ws, _is_final, limit=_LIMIT)


def test_nemotron_held_final_and_decode_in_progress_do_not_leak(nemotron_client):
    """A pre-stop endpoint is held when listen_stop comes in while its step is still
    decoding; listen_start and speech follow at once. The held final is never
    confirmed, nothing stitches onto the new speech, and the new utterance is
    delivered."""
    client = nemotron_client(_NEMOTRON_SCRIPT, hold=True)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(_ONE_CHUNK)
        assert nemotron_client.made[0].started.wait(_GATE_S)
        ws.send_json(STOP)
        ws.send_json(START)
        nemotron_client.made[0].gate.set()
        frames = recv_until(ws, is_type("listen_stopped"), limit=_LIMIT)
        frames += _nemotron_post(ws)
    before = _transcriptions(frames[:[f.get("type") for f in frames].index("listen_stopped")])
    assert [(f["text"], f["endpoint"]) for f in before] == [("pre", True)]
    after = _assert_nothing_pre_after_stopped(frames)
    assert [f["text"] for f in after if f["is_final"]] == ["post"]


def test_nemotron_hold_does_not_confirm_in_a_stopped_gap(nemotron_client):
    """Stopped longer than the hold window: its wall-clock fallback sends nothing."""
    client = nemotron_client(_NEMOTRON_SCRIPT)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(_ONE_CHUNK)
        prov = ws.receive_json()
        assert prov["endpoint"] is True and prov["text"] == "pre"
        ws.send_json(STOP)
        frames = recv_until(ws, is_type("listen_stopped"), limit=_LIMIT)
        time.sleep(0.7)                        # over twice the 300 ms hold window
        frames += barrier(ws)
        assert _after_stopped(frames) == []
        ws.send_json(START)
        frames += _nemotron_post(ws)
    after = _assert_nothing_pre_after_stopped(frames)
    assert [f["text"] for f in after if f["is_final"]] == ["post"]


def test_nemotron_open_utterance_does_not_carry_into_the_first_partial_after_start(
        settings):
    """NeMo keeps a running hypothesis per stream: an utterance still open at the stop
    would show up in the first partial after listen_start unless the stream is
    replaced. Here the stand-in keeps that state from one step to the next."""
    _set(settings, VM_STT_ENDPOINT_HOLD_MS="300")
    shared = StatefulShared()
    settings.setattr("voicemanager.pipeline.nemotron_stt.NemotronSTTPipeline",
                     lambda **kwargs: StatefulNemotron(**kwargs))
    app = create_app()
    app.state.stt_engine_type = "nemotron"
    app.state.nemotron_pipeline = shared
    app.state.session_registry = SessionRegistry()
    client = TestClient(app)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_bytes(STATEFUL_PRE)
        assert ws.receive_json()["text"] == "pre"
        ws.send_bytes(STATEFUL_PRE[:100])     # a sub-chunk tail, still buffered
        ws.send_json(STOP)
        ws.send_json(START)
        ws.send_bytes(STATEFUL_POST)
        frames = recv_until(ws, is_type("listen_stopped"), limit=_LIMIT)
        frames += barrier(ws)
    after = _assert_nothing_pre_after_stopped(frames)
    assert [(f["text"], f["is_final"]) for f in after] == [("post", False)]
    assert len(shared.inits) == 2               # a fresh stream after the stop


def test_endpoint_hold_clear_drops_the_hold_and_the_stitch():
    machine = EndpointHoldMachine(100)
    machine.begin_hold("one", "en", 0, turn_id=3)
    machine.reopen()                           # "one" becomes a stitch prefix
    machine.begin_hold("two", "en", 0, turn_id=4)
    machine.reopen()
    assert machine.stitch_prefix == "two"
    machine.begin_hold("two three", "en", 0, turn_id=5)
    assert machine.is_holding

    machine.clear()

    assert not machine.is_holding
    assert machine.stitch_prefix == "" and machine.stitch_turn == 0
    assert machine.stitch("four") == "four"
    assert machine.advance(10_000) is None
    assert machine.take_held() is None
    # The segment mapping stays monotonic: emitted indexes are never reused.
    assert machine.remap(5) == 3


# --- audio.cpp over the fake sidecar -------------------------------------------------

def _audiocpp_client(fake: FakeSidecar) -> TestClient:
    app = create_app()
    app.state.stt_engine_type = "audiocpp"
    app.state.audiocpp_config = {
        "sidecar_bin": "", "socket": fake.socket_path, "model": "",
        "model_task": "nemotron-3.5-asr-streaming", "backend": "cpu", "device": 0,
        "threads": 4, "assets_root": "",
    }
    app.state.session_registry = SessionRegistry()
    return TestClient(app)


@pytest.fixture
def audiocpp_quiet(settings):
    """A 200 ms quiet flush (VM_AUDIOCPP_SILENCE_MS) and no silero: the frame gap is
    the only endpoint."""
    _set(settings, VM_AUDIOCPP_SILENCE_MS="200")


def _wait_for(predicate, timeout: float = _GATE_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_audiocpp_late_sidecar_reply_is_not_delivered_after_listen_stopped(audiocpp_quiet):
    """The sidecar is still decoding the pre-stop frame at listen_stop, and
    listen_start and new audio follow at once. Its late delta and the final of that
    utterance never reach the client; the new utterance does."""
    gate = threading.Event()
    script = [([], "pre final"), ([], "post final")]
    with FakeSidecar(audio_delta_script=["pre ", "post "], flush_script=script,
                     audio_reply_gate=gate) as fake:
        client = _audiocpp_client(fake)
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(_PRE)
            assert _wait_for(lambda: fake.connections and fake.connections[0].audio_frames == 1)
            ws.send_json(STOP)
            ws.send_json(START)
            ws.send_bytes(_POST)
            gate.set()
            frames = recv_until(ws, is_type("listen_stopped"), limit=_LIMIT)
            frames += recv_until(ws, _is_final, limit=_LIMIT)
    after = _assert_nothing_pre_after_stopped(frames)
    assert [f["text"] for f in after if f["is_final"]] == ["post final"]
    assert [f["text"] for f in after if not f["is_final"]][-1:] == ["post "]


def test_audiocpp_quiet_flush_does_not_fire_in_a_stopped_gap(audiocpp_quiet):
    script = [(["pre "], "pre final"), (["post "], "post final")]
    with FakeSidecar(flush_script=script) as fake:
        client = _audiocpp_client(fake)
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(_PRE)
            ws.send_json(STOP)
            frames = recv_until(ws, is_type("listen_stopped"), limit=_LIMIT)
            time.sleep(0.7)                    # over three times the 200 ms window
            frames += barrier(ws)
            assert _after_stopped(frames) == []
            ws.send_json(START)
            ws.send_bytes(_POST)
            frames += recv_until(ws, _is_final, limit=_LIMIT)
    after = _assert_nothing_pre_after_stopped(frames)
    assert [(f["text"], f["is_final"]) for f in after] == [
        ("post ", False), ("post final", True)]


# --- audio.cpp under a turn engine: a turn-end flush running at the boundary ----------

class _NamingFlushSTT:
    """An audiocpp stand-in that commits only at a flush, naming the audio it was fed.

    The live partial is the names of the speech frames fed since the last reset; a
    flush makes it the final. ``hold`` makes the first flush wait on ``gate``;
    ``flush_started`` is set when it is running.
    """

    def __init__(self, hold: bool = False) -> None:
        self.gate = threading.Event()
        self.flush_started = threading.Event()
        self._hold = hold
        self._names: list[str] = []
        self._pending = False
        self._segment = 0

    async def feed(self, data: bytes) -> bool:
        self._pending = True
        name = _label_of(data)
        if name is None:
            return False
        if name not in self._names:
            self._names.append(name)
        return True

    def get_current_text(self):
        return " ".join(self._names), "en", self._segment

    def take_final(self):
        return None

    async def finalize(self):
        import asyncio
        if self._hold:
            self._hold = False
            self.flush_started.set()
            await asyncio.to_thread(self.gate.wait, _GATE_S)
        if not self._pending:
            return "", "en", self._segment
        self._pending = False
        return " ".join(self._names), "en", self._segment

    def reset(self) -> None:
        self._segment += 1
        self._names = []

    def dispose(self) -> None:
        pass


@pytest.fixture
def tyger():
    fake = FakeTygerSidecar().start()
    try:
        yield fake
    finally:
        fake.stop()


def test_audiocpp_turn_end_flush_in_progress_is_not_delivered_after_listen_stopped(
        tyger, settings):
    """A turn end flushes the open utterance and that flush is still running when
    listen_stop comes in; listen_start and new speech follow at once. Neither the
    flush's partial nor its final reaches the client after listen_stopped, and the
    new utterance is delivered."""
    _set(settings, VM_AUDIOCPP_SILENCE_MS="1000")
    made: list[_NamingFlushSTT] = []

    def factory(config=None, **kwargs):
        made.append(_NamingFlushSTT(hold=True))
        return made[-1]

    settings.setattr(audiocpp_stt, "AudioCppSTTPipeline", factory)
    app = make_app(sidecar=StandInSidecar(tyger.url))
    app.state.audiocpp_config = {"vad_model_factory": _AmplitudeVad}
    client = TestClient(app)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        opt_in(ws)
        assert tyger.wait_for(lambda: len(tyger.sessions) == 1)
        for _ in range(8):
            ws.send_bytes(_PRE)
        frames = [ws.receive_json() for _ in range(8)]   # one interim per frame fed
        assert [f["text"] for f in frames] == ["pre"] * 8
        tyger.send(0, wire.EndpointEvent(by="model", mic_sample=4096))
        assert made[0].flush_started.wait(_GATE_S)   # the turn-end flush is running
        ws.send_json(STOP)
        ws.send_json(START)
        for _ in range(8):
            ws.send_bytes(_POST)
        for _ in range(45):                    # past the 1 s window: silero ends it
            ws.send_bytes(_SILENCE)
        made[0].gate.set()
        frames += _to_first_final_after_stop(ws)
    after = _assert_nothing_pre_after_stopped(frames)
    assert [f["text"] for f in after if f["is_final"]] == ["post"]
    assert not [f for f in frames if is_type("error")(f)]
