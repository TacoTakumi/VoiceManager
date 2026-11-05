"""The Tyger sidecar server, run as a script with a fake scorer (no model, no GPU).

The test starts ``server.py --fake-scorer SCRIPT`` with the test interpreter on a
free port, waits for /health and talks to it over its WebSocket, one connection per
session. The fake scorer reads each session's per-frame scores from the script; the
server feeds them through that session's turn rules and sends the event lines.

Covers: two sessions fed interleaved hops each get exactly the events their own
script gives alone (and the same as a later connection fed alone); the rule
settings come from the command line; a malformed hop, a text message and a scorer
failure each give an error event line and the session goes on; a bad rule setting
stops the script at once; scorer.py imports with the standard library alone;
--torch-threads defaults to 1, refuses 0 and reaches the real scorer's load.
"""
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from websockets.sync.client import connect

from voicemanager.pipeline.tyger import wire
from voicemanager.pipeline.tyger.rules import FRAME_SAMPLES, TurnRules, TurnRuleSettings

TYGER_DIR = Path(__file__).resolve().parents[1] / "src" / "voicemanager" / "pipeline" / "tyger"
SERVER = TYGER_DIR / "server.py"
HOP_FRAMES = wire.HOP_SAMPLES // FRAME_SAMPLES

LISTEN, SPEAK = wire.MODE_LISTENING, wire.MODE_SPEAKING
VOICED, QUIET = -30.0, -80.0

# Rule settings given on the command line, chosen so that the script's events differ
# from what the defaults give.
CLI_RULES = {"endpoint_debounce_frames": 3, "silence_fallback_ms": 800.0}
CLI_ARGS = ["--endpoint-debounce", "3", "--silence-fallback-ms", "800"]


def _hops(*groups):
    """(mode, n_hops, frame) groups -> [(mode, [frame] * 4)] hops."""
    out = []
    for mode, n, frame in groups:
        out += [(mode, [list(frame)] * HOP_FRAMES) for _ in range(n)]
    return out


# Session "a": a spoken turn the endpoint head ends (two confident frames, then
# three), then a turn that silence ends.
SESSION_A = (
    _hops((LISTEN, 3, (0.1, 0.1, VOICED)))
    + [(LISTEN, [[0.95, 0.1, VOICED], [0.95, 0.1, VOICED], [0.1, 0.1, VOICED], [0.1, 0.1, VOICED]]),
       (LISTEN, [[0.96, 0.1, VOICED], [0.96, 0.1, VOICED], [0.96, 0.1, VOICED], [0.1, 0.1, VOICED]])]
    + _hops((LISTEN, 2, (0.1, 0.1, VOICED)), (LISTEN, 4, (0.1, 0.1, QUIET)))
)
# Session "b": quiet, then a reply plays and the user cuts in, then a new turn.
SESSION_B = (
    _hops((LISTEN, 2, (0.0, 0.0, QUIET)), (SPEAK, 2, (0.0, 0.2, VOICED)),
          (SPEAK, 2, (0.0, 0.9, VOICED)), (LISTEN, 3, (0.3, 0.1, VOICED)),
          (LISTEN, 5, (0.3, 0.1, QUIET)))
)
B_OFFSET = 1000  # session b's mic stream starts elsewhere, so its sample positions differ


def _script(sessions: dict) -> dict:
    return {"sessions": {key: [f for _mode, frames in hops for f in frames]
                         for key, hops in sessions.items()}}


def _hop_end(index: int, offset: int = 0) -> int:
    return (index + 1) * wire.HOP_SAMPLES + offset


def _hop(mode: str, mic_sample: int) -> bytes:
    silence = bytes(wire.CHANNEL_BYTES)
    return wire.pack_hop(mode, mic_sample, silence, silence)


def _expected(hops, offset: int = 0, rules: dict | None = None) -> list:
    """What one session's script gives alone: the turn rules fed its frames, the
    way the sidecar feeds them (frame ends on the mic stream)."""
    engine = TurnRules(TurnRuleSettings(**(rules if rules is not None else CLI_RULES)))
    events = []
    for k, (mode, frames) in enumerate(hops):
        engine.begin_hop(mode)
        end = _hop_end(k, offset)
        for j, (p_end, yld, db) in enumerate(frames):
            mic = end - (HOP_FRAMES - 1 - j) * FRAME_SAMPLES
            events += engine.on_frame(mode, p_end, yld, db, mic).events
    return events


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_health(port: int) -> tuple[int, dict]:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, {}


class Sidecar:
    def __init__(self, proc: subprocess.Popen, port: int):
        self.proc, self.port = proc, port

    def session(self, key: str | None = None):
        query = f"?session={key}" if key is not None else ""
        return connect(f"ws://127.0.0.1:{self.port}/tyger{query}", max_size=None,
                       open_timeout=5)


@pytest.fixture(scope="module")
def sidecar(tmp_path_factory):
    script = tmp_path_factory.mktemp("tyger") / "script.json"
    sessions = {"a": SESSION_A, "b": SESSION_B, "c": SESSION_A}
    data = _script(sessions)
    # Session "e": its scorer fails on the third hop (the entry stands for that hop's
    # first frame); the stream goes on after it.
    e_frames = data["sessions"]["c"]
    data["sessions"]["e"] = (e_frames[:2 * HOP_FRAMES] + [{"error": "fake scorer failure"}]
                             + e_frames[2 * HOP_FRAMES:])
    script.write_text(json.dumps(data))
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, str(SERVER), "--fake-scorer", str(script), "--host", "127.0.0.1",
         "--port", str(port), "--log-level", "warning", *CLI_ARGS],
        stdin=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 20
        while True:
            assert proc.poll() is None, f"sidecar exited with {proc.returncode}"
            try:
                if _get_health(port)[0] == 200:
                    break
            except OSError:
                pass
            assert time.monotonic() < deadline, "sidecar /health never answered 200"
            time.sleep(0.05)
        yield Sidecar(proc, port)
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(5)


def _collect_until_error(ws) -> list:
    """Every event the session sent, through the first error event."""
    events = []
    while True:
        batch = wire.parse_events(ws.recv(timeout=5))
        for event in batch:
            events.append(event)
        if any(isinstance(e, wire.ErrorEvent) for e in batch):
            return events


def test_health_reports_the_fake_scorer(sidecar):
    status, body = _get_health(sidecar.port)
    assert status == 200
    assert body["status"] == "ok"
    assert body["scorer"] == "fake"


def test_the_scripts_give_events_and_the_cli_rules_count():
    a, b = _expected(SESSION_A), _expected(SESSION_B, B_OFFSET)
    assert [(type(e), getattr(e, "by", None)) for e in a] == [
        (wire.EndpointEvent, "model"), (wire.EndpointEvent, "silence")]
    assert any(isinstance(e, wire.BargeEvent) for e in b)
    assert _expected(SESSION_A, rules={}) != a


def test_two_interleaved_sessions_each_get_their_own_events(sidecar):
    marker = b"not a hop"
    with sidecar.session("a") as ws_a, sidecar.session("b") as ws_b:
        for k in range(max(len(SESSION_A), len(SESSION_B))):
            if k < len(SESSION_A):
                ws_a.send(_hop(SESSION_A[k][0], _hop_end(k)))
            if k < len(SESSION_B):
                ws_b.send(_hop(SESSION_B[k][0], _hop_end(k, B_OFFSET)))
        ws_a.send(marker)
        ws_b.send(marker)
        got_a, got_b = _collect_until_error(ws_a), _collect_until_error(ws_b)
    assert got_a[:-1] == _expected(SESSION_A)
    assert got_b[:-1] == _expected(SESSION_B, B_OFFSET)
    assert isinstance(got_a[-1], wire.ErrorEvent) and isinstance(got_b[-1], wire.ErrorEvent)

    # A later connection with the same script, fed alone, gives the same events.
    with sidecar.session("a") as ws:
        for k, (mode, _frames) in enumerate(SESSION_A):
            ws.send(_hop(mode, _hop_end(k)))
        ws.send(marker)
        assert _collect_until_error(ws)[:-1] == got_a[:-1]


def test_a_bad_message_gives_an_error_event_and_the_session_goes_on(sidecar):
    good = _hop(LISTEN, _hop_end(0))
    bad_magic = b"XXXX" + good[4:]
    with sidecar.session("c") as ws:
        for k in range(2):
            ws.send(_hop(SESSION_A[k][0], _hop_end(k)))
        ws.send(good[:-2])
        first = wire.parse_events(ws.recv(timeout=5))
        ws.send("a text message")
        second = wire.parse_events(ws.recv(timeout=5))
        ws.send(bad_magic)
        third = wire.parse_events(ws.recv(timeout=5))
        for k in range(2, len(SESSION_A)):
            ws.send(_hop(SESSION_A[k][0], _hop_end(k)))
        ws.send(b"end")
        rest = _collect_until_error(ws)
    for errors in (first, second, third):
        assert len(errors) == 1 and isinstance(errors[0], wire.ErrorEvent)
    assert "hop" in first[0].message
    assert "binary" in second[0].message
    assert "magic" in third[0].message
    # The bad messages changed nothing: the good hops give the script's events.
    assert rest[:-1] == _expected(SESSION_A)


def test_a_scorer_failure_gives_an_error_event_and_the_session_goes_on(sidecar):
    with sidecar.session("e") as ws:
        for k in range(2):
            ws.send(_hop(SESSION_A[k][0], _hop_end(k)))
        ws.send(_hop(LISTEN, _hop_end(2)))       # the scorer fails on this hop
        failed = wire.parse_events(ws.recv(timeout=5))
        for k in range(2, len(SESSION_A)):
            ws.send(_hop(SESSION_A[k][0], _hop_end(k + 1)))
        ws.send(b"end")
        rest = _collect_until_error(ws)
    assert len(failed) == 1 and isinstance(failed[0], wire.ErrorEvent)
    assert "fake scorer failure" in failed[0].message
    # The failed hop scored no frame: the rest is the script's turn, one hop later.
    expected = _expected(SESSION_A[:2] + [(LISTEN, [])] + SESSION_A[2:])
    assert rest[:-1] == expected


def test_an_unscripted_session_runs_on_quiet_scores(sidecar):
    # An unlabelled session is keyed by its connection order; with no stream under
    # that key it scores nothing confident and the quiet audio gives no events.
    with sidecar.session() as ws:
        ws.send(_hop(LISTEN, _hop_end(0)))
        ws.send(b"end")
        events = _collect_until_error(ws)
    assert len(events) == 1 and isinstance(events[0], wire.ErrorEvent)


def test_a_bad_rule_setting_stops_the_script():
    proc = subprocess.run(
        [sys.executable, str(SERVER), "--fake-scorer", "/nonexistent.json",
         "--endpoint-threshold", "1.5"],
        capture_output=True, text=True, timeout=30)
    assert proc.returncode == 2
    assert "endpoint_threshold" in proc.stderr


def test_the_scorer_module_imports_with_the_standard_library_alone(tmp_path):
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"sessions": {"x": [[0.5, 0.25, -20.0]]}}))
    code = f"""
import importlib.util, sys
from pathlib import Path
d = Path({str(TYGER_DIR)!r})
spec = importlib.util.spec_from_file_location(
    "tyger", d / "__init__.py", submodule_search_locations=[str(d)])
sys.modules["tyger"] = pkg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pkg)
import tyger.scorer
factory = tyger.scorer.FakeScorer.from_file({str(script)!r})
s = factory.new_session("x", 0)
frames = list(s.score_hop(bytes(10240), bytes(10240), "listening"))
assert frames[0] == (0.5, 0.25, -20.0), frames
assert len(frames) == 4
assert frames[1].energy_db < -100, frames
for name in ("numpy", "torch", "transformers", "fastapi", "pydantic"):
    assert name not in sys.modules, name
print("ok")
"""
    proc = subprocess.run([sys.executable, "-I", "-S", "-c", code],
                          capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_torch_threads_defaults_to_one_and_refuses_zero(capsys):
    from voicemanager.pipeline.tyger import server
    assert server.build_parser().parse_args([]).torch_threads == 1
    assert server.build_parser().parse_args(["--torch-threads", "4"]).torch_threads == 4
    for bad in ("0", "-2", "two"):
        with pytest.raises(SystemExit) as exc:
            server.build_parser().parse_args(["--torch-threads", bad])
        assert exc.value.code == 2
        assert "--torch-threads" in capsys.readouterr().err


def test_main_passes_torch_threads_to_the_scorer_load(monkeypatch):
    """main hands --torch-threads to TygerScorer.load; nothing loads or serves."""
    import types

    import uvicorn

    from voicemanager.pipeline.tyger import server
    seen = {}

    def fake_create_app(settings, **kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(state=types.SimpleNamespace(load_error=None))

    def fake_load(model, revision, device, threads):
        seen["load"] = (model, revision, device, threads)
        return "loaded"

    class FakeServer:
        def __init__(self, config):
            pass

        def run(self):
            seen["ran"] = True

    monkeypatch.setattr(server, "create_app", fake_create_app)
    monkeypatch.setattr(server.scorer.TygerScorer, "load", staticmethod(fake_load))
    monkeypatch.setattr(uvicorn, "Config", lambda *a, **k: None)
    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    server.main(["--device", "cpu", "--torch-threads", "3", "--revision", "abc"])
    assert seen["ran"] is True
    assert seen["load_scorer"]() == "loaded"
    assert seen["load"] == (server.DEFAULT_MODEL, "abc", "cpu", 3)


# --- the experimental pre-filter, through the sidecar app -------------------------------

class _RecordingSession:
    def __init__(self, store):
        self.store = store

    def reset(self):
        pass

    def score_hop(self, user, agent, mode):
        from voicemanager.pipeline.tyger.scorer import FrameScore
        self.store.append((bytes(user), bytes(agent), mode))
        return iter([FrameScore(0.0, 0.0, -90.0)] * 4)


class _RecordingScorer:
    name = "recording"

    def __init__(self):
        self.hops = []

    def new_session(self, key, index):
        return _RecordingSession(self.hops)


class _SilentFactory:
    """Enrolls at once into a filter that outputs silence, 384 samples late."""

    def enroll(self, sample):
        import numpy as np

        class _F:
            delay = 384

            def push(self, samples):
                return np.zeros(len(samples), np.float32)
        return _F()


def _feed_app(prefilter_factory):
    import numpy as np
    from fastapi.testclient import TestClient
    from voicemanager.pipeline.tyger import server
    rec = _RecordingScorer()
    app = server.create_app(turn_scorer=rec, prefilter_factory=prefilter_factory)
    rng = np.random.default_rng(5)
    voiced = lambda: (rng.uniform(-0.1, 0.1, wire.HOP_SAMPLES) * 32768).astype("<i2").tobytes()
    agent = (rng.uniform(-0.3, 0.3, wire.HOP_SAMPLES) * 32768).astype("<i2").tobytes()
    sent = []
    for i in range(12):        # 3.84 s of voice while listening: enough to enroll
        sent.append((voiced(), bytes(wire.HOP_SAMPLES * 2), wire.MODE_LISTENING))
    for i in range(3):
        sent.append((voiced(), agent, wire.MODE_SPEAKING))
    # With progress lines every hop gets an answer once it is scored.
    with TestClient(app) as client, \
            client.websocket_connect(f"/tyger?session=p&{wire.PROGRESS_PARAM}=1") as ws:
        ws.receive_text()
        for k, (u, a, m) in enumerate(sent):
            ws.send_bytes(wire.pack_hop(m, (k + 1) * wire.HOP_SAMPLES, u, a))
            ws.receive_text()
    return sent, rec.hops


def test_with_no_prefilter_the_scorer_sees_every_hop_as_sent():
    sent, seen = _feed_app(None)
    assert seen == sent


def test_with_a_prefilter_speaking_hops_reach_the_scorer_filtered_with_a_silent_agent():
    sent, seen = _feed_app(_SilentFactory())
    assert len(seen) == len(sent)
    assert seen[:12] == sent[:12]                  # listening hops: unchanged
    for (u, a, m), (su, _sa, _sm) in zip(seen[12:], sent[12:]):
        assert m == wire.MODE_SPEAKING
        assert a == bytes(wire.HOP_SAMPLES * 2)
        assert u != su and len(u) == wire.HOP_SAMPLES * 2
