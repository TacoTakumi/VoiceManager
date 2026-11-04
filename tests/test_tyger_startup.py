"""The Tyger sidecar's startup and shutdown from the server lifespan, model-free.

The sidecar is the fake process from tests/tyger_fake_sidecar.py, started with the
test interpreter in place of the Tyger environment's python. Covers: the command
line (the configured interpreter, the sidecar script, the model, the device and
every rule value), the device that a blank VM_TYGER_DEVICE inherits, a ready fake
making the app start with the sidecar on app.state, a missing interpreter and a
never-ready fake failing startup with the setting named, a port already in use,
and the setting off spawning nothing.
"""
import concurrent.futures
import json
import socket
import sys
import time
from pathlib import Path

import pytest

import authkit  # noqa: F401 (test auth env; must be imported before voicemanager)
from fastapi.testclient import TestClient

import voicemanager.app as app_module
import voicemanager.models as models
import voicemanager.pipeline.tyger.sidecar as sidecar_module
from voicemanager.config import Settings
from voicemanager.pipeline.tyger.sidecar import TygerSidecar, TygerStartupError

FAKE_SIDECAR = Path(__file__).with_name("tyger_fake_sidecar.py")
REAL_SCRIPT = (Path(__file__).resolve().parent.parent / "src" / "voicemanager" / "pipeline"
               / "tyger" / "server.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        turn_engine="tyger",
        device="cpu",
        tyger_py=sys.executable,
        tyger_port=_free_port(),
        tyger_ready_timeout=10.0,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def fake_sidecar(monkeypatch):
    """Point the sidecar manager at the fake script; yields a mode setter."""
    monkeypatch.setattr(sidecar_module, "SERVER_SCRIPT", FAKE_SIDECAR)

    def set_mode(mode: str, **env):
        monkeypatch.setenv("TYGER_FAKE_MODE", mode)
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))
    set_mode("ready")
    return set_mode


def _stub_engines(monkeypatch, settings: Settings) -> None:
    """Run the real lifespan with stand-in STT and TTS engines."""
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(app_module, "load_models", lambda s: {
        "stt_engine_type": "qwen", "device": "cpu", "qwen_model": object()})
    monkeypatch.setattr(app_module, "load_tts_models", lambda s: {
        "tts_engine": object(),
        "tts_executor": concurrent.futures.ThreadPoolExecutor(max_workers=1),
        "tts_engine_type": "openvoice"})


def _wait_exit(proc, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert proc.poll() is not None


# --- the command line ----------------------------------------------------------------

def test_the_command_runs_the_sidecar_script_by_path_with_the_configured_interpreter():
    sidecar = TygerSidecar.from_settings(_settings(), device="cuda:3")
    cmd = sidecar.command()
    assert cmd[0] == sys.executable
    assert Path(cmd[1]) == REAL_SCRIPT
    assert cmd[cmd.index("--device") + 1] == "cuda:3"


def test_the_sidecar_is_spawned_with_the_model_device_and_every_rule_value(
        fake_sidecar, tmp_path):
    args_file = tmp_path / "args.json"
    fake_sidecar("ready", TYGER_FAKE_ARGS_FILE=args_file)
    settings = _settings(
        device="cuda:3",
        tyger_model="/home/you/bundles/tyger",
        tyger_revision="abc123",
        tyger_endpoint_threshold=0.8,
        tyger_endpoint_debounce=3,
        tyger_barge_threshold=0.55,
        tyger_barge_debounce=4,
        tyger_warmup_frames=5,
        tyger_min_speech_ms=300,
        tyger_voice_gate_db=-45.0,
        tyger_silence_fallback_ms=900,
    )
    sidecar = models.load_turn_engine(settings)
    try:
        assert isinstance(sidecar, TygerSidecar)
        assert sidecar.up
        argv = json.loads(args_file.read_text())

        def value(flag: str) -> str:
            return argv[argv.index(flag) + 1]

        assert value("--host") == "127.0.0.1"
        assert value("--port") == str(settings.tyger_port)
        assert value("--model") == "/home/you/bundles/tyger"
        assert value("--revision") == "abc123"
        assert value("--device") == "cuda:3"  # blank VM_TYGER_DEVICE: VM_DEVICE's device
        assert float(value("--endpoint-threshold")) == 0.8
        assert int(value("--endpoint-debounce")) == 3
        assert float(value("--barge-threshold")) == 0.55
        assert int(value("--barge-debounce")) == 4
        assert int(value("--warmup-frames")) == 5
        assert float(value("--min-speech-ms")) == 300
        assert float(value("--voice-gate-db")) == -45.0
        assert float(value("--silence-fallback-ms")) == 900
        # The session settings belong to the server, not the sidecar.
        assert not any(a.startswith(("--hold", "--start-wait")) for a in argv)
        assert sidecar.url == f"ws://127.0.0.1:{settings.tyger_port}/tyger"
        assert sidecar.session_url("conn 1") == (
            f"ws://127.0.0.1:{settings.tyger_port}/tyger?session=conn+1")
        proc = sidecar.process
    finally:
        sidecar.close()
    _wait_exit(proc)
    assert not sidecar.alive and not sidecar.up


def test_an_explicit_tyger_device_wins_over_vm_device(fake_sidecar, tmp_path):
    args_file = tmp_path / "args.json"
    fake_sidecar("ready", TYGER_FAKE_ARGS_FILE=args_file)
    sidecar = models.load_turn_engine(_settings(device="cuda:3", tyger_device="cpu"))
    try:
        argv = json.loads(args_file.read_text())
        assert argv[argv.index("--device") + 1] == "cpu"
    finally:
        sidecar.close()


def test_the_rule_values_match_the_sidecar_scripts_flags():
    # Every flag the manager passes is one the real script's parser knows; the
    # check reads the script's text, so nothing from it is imported here.
    text = REAL_SCRIPT.read_text()
    sidecar = TygerSidecar.from_settings(_settings(), device="cpu")
    flags = [a for a in sidecar.command()[2:] if a.startswith("--")]
    assert len(flags) == 13, flags
    for flag in flags:
        assert f'"{flag}"' in text, flag


# --- failures name the setting -------------------------------------------------------

def test_missing_interpreter_fails_naming_the_setting(fake_sidecar, tmp_path):
    settings = _settings(tyger_py=str(tmp_path / "no-such-venv" / "bin" / "python"))
    with pytest.raises(TygerStartupError) as exc:
        models.load_turn_engine(settings)
    assert "VM_TYGER_PY" in str(exc.value)


def test_never_ready_fake_fails_naming_the_timeout_and_leaves_nothing(fake_sidecar):
    fake_sidecar("never")
    sidecar = TygerSidecar.from_settings(_settings(tyger_ready_timeout=1.0), device="cpu")
    t0 = time.monotonic()
    with pytest.raises(TygerStartupError) as exc:
        sidecar.start()
    assert "VM_TYGER_READY_TIMEOUT" in str(exc.value)
    assert time.monotonic() - t0 < 5
    assert sidecar.process is None
    assert not sidecar.alive and not sidecar.up


def test_fake_that_exits_fails_with_its_code_naming_the_interpreter(fake_sidecar):
    fake_sidecar("exit")
    sidecar = TygerSidecar.from_settings(_settings(), device="cpu")
    with pytest.raises(TygerStartupError) as exc:
        sidecar.start()
    assert "code 3" in str(exc.value)
    assert "VM_TYGER_PY" in str(exc.value)
    assert sidecar.process is None


def test_slow_fake_is_waited_for(fake_sidecar):
    fake_sidecar("slow", TYGER_FAKE_DELAY_S=0.5)
    sidecar = TygerSidecar.from_settings(_settings(), device="cpu")
    try:
        t0 = time.monotonic()
        sidecar.start()
        assert time.monotonic() - t0 >= 0.4
        assert sidecar.up
    finally:
        sidecar.close()


def test_a_port_already_in_use_refuses_startup_naming_the_port_setting(fake_sidecar, tmp_path):
    args_file = tmp_path / "args.json"
    fake_sidecar("ready", TYGER_FAKE_ARGS_FILE=args_file)
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = taken.getsockname()[1]
        sidecar = TygerSidecar.from_settings(_settings(tyger_port=port), device="cpu")
        with pytest.raises(TygerStartupError) as exc:
            sidecar.start()
    assert "VM_TYGER_PORT" in str(exc.value)
    assert sidecar.process is None
    assert not args_file.exists()  # nothing was spawned


# --- the lifespan --------------------------------------------------------------------

def test_the_app_starts_the_sidecar_and_stops_it_at_shutdown(fake_sidecar, monkeypatch):
    _stub_engines(monkeypatch, _settings())
    with TestClient(app_module.create_app()) as client:
        sidecar = client.app.state.turn_engine_sidecar
        assert isinstance(sidecar, TygerSidecar)
        assert sidecar.up and sidecar.generation == 1
        watcher = client.app.state.turn_engine_watcher
        assert watcher is not None and not watcher.done()
        proc = sidecar.process
        assert proc.poll() is None
    _wait_exit(proc)
    assert watcher.done()
    assert sidecar.process is None and not sidecar.up


def test_a_missing_interpreter_refuses_to_start_the_server(fake_sidecar, monkeypatch, tmp_path):
    _stub_engines(monkeypatch, _settings(tyger_py=str(tmp_path / "absent" / "python")))
    with pytest.raises(TygerStartupError) as exc:
        with TestClient(app_module.create_app()):
            pass
    assert "VM_TYGER_PY" in str(exc.value)


def test_setting_off_spawns_nothing(monkeypatch):
    def no_spawn(*args, **kwargs):
        raise AssertionError("a process was spawned with the turn engine off")
    monkeypatch.setattr(sidecar_module.subprocess, "Popen", no_spawn)
    assert models.load_turn_engine(_settings(turn_engine="off")) is None
    _stub_engines(monkeypatch, _settings(turn_engine="off"))
    with TestClient(app_module.create_app()) as client:
        assert client.app.state.turn_engine_sidecar is None
        assert client.app.state.turn_engine_watcher is None
        assert client.get("/health").status_code == 200


def test_the_default_setting_is_off():
    assert Settings(_env_file=None).turn_engine == "off"
