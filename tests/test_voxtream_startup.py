"""The voxtream engine's startup and shutdown, model-free.

The sidecar is a fake process (tests/voxtream_fake_sidecar.py) started with the test
interpreter in place of the VoXtream environment's python. Covers: the settings and
their defaults, the device argument, a ready fake making the engine load and the app
start with engine type voxtream, a missing interpreter and a never-ready fake failing
startup with the setting named, the process gone after shutdown, the presets
validated before any spawn, and the unknown-voice error.
"""
import json
import socket
import sys
import time
from pathlib import Path

import pytest

import authkit  # test auth env; must be imported before voicemanager
from fastapi.testclient import TestClient

import voicemanager.app as app_module
import voicemanager.models as models
from voicemanager.config import Settings, resolve_home
from voicemanager.pipeline.audiocpp_tts import PresetLoadError, UnknownVoiceError
from voicemanager.pipeline.voxtream_tts import (
    VoxtreamSidecar, VoxtreamStartupError, VoxtreamTTSEngine)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_REFS = REPO_ROOT / "voices" / "designed" / "refs.json"
FAKE_SIDECAR = Path(__file__).with_name("voxtream_fake_sidecar.py")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _settings(**overrides) -> Settings:
    base = dict(
        _env_file=None,
        tts_engine="voxtream",
        device="cpu",
        voxtream_py=sys.executable,
        voxtream_port=_free_port(),
        voxtream_voices=str(REAL_REFS),
        voxtream_ready_timeout=10.0,
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def fake_sidecar(monkeypatch):
    """Point the sidecar manager at the fake script; yields a mode setter."""
    monkeypatch.setattr("voicemanager.pipeline.voxtream_tts.SERVER_SCRIPT", FAKE_SIDECAR)

    def set_mode(mode: str, **env):
        monkeypatch.setenv("VOXTREAM_FAKE_MODE", mode)
        for k, v in env.items():
            monkeypatch.setenv(k, str(v))
    set_mode("ready")
    return set_mode


# --- settings ------------------------------------------------------------------------

def test_settings_expose_voxtream_fields_with_defaults():
    s = Settings(_env_file=None, tts_engine="voxtream")
    assert s.tts_engine == "voxtream"
    assert s.voxtream_py == str(resolve_home() / "venv-voxtream/bin/python")
    assert s.voxtream_port == 8256
    assert s.voxtream_voices == str(resolve_home() / "voices/designed/refs.json")
    assert s.voxtream_voice == "designed_female_s101"
    assert s.voxtream_ready_timeout == 180.0
    assert s.voxtream_compile is False


def test_load_tts_models_routes_voxtream_to_its_loader(monkeypatch):
    monkeypatch.setattr(models, "_load_voxtream_tts", lambda s: {"tts_engine_type": "voxtream"})
    import types
    assert models.load_tts_models(types.SimpleNamespace(tts_engine="voxtream")) == {
        "tts_engine_type": "voxtream"}


def test_device_argument_maps_resolved_device():
    assert VoxtreamSidecar.device_argument("cuda:3") == "3"
    assert VoxtreamSidecar.device_argument("cuda") is None
    assert VoxtreamSidecar.device_argument("cpu") == "-1"


# --- spawn, ready, shutdown ----------------------------------------------------------

def test_engine_loads_against_a_ready_fake_and_stops_it(fake_sidecar, tmp_path):
    args_file = tmp_path / "args.json"
    fake_sidecar("ready", VOXTREAM_FAKE_ARGS_FILE=args_file)
    settings = _settings(device="cuda:3")
    result = models._load_voxtream_tts(settings)
    engine = result["tts_engine"]
    try:
        assert result["tts_engine_type"] == "voxtream"
        assert isinstance(engine, VoxtreamTTSEngine)
        assert engine.sample_rate == 24000
        assert engine.sidecar.alive
        assert engine.sidecar.url == f"ws://127.0.0.1:{settings.voxtream_port}/voxtream"
        argv = json.loads(args_file.read_text())
        assert argv[argv.index("--port") + 1] == str(settings.voxtream_port)
        assert argv[argv.index("--host") + 1] == "127.0.0.1"
        assert "--compile" not in argv
        proc = engine.sidecar.process
    finally:
        engine.dispose()
        result["tts_executor"].shutdown(wait=True)
    deadline = time.monotonic() + 5
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert proc.poll() is not None
    assert not engine.sidecar.alive


def test_slow_fake_is_waited_for(fake_sidecar):
    fake_sidecar("slow", VOXTREAM_FAKE_DELAY_S=0.5)
    engine = VoxtreamTTSEngine(
        VoxtreamSidecar(sys.executable, _free_port(), ready_timeout_s=10.0),
        default_voice="designed_female_s101")
    try:
        t0 = time.monotonic()
        engine.load(voices_path=REAL_REFS)
        assert time.monotonic() - t0 >= 0.4
        assert engine.sidecar.alive
    finally:
        engine.dispose()


def test_app_starts_with_engine_type_voxtream_and_stops_the_sidecar(fake_sidecar, monkeypatch):
    settings = _settings()
    monkeypatch.setattr(app_module, "get_settings", lambda: settings)
    monkeypatch.setattr(app_module, "load_models", lambda s: {
        "stt_engine_type": "qwen", "device": "cpu", "qwen_model": object()})
    captured = {}
    real_loader = models._load_voxtream_tts

    def loader(s):
        result = real_loader(s)
        captured["engine"] = result["tts_engine"]
        return result
    monkeypatch.setattr(models, "_load_voxtream_tts", loader)
    with TestClient(app_module.create_app()) as client:
        assert client.app.state.tts_engine_type == "voxtream"
        proc = captured["engine"].sidecar.process
        assert proc.poll() is None
    deadline = time.monotonic() + 5
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert proc.poll() is not None


def test_missing_interpreter_fails_naming_the_setting(fake_sidecar, tmp_path):
    settings = _settings(voxtream_py=str(tmp_path / "no-such-venv" / "bin" / "python"))
    with pytest.raises(VoxtreamStartupError) as exc:
        models._load_voxtream_tts(settings)
    assert "VM_VOXTREAM_PY" in str(exc.value)


def test_never_ready_fake_fails_naming_the_timeout_and_leaves_nothing(fake_sidecar):
    fake_sidecar("never")
    sidecar = VoxtreamSidecar(sys.executable, _free_port(), ready_timeout_s=1.0)
    engine = VoxtreamTTSEngine(sidecar, default_voice="designed_female_s101")
    t0 = time.monotonic()
    with pytest.raises(VoxtreamStartupError) as exc:
        engine.load(voices_path=REAL_REFS)
    assert "VM_VOXTREAM_READY_TIMEOUT" in str(exc.value)
    assert time.monotonic() - t0 < 5
    assert sidecar.process is None
    assert not sidecar.alive


def test_fake_that_exits_fails_with_its_code(fake_sidecar):
    fake_sidecar("exit")
    sidecar = VoxtreamSidecar(sys.executable, _free_port(), ready_timeout_s=5.0)
    with pytest.raises(VoxtreamStartupError) as exc:
        sidecar.start()
    assert "code 3" in str(exc.value)
    assert sidecar.process is None


# --- presets -------------------------------------------------------------------------

def test_bad_presets_refuse_to_start_before_any_spawn(fake_sidecar, tmp_path):
    bad = tmp_path / "refs.json"
    bad.write_text('{"designed_female_s101": {"file": "absent.wav", "ref_text": "x"}}',
                   encoding="utf-8")
    engine = VoxtreamTTSEngine(
        VoxtreamSidecar(sys.executable, _free_port()), default_voice="designed_female_s101")
    with pytest.raises(PresetLoadError):
        engine.load(voices_path=bad)
    assert engine.sidecar.process is None


def test_default_voice_must_be_a_preset(fake_sidecar):
    engine = VoxtreamTTSEngine(
        VoxtreamSidecar(sys.executable, _free_port()), default_voice="not_a_preset")
    with pytest.raises(VoxtreamStartupError) as exc:
        engine.load(voices_path=REAL_REFS)
    assert "VM_VOXTREAM_VOICE" in str(exc.value)
    assert engine.sidecar.process is None


def test_resolve_voice_default_other_and_unknown(fake_sidecar):
    engine = VoxtreamTTSEngine(
        VoxtreamSidecar(sys.executable, _free_port()), default_voice="designed_female_s101")
    engine.load(voices_path=REAL_REFS)
    try:
        assert engine.resolve_voice(None).name == "designed_female_s101"
        assert engine.resolve_voice("").name == "designed_female_s101"
        assert engine.resolve_voice("designed_gm_02").wav_path.endswith("designed_gm_02.wav")
        assert "designed_gm_02" in engine.available_voices
        with pytest.raises(UnknownVoiceError) as exc:
            engine.resolve_voice("nope")
        assert "designed_female_s101" in str(exc.value)
    finally:
        engine.dispose()


# --- /voice/upload: presets only -----------------------------------------------------

def test_upload_under_voxtream_returns_400_naming_the_limitation():
    from voicemanager.session import SessionRegistry

    app = app_module.create_app()
    app.state.tts_engine_type = "voxtream"
    app.state.session_registry = SessionRegistry()
    client = TestClient(app)
    response = client.post(
        "/voice/upload",
        params={**authkit.auth_params(), "session_id": "s1"},
        headers=authkit.auth_headers(),
        files={"file": ("v.wav", b"RIFF" + b"\x00" * 40, "audio/wav")},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "voxtream" in detail
    assert "preset" in detail.lower()
