"""Engine selection, VM_* config, startup wiring, upload rejection.

Model-free and server-free (the CLAUDE.md invariant): sentinel loaders assert the
load_tts_models dispatch, Settings(_env_file=None) asserts the new VM_AUDIOCPP_TTS_*
surface isolated from the operator .env, a temp refs.json asserts fail-fast startup,
and a hand-wired TestClient asserts the /voice/upload 400 under audiocpp.
"""
import types
from pathlib import Path

import pytest

import authkit  # test auth env; must be imported before voicemanager

from fastapi.testclient import TestClient

import voicemanager.models as models
from voicemanager.app import create_app
from voicemanager.config import Settings, resolve_home
from voicemanager.pipeline.audiocpp_tts import AudioCppTTSEngine, PresetLoadError
from voicemanager.pipeline.audiocpp_tts import BREEZE_TTS_MODEL_TASK, DEFAULT_TTS_MODEL_TASK
from voicemanager.pipeline.audiocpp_tts import QWEN3_TTS_MODEL_TASK
from voicemanager.session import SessionRegistry

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_REFS = REPO_ROOT / "voices" / "designed" / "refs.json"


# --- load_tts_models dispatch (config-level, no model load) --------------------------

@pytest.mark.parametrize("engine", ["openvoice", "vibevoice", "audiocpp"])
def test_load_tts_models_routes_each_engine_to_its_own_loader(engine, monkeypatch):
    """Each VM_TTS_ENGINE value dispatches to exactly its own loader; the two
    prior values keep routing to their existing loaders (byte-identical
    selection behavior for openvoice and vibevoice)."""
    for name, who in [
        ("_load_openvoice_tts", "openvoice"),
        ("_load_vibevoice_tts", "vibevoice"),
        ("_load_audiocpp_tts", "audiocpp"),
    ]:
        monkeypatch.setattr(
            models, name, lambda s, who=who: {"tts_engine_type": who, "_who": who}
        )
    settings = types.SimpleNamespace(tts_engine=engine)
    result = models.load_tts_models(settings)
    assert result["_who"] == engine
    assert result["tts_engine_type"] == engine


def test_load_tts_models_unknown_engine_fails_fast(monkeypatch):
    """An unknown VM_TTS_ENGINE fails fast with the defined error instead of the
    old silent fall-through to openvoice."""
    for name in ("_load_openvoice_tts", "_load_vibevoice_tts", "_load_audiocpp_tts"):
        def _boom(s, name=name):
            raise AssertionError(f"{name} must not be called for an unknown engine")
        monkeypatch.setattr(models, name, _boom)
    settings = types.SimpleNamespace(tts_engine="bogus-tts-engine")
    with pytest.raises(ValueError) as exc:
        models.load_tts_models(settings)
    assert "bogus-tts-engine" in str(exc.value)


# --- config surface ------------------------------------------------------------------

def test_settings_expose_audiocpp_tts_fields():
    """The audiocpp TTS engine value + its VM_AUDIOCPP_TTS_* settings exist and
    bind, with the spec'd defaults (isolated from the operator .env)."""
    s = Settings(
        _env_file=None,
        tts_engine="audiocpp",
        audiocpp_tts_model="/models/dots-tts-soar-q8_0.gguf",
    )
    assert s.tts_engine == "audiocpp"
    assert s.audiocpp_tts_model.endswith("q8_0.gguf")
    assert s.audiocpp_tts_model_task == "breeze-tts-streaming"
    assert s.audiocpp_tts_voices == str(resolve_home() / "voices/designed/refs.json")
    assert s.audiocpp_tts_voice == "designed_female_s101"
    assert s.audiocpp_tts_seam_silence_ms == 400  # 0 disables the trim
    assert s.audiocpp_tts_request_options == "seed=2"  # fixed seed by default


# --- model_task selection: dots default, voxcpm2 opt-in ---------

def test_dots_model_task_default_requires_ref_text():
    """The default VM_AUDIOCPP_TTS_MODEL_TASK resolves to dots and derives the
    per-model ref_text rule (required)."""
    result = models._load_audiocpp_tts(_tts_settings(str(REAL_REFS)))
    try:
        engine = result["tts_engine"]
        assert engine._config["model_task"] == "dots-tts-streaming"
        assert engine._ref_text_required is True
    finally:
        result["tts_executor"].shutdown(wait=False)


def test_voxcpm2_model_task_selects_end_to_end():
    """VM_AUDIOCPP_TTS_MODEL_TASK=voxcpm2-tts-streaming selects voxcpm2: the loaded
    engine carries the task AND derives ref_text_required=False, so the per-model
    behavior follows the deployment-level selection."""
    settings = _tts_settings(str(REAL_REFS))
    settings.audiocpp_tts_model_task = "voxcpm2-tts-streaming"
    result = models._load_audiocpp_tts(settings)
    try:
        engine = result["tts_engine"]
        assert engine._config["model_task"] == "voxcpm2-tts-streaming"
        assert engine._ref_text_required is False
    finally:
        result["tts_executor"].shutdown(wait=False)


def test_voxcpm2_model_task_is_the_hello_wire_id():
    """The selected model_task rides the sidecar HELLO verbatim: selection
    is deployment-level, never a per-request field."""
    from audiocpp_tts_fake import FakeTtsSidecar
    with FakeTtsSidecar() as fake:
        engine = AudioCppTTSEngine({"socket": fake.socket_path,
                                    "model_task": "voxcpm2-tts-streaming"})
        engine.load(voices_path=str(REAL_REFS))
        try:
            engine.connect()  # in-process fake: blocking handshake completes inline
        finally:
            engine.dispose()
        assert fake.hello["model_task"] == "voxcpm2-tts-streaming"


# --- breeze opt-in -------------------------------------------------------------------

def test_breeze_model_task_selects_end_to_end():
    """VM_AUDIOCPP_TTS_MODEL_TASK=breeze-tts-streaming selects breeze: the loaded
    engine carries the task AND derives ref_text_required=True (breeze clones from
    the reference clip plus its exact transcript)."""
    settings = _tts_settings(str(REAL_REFS))
    settings.audiocpp_tts_model_task = BREEZE_TTS_MODEL_TASK
    result = models._load_audiocpp_tts(settings)
    try:
        engine = result["tts_engine"]
        assert engine._config["model_task"] == "breeze-tts-streaming"
        assert engine._ref_text_required is True
    finally:
        result["tts_executor"].shutdown(wait=False)


def test_breeze_model_task_is_the_hello_wire_id():
    """The breeze model_task rides the sidecar HELLO verbatim, like voxcpm2."""
    from audiocpp_tts_fake import FakeTtsSidecar
    with FakeTtsSidecar() as fake:
        engine = AudioCppTTSEngine({"socket": fake.socket_path,
                                    "model_task": BREEZE_TTS_MODEL_TASK})
        engine.load(voices_path=str(REAL_REFS))
        try:
            engine.connect()
        finally:
            engine.dispose()
        assert fake.hello["model_task"] == "breeze-tts-streaming"


# --- qwen3 opt-in --------------------------------------------------------------------

def test_qwen3_model_task_selects_end_to_end():
    """VM_AUDIOCPP_TTS_MODEL_TASK=qwen3-tts-streaming selects Qwen3-TTS: the loaded
    engine carries the task and requires ref_text (Base clones from the reference
    clip plus its transcript)."""
    settings = _tts_settings(str(REAL_REFS))
    settings.audiocpp_tts_model_task = QWEN3_TTS_MODEL_TASK
    result = models._load_audiocpp_tts(settings)
    try:
        engine = result["tts_engine"]
        assert engine._config["model_task"] == "qwen3-tts-streaming"
        assert engine._ref_text_required is True
    finally:
        result["tts_executor"].shutdown(wait=False)


def test_qwen3_model_task_is_the_hello_wire_id():
    """The qwen3 model_task rides the sidecar HELLO verbatim."""
    from audiocpp_tts_fake import FakeTtsSidecar
    with FakeTtsSidecar() as fake:
        engine = AudioCppTTSEngine({"socket": fake.socket_path,
                                    "model_task": QWEN3_TTS_MODEL_TASK})
        engine.load(voices_path=str(REAL_REFS))
        try:
            engine.connect()
        finally:
            engine.dispose()
        assert fake.hello["model_task"] == "qwen3-tts-streaming"


def test_default_model_task_stays_dots_after_breeze():
    """Adding breeze changes no default: the settings default and the engine
    constant are still dots-tts-streaming."""
    assert DEFAULT_TTS_MODEL_TASK == "dots-tts-streaming"
    assert _tts_settings(str(REAL_REFS)).audiocpp_tts_model_task == "dots-tts-streaming"
    assert BREEZE_TTS_MODEL_TASK != DEFAULT_TTS_MODEL_TASK


def test_ws_tts_schemas_have_no_per_request_model_field():
    """No per-request model routing field exists on the WS TTS schemas."""
    from voicemanager import protocol
    for cls in (protocol.TTSRequestMessage, protocol.TTSStreamMessage,
                protocol.TTSCancelMessage):
        fields = set(cls.model_fields)
        assert "model" not in fields, cls
        assert "model_task" not in fields, cls
        assert "engine" not in fields, cls


def test_sidecar_protocol_version_stays_1():
    """The sidecar wire protocol version stays 1 - no bump for voxcpm2."""
    proto_h = REPO_ROOT / "sidecar" / "src" / "protocol.h"
    assert "constexpr int kProtocolVersion = 1;" in proto_h.read_text(encoding="utf-8")


# --- audiocpp startup loader: fail-fast preset validation -------------------

def _tts_settings(voices: str, model: str = "/models/dots.gguf") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        tts_engine="audiocpp",
        audiocpp_sidecar_bin="sidecar/build/bin/audiocpp_sidecar",
        audiocpp_socket="",
        audiocpp_backend="cuda",
        audiocpp_device=0,
        audiocpp_threads=4,
        audiocpp_tts_model=model,
        audiocpp_tts_model_task="dots-tts-streaming",
        audiocpp_tts_voices=voices,
        audiocpp_tts_voice="designed_female_s101",
        audiocpp_tts_seam_silence_ms=250,
        audiocpp_tts_request_options="text_chunk_size=320,seed=7",
    )


def test_audiocpp_tts_loader_returns_a_loaded_engine():
    result = models._load_audiocpp_tts(_tts_settings(str(REAL_REFS)))
    try:
        assert result["tts_engine_type"] == "audiocpp"
        assert isinstance(result["tts_engine"], AudioCppTTSEngine)
        assert result["tts_executor"] is not None
        # Presets validated at startup: the engine can resolve without touching a sidecar.
        assert result["tts_engine"].sample_rate == 48000
    finally:
        result["tts_executor"].shutdown(wait=False)


def test_audiocpp_tts_seam_silence_ms_reaches_the_spawn_args(monkeypatch):
    """VM_AUDIOCPP_TTS_SEAM_SILENCE_MS flows loader -> engine config ->
    the sidecar spawn argv as --tts-seam-silence-ms."""
    result = models._load_audiocpp_tts(_tts_settings(str(REAL_REFS)))
    result["tts_executor"].shutdown(wait=False)
    engine = result["tts_engine"]
    assert engine._config.get("seam_silence_ms") == 250

    spawned: list[list[str]] = []

    def _fake_popen(argv, *args, **kwargs):
        spawned.append(list(argv))
        return types.SimpleNamespace(pid=4242)

    import voicemanager.pipeline.audiocpp_tts as tts_mod
    monkeypatch.setattr(tts_mod.subprocess, "Popen", _fake_popen)
    engine._client._spawn("/bin/fake_sidecar", "/tmp/fake.sock")
    assert len(spawned) == 1
    argv = spawned[0]
    assert "--tts-seam-silence-ms" in argv
    assert argv[argv.index("--tts-seam-silence-ms") + 1] == "250"
    # Operator request-option overrides ride as repeatable flags,
    # in order, opaque - the sidecar merges them over its registry defaults.
    pairs = [argv[i + 1] for i, a in enumerate(argv) if a == "--request-option"]
    assert pairs == ["text_chunk_size=320", "seed=7"]


def test_audiocpp_tts_request_options_default_empty_sends_no_flags(monkeypatch):
    """An unset VM_AUDIOCPP_TTS_REQUEST_OPTIONS must add nothing to the argv, so
    the registry pins (text_chunk_size=200) stand untouched."""
    settings = _tts_settings(str(REAL_REFS))
    settings.audiocpp_tts_request_options = ""
    result = models._load_audiocpp_tts(settings)
    result["tts_executor"].shutdown(wait=False)
    engine = result["tts_engine"]

    spawned: list[list[str]] = []

    def _fake_popen(argv, *args, **kwargs):
        spawned.append(list(argv))
        return types.SimpleNamespace(pid=4242)

    import voicemanager.pipeline.audiocpp_tts as tts_mod
    monkeypatch.setattr(tts_mod.subprocess, "Popen", _fake_popen)
    engine._client._spawn("/bin/fake_sidecar", "/tmp/fake.sock")
    assert "--request-option" not in spawned[0]


def test_audiocpp_tts_request_options_typo_refuses_startup():
    """A malformed item (no KEY=) fails the loader fast, like the STT twin."""
    settings = _tts_settings(str(REAL_REFS))
    settings.audiocpp_tts_request_options = "text_chunk_size"
    with pytest.raises(ValueError):
        models._load_audiocpp_tts(settings)


def test_audiocpp_tts_loader_fails_fast_on_bad_presets(tmp_path):
    """A refs.json entry lacking ref_text refuses startup, before any
    request is served and before any sidecar exists."""
    bad = tmp_path / "refs.json"
    bad.write_text('{"designed_female_s101": {"file": "a.wav"}}', encoding="utf-8")
    (tmp_path / "a.wav").write_bytes(b"RIFFfake")
    with pytest.raises(PresetLoadError):
        models._load_audiocpp_tts(_tts_settings(str(bad)))


# --- /voice/upload under audiocpp: presets only -----------------------

def _upload_app(tts_engine_type: str) -> TestClient:
    app = create_app()
    app.state.tts_engine_type = tts_engine_type
    app.state.session_registry = SessionRegistry()
    return TestClient(app)


def test_upload_under_audiocpp_returns_400_naming_the_limitation():
    client = _upload_app("audiocpp")
    response = client.post(
        "/voice/upload",
        params={**authkit.auth_params(), "session_id": "s1"},
        headers=authkit.auth_headers(),
        files={"file": ("v.wav", b"RIFF" + b"\x00" * 40, "audio/wav")},
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "audiocpp" in detail
    assert "preset" in detail.lower()


def test_upload_under_openvoice_keeps_its_path():
    """The OpenVoice upload path is untouched: same api-key gate, then the
    session lookup (404 for an unknown session), never the engine 400."""
    client = _upload_app("openvoice")
    response = client.post(
        "/voice/upload",
        params={**authkit.auth_params(), "session_id": "missing"},
        headers=authkit.auth_headers(),
        files={"file": ("v.wav", b"RIFF" + b"\x00" * 40, "audio/wav")},
    )
    assert response.status_code == 404
