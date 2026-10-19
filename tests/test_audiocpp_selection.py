"""Engine-selection wiring for the audio.cpp STT backend.

Model-free and server-free (fakes / TestClient, no GPU, no sidecar, no model
download) — the CLAUDE.md test-suite invariant. These tests assert only the
*selection/routing*: that VM_STT_ENGINE=audiocpp routes startup + /ws/stream to
the audio.cpp backend, that each prior engine value is unchanged, and that an
unknown value fails fast. The pipeline itself (over the real sidecar) is out of scope here
and is exercised by its own fake-sidecar tests / the manual live probes.
"""
import sys
import time
import types

import pytest

from fastapi.testclient import TestClient

# Set up the suite auth store before importing app / config (mirrors the other WS tests).
import authkit

import voicemanager.models as models
from voicemanager.app import create_app
from voicemanager.config import Settings
from voicemanager.session import SessionRegistry


# --- load_models dispatch (config-level, no model load) --------------------

@pytest.mark.parametrize("engine", ["whisper", "qwen", "nemotron", "audiocpp"])
def test_load_models_routes_each_engine_to_its_own_loader(engine, monkeypatch):
    """Each VM_STT_ENGINE value dispatches to exactly its own loader.

    The per-engine loaders are replaced with sentinels, so no real model is
    loaded — this asserts the *routing*, which is what the selection wires. The new
    'audiocpp' value routes to _load_audiocpp_models; the three prior values
    keep routing to their existing loaders (no behavior change).
    """
    for name, who in [
        ("_load_whisper_models", "whisper"),
        ("_load_qwen_models", "qwen"),
        ("_load_nemotron_models", "nemotron"),
        ("_load_audiocpp_models", "audiocpp"),
    ]:
        monkeypatch.setattr(
            models, name, lambda s, who=who: {"stt_engine_type": who, "_who": who}
        )

    settings = types.SimpleNamespace(stt_engine=engine)
    result = models.load_models(settings)
    assert result["_who"] == engine
    assert result["stt_engine_type"] == engine


def test_load_models_unknown_engine_fails_fast(monkeypatch):
    """An unknown VM_STT_ENGINE value fails fast instead of silently defaulting.

    Guard the real loaders so a routing bug can't accidentally load a model; the
    dispatch must raise before reaching any of them.
    """
    for name in ("_load_whisper_models", "_load_qwen_models",
                 "_load_nemotron_models", "_load_audiocpp_models"):
        def _boom(s, name=name):
            raise AssertionError(f"{name} must not be called for an unknown engine")
        monkeypatch.setattr(models, name, _boom)

    settings = types.SimpleNamespace(stt_engine="bogus-engine")
    with pytest.raises(ValueError) as exc:
        models.load_models(settings)
    assert "bogus-engine" in str(exc.value)


# --- config surface --------------------------------------------------------

def test_settings_expose_audiocpp_engine_and_per_engine_fields():
    """The audio.cpp engine value + its VM_AUDIOCPP_* settings exist and bind."""
    # _env_file=None: this test asserts the REGISTRY DEFAULT for model_task, so the
    # operator .env (which pins its own VM_AUDIOCPP_MODEL_TASK) must not bind.
    s = Settings(
        _env_file=None,
        stt_engine="audiocpp",
        audiocpp_model="/models/nemotron-3.5-asr-streaming-0.6b-f16.gguf",
        audiocpp_backend="cuda",
        audiocpp_device=2,
        audiocpp_threads=6,
    )
    assert s.stt_engine == "audiocpp"
    assert s.audiocpp_model.endswith("f16.gguf")
    assert s.audiocpp_backend == "cuda"
    assert s.audiocpp_device == 2
    assert s.audiocpp_threads == 6
    # The wire model/task id defaults to the sidecar registry's one entry.
    assert s.audiocpp_model_task == "nemotron-3.5-asr-streaming"


# --- /ws/stream routing (handshake, no lifespan/model load) ----------------

class _FakeAudioCppPipeline:
    """Stand-in for the real AudioCppSTTPipeline: records that the selection
    branch chose it, and satisfies only the handler's connect+teardown contract
    (construct, finalize, reset, dispose)."""

    instances: list = []

    def __init__(self, config=None):
        self.config = config
        _FakeAudioCppPipeline.instances.append(self)

    async def finalize(self):
        return ("", "en", 0)

    def reset(self):
        pass

    def dispose(self):
        pass


def _inject_fake_audiocpp_module(monkeypatch):
    module = types.ModuleType("voicemanager.pipeline.audiocpp_stt")
    module.AudioCppSTTPipeline = _FakeAudioCppPipeline
    monkeypatch.setitem(sys.modules, "voicemanager.pipeline.audiocpp_stt", module)


def test_ws_handshake_reports_audiocpp_and_selects_its_pipeline(monkeypatch):
    """Under VM_STT_ENGINE=audiocpp the handshake advertises stt_engine=audiocpp
    and /ws/stream instantiates the audio.cpp pipeline (routing proof)."""
    _FakeAudioCppPipeline.instances.clear()
    _inject_fake_audiocpp_module(monkeypatch)

    app = create_app()
    app.state.stt_engine_type = "audiocpp"        # what _load_audiocpp_models records
    app.state.audiocpp_config = {"model": "/models/x.gguf"}
    app.state.session_registry = SessionRegistry()
    client = TestClient(app)                        # no `with client:` -> no lifespan

    with authkit.ws_connect(client, "/ws/stream") as ws:
        handshake = ws.receive_json()
        assert handshake["type"] == "ready"
        assert handshake["stt_engine"] == "audiocpp"

    assert len(_FakeAudioCppPipeline.instances) == 1
    assert _FakeAudioCppPipeline.instances[0].config == {"model": "/models/x.gguf"}


# --- model_task selection: two registered ids + unknown ----------------------------
#
# The sidecar registry now serves TWO model_tasks; which one a session gets is purely
# VM_AUDIOCPP_MODEL_TASK -> config["model_task"] -> HELLO. These tests drive the real
# _SidecarClient handshake against the fake sidecar (which validates the id against
# the registry's known set exactly like sidecar/src/main.cpp): both registered ids
# handshake and reach the wire verbatim, an unknown id gets the defined
# unknown_model_task error naming BOTH valid ids, and it stays a permanent
# (do-not-retry, close 4001) config failure.

NEMOTRON_ID = "nemotron-3.5-asr-streaming"
VOXTRAL_ID = "voxtral-mini-4b-realtime-asr-streaming"


def _dial_config(socket_path: str, model_task: str) -> dict:
    """Dial-only config (no spawn): connect to `socket_path` with this model_task."""
    return {
        "sidecar_bin": "",
        "socket": socket_path,
        "model": "",
        "model_task": model_task,
        "backend": "cpu",
        "device": 0,
        "threads": 4,
        "assets_root": "",
    }


def test_settings_bind_the_voxtral_model_task():
    """VM_AUDIOCPP_MODEL_TASK accepts the second registered id (config surface)."""
    s = Settings(stt_engine="audiocpp", audiocpp_model_task=VOXTRAL_ID)
    assert s.audiocpp_model_task == VOXTRAL_ID


@pytest.mark.parametrize("model_task,family", [
    (NEMOTRON_ID, "nemotron_asr"),          # the shipped default, byte-for-byte unchanged
    (VOXTRAL_ID, "voxtral_realtime"),       # the new second model
])
def test_configured_model_task_reaches_hello_and_handshakes(model_task, family):
    """The configured id goes onto the wire verbatim and READY acks it with its family."""
    from voicemanager.pipeline.audiocpp_stt import _SidecarClient

    from audiocpp_fake import FakeSidecar

    with FakeSidecar() as fake:
        client = _SidecarClient(_dial_config(fake.socket_path, model_task))
        try:
            client.connect()
            assert fake.connections[0].hello["model_task"] == model_task
            assert client._ready["model_task"] == model_task
            assert client._ready["family"] == family
        finally:
            client.close()


def test_unknown_model_task_fails_with_the_defined_error_naming_both_ids():
    """An unregistered id is refused with unknown_model_task, its message naming BOTH
    valid ids (the operator's fix is right there), and stays a permanent config
    failure -- the do-not-retry 4001 class, not a retriable fault."""
    from voicemanager.pipeline.audiocpp_stt import SidecarError, _SidecarClient

    from audiocpp_fake import FakeSidecar

    with FakeSidecar() as fake:
        client = _SidecarClient(_dial_config(fake.socket_path, "no-such-model-task"))
        with pytest.raises(SidecarError) as excinfo:
            client.connect()
    assert excinfo.value.code == "unknown_model_task"
    assert NEMOTRON_ID in excinfo.value.message, "the error must name the nemotron id"
    assert VOXTRAL_ID in excinfo.value.message, "the error must name the voxtral id"
    assert excinfo.value.is_permanent is True, (
        "unknown_model_task is an operator-configuration failure and must keep the "
        "do-not-retry contract")


def test_ws_stream_routes_the_voxtral_model_task_to_the_sidecar(monkeypatch):
    """End to end, model-free: VM_AUDIOCPP_MODEL_TASK=<voxtral id> on /ws/stream puts
    the voxtral id in the sidecar HELLO -- the whole model routing, since
    which model serves a session is exactly which model_task its HELLO requests."""
    import voicemanager.config as cfg

    from audiocpp_fake import FakeSidecar

    monkeypatch.setenv("VM_SESSION_TIMEOUT", "30")
    cfg._settings = None
    with FakeSidecar() as fake:
        app = create_app()
        app.state.stt_engine_type = "audiocpp"
        app.state.audiocpp_config = _dial_config(fake.socket_path, VOXTRAL_ID)
        app.state.session_registry = SessionRegistry()
        client = TestClient(app)
        with authkit.ws_connect(client, "/ws/stream") as ws:
            assert ws.receive_json()["type"] == "ready"
            ws.send_bytes(b"\x11\x22" * 512)  # first feed triggers the lazy connect
            deadline = time.monotonic() + 5.0
            while not fake.connections and time.monotonic() < deadline:
                time.sleep(0.01)
    assert fake.connections, "the WS audio frame never reached the fake sidecar"
    assert fake.connections[0].hello["model_task"] == VOXTRAL_ID
