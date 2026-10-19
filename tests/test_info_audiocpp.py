"""/info and /health report the audiocpp engine's loaded model.

Model-free and server-free (the CLAUDE.md invariant): TestClient without the
lifespan context, app.state wired by hand exactly as tests/test_audiocpp_crash.py
does. Before this fix both endpoints fell through every engine branch to
models.stt=null under VM_STT_ENGINE=audiocpp, which the HermesConsole wall
renders as NO DATA (live-confirmed 2026-08-14).
"""
import authkit  # test auth env; must be imported before voicemanager

from fastapi.testclient import TestClient

from voicemanager.app import create_app
from voicemanager.session import SessionRegistry

_GGUF = "/home/user/audiocpp-models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q4_k.gguf"
_TASK = "voxtral-mini-4b-realtime-asr-streaming"


def _get(client: TestClient, path: str) -> dict:
    """GET an endpoint with the suite's auth, returning the JSON body."""
    return client.get(path, headers=authkit.auth_headers(),
                      params=authkit.auth_params()).json()


def _app(model: str = _GGUF, config: bool = True):
    app = create_app()
    app.state.stt_engine_type = "audiocpp"
    if config:
        app.state.audiocpp_config = {
            "sidecar_bin": "",
            "socket": "",
            "model": model,
            "model_task": _TASK,
            "backend": "cpu",
            "device": 0,
            "threads": 4,
            "assets_root": "",
        }
    app.state.session_registry = SessionRegistry()
    return app


def test_info_reports_gguf_basename_and_task():
    info = _get(TestClient(_app()), "/info")
    assert info["stt_engine"] == "audiocpp"
    # Basename, not the full path: /info is unauthenticated, the filesystem
    # layout stays server-side; the basename still names the weights + quant.
    assert info["models"]["stt"] == "voxtral-mini-4b-realtime-2602-q4_k.gguf"
    assert info["audiocpp_model"] == "voxtral-mini-4b-realtime-2602-q4_k.gguf"
    assert info["audiocpp_model_task"] == _TASK


def test_info_falls_back_to_model_task_without_gguf_path():
    # The crash/capacity tests run the fake sidecar with model="" -- the task id
    # is still the honest name for what is serving.
    info = _get(TestClient(_app(model="")), "/info")
    assert info["models"]["stt"] == _TASK
    assert info["audiocpp_model"] is None
    assert info["audiocpp_model_task"] == _TASK


def _tts_app(model_task: str):
    # The startup TTS engine validates presets and never opens a sidecar, so its
    # rate has to come from the configured model_task, not a connection.
    from voicemanager.pipeline.audiocpp_tts import AudioCppTTSEngine
    app = _app()
    app.state.tts_engine_type = "audiocpp"
    app.state.tts_engine = AudioCppTTSEngine({"socket": "", "model": "",
                                              "model_task": model_task})
    return app


def test_info_tts_sample_rate_follows_the_configured_model_task():
    breeze = _get(TestClient(_tts_app("breeze-tts-streaming")), "/info")
    assert breeze["tts_engine"] == "audiocpp"
    assert breeze["tts_sample_rate"] == 24000
    dots = _get(TestClient(_tts_app("dots-tts-streaming")), "/info")
    assert dots["tts_sample_rate"] == 48000


def test_info_tts_sample_rate_is_24k_for_qwen3():
    """/info reads the rate from the configured model_task before any sidecar runs."""
    qwen3 = _get(TestClient(_tts_app("qwen3-tts-streaming")), "/info")
    assert qwen3["tts_sample_rate"] == 24000


def test_health_reports_model_task():
    health = _get(TestClient(_app()), "/health")
    assert health["models"]["stt"] == _TASK


def test_unloaded_audiocpp_stays_null():
    # Engine selected but nothing loaded (no lifespan): the honest null the
    # other engines report in the same state, never an invented name.
    info_client = TestClient(_app(config=False))
    info = _get(info_client, "/info")
    assert info["models"]["stt"] is None
    assert info["audiocpp_model"] is None
    assert info["audiocpp_model_task"] is None
    assert _get(info_client, "/health")["models"]["stt"] is None
