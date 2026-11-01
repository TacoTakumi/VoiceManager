"""/info and /health name the voxtream engine; /health reports its sidecar's state.

Model-free and server-free: TestClient without the lifespan, app.state wired by
hand with a voxtream engine that never starts its sidecar. Before this fix both
endpoints fell through the engine branches to the openvoice label under
VM_TTS_ENGINE=voxtream (seen live on the rig's /health). /health gets the engine
and model label the other engines get; /info keeps its documented convention and
reports the bare model id, here the repository the shipped generator config loads.
Both come from one engine table in the transport module.

The engine's ``ready`` flag (false while the sidecar is dead or starting again)
drives /health: 503 with status degraded and tts null while false, the usual 200
ok body once true. The flag is patched at the class here; the watcher tests cover
what sets it.
"""
import inspect
import sys
from pathlib import Path

import authkit  # test auth env; must be imported before voicemanager
from fastapi.testclient import TestClient

from voicemanager.app import create_app
from voicemanager.pipeline.voxtream_tts import VoxtreamSidecar, VoxtreamTTSEngine
from voicemanager.session import SessionRegistry
from voicemanager.transport import http as http_module

REAL_REFS = Path(__file__).resolve().parent.parent / "voices" / "designed" / "refs.json"
REPO = "herimor/voxtream2.1"
LABEL = f"voxtream ({REPO})"


def _get(client: TestClient, path: str) -> dict:
    return client.get(path, headers=authkit.auth_headers(),
                      params=authkit.auth_params()).json()


def _app(monkeypatch=None, ready: dict | None = None):
    """An app wired by hand; ``ready`` is a holder the test flips the flag with."""
    if monkeypatch is not None:
        monkeypatch.setattr(VoxtreamTTSEngine, "ready",
                            property(lambda self: ready["ready"]))
    app = create_app()
    app.state.stt_engine_type = "whisper"
    app.state.tts_engine_type = "voxtream"
    engine = VoxtreamTTSEngine(VoxtreamSidecar(sys.executable, 1), default_voice="designed_female_s101")
    engine.load(voices_path=REAL_REFS, start_sidecar=False)
    app.state.tts_engine = engine
    app.state.session_registry = SessionRegistry()
    return app


def test_health_names_the_voxtream_engine_and_its_model_repo(monkeypatch):
    client = TestClient(_app(monkeypatch, {"ready": True}))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["models"]["tts"] == LABEL


def test_health_is_503_degraded_while_the_sidecar_is_not_ready_and_recovers(monkeypatch):
    flag = {"ready": True}
    client = TestClient(_app(monkeypatch, flag))
    ready_body = client.get("/health").json()

    flag["ready"] = False
    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["models"]["tts"] is None
    assert body["models"]["stt"] == ready_body["models"]["stt"]
    assert body["models"]["vad"] == ready_body["models"]["vad"]
    assert set(body) == set(ready_body) and set(body["models"]) == set(ready_body["models"])

    flag["ready"] = True
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == ready_body


def test_health_without_the_sidecar_started_is_degraded():
    # No patching: an engine whose sidecar was never started reports not ready.
    response = TestClient(_app()).get("/health")
    assert response.status_code == 503
    assert response.json()["models"]["tts"] is None


def test_info_names_the_voxtream_engine_the_same_way():
    info = _get(TestClient(_app()), "/info")
    assert info["tts_engine"] == "voxtream"
    assert info["models"]["tts"] == LABEL
    assert info["tts_model"] == REPO
    assert info["tts_sample_rate"] == 24000


def test_the_label_comes_from_the_shipped_generator_config():
    import json
    from voicemanager.pipeline.voxtream_server import CONFIG_DIR
    assert json.loads((CONFIG_DIR / "generator.json").read_text())["model_repo"] == REPO


def test_info_and_health_share_one_engine_table():
    # The /info body must not carry its own copy of the engine table: the
    # openvoice fallback literal appears once in the module, inside the id helper
    # that the /health label helper wraps.
    source = inspect.getsource(http_module)
    assert source.count('"openvoice-v2"') == 1
    assert '"openvoice-v2"' in inspect.getsource(http_module._tts_model_id)
    assert "_tts_model_id(" in inspect.getsource(http_module._tts_model_name)
