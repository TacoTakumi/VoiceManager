"""Preset engines report their voice list and default voice.

Model-free and server-free: the audiocpp engine loads its presets from a temp
refs.json manifest (no sidecar is ever reached), and the VibeVoice engine is
constructed without load(), so no model is touched. The WS and /info checks run
the real app through a TestClient with the engine wired onto app.state by hand.
"""
import json
from pathlib import Path

import authkit  # test auth env; must be imported before voicemanager

import pytest
from fastapi.testclient import TestClient

from voicemanager.app import create_app
from voicemanager.pipeline.audiocpp_tts import DEFAULT_VOICE, AudioCppTTSEngine
from voicemanager.pipeline.vibevoice_tts import VibeVoiceTTSEngine
from voicemanager.session import SessionRegistry
from voicemanager.transport import websocket as ws_module

_PRESETS = ["zeta_voice", DEFAULT_VOICE, "alpha_voice"]


def _write_manifest(tmp_path: Path) -> Path:
    entries: dict = {"_comment": "metadata key, not a preset"}
    for i, name in enumerate(_PRESETS):
        wav = tmp_path / f"{i}.wav"
        wav.write_bytes(b"RIFFfake")
        entries[name] = {"file": wav.name, "ref_text": f"transcript {i}"}
    refs = tmp_path / "refs.json"
    refs.write_text(json.dumps(entries), encoding="utf-8")
    return refs


def _engine(tmp_path: Path, config: dict | None = None) -> AudioCppTTSEngine:
    # No socket and no sidecar_bin: load() only reads the manifest.
    engine = AudioCppTTSEngine({"socket": "", "sidecar_bin": "", **(config or {})})
    engine.load(voices_path=str(_write_manifest(tmp_path)))
    return engine


# --- engine surface -------------------------------------------------------------

def test_audiocpp_available_voices_are_the_manifest_presets(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    assert engine.available_voices == sorted(_PRESETS)


def test_audiocpp_available_voices_empty_before_load() -> None:
    assert AudioCppTTSEngine({"socket": ""}).available_voices == []


def test_audiocpp_default_voice_is_the_built_in_default(tmp_path: Path) -> None:
    assert _engine(tmp_path).default_voice == DEFAULT_VOICE


def test_audiocpp_default_voice_follows_config(tmp_path: Path) -> None:
    engine = _engine(tmp_path, {"voice": "alpha_voice"})
    assert engine.default_voice == "alpha_voice"


def test_audiocpp_session_engine_keeps_voices_and_default(tmp_path: Path) -> None:
    session = _engine(tmp_path, {"voice": "zeta_voice"}).session_engine()
    assert session.available_voices == sorted(_PRESETS)
    assert session.default_voice == "zeta_voice"


def test_vibevoice_default_voice_is_carter_before_load() -> None:
    assert VibeVoiceTTSEngine().default_voice == "Carter"


def test_vibevoice_default_voice_follows_the_configured_voice() -> None:
    engine = VibeVoiceTTSEngine()
    engine._default_voice = "Emma"  # what load(default_voice=...) stores
    assert engine.default_voice == "Emma"


# --- handshake and /info --------------------------------------------------------

class _InstantSTT:
    """Minimal STT stand-in: the receive loop needs an STT pipeline."""

    def __init__(self, asr_model=None, **kwargs):
        pass

    async def feed(self, data: bytes) -> bool:
        return False

    def get_current_text(self):
        return ("", "en", 0)

    async def finalize(self):
        return ("", "en", 0)

    def reset(self) -> None:
        pass

    def dispose(self) -> None:
        pass


@pytest.fixture
def audiocpp_app(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ws_module, "QwenSTTPipeline", _InstantSTT)
    app = create_app()
    app.state.stt_engine_type = "qwen"  # swapped for _InstantSTT above
    app.state.qwen_model = object()
    app.state.tts_engine_type = "audiocpp"
    app.state.tts_engine = _engine(tmp_path)
    app.state.session_registry = SessionRegistry()
    return app


def test_handshake_lists_the_audiocpp_presets(audiocpp_app) -> None:
    client = TestClient(audiocpp_app)
    with authkit.ws_connect(client, "/ws/stream") as ws:
        handshake = ws.receive_json()
    assert handshake["type"] == "ready"
    assert handshake["tts_engine"] == "audiocpp"
    assert handshake["available_voices"] == sorted(_PRESETS)


def test_info_lists_the_audiocpp_presets(audiocpp_app) -> None:
    client = TestClient(audiocpp_app)
    info = client.get("/info", headers=authkit.auth_headers(),
                      params=authkit.auth_params()).json()
    assert info["tts_engine"] == "audiocpp"
    assert info["available_voices"] == sorted(_PRESETS)
