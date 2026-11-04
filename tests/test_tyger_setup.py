"""python setup_models.py tyger, with the download mocked, and the sidecar's requirements.

Model-free and network-free. snapshot_download is replaced by a fake that records its
arguments and returns a small fake bundle whose config.json carries the checksums of
its files, the layout the real bundle uses. Covers: the pinned repo and revision are
requested with only the files the sidecar loads, a good bundle passes, a changed or
missing file fails the setup with a message naming it, a local bundle directory is
checked without a download, setup_models does not import torch, and
requirements-tyger.txt lists transformers at 5.13 or newer.
"""
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import authkit  # test auth env; must be imported before voicemanager  # noqa: F401

from voicemanager.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[1]
PINNED_REPO = "nineninesix/tyger"
PINNED_REVISION = "a5d4d73bb74fa4a7082dcd4387e44a1e5ac151b6"
NEMOTRON_FILES = ("config.json", "generation_config.json", "processor_config.json",
                  "tokenizer.json", "tokenizer_config.json")


def _load_setup_models():
    spec = importlib.util.spec_from_file_location(
        "setup_models_under_test", REPO_ROOT / "setup_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup_models(monkeypatch):
    module = _load_setup_models()
    monkeypatch.setattr(module, "get_settings", lambda: Settings(_env_file=None))
    return module


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_bundle(folder: Path) -> Path:
    """A fake bundle: weights, nemotron's files and a config.json with their checksums."""
    (folder / "nemotron").mkdir(parents=True)
    weights = b"fake safetensors bytes"
    (folder / "model.safetensors").write_bytes(weights)
    digests = {}
    for name in NEMOTRON_FILES:
        data = f"{{\"name\": \"{name}\"}}".encode()
        (folder / "nemotron" / name).write_bytes(data)
        digests[name] = _sha(data)
    config = {
        "model_type": "tyger_nemotron_gru",
        "format_version": 1,
        "weights_sha256": _sha(weights),
        "frontend_file_sha256": digests,
    }
    (folder / "config.json").write_text(json.dumps(config))
    return folder


@pytest.fixture
def fake_download(setup_models, monkeypatch, tmp_path):
    """Replace snapshot_download; the bundle it returns and its recorded calls."""
    bundle = _make_bundle(tmp_path / "snapshot")
    calls = []

    def fake_snapshot_download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(bundle)

    monkeypatch.setattr(setup_models, "snapshot_download", fake_snapshot_download)
    return bundle, calls


# --- download ------------------------------------------------------------------------

def test_tyger_is_a_setup_target(setup_models):
    assert "tyger" in setup_models.TARGETS


def test_the_pinned_repo_and_revision_are_requested(setup_models, fake_download):
    _bundle, calls = fake_download
    setup_models.setup_tyger()
    assert len(calls) == 1
    args, kwargs = calls[0]
    repo = kwargs.get("repo_id", args[0] if args else None)
    assert repo == PINNED_REPO
    assert kwargs["revision"] == PINNED_REVISION
    assert sorted(kwargs["allow_patterns"]) == sorted(
        ["config.json", "model.safetensors", "nemotron/*.json"])


def test_the_configured_repo_and_revision_are_requested(setup_models, fake_download,
                                                        monkeypatch):
    _bundle, calls = fake_download
    monkeypatch.setattr(setup_models, "get_settings", lambda: Settings(
        _env_file=None, tyger_model="org/other", tyger_revision="abc123"))
    setup_models.setup_tyger()
    args, kwargs = calls[0]
    assert kwargs.get("repo_id", args[0] if args else None) == "org/other"
    assert kwargs["revision"] == "abc123"


def test_a_good_bundle_passes_its_checksums(setup_models, fake_download, capsys):
    bundle, _calls = fake_download
    setup_models.setup_tyger()
    assert str(bundle) in capsys.readouterr().out


# --- checksums -----------------------------------------------------------------------

def test_changed_weights_fail_with_the_file_named(setup_models, fake_download):
    bundle, _calls = fake_download
    (bundle / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(SystemExit) as exc:
        setup_models.setup_tyger()
    message = str(exc.value.code)
    assert "model.safetensors" in message
    assert "checksum" in message


def test_a_changed_nemotron_file_fails_with_the_file_named(setup_models, fake_download):
    bundle, _calls = fake_download
    (bundle / "nemotron" / "tokenizer.json").write_bytes(b"{}")
    with pytest.raises(SystemExit) as exc:
        setup_models.setup_tyger()
    message = str(exc.value.code)
    assert "nemotron/tokenizer.json" in message
    assert "checksum" in message


def test_a_missing_file_fails_with_the_file_named(setup_models, fake_download):
    bundle, _calls = fake_download
    (bundle / "nemotron" / "processor_config.json").unlink()
    with pytest.raises(SystemExit) as exc:
        setup_models.setup_tyger()
    assert "nemotron/processor_config.json" in str(exc.value.code)


def test_a_config_without_checksums_fails(setup_models, fake_download):
    bundle, _calls = fake_download
    (bundle / "config.json").write_text(json.dumps({"model_type": "tyger_nemotron_gru"}))
    with pytest.raises(SystemExit) as exc:
        setup_models.setup_tyger()
    assert "config.json" in str(exc.value.code)


def test_verify_returns_quietly_on_a_good_bundle(setup_models, tmp_path):
    setup_models.verify_tyger_bundle(_make_bundle(tmp_path / "b"))


def test_a_local_bundle_directory_is_checked_without_a_download(setup_models, monkeypatch,
                                                                tmp_path):
    bundle = _make_bundle(tmp_path / "local")
    monkeypatch.setattr(setup_models, "get_settings",
                        lambda: Settings(_env_file=None, tyger_model=str(bundle)))

    def no_download(*args, **kwargs):
        raise AssertionError("a local bundle must not be downloaded")

    monkeypatch.setattr(setup_models, "snapshot_download", no_download)
    setup_models.setup_tyger()
    (bundle / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(SystemExit, match="model.safetensors"):
        setup_models.setup_tyger()


def test_setup_models_does_not_import_torch():
    """The tyger target runs in the server venv and must stay cheap: no torch."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)])
    code = ("import sys, setup_models; "
            "assert hasattr(setup_models, 'setup_tyger'); "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


# --- the sidecar's requirements ------------------------------------------------------

def _requirement_lines() -> list[str]:
    text = (REPO_ROOT / "requirements-tyger.txt").read_text()
    lines = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def test_requirements_list_transformers_at_5_13_or_newer():
    pins = [line for line in _requirement_lines()
            if re.match(r"transformers\b", line, re.IGNORECASE)]
    assert len(pins) == 1, pins
    match = re.fullmatch(r"transformers\s*(==|>=)\s*(\d+)\.(\d+)(?:\.(\d+))?", pins[0])
    assert match, pins[0]
    major, minor = int(match.group(2)), int(match.group(3))
    assert (major, minor) >= (5, 13)


def test_requirements_cover_the_scorer_and_the_sidecar_server():
    names = {re.split(r"[<>=!~\[ ;@]", line, 1)[0].lower() for line in _requirement_lines()}
    for needed in ("torch", "transformers", "safetensors", "huggingface_hub", "numpy",
                   "librosa", "fastapi", "uvicorn", "websockets"):
        assert needed in names, needed
