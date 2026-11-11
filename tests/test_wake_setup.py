"""python setup_models.py wake, with the download mocked, and the wake requirements file.

Model-free and network-free: the download is replaced by a function that writes a
small archive with the model's file names. Covers: wake is a setup target; the pinned
archive is asked for; the archive is checked against the pinned SHA-256 before
anything is installed; only the files the spotter loads are written, by name; a
directory that has them is left alone; a missing setting, a failed download and an
archive without a file each stop with a message; the packages are offered and their
absence does not stop the download; and the manual probe script is not collected.
"""
import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

import authkit  # test auth env; must be imported before voicemanager  # noqa: F401

from voicemanager.config import Settings
from voicemanager.wake import find_model_files

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_setup_models():
    spec = importlib.util.spec_from_file_location(
        "setup_models_under_test", REPO_ROOT / "setup_models.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _archive(module, *, leave_out: str | None = None, extra: dict | None = None) -> bytes:
    """A .tar.bz2 with the model's files under its top directory, as the release has."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        names = [n for n in module.WAKE_MODEL_FILES if n != leave_out]
        # Files of the real archive that the spotter does not load.
        names += ["encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx", "README.md",
                  "test_wavs/0.wav"]
        members = {f"{module.WAKE_MODEL}/{name}": f"bytes of {name}".encode() for name in names}
        members.update(extra or {})
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture
def setup_models(monkeypatch, tmp_path):
    """setup_models with VM_WAKE_MODEL_DIR set, the packages present, and no network."""
    module = _load_setup_models()
    module.model_dir = tmp_path / "kws"
    monkeypatch.setattr(module, "get_settings", lambda: Settings(
        _env_file=None, wake_model_dir=str(module.model_dir)))
    module.asked = []
    monkeypatch.setattr(module, "ensure_requirements",
                        lambda engine: module.asked.append(engine) or True)

    def no_network(url, dest):
        raise AssertionError(f"unexpected download of {url}")

    monkeypatch.setattr(module, "_fetch", no_network)
    return module


def _serve(monkeypatch, module, data: bytes, *, pin: bool = True) -> list:
    """Make the download write ``data``; with ``pin`` its SHA-256 is the pinned one."""
    urls = []

    def fetch(url, dest):
        urls.append(url)
        Path(dest).write_bytes(data)

    monkeypatch.setattr(module, "_fetch", fetch)
    if pin:
        monkeypatch.setattr(module, "WAKE_MODEL_SHA256", hashlib.sha256(data).hexdigest())
    return urls


def test_wake_is_a_setup_target(setup_models):
    assert setup_models.TARGETS["wake"] is setup_models.setup_wake


def test_the_pinned_archive_is_downloaded_and_the_spotter_files_are_installed(
        setup_models, monkeypatch, capsys):
    urls = _serve(monkeypatch, setup_models, _archive(setup_models))
    setup_models.setup_wake()
    assert urls == [setup_models.WAKE_MODEL_URL]
    assert urls[0].startswith("https://github.com/k2-fsa/sherpa-onnx/releases/download/")
    assert urls[0].endswith(f"{setup_models.WAKE_MODEL}.tar.bz2")
    folder = setup_models.model_dir
    assert sorted(p.name for p in folder.iterdir()) == sorted(setup_models.WAKE_MODEL_FILES)
    for name in setup_models.WAKE_MODEL_FILES:
        assert (folder / name).read_bytes() == f"bytes of {name}".encode()
    # What was installed is what the server's loader looks for.
    assert set(find_model_files(str(folder))) == {
        "encoder", "decoder", "joiner", "tokens.txt", "bpe.model"}
    assert "ready" in capsys.readouterr().out


def test_the_pinned_checksum_is_a_sha256_digest(setup_models):
    assert len(setup_models.WAKE_MODEL_SHA256) == 64
    int(setup_models.WAKE_MODEL_SHA256, 16)


def test_an_archive_with_another_checksum_installs_nothing(setup_models, monkeypatch):
    _serve(monkeypatch, setup_models, _archive(setup_models), pin=False)
    with pytest.raises(SystemExit) as stop:
        setup_models.setup_wake()
    assert "SHA-256" in str(stop.value)
    assert not setup_models.model_dir.exists()


@pytest.mark.parametrize("name", ["tokens.txt", "bpe.model",
                                  "joiner-epoch-12-avg-2-chunk-16-left-64.onnx"])
def test_an_archive_without_a_file_installs_nothing(setup_models, monkeypatch, name):
    _serve(monkeypatch, setup_models, _archive(setup_models, leave_out=name))
    with pytest.raises(SystemExit) as stop:
        setup_models.setup_wake()
    assert name in str(stop.value)
    assert not setup_models.model_dir.exists()


def test_a_member_with_a_path_outside_the_directory_is_not_written(
        setup_models, monkeypatch, tmp_path):
    """Files are taken out by name, so a member of the archive cannot place itself."""
    _serve(monkeypatch, setup_models, _archive(
        setup_models, extra={"../escaped.txt": b"x", "/tmp/absolute-escaped.txt": b"x"}))
    setup_models.setup_wake()
    assert not (tmp_path / "escaped.txt").exists()
    assert sorted(p.name for p in setup_models.model_dir.iterdir()) == sorted(
        setup_models.WAKE_MODEL_FILES)


def test_a_directory_that_has_the_files_is_left_alone(setup_models, capsys):
    folder = setup_models.model_dir
    folder.mkdir()
    for name in setup_models.WAKE_MODEL_FILES:
        (folder / name).write_text("mine")
    setup_models.setup_wake()          # the fixture's download raises if it is called
    assert all((folder / n).read_text() == "mine" for n in setup_models.WAKE_MODEL_FILES)
    assert "already" in capsys.readouterr().out


def test_a_failed_download_stops_with_the_url(setup_models, monkeypatch):
    def fetch(url, dest):
        raise OSError("network is unreachable")

    monkeypatch.setattr(setup_models, "_fetch", fetch)
    with pytest.raises(SystemExit) as stop:
        setup_models.setup_wake()
    assert setup_models.WAKE_MODEL_URL in str(stop.value)
    assert "network is unreachable" in str(stop.value)


def test_no_model_directory_setting_stops_with_the_setting_named(setup_models, monkeypatch):
    monkeypatch.setattr(setup_models, "get_settings", lambda: Settings(_env_file=None))
    with pytest.raises(SystemExit) as stop:
        setup_models.setup_wake()
    assert "VM_WAKE_MODEL_DIR" in str(stop.value)


def test_the_packages_are_offered_after_the_model_and_do_not_gate_it(
        setup_models, monkeypatch):
    _serve(monkeypatch, setup_models, _archive(setup_models))
    monkeypatch.setattr(setup_models, "ensure_requirements",
                        lambda engine: setup_models.asked.append(engine) and False)
    setup_models.setup_wake()
    assert setup_models.asked == ["wake"]
    assert (setup_models.model_dir / "tokens.txt").is_file()


def test_the_wake_requirements_name_the_two_packages(setup_models):
    modules, req_file = setup_models.ENGINE_REQUIREMENTS["wake"]
    assert modules == ("sherpa_onnx", "sentencepiece")
    lines = [line.split("#")[0].strip()
             for line in (REPO_ROOT / req_file).read_text().splitlines()]
    packages = [line for line in lines if line]
    assert any(p.startswith("sherpa-onnx") for p in packages)
    assert "sentencepiece" in packages
    assert len(packages) == 2


def test_the_default_requirements_do_not_install_the_wake_engine():
    """The wake engine is optional: a default install does not pull sherpa-onnx."""
    text = (REPO_ROOT / "requirements.txt").read_text()
    assert "sherpa" not in text.lower()


def test_the_probe_is_a_manual_script_and_not_a_test():
    probe = REPO_ROOT / "scripts" / "wake_probe.py"
    assert probe.is_file()
    # pytest collects test_*.py under tests/ only: the probe needs the real model.
    assert not probe.name.startswith("test_")
    assert "SherpaWakeSpotter" in probe.read_text()
