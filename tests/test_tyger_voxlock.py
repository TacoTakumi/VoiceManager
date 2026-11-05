"""The vendored VoxLock filter of the experimental pre-filter, with a stubbed onnxruntime.

Model-free and GPU-free. A stub onnxruntime stands in for the real one: its sessions
report the providers the test chooses and run a trivial graph. Covers: the filter
opens both models on the CUDA provider of the sidecar's device; a session left on
the CPU provider, a CPU device, or missing model files each stop it with a message
that says why; the stream's output lines up with its input after the fixed delay;
the licence files sit beside the code; and the sidecar script exits with code 1
when the pre-filter is on and its files are missing.
"""
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from voicemanager.pipeline.tyger import voxlock

TYGER_DIR = Path(voxlock.__file__).resolve().parent.parent
SERVER = TYGER_DIR / "server.py"


class _Input:
    def __init__(self, name, shape):
        self.name, self.shape = name, shape


class _StubSession:
    """enroll -> three fixed arrays; chunk -> one frame per 128 samples, all ones."""

    def __init__(self, path, sess_options=None, providers=None, active=("CUDAExecutionProvider",)):
        self.path = str(path)
        self.requested = providers
        self._active = list(active)

    def get_providers(self):
        return self._active

    def get_inputs(self):
        if self.path.endswith("chunk.onnx"):
            return [_Input("wav", [1, "n"]), _Input("h", [2, 1, 8]), _Input("c", [2, 1, 8])]
        return [_Input("enroll", [1, 48000])]

    def run(self, _outputs, feeds):
        if "enroll" in feeds:
            return [np.zeros(1, np.float32), np.zeros(1, np.float32), np.zeros(1, np.float32)]
        wav = feeds["wav"][0]
        k = 1 + (len(wav) - voxlock.NFFT) // voxlock.HOP
        frames = np.ones((1, k, voxlock.NFFT), np.float32) * voxlock.WIN2
        return [frames, feeds["h"], feeds["c"]]


def _stub_ort(active=("CUDAExecutionProvider",)):
    made = []

    def session(path, sess_options=None, providers=None):
        s = _StubSession(path, sess_options, providers, active)
        made.append(s)
        return s
    mod = types.SimpleNamespace(InferenceSession=session, SessionOptions=lambda: object(),
                                preload_dlls=lambda: None)
    return mod, made


@pytest.fixture
def model_dir(tmp_path):
    for name in voxlock.FILES:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"stub")
    return tmp_path


def test_both_models_open_on_the_cuda_provider_of_the_device(model_dir):
    ort, made = _stub_ort()
    voxlock.VoxLockFactory(model_dir, "cuda:3", ort=ort)
    assert len(made) == 2
    for s in made:
        assert s.requested == [("CUDAExecutionProvider", {"device_id": 3})]


def test_plain_cuda_is_device_zero(model_dir):
    ort, made = _stub_ort()
    voxlock.VoxLockFactory(model_dir, "cuda", ort=ort)
    assert made[0].requested == [("CUDAExecutionProvider", {"device_id": 0})]


def test_a_session_left_on_the_cpu_stops_the_filter_naming_cuda(model_dir):
    ort, _made = _stub_ort(active=("CPUExecutionProvider",))
    with pytest.raises(voxlock.PrefilterLoadError, match="CUDA"):
        voxlock.VoxLockFactory(model_dir, "cuda:3", ort=ort)


def test_a_cpu_device_is_refused_naming_cuda(model_dir):
    ort, made = _stub_ort()
    with pytest.raises(voxlock.PrefilterLoadError, match="CUDA"):
        voxlock.VoxLockFactory(model_dir, "cpu", ort=ort)
    assert made == []


def test_missing_model_files_stop_the_filter_naming_the_setup_step(tmp_path):
    ort, _made = _stub_ort()
    with pytest.raises(voxlock.PrefilterLoadError, match="setup_models.py tyger"):
        voxlock.VoxLockFactory(tmp_path, "cuda:3", ort=ort)


def test_the_files_are_looked_up_in_the_cache_at_the_pinned_revision(monkeypatch, tmp_path):
    seen = {}

    def fake_snapshot(**kw):
        seen.update(kw)
        raise OSError("not cached")
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(snapshot_download=fake_snapshot))
    with pytest.raises(voxlock.PrefilterLoadError, match="setup_models.py tyger"):
        voxlock.resolve_files()
    assert seen["repo_id"] == voxlock.REPO and seen["revision"] == voxlock.REVISION
    assert seen["local_files_only"] is True


def test_the_stream_output_lines_up_with_the_input_after_its_delay(model_dir):
    ort, _made = _stub_ort()
    stream = voxlock.VoxLockFactory(model_dir, "cuda:3", ort=ort).enroll(np.zeros(48000, np.float32))
    assert stream.delay == voxlock.NFFT - voxlock.HOP == 384
    hop = 5120
    first = stream.push(np.zeros(hop, np.float32))
    assert len(first) == hop - stream.delay
    for _ in range(5):
        assert len(stream.push(np.zeros(hop, np.float32))) == hop
    # With every frame all ones under the squared window, overlap-add gives a flat GAIN.
    out = stream.push(np.zeros(hop, np.float32))
    assert np.allclose(out, voxlock.GAIN, atol=1e-5)


def test_enrollment_needs_exactly_3_s(model_dir):
    ort, _made = _stub_ort()
    factory = voxlock.VoxLockFactory(model_dir, "cuda:3", ort=ort)
    with pytest.raises(ValueError):
        factory.enroll(np.zeros(1000, np.float32))


def test_the_licence_files_sit_beside_the_vendored_code():
    here = Path(voxlock.__file__).resolve().parent
    licence = (here / "LICENSE").read_text()
    notice = (here / "NOTICE").read_text()
    assert "Apache License" in licence and "Version 2.0" in licence
    assert "voxlock 1.3.0" in notice and "Arbaz Siddiqui" in notice


def test_the_sidecar_exits_1_when_the_prefilter_files_are_missing(tmp_path):
    script = tmp_path / "script.json"
    script.write_text('{"sessions": {}}')
    env = {"HF_HUB_CACHE": str(tmp_path / "empty-cache"), "HF_HUB_OFFLINE": "1",
           "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        [sys.executable, str(SERVER), "--fake-scorer", str(script), "--port", "1",
         "--prefilter", "voxlock", "--device", "cuda:3"],
        capture_output=True, text=True, timeout=60, env=env)
    assert proc.returncode == 1
    assert "setup_models.py tyger" in proc.stderr
