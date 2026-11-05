"""VoxLock, the target-speaker filter of the experimental pre-filter.

Adapted from voxlock 1.3.0 (voxlock/core.py, https://github.com/arbazsiddiqui/voxlock),
Apache-2.0; see LICENSE and NOTICE in this directory. Kept: the enroll and chunk
ONNX models (fp32) and the streaming overlap-add. Changed: both models run on the
CUDA provider of the sidecar's device and nowhere else; the files come from the
Hugging Face cache, where ``python setup_models.py tyger`` puts them.

``VoxLockFactory`` is the pre-filter's filter factory: ``enroll`` takes exactly 3 s
of the user's voice (16 kHz float32) and returns a ``VoxLockStream``, whose
``push`` takes any number of samples and returns the filtered samples that are
ready. Joined, the output lines up with the input sample for sample; the last
``delay`` samples (384, 24 ms) are always still held back.
"""
import os
from pathlib import Path

import numpy as np

REPO = "arbazsiddiqui/voxlock"
REVISION = "2413b292bc2c337bfc19a9802aaf3e011e68e9c4"
FILES = ("onnx/v2/enroll.onnx", "onnx/v2/chunk.onnx")

SR, HOP, NFFT, PAD = 16000, 128, 512, 256
SAMPLE = 3 * SR                    # the network takes a voice sample of exactly 3 s
DELAY = NFFT - HOP                 # samples a push holds back once the stream runs
WIN2 = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / NFFT)).astype(np.float32) ** 2
# The network returns the caller at about -2x (trained with a scale-invariant loss);
# this restores the input level.
GAIN = -0.5
FRAMES_PER_RUN = 100
_CUDA = "CUDAExecutionProvider"


class PrefilterLoadError(RuntimeError):
    """The pre-filter cannot run; the message says why and what to do."""


def resolve_files(repo: str = REPO, revision: str = REVISION) -> Path:
    """The folder holding the two model files in the Hugging Face cache. Nothing is
    downloaded."""
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=repo, revision=revision, allow_patterns=list(FILES),
                                      local_files_only=True))
    except Exception as exc:
        raise PrefilterLoadError(
            f"the VoxLock models {repo} at {revision} are not in the Hugging Face cache "
            f"({exc}); fetch them with 'python setup_models.py tyger'") from exc


def _device_id(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:") and device[5:].isdigit():
        return int(device[5:])
    raise PrefilterLoadError(f"the pre-filter runs on a CUDA device only, not {device!r}")


class VoxLockFactory:
    """Both models on the CUDA provider, shared by every session's stream."""

    def __init__(self, folder, device: str, ort=None) -> None:
        folder = Path(folder)
        index = _device_id(device)
        missing = [name for name in FILES if not (folder / name).is_file()]
        if missing:
            raise PrefilterLoadError(
                f"the VoxLock model files {', '.join(missing)} are missing from {folder}; "
                "fetch them with 'python setup_models.py tyger'")
        # cuda:N counts GPUs in PCI bus order, as the scorer's device does.
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        if ort is None:
            import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()     # the CUDA libraries the onnxruntime-gpu extras installed
        providers = [(_CUDA, {"device_id": index})]
        self.enroll_session = ort.InferenceSession(str(folder / FILES[0]), providers=providers)
        self.chunk_session = ort.InferenceSession(str(folder / FILES[1]), providers=providers)
        for session in (self.enroll_session, self.chunk_session):
            active = list(session.get_providers())
            if not active or active[0] != _CUDA:
                raise PrefilterLoadError(
                    f"the VoxLock model {session_path(session)} did not start on CUDA "
                    f"(providers {active}); the pre-filter never runs on the CPU. Check that "
                    "onnxruntime-gpu and its cuda and cudnn extras are installed in the "
                    "sidecar's environment")
        inputs = {i.name: i.shape for i in self.chunk_session.get_inputs()}
        self.state_shape = tuple(int(d) for d in inputs["h"])

    @classmethod
    def load(cls, device: str) -> "VoxLockFactory":
        return cls(resolve_files(), device)

    def enroll(self, sample) -> "VoxLockStream":
        sample = np.asarray(sample, np.float32)
        if len(sample) != SAMPLE:
            raise ValueError(f"the voice sample must be exactly {SAMPLE} samples, got {len(sample)}")
        voice = self.enroll_session.run(None, {"enroll": sample[None]})
        return VoxLockStream(self, dict(zip(("enroll_n", "k", "v"), voice)))


def session_path(session) -> str:
    return getattr(session, "path", None) or getattr(session, "_model_path", "") or "?"


class VoxLockStream:
    """One session's filter state: the LSTM state and the overlap-add tail."""

    delay = DELAY

    def __init__(self, factory: VoxLockFactory, voice: dict) -> None:
        self._chunk = factory.chunk_session
        self._voice = voice
        self._h = np.zeros(factory.state_shape, np.float32)
        self._c = np.zeros_like(self._h)
        self._head = np.zeros(0, np.float32)   # first samples, until the start padding can be built
        self._buf = None                       # padded input no frame has consumed yet
        self._ola = np.zeros(NFFT, np.float32)
        self._env = np.zeros(NFFT, np.float32)
        self._dropped = 0                      # leading padded samples still to drop

    def push(self, samples) -> np.ndarray:
        samples = np.asarray(samples, np.float32)
        if self._buf is None:
            self._head = np.concatenate([self._head, samples])
            if len(self._head) <= PAD:
                return np.zeros(0, np.float32)
            # torch.stft centre padding (reflect)
            self._buf = np.concatenate([self._head[1:PAD + 1][::-1], self._head])
            self._dropped = PAD
        else:
            self._buf = np.concatenate([self._buf, samples])
        return self._run()

    def _run(self) -> np.ndarray:
        out = []
        while len(self._buf) >= NFFT:
            k = min(FRAMES_PER_RUN, 1 + (len(self._buf) - NFFT) // HOP)
            seg = self._buf[:(k - 1) * HOP + NFFT]
            fr, self._h, self._c = self._chunk.run(
                None, {"wav": seg[None], **self._voice, "h": self._h, "c": self._c})
            ola = np.concatenate([self._ola, np.zeros(k * HOP, np.float32)])
            env = np.concatenate([self._env, np.zeros(k * HOP, np.float32)])
            for j in range(k):
                ola[j * HOP:j * HOP + NFFT] += fr[0, j]
                env[j * HOP:j * HOP + NFFT] += WIN2
            done = k * HOP                       # samples no later frame will touch
            y = ola[:done] / np.where(env[:done] > 1e-11, env[:done], 1.0)
            self._ola, self._env = ola[done:], env[done:]
            self._buf = self._buf[done:]
            cut = min(self._dropped, len(y))
            self._dropped -= cut
            out.append(y[cut:] * GAIN)
        return np.concatenate(out) if out else np.zeros(0, np.float32)
