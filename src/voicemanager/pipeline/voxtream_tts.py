"""VoXtream TTS engine: one continuous generation per streamed request.

VoXtream (herimor/voxtream) reads text at the speaking rate: a monotonic phoneme
pointer with a few phonemes of lookahead, so words can arrive while it speaks and
one generation carries a whole reply with no per-piece seams. It pins an older torch
than VoiceManager, so it runs as a separate process from its own virtual environment:
``VoxtreamSidecar`` spawns ``voxtream_server.py`` with that environment's python,
waits for its ``/health`` and stops it at shutdown. ``VoxtreamTTSEngine`` is the
engine object the pipelines hold; it resolves preset voices and opens one WebSocket
connection to the sidecar per generation.

The voice is a preset from the designed presets file (the reference WAV only; the
transcript is unused). Cloning from an uploaded WAV is not offered on this engine.
"""
import logging
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from voicemanager.pipeline import voxtream_server
from voicemanager.pipeline.audiocpp_tts import PresetVoice, UnknownVoiceError, load_presets
from voicemanager.pipeline.tts import TTSEngine

logger = logging.getLogger(__name__)

SAMPLE_RATE = voxtream_server.SAMPLE_RATE
SERVER_SCRIPT = Path(voxtream_server.__file__)
# Settings named in startup errors, so the operator knows which line of .env to fix.
PY_SETTING = "VM_VOXTREAM_PY"
READY_TIMEOUT_SETTING = "VM_VOXTREAM_READY_TIMEOUT"
VOICE_SETTING = "VM_VOXTREAM_VOICE"
_STOP_GRACE_S = 5.0
_HEALTH_POLL_S = 0.1


class VoxtreamStartupError(RuntimeError):
    """The sidecar could not be started; the message names the setting at fault."""


class VoxtreamSidecar:
    """The VoXtream process: spawn, wait for ready, stop."""

    def __init__(self, python: str | Path, port: int, device: str | None = None,
                 ready_timeout_s: float = 180.0, compile_model: bool = False,
                 host: str = "127.0.0.1", server_script: str | Path | None = None,
                 extra_args: list[str] | None = None) -> None:
        self.python = Path(python).expanduser()
        self.port = int(port)
        self.host = host
        self.device = device
        self.ready_timeout_s = float(ready_timeout_s)
        self.compile_model = compile_model
        # Resolved at call time so a test can point the module at a fake script.
        self.server_script = Path(server_script) if server_script else Path(SERVER_SCRIPT)
        self.extra_args = list(extra_args or [])
        self.process: subprocess.Popen | None = None

    @staticmethod
    def device_argument(resolved_device: str) -> str | None:
        """Map a resolved VM_DEVICE to the sidecar's --device value.

        ``cuda:N`` pins GPU N; ``cuda`` inherits whatever CUDA_VISIBLE_DEVICES says
        (None, no argument); ``cpu`` hides every GPU with -1.
        """
        if resolved_device.startswith("cuda:"):
            return resolved_device.split(":", 1)[1]
        if resolved_device == "cpu":
            return "-1"
        return None

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}/voxtream"

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    def command(self) -> list[str]:
        cmd = [str(self.python), str(self.server_script), "--host", self.host,
               "--port", str(self.port)]
        if self.device is not None:
            cmd += ["--device", str(self.device)]
        if self.compile_model:
            cmd.append("--compile")
        return cmd + self.extra_args

    def start(self) -> None:
        """Spawn the sidecar and block until /health answers 200.

        Raises VoxtreamStartupError naming VM_VOXTREAM_PY when the interpreter is
        missing or the process dies, and naming VM_VOXTREAM_READY_TIMEOUT when the
        process lives but is not ready in time. Nothing is left running on failure.
        """
        if not self.python.is_file():
            raise VoxtreamStartupError(
                f"{PY_SETTING}={self.python} is not a file; it must be the python of "
                "the VoXtream virtual environment (see the README section on the "
                "voxtream engine)")
        if self.process is not None and self.process.poll() is None:
            return
        cmd = self.command()
        logger.info("Starting VoXtream sidecar: %s", " ".join(cmd))
        try:
            self.process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
        except OSError as exc:
            raise VoxtreamStartupError(
                f"{PY_SETTING}={self.python} could not be started: {exc}") from exc
        deadline = time.monotonic() + self.ready_timeout_s
        while True:
            code = self.process.poll()
            if code is not None:
                self.process = None
                raise VoxtreamStartupError(
                    f"the VoXtream sidecar exited with code {code} during startup "
                    f"({PY_SETTING}={self.python}); its log says why")
            if self.is_ready():
                logger.info("VoXtream sidecar ready at %s", self.url)
                return
            if time.monotonic() >= deadline:
                self.stop()
                raise VoxtreamStartupError(
                    f"the VoXtream sidecar was not ready within "
                    f"{READY_TIMEOUT_SETTING}={self.ready_timeout_s:g} s; the first run "
                    "downloads the weights, so raise the timeout or check its log")
            time.sleep(_HEALTH_POLL_S)

    def is_ready(self) -> bool:
        try:
            with urllib.request.urlopen(self.health_url, timeout=1.0) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def stop(self) -> None:
        """Terminate the process, then kill it if it ignores the request."""
        proc = self.process
        if proc is None:
            return
        self.process = None
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=_STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=_STOP_GRACE_S)
        logger.info("VoXtream sidecar stopped")

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None


class VoxtreamTTSEngine(TTSEngine):
    """The engine object: presets plus the sidecar it speaks through.

    Loading validates the presets and starts the sidecar; nothing from VoXtream is
    imported into this process. Synthesis runs over the sidecar's WebSocket.
    """

    is_batch = False

    def __init__(self, sidecar: VoxtreamSidecar, default_voice: str) -> None:
        self._sidecar = sidecar
        self._default_voice = default_voice
        self._presets: dict[str, PresetVoice] = {}

    @property
    def sidecar(self) -> VoxtreamSidecar:
        return self._sidecar

    def load(self, device: str = "", voices_path: str | Path | None = None) -> None:
        """Validate the presets, then spawn the sidecar and wait for it."""
        if voices_path is None:
            raise ValueError("voices_path is required: the presets file for this engine")
        self._presets = load_presets(voices_path, ref_text_required=False)
        if self._default_voice not in self._presets:
            raise VoxtreamStartupError(
                f"{VOICE_SETTING}={self._default_voice!r} is not a preset in "
                f"{voices_path}; valid presets: {', '.join(sorted(self._presets))}")
        self._sidecar.start()

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def available_voices(self) -> list[str]:
        return sorted(self._presets)

    @property
    def default_voice(self) -> str:
        return self._default_voice

    def resolve_voice(self, voice: str | None) -> PresetVoice:
        """The preset for a request's voice; no name means the engine default.

        Raises UnknownVoiceError listing the valid presets, never a silent fallback.
        """
        name = voice or self._default_voice
        preset = self._presets.get(name)
        if preset is None:
            raise UnknownVoiceError(
                f"unknown voice {name!r}; valid presets: {', '.join(sorted(self._presets))}")
        return preset

    def synthesize(self, text: str, speaker_key: str = "EN-US", speed: float = 1.0,
                   voice: str | None = None) -> tuple[np.ndarray, int]:
        raise NotImplementedError("voxtream synthesis arrives with the stream pipeline")

    async def stream_synthesize(self, text: str, **kwargs) -> AsyncIterator[np.ndarray]:
        raise NotImplementedError("voxtream synthesis arrives with the stream pipeline")
        yield  # pragma: no cover

    def dispose(self) -> None:
        self._sidecar.stop()
