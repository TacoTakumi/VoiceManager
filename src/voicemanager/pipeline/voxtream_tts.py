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
import asyncio
import json
import logging
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import AsyncIterator

import numpy as np
import websockets.asyncio.client
import websockets.exceptions

from voicemanager.pipeline import voxtream_server
from voicemanager.pipeline.audiocpp_tts import PresetVoice, UnknownVoiceError, load_presets
from voicemanager.pipeline.tts import TTSEngine, TTSStreamPipeline, float32_to_pcm16

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

    def load(self, device: str = "", voices_path: str | Path | None = None,
             start_sidecar: bool = True) -> None:
        """Validate the presets, then spawn the sidecar and wait for it.

        ``start_sidecar=False`` loads the presets only, for an engine pointed at a
        sidecar that is already running (tests use a fake one).
        """
        if voices_path is None:
            raise ValueError("voices_path is required: the presets file for this engine")
        self._presets = load_presets(voices_path, ref_text_required=False)
        if self._default_voice not in self._presets:
            raise VoxtreamStartupError(
                f"{VOICE_SETTING}={self._default_voice!r} is not a preset in "
                f"{voices_path}; valid presets: {', '.join(sorted(self._presets))}")
        if start_sidecar:
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

    def open_generation(self, voice: str | None) -> "VoxtreamGeneration":
        """A generation for the preset ``voice`` names; not yet connected.

        Raises UnknownVoiceError before any connection is opened.
        """
        preset = self.resolve_voice(voice)
        return VoxtreamGeneration(self._sidecar.url, preset)

    def synthesize(self, text: str, speaker_key: str = "EN-US", speed: float = 1.0,
                   voice: str | None = None) -> tuple[np.ndarray, int]:
        """One generation for the whole text, collected. Call from a worker thread."""
        async def collect() -> list[np.ndarray]:
            return [frame async for frame in self.stream_synthesize(text, voice=voice)]

        frames = asyncio.run(collect())
        if not frames:
            return np.array([], dtype=np.float32), SAMPLE_RATE
        return np.concatenate(frames), SAMPLE_RATE

    async def stream_synthesize(self, text: str, **kwargs) -> AsyncIterator[np.ndarray]:
        """Yield float32 frames of one generation that speaks the whole text.

        The one-shot ``tts`` request lands here with its complete text; the
        ``tts_stream`` request goes through VoxtreamTTSStreamPipeline instead, which
        feeds the same kind of generation as the chunks arrive.
        """
        generation = self.open_generation(kwargs.get("voice"))
        try:
            await generation.start()  # inside the try: an init error must not leak the socket
            await generation.push(text)
            await generation.end()
            async for frame in generation.frames():
                yield frame
        finally:
            await generation.close()

    def dispose(self) -> None:
        self._sidecar.stop()


class VoxtreamGenerationError(RuntimeError):
    """The sidecar answered a generation with an error frame."""


class VoxtreamGeneration:
    """One generation on the sidecar: one WebSocket connection, text in, frames out.

    ``start`` connects and sends the init frame; ``push`` and ``end`` may be called
    from one task while another task iterates ``frames``. ``close`` drops the
    connection, which is how a cancel stops the sidecar's generation.
    """

    def __init__(self, url: str, preset: PresetVoice) -> None:
        self.url = url
        self.preset = preset
        self.sample_rate = SAMPLE_RATE
        self._ws = None

    async def start(self) -> None:
        self._ws = await websockets.asyncio.client.connect(self.url, max_size=None)
        await self._ws.send(json.dumps({
            "event": "init", "prompt_audio_path": self.preset.wav_path, "full_stream": True,
        }))
        first = await self._ws.recv()
        payload = json.loads(first) if isinstance(first, str) else {}
        if payload.get("type") == "error":
            raise VoxtreamGenerationError(payload.get("message", "sidecar refused the request"))
        if payload.get("type") != "config":
            raise VoxtreamGenerationError(f"unexpected first frame from the sidecar: {first!r}")
        self.sample_rate = int(payload.get("sample_rate", SAMPLE_RATE))

    async def push(self, chunk: str) -> None:
        if chunk and self._ws is not None:
            await self._ws.send(json.dumps({"event": "text", "chunk": chunk}))

    async def end(self) -> None:
        if self._ws is not None:
            await self._ws.send(json.dumps({"event": "eot"}))

    async def frames(self) -> AsyncIterator[np.ndarray]:
        """Yield each binary frame as float32 samples until eos; raise on error."""
        if self._ws is None:
            return
        try:
            async for message in self._ws:
                if isinstance(message, (bytes, bytearray)):
                    yield np.frombuffer(bytes(message), dtype=np.float32)
                    continue
                payload = json.loads(message)
                kind = payload.get("type")
                if kind == "eos":
                    return
                if kind == "error":
                    raise VoxtreamGenerationError(payload.get("message", "generation failed"))
        except websockets.exceptions.ConnectionClosedOK:
            return
        except websockets.exceptions.ConnectionClosedError as exc:
            raise VoxtreamGenerationError(f"sidecar connection lost: {exc}") from exc

    async def close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:
            logger.debug("Closing a VoXtream generation failed", exc_info=True)


class VoxtreamTTSStreamPipeline(TTSStreamPipeline):
    """The tts_stream pipeline for the voxtream engine: one generation per request.

    The first chunk of a request resolves its voice (an unknown voice fails here,
    before anything is opened), starts the synthesis task and opens the generation.
    Every chunk goes straight into that generation as text; ``flush`` has nothing to
    do, since the engine paces itself on the words it has; ``done`` and the flush
    timeout end the text, and the request ends with one audio_end after the
    sidecar's eos. The two chunking values behave the same and the clause settings
    are unused. Cancel, supersede and dispose close the generation through the base
    class's abandon and dispose, which cancel the synthesis task.
    """

    async def feed_text(self, chunk: str, done: bool, websocket, request_id: str,
                        voice: str | None = None, flush: bool = False,
                        chunking: str = "sentence") -> None:
        if self._request_id != request_id:
            self.abandon()
            self._request_id = request_id
            self._text_buffer = ""
            self._done_received = False
            self._audio_started = False
            self._sentence_queue = asyncio.Queue()
            self._voice = voice
            self._chunking = chunking
            # The voice is resolved inside the task: an unknown name must end this
            # request with an error frame, not raise into the receive loop and close
            # the whole connection.
            self._synthesis_task = asyncio.create_task(
                self._run_generation(websocket, request_id,
                                     voice or self._tts_pipeline._voice))
        if self._done_received:
            return  # a frame that arrived after done: the request is closed
        if chunk:
            await self._sentence_queue.put(chunk)
        if done:
            self._done_received = True
            await self._sentence_queue.put(None)
        # flush: nothing to do, the generation already has every word

    async def _run_generation(self, websocket, request_id: str, voice: str | None) -> None:
        from voicemanager.protocol import AudioEndMessage, AudioStartMessage, ErrorMessage

        engine = self._tts_pipeline._engine
        queue = self._sentence_queue
        generation: VoxtreamGeneration | None = None
        pump: asyncio.Task | None = None
        frames = 0

        async def pump_text() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    await generation.end()
                    return
                await generation.push(item)

        try:
            try:
                generation = engine.open_generation(voice)  # before any connection
            except UnknownVoiceError as exc:
                logger.warning("tts_stream request_id=%s: %s", request_id, exc)
                await websocket.send_text(ErrorMessage(
                    error_type="unknown_voice", message=str(exc)).model_dump_json())
                return  # the finally block ends the request on the wire
            await generation.start()
            pump = asyncio.create_task(pump_text())
            async for frame in generation.frames():
                if not self._audio_started:
                    await websocket.send_text(AudioStartMessage(
                        request_id=request_id, sample_rate=generation.sample_rate,
                        channels=1, bit_depth=16, encoding="pcm").model_dump_json())
                    self._audio_started = True
                await websocket.send_bytes(float32_to_pcm16(frame))
                frames += 1
                self._report_audio(request_id, len(frame), generation.sample_rate)
            if pump is not None and not pump.done():
                # eos before the text ended: the sidecar stopped on its own
                pump.cancel()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("VoXtream generation failed for request_id=%s", request_id)
        finally:
            if pump is not None:
                pump.cancel()
            if generation is not None:
                await generation.close()
            if not self._audio_started:
                try:
                    await websocket.send_text(AudioStartMessage(
                        request_id=request_id, sample_rate=engine.sample_rate).model_dump_json())
                except Exception:
                    pass
            try:
                await websocket.send_text(AudioEndMessage(request_id=request_id).model_dump_json())
            except Exception:
                logger.debug("Could not send audio_end for request_id=%s", request_id)
            logger.debug("VoXtream generation complete: request_id=%s frames=%d",
                         request_id, frames)
