"""VoXtream sidecar server: one streaming TTS generation per WebSocket connection.

This file runs inside the VoXtream virtual environment, by file path, not inside
VoiceManager's own environment (VoXtream pins an older torch). It therefore imports
nothing from the voicemanager package and imports voxtream itself only in ``main()``.
VoiceManager's tests import ``create_app`` and inject a fake generator.

Wire protocol (the shapes of VoXtream's own ``voxtream-server``, kept so its client
works unchanged):

- The client connects to ``/voxtream`` and sends one JSON text frame
  ``{"event": "init", "prompt_audio_path": "...", "full_stream": true}``. An optional
  ``"text"`` field is pushed first; with ``full_stream`` false it is the whole text.
- Then any number of ``{"event": "text", "chunk": "..."}`` frames and one
  ``{"event": "eot"}`` when the text is finished.
- The server answers with exactly one
  ``{"type": "config", "sample_rate": 24000, "dtype": "float32", "channels": 1}``,
  then one binary frame of little-endian float32 samples per generated audio frame,
  then ``{"type": "eos"}``. A failure inside generation sends
  ``{"type": "error", "message": "..."}`` instead of eos.

Differences from VoXtream's demo server, all deliberate: the config frame is sent
once; generations run one at a time (a second connection waits for the first to end);
a connection that closes early stops its generation within one frame instead of
leaving a thread blocked; the config files are read from a fixed directory, not the
working directory; the default bind address is loopback.

Readiness: ``GET /health`` answers 200 once the model is loaded and 503 before.

    python voxtream_server.py --port 8256 --device 3
"""
import argparse
import asyncio
import json
import logging
import os
import queue
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

logger = logging.getLogger("voxtream_server")

SAMPLE_RATE = 24000
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8256
CONFIG_DIR = Path(__file__).with_name("voxtream_configs")
# How long the connection handler waits for the generation thread to wind down after
# a cancel before giving the generation lock back. A frame is 80 ms; the generator
# polls its text feed at least once per frame, so one second is generous.
_WORKER_JOIN_S = 5.0


class GenerationCancelled(Exception):
    """Raised inside the generation thread when the client went away."""


class TextFeed:
    """The text iterator VoXtream's generator polls, fed from the socket.

    VoXtream calls ``next()`` at least once per audio frame and treats ``None`` as
    "no new text yet", so ``__next__`` never blocks: it returns a queued chunk, or
    None, raises StopIteration once the text has ended and the queue is drained, and
    raises GenerationCancelled on the first poll after ``cancel()``.
    """

    def __init__(self) -> None:
        self._q: queue.Queue[str | None] = queue.Queue()
        self._ended = False
        self._cancelled = False

    def push(self, chunk: str) -> None:
        if chunk:
            self._q.put(chunk)

    def end(self) -> None:
        self._ended = True

    def cancel(self) -> None:
        self._cancelled = True

    def __iter__(self) -> "TextFeed":
        return self

    def __next__(self) -> str | None:
        if self._cancelled:
            raise GenerationCancelled()
        try:
            return self._q.get_nowait()
        except queue.Empty:
            if self._ended:
                raise StopIteration
            return None


@dataclass
class _Done:
    error: str | None


def _run_generation(generator: Any, prompt_path: Path, feed: TextFeed,
                    out: "asyncio.Queue[Any]", loop: asyncio.AbstractEventLoop,
                    cancelled: threading.Event) -> None:
    """Generation thread body: pull frames from the generator into ``out``.

    Ends on eos, on an error, or on cancel. The generator is closed from this same
    thread (it is suspended at a yield here, never executing), so no further frame
    can be produced after this function returns.
    """
    gen = None
    error: str | None = None
    try:
        gen = generator.generate_stream(prompt_audio_path=prompt_path, text=feed)
        for frame, _meta in gen:
            if cancelled.is_set():
                break
            loop.call_soon_threadsafe(out.put_nowait, frame)
    except GenerationCancelled:
        pass
    except Exception as exc:  # the defined error surface of the protocol
        logger.exception("VoXtream generation failed")
        error = str(exc) or exc.__class__.__name__
    finally:
        if gen is not None:
            try:
                gen.close()
            except Exception:
                logger.exception("Closing the VoXtream generator failed")
        loop.call_soon_threadsafe(out.put_nowait, _Done(error))


def create_app(generator: Any | None = None, config_dir: Path | None = None,
               device: str | None = None, compile_model: bool = False) -> FastAPI:
    """Build the sidecar app.

    With ``generator`` given (tests), it is served at once. Without one, the lifespan
    loads VoXtream from ``config_dir`` in a background thread and ``/health`` answers
    503 until that is done.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.generation_lock = asyncio.Lock()
        load_task = None
        if app.state.generator is None:
            load_task = asyncio.create_task(_load_generator(app, config_dir or CONFIG_DIR,
                                                            device, compile_model))
        try:
            yield
        finally:
            if load_task is not None and not load_task.done():
                load_task.cancel()

    app = FastAPI(title="VoXtream sidecar", lifespan=lifespan)
    app.state.generator = generator

    @app.get("/health")
    async def health():
        if app.state.generator is None:
            return JSONResponse({"status": "loading"}, status_code=503)
        return {"status": "ok", "sample_rate": SAMPLE_RATE}

    @app.websocket("/voxtream")
    async def synthesis(ws: WebSocket):
        await ws.accept()
        generator = app.state.generator
        if generator is None:
            await _send_error(ws, "Model is still loading.")
            await _close(ws, code=1013)
            return
        try:
            init = json.loads(await ws.receive_text())
        except (WebSocketDisconnect, RuntimeError):
            return
        except json.JSONDecodeError:
            await _send_error(ws, "First message must be a JSON init frame.")
            await _close(ws)
            return
        if not isinstance(init, dict) or init.get("event") != "init":
            await _send_error(ws, "First message must be {'event': 'init', ...}.")
            await _close(ws)
            return
        prompt_path = Path(str(init.get("prompt_audio_path") or ""))
        if not init.get("prompt_audio_path") or not prompt_path.is_file():
            await _send_error(ws, f"Invalid prompt audio: {init.get('prompt_audio_path')!r}")
            await _close(ws)
            return
        text_initial = init.get("text") or ""
        full_stream = bool(init.get("full_stream", False))

        async with app.state.generation_lock:
            await _serve_generation(ws, generator, prompt_path, text_initial, full_stream)

    return app


async def _serve_generation(ws: WebSocket, generator: Any, prompt_path: Path,
                            text_initial: str, full_stream: bool) -> None:
    loop = asyncio.get_running_loop()
    feed = TextFeed()
    feed.push(text_initial)
    if not full_stream:
        feed.end()
    out: asyncio.Queue[Any] = asyncio.Queue()
    cancelled = threading.Event()

    def stop() -> None:
        cancelled.set()
        feed.cancel()

    async def receive_text_frames() -> None:
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    stop()
                    return
                text = msg.get("text")
                if text is None:
                    continue  # binary frames are not part of the protocol
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    feed.push(text)  # a bare text frame is a chunk
                    continue
                event = payload.get("event") if isinstance(payload, dict) else None
                if event == "text":
                    feed.push(str(payload.get("chunk", "")))
                elif event == "eot":
                    feed.end()
        except (WebSocketDisconnect, RuntimeError):
            stop()

    await ws.send_text(json.dumps({"type": "config", "sample_rate": SAMPLE_RATE,
                                   "dtype": "float32", "channels": 1}))
    worker = threading.Thread(
        target=_run_generation, args=(generator, prompt_path, feed, out, loop, cancelled),
        name="voxtream-generation", daemon=True)
    worker.start()
    receiver = asyncio.create_task(receive_text_frames())
    error: str | None = None
    client_gone = False
    try:
        while True:
            item = await out.get()
            if isinstance(item, _Done):
                error = item.error
                break
            frame = np.ascontiguousarray(item, dtype=np.float32)
            try:
                await ws.send_bytes(frame.tobytes())
            except Exception:
                client_gone = True
                stop()
                break
        if not client_gone and not cancelled.is_set():
            if error is not None:
                await _send_error(ws, error)
            else:
                await ws.send_text(json.dumps({"type": "eos"}))
            await _close(ws)
    finally:
        stop()
        receiver.cancel()
        # The generation lock is held until the thread has closed its generator, so the
        # next connection never overlaps this one on the model.
        await asyncio.to_thread(worker.join, _WORKER_JOIN_S)
        if worker.is_alive():
            logger.warning("VoXtream generation thread did not stop within %.0f s",
                           _WORKER_JOIN_S)


async def _send_error(ws: WebSocket, message: str) -> None:
    try:
        await ws.send_text(json.dumps({"type": "error", "message": message}))
    except Exception:
        pass


async def _close(ws: WebSocket, code: int = 1000) -> None:
    try:
        await ws.close(code=code)
    except Exception:
        pass


async def _load_generator(app: FastAPI, config_dir: Path, device: str | None,
                          compile_model: bool) -> None:
    def load() -> Any:
        if device is not None and device != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        from voxtream.generator import SpeechGenerator, SpeechGeneratorConfig
        from voxtream.utils.generator import set_seed

        set_seed()
        with open(config_dir / "generator.json") as f:
            config = SpeechGeneratorConfig(**json.load(f))
        with open(config_dir / "speaking_rate.json") as f:
            spk_rate_config = json.load(f)
        return SpeechGenerator(config, spk_rate_config, compile=compile_model)

    logger.info("Loading VoXtream (configs=%s, device=%s, compile=%s)",
                config_dir, device or "default", compile_model)
    app.state.generator = await asyncio.to_thread(load)
    logger.info("VoXtream ready")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="bind address (default loopback; the server has no auth)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR,
                        help="directory with generator.json and speaking_rate.json")
    parser.add_argument("--device", default=None,
                        help="CUDA device index for CUDA_VISIBLE_DEVICES (default: inherit)")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    app = create_app(config_dir=args.config_dir, device=args.device,
                     compile_model=args.compile)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
