"""A stand-in for the VoXtream sidecar process, for model-free tests.

Run with the test interpreter as if it were the VoXtream environment's python:

    python tests/voxtream_fake_sidecar.py --host 127.0.0.1 --port N [--device D]

It accepts the real server's arguments and serves ``GET /health`` with 200. The
environment variable ``VOXTREAM_FAKE_MODE`` changes its behaviour:

- ``ready`` (default): answer 200 at once.
- ``slow``: answer 503 for ``VOXTREAM_FAKE_DELAY_S`` seconds (default 1), then 200,
  the way the real sidecar does while the model loads.
- ``never``: answer 503 forever.
- ``exit``: exit with code 3 at once, like a sidecar whose import failed.

``VOXTREAM_FAKE_ARGS_FILE`` names a file the fake writes its argument list to, so a
test can check what it was started with. The fake keeps running until it is
terminated.
"""
import argparse
import asyncio
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    args_file = os.environ.get("VOXTREAM_FAKE_ARGS_FILE")
    if args_file:
        with open(args_file, "w", encoding="utf-8") as f:
            json.dump(sys.argv[1:], f)

    mode = os.environ.get("VOXTREAM_FAKE_MODE", "ready")
    if mode == "exit":
        sys.exit(3)
    ready_at = time.monotonic()
    if mode == "slow":
        ready_at += float(os.environ.get("VOXTREAM_FAKE_DELAY_S", "1"))
    if mode == "never":
        ready_at = float("inf")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server naming)
            if self.path != "/health":
                self.send_response(404)
                self.end_headers()
                return
            ready = time.monotonic() >= ready_at
            body = json.dumps({"status": "ok" if ready else "loading",
                               "sample_rate": 24000}).encode()
            self.send_response(200 if ready else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):  # quiet
            return

    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()


# --- in-process fake for the engine tests ------------------------------------------------
#
# FakeVoxtreamSidecar serves the wire contract on a loopback WebSocket from its own
# thread and event loop, so it works under a plain asyncio test and under the FastAPI
# TestClient alike. It records every init, text chunk, eot and close, and it emits
# ``frames_per_chunk`` float32 frames for each text chunk it receives (at once), then
# eos after eot. ``hold_eos`` keeps the connection open after eot so a test can check
# the client side while the sidecar is still "speaking".

FRAME_SAMPLES = 1920  # 80 ms at 24 kHz


class FakeVoxtreamSidecar:
    def __init__(self, frames_per_chunk: int = 1, hold_eos: bool = False,
                 keep_emitting_after_eot: bool = False, init_error: str | None = None):
        self.frames_per_chunk = frames_per_chunk
        self.hold_eos = hold_eos
        self.keep_emitting_after_eot = keep_emitting_after_eot
        # Answer every init with this error message instead of config, then keep the
        # connection open until the client closes it (so a leaked client socket shows).
        self.init_error = init_error
        self.inits: list[dict] = []
        self.chunks: list[list[str]] = []  # per connection, in order
        self.eots: list[int] = []  # connection indexes that sent eot
        self.closes: list[tuple[int, float]] = []  # (connection index, monotonic time)
        self.frames_sent: list[int] = []  # per connection
        self.connections = 0
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server = None
        self.port: int | None = None
        self._started = threading.Event()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/voxtream"

    def start(self) -> "FakeVoxtreamSidecar":
        self._thread = threading.Thread(target=self._run, name="fake-voxtream", daemon=True)
        self._thread.start()
        if not self._started.wait(5):
            raise RuntimeError("fake sidecar did not start")
        return self

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(5)

    def wait_for(self, predicate, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.005)
        return predicate()

    def _run(self) -> None:
        import websockets.asyncio.server

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def serve():
            self._server = await websockets.asyncio.server.serve(
                self._handle, "127.0.0.1", 0, max_size=None)
            self.port = self._server.sockets[0].getsockname()[1]
            self._started.set()

        self._loop.run_until_complete(serve())
        try:
            self._loop.run_forever()
        finally:
            self._server.close()
            self._loop.run_until_complete(self._server.wait_closed())
            self._loop.close()

    async def _handle(self, ws) -> None:
        with self._lock:
            index = self.connections
            self.connections += 1
            self.chunks.append([])
            self.frames_sent.append(0)
        frame_no = 0
        try:
            init = json.loads(await ws.recv())
            with self._lock:
                self.inits.append(init)
            if self.init_error is not None:
                await ws.send(json.dumps({"type": "error", "message": self.init_error}))
                while True:
                    await ws.recv()  # ends with ConnectionClosed when the client closes
            await ws.send(json.dumps({"type": "config", "sample_rate": 24000,
                                      "dtype": "float32", "channels": 1}))
            if init.get("text"):
                self.chunks[index].append(init["text"])
            eot = not init.get("full_stream", False)
            while not eot:
                message = await ws.recv()
                payload = json.loads(message)
                if payload.get("event") == "text":
                    self.chunks[index].append(payload["chunk"])
                    for _ in range(self.frames_per_chunk):
                        frame_no += 1
                        await ws.send(np.full(FRAME_SAMPLES, frame_no / 100.0,
                                              dtype=np.float32).tobytes())
                        self.frames_sent[index] = frame_no
                elif payload.get("event") == "eot":
                    eot = True
            with self._lock:
                self.eots.append(index)
            if self.keep_emitting_after_eot:
                while True:
                    frame_no += 1
                    await ws.send(np.full(FRAME_SAMPLES, frame_no / 100.0,
                                          dtype=np.float32).tobytes())
                    self.frames_sent[index] = frame_no
                    await asyncio.sleep(0.01)
            if self.hold_eos:
                await asyncio.sleep(3600)
            await ws.send(json.dumps({"type": "eos"}))
            await ws.close()
        except Exception:
            pass
        finally:
            with self._lock:
                self.closes.append((index, time.monotonic()))


if __name__ == "__main__":
    main()
