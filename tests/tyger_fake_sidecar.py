"""A stand-in for the Tyger sidecar, for model-free server tests.

It speaks the sidecar's wire contract (``voicemanager.pipeline.tyger.wire``): GET
/health, and one WebSocket per session at ``/tyger`` (optionally
``/tyger?session=LABEL``) that takes binary hops and answers with event lines. It
does not score anything: a test scripts the events each session gets and reads
back every hop the session sent. A session's key is its label, else its connection
order as a string ("0" for the first), as in the real sidecar.

Two ways to use it:

* In-process, for the session tests: ``FakeTygerSidecar(...).start()`` serves on a
  free loopback port from its own thread and event loop, so it works under a plain
  asyncio test and under the FastAPI TestClient alike. Script events per session
  and hop with ``events={key: {hop_number: [Event, ...]}}`` (hop_number counts the
  session's valid hops from 0; the events go out right after that hop), or with an
  ``on_hop(session, hop_number, hop)`` callback returning events. ``send(i, ...)``
  pushes events to the i-th session at any time and ``drop(i)`` closes it with
  code 1011. ``sessions`` holds a ``FakeSession`` per connection with its key,
  path, unpacked hops and any bad messages. ``ready = False`` makes /health 503.
  A message that is not a hop gets an error event line, as from the real sidecar.

* As a process, for the spawn and watch tests: run it with the test interpreter
  as if it were the Tyger environment's python::

      python tests/tyger_fake_sidecar.py --host 127.0.0.1 --port N [server.py args]

  It accepts every argument of the real server.py. Environment variables:

  - ``TYGER_FAKE_MODE``: ``ready`` (default) answers 200 at once; ``slow`` answers
    503 for ``TYGER_FAKE_DELAY_S`` seconds (default 1), then 200; ``never`` answers
    503 forever; ``exit`` exits with code 3 at once, like a sidecar whose import
    failed.
  - ``TYGER_FAKE_ARGS_FILE``: a file the fake writes its argument list to (JSON).
  - ``TYGER_FAKE_EVENTS``: a JSON file ``{key: {hop_number: [event, ...]}}`` with
    each event an object in the event-line format, e.g.
    ``{"event": "endpoint", "by": "model", "mic_sample": 5120}``.
  - ``TYGER_FAKE_RECORD``: a file the fake appends one JSON line to per hop:
    ``{"key", "index", "hop", "mode", "mic_sample", "user", "agent"}``, the
    channels as base64 of their little-endian int16 bytes.

  It keeps running until it is terminated.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import http
import importlib.util
import json
import os
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

if __name__ == "__main__":
    # A process of its own: load the tyger directory as a package by path (the
    # voicemanager package would import torch on the way), the way the real
    # sidecar does.
    _TYGER = Path(__file__).resolve().parents[1] / "src" / "voicemanager" / "pipeline" / "tyger"
    _spec = importlib.util.spec_from_file_location(
        "tyger", _TYGER / "__init__.py", submodule_search_locations=[str(_TYGER)])
    sys.modules["tyger"] = _pkg = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_pkg)
    import tyger.wire as wire
else:
    from voicemanager.pipeline.tyger import wire


@dataclass
class FakeSession:
    """What one connection sent."""
    index: int
    key: str
    path: str
    hops: list = field(default_factory=list)       # wire.Hop, in order
    bad: list = field(default_factory=list)        # messages that were not hops
    sent: list = field(default_factory=list)       # events sent to it, in order
    closed: bool = False
    ws: object = None


class FakeTygerSidecar:
    def __init__(self, events: dict | None = None,
                 on_hop: Callable[[FakeSession, int, "wire.Hop"], list | None] | None = None,
                 host: str = "127.0.0.1", port: int = 0,
                 record: Callable[[FakeSession, int, "wire.Hop"], None] | None = None) -> None:
        self.events = {str(k): {int(n): list(v) for n, v in hops.items()}
                       for k, hops in (events or {}).items()}
        self.on_hop = on_hop
        self.record = record
        self.host = host
        self.port: int | None = port or None
        self._bind_port = port
        self.ready = True
        self.sessions: list[FakeSession] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server = None
        self._started = threading.Event()

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}/tyger"

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    # --- lifecycle -----------------------------------------------------------------

    def start(self) -> "FakeTygerSidecar":
        self._thread = threading.Thread(target=self._run, name="fake-tyger", daemon=True)
        self._thread.start()
        if not self._started.wait(5):
            raise RuntimeError("fake Tyger sidecar did not start")
        return self

    def stop(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(5)

    def serve_forever(self) -> None:
        """Serve from the calling thread until the process is terminated."""
        self._run()

    def wait_for(self, predicate, timeout: float = 2.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.005)
        return predicate()

    # --- driving sessions from a test ------------------------------------------------

    def send(self, index: int, *events) -> None:
        """Send events to the index-th session now, as one message."""
        session = self.sessions[index]
        asyncio.run_coroutine_threadsafe(self._send(session, list(events)), self._loop).result(5)

    def drop(self, index: int, code: int = 1011) -> None:
        """Close the index-th session's connection, as a failing sidecar would."""
        ws = self.sessions[index].ws
        asyncio.run_coroutine_threadsafe(ws.close(code=code, reason="fake sidecar drop"),
                                         self._loop).result(5)

    # --- serving --------------------------------------------------------------------

    def _run(self) -> None:
        import websockets.asyncio.server

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def serve():
            self._server = await websockets.asyncio.server.serve(
                self._handle, self.host, self._bind_port, max_size=None,
                process_request=self._process_request)
            self.port = self._server.sockets[0].getsockname()[1]
            self._started.set()

        self._loop.run_until_complete(serve())
        try:
            self._loop.run_forever()
        finally:
            self._server.close()
            self._loop.run_until_complete(self._server.wait_closed())
            self._loop.close()

    def _process_request(self, connection, request):
        path = urllib.parse.urlsplit(request.path).path
        if path == "/health":
            ready = self.ready() if callable(self.ready) else self.ready
            body = json.dumps({"status": "ok" if ready else "loading", "scorer": "fake",
                               "sessions": sum(not s.closed for s in self.sessions)})
            status = http.HTTPStatus.OK if ready else http.HTTPStatus.SERVICE_UNAVAILABLE
            response = connection.respond(status, body)
            response.headers["Content-Type"] = "application/json"
            return response
        if path != "/tyger":
            return connection.respond(http.HTTPStatus.NOT_FOUND, "not found\n")
        return None

    async def _send(self, session: FakeSession, events: list) -> None:
        if not events:
            return
        await session.ws.send("\n".join(wire.format_event(e) for e in events))
        with self._lock:
            session.sent.extend(events)

    async def _handle(self, ws) -> None:
        split = urllib.parse.urlsplit(ws.request.path)
        label = urllib.parse.parse_qs(split.query).get("session", [""])[0]
        with self._lock:
            index = len(self.sessions)
            session = FakeSession(index=index, key=label or str(index), path=ws.request.path,
                                  ws=ws)
            self.sessions.append(session)
        try:
            async for message in ws:
                try:
                    hop = wire.unpack_hop(message) if isinstance(message, bytes) else None
                except ValueError as exc:
                    hop, error = None, f"bad hop: {exc}"
                else:
                    error = "expected a binary hop, got a text message"
                if hop is None:
                    with self._lock:
                        session.bad.append(message)
                    await self._send(session, [wire.ErrorEvent(error)])
                    continue
                with self._lock:
                    number = len(session.hops)
                    session.hops.append(hop)
                if self.record is not None:
                    self.record(session, number, hop)
                events = list(self.events.get(session.key, {}).get(number, []))
                if self.on_hop is not None:
                    events += self.on_hop(session, number, hop) or []
                await self._send(session, events)
        except Exception:
            pass
        finally:
            session.closed = True


def _file_recorder(path: str):
    lock = threading.Lock()

    def record(session: FakeSession, number: int, hop) -> None:
        line = json.dumps({"key": session.key, "index": session.index, "hop": number,
                           "mode": hop.mode, "mic_sample": hop.mic_sample,
                           "user": base64.b64encode(hop.user).decode("ascii"),
                           "agent": base64.b64encode(hop.agent).decode("ascii")})
        with lock, open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    # The real server's other arguments, accepted and ignored.
    for flag in ("--model", "--revision", "--device", "--fake-scorer", "--log-level",
                 "--endpoint-threshold", "--endpoint-debounce", "--barge-threshold",
                 "--barge-debounce", "--warmup-frames", "--min-speech-ms", "--voice-gate-db",
                 "--silence-fallback-ms"):
        parser.add_argument(flag, default=None)
    args = parser.parse_args(argv)

    args_file = os.environ.get("TYGER_FAKE_ARGS_FILE")
    if args_file:
        with open(args_file, "w", encoding="utf-8") as f:
            json.dump(sys.argv[1:], f)

    mode = os.environ.get("TYGER_FAKE_MODE", "ready")
    if mode == "exit":
        sys.exit(3)
    ready_at = time.monotonic()
    if mode == "slow":
        ready_at += float(os.environ.get("TYGER_FAKE_DELAY_S", "1"))
    if mode == "never":
        ready_at = float("inf")

    events = {}
    events_file = os.environ.get("TYGER_FAKE_EVENTS")
    if events_file:
        raw = json.loads(Path(events_file).read_text())
        events = {key: {n: [wire.parse_event_line(json.dumps(e)) for e in evs]
                        for n, evs in hops.items()}
                  for key, hops in raw.items()}
    record_file = os.environ.get("TYGER_FAKE_RECORD")
    fake = FakeTygerSidecar(events=events, host=args.host, port=args.port,
                            record=_file_recorder(record_file) if record_file else None)
    fake.ready = lambda: time.monotonic() >= ready_at
    fake.serve_forever()


if __name__ == "__main__":
    main()
