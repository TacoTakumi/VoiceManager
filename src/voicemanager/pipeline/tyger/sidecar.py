"""The Tyger sidecar process, as the server sees it: spawn, wait, watch, restart, stop.

Tyger's model pins its own torch and transformers 5.x, so it runs in a separate
process from its own virtual environment (requirements-tyger.txt). ``TygerSidecar``
starts ``server.py`` in this directory by file path with that environment's python
(VM_TYGER_PY), waits for its ``/health`` to answer 200, and stops it at shutdown.
Nothing of the sidecar's code is imported here: the scorer and the server script
run only in the sidecar's interpreter.

The sidecar is one shared child process. ``watch`` notices when it exits and
starts it again, waiting longer after each failed try. Code that holds sessions on
the sidecar learns of a loss and of the return through ``add_listener`` (a callback
``(up, generation)`` run on the event loop), ``wait_lost`` and ``wait_up``, or by
reading ``up`` and ``generation``: the generation counts the starts that reached
ready, so a session opened on generation N knows it is stale once that changes.
"""
import asyncio
import logging
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# The sidecar script, by path. Resolved at construction time, so a test can point
# the module at a fake script.
SERVER_SCRIPT = Path(__file__).with_name("server.py")
# Settings named in startup errors, so the operator knows which line of .env to fix.
PY_SETTING = "VM_TYGER_PY"
READY_TIMEOUT_SETTING = "VM_TYGER_READY_TIMEOUT"
PORT_SETTING = "VM_TYGER_PORT"
_STOP_GRACE_S = 5.0
_HEALTH_POLL_S = 0.1
# The watcher: how often it reads the sidecar's exit status, how long it waits after
# the first failed start, and the cap the wait doubles up to after each further one.
_WATCH_POLL_S = 1.0
_WATCH_RETRY_S = 1.0
_WATCH_RETRY_MAX_S = 30.0

# (server.py flag, Settings field) for every turn-rule value the sidecar takes.
RULE_ARGUMENTS = (
    ("--endpoint-threshold", "tyger_endpoint_threshold"),
    ("--endpoint-debounce", "tyger_endpoint_debounce"),
    ("--barge-threshold", "tyger_barge_threshold"),
    ("--barge-debounce", "tyger_barge_debounce"),
    ("--warmup-frames", "tyger_warmup_frames"),
    ("--min-speech-ms", "tyger_min_speech_ms"),
    ("--voice-gate-db", "tyger_voice_gate_db"),
    ("--silence-fallback-ms", "tyger_silence_fallback_ms"),
)

Listener = Callable[[bool, int], None]


class TygerStartupError(RuntimeError):
    """The sidecar could not be started; the message names the setting at fault."""


class TygerSidecar:
    """The Tyger process: spawn, wait for ready, watch and restart, stop."""

    def __init__(self, python: str | Path, port: int, device: str, model: str,
                 revision: str | None = None, rule_args: list[str] | None = None,
                 ready_timeout_s: float = 180.0, host: str = "127.0.0.1",
                 server_script: str | Path | None = None) -> None:
        self.python = Path(python).expanduser()
        self.port = int(port)
        self.host = host
        self.device = device
        self.model = model
        self.revision = revision
        self.rule_args = list(rule_args or [])
        self.ready_timeout_s = float(ready_timeout_s)
        self.server_script = Path(server_script) if server_script else Path(SERVER_SCRIPT)
        self.process: subprocess.Popen | None = None
        # Counts the starts that reached ready; 0 before the first.
        self.generation = 0
        # Set once the spawned process answered /health; cleared on every spawn and
        # stop, so ``up`` is false while a start is still waiting.
        self._answered_health = False
        # Set by close(): a start still waiting ends and no start begins, so a server
        # shutting down never leaves a sidecar behind.
        self._closing = False
        self._lock = threading.Lock()
        self._listeners: list[Listener] = []

    @classmethod
    def from_settings(cls, settings, device: str) -> "TygerSidecar":
        """The sidecar the settings describe, on ``device`` (cpu, cuda or cuda:N)."""
        rule_args: list[str] = []
        for flag, field in RULE_ARGUMENTS:
            rule_args += [flag, str(getattr(settings, field))]
        return cls(python=settings.tyger_py, port=settings.tyger_port, device=device,
                   model=settings.tyger_model, revision=settings.tyger_revision,
                   rule_args=rule_args, ready_timeout_s=settings.tyger_ready_timeout)

    # --- addresses ------------------------------------------------------------------

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}/tyger"

    def session_url(self, label: str) -> str:
        """The WebSocket URL of one session, labelled for the sidecar's log."""
        return f"{self.url}?{urllib.parse.urlencode({'session': label})}"

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/health"

    def command(self) -> list[str]:
        cmd = [str(self.python), str(self.server_script), "--host", self.host,
               "--port", str(self.port), "--model", self.model]
        if self.revision:
            cmd += ["--revision", self.revision]
        return cmd + ["--device", self.device] + self.rule_args

    # --- start and stop -------------------------------------------------------------

    def start(self) -> None:
        """Spawn the sidecar and block until /health answers 200.

        Raises TygerStartupError naming VM_TYGER_PY when the interpreter is missing
        or the process dies, and naming VM_TYGER_READY_TIMEOUT when the process
        lives but is not ready in time. Nothing is left running on failure.
        """
        if not self.python.is_file():
            raise TygerStartupError(
                f"{PY_SETTING}={self.python} is not a file; it must be the python of "
                "the Tyger virtual environment (requirements-tyger.txt)")
        if self.alive:
            return
        if self._closing:
            raise TygerStartupError("the Tyger sidecar is closing; no start begins")
        self._answered_health = False
        if self._port_in_use():
            # A stranger on the port (an orphaned sidecar from a killed server, another
            # service) would answer /health and be taken for the child we never
            # managed to start. Refuse instead.
            raise TygerStartupError(
                f"{PORT_SETTING}={self.port} is already in use on {self.host}; stop the "
                "process that holds it (an orphaned sidecar?) or pick another port")
        cmd = self.command()
        logger.info("Starting Tyger sidecar: %s", " ".join(cmd))
        try:
            # Its own session: a service stop or a Ctrl-C signals the server's process
            # group, and the sidecar must not die ahead of the lifespan, which stops
            # it itself (otherwise the watcher starts one the shutdown then kills).
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            raise TygerStartupError(
                f"{PY_SETTING}={self.python} could not be started: {exc}") from exc
        self.process = proc
        deadline = time.monotonic() + self.ready_timeout_s
        while True:
            if self._closing:
                # close() ran while this start waited: its stop() may already have
                # taken the process; stop again so both paths leave nothing running.
                self.stop()
                raise TygerStartupError(
                    "the Tyger sidecar was stopped during startup: the server is "
                    "shutting down")
            code = proc.poll()
            if code is not None:
                self.process = None
                raise TygerStartupError(
                    f"the Tyger sidecar exited with code {code} during startup "
                    f"({PY_SETTING}={self.python}); its log says why")
            if self.is_ready():
                with self._lock:
                    self._answered_health = True
                    self.generation += 1
                logger.info("Tyger sidecar ready at %s (start %d)", self.url, self.generation)
                return
            if time.monotonic() >= deadline:
                self.stop()
                raise TygerStartupError(
                    f"the Tyger sidecar was not ready within "
                    f"{READY_TIMEOUT_SETTING}={self.ready_timeout_s:g} s; check its log "
                    "(is the bundle fetched with `python setup_models.py tyger`?)")
            time.sleep(_HEALTH_POLL_S)

    def _port_in_use(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=0.5):
                return True
        except OSError:
            return False

    def is_ready(self) -> bool:
        """Ask /health; True on 200 (blocking, up to a second)."""
        try:
            with urllib.request.urlopen(self.health_url, timeout=1.0) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def close(self) -> None:
        """Stop the process for good: a start still waiting ends and no start begins."""
        self._closing = True
        self.stop()

    def stop(self) -> None:
        """Terminate the process, then kill it if it ignores the request."""
        with self._lock:
            proc, self.process = self.process, None
            self._answered_health = False
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=_STOP_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=_STOP_GRACE_S)
        logger.info("Tyger sidecar stopped")

    # --- state ----------------------------------------------------------------------

    @property
    def alive(self) -> bool:
        """The process runs (its exit status is not set); no request is made."""
        proc = self.process
        return proc is not None and proc.poll() is None

    @property
    def up(self) -> bool:
        """The process runs and has answered /health since it was spawned."""
        return self._answered_health and self.alive

    def add_listener(self, listener: Listener) -> Callable[[], None]:
        """Call ``listener(up, generation)`` when the watcher sees a loss or a return.

        A loss passes False and the generation that was lost; a return passes True
        and the new generation. Listeners run on the watcher's event loop and must
        not block; one that raises is logged and the others still run. Returns a
        function that removes the listener.
        """
        self._listeners.append(listener)

        def remove() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)
        return remove

    def _notify(self, up: bool, generation: int) -> None:
        for listener in list(self._listeners):
            try:
                listener(up, generation)
            except Exception:
                logger.exception("Tyger sidecar listener failed")

    async def wait_lost(self) -> int:
        """Return the lost generation once the watcher reports a loss.

        Returns at once with the current generation when the sidecar is not up now.
        """
        if not self.up:
            return self.generation
        return await self._next(up=False)

    async def wait_up(self) -> int:
        """Return the generation once the sidecar is up (at once if it is now)."""
        if self.up:
            return self.generation
        return await self._next(up=True)

    async def _next(self, up: bool) -> int:
        future = asyncio.get_running_loop().create_future()

        def listener(now_up: bool, generation: int) -> None:
            if now_up is up and not future.done():
                future.set_result(generation)
        remove = self.add_listener(listener)
        try:
            return await future
        finally:
            remove()

    # --- the watcher ----------------------------------------------------------------

    async def watch(self, poll_interval: float | None = None,
                    retry_delay: float | None = None,
                    max_retry_delay: float | None = None) -> None:
        """Start the sidecar again whenever it exits; runs until cancelled.

        A dead sidecar is reported to the listeners once, then started through the
        same path as at startup (the ready wait and the port-in-use refusal apply),
        in a worker thread so the event loop keeps serving. A start that fails is
        logged and tried again after ``retry_delay``, doubling after each further
        failure up to ``max_retry_delay``; a start that works resets the wait and is
        reported to the listeners. Nothing raises out of here. The intervals default
        to the module constants, read at call time.
        """
        if poll_interval is None:
            poll_interval = _WATCH_POLL_S
        if retry_delay is None:
            retry_delay = _WATCH_RETRY_S
        if max_retry_delay is None:
            max_retry_delay = max(_WATCH_RETRY_MAX_S, retry_delay)
        delay = retry_delay
        # The generation whose loss is still to be reported: the one that reached
        # ready last, even if it died before this coroutine first ran.
        reported_generation = self.generation or None
        while True:
            if self.alive:
                await asyncio.sleep(poll_interval)
                continue
            if reported_generation is not None:
                # A process that reached ready has exited: one loss, reported once.
                # A failed restart leaves no process; its error is the whole story.
                proc = self.process
                code = proc.poll() if proc is not None else None
                logger.warning("Tyger sidecar is gone (exit code %s); starting it again",
                               code)
                self._notify(False, reported_generation)
                reported_generation = None
            try:
                await asyncio.to_thread(self.start)
            except TygerStartupError as exc:
                logger.error("Tyger sidecar restart failed: %s; next try in %g s",
                             exc, delay)
            except Exception:
                logger.exception("Tyger sidecar restart failed; next try in %g s", delay)
            else:
                delay = retry_delay
                reported_generation = self.generation
                logger.info("Tyger sidecar is back (start %d)", self.generation)
                self._notify(True, self.generation)
                continue
            await asyncio.sleep(delay)
            delay = min(delay * 2, max_retry_delay)
