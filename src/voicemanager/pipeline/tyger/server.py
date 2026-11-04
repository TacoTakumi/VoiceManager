"""Tyger sidecar server: turn scores and turn rules for every opted-in session.

This file runs inside the Tyger virtual environment (requirements-tyger.txt), by
file path, not inside VoiceManager's own environment: Tyger pins its own torch and
transformers 5.x. It imports nothing from the voicemanager package (whose pipeline
package imports torch and silero); it loads this directory as a package by path
instead, so ``rules``, ``scorer`` and ``wire`` resolve without it. The real model
is imported only when the script runs without ``--fake-scorer``.

Wire protocol (``wire`` has the formats):

- One WebSocket per session at ``/tyger``, optionally labelled
  ``/tyger?session=LABEL`` (the label shows in the log and picks the fake
  scorer's script). ``progress=1`` in the query asks for progress lines.
- The server sends one binary message per 320 ms hop. The sidecar scores the hop's
  four 80 ms frames with the session's scorer, feeds them through the session's
  turn rules, and sends the hop's events, if any, as one text message of event
  lines (endpoint, barge).
- A message that is not a valid hop, or a hop the scorer fails on, gets one error
  event line, and the session goes on with the next hop.
- A session that asked for progress gets a scored line at 0 when it opens, and one
  at the hop's mic_sample last in the message of every valid hop, scored or failed,
  so each hop then gets a message.

Each session has its own turn rules and scorer state; the model is shared. Scoring
runs on one thread, so the GPU serves one hop at a time and the event loop keeps
every session's I/O moving meanwhile.

Readiness: ``GET /health`` answers 200 once the scorer is loaded and 503 before. A
model that fails to load stops the process with exit code 1.

    python server.py --port 8257 --device cuda:3 --model nineninesix/tyger --revision REV
    python server.py --port 8257 --fake-scorer script.json
"""
import argparse
import asyncio
import importlib.util
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

_HERE = Path(__file__).resolve().parent

if __package__:
    from . import rules, scorer, wire
else:
    # Run as a script: sys.path[0] is this directory, where wire.py and rules.py
    # would import as top-level modules of their own. Load the directory as the
    # package "tyger" instead, so every module is loaded once, through it.
    if sys.path and Path(sys.path[0] or ".").resolve() == _HERE:
        sys.path.pop(0)
    if "tyger" not in sys.modules:
        _spec = importlib.util.spec_from_file_location(
            "tyger", _HERE / "__init__.py", submodule_search_locations=[str(_HERE)])
        sys.modules["tyger"] = _pkg = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_pkg)
    import tyger.rules as rules
    import tyger.scorer as scorer
    import tyger.wire as wire

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

logger = logging.getLogger("tyger_server")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8257
DEFAULT_MODEL = "nineninesix/tyger"

# Command-line flags of the rule settings: (flag, TurnRuleSettings field, type, help).
RULE_FLAGS = (
    ("--endpoint-threshold", "endpoint_threshold", float,
     "p_end at or above this counts toward an endpoint"),
    ("--endpoint-debounce", "endpoint_debounce_frames", int,
     "frames in a row at the endpoint threshold that end a turn"),
    ("--barge-threshold", "barge_threshold", float,
     "yield at or above this counts toward a barge"),
    ("--barge-debounce", "barge_debounce_frames", int,
     "frames in a row at the barge threshold that make a barge"),
    ("--warmup-frames", "warmup_frames", int,
     "frames of fresh context before the heads' decisions count"),
    ("--min-speech-ms", "min_speech_ms", float,
     "voiced audio a turn needs before it can end"),
    ("--voice-gate-db", "voice_gate_db", float,
     "a frame above this energy (dBFS) is voiced"),
    ("--silence-fallback-ms", "silence_fallback_ms", float,
     "quiet below the voice gate that ends a spoken turn the head has not"),
)


class Session:
    """One WebSocket's turn state: its rules and its scorer."""

    def __init__(self, key: str, rules_engine: "rules.TurnRules",
                 session_scorer: "scorer.SessionScorer") -> None:
        self.key = key
        self.rules = rules_engine
        self.scorer = session_scorer
        self.hops = 0


def process_hop(session: Session, hop: "wire.Hop") -> list:
    """Score one hop and run its frames through the session's rules -> its events.

    Blocking (the real scorer runs the model): called on the scoring thread. A
    reset the rules ask for, before the hop or after a frame, is applied to the
    scorer before the next frame is scored.
    """
    session.hops += 1
    if session.rules.begin_hop(hop.mode):
        session.scorer.reset()
    events = []
    frames = 0
    for frame in session.scorer.score_hop(hop.user, hop.agent, hop.mode):
        if frames == rules.HOP_FRAMES:
            raise scorer.ScorerError(f"the scorer gave more than {rules.HOP_FRAMES} frames")
        # The frame's end on the mic stream: the hop's end, less the frames after it.
        mic_sample = max(0, hop.mic_sample - (rules.HOP_FRAMES - 1 - frames) * rules.FRAME_SAMPLES)
        result = session.rules.on_frame(hop.mode, frame.p_end, frame.yield_prob,
                                        frame.energy_db, mic_sample)
        events.extend(result.events)
        if result.reset:
            session.scorer.reset()
        frames += 1
    if frames != rules.HOP_FRAMES:
        raise scorer.ScorerError(f"the scorer gave {frames} frames, not {rules.HOP_FRAMES}")
    return events


def create_app(settings: "rules.TurnRuleSettings | None" = None,
               turn_scorer: "scorer.Scorer | None" = None,
               load_scorer: "Callable[[], scorer.Scorer] | None" = None,
               on_load_error: Callable[[], None] | None = None) -> FastAPI:
    """Build the sidecar app.

    With ``turn_scorer`` given it is served at once. Otherwise the lifespan runs
    ``load_scorer`` on the scoring thread and /health answers 503 until it is
    done; if it fails, ``app.state.load_error`` holds the error and
    ``on_load_error`` is called (``main`` stops the server with it).
    """
    rule_settings = settings if settings is not None else rules.TurnRuleSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tyger-score")
        app.state.executor = executor
        load_task = None
        if app.state.scorer is None and load_scorer is not None:
            load_task = asyncio.create_task(_load(app, executor, load_scorer, on_load_error))
        try:
            yield
        finally:
            if load_task is not None and not load_task.done():
                load_task.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="Tyger sidecar", lifespan=lifespan)
    app.state.scorer = turn_scorer
    app.state.load_error = None
    app.state.connections = 0
    app.state.sessions = 0

    @app.get("/health")
    async def health():
        if app.state.scorer is None:
            return JSONResponse({"status": "loading"}, status_code=503)
        return {"status": "ok", "scorer": app.state.scorer.name,
                "sessions": app.state.sessions}

    @app.websocket("/tyger")
    async def tyger_session(ws: WebSocket):
        await ws.accept()
        index = app.state.connections
        app.state.connections += 1
        active = app.state.scorer
        if active is None:
            await _send_events(ws, [wire.ErrorEvent("the Tyger model is still loading")])
            await _close(ws, code=1013)
            return
        key = ws.query_params.get("session") or str(index)
        progress = ws.query_params.get(wire.PROGRESS_PARAM) == "1"
        session = Session(key, rules.TurnRules(rule_settings), active.new_session(key, index))
        app.state.sessions += 1
        logger.info("session %s opened", key)
        try:
            if progress and not await _send_events(ws, [wire.ScoredEvent(mic_sample=0)]):
                return
            await _serve_session(ws, session, app.state.executor, progress)
        finally:
            app.state.sessions -= 1
            logger.info("session %s closed after %d hops", key, session.hops)

    return app


async def _serve_session(ws: WebSocket, session: Session, executor: ThreadPoolExecutor,
                         progress: bool = False) -> None:
    """Receive hops in order; each is scored before the next is read, so a
    session's events keep the order of its hops. With ``progress``, each valid
    hop's message ends with its scored line."""
    loop = asyncio.get_running_loop()
    while True:
        try:
            message = await ws.receive()
        except (WebSocketDisconnect, RuntimeError):
            return
        if message["type"] == "websocket.disconnect":
            return
        data = message.get("bytes")
        if data is None:
            events: list = [wire.ErrorEvent("expected a binary hop, got a text message")]
        else:
            try:
                hop = wire.unpack_hop(data)
            except ValueError as exc:
                events = [wire.ErrorEvent(f"bad hop: {exc}")]
            else:
                try:
                    events = await loop.run_in_executor(executor, process_hop, session, hop)
                except Exception as exc:  # one hop's failure; the session goes on
                    logger.exception("session %s: scoring a hop failed", session.key)
                    events = [wire.ErrorEvent(f"scoring failed: {exc or type(exc).__name__}")]
                if progress:
                    events.append(wire.ScoredEvent(mic_sample=hop.mic_sample))
        if events and not await _send_events(ws, events):
            return


async def _send_events(ws: WebSocket, events: list) -> bool:
    """Send events as one text message of event lines. False if the peer is gone."""
    try:
        await ws.send_text("\n".join(wire.format_event(e) for e in events))
        return True
    except Exception:
        return False


async def _close(ws: WebSocket, code: int = 1000) -> None:
    try:
        await ws.close(code=code)
    except Exception:
        pass


async def _load(app: FastAPI, executor: ThreadPoolExecutor,
                load_scorer: "Callable[[], scorer.Scorer]",
                on_load_error: Callable[[], None] | None) -> None:
    loop = asyncio.get_running_loop()
    try:
        app.state.scorer = await loop.run_in_executor(executor, load_scorer)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("Loading the Tyger scorer failed")
        app.state.load_error = exc
        if on_load_error is not None:
            on_load_error()
        return
    logger.info("Tyger scorer ready (%s)", app.state.scorer.name)


def _device(value: str) -> str:
    try:
        return scorer.check_device(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, not {number}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help="bind address (default loopback; the server has no auth)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="the bundle: a Hugging Face repo in the local cache, or a directory")
    parser.add_argument("--revision", default=None, help="the bundle's revision in the cache")
    parser.add_argument("--device", type=_device, default="cuda", help="cpu, cuda or cuda:N")
    parser.add_argument("--torch-threads", type=_positive_int, default=1, metavar="N",
                        help="limit torch's CPU intra-op threads to N (default 1): the "
                             "per-hop encode is CPU-bound, and many threads slow it down "
                             "on a busy CPU")
    parser.add_argument("--fake-scorer", type=Path, default=None, metavar="SCRIPT",
                        help="score from this JSON script instead of the model (tests)")
    defaults = rules.TurnRuleSettings()
    for flag, field, kind, text in RULE_FLAGS:
        parser.add_argument(flag, dest=field, type=kind, default=getattr(defaults, field),
                            help=f"{text} (default {getattr(defaults, field)})")
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: list[str] | None = None) -> None:
    import uvicorn

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = rules.TurnRuleSettings(**{field: getattr(args, field)
                                             for _flag, field, _kind, _text in RULE_FLAGS})
    except ValueError as exc:
        parser.error(str(exc))
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    turn_scorer: Any = None
    load_scorer = None
    if args.fake_scorer is not None:
        try:
            turn_scorer = scorer.FakeScorer.from_file(args.fake_scorer)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        def load_scorer():
            return scorer.TygerScorer.load(args.model, args.revision, args.device,
                                           threads=args.torch_threads)

    server: Any = None

    def stop() -> None:
        server.should_exit = True

    app = create_app(settings, turn_scorer=turn_scorer, load_scorer=load_scorer,
                     on_load_error=stop)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port,
                                           log_level=args.log_level))
    server.run()
    if app.state.load_error is not None:
        sys.exit(1)


if __name__ == "__main__":
    main()
