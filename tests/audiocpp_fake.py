"""A reusable in-process fake audio.cpp sidecar for the model-free test suite.

Speaks the real length-prefixed wire protocol from sidecar/PROTOCOL.md over a genuine
AF_UNIX socket (no GPU, no native binary, no model), so tests exercise the pipeline's
actual framing + socket round-trips rather than a stubbed method seam. Reused by the
contract tests and the frame-pattern, crash and PCM tests.

This is a test HELPER, not a test module: it defines no test_* functions, so pytest
never collects it (and it never spawns a server at import time -- only when a test
constructs FakeSidecar()).

ACCEPT BEHAVIOUR IS SINGLE-SESSION, ON PURPOSE. One sidecar process
serves exactly ONE streaming session at a time, and that limit is explicit:
`sidecar/src/main.cpp` admits a connection under a process-wide session lock and
refuses a concurrent second one immediately with `ERROR busy`. Concurrency at the
VoiceManager level comes from giving every /ws/stream session its OWN sidecar process
on its own private socket path (pipeline/audiocpp_stt.py "owned mode"), which is also
what makes the fault isolation real -- a process-fatal fault can only take down
the one session whose sidecar it is.

So the default here is `accept_mode="single"`, mirroring main.cpp exactly. Before the
session lock, main.cpp ran handle_connection() INLINE and a second client simply sat in the
listen backlog until the first disconnected (`accept_mode="serial"`, kept for reference). What
this fake must never do is out-capability the runtime it stands in for: it used to
spawn a thread per connection, and that is how a "concurrent sessions unaffected"
 assertion passed model-free against a runtime that was single-client. It was the
second such trap -- a live run found it emitted
mid-stream deltas the real Nemotron-3.5 model never produces. Multi-session concurrency
remains a LIVE claim (scripts/audiocpp_two_session_repro.py), never a fake-sidecar one.

Scriptable per connection:
  * ready_delay  -- sleep before sending READY (drives the "feed offloads" assertion)
  * hello_error  -- reply to HELLO with ERROR <code> and close (handshake-error path)
  * audio_events -- EVENT partial texts emitted per AUDIO frame (a model that streams
                    from process_audio_chunk; the real Nemotron-3.5 sends none here)
  * audio_delta_script -- ONE scripted EVENT popped per AUDIO frame, in order; once
                    exhausted, further frames emit nothing. This is the delta-streaming
                    shape: audio.cpp emits partial_text as token/char DELTAS, so
                    a faithful fake sends each fragment exactly once. Entries are either
                    a str (an is_final=false delta), a dict dumped verbatim as the
                    EVENT payload -- the dict form scripts is_final=true marker events
                    and other exact wire shapes -- or None to emit NOTHING for that
                    frame (a model below its output threshold, e.g. voxtral fed under
                    ~0.6 s of speech). Takes over from audio_events
                    when set. flush_events / flush_script event lists accept the same
                    str|dict entries.
  * audio_final  -- a FINAL text emitted after each AUDIO frame (volunteered final)
  * flush_events -- EVENT partial texts emitted on FLUSH (the pre-final burst)
  * flush_final  -- the FINAL text emitted on FLUSH (None to emit no FINAL)
  * flush_final_delay -- sleep this long before the FLUSH's FINAL: a legitimately SLOW
                    single decode (one long pause, then the FINAL), the shape the
                    defensive whole-exchange bound must NOT break
  * flush_event_flood -- on FLUSH, drip EVENT frames carrying this text endlessly (a few
                    ms apart) and NEVER send a FINAL. A deliberately MISBEHAVING sidecar:
                    PROTOCOL.md guarantees exactly one FINAL per FLUSH and no real-sidecar
                    path to this shape is known, but every EVENT resets the client's
                    per-recv timeout, so without the whole-exchange bound this hangs
                    the session forever. Takes over the FLUSH branch entirely.
  * flush_error  -- ERROR <code> in place of the FIRST FLUSH's FINAL, after which this
                    connection serves normally again -- exactly what drive_loop.cpp does
                    when session.finalize() throws: it writes ERROR `internal` from the
                    catch block and CONTINUES its loop, still serving
  * die_on_audio -- close the socket abruptly on the first AUDIO frame (fault sim)
  * die_on_audio_conn -- die like die_on_audio, but ONLY for the Nth connection (0-based),
                    leaving every other connection healthy -- so one session can be
                    faulted while a LATER session on the same sidecar still works

Constructor-level:
  * socket_path  -- bind here instead of a private temp path (lets a test stand a fake up
                    on the per-session path an owned-mode spawn asked for)
  * accept_mode  -- "single" (default, faithful to sidecar/src/main.cpp: one
                    session at a time under a process-wide lock, a concurrent second
                    connection refused immediately with ERROR `busy`), "serial" (the
                    older shape: the second connection waits in the listen backlog),
                    or "concurrent" (thread per connection). "concurrent" is a DELIBERATE
                    over-capability -- the real sidecar does not serve two sessions in one
                    process, so nothing may use it to make a fault-isolation concurrency claim.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import tempfile
import threading
import time

# The real registry's model_task ids and their families -- must mirror
# sidecar/src/task_registry.cpp. The fake validates HELLO against this
# the way main.cpp does (unknown id -> ERROR unknown_model_task naming the known ids),
# so a selection test's unknown-id path exercises the same wire shape the sidecar sends.
KNOWN_MODEL_TASKS = {
    "nemotron-3.5-asr-streaming": "nemotron_asr",
    "voxtral-mini-4b-realtime-asr-streaming": "voxtral_realtime",
}

# Message type tags -- must match sidecar/PROTOCOL.md.
HELLO = 0x01
READY = 0x02
AUDIO = 0x03
EVENT = 0x04
FLUSH = 0x05
FINAL = 0x06
RESET = 0x07
ERROR = 0x08
BYE = 0x09

_HEADER = struct.Struct(">BI")


def _event_payload(text: str, lang: str = "en") -> bytes:
    return json.dumps({
        "is_final": False,
        "partial_text": {"text": text, "language": lang},
        "voice_activity": [],
        "word_timestamps": [],
    }).encode("utf-8")


def _scripted_event_payload(entry, lang: str) -> bytes:
    """Wire payload for one scripted EVENT entry: str -> an is_final=false delta;
    dict -> dumped verbatim (lets a test script is_final=true markers, contentless
    events, or any exact wire shape)."""
    if isinstance(entry, str):
        return _event_payload(entry, lang)
    return json.dumps(entry).encode("utf-8")


def _final_payload(text: str, lang: str = "en") -> bytes:
    return json.dumps({
        "is_final": True,
        "text_output": {"text": text, "language": lang},
        "word_timestamps": [],
    }).encode("utf-8")


class Connection:
    """Per-connection record the tests assert against."""

    def __init__(self, index: int = 0):
        self.index = index  # 0-based accept order (drives die_on_audio_conn)
        self.hello: dict | None = None
        self.audio_frames = 0
        # Every AUDIO payload byte in arrival order -- what the session was actually
        # fed. The idle-gate tests assert speech survives gating
        # (its bytes appear contiguously here) and silence does not (no growth).
        self.audio_bytes = bytearray()
        self.flush_count = 0
        self.reset_count = 0
        self.closed_by_bye = False


class FakeSidecar:
    """A fake sidecar listening on a temp AF_UNIX socket. Use as a context manager."""

    def __init__(self, *, ready_delay: float = 0.0, hello_error: str | None = None,
                 audio_events: list[str] | None = None, audio_final: str | None = None,
                 audio_delta_script: list | None = None,
                 flush_events: list | None = None,
                 flush_final: str | None = "final text", flush_error: str | None = None,
                 flush_final_delay: float = 0.0,
                 flush_event_flood: str | None = None,
                 flush_script: list[tuple[list[str], str]] | None = None,
                 die_on_audio: bool = False,
                 die_on_audio_conn: int | None = None, lang: str = "en",
                 socket_path: str | None = None, accept_mode: str = "single"):
        if accept_mode not in ("single", "serial", "concurrent"):
            raise ValueError(
                f"accept_mode must be 'single', 'serial' or 'concurrent', got {accept_mode!r}")
        self.accept_mode = accept_mode
        self.ready_delay = ready_delay
        self.hello_error = hello_error
        self.audio_events = audio_events or []
        self.audio_final = audio_final
        # Per-AUDIO-frame delta scripting: pop one entry per frame, exhausted ->
        # emit nothing. Faithful to the delta wire shape, where each fragment is sent
        # exactly once (repeating audio_events every frame models replace semantics,
        # which the accumulating client no longer speaks).
        self.audio_delta_script = audio_delta_script
        self.flush_events = flush_events if flush_events is not None else []
        self.flush_final = flush_final
        self.flush_error = flush_error
        self.flush_final_delay = flush_final_delay
        self.flush_event_flood = flush_event_flood
        # Per-FLUSH scripting: each FLUSH on a connection pops the next
        # (events, final) pair; once exhausted, replies with no EVENTs and a FINAL of ""
        # -- the content-faithful shape for decoding accumulated silence (the real model
        # returns empty text, and PROTOCOL.md's exactly-one-FINAL-per-FLUSH still holds).
        # The single flush_events/flush_final pair above is content-BLIND (every FLUSH
        # answers with the same text no matter what was streamed), which is fine until a
        # test streams utterances of different content on one connection. When set, this
        # takes over the FLUSH branch entirely (flush_events/flush_final/flush_error are
        # ignored).
        self.flush_script = flush_script
        self.die_on_audio = die_on_audio
        self.die_on_audio_conn = die_on_audio_conn
        self.lang = lang

        # A caller-supplied path lets a test stand this fake up exactly where an
        # owned-mode spawn asked for its private sidecar; otherwise we own a
        # private temp dir and clean it up in stop().
        if socket_path is not None:
            self._dir: str | None = None
            self.socket_path = socket_path
        else:
            self._dir = tempfile.mkdtemp(prefix="fake-audiocpp-")
            self.socket_path = os.path.join(self._dir, "sidecar.sock")
        self._busy = threading.Lock()   # "single" mode: admits exactly one live session
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(self.socket_path)
        self._listener.listen(8)
        self._listener.settimeout(0.2)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.connections: list[Connection] = []
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)

    # --- lifecycle ----------------------------------------------------------------
    def __enter__(self) -> "FakeSidecar":
        self._accept_thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._listener.close()
        except OSError:
            pass
        self._accept_thread.join(timeout=2.0)
        for t in self._threads:
            t.join(timeout=2.0)
        try:
            os.path.exists(self.socket_path) and os.unlink(self.socket_path)
        except OSError:
            pass
        if self._dir is not None:
            try:
                os.rmdir(self._dir)
            except OSError:
                pass

    # --- server -------------------------------------------------------------------
    def _accept_loop(self) -> None:
        """Accept connections the way sidecar/src/main.cpp does: one session at a time.

        In the default "single" mode a connection is served on its own thread while a
        process-wide lock admits exactly ONE live session; a second connection arriving
        while that lock is held is refused immediately with ERROR `busy` and closed,
        mirroring main.cpp. "serial" is the older, pre-lock shape (served inline, so
        the next accept() waits and a second client sits unanswered in the backlog).
        "concurrent" is thread-per-connection with no lock and is NOT faithful to the
        sidecar (see the module docstring).
        """
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            if self.accept_mode == "serial":
                self._serve(conn)  # blocks until this session ends -- pre-lock shape
                continue
            target = self._serve_single if self.accept_mode == "single" else self._serve
            t = threading.Thread(target=target, args=(conn,), daemon=True)
            t.start()
            self._threads.append(t)

    def _serve_single(self, conn: socket.socket) -> None:
        """Serve under the one-session-per-process lock; refuse a concurrent second."""
        if not self._busy.acquire(blocking=False):
            try:
                self._send(conn, ERROR, json.dumps({
                    "ok": False, "code": "busy",
                    "message": "sidecar already has a live session",
                }).encode("utf-8"))
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
            return
        try:
            self._serve(conn)
        finally:
            self._busy.release()

    def _serve(self, conn: socket.socket) -> None:
        record = Connection(index=len(self.connections))
        self.connections.append(record)
        # This connection dies on AUDIO if globally armed, or if it is the specific
        # connection ordinal armed for a targeted fault.
        die_on_audio = self.die_on_audio or record.index == self.die_on_audio_conn
        conn.settimeout(0.5)
        # A volunteered audio_final is emitted once per connection so take_final() drains
        # to exactly one; audio_events stream on every frame (partials are continuous).
        audio_final_left = 1 if self.audio_final is not None else 0
        # Only the FIRST flush errors; the connection then serves normally, mirroring the
        # drive loop, which writes the ERROR from its catch block and keeps looping.
        flush_error_left = 1 if self.flush_error is not None else 0
        flush_idx = 0  # position in flush_script for THIS connection
        delta_idx = 0  # position in audio_delta_script for THIS connection
        try:
            if not self._handshake(conn, record):
                return
            while not self._stop.is_set():
                frame = self._recv_frame(conn)
                if frame is None:
                    return  # EOF
                mtype, payload = frame
                if mtype == AUDIO:
                    record.audio_frames += 1
                    record.audio_bytes.extend(payload)
                    if die_on_audio:
                        conn.close()  # abrupt EOF, no BYE -- the crash fault shape
                        return
                    if self.audio_delta_script is not None:
                        if delta_idx < len(self.audio_delta_script):
                            entry = self.audio_delta_script[delta_idx]
                            delta_idx += 1
                            if entry is not None:  # None -> below output threshold
                                self._send(conn, EVENT,
                                           _scripted_event_payload(entry, self.lang))
                    else:
                        for text in self.audio_events:
                            self._send(conn, EVENT, _event_payload(text, self.lang))
                    if audio_final_left:
                        self._send(conn, FINAL, _final_payload(self.audio_final, self.lang))
                        audio_final_left = 0
                elif mtype == FLUSH:
                    record.flush_count += 1
                    if self.flush_event_flood is not None:
                        # Misbehaving sidecar: endless EVENT drip, never a FINAL.
                        # Ends only when the peer goes away (send raises) or stop() is
                        # called -- exactly the shape the client-side bound must cut off.
                        while not self._stop.is_set():
                            self._send(conn, EVENT,
                                       _event_payload(self.flush_event_flood, self.lang))
                            time.sleep(0.005)
                        return
                    if self.flush_script is not None:
                        events, final = (self.flush_script[flush_idx]
                                         if flush_idx < len(self.flush_script)
                                         else ([], ""))
                        flush_idx += 1
                        for entry in events:
                            self._send(conn, EVENT, _scripted_event_payload(entry, self.lang))
                        self._send(conn, FINAL, _final_payload(final, self.lang))
                        continue
                    for entry in self.flush_events:
                        self._send(conn, EVENT, _scripted_event_payload(entry, self.lang))
                    if flush_error_left:
                        flush_error_left = 0
                        self._send(conn, ERROR, json.dumps({
                            "ok": False, "code": self.flush_error,
                            "message": "finalize failed: simulated decode error",
                        }).encode("utf-8"))
                    elif self.flush_final is not None:
                        if self.flush_final_delay > 0:
                            time.sleep(self.flush_final_delay)  # a slow single decode
                        self._send(conn, FINAL, _final_payload(self.flush_final, self.lang))
                elif mtype == RESET:
                    record.reset_count += 1
                elif mtype == BYE:
                    record.closed_by_bye = True
                    return
        except OSError:
            return
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _handshake(self, conn: socket.socket, record: Connection) -> bool:
        frame = self._recv_frame(conn)
        if frame is None:
            return False
        _, payload = frame
        try:
            record.hello = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            record.hello = None
        if self.hello_error is not None:
            self._send(conn, ERROR, json.dumps(
                {"ok": False, "code": self.hello_error, "message": "handshake rejected"}
            ).encode("utf-8"))
            return False
        # Registry lookup, faithful to main.cpp: an unregistered model_task is refused
        # with the defined error naming every known id.
        model_task = (record.hello or {}).get("model_task", "nemotron-3.5-asr-streaming")
        if model_task not in KNOWN_MODEL_TASKS:
            known = ", ".join(f"'{mid}'" for mid in KNOWN_MODEL_TASKS)
            self._send(conn, ERROR, json.dumps({
                "ok": False, "code": "unknown_model_task",
                "message": f"no such model_task; known: {known}",
            }).encode("utf-8"))
            return False
        if self.ready_delay:
            time.sleep(self.ready_delay)
        self._send(conn, READY, json.dumps({
            "ok": True,
            "protocol": 1,
            "model_task": model_task,
            "family": KNOWN_MODEL_TASKS[model_task],
            "audio": {"sample_rate": 16000, "channels": 1, "format": "s16le"},
            "preferred_chunk_samples": 512,
        }).encode("utf-8"))
        return True

    # --- framing ------------------------------------------------------------------
    def _send(self, conn: socket.socket, mtype: int, payload: bytes = b"") -> None:
        conn.sendall(_HEADER.pack(mtype, len(payload)) + payload)

    def _recv_frame(self, conn: socket.socket) -> tuple[int, bytes] | None:
        header = self._recv_exactly(conn, _HEADER.size)
        if header is None:
            return None
        mtype, length = _HEADER.unpack(header)
        payload = self._recv_exactly(conn, length) if length else b""
        if payload is None:
            return None
        return mtype, payload

    def _recv_exactly(self, conn: socket.socket, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            if self._stop.is_set():
                return None
            try:
                chunk = conn.recv(n - len(buf))
            except socket.timeout:
                continue
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)
