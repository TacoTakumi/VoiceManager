"""A reusable in-process fake audio.cpp TTS sidecar for the model-free test suite.

Speaks the real length-prefixed TTS wire protocol from sidecar/PROTOCOL.md (the TTS
additive frame set: TEXT 0x0A -> TTS_AUDIO 0x0B stream -> TTS_DONE 0x0C) over a
genuine AF_UNIX socket - no GPU, no native binary, no model - so tests exercise the
engine's actual framing + socket round-trips rather than a stubbed method seam.
Companion to tests/audiocpp_fake.py (the ASR fake); same conventions: a test HELPER,
not a test module - it defines no test_* functions and only serves once constructed.

Like the real sidecar, one connection is one session, and a TEXT request is answered
by a stream of TTS_AUDIO payloads, a flush tail, then exactly one TTS_DONE after the
last audio byte.

Scriptable:
  * chunks       -- the s16le TTS_AUDIO payloads sent per TEXT request, in order
                    (defaults to two deterministic patterns)
  * flush_chunk  -- the trailing vocoder-flush payload, sent last before TTS_DONE
  * gate_after_first -- after sending the FIRST chunk of a request, wait for
                    .proceed (threading.Event) before sending the rest: holds a
                    request "in flight" so a test can prove the engine yields the
                    first chunk before the fake finishes generating
  * hello_error  -- reply to HELLO with ERROR <code> and close (handshake-error path)
  * text_error   -- reply to the FIRST TEXT with ERROR <code> instead of audio; the
                    connection then serves normally (mirrors the drive path, which
                    reports a failed request and keeps its loop)
  * die_on_text  -- close the socket abruptly on the first TEXT frame (fault sim,
                    for the crash tests)
  * die_on_text_containing -- close abruptly only on a TEXT whose text contains
                    this marker; other requests (and other connections) serve
                    normally. Lets the two-session crash test kill exactly ONE
                    session's connection deterministically, regardless of the
                    order the sessions' warm-up connects land in.
  * ready_sample_rate -- the rate the READY ack advertises (default 48000)

Observed state the tests assert against:
  * requests     -- every parsed TEXT payload dict, in arrival order
  * cancels      -- every parsed CANCEL payload dict, in arrival order
  * request_done -- threading.Event set when the fake has finished sending a
                    request's full frame sequence (audio + flush + TTS_DONE);
                    cleared when the next TEXT arrives

Cancel behavior mirrors the real drive path: while answering a request the
fake polls its socket; a CANCEL frame stops the chunk stream (no flush tail) and
closes the request with TTS_DONE {"ok":true,"cancelled":true}. A CANCEL between
requests is recorded and ignored, like the sidecar's benign stray-cancel race.
"""
from __future__ import annotations

import json
import os
import select
import socket
import struct
import tempfile
import threading
import time

# Message type tags - must match sidecar/PROTOCOL.md.
HELLO = 0x01
READY = 0x02
ERROR = 0x08
BYE = 0x09
TEXT = 0x0A
TTS_AUDIO = 0x0B
TTS_DONE = 0x0C
CANCEL = 0x0D

TTS_MODEL_TASK = "dots-tts-streaming"

_HEADER = struct.Struct(">BI")


def s16le(*values: int) -> bytes:
    """Pack int16 sample values as raw s16le bytes."""
    return struct.pack(f"<{len(values)}h", *values)


#: Default per-request payloads: two mid-generation chunks and the flush tail,
#: byte patterns chosen so sample counts and values are assertable end to end.
DEFAULT_CHUNKS = [s16le(0, 1000, -1000, 32767), s16le(-32768, 42, -42)]
DEFAULT_FLUSH = s16le(7, -7, 7)


class FakeTtsSidecar:
    """A fake TTS sidecar listening on a temp AF_UNIX socket. Context manager."""

    def __init__(self, *, chunks: list[bytes] | None = None,
                 flush_chunk: bytes | None = DEFAULT_FLUSH,
                 gate_after_first: bool = False,
                 hello_error: str | None = None,
                 text_error: str | None = None,
                 die_on_text: bool = False,
                 die_on_text_containing: str | None = None,
                 ready_sample_rate: int = 48000,
                 socket_path: str | None = None):
        self.chunks = chunks if chunks is not None else list(DEFAULT_CHUNKS)
        self.flush_chunk = flush_chunk
        self.gate_after_first = gate_after_first
        self.hello_error = hello_error
        self.text_error = text_error
        self.die_on_text = die_on_text
        self.die_on_text_containing = die_on_text_containing
        self.ready_sample_rate = ready_sample_rate

        self.requests: list[dict] = []
        self.cancels: list[dict] = []
        self.hello: dict | None = None
        self.proceed = threading.Event()
        self.request_done = threading.Event()
        self.gate_timed_out = False

        if socket_path is None:
            self._dir: str | None = tempfile.mkdtemp(prefix="vm-tts-fake-")
            self.socket_path = os.path.join(self._dir, "sidecar.sock")
        else:
            self._dir = None
            self.socket_path = socket_path
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(self.socket_path)
        self._listener.listen(8)
        self._listener.settimeout(0.2)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._accept_thread = threading.Thread(target=self._accept_loop, daemon=True)

    # --- lifecycle ----------------------------------------------------------------
    def __enter__(self) -> "FakeTtsSidecar":
        self._accept_thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        self.proceed.set()  # release any gated request so serve threads can exit
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
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            t = threading.Thread(target=self._serve, args=(conn,), daemon=True)
            t.start()
            self._threads.append(t)

    def _serve(self, conn: socket.socket) -> None:
        conn.settimeout(0.5)
        text_error_left = 1 if self.text_error is not None else 0
        try:
            if not self._handshake(conn):
                return
            while not self._stop.is_set():
                frame = self._recv_frame(conn)
                if frame is None:
                    return  # EOF
                mtype, payload = frame
                if mtype == TEXT:
                    request = json.loads(payload.decode("utf-8"))
                    self.requests.append(request)
                    self.request_done.clear()
                    if self.die_on_text or (
                            self.die_on_text_containing is not None
                            and self.die_on_text_containing in request.get("text", "")):
                        conn.close()  # abrupt EOF, no TTS_DONE - the crash shape
                        return
                    if text_error_left:
                        text_error_left = 0
                        self._send(conn, ERROR, json.dumps({
                            "ok": False, "code": self.text_error,
                            "message": f"scripted {self.text_error}",
                        }).encode("utf-8"))
                        continue
                    self._answer_request(conn)
                elif mtype == CANCEL:
                    # A stray CANCEL between requests: benign race, record + ignore.
                    self._record_cancel(payload)
                elif mtype == BYE:
                    return
                else:
                    self._send(conn, ERROR, json.dumps({
                        "ok": False, "code": "bad_frame",
                        "message": "unexpected frame type for a TTS session",
                    }).encode("utf-8"))
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _record_cancel(self, payload: bytes) -> None:
        try:
            self.cancels.append(json.loads(payload.decode("utf-8")) if payload else {})
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.cancels.append({})

    def _poll_mid_request(self, conn: socket.socket) -> bool:
        """Consume any frames pending mid-request; True once a CANCEL arrived."""
        cancelled = False
        while True:
            readable, _, _ = select.select([conn], [], [], 0)
            if not readable:
                return cancelled
            frame = self._recv_frame(conn)
            if frame is None:
                raise OSError("peer vanished mid-request")
            mtype, payload = frame
            if mtype == CANCEL:
                self._record_cancel(payload)
                cancelled = True
            # Anything else mid-request is ignored, like the real drive path.

    def _answer_request(self, conn: socket.socket) -> None:
        """The TTS reply sequence: audio chunks, the flush tail, one TTS_DONE.

        Cancellable mid-stream: a CANCEL stops the chunk stream (no flush
        tail) and the request closes with TTS_DONE {"ok":true,"cancelled":true}.
        """
        cancelled = False
        for i, chunk in enumerate(self.chunks):
            if self._poll_mid_request(conn):
                cancelled = True
                break
            self._send(conn, TTS_AUDIO, chunk)
            if i == 0 and self.gate_after_first:
                # Hold the request in flight until the test releases it (the
                # engine must have yielded this first chunk already) - or until a
                # CANCEL breaks the hold, which is exactly how a cancel-mid-request
                # test keeps the request in flight while it cancels.
                deadline = time.monotonic() + 5.0
                while not self.proceed.wait(timeout=0.01):
                    if self._poll_mid_request(conn):
                        cancelled = True
                        break
                    if time.monotonic() > deadline:
                        self.gate_timed_out = True
                        break
                if cancelled:
                    break
        if not cancelled and self._poll_mid_request(conn):
            cancelled = True
        if cancelled:
            self._send(conn, TTS_DONE, b'{"ok":true,"cancelled":true}')
        else:
            if self.flush_chunk:
                self._send(conn, TTS_AUDIO, self.flush_chunk)
            self._send(conn, TTS_DONE, b'{"ok":true}')
        self.request_done.set()

    def _handshake(self, conn: socket.socket) -> bool:
        frame = self._recv_frame(conn)
        if frame is None:
            return False
        mtype, payload = frame
        if mtype != HELLO:
            return False
        self.hello = json.loads(payload.decode("utf-8"))
        if self.hello_error is not None:
            self._send(conn, ERROR, json.dumps({
                "ok": False, "code": self.hello_error,
                "message": f"scripted {self.hello_error}",
            }).encode("utf-8"))
            return False
        self._send(conn, READY, json.dumps({
            "ok": True,
            "protocol": 1,
            "model_task": self.hello.get("model_task", TTS_MODEL_TASK),
            "family": "dots_tts",
            "task": "tts",
            "mode": "streaming",
            "input": "none",
            "audio": {"sample_rate": self.ready_sample_rate, "channels": 1,
                      "format": "s16le"},
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
        if length == 0:
            return mtype, b""
        payload = self._recv_exactly(conn, length)
        if payload is None:
            return None
        return mtype, payload

    def _recv_exactly(self, conn: socket.socket, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            if self._stop.is_set():
                return None
            try:
                data = conn.recv(n - len(buf))
            except socket.timeout:
                continue
            except OSError:
                return None
            if not data:
                return None
            buf.extend(data)
        return bytes(buf)
