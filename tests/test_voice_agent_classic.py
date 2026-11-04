"""The reference voice agent's classic mode, over fakes.

Model-free and server-free. The duplex session, the chat stream and the player are
fakes; one test runs the real ChatStream against a loopback HTTP server that speaks
the OpenAI streaming shape. Covers: dispatch on the endpoint with the final as the
fallback; supersede when the final revises the dispatched text; other turns ignored;
tokens forwarded one by one and done after the last; the history carries the previous
reply; an HTTP failure is one printed line and the loop goes on; barge-in cancels
exactly once and the new utterance dispatches at its endpoint; the utterance record
and the run-file paths.
"""
import argparse
import asyncio
import importlib.util
import io
import json
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
if "voice_agent" in sys.modules:
    va = sys.modules["voice_agent"]
else:
    _spec = importlib.util.spec_from_file_location("voice_agent", SCRIPTS / "voice_agent.py")
    va = importlib.util.module_from_spec(_spec)
    sys.modules["voice_agent"] = va
    _spec.loader.exec_module(va)


# --- fakes -----------------------------------------------------------------------------

class FakeRequest:
    def __init__(self, duplex, request_id: str, voice, chunking):
        self.duplex = duplex
        self.request_id = request_id
        self.voice = voice
        self.chunking = chunking
        self.sent: list[str] = []
        self.done_sent = False
        self.cancelled = False

    async def send(self, text: str, flush: bool = False) -> None:
        self.sent.append(text)
        self.duplex.events.append(("send", self.request_id, text))

    async def done(self) -> None:
        self.done_sent = True
        self.duplex.events.append(("done", self.request_id))

    async def cancel(self) -> None:
        self.cancelled = True
        self.duplex.events.append(("cancel", self.request_id))


class FakeDuplex:
    def __init__(self):
        self.turns = 0
        self.requests: list[FakeRequest] = []
        self.events: list[tuple] = []

    async def turn_start(self) -> int:
        self.turns += 1
        self.events.append(("turn_start", self.turns))
        return self.turns

    def tts_request(self, voice=None, chunking=None, request_id=None):
        req = FakeRequest(self, request_id or f"r{len(self.requests) + 1}", voice, chunking)
        self.requests.append(req)
        return req


class FakeChat:
    """A chat stream that yields scripted deltas; with ``hold`` it never ends on its own."""

    def __init__(self, deltas: list[str], hold: bool = False, fail: Exception | None = None):
        self.scripted = list(deltas)
        self.hold = hold
        self.fail = fail
        self.text = ""
        self.closed = False
        self.started = False
        self.base_url = "http://fake"
        self.timeout = 1.0

    def start(self, loop) -> None:
        self.started = True

    async def deltas(self):
        if self.fail is not None:
            raise self.fail
        for delta in self.scripted:
            self.text += delta
            yield delta
            await asyncio.sleep(0)
        if self.hold:
            await asyncio.Event().wait()

    def close(self) -> None:
        self.closed = True


class ChatFactory:
    """Hands out FakeChats in order and records the messages each was given."""

    def __init__(self, *chats: FakeChat):
        self.chats = list(chats)
        self.messages: list[list[dict]] = []

    def __call__(self, messages: list[dict]) -> FakeChat:
        self.messages.append([dict(m) for m in messages])
        return self.chats.pop(0)


class FakePlayer:
    def __init__(self):
        self.rate = None
        self.chunks: list[bytes] = []
        self.cleared = 0

    def start(self, rate: int) -> None:
        self.rate = rate

    def enqueue(self, pcm: bytes) -> None:
        self.chunks.append(pcm)

    def clear(self) -> None:
        self.cleared += 1

    def is_playing(self) -> bool:
        return bool(self.chunks)


def _frame(text: str, turn_id: int = 1, seg: int = 0, endpoint: bool = False,
           final: bool = False) -> dict:
    return {"type": "transcription", "text": text, "segment_index": seg, "turn_id": turn_id,
            "endpoint": endpoint, "is_final": final}


def _session(factory: ChatFactory, errors: list[str] | None = None):
    duplex, player, log = FakeDuplex(), FakePlayer(), io.StringIO()
    clock = {"t": 100.0}
    session = va.ClassicSession(duplex, factory, player, log, clock=lambda: clock["t"],
                                voice="alice", chunking="clause",
                                on_error=(errors.append if errors is not None else print))
    return session, duplex, player, log, clock


async def _settle():
    for _ in range(5):
        await asyncio.sleep(0)


async def _finish_reply(session, duplex, req, clock, pcm: bytes = b"\x01\x00") -> None:
    """The server's audio for a request: start, one chunk, end."""
    await session.on_event(("audio_start", {"type": "audio_start", "request_id": req.request_id,
                                            "sample_rate": 24000}), clock["t"])
    await session.on_event(("pcm", pcm, req.request_id), clock["t"])
    await session.on_event(("audio_end", {"type": "audio_end", "request_id": req.request_id,
                                          "reason": "complete"}), clock["t"])


def _lines(log: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in log.getvalue().splitlines()]


# --- dispatch rule -----------------------------------------------------------------------

def test_interim_endpoint_and_final_with_the_same_text_dispatch_once():
    factory = ChatFactory(FakeChat(["Paris", "."]))
    session, duplex, player, log, clock = _session(factory)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("what is the capital"), 100.2)
        await session.on_transcript(_frame("what is the capital of France", endpoint=True), 100.9)
        await _settle()
        await session.on_transcript(_frame("What is the capital of France?", final=True), 101.9)
        await _settle()
        clock["t"] = 102.2
        await _finish_reply(session, duplex, duplex.requests[0], clock)
    asyncio.run(run())
    assert len(duplex.requests) == 1
    req = duplex.requests[0]
    assert req.sent == ["Paris", "."] and req.done_sent and not req.cancelled
    assert req.voice == "alice" and req.chunking == "clause"
    assert factory.messages[0][-1] == {"role": "user", "content": "what is the capital of France"}
    assert duplex.turns == 2  # the reply ended on its own: the next turn is open
    assert session.history == [{"role": "user", "content": "what is the capital of France"},
                               {"role": "assistant", "content": "Paris."}]


def test_an_audio_end_with_reason_error_is_recorded_as_an_error():
    # The server ends a reply it could not synthesize (an unknown voice, a sidecar
    # failure) with audio_end reason error and no audio: the record says so.
    factory = ChatFactory(FakeChat(["Paris", "."]))
    session, duplex, player, log, clock = _session(factory)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("what is the capital of France", endpoint=True), 100.9)
        await _settle()
        clock["t"] = 101.5
        req = duplex.requests[0]
        await session.on_event(("audio_end", {"type": "audio_end", "request_id": req.request_id,
                                              "reason": "error"}), clock["t"])
    asyncio.run(run())
    lines = _lines(log)
    assert len(lines) == 1
    assert lines[0]["reason"] == "error"
    assert lines[0]["cancelled"] is False
    assert lines[0]["first_audio_ms"] is None
    assert duplex.turns == 2  # the next turn still opens


def test_a_final_that_revises_the_dispatched_text_supersedes_the_reply():
    factory = ChatFactory(FakeChat(["Where", " to?"], hold=True), FakeChat(["Booking", " Boston."]))
    session, duplex, player, log, clock = _session(factory)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("book a flight to", endpoint=True), 100.9)
        await _settle()
        await session.on_transcript(_frame("book a flight to boston", final=True), 101.9)
        await _settle()
        await _finish_reply(session, duplex, duplex.requests[1], clock)
    asyncio.run(run())
    first, second = duplex.requests
    assert first.cancelled and factory.chats == [] and first.sent == ["Where", " to?"]
    assert second.sent == ["Booking", " Boston."] and second.done_sent
    assert factory.messages[1][-1] == {"role": "user", "content": "book a flight to boston"}
    assert [m["role"] for m in factory.messages[1]] == ["system", "user"]  # one user message
    assert player.cleared == 1
    lines = _lines(log)
    assert [(l["cancelled"], l["reason"]) for l in lines] == [(True, "superseded"),
                                                               (False, "complete")]


def test_frames_of_another_turn_are_ignored():
    factory = ChatFactory(FakeChat(["Hi."]))
    session, duplex, player, log, clock = _session(factory)

    async def run():
        await session.open_turn()  # turn 1
        await session.on_transcript(_frame("old words", turn_id=0, final=True), 100.5)
        await session.on_transcript(_frame("stale", turn_id=2, final=True), 100.6)
        await _settle()
        assert duplex.requests == []
        await session.on_transcript(_frame("hello there", turn_id=1, final=True), 100.9)
        await _settle()
    asyncio.run(run())
    assert len(duplex.requests) == 1
    assert factory.messages[0][-1]["content"] == "hello there"


def test_an_engine_without_endpoint_frames_dispatches_on_the_final():
    factory = ChatFactory(FakeChat(["Sure."]))
    session, duplex, player, log, clock = _session(factory)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("turn on the"), 100.2)
        await session.on_transcript(_frame("turn on the lights"), 100.5)
        await _settle()
        assert duplex.requests == []
        await session.on_transcript(_frame("Turn on the lights.", final=True), 101.5)
        await _settle()
    asyncio.run(run())
    assert len(duplex.requests) == 1 and duplex.requests[0].sent == ["Sure."]


# --- streaming the reply -----------------------------------------------------------------

def test_tokens_are_forwarded_one_by_one_then_done_and_the_history_carries_the_reply():
    factory = ChatFactory(FakeChat(["It", " is", " sunny", "."]), FakeChat(["Twenty", " degrees."]))
    session, duplex, player, log, clock = _session(factory)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("how is the weather", endpoint=True), 100.9)
        await _settle()
        await _finish_reply(session, duplex, duplex.requests[0], clock)
        await session.on_transcript(_frame("and the temperature", turn_id=2, endpoint=True), 105.0)
        await _settle()
    asyncio.run(run())
    first, second = duplex.requests
    assert duplex.events[:6] == [("turn_start", 1), ("send", "r1", "It"), ("send", "r1", " is"),
                                 ("send", "r1", " sunny"), ("send", "r1", "."), ("done", "r1")]
    assert [m["content"] for m in factory.messages[1][1:]] == [
        "how is the weather", "It is sunny.", "and the temperature"]
    assert second.sent == ["Twenty", " degrees."]


class _SSEHandler(BaseHTTPRequestHandler):
    status = 200
    deltas = ["Hello", " there", "."]
    bodies: list[dict] = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        _SSEHandler.bodies.append(json.loads(self.rfile.read(length)))
        if self.status != 200:
            self.send_error(self.status, "boom")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for delta in self.deltas:
            chunk = {"choices": [{"delta": {"content": delta}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *_args):
        pass


@pytest.fixture
def sse_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SSEHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _SSEHandler.status, _SSEHandler.bodies = 200, []
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def test_chat_stream_yields_the_deltas_in_order(sse_server):
    async def run():
        chat = va.ChatStream(sse_server, "m1", [{"role": "user", "content": "hi"}])
        chat.start(asyncio.get_running_loop())
        return [d async for d in chat.deltas()], chat.text
    deltas, text = asyncio.run(run())
    assert deltas == ["Hello", " there", "."] and text == "Hello there."
    assert _SSEHandler.bodies[0]["model"] == "m1" and _SSEHandler.bodies[0]["stream"] is True
    # Thinking is off unless asked: a reasoning pass delays the first spoken word.
    assert _SSEHandler.bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_chat_stream_keeps_thinking_when_asked(sse_server):
    async def run():
        chat = va.ChatStream(sse_server, None, [{"role": "user", "content": "hi"}], think=True)
        chat.start(asyncio.get_running_loop())
        return [d async for d in chat.deltas()]
    asyncio.run(run())
    assert "chat_template_kwargs" not in _SSEHandler.bodies[-1]
    assert "model" not in _SSEHandler.bodies[-1]


def test_an_http_failure_is_one_line_and_the_loop_goes_on(sse_server):
    errors: list[str] = []
    _SSEHandler.status = 500
    factory_calls = []

    def factory(messages):
        factory_calls.append(messages)
        return va.ChatStream(sse_server, None, messages)

    duplex, player, log = FakeDuplex(), FakePlayer(), io.StringIO()
    session = va.ClassicSession(duplex, factory, player, log, clock=lambda: 100.0,
                                on_error=errors.append)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("hello", final=True), 100.9)
        await asyncio.wait_for(session.reply.task, 5.0)
        # The server still answers the empty request with an audio_end: the next turn opens.
        await session.on_event(("audio_end", {"type": "audio_end",
                                              "request_id": duplex.requests[0].request_id,
                                              "reason": "complete"}), 101.0)
        _SSEHandler.status = 200
        await session.on_transcript(_frame("hello again", turn_id=2, final=True), 102.0)
        await asyncio.wait_for(session.reply.task, 5.0)
    asyncio.run(run())
    assert len(errors) == 1 and errors[0].startswith("error:") and "HTTP 500" in errors[0]
    assert "\n" not in errors[0]
    assert duplex.requests[0].done_sent  # the failed reply still closed its request
    assert duplex.requests[1].sent == ["Hello", " there", "."]
    assert _lines(log)[0]["failed"] is True


def test_an_unreachable_server_is_one_line():
    errors: list[str] = []
    duplex, player, log = FakeDuplex(), FakePlayer(), io.StringIO()
    session = va.ClassicSession(duplex, lambda m: va.ChatStream("http://127.0.0.1:9", None, m),
                                player, log, clock=lambda: 100.0, on_error=errors.append)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("hello", final=True), 100.9)
        await asyncio.wait_for(session.reply.task, 10.0)
    asyncio.run(run())
    assert len(errors) == 1 and errors[0].startswith("error: cannot reach")


# --- barge-in and the record -------------------------------------------------------------

def test_new_words_during_playback_cancel_once_and_the_new_utterance_dispatches_at_its_endpoint():
    factory = ChatFactory(FakeChat(["Long", " answer"], hold=True), FakeChat(["Okay."]))
    session, duplex, player, log, clock = _session(factory)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("tell me a story", endpoint=True), 100.9)
        await _settle()
        first = duplex.requests[0]
        clock["t"] = 101.3
        await session.on_event(("audio_start", {"type": "audio_start", "request_id": "r1",
                                                "sample_rate": 24000}), 101.3)
        await session.on_event(("pcm", b"\x01\x00", "r1"), 101.3)
        # A re-cased final of the same words: a confirmation, no cancel.
        await session.on_transcript(_frame("Tell me a story.", final=True), 101.9)
        assert not first.cancelled
        # The person talks over it: new words, not yet endpointed.
        await session.on_transcript(_frame("wait", seg=1), 102.5)
        await session.on_transcript(_frame("wait stop", seg=1), 102.7)
        await _settle()
        assert first.cancelled and duplex.events.count(("cancel", "r1")) == 1
        # Late audio of the cancelled reply is not played; its ack keeps the turn open.
        await session.on_event(("pcm", b"\x02\x00", "r1"), 102.8)
        await session.on_event(("audio_end", {"type": "audio_end", "request_id": "r1",
                                              "reason": "cancelled"}), 102.9)
        assert duplex.turns == 1
        await session.on_transcript(_frame("wait stop that", seg=1, endpoint=True), 103.2)
        await _settle()
    asyncio.run(run())
    assert len(duplex.requests) == 2
    assert player.chunks == [b"\x01\x00"] and player.cleared == 1
    # The interruption is its own utterance: the answered segment is not repeated, and
    # the history keeps the question with the partial reply before the new message.
    assert factory.messages[1][-1]["content"] == "wait stop that"
    assert [m["content"] for m in factory.messages[1][1:]] == [
        "tell me a story", "Long answer", "wait stop that"]
    lines = _lines(log)
    assert lines[0]["cancelled"] is True and lines[0]["reason"] == "barge-in"
    assert lines[0]["first_audio_ms"] == pytest.approx(400.0)
    assert lines[0]["reply"] == "Long answer"


def test_utterance_record_fields_and_summary():
    factory = ChatFactory(FakeChat(["Hi."]))
    session, duplex, player, log, clock = _session(factory)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("hello", endpoint=True), 100.5)
        await _settle()
        clock["t"] = 100.8
        await _finish_reply(session, duplex, duplex.requests[0], clock)
    asyncio.run(run())
    line = _lines(log)[0]
    assert set(line) >= {"type", "turn_id", "text", "t", "first_audio_ms", "cancelled",
                         "reason", "reply", "failed"}
    assert line["type"] == "utterance" and line["turn_id"] == 1 and line["text"] == "hello"
    assert line["t"] == pytest.approx(0.5) and line["first_audio_ms"] == pytest.approx(300.0)
    assert line["cancelled"] is False and line["reply"] == "Hi."
    assert "1 replies" in session.summary_line() and "300 ms" in session.summary_line()


def test_default_run_paths_are_under_runs_classic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(log=None, wav=None)
    log_path, wav_path = va.classic_output_paths(args)
    assert pathlib.Path(log_path).parent == pathlib.Path("runs/classic")
    assert pathlib.Path(wav_path).parent == pathlib.Path("runs/classic")
    assert (tmp_path / "runs" / "classic").is_dir()
    assert va.classic_output_paths(argparse.Namespace(log="a.jsonl", wav="b.wav")) == (
        "a.jsonl", "b.wav")


def test_terminal_lines_per_utterance_and_reply():
    factory = ChatFactory(FakeChat(["Hi", " there."]))
    duplex, player, log = FakeDuplex(), FakePlayer(), io.StringIO()
    said: list[str] = []
    clock = {"t": 100.0}
    session = va.ClassicSession(duplex, factory, player, log, clock=lambda: clock["t"],
                                say=said.append)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("hello", endpoint=True), 100.5)
        await _settle()
        clock["t"] = 100.8
        await _finish_reply(session, duplex, duplex.requests[0], clock)
    asyncio.run(run())
    assert said == ["... hello", "> hello", "< Hi there.  [first audio 300 ms]"]


def test_warm_up_reports_ready_or_the_failure(sse_server):
    said: list[str] = []
    assert asyncio.run(va.warm_up_chat(sse_server, "m1", say=said.append)) is True
    assert said[-1] == "chat model ready" and _SSEHandler.bodies[0]["model"] == "m1"
    _SSEHandler.status = 503
    said.clear()
    assert asyncio.run(va.warm_up_chat(sse_server, None, say=said.append)) is False
    assert said[-1].startswith("error:") and "HTTP 503" in said[-1]


def test_words_after_audio_end_while_the_tail_still_plays_clear_the_speaker():
    factory = ChatFactory(FakeChat(["A", " long", " fact."]), FakeChat(["Sure."]))
    session, duplex, player, log, clock = _session(factory)
    said: list[str] = []
    session.say = said.append

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("tell me a fact", endpoint=True), 100.9)
        await _settle()
        # The server has sent every byte (audio_end), but the player still holds audio.
        await _finish_reply(session, duplex, duplex.requests[0], clock)
        assert duplex.turns == 2 and player.is_playing()
        await session.on_transcript(_frame("that is", turn_id=2), 103.0)
        assert player.cleared == 1
        await session.on_transcript(_frame("that is enough", turn_id=2, endpoint=True), 103.6)
        await _settle()
    asyncio.run(run())
    assert "  (playback stopped)" in said
    assert len(duplex.requests) == 2 and not duplex.requests[0].cancelled


def test_the_question_growing_in_the_same_segment_keeps_the_whole_text():
    # The person paused mid-question (endpoint), the reply started, and they went on
    # in the same segment: the whole question is dispatched again, once.
    factory = ChatFactory(FakeChat(["Where?"], hold=True), FakeChat(["Booked."]))
    session, duplex, player, log, clock = _session(factory)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("book me a flight", endpoint=True), 100.9)
        await _settle()
        await session.on_transcript(_frame("book me a flight to"), 101.4)  # same segment grows
        await _settle()
        assert duplex.requests[0].cancelled
        await session.on_transcript(_frame("book me a flight to boston", endpoint=True), 102.0)
        await _settle()
    asyncio.run(run())
    assert factory.messages[1][-1]["content"] == "book me a flight to boston"
    assert [m["role"] for m in factory.messages[1]] == ["system", "user"]


# --- history after cancels and failures ------------------------------------------------

def test_a_barge_in_after_the_stream_ended_records_the_reply_once():
    # The chat stream finished and done() was sent; the audio is still arriving when
    # the person interrupts. The reply must appear once in the history.
    factory = ChatFactory(FakeChat(["Short", " reply."]), FakeChat(["Yes."]))
    session, duplex, player, log, clock = _session(factory)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("question", endpoint=True), 100.9)
        await asyncio.wait_for(session.reply.task, 2.0)  # the stream ended, done() sent
        await session.on_event(("audio_start", {"type": "audio_start", "request_id": "r1",
                                                "sample_rate": 24000}), 101.2)
        await session.on_event(("pcm", b"\x01\x00", "r1"), 101.2)
        await session.on_transcript(_frame("wait", seg=1), 101.5)  # new segment: barge-in
        await _settle()
        await session.on_transcript(_frame("wait a second", seg=1, endpoint=True), 102.0)
        await _settle()
    asyncio.run(run())
    assert [m["content"] for m in factory.messages[1][1:]] == [
        "question", "Short reply.", "wait a second"]
    assert [m["role"] for m in factory.messages[1]] == ["system", "user", "assistant", "user"]


def test_on_a_finals_only_engine_an_interruption_is_a_barge_in_not_a_supersede():
    # Whisper sends only finals. An interruption arrives as a final of a new segment:
    # it is the person's next utterance, not a revision of the dispatched one.
    factory = ChatFactory(FakeChat(["Long", " answer"], hold=True), FakeChat(["Fine."]))
    session, duplex, player, log, clock = _session(factory)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("first question", final=True), 100.9)
        await _settle()
        await session.on_event(("audio_start", {"type": "audio_start", "request_id": "r1",
                                                "sample_rate": 24000}), 101.2)
        await session.on_event(("pcm", b"\x01\x00", "r1"), 101.2)
        await session.on_transcript(_frame("Stop that.", seg=1, final=True), 102.0)
        await _settle()
    asyncio.run(run())
    assert duplex.requests[0].cancelled
    assert _lines(log)[0]["reason"] == "barge-in"
    assert [m["content"] for m in factory.messages[1][1:]] == [
        "first question", "Long answer", "Stop that."]


def test_a_revision_of_the_dispatched_segment_is_superseded_and_replaces_the_message():
    factory = ChatFactory(FakeChat(["Where", " to?"], hold=True), FakeChat(["Boston."]))
    session, duplex, player, log, clock = _session(factory)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("fly me to", endpoint=True), 100.9)
        await _settle()
        await session.on_transcript(_frame("fly me to boston", final=True), 101.9)
        await _settle()
    asyncio.run(run())
    assert _lines(log)[0]["reason"] == "superseded"
    assert [m["role"] for m in factory.messages[1]] == ["system", "user"]
    assert factory.messages[1][-1]["content"] == "fly me to boston"


def test_no_user_message_is_left_without_a_reply():
    # A reply that fails before any text, then a barge-in before the first token: the
    # next request must still alternate user and assistant.
    failed = FakeChat([], fail=RuntimeError("down"))
    factory = ChatFactory(failed, FakeChat([], hold=True), FakeChat(["Okay."]))
    errors: list[str] = []
    session, duplex, player, log, clock = _session(factory, errors)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("one", endpoint=True), 100.9)
        await asyncio.wait_for(session.reply.task, 2.0)
        await session.on_event(("audio_end", {"type": "audio_end", "request_id": "r1",
                                              "reason": "complete"}), 101.0)
        assert session.history == []  # nothing was answered
        await session.on_transcript(_frame("two", turn_id=2, endpoint=True), 102.0)
        await _settle()
        await session.on_transcript(_frame("three", turn_id=2, seg=1), 102.5)  # before any token
        await _settle()
        await session.on_transcript(_frame("three four", turn_id=2, seg=1, endpoint=True), 103.0)
        await _settle()
    asyncio.run(run())
    assert len(errors) == 1
    roles = [m["role"] for m in factory.messages[2]]
    assert roles == ["system", "user"], roles
    assert factory.messages[2][-1]["content"] == "three four"


def test_drain_errors_prints_one_line_per_error_frame():
    class _Duplex:
        async def errors(self):
            yield {"type": "error", "error_type": "unknown_voice", "message": "no such voice"}
            yield {"type": "error", "error_type": "invalid_request"}

    said: list[str] = []
    asyncio.run(va.drain_errors(_Duplex(), said.append))
    assert said == ["server error: unknown_voice: no such voice",
                    "server error: invalid_request: "]


# --- late frames of an answered segment, and the window while the next turn opens -------

def test_a_late_final_of_the_answered_segment_after_a_barge_in_is_ignored():
    factory = ChatFactory(FakeChat(["Long", " answer"], hold=True), FakeChat(["Okay."]))
    session, duplex, player, log, clock = _session(factory)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("tell me a story", endpoint=True), 100.9)
        await _settle()
        await session.on_event(("audio_start", {"type": "audio_start", "request_id": "r1",
                                                "sample_rate": 24000}), 101.2)
        await session.on_event(("pcm", b"\x01\x00", "r1"), 101.2)
        await session.on_transcript(_frame("wait", seg=1), 101.5)  # barge-in, new segment
        await _settle()
        # The endpoint hold confirms the answered segment late: not a new utterance.
        await session.on_transcript(_frame("Tell me a story.", seg=0, final=True), 101.8)
        await _settle()
        await session.on_transcript(_frame("wait stop", seg=1, endpoint=True), 102.2)
        await _settle()
    asyncio.run(run())
    assert len(duplex.requests) == 2
    assert factory.messages[1][-1]["content"] == "wait stop"


def test_a_frame_arriving_while_the_next_turn_opens_does_not_touch_the_answered_exchange():
    class GatedDuplex(FakeDuplex):
        def __init__(self):
            super().__init__()
            self.gate = asyncio.Event()

        async def turn_start(self) -> int:
            if self.turns >= 1:
                await self.gate.wait()  # the server's ack is in flight
            return await super().turn_start()

    factory = ChatFactory(FakeChat(["First."]), FakeChat(["Second."]))
    duplex, player, log = GatedDuplex(), FakePlayer(), io.StringIO()
    session = va.ClassicSession(duplex, factory, player, log, clock=lambda: 100.0)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("question one", endpoint=True), 100.9)
        await asyncio.wait_for(session.reply.task, 2.0)
        ending = asyncio.create_task(session.on_event(
            ("audio_end", {"type": "audio_end", "request_id": "r1", "reason": "complete"}),
            101.5))
        await _settle()  # on_audio_end is now waiting for the turn_started ack
        assert not ending.done()
        await session.on_transcript(_frame("question two", seg=1, endpoint=True), 101.6)
        await _settle()
        duplex.gate.set()
        await asyncio.wait_for(ending, 2.0)
        await _settle()
    asyncio.run(run())
    assert len(duplex.requests) == 2
    assert [m["content"] for m in factory.messages[1][1:]] == [
        "question one", "First.", "question two"]
    assert session.history[:2] == [{"role": "user", "content": "question one"},
                                   {"role": "assistant", "content": "First."}]


# --- the turn engine: playback edges on the mic clock, turn_end, barge ------------------

import types  # noqa: E402


class TurnDuplex(FakeDuplex):
    """A duplex that also takes playback reports."""

    def __init__(self):
        super().__init__()
        self.playback: list[tuple] = []

    async def playback_start(self, request, mic_sample=None) -> None:
        self.playback.append(("start", request, mic_sample))
        self.events.append(("playback_start", request, mic_sample))

    async def playback_stop(self, request, mic_sample=None, *, reply_sample) -> None:
        self.playback.append(("stop", request, mic_sample, reply_sample))
        self.events.append(("playback_stop", request, mic_sample, reply_sample))


class TurnPlayer(FakePlayer):
    """A fake player that also notes which reply its audio belongs to."""

    def __init__(self):
        super().__init__()
        self.begun: list[str] = []
        self.ended: list[str] = []

    def begin_reply(self, request_id: str) -> None:
        self.begun.append(request_id)

    def end_reply(self, request_id: str) -> None:
        self.ended.append(request_id)


def _turn_session(factory: ChatFactory, mic=None):
    duplex, player, log = TurnDuplex(), TurnPlayer(), io.StringIO()
    clock = {"t": 100.0}
    session = va.ClassicSession(duplex, factory, player, log, clock=lambda: clock["t"],
                                voice="alice", chunking="clause", turn_engine="tyger",
                                mic_clock=mic)
    return session, duplex, player, log, clock


def _turn_end(text: str, turn_id: int = 1, mic_sample: int = 32000, by: str = "model") -> dict:
    return {"type": "turn_end", "turn_id": turn_id, "text": text, "mic_sample": mic_sample,
            "by": by}


def _mic(adc: float | None = 10.0, mono: float | None = 500.0):
    mic = va.MicClock()
    mic.add(b"\x00\x00", types.SimpleNamespace(inputBufferAdcTime=adc))
    if mono is not None:
        mic.first_monotonic = mono
    return mic


def test_mic_sample_at_maps_a_dac_time_onto_the_mic_stream():
    assert va.mic_sample_at(10.5, 10.0) == 8000
    assert va.mic_sample_at(10.0, 10.0) == 0
    assert va.mic_sample_at(12.00003, 10.0) == 32000  # rounded to the nearest sample
    assert va.mic_sample_at(12.00004, 10.0) == 32001
    assert va.mic_sample_at(9.0, 10.0) == 0  # an edge before the first mic sample


def test_playback_mic_sample_uses_stream_time_else_the_monotonic_clock():
    mic = _mic(adc=10.0, mono=500.0)
    assert va.playback_mic_sample(10.25, 777.0, mic) == 4000
    # A host API that reports no stream time (None or zero) falls back to monotonic.
    assert va.playback_mic_sample(None, 500.5, mic) == 8000
    assert va.playback_mic_sample(10.25, 500.5, _mic(adc=0.0, mono=500.0)) == 8000
    # Before the mic has captured anything there is no clock to map onto.
    assert va.playback_mic_sample(10.25, 500.5, va.MicClock()) is None
    assert va.playback_mic_sample(10.25, 500.5, None) is None


def test_mic_clock_keeps_only_the_first_block_times():
    mic = va.MicClock()
    mic.add(b"\x01\x00" * 1600, types.SimpleNamespace(inputBufferAdcTime=3.25))
    first = mic.first_monotonic
    mic.add(b"\x02\x00" * 1600, types.SimpleNamespace(inputBufferAdcTime=3.35))
    assert mic.first_adc_time == 3.25 and mic.first_monotonic == first is not None


# --- the tracked player ---------------------------------------------------------------

def _tracked(rate: int = 16000):
    player = va.TrackedPlayer()
    player.sample_rate = rate
    edges: list = []
    player.on_edge = edges.append
    return player, edges


def _play(player, frames: int, dac: float) -> bytes:
    out = bytearray(frames * 2)
    player._callback(out, frames, types.SimpleNamespace(outputBufferDacTime=dac,
                                                        currentTime=dac - 0.05), None)
    return bytes(out)


def test_tracked_player_reports_the_first_sample_at_its_dac_time():
    player, edges = _tracked()
    _play(player, 4, 1.0)  # silence before the reply: no edge
    player.begin_reply("r1")
    player.enqueue(b"\x01\x00" * 6)
    out = _play(player, 4, 2.0)
    assert out == b"\x01\x00" * 4
    assert [(e.event, e.request_id, e.dac_time) for e in edges] == [("start", "r1", 2.0)]
    assert edges[0].monotonic is not None and edges[0].reply_sample is None


def test_tracked_player_start_inside_a_block_is_offset_by_the_samples_before_it():
    player, edges = _tracked(rate=24000)
    player.enqueue(b"\x05\x00" * 3)  # audio of no reply (another request's tail)
    player.begin_reply("r2")
    player.enqueue(b"\x01\x00" * 10)
    _play(player, 12, 4.0)
    assert [(e.event, e.request_id) for e in edges] == [("start", "r2")]
    assert edges[0].dac_time == pytest.approx(4.0 + 3 / 24000)


def test_tracked_player_reports_the_stop_once_the_ended_reply_has_drained():
    player, edges = _tracked()
    player.begin_reply("r1")
    player.enqueue(b"\x01\x00" * 6)
    player.end_reply("r1")  # audio_end: the reply is whole, but not yet played
    _play(player, 4, 2.0)
    assert [e.event for e in edges] == ["start"]
    _play(player, 4, 2.00025)  # 2 more samples, then silence
    assert [e.event for e in edges] == ["start", "stop"]
    stop = edges[1]
    assert stop.request_id == "r1" and stop.reply_sample == 6
    # The edge is the end of the last sample played.
    assert stop.dac_time == pytest.approx(2.00025 + 2 / 16000)


def test_tracked_player_stops_at_audio_end_when_the_reply_already_played_out():
    player, edges = _tracked()
    player.begin_reply("r1")
    player.enqueue(b"\x01\x00" * 4)
    _play(player, 4, 2.0)
    _play(player, 4, 2.00025)  # an underrun: the reply is open, so no stop yet
    assert [e.event for e in edges] == ["start"]
    player.end_reply("r1")
    assert [e.event for e in edges] == ["start", "stop"]
    assert edges[1].reply_sample == 4 and edges[1].dac_time == pytest.approx(2.00025)


def test_tracked_player_clear_stops_a_playing_reply_at_what_played():
    player, edges = _tracked()
    player.begin_reply("r1")
    player.enqueue(b"\x01\x00" * 100)
    _play(player, 10, 3.0)
    player.clear()
    assert [(e.event, e.reply_sample) for e in edges] == [("start", None), ("stop", 10)]
    assert edges[1].dac_time == pytest.approx(3.0 + 10 / 16000)
    assert player.is_playing() is False
    _play(player, 10, 3.1)
    assert len(edges) == 2  # nothing more for the cleared reply


def test_tracked_player_clear_before_the_first_sample_reports_nothing():
    player, edges = _tracked()
    player.begin_reply("r1")
    player.enqueue(b"\x01\x00" * 10)
    player.clear()
    player.end_reply("r1")
    _play(player, 10, 3.0)
    assert edges == []


def test_tracked_player_two_replies_back_to_back():
    player, edges = _tracked()
    player.begin_reply("r1")
    player.enqueue(b"\x01\x00" * 3)
    player.end_reply("r1")
    player.begin_reply("r2")
    player.enqueue(b"\x02\x00" * 3)
    _play(player, 8, 5.0)
    assert [(e.event, e.request_id) for e in edges] == [
        ("start", "r1"), ("stop", "r1"), ("start", "r2")]
    assert edges[1].reply_sample == 3
    assert edges[1].dac_time == pytest.approx(5.0 + 3 / 16000)
    assert edges[2].dac_time == pytest.approx(5.0 + 3 / 16000)


def test_tracked_player_without_replies_plays_like_the_plain_player():
    player, edges = _tracked()
    player.enqueue(b"\x01\x02\x03\x04")
    assert player.pull(6) == b"\x01\x02\x03\x04\x00\x00"
    assert bytes(player.played) == b"\x01\x02\x03\x04" and edges == []


# --- the classic session with the turn engine -----------------------------------------

def test_with_the_turn_engine_finals_do_not_end_the_turn_but_turn_end_does():
    factory = ChatFactory(FakeChat(["Paris", "."]))
    session, duplex, player, log, clock = _turn_session(factory)
    said: list[str] = []
    session.say = said.append

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("what is the capital", endpoint=True), 100.5)
        await session.on_transcript(_frame("what is the capital", final=True), 100.6)
        await session.on_transcript(_frame("of France", seg=1, final=True), 101.4)
        await _settle()
        assert duplex.requests == []  # the endpoint and the finals end no turn
        await session.on_turn_end(_turn_end("what is the capital of France"), 101.6)
        await _settle()
    asyncio.run(run())
    assert len(duplex.requests) == 1
    assert duplex.requests[0].sent == ["Paris", "."]
    assert factory.messages[0][-1] == {"role": "user",
                                       "content": "what is the capital of France"}
    assert "... what is the capital" in said and "> what is the capital of France" in said
    turn_ends = [line for line in _lines(log) if line["type"] == "turn_end"]
    assert turn_ends[0]["text"] == "what is the capital of France"
    assert turn_ends[0]["by"] == "model" and turn_ends[0]["mic_sample"] == 32000


def test_a_turn_end_without_words_dispatches_nothing():
    factory = ChatFactory()
    session, duplex, player, log, clock = _turn_session(factory)

    async def run():
        await session.open_turn()
        await session.on_turn_end(_turn_end("  "), 101.0)
    asyncio.run(run())
    assert duplex.requests == []


def test_audio_events_tag_the_player_with_the_reply():
    factory = ChatFactory(FakeChat(["Hi."]))
    session, duplex, player, log, clock = _turn_session(factory)

    async def run():
        await session.open_turn()
        await session.on_turn_end(_turn_end("hello"), 100.9)
        await _settle()
        await _finish_reply(session, duplex, duplex.requests[0], clock)
    asyncio.run(run())
    assert player.begun == ["r1"] and player.ended == ["r1"]
    assert player.chunks == [b"\x01\x00"] and duplex.turns == 2


def test_without_the_turn_engine_the_player_is_not_tagged():
    factory = ChatFactory(FakeChat(["Hi."]))
    duplex, player, log = FakeDuplex(), TurnPlayer(), io.StringIO()
    session = va.ClassicSession(duplex, factory, player, log, clock=lambda: 100.0)

    async def run():
        await session.open_turn()
        await session.on_transcript(_frame("hello", endpoint=True), 100.9)
        await _settle()
        await _finish_reply(session, duplex, duplex.requests[0], {"t": 101.0})
        assert await session.on_turn_end(_turn_end("ignored"), 101.5) is None
    asyncio.run(run())
    assert player.begun == [] and player.ended == [] and len(duplex.requests) == 1


def test_a_barge_clears_the_player_and_the_next_turn_end_dispatches():
    factory = ChatFactory(FakeChat(["Long", " answer"], hold=True), FakeChat(["Okay."]))
    session, duplex, player, log, clock = _turn_session(factory)

    async def run():
        await session.open_turn()
        await session.on_turn_end(_turn_end("tell me a story"), 100.9)
        await _settle()
        first = duplex.requests[0]
        await session.on_event(("audio_start", {"type": "audio_start", "request_id": "r1",
                                                "sample_rate": 24000}), 101.3)
        await session.on_event(("pcm", b"\x01\x00", "r1"), 101.3)
        await session.on_event(("barge", {"type": "barge", "request_id": "r1",
                                          "onset_mic_sample": 40000, "mic_sample": 44800}),
                               102.0)
        await _settle()
        assert player.cleared == 1
        # The server cancelled the request itself: no tts_cancel goes back.
        assert not first.cancelled and ("cancel", "r1") not in duplex.events
        # Its audio_end (interrupted) is the ack of the barge: the turn stays open.
        await session.on_event(("audio_end", {"type": "audio_end", "request_id": "r1",
                                              "reason": "interrupted"}), 102.1)
        assert duplex.turns == 1
        await session.on_turn_end(_turn_end("wait stop that", mic_sample=60000), 103.0)
        await _settle()
    asyncio.run(run())
    assert len(duplex.requests) == 2
    assert [m["content"] for m in factory.messages[1][1:]] == [
        "tell me a story", "Long answer", "wait stop that"]
    lines = _lines(log)
    barges = [line for line in lines if line["type"] == "barge"]
    assert barges[0]["request_id"] == "r1" and barges[0]["onset_mic_sample"] == 40000
    utterances = [line for line in lines if line["type"] == "utterance"]
    assert utterances[0]["cancelled"] is True and utterances[0]["reason"] == "barge"


def test_a_barge_of_an_ended_reply_still_playing_clears_the_speaker():
    factory = ChatFactory(FakeChat(["Hi."]))
    session, duplex, player, log, clock = _turn_session(factory)

    async def run():
        await session.open_turn()
        await session.on_turn_end(_turn_end("hello"), 100.9)
        await _settle()
        await _finish_reply(session, duplex, duplex.requests[0], clock)
        await session.on_event(("barge", {"type": "barge", "request_id": "r1",
                                          "onset_mic_sample": 1, "mic_sample": 2}), 101.5)
        await session.on_event(("barge", {"type": "barge", "request_id": "zz",
                                          "onset_mic_sample": 1, "mic_sample": 2}), 101.6)
    asyncio.run(run())
    assert player.cleared == 1


def test_a_turn_end_over_a_reply_with_no_words_yet_joins_the_two():
    factory = ChatFactory(FakeChat([], hold=True), FakeChat(["Sure."]))
    session, duplex, player, log, clock = _turn_session(factory)

    async def run():
        await session.open_turn()
        await session.on_turn_end(_turn_end("book a table"), 100.9)
        await _settle()
        await session.on_turn_end(_turn_end("for two"), 101.4)
        await _settle()
    asyncio.run(run())
    assert len(duplex.requests) == 2 and duplex.requests[0].cancelled
    assert factory.messages[1][1:] == [{"role": "user", "content": "book a table for two"}]


def test_a_turn_end_over_a_reply_that_spoke_keeps_the_exchange():
    factory = ChatFactory(FakeChat(["Which", " day?"], hold=True), FakeChat(["Done."]))
    session, duplex, player, log, clock = _turn_session(factory)

    async def run():
        await session.open_turn()
        await session.on_turn_end(_turn_end("book a table"), 100.9)
        await _settle()
        await session.on_turn_end(_turn_end("on friday"), 101.4)
        await _settle()
    asyncio.run(run())
    assert duplex.requests[0].cancelled
    assert [m["content"] for m in factory.messages[1][1:]] == [
        "book a table", "Which day?", "on friday"]


def test_playback_edges_go_out_on_the_mic_clock():
    factory = ChatFactory()
    session, duplex, player, log, clock = _turn_session(factory, mic=_mic(adc=10.0))

    async def run():
        await session.on_playback_edge(va.PlaybackEdge("start", "r1", 10.5, 900.0, None))
        await session.on_playback_edge(va.PlaybackEdge("stop", "r1", 12.0, 901.0, 36000))
    asyncio.run(run())
    assert duplex.playback == [("start", "r1", 8000), ("stop", "r1", 32000, 36000)]
    lines = [line for line in _lines(log) if line["type"] == "playback"]
    assert [(line["event"], line["mic_sample"]) for line in lines] == [
        ("start", 8000), ("stop", 32000)]
    assert lines[1]["reply_sample"] == 36000


def test_tracked_player_edges_reach_the_server_through_the_session():
    """The whole path: queued reply audio, the output callback, the playback frames."""
    factory = ChatFactory()
    session, duplex, _, log, clock = _turn_session(factory, mic=_mic(adc=10.0))
    player, edges = _tracked(rate=16000)

    async def run():
        player.begin_reply("r1")
        player.enqueue(b"\x01\x00" * 1600)
        player.end_reply("r1")
        _play(player, 1600, 10.5)
        _play(player, 1600, 10.6)
        for edge in edges:
            await session.on_playback_edge(edge)
    asyncio.run(run())
    assert duplex.playback == [("start", "r1", 8000), ("stop", "r1", 9600, 1600)]


def test_playback_edges_are_not_sent_without_the_turn_engine():
    duplex = TurnDuplex()
    session = va.ClassicSession(duplex, ChatFactory(), FakePlayer(), io.StringIO(),
                                mic_clock=_mic())
    asyncio.run(session.on_playback_edge(va.PlaybackEdge("start", "r1", 10.5, 900.0, None)))
    assert duplex.playback == []


def test_a_lost_turn_engine_goes_back_to_the_endpoint_frames():
    factory = ChatFactory(FakeChat(["Hi."]))
    session, duplex, player, log, clock = _turn_session(factory)
    said: list[str] = []
    session.say = said.append

    async def run():
        await session.open_turn()
        session.on_error_frame({"type": "error", "error_type": "invalid_request"})
        assert session.turn_engine == "tyger"
        session.on_error_frame({"type": "error", "error_type": "turn_engine_lost",
                                "message": "sidecar exited"})
        assert session.turn_engine is None
        await session.on_transcript(_frame("hello", endpoint=True), 101.0)
        await _settle()
        await session.on_playback_edge(va.PlaybackEdge("start", "r1", 10.5, 900.0, None))
    asyncio.run(run())
    assert len(duplex.requests) == 1 and duplex.playback == []
    assert any("turn engine lost" in line for line in said)


def test_drain_errors_hands_each_frame_on():
    class _Duplex:
        async def errors(self):
            yield {"type": "error", "error_type": "turn_engine_lost", "message": "gone"}

    seen: list[dict] = []
    asyncio.run(va.drain_errors(_Duplex(), lambda line: None, on_frame=seen.append))
    assert seen[0]["error_type"] == "turn_engine_lost"


# --- the classic run, end to end over fakes -------------------------------------------

class _LiveRequest:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self.sent: list[str] = []

    async def send(self, text: str, flush: bool = False) -> None:
        self.sent.append(text)

    async def done(self) -> None:
        pass

    async def cancel(self) -> None:
        pass


class _LiveDuplex:
    def __init__(self, refuse: bool = False):
        self.calls: list = []
        self.refuse = refuse
        self.requests: list[_LiveRequest] = []
        self.ready = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def enable_turn_engine(self, engine: str = "tyger", timeout: float = 10.0) -> str:
        self.calls.append(("enable_turn_engine", engine))
        if self.refuse:
            from voicemanager.client import TurnEngineError
            raise TurnEngineError("not on this engine", error_type="turn_engine_unsupported")
        return engine

    async def turn_start(self) -> int:
        self.calls.append(("turn_start",))
        return 1

    async def send_audio(self, pcm: bytes) -> None:
        self.calls.append(("send_audio", len(pcm)))

    def tts_request(self, voice=None, chunking=None, request_id=None):
        req = _LiveRequest(f"r{len(self.requests) + 1}")
        self.requests.append(req)
        return req

    async def transcripts(self):
        await asyncio.Event().wait()
        yield {}

    async def audio(self):
        await asyncio.Event().wait()
        yield ()

    async def turn_ends(self):
        yield _turn_end("hello there")
        await asyncio.Event().wait()

    async def errors(self):
        await asyncio.Event().wait()
        yield {}

    async def playback_start(self, request, mic_sample=None) -> None:
        pass

    async def playback_stop(self, request, mic_sample=None, *, reply_sample) -> None:
        pass


def _run_classic(monkeypatch, tmp_path, argv: list[str], duplex: _LiveDuplex):
    import voicemanager.client as vmc

    class FakeClient:
        stt_engine, tts_engine, last_tts_sample_rate = "audiocpp", "vibevoice", 24000

        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def duplex(self):
            return duplex

    async def fake_mic(dx, capture=None):
        for i in range(2):
            block = b"\x03\x00" * 1600
            if capture is not None:
                capture.add(block, types.SimpleNamespace(inputBufferAdcTime=20.0 + 0.1 * i))
            await dx.send_audio(block)
        await asyncio.Event().wait()

    async def ready(*args, **kwargs):
        return True

    def fake_start(self, rate):
        self.sample_rate = rate

    monkeypatch.setenv("VOICEMANAGER_TOKEN", "vmp_test")
    monkeypatch.setattr(vmc, "VoiceManagerClient", FakeClient)
    monkeypatch.setattr(va, "warm_up_chat", ready)
    monkeypatch.setattr(va.microturn_client, "_mic_task", fake_mic)
    monkeypatch.setattr(va.microturn_client.AudioPlayer, "start", fake_start)
    monkeypatch.setattr(va, "ChatStream", lambda *a, **k: FakeChat(["Hi."]))
    wav = tmp_path / "call.wav"
    args = va.build_parser().parse_args(
        ["classic", "--log", str(tmp_path / "call.jsonl"), "--wav", str(wav),
         "--duration", "0.3", *argv])
    asyncio.run(va.classic(args))
    return wav


def test_classic_run_opts_in_before_the_mic_and_dispatches_on_turn_end(monkeypatch, tmp_path):
    duplex = _LiveDuplex()
    _run_classic(monkeypatch, tmp_path, ["--turn-engine", "tyger"], duplex)
    names = [c[0] for c in duplex.calls]
    assert names[0] == "enable_turn_engine" and duplex.calls[0][1] == "tyger"
    assert names.index("enable_turn_engine") < names.index("send_audio")
    assert len(duplex.requests) == 1 and duplex.requests[0].sent == ["Hi."]


def test_classic_run_exits_when_the_server_refuses_the_turn_engine(monkeypatch, tmp_path):
    duplex = _LiveDuplex(refuse=True)
    with pytest.raises(SystemExit) as info:
        _run_classic(monkeypatch, tmp_path, ["--turn-engine", "tyger"], duplex)
    assert "turn_engine_unsupported" in str(info.value)
    assert ("send_audio", 3200) not in duplex.calls


def test_classic_run_without_the_flag_does_not_opt_in(monkeypatch, tmp_path):
    duplex = _LiveDuplex()
    wav = _run_classic(monkeypatch, tmp_path, [], duplex)
    assert all(c[0] != "enable_turn_engine" for c in duplex.calls)
    assert duplex.requests == []  # turn_end is not read without the opt-in
    assert not pathlib.Path(va.microturn_client.capture_paths(wav)[2]).exists()


def test_classic_capture_writes_the_mic_the_output_timeline_and_the_times(monkeypatch,
                                                                         tmp_path):
    import wave

    duplex = _LiveDuplex()
    wav = _run_classic(monkeypatch, tmp_path, ["--capture"], duplex)
    mic_path, out_path, json_path = va.microturn_client.capture_paths(wav)
    with wave.open(mic_path, "rb") as wf:
        assert wf.getframerate() == 16000 and wf.getnframes() == 3200
    assert pathlib.Path(out_path).exists()
    meta = json.loads(pathlib.Path(json_path).read_text())
    assert meta["mic"]["first_adc_time"] == 20.0 and meta["mic"]["samples"] == 3200
    assert meta["log"]["t0_monotonic"] is not None
