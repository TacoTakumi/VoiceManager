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
