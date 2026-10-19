"""Tests for the additive `flush` field on tts_stream.

`flush=true` dispatches the ENTIRE current text buffer for synthesis exactly
as-is -- complete sentence or not -- and leaves the request OPEN: no
end-of-sentence sentinel is queued, no audio_end is emitted, and the same
request_id keeps accepting chunks.

Because a flush leaves the request open, the VM_TTS_STREAM_FLUSH_TIMEOUT safety
path is what eventually closes an abandoned one -- so flushing must not
disturb it: the timeout still fires, still closes the request exactly
once, and never re-speaks text the flush already dispatched.

Model-free: a scripted fake engine records the exact text it was asked to
synthesize and yields one deterministic chunk per call, so PCM can be tied back
to the text that produced it. A fake WebSocket records the frame sequence, which
is what the acceptance criteria are asserted against.
"""
import asyncio
import json

import authkit  # noqa: F401  (test auth env; import before voicemanager)

import numpy as np

from voicemanager.pipeline.tts import TTSPipeline, TTSStreamPipeline, split_sentences
from voicemanager.protocol import TTSStreamMessage


# --- test doubles ----------------------------------------------------------

class _FakeEngine:
    """Records synthesis calls; yields one 8-sample chunk per call."""

    def __init__(self):
        self.calls: list[str] = []

    @property
    def sample_rate(self) -> int:
        return 24000

    async def stream_synthesize(self, text: str, **kwargs):
        self.calls.append(text)
        yield np.zeros(8, dtype=np.float32)


class _FakeWebSocket:
    """Records the outbound frame sequence as ("text"|"bytes", payload) tuples.

    The test also appends its own ("client", ...) markers to this same list, so
    a single ordered list captures both directions -- that is what makes the
    "first PCM arrives before the client sent done" claim an assertion
    about wire ordering rather than wall-clock timing.
    """

    def __init__(self):
        self.frames: list[tuple[str, object]] = []

    async def send_text(self, data: str) -> None:
        self.frames.append(("text", json.loads(data)))

    async def send_bytes(self, data: bytes) -> None:
        self.frames.append(("bytes", data))

    # test-side helpers
    def mark(self, label: str) -> None:
        self.frames.append(("client", label))

    def texts(self) -> list[dict]:
        return [p for kind, p in self.frames if kind == "text"]

    def types(self) -> list[str]:
        return [p["type"] for kind, p in self.frames if kind == "text"]

    def kinds(self) -> list[str]:
        return [kind if kind != "text" else p["type"] for kind, p in self.frames]


def _make_pipeline(engine: _FakeEngine) -> TTSStreamPipeline:
    return TTSStreamPipeline(TTSPipeline(engine=engine, executor=None))


async def _settle(ws: _FakeWebSocket, predicate, timeout: float = 2.0) -> None:
    """Yield to the event loop until the synthesis task has produced frames."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if predicate(ws.frames):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting on frames: {ws.kinds()}")


# --- the field is additive ------------------------------------------------

def test_tts_stream_flush_defaults_false():
    msg = TTSStreamMessage(request_id="r1", text="hi")
    assert msg.flush is False
    assert json.loads(msg.model_dump_json())["flush"] is False


def test_tts_stream_flush_and_done_are_distinct_fields():
    msg = TTSStreamMessage(request_id="r1", text="hi", flush=True)
    assert msg.flush is True
    assert msg.done is False  # flush does not imply done


# --- flush dispatches the buffer and leaves the request open --------------

def test_flush_precondition_text_is_not_a_complete_sentence():
    # The acceptance rests on 'Hello there' being a fragment the sentence
    # splitter would NOT dispatch on its own under done=false.
    assert split_sentences("Hello there") == ["Hello there"]  # single, un-dispatchable


def test_flush_speaks_a_fragment_then_done_closes_the_same_request():
    engine = _FakeEngine()
    pipe = _make_pipeline(engine)
    ws = _FakeWebSocket()

    async def run():
        # 1. flush a fragment mid-request
        await pipe.feed_text(
            chunk="Hello there", done=False, flush=True,
            websocket=ws, request_id="r1",
        )
        await _settle(ws, lambda f: any(k == "bytes" for k, _ in f))

        # Everything the client has seen so far: audio_start + PCM, no audio_end.
        assert ws.kinds() == ["audio_start", "bytes"]
        assert engine.calls == ["Hello there"]  # PCM is for exactly the flushed text
        assert ws.texts()[0]["request_id"] == "r1"

        # 2. only NOW does the client send done=true
        ws.mark("done=true")
        await pipe.feed_text(
            chunk="General Kenobi.", done=True,
            websocket=ws, request_id="r1",
        )
        await pipe._synthesis_task

        # First PCM byte preceded the client's done frame, on the wire.
        first_pcm = next(i for i, (k, _) in enumerate(ws.frames) if k == "bytes")
        client_done = next(i for i, (k, p) in enumerate(ws.frames) if k == "client")
        assert first_pcm < client_done

        # Same request_id continued -- one audio_start, one audio_end.
        assert ws.types().count("audio_start") == 1
        assert ws.types().count("audio_end") == 1
        assert ws.kinds() == ["audio_start", "bytes", "client", "bytes", "audio_end"]
        assert engine.calls == ["Hello there", "General Kenobi."]
        assert ws.texts()[-1]["request_id"] == "r1"

    asyncio.run(run())


def test_flush_emits_no_audio_end_while_the_request_stays_open():
    engine = _FakeEngine()
    pipe = _make_pipeline(engine)
    ws = _FakeWebSocket()

    async def run():
        await pipe.feed_text(
            chunk="Hello there", done=False, flush=True,
            websocket=ws, request_id="r1",
        )
        await _settle(ws, lambda f: any(k == "bytes" for k, _ in f))
        # Give the synthesis loop room to (wrongly) close the request.
        await asyncio.sleep(0.05)
        assert "audio_end" not in ws.types()
        assert pipe._synthesis_task is not None and not pipe._synthesis_task.done()
        await pipe.cancel()

    asyncio.run(run())


def test_flush_clears_the_buffer_so_text_is_not_spoken_twice():
    engine = _FakeEngine()
    pipe = _make_pipeline(engine)
    ws = _FakeWebSocket()

    async def run():
        await pipe.feed_text(
            chunk="Hello there", done=False, flush=True,
            websocket=ws, request_id="r1",
        )
        await _settle(ws, lambda f: any(k == "bytes" for k, _ in f))
        await pipe.feed_text(
            chunk=" General Kenobi.", done=True,
            websocket=ws, request_id="r1",
        )
        await pipe._synthesis_task
        assert engine.calls == ["Hello there", "General Kenobi."]

    asyncio.run(run())


# --- an empty / whitespace-only flush is a silent no-op --------------------

def test_empty_flush_emits_no_frames_and_the_request_survives_it():
    # A flush with nothing buffered must not close the request, must not
    # emit audio_start/audio_end, and must not produce PCM. The request stays
    # open, so a later done=true still synthesizes on the SAME request_id.
    engine = _FakeEngine()
    pipe = _make_pipeline(engine)
    ws = _FakeWebSocket()

    async def run():
        await pipe.feed_text(
            chunk="", done=False, flush=True,
            websocket=ws, request_id="r1",
        )
        # Give the synthesis task room to (wrongly) emit an empty audio_start /
        # audio_end pair before we assert on silence.
        await asyncio.sleep(0.05)

        assert ws.frames == []          # zero frames of any kind
        assert engine.calls == []       # nothing was synthesized
        # The request is still open: its consumer task is alive and unclosed.
        assert pipe._synthesis_task is not None and not pipe._synthesis_task.done()

        # The request survived the no-op flush -- done=true still speaks on r1.
        await pipe.feed_text(
            chunk="Hi.", done=True,
            websocket=ws, request_id="r1",
        )
        await pipe._synthesis_task

        assert ws.kinds() == ["audio_start", "bytes", "audio_end"]
        assert ws.types().count("audio_start") == 1
        assert ws.types().count("audio_end") == 1
        assert engine.calls == ["Hi."]
        assert all(p["request_id"] == "r1" for p in ws.texts())

    asyncio.run(run())


def test_whitespace_only_flush_counts_as_empty_and_stays_silent():
    # 'empty' means the accumulated buffer has no speakable content -- whitespace
    # is not text to synthesize, so it takes the same silent path.
    engine = _FakeEngine()
    pipe = _make_pipeline(engine)
    ws = _FakeWebSocket()

    async def run():
        await pipe.feed_text(
            chunk="   \n\t ", done=False, flush=True,
            websocket=ws, request_id="r1",
        )
        await asyncio.sleep(0.05)

        assert ws.frames == []
        assert engine.calls == []
        assert pipe._synthesis_task is not None and not pipe._synthesis_task.done()

        # Held whitespace is not lost so much as harmless: the next real chunk
        # is still spoken, and the leading blanks are stripped off the synthesis.
        await pipe.feed_text(
            chunk="Hi.", done=True,
            websocket=ws, request_id="r1",
        )
        await pipe._synthesis_task

        assert ws.kinds() == ["audio_start", "bytes", "audio_end"]
        assert engine.calls == ["Hi."]

    asyncio.run(run())


def test_repeated_empty_flushes_stay_silent_mid_request():
    # A client that flushes on every LLM token boundary will hand us empty
    # buffers constantly -- after a real flush drains the buffer, the next flush
    # has nothing left. That must stay silent, not emit a stray audio frame.
    engine = _FakeEngine()
    pipe = _make_pipeline(engine)
    ws = _FakeWebSocket()

    async def run():
        await pipe.feed_text(
            chunk="Hello there", done=False, flush=True,
            websocket=ws, request_id="r1",
        )
        await _settle(ws, lambda f: any(k == "bytes" for k, _ in f))
        assert ws.kinds() == ["audio_start", "bytes"]

        # Buffer is now drained. Flush twice more with nothing in it.
        for _ in range(2):
            await pipe.feed_text(
                chunk="", done=False, flush=True,
                websocket=ws, request_id="r1",
            )
        await asyncio.sleep(0.05)

        # No new frames, no second audio_start, no premature audio_end.
        assert ws.kinds() == ["audio_start", "bytes"]
        assert engine.calls == ["Hello there"]
        await pipe.cancel()

    asyncio.run(run())


# --- the flush timeout safety path survives an explicit flush ----------------

async def _await_close(pipe: TTSStreamPipeline, timeout: float = 2.0) -> None:
    """Await the request's close, failing loudly if it never comes.

    A timeout-path regression where nothing queues the end sentinel leaves the
    synthesis task waiting on the queue forever. Bounded here so that shows up
    as a failed test rather than a hung suite.
    """
    try:
        await asyncio.wait_for(pipe._synthesis_task, timeout)
    except asyncio.TimeoutError:
        raise AssertionError(
            f"request never closed: synthesis task still running after {timeout}s"
        )


def _spy_on_queue_puts(pipe: TTSStreamPipeline) -> list:
    """Record every item put on the request's sentence queue, in order.

    The end-of-sentence sentinel is `None`, and it is what closes a request
    (audio_end). Recording the puts is how the structural half of the timeout rule -- "an
    explicit flush never queues the sentinel; only the timeout and done=true do"
    -- is asserted directly, rather than inferred from the frames.
    """
    puts: list = []
    queue = pipe._sentence_queue
    real_put = queue.put

    async def spy_put(item):
        puts.append(item)
        await real_put(item)

    queue.put = spy_put
    return puts


def test_flush_then_idle_timeout_closes_the_request_exactly_once():
    # Flush a fragment, hear it, then go silent. The flush timeout
    # is still armed on the still-open request, and it -- not the flush -- is
    # what closes r1: exactly one audio_end, and no second rendering of the text
    # the flush already spoke.
    engine = _FakeEngine()
    pipe = _make_pipeline(engine)
    ws = _FakeWebSocket()

    async def run():
        await pipe.feed_text(
            chunk="Hello.", done=False, flush=True,
            websocket=ws, request_id="r1",
        )
        await _settle(ws, lambda f: any(k == "bytes" for k, _ in f))

        # Before the timeout fires the request is OPEN -- nothing has closed it.
        assert ws.kinds() == ["audio_start", "bytes"]
        assert "audio_end" not in ws.types()
        assert pipe._synthesis_task is not None and not pipe._synthesis_task.done()

        # ... the client now sends nothing for > VM_TTS_STREAM_FLUSH_TIMEOUT, so
        # the WS receive loop calls flush_timeout() on the active request.
        ws.mark("flush timeout")
        await pipe.flush_timeout(ws, "r1")
        await _await_close(pipe)

        # Closed once, by the timeout, with no re-synthesis of the flushed text.
        assert ws.kinds() == ["audio_start", "bytes", "client", "audio_end"]
        assert ws.types().count("audio_end") == 1
        assert ws.types().count("audio_start") == 1
        assert engine.calls == ["Hello."]
        assert all(p["request_id"] == "r1" for p in ws.texts())

    asyncio.run(run())


def test_explicit_flush_queues_no_end_sentinel_so_only_the_timeout_closes():
    # The structural half of the timeout rule, and the reason the two paths can never
    # double-close: the sentinel is queued by the timeout (and by done=true),
    # never by a flush.
    engine = _FakeEngine()
    pipe = _make_pipeline(engine)
    ws = _FakeWebSocket()

    async def run():
        # Open the request with a chunk that dispatches nothing, so the sentence
        # queue exists and can be spied on before the flush touches it.
        await pipe.feed_text(
            chunk="", done=False,
            websocket=ws, request_id="r1",
        )
        puts = _spy_on_queue_puts(pipe)

        await pipe.feed_text(
            chunk="Hello.", done=False, flush=True,
            websocket=ws, request_id="r1",
        )
        await _settle(ws, lambda f: any(k == "bytes" for k, _ in f))

        # The flush dispatched the text and nothing else -- no sentinel.
        assert puts == ["Hello."]

        await pipe.flush_timeout(ws, "r1")
        await _await_close(pipe)

        # The timeout owns the close: it queued the one and only sentinel, and it
        # re-queued no text, because the flush had already drained the buffer.
        assert puts == ["Hello.", None]
        assert puts.count(None) == 1

    asyncio.run(run())


def test_timeout_after_a_flush_speaks_only_the_text_added_since_the_flush():
    # The flush drains the buffer, so when the timeout later flushes-and-closes it
    # finds only what arrived afterwards -- the flushed text is not spoken twice.
    engine = _FakeEngine()
    pipe = _make_pipeline(engine)
    ws = _FakeWebSocket()

    async def run():
        await pipe.feed_text(
            chunk="Hello.", done=False, flush=True,
            websocket=ws, request_id="r1",
        )
        await _settle(ws, lambda f: any(k == "bytes" for k, _ in f))

        # A trailing fragment arrives, then the client goes quiet mid-sentence.
        await pipe.feed_text(
            chunk=" General Kenobi", done=False,
            websocket=ws, request_id="r1",
        )
        await pipe.flush_timeout(ws, "r1")
        await _await_close(pipe)

        assert engine.calls == ["Hello.", "General Kenobi"]
        assert ws.types().count("audio_end") == 1
        assert ws.kinds() == ["audio_start", "bytes", "bytes", "audio_end"]

    asyncio.run(run())


def test_flush_timeout_on_a_fully_drained_buffer_still_closes_the_request():
    # The pathological case for a token-boundary flusher: every scrap of text has
    # already been flushed, so the timeout finds an empty buffer. It must still
    # close the request (one audio_end) and must not emit a stray second
    # audio_start for the empty flush.
    engine = _FakeEngine()
    pipe = _make_pipeline(engine)
    ws = _FakeWebSocket()

    async def run():
        await pipe.feed_text(
            chunk="Hello.", done=False, flush=True,
            websocket=ws, request_id="r1",
        )
        await _settle(ws, lambda f: any(k == "bytes" for k, _ in f))
        await pipe.feed_text(  # empty flush: no-op, buffer already drained
            chunk="", done=False, flush=True,
            websocket=ws, request_id="r1",
        )

        await pipe.flush_timeout(ws, "r1")
        await _await_close(pipe)

        assert ws.kinds() == ["audio_start", "bytes", "audio_end"]
        assert engine.calls == ["Hello."]

    asyncio.run(run())


def test_without_flush_a_fragment_is_still_held_in_the_buffer():
    # The default path is unchanged: a fragment under done=false is
    # buffered, not synthesized.
    engine = _FakeEngine()
    pipe = _make_pipeline(engine)
    ws = _FakeWebSocket()

    async def run():
        await pipe.feed_text(
            chunk="Hello there", done=False,
            websocket=ws, request_id="r1",
        )
        await asyncio.sleep(0.05)
        assert engine.calls == []
        assert ws.frames == []
        await pipe.cancel()

    asyncio.run(run())
