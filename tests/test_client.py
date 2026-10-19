"""Unit tests for VoiceManagerClient.

Tests cover: exception hierarchy, connection lifecycle, auth failure,
reader task frame routing, TTS audio collection, WAV helper round-trip,
stt_engine capture from handshake, stream_stt_live() async iterator.
All tests run without a running server.
"""
import asyncio
import io
import json
import tempfile
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voicemanager.client import (
    AuthenticationError,
    ConnectionError,
    STTError,
    TTSError,
    VoiceManagerClient,
    VoiceManagerError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HANDSHAKE_MSG = json.dumps({
    "type": "ready",
    "session_id": "test-session-123",
    "audio_format": {
        "sample_rate": 16000,
        "channels": 1,
        "bit_depth": 16,
        "encoding": "pcm",
    },
    "stt_engine": "whisper",
})

HANDSHAKE_MSG_QWEN = json.dumps({
    "type": "ready",
    "session_id": "test-session-456",
    "audio_format": {
        "sample_rate": 16000,
        "channels": 1,
        "bit_depth": 16,
        "encoding": "pcm",
    },
    "stt_engine": "qwen",
})

HANDSHAKE_MSG_NO_ENGINE = json.dumps({
    "type": "ready",
    "session_id": "test-session-789",
    "audio_format": {
        "sample_rate": 16000,
        "channels": 1,
        "bit_depth": 16,
        "encoding": "pcm",
    },
})


@pytest.fixture
def handshake_msg():
    return HANDSHAKE_MSG


def _make_mock_ws(handshake=HANDSHAKE_MSG):
    """Create a mock WebSocket that returns handshake on recv() and yields nothing."""
    ws = AsyncMock()
    ws.recv = AsyncMock(return_value=handshake)
    ws.send = AsyncMock()
    ws.close = AsyncMock()

    # Default __aiter__ yields nothing
    async def empty_aiter():
        return
        yield  # make it an async generator

    ws.__aiter__ = empty_aiter
    return ws


@pytest.fixture
def mock_ws():
    return _make_mock_ws()


@pytest.fixture
def client():
    return VoiceManagerClient(host="localhost", port=8000, token="test-key")


def _patch_connect(mock_ws_instance):
    """Return a patch context for websockets.asyncio.client.connect that yields mock_ws.

    websockets.asyncio.client.connect(url, ...) returns a connect object which is
    awaited to produce the actual WebSocket. We need our mock to mimic that:
    calling connect(url, ...) should return something awaitable that produces mock_ws.
    """
    # Create an async function that returns the mock_ws
    async def _fake_connect(url, **kwargs):
        return mock_ws_instance

    return patch("websockets.asyncio.client.connect", new=_fake_connect)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

class _AsyncIterWS:
    """Minimal mock WebSocket that yields a fixed list of messages."""

    def __init__(self, messages):
        self._messages = messages
        self.recv = AsyncMock()
        self.send = AsyncMock()
        self.close = AsyncMock()

    def __aiter__(self):
        return self._aiter_impl()

    async def _aiter_impl(self):
        for msg in self._messages:
            yield msg


async def _make_ws_iter(messages):
    """Create a mock ws that yields specific messages then stops."""
    return _AsyncIterWS(messages)


# ---------------------------------------------------------------------------
# TestExceptions
# ---------------------------------------------------------------------------

class TestExceptions:
    def test_hierarchy(self):
        assert issubclass(AuthenticationError, VoiceManagerError)
        assert issubclass(ConnectionError, VoiceManagerError)
        assert issubclass(STTError, VoiceManagerError)
        assert issubclass(TTSError, VoiceManagerError)

    def test_exception_message(self):
        for exc_class in (VoiceManagerError, AuthenticationError, ConnectionError, STTError, TTSError):
            exc = exc_class("test message")
            assert "test message" in str(exc)


# ---------------------------------------------------------------------------
# TestConnection
# ---------------------------------------------------------------------------

class TestConnection:
    async def test_connect_sets_session_id(self, client, mock_ws):
        with _patch_connect(mock_ws):
            await client._connect()

        assert client.session_id == "test-session-123"
        assert client.connected is True
        # Cleanup: cancel the reader task
        client._reader.cancel()
        try:
            await client._reader
        except asyncio.CancelledError:
            pass

    async def test_connect_auth_failure(self, client):
        import websockets.exceptions

        mock_response = MagicMock()
        mock_response.status_code = 403

        exc = websockets.exceptions.InvalidStatus(mock_response)

        async def _fail(url, **kwargs):
            raise exc

        with patch("websockets.asyncio.client.connect", new=_fail):
            with pytest.raises(AuthenticationError):
                await client._connect()

    async def test_connect_non_auth_failure(self, client):
        """Non-403 InvalidStatus should raise ConnectionError (not AuthenticationError)."""
        import websockets.exceptions

        mock_response = MagicMock()
        mock_response.status_code = 500

        exc = websockets.exceptions.InvalidStatus(mock_response)

        async def _fail(url, **kwargs):
            raise exc

        with patch("websockets.asyncio.client.connect", new=_fail):
            with pytest.raises(ConnectionError):
                await client._connect()

    async def test_disconnect_cancels_reader(self, client, mock_ws):
        with _patch_connect(mock_ws):
            await client._connect()

        await client._disconnect()

        assert client.connected is False
        assert client._ws is None

    async def test_context_manager(self, mock_ws):
        c = VoiceManagerClient(host="localhost", port=8000, token="test-key")
        with _patch_connect(mock_ws):
            async with c:
                assert c.connected is True
        assert c.connected is False

    async def test_connect_captures_stt_engine_whisper(self, client):
        """stt_engine is 'whisper' when handshake explicitly sends stt_engine='whisper'."""
        ws = _make_mock_ws(handshake=HANDSHAKE_MSG)
        with _patch_connect(ws):
            await client._connect()

        assert client.stt_engine == "whisper"
        client._reader.cancel()
        try:
            await client._reader
        except asyncio.CancelledError:
            pass

    async def test_connect_captures_stt_engine_qwen(self, client):
        """stt_engine is 'qwen' when handshake sends stt_engine='qwen'."""
        ws = _make_mock_ws(handshake=HANDSHAKE_MSG_QWEN)
        with _patch_connect(ws):
            await client._connect()

        assert client.stt_engine == "qwen"
        client._reader.cancel()
        try:
            await client._reader
        except asyncio.CancelledError:
            pass

    async def test_connect_stt_engine_defaults_to_whisper_when_missing(self, client):
        """stt_engine defaults to 'whisper' when field absent from handshake (backward compat)."""
        ws = _make_mock_ws(handshake=HANDSHAKE_MSG_NO_ENGINE)
        with _patch_connect(ws):
            await client._connect()

        assert client.stt_engine == "whisper"
        client._reader.cancel()
        try:
            await client._reader
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# TestReconnection
# ---------------------------------------------------------------------------

class TestReconnection:
    async def test_reconnect_sets_flag(self, client):
        import websockets.exceptions

        good_ws = _make_mock_ws()

        call_count = 0

        async def _connect_side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise websockets.exceptions.ConnectionClosed(None, None)
            return good_ws

        with patch("websockets.asyncio.client.connect", new=_connect_side_effect):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await client._reconnect()

        assert client.reconnected is True
        # Cleanup reader
        if client._reader:
            client._reader.cancel()
            try:
                await client._reader
            except asyncio.CancelledError:
                pass

    async def test_reconnect_max_retries(self, client):
        import websockets.exceptions

        async def _always_fail(url, **kwargs):
            raise websockets.exceptions.ConnectionClosed(None, None)

        with patch("websockets.asyncio.client.connect", new=_always_fail):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(ConnectionError, match="Failed to reconnect"):
                    await client._reconnect()


# ---------------------------------------------------------------------------
# TestReaderTask
# ---------------------------------------------------------------------------

class TestReaderTask:
    async def test_transcription_routed_to_stt_queue(self, client):
        transcription = json.dumps({
            "type": "transcription",
            "text": "hello world",
            "is_final": True,
            "segment_index": 0,
        })
        ws = await _make_ws_iter([transcription])
        client._ws = ws

        # Run the reader loop to completion (it ends when aiter exhausts)
        await client._reader_loop()

        # The stt_queue should have our transcription (before the None sentinel)
        items = []
        while not client._stt_queue.empty():
            items.append(client._stt_queue.get_nowait())

        text_items = [i for i in items if i is not None]
        assert len(text_items) == 1
        assert text_items[0]["type"] == "transcription"
        assert text_items[0]["text"] == "hello world"

    async def test_binary_routed_to_tts_queue(self, client):
        binary_data = b"\x00\x01\x02\x03"
        ws = await _make_ws_iter([binary_data])
        client._ws = ws

        await client._reader_loop()

        items = []
        while not client._tts_queue.empty():
            items.append(client._tts_queue.get_nowait())

        non_none = [i for i in items if i is not None]
        assert non_none == [binary_data]

    async def test_audio_start_routed_to_tts_queue(self, client):
        audio_start = json.dumps({
            "type": "audio_start",
            "request_id": "req-abc",
            "sample_rate": 22050,
        })
        ws = await _make_ws_iter([audio_start])
        client._ws = ws

        await client._reader_loop()

        items = []
        while not client._tts_queue.empty():
            items.append(client._tts_queue.get_nowait())

        non_none = [i for i in items if i is not None]
        assert len(non_none) == 1
        assert non_none[0]["type"] == "audio_start"
        assert non_none[0]["request_id"] == "req-abc"

    async def test_audio_end_routed_to_tts_queue(self, client):
        audio_end = json.dumps({
            "type": "audio_end",
            "request_id": "req-abc",
        })
        ws = await _make_ws_iter([audio_end])
        client._ws = ws

        await client._reader_loop()

        items = []
        while not client._tts_queue.empty():
            items.append(client._tts_queue.get_nowait())

        non_none = [i for i in items if i is not None]
        assert len(non_none) == 1
        assert non_none[0]["type"] == "audio_end"

    async def test_error_routed_to_error_queue(self, client):
        error_msg = json.dumps({
            "type": "error",
            "error_type": "timeout_warning",
            "message": "closing soon",
        })
        ws = await _make_ws_iter([error_msg])
        client._ws = ws

        await client._reader_loop()

        items = []
        while not client._error_queue.empty():
            items.append(client._error_queue.get_nowait())

        non_none = [i for i in items if i is not None]
        assert len(non_none) == 1
        assert non_none[0]["type"] == "error"
        assert non_none[0]["error_type"] == "timeout_warning"
        assert non_none[0]["message"] == "closing soon"


# ---------------------------------------------------------------------------
# TestTTS
# ---------------------------------------------------------------------------

class TestTTS:
    async def test_synthesize_tts_returns_audio_and_sample_rate(self, client):
        """Bypass reader loop — pre-populate tts_queue with protocol frames."""
        request_id = "REQ1"

        # Pre-populate the queue with the expected protocol sequence
        await client._tts_queue.put({
            "type": "audio_start",
            "request_id": request_id,
            "sample_rate": 22050,
        })
        await client._tts_queue.put(b"\x00\x01\x02\x03")
        await client._tts_queue.put({
            "type": "audio_end",
            "request_id": request_id,
        })

        # Mock the WebSocket send and uuid4
        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        with patch("uuid.uuid4", return_value=MagicMock(__str__=lambda self: request_id)):
            pcm, rate = await client.synthesize_tts("hello")

        assert pcm == b"\x00\x01\x02\x03"
        assert rate == 22050

    async def test_synthesize_tts_handles_stale_requests(self, client):
        """audio_end for wrong request_id is discarded; correct one terminates loop."""
        request_id = "REQ2"

        await client._tts_queue.put({
            "type": "audio_start",
            "request_id": "STALE_REQ",
            "sample_rate": 22050,
        })
        await client._tts_queue.put({
            "type": "audio_end",
            "request_id": "STALE_REQ",
        })
        await client._tts_queue.put({
            "type": "audio_start",
            "request_id": request_id,
            "sample_rate": 22050,
        })
        await client._tts_queue.put(b"\xAA\xBB")
        await client._tts_queue.put({
            "type": "audio_end",
            "request_id": request_id,
        })

        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        with patch("uuid.uuid4", return_value=MagicMock(__str__=lambda self: request_id)):
            pcm, rate = await client.synthesize_tts("test stale")

        assert pcm == b"\xAA\xBB"
        assert rate == 22050


# ---------------------------------------------------------------------------
# TestWAVHelpers
# ---------------------------------------------------------------------------

class TestWAVHelpers:
    def test_save_tts_to_file(self, tmp_path):
        # 100 frames of silent 16-bit PCM = 200 bytes of zeros
        pcm = bytes(200)
        out_path = str(tmp_path / "output.wav")

        VoiceManagerClient.save_tts_to_file(pcm, out_path, 22050)

        with wave.open(out_path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 22050
            assert wf.getnframes() == 100

    def test_pcm_to_wav_bytes_starts_with_riff(self):
        pcm = bytes(200)
        result = VoiceManagerClient.pcm_to_wav_bytes(pcm, 22050)

        assert result[:4] == b"RIFF"

    def test_pcm_to_wav_bytes_readable_by_wave(self):
        pcm = bytes(200)
        result = VoiceManagerClient.pcm_to_wav_bytes(pcm, 22050)

        with wave.open(io.BytesIO(result), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 22050
            assert wf.getnframes() == 100

    def test_pcm_to_wav_roundtrip(self):
        # Generate simple PCM pattern
        pcm = bytes(range(256)) * 4  # 1024 bytes = 512 frames at 16-bit
        result = VoiceManagerClient.pcm_to_wav_bytes(pcm, 16000)

        with wave.open(io.BytesIO(result), "rb") as wf:
            frames = wf.readframes(wf.getnframes())

        assert frames == pcm


# ---------------------------------------------------------------------------
# TestSTTLive
# ---------------------------------------------------------------------------

async def _collect_stream_stt_live(client, chunks):
    """Helper: run stream_stt_live and collect all yielded items."""
    results = []
    async def _aiter():
        for c in chunks:
            yield c

    async for item in client.stream_stt_live(_aiter()):
        results.append(item)
    return results


class TestSTTLive:
    async def test_stream_stt_live_yields_items_from_queue(self, client):
        """stream_stt_live yields transcription dicts that appear in _stt_queue."""
        transcription = {
            "type": "transcription",
            "text": "hello",
            "is_final": True,
            "segment_index": 0,
        }

        # Pre-populate queue: one result then sentinel
        await client._stt_queue.put(transcription)
        await client._stt_queue.put(None)

        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        results = await _collect_stream_stt_live(client, [b"\x00\x01"])

        assert len(results) == 1
        assert results[0]["text"] == "hello"
        assert results[0]["is_final"] is True

    async def test_stream_stt_live_yields_interim_and_final(self, client):
        """stream_stt_live yields both is_final=False and is_final=True dicts."""
        interim = {
            "type": "transcription",
            "text": "hel",
            "is_final": False,
            "segment_index": 0,
        }
        final = {
            "type": "transcription",
            "text": "hello",
            "is_final": True,
            "segment_index": 0,
        }

        await client._stt_queue.put(interim)
        await client._stt_queue.put(final)
        await client._stt_queue.put(None)

        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        results = await _collect_stream_stt_live(client, [b"\x00\x01"])

        assert len(results) == 2
        assert results[0]["is_final"] is False
        assert results[1]["is_final"] is True

    async def test_stream_stt_live_terminates_on_none_sentinel(self, client):
        """stream_stt_live stops when None sentinel received (connection closed)."""
        await client._stt_queue.put(None)

        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        results = await _collect_stream_stt_live(client, [b"\x00\x01"])

        assert results == []

    async def test_stream_stt_live_raises_stt_error_on_connection_lost(self, client):
        """stream_stt_live raises STTError when send raises ConnectionClosed."""
        import websockets.exceptions

        client._ws = AsyncMock()
        client._ws.send = AsyncMock(
            side_effect=websockets.exceptions.ConnectionClosed(None, None)
        )

        # Mock _reconnect so it doesn't actually try to reconnect
        async def _noop_reconnect():
            pass
        client._reconnect = _noop_reconnect

        async def _aiter():
            yield b"\x00\x01"

        with pytest.raises(STTError, match="Connection lost"):
            async for _ in client.stream_stt_live(_aiter()):
                pass

    async def test_existing_stream_stt_returns_list(self, client):
        """Existing stream_stt() still returns a list of dicts (unchanged behavior)."""
        transcription = {
            "type": "transcription",
            "text": "world",
            "is_final": True,
            "segment_index": 0,
        }
        await client._stt_queue.put(transcription)
        await client._stt_queue.put(None)

        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        async def _aiter():
            yield b"\x00\x01"

        result = await client.stream_stt(_aiter())

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == "world"


# ---------------------------------------------------------------------------
# TestSTTEndpoint — the endpoint flag surfaces in yielded dicts
# ---------------------------------------------------------------------------

class TestSTTEndpoint:
    """The client surfaces the additive speculative-endpoint flag from each
    frame: endpoint=true on the provisional endpoint frame,
    endpoint=false on ordinary interim and confirmed-final frames."""

    async def test_stream_stt_live_surfaces_endpoint_flag(self, client):
        """A provisional endpoint frame yields endpoint=true; ordinary interim
        and confirmed-final frames yield endpoint=false."""
        interim = {
            "type": "transcription",
            "text": "hel",
            "is_final": False,
            "endpoint": False,
            "segment_index": 0,
        }
        # Provisional end-of-utterance: the nemotron path's speculative frame
        # (is_final=False so it can still be revised, endpoint=True).
        provisional_endpoint = {
            "type": "transcription",
            "text": "hello there",
            "is_final": False,
            "endpoint": True,
            "segment_index": 0,
        }
        confirmed_final = {
            "type": "transcription",
            "text": "hello there",
            "is_final": True,
            "endpoint": False,
            "segment_index": 0,
        }

        await client._stt_queue.put(interim)
        await client._stt_queue.put(provisional_endpoint)
        await client._stt_queue.put(confirmed_final)
        await client._stt_queue.put(None)

        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        results = await _collect_stream_stt_live(client, [b"\x00\x01"])

        assert len(results) == 3
        # Ordinary interim: endpoint=false
        assert results[0]["is_final"] is False
        assert results[0]["endpoint"] is False
        # Provisional endpoint: endpoint=true
        assert results[1]["endpoint"] is True
        # Confirmed-final: endpoint=false, never retracted
        assert results[2]["is_final"] is True
        assert results[2]["endpoint"] is False

    async def test_stream_stt_surfaces_endpoint_flag(self, client):
        """The collecting stream_stt() also carries endpoint through each dict."""
        provisional_endpoint = {
            "type": "transcription",
            "text": "done",
            "is_final": False,
            "endpoint": True,
            "segment_index": 0,
        }
        await client._stt_queue.put(provisional_endpoint)
        await client._stt_queue.put(None)

        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        async def _aiter():
            yield b"\x00\x01"

        result = await client.stream_stt(_aiter())

        assert len(result) == 1
        assert result[0]["endpoint"] is True


# ---------------------------------------------------------------------------
# TestTTSLiveRequestId — the caller can obtain/supply the
# request_id of the stream_tts_live request it just started, so cancel_tts()
# can actually be called on the in-flight request (barge-in).
# ---------------------------------------------------------------------------

class TestTTSLiveRequestId:
    async def test_stream_tts_live_accepts_caller_supplied_request_id(self, client):
        """The caller supplies the request_id, so it can cancel the in-flight
        request WITHOUT parsing any server frame; the iteration then terminates."""
        rid = "CALLER-RID"
        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        # Frames the reader would have queued: synthesis started + one PCM chunk.
        await client._tts_queue.put(
            {"type": "audio_start", "request_id": rid, "sample_rate": 24000}
        )
        await client._tts_queue.put(b"\x11\x22")

        async def _text():
            yield "Hello there"

        yielded = []
        async for pcm in client.stream_tts_live(_text(), request_id=rid):
            yielded.append(pcm)
            # The caller already has the id (it supplied it) — cancel on it.
            await client.cancel_tts(rid)
            # Server acks the cancel with a prompt audio_end for that id.
            await client._tts_queue.put(
                {"type": "audio_end", "request_id": rid}
            )

        # Exactly the one pre-cancel PCM chunk was yielded, then iteration ended.
        assert yielded == [b"\x11\x22"]
        sent = [json.loads(c.args[0]) for c in client._ws.send.await_args_list]
        assert {"type": "tts_cancel", "request_id": rid} in sent

    async def test_stream_tts_live_sends_supplied_request_id_on_wire(self, client):
        """A caller-supplied request_id is the one carried on the tts_stream
        frames, so the id the caller holds matches the in-flight request."""
        rid = "WIRE-RID"
        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        await client._tts_queue.put(
            {"type": "audio_start", "request_id": rid, "sample_rate": 24000}
        )
        await client._tts_queue.put({"type": "audio_end", "request_id": rid})

        async def _text():
            yield "Hi"

        async for _ in client.stream_tts_live(_text(), request_id=rid):
            pass

        sent = [json.loads(c.args[0]) for c in client._ws.send.await_args_list]
        stream_frames = [f for f in sent if f.get("type") == "tts_stream"]
        assert stream_frames  # at least one text frame went out
        assert all(f["request_id"] == rid for f in stream_frames)

    async def test_stream_tts_live_exposes_generated_request_id(self, client):
        """When the caller does NOT supply a request_id, the one stream_tts_live
        mints is still reachable via client.last_tts_request_id — no code path
        generates an id the caller cannot reach (a structural check)."""
        rid = "GENERATED-RID"
        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        await client._tts_queue.put(
            {"type": "audio_start", "request_id": rid, "sample_rate": 24000}
        )
        await client._tts_queue.put(b"\x33\x44")
        await client._tts_queue.put({"type": "audio_end", "request_id": rid})

        async def _text():
            yield "Hi"

        yielded = []
        with patch("uuid.uuid4", return_value=MagicMock(__str__=lambda self: rid)):
            async for pcm in client.stream_tts_live(_text()):
                yielded.append(pcm)

        assert yielded == [b"\x33\x44"]
        assert client.last_tts_request_id == rid


# ---------------------------------------------------------------------------
# TestCancelTTS — cancel_tts has an honest, non-blocking
# contract: it does NOT wait for audio_end before returning, and the caller
# yields zero PCM for the cancelled request after cancel_tts returns — binary
# frames are discarded until the request's own audio_end arrives.
# ---------------------------------------------------------------------------

class TestCancelTTS:
    async def test_cancel_tts_returns_without_waiting_for_audio_end(self, client):
        """cancel_tts() returns promptly (it only marks + sends tts_cancel); it
        never blocks on the server's audio_end, which may lag while the executor
        unwinds a >=2 s synthesis."""
        rid = "CANCEL-RID"
        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        # No audio_end is ever queued — if cancel_tts waited for it, this would
        # time out. 50 ms is generous for a mark-a-set-and-send coroutine.
        await asyncio.wait_for(client.cancel_tts(rid), timeout=0.05)

        sent = [json.loads(c.args[0]) for c in client._ws.send.await_args_list]
        assert {"type": "tts_cancel", "request_id": rid} in sent

    async def test_cancel_tts_discards_binary_until_audio_end(self, client):
        """After cancel_tts() returns, binary frames for the cancelled request
        are discarded (never yielded) until that request's audio_end arrives,
        which then ends the iteration."""
        rid = "CANCEL-DISCARD-RID"
        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        # Pre-cancel: one legitimate PCM chunk the caller should receive.
        await client._tts_queue.put(
            {"type": "audio_start", "request_id": rid, "sample_rate": 24000}
        )
        await client._tts_queue.put(b"\x11\x22")

        async def _text():
            yield "Hello there"

        cancelled = False
        yielded = []
        async for pcm in client.stream_tts_live(_text(), request_id=rid):
            yielded.append(pcm)
            if not cancelled:
                cancelled = True
                await client.cancel_tts(rid)
                # Frames still in flight while the executor unwinds — these
                # belong to the cancelled request and MUST be discarded.
                await client._tts_queue.put(b"\x33\x44")
                await client._tts_queue.put(b"\x55\x66")
                # Server finally acks the cancel with audio_end for this id.
                await client._tts_queue.put(
                    {"type": "audio_end", "request_id": rid}
                )

        # Only the pre-cancel chunk was yielded; the two post-cancel chunks were
        # discarded, and audio_end ended the iteration.
        assert yielded == [b"\x11\x22"]

    def test_cancel_tts_docstring_has_no_false_synthesize_claim(self):
        """Structural: the docstring no longer claims request_id is
        'returned by synthesize_tts()' — synthesize_tts() returns (bytes, rate)."""
        doc = VoiceManagerClient.cancel_tts.__doc__ or ""
        assert "returned by synthesize_tts" not in doc


# ---------------------------------------------------------------------------
# TestFlushTTS — the client exposes a flush affordance that puts
# exactly ONE tts_stream frame with flush=true and done=false on the wire for
# the active request_id, and whose docstring warns that flushing MID-sentence
# audibly cuts a sentence in half (the correct use is a client-detected boundary).
# ---------------------------------------------------------------------------

class TestFlushTTS:
    async def test_flush_tts_sends_exactly_one_flush_frame(self, client):
        """Calling flush_tts(rid) puts exactly ONE tts_stream frame carrying
        flush=true and done=false for that request_id on the wire."""
        rid = "FLUSH-RID"
        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        await client.flush_tts(rid)

        sent = [json.loads(c.args[0]) for c in client._ws.send.await_args_list]
        flush_frames = [f for f in sent if f.get("flush") is True]
        assert len(flush_frames) == 1
        frame = flush_frames[0]
        assert frame["type"] == "tts_stream"
        assert frame["request_id"] == rid
        assert frame["done"] is False

    async def test_flush_tts_returns_without_waiting_for_audio(self, client):
        """flush_tts() only sends the flush frame — it never blocks on audio; a
        flush leaves the request OPEN and emits no audio_end."""
        rid = "FLUSH-NOWAIT-RID"
        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        # No audio_start/audio_end is ever queued — if flush_tts waited on one,
        # this would time out. 50 ms is generous for a single send.
        await asyncio.wait_for(client.flush_tts(rid), timeout=0.05)

    def test_flush_tts_docstring_warns_mid_sentence(self):
        """Structural: the docstring warns that flushing MID-sentence
        cuts a sentence in half, and that the correct use is a client-detected
        boundary."""
        doc = (VoiceManagerClient.flush_tts.__doc__ or "").lower()
        assert "mid-sentence" in doc
        assert "cut" in doc and "half" in doc
        assert "boundary" in doc


# ---------------------------------------------------------------------------
# TestTurns — client.turn_start() opens a turn and returns the
# server-assigned turn_id, and every transcription dict a Python consumer sees
# carries a turn_id key (0 when no turn was ever opened) so it can filter by
# turn without dropping to raw WebSocket.
# ---------------------------------------------------------------------------

class TestTurns:
    async def test_turn_start_sends_frame_and_returns_ack_turn_id(self, client):
        """turn_start() puts the bare turn_start control frame on the wire and
        returns the turn_id from the server's turn_started ack."""
        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        # The reader would have routed the server's ack here.
        await client._turn_queue.put({"type": "turn_started", "turn_id": 7})

        turn_id = await client.turn_start()

        assert turn_id == 7
        sent = [json.loads(c.args[0]) for c in client._ws.send.await_args_list]
        assert {"type": "turn_start"} in sent

    async def test_turn_started_routed_to_turn_queue(self, client):
        """The reader routes a turn_started frame to _turn_queue (not stt/tts),
        so turn_start() can await its ack there."""
        ack = json.dumps({"type": "turn_started", "turn_id": 3})
        ws = await _make_ws_iter([ack])
        client._ws = ws

        await client._reader_loop()

        items = []
        while not client._turn_queue.empty():
            items.append(client._turn_queue.get_nowait())
        non_none = [i for i in items if i is not None]
        assert len(non_none) == 1
        assert non_none[0]["type"] == "turn_started"
        assert non_none[0]["turn_id"] == 3

        # It did NOT leak onto the STT or TTS queues (only their None sentinels,
        # which the reader's teardown puts on every queue).
        while not client._stt_queue.empty():
            assert client._stt_queue.get_nowait() is None
        while not client._tts_queue.empty():
            assert client._tts_queue.get_nowait() is None

    async def test_turn_start_raises_stt_error_on_timeout(self, client):
        """turn_start() raises STTError when the turn_started ack never arrives
        rather than hanging forever."""
        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        with pytest.raises(STTError, match="[Tt]imed out"):
            await client.turn_start(timeout=0.05)

    async def test_stream_stt_live_dicts_carry_turn_id(self, client):
        """Every dict yielded by stream_stt_live() carries a turn_id key; a frame
        the server stamped keeps its value, a frame lacking one defaults to 0."""
        stamped = {
            "type": "transcription",
            "text": "in a turn",
            "is_final": True,
            "segment_index": 0,
            "turn_id": 5,
        }
        # A frame with no turn_id at all — the client must still expose the key.
        unstamped = {
            "type": "transcription",
            "text": "no turn",
            "is_final": True,
            "segment_index": 1,
        }
        await client._stt_queue.put(stamped)
        await client._stt_queue.put(unstamped)
        await client._stt_queue.put(None)

        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        results = await _collect_stream_stt_live(client, [b"\x00\x01"])

        assert len(results) == 2
        assert all("turn_id" in r for r in results)
        assert results[0]["turn_id"] == 5
        assert results[1]["turn_id"] == 0

    async def test_stream_stt_dicts_carry_turn_id(self, client):
        """Every dict returned by stream_stt() carries a turn_id key, defaulting
        to 0 when the frame did not stamp one."""
        unstamped = {
            "type": "transcription",
            "text": "world",
            "is_final": True,
            "segment_index": 0,
        }
        await client._stt_queue.put(unstamped)
        await client._stt_queue.put(None)

        client._ws = AsyncMock()
        client._ws.send = AsyncMock()

        async def _aiter():
            yield b"\x00\x01"

        result = await client.stream_stt(_aiter())

        assert len(result) == 1
        assert "turn_id" in result[0]
        assert result[0]["turn_id"] == 0


# ---------------------------------------------------------------------------
# TestTokenAuth: Bearer token on the WebSocket handshake and every REST call
# ---------------------------------------------------------------------------

class _FakeHTTPResponse:
    """Minimal stand-in for an httpx.Response."""

    def __init__(self, status_code: int = 200, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload if payload is not None else {"status": "ok"}
        self.text = json.dumps(self._payload)

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _RecordingAsyncClient:
    """Fake httpx.AsyncClient that records every request it is asked to make."""

    calls: list[dict] = []
    status_code: int = 200

    def __init__(self, *args, **kwargs) -> None:
        self.init_kwargs = kwargs

    async def __aenter__(self) -> "_RecordingAsyncClient":
        return self

    async def __aexit__(self, *exc_info) -> None:
        return None

    async def _record(self, method: str, url: str, **kwargs) -> _FakeHTTPResponse:
        _RecordingAsyncClient.calls.append({
            "method": method,
            "url": url,
            "init_kwargs": self.init_kwargs,
            **kwargs,
        })
        return _FakeHTTPResponse(_RecordingAsyncClient.status_code)

    async def get(self, url: str, **kwargs) -> _FakeHTTPResponse:
        return await self._record("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> _FakeHTTPResponse:
        return await self._record("POST", url, **kwargs)


@pytest.fixture
def recording_http():
    """Patch httpx.AsyncClient in the client module with the recording fake."""
    _RecordingAsyncClient.calls = []
    _RecordingAsyncClient.status_code = 200
    with patch("voicemanager.client.httpx.AsyncClient", new=_RecordingAsyncClient):
        yield _RecordingAsyncClient


def _request_headers(call: dict) -> dict:
    """Merge client-level and request-level headers of one recorded call."""
    headers: dict = {}
    headers.update(call["init_kwargs"].get("headers") or {})
    headers.update(call.get("headers") or {})
    return headers


def _request_has_no_token_in_url(call: dict, token: str) -> bool:
    """True when neither the URL nor its query params carry the token."""
    params = call.get("params") or {}
    return token not in call["url"] and token not in json.dumps(params)


class TestTokenAuth:
    async def test_constructor_takes_token(self):
        c = VoiceManagerClient(host="localhost", port=8000, token="tok-abc")
        assert c.token == "tok-abc"

    async def test_constructor_rejects_old_keyword(self):
        with pytest.raises(TypeError):
            VoiceManagerClient(host="localhost", port=8000, **{"api_" + "key": "x"})

    async def test_ws_connect_sends_bearer_header_and_bare_url(self, mock_ws):
        c = VoiceManagerClient(host="example", port=9000, token="tok-abc")
        seen: dict = {}

        async def _fake_connect(url, **kwargs):
            seen["url"] = url
            seen["kwargs"] = kwargs
            return mock_ws

        with patch("websockets.asyncio.client.connect", new=_fake_connect):
            await c._connect()

        assert seen["url"] == "ws://example:9000/ws/stream"
        assert "?" not in seen["url"]
        assert "tok-abc" not in seen["url"]
        headers = dict(seen["kwargs"]["additional_headers"])
        assert headers["Authorization"] == "Bearer tok-abc"

        c._reader.cancel()
        try:
            await c._reader
        except asyncio.CancelledError:
            pass

    async def test_ws_connect_uses_wss_with_ssl(self, mock_ws):
        c = VoiceManagerClient(host="example", port=443, token="tok", use_ssl=True)
        seen: dict = {}

        async def _fake_connect(url, **kwargs):
            seen["url"] = url
            return mock_ws

        with patch("websockets.asyncio.client.connect", new=_fake_connect):
            await c._connect()

        assert seen["url"] == "wss://example:443/ws/stream"
        c._reader.cancel()
        try:
            await c._reader
        except asyncio.CancelledError:
            pass

    @pytest.mark.parametrize("status", [401, 403])
    async def test_ws_rejection_raises_authentication_error(self, client, status):
        import websockets.exceptions

        mock_response = MagicMock()
        mock_response.status_code = status
        exc = websockets.exceptions.InvalidStatus(mock_response)

        async def _fail(url, **kwargs):
            raise exc

        with patch("websockets.asyncio.client.connect", new=_fail):
            with pytest.raises(AuthenticationError):
                await client._connect()

    async def test_health_sends_bearer_header(self, recording_http):
        c = VoiceManagerClient(host="example", port=9000, token="tok-abc")
        await c.health()

        assert len(recording_http.calls) == 1
        call = recording_http.calls[0]
        assert call["url"] == "http://example:9000/health"
        assert _request_headers(call)["Authorization"] == "Bearer tok-abc"
        assert _request_has_no_token_in_url(call, "tok-abc")

    async def test_info_sends_bearer_header(self, recording_http):
        c = VoiceManagerClient(host="example", port=9000, token="tok-abc")
        await c.info()

        assert len(recording_http.calls) == 1
        call = recording_http.calls[0]
        assert call["url"] == "http://example:9000/info"
        assert _request_headers(call)["Authorization"] == "Bearer tok-abc"
        assert _request_has_no_token_in_url(call, "tok-abc")

    async def test_upload_voice_sends_bearer_header(self, recording_http):
        c = VoiceManagerClient(host="example", port=9000, token="tok-abc")
        c.session_id = "sess-1"
        await c.upload_voice(b"RIFF....", filename="v.wav")

        assert len(recording_http.calls) == 1
        call = recording_http.calls[0]
        assert call["url"] == "http://example:9000/voice/upload"
        assert _request_headers(call)["Authorization"] == "Bearer tok-abc"
        assert _request_has_no_token_in_url(call, "tok-abc")
        assert call["params"] == {"session_id": "sess-1"}

    @pytest.mark.parametrize("status", [401, 403])
    async def test_upload_voice_rejection_raises_authentication_error(
        self, recording_http, status
    ):
        recording_http.status_code = status
        c = VoiceManagerClient(host="example", port=9000, token="bad")
        c.session_id = "sess-1"
        with pytest.raises(AuthenticationError):
            await c.upload_voice(b"RIFF....")

    @pytest.mark.parametrize("status", [401, 403])
    async def test_info_rejection_raises_authentication_error(self, recording_http, status):
        recording_http.status_code = status
        c = VoiceManagerClient(host="example", port=9000, token="bad")
        with pytest.raises(AuthenticationError):
            await c.info()

    def test_client_source_has_no_legacy_key_name(self):
        import inspect

        import voicemanager.client as client_module

        source = inspect.getsource(client_module)
        assert ("api" + "_key") not in source


class TestDefaultVoice:
    async def test_handshake_default_voice_is_read(self, client):
        handshake = json.dumps({
            "type": "ready",
            "session_id": "s1",
            "audio_format": {},
            "tts_engine": "audiocpp",
            "available_voices": ["majel", "narrator"],
            "default_voice": "majel",
        })
        with _patch_connect(_make_mock_ws(handshake)):
            await client._connect()
        assert client.default_voice == "majel"
        assert client.available_voices == ["majel", "narrator"]
        client._reader.cancel()
        try:
            await client._reader
        except asyncio.CancelledError:
            pass

    async def test_default_voice_is_none_when_absent(self, client, mock_ws):
        with _patch_connect(mock_ws):
            await client._connect()
        assert client.default_voice is None
        client._reader.cancel()
        try:
            await client._reader
        except asyncio.CancelledError:
            pass
