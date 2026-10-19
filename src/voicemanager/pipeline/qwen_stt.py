"""Qwen3-ASR streaming STT pipeline.

Per-connection state machine wrapping qwen-asr's streaming API.
One instance per WebSocket connection — created on connect, disposed on disconnect.
"""
import asyncio
import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


def _pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert 16-bit PCM bytes to float32 normalized [-1.0, 1.0].

    Same conversion as pipeline/__init__.py but duplicated here to keep
    qwen_stt.py self-contained (no cross-import with whisper pipeline).
    """
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


class QwenSTTPipeline:
    """Per-connection Qwen3-ASR streaming pipeline.

    Analogous to STTPipeline but uses qwen-asr streaming API instead of
    faster-whisper + Silero VAD. One instance per WebSocket connection.

    The qwen-asr streaming state accumulates audio and updates state.text
    with progressive transcription results. No external VAD needed — the
    model handles voice detection internally.

    IMPORTANT: feed_sync() calls streaming_transcribe() which blocks.
    Always call via asyncio.to_thread() from async code.
    """

    def __init__(self, asr_model, chunk_size_sec: float = 2.0):
        """Initialize streaming state for this connection.

        Args:
            asr_model: Qwen3ASRModel.LLM() instance from app.state.
            chunk_size_sec: Audio segment size in seconds for streaming.
        """
        self._asr = asr_model
        self._chunk_size_sec = chunk_size_sec
        self._state = asr_model.init_streaming_state(
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=chunk_size_sec,
        )
        self._segment_index = 0
        self._last_text = ""

    def feed_sync(self, pcm_bytes: bytes) -> bool:
        """Feed PCM bytes to streaming engine (SYNCHRONOUS — blocks).

        Converts PCM to float32, calls streaming_transcribe(), checks for
        text delta. Returns True if state.text changed from last call.

        MUST be called via asyncio.to_thread() from async code.
        """
        audio_f32 = _pcm_to_float32(pcm_bytes)
        self._asr.streaming_transcribe(audio_f32, self._state)
        new_text = self._state.text
        if new_text != self._last_text:
            self._last_text = new_text
            return True
        return False

    async def feed(self, pcm_bytes: bytes) -> bool:
        """Async wrapper for feed_sync - runs in thread pool.

        Returns True if new text is available (state.text changed).
        """
        return await asyncio.to_thread(self.feed_sync, pcm_bytes)

    def get_current_text(self) -> tuple[str, str, int]:
        """Return (text, language, segment_index) of current partial result.

        Call after feed() returns True to get the interim transcription.
        """
        text = self._state.text
        language = getattr(self._state, "language", "")
        return text, language, self._segment_index

    def finalize_sync(self) -> tuple[str, str, int]:
        """Flush remaining audio and return final result (SYNCHRONOUS — blocks).

        Calls finish_streaming_transcribe() to process buffered audio.
        Returns (text, language, segment_index).

        MUST be called via asyncio.to_thread() from async code.
        """
        t0 = time.monotonic()
        self._asr.finish_streaming_transcribe(self._state)
        duration = time.monotonic() - t0
        text = self._state.text
        language = getattr(self._state, "language", "")
        logger.debug("Finalize completed in %.3fs, text length=%d", duration, len(text))
        return text, language, self._segment_index

    async def finalize(self) -> tuple[str, str, int]:
        """Async wrapper for finalize_sync — runs in thread pool.

        Returns (text, language, segment_index) of the final transcription.
        """
        return await asyncio.to_thread(self.finalize_sync)

    def reset(self):
        """Reset streaming state for a new utterance after emitting a final.

        Re-initialises the qwen-asr streaming state so the next audio
        is treated as a fresh segment.  Called after silence-triggered
        finalization to allow back-to-back utterances on the same
        connection.
        """
        self._state = self._asr.init_streaming_state(
            unfixed_chunk_num=2,
            unfixed_token_num=5,
            chunk_size_sec=self._chunk_size_sec,
        )
        self._segment_index += 1
        self._last_text = ""
        logger.debug("QwenSTTPipeline reset for segment %d", self._segment_index)

    def dispose(self):
        """Release pipeline resources on disconnect."""
        self._last_text = ""
        self._segment_index = 0
        # State object is per-pipeline; GC handles cleanup
        logger.debug("QwenSTTPipeline disposed")
