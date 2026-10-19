"""Nemotron cache-aware streaming STT pipeline.

Per-connection wrapper over a SHARED nemo.collections.asr.inference
CacheAwareRNNTPipeline. One NeMo *stream* per WebSocket connection (a unique
stream_id + its slot in the pipeline's per-stream state pool); the model itself
is loaded once on app.state. Replaces the retired disjoint-chunk STT loop.

Finals are driven by the pipeline's built-in acoustic end-of-utterance (EoU)
endpointer, surfaced additively via take_final() so the whisper/qwen
five-method contract is untouched.

Three facts about the NeMo 26.06 cache-aware API drive this design (verified against
the installed source under nemo/collections/asr/inference/ and the re-gate harness
scripts/spike_nemotron_2606.py):

  1. transcribe_step(requests) routes by Frame.stream_id and auto-manages per-stream
     state: a Frame with is_first=True creates the state (init_state), is_last=True
     deletes it AND frees the stream's slot in the shared feature bufferer. We rely on
     this instead of open_session()/close_session() — those reset the ENTIRE shared
     state pool (every concurrent connection), so a per-connection wrapper must never
     call them.
  2. The shared bufferer advances exactly one chunk_size_in_secs step per Frame, using
     the last chunk_size of its rolling buffer. Frames therefore MUST be chunk-aligned:
     feed() buffers incoming PCM and emits one Frame per full chunk (the live-feeding
     recipe). A short/long frame would drop or re-process audio.
  3. transcribe_step mutates shared tensors (feature buffer, cache context), so
     concurrent connections must serialize it. We hold the shared pipeline's
     `transcribe_lock` (attached at load time) around every step.

IMPORTANT: the NeMo transcribe step blocks. feed()/finalize() offload it via
asyncio.to_thread() (same pattern as qwen_stt.py).
"""
import asyncio
import logging

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


def _pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert 16-bit PCM bytes to float32 normalized [-1.0, 1.0].

    Duplicated for module self-containment (same pattern as qwen_stt.py).
    """
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


class NemotronSTTPipeline:
    """Per-connection cache-aware streaming pipeline."""

    def __init__(self, shared_pipeline, stream_id: int, att_context_size: list[int]):
        """Initialize one NeMo stream against the shared pipeline.

        Args:
            shared_pipeline: the single CacheAwareRNNTPipeline on app.state.
            stream_id: unique per-connection stream id (keys this connection's
                state + bufferer slot; must be unique across live connections).
            att_context_size: [left, right] cache-aware latency mode (informational
                here — the shared model's att_context_size is set once at load).
        """
        self._pipeline = shared_pipeline
        self._stream_id = stream_id
        self._att_context_size = att_context_size

        # Chunk size the bufferer advances per step (fact 2). feed() aligns to it.
        chunk_secs = getattr(shared_pipeline, "chunk_size_in_secs", None)
        if chunk_secs is None:
            raise ValueError(
                "shared_pipeline is missing chunk_size_in_secs — cannot size the "
                "cache-aware feed buffer"
            )
        self._chunk_samples = int(round(chunk_secs * SAMPLE_RATE))
        self._chunk_bytes = self._chunk_samples * 2  # 16-bit PCM
        self._buf = bytearray()                       # PCM awaiting a full chunk

        self._first_chunk_sent = False  # drives Frame.is_first on the first chunk
        self._closed = False            # True once the is_last flush frame is sent

        # Create this stream's state up front (idempotent with the is_first frame).
        self._state = self._pipeline.init_state(stream_id, self._make_options())
        self._segment_index = 0
        self._last_partial = ""
        self._pending_finals: list[tuple[str, str, int]] = []

    # --- NeMo-touching seams (filled from the spike recipe, en-0.6b model) ----------
    def _make_options(self):
        """Build the per-stream ASRRequestOptions.

        The English-only en-0.6b model is NOT prompt-enabled, so create_state needs
        no language_code; unset fields inherit the pipeline cfg defaults
        (stop_history_eou, asr_output_granularity). Overridden in unit tests so they
        never import NeMo / load a model.
        """
        from nemo.collections.asr.inference.streaming.framing.request_options import (
            ASRRequestOptions,
        )
        return ASRRequestOptions()

    def _build_frame(self, audio_f32: np.ndarray, *, is_last: bool):
        """Build one chunk-aligned Frame for this stream (zero-pads a short tail)."""
        import torch
        from nemo.collections.asr.inference.streaming.framing.request import Frame

        valid = int(audio_f32.shape[0])
        if valid < self._chunk_samples:
            padded = np.zeros(self._chunk_samples, dtype=np.float32)
            padded[:valid] = audio_f32
            audio_f32 = padded
        samples = torch.from_numpy(audio_f32.copy())
        is_first = not self._first_chunk_sent
        frame = Frame(
            samples=samples,
            stream_id=self._stream_id,
            is_first=is_first,
            is_last=is_last,
            length=valid,
            options=self._make_options() if is_first else None,
        )
        self._first_chunk_sent = True
        return frame

    def _run_step(self, frame):
        """Drive one transcribe_step under the shared pipeline lock (fact 3)."""
        lock = getattr(self._pipeline, "transcribe_lock", None)
        if lock is not None:
            with lock:
                return self._pipeline.transcribe_step([frame])[0]
        return self._pipeline.transcribe_step([frame])[0]

    def _transcribe(self, audio_f32: np.ndarray) -> tuple[str, str | None]:
        """Drive one cache-aware transcribe step for this stream (SYNC, blocks).

        Returns (current_partial, final_text_or_None); a non-None final_text means
        the endpointer fired an EoU on this step. Overridden in unit tests.
        """
        out = self._run_step(self._build_frame(audio_f32, is_last=False))
        final = out.final_transcript if out.final_transcript.strip() else None
        return out.partial_transcript, final
    # -------------------------------------------------------------------------

    async def feed(self, pcm_bytes: bytes) -> bool:
        """Buffer PCM, drive a step per full chunk; True if a new interim is available.

        Each EoU step completes an utterance: queue exactly one final
        carrying the current segment_index, advance the segment, and do NOT count it
        as an interim — the completed text is a final, surfaced via take_final(), so
        it is never emitted twice. A single feed() may span zero, one, or several
        chunks depending on frame size; interim/final bookkeeping is per chunk.
        """
        self._buf.extend(pcm_bytes)
        interim_available = False
        while len(self._buf) >= self._chunk_bytes:
            chunk = bytes(self._buf[: self._chunk_bytes])
            del self._buf[: self._chunk_bytes]
            audio_f32 = _pcm_to_float32(chunk)
            partial, final_text = await asyncio.to_thread(self._transcribe, audio_f32)
            if final_text is not None and final_text.strip():
                self._pending_finals.append((final_text, "en", self._segment_index))
                self._segment_index += 1
                self._last_partial = ""
            elif partial != self._last_partial:
                self._last_partial = partial
                if partial.strip():
                    interim_available = True
        return interim_available

    def get_current_text(self) -> tuple[str, str, int]:
        """Current live partial: (text, "en", segment_index)."""
        return self._last_partial, "en", self._segment_index

    def take_final(self) -> tuple[str, str, int] | None:
        """Pop the next EoU-finalized utterance, or None."""
        if self._pending_finals:
            return self._pending_finals.pop(0)
        return None

    async def finalize(self) -> tuple[str, str, int]:
        """Flush trailing audio at disconnect; return the last final.

        Sends one is_last frame (the buffered tail, or a silence chunk if the buffer
        is empty) which flushes the endpointer's residue AND frees this stream's slot
        in the shared bufferer + deletes its state.
        """
        flushed = await asyncio.to_thread(self._finalize_sync)
        if flushed and flushed.strip():
            result = (flushed, "en", self._segment_index)
            self._segment_index += 1
            return result
        if self._pending_finals:
            return self._pending_finals.pop()
        return self._last_partial, "en", self._segment_index

    def _finalize_sync(self) -> str:
        """Send the is_last flush frame and read the tail transcript (SYNC)."""
        if self._closed or not self._first_chunk_sent:
            # Nothing was ever streamed (or already flushed): no slot/state to free.
            self._closed = True
            return ""
        remaining = bytes(self._buf)
        self._buf.clear()
        if remaining:
            audio_f32 = _pcm_to_float32(remaining)
        else:
            # No partial chunk pending: send a silence chunk so is_last frees the slot.
            audio_f32 = np.zeros(self._chunk_samples, dtype=np.float32)
        out = self._run_step(self._build_frame(audio_f32, is_last=True))
        self._closed = True
        return out.final_transcript

    def reset(self) -> None:
        """Clear the live partial after a final.

        Does NOT advance segment_index — feed() already advanced it on the EoU step
        (unlike qwen, where reset() owns the increment).
        """
        self._last_partial = ""
        logger.debug("NemotronSTTPipeline reset for stream %d", self._stream_id)

    def dispose(self) -> None:
        """Release the stream's state on disconnect.

        delete_state() drops the per-stream state pool entry. The bufferer slot is
        freed by the is_last frame in finalize() (called first in the WS finally
        block); this is the safety net if state lingers.
        """
        try:
            self._pipeline.delete_state(self._stream_id)
        except Exception:
            logger.debug("delete_state failed for stream %d (already gone)", self._stream_id)
        self._pending_finals.clear()
        self._last_partial = ""
        self._buf.clear()
        logger.debug("NemotronSTTPipeline disposed for stream %d", self._stream_id)
