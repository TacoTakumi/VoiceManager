"""Audio processing pipeline for STT: buffer accumulation, VAD gating, async inference."""
import asyncio
import functools
import logging
import time

import numpy as np
import torch
from silero_vad import VADIterator

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VAD_CHUNK_SAMPLES = 512                     # required by Silero VAD at 16kHz
VAD_CHUNK_BYTES = VAD_CHUNK_SAMPLES * 2     # int16 = 2 bytes per sample
MAX_BUFFER_SAMPLES = SAMPLE_RATE * 30       # 30s cap


class STTPipeline:
    """Per-connection STT pipeline: buffer + VAD + inference.

    One instance per WebSocket connection. Created on connect, disposed on disconnect.
    """

    def __init__(self, vad_model, whisper_model, executor, vad_threshold: float, silence_ms: int):
        self._whisper = whisper_model
        self._executor = executor
        # vad_model may be None when TestClient skips lifespan — VAD disabled in that case
        self._vad_iter = (
            VADIterator(
                vad_model,
                threshold=vad_threshold,
                sampling_rate=SAMPLE_RATE,
                min_silence_duration_ms=silence_ms,
            )
            if vad_model is not None
            else None
        )
        self._buffer = bytearray()
        self._vad_leftover = bytearray()
        self._speech_detected = False

    def feed(self, data: bytes) -> bool:
        """Append PCM bytes to buffer. Returns True if inference should be triggered.

        VAD processes 512-sample chunks. Partial chunks are carried to next call.
        Returns True on VAD silence-after-speech boundary or 30s buffer cap.
        """
        self._buffer.extend(data)
        self._vad_leftover.extend(data)

        triggered = False
        if self._vad_iter is not None:
            while len(self._vad_leftover) >= VAD_CHUNK_BYTES:
                chunk_bytes = bytes(self._vad_leftover[:VAD_CHUNK_BYTES])
                self._vad_leftover = self._vad_leftover[VAD_CHUNK_BYTES:]
                chunk_f32 = _pcm_to_float32(chunk_bytes)
                result = self._vad_iter(torch.from_numpy(chunk_f32))
                if result is not None:
                    if "start" in result:
                        self._speech_detected = True
                    elif "end" in result and self._speech_detected:
                        triggered = True

        # Force inference if buffer exceeds 30s cap
        buffer_samples = len(self._buffer) // 2
        if buffer_samples >= MAX_BUFFER_SAMPLES:
            triggered = True

        return triggered

    async def run_inference(self, loop: asyncio.AbstractEventLoop):
        """Run transcription in executor thread, return segments and info.

        CRITICAL: list(segments) consumed INSIDE executor — the generator is lazy
        and calling next() triggers CTranslate2 inference. Must not iterate in event loop.
        """
        audio_np = _pcm_to_float32(bytes(self._buffer))
        t0 = time.monotonic()
        segments, info = await loop.run_in_executor(
            self._executor,
            functools.partial(_transcribe_sync, self._whisper, audio_np),
        )
        duration = time.monotonic() - t0
        # Full buffer clear after inference
        self._buffer.clear()
        self._vad_leftover.clear()
        if self._vad_iter is not None:
            self._vad_iter.reset_states()
        self._speech_detected = False
        logger.debug("Inference completed in %.3fs, %d segments", duration, len(segments))
        return segments, info, duration

    def dispose(self):
        """Release pipeline resources (called on disconnect)."""
        if self._vad_iter is not None:
            self._vad_iter.reset_states()
        self._buffer.clear()
        self._vad_leftover.clear()


class AcousticEndpointer:
    """Content-based end-of-utterance detector for engines with no native EoU.

    The audio.cpp backend's original endpoint trigger keyed on a GAP in inbound frames,
    which a continuously-streaming mic client (HermesConsole) never produces --
    its silence arrives as frames of quiet audio, not as an absence of frames. This
    class supplies the missing signal from the audio CONTENT: fed the same PCM the
    pipeline receives, it returns True once speech is followed by at least silence_ms
    of consecutive acoustic silence -- the moment the WS handler should FLUSH.

    Same silero VADIterator + 512-sample chunking as STTPipeline above; partial chunks
    carry to the next call. Sync and cheap (a per-chunk CPU inference well under 1 ms,
    the same cost the whisper path already pays on the event loop). With no vad_model
    (config dicts not built by _load_audiocpp_models, e.g. hand-built test configs) it
    is inert and the frame-gap fallback remains the only trigger.
    """

    def __init__(self, vad_model, vad_threshold: float, silence_ms: int):
        self._vad_iter = (
            VADIterator(
                vad_model,
                threshold=vad_threshold,
                sampling_rate=SAMPLE_RATE,
                min_silence_duration_ms=silence_ms,
            )
            if vad_model is not None
            else None
        )
        self._leftover = bytearray()
        self._speech_detected = False

    @property
    def armed(self) -> bool:
        """True when a VAD is present, i.e. classification is live. The idle-feed
        gate requires this: with no VAD every frame classifies as non-speech, so gating
        on an unarmed endpointer would gate ALL audio."""
        return self._vad_iter is not None

    @property
    def speech_active(self) -> bool:
        """True between silero's speech-start and speech-end events -- inside an
        utterance, including pauses shorter than silence_ms. Frames outside this span
        (and not carrying the end event) are inter-utterance audio, the idle-feed
        gate's definition of not-billed-to-the-decoder silence."""
        return self._speech_detected

    def feed(self, data: bytes) -> bool:
        """Classify one PCM frame; True when an utterance just ended acoustically."""
        if self._vad_iter is None:
            return False
        self._leftover.extend(data)
        endpointed = False
        while len(self._leftover) >= VAD_CHUNK_BYTES:
            chunk_bytes = bytes(self._leftover[:VAD_CHUNK_BYTES])
            self._leftover = self._leftover[VAD_CHUNK_BYTES:]
            chunk_f32 = _pcm_to_float32(chunk_bytes)
            result = self._vad_iter(torch.from_numpy(chunk_f32))
            if result is not None:
                if "start" in result:
                    self._speech_detected = True
                elif "end" in result and self._speech_detected:
                    self._speech_detected = False
                    endpointed = True
        return endpointed


def _pcm_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert 16-bit PCM bytes to float32 normalized [-1.0, 1.0]."""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def _transcribe_sync(whisper_model, audio_np: np.ndarray):
    """Synchronous wrapper for ThreadPoolExecutor.

    MUST call list() to consume the lazy generator before returning.
    The generator triggers CTranslate2 inference on each next() call.
    """
    segments_iter, info = whisper_model.transcribe(audio_np)
    return list(segments_iter), info
