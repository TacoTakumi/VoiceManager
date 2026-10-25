"""TTS engine abstraction, OpenVoice v2 implementation, and per-connection TTSPipeline.

TTSEngine ABC defines the interface for swappable TTS engines.
OpenVoiceTTSEngine implements it using MeloTTS + ToneColorConverter v2.
TTSPipeline orchestrates per-sentence synthesis and streaming over WebSocket.
"""
from abc import ABC, abstractmethod
import asyncio
import functools
import io
import logging
import os
import tempfile
import time
from collections.abc import AsyncIterator, Callable

import numpy as np
import soundfile as sf
import torch

from voicemanager.pipeline.clause_chunker import ClauseChunker
from voicemanager.pipeline.sentence_split import split_sentence

logger = logging.getLogger(__name__)

# Floor on the clause timer's wait when a deadline lands within rounding of now.
_CLAUSE_TIMER_MIN_S = 0.002


class TTSEngine(ABC):
    """Modular TTS engine interface. Swap engines by implementing this ABC."""

    #: True when this engine synthesizes in one blocking batch, so its
    #: stream_synthesize() does its inference in the CALLING context rather than
    #: yielding from a background thread. The pipeline reads this to decide it must
    #: run synthesize() in the TTS executor instead of consuming the generator on
    #: the event loop -- otherwise the loop is dead for the whole synthesis and the
    #: server reads no frames, including the barging-in human's STT audio and their
    #: tts_cancel.
    #:
    #: Engines that genuinely stream (VibeVoice, yielding from AsyncAudioStreamer)
    #: leave this False: they must be consumed on the loop, which is what delivers
    #: their chunks as they arrive.
    is_batch: bool = False

    @abstractmethod
    def load(self, device: str) -> None:
        """Load model weights onto device. Called once at startup."""
        ...

    @abstractmethod
    def synthesize(self, text: str, speaker_key: str = "EN-US", speed: float = 1.0) -> tuple[np.ndarray, int]:
        """Synthesize a single sentence. Returns (float32 numpy array, sample_rate).

        MUST be called from ThreadPoolExecutor. Does NOT split sentences.
        """
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Native output sample rate of the engine."""
        ...

    @abstractmethod
    async def stream_synthesize(self, text: str, **kwargs) -> "AsyncIterator[np.ndarray]":
        """Yield float32 numpy audio chunks asynchronously.

        For batch engines (OpenVoice), yields a single array via async wrapper.
        For streaming engines (VibeVoice), yields incremental chunks from
        AsyncAudioStreamer as they arrive from the generation thread.

        This is an async generator -- consume with `async for chunk in engine.stream_synthesize(...)`.
        """
        ...
        # Make it a generator (abstract async generators need a yield)
        yield  # type: ignore  # pragma: no cover


class OpenVoiceTTSEngine(TTSEngine):
    """OpenVoice v2 + MeloTTS engine.

    Synthesis pipeline per sentence:
    1. MeloTTS tts_to_file() -> temp WAV
    2. ToneColorConverter.convert() with src_se/tgt_se -> float32 numpy at 22050Hz
    """

    is_batch = True  # blocking synthesis -- the pipeline must offload it

    def __init__(self):
        self._melo_model = None
        self._tone_color_converter = None
        self._src_se = None
        self._device = "cpu"
        self._sample_rate_val = 44100  # MeloTTS native output rate

    def load(self, device: str, checkpoint_dir: str, base_speaker_dir: str, language: str = "EN") -> None:
        """Load MeloTTS + ToneColorConverter + source speaker embedding.

        Args:
            device: "cpu" or "cuda"
            checkpoint_dir: path to checkpoints_v2/converter (contains config.json + checkpoint.pth)
            base_speaker_dir: path to checkpoints_v2/base_speakers/ses (contains en-us.pth etc.)
            language: MeloTTS language code, default "EN"
        """
        from openvoice.api import ToneColorConverter
        from melo.api import TTS as MeloTTS

        self._device = device

        logger.info("Loading ToneColorConverter from %s", checkpoint_dir)
        self._tone_color_converter = ToneColorConverter(
            f"{checkpoint_dir}/config.json",
            device=device,
        )
        self._tone_color_converter.load_ckpt(f"{checkpoint_dir}/checkpoint.pth")

        logger.info("Loading MeloTTS (language=%s, device=%s)", language, device)
        self._melo_model = MeloTTS(language=language, device=device)

        se_path = f"{base_speaker_dir}/{language.lower()}-us.pth"
        logger.info("Loading source speaker embedding from %s", se_path)
        self._src_se = torch.load(se_path, map_location=device)

        logger.info("OpenVoiceTTSEngine loaded successfully")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate_val

    @property
    def tone_color_converter(self):
        """Access ToneColorConverter for voice upload se_extractor calls."""
        return self._tone_color_converter

    @property
    def src_se(self):
        """Source speaker embedding for tone color conversion."""
        return self._src_se

    def synthesize(self, text: str, speaker_key: str = "EN-US", speed: float = 1.0) -> tuple[np.ndarray, int]:
        """Synthesize one sentence via MeloTTS + ToneColorConverter (runs in executor).

        Returns (float32 numpy array, sample_rate). If no tgt_se is set, returns
        base MeloTTS output without voice cloning.
        """
        return self._synthesize_base(text, speaker_key, speed), self._sample_rate_val

    async def stream_synthesize(self, text: str, **kwargs) -> AsyncIterator[np.ndarray]:
        """Wrap synthesize() as single-chunk async yield for ABC compliance.

        OpenVoice is synchronous batch synthesis. This runs synthesize() in the
        calling context and yields the result as a single chunk. The caller
        (TTSPipeline) is responsible for running this inside an executor if needed.
        """
        speaker_key = kwargs.get("speaker_key", "EN-US")
        speed = kwargs.get("speed", 1.0)
        audio_np, _ = self.synthesize(text, speaker_key=speaker_key, speed=speed)
        yield audio_np

    def synthesize_cloned(self, text: str, tgt_se, speaker_key: str = "EN-US", speed: float = 1.0) -> tuple[np.ndarray, int]:
        """Synthesize with voice cloning — applies tone color conversion with target speaker embedding.

        Args:
            text: Single sentence to synthesize
            tgt_se: Target speaker embedding from se_extractor.get_se()
            speaker_key: MeloTTS speaker key (e.g., "EN-US")
            speed: Speech speed multiplier

        Returns (float32 numpy array at 22050Hz, sample_rate).
        """
        base_audio = self._synthesize_base(text, speaker_key, speed)

        # Wrap in BytesIO WAV for ToneColorConverter.convert()
        buf = io.BytesIO()
        sf.write(buf, base_audio, self._sample_rate_val, format="WAV", subtype="FLOAT")
        buf.seek(0)

        # Write to temp file because ToneColorConverter.convert() needs a path
        # (librosa.load inside convert() works with paths reliably)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(buf.getvalue())
            # Tone color conversion: src_se -> tgt_se
            converted = self._tone_color_converter.convert(
                audio_src_path=tmp_path,
                src_se=self._src_se,
                tgt_se=tgt_se,
                output_path=None,
                message="@VoiceManager",
            )
        finally:
            os.unlink(tmp_path)

        # ToneColorConverter resamples to 22050 Hz (per converter/config.json)
        return converted, 22050

    def _synthesize_base(self, text: str, speaker_key: str, speed: float) -> np.ndarray:
        """Run MeloTTS base synthesis to temp file, read back as float32 numpy."""
        spk2id = self._melo_model.hps.data.spk2id
        speaker_id = dict(spk2id).get(speaker_key, 0)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(tmp_fd)
        try:
            self._melo_model.tts_to_file(text, speaker_id, tmp_path, speed=speed)
            audio_np, sr = sf.read(tmp_path, dtype="float32")
            return audio_np
        finally:
            os.unlink(tmp_path)


def float32_to_pcm16(audio_np: np.ndarray) -> bytes:
    """Convert float32 [-1.0, 1.0] numpy audio to 16-bit PCM bytes.

    Output is always frame-aligned (even byte count) because int16 = 2 bytes/sample.
    """
    return (audio_np * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


def split_sentences(text: str, language: str = "EN") -> list[str]:
    """Split text into sentences using OpenVoice's splitter.

    Uses split_sentence() from sentence_split, a copy of openvoice.utils, so every
    engine can split without OpenVoice installed. It handles English and Chinese and
    merges short fragments. Add nltk.sent_tokenize as upgrade if abbreviation edge
    cases surface.
    """
    sentences = split_sentence(text, language_str=language)
    return [s.strip() for s in sentences if s.strip()]


class TTSPipeline:
    """Per-connection TTS pipeline: sentence splitting, async synthesis, streaming.

    One instance per WebSocket connection. Handles synthesis dispatch and cancellation.
    """

    def __init__(
        self,
        engine: TTSEngine,
        executor,
        language: str = "EN",
        speaker_key: str = "EN-US",
        speed: float = 1.0,
        voice: str | None = None,
    ):
        self._engine = engine
        self._executor = executor
        self._language = language
        self._speaker_key = speaker_key
        self._speed = speed
        self._voice = voice
        self.target_se = None  # per-session voice profile

    def dispose(self):
        """Release voice profile tensor reference."""
        self.target_se = None

    async def synthesize_and_stream(
        self,
        websocket,
        text: str,
        request_id: str,
        tgt_se=None,
        voice: str | None = None,
    ) -> None:
        """Synthesize text and stream PCM audio over WebSocket via stream_synthesize().

        For OpenVoice: splits into sentences, synthesizes each via stream_synthesize()
        (which yields a single chunk per sentence).
        For VibeVoice: feeds entire text to stream_synthesize() without splitting,
        yields multiple incremental chunks as they arrive from AsyncAudioStreamer.

        Sends audio_start before first chunk, audio_end after last.
        Cancellation via asyncio.CancelledError propagation.

        A synthesis that raises after audio_start was sent still terminates
        on the wire - exactly one audio_end with reason="error" for this
        request_id goes out (best-effort: the socket may already be gone on a
        fault path) before the exception propagates. Engine-agnostic; the happy
        path and the cancel path (whose terminal the session owns) are unchanged.
        """
        from voicemanager.protocol import AudioStartMessage, AudioEndMessage

        effective_tgt_se = self.target_se if self.target_se is not None else tgt_se
        effective_voice = voice or self._voice

        # An engine whose declared sample rate is only known once its backend has
        # answered (audiocpp: READY states the model's rate, 48 kHz or 24 kHz) exposes
        # ensure_ready(); await it so audio_start below carries the real rate. Engines
        # without it are untouched. A failure here is NOT raised: the synthesis below
        # re-raises the same failure after audio_start, so the request still ends with
        # its error audio_end.
        ensure_ready = getattr(self._engine, "ensure_ready", None)
        if ensure_ready is not None:
            try:
                await ensure_ready()
            except Exception:
                logger.debug("TTS engine ensure_ready failed before audio_start; "
                             "the synthesis will surface it", exc_info=True)

        # VibeVoice handles text internally -- no sentence splitting
        # OpenVoice still needs sentence splitting for its batch synthesis
        use_sentence_splitting = hasattr(self._engine, 'synthesize_cloned')

        if use_sentence_splitting:
            segments = split_sentences(text, self._language)
            # ToneColorConverter resamples to 22050 Hz; base MeloTTS outputs at native rate
            effective_sample_rate = 22050 if effective_tgt_se is not None else self._engine.sample_rate
        else:
            segments = [text] if text.strip() else []
            effective_sample_rate = self._engine.sample_rate

        if not segments:
            await websocket.send_text(
                AudioStartMessage(
                    request_id=request_id,
                    sample_rate=effective_sample_rate,
                ).model_dump_json()
            )
            await websocket.send_text(
                AudioEndMessage(request_id=request_id).model_dump_json()
            )
            return

        loop = asyncio.get_event_loop()

        # Send audio_start before first chunk
        await websocket.send_text(
            AudioStartMessage(
                request_id=request_id,
                sample_rate=effective_sample_rate,
                channels=1,
                bit_depth=16,
                encoding="pcm",
            ).model_dump_json()
        )

        try:
            await self._stream_segments(
                websocket, segments, request_id, effective_tgt_se, effective_voice,
                use_sentence_splitting, loop)
        except Exception:
            # The client has an OPEN audio stream (audio_start went out) and
            # would otherwise get eternal silence for this request_id. Terminate it
            # with exactly one audio_end reason="error", then let the exception
            # propagate (the session's done-callback logs it). Best-effort send: on
            # a fault path the socket may already be closed. CancelledError is a
            # BaseException and deliberately NOT caught - the cancel path's
            # terminal (reason="cancelled") is the session's to send.
            try:
                await websocket.send_text(
                    AudioEndMessage(request_id=request_id, reason="error")
                    .model_dump_json()
                )
            except Exception:
                logger.debug(
                    "Could not send error audio_end for request_id=%s", request_id)
            raise

        # Signal end of audio stream
        await websocket.send_text(
            AudioEndMessage(request_id=request_id).model_dump_json()
        )
        logger.debug("TTS complete: request_id=%s segments=%d", request_id, len(segments))

    async def _stream_segments(
        self,
        websocket,
        segments: list[str],
        request_id: str,
        effective_tgt_se,
        effective_voice: str | None,
        use_sentence_splitting: bool,
        loop,
    ) -> None:
        """The per-segment synthesis loop of synthesize_and_stream (unchanged
        behavior, hoisted so the terminal-frame guard wraps it whole)."""
        for segment in segments:
            # Use stream_synthesize() as primary inference path
            if use_sentence_splitting and effective_tgt_se is not None:
                # OpenVoice with voice cloning: use synthesize_cloned (stream_synthesize doesn't support cloning)
                audio_np, sr = await loop.run_in_executor(
                    self._executor,
                    functools.partial(
                        self._engine.synthesize_cloned,
                        segment,
                        effective_tgt_se,
                        self._speaker_key,
                        self._speed,
                    ),
                )
                pcm_bytes = float32_to_pcm16(audio_np)
                await websocket.send_bytes(pcm_bytes)
            elif getattr(self._engine, "is_batch", False):
                # A batch engine with no voice profile -- the default OpenVoice
                # setup. Its stream_synthesize() would do the inference right here on the
                # event loop, leaving the receive loop deaf for the whole synthesis. Offload
                # it exactly as the cloned path above does, so the server keeps reading
                # frames while we speak.
                audio_np, sr = await loop.run_in_executor(
                    self._executor,
                    functools.partial(
                        self._engine.synthesize,
                        segment,
                        self._speaker_key,
                        self._speed,
                    ),
                )
                pcm_bytes = float32_to_pcm16(audio_np)
                await websocket.send_bytes(pcm_bytes)
            else:
                # Streaming engines only: consume async stream_synthesize() directly on
                # the event loop -- VibeVoice yields incremental chunks from
                # AsyncAudioStreamer as they arrive, which is the point of it. Batch
                # engines are offloaded above instead.
                kwargs = {
                    "speaker_key": self._speaker_key,
                    "speed": self._speed,
                }
                if effective_voice is not None:
                    kwargs["voice"] = effective_voice

                async for chunk_np in self._engine.stream_synthesize(segment, **kwargs):
                    pcm_bytes = float32_to_pcm16(chunk_np)
                    await websocket.send_bytes(pcm_bytes)
                    logger.debug(
                        "Sent TTS audio chunk: request_id=%s bytes=%d",
                        request_id, len(pcm_bytes),
                    )


class TTSStreamPipeline:
    """Streaming text accumulation and sentence dispatch for TTS.

    Wraps a TTSPipeline via composition. Accumulates text chunks from tts_stream
    messages, dispatches complete sentences for synthesis as they are detected,
    and flushes the remainder on done=True or timeout.

    One instance per WebSocket connection, created alongside TTSPipeline.

    A request opened with chunking="clause" takes a different cut: its text goes
    through a ClauseChunker (one per request) instead of the sentence splitter, a
    per-request timer task polls the chunker at its next deadline, and the synthesis
    loop reports every PCM chunk it sends through ``_report_audio`` so the chunker
    knows how much audio is still unplayed. A sentence request never touches any of
    that.
    """

    def __init__(
        self,
        tts_pipeline: TTSPipeline,
        language: str = "EN",
        voice: str | None = None,
        clause_first_words: int = 4,
        clause_first_deadline_ms: int = 600,
        clause_later_words: int = 12,
        clause_margin_ms: int = 1000,
        clause_mark_min_words: int = 3,
        clause_clock: Callable[[], float] = time.monotonic,
    ):
        self._tts_pipeline = tts_pipeline
        self._language = language
        self._text_buffer: str = ""
        self._request_id: str | None = None
        self._synthesis_task: asyncio.Task | None = None
        # Tasks cut loose by abandon() and still unwinding in an executor thread. Held only
        # so the event loop keeps a strong reference to them: a pending task nobody holds
        # can be garbage-collected mid-unwind.
        self._abandoned: list[asyncio.Task] = []
        self._sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._done_received: bool = False
        self._audio_started: bool = False
        self._voice: str | None = voice
        # The open request's chunking mode, fixed at request start like voice.
        self._chunking: str = "sentence"
        # Clause-mode thresholds; each clause request gets a fresh chunker built from
        # them. The clock is injectable so tests can drive the deadline rule.
        self._clause_settings = {
            "first_piece_words": clause_first_words,
            "first_piece_deadline_ms": clause_first_deadline_ms,
            "later_piece_words": clause_later_words,
            "unplayed_margin_ms": clause_margin_ms,
            "mark_min_words": clause_mark_min_words,
        }
        self._clause_clock = clause_clock
        # Per-request clause state: None on a sentence request.
        self._clause_chunker: ClauseChunker | None = None
        self._clause_timer: asyncio.Task | None = None
        self._clause_wake: asyncio.Event | None = None

    @property
    def chunking(self) -> str:
        """The chunking mode of the current request ("sentence" or "clause")."""
        return self._chunking

    @property
    def synthesis_task(self) -> asyncio.Task | None:
        """The live synthesis task, if any — read-only.

        TTSSession needs to see this one to enforce "at most one TTS request in flight
        per connection" across both it and the one-shot `tts` task it owns.
        """
        return self._synthesis_task

    async def feed_text(
        self,
        chunk: str,
        done: bool,
        websocket,
        request_id: str,
        voice: str | None = None,
        flush: bool = False,
        chunking: str = "sentence",
    ) -> None:
        """Accumulate a text chunk and dispatch complete sentences.

        Voice and chunking are fixed at request start. If a subsequent feed_text()
        call for the same request_id specifies a different value, it is silently
        ignored. Either only takes effect when a new request_id begins.

        On first call for a new request_id, starts a synthesis consumer task.
        Dispatches all-but-last sentence from split_sentences() result.
        On done=True, flushes remaining buffer as final sentence.

        Args:
            chunk: Text fragment from tts_stream message
            done: True if this is the final chunk
            websocket: Active WebSocket connection for audio streaming
            request_id: Client correlation ID
            voice: Optional voice name override (VibeVoice only); fixed at request start
            flush: Dispatch the whole buffer as-is and leave the request open.
                A flush with an empty or whitespace-only buffer is
                a silent no-op. Ignored when done=True, which closes the
                request anyway.
            chunking: "sentence" or "clause"; fixed at request start. Validated
                upstream by the protocol model.
        """
        # New request — start fresh
        if self._request_id != request_id:
            # The superseded synthesis is cut loose, never awaited. cancel() would
            # await it, and a task blocked inside run_in_executor cannot be interrupted --
            # so awaiting here parked whoever called us (the receive loop) for the rest of
            # the synthesis, deaf to the audio of the human barging in. TTSSession
            # has already silenced the old request by the time we get here; this only has
            # to let go of it.
            self.abandon()
            self._request_id = request_id
            self._text_buffer = ""
            self._done_received = False
            self._audio_started = False
            self._sentence_queue = asyncio.Queue()
            # Voice and chunking are fixed at request start
            self._voice = voice
            self._chunking = chunking
            # Start synthesis consumer task (asyncio.create_task for full-duplex)
            self._synthesis_task = asyncio.create_task(
                self._synthesis_loop(websocket, request_id)
            )
            if chunking == "clause":
                self._start_clause(request_id)

        if self._clause_chunker is not None:
            await self._feed_clause(chunk, done, flush)
            return

        self._text_buffer += chunk

        if done:
            # Flush remaining buffer as final sentence
            self._done_received = True
            if self._text_buffer.strip():
                await self._sentence_queue.put(self._text_buffer.strip())
            self._text_buffer = ""
            # Signal no more sentences
            await self._sentence_queue.put(None)
        elif flush:
            # Dispatch the entire buffer as-is - sentence or fragment - and
            # leave the request OPEN: no None sentinel, so no audio_end, and this
            # request_id keeps accepting chunks.
            # An empty/whitespace-only buffer dispatches NOTHING - a client
            # flushing on every token boundary hands us empty buffers routinely, and
            # each must be silent: no queue entry, hence no audio_start/PCM, and (as
            # on any flush) no sentinel, hence no audio_end.
            if self._text_buffer.strip():
                await self._sentence_queue.put(self._text_buffer.strip())
                self._text_buffer = ""
        else:
            # Dispatch complete sentences, keep fragment in buffer
            sentences = split_sentences(self._text_buffer, self._language)
            if len(sentences) > 1:
                for s in sentences[:-1]:
                    await self._sentence_queue.put(s)
                self._text_buffer = sentences[-1]
            # else: single element or empty — keep in buffer, may be fragment

    async def flush_timeout(self, websocket, request_id: str) -> None:
        """Safety flush when no chunk arrives within timeout.

        Called by WebSocket handler when asyncio.wait_for times out during
        active streaming. Flushes buffer and signals end of sentences.
        """
        if self._clause_chunker is not None and self._request_id == request_id:
            self._stop_clause_timer()
            for piece in self._clause_chunker.drain():
                logger.warning(
                    "Flush timeout triggered for request_id=%s, flushing %d held chars",
                    request_id, len(piece),
                )
                await self._sentence_queue.put(piece)
        elif self._request_id == request_id and self._text_buffer.strip():
            logger.warning(
                "Flush timeout triggered for request_id=%s, flushing %d chars",
                request_id, len(self._text_buffer),
            )
            await self._sentence_queue.put(self._text_buffer.strip())
            self._text_buffer = ""
        self._done_received = True
        await self._sentence_queue.put(None)

    # --- clause mode ----------------------------------------------------------

    def _start_clause(self, request_id: str) -> None:
        """Give a new clause request its own chunker and timer task."""
        chunker = ClauseChunker(**self._clause_settings, clock=self._clause_clock)
        wake = asyncio.Event()
        timer = asyncio.create_task(
            self._clause_timer_loop(chunker, wake, self._sentence_queue))
        self._clause_chunker = chunker
        self._clause_wake = wake
        self._clause_timer = timer
        # The timer lives only as long as the request's synthesis: a synthesis that
        # ends on its own (an error included) takes the timer with it.
        self._synthesis_task.add_done_callback(lambda _task: timer.cancel())
        logger.debug("Clause chunking for request_id=%s", request_id)

    async def _feed_clause(self, chunk: str, done: bool, flush: bool) -> None:
        """feed_text for a clause request: hold the text and send what the rule cuts.

        done and flush both send everything held as one piece; done then ends the
        request with the sentinel (one audio_end), flush leaves it open.
        """
        chunker = self._clause_chunker
        chunker.feed(chunk)
        if done:
            self._done_received = True
            self._stop_clause_timer()
            for piece in chunker.drain():
                await self._sentence_queue.put(piece)
            await self._sentence_queue.put(None)
            return
        for piece in chunker.drain() if flush else chunker.poll():
            await self._sentence_queue.put(piece)
        self._wake_clause_timer()

    def _report_audio(self, request_id: str, n_samples: int, sample_rate: int) -> None:
        """Tell the clause chunker that a PCM chunk of the request was just sent.

        Every synthesis loop calls this after each chunk it sends, so the chunker's
        unplayed-audio estimate tracks what the client is playing. A no-op on a
        sentence request and for a request that is no longer the current one (a
        superseded task still unwinding).
        """
        chunker = self._clause_chunker
        if chunker is None or request_id != self._request_id or sample_rate <= 0:
            return
        chunker.record_audio(n_samples / sample_rate)
        if self._done_received:
            return  # done already sent everything held
        for piece in chunker.poll():
            self._sentence_queue.put_nowait(piece)
        self._wake_clause_timer()

    async def _clause_timer_loop(
        self, chunker: ClauseChunker, wake: asyncio.Event, queue: asyncio.Queue
    ) -> None:
        """Poll the chunker at its next deadline, so a time trigger is not missed.

        Sleeps until next_deadline() or until ``wake`` is set (a feed or an audio
        report changed the deadline), then asks again. Runs until cancelled with the
        request.
        """
        while True:
            wake.clear()
            deadline = chunker.next_deadline()
            if deadline is None:
                await wake.wait()  # nothing held: only a new feed can cut a piece
                continue
            delay = deadline - self._clause_clock()
            if delay > 0:
                try:
                    await asyncio.wait_for(wake.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                else:
                    continue  # re-armed: the deadline may have moved
            pieces = chunker.poll()
            for piece in pieces:
                logger.debug("Clause timer sent %d chars", len(piece))
                queue.put_nowait(piece)
            if not pieces:
                # The deadline sits within rounding of now; give the clock a moment
                # rather than spin.
                await asyncio.sleep(_CLAUSE_TIMER_MIN_S)

    def _wake_clause_timer(self) -> None:
        if self._clause_wake is not None:
            self._clause_wake.set()

    def _stop_clause_timer(self) -> None:
        if self._clause_timer is not None:
            self._clause_timer.cancel()
        self._clause_timer = None
        self._clause_wake = None

    def _stop_clause(self) -> None:
        """Drop the request's clause state: timer cancelled, chunker gone."""
        self._stop_clause_timer()
        self._clause_chunker = None

    async def _synthesis_loop(self, websocket, request_id: str) -> None:
        """Consume sentences from queue and synthesize sequentially.

        Sends audio_start before first sentence audio.
        Sends audio_end after all sentences are synthesized.
        """
        from voicemanager.protocol import AudioStartMessage, AudioEndMessage

        effective_tgt_se = self._tts_pipeline.target_se
        effective_sample_rate = 22050 if effective_tgt_se is not None else self._tts_pipeline._engine.sample_rate
        sentences_synthesized = 0

        try:
            while True:
                sentence = await self._sentence_queue.get()
                if sentence is None:
                    break  # No more sentences

                if not sentence.strip():
                    continue

                # Send audio_start before first sentence audio
                if not self._audio_started:
                    await websocket.send_text(
                        AudioStartMessage(
                            request_id=request_id,
                            sample_rate=effective_sample_rate,
                            channels=1,
                            bit_depth=16,
                            encoding="pcm",
                        ).model_dump_json()
                    )
                    self._audio_started = True

                if self._chunking == "clause":
                    # Makes a dropped or altered piece visible: the exact text the
                    # engine gets, once per piece, whichever path runs below.
                    logger.info(
                        "Clause TTS engine request: request_id=%s parts=1 text=%r",
                        request_id, sentence,
                    )

                # Use async stream_synthesize() for both engines
                loop = asyncio.get_event_loop()
                if effective_tgt_se is not None and hasattr(self._tts_pipeline._engine, 'synthesize_cloned'):
                    # OpenVoice with voice cloning
                    audio_np, sr = await loop.run_in_executor(
                        self._tts_pipeline._executor,
                        functools.partial(
                            self._tts_pipeline._engine.synthesize_cloned,
                            sentence,
                            effective_tgt_se,
                            self._tts_pipeline._speaker_key,
                            self._tts_pipeline._speed,
                        ),
                    )
                    pcm_bytes = float32_to_pcm16(audio_np)
                    await websocket.send_bytes(pcm_bytes)
                    self._report_audio(request_id, len(audio_np), sr)
                elif getattr(self._tts_pipeline._engine, "is_batch", False):
                    # Batch engine, no voice profile -- offload the blocking
                    # inference so the receive loop keeps reading while we speak.
                    audio_np, sr = await loop.run_in_executor(
                        self._tts_pipeline._executor,
                        functools.partial(
                            self._tts_pipeline._engine.synthesize,
                            sentence,
                            self._tts_pipeline._speaker_key,
                            self._tts_pipeline._speed,
                        ),
                    )
                    pcm_bytes = float32_to_pcm16(audio_np)
                    await websocket.send_bytes(pcm_bytes)
                    self._report_audio(request_id, len(audio_np), sr)
                else:
                    kwargs = {
                        "speaker_key": self._tts_pipeline._speaker_key,
                        "speed": self._tts_pipeline._speed,
                    }
                    effective_voice = self._voice or self._tts_pipeline._voice
                    if effective_voice is not None:
                        kwargs["voice"] = effective_voice

                    # Consume async stream_synthesize() directly -- chunks sent as they arrive
                    async for chunk_np in self._tts_pipeline._engine.stream_synthesize(sentence, **kwargs):
                        pcm_bytes = float32_to_pcm16(chunk_np)
                        await websocket.send_bytes(pcm_bytes)
                        self._report_audio(
                            request_id, len(chunk_np), self._tts_pipeline._engine.sample_rate)
                sentences_synthesized += 1
                logger.debug(
                    "Streamed TTS sentence: request_id=%s sentence=%s",
                    request_id, sentence[:30],
                )
        finally:
            # Send audio_end after all sentences synthesized
            if not self._audio_started:
                # No sentences were synthesized — still send audio_start + audio_end pair
                await websocket.send_text(
                    AudioStartMessage(
                        request_id=request_id,
                        sample_rate=effective_sample_rate,
                    ).model_dump_json()
                )
            try:
                await websocket.send_text(
                    AudioEndMessage(request_id=request_id).model_dump_json()
                )
            except Exception:
                logger.debug("Could not send audio_end for request_id=%s", request_id)
            logger.debug(
                "TTS stream complete: request_id=%s sentences=%d",
                request_id, sentences_synthesized,
            )

    async def cancel(self) -> None:
        """Cancel active streaming synthesis. Clear buffer and task.

        AWAITS the cancelled task, so it parks the caller until a blocked executor
        thread finishes. Fine when superseding a request on a path that can afford to
        wait; NOT usable on the cancel path — see abandon().
        """
        if self._synthesis_task is not None and not self._synthesis_task.done():
            self._synthesis_task.cancel()
            try:
                await self._synthesis_task
            except asyncio.CancelledError:
                pass
        self._stop_clause()
        self._text_buffer = ""
        self._request_id = None
        self._done_received = False
        self._audio_started = False
        self._synthesis_task = None

    def abandon(self) -> asyncio.Task | None:
        """Cancel the synthesis task WITHOUT awaiting it; drop the buffer.

        An executor thread cannot be interrupted, so awaiting the cancelled task parks
        the caller for the rest of the synthesis - an 858-1089ms stall before the cancel
        could be acked. Instead the task is cut loose and left to unwind on its own time,
        and the caller returns immediately to ack the cancel.

        Returns the abandoned task so the caller can keep watch on it (TTSSession does:
        the frames it still tries to send while unwinding must go nowhere).
        """
        task = self._synthesis_task
        if task is not None and not task.done():
            task.cancel()
            self._abandoned = [t for t in self._abandoned if not t.done()]
            self._abandoned.append(task)
        self._stop_clause()
        self._text_buffer = ""
        self._request_id = None
        self._done_received = False
        self._audio_started = False
        self._synthesis_task = None
        return task

    def dispose(self) -> None:
        """Clean up on disconnect. Cancel task, clear state."""
        if self._synthesis_task is not None and not self._synthesis_task.done():
            self._synthesis_task.cancel()
        self._stop_clause()
        self._text_buffer = ""
        self._request_id = None
        self._synthesis_task = None
