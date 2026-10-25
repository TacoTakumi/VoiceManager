"""WebSocket transport handler with auth, handshake, audio validation, STT and TTS pipelines."""
import asyncio
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError
from voicemanager.auth.deps import websocket_token
from voicemanager.config import get_settings
from voicemanager.pipeline import AcousticEndpointer, STTPipeline
from voicemanager.pipeline.audiocpp_stt import SidecarError, SidecarFault
from voicemanager.pipeline.audiocpp_tts import AudioCppTTSStreamPipeline
from voicemanager.pipeline.qwen_stt import QwenSTTPipeline
from voicemanager.pipeline.tts import TTSPipeline, TTSStreamPipeline
from voicemanager.pipeline.tts_session import TTSSession
from voicemanager.protocol import (
    HandshakeMessage, ErrorMessage, TranscriptionMessage, AUDIO_FORMAT_SPEC,
    TTSRequestMessage, TTSCancelMessage, TTSStreamMessage, TurnStartedMessage,
    CHUNKING_MODES,
)
from voicemanager.session import SessionRegistry, SessionEntry
from voicemanager.transport.endpoint_hold import EndpointHoldMachine
from voicemanager.transport.turns import TurnTracker

logger = logging.getLogger(__name__)

websocket_router = APIRouter()


def _is_chunking_only_fault(exc: ValidationError) -> bool:
    """True when every validation error of a tts_stream frame is on its chunking field.

    Such a frame is a client asking for a mode the server does not offer: it gets an
    invalid_request error frame and is dropped, and the connection stays open. Any other
    fault (a missing request_id, a bad text) keeps the generic server_error path.
    """
    errors = exc.errors()
    return bool(errors) and all(tuple(e.get("loc", ())) == ("chunking",) for e in errors)


def _make_hold_machine(settings) -> EndpointHoldMachine | None:
    """Build the speculative endpoint-hold machine, or None when disabled.

    VM_STT_ENDPOINT_HOLD_MS=0 -> None -> the nemotron branch bypasses the hold
    machine entirely (immediate finals, no endpoint frames — pre-feature behavior).
    """
    if settings.stt_endpoint_hold_ms > 0:
        return EndpointHoldMachine(settings.stt_endpoint_hold_ms)
    return None


async def _drive_nemotron_stt(pipeline, hold_machine, data: bytes, send, *, session_id: str = "") -> None:
    """Drive one inbound nemotron audio frame; emit TranscriptionMessages via `send`.

    `send` is an async callable taking a TranscriptionMessage (websocket send in
    production, a list collector in tests) — the WS handler and the tests share this
    one code path so behavior can't drift.

    Two regimes:
      * hold_machine is None — speculation disabled (VM_STT_ENDPOINT_HOLD_MS=0):
        legacy immediate-final path, byte-for-byte pre-feature behavior.
      * hold_machine set — speculative endpoint-hold: on
        acoustic EoU emit a provisional endpoint frame (is_final=false,
        endpoint=true) and start an audio-clock hold; once a hold-window's worth of
        post-EoU audio has been fed, emit the immutable confirmed final.
    """
    # Disable path: EoU emits the confirmed final immediately, no endpoint frame.
    if hold_machine is None:
        has_update = await pipeline.feed(data)
        if has_update:
            text, language, seg_idx = pipeline.get_current_text()
            if text.strip():
                await send(TranscriptionMessage(
                    text=text, is_final=False, segment_index=seg_idx, language=language))
        while (final := pipeline.take_final()) is not None:
            f_text, f_lang, f_seg = final
            if f_text.strip():
                await send(TranscriptionMessage(
                    text=f_text, is_final=True, segment_index=f_seg, language=f_lang))
            pipeline.reset()
        return

    # Speculative hold path. Whether we were
    # already holding *before* this frame decides both whether this frame's audio
    # advances the hold clock (the EoU frame is the boundary, its audio is not counted)
    # and whether decoded speech in it reopens the held turn.
    was_holding = hold_machine.is_holding
    n_samples = len(data) // 2  # 16-bit PCM

    has_update = await pipeline.feed(data)
    interim_text = pipeline.get_current_text()[0] if has_update else ""

    # Collect this frame's EoU finals before emitting: a final arriving mid-hold is
    # itself decoded speech and reopens the turn (the resume check below handles it).
    finals: list[tuple[str, str, int]] = []
    while (final := pipeline.take_final()) is not None:
        finals.append(final)

    # Decoded speech (an interim partial or a fresh final) during the hold
    # reopens the held turn — cancel its pending confirm, fold the held text into a
    # stitch prefix, and remap the resumed NeMo segment onto the held segment_index.
    # The complement: once a hold has confirmed, was_holding is False here,
    # so a later turn never reopens and simply takes the next remapped segment_index.
    if was_holding and (interim_text.strip() or any(f[0].strip() for f in finals)):
        cancelled = hold_machine.reopen()
        was_holding = False  # hold cancelled; nothing to advance/confirm below
        logger.debug("Nemotron reopen (stitch) session=%s held=%s",
                     session_id, cancelled[0][:50] if cancelled else "")

    # Each acoustic EoU begins a hold and emits a provisional endpoint frame.
    # After a reopen the stitch prefix folds into the final text and the segment is
    # remapped, so the provisional (and its eventual confirm) carry the whole turn.
    new_eou = False
    for f_text, f_lang, f_seg in finals:
        if not f_text.strip():
            continue
        text = hold_machine.stitch(f_text)
        seg = hold_machine.remap(f_seg)
        prov = TranscriptionMessage(
            text=text, is_final=False, endpoint=True,
            segment_index=seg, language=f_lang)
        await send(prov)
        # The sink stamps `prov` with the turn open when this audio was received;
        # carry that turn with the held text so a confirm after a later turn_start keeps it.
        hold_machine.begin_hold(text, f_lang, seg, prov.turn_id)  # stored value is already stitched
        new_eou = True
        logger.debug("Nemotron endpoint (provisional) session=%s seg=%d %s",
                     session_id, seg, text[:50])

    # Audio fed after the EoU point advances the audio clock; once the
    # hold window elapses, confirm the immutable final. Skip on a frame that just
    # (re)started a hold or reopened — that frame's audio is the boundary, not the hold.
    if was_holding and not new_eou:
        confirmed = hold_machine.advance(n_samples)
        if confirmed is not None:
            c_text, c_lang, c_seg = confirmed
            final = TranscriptionMessage(
                text=c_text, is_final=True, endpoint=False,
                segment_index=c_seg, language=c_lang)
            # A final crossing a turn boundary keeps the turn its audio was
            # received in (carried by the hold), not the send-time turn.
            final.turn_id = hold_machine.last_final_turn
            await send(final)
            pipeline.reset()  # confirmed final is the commit boundary
            logger.debug("Nemotron confirmed final session=%s seg=%d %s",
                         session_id, c_seg, c_text[:50])

    # Interim partials (endpoint=false); silence yields none. After a reopen these carry
    # the stitched text under the remapped (held) segment_index.
    if interim_text.strip():
        _, language, seg_idx = pipeline.get_current_text()
        await send(TranscriptionMessage(
            text=hold_machine.stitch(interim_text), is_final=False, endpoint=False,
            segment_index=hold_machine.remap(seg_idx), language=language))


async def _teardown_nemotron_stt(pipeline, hold_machine, send, *, session_id: str = "") -> None:
    """Flush held/stitched nemotron text as the final confirmed frame at teardown.

    Called on every path that ends STT for a connection while a send may still be possible
    (the WS finally-block after client end/disconnect). Runs pipeline.finalize() to flush the
    NeMo residue, then emits ONE is_final=true frame that never silently drops held text:

      * hold pending — the held text is already stitched and at its emitted segment_index;
        merge any finalize tail into the same turn (one final per turn).
      * mid-stitch (reopened, awaiting the resumed EoU) — fold the stitch prefix in and remap
        the pipeline's tail so the held prefix is never lost.
      * hold disabled (hold_machine None) — legacy tail-only final at the pipeline segment.

    Idempotent by state: after emitting, the flush clears its sources —
    take_held()/finalize() gate their own state, and pipeline.reset() + clear_stitch()
    drop the live partial and mid-stitch prefix — so a second call emits nothing,
    regardless of whether the socket is still open. `send` is the same async sink used
    by _drive_nemotron_stt so behavior can't drift.
    """
    tail_text, lang, pipeline_seg = await pipeline.finalize()

    # Held/stitched text carries the turn its audio was received in; the disable
    # path leaves turn_id at the default so the sink stamps the receive-time turn.
    turn_id = 0
    if hold_machine is None:
        text, seg = tail_text, pipeline_seg
    else:
        held = hold_machine.take_held()
        if held is not None:
            h_text, lang, seg = held
            text = f"{h_text} {tail_text}".strip() if tail_text.strip() else h_text
            turn_id = hold_machine.last_final_turn  # the held turn crosses teardown intact
        else:
            # Mid-stitch or plain trailing partial: fold any prefix in and remap.
            turn_id = hold_machine.stitch_turn      # the stitched turn's original turn
            text = hold_machine.stitch(tail_text)
            seg = hold_machine.remap(pipeline_seg)

    if text.strip():
        msg = TranscriptionMessage(
            text=text, is_final=True, endpoint=False, segment_index=seg, language=lang)
        msg.turn_id = turn_id
        await send(msg)
        logger.debug("Nemotron teardown final session=%s seg=%d %s", session_id, seg, text[:50])

    # A teardown flush is a commit boundary: clear the sources of the emitted
    # text so a second call is inert by state, not merely by a closed socket.
    # pipeline.reset() clears the live partial; clear_stitch() drops the mid-stitch
    # prefix. finalize() already gated its own residue (self._closed). Keeps
    # "at most one final per segment_index" true even if teardown runs twice.
    pipeline.reset()
    if hold_machine is not None:
        hold_machine.clear_stitch()


async def _finalize_audiocpp_utterance(pipeline, send, *, session_id: str = "") -> None:
    """FLUSH one audio.cpp utterance mid-session and emit its frames.

    The shipping Nemotron-3.5 model streams growing interim deltas during AUDIO (surfaced
    live by the audio branch of the handler) and decodes only the utterance tail at FLUSH.
    pipeline.finalize() runs the FLUSH (reading any tail deltas into the live partial and
    returning the final), then this emits the last partial as one is_final=False frame and
    the final as the single is_final=True.

    `send` is the handler's shared stamping sink (_emit), so these frames carry the turn
    their audio was received in like every other engine's. reset() then advances
    to the next (monotonic, never-rewound) segment_index so the following utterance is
    distinct; feed() re-arms finalize()'s pending flag, so the next utterance flushes too.
    """
    f_text, f_lang, f_seg = await pipeline.finalize()
    # get_current_text() now holds the ACCUMULATED transcript-so-far: the deltas that
    # arrived during AUDIO plus the tail deltas from the flush
    # (finalize() folds each EVENT delta into the live partial). Emit it as
    # the interim before the committed final; both carry the pre-reset segment_index, so
    # the partial and its final line up. The reset() below is the commit boundary that
    # clears the accumulation.
    p_text, p_lang, p_seg = pipeline.get_current_text()
    if p_text.strip():
        await send(TranscriptionMessage(
            text=p_text, is_final=False, segment_index=p_seg, language=p_lang))
    if f_text.strip():
        await send(TranscriptionMessage(
            text=f_text, is_final=True, segment_index=f_seg, language=f_lang))
        logger.debug("audio.cpp mid-session final session=%s seg=%d %s",
                     session_id, f_seg, f_text[:50])
    pipeline.reset()  # advance the segment; feed() re-arms the pending flush for the next


def _connection_default_voice(engine, token) -> str | None:
    """The voice a TTS request without 'voice' gets on this connection.

    The token's voice when the engine has it, else the engine's default. An
    engine without presets (OpenVoice: no available_voices) gives None and the
    token voice is ignored. A token voice the engine lacks (an engine switch, a
    removed preset) logs one warning naming the token and the voice - never
    the secret. Called once per connection, so the warning is once too.
    """
    if engine is None or not hasattr(engine, "available_voices"):
        return None
    default = getattr(engine, "default_voice", None)
    voice = getattr(token, "voice", None)
    if not voice:
        return default
    if voice in engine.available_voices:
        return voice
    logger.warning(
        "Token %r of %r has voice %r, which the TTS engine does not have; "
        "using the default voice %r",
        token.name, token.owner, voice, default,
    )
    return default


def _is_valid_audio(data: bytes) -> bool:
    """Validate audio frame format.

    Accepts:
    - WAV container (starts with RIFF...WAVE magic bytes)
    - Raw 16-bit PCM (even byte count = 2-byte sample alignment)

    Rejects:
    - Odd byte count (incomplete 16-bit sample)
    - Empty frames
    """
    if len(data) == 0:
        return False
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return True
    return len(data) % 2 == 0


@websocket_router.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket handler for full-duplex audio streaming (STT + TTS).

    Protocol flow:
    1. Auth check before accept — the handshake request must carry
       'Authorization: Bearer <token>'; without a valid token it is rejected during
       the handshake, which clients see as an HTTP 403. The token is kept
       on websocket.state.token for the rest of the connection
    2. Accept and send handshake — session_id + audio_format
    3. Create STTPipeline and TTSPipeline per connection
    4. Register session in registry
    5. Unified receive() loop with inactivity timeout: binary frames -> STT, text frames -> TTS routing
    6. First binary frame validated — error frame + close 1003 on mismatch
    7. TTS dispatched through the connection's TTSSession, which owns the single
       in-flight request and runs it as an asyncio.Task for full-duplex
    8. TTS cancellation is a method on that session, not logic inlined here — the
       tts_cancel branch is a thin delegation
    9. TTSPipeline disposed and session removed in finally block
    """
    settings = get_settings()

    # Reject before accept without a valid bearer token. websocket_token
    # closes the socket with 1008 before accept(), so no WebSocket is ever
    # established and this code never reaches the client — the ASGI server rejects
    # the handshake with HTTP 403 instead. A token in the query string is
    # ignored; the store lookup runs in the thread pool.
    token = await websocket_token(websocket)
    if token is None:
        return
    websocket.state.token = token

    await websocket.accept()

    # Send handshake with session_id and audio_format
    session_id = str(uuid.uuid4())
    logger.info("WebSocket connected: session_id=%s", session_id)

    stt_engine_type = getattr(websocket.app.state, "stt_engine_type", "whisper")
    tts_engine_type = getattr(websocket.app.state, "tts_engine_type", "openvoice")
    tts_engine_obj = getattr(websocket.app.state, "tts_engine", None)
    available_voices = (
        tts_engine_obj.available_voices
        if hasattr(tts_engine_obj, "available_voices")
        else []
    )
    # Resolved once per connection: the token's voice, else the engine default.
    # Requests without 'voice' get it through the TTSPipeline below.
    default_voice = _connection_default_voice(tts_engine_obj, token)

    await websocket.send_text(
        HandshakeMessage(
            session_id=session_id,
            audio_format=AUDIO_FORMAT_SPEC,
            stt_engine=stt_engine_type,
            tts_engine=tts_engine_type,
            available_voices=available_voices,
            default_voice=default_voice,
        ).model_dump_json()
    )

    # Create per-connection STT pipeline based on active engine
    hold_machine: EndpointHoldMachine | None = None  # nemotron speculative endpoint-hold
    if stt_engine_type == "qwen":
        qwen_model = getattr(websocket.app.state, "qwen_model", None)
        pipeline = QwenSTTPipeline(asr_model=qwen_model)
    elif stt_engine_type == "nemotron":
        from voicemanager.pipeline.nemotron_stt import NemotronSTTPipeline
        nemotron_pipeline = getattr(websocket.app.state, "nemotron_pipeline", None)
        stream_id = uuid.uuid4().int & 0xFFFFFFFF  # per-connection NeMo stream id
        pipeline = NemotronSTTPipeline(
            shared_pipeline=nemotron_pipeline,
            stream_id=stream_id,
            att_context_size=[settings.nemotron_att_context_left,
                              settings.nemotron_att_context_right],
        )
        # hold_ms>0 enables speculative endpoint-hold; 0 disables it entirely.
        hold_machine = _make_hold_machine(settings)
    elif stt_engine_type == "audiocpp":
        # audio.cpp path: the STT runtime is a separate sidecar process
        # reached over a Unix socket (crash isolation). The pipeline implements the
        # same per-connection streaming contract as nemotron (feed/get_current_text/
        # take_final/finalize/reset/dispose), so it drives through the SAME generic path
        # below with hold_machine left None: interim deltas -> is_final=False and finals ->
        # is_final=True via take_final()/finalize(). The speculative endpoint-hold
        # is a nemotron-only feature and stays off here. Lazy import keeps the module (and
        # its sidecar client) untouched unless this engine is active (mirrors nemotron).
        from voicemanager.pipeline.audiocpp_stt import AudioCppSTTPipeline
        audiocpp_config = getattr(websocket.app.state, "audiocpp_config", None)
        pipeline = AudioCppSTTPipeline(config=audiocpp_config)
        # Acoustic end-of-utterance from frame CONTENT. A continuously-
        # streaming mic client (HermesConsole) never stops sending, so the frame-gap
        # trigger below never fires for it -- its silence is frames of quiet audio. Each
        # connection gets its OWN silero instance (the net keeps LSTM state inside the
        # model object; sharing one would cross-contaminate concurrent sessions), built
        # off-loop. No factory configured -> inert, and the frame gap is the only trigger.
        vad_factory = (audiocpp_config or {}).get("vad_model_factory")
        audiocpp_endpointer = AcousticEndpointer(
            vad_model=await asyncio.to_thread(vad_factory) if vad_factory else None,
            vad_threshold=settings.audiocpp_vad_threshold,
            silence_ms=settings.audiocpp_silence_ms,
        )
        # Idle-feed gate. Inter-utterance silence frames are not
        # fed to the (4B) decoder; a rolling pre-roll is replayed at speech onset so
        # the attack silero needed a chunk to confirm is not clipped. Armed only when
        # the endpointer can actually classify -- with no VAD every frame classifies
        # as non-speech and gating would starve the session of ALL audio.
        audiocpp_idle_gate = settings.audiocpp_idle_gate and audiocpp_endpointer.armed
        audiocpp_preroll_max = max(0, settings.audiocpp_onset_preroll_ms) * 32  # bytes
        audiocpp_preroll = bytearray()
    else:
        # Whisper path: existing behavior
        # Use getattr for safe access — app.state attributes absent when TestClient skips lifespan
        vad_model = getattr(websocket.app.state, "vad_model", None)
        whisper_model = getattr(websocket.app.state, "whisper_model", None)
        executor = getattr(websocket.app.state, "executor", None)
        pipeline = STTPipeline(
            vad_model=vad_model,
            whisper_model=whisper_model,
            executor=executor,
            vad_threshold=settings.vad_threshold,
            silence_ms=settings.vad_silence_ms,
        )

    # Server-owned turn counter for this connection. Opening a turn is
    # a method call (turn_tracker.advance, wired to the turn_start frame), so
    # the counter can also be advanced from the STT path by a future server-side
    # barge-in. Frames are stamped with the turn open when their audio was received.
    #
    # advance() also clears the STT pipeline's uncommitted live partial, so
    # pre-turn_start words cannot stitch into the new turn's first frame (turn_id fences
    # whole FRAMES but not text INSIDE one). The clear applies to whichever engines hold a
    # live partial: nemotron (raw partial + mid-stitch prefix) and qwen (its accumulating
    # streaming state). It never drops a pending endpoint-hold final (is_holding) -- that
    # resolves later, still tagged with the old turn. Whisper emits only VAD-gated
    # finals and keeps no partial text to stitch, so its clear_partial stays None and
    # advance() is a counter-only bump; an engine with no live partial has nothing to clear.
    clear_partial = None
    if stt_engine_type == "nemotron":
        def clear_partial() -> None:
            # Drop the raw live partial (_last_partial) and any mid-stitch reopen prefix
            # -- both uncommitted. A pending hold's _held/_pending_finals are untouched, so
            # the held/endpointed final still emits and is never dropped.
            pipeline.reset()
            if hold_machine is not None:
                hold_machine.clear_stitch()
    elif stt_engine_type in ("qwen", "audiocpp"):
        def clear_partial() -> None:
            # Drop the streaming engine's uncommitted partial. reset() reinitialises the
            # streaming state and advances to the next (monotonic, never rewound)
            # segment_index. Applies to qwen and audio.cpp — both keep a live
            # get_current_text() partial and no speculative hold to preserve.
            pipeline.reset()
    turn_tracker = TurnTracker(clear_partial=clear_partial)

    # One TTSPipeline per connection (when TTS engine available)
    tts_engine = getattr(websocket.app.state, "tts_engine", None)
    tts_executor = getattr(websocket.app.state, "tts_executor", None)

    # audiocpp: one TTS sidecar process per WS session on a private
    # socket path, mirroring the STT ownership pattern. The startup engine only
    # validated config + presets; THIS session gets its own engine (own sidecar
    # client), warmed in the background so the handshake is not stalled by the
    # model load and the first tts request finds the session already warm. A
    # warm-up failure is logged, not fatal: each request retries connect() itself
    # and surfaces the defined error.
    session_tts_engine = None
    tts_warm_task: asyncio.Task | None = None
    if tts_engine is not None and tts_engine_type == "audiocpp":
        session_tts_engine = tts_engine.session_engine()
        tts_engine = session_tts_engine
        tts_warm_task = asyncio.create_task(asyncio.to_thread(session_tts_engine.connect))

        def _log_warm_failure(task: asyncio.Task, sid: str = session_id) -> None:
            if not task.cancelled() and task.exception() is not None:
                logger.warning("TTS sidecar warm-up failed for session %s: %s",
                               sid, task.exception())
        tts_warm_task.add_done_callback(_log_warm_failure)

        # The defined error surface for a TTS sidecar death. The
        # synthesis runs in a detached task this receive loop never awaits, so a
        # mid-request SidecarFault cannot reach the except-clauses below - the
        # engine awaits this callback from the task's own context instead. Same
        # defined shape as the STT fault path (an EXPECTED, CONTAINED condition:
        # one error frame + a 1011 close, a clean warning, no bug traceback), and
        # scoped to THIS session only - the sidecar is per-session, so a death's
        # blast radius is one connection. The close also wakes the pending
        # receive(), so the loop exits with no hang.
        async def _surface_tts_fault(exc: BaseException) -> None:
            logger.warning("audio.cpp TTS sidecar fault, closing session %s: %s",
                           session_id, exc)
            try:
                await websocket.send_text(
                    ErrorMessage(
                        error_type="tts_backend_fault",
                        message="TTS backend faulted; this session was closed.",
                    ).model_dump_json()
                )
                await websocket.close(code=1011)
            except Exception:
                pass

        session_tts_engine.on_fault = _surface_tts_fault

    tts_pipeline = None
    if tts_engine is not None and tts_executor is not None:
        tts_pipeline = TTSPipeline(
            engine=tts_engine,
            executor=tts_executor,
            language=settings.tts_language,
            speaker_key=settings.tts_speaker_key,
            speed=settings.tts_speed,
            # A request's own 'voice' wins; without one, both the tts and the
            # tts_stream paths fall back to this connection default.
            voice=default_voice,
        )

    # One TTSStreamPipeline per connection for streaming TTS input.
    # audiocpp gets a batching variant: first sentence
    # dispatches immediately, busy-period text coalesces into one sidecar
    # request. Every other engine keeps the base pipeline, byte-identical.
    tts_stream_pipeline = None
    if tts_pipeline is not None:
        stream_kwargs = {}
        if tts_engine_type == "audiocpp":
            stream_pipeline_cls = AudioCppTTSStreamPipeline
            # Only the audiocpp variant takes the coalescing window, so the base
            # pipeline's call stays byte-identical for every other engine.
            stream_kwargs["coalesce_ms"] = settings.audiocpp_tts_coalesce_ms
        else:
            stream_pipeline_cls = TTSStreamPipeline
        tts_stream_pipeline = stream_pipeline_cls(
            tts_pipeline=tts_pipeline,
            language=settings.tts_language,
            # Thresholds for chunking="clause" requests; sentence requests ignore them.
            clause_first_words=settings.tts_clause_first_words,
            clause_first_deadline_ms=settings.tts_clause_first_deadline_ms,
            clause_later_words=settings.tts_clause_later_words,
            clause_margin_ms=settings.tts_clause_margin_ms,
            **stream_kwargs,
        )

    # Register session in registry
    registry: SessionRegistry | None = getattr(websocket.app.state, "session_registry", None)
    if registry is not None:
        entry = SessionEntry(
            session_id=session_id,
            connected_at=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc),
            has_voice_profile=False,
            tts_pipeline=tts_pipeline,
            owner=websocket.state.token.owner,
        )
        await registry.register(entry)

    # The connection's single in-flight TTS request lives on the session,
    # not in this loop. Both the `tts` and `tts_stream` branches dispatch through it, so
    # at most one synthesis task can ever be live here — and cancellation is a method any
    # caller can reach (a future server-side barge-in fires from the STT path).
    tts_session: TTSSession | None = None
    if tts_pipeline is not None:
        tts_session = TTSSession(
            websocket=websocket,
            tts_pipeline=tts_pipeline,
            tts_stream_pipeline=tts_stream_pipeline,
        )

    # Track last streaming STT text update for silence-triggered finalization (qwen)
    streaming_last_update: float | None = None
    # Track the last inbound audio frame for the audio.cpp mid-session endpoint trigger.
    # Unlike qwen (which finalizes when the interim TEXT stops changing), the
    # audio.cpp model has no native end-of-utterance signal, so end-of-utterance is a GAP in
    # inbound AUDIO: this holds the monotonic time of the last audio frame while an
    # utterance is un-flushed, and is None when there is nothing pending to finalize.
    audiocpp_last_audio: float | None = None

    async def _emit(msg: TranscriptionMessage) -> None:
        """Send one transcription frame, stamped with its turn (shared cross-engine sink).

        The turn_id is the turn open when this utterance's audio was received
        (turn_tracker.received), applied here at the sink so every frame carries it without
        each construction site repeating the stamp. Every STT engine emits through this sink
        -- nemotron (_drive/_teardown), qwen (interim/final) and whisper (_send_transcription)
        -- so no engine can emit an un-stamped frame.

        One exception: a held final (is_final under the endpoint-hold) already carries
        the turn its audio was received in — set from the hold, which may predate the
        current turn when it confirms across a turn_start. Do NOT overwrite that with the
        send-time turn; the id on the frame is the race-free fence the client reads. Only the
        nemotron hold path has held finals; whisper/qwen (hold_machine None) stamp normally.
        """
        if not (hold_machine is not None and msg.is_final):
            turn_tracker.stamp(msg)
        await websocket.send_text(msg.model_dump_json())

    try:
        loop = asyncio.get_event_loop()
        first_frame_validated = False

        # Inactivity timeout
        last_activity = time.monotonic()
        warning_sent = False
        timeout = settings.session_timeout

        while True:
            elapsed = time.monotonic() - last_activity
            remaining = timeout - elapsed

            if remaining <= 0:
                # Timeout expired — close connection.
                # Flush any held/stitched nemotron text as the confirmed
                # final BEFORE closing. The finally-block teardown runs after close()
                # (its send would fail on a closed socket), and the mid-stitch state
                # (reopened, is_holding=False) has no wall-clock fallback — so without
                # this the stitched turn is silently dropped at the inactivity timeout.
                # _teardown_nemotron_stt is idempotent, so the finally call is then a no-op.
                if stt_engine_type in ("nemotron", "audiocpp"):
                    try:
                        await _teardown_nemotron_stt(
                            pipeline, hold_machine, _emit, session_id=session_id)
                    except Exception:
                        logger.debug(
                            "Could not flush streaming final before timeout close (session %s)",
                            session_id,
                        )
                try:
                    await websocket.send_text(
                        ErrorMessage(
                            error_type="timeout",
                            message="Session closed due to inactivity.",
                        ).model_dump_json()
                    )
                    await websocket.close(code=1001)
                except Exception:
                    pass
                break

            WARNING_LEAD = 30
            if remaining <= WARNING_LEAD:
                wait_for = remaining
            else:
                wait_for = remaining - WARNING_LEAD

            # Use flush timeout when streaming TTS is active
            effective_timeout = wait_for
            if tts_session is not None and tts_session.active_stream_request_id is not None:
                effective_timeout = min(wait_for, settings.tts_stream_flush_timeout)
            # While a speculative endpoint is held, cap the receive timeout to
            # the hold window so a client that stops sending audio still gets its confirmed
            # final within a bounded wall-clock time (via this same receive-timeout path),
            # never only at session close. The audio clock still drives confirmation
            # while audio keeps arriving; this is the silent-client backstop.
            if hold_machine is not None and hold_machine.is_holding:
                effective_timeout = min(effective_timeout, settings.stt_endpoint_hold_ms / 1000.0)
            # While an audio.cpp utterance is un-flushed, cap the receive timeout to the
            # silence window so a gap in inbound audio wakes the loop at the endpoint threshold
            # (below) instead of only at session close — the mid-session FLUSH that surfaces
            # per-utterance partials + a final. Bounded like the nemotron hold's silent-client
            # backstop; while audio keeps arriving each frame returns before this elapses.
            if stt_engine_type == "audiocpp" and audiocpp_last_audio is not None:
                effective_timeout = min(effective_timeout, settings.audiocpp_silence_ms / 1000.0)

            try:
                msg = await asyncio.wait_for(websocket.receive(), timeout=effective_timeout)
            except asyncio.TimeoutError:
                # Flush streaming TTS buffer on timeout
                if tts_session is not None and tts_session.active_stream_request_id is not None:
                    await tts_session.flush_timeout()
                    continue
                # Wall-clock fallback - flush the held (possibly stitched) text as
                # the immutable confirmed final, then keep the session open. take_held()
                # clears the hold, so this never double-emits with the audio-clock confirm
                # (at most one is_final per segment_index).
                if hold_machine is not None:
                    held = hold_machine.take_held()
                    if held is not None:
                        h_text, h_lang, h_seg = held
                        fallback = TranscriptionMessage(
                            text=h_text, is_final=True, endpoint=False,
                            segment_index=h_seg, language=h_lang)
                        # Keep the turn the held audio was received in, not the
                        # send-time turn (which may have advanced past a turn_start).
                        fallback.turn_id = hold_machine.last_final_turn
                        await _emit(fallback)
                        pipeline.reset()  # confirmed final is the commit boundary
                        logger.debug("Nemotron wall-clock fallback final session=%s seg=%d %s",
                                     session_id, h_seg, h_text[:50])
                        continue
                # audio.cpp mid-session endpoint trigger. A gap of at least
                # audiocpp_silence_ms with an utterance still un-flushed is our end-of-
                # utterance signal (audio.cpp has no native acoustic EoU). FLUSH the
                # utterance, surface at most one is_final=False partial (the whole
                # accumulated transcript, none if it is empty) + exactly one is_final=True
                # final DURING the session, then reset for the next utterance and keep the
                # connection OPEN. The elapsed re-check guards against a timeout that fired
                # for a shorter reason.
                if stt_engine_type == "audiocpp" and audiocpp_last_audio is not None:
                    silence_ms = settings.audiocpp_silence_ms
                    if time.monotonic() - audiocpp_last_audio >= silence_ms / 1000.0:
                        await _finalize_audiocpp_utterance(pipeline, _emit, session_id=session_id)
                        audiocpp_last_audio = None
                        continue
                if remaining > WARNING_LEAD and not warning_sent:
                    # A receive timeout stopped implying
                    # "the client has been quiet for the whole pre-warning leg" once the
                    # audio.cpp silence cap (and the TTS-flush/hold caps) made sub-second
                    # wakes routine. A capped wake the branches above declined used to
                    # fall through here and warn a HEALTHY session ~1 s in -- and latch
                    # warning_sent, suppressing the real warning later. Warn only when
                    # the client's ACTUAL quiet time has consumed the pre-warning leg;
                    # a declined capped wake falls through silently and just re-arms.
                    if timeout - (time.monotonic() - last_activity) <= WARNING_LEAD:
                        # Send timeout warning ~30s before close
                        await websocket.send_text(
                            ErrorMessage(
                                error_type="timeout_warning",
                                message=f"Session will close in {WARNING_LEAD}s due to inactivity. Send any message to reset.",
                            ).model_dump_json()
                        )
                        warning_sent = True
                continue

            # Reset activity tracking on any received message
            last_activity = time.monotonic()
            warning_sent = False

            # Update registry activity timestamp
            if registry is not None:
                await registry.update_activity(session_id)

            # Handle disconnect (ASGI sends disconnect message)
            if msg.get("type") == "websocket.disconnect":
                break

            if msg.get("bytes") is not None:
                data = msg["bytes"]

                # Validate first binary frame
                if not first_frame_validated:
                    if not _is_valid_audio(data):
                        await websocket.send_text(
                            ErrorMessage(
                                error_type="invalid_audio_format",
                                message="Expected 16kHz mono 16-bit PCM or WAV. Got unrecognized format.",
                            ).model_dump_json()
                        )
                        await websocket.close(code=1003)
                        logger.warning("Invalid audio format from session %s, disconnected", session_id)
                        return
                    first_frame_validated = True
                    logger.debug("First audio frame validated for session %s (%d bytes)", session_id, len(data))

                # Snapshot the open turn as this audio's turn BEFORE transcribing
                # it, so frames it produces are tagged with the turn open at receive time
                # (not the turn that may open while its inference is still in flight).
                turn_tracker.mark_received()

                # STT path — route to correct pipeline
                if stt_engine_type == "nemotron":
                    # Generic streaming drive path over the feed/get_current_text/take_final
                    # contract: acoustic-EoU driven, with a speculative endpoint-hold when
                    # enabled (hold_machine None -> legacy immediate finals).
                    await _drive_nemotron_stt(
                        pipeline, hold_machine, data, _emit, session_id=session_id,
                    )
                elif stt_engine_type == "audiocpp":
                    # audio.cpp shares the feed/get_current_text/take_final
                    # contract but has NO native acoustic EoU (its final commits only at
                    # FLUSH) - so unlike nemotron it can't self-endpoint.
                    # The silence trigger in the receive-timeout path above FLUSHes the
                    # un-flushed utterance per utterance.
                    #
                    # Classify this frame's CONTENT FIRST --
                    # both the endpoint decision and the idle-gate feed decision key on it.
                    # acoustic_endpoint True = speech just ended acoustically, the endpoint
                    # signal a continuously-streaming client actually produces (the
                    # frame-gap trigger below only ever fires when transmission stops).
                    #
                    # Off-loop: this is real silero inference on EVERY 32 ms
                    # chunk, so on the loop it serializes against every other session's I/O
                    # and the cost multiplies with concurrency. AWAITED here, not fired off:
                    # each frame's endpoint decision must be applied before the next frame
                    # is read, or the trailing-silence trigger would race the audio it is
                    # classifying. (The whisper engine's own on-loop VAD is untouched -- the
                    # off-loop rule binds only this engine, and the blast radius stays additive.)
                    was_speaking = audiocpp_endpointer.speech_active
                    acoustic_endpoint = await asyncio.to_thread(audiocpp_endpointer.feed, data)
                    # In-utterance = inside silero's start..end span before or after this
                    # frame, or carrying the end event itself (its tail belongs to the
                    # utterance). Everything else is inter-utterance audio.
                    in_utterance = (was_speaking or audiocpp_endpointer.speech_active
                                    or acoustic_endpoint)
                    if audiocpp_idle_gate and not in_utterance:
                        # Inter-utterance silence is NOT billed to the
                        # decoder -- not fed, and (fed nothing) it cannot arm the frame-gap
                        # trigger either, so no empty inter-utterance flush ever consumes a
                        # segment index. Keep the tail in the rolling onset pre-roll.
                        audiocpp_preroll.extend(data)
                        if len(audiocpp_preroll) > audiocpp_preroll_max:
                            del audiocpp_preroll[:len(audiocpp_preroll) - audiocpp_preroll_max]
                        continue
                    if audiocpp_idle_gate and not was_speaking and audiocpp_preroll:
                        # Speech onset: replay the gated tail ahead of this frame so the
                        # attack silero took a chunk to confirm is not clipped.
                        await pipeline.feed(bytes(audiocpp_preroll))
                        audiocpp_preroll.clear()
                    # Feed the chunk off-loop and mark the utterance un-flushed.
                    has_update = await pipeline.feed(data)
                    audiocpp_last_audio = time.monotonic()
                    # Interim deltas the model volunteers mid-AUDIO (generic forwarding)
                    # surface live here; the shipping Nemotron-3.5 streams its growing
                    # partials through this branch while the speaker talks.
                    if has_update:
                        text, language, seg_idx = pipeline.get_current_text()
                        if text.strip():
                            await _emit(TranscriptionMessage(
                                text=text, is_final=False, segment_index=seg_idx, language=language))
                    # A volunteered mid-stream FINAL commits immediately and ends
                    # the utterance, exactly like nemotron's legacy path.
                    committed = False
                    while (final := pipeline.take_final()) is not None:
                        committed = True
                        f_text, f_lang, f_seg = final
                        if f_text.strip():
                            await _emit(TranscriptionMessage(
                                text=f_text, is_final=True, segment_index=f_seg, language=f_lang))
                        pipeline.reset()
                        audiocpp_last_audio = None  # committed; nothing pending until next audio
                    # Acoustic FLUSH: same finalize + emit + reset as the frame-gap
                    # path, fired inline because frames are still arriving (the receive
                    # timeout that hosts the gap trigger never elapses for this client).
                    # Skipped when a volunteered FINAL just committed this utterance.
                    if acoustic_endpoint and not committed:
                        await _finalize_audiocpp_utterance(pipeline, _emit, session_id=session_id)
                        audiocpp_last_audio = None
                elif stt_engine_type == "qwen":
                    # Streaming path: qwen interim result protocol
                    has_update = await pipeline.feed(data)
                    if has_update:
                        streaming_last_update = time.monotonic()
                        text, language, seg_idx = pipeline.get_current_text()
                        if text.strip():  # Suppress empty partials
                            msg_out = TranscriptionMessage(
                                text=text,
                                is_final=False,
                                segment_index=seg_idx,
                                language=language,
                            )
                            await _emit(msg_out)  # stamp turn_id via the shared sink
                            logger.debug(
                                "Sent interim transcription for session %s: %s",
                                session_id, text[:50],
                            )
                    elif streaming_last_update is not None:
                        # Silence detection: no text change -- check threshold
                        silence_ms = settings.qwen_silence_ms
                        silence_elapsed = time.monotonic() - streaming_last_update
                        if silence_elapsed >= silence_ms / 1000.0:
                            text, language, seg_idx = await pipeline.finalize()
                            if text.strip():
                                final_msg = TranscriptionMessage(
                                    text=text,
                                    is_final=True,
                                    segment_index=seg_idx,
                                    language=language,
                                )
                                await _emit(final_msg)  # stamp turn_id via the shared sink
                                logger.debug(
                                    "Sent silence-triggered final for session %s: %s",
                                    session_id, text[:50],
                                )
                            pipeline.reset()
                            streaming_last_update = None
                else:
                    # Whisper path: VAD-gated finals, stamped via the shared sink.
                    if pipeline.feed(data):
                        await _send_transcription(pipeline, loop, _emit, session_id)

            elif msg.get("text") is not None:
                try:
                    parsed = json.loads(msg["text"])
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from session %s", session_id)
                    continue

                msg_type = parsed.get("type")

                if msg_type == "tts" and tts_session is not None:
                    tts_req = TTSRequestMessage.model_validate(parsed)
                    logger.info(
                        "TTS request: session=%s request_id=%s text=%s voice=%s",
                        session_id, tts_req.request_id, tts_req.text[:50],
                        tts_req.voice,
                    )
                    # The session dispatches the synthesis task and supersedes
                    # whatever was in flight — one TTS request per connection.
                    await tts_session.speak(
                        text=tts_req.text,
                        request_id=tts_req.request_id,
                        voice=tts_req.voice,  # voice parameter routing
                    )

                elif msg_type == "tts_stream" and tts_session is not None:
                    try:
                        tts_stream_msg = TTSStreamMessage.model_validate(parsed)
                    except ValidationError as exc:
                        if not _is_chunking_only_fault(exc):
                            raise
                        # An unknown chunking mode: tell the client and drop the frame.
                        # Nothing is opened or disturbed, and STT keeps running.
                        logger.warning(
                            "Dropped tts_stream with bad chunking=%r: session=%s",
                            parsed.get("chunking"), session_id,
                        )
                        await websocket.send_text(
                            ErrorMessage(
                                error_type="invalid_request",
                                message=(
                                    "tts_stream chunking must be one of: "
                                    + ", ".join(CHUNKING_MODES)
                                    + "; the frame was dropped."
                                ),
                            ).model_dump_json()
                        )
                        continue
                    logger.info(
                        "TTS stream chunk: session=%s request_id=%s done=%s flush=%s len=%d "
                        "voice=%s chunking=%s",
                        session_id, tts_stream_msg.request_id, tts_stream_msg.done,
                        tts_stream_msg.flush, len(tts_stream_msg.text), tts_stream_msg.voice,
                        tts_stream_msg.chunking,
                    )
                    await tts_session.feed_stream(
                        chunk=tts_stream_msg.text,
                        done=tts_stream_msg.done,
                        request_id=tts_stream_msg.request_id,
                        voice=tts_stream_msg.voice,  # voice parameter routing
                        flush=tts_stream_msg.flush,  # speak the buffer now
                        chunking=tts_stream_msg.chunking,  # fixed at request start
                    )

                elif msg_type == "tts_cancel" and tts_session is not None:
                    cancel_req = TTSCancelMessage.model_validate(parsed)
                    logger.info(
                        "TTS cancel: session=%s request_id=%s",
                        session_id, cancel_req.request_id,
                    )
                    # A thin delegation. The cancellation logic lives on the
                    # session so the STT path can reach it too.
                    await tts_session.cancel(cancel_req.request_id)

                elif msg_type == "turn_start":
                    # A thin delegation to the server-owned turn counter.
                    # Opening a turn -- bumping the counter and clearing the STT live
                    # partial -- lives entirely in turn_tracker.advance(), so a future
                    # server-side barge-in can open a turn from the STT path by calling
                    # the same method. The handler only relays the new id back.
                    # Unguarded by TTS: turns are an STT/conversation concept.
                    new_turn_id = turn_tracker.advance()
                    logger.info("Turn opened: session=%s turn_id=%s", session_id, new_turn_id)
                    await websocket.send_text(
                        TurnStartedMessage(turn_id=new_turn_id).model_dump_json()
                    )

    except WebSocketDisconnect:
        logger.info("Client disconnected: session_id=%s", session_id)
    except SidecarFault as exc:
        # A process-fatal fault in the audio.cpp sidecar (induced GGML_ASSERT,
        # SIGKILL, or a bare socket EOF mid-stream) surfaces here as a SidecarFault. It is
        # an EXPECTED, CONTAINED condition, not a VoiceManager bug: the runtime is isolated
        # in its own process, so the fault crashes neither this server nor any concurrent
        # session (each WS connection is its own asyncio task). Give just this session a
        # DEFINED error frame + close code (distinct from the generic server_error path)
        # and log a clean warning -- no bug traceback, because the crash isn't ours.
        logger.warning("audio.cpp sidecar fault, closing session %s: %s", session_id, exc)
        try:
            await websocket.send_text(
                ErrorMessage(
                    error_type="stt_backend_fault",
                    message="Speech backend faulted; this session was closed.",
                ).model_dump_json()
            )
            await websocket.close(code=1011)
        except Exception:
            pass
    except SidecarError as exc:
        # The audio.cpp sidecar refused to serve this session. That is not a VoiceManager
        # bug, so it gets its own defined frame + close code instead of the catch-all's
        # generic server_error and a full bug traceback.
        #
        # The code is chosen from exc.code, NOT from the exception class, because
        # `SidecarError` spans two classes that ask the client for opposite things. An
        # earlier version collapsed them onto 4001 and so actively told clients to stop
        # retrying a fault that would have cleared on the next attempt:
        #
        #   * PERMANENT (`load_failed` for a missing/bad GGUF, `unknown_model_task`,
        #     `unsupported_protocol`, `dial_failed`, `spawn_failed`, `no_sidecar`) --
        #     4001 stt_backend_unavailable: this server cannot serve this engine AT ALL
        #     until an operator fixes VM_AUDIOCPP_*, so retrying is futile. Logged at
        #     ERROR: a human has to act.
        #   * TRANSIENT (`busy` from a sidecar that already has a live session, and any
        #     code not on the permanent allow-list) -- 4002 stt_backend_unready: the
        #     backend could not take this session RIGHT NOW; retry with backoff. Logged at
        #     WARNING, because nothing is misconfigured.
        #
        # Both sit in the RFC 6455 private 4000-4999 range -- unmistakably
        # application-defined, never confusable with a transport-level failure -- and both
        # are distinct from the SidecarFault path above (1011/stt_backend_fault: the
        # backend WAS serving and broke mid-session).
        if exc.is_permanent:
            error_type, close_code = "stt_backend_unavailable", 4001
            message = f"Speech backend unavailable ({exc.code}); this session was closed."
            logger.error(
                "audio.cpp sidecar unavailable, closing session %s: %s", session_id, exc)
        else:
            error_type, close_code = "stt_backend_unready", 4002
            message = (f"Speech backend not ready ({exc.code}); this session was closed, "
                       "retry shortly.")
            logger.warning(
                "audio.cpp sidecar not ready (transient), closing session %s: %s",
                session_id, exc)
        try:
            await websocket.send_text(
                ErrorMessage(error_type=error_type, message=message).model_dump_json()
            )
            await websocket.close(code=close_code)
        except Exception:
            pass
    except Exception:
        logger.exception("Unexpected error in WebSocket handler for session %s", session_id)
        try:
            await websocket.send_text(
                ErrorMessage(
                    error_type="server_error",
                    message="Internal server error.",
                ).model_dump_json()
            )
            await websocket.close(code=1011)
        except Exception:
            pass
    finally:
        # Stop whatever TTS is in flight before general cleanup
        if tts_session is not None:
            await tts_session.aclose()
        # Finalize streaming state if active (flush buffered audio)
        if stt_engine_type in ("nemotron", "audiocpp"):
            # Flush the trailing final before the socket closes so text is never
            # silently dropped. nemotron merges held/stitched text with the finalize tail;
            # audio.cpp (hold_machine None) flushes its finalize() tail as the one final.
            try:
                await _teardown_nemotron_stt(pipeline, hold_machine, _emit, session_id=session_id)
            except Exception:
                logger.debug(
                    "Could not send streaming teardown final for session %s (connection may be closed)",
                    session_id,
                )
        elif stt_engine_type == "qwen":
            try:
                text, language, seg_idx = await pipeline.finalize()
                if text.strip():
                    final_msg = TranscriptionMessage(
                        text=text,
                        is_final=True,
                        segment_index=seg_idx,
                        language=language,
                    )
                    await _emit(final_msg)  # stamp turn_id via the shared sink
                    logger.debug(
                        "Sent final transcription for session %s: %s",
                        session_id, text[:50],
                    )
            except Exception:
                logger.debug(
                    "Could not send final transcription for session %s (connection may be closed)",
                    session_id,
                )
        pipeline.dispose()  # STTPipeline/QwenSTTPipeline cleanup
        if tts_pipeline is not None:
            tts_pipeline.dispose()  # release target_se tensor
        if tts_stream_pipeline is not None:
            tts_stream_pipeline.dispose()
        # audiocpp: reap THIS session's TTS sidecar. dispose() is
        # non-blocking (signal + daemon-thread reap, the STT close() pattern) and is
        # deliberately called WITHOUT await: an awaited teardown can be cancelled out
        # from under the reap when this endpoint task is torn down on disconnect.
        # The client's lifecycle lock makes it safe against a warm-up connect still
        # in flight.
        if session_tts_engine is not None:
            try:
                session_tts_engine.dispose()
            except Exception:
                logger.exception("TTS sidecar teardown failed for session %s", session_id)
        # Remove from session registry
        if registry is not None:
            await registry.remove(session_id)
        logger.debug("Pipelines disposed and session removed for session %s", session_id)


async def _send_transcription(
    pipeline: STTPipeline,
    loop: asyncio.AbstractEventLoop,
    send,
    session_id: str,
) -> None:
    """Run inference and emit one TranscriptionMessage per segment.

    Each segment from faster-whisper becomes a separate transcription frame. `send` is the
    handler's shared stamping sink (_emit), so whisper's finals carry the receive-time
    turn_id like every other engine's frames — this function never touches the
    socket directly.
    """
    segments, info, duration = await pipeline.run_inference(loop)
    for idx, seg in enumerate(segments):
        confidence = math.exp(seg.avg_logprob) if seg.avg_logprob else 0.0
        msg = TranscriptionMessage(
            text=seg.text.strip(),
            is_final=True,
            segment_index=idx,
            language=info.language,
            inference_duration=round(duration, 4),
            confidence=round(confidence, 4),
            start=round(seg.start, 3),
            end=round(seg.end, 3),
        )
        await send(msg)
        logger.debug(
            "Sent transcription segment %d for session %s: %s",
            idx, session_id, seg.text.strip()[:50],
        )
