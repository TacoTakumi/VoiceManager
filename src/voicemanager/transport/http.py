"""HTTP endpoints for health checks, server info, and voice upload."""
import asyncio
import functools
import os
import tempfile
from fastapi import APIRouter, Depends, Request, UploadFile, File, HTTPException, Query
from voicemanager.auth.deps import require_token
from voicemanager.auth.store import Token
from voicemanager.config import get_settings

http_router = APIRouter(tags=["status"])


def _audiocpp_stt_model_name(config: dict) -> str | None:
    """Human-readable audiocpp model name for /info.

    The GGUF basename, not the full path — the filesystem layout stays
    server-side while the basename still names the weights + quant. Falls back to the model_task when no path is configured.
    """
    return os.path.basename(config.get("model", "")) or config.get("model_task")


def _tts_model_name(request: Request) -> str:
    """Return human-readable TTS model name based on active engine type."""
    tts_engine_type = getattr(request.app.state, "tts_engine_type", "openvoice")
    if tts_engine_type == "vibevoice":
        settings = get_settings()
        return f"vibevoice-realtime ({settings.vibevoice_model})"
    if tts_engine_type == "audiocpp":
        settings = get_settings()
        # Basename, not the full path (the /info convention for the STT sidecar):
        # falls back to the model_task id when no GGUF path is configured.
        name = os.path.basename(settings.audiocpp_tts_model) or settings.audiocpp_tts_model_task
        return f"audiocpp ({name})"
    return "openvoice-v2"


@http_router.get("/health")
async def health(request: Request):
    """Model readiness status.

    Reports actual model load state from app.state.
    Reflects active STT engine (whisper or qwen).
    """
    settings = get_settings()
    stt_engine_type = getattr(request.app.state, "stt_engine_type", settings.stt_engine)
    whisper = getattr(request.app.state, "whisper_model", None)
    qwen = getattr(request.app.state, "qwen_model", None)
    nemotron = getattr(request.app.state, "nemotron_pipeline", None)
    audiocpp = getattr(request.app.state, "audiocpp_config", None)
    vad = getattr(request.app.state, "vad_model", None)
    tts_engine = getattr(request.app.state, "tts_engine", None)
    return {
        "status": "ok",
        "models": {
            # audiocpp reports the model_task: the family label analog of
            # "nemotron-streaming"/"qwen3-asr" — which family is serving depends on
            # the loaded GGUF, and the task id names exactly that.
            "stt": "qwen3-asr" if stt_engine_type == "qwen" and qwen is not None
                   else "nemotron-streaming" if stt_engine_type == "nemotron" and nemotron is not None
                   else audiocpp.get("model_task") if stt_engine_type == "audiocpp" and audiocpp is not None
                   else "faster-whisper" if whisper is not None
                   else None,
            "vad": "silero-vad" if vad is not None else None,
            "tts": _tts_model_name(request) if tts_engine is not None else None,
        },
    }


@http_router.get("/info")
async def info(request: Request, token: Token = Depends(require_token)):
    """Loaded model names, device, and active sessions.

    Reflects active STT engine (whisper or qwen). Needs a bearer token (401 without).
    Lists only the sessions opened with the caller's tokens.
    """
    settings = get_settings()
    device = getattr(request.app.state, "device", settings.device)
    stt_engine_type = getattr(request.app.state, "stt_engine_type", settings.stt_engine)
    tts_engine_type = getattr(request.app.state, "tts_engine_type", "openvoice")
    whisper = getattr(request.app.state, "whisper_model", None)
    qwen = getattr(request.app.state, "qwen_model", None)
    nemotron = getattr(request.app.state, "nemotron_pipeline", None)
    audiocpp = getattr(request.app.state, "audiocpp_config", None)
    tts_engine = getattr(request.app.state, "tts_engine", None)

    # Session list for debugging, limited to the caller's own sessions
    registry = getattr(request.app.state, "session_registry", None)
    sessions = await registry.snapshot(owner=token.owner) if registry is not None else []

    return {
        "stt_engine": stt_engine_type,
        "tts_engine": tts_engine_type,
        "models": {
            "stt": settings.qwen_model if stt_engine_type == "qwen" and qwen is not None
                   else settings.nemotron_model if stt_engine_type == "nemotron" and nemotron is not None
                   else _audiocpp_stt_model_name(audiocpp) if stt_engine_type == "audiocpp" and audiocpp is not None
                   else settings.whisper_model if whisper is not None
                   else None,
            "tts": _tts_model_name(request) if tts_engine is not None else None,
        },
        "device": device,
        "whisper_model": settings.whisper_model if stt_engine_type == "whisper" else None,
        "qwen_model": settings.qwen_model if stt_engine_type == "qwen" else None,
        "nemotron_model": settings.nemotron_model if stt_engine_type == "nemotron" else None,
        # Sourced from app.state (the loaded truth), not settings, and gated on the
        # config actually being present — mirrors the per-engine field pattern while
        # keeping the unloaded state an honest null.
        "audiocpp_model": (
            os.path.basename(audiocpp.get("model", "")) or None
            if stt_engine_type == "audiocpp" and audiocpp is not None
            else None
        ),
        "audiocpp_model_task": (
            audiocpp.get("model_task")
            if stt_engine_type == "audiocpp" and audiocpp is not None
            else None
        ),
        "tts_model": (
            settings.vibevoice_model if tts_engine_type == "vibevoice"
            else (os.path.basename(settings.audiocpp_tts_model)
                  or settings.audiocpp_tts_model_task) if tts_engine_type == "audiocpp"
            else "openvoice-v2"
        ),
        "tts_sample_rate": tts_engine.sample_rate if tts_engine is not None else None,
        "available_voices": (
            tts_engine.available_voices
            if tts_engine is not None and hasattr(tts_engine, "available_voices")
            else []
        ),
        "sessions": sessions,
    }


@http_router.post("/voice/upload")
async def upload_voice(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Query(default=""),
    token: Token = Depends(require_token),
):
    """Upload voice sample for cloning. Extracts tone color embedding.

    Needs a bearer token (401 without). Routes to a specific session's
    TTSPipeline via session_id query parameter.
    Returns 404 if session_id is not found in the registry, or belongs to
    another user: that answer is the same as for an unknown id, so it does not
    reveal that the session exists.
    """
    # Voice cloning not supported with VibeVoice engine
    tts_engine_type = getattr(request.app.state, "tts_engine_type", "openvoice")
    if tts_engine_type == "vibevoice":
        raise HTTPException(
            status_code=400,
            detail="Voice cloning not supported with VibeVoice engine. Use VM_TTS_ENGINE=openvoice for voice cloning.",
        )
    # The audiocpp engine serves named presets only;
    # per-session upload-cloning is deferred. The OpenVoice path is untouched.
    if tts_engine_type == "audiocpp":
        raise HTTPException(
            status_code=400,
            detail="Voice upload not supported with the audiocpp engine (presets only: "
                   "select a voice from voices/designed/refs.json via the request's "
                   "voice field). Use VM_TTS_ENGINE=openvoice for upload cloning.",
        )
    # The voxtream engine likewise speaks named presets only.
    if tts_engine_type == "voxtream":
        raise HTTPException(
            status_code=400,
            detail="Voice upload not supported with the voxtream engine (presets only: "
                   "select a voice from the VM_VOXTREAM_VOICES presets via the request's "
                   "voice field). Use VM_TTS_ENGINE=openvoice for upload cloning.",
        )

    # Route voice upload to specific session via registry
    registry = getattr(request.app.state, "session_registry", None)
    if registry is None:
        raise HTTPException(status_code=503, detail="Session registry not available")

    entry = await registry.get(session_id)
    if entry is None or not entry.visible_to(token.owner):
        raise HTTPException(status_code=404, detail="Session not found")

    tts_engine = getattr(request.app.state, "tts_engine", None)
    if tts_engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not loaded")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"

    # se_extractor.get_se() requires a file path (pydub needs disk access for VAD splitting)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(content)

        loop = asyncio.get_event_loop()
        tts_executor = getattr(request.app.state, "tts_executor", None)

        from openvoice import se_extractor
        try:
            target_se, _ = await loop.run_in_executor(
                tts_executor,
                functools.partial(
                    se_extractor.get_se,
                    tmp_path,
                    tts_engine.tone_color_converter,
                    vad=True,
                ),
            )
        except AssertionError as exc:
            if "too short" in str(exc):
                raise HTTPException(
                    status_code=400,
                    detail="Voice sample too short — need at least a few seconds of clear speech",
                )
            raise
        # Set voice profile on session's TTSPipeline (overwrite on re-upload)
        if entry.tts_pipeline is not None:
            entry.tts_pipeline.target_se = target_se
        await registry.set_voice_profile(session_id, True)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return {"status": "voice profile updated"}
