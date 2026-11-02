"""HTTP endpoints for health checks, server info, and voice upload."""
import asyncio
import functools
import os
import tempfile
from fastapi import APIRouter, Depends, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
from starlette.requests import ClientDisconnect
from voicemanager.auth.deps import require_token
from voicemanager.auth.store import Token
from voicemanager.config import get_settings

http_router = APIRouter(tags=["status"])


def _audiocpp_stt_model_name(config: dict) -> str | None:
    """Human-readable audiocpp model name for /info.

    The GGUF basename, not the full path - the filesystem layout stays
    server-side while the basename still names the weights + quant. Falls back to the model_task when no path is configured.
    """
    return os.path.basename(config.get("model", "")) or config.get("model_task")


def _tts_model_id(request: Request) -> str:
    """The bare model id of the active TTS engine, as /info reports it.

    The vibevoice model id, the audiocpp GGUF basename (the filesystem layout
    stays server-side; falls back to the model_task id when no path is
    configured), the VoXtream model repository, else the OpenVoice checkpoint.
    """
    tts_engine_type = getattr(request.app.state, "tts_engine_type", "openvoice")
    if tts_engine_type == "vibevoice":
        return get_settings().vibevoice_model
    if tts_engine_type == "audiocpp":
        settings = get_settings()
        return os.path.basename(settings.audiocpp_tts_model) or settings.audiocpp_tts_model_task
    if tts_engine_type == "voxtream":
        return _voxtream_model_repo()
    return "openvoice-v2"


def _tts_model_name(request: Request) -> str:
    """Human-readable TTS model label for /health: the engine and its model id."""
    tts_engine_type = getattr(request.app.state, "tts_engine_type", "openvoice")
    model = _tts_model_id(request)
    if tts_engine_type == "vibevoice":
        return f"vibevoice-realtime ({model})"
    if tts_engine_type in ("audiocpp", "voxtream"):
        return f"{tts_engine_type} ({model})"
    return model


@functools.cache
def _voxtream_model_repo() -> str:
    """The Hugging Face repository the shipped VoXtream generator config loads."""
    import json
    from voicemanager.pipeline.voxtream_server import CONFIG_DIR
    return json.loads((CONFIG_DIR / "generator.json").read_text())["model_repo"]


@http_router.get("/health")
async def health(request: Request):
    """Model readiness status.

    Reports actual model load state from app.state.
    Reflects active STT engine (whisper or qwen).

    An engine with a ``ready`` flag (voxtream: its sidecar process) is reported
    as loaded only while the flag is true. While it is false the answer is 503
    with status "degraded" and that slot null, so a monitor that reads the
    status code and a reader of the body both see TTS is out, and the body
    keeps its shape.
    """
    settings = get_settings()
    stt_engine_type = getattr(request.app.state, "stt_engine_type", settings.stt_engine)
    whisper = getattr(request.app.state, "whisper_model", None)
    qwen = getattr(request.app.state, "qwen_model", None)
    nemotron = getattr(request.app.state, "nemotron_pipeline", None)
    audiocpp = getattr(request.app.state, "audiocpp_config", None)
    vad = getattr(request.app.state, "vad_model", None)
    tts_engine = getattr(request.app.state, "tts_engine", None)
    tts_ready = tts_engine is not None and bool(getattr(tts_engine, "ready", True))
    degraded = tts_engine is not None and not tts_ready
    body = {
        "status": "degraded" if degraded else "ok",
        "models": {
            # audiocpp reports the model_task: the family label analog of
            # "nemotron-streaming"/"qwen3-asr" - which family is serving depends on
            # the loaded GGUF, and the task id names exactly that.
            "stt": "qwen3-asr" if stt_engine_type == "qwen" and qwen is not None
                   else "nemotron-streaming" if stt_engine_type == "nemotron" and nemotron is not None
                   else audiocpp.get("model_task") if stt_engine_type == "audiocpp" and audiocpp is not None
                   else "faster-whisper" if whisper is not None
                   else None,
            "vad": "silero-vad" if vad is not None else None,
            "tts": _tts_model_name(request) if tts_ready else None,
        },
    }
    if degraded:
        return JSONResponse(body, status_code=503)
    return body


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
        # config actually being present - mirrors the per-engine field pattern while
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
        "tts_model": _tts_model_id(request),
        "tts_sample_rate": tts_engine.sample_rate if tts_engine is not None else None,
        "available_voices": (
            tts_engine.available_voices
            if tts_engine is not None and hasattr(tts_engine, "available_voices")
            else []
        ),
        "sessions": sessions,
    }


class _BodyTooLarge(Exception):
    """Raised from the capped receive channel once the body passes the limit."""


def _capped_receive(receive, limit: int):
    """Wrap an ASGI receive so the body bytes it hands out stop at ``limit``.

    The multipart parser pulls the body through this channel, so an upload sent
    without a Content-Length (chunked) is cut off within one chunk of the cap
    instead of spooling to disk until the client stops.
    """
    total = 0

    async def receive_capped():
        nonlocal total
        message = await receive()
        if message["type"] == "http.request":
            total += len(message.get("body", b""))
            if total > limit:
                raise _BodyTooLarge()
        return message

    return receive_capped


def _too_large(limit: int) -> HTTPException:
    """The 413 for an upload over the cap.

    Built at each raise site, never ahead of time: an exception held in the
    handler's locals is kept alive by its own traceback, and with it the parser
    frames and the spooled file of the refused upload, until the cyclic GC runs.
    """
    return HTTPException(
        status_code=413,
        detail=f"Upload larger than the {limit} byte limit (VM_UPLOAD_MAX_BYTES)",
    )


@http_router.post("/voice/upload")
async def upload_voice(
    request: Request,
    session_id: str = Query(default=""),
    token: Token = Depends(require_token),
):
    """Upload voice sample for cloning. Extracts tone color embedding.

    Needs a bearer token (401 without). Routes to a specific session's
    TTSPipeline via session_id query parameter.
    Returns 404 if session_id is not found in the registry, or belongs to
    another user: that answer is the same as for an unknown id, so it does not
    reveal that the session exists.

    The multipart body is not a parameter: FastAPI reads a declared file before
    it runs any dependency, so the whole upload would be spooled to disk before
    the 401, 400 or 404. The handler reads the form itself, after every check,
    and refuses a body over VM_UPLOAD_MAX_BYTES with 413: on the Content-Length
    before any read, or at the cap while a chunked body streams in.
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

    limit = get_settings().upload_max_bytes
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise _too_large(limit)
    capped = Request(request.scope, _capped_receive(request.receive, limit))
    try:
        async with capped.form() as form:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise HTTPException(status_code=400, detail="Missing file field")
            content = await upload.read()
            filename = upload.filename
    except _BodyTooLarge:
        raise _too_large(limit)
    except ClientDisconnect:
        # The client left mid-upload. Nobody reads this answer; it only keeps the
        # abort out of the server log, where an unhandled exception is a traceback.
        raise HTTPException(status_code=400, detail="Client disconnected during upload")
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    suffix = os.path.splitext(filename or "audio.wav")[1] or ".wav"

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
                    detail="Voice sample too short - need at least a few seconds of clear speech",
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
