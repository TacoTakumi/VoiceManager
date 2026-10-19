"""FastAPI application factory."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from voicemanager.config import get_settings
from voicemanager.models import load_models, load_tts_models
from voicemanager.session import SessionRegistry
from voicemanager.transport.http import http_router
from voicemanager.transport.websocket import websocket_router
from voicemanager.web.account_routes import account_router
from voicemanager.web.render import STATIC_DIR
from voicemanager.web.session import SessionCookieMiddleware, SignInRequired
from voicemanager.web.signin_routes import sign_in_redirect, web_router
from voicemanager.web.token_routes import token_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown.

    Loads STT models (faster-whisper, Qwen3-ASR, or Nemotron cache-aware) and Silero VAD.
    Fails fast if model loading fails.
    Engine selected via VM_STT_ENGINE (whisper|qwen|nemotron).
    """
    settings = get_settings()
    logger.info(
        "VoiceManager starting (device=%s, stt_engine=%s, tts_engine=%s)",
        settings.device,
        settings.stt_engine,
        settings.tts_engine,
    )
    try:
        models = load_models(settings)
    except Exception as e:
        logger.error("Model loading failed — refusing to start: %s", e)
        raise

    # Store engine type for WebSocket handler routing
    stt_engine_type = models.get("stt_engine_type", "whisper")
    app.state.stt_engine_type = stt_engine_type
    app.state.device = models["device"]

    if stt_engine_type == "qwen":
        # Qwen path: store model, no VAD/executor needed
        app.state.qwen_model = models["qwen_model"]
        app.state.whisper_model = None
        app.state.vad_model = None
        app.state.executor = None
        app.state.nemotron_pipeline = None
        logger.info("Qwen3-ASR loaded on device=%s", models["device"])
    elif stt_engine_type == "nemotron":
        # Nemotron path: one shared cache-aware pipeline; per-connection streams
        app.state.nemotron_pipeline = models["nemotron_pipeline"]
        app.state.whisper_model = None
        app.state.vad_model = None
        app.state.executor = None
        app.state.qwen_model = None
        logger.info("Nemotron cache-aware pipeline loaded on device=%s", models["device"])
    elif stt_engine_type == "audiocpp":
        # audio.cpp path: the STT runtime lives in a separate sidecar process. No in-process
        # model handle here - the per-connection pipeline owns the
        # sidecar client; we only carry the resolved config for it to consume.
        app.state.audiocpp_config = models["audiocpp_config"]
        app.state.whisper_model = None
        app.state.vad_model = None
        app.state.executor = None
        app.state.qwen_model = None
        app.state.nemotron_pipeline = None
        logger.info("audio.cpp STT backend selected on device=%s", models["device"])
    else:
        # Whisper path: existing behavior
        app.state.whisper_model = models["whisper"]
        app.state.vad_model = models["vad_model"]
        app.state.executor = models["executor"]
        app.state.qwen_model = None
        app.state.nemotron_pipeline = None
        logger.info("faster-whisper loaded on device=%s", models["device"])

    try:
        tts_models = load_tts_models(settings)
    except Exception as e:
        logger.error("TTS model loading failed — refusing to start: %s", e)
        raise
    app.state.tts_engine = tts_models["tts_engine"]
    app.state.tts_executor = tts_models["tts_executor"]
    app.state.tts_engine_type = tts_models.get("tts_engine_type", "openvoice")  # NEW
    app.state.session_registry = SessionRegistry()
    logger.info(
        "TTS models loaded (engine=%s), session registry initialized",
        tts_models.get("tts_engine_type", "openvoice"),
    )
    yield
    # Shutdown
    tts_models["tts_executor"].shutdown(wait=True)
    if stt_engine_type == "whisper":
        models["executor"].shutdown(wait=True)
    logger.info("VoiceManager shutting down")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(title="VoiceManager", lifespan=lifespan)
    app.include_router(http_router)
    app.include_router(websocket_router)
    # The web UI shares the app and port with the API. A page asked
    # for without a session answers with a redirect to the sign-in page.
    app.include_router(web_router)
    app.include_router(token_router)
    app.include_router(account_router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.add_exception_handler(SignInRequired, sign_in_redirect)
    # Sets the session cookie again when a request moved the session's expiry.
    app.add_middleware(SessionCookieMiddleware)
    return app
