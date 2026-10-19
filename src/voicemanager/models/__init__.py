"""ML model lifecycle management for STT and TTS pipelines."""
import concurrent.futures
import logging
import threading

import torch
from silero_vad import load_silero_vad

from voicemanager.config import parse_option_list
from voicemanager.pipeline.tts import OpenVoiceTTSEngine

logger = logging.getLogger(__name__)

# Cache-aware bufferer/context-manager slots for the shared Nemotron pipeline. Each
# concurrent WebSocket connection consumes one slot for its lifetime (freed on the
# is_last flush frame). Must be >= the max number of simultaneous nemotron connections,
# or the bufferer raises RuntimeError("No free slots available").
_NEMOTRON_NUM_SLOTS = 8


def load_models(settings) -> dict:
    """Load STT models at startup. Fail fast if loading fails.

    Dispatches on settings.stt_engine to the matching loader. An unrecognized value
    fails fast with a ValueError rather than silently falling back to whisper - the
    lifespan turns that into a refusal to start. Each known engine keeps routing to its own
    loader, so the pre-existing engines are behaviourally unchanged.

    Args:
        settings: voicemanager.config.Settings instance

    Returns:
        dict — keys depend on engine: whisper path returns whisper/vad_model/executor/device;
        qwen/nemotron/audiocpp paths return their engine handle + stt_engine_type/device
    """
    engine = settings.stt_engine
    if engine == "whisper":
        return _load_whisper_models(settings)
    if engine == "qwen":
        return _load_qwen_models(settings)
    if engine == "nemotron":
        return _load_nemotron_models(settings)
    if engine == "audiocpp":
        return _load_audiocpp_models(settings)
    raise ValueError(
        f"Unknown VM_STT_ENGINE={engine!r}; valid: whisper, qwen, nemotron, audiocpp"
    )


def _load_whisper_models(settings) -> dict:
    """Load faster-whisper and Silero VAD at startup. Fail fast if either fails.

    Args:
        settings: voicemanager.config.Settings instance

    Returns:
        dict with keys: whisper, vad_model, executor, device
    """
    # Deferred: faster-whisper is in requirements-whisper.txt, not the core install.
    from faster_whisper import WhisperModel

    resolved_device = _resolve_device(settings.device)
    logger.info("Resolved device: %s", resolved_device)

    logger.info(
        "Loading faster-whisper model=%s device=%s compute_type=int8",
        settings.whisper_model,
        resolved_device,
    )
    fw_device = resolved_device
    fw_device_index = 0
    if resolved_device.startswith("cuda:"):
        fw_device = "cuda"
        fw_device_index = int(resolved_device.split(":")[1])
    whisper = WhisperModel(
        settings.whisper_model,
        device=fw_device,
        device_index=fw_device_index,
        compute_type="int8",
    )
    logger.info("faster-whisper loaded successfully")

    logger.info("Loading Silero VAD model")
    vad_model = load_silero_vad()
    logger.info("Silero VAD loaded successfully")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    logger.info("ThreadPoolExecutor created (max_workers=1)")

    return {
        "whisper": whisper,
        "vad_model": vad_model,
        "executor": executor,
        "device": resolved_device,
    }


def _load_qwen_models(settings) -> dict:
    """Load Qwen3-ASR model at startup. Fail fast if loading fails.

    Uses qwen-asr library wrapping vLLM. No ThreadPoolExecutor needed —
    QwenSTTPipeline uses asyncio.to_thread() per-call.
    No VAD model needed — Qwen3-ASR handles voice detection internally.

    Args:
        settings: voicemanager.config.Settings instance

    Returns:
        dict with keys: qwen_model, stt_engine_type, device
    """
    from qwen_asr import Qwen3ASRModel

    resolved_device = _resolve_device(settings.device)
    logger.info("Resolved device: %s", resolved_device)
    logger.info(
        "Loading Qwen3-ASR model=%s gpu_memory_utilization=%.2f",
        settings.qwen_model,
        settings.qwen_gpu_memory_utilization,
    )
    asr = Qwen3ASRModel.LLM(
        model=settings.qwen_model,
        gpu_memory_utilization=settings.qwen_gpu_memory_utilization,
        max_new_tokens=settings.qwen_max_new_tokens,
        max_model_len=settings.qwen_max_model_len,
    )
    logger.info("Qwen3-ASR model loaded successfully")
    return {
        "qwen_model": asr,
        "stt_engine_type": "qwen",
        "device": resolved_device,
    }


def _load_nemotron_models(settings) -> dict:
    """Load the shared cache-aware Nemotron streaming pipeline once.

    Builds a single CacheAwareRNNTPipeline via PipelineBuilder (one VRAM copy);
    per-connection streams share it via stream_id. Lazy NeMo import - only when
    VM_STT_ENGINE=nemotron. A threading.Lock is attached as `transcribe_lock`
    so per-connection feed() calls serialize transcribe_step, which mutates the shared
    cache/feature tensors.

    Args:
        settings: voicemanager.config.Settings instance

    Returns:
        dict with keys: nemotron_pipeline, stt_engine_type, device
    """
    import os
    # PyTorch 2.6+ defaults to weights_only=True which breaks NeMo checkpoints
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    resolved_device = _resolve_device(settings.device)
    logger.info("Resolved device: %s", resolved_device)

    # NeMo's from_pretrained loads weights via torch.load with map_location defaulting
    # to GPU 0; set the default CUDA device so bare "cuda" references target the right GPU.
    if resolved_device.startswith("cuda:"):
        gpu_idx = int(resolved_device.split(":")[1])
        torch.cuda.set_device(gpu_idx)
        logger.info("Set default CUDA device to %d for NeMo loading", gpu_idx)

    logger.info(
        "Loading Nemotron cache-aware pipeline model=%s device=%s num_slots=%d",
        settings.nemotron_model, resolved_device, _NEMOTRON_NUM_SLOTS,
    )

    # Lazy NeMo import. Build the single shared pipeline per the verified 26.06
    # recipe (scripts/spike_nemotron_2606.py); PipelineBuilder.build_pipeline loads the
    # model once and returns the cache-aware pipeline.
    from nemo.collections.asr.inference.factory.pipeline_builder import PipelineBuilder

    cfg = _build_nemotron_cfg(settings, resolved_device)
    nemotron_pipeline = PipelineBuilder.build_pipeline(cfg)

    # Serialize transcribe_step across concurrent connections (shared mutable tensors).
    nemotron_pipeline.transcribe_lock = threading.Lock()

    logger.info(
        "Nemotron cache-aware pipeline loaded (chunk_size_in_secs=%.3f, att_context_size=%s)",
        getattr(nemotron_pipeline, "chunk_size_in_secs", float("nan")),
        [settings.nemotron_att_context_left, settings.nemotron_att_context_right],
    )
    return {
        "nemotron_pipeline": nemotron_pipeline,
        "stt_engine_type": "nemotron",
        "device": resolved_device,
    }


def _build_nemotron_cfg(settings, device: str):
    """Assemble the cache-aware pipeline DictConfig (verified 26.06 recipe).

    Mirrors scripts/spike_nemotron_2606.py:build_cfg verbatim (which mirrors
    examples/asr/conf/asr_streaming_inference/cache_aware_rnnt.yaml), parameterized by
    VM_NEMOTRON_* settings. Key 26.06 schema notes: NO top-level enable_pnc;
    decoding.strategy="greedy_batch" with a structured greedy block; rewritten
    confidence block; top-level return_tail_result + lang; num_slots bumped for
    concurrency.
    """
    from omegaconf import OmegaConf

    if device.startswith("cuda"):
        asr_device = "cuda"
        device_id = int(device.split(":")[1]) if ":" in device else 0
    else:
        asr_device = "cpu"
        device_id = 0

    cfg = {
        "asr": {
            "model_name": settings.nemotron_model,
            "device": asr_device,
            "device_id": device_id,
            "compute_dtype": "bfloat16",
            "use_amp": False,
            "decoding": {
                "strategy": "greedy_batch",
                "preserve_alignments": False,
                "fused_batch_size": -1,
                "greedy": {
                    "use_cuda_graph_decoder": False,
                    # Disabled: we configure no contextual biasing (no key phrases,
                    # boosting_tree.model_path=None), so this only drags in the
                    # ngram_multi_advance Triton kernel — which has no compiled image
                    # for Blackwell/sm_120 (RTX 5060 Ti), crashing transcribe with
                    # "Triton Error [CUDA]: no kernel image is available". Off => the
                    # fusion-model advance path is skipped; transcripts are unchanged.
                    "enable_per_stream_biasing": False,
                    "preserve_frame_confidence": False,
                    "max_symbols": 10,
                    "ngram_lm_model": None,
                    "ngram_lm_alpha": 0.0,
                    "boosting_tree": {
                        "model_path": None,
                        "key_phrases_file": None,
                        "key_phrases_list": None,
                        "key_phrase_items_list": None,
                        "source_lang": "en",
                    },
                    "boosting_tree_alpha": 0.0,
                },
            },
        },
        "itn": {
            "input_case": "lower_cased",
            "whitelist": None,
            "overwrite_cache": False,
            "max_number_of_permutations_per_split": 729,
            "left_padding_size": 4,
            "batch_size": 32,
            "n_jobs": 16,
        },
        "confidence": {
            "exclude_blank": True,
            "aggregation": "mean",
            "method_cfg": {
                "name": "entropy",
                "entropy_type": "tsallis",
                "alpha": 0.5,
                "entropy_norm": "exp",
            },
        },
        "endpointing": {
            "stop_history_eou": settings.nemotron_eou_stop_history_ms,
            "residue_tokens_at_end": settings.nemotron_eou_residue_tokens,
        },
        "streaming": {
            "sample_rate": 16000,
            "batch_size": 1,
            "word_boundary_tolerance": 4,
            "att_context_size": [
                settings.nemotron_att_context_left,
                settings.nemotron_att_context_right,
            ],
            "use_cache": True,
            "use_feat_cache": True,
            "chunk_size_in_secs": None,  # derived from the model's chunk size
            "request_type": "frame",
            "num_slots": _NEMOTRON_NUM_SLOTS,
        },
        "matmul_precision": "high",
        "log_level": 30,
        "pipeline_type": "cache_aware",
        "asr_decoding_type": "rnnt",
        "audio_file": None,
        "output_filename": None,
        "output_dir": None,
        "enable_itn": False,
        "enable_nmt": False,
        "asr_output_granularity": "segment",
        "cache_dir": None,
        "lang": "en",
        "return_tail_result": False,
        "calculate_wer": False,
        "calculate_bleu": False,
        "warmup_steps": 0,
        "run_steps": 1,
    }
    return OmegaConf.create(cfg)


def _load_audiocpp_models(settings) -> dict:
    """Select the audio.cpp streaming STT backend at startup.

    Unlike the in-process engines, the audio.cpp runtime lives in a separate sidecar
    process reached over a Unix-domain socket (crash isolation). This loader
    does NOT spawn the sidecar or load the GGUF — the per-connection pipeline owns the
    sidecar client lifecycle. It only resolves the device (for logging/record)
    and gathers the VM_AUDIOCPP_* config the pipeline will need, returning the engine
    type so app routing + /ws/stream dispatch to the audio.cpp path.

    Args:
        settings: voicemanager.config.Settings instance

    Returns:
        dict with keys: stt_engine_type, audiocpp_config, device
    """
    # Deferred like the other engine imports: nothing in the audio.cpp client is loaded
    # unless this engine is the selected one.
    from voicemanager.pipeline.audiocpp_stt import SidecarSlots

    resolved_device = _resolve_device(settings.device)
    logger.info("Resolved device: %s", resolved_device)

    audiocpp_config = {
        "sidecar_bin": settings.audiocpp_sidecar_bin,
        "socket": settings.audiocpp_socket,
        "model": settings.audiocpp_model,
        "model_task": settings.audiocpp_model_task,
        "backend": settings.audiocpp_backend,
        "device": settings.audiocpp_device,
        "threads": settings.audiocpp_threads,
        "assets_root": settings.audiocpp_assets_root,
        # Parsed HERE so a malformed VM_AUDIOCPP_*_OPTIONS refuses to start the server
        # rather than surfacing per-session at the first spawn.
        "session_options": parse_option_list(settings.audiocpp_session_options),
        "request_options": parse_option_list(settings.audiocpp_request_options),
        # ONE limiter per server, shared by every connection's pipeline: that is what
        # makes the count process-wide without a module-level mutable global.
        "slots": SidecarSlots(settings.audiocpp_max_sidecars),
        # Silero VAD for the acoustic endpoint trigger: audio.cpp has no
        # native acoustic EoU, and a continuously-streaming mic client (HermesConsole)
        # never produces the frame GAP that the fallback trigger keys on -- its end-of-speech is
        # frames of quiet audio. A FACTORY, not a shared instance: the silero net keeps
        # its LSTM state inside the model object, so one instance fed by interleaved
        # sessions would cross-contaminate their speech probabilities -- each connection
        # loads its own (~25 ms, bundled with the package, no download).
        "vad_model_factory": load_silero_vad,
    }
    # Prove the bundled silero model loads NOW, not on the first connection: a broken
    # install should refuse to start, not hand every session a server_error.
    load_silero_vad()
    # A sidecar serves one session at a time and a fatal audio.cpp fault kills every
    # session on its process, so the default is one sidecar PER SESSION on a private
    # socket path. A configured VM_AUDIOCPP_SOCKET opts into a shared external
    # listener instead -- fine for measurement, but single-session: concurrent /ws/stream
    # consumers will get ERROR `busy`. Worth saying out loud at startup.
    if audiocpp_config["socket"]:
        logger.warning(
            "audio.cpp STT backend: VM_AUDIOCPP_SOCKET=%s set -- using that EXTERNAL "
            "sidecar for every session instead of spawning one per session. It serves "
            "one session at a time, so concurrent /ws/stream sessions will be refused "
            "(ERROR busy). Leave VM_AUDIOCPP_SOCKET blank for per-session sidecars.",
            audiocpp_config["socket"],
        )
    # Say the concurrency ceiling out loud at startup, so an operator sizing a deployment
    # reads it from the log rather than discovering it when the GPU runs out.
    max_sidecars = settings.audiocpp_max_sidecars
    if audiocpp_config["socket"]:
        pass                                  # external mode: not ours to spawn or count
    elif max_sidecars:
        logger.info(
            "audio.cpp STT backend: at most %d concurrent sessions (VM_AUDIOCPP_MAX_SIDECARS). "
            "Each runs its own sidecar process holding a CUDA context and a resident model "
            "on device %d (~1.5 GiB VRAM per session, measured); sessions over the cap are "
            "refused with a retriable at_capacity (WS close 4002), not an out-of-VRAM "
            "load_failed.", max_sidecars, audiocpp_config["device"],
        )
    else:
        logger.warning(
            "audio.cpp STT backend: VM_AUDIOCPP_MAX_SIDECARS=0 -- concurrent sessions are "
            "UNBOUNDED. Each spawns its own sidecar holding a CUDA context and a resident "
            "model on device %d (~1.5 GiB VRAM per session, measured), so the ceiling is "
            "now enforced by that device running out of VRAM, which surfaces to the client "
            "as a config-fault load_failed rather than a retriable capacity error.",
            audiocpp_config["device"],
        )
    logger.info(
        "audio.cpp STT backend selected (sidecar=%s, mode=%s, model_task=%s, backend=%s, device=%d)",
        audiocpp_config["sidecar_bin"],
        "external" if audiocpp_config["socket"] else "per-session",
        audiocpp_config["model_task"],
        audiocpp_config["backend"], audiocpp_config["device"],
    )
    return {
        "stt_engine_type": "audiocpp",
        "audiocpp_config": audiocpp_config,
        "device": resolved_device,
    }


def load_tts_models(settings) -> dict:
    """Load TTS engine at startup. Branches on settings.tts_engine.

    Dispatches on settings.tts_engine to the matching loader. An unrecognized value
    fails fast with a ValueError (mirroring load_models) rather than
    the old silent fall-through to openvoice - the lifespan turns that into a
    refusal to start. Each known engine keeps routing to its own loader, so the
    pre-existing engines are behaviourally unchanged.

    Args:
        settings: voicemanager.config.Settings instance

    Returns:
        dict with keys: tts_engine, tts_executor, tts_engine_type
    """
    engine = settings.tts_engine
    if engine == "openvoice":
        return _load_openvoice_tts(settings)
    if engine == "vibevoice":
        return _load_vibevoice_tts(settings)
    if engine == "audiocpp":
        return _load_audiocpp_tts(settings)
    raise ValueError(
        f"unknown VM_TTS_ENGINE {engine!r}: expected 'openvoice', 'vibevoice', "
        "or 'audiocpp'")


def _load_openvoice_tts(settings) -> dict:
    """Load OpenVoice TTS engine at startup. Fail fast if loading fails.

    Args:
        settings: voicemanager.config.Settings instance

    Returns:
        dict with keys: tts_engine, tts_executor, tts_engine_type
    """
    resolved_device = _resolve_device(settings.device)

    logger.info("Loading TTS engine (device=%s)", resolved_device)
    engine = OpenVoiceTTSEngine()
    engine.load(
        device=resolved_device,
        checkpoint_dir=settings.tts_checkpoint_dir,
        base_speaker_dir=settings.tts_base_speaker_dir,
        language=settings.tts_language,
    )
    logger.info("TTS engine loaded (sample_rate=%d)", engine.sample_rate)

    tts_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    logger.info("TTS ThreadPoolExecutor created (max_workers=1, separate from STT)")

    return {
        "tts_engine": engine,
        "tts_executor": tts_executor,
        "tts_engine_type": "openvoice",
    }


def _load_vibevoice_tts(settings) -> dict:
    """Load VibeVoice TTS engine at startup. Fail fast if loading fails.

    Lazy import prevents ImportError when openvoice engine is active.
    Mirrors _load_qwen_models() pattern.

    Args:
        settings: voicemanager.config.Settings instance

    Returns:
        dict with keys: tts_engine, tts_executor, tts_engine_type
    """
    from voicemanager.pipeline.vibevoice_tts import VibeVoiceTTSEngine

    resolved_device = _resolve_device(settings.device)

    logger.info(
        "Loading VibeVoice TTS engine (model=%s, device=%s, voice=%s)",
        settings.vibevoice_model, resolved_device, settings.vibevoice_voice,
    )
    engine = VibeVoiceTTSEngine()
    engine.load(
        device=resolved_device,
        model_name=settings.vibevoice_model,
        default_voice=settings.vibevoice_voice,
        cfg_scale=settings.vibevoice_cfg_scale,
        voices_dir=settings.vibevoice_voices_dir,
    )
    logger.info("VibeVoice TTS engine loaded (sample_rate=%d)", engine.sample_rate)

    tts_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    logger.info("TTS ThreadPoolExecutor created (max_workers=1, separate from STT)")

    return {
        "tts_engine": engine,
        "tts_executor": tts_executor,
        "tts_engine_type": "vibevoice",
    }


def _load_audiocpp_tts(settings) -> dict:
    """Load the audio.cpp dots_tts TTS engine at startup.

    No model loads in THIS process - the sidecar owns the GGUF and the GPU
    (VM_AUDIOCPP_DEVICE side), spawned per session. What startup does is
    validate the frozen preset voice assets fail-fast: a refs.json
    entry with a missing/empty ref_text or an absent WAV refuses to start.

    Lazy import mirrors the other engines: no audiocpp module cost unless selected.

    Args:
        settings: voicemanager.config.Settings instance

    Returns:
        dict with keys: tts_engine, tts_executor, tts_engine_type
    """
    from voicemanager.pipeline.audiocpp_tts import AudioCppTTSEngine

    logger.info(
        "Loading audiocpp TTS engine (model=%s, model_task=%s, voices=%s, voice=%s)",
        settings.audiocpp_tts_model, settings.audiocpp_tts_model_task,
        settings.audiocpp_tts_voices, settings.audiocpp_tts_voice,
    )
    engine = AudioCppTTSEngine({
        # The SAME sidecar binary as STT; TTS selects its task via HELLO.
        "sidecar_bin": settings.audiocpp_sidecar_bin,
        "model": settings.audiocpp_tts_model,
        "model_task": settings.audiocpp_tts_model_task,
        "backend": settings.audiocpp_backend,
        "device": settings.audiocpp_device,
        "threads": settings.audiocpp_threads,
        "voices": settings.audiocpp_tts_voices,
        "voice": settings.audiocpp_tts_voice,
        "seam_silence_ms": settings.audiocpp_tts_seam_silence_ms,
        "request_options": parse_option_list(settings.audiocpp_tts_request_options),
    })
    engine.load()  # fail-fast preset validation; no sidecar spawn here
    logger.info("audiocpp TTS engine loaded (sample_rate=%d)", engine.sample_rate)

    tts_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    logger.info("TTS ThreadPoolExecutor created (max_workers=1, separate from STT)")

    return {
        "tts_engine": engine,
        "tts_executor": tts_executor,
        "tts_engine_type": "audiocpp",
    }


def _resolve_device(device_setting: str) -> str:
    """Auto-detect or use explicit device setting.

    "auto" picks the CUDA GPU with the most free memory, or CPU if none available.
    Explicit values like "cpu", "cuda", "cuda:0", "cuda:1" are passed through.
    """
    if device_setting == "auto":
        if not torch.cuda.is_available():
            logger.info("Device auto-detect: cpu (no CUDA available)")
            return "cpu"
        gpu_count = torch.cuda.device_count()
        if gpu_count <= 1:
            logger.info("Device auto-detect: cuda (single GPU)")
            return "cuda"
        best_idx = 0
        best_free = 0
        for i in range(gpu_count):
            free, total = torch.cuda.mem_get_info(i)
            logger.info("GPU %d: %.1f GiB free / %.1f GiB total", i, free / 2**30, total / 2**30)
            if free > best_free:
                best_free = free
                best_idx = i
        resolved = f"cuda:{best_idx}"
        logger.info("Device auto-detect: %s (most free memory: %.1f GiB)", resolved, best_free / 2**30)
        return resolved
    return device_setting
