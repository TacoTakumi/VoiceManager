"""VibeVoice-Realtime-0.5B TTS engine implementation.

All vibevoice imports are lazy to avoid ImportError when this engine
is not active.
"""
import asyncio
import copy
import logging
from collections.abc import AsyncIterator

import numpy as np
import torch

from voicemanager.pipeline.tts import TTSEngine

logger = logging.getLogger(__name__)

# Available voice presets (verified from VibeVoice package)
VIBEVOICE_VOICES = ["Carter", "Davis", "Emma", "Frank", "Grace", "Mike"]


def _dtype_and_attn(device: str) -> tuple[torch.dtype, str]:
    """Pick (torch_dtype, attn_implementation) for a device string.

    Any CUDA device — bare ``"cuda"`` or an explicit ``"cuda:N"`` — uses
    bfloat16 + flash_attention_2. CPU (or anything non-CUDA) uses float32 +
    sdpa. The ``startswith`` check is deliberate: a plain ``== "cuda"`` would
    send ``cuda:N`` into the CPU branch and silently load in fp32.
    """
    if device.startswith("cuda"):
        return torch.bfloat16, "flash_attention_2"
    return torch.float32, "sdpa"


class VibeVoiceTTSEngine(TTSEngine):
    """VibeVoice-Realtime-0.5B streaming TTS engine.

    Uses AsyncAudioStreamer for streaming generation and VoiceMapper
    for preset voice selection. All vibevoice imports are lazy.
    """

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._voice_cache: dict[str, dict] = {}  # voice_name -> all_prefilled_outputs
        self._sample_rate_val = 24000  # VibeVoice native output rate
        self._default_voice = "Carter"
        self._default_cfg_scale = 1.5

    def load(
        self,
        device: str,
        model_name: str = "microsoft/VibeVoice-Realtime-0.5B",
        default_voice: str = "Carter",
        cfg_scale: float = 1.5,
        voices_dir: str = "voices/streaming_model",
    ) -> None:
        """Load VibeVoice model, processor, and cache voice presets.

        Attempts flash_attention_2 on CUDA first, falls back to sdpa.

        Args:
            device: "cpu", "cuda", or an explicit "cuda:N"
            model_name: HuggingFace model ID or local path
            default_voice: Default voice preset name
            cfg_scale: Classifier-free guidance scale
            voices_dir: Path to directory containing voice preset .pt files
        """
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_streaming_processor import (
            VibeVoiceStreamingProcessor,
        )

        self._device = device
        self._default_voice = default_voice
        self._default_cfg_scale = cfg_scale
        self._voices_dir = voices_dir

        # Device-specific dtype and attention (flash_attention_2 fallback).
        # Handles "cuda:N" as well as bare "cuda" (see _dtype_and_attn).
        load_dtype, attn_impl = _dtype_and_attn(device)

        logger.info("Loading VibeVoice processor from %s", model_name)
        self._processor = VibeVoiceStreamingProcessor.from_pretrained(model_name)

        logger.info(
            "Loading VibeVoice model from %s (dtype=%s, attn=%s, device=%s)",
            model_name, load_dtype, attn_impl, device,
        )
        try:
            self._model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                model_name,
                torch_dtype=load_dtype,
                device_map=device,
                attn_implementation=attn_impl,
            )
        except (ImportError, ValueError) as exc:
            if device.startswith("cuda") and attn_impl == "flash_attention_2":
                logger.warning(
                    "flash_attention_2 unavailable (%s), falling back to sdpa "
                    "(may affect audio quality per VibeVoice docs)",
                    exc,
                )
                self._model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    model_name,
                    torch_dtype=load_dtype,
                    device_map=device,
                    attn_implementation="sdpa",
                )
            else:
                raise

        # Cache voice presets
        self._load_voice_presets()

        logger.info(
            "VibeVoiceTTSEngine loaded: device=%s, voices=%s, default=%s",
            device, list(self._voice_cache.keys()), self._default_voice,
        )

    def _load_voice_presets(self) -> None:
        """Scan voices_dir for .pt files and cache them.

        Voice preset files use the naming convention from the VibeVoice repo:
        ``en-Carter_man.pt``, ``en-Emma_woman.pt``, etc. The short name
        (e.g. "Carter") is extracted from the filename for use as the voice
        selection key.
        """
        import pathlib

        voices_path = pathlib.Path(self._voices_dir)
        if not voices_path.exists():
            raise RuntimeError(
                f"VibeVoice voices directory not found: {voices_path.resolve()}\n"
                "Run: python setup_models.py vibevoice"
            )

        pt_files = sorted(voices_path.glob("*.pt"))
        if not pt_files:
            raise RuntimeError(
                f"No .pt voice preset files in {voices_path.resolve()}\n"
                "Run: python setup_models.py vibevoice"
            )

        for pt_file in pt_files:
            # Extract short name: "en-Carter_man.pt" -> "Carter"
            stem = pt_file.stem  # "en-Carter_man"
            parts = stem.split("-", 1)
            if len(parts) == 2:
                # "Carter_man" -> "Carter"
                short_name = parts[1].rsplit("_", 1)[0]
            else:
                short_name = stem

            try:
                all_prefilled = torch.load(pt_file, map_location=self._device, weights_only=False)
                self._voice_cache[short_name] = all_prefilled
                logger.debug("Cached voice preset: %s from %s", short_name, pt_file)
            except Exception as exc:
                logger.warning("Failed to load voice preset %s: %s", short_name, exc)

        if not self._voice_cache:
            raise RuntimeError(
                f"No voice presets loaded from {voices_path.resolve()}"
            )

    @property
    def sample_rate(self) -> int:
        """24kHz native output rate."""
        return self._sample_rate_val

    @property
    def available_voices(self) -> list[str]:
        """List of loaded voice preset names."""
        return sorted(self._voice_cache.keys())

    @property
    def default_voice(self) -> str:
        """The preset a request that names no voice gets (the configured voice)."""
        return self._default_voice

    def synthesize(self, text: str, speaker_key: str = "EN-US", speed: float = 1.0) -> tuple[np.ndarray, int]:
        """Synthesize entire text at once (collects all streaming chunks).

        For VibeVoice, this runs stream_synthesize() synchronously via asyncio.
        The speaker_key parameter is ignored -- use voice kwarg instead.
        """
        # Use a synchronous AudioStreamer for the non-async path
        chunks = []

        from vibevoice.modular.streamer import AudioStreamer
        import threading

        voice = self._default_voice
        cfg_scale = self._default_cfg_scale

        all_prefilled = copy.deepcopy(self._voice_cache[voice])
        inputs = self._processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=all_prefilled,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}

        streamer = AudioStreamer(batch_size=1)

        def _generate():
            self._model.generate(
                **inputs,
                cfg_scale=cfg_scale,
                tokenizer=self._processor.tokenizer,
                generation_config={"do_sample": False},
                audio_streamer=streamer,
                all_prefilled_outputs=copy.deepcopy(all_prefilled),
            )

        gen_thread = threading.Thread(target=_generate, daemon=True)
        gen_thread.start()

        for chunk_tensor in streamer.get_stream(sample_idx=0):
            audio_np = chunk_tensor.detach().cpu().float().numpy()
            chunks.append(audio_np)

        gen_thread.join(timeout=30)

        if not chunks:
            return np.array([], dtype=np.float32), self._sample_rate_val
        return np.concatenate(chunks), self._sample_rate_val

    async def stream_synthesize(self, text: str, **kwargs) -> AsyncIterator[np.ndarray]:
        """Yield float32 numpy audio chunks at 24kHz from VibeVoice streaming generation.

        Uses AsyncAudioStreamer (asyncio.Queue-based) to bridge from the background
        generation thread to the event loop. Audio chunks are yielded as they arrive
        from model.generate() -- delivering VibeVoice's ~300ms first-audio latency.

        The generation thread runs model.generate() which pushes chunks to the
        AsyncAudioStreamer via call_soon_threadsafe(). The async for loop in this
        method consumes them from the asyncio.Queue on the event loop.

        AsyncAudioStreamer requires loop parameter -- must be current event loop.
        all_prefilled_outputs must be deepcopied before each generate() call.
        Short text (<= 3 words) may produce garbled audio -- log warning but proceed.

        Args:
            text: Text to synthesize
            **kwargs:
                voice: str -- voice preset name (default: self._default_voice)
                cfg_scale: float -- classifier-free guidance scale (default: self._default_cfg_scale)
        """
        from vibevoice.modular.streamer import AsyncAudioStreamer
        import threading

        voice = kwargs.get("voice", self._default_voice)
        cfg_scale = kwargs.get("cfg_scale", self._default_cfg_scale)

        # Resolve voice preset
        if voice not in self._voice_cache:
            logger.warning(
                "Voice '%s' not found, falling back to '%s'",
                voice, self._default_voice,
            )
            voice = self._default_voice

        # Warn on very short text
        word_count = len(text.strip().split())
        if word_count <= 3:
            logger.warning(
                "VibeVoice short text warning: %d words ('%s') -- may produce unstable audio",
                word_count, text[:50],
            )

        # Deepcopy voice preset before generate() (mutates in-place)
        all_prefilled = copy.deepcopy(self._voice_cache[voice])

        # Prepare model inputs
        inputs = self._processor.process_input_with_cached_prompt(
            text=text,
            cached_prompt=all_prefilled,
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # AsyncAudioStreamer gets the running loop internally
        streamer = AsyncAudioStreamer(batch_size=1)

        # Run generate in a background thread
        # threading.Thread rather than executor avoids queue contention during streaming
        def _generate():
            self._model.generate(
                **inputs,
                cfg_scale=cfg_scale,
                tokenizer=self._processor.tokenizer,
                generation_config={"do_sample": False},
                audio_streamer=streamer,
                all_prefilled_outputs=copy.deepcopy(all_prefilled),
            )

        gen_thread = threading.Thread(target=_generate, daemon=True)
        gen_thread.start()

        # Consume audio chunks as they arrive from the async generator
        try:
            async for chunk_tensor in streamer.get_stream(sample_idx=0):
                audio_np = chunk_tensor.detach().cpu().float().numpy()
                yield audio_np
        finally:
            gen_thread.join(timeout=30)
            if gen_thread.is_alive():
                logger.warning("VibeVoice generate thread did not finish within 30s timeout")

    def dispose(self) -> None:
        """Release model and cache resources."""
        self._voice_cache.clear()
        self._model = None
        self._processor = None
        logger.info("VibeVoiceTTSEngine disposed")
