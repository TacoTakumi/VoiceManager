"""VoiceManager configuration via pydantic-settings."""
import logging
import os
import sys
from pathlib import Path

from pydantic import Field, ValidationError, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

logger = logging.getLogger(__name__)


def parse_option_list(raw: str) -> list[str]:
    """Parse a comma-separated KEY=VALUE list into its items.

    The shape VM_AUDIOCPP_SESSION_OPTIONS / VM_AUDIOCPP_REQUEST_OPTIONS take, e.g.
    ``"voxtral_realtime.stream_batch_tokens=2,temperature=0.0"``. Items are stripped
    and empties dropped; a non-empty item without a ``KEY=`` prefix raises ValueError
    so a typo refuses to start the server instead of being silently ignored.
    Values may themselves contain ``=``; only the first one splits.
    """
    options: list[str] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, eq, _value = item.partition("=")
        if not eq or not key.strip():
            raise ValueError(
                f"expected KEY=VALUE in option list, got {item!r} (from {raw!r})")
        options.append(item)
    return options


HOME_ENV = "VOICEMANAGER_HOME"


def resolve_home() -> Path:
    """The directory that holds .env and anchors every relative path setting.

    In order: ``$VOICEMANAGER_HOME`` when set; the clone root when the package
    runs from a clone (``pyproject.toml`` two levels above this package, the
    ``pip install -e .`` layout); else ``$XDG_CONFIG_HOME/voicemanager``, or
    ``~/.config/voicemanager`` when XDG_CONFIG_HOME is unset. It does not depend
    on the current directory, so the CLI and the server agree from anywhere.
    """
    override = os.environ.get(HOME_ENV)
    if override:
        return Path(override).expanduser()
    clone_root = Path(__file__).resolve().parents[2]
    if (clone_root / "pyproject.toml").is_file():
        return clone_root
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home) if config_home else Path.home() / ".config"
    return base / "voicemanager"


def resolve_db_path(db_path: str) -> Path:
    """Where the auth store's SQLite file lives, with its parent dirs created.

    ``db_path`` is the VM_DB_PATH setting. When it is blank the file is
    ``$XDG_STATE_HOME/voicemanager/state.db``, or
    ``~/.local/state/voicemanager/state.db`` when XDG_STATE_HOME is unset, so
    the CLI and the server find the same store from the same environment.
    """
    if db_path:
        path = Path(db_path).expanduser()
    else:
        state_home = os.environ.get("XDG_STATE_HOME")
        base = Path(state_home) if state_home else Path.home() / ".local" / "state"
        path = base / "voicemanager" / "state.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# The shared API key setting, removed in favour of per-user bearer tokens. Set
# anywhere, it refuses to load rather than being silently ignored.
REMOVED_KEY_ENV = "VM_API_KEY"
REMOVED_KEY_MESSAGE = (
    f"{REMOVED_KEY_ENV} was removed: remove it from the environment and .env, "
    "and create a bearer token with 'voicemanager user token add'."
)


class RemovedSettingError(ValueError):
    """A setting VoiceManager no longer has is still set."""


class _RemovedKeyGuard(PydanticBaseSettingsSource):
    """A settings source that contributes nothing and refuses the removed key.

    It reads the env and .env sources' own variable maps (lowercased when the
    settings are case-insensitive), so it sees exactly the files and variables
    the real sources would load.
    """

    def __init__(self, settings_cls, *sources: PydanticBaseSettingsSource) -> None:
        super().__init__(settings_cls)
        self._sources = sources

    def get_field_value(self, field, field_name):
        return None, field_name, False

    def __call__(self) -> dict:
        for source in self._sources:
            names = getattr(source, "env_vars", {}) or {}
            if any(name.upper() == REMOVED_KEY_ENV for name in names):
                raise RemovedSettingError(REMOVED_KEY_MESSAGE)
        return {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VM_",
        # The default .env is <home>/.env (see __init__), not ./.env.
        env_file=".env",
        env_file_encoding="utf-8",
        dotenv_filtering="match_prefix",
    )
    host: str = Field(default="0.0.0.0")         # VM_HOST
    port: int = Field(default=8255)              # VM_PORT
    # Auth store file; blank means the XDG state dir (see resolve_db_path).
    db_path: str = Field(default="")             # VM_DB_PATH
    whisper_model: str = Field(default="tiny")   # VM_WHISPER_MODEL
    device: str = Field(default="auto")          # VM_DEVICE: auto|cpu|cuda|cuda:0|cuda:1
    vad_threshold: float = Field(default=0.5)    # VM_VAD_THRESHOLD
    vad_silence_ms: int = Field(default=500)     # VM_VAD_SILENCE_MS
    # OpenVoice v2 checkpoints, where `python setup_models.py openvoice` downloads them.
    tts_checkpoint_dir: str = Field(
        default="checkpoints_v2/converter"
    )  # VM_TTS_CHECKPOINT_DIR
    tts_base_speaker_dir: str = Field(
        default="checkpoints_v2/base_speakers/ses"
    )  # VM_TTS_BASE_SPEAKER_DIR
    tts_language: str = Field(default="EN")      # VM_TTS_LANGUAGE
    tts_speaker_key: str = Field(default="EN-US")  # VM_TTS_SPEAKER_KEY
    tts_speed: float = Field(default=1.0)        # VM_TTS_SPEED
    session_timeout: int = Field(default=300)    # VM_SESSION_TIMEOUT (seconds)
    tts_stream_flush_timeout: int = Field(default=10)  # VM_TTS_STREAM_FLUSH_TIMEOUT (seconds)
    stt_engine: str = Field(default="audiocpp")             # VM_STT_ENGINE: whisper|qwen|nemotron|audiocpp
    qwen_model: str = Field(default="Qwen/Qwen3-ASR-0.6B") # VM_QWEN_MODEL
    qwen_gpu_memory_utilization: float = Field(default=0.5) # VM_QWEN_GPU_MEMORY_UTILIZATION
    qwen_max_new_tokens: int = Field(default=256)           # VM_QWEN_MAX_NEW_TOKENS
    qwen_max_model_len: int = Field(default=4096)            # VM_QWEN_MAX_MODEL_LEN
    qwen_silence_ms: int = Field(default=2000)               # VM_QWEN_SILENCE_MS
    # Nemotron cache-aware streaming STT (only used when VM_STT_ENGINE=nemotron)
    nemotron_model: str = Field(default="nvidia/nemotron-speech-streaming-en-0.6b")  # VM_NEMOTRON_MODEL
    # Cache-aware latency mode: att_context_size = [left, right] in 80ms frames.
    # right in {0,1,6,13}; lower right = lower latency. Confirmed for the en-0.6b model
    # by the NeMo 26.06 re-gate (scripts/spike_nemotron_2606.py).
    nemotron_att_context_left: int = Field(default=70)    # VM_NEMOTRON_ATT_CONTEXT_LEFT
    nemotron_att_context_right: int = Field(default=13)   # VM_NEMOTRON_ATT_CONTEXT_RIGHT
    # End-of-utterance endpointing thresholds (cfg.endpointing.*). Confirmed 26.06 defaults.
    nemotron_eou_stop_history_ms: int = Field(default=800)  # VM_NEMOTRON_EOU_STOP_HISTORY_MS
    nemotron_eou_residue_tokens: int = Field(default=2)     # VM_NEMOTRON_EOU_RESIDUE_TOKENS
    # Speculative endpoint-hold window. On EoU the nemotron branch
    # emits a provisional endpoint frame and holds this long before confirming an
    # immutable final; 0 disables speculation (immediate final, no endpoint frame).
    stt_endpoint_hold_ms: int = Field(default=1000)          # VM_STT_ENDPOINT_HOLD_MS
    # audio.cpp Nemotron-3.5 streaming STT sidecar (only used when VM_STT_ENGINE=audiocpp).
    # These mirror the sidecar CLI (sidecar/README.md): a separate process that runs the
    # audio.cpp runtime and talks to us over a Unix-domain socket, so a native CUDA fault
    # takes down only the sidecar, not this server.
    audiocpp_sidecar_bin: str = Field(
        default="sidecar/build/bin/audiocpp_sidecar")        # VM_AUDIOCPP_SIDECAR_BIN
    # Blank (the default) means OWNED mode: every /ws/stream session spawns its OWN sidecar
    # on a private socket path it mints. A sidecar serves one session at a time and a
    # fatal audio.cpp fault kills every session on that process, so one-per-session is what
    # leaves concurrent sessions unaffected by a fault in one. Setting this to a path
    # switches to EXTERNAL mode: dial that operator-managed listener, never spawn, never
    # terminate (as used for warm-sidecar measurements) -- single-session, so a second
    # concurrent consumer of it gets ERROR `busy`.
    audiocpp_socket: str = Field(default="")                 # VM_AUDIOCPP_SOCKET
    # Default GGUF paths are where `python setup_models.py` downloads them (models/audiocpp/).
    audiocpp_model: str = Field(
        default="models/audiocpp/Nemotron-3.5-ASR-Streaming-0.6B-GGUF/"
                "nemotron-3.5-asr-streaming-0.6b-f16.gguf")  # VM_AUDIOCPP_MODEL
    # Wire model/task id - must be one of the sidecar task registry's entries:
    # "nemotron-3.5-asr-streaming" (the default, unchanged) or
    # "voxtral-mini-4b-realtime-asr-streaming" (point VM_AUDIOCPP_MODEL at the matching
    # q4_k GGUF). An unknown id is refused at HELLO with `unknown_model_task` naming
    # the valid ids.
    audiocpp_model_task: str = Field(default="nemotron-3.5-asr-streaming")  # VM_AUDIOCPP_MODEL_TASK
    audiocpp_backend: str = Field(default="cuda")            # VM_AUDIOCPP_BACKEND: cpu|cuda|vulkan|metal|auto
    audiocpp_device: int = Field(default=0)                  # VM_AUDIOCPP_DEVICE (backend device index)
    audiocpp_threads: int = Field(default=4)                 # VM_AUDIOCPP_THREADS
    # RESOURCE COST / CONCURRENCY CEILING. In OWNED mode every concurrent
    # /ws/stream session runs its OWN sidecar process, and each one costs: one process,
    # one CUDA context, and one resident f16 GGUF -- all on the single device named by
    # VM_AUDIOCPP_DEVICE above. Measured on this box (RTX 4060 Ti as cuda:3, the
    # shipped Nemotron-3.5 f16 model): ~1785 MiB of VRAM per concurrent session at first,
    # ~1480 MiB on the current audio.cpp pin, on top of the ~2.8 GiB the VoiceManager
    # server itself holds. So the practical ceiling is (device VRAM - server baseline) /
    # ~1.5 GiB -- about 8 on a 16 GiB card.
    # This cap is the ADMISSION policy for that ceiling: a session over it is refused
    # up front with a defined retriable error (`at_capacity` -> WS close 4002
    # stt_backend_unready, "retry with backoff"), instead of being allowed to spawn and
    # then die on an out-of-VRAM `load_failed`, which reports a merely-full server to
    # the client as a permanent operator misconfiguration. The default is deliberately
    # below the measured ceiling to leave headroom for fragmentation and graph arenas.
    # Raise it if your device has the VRAM; 0 disables the cap entirely (unbounded --
    # for an operator who has measured their own ceiling and accepts enforcing it by
    # exhaustion). Ignored in EXTERNAL mode (VM_AUDIOCPP_SOCKET), where the single
    # operator-managed sidecar is not ours to spawn or count.
    audiocpp_max_sidecars: int = Field(default=4)            # VM_AUDIOCPP_MAX_SIDECARS
    # chdir target so the vendored model_specs/ resolve; blank = the sidecar's built-in default.
    audiocpp_assets_root: str = Field(default="")            # VM_AUDIOCPP_ASSETS_ROOT
    # Operator-level option overrides: comma-separated KEY=VALUE lists
    # (see parse_option_list) forwarded verbatim to the sidecar as repeatable
    # --session-option / --request-option args, where they override the model_task's
    # registry defaults. Opaque end to end: nothing in VoiceManager interprets a key.
    audiocpp_session_options: str = Field(default="")        # VM_AUDIOCPP_SESSION_OPTIONS
    audiocpp_request_options: str = Field(default="")        # VM_AUDIOCPP_REQUEST_OPTIONS
    # Mid-session endpoint triggers. audio.cpp has no native acoustic EoU
    # and the Nemotron-3.5 model commits its final only at FLUSH (its interim deltas stream
    # during AUDIO), so a per-utterance FLUSH must come from VoiceManager. This many ms of
    # silence ends the utterance, detected two ways: acoustically -- silero-vad classifies
    # inbound frame CONTENT, so a client that streams its mic continuously (HermesConsole)
    # endpoints when the speaker goes quiet -- and as a frame GAP, the fallback for a
    # client that simply stops transmitting between utterances (mirrors qwen_silence_ms).
    audiocpp_silence_ms: int = Field(default=800)            # VM_AUDIOCPP_SILENCE_MS
    # Silero speech-probability threshold for the acoustic trigger (mirrors VM_VAD_THRESHOLD,
    # but per-engine so tuning one engine's VAD cannot shift another's).
    audiocpp_vad_threshold: float = Field(default=0.5)       # VM_AUDIOCPP_VAD_THRESHOLD
    # Idle-feed gating. A continuously-streaming mic
    # client sends inter-utterance silence as FRAMES of quiet audio; without gating every
    # one is billed to the decoder (a real cost on the 4B Voxtral model). With the gate on,
    # frames the acoustic endpointer classifies as OUTSIDE an utterance (before silero's
    # speech-start, after its speech-end) are not fed to the sidecar session. Engine-scoped
    # (audiocpp only) and inert unless the endpointer has a VAD; 0/false restores
    # feed-everything.
    audiocpp_idle_gate: bool = Field(default=True)           # VM_AUDIOCPP_IDLE_GATE
    # Rolling pre-roll replayed into the session at speech onset, so the attack silero
    # needed a chunk-or-two to confirm is not clipped off the front of the utterance.
    # 16 kHz mono s16 = 32 bytes/ms; 200 ms = 6.4 kB kept while gated.
    audiocpp_onset_preroll_ms: int = Field(default=200)      # VM_AUDIOCPP_ONSET_PREROLL_MS
    tts_engine: str = Field(default="audiocpp")              # VM_TTS_ENGINE: openvoice|vibevoice|audiocpp
    vibevoice_model: str = Field(default="microsoft/VibeVoice-Realtime-0.5B")  # VM_VIBEVOICE_MODEL
    vibevoice_voice: str = Field(default="Carter")            # VM_VIBEVOICE_VOICE
    vibevoice_cfg_scale: float = Field(default=1.5)           # VM_VIBEVOICE_CFG_SCALE
    vibevoice_voices_dir: str = Field(default="voices/streaming_model")  # VM_VIBEVOICE_VOICES_DIR

    # audio.cpp dots_tts TTS sidecar (only used when VM_TTS_ENGINE=audiocpp).
    # Shares the sidecar binary and backend/device/threads knobs with the
    # STT engine above - it is the SAME binary, serving TTS when HELLO selects
    # the dots-tts-streaming model_task.
    audiocpp_tts_model: str = Field(
        default="models/audiocpp/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf")  # VM_AUDIOCPP_TTS_MODEL
    audiocpp_tts_model_task: str = Field(default="breeze-tts-streaming")  # VM_AUDIOCPP_TTS_MODEL_TASK
    # Frozen preset voice assets: refs.json pairing reference WAVs with
    # their exact transcripts. Validated fail-fast at startup.
    audiocpp_tts_voices: str = Field(default="voices/designed/refs.json")  # VM_AUDIOCPP_TTS_VOICES
    audiocpp_tts_voice: str = Field(default="designed_female_s101")  # VM_AUDIOCPP_TTS_VOICE (default preset)
    # Seam-silence trim: the sidecar caps a silence run that crosses a
    # dots chunk boundary at this many ms (measured seams run 400-900 ms against
    # 340-620 ms for natural in-chunk pauses). 0 disables the trim.
    audiocpp_tts_seam_silence_ms: int = Field(default=400)   # VM_AUDIOCPP_TTS_SEAM_SILENCE_MS
    # Operator request-option overrides for the TTS sidecar: the same
    # comma-separated KEY=VALUE shape as VM_AUDIOCPP_REQUEST_OPTIONS, forwarded
    # opaquely and merged sidecar-side over the registry pins (text_chunk_size=200,
    # max_tokens=500, text_chunk_mode=default). Empty = the pins stand.
    # seed=2 keeps Breeze output the same from run to run.
    audiocpp_tts_request_options: str = Field(default="seed=2")  # VM_AUDIOCPP_TTS_REQUEST_OPTIONS
    # tts_stream coalescing window: after the FIRST batch has dispatched, keep
    # draining the sentence queue for this many ms before synthesizing. A
    # token-by-token producer completes one sentence at a time and synthesis
    # outruns it, so without this every sentence becomes its own clone request and
    # Breeze re-samples the speaker on each one. 0 restores the pure get_nowait
    # drain. Never applies to the first batch - first audio is never gated on
    # text that has not arrived yet.
    audiocpp_tts_coalesce_ms: int = Field(default=350)       # VM_AUDIOCPP_TTS_COALESCE_MS
    # Composed reference for the later pieces of a chunking="clause" request.
    # Breeze re-conditions the speaker from the reference on every request, and
    # audio.cpp continues a voice only inside one request, so the seam after the
    # first piece of a reply is audible. On, every engine request after the first
    # one clones from a temporary WAV of the preset clip followed by the previous
    # piece's own audio, with the preset transcript plus that piece's text as its
    # ref_text. The first piece, sentence requests and other engines are untouched.
    audiocpp_tts_clause_continue: bool = Field(default=False)  # VM_AUDIOCPP_TTS_CLAUSE_CONTINUE
    # How much of the previous piece's audio joins the composed reference: its
    # last this-many ms (the head is dropped), so a long piece stays a short clip.
    audiocpp_tts_clause_tail_ms: int = Field(default=4000, gt=0)  # VM_AUDIOCPP_TTS_CLAUSE_TAIL_MS
    # Start buffer per engine request: hold its first audio until this many ms
    # are in hand and send them as one chunk. The first frames of a synthesis
    # arrive at about real-time speed, so a client that plays on arrival runs
    # dry between them; later frames run ahead of playback. 0 sends on arrival.
    audiocpp_tts_start_buffer_ms: int = Field(default=250, ge=0)  # VM_AUDIOCPP_TTS_START_BUFFER_MS

    # VoXtream TTS sidecar (only used when VM_TTS_ENGINE=voxtream). VoXtream reads
    # streamed text at the speaking rate, so one generation speaks a whole tts_stream
    # request with no seams. It pins its own torch, so it runs as one separate process
    # from its own virtual environment: VM_VOXTREAM_PY is that environment's python.
    # VoiceManager spawns the sidecar at startup, waits for it to be ready, and stops
    # it at shutdown. The GPU is the one VM_DEVICE resolves to.
    voxtream_py: str = Field(default="venv-voxtream/bin/python")  # VM_VOXTREAM_PY
    voxtream_port: int = Field(default=8256, gt=0)             # VM_VOXTREAM_PORT (loopback)
    voxtream_voices: str = Field(default="voices/designed/refs.json")  # VM_VOXTREAM_VOICES
    voxtream_voice: str = Field(default="designed_female_s101")  # VM_VOXTREAM_VOICE (default preset)
    # Seconds to wait for the sidecar's /health to answer 200 after the spawn. The
    # model load takes a few seconds; the first run also downloads the weights.
    voxtream_ready_timeout: float = Field(default=180.0, gt=0)  # VM_VOXTREAM_READY_TIMEOUT
    voxtream_compile: bool = Field(default=False)              # VM_VOXTREAM_COMPILE (torch.compile)

    # tts_stream "clause" chunking: text is cut at clause marks, word counts and
    # playback-aware deadlines instead of whole sentences. A clause mark cuts only
    # when the piece up to it holds at least mark_min_words words. The first piece
    # goes out at such a mark, after this many words or this many ms, whichever
    # comes first; a later piece goes out at such a mark, after this many words,
    # or once less than the margin of audio is still unplayed, whichever comes
    # first (the unplayed audio is estimated from a play cursor, so a playback
    # gap is not counted as audio played).
    tts_clause_first_words: int = Field(default=3, gt=0)          # VM_TTS_CLAUSE_FIRST_WORDS
    tts_clause_first_deadline_ms: int = Field(default=300, gt=0)  # VM_TTS_CLAUSE_FIRST_DEADLINE_MS
    tts_clause_later_words: int = Field(default=12, gt=0)         # VM_TTS_CLAUSE_LATER_WORDS
    tts_clause_margin_ms: int = Field(default=1000, gt=0)         # VM_TTS_CLAUSE_MARGIN_MS
    tts_clause_mark_min_words: int = Field(default=3, gt=0)       # VM_TTS_CLAUSE_MARK_MIN_WORDS

    # Settings that hold a file or directory path. A relative value resolves
    # against resolve_home(), so it means the same file from any directory.
    _HOME_RELATIVE_PATHS = (
        "tts_checkpoint_dir",
        "tts_base_speaker_dir",
        "audiocpp_sidecar_bin",
        "audiocpp_model",
        "audiocpp_tts_model",
        "audiocpp_tts_voices",
        "vibevoice_voices_dir",
        "voxtream_py",
        "voxtream_voices",
    )

    def __init__(self, **values) -> None:
        values.setdefault("_env_file", resolve_home() / ".env")
        super().__init__(**values)

    @model_validator(mode="after")
    def _anchor_relative_paths(self) -> "Settings":
        home = resolve_home()
        for name in self._HOME_RELATIVE_PATHS:
            value = getattr(self, name)
            if value and not Path(value).expanduser().is_absolute():
                setattr(self, name, str(home / value))
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """The default sources, after a check that the removed key is not set."""
        guard = _RemovedKeyGuard(settings_cls, env_settings, dotenv_settings)
        return guard, init_settings, env_settings, dotenv_settings, file_secret_settings


_settings: Settings | None = None


def get_settings() -> Settings:
    """Lazy accessor for Settings singleton. Fails fast with a clear message on bad config,
    including the removed VM_API_KEY still being set."""
    global _settings
    if _settings is None:
        try:
            _settings = Settings()
        except RemovedSettingError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        except ValidationError as e:
            print(
                "ERROR: Configuration validation failed.\n"
                f"{e}\n\n"
                "Hint: check the VM_* values in the environment and .env "
                "(.env.example lists them).",
                file=sys.stderr,
            )
            sys.exit(1)
    return _settings
