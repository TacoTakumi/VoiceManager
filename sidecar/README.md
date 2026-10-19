# audio.cpp streaming sidecar

A separate process that runs the audio.cpp Nemotron-3.5 streaming STT engine and talks
to VoiceManager over a Unix-domain-socket wire protocol. The process boundary is the
crash isolation: a fatal fault in the native CUDA runtime (a GGML_ASSERT, etc.) takes
down only the sidecar, not the VoiceManager server. The sidecar drives audio.cpp
through its generic runtime interface (`IStreamingVoiceTaskSession`) plus the model
registry, never a concrete model class, so adding a model later is a registry entry, not
a protocol change.

The wire protocol is documented in [PROTOCOL.md](PROTOCOL.md).

## Status

- The socket server and the v1 handshake (HELLO -> READY/ERROR).
- The streaming drive loop (AUDIO -> EVENT interim deltas, FLUSH ->
  FINAL, RESET) over `IStreamingVoiceTaskSession` + the registry; loads the model on the
  handshake and links the prebuilt `engine_runtime` from audio.cpp.
- Containment of a process-fatal fault as a detectable client signal.

## Build

The sidecar links the **prebuilt** static engine from the pinned audio.cpp checkout
(`3rdParty/audio.cpp/build/linux-cuda-release/`) rather than rebuilding the CUDA runtime,
so build audio.cpp first (its `linux-cuda-release` preset). One engine build tree
compiles every shipped family via audio.cpp's composite model flag; configure
it with:

    cmake ... -DAUDIOCPP_MODEL_SET=custom \
        -DAUDIOCPP_MODELS="nemotron_asr;voxtral_realtime;dots_tts"

Exactly one sidecar binary links against that tree and serves both ASR and TTS
sessions. Then, from the repo root:

    cmake -S sidecar -B sidecar/build -G Ninja
    cmake --build sidecar/build

The binary is `sidecar/build/bin/audiocpp_sidecar`.

If the vendored audio.cpp or its build tree lives elsewhere, override the paths:

    cmake -S sidecar -B sidecar/build \
        -DAUDIOCPP_ROOT=/path/to/audio.cpp \
        -DAUDIOCPP_BUILD=/path/to/audio.cpp/build/linux-cuda-release

## Run

    sidecar/build/bin/audiocpp_sidecar \
        --socket /tmp/audiocpp-sidecar.sock \
        --model /path/to/nemotron-3.5-asr-streaming-0.6b-f16.gguf \
        --backend cuda --device 0 --threads 4

`--model` is required. `--backend`/`--device`/`--threads` pick the inference backend
(default `cpu`, device 0). The sidecar loads the model on a valid handshake, so a READY
ack means the session is warm. It resolves the vendored `model_specs/` by chdir-ing into
the audio.cpp tree it was built against; override with `--assets-root <dir>`. Send it
SIGINT/SIGTERM to shut down; it unlinks the socket on exit.

## Verify

Both probes are **manual** scripts (outside `tests/`, so `pytest tests/` never collects
them). Each can spawn the sidecar on a temp socket, run its checks, and tear it down.

`handshake_probe.py` - the handshake only. Needs no GPU and loads no model:

    python sidecar/handshake_probe.py --spawn sidecar/build/bin/audiocpp_sidecar

`stream_probe.py` - the streaming drive loop end to end. Needs the model and a GPU. It
streams a chunked reference utterance (`voices/designed/designed_female_s101.wav`), then
asserts >=1 interim EVENT delta + exactly one FINAL, with real transcript text, and a
warm second utterance after RESET:

    python sidecar/stream_probe.py \
        --spawn sidecar/build/bin/audiocpp_sidecar \
        --model /path/to/nemotron-3.5-asr-streaming-0.6b-f16.gguf \
        --backend cuda --device 0
