# VoiceManager

A local server for streaming speech-to-text (STT) and text-to-speech (TTS) over one WebSocket. A client sends 16 kHz mono PCM and gets interim and final transcripts back. On the same socket it sends text and gets PCM audio back. STT and TTS run at the same time (full duplex).

The WebSocket protocol is the contract. Clients can be written in any language. This repo ships one Python client, `voicemanager.client`, and full client integration docs in [docs/client-integration.md](docs/client-integration.md).

## Built for conversation

VoiceManager is made for voice agents that talk with a person in real time. One socket carries both directions at once, so the agent can listen while it speaks.

- **Streaming, not only batch.** For STT, the client sends audio as it is captured. It gets interim transcripts (`is_final=false`) while the person speaks, and a final transcript (`is_final=true`) when they stop. To transcribe a recording, `transcribe_file` in the Python client streams the file and returns all the transcripts at once. For TTS there are two modes. `tts` is batch: it takes the full text and returns its audio. `tts_stream` takes text fragments as an LLM produces them, so the audio starts before the reply is complete. Forward every token as it arrives. The server finds the sentence boundaries and groups the text itself.
- **End of speech.** The audiocpp STT engine uses Silero VAD to detect when the person stops speaking, also when the client streams the microphone without a pause.
- **Barge-in.** `tts_cancel` stops a reply at once. The server sends one `audio_end` with `reason: "cancelled"` and no more audio for that request, so the agent can stop talking when the person starts.
- **Turns.** `turn_start` opens a new turn, and each transcript carries the `turn_id` of the audio it came from. A client can then drop late transcripts that belong to a turn it already ended.
- **Flush.** A `tts_stream` frame with `flush: true` speaks the buffered text now and keeps the request open for more text.
- **Clause mode.** A client that streams a few words at a time can set `chunking: "clause"` on the first `tts_stream` frame. The server then holds the text and cuts pieces by clause marks, word counts and how much audio is still unplayed, so the speech is smoother. The default is `"sentence"`. `flush` and `done` keep their meaning. Five settings tune it: `VM_TTS_CLAUSE_FIRST_WORDS` (4), `VM_TTS_CLAUSE_FIRST_DEADLINE_MS` (600), `VM_TTS_CLAUSE_LATER_WORDS` (12), `VM_TTS_CLAUSE_MARGIN_MS` (1000) and `VM_TTS_CLAUSE_MARK_MIN_WORDS` (3). See [Clause mode](docs/client-integration.md#2-clause-mode-chunking-clause).

These features are optional. A client that does not use them gets plain STT and TTS. [Turn-taking protocol surface](docs/client-integration.md#turn-taking-protocol-surface) in `docs/client-integration.md` gives the frames and the rules a client must follow.

## Engines

You select one STT engine and one TTS engine at startup with `.env`.

| Kind | `.env` value | Engine | Notes |
|---|---|---|---|
| STT | `VM_STT_ENGINE=audiocpp` | Nemotron-3.5 ASR streaming (or Voxtral Realtime) in an audio.cpp sidecar | Default. Lowest latency. Needs the sidecar build below. |
| STT | `VM_STT_ENGINE=whisper` | faster-whisper + silero-vad | Works on CPU. |
| STT | `VM_STT_ENGINE=nemotron` | NVIDIA NeMo, Nemotron streaming | Needs `requirements-nemotron.txt`. |
| STT | `VM_STT_ENGINE=qwen` | Qwen3-ASR through vLLM | Needs `requirements-qwen.txt`. |
| TTS | `VM_TTS_ENGINE=audiocpp` | Breeze-TTS 2, Qwen3-TTS, dots.tts or VoxCPM2 in an audio.cpp sidecar | Default (Breeze-TTS 2). Preset voices from `voices/designed/refs.json`. |
| TTS | `VM_TTS_ENGINE=openvoice` | OpenVoice v2 + MeloTTS | The only engine with voice cloning (`/voice/upload`). |
| TTS | `VM_TTS_ENGINE=vibevoice` | VibeVoice-Realtime-0.5B | Preset voices. |

### Why there are so many engines

VoiceManager started with Whisper for STT and OpenVoice for TTS. The first goal was the interface: one WebSocket protocol that hides the engine from the client. With that protocol stable, engines became cheap to add, so we tried new models as they came out. Each engine was a step toward low-latency, real-time streaming in both directions:

- Whisper and OpenVoice transcribe and speak in whole segments. They are simple and run on a CPU, but the client waits for each segment.
- Qwen3-ASR added interim transcripts while the user speaks. VibeVoice added streaming TTS.
- Parakeet, and later Nemotron through NeMo, added cache-aware streaming STT.
- audio.cpp runs GGUF models for both STT and TTS in a native sidecar process, one per session. It gave the lowest latency we measured: on a warm session the first TTS audio arrives in about 110-170 ms with Breeze-TTS 2 and 50-85 ms with Qwen3-TTS. The first request after a sidecar starts is slower, because the sidecar loads the model then. It needs no Python model stack, and a crash in the engine stops only that session's sidecar. New models often arrive there quickly, which is how Breeze-TTS 2, Qwen3-TTS, dots.tts, VoxCPM2 and Voxtral Realtime were added. It is now the default for both STT and TTS.

We kept the older engines. The automated tests still cover each one, and each has a use: Whisper and OpenVoice run without a GPU or a C++ build, OpenVoice is the only engine that clones a voice from an upload, and the others are a starting point for anyone who wants to compare models behind the same protocol.

## Requirements

- Linux (tested on Ubuntu) and Python 3.12+
- An NVIDIA GPU with CUDA is strongly recommended. CPU works for Whisper and OpenVoice, but slowly.
- For the audio.cpp engine: CMake, Ninja, a C++ compiler and the CUDA toolkit

## Install

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .        # installs the voicemanager command
cp .env.example .env
```

### What gets installed

`requirements.txt` installs the server, the Python client and the default audiocpp STT and TTS engines. Each other engine has its own file. Install it only if you use that engine:

| File | Engine | Notes |
|---|---|---|
| `requirements.txt` | audiocpp (default) | Includes PyTorch, which Silero VAD needs. |
| `requirements-whisper.txt` | `whisper` | faster-whisper. |
| `requirements-openvoice.txt` | `openvoice` | MeloTTS from GitHub. OpenVoice itself comes from a clone (see below). |
| `requirements-vibevoice.txt` | `vibevoice` | VibeVoice from GitHub and diffusers. |
| `requirements-nemotron.txt` | `nemotron` | NeMo from a pinned git commit. Large. |
| `requirements-qwen.txt` | `qwen` | qwen-asr and vLLM. Large. |

`python setup_models.py` checks the engines you configure or name. If their packages are missing, it asks before it installs the file.

The audiocpp engines run their models in the sidecar. They use PyTorch only for Silero VAD, which runs on the CPU. To skip the CUDA build of PyTorch (several GB), install the CPU wheels first. Skip this if you also run Whisper, OpenVoice, VibeVoice or NeMo on the GPU, because those need the CUDA build.

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Then set up the engines you want. Each subsection below is independent.

### audio.cpp engine (STT and TTS)

The audio.cpp engine runs each session in its own sidecar process (`sidecar/`), which links a prebuilt audio.cpp engine. VoiceManager is tested against audio.cpp commit `624cf918e51263d39c2a0d28a6c5e3d55990b79f`. Use that commit.

1. Get audio.cpp at the pinned commit:

   ```bash
   git clone https://github.com/0xShug0/audio.cpp 3rdParty/audio.cpp
   git -C 3rdParty/audio.cpp checkout 624cf918e51263d39c2a0d28a6c5e3d55990b79f
   ```

2. Build the audio.cpp engine with CUDA and the model families the sidecar serves:

   ```bash
   cmake -S 3rdParty/audio.cpp -B 3rdParty/audio.cpp/build/linux-cuda-release -G Ninja \
       -DCMAKE_BUILD_TYPE=RelWithDebInfo -DGGML_CUDA=ON \
       -DAUDIOCPP_MODEL_SET=custom \
       -DAUDIOCPP_MODELS="nemotron_asr;voxtral_realtime;dots_tts;voxcpm2;breeze_tts;qwen3_tts"
   cmake --build 3rdParty/audio.cpp/build/linux-cuda-release
   ```

3. Build the sidecar:

   ```bash
   cmake -S sidecar -B sidecar/build -G Ninja
   cmake --build sidecar/build
   ```

   The binary is `sidecar/build/bin/audiocpp_sidecar`, which is the default for `VM_AUDIOCPP_SIDECAR_BIN`. `sidecar/README.md` tells how to build against an audio.cpp tree in a different location.

4. Download the GGUF models:

   ```bash
   python setup_models.py
   ```

   With no arguments, `setup_models.py` sets up only the engines that are configured. With the defaults, it downloads the Nemotron-3.5 ASR f16 and Breeze-TTS 2 q8_0 GGUFs from the Hugging Face repo `audio-cpp/audio.cpp-gguf` into `models/audiocpp/`. That is where `VM_AUDIOCPP_MODEL` and `VM_AUDIOCPP_TTS_MODEL` point by default. Read each model's license before you use it. Breeze-TTS 2 has a non-commercial license.

5. The defaults need no `.env` change. They are:

   ```
   VM_STT_ENGINE=audiocpp
   VM_AUDIOCPP_MODEL=models/audiocpp/Nemotron-3.5-ASR-Streaming-0.6B-GGUF/nemotron-3.5-asr-streaming-0.6b-f16.gguf
   VM_AUDIOCPP_MODEL_TASK=nemotron-3.5-asr-streaming

   VM_TTS_ENGINE=audiocpp
   VM_AUDIOCPP_TTS_MODEL=models/audiocpp/Breeze-TTS-2-GGUF/breeze-tts-2-q8_0.gguf
   VM_AUDIOCPP_TTS_MODEL_TASK=breeze-tts-streaming
   VM_AUDIOCPP_TTS_REQUEST_OPTIONS=seed=2

   VM_AUDIOCPP_DEVICE=0
   ```

   Relative paths resolve against the clone (see Configuration), so the server finds them from any directory. To use a different GGUF, set its path in `.env` and run `python setup_models.py` again. The other TTS tasks are `qwen3-tts-streaming`, `dots-tts-streaming` and `voxcpm2-tts-streaming`. The other STT task is `voxtral-mini-4b-realtime-asr-streaming`. `.env.example` lists the GGUF that each task needs, and the VRAM and concurrency settings.

Each WebSocket session starts its own sidecar on `VM_AUDIOCPP_DEVICE`, so each session costs GPU memory. `VM_AUDIOCPP_MAX_SIDECARS` (default 4) limits the number of concurrent sessions. The server refuses a session over the limit with close code 4002.

The TTS voices are the presets in `voices/designed/refs.json`. Each entry pairs a reference WAV with its exact transcript. The default voice is `designed_female_s101` (`VM_AUDIOCPP_TTS_VOICE`). To add a voice, add a WAV and an entry to that file.

Copyright notice: the `lcars_demo` voice is made from computer voice clips from Star Trek: The Next Generation. That audio is copyrighted by its owners and is included for personal, non-commercial use only. VoiceManager is not affiliated with or endorsed by the owners of Star Trek. The `designed_*` voices are synthetic and were made with voice design.

### Whisper and OpenVoice

```bash
pip install -r requirements-whisper.txt
python setup_models.py whisper     # the faster-whisper model in VM_WHISPER_MODEL

pip install -r requirements-openvoice.txt
git clone https://github.com/myshell-ai/OpenVoice.git ../OpenVoice
pip install -e ../OpenVoice
python setup_models.py openvoice   # OpenVoice v2 checkpoints and NLTK data
```

### VibeVoice

```bash
pip install -r requirements-vibevoice.txt
python setup_models.py vibevoice
```

Set `VM_TTS_ENGINE=vibevoice`. The voice is `VM_VIBEVOICE_VOICE` (for example Carter, Davis, Emma, Frank, Grace or Mike).

## Configuration

All settings are `VM_*` variables in `.env`. `.env.example` lists each one with its default and an explanation. The server does not start if `.env` contains an unknown `VM_` key.

`.env` lives in the VoiceManager home directory, and relative paths in settings (models, the sidecar binary, voice files) resolve against it. The server and the `voicemanager` command therefore behave the same from any directory. The home is `$VOICEMANAGER_HOME` when set, else the clone (for a `pip install -e .` install), else `~/.config/voicemanager/`.

Some important settings:

- `VM_HOST`, `VM_PORT`: the listen address (default `0.0.0.0:8255`). 8255 spells TALK on a phone keypad.
- `VM_DEVICE`: `auto` (the CUDA GPU with the most free memory), `cpu`, `cuda` or `cuda:N`. The audio.cpp engine uses `VM_AUDIOCPP_DEVICE` instead.
- `VM_DB_PATH`: the user store (default `~/.local/state/voicemanager/state.db`)
- `VM_SESSION_TIMEOUT`: close an inactive session after this many seconds (default 300)

## Users and tokens

Clients authenticate with a bearer token in the `Authorization: Bearer <token>` header. Each token belongs to a user. The server keeps users and tokens in a SQLite file (`VM_DB_PATH`) and stores only the SHA-256 of each token.

Make a user and a token on the server host:

```bash
voicemanager user add myuser                       # asks for a password
# Now you can sign in to the web UI (see Web UI below) to make tokens there,
# or continue with the commands below.
voicemanager user token add myuser my-client       # prints the vmp_... token one time
voicemanager user token voice myuser my-client designed_male_s202   # optional
```

A token can have its own voice. The server selects the voice in this order: the `voice` field of the request, then the voice of the token, then the `.env` default.

Other commands: `voicemanager user list|passwd|remove` and `voicemanager user token list|revoke`. Add `--help` to any command for its options.

### Web UI

Open `http://<host>:<port>/` and sign in. A user can make and revoke their own tokens, set the voice of a token and change their password. A user that was added with `--no-password` cannot sign in, but can own tokens. The web UI uses no JavaScript. After 5 failed sign-ins in 15 minutes, the server locks the sign-in for that user.

## Run

```bash
source venv/bin/activate
voicemanager serve
```

`python -m voicemanager` does the same thing. The server loads the models at startup and does not start if a model does not load. `GET /health` does not need a token.

### Run as a service

`contrib/systemd/` has two sample units. Each file starts with its install steps and the lines to adjust.

- `contrib/systemd/system/voicemanager.service` is a system unit that runs the server as one user. Replace `you` with that account.
- `contrib/systemd/user/voicemanager.service` is a user unit for `systemctl --user`.

The server and its audio.cpp sidecars write their logs to stderr, so systemd puts them in the journal: `journalctl -u voicemanager -f` for the system unit, and `journalctl --user -u voicemanager -f` for the user unit.

## Use

Connect to `ws://<host>:<port>/ws/stream` with the header `Authorization: Bearer <token>`. The server starts with a `ready` message that gives the session id, the active engines, `available_voices` and `default_voice`. Binary frames go to STT. JSON text frames (`tts`, `tts_stream`, `tts_cancel`) go to TTS.

The Python client wraps the protocol:

```python
from voicemanager.client import VoiceManagerClient

async with VoiceManagerClient(token="vmp_...") as client:
    results = await client.transcribe_file("audio.wav")           # STT

    pcm, rate = await client.synthesize_tts("Hello world")        # TTS
    client.save_tts_to_file(pcm, "output.wav", rate)

    pcm, rate = await client.synthesize_tts("Hello", voice="designed_male_s202")
```

**[docs/client-integration.md](docs/client-integration.md)** is the full reference: message schemas, close codes, REST endpoints (`/health`, `/info`, `/voice/upload`), the client methods and known problems. Use it to write a client in a different language.

## Tests

```bash
pip install -r requirements-dev.txt
./venv/bin/python -m pytest tests/
```

The automated tests do not need a server, a GPU, models or audio hardware. Some `tests/test_*.py` files are manual scripts for a running server. `tests/conftest.py` excludes them from the pytest run.

## License

The VoiceManager code is under the MIT License. See `LICENSE`.

Other parts have their own terms:

- Model weights have their own licenses. Breeze-TTS 2 has a non-commercial license. Read each model's license before you use it.
- The `lcars_demo` voice is for personal, non-commercial use only. See the copyright notice above.
- The openvoice extra (`requirements-openvoice.txt`) installs GPL packages: Unidecode and phonemizer.
