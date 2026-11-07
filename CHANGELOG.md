# Changelog

All notable changes to VoiceManager are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.1]

### Fixed

- Clause-mode `tts_stream` with a client that streams LLM tokens. A word split
  across tokens ("subsystem" + "s.") is no longer cut in two: a word-count or time
  trigger cuts at the last word end, which is whitespace, or the end of the held
  text once no new frame arrived for the word grace. A piece with no letter or
  digit (a lone ".") is never sent to the engine: it starts the next piece, or is
  dropped at `done` or `flush`. Pieces keep the client's whitespace, so the
  audiocpp engine no longer puts a space inside a split word or before
  punctuation ("tri age", "categories ,"), and a paragraph break no longer joins
  two sentences ("use.One").

### Added

- Setting `VM_TTS_CLAUSE_WORD_GRACE_MS` (250). The end of the held text counts as
  a word end once no new frame arrived for this long. 0 makes every frame end a
  word end, the behaviour before this release. A client that sends whole words
  with no trailing space gets each count cut, and the first-piece deadline cut,
  at most this much later. The margin cut does not wait: it cuts at the last
  whitespace, so the last word can start the next piece.

## [0.8.0]

### Added

- A listening boundary on one connection. A client sends `listen_stop` when it
  stops sending mic audio (stop listening, end of a conversation, idle) and
  `listen_start` before its first audio after the stop. The server answers each
  `listen_stop` with `listen_stopped`, whose `mic_sample` is the count of mic
  samples it accepted on the connection; `listen_start` gets no answer. Both
  frames are idempotent and never send an `error` frame. A connection starts
  listening, so a client that never sends them sees no change, and older servers
  drop them silently. See "Listening boundary" in `docs/client-integration.md`.
- While stopped, mic frames are dropped with no error and are not counted. At the
  stop the server drops every transcription from audio before it that is not sent
  yet and discards the STT engine's open utterance, on every STT engine. On a
  turn-engine connection it also drops the pending finals, the partial hop and the
  mic hold, and sends no `turn_end` or `barge` for audio before the stop.
- A stopped connection is exempt from `VM_SESSION_TIMEOUT`: no `timeout_warning`,
  no `1001` close. After `listen_start` the timer runs again from that frame.
- The Tyger turn engine keeps the connection's calibrated voice gate
  (`VM_TYGER_GATE_MARGIN_DB`, `VM_TYGER_GATE_MAX_DB`) and, with the experimental
  pre-filter (`VM_TYGER_PREFILTER=voxlock`), its VoxLock enrollment across a
  stop and a start, so the user does not calibrate or give the 3 s voice sample
  again. The turn-engine opt-in and the `turn_id` counter are kept too.
- `listen_stop` does not cancel TTS: a `tts` or `tts_stream` request while stopped
  is synthesized (for example a spoken goodbye). On a turn-engine connection,
  `playback` frames while stopped are dropped with no answer, and a reply begun
  while stopped is not tracked for barge.
- Python client: `DuplexSession.listen_stop()` stops listening and returns the
  `mic_sample` of `listen_stopped` (`None` from an older server, or when no answer
  comes in time), `DuplexSession.listen_start()` listens again, and the
  `DuplexSession.listening` property tells the state. While stopped, `send_audio`
  sends and counts nothing, so `mic_sample` keeps matching the server's count.

## [0.7.0]

### Added

- `VM_TYGER_GATE_MARGIN_DB` (default 8 dB) and `VM_TYGER_GATE_MAX_DB` (default
  -38 dBFS): the margin and the cap of the calibrated voice gate below.

### Changed

- The Tyger turn engine calibrates its voice gate per connection. Each session
  measures the room from its first 2 s of mic audio heard while no reply plays,
  and sets its gate to the room level plus `VM_TYGER_GATE_MARGIN_DB`, no lower
  than `VM_TYGER_VOICE_GATE_DB` and no higher than `VM_TYGER_GATE_MAX_DB`.
  `VM_TYGER_VOICE_GATE_DB` (default -50 dBFS) is now the gate before calibration
  and the lowest gate. On the mics tested the room sat at -50 to -52 dBFS, so the
  fixed -50 dBFS gate rarely saw silence and a turn could wait 20 s or more for
  the silence fallback. To pin a fixed gate as before, set the margin to 0 and the
  max to the gate.
- Experimental pre-filter (`VM_TYGER_PREFILTER=voxlock`): the session takes its
  3 s voice sample only after the gate is calibrated, and only from audio above
  that gate, so it no longer learns room noise. The order on a connection is:
  calibrate the gate, learn the voice, then allow barge-in.
- The README's server-side barge-in steps now say what to expect on speakers
  (calibration, then the voice sample, then barge-in), the hardware it was tested
  with, and how to record a session and replay it offline.

### Fixed

- Experimental pre-filter (`VM_TYGER_PREFILTER=voxlock`): no barge fires until the
  connection has its 3 s voice sample. Before that, Tyger heard the raw mic, so on
  speakers an agent that speaks first (before the user has said 3 s) could stop
  its own reply on its echo, and the held echo went to STT as a user turn. The
  user can now cut into replies only after they have spoken for about 3 s.
- Experimental pre-filter: VoxLock is now run once over its 3 s voice sample when
  it learns it. Its running state used to start with only a few seconds of the
  user, so on speakers it could pass the echo of the first reply after it learned
  the voice, and that echo stopped the reply.

## [0.6.0]

### Added

- Experimental: barge-in on speakers with the Tyger turn engine.
  `VM_TYGER_PREFILTER=voxlock` puts a target-speaker filter (VoxLock) in the
  Tyger sidecar. Each connection learns the user's voice from its first 3 s of
  speech while no reply plays; from then on, while a reply plays, Tyger hears that
  voice filtered out of the mic, plus noise at `VM_TYGER_PREFILTER_FLOOR_DB`
  (default -52 dBFS), and no reply channel, so a barge in the middle of a reply
  on speakers is heard as on headphones. Off by default; no protocol change. It
  needs `onnxruntime-gpu` (now in `requirements-tyger.txt`) and runs on the
  sidecar's GPU only; `python setup_models.py tyger` fetches its models when the
  setting is on. The VoxLock code is adapted from voxlock 1.3.0 and ships with its
  Apache-2.0 `LICENSE` and a `NOTICE`.

## [0.5.1]

### Changed

- The README's Tyger section and the client guide's Turn engine overview point to
  `scripts/voice_agent.py classic --turn-engine tyger` as an example client.

## [0.5.0]

### Added

- A turn engine: `VM_TURN_ENGINE=tyger` starts the Tyger turn-taking model
  (NineNineSix) as one sidecar process from its own virtual environment
  (`venv-tyger`, `requirements-tyger.txt`, `VM_TYGER_PY`), on `VM_TYGER_DEVICE`.
  It hears the user's mic and the reply as it plays at the client, and decides
  when the user's turn ended and when the user cut in. It works with the audiocpp
  STT engine only. `python setup_models.py tyger` downloads the bundle at a pinned
  revision and checks its checksums. The turn rules are `VM_TYGER_*` settings
  (see `.env.example`). The server watches the sidecar and starts it again when
  it dies. The bundle is under OpenMDW-1.1 (NVIDIA's Nemotron weights in it under
  NVIDIA's own OpenMDW-1.1 grant); the turn-rule code adapted from tyger-s2s
  ships with its Apache-2.0 `LICENSE` and a `NOTICE`.
- Opt-in turn-engine frames on the WebSocket, all additive: a client sends
  `turn_engine` before its first audio frame and gets `turn_engine_ready`, reports
  each reply's playback edges with `playback` (`start` and `stop`, placed on its
  own mic stream by `mic_sample`), and then gets `barge` when the user cut in over
  a reply and `turn_end` (the finals of the turn joined, `by` `model` or
  `silence`) when the user's turn ended. New `error_type` values
  `turn_engine_unavailable`, `turn_engine_unsupported` and `turn_engine_lost`.
  A connection that does not opt in, and a server with `VM_TURN_ENGINE=off`,
  behave as before.
- For an opted-in connection the server holds the mic audio back from the STT
  while a reply plays, so the STT does not transcribe the reply's echo; after a
  barge, the user's speech from its onset is transcribed. If the sidecar is lost,
  each opted-in connection gets one `turn_engine_lost` frame and goes on as a
  plain connection.
- Python client: `DuplexSession.enable_turn_engine()`, `playback_start()`,
  `playback_stop()`, the `mic_sample` property, `turn_ends()`, `("barge", frame)`
  on `audio()` and `TurnEngineError`.
- `scripts/voice_agent.py classic --turn-engine tyger` opts in, reports playback
  edges from the output DAC time and the mic's first ADC time, ends a turn on
  `turn_end` and clears its playback on a barge. `classic --capture` saves the
  same mic, output and timing files as `microturn --capture`.
- `docs/client-integration.md` documents the turn engine for clients in any
  language, and the README covers its setup and licences.

### Changed

- `audio_end` reason `interrupted`, reserved until now, is sent: it ends a reply
  the turn engine took as barged, after that request's `barge` frame. Only an
  opted-in connection can get it.

## [0.4.2]

### Added

- `scripts/voice_agent.py microturn --capture` also saves the mic as captured
  (`<stem>-mic.wav`, 16 kHz), everything sent to the output device with its
  silence (`<stem>-out.wav`) and both streams' first-sample clock times
  (`<stem>-capture.json`), so an offline turn-model replay can line the two
  channels up. Off by default; the existing WAV is unchanged.
  The JSON also holds `log.t0_monotonic`, the clock time the call log's `t`
  counts from, so each log line can be placed on the mic's clock.
- `scripts/voice_agent.py microturn --no-cut-in` lets a playing reply play out
  when the user's words arrive over it, for echo tests on speakers. The cut-in
  stays on by default.

## [0.4.1]

A sweep of the small corrections that earlier review rounds left open.

### Fixed

- `cancel_tts` and `flush_tts` on the Python client raise `TTSError` and
  reconnect on a lost connection, as `send` and `done` already did; before,
  the websockets `ConnectionClosed` leaked out to the caller.
- A VoXtream generation reports `audio_end` reason `complete` only when the
  sidecar sent its `eos` frame; a connection the sidecar closed cleanly without
  one is reason `error`.
- The `tts_stream` idle flush is re-armed only by a frame of the open request;
  a late frame for a cancelled or superseded request no longer moves it.
- The VoXtream watcher logs the "sidecar is gone" line once per death; a
  retry after a failed restart logs the restart error alone.
- `POST /voice/upload` accepts only plain ASCII digits in `Content-Length`, and
  its OpenAPI entry carries the multipart request body again (it was lost when
  the handler started reading the body itself).
- The `Authorization` scheme is matched case-insensitively (`bearer` works, as
  RFC 7235 requires); the secret keeps its case. A new auth store file is
  created 0600 in a 0700 directory. Token names are stored stripped whether
  they come from the CLI or the web form.
- The reference voice agent records an `audio_end` with reason `error` as an
  error, and the micro-turn client's TTS sink forgets a request at its
  `audio_end` instead of keeping an entry for the life of the call.
- The audiocpp TTS evidence script takes the sample rate from `audio_start`,
  so a 24 kHz engine's long-form WAV, duration and RTF are labelled right.

### Changed

- Every Python file in the package is ASCII: em dashes became ` - `, and the
  sentence splitter writes its CJK and quote characters as escapes. A test
  keeps non-ASCII bytes out of the package. The voice upload "too short"
  message lost its em dash too.
- The client guide's clause-mode debug pointer quotes the current log line and
  the first-audio line that follows it, and the classic outline says where it
  differs from the full script (one dispatch per turn, a new turn per reply).
- Stale comments corrected: the sidecar registry says six entries ship, a drive
  test states its claim instead of pointing at a review round, and the Python
  tests lose a bare marker, a narrated task list and lowercase sentence starts.

## [0.4.0]

### Added

- The VoXtream sidecar is watched after startup. When the sidecar process dies
  the server starts it again through the same path as at startup, and a start
  that fails is logged and tried again every few seconds. Before, a dead
  sidecar left every TTS request failing until a person restarted the server.
  At shutdown the watcher stops before the sidecar, and a restart still waiting
  for the sidecar is interrupted, so no sidecar outlives the server.

### Changed

- `GET /health` answers 503 with `status: "degraded"` and `tts: null` while the
  VoXtream sidecar is dead or loading again, and 200 `ok` once it is ready. The
  body keeps its shape; the STT and VAD slots are unchanged. This is why the
  version is a minor bump: a documented REST endpoint has a new answer.
- The VoXtream sidecar runs in its own session, and the contrib systemd units
  set `KillMode=mixed`, so a service stop or a Ctrl-C signals the server only
  and the server stops the sidecar itself. Before, the stop signal reached the
  sidecar first and the watcher could start a new one that the shutdown then
  killed. An installed unit needs the new line and a `daemon-reload`.
- A terminal hangup (a dropped SSH session, a closed terminal pane) now runs
  the server's orderly shutdown, which stops the sidecar. Before, the hangup
  ended the server without its lifespan; with the sidecar in its own session
  that would have left it running on its port. A hangup that is already
  ignored (`nohup`) stays ignored.

- The micro-turn reference client (`scripts/microturn_client.py`) sends the
  model its transcript fragments in the form the adapter was trained on. A
  word the STT cut at a tick boundary is completed by the next fragment, which
  starts with the missing letters ("tell me a ra", then "ndom fact"), and a
  mark the STT adds late to a sent word is sent the same way. When a
  segment's final arrives without an end mark, the client sends one: "?" for a
  segment that opens with a question word, otherwise ".", on the last word
  sent for that segment in the tick or alone, once per segment. In a text
  replay of seven live runs this form raised the answered requests from 6 of
  38 to 18 of 38. The server, the protocol and the classic voice agent are
  unchanged.

### Fixed

- The micro-turn client's late end mark is judged on the words the model
  received, not on the final's text: a final that revises the last sent word
  ("honey" to "money?") sends no words but still ends the model's history with
  a mark, the final's own when it has one. After a shrink, regrowth along the
  words the model received counts no second revision, however many ticks it
  takes or where its frames cut a word, so the tick log's `revised` and
  `revisions` fields no longer overcount a mid-word shrink.
- `POST /voice/upload` checks the bearer token, the engine and the session
  before it reads the multipart body. FastAPI read the declared file parameter
  before any dependency ran, so an unauthenticated client could push an
  arbitrarily large upload that was spooled to disk before the 401. The body is
  now capped by a new `VM_UPLOAD_MAX_BYTES` setting (default 20 MiB): a
  `Content-Length` above it answers 413 before any read, and a chunked body is
  cut off at the cap with 413. A form without a `file` field answers 400
  instead of 422. The accepted path is unchanged.

## [0.3.0]

### Added

- A `voxtream` TTS engine (`VM_TTS_ENGINE=voxtream`): VoXtream2.1 reads streamed
  text at the speaking rate, so a `tts_stream` request is one continuous
  generation from its first word to its last, with no seams between pieces.
  `flush` has no effect on it, `chunking` makes no difference, and the client
  protocol is unchanged. VoXtream pins its own PyTorch, so it runs as one
  sidecar process from its own virtual environment (`venv-voxtream`, three
  commands in the README), spawned at startup on the `VM_DEVICE` GPU and
  stopped at shutdown; startup refuses with the setting named when the
  interpreter is missing, the sidecar is not ready in time, or
  `VM_VOXTREAM_PORT` is already in use (an orphaned sidecar is never adopted).
  The voice is a preset from the designed presets file (the reference WAV
  only); an unknown `voice` on `tts_stream` ends that request with an `error`
  frame of `error_type` `unknown_voice` naming the presets, then `audio_start`
  and `audio_end` with reason `error`, and the connection stays open. A
  generation that fails before the engine finishes (sidecar unreachable, an
  error frame, a dropped connection) ends with `audio_end` reason `error`, as
  the one-shot `tts` path already did. One generation is capped at 60 s of
  audio by the shipped generator config; a longer request is cut there with a
  normal `audio_end` and the rest of its text is dropped. `/voice/upload`
  answers 400 under this engine (no cloning). `/health` names the engine as
  `voxtream (herimor/voxtream2.1)` and `/info` reports `tts_model`
  `herimor/voxtream2.1`. Settings `VM_VOXTREAM_PY`, `VM_VOXTREAM_PORT`,
  `VM_VOXTREAM_VOICES`, `VM_VOXTREAM_VOICE`, `VM_VOXTREAM_READY_TIMEOUT` and
  `VM_VOXTREAM_COMPILE`. The encoded reference clip of each preset is cached
  beside its WAV (`<name>.prompt.npy`) after the first request. English only;
  the weights are CC BY-NC 4.0.
- `tts_stream` has a `chunking` field: `"sentence"` (the default, unchanged) or
  `"clause"`. In clause mode the server holds the streamed text and cuts
  playback-aware pieces at clause marks, word counts or when the unplayed
  audio runs low (a clause mark cuts only after at least 3 words, and the
  unplayed estimate is a play cursor that ignores playback gaps), so a client
  that sends a few words at a time gets smooth speech. The first frame of a
  request fixes the mode. Any other value gets an
  `invalid_request` error frame and the connection stays open. `flush` and
  `done` keep their meaning. Works on the shared and the audiocpp pipelines.
  Each engine request of a clause-mode request logs its text at INFO.
- Settings `VM_TTS_CLAUSE_FIRST_WORDS` (3), `VM_TTS_CLAUSE_FIRST_DEADLINE_MS`
  (300), `VM_TTS_CLAUSE_LATER_WORDS` (12), `VM_TTS_CLAUSE_MARGIN_MS` (1000) and
  `VM_TTS_CLAUSE_MARK_MIN_WORDS` (3).
- Settings `VM_AUDIOCPP_TTS_CLAUSE_CONTINUE` (off) and
  `VM_AUDIOCPP_TTS_CLAUSE_TAIL_MS` (4000). With continuation on, each engine
  request of a clause-mode request after the first clones from the preset clip
  plus the last `VM_AUDIOCPP_TTS_CLAUSE_TAIL_MS` ms of the previous piece's
  audio and its text, to carry the voice across the seam. It is off by default:
  those pieces start about 440 ms later and the voice still changed at seams.
- Setting `VM_AUDIOCPP_TTS_START_BUFFER_MS` (250, `0` turns it off). On
  `tts_stream`, the audiocpp engine holds the first 250 ms of audio of each
  engine request (each clause piece or batch) and sends it as one chunk. Later
  chunks go out as they arrive. A request shorter than that is sent when it
  ends. The first frames of a synthesis arrive at about real-time speed, so a
  client that played on arrival ran dry several times at the start of each
  utterance. The cost is up to about 250 ms more to the first audio of each
  piece. No protocol change.
- `chunking` argument in `stream_tts` and `stream_tts_live` of the Python
  client. It is sent on the first frame only.
- `--tts-mode clause|flush` in the micro-turn live mode (clause is the
  default), and its run files go to `runs/microturn/` by default.
- `scripts/voice_agent.py`, the reference voice agent, with three subcommands.
  `classic` is the turn pattern with any OpenAI-compatible chat model: one
  `turn_start` per utterance, dispatch on the first endpointed transcript (the
  final on engines without an endpoint frame), supersede when the final revises
  the text, the reply streamed token by token into one clause-mode `tts_stream`
  request, cancel when the person talks over it (also during the tail of a
  reply the server has finished sending), thinking off in the chat request
  unless `--think` is given (a thinking model otherwise reasons for seconds
  before its first spoken word), a warm-up request at startup so the model is
  loaded before the first turn, run files under `runs/classic/`. `microturn`
  is the micro-turn full-duplex loop: every 0.6 s
  a text LLM with the micro-turn LoRA decides to speak a few words or to
  listen, over a real call with headphones, logging the turn time and the
  delay to the first audio. `replay` runs that loop on a scripted call as
  text. The micro-turn loop stays in `scripts/microturn_client.py` as a module;
  its own command line is gone.
- `VoiceManagerClient.duplex()`: a `DuplexSession` that runs STT and TTS at the
  same time on one connection, with `send_audio`, `transcripts()`,
  `turn_start()`, `tts_request()` (a `TTSRequest` with `send`, `flush`, `done`
  and `cancel`), `audio()` and `errors()`. The one-sided methods raise while a
  session is open.

- README: a "Conversation with a text LLM" section on the two ways to build
  an agent on VoiceManager, the classic turn pattern with any LLM and the
  experimental micro-turn pattern with the published LoRA, its links, its state
  and where it is going.

### Fixed

- The `tts_stream` idle flush (`VM_TTS_STREAM_FLUSH_TIMEOUT`) now counts from
  the last `tts_stream` frame of the request. Before, every frame on the socket
  restarted it, so a client that streamed microphone audio on the same socket
  kept the buffered text of a request from ever being flushed.

## [0.2.0]

### Changed

- **Breaking:** the default port is now 8255 (TALK on a phone keypad), not
  8000. Port 8000 is the default of many other servers. To keep the old port,
  set `VM_PORT=8000` in `.env` and pass `port=8000` to `VoiceManagerClient`.

## [0.1.0] - 2026-10-18

First public release. Developed privately since March 2026.

- Streaming STT and TTS over one WebSocket (`/ws/stream`), full duplex: interim
  and final transcripts, batch `tts` and streaming `tts_stream`, `tts_cancel`
  for barge-in, turns and flush. The protocol is documented in
  `docs/client-integration.md`, so a client can be written in any language.
- Python client library `voicemanager.client`.
- Pluggable engines. Default: Nemotron-3.5 ASR and Breeze-TTS 2 in an
  audio.cpp sidecar. Also faster-whisper, NeMo Nemotron and Qwen3-ASR for STT,
  and OpenVoice v2 (voice cloning) and VibeVoice for TTS.
- Users and bearer tokens in a SQLite store, each token with an optional voice.
  The `voicemanager` command runs the server and manages users and tokens.
- Web UI without JavaScript, where users manage their tokens and password.
- Preset voices in `voices/designed/`, and sample systemd units in
  `contrib/systemd/`.
