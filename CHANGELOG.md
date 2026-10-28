# Changelog

All notable changes to VoiceManager are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Added

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
- `--tts-mode clause|flush` in `scripts/microturn_client.py` live mode (clause
  is the default), and run files go to `runs/microturn/` by default.
- `scripts/microturn_client.py`, a reference client for micro-turn full-duplex
  voice: every 0.6 s a text LLM with the micro-turn LoRA decides to speak a few
  words or to listen. Replay mode runs a scripted call as text. Live mode runs
  a real call over VoiceManager (headphones required) and logs the turn time
  and the delay to the first audio.

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
