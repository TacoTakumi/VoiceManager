# Changelog

All notable changes to VoiceManager are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0]

### Added

- `tts_stream` has a `chunking` field: `"sentence"` (the default, unchanged) or
  `"clause"`. In clause mode the server holds the streamed text and cuts
  playback-aware pieces at clause marks, word counts or when the unplayed
  audio runs low, so a client that sends a few words at a time gets smooth
  speech. The first frame of a request fixes the mode. Any other value gets an
  `invalid_request` error frame and the connection stays open. `flush` and
  `done` keep their meaning. Works on the shared and the audiocpp pipelines.
- Settings `VM_TTS_CLAUSE_FIRST_WORDS` (4), `VM_TTS_CLAUSE_FIRST_DEADLINE_MS`
  (600), `VM_TTS_CLAUSE_LATER_WORDS` (12) and `VM_TTS_CLAUSE_MARGIN_MS` (500).
- `chunking` argument in `stream_tts` and `stream_tts_live` of the Python
  client. It is sent on the first frame only.
- `--tts-mode clause|flush` in `scripts/microturn_client.py` live mode (clause
  is the default), and run files go to `runs/microturn/` by default.
- `scripts/microturn_client.py`, a reference client for micro-turn full-duplex
  voice: every 0.6 s a text LLM with the micro-turn LoRA decides to speak a few
  words or to listen. Replay mode runs a scripted call as text. Live mode runs
  a real call over VoiceManager (headphones required) and logs the turn time
  and the delay to the first audio.

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
