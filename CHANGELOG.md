# Changelog

All notable changes to VoiceManager are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
