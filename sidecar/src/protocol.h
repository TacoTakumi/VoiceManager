// Wire-protocol constants for the audio.cpp streaming sidecar (v1).
//
// This is the single source of truth for the frame tags and fixed limits that
// PROTOCOL.md documents. The framing is a 5-byte header (1-byte type tag + a
// 4-byte big-endian payload length) followed by the payload; see frame.h.
#pragma once

#include <cstddef>
#include <cstdint>

namespace sidecar {

// Bumped only on a breaking wire change; carried in HELLO/READY as "protocol".
constexpr int kProtocolVersion = 1;

// A frame's declared payload length may not exceed this; larger is malformed.
constexpr std::uint32_t kMaxPayloadBytes = 16u * 1024u * 1024u;  // 16 MiB

// Fixed 5-byte frame header: type (u8) + length (u32, big-endian).
constexpr std::size_t kFrameHeaderBytes = 5;

// Standard audio contract: the same 16 kHz mono 16-bit PCM VoiceManager
// already validates on the first frame. AUDIO-frame payloads are raw s16le.
constexpr int kAudioSampleRate = 16000;
constexpr int kAudioChannels = 1;
constexpr const char * kAudioFormat = "s16le";

// TTS output-audio contract: TTS_AUDIO payloads are raw s16le mono at the
// model's vocoder rate. The DEFAULT rate, and the rate of every entry that does not
// override it (dots, voxcpm2); READY advertises the selected descriptor's
// TaskDescriptor::sample_rate (breeze is 24000).
constexpr int kTtsAudioSampleRate = 48000;

// Hint mirroring StreamingPolicy::preferred_audio_chunk_samples (session default 512).
constexpr int kPreferredChunkSamples = 512;

// Message type tags. See PROTOCOL.md for the payload of each. The 0x0A-0x0C TTS set is
// additive under protocol 1: a TTS model_task in HELLO implies it, and the
// 0x01-0x09 ASR set is byte-identical to what shipped before.
enum class MsgType : std::uint8_t {
    Hello = 0x01,  // C->S  JSON: { protocol, model_task }
    Ready = 0x02,  // S->C  JSON: handshake ack
    Audio = 0x03,  // C->S  raw s16le PCM chunk            (drive loop)
    Event = 0x04,  // S->C  JSON: interim StreamEvent      (drive loop)
    Flush = 0x05,  // C->S  JSON: request finalize         (drive loop)
    Final = 0x06,  // S->C  JSON: utterance-final result   (drive loop)
    Reset = 0x07,  // C->S  JSON: reset session state      (drive loop)
    Error = 0x08,  // S->C  JSON: { ok:false, code, message }
    Bye = 0x09,    // C->S  JSON: graceful close
    Text = 0x0A,     // C->S  JSON: { text, voice_wav, ref_text } synthesis request
    TtsAudio = 0x0B, // S->C  raw s16le mono 48 kHz PCM chunk
    TtsDone = 0x0C,  // S->C  JSON: end-of-request marker, after the last audio byte
    Cancel = 0x0D,   // C->S  JSON: stop the in-flight TTS request
};

}  // namespace sidecar
