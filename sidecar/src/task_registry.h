// The sidecar's model/task registry.
//
// Maps each wire `model_task` id to the generic descriptor the drive loop hands
// to audio.cpp's own registry: registry.load({model_path, family_hint = family}) then
// create_task_session({VoiceTaskKind, RunMode}, options). The fields here are the
// lowercase spellings of audio.cpp's taxonomy so no concrete session class name ever
// appears on the wire or in the loop. Five entries ship: nemotron (the first cut),
// voxtral (the second model), dots (the first TTS entry), voxcpm2 (the second TTS
// entry), and breeze (the third TTS entry).
#pragma once

#include "protocol.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

namespace sidecar {

// Opaque string option maps, the shape SessionOptions.options and TaskRequest.options
// take in the engine. The sidecar merges and forwards them; it never interprets a key.
using OptionMap = std::unordered_map<std::string, std::string>;

// `overrides` overlaid on `defaults`, overrides winning on a shared key.
OptionMap merge_options(const OptionMap & defaults, const OptionMap & overrides);

struct TaskDescriptor {
    std::string id;          // wire model_task id, e.g. "nemotron-3.5-asr-streaming"
    std::string family;      // audio.cpp family_hint, e.g. "nemotron_asr"
    std::string task;        // VoiceTaskKind::Asr        -> "asr"
    std::string mode;        // RunMode::Streaming        -> "streaming"
    std::string input_kind;  // StreamingInputKind::AudioChunks -> "audio_chunks"
    // model_path is resolved at session start from sidecar config, not baked in
    // here, so no machine-specific path is committed. Empty until then.
    std::string model_path;
    // Per-model defaults, overridable by the sidecar CLI. The drive loop
    // and resolver forward them opaquely - per-model behavior lives HERE, never in code.
    OptionMap session_options;  // -> SessionOptions.options at session creation
    OptionMap request_options;  // -> TaskRequest.options at every start_stream/re-arm
    // Per-model ref_text requirement: dots requires a non-empty reference transcript
    // on every TEXT frame; voxcpm2 does not (it ignores the transcript unless
    // continuation audio is also supplied). The TTS drive path enforces this flag -
    // never a family string comparison.
    bool ref_text_required = true;
    // Output sample rate of this model's TTS_AUDIO payloads, advertised in READY.
    // Defaults to the 48 kHz constant so the dots and voxcpm2 entries are unchanged;
    // breeze overrides it to 24000. Meaningless for ASR entries (READY states the
    // capture-rate input contract there).
    int sample_rate = kTtsAudioSampleRate;
};

// The READY handshake ack for `task` as a JSON payload. For a TTS model_task `audio`
// is the OUTPUT contract at task.sample_rate and the input-side chunk hint is absent;
// the ASR ack states the capture-rate input contract and is byte-identical to what
// shipped. Lives beside the registry so a test can construct READY per descriptor
// without the server main.
std::string build_ready(const TaskDescriptor & task);

class TaskRegistry {
public:
    // The shipped registry: nemotron-3.5-asr-streaming (no default options),
    // voxtral-mini-4b-realtime-asr-streaming (arena/context recipe pinned as its
    // session-option defaults), dots-tts-streaming (upstream long-form guards
    // text_chunk_size=200 / max_tokens=500 pinned as its request-option defaults), and
    // voxcpm2-tts-streaming (retry_badcase=false pinned as its sole request-option
    // default, ref_text optional), and breeze-tts-streaming (text_chunk_size=200 /
    // guidance_scale=1.0 pinned as its request-option defaults, ref_text required,
    // 24 kHz output), and qwen3-tts-streaming (no pinned options, ref_text required,
    // 24 kHz output). Adding a model is one more entry here plus its payload - the
    // drive loop does not change.
    static TaskRegistry default_registry();

    // Returns the descriptor for `id`, or nullptr if `id` is not registered.
    const TaskDescriptor * find(const std::string & id) const;

    std::vector<std::string> ids() const;
    std::size_t size() const noexcept { return entries_.size(); }

private:
    std::vector<TaskDescriptor> entries_;
};

}  // namespace sidecar
