#include "task_registry.h"

#include "cJSON.h"

namespace sidecar {

OptionMap merge_options(const OptionMap & defaults, const OptionMap & overrides) {
    OptionMap merged = defaults;
    for (const auto & [key, value] : overrides) {
        merged[key] = value;
    }
    return merged;
}

TaskRegistry TaskRegistry::default_registry() {
    TaskRegistry registry;
    // Nemotron-3.5 streaming ASR, the original first-cut entry.
    // No default options: the shipped nemotron behavior is exactly the engine defaults.
    registry.entries_.push_back(TaskDescriptor{
        /*id=*/"nemotron-3.5-asr-streaming",
        /*family=*/"nemotron_asr",
        /*task=*/"asr",
        /*mode=*/"streaming",
        /*input_kind=*/"audio_chunks",
        /*model_path=*/"",
        /*session_options=*/{},
        /*request_options=*/{},
    });
    // Voxtral-Mini-4B-Realtime streaming ASR, the second model. The
    // session options pin the arena/context recipe from the measurement spike: every
    // voxtral_realtime.* knob should stay at its engine default (arenas 512/512/512 MB,
    // weight contexts 128/128 MB, decode cache 1024 steps ~ the full ~82 s context;
    // peak 3,579 MiB/session on this rig). Pinned explicitly HERE - where per-model
    // behavior lives - so the measured recipe holds even if engine defaults move.
    registry.entries_.push_back(TaskDescriptor{
        /*id=*/"voxtral-mini-4b-realtime-asr-streaming",
        /*family=*/"voxtral_realtime",
        /*task=*/"asr",
        /*mode=*/"streaming",
        /*input_kind=*/"audio_chunks",
        /*model_path=*/"",
        /*session_options=*/{
            {"voxtral_realtime.audio_encoder_graph_arena_mb", "512"},
            {"voxtral_realtime.audio_encoder_weight_context_mb", "128"},
            {"voxtral_realtime.text_decoder_prefill_graph_arena_mb", "512"},
            {"voxtral_realtime.text_decoder_decode_graph_arena_mb", "512"},
            {"voxtral_realtime.text_decoder_weight_context_mb", "128"},
            {"voxtral_realtime.stream_decode_cache_steps", "1024"},
        },
        /*request_options=*/{},
    });
    // Dots-TTS streaming synthesis, the first TTS entry. input_kind=none:
    // the request text arrives with start_stream, there is no streaming input feed. The
    // request-option defaults pin the upstream long-form guards (audio.cpp's dots_tts
    // guidance): text_chunk_size=200 keeps long text chunked below the model's drift
    // range, max_tokens=500 bounds each chunk's token budget. text_chunk_mode=default
    // replaces dots' tag_aware built-in: over-budget sentences backtrack
    // to clause breaks instead of hard word splits (v1 sends no inline [tags], so tag
    // awareness buys nothing). Pinned HERE - where per-model behavior lives - so the
    // guards hold even if engine defaults move.
    registry.entries_.push_back(TaskDescriptor{
        /*id=*/"dots-tts-streaming",
        /*family=*/"dots_tts",
        /*task=*/"tts",
        /*mode=*/"streaming",
        /*input_kind=*/"none",
        /*model_path=*/"",
        /*session_options=*/{},
        /*request_options=*/{
            {"text_chunk_size", "200"},
            {"max_tokens", "500"},
            {"text_chunk_mode", "default"},
        },
        /*ref_text_required=*/true,
    });
    // VoxCPM2 streaming synthesis, the second TTS entry. input_kind=none like dots.
    // Two pinned request-option defaults:
    // retry_badcase=false - generate_streaming THROWS without it (generator.cpp
    // guards "streaming generation requires retry_badcase=false") - and
    // voxcpm2.chunk_strategy=stateless. The engine reads chunk_strategy from the
    // REQUEST options (voxcpm2/session.cpp parse_chunk_strategy(request.options)) and
    // its default, continuation, carries each chunk's audio into the next prompt;
    // release-0.6 had no such option and synthesised every chunk from the reference
    // prompt alone, so stateless keeps long-form chunking as it shipped. No
    // text_chunk_size/max_tokens pins: voxcpm2's own defaults (2048/4096) are trusted,
    // and a chunk retune is considered only if the live listen shows seam
    // problems. ref_text_required=false makes the reference transcript optional
    // for this model - voxcpm2 ignores prompt_text unless continuation audio is also
    // supplied.
    registry.entries_.push_back(TaskDescriptor{
        /*id=*/"voxcpm2-tts-streaming",
        /*family=*/"voxcpm2",
        /*task=*/"tts",
        /*mode=*/"streaming",
        /*input_kind=*/"none",
        /*model_path=*/"",
        /*session_options=*/{},
        /*request_options=*/{
            {"retry_badcase", "false"},
            {"voxcpm2.chunk_strategy", "stateless"},
        },
        /*ref_text_required=*/false,
    });
    // Breeze-TTS 2 streaming synthesis, the third TTS entry. input_kind=none like dots
    // and voxcpm2. Request-option defaults pin
    // upstream's documented streaming recipe: text_chunk_size=200 keeps long text below
    // the model's drift range and guidance_scale=1.0 is the documented CFG default,
    // written down HERE so the recipe holds even if the dev engine's defaults move. No
    // other request or session option is set. ref_text_required=true: breeze clones from
    // a reference clip plus its transcript and rejects an empty one. sample_rate=24000:
    // breeze's vocoder emits 24 kHz (speech_decoder.cpp kSampleRate) and READY
    // advertises it.
    registry.entries_.push_back(TaskDescriptor{
        /*id=*/"breeze-tts-streaming",
        /*family=*/"breeze_tts",
        /*task=*/"tts",
        /*mode=*/"streaming",
        /*input_kind=*/"none",
        /*model_path=*/"",
        /*session_options=*/{},
        /*request_options=*/{
            {"text_chunk_size", "200"},
            {"guidance_scale", "1.0"},
        },
        /*ref_text_required=*/true,
        /*sample_rate=*/24000,
    });
    // Qwen3-TTS 1.7B Base streaming synthesis, the fourth TTS entry. input_kind=none like
    // the other TTS entries. No request or session options are pinned: a one-minute
    // long-form run on the engine defaults streamed at RTF 0.29 with no playback stalls,
    // so there is nothing to tune yet. ref_text_required=true: Base clones from the
    // reference clip plus its transcript by default (x_vector_only_mode=false).
    // sample_rate=24000: the 12 Hz speech decoder emits 24 kHz.
    registry.entries_.push_back(TaskDescriptor{
        /*id=*/"qwen3-tts-streaming",
        /*family=*/"qwen3_tts",
        /*task=*/"tts",
        /*mode=*/"streaming",
        /*input_kind=*/"none",
        /*model_path=*/"",
        /*session_options=*/{},
        /*request_options=*/{},
        /*ref_text_required=*/true,
        /*sample_rate=*/24000,
    });
    return registry;
}

std::string build_ready(const TaskDescriptor & task) {
    // The task kind selects the audio contract the ack states: for a TTS
    // model_task `audio` is the OUTPUT contract (TTS_AUDIO payloads at the descriptor's
    // sample_rate) and the input-side chunk hint is absent; the ASR ack is
    // byte-identical to what shipped.
    const bool is_tts = (task.task == "tts");
    cJSON * root = cJSON_CreateObject();
    cJSON_AddBoolToObject(root, "ok", true);
    cJSON_AddNumberToObject(root, "protocol", kProtocolVersion);
    cJSON_AddStringToObject(root, "model_task", task.id.c_str());
    cJSON_AddStringToObject(root, "family", task.family.c_str());
    cJSON_AddStringToObject(root, "task", task.task.c_str());
    cJSON_AddStringToObject(root, "mode", task.mode.c_str());
    cJSON_AddStringToObject(root, "input", task.input_kind.c_str());
    cJSON * audio = cJSON_AddObjectToObject(root, "audio");
    cJSON_AddNumberToObject(audio, "sample_rate", is_tts ? task.sample_rate : kAudioSampleRate);
    cJSON_AddNumberToObject(audio, "channels", kAudioChannels);
    cJSON_AddStringToObject(audio, "format", kAudioFormat);
    if (!is_tts) {
        cJSON_AddNumberToObject(root, "preferred_chunk_samples", kPreferredChunkSamples);
    }
    char * printed = cJSON_PrintUnformatted(root);
    std::string out = (printed != nullptr) ? printed : "{}";
    cJSON_free(printed);
    cJSON_Delete(root);
    return out;
}

const TaskDescriptor * TaskRegistry::find(const std::string & id) const {
    for (const auto & entry : entries_) {
        if (entry.id == id) {
            return &entry;
        }
    }
    return nullptr;
}

std::vector<std::string> TaskRegistry::ids() const {
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const auto & entry : entries_) {
        out.push_back(entry.id);
    }
    return out;
}

}  // namespace sidecar
