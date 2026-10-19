// The registry contract for the breeze entry and the per-descriptor READY sample
// rate. No model, no GPU, no engine libs - the registry and the READY builder are
// pure sidecar code.
//
// What must hold (GREEN):
//   - exactly six ids: the four earlier entries, breeze-tts-streaming and
//     qwen3-tts-streaming
//   - breeze descriptor: family breeze_tts, task tts, mode streaming, input_kind none,
//     ref_text_required true, sample_rate 24000, request options EXACTLY
//     text_chunk_size=200 and guidance_scale=1.0, no session options
//   - qwen3 descriptor: family qwen3_tts, task tts, mode streaming, input_kind none,
//     ref_text_required true, sample_rate 24000, no request or session options
//   - READY reports the descriptor's rate: 48000 for dots and voxcpm2, 24000 for breeze
//     and qwen3
//   - the default sample_rate on a descriptor is the 48000 constant
#include "protocol.h"
#include "task_registry.h"

#include "cJSON.h"

#include <algorithm>
#include <cstdio>
#include <string>

extern int g_failures;  // shared with the rearm test's CHECK counter (one binary)

namespace {

#define CHECK(cond, what)                                                          \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (what));  \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

// audio.sample_rate out of a READY payload, or -1 when absent/malformed.
int ready_sample_rate(const std::string & payload) {
    cJSON * root = cJSON_Parse(payload.c_str());
    if (root == nullptr) {
        return -1;
    }
    const cJSON * audio = cJSON_GetObjectItemCaseSensitive(root, "audio");
    const cJSON * rate =
        (audio != nullptr) ? cJSON_GetObjectItemCaseSensitive(audio, "sample_rate") : nullptr;
    const int out = (rate != nullptr && cJSON_IsNumber(rate)) ? rate->valueint : -1;
    cJSON_Delete(root);
    return out;
}

void test_six_ids() {
    const sidecar::TaskRegistry registry = sidecar::TaskRegistry::default_registry();
    CHECK(registry.size() == 6, "the registry must list exactly six model_tasks");
    const auto ids = registry.ids();
    for (const char * id : {"nemotron-3.5-asr-streaming", "voxtral-mini-4b-realtime-asr-streaming",
                            "dots-tts-streaming", "voxcpm2-tts-streaming", "breeze-tts-streaming",
                            "qwen3-tts-streaming"}) {
        CHECK(std::find(ids.begin(), ids.end(), id) != ids.end(), id);
    }
}

void test_breeze_descriptor() {
    const sidecar::TaskRegistry registry = sidecar::TaskRegistry::default_registry();
    const sidecar::TaskDescriptor * breeze = registry.find("breeze-tts-streaming");
    CHECK(breeze != nullptr, "the breeze model_task must resolve");
    if (breeze == nullptr) {
        return;
    }
    CHECK(breeze->family == "breeze_tts", "breeze family_hint");
    CHECK(breeze->task == "tts" && breeze->mode == "streaming" && breeze->input_kind == "none",
          "breeze taxonomy: streaming TTS, no streaming input feed");
    CHECK(breeze->ref_text_required, "breeze requires a reference transcript");
    CHECK(breeze->sample_rate == 24000, "breeze output rate is 24 kHz");
    const sidecar::OptionMap expected{{"text_chunk_size", "200"}, {"guidance_scale", "1.0"}};
    CHECK(breeze->request_options == expected,
          "breeze request options are EXACTLY text_chunk_size=200 and guidance_scale=1.0");
    CHECK(breeze->session_options.empty(), "breeze pins no session options");
}

void test_qwen3_descriptor() {
    const sidecar::TaskRegistry registry = sidecar::TaskRegistry::default_registry();
    const sidecar::TaskDescriptor * qwen3 = registry.find("qwen3-tts-streaming");
    CHECK(qwen3 != nullptr, "the qwen3 model_task must resolve");
    if (qwen3 == nullptr) {
        return;
    }
    CHECK(qwen3->family == "qwen3_tts", "qwen3 family_hint");
    CHECK(qwen3->task == "tts" && qwen3->mode == "streaming" && qwen3->input_kind == "none",
          "qwen3 taxonomy: streaming TTS, no streaming input feed");
    CHECK(qwen3->ref_text_required, "qwen3 Base clones from the reference clip plus its transcript");
    CHECK(qwen3->sample_rate == 24000, "qwen3 output rate is 24 kHz");
    CHECK(qwen3->request_options.empty(), "qwen3 pins no request options (engine defaults)");
    CHECK(qwen3->session_options.empty(), "qwen3 pins no session options");
}

void test_voxcpm2_chunk_strategy_pin() {
    // v0.7.3 reads voxcpm2.chunk_strategy from the REQUEST options (session.cpp
    // parse_chunk_strategy(request.options)) and defaults to continuation, which carries
    // the previous chunk's audio into the next prompt. release-0.6 had no such option and
    // synthesised every chunk from the reference prompt alone, so the entry pins stateless
    // as a request-option default to keep long-form chunking as it shipped.
    const sidecar::TaskRegistry registry = sidecar::TaskRegistry::default_registry();
    const sidecar::TaskDescriptor * voxcpm2 = registry.find("voxcpm2-tts-streaming");
    CHECK(voxcpm2 != nullptr, "the voxcpm2 model_task must resolve");
    if (voxcpm2 == nullptr) {
        return;
    }
    const auto it = voxcpm2->request_options.find("voxcpm2.chunk_strategy");
    CHECK(it != voxcpm2->request_options.end() && it->second == "stateless",
          "voxcpm2 request options pin voxcpm2.chunk_strategy=stateless");
    CHECK(voxcpm2->session_options.find("voxcpm2.chunk_strategy") == voxcpm2->session_options.end(),
          "chunk_strategy is a request option in the engine, not a session option");
    CHECK(voxcpm2->request_options.at("retry_badcase") == "false",
          "the retry_badcase=false pin is kept alongside the chunk strategy");
}

void test_default_sample_rate_is_48k() {
    const sidecar::TaskDescriptor blank;
    CHECK(blank.sample_rate == sidecar::kTtsAudioSampleRate && blank.sample_rate == 48000,
          "a descriptor's default sample_rate is the 48000 constant");
    const sidecar::TaskRegistry registry = sidecar::TaskRegistry::default_registry();
    const sidecar::TaskDescriptor * dots = registry.find("dots-tts-streaming");
    const sidecar::TaskDescriptor * voxcpm2 = registry.find("voxcpm2-tts-streaming");
    CHECK(dots != nullptr && dots->sample_rate == 48000, "dots stays at 48000");
    CHECK(voxcpm2 != nullptr && voxcpm2->sample_rate == 48000, "voxcpm2 stays at 48000");
}

void test_ready_reports_descriptor_rate() {
    const sidecar::TaskRegistry registry = sidecar::TaskRegistry::default_registry();
    struct Case { const char * id; int rate; };
    for (const Case & c : {Case{"dots-tts-streaming", 48000}, Case{"voxcpm2-tts-streaming", 48000},
                           Case{"breeze-tts-streaming", 24000}, Case{"qwen3-tts-streaming", 24000}}) {
        const sidecar::TaskDescriptor * task = registry.find(c.id);
        CHECK(task != nullptr, c.id);
        if (task == nullptr) {
            continue;
        }
        const std::string ready = sidecar::build_ready(*task);
        CHECK(ready_sample_rate(ready) == c.rate, c.id);
    }
    // The ASR ack is untouched: input contract at the 16 kHz capture rate.
    const sidecar::TaskDescriptor * asr = registry.find("nemotron-3.5-asr-streaming");
    CHECK(asr != nullptr && ready_sample_rate(sidecar::build_ready(*asr)) == sidecar::kAudioSampleRate,
          "ASR READY keeps the capture-rate audio contract");
}

}  // namespace

int run_task_registry_tests() {
    test_six_ids();
    test_breeze_descriptor();
    test_qwen3_descriptor();
    test_voxcpm2_chunk_strategy_pin();
    test_default_sample_rate_is_48k();
    test_ready_reports_descriptor_rate();
    return g_failures;
}
