// Sink-throw cancel probe - is the streaming session reusable after a sentinel unwind?
//
// This is a MANUAL probe, not part of any automated suite. It needs a real GGUF and a
// CUDA GPU. It answers one empirical question, recorded for the cancel-mechanism choice:
//
//   1. start_stream a long generation with a stream-event sink that THROWS a sentinel
//      exception after the Nth audio event - measure how quickly the throw unwinds out
//      of start_stream.
//   2. On the SAME session, run a fresh start_stream with a normal sink - does it
//      complete correctly? The post-cancel audio is written to a WAV for a listening
//      check (corruption would be audible).
//
// Either technical outcome is a pass; what matters is that the verdict is recorded.
// If the sentinel does NOT propagate (the engine swallows it) or the process dies,
// that is a "not reusable" verdict and the kill-and-respawn cancel is used instead.
//
// The selected model_task determines the request options (the registry's defaults,
// e.g. retry_badcase=false for voxcpm2) exactly like the sidecar merges them, so the
// same probe runs for dots and voxcpm2.
//
// Usage (voxcpm2):
//   sidecar/build-0.6/bin/sink_throw_probe \
//       --model-task voxcpm2-tts-streaming \
//       --model /home/user/audiocpp-models/VoxCPM2-GGUF/voxcpm2-q8_0.gguf \
//       --backend cuda --device 3 \
//       --voice-wav voices/designed/designed_female_s101.wav \
//       --ref-text "<the preset's exact transcript>" \
//       --out /tmp/voxcpm2_post_cancel.wav
#include "engine_session.h"
#include "task_registry.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/runtime/session.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace rt = engine::runtime;
using Clock = std::chrono::steady_clock;

namespace {

struct SentinelCancel {};  // deliberately NOT std::exception: nothing else throws this

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

const char * kLongText =
    "Long range sensors have completed their sweep of the outer system. "
    "Three vessels are holding station near the fifth planet, and their transponder "
    "codes do not match any registry on file. Shields are at full strength, weapons "
    "are on standby, and all decks report ready for extended operations. "
    "Additional telemetry from the forward array is being processed now.";

const char * kPostCancelText =
    "Cancellation complete. The synthesis pipeline has been re-armed and is "
    "functioning within normal parameters.";

rt::TaskRequest build_request(const std::string & text, const rt::AudioBuffer & reference,
                              const std::string & ref_text,
                              const sidecar::OptionMap & options) {
    rt::TaskRequest request;
    request.text_input = rt::Transcript{text, ""};
    rt::VoiceReference speaker;
    speaker.audio = reference;
    rt::VoiceCondition voice;
    voice.speaker = speaker;
    request.voice = voice;
    // The registry's per-model defaults (text_chunk_size/max_tokens for dots,
    // retry_badcase=false for voxcpm2), merged exactly like the sidecar does.
    request.options = options;
    if (!ref_text.empty()) {
        request.options["reference_text"] = ref_text;
    }
    return request;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string model_path;
    std::string model_task = "dots-tts-streaming";
    std::string backend = "cuda";
    int device = 0;
    std::string voice_wav;
    std::string ref_text;
    std::string out_path = "post_cancel.wav";
    int throw_after_events = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) { model_path = argv[++i]; continue; }
        if (arg == "--model-task" && i + 1 < argc) { model_task = argv[++i]; continue; }
        if (arg == "--backend" && i + 1 < argc) { backend = argv[++i]; continue; }
        if (arg == "--device" && i + 1 < argc) { device = std::atoi(argv[++i]); continue; }
        if (arg == "--voice-wav" && i + 1 < argc) { voice_wav = argv[++i]; continue; }
        if (arg == "--ref-text" && i + 1 < argc) { ref_text = argv[++i]; continue; }
        if (arg == "--out" && i + 1 < argc) { out_path = argv[++i]; continue; }
        if (arg == "--throw-after" && i + 1 < argc) { throw_after_events = std::atoi(argv[++i]); continue; }
        std::fprintf(stderr, "unknown/incomplete argument: %s\n", arg.c_str());
        return 2;
    }
    if (model_path.empty() || voice_wav.empty()) {
        std::fprintf(stderr, "usage: sink_throw_probe --model <gguf> --voice-wav <wav> "
                             "[--model-task dots-tts-streaming|voxcpm2-tts-streaming] "
                             "[--ref-text <transcript> --backend cuda --device N --out out.wav]\n");
        return 2;
    }

    const sidecar::TaskRegistry registry = sidecar::TaskRegistry::default_registry();
    const sidecar::TaskDescriptor * task = registry.find(model_task);
    if (task == nullptr) {
        std::fprintf(stderr, "%s not registered\n", model_task.c_str());
        return 1;
    }
    sidecar::EngineRuntimeConfig config;
    config.model_path = model_path;
    config.backend = backend;
    config.device = device;

    // The per-model request-option defaults ride every request, merged like the
    // sidecar's HELLO path (retry_badcase=false for voxcpm2, the 200/500 guards for
    // dots).
    const sidecar::OptionMap request_opts = sidecar::merge_options(task->request_options, {});

    std::printf("[probe] loading %s (backend=%s device=%d)\n", model_path.c_str(),
                backend.c_str(), device);
    const auto load_start = Clock::now();
    sidecar::StreamingSession live = sidecar::open_streaming_session(*task, config);
    live.streaming->prepare(rt::SessionPreparationRequest{});
    std::printf("[probe] session warm in %.1f s\n", ms_since(load_start) / 1000.0);

    const auto wav = engine::audio::read_wav_f32(std::filesystem::path(voice_wav));
    const rt::AudioBuffer reference{wav.sample_rate, wav.channels, wav.samples};

    // --- Round 1: throw a sentinel from the sink mid-generation -----------------------
    int events_seen = 0;
    long samples_before_throw = 0;
    Clock::time_point throw_at{};
    live.streaming->set_stream_event_sink(
        [&](const rt::StreamEvent & event) {
            for (const auto & named : event.named_audio_outputs) {
                samples_before_throw += static_cast<long>(named.audio.samples.size());
            }
            if (++events_seen == throw_after_events) {
                throw_at = Clock::now();
                std::printf("[probe] throwing sentinel from the sink (event %d, %.2f s of "
                            "audio emitted)\n",
                            events_seen, static_cast<double>(samples_before_throw) / 48000.0);
                throw SentinelCancel{};
            }
        });

    const auto gen_start = Clock::now();
    bool sentinel_propagated = false;
    double unwind_ms = -1.0;
    try {
        live.streaming->start_stream(build_request(kLongText, reference, ref_text,
                                                   request_opts));
        std::printf("[probe] start_stream returned NORMALLY - the sentinel was swallowed "
                    "somewhere inside the engine\n");
    } catch (const SentinelCancel &) {
        sentinel_propagated = true;
        unwind_ms = ms_since(throw_at);
        std::printf("[probe] sentinel unwound out of start_stream in %.0f ms "
                    "(%.2f s into generation)\n",
                    unwind_ms, ms_since(gen_start) / 1000.0);
    } catch (const std::exception & ex) {
        std::printf("[probe] start_stream threw a DIFFERENT exception: %s\n", ex.what());
    }

    // --- Round 2: fresh start_stream on the SAME session -------------------------------
    std::vector<float> post_cancel_audio;
    live.streaming->set_stream_event_sink([&](const rt::StreamEvent & event) {
        for (const auto & named : event.named_audio_outputs) {
            post_cancel_audio.insert(post_cancel_audio.end(), named.audio.samples.begin(),
                                     named.audio.samples.end());
        }
    });
    bool reuse_ok = false;
    const auto reuse_start = Clock::now();
    try {
        live.streaming->start_stream(build_request(kPostCancelText, reference, ref_text,
                                                   request_opts));
        reuse_ok = !post_cancel_audio.empty();
        std::printf("[probe] post-cancel start_stream completed in %.2f s, %.2f s of audio\n",
                    ms_since(reuse_start) / 1000.0,
                    static_cast<double>(post_cancel_audio.size()) / 48000.0);
    } catch (const std::exception & ex) {
        std::printf("[probe] post-cancel start_stream FAILED: %s\n", ex.what());
    }
    live.streaming->set_stream_event_sink(nullptr);

    if (!post_cancel_audio.empty()) {
        engine::audio::write_pcm16_wav(std::filesystem::path(out_path), 48000, 1,
                                       post_cancel_audio);
        std::printf("[probe] post-cancel audio written to %s - LISTEN for corruption\n",
                    out_path.c_str());
    }

    std::printf("\n[probe] VERDICT: sentinel_propagated=%s unwind_ms=%.0f reuse_ok=%s\n",
                sentinel_propagated ? "yes" : "no", unwind_ms, reuse_ok ? "yes" : "no");
    return (sentinel_propagated && reuse_ok) ? 0 : 3;
}
