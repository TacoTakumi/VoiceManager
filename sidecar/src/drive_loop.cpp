#include "drive_loop.h"

#include "frame.h"
#include "protocol.h"

#include "cJSON.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sidecar {

namespace {

namespace rt = engine::runtime;

void log_line(const char * level, const std::string & msg) {
    std::fprintf(stderr, "[sidecar] %s: %s\n", level, msg.c_str());
}

std::string json_to_string_and_free(cJSON * obj) {
    char * printed = cJSON_PrintUnformatted(obj);
    std::string out = (printed != nullptr) ? printed : "{}";
    cJSON_free(printed);
    cJSON_Delete(obj);
    return out;
}

std::string build_error(const std::string & code, const std::string & message) {
    cJSON * root = cJSON_CreateObject();
    cJSON_AddBoolToObject(root, "ok", false);
    cJSON_AddStringToObject(root, "code", code.c_str());
    cJSON_AddStringToObject(root, "message", message.c_str());
    return json_to_string_and_free(root);
}

// { text, language } from a Transcript, matching PROTOCOL.md's partial_text/text_output.
cJSON * transcript_to_json(const rt::Transcript & transcript) {
    cJSON * obj = cJSON_CreateObject();
    cJSON_AddStringToObject(obj, "text", transcript.text.c_str());
    cJSON_AddStringToObject(obj, "language", transcript.language.c_str());
    return obj;
}

// [{ start_sample, end_sample, word, confidence }, ...] from WordTimestamp, mapping the
// same-named StreamEvent/TaskResult field.
cJSON * word_timestamps_to_json(const std::vector<rt::WordTimestamp> & words) {
    cJSON * arr = cJSON_CreateArray();
    for (const auto & word : words) {
        cJSON * item = cJSON_CreateObject();
        cJSON_AddNumberToObject(item, "start_sample", static_cast<double>(word.span.start_sample));
        cJSON_AddNumberToObject(item, "end_sample", static_cast<double>(word.span.end_sample));
        cJSON_AddStringToObject(item, "word", word.word.c_str());
        cJSON_AddNumberToObject(item, "confidence", word.confidence);
        cJSON_AddItemToArray(arr, item);
    }
    return arr;
}

// True if the StreamEvent carries anything worth forwarding: an empty accumulate-only
// event is not sent, and neither is a bare end-of-stream MARKER (is_final set with no
// text or word content) - the FINAL frame written after finalize() returns is the
// single utterance-final on this wire.
bool has_content(const rt::StreamEvent & event) {
    return event.partial_text.has_value() ||
           !event.voice_activity.empty() ||
           !event.word_timestamps.empty();
}

std::string build_event(const rt::StreamEvent & event) {
    cJSON * root = cJSON_CreateObject();
    // The model's real flag: a text-carrying final event reaches the
    // client as is_final=true instead of masquerading as one more interim delta.
    cJSON_AddBoolToObject(root, "is_final", event.is_final);
    if (event.partial_text.has_value()) {
        cJSON_AddItemToObject(root, "partial_text", transcript_to_json(*event.partial_text));
    }
    cJSON_AddItemToObject(root, "voice_activity", cJSON_CreateArray());
    cJSON_AddItemToObject(root, "word_timestamps", word_timestamps_to_json(event.word_timestamps));
    return json_to_string_and_free(root);
}

std::string build_final(const rt::TaskResult & result) {
    cJSON * root = cJSON_CreateObject();
    cJSON_AddBoolToObject(root, "is_final", true);
    if (result.text_output.has_value()) {
        cJSON_AddItemToObject(root, "text_output", transcript_to_json(*result.text_output));
    }
    cJSON_AddItemToObject(root, "word_timestamps", word_timestamps_to_json(result.word_timestamps));
    return json_to_string_and_free(root);
}

// Decode an s16le PCM payload into float samples in [-1, 1), the AudioBuffer convention
// (audio.cpp scales s16 by 1/32768; see wav_reader.cpp). A payload with a trailing odd
// byte is malformed and rejected by the caller before this runs.
std::vector<float> s16le_to_float(const std::string & payload) {
    const std::size_t count = payload.size() / 2;
    std::vector<float> samples;
    samples.reserve(count);
    const auto * bytes = reinterpret_cast<const unsigned char *>(payload.data());
    for (std::size_t i = 0; i < count; ++i) {
        const std::int16_t s =
            static_cast<std::int16_t>(bytes[2 * i] | (bytes[2 * i + 1] << 8));
        samples.push_back(static_cast<float>(s) / 32768.0f);
    }
    return samples;
}

// The streaming audio contract: standard 16 kHz mono. max_input_samples is a
// generous upper bound - the streaming session only traces it, since capacity is sized
// from the model's lookahead, not the (unknown-ahead-of-time) utterance length.
rt::SessionPreparationRequest streaming_preparation() {
    rt::SessionPreparationRequest prep;
    prep.audio = rt::AudioPreparationContract{
        kAudioSampleRate,
        kAudioChannels,
        static_cast<std::int64_t>(kAudioSampleRate) * 600,  // 10 min ceiling (hint)
    };
    return prep;
}

}  // namespace

void drive_streaming_session(int fd, rt::IStreamingVoiceTaskSession & session,
                             const OptionMap & request_options) {
    // Interim deltas are pushed by the session (during process_audio_chunk and finalize,
    // for the ASR streaming model) through this sink; forward each as an EVENT frame.
    // Installed for the life of the connection and cleared before returning so the
    // captured fd never dangles.
    //
    // NOTHING MAY ESCAPE THIS LAMBDA. write_frame throws
    // std::runtime_error on a write failure, and this sink is invoked from INSIDE
    // session.process_audio_chunk() and session.finalize(): for the shipped ASR streaming
    // model, from a callback threaded through its decode loop. So a client that
    // disconnects mid-decode turns an EPIPE into an exception unwinding through
    // audio.cpp's own frames, mid-decode, past whatever ggml/CUDA state the decode is
    // holding. An earlier audit of the finalize-only decode path found no `noexcept`
    // annotation; the process_audio_chunk path has not been audited the same way, so a
    // std::terminate from an escaping exception cannot be ruled out there, and unwinding
    // mid-decode is a bad bet regardless. Catching here is four lines and removes the
    // whole class.
    //
    // Dropping the frame is the correct recovery: interim EVENTs are best-effort, and a
    // dead socket is discovered properly by the next read_frame()/write_frame() in the
    // loop below, which ends the connection through the normal path. The flag is a
    // shared_ptr rather than a captured reference because the sink outlives this stack
    // frame if anything ever escapes the outer catch.
    auto sink_write_failed = std::make_shared<bool>(false);
    session.set_stream_event_sink([fd, sink_write_failed](const rt::StreamEvent & event) {
        if (*sink_write_failed || !has_content(event)) {
            return;
        }
        try {
            write_frame(fd, MsgType::Event, build_event(event));
        } catch (const std::exception & ex) {
            *sink_write_failed = true;   // log once, then stay quiet for this connection
            log_line("warn", std::string("dropping interim EVENT, peer write failed: ")
                                 + ex.what());
        }
    });

    try {
        session.prepare(streaming_preparation());
        // One request for the connection, reused by every per-utterance re-arm below so
        // the caller-merged option map stays identical across utterances. The map
        // is opaque here: no key is read, no model family consulted.
        rt::TaskRequest stream_request{};
        stream_request.options = request_options;
        session.start_stream(stream_request);  // begin the first utterance (resets state)

        std::int64_t start_sample = 0;

        // Re-arm the stream for the next utterance. Session types with a
        // started-guard clear their started state on finalize() and
        // reset(), so without this the second utterance on a connection throws; for
        // sessions without the guard start_stream() is a reset-equivalent, the same
        // turnover a RESET already performed. A re-arm failure is reported and leaves the stream
        // unarmed - the next FLUSH then yields an ERROR (finalize's started-guard), and
        // a later RESET retries the re-arm; the connection itself stays alive.
        const auto rearm_stream = [&]() {
            try {
                session.start_stream(stream_request);
                start_sample = 0;
            } catch (const std::exception & ex) {
                write_frame(fd, MsgType::Error,
                            build_error("internal",
                                        std::string("start_stream failed: ") + ex.what()));
            }
        };

        Frame frame;
        while (read_frame(fd, frame)) {
            switch (frame.type) {
            case MsgType::Audio: {
                if (frame.payload.size() % 2 != 0) {
                    write_frame(fd, MsgType::Error,
                                build_error("bad_frame", "AUDIO payload is not whole s16le samples"));
                    break;
                }
                rt::AudioChunk chunk;
                chunk.sample_rate = kAudioSampleRate;
                chunk.channels = kAudioChannels;
                chunk.start_sample = start_sample;
                chunk.samples = s16le_to_float(frame.payload);
                start_sample += static_cast<std::int64_t>(chunk.samples.size());
                // Some models return partials from process_audio_chunk directly; forward
                // them. The ASR streaming model encodes each chunk here and pushes its
                // deltas through the sink during speech; finalize() emits only the tail.
                const rt::StreamEvent event = session.process_audio_chunk(chunk);
                if (has_content(event)) {
                    write_frame(fd, MsgType::Event, build_event(event));
                }
                break;
            }
            case MsgType::Flush: {
                // The client's endpointer decided to finalize. finalize() drives
                // the decode, emitting interim deltas via the sink, then returns the
                // utterance-final result: exactly one FINAL per FLUSH.
                try {
                    const rt::TaskResult result = session.finalize();
                    write_frame(fd, MsgType::Final, build_final(result));
                    rearm_stream();  // successful finalize ended the stream; arm the next
                } catch (const std::exception & ex) {
                    // A failed finalize did not consume the started state; do not re-arm.
                    write_frame(fd, MsgType::Error,
                                build_error("internal", std::string("finalize failed: ") + ex.what()));
                }
                break;
            }
            case MsgType::Reset:
                session.reset();  // discards the utterance and clears any started state...
                start_sample = 0;
                rearm_stream();   // ...so arm the next utterance
                break;
            case MsgType::Bye:
                session.set_stream_event_sink(nullptr);
                return;
            default:
                write_frame(fd, MsgType::Error,
                            build_error("bad_frame", "unexpected frame type in drive loop"));
                break;
            }
        }
    } catch (const std::exception & ex) {
        // An I/O failure (peer close mid-reply) or a recoverable session error ends the
        // connection here; a process-fatal fault is caught one level up by the fatal-signal
        // handler, which emits ERROR `sidecar_fault` before the process dies.
        log_line("warn", std::string("drive loop ended: ") + ex.what());
    }

    session.set_stream_event_sink(nullptr);
}

}  // namespace sidecar
