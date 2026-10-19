#include "tts_drive.h"

#include "frame.h"
#include "protocol.h"

#include "cJSON.h"

#include <poll.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
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

// Encode float samples in [-1, 1) to raw s16le bytes - the exact inverse of the
// engine's s16 -> float convention (scale by 32768, see wav_reader.cpp), clamped so a
// full-scale +1.0 cannot wrap.
std::string float_to_s16le(const std::vector<float> & samples) {
    std::string out;
    out.resize(samples.size() * 2);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        long v = std::lrintf(samples[i] * 32768.0f);
        if (v > 32767) {
            v = 32767;
        } else if (v < -32768) {
            v = -32768;
        }
        const auto s = static_cast<std::int16_t>(v);
        out[2 * i] = static_cast<char>(s & 0xFF);
        out[2 * i + 1] = static_cast<char>((s >> 8) & 0xFF);
    }
    return out;
}

// Caps the silence run that spans a dots segment boundary. The session
// names every sink buffer segment_<i>_chunk_<n> (the vocoder-tail "flush" buffer has
// no segment), so the seam is visible here: a run of below-threshold samples whose
// bounding voiced samples belong to DIFFERENT segments is a seam gap and is capped;
// a run inside one segment (an in-chunk pause, the request tail) is emitted whole.
// Trailing silence is held back until its fate is known - holding silence never
// delays a voiced sample, so streaming latency is untouched. Used on the worker
// (sink) thread during generation and on the connection thread after join; the join
// is the synchronization point.
class SeamSilenceTrimmer {
public:
    explicit SeamSilenceTrimmer(const TtsDriveConfig & config)
        : cap_(config.seam_silence_cap_samples), threshold_(config.seam_silence_threshold) {}

    bool enabled() const { return cap_ > 0; }

    // Feed one named buffer; appends every sample due NOW to `out`.
    void feed(const std::string & id, const std::vector<float> & samples,
              std::vector<float> & out) {
        const std::optional<long long> seg = parse_segment(id);
        for (const float sample : samples) {
            if (std::fabs(sample) < threshold_) {
                // Request-leading silence - before ANY voiced sample - can never be
                // a seam: stream it straight through so the client's first frame is
                // never delayed. Only silence after voice is held for a verdict.
                if (!last_voiced_segment_.has_value()) {
                    out.push_back(sample);
                } else {
                    held_.push_back(sample);
                }
                continue;
            }
            release_held(seg, out);
            out.push_back(sample);
            if (seg.has_value()) {
                last_voiced_segment_ = seg;
            }
        }
    }

    // End of request: held silence never precedes another voiced sample, so it is
    // request-tail silence - emitted whole, never treated as a seam.
    void finish(std::vector<float> & out) {
        out.insert(out.end(), held_.begin(), held_.end());
        held_.clear();
    }

private:
    static std::optional<long long> parse_segment(const std::string & id) {
        // "segment_<i>_chunk_<n>"; anything else (the "flush" tail, an unnamed
        // buffer) states no segment and can neither open nor close a seam.
        constexpr const char * kPrefix = "segment_";
        constexpr std::size_t kPrefixLen = 8;
        if (id.compare(0, kPrefixLen, kPrefix) != 0 || id.size() <= kPrefixLen ||
            id[kPrefixLen] < '0' || id[kPrefixLen] > '9') {
            return std::nullopt;
        }
        return std::strtoll(id.c_str() + kPrefixLen, nullptr, 10);
    }

    void release_held(const std::optional<long long> & seg, std::vector<float> & out) {
        if (held_.empty()) {
            return;
        }
        const bool seam = last_voiced_segment_.has_value() && seg.has_value() &&
                          *seg != *last_voiced_segment_;
        const std::size_t keep =
            seam ? std::min(held_.size(), static_cast<std::size_t>(cap_)) : held_.size();
        out.insert(out.end(), held_.begin(),
                   held_.begin() + static_cast<std::ptrdiff_t>(keep));
        held_.clear();
    }

    long long cap_;
    float threshold_;
    std::vector<float> held_;
    std::optional<long long> last_voiced_segment_;
};

// The cancel sentinel (validated by the sink-throw probe): deliberately NOT derived from
// std::exception, so neither the sink's own std::exception guard nor anything inside
// the pinned bb15edd7 engine frames can swallow it. The sink throws it when a wire
// CANCEL has been seen; it unwinds out of start_stream (measured 3 ms, twice) and the
// worker catches it by type - the session stays warm for the next request.
struct CancelSentinel {};

// The three required TEXT fields (PROTOCOL.md). `error` explains the first violation.
struct TextRequest {
    std::string text;
    std::string voice_wav;
    std::string ref_text;
};

bool parse_text_request(const std::string & payload, TextRequest & out, std::string & error,
                       bool ref_text_required) {
    cJSON * root = cJSON_Parse(payload.c_str());
    if (root == nullptr) {
        error = "TEXT payload is not valid JSON";
        return false;
    }
    const cJSON * text = cJSON_GetObjectItemCaseSensitive(root, "text");
    const cJSON * voice_wav = cJSON_GetObjectItemCaseSensitive(root, "voice_wav");
    const cJSON * ref_text = cJSON_GetObjectItemCaseSensitive(root, "ref_text");
    bool ok = false;
    if (!cJSON_IsString(text) || text->valuestring == nullptr) {
        error = "TEXT requires a string 'text'";
    } else if (!cJSON_IsString(voice_wav) || voice_wav->valuestring == nullptr ||
               voice_wav->valuestring[0] == '\0') {
        error = "TEXT requires a non-empty string 'voice_wav'";
    } else if (ref_text_required &&
               (!cJSON_IsString(ref_text) || ref_text->valuestring == nullptr ||
                ref_text->valuestring[0] == '\0')) {
        // ref_text is MANDATORY and non-empty only for models that require it
        // (dots): those models condition on the reference WAV and its exact
        // transcript together. voxcpm2 does not require it - a missing or
        // empty transcript is accepted and simply omitted from the request options.
        error = "TEXT requires a non-empty string 'ref_text' (the reference transcript)";
    } else {
        out.text = text->valuestring;
        out.voice_wav = voice_wav->valuestring;
        out.ref_text = (cJSON_IsString(ref_text) && ref_text->valuestring != nullptr)
                           ? ref_text->valuestring
                           : "";
        ok = true;
    }
    cJSON_Delete(root);
    return ok;
}

// Services ONE TEXT request: dispatches the blocking start_stream on a worker thread
// with the sink forwarding audio to `fd`, then sends TTS_DONE after the last audio
// byte. Returns false only when the connection is known dead (a reply write failed or
// EOF was read mid-request); recoverable request errors are reported as ERROR frames
// and return true. While the request is in flight the drain loop also POLLS the fd:
// a CANCEL frame arms the sink-throw sentinel and the request is closed early
// by TTS_DONE {cancelled:true} with its remaining audio discarded; a BYE read here is
// remembered in `bye_seen` and honoured by the caller AFTER the reply is finished,
// preserving the queued-BYE behavior from before CANCEL.
bool service_text_request(int fd, rt::IStreamingVoiceTaskSession & session,
                          const OptionMap & request_options, const WavLoader & load_wav,
                          const TtsDriveConfig & config, const std::string & payload,
                          bool & bye_seen, bool ref_text_required) {
    TextRequest text_request;
    std::string parse_error;
    if (!parse_text_request(payload, text_request, parse_error, ref_text_required)) {
        write_frame(fd, MsgType::Error, build_error("bad_frame", parse_error));
        return true;
    }

    // The voice reference is loaded per request, before anything is dispatched, so an
    // unreadable WAV is a defined per-request error, never a mid-generation surprise.
    rt::AudioBuffer reference_audio;
    try {
        reference_audio = load_wav(text_request.voice_wav);
    } catch (const std::exception & ex) {
        write_frame(fd, MsgType::Error,
                    build_error("bad_frame", std::string("could not load voice_wav '") +
                                                 text_request.voice_wav + "': " + ex.what()));
        return true;
    }

    rt::TaskRequest request;
    request.text_input = rt::Transcript{text_request.text, ""};
    rt::VoiceReference speaker;
    speaker.audio = std::move(reference_audio);
    rt::VoiceCondition voice;
    voice.speaker = std::move(speaker);
    request.voice = std::move(voice);
    // The caller-merged option map, opaque as ever; the wire ref_text rides the
    // engine's `reference_text` option key. Omitted entirely when the model does
    // not require it and the frame carried no transcript (voxcpm2) - never
    // an empty-string key, so a voxcpm2 request's options stay exactly its pins.
    request.options = request_options;
    if (!text_request.ref_text.empty()) {
        request.options["reference_text"] = text_request.ref_text;
    }

    // The buffer-and-drain pump. The sink - invoked on the worker
    // thread INSIDE start_stream - only converts and appends to this in-process queue,
    // so its push NEVER blocks on the socket and a slow reader can never stall
    // generation. The connection thread below is the single drainer, writing queued
    // payloads to the socket in FIFO order: every byte delivered in order, none
    // dropped. Memory is bounded by the synthesis length, which the session itself
    // already holds in full (its merged result buffer), so the queue adds no new
    // asymptotic cost.
    struct AudioPump {
        std::mutex mutex;
        std::condition_variable cv;
        std::deque<std::string> queue;
        bool generation_done = false;
    };
    AudioPump pump;

    // Armed by a wire CANCEL; checked by the sink on every event so the throw lands
    // between vocoder merge groups (worst case one group, ~0.35 s of audio at the
    // pinned defaults - the bound the sink-throw probe measured).
    std::atomic<bool> cancel_requested{false};

    // NOTHING MAY ESCAPE THIS LAMBDA except the CancelSentinel (the drive_loop.cpp
    // sink rule): the only realistic accidental throw here is an allocation failure,
    // and unwinding that mid-decode through audio.cpp's frames is never worth it -
    // drop the buffer instead. The sentinel is the ONE deliberate exception: the
    // sink-throw probe proved (twice) it unwinds cleanly and the session stays reusable.
    // The seam trimmer sits between the sink and the pump. Disabled
    // (cap <= 0) it is bypassed entirely and the pre-trim path runs bit-identically.
    SeamSilenceTrimmer trimmer(config);

    // ONE forwarder for both output kinds: the
    // push-style sink below and the pull loop in the worker both hand every event
    // here, so a pulled chunk crosses exactly the same trimmer and pump as a pushed
    // one - the wire cannot tell the two session kinds apart.
    auto forward_event = [&pump, &trimmer](const rt::StreamEvent & event) {
        try {
            std::vector<std::string> payloads;
            if (trimmer.enabled()) {
                std::vector<float> out;
                if (event.audio_output.has_value() && !event.audio_output->samples.empty()) {
                    trimmer.feed("", event.audio_output->samples, out);
                }
                for (const auto & named : event.named_audio_outputs) {
                    if (!named.audio.samples.empty()) {
                        trimmer.feed(named.id, named.audio.samples, out);
                    }
                }
                if (!out.empty()) {
                    payloads.push_back(float_to_s16le(out));
                }
            } else {
                if (event.audio_output.has_value() && !event.audio_output->samples.empty()) {
                    payloads.push_back(float_to_s16le(event.audio_output->samples));
                }
                for (const auto & named : event.named_audio_outputs) {
                    if (!named.audio.samples.empty()) {
                        payloads.push_back(float_to_s16le(named.audio.samples));
                    }
                }
            }
            if (payloads.empty()) {
                return;
            }
            {
                std::lock_guard<std::mutex> lock(pump.mutex);
                for (auto & payload : payloads) {
                    pump.queue.push_back(std::move(payload));
                }
            }
            pump.cv.notify_one();
        } catch (const std::exception & ex) {
            log_line("warn", std::string("dropping TTS audio buffer: ") + ex.what());
        }
    };

    session.set_stream_event_sink([&cancel_requested,
                                   &forward_event](const rt::StreamEvent & event) {
        if (cancel_requested.load(std::memory_order_relaxed)) {
            throw CancelSentinel{};
        }
        forward_event(event);
    });

    // The session's declared output kind selects HOW audio leaves it: a
    // FinalResult session pushes through the sink from inside the blocking
    // start_stream; a PullEvents session stages the request in start_stream and hands
    // one chunk per next_stream_event() until nullopt. The kind is read from the
    // streaming policy and nothing else - no family enters this decision.
    const bool pull_events =
        session.streaming_policy().output == rt::StreamingOutputKind::PullEvents;

    // The blocking generation on a worker thread. `worker_error` and
    // `worker_cancelled` are written by the worker and read only after join. The
    // sentinel is caught by type FIRST: a cancelled generation is a clean outcome,
    // never an error.
    std::string worker_error;
    bool worker_cancelled = false;
    std::thread worker([&session, &request, &worker_error, &worker_cancelled, &pump,
                        pull_events, &cancel_requested, &forward_event]() {
        try {
            session.start_stream(request);
            if (pull_events) {
                // Pull on this worker thread so the connection thread stays the sole
                // drainer. A wire CANCEL ends the pull between chunks - the
                // sentinel is never needed here because this loop owns the calls; the
                // next start_stream resets the session's staged request.
                for (;;) {
                    if (cancel_requested.load(std::memory_order_relaxed)) {
                        worker_cancelled = true;
                        break;
                    }
                    std::optional<rt::StreamEvent> event = session.next_stream_event();
                    if (!event.has_value()) {
                        break;
                    }
                    forward_event(*event);
                }
            }
        } catch (const CancelSentinel &) {
            worker_cancelled = true;
        } catch (const std::exception & ex) {
            worker_error = ex.what();
        } catch (...) {
            worker_error = "unknown synthesis error";
        }
        {
            std::lock_guard<std::mutex> lock(pump.mutex);
            pump.generation_done = true;
        }
        pump.cv.notify_one();
    });

    // Drain until generation is done AND the queue is empty, so every sink-pushed
    // byte is on the wire before TTS_DONE. A write failure - the peer vanished
    // mid-generation - marks the connection dead; later payloads are discarded (the
    // queue still drains so the loop terminates and memory is released).
    //
    // The wait is a short wait_for so the loop can also POLL the fd for
    // mid-request frames. A CANCEL arms the sentinel AND switches the drain to
    // discard: from that moment the client receives no further audio from this
    // request - what is already queued is dropped here, and the sink throw
    // stops more from being generated. A BYE is remembered for the caller; EOF or a
    // read error marks the connection dead exactly like a write failure.
    bool write_failed = false;
    bool conn_dead = false;
    for (;;) {
        std::string payload;
        bool have_payload = false;
        bool drained = false;
        {
            std::unique_lock<std::mutex> lock(pump.mutex);
            pump.cv.wait_for(lock, std::chrono::milliseconds(10), [&pump] {
                return !pump.queue.empty() || pump.generation_done;
            });
            if (!pump.queue.empty()) {
                payload = std::move(pump.queue.front());
                pump.queue.pop_front();
                have_payload = true;
            } else if (pump.generation_done) {
                drained = true;
            }
        }
        if (have_payload && !write_failed && !conn_dead && !cancel_requested.load()) {
            try {
                write_frame(fd, MsgType::TtsAudio, payload);
            } catch (const std::exception & ex) {
                write_failed = true;  // log once, discard the rest for this request
                log_line("warn",
                         std::string("dropping TTS audio, peer write failed: ") + ex.what());
            }
        }
        if (drained) {
            break;  // generation_done and nothing left to drain
        }
        // Poll for a mid-request frame without ever blocking the drain.
        //
        // A note on the poll-then-read gap: read_frame blocks until the WHOLE
        // frame has arrived, so POLLIN with only a partial frame present would park
        // this loop mid-drain. That is harmless here: the only frames a client may
        // send mid-request are the 7-byte CANCEL/BYE controls ([type u8][len u32] +
        // "{}"), every real client writes a frame in ONE send, and AF_UNIX
        // SOCK_STREAM delivers a small single write intact - so POLLIN implies the
        // whole control frame is already readable. A peer that deliberately splits
        // a frame across writes stalls only the gap between ITS OWN writes (the
        // cancel still lands - pinned by the split-write test), and an EOF
        // mid-frame still surfaces as conn_dead via read_frame's throw.
        while (!conn_dead) {
            struct pollfd pfd{fd, POLLIN, 0};
            if (::poll(&pfd, 1, 0) <= 0 || (pfd.revents & POLLIN) == 0) {
                break;
            }
            Frame in;
            bool got = false;
            try {
                got = read_frame(fd, in);
            } catch (const std::exception &) {
                got = false;
            }
            if (!got) {
                conn_dead = true;  // EOF/read error: the peer is gone mid-request
                break;
            }
            if (in.type == MsgType::Cancel) {
                cancel_requested.store(true);
            } else if (in.type == MsgType::Bye) {
                bye_seen = true;  // honoured after the reply is finished (pre-CANCEL shape)
            } else {
                log_line("warn", "ignoring an unexpected frame during a TTS request");
            }
        }
    }
    worker.join();
    session.set_stream_event_sink(nullptr);

    if (conn_dead) {
        return false;  // nothing can be replied; the caller ends the connection
    }
    try {
        if (!worker_error.empty()) {
            // A failed request is closed by ERROR instead of TTS_DONE (PROTOCOL.md).
            write_frame(fd, MsgType::Error,
                        build_error("internal", "synthesis failed: " + worker_error));
            return true;
        }
        if (write_failed) {
            return false;  // the peer is gone; the read loop will see EOF next
        }
        // The drain loop above emptied the queue after generation_done, so this
        // really is sent only after the last audio byte of the request.
        // A cancelled request is still a cleanly-closed one: the marker says so,
        // whether the sentinel fired or generation beat the cancel to the finish.
        if (worker_cancelled || cancel_requested.load()) {
            write_frame(fd, MsgType::TtsDone, "{\"ok\":true,\"cancelled\":true}");
        } else {
            // The trimmer's held request-tail silence (never a seam) goes out
            // ahead of TTS_DONE. The worker joined above, so this thread owns the
            // trimmer now.
            std::vector<float> tail;
            trimmer.finish(tail);
            if (!tail.empty()) {
                write_frame(fd, MsgType::TtsAudio, float_to_s16le(tail));
            }
            write_frame(fd, MsgType::TtsDone, "{\"ok\":true}");
        }
    } catch (const std::exception & ex) {
        log_line("warn", std::string("TTS reply write failed: ") + ex.what());
        return false;
    }
    return true;
}

}  // namespace

void drive_tts_session(int fd, rt::IStreamingVoiceTaskSession & session,
                       const OptionMap & request_options, const WavLoader & load_wav,
                       const TtsDriveConfig & config, bool ref_text_required) {
    try {
        // No streaming-input audio contract to state (input_kind=none); prepare with
        // an empty request so the session validates and marks itself ready.
        // A prepare() failure is answered with a refusal-class ERROR before the
        // connection ends - the client completed HELLO/READY and is blocked reading
        // this socket, and a bare EOF here is indistinguishable from a crash.
        try {
            session.prepare(rt::SessionPreparationRequest{});
        } catch (const std::exception & ex) {
            log_line("warn", std::string("TTS session prepare failed: ") + ex.what());
            write_frame(fd, MsgType::Error,
                        build_error("prepare_failed",
                                    std::string("TTS session prepare failed: ") +
                                        ex.what()));
            return;
        }

        Frame frame;
        while (read_frame(fd, frame)) {
            switch (frame.type) {
            case MsgType::Text: {
                bool bye_seen = false;
                if (!service_text_request(fd, session, request_options, load_wav,
                                          config, frame.payload, bye_seen, ref_text_required)) {
                    session.set_stream_event_sink(nullptr);
                    return;  // connection known dead
                }
                if (bye_seen) {
                    // A BYE read mid-request ends the connection only AFTER its
                    // reply was finished - the pre-CANCEL queued-BYE behavior.
                    session.set_stream_event_sink(nullptr);
                    return;
                }
                break;
            }
            case MsgType::Cancel:
                // A stray CANCEL with nothing in flight (the client cancelled just
                // as the request finished) is a benign race, not a violation.
                break;
            case MsgType::Bye:
                session.set_stream_event_sink(nullptr);
                return;
            default:
                // The ASR frame set (AUDIO/FLUSH/RESET) is not valid on a TTS
                // connection - the HELLO's model_task implied THIS frame set.
                write_frame(fd, MsgType::Error,
                            build_error("bad_frame",
                                        "unexpected frame type for a TTS session"));
                break;
            }
        }
    } catch (const std::exception & ex) {
        // An I/O failure (peer close mid-reply) or an unexpected session error ends
        // the connection; a process-fatal fault is handled one level up.
        log_line("warn", std::string("TTS drive loop ended: ") + ex.what());
    }
    session.set_stream_event_sink(nullptr);
}

}  // namespace sidecar
