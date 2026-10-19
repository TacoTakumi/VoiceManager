// The REAL drive_streaming_session against a fake streaming session
// over a socketpair - no model, no GPU, no engine libs.
//
// The fake mirrors the voxtral_realtime session's started-state contract
// (src/models/voxtral_realtime/session.cpp at the pinned bb15edd7):
//   - process_audio_chunk() and finalize() THROW without a prior start_stream()
//   - start_stream() resets per-utterance state and sets started
//   - finalize() clears started on success; reset() clears started
//
// What must hold (GREEN): the drive loop re-arms the stream per utterance - after a
// successful finalize and after Reset - so back-to-back utterances on one connection
// both produce their FINAL, and a FLUSH with no started stream yields an ERROR frame,
// never a dead connection. Nemotron is unaffected: its start_stream() is reset() plus
// option stashing, so a per-utterance re-arm is exactly the RESET it already expects.
#include "drive_loop.h"
#include "frame.h"
#include "protocol.h"
#include "task_registry.h"

#include "cJSON.h"

#include <csignal>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rt = engine::runtime;

// Shared CHECK counter and the TTS drive tests (tts_drive_test.cpp): one test
// binary keeps the pytest wrapper a single entry.
int g_failures = 0;
int run_tts_drive_tests();
int run_task_registry_tests();  // task_registry_test.cpp

namespace {

#define CHECK(cond, what)                                                          \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (what));  \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

// Mirrors VoxtralRealtimeSession's streaming state machine, with a controllable
// start_stream failure for the no-started-stream scenario.
class FakeStreamingSession : public rt::IStreamingVoiceTaskSession {
public:
    int fail_start_call = -1;  // 1-based start_stream() call index that throws
    // request.options as observed by each start_stream() call, in call order.
    std::vector<std::unordered_map<std::string, std::string>> start_request_options;
    // Events pushed through the stream-event sink at the START of each finalize(),
    // mimicking a model that emits deltas/markers during its finalize decode.
    std::vector<rt::StreamEvent> finalize_sink_events;

    std::string family() const override { return "fake_streaming"; }
    rt::VoiceTaskKind task_kind() const override { return rt::VoiceTaskKind::Asr; }
    rt::RunMode run_mode() const override { return rt::RunMode::Streaming; }
    void prepare(const rt::SessionPreparationRequest &) override { prepared_ = true; }

    void start_stream(const rt::TaskRequest & request) override {
        require_prepared("start_stream");
        start_request_options.push_back(request.options);
        if (++start_calls_ == fail_start_call) {
            throw std::runtime_error("fake start_stream() induced failure");
        }
        reset();  // voxtral: start_stream() resets per-utterance state...
        started_ = true;  // ...then arms the stream
    }

    rt::StreamEvent process_audio_chunk(const rt::AudioChunk & chunk) override {
        require_prepared("process_audio_chunk");
        if (!started_) {
            throw std::runtime_error("fake process_audio_chunk() requires start_stream");
        }
        samples_ += static_cast<long>(chunk.samples.size());
        return rt::StreamEvent{};  // accumulate-only, like the shipped ASR models
    }

    rt::TaskResult finalize() override {
        require_prepared("finalize");
        if (!started_) {
            throw std::runtime_error("fake finalize() requires start_stream");
        }
        if (sink_) {
            for (const auto & event : finalize_sink_events) {
                sink_(event);
            }
        }
        rt::TaskResult result;
        result.text_output = rt::Transcript{
            "final#" + std::to_string(++finals_) + " samples=" + std::to_string(samples_),
            "en"};
        started_ = false;  // voxtral: finalize() ends the armed stream
        return result;
    }

    void reset() override {
        require_prepared("reset");
        samples_ = 0;
        started_ = false;  // voxtral: reset() clears the started state
    }

    void set_stream_event_sink(rt::StreamEventCallback sink) override {
        sink_ = std::move(sink);
    }

    bool started() const { return started_; }

private:
    void require_prepared(const char * what) const {
        if (!prepared_) {
            throw std::runtime_error(std::string("fake ") + what + " before prepare");
        }
    }

    bool prepared_ = false;
    bool started_ = false;
    int start_calls_ = 0;
    int finals_ = 0;
    long samples_ = 0;
    rt::StreamEventCallback sink_;
};

// --- client-side helpers over the socketpair ------------------------------------------

std::string audio_payload(std::size_t n_samples) {
    return std::string(n_samples * 2, '\0');  // silent s16le is fine for a fake
}

// Reads frames until one of `type` arrives; returns false on EOF / error (a dead
// connection - the RED failure mode). ERROR frames encountered on the way are counted.
bool read_until(int fd, sidecar::MsgType type, sidecar::Frame & out, int * errors_seen) {
    sidecar::Frame frame;
    try {
        while (sidecar::read_frame(fd, frame)) {
            if (frame.type == sidecar::MsgType::Error && errors_seen != nullptr) {
                ++*errors_seen;
            }
            if (frame.type == type) {
                out = frame;
                return true;
            }
        }
    } catch (const std::exception &) {
        return false;
    }
    return false;
}

std::string final_text(const sidecar::Frame & frame) {
    cJSON * root = cJSON_Parse(frame.payload.c_str());
    if (root == nullptr) {
        return "<unparseable>";
    }
    std::string text = "<missing>";
    const cJSON * output = cJSON_GetObjectItemCaseSensitive(root, "text_output");
    const cJSON * text_item =
        (output != nullptr) ? cJSON_GetObjectItemCaseSensitive(output, "text") : nullptr;
    if (cJSON_IsString(text_item)) {
        text = text_item->valuestring;
    }
    cJSON_Delete(root);
    return text;
}

struct Connection {
    FakeStreamingSession session;
    int client_fd = -1;
    std::thread server;

    explicit Connection(int fail_start_call = -1,
                        std::unordered_map<std::string, std::string> request_options = {}) {
        session.fail_start_call = fail_start_call;
        int fds[2] = {-1, -1};
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
            throw std::runtime_error("socketpair failed");
        }
        client_fd = fds[0];
        const int server_fd = fds[1];
        server = std::thread([this, server_fd, options = std::move(request_options)] {
            sidecar::drive_streaming_session(server_fd, session, options);
            ::close(server_fd);
        });
    }

    ~Connection() {
        if (client_fd >= 0) {
            ::close(client_fd);
        }
        if (server.joinable()) {
            server.join();
        }
    }

    // A send to a dead server is itself a RED symptom; keep it non-fatal so the
    // scenario's read assertions report the failure instead of aborting the binary.
    void send(sidecar::MsgType type, const std::string & payload) {
        try {
            sidecar::write_frame(client_fd, type, payload);
        } catch (const std::exception & ex) {
            std::fprintf(stderr, "note: client send failed (%s)\n", ex.what());
        }
    }
};

// --- scenarios ------------------------------------------------------------------------

// The fake itself honours the voxtral started-state contract.
void test_fake_contract() {
    FakeStreamingSession fake;
    fake.prepare(rt::SessionPreparationRequest{});
    bool threw = false;
    try {
        fake.process_audio_chunk(rt::AudioChunk{});
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(threw, "process_audio_chunk without start_stream must throw");
    threw = false;
    try {
        fake.finalize();
    } catch (const std::exception &) {
        threw = true;
    }
    CHECK(threw, "finalize without start_stream must throw");
    fake.start_stream(rt::TaskRequest{});
    CHECK(fake.started(), "start_stream must arm the stream");
    fake.finalize();
    CHECK(!fake.started(), "finalize must clear the started state");
    fake.start_stream(rt::TaskRequest{});
    fake.reset();
    CHECK(!fake.started(), "reset must clear the started state");
}

// Back-to-back utterances with FLUSH-only turnover both produce their FINAL.
void test_back_to_back_flush() {
    Connection conn;
    sidecar::Frame frame;

    conn.send(sidecar::MsgType::Audio, audio_payload(160));
    conn.send(sidecar::MsgType::Flush, "{}");
    CHECK(read_until(conn.client_fd, sidecar::MsgType::Final, frame, nullptr),
          "utterance 1: FINAL must arrive");
    CHECK(final_text(frame) == "final#1 samples=160", "utterance 1: wrong FINAL text");

    conn.send(sidecar::MsgType::Audio, audio_payload(320));
    conn.send(sidecar::MsgType::Flush, "{}");
    CHECK(read_until(conn.client_fd, sidecar::MsgType::Final, frame, nullptr),
          "utterance 2 after finalize: FINAL must arrive (re-arm after finalize)");
    CHECK(final_text(frame) == "final#2 samples=320", "utterance 2: wrong FINAL text");

    conn.send(sidecar::MsgType::Bye, "{}");
}

// RESET between utterances re-arms, and discards audio fed before the RESET.
void test_reset_turnover() {
    Connection conn;
    sidecar::Frame frame;

    conn.send(sidecar::MsgType::Audio, audio_payload(160));
    conn.send(sidecar::MsgType::Flush, "{}");
    CHECK(read_until(conn.client_fd, sidecar::MsgType::Final, frame, nullptr),
          "utterance 1: FINAL must arrive");

    conn.send(sidecar::MsgType::Audio, audio_payload(100));  // abandoned pre-RESET audio
    conn.send(sidecar::MsgType::Reset, "{}");
    conn.send(sidecar::MsgType::Audio, audio_payload(480));
    conn.send(sidecar::MsgType::Flush, "{}");
    CHECK(read_until(conn.client_fd, sidecar::MsgType::Final, frame, nullptr),
          "utterance 2 after RESET: FINAL must arrive (re-arm after reset)");
    CHECK(final_text(frame) == "final#2 samples=480",
          "utterance 2 after RESET: RESET must discard pre-RESET audio");

    conn.send(sidecar::MsgType::Bye, "{}");
}

// A failed re-arm leaves an unstarted stream: the following FLUSH must produce an
// ERROR frame (not a dead connection), and a RESET must recover the session.
void test_flush_with_no_started_stream() {
    Connection conn(/*fail_start_call=*/2);  // call 1 = connection open; 2 = first re-arm
    sidecar::Frame frame;
    int errors = 0;

    conn.send(sidecar::MsgType::Audio, audio_payload(160));
    conn.send(sidecar::MsgType::Flush, "{}");
    CHECK(read_until(conn.client_fd, sidecar::MsgType::Final, frame, &errors),
          "utterance 1: FINAL must arrive");

    // The re-arm after finalize threw; the stream is NOT started. FLUSH must be answered
    // with an ERROR frame and the connection must survive.
    conn.send(sidecar::MsgType::Flush, "{}");
    errors = 0;
    CHECK(read_until(conn.client_fd, sidecar::MsgType::Error, frame, &errors),
          "FLUSH with no started stream: ERROR frame must arrive, connection alive");

    // RESET retries the re-arm (call 3 succeeds) and the connection keeps working.
    conn.send(sidecar::MsgType::Reset, "{}");
    conn.send(sidecar::MsgType::Audio, audio_payload(320));
    conn.send(sidecar::MsgType::Flush, "{}");
    CHECK(read_until(conn.client_fd, sidecar::MsgType::Final, frame, nullptr),
          "post-recovery utterance: FINAL must arrive");
    CHECK(final_text(frame) == "final#2 samples=320",
          "post-recovery utterance: wrong FINAL text");

    conn.send(sidecar::MsgType::Bye, "{}");
}

// Reads frames until a FINAL arrives, collecting every EVENT frame on the way.
// Returns false on EOF/error before the FINAL.
bool collect_events_until_final(int fd, std::vector<sidecar::Frame> & events,
                                sidecar::Frame & final_frame) {
    sidecar::Frame frame;
    try {
        while (sidecar::read_frame(fd, frame)) {
            if (frame.type == sidecar::MsgType::Event) {
                events.push_back(frame);
            } else if (frame.type == sidecar::MsgType::Final) {
                final_frame = frame;
                return true;
            }
        }
    } catch (const std::exception &) {
        return false;
    }
    return false;
}

// (is_final, partial text) of an EVENT frame's JSON payload.
std::pair<bool, std::string> event_fields(const sidecar::Frame & frame) {
    cJSON * root = cJSON_Parse(frame.payload.c_str());
    if (root == nullptr) {
        return {false, "<unparseable>"};
    }
    const cJSON * is_final = cJSON_GetObjectItemCaseSensitive(root, "is_final");
    std::pair<bool, std::string> out{cJSON_IsTrue(is_final), "<missing>"};
    const cJSON * partial = cJSON_GetObjectItemCaseSensitive(root, "partial_text");
    const cJSON * text =
        (partial != nullptr) ? cJSON_GetObjectItemCaseSensitive(partial, "text") : nullptr;
    if (cJSON_IsString(text)) {
        out.second = text->valuestring;
    }
    cJSON_Delete(root);
    return out;
}

rt::StreamEvent text_event(const std::string & text, bool is_final) {
    rt::StreamEvent event;
    event.partial_text = rt::Transcript{text, "en"};
    event.is_final = is_final;
    return event;
}

// Sink events carry the model's real is_final on the wire, and a bare final MARKER
// (is_final set, no text/word content) is not forwarded at all - the FINAL frame from
// finalize stays the single utterance-final.
void test_event_hygiene() {
    Connection conn;
    conn.session.finalize_sink_events = {
        text_event("Hel", /*is_final=*/false),   // a mid-decode delta
        [] {                                     // a contentless end-of-stream marker
            rt::StreamEvent marker;
            marker.is_final = true;
            return marker;
        }(),
        text_event("Hello", /*is_final=*/true),  // a text-carrying final event
    };

    conn.send(sidecar::MsgType::Audio, audio_payload(160));
    conn.send(sidecar::MsgType::Flush, "{}");
    std::vector<sidecar::Frame> events;
    sidecar::Frame final_frame;
    CHECK(collect_events_until_final(conn.client_fd, events, final_frame),
          "event hygiene: the FINAL frame must still arrive");
    CHECK(events.size() == 2,
          "the contentless final marker must NOT be forwarded as an EVENT frame");
    if (events.size() == 2) {
        const auto delta = event_fields(events[0]);
        CHECK(!delta.first, "a mid-decode delta must carry is_final=false");
        CHECK(delta.second == "Hel", "the delta's text must be forwarded verbatim");
        const auto final_event = event_fields(events[1]);
        CHECK(final_event.first,
              "a text-carrying final event must carry the model's is_final=true");
        CHECK(final_event.second == "Hello", "the final event's text must be forwarded");
    }
    conn.send(sidecar::MsgType::Bye, "{}");
}

// The opaque request-option map reaches the session at EVERY start_stream: the initial
// arm and each per-utterance re-arm. The drive loop never inspects it.
void test_request_options_reach_every_start_stream() {
    const std::unordered_map<std::string, std::string> options{
        {"voxtral_realtime.stream_batch_tokens", "2"},
        {"temperature", "0.0"},
    };
    sidecar::Frame frame;
    Connection conn(/*fail_start_call=*/-1, options);

    conn.send(sidecar::MsgType::Audio, audio_payload(160));
    conn.send(sidecar::MsgType::Flush, "{}");
    CHECK(read_until(conn.client_fd, sidecar::MsgType::Final, frame, nullptr),
          "options run, utterance 1: FINAL must arrive");
    conn.send(sidecar::MsgType::Reset, "{}");
    conn.send(sidecar::MsgType::Audio, audio_payload(320));
    conn.send(sidecar::MsgType::Flush, "{}");
    CHECK(read_until(conn.client_fd, sidecar::MsgType::Final, frame, nullptr),
          "options run, utterance 2: FINAL must arrive");
    conn.send(sidecar::MsgType::Bye, "{}");
    conn.server.join();  // all start_stream() calls have happened once the loop returns

    // initial arm + re-arm after finalize + re-arm after RESET + re-arm after finalize.
    CHECK(conn.session.start_request_options.size() >= 3,
          "start_stream must have been called for the initial arm and every re-arm");
    for (const auto & seen : conn.session.start_request_options) {
        CHECK(seen == options, "every start_stream must carry the same opaque option map");
    }
}

// The additive protocol-1 TTS frame types. The codec is type-agnostic,
// so what this pins is the wire contract itself: the tag values (0x0A/0x0B/0x0C), the
// protocol field still 1, byte-exact round-trips for each new type (including binary
// PCM with embedded NULs and high bytes), and oversize rejection identical to the
// existing framing rules.
void test_tts_frame_codec() {
    CHECK(sidecar::kProtocolVersion == 1, "TTS frames are additive: protocol stays 1");
    CHECK(static_cast<std::uint8_t>(sidecar::MsgType::Text) == 0x0A, "TEXT tag is 0x0A");
    CHECK(static_cast<std::uint8_t>(sidecar::MsgType::TtsAudio) == 0x0B,
          "TTS_AUDIO tag is 0x0B");
    CHECK(static_cast<std::uint8_t>(sidecar::MsgType::TtsDone) == 0x0C,
          "TTS_DONE tag is 0x0C");

    int fds[2] = {-1, -1};
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
        throw std::runtime_error("socketpair failed");
    }

    // Byte-exact round-trip of each new type. The TTS_AUDIO payload is deliberately
    // binary (s16le-ish bytes with NULs and >0x7F values), not text.
    std::string pcm;
    for (int i = 0; i < 256; ++i) {
        pcm.push_back(static_cast<char>(i));
        pcm.push_back('\0');
    }
    const std::pair<sidecar::MsgType, std::string> cases[] = {
        {sidecar::MsgType::Text,
         R"({"text":"hello","voice_wav":"/v/s101.wav","ref_text":"reference words"})"},
        {sidecar::MsgType::TtsAudio, pcm},
        {sidecar::MsgType::TtsDone, R"({"ok":true})"},
    };
    for (const auto & [type, payload] : cases) {
        sidecar::write_frame(fds[0], type, payload);
        sidecar::Frame got;
        CHECK(sidecar::read_frame(fds[1], got), "round-trip must yield a complete frame");
        CHECK(got.type == type, "round-trip must preserve the type tag");
        CHECK(got.payload == payload, "round-trip must preserve the payload byte-exactly");
    }

    // Oversize rejection, identical to the existing framing rules: an outgoing TTS
    // frame larger than kMaxPayloadBytes throws before any byte hits the wire, and an
    // incoming header declaring an oversize length throws on read.
    bool write_threw = false;
    try {
        sidecar::write_frame(fds[0], sidecar::MsgType::TtsAudio,
                             std::string(sidecar::kMaxPayloadBytes + 1, 'x'));
    } catch (const std::exception &) {
        write_threw = true;
    }
    CHECK(write_threw, "an oversize outgoing TTS_AUDIO frame must be rejected");

    const std::uint32_t bad_len = sidecar::kMaxPayloadBytes + 1;
    const unsigned char bad_header[sidecar::kFrameHeaderBytes] = {
        static_cast<unsigned char>(sidecar::MsgType::Text),
        static_cast<unsigned char>((bad_len >> 24) & 0xFF),
        static_cast<unsigned char>((bad_len >> 16) & 0xFF),
        static_cast<unsigned char>((bad_len >> 8) & 0xFF),
        static_cast<unsigned char>(bad_len & 0xFF),
    };
    if (::write(fds[0], bad_header, sizeof(bad_header)) !=
        static_cast<ssize_t>(sizeof(bad_header))) {
        throw std::runtime_error("short write of the oversize header");
    }
    bool read_threw = false;
    try {
        sidecar::Frame got;
        sidecar::read_frame(fds[1], got);
    } catch (const std::exception &) {
        read_threw = true;
    }
    CHECK(read_threw, "an oversize declared TEXT length must be rejected on read");

    ::close(fds[0]);
    ::close(fds[1]);
}

// merge_options: registry defaults overlaid by operator overrides, overrides winning.
void test_merge_options() {
    const sidecar::OptionMap defaults{{"a", "1"}, {"b", "2"}};
    const sidecar::OptionMap overrides{{"b", "3"}, {"c", "4"}};
    const sidecar::OptionMap merged = sidecar::merge_options(defaults, overrides);
    const sidecar::OptionMap expected{{"a", "1"}, {"b", "3"}, {"c", "4"}};
    CHECK(merged == expected, "overrides must win over defaults, others carried through");
    CHECK(sidecar::merge_options({}, {}).empty(), "empty maps merge to empty");
}

// The default registry serves all four shipped model_tasks. The nemotron entry stays
// byte-for-byte what it was (no options - shipped behavior is exactly the engine
// defaults), the voxtral entry pins the measured arena/context recipe in its session-option
// defaults, the dots entry pins the upstream long-form guards (text_chunk_size=200,
// max_tokens=500, text_chunk_mode=default) with ref_text required, and the voxcpm2
// entry pins retry_badcase=false and voxcpm2.chunk_strategy=stateless with ref_text
// optional.
void test_registry_entries() {
    const sidecar::TaskRegistry registry = sidecar::TaskRegistry::default_registry();
    CHECK(registry.size() == 6, "the registry must list exactly the six shipped model_tasks");

    const auto ids = registry.ids();
    CHECK(std::find(ids.begin(), ids.end(), "nemotron-3.5-asr-streaming") != ids.end(),
          "ids() must list the nemotron model_task");
    CHECK(std::find(ids.begin(), ids.end(), "voxtral-mini-4b-realtime-asr-streaming") != ids.end(),
          "ids() must list the voxtral model_task");
    CHECK(std::find(ids.begin(), ids.end(), "dots-tts-streaming") != ids.end(),
          "ids() must list the dots TTS model_task");
    CHECK(std::find(ids.begin(), ids.end(), "voxcpm2-tts-streaming") != ids.end(),
          "ids() must list the voxcpm2 TTS model_task");
    CHECK(registry.find("no-such-model-task") == nullptr,
          "an unregistered id must not resolve");

    const sidecar::TaskDescriptor * nemotron = registry.find("nemotron-3.5-asr-streaming");
    CHECK(nemotron != nullptr, "the nemotron model_task must resolve");
    if (nemotron != nullptr) {
        CHECK(nemotron->family == "nemotron_asr", "nemotron family unchanged");
        CHECK(nemotron->task == "asr" && nemotron->mode == "streaming" &&
                  nemotron->input_kind == "audio_chunks",
              "nemotron taxonomy unchanged");
        CHECK(nemotron->session_options.empty() && nemotron->request_options.empty(),
              "nemotron must keep NO default options - byte-for-byte current behavior");
    }

    const sidecar::TaskDescriptor * voxtral =
        registry.find("voxtral-mini-4b-realtime-asr-streaming");
    CHECK(voxtral != nullptr, "the voxtral model_task must resolve");
    if (voxtral != nullptr) {
        CHECK(voxtral->family == "voxtral_realtime", "voxtral family_hint");
        CHECK(voxtral->task == "asr" && voxtral->mode == "streaming" &&
                  voxtral->input_kind == "audio_chunks",
              "voxtral taxonomy: streaming ASR over audio chunks");
        const sidecar::OptionMap recipe{
            {"voxtral_realtime.audio_encoder_graph_arena_mb", "512"},
            {"voxtral_realtime.audio_encoder_weight_context_mb", "128"},
            {"voxtral_realtime.text_decoder_prefill_graph_arena_mb", "512"},
            {"voxtral_realtime.text_decoder_decode_graph_arena_mb", "512"},
            {"voxtral_realtime.text_decoder_weight_context_mb", "128"},
            {"voxtral_realtime.stream_decode_cache_steps", "1024"},
        };
        CHECK(voxtral->session_options == recipe,
              "voxtral session-option defaults must pin the measured arena/context recipe");
        CHECK(voxtral->request_options.empty(),
              "voxtral request options: engine generation defaults, nothing pinned");
    }

    const sidecar::TaskDescriptor * dots = registry.find("dots-tts-streaming");
    CHECK(dots != nullptr, "the dots TTS model_task must resolve");
    if (dots != nullptr) {
        CHECK(dots->family == "dots_tts", "dots family_hint");
        CHECK(dots->task == "tts" && dots->mode == "streaming" && dots->input_kind == "none",
              "dots taxonomy: streaming TTS, no streaming input");
        CHECK(dots->session_options.empty(),
              "dots session options: engine defaults, nothing pinned");
        const sidecar::OptionMap guards{
            {"text_chunk_size", "200"},
            {"max_tokens", "500"},
            {"text_chunk_mode", "default"},  // clause-break cuts for over-budget sentences
        };
        CHECK(dots->request_options == guards,
              "dots request-option defaults must pin the upstream long-form guards");
        CHECK(dots->ref_text_required,
              "dots requires ref_text - the transcript is mandatory");
    }

    const sidecar::TaskDescriptor * voxcpm2 = registry.find("voxcpm2-tts-streaming");
    CHECK(voxcpm2 != nullptr, "the voxcpm2 TTS model_task must resolve");
    if (voxcpm2 != nullptr) {
        CHECK(voxcpm2->family == "voxcpm2", "voxcpm2 family_hint");
        CHECK(voxcpm2->task == "tts" && voxcpm2->mode == "streaming" &&
                  voxcpm2->input_kind == "none",
              "voxcpm2 taxonomy: streaming TTS, no streaming input");
        CHECK(voxcpm2->session_options.empty(),
              "voxcpm2 session options: engine defaults, nothing pinned");
        const sidecar::OptionMap voxcpm2_pins{
            {"retry_badcase", "false"},
            {"voxcpm2.chunk_strategy", "stateless"},  // keeps 0.6's per-chunk prompt reuse
        };
        CHECK(voxcpm2->request_options == voxcpm2_pins,
              "voxcpm2 request-option defaults must pin ONLY retry_badcase=false and "
              "voxcpm2.chunk_strategy=stateless");
        CHECK(!voxcpm2->ref_text_required,
              "voxcpm2 does not require ref_text - the transcript is optional");
    }
}

}  // namespace

int main() {
    std::signal(SIGPIPE, SIG_IGN);  // a dead-peer write must surface as EPIPE, not kill us
    test_fake_contract();
    test_back_to_back_flush();
    test_reset_turnover();
    test_flush_with_no_started_stream();
    test_request_options_reach_every_start_stream();
    test_tts_frame_codec();
    test_merge_options();
    test_registry_entries();
    test_event_hygiene();
    run_tts_drive_tests();
    run_task_registry_tests();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d assertion(s) failed\n", g_failures);
        return 1;
    }
    std::printf("drive_loop re-arm tests: all passed\n");
    return 0;
}
