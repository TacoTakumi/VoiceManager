// The REAL drive_tts_session against a fake TTS
// streaming session over a socketpair - no model, no GPU, no engine libs.
//
// The fake mirrors the dots_tts session's streaming contract (session.cpp at the
// pinned bb15edd7): start_stream() is BLOCKING - it runs the whole generation,
// pushing StreamEvents whose named_audio_outputs carry the audio (ids like
// "segment_0_chunk_0", closed by an id "flush" buffer with meta flush=1) through the
// stream-event sink, and returns only when synthesis is complete.
//
// What must hold (GREEN):
//   - incremental: audio pushed mid-generation reaches the SOCKET before the fake's
//     start_stream returns (the fake gates on the client having read the first chunk,
//     so a batch-at-completion drive path deadlocks the gate and fails its timeout)
//   - the trailing flush-named buffer is forwarded before TTS_DONE
//   - the dispatched TaskRequest carries the text, the loaded voice reference, and
//     options: reference_text merged over the caller's request options
//   - back-to-back TEXT requests on one connection both complete
//   - BYE/EOF mid-generation ends cleanly with no hang
//   - a TEXT missing/empty ref_text, or an unloadable WAV, is never dispatched
//     (ERROR bad_frame instead) and the connection survives
#include "tts_drive.h"

#include "frame.h"
#include "protocol.h"
#include "task_registry.h"

#include "engine/framework/text/chunking.h"

#include "cJSON.h"

#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace rt = engine::runtime;

extern int g_failures;  // shared with the rearm test's CHECK counter (one binary)

namespace {

#define CHECK(cond, what)                                                          \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (what));  \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

// Known sample values so the expected s16le bytes are exact: the drive path encodes
// float -> s16le with round(x * 32768) clamped, the inverse of the engine's /32768.
std::vector<float> floats_from_s16(std::initializer_list<std::int16_t> values) {
    std::vector<float> out;
    for (const std::int16_t v : values) {
        out.push_back(static_cast<float>(v) / 32768.0f);
    }
    return out;
}

std::string s16le_bytes(std::initializer_list<std::int16_t> values) {
    std::string out;
    for (const std::int16_t v : values) {
        out.push_back(static_cast<char>(v & 0xFF));
        out.push_back(static_cast<char>((v >> 8) & 0xFF));
    }
    return out;
}

// The fake's scripted generation: two mid-generation chunks then the vocoder flush.
const std::initializer_list<std::int16_t> kChunkA{0, 1000, -1000, 32767};
const std::initializer_list<std::int16_t> kChunkB{-32768, 42, -42};
const std::initializer_list<std::int16_t> kFlush{7, -7};

// Mirrors DotsSession's streaming contract: start_stream() runs the whole generation
// synchronously, pushing named audio buffers through the sink as it goes, flush last.
class FakeTtsSession : public rt::IStreamingVoiceTaskSession {
public:
    // When set, start_stream() blocks after pushing the FIRST chunk until the test
    // signals proceed_ - used to hold generation "in flight" while the client reads
    // the socket (incremental proof), closes it (EOF), or sends BYE.
    bool gate_after_first_chunk = false;
    std::atomic<bool> proceed{false};
    std::atomic<bool> gate_timed_out{false};

    // Bulk mode: when bulk_chunks > 0, start_stream() pushes that many
    // deterministic-pattern buffers of bulk_chunk_samples each (plus the flush) as
    // fast as it can - far more bytes than a socket buffer holds - and raises
    // generation_done as its last act, so a test can prove the sink push never
    // blocked on a reader that is not draining.
    int bulk_chunks = 0;
    int bulk_chunk_samples = 0;
    std::atomic<bool> generation_done{false};

    // Drip mode: push up to drip_chunks small buffers, 5 ms apart, so a wire
    // CANCEL can land mid-generation. The sink-throw sentinel unwinds out of
    // start_stream through push_samples - nothing in the fake may catch it, exactly
    // like the real dots session (the sink-throw probe, twice).
    int drip_chunks = 0;
    std::atomic<int> drip_pushed{0};

    // Scripted mode: when non-empty, start_stream pushes exactly these named
    // buffers in order (no gate, no flush tail) - the seam-trim tests script segment
    // boundaries and silence runs precisely.
    struct ScriptedPush {
        std::string id;
        std::vector<float> samples;
    };
    std::vector<ScriptedPush> scripted;

    std::vector<rt::TaskRequest> start_requests;  // copy of every dispatched request

    // When set, prepare() throws - the scripted engine-side load/validate
    // failure the drive path must answer with ERROR prepare_failed, not bare EOF.
    bool prepare_throws = false;

    // Pull-events mode: mirrors BreezeTTSSession's
    // streaming contract at the 820cddc pin - streaming_policy().output is PullEvents,
    // start_stream() only stages the request (it pushes NOTHING through the sink, which
    // the real session ignores), and each next_stream_event() returns one named chunk
    // "chunk_<i>" until `pull_chunks` are out, then nullopt. Calling it before
    // start_stream throws, exactly like the real session.
    rt::StreamingOutputKind output_kind = rt::StreamingOutputKind::FinalResult;
    int pull_chunks = 0;
    int pull_chunk_samples = 0;
    std::atomic<int> pull_calls{0};  // next_stream_event() calls, nullopt one included
    // When >= 0, the pull of that chunk index blocks on `proceed` - the
    // in-flight pull a wire CANCEL lands during. finish_stream() must NEVER be called by
    // the drive; the counter proves it.
    int pull_gate_at_chunk = -1;
    std::atomic<int> finish_stream_calls{0};

    rt::TaskResult finish_stream() override {
        ++finish_stream_calls;
        return rt::TaskResult{};
    }

    rt::StreamingPolicy streaming_policy() const override {
        rt::StreamingPolicy policy;
        policy.input = rt::StreamingInputKind::None;
        policy.output = output_kind;
        return policy;
    }

    std::optional<rt::StreamEvent> next_stream_event() override {
        ++pull_calls;
        if (!pull_started_) {
            throw std::runtime_error("fake pull streaming has not been started");
        }
        if (pull_index_ >= pull_chunks) {
            return std::nullopt;
        }
        const int chunk_index = pull_index_++;
        if (chunk_index == pull_gate_at_chunk) {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!proceed.load()) {
                if (std::chrono::steady_clock::now() > deadline) {
                    gate_timed_out = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        rt::NamedAudioBuffer named;
        named.id = "chunk_" + std::to_string(chunk_index);
        named.audio = rt::AudioBuffer{24000, 1, bulk_pattern(chunk_index, pull_chunk_samples)};
        rt::StreamEvent event;
        event.named_audio_outputs.push_back(std::move(named));
        return event;
    }

    std::string family() const override { return "fake_tts"; }
    rt::VoiceTaskKind task_kind() const override { return rt::VoiceTaskKind::Tts; }
    rt::RunMode run_mode() const override { return rt::RunMode::Streaming; }
    void prepare(const rt::SessionPreparationRequest &) override {
        if (prepare_throws) {
            throw std::runtime_error("scripted prepare failure");
        }
        prepared_ = true;
    }

    // The deterministic bulk pattern, reproducible on the client side for the
    // byte-exact order/completeness comparison.
    static std::vector<float> bulk_pattern(int chunk_index, int samples) {
        std::vector<float> out;
        out.reserve(static_cast<std::size_t>(samples));
        for (int i = 0; i < samples; ++i) {
            const auto s = static_cast<std::int16_t>(((chunk_index * 7919 + i * 31) % 65536) - 32768);
            out.push_back(static_cast<float>(s) / 32768.0f);
        }
        return out;
    }

    void start_stream(const rt::TaskRequest & request) override {
        require_prepared("start_stream");
        start_requests.push_back(request);
        if (output_kind == rt::StreamingOutputKind::PullEvents) {
            pull_index_ = 0;
            pull_started_ = true;  // stage only; the drive must PULL the chunks
            return;
        }
        if (!scripted.empty()) {
            for (const auto & push : scripted) {
                push_samples(push.id, push.samples, /*flush=*/false);
            }
            generation_done = true;
            return;
        }
        if (bulk_chunks > 0) {
            for (int c = 0; c < bulk_chunks; ++c) {
                push_samples("segment_0_chunk_" + std::to_string(c),
                             bulk_pattern(c, bulk_chunk_samples), /*flush=*/false);
            }
            push_chunk("flush", kFlush, /*flush=*/true);
            generation_done = true;  // last act: generation returned without blocking
            return;
        }
        if (drip_chunks > 0) {
            for (int c = 0; c < drip_chunks; ++c) {
                push_chunk("segment_0_chunk_" + std::to_string(c), kChunkA,
                           /*flush=*/false);
                drip_pushed = c + 1;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
            push_chunk("flush", kFlush, /*flush=*/true);
            return;
        }
        push_chunk("segment_0_chunk_0", kChunkA, /*flush=*/false);
        if (gate_after_first_chunk) {
            // A drive path that batches audio until completion never lets the client
            // read the first chunk, so this gate times out and the test fails.
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!proceed.load()) {
                if (std::chrono::steady_clock::now() > deadline) {
                    gate_timed_out = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
        push_chunk("segment_0_chunk_1", kChunkB, /*flush=*/false);
        push_chunk("flush", kFlush, /*flush=*/true);
    }

    rt::StreamEvent process_audio_chunk(const rt::AudioChunk &) override {
        throw std::runtime_error("fake TTS session has no streaming audio input");
    }
    rt::TaskResult finalize() override {
        throw std::runtime_error("fake TTS session has no finalize path");
    }
    void reset() override { require_prepared("reset"); }
    void set_stream_event_sink(rt::StreamEventCallback sink) override {
        sink_ = std::move(sink);
    }

private:
    void require_prepared(const char * what) const {
        if (!prepared_) {
            throw std::runtime_error(std::string("fake ") + what + " before prepare");
        }
    }

    void push_chunk(const std::string & id, std::initializer_list<std::int16_t> values,
                    bool flush) {
        push_samples(id, floats_from_s16(values), flush);
    }

    void push_samples(const std::string & id, std::vector<float> samples, bool flush) {
        if (!sink_) {
            return;
        }
        rt::NamedAudioBuffer named;
        named.id = id;
        named.audio = rt::AudioBuffer{48000, 1, std::move(samples)};
        if (flush) {
            named.meta.insert_or_assign("flush", "1");
        }
        rt::StreamEvent event;
        event.named_audio_outputs.push_back(std::move(named));
        sink_(event);
    }

    bool prepared_ = false;
    bool pull_started_ = false;
    int pull_index_ = 0;
    rt::StreamEventCallback sink_;
};

// --- client-side helpers over the socketpair ------------------------------------------

std::string tts_request_json(const std::string & text, const std::string & voice_wav,
                             const char * ref_text) {
    cJSON * root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "text", text.c_str());
    cJSON_AddStringToObject(root, "voice_wav", voice_wav.c_str());
    if (ref_text != nullptr) {
        cJSON_AddStringToObject(root, "ref_text", ref_text);
    }
    char * printed = cJSON_PrintUnformatted(root);
    std::string out = (printed != nullptr) ? printed : "{}";
    cJSON_free(printed);
    cJSON_Delete(root);
    return out;
}

bool json_bool_field(const std::string & payload, const char * key) {
    cJSON * root = cJSON_Parse(payload.c_str());
    if (root == nullptr) {
        return false;
    }
    const cJSON * item = cJSON_GetObjectItemCaseSensitive(root, key);
    const bool out = cJSON_IsTrue(item);
    cJSON_Delete(root);
    return out;
}

std::string json_string_field(const std::string & payload, const char * key) {
    cJSON * root = cJSON_Parse(payload.c_str());
    if (root == nullptr) {
        return "<unparseable>";
    }
    std::string out = "<missing>";
    const cJSON * item = cJSON_GetObjectItemCaseSensitive(root, key);
    if (cJSON_IsString(item) && item->valuestring != nullptr) {
        out = item->valuestring;
    }
    cJSON_Delete(root);
    return out;
}

// The REAL drive_tts_session on its own thread over a socketpair, with a fake WAV
// loader (no engine libs): the loader records the requested path and returns a fixed
// buffer, or throws for the missing-wav scenario.
struct TtsConnection {
    FakeTtsSession session;
    int client_fd = -1;
    std::thread server;
    std::shared_ptr<std::atomic<bool>> server_done = std::make_shared<std::atomic<bool>>(false);
    std::vector<std::string> loaded_wavs;  // written on the server thread, read after join

    explicit TtsConnection(sidecar::OptionMap request_options = {},
                           sidecar::TtsDriveConfig drive_config = {},
                           bool prepare_throws = false,
                           bool ref_text_required = true) {
        // Fake knobs that the drive path reads immediately (prepare runs the
        // moment the server thread starts) must be set BEFORE the thread spawns.
        session.prepare_throws = prepare_throws;
        int fds[2] = {-1, -1};
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
            throw std::runtime_error("socketpair failed");
        }
        client_fd = fds[0];
        const int server_fd = fds[1];
        auto done = server_done;
        server = std::thread([this, server_fd, request_options, drive_config,
                              ref_text_required, done]() {
            sidecar::drive_tts_session(
                server_fd, session, request_options,
                [this](const std::string & path) -> rt::AudioBuffer {
                    loaded_wavs.push_back(path);
                    if (path.find("missing") != std::string::npos) {
                        throw std::runtime_error("no such wav file");
                    }
                    return rt::AudioBuffer{24000, 1, {0.25f, -0.5f, 0.125f}};
                },
                drive_config, ref_text_required);
            ::close(server_fd);
            done->store(true);
        });
    }

    // The no-hang assertion: the drive path must have returned; a hung server thread
    // would otherwise block this binary forever, so give up loudly instead.
    void join_or_die(const char * what) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!server_done->load()) {
            if (std::chrono::steady_clock::now() > deadline) {
                std::fprintf(stderr, "FAIL: %s: drive_tts_session hung\n", what);
                std::_Exit(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        server.join();
    }

    void send_text(const std::string & text, const std::string & wav, const char * ref) {
        sidecar::write_frame(client_fd, sidecar::MsgType::Text,
                             tts_request_json(text, wav, ref));
    }

    ~TtsConnection() {
        if (client_fd >= 0) {
            ::close(client_fd);
        }
        if (server.joinable()) {
            server.join();
        }
    }
};

// Reads TTS_AUDIO frames, concatenating payloads, until TTS_DONE; false on EOF/error.
bool read_audio_until_done(int fd, std::string & audio, std::string & done_payload) {
    sidecar::Frame frame;
    audio.clear();
    try {
        while (sidecar::read_frame(fd, frame)) {
            if (frame.type == sidecar::MsgType::TtsAudio) {
                audio += frame.payload;
                continue;
            }
            if (frame.type == sidecar::MsgType::TtsDone) {
                done_payload = frame.payload;
                return true;
            }
            std::fprintf(stderr, "FAIL: unexpected frame type 0x%02x before TTS_DONE\n",
                         static_cast<unsigned>(frame.type));
            return false;
        }
    } catch (const std::exception &) {
        return false;
    }
    return false;
}

std::string full_audio_bytes() {
    return s16le_bytes(kChunkA) + s16le_bytes(kChunkB) + s16le_bytes(kFlush);
}

// The full scripted exchange: TEXT -> incremental audio -> flush tail ->
// TTS_DONE, with the request carrying text, voice reference, and merged options.
void test_tts_full_exchange() {
    TtsConnection conn(sidecar::OptionMap{
        {"text_chunk_size", "200"},
        {"max_tokens", "500"},
    });
    conn.session.gate_after_first_chunk = true;

    conn.send_text("Warp core is stable.", "/voices/ref.wav", "reference words");

    // Incremental: the first chunk must be readable NOW, while the fake's
    // start_stream is still blocked on its gate - i.e. before generation returns.
    sidecar::Frame frame;
    bool got_first = false;
    try {
        got_first = sidecar::read_frame(conn.client_fd, frame);
    } catch (const std::exception &) {
        got_first = false;
    }
    CHECK(got_first && frame.type == sidecar::MsgType::TtsAudio,
          "the first TTS_AUDIO frame must arrive while generation is in flight");
    if (got_first) {
        CHECK(frame.payload == s16le_bytes(kChunkA),
              "first chunk payload must be the exact s16le bytes");
    }
    conn.session.proceed = true;  // release the fake; it pushes chunk B then flush

    // Ordering: remaining audio (chunk B, then the flush-named tail), then TTS_DONE.
    CHECK(sidecar::read_frame(conn.client_fd, frame) &&
              frame.type == sidecar::MsgType::TtsAudio &&
              frame.payload == s16le_bytes(kChunkB),
          "second chunk must follow, byte-exact");
    CHECK(sidecar::read_frame(conn.client_fd, frame) &&
              frame.type == sidecar::MsgType::TtsAudio &&
              frame.payload == s16le_bytes(kFlush),
          "the flush-named buffer must be forwarded as audio before TTS_DONE");
    CHECK(sidecar::read_frame(conn.client_fd, frame) &&
              frame.type == sidecar::MsgType::TtsDone,
          "TTS_DONE must close the request after the last audio byte");
    CHECK(!conn.session.gate_timed_out.load(),
          "generation must not have had to time out waiting for the socket read");

    // The dispatched request: text, voice reference, merged options.
    CHECK(conn.session.start_requests.size() == 1, "exactly one start_stream dispatch");
    if (!conn.session.start_requests.empty()) {
        const rt::TaskRequest & req = conn.session.start_requests.front();
        CHECK(req.text_input.has_value() && req.text_input->text == "Warp core is stable.",
              "the synthesis text must ride TaskRequest.text_input");
        const bool has_voice = req.voice.has_value() && req.voice->speaker.has_value() &&
                               req.voice->speaker->audio.has_value();
        CHECK(has_voice, "the loaded voice reference must ride TaskRequest.voice");
        if (has_voice) {
            CHECK(req.voice->speaker->audio->sample_rate == 24000 &&
                      req.voice->speaker->audio->samples.size() == 3,
                  "the voice reference must be the loader's buffer, unresampled");
        }
        CHECK(req.options.count("reference_text") != 0 &&
                  req.options.at("reference_text") == "reference words",
              "ref_text must ride options[reference_text]");
        CHECK(req.options.count("text_chunk_size") != 0 &&
                  req.options.at("text_chunk_size") == "200" &&
                  req.options.count("max_tokens") != 0 &&
                  req.options.at("max_tokens") == "500",
              "the caller's request options must be carried through");
    }
    CHECK(conn.loaded_wavs == std::vector<std::string>{"/voices/ref.wav"},
          "the WAV loader must be asked for exactly the requested voice_wav");

    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.join_or_die("full exchange");
}

// Back-to-back requests on one connection both complete (the session is
// re-armed per request exactly like dots' own start_stream-resets contract).
void test_tts_back_to_back() {
    TtsConnection conn;
    for (int i = 0; i < 2; ++i) {
        conn.send_text("request " + std::to_string(i), "/voices/ref.wav", "ref");
        std::string audio;
        std::string done_payload;
        CHECK(read_audio_until_done(conn.client_fd, audio, done_payload),
              "each back-to-back request must complete with TTS_DONE");
        CHECK(audio == full_audio_bytes(),
              "each request must deliver the full audio byte-exactly");
    }
    CHECK(conn.session.start_requests.size() == 2,
          "both requests must have been dispatched");
    if (conn.session.start_requests.size() == 2) {
        CHECK(conn.session.start_requests[0].text_input->text == "request 0" &&
                  conn.session.start_requests[1].text_input->text == "request 1",
              "each dispatch must carry its own text");
    }
    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.join_or_die("back-to-back");
}

// ref_text is mandatory: a TEXT without it, with it empty, or with an
// unloadable WAV is rejected with ERROR bad_frame, dispatches NOTHING, and leaves
// the connection alive for a corrected request.
void test_tts_bad_requests_never_dispatch() {
    TtsConnection conn;  // ref_text_required defaults true: the dots rule
    sidecar::Frame frame;

    conn.send_text("no ref text", "/voices/ref.wav", nullptr);
    CHECK(sidecar::read_frame(conn.client_fd, frame) &&
              frame.type == sidecar::MsgType::Error &&
              json_string_field(frame.payload, "code") == "bad_frame",
          "a TEXT missing ref_text must be rejected with ERROR bad_frame");

    conn.send_text("empty ref text", "/voices/ref.wav", "");
    CHECK(sidecar::read_frame(conn.client_fd, frame) &&
              frame.type == sidecar::MsgType::Error &&
              json_string_field(frame.payload, "code") == "bad_frame",
          "a TEXT with an EMPTY ref_text must be rejected with ERROR bad_frame");

    conn.send_text("wav gone", "/voices/missing.wav", "ref");
    CHECK(sidecar::read_frame(conn.client_fd, frame) &&
              frame.type == sidecar::MsgType::Error &&
              json_string_field(frame.payload, "code") == "bad_frame",
          "an unloadable voice_wav must be rejected with ERROR bad_frame");

    CHECK(conn.session.start_requests.empty(),
          "no invalid request may ever be dispatched");

    // An ASR-set frame is not valid on a TTS connection.
    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Audio, std::string(4, '\0'));
    CHECK(sidecar::read_frame(conn.client_fd, frame) &&
              frame.type == sidecar::MsgType::Error &&
              json_string_field(frame.payload, "code") == "bad_frame",
          "an ASR-set frame on a TTS connection must be rejected with ERROR bad_frame");

    // The connection survives all of the above: a corrected request completes.
    conn.send_text("now valid", "/voices/ref.wav", "ref");
    std::string audio;
    std::string done_payload;
    CHECK(read_audio_until_done(conn.client_fd, audio, done_payload) &&
              audio == full_audio_bytes(),
          "a corrected request after rejections must complete normally");
    CHECK(conn.session.start_requests.size() == 1,
          "only the corrected request may have been dispatched");

    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.join_or_die("bad requests");
}

// ref_text is per-model: under voxcpm2
// (ref_text_required=false) a TEXT frame with a missing or empty ref_text is
// ACCEPTED and dispatched with the reference_text option key OMITTED - the dots
// rejection (ref_text_required=true) is pinned by test_tts_bad_requests_never_dispatch.
void test_voxcpm2_accepts_ref_text_less_text() {
    sidecar::TtsDriveConfig config;
    TtsConnection conn({}, config, /*prepare_throws=*/false, /*ref_text_required=*/false);

    conn.send_text("no transcript", "/voices/ref.wav", nullptr);
    std::string audio;
    std::string done_payload;
    CHECK(read_audio_until_done(conn.client_fd, audio, done_payload),
          "a ref_text-less TEXT under voxcpm2 must complete with TTS_DONE");
    CHECK(audio == full_audio_bytes(),
          "the ref_text-less voxcpm2 request must synthesize byte-exactly");

    conn.send_text("empty transcript", "/voices/ref.wav", "");
    CHECK(read_audio_until_done(conn.client_fd, audio, done_payload),
          "a TEXT with an EMPTY ref_text under voxcpm2 must complete with TTS_DONE");

    CHECK(conn.session.start_requests.size() == 2,
          "both ref_text-less voxcpm2 requests must dispatch");
    for (const rt::TaskRequest & req : conn.session.start_requests) {
        CHECK(req.options.count("reference_text") == 0,
              "a ref_text-less voxcpm2 request must OMIT the reference_text option key");
        CHECK(req.text_input.has_value() && !req.text_input->text.empty(),
              "the synthesis text must still ride the voxcpm2 request");
        CHECK(req.voice.has_value() && req.voice->speaker.has_value() &&
                  req.voice->speaker->audio.has_value(),
              "the voice reference must still ride the voxcpm2 request");
    }

    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.join_or_die("voxcpm2 ref_text optional");
}

// Buffer-and-drain: with a reader that deliberately drains
// NOTHING, generation must still run to completion - the sink push lands in the
// in-process pump queue, never blocking on the socket. The pushed volume (~2 MiB)
// is far beyond any AF_UNIX socket buffer, so a sink that writes straight to the
// socket wedges mid-generation and fails the completion check. Afterwards the
// throttled reader drains everything and every byte must arrive in order, none
// dropped, with TTS_DONE after the last of them.
void test_tts_throttled_reader_never_stalls_sink() {
    TtsConnection conn;
    conn.session.bulk_chunks = 64;
    conn.session.bulk_chunk_samples = 16384;  // 64 x 32 KiB = 2 MiB of s16le audio

    conn.send_text("bulk request", "/voices/ref.wav", "ref");

    // Throttled reader: read NOTHING until generation has returned.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!conn.session.generation_done.load() &&
           std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    CHECK(conn.session.generation_done.load(),
          "generation must complete before the reader drains anything - the sink "
          "push may never block on the socket");

    // Now drain: every sink-pushed byte, in order, none dropped, then TTS_DONE.
    std::string audio;
    std::string done_payload;
    CHECK(read_audio_until_done(conn.client_fd, audio, done_payload),
          "the throttled drain must still end in TTS_DONE");
    std::string expected;
    for (int c = 0; c < conn.session.bulk_chunks; ++c) {
        for (const float sample :
             FakeTtsSession::bulk_pattern(c, conn.session.bulk_chunk_samples)) {
            const auto s = static_cast<std::int16_t>(std::lrintf(sample * 32768.0f));
            expected.push_back(static_cast<char>(s & 0xFF));
            expected.push_back(static_cast<char>((s >> 8) & 0xFF));
        }
    }
    expected += s16le_bytes(kFlush);
    CHECK(audio == expected,
          "every sink-pushed byte must arrive in order with none dropped");

    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.join_or_die("throttled reader");
}

// EOF mid-generation: the client vanishes while the fake is still generating; the
// drive path must let generation finish into the dead socket and return - no hang.
void test_tts_eof_mid_generation() {
    TtsConnection conn;
    conn.session.gate_after_first_chunk = true;

    conn.send_text("doomed request", "/voices/ref.wav", "ref");
    sidecar::Frame frame;
    CHECK(sidecar::read_frame(conn.client_fd, frame) &&
              frame.type == sidecar::MsgType::TtsAudio,
          "the first chunk must arrive before the client disconnects");

    ::close(conn.client_fd);  // peer gone mid-generation
    conn.client_fd = -1;
    conn.session.proceed = true;  // fake keeps pushing into the dead socket

    conn.join_or_die("EOF mid-generation");  // returns cleanly, no hang
    CHECK(conn.session.start_requests.size() == 1, "the request was in flight");
}

// BYE mid-generation: the in-flight reply is finished first (audio tail + TTS_DONE
// still arrive), then the BYE ends the connection - no hang.
void test_tts_bye_mid_generation() {
    TtsConnection conn;
    conn.session.gate_after_first_chunk = true;

    conn.send_text("parting request", "/voices/ref.wav", "ref");
    sidecar::Frame frame;
    CHECK(sidecar::read_frame(conn.client_fd, frame) &&
              frame.type == sidecar::MsgType::TtsAudio &&
              frame.payload == s16le_bytes(kChunkA),
          "the first chunk must arrive before BYE is sent");

    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.session.proceed = true;

    CHECK(sidecar::read_frame(conn.client_fd, frame) &&
              frame.type == sidecar::MsgType::TtsAudio &&
              frame.payload == s16le_bytes(kChunkB),
          "the in-flight reply must be finished after BYE: chunk B");
    CHECK(sidecar::read_frame(conn.client_fd, frame) &&
              frame.type == sidecar::MsgType::TtsAudio &&
              frame.payload == s16le_bytes(kFlush),
          "the in-flight reply must be finished after BYE: flush tail");
    CHECK(sidecar::read_frame(conn.client_fd, frame) &&
              frame.type == sidecar::MsgType::TtsDone,
          "the in-flight reply must be finished after BYE: TTS_DONE");

    conn.join_or_die("BYE mid-generation");  // the queued BYE then ends the loop
}

// Cancel: a CANCEL frame mid-generation makes the sink
// throw the sentinel, generation stops early (well short of its scripted length), the
// request is closed by TTS_DONE {cancelled:true}, and the SAME session then serves
// the next request byte-exactly - sink-throw session reuse, not kill-and-respawn.
void test_tts_cancel_stops_generation_and_session_survives() {
    TtsConnection conn;
    conn.session.drip_chunks = 200;  // ~1 s of dripped generation if never cancelled

    conn.send_text("long doomed request", "/voices/ref.wav", "ref");

    // The request must be streaming before the cancel lands.
    sidecar::Frame frame;
    for (int i = 0; i < 2; ++i) {
        CHECK(sidecar::read_frame(conn.client_fd, frame) &&
                  frame.type == sidecar::MsgType::TtsAudio,
              "audio must be streaming before the cancel");
    }

    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Cancel, "{}");

    // The reply must end in TTS_DONE {cancelled:true}. Audio frames already queued
    // when the cancel landed may still arrive first; anything else is a violation.
    bool got_done = false;
    std::string done_payload;
    try {
        while (sidecar::read_frame(conn.client_fd, frame)) {
            if (frame.type == sidecar::MsgType::TtsAudio) {
                continue;
            }
            if (frame.type == sidecar::MsgType::TtsDone) {
                got_done = true;
                done_payload = frame.payload;
            } else {
                CHECK(false, "only TTS_AUDIO/TTS_DONE may arrive while a cancel resolves");
            }
            break;
        }
    } catch (const std::exception &) {
    }
    CHECK(got_done, "a cancelled request must still be closed by TTS_DONE");
    CHECK(json_bool_field(done_payload, "cancelled"),
          "the closing TTS_DONE must carry cancelled:true");
    CHECK(conn.session.drip_pushed.load() < conn.session.drip_chunks,
          "the sink-throw must stop generation early, not run the full script");

    // Session reuse (the sink-throw probe verdict): the SAME warm session serves the next
    // request completely and byte-exactly - the probe's round-2 claim, model-free.
    conn.session.drip_chunks = 0;
    conn.send_text("follow-up request", "/voices/ref.wav", "ref");
    std::string audio;
    std::string done2;
    CHECK(read_audio_until_done(conn.client_fd, audio, done2) &&
              audio == full_audio_bytes(),
          "the session must stay warm and serve the next request byte-exactly");
    CHECK(!json_bool_field(done2, "cancelled"),
          "a completed follow-up must not be marked cancelled");
    CHECK(conn.session.start_requests.size() == 2,
          "both requests must have dispatched on the ONE session");

    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.join_or_die("cancel mid-generation");
}

// A stray CANCEL with no request in flight (the client cancelled just as the request
// finished) is ignored - not answered with ERROR - and the next request is unharmed.
void test_tts_stray_cancel_between_requests_is_ignored() {
    TtsConnection conn;
    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Cancel, "{}");
    conn.send_text("after stray cancel", "/voices/ref.wav", "ref");
    std::string audio;
    std::string done_payload;
    CHECK(read_audio_until_done(conn.client_fd, audio, done_payload) &&
              audio == full_audio_bytes(),
          "a stray CANCEL between requests must be ignored, not answered with ERROR");
    CHECK(conn.session.start_requests.size() == 1,
          "the request after a stray cancel must still dispatch");
    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.join_or_die("stray cancel");
}

// A session.prepare() failure is a DEFINED refusal.
// The client completed HELLO/READY and is waiting on this connection; ending it
// with bare EOF is indistinguishable from a crash. The drive path must answer
// with ERROR prepare_failed before the connection ends.
void test_tts_prepare_failure_sends_error_frame() {
    TtsConnection conn({}, {}, /*prepare_throws=*/true);
    sidecar::Frame frame;
    bool got = false;
    try {
        got = sidecar::read_frame(conn.client_fd, frame);
    } catch (const std::exception &) {
        got = false;
    }
    CHECK(got, "a prepare() failure must be answered with a frame, not bare EOF");
    if (got) {
        CHECK(frame.type == sidecar::MsgType::Error,
              "the prepare-failure reply must be an ERROR frame");
        CHECK(json_string_field(frame.payload, "code") == "prepare_failed",
              "the prepare-failure ERROR must carry the refusal code prepare_failed");
    }
    // After the refusal the connection ends cleanly (EOF at a frame boundary).
    bool eof = false;
    try {
        eof = !sidecar::read_frame(conn.client_fd, frame);
    } catch (const std::exception &) {
        eof = true;
    }
    CHECK(eof, "the connection must end after the prepare-failure ERROR");
    CHECK(conn.session.start_requests.empty(),
          "nothing may be dispatched on a session whose prepare failed");
    conn.join_or_die("prepare failure");
}

// The drain loop's poll-then-read gap. read_frame blocks
// until a whole frame has arrived, so a client that splits its 7-byte CANCEL
// across two writes makes the drain wait out the gap between them - and nothing
// worse: the cancel still lands, generation still stops early, and the session
// still serves the next request. This pins the harmlessness the drive path's
// comment asserts (a well-behaved client sends a control frame in ONE write,
// which AF_UNIX SOCK_STREAM delivers intact).
void test_tts_cancel_split_across_writes_still_lands() {
    TtsConnection conn;
    conn.session.drip_chunks = 200;  // ~1 s of dripped generation if never cancelled

    conn.send_text("long doomed request", "/voices/ref.wav", "ref");

    sidecar::Frame frame;
    for (int i = 0; i < 2; ++i) {
        CHECK(sidecar::read_frame(conn.client_fd, frame) &&
                  frame.type == sidecar::MsgType::TtsAudio,
              "audio must be streaming before the split cancel");
    }

    // The CANCEL frame's exact 7 wire bytes ([type u8][len u32 BE] + "{}"),
    // deliberately split mid-HEADER across two writes with a real gap between.
    const unsigned char wire[7] = {
        static_cast<unsigned char>(sidecar::MsgType::Cancel), 0x00, 0x00, 0x00, 0x02,
        '{', '}'};
    auto send_all = [&](const unsigned char * data, std::size_t n) {
        std::size_t off = 0;
        while (off < n) {
            const ssize_t w = ::send(conn.client_fd, data + off, n - off, 0);
            CHECK(w > 0, "split-cancel send must succeed");
            if (w <= 0) {
                return;
            }
            off += static_cast<std::size_t>(w);
        }
    };
    send_all(wire, 3);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    send_all(wire + 3, 4);

    bool got_done = false;
    std::string done_payload;
    try {
        while (sidecar::read_frame(conn.client_fd, frame)) {
            if (frame.type == sidecar::MsgType::TtsAudio) {
                continue;
            }
            if (frame.type == sidecar::MsgType::TtsDone) {
                got_done = true;
                done_payload = frame.payload;
            } else {
                CHECK(false, "only TTS_AUDIO/TTS_DONE may arrive while a cancel resolves");
            }
            break;
        }
    } catch (const std::exception &) {
    }
    CHECK(got_done, "a split-write CANCEL must still close the request with TTS_DONE");
    CHECK(json_bool_field(done_payload, "cancelled"),
          "the closing TTS_DONE must carry cancelled:true");
    CHECK(conn.session.drip_pushed.load() < conn.session.drip_chunks,
          "the split-write CANCEL must still stop generation early");

    // The session survives the split cancel exactly like a whole-frame one.
    conn.session.drip_chunks = 0;
    conn.send_text("follow-up request", "/voices/ref.wav", "ref");
    std::string audio;
    std::string done2;
    CHECK(read_audio_until_done(conn.client_fd, audio, done2) &&
              audio == full_audio_bytes(),
          "the session must serve the next request after a split-write cancel");

    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.join_or_die("split-write cancel");
}

// --- Seam-silence trim ---------------------------------------------------------

std::string s16le_of(const std::vector<float> & samples) {
    // The drive path's exact float -> s16le encoding, for byte-exact expectations.
    std::string out;
    for (const float f : samples) {
        long v = std::lrintf(f * 32768.0f);
        if (v > 32767) {
            v = 32767;
        } else if (v < -32768) {
            v = -32768;
        }
        const auto s = static_cast<std::int16_t>(v);
        out.push_back(static_cast<char>(s & 0xFF));
        out.push_back(static_cast<char>((s >> 8) & 0xFF));
    }
    return out;
}

std::vector<float> silence(std::size_t n) {
    return std::vector<float>(n, 0.0f);
}

std::vector<float> cat(const std::vector<std::vector<float>> & parts) {
    std::vector<float> out;
    for (const auto & part : parts) {
        out.insert(out.end(), part.begin(), part.end());
    }
    return out;
}

// A silence run that crosses a segment-id change is capped to the configured
// maximum - tail-side, head-side, or both - while every voiced sample and the
// request-tail silence arrive byte-exactly.
void test_seam_silence_crossing_boundary_is_capped() {
    sidecar::TtsDriveConfig config;
    config.seam_silence_cap_samples = 4;
    TtsConnection conn({}, config);
    const auto voiced_a = floats_from_s16({1000, -2000, 3000});
    const auto voiced_b = floats_from_s16({4000, -4000});
    const auto voiced_c = floats_from_s16({5000, -5000, 6000});
    conn.session.scripted = {
        {"segment_0_chunk_0", cat({voiced_a, silence(6)})},
        {"segment_1_chunk_0", cat({silence(6), voiced_b})},  // 12-sample seam run
        {"segment_2_chunk_0", cat({silence(8), voiced_c})},  // head-only 8-sample run
        {"segment_2_chunk_1", silence(5)},                   // request tail, no seam
    };
    conn.send_text("seam", "/voices/ref.wav", "ref");
    std::string audio;
    std::string done_payload;
    CHECK(read_audio_until_done(conn.client_fd, audio, done_payload),
          "the seam-trimmed request must complete with TTS_DONE");
    const std::string expected = s16le_of(
        cat({voiced_a, silence(4), voiced_b, silence(4), voiced_c, silence(5)}));
    CHECK(audio == expected,
          "both seam runs must be capped to 4 samples; voiced audio and the "
          "request tail must arrive byte-exactly");
    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.join_or_die("seam cap");
}

// Silence that stays inside ONE segment - an in-chunk pause, however long - is
// never trimmed, and a boundary with no silence loses nothing.
void test_within_segment_silence_never_trimmed() {
    sidecar::TtsDriveConfig config;
    config.seam_silence_cap_samples = 4;
    TtsConnection conn({}, config);
    const auto voiced_a = floats_from_s16({1000, -2000, 3000});
    const auto voiced_b = floats_from_s16({4000, -4000});
    const auto voiced_c = floats_from_s16({5000, -5000, 6000});
    conn.session.scripted = {
        {"segment_0_chunk_0", cat({voiced_a, silence(10), voiced_b})},  // in-buffer pause
        {"segment_0_chunk_1", silence(10)},                             // cross-buffer pause
        {"segment_0_chunk_2", voiced_c},                                // same segment
        {"segment_1_chunk_0", voiced_a},  // boundary with no silence: nothing to trim
    };
    conn.send_text("pauses", "/voices/ref.wav", "ref");
    std::string audio;
    std::string done_payload;
    CHECK(read_audio_until_done(conn.client_fd, audio, done_payload),
          "the in-chunk-pause request must complete with TTS_DONE");
    const std::string expected = s16le_of(
        cat({voiced_a, silence(10), voiced_b, silence(10), voiced_c, voiced_a}));
    CHECK(audio == expected,
          "within-segment silence must pass through whole; a silence-free "
          "boundary must lose nothing");
    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.join_or_die("in-chunk pauses");
}

// Request-leading silence - before ANY voiced sample - can never be a seam, so
// the trimmer must stream it through immediately, never holding it back until
// voice arrives: holding would delay the client's first frame for nothing.
void test_request_leading_silence_streams_immediately() {
    sidecar::TtsDriveConfig config;
    config.seam_silence_cap_samples = 4;
    TtsConnection conn({}, config);
    const auto voiced_a = floats_from_s16({1000, -2000, 3000});
    conn.session.scripted = {
        {"segment_0_chunk_0", silence(7)},  // model lead-in, longer than the cap
        {"segment_0_chunk_1", voiced_a},
    };
    conn.send_text("lead-in", "/voices/ref.wav", "ref");
    // Framing carries the property: a held lead-in would merge into the first
    // voiced frame, so the FIRST frame must be exactly the silence buffer.
    sidecar::Frame frame;
    CHECK(sidecar::read_frame(conn.client_fd, frame) &&
              frame.type == sidecar::MsgType::TtsAudio &&
              frame.payload == s16le_of(silence(7)),
          "the pure-silence lead-in buffer must be forwarded as its own frame, "
          "not held until the first voiced sample");
    std::string audio;
    std::string done_payload;
    CHECK(read_audio_until_done(conn.client_fd, audio, done_payload),
          "the lead-in request must complete with TTS_DONE");
    CHECK(audio == s16le_of(voiced_a), "the voiced audio must follow byte-exactly");
    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.join_or_die("leading silence");
}

// A cap of 0 (the TtsDriveConfig default) disables the trim: everything passes
// through bit-exactly, boundary silence included.
void test_seam_trim_disabled_by_zero_cap() {
    TtsConnection conn;  // default TtsDriveConfig: trim disabled
    const auto voiced_a = floats_from_s16({1000, -2000, 3000});
    const auto voiced_b = floats_from_s16({4000, -4000});
    conn.session.scripted = {
        {"segment_0_chunk_0", cat({voiced_a, silence(6)})},
        {"segment_1_chunk_0", cat({silence(6), voiced_b})},
        {"segment_1_chunk_1", silence(5)},
    };
    conn.send_text("no trim", "/voices/ref.wav", "ref");
    std::string audio;
    std::string done_payload;
    CHECK(read_audio_until_done(conn.client_fd, audio, done_payload),
          "the untrimmed request must complete with TTS_DONE");
    const std::string expected = s16le_of(
        cat({voiced_a, silence(6), silence(6), voiced_b, silence(5)}));
    CHECK(audio == expected, "a zero cap must pass every sample through untouched");
    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.join_or_die("trim disabled");
}

// The seam trimmer is a no-op for voxcpm2 by construction.
// voxcpm2 names its sink buffers "chunk_N" (session.cpp: named.id = "chunk_" +
// index, no segment concept), so parse_segment returns nothing, the trimmer can
// never open a seam, and every sample - silence included - passes through
// byte-identical even with the cap ENABLED. Naming-driven, never a family check.
void test_voxcpm2_chunk_naming_passes_through_trim() {
    sidecar::TtsDriveConfig config;
    config.seam_silence_cap_samples = 4;  // trimmer ENABLED: would cap a dots seam
    TtsConnection conn({}, config);
    const auto voiced_a = floats_from_s16({1000, -2000, 3000});
    const auto voiced_b = floats_from_s16({4000, -4000});
    conn.session.scripted = {
        {"chunk_0", cat({voiced_a, silence(6)})},
        {"chunk_1", cat({silence(6), voiced_b})},   // silence run spans a chunk boundary
        {"chunk_2", cat({silence(8), voiced_a})},   // head-only silence run
        {"chunk_3", silence(5)},                    // trailing silence
    };
    conn.send_text("voxcpm2 pass-through", "/voices/ref.wav", "ref");
    std::string audio;
    std::string done_payload;
    CHECK(read_audio_until_done(conn.client_fd, audio, done_payload),
          "the voxcpm2-named request must complete with TTS_DONE");
    const std::string expected = s16le_of(
        cat({voiced_a, silence(6), silence(6), voiced_b, silence(8), voiced_a, silence(5)}));
    CHECK(audio == expected,
          "chunk_N carries no segment id, so the trimmer must pass every sample "
          "through byte-identical - silence runs untouched despite the cap");
    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.join_or_die("voxcpm2 chunk_N pass-through");
}

bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// The live seam ear test's over-budget sentence (211 codepoints > the pinned
// text_chunk_size=200) and the full long-form passage it came from - the exact
// text of tts_evidence_longform.wav, so the chunk assertions below describe the
// seams heard in that ear test.
const std::string kOverBudgetSentence =
    "Tomorrow we make orbit around the third planet, a cold world of shallow "
    "seas and long grey beaches, where the probes have detected geometric "
    "formations that may be the remains of an ancient shoreline settlement.";

const std::string kLongformPassage =
    "Captain's log, stardate four seven six three point nine. We have entered the "
    "Veridian system after a six day crossing at warp seven, and the ship has "
    "settled into the easy rhythm of survey work. The science teams are mapping "
    "the second planet's magnetosphere while engineering runs a full diagnostic "
    "on the port nacelle, which has been reading two percent warm since the "
    "nebula passage. Counselor Reyes reports that morale is high, helped in no "
    "small part by the hydroponics bay's first successful tomato harvest. "
    "Tomorrow we make orbit around the third planet, a cold world of shallow "
    "seas and long grey beaches, where the probes have detected geometric "
    "formations that may be the remains of an ancient shoreline settlement. If "
    "the away team confirms artificial origin, this system will be referred to "
    "the archaeology council for a full expedition. For now the bridge is quiet, "
    "the stars are steady, and the ship feels very much like home.";

// The dots registry request-option defaults carry the
// chunk-mode retune, and the merged defaults ride EVERY dispatched TTS request -
// exactly the map main.cpp hands drive_tts_session for this model_task.
void test_registry_chunk_mode_rides_every_dispatch() {
    const sidecar::TaskRegistry registry = sidecar::TaskRegistry::default_registry();
    const sidecar::TaskDescriptor * dots = registry.find("dots-tts-streaming");
    CHECK(dots != nullptr, "the dots TTS model_task must resolve");
    if (dots == nullptr) {
        return;
    }
    TtsConnection conn(sidecar::merge_options(dots->request_options, {}));
    for (int i = 0; i < 2; ++i) {
        conn.send_text("request " + std::to_string(i), "/voices/ref.wav", "ref");
        std::string audio;
        std::string done_payload;
        CHECK(read_audio_until_done(conn.client_fd, audio, done_payload),
              "each request under registry defaults must complete with TTS_DONE");
    }
    CHECK(conn.session.start_requests.size() == 2, "both requests must dispatch");
    for (const rt::TaskRequest & req : conn.session.start_requests) {
        CHECK(req.options.count("text_chunk_mode") != 0 &&
                  req.options.at("text_chunk_mode") == "default",
              "text_chunk_mode=default must ride every dispatched request");
        CHECK(req.options.count("text_chunk_size") != 0 &&
                  req.options.at("text_chunk_size") == "200" &&
                  req.options.count("max_tokens") != 0 &&
                  req.options.at("max_tokens") == "500",
              "the pinned long-form guards must still ride alongside the mode");
    }
    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.join_or_die("registry chunk mode");
}

// Against the REAL vendored chunker (pinned bb15edd7), the
// registry's mode cuts the ear test's over-budget sentence at its clause break - no
// mid-phrase piece, no one-word chunk - while under-budget sentences keep their
// sentence-boundary seams.
void test_chunk_mode_default_cuts_at_clause_breaks() {
    namespace et = engine::text;

    // The registry's option string must reach et::TextChunkMode::Default through
    // the same parse dots' own request path applies to TaskRequest.options.
    const sidecar::TaskRegistry registry = sidecar::TaskRegistry::default_registry();
    const sidecar::TaskDescriptor * dots = registry.find("dots-tts-streaming");
    CHECK(dots != nullptr, "the dots TTS model_task must resolve");
    if (dots == nullptr) {
        return;
    }
    const auto mode = et::parse_text_chunk_mode_override(dots->request_options);
    CHECK(mode.has_value() && *mode == et::TextChunkMode::Default,
          "the registry defaults must parse to TextChunkMode::Default");

    // The over-budget sentence backtracks to the clause break after "beaches,":
    // two whole-clause chunks, every word attached to its phrase.
    const auto clauses =
        et::split_text_chunks(kOverBudgetSentence, 200, et::TextChunkMode::Default);
    CHECK(clauses.size() == 2, "the 211-codepoint sentence must split into two clauses");
    if (clauses.size() == 2) {
        CHECK(ends_with(clauses[0], "long grey beaches,"),
              "the cut must land at the clause boundary after beaches,");
        CHECK(clauses[1] ==
                  "where the probes have detected geometric formations that may be "
                  "the remains of an ancient shoreline settlement.",
              "the second chunk must be the whole trailing clause");
    }
    for (const auto & chunk : clauses) {
        CHECK(chunk.find(' ') != std::string::npos, "no one-word chunk may remain");
    }

    // The defect this retunes away, pinned as characterization: tag_aware packs
    // words with no clause backtracking, stranding the one-word "settlement."
    // chunk behind a mid-noun-phrase seam (the obvious ear-test seam at ~41-42 s).
    const auto tag_aware =
        et::split_text_chunks(kOverBudgetSentence, 200, et::TextChunkMode::TagAware);
    CHECK(!tag_aware.empty() && tag_aware.back() == "settlement.",
          "characterization: tag_aware strands the one-word chunk the retune removes");

    // The full long-form passage: under-budget sentences pack whole into chunks with
    // seams at sentence ends (the first chunk carries its first TWO sentences),
    // the over-budget sentence contributes its clause pair, and every seam in
    // the clip lands on punctuation.
    const auto passage =
        et::split_text_chunks(kLongformPassage, 200, et::TextChunkMode::Default);
    CHECK(passage.size() == 8, "the long-form passage must chunk to 8 punctuation-bounded pieces");
    CHECK(!passage.empty() && ends_with(passage.front(), "survey work."),
          "under-budget sentences must keep their sentence-boundary seam");
    for (const auto & chunk : passage) {
        const char last = chunk.empty() ? '\0' : chunk.back();
        CHECK(last == '.' || last == ',' || last == '!' || last == '?',
              "every seam in the passage must land on punctuation");
        CHECK(chunk.find(' ') != std::string::npos, "no one-word chunk in the passage");
    }
    if (passage.size() == 8) {
        CHECK(ends_with(passage[4], "long grey beaches,") &&
                  ends_with(passage[5], "shoreline settlement."),
              "the over-budget sentence must appear as its adjacent clause pair");
    }
}

// A PullEvents session. The drive must
// read the kind from streaming_policy(), run start_stream, then pull
// next_stream_event on the worker thread until nullopt, forwarding every chunk in
// order as TTS_AUDIO through the existing trimmer and pump, then one TTS_DONE.
void test_pull_events_session_is_pulled_in_order() {
    constexpr int kChunks = 5;
    constexpr int kSamples = 37;
    TtsConnection conn;
    conn.session.output_kind = rt::StreamingOutputKind::PullEvents;
    conn.session.pull_chunks = kChunks;
    conn.session.pull_chunk_samples = kSamples;

    conn.send_text("Pulled, not pushed.", "/voices/ref.wav", "reference words");

    sidecar::Frame frame;
    int audio_frames = 0;
    bool in_order = true;
    bool got_done = false;
    for (;;) {
        bool got = false;
        try {
            got = sidecar::read_frame(conn.client_fd, frame);
        } catch (const std::exception &) {
            got = false;
        }
        if (!got) {
            break;
        }
        if (frame.type == sidecar::MsgType::TtsAudio) {
            std::string expected;
            for (const float f : FakeTtsSession::bulk_pattern(audio_frames, kSamples)) {
                const auto v = static_cast<std::int16_t>(f * 32768.0f);
                expected.push_back(static_cast<char>(v & 0xFF));
                expected.push_back(static_cast<char>((v >> 8) & 0xFF));
            }
            in_order = in_order && (frame.payload == expected);
            ++audio_frames;
            continue;
        }
        got_done = (frame.type == sidecar::MsgType::TtsDone);
        break;
    }
    CHECK(audio_frames == kChunks, "every pulled chunk must arrive as its own TTS_AUDIO frame");
    CHECK(in_order, "chunk payloads must arrive in pull order, byte-exact");
    CHECK(got_done, "exactly one TTS_DONE must follow the last pulled chunk");
    CHECK(conn.session.pull_calls.load() == kChunks + 1,
          "the drive must pull until next_stream_event returns nullopt");
    CHECK(conn.session.start_requests.size() == 1, "exactly one start_stream dispatch");

    // A second request on the same connection is pulled again from chunk 0.
    conn.send_text("Again.", "/voices/ref.wav", "reference words");
    std::string audio;
    std::string done_payload;
    CHECK(read_audio_until_done(conn.client_fd, audio, done_payload),
          "a second pulled request must complete with TTS_DONE");
    CHECK(audio.size() == static_cast<std::size_t>(kChunks * kSamples * 2),
          "the second request must deliver every chunk again");
    CHECK(conn.session.pull_calls.load() == 2 * (kChunks + 1),
          "the second request must pull until nullopt again");

    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.join_or_die("pull-events");
}

// CANCEL during a PullEvents request. The
// drive stops after the in-flight pull returns - no further pull, finish_stream never
// called, queued audio discarded, TTS_DONE {cancelled:true} - and the next TEXT on
// the same session completes normally.
void test_pull_events_cancel_stops_after_in_flight_pull() {
    constexpr int kChunks = 5;
    constexpr int kSamples = 37;
    TtsConnection conn;
    conn.session.output_kind = rt::StreamingOutputKind::PullEvents;
    conn.session.pull_chunks = kChunks;
    conn.session.pull_chunk_samples = kSamples;
    conn.session.pull_gate_at_chunk = 1;  // chunk 1's pull is the in-flight one

    conn.send_text("Cancel me mid-pull.", "/voices/ref.wav", "reference words");

    // Chunk 0 is on the wire while the pull of chunk 1 is parked on the gate.
    sidecar::Frame frame;
    CHECK(sidecar::read_frame(conn.client_fd, frame) && frame.type == sidecar::MsgType::TtsAudio,
          "chunk 0 must stream before the cancel");

    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Cancel, "{}");
    // Give the drain loop's fd poll (10 ms cadence) time to arm the cancel before the
    // in-flight pull is released; the assertions below prove the arming took effect.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    conn.session.proceed = true;

    bool got_done = false;
    std::string done_payload;
    int audio_after_cancel = 0;
    try {
        while (sidecar::read_frame(conn.client_fd, frame)) {
            if (frame.type == sidecar::MsgType::TtsAudio) {
                ++audio_after_cancel;
                continue;
            }
            if (frame.type == sidecar::MsgType::TtsDone) {
                got_done = true;
                done_payload = frame.payload;
            } else {
                CHECK(false, "only TTS_AUDIO/TTS_DONE may arrive while a cancel resolves");
            }
            break;
        }
    } catch (const std::exception &) {
    }
    CHECK(got_done, "a cancelled pull request must still be closed by TTS_DONE");
    CHECK(json_bool_field(done_payload, "cancelled"), "TTS_DONE must carry cancelled:true");
    CHECK(audio_after_cancel == 0, "queued audio must be discarded once the cancel is armed");
    CHECK(!conn.session.gate_timed_out.load(), "the in-flight pull must have been released");
    CHECK(conn.session.pull_calls.load() == 2,
          "the pull count must stop at the in-flight chunk - no further pull after CANCEL");
    CHECK(conn.session.finish_stream_calls.load() == 0, "finish_stream must never be called");

    // The next TEXT on the same session completes normally, pulled from chunk 0.
    conn.session.pull_gate_at_chunk = -1;
    conn.send_text("Follow-up.", "/voices/ref.wav", "reference words");
    std::string audio;
    std::string done2;
    CHECK(read_audio_until_done(conn.client_fd, audio, done2),
          "the follow-up pull request must complete with TTS_DONE");
    CHECK(audio.size() == static_cast<std::size_t>(kChunks * kSamples * 2),
          "the follow-up must deliver every chunk");
    CHECK(!json_bool_field(done2, "cancelled"), "the follow-up must not be marked cancelled");
    CHECK(conn.session.pull_calls.load() == 2 + kChunks + 1,
          "the follow-up must pull until nullopt");
    CHECK(conn.session.finish_stream_calls.load() == 0, "finish_stream still never called");
    CHECK(conn.session.start_requests.size() == 2, "both requests dispatched on the ONE session");

    sidecar::write_frame(conn.client_fd, sidecar::MsgType::Bye, "{}");
    conn.join_or_die("pull-events cancel");
}

}  // namespace

int run_tts_drive_tests() {
    test_tts_full_exchange();
    test_tts_back_to_back();
    test_tts_bad_requests_never_dispatch();
    test_voxcpm2_accepts_ref_text_less_text();
    test_tts_throttled_reader_never_stalls_sink();
    test_tts_eof_mid_generation();
    test_tts_bye_mid_generation();
    test_tts_cancel_stops_generation_and_session_survives();
    test_pull_events_session_is_pulled_in_order();
    test_pull_events_cancel_stops_after_in_flight_pull();
    test_tts_stray_cancel_between_requests_is_ignored();
    test_tts_prepare_failure_sends_error_frame();
    test_tts_cancel_split_across_writes_still_lands();
    test_registry_chunk_mode_rides_every_dispatch();
    test_chunk_mode_default_cuts_at_clause_breaks();
    test_seam_silence_crossing_boundary_is_capped();
    test_within_segment_silence_never_trimmed();
    test_request_leading_silence_streams_immediately();
    test_seam_trim_disabled_by_zero_cap();
    test_voxcpm2_chunk_naming_passes_through_trim();
    return 0;
}
