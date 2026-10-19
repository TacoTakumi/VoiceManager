// audio.cpp streaming sidecar - process boundary for the Nemotron-3.5 STT engine.
//
// The sidecar binds an AF_UNIX socket, accepts a connection, reads a HELLO carrying the
// wire model/task id, validates it against the one-entry task registry, and -
// on a valid HELLO - loads the selected model and creates its streaming session before
// replying READY (so READY means the session is warm). It then runs the generic drive
// loop over IStreamingVoiceTaskSession: AUDIO chunks in, interim EVENT deltas and
// exactly one FINAL per FLUSH out. A process-fatal fault is contained to this
// process and surfaced to the client as a best-effort ERROR `sidecar_fault`. See
// PROTOCOL.md.

#include "drive_loop.h"
#include "engine_session.h"
#include "frame.h"
#include "protocol.h"
#include "task_registry.h"
#include "tts_drive.h"

#include "engine/framework/audio/wav_reader.h"

#include "cJSON.h"

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

std::atomic<bool> g_stop{false};

// Self-pipe that wakes main's poll() on SIGTERM/SIGINT. The kernel may hand
// a process-directed signal to ANY thread, and a session thread's read loop swallows
// EINTR (frame.cpp read_exact) - so setting g_stop alone is not guaranteed to wake the
// accept loop. write() is async-signal-safe; the write end is non-blocking so a full
// pipe can never block the handler (the byte only needs to be there once). Written
// before the handlers are installed and never changed after, so the handler's read of
// the fd is race-free.
int g_stop_pipe[2] = {-1, -1};

void on_signal(int) {
    g_stop.store(true);
    if (g_stop_pipe[1] >= 0) {
        const char byte = 1;
        const ssize_t rc = ::write(g_stop_pipe[1], &byte, 1);
        (void) rc;
    }
}

// --- One streaming session per process -------------------------------------
//
// This process serves exactly ONE session at a time, and that is deliberate rather than
// a limitation to work around. The process boundary is what contains a process-fatal
// audio.cpp fault (a GGML_ASSERT -> abort() on a CUDA alloc, a SIGSEGV): such a fault
// kills the whole process, so every session sharing it dies together. Containment requires
// a fault to leave *concurrent sessions unaffected*, which holds only while a fault can
// reach exactly one session - so VoiceManager gives every /ws/stream session its own
// sidecar on its own private socket path (pipeline/audiocpp_stt.py "owned mode").
//
// The limit used to be implicit and hostile: handle_connection ran INLINE in the accept
// loop, so a second client completed the kernel-level connect via the listen backlog and
// then sat unanswered until its read timed out 120 s later, having transcribed nothing.
// Now the session runs on its own thread while this lock admits
// one at a time, and a concurrent second connection is refused IMMEDIATELY with a defined
// ERROR `busy`. Exactly one thread ever drives audio.cpp, so no concurrent use of the
// native runtime is introduced.
std::mutex g_session_lock;

// --- Deterministic shutdown ------------------------------------------------
//
// Session threads are joinable and joined before main returns, so a clean SIGTERM can
// never run global destructors (exit) underneath a thread still holding live
// audio.cpp/CUDA state - that race ended in the fatal-signal handler reporting a
// SPURIOUS ERROR sidecar_fault for an operator-initiated shutdown. To make the
// join prompt when a client is still attached, main shutdown()s the live connection,
// which forces the session thread's blocking read to EOF and ends the session through
// its normal path. The pair below closes the register/shutdown race: either the
// session thread registers first (main's sweep sees the fd and shuts it down) or main
// flips g_shutting_down first (the thread sees it and refuses to start a session).
std::mutex g_conn_mutex;
int g_live_conn = -1;          // fd of the live session's connection; guarded by g_conn_mutex
bool g_shutting_down = false;  // guarded by g_conn_mutex

// --- Process-fatal-fault handling ------------------------------------------
//
// The audio.cpp runtime is native CUDA code whose fatal faults (a GGML_ASSERT ->
// abort() on a CUDA alloc, a SIGSEGV, ...) cannot be caught and recovered. The process
// boundary is the crash isolation: such a fault dies here in the sidecar and never
// reaches the VoiceManager server. On the way down we make one best-effort, async-
// signal-safe attempt to hand the connected client a defined ERROR `sidecar_fault`
// frame, so it observes an explicit fault signal rather than only a bare socket EOF -
// then we re-raise so the process still dies with the real fault's exit status.

// fd of the currently-served connection, or -1 between connections. Read from the signal
// handler, so it is a plain sig_atomic_t (not std::atomic) per async-signal-safety rules.
volatile std::sig_atomic_t g_conn_fd = -1;

// The ERROR `sidecar_fault` frame, serialized once at startup into a fixed buffer so the
// handler can emit it with a single write() - no allocation, no cJSON (neither is
// async-signal-safe). Wire format matches frame.cpp: 5-byte header + JSON payload.
char g_fault_frame[256];
std::size_t g_fault_frame_len = 0;

void init_fault_frame() {
    static const char payload[] =
        "{\"ok\":false,\"code\":\"sidecar_fault\","
        "\"message\":\"sidecar hit a process-fatal fault\"}";
    const std::size_t plen = sizeof(payload) - 1;  // drop the trailing NUL
    g_fault_frame[0] = static_cast<char>(static_cast<std::uint8_t>(sidecar::MsgType::Error));
    g_fault_frame[1] = static_cast<char>((plen >> 24) & 0xFF);
    g_fault_frame[2] = static_cast<char>((plen >> 16) & 0xFF);
    g_fault_frame[3] = static_cast<char>((plen >> 8) & 0xFF);
    g_fault_frame[4] = static_cast<char>(plen & 0xFF);
    std::memcpy(g_fault_frame + sidecar::kFrameHeaderBytes, payload, plen);
    g_fault_frame_len = sidecar::kFrameHeaderBytes + plen;
}

void on_fatal_signal(int sig) {
    // Best-effort: a partial/failed write is ignored - the client still gets EOF a moment
    // later when this process exits. (If the fault struck mid-frame-write the bytes may
    // interleave, but the dominant case - a fault during compute - is a clean append.)
    const int fd = g_conn_fd;
    if (fd >= 0 && g_fault_frame_len > 0) {
        const ssize_t rc = ::write(fd, g_fault_frame, g_fault_frame_len);
        (void) rc;
    }
    // Re-raise with the default disposition so the process still dies from `sig`: the fault
    // stays contained to this process and the exit status reflects the real fault.
    ::signal(sig, SIG_DFL);
    ::raise(sig);
}

void install_fatal_handlers() {
    struct sigaction sa{};
    sa.sa_handler = on_fatal_signal;
    ::sigemptyset(&sa.sa_mask);
    // SA_RESETHAND + SA_NODEFER: after the handler runs the disposition is default and the
    // signal is unblocked, so the re-raise terminates immediately with the real signal.
    sa.sa_flags = SA_RESETHAND | SA_NODEFER;
    for (int sig : {SIGSEGV, SIGABRT, SIGBUS, SIGFPE, SIGILL}) {
        ::sigaction(sig, &sa, nullptr);
    }
}

void log_line(const char * level, const std::string & msg) {
    std::fprintf(stderr, "[sidecar] %s: %s\n", level, msg.c_str());
}

// --- JSON builders (cJSON) -------------------------------------------------

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

std::string quote_join(const std::vector<std::string> & items) {
    std::string out = "[";
    for (std::size_t i = 0; i < items.size(); ++i) {
        out += "\"" + items[i] + "\"";
        if (i + 1 < items.size()) {
            out += ", ";
        }
    }
    out += "]";
    return out;
}

// --- Per-connection handling ----------------------------------------------

// Completes the handshake on `fd` against `registry`, opens the selected model's
// streaming session under `engine_cfg`, and runs the drive loop until BYE/EOF. Never
// throws: I/O and protocol errors are logged and end the connection.
void handle_connection(int fd, const sidecar::TaskRegistry & registry,
                       const sidecar::EngineRuntimeConfig & engine_cfg,
                       int tts_seam_silence_ms) {
    using sidecar::Frame;
    using sidecar::MsgType;
    try {
        Frame frame;
        if (!sidecar::read_frame(fd, frame)) {
            return;  // client closed before sending anything
        }
        if (frame.type != MsgType::Hello) {
            sidecar::write_frame(fd, MsgType::Error,
                                 build_error("bad_handshake", "expected HELLO as the first frame"));
            return;
        }

        cJSON * root = cJSON_Parse(frame.payload.c_str());
        if (root == nullptr) {
            sidecar::write_frame(fd, MsgType::Error,
                                 build_error("bad_handshake", "HELLO payload is not valid JSON"));
            return;
        }

        const cJSON * proto = cJSON_GetObjectItemCaseSensitive(root, "protocol");
        const cJSON * model_task = cJSON_GetObjectItemCaseSensitive(root, "model_task");
        if (!cJSON_IsNumber(proto) || !cJSON_IsString(model_task) ||
            model_task->valuestring == nullptr) {
            cJSON_Delete(root);
            sidecar::write_frame(fd, MsgType::Error,
                                 build_error("bad_handshake",
                                             "HELLO requires numeric 'protocol' and string 'model_task'"));
            return;
        }

        const int client_proto = proto->valueint;
        const std::string requested_id = model_task->valuestring;
        cJSON_Delete(root);

        if (client_proto != sidecar::kProtocolVersion) {
            sidecar::write_frame(fd, MsgType::Error,
                                 build_error("unsupported_protocol",
                                             "sidecar speaks protocol " +
                                                 std::to_string(sidecar::kProtocolVersion)));
            return;
        }

        const sidecar::TaskDescriptor * task = registry.find(requested_id);
        if (task == nullptr) {
            sidecar::write_frame(fd, MsgType::Error,
                                 build_error("unknown_model_task",
                                             "no such model_task; known: " + quote_join(registry.ids())));
            return;
        }

        // Load the model and create its streaming session before acking, so a READY
        // genuinely means the session is warm. A load failure is reported and closes.
        sidecar::StreamingSession live;
        try {
            live = sidecar::open_streaming_session(*task, engine_cfg);
        } catch (const std::exception & ex) {
            sidecar::write_frame(fd, MsgType::Error,
                                 build_error("load_failed",
                                             std::string("could not open '") + requested_id + "': " + ex.what()));
            log_line("error", std::string("model load failed for ") + requested_id + ": " + ex.what());
            return;
        }

        sidecar::write_frame(fd, MsgType::Ready, sidecar::build_ready(*task));
        log_line("info", "handshake ok, session warm: model_task=" + requested_id);

        // Generic drive loops over IStreamingVoiceTaskSession (no concrete session
        // class), selected by the DESCRIPTOR'S TASK KIND, never the model family.
        // Request options = registry defaults overlaid by
        // --request-option overrides; one opaque merged map.
        const sidecar::OptionMap merged_request_options =
            sidecar::merge_options(task->request_options, engine_cfg.request_options);
        if (task->task == "tts") {
            // The seam-silence cap in samples at THIS descriptor's output
            // rate (48 kHz is a per-model_task value, not a constant - breeze
            // is 24 kHz, so a cap fixed at kTtsAudioSampleRate would double the
            // trimmed window there); <= 0 disables the trim. Only the TTS path reads it.
            sidecar::TtsDriveConfig tts_cfg;
            tts_cfg.seam_silence_cap_samples =
                static_cast<long long>(tts_seam_silence_ms) * task->sample_rate / 1000;
            // TEXT -> start_stream on a worker with the sink pumping TTS_AUDIO,
            // exactly one TTS_DONE per request. The reference WAV is loaded through
            // the engine's own reader so its float convention matches training.
            sidecar::drive_tts_session(
                fd, *live.streaming, merged_request_options,
                [](const std::string & path) {
                    const auto wav = engine::audio::read_wav_f32(std::filesystem::path(path));
                    return engine::runtime::AudioBuffer{wav.sample_rate, wav.channels,
                                                        wav.samples};
                },
                tts_cfg, task->ref_text_required);
        } else {
            // AUDIO -> process_audio_chunk, FLUSH -> finalize (one FINAL), RESET ->
            // reset; the merged map is applied at every start_stream including re-arms.
            sidecar::drive_streaming_session(fd, *live.streaming, merged_request_options);
        }
    } catch (const std::exception & ex) {
        log_line("warn", std::string("connection ended: ") + ex.what());
    }
}

// --- Socket setup ----------------------------------------------------------

int bind_and_listen(const std::string & socket_path) {
    if (socket_path.size() + 1 > sizeof(sockaddr_un::sun_path)) {
        throw std::runtime_error("socket path is too long for sun_path");
    }
    // A stale socket file from a previous run would make bind() fail with EADDRINUSE.
    ::unlink(socket_path.c_str());

    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error(std::string("socket() failed: ") + std::strerror(errno));
    }
    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);
    if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        const std::string err = std::strerror(errno);
        ::close(fd);
        throw std::runtime_error("bind() failed: " + err);
    }
    if (::listen(fd, 16) < 0) {
        const std::string err = std::strerror(errno);
        ::close(fd);
        ::unlink(socket_path.c_str());
        throw std::runtime_error("listen() failed: " + err);
    }
    // Non-blocking: the accept loop poll()s for readiness first, and a pending
    // connection that aborts between poll() and accept() must not stall the loop.
    const int flags = ::fcntl(fd, F_GETFL, 0);
    if (flags >= 0) {
        ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }
    return fd;
}

#ifndef SIDECAR_DEFAULT_ASSETS_ROOT
#define SIDECAR_DEFAULT_ASSETS_ROOT ""
#endif

void print_usage() {
    std::fprintf(stderr,
                 "usage: audiocpp_sidecar --socket <path> --model <gguf> [options]\n"
                 "  --socket <path>   AF_UNIX socket path to bind and listen on (required)\n"
                 "  --model <path>    model file (GGUF) to load for the streaming session (required)\n"
                 "  --backend <name>  inference backend: cpu|cuda|vulkan|metal|auto (default cpu)\n"
                 "  --device <n>      backend device index (default 0)\n"
                 "  --threads <n>     inference threads (default 4)\n"
                 "  --session-option KEY=VALUE  session-creation option override (repeatable);\n"
                 "                    merged over the model_task's registry defaults\n"
                 "  --request-option KEY=VALUE  start_stream request option override (repeatable);\n"
                 "                    merged over the model_task's registry defaults\n"
                 "  --assets-root <d> chdir here so the vendored model_specs/ resolve (default:\n"
                 "                    the audio.cpp tree the sidecar was built against)\n"
                 "  --tts-seam-silence-ms <n>  cap for a silence run crossing a TTS segment\n"
                 "                    boundary, in ms at the output rate (default 400; 0 disables)\n"
                 "  --help            show this help\n");
}

}  // namespace

int main(int argc, char ** argv) {
    std::string socket_path;
    std::string assets_root = SIDECAR_DEFAULT_ASSETS_ROOT;
    sidecar::EngineRuntimeConfig engine_cfg;
    int tts_seam_silence_ms = 400;  // 0 disables the seam trim
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        }
        if (arg == "--socket" && i + 1 < argc) {
            socket_path = argv[++i];
            continue;
        }
        if (arg == "--assets-root" && i + 1 < argc) {
            assets_root = argv[++i];
            continue;
        }
        if (arg == "--model" && i + 1 < argc) {
            engine_cfg.model_path = argv[++i];
            continue;
        }
        if (arg == "--backend" && i + 1 < argc) {
            engine_cfg.backend = argv[++i];
            continue;
        }
        if (arg == "--device" && i + 1 < argc) {
            engine_cfg.device = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--threads" && i + 1 < argc) {
            engine_cfg.threads = std::atoi(argv[++i]);
            continue;
        }
        if (arg == "--tts-seam-silence-ms" && i + 1 < argc) {
            tts_seam_silence_ms = std::atoi(argv[++i]);
            continue;
        }
        if ((arg == "--session-option" || arg == "--request-option") && i + 1 < argc) {
            const std::string pair = argv[++i];
            const std::size_t eq = pair.find('=');
            if (eq == std::string::npos || eq == 0) {
                std::fprintf(stderr, "[sidecar] error: %s expects KEY=VALUE, got '%s'\n",
                             arg.c_str(), pair.c_str());
                return 2;
            }
            auto & target = (arg == "--session-option") ? engine_cfg.session_options
                                                        : engine_cfg.request_options;
            target[pair.substr(0, eq)] = pair.substr(eq + 1);
            continue;
        }
        std::fprintf(stderr, "[sidecar] error: unknown or incomplete argument: %s\n", arg.c_str());
        print_usage();
        return 2;
    }
    if (socket_path.empty()) {
        std::fprintf(stderr, "[sidecar] error: --socket is required\n");
        print_usage();
        return 2;
    }
    if (engine_cfg.model_path.empty()) {
        std::fprintf(stderr, "[sidecar] error: --model is required\n");
        print_usage();
        return 2;
    }

    // audio.cpp resolves a model package spec (model_specs/<family>.json) by walking up
    // from cwd, so chdir into the vendored assets root before binding. Absolute --socket
    // and --model paths (the norm) are unaffected. Done before bind_and_listen so a
    // relative --socket still binds where the user meant.
    if (!assets_root.empty()) {
        if (::chdir(assets_root.c_str()) != 0) {
            std::fprintf(stderr, "[sidecar] error: could not chdir to assets root '%s': %s\n",
                         assets_root.c_str(), std::strerror(errno));
            return 1;
        }
    }

    // Writing to a peer that has closed its end must not kill the process (EPIPE
    // surfaces from write() instead and is handled per-connection).
    std::signal(SIGPIPE, SIG_IGN);

    // The stop self-pipe must exist before the handlers that write to it.
    if (::pipe(g_stop_pipe) != 0) {
        std::fprintf(stderr, "[sidecar] error: pipe() failed: %s\n", std::strerror(errno));
        return 1;
    }
    ::fcntl(g_stop_pipe[1], F_SETFL, O_NONBLOCK);

    struct sigaction sa{};
    sa.sa_handler = on_signal;  // wakes main's poll() via the self-pipe
    ::sigemptyset(&sa.sa_mask);
    ::sigaction(SIGINT, &sa, nullptr);
    ::sigaction(SIGTERM, &sa, nullptr);

    // A process-fatal fault (GGML_ASSERT/abort, SIGSEGV, ...) surfaces to the client as a
    // best-effort ERROR `sidecar_fault` before the process dies.
    init_fault_frame();
    install_fatal_handlers();

    const sidecar::TaskRegistry registry = sidecar::TaskRegistry::default_registry();

    int listen_fd = -1;
    try {
        listen_fd = bind_and_listen(socket_path);
    } catch (const std::exception & ex) {
        log_line("error", ex.what());
        return 1;
    }
    log_line("info", "listening on " + socket_path + "; registered model_task ids: " +
                         quote_join(registry.ids()));
    log_line("info", "model=" + engine_cfg.model_path + " backend=" + engine_cfg.backend +
                         " device=" + std::to_string(engine_cfg.device));

    // Worker threads are joinable and joined before main returns - but joining
    // ONLY there let finished handles pile up for the process's life. That
    // is harmless in VoiceManager's owned mode (one session per process) and unbounded in
    // an EXTERNAL sidecar under sequential reconnects, or a client retrying into busy
    // refusals: a joinable thread's 8 MiB stack is glibc's to keep until join, so every
    // served connection cost ~9.3 MiB of address space that was never given back
    // (measured, sidecar/reap_probe.py). Note the KERNEL task does exit either way, so
    // /proc/<pid>/task never showed this - only VmSize did.
    //
    // Each worker raises `done` as its very last act and the accept loop sweeps the
    // flagged ones per iteration, so a finished thread is joined at the next wakeup
    // instead of at shutdown. The join-all below is untouched and still quiesces every
    // LIVE thread before global destructors run.
    struct Worker {
        std::thread thread;
        std::shared_ptr<std::atomic<bool>> done;
    };
    std::vector<Worker> workers;

    // Join and drop every worker that has finished. Each join here is immediate: the
    // flag is only set once the thread's body has returned, so this never blocks the
    // accept loop behind a live session.
    const auto reap_finished_workers = [&workers]() {
        for (auto it = workers.begin(); it != workers.end();) {
            if (it->done->load(std::memory_order_acquire)) {
                it->thread.join();
                it = workers.erase(it);
            } else {
                ++it;
            }
        }
    };

    while (!g_stop.load()) {
        reap_finished_workers();
        pollfd fds[2];
        fds[0] = pollfd{listen_fd, POLLIN, 0};
        fds[1] = pollfd{g_stop_pipe[0], POLLIN, 0};
        if (::poll(fds, 2, -1) < 0) {
            if (errno == EINTR) {
                continue;  // signal landed on this thread: re-check g_stop
            }
            log_line("warn", std::string("poll() failed: ") + std::strerror(errno));
            continue;
        }
        if (fds[1].revents != 0) {
            break;  // stop pipe readable: SIGTERM/SIGINT landed on SOME thread
        }
        if ((fds[0].revents & POLLIN) == 0) {
            continue;
        }
        const int conn = ::accept(listen_fd, nullptr, nullptr);
        if (conn < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK ||
                errno == ECONNABORTED) {
                continue;  // spurious wakeup or the peer aborted between poll and accept
            }
            log_line("warn", std::string("accept() failed: ") + std::strerror(errno));
            continue;
        }
        // Serve on its own thread so the accept loop stays free to answer - rather than
        // strand - a second client. The registry and config are copied in.
        auto done = std::make_shared<std::atomic<bool>>(false);
        workers.push_back(Worker{
            std::thread([conn, registry, engine_cfg, tts_seam_silence_ms, done]() {
                // Raise `done` on EVERY exit path, including the two early refusals below,
                // so the accept loop's sweep can join this thread as soon as it finishes.
                // Declared first, so it is destroyed LAST - after the session lock
                // is released and the connection closed, this really is the thread's final
                // act, and a sweep that sees the flag has nothing left to wait on.
                struct DoneGuard {
                    std::shared_ptr<std::atomic<bool>> flag;
                    ~DoneGuard() { flag->store(true, std::memory_order_release); }
                } done_guard{done};
                // One session per process: a concurrent second connection is refused now,
                // with a defined code, instead of waiting out the client's read timeout.
                std::unique_lock<std::mutex> session(g_session_lock, std::try_to_lock);
                if (!session.owns_lock()) {
                    try {
                        sidecar::write_frame(
                            conn, sidecar::MsgType::Error,
                            build_error("busy",
                                        "sidecar already has a live session; this build "
                                        "serves one session per process"));
                    } catch (const std::exception &) {
                        // The peer may already be gone; the close below is the real answer.
                    }
                    log_line("warn", "refused a concurrent connection: session already live");
                    ::close(conn);
                    return;
                }
                // Register as the live connection so main's shutdown sweep can force this
                // thread's blocking read to EOF; refuse to START a session once shutdown
                // has begun, or the join below could wait out a whole client session.
                {
                    std::lock_guard<std::mutex> reg(g_conn_mutex);
                    if (g_shutting_down) {
                        ::close(conn);
                        return;
                    }
                    g_live_conn = conn;
                }
                // Expose this fd to the fatal-signal handler for the life of the
                // connection, so a process-fatal fault mid-session can emit a best-effort
                // ERROR to the client. Safe as a single global: only one session is live.
                g_conn_fd = conn;
                handle_connection(conn, registry, engine_cfg, tts_seam_silence_ms);
                g_conn_fd = -1;
                {
                    std::lock_guard<std::mutex> reg(g_conn_mutex);
                    g_live_conn = -1;
                }
                // close() only after deregistering, so main can never shut down a stale fd.
                ::close(conn);
            }),
            done});
    }

    // Deterministic quiesce: stop admitting connections, force the live
    // session (if any) to EOF, then join every worker - so no thread still holds
    // audio.cpp/CUDA state when main returns and global destructors run. A truly wedged
    // native compute is still bounded by the Python reaper's SIGKILL escalation.
    ::close(listen_fd);
    {
        std::lock_guard<std::mutex> reg(g_conn_mutex);
        g_shutting_down = true;
        if (g_live_conn >= 0) {
            ::shutdown(g_live_conn, SHUT_RDWR);
        }
    }
    for (Worker & worker : workers) {
        worker.thread.join();
    }
    ::unlink(socket_path.c_str());
    log_line("info", "shut down cleanly");
    return 0;
}
