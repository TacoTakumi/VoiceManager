// Resolve a wire model/task descriptor to a generic audio.cpp streaming session.
//
// This is the ONE place that touches audio.cpp's model registry. It turns a
// `TaskDescriptor` (the lowercase spelling of audio.cpp's taxonomy, keyed by the wire
// model_task id) into an `IStreamingVoiceTaskSession` via the generic path:
//
//   make_default_registry()
//     -> registry.load({model_path, family_hint = descriptor.family})
//     -> model->create_task_session({parse(task), parse(mode)}, options)
//     -> dynamic_cast<IStreamingVoiceTaskSession *>
//
// No concrete session class name (e.g. the nemotron streaming-session class) appears
// here or in the drive loop; the descriptor's string fields select the model. Adding a
// second model+task is a registry entry plus a payload, not a code change to this resolver.
#pragma once

#include "task_registry.h"

#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"

#include <memory>
#include <string>

namespace sidecar {

// Backend/runtime knobs the sidecar resolves from its CLI (never baked into the
// descriptor, so no machine-specific path or device is committed).
struct EngineRuntimeConfig {
    std::string model_path;      // resolved GGUF path (--model)
    std::string backend = "cpu"; // "cpu" | "cuda" | "vulkan" | "metal" | "auto"
    int device = 0;              // backend device index (--device)
    int threads = 4;             // inference threads (--threads)
    // Operator-level overrides: merged over the registry entry's per-model
    // defaults, overrides winning, then forwarded opaquely.
    OptionMap session_options;   // --session-option KEY=VALUE (repeatable)
    OptionMap request_options;   // --request-option KEY=VALUE (repeatable)
};

// A loaded model + its streaming session, held warm for the life of a connection. The
// model (which owns the weights) must outlive the session, so both are kept here; the
// registry is retained too since it is cheap and keeps the loaders alive. `streaming`
// is a non-owning view into `session` (stable across moves - it points at the heap
// object the unique_ptr owns).
struct StreamingSession {
    engine::runtime::ModelRegistry registry;
    std::unique_ptr<engine::runtime::ILoadedVoiceModel> model;
    std::unique_ptr<engine::runtime::IVoiceTaskSession> session;
    engine::runtime::IStreamingVoiceTaskSession * streaming = nullptr;
};

// Loads the model named by `task` and creates its streaming session under `config`.
// Throws std::runtime_error if the model fails to load or the resolved task session does
// not support streaming execution.
StreamingSession open_streaming_session(const TaskDescriptor & task, const EngineRuntimeConfig & config);

}  // namespace sidecar
