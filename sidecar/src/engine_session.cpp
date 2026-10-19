#include "engine_session.h"

#include "engine/framework/core/module.h"

#include <stdexcept>

namespace sidecar {

namespace {

engine::core::BackendType parse_backend(const std::string & value) {
    if (value == "cpu") {
        return engine::core::BackendType::Cpu;
    }
    if (value == "cuda") {
        return engine::core::BackendType::Cuda;
    }
    if (value == "vulkan") {
        return engine::core::BackendType::Vulkan;
    }
    if (value == "metal") {
        return engine::core::BackendType::Metal;
    }
    if (value == "auto" || value == "best") {
        return engine::core::BackendType::BestAvailable;
    }
    throw std::runtime_error("unsupported backend: " + value);
}

}  // namespace

StreamingSession open_streaming_session(const TaskDescriptor & task, const EngineRuntimeConfig & config) {
    if (config.model_path.empty()) {
        throw std::runtime_error("no model path configured (pass --model)");
    }

    StreamingSession handle;
    handle.registry = engine::runtime::make_default_registry();

    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = config.model_path;
    // Descriptor drives the family; the registry auto-detects if this is empty, but
    // stating it keeps the wire id authoritative over what gets loaded.
    if (!task.family.empty()) {
        load_request.family_hint = task.family;
    }

    engine::runtime::TaskSpec task_spec{
        engine::runtime::parse_voice_task_kind(task.task),
        engine::runtime::parse_run_mode(task.mode),
    };

    engine::runtime::SessionOptions session_options;
    session_options.backend.type = parse_backend(config.backend);
    session_options.backend.device = config.device;
    session_options.backend.threads = config.threads > 0 ? config.threads : 1;
    // Per-model registry defaults overlaid by operator --session-option overrides;
    // forwarded opaquely - no key is interpreted here.
    session_options.options = merge_options(task.session_options, config.session_options);

    handle.model = handle.registry.load(load_request);
    handle.session = handle.model->create_task_session(task_spec, session_options);

    handle.streaming =
        dynamic_cast<engine::runtime::IStreamingVoiceTaskSession *>(handle.session.get());
    if (handle.streaming == nullptr) {
        throw std::runtime_error("model_task '" + task.id + "' does not support streaming execution");
    }
    return handle;
}

}  // namespace sidecar
