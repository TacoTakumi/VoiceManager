cmake_minimum_required(VERSION 3.20)
project(audiocpp_sidecar LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# The vendored, pinned audio.cpp tree (never a moving branch ref). We link its
# prebuilt static engine (built by audio.cpp's own CMake preset) rather than rebuilding
# the CUDA runtime here, and reuse its bundled cJSON as the single source of truth.
set(AUDIOCPP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../3rdParty/audio.cpp"
    CACHE PATH "Path to the vendored audio.cpp checkout")
get_filename_component(AUDIOCPP_ROOT_ABS "${AUDIOCPP_ROOT}" ABSOLUTE)
set(AUDIOCPP_BUILD "${AUDIOCPP_ROOT}/build/linux-cuda-release"
    CACHE PATH "Path to the audio.cpp build tree holding the prebuilt static libraries")
set(CJSON_DIR "${AUDIOCPP_ROOT}/external/cJSON")

if (NOT EXISTS "${AUDIOCPP_BUILD}/libengine_runtime.a")
    message(FATAL_ERROR
        "prebuilt engine not found at ${AUDIOCPP_BUILD}/libengine_runtime.a; build audio.cpp "
        "first (its linux-cuda-release preset) or set -DAUDIOCPP_BUILD=<audio.cpp build dir>")
endif()

# The generic streaming drive loop over engine_runtime (IStreamingVoiceTaskSession
# + the model registry). cJSON is provided by the engine's libcjson_vendor.a (linked
# below), so we do NOT compile external/cJSON/cJSON.c here - that would duplicate its
# symbols at the final link.
add_executable(audiocpp_sidecar
    src/main.cpp
    src/frame.cpp
    src/task_registry.cpp
    src/engine_session.cpp
    src/drive_loop.cpp
    src/tts_drive.cpp
)
target_include_directories(audiocpp_sidecar PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CJSON_DIR}
    ${AUDIOCPP_ROOT}/include
    ${AUDIOCPP_ROOT}/external/ggml/include
)
# Match engine_runtime's PUBLIC compile definitions so shared headers compile identically.
# SIDECAR_DEFAULT_ASSETS_ROOT lets the sidecar find the vendored model_specs/ (audio.cpp
# resolves a model package spec by walking up from cwd) without the caller setting cwd.
target_compile_definitions(audiocpp_sidecar PRIVATE
    ENGINE_HAS_CUDA_ISTFT=1
    ENGINE_HAS_CUDA_TORCH_RANDOM=1
    SIDECAR_DEFAULT_ASSETS_ROOT="${AUDIOCPP_ROOT_ABS}"
)
target_compile_options(audiocpp_sidecar PRIVATE -O2 -Wall -Wextra)

# --- Link the prebuilt audio.cpp engine ------------------------------------------------
# The archive set has circular references (engine_runtime <-> ggml <-> ...), so it is
# wrapped in a linker group. The CUDA/CPU/system dependency set mirrors audio.cpp's own
# audiocpp_cli link line (its CMake-generated LINK_LIBRARIES for the linux-cuda-release
# build): cudart/cufft/cublas(+Lt)/culibos, the CUDA driver stub, nccl, OpenMP, threads.
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_library(NCCL_LIBRARY NAMES nccl)
if (NOT NCCL_LIBRARY)
    message(FATAL_ERROR "libnccl not found; the prebuilt ggml-cuda links against it")
endif()

# release-0.6 moved ggml's archive output from <build>/external/ggml/src to
# <build>/ggml/src (source stays at external/ggml); probe the build tree so the same
# link line works against either the pinned release-0.3 layout or a 0.6 build.
if (EXISTS "${AUDIOCPP_BUILD}/ggml/src/libggml.a")
    set(AUDIOCPP_GGML_BUILD "${AUDIOCPP_BUILD}/ggml/src")
else()
    set(AUDIOCPP_GGML_BUILD "${AUDIOCPP_BUILD}/external/ggml/src")
endif()

set(AUDIOCPP_STATIC_LIBS
    ${AUDIOCPP_BUILD}/libengine_runtime.a
    ${AUDIOCPP_GGML_BUILD}/libggml.a
    ${AUDIOCPP_GGML_BUILD}/libggml-cpu.a
    ${AUDIOCPP_GGML_BUILD}/ggml-cuda/libggml-cuda.a
    ${AUDIOCPP_GGML_BUILD}/libggml-base.a
    ${AUDIOCPP_BUILD}/external/sentencepiece/src/libsentencepiece.a
    ${AUDIOCPP_BUILD}/libcjson_vendor.a
    ${AUDIOCPP_BUILD}/libyaml_vendor.a
)

target_link_libraries(audiocpp_sidecar PRIVATE
    -Wl,--start-group ${AUDIOCPP_STATIC_LIBS} -Wl,--end-group
    CUDA::cudart
    CUDA::cufft
    CUDA::cublas
    CUDA::cublasLt
    CUDA::culibos
    CUDA::cuda_driver
    ${NCCL_LIBRARY}
    OpenMP::OpenMP_CXX
    Threads::Threads
    ${CMAKE_DL_LIBS}
    rt
    m
)

# --- Sink-throw cancel probe ------------------------------------------------------------
# MANUAL probe, never run by any suite: throws a sentinel from the stream-event sink
# mid-generation against the REAL dots_tts session and reports whether the session is
# reusable afterwards (the recorded verdict selects the TTS cancel mechanism). Links the
# same prebuilt engine as the sidecar.
add_executable(sink_throw_probe
    probes/sink_throw_probe.cpp
    src/engine_session.cpp
    src/task_registry.cpp
)
target_include_directories(sink_throw_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CJSON_DIR}
    ${AUDIOCPP_ROOT}/include
    ${AUDIOCPP_ROOT}/external/ggml/include
)
target_compile_definitions(sink_throw_probe PRIVATE
    ENGINE_HAS_CUDA_ISTFT=1
    ENGINE_HAS_CUDA_TORCH_RANDOM=1
)
target_compile_options(sink_throw_probe PRIVATE -O2 -Wall -Wextra)
target_link_libraries(sink_throw_probe PRIVATE
    -Wl,--start-group ${AUDIOCPP_STATIC_LIBS} -Wl,--end-group
    CUDA::cudart
    CUDA::cufft
    CUDA::cublas
    CUDA::cublasLt
    CUDA::culibos
    CUDA::cuda_driver
    ${NCCL_LIBRARY}
    OpenMP::OpenMP_CXX
    Threads::Threads
    ${CMAKE_DL_LIBS}
    rt
    m
)

# --- Model-free drive-loop tests ------------------------------------------------------
# The REAL drive loop against a fake IStreamingVoiceTaskSession over a socketpair: no
# model, no GPU, and deliberately NO engine libraries - the runtime session interface is
# header-only, and cJSON is compiled directly from the vendored source. The pytest
# wrapper (tests/test_sidecar_drive_loop.py) runs this binary and skips when it is
# absent.
#
# The chunk-mode tests compile the REAL vendored text chunker into this binary
# from three engine TUs. options.cpp's tensor-storage half references ggml/engine
# symbols the chunker never calls, so the target builds with function/data sections and
# links with --gc-sections: the unused half is discarded and the no-engine-libs
# property of this binary holds.
add_executable(audiocpp_sidecar_tests
    tests/drive_loop_rearm_test.cpp
    tests/tts_drive_test.cpp
    tests/task_registry_test.cpp
    src/drive_loop.cpp
    src/tts_drive.cpp
    src/frame.cpp
    src/task_registry.cpp
    ${CJSON_DIR}/cJSON.c
    ${AUDIOCPP_ROOT}/src/framework/text/chunking.cpp
    ${AUDIOCPP_ROOT}/src/framework/io/text.cpp
    ${AUDIOCPP_ROOT}/src/framework/runtime/options.cpp
    # Since main 624cf918, TaskResult holds a json::Value (CustomSchemaOutput), so its
    # destructor is referenced from the session header.
    ${AUDIOCPP_ROOT}/src/framework/io/json.cpp
)
target_include_directories(audiocpp_sidecar_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CJSON_DIR}
    ${AUDIOCPP_ROOT}/include
    ${AUDIOCPP_ROOT}/external/ggml/include
)
target_compile_definitions(audiocpp_sidecar_tests PRIVATE
    ENGINE_HAS_CUDA_ISTFT=1
    ENGINE_HAS_CUDA_TORCH_RANDOM=1
)
target_compile_options(audiocpp_sidecar_tests PRIVATE
    -O2 -Wall -Wextra -ffunction-sections -fdata-sections)
target_link_options(audiocpp_sidecar_tests PRIVATE -Wl,--gc-sections)
target_link_libraries(audiocpp_sidecar_tests PRIVATE Threads::Threads)
