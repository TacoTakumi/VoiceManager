// The TTS drive path: TEXT -> blocking start_stream on a worker
// thread with the stream-event sink forwarding audio to the socket -> TTS_DONE.
//
// Selected by the descriptor's task kind (task=tts), never by the model family:
// like drive_loop.h it references ONLY the generic
// `engine::runtime::IStreamingVoiceTaskSession`, so the same path drives any
// registered streaming-TTS model+task:
//
//   TEXT {text, voice_wav, ref_text} -> session.start_stream(request)
//     - the request carries text_input, the loaded voice reference, and
//       options[reference_text] (ref_text is per-model: REQUIRED for dots,
//       optional for voxcpm2 - a dots request without it is rejected with
//       ERROR bad_frame, never dispatched;
//       a voxcpm2 request omits the reference_text option key when absent)
//     - a FinalResult session pushes audio through the stream-event sink WHILE
//       generation runs; a PullEvents session (streaming_policy().output, read
//       there and nowhere else) stages
//       the request in start_stream and the worker pulls next_stream_event until
//       nullopt; either way each buffer (the trailing flush-named one included) is forwarded
//       as a TTS_AUDIO frame (s16le mono, the READY-advertised rate) through the
//       buffer-and-drain pump: the sink push only queues in-process
//       and the connection thread drains to the socket, so a slow reader never
//       stalls generation and no byte is dropped or reordered
//   -> exactly one TTS_DONE after the last audio byte of the request
//   BYE / EOF -> return, freeing the session (an in-flight request is finished
//   first: generation is not cancellable here, and PROTOCOL.md's BYE contract is
//   to finish the in-flight reply)
#pragma once

#include "task_registry.h"

#include "engine/framework/runtime/session.h"

#include <functional>
#include <string>

namespace sidecar {

// Loads a reference WAV from an absolute path into the engine's AudioBuffer float
// convention. Injected so the drive path (and its model-free tests) never links an
// engine library symbol; main.cpp passes a loader wrapping engine::audio::read_wav_f32.
// Throws on an unreadable/undecodable file.
using WavLoader = std::function<engine::runtime::AudioBuffer(const std::string & path)>;

// Seam-silence trim configuration. The dots session generates each text
// chunk as its own utterance, so a seam carries an end-of-utterance tail plus a
// start-of-utterance lead of silence (measured 400-900 ms against 340-620 ms for
// natural in-chunk pauses). The drive path caps any silence run that CROSSES a
// segment-id change (the sink buffers are named segment_<i>_chunk_<n>) to
// `seam_silence_cap_samples`; runs that stay inside one segment - in-chunk pauses and
// the request tail - are never touched, and no voiced sample is ever dropped. This is
// the ONE deliberate exception to the pump's no-drop rule, bounded and silence-only;
// the backpressure path still never drops a byte.
struct TtsDriveConfig {
    // Cap in samples at the output rate; <= 0 disables the trim entirely
    // (bit-exact pass-through, unchanged buffer framing).
    long long seam_silence_cap_samples = 0;
    // Amplitude (float domain, engine's /32768 convention) below which a sample
    // counts as silence for the seam run.
    float seam_silence_threshold = 0.004f;
};

// Drives one TTS session over the connection `fd` until BYE or EOF. Assumes the
// handshake has already completed and READY was sent. Never throws: recoverable
// per-request errors are reported as ERROR frames; an I/O failure or peer close ends
// the loop.
//
// `request_options` is the caller-merged (registry defaults + operator overrides)
// map placed opaquely on every TaskRequest.options; the loop never reads a key from
// it (the wire ref_text is added under the engine's `reference_text` option key).
// `ref_text_required` is the descriptor's per-model flag: dots requires a non-empty
// reference transcript, voxcpm2 does not. The loop enforces THIS flag - never a family
// string comparison.
void drive_tts_session(int fd, engine::runtime::IStreamingVoiceTaskSession & session,
                       const OptionMap & request_options, const WavLoader & load_wav,
                       const TtsDriveConfig & config = TtsDriveConfig{},
                       bool ref_text_required = true);

}  // namespace sidecar
