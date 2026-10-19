// The generic streaming drive loop.
//
// Translates the wire protocol (PROTOCOL.md) to and from audio.cpp's generic streaming
// interface. It references ONLY `engine::runtime::IStreamingVoiceTaskSession` - no
// concrete session class - so the same loop drives any registered model+task:
//
//   AUDIO (s16le PCM)  -> session.process_audio_chunk(chunk)   [-> EVENT if it emits]
//   FLUSH              -> session.finalize()                   -> exactly one FINAL
//                         (interim deltas arrive via the stream-event sink as EVENTs)
//   RESET              -> session.reset()                      (new utterance, warm model)
//   BYE / EOF          -> return, freeing the session
//
// Endpointing stays on the client's side: a final is produced only by a client
// FLUSH, never by a native end-of-utterance event.
#pragma once

#include "task_registry.h"

#include "engine/framework/runtime/session.h"

namespace sidecar {

// Drives one streaming session over the connection `fd` until BYE or EOF. Assumes the
// handshake has already completed and READY was sent. Prepares and starts the session,
// then services AUDIO/FLUSH/RESET frames. Never throws: recoverable per-frame errors are
// reported as ERROR frames; an I/O failure or peer close ends the loop.
//
// `request_options` is the caller-merged (registry defaults + operator overrides)
// map placed opaquely on TaskRequest.options at the initial start_stream and
// every per-utterance re-arm. The loop never reads a key from it.
void drive_streaming_session(int fd, engine::runtime::IStreamingVoiceTaskSession & session,
                             const OptionMap & request_options = {});

}  // namespace sidecar
