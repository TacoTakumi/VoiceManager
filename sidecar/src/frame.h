// Length-prefixed frame I/O over a blocking socket fd.
//
// Wire format (see PROTOCOL.md): a 5-byte header of [type:u8][length:u32 big-endian]
// followed by exactly `length` payload bytes. read_frame/write_frame handle partial
// reads/writes and EINTR, and enforce the kMaxPayloadBytes bound.
#pragma once

#include "protocol.h"

#include <string>

namespace sidecar {

struct Frame {
    MsgType type = MsgType::Error;
    std::string payload;  // raw bytes: UTF-8 JSON for control frames, s16le PCM for AUDIO
};

// Read one full frame from `fd` into `out`.
// Returns true on a complete frame; false on a clean EOF at a frame boundary (peer
// closed before sending another frame). Throws std::runtime_error on a short read
// mid-frame or an oversized declared length.
bool read_frame(int fd, Frame & out);

// Write a full frame (header + payload) to `fd`. Throws std::runtime_error on a write
// error (including a peer that closed the connection).
void write_frame(int fd, MsgType type, const std::string & payload);

}  // namespace sidecar
