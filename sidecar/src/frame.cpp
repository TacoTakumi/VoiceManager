#include "frame.h"

#include <cerrno>
#include <cstring>
#include <stdexcept>

#include <unistd.h>

namespace sidecar {

namespace {

// Read exactly `n` bytes into `dst`. Returns false only on a clean EOF before the
// first byte (peer closed at a frame boundary). Throws on EOF mid-read or an I/O error.
bool read_exact(int fd, void * dst, std::size_t n, bool allow_eof_at_start) {
    auto * p = static_cast<unsigned char *>(dst);
    std::size_t got = 0;
    while (got < n) {
        const ssize_t r = ::read(fd, p + got, n - got);
        if (r > 0) {
            got += static_cast<std::size_t>(r);
            continue;
        }
        if (r == 0) {  // EOF
            if (got == 0 && allow_eof_at_start) {
                return false;
            }
            throw std::runtime_error("unexpected EOF mid-frame");
        }
        if (errno == EINTR) {
            continue;
        }
        throw std::runtime_error(std::string("read failed: ") + std::strerror(errno));
    }
    return true;
}

void write_all(int fd, const void * src, std::size_t n) {
    const auto * p = static_cast<const unsigned char *>(src);
    std::size_t sent = 0;
    while (sent < n) {
        const ssize_t w = ::write(fd, p + sent, n - sent);
        if (w > 0) {
            sent += static_cast<std::size_t>(w);
            continue;
        }
        if (w < 0 && errno == EINTR) {
            continue;
        }
        throw std::runtime_error(std::string("write failed: ") + std::strerror(errno));
    }
}

}  // namespace

bool read_frame(int fd, Frame & out) {
    unsigned char header[kFrameHeaderBytes];
    if (!read_exact(fd, header, kFrameHeaderBytes, /*allow_eof_at_start=*/true)) {
        return false;
    }
    const std::uint8_t type = header[0];
    const std::uint32_t length = (static_cast<std::uint32_t>(header[1]) << 24) |
                                 (static_cast<std::uint32_t>(header[2]) << 16) |
                                 (static_cast<std::uint32_t>(header[3]) << 8) |
                                 (static_cast<std::uint32_t>(header[4]));
    if (length > kMaxPayloadBytes) {
        throw std::runtime_error("frame length exceeds maximum");
    }
    out.type = static_cast<MsgType>(type);
    out.payload.resize(length);
    if (length > 0) {
        read_exact(fd, out.payload.data(), length, /*allow_eof_at_start=*/false);
    }
    return true;
}

void write_frame(int fd, MsgType type, const std::string & payload) {
    if (payload.size() > kMaxPayloadBytes) {
        throw std::runtime_error("outgoing frame exceeds maximum");
    }
    const std::uint32_t length = static_cast<std::uint32_t>(payload.size());
    unsigned char header[kFrameHeaderBytes];
    header[0] = static_cast<std::uint8_t>(type);
    header[1] = static_cast<unsigned char>((length >> 24) & 0xFF);
    header[2] = static_cast<unsigned char>((length >> 16) & 0xFF);
    header[3] = static_cast<unsigned char>((length >> 8) & 0xFF);
    header[4] = static_cast<unsigned char>(length & 0xFF);
    write_all(fd, header, kFrameHeaderBytes);
    if (length > 0) {
        write_all(fd, payload.data(), length);
    }
}

}  // namespace sidecar
