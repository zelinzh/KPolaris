#pragma once
#include "grmhd/ddc_host_buffer.hpp"
#ifdef KPOLARIS_DDC_COMPACT_CUDA
#include "ddc_compact_receive.hpp"
#endif

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef _WIN32
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace kpolaris {

struct DDCNativeFrame {
#ifdef KPOLARIS_DDC_COMPACT_CUDA
    std::shared_ptr<ddc_transport::CompactFrame> compact;
#endif
    long long sequence = -1;
    double time = 0.0;
    size_t num_meshblocks = 0;
    int nx1_mb = 0;
    int nx2_mb = 0;
    int nx3_mb = 0;
    std::string par_text;
    std::vector<long long> block_order;
    ddc_transport::HostBuffer<float> rho;
    ddc_transport::HostBuffer<float> uu;
    ddc_transport::HostBuffer<float> uvec;
    ddc_transport::HostBuffer<float> bvec;
};

namespace ddc_detail {

// A native name is a sequence identifier, not a materialized-file cache lookup.
inline int native_sequence_from_path(const std::string& path) {
    const std::string filename = path.substr(path.find_last_of("/\\") + 1);
    const std::string prefix = "ddc_frame_", suffix = ".phdf";
    if (filename.compare(0, prefix.size(), prefix) != 0) return -1;
    if (filename.size() <= prefix.size() + suffix.size() ||
        filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) != 0) {
        throw std::runtime_error("Invalid DDC frame name: " + filename);
    }
    int sequence = 0;
    for (size_t i = prefix.size(); i < filename.size() - suffix.size(); ++i) {
        const char c = filename[i];
        if (c < '0' || c > '9' || sequence > (std::numeric_limits<int>::max() - (c - '0')) / 10) {
            throw std::runtime_error("Invalid DDC frame sequence: " + filename);
        }
        sequence = sequence * 10 + (c - '0');
    }
    return sequence;
}

inline DDCNativeFrame request_socket_staged_frame(
    const std::string& socket_path,
    int sequence,
    int timeout_seconds,
    DDCNativeFrame frame = {}) {
#ifdef _WIN32
    (void)socket_path;
    (void)sequence;
    (void)timeout_seconds;
    throw std::runtime_error(
        "KPolaris DDC native Unix socket service is unavailable on Windows");
#else
    static_assert(sizeof(long long) == 8 && sizeof(float) == 4 &&
                  std::numeric_limits<float>::is_iec559,
                  "STAGE1 requires int64 and IEEE float32");
    if (sequence < 0 || timeout_seconds <= 0) {
        throw std::runtime_error("DDC sequence must be nonnegative and timeout positive");
    }
    const unsigned short endian_probe = 1;
    if (*reinterpret_cast<const unsigned char*>(&endian_probe) != 1) {
        throw std::runtime_error("KPolaris DDC native transport requires a little-endian host");
    }
    sockaddr_un address{};
    if (socket_path.empty()) {
        throw std::runtime_error("KPOLARIS_DDC_SOCKET is required for native DDC input");
    }
    if (socket_path.size() >= sizeof(address.sun_path)) {
        throw std::runtime_error("KPOLARIS_DDC_SOCKET path is too long");
    }
    const int descriptor = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (descriptor < 0) {
        throw std::runtime_error(
            "Could not create DDC native Unix socket: " + std::string(std::strerror(errno)));
    }
    auto close_descriptor = [&]() { ::close(descriptor); };
    try {
        timeval timeout{};
        timeout.tv_sec = timeout_seconds;
        if (::setsockopt(descriptor, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
            ::setsockopt(descriptor, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) {
            throw std::runtime_error(
                "Could not configure DDC native socket timeout: " +
                std::string(std::strerror(errno)));
        }
        address.sun_family = AF_UNIX;
        std::memcpy(address.sun_path, socket_path.c_str(), socket_path.size() + 1);
        if (::connect(descriptor, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
            throw std::runtime_error(
                "Could not connect to DDC native service " + socket_path + ": " +
                std::string(std::strerror(errno)));
        }
#ifdef SO_NOSIGPIPE
        const int no_sigpipe = 1;
        if (::setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe)) != 0) {
            throw std::runtime_error("Could not configure DDC socket SO_NOSIGPIPE");
        }
#endif
#ifdef MSG_NOSIGNAL
        constexpr int send_flags = MSG_NOSIGNAL;
#else
        constexpr int send_flags = 0;
#endif
        std::string request = "STAGE " + std::to_string(sequence) + "\n";
        const bool compact = ddc_transport::enabled("KPOLARIS_DDC_COMPACT");
#ifdef KPOLARIS_DDC_COMPACT_CUDA
        if (compact) request = ddc_transport::compact_request(socket_path, sequence);
#else
        if (compact) throw std::runtime_error("This build does not support compact DDC CUDA input");
#endif
        size_t sent = 0;
        while (sent < request.size()) {
            const ssize_t count = ::send(
                descriptor, request.data() + sent, request.size() - sent, send_flags);
            if (count < 0 && errno == EINTR) continue;
            if (count <= 0) {
                throw std::runtime_error(
                    "Could not send DDC native request: " + std::string(std::strerror(errno)));
            }
            sent += static_cast<size_t>(count);
        }
        std::string header;
        while (header.find('\n') == std::string::npos) {
            char character = 0;
            const ssize_t count = ::recv(descriptor, &character, 1, 0);
            if (count < 0 && errno == EINTR) continue;
            if (count < 0) {
                throw std::runtime_error(
                    "Could not receive DDC native header: " + std::string(std::strerror(errno)));
            }
            if (count == 0) {
                throw std::runtime_error("DDC native service closed before the header");
            }
            header.push_back(character);
            if (header.size() > 65536) {
                throw std::runtime_error("DDC native header exceeded 65536 bytes");
            }
        }
        header.pop_back();
        if (header.compare(0, 6, "ERROR ") == 0) {
            throw std::runtime_error("DDC native service failed: " + header);
        }
        std::istringstream input(header);
        std::string format;
        long long response_sequence = -1;
        double response_time = 0.0;
        unsigned long long num_meshblocks = 0;
        int nx1_mb = 0, nx2_mb = 0, nx3_mb = 0;
        unsigned long long par_bytes = 0, block_count = 0;
        unsigned long long rho_count = 0, u_count = 0, uvec_count = 0, b_count = 0;
        unsigned long long payload_bytes = 0;
        if (!(input >> format >> response_sequence >> response_time >> num_meshblocks
                    >> nx1_mb >> nx2_mb >> nx3_mb >> par_bytes >> block_count
                    >> rho_count >> u_count >> uvec_count >> b_count >> payload_bytes) ||
            format != (compact ? "STAGEQ1" : "STAGE1")) {
            throw std::runtime_error("Invalid DDC native header: " + header);
        }
        std::string trailing;
        if (input >> trailing) {
            throw std::runtime_error("Unexpected trailing DDC native header fields");
        }
        if (response_sequence != sequence || !std::isfinite(response_time) || num_meshblocks == 0 ||
            nx1_mb <= 0 || nx2_mb <= 0 || nx3_mb <= 0) {
            throw std::runtime_error("Invalid DDC native sequence or mesh dimensions");
        }
        auto checked_size = [](unsigned long long value, const char* label) -> size_t {
            if (value > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
                throw std::runtime_error(std::string("DDC native ") + label + " exceeds size_t");
            }
            return static_cast<size_t>(value);
        };
        auto checked_multiply = [](size_t left, size_t right, const char* label) -> size_t {
            if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
                throw std::runtime_error(std::string("DDC native ") + label + " size overflow");
            }
            return left * right;
        };
        const size_t nmb = checked_size(num_meshblocks, "meshblock count");
        size_t scalar_count = checked_multiply(nmb, static_cast<size_t>(nx1_mb), "scalar");
        scalar_count = checked_multiply(scalar_count, static_cast<size_t>(nx2_mb), "scalar");
        scalar_count = checked_multiply(scalar_count, static_cast<size_t>(nx3_mb), "scalar");
        // The grid materializer uses int cell indices; reject impossible frames
        // before allocating primitive buffers.
        if (scalar_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::runtime_error("DDC native cell count exceeds the supported int index range");
        }
        const size_t vector_count = checked_multiply(scalar_count, size_t(3), "vector");
        if (checked_size(block_count, "block count") != checked_multiply(nmb, size_t(3), "block") ||
            checked_size(rho_count, "rho count") != scalar_count ||
            checked_size(u_count, "u count") != scalar_count ||
            checked_size(uvec_count, "uvec count") != vector_count ||
            checked_size(b_count, "B count") != vector_count) {
            throw std::runtime_error("DDC native array counts do not match the mesh layout");
        }
        const size_t par_size = checked_size(par_bytes, "parameter bytes");
        if (par_size == 0 || par_size > 16 * 1024 * 1024) {
            throw std::runtime_error("DDC native parameter text must contain 1..16777216 bytes");
        }
        size_t expected_bytes = par_size;
        auto add_bytes = [&](size_t count, size_t item_size) {
            const size_t bytes = checked_multiply(count, item_size, "payload");
            if (expected_bytes > std::numeric_limits<size_t>::max() - bytes) {
                throw std::runtime_error("DDC native payload size overflow");
            }
            expected_bytes += bytes;
        };
        add_bytes(checked_size(block_count, "block count"), sizeof(long long));
        add_bytes(scalar_count, sizeof(float));
        add_bytes(scalar_count, sizeof(float));
        add_bytes(vector_count, sizeof(float));
        add_bytes(vector_count, sizeof(float));
        if (!compact && expected_bytes != checked_size(payload_bytes, "payload bytes")) {
            throw std::runtime_error("DDC native payload byte count mismatch");
        }
        auto receive_exact = [&](void* output, size_t bytes) {
            char* destination = static_cast<char*>(output);
            size_t received = 0;
            while (received < bytes) {
                const ssize_t count = ::recv(descriptor, destination + received, bytes - received, 0);
                if (count < 0 && errno == EINTR) continue;
                if (count < 0) {
                    throw std::runtime_error(
                        "Could not receive DDC native payload: " +
                        std::string(std::strerror(errno)));
                }
                if (count == 0) {
                    throw std::runtime_error("DDC native service closed during payload transfer");
                }
                received += static_cast<size_t>(count);
            }
        };
        frame.sequence = response_sequence;
        frame.time = response_time;
        frame.num_meshblocks = nmb;
        frame.nx1_mb = nx1_mb;
        frame.nx2_mb = nx2_mb;
        frame.nx3_mb = nx3_mb;
        frame.par_text.resize(par_size);
        frame.block_order.resize(checked_size(block_count, "block count"));
        if (par_size) receive_exact(frame.par_text.data(), par_size);
        receive_exact(frame.block_order.data(), frame.block_order.size() * sizeof(long long));
#ifdef KPOLARIS_DDC_COMPACT_CUDA
        if (compact) {
            const size_t prefix_bytes = par_size + frame.block_order.size()*sizeof(long long);
            if (payload_bytes < prefix_bytes) throw std::runtime_error("Truncated compact payload");
            frame.compact = ddc_transport::receive_compact(receive_exact, socket_path, scalar_count,
                nx3_mb, nx2_mb, nx1_mb, payload_bytes-prefix_bytes, sequence);
            frame.rho = frame.compact->first.slice(0, scalar_count);
            frame.uu = frame.compact->first.slice(scalar_count, scalar_count);
            frame.uvec = frame.compact->first.slice(2*scalar_count, vector_count);
            frame.bvec = frame.compact->first.slice(5*scalar_count, vector_count);
            close_descriptor(); return frame;
        }
#endif
        if (ddc_transport::enabled("KPOLARIS_DDC_PINNED")) {
            auto storage = std::move(frame.rho);
            frame.uu = ddc_transport::HostBuffer<float>{};
            frame.uvec = ddc_transport::HostBuffer<float>{};
            frame.bvec = ddc_transport::HostBuffer<float>{};
            storage.resize(8*scalar_count,true);
            frame.rho = storage.slice(0,scalar_count);
            frame.uu = storage.slice(scalar_count,scalar_count);
            frame.uvec = storage.slice(2*scalar_count,vector_count);
            frame.bvec = storage.slice(5*scalar_count,vector_count);
            receive_exact(storage.data(),storage.size()*sizeof(float));
            close_descriptor(); return frame;
        }
        frame.rho.resize(scalar_count);
        frame.uu.resize(scalar_count);
        frame.uvec.resize(vector_count);
        frame.bvec.resize(vector_count);
        receive_exact(frame.rho.data(), frame.rho.size() * sizeof(float));
        receive_exact(frame.uu.data(), frame.uu.size() * sizeof(float));
        receive_exact(frame.uvec.data(), frame.uvec.size() * sizeof(float));
        receive_exact(frame.bvec.data(), frame.bvec.size() * sizeof(float));
        close_descriptor();
        return frame;
    } catch (...) {
        close_descriptor();
        throw;
    }
#endif
}

} // namespace ddc_detail
} // namespace kpolaris
