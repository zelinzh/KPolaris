#pragma once

#include <cstdlib>
#include <limits>
#include <ostream>
#include <set>
#include <vector>
#include <stdexcept>
#include <string>

namespace kpolaris {

// Environment defaults retain compatibility with dense-dump-codec launchers.
// Tool parameter files / CLI override these values without changing the process environment.
struct DDCInputOptions {
    int native = 0;
    std::string socket_path;
    std::string manifest; // Declared provenance; STAGE1 does not authenticate a manifest.
    int timeout_seconds = 7200;

    static int parse_integer(const std::string& value, const std::string& name) {
        int result = 0;
        if (value.empty()) throw std::runtime_error(name + " requires an integer");
        for (char c : value) {
            if (c < '0' || c > '9' || result > (std::numeric_limits<int>::max() - (c - '0')) / 10) {
                throw std::runtime_error(name + " requires a nonnegative integer");
            }
            result = result * 10 + (c - '0');
        }
        return result;
    }

    bool parse(const std::string& key, const std::string& value) {
        if (key == "kharma_ddc_native") native = parse_integer(value, key);
        else if (key == "kharma_ddc_socket") socket_path = value == "none" ? "" : value;
        else if (key == "kharma_ddc_manifest") manifest = value == "none" ? "" : value;
        else if (key == "kharma_ddc_timeout_seconds") timeout_seconds = parse_integer(value, key);
        else return false;
        return true;
    }

    void validate() const {
        if (native != 0 && native != 1) throw std::runtime_error("kharma_ddc_native must be 0 or 1");
        if (timeout_seconds <= 0) throw std::runtime_error("kharma_ddc_timeout_seconds must be positive");
        if (native && socket_path.empty()) throw std::runtime_error("kharma_ddc_socket / KPOLARIS_DDC_SOCKET is required for native DDC input");
    }

    static DDCInputOptions from_environment() {
        DDCInputOptions result;
        if (const char* v = std::getenv("KPOLARIS_DDC_NATIVE")) result.native = parse_integer(v, "KPOLARIS_DDC_NATIVE");
        if (const char* v = std::getenv("KPOLARIS_DDC_SOCKET")) result.socket_path = v;
        if (const char* v = std::getenv("KPOLARIS_DDC_MANIFEST")) result.manifest = v;
        if (const char* v = std::getenv("KPOLARIS_DDC_SOCKET_TIMEOUT_SECONDS")) result.timeout_seconds = parse_integer(v, "KPOLARIS_DDC_SOCKET_TIMEOUT_SECONDS");
        return result;
    }

    // One native sequence has one physical time. A repeated virtual name at
    // different scheduled times must not reuse the first occurrence's check.
    void validate_schedule(const std::vector<std::string>& paths) const {
        if (!native) return;
        std::set<std::string> frames;
        for (const auto& path : paths) {
            const std::string name = path.substr(path.find_last_of("/\\") + 1);
            if (name.compare(0, 10, "ddc_frame_") == 0 && !frames.insert(name).second) {
                throw std::runtime_error("DDC frame occurs more than once in slow-light schedule: " + name);
            }
        }
    }

    void write_parameters(std::ostream& out, const char* prefix = "") const {
        out << prefix << "kharma_ddc_native=" << native << "\n";
        out << prefix << "kharma_ddc_socket=" << (socket_path.empty() ? "none" : socket_path) << "\n";
        out << prefix << "kharma_ddc_manifest=" << (manifest.empty() ? "none" : manifest) << "\n";
        out << prefix << "kharma_ddc_timeout_seconds=" << timeout_seconds << "\n";
    }
};

} // namespace kpolaris
