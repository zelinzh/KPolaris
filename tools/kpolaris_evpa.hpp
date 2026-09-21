#pragma once

#include <stdexcept>
#include <string>
#include <vector>

// Host-side output convention only. Transport always uses the original
// right-handed camera screen. N is its passive 90-degree rotation; W preserves it.
inline std::string parse_evpa_zero(const std::string& value) {
    if (value == "N" || value == "n") return "N";
    if (value == "W" || value == "w") return "W";
    throw std::invalid_argument("evpa_0 must be N or W");
}

inline int output_qu_sign(const std::string& evpa_0) {
    return evpa_0 == "N" ? -1 : 1;
}

template<class T>
std::vector<T> output_linear_component(std::vector<T> values, const std::string& evpa_0) {
    if (evpa_0 == "N") for (auto& value : values) value = -value;
    return values;
}
