#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace kpolaris {

// Host-side empirical radius quantiles. Path order is deliberately discarded:
// one lensed ray may visit the same radial shell on both sides of its turning point.
inline std::array<double, 3> emissivity_radius_quantiles(
    std::vector<std::pair<double, double>> radius_weight) {
    double largest = 0.0;
    for (const auto& sample : radius_weight) {
        if (!std::isfinite(sample.first) || !std::isfinite(sample.second) ||
            sample.second < 0.0) {
            throw std::invalid_argument("radius quantiles require finite, nonnegative weights");
        }
        largest = std::max(largest, sample.second);
    }
    if (largest == 0.0) return {};
    std::sort(radius_weight.begin(), radius_weight.end());
    double total = 0.0;
    for (const auto& sample : radius_weight) total += sample.second / largest;
    const std::array<double, 3> fractions{0.05, 0.50, 0.95};
    std::array<double, 3> result{};
    double cumulative = 0.0;
    size_t target = 0;
    for (const auto& sample : radius_weight) {
        cumulative += sample.second / largest;
        while (target < result.size() && cumulative >= fractions[target] * total) {
            result[target++] = sample.first;
        }
    }
    return result;
}

// Stability relative to the final signal, not a possibly much larger earlier
// peak. A non-radiating or non-finite history has no physical freeze sample.
template<class SampleValue>
inline int intensity_freeze_sample(int count, double final_i, SampleValue value_at,
                                   double relative_tolerance = 0.01) {
    if (!(final_i > 0.0) || !std::isfinite(final_i)) return -1;
    double suffix_max = 0.0;
    int freeze = -1;
    for (int sample = count - 1; sample >= 0; --sample) {
        const double value = value_at(sample);
        if (!std::isfinite(value)) return -1;
        suffix_max = std::max(suffix_max, std::abs(value - final_i));
        if (suffix_max <= relative_tolerance * final_i) freeze = sample;
    }
    return freeze;
}

} // namespace kpolaris
