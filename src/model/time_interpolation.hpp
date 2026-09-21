#pragma once

#include "common/math.hpp"

namespace kpolaris {

// Snapshot mode reproduces the original pair-by-pair geodesic stepping.
// Block mode clips only at the end of the resident time range; coefficient
// samples still use the adjacent original snapshots enclosing their own time.
enum class SlowLightStepMode { snapshot, block, decoupled };
inline constexpr int decoupled_cache_halo_snapshots = 4;
inline constexpr SlowLightStepMode default_slow_light_step_mode = SlowLightStepMode::decoupled;
inline const char* slow_light_step_mode_name(SlowLightStepMode mode) {
    if (mode == SlowLightStepMode::decoupled) return "decoupled";
    return mode == SlowLightStepMode::block ? "block" : "snapshot";
}

enum class SlowLightInterpolation { coefficients, fluid };

inline constexpr SlowLightInterpolation default_slow_light_interpolation = SlowLightInterpolation::fluid;

inline const char* slow_light_interpolation_name(SlowLightInterpolation method) {
    return method == SlowLightInterpolation::fluid ? "fluid" : "coefficients";
}

// Preserve the endpoint and static-snapshot limits without roundoff from mixing.
template<class Real>
KPOLARIS_INLINE Real interpolate_time_value(Real a, Real b, Real fraction) {
    if (fraction <= Real(0) || a == b) return a;
    if (fraction >= Real(1)) return b;
    return (Real(1) - fraction) * a + fraction * b;
}

} // namespace kpolaris
