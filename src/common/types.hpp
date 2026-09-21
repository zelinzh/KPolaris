#pragma once

#include <Kokkos_Core.hpp>

#ifndef KPOLARIS_MAX_FREQUENCIES
#define KPOLARIS_MAX_FREQUENCIES 1
#endif

#ifndef KPOLARIS_MAX_ANALYSIS_RADIAL_BINS
#define KPOLARIS_MAX_ANALYSIS_RADIAL_BINS 24
#endif

#ifndef KPOLARIS_SPECIAL_FUNCTION_FLOAT
#define KPOLARIS_SPECIAL_FUNCTION_FLOAT 0
#endif

namespace kpolaris {

#if KPOLARIS_USE_FLOAT
using DefaultReal = float;
#else
using DefaultReal = double;
#endif

#if KPOLARIS_SPECIAL_FUNCTION_FLOAT
using SpecialFunctionReal = float;
#else
using SpecialFunctionReal = DefaultReal;
#endif

constexpr int ndim = 4;
constexpr int max_frequencies = KPOLARIS_MAX_FREQUENCIES;
constexpr int max_analysis_radial_bins = KPOLARIS_MAX_ANALYSIS_RADIAL_BINS;

#define KPOLARIS_INLINE KOKKOS_INLINE_FUNCTION

enum class TerminationReason : int {
    none = 0,
    reached_camera = 1,
    reached_inner_boundary = 2,
    escaped_domain = 3,
    max_steps = 4,
    invalid_state = 5,
    slow_light_time_exhausted = 6,
    adaptive_step_underflow = 7,
    metric_time_exhausted = 8,
    reached_direct_turn = 9
};

enum class CoordinateSystem : int {
    CartesianKS = 0,
    BoyerLindquist = 1,
    SphericalKS = 2,
    FMKS = 3,
    MKS = 4,
};

template<class Real>
struct RayDiagnostics {
    Real null_error = Real(0);
    Real frame_error = Real(0);
    Real closure_x = Real(0);
    Real closure_k = Real(0);
    int steps = 0;
    TerminationReason reason = TerminationReason::none;
};

} // namespace kpolaris
