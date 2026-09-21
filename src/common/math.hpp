#pragma once

#include <Kokkos_Core.hpp>
#include <limits>

#include "common/types.hpp"

namespace kpolaris {

template<class Real>
KPOLARIS_INLINE Real abs_val(Real x) {
    return x < Real(0) ? -x : x;
}

template<class Real>
KPOLARIS_INLINE Real square(Real x) {
    return x * x;
}

template<class Real>
KPOLARIS_INLINE Real min_val(Real a, Real b) {
    return a < b ? a : b;
}

template<class Real>
KPOLARIS_INLINE Real max_val(Real a, Real b) {
    return a > b ? a : b;
}

template<class Real>
KPOLARIS_INLINE Real clamp(Real x, Real lo, Real hi) {
    return min_val(max_val(x, lo), hi);
}

// A nonzero denominator floor that remains representable in both supported
// main precisions.  Casting 1e-300 directly to float produces zero and can
// turn otherwise safe fallback branches into divisions by zero.
template<class Real>
KPOLARIS_INLINE Real tiny_positive() {
    if constexpr (sizeof(Real) <= sizeof(float)) {
        return Real(1e-30f);
    } else {
        return Real(1e-300);
    }
}

template<class Real>
KPOLARIS_INLINE Real large_positive() {
    if constexpr (sizeof(Real) <= sizeof(float)) {
        return Real(1e30f);
    } else {
        return Real(1e300);
    }
}

template<class Real>
KPOLARIS_INLINE Real adaptive_tolerance_floor() {
    if constexpr (sizeof(Real) <= sizeof(float)) {
        return Real(8) * Real(1.1920928955078125e-7f);
    } else {
        return Real(8) * Real(2.220446049250313080847263336181640625e-16);
    }
}

template<class Real>
KPOLARIS_INLINE Real stable_attenuation(Real kappa, Real dlambda) {
    const Real x = kappa * dlambda;
    const Real max_log = Real(700);
    if (x > max_log) {
        return Real(0);
    }
    return Kokkos::exp(-x);
}

template<class Real>
KPOLARIS_INLINE Real stable_source_factor(Real kappa, Real dlambda) {
    const Real x = kappa * dlambda;
    if (abs_val(x) < Real(1e-6)) {
        return dlambda *
               (Real(1) - x / Real(2) + x * x / Real(6) -
                x * x * x / Real(24));
    }

    return (Real(1) - Kokkos::exp(-x)) / kappa;
}

} // namespace kpolaris
