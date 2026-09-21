#pragma once

#include "geodesic/metric_domain.hpp"
#include "geodesic/rk4.hpp"
#include "radiation/stokes.hpp"

namespace kpolaris {

// A source selection at fixed plasma: outside the wedge only j is removed.
// Absorption and conversion remain active, and rotation is controlled separately.
template<class Real = DefaultReal>
struct EmissionSelection {
    Real equatorial_h_over_r = Real(0); // 0 disabled; 1 includes the whole sphere.
    int equatorial_samples = 8; // Minimum samples across 2h/r in a monotone crossing.
    int faraday_rotation = 1;
    KPOLARIS_INLINE bool narrow_wedge() const {
        return equatorial_h_over_r > Real(0) && equatorial_h_over_r < Real(1);
    }
};

// h is KS height |z| and r is the Kerr/Boyer-Lindquist radius, not cylindrical R.
// Native FMKS x2 must first be mapped to physical theta.
template<class Metric, class Real>
KPOLARIS_INLINE Real equatorial_height_ratio(const Metric& metric, const Vec4<Real>& x) {
    if constexpr (Metric::coordinate_system == CoordinateSystem::FMKS) {
        return Kokkos::cos(metric.theta_from_x2(x[1], x[2]));
    } else if constexpr (Metric::coordinate_system == CoordinateSystem::CartesianKS) {
        const Real r = metric_domain_radius(metric, x);
        return r > Real(0) ? x[3] / r : Real(0);
    } else {
        return Kokkos::cos(x[2]);
    }
}

template<class Metric, class Real>
KPOLARIS_INLINE void apply_emission_selection(const EmissionSelection<Real>& selection,
    const Metric& metric, const TransportState<Real>& state, TransferCoeffs<Real>& coeffs) {
    if (selection.narrow_wedge() &&
        abs_val(equatorial_height_ratio(metric, state.x)) > selection.equatorial_h_over_r) {
        coeffs.jI = coeffs.jQ = coeffs.jU = coeffs.jV = Real(0);
    }
    if (!selection.faraday_rotation) coeffs.rV = Real(0);
}

template<class Metric, class Real>
KPOLARIS_INLINE int equatorial_resolution_substeps(const EmissionSelection<Real>& selection,
    const Metric& metric, const TransportState<Real>& first,
    const TransportState<Real>& middle, const TransportState<Real>& last, Real outer_radius) {
    if (!selection.narrow_wedge()) return 1;
    if (metric_domain_radius(metric, first.x) > outer_radius &&
        metric_domain_radius(metric, middle.x) > outer_radius &&
        metric_domain_radius(metric, last.x) > outer_radius) return 1;
    const Real q0 = equatorial_height_ratio(metric, first.x);
    const Real qm = equatorial_height_ratio(metric, middle.x);
    const Real q1 = equatorial_height_ratio(metric, last.x);
    const Real span = abs_val(qm-q0) + abs_val(q1-qm);
    const Real allowed = Real(2) * selection.equatorial_h_over_r / Real(selection.equatorial_samples);
    const Real ratio = span / allowed;
    return ratio > Real(1.000001) ? int(min_val(Kokkos::ceil(ratio), Real(1000000))) : 1;
}

template<class Metric, class Real>
KPOLARIS_INLINE TransportState<Real> emission_selection_path_state(
    const Metric& metric, const TransportState<Real>& first, Real h, int composed) {
    if (!composed) return rk4_step(metric, first, h);
    return rk4_step(metric, rk4_step(metric, first, h*Real(0.5)), h*Real(0.5));
}

// Split a resolved accepted step at the first wedge surface it crosses. This
// prevents boundary phase from changing the thin-source column by O(1/Nsample).
// Ignore an initial point within roundoff of a surface to make forward progress.
template<class Metric, class Real>
KPOLARIS_INLINE int refine_equatorial_boundary(const EmissionSelection<Real>& selection,
    const Metric& metric, const TransportState<Real>& first,
    TransportState<Real>& middle, TransportState<Real>& last, Real& h,
    int composed, Real outer_radius) {
    if (!selection.narrow_wedge() || h == Real(0)) return 0;
    if (metric_domain_radius(metric, first.x) > outer_radius &&
        metric_domain_radius(metric, middle.x) > outer_radius &&
        metric_domain_radius(metric, last.x) > outer_radius) return 0;
    constexpr Real epsilon = sizeof(Real) <= sizeof(float) ? Real(0x1p-23) : Real(0x1p-52);
    const Real tolerance = Real(128) * epsilon;
    const Real q0 = equatorial_height_ratio(metric, first.x);
    const Real qm = equatorial_height_ratio(metric, middle.x);
    const Real q1 = equatorial_height_ratio(metric, last.x);
    Real edge = Real(0), lo = Real(0), hi = Real(1);
    int found = 0;
    for (int half = 0; half < 2 && !found; ++half) {
        const Real a = half ? qm : q0;
        const Real b = half ? q1 : qm;
        Real earliest = Real(2);
        for (int sign = -1; sign <= 1; sign += 2) {
            const Real surface = Real(sign) * selection.equatorial_h_over_r;
            if (abs_val(q0-surface) <= tolerance || abs_val(a-surface) <= tolerance) continue;
            if ((a-surface)*(b-surface) > Real(0)) continue;
            const Real fraction = (surface-a)/(b-a);
            if (fraction < earliest) { earliest=fraction; edge=surface; found=1; }
        }
        if (found) { lo=half ? Real(0.5) : Real(0); hi=half ? Real(1) : Real(0.5); }
    }
    if (!found) return 0;
    // Use the same RK composition for every root evaluation and the endpoint.
    auto low_state = emission_selection_path_state(metric, first, h*lo, composed);
    const Real low_value = equatorial_height_ratio(metric, low_state.x)-edge;
    constexpr int iterations = sizeof(Real) <= sizeof(float) ? 24 : 48;
#if defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
    for (int iter=0; iter<iterations; ++iter) {
        const Real mid=(lo+hi)*Real(0.5);
        if (mid==lo || mid==hi) break;
        const auto trial=emission_selection_path_state(metric, first, h*mid, composed);
        const Real value=equatorial_height_ratio(metric, trial.x)-edge;
        if ((value > Real(0)) == (low_value > Real(0))) lo=mid; else hi=mid;
    }
    const Real fraction=(lo+hi)*Real(0.5);
    if (fraction <= tolerance || fraction >= Real(1)-tolerance) return 0;
    h *= fraction;
    last=emission_selection_path_state(metric, first, h, composed);
    middle=rk4_step(metric, first, h*Real(0.5));
    return 1;
}

} // namespace kpolaris
