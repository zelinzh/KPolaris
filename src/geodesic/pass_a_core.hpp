#pragma once

#include "camera/camera.hpp"
#include "geodesic/metric_domain.hpp"
#include "geodesic/rk4.hpp"
#include "radiation/emission_selection.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct PassAParams {
    CameraParams<Real> camera;
    EmissionSelection<Real> emission_selection;
    Real mass = Real(1);
    Real spin = Real(0);
    Real inner_radius = Real(2.2);
    Real outer_radius = Real(200);
    Real step = Real(0.05);
    int max_steps = 4096;
    int adaptive = 0;
    int direct_only = 0; // Observer-adjacent dz/ds turning segment (arXiv:2512.09641).
    Real adaptive_tolerance = Real(1e-7);
    Real min_step = Real(1e-5);
    Real max_step = Real(1);
    Real max_radiation_step = Real(1);
    Real max_radiation_depth = Real(1);
    Real max_absorption_depth = Real(-1);
    Real max_faraday_depth = Real(-1);
    Real fmks_startx1 = Real(0);
    Real fmks_hslope = Real(0.3);
    Real fmks_mks_smooth = Real(0.5);
    Real fmks_poly_alpha = Real(14);
    Real fmks_poly_xt = Real(0.82);
    Real fmks_poly_norm = Real(1);
    CoordinateSystem coordinate_system = CoordinateSystem::CartesianKS;
};

template<class Real = DefaultReal>
struct PassAResult {
    TransportState<Real> state;
    Real initial_null = Real(0);
    Real final_null = Real(0);
    Real min_radius = Real(0);
    Real final_radius = Real(0);
    Real frame_error = Real(0);
    Real path_length = Real(0);
    int steps = 0;
    TerminationReason reason = TerminationReason::none;
};

template<class Real = DefaultReal>
struct MetricSurfaceEvent {
    TransportState<Real> state;
    Real used_h = Real(0);
    Real surface_value = Real(0);
};

template<class Real>
KPOLARIS_INLINE int finite_transport_state(const TransportState<Real>& state) {
    for (int mu = 0; mu < ndim; ++mu) {
        const Real values[4] = {
            state.x[mu], state.k[mu], state.e1[mu], state.e2[mu]};
        for (int field = 0; field < 4; ++field) {
            if (!(values[field] == values[field]) ||
                abs_val(values[field]) >= large_positive<Real>()) {
                return 0;
            }
        }
    }
    return 1;
}

// Evaluate the same numerical path used by the accepted parent step. Adaptive
// RK4 accepts two half steps, so using a single RK4 trial here would invalidate
// the endpoint sign bracket even as fraction -> 1.
template<class Metric, class Real>
KPOLARIS_INLINE TransportState<Real> metric_surface_path_state(
    const Metric& metric,
    const TransportState<Real>& first_state,
    Real h,
    int composed_two_half_steps) {
    if (composed_two_half_steps) {
        const TransportState<Real> half =
            rk4_step(metric, first_state, h * Real(0.5));
        return rk4_step(metric, half, h * Real(0.5));
    }
    return rk4_step(metric, first_state, h);
}

// Refine a bracketed metric-domain crossing. Every trial follows the same RK
// composition as the already accepted parent step. The returned endpoint is
// the last finite point on the original (physical-domain) side of the surface,
// never an unvalidated state below an excision boundary.
template<class Metric, class Real>
KPOLARIS_INLINE MetricSurfaceEvent<Real> locate_metric_surface_crossing(
    const Metric& metric,
    const TransportState<Real>& first_state,
    const TransportState<Real>& second_state,
    Real full_h,
    Real surface_parameter,
    int inner_surface,
    int composed_two_half_steps) {
    MetricSurfaceEvent<Real> out;
    out.state = first_state;
    out.used_h = Real(0);
    Real lo = Real(0);
    Real hi = Real(1);
    Real flo = inner_surface ?
        metric_inner_surface_value(metric, first_state.x, surface_parameter) :
        metric_domain_radius(metric, first_state.x) - surface_parameter;
    out.surface_value = flo;

    if (flo == Real(0)) {
        return out;
    }
    if (!finite_transport_state(second_state)) {
        return out;
    }

    // Recompute hi with the trial evaluator. For an adaptive parent this is the
    // same two-half composition as second_state; checking it explicitly keeps
    // the bracket valid if a future step implementation changes.
    const TransportState<Real> hi_state = metric_surface_path_state(
        metric, first_state, full_h, composed_two_half_steps);
    if (!finite_transport_state(hi_state)) {
        return out;
    }
    Real fhi = inner_surface ?
        metric_inner_surface_value(metric, hi_state.x, surface_parameter) :
        metric_domain_radius(metric, hi_state.x) - surface_parameter;
    if (!(fhi == fhi) || abs_val(fhi) >= large_positive<Real>()) {
        return out;
    }
    if (fhi == Real(0)) {
        out.state = hi_state;
        out.used_h = full_h;
        out.surface_value = fhi;
        return out;
    }
    if ((flo > Real(0)) == (fhi > Real(0))) {
        return out;
    }

    // In float, a secant trial can round just inside the surface even for a
    // linear moving boundary. The exterior endpoint then needs bisection;
    // ten trials leave an O(1e-3) gap. Allow a float mantissa's worth of
    // refinements while keeping the production double path unchanged.
    constexpr int iterations = sizeof(Real) <= sizeof(float) ? 24 : 12;
    int use_secant = 1;
#if defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
    for (int iter = 0; iter < iterations; ++iter) {
        const Real width = hi - lo;
        Real trial = (lo + hi) * Real(0.5);
        const Real denom = flo - fhi;
        if (use_secant && abs_val(denom) > tiny_positive<Real>()) {
            const Real secant = lo + width * flo / denom;
            const Real guard = width * Real(0.1);
            if (secant > lo + guard && secant < hi - guard) {
                trial = secant;
            }
        }
        const TransportState<Real> trial_state = metric_surface_path_state(
            metric, first_state, full_h * trial, composed_two_half_steps);
        if (!finite_transport_state(trial_state)) {
            hi = trial;
            use_secant = 0;
            continue;
        }
        const Real ftrial = inner_surface ?
            metric_inner_surface_value(metric, trial_state.x, surface_parameter) :
            metric_domain_radius(metric, trial_state.x) - surface_parameter;
        if (!(ftrial == ftrial) || abs_val(ftrial) >= large_positive<Real>()) {
            hi = trial;
            use_secant = 0;
            continue;
        }
        if (ftrial == Real(0)) {
            out.state = trial_state;
            out.used_h = full_h * trial;
            out.surface_value = ftrial;
            return out;
        }
        if ((flo > Real(0)) == (ftrial > Real(0))) {
            lo = trial;
            flo = ftrial;
            out.state = trial_state;
            out.used_h = full_h * trial;
            out.surface_value = ftrial;
        } else {
            hi = trial;
            fhi = ftrial;
        }
    }
    return out;
}

// Spin-axis height z=r cos(theta), equal to Cartesian KS z. The FMKS
// polar coordinate depends on BOTH native x1 and x2; omitting dtheta/dx1
// changes segment membership. Use the signed affine step to determine the tracing direction.
template<class Metric, class Real>
KPOLARIS_INLINE void vertical_position_velocity(
    const Metric& metric, const TransportState<Real>& state, Real& z, Real& dzds) {
    if constexpr (Metric::coordinate_system == CoordinateSystem::FMKS) {
        const Real r = Kokkos::exp(state.x[1]);
        const Real theta = metric.theta_from_x2(state.x[1], state.x[2]);
        const Real dtheta = metric.dtheta_dx1(state.x[1], state.x[2]) * state.k[1] +
                            metric.dtheta_dx2(state.x[1], state.x[2]) * state.k[2];
        z = r * Kokkos::cos(theta);
        dzds = z * state.k[1] - r * Kokkos::sin(theta) * dtheta;
    } else if constexpr (Metric::coordinate_system == CoordinateSystem::SphericalKS ||
                         Metric::coordinate_system == CoordinateSystem::BoyerLindquist ||
                         Metric::coordinate_system == CoordinateSystem::MKS) {
        z = state.x[1] * Kokkos::cos(state.x[2]);
        dzds = Kokkos::cos(state.x[2]) * state.k[1] -
                state.x[1] * Kokkos::sin(state.x[2]) * state.k[2];
    } else {
        z = state.x[3];
        dzds = state.k[3];
    }
}

// Only an away-from-midplane -> toward-midplane reversal is a boundary.
// An equatorial crossing, a theta turning point, a tangency without reversal,
// or an exactly equatorial orbit is not a vertical turning event.
template<class Real>
KPOLARIS_INLINE int vertical_turn_bracket(Real z0, Real v0, Real z1, Real v1,
                                         Real h, Real radius_scale) {
    // KPolaris supports IEEE float/double; hex constants are device-safe.
    constexpr Real epsilon = sizeof(Real) <= sizeof(float) ? Real(0x1p-23) : Real(0x1p-52);
    const Real floor = Real(64) * epsilon * radius_scale;
    return abs_val(z0) > floor && abs_val(z1) > floor &&
           ((z0 > Real(0)) == (z1 > Real(0))) &&
           z0 * v0 * h > Real(0) && z1 * v1 * h <= Real(0);
}

// Refine on the same RK composition as the parent step and stop at the
// observer-side endpoint of the bracket. With zero incident radiation,
// starting transfer here with S=0 is exactly equivalent to setting j=0
// on every more distant segment while retaining the full propagation K.
// Foreground emission, absorption and Faraday effects remain unchanged.
template<class Metric, class Real>
KPOLARIS_INLINE int truncate_at_first_vertical_turn(
    const Metric& metric, const TransportState<Real>& first,
    TransportState<Real>& last, Real& h, int composed_two_half_steps) {
    Real z0, v0, z1, v1;
    vertical_position_velocity(metric, first, z0, v0);
    vertical_position_velocity(metric, last, z1, v1);
    const Real radius_scale = max_val(metric_domain_radius(metric, first.x),
                                      metric_domain_radius(metric, last.x));
    if (!finite_transport_state(last) ||
        !vertical_turn_bracket(z0, v0, z1, v1, h, radius_scale)) return 0;
    Real lo = Real(0), hi = Real(1);
    TransportState<Real> endpoint = first;
    constexpr int iterations = sizeof(Real) <= sizeof(float) ? 24 : 48;
#if defined(__CUDA_ARCH__)
#pragma unroll 1
#endif
    for (int iter = 0; iter < iterations; ++iter) {
        const Real mid = (lo + hi) * Real(0.5);
        if (mid == lo || mid == hi) break;
        const auto trial = metric_surface_path_state(metric, first, h * mid,
                                                     composed_two_half_steps);
        Real z, v;
        vertical_position_velocity(metric, trial, z, v);
        if (!finite_transport_state(trial)) { hi = mid; continue; }
        if ((v > Real(0)) == (v0 > Real(0)) && v != Real(0)) {
            lo = mid; endpoint = trial;
        } else { hi = mid; }
    }
    last = endpoint;
    h *= lo;
    return 1;
}

template<class Metric, class Real>
KPOLARIS_INLINE Real metric_radius(const Metric& metric, const Vec4<Real>& x) {
    return metric_domain_radius(metric, x);
}

template<class Metric, class Real>
KPOLARIS_INLINE PassAResult<Real> trace_pass_a_pixel_metric(int pixel,
                                                         const PassAParams<Real>& params,
                                                         const Metric& metric) {
    PassAResult<Real> out;
    out.state = initialize_camera_ray(metric, pixel, params.camera);
    out.initial_null = metric.dot(out.state.x, out.state.k, out.state.k);
    out.min_radius = metric_radius(metric, out.state.x);
    out.final_radius = out.min_radius;
    out.reason = TerminationReason::max_steps;

    Real h_current = -abs_val(params.step);
    for (int n = 0; n < params.max_steps; ++n) {
        if (!metric_time_domain_valid(metric, out.state.x)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const Real r = metric_radius(metric, out.state.x);
        out.min_radius = min_val(out.min_radius, r);
        if (metric_inner_surface_value(metric, out.state.x,
                                       params.inner_radius) <= Real(0)) {
            out.reason = TerminationReason::reached_inner_boundary;
            break;
        }
        if (r >= params.outer_radius && n > 0) {
            out.reason = TerminationReason::escaped_domain;
            break;
        }

        const TransportState<Real> old_state = out.state;
        const Real effective_min_step = metric_effective_min_step(
            metric, old_state.k, params.min_step);
        if (metric_time_domain_step_is_terminal(
                metric, old_state.x, old_state.k,
                h_current, effective_min_step)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        if (metric_inner_surface_step_is_terminal(
                metric, old_state.x, old_state.k,
                params.inner_radius, h_current, effective_min_step)) {
            out.reason = TerminationReason::reached_inner_boundary;
            break;
        }
        Real used_h = metric_limit_time_domain_step(
            metric, old_state.x, old_state.k, h_current);
        used_h = metric_limit_inner_surface_step(
            metric, old_state.x, old_state.k, params.inner_radius, used_h);
        TransportState<Real> next_state;
        if (params.adaptive) {
            AdaptiveRK4Control<Real> control;
            control.tolerance = params.adaptive_tolerance;
            control.min_step = effective_min_step;
            control.max_step = params.max_step;
            if (abs_val(used_h) < control.min_step) {
                control.min_step = abs_val(used_h);
            }
            const auto step = adaptive_rk4_step(metric, old_state, used_h, control);
            if (!step.accepted) {
                h_current = step.next_h;
                if (abs_val(h_current) <= effective_min_step * Real(1.0001)) {
                    out.reason = TerminationReason::adaptive_step_underflow;
                    break;
                }
                continue;
            }
            next_state = step.state;
            used_h = step.used_h;
            h_current = step.next_h;
        } else {
            next_state = rk4_step(metric, old_state, used_h);
        }

        if (!metric_time_domain_valid(metric, next_state.x)) {
            // Keep the last supported event.  The RK stages may use the
            // provider's finite one-interval continuation to detect this
            // crossing, but unsupported trajectory values never become part
            // of the accepted ray.
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const int direct_turn = params.direct_only && truncate_at_first_vertical_turn(
            metric, old_state, next_state, used_h, params.adaptive);
        out.state = next_state;
        out.path_length += abs_val(used_h);
        out.steps += 1;
        if (direct_turn) {
            out.reason = TerminationReason::reached_direct_turn;
            break;
        }
    }

    out.final_radius = metric_radius(metric, out.state.x);
    out.final_null = metric.dot(out.state.x, out.state.k, out.state.k);
    out.frame_error = max_frame_error(frame_errors(metric, out.state));
    return out;
}

} // namespace kpolaris
