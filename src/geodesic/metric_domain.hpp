#pragma once

#include <type_traits>
#include <utility>

#include "common/vec.hpp"

namespace kpolaris {

template<class Metric, class Real, class = void>
struct MetricHasRadialCoordinate : std::false_type {};

template<class Metric, class Real>
struct MetricHasRadialCoordinate<Metric, Real, std::void_t<decltype(
    std::declval<const Metric&>().radial_coordinate(
        std::declval<const Vec4<Real>&>()))>> : std::true_type {};

template<class Metric, class Real, class = void>
struct MetricHasInnerBoundaryValue : std::false_type {};

template<class Metric, class Real>
struct MetricHasInnerBoundaryValue<Metric, Real, std::void_t<decltype(
    std::declval<const Metric&>().inner_boundary_value(
        std::declval<const Vec4<Real>&>(), std::declval<Real>()))>> :
    std::true_type {};

// Metrics with a moving or non-spherical excision surface can use this hook to
// keep an integration trial from stepping far beyond that surface.  The hook
// returns an absolute affine-step limit; returning a non-positive value means
// "no additional limit".  Metrics without the hook retain the legacy step.
template<class Metric, class Real, class = void>
struct MetricHasInnerBoundaryStepLimit : std::false_type {};

template<class Metric, class Real>
struct MetricHasInnerBoundaryStepLimit<Metric, Real, std::void_t<decltype(
    std::declval<const Metric&>().inner_boundary_step_limit(
        std::declval<const Vec4<Real>&>(),
        std::declval<const Vec4<Real>&>(),
        std::declval<Real>(), std::declval<Real>()))>> : std::true_type {};

// An excision-aware metric may explicitly classify an imminent transverse
// inner-boundary event before any RK trial enters its excluded region.  This
// is a separate opt-in so ordinary adaptive failures remain step underflows.
template<class Metric, class Real, class = void>
struct MetricHasInnerBoundaryTerminalEvent : std::false_type {};

template<class Metric, class Real>
struct MetricHasInnerBoundaryTerminalEvent<Metric, Real, std::void_t<decltype(
    std::declval<const Metric&>().inner_boundary_step_is_terminal(
        std::declval<const Vec4<Real>&>(),
        std::declval<const Vec4<Real>&>(),
        std::declval<Real>(), std::declval<Real>()))>> : std::true_type {};

// An affine parameter has no physical normalization: rescaling k^mu by a
// constant rescales every affine step by its inverse.  Metrics whose ray
// normalization can become large may opt in to a local minimum-step hook so
// that a configured physical displacement floor is not mistaken for an
// affine-step floor.  Metrics without the hook retain the configured value.
template<class Metric, class Real, class = void>
struct MetricHasEffectiveMinStep : std::false_type {};

template<class Metric, class Real>
struct MetricHasEffectiveMinStep<Metric, Real, std::void_t<decltype(
    std::declval<const Metric&>().effective_min_step(
        std::declval<const Vec4<Real>&>(), std::declval<Real>()))>> :
    std::true_type {};

// Dynamic metrics backed by finite trajectory data can expose their temporal
// support explicitly.  A false result is a physical data-domain boundary, not
// an RK failure; callers must terminate with metric_time_exhausted instead of
// evaluating a silently clamped trajectory.
template<class Metric, class Real, class = void>
struct MetricHasTimeDomain : std::false_type {};

template<class Metric, class Real>
struct MetricHasTimeDomain<Metric, Real, std::void_t<decltype(
    std::declval<const Metric&>().time_domain_valid(
        std::declval<const Vec4<Real>&>()))>> : std::true_type {};

template<class Metric, class Real>
KPOLARIS_INLINE int metric_time_domain_valid(
    const Metric& metric,
    const Vec4<Real>& x) {
    if constexpr (MetricHasTimeDomain<Metric, Real>::value) {
        return metric.time_domain_valid(x);
    }
    return 1;
}

// A finite, tabulated spacetime may limit a trial before an RK stage leaves
// the available time interval.  The metric receives the signed proposed step
// so it can distinguish forward and backward affine evolution; it returns an
// absolute step limit, following the inner-boundary hook convention.
template<class Metric, class Real, class = void>
struct MetricHasTimeDomainStepLimit : std::false_type {};

template<class Metric, class Real>
struct MetricHasTimeDomainStepLimit<Metric, Real, std::void_t<decltype(
    std::declval<const Metric&>().time_domain_step_limit(
        std::declval<const Vec4<Real>&>(),
        std::declval<const Vec4<Real>&>(),
        std::declval<Real>()))>> : std::true_type {};

// Once the remaining affine distance to a finite time boundary is at the
// effective minimum-step scale, terminate explicitly instead of asking the
// adaptive controller to resolve an endpoint kink until it underflows.
template<class Metric, class Real, class = void>
struct MetricHasTimeDomainTerminalEvent : std::false_type {};

template<class Metric, class Real>
struct MetricHasTimeDomainTerminalEvent<Metric, Real, std::void_t<decltype(
    std::declval<const Metric&>().time_domain_step_is_terminal(
        std::declval<const Vec4<Real>&>(),
        std::declval<const Vec4<Real>&>(),
        std::declval<Real>(), std::declval<Real>()))>> : std::true_type {};

template<class Metric, class Real>
KPOLARIS_INLINE Real metric_limit_time_domain_step(
    const Metric& metric,
    const Vec4<Real>& x,
    const Vec4<Real>& k,
    Real proposed_h) {
    if constexpr (MetricHasTimeDomainStepLimit<Metric, Real>::value) {
        const Real limit = metric.time_domain_step_limit(
            x, k, proposed_h);
        if (limit > tiny_positive<Real>()) {
            const Real limited_abs = min_val(abs_val(proposed_h), limit);
            return proposed_h < Real(0) ? -limited_abs : limited_abs;
        }
    }
    return proposed_h;
}

template<class Metric, class Real>
KPOLARIS_INLINE int metric_time_domain_step_is_terminal(
    const Metric& metric,
    const Vec4<Real>& x,
    const Vec4<Real>& k,
    Real proposed_h,
    Real event_tolerance) {
    if constexpr (MetricHasTimeDomainTerminalEvent<Metric, Real>::value) {
        return metric.time_domain_step_is_terminal(
            x, k, proposed_h, abs_val(event_tolerance));
    }
    return 0;
}

template<class Metric, class Real>
KPOLARIS_INLINE Real metric_effective_min_step(
    const Metric& metric,
    const Vec4<Real>& k,
    Real configured_min_step) {
    if constexpr (MetricHasEffectiveMinStep<Metric, Real>::value) {
        return metric.effective_min_step(k, abs_val(configured_min_step));
    }
    return abs_val(configured_min_step);
}

template<class Metric, class Real>
KPOLARIS_INLINE Real metric_domain_radius(const Metric& metric,
                                          const Vec4<Real>& x) {
    if constexpr (MetricHasRadialCoordinate<Metric, Real>::value) {
        return metric.radial_coordinate(x);
    } else if constexpr (Metric::coordinate_system == CoordinateSystem::FMKS) {
        return Kokkos::exp(x[1]);
    } else if constexpr (Metric::coordinate_system == CoordinateSystem::SphericalKS ||
                         Metric::coordinate_system == CoordinateSystem::BoyerLindquist) {
        return x[1];
    } else {
        return metric.build_work(x).r;
    }
}

// Positive outside the capture surface and non-positive inside.  Existing
// single-hole metrics retain their old radial inner boundary; binary metrics
// may provide a union of moving capture surfaces.
template<class Metric, class Real>
KPOLARIS_INLINE Real metric_inner_surface_value(const Metric& metric,
                                                const Vec4<Real>& x,
                                                Real inner_parameter) {
    if constexpr (MetricHasInnerBoundaryValue<Metric, Real>::value) {
        return metric.inner_boundary_value(x, inner_parameter);
    } else {
        return metric_domain_radius(metric, x) - inner_parameter;
    }
}

template<class Metric, class Real>
KPOLARIS_INLINE Real metric_limit_inner_surface_step(
    const Metric& metric,
    const Vec4<Real>& x,
    const Vec4<Real>& k,
    Real inner_parameter,
    Real proposed_h) {
    if constexpr (MetricHasInnerBoundaryStepLimit<Metric, Real>::value) {
        // The metric hook uses a traversal tangent and a positive step length.
        // Convert locally; the stored photon wavevector remains future directed.
        const Vec4<Real> direction = proposed_h < Real(0) ? k * Real(-1) : k;
        const Real limit = metric.inner_boundary_step_limit(
            x, direction, inner_parameter, abs_val(proposed_h));
        if (limit > tiny_positive<Real>()) {
            const Real limited_abs = min_val(abs_val(proposed_h), limit);
            return proposed_h < Real(0) ? -limited_abs : limited_abs;
        }
    }
    return proposed_h;
}

template<class Metric, class Real>
KPOLARIS_INLINE int metric_inner_surface_step_is_terminal(
    const Metric& metric,
    const Vec4<Real>& x,
    const Vec4<Real>& k,
    Real inner_parameter,
    Real proposed_h,
    Real event_tolerance) {
    if constexpr (MetricHasInnerBoundaryTerminalEvent<Metric, Real>::value) {
        const Vec4<Real> direction = proposed_h < Real(0) ? k * Real(-1) : k;
        return metric.inner_boundary_step_is_terminal(
            x, direction, inner_parameter, abs_val(event_tolerance));
    }
    return 0;
}

} // namespace kpolaris
