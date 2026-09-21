#pragma once

#include "geodesic/state.hpp"

namespace kpolaris {

template<class Metric, class Real>
KPOLARIS_INLINE TransportState<Real> transport_rhs(const Metric& metric,
                                                const TransportState<Real>& s) {
    Real gamma[ndim][ndim][ndim];
    metric.connection(s.x, gamma);

    TransportState<Real> d;
    d.x = s.k;

    for (int mu = 0; mu < ndim; ++mu) {
        Real dk = Real(0);
        Real de1 = Real(0);
        Real de2 = Real(0);
        for (int a = 0; a < ndim; ++a) {
            for (int b = 0; b < ndim; ++b) {
                const Real g = gamma[mu][a][b];
                dk -= g * s.k[a] * s.k[b];
                de1 -= g * s.k[a] * s.e1[b];
                de2 -= g * s.k[a] * s.e2[b];
            }
        }
        d.k[mu] = dk;
        d.e1[mu] = de1;
        d.e2[mu] = de2;
    }

    return d;
}

template<class Metric, class Real>
KPOLARIS_INLINE TransportState<Real> rk4_step(const Metric& metric,
                                           const TransportState<Real>& s,
                                           Real h) {
    const TransportState<Real> k1 = transport_rhs(metric, s);
    const TransportState<Real> k2 = transport_rhs(metric, axpy(s, Real(0.5) * h, k1));
    const TransportState<Real> k3 = transport_rhs(metric, axpy(s, Real(0.5) * h, k2));
    const TransportState<Real> k4 = transport_rhs(metric, axpy(s, h, k3));

    return s + (k1 + Real(2) * k2 + Real(2) * k3 + k4) * (h / Real(6));
}


template<class Metric, class Real>
KPOLARIS_INLINE TransportState<Real> rk4_step_from_k1(const Metric& metric,
                                                   const TransportState<Real>& s,
                                                   Real h,
                                                   const TransportState<Real>& k1) {
    const TransportState<Real> k2 = transport_rhs(metric, axpy(s, Real(0.5) * h, k1));
    const TransportState<Real> k3 = transport_rhs(metric, axpy(s, Real(0.5) * h, k2));
    const TransportState<Real> k4 = transport_rhs(metric, axpy(s, h, k3));

    return s + (k1 + Real(2) * k2 + Real(2) * k3 + k4) * (h / Real(6));
}

template<class Metric, class Real>
KPOLARIS_INLINE TransportState<Real> rk4_midpoint_state(const Metric& metric,
                                                     const TransportState<Real>& s,
                                                     Real h) {
    const TransportState<Real> k1 = transport_rhs(metric, s);
    const TransportState<Real> k2 = transport_rhs(metric, axpy(s, Real(0.5) * h, k1));
    return axpy(s, Real(0.5) * h, k2);
}


// Adaptive step acceptance compares both geodesic variables and the
// transported screen basis.  x/k control the ray, while e1/e2 control the
// camera-frame polarization projection; both must converge for polarized images.
template<class Real>
KPOLARIS_INLINE Real state_error_norm(const TransportState<Real>& a,
                                   const TransportState<Real>& b) {
    Real err = Real(0);
    for (int mu = 0; mu < ndim; ++mu) {
        const Real sx = max_val(Real(1), max_val(abs_val(a.x[mu]), abs_val(b.x[mu])));
        const Real sk = max_val(Real(1), max_val(abs_val(a.k[mu]), abs_val(b.k[mu])));
        const Real se1 = max_val(Real(1), max_val(abs_val(a.e1[mu]), abs_val(b.e1[mu])));
        const Real se2 = max_val(Real(1), max_val(abs_val(a.e2[mu]), abs_val(b.e2[mu])));
        err = max_val(err, abs_val(a.x[mu] - b.x[mu]) / sx);
        err = max_val(err, abs_val(a.k[mu] - b.k[mu]) / sk);
        err = max_val(err, abs_val(a.e1[mu] - b.e1[mu]) / se1);
        err = max_val(err, abs_val(a.e2[mu] - b.e2[mu]) / se2);
    }
    return err;
}

template<class Real = DefaultReal>
struct AdaptiveRK4Control {
    Real tolerance = Real(1e-7);
    Real min_step = Real(1e-5);
    Real max_step = Real(1);
    int max_attempts = 10;
};

template<class Real = DefaultReal>
struct AdaptiveRK4Step {
    TransportState<Real> state;
    TransportState<Real> mid_state;
    Real used_h = Real(0);
    Real next_h = Real(0);
    Real error = Real(0);
    int accepted = 0;
};

template<class Metric, class Real>
KPOLARIS_INLINE AdaptiveRK4Step<Real> adaptive_rk4_step(const Metric& metric,
                                                     const TransportState<Real>& state,
                                                     Real requested_h,
                                                     AdaptiveRK4Control<Real> control) {
    AdaptiveRK4Step<Real> out;
    Real h = min_val(max_val(abs_val(requested_h), control.min_step), control.max_step);
    if (requested_h < Real(0)) {
        h = -h;
    }

    Real last_err = large_positive<Real>();
    for (int attempt = 0; attempt < control.max_attempts; ++attempt) {
        const TransportState<Real> k1 = transport_rhs(metric, state);
        const TransportState<Real> full = rk4_step_from_k1(metric, state, h, k1);
        const TransportState<Real> half = rk4_step_from_k1(metric, state, h * Real(0.5), k1);
        const TransportState<Real> two_half = rk4_step(metric, half, h * Real(0.5));
        const Real err = state_error_norm(full, two_half);
        last_err = err;
        const Real tol = max_val(control.tolerance, adaptive_tolerance_floor<Real>());
        if (err <= tol) {
            Real factor = Real(2);
            if (err > Real(0)) {
                factor = Real(0.9) * Kokkos::pow(tol / err, Real(0.2));
                factor = min_val(Real(2), max_val(Real(0.2), factor));
            }
            out.state = two_half;
            out.mid_state = half;
            out.used_h = h;
            out.next_h = min_val(control.max_step, max_val(control.min_step, abs_val(h) * factor));
            if (requested_h < Real(0)) {
                out.next_h = -out.next_h;
            }
            out.error = err;
            out.accepted = 1;
            return out;
        }
        if (abs_val(h) <= control.min_step * Real(1.0001)) {
            break;
        }
        Real factor = Real(0.9) * Kokkos::pow(tol / max_val(err, tiny_positive<Real>()), Real(0.25));
        factor = min_val(Real(0.8), max_val(Real(0.1), factor));
        h *= factor;
    }
    const Real next_abs = max_val(control.min_step, abs_val(h) * Real(0.5));
    out.state = state;
    out.mid_state = state;
    out.used_h = Real(0);
    out.next_h = requested_h < Real(0) ? -next_abs : next_abs;
    out.error = last_err;
    out.accepted = 0;
    return out;
}


template<class Metric, class Real>
KPOLARIS_INLINE AdaptiveRK4Step<Real> adaptive_rkf45_endpoint_step(
    const Metric& metric,
    const TransportState<Real>& state,
    Real requested_h,
    AdaptiveRK4Control<Real> control) {
    AdaptiveRK4Step<Real> out;
    Real h = min_val(max_val(abs_val(requested_h), control.min_step), control.max_step);
    if (requested_h < Real(0)) {
        h = -h;
    }

    Real last_err = large_positive<Real>();
    for (int attempt = 0; attempt < control.max_attempts; ++attempt) {
        const TransportState<Real> k1 = transport_rhs(metric, state);
        const TransportState<Real> k2 =
            transport_rhs(metric, axpy(state, h * Real(1) / Real(5), k1));
        const TransportState<Real> k3 = transport_rhs(
            metric,
            state + (k1 * (Real(3) / Real(40)) +
                     k2 * (Real(9) / Real(40))) * h);
        const TransportState<Real> k4 = transport_rhs(
            metric,
            state + (k1 * (Real(3) / Real(10)) +
                     k2 * (Real(-9) / Real(10)) +
                     k3 * (Real(6) / Real(5))) * h);
        const TransportState<Real> k5 = transport_rhs(
            metric,
            state + (k1 * (Real(-11) / Real(54)) +
                     k2 * (Real(5) / Real(2)) +
                     k3 * (Real(-70) / Real(27)) +
                     k4 * (Real(35) / Real(27))) * h);
        const TransportState<Real> k6 = transport_rhs(
            metric,
            state + (k1 * (Real(1631) / Real(55296)) +
                     k2 * (Real(175) / Real(512)) +
                     k3 * (Real(575) / Real(13824)) +
                     k4 * (Real(44275) / Real(110592)) +
                     k5 * (Real(253) / Real(4096))) * h);

        const TransportState<Real> fifth =
            state + (k1 * (Real(37) / Real(378)) +
                     k3 * (Real(250) / Real(621)) +
                     k4 * (Real(125) / Real(594)) +
                     k6 * (Real(512) / Real(1771))) * h;
        const TransportState<Real> fourth =
            state + (k1 * (Real(2825) / Real(27648)) +
                     k3 * (Real(18575) / Real(48384)) +
                     k4 * (Real(13525) / Real(55296)) +
                     k5 * (Real(277) / Real(14336)) +
                     k6 * (Real(1) / Real(4))) * h;
        const Real err = state_error_norm(fifth, fourth);
        last_err = err;
        const Real tol = max_val(control.tolerance, adaptive_tolerance_floor<Real>());
        if (err <= tol) {
            Real factor = Real(2);
            if (err > Real(0)) {
                factor = Real(0.9) * Kokkos::pow(tol / err, Real(0.2));
                factor = min_val(Real(2), max_val(Real(0.2), factor));
            }
            out.state = fifth;
            out.mid_state = fifth;
            out.used_h = h;
            out.next_h = min_val(control.max_step, max_val(control.min_step, abs_val(h) * factor));
            if (requested_h < Real(0)) {
                out.next_h = -out.next_h;
            }
            out.error = err;
            out.accepted = 1;
            return out;
        }
        if (abs_val(h) <= control.min_step * Real(1.0001)) {
            break;
        }
        Real factor = Real(0.9) * Kokkos::pow(tol / max_val(err, tiny_positive<Real>()), Real(0.2));
        factor = min_val(Real(0.8), max_val(Real(0.1), factor));
        h *= factor;
    }
    const Real next_abs = max_val(control.min_step, abs_val(h) * Real(0.5));
    out.state = state;
    out.mid_state = state;
    out.used_h = Real(0);
    out.next_h = requested_h < Real(0) ? -next_abs : next_abs;
    out.error = last_err;
    out.accepted = 0;
    return out;
}

template<class Metric, class Real>
KPOLARIS_INLINE TransportState<Real> integrate_fixed_rk4(const Metric& metric,
                                                      TransportState<Real> s,
                                                      Real h,
                                                      int nsteps) {
    for (int n = 0; n < nsteps; ++n) {
        s = rk4_step(metric, s, h);
    }
    return s;
}

} // namespace kpolaris
