#pragma once

#include <Kokkos_Core.hpp>

#include "geodesic/rk4.hpp"
#include "geometry/kerr_schild_cartesian.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct KerrGeodesicSmokeParams {
    int nrays = 32;
    int nsteps = 128;
    Real mass = Real(1);
    Real spin = Real(0);
    Real r0 = Real(20);
    Real impact_span = Real(0.5);
    Real step = Real(0.01);
};

template<class Real>
KPOLARIS_INLINE TransportState<Real> make_kerr_smoke_state(
    int ray,
    const KerrGeodesicSmokeParams<Real>& p) {
    const Real frac = p.nrays > 1 ? Real(ray) / Real(p.nrays - 1) : Real(0.5);
    const Real y = (frac - Real(0.5)) * p.impact_span;
    const Real xcoord = Kokkos::sqrt(max_val(p.r0 * p.r0 - y * y, Real(0)));
    const Real inv_r = Real(1) / p.r0;
    const Real nx = xcoord * inv_r;
    const Real ny = y * inv_r;

    TransportState<Real> s;
    s.x = Vec4<Real>(Real(0), xcoord, y, Real(0));
    s.k = Vec4<Real>(Real(1), -nx, -ny, Real(0));
    s.e1 = Vec4<Real>(Real(0), -ny, nx, Real(0));
    s.e2 = Vec4<Real>(Real(0), Real(0), Real(0), Real(1));
    return s;
}

template<class ExecSpace, class Real>
void run_kerr_geodesic_smoke(const KerrGeodesicSmokeParams<Real>& params,
                             Kokkos::View<Real*, ExecSpace> initial_null,
                             Kokkos::View<Real*, ExecSpace> final_null,
                             Kokkos::View<Real*, ExecSpace> initial_r,
                             Kokkos::View<Real*, ExecSpace> final_r,
                             Kokkos::View<Real*, ExecSpace> frame_error) {
    Kokkos::parallel_for(
        "KPOLARISKerrGeodesicSmoke",
        Kokkos::RangePolicy<ExecSpace>(0, params.nrays),
        KOKKOS_LAMBDA(const int ray) {
            const KerrSchildInMetric<Real> metric(params.mass, params.spin);
            TransportState<Real> state = make_kerr_smoke_state(ray, params);
            initial_null(ray) = metric.dot(state.x, state.k, state.k);
            initial_r(ray) = spatial_radius(state.x);
            for (int n = 0; n < params.nsteps; ++n) {
                state = rk4_step(metric, state, params.step);
            }
            final_null(ray) = metric.dot(state.x, state.k, state.k);
            final_r(ray) = spatial_radius(state.x);
            frame_error(ray) = max_frame_error(frame_errors(metric, state));
        });
}

} // namespace kpolaris
