#pragma once

#include <Kokkos_Core.hpp>

#include "geodesic/rk4.hpp"
#include "geometry/minkowski.hpp"
#include "radiation/semi_analytic.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct DoublePassToyParams {
    int nx = 16;
    int ny = 16;
    int nsteps = 64;
    Real camera_z = Real(1);
    Real inner_z = Real(0);
    Real fov = Real(1);
    TransferCoeffs<Real> coeffs;
};

template<class Real = DefaultReal>
struct DoublePassToyResult {
    Stokes<Real> stokes;
    Real closure_x = Real(0);
    Real frame_error = Real(0);
    int steps = 0;
};

template<class Real>
KPOLARIS_INLINE TransportState<Real> make_camera_state(int pixel,
                                                    const DoublePassToyParams<Real>& p,
                                                    bool inward) {
    const int i = pixel % p.nx;
    const int j = pixel / p.nx;
    const Real sx = ((Real(i) + Real(0.5)) / Real(p.nx) - Real(0.5)) * p.fov;
    const Real sy = ((Real(j) + Real(0.5)) / Real(p.ny) - Real(0.5)) * p.fov;
    const Real kz = inward ? Real(-1) : Real(1);

    TransportState<Real> s;
    s.x = Vec4<Real>(Real(0), sx, sy, inward ? p.camera_z : p.inner_z);
    s.k = Vec4<Real>(Real(1), Real(0), Real(0), kz);
    s.e1 = Vec4<Real>(Real(0), Real(1), Real(0), Real(0));
    s.e2 = Vec4<Real>(Real(0), Real(0), Real(1), Real(0));
    return s;
}

template<class Real>
KPOLARIS_INLINE DoublePassToyResult<Real> trace_double_pass_toy_pixel(
    int pixel,
    const DoublePassToyParams<Real>& p) {
    const MinkowskiMetric<Real> metric;
    const Real span = p.camera_z - p.inner_z;
    const Real h = span / Real(p.nsteps);

    TransportState<Real> pass_a = make_camera_state(pixel, p, true);
    pass_a = integrate_fixed_rk4(metric, pass_a, h, p.nsteps);

    TransportState<Real> pass_b = pass_a;
    pass_b.k = Vec4<Real>(Real(1), Real(0), Real(0), Real(1));
    pass_b.e1 = Vec4<Real>(Real(0), Real(1), Real(0), Real(0));
    pass_b.e2 = Vec4<Real>(Real(0), Real(0), Real(1), Real(0));

    Stokes<Real> stokes;
    const Real dl = h;
    for (int n = 0; n < p.nsteps; ++n) {
        pass_b = rk4_step(metric, pass_b, h);
        semi_analytic_stokes_step(stokes, p.coeffs, dl);
    }

    const Real target_z = p.camera_z;
    const Real closure = abs_val(pass_b.x[3] - target_z);
    const Real frame_err = max_frame_error(frame_errors(metric, pass_b));

    DoublePassToyResult<Real> result;
    result.stokes = stokes;
    result.closure_x = closure;
    result.frame_error = frame_err;
    result.steps = 2 * p.nsteps;
    return result;
}

template<class ExecSpace, class Real>
void run_double_pass_toy(const DoublePassToyParams<Real>& params,
                         Kokkos::View<Real*, ExecSpace> image_i,
                         Kokkos::View<Real*, ExecSpace> image_q,
                         Kokkos::View<Real*, ExecSpace> image_u,
                         Kokkos::View<Real*, ExecSpace> image_v,
                         Kokkos::View<Real*, ExecSpace> closure,
                         Kokkos::View<Real*, ExecSpace> frame_error) {
    const int npix = params.nx * params.ny;
    Kokkos::parallel_for(
        "KPOLARISDoublePassToy",
        Kokkos::RangePolicy<ExecSpace>(0, npix),
        KOKKOS_LAMBDA(const int pixel) {
            const DoublePassToyResult<Real> r =
                trace_double_pass_toy_pixel(pixel, params);
            image_i(pixel) = r.stokes.I;
            image_q(pixel) = r.stokes.Q;
            image_u(pixel) = r.stokes.U;
            image_v(pixel) = r.stokes.V;
            closure(pixel) = r.closure_x;
            frame_error(pixel) = r.frame_error;
        });
}

} // namespace kpolaris
