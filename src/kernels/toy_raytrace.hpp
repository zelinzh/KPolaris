#pragma once

#include <Kokkos_Core.hpp>

#include "radiation/semi_analytic.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct ToyRaytraceParams {
    int nx = 16;
    int ny = 16;
    int nsteps = 64;
    Real length = Real(1);
    TransferCoeffs<Real> coeffs;
};

template<class Real>
KPOLARIS_INLINE Stokes<Real> trace_toy_pixel(int pixel,
                                          const ToyRaytraceParams<Real>& params) {
    const int i = pixel % params.nx;
    const int j = pixel / params.nx;
    const Real x = (Real(i) + Real(0.5)) / Real(params.nx) - Real(0.5);
    const Real y = (Real(j) + Real(0.5)) / Real(params.ny) - Real(0.5);

    TransferCoeffs<Real> coeffs = params.coeffs;
    coeffs.jI *= Real(1) + Real(0.2) * x;
    coeffs.jQ *= Real(1) + Real(0.1) * y;

    Stokes<Real> stokes;
    const Real dl = params.length / Real(params.nsteps);
    for (int n = 0; n < params.nsteps; ++n) {
        semi_analytic_stokes_step(stokes, coeffs, dl);
    }
    return stokes;
}

template<class ExecSpace, class Real>
void run_toy_raytrace(const ToyRaytraceParams<Real>& params,
                      Kokkos::View<Real*, ExecSpace> image_i,
                      Kokkos::View<Real*, ExecSpace> image_q,
                      Kokkos::View<Real*, ExecSpace> image_u,
                      Kokkos::View<Real*, ExecSpace> image_v) {
    const int npix = params.nx * params.ny;
    Kokkos::parallel_for(
        "KPOLARISToyRaytrace",
        Kokkos::RangePolicy<ExecSpace>(0, npix),
        KOKKOS_LAMBDA(const int pixel) {
            const Stokes<Real> s = trace_toy_pixel(pixel, params);
            image_i(pixel) = s.I;
            image_q(pixel) = s.Q;
            image_u(pixel) = s.U;
            image_v(pixel) = s.V;
        });
}

} // namespace kpolaris
