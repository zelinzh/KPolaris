#pragma once

#include <Kokkos_Core.hpp>

#include "geodesic/pass_a.hpp"

namespace kpolaris {

template<class ExecSpace, class Real>
void run_pass_a(const PassAParams<Real>& params,
                Kokkos::View<Real*, ExecSpace> final_r,
                Kokkos::View<Real*, ExecSpace> min_r,
                Kokkos::View<Real*, ExecSpace> initial_null,
                Kokkos::View<Real*, ExecSpace> final_null,
                Kokkos::View<Real*, ExecSpace> frame_error,
                Kokkos::View<int*, ExecSpace> steps,
                Kokkos::View<int*, ExecSpace> reason) {
    const int npix = params.camera.nx * params.camera.ny;
    Kokkos::parallel_for(
        "KPOLARISPassA",
        Kokkos::RangePolicy<ExecSpace>(0, npix),
        KOKKOS_LAMBDA(const int pixel) {
            const PassAResult<Real> r = trace_pass_a_pixel(pixel, params);
            final_r(pixel) = r.final_radius;
            min_r(pixel) = r.min_radius;
            initial_null(pixel) = r.initial_null;
            final_null(pixel) = r.final_null;
            frame_error(pixel) = r.frame_error;
            steps(pixel) = r.steps;
            reason(pixel) = static_cast<int>(r.reason);
        });
}

} // namespace kpolaris
