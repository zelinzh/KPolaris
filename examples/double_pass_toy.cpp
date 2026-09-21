#include <cmath>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "KPolaris.hpp"

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        using Real = kpolaris::DefaultReal;
        using ExecSpace = Kokkos::DefaultExecutionSpace;

        kpolaris::DoublePassToyParams<Real> params;
        params.nx = 64;
        params.ny = 64;
        params.nsteps = 96;
        params.camera_z = Real(2);
        params.inner_z = Real(0.2);
        params.fov = Real(1);
        params.coeffs.jI = Real(1);
        params.coeffs.jQ = Real(0.08);
        params.coeffs.aI = Real(0.15);
        params.coeffs.rV = Real(0.4);

        const int npix = params.nx * params.ny;
        Kokkos::View<Real*, ExecSpace> image_i("I", npix);
        Kokkos::View<Real*, ExecSpace> image_q("Q", npix);
        Kokkos::View<Real*, ExecSpace> image_u("U", npix);
        Kokkos::View<Real*, ExecSpace> image_v("V", npix);
        Kokkos::View<Real*, ExecSpace> closure("closure", npix);
        Kokkos::View<Real*, ExecSpace> frame_error("frame_error", npix);

        kpolaris::run_double_pass_toy<ExecSpace>(params, image_i, image_q, image_u,
                                              image_v, closure, frame_error);
        Kokkos::fence();

        auto h_i = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_i);
        auto h_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_q);
        auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_u);
        auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_v);
        auto h_closure = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), closure);
        auto h_frame_error = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), frame_error);

        Real sum_i = Real(0);
        Real sum_q = Real(0);
        Real sum_u = Real(0);
        Real sum_v = Real(0);
        Real max_closure = Real(0);
        Real max_frame = Real(0);
        for (int p = 0; p < npix; ++p) {
            sum_i += h_i(p);
            sum_q += h_q(p);
            sum_u += h_u(p);
            sum_v += h_v(p);
            max_closure = std::max(max_closure, std::abs(h_closure(p)));
            max_frame = std::max(max_frame, std::abs(h_frame_error(p)));
        }

        std::cout << "KPolaris double-pass toy\n";
        std::cout << "pixels " << npix << "\n";
        std::cout << "I " << sum_i << "\n";
        std::cout << "Q " << sum_q << "\n";
        std::cout << "U " << sum_u << "\n";
        std::cout << "V " << sum_v << "\n";
        std::cout << "max_closure " << max_closure << "\n";
        std::cout << "max_frame_error " << max_frame << "\n";
    }
    Kokkos::finalize();
    return 0;
}
