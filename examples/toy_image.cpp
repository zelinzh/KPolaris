#include <iostream>

#include <Kokkos_Core.hpp>

#include "KPolaris.hpp"

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        using Real = kpolaris::DefaultReal;
        using ExecSpace = Kokkos::DefaultExecutionSpace;

        kpolaris::ToyRaytraceParams<Real> params;
        params.nx = 32;
        params.ny = 32;
        params.nsteps = 64;
        params.length = Real(1);
        params.coeffs.jI = Real(1);
        params.coeffs.jQ = Real(0.15);
        params.coeffs.aI = Real(0.2);
        params.coeffs.rV = Real(0.7);

        const int npix = params.nx * params.ny;
        Kokkos::View<Real*, ExecSpace> image_i("I", npix);
        Kokkos::View<Real*, ExecSpace> image_q("Q", npix);
        Kokkos::View<Real*, ExecSpace> image_u("U", npix);
        Kokkos::View<Real*, ExecSpace> image_v("V", npix);

        kpolaris::run_toy_raytrace<ExecSpace>(params, image_i, image_q, image_u,
                                           image_v);
        Kokkos::fence();

        auto h_i =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_i);
        auto h_q =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_q);
        auto h_u =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_u);
        auto h_v =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_v);

        Real sum_i = Real(0);
        Real sum_q = Real(0);
        Real sum_u = Real(0);
        Real sum_v = Real(0);
        for (int p = 0; p < npix; ++p) {
            sum_i += h_i(p);
            sum_q += h_q(p);
            sum_u += h_u(p);
            sum_v += h_v(p);
        }

        std::cout << "KPolaris toy image\n";
        std::cout << "pixels " << npix << "\n";
        std::cout << "I " << sum_i << "\n";
        std::cout << "Q " << sum_q << "\n";
        std::cout << "U " << sum_u << "\n";
        std::cout << "V " << sum_v << "\n";
    }
    Kokkos::finalize();
    return 0;
}
