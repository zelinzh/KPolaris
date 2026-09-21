#include <cmath>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "KPolaris.hpp"

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        using Real = kpolaris::DefaultReal;
        using ExecSpace = Kokkos::DefaultExecutionSpace;

        kpolaris::PassAParams<Real> params;
        params.camera.nx = 128;
        params.camera.ny = 128;
        params.camera.radius = Real(40);
        params.camera.inclination = Real(1.04719755119659774615);
        params.camera.fov = Real(0.12);
        params.mass = Real(1);
        params.spin = Real(0.9375);
        params.inner_radius = Real(2.0);
        params.outer_radius = Real(100);
        params.step = Real(0.025);
        params.max_steps = 4096;

        const int npix = params.camera.nx * params.camera.ny;
        Kokkos::View<Real*, ExecSpace> final_r("final_r", npix);
        Kokkos::View<Real*, ExecSpace> min_r("min_r", npix);
        Kokkos::View<Real*, ExecSpace> initial_null("initial_null", npix);
        Kokkos::View<Real*, ExecSpace> final_null("final_null", npix);
        Kokkos::View<Real*, ExecSpace> frame_error("frame_error", npix);
        Kokkos::View<int*, ExecSpace> steps("steps", npix);
        Kokkos::View<int*, ExecSpace> reason("reason", npix);

        kpolaris::run_pass_a<ExecSpace>(params, final_r, min_r, initial_null,
                                     final_null, frame_error, steps, reason);
        Kokkos::fence();

        auto h_final_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), final_r);
        auto h_min_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), min_r);
        auto h_initial_null = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), initial_null);
        auto h_final_null = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), final_null);
        auto h_frame_error = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), frame_error);
        auto h_steps = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), steps);
        auto h_reason = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reason);

        int inner_hits = 0;
        int escaped = 0;
        int maxed = 0;
        int min_steps = params.max_steps;
        int max_steps = 0;
        Real min_radius = params.camera.radius;
        Real max_initial_null = Real(0);
        Real max_final_null = Real(0);
        Real max_frame = Real(0);
        Real min_final_r = params.camera.radius;
        Real max_final_r = Real(0);
        for (int p = 0; p < npix; ++p) {
            const auto rr = static_cast<kpolaris::TerminationReason>(h_reason(p));
            if (rr == kpolaris::TerminationReason::reached_inner_boundary) {
                inner_hits += 1;
            } else if (rr == kpolaris::TerminationReason::escaped_domain) {
                escaped += 1;
            } else if (rr == kpolaris::TerminationReason::max_steps) {
                maxed += 1;
            }
            min_steps = std::min(min_steps, h_steps(p));
            max_steps = std::max(max_steps, h_steps(p));
            min_radius = std::min(min_radius, h_min_r(p));
            min_final_r = std::min(min_final_r, h_final_r(p));
            max_final_r = std::max(max_final_r, h_final_r(p));
            max_initial_null = std::max(max_initial_null, std::abs(h_initial_null(p)));
            max_final_null = std::max(max_final_null, std::abs(h_final_null(p)));
            max_frame = std::max(max_frame, std::abs(h_frame_error(p)));
        }

        std::cout << "KPolaris Pass A smoke\n";
        std::cout << "pixels " << npix << "\n";
        std::cout << "inner_hits " << inner_hits << "\n";
        std::cout << "escaped " << escaped << "\n";
        std::cout << "max_steps_terminated " << maxed << "\n";
        std::cout << "step_range " << min_steps << " " << max_steps << "\n";
        std::cout << "min_radius " << min_radius << "\n";
        std::cout << "final_r_range " << min_final_r << " " << max_final_r << "\n";
        std::cout << "max_initial_null " << max_initial_null << "\n";
        std::cout << "max_final_null " << max_final_null << "\n";
        std::cout << "max_frame_error " << max_frame << "\n";
    }
    Kokkos::finalize();
    return 0;
}
