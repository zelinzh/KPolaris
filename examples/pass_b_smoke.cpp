#include <cmath>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "KPolaris.hpp"

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        using Real = kpolaris::DefaultReal;
        using ExecSpace = Kokkos::DefaultExecutionSpace;

        kpolaris::PassBParams<Real> params;
        params.pass_a.camera.nx = 96;
        params.pass_a.camera.ny = 96;
        params.pass_a.camera.radius = Real(35);
        params.pass_a.camera.inclination = Real(1.04719755119659774615);
        params.pass_a.camera.fov = Real(0.08);
        params.pass_a.mass = Real(1);
        params.pass_a.spin = Real(0.9375);
        params.pass_a.inner_radius = Real(2.0);
        params.pass_a.outer_radius = Real(90);
        params.pass_a.step = Real(0.025);
        params.pass_a.max_steps = 4096;
        params.coeffs.jI = Real(0.8);
        params.coeffs.jQ = Real(0.05);
        params.coeffs.aI = Real(0.1);
        params.coeffs.rV = Real(0.2);
        params.radiation_substeps = 1;

        params.radiation_substeps = 2;

        kpolaris::RIAFAnalyticRadiationModel<Real> radiation_model;
        radiation_model.nth0 = Real(1.0);
        radiation_model.Te0 = Real(1.0);
        radiation_model.disk_h = Real(0.35);
        radiation_model.pow_nth = Real(-1.1);
        radiation_model.pow_T = Real(-0.84);
        radiation_model.r_min = Real(2.1);
        radiation_model.r_max = Real(35);
        radiation_model.emission_scale = Real(1);
        radiation_model.absorption_scale = Real(1);
        radiation_model.faraday_scale = Real(1);

        const int npix = params.pass_a.camera.nx * params.pass_a.camera.ny;
        Kokkos::View<Real*, ExecSpace> image_i("I", npix);
        Kokkos::View<Real*, ExecSpace> image_q("Q", npix);
        Kokkos::View<Real*, ExecSpace> image_u("U", npix);
        Kokkos::View<Real*, ExecSpace> image_v("V", npix);
        Kokkos::View<Real*, ExecSpace> closure_x("closure_x", npix);
        Kokkos::View<Real*, ExecSpace> closure_k("closure_k", npix);
        Kokkos::View<Real*, ExecSpace> final_null("final_null", npix);
        Kokkos::View<Real*, ExecSpace> frame_error("frame_error", npix);
        Kokkos::View<Real*, ExecSpace> det_r("det_r", npix);
        Kokkos::View<int*, ExecSpace> steps("steps", npix);
        Kokkos::View<int*, ExecSpace> reason("reason", npix);

        kpolaris::run_pass_b_model<ExecSpace>(params.pass_a, radiation_model,
                                           params.radiation_substeps, image_i,
                                           image_q, image_u, image_v, closure_x,
                                           closure_k, final_null, frame_error,
                                           det_r, steps, reason);
        Kokkos::fence();

        auto h_i = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_i);
        auto h_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_q);
        auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_u);
        auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_v);
        auto h_closure_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), closure_x);
        auto h_closure_k = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), closure_k);
        auto h_final_null = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), final_null);
        auto h_frame_error = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), frame_error);
        auto h_det_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), det_r);
        auto h_steps = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), steps);
        auto h_reason = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reason);

        int returned = 0;
        int failed = 0;
        int min_steps = params.pass_a.max_steps;
        int max_steps = 0;
        Real sum_i = Real(0);
        Real sum_q = Real(0);
        Real sum_u = Real(0);
        Real sum_v = Real(0);
        Real max_closure_x = Real(0);
        Real max_closure_k = Real(0);
        Real max_final_null = Real(0);
        Real max_frame = Real(0);
        Real min_det = Real(10);
        Real max_det = Real(-10);

        for (int p = 0; p < npix; ++p) {
            if (h_reason(p) == static_cast<int>(kpolaris::TerminationReason::reached_camera)) {
                returned += 1;
                sum_i += h_i(p);
                sum_q += h_q(p);
                sum_u += h_u(p);
                sum_v += h_v(p);
                min_steps = std::min(min_steps, h_steps(p));
                max_steps = std::max(max_steps, h_steps(p));
                max_closure_x = std::max(max_closure_x, std::abs(h_closure_x(p)));
                max_closure_k = std::max(max_closure_k, std::abs(h_closure_k(p)));
                max_final_null = std::max(max_final_null, std::abs(h_final_null(p)));
                max_frame = std::max(max_frame, std::abs(h_frame_error(p)));
                min_det = std::min(min_det, h_det_r(p));
                max_det = std::max(max_det, h_det_r(p));
            } else {
                failed += 1;
            }
        }

        std::cout << "KPolaris Pass B RIAF-model smoke\n";
        std::cout << "pixels " << npix << "\n";
        std::cout << "returned " << returned << "\n";
        std::cout << "failed " << failed << "\n";
        std::cout << "step_range " << min_steps << " " << max_steps << "\n";
        std::cout << "I " << sum_i << "\n";
        std::cout << "Q " << sum_q << "\n";
        std::cout << "U " << sum_u << "\n";
        std::cout << "V " << sum_v << "\n";
        std::cout << "max_closure_x " << max_closure_x << "\n";
        std::cout << "max_closure_k " << max_closure_k << "\n";
        std::cout << "max_final_null " << max_final_null << "\n";
        std::cout << "max_frame_error " << max_frame << "\n";
        std::cout << "det_range " << min_det << " " << max_det << "\n";
    }
    Kokkos::finalize();
    return 0;
}
