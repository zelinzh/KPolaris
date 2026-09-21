#pragma once

#include <iostream>
#include <syncstream>
#include <vector>

#include <Kokkos_Core.hpp>

#include "geodesic/pass_b.hpp"
#include "image/result.hpp"

namespace kpolaris_image_detail {

using Real = kpolaris::DefaultReal;

template<class View>
std::vector<Real> copy_real_view_1d(const View& view) {
    const size_t n = static_cast<size_t>(view.extent(0));
    std::vector<Real> out(n);
    using HostView = Kokkos::View<Real*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    HostView host(out.data(), n);
    Kokkos::deep_copy(host, view);
    return out;
}

template<class View>
std::vector<int> copy_int_view_1d(const View& view) {
    const size_t n = static_cast<size_t>(view.extent(0));
    std::vector<int> out(n);
    using HostView = Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    HostView host(out.data(), n);
    Kokkos::deep_copy(host, view);
    return out;
}

template<class View>
std::vector<Real> copy_real_view_2d(const View& view) {
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    const int nfreq = static_cast<int>(view.extent(0));
    const int npix = static_cast<int>(view.extent(1));
    std::vector<Real> out(static_cast<size_t>(nfreq) * static_cast<size_t>(npix));
    for (int f = 0; f < nfreq; ++f) {
        for (int p = 0; p < npix; ++p) {
            out[static_cast<size_t>(f) * static_cast<size_t>(npix) + static_cast<size_t>(p)] = host(f, p);
        }
    }
    return out;
}

inline void report_image_timing(int timing, const char* name, double seconds) {
    if (timing) {
        std::osyncstream(std::cout) << "timing " << name << ' ' << seconds << " s\n";
    }
}

template<class ExecSpace, class RealT, class RadiationModel, class Metric>
void run_pass_b_segment_model_with_overlap_metric(
    const kpolaris::PassAParams<RealT>& params,
    const RadiationModel& radiation_model,
    int radiation_substeps,
    const Metric& metric,
    Kokkos::View<RealT*, ExecSpace> image_i,
    Kokkos::View<RealT*, ExecSpace> image_q,
    Kokkos::View<RealT*, ExecSpace> image_u,
    Kokkos::View<RealT*, ExecSpace> image_v,
    Kokkos::View<RealT*, ExecSpace> closure_x,
    Kokkos::View<RealT*, ExecSpace> closure_k,
    Kokkos::View<RealT*, ExecSpace> final_null,
    Kokkos::View<RealT*, ExecSpace> frame_error,
    Kokkos::View<RealT*, ExecSpace> det_r,
    Kokkos::View<RealT*, ExecSpace> overlap_r11,
    Kokkos::View<RealT*, ExecSpace> overlap_r12,
    Kokkos::View<RealT*, ExecSpace> overlap_r21,
    Kokkos::View<RealT*, ExecSpace> overlap_r22,
    Kokkos::View<RealT*, ExecSpace> basis_identity_error,
    Kokkos::View<RealT*, ExecSpace> basis_rotation_angle,
    Kokkos::View<int*, ExecSpace> pass_a_steps,
    Kokkos::View<int*, ExecSpace> steps,
    Kokkos::View<int*, ExecSpace> reason,
    const ExecSpace& exec = ExecSpace()) {
    const int npix = params.camera.nx * params.camera.ny;
    Kokkos::parallel_for(
        "KPOLARISImageMetric",
        Kokkos::RangePolicy<ExecSpace>(exec, 0, npix),
        KOKKOS_LAMBDA(const int pixel) {
            const kpolaris::PassBResult<RealT> r =
                kpolaris::trace_pass_b_segment_model_pixel_metric(
                    pixel, params, radiation_model, radiation_substeps, metric);
            image_i(pixel) = r.observed_stokes.I;
            image_q(pixel) = r.observed_stokes.Q;
            image_u(pixel) = r.observed_stokes.U;
            image_v(pixel) = r.observed_stokes.V;
            closure_x(pixel) = r.closure_x;
            closure_k(pixel) = r.closure_k;
            final_null(pixel) = r.final_null;
            frame_error(pixel) = r.frame_error;
            det_r(pixel) = r.overlap.det();
            overlap_r11(pixel) = r.overlap.r11;
            overlap_r12(pixel) = r.overlap.r12;
            overlap_r21(pixel) = r.overlap.r21;
            overlap_r22(pixel) = r.overlap.r22;
            basis_identity_error(pixel) =
                kpolaris::max_val(kpolaris::max_val(kpolaris::abs_val(r.overlap.r11 - RealT(1)),
                                             kpolaris::abs_val(r.overlap.r22 - RealT(1))),
                               kpolaris::max_val(kpolaris::abs_val(r.overlap.r12),
                                             kpolaris::abs_val(r.overlap.r21)));
            basis_rotation_angle(pixel) =
                Kokkos::atan2(r.overlap.r21 - r.overlap.r12,
                              r.overlap.r11 + r.overlap.r22);
            pass_a_steps(pixel) = r.pass_a.steps;
            steps(pixel) = r.steps;
            reason(pixel) = static_cast<int>(r.reason);
        });
}

template<class Metric, class RadiationModel>
ImageHostData run_image_metric(
    const kpolaris::PassAParams<Real>& pass_a,
    RadiationModel model,
    int radiation_substeps,
    int timing,
    const Metric& metric,
    const char* timing_suffix) {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    const int npix = pass_a.camera.nx * pass_a.camera.ny;

    Kokkos::View<Real*, ExecSpace> image_i("I", npix);
    Kokkos::View<Real*, ExecSpace> image_q("Q", npix);
    Kokkos::View<Real*, ExecSpace> image_u("U", npix);
    Kokkos::View<Real*, ExecSpace> image_v("V", npix);
    Kokkos::View<Real*, ExecSpace> closure_x("closure_x", npix);
    Kokkos::View<Real*, ExecSpace> closure_k("closure_k", npix);
    Kokkos::View<Real*, ExecSpace> final_null("final_null", npix);
    Kokkos::View<Real*, ExecSpace> frame_error("frame_error", npix);
    Kokkos::View<Real*, ExecSpace> det_r("det_r", npix);
    Kokkos::View<Real*, ExecSpace> overlap_r11("overlap_r11", npix);
    Kokkos::View<Real*, ExecSpace> overlap_r12("overlap_r12", npix);
    Kokkos::View<Real*, ExecSpace> overlap_r21("overlap_r21", npix);
    Kokkos::View<Real*, ExecSpace> overlap_r22("overlap_r22", npix);
    Kokkos::View<Real*, ExecSpace> basis_identity_error("basis_identity_error", npix);
    Kokkos::View<Real*, ExecSpace> basis_rotation_angle("basis_rotation_angle", npix);
    Kokkos::View<int*, ExecSpace> pass_a_steps("pass_a_steps", npix);
    Kokkos::View<int*, ExecSpace> steps("steps", npix);
    Kokkos::View<int*, ExecSpace> reason("reason", npix);

    Kokkos::Timer kernel_timer;
    run_pass_b_segment_model_with_overlap_metric<ExecSpace>(
        pass_a, model, radiation_substeps, metric,
        image_i, image_q, image_u, image_v,
        closure_x, closure_k, final_null, frame_error, det_r,
        overlap_r11, overlap_r12, overlap_r21, overlap_r22,
        basis_identity_error, basis_rotation_angle, pass_a_steps, steps, reason);
    Kokkos::fence();
    (void)timing_suffix;
    report_image_timing(timing, "image_kernel_single", kernel_timer.seconds());

    Kokkos::Timer copy_timer;
    ImageHostData out;
    out.npix = npix;
    out.nfreq = 1;
    out.image_i = copy_real_view_1d(image_i);
    out.image_q = copy_real_view_1d(image_q);
    out.image_u = copy_real_view_1d(image_u);
    out.image_v = copy_real_view_1d(image_v);
    out.closure_x = copy_real_view_1d(closure_x);
    out.closure_k = copy_real_view_1d(closure_k);
    out.final_null = copy_real_view_1d(final_null);
    out.frame_error = copy_real_view_1d(frame_error);
    out.det_r = copy_real_view_1d(det_r);
    out.overlap_r11 = copy_real_view_1d(overlap_r11);
    out.overlap_r12 = copy_real_view_1d(overlap_r12);
    out.overlap_r21 = copy_real_view_1d(overlap_r21);
    out.overlap_r22 = copy_real_view_1d(overlap_r22);
    out.basis_identity_error = copy_real_view_1d(basis_identity_error);
    out.basis_rotation_angle = copy_real_view_1d(basis_rotation_angle);
    out.pass_a_steps = copy_int_view_1d(pass_a_steps);
    out.steps = copy_int_view_1d(steps);
    out.reason = copy_int_view_1d(reason);
    report_image_timing(timing, "image_device_to_host_single", copy_timer.seconds());
    return out;
}

} // namespace kpolaris_image_detail
