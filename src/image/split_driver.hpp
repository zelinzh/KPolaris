#pragma once

#include "image/driver.hpp"

namespace kpolaris_image_detail {

template<class ExecSpace, class RealT, class RadiationModel, class Metric>
void run_split_pass_b_segment_model_with_overlap_metric(
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
    int timing) {
    const int npix = params.camera.nx * params.camera.ny;
    Kokkos::View<RealT*, ExecSpace> endpoint_x("endpoint_x", npix * kpolaris::ndim);
    Kokkos::View<RealT*, ExecSpace> endpoint_k("endpoint_k", npix * kpolaris::ndim);
    Kokkos::View<RealT*, ExecSpace> endpoint_e1("endpoint_e1", npix * kpolaris::ndim);
    Kokkos::View<RealT*, ExecSpace> endpoint_e2("endpoint_e2", npix * kpolaris::ndim);
    Kokkos::View<int*, ExecSpace> endpoint_valid("endpoint_valid", npix);
    Kokkos::View<int*, ExecSpace> endpoint_reason("endpoint_reason", npix);

    Kokkos::Timer pass_a_timer;
    Kokkos::parallel_for(
        "KPOLARISPassAEndpointMetric",
        Kokkos::RangePolicy<ExecSpace>(0, npix),
        KOKKOS_LAMBDA(const int pixel) {
            const kpolaris::PassAEndpointResult<RealT> r =
                kpolaris::trace_pass_a_segment_endpoint_pixel_metric(pixel, params, metric);
            const int base = pixel * kpolaris::ndim;
            for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                endpoint_x(base + mu) = r.state.x[mu];
                endpoint_k(base + mu) = r.state.k[mu];
                endpoint_e1(base + mu) = r.state.e1[mu];
                endpoint_e2(base + mu) = r.state.e2[mu];
            }
            endpoint_valid(pixel) = r.valid;
            pass_a_steps(pixel) = r.steps;
            endpoint_reason(pixel) = static_cast<int>(r.reason);
        });
    Kokkos::fence();
    report_image_timing(timing, "image_pass_a_endpoint", pass_a_timer.seconds());

    Kokkos::Timer pass_b_timer;
    Kokkos::parallel_for(
        "KPOLARISPassBFromEndpointMetricOverlap",
        Kokkos::RangePolicy<ExecSpace>(0, npix),
        KOKKOS_LAMBDA(const int pixel) {
            kpolaris::TransportState<RealT> endpoint_state;
            const int base = pixel * kpolaris::ndim;
            for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                endpoint_state.x[mu] = endpoint_x(base + mu);
                endpoint_state.k[mu] = endpoint_k(base + mu);
                endpoint_state.e1[mu] = endpoint_e1(base + mu);
                endpoint_state.e2[mu] = endpoint_e2(base + mu);
            }
            const kpolaris::PassBResult<RealT> r =
                kpolaris::trace_pass_b_segment_model_endpoint_pixel_metric(
                    pixel, params, radiation_model, radiation_substeps, endpoint_state,
                    pass_a_steps(pixel),
                    static_cast<kpolaris::TerminationReason>(endpoint_reason(pixel)),
                    endpoint_valid(pixel), metric);
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
    Kokkos::fence();
    report_image_timing(timing, "image_pass_b_transport", pass_b_timer.seconds());
}

template<class Metric, class RadiationModel>
ImageHostData run_image_metric_with_optional_split(
    const kpolaris::PassAParams<Real>& pass_a,
    RadiationModel model,
    int radiation_substeps,
    int timing,
    int split_transport,
    const Metric& metric,
    const char* timing_suffix) {
    if (!split_transport) {
        return run_image_metric(pass_a, model, radiation_substeps, timing, metric, timing_suffix);
    }

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
    run_split_pass_b_segment_model_with_overlap_metric<ExecSpace>(
        pass_a, model, radiation_substeps, metric,
        image_i, image_q, image_u, image_v,
        closure_x, closure_k, final_null, frame_error, det_r,
        overlap_r11, overlap_r12, overlap_r21, overlap_r22,
        basis_identity_error, basis_rotation_angle, pass_a_steps, steps, reason,
        timing);
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
