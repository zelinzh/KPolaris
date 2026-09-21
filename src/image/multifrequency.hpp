#pragma once

#include <stdexcept>

#include "image/driver.hpp"

#if KPOLARIS_MAX_FREQUENCIES > 1
namespace kpolaris_image_detail {

template<class ExecSpace, class RealT, class RadiationModel, class Metric>
void run_pass_b_segment_model_multifrequency_with_overlap_metric(
    const kpolaris::PassAParams<RealT>& params,
    RadiationModel radiation_model,
    int radiation_substeps,
    const RealT* frequencies,
    int nfreq,
    const Metric& metric,
    Kokkos::View<RealT**, ExecSpace> image_i,
    Kokkos::View<RealT**, ExecSpace> image_q,
    Kokkos::View<RealT**, ExecSpace> image_u,
    Kokkos::View<RealT**, ExecSpace> image_v,
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
    Kokkos::View<int*, ExecSpace> reason) {
    const int npix = params.camera.nx * params.camera.ny;
    Kokkos::parallel_for(
        "KPOLARISImageMetricMultiFreq",
        Kokkos::RangePolicy<ExecSpace>(0, npix),
        KOKKOS_LAMBDA(const int pixel) {
            RealT local_frequencies[KPOLARIS_MAX_FREQUENCIES];
            kpolaris::Stokes<RealT> local_stokes[KPOLARIS_MAX_FREQUENCIES];
            for (int f = 0; f < nfreq; ++f) {
                local_frequencies[f] = frequencies[f];
            }
            const kpolaris::MultiFrequencyPassBResult<RealT> r =
                kpolaris::trace_pass_b_segment_model_multifrequency_pixel_metric(
                    pixel, params, radiation_model, radiation_substeps,
                    local_frequencies, nfreq, local_stokes, metric);
            for (int f = 0; f < nfreq; ++f) {
                image_i(f, pixel) = local_stokes[f].I;
                image_q(f, pixel) = local_stokes[f].Q;
                image_u(f, pixel) = local_stokes[f].U;
                image_v(f, pixel) = local_stokes[f].V;
            }
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

template<class ExecSpace, class RealT, class RadiationModel, class Metric>
void run_pass_b_segment_model_multifrequency_control_with_overlap_metric(
    const kpolaris::PassAParams<RealT>& params,
    RadiationModel radiation_model,
    int radiation_substeps,
    const RealT* frequencies,
    int nfreq,
    const RealT* step_control_frequencies,
    int step_control_nfreq,
    const Metric& metric,
    Kokkos::View<RealT**, ExecSpace> image_i,
    Kokkos::View<RealT**, ExecSpace> image_q,
    Kokkos::View<RealT**, ExecSpace> image_u,
    Kokkos::View<RealT**, ExecSpace> image_v,
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
    Kokkos::View<int*, ExecSpace> reason) {
    const int npix = params.camera.nx * params.camera.ny;
    Kokkos::parallel_for(
        "KPOLARISImageMetricMultiFreqControl",
        Kokkos::RangePolicy<ExecSpace>(0, npix),
        KOKKOS_LAMBDA(const int pixel) {
            RealT local_frequencies[KPOLARIS_MAX_FREQUENCIES];
            kpolaris::Stokes<RealT> local_stokes[KPOLARIS_MAX_FREQUENCIES];
            for (int f = 0; f < nfreq; ++f) {
                local_frequencies[f] = frequencies[f];
            }
            const kpolaris::MultiFrequencyPassBResult<RealT> r =
                kpolaris::trace_pass_b_segment_model_multifrequency_control_pixel_metric(
                    pixel, params, radiation_model, radiation_substeps,
                    local_frequencies, nfreq, step_control_frequencies,
                    step_control_nfreq, local_stokes, metric);
            for (int f = 0; f < nfreq; ++f) {
                image_i(f, pixel) = local_stokes[f].I;
                image_q(f, pixel) = local_stokes[f].Q;
                image_u(f, pixel) = local_stokes[f].U;
                image_v(f, pixel) = local_stokes[f].V;
            }
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
ImageHostData run_multifrequency_image_metric_with_step_control(
    const kpolaris::PassAParams<Real>& pass_a,
    RadiationModel model,
    int radiation_substeps,
    const std::vector<Real>& frequencies,
    const std::vector<Real>& step_control_frequencies,
    int timing,
    const Metric& metric,
    const char* timing_suffix) {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    const int npix = pass_a.camera.nx * pass_a.camera.ny;
    const int nfreq = static_cast<int>(frequencies.size());
    const int step_control_nfreq = static_cast<int>(step_control_frequencies.size());
    if (nfreq < 1 || nfreq > KPOLARIS_MAX_FREQUENCIES) {
        throw std::invalid_argument(
            "multifrequency image chunk size must be between 1 and "
            "KPOLARIS_MAX_FREQUENCIES");
    }
    if (step_control_nfreq < 1) {
        throw std::invalid_argument(
            "multifrequency step-control frequency list must not be empty");
    }
    bool separate_step_control = step_control_nfreq != nfreq;
    if (!separate_step_control) {
        for (int f = 0; f < nfreq; ++f) {
            if (frequencies[static_cast<size_t>(f)] != step_control_frequencies[static_cast<size_t>(f)]) {
                separate_step_control = true;
                break;
            }
        }
    }

    Kokkos::View<Real*, ExecSpace> freq_view("frequencies", nfreq);
    auto h_freq = Kokkos::create_mirror_view(freq_view);
    for (int f = 0; f < nfreq; ++f) {
        h_freq(f) = frequencies[static_cast<size_t>(f)];
    }
    Kokkos::deep_copy(freq_view, h_freq);

    Kokkos::View<Real*, ExecSpace> step_control_freq_view("step_control_frequencies", step_control_nfreq);
    auto h_step_control_freq = Kokkos::create_mirror_view(step_control_freq_view);
    for (int f = 0; f < step_control_nfreq; ++f) {
        h_step_control_freq(f) = step_control_frequencies[static_cast<size_t>(f)];
    }
    Kokkos::deep_copy(step_control_freq_view, h_step_control_freq);

    Kokkos::View<Real**, ExecSpace> image_i("I_multi", nfreq, npix);
    Kokkos::View<Real**, ExecSpace> image_q("Q_multi", nfreq, npix);
    Kokkos::View<Real**, ExecSpace> image_u("U_multi", nfreq, npix);
    Kokkos::View<Real**, ExecSpace> image_v("V_multi", nfreq, npix);
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
    if (separate_step_control) {
        run_pass_b_segment_model_multifrequency_control_with_overlap_metric<ExecSpace>(
            pass_a, model, radiation_substeps, freq_view.data(), nfreq,
            step_control_freq_view.data(), step_control_nfreq, metric,
            image_i, image_q, image_u, image_v,
            closure_x, closure_k, final_null, frame_error, det_r,
            overlap_r11, overlap_r12, overlap_r21, overlap_r22,
            basis_identity_error, basis_rotation_angle, pass_a_steps, steps, reason);
    } else {
        run_pass_b_segment_model_multifrequency_with_overlap_metric<ExecSpace>(
            pass_a, model, radiation_substeps, freq_view.data(), nfreq, metric,
            image_i, image_q, image_u, image_v,
            closure_x, closure_k, final_null, frame_error, det_r,
            overlap_r11, overlap_r12, overlap_r21, overlap_r22,
            basis_identity_error, basis_rotation_angle, pass_a_steps, steps, reason);
    }
    Kokkos::fence();
    (void)timing_suffix;
    report_image_timing(timing, "image_kernel_multifrequency", kernel_timer.seconds());

    Kokkos::Timer copy_timer;
    ImageHostData out;
    out.npix = npix;
    out.nfreq = nfreq;
    out.image_i = copy_real_view_2d(image_i);
    out.image_q = copy_real_view_2d(image_q);
    out.image_u = copy_real_view_2d(image_u);
    out.image_v = copy_real_view_2d(image_v);
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
    report_image_timing(timing, "image_device_to_host_multifrequency", copy_timer.seconds());
    return out;
}


template<class Metric, class RadiationModel>
ImageHostData run_multifrequency_image_metric(
    const kpolaris::PassAParams<Real>& pass_a,
    RadiationModel model,
    int radiation_substeps,
    const std::vector<Real>& frequencies,
    int timing,
    const Metric& metric,
    const char* timing_suffix) {
    return run_multifrequency_image_metric_with_step_control(
        pass_a, model, radiation_substeps, frequencies, frequencies,
        timing, metric, timing_suffix);
}

} // namespace kpolaris_image_detail
#endif
