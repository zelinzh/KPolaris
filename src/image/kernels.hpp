#pragma once

#include <Kokkos_Core.hpp>
#include <type_traits>

#include "geodesic/pass_b.hpp"

namespace kpolaris {

template<class ExecSpace>
inline auto pixel_range_policy(int npix) {
#ifdef KOKKOS_ENABLE_OPENMP
    if constexpr (std::is_same_v<ExecSpace, Kokkos::OpenMP>) {
        return Kokkos::RangePolicy<ExecSpace, Kokkos::Schedule<Kokkos::Dynamic>>(0, npix);
    } else
#endif
    {
        return Kokkos::RangePolicy<ExecSpace>(0, npix);
    }
}

template<class ExecSpace, class Real>
void run_pass_b(const PassBParams<Real>& params,
                Kokkos::View<Real*, ExecSpace> image_i,
                Kokkos::View<Real*, ExecSpace> image_q,
                Kokkos::View<Real*, ExecSpace> image_u,
                Kokkos::View<Real*, ExecSpace> image_v,
                Kokkos::View<Real*, ExecSpace> closure_x,
                Kokkos::View<Real*, ExecSpace> closure_k,
                Kokkos::View<Real*, ExecSpace> final_null,
                Kokkos::View<Real*, ExecSpace> frame_error,
                Kokkos::View<Real*, ExecSpace> det_r,
                Kokkos::View<int*, ExecSpace> steps,
                Kokkos::View<int*, ExecSpace> reason) {
    const int npix = params.pass_a.camera.nx * params.pass_a.camera.ny;
    Kokkos::parallel_for(
        "KPOLARISPassB",
        pixel_range_policy<ExecSpace>(npix),
        KOKKOS_LAMBDA(const int pixel) {
            const PassBResult<Real> r = trace_pass_b_pixel(pixel, params);
            image_i(pixel) = r.observed_stokes.I;
            image_q(pixel) = r.observed_stokes.Q;
            image_u(pixel) = r.observed_stokes.U;
            image_v(pixel) = r.observed_stokes.V;
            closure_x(pixel) = r.closure_x;
            closure_k(pixel) = r.closure_k;
            final_null(pixel) = r.final_null;
            frame_error(pixel) = r.frame_error;
            det_r(pixel) = r.overlap.det();
            steps(pixel) = r.steps;
            reason(pixel) = static_cast<int>(r.reason);
        });
}


template<class ExecSpace, class Real, class RadiationModel>
void run_pass_b_model(const PassAParams<Real>& pass_a_params,
                      const RadiationModel& radiation_model,
                      int radiation_substeps,
                      Kokkos::View<Real*, ExecSpace> image_i,
                      Kokkos::View<Real*, ExecSpace> image_q,
                      Kokkos::View<Real*, ExecSpace> image_u,
                      Kokkos::View<Real*, ExecSpace> image_v,
                      Kokkos::View<Real*, ExecSpace> closure_x,
                      Kokkos::View<Real*, ExecSpace> closure_k,
                      Kokkos::View<Real*, ExecSpace> final_null,
                      Kokkos::View<Real*, ExecSpace> frame_error,
                      Kokkos::View<Real*, ExecSpace> det_r,
                      Kokkos::View<int*, ExecSpace> steps,
                      Kokkos::View<int*, ExecSpace> reason) {
    const int npix = pass_a_params.camera.nx * pass_a_params.camera.ny;
    Kokkos::parallel_for(
        "KPOLARISPassBModel",
        pixel_range_policy<ExecSpace>(npix),
        KOKKOS_LAMBDA(const int pixel) {
            const PassBResult<Real> r = trace_pass_b_model_pixel(
                pixel, pass_a_params, radiation_model, radiation_substeps);
            image_i(pixel) = r.observed_stokes.I;
            image_q(pixel) = r.observed_stokes.Q;
            image_u(pixel) = r.observed_stokes.U;
            image_v(pixel) = r.observed_stokes.V;
            closure_x(pixel) = r.closure_x;
            closure_k(pixel) = r.closure_k;
            final_null(pixel) = r.final_null;
            frame_error(pixel) = r.frame_error;
            det_r(pixel) = r.overlap.det();
            steps(pixel) = r.steps;
            reason(pixel) = static_cast<int>(r.reason);
        });
}


template<class ExecSpace, class Real, class RadiationModel>
void run_pass_b_segment_model(const PassAParams<Real>& params,
                              const RadiationModel& radiation_model,
                              int radiation_substeps,
                              Kokkos::View<Real*, ExecSpace> image_i,
                              Kokkos::View<Real*, ExecSpace> image_q,
                              Kokkos::View<Real*, ExecSpace> image_u,
                              Kokkos::View<Real*, ExecSpace> image_v,
                              Kokkos::View<Real*, ExecSpace> closure_x,
                              Kokkos::View<Real*, ExecSpace> closure_k,
                              Kokkos::View<Real*, ExecSpace> final_null,
                              Kokkos::View<Real*, ExecSpace> frame_error,
                              Kokkos::View<Real*, ExecSpace> det_r,
                              Kokkos::View<int*, ExecSpace> steps,
                              Kokkos::View<int*, ExecSpace> reason) {
    const int npix = params.camera.nx * params.camera.ny;
    Kokkos::parallel_for(
        "KPOLARISPassBSegmentModel",
        pixel_range_policy<ExecSpace>(npix),
        KOKKOS_LAMBDA(const int pixel) {
            const PassBResult<Real> r = trace_pass_b_segment_model_pixel(
                pixel, params, radiation_model, radiation_substeps);
            image_i(pixel) = r.observed_stokes.I;
            image_q(pixel) = r.observed_stokes.Q;
            image_u(pixel) = r.observed_stokes.U;
            image_v(pixel) = r.observed_stokes.V;
            closure_x(pixel) = r.closure_x;
            closure_k(pixel) = r.closure_k;
            final_null(pixel) = r.final_null;
            frame_error(pixel) = r.frame_error;
            det_r(pixel) = r.overlap.det();
            steps(pixel) = r.steps;
            reason(pixel) = static_cast<int>(r.reason);
        });
}




template<class ExecSpace, class Real>
void run_pass_a_segment_endpoints(const PassAParams<Real>& params,
                                  Kokkos::View<Real*, ExecSpace> endpoint_x,
                                  Kokkos::View<Real*, ExecSpace> endpoint_k,
                                  Kokkos::View<Real*, ExecSpace> endpoint_e1,
                                  Kokkos::View<Real*, ExecSpace> endpoint_e2,
                                  Kokkos::View<int*, ExecSpace> endpoint_valid,
                                  Kokkos::View<int*, ExecSpace> pass_a_steps,
                                  Kokkos::View<int*, ExecSpace> pass_a_reason) {
    const int npix = params.camera.nx * params.camera.ny;
    Kokkos::parallel_for(
        "KPOLARISPassAEndpoint",
        pixel_range_policy<ExecSpace>(npix),
        KOKKOS_LAMBDA(const int pixel) {
            const PassAEndpointResult<Real> r =
                trace_pass_a_segment_endpoint_pixel(pixel, params);
            const int base = pixel * ndim;
            for (int mu = 0; mu < ndim; ++mu) {
                endpoint_x(base + mu) = r.state.x[mu];
                endpoint_k(base + mu) = r.state.k[mu];
                endpoint_e1(base + mu) = r.state.e1[mu];
                endpoint_e2(base + mu) = r.state.e2[mu];
            }
            endpoint_valid(pixel) = r.valid;
            pass_a_steps(pixel) = r.steps;
            pass_a_reason(pixel) = static_cast<int>(r.reason);
        });
}

template<class ExecSpace, class Real, class RadiationModel>
void run_pass_b_segment_model_from_endpoints_with_overlap(
    const PassAParams<Real>& params,
    const RadiationModel& radiation_model,
    int radiation_substeps,
    Kokkos::View<Real*, ExecSpace> endpoint_x,
    Kokkos::View<Real*, ExecSpace> endpoint_k,
    Kokkos::View<Real*, ExecSpace> endpoint_e1,
    Kokkos::View<Real*, ExecSpace> endpoint_e2,
    Kokkos::View<int*, ExecSpace> endpoint_valid,
    Kokkos::View<int*, ExecSpace> stored_pass_a_steps,
    Kokkos::View<int*, ExecSpace> stored_pass_a_reason,
    Kokkos::View<Real*, ExecSpace> image_i,
    Kokkos::View<Real*, ExecSpace> image_q,
    Kokkos::View<Real*, ExecSpace> image_u,
    Kokkos::View<Real*, ExecSpace> image_v,
    Kokkos::View<Real*, ExecSpace> closure_x,
    Kokkos::View<Real*, ExecSpace> closure_k,
    Kokkos::View<Real*, ExecSpace> final_null,
    Kokkos::View<Real*, ExecSpace> frame_error,
    Kokkos::View<Real*, ExecSpace> det_r,
    Kokkos::View<Real*, ExecSpace> overlap_r11,
    Kokkos::View<Real*, ExecSpace> overlap_r12,
    Kokkos::View<Real*, ExecSpace> overlap_r21,
    Kokkos::View<Real*, ExecSpace> overlap_r22,
    Kokkos::View<Real*, ExecSpace> basis_identity_error,
    Kokkos::View<Real*, ExecSpace> basis_rotation_angle,
    Kokkos::View<int*, ExecSpace> pass_a_steps,
    Kokkos::View<int*, ExecSpace> steps,
    Kokkos::View<int*, ExecSpace> reason) {
    const int npix = params.camera.nx * params.camera.ny;
    Kokkos::parallel_for(
        "KPOLARISPassBFromEndpointOverlap",
        pixel_range_policy<ExecSpace>(npix),
        KOKKOS_LAMBDA(const int pixel) {
            TransportState<Real> endpoint_state;
            const int base = pixel * ndim;
            for (int mu = 0; mu < ndim; ++mu) {
                endpoint_state.x[mu] = endpoint_x(base + mu);
                endpoint_state.k[mu] = endpoint_k(base + mu);
                endpoint_state.e1[mu] = endpoint_e1(base + mu);
                endpoint_state.e2[mu] = endpoint_e2(base + mu);
            }
            const PassBResult<Real> r = trace_pass_b_segment_model_endpoint_pixel(
                pixel, params, radiation_model, radiation_substeps, endpoint_state,
                stored_pass_a_steps(pixel),
                static_cast<TerminationReason>(stored_pass_a_reason(pixel)),
                endpoint_valid(pixel));
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
                max_val(max_val(abs_val(r.overlap.r11 - Real(1)),
                                abs_val(r.overlap.r22 - Real(1))),
                        max_val(abs_val(r.overlap.r12),
                                abs_val(r.overlap.r21)));
            basis_rotation_angle(pixel) =
                Kokkos::atan2(r.overlap.r21 - r.overlap.r12,
                              r.overlap.r11 + r.overlap.r22);
            pass_a_steps(pixel) = r.pass_a.steps;
            steps(pixel) = r.steps;
            reason(pixel) = static_cast<int>(r.reason);
        });
}

template<class ExecSpace, class Real, class RadiationModel>
void run_pass_b_segment_model_with_overlap(const PassAParams<Real>& params,
                                           const RadiationModel& radiation_model,
                                           int radiation_substeps,
                                           Kokkos::View<Real*, ExecSpace> image_i,
                                           Kokkos::View<Real*, ExecSpace> image_q,
                                           Kokkos::View<Real*, ExecSpace> image_u,
                                           Kokkos::View<Real*, ExecSpace> image_v,
                                           Kokkos::View<Real*, ExecSpace> closure_x,
                                           Kokkos::View<Real*, ExecSpace> closure_k,
                                           Kokkos::View<Real*, ExecSpace> final_null,
                                           Kokkos::View<Real*, ExecSpace> frame_error,
                                           Kokkos::View<Real*, ExecSpace> det_r,
                                           Kokkos::View<Real*, ExecSpace> overlap_r11,
                                           Kokkos::View<Real*, ExecSpace> overlap_r12,
                                           Kokkos::View<Real*, ExecSpace> overlap_r21,
                                           Kokkos::View<Real*, ExecSpace> overlap_r22,
                                           Kokkos::View<Real*, ExecSpace> basis_identity_error,
                                           Kokkos::View<Real*, ExecSpace> basis_rotation_angle,
                                           Kokkos::View<int*, ExecSpace> pass_a_steps,
                                           Kokkos::View<int*, ExecSpace> steps,
                                           Kokkos::View<int*, ExecSpace> reason) {
    const int npix = params.camera.nx * params.camera.ny;
    Kokkos::parallel_for(
        "KPOLARISPassBSegmentModelOverlap",
        pixel_range_policy<ExecSpace>(npix),
        KOKKOS_LAMBDA(const int pixel) {
            const PassBResult<Real> r = trace_pass_b_segment_model_pixel(
                pixel, params, radiation_model, radiation_substeps);
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
                max_val(max_val(abs_val(r.overlap.r11 - Real(1)),
                                abs_val(r.overlap.r22 - Real(1))),
                        max_val(abs_val(r.overlap.r12),
                                abs_val(r.overlap.r21)));
            basis_rotation_angle(pixel) =
                Kokkos::atan2(r.overlap.r21 - r.overlap.r12,
                              r.overlap.r11 + r.overlap.r22);
            pass_a_steps(pixel) = r.pass_a.steps;
            steps(pixel) = r.steps;
            reason(pixel) = static_cast<int>(r.reason);
        });
}


#if KPOLARIS_MAX_FREQUENCIES > 1
template<class ExecSpace, class Real, class RadiationModel>
void run_pass_b_segment_model_multifrequency_with_overlap(
    const PassAParams<Real>& params,
    const RadiationModel& radiation_model,
    int radiation_substeps,
    Kokkos::View<Real*, ExecSpace> frequencies,
    Kokkos::View<Real**, ExecSpace> image_i,
    Kokkos::View<Real**, ExecSpace> image_q,
    Kokkos::View<Real**, ExecSpace> image_u,
    Kokkos::View<Real**, ExecSpace> image_v,
    Kokkos::View<Real*, ExecSpace> closure_x,
    Kokkos::View<Real*, ExecSpace> closure_k,
    Kokkos::View<Real*, ExecSpace> final_null,
    Kokkos::View<Real*, ExecSpace> frame_error,
    Kokkos::View<Real*, ExecSpace> det_r,
    Kokkos::View<Real*, ExecSpace> overlap_r11,
    Kokkos::View<Real*, ExecSpace> overlap_r12,
    Kokkos::View<Real*, ExecSpace> overlap_r21,
    Kokkos::View<Real*, ExecSpace> overlap_r22,
    Kokkos::View<Real*, ExecSpace> basis_identity_error,
    Kokkos::View<Real*, ExecSpace> basis_rotation_angle,
    Kokkos::View<int*, ExecSpace> pass_a_steps,
    Kokkos::View<int*, ExecSpace> steps,
    Kokkos::View<int*, ExecSpace> reason) {
    const int npix = params.camera.nx * params.camera.ny;
    const int nfreq = static_cast<int>(frequencies.extent(0));
    Kokkos::parallel_for(
        "KPOLARISPassBSegmentModelMultiFreqOverlap",
        pixel_range_policy<ExecSpace>(npix),
        KOKKOS_LAMBDA(const int pixel) {
            Real local_frequencies[KPOLARIS_MAX_FREQUENCIES];
            Stokes<Real> local_stokes[KPOLARIS_MAX_FREQUENCIES];
            for (int f = 0; f < nfreq; ++f) {
                local_frequencies[f] = frequencies(f);
            }
            const MultiFrequencyPassBResult<Real> r =
                trace_pass_b_segment_model_multifrequency_pixel(
                    pixel, params, radiation_model, radiation_substeps,
                    local_frequencies, nfreq, local_stokes);
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
                max_val(max_val(abs_val(r.overlap.r11 - Real(1)),
                                abs_val(r.overlap.r22 - Real(1))),
                        max_val(abs_val(r.overlap.r12),
                                abs_val(r.overlap.r21)));
            basis_rotation_angle(pixel) =
                Kokkos::atan2(r.overlap.r21 - r.overlap.r12,
                              r.overlap.r11 + r.overlap.r22);
            pass_a_steps(pixel) = r.pass_a.steps;
            steps(pixel) = r.steps;
            reason(pixel) = static_cast<int>(r.reason);
        });
}
#endif

} // namespace kpolaris
