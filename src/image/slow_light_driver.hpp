#pragma once

#include <algorithm>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <Kokkos_Core.hpp>

#include "geodesic/pass_b.hpp"
#include "model/time_interpolation.hpp"
#include "image/driver.hpp"
#include "image/analysis_driver.hpp"
#ifndef KPOLARIS_ENABLE_ANALYSIS_MODE
#define KPOLARIS_ENABLE_ANALYSIS_MODE 0
#endif

namespace kpolaris_image_detail {

template<class Model, class = void>
struct HasFluidTimeInterpolation : std::false_type {};
template<class Model>
struct HasFluidTimeInterpolation<Model, std::void_t<decltype(Model::supports_fluid_time_interpolation)>>
    : std::integral_constant<bool, Model::supports_fluid_time_interpolation> {};

template<class Model>
void validate_slow_light_interpolation(const Model& lower, const Model& upper,
                                      kpolaris::SlowLightInterpolation method) {
    if (method != kpolaris::SlowLightInterpolation::fluid) return;
    if constexpr (HasFluidTimeInterpolation<Model>::value) {
        if (!lower.can_interpolate_fluid_with(upper))
            throw std::invalid_argument("slow_light_interpolation=fluid requires native fluid variables, "
                "a common coordinate mapping/domain, and matching scalar reconstruction; "
                "precomputed four-vector resampling is not supported");
    } else {
        throw std::invalid_argument("this radiation model does not support slow_light_interpolation=fluid");
    }
}

template<class RealT, class RadiationModel>
struct SlowLightTemporalModel {
    RadiationModel lower;
    RadiationModel upper;
    RealT lower_time = RealT(0);
    RealT upper_time = RealT(1);
    RealT observation_time = RealT(0);
    RealT freq_cgs = RealT(0);
    kpolaris::SlowLightInterpolation interpolation = kpolaris::default_slow_light_interpolation;

    KPOLARIS_INLINE void set_frequency(RealT frequency) {
        freq_cgs = lower.freq_cgs = upper.freq_cgs = frequency;
    }

    KPOLARIS_INLINE RealT dlambda_scale() const {
        return lower.dlambda_scale();
    }

    KPOLARIS_INLINE RealT interpolation_fraction(const kpolaris::TransportState<RealT>& state) const {
        const RealT denom = upper_time - lower_time;
        RealT f = RealT(0);
        if (kpolaris::abs_val(denom) > kpolaris::tiny_positive<RealT>()) {
            f = (observation_time + state.x[0] - lower_time) / denom;
        }
        return kpolaris::min_val(RealT(1), kpolaris::max_val(RealT(0), f));
    }

    template<class Metric>
    KPOLARIS_INLINE void bl_coordinates_for_metric(
        const Metric& metric,
        const kpolaris::Vec4<RealT>& x,
        RealT& r,
        RealT& th,
        RealT& cp,
        RealT& sp) const {
        lower.bl_coordinates_for_metric(metric, x, r, th, cp, sp);
    }

    template<class Metric>
    KPOLARIS_INLINE kpolaris::Vec4<RealT> lower_vector(
        const Metric& metric,
        const kpolaris::Vec4<RealT>& x,
        const kpolaris::Vec4<RealT>& v) const {
        return lower.lower_vector(metric, x, v);
    }

    template<class Metric>
    KPOLARIS_INLINE RealT screen_inner_product(
        const Metric& metric,
        const kpolaris::Vec4<RealT>& x,
        const kpolaris::Vec4<RealT>& ucon,
        const kpolaris::Vec4<RealT>& kcon,
        const kpolaris::Vec4<RealT>& econ,
        const kpolaris::Vec4<RealT>& bcon) const {
        return lower.screen_inner_product(metric, x, ucon, kcon, econ, bcon);
    }

    template<class Metric>
    KPOLARIS_INLINE int fluid_state(
        const Metric& metric,
        const kpolaris::TransportState<RealT>& state,
        RealT& rho,
        RealT& uu,
        kpolaris::Vec4<RealT>& ucon,
        kpolaris::Vec4<RealT>& bcon,
        RealT& ne_cgs,
        RealT& thetae,
        RealT& b_cgs,
        RealT& sigma,
        RealT& beta) const {
        if constexpr (HasFluidTimeInterpolation<RadiationModel>::value) {
            if (interpolation == kpolaris::SlowLightInterpolation::fluid)
                return lower.temporal_fluid_state(upper, interpolation_fraction(state), metric, state,
                    rho, uu, ucon, bcon, ne_cgs, thetae, b_cgs, sigma, beta);
        }
        RealT rho0 = RealT(0), uu0 = RealT(0), ne0 = RealT(0), theta0 = RealT(0);
        RealT b0 = RealT(0), sigma0 = RealT(0), beta0 = RealT(0);
        RealT rho1 = RealT(0), uu1 = RealT(0), ne1 = RealT(0), theta1 = RealT(0);
        RealT b1 = RealT(0), sigma1 = RealT(0), beta1 = RealT(0);
        kpolaris::Vec4<RealT> u0, bcon0, u1, bcon1;
        const int ok0 = lower.fluid_state(metric, state, rho0, uu0, u0, bcon0,
                                          ne0, theta0, b0, sigma0, beta0);
        const int ok1 = upper.fluid_state(metric, state, rho1, uu1, u1, bcon1,
                                          ne1, theta1, b1, sigma1, beta1);
        if (!ok0 && !ok1) {
            return 0;
        }
        const RealT f = interpolation_fraction(state);
        const RealT g = RealT(1) - f;
        if (ok0 && ok1) {
            rho = g * rho0 + f * rho1;
            uu = g * uu0 + f * uu1;
            ne_cgs = g * ne0 + f * ne1;
            thetae = g * theta0 + f * theta1;
            b_cgs = g * b0 + f * b1;
            sigma = g * sigma0 + f * sigma1;
            beta = g * beta0 + f * beta1;
            for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                ucon[mu] = g * u0[mu] + f * u1[mu];
                bcon[mu] = g * bcon0[mu] + f * bcon1[mu];
            }
            return 1;
        }
        rho = ok0 ? rho0 : rho1;
        uu = ok0 ? uu0 : uu1;
        ne_cgs = ok0 ? ne0 : ne1;
        thetae = ok0 ? theta0 : theta1;
        b_cgs = ok0 ? b0 : b1;
        sigma = ok0 ? sigma0 : sigma1;
        beta = ok0 ? beta0 : beta1;
        const kpolaris::Vec4<RealT>& usrc = ok0 ? u0 : u1;
        const kpolaris::Vec4<RealT>& bsrc = ok0 ? bcon0 : bcon1;
        for (int mu = 0; mu < kpolaris::ndim; ++mu) {
            ucon[mu] = usrc[mu];
            bcon[mu] = bsrc[mu];
        }
        return 1;
    }

    template<class Metric>
    KPOLARIS_INLINE kpolaris::TransferCoeffs<RealT> coefficients(
        const Metric& metric,
        const kpolaris::TransportState<RealT>& state,
        RealT phase,
        const kpolaris::PlasmaPerturbation<RealT>& perturbation = {}) const {
        const RealT f = interpolation_fraction(state);
        if constexpr (HasFluidTimeInterpolation<RadiationModel>::value) {
            if (interpolation == kpolaris::SlowLightInterpolation::fluid) {
                if (f <= RealT(0)) return lower.coefficients(metric, state, phase, perturbation);
                if (f >= RealT(1)) return upper.coefficients(metric, state, phase, perturbation);
                return lower.coefficients(metric, state, phase, perturbation, &upper, f);
            }
        }
        const auto a = lower.coefficients(metric, state, phase, perturbation);
        const auto b = upper.coefficients(metric, state, phase, perturbation);
        const RealT g = RealT(1) - f;
        kpolaris::TransferCoeffs<RealT> out;
        out.jI = g * a.jI + f * b.jI;
        out.jQ = g * a.jQ + f * b.jQ;
        out.jU = g * a.jU + f * b.jU;
        out.jV = g * a.jV + f * b.jV;
        out.aI = g * a.aI + f * b.aI;
        out.aQ = g * a.aQ + f * b.aQ;
        out.aU = g * a.aU + f * b.aU;
        out.aV = g * a.aV + f * b.aV;
        out.rQ = g * a.rQ + f * b.rQ;
        out.rU = g * a.rU + f * b.rU;
        out.rV = g * a.rV + f * b.rV;
        return out;
    }
};

template<class ExecSpace, class RealT>
struct SlowLightViews {
    Kokkos::View<RealT*, ExecSpace> state_x;
    Kokkos::View<RealT*, ExecSpace> state_k;
    Kokkos::View<RealT*, ExecSpace> state_e1;
    Kokkos::View<RealT*, ExecSpace> state_e2;
    Kokkos::View<RealT*, ExecSpace> stokes_i;
    Kokkos::View<RealT*, ExecSpace> stokes_q;
    Kokkos::View<RealT*, ExecSpace> stokes_u;
    Kokkos::View<RealT*, ExecSpace> stokes_v;
    Kokkos::View<RealT*, ExecSpace> h_current;
    Kokkos::View<RealT*, ExecSpace> requested_time;
    Kokkos::View<int*, ExecSpace> waiting_for_data;
    Kokkos::View<RealT*, ExecSpace> radiation_step_cap;
    Kokkos::View<RealT*, ExecSpace> image_i;
    Kokkos::View<RealT*, ExecSpace> image_q;
    Kokkos::View<RealT*, ExecSpace> image_u;
    Kokkos::View<RealT*, ExecSpace> image_v;
    Kokkos::View<RealT*, ExecSpace> closure_x;
    Kokkos::View<RealT*, ExecSpace> closure_k;
    Kokkos::View<RealT*, ExecSpace> final_null;
    Kokkos::View<RealT*, ExecSpace> frame_error;
    Kokkos::View<RealT*, ExecSpace> det_r;
    Kokkos::View<RealT*, ExecSpace> overlap_r11;
    Kokkos::View<RealT*, ExecSpace> overlap_r12;
    Kokkos::View<RealT*, ExecSpace> overlap_r21;
    Kokkos::View<RealT*, ExecSpace> overlap_r22;
    Kokkos::View<RealT*, ExecSpace> basis_identity_error;
    Kokkos::View<RealT*, ExecSpace> basis_rotation_angle;
#if KPOLARIS_ENABLE_ANALYSIS_MODE
    Kokkos::View<RealT*, ExecSpace> response_data;
    Kokkos::View<RealT*, ExecSpace> radiating_path_length;
    Kokkos::View<RealT*, ExecSpace> emission_weight;
    Kokkos::View<RealT*, ExecSpace> emission_weighted_radius;
    Kokkos::View<RealT*, ExecSpace> emission_weighted_optical_depth_to_camera;
    Kokkos::View<RealT*, ExecSpace> absorption_depth;
    Kokkos::View<RealT*, ExecSpace> absorption_operator_depth;
    Kokkos::View<RealT*, ExecSpace> faraday_rotation_depth;
    Kokkos::View<RealT*, ExecSpace> faraday_conversion_depth;
    Kokkos::View<RealT*, ExecSpace> faraday_operator_depth;
    Kokkos::View<RealT*, ExecSpace> dominant_emission_radius;
    Kokkos::View<RealT*, ExecSpace> dominant_emission_weight;
    Kokkos::View<RealT*, ExecSpace> dominant_ne_cgs;
    Kokkos::View<RealT*, ExecSpace> dominant_thetae;
    Kokkos::View<RealT*, ExecSpace> dominant_b_cgs;
    Kokkos::View<RealT*, ExecSpace> dominant_beta;
    Kokkos::View<RealT*, ExecSpace> dominant_sigma;
    Kokkos::View<RealT*, ExecSpace> emission_weighted_ne_cgs;
    Kokkos::View<RealT*, ExecSpace> emission_weighted_thetae;
    Kokkos::View<RealT*, ExecSpace> emission_weighted_b_cgs;
    Kokkos::View<RealT*, ExecSpace> emission_weighted_beta;
    Kokkos::View<RealT*, ExecSpace> emission_weighted_sigma;
    Kokkos::View<RealT*, ExecSpace> photon_ring_winding_estimate;
    Kokkos::View<RealT*, ExecSpace> previous_phi;
    Kokkos::View<int*, ExecSpace> has_previous_phi;
    Kokkos::View<int*, ExecSpace> radiation_substeps;
    Kokkos::View<int*, ExecSpace> dominant_emission_region;
    Kokkos::View<RealT*, ExecSpace> radial_stokes_i;
    Kokkos::View<RealT*, ExecSpace> radial_stokes_q;
    Kokkos::View<RealT*, ExecSpace> radial_stokes_u;
    Kokkos::View<RealT*, ExecSpace> radial_stokes_v;
    Kokkos::View<RealT*, ExecSpace> radial_absorption_depth;
    Kokkos::View<RealT*, ExecSpace> radial_faraday_rotation_depth;
    Kokkos::View<RealT*, ExecSpace> radial_faraday_conversion_depth;
    Kokkos::View<RealT*, ExecSpace> radial_faraday_operator_depth;
#endif
    Kokkos::View<int*, ExecSpace> active;
    Kokkos::View<int*, ExecSpace> pass_a_steps;
    Kokkos::View<int*, ExecSpace> steps;
    Kokkos::View<int*, ExecSpace> reason;
};

template<class ExecSpace, class RealT>
SlowLightViews<ExecSpace, RealT> allocate_slow_light_views(
    int npix,
    int analysis_mode = 0,
    const kpolaris::AnalysisConfig<RealT>& analysis_config = {}) {
    SlowLightViews<ExecSpace, RealT> v;
    v.state_x = Kokkos::View<RealT*, ExecSpace>("slow_state_x", npix * kpolaris::ndim);
    v.state_k = Kokkos::View<RealT*, ExecSpace>("slow_state_k", npix * kpolaris::ndim);
    v.state_e1 = Kokkos::View<RealT*, ExecSpace>("slow_state_e1", npix * kpolaris::ndim);
    v.state_e2 = Kokkos::View<RealT*, ExecSpace>("slow_state_e2", npix * kpolaris::ndim);
    v.stokes_i = Kokkos::View<RealT*, ExecSpace>("slow_stokes_i", npix);
    v.stokes_q = Kokkos::View<RealT*, ExecSpace>("slow_stokes_q", npix);
    v.stokes_u = Kokkos::View<RealT*, ExecSpace>("slow_stokes_u", npix);
    v.stokes_v = Kokkos::View<RealT*, ExecSpace>("slow_stokes_v", npix);
    v.h_current = Kokkos::View<RealT*, ExecSpace>("slow_h_current", npix);
    v.requested_time = Kokkos::View<RealT*, ExecSpace>("slow_requested_time", npix);
    v.waiting_for_data = Kokkos::View<int*, ExecSpace>("slow_waiting_for_data", npix);
    v.radiation_step_cap = Kokkos::View<RealT*, ExecSpace>("slow_radiation_step_cap", npix);
    v.image_i = Kokkos::View<RealT*, ExecSpace>("slow_I", npix);
    v.image_q = Kokkos::View<RealT*, ExecSpace>("slow_Q", npix);
    v.image_u = Kokkos::View<RealT*, ExecSpace>("slow_U", npix);
    v.image_v = Kokkos::View<RealT*, ExecSpace>("slow_V", npix);
    v.closure_x = Kokkos::View<RealT*, ExecSpace>("slow_closure_x", npix);
    v.closure_k = Kokkos::View<RealT*, ExecSpace>("slow_closure_k", npix);
    v.final_null = Kokkos::View<RealT*, ExecSpace>("slow_final_null", npix);
    v.frame_error = Kokkos::View<RealT*, ExecSpace>("slow_frame_error", npix);
    v.det_r = Kokkos::View<RealT*, ExecSpace>("slow_det_r", npix);
    v.overlap_r11 = Kokkos::View<RealT*, ExecSpace>("slow_overlap_r11", npix);
    v.overlap_r12 = Kokkos::View<RealT*, ExecSpace>("slow_overlap_r12", npix);
    v.overlap_r21 = Kokkos::View<RealT*, ExecSpace>("slow_overlap_r21", npix);
    v.overlap_r22 = Kokkos::View<RealT*, ExecSpace>("slow_overlap_r22", npix);
    v.basis_identity_error = Kokkos::View<RealT*, ExecSpace>("slow_basis_identity_error", npix);
    v.basis_rotation_angle = Kokkos::View<RealT*, ExecSpace>("slow_basis_rotation_angle", npix);
#if KPOLARIS_ENABLE_ANALYSIS_MODE
    if (analysis_mode) {
        v.radiating_path_length = Kokkos::View<RealT*, ExecSpace>("slow_analysis_radiating_path_length", npix);
        v.emission_weight = Kokkos::View<RealT*, ExecSpace>("slow_analysis_emission_weight", npix);
        v.emission_weighted_radius = Kokkos::View<RealT*, ExecSpace>("slow_analysis_emission_weighted_radius", npix);
        v.emission_weighted_optical_depth_to_camera = Kokkos::View<RealT*, ExecSpace>("slow_analysis_emission_weighted_tau", npix);
        v.absorption_depth = Kokkos::View<RealT*, ExecSpace>("slow_analysis_absorption_depth", npix);
        v.absorption_operator_depth = Kokkos::View<RealT*, ExecSpace>("slow_analysis_absorption_operator_depth", npix);
        v.faraday_rotation_depth = Kokkos::View<RealT*, ExecSpace>("slow_analysis_faraday_rotation_depth", npix);
        v.faraday_conversion_depth = Kokkos::View<RealT*, ExecSpace>("slow_analysis_faraday_conversion_depth", npix);
        v.faraday_operator_depth = Kokkos::View<RealT*, ExecSpace>("slow_analysis_faraday_operator_depth", npix);
        v.dominant_emission_radius = Kokkos::View<RealT*, ExecSpace>("slow_analysis_dominant_emission_radius", npix);
        v.dominant_emission_weight = Kokkos::View<RealT*, ExecSpace>("slow_analysis_dominant_emission_weight", npix);
        v.dominant_ne_cgs = Kokkos::View<RealT*, ExecSpace>("slow_analysis_dominant_ne_cgs", npix);
        v.dominant_thetae = Kokkos::View<RealT*, ExecSpace>("slow_analysis_dominant_thetae", npix);
        v.dominant_b_cgs = Kokkos::View<RealT*, ExecSpace>("slow_analysis_dominant_b_cgs", npix);
        v.dominant_beta = Kokkos::View<RealT*, ExecSpace>("slow_analysis_dominant_beta", npix);
        v.dominant_sigma = Kokkos::View<RealT*, ExecSpace>("slow_analysis_dominant_sigma", npix);
        v.emission_weighted_ne_cgs = Kokkos::View<RealT*, ExecSpace>("slow_analysis_emission_weighted_ne_cgs", npix);
        v.emission_weighted_thetae = Kokkos::View<RealT*, ExecSpace>("slow_analysis_emission_weighted_thetae", npix);
        v.emission_weighted_b_cgs = Kokkos::View<RealT*, ExecSpace>("slow_analysis_emission_weighted_b_cgs", npix);
        v.emission_weighted_beta = Kokkos::View<RealT*, ExecSpace>("slow_analysis_emission_weighted_beta", npix);
        v.emission_weighted_sigma = Kokkos::View<RealT*, ExecSpace>("slow_analysis_emission_weighted_sigma", npix);
        v.photon_ring_winding_estimate = Kokkos::View<RealT*, ExecSpace>("slow_analysis_photon_ring_winding", npix);
        v.previous_phi = Kokkos::View<RealT*, ExecSpace>("slow_analysis_previous_phi", npix);
        v.has_previous_phi = Kokkos::View<int*, ExecSpace>("slow_analysis_has_previous_phi", npix);
        v.radiation_substeps = Kokkos::View<int*, ExecSpace>("slow_analysis_radiation_substeps", npix);
        v.dominant_emission_region = Kokkos::View<int*, ExecSpace>("slow_analysis_dominant_region", npix);
        if (analysis_config.response.enabled())
            v.response_data = Kokkos::View<RealT*, ExecSpace>("slow_analysis_response", size_t(4) * analysis_config.response.channels() * npix);
        const int radial_values = npix * analysis_config.radial_bins;
        v.radial_stokes_i = Kokkos::View<RealT*, ExecSpace>("slow_analysis_radial_stokes_i", radial_values);
        v.radial_stokes_q = Kokkos::View<RealT*, ExecSpace>("slow_analysis_radial_stokes_q", radial_values);
        v.radial_stokes_u = Kokkos::View<RealT*, ExecSpace>("slow_analysis_radial_stokes_u", radial_values);
        v.radial_stokes_v = Kokkos::View<RealT*, ExecSpace>("slow_analysis_radial_stokes_v", radial_values);
        v.radial_absorption_depth = Kokkos::View<RealT*, ExecSpace>("slow_analysis_radial_absorption_depth", radial_values);
        v.radial_faraday_rotation_depth = Kokkos::View<RealT*, ExecSpace>("slow_analysis_radial_faraday_rotation_depth", radial_values);
        v.radial_faraday_conversion_depth = Kokkos::View<RealT*, ExecSpace>("slow_analysis_radial_faraday_conversion_depth", radial_values);
        v.radial_faraday_operator_depth = Kokkos::View<RealT*, ExecSpace>("slow_analysis_radial_faraday_operator_depth", radial_values);
    }
#else
    (void)analysis_mode;
    (void)analysis_config;
#endif
    v.active = Kokkos::View<int*, ExecSpace>("slow_active", npix);
    v.pass_a_steps = Kokkos::View<int*, ExecSpace>("slow_pass_a_steps", npix);
    v.steps = Kokkos::View<int*, ExecSpace>("slow_steps", npix);
    v.reason = Kokkos::View<int*, ExecSpace>("slow_reason", npix);
    return v;
}

template<class ExecSpace, class RealT, class Metric>
void initialize_slow_light_states_metric(
    const kpolaris::PassAParams<RealT>& params,
    const Metric& metric,
    SlowLightViews<ExecSpace, RealT>& v,
    int timing,
    int analysis_mode = 0,
    const kpolaris::AnalysisConfig<RealT>& analysis_config = {}) {
    const int npix = params.camera.nx * params.camera.ny;
    Kokkos::Timer timer;
    Kokkos::parallel_for(
        "KPOLARISSlowLightInit",
        Kokkos::RangePolicy<ExecSpace>(0, npix),
        KOKKOS_LAMBDA(const int pixel) {
            const auto endpoint =
                kpolaris::trace_pass_a_segment_endpoint_pixel_metric(pixel, params, metric);
            const auto camera_state = kpolaris::initialize_camera_ray(metric, pixel, params.camera);
            kpolaris::TransportState<RealT> state = endpoint.state;
            const int base = pixel * kpolaris::ndim;
            for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                v.state_x(base + mu) = state.x[mu];
                v.state_k(base + mu) = state.k[mu];
                v.state_e1(base + mu) = state.e1[mu];
                v.state_e2(base + mu) = state.e2[mu];
            }
            v.stokes_i(pixel) = RealT(0);
            v.stokes_q(pixel) = RealT(0);
            v.stokes_u(pixel) = RealT(0);
            v.stokes_v(pixel) = RealT(0);
            v.h_current(pixel) = params.step;
            v.waiting_for_data(pixel) = 0;
            v.radiation_step_cap(pixel) = params.max_radiation_step > RealT(0) ?
                params.max_radiation_step : params.max_step;
            v.active(pixel) = endpoint.valid ? 1 : 0;
            v.pass_a_steps(pixel) = endpoint.steps;
            v.steps(pixel) = 0;
            v.reason(pixel) = static_cast<int>(endpoint.reason);
            v.image_i(pixel) = RealT(0);
            v.image_q(pixel) = RealT(0);
            v.image_u(pixel) = RealT(0);
            v.image_v(pixel) = RealT(0);
            v.det_r(pixel) = RealT(0);
            v.overlap_r11(pixel) = RealT(0);
            v.overlap_r12(pixel) = RealT(0);
            v.overlap_r21(pixel) = RealT(0);
            v.overlap_r22(pixel) = RealT(0);
            v.basis_identity_error(pixel) = RealT(0);
            v.basis_rotation_angle(pixel) = RealT(0);
#if KPOLARIS_ENABLE_ANALYSIS_MODE
            if (analysis_mode) {
                v.radiating_path_length(pixel) = RealT(0);
                v.emission_weight(pixel) = RealT(0);
                v.emission_weighted_radius(pixel) = RealT(0);
                v.emission_weighted_optical_depth_to_camera(pixel) = RealT(0);
                v.absorption_depth(pixel) = RealT(0);
                v.absorption_operator_depth(pixel) = RealT(0);
                v.faraday_rotation_depth(pixel) = RealT(0);
                v.faraday_conversion_depth(pixel) = RealT(0);
                v.faraday_operator_depth(pixel) = RealT(0);
                v.dominant_emission_radius(pixel) = RealT(0);
                v.dominant_emission_weight(pixel) = RealT(0);
                v.dominant_ne_cgs(pixel) = RealT(0);
                v.dominant_thetae(pixel) = RealT(0);
                v.dominant_b_cgs(pixel) = RealT(0);
                v.dominant_beta(pixel) = RealT(0);
                v.dominant_sigma(pixel) = RealT(0);
                v.emission_weighted_ne_cgs(pixel) = RealT(0);
                v.emission_weighted_thetae(pixel) = RealT(0);
                v.emission_weighted_b_cgs(pixel) = RealT(0);
                v.emission_weighted_beta(pixel) = RealT(0);
                v.emission_weighted_sigma(pixel) = RealT(0);
                v.photon_ring_winding_estimate(pixel) = RealT(0);
                v.previous_phi(pixel) = RealT(0);
                v.has_previous_phi(pixel) = 0;
                v.radiation_substeps(pixel) = 0;
                v.dominant_emission_region(pixel) = 0;
                for (int ch=0; ch<analysis_config.response.channels(); ++ch)
                    kpolaris::response_write(v.response_data, pixel, npix, ch, kpolaris::Stokes<RealT>{});
                for (int bin = 0; bin < analysis_config.radial_bins; ++bin) {
                    const int index = bin * npix + pixel;
                    v.radial_stokes_i(index) = RealT(0);
                    v.radial_stokes_q(index) = RealT(0);
                    v.radial_stokes_u(index) = RealT(0);
                    v.radial_stokes_v(index) = RealT(0);
                    v.radial_absorption_depth(index) = RealT(0);
                    v.radial_faraday_rotation_depth(index) = RealT(0);
                    v.radial_faraday_conversion_depth(index) = RealT(0);
                    v.radial_faraday_operator_depth(index) = RealT(0);
                }
            }
#else
            (void)analysis_mode;
            (void)analysis_config;
#endif
            if (!endpoint.valid) {
                v.closure_x(pixel) = kpolaris::spatial_distance(state.x, camera_state.x);
                v.closure_k(pixel) = kpolaris::vector_max_abs_difference(state.k, camera_state.k);
                v.final_null(pixel) = metric.dot(state.x, state.k, state.k);
                v.frame_error(pixel) = kpolaris::max_frame_error(kpolaris::frame_errors(metric, state));
            } else {
                v.closure_x(pixel) = RealT(0);
                v.closure_k(pixel) = RealT(0);
                v.final_null(pixel) = RealT(0);
                v.frame_error(pixel) = RealT(0);
            }
        });
    Kokkos::fence();
    report_image_timing(timing, "slow_light_pass_a_endpoint", timer.seconds());
}

template<class ExecSpace, class RealT>
void copy_slow_light_views(
    const SlowLightViews<ExecSpace, RealT>& src,
    SlowLightViews<ExecSpace, RealT>& dst,
    int timing,
    int analysis_mode = 0) {
    Kokkos::Timer timer;
    Kokkos::deep_copy(dst.state_x, src.state_x);
    Kokkos::deep_copy(dst.state_k, src.state_k);
    Kokkos::deep_copy(dst.state_e1, src.state_e1);
    Kokkos::deep_copy(dst.state_e2, src.state_e2);
    Kokkos::deep_copy(dst.stokes_i, src.stokes_i);
    Kokkos::deep_copy(dst.stokes_q, src.stokes_q);
    Kokkos::deep_copy(dst.stokes_u, src.stokes_u);
    Kokkos::deep_copy(dst.stokes_v, src.stokes_v);
    Kokkos::deep_copy(dst.h_current, src.h_current);
    Kokkos::deep_copy(dst.requested_time, src.requested_time);
    Kokkos::deep_copy(dst.waiting_for_data, src.waiting_for_data);
    Kokkos::deep_copy(dst.radiation_step_cap, src.radiation_step_cap);
    Kokkos::deep_copy(dst.image_i, src.image_i);
    Kokkos::deep_copy(dst.image_q, src.image_q);
    Kokkos::deep_copy(dst.image_u, src.image_u);
    Kokkos::deep_copy(dst.image_v, src.image_v);
    Kokkos::deep_copy(dst.closure_x, src.closure_x);
    Kokkos::deep_copy(dst.closure_k, src.closure_k);
    Kokkos::deep_copy(dst.final_null, src.final_null);
    Kokkos::deep_copy(dst.frame_error, src.frame_error);
    Kokkos::deep_copy(dst.det_r, src.det_r);
    Kokkos::deep_copy(dst.overlap_r11, src.overlap_r11);
    Kokkos::deep_copy(dst.overlap_r12, src.overlap_r12);
    Kokkos::deep_copy(dst.overlap_r21, src.overlap_r21);
    Kokkos::deep_copy(dst.overlap_r22, src.overlap_r22);
    Kokkos::deep_copy(dst.basis_identity_error, src.basis_identity_error);
    Kokkos::deep_copy(dst.basis_rotation_angle, src.basis_rotation_angle);
#if KPOLARIS_ENABLE_ANALYSIS_MODE
    if (analysis_mode) {
        Kokkos::deep_copy(dst.radiating_path_length, src.radiating_path_length);
        Kokkos::deep_copy(dst.emission_weight, src.emission_weight);
        Kokkos::deep_copy(dst.emission_weighted_radius, src.emission_weighted_radius);
        Kokkos::deep_copy(dst.emission_weighted_optical_depth_to_camera, src.emission_weighted_optical_depth_to_camera);
        Kokkos::deep_copy(dst.absorption_depth, src.absorption_depth);
        Kokkos::deep_copy(dst.absorption_operator_depth, src.absorption_operator_depth);
        Kokkos::deep_copy(dst.faraday_rotation_depth, src.faraday_rotation_depth);
        Kokkos::deep_copy(dst.faraday_conversion_depth, src.faraday_conversion_depth);
        Kokkos::deep_copy(dst.faraday_operator_depth, src.faraday_operator_depth);
        Kokkos::deep_copy(dst.dominant_emission_radius, src.dominant_emission_radius);
        Kokkos::deep_copy(dst.dominant_emission_weight, src.dominant_emission_weight);
        Kokkos::deep_copy(dst.dominant_ne_cgs, src.dominant_ne_cgs);
        Kokkos::deep_copy(dst.dominant_thetae, src.dominant_thetae);
        Kokkos::deep_copy(dst.dominant_b_cgs, src.dominant_b_cgs);
        Kokkos::deep_copy(dst.dominant_beta, src.dominant_beta);
        Kokkos::deep_copy(dst.dominant_sigma, src.dominant_sigma);
        Kokkos::deep_copy(dst.emission_weighted_ne_cgs, src.emission_weighted_ne_cgs);
        Kokkos::deep_copy(dst.emission_weighted_thetae, src.emission_weighted_thetae);
        Kokkos::deep_copy(dst.emission_weighted_b_cgs, src.emission_weighted_b_cgs);
        Kokkos::deep_copy(dst.emission_weighted_beta, src.emission_weighted_beta);
        Kokkos::deep_copy(dst.emission_weighted_sigma, src.emission_weighted_sigma);
        Kokkos::deep_copy(dst.photon_ring_winding_estimate, src.photon_ring_winding_estimate);
        Kokkos::deep_copy(dst.previous_phi, src.previous_phi);
        Kokkos::deep_copy(dst.has_previous_phi, src.has_previous_phi);
        Kokkos::deep_copy(dst.radiation_substeps, src.radiation_substeps);
        Kokkos::deep_copy(dst.dominant_emission_region, src.dominant_emission_region);
        if (src.response_data.extent(0)) Kokkos::deep_copy(dst.response_data, src.response_data);
        Kokkos::deep_copy(dst.radial_stokes_i, src.radial_stokes_i);
        Kokkos::deep_copy(dst.radial_stokes_q, src.radial_stokes_q);
        Kokkos::deep_copy(dst.radial_stokes_u, src.radial_stokes_u);
        Kokkos::deep_copy(dst.radial_stokes_v, src.radial_stokes_v);
        Kokkos::deep_copy(dst.radial_absorption_depth, src.radial_absorption_depth);
        Kokkos::deep_copy(dst.radial_faraday_rotation_depth, src.radial_faraday_rotation_depth);
        Kokkos::deep_copy(dst.radial_faraday_conversion_depth, src.radial_faraday_conversion_depth);
        Kokkos::deep_copy(dst.radial_faraday_operator_depth, src.radial_faraday_operator_depth);
    }
#else
    (void)analysis_mode;
#endif
    Kokkos::deep_copy(dst.active, src.active);
    Kokkos::deep_copy(dst.pass_a_steps, src.pass_a_steps);
    Kokkos::deep_copy(dst.steps, src.steps);
    Kokkos::deep_copy(dst.reason, src.reason);
    Kokkos::fence();
    report_image_timing(timing, "slow_light_pass_a_clone", timer.seconds());
}

template<class ExecSpace, class RealT>
int count_active_rays(const Kokkos::View<int*, ExecSpace>& active) {
    int count = 0;
    Kokkos::parallel_reduce(
        "KPOLARISSlowLightActiveCount",
        Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(active.extent(0))),
        KOKKOS_LAMBDA(const int i, int& local) {
            local += active(i) ? 1 : 0;
        }, count);
    Kokkos::fence();
    return count;
}

template<class LoadModel, class Model, class = void>
struct SlowLightLoaderTraits {
    using PrefetchResult = Model;

    static Model load(LoadModel& load_model, const std::string& path) {
        return load_model(path);
    }

    static PrefetchResult prefetch(LoadModel load_model, const std::string& path) {
        return load_model(path);
    }

    static Model materialize(LoadModel&, PrefetchResult&& prefetched) {
        return std::move(prefetched);
    }
};

template<class LoadModel, class Model>
struct SlowLightLoaderTraits<LoadModel, Model,
                             std::void_t<typename LoadModel::slow_light_prefetch_result_type> > {
    using PrefetchResult = typename LoadModel::slow_light_prefetch_result_type;

    static Model load(LoadModel& load_model, const std::string& path) {
        return load_model.load(path);
    }

    static PrefetchResult prefetch(LoadModel load_model, const std::string& path) {
        return load_model.prefetch(path);
    }

    static Model materialize(LoadModel& load_model, PrefetchResult&& prefetched) {
        return load_model.materialize(std::move(prefetched));
    }
};

template<class LoadTraits, class LoadModel>
auto launch_slow_light_prefetch(LoadModel load_model, std::string path)
    -> std::future<typename LoadTraits::PrefetchResult> {
    return std::async(std::launch::async,
                      [load_model, path = std::move(path)]() mutable {
                          return LoadTraits::prefetch(load_model, path);
                      });
}

inline void report_slow_light_prefetch_dump(int timing, size_t dump_index) {
    if (timing) {
        std::cout << "timing slow_light_prefetch_dump_index " << dump_index << "\n";
    }
}

inline void report_slow_light_prefetch_wait(int timing, const char* label, double seconds) {
    if (timing) {
        std::cout << "timing " << label << ' ' << seconds << " s\n";
    }
}

struct SlowLightTimeProbeHostResult {
    Real min_x0 = Real(0);
    Real max_x0 = Real(0);
    Real min_fluid_time = Real(0);
    Real max_fluid_time = Real(0);
    Real min_camera_time = Real(0);
    Real max_camera_time = Real(0);
    long long radiating_samples = 0;
    long long pass_a_steps = 0;
    long long pass_b_steps = 0;
    int pixels = 0;
    int valid_endpoint_pixels = 0;
    int returned_pixels = 0;
    int radiating_pixels = 0;
    int camera_time_pixels = 0;
    int max_pass_a_steps = 0;
    int max_pass_b_steps = 0;
};

template<class ExecSpace, class RealT, class Metric>
SlowLightTimeProbeHostResult run_slow_light_time_probe_metric(
    const kpolaris::PassAParams<RealT>& params,
    RealT observation_time,
    const Metric& metric,
    int timing) {
    const int npix = params.camera.nx * params.camera.ny;
    const RealT huge = std::numeric_limits<RealT>::max();
    Kokkos::View<RealT*, ExecSpace> min_x0("slow_probe_min_x0", npix);
    Kokkos::View<RealT*, ExecSpace> max_x0("slow_probe_max_x0", npix);
    Kokkos::View<RealT*, ExecSpace> min_time("slow_probe_min_time", npix);
    Kokkos::View<RealT*, ExecSpace> max_time("slow_probe_max_time", npix);
    Kokkos::View<RealT*, ExecSpace> camera_time("slow_probe_camera_time", npix);
    Kokkos::View<int*, ExecSpace> sample_count("slow_probe_sample_count", npix);
    Kokkos::View<int*, ExecSpace> endpoint_valid("slow_probe_endpoint_valid", npix);
    Kokkos::View<int*, ExecSpace> returned("slow_probe_returned", npix);
    Kokkos::View<int*, ExecSpace> pass_a_steps("slow_probe_pass_a_steps", npix);
    Kokkos::View<int*, ExecSpace> pass_b_steps("slow_probe_pass_b_steps", npix);

    Kokkos::Timer timer;
    Kokkos::parallel_for(
        "KPOLARISSlowLightTimeProbe",
        Kokkos::RangePolicy<ExecSpace>(0, npix),
        KOKKOS_LAMBDA(const int pixel) {
            min_x0(pixel) = huge;
            max_x0(pixel) = -huge;
            min_time(pixel) = huge;
            max_time(pixel) = -huge;
            camera_time(pixel) = -huge;
            sample_count(pixel) = 0;
            endpoint_valid(pixel) = 0;
            returned(pixel) = 0;
            pass_a_steps(pixel) = 0;
            pass_b_steps(pixel) = 0;

            const auto endpoint =
                kpolaris::trace_pass_a_segment_endpoint_pixel_metric(pixel, params, metric);
            pass_a_steps(pixel) = endpoint.steps;
            if (!endpoint.valid) {
                return;
            }
            endpoint_valid(pixel) = 1;

            kpolaris::TransportState<RealT> state = endpoint.state;

            kpolaris::AdaptiveRK4Control<RealT> transfer_control;
            transfer_control.tolerance = params.adaptive_tolerance;
            transfer_control.min_step = params.min_step;
            transfer_control.max_step = params.max_step;
            RealT h_current = params.step;
            int local_steps = 0;
            int reached_camera = 0;
            while (!reached_camera && local_steps < params.max_steps) {
                RealT h = h_current;
                const kpolaris::TransportState<RealT> old_state = state;
                kpolaris::TransportState<RealT> next_state;
                kpolaris::TransportState<RealT> sample_state;
                if (params.adaptive) {
                    const auto proposed = kpolaris::adaptive_rk4_step(metric, old_state, h, transfer_control);
                    if (!proposed.accepted) {
                        h_current = kpolaris::abs_val(proposed.next_h);
                        if (h_current <= params.min_step * RealT(1.0001)) {
                            break;
                        }
                        continue;
                    }
                    h = kpolaris::abs_val(proposed.used_h);
                    h_current = kpolaris::abs_val(proposed.next_h);
                    next_state = proposed.state;
                    sample_state = proposed.mid_state;
                } else {
                    next_state = kpolaris::rk4_step(metric, old_state, h);
                    sample_state = next_state;
                }

                const RealT s0 = kpolaris::camera_surface_value(metric, params.camera, old_state);
                const RealT s1 = kpolaris::camera_surface_value(metric, params.camera, next_state);
                const int crosses_camera = kpolaris::camera_surface_crossed(s0, s1);
                if (crosses_camera) {
                    const RealT frac = kpolaris::camera_surface_crossing_fraction(s0, s1);
                    const RealT h_cross = h * frac;
                    if (kpolaris::abs_val(h_cross) <= RealT(1e-14)) {
                        h = RealT(0);
                        next_state = old_state;
                        sample_state = old_state;
                    } else if (frac < RealT(0.999999999999)) {
                        h = h_cross;
                        if (params.adaptive) {
                            kpolaris::AdaptiveRK4Control<RealT> crossing_control = transfer_control;
                            crossing_control.min_step = kpolaris::min_val(crossing_control.min_step, kpolaris::abs_val(h));
                            crossing_control.max_step = kpolaris::max_val(crossing_control.min_step, kpolaris::abs_val(h));
                            const auto crossing_step = kpolaris::adaptive_rk4_step(metric, old_state, h, crossing_control);
                            if (!crossing_step.accepted) {
                                h_current = kpolaris::abs_val(crossing_step.next_h);
                                continue;
                            }
                            h = kpolaris::abs_val(crossing_step.used_h);
                            next_state = crossing_step.state;
                            sample_state = crossing_step.mid_state;
                        } else {
                            next_state = kpolaris::rk4_step(metric, old_state, h);
                            sample_state = next_state;
                        }
                    }
                }

                const RealT sample_r = kpolaris::radial_coordinate(metric, sample_state.x);
                if (h > RealT(0) && sample_r >= params.inner_radius && sample_r < params.outer_radius) {
                    const RealT x0 = sample_state.x[0];
                    const RealT t = observation_time + x0;
                    min_x0(pixel) = kpolaris::min_val(min_x0(pixel), x0);
                    max_x0(pixel) = kpolaris::max_val(max_x0(pixel), x0);
                    min_time(pixel) = kpolaris::min_val(min_time(pixel), t);
                    max_time(pixel) = kpolaris::max_val(max_time(pixel), t);
                    sample_count(pixel) += 1;
                }

                state = next_state;
                local_steps += 1;
                if (crosses_camera) {
                    camera_time(pixel) = observation_time + next_state.x[0];
                    reached_camera = 1;
                }
            }
            returned(pixel) = reached_camera;
            pass_b_steps(pixel) = local_steps;
        });
    Kokkos::fence();
    report_image_timing(timing, "slow_light_time_probe_kernel", timer.seconds());

    const std::vector<Real> h_min_x0 = copy_real_view_1d(min_x0);
    const std::vector<Real> h_max_x0 = copy_real_view_1d(max_x0);
    const std::vector<Real> h_min_time = copy_real_view_1d(min_time);
    const std::vector<Real> h_max_time = copy_real_view_1d(max_time);
    const std::vector<Real> h_camera_time = copy_real_view_1d(camera_time);
    const std::vector<int> h_sample_count = copy_int_view_1d(sample_count);
    const std::vector<int> h_endpoint_valid = copy_int_view_1d(endpoint_valid);
    const std::vector<int> h_returned = copy_int_view_1d(returned);
    const std::vector<int> h_pass_a_steps = copy_int_view_1d(pass_a_steps);
    const std::vector<int> h_pass_b_steps = copy_int_view_1d(pass_b_steps);

    SlowLightTimeProbeHostResult out;
    out.pixels = npix;
    out.min_x0 = huge;
    out.max_x0 = -huge;
    out.min_fluid_time = huge;
    out.max_fluid_time = -huge;
    out.min_camera_time = huge;
    out.max_camera_time = -huge;
    for (int p = 0; p < npix; ++p) {
        out.pass_a_steps += h_pass_a_steps[static_cast<size_t>(p)];
        out.pass_b_steps += h_pass_b_steps[static_cast<size_t>(p)];
        out.max_pass_a_steps = std::max(out.max_pass_a_steps, h_pass_a_steps[static_cast<size_t>(p)]);
        out.max_pass_b_steps = std::max(out.max_pass_b_steps, h_pass_b_steps[static_cast<size_t>(p)]);
        out.valid_endpoint_pixels += h_endpoint_valid[static_cast<size_t>(p)] ? 1 : 0;
        out.returned_pixels += h_returned[static_cast<size_t>(p)] ? 1 : 0;
        const int samples = h_sample_count[static_cast<size_t>(p)];
        if (samples > 0) {
            out.radiating_pixels += 1;
            out.radiating_samples += samples;
            out.min_x0 = std::min(out.min_x0, h_min_x0[static_cast<size_t>(p)]);
            out.max_x0 = std::max(out.max_x0, h_max_x0[static_cast<size_t>(p)]);
            out.min_fluid_time = std::min(out.min_fluid_time, h_min_time[static_cast<size_t>(p)]);
            out.max_fluid_time = std::max(out.max_fluid_time, h_max_time[static_cast<size_t>(p)]);
        }
        if (h_returned[static_cast<size_t>(p)]) {
            out.camera_time_pixels += 1;
            out.min_camera_time = std::min(out.min_camera_time, h_camera_time[static_cast<size_t>(p)]);
            out.max_camera_time = std::max(out.max_camera_time, h_camera_time[static_cast<size_t>(p)]);
        }
    }
    if (out.radiating_samples == 0) {
        out.min_x0 = Real(0);
        out.max_x0 = Real(0);
        out.min_fluid_time = observation_time;
        out.max_fluid_time = observation_time;
    }
    if (out.camera_time_pixels == 0) {
        out.min_camera_time = observation_time;
        out.max_camera_time = observation_time;
    }
    return out;
}


#if KPOLARIS_ENABLE_ANALYSIS_MODE
template<class ExecSpace, class RealT>
KPOLARIS_INLINE kpolaris::PassBAnalysisDiagnostics<RealT> load_slow_light_analysis(
    const SlowLightViews<ExecSpace, RealT>& v,
    int pixel) {
    kpolaris::PassBAnalysisDiagnostics<RealT> a;
    a.radiating_path_length = v.radiating_path_length(pixel);
    a.emission_weight = v.emission_weight(pixel);
    a.emission_weighted_radius = v.emission_weighted_radius(pixel);
    a.emission_weighted_optical_depth_to_camera = v.emission_weighted_optical_depth_to_camera(pixel);
    a.absorption_depth = v.absorption_depth(pixel);
    a.absorption_operator_depth = v.absorption_operator_depth(pixel);
    a.faraday_rotation_depth = v.faraday_rotation_depth(pixel);
    a.faraday_conversion_depth = v.faraday_conversion_depth(pixel);
    a.faraday_operator_depth = v.faraday_operator_depth(pixel);
    a.dominant_emission_radius = v.dominant_emission_radius(pixel);
    a.dominant_emission_weight = v.dominant_emission_weight(pixel);
    a.dominant_ne_cgs = v.dominant_ne_cgs(pixel);
    a.dominant_thetae = v.dominant_thetae(pixel);
    a.dominant_b_cgs = v.dominant_b_cgs(pixel);
    a.dominant_beta = v.dominant_beta(pixel);
    a.dominant_sigma = v.dominant_sigma(pixel);
    a.emission_weighted_ne_cgs = v.emission_weighted_ne_cgs(pixel);
    a.emission_weighted_thetae = v.emission_weighted_thetae(pixel);
    a.emission_weighted_b_cgs = v.emission_weighted_b_cgs(pixel);
    a.emission_weighted_beta = v.emission_weighted_beta(pixel);
    a.emission_weighted_sigma = v.emission_weighted_sigma(pixel);
    a.photon_ring_winding_estimate = v.photon_ring_winding_estimate(pixel);
    a.previous_phi = v.previous_phi(pixel);
    a.has_previous_phi = v.has_previous_phi(pixel);
    a.radiation_substeps = v.radiation_substeps(pixel);
    a.dominant_emission_region = v.dominant_emission_region(pixel);
    return a;
}

template<class ExecSpace, class RealT>
KPOLARIS_INLINE void store_slow_light_analysis_raw(
    const kpolaris::PassBAnalysisDiagnostics<RealT>& a,
    const SlowLightViews<ExecSpace, RealT>& v,
    int pixel) {
    v.radiating_path_length(pixel) = a.radiating_path_length;
    v.emission_weight(pixel) = a.emission_weight;
    v.emission_weighted_radius(pixel) = a.emission_weighted_radius;
    v.emission_weighted_optical_depth_to_camera(pixel) = a.emission_weighted_optical_depth_to_camera;
    v.absorption_depth(pixel) = a.absorption_depth;
    v.absorption_operator_depth(pixel) = a.absorption_operator_depth;
    v.faraday_rotation_depth(pixel) = a.faraday_rotation_depth;
    v.faraday_conversion_depth(pixel) = a.faraday_conversion_depth;
    v.faraday_operator_depth(pixel) = a.faraday_operator_depth;
    v.dominant_emission_radius(pixel) = a.dominant_emission_radius;
    v.dominant_emission_weight(pixel) = a.dominant_emission_weight;
    v.dominant_ne_cgs(pixel) = a.dominant_ne_cgs;
    v.dominant_thetae(pixel) = a.dominant_thetae;
    v.dominant_b_cgs(pixel) = a.dominant_b_cgs;
    v.dominant_beta(pixel) = a.dominant_beta;
    v.dominant_sigma(pixel) = a.dominant_sigma;
    v.emission_weighted_ne_cgs(pixel) = a.emission_weighted_ne_cgs;
    v.emission_weighted_thetae(pixel) = a.emission_weighted_thetae;
    v.emission_weighted_b_cgs(pixel) = a.emission_weighted_b_cgs;
    v.emission_weighted_beta(pixel) = a.emission_weighted_beta;
    v.emission_weighted_sigma(pixel) = a.emission_weighted_sigma;
    v.photon_ring_winding_estimate(pixel) = a.photon_ring_winding_estimate;
    v.previous_phi(pixel) = a.previous_phi;
    v.has_previous_phi(pixel) = a.has_previous_phi;
    v.radiation_substeps(pixel) = a.radiation_substeps;
    v.dominant_emission_region(pixel) = a.dominant_emission_region;
}

template<class ExecSpace, class RealT>
KPOLARIS_INLINE void store_slow_light_analysis_final(
    kpolaris::PassBAnalysisDiagnostics<RealT> a,
    const SlowLightViews<ExecSpace, RealT>& v,
    int pixel) {
    kpolaris::finalize_analysis(a);
    store_slow_light_analysis_raw(a, v, pixel);
}

template<class ExecSpace, class RealT>
KPOLARIS_INLINE kpolaris::PassBAnalysisResult<RealT>
load_slow_light_analysis_result(
    const SlowLightViews<ExecSpace, RealT>& v,
    int pixel,
    const kpolaris::AnalysisConfig<RealT>& config) {
    kpolaris::PassBAnalysisResult<RealT> out;
    out.analysis = load_slow_light_analysis(v, pixel);
    const int npix = v.image_i.extent_int(0);
    for (int bin = 0; bin < config.radial_bins; ++bin) {
        const int index = bin * npix + pixel;
        out.observer_radial_stokes[bin] = kpolaris::Stokes<RealT>(
            v.radial_stokes_i(index), v.radial_stokes_q(index),
            v.radial_stokes_u(index), v.radial_stokes_v(index));
        out.analysis.radial_absorption_depth[bin] =
            v.radial_absorption_depth(index);
        out.analysis.radial_faraday_rotation_depth[bin] =
            v.radial_faraday_rotation_depth(index);
        out.analysis.radial_faraday_conversion_depth[bin] =
            v.radial_faraday_conversion_depth(index);
        out.analysis.radial_faraday_operator_depth[bin] =
            v.radial_faraday_operator_depth(index);
    }
    return out;
}

template<class ExecSpace, class RealT>
KPOLARIS_INLINE void store_slow_light_analysis_result_raw(
    const kpolaris::PassBAnalysisResult<RealT>& out,
    const SlowLightViews<ExecSpace, RealT>& v,
    int pixel,
    const kpolaris::AnalysisConfig<RealT>& config) {
    store_slow_light_analysis_raw(out.analysis, v, pixel);
    const int npix = v.image_i.extent_int(0);
    for (int bin = 0; bin < config.radial_bins; ++bin) {
        const int index = bin * npix + pixel;
        v.radial_stokes_i(index) = out.observer_radial_stokes[bin].I;
        v.radial_stokes_q(index) = out.observer_radial_stokes[bin].Q;
        v.radial_stokes_u(index) = out.observer_radial_stokes[bin].U;
        v.radial_stokes_v(index) = out.observer_radial_stokes[bin].V;
        v.radial_absorption_depth(index) =
            out.analysis.radial_absorption_depth[bin];
        v.radial_faraday_rotation_depth(index) =
            out.analysis.radial_faraday_rotation_depth[bin];
        v.radial_faraday_conversion_depth(index) =
            out.analysis.radial_faraday_conversion_depth[bin];
        v.radial_faraday_operator_depth(index) =
            out.analysis.radial_faraday_operator_depth[bin];
    }
}

template<class ExecSpace, class RealT>
KPOLARIS_INLINE void finalize_slow_light_analysis_result(
    kpolaris::PassBAnalysisResult<RealT>& out,
    const kpolaris::BasisOverlap2<RealT>& overlap,
    const kpolaris::Stokes<RealT>& observed,
    const SlowLightViews<ExecSpace, RealT>& v,
    int pixel,
    const kpolaris::AnalysisConfig<RealT>& config) {
    kpolaris::finalize_analysis(out.analysis);
    out.transport.observed_stokes = observed;
    for (int bin = 0; bin < config.radial_bins; ++bin) {
        out.observer_radial_stokes[bin] = kpolaris::transform_to_observer_basis(
            out.observer_radial_stokes[bin], overlap);
    }
    kpolaris::finalize_response_basis(config.response, v.response_data, pixel,
        v.image_i.extent_int(0), overlap);
    kpolaris::finalize_observer_weighted_analysis(out, config);
    store_slow_light_analysis_result_raw(out, v, pixel, config);
}
#endif

// The scalar adapter keeps the existing window API and its timing semantics.
template<class RealT, class TemporalModel>
struct SlowLightSingleWindow {
    TemporalModel temporal;
    RealT end_time;
    KPOLARIS_INLINE int count() const { return 1; }
    KPOLARIS_INLINE bool decoupled() const { return false; }
    KPOLARIS_INLINE bool contains(RealT) const { return true; }
    KPOLARIS_INLINE int integration_windows() const { return 1; }
    KPOLARIS_INLINE int sample_window(RealT, int) const { return 0; }
    KPOLARIS_INLINE RealT observation_time() const { return temporal.observation_time; }
    KPOLARIS_INLINE RealT last_time() const { return end_time; }
    KPOLARIS_INLINE RealT upper_time(int) const { return end_time; }
    KPOLARIS_INLINE TemporalModel model(int) const { return temporal; }
};

template<class ExecSpace, class RealT, class RadiationModel>
struct SlowLightResidentWindows {
    Kokkos::View<RadiationModel*, ExecSpace> models;
    Kokkos::View<RealT*, ExecSpace> times;
    RealT observer_time;
    int first, last; // inclusive snapshot indices within this resident block
    kpolaris::SlowLightInterpolation interpolation = kpolaris::default_slow_light_interpolation;
    kpolaris::SlowLightStepMode step_mode = kpolaris::default_slow_light_step_mode;
    RealT frequency_override = RealT(0); // zero keeps the snapshot model frequency
    KPOLARIS_INLINE int count() const { return last - first; }
    KPOLARIS_INLINE bool decoupled() const {
        return step_mode == kpolaris::SlowLightStepMode::decoupled;
    }
    KPOLARIS_INLINE bool contains(RealT time) const {
        return time >= times(first) && time <= times(last);
    }
    KPOLARIS_INLINE int integration_windows() const {
        return step_mode == kpolaris::SlowLightStepMode::snapshot ? count() : 1;
    }
    KPOLARIS_INLINE int sample_window(RealT time, int integration_window) const {
        if (step_mode == kpolaris::SlowLightStepMode::snapshot) return integration_window;
        // Real snapshot timestamps can be nonuniform. Do not assume a cadence
        // or interpolate across all snapshots in the resident block at once.
        int lo = first, hi = last;
        while (lo + 1 < hi) {
            const int mid = lo + (hi - lo) / 2;
            if (time < times(mid)) hi = mid;
            else lo = mid;
        }
        return lo - first;
    }
    KPOLARIS_INLINE RealT observation_time() const { return observer_time; }
    KPOLARIS_INLINE RealT last_time() const { return times(last); }
    KPOLARIS_INLINE RealT upper_time(int i) const {
        return step_mode == kpolaris::SlowLightStepMode::block ? last_time() : times(first + i + 1);
    }
    KPOLARIS_INLINE SlowLightTemporalModel<RealT, RadiationModel> model(int i) const {
        const int k = first + i;
        SlowLightTemporalModel<RealT, RadiationModel> temporal{
            models(k), models(k + 1), times(k), times(k + 1),
            observer_time, models(k).freq_cgs, interpolation};
        if (frequency_override > RealT(0)) temporal.set_frequency(frequency_override);
        return temporal;
    }
};

// Image-major ordering retains adjacent-pixel memory access while letting the
// GPU schedule the next image before the slowest rays of an earlier one finish.
template<class ExecSpace, class RealT, class Windows>
struct SlowLightSingleRayImage {
    static constexpr int max_frequencies = 1;
    SlowLightViews<ExecSpace, RealT> image;
    Windows windows;
    KPOLARIS_INLINE int size() const { return 1; }
    KPOLARIS_INLINE int image_index(int, int) const { return 0; }
    KPOLARIS_INLINE int pixel_index(int flat, int) const { return flat; }
    KPOLARIS_INLINE const SlowLightViews<ExecSpace, RealT>& states(int) const { return image; }
    KPOLARIS_INLINE int frequency_count(int) const { return 1; }
    KPOLARIS_INLINE RealT frequency(int, int) const { return RealT(0); }
    KPOLARIS_INLINE const SlowLightViews<ExecSpace, RealT>& frequency_states(int, int) const { return image; }
    KPOLARIS_INLINE const Windows& temporal_windows(int) const { return windows; }
};

template<class ExecSpace, class RealT, class Windows>
struct SlowLightRayBatch {
    static constexpr int max_frequencies = 1;
    Kokkos::View<SlowLightViews<ExecSpace, RealT>*, ExecSpace> images;
    Kokkos::View<Windows*, ExecSpace> windows;
    KPOLARIS_INLINE int size() const { return images.extent_int(0); }
    KPOLARIS_INLINE int image_index(int flat, int npix) const { return flat / npix; }
    KPOLARIS_INLINE int pixel_index(int flat, int npix) const { return flat % npix; }
    KPOLARIS_INLINE const SlowLightViews<ExecSpace, RealT>& states(int image) const { return images(image); }
    KPOLARIS_INLINE int frequency_count(int) const { return 1; }
    KPOLARIS_INLINE RealT frequency(int, int) const { return RealT(0); }
    KPOLARIS_INLINE const SlowLightViews<ExecSpace, RealT>& frequency_states(int image, int) const { return images(image); }
    KPOLARIS_INLINE const Windows& temporal_windows(int image) const { return windows(image); }
};

// Each group shares its geometric step. Frequency-dependent state stays in
// separate views; all groups refer to the same resident snapshot descriptors.
// A capacity-one specialization keeps single-frequency register usage small.
template<class ExecSpace, class RealT, class Windows, int Capacity>
struct SlowLightFrequencyRayBatch {
    static constexpr int max_frequencies = Capacity;
    Kokkos::View<SlowLightViews<ExecSpace, RealT>*, ExecSpace> images;
    Kokkos::View<Windows*, ExecSpace> windows;
    Kokkos::View<int*, ExecSpace> offsets, counts;
    Kokkos::View<RealT*, ExecSpace> frequencies;
    KPOLARIS_INLINE int size() const { return windows.extent_int(0); }
    KPOLARIS_INLINE int image_index(int flat, int npix) const { return flat / npix; }
    KPOLARIS_INLINE int pixel_index(int flat, int npix) const { return flat % npix; }
    KPOLARIS_INLINE int frequency_count(int image) const { return counts(image); }
    KPOLARIS_INLINE RealT frequency(int image, int f) const { return frequencies(offsets(image) + f); }
    KPOLARIS_INLINE const SlowLightViews<ExecSpace, RealT>& states(int image) const { return images(offsets(image)); }
    KPOLARIS_INLINE const SlowLightViews<ExecSpace, RealT>& frequency_states(int image, int f) const {
        return images(offsets(image) + f);
    }
    KPOLARIS_INLINE const Windows& temporal_windows(int image) const { return windows(image); }
};

template<bool AnalysisMode, class ExecSpace, class RealT, class Rays, class Metric>
void launch_slow_light_rays_metric(
    const kpolaris::PassAParams<RealT>& params,
    const Rays& rays,
    const Metric& metric,
    int timing,
    const kpolaris::AnalysisConfig<RealT>& analysis_config = {},
    bool synchronize = true) {
    // The ordinary image kernel must not reserve per-ray analysis scratch or
    // registers merely because the executable also supports diagnostics.
    constexpr int analysis_mode = AnalysisMode ? 1 : 0;
    const int npix = params.camera.nx * params.camera.ny;
    if (rays.size() > std::numeric_limits<int>::max() / npix)
        throw std::invalid_argument("slow-light ray batch exceeds int-sized indexing");
    Kokkos::Timer timer;
    Kokkos::parallel_for(
        "KPOLARISSlowLightWindow",
        Kokkos::RangePolicy<ExecSpace>(0, npix * rays.size()),
        KOKKOS_LAMBDA(const int flat) {
            const int image = rays.image_index(flat, npix);
            const int pixel = rays.pixel_index(flat, npix);
            const auto& v = rays.states(image);
            const auto& windows = rays.temporal_windows(image);
            if (!v.active(pixel)) {
                return;
            }
            const int base = pixel * kpolaris::ndim;
            // A ray may not enter this block until a later fluid time. Avoid
            // loading its complete transport/analysis state in empty windows.
            // Step-limited rays must still pass through the finalization below.
            if (windows.decoupled() && v.waiting_for_data(pixel) &&
                !windows.contains(v.requested_time(pixel))) return;
            if (!windows.decoupled() && windows.observation_time() + v.state_x(base) >= windows.last_time() &&
                v.steps(pixel) < params.max_steps) return;
            kpolaris::TransportState<RealT> state;
            for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                state.x[mu] = v.state_x(base + mu);
                state.k[mu] = v.state_k(base + mu);
                state.e1[mu] = v.state_e1(base + mu);
                state.e2[mu] = v.state_e2(base + mu);
            }
            constexpr int capacity = Rays::max_frequencies;
            const int nfreq = rays.frequency_count(image);
            kpolaris::Stokes<RealT> stokes[capacity];
            for (int f = 0; f < nfreq; ++f) {
                const auto& vf = rays.frequency_states(image, f);
                stokes[f] = {vf.stokes_i(pixel), vf.stokes_q(pixel), vf.stokes_u(pixel), vf.stokes_v(pixel)};
            }
            const int screen_orientation = kpolaris::backward_screen_orientation_sign(params.camera);
            kpolaris::AdaptiveRK4Control<RealT> transfer_control;
            transfer_control.tolerance = params.adaptive_tolerance;
            transfer_control.min_step = params.min_step;
            transfer_control.max_step = params.max_step;
            RealT h_current = v.h_current(pixel);
            RealT radiation_step_cap = v.radiation_step_cap(pixel);
            int local_steps = v.steps(pixel);
            int reached_camera = 0;
#if KPOLARIS_ENABLE_ANALYSIS_MODE
            kpolaris::PassBAnalysisResult<RealT> analysis_result;
            if (analysis_mode && capacity == 1) {
                analysis_result = load_slow_light_analysis_result(
                    v, pixel, analysis_config);
            }
#else
            (void)analysis_mode;
#endif

            for (int window = 0; window < windows.integration_windows(); ++window) {
                if (reached_camera || !v.active(pixel)) break;
                auto radiation_model = windows.model(window);
                int coefficient_window = window;
                const RealT window_upper_time = windows.upper_time(window);
                while (!reached_camera && local_steps < params.max_steps) {
                    const RealT retry_h = h_current;
                    RealT h = h_current;
                    const kpolaris::TransportState<RealT> old_state = state;
                    const RealT old_sample_time = radiation_model.observation_time + old_state.x[0];
                    if (!windows.decoupled() && old_sample_time >= window_upper_time) {
                        break;
                    }
                    const RealT start_r_bl = kpolaris::radial_coordinate(metric, old_state.x);
                    if (start_r_bl <= params.outer_radius && radiation_step_cap > RealT(0)) {
                        h = kpolaris::min_val(h, radiation_step_cap);
                    }
                    kpolaris::TransportState<RealT> next_state;
                    kpolaris::TransportState<RealT> sample_state;
                    if (params.adaptive) {
                        const auto proposed = kpolaris::adaptive_rk4_step(metric, old_state, h, transfer_control);
                        if (!proposed.accepted) {
                            h_current = kpolaris::abs_val(proposed.next_h);
                            if (h_current <= params.min_step * RealT(1.0001)) {
                                v.reason(pixel) = static_cast<int>(kpolaris::TerminationReason::max_steps);
                                v.active(pixel) = 0;
                                break;
                            }
                            continue;
                        }
                        h = kpolaris::abs_val(proposed.used_h);
                        h_current = kpolaris::abs_val(proposed.next_h);
                        next_state = proposed.state;
                        sample_state = proposed.mid_state;
                    } else {
                        next_state = kpolaris::rk4_step(metric, old_state, h);
                        sample_state = next_state;
                    }

                    const RealT s0 = kpolaris::camera_surface_value(metric, params.camera, old_state);
                    const RealT s1 = kpolaris::camera_surface_value(metric, params.camera, next_state);
                    int crosses_camera = kpolaris::camera_surface_crossed(s0, s1);
                    RealT camera_frac = RealT(2);
                    if (crosses_camera) {
                        camera_frac = kpolaris::camera_surface_crossing_fraction(s0, s1);
                    }
                    const RealT trial_next_sample_time = radiation_model.observation_time + next_state.x[0];
                    const int crosses_time = (!windows.decoupled() && trial_next_sample_time > window_upper_time &&
                                              trial_next_sample_time > old_sample_time);
                    RealT time_frac = RealT(2);
                    if (crosses_time) {
                        time_frac = kpolaris::min_val(RealT(1), kpolaris::max_val(RealT(0),
                            (window_upper_time - old_sample_time) / (trial_next_sample_time - old_sample_time)));
                    }

                    if (crosses_time && (!crosses_camera || time_frac < camera_frac)) {
                        crosses_camera = 0;
                        const RealT h_boundary = h * time_frac;
                        if (kpolaris::abs_val(h_boundary) <= RealT(1e-14)) {
                            break;
                        }
                        h = h_boundary;
                        if (params.adaptive) {
                            kpolaris::AdaptiveRK4Control<RealT> boundary_control = transfer_control;
                            boundary_control.min_step = kpolaris::min_val(boundary_control.min_step, kpolaris::abs_val(h));
                            boundary_control.max_step = kpolaris::max_val(boundary_control.min_step, kpolaris::abs_val(h));
                            const auto boundary_step = kpolaris::adaptive_rk4_step(metric, old_state, h, boundary_control);
                            if (!boundary_step.accepted) {
                                h_current = kpolaris::abs_val(boundary_step.next_h);
                                continue;
                            }
                            h = kpolaris::abs_val(boundary_step.used_h);
                            next_state = boundary_step.state;
                            sample_state = boundary_step.mid_state;
                        } else {
                            next_state = kpolaris::rk4_step(metric, old_state, h);
                            sample_state = next_state;
                        }
                        // Two half RK steps can each round back to the old time
                        // when the remaining interval is one time-coordinate ulp.
                        // Do not accumulate emission repeatedly at the same point.
                        if (next_state.x[0] == old_state.x[0]) {
                            const RealT time_scale = kpolaris::max_val(
                                kpolaris::abs_val(old_state.x[0]),
                                kpolaris::max_val(kpolaris::abs_val(window_upper_time),
                                                  kpolaris::abs_val(old_sample_time - old_state.x[0])));
                            const RealT rounding = RealT(0.5) * kpolaris::adaptive_tolerance_floor<RealT>() * time_scale;
                            if (kpolaris::abs_val(window_upper_time - old_sample_time) > rounding) {
                                v.reason(pixel) = static_cast<int>(kpolaris::TerminationReason::adaptive_step_underflow);
                                v.active(pixel) = 0;
                            }
                            // Within rounding resolution, let the next snapshot
                            // interval continue from the unchanged physical state.
                            break;
                        }
                    } else if (crosses_camera) {
                        const RealT h_cross = h * camera_frac;
                        if (kpolaris::abs_val(h_cross) <= RealT(1e-14)) {
                            h = RealT(0);
                            next_state = old_state;
                            sample_state = old_state;
                        } else if (camera_frac < RealT(0.999999999999)) {
                            h = h_cross;
                            if (params.adaptive) {
                                kpolaris::AdaptiveRK4Control<RealT> crossing_control = transfer_control;
                                crossing_control.min_step = kpolaris::min_val(crossing_control.min_step, kpolaris::abs_val(h));
                                crossing_control.max_step = kpolaris::max_val(crossing_control.min_step, kpolaris::abs_val(h));
                                const auto crossing_step = kpolaris::adaptive_rk4_step(metric, old_state, h, crossing_control);
                                if (!crossing_step.accepted) {
                                    h_current = kpolaris::abs_val(crossing_step.next_h);
                                    continue;
                                }
                                h = kpolaris::abs_val(crossing_step.used_h);
                                next_state = crossing_step.state;
                                sample_state = crossing_step.mid_state;
                            } else {
                                next_state = kpolaris::rk4_step(metric, old_state, h);
                                sample_state = next_state;
                            }
                        }
                    }

                    if (params.emission_selection.narrow_wedge()) {
                        if (!params.adaptive) sample_state = kpolaris::rk4_step(metric, old_state, h*RealT(0.5));
                        const int wedge_substeps = kpolaris::equatorial_resolution_substeps(params.emission_selection,
                            metric, old_state, sample_state, next_state, params.outer_radius);
                        if (wedge_substeps > 1) {
                            if (h <= params.min_step*RealT(1.0001)) {
                                v.reason(pixel) = static_cast<int>(kpolaris::TerminationReason::adaptive_step_underflow);
                                v.active(pixel) = 0;
                                break;
                            }
                            h_current = kpolaris::max_val(params.min_step, h/RealT(wedge_substeps));
                            radiation_step_cap = h_current;
                            continue;
                        }
                        if (kpolaris::refine_equatorial_boundary(params.emission_selection, metric, old_state,
                                sample_state, next_state, h, params.adaptive, params.outer_radius)) {
                            crosses_camera = kpolaris::camera_surface_crossed(s0,
                                kpolaris::camera_surface_value(metric, params.camera, next_state));
                        }
                    }

                    const RealT old_r_bl = kpolaris::radial_coordinate(metric, old_state.x);
                    const RealT sample_r_bl = kpolaris::radial_coordinate(metric, sample_state.x);
                    const RealT next_r_bl = kpolaris::radial_coordinate(metric, next_state.x);
                    const int radiation_active =
                        (sample_r_bl >= params.inner_radius && sample_r_bl < params.outer_radius) ? 1 : 0;
                    const RealT coefficient_time = windows.observation_time() + sample_state.x[0];
                    if (windows.decoupled() && radiation_active && !windows.contains(coefficient_time)) {
                        v.requested_time(pixel) = coefficient_time;
                        v.waiting_for_data(pixel) = 1;
                        h_current = retry_h;
                        break;
                    }
                    v.waiting_for_data(pixel) = 0;
                    const int sample_window = windows.decoupled() && !radiation_active ?
                        coefficient_window : windows.sample_window(coefficient_time, window);
                    if (sample_window != coefficient_window) {
                        radiation_model = windows.model(sample_window);
                        coefficient_window = sample_window;
                    }
                    const RealT local_max_radiation_step =
                        (params.max_radiation_step > RealT(0) &&
                         (old_r_bl <= params.outer_radius || sample_r_bl <= params.outer_radius ||
                          next_r_bl <= params.outer_radius)) ? params.max_radiation_step : RealT(0);
                    kpolaris::TransferCoeffs<RealT> coeffs[capacity]{};
                    RealT dlambda_scales[capacity];
                    int required_steps = 1, predictive_steps = 1;
                    for (int f = 0; f < nfreq; ++f) {
                        const RealT frequency = rays.frequency(image, f);
                        if (frequency > RealT(0)) radiation_model.set_frequency(frequency);
                        dlambda_scales[f] = radiation_model.dlambda_scale();
                        if (radiation_active) {
                            coeffs[f] = radiation_model.coefficients(metric, sample_state, RealT(0.5));
                            kpolaris::apply_emission_selection(params.emission_selection, metric, sample_state, coeffs[f]);
                            kpolaris::transform_axial_coefficients_to_screen_orientation(coeffs[f], screen_orientation);
                        }
                        const int candidate = kpolaris::radiation_substep_count(
                            coeffs[f], h, dlambda_scales[f], 1, local_max_radiation_step,
                            params.max_radiation_depth, params.max_absorption_depth, params.max_faraday_depth);
                        required_steps = candidate > required_steps ? candidate : required_steps;
                        const int predictive = kpolaris::radiation_substep_count(
                            coeffs[f], h_current, dlambda_scales[f], 1, local_max_radiation_step,
                            params.max_radiation_depth, params.max_absorption_depth, params.max_faraday_depth);
                        predictive_steps = predictive > predictive_steps ? predictive : predictive_steps;
                    }
                    if (required_steps > 1 && h > params.min_step * RealT(1.0001)) {
                        h_current = kpolaris::max_val(params.min_step, h / RealT(required_steps));
                        radiation_step_cap = h_current;
                        continue;
                    }
                    radiation_step_cap = predictive_steps > 1 ?
                        kpolaris::max_val(params.min_step, h_current / RealT(predictive_steps)) : params.max_step;
                    // Commit only after geometry, data availability, and every
                    // frequency's radiation-step check have succeeded.
                    if (h > RealT(0)) {
                        for (int f = 0; f < nfreq; ++f) {
                            const auto& vf = rays.frequency_states(image, f);
#if KPOLARIS_ENABLE_ANALYSIS_MODE
                            if (analysis_mode) {
                                const RealT frequency = rays.frequency(image, f);
                                if (frequency > RealT(0)) radiation_model.set_frequency(frequency);
                                // Keep scalar diagnostics in registers. Fused diagnostics
                                // use one scratch result at a time, not nfreq large arrays.
                                if constexpr (capacity > 1)
                                    analysis_result = load_slow_light_analysis_result(vf, pixel, analysis_config);
                                kpolaris::accumulate_analysis_phi(
                                    metric, radiation_model, sample_state,
                                    analysis_result.analysis);
                                if (radiation_active) {
                                    const RealT dlambda = kpolaris::abs_val(h) * dlambda_scales[f];
                                    RealT analysis_r = RealT(0), analysis_th = RealT(0);
                                    RealT analysis_cp = RealT(1), analysis_sp = RealT(0);
                                    radiation_model.bl_coordinates_for_metric(
                                        metric, sample_state.x, analysis_r, analysis_th,
                                        analysis_cp, analysis_sp);
                                    (void)analysis_th;
                                    (void)analysis_cp;
                                    (void)analysis_sp;
                                    const int radial_bin = kpolaris::analysis_radial_bin(
                                        analysis_config, analysis_r);
                                    kpolaris::accumulate_physical_analysis_sample(
                                        metric, radiation_model, sample_state, coeffs[f],
                                        dlambda, analysis_r, radial_bin,
                                        analysis_result.analysis);
                                    kpolaris::accumulate_response_sample(metric, radiation_model,
                                        sample_state, stokes[f], coeffs[f], dlambda, screen_orientation,
                                        analysis_config.response, vf.response_data, pixel, npix, params.emission_selection);
                                    const auto propagation =
                                        kpolaris::transfer_without_emission(coeffs[f]);
                                    for (int bin = 0; bin < analysis_config.radial_bins; ++bin) {
                                        kpolaris::semi_analytic_stokes_step(
                                            analysis_result.observer_radial_stokes[bin],
                                            bin == radial_bin ? coeffs[f] : propagation,
                                            dlambda);
                                    }
                                    analysis_result.analysis.radiation_substeps += 1;
                                }
                                if constexpr (capacity > 1)
                                    store_slow_light_analysis_result_raw(analysis_result, vf, pixel, analysis_config);
                            }
#endif
                            kpolaris::semi_analytic_stokes_step(stokes[f], coeffs[f],
                                kpolaris::abs_val(h) * dlambda_scales[f]);
                        }
                    }
                    state = next_state;
                    local_steps += 1;
                    if (crosses_camera) {
                        reached_camera = 1;
                    }
                }

            } // resident time windows

            const int common_reason = v.reason(pixel);
            const int common_active = v.active(pixel);
            const RealT requested_time = v.requested_time(pixel);
            const int waiting_for_data = v.waiting_for_data(pixel);
            for (int f = 0; f < nfreq; ++f) {
                const auto& vf = rays.frequency_states(image, f);
                vf.reason(pixel) = common_reason;
                vf.active(pixel) = common_active;
                vf.requested_time(pixel) = requested_time;
                vf.waiting_for_data(pixel) = waiting_for_data;
                for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                    vf.state_x(base + mu) = state.x[mu];
                    vf.state_k(base + mu) = state.k[mu];
                    vf.state_e1(base + mu) = state.e1[mu];
                    vf.state_e2(base + mu) = state.e2[mu];
                }
                vf.stokes_i(pixel) = stokes[f].I;
                vf.stokes_q(pixel) = stokes[f].Q;
                vf.stokes_u(pixel) = stokes[f].U;
                vf.stokes_v(pixel) = stokes[f].V;
                vf.h_current(pixel) = h_current;
                vf.radiation_step_cap(pixel) = radiation_step_cap;
                vf.steps(pixel) = local_steps;
#if KPOLARIS_ENABLE_ANALYSIS_MODE
                if (analysis_mode && capacity == 1) {
                    store_slow_light_analysis_result_raw(
                        analysis_result, vf, pixel, analysis_config);
                }
#endif

                if (reached_camera || local_steps >= params.max_steps || !vf.active(pixel)) {
                    // The camera ray is used only to express the completed image in
                    // the observer basis, never to advance an intermediate window.
                    const auto camera_state = kpolaris::initialize_camera_ray(metric, pixel, params.camera);
                    const auto overlap = kpolaris::screen_overlap(metric, state.x, camera_state, state);
                    const auto observed = kpolaris::transform_to_observer_basis(stokes[f], overlap);
                    vf.image_i(pixel) = observed.I;
                    vf.image_q(pixel) = observed.Q;
                    vf.image_u(pixel) = observed.U;
                    vf.image_v(pixel) = observed.V;
                    vf.closure_x(pixel) = kpolaris::spatial_distance(state.x, camera_state.x);
                    kpolaris::Vec4<RealT> target_k;
                    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                        target_k[mu] = camera_state.k[mu];
                    }
                    vf.closure_k(pixel) = kpolaris::vector_max_abs_difference(state.k, target_k);
                    vf.final_null(pixel) = metric.dot(state.x, state.k, state.k);
                    vf.frame_error(pixel) = kpolaris::max_frame_error(kpolaris::frame_errors(metric, state));
                    vf.det_r(pixel) = overlap.det();
                    vf.overlap_r11(pixel) = overlap.r11;
                    vf.overlap_r12(pixel) = overlap.r12;
                    vf.overlap_r21(pixel) = overlap.r21;
                    vf.overlap_r22(pixel) = overlap.r22;
                    vf.basis_identity_error(pixel) =
                        kpolaris::max_val(kpolaris::max_val(kpolaris::abs_val(overlap.r11 - RealT(1)),
                                                            kpolaris::abs_val(overlap.r22 - RealT(1))),
                                          kpolaris::max_val(kpolaris::abs_val(overlap.r12),
                                                            kpolaris::abs_val(overlap.r21)));
                    vf.basis_rotation_angle(pixel) = Kokkos::atan2(overlap.r21 - overlap.r12,
                                                                  overlap.r11 + overlap.r22);
#if KPOLARIS_ENABLE_ANALYSIS_MODE
                    if (analysis_mode) {
                        if constexpr (capacity > 1)
                            analysis_result = load_slow_light_analysis_result(vf, pixel, analysis_config);
                        finalize_slow_light_analysis_result(
                            analysis_result, overlap, observed, vf, pixel,
                            analysis_config);
                    }
#endif
                    if (reached_camera)
                        vf.reason(pixel) = static_cast<int>(kpolaris::TerminationReason::reached_camera);
                    else if (vf.reason(pixel) != static_cast<int>(kpolaris::TerminationReason::adaptive_step_underflow))
                        vf.reason(pixel) = static_cast<int>(kpolaris::TerminationReason::max_steps);
                    vf.active(pixel) = 0;
                }
            } // frequency channels
        });
    if (synchronize || timing) Kokkos::fence();
    report_image_timing(timing, "slow_light_pass_b_window", timer.seconds());
}

template<class ExecSpace, class RealT, class Rays, class Metric>
void run_slow_light_rays_metric(
    const kpolaris::PassAParams<RealT>& params,
    const Rays& rays,
    const Metric& metric,
    int timing,
    int analysis_mode = 0,
    const kpolaris::AnalysisConfig<RealT>& analysis_config = {},
    bool synchronize = true) {
    if (analysis_mode) {
#if KPOLARIS_ENABLE_ANALYSIS_MODE
        launch_slow_light_rays_metric<true, ExecSpace>(params, rays, metric, timing, analysis_config, synchronize);
#else
        throw std::runtime_error("slow-light analysis is disabled in this build");
#endif
    } else {
        launch_slow_light_rays_metric<false, ExecSpace>(params, rays, metric, timing, analysis_config, synchronize);
    }
}

template<class ExecSpace, class RealT, class Windows, class Metric>
void run_slow_light_windows_metric(
    const kpolaris::PassAParams<RealT>& params,
    const Windows& windows,
    const Metric& metric,
    SlowLightViews<ExecSpace, RealT>& v,
    int timing,
    int analysis_mode = 0,
    const kpolaris::AnalysisConfig<RealT>& analysis_config = {},
    bool synchronize = true) {
    const SlowLightSingleRayImage<ExecSpace, RealT, Windows> rays{v, windows};
    run_slow_light_rays_metric<ExecSpace>(params, rays, metric, timing,
                                         analysis_mode, analysis_config, synchronize);
}

template<class ExecSpace, class RealT, class TemporalModel, class Metric>
void run_slow_light_window_metric(
    const kpolaris::PassAParams<RealT>& params,
    const TemporalModel& radiation_model,
    const Metric& metric,
    SlowLightViews<ExecSpace, RealT>& v,
    RealT window_upper_time,
    int timing,
    int analysis_mode = 0,
    const kpolaris::AnalysisConfig<RealT>& analysis_config = {}) {
    const SlowLightSingleWindow<RealT, TemporalModel> window{radiation_model, window_upper_time};
    run_slow_light_windows_metric(params, window, metric, v, timing, analysis_mode, analysis_config);
}

template<class ExecSpace, class RadiationModel>
Kokkos::View<RadiationModel*, ExecSpace> copy_slow_light_models_to_device(
    const std::vector<RadiationModel>& models,
    const char* label) {
    Kokkos::View<RadiationModel*, ExecSpace> device_models(label, models.size());
    auto host_models = Kokkos::create_mirror_view(device_models);
    for (size_t i = 0; i < models.size(); ++i) {
        host_models(static_cast<int>(i)) = models[i];
    }
    Kokkos::deep_copy(device_models, host_models);
    return device_models;
}

struct SlowLightVacuumNoOpObserver {
    template<class State>
    KPOLARIS_INLINE void operator()(const State&) const {}
};

#if KPOLARIS_ENABLE_ANALYSIS_MODE
template<class Metric, class RadiationModel, class RealT>
struct SlowLightVacuumAnalysisObserver {
    const Metric& metric;
    const RadiationModel& radiation_model;
    kpolaris::PassBAnalysisDiagnostics<RealT>& analysis;
    int enabled = 0;

    KPOLARIS_INLINE void operator()(
        const kpolaris::TransportState<RealT>& sample_state) const {
        if (enabled) {
            kpolaris::accumulate_analysis_phi(
                metric, radiation_model, sample_state, analysis);
        }
    }
};
#endif

template<class RealT, class Metric, class StepObserver>
KPOLARIS_INLINE kpolaris::TerminationReason advance_slow_light_vacuum_to_camera(
    const kpolaris::PassAParams<RealT>& params,
    const Metric& metric,
    kpolaris::TransportState<RealT>& state,
    RealT& h_current,
    int& local_steps,
    StepObserver observer) {
    const RealT radial_tolerance =
        RealT(1e-8) * kpolaris::max_val(RealT(1), kpolaris::abs_val(params.outer_radius));
    if (kpolaris::radial_coordinate(metric, state.x) <
        params.outer_radius - radial_tolerance) {
        return kpolaris::TerminationReason::slow_light_time_exhausted;
    }

    kpolaris::AdaptiveRK4Control<RealT> transfer_control;
    transfer_control.tolerance = params.adaptive_tolerance;
    transfer_control.min_step = params.min_step;
    transfer_control.max_step = params.max_step;
    while (local_steps < params.max_steps) {
        RealT h = h_current;
        const kpolaris::TransportState<RealT> old_state = state;
        kpolaris::TransportState<RealT> next_state;
        kpolaris::TransportState<RealT> sample_state;
        if (params.adaptive) {
            const auto proposed =
                kpolaris::adaptive_rk4_step(metric, old_state, h, transfer_control);
            if (!proposed.accepted) {
                h_current = kpolaris::abs_val(proposed.next_h);
                if (h_current <= params.min_step * RealT(1.0001)) {
                    return kpolaris::TerminationReason::max_steps;
                }
                continue;
            }
            h = kpolaris::abs_val(proposed.used_h);
            h_current = kpolaris::abs_val(proposed.next_h);
            next_state = proposed.state;
            sample_state = proposed.mid_state;
        } else {
            next_state = kpolaris::rk4_step(metric, old_state, h);
            sample_state = next_state;
        }

        const RealT s0 = kpolaris::camera_surface_value(metric, params.camera, old_state);
        const RealT s1 = kpolaris::camera_surface_value(metric, params.camera, next_state);
        const int crosses_camera = kpolaris::camera_surface_crossed(s0, s1);
        if (crosses_camera) {
            const RealT fraction = kpolaris::camera_surface_crossing_fraction(s0, s1);
            const RealT h_cross = h * fraction;
            if (kpolaris::abs_val(h_cross) <= RealT(1e-14)) {
                h = RealT(0);
                next_state = old_state;
                sample_state = old_state;
            } else if (fraction < RealT(0.999999999999)) {
                h = h_cross;
                if (params.adaptive) {
                    kpolaris::AdaptiveRK4Control<RealT> crossing_control = transfer_control;
                    crossing_control.min_step = kpolaris::min_val(
                        crossing_control.min_step, kpolaris::abs_val(h));
                    crossing_control.max_step = kpolaris::max_val(
                        crossing_control.min_step, kpolaris::abs_val(h));
                    const auto crossing_step =
                        kpolaris::adaptive_rk4_step(metric, old_state, h, crossing_control);
                    if (!crossing_step.accepted) {
                        h_current = kpolaris::abs_val(crossing_step.next_h);
                        continue;
                    }
                    h = kpolaris::abs_val(crossing_step.used_h);
                    next_state = crossing_step.state;
                    sample_state = crossing_step.mid_state;
                } else {
                    next_state = kpolaris::rk4_step(metric, old_state, h);
                    sample_state = next_state;
                }
            }
        }

        // A missing fluid window may only be crossed in outward vacuum.
        // Do not extrapolate plasma if the ray re-enters the emitting region.
        const RealT old_radius = kpolaris::radial_coordinate(metric, old_state.x);
        const RealT next_radius = kpolaris::radial_coordinate(metric, next_state.x);
        const RealT sample_radius = kpolaris::radial_coordinate(metric, sample_state.x);
        if (!Kokkos::isfinite(next_radius) || !Kokkos::isfinite(sample_radius)) {
            return kpolaris::TerminationReason::max_steps;
        }
        if (next_radius < old_radius - radial_tolerance ||
            sample_radius < params.outer_radius - radial_tolerance ||
            next_radius < params.outer_radius - radial_tolerance) {
            return kpolaris::TerminationReason::slow_light_time_exhausted;
        }
        if (h > RealT(0)) {
            observer(sample_state);
        }
        state = next_state;
        local_steps += 1;
        if (crosses_camera) {
            return kpolaris::TerminationReason::reached_camera;
        }
    }
    return kpolaris::TerminationReason::max_steps;
}

template<class ExecSpace, class RealT, class Metric, class RadiationModel>
void finalize_slow_light_unfinished_metric(
    const kpolaris::PassAParams<RealT>& params,
    const Metric& metric,
    const RadiationModel& geometry_model,
    SlowLightViews<ExecSpace, RealT>& v,
    int timing,
    int analysis_mode = 0,
    const kpolaris::AnalysisConfig<RealT>& analysis_config = {}) {
    const int npix = params.camera.nx * params.camera.ny;
    Kokkos::Timer timer;
    Kokkos::parallel_for(
        "KPOLARISSlowLightFinalizeUnfinished",
        Kokkos::RangePolicy<ExecSpace>(0, npix),
        KOKKOS_LAMBDA(const int pixel) {
            if (!v.active(pixel)) {
                return;
            }
            const int base = pixel * kpolaris::ndim;
            kpolaris::TransportState<RealT> state;
            for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                state.x[mu] = v.state_x(base + mu);
                state.k[mu] = v.state_k(base + mu);
                state.e1[mu] = v.state_e1(base + mu);
                state.e2[mu] = v.state_e2(base + mu);
            }
            const auto camera_state = kpolaris::initialize_camera_ray(metric, pixel, params.camera);
            const kpolaris::Stokes<RealT> stokes(v.stokes_i(pixel), v.stokes_q(pixel),
                                                 v.stokes_u(pixel), v.stokes_v(pixel));
#if KPOLARIS_ENABLE_ANALYSIS_MODE
            kpolaris::PassBAnalysisResult<RealT> analysis_result;
            if (analysis_mode) {
                analysis_result = load_slow_light_analysis_result(
                    v, pixel, analysis_config);
            }
            SlowLightVacuumAnalysisObserver<Metric, RadiationModel, RealT> observer{
                metric, geometry_model, analysis_result.analysis, analysis_mode};
#else
            (void)analysis_mode;
            (void)analysis_config;
#endif
            RealT h_current = v.h_current(pixel);
            int local_steps = v.steps(pixel);
#if KPOLARIS_ENABLE_ANALYSIS_MODE
            const auto reason = advance_slow_light_vacuum_to_camera(
                params, metric, state, h_current, local_steps, observer);
#else
            const auto reason = advance_slow_light_vacuum_to_camera(
                params, metric, state, h_current, local_steps,
                SlowLightVacuumNoOpObserver{});
#endif
            for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                v.state_x(base + mu) = state.x[mu];
                v.state_k(base + mu) = state.k[mu];
                v.state_e1(base + mu) = state.e1[mu];
                v.state_e2(base + mu) = state.e2[mu];
            }
            v.h_current(pixel) = h_current;
            v.steps(pixel) = local_steps;
            const auto overlap = kpolaris::screen_overlap(metric, state.x, camera_state, state);
            const auto observed = kpolaris::transform_to_observer_basis(stokes, overlap);
            v.image_i(pixel) = observed.I;
            v.image_q(pixel) = observed.Q;
            v.image_u(pixel) = observed.U;
            v.image_v(pixel) = observed.V;
            v.closure_x(pixel) = kpolaris::spatial_distance(state.x, camera_state.x);
            kpolaris::Vec4<RealT> target_k;
            for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                target_k[mu] = camera_state.k[mu];
            }
            v.closure_k(pixel) = kpolaris::vector_max_abs_difference(state.k, target_k);
            v.final_null(pixel) = metric.dot(state.x, state.k, state.k);
            v.frame_error(pixel) = kpolaris::max_frame_error(kpolaris::frame_errors(metric, state));
            v.det_r(pixel) = overlap.det();
            v.overlap_r11(pixel) = overlap.r11;
            v.overlap_r12(pixel) = overlap.r12;
            v.overlap_r21(pixel) = overlap.r21;
            v.overlap_r22(pixel) = overlap.r22;
            v.basis_identity_error(pixel) =
                kpolaris::max_val(kpolaris::max_val(kpolaris::abs_val(overlap.r11 - RealT(1)),
                                                    kpolaris::abs_val(overlap.r22 - RealT(1))),
                                  kpolaris::max_val(kpolaris::abs_val(overlap.r12),
                                                    kpolaris::abs_val(overlap.r21)));
            v.basis_rotation_angle(pixel) = Kokkos::atan2(overlap.r21 - overlap.r12,
                                                          overlap.r11 + overlap.r22);
#if KPOLARIS_ENABLE_ANALYSIS_MODE
            if (analysis_mode) {
                finalize_slow_light_analysis_result(
                    analysis_result, overlap, observed, v, pixel,
                    analysis_config);
            }
#endif
            v.reason(pixel) = static_cast<int>(reason);
            v.active(pixel) = 0;
        });
    Kokkos::fence();
    report_image_timing(timing, "slow_light_finalize_vacuum", timer.seconds());
}

template<class ExecSpace, class RealT>
ImageHostData copy_slow_light_views_to_image_host_data(
    const SlowLightViews<ExecSpace, RealT>& views,
    int npix,
    int timing,
    int analysis_mode = 0,
    const kpolaris::AnalysisConfig<RealT>& analysis_config = {}) {
    Kokkos::Timer copy_timer;
    ImageHostData out;
    out.npix = npix;
    out.nfreq = 1;
    out.image_i = copy_real_view_1d(views.image_i);
    out.image_q = copy_real_view_1d(views.image_q);
    out.image_u = copy_real_view_1d(views.image_u);
    out.image_v = copy_real_view_1d(views.image_v);
    out.closure_x = copy_real_view_1d(views.closure_x);
    out.closure_k = copy_real_view_1d(views.closure_k);
    out.final_null = copy_real_view_1d(views.final_null);
    out.frame_error = copy_real_view_1d(views.frame_error);
    out.det_r = copy_real_view_1d(views.det_r);
    out.overlap_r11 = copy_real_view_1d(views.overlap_r11);
    out.overlap_r12 = copy_real_view_1d(views.overlap_r12);
    out.overlap_r21 = copy_real_view_1d(views.overlap_r21);
    out.overlap_r22 = copy_real_view_1d(views.overlap_r22);
    out.basis_identity_error = copy_real_view_1d(views.basis_identity_error);
    out.basis_rotation_angle = copy_real_view_1d(views.basis_rotation_angle);
    out.pass_a_steps = copy_int_view_1d(views.pass_a_steps);
    out.steps = copy_int_view_1d(views.steps);
    out.reason = copy_int_view_1d(views.reason);
#if KPOLARIS_ENABLE_ANALYSIS_MODE
    if (analysis_mode) {
        out.has_analysis = 1;
        out.radiating_path_length = copy_real_view_1d(views.radiating_path_length);
        out.emission_weight = copy_real_view_1d(views.emission_weight);
        out.emission_weighted_radius = copy_real_view_1d(views.emission_weighted_radius);
        out.emission_weighted_optical_depth_to_camera = copy_real_view_1d(views.emission_weighted_optical_depth_to_camera);
        out.absorption_depth = copy_real_view_1d(views.absorption_depth);
        out.absorption_operator_depth = copy_real_view_1d(views.absorption_operator_depth);
        out.faraday_rotation_depth = copy_real_view_1d(views.faraday_rotation_depth);
        out.faraday_conversion_depth = copy_real_view_1d(views.faraday_conversion_depth);
        out.faraday_operator_depth = copy_real_view_1d(views.faraday_operator_depth);
        out.dominant_emission_radius = copy_real_view_1d(views.dominant_emission_radius);
        out.dominant_emission_region = copy_int_view_1d(views.dominant_emission_region);
        out.dominant_ne_cgs = copy_real_view_1d(views.dominant_ne_cgs);
        out.dominant_thetae = copy_real_view_1d(views.dominant_thetae);
        out.dominant_b_cgs = copy_real_view_1d(views.dominant_b_cgs);
        out.dominant_beta = copy_real_view_1d(views.dominant_beta);
        out.dominant_sigma = copy_real_view_1d(views.dominant_sigma);
        out.emission_weighted_ne_cgs = copy_real_view_1d(views.emission_weighted_ne_cgs);
        out.emission_weighted_thetae = copy_real_view_1d(views.emission_weighted_thetae);
        out.emission_weighted_b_cgs = copy_real_view_1d(views.emission_weighted_b_cgs);
        out.emission_weighted_beta = copy_real_view_1d(views.emission_weighted_beta);
        out.emission_weighted_sigma = copy_real_view_1d(views.emission_weighted_sigma);
        out.photon_ring_winding_estimate = copy_real_view_1d(views.photon_ring_winding_estimate);
        out.radiation_substeps = copy_int_view_1d(views.radiation_substeps);
        out.response_config = analysis_config.response;
        if (analysis_config.response.enabled()) out.response_data = copy_real_view_1d(views.response_data);
        out.analysis_radial_bins = analysis_config.radial_bins;
        out.analysis_radial_min = analysis_config.radial_min;
        out.analysis_radial_max = analysis_config.radial_max;
        out.analysis_formation_fraction = analysis_config.formation_fraction;
        out.analysis_radial_bin_edges.resize(
            static_cast<size_t>(analysis_config.radial_bins) + 1);
        out.analysis_radial_bin_centers.resize(
            static_cast<size_t>(analysis_config.radial_bins));
        const RealT log_min = Kokkos::log(analysis_config.radial_min);
        const RealT log_max = Kokkos::log(analysis_config.radial_max);
        for (int bin = 0; bin <= analysis_config.radial_bins; ++bin) {
            out.analysis_radial_bin_edges[static_cast<size_t>(bin)] =
                Kokkos::exp(log_min + RealT(bin) / RealT(analysis_config.radial_bins) *
                            (log_max - log_min));
            if (bin < analysis_config.radial_bins) {
                out.analysis_radial_bin_centers[static_cast<size_t>(bin)] =
                    kpolaris::analysis_radial_bin_center(analysis_config, bin);
            }
        }
        out.radial_stokes_i_contribution = copy_real_view_1d(views.radial_stokes_i);
        out.radial_stokes_q_contribution = copy_real_view_1d(views.radial_stokes_q);
        out.radial_stokes_u_contribution = copy_real_view_1d(views.radial_stokes_u);
        out.radial_stokes_v_contribution = copy_real_view_1d(views.radial_stokes_v);
        out.radial_absorption_depth = copy_real_view_1d(views.radial_absorption_depth);
        out.radial_faraday_rotation_depth = copy_real_view_1d(views.radial_faraday_rotation_depth);
        out.radial_faraday_conversion_depth = copy_real_view_1d(views.radial_faraday_conversion_depth);
        out.radial_faraday_operator_depth = copy_real_view_1d(views.radial_faraday_operator_depth);

        out.observer_weighted_radius_i.resize(npix);
        out.observer_weighted_radius_linear.resize(npix);
        out.observer_weighted_radius_circular.resize(npix);
        out.intensity_formation_radius_low.resize(npix);
        out.intensity_formation_radius_median.resize(npix);
        out.intensity_formation_radius_high.resize(npix);
        out.linear_formation_radius_low.resize(npix);
        out.linear_formation_radius_median.resize(npix);
        out.linear_formation_radius_high.resize(npix);
        out.circular_formation_radius_low.resize(npix);
        out.circular_formation_radius_median.resize(npix);
        out.circular_formation_radius_high.resize(npix);
        out.los_linear_coherence.resize(npix);
        out.los_circular_coherence.resize(npix);
        out.contribution_closure_max_abs.resize(npix);
        out.contribution_closure_relative_l1.resize(npix);
        out.foreground_absorption_depth.resize(npix);
        out.foreground_faraday_rotation_depth.resize(npix);
        out.foreground_faraday_conversion_depth.resize(npix);
        out.foreground_faraday_operator_depth.resize(npix);
        out.foreground_faraday_operator_fraction.resize(npix);
        for (int pixel = 0; pixel < npix; ++pixel) {
            kpolaris::PassBAnalysisResult<RealT> result;
            result.transport.observed_stokes = kpolaris::Stokes<RealT>(
                out.image_i[static_cast<size_t>(pixel)],
                out.image_q[static_cast<size_t>(pixel)],
                out.image_u[static_cast<size_t>(pixel)],
                out.image_v[static_cast<size_t>(pixel)]);
            result.analysis.faraday_operator_depth =
                out.faraday_operator_depth[static_cast<size_t>(pixel)];
            for (int bin = 0; bin < analysis_config.radial_bins; ++bin) {
                const size_t index = static_cast<size_t>(bin) *
                                     static_cast<size_t>(npix) +
                                     static_cast<size_t>(pixel);
                result.observer_radial_stokes[bin] = kpolaris::Stokes<RealT>(
                    out.radial_stokes_i_contribution[index],
                    out.radial_stokes_q_contribution[index],
                    out.radial_stokes_u_contribution[index],
                    out.radial_stokes_v_contribution[index]);
                result.analysis.radial_absorption_depth[bin] =
                    out.radial_absorption_depth[index];
                result.analysis.radial_faraday_rotation_depth[bin] =
                    out.radial_faraday_rotation_depth[index];
                result.analysis.radial_faraday_conversion_depth[bin] =
                    out.radial_faraday_conversion_depth[index];
                result.analysis.radial_faraday_operator_depth[bin] =
                    out.radial_faraday_operator_depth[index];
            }
            kpolaris::finalize_observer_weighted_analysis(result, analysis_config);
            const auto& a = result.analysis;
            const size_t p = static_cast<size_t>(pixel);
            out.observer_weighted_radius_i[p] = a.observer_weighted_radius_i;
            out.observer_weighted_radius_linear[p] = a.observer_weighted_radius_linear;
            out.observer_weighted_radius_circular[p] = a.observer_weighted_radius_circular;
            out.intensity_formation_radius_low[p] = a.intensity_formation_radius_low;
            out.intensity_formation_radius_median[p] = a.intensity_formation_radius_median;
            out.intensity_formation_radius_high[p] = a.intensity_formation_radius_high;
            out.linear_formation_radius_low[p] = a.linear_formation_radius_low;
            out.linear_formation_radius_median[p] = a.linear_formation_radius_median;
            out.linear_formation_radius_high[p] = a.linear_formation_radius_high;
            out.circular_formation_radius_low[p] = a.circular_formation_radius_low;
            out.circular_formation_radius_median[p] = a.circular_formation_radius_median;
            out.circular_formation_radius_high[p] = a.circular_formation_radius_high;
            out.los_linear_coherence[p] = a.los_linear_coherence;
            out.los_circular_coherence[p] = a.los_circular_coherence;
            out.contribution_closure_max_abs[p] = a.contribution_closure_max_abs;
            out.contribution_closure_relative_l1[p] = a.contribution_closure_relative_l1;
            out.foreground_absorption_depth[p] = a.foreground_absorption_depth;
            out.foreground_faraday_rotation_depth[p] = a.foreground_faraday_rotation_depth;
            out.foreground_faraday_conversion_depth[p] = a.foreground_faraday_conversion_depth;
            out.foreground_faraday_operator_depth[p] = a.foreground_faraday_operator_depth;
            out.foreground_faraday_operator_fraction[p] = a.foreground_faraday_operator_fraction;
        }
    }
#else
    (void)analysis_mode;
    (void)analysis_config;
#endif
    report_image_timing(timing, "slow_light_device_to_host", copy_timer.seconds());
    return out;
}

template<class Metric, class RadiationModel, class LoadModel>
ImageHostData run_slow_light_image_metric(
    const kpolaris::PassAParams<Real>& pass_a,
    const std::vector<std::string>& dump_paths,
    const std::vector<Real>& dump_times,
    Real observation_time,
    RadiationModel lower,
    LoadModel load_model,
    int timing,
    const Metric& metric,
    const char* timing_suffix,
    int analysis_mode = 0,
    int prefetch = 1,
    kpolaris::AnalysisConfig<Real> analysis_config = {},
    kpolaris::SlowLightInterpolation interpolation = kpolaris::default_slow_light_interpolation) {
    if (dump_paths.size() != dump_times.size()) {
        throw std::runtime_error("slow-light dump path/time list size mismatch");
    }
    if (dump_paths.size() < 2) {
        throw std::runtime_error("slow-light requires at least two dump time slices");
    }
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    const int npix = pass_a.camera.nx * pass_a.camera.ny;
    auto views = allocate_slow_light_views<ExecSpace, Real>(
        npix, analysis_mode, analysis_config);
    initialize_slow_light_states_metric<ExecSpace>(
        pass_a, metric, views, timing, analysis_mode, analysis_config);

    using LoadTraits = SlowLightLoaderTraits<LoadModel, RadiationModel>;
    RadiationModel upper = LoadTraits::load(load_model, dump_paths[1]);
    int active = npix;
    for (size_t i = 0; i + 1 < dump_paths.size(); ++i) {
        const bool start_prefetch = prefetch && active > 0 && i + 2 < dump_paths.size();
        std::future<typename LoadTraits::PrefetchResult> next_upper_future;
        if (start_prefetch) {
            report_slow_light_prefetch_dump(timing, i + 2);
            next_upper_future = launch_slow_light_prefetch<LoadTraits>(load_model, dump_paths[i + 2]);
        }

        SlowLightTemporalModel<Real, RadiationModel> temporal;
        temporal.lower = lower;
        temporal.upper = upper;
        temporal.lower_time = dump_times[i];
        temporal.upper_time = dump_times[i + 1];
        temporal.observation_time = observation_time;
        temporal.freq_cgs = lower.freq_cgs;
        temporal.interpolation = interpolation;
        validate_slow_light_interpolation(lower, upper, interpolation);
        if (timing) {
            std::cout << "timing slow_light_window_index " << i
                      << " t0 " << static_cast<double>(dump_times[i])
                      << " t1 " << static_cast<double>(dump_times[i + 1]) << "\n";
        }
        run_slow_light_window_metric<ExecSpace>(pass_a, temporal, metric, views,
                                                dump_times[i + 1], timing,
                                                analysis_mode, analysis_config);
        active = count_active_rays<ExecSpace, Real>(views.active);
        if (timing) {
            std::cout << "timing slow_light_active_rays " << active << "\n";
        }
        lower = std::move(upper);
        if (active == 0) {
            if (start_prefetch) {
                Kokkos::Timer wait_timer;
                next_upper_future.wait();
                report_slow_light_prefetch_wait(timing, "slow_light_prefetch_discard_wait", wait_timer.seconds());
            }
            break;
        }
        if (i + 2 < dump_paths.size()) {
            if (start_prefetch) {
                Kokkos::Timer wait_timer;
                upper = LoadTraits::materialize(load_model, next_upper_future.get());
                report_slow_light_prefetch_wait(timing, "slow_light_prefetch_wait", wait_timer.seconds());
            } else {
                upper = LoadTraits::load(load_model, dump_paths[i + 2]);
            }
        }
    }
    finalize_slow_light_unfinished_metric<ExecSpace>(
        pass_a, metric, lower, views, timing, analysis_mode, analysis_config);
    (void)timing_suffix;

    return copy_slow_light_views_to_image_host_data<ExecSpace>(
        views, npix, timing, analysis_mode, analysis_config);
}


} // namespace kpolaris_image_detail
