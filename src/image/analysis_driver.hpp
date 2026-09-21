#pragma once

#include <iostream>
#include <vector>

#include <Kokkos_Core.hpp>

#include "diagnostics/plasma.hpp"
#include "diagnostics/response.hpp"
#include "geodesic/pass_b.hpp"
#include "image/driver.hpp"
#include "image/result.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct AnalysisConfig {
    ResponseConfig<Real> response;
    int radial_bins = 16;
    Real radial_min = Real(2);
    Real radial_max = Real(100);
    Real formation_fraction = Real(0.9);
};

template<class Real = DefaultReal>
struct PassBAnalysisDiagnostics {
    Real radiating_path_length = Real(0);
    Real emission_weight = Real(0);
    Real emission_weighted_radius = Real(0);
    Real emission_weighted_optical_depth_to_camera = Real(0);
    Real absorption_depth = Real(0);
    Real absorption_operator_depth = Real(0);
    Real faraday_rotation_depth = Real(0);
    Real faraday_conversion_depth = Real(0);
    Real faraday_operator_depth = Real(0);
    Real dominant_emission_radius = Real(0);
    Real dominant_emission_weight = Real(0);
    Real dominant_ne_cgs = Real(0);
    Real dominant_thetae = Real(0);
    Real dominant_b_cgs = Real(0);
    Real dominant_beta = Real(0);
    Real dominant_sigma = Real(0);
    Real emission_weighted_ne_cgs = Real(0);
    Real emission_weighted_thetae = Real(0);
    Real emission_weighted_b_cgs = Real(0);
    Real emission_weighted_beta = Real(0);
    Real emission_weighted_sigma = Real(0);
    Real photon_ring_winding_estimate = Real(0);
    Real previous_phi = Real(0);
    int has_previous_phi = 0;
    int radiation_substeps = 0;
    int dominant_emission_region = 0;
    Real radial_absorption_depth[KPOLARIS_MAX_ANALYSIS_RADIAL_BINS]{};
    Real radial_faraday_rotation_depth[KPOLARIS_MAX_ANALYSIS_RADIAL_BINS]{};
    Real radial_faraday_conversion_depth[KPOLARIS_MAX_ANALYSIS_RADIAL_BINS]{};
    Real radial_faraday_operator_depth[KPOLARIS_MAX_ANALYSIS_RADIAL_BINS]{};
    Real observer_weighted_radius_i = Real(0);
    Real observer_weighted_radius_linear = Real(0);
    Real observer_weighted_radius_circular = Real(0);
    Real intensity_formation_radius_low = Real(0);
    Real intensity_formation_radius_median = Real(0);
    Real intensity_formation_radius_high = Real(0);
    Real linear_formation_radius_low = Real(0);
    Real linear_formation_radius_median = Real(0);
    Real linear_formation_radius_high = Real(0);
    Real circular_formation_radius_low = Real(0);
    Real circular_formation_radius_median = Real(0);
    Real circular_formation_radius_high = Real(0);
    Real los_linear_coherence = Real(0);
    Real los_circular_coherence = Real(0);
    Real contribution_closure_max_abs = Real(0);
    Real contribution_closure_relative_l1 = Real(0);
    Real foreground_absorption_depth = Real(0);
    Real foreground_faraday_rotation_depth = Real(0);
    Real foreground_faraday_conversion_depth = Real(0);
    Real foreground_faraday_operator_depth = Real(0);
    Real foreground_faraday_operator_fraction = Real(0);
};

template<class Real = DefaultReal>
struct PassBAnalysisResult {
    PassBResult<Real> transport;
    PassBAnalysisDiagnostics<Real> analysis;
    Stokes<Real> observer_radial_stokes[KPOLARIS_MAX_ANALYSIS_RADIAL_BINS]{};
};

template<class Real>
KPOLARIS_INLINE int analysis_radial_bin(const AnalysisConfig<Real>& config, Real r) {
    if (config.radial_bins <= 1 || !(config.radial_max > config.radial_min) ||
        !(config.radial_min > Real(0)) || !(r > config.radial_min)) {
        return 0;
    }
    if (r >= config.radial_max) {
        return config.radial_bins - 1;
    }
    const Real fraction = Kokkos::log(r / config.radial_min) /
                          Kokkos::log(config.radial_max / config.radial_min);
    int bin = static_cast<int>(fraction * Real(config.radial_bins));
    if (bin < 0) bin = 0;
    if (bin >= config.radial_bins) bin = config.radial_bins - 1;
    return bin;
}

template<class Real>
KPOLARIS_INLINE Real analysis_radial_bin_center(const AnalysisConfig<Real>& config,
                                                int bin) {
    if (config.radial_bins <= 0 || !(config.radial_max > config.radial_min) ||
        !(config.radial_min > Real(0))) {
        return Real(0);
    }
    const Real log_min = Kokkos::log(config.radial_min);
    const Real log_max = Kokkos::log(config.radial_max);
    return Kokkos::exp(log_min + (Real(bin) + Real(0.5)) /
                        Real(config.radial_bins) * (log_max - log_min));
}

template<class Real>
KPOLARIS_INLINE TransferCoeffs<Real> transfer_without_emission(
    const TransferCoeffs<Real>& coeffs) {
    TransferCoeffs<Real> propagation = coeffs;
    propagation.jI = Real(0);
    propagation.jQ = Real(0);
    propagation.jU = Real(0);
    propagation.jV = Real(0);
    return propagation;
}

template<class Real>
KPOLARIS_INLINE Real radial_contribution_weight(const Stokes<Real>& stokes,
                                                int component) {
    if (component == 0) {
        return max_val(stokes.I, Real(0));
    }
    if (component == 1) {
        return Kokkos::sqrt(square(stokes.Q) + square(stokes.U));
    }
    return abs_val(stokes.V);
}

template<class Real>
KPOLARIS_INLINE Real radial_formation_quantile(
    const PassBAnalysisResult<Real>& out,
    const AnalysisConfig<Real>& config,
    int component,
    Real quantile) {
    Real total = Real(0);
    for (int bin = 0; bin < config.radial_bins; ++bin) {
        total += radial_contribution_weight(out.observer_radial_stokes[bin], component);
    }
    if (!(total > Real(0))) {
        return Real(0);
    }
    const Real target = min_val(Real(1), max_val(Real(0), quantile)) * total;
    Real cumulative = Real(0);
    for (int bin = 0; bin < config.radial_bins; ++bin) {
        cumulative += radial_contribution_weight(out.observer_radial_stokes[bin], component);
        if (cumulative >= target) {
            return analysis_radial_bin_center(config, bin);
        }
    }
    return analysis_radial_bin_center(config, config.radial_bins - 1);
}

template<class Real>
KPOLARIS_INLINE Real radial_weighted_radius(
    const PassBAnalysisResult<Real>& out,
    const AnalysisConfig<Real>& config,
    int component) {
    Real weighted = Real(0);
    Real total = Real(0);
    for (int bin = 0; bin < config.radial_bins; ++bin) {
        const Real weight = radial_contribution_weight(out.observer_radial_stokes[bin], component);
        weighted += weight * analysis_radial_bin_center(config, bin);
        total += weight;
    }
    return total > Real(0) ? weighted / total : Real(0);
}

template<class Real>
KPOLARIS_INLINE void finalize_observer_weighted_analysis(
    PassBAnalysisResult<Real>& out,
    const AnalysisConfig<Real>& config) {
    Stokes<Real> sum;
    Real sum_abs_i = Real(0), sum_linear = Real(0), sum_abs_v = Real(0);
    Real contribution_l1 = Real(0);
    for (int bin = 0; bin < config.radial_bins; ++bin) {
        const Stokes<Real>& radial = out.observer_radial_stokes[bin];
        sum.I += radial.I;
        sum.Q += radial.Q;
        sum.U += radial.U;
        sum.V += radial.V;
        sum_abs_i += abs_val(radial.I);
        sum_linear += Kokkos::sqrt(square(radial.Q) + square(radial.U));
        sum_abs_v += abs_val(radial.V);
        contribution_l1 += abs_val(radial.I) + abs_val(radial.Q) +
                           abs_val(radial.U) + abs_val(radial.V);
    }

    const Stokes<Real>& observed = out.transport.observed_stokes;
    const Real d_i = sum.I - observed.I;
    const Real d_q = sum.Q - observed.Q;
    const Real d_u = sum.U - observed.U;
    const Real d_v = sum.V - observed.V;
    out.analysis.contribution_closure_max_abs =
        max_val(max_val(abs_val(d_i), abs_val(d_q)),
                max_val(abs_val(d_u), abs_val(d_v)));
    const Real residual_l1 = abs_val(d_i) + abs_val(d_q) + abs_val(d_u) + abs_val(d_v);
    out.analysis.contribution_closure_relative_l1 =
        residual_l1 / max_val(contribution_l1, tiny_positive<Real>());

    out.analysis.los_linear_coherence = sum_linear > Real(0) ?
        Kokkos::sqrt(square(sum.Q) + square(sum.U)) / sum_linear : Real(0);
    out.analysis.los_circular_coherence = sum_abs_v > Real(0) ?
        abs_val(sum.V) / sum_abs_v : Real(0);
    (void)sum_abs_i;

    out.analysis.observer_weighted_radius_i = radial_weighted_radius(out, config, 0);
    out.analysis.observer_weighted_radius_linear = radial_weighted_radius(out, config, 1);
    out.analysis.observer_weighted_radius_circular = radial_weighted_radius(out, config, 2);
    const Real tail = Real(0.5) * (Real(1) - config.formation_fraction);
    out.analysis.intensity_formation_radius_low =
        radial_formation_quantile(out, config, 0, tail);
    out.analysis.intensity_formation_radius_median =
        radial_formation_quantile(out, config, 0, Real(0.5));
    out.analysis.intensity_formation_radius_high =
        radial_formation_quantile(out, config, 0, Real(1) - tail);
    out.analysis.linear_formation_radius_low =
        radial_formation_quantile(out, config, 1, tail);
    out.analysis.linear_formation_radius_median =
        radial_formation_quantile(out, config, 1, Real(0.5));
    out.analysis.linear_formation_radius_high =
        radial_formation_quantile(out, config, 1, Real(1) - tail);
    out.analysis.circular_formation_radius_low =
        radial_formation_quantile(out, config, 2, tail);
    out.analysis.circular_formation_radius_median =
        radial_formation_quantile(out, config, 2, Real(0.5));
    out.analysis.circular_formation_radius_high =
        radial_formation_quantile(out, config, 2, Real(1) - tail);

    const Real formation_outer = out.analysis.intensity_formation_radius_high;
    if (formation_outer > Real(0)) {
        for (int bin = 0; bin < config.radial_bins; ++bin) {
            if (analysis_radial_bin_center(config, bin) > formation_outer) {
                out.analysis.foreground_absorption_depth +=
                    out.analysis.radial_absorption_depth[bin];
                out.analysis.foreground_faraday_rotation_depth +=
                    out.analysis.radial_faraday_rotation_depth[bin];
                out.analysis.foreground_faraday_conversion_depth +=
                    out.analysis.radial_faraday_conversion_depth[bin];
                out.analysis.foreground_faraday_operator_depth +=
                    out.analysis.radial_faraday_operator_depth[bin];
            }
        }
    }
    if (out.analysis.faraday_operator_depth > Real(0)) {
        out.analysis.foreground_faraday_operator_fraction =
            out.analysis.foreground_faraday_operator_depth /
            out.analysis.faraday_operator_depth;
    }
}

template<class Real>
KPOLARIS_INLINE int dominant_emission_region_code(Real r) {
    if (!(r > Real(0))) {
        return 0;
    }
    if (r < Real(5)) {
        return 1;
    }
    if (r < Real(20)) {
        return 2;
    }
    return 3;
}

template<class Real>
KPOLARIS_INLINE Real wrapped_delta_phi(Real phi, Real previous_phi) {
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real two_pi = Real(6.283185307179586476925286766559005768);
    Real dphi = phi - previous_phi;
    if (dphi > pi) {
        dphi -= two_pi;
    } else if (dphi < -pi) {
        dphi += two_pi;
    }
    return dphi;
}

template<class Metric, class Real, class RadiationModel>
KPOLARIS_INLINE void accumulate_analysis_phi(const Metric& metric,
                                             const RadiationModel& radiation_model,
                                             const TransportState<Real>& sample_state,
                                             PassBAnalysisDiagnostics<Real>& analysis) {
    Real r = Real(0), th = Real(0), cp = Real(1), sp = Real(0);
    radiation_model.bl_coordinates_for_metric(metric, sample_state.x, r, th, cp, sp);
    (void)r;
    (void)th;
    Real phi = Kokkos::atan2(sp, cp);
    if (phi < Real(0)) {
        phi += Real(6.283185307179586476925286766559005768);
    }
    if (analysis.has_previous_phi) {
        analysis.photon_ring_winding_estimate += abs_val(wrapped_delta_phi(phi, analysis.previous_phi)) /
                                        Real(6.283185307179586476925286766559005768);
    }
    analysis.previous_phi = phi;
    analysis.has_previous_phi = 1;
}

template<class Metric, class Real, class RadiationModel>
KPOLARIS_INLINE void accumulate_physical_analysis_sample(
    const Metric& metric,
    const RadiationModel& radiation_model,
    const TransportState<Real>& sample_state,
    const TransferCoeffs<Real>& coeffs,
    Real dlambda,
    Real r,
    int radial_bin,
    PassBAnalysisDiagnostics<Real>& analysis) {
    const Real dl = abs_val(dlambda);
    const Real absorption_delta = max_val(coeffs.aI, Real(0)) * dl;
    const Real absorption_operator_delta = absorption_operator_rate(coeffs) * dl;
    const Real faraday_rotation_delta = coeffs.rV * dl;
    const Real faraday_conversion_delta = Kokkos::sqrt(square(coeffs.rQ) + square(coeffs.rU)) * dl;
    const Real faraday_operator_delta = faraday_operator_rate(coeffs) * dl;
    const Real tau_from_source_mid = analysis.absorption_depth + Real(0.5) * absorption_delta;
    const Real weight = max_val(coeffs.jI, Real(0)) * dl;

    analysis.radiating_path_length += dl;
    if (weight > Real(0)) {
        analysis.emission_weight += weight;
        analysis.emission_weighted_radius += weight * r;
        analysis.emission_weighted_optical_depth_to_camera += weight * tau_from_source_mid;
        const AnalysisPlasmaDiagnostics<Real> plasma =
            analysis_plasma_diagnostics(metric, radiation_model, sample_state);
        if (weight > analysis.dominant_emission_weight) {
            analysis.dominant_emission_weight = weight;
            analysis.dominant_emission_radius = r;
            analysis.dominant_emission_region = dominant_emission_region_code(r);
            if (plasma.valid) {
                analysis.dominant_ne_cgs = plasma.ne_cgs;
                analysis.dominant_thetae = plasma.thetae;
                analysis.dominant_b_cgs = plasma.b_cgs;
                analysis.dominant_beta = plasma.beta;
                analysis.dominant_sigma = plasma.sigma;
            }
        }
        if (plasma.valid) {
            analysis.emission_weighted_ne_cgs += weight * plasma.ne_cgs;
            analysis.emission_weighted_thetae += weight * plasma.thetae;
            analysis.emission_weighted_b_cgs += weight * plasma.b_cgs;
            analysis.emission_weighted_beta += weight * plasma.beta;
            analysis.emission_weighted_sigma += weight * plasma.sigma;
        }
    }

    analysis.absorption_depth += absorption_delta;
    analysis.absorption_operator_depth += absorption_operator_delta;
    analysis.faraday_rotation_depth += faraday_rotation_delta;
    analysis.faraday_conversion_depth += faraday_conversion_delta;
    analysis.faraday_operator_depth += faraday_operator_delta;
    analysis.radial_absorption_depth[radial_bin] += absorption_delta;
    analysis.radial_faraday_rotation_depth[radial_bin] += faraday_rotation_delta;
    analysis.radial_faraday_conversion_depth[radial_bin] += faraday_conversion_delta;
    analysis.radial_faraday_operator_depth[radial_bin] += faraday_operator_delta;
}

// Compatibility entry point for analysis paths that have not requested a
// radial decomposition (notably the slow-light driver).  Their scalar LOS
// moments retain the previous behavior while all depth is assigned to bin 0.
template<class Metric, class Real, class RadiationModel>
KPOLARIS_INLINE void accumulate_physical_analysis_sample(
    const Metric& metric,
    const RadiationModel& radiation_model,
    const TransportState<Real>& sample_state,
    const TransferCoeffs<Real>& coeffs,
    Real dlambda,
    PassBAnalysisDiagnostics<Real>& analysis) {
    Real r = Real(0), th = Real(0), cp = Real(1), sp = Real(0);
    radiation_model.bl_coordinates_for_metric(metric, sample_state.x, r, th, cp, sp);
    (void)th;
    (void)cp;
    (void)sp;
    accumulate_physical_analysis_sample(
        metric, radiation_model, sample_state, coeffs, dlambda, r, 0, analysis);
}

template<class Real>
KPOLARIS_INLINE void finalize_analysis(PassBAnalysisDiagnostics<Real>& analysis) {
    if (analysis.emission_weight > Real(0)) {
        const Real inv_weight = Real(1) / analysis.emission_weight;
        analysis.emission_weighted_radius *= inv_weight;
        const Real mean_tau_from_source = analysis.emission_weighted_optical_depth_to_camera * inv_weight;
        analysis.emission_weighted_optical_depth_to_camera =
            max_val(Real(0), analysis.absorption_depth - mean_tau_from_source);
        analysis.emission_weighted_ne_cgs *= inv_weight;
        analysis.emission_weighted_thetae *= inv_weight;
        analysis.emission_weighted_b_cgs *= inv_weight;
        analysis.emission_weighted_beta *= inv_weight;
        analysis.emission_weighted_sigma *= inv_weight;
    } else {
        analysis.emission_weighted_radius = Real(0);
        analysis.emission_weighted_optical_depth_to_camera = Real(0);
        analysis.dominant_emission_radius = Real(0);
        analysis.dominant_emission_region = 0;
    }
}

template<class Metric, class Real, class RadiationModel, class ResponseView = Kokkos::View<Real*>>
KPOLARIS_INLINE PassBAnalysisResult<Real> trace_pass_b_segment_model_endpoint_analysis_pixel_metric(
    int pixel,
    const PassAParams<Real>& params,
    const AnalysisConfig<Real>& analysis_config,
    const RadiationModel& radiation_model,
    int radiation_substeps,
    const TransportState<Real>& endpoint_state,
    int pass_a_steps,
    TerminationReason pass_a_reason,
    int endpoint_valid,
    const Metric& metric,
    const ResponseView& response_data = {}) {
    (void)radiation_substeps;
    PassBAnalysisResult<Real> out;
    PassBResult<Real>& transport = out.transport;
    const TransportState<Real> camera_state = initialize_camera_ray(metric, pixel, params.camera);

    transport.state = endpoint_state;
    transport.pass_a.state = endpoint_state;
    transport.pass_a.steps = pass_a_steps;
    transport.pass_a.reason = pass_a_reason;
    transport.pass_a.final_null = metric.dot(endpoint_state.x, endpoint_state.k, endpoint_state.k);
    transport.pass_a.frame_error = max_frame_error(frame_errors(metric, endpoint_state));
    transport.reason = pass_a_reason;

    if (!endpoint_valid) {
        transport.closure_x = spatial_distance(transport.state.x, camera_state.x);
        transport.closure_k = vector_max_abs_difference(transport.state.k, camera_state.k);
        transport.final_null = transport.pass_a.final_null;
        transport.frame_error = transport.pass_a.frame_error;
        return out;
    }

    const int screen_orientation = backward_screen_orientation_sign(params.camera);

    AdaptiveRK4Control<Real> transfer_control;
    transfer_control.tolerance = params.adaptive_tolerance;
    transfer_control.min_step = params.min_step;
    transfer_control.max_step = params.max_step;

    Stokes<Real> stokes;
    Stokes<Real> radial_stokes[KPOLARIS_MAX_ANALYSIS_RADIAL_BINS]{};
    const Real dlambda_scale = radiation_model.dlambda_scale();
    int reached_camera = 0;
    Real h_current = params.step;
    Real radiation_step_cap = params.max_radiation_step > Real(0) ? params.max_radiation_step : params.max_step;
    while (!reached_camera && transport.steps < params.max_steps) {
        Real h = h_current;
        const TransportState<Real> old_state = transport.state;
        const Real start_r_bl = radial_coordinate(metric, old_state.x);
        if (start_r_bl <= params.outer_radius && radiation_step_cap > Real(0)) {
            h = min_val(h, radiation_step_cap);
        }
        TransportState<Real> next_state;
        TransportState<Real> sample_state;
        if (params.adaptive) {
            const auto proposed = adaptive_rk4_step(metric, old_state, h, transfer_control);
            if (!proposed.accepted) {
                h_current = abs_val(proposed.next_h);
                if (h_current <= params.min_step * Real(1.0001)) {
                    transport.reason = TerminationReason::max_steps;
                    break;
                }
                continue;
            }
            h = abs_val(proposed.used_h);
            h_current = abs_val(proposed.next_h);
            next_state = proposed.state;
            sample_state = proposed.mid_state;
        } else {
            next_state = rk4_step(metric, old_state, h);
            sample_state = next_state;
        }

        const Real s0 = camera_surface_value(metric, params.camera, old_state);
        const Real s1 = camera_surface_value(metric, params.camera, next_state);
        int crosses_camera = camera_surface_crossed(s0, s1);
        if (crosses_camera) {
            const Real frac = camera_surface_crossing_fraction(s0, s1);
            const Real h_cross = h * frac;
            if (abs_val(h_cross) <= Real(1e-14)) {
                h = Real(0);
                next_state = old_state;
                sample_state = old_state;
            } else if (frac < Real(0.999999999999)) {
                h = h_cross;
                if (params.adaptive) {
                    AdaptiveRK4Control<Real> crossing_control = transfer_control;
                    crossing_control.min_step = min_val(crossing_control.min_step, abs_val(h));
                    crossing_control.max_step = max_val(crossing_control.min_step, abs_val(h));
                    const auto crossing_step = adaptive_rk4_step(metric, old_state, h, crossing_control);
                    if (!crossing_step.accepted) {
                        h_current = abs_val(crossing_step.next_h);
                        continue;
                    }
                    h = abs_val(crossing_step.used_h);
                    next_state = crossing_step.state;
                    sample_state = crossing_step.mid_state;
                } else {
                    next_state = rk4_step(metric, old_state, h);
                    sample_state = next_state;
                }
            }
        }

        if (params.emission_selection.narrow_wedge()) {
            if (!params.adaptive) sample_state = rk4_step(metric, old_state, h*Real(0.5));
            const int wedge_substeps = equatorial_resolution_substeps(params.emission_selection,
                metric, old_state, sample_state, next_state, params.outer_radius);
            if (wedge_substeps > 1) {
                if (h <= params.min_step*Real(1.0001)) {
                    transport.reason = TerminationReason::adaptive_step_underflow;
                    break;
                }
                h_current = max_val(params.min_step, h/Real(wedge_substeps));
                radiation_step_cap = h_current;
                continue;
            }
            if (refine_equatorial_boundary(params.emission_selection, metric, old_state,
                    sample_state, next_state, h, params.adaptive, params.outer_radius)) {
                crosses_camera = camera_surface_crossed(s0, camera_surface_value(metric, params.camera, next_state));
            }
        }

        const Real old_r_bl = radial_coordinate(metric, old_state.x);
        const Real sample_r_bl = radial_coordinate(metric, sample_state.x);
        const Real next_r_bl = radial_coordinate(metric, next_state.x);
        const int radiation_active =
            (sample_r_bl >= params.inner_radius && sample_r_bl < params.outer_radius) ? 1 : 0;
        TransferCoeffs<Real> coeffs;
        if (radiation_active) {
            coeffs = radiation_model.coefficients(metric, sample_state, Real(0.5));
            apply_emission_selection(params.emission_selection, metric, sample_state, coeffs);
            transform_axial_coefficients_to_screen_orientation(coeffs, screen_orientation);
        }
        const Real local_max_radiation_step =
            (params.max_radiation_step > Real(0) &&
             (old_r_bl <= params.outer_radius || sample_r_bl <= params.outer_radius ||
              next_r_bl <= params.outer_radius)) ? params.max_radiation_step : Real(0);
        const int required_steps = radiation_substep_count(
            coeffs, h, dlambda_scale, 1,
            local_max_radiation_step, params.max_radiation_depth,
            params.max_absorption_depth, params.max_faraday_depth);
        if (required_steps > 1 && h > params.min_step * Real(1.0001)) {
            h_current = max_val(params.min_step, h / Real(required_steps));
            radiation_step_cap = h_current;
            continue;
        }
        if (local_max_radiation_step > Real(0) || params.max_radiation_depth > Real(0) ||
            params.max_absorption_depth > Real(0) || params.max_faraday_depth > Real(0)) {
            const int predictive_steps = radiation_substep_count(
                coeffs, h_current, dlambda_scale, 1,
                local_max_radiation_step, params.max_radiation_depth,
                params.max_absorption_depth, params.max_faraday_depth);
            radiation_step_cap = predictive_steps > 1 ?
                max_val(params.min_step, h_current / Real(predictive_steps)) : params.max_step;
        } else {
            radiation_step_cap = params.max_step;
        }
        if (h > Real(0)) {
            accumulate_analysis_phi(metric, radiation_model, sample_state, out.analysis);
            if (radiation_active) {
                const Real dlambda = abs_val(h) * dlambda_scale;
                Real analysis_r = Real(0), analysis_th = Real(0);
                Real analysis_cp = Real(1), analysis_sp = Real(0);
                radiation_model.bl_coordinates_for_metric(
                    metric, sample_state.x, analysis_r, analysis_th,
                    analysis_cp, analysis_sp);
                (void)analysis_th;
                (void)analysis_cp;
                (void)analysis_sp;
                const int radial_bin = analysis_radial_bin(analysis_config, analysis_r);
                accumulate_physical_analysis_sample(metric, radiation_model, sample_state,
                                                    coeffs, dlambda, analysis_r,
                                                    radial_bin, out.analysis);
                accumulate_response_sample(metric, radiation_model, sample_state, stokes,
                    coeffs, dlambda, screen_orientation, analysis_config.response,
                    response_data, pixel, params.camera.nx * params.camera.ny, params.emission_selection);
                const TransferCoeffs<Real> propagation = transfer_without_emission(coeffs);
                for (int bin = 0; bin < analysis_config.radial_bins; ++bin) {
                    semi_analytic_stokes_step(
                        radial_stokes[bin], bin == radial_bin ? coeffs : propagation,
                        dlambda);
                }
                out.analysis.radiation_substeps += 1;
            }
            semi_analytic_stokes_step(stokes, coeffs, abs_val(h) * dlambda_scale);
        }
        transport.state = next_state;
        transport.steps += 1;
        if (crosses_camera) {
            reached_camera = 1;
        }
    }

    finalize_analysis(out.analysis);
    transport.propagated_stokes = stokes;
    transport.overlap = screen_overlap(metric, transport.state.x, camera_state, transport.state);
    transport.observed_stokes = transform_to_observer_basis(stokes, transport.overlap);
    for (int bin = 0; bin < analysis_config.radial_bins; ++bin) {
        out.observer_radial_stokes[bin] =
            transform_to_observer_basis(radial_stokes[bin], transport.overlap);
    }
    finalize_response_basis(analysis_config.response, response_data, pixel,
        params.camera.nx * params.camera.ny, transport.overlap);
    finalize_observer_weighted_analysis(out, analysis_config);
    transport.closure_x = spatial_distance(transport.state.x, camera_state.x);
    Vec4<Real> target_k;
    for (int mu = 0; mu < ndim; ++mu) {
        target_k[mu] = camera_state.k[mu];
    }
    transport.closure_k = vector_max_abs_difference(transport.state.k, target_k);
    transport.final_null = metric.dot(transport.state.x, transport.state.k, transport.state.k);
    transport.frame_error = max_frame_error(frame_errors(metric, transport.state));
    if (reached_camera) transport.reason = TerminationReason::reached_camera;
    else if (transport.reason != TerminationReason::adaptive_step_underflow)
        transport.reason = TerminationReason::max_steps;
    return out;
}

template<class Metric, class Real, class RadiationModel, class ResponseView = Kokkos::View<Real*>>
KPOLARIS_INLINE PassBAnalysisResult<Real> trace_pass_b_segment_model_analysis_pixel_metric(
    int pixel,
    const PassAParams<Real>& params,
    const AnalysisConfig<Real>& analysis_config,
    const RadiationModel& radiation_model,
    int radiation_substeps,
    const Metric& metric,
    const ResponseView& response_data = {}) {
    const PassAEndpointResult<Real> endpoint = trace_pass_a_segment_endpoint_pixel_metric(pixel, params, metric);
    return trace_pass_b_segment_model_endpoint_analysis_pixel_metric(
        pixel, params, analysis_config, radiation_model, radiation_substeps, endpoint.state,
        endpoint.steps, endpoint.reason, endpoint.valid, metric, response_data);
}

} // namespace kpolaris

namespace kpolaris_image_detail {

template<class RealT, class ExecSpace>
struct AnalysisImageViews {
    Kokkos::View<RealT*, ExecSpace> response_data;
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
    Kokkos::View<int*, ExecSpace> pass_a_steps;
    Kokkos::View<int*, ExecSpace> steps;
    Kokkos::View<int*, ExecSpace> reason;

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
    Kokkos::View<int*, ExecSpace> dominant_emission_region;
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
    Kokkos::View<int*, ExecSpace> radiation_substeps;
    Kokkos::View<RealT*, ExecSpace> radial_stokes_i_contribution;
    Kokkos::View<RealT*, ExecSpace> radial_stokes_q_contribution;
    Kokkos::View<RealT*, ExecSpace> radial_stokes_u_contribution;
    Kokkos::View<RealT*, ExecSpace> radial_stokes_v_contribution;
    Kokkos::View<RealT*, ExecSpace> radial_absorption_depth;
    Kokkos::View<RealT*, ExecSpace> radial_faraday_rotation_depth;
    Kokkos::View<RealT*, ExecSpace> radial_faraday_conversion_depth;
    Kokkos::View<RealT*, ExecSpace> radial_faraday_operator_depth;
    Kokkos::View<RealT*, ExecSpace> observer_weighted_radius_i;
    Kokkos::View<RealT*, ExecSpace> observer_weighted_radius_linear;
    Kokkos::View<RealT*, ExecSpace> observer_weighted_radius_circular;
    Kokkos::View<RealT*, ExecSpace> intensity_formation_radius_low;
    Kokkos::View<RealT*, ExecSpace> intensity_formation_radius_median;
    Kokkos::View<RealT*, ExecSpace> intensity_formation_radius_high;
    Kokkos::View<RealT*, ExecSpace> linear_formation_radius_low;
    Kokkos::View<RealT*, ExecSpace> linear_formation_radius_median;
    Kokkos::View<RealT*, ExecSpace> linear_formation_radius_high;
    Kokkos::View<RealT*, ExecSpace> circular_formation_radius_low;
    Kokkos::View<RealT*, ExecSpace> circular_formation_radius_median;
    Kokkos::View<RealT*, ExecSpace> circular_formation_radius_high;
    Kokkos::View<RealT*, ExecSpace> los_linear_coherence;
    Kokkos::View<RealT*, ExecSpace> los_circular_coherence;
    Kokkos::View<RealT*, ExecSpace> contribution_closure_max_abs;
    Kokkos::View<RealT*, ExecSpace> contribution_closure_relative_l1;
    Kokkos::View<RealT*, ExecSpace> foreground_absorption_depth;
    Kokkos::View<RealT*, ExecSpace> foreground_faraday_rotation_depth;
    Kokkos::View<RealT*, ExecSpace> foreground_faraday_conversion_depth;
    Kokkos::View<RealT*, ExecSpace> foreground_faraday_operator_depth;
    Kokkos::View<RealT*, ExecSpace> foreground_faraday_operator_fraction;
};

template<class RealT, class ExecSpace>
AnalysisImageViews<RealT, ExecSpace> make_analysis_image_views(
    int npix, const kpolaris::AnalysisConfig<RealT>& config) {
    AnalysisImageViews<RealT, ExecSpace> v;
    if (config.response.enabled())
        v.response_data = Kokkos::View<RealT*, ExecSpace>("analysis_response", size_t(4) * config.response.channels() * npix);
    v.image_i = Kokkos::View<RealT*, ExecSpace>("analysis_I", npix);
    v.image_q = Kokkos::View<RealT*, ExecSpace>("analysis_Q", npix);
    v.image_u = Kokkos::View<RealT*, ExecSpace>("analysis_U", npix);
    v.image_v = Kokkos::View<RealT*, ExecSpace>("analysis_V", npix);
    v.closure_x = Kokkos::View<RealT*, ExecSpace>("analysis_closure_x", npix);
    v.closure_k = Kokkos::View<RealT*, ExecSpace>("analysis_closure_k", npix);
    v.final_null = Kokkos::View<RealT*, ExecSpace>("analysis_final_null", npix);
    v.frame_error = Kokkos::View<RealT*, ExecSpace>("analysis_frame_error", npix);
    v.det_r = Kokkos::View<RealT*, ExecSpace>("analysis_det_r", npix);
    v.overlap_r11 = Kokkos::View<RealT*, ExecSpace>("analysis_overlap_r11", npix);
    v.overlap_r12 = Kokkos::View<RealT*, ExecSpace>("analysis_overlap_r12", npix);
    v.overlap_r21 = Kokkos::View<RealT*, ExecSpace>("analysis_overlap_r21", npix);
    v.overlap_r22 = Kokkos::View<RealT*, ExecSpace>("analysis_overlap_r22", npix);
    v.basis_identity_error = Kokkos::View<RealT*, ExecSpace>("analysis_basis_identity_error", npix);
    v.basis_rotation_angle = Kokkos::View<RealT*, ExecSpace>("analysis_basis_rotation_angle", npix);
    v.pass_a_steps = Kokkos::View<int*, ExecSpace>("analysis_pass_a_steps", npix);
    v.steps = Kokkos::View<int*, ExecSpace>("analysis_steps", npix);
    v.reason = Kokkos::View<int*, ExecSpace>("analysis_reason", npix);

    v.radiating_path_length = Kokkos::View<RealT*, ExecSpace>("analysis_radiating_path_length", npix);
    v.emission_weight = Kokkos::View<RealT*, ExecSpace>("analysis_emission_weight", npix);
    v.emission_weighted_radius = Kokkos::View<RealT*, ExecSpace>("analysis_emission_weighted_radius", npix);
    v.emission_weighted_optical_depth_to_camera = Kokkos::View<RealT*, ExecSpace>("analysis_emission_weighted_optical_depth_to_camera", npix);
    v.absorption_depth = Kokkos::View<RealT*, ExecSpace>("analysis_absorption_depth", npix);
    v.absorption_operator_depth = Kokkos::View<RealT*, ExecSpace>("analysis_absorption_operator_depth", npix);
    v.faraday_rotation_depth = Kokkos::View<RealT*, ExecSpace>("analysis_faraday_rotation_depth", npix);
    v.faraday_conversion_depth = Kokkos::View<RealT*, ExecSpace>("analysis_faraday_conversion_depth", npix);
    v.faraday_operator_depth = Kokkos::View<RealT*, ExecSpace>("analysis_faraday_operator_depth", npix);
    v.dominant_emission_radius = Kokkos::View<RealT*, ExecSpace>("analysis_dominant_emission_radius", npix);
    v.dominant_emission_region = Kokkos::View<int*, ExecSpace>("analysis_dominant_emission_region", npix);
    v.dominant_ne_cgs = Kokkos::View<RealT*, ExecSpace>("analysis_dominant_ne_cgs", npix);
    v.dominant_thetae = Kokkos::View<RealT*, ExecSpace>("analysis_dominant_thetae", npix);
    v.dominant_b_cgs = Kokkos::View<RealT*, ExecSpace>("analysis_dominant_b_cgs", npix);
    v.dominant_beta = Kokkos::View<RealT*, ExecSpace>("analysis_dominant_beta", npix);
    v.dominant_sigma = Kokkos::View<RealT*, ExecSpace>("analysis_dominant_sigma", npix);
    v.emission_weighted_ne_cgs = Kokkos::View<RealT*, ExecSpace>("analysis_emission_weighted_ne_cgs", npix);
    v.emission_weighted_thetae = Kokkos::View<RealT*, ExecSpace>("analysis_emission_weighted_thetae", npix);
    v.emission_weighted_b_cgs = Kokkos::View<RealT*, ExecSpace>("analysis_emission_weighted_b_cgs", npix);
    v.emission_weighted_beta = Kokkos::View<RealT*, ExecSpace>("analysis_emission_weighted_beta", npix);
    v.emission_weighted_sigma = Kokkos::View<RealT*, ExecSpace>("analysis_emission_weighted_sigma", npix);
    v.photon_ring_winding_estimate = Kokkos::View<RealT*, ExecSpace>("analysis_photon_ring_winding_estimate", npix);
    v.radiation_substeps = Kokkos::View<int*, ExecSpace>("analysis_radiation_substeps", npix);
    const int radial_values = npix * config.radial_bins;
    v.radial_stokes_i_contribution = Kokkos::View<RealT*, ExecSpace>("analysis_radial_stokes_i_contribution", radial_values);
    v.radial_stokes_q_contribution = Kokkos::View<RealT*, ExecSpace>("analysis_radial_stokes_q_contribution", radial_values);
    v.radial_stokes_u_contribution = Kokkos::View<RealT*, ExecSpace>("analysis_radial_stokes_u_contribution", radial_values);
    v.radial_stokes_v_contribution = Kokkos::View<RealT*, ExecSpace>("analysis_radial_stokes_v_contribution", radial_values);
    v.radial_absorption_depth = Kokkos::View<RealT*, ExecSpace>("analysis_radial_absorption_depth", radial_values);
    v.radial_faraday_rotation_depth = Kokkos::View<RealT*, ExecSpace>("analysis_radial_faraday_rotation_depth", radial_values);
    v.radial_faraday_conversion_depth = Kokkos::View<RealT*, ExecSpace>("analysis_radial_faraday_conversion_depth", radial_values);
    v.radial_faraday_operator_depth = Kokkos::View<RealT*, ExecSpace>("analysis_radial_faraday_operator_depth", radial_values);
    v.observer_weighted_radius_i = Kokkos::View<RealT*, ExecSpace>("analysis_observer_weighted_radius_i", npix);
    v.observer_weighted_radius_linear = Kokkos::View<RealT*, ExecSpace>("analysis_observer_weighted_radius_linear", npix);
    v.observer_weighted_radius_circular = Kokkos::View<RealT*, ExecSpace>("analysis_observer_weighted_radius_circular", npix);
    v.intensity_formation_radius_low = Kokkos::View<RealT*, ExecSpace>("analysis_intensity_formation_radius_low", npix);
    v.intensity_formation_radius_median = Kokkos::View<RealT*, ExecSpace>("analysis_intensity_formation_radius_median", npix);
    v.intensity_formation_radius_high = Kokkos::View<RealT*, ExecSpace>("analysis_intensity_formation_radius_high", npix);
    v.linear_formation_radius_low = Kokkos::View<RealT*, ExecSpace>("analysis_linear_formation_radius_low", npix);
    v.linear_formation_radius_median = Kokkos::View<RealT*, ExecSpace>("analysis_linear_formation_radius_median", npix);
    v.linear_formation_radius_high = Kokkos::View<RealT*, ExecSpace>("analysis_linear_formation_radius_high", npix);
    v.circular_formation_radius_low = Kokkos::View<RealT*, ExecSpace>("analysis_circular_formation_radius_low", npix);
    v.circular_formation_radius_median = Kokkos::View<RealT*, ExecSpace>("analysis_circular_formation_radius_median", npix);
    v.circular_formation_radius_high = Kokkos::View<RealT*, ExecSpace>("analysis_circular_formation_radius_high", npix);
    v.los_linear_coherence = Kokkos::View<RealT*, ExecSpace>("analysis_los_linear_coherence", npix);
    v.los_circular_coherence = Kokkos::View<RealT*, ExecSpace>("analysis_los_circular_coherence", npix);
    v.contribution_closure_max_abs = Kokkos::View<RealT*, ExecSpace>("analysis_contribution_closure_max_abs", npix);
    v.contribution_closure_relative_l1 = Kokkos::View<RealT*, ExecSpace>("analysis_contribution_closure_relative_l1", npix);
    v.foreground_absorption_depth = Kokkos::View<RealT*, ExecSpace>("analysis_foreground_absorption_depth", npix);
    v.foreground_faraday_rotation_depth = Kokkos::View<RealT*, ExecSpace>("analysis_foreground_faraday_rotation_depth", npix);
    v.foreground_faraday_conversion_depth = Kokkos::View<RealT*, ExecSpace>("analysis_foreground_faraday_conversion_depth", npix);
    v.foreground_faraday_operator_depth = Kokkos::View<RealT*, ExecSpace>("analysis_foreground_faraday_operator_depth", npix);
    v.foreground_faraday_operator_fraction = Kokkos::View<RealT*, ExecSpace>("analysis_foreground_faraday_operator_fraction", npix);
    return v;
}

template<class RealT, class ExecSpace>
KPOLARIS_INLINE void store_analysis_pixel_result(
    int pixel,
    const kpolaris::PassBAnalysisResult<RealT>& ar,
    const kpolaris::AnalysisConfig<RealT>& config,
    const AnalysisImageViews<RealT, ExecSpace>& v) {
    const kpolaris::PassBResult<RealT>& r = ar.transport;
    v.image_i(pixel) = r.observed_stokes.I;
    v.image_q(pixel) = r.observed_stokes.Q;
    v.image_u(pixel) = r.observed_stokes.U;
    v.image_v(pixel) = r.observed_stokes.V;
    v.closure_x(pixel) = r.closure_x;
    v.closure_k(pixel) = r.closure_k;
    v.final_null(pixel) = r.final_null;
    v.frame_error(pixel) = r.frame_error;
    v.det_r(pixel) = r.overlap.det();
    v.overlap_r11(pixel) = r.overlap.r11;
    v.overlap_r12(pixel) = r.overlap.r12;
    v.overlap_r21(pixel) = r.overlap.r21;
    v.overlap_r22(pixel) = r.overlap.r22;
    v.basis_identity_error(pixel) =
        kpolaris::max_val(kpolaris::max_val(kpolaris::abs_val(r.overlap.r11 - RealT(1)),
                                            kpolaris::abs_val(r.overlap.r22 - RealT(1))),
                          kpolaris::max_val(kpolaris::abs_val(r.overlap.r12),
                                            kpolaris::abs_val(r.overlap.r21)));
    v.basis_rotation_angle(pixel) = Kokkos::atan2(r.overlap.r21 - r.overlap.r12,
                                                  r.overlap.r11 + r.overlap.r22);
    v.pass_a_steps(pixel) = r.pass_a.steps;
    v.steps(pixel) = r.steps;
    v.reason(pixel) = static_cast<int>(r.reason);

    v.radiating_path_length(pixel) = ar.analysis.radiating_path_length;
    v.emission_weight(pixel) = ar.analysis.emission_weight;
    v.emission_weighted_radius(pixel) = ar.analysis.emission_weighted_radius;
    v.emission_weighted_optical_depth_to_camera(pixel) = ar.analysis.emission_weighted_optical_depth_to_camera;
    v.absorption_depth(pixel) = ar.analysis.absorption_depth;
    v.absorption_operator_depth(pixel) = ar.analysis.absorption_operator_depth;
    v.faraday_rotation_depth(pixel) = ar.analysis.faraday_rotation_depth;
    v.faraday_conversion_depth(pixel) = ar.analysis.faraday_conversion_depth;
    v.faraday_operator_depth(pixel) = ar.analysis.faraday_operator_depth;
    v.dominant_emission_radius(pixel) = ar.analysis.dominant_emission_radius;
    v.dominant_emission_region(pixel) = ar.analysis.dominant_emission_region;
    v.dominant_ne_cgs(pixel) = ar.analysis.dominant_ne_cgs;
    v.dominant_thetae(pixel) = ar.analysis.dominant_thetae;
    v.dominant_b_cgs(pixel) = ar.analysis.dominant_b_cgs;
    v.dominant_beta(pixel) = ar.analysis.dominant_beta;
    v.dominant_sigma(pixel) = ar.analysis.dominant_sigma;
    v.emission_weighted_ne_cgs(pixel) = ar.analysis.emission_weighted_ne_cgs;
    v.emission_weighted_thetae(pixel) = ar.analysis.emission_weighted_thetae;
    v.emission_weighted_b_cgs(pixel) = ar.analysis.emission_weighted_b_cgs;
    v.emission_weighted_beta(pixel) = ar.analysis.emission_weighted_beta;
    v.emission_weighted_sigma(pixel) = ar.analysis.emission_weighted_sigma;
    v.photon_ring_winding_estimate(pixel) = ar.analysis.photon_ring_winding_estimate;
    v.radiation_substeps(pixel) = ar.analysis.radiation_substeps;
    const int npix = v.image_i.extent_int(0);
    for (int bin = 0; bin < config.radial_bins; ++bin) {
        const int index = bin * npix + pixel;
        v.radial_stokes_i_contribution(index) = ar.observer_radial_stokes[bin].I;
        v.radial_stokes_q_contribution(index) = ar.observer_radial_stokes[bin].Q;
        v.radial_stokes_u_contribution(index) = ar.observer_radial_stokes[bin].U;
        v.radial_stokes_v_contribution(index) = ar.observer_radial_stokes[bin].V;
        v.radial_absorption_depth(index) = ar.analysis.radial_absorption_depth[bin];
        v.radial_faraday_rotation_depth(index) = ar.analysis.radial_faraday_rotation_depth[bin];
        v.radial_faraday_conversion_depth(index) = ar.analysis.radial_faraday_conversion_depth[bin];
        v.radial_faraday_operator_depth(index) = ar.analysis.radial_faraday_operator_depth[bin];
    }
    v.observer_weighted_radius_i(pixel) = ar.analysis.observer_weighted_radius_i;
    v.observer_weighted_radius_linear(pixel) = ar.analysis.observer_weighted_radius_linear;
    v.observer_weighted_radius_circular(pixel) = ar.analysis.observer_weighted_radius_circular;
    v.intensity_formation_radius_low(pixel) = ar.analysis.intensity_formation_radius_low;
    v.intensity_formation_radius_median(pixel) = ar.analysis.intensity_formation_radius_median;
    v.intensity_formation_radius_high(pixel) = ar.analysis.intensity_formation_radius_high;
    v.linear_formation_radius_low(pixel) = ar.analysis.linear_formation_radius_low;
    v.linear_formation_radius_median(pixel) = ar.analysis.linear_formation_radius_median;
    v.linear_formation_radius_high(pixel) = ar.analysis.linear_formation_radius_high;
    v.circular_formation_radius_low(pixel) = ar.analysis.circular_formation_radius_low;
    v.circular_formation_radius_median(pixel) = ar.analysis.circular_formation_radius_median;
    v.circular_formation_radius_high(pixel) = ar.analysis.circular_formation_radius_high;
    v.los_linear_coherence(pixel) = ar.analysis.los_linear_coherence;
    v.los_circular_coherence(pixel) = ar.analysis.los_circular_coherence;
    v.contribution_closure_max_abs(pixel) = ar.analysis.contribution_closure_max_abs;
    v.contribution_closure_relative_l1(pixel) = ar.analysis.contribution_closure_relative_l1;
    v.foreground_absorption_depth(pixel) = ar.analysis.foreground_absorption_depth;
    v.foreground_faraday_rotation_depth(pixel) = ar.analysis.foreground_faraday_rotation_depth;
    v.foreground_faraday_conversion_depth(pixel) = ar.analysis.foreground_faraday_conversion_depth;
    v.foreground_faraday_operator_depth(pixel) = ar.analysis.foreground_faraday_operator_depth;
    v.foreground_faraday_operator_fraction(pixel) = ar.analysis.foreground_faraday_operator_fraction;
}

template<class ExecSpace, class RealT, class RadiationModel, class Metric>
void run_analysis_pass_b_segment_model_metric(
    const kpolaris::PassAParams<RealT>& params,
    const kpolaris::AnalysisConfig<RealT>& analysis_config,
    const RadiationModel& radiation_model,
    int radiation_substeps_arg,
    const Metric& metric,
    const AnalysisImageViews<RealT, ExecSpace>& views) {
    const int npix = params.camera.nx * params.camera.ny;
    const AnalysisImageViews<RealT, ExecSpace> local_views = views;
    const kpolaris::AnalysisConfig<RealT> local_analysis_config = analysis_config;
    Kokkos::parallel_for(
        "KPOLARISAnalysisImageMetric",
        Kokkos::RangePolicy<ExecSpace>(0, npix),
        KOKKOS_LAMBDA(const int pixel) {
            const kpolaris::PassBAnalysisResult<RealT> ar =
                kpolaris::trace_pass_b_segment_model_analysis_pixel_metric(
                    pixel, params, local_analysis_config, radiation_model,
                    radiation_substeps_arg, metric, local_views.response_data);
            store_analysis_pixel_result(pixel, ar, local_analysis_config, local_views);
        });
}

template<class ExecSpace, class RealT, class RadiationModel, class Metric>
void run_split_analysis_pass_b_segment_model_metric(
    const kpolaris::PassAParams<RealT>& params,
    const kpolaris::AnalysisConfig<RealT>& analysis_config,
    const RadiationModel& radiation_model,
    int radiation_substeps_arg,
    const Metric& metric,
    const AnalysisImageViews<RealT, ExecSpace>& views,
    int timing) {
    const int npix = params.camera.nx * params.camera.ny;
    const AnalysisImageViews<RealT, ExecSpace> local_views = views;
    const kpolaris::AnalysisConfig<RealT> local_analysis_config = analysis_config;
    Kokkos::View<RealT*, ExecSpace> endpoint_x("analysis_endpoint_x", npix * kpolaris::ndim);
    Kokkos::View<RealT*, ExecSpace> endpoint_k("analysis_endpoint_k", npix * kpolaris::ndim);
    Kokkos::View<RealT*, ExecSpace> endpoint_e1("analysis_endpoint_e1", npix * kpolaris::ndim);
    Kokkos::View<RealT*, ExecSpace> endpoint_e2("analysis_endpoint_e2", npix * kpolaris::ndim);
    Kokkos::View<int*, ExecSpace> endpoint_valid("analysis_endpoint_valid", npix);
    Kokkos::View<int*, ExecSpace> endpoint_reason("analysis_endpoint_reason", npix);

    Kokkos::Timer pass_a_timer;
    Kokkos::parallel_for(
        "KPOLARISAnalysisPassAEndpointMetric",
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
            local_views.pass_a_steps(pixel) = r.steps;
            endpoint_reason(pixel) = static_cast<int>(r.reason);
        });
    Kokkos::fence();
    report_image_timing(timing, "image_analysis_pass_a_endpoint", pass_a_timer.seconds());

    Kokkos::Timer pass_b_timer;
    Kokkos::parallel_for(
        "KPOLARISAnalysisPassBFromEndpointMetric",
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
            const kpolaris::PassBAnalysisResult<RealT> ar =
                kpolaris::trace_pass_b_segment_model_endpoint_analysis_pixel_metric(
                    pixel, params, local_analysis_config, radiation_model,
                    radiation_substeps_arg, endpoint_state,
                    local_views.pass_a_steps(pixel),
                    static_cast<kpolaris::TerminationReason>(endpoint_reason(pixel)),
                    endpoint_valid(pixel), metric, local_views.response_data);
            store_analysis_pixel_result(pixel, ar, local_analysis_config, local_views);
        });
    Kokkos::fence();
    report_image_timing(timing, "image_analysis_pass_b_transport", pass_b_timer.seconds());
}

template<class Metric, class RadiationModel>
ImageHostData run_analysis_image_metric_with_optional_split(
    const kpolaris::PassAParams<Real>& pass_a,
    const kpolaris::AnalysisConfig<Real>& analysis_config,
    RadiationModel model,
    int radiation_substeps_arg,
    int timing,
    int split_transport,
    const Metric& metric,
    const char* timing_suffix) {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    const int npix = pass_a.camera.nx * pass_a.camera.ny;
    AnalysisImageViews<Real, ExecSpace> views =
        make_analysis_image_views<Real, ExecSpace>(npix, analysis_config);

    Kokkos::Timer kernel_timer;
    if (split_transport) {
        run_split_analysis_pass_b_segment_model_metric<ExecSpace>(
            pass_a, analysis_config, model, radiation_substeps_arg, metric, views, timing);
    } else {
        run_analysis_pass_b_segment_model_metric<ExecSpace>(
            pass_a, analysis_config, model, radiation_substeps_arg, metric, views);
        Kokkos::fence();
    }
    (void)timing_suffix;
    report_image_timing(timing, "image_analysis_kernel_single", kernel_timer.seconds());

    Kokkos::Timer copy_timer;
    ImageHostData out;
    out.npix = npix;
    out.nfreq = 1;
    out.has_analysis = 1;
    out.response_config = analysis_config.response;
    if (analysis_config.response.enabled()) out.response_data = copy_real_view_1d(views.response_data);
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
    out.analysis_radial_bins = analysis_config.radial_bins;
    out.analysis_radial_min = analysis_config.radial_min;
    out.analysis_radial_max = analysis_config.radial_max;
    out.analysis_formation_fraction = analysis_config.formation_fraction;
    out.analysis_radial_bin_edges.resize(static_cast<size_t>(analysis_config.radial_bins) + 1);
    out.analysis_radial_bin_centers.resize(static_cast<size_t>(analysis_config.radial_bins));
    const Real log_min = Kokkos::log(analysis_config.radial_min);
    const Real log_max = Kokkos::log(analysis_config.radial_max);
    for (int bin = 0; bin <= analysis_config.radial_bins; ++bin) {
        out.analysis_radial_bin_edges[static_cast<size_t>(bin)] =
            Kokkos::exp(log_min + Real(bin) / Real(analysis_config.radial_bins) *
                        (log_max - log_min));
        if (bin < analysis_config.radial_bins) {
            out.analysis_radial_bin_centers[static_cast<size_t>(bin)] =
                kpolaris::analysis_radial_bin_center(analysis_config, bin);
        }
    }
    out.radial_stokes_i_contribution = copy_real_view_1d(views.radial_stokes_i_contribution);
    out.radial_stokes_q_contribution = copy_real_view_1d(views.radial_stokes_q_contribution);
    out.radial_stokes_u_contribution = copy_real_view_1d(views.radial_stokes_u_contribution);
    out.radial_stokes_v_contribution = copy_real_view_1d(views.radial_stokes_v_contribution);
    out.radial_absorption_depth = copy_real_view_1d(views.radial_absorption_depth);
    out.radial_faraday_rotation_depth = copy_real_view_1d(views.radial_faraday_rotation_depth);
    out.radial_faraday_conversion_depth = copy_real_view_1d(views.radial_faraday_conversion_depth);
    out.radial_faraday_operator_depth = copy_real_view_1d(views.radial_faraday_operator_depth);
    out.observer_weighted_radius_i = copy_real_view_1d(views.observer_weighted_radius_i);
    out.observer_weighted_radius_linear = copy_real_view_1d(views.observer_weighted_radius_linear);
    out.observer_weighted_radius_circular = copy_real_view_1d(views.observer_weighted_radius_circular);
    out.intensity_formation_radius_low = copy_real_view_1d(views.intensity_formation_radius_low);
    out.intensity_formation_radius_median = copy_real_view_1d(views.intensity_formation_radius_median);
    out.intensity_formation_radius_high = copy_real_view_1d(views.intensity_formation_radius_high);
    out.linear_formation_radius_low = copy_real_view_1d(views.linear_formation_radius_low);
    out.linear_formation_radius_median = copy_real_view_1d(views.linear_formation_radius_median);
    out.linear_formation_radius_high = copy_real_view_1d(views.linear_formation_radius_high);
    out.circular_formation_radius_low = copy_real_view_1d(views.circular_formation_radius_low);
    out.circular_formation_radius_median = copy_real_view_1d(views.circular_formation_radius_median);
    out.circular_formation_radius_high = copy_real_view_1d(views.circular_formation_radius_high);
    out.los_linear_coherence = copy_real_view_1d(views.los_linear_coherence);
    out.los_circular_coherence = copy_real_view_1d(views.los_circular_coherence);
    out.contribution_closure_max_abs = copy_real_view_1d(views.contribution_closure_max_abs);
    out.contribution_closure_relative_l1 = copy_real_view_1d(views.contribution_closure_relative_l1);
    out.foreground_absorption_depth = copy_real_view_1d(views.foreground_absorption_depth);
    out.foreground_faraday_rotation_depth = copy_real_view_1d(views.foreground_faraday_rotation_depth);
    out.foreground_faraday_conversion_depth = copy_real_view_1d(views.foreground_faraday_conversion_depth);
    out.foreground_faraday_operator_depth = copy_real_view_1d(views.foreground_faraday_operator_depth);
    out.foreground_faraday_operator_fraction = copy_real_view_1d(views.foreground_faraday_operator_fraction);
    report_image_timing(timing, "image_analysis_device_to_host_single", copy_timer.seconds());
    return out;
}

} // namespace kpolaris_image_detail
