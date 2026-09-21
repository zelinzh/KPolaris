#pragma once

#include "frame/stokes_rotation.hpp"
#include "geodesic/pass_a.hpp"
#include "model/model.hpp"
#include "radiation/semi_analytic.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct PassBParams {
    PassAParams<Real> pass_a;
    TransferCoeffs<Real> coeffs;
    int radiation_substeps = 1;
};

template<class Real = DefaultReal>
struct PassBResult {
    PassAResult<Real> pass_a;
    TransportState<Real> state;
    Stokes<Real> propagated_stokes;
    Stokes<Real> observed_stokes;
    BasisOverlap2<Real> overlap;
    Real closure_x = Real(0);
    Real closure_k = Real(0);
    Real final_null = Real(0);
    Real frame_error = Real(0);
    int steps = 0;
    TerminationReason reason = TerminationReason::none;
};

template<class Real>
KPOLARIS_INLINE Real spatial_distance(const Vec4<Real>& a, const Vec4<Real>& b) {
    return Kokkos::sqrt(square(a[1] - b[1]) + square(a[2] - b[2]) +
                        square(a[3] - b[3]));
}


template<class Metric, class Real>
KPOLARIS_INLINE Real radial_coordinate(const Metric& metric, const Vec4<Real>& x) {
    return metric_domain_radius(metric, x);
}

template<class Real>
KPOLARIS_INLINE Real vector_max_abs_difference(const Vec4<Real>& a,
                                            const Vec4<Real>& b) {
    Real out = abs_val(a[0] - b[0]);
    for (int i = 1; i < ndim; ++i) {
        out = max_val(out, abs_val(a[i] - b[i]));
    }
    return out;
}

template<class Metric, class Real>
KPOLARIS_INLINE Real camera_surface_value(const Metric& metric,
                                       const CameraParams<Real>& camera,
                                       const TransportState<Real>& state) {
    if (camera.model == CameraModel::Pinhole) {
        return radial_coordinate(metric, state.x) - camera.radius;
    }

    const CameraBasis3<Real> basis = make_parallel_plane_basis(camera);
    const Vec3<Real> dx(state.x[1] - basis.position.x,
                        state.x[2] - basis.position.y,
                        state.x[3] - basis.position.z);
    return dot(dx, basis.forward);
}

template<class Real>
KPOLARIS_INLINE int camera_surface_crossed(Real s0, Real s1) {
    if (s0 == Real(0) || s1 == Real(0)) {
        return 1;
    }
    return (s0 < Real(0) && s1 > Real(0)) ||
           (s0 > Real(0) && s1 < Real(0));
}

template<class Real>
KPOLARIS_INLINE Real camera_surface_crossing_fraction(Real s0, Real s1) {
    const Real denom = s0 - s1;
    if (abs_val(denom) <= tiny_positive<Real>()) {
        return Real(1);
    }
    return min_val(Real(1), max_val(Real(0), s0 / denom));
}

template<class Metric, class Real>
KPOLARIS_INLINE BasisOverlap2<Real> screen_overlap(const Metric& metric,
                                                const Vec4<Real>& x,
                                                const TransportState<Real>& camera_state,
                                                const TransportState<Real>& propagated_state) {
    BasisOverlap2<Real> overlap;
    overlap.r11 = metric.dot(x, camera_state.e1, propagated_state.e1);
    overlap.r12 = metric.dot(x, camera_state.e1, propagated_state.e2);
    overlap.r21 = metric.dot(x, camera_state.e2, propagated_state.e1);
    overlap.r22 = metric.dot(x, camera_state.e2, propagated_state.e2);
    return overlap;
}

// Both passes keep the physical future-directed wavevector. Pass A transports
// the camera screen frame to the endpoint with negative affine steps; Pass B
// uses positive steps with the same frame and no axial sign correction.
template<class Real>
KPOLARIS_INLINE int backward_screen_orientation_sign(const CameraParams<Real>&) {
    return 1;
}

template<class Metric, class Real>
KPOLARIS_INLINE void initialize_backward_propagation_screen_frame(
    const Metric& metric,
    const CameraParams<Real>&,
    TransportState<Real>& state) {
    initialize_screen_frame_from_reference(metric, state.x, state.k,
                                           state.e1, state.e2);
}

// Q/U emissivities follow the e1/e2 projections in each radiation model.
// V and the Faraday axial terms carry the screen orientation explicitly.
template<class Real>
KPOLARIS_INLINE void transform_axial_coefficients_to_screen_orientation(
    TransferCoeffs<Real>& coeffs,
    int orientation_sign) {
    if (orientation_sign < 0) {
        coeffs.jV = -coeffs.jV;
        coeffs.aV = -coeffs.aV;
        coeffs.rV = -coeffs.rV;
    }
}


template<class Real>
KPOLARIS_INLINE Real absorption_operator_rate(const TransferCoeffs<Real>& coeffs) {
    return abs_val(coeffs.aI) +
        Kokkos::sqrt(square(coeffs.aQ) + square(coeffs.aU) + square(coeffs.aV));
}

template<class Real>
KPOLARIS_INLINE Real faraday_operator_rate(const TransferCoeffs<Real>& coeffs) {
    return Kokkos::sqrt(square(coeffs.rQ) + square(coeffs.rU) + square(coeffs.rV));
}

template<class Real>
KPOLARIS_INLINE Real effective_depth_limit(Real specific_limit, Real legacy_limit) {
    return specific_limit >= Real(0) ? specific_limit : legacy_limit;
}

template<class Real>
KPOLARIS_INLINE int radiation_substep_count(const TransferCoeffs<Real>& coeffs,
                                         Real geometric_step,
                                         Real dlambda_scale,
                                         int requested_substeps,
                                         Real max_radiation_step,
                                         Real max_radiation_depth,
                                         Real max_absorption_depth,
                                         Real max_faraday_depth) {
    int nsub = requested_substeps > 0 ? requested_substeps : 1;
    const Real abs_h = abs_val(geometric_step);
    if (max_radiation_step > Real(0)) {
        while (abs_h / Real(nsub) > max_radiation_step && nsub < 4096) {
            nsub *= 2;
        }
    }
    const Real dl = abs_h * abs_val(dlambda_scale);
    const Real absorption_limit = effective_depth_limit(max_absorption_depth, max_radiation_depth);
    if (absorption_limit > Real(0)) {
        const Real depth = absorption_operator_rate(coeffs) * dl;
        while (depth / Real(nsub) > absorption_limit && nsub < 4096) {
            nsub *= 2;
        }
    }
    const Real faraday_limit = effective_depth_limit(max_faraday_depth, max_radiation_depth);
    if (faraday_limit > Real(0)) {
        const Real depth = faraday_operator_rate(coeffs) * dl;
        while (depth / Real(nsub) > faraday_limit && nsub < 4096) {
            nsub *= 2;
        }
    }
    return nsub;
}


template<class Real>
KPOLARIS_INLINE void radiation_step_with_substeps(Stokes<Real>& stokes,
                                               const TransferCoeffs<Real>& coeffs,
                                               Real dlambda,
                                               int nsub) {
    const int safe_nsub = nsub > 0 ? nsub : 1;
    const Real sub_dl = dlambda / Real(safe_nsub);
    for (int s = 0; s < safe_nsub; ++s) {
        semi_analytic_stokes_step(stokes, coeffs, sub_dl);
    }
}


template<class Metric, class Real, class RadiationModel>
KPOLARIS_INLINE PassBResult<Real> trace_pass_b_model_pixel_metric(int pixel,
                                                               const PassAParams<Real>& pass_a_params,
                                                               const RadiationModel& radiation_model,
                                                               int radiation_substeps,
                                                               const Metric& metric) {
    PassBResult<Real> out;
    out.pass_a = trace_pass_a_pixel_metric(pixel, pass_a_params, metric);
    out.state = out.pass_a.state;
    out.reason = out.pass_a.reason;

    const TransportState<Real> camera_state =
        initialize_camera_ray(metric, pixel, pass_a_params.camera);

    if (out.pass_a.reason != TerminationReason::reached_inner_boundary &&
        out.pass_a.reason != TerminationReason::reached_direct_turn) {
        out.closure_x = spatial_distance(out.state.x, camera_state.x);
        out.closure_k = vector_max_abs_difference(out.state.k, camera_state.k);
        out.final_null = metric.dot(out.state.x, out.state.k, out.state.k);
        out.frame_error = max_frame_error(frame_errors(metric, out.state));
        return out;
    }

    // Switch to positive affine steps while retaining the physical wavevector
    // and the camera screen basis transported to this endpoint by Pass A.
    const int screen_orientation =
        backward_screen_orientation_sign(pass_a_params.camera);

    Stokes<Real> stokes;
    const Real nominal_h = pass_a_params.step;
    const int safe_nsub = radiation_substeps > 0 ? radiation_substeps : 1;
    for (int n = 0; n < out.pass_a.steps; ++n) {
        Real h = nominal_h;
        if (!pass_a_params.adaptive && n == 0 && out.pass_a.steps > 0) {
            const Real final_segment = out.pass_a.path_length -
                abs_val(nominal_h) * Real(out.pass_a.steps - 1);
            if (final_segment >= Real(0) &&
                final_segment <= abs_val(nominal_h) * Real(1.0001)) {
                h = nominal_h < Real(0) ? -final_segment : final_segment;
            }
        }
        if (!metric_time_domain_valid(metric, out.state.x)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const Real effective_min_step = metric_effective_min_step(
            metric, out.state.k, pass_a_params.min_step);
        if (metric_time_domain_step_is_terminal(
                metric, out.state.x, out.state.k,
                h, effective_min_step)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        h = metric_limit_time_domain_step(
            metric, out.state.x, out.state.k, h);
        const TransportState<Real> next_state = rk4_step(metric, out.state, h);
        if (!metric_time_domain_valid(metric, next_state.x)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        TransferCoeffs<Real> coeffs =
            radiation_model.coefficients(metric, next_state, Real(1));
        apply_emission_selection(pass_a_params.emission_selection, metric, next_state, coeffs);
        transform_axial_coefficients_to_screen_orientation(coeffs,
                                                           screen_orientation);
        const int required_steps = radiation_substep_count(
            coeffs, h, radiation_model.dlambda_scale(), safe_nsub,
            pass_a_params.max_radiation_step, pass_a_params.max_radiation_depth,
            pass_a_params.max_absorption_depth, pass_a_params.max_faraday_depth);
        const Real dlambda = abs_val(h) * radiation_model.dlambda_scale() /
                             Real(required_steps > 0 ? required_steps : 1);
        for (int s = 0; s < required_steps; ++s) {
            semi_analytic_stokes_step(stokes, coeffs, dlambda);
        }
        out.state = next_state;
        out.steps += 1;
    }

    out.propagated_stokes = stokes;
    out.overlap = screen_overlap(metric, out.state.x, camera_state, out.state);
    out.observed_stokes = transform_to_observer_basis(stokes, out.overlap);
    out.closure_x = spatial_distance(out.state.x, camera_state.x);
    Vec4<Real> target_k;
    for (int mu = 0; mu < ndim; ++mu) {
        target_k[mu] = camera_state.k[mu];
    }
    out.closure_k = vector_max_abs_difference(out.state.k, target_k);
    out.final_null = metric.dot(out.state.x, out.state.k, out.state.k);
    out.frame_error = max_frame_error(frame_errors(metric, out.state));
    if (out.reason != TerminationReason::metric_time_exhausted) {
        out.reason = TerminationReason::reached_camera;
    }
    return out;
}


template<class Real, class RadiationModel>
KPOLARIS_INLINE PassBResult<Real> trace_pass_b_model_pixel(int pixel,
                                                        const PassAParams<Real>& pass_a_params,
                                                        const RadiationModel& radiation_model,
                                                        int radiation_substeps) {
    if (pass_a_params.coordinate_system == CoordinateSystem::BoyerLindquist) {
        const KerrBoyerLindquistMetric<Real> metric(pass_a_params.mass, pass_a_params.spin);
        return trace_pass_b_model_pixel_metric(pixel, pass_a_params, radiation_model,
                                               radiation_substeps, metric);
    }
    if (pass_a_params.coordinate_system == CoordinateSystem::SphericalKS) {
        const KerrSchildSphericalMetric<Real> metric(pass_a_params.mass, pass_a_params.spin);
        return trace_pass_b_model_pixel_metric(pixel, pass_a_params, radiation_model,
                                               radiation_substeps, metric);
    }
    if (pass_a_params.coordinate_system == CoordinateSystem::MKS) {
        const KerrSchildSphericalMetric<Real> metric(pass_a_params.mass, pass_a_params.spin);
        return trace_pass_b_model_pixel_metric(pixel, pass_a_params, radiation_model,
                                               radiation_substeps, metric);
    }
    if (pass_a_params.coordinate_system == CoordinateSystem::FMKS) {
        const KerrFMKSMetric<Real> metric(pass_a_params.mass, pass_a_params.spin, pass_a_params.fmks_startx1,
                                         pass_a_params.fmks_hslope, pass_a_params.fmks_mks_smooth,
                                         pass_a_params.fmks_poly_alpha, pass_a_params.fmks_poly_xt,
                                         pass_a_params.fmks_poly_norm);
        return trace_pass_b_model_pixel_metric(pixel, pass_a_params, radiation_model,
                                               radiation_substeps, metric);
    }
    const KerrSchildInMetric<Real> metric(pass_a_params.mass, pass_a_params.spin);
    return trace_pass_b_model_pixel_metric(pixel, pass_a_params, radiation_model,
                                           radiation_substeps, metric);
}




template<class Real = DefaultReal>
struct PassAEndpointResult {
    TransportState<Real> state;
    Real final_null = Real(0);
    Real frame_error = Real(0);
    int steps = 0;
    int valid = 0;
    TerminationReason reason = TerminationReason::none;
};

template<class Metric, class Real>
KPOLARIS_INLINE PassAEndpointResult<Real> trace_pass_a_segment_endpoint_pixel_metric(
    int pixel,
    const PassAParams<Real>& params,
    const Metric& metric) {
    PassAEndpointResult<Real> out;
    out.state = initialize_camera_ray(metric, pixel, params.camera);
    out.reason = TerminationReason::max_steps;

    AdaptiveRK4Control<Real> adaptive_control;
    adaptive_control.tolerance = params.adaptive_tolerance;
    adaptive_control.min_step = params.min_step;
    adaptive_control.max_step = params.max_step;
    Real h_current = -abs_val(params.step);
    bool entered_radiation_sphere = false;
    int steps_inside_or_after_entry = 0;
    for (int n = 0; n < params.max_steps; ++n) {
        if (!metric_time_domain_valid(metric, out.state.x)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const Real r_bl = radial_coordinate(metric, out.state.x);

        if (!entered_radiation_sphere) {
            if (r_bl <= params.outer_radius) {
                entered_radiation_sphere = true;
                steps_inside_or_after_entry = 0;
            }
        } else {
            if (metric_inner_surface_value(metric, out.state.x,
                                               params.inner_radius) <= Real(0)) {
                out.reason = TerminationReason::reached_inner_boundary;
                break;
            }
            if (steps_inside_or_after_entry > 0 && r_bl >= params.outer_radius) {
                out.reason = TerminationReason::escaped_domain;
                break;
            }
        }

        const TransportState<Real> old_state = out.state;
        const Real effective_min_step = metric_effective_min_step(
            metric, old_state.k, params.min_step);
        if (metric_time_domain_step_is_terminal(
                metric, old_state.x, old_state.k,
                h_current, effective_min_step)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const Real old_r = r_bl;
        const Real old_inner_value = metric_inner_surface_value(
            metric, old_state.x, params.inner_radius);
        if (entered_radiation_sphere &&
            metric_inner_surface_step_is_terminal(
                metric, old_state.x, old_state.k,
                params.inner_radius, h_current, effective_min_step)) {
            out.reason = TerminationReason::reached_inner_boundary;
            break;
        }
        Real used_h = metric_limit_time_domain_step(
            metric, old_state.x, old_state.k, h_current);
        used_h = metric_limit_inner_surface_step(
            metric, old_state.x, old_state.k, params.inner_radius, used_h);
        TransportState<Real> next_state;
        if (params.adaptive) {
            AdaptiveRK4Control<Real> trial_control = adaptive_control;
            trial_control.min_step = effective_min_step;
            if (abs_val(used_h) < trial_control.min_step) {
                trial_control.min_step = abs_val(used_h);
            }
            const auto step = adaptive_rk4_step(metric, old_state, used_h, trial_control);
            if (!step.accepted) {
                h_current = step.next_h;
                if (abs_val(h_current) <= effective_min_step * Real(1.0001)) {
                    out.reason = TerminationReason::adaptive_step_underflow;
                    break;
                }
                continue;
            }
            next_state = step.state;
            used_h = step.used_h;
            h_current = step.next_h;
        } else {
            next_state = rk4_step(metric, out.state, used_h);
        }

        if (!metric_time_domain_valid(metric, next_state.x)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const Real next_r = radial_coordinate(metric, next_state.x);
        const Real next_inner_value = metric_inner_surface_value(
            metric, next_state.x, params.inner_radius);
        if (!entered_radiation_sphere && next_r <= params.outer_radius) {
            entered_radiation_sphere = true;
            steps_inside_or_after_entry = 0;
        }
        else if (!entered_radiation_sphere && next_r >= old_r) {
            out.state = next_state;
            out.reason = TerminationReason::escaped_domain;
            out.steps += 1;
            break;
        }

        TerminationReason crossing_reason = TerminationReason::none;
        if (entered_radiation_sphere && old_inner_value > Real(0) &&
            next_inner_value <= Real(0)) {
            crossing_reason = TerminationReason::reached_inner_boundary;
        } else if (entered_radiation_sphere && steps_inside_or_after_entry > 0 &&
                   old_r < params.outer_radius && next_r >= params.outer_radius) {
            crossing_reason = TerminationReason::escaped_domain;
        }

        if (crossing_reason != TerminationReason::none) {
            const int inner_event =
                crossing_reason == TerminationReason::reached_inner_boundary;
            const Real surface_parameter = inner_event ?
                params.inner_radius : params.outer_radius;
            const MetricSurfaceEvent<Real> event = locate_metric_surface_crossing(
                metric, old_state, next_state, used_h, surface_parameter,
                inner_event, params.adaptive);
            next_state = event.state;
            used_h = event.used_h;
        }
        if (params.direct_only && truncate_at_first_vertical_turn(
                metric, old_state, next_state, used_h, params.adaptive)) {
            crossing_reason = TerminationReason::reached_direct_turn;
        }
        if (crossing_reason != TerminationReason::none) {
            out.state = next_state;
            out.reason = crossing_reason;
            out.steps += 1;
            break;
        }

        out.state = next_state;
        out.steps += 1;
        if (entered_radiation_sphere) {
            steps_inside_or_after_entry += 1;
        }
    }

    out.valid = out.reason == TerminationReason::reached_direct_turn ||
                (entered_radiation_sphere &&
                 (out.reason == TerminationReason::reached_inner_boundary ||
                  out.reason == TerminationReason::escaped_domain));
    out.final_null = metric.dot(out.state.x, out.state.k, out.state.k);
    out.frame_error = max_frame_error(frame_errors(metric, out.state));
    return out;
}

template<class Real>
KPOLARIS_INLINE PassAEndpointResult<Real> trace_pass_a_segment_endpoint_pixel(
    int pixel,
    const PassAParams<Real>& params) {
    if (params.coordinate_system == CoordinateSystem::BoyerLindquist) {
        const KerrBoyerLindquistMetric<Real> metric(params.mass, params.spin);
        return trace_pass_a_segment_endpoint_pixel_metric(pixel, params, metric);
    }
    if (params.coordinate_system == CoordinateSystem::SphericalKS) {
        const KerrSchildSphericalMetric<Real> metric(params.mass, params.spin);
        return trace_pass_a_segment_endpoint_pixel_metric(pixel, params, metric);
    }
    if (params.coordinate_system == CoordinateSystem::MKS) {
        const KerrSchildSphericalMetric<Real> metric(params.mass, params.spin);
        return trace_pass_a_segment_endpoint_pixel_metric(pixel, params, metric);
    }
    if (params.coordinate_system == CoordinateSystem::FMKS) {
        const KerrFMKSMetric<Real> metric(params.mass, params.spin, params.fmks_startx1,
                                         params.fmks_hslope, params.fmks_mks_smooth,
                                         params.fmks_poly_alpha, params.fmks_poly_xt,
                                         params.fmks_poly_norm);
        return trace_pass_a_segment_endpoint_pixel_metric(pixel, params, metric);
    }
    const KerrSchildInMetric<Real> metric(params.mass, params.spin);
    return trace_pass_a_segment_endpoint_pixel_metric(pixel, params, metric);
}

template<class Metric, class Real, class RadiationModel>
KPOLARIS_INLINE PassBResult<Real> trace_pass_b_segment_model_endpoint_pixel_metric(
    int pixel,
    const PassAParams<Real>& params,
    const RadiationModel& radiation_model,
    int radiation_substeps,
    const TransportState<Real>& endpoint_state,
    int pass_a_steps,
    TerminationReason pass_a_reason,
    int endpoint_valid,
    const Metric& metric) {
    (void)radiation_substeps;
    PassBResult<Real> out;
    const TransportState<Real> camera_state =
        initialize_camera_ray(metric, pixel, params.camera);

    out.state = endpoint_state;
    out.pass_a.state = endpoint_state;
    out.pass_a.steps = pass_a_steps;
    out.pass_a.reason = pass_a_reason;
    out.pass_a.final_null = metric.dot(endpoint_state.x, endpoint_state.k, endpoint_state.k);
    out.pass_a.frame_error = max_frame_error(frame_errors(metric, endpoint_state));
    out.reason = pass_a_reason;

    if (!endpoint_valid) {
        out.closure_x = spatial_distance(out.state.x, camera_state.x);
        out.closure_k = vector_max_abs_difference(out.state.k, camera_state.k);
        out.final_null = out.pass_a.final_null;
        out.frame_error = out.pass_a.frame_error;
        return out;
    }

    const int screen_orientation = backward_screen_orientation_sign(params.camera);

    AdaptiveRK4Control<Real> transfer_control;
    transfer_control.tolerance = params.adaptive_tolerance;
    transfer_control.min_step = params.min_step;
    transfer_control.max_step = params.max_step;

    Stokes<Real> stokes;
    const Real dlambda_scale = radiation_model.dlambda_scale();
    int reached_camera = 0;
    Real h_current = params.step;
    Real radiation_step_cap = params.max_radiation_step > Real(0) ? params.max_radiation_step : params.max_step;
    while (!reached_camera && out.steps < params.max_steps) {
        if (!metric_time_domain_valid(metric, out.state.x)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        Real h = h_current;
        const TransportState<Real> old_state = out.state;
        const Real effective_min_step = metric_effective_min_step(
            metric, old_state.k, params.min_step);
        const Real start_r_bl = radial_coordinate(metric, old_state.x);
        if (start_r_bl <= params.outer_radius && radiation_step_cap > Real(0)) {
            h = min_val(h, radiation_step_cap);
        }
        if (metric_time_domain_step_is_terminal(
                metric, old_state.x, old_state.k,
                h, effective_min_step)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        h = metric_limit_time_domain_step(
            metric, old_state.x, old_state.k, h);
        TransportState<Real> next_state;
        TransportState<Real> sample_state;
        if (params.adaptive) {
            AdaptiveRK4Control<Real> trial_control = transfer_control;
            trial_control.min_step = effective_min_step;
            const auto proposed = adaptive_rk4_step(metric, old_state, h, trial_control);
            if (!proposed.accepted) {
                h_current = abs_val(proposed.next_h);
                if (h_current <= effective_min_step * Real(1.0001)) {
                    out.reason = TerminationReason::adaptive_step_underflow;
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

        if (!metric_time_domain_valid(metric, next_state.x)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
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
                    crossing_control.min_step = min_val(effective_min_step, abs_val(h));
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
                if (h <= effective_min_step*Real(1.0001)) {
                    out.reason = TerminationReason::adaptive_step_underflow;
                    break;
                }
                h_current = max_val(effective_min_step, h/Real(wedge_substeps));
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
        TransferCoeffs<Real> coeffs;
        if (metric_inner_surface_value(metric, sample_state.x,
                                       params.inner_radius) > Real(0) &&
            sample_r_bl < params.outer_radius) {
            coeffs = radiation_model.coefficients(metric, sample_state, Real(0.5));
            apply_emission_selection(params.emission_selection, metric, sample_state, coeffs);
            transform_axial_coefficients_to_screen_orientation(coeffs,
                                                               screen_orientation);
        }
        const Real local_max_radiation_step =
            (params.max_radiation_step > Real(0) &&
             (old_r_bl <= params.outer_radius || sample_r_bl <= params.outer_radius ||
              next_r_bl <= params.outer_radius)) ? params.max_radiation_step : Real(0);
        const int required_steps = radiation_substep_count(
            coeffs, h, dlambda_scale, 1,
            local_max_radiation_step, params.max_radiation_depth,
                params.max_absorption_depth, params.max_faraday_depth);
        if (required_steps > 1 && h > effective_min_step * Real(1.0001)) {
            h_current = max_val(effective_min_step, h / Real(required_steps));
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
                max_val(effective_min_step, h_current / Real(predictive_steps)) : params.max_step;
        } else {
            radiation_step_cap = params.max_step;
        }
        if (h > Real(0)) {
            semi_analytic_stokes_step(stokes, coeffs,
                                      abs_val(h) * dlambda_scale);
        }
        out.state = next_state;
        out.steps += 1;
        if (crosses_camera) {
            reached_camera = 1;
        }
    }

    out.propagated_stokes = stokes;
    out.overlap = screen_overlap(metric, out.state.x, camera_state, out.state);
    out.observed_stokes = transform_to_observer_basis(stokes, out.overlap);
    out.closure_x = spatial_distance(out.state.x, camera_state.x);
    Vec4<Real> target_k;
    for (int mu = 0; mu < ndim; ++mu) {
        target_k[mu] = camera_state.k[mu];
    }
    out.closure_k = vector_max_abs_difference(out.state.k, target_k);
    out.final_null = metric.dot(out.state.x, out.state.k, out.state.k);
    out.frame_error = max_frame_error(frame_errors(metric, out.state));
    if (reached_camera) {
        out.reason = TerminationReason::reached_camera;
    } else if (out.reason != TerminationReason::adaptive_step_underflow &&
               out.reason != TerminationReason::metric_time_exhausted) {
        out.reason = TerminationReason::max_steps;
    }
    return out;
}


template<class Real, class RadiationModel>
KPOLARIS_INLINE PassBResult<Real> trace_pass_b_segment_model_endpoint_pixel(
    int pixel,
    const PassAParams<Real>& params,
    const RadiationModel& radiation_model,
    int radiation_substeps,
    const TransportState<Real>& endpoint_state,
    int pass_a_steps,
    TerminationReason pass_a_reason,
    int endpoint_valid) {
    if (params.coordinate_system == CoordinateSystem::BoyerLindquist) {
        const KerrBoyerLindquistMetric<Real> metric(params.mass, params.spin);
        return trace_pass_b_segment_model_endpoint_pixel_metric(
            pixel, params, radiation_model, radiation_substeps, endpoint_state,
            pass_a_steps, pass_a_reason, endpoint_valid, metric);
    }
    if (params.coordinate_system == CoordinateSystem::SphericalKS) {
        const KerrSchildSphericalMetric<Real> metric(params.mass, params.spin);
        return trace_pass_b_segment_model_endpoint_pixel_metric(
            pixel, params, radiation_model, radiation_substeps, endpoint_state,
            pass_a_steps, pass_a_reason, endpoint_valid, metric);
    }
    if (params.coordinate_system == CoordinateSystem::MKS) {
        const KerrSchildSphericalMetric<Real> metric(params.mass, params.spin);
        return trace_pass_b_segment_model_endpoint_pixel_metric(
            pixel, params, radiation_model, radiation_substeps, endpoint_state,
            pass_a_steps, pass_a_reason, endpoint_valid, metric);
    }
    if (params.coordinate_system == CoordinateSystem::FMKS) {
        const KerrFMKSMetric<Real> metric(params.mass, params.spin, params.fmks_startx1,
                                         params.fmks_hslope, params.fmks_mks_smooth,
                                         params.fmks_poly_alpha, params.fmks_poly_xt,
                                         params.fmks_poly_norm);
        return trace_pass_b_segment_model_endpoint_pixel_metric(
            pixel, params, radiation_model, radiation_substeps, endpoint_state,
            pass_a_steps, pass_a_reason, endpoint_valid, metric);
    }
    const KerrSchildInMetric<Real> metric(params.mass, params.spin);
    return trace_pass_b_segment_model_endpoint_pixel_metric(
        pixel, params, radiation_model, radiation_substeps, endpoint_state,
        pass_a_steps, pass_a_reason, endpoint_valid, metric);
}

template<class Metric, class Real, class RadiationModel>
KPOLARIS_INLINE PassBResult<Real> trace_pass_b_segment_model_pixel_metric(int pixel,
                                                                       const PassAParams<Real>& params,
                                                                       const RadiationModel& radiation_model,
                                                                       int radiation_substeps,
                                                                       const Metric& metric) {
    PassBResult<Real> out;
    const TransportState<Real> camera_state =
        initialize_camera_ray(metric, pixel, params.camera);

    out.state = camera_state;
    out.pass_a.state = camera_state;
    out.pass_a.initial_null = metric.dot(out.state.x, out.state.k, out.state.k);
    out.pass_a.min_radius = radial_coordinate(metric, out.state.x);
    out.pass_a.final_radius = out.pass_a.min_radius;
    out.pass_a.reason = TerminationReason::max_steps;
    out.reason = TerminationReason::max_steps;

    AdaptiveRK4Control<Real> adaptive_control;
    adaptive_control.tolerance = params.adaptive_tolerance;
    adaptive_control.min_step = params.min_step;
    adaptive_control.max_step = params.max_step;
    Real h_current = -abs_val(params.step);
    bool entered_radiation_sphere = false;
    int steps_inside_or_after_entry = 0;
    for (int n = 0; n < params.max_steps; ++n) {
        if (!metric_time_domain_valid(metric, out.state.x)) {
            out.pass_a.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const Real r_bl = radial_coordinate(metric, out.state.x);
        out.pass_a.min_radius = min_val(out.pass_a.min_radius, r_bl);

        if (!entered_radiation_sphere) {
            if (r_bl <= params.outer_radius) {
                entered_radiation_sphere = true;
                steps_inside_or_after_entry = 0;
            }
        } else {
            if (metric_inner_surface_value(metric, out.state.x,
                                           params.inner_radius) <= Real(0)) {
                out.pass_a.reason = TerminationReason::reached_inner_boundary;
                break;
            }
            if (steps_inside_or_after_entry > 0 && r_bl >= params.outer_radius) {
                out.pass_a.reason = TerminationReason::escaped_domain;
                break;
            }
        }

        const TransportState<Real> old_state = out.state;
        const Real effective_min_step = metric_effective_min_step(
            metric, old_state.k, params.min_step);
        if (metric_time_domain_step_is_terminal(
                metric, old_state.x, old_state.k,
                h_current, effective_min_step)) {
            out.pass_a.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const Real old_r = r_bl;
        const Real old_inner_value = metric_inner_surface_value(
            metric, old_state.x, params.inner_radius);
        if (entered_radiation_sphere &&
            metric_inner_surface_step_is_terminal(
                metric, old_state.x, old_state.k,
                params.inner_radius, h_current, effective_min_step)) {
            out.pass_a.reason = TerminationReason::reached_inner_boundary;
            break;
        }
        Real used_h = metric_limit_time_domain_step(
            metric, old_state.x, old_state.k, h_current);
        used_h = metric_limit_inner_surface_step(
            metric, old_state.x, old_state.k, params.inner_radius, used_h);
        TransportState<Real> next_state;
        if (params.adaptive) {
            AdaptiveRK4Control<Real> trial_control = adaptive_control;
            trial_control.min_step = effective_min_step;
            if (abs_val(used_h) < trial_control.min_step) {
                trial_control.min_step = abs_val(used_h);
            }
            const auto step = adaptive_rk4_step(metric, old_state, used_h, trial_control);
            if (!step.accepted) {
                h_current = step.next_h;
                if (abs_val(h_current) <= effective_min_step * Real(1.0001)) {
                    out.pass_a.reason = TerminationReason::adaptive_step_underflow;
                    break;
                }
                continue;
            }
            next_state = step.state;
            used_h = step.used_h;
            h_current = step.next_h;
        } else {
            next_state = rk4_step(metric, out.state, used_h);
        }

        if (!metric_time_domain_valid(metric, next_state.x)) {
            out.pass_a.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const Real next_r = radial_coordinate(metric, next_state.x);
        const Real next_inner_value = metric_inner_surface_value(
            metric, next_state.x, params.inner_radius);
        if (!entered_radiation_sphere && next_r <= params.outer_radius) {
            entered_radiation_sphere = true;
            steps_inside_or_after_entry = 0;
        }
        else if (!entered_radiation_sphere && next_r >= old_r) {
            out.state = next_state;
            out.pass_a.path_length += abs_val(used_h);
            out.pass_a.steps += 1;
            out.pass_a.reason = TerminationReason::escaped_domain;
            break;
        }

        TerminationReason crossing_reason = TerminationReason::none;
        if (entered_radiation_sphere && old_inner_value > Real(0) &&
            next_inner_value <= Real(0)) {
            crossing_reason = TerminationReason::reached_inner_boundary;
        } else if (entered_radiation_sphere && steps_inside_or_after_entry > 0 &&
                   old_r < params.outer_radius && next_r >= params.outer_radius) {
            crossing_reason = TerminationReason::escaped_domain;
        }

        if (crossing_reason != TerminationReason::none) {
            const int inner_event =
                crossing_reason == TerminationReason::reached_inner_boundary;
            const Real surface_parameter = inner_event ?
                params.inner_radius : params.outer_radius;
            const MetricSurfaceEvent<Real> event = locate_metric_surface_crossing(
                metric, old_state, next_state, used_h, surface_parameter,
                inner_event, params.adaptive);
            next_state = event.state;
            used_h = event.used_h;
        }
        if (params.direct_only && truncate_at_first_vertical_turn(
                metric, old_state, next_state, used_h, params.adaptive)) {
            crossing_reason = TerminationReason::reached_direct_turn;
        }
        if (crossing_reason != TerminationReason::none) {
            out.state = next_state;
            out.pass_a.path_length += abs_val(used_h);
            out.pass_a.steps += 1;
            out.pass_a.reason = crossing_reason;
            break;
        }

        out.state = next_state;
        out.pass_a.path_length += abs_val(used_h);
        out.pass_a.steps += 1;
        if (entered_radiation_sphere) {
            steps_inside_or_after_entry += 1;
        }
    }

    out.pass_a.state = out.state;
    out.pass_a.final_radius = radial_coordinate(metric, out.state.x);
    out.pass_a.final_null = metric.dot(out.state.x, out.state.k, out.state.k);
    out.pass_a.frame_error = max_frame_error(frame_errors(metric, out.state));
    out.reason = out.pass_a.reason;

    if (out.pass_a.reason != TerminationReason::reached_direct_turn &&
        (!entered_radiation_sphere ||
         (out.pass_a.reason != TerminationReason::reached_inner_boundary &&
          out.pass_a.reason != TerminationReason::escaped_domain))) {
        out.closure_x = spatial_distance(out.state.x, camera_state.x);
        out.closure_k = vector_max_abs_difference(out.state.k, camera_state.k);
        out.final_null = out.pass_a.final_null;
        out.frame_error = out.pass_a.frame_error;
        return out;
    }

    // Switch to positive affine steps while retaining the physical wavevector
    // and the camera screen basis transported to this endpoint by Pass A.
    const int screen_orientation = backward_screen_orientation_sign(params.camera);

    Stokes<Real> stokes;
    const Real dlambda_scale = radiation_model.dlambda_scale();
    (void)radiation_substeps;
    AdaptiveRK4Control<Real> transfer_control = adaptive_control;
    int reached_camera = 0;
    h_current = params.step;
    Real radiation_step_cap = params.max_radiation_step > Real(0) ? params.max_radiation_step : params.max_step;
    while (!reached_camera && out.steps < params.max_steps) {
        if (!metric_time_domain_valid(metric, out.state.x)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        Real h = h_current;
        const TransportState<Real> old_state = out.state;
        const Real effective_min_step = metric_effective_min_step(
            metric, old_state.k, params.min_step);
        const Real start_r_bl = radial_coordinate(metric, old_state.x);
        if (start_r_bl <= params.outer_radius && radiation_step_cap > Real(0)) {
            h = min_val(h, radiation_step_cap);
        }
        if (metric_time_domain_step_is_terminal(
                metric, old_state.x, old_state.k,
                h, effective_min_step)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        h = metric_limit_time_domain_step(
            metric, old_state.x, old_state.k, h);
        TransportState<Real> next_state;
        TransportState<Real> sample_state;
        if (params.adaptive) {
            AdaptiveRK4Control<Real> trial_control = transfer_control;
            trial_control.min_step = effective_min_step;
            const auto proposed = adaptive_rk4_step(metric, old_state, h, trial_control);
            if (!proposed.accepted) {
                h_current = abs_val(proposed.next_h);
                if (h_current <= effective_min_step * Real(1.0001)) {
                    out.reason = TerminationReason::adaptive_step_underflow;
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

        if (!metric_time_domain_valid(metric, next_state.x)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
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
                    crossing_control.min_step = min_val(effective_min_step, abs_val(h));
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
                if (h <= effective_min_step*Real(1.0001)) {
                    out.reason = TerminationReason::adaptive_step_underflow;
                    break;
                }
                h_current = max_val(effective_min_step, h/Real(wedge_substeps));
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
        TransferCoeffs<Real> coeffs;
        if (metric_inner_surface_value(metric, sample_state.x,
                                       params.inner_radius) > Real(0) &&
            sample_r_bl < params.outer_radius) {
            coeffs = radiation_model.coefficients(metric, sample_state, Real(0.5));
            apply_emission_selection(params.emission_selection, metric, sample_state, coeffs);
            transform_axial_coefficients_to_screen_orientation(coeffs,
                                                               screen_orientation);
        }
        const Real local_max_radiation_step =
            (params.max_radiation_step > Real(0) &&
             (old_r_bl <= params.outer_radius || sample_r_bl <= params.outer_radius ||
              next_r_bl <= params.outer_radius)) ? params.max_radiation_step : Real(0);
        const int required_steps = radiation_substep_count(
            coeffs, h, dlambda_scale, 1,
            local_max_radiation_step, params.max_radiation_depth,
                params.max_absorption_depth, params.max_faraday_depth);
        if (required_steps > 1 && h > effective_min_step * Real(1.0001)) {
            h_current = max_val(effective_min_step, h / Real(required_steps));
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
                max_val(effective_min_step, h_current / Real(predictive_steps)) : params.max_step;
        } else {
            radiation_step_cap = params.max_step;
        }
        if (h > Real(0)) {
            semi_analytic_stokes_step(stokes, coeffs,
                                      abs_val(h) * dlambda_scale);
        }
        out.state = next_state;
        out.steps += 1;
        if (crosses_camera) {
            reached_camera = 1;
        }
    }

    out.propagated_stokes = stokes;
    out.overlap = screen_overlap(metric, out.state.x, camera_state, out.state);
    out.observed_stokes = transform_to_observer_basis(stokes, out.overlap);
    out.closure_x = spatial_distance(out.state.x, camera_state.x);
    Vec4<Real> target_k;
    for (int mu = 0; mu < ndim; ++mu) {
        target_k[mu] = camera_state.k[mu];
    }
    out.closure_k = vector_max_abs_difference(out.state.k, target_k);
    out.final_null = metric.dot(out.state.x, out.state.k, out.state.k);
    out.frame_error = max_frame_error(frame_errors(metric, out.state));
    if (reached_camera) {
        out.reason = TerminationReason::reached_camera;
    } else if (out.reason != TerminationReason::adaptive_step_underflow &&
               out.reason != TerminationReason::metric_time_exhausted) {
        out.reason = TerminationReason::max_steps;
    }
    return out;
}


template<class Real = DefaultReal>
struct MultiFrequencyPassBResult {
    PassAResult<Real> pass_a;
    TransportState<Real> state;
    BasisOverlap2<Real> overlap;
    Real closure_x = Real(0);
    Real closure_k = Real(0);
    Real final_null = Real(0);
    Real frame_error = Real(0);
    int steps = 0;
    TerminationReason reason = TerminationReason::none;
};

template<class Metric, class Real, class RadiationModel>
KPOLARIS_INLINE MultiFrequencyPassBResult<Real> trace_pass_b_segment_model_multifrequency_pixel_metric(
    int pixel,
    const PassAParams<Real>& params,
    RadiationModel radiation_model,
    int radiation_substeps,
    const Real* frequencies,
    int nfreq,
    Stokes<Real>* observed_stokes,
    const Metric& metric) {
    MultiFrequencyPassBResult<Real> out;
    if (nfreq < 1 || nfreq > KPOLARIS_MAX_FREQUENCIES ||
        frequencies == nullptr || observed_stokes == nullptr) {
        out.pass_a.reason = TerminationReason::invalid_state;
        out.reason = TerminationReason::invalid_state;
        return out;
    }
    const TransportState<Real> camera_state =
        initialize_camera_ray(metric, pixel, params.camera);

    out.state = camera_state;
    out.pass_a.state = camera_state;
    out.pass_a.initial_null = metric.dot(out.state.x, out.state.k, out.state.k);
    out.pass_a.min_radius = radial_coordinate(metric, out.state.x);
    out.pass_a.final_radius = out.pass_a.min_radius;
    out.pass_a.reason = TerminationReason::max_steps;
    out.reason = TerminationReason::max_steps;

    AdaptiveRK4Control<Real> adaptive_control;
    adaptive_control.tolerance = params.adaptive_tolerance;
    adaptive_control.min_step = params.min_step;
    adaptive_control.max_step = params.max_step;
    Real h_current = -abs_val(params.step);
    bool entered_radiation_sphere = false;
    int steps_inside_or_after_entry = 0;
    for (int n = 0; n < params.max_steps; ++n) {
        if (!metric_time_domain_valid(metric, out.state.x)) {
            out.pass_a.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const Real r_bl = radial_coordinate(metric, out.state.x);
        out.pass_a.min_radius = min_val(out.pass_a.min_radius, r_bl);

        if (!entered_radiation_sphere) {
            if (r_bl <= params.outer_radius) {
                entered_radiation_sphere = true;
                steps_inside_or_after_entry = 0;
            }
        } else {
            if (metric_inner_surface_value(metric, out.state.x,
                                           params.inner_radius) <= Real(0)) {
                out.pass_a.reason = TerminationReason::reached_inner_boundary;
                break;
            }
            if (steps_inside_or_after_entry > 0 && r_bl >= params.outer_radius) {
                out.pass_a.reason = TerminationReason::escaped_domain;
                break;
            }
        }

        const TransportState<Real> old_state = out.state;
        const Real effective_min_step = metric_effective_min_step(
            metric, old_state.k, params.min_step);
        if (metric_time_domain_step_is_terminal(
                metric, old_state.x, old_state.k,
                h_current, effective_min_step)) {
            out.pass_a.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const Real old_r = r_bl;
        const Real old_inner_value = metric_inner_surface_value(
            metric, old_state.x, params.inner_radius);
        if (entered_radiation_sphere &&
            metric_inner_surface_step_is_terminal(
                metric, old_state.x, old_state.k,
                params.inner_radius, h_current, effective_min_step)) {
            out.pass_a.reason = TerminationReason::reached_inner_boundary;
            break;
        }
        Real used_h = metric_limit_time_domain_step(
            metric, old_state.x, old_state.k, h_current);
        used_h = metric_limit_inner_surface_step(
            metric, old_state.x, old_state.k, params.inner_radius, used_h);
        TransportState<Real> next_state;
        if (params.adaptive) {
            AdaptiveRK4Control<Real> trial_control = adaptive_control;
            trial_control.min_step = effective_min_step;
            if (abs_val(used_h) < trial_control.min_step) {
                trial_control.min_step = abs_val(used_h);
            }
            const auto step = adaptive_rk4_step(metric, old_state, used_h, trial_control);
            if (!step.accepted) {
                h_current = step.next_h;
                if (abs_val(h_current) <= effective_min_step * Real(1.0001)) {
                    out.pass_a.reason = TerminationReason::adaptive_step_underflow;
                    break;
                }
                continue;
            }
            next_state = step.state;
            used_h = step.used_h;
            h_current = step.next_h;
        } else {
            next_state = rk4_step(metric, out.state, used_h);
        }

        if (!metric_time_domain_valid(metric, next_state.x)) {
            out.pass_a.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const Real next_r = radial_coordinate(metric, next_state.x);
        const Real next_inner_value = metric_inner_surface_value(
            metric, next_state.x, params.inner_radius);
        if (!entered_radiation_sphere && next_r <= params.outer_radius) {
            entered_radiation_sphere = true;
            steps_inside_or_after_entry = 0;
        }
        else if (!entered_radiation_sphere && next_r >= old_r) {
            out.state = next_state;
            out.pass_a.path_length += abs_val(used_h);
            out.pass_a.steps += 1;
            out.pass_a.reason = TerminationReason::escaped_domain;
            break;
        }

        TerminationReason crossing_reason = TerminationReason::none;
        if (entered_radiation_sphere && old_inner_value > Real(0) &&
            next_inner_value <= Real(0)) {
            crossing_reason = TerminationReason::reached_inner_boundary;
        } else if (entered_radiation_sphere && steps_inside_or_after_entry > 0 &&
                   old_r < params.outer_radius && next_r >= params.outer_radius) {
            crossing_reason = TerminationReason::escaped_domain;
        }

        if (crossing_reason != TerminationReason::none) {
            const int inner_event =
                crossing_reason == TerminationReason::reached_inner_boundary;
            const Real surface_parameter = inner_event ?
                params.inner_radius : params.outer_radius;
            const MetricSurfaceEvent<Real> event = locate_metric_surface_crossing(
                metric, old_state, next_state, used_h, surface_parameter,
                inner_event, params.adaptive);
            next_state = event.state;
            used_h = event.used_h;
        }
        if (params.direct_only && truncate_at_first_vertical_turn(
                metric, old_state, next_state, used_h, params.adaptive)) {
            crossing_reason = TerminationReason::reached_direct_turn;
        }
        if (crossing_reason != TerminationReason::none) {
            out.state = next_state;
            out.pass_a.path_length += abs_val(used_h);
            out.pass_a.steps += 1;
            out.pass_a.reason = crossing_reason;
            break;
        }

        out.state = next_state;
        out.pass_a.path_length += abs_val(used_h);
        out.pass_a.steps += 1;
        if (entered_radiation_sphere) {
            steps_inside_or_after_entry += 1;
        }
    }

    out.pass_a.state = out.state;
    out.pass_a.final_radius = radial_coordinate(metric, out.state.x);
    out.pass_a.final_null = metric.dot(out.state.x, out.state.k, out.state.k);
    out.pass_a.frame_error = max_frame_error(frame_errors(metric, out.state));
    out.reason = out.pass_a.reason;

    if (out.pass_a.reason != TerminationReason::reached_direct_turn &&
        (!entered_radiation_sphere ||
         (out.pass_a.reason != TerminationReason::reached_inner_boundary &&
          out.pass_a.reason != TerminationReason::escaped_domain))) {
        out.closure_x = spatial_distance(out.state.x, camera_state.x);
        out.closure_k = vector_max_abs_difference(out.state.k, camera_state.k);
        out.final_null = out.pass_a.final_null;
        out.frame_error = out.pass_a.frame_error;
        return out;
    }

    const int screen_orientation = backward_screen_orientation_sign(params.camera);

    Stokes<Real> stokes[KPOLARIS_MAX_FREQUENCIES];
    for (int f = 0; f < nfreq; ++f) {
        stokes[f] = Stokes<Real>();
    }

    (void)radiation_substeps;
    AdaptiveRK4Control<Real> transfer_control = adaptive_control;
    int reached_camera = 0;
    h_current = params.step;
    Real radiation_step_cap = params.max_radiation_step > Real(0) ? params.max_radiation_step : params.max_step;
    while (!reached_camera && out.steps < params.max_steps) {
        if (!metric_time_domain_valid(metric, out.state.x)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        Real h = h_current;
        const TransportState<Real> old_state = out.state;
        const Real effective_min_step = metric_effective_min_step(
            metric, old_state.k, params.min_step);
        const Real start_r_bl = radial_coordinate(metric, old_state.x);
        if (start_r_bl <= params.outer_radius && radiation_step_cap > Real(0)) {
            h = min_val(h, radiation_step_cap);
        }
        if (metric_time_domain_step_is_terminal(
                metric, old_state.x, old_state.k,
                h, effective_min_step)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        h = metric_limit_time_domain_step(
            metric, old_state.x, old_state.k, h);
        TransportState<Real> next_state;
        TransportState<Real> sample_state;
        if (params.adaptive) {
            AdaptiveRK4Control<Real> trial_control = transfer_control;
            trial_control.min_step = effective_min_step;
            const auto proposed = adaptive_rk4_step(metric, old_state, h, trial_control);
            if (!proposed.accepted) {
                h_current = abs_val(proposed.next_h);
                if (h_current <= effective_min_step * Real(1.0001)) {
                    out.reason = TerminationReason::adaptive_step_underflow;
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

        if (!metric_time_domain_valid(metric, next_state.x)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
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
                    crossing_control.min_step = min_val(effective_min_step, abs_val(h));
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
                if (h <= effective_min_step*Real(1.0001)) {
                    out.reason = TerminationReason::adaptive_step_underflow;
                    break;
                }
                h_current = max_val(effective_min_step, h/Real(wedge_substeps));
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
        const Real local_max_radiation_step =
            (params.max_radiation_step > Real(0) &&
             (old_r_bl <= params.outer_radius || sample_r_bl <= params.outer_radius ||
              next_r_bl <= params.outer_radius)) ? params.max_radiation_step : Real(0);
        int required_steps = 1;
        TransferCoeffs<Real> coeffs[KPOLARIS_MAX_FREQUENCIES];
        Real dlambda_scales[KPOLARIS_MAX_FREQUENCIES];
        for (int f = 0; f < nfreq; ++f) {
            RadiationModel freq_model = radiation_model;
            freq_model.freq_cgs = frequencies[f];
            if (metric_inner_surface_value(metric, sample_state.x, params.inner_radius) > Real(0) &&
                sample_r_bl < params.outer_radius) {
                coeffs[f] = freq_model.coefficients(metric, sample_state, Real(0.5));
                apply_emission_selection(params.emission_selection, metric, sample_state, coeffs[f]);
            }
            transform_axial_coefficients_to_screen_orientation(coeffs[f],
                                                               screen_orientation);
            dlambda_scales[f] = freq_model.dlambda_scale();
            const int candidate = radiation_substep_count(
                coeffs[f], h, dlambda_scales[f], 1,
                local_max_radiation_step, params.max_radiation_depth,
                params.max_absorption_depth, params.max_faraday_depth);
            required_steps = candidate > required_steps ? candidate : required_steps;
        }
        if (required_steps > 1 && h > effective_min_step * Real(1.0001)) {
            h_current = max_val(effective_min_step, h / Real(required_steps));
            radiation_step_cap = h_current;
            continue;
        }
        if (local_max_radiation_step > Real(0) || params.max_radiation_depth > Real(0) ||
            params.max_absorption_depth > Real(0) || params.max_faraday_depth > Real(0)) {
            int predictive_steps = 1;
            for (int f = 0; f < nfreq; ++f) {
                const int candidate = radiation_substep_count(
                    coeffs[f], h_current, dlambda_scales[f], 1,
                    local_max_radiation_step, params.max_radiation_depth,
                params.max_absorption_depth, params.max_faraday_depth);
                predictive_steps = candidate > predictive_steps ? candidate : predictive_steps;
            }
            radiation_step_cap = predictive_steps > 1 ?
                max_val(effective_min_step, h_current / Real(predictive_steps)) : params.max_step;
        } else {
            radiation_step_cap = params.max_step;
        }
        if (h > Real(0)) {
            for (int f = 0; f < nfreq; ++f) {
                semi_analytic_stokes_step(stokes[f], coeffs[f],
                                          abs_val(h) * dlambda_scales[f]);
            }
        }
        out.state = next_state;
        out.steps += 1;
        if (crosses_camera) {
            reached_camera = 1;
        }
    }

    out.overlap = screen_overlap(metric, out.state.x, camera_state, out.state);
    for (int f = 0; f < nfreq; ++f) {
        observed_stokes[f] = transform_to_observer_basis(stokes[f], out.overlap);
    }
    out.closure_x = spatial_distance(out.state.x, camera_state.x);
    Vec4<Real> target_k;
    for (int mu = 0; mu < ndim; ++mu) {
        target_k[mu] = camera_state.k[mu];
    }
    out.closure_k = vector_max_abs_difference(out.state.k, target_k);
    out.final_null = metric.dot(out.state.x, out.state.k, out.state.k);
    out.frame_error = max_frame_error(frame_errors(metric, out.state));
    if (reached_camera) {
        out.reason = TerminationReason::reached_camera;
    } else if (out.reason != TerminationReason::adaptive_step_underflow &&
               out.reason != TerminationReason::metric_time_exhausted) {
        out.reason = TerminationReason::max_steps;
    }
    return out;
}
// Keep the safe chunked path separate so the default fused kernel can reuse
// per-frequency coefficients without paying the extra full-control loop.
template<class Metric, class Real, class RadiationModel>
KPOLARIS_INLINE MultiFrequencyPassBResult<Real> trace_pass_b_segment_model_multifrequency_control_pixel_metric(
    int pixel,
    const PassAParams<Real>& params,
    RadiationModel radiation_model,
    int radiation_substeps,
    const Real* frequencies,
    int nfreq,
    const Real* step_control_frequencies,
    int step_control_nfreq,
    Stokes<Real>* observed_stokes,
    const Metric& metric) {
    MultiFrequencyPassBResult<Real> out;
    if (nfreq < 1 || nfreq > KPOLARIS_MAX_FREQUENCIES ||
        step_control_nfreq < 1 || frequencies == nullptr ||
        step_control_frequencies == nullptr || observed_stokes == nullptr) {
        out.pass_a.reason = TerminationReason::invalid_state;
        out.reason = TerminationReason::invalid_state;
        return out;
    }
    const TransportState<Real> camera_state =
        initialize_camera_ray(metric, pixel, params.camera);

    out.state = camera_state;
    out.pass_a.state = camera_state;
    out.pass_a.initial_null = metric.dot(out.state.x, out.state.k, out.state.k);
    out.pass_a.min_radius = radial_coordinate(metric, out.state.x);
    out.pass_a.final_radius = out.pass_a.min_radius;
    out.pass_a.reason = TerminationReason::max_steps;
    out.reason = TerminationReason::max_steps;

    AdaptiveRK4Control<Real> adaptive_control;
    adaptive_control.tolerance = params.adaptive_tolerance;
    adaptive_control.min_step = params.min_step;
    adaptive_control.max_step = params.max_step;
    Real h_current = -abs_val(params.step);
    bool entered_radiation_sphere = false;
    int steps_inside_or_after_entry = 0;
    for (int n = 0; n < params.max_steps; ++n) {
        if (!metric_time_domain_valid(metric, out.state.x)) {
            out.pass_a.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const Real r_bl = radial_coordinate(metric, out.state.x);
        out.pass_a.min_radius = min_val(out.pass_a.min_radius, r_bl);

        if (!entered_radiation_sphere) {
            if (r_bl <= params.outer_radius) {
                entered_radiation_sphere = true;
                steps_inside_or_after_entry = 0;
            }
        } else {
            if (metric_inner_surface_value(metric, out.state.x,
                                           params.inner_radius) <= Real(0)) {
                out.pass_a.reason = TerminationReason::reached_inner_boundary;
                break;
            }
            if (steps_inside_or_after_entry > 0 && r_bl >= params.outer_radius) {
                out.pass_a.reason = TerminationReason::escaped_domain;
                break;
            }
        }

        const TransportState<Real> old_state = out.state;
        const Real effective_min_step = metric_effective_min_step(
            metric, old_state.k, params.min_step);
        if (metric_time_domain_step_is_terminal(
                metric, old_state.x, old_state.k,
                h_current, effective_min_step)) {
            out.pass_a.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const Real old_r = r_bl;
        const Real old_inner_value = metric_inner_surface_value(
            metric, old_state.x, params.inner_radius);
        if (entered_radiation_sphere &&
            metric_inner_surface_step_is_terminal(
                metric, old_state.x, old_state.k,
                params.inner_radius, h_current, effective_min_step)) {
            out.pass_a.reason = TerminationReason::reached_inner_boundary;
            break;
        }
        Real used_h = metric_limit_time_domain_step(
            metric, old_state.x, old_state.k, h_current);
        used_h = metric_limit_inner_surface_step(
            metric, old_state.x, old_state.k, params.inner_radius, used_h);
        TransportState<Real> next_state;
        if (params.adaptive) {
            AdaptiveRK4Control<Real> trial_control = adaptive_control;
            trial_control.min_step = effective_min_step;
            if (abs_val(used_h) < trial_control.min_step) {
                trial_control.min_step = abs_val(used_h);
            }
            const auto step = adaptive_rk4_step(metric, old_state, used_h, trial_control);
            if (!step.accepted) {
                h_current = step.next_h;
                if (abs_val(h_current) <= effective_min_step * Real(1.0001)) {
                    out.pass_a.reason = TerminationReason::adaptive_step_underflow;
                    break;
                }
                continue;
            }
            next_state = step.state;
            used_h = step.used_h;
            h_current = step.next_h;
        } else {
            next_state = rk4_step(metric, out.state, used_h);
        }

        if (!metric_time_domain_valid(metric, next_state.x)) {
            out.pass_a.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const Real next_r = radial_coordinate(metric, next_state.x);
        const Real next_inner_value = metric_inner_surface_value(
            metric, next_state.x, params.inner_radius);
        if (!entered_radiation_sphere && next_r <= params.outer_radius) {
            entered_radiation_sphere = true;
            steps_inside_or_after_entry = 0;
        }
        else if (!entered_radiation_sphere && next_r >= old_r) {
            out.state = next_state;
            out.pass_a.path_length += abs_val(used_h);
            out.pass_a.steps += 1;
            out.pass_a.reason = TerminationReason::escaped_domain;
            break;
        }

        TerminationReason crossing_reason = TerminationReason::none;
        if (entered_radiation_sphere && old_inner_value > Real(0) &&
            next_inner_value <= Real(0)) {
            crossing_reason = TerminationReason::reached_inner_boundary;
        } else if (entered_radiation_sphere && steps_inside_or_after_entry > 0 &&
                   old_r < params.outer_radius && next_r >= params.outer_radius) {
            crossing_reason = TerminationReason::escaped_domain;
        }

        if (crossing_reason != TerminationReason::none) {
            const int inner_event =
                crossing_reason == TerminationReason::reached_inner_boundary;
            const Real surface_parameter = inner_event ?
                params.inner_radius : params.outer_radius;
            const MetricSurfaceEvent<Real> event = locate_metric_surface_crossing(
                metric, old_state, next_state, used_h, surface_parameter,
                inner_event, params.adaptive);
            next_state = event.state;
            used_h = event.used_h;
        }
        if (params.direct_only && truncate_at_first_vertical_turn(
                metric, old_state, next_state, used_h, params.adaptive)) {
            crossing_reason = TerminationReason::reached_direct_turn;
        }
        if (crossing_reason != TerminationReason::none) {
            out.state = next_state;
            out.pass_a.path_length += abs_val(used_h);
            out.pass_a.steps += 1;
            out.pass_a.reason = crossing_reason;
            break;
        }

        out.state = next_state;
        out.pass_a.path_length += abs_val(used_h);
        out.pass_a.steps += 1;
        if (entered_radiation_sphere) {
            steps_inside_or_after_entry += 1;
        }
    }

    out.pass_a.state = out.state;
    out.pass_a.final_radius = radial_coordinate(metric, out.state.x);
    out.pass_a.final_null = metric.dot(out.state.x, out.state.k, out.state.k);
    out.pass_a.frame_error = max_frame_error(frame_errors(metric, out.state));
    out.reason = out.pass_a.reason;

    if (out.pass_a.reason != TerminationReason::reached_direct_turn &&
        (!entered_radiation_sphere ||
         (out.pass_a.reason != TerminationReason::reached_inner_boundary &&
          out.pass_a.reason != TerminationReason::escaped_domain))) {
        out.closure_x = spatial_distance(out.state.x, camera_state.x);
        out.closure_k = vector_max_abs_difference(out.state.k, camera_state.k);
        out.final_null = out.pass_a.final_null;
        out.frame_error = out.pass_a.frame_error;
        return out;
    }

    const int screen_orientation = backward_screen_orientation_sign(params.camera);

    Stokes<Real> stokes[KPOLARIS_MAX_FREQUENCIES];
    for (int f = 0; f < nfreq; ++f) {
        stokes[f] = Stokes<Real>();
    }

    (void)radiation_substeps;
    AdaptiveRK4Control<Real> transfer_control = adaptive_control;
    int reached_camera = 0;
    h_current = params.step;
    Real radiation_step_cap = params.max_radiation_step > Real(0) ? params.max_radiation_step : params.max_step;
    while (!reached_camera && out.steps < params.max_steps) {
        if (!metric_time_domain_valid(metric, out.state.x)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        Real h = h_current;
        const TransportState<Real> old_state = out.state;
        const Real effective_min_step = metric_effective_min_step(
            metric, old_state.k, params.min_step);
        const Real start_r_bl = radial_coordinate(metric, old_state.x);
        if (start_r_bl <= params.outer_radius && radiation_step_cap > Real(0)) {
            h = min_val(h, radiation_step_cap);
        }
        if (metric_time_domain_step_is_terminal(
                metric, old_state.x, old_state.k,
                h, effective_min_step)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        h = metric_limit_time_domain_step(
            metric, old_state.x, old_state.k, h);
        TransportState<Real> next_state;
        TransportState<Real> sample_state;
        if (params.adaptive) {
            AdaptiveRK4Control<Real> trial_control = transfer_control;
            trial_control.min_step = effective_min_step;
            const auto proposed = adaptive_rk4_step(metric, old_state, h, trial_control);
            if (!proposed.accepted) {
                h_current = abs_val(proposed.next_h);
                if (h_current <= effective_min_step * Real(1.0001)) {
                    out.reason = TerminationReason::adaptive_step_underflow;
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

        if (!metric_time_domain_valid(metric, next_state.x)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
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
                    crossing_control.min_step = min_val(effective_min_step, abs_val(h));
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
                if (h <= effective_min_step*Real(1.0001)) {
                    out.reason = TerminationReason::adaptive_step_underflow;
                    break;
                }
                h_current = max_val(effective_min_step, h/Real(wedge_substeps));
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
        const Real local_max_radiation_step =
            (params.max_radiation_step > Real(0) &&
             (old_r_bl <= params.outer_radius || sample_r_bl <= params.outer_radius ||
              next_r_bl <= params.outer_radius)) ? params.max_radiation_step : Real(0);
        int required_steps = 1;
        for (int f = 0; f < step_control_nfreq; ++f) {
            RadiationModel freq_model = radiation_model;
            freq_model.freq_cgs = step_control_frequencies[f];
            TransferCoeffs<Real> coeffs;
            if (metric_inner_surface_value(metric, sample_state.x, params.inner_radius) > Real(0) &&
                sample_r_bl < params.outer_radius) {
                coeffs = freq_model.coefficients(metric, sample_state, Real(0.5));
                apply_emission_selection(params.emission_selection, metric, sample_state, coeffs);
            }
            transform_axial_coefficients_to_screen_orientation(coeffs, screen_orientation);
            const Real dlambda_scale = freq_model.dlambda_scale();
            const int candidate = radiation_substep_count(
                coeffs, h, dlambda_scale, 1,
                local_max_radiation_step, params.max_radiation_depth,
                params.max_absorption_depth, params.max_faraday_depth);
            required_steps = candidate > required_steps ? candidate : required_steps;
        }
        if (required_steps > 1 && h > effective_min_step * Real(1.0001)) {
            h_current = max_val(effective_min_step, h / Real(required_steps));
            radiation_step_cap = h_current;
            continue;
        }
        if (local_max_radiation_step > Real(0) || params.max_radiation_depth > Real(0) ||
            params.max_absorption_depth > Real(0) || params.max_faraday_depth > Real(0)) {
            int predictive_steps = 1;
            for (int f = 0; f < step_control_nfreq; ++f) {
                RadiationModel freq_model = radiation_model;
                freq_model.freq_cgs = step_control_frequencies[f];
                TransferCoeffs<Real> coeffs;
                if (metric_inner_surface_value(metric, sample_state.x, params.inner_radius) > Real(0) &&
                    sample_r_bl < params.outer_radius) {
                    coeffs = freq_model.coefficients(metric, sample_state, Real(0.5));
                    apply_emission_selection(params.emission_selection, metric, sample_state, coeffs);
                }
                transform_axial_coefficients_to_screen_orientation(coeffs, screen_orientation);
                const Real dlambda_scale = freq_model.dlambda_scale();
                const int candidate = radiation_substep_count(
                    coeffs, h_current, dlambda_scale, 1,
                    local_max_radiation_step, params.max_radiation_depth,
                    params.max_absorption_depth, params.max_faraday_depth);
                predictive_steps = candidate > predictive_steps ? candidate : predictive_steps;
            }
            radiation_step_cap = predictive_steps > 1 ?
                max_val(effective_min_step, h_current / Real(predictive_steps)) : params.max_step;
        } else {
            radiation_step_cap = params.max_step;
        }
        if (h > Real(0)) {
            for (int f = 0; f < nfreq; ++f) {
                RadiationModel freq_model = radiation_model;
                freq_model.freq_cgs = frequencies[f];
                TransferCoeffs<Real> coeffs;
                if (metric_inner_surface_value(metric, sample_state.x, params.inner_radius) > Real(0) &&
                    sample_r_bl < params.outer_radius) {
                    coeffs = freq_model.coefficients(metric, sample_state, Real(0.5));
                    apply_emission_selection(params.emission_selection, metric, sample_state, coeffs);
                }
                transform_axial_coefficients_to_screen_orientation(coeffs, screen_orientation);
                semi_analytic_stokes_step(stokes[f], coeffs,
                                          abs_val(h) * freq_model.dlambda_scale());
            }
        }
        out.state = next_state;
        out.steps += 1;
        if (crosses_camera) {
            reached_camera = 1;
        }
    }

    out.overlap = screen_overlap(metric, out.state.x, camera_state, out.state);
    for (int f = 0; f < nfreq; ++f) {
        observed_stokes[f] = transform_to_observer_basis(stokes[f], out.overlap);
    }
    out.closure_x = spatial_distance(out.state.x, camera_state.x);
    Vec4<Real> target_k;
    for (int mu = 0; mu < ndim; ++mu) {
        target_k[mu] = camera_state.k[mu];
    }
    out.closure_k = vector_max_abs_difference(out.state.k, target_k);
    out.final_null = metric.dot(out.state.x, out.state.k, out.state.k);
    out.frame_error = max_frame_error(frame_errors(metric, out.state));
    if (reached_camera) {
        out.reason = TerminationReason::reached_camera;
    } else if (out.reason != TerminationReason::adaptive_step_underflow &&
               out.reason != TerminationReason::metric_time_exhausted) {
        out.reason = TerminationReason::max_steps;
    }
    return out;
}

template<class Real, class RadiationModel>
KPOLARIS_INLINE MultiFrequencyPassBResult<Real> trace_pass_b_segment_model_multifrequency_pixel(
    int pixel,
    const PassAParams<Real>& params,
    const RadiationModel& radiation_model,
    int radiation_substeps,
    const Real* frequencies,
    int nfreq,
    Stokes<Real>* observed_stokes) {
    if (params.coordinate_system == CoordinateSystem::BoyerLindquist) {
        const KerrBoyerLindquistMetric<Real> metric(params.mass, params.spin);
        return trace_pass_b_segment_model_multifrequency_pixel_metric(
            pixel, params, radiation_model, radiation_substeps, frequencies,
            nfreq, observed_stokes, metric);
    }
    if (params.coordinate_system == CoordinateSystem::SphericalKS) {
        const KerrSchildSphericalMetric<Real> metric(params.mass, params.spin);
        return trace_pass_b_segment_model_multifrequency_pixel_metric(
            pixel, params, radiation_model, radiation_substeps, frequencies,
            nfreq, observed_stokes, metric);
    }
    if (params.coordinate_system == CoordinateSystem::MKS) {
        const KerrSchildSphericalMetric<Real> metric(params.mass, params.spin);
        return trace_pass_b_segment_model_multifrequency_pixel_metric(
            pixel, params, radiation_model, radiation_substeps, frequencies,
            nfreq, observed_stokes, metric);
    }
    if (params.coordinate_system == CoordinateSystem::FMKS) {
        const KerrFMKSMetric<Real> metric(params.mass, params.spin, params.fmks_startx1,
                                         params.fmks_hslope, params.fmks_mks_smooth,
                                         params.fmks_poly_alpha, params.fmks_poly_xt,
                                         params.fmks_poly_norm);
        return trace_pass_b_segment_model_multifrequency_pixel_metric(
            pixel, params, radiation_model, radiation_substeps, frequencies,
            nfreq, observed_stokes, metric);
    }
    const KerrSchildInMetric<Real> metric(params.mass, params.spin);
    return trace_pass_b_segment_model_multifrequency_pixel_metric(
        pixel, params, radiation_model, radiation_substeps, frequencies,
        nfreq, observed_stokes, metric);
}


template<class Real, class RadiationModel>
KPOLARIS_INLINE PassBResult<Real> trace_pass_b_segment_model_pixel(int pixel,
                                                                const PassAParams<Real>& params,
                                                                const RadiationModel& radiation_model,
                                                                int radiation_substeps) {
    if (params.coordinate_system == CoordinateSystem::BoyerLindquist) {
        const KerrBoyerLindquistMetric<Real> metric(params.mass, params.spin);
        return trace_pass_b_segment_model_pixel_metric(pixel, params, radiation_model,
                                                       radiation_substeps, metric);
    }
    if (params.coordinate_system == CoordinateSystem::SphericalKS) {
        const KerrSchildSphericalMetric<Real> metric(params.mass, params.spin);
        return trace_pass_b_segment_model_pixel_metric(pixel, params, radiation_model,
                                                       radiation_substeps, metric);
    }
    if (params.coordinate_system == CoordinateSystem::MKS) {
        const KerrSchildSphericalMetric<Real> metric(params.mass, params.spin);
        return trace_pass_b_segment_model_pixel_metric(pixel, params, radiation_model,
                                                       radiation_substeps, metric);
    }
    if (params.coordinate_system == CoordinateSystem::FMKS) {
        const KerrFMKSMetric<Real> metric(params.mass, params.spin, params.fmks_startx1,
                                         params.fmks_hslope, params.fmks_mks_smooth,
                                         params.fmks_poly_alpha, params.fmks_poly_xt,
                                         params.fmks_poly_norm);
        return trace_pass_b_segment_model_pixel_metric(pixel, params, radiation_model,
                                                       radiation_substeps, metric);
    }
    const KerrSchildInMetric<Real> metric(params.mass, params.spin);
    return trace_pass_b_segment_model_pixel_metric(pixel, params, radiation_model,
                                                   radiation_substeps, metric);
}

template<class Metric, class Real>
KPOLARIS_INLINE PassBResult<Real> trace_pass_b_pixel_metric(int pixel,
                                                         const PassBParams<Real>& params,
                                                         const Metric& metric) {
    PassBResult<Real> out;
    out.pass_a = trace_pass_a_pixel_metric(pixel, params.pass_a, metric);
    out.state = out.pass_a.state;
    out.reason = out.pass_a.reason;

    const TransportState<Real> camera_state =
        initialize_camera_ray(metric, pixel, params.pass_a.camera);

    if (out.pass_a.reason != TerminationReason::reached_inner_boundary &&
        out.pass_a.reason != TerminationReason::reached_direct_turn) {
        out.closure_x = spatial_distance(out.state.x, camera_state.x);
        out.closure_k = vector_max_abs_difference(out.state.k, camera_state.k);
        out.final_null = metric.dot(out.state.x, out.state.k, out.state.k);
        out.frame_error = max_frame_error(frame_errors(metric, out.state));
        return out;
    }

    // Switch to positive affine steps while retaining the physical wavevector
    // and the camera screen basis transported to this endpoint by Pass A.
    const int screen_orientation =
        backward_screen_orientation_sign(params.pass_a.camera);

    Stokes<Real> stokes;
    TransferCoeffs<Real> coeffs = params.coeffs;
    transform_axial_coefficients_to_screen_orientation(coeffs,
                                                       screen_orientation);
    const Real nominal_h = params.pass_a.step;
    for (int n = 0; n < out.pass_a.steps; ++n) {
        Real h = nominal_h;
        if (!metric_time_domain_valid(metric, out.state.x)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const Real effective_min_step = metric_effective_min_step(
            metric, out.state.k, params.pass_a.min_step);
        if (metric_time_domain_step_is_terminal(
                metric, out.state.x, out.state.k,
                h, effective_min_step)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        h = metric_limit_time_domain_step(
            metric, out.state.x, out.state.k, h);
        const TransportState<Real> next_state =
            rk4_step(metric, out.state, h);
        if (!metric_time_domain_valid(metric, next_state.x)) {
            out.reason = TerminationReason::metric_time_exhausted;
            break;
        }
        out.state = next_state;
        const int nrad = radiation_substep_count(
            coeffs, h, Real(1), params.radiation_substeps,
            params.pass_a.max_radiation_step, params.pass_a.max_radiation_depth,
            params.pass_a.max_absorption_depth, params.pass_a.max_faraday_depth);
        radiation_step_with_substeps(stokes, coeffs, abs_val(h), nrad);
        out.steps += 1;
    }

    out.propagated_stokes = stokes;
    out.overlap = screen_overlap(metric, out.state.x, camera_state, out.state);
    out.observed_stokes = transform_to_observer_basis(stokes, out.overlap);
    out.closure_x = spatial_distance(out.state.x, camera_state.x);
    Vec4<Real> target_k;
    for (int mu = 0; mu < ndim; ++mu) {
        target_k[mu] = camera_state.k[mu];
    }
    out.closure_k = vector_max_abs_difference(out.state.k, target_k);
    out.final_null = metric.dot(out.state.x, out.state.k, out.state.k);
    out.frame_error = max_frame_error(frame_errors(metric, out.state));
    if (out.reason != TerminationReason::metric_time_exhausted) {
        out.reason = TerminationReason::reached_camera;
    }
    return out;
}

template<class Real>
KPOLARIS_INLINE PassBResult<Real> trace_pass_b_pixel(int pixel,
                                                  const PassBParams<Real>& params) {
    if (params.pass_a.coordinate_system == CoordinateSystem::BoyerLindquist) {
        const KerrBoyerLindquistMetric<Real> metric(params.pass_a.mass, params.pass_a.spin);
        return trace_pass_b_pixel_metric(pixel, params, metric);
    }
    if (params.pass_a.coordinate_system == CoordinateSystem::SphericalKS) {
        const KerrSchildSphericalMetric<Real> metric(params.pass_a.mass, params.pass_a.spin);
        return trace_pass_b_pixel_metric(pixel, params, metric);
    }
    if (params.pass_a.coordinate_system == CoordinateSystem::MKS) {
        const KerrSchildSphericalMetric<Real> metric(params.pass_a.mass, params.pass_a.spin);
        return trace_pass_b_pixel_metric(pixel, params, metric);
    }
    if (params.pass_a.coordinate_system == CoordinateSystem::FMKS) {
        const KerrFMKSMetric<Real> metric(params.pass_a.mass, params.pass_a.spin, params.pass_a.fmks_startx1,
                                         params.pass_a.fmks_hslope, params.pass_a.fmks_mks_smooth,
                                         params.pass_a.fmks_poly_alpha, params.pass_a.fmks_poly_xt,
                                         params.pass_a.fmks_poly_norm);
        return trace_pass_b_pixel_metric(pixel, params, metric);
    }
    const KerrSchildInMetric<Real> metric(params.pass_a.mass, params.pass_a.spin);
    return trace_pass_b_pixel_metric(pixel, params, metric);
}

} // namespace kpolaris
