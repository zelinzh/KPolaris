#pragma once

#include <Kokkos_Core.hpp>

#include "geodesic/pass_b.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct TraceConfig {
    int first_pixel = 0;
    int ray_count = 1;
    int max_samples = 1024;
    int sample_stride = 1;
    int record_lambda = 1;
    int record_coords = 1;
    int record_plasma = 1;
    int record_x = 0;
    int record_k = 0;
    int record_e1 = 0;
    int record_e2 = 0;
    int record_coeffs = 1;
    int record_stokes = 1;
};

template<class StoreReal, class ExecSpace>
struct TraceViews {
    Kokkos::View<int*, ExecSpace> pixel;
    Kokkos::View<int*, ExecSpace> sample_count;
    Kokkos::View<int*, ExecSpace> pass_a_steps;
    Kokkos::View<int*, ExecSpace> pass_b_steps;
    Kokkos::View<int*, ExecSpace> reason;
    Kokkos::View<StoreReal*, ExecSpace> closure_x;
    Kokkos::View<StoreReal*, ExecSpace> closure_k;
    Kokkos::View<StoreReal*, ExecSpace> final_null;
    Kokkos::View<StoreReal*, ExecSpace> frame_error;
    Kokkos::View<StoreReal*, ExecSpace> final_propagated_i;
    Kokkos::View<StoreReal*, ExecSpace> final_propagated_q;
    Kokkos::View<StoreReal*, ExecSpace> final_propagated_u;
    Kokkos::View<StoreReal*, ExecSpace> final_propagated_v;
    Kokkos::View<StoreReal*, ExecSpace> final_observed_i;
    Kokkos::View<StoreReal*, ExecSpace> final_observed_q;
    Kokkos::View<StoreReal*, ExecSpace> final_observed_u;
    Kokkos::View<StoreReal*, ExecSpace> final_observed_v;

    Kokkos::View<StoreReal**, ExecSpace> lambda;
    Kokkos::View<StoreReal**, ExecSpace> dlambda;
    Kokkos::View<StoreReal**, ExecSpace> r;
    Kokkos::View<StoreReal**, ExecSpace> theta;
    Kokkos::View<StoreReal**, ExecSpace> phi;
    Kokkos::View<StoreReal**, ExecSpace> ne_cgs;
    Kokkos::View<StoreReal**, ExecSpace> thetae;
    Kokkos::View<StoreReal**, ExecSpace> b_cgs;
    Kokkos::View<StoreReal**, ExecSpace> beta;
    Kokkos::View<StoreReal**, ExecSpace> sigma;
    Kokkos::View<StoreReal**, ExecSpace> nu_fluid_hz;
    Kokkos::View<StoreReal**, ExecSpace> theta_bk;
    Kokkos::View<StoreReal**, ExecSpace> b1;
    Kokkos::View<StoreReal**, ExecSpace> b2;
    Kokkos::View<StoreReal**, ExecSpace> cos2chi;
    Kokkos::View<StoreReal**, ExecSpace> sin2chi;
    Kokkos::View<StoreReal**, ExecSpace> x0;
    Kokkos::View<StoreReal**, ExecSpace> x1;
    Kokkos::View<StoreReal**, ExecSpace> x2;
    Kokkos::View<StoreReal**, ExecSpace> x3;
    Kokkos::View<StoreReal**, ExecSpace> k0;
    Kokkos::View<StoreReal**, ExecSpace> k1;
    Kokkos::View<StoreReal**, ExecSpace> k2;
    Kokkos::View<StoreReal**, ExecSpace> k3;
    Kokkos::View<StoreReal**, ExecSpace> e10;
    Kokkos::View<StoreReal**, ExecSpace> e11;
    Kokkos::View<StoreReal**, ExecSpace> e12;
    Kokkos::View<StoreReal**, ExecSpace> e13;
    Kokkos::View<StoreReal**, ExecSpace> e20;
    Kokkos::View<StoreReal**, ExecSpace> e21;
    Kokkos::View<StoreReal**, ExecSpace> e22;
    Kokkos::View<StoreReal**, ExecSpace> e23;
    Kokkos::View<StoreReal**, ExecSpace> jI;
    Kokkos::View<StoreReal**, ExecSpace> jQ;
    Kokkos::View<StoreReal**, ExecSpace> jU;
    Kokkos::View<StoreReal**, ExecSpace> jV;
    Kokkos::View<StoreReal**, ExecSpace> aI;
    Kokkos::View<StoreReal**, ExecSpace> aQ;
    Kokkos::View<StoreReal**, ExecSpace> aU;
    Kokkos::View<StoreReal**, ExecSpace> aV;
    Kokkos::View<StoreReal**, ExecSpace> rQ;
    Kokkos::View<StoreReal**, ExecSpace> rU;
    Kokkos::View<StoreReal**, ExecSpace> rV;
    Kokkos::View<StoreReal**, ExecSpace> SI;
    Kokkos::View<StoreReal**, ExecSpace> SQ;
    Kokkos::View<StoreReal**, ExecSpace> SU;
    Kokkos::View<StoreReal**, ExecSpace> SV;
};

template<class StoreReal, class View, class Real>
KPOLARIS_INLINE void trace_store(View view, int ray, int sample, Real value) {
    if (view.extent(0) > 0) {
        view(ray, sample) = static_cast<StoreReal>(value);
    }
}

template<class Real>
struct PlasmaTraceDiagnostics {
    Real ne_cgs = Real(0);
    Real thetae = Real(0);
    Real b_cgs = Real(0);
    Real beta = Real(0);
    Real sigma = Real(0);
    Real nu_fluid_hz = Real(0);
    Real theta_bk = Real(0);
    Real b1 = Real(0);
    Real b2 = Real(0);
    Real cos2chi = Real(0);
    Real sin2chi = Real(0);
};

template<class Metric, class Real>
KPOLARIS_INLINE PlasmaTraceDiagnostics<Real> trace_plasma_diagnostics(
    const Metric& metric,
    const RIAFAnalyticRadiationModel<Real>& model,
    const TransportState<Real>& state) {
    PlasmaTraceDiagnostics<Real> out;
    Real r = Real(0), th = Real(0), cosphi = Real(1), sinphi = Real(0);
    model.bl_coordinates_for_metric(metric, state.x, r, th, cosphi, sinphi);
    const Real r_bl = max_val(r, model.min_radius);
    const Real rh = metric.horizon_radius();
    if (r_bl <= rh + model.horizon_buffer || r_bl < model.r_min || r_bl > model.r_max) {
        return out;
    }
    const Real ne_norm = model.density_profile_bl(r_bl, th);
    if (!(ne_norm > Real(0))) {
        return out;
    }
    out.ne_cgs = ne_norm * model.ne_unit;
    out.thetae = model.thetae_profile(r_bl);
    out.b_cgs = model.magnetic_field_cgs(out.ne_cgs, r_bl);
    const Vec4<Real> ucon = model.fluid_four_velocity(metric, state.x, r_bl);
    const Vec4<Real> ucov = model.lower_vector(metric, state.x, ucon);
    Real uk = Real(0);
    for (int mu = 0; mu < ndim; ++mu) {
        uk += ucov[mu] * state.k[mu];
    }
    const Real nu_scale = max_val(abs_val(-uk), model.min_frequency_scale);
    out.nu_fluid_hz = nu_scale * model.freq_cgs;
    const Vec4<Real> bcon = model.magnetic_unit_four_vector(metric, state.x, ucon);
    const Vec4<Real> bcov = model.lower_vector(metric, state.x, bcon);
    Real kdotb = Real(0);
    for (int mu = 0; mu < ndim; ++mu) {
        kdotb += state.k[mu] * bcov[mu];
    }
    const Real cos_theta = clamp(kdotb / max_val(nu_scale, Real(1e-30)), Real(-1), Real(1));
    out.theta_bk = Kokkos::acos(cos_theta);
    out.b1 = model.screen_inner_product(metric, state.x, ucon, state.k, state.e1, bcon);
    out.b2 = model.screen_inner_product(metric, state.x, ucon, state.k, state.e2, bcon);
    const Real bproj2 = out.b1 * out.b1 + out.b2 * out.b2;
    if (bproj2 > Real(1e-30)) {
        out.cos2chi = (out.b2 * out.b2 - out.b1 * out.b1) / bproj2;
        out.sin2chi = -Real(2) * out.b1 * out.b2 / bproj2;
    }
    return out;
}

template<class Metric, class Real>
KPOLARIS_INLINE PlasmaTraceDiagnostics<Real> trace_plasma_diagnostics(
    const Metric& metric,
    const MagnetizedTorusRadiationModel<Real>& model,
    const TransportState<Real>& state) {
    PlasmaTraceDiagnostics<Real> out;
    Real r = Real(0), th = Real(0), cosphi = Real(1), sinphi = Real(0);
    model.bl_coordinates_for_metric(metric, state.x, r, th, cosphi, sinphi);
    if (r <= metric.horizon_radius() * Real(1.05) || (model.r_outer > Real(0) && r > model.r_outer * Real(1.15))) {
        return out;
    }
    const Real gtt = model.bl_gtt(r, th);
    const Real gtphi = model.bl_gtphi(r, th);
    const Real gphiphi = model.bl_gphiphi(r, th);
    const Real d2 = gtt * model.l0 * model.l0 + Real(2) * gtphi * model.l0 + gphiphi;
    const Real Wpot = model.potential(r, th);
    if (!(Wpot <= model.Win && Wpot >= model.Wc) || !(d2 > Real(0)) || !(model.KK > Real(0))) {
        return out;
    }
    const Real omega_density = Kokkos::pow((model.Win - Wpot) * ((model.kappa - Real(1)) / model.kappa) /
                                           ((Real(1) + Real(1) / model.beta) * model.KK),
                                           Real(1) / (model.kappa - Real(1)));
    if (!(omega_density > Real(1e-12))) {
        return out;
    }
    const Real p_code = model.KK * Kokkos::pow(omega_density, model.kappa);
    const Real pm_code = p_code / model.beta;
    const Real rho_code = omega_density - model.kappa / (model.kappa - Real(1)) * p_code;
    if (!(rho_code > Real(0) && p_code > Real(0) && pm_code > Real(0))) {
        return out;
    }
    const Real Omega = -(gtt * model.l0 + gtphi) / (gtphi * model.l0 + gphiphi);
    const Real u0 = Kokkos::sqrt(max_val(-Real(1) /
                         (gtt + Real(2) * gtphi * Omega + gphiphi * Omega * Omega), Real(0)));
    const Real u3 = Omega * u0;
    const Vec4<Real> ucon = model.axisymmetric_bl_vector_to_metric(metric, u0, u3, r, th, cosphi, sinphi);
    const Real b3 = Kokkos::sqrt(max_val(Real(2) * pm_code / d2, Real(0)));
    const Real b0 = model.l0 * b3;
    const Vec4<Real> bcon = model.axisymmetric_bl_vector_to_metric(metric, b0, b3, r, th, cosphi, sinphi);
    const Real bsq_geom = max_val(metric.dot(state.x, bcon, bcon), Real(0));
    if (!(bsq_geom > Real(0))) {
        return out;
    }
    const Real bnorm_geom = Kokkos::sqrt(bsq_geom);
    const Vec4<Real> bunit = bcon * (Real(1) / bnorm_geom);
    const Real rho_cgs = rho_code * model.density_unit_cgs();
    const Real internal_energy_cgs = p_code / (model.kappa - Real(1)) * model.pressure_unit_cgs();
    const Real mp = Real(1.67262192369e-24);
    const Real me = Real(9.1093837015e-28);
    const Real cl = Real(2.99792458e10);
    const Real kbol = Real(1.380649e-16);
    out.ne_cgs = rho_cgs / mp;
    const Real temperature = Real(2) * mp * internal_energy_cgs /
                             max_val(Real(3) * kbol * rho_cgs * (Real(2) + model.Rhigh), Real(1e-300));
    out.thetae = max_val(kbol * temperature / (me * cl * cl), model.thetae_min);
    out.b_cgs = bnorm_geom * model.bfield_unit_cgs();
    out.beta = p_code / max_val(pm_code, Real(1e-300));
    out.sigma = Real(2) * pm_code / max_val(rho_code, Real(1e-300));
    const Vec4<Real> ucov = model.lower_vector(metric, state.x, ucon);
    Real uk = Real(0);
    for (int mu = 0; mu < ndim; ++mu) {
        uk += ucov[mu] * state.k[mu];
    }
    const Real nu_scale = max_val(abs_val(-uk) / max_val(model.observer_energy_scale, Real(1e-300)), Real(1e-8));
    out.nu_fluid_hz = model.freq_cgs * nu_scale;
    const Vec4<Real> bcov = model.lower_vector(metric, state.x, bunit);
    Real kdotb = Real(0);
    for (int mu = 0; mu < ndim; ++mu) {
        kdotb += state.k[mu] * bcov[mu];
    }
    const Real cos_theta = clamp(kdotb / max_val(abs_val(-uk), Real(1e-300)), Real(-1), Real(1));
    out.theta_bk = Kokkos::acos(cos_theta);
    out.b1 = model.screen_inner_product(metric, state.x, ucon, state.k, state.e1, bunit);
    out.b2 = model.screen_inner_product(metric, state.x, ucon, state.k, state.e2, bunit);
    const Real bproj2 = out.b1 * out.b1 + out.b2 * out.b2;
    if (bproj2 > Real(1e-30)) {
        out.cos2chi = (out.b2 * out.b2 - out.b1 * out.b1) / bproj2;
        out.sin2chi = -Real(2) * out.b1 * out.b2 / bproj2;
    }
    return out;
}


template<class Metric, class Real, class Model>
KPOLARIS_INLINE PlasmaTraceDiagnostics<Real> trace_plasma_diagnostics(
    const Metric& metric,
    const Model& model,
    const TransportState<Real>& state) {
    PlasmaTraceDiagnostics<Real> out;
    Real rho = Real(0), uu = Real(0);
    Vec4<Real> ucon, bcon;
    if (!model.fluid_state(metric, state, rho, uu, ucon, bcon,
                           out.ne_cgs, out.thetae, out.b_cgs, out.sigma, out.beta)) {
        return out;
    }
    const Vec4<Real> ucov = model.lower_vector(metric, state.x, ucon);
    Real uk = Real(0);
    for (int mu = 0; mu < ndim; ++mu) uk += ucov[mu] * state.k[mu];
    const Real nu_scale = max_val(abs_val(-uk), Real(1e-8));
    out.nu_fluid_hz = model.freq_cgs * nu_scale;
    const Vec4<Real> bcov = model.lower_vector(metric, state.x, bcon);
    Real kdotb = Real(0);
    for (int mu = 0; mu < ndim; ++mu) kdotb += state.k[mu] * bcov[mu];
    const Real bnorm = Kokkos::sqrt(max_val(metric.dot(state.x, bcon, bcon), Real(1e-300)));
    const Vec4<Real> bunit = bcon * (Real(1) / bnorm);
    const Real cos_theta = clamp(kdotb / max_val(nu_scale * bnorm, Real(1e-30)), Real(-1), Real(1));
    out.theta_bk = Kokkos::acos(cos_theta);
    out.b1 = model.screen_inner_product(metric, state.x, ucon, state.k, state.e1, bunit);
    out.b2 = model.screen_inner_product(metric, state.x, ucon, state.k, state.e2, bunit);
    const Real bproj2 = out.b1 * out.b1 + out.b2 * out.b2;
    if (bproj2 > Real(1e-30)) {
        out.cos2chi = (out.b2 * out.b2 - out.b1 * out.b1) / bproj2;
        out.sin2chi = -Real(2) * out.b1 * out.b2 / bproj2;
    }
    return out;
}

template<class StoreReal, class ExecSpace, class Metric, class Real, class Model>
KPOLARIS_INLINE void record_trace_sample(const Metric& metric,
                                      const Model& model,
                                      const TransportState<Real>& state,
                                      const TransferCoeffs<Real>& coeffs,
                                      const Stokes<Real>& stokes,
                                      Real lambda,
                                      Real dlambda,
                                      int ray,
                                      int sample,
                                      const TraceConfig<Real>& config,
                                      const TraceViews<StoreReal, ExecSpace>& views) {
    if (config.record_lambda) {
        trace_store<StoreReal>(views.lambda, ray, sample, lambda);
        trace_store<StoreReal>(views.dlambda, ray, sample, dlambda);
    }
    if (config.record_coords) {
        Real r = Real(0), th = Real(0), cp = Real(1), sp = Real(0);
        model.bl_coordinates_for_metric(metric, state.x, r, th, cp, sp);
        Real phi = Kokkos::atan2(sp, cp);
        if (phi < Real(0)) {
            phi += Real(6.283185307179586476925286766559005768);
        }
        trace_store<StoreReal>(views.r, ray, sample, r);
        trace_store<StoreReal>(views.theta, ray, sample, th);
        trace_store<StoreReal>(views.phi, ray, sample, phi);
    }

    if (config.record_plasma) {
        const PlasmaTraceDiagnostics<Real> plasma = trace_plasma_diagnostics(metric, model, state);
        trace_store<StoreReal>(views.ne_cgs, ray, sample, plasma.ne_cgs);
        trace_store<StoreReal>(views.thetae, ray, sample, plasma.thetae);
        trace_store<StoreReal>(views.b_cgs, ray, sample, plasma.b_cgs);
        trace_store<StoreReal>(views.beta, ray, sample, plasma.beta);
        trace_store<StoreReal>(views.sigma, ray, sample, plasma.sigma);
        trace_store<StoreReal>(views.nu_fluid_hz, ray, sample, plasma.nu_fluid_hz);
        trace_store<StoreReal>(views.theta_bk, ray, sample, plasma.theta_bk);
        trace_store<StoreReal>(views.b1, ray, sample, plasma.b1);
        trace_store<StoreReal>(views.b2, ray, sample, plasma.b2);
        trace_store<StoreReal>(views.cos2chi, ray, sample, plasma.cos2chi);
        trace_store<StoreReal>(views.sin2chi, ray, sample, plasma.sin2chi);
    }

    if (config.record_x) {
        trace_store<StoreReal>(views.x0, ray, sample, state.x[0]);
        trace_store<StoreReal>(views.x1, ray, sample, state.x[1]);
        trace_store<StoreReal>(views.x2, ray, sample, state.x[2]);
        trace_store<StoreReal>(views.x3, ray, sample, state.x[3]);
    }
    if (config.record_k) {
        trace_store<StoreReal>(views.k0, ray, sample, state.k[0]);
        trace_store<StoreReal>(views.k1, ray, sample, state.k[1]);
        trace_store<StoreReal>(views.k2, ray, sample, state.k[2]);
        trace_store<StoreReal>(views.k3, ray, sample, state.k[3]);
    }
    if (config.record_e1) {
        trace_store<StoreReal>(views.e10, ray, sample, state.e1[0]);
        trace_store<StoreReal>(views.e11, ray, sample, state.e1[1]);
        trace_store<StoreReal>(views.e12, ray, sample, state.e1[2]);
        trace_store<StoreReal>(views.e13, ray, sample, state.e1[3]);
    }
    if (config.record_e2) {
        trace_store<StoreReal>(views.e20, ray, sample, state.e2[0]);
        trace_store<StoreReal>(views.e21, ray, sample, state.e2[1]);
        trace_store<StoreReal>(views.e22, ray, sample, state.e2[2]);
        trace_store<StoreReal>(views.e23, ray, sample, state.e2[3]);
    }

    if (config.record_coeffs) {
        trace_store<StoreReal>(views.jI, ray, sample, coeffs.jI);
        trace_store<StoreReal>(views.jQ, ray, sample, coeffs.jQ);
        trace_store<StoreReal>(views.jU, ray, sample, coeffs.jU);
        trace_store<StoreReal>(views.jV, ray, sample, coeffs.jV);
        trace_store<StoreReal>(views.aI, ray, sample, coeffs.aI);
        trace_store<StoreReal>(views.aQ, ray, sample, coeffs.aQ);
        trace_store<StoreReal>(views.aU, ray, sample, coeffs.aU);
        trace_store<StoreReal>(views.aV, ray, sample, coeffs.aV);
        trace_store<StoreReal>(views.rQ, ray, sample, coeffs.rQ);
        trace_store<StoreReal>(views.rU, ray, sample, coeffs.rU);
        trace_store<StoreReal>(views.rV, ray, sample, coeffs.rV);
    }

    if (config.record_stokes) {
        trace_store<StoreReal>(views.SI, ray, sample, stokes.I);
        trace_store<StoreReal>(views.SQ, ray, sample, stokes.Q);
        trace_store<StoreReal>(views.SU, ray, sample, stokes.U);
        trace_store<StoreReal>(views.SV, ray, sample, stokes.V);
    }
}

template<class StoreReal, class ExecSpace, class Metric, class Real, class RadiationModel>
KPOLARIS_INLINE void trace_pass_b_segment_samples_pixel_metric(
    int ray,
    int pixel,
    const PassAParams<Real>& params,
    const RadiationModel& radiation_model,
    const TraceConfig<Real>& config,
    const Metric& metric,
    const Real* step_control_frequencies,
    int step_control_nfreq,
    const TraceViews<StoreReal, ExecSpace>& views) {
    views.pixel(ray) = pixel;
    views.sample_count(ray) = 0;
    views.pass_a_steps(ray) = 0;
    views.pass_b_steps(ray) = 0;
    views.reason(ray) = static_cast<int>(TerminationReason::max_steps);
    views.closure_x(ray) = StoreReal(0);
    views.closure_k(ray) = StoreReal(0);
    views.final_null(ray) = StoreReal(0);
    views.frame_error(ray) = StoreReal(0);
    views.final_propagated_i(ray) = StoreReal(0);
    views.final_propagated_q(ray) = StoreReal(0);
    views.final_propagated_u(ray) = StoreReal(0);
    views.final_propagated_v(ray) = StoreReal(0);
    views.final_observed_i(ray) = StoreReal(0);
    views.final_observed_q(ray) = StoreReal(0);
    views.final_observed_u(ray) = StoreReal(0);
    views.final_observed_v(ray) = StoreReal(0);

    const TransportState<Real> camera_state = initialize_camera_ray(metric, pixel, params.camera);
    const auto endpoint = trace_pass_a_segment_endpoint_pixel_metric(pixel, params, metric);
    TransportState<Real> state = endpoint.state;
    views.pass_a_steps(ray) = endpoint.steps;
    views.reason(ray) = static_cast<int>(endpoint.reason);

    if (!endpoint.valid) {
        views.closure_x(ray) = static_cast<StoreReal>(spatial_distance(state.x, camera_state.x));
        views.closure_k(ray) = static_cast<StoreReal>(vector_max_abs_difference(state.k, camera_state.k));
        views.final_null(ray) = static_cast<StoreReal>(metric.dot(state.x, state.k, state.k));
        views.frame_error(ray) = static_cast<StoreReal>(max_frame_error(frame_errors(metric, state)));
        return;
    }

    const int screen_orientation = backward_screen_orientation_sign(params.camera);

    Stokes<Real> stokes;
    int reached_camera = 0;
    TerminationReason pass_b_reason = TerminationReason::max_steps;
    int pass_b_steps = 0;
    int sample_count = 0;
    Real lambda = Real(0);
    Real h_current = params.step;
    Real radiation_step_cap = params.max_radiation_step > Real(0) ?
        params.max_radiation_step : params.max_step;
    AdaptiveRK4Control<Real> adaptive_control;
    adaptive_control.tolerance = params.adaptive_tolerance;
    adaptive_control.min_step = params.min_step;
    adaptive_control.max_step = params.max_step;
    const int sample_stride = config.sample_stride > 0 ? config.sample_stride : 1;
    const int max_samples = config.max_samples > 0 ? config.max_samples : 0;
    while (!reached_camera && pass_b_steps < params.max_steps) {
        if (!metric_time_domain_valid(metric, state.x)) {
            pass_b_reason = TerminationReason::metric_time_exhausted;
            break;
        }
        Real h = h_current;
        const TransportState<Real> old_state = state;
        const Real effective_min_step = metric_effective_min_step(
            metric, old_state.k, params.min_step);
        const Real start_r_bl = radial_coordinate(metric, old_state.x);
        if (start_r_bl <= params.outer_radius && radiation_step_cap > Real(0)) {
            h = min_val(h, radiation_step_cap);
        }
        if (metric_time_domain_step_is_terminal(
                metric, old_state.x, old_state.k,
                h, effective_min_step)) {
            pass_b_reason = TerminationReason::metric_time_exhausted;
            break;
        }
        h = metric_limit_time_domain_step(
            metric, old_state.x, old_state.k, h);
        TransportState<Real> next_state;
        TransportState<Real> sample_state;
        if (params.adaptive) {
            AdaptiveRK4Control<Real> trial_control = adaptive_control;
            trial_control.min_step = effective_min_step;
            const auto proposed = adaptive_rk4_step(metric, old_state, h, trial_control);
            if (!proposed.accepted) {
                h_current = abs_val(proposed.next_h);
                if (h_current <= effective_min_step * Real(1.0001)) {
                    pass_b_reason = TerminationReason::adaptive_step_underflow;
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
            pass_b_reason = TerminationReason::metric_time_exhausted;
            break;
        }
        const Real s0 = camera_surface_value(metric, params.camera, old_state);
        const Real s1 = camera_surface_value(metric, params.camera, next_state);
        const int crosses_camera = camera_surface_crossed(s0, s1);
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
                    AdaptiveRK4Control<Real> crossing_control = adaptive_control;
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

        const Real old_r_bl = radial_coordinate(metric, old_state.x);
        const Real sample_r_bl = radial_coordinate(metric, sample_state.x);
        const Real next_r_bl = radial_coordinate(metric, next_state.x);
        TransferCoeffs<Real> coeffs;
        if (sample_r_bl >= params.inner_radius && sample_r_bl < params.outer_radius) {
            coeffs = radiation_model.coefficients(metric, sample_state, Real(0.5));
            transform_axial_coefficients_to_screen_orientation(coeffs, screen_orientation);
        }
        const Real local_max_radiation_step =
            (params.max_radiation_step > Real(0) &&
             (old_r_bl <= params.outer_radius || sample_r_bl <= params.outer_radius ||
              next_r_bl <= params.outer_radius)) ? params.max_radiation_step : Real(0);
        int required_steps = radiation_substep_count(
            coeffs, h, radiation_model.dlambda_scale(), 1,
            local_max_radiation_step, params.max_radiation_depth,
            params.max_absorption_depth, params.max_faraday_depth);
        for (int f = 0; f < step_control_nfreq; ++f) {
            RadiationModel control_model = radiation_model;
            control_model.freq_cgs = step_control_frequencies[f];
            TransferCoeffs<Real> control_coeffs = control_model.coefficients(metric, sample_state, Real(0.5));
            transform_axial_coefficients_to_screen_orientation(control_coeffs, screen_orientation);
            const int control_steps = radiation_substep_count(
                control_coeffs, h, control_model.dlambda_scale(), 1,
                local_max_radiation_step, params.max_radiation_depth,
                params.max_absorption_depth, params.max_faraday_depth);
            required_steps = max_val(required_steps, control_steps);
        }
        if (required_steps > 1 && h > effective_min_step * Real(1.0001)) {
            h_current = max_val(effective_min_step, h / Real(required_steps));
            radiation_step_cap = h_current;
            continue;
        }

        if (local_max_radiation_step > Real(0) || params.max_radiation_depth > Real(0) ||
            params.max_absorption_depth > Real(0) || params.max_faraday_depth > Real(0)) {
            int predictive_steps = radiation_substep_count(
                coeffs, h_current, radiation_model.dlambda_scale(), 1,
                local_max_radiation_step, params.max_radiation_depth,
                params.max_absorption_depth, params.max_faraday_depth);
            for (int f = 0; f < step_control_nfreq; ++f) {
                RadiationModel control_model = radiation_model;
                control_model.freq_cgs = step_control_frequencies[f];
                TransferCoeffs<Real> control_coeffs;
                if (sample_r_bl >= params.inner_radius && sample_r_bl < params.outer_radius) {
                    control_coeffs = control_model.coefficients(metric, sample_state, Real(0.5));
                    transform_axial_coefficients_to_screen_orientation(control_coeffs,
                                                                       screen_orientation);
                }
                const int control_steps = radiation_substep_count(
                    control_coeffs, h_current, control_model.dlambda_scale(), 1,
                    local_max_radiation_step, params.max_radiation_depth,
                    params.max_absorption_depth, params.max_faraday_depth);
                predictive_steps = max_val(predictive_steps, control_steps);
            }
            radiation_step_cap = predictive_steps > 1 ?
                max_val(effective_min_step, h_current / Real(predictive_steps)) : params.max_step;
        } else {
            radiation_step_cap = params.max_step;
        }

        if ((pass_b_steps % sample_stride) == 0 && sample_count < max_samples) {
            record_trace_sample(metric, radiation_model, sample_state, coeffs, stokes,
                                lambda + Real(0.5) * h,
                                abs_val(h) * radiation_model.dlambda_scale(),
                                ray, sample_count, config, views);
            ++sample_count;
        }
        if (h > Real(0)) {
            semi_analytic_stokes_step(stokes, coeffs,
                                      abs_val(h) * radiation_model.dlambda_scale());
        }
        state = next_state;
        lambda += h;
        ++pass_b_steps;
        if (crosses_camera) {
            reached_camera = 1;
        }
    }

    views.sample_count(ray) = sample_count;
    views.pass_b_steps(ray) = pass_b_steps;
    views.reason(ray) = static_cast<int>(
        reached_camera ? TerminationReason::reached_camera : pass_b_reason);
    views.closure_x(ray) = static_cast<StoreReal>(spatial_distance(state.x, camera_state.x));
    Vec4<Real> target_k;
    for (int mu = 0; mu < ndim; ++mu) {
        target_k[mu] = camera_state.k[mu];
    }
    views.closure_k(ray) = static_cast<StoreReal>(vector_max_abs_difference(state.k, target_k));
    views.final_null(ray) = static_cast<StoreReal>(metric.dot(state.x, state.k, state.k));
    views.frame_error(ray) = static_cast<StoreReal>(max_frame_error(frame_errors(metric, state)));
    views.final_propagated_i(ray) = static_cast<StoreReal>(stokes.I);
    views.final_propagated_q(ray) = static_cast<StoreReal>(stokes.Q);
    views.final_propagated_u(ray) = static_cast<StoreReal>(stokes.U);
    views.final_propagated_v(ray) = static_cast<StoreReal>(stokes.V);
    const auto overlap = screen_overlap(metric, state.x, camera_state, state);
    const Stokes<Real> observed = transform_to_observer_basis(stokes, overlap);
    views.final_observed_i(ray) = static_cast<StoreReal>(observed.I);
    views.final_observed_q(ray) = static_cast<StoreReal>(observed.Q);
    views.final_observed_u(ray) = static_cast<StoreReal>(observed.U);
    views.final_observed_v(ray) = static_cast<StoreReal>(observed.V);
}

template<class ExecSpace, class StoreReal, class Real, class RadiationModel>
void run_trace_pass_b_segment_model(const PassAParams<Real>& params,
                                    const RadiationModel& radiation_model,
                                    const TraceConfig<Real>& config,
                                    const TraceViews<StoreReal, ExecSpace>& views) {
    const int ray_count = config.ray_count;
    if (params.coordinate_system == CoordinateSystem::BoyerLindquist) {
        const KerrBoyerLindquistMetric<Real> metric(params.mass, params.spin);
        Kokkos::parallel_for(
            "KPOLARISTracePassBModelBL",
            Kokkos::RangePolicy<ExecSpace>(0, ray_count),
            KOKKOS_LAMBDA(const int ray) {
                trace_pass_b_segment_samples_pixel_metric(
                    ray, config.first_pixel + ray, params, radiation_model,
                    config, metric, static_cast<const Real*>(nullptr), 0, views);
            });
    } else if (params.coordinate_system == CoordinateSystem::SphericalKS) {
        const KerrSchildSphericalMetric<Real> metric(params.mass, params.spin);
        Kokkos::parallel_for(
            "KPOLARISTracePassBModelSphericalKS",
            Kokkos::RangePolicy<ExecSpace>(0, ray_count),
            KOKKOS_LAMBDA(const int ray) {
                trace_pass_b_segment_samples_pixel_metric(
                    ray, config.first_pixel + ray, params, radiation_model,
                    config, metric, static_cast<const Real*>(nullptr), 0, views);
            });
    } else if (params.coordinate_system == CoordinateSystem::MKS) {
        const KerrSchildSphericalMetric<Real> metric(params.mass, params.spin);
        Kokkos::parallel_for(
            "KPOLARISTracePassBModelMKS",
            Kokkos::RangePolicy<ExecSpace>(0, ray_count),
            KOKKOS_LAMBDA(const int ray) {
                trace_pass_b_segment_samples_pixel_metric(
                    ray, config.first_pixel + ray, params, radiation_model,
                    config, metric, static_cast<const Real*>(nullptr), 0, views);
            });
    } else if (params.coordinate_system == CoordinateSystem::FMKS) {
        const KerrFMKSMetric<Real> metric(params.mass, params.spin, params.fmks_startx1,
                                         params.fmks_hslope, params.fmks_mks_smooth,
                                         params.fmks_poly_alpha, params.fmks_poly_xt,
                                         params.fmks_poly_norm);
        Kokkos::parallel_for(
            "KPOLARISTracePassBModelFMKS",
            Kokkos::RangePolicy<ExecSpace>(0, ray_count),
            KOKKOS_LAMBDA(const int ray) {
                trace_pass_b_segment_samples_pixel_metric(
                    ray, config.first_pixel + ray, params, radiation_model,
                    config, metric, static_cast<const Real*>(nullptr), 0, views);
            });
    } else {
        const KerrSchildInMetric<Real> metric(params.mass, params.spin);
        Kokkos::parallel_for(
            "KPOLARISTracePassBModelKS",
            Kokkos::RangePolicy<ExecSpace>(0, ray_count),
            KOKKOS_LAMBDA(const int ray) {
                trace_pass_b_segment_samples_pixel_metric(
                    ray, config.first_pixel + ray, params, radiation_model,
                    config, metric, static_cast<const Real*>(nullptr), 0, views);
            });
    }
}

template<class ExecSpace, class StoreReal, class Real, class RadiationModel>
void run_trace_pass_b_segment_model_control(const PassAParams<Real>& params,
                                            const RadiationModel& radiation_model,
                                            const TraceConfig<Real>& config,
                                            const Real* step_control_frequencies,
                                            int step_control_nfreq,
                                            const TraceViews<StoreReal, ExecSpace>& views) {
    const int ray_count = config.ray_count;
    if (params.coordinate_system == CoordinateSystem::BoyerLindquist) {
        const KerrBoyerLindquistMetric<Real> metric(params.mass, params.spin);
        Kokkos::parallel_for(
            "KPOLARISTracePassBModelBLControl",
            Kokkos::RangePolicy<ExecSpace>(0, ray_count),
            KOKKOS_LAMBDA(const int ray) {
                trace_pass_b_segment_samples_pixel_metric(
                    ray, config.first_pixel + ray, params, radiation_model,
                    config, metric, step_control_frequencies, step_control_nfreq, views);
            });
    } else if (params.coordinate_system == CoordinateSystem::SphericalKS) {
        const KerrSchildSphericalMetric<Real> metric(params.mass, params.spin);
        Kokkos::parallel_for(
            "KPOLARISTracePassBModelSphericalKSControl",
            Kokkos::RangePolicy<ExecSpace>(0, ray_count),
            KOKKOS_LAMBDA(const int ray) {
                trace_pass_b_segment_samples_pixel_metric(
                    ray, config.first_pixel + ray, params, radiation_model,
                    config, metric, step_control_frequencies, step_control_nfreq, views);
            });
    } else if (params.coordinate_system == CoordinateSystem::FMKS) {
        const KerrFMKSMetric<Real> metric(params.mass, params.spin, params.fmks_startx1,
                                         params.fmks_hslope, params.fmks_mks_smooth,
                                         params.fmks_poly_alpha, params.fmks_poly_xt,
                                         params.fmks_poly_norm);
        Kokkos::parallel_for(
            "KPOLARISTracePassBModelFMKSControl",
            Kokkos::RangePolicy<ExecSpace>(0, ray_count),
            KOKKOS_LAMBDA(const int ray) {
                trace_pass_b_segment_samples_pixel_metric(
                    ray, config.first_pixel + ray, params, radiation_model,
                    config, metric, step_control_frequencies, step_control_nfreq, views);
            });
    } else if (params.coordinate_system == CoordinateSystem::MKS) {
        const KerrSchildSphericalMetric<Real> metric(params.mass, params.spin);
        Kokkos::parallel_for(
            "KPOLARISTracePassBModelMKSControl",
            Kokkos::RangePolicy<ExecSpace>(0, ray_count),
            KOKKOS_LAMBDA(const int ray) {
                trace_pass_b_segment_samples_pixel_metric(
                    ray, config.first_pixel + ray, params, radiation_model,
                    config, metric, step_control_frequencies, step_control_nfreq, views);
            });
    } else {
        const KerrSchildInMetric<Real> metric(params.mass, params.spin);
        Kokkos::parallel_for(
            "KPOLARISTracePassBModelCKSControl",
            Kokkos::RangePolicy<ExecSpace>(0, ray_count),
            KOKKOS_LAMBDA(const int ray) {
                trace_pass_b_segment_samples_pixel_metric(
                    ray, config.first_pixel + ray, params, radiation_model,
                    config, metric, step_control_frequencies, step_control_nfreq, views);
            });
    }
}

} // namespace kpolaris
