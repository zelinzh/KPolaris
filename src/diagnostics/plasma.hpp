#pragma once

#include "common/math.hpp"
#include "common/vec.hpp"
#include "geodesic/state.hpp"
#include "model/magnetized_torus.hpp"
#include "model/riaf.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct AnalysisPlasmaDiagnostics {
    Real ne_cgs = Real(0);
    Real thetae = Real(0);
    Real b_cgs = Real(0);
    Real beta = Real(0);
    Real sigma = Real(0);
    int valid = 0;
    int magnetization_valid = 0;
};

template<class Metric, class Real>
KPOLARIS_INLINE AnalysisPlasmaDiagnostics<Real> analysis_plasma_diagnostics(
    const Metric& metric,
    const RIAFAnalyticRadiationModel<Real>& model,
    const TransportState<Real>& state) {
    AnalysisPlasmaDiagnostics<Real> out;
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
    out.valid = (out.ne_cgs > Real(0) && out.thetae > Real(0) && out.b_cgs > Real(0)) ? 1 : 0;
    return out;
}

template<class Metric, class Real>
KPOLARIS_INLINE AnalysisPlasmaDiagnostics<Real> analysis_plasma_diagnostics(
    const Metric& metric,
    const MagnetizedTorusRadiationModel<Real>& model,
    const TransportState<Real>& state) {
    AnalysisPlasmaDiagnostics<Real> out;
    Real r = Real(0), th = Real(0), cosphi = Real(1), sinphi = Real(0);
    model.bl_coordinates_for_metric(metric, state.x, r, th, cosphi, sinphi);
    if (r <= metric.horizon_radius() * Real(1.05) ||
        (model.r_outer > Real(0) && r > model.r_outer * Real(1.15))) {
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
    const Real rho_cgs = rho_code * model.density_unit_cgs();
    const Real internal_energy_cgs = p_code / (model.kappa - Real(1)) * model.pressure_unit_cgs();
    const Real mp = Real(1.67262192369e-24);
    const Real me = Real(9.1093837015e-28);
    const Real cl = Real(2.99792458e10);
    const Real kbol = Real(1.380649e-16);
    const Real temperature = Real(2) * mp * internal_energy_cgs /
                             max_val(Real(3) * kbol * rho_cgs * (Real(2) + model.Rhigh), Real(1e-300));
    out.ne_cgs = rho_cgs / mp;
    out.thetae = max_val(kbol * temperature / (me * cl * cl), model.thetae_min);
    out.b_cgs = Kokkos::sqrt(max_val(Real(2) * pm_code, Real(0))) * model.bfield_unit_cgs();
    out.magnetization_valid = 1;
    out.beta = p_code / max_val(pm_code, Real(1e-300));
    out.sigma = Real(2) * pm_code / max_val(rho_code, Real(1e-300));
    out.valid = (out.ne_cgs > Real(0) && out.thetae > Real(0) && out.b_cgs > Real(0)) ? 1 : 0;
    return out;
}

template<class Metric, class Real, class Model>
KPOLARIS_INLINE AnalysisPlasmaDiagnostics<Real> analysis_plasma_diagnostics(
    const Metric& metric,
    const Model& model,
    const TransportState<Real>& state) {
    AnalysisPlasmaDiagnostics<Real> out;
    Real rho = Real(0), uu = Real(0);
    Vec4<Real> ucon, bcon;
    if (!model.fluid_state(metric, state, rho, uu, ucon, bcon,
                           out.ne_cgs, out.thetae, out.b_cgs, out.sigma, out.beta)) {
        return out;
    }
    out.magnetization_valid = 1;
    out.valid = (out.ne_cgs > Real(0) && out.thetae > Real(0) && out.b_cgs > Real(0)) ? 1 : 0;
    return out;
}

} // namespace kpolaris
