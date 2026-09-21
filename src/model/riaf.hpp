#pragma once

#include "radiation/plasma_perturbation.hpp"

#include "common/math.hpp"
#include "geodesic/state.hpp"
#include "radiation/stokes.hpp"
#include "radiation/thermal_synchrotron.hpp"
#include "radiation/nonthermal_synchrotron.hpp"

#ifndef KPOLARIS_RIAF_COMPILED_EMISSION_TYPE
#define KPOLARIS_RIAF_COMPILED_EMISSION_TYPE 0
#endif

namespace kpolaris {

template<class Real = DefaultReal>
struct RIAFAnalyticRadiationModel {
    Real nth0 = Real(1);
    Real Te0 = Real(1);
    Real disk_h = Real(0.35);
    Real pow_nth = Real(-1.1);
    Real pow_T = Real(-0.84);
    Real ne_unit = Real(5.0e6);
    Real te_unit = Real(1.0e11);
    Real b_unit = Real(30);
    Real freq_cgs = Real(230.0e9);
    Real mbh_solar = Real(4.3e6);
    Real r_min = Real(1);
    Real r_max = Real(100);
    Real horizon_buffer = Real(0.1);
    Real emission_scale = Real(1);
    Real absorption_scale = Real(1);
    Real faraday_scale = Real(1);
    Real max_pol_frac = Real(0.99);
    Real nonthermal_kappa = Real(3.5);
    int variable_kappa = 0;
    Real variable_kappa_min = Real(3.1);
    Real variable_kappa_interp_start = Real(1e20);
    Real variable_kappa_max = Real(7.0);
    Real powerlaw_p = Real(3.25);
    Real powerlaw_eta = Real(0.02);
    Real powerlaw_gamma_min = Real(1e2);
    Real powerlaw_gamma_max = Real(1e5);
    Real powerlaw_gamma_cutoff = Real(1e10);
    Real keplerian_factor = Real(1);
    Real infall_factor = Real(0);
    int emission_type = 1;
    Real min_frequency_scale = Real(1e-6);
    Real min_sin_theta = Real(1e-4);
    Real min_thetae = Real(1e-3);
    Real min_radius = Real(1e-6);

    KPOLARIS_INLINE Real density_profile_bl(Real r_bl, Real th) const {
        const Real zc = r_bl * Kokkos::cos(th);
        const Real rc = max_val(r_bl * Kokkos::sin(th), min_radius);
        const Real h = max_val(disk_h, Real(1e-6));
        const Real vertical = Kokkos::exp(-(zc * zc) / (Real(2) * rc * rc * h * h));
        return nth0 * vertical * Kokkos::pow(max_val(r_bl, min_radius), pow_nth);
    }

    KPOLARIS_INLINE Real density_profile(const Vec4<Real>& x, Real r_bl) const {
        const Real th = Kokkos::acos(clamp(x[3] / max_val(r_bl, tiny_positive<Real>()), Real(-1), Real(1)));
        return density_profile_bl(r_bl, th);
    }

    KPOLARIS_INLINE Real thetae_profile(Real r_bl) const {
        const Real kbol = Real(1.380649e-16);
        const Real me = Real(9.1093837015e-28);
        const Real cl = Real(2.99792458e10);
        const Real thetae_unit = te_unit * kbol / (me * cl * cl);
        return max_val(Te0 * Kokkos::pow(max_val(r_bl, min_radius), pow_T) * thetae_unit,
                       min_thetae);
    }

    KPOLARIS_INLINE Real length_unit_cgs() const {
        const Real gnewt = Real(6.67430e-8);
        const Real msun = Real(1.98847e33);
        const Real cl = Real(2.99792458e10);
        return gnewt * mbh_solar * msun / (cl * cl);
    }

    KPOLARIS_INLINE Real dlambda_scale() const {
        return length_unit_cgs() / max_val(freq_cgs, Real(1));
    }

    KPOLARIS_INLINE Real magnetic_field_cgs(Real ne_cgs, Real r_bl) const {
        const Real pi = Real(3.141592653589793238462643383279502884);
        const Real mp = Real(1.67262192369e-24);
        const Real cl = Real(2.99792458e10);
        const Real eps = Real(0.1);
        const Real b = Kokkos::sqrt(max_val(Real(8) * pi * eps * ne_cgs * mp * cl * cl /
                                            (Real(6) * max_val(r_bl, min_radius)),
                                            Real(0)));
        return max_val(b, Real(1e-12));
    }

    KPOLARIS_INLINE void bl_coordinates(const Vec4<Real>& x, Real spin,
                                     Real& r, Real& th,
                                     Real& cosphi, Real& sinphi) const {
        const Real xx = x[1];
        const Real yy = x[2];
        const Real zz = x[3];
        const Real a2 = spin * spin;
        const Real radius2 = xx * xx + yy * yy + zz * zz;
        const Real sqsum = radius2 - a2;
        const Real discr = Kokkos::sqrt(sqsum * sqsum + Real(4) * a2 * zz * zz);
        r = Kokkos::sqrt(max_val(Real(0.5) * (sqsum + discr), tiny_positive<Real>()));
        th = Kokkos::acos(clamp(zz / max_val(r, tiny_positive<Real>()), Real(-1), Real(1)));
        const Real sinth = max_val(Kokkos::sin(th), Real(1e-30));
        const Real denom = max_val(sinth * (r * r + a2), tiny_positive<Real>());
        // Match KerrSchildInMetric and the FMKS -> Cartesian-KS map:
        //   x = (r cos(phi) - a sin(phi)) sin(theta)
        //   y = (r sin(phi) + a cos(phi)) sin(theta).
        // The opposite signs silently leave circular axial vectors unchanged,
        // but rotate every vector with a radial/polar component incorrectly.
        cosphi = (r * xx + spin * yy) / denom;
        sinphi = (r * yy - spin * xx) / denom;
        const Real normp = Kokkos::sqrt(max_val(cosphi * cosphi + sinphi * sinphi, tiny_positive<Real>()));
        cosphi /= normp;
        sinphi /= normp;
    }

    template<class Metric>
    KPOLARIS_INLINE void bl_coordinates_for_metric(const Metric& metric,
                                                const Vec4<Real>& x,
                                                Real& r, Real& th,
                                                Real& cosphi, Real& sinphi) const {
        if constexpr (Metric::coordinate_system == CoordinateSystem::BoyerLindquist) {
            r = x[1];
            th = x[2];
            cosphi = Kokkos::cos(x[3]);
            sinphi = Kokkos::sin(x[3]);
        } else {
            bl_coordinates(x, metric.spin, r, th, cosphi, sinphi);
        }
    }

    KPOLARIS_INLINE void bl_metric(Real r, Real th, Real spin,
                                Real gcov[ndim][ndim], Real gcon[ndim][ndim]) const {
        const Real a = spin;
        const Real a2 = a * a;
        const Real sinth = Kokkos::sin(th);
        const Real sin2 = max_val(sinth * sinth, Real(1e-30));
        const Real costh = Kokkos::cos(th);
        const Real sigma = r * r + a2 * costh * costh;
        const Real delta = r * r - Real(2) * r + a2;
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                gcov[mu][nu] = Real(0);
                gcon[mu][nu] = Real(0);
            }
        }
        gcov[0][0] = -(Real(1) - Real(2) * r / sigma);
        gcov[0][3] = gcov[3][0] = -Real(2) * a * r * sin2 / sigma;
        gcov[1][1] = sigma / max_val(delta, tiny_positive<Real>());
        gcov[2][2] = sigma;
        gcov[3][3] = (r * r + a2 + Real(2) * a2 * r * sin2 / sigma) * sin2;

        const Real inv_sigma_delta = Real(1) / max_val(sigma * delta, tiny_positive<Real>());
        gcon[0][0] = -((r * r + a2) * (r * r + a2) - a2 * delta * sin2) * inv_sigma_delta;
        gcon[0][3] = gcon[3][0] = -Real(2) * a * r * inv_sigma_delta;
        gcon[1][1] = delta / sigma;
        gcon[2][2] = Real(1) / sigma;
        gcon[3][3] = (delta - a2 * sin2) / max_val(sigma * delta * sin2, tiny_positive<Real>());
    }

    KPOLARIS_INLINE Real isco_radius(Real spin) const {
        const Real a = spin;
        const Real one = Real(1);
        const Real z1 = one + Kokkos::cbrt(one - a * a) *
                        (Kokkos::cbrt(one + a) + Kokkos::cbrt(one - a));
        const Real z2 = Kokkos::sqrt(Real(3) * a * a + z1 * z1);
        const Real sign = a < Real(0) ? Real(-1) : Real(1);
        return Real(3) + z2 - sign * Kokkos::sqrt((Real(3) - z1) * (Real(3) + z1 + Real(2) * z2));
    }

    KPOLARIS_INLINE Real signed_nonzero_denominator(Real x) const {
        const Real floor = tiny_positive<Real>();
        if (abs_val(x) > floor) {
            return x;
        }
        return x < Real(0) ? -floor : floor;
    }

    KPOLARIS_INLINE Vec4<Real> bl_to_cartesian_ks_vector(Real vt_bl, Real vr_bl, Real vth_bl, Real vphi_bl,
                                                      Real r, Real th, Real cosphi, Real sinphi,
                                                      Real spin) const {
        const Real a = spin;
        const Real delta = max_val(r * r - Real(2) * r + a * a, tiny_positive<Real>());
        const Real vt_ks = vt_bl + Real(2) * r / delta * vr_bl;
        const Real vphi_ks = vphi_bl + a / delta * vr_bl;
        const Real sinth = Kokkos::sin(th);
        const Real costh = Kokkos::cos(th);
        const Real dx_dr = cosphi * sinth;
        const Real dy_dr = sinphi * sinth;
        const Real dz_dr = costh;
        const Real dx_dth = (r * cosphi - a * sinphi) * costh;
        const Real dy_dth = (r * sinphi + a * cosphi) * costh;
        const Real dz_dth = -r * sinth;
        const Real dx_dphi = (-r * sinphi - a * cosphi) * sinth;
        const Real dy_dphi = ( r * cosphi - a * sinphi) * sinth;
        const Real dz_dphi = Real(0);
        return Vec4<Real>(vt_ks,
                          vr_bl * dx_dr + vth_bl * dx_dth + vphi_ks * dx_dphi,
                          vr_bl * dy_dr + vth_bl * dy_dth + vphi_ks * dy_dphi,
                          vr_bl * dz_dr + vth_bl * dz_dth + vphi_ks * dz_dphi);
    }

    template<class Metric>
    KPOLARIS_INLINE Vec4<Real> fluid_four_velocity_from_bl_coords(
        const Metric& metric,
        [[maybe_unused]] const Vec4<Real>& x,
        Real r,
        Real th,
        Real cosphi,
        Real sinphi,
        const Real metric_gcov[ndim][ndim]) const {
        const Real rh = metric.horizon_radius();
        const Real risco = isco_radius(metric.spin);

        Real gcov[ndim][ndim], gcon[ndim][ndim];
        bl_metric(r, th, metric.spin, gcov, gcon);

        Real vt = Real(1), vr = Real(0), vth = Real(0), vphi = Real(0);
        if (r < rh) {
            vt = Real(1) / Kokkos::sqrt(max_val(-gcon[0][0], tiny_positive<Real>()));
        } else if (r < risco) {
            Real gcov_i[ndim][ndim], gcon_i[ndim][ndim];
            bl_metric(risco, th, metric.spin, gcov_i, gcon_i);
            const Real omega_isco = Real(1) / (risco * Kokkos::sqrt(risco) + metric.spin);
            const Real qi = gcov_i[0][0] + Real(2) * omega_isco * gcov_i[0][3] +
                            omega_isco * omega_isco * gcov_i[3][3];
            const Real ut_i = Real(1) / Kokkos::sqrt(max_val(-qi, tiny_positive<Real>()));
            const Real e = (gcov_i[0][0] + gcov_i[0][3] * omega_isco) * ut_i;
            const Real l = (gcov_i[3][0] + gcov_i[3][3] * omega_isco) * ut_i;
            const Real k_con = gcon[0][0] * e * e + Real(2) * gcon[0][3] * e * l +
                               gcon[3][3] * l * l;
            const Real ur_cov = -Kokkos::sqrt(max_val(-(Real(1) + k_con) / max_val(gcon[1][1], tiny_positive<Real>()), Real(0)));
            const Real vr_tmp = gcon[1][1] * ur_cov;
            const Real vt_tmp = gcon[0][0] * e + gcon[0][3] * l;
            const Real vphi_tmp = gcon[3][0] * e + gcon[3][3] * l;
            const Real omega_k = vphi_tmp / max_val(vt_tmp, tiny_positive<Real>());
            const Real omega_ff = gcon[0][3] / signed_nonzero_denominator(gcon[0][0]);
            const Real omega = omega_k + (Real(1) - keplerian_factor) * (omega_ff - omega_k);
            const Real ur_ff = -Kokkos::sqrt(max_val(-(Real(1) + gcon[0][0]) * gcon[1][1], Real(0)));
            vr = vr_tmp + infall_factor * (ur_ff - vr_tmp);
            const Real q = gcov[0][0] + Real(2) * omega * gcov[0][3] + omega * omega * gcov[3][3];
            vt = Kokkos::sqrt(max_val(-(Real(1) + vr * vr * gcov[1][1]) / q, Real(0)));
            vphi = omega * vt;
        } else {
            const Real r_eff = max_val(r, min_radius);
            const Real omega_k = Real(1) / (r_eff * Kokkos::sqrt(r_eff) + metric.spin);
            const Real omega_ff = gcon[0][3] / signed_nonzero_denominator(gcon[0][0]);
            const Real omega = omega_k + (Real(1) - keplerian_factor) * (omega_ff - omega_k);
            vr = infall_factor * -Kokkos::sqrt(max_val(-(Real(1) + gcon[0][0]) * gcon[1][1], Real(0)));
            const Real q = gcov[0][0] + Real(2) * omega * gcov[0][3] + omega * omega * gcov[3][3];
            vt = Kokkos::sqrt(max_val(-(Real(1) + vr * vr * gcov[1][1]) / q, Real(0)));
            vphi = omega * vt;
        }

        Vec4<Real> u;
        if constexpr (Metric::coordinate_system == CoordinateSystem::BoyerLindquist) {
            u = Vec4<Real>(vt, vr, vth, vphi);
        } else {
            u = bl_to_cartesian_ks_vector(vt, vr, vth, vphi, r, th, cosphi, sinphi, metric.spin);
        }
        const Real norm2 = dot_with_gcov(metric_gcov, u, u);
        if (norm2 < Real(0)) {
            u = u * (Real(1) / Kokkos::sqrt(max_val(-norm2, tiny_positive<Real>())));
        }
        return u;
    }

    template<class Metric>
    KPOLARIS_INLINE Vec4<Real> fluid_four_velocity(const Metric& metric,
                                                const Vec4<Real>& x,
                                                Real r_bl) const {
        Real r = r_bl, th = Real(0), cosphi = Real(1), sinphi = Real(0);
        bl_coordinates_for_metric(metric, x, r, th, cosphi, sinphi);
        Real gcov_metric[ndim][ndim];
        metric.gcov_matrix(x, gcov_metric);
        return fluid_four_velocity_from_bl_coords(metric, x, r, th, cosphi, sinphi, gcov_metric);
    }

    KPOLARIS_INLINE Real dot_with_gcov(const Real gcov[ndim][ndim],
                                    const Vec4<Real>& u,
                                    const Vec4<Real>& v) const {
        Real out = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                out += gcov[mu][nu] * u[mu] * v[nu];
            }
        }
        return out;
    }

    KPOLARIS_INLINE Vec4<Real> lower_vector_with_gcov(const Real gcov[ndim][ndim],
                                                   const Vec4<Real>& v) const {
        Vec4<Real> out;
        for (int mu = 0; mu < ndim; ++mu) {
            Real sum = Real(0);
            for (int nu = 0; nu < ndim; ++nu) {
                sum += gcov[mu][nu] * v[nu];
            }
            out[mu] = sum;
        }
        return out;
    }

    template<class Metric>
    KPOLARIS_INLINE Vec4<Real> lower_vector(const Metric& metric,
                                         const Vec4<Real>& x,
                                         const Vec4<Real>& v) const {
        Real gcov[ndim][ndim];
        metric.gcov_matrix(x, gcov);
        return lower_vector_with_gcov(gcov, v);
    }

    KPOLARIS_INLINE Real screen_inner_product_with_gcov(const Real gcov[ndim][ndim],
                                                     const Vec4<Real>& u,
                                                     const Vec4<Real>& k,
                                                     const Vec4<Real>& a,
                                                     const Vec4<Real>& b) const {
        // Match Arcmancer's screen projection: compare polarization axes in
        // the two-plane orthogonal to both the local fluid velocity u and
        // photon direction k, not by a raw 4D dot product.
        const Real uk = dot_with_gcov(gcov, u, k);
        if (abs_val(uk) <= tiny_positive<Real>()) {
            return dot_with_gcov(gcov, a, b);
        }
        const Vec4<Real> eK = k * (Real(-1) / uk) - u;
        return dot_with_gcov(gcov, a, b) +
               dot_with_gcov(gcov, u, a) * dot_with_gcov(gcov, u, b) -
               dot_with_gcov(gcov, eK, a) * dot_with_gcov(gcov, eK, b);
    }

    template<class Metric>
    KPOLARIS_INLINE Real screen_inner_product(const Metric& metric,
                                           const Vec4<Real>& x,
                                           const Vec4<Real>& u,
                                           const Vec4<Real>& k,
                                           const Vec4<Real>& a,
                                           const Vec4<Real>& b) const {
        Real gcov[ndim][ndim];
        metric.gcov_matrix(x, gcov);
        return screen_inner_product_with_gcov(gcov, u, k, a, b);
    }

    template<class Metric>
    KPOLARIS_INLINE Vec4<Real> magnetic_unit_four_vector_with_gcov(
        const Metric& metric,
        const Vec4<Real>& x,
        const Vec4<Real>& ucon,
        const Real gcov[ndim][ndim]) const {
        Real r = Real(0), th = Real(0), cosphi = Real(1), sinphi = Real(0);
        bl_coordinates_for_metric(metric, x, r, th, cosphi, sinphi);
        Vec4<Real> btrial;
        if constexpr (Metric::coordinate_system == CoordinateSystem::BoyerLindquist) {
            btrial = Vec4<Real>(Real(0), Real(0), Real(0), Real(1));
        } else {
            btrial = bl_to_cartesian_ks_vector(Real(0), Real(0), Real(0), Real(1),
                                               r, th, cosphi, sinphi, metric.spin);
        }

        const Real udotb = dot_with_gcov(gcov, ucon, btrial);
        Vec4<Real> bcon = btrial + ucon * udotb;
        Real bsq = dot_with_gcov(gcov, bcon, bcon);
        if (!(bsq > Real(1e-30))) {
            btrial = Vec4<Real>(Real(0), Real(0), Real(0), Real(1));
            const Real udotz = dot_with_gcov(gcov, ucon, btrial);
            bcon = btrial + ucon * udotz;
            bsq = dot_with_gcov(gcov, bcon, bcon);
        }
        if (!(bsq > Real(1e-30))) {
            btrial = Vec4<Real>(Real(0), Real(1), Real(0), Real(0));
            const Real udotx = dot_with_gcov(gcov, ucon, btrial);
            bcon = btrial + ucon * udotx;
            bsq = dot_with_gcov(gcov, bcon, bcon);
        }

        if (bsq > Real(1e-30)) {
            bcon = bcon * (Real(1) / Kokkos::sqrt(bsq));
        }
        return bcon;
    }

    template<class Metric>
    KPOLARIS_INLINE Vec4<Real> magnetic_unit_four_vector(const Metric& metric,
                                                      const Vec4<Real>& x,
                                                      const Vec4<Real>& ucon) const {
        Real gcov[ndim][ndim];
        metric.gcov_matrix(x, gcov);
        return magnetic_unit_four_vector_with_gcov(metric, x, ucon, gcov);
    }

    template<class Metric>
    KPOLARIS_INLINE Real fluid_frequency_scale(const Metric& metric,
                                            const TransportState<Real>& state,
                                            Real r_bl) const {
        const Vec4<Real> ucon = fluid_four_velocity(metric, state.x, r_bl);
        const Vec4<Real> ucov = lower_vector(metric, state.x, ucon);
        Real uk = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            uk += ucov[mu] * state.k[mu];
        }
        return max_val(abs_val(-uk), min_frequency_scale);
    }

    template<class Metric>
    KPOLARIS_INLINE TransferCoeffs<Real> coefficients(const Metric& metric,
                                                   const TransportState<Real>& state,
                                                   Real,
                                                   const PlasmaPerturbation<Real>& perturbation = {}) const {
        TransferCoeffs<Real> coeffs;
        Real r_coord = Real(0), th_coord = Real(0), cosphi = Real(1), sinphi = Real(0);
        bl_coordinates_for_metric(metric, state.x, r_coord, th_coord, cosphi, sinphi);
        const Real r_bl = max_val(r_coord, min_radius);
        const Real rh = metric.horizon_radius();
        if (r_bl <= rh + horizon_buffer || r_bl < r_min || r_bl > r_max) {
            return coeffs;
        }

        const Real ne_norm = density_profile_bl(r_bl, th_coord);
        if (!(ne_norm > Real(0))) {
            return coeffs;
        }
        Real ne_cgs = ne_norm * ne_unit;
        Real thetae = thetae_profile(r_bl);
        Real b_cgs = magnetic_field_cgs(ne_cgs, r_bl);
        Real sigma_unused = Real(0), beta_unused = Real(0);
        perturbation.apply(ne_cgs, thetae, b_cgs, sigma_unused, beta_unused);

        Real gcov_local[ndim][ndim];
        metric.gcov_matrix(state.x, gcov_local);
        const Vec4<Real> ucon = fluid_four_velocity_from_bl_coords(
            metric, state.x, r_bl, th_coord, cosphi, sinphi, gcov_local);
        const Vec4<Real> ucov = lower_vector_with_gcov(gcov_local, ucon);
        Real uk = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            uk += ucov[mu] * state.k[mu];
        }
        const Real nu_fluid_scale = max_val(abs_val(-uk), min_frequency_scale);

        const Vec4<Real> bcon_unit = magnetic_unit_four_vector_with_gcov(metric, state.x, ucon, gcov_local);
        const Vec4<Real> bcov_unit = lower_vector_with_gcov(gcov_local, bcon_unit);
        Real kdotb = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            kdotb += state.k[mu] * bcov_unit[mu];
        }
        const Real cos_theta = clamp(kdotb / max_val(nu_fluid_scale, Real(1e-30)), Real(-1), Real(1));
        const Real sin_theta = max_val(Kokkos::sqrt(max_val(Real(0), Real(1) - cos_theta * cos_theta)),
                                       min_sin_theta);

        const Real theta = Kokkos::acos(clamp(cos_theta, Real(-1), Real(1)));
        const Real nu = max_val(freq_cgs * nu_fluid_scale, Real(1));

        ThermalSynchrotronParams<Real> thermal_params;
#if KPOLARIS_RIAF_COMPILED_EMISSION_TYPE > 0
        thermal_params.fit = KPOLARIS_RIAF_COMPILED_EMISSION_TYPE;
#else
        thermal_params.fit = emission_type;
#endif
        thermal_params.max_pol_frac = max_pol_frac;
        thermal_params.min_sin_theta = min_sin_theta;
        thermal_params.emission_scale = emission_scale;
        thermal_params.absorption_scale = absorption_scale;
        thermal_params.faraday_scale = faraday_scale;

        const Real pi = Real(3.141592653589793238462643383279502884);
        if (theta <= Real(0) || theta >= pi) {
            coeffs.rV = faraday_scale * symphony_rho_V(nu, ne_cgs, thetae,
                                                       b_cgs, theta,
                                                       min_sin_theta) * nu;
            return coeffs;
        }

        LocalThermalSynchrotronState<Real> thermal_state;
        thermal_state.nu = nu;
        thermal_state.ne = ne_cgs;
        thermal_state.thetae = thetae;
        thermal_state.b_cgs = b_cgs;
        thermal_state.theta = theta;
        thermal_state.sin_theta = sin_theta;
        thermal_state.cos_theta = cos_theta;

        TransferCoeffs<Real> magnetic_coeffs;
#if KPOLARIS_RIAF_COMPILED_EMISSION_TYPE == KPOLARIS_THERMAL_SYNCHROTRON_DEXTER
        magnetic_coeffs =
            thermal_synchrotron_magnetic_basis_coefficients_fit<ThermalSynchrotronDexter>(thermal_state,
                                                                                         thermal_params);
#elif KPOLARIS_RIAF_COMPILED_EMISSION_TYPE == KPOLARIS_THERMAL_SYNCHROTRON_PANDYA
        magnetic_coeffs =
            thermal_synchrotron_magnetic_basis_coefficients_fit<ThermalSynchrotronPandya>(thermal_state,
                                                                                         thermal_params);
#else
        if (emission_type == SynchrotronKappa || emission_type == SynchrotronPowerLaw) {
            NonthermalSynchrotronParams<Real> nonthermal_params;
            nonthermal_params.distribution = emission_type;
            nonthermal_params.max_pol_frac_emission = max_pol_frac;
            nonthermal_params.max_pol_frac_absorption = max_pol_frac;
            nonthermal_params.min_sin_theta = min_sin_theta;
            nonthermal_params.emission_scale = emission_scale;
            nonthermal_params.absorption_scale = absorption_scale;
            nonthermal_params.faraday_scale = faraday_scale;
            nonthermal_params.kappa = nonthermal_kappa;
            nonthermal_params.kappa_interp_begin = variable_kappa_interp_start;
            nonthermal_params.kappa_interp_end = variable_kappa_max;
            nonthermal_params.power_law_p = powerlaw_p;
            nonthermal_params.power_law_eta = powerlaw_eta;
            nonthermal_params.power_law_gamma_min = powerlaw_gamma_min;
            nonthermal_params.power_law_gamma_max = powerlaw_gamma_max;
            nonthermal_params.power_law_gamma_cutoff = powerlaw_gamma_cutoff;
            LocalNonthermalSynchrotronState<Real> nonthermal_state;
            nonthermal_state.nu = nu;
            nonthermal_state.ne = ne_cgs;
            nonthermal_state.thetae = thetae;
            nonthermal_state.b_cgs = b_cgs;
            nonthermal_state.sin_theta = sin_theta;
            nonthermal_state.cos_theta = cos_theta;
            magnetic_coeffs = nonthermal_synchrotron_magnetic_basis_coefficients(nonthermal_state,
                                                                                 nonthermal_params);
        } else {
            magnetic_coeffs = thermal_synchrotron_magnetic_basis_coefficients(thermal_state,
                                                                              thermal_params);
        }
#endif

        const Real b1 = screen_inner_product_with_gcov(gcov_local, ucon, state.k, state.e1, bcon_unit);
        const Real b2 = screen_inner_product_with_gcov(gcov_local, ucon, state.k, state.e2, bcon_unit);
        const Real bproj2 = b1 * b1 + b2 * b2;
        coeffs.jI = magnetic_coeffs.jI;
        coeffs.jV = magnetic_coeffs.jV;
        coeffs.aI = magnetic_coeffs.aI;
        coeffs.aV = magnetic_coeffs.aV;
        coeffs.rV = magnetic_coeffs.rV;

        if (bproj2 > Real(1e-30)) {
            const Real cos2chi = (b2 * b2 - b1 * b1) / bproj2;
            const Real sin2chi = -Real(2) * b1 * b2 / bproj2;
            coeffs.jQ = magnetic_coeffs.jQ * cos2chi;
            coeffs.jU = magnetic_coeffs.jQ * sin2chi;
            coeffs.aQ = magnetic_coeffs.aQ * cos2chi;
            coeffs.aU = magnetic_coeffs.aQ * sin2chi;
            coeffs.rQ = magnetic_coeffs.rQ * cos2chi;
            coeffs.rU = magnetic_coeffs.rQ * sin2chi;
        }
        return coeffs;
    }
};

} // namespace kpolaris
