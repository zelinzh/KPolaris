#pragma once

#include "radiation/plasma_perturbation.hpp"

#include <cmath>

#include "common/math.hpp"
#include "geodesic/state.hpp"
#include "radiation/stokes.hpp"
#include "radiation/thermal_synchrotron.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct MagnetizedTorusRadiationModel {
    Real spin = Real(0.9);
    Real l_lambda = Real(0.78);
    Real wwin = Real(1);
    Real kappa = Real(4) / Real(3);
    Real omegac = Real(1);
    Real betac = Real(10);
    Real beta = Real(10);
    Real Rhigh = Real(1);
    Real bh_mass_solar = Real(4.0e6);
    Real accretion_rate_cgs = Real(1.57e15);
    Real accretion_rate_code = Real(3.0e-3);
    Real freq_cgs = Real(230.0e9);
    Real thetae_min = Real(5.0e-2);
    Real observer_energy_scale = Real(1);
    Real emission_scale = Real(1);
    Real absorption_scale = Real(1);
    Real faraday_scale = Real(1);
    Real max_pol_frac = Real(0.999);
    int scalar_transport = 0;

    Real l0 = Real(0);
    Real rcusp = Real(0);
    Real rc = Real(0);
    Real r_outer = Real(0);
    Real Wc = Real(0);
    Real Win = Real(0);
    Real pc = Real(0);
    Real KK = Real(0);

    MagnetizedTorusRadiationModel() { initialize_default_torus(); }

    KPOLARIS_INLINE static Real pi() { return Real(3.141592653589793238462643383279502884); }

    Real bl_gtt_host(Real r, Real th) const {
        const Real r2 = r * r;
        const Real sigma = r2 + spin * spin * std::cos(th) * std::cos(th);
        return -(Real(1) - Real(2) * r / sigma);
    }

    Real bl_gtphi_host(Real r, Real th) const {
        const Real r2 = r * r;
        const Real sigma = r2 + spin * spin * std::cos(th) * std::cos(th);
        const Real sin2 = std::sin(th) * std::sin(th);
        return -Real(2) * r * spin * sin2 / sigma;
    }

    Real bl_gphiphi_host(Real r, Real th) const {
        const Real r2 = r * r;
        const Real a2 = spin * spin;
        const Real sigma = r2 + a2 * std::cos(th) * std::cos(th);
        const Real sin2 = std::sin(th) * std::sin(th);
        return (r2 + a2 + Real(2) * r * a2 * sin2 / sigma) * sin2;
    }

    Real potential_host(Real r, Real th) const {
        const Real gtt = bl_gtt_host(r, th);
        const Real gtphi = bl_gtphi_host(r, th);
        const Real gphiphi = bl_gphiphi_host(r, th);
        const Real rcyl2 = gtphi * gtphi - gtt * gphiphi;
        const Real d2 = gtt * l0 * l0 + Real(2) * gtphi * l0 + gphiphi;
        return Real(0.5) * std::log(rcyl2 / d2);
    }

    Real dpotential_dr_host(Real r) const {
        const Real h = std::max(Real(1e-5), Real(1e-5) * r);
        return (potential_host(r + h, pi() / Real(2)) -
                potential_host(r - h, pi() / Real(2))) / (Real(2) * h);
    }

    Real bisect_dW_root_host(Real a, Real b) const {
        Real fa = dpotential_dr_host(a);
        for (int it = 0; it < 80; ++it) {
            const Real m = Real(0.5) * (a + b);
            const Real fm = dpotential_dr_host(m);
            if ((fa <= Real(0) && fm <= Real(0)) || (fa >= Real(0) && fm >= Real(0))) {
                a = m;
                fa = fm;
            } else {
                b = m;
            }
        }
        return Real(0.5) * (a + b);
    }

    void initialize_default_torus() {
        const Real z1 = Real(1) + std::cbrt(Real(1) - spin * spin) *
                        (std::cbrt(Real(1) + spin) + std::cbrt(Real(1) - spin));
        const Real z2 = std::sqrt(Real(3) * spin * spin + z1 * z1);
        const Real rms = Real(3) + z2 - std::sqrt((Real(3) - z1) * (Real(3) + z1 + Real(2) * z2));
        const Real rmb = Real(2) - spin + Real(2) * std::sqrt(Real(1) - spin);
        const auto l_kep = [this](Real r) {
            return (r * r - Real(2) * spin * std::sqrt(r) + spin * spin) /
                   (std::pow(r, Real(1.5)) - Real(2) * std::sqrt(r) + spin);
        };
        const Real lms = l_kep(rms);
        const Real lmb = l_kep(rmb);
        l0 = l_lambda * (lmb - lms) + lms;

        const Real rh = Real(1) + std::sqrt(std::max(Real(0), Real(1) - spin * spin));
        Real roots[2] = {Real(0), Real(0)};
        int nroots = 0;
        Real left = rh + Real(0.01);
        Real fleft = dpotential_dr_host(left);
        for (Real right = left + Real(0.01); right <= Real(20) && nroots < 2; right += Real(0.01)) {
            const Real fright = dpotential_dr_host(right);
            if ((fleft <= Real(0) && fright >= Real(0)) || (fleft >= Real(0) && fright <= Real(0))) {
                roots[nroots++] = bisect_dW_root_host(left, right);
                left = roots[nroots - 1] + Real(0.01);
                fleft = dpotential_dr_host(left);
            } else {
                left = right;
                fleft = fright;
            }
        }
        if (nroots >= 2) {
            rcusp = roots[0];
            rc = roots[1];
        } else {
            rcusp = rh + Real(0.2);
            rc = Real(6);
        }
        const Real Wcusp = potential_host(rcusp, pi() / Real(2));
        Wc = potential_host(rc, pi() / Real(2));
        Win = wwin * (Wcusp - Wc) + Wc;
        r_outer = Real(10) * rc;

        const Real gtt = bl_gtt_host(rc, pi() / Real(2));
        const Real gtphi = bl_gtphi_host(rc, pi() / Real(2));
        const Real gphiphi = bl_gphiphi_host(rc, pi() / Real(2));
        const Real rcyl2 = gtphi * gtphi - gtt * gphiphi;
        pc = omegac * (Win - Wc) * (kappa - Real(1)) / kappa / (Real(1) + Real(1) / betac);
        KK = pc / std::pow(omegac, kappa);
        (void)rcyl2;
    }

    KPOLARIS_INLINE Real length_unit_cgs() const {
        const Real gnewt = Real(6.67430e-8);
        const Real msun = Real(1.98847e33);
        const Real cl = Real(2.99792458e10);
        return gnewt * bh_mass_solar * msun / (cl * cl);
    }

    KPOLARIS_INLINE Real dlambda_scale() const {
        return observer_energy_scale * length_unit_cgs() / max_val(freq_cgs, Real(1));
    }

    KPOLARIS_INLINE Real density_unit_cgs() const {
        const Real cl = Real(2.99792458e10);
        const Real lunit = length_unit_cgs();
        const Real tunit = lunit / cl;
        return accretion_rate_cgs / max_val(accretion_rate_code, Real(1e-300)) * tunit /
               max_val(lunit * lunit * lunit, Real(1e-300));
    }

    KPOLARIS_INLINE Real pressure_unit_cgs() const {
        const Real cl = Real(2.99792458e10);
        return density_unit_cgs() * cl * cl;
    }

    KPOLARIS_INLINE Real bfield_unit_cgs() const {
        const Real cl = Real(2.99792458e10);
        return Kokkos::sqrt(Real(4) * pi() * density_unit_cgs()) * cl;
    }

    KPOLARIS_INLINE Real bl_gtt(Real r, Real th) const {
        const Real sigma = r * r + spin * spin * Kokkos::cos(th) * Kokkos::cos(th);
        return -(Real(1) - Real(2) * r / sigma);
    }

    KPOLARIS_INLINE Real bl_gtphi(Real r, Real th) const {
        const Real sigma = r * r + spin * spin * Kokkos::cos(th) * Kokkos::cos(th);
        const Real sin2 = Kokkos::sin(th) * Kokkos::sin(th);
        return -Real(2) * r * spin * sin2 / sigma;
    }

    KPOLARIS_INLINE Real bl_gphiphi(Real r, Real th) const {
        const Real a2 = spin * spin;
        const Real sigma = r * r + a2 * Kokkos::cos(th) * Kokkos::cos(th);
        const Real sin2 = Kokkos::sin(th) * Kokkos::sin(th);
        return (r * r + a2 + Real(2) * r * a2 * sin2 / sigma) * sin2;
    }

    KPOLARIS_INLINE Real potential(Real r, Real th) const {
        const Real gtt = bl_gtt(r, th);
        const Real gtphi = bl_gtphi(r, th);
        const Real gphiphi = bl_gphiphi(r, th);
        const Real rcyl2 = gtphi * gtphi - gtt * gphiphi;
        const Real d2 = gtt * l0 * l0 + Real(2) * gtphi * l0 + gphiphi;
        return Real(0.5) * Kokkos::log(rcyl2 / max_val(d2, Real(1e-300)));
    }

    KPOLARIS_INLINE void bl_coordinates(const Vec4<Real>& x, Real& r, Real& th,
                                     Real& cosphi, Real& sinphi) const {
        const Real xx = x[1];
        const Real yy = x[2];
        const Real zz = x[3];
        const Real a2 = spin * spin;
        const Real radius2 = xx * xx + yy * yy + zz * zz;
        const Real sqsum = radius2 - a2;
        const Real discr = Kokkos::sqrt(sqsum * sqsum + Real(4) * a2 * zz * zz);
        r = Kokkos::sqrt(max_val(Real(0.5) * (sqsum + discr), Real(1e-300)));
        th = Kokkos::acos(clamp(zz / max_val(r, Real(1e-300)), Real(-1), Real(1)));
        const Real sinth = max_val(Kokkos::sin(th), Real(1e-30));
        const Real denom = max_val(sinth * (r * r + a2), Real(1e-300));
        cosphi = (r * xx + spin * yy) / denom;
        sinphi = (r * yy - spin * xx) / denom;
        const Real normp = Kokkos::sqrt(max_val(cosphi * cosphi + sinphi * sinphi, Real(1e-300)));
        cosphi /= normp;
        sinphi /= normp;
    }

    template<class Metric>
    KPOLARIS_INLINE void bl_coordinates_for_metric([[maybe_unused]] const Metric& metric,
                                                const Vec4<Real>& x,
                                                Real& r, Real& th,
                                                Real& cosphi, Real& sinphi) const {
        if constexpr (Metric::coordinate_system == CoordinateSystem::BoyerLindquist) {
            r = x[1];
            th = x[2];
            cosphi = Kokkos::cos(x[3]);
            sinphi = Kokkos::sin(x[3]);
        } else {
            bl_coordinates(x, r, th, cosphi, sinphi);
        }
    }

    KPOLARIS_INLINE Vec4<Real> axisymmetric_bl_vector_to_cart(Real vt, Real vphi,
                                                           Real r, Real th,
                                                           Real cosphi,
                                                           Real sinphi) const {
        const Real sinth = Kokkos::sin(th);
        const Real dx_dphi = (-r * sinphi - spin * cosphi) * sinth;
        const Real dy_dphi = ( r * cosphi - spin * sinphi) * sinth;
        return Vec4<Real>(vt, vphi * dx_dphi, vphi * dy_dphi, Real(0));
    }

    template<class Metric>
    KPOLARIS_INLINE Vec4<Real> axisymmetric_bl_vector_to_metric(const Metric&,
                                                             Real vt, Real vphi,
                                                             Real r, Real th,
                                                             Real cosphi, Real sinphi) const {
        if constexpr (Metric::coordinate_system == CoordinateSystem::BoyerLindquist) {
            return Vec4<Real>(vt, Real(0), Real(0), vphi);
        } else {
            return axisymmetric_bl_vector_to_cart(vt, vphi, r, th, cosphi, sinphi);
        }
    }

    template<class Metric>
    KPOLARIS_INLINE Vec4<Real> lower_vector(const Metric& metric,
                                         const Vec4<Real>& x,
                                         const Vec4<Real>& v) const {
        Vec4<Real> out;
        for (int mu = 0; mu < ndim; ++mu) {
            Real sum = Real(0);
            for (int nu = 0; nu < ndim; ++nu) {
                sum += metric.gcov(mu, nu, x) * v[nu];
            }
            out[mu] = sum;
        }
        return out;
    }

    template<class Metric>
    KPOLARIS_INLINE Real screen_inner_product(const Metric& metric,
                                           const Vec4<Real>& x,
                                           const Vec4<Real>& u,
                                           const Vec4<Real>& k,
                                           const Vec4<Real>& a,
                                           const Vec4<Real>& b) const {
        // Match Arcmancer's screen projection: compare polarization axes in
        // the two-plane orthogonal to both the local fluid velocity u and
        // photon direction k, not by a raw 4D dot product.
        const Real uk = metric.dot(x, u, k);
        if (abs_val(uk) <= Real(1e-300)) {
            return metric.dot(x, a, b);
        }
        const Vec4<Real> eK = k * (Real(-1) / uk) - u;
        return metric.dot(x, a, b) +
               metric.dot(x, u, a) * metric.dot(x, u, b) -
               metric.dot(x, eK, a) * metric.dot(x, eK, b);
    }

    KPOLARIS_INLINE Real planck_flux(Real nu, Real temperature) const {
        const Real hpl = Real(6.62607015e-27);
        const Real kbol = Real(1.380649e-16);
        const Real cl = Real(2.99792458e10);
        const Real x = hpl * nu / max_val(kbol * temperature, Real(1e-300));
        Real expm1_val = Real(0);
        if (abs_val(x) <= Kokkos::sqrt(Real(2.220446049250313e-16))) {
            expm1_val = x + x * x / Real(2);
        } else {
            expm1_val = Kokkos::exp(x) - Real(1);
        }
        return Real(2) * hpl * nu * nu * nu / (cl * cl) / max_val(expm1_val, Real(1e-300));
    }

    KPOLARIS_INLINE Real f_function(Real x) const {
        return Real(2.011) * Kokkos::exp(-Kokkos::pow(x, Real(1.035)) / Real(4.7)) -
               Kokkos::cos(x / Real(2)) * Kokkos::exp(-Kokkos::pow(x, Real(1.2)) / Real(2.73)) -
               Real(0.011) * Kokkos::exp(-x / Real(47.2));
    }

    KPOLARIS_INLINE Real f_function_D16(Real x) const {
        return f_function(x) +
               (Real(0.011) * Kokkos::exp(-x / Real(47.2)) -
                Real(369.69342869090815) *
                Kokkos::pow(x + Real(1e-16), Real(-8) / Real(3))) *
               Real(0.5) * (Real(1) + Kokkos::tanh(Real(10) * Kokkos::log(x / Real(120))));
    }

    KPOLARIS_INLINE Real g_function_D16(Real x) const {
        return Real(0.43793091) * Kokkos::log(Real(1) + Real(0.00185777) * Kokkos::pow(x, Real(1.50316886)));
    }

    KPOLARIS_INLINE Real shape_i(Real x) const {
        const Real xs = max_val(x, Real(1e-12));
        const Real xm13 = Real(1) / Kokkos::cbrt(xs);
        return Real(2.5651) * (Real(1) + Real(1.92) * xm13 + Real(0.9977) * xm13 * xm13) *
               Kokkos::exp(-Real(1.8899) / xm13);
    }

    KPOLARIS_INLINE Real shape_q(Real x) const {
        const Real xs = max_val(x, Real(1e-12));
        const Real xm13 = Real(1) / Kokkos::cbrt(xs);
        return Real(2.5651) * (Real(1) + Real(0.93193) * xm13 + Real(0.499873) * xm13 * xm13) *
               Kokkos::exp(-Real(1.8899) / xm13);
    }

    KPOLARIS_INLINE Real shape_v(Real x) const {
        const Real xs = max_val(x, Real(1e-12));
        const Real xm13 = Real(1) / Kokkos::cbrt(xs);
        return (Real(1.81384) / xs + Real(3.42319) * xm13 * xm13 +
                Real(0.0292545) / Kokkos::sqrt(xs) + Real(2.03773) * xm13) *
               Kokkos::exp(-Real(1.8899) / xm13);
    }

    template<class Metric>
    KPOLARIS_INLINE TransferCoeffs<Real> coefficients(const Metric& metric,
                                                   const TransportState<Real>& state,
                                                   Real,
                                                   const PlasmaPerturbation<Real>& perturbation = {}) const {
        TransferCoeffs<Real> coeffs;
        Real r = Real(0), th = Real(0), cosphi = Real(1), sinphi = Real(0);
        bl_coordinates_for_metric(metric, state.x, r, th, cosphi, sinphi);
        if (r <= metric.horizon_radius() * Real(1.05) ||
            (r_outer > Real(0) && r > r_outer * Real(1.15))) {
            return coeffs;
        }

        const Real gtt = bl_gtt(r, th);
        const Real gtphi = bl_gtphi(r, th);
        const Real gphiphi = bl_gphiphi(r, th);
        const Real d2 = gtt * l0 * l0 + Real(2) * gtphi * l0 + gphiphi;
        const Real Wpot = potential(r, th);
        if (!(Wpot <= Win && Wpot >= Wc) || !(d2 > Real(0)) || !(KK > Real(0))) {
            return coeffs;
        }

        const Real omega_density = Kokkos::pow((Win - Wpot) * ((kappa - Real(1)) / kappa) /
                                               ((Real(1) + Real(1) / beta) * KK),
                                               Real(1) / (kappa - Real(1)));
        if (!(omega_density > Real(1e-12))) {
            return coeffs;
        }
        const Real p_code = KK * Kokkos::pow(omega_density, kappa);
        const Real pm_code = p_code / beta;
        const Real rho_code = omega_density - kappa / (kappa - Real(1)) * p_code;
        if (!(rho_code > Real(0) && p_code > Real(0) && pm_code > Real(0))) {
            return coeffs;
        }

        const Real Omega = -(gtt * l0 + gtphi) / (gtphi * l0 + gphiphi);
        const Real u0 = Kokkos::sqrt(max_val(-Real(1) /
                             (gtt + Real(2) * gtphi * Omega + gphiphi * Omega * Omega), Real(0)));
        const Real u3 = Omega * u0;
        const Vec4<Real> ucon = axisymmetric_bl_vector_to_metric(metric, u0, u3, r, th, cosphi, sinphi);

        const Real b3 = Kokkos::sqrt(max_val(Real(2) * pm_code / d2, Real(0)));
        const Real b0 = l0 * b3;
        const Vec4<Real> bcon = axisymmetric_bl_vector_to_metric(metric, b0, b3, r, th, cosphi, sinphi);
        const Real bsq_geom = max_val(metric.dot(state.x, bcon, bcon), Real(0));
        if (!(bsq_geom > Real(0))) {
            return coeffs;
        }
        const Real bnorm_geom = Kokkos::sqrt(bsq_geom);
        const Vec4<Real> bunit = bcon * (Real(1) / bnorm_geom);

        const Real rho_cgs = rho_code * density_unit_cgs();
        const Real internal_energy_cgs = p_code / (kappa - Real(1)) * pressure_unit_cgs();
        const Real mp = Real(1.67262192369e-24);
        const Real me = Real(9.1093837015e-28);
        const Real cl = Real(2.99792458e10);
        const Real kbol = Real(1.380649e-16);
        const Real ee = Real(4.803204712570263e-10);
        Real ne_cgs = rho_cgs / mp;
        Real temperature = Real(2) * mp * internal_energy_cgs /
                                 max_val(Real(3) * kbol * rho_cgs * (Real(2) + Rhigh), Real(1e-300));
        Real thetae = max_val(kbol * temperature / (me * cl * cl), thetae_min);
        Real b_cgs = bnorm_geom * bfield_unit_cgs();
        if (!(ne_cgs > Real(0) && temperature > Real(0) && b_cgs > Real(0))) {
            return coeffs;
        }

        const Vec4<Real> ucov = lower_vector(metric, state.x, ucon);
        Real uk = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            uk += ucov[mu] * state.k[mu];
        }
        const Real nu_scale = max_val(abs_val(-uk) / max_val(observer_energy_scale, Real(1e-300)), Real(1e-8));
        const Real nu = max_val(freq_cgs * nu_scale, Real(1));

        const Vec4<Real> bcov = lower_vector(metric, state.x, bunit);
        Real kdotb = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            kdotb += state.k[mu] * bcov[mu];
        }
        const Real cos_theta = clamp(kdotb / max_val(abs_val(-uk), Real(1e-300)), Real(-1), Real(1));
        const Real sin_theta = max_val(Kokkos::sqrt(max_val(Real(0), Real(1) - cos_theta * cos_theta)), Real(1e-8));
        const Real theta = Kokkos::acos(cos_theta);

        Real sigma_local = Real(2) * pm_code / rho_code;
        Real beta_local = beta;
        perturbation.apply(ne_cgs, thetae, b_cgs, sigma_local, beta_local);
        if (perturbation.parameter == PlasmaParameter::temperature_scale)
            temperature *= Kokkos::exp(perturbation.log_scale);

        const Real nu_B = ee * b_cgs / (Real(2) * pi() * me * cl);
        const Real nu_c = max_val(Real(1.5) * nu_B * sin_theta * thetae * thetae, Real(1));
        const Real x = nu / max_val(nu_c, Real(1e-300));
        const Real pref = ne_cgs * ee * ee * nu /
                          (Real(2) * Kokkos::sqrt(Real(3)) * cl * thetae * thetae);
        Real jI_phys = pref * shape_i(x);
        Real jQ_phys = pref * shape_q(x);
        Real tan_theta = Kokkos::tan(theta);
        if (abs_val(tan_theta) < Real(1e-12)) {
            tan_theta = tan_theta >= Real(0) ? Real(1e-12) : Real(-1e-12);
        }
        Real jV_phys = Real(2) * ne_cgs * ee * ee * nu /
                       (Real(3) * Kokkos::sqrt(Real(3)) * cl * thetae * thetae * thetae) *
                       (Real(1) / tan_theta) * shape_v(x);

        const Real Bnu = planck_flux(nu, temperature);
        Real aI_phys = jI_phys / max_val(Bnu, Real(1e-300));
        Real aQ_phys = jQ_phys / max_val(Bnu, Real(1e-300));
        Real aV_phys = jV_phys / max_val(Bnu, Real(1e-300));

        const Real w = Real(2) * pi() * nu;
        const Real wp2 = Real(4) * pi() * ne_cgs * ee * ee / me;
        const Real Omega0 = Real(2) * pi() * nu_B;
        const Real xpar = thetae * Kokkos::sqrt(max_val(Real(0), Kokkos::sqrt(Real(2)) * sin_theta *
                                                        Real(1e3) * Omega0 / max_val(w, Real(1e-300))));
        Real k0 = Real(0), k1 = Real(0), k2 = Real(0);
        thermal_bessel_k0k1k2(Real(1) / thetae, k0, k1, k2);
        const Real diagonal = f_function_D16(max_val(xpar, Real(1e-12))) * wp2 * Omega0 * Omega0 /
                              (w * w * w * w) * (k1 / max_val(k2, Real(1e-300)) + Real(6) * thetae) *
                              sin_theta * sin_theta;
        const Real non_diagonal = wp2 * Omega0 / (w * w * w) *
                                  (k0 - g_function_D16(max_val(xpar, Real(1e-12)))) /
                                  max_val(k2, Real(1e-300)) * cos_theta;
        Real rQ_phys = w / (Real(2) * cl) * diagonal;
        Real rV_phys = w / cl * non_diagonal;

        if (scalar_transport) {
            jQ_phys = jV_phys = Real(0);
            aQ_phys = aV_phys = Real(0);
            rQ_phys = rV_phys = Real(0);
        }

        const Real nusq = nu * nu;
        Real jI_inv = jI_phys / max_val(nusq, Real(1e-300));
        Real jQ_inv = jQ_phys / max_val(nusq, Real(1e-300));
        Real jV_inv = jV_phys / max_val(nusq, Real(1e-300));
        const Real jP = Kokkos::sqrt(jQ_inv * jQ_inv + jV_inv * jV_inv);
        if (jI_inv > Real(0) && jP > Real(0) && jP > max_pol_frac * jI_inv) {
            const Real scale = max_pol_frac * jI_inv / jP;
            jQ_inv *= scale;
            jV_inv *= scale;
        }

        const Real aI_inv = aI_phys * nu;
        const Real aQ_inv = aQ_phys * nu;
        const Real aV_inv = aV_phys * nu;
        const Real rQ_inv = rQ_phys * nu;
        const Real rV_inv = rV_phys * nu;

        coeffs.jI = emission_scale * jI_inv;
        coeffs.jV = emission_scale * jV_inv;
        coeffs.aI = absorption_scale * aI_inv;
        coeffs.aV = absorption_scale * aV_inv;
        coeffs.rV = faraday_scale * rV_inv;

        const Real b1 = screen_inner_product(metric, state.x, ucon, state.k, state.e1, bunit);
        const Real b2 = screen_inner_product(metric, state.x, ucon, state.k, state.e2, bunit);
        const Real bproj2 = b1 * b1 + b2 * b2;
        if (bproj2 > Real(1e-30)) {
            const Real cos2chi = (b2 * b2 - b1 * b1) / bproj2;
            const Real sin2chi = -Real(2) * b1 * b2 / bproj2;
            coeffs.jQ = emission_scale * jQ_inv * cos2chi;
            coeffs.jU = emission_scale * jQ_inv * sin2chi;
            coeffs.aQ = absorption_scale * aQ_inv * cos2chi;
            coeffs.aU = absorption_scale * aQ_inv * sin2chi;
            coeffs.rQ = faraday_scale * rQ_inv * cos2chi;
            coeffs.rU = faraday_scale * rQ_inv * sin2chi;
        }
        return coeffs;
    }
};

} // namespace kpolaris
