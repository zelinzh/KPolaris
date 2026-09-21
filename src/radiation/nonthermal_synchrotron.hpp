#pragma once

#include "common/math.hpp"
#include "radiation/stokes.hpp"
#include "radiation/thermal_synchrotron.hpp"

namespace kpolaris {

enum SynchrotronEmissionType : int {
    SynchrotronThermalPandya = 1,
    SynchrotronKappa = 2,
    SynchrotronPowerLaw = 3,
    SynchrotronThermalDexter = 4
};

template<class Real = DefaultReal>
struct NonthermalSynchrotronParams {
    int distribution = SynchrotronPowerLaw;
    Real max_pol_frac_emission = Real(0.99);
    Real max_pol_frac_absorption = Real(0.99);
    Real min_sin_theta = Real(1e-4);
    Real emission_scale = Real(1);
    Real absorption_scale = Real(1);
    Real faraday_scale = Real(1);

    Real power_law_p = Real(3.25);
    Real power_law_eta = Real(0.02);
    Real power_law_gamma_min = Real(1e2);
    Real power_law_gamma_max = Real(1e5);
    Real power_law_gamma_cutoff = Real(1e10);

    Real kappa = Real(3.5);
    Real kappa_width = Real(-1);
    Real kappa_interp_begin = Real(1e20);
    Real kappa_interp_end = Real(7.0);
};

template<class Real = DefaultReal>
struct LocalNonthermalSynchrotronState {
    Real nu = Real(0);
    Real ne = Real(0);
    Real thetae = Real(0);
    Real b_cgs = Real(0);
    Real sin_theta = Real(0);
    Real cos_theta = Real(1);
};

namespace detail {

template<class Real>
KPOLARIS_INLINE Real gamma_fn(Real x) {
    return Real(::tgamma(double(x)));
}

template<class Real>
KPOLARIS_INLINE Real hyp2f1_series(Real a, Real b, Real c, Real z) {
    Real term = Real(1);
    Real sum = Real(1);
    for (int n = 1; n <= 200; ++n) {
        const Real rn = Real(n);
        Real denom = (c + rn - Real(1)) * rn;
        if (abs_val(denom) < tiny_positive<Real>()) {
            denom = denom >= Real(0) ? tiny_positive<Real>() : -tiny_positive<Real>();
        }
        term *= (a + rn - Real(1)) * (b + rn - Real(1)) * z / denom;
        sum += term;
        if (abs_val(term) < abs_val(sum) * Real(1e-13)) {
            break;
        }
    }
    return sum;
}

template<class Real>
KPOLARIS_INLINE Real signed_nonzero(Real x) {
    if (abs_val(x) < tiny_positive<Real>()) {
        return x >= Real(0) ? tiny_positive<Real>() : -tiny_positive<Real>();
    }
    return x;
}

template<class Real>
KPOLARIS_INLINE Real stable_hyp2f1(Real a, Real b, Real c, Real z) {
    if (z > Real(-1) && z < Real(1)) {
        return hyp2f1_series(a, b, c, z);
    }
    const Real zp = Real(1) / (Real(1) - z);
    const Real one_minus_z = Real(1) - z;
    const Real denom1 = signed_nonzero(gamma_fn(b) * gamma_fn(c - a));
    const Real denom2 = signed_nonzero(gamma_fn(a) * gamma_fn(c - b));
    const Real p1 = Kokkos::pow(one_minus_z, -a) * gamma_fn(c) * gamma_fn(b - a) / denom1 *
                    hyp2f1_series(a, c - b, a - b + Real(1), zp);
    const Real p2 = Kokkos::pow(one_minus_z, -b) * gamma_fn(c) * gamma_fn(a - b) / denom2 *
                    hyp2f1_series(b, c - a, b - a + Real(1), zp);
    return p1 + p2;
}

template<class Real>
KPOLARIS_INLINE Real nu_c(Real b_cgs) {
    const Real ee = Real(4.80320680e-10);
    const Real me = Real(9.1093826e-28);
    const Real cl = Real(2.99792458e10);
    const Real pi = Real(3.141592653589793238462643383279502884);
    return ee * b_cgs / (Real(2) * pi * me * cl);
}

template<class Real>
KPOLARIS_INLINE Real power_law_density_from_eta(Real b_cgs, const NonthermalSynchrotronParams<Real>& p) {
    const Real me = Real(9.1093826e-28);
    const Real cl = Real(2.99792458e10);
    if (!(p.power_law_p > Real(2)) || !(p.power_law_gamma_min >= Real(1))) {
        return Real(0);
    }
    const Real u_nth = p.power_law_eta * b_cgs * b_cgs / Real(2);
    return u_nth * (p.power_law_p - Real(2)) / (p.power_law_p - Real(1)) /
           max_val(me * cl * cl * p.power_law_gamma_min, tiny_positive<Real>());
}

template<class Real>
KPOLARIS_INLINE Real power_law_I(const LocalNonthermalSynchrotronState<Real>& s,
                              const NonthermalSynchrotronParams<Real>& p,
                              Real ne_pl) {
    const Real ee = Real(4.80320680e-10);
    const Real cl = Real(2.99792458e10);
    const Real nuc = nu_c(s.b_cgs);
    const Real sint = max_val(s.sin_theta, p.min_sin_theta);
    const Real pp = p.power_law_p;
    if (!(s.nu > Real(0)) || !(nuc > Real(0)) || !(ne_pl > Real(0)) || !(pp > Real(1))) return Real(0);
    const Real pref = ne_pl * ee * ee * nuc / cl;
    const Real term1 = Kokkos::pow(Real(3), pp / Real(2)) * (pp - Real(1)) * sint;
    const Real term2 = Real(2) * (pp + Real(1)) *
        (Kokkos::pow(p.power_law_gamma_min, Real(1) - pp) -
         Kokkos::pow(p.power_law_gamma_max, Real(1) - pp));
    const Real term3 = gamma_fn((Real(3) * pp - Real(1)) / Real(12)) *
                       gamma_fn((Real(3) * pp + Real(19)) / Real(12));
    const Real term4 = Kokkos::pow(s.nu / max_val(nuc * sint, tiny_positive<Real>()),
                                   -(pp - Real(1)) / Real(2));
    return pref * term1 / max_val(term2, tiny_positive<Real>()) * term3 * term4;
}

template<class Real>
KPOLARIS_INLINE Real power_law_Q(const LocalNonthermalSynchrotronState<Real>& s,
                              const NonthermalSynchrotronParams<Real>& p,
                              Real ne_pl) {
    return -(p.power_law_p + Real(1)) / (p.power_law_p + Real(7) / Real(3)) *
           power_law_I(s, p, ne_pl);
}

template<class Real>
KPOLARIS_INLINE Real power_law_V(const LocalNonthermalSynchrotronState<Real>& s,
                              const NonthermalSynchrotronParams<Real>& p,
                              Real ne_pl) {
    const Real nuc = nu_c(s.b_cgs);
    const Real sint = max_val(s.sin_theta, p.min_sin_theta);
    Real cost = s.cos_theta;
    if (abs_val(cost) < Real(1e-12)) cost = cost >= Real(0) ? Real(1e-12) : Real(-1e-12);
    const Real term1 = -(Real(171) / Real(250)) * Kokkos::pow(p.power_law_p, Real(49) / Real(100));
    const Real term2 = cost / sint * Kokkos::pow(s.nu / max_val(Real(3) * nuc * sint, tiny_positive<Real>()), Real(-0.5));
    return -term1 * term2 * power_law_I(s, p, ne_pl);
}

template<class Real>
KPOLARIS_INLINE Real power_law_I_abs(const LocalNonthermalSynchrotronState<Real>& s,
                                  const NonthermalSynchrotronParams<Real>& p,
                                  Real ne_pl) {
    const Real ee = Real(4.80320680e-10);
    const Real me = Real(9.1093826e-28);
    const Real cl = Real(2.99792458e10);
    const Real nuc = nu_c(s.b_cgs);
    const Real sint = max_val(s.sin_theta, p.min_sin_theta);
    const Real pp = p.power_law_p;
    if (!(s.nu > Real(0)) || !(nuc > Real(0)) || !(ne_pl > Real(0)) || !(pp > Real(1))) return Real(0);
    const Real pref = ne_pl * ee * ee / max_val(s.nu * me * cl, tiny_positive<Real>());
    const Real term1 = Kokkos::pow(Real(3), (pp + Real(1)) / Real(2)) * (pp - Real(1));
    const Real term2 = Real(4) * (Kokkos::pow(p.power_law_gamma_min, Real(1) - pp) -
                                  Kokkos::pow(p.power_law_gamma_max, Real(1) - pp));
    const Real term3 = gamma_fn((Real(3) * pp + Real(2)) / Real(12)) *
                       gamma_fn((Real(3) * pp + Real(22)) / Real(12));
    const Real term4 = Kokkos::pow(s.nu / max_val(nuc * sint, tiny_positive<Real>()),
                                   -(pp + Real(2)) / Real(2));
    return pref * term1 / max_val(term2, tiny_positive<Real>()) * term3 * term4;
}

template<class Real>
KPOLARIS_INLINE Real power_law_Q_abs(const LocalNonthermalSynchrotronState<Real>& s,
                                  const NonthermalSynchrotronParams<Real>& p,
                                  Real ne_pl) {
    const Real term5 = -Kokkos::pow((Real(17) / Real(500)) * p.power_law_p - Real(43) / Real(1250),
                                    Real(43) / Real(500));
    return power_law_I_abs(s, p, ne_pl) * term5;
}

template<class Real>
KPOLARIS_INLINE Real power_law_V_abs(const LocalNonthermalSynchrotronState<Real>& s,
                                  const NonthermalSynchrotronParams<Real>& p,
                                  Real ne_pl) {
    const Real nuc = nu_c(s.b_cgs);
    const Real sint = max_val(s.sin_theta, p.min_sin_theta);
    const Real term5 = -Kokkos::pow((Real(71) / Real(100)) * p.power_law_p + Real(22) / Real(625),
                                    Real(197) / Real(500));
    const Real term6_base = max_val((Real(31) / Real(10)) * Kokkos::pow(sint, -Real(48) / Real(25)) -
                                    Real(31) / Real(10), Real(0));
    const Real term6 = Kokkos::pow(term6_base, Real(64) / Real(125));
    const Real term7 = Kokkos::pow(s.nu / max_val(nuc * sint, tiny_positive<Real>()), Real(-0.5));
    const Real sign = s.cos_theta >= Real(0) ? Real(1) : Real(-1);
    return -power_law_I_abs(s, p, ne_pl) * term5 * term6 * term7 * sign;
}

template<class Real>
KPOLARIS_INLINE Real kappa_width_value(const LocalNonthermalSynchrotronState<Real>& s,
                                    const NonthermalSynchrotronParams<Real>& p) {
    if (p.kappa_width > Real(0)) return p.kappa_width;
    return (p.kappa - Real(3)) / max_val(p.kappa, tiny_positive<Real>()) * s.thetae;
}

template<class Real>
KPOLARIS_INLINE Real kappa_X(const LocalNonthermalSynchrotronState<Real>& s,
                          const NonthermalSynchrotronParams<Real>& p,
                          Real w) {
    const Real nuc = nu_c(s.b_cgs);
    const Real sint = max_val(s.sin_theta, p.min_sin_theta);
    const Real nuw = w * w * p.kappa * p.kappa * nuc * sint;
    return s.nu / max_val(nuw, tiny_positive<Real>());
}

template<class Real>
KPOLARIS_INLINE Real kappa_I(const LocalNonthermalSynchrotronState<Real>& s,
                          const NonthermalSynchrotronParams<Real>& p) {
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real ee = Real(4.80320680e-10);
    const Real cl = Real(2.99792458e10);
    const Real nuc = nu_c(s.b_cgs);
    const Real sint = max_val(s.sin_theta, p.min_sin_theta);
    const Real w = kappa_width_value(s, p);
    const Real X = kappa_X(s, p, w);
    if (!(p.kappa > Real(2)) || !(w > Real(0)) || !(X > Real(0))) return Real(0);
    const Real pref = s.ne * ee * ee * nuc * sint / cl;
    const Real Nlow = Real(4) * pi * gamma_fn(p.kappa - Real(4) / Real(3)) /
                      (Kokkos::pow(Real(3), Real(7) / Real(3)) * gamma_fn(p.kappa - Real(2)));
    const Real Nhigh = Real(0.25) * Kokkos::pow(Real(3), (p.kappa - Real(1)) / Real(2)) *
        (p.kappa - Real(2)) * (p.kappa - Real(1)) *
        gamma_fn(p.kappa / Real(4) - Real(1) / Real(3)) *
        gamma_fn(p.kappa / Real(4) + Real(4) / Real(3));
    const Real x = Real(3) * Kokkos::pow(p.kappa, -Real(1.5));
    return pref * Nlow * Kokkos::pow(X, Real(1) / Real(3)) *
           Kokkos::pow(Real(1) + Kokkos::pow(X, x * (Real(3) * p.kappa - Real(4)) / Real(6)) *
                       Kokkos::pow(Nlow / Nhigh, x), -Real(1) / x);
}

template<class Real>
KPOLARIS_INLINE Real kappa_Q(const LocalNonthermalSynchrotronState<Real>& s,
                          const NonthermalSynchrotronParams<Real>& p) {
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real ee = Real(4.80320680e-10);
    const Real cl = Real(2.99792458e10);
    const Real nuc = nu_c(s.b_cgs);
    const Real sint = max_val(s.sin_theta, p.min_sin_theta);
    const Real w = kappa_width_value(s, p);
    const Real X = kappa_X(s, p, w);
    if (!(p.kappa > Real(2)) || !(w > Real(0)) || !(X > Real(0))) return Real(0);
    const Real pref = s.ne * ee * ee * nuc * sint / cl;
    const Real Nlow = -Real(0.5) * Real(4) * pi * gamma_fn(p.kappa - Real(4) / Real(3)) /
                      (Kokkos::pow(Real(3), Real(7) / Real(3)) * gamma_fn(p.kappa - Real(2)));
    const Real Nhigh = -(Kokkos::pow(Real(4) / Real(5), Real(2)) + p.kappa / Real(50)) * Real(0.25) *
        Kokkos::pow(Real(3), (p.kappa - Real(1)) / Real(2)) * (p.kappa - Real(2)) *
        (p.kappa - Real(1)) * gamma_fn(p.kappa / Real(4) - Real(1) / Real(3)) *
        gamma_fn(p.kappa / Real(4) + Real(4) / Real(3));
    const Real x = Real(3.7) * Kokkos::pow(p.kappa, -Real(1.6));
    return pref * Nlow * Kokkos::pow(X, Real(1) / Real(3)) *
           Kokkos::pow(Real(1) + Kokkos::pow(X, x * (Real(3) * p.kappa - Real(4)) / Real(6)) *
                       Kokkos::pow(Nlow / Nhigh, x), -Real(1) / x);
}

template<class Real>
KPOLARIS_INLINE Real kappa_V(const LocalNonthermalSynchrotronState<Real>& s,
                          const NonthermalSynchrotronParams<Real>& p) {
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real ee = Real(4.80320680e-10);
    const Real cl = Real(2.99792458e10);
    const Real nuc = nu_c(s.b_cgs);
    const Real sint = max_val(s.sin_theta, p.min_sin_theta);
    const Real w = kappa_width_value(s, p);
    const Real X = kappa_X(s, p, w);
    if (!(p.kappa > Real(2)) || !(w > Real(0)) || !(X > Real(0))) return Real(0);
    const Real pref = s.ne * ee * ee * nuc * sint / cl;
    const Real angle1 = max_val(Kokkos::pow(sint, -Real(12) / Real(5)) - Real(1), Real(0));
    const Real angle2 = max_val(Kokkos::pow(sint, -Real(5) / Real(2)) - Real(1), Real(0));
    const Real Nlow = -Kokkos::pow(Real(3) / Real(4), Real(2)) * Kokkos::pow(angle1, Real(12) / Real(25)) *
        Kokkos::pow(p.kappa, -Real(66) / Real(125)) / w * Kokkos::pow(X, -Real(7) / Real(20)) *
        Real(4) * pi * gamma_fn(p.kappa - Real(4) / Real(3)) /
        (Kokkos::pow(Real(3), Real(7) / Real(3)) * gamma_fn(p.kappa - Real(2)));
    const Real Nhigh = -Kokkos::pow(Real(7) / Real(8), Real(2)) * Kokkos::pow(angle2, Real(11) / Real(25)) *
        Kokkos::pow(p.kappa, -Real(11) / Real(25)) / w * Kokkos::pow(X, -Real(0.5)) * Real(0.25) *
        Kokkos::pow(Real(3), (p.kappa - Real(1)) / Real(2)) * (p.kappa - Real(2)) *
        (p.kappa - Real(1)) * gamma_fn(p.kappa / Real(4) - Real(1) / Real(3)) *
        gamma_fn(p.kappa / Real(4) + Real(4) / Real(3));
    const Real x = Real(3) * Kokkos::pow(p.kappa, -Real(1.5));
    const Real ans = pref * Nlow * Kokkos::pow(X, Real(1) / Real(3)) *
        Kokkos::pow(Real(1) + Kokkos::pow(X, x * (Real(3) * p.kappa - Real(4)) / Real(6)) *
                    Kokkos::pow(Nlow / Nhigh, x), -Real(1) / x);
    const Real sign = s.cos_theta >= Real(0) ? Real(1) : Real(-1);
    return -ans * sign;
}

template<class Real>
KPOLARIS_INLINE void kappa_abs_common(const LocalNonthermalSynchrotronState<Real>& s,
                                   const NonthermalSynchrotronParams<Real>& p,
                                   Real& X, Real& pref, Real& hyp) {
    const Real ee = Real(4.80320680e-10);
    const Real sint = max_val(s.sin_theta, p.min_sin_theta);
    const Real w = kappa_width_value(s, p);
    X = kappa_X(s, p, w);
    pref = s.ne * ee / max_val(s.b_cgs * sint, tiny_positive<Real>());
    hyp = stable_hyp2f1(p.kappa - Real(1) / Real(3), p.kappa + Real(1),
                        p.kappa + Real(2) / Real(3), -p.kappa * w);
}

template<class Real>
KPOLARIS_INLINE Real kappa_I_abs(const LocalNonthermalSynchrotronState<Real>& s,
                              const NonthermalSynchrotronParams<Real>& p) {
    const Real pi = Real(3.141592653589793238462643383279502884);
    Real X, pref, hyp;
    kappa_abs_common(s, p, X, pref, hyp);
    const Real w = kappa_width_value(s, p);
    if (!(p.kappa > Real(2)) || !(w > Real(0)) || !(X > Real(0)) || !(abs_val(hyp) > tiny_positive<Real>())) return Real(0);
    const Real Nlow = Kokkos::pow(Real(3), Real(1) / Real(6)) * Real(10) / Real(41) *
        Kokkos::pow(Real(2) * pi, Real(2)) / Kokkos::pow(w * p.kappa, Real(16) / Real(3) - p.kappa) *
        (p.kappa - Real(2)) * (p.kappa - Real(1)) * p.kappa / (Real(3) * p.kappa - Real(1)) *
        gamma_fn(Real(5) / Real(3)) * hyp;
    const Real Nhigh = Real(2) * Kokkos::pow(pi, Real(2.5)) / Real(3) * (p.kappa - Real(2)) *
        (p.kappa - Real(1)) * p.kappa / Kokkos::pow(w * p.kappa, Real(5)) *
        (Real(2) * gamma_fn(Real(2) + p.kappa / Real(2)) / (Real(2) + p.kappa) - Real(1)) *
        (Kokkos::pow(Real(3) / p.kappa, Real(19) / Real(4)) + Real(3) / Real(5)) + tiny_positive<Real>();
    const Real x = Kokkos::pow(-Real(7) / Real(4) + Real(8) * p.kappa / Real(5), -Real(43) / Real(50));
    return pref * Nlow * Kokkos::pow(X, -Real(5) / Real(3)) *
        Kokkos::pow(Real(1) + Kokkos::pow(X, x * (Real(3) * p.kappa - Real(1)) / Real(6)) *
                    Kokkos::pow(Nlow / Nhigh, x), -Real(1) / x);
}

template<class Real>
KPOLARIS_INLINE Real kappa_Q_abs(const LocalNonthermalSynchrotronState<Real>& s,
                              const NonthermalSynchrotronParams<Real>& p) {
    const Real pi = Real(3.141592653589793238462643383279502884);
    Real X, pref, hyp;
    kappa_abs_common(s, p, X, pref, hyp);
    const Real w = kappa_width_value(s, p);
    if (!(p.kappa > Real(2)) || !(w > Real(0)) || !(X > Real(0)) || !(abs_val(hyp) > tiny_positive<Real>())) return Real(0);
    const Real Nlow = -Real(25) / Real(48) * Kokkos::pow(Real(3), Real(1) / Real(6)) * Real(10) / Real(41) *
        Kokkos::pow(Real(2) * pi, Real(2)) / Kokkos::pow(w * p.kappa, Real(16) / Real(3) - p.kappa) *
        (p.kappa - Real(2)) * (p.kappa - Real(1)) * p.kappa / (Real(3) * p.kappa - Real(1)) *
        gamma_fn(Real(5) / Real(3)) * hyp;
    const Real Nhigh = -(Kokkos::pow(Real(21), Real(2)) * Kokkos::pow(p.kappa, -Real(144) / Real(25)) + Real(11) / Real(20)) *
        Real(2) * Kokkos::pow(pi, Real(2.5)) / Real(3) * (p.kappa - Real(2)) * (p.kappa - Real(1)) *
        p.kappa / Kokkos::pow(w * p.kappa, Real(5)) *
        (Real(2) * gamma_fn(Real(2) + p.kappa / Real(2)) / (Real(2) + p.kappa) - Real(1)) + tiny_positive<Real>();
    const Real x = Real(7) / Real(5) * Kokkos::pow(p.kappa, -Real(23) / Real(20));
    return pref * Nlow * Kokkos::pow(X, -Real(5) / Real(3)) *
        Kokkos::pow(Real(1) + Kokkos::pow(X, x * (Real(3) * p.kappa - Real(1)) / Real(6)) *
                    Kokkos::pow(Nlow / Nhigh, x), -Real(1) / x);
}

template<class Real>
KPOLARIS_INLINE Real kappa_V_abs(const LocalNonthermalSynchrotronState<Real>& s,
                              const NonthermalSynchrotronParams<Real>& p) {
    const Real pi = Real(3.141592653589793238462643383279502884);
    Real X, pref, hyp;
    kappa_abs_common(s, p, X, pref, hyp);
    const Real w = kappa_width_value(s, p);
    const Real sint = max_val(s.sin_theta, p.min_sin_theta);
    if (!(p.kappa > Real(2)) || !(w > Real(0)) || !(X > Real(0)) || !(abs_val(hyp) > tiny_positive<Real>())) return Real(0);
    const Real angle1 = max_val(Kokkos::pow(sint, -Real(114) / Real(50)) - Real(1), Real(0));
    const Real angle2 = max_val(Kokkos::pow(sint, -Real(41) / Real(20)) - Real(1), Real(0));
    const Real Nlow = -Real(77) / (Real(100) * w) * Kokkos::pow(angle1, Real(223) / Real(500)) *
        Kokkos::pow(X, -Real(7) / Real(20)) * Kokkos::pow(p.kappa, -Real(7) / Real(10)) *
        Kokkos::pow(Real(3), Real(1) / Real(6)) * Real(10) / Real(41) * Kokkos::pow(Real(2) * pi, Real(2)) /
        Kokkos::pow(w * p.kappa, Real(16) / Real(3) - p.kappa) * (p.kappa - Real(2)) *
        (p.kappa - Real(1)) * p.kappa / (Real(3) * p.kappa - Real(1)) * gamma_fn(Real(5) / Real(3)) * hyp;
    const Real nfac = Real(13) * Real(13) * Kokkos::pow(p.kappa, -Real(8)) + Real(13) / Real(2500) * p.kappa -
                      Real(263) / Real(5000) + Real(47) / (Real(200) * p.kappa);
    const Real Nhigh = -Real(143) / Real(10) * Kokkos::pow(w, -Real(116) / Real(125)) *
        Kokkos::pow(angle2, Real(0.5)) * nfac * Kokkos::pow(X, -Real(0.5)) * Real(2) *
        Kokkos::pow(pi, Real(2.5)) / Real(3) * (p.kappa - Real(2)) * (p.kappa - Real(1)) * p.kappa /
        Kokkos::pow(w * p.kappa, Real(5)) *
        (Real(2) * gamma_fn(Real(2) + p.kappa / Real(2)) / (Real(2) + p.kappa) - Real(1)) + tiny_positive<Real>();
    const Real x = Real(61) / Real(50) * Kokkos::pow(p.kappa, -Real(142) / Real(125)) + Real(7) / Real(1000);
    const Real ans = pref * Nlow * Kokkos::pow(X, -Real(5) / Real(3)) *
        Kokkos::pow(Real(1) + Kokkos::pow(X, x * (Real(3) * p.kappa - Real(1)) / Real(6)) *
                    Kokkos::pow(Nlow / Nhigh, x), -Real(1) / x);
    const Real sign = s.cos_theta >= Real(0) ? Real(1) : Real(-1);
    return -ans * sign;
}

template<class Real>
KPOLARIS_INLINE Real kappa_rho_Q_at(const LocalNonthermalSynchrotronState<Real>& s,
                                 const NonthermalSynchrotronParams<Real>& p,
                                 Real kappa_anchor) {
    const Real w = kappa_width_value(s, p);
    const Real X = kappa_X(s, p, w);
    const Real ee = Real(4.80320680e-10);
    const Real me = Real(9.1093826e-28);
    const Real cl = Real(2.99792458e10);
    const Real nuc = nu_c(s.b_cgs);
    const Real sint = max_val(s.sin_theta, p.min_sin_theta);
    if (!(X > Real(0))) return Real(0);
    const Real pref = -(s.ne * ee * ee * nuc * nuc * sint * sint) /
                      max_val(me * cl * s.nu * s.nu * s.nu, tiny_positive<Real>());
    Real wterm, fx;
    if (kappa_anchor < Real(3.75)) {
        wterm = Real(17) * w - Real(3) * Kokkos::sqrt(w) + Real(7) * Kokkos::sqrt(w) * Kokkos::exp(-Real(5) * w);
        fx = Real(1) - Kokkos::exp(-Kokkos::pow(X, Real(0.84)) / Real(30)) -
             Kokkos::sin(X / Real(10)) * Kokkos::exp(-Real(1.5) * Kokkos::pow(X, Real(0.471)));
    } else if (kappa_anchor < Real(4.25)) {
        wterm = Real(46) / Real(3) * w - Real(5) / Real(3) * Kokkos::sqrt(w) +
                Real(17) / Real(3) * Kokkos::sqrt(w) * Kokkos::exp(-Real(5) * w);
        fx = Real(1) - Kokkos::exp(-Kokkos::pow(X, Real(0.84)) / Real(18)) -
             Kokkos::sin(X / Real(6)) * Kokkos::exp(-Real(7) * Kokkos::sqrt(X) / Real(4));
    } else if (kappa_anchor < Real(4.75)) {
        wterm = Real(14) * w - Real(13) / Real(8) * Kokkos::sqrt(w) +
                Real(9) / Real(2) * Kokkos::sqrt(w) * Kokkos::exp(-Real(5) * w);
        fx = Real(1) - Kokkos::exp(-Kokkos::pow(X, Real(0.84)) / Real(12)) -
             Kokkos::sin(X / Real(4)) * Kokkos::exp(-Real(2) * Kokkos::pow(X, Real(0.525)));
    } else {
        wterm = Real(25) / Real(2) * w - Kokkos::sqrt(w) + Real(5) * Kokkos::sqrt(w) * Kokkos::exp(-Real(5) * w);
        fx = Real(1) - Kokkos::exp(-Kokkos::pow(X, Real(0.84)) / Real(8)) -
             Kokkos::sin(Real(3) * X / Real(8)) * Kokkos::exp(-Real(9) * Kokkos::pow(X, Real(0.541)) / Real(4));
    }
    return pref * fx * wterm;
}

template<class Real>
KPOLARIS_INLINE Real kappa_rho_V_at(const LocalNonthermalSynchrotronState<Real>& s,
                                 const NonthermalSynchrotronParams<Real>& p,
                                 Real kappa_anchor) {
    const Real w = kappa_width_value(s, p);
    const Real X = kappa_X(s, p, w);
    const Real ee = Real(4.80320680e-10);
    const Real me = Real(9.1093826e-28);
    const Real cl = Real(2.99792458e10);
    const Real nuc = nu_c(s.b_cgs);
    if (!(X > Real(0)) || !(w > Real(0))) return Real(0);
    const Real pref = Real(2) * s.ne * ee * ee * nuc * s.cos_theta /
                      max_val(me * cl * s.nu * s.nu, tiny_positive<Real>());
    Real k0 = Real(0), k2 = Real(0);
    thermal_bessel_k0k2(Real(1) / w, k0, k2);
    const Real bessel_term = k0 / max_val(k2 + tiny_positive<Real>(), tiny_positive<Real>());
    Real wterm, gx;
    if (kappa_anchor < Real(3.75)) {
        wterm = (w * w + Real(2) * w + Real(1)) / (Real(25) / Real(8) * w * w + Real(4) * w + Real(1));
        gx = Real(1) - Real(0.17) * Kokkos::log(Real(1) + Real(0.447) * Kokkos::pow(X, Real(-0.5)));
    } else if (kappa_anchor < Real(4.25)) {
        wterm = (w * w + Real(54) * w + Real(50)) / (Real(30) / Real(11) * w * w + Real(134) * w + Real(50));
        gx = Real(1) - Real(0.17) * Kokkos::log(Real(1) + Real(0.391) * Kokkos::pow(X, Real(-0.5)));
    } else if (kappa_anchor < Real(4.75)) {
        wterm = (w * w + Real(43) * w + Real(38)) / (Real(7) / Real(3) * w * w + Real(185) / Real(2) * w + Real(38));
        gx = Real(1) - Real(0.17) * Kokkos::log(Real(1) + Real(0.348) * Kokkos::pow(X, Real(-0.5)));
    } else {
        wterm = (w + Real(13) / Real(14)) / (Real(2) * w + Real(13) / Real(14));
        gx = Real(1) - Real(0.17) * Kokkos::log(Real(1) + Real(0.313) * Kokkos::pow(X, Real(-0.5)));
    }
    return pref * bessel_term * gx * wterm;
}

template<class Real>
KPOLARIS_INLINE Real interpolate_kappa_rho_Q(const LocalNonthermalSynchrotronState<Real>& s,
                                          const NonthermalSynchrotronParams<Real>& p) {
    // Symphony's fast rho_nu_fit is defined only for 3.5 <= kappa <= 5.
    // Outside that interval the original fit wrapper returns 0 rather than
    // clamping or blending toward a thermal rotativity.
    if (p.kappa < Real(3.5) || p.kappa > Real(5.0)) return Real(0);
    if (p.kappa < Real(4.0)) return ((Real(4.0) - p.kappa) * kappa_rho_Q_at(s, p, Real(3.5)) + (p.kappa - Real(3.5)) * kappa_rho_Q_at(s, p, Real(4.0))) / Real(0.5);
    if (p.kappa < Real(4.5)) return ((Real(4.5) - p.kappa) * kappa_rho_Q_at(s, p, Real(4.0)) + (p.kappa - Real(4.0)) * kappa_rho_Q_at(s, p, Real(4.5))) / Real(0.5);
    if (p.kappa < Real(5.0)) return ((Real(5.0) - p.kappa) * kappa_rho_Q_at(s, p, Real(4.5)) + (p.kappa - Real(4.5)) * kappa_rho_Q_at(s, p, Real(5.0))) / Real(0.5);
    return kappa_rho_Q_at(s, p, Real(5.0));
}

template<class Real>
KPOLARIS_INLINE Real interpolate_kappa_rho_V(const LocalNonthermalSynchrotronState<Real>& s,
                                          const NonthermalSynchrotronParams<Real>& p) {
    // Symphony's fast rho_nu_fit is defined only for 3.5 <= kappa <= 5.
    // Outside that interval the original fit wrapper returns 0 rather than
    // clamping or blending toward a thermal rotativity.
    if (p.kappa < Real(3.5) || p.kappa > Real(5.0)) return Real(0);
    if (p.kappa < Real(4.0)) return ((Real(4.0) - p.kappa) * kappa_rho_V_at(s, p, Real(3.5)) + (p.kappa - Real(3.5)) * kappa_rho_V_at(s, p, Real(4.0))) / Real(0.5);
    if (p.kappa < Real(4.5)) return ((Real(4.5) - p.kappa) * kappa_rho_V_at(s, p, Real(4.0)) + (p.kappa - Real(4.0)) * kappa_rho_V_at(s, p, Real(4.5))) / Real(0.5);
    if (p.kappa < Real(5.0)) return ((Real(5.0) - p.kappa) * kappa_rho_V_at(s, p, Real(4.5)) + (p.kappa - Real(4.5)) * kappa_rho_V_at(s, p, Real(5.0))) / Real(0.5);
    return kappa_rho_V_at(s, p, Real(5.0));
}

} // namespace detail

template<class Real>
KPOLARIS_INLINE Real variable_kappa_from_sigma_beta(Real sigma, Real beta,
                                                 const NonthermalSynchrotronParams<Real>& params) {
    if (!(sigma > Real(0)) || !(beta >= Real(0))) {
        return params.kappa;
    }
    Real kappa = Real(2.8) + Real(0.7) / Kokkos::sqrt(sigma) +
        Real(3.7) * Kokkos::pow(sigma, Real(-0.19)) *
        Kokkos::tanh(Real(23.4) * Kokkos::pow(sigma, Real(0.26)) * beta);
    if (kappa < Real(3.1)) kappa = Real(3.1);
    return kappa;
}

template<class Real>
KPOLARIS_INLINE TransferCoeffs<Real>
nonthermal_synchrotron_magnetic_basis_coefficients(
    const LocalNonthermalSynchrotronState<Real>& state,
    NonthermalSynchrotronParams<Real> params) {
    TransferCoeffs<Real> coeffs;
    if (!(state.nu > Real(0)) || !(state.b_cgs > Real(0)) || !(state.ne > Real(0))) {
        return coeffs;
    }
    const Real nu = max_val(state.nu, Real(1));
    const Real nusq = nu * nu;

    if (params.distribution == SynchrotronPowerLaw) {
        const Real ne_pl = detail::power_law_density_from_eta(state.b_cgs, params);
        Real jI = detail::power_law_I(state, params, ne_pl) / max_val(nusq, tiny_positive<Real>());
        Real jQ = -detail::power_law_Q(state, params, ne_pl) / max_val(nusq, tiny_positive<Real>());
        Real jV = detail::power_law_V(state, params, ne_pl) / max_val(nusq, tiny_positive<Real>());
        const Real jP = Kokkos::sqrt(jQ * jQ + jV * jV);
        if (jI > Real(0) && jP > Real(0) && jI < jP / max_val(params.max_pol_frac_emission, Real(1e-30))) {
            const Real scale = jI / jP * params.max_pol_frac_emission;
            jQ *= scale;
            jV *= scale;
        }
        Real aI = detail::power_law_I_abs(state, params, ne_pl) * nu;
        Real aQ = -detail::power_law_Q_abs(state, params, ne_pl) * nu;
        Real aV = detail::power_law_V_abs(state, params, ne_pl) * nu;
        const Real aP = Kokkos::sqrt(aQ * aQ + aV * aV);
        if (aI > Real(0) && aP > Real(0) && aI < aP / max_val(params.max_pol_frac_absorption, Real(1e-30))) {
            const Real scale = aI / aP * params.max_pol_frac_absorption;
            aQ *= scale;
            aV *= scale;
        }
        coeffs.jI = params.emission_scale * jI;
        coeffs.jQ = params.emission_scale * jQ;
        coeffs.jV = params.emission_scale * jV;
        coeffs.aI = params.absorption_scale * aI;
        coeffs.aQ = params.absorption_scale * aQ;
        coeffs.aV = params.absorption_scale * aV;
        // Symphony's fast rho_nu_fit does not include power-law rotativity.
        // The original rho_nu susceptibility integrator can evaluate it, but
        // it is too expensive for per-step transport and is not used here.
        return coeffs;
    }

    if (params.distribution == SynchrotronKappa) {
        if (params.kappa_width <= Real(0)) {
            params.kappa_width = (params.kappa - Real(3)) / max_val(params.kappa, tiny_positive<Real>()) * state.thetae;
        }
        ThermalSynchrotronParams<Real> thermal_params;
        thermal_params.fit = ThermalSynchrotronPandya;
        thermal_params.max_pol_frac = params.max_pol_frac_emission;
        thermal_params.min_sin_theta = params.min_sin_theta;
        LocalThermalSynchrotronState<Real> thermal_state;
        thermal_state.nu = state.nu;
        thermal_state.ne = state.ne;
        thermal_state.thetae = state.thetae;
        thermal_state.b_cgs = state.b_cgs;
        thermal_state.sin_theta = state.sin_theta;
        thermal_state.cos_theta = state.cos_theta;
        const TransferCoeffs<Real> thermal =
            thermal_synchrotron_magnetic_basis_coefficients_fit<ThermalSynchrotronPandya>(thermal_state, thermal_params);

        TransferCoeffs<Real> kappa;
        kappa.jI = detail::kappa_I(state, params) / max_val(nusq, tiny_positive<Real>());
        kappa.jQ = -detail::kappa_Q(state, params) / max_val(nusq, tiny_positive<Real>());
        kappa.jV = detail::kappa_V(state, params) / max_val(nusq, tiny_positive<Real>());
        kappa.aI = detail::kappa_I_abs(state, params) * nu;
        kappa.aQ = -detail::kappa_Q_abs(state, params) * nu;
        kappa.aV = detail::kappa_V_abs(state, params) * nu;
        kappa.rQ = detail::interpolate_kappa_rho_Q(state, params) * nu;
        kappa.rV = detail::interpolate_kappa_rho_V(state, params) * nu;

        Real mix = Real(0);
        if (params.kappa >= params.kappa_interp_begin && params.kappa < params.kappa_interp_end) {
            mix = (params.kappa - params.kappa_interp_begin) /
                  max_val(params.kappa_interp_end - params.kappa_interp_begin, tiny_positive<Real>());
        } else if (params.kappa >= params.kappa_interp_end) {
            mix = Real(1);
        }
        coeffs.jI = (Real(1) - mix) * kappa.jI + mix * thermal.jI;
        coeffs.jQ = (Real(1) - mix) * kappa.jQ + mix * thermal.jQ;
        coeffs.jV = (Real(1) - mix) * kappa.jV + mix * thermal.jV;
        coeffs.aI = (Real(1) - mix) * kappa.aI + mix * thermal.aI;
        coeffs.aQ = (Real(1) - mix) * kappa.aQ + mix * thermal.aQ;
        coeffs.aV = (Real(1) - mix) * kappa.aV + mix * thermal.aV;
        coeffs.rQ = kappa.rQ;
        coeffs.rV = kappa.rV;

        const Real jP = Kokkos::sqrt(coeffs.jQ * coeffs.jQ + coeffs.jV * coeffs.jV);
        if (coeffs.jI > Real(0) && jP > Real(0) && coeffs.jI < jP / max_val(params.max_pol_frac_emission, Real(1e-30))) {
            const Real scale = coeffs.jI / jP * params.max_pol_frac_emission;
            coeffs.jQ *= scale;
            coeffs.jV *= scale;
        }
        const Real aP = Kokkos::sqrt(coeffs.aQ * coeffs.aQ + coeffs.aV * coeffs.aV);
        if (coeffs.aI > Real(0) && aP > Real(0) && coeffs.aI < aP / max_val(params.max_pol_frac_absorption, Real(1e-30))) {
            const Real scale = coeffs.aI / aP * params.max_pol_frac_absorption;
            coeffs.aQ *= scale;
            coeffs.aV *= scale;
        }
        coeffs.jI *= params.emission_scale;
        coeffs.jQ *= params.emission_scale;
        coeffs.jV *= params.emission_scale;
        coeffs.aI *= params.absorption_scale;
        coeffs.aQ *= params.absorption_scale;
        coeffs.aV *= params.absorption_scale;
        coeffs.rQ *= params.faraday_scale;
        coeffs.rV *= params.faraday_scale;
        return coeffs;
    }

    return coeffs;
}

} // namespace kpolaris
