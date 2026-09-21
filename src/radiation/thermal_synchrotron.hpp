#pragma once

#include "common/math.hpp"
#include "radiation/stokes.hpp"

// These constants must be macros because compile-time model selection is used
// inside #if expressions. C++ enum names are replaced with zero by the
// preprocessor and therefore cannot be compared safely there.
#define KPOLARIS_THERMAL_SYNCHROTRON_PANDYA 1
#define KPOLARIS_THERMAL_SYNCHROTRON_DEXTER 4

namespace kpolaris {

enum ThermalSynchrotronFit : int {
    ThermalSynchrotronPandya = KPOLARIS_THERMAL_SYNCHROTRON_PANDYA,
    ThermalSynchrotronDexter = KPOLARIS_THERMAL_SYNCHROTRON_DEXTER
};

template<class Real = DefaultReal>
struct ThermalSynchrotronParams {
    int fit = ThermalSynchrotronPandya;
    Real max_pol_frac = Real(0.99);
    Real min_sin_theta = Real(1e-4);
    Real emission_scale = Real(1);
    Real absorption_scale = Real(1);
    Real faraday_scale = Real(1);
};

template<class Real = DefaultReal>
struct LocalThermalSynchrotronState {
    Real nu = Real(0);
    Real ne = Real(0);
    Real thetae = Real(0);
    Real b_cgs = Real(0);
    Real theta = Real(0);
    Real sin_theta = Real(0);
    Real cos_theta = Real(1);
};

namespace detail {

template<class Calc>
KPOLARIS_INLINE Calc bessel_small_floor() {
    if constexpr (sizeof(Calc) <= sizeof(float)) {
        return Calc(1e-30f);
    } else {
        return Calc(1e-300);
    }
}

template<class Calc>
KPOLARIS_INLINE Calc bessel_large_value() {
    if constexpr (sizeof(Calc) <= sizeof(float)) {
        return Calc(1e30f);
    } else {
        return Calc(1e300);
    }
}

template<class Calc>
KPOLARIS_INLINE Calc thermal_bessel_i0_nr_calc(Calc ax) {
    if (ax < Calc(3.75)) {
        Calc y = ax / Calc(3.75);
        y *= y;
        Calc poly = Calc(0.0045813);
        poly = Calc(0.0360768) + y * poly;
        poly = Calc(0.2659732) + y * poly;
        poly = Calc(1.2067492) + y * poly;
        poly = Calc(3.0899424) + y * poly;
        poly = Calc(3.5156229) + y * poly;
        return Calc(1) + y * poly;
    }
    Calc y = Calc(3.75) / ax;
    Calc poly = Calc(0.00392377);
    poly = Calc(-0.01647633) + y * poly;
    poly = Calc(0.02635537) + y * poly;
    poly = Calc(-0.02057706) + y * poly;
    poly = Calc(0.00916281) + y * poly;
    poly = Calc(-0.00157565) + y * poly;
    poly = Calc(0.00225319) + y * poly;
    poly = Calc(0.01328592) + y * poly;
    poly = Calc(0.39894228) + y * poly;
    return Kokkos::exp(ax) / Kokkos::sqrt(ax) * poly;
}

template<class Calc>
KPOLARIS_INLINE Calc thermal_bessel_i1_nr_calc(Calc ax) {
    if (ax < Calc(3.75)) {
        Calc y = ax / Calc(3.75);
        y *= y;
        Calc poly = Calc(0.00032411);
        poly = Calc(0.00301532) + y * poly;
        poly = Calc(0.02658733) + y * poly;
        poly = Calc(0.15084934) + y * poly;
        poly = Calc(0.51498869) + y * poly;
        poly = Calc(0.87890594) + y * poly;
        poly = Calc(0.5) + y * poly;
        return ax * poly;
    }
    Calc y = Calc(3.75) / ax;
    Calc poly = Calc(-0.00420059);
    poly = Calc(0.01787654) + y * poly;
    poly = Calc(-0.02895312) + y * poly;
    poly = Calc(0.02282967) + y * poly;
    poly = Calc(-0.01031555) + y * poly;
    poly = Calc(0.00163801) + y * poly;
    poly = Calc(-0.00362018) + y * poly;
    poly = Calc(-0.03988024) + y * poly;
    poly = Calc(0.39894228) + y * poly;
    return Kokkos::exp(ax) / Kokkos::sqrt(ax) * poly;
}

template<class Calc>
KPOLARIS_INLINE Calc thermal_bessel_k0_nr_calc(Calc x) {
    if (x <= Calc(0)) {
        return bessel_large_value<Calc>();
    }
    if (x <= Calc(2)) {
        Calc y = x * x * Calc(0.25);
        Calc poly = Calc(0.00000740);
        poly = Calc(0.00010750) + y * poly;
        poly = Calc(0.00262698) + y * poly;
        poly = Calc(0.03488590) + y * poly;
        poly = Calc(0.23069756) + y * poly;
        poly = Calc(0.42278420) + y * poly;
        poly = Calc(-0.57721566) + y * poly;
        return -Kokkos::log(x * Calc(0.5)) * thermal_bessel_i0_nr_calc(x) + poly;
    }
    Calc y = Calc(2) / x;
    Calc poly = Calc(0.00053208);
    poly = Calc(-0.00251540) + y * poly;
    poly = Calc(0.00587872) + y * poly;
    poly = Calc(-0.01062446) + y * poly;
    poly = Calc(0.02189568) + y * poly;
    poly = Calc(-0.07832358) + y * poly;
    poly = Calc(1.25331414) + y * poly;
    return Kokkos::exp(-x) / Kokkos::sqrt(x) * poly;
}

template<class Calc>
KPOLARIS_INLINE Calc thermal_bessel_k1_nr_calc(Calc x) {
    if (x <= Calc(0)) {
        return bessel_large_value<Calc>();
    }
    if (x <= Calc(2)) {
        Calc y = x * x * Calc(0.25);
        Calc poly = Calc(-0.00004686);
        poly = Calc(-0.00110404) + y * poly;
        poly = Calc(-0.01919402) + y * poly;
        poly = Calc(-0.18156897) + y * poly;
        poly = Calc(-0.67278579) + y * poly;
        poly = Calc(0.15443144) + y * poly;
        poly = Calc(1) + y * poly;
        return Kokkos::log(x * Calc(0.5)) * thermal_bessel_i1_nr_calc(x) + poly / x;
    }
    Calc y = Calc(2) / x;
    Calc poly = Calc(-0.00068245);
    poly = Calc(0.00325614) + y * poly;
    poly = Calc(-0.00780353) + y * poly;
    poly = Calc(0.01504268) + y * poly;
    poly = Calc(-0.03655620) + y * poly;
    poly = Calc(0.23498619) + y * poly;
    poly = Calc(1.25331414) + y * poly;
    return Kokkos::exp(-x) / Kokkos::sqrt(x) * poly;
}

} // namespace detail

template<class Real>
KPOLARIS_INLINE Real thermal_bessel_i0_nr(Real ax) {
    using Calc = SpecialFunctionReal;
    return Real(detail::thermal_bessel_i0_nr_calc<Calc>(Calc(ax)));
}

template<class Real>
KPOLARIS_INLINE Real thermal_bessel_i1_nr(Real ax) {
    using Calc = SpecialFunctionReal;
    return Real(detail::thermal_bessel_i1_nr_calc<Calc>(Calc(ax)));
}

template<class Real>
KPOLARIS_INLINE Real thermal_bessel_k0_nr(Real x) {
    using Calc = SpecialFunctionReal;
    return Real(detail::thermal_bessel_k0_nr_calc<Calc>(Calc(x)));
}

template<class Real>
KPOLARIS_INLINE Real thermal_bessel_k1_nr(Real x) {
    using Calc = SpecialFunctionReal;
    return Real(detail::thermal_bessel_k1_nr_calc<Calc>(Calc(x)));
}

template<class Real>
KPOLARIS_INLINE void thermal_bessel_k0k2(Real x, Real& k0, Real& k2) {
    using Calc = SpecialFunctionReal;
    const Calc xc = Calc(x);
    const Calc k0c = detail::thermal_bessel_k0_nr_calc<Calc>(xc);
    const Calc k1c = detail::thermal_bessel_k1_nr_calc<Calc>(xc);
    const Calc k2c = max_val(k0c + Calc(2) * k1c /
                             max_val(xc, detail::bessel_small_floor<Calc>()),
                             detail::bessel_small_floor<Calc>());
    k0 = Real(k0c);
    k2 = Real(k2c);
}

template<class Real>
KPOLARIS_INLINE void thermal_bessel_k1k2(Real x, Real& k1, Real& k2) {
    using Calc = SpecialFunctionReal;
    const Calc xc = Calc(x);
    const Calc k0c = detail::thermal_bessel_k0_nr_calc<Calc>(xc);
    const Calc k1c = detail::thermal_bessel_k1_nr_calc<Calc>(xc);
    const Calc k2c = max_val(k0c + Calc(2) * k1c /
                             max_val(xc, detail::bessel_small_floor<Calc>()),
                             detail::bessel_small_floor<Calc>());
    k1 = Real(k1c);
    k2 = Real(k2c);
}

template<class Real>
KPOLARIS_INLINE void thermal_bessel_k0k1k2(Real x, Real& k0, Real& k1, Real& k2) {
    using Calc = SpecialFunctionReal;
    const Calc xc = Calc(x);
    const Calc k0c = detail::thermal_bessel_k0_nr_calc<Calc>(xc);
    const Calc k1c = detail::thermal_bessel_k1_nr_calc<Calc>(xc);
    const Calc k2c = max_val(k0c + Calc(2) * k1c /
                             max_val(xc, detail::bessel_small_floor<Calc>()),
                             detail::bessel_small_floor<Calc>());
    k0 = Real(k0c);
    k1 = Real(k1c);
    k2 = Real(k2c);
}

template<class Real>
KPOLARIS_INLINE Real symphony_bnu_inv(Real nu, Real thetae) {
    const Real hpl = Real(6.62607015e-27);
    const Real me = Real(9.1093837015e-28);
    const Real cl = Real(2.99792458e10);
    if (!(thetae > Real(0))) {
        return Real(0);
    }
    const Real x = hpl * nu / (me * cl * cl * thetae);
    if (x < Real(2e-3)) {
        const Real denom = x / Real(24) *
                           (Real(24) + x * (Real(12) + x * (Real(4) + x)));
        return (Real(2) * hpl / max_val(denom, tiny_positive<Real>())) /
               (cl * cl);
    }
    const Real denom = Kokkos::exp(x) - Real(1);
    return (Real(2) * hpl / max_val(denom, tiny_positive<Real>())) /
           (cl * cl);
}

template<class Real>
KPOLARIS_INLINE Real symphony_dexter_jI(Real nu, Real ne, Real thetae,
                                     Real b_cgs, Real sin_theta) {
    const Real ee = Real(4.803204712570263e-10);
    const Real me = Real(9.1093837015e-28);
    const Real cl = Real(2.99792458e10);
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real nus = Real(3) * ee * b_cgs * sin_theta /
                     (Real(4) * pi * me * cl) * thetae * thetae + Real(1);
    const Real x = max_val(nu / max_val(nus, tiny_positive<Real>()), tiny_positive<Real>());
    const Real xm13 = Real(1) / Kokkos::cbrt(x);
    const Real i_val = Real(2.5651) * (Real(1) + Real(1.92) * xm13 +
                       Real(0.9977) * xm13 * xm13) *
                       Kokkos::exp(-Real(1.8899) / xm13);
    return ne * ee * ee * nu / (Real(2) * Kokkos::sqrt(Real(3)) * cl * thetae * thetae) * i_val;
}

template<class Real>
KPOLARIS_INLINE Real symphony_dexter_jQ_fit(Real nu, Real ne, Real thetae,
                                         Real b_cgs, Real sin_theta) {
    const Real ee = Real(4.803204712570263e-10);
    const Real me = Real(9.1093837015e-28);
    const Real cl = Real(2.99792458e10);
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real nus = Real(3) * ee * b_cgs * sin_theta /
                     (Real(4) * pi * me * cl) * thetae * thetae + Real(1);
    const Real x = max_val(nu / max_val(nus, tiny_positive<Real>()), tiny_positive<Real>());
    const Real xm13 = Real(1) / Kokkos::cbrt(x);
    const Real iq = Real(2.5651) * (Real(1) + Real(0.93193) * xm13 +
                    Real(0.499873) * xm13 * xm13) *
                    Kokkos::exp(-Real(1.8899) / xm13);
    return -ne * ee * ee * nu / (Real(2) * Kokkos::sqrt(Real(3)) * cl * thetae * thetae) * iq;
}

template<class Real>
KPOLARIS_INLINE Real symphony_dexter_jV(Real nu, Real ne, Real thetae,
                                     Real b_cgs, Real theta,
                                     Real min_sin_theta) {
    const Real ee = Real(4.803204712570263e-10);
    const Real me = Real(9.1093837015e-28);
    const Real cl = Real(2.99792458e10);
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real sin_theta = max_val(Kokkos::sin(theta), min_sin_theta);
    const Real nus = Real(3) * ee * b_cgs * sin_theta /
                     (Real(4) * pi * me * cl) * thetae * thetae + Real(1);
    const Real x = max_val(nu / max_val(nus, tiny_positive<Real>()), tiny_positive<Real>());
    const Real xm13 = Real(1) / Kokkos::cbrt(x);
    const Real iv = (Real(1.81384) / x + Real(3.42319) * xm13 * xm13 +
                     Real(0.0292545) / Kokkos::sqrt(x) +
                     Real(2.03773) * xm13) * Kokkos::exp(-Real(1.8899) / xm13);
    Real tan_theta = Kokkos::tan(theta);
    if (abs_val(tan_theta) < Real(1e-10)) {
        tan_theta = tan_theta >= Real(0) ? Real(1e-10) : Real(-1e-10);
    }
    return Real(2) * ne * ee * ee * nu /
           (tan_theta * Real(3) * Kokkos::sqrt(Real(3)) * cl * thetae * thetae * thetae) * iv;
}

template<class Real>
KPOLARIS_INLINE Real symphony_pandya_nu_c(Real b_cgs) {
    const Real ee = Real(4.803204712570263e-10);
    const Real me = Real(9.1093837015e-28);
    const Real cl = Real(2.99792458e10);
    const Real pi = Real(3.141592653589793238462643383279502884);
    return ee * b_cgs / (Real(2) * pi * me * cl);
}

template<class Real>
KPOLARIS_INLINE Real symphony_pandya_x(Real nu, Real thetae,
                                    Real b_cgs, Real sin_theta) {
    const Real nu_c = symphony_pandya_nu_c(b_cgs);
    const Real nu_s = Real(2) / Real(9) * nu_c * sin_theta * thetae * thetae;
    return max_val(nu / max_val(nu_s, tiny_positive<Real>()), tiny_positive<Real>());
}

template<class Real>
KPOLARIS_INLINE Real symphony_pandya_jI(Real nu, Real ne, Real thetae,
                                     Real b_cgs, Real sin_theta) {
    if (!(thetae > Real(0)) || !(ne > Real(0)) || !(b_cgs > Real(0)) ||
        !(sin_theta > Real(0))) {
        return Real(0);
    }
    const Real ee = Real(4.803204712570263e-10);
    const Real cl = Real(2.99792458e10);
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real nu_c = symphony_pandya_nu_c(b_cgs);
    const Real x = symphony_pandya_x(nu, thetae, b_cgs, sin_theta);
    const Real prefactor = ne * ee * ee * nu_c / cl;
    const Real term1 = Kokkos::sqrt(Real(2)) * pi / Real(27) * sin_theta;
    const Real x13 = Kokkos::cbrt(x);
    const Real shape = Kokkos::sqrt(x) + Real(1.8877486253633869933) * Kokkos::sqrt(x13);
    const Real term2 = shape * shape;
    return prefactor * term1 * term2 * Kokkos::exp(-x13);
}

template<class Real>
KPOLARIS_INLINE Real symphony_pandya_jQ_ipole(Real nu, Real ne, Real thetae,
                                           Real b_cgs, Real sin_theta) {
    if (!(thetae > Real(0)) || !(ne > Real(0)) || !(b_cgs > Real(0)) ||
        !(sin_theta > Real(0))) {
        return Real(0);
    }
    const Real ee = Real(4.803204712570263e-10);
    const Real cl = Real(2.99792458e10);
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real nu_c = symphony_pandya_nu_c(b_cgs);
    const Real x = symphony_pandya_x(nu, thetae, b_cgs, sin_theta);
    const Real thetae_2425 = Kokkos::pow(thetae, Real(24) / Real(25));
    const Real prefactor = ne * ee * ee * nu_c / cl;
    const Real term1 = Kokkos::sqrt(Real(2)) * pi / Real(27) * sin_theta;
    const Real term2 = (Real(7) * thetae_2425 + Real(35)) /
                       (Real(10) * thetae_2425 + Real(75));
    const Real x13 = Kokkos::cbrt(x);
    const Real shape = Kokkos::sqrt(x) + term2 * Real(1.8877486253633869933) *
                       Kokkos::sqrt(x13);
    const Real term3 = shape * shape;
    return prefactor * term1 * term3 * Kokkos::exp(-x13);
}

template<class Real>
KPOLARIS_INLINE Real symphony_pandya_jV(Real nu, Real ne, Real thetae,
                                     Real b_cgs, Real sin_theta,
                                     Real cos_theta) {
    if (!(thetae > Real(0)) || !(ne > Real(0)) || !(b_cgs > Real(0)) ||
        !(sin_theta > Real(0))) {
        return Real(0);
    }
    const Real ee = Real(4.803204712570263e-10);
    const Real cl = Real(2.99792458e10);
    const Real nu_c = symphony_pandya_nu_c(b_cgs);
    const Real x = symphony_pandya_x(nu, thetae, b_cgs, sin_theta);
    const Real prefactor = ne * ee * ee * nu_c / cl;
    const Real sin_shift = sin_theta * Real(0.4356824462767121) -
                           cos_theta * Real(0.9001004421765051);
    const Real term1 = (Real(37) - Real(87) * sin_shift) /
                       (Real(100) * (thetae + Real(1)));
    const Real term2 = Kokkos::pow(Real(1) +
                                   (Kokkos::pow(thetae, Real(3) / Real(5)) / Real(25) +
                                    Real(7) / Real(10)) *
                                   Kokkos::pow(x, Real(9) / Real(25)),
                                   Real(5) / Real(3));
    return prefactor * term1 * term2 * Kokkos::exp(-Kokkos::cbrt(x));
}

template<class Real>
KPOLARIS_INLINE Real symphony_rho_Q(Real nu, Real ne, Real thetae,
                                 Real b_cgs, Real theta,
                                 Real min_sin_theta) {
    const Real ee = Real(4.803204712570263e-10);
    const Real me = Real(9.1093837015e-28);
    const Real cl = Real(2.99792458e10);
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real sin_theta = max_val(Kokkos::sin(theta), min_sin_theta);
    const Real omega0 = ee * b_cgs / (me * cl);
    const Real wp2 = Real(4) * pi * ne * ee * ee / me;
    const Real two_pi_nu = Real(2) * pi * nu;
    const Real x = max_val(thetae * Kokkos::sqrt(max_val(Real(0), Kokkos::sqrt(Real(2)) * sin_theta *
                   (Real(1e3) * omega0 / max_val(two_pi_nu, tiny_positive<Real>())))), Real(1e-12));
    const Real exp_x_47 = Kokkos::exp(-x / Real(47.2));
    const Real extraterm = (Real(0.011) * exp_x_47 -
                           Real(369.69342869090815) *
                           Kokkos::pow(x + Real(1e-16), Real(-8) / Real(3))) *
                          (Real(0.5) + Real(0.5) *
                           Kokkos::tanh((Kokkos::log(x) - Real(4.787491742782046)) / Real(0.1)));
    const Real jffunc = Real(2.011) * Kokkos::exp(-Kokkos::pow(x, Real(1.035)) / Real(4.7)) -
                          Kokkos::cos(x / Real(2)) * Kokkos::exp(-Kokkos::pow(x, Real(1.2)) / Real(2.73)) -
                          Real(0.011) * exp_x_47 + extraterm;
    Real k1 = Real(0), k2 = Real(0);
    thermal_bessel_k1k2(Real(1) / thetae, k1, k2);
    const Real k_ratio = k2 > Real(0) ? k1 / k2 : Real(1);
    const Real safe_two_pi_nu = max_val(two_pi_nu, tiny_positive<Real>());
    Real eps11m22 = jffunc * wp2 * omega0 * omega0;
    eps11m22 /= safe_two_pi_nu;
    eps11m22 /= safe_two_pi_nu;
    eps11m22 /= safe_two_pi_nu;
    eps11m22 /= safe_two_pi_nu;
    eps11m22 *= (k_ratio + Real(6) * thetae) * sin_theta * sin_theta;
    return pi * nu / cl * eps11m22;
}

template<class Real>
KPOLARIS_INLINE Real symphony_rho_V(Real nu, Real ne, Real thetae,
                                 Real b_cgs, Real theta,
                                 Real min_sin_theta) {
    const Real ee = Real(4.803204712570263e-10);
    const Real me = Real(9.1093837015e-28);
    const Real cl = Real(2.99792458e10);
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real sin_theta = max_val(Kokkos::sin(theta), min_sin_theta);
    const Real omega0 = ee * b_cgs / (me * cl);
    const Real wp2 = Real(4) * pi * ne * ee * ee / me;
    const Real two_pi_nu = Real(2) * pi * nu;
    const Real x = max_val(thetae * Kokkos::sqrt(max_val(Real(0), Kokkos::sqrt(Real(2)) * sin_theta *
                   (Real(1e3) * omega0 / max_val(two_pi_nu, tiny_positive<Real>())))), Real(1e-12));
    Real k0 = Real(0), k2 = Real(0);
    thermal_bessel_k0k2(Real(1) / thetae, k0, k2);
    const Real shgmfunc = Real(1) - Real(0.11) * Kokkos::log(Real(1) + Real(0.035) * x);
    const Real k_ratio = k2 > Real(0) ? k0 / k2 : Real(1);
    const Real fit_factor = k_ratio * shgmfunc;
    const Real safe_two_pi_nu = max_val(two_pi_nu, tiny_positive<Real>());
    Real eps12 = wp2 * omega0;
    eps12 /= safe_two_pi_nu;
    eps12 /= safe_two_pi_nu;
    eps12 /= safe_two_pi_nu;
    eps12 *= fit_factor * Kokkos::cos(theta);
    return Real(2) * pi * nu / cl * eps12;
}

template<class Real>
KPOLARIS_INLINE void symphony_rho_QV(Real nu, Real ne, Real thetae,
                                  Real b_cgs, Real sin_theta,
                                  Real cos_theta,
                                  Real& rhoQ, Real& rhoV) {
    const Real ee = Real(4.803204712570263e-10);
    const Real me = Real(9.1093837015e-28);
    const Real cl = Real(2.99792458e10);
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real omega0 = ee * b_cgs / (me * cl);
    const Real wp2 = Real(4) * pi * ne * ee * ee / me;
    const Real two_pi_nu = Real(2) * pi * nu;
    const Real x = max_val(thetae * Kokkos::sqrt(max_val(Real(0), Kokkos::sqrt(Real(2)) * sin_theta *
                   (Real(1e3) * omega0 / max_val(two_pi_nu, tiny_positive<Real>())))), Real(1e-12));
    Real k0 = Real(0), k1 = Real(0), k2 = Real(0);
    thermal_bessel_k0k1k2(Real(1) / thetae, k0, k1, k2);

    const Real exp_x_47 = Kokkos::exp(-x / Real(47.2));
    const Real extraterm = (Real(0.011) * exp_x_47 -
                           Real(369.69342869090815) *
                           Kokkos::pow(x + Real(1e-16), Real(-8) / Real(3))) *
                          (Real(0.5) + Real(0.5) *
                           Kokkos::tanh((Kokkos::log(x) - Real(4.787491742782046)) / Real(0.1)));
    const Real jffunc = Real(2.011) * Kokkos::exp(-Kokkos::pow(x, Real(1.035)) / Real(4.7)) -
                          Kokkos::cos(x / Real(2)) * Kokkos::exp(-Kokkos::pow(x, Real(1.2)) / Real(2.73)) -
                          Real(0.011) * exp_x_47 + extraterm;
    const Real k1_ratio = k2 > Real(0) ? k1 / k2 : Real(1);
    const Real safe_two_pi_nu = max_val(two_pi_nu, tiny_positive<Real>());
    Real eps11m22 = jffunc * wp2 * omega0 * omega0;
    eps11m22 /= safe_two_pi_nu;
    eps11m22 /= safe_two_pi_nu;
    eps11m22 /= safe_two_pi_nu;
    eps11m22 /= safe_two_pi_nu;
    eps11m22 *= (k1_ratio + Real(6) * thetae) * sin_theta * sin_theta;
    rhoQ = pi * nu / cl * eps11m22;

    const Real shgmfunc = Real(1) - Real(0.11) * Kokkos::log(Real(1) + Real(0.035) * x);
    const Real k0_ratio = k2 > Real(0) ? k0 / k2 : Real(1);
    const Real fit_factor = k0_ratio * shgmfunc;
    Real eps12 = wp2 * omega0;
    eps12 /= safe_two_pi_nu;
    eps12 /= safe_two_pi_nu;
    eps12 /= safe_two_pi_nu;
    eps12 *= fit_factor * cos_theta;
    rhoV = Real(2) * pi * nu / cl * eps12;
}

// Returns invariant transfer coefficients with e2 along projected B and
// (e1,e2,k_spatial) right handed. Positive optically thin synchrotron jQ
// therefore gives electric polarization perpendicular to projected B.
// Rotate jQ, aQ and rho_Q together into the transported camera/screen basis.
template<int Fit, class Real>
KPOLARIS_INLINE TransferCoeffs<Real>
thermal_synchrotron_magnetic_basis_coefficients_fit(
    const LocalThermalSynchrotronState<Real>& state,
    const ThermalSynchrotronParams<Real>& params) {
    TransferCoeffs<Real> coeffs;
    if (!(state.nu > Real(0)) || !(state.ne > Real(0)) ||
        !(state.thetae > Real(0)) || !(state.b_cgs > Real(0))) {
        return coeffs;
    }
    const Real nu = max_val(state.nu, Real(1));
    const Real sin_theta = max_val(state.sin_theta, params.min_sin_theta);
    [[maybe_unused]] const Real theta = state.theta;

    Real jI_phys = Real(0);
    Real jQ_phys_ipole = Real(0);
    Real jV_phys = Real(0);
    if constexpr (Fit == ThermalSynchrotronDexter) {
        const Real ee = Real(4.803204712570263e-10);
        const Real me = Real(9.1093837015e-28);
        const Real cl = Real(2.99792458e10);
        const Real pi = Real(3.141592653589793238462643383279502884);
        const Real nus = Real(3) * ee * state.b_cgs * sin_theta /
                         (Real(4) * pi * me * cl) * state.thetae * state.thetae + Real(1);
        const Real x = max_val(nu / max_val(nus, tiny_positive<Real>()), tiny_positive<Real>());
        const Real xm13 = Real(1) / Kokkos::cbrt(x);
        const Real exp_shape = Kokkos::exp(-Real(1.8899) / xm13);
        const Real pref = state.ne * ee * ee * nu /
            (Real(2) * Kokkos::sqrt(Real(3)) * cl * state.thetae * state.thetae);
        const Real i_val = Real(2.5651) * (Real(1) + Real(1.92) * xm13 +
                           Real(0.9977) * xm13 * xm13) * exp_shape;
        const Real iq = Real(2.5651) * (Real(1) + Real(0.93193) * xm13 +
                        Real(0.499873) * xm13 * xm13) * exp_shape;
        jI_phys = pref * i_val;
        jQ_phys_ipole = pref * iq;

        const Real sin_theta_v = sin_theta;
        const Real nus_v = Real(3) * ee * state.b_cgs * sin_theta_v /
                           (Real(4) * pi * me * cl) * state.thetae * state.thetae + Real(1);
        const Real xv = max_val(nu / max_val(nus_v, tiny_positive<Real>()), tiny_positive<Real>());
        const Real xv_m13 = Real(1) / Kokkos::cbrt(xv);
        const Real iv = (Real(1.81384) / xv + Real(3.42319) * xv_m13 * xv_m13 +
                         Real(0.0292545) / Kokkos::sqrt(xv) +
                         Real(2.03773) * xv_m13) * Kokkos::exp(-Real(1.8899) / xv_m13);
        Real cos_for_tan = state.cos_theta;
        if (abs_val(cos_for_tan) < Real(1e-10)) {
            cos_for_tan = cos_for_tan >= Real(0) ? Real(1e-10) : Real(-1e-10);
        }
        const Real tan_theta = sin_theta_v / cos_for_tan;
        jV_phys = Real(2) * state.ne * ee * ee * nu /
            (tan_theta * Real(3) * Kokkos::sqrt(Real(3)) * cl *
             state.thetae * state.thetae * state.thetae) * iv;
    } else {
        static_assert(Fit == ThermalSynchrotronPandya,
                      "unsupported thermal synchrotron fit");
        jI_phys = symphony_pandya_jI(nu, state.ne, state.thetae, state.b_cgs, sin_theta);
        jQ_phys_ipole = symphony_pandya_jQ_ipole(nu, state.ne, state.thetae, state.b_cgs, sin_theta);
        jV_phys = symphony_pandya_jV(nu, state.ne, state.thetae, state.b_cgs, sin_theta,
                                    state.cos_theta);
    }

    const Real nusq = nu * nu;
    Real jI_inv = jI_phys / max_val(nusq, tiny_positive<Real>());
    Real jQ_inv = jQ_phys_ipole / max_val(nusq, tiny_positive<Real>());
    Real jV_inv = jV_phys / max_val(nusq, tiny_positive<Real>());

    const Real jP_scale = max_val(abs_val(jQ_inv), abs_val(jV_inv));
    const Real jP = jP_scale > Real(0)
        ? jP_scale * Kokkos::sqrt(square(jQ_inv / jP_scale) +
                                  square(jV_inv / jP_scale))
        : Real(0);
    if (jI_inv > Real(0) && jP > Real(0) &&
        jI_inv < jP / max_val(params.max_pol_frac, Real(1e-30))) {
        const Real scale = jI_inv / jP * params.max_pol_frac;
        jQ_inv *= scale;
        jV_inv *= scale;
    }

    Real aI_inv = Real(0);
    Real aQ_inv = Real(0);
    Real aV_inv = Real(0);
    const Real bnu_inv = symphony_bnu_inv(nu, state.thetae);
    if (bnu_inv > Real(0)) {
        aI_inv = jI_inv / bnu_inv;
        aQ_inv = jQ_inv / bnu_inv;
        aV_inv = jV_inv / bnu_inv;
    }

    Real rhoQ_phys = Real(0);
    Real rhoV_phys = Real(0);
    symphony_rho_QV(nu, state.ne, state.thetae, state.b_cgs, sin_theta,
                    state.cos_theta, rhoQ_phys, rhoV_phys);
    const Real rhoQ_inv = rhoQ_phys * nu;
    const Real rhoV_inv = rhoV_phys * nu;

    coeffs.jI = params.emission_scale * jI_inv;
    coeffs.jQ = params.emission_scale * jQ_inv;
    coeffs.jV = params.emission_scale * jV_inv;
    coeffs.aI = params.absorption_scale * aI_inv;
    coeffs.aQ = params.absorption_scale * aQ_inv;
    coeffs.aV = params.absorption_scale * aV_inv;
    coeffs.rQ = params.faraday_scale * rhoQ_inv;
    coeffs.rV = params.faraday_scale * rhoV_inv;
    return coeffs;
}

template<class Real>
KPOLARIS_INLINE TransferCoeffs<Real>
thermal_synchrotron_magnetic_basis_coefficients(
    const LocalThermalSynchrotronState<Real>& state,
    const ThermalSynchrotronParams<Real>& params) {
    if (params.fit == ThermalSynchrotronDexter) {
        return thermal_synchrotron_magnetic_basis_coefficients_fit<ThermalSynchrotronDexter>(state, params);
    }
    return thermal_synchrotron_magnetic_basis_coefficients_fit<ThermalSynchrotronPandya>(state, params);
}

} // namespace kpolaris
