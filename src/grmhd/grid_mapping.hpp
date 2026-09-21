#pragma once

#include "common/math.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct MKSGridMapping {
    Real hslope = Real(0.3);

    KPOLARIS_INLINE Real theta_from_x2(Real, Real x2) const {
        const Real pi = Real(3.141592653589793238462643383279502884);
        return pi * x2 + ((Real(1) - hslope) / Real(2)) * Kokkos::sin(Real(2) * pi * x2);
    }

    KPOLARIS_INLINE Real dtheta_dx1(Real, Real) const { return Real(0); }

    KPOLARIS_INLINE Real dtheta_dx2(Real, Real x2) const {
        const Real pi = Real(3.141592653589793238462643383279502884);
        return pi + (Real(1) - hslope) * pi * Kokkos::cos(Real(2) * pi * x2);
    }

    KPOLARIS_INLINE Real native_x2_from_theta(Real x1, Real theta) const {
        Real lo = Real(0);
        Real hi = Real(1);
        for (int it = 0; it < 64; ++it) {
            const Real mid = Real(0.5) * (lo + hi);
            const Real th = theta_from_x2(x1, mid);
            if (th < theta) lo = mid;
            else hi = mid;
        }
        return Real(0.5) * (lo + hi);
    }
};

template<class Real = DefaultReal>
struct FMKSGridMapping {
    Real startx1 = Real(0);
    Real hslope = Real(0.3);
    Real mks_smooth = Real(0.5);
    Real poly_alpha = Real(14);
    Real poly_xt = Real(0.82);
    Real poly_norm = Real(1);

    KPOLARIS_INLINE Real power(Real x, Real p) const {
        const int n = static_cast<int>(p + Real(0.5));
        if (abs_val(p - Real(n)) < Real(1e-10) && n >= 0) {
            Real y = Real(1);
            const Real ax = abs_val(x);
            for (int i = 0; i < n; ++i) y *= ax;
            if (x < Real(0) && (n % 2) == 1) y = -y;
            return y;
        }
        return Kokkos::pow(abs_val(x), p);
    }

    KPOLARIS_INLINE Real mks_theta(Real x2) const {
        const Real pi = Real(3.141592653589793238462643383279502884);
        return pi * x2 + ((Real(1) - hslope) / Real(2)) * Kokkos::sin(Real(2) * pi * x2);
    }

    KPOLARIS_INLINE Real jet_theta(Real x2) const {
        const Real pi = Real(3.141592653589793238462643383279502884);
        const Real y = Real(2) * x2 - Real(1);
        return poly_norm * y *
            (Real(1) + power(y / poly_xt, poly_alpha) / (poly_alpha + Real(1))) + Real(0.5) * pi;
    }

    KPOLARIS_INLINE Real theta_from_x2(Real x1, Real x2) const {
        const Real thG = mks_theta(x2);
        const Real thJ = jet_theta(x2);
        return thG + Kokkos::exp(mks_smooth * (startx1 - x1)) * (thJ - thG);
    }

    KPOLARIS_INLINE Real dtheta_dx1(Real x1, Real x2) const {
        return -mks_smooth * Kokkos::exp(mks_smooth * (startx1 - x1)) * (jet_theta(x2) - mks_theta(x2));
    }

    KPOLARIS_INLINE Real dtheta_dx2(Real x1, Real x2) const {
        const Real pi = Real(3.141592653589793238462643383279502884);
        const Real y = Real(2) * x2 - Real(1);
        const Real expfac = Kokkos::exp(mks_smooth * (startx1 - x1));
        const Real dthG = pi + (Real(1) - hslope) * pi * Kokkos::cos(Real(2) * pi * x2);
        const Real dthJ = Real(2) * poly_norm *
            (Real(1) + power(y / poly_xt, poly_alpha) / (poly_alpha + Real(1))) +
            (Real(2) * poly_alpha * poly_norm * y * power(y / poly_xt, poly_alpha - Real(1))) /
                ((poly_alpha + Real(1)) * poly_xt);
        return dthG + expfac * (dthJ - dthG);
    }

    KPOLARIS_INLINE Real native_x2_from_theta(Real x1, Real theta) const {
        Real lo = Real(0);
        Real hi = Real(1);
        for (int it = 0; it < 64; ++it) {
            const Real mid = Real(0.5) * (lo + hi);
            const Real th = theta_from_x2(x1, mid);
            if (th < theta) lo = mid;
            else hi = mid;
        }
        return Real(0.5) * (lo + hi);
    }
};

template<class Real>
KPOLARIS_INLINE FMKSGridMapping<Real> make_fmks_grid_mapping(Real startx1,
                                                          Real hslope,
                                                          Real mks_smooth,
                                                          Real poly_alpha,
                                                          Real poly_xt,
                                                          Real poly_norm) {
    FMKSGridMapping<Real> mapping;
    mapping.startx1 = startx1;
    mapping.hslope = hslope;
    mapping.mks_smooth = mks_smooth;
    mapping.poly_alpha = poly_alpha;
    mapping.poly_xt = poly_xt;
    mapping.poly_norm = poly_norm;
    return mapping;
}

} // namespace kpolaris
