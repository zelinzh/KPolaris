#pragma once

#include "common/vec.hpp"
#include "geometry/connection.hpp"
#include "geometry/kerr_schild_spherical.hpp"
#include "geodesic/state.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct KerrFMKSMetric {
    Real mass = Real(1);
    Real spin = Real(0);
    Real startx1 = Real(0);
    Real hslope = Real(0.3);
    Real mks_smooth = Real(0.5);
    Real poly_alpha = Real(14);
    Real poly_xt = Real(0.82);
    Real poly_norm = Real(1);
    static constexpr CoordinateSystem coordinate_system = CoordinateSystem::FMKS;

    KPOLARIS_INLINE KerrFMKSMetric() {}
    KPOLARIS_INLINE KerrFMKSMetric(Real mass_in, Real spin_in, Real startx1_in,
                                Real hslope_in, Real mks_smooth_in,
                                Real poly_alpha_in, Real poly_xt_in,
                                Real poly_norm_in)
        : mass(mass_in), spin(spin_in), startx1(startx1_in), hslope(hslope_in),
          mks_smooth(mks_smooth_in), poly_alpha(poly_alpha_in),
          poly_xt(poly_xt_in), poly_norm(poly_norm_in) {}

    struct Work {
        Real x1 = Real(0), x2 = Real(0), r = Real(0), th = Real(0);
        Real sth = Real(0), cth = Real(0), rho2 = Real(0);
    };

    KPOLARIS_INLINE Real fmks_power(Real x, Real p) const {
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

    KPOLARIS_INLINE Real theta_from_x2(Real x1, Real x2) const {
        const Real pi = Real(3.141592653589793238462643383279502884);
        const Real thG = pi * x2 + ((Real(1) - hslope) / Real(2)) * Kokkos::sin(Real(2) * pi * x2);
        const Real y = Real(2) * x2 - Real(1);
        const Real thJ = poly_norm * y *
            (Real(1) + fmks_power(y / poly_xt, poly_alpha) / (poly_alpha + Real(1))) + Real(0.5) * pi;
        return thG + Kokkos::exp(mks_smooth * (startx1 - x1)) * (thJ - thG);
    }

    KPOLARIS_INLINE Real dtheta_dx1(Real x1, Real x2) const {
        const Real pi = Real(3.141592653589793238462643383279502884);
        const Real thG = pi * x2 + ((Real(1) - hslope) / Real(2)) * Kokkos::sin(Real(2) * pi * x2);
        const Real y = Real(2) * x2 - Real(1);
        const Real thJ = poly_norm * y *
            (Real(1) + fmks_power(y / poly_xt, poly_alpha) / (poly_alpha + Real(1))) + Real(0.5) * pi;
        return -mks_smooth * Kokkos::exp(mks_smooth * (startx1 - x1)) * (thJ - thG);
    }

    KPOLARIS_INLINE Real dtheta_dx2(Real x1, Real x2) const {
        const Real pi = Real(3.141592653589793238462643383279502884);
        const Real y = Real(2) * x2 - Real(1);
        const Real expfac = Kokkos::exp(mks_smooth * (startx1 - x1));
        const Real dthG = pi + (Real(1) - hslope) * pi * Kokkos::cos(Real(2) * pi * x2);
        const Real dthJ = Real(2) * poly_norm *
            (Real(1) + fmks_power(y / poly_xt, poly_alpha) / (poly_alpha + Real(1))) +
            (Real(2) * poly_alpha * poly_norm * y * fmks_power(y / poly_xt, poly_alpha - Real(1))) /
                ((poly_alpha + Real(1)) * poly_xt);
        return dthG + expfac * (dthJ - dthG);
    }

    KPOLARIS_INLINE Real d2theta_dx1dx1(Real x1, Real x2) const {
        return -mks_smooth * dtheta_dx1(x1, x2);
    }

    KPOLARIS_INLINE Real d2theta_dx1dx2(Real x1, Real x2) const {
        const Real pi = Real(3.141592653589793238462643383279502884);
        const Real y = Real(2) * x2 - Real(1);
        const Real expfac = Kokkos::exp(mks_smooth * (startx1 - x1));
        const Real dthG = pi + (Real(1) - hslope) * pi * Kokkos::cos(Real(2) * pi * x2);
        const Real dthJ = Real(2) * poly_norm *
            (Real(1) + fmks_power(y / poly_xt, poly_alpha) / (poly_alpha + Real(1))) +
            (Real(2) * poly_alpha * poly_norm * y * fmks_power(y / poly_xt, poly_alpha - Real(1))) /
                ((poly_alpha + Real(1)) * poly_xt);
        return -mks_smooth * expfac * (dthJ - dthG);
    }

    KPOLARIS_INLINE Real d2theta_dx2dx2(Real x1, Real x2) const {
        const Real pi = Real(3.141592653589793238462643383279502884);
        const Real y = Real(2) * x2 - Real(1);
        const Real z = y / poly_xt;
        const Real expfac = Kokkos::exp(mks_smooth * (startx1 - x1));
        const Real d2thG = -Real(2) * (Real(1) - hslope) * pi * pi * Kokkos::sin(Real(2) * pi * x2);
        const Real d2thJ = Real(4) * poly_norm * poly_alpha / (poly_alpha + Real(1)) *
            (Real(2) * fmks_power(z, poly_alpha - Real(1)) / poly_xt +
             y * (poly_alpha - Real(1)) * fmks_power(z, poly_alpha - Real(2)) / (poly_xt * poly_xt));
        return d2thG + expfac * (d2thJ - d2thG);
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

    KPOLARIS_INLINE Vec4<Real> camera_position_phi0(Real radius, Real inclination) const {
        const Real x1 = Kokkos::log(max_val(radius, tiny_positive<Real>()));
        return Vec4<Real>(Real(0), x1, native_x2_from_theta(x1, inclination), Real(0));
    }

    KPOLARIS_INLINE Work build_work(const Vec4<Real>& X) const {
        Work w;
        w.x1 = X[1];
        w.x2 = X[2];
        w.r = Kokkos::exp(w.x1);
        w.th = theta_from_x2(w.x1, w.x2);
        w.sth = Kokkos::sin(w.th);
        w.cth = Kokkos::cos(w.th);
        w.rho2 = w.r * w.r + spin * spin * w.cth * w.cth;
        return w;
    }

    KPOLARIS_INLINE void ks_gcov_matrix(const Work& w, Real g[ndim][ndim]) const {
        const Real s2 = w.sth * w.sth;
        const Real a2 = spin * spin;
        for (int mu = 0; mu < ndim; ++mu) for (int nu = 0; nu < ndim; ++nu) g[mu][nu] = Real(0);
        const Real f = Real(2) * mass * w.r / w.rho2;
        g[0][0] = Real(-1) + f;
        g[0][1] = f;
        g[0][3] = -spin * f * s2;
        g[1][0] = g[0][1];
        g[1][1] = Real(1) + f;
        g[1][3] = -spin * s2 * (Real(1) + f);
        g[2][2] = w.rho2;
        g[3][0] = g[0][3];
        g[3][1] = g[1][3];
        g[3][3] = s2 * (w.rho2 + a2 * s2 * (Real(1) + f));
    }

    KPOLARIS_INLINE void dxdX_matrix(const Work& w, Real jac[ndim][ndim]) const {
        for (int mu = 0; mu < ndim; ++mu) for (int nu = 0; nu < ndim; ++nu) jac[mu][nu] = (mu == nu) ? Real(1) : Real(0);
        jac[1][1] = w.r;
        jac[2][1] = dtheta_dx1(w.x1, w.x2);
        jac[2][2] = dtheta_dx2(w.x1, w.x2);
    }

    KPOLARIS_INLINE void gcov_matrix(const Vec4<Real>& X, Real g[ndim][ndim]) const {
        const Work w = build_work(X);
        Real gks[ndim][ndim];
        Real jac[ndim][ndim];
        ks_gcov_matrix(w, gks);
        dxdX_matrix(w, jac);
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                Real sum = Real(0);
                for (int a = 0; a < ndim; ++a) {
                    for (int b = 0; b < ndim; ++b) sum += gks[a][b] * jac[a][mu] * jac[b][nu];
                }
                g[mu][nu] = sum;
            }
        }
    }

    KPOLARIS_INLINE void invert_4x4(const Real a[ndim][ndim], Real inv[ndim][ndim]) const {
        Real m[ndim][2 * ndim];
        for (int i = 0; i < ndim; ++i) {
            for (int j = 0; j < ndim; ++j) {
                m[i][j] = a[i][j];
                m[i][j + ndim] = (i == j) ? Real(1) : Real(0);
            }
        }
        for (int col = 0; col < ndim; ++col) {
            Real pivot = m[col][col];
            if (abs_val(pivot) < tiny_positive<Real>()) {
                pivot = pivot < Real(0) ? -tiny_positive<Real>() : tiny_positive<Real>();
            }
            const Real inv_pivot = Real(1) / pivot;
            for (int j = 0; j < 2 * ndim; ++j) m[col][j] *= inv_pivot;
            for (int row = 0; row < ndim; ++row) {
                if (row == col) continue;
                const Real factor = m[row][col];
                for (int j = 0; j < 2 * ndim; ++j) m[row][j] -= factor * m[col][j];
            }
        }
        for (int i = 0; i < ndim; ++i) for (int j = 0; j < ndim; ++j) inv[i][j] = m[i][j + ndim];
    }

    KPOLARIS_INLINE Real gcov(int mu, int nu, const Vec4<Real>& X) const {
        Real g[ndim][ndim];
        gcov_matrix(X, g);
        return g[mu][nu];
    }

    KPOLARIS_INLINE Real gcon(int mu, int nu, const Vec4<Real>& X) const {
        Real g[ndim][ndim];
        gcon_matrix(X, g);
        return g[mu][nu];
    }

    KPOLARIS_INLINE void gcon_matrix(const Vec4<Real>& X, Real g[ndim][ndim]) const {
        Real cov[ndim][ndim];
        gcov_matrix(X, cov);
        invert_4x4(cov, g);
    }

    KPOLARIS_INLINE Real dot(const Vec4<Real>& X, const Vec4<Real>& u, const Vec4<Real>& v) const {
        Real g[ndim][ndim];
        gcov_matrix(X, g);
        Real out = Real(0);
        for (int mu = 0; mu < ndim; ++mu) for (int nu = 0; nu < ndim; ++nu) out += g[mu][nu] * u[mu] * v[nu];
        return out;
    }

    KPOLARIS_INLINE void dks_gcov_matrix(const Work& w, Real dgr[ndim][ndim], Real dgth[ndim][ndim]) const {
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                dgr[mu][nu] = Real(0);
                dgth[mu][nu] = Real(0);
            }
        }

        const Real r = w.r;
        const Real a = spin;
        const Real a2 = a * a;
        const Real s = w.sth;
        const Real c = w.cth;
        const Real s2 = s * s;
        const Real rho2 = w.rho2;
        const Real rho4 = rho2 * rho2;
        const Real drho_dr = Real(2) * r;
        const Real drho_dth = -Real(2) * a2 * s * c;
        const Real ds2_dth = Real(2) * s * c;
        const Real f = Real(2) * mass * r / rho2;
        const Real df_dr = Real(2) * mass * (rho2 - r * drho_dr) / rho4;
        const Real df_dth = -Real(2) * mass * r * drho_dth / rho4;

        auto set_sym_r = [&](int mu, int nu, Real val) { dgr[mu][nu] = val; dgr[nu][mu] = val; };
        auto set_sym_t = [&](int mu, int nu, Real val) { dgth[mu][nu] = val; dgth[nu][mu] = val; };

        set_sym_r(0, 0, df_dr);
        set_sym_t(0, 0, df_dth);
        set_sym_r(0, 1, df_dr);
        set_sym_t(0, 1, df_dth);
        set_sym_r(0, 3, -a * s2 * df_dr);
        set_sym_t(0, 3, -a * (s2 * df_dth + ds2_dth * f));

        set_sym_r(1, 1, df_dr);
        set_sym_t(1, 1, df_dth);
        set_sym_r(1, 3, -a * s2 * df_dr);
        set_sym_t(1, 3, -a * (ds2_dth * (Real(1) + f) + s2 * df_dth));

        set_sym_r(2, 2, drho_dr);
        set_sym_t(2, 2, drho_dth);

        const Real bracket = rho2 + a2 * s2 * (Real(1) + f);
        const Real dbracket_dr = drho_dr + a2 * s2 * df_dr;
        const Real dbracket_dth = drho_dth + a2 * (ds2_dth * (Real(1) + f) + s2 * df_dth);
        set_sym_r(3, 3, s2 * dbracket_dr);
        set_sym_t(3, 3, ds2_dth * bracket + s2 * dbracket_dth);
    }

    KPOLARIS_INLINE void dxdX_derivs(const Work& w, Real dJ[ndim][ndim][ndim]) const {
        for (int alpha = 0; alpha < ndim; ++alpha) {
            for (int mu = 0; mu < ndim; ++mu) {
                for (int nu = 0; nu < ndim; ++nu) dJ[alpha][mu][nu] = Real(0);
            }
        }
        dJ[1][1][1] = w.r;
        dJ[1][2][1] = d2theta_dx1dx1(w.x1, w.x2);
        dJ[1][2][2] = d2theta_dx1dx2(w.x1, w.x2);
        dJ[2][2][1] = d2theta_dx1dx2(w.x1, w.x2);
        dJ[2][2][2] = d2theta_dx2dx2(w.x1, w.x2);
    }

    KPOLARIS_INLINE void dgcov(const Vec4<Real>& X, Real dg[ndim][ndim][ndim]) const {
        const Work w = build_work(X);
        Real gks[ndim][ndim];
        Real dgr[ndim][ndim];
        Real dgth[ndim][ndim];
        Real jac[ndim][ndim];
        Real dJ[ndim][ndim][ndim];
        ks_gcov_matrix(w, gks);
        dks_gcov_matrix(w, dgr, dgth);
        dxdX_matrix(w, jac);
        dxdX_derivs(w, dJ);

        const Real drdX[ndim] = {Real(0), w.r, Real(0), Real(0)};
        const Real dthdX[ndim] = {Real(0), jac[2][1], jac[2][2], Real(0)};

        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                for (int alpha = 0; alpha < ndim; ++alpha) {
                    Real sum = Real(0);
                    for (int a = 0; a < ndim; ++a) {
                        for (int b = 0; b < ndim; ++b) {
                            const Real dgks = dgr[a][b] * drdX[alpha] + dgth[a][b] * dthdX[alpha];
                            sum += dgks * jac[a][mu] * jac[b][nu];
                            sum += gks[a][b] * dJ[alpha][a][mu] * jac[b][nu];
                            sum += gks[a][b] * jac[a][mu] * dJ[alpha][b][nu];
                        }
                    }
                    dg[mu][nu][alpha] = sum;
                }
            }
        }
    }

    KPOLARIS_INLINE void connection(const Vec4<Real>& X, Real gamma[ndim][ndim][ndim]) const {
        Real inv[ndim][ndim];
        Real dg[ndim][ndim][ndim];
        gcon_matrix(X, inv);
        dgcov(X, dg);
        fill_connection_from_metric_derivs(inv, dg, gamma);
    }

    KPOLARIS_INLINE Real horizon_radius() const {
        return mass + Kokkos::sqrt(max_val(mass * mass - spin * spin, Real(0)));
    }
};


template<class Real>
struct FMKSConnectionWork {
    Real r = Real(0);
    Real th = Real(0);
    Real dth1 = Real(0);
    Real dth2 = Real(1);
    Real d2th11 = Real(0);
    Real d2th12 = Real(0);
    Real d2th22 = Real(0);
    SphericalKSConnectionWork<Real> spherical;
};

template<class Real>
KPOLARIS_INLINE FMKSConnectionWork<Real> build_fmks_connection_work(
    const KerrFMKSMetric<Real>& metric,
    const Vec4<Real>& X) {
    const Real x1 = X[1];
    const Real x2 = X[2];
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real y = Real(2) * x2 - Real(1);
    const Real z = y / metric.poly_xt;
    const Real expfac = Kokkos::exp(metric.mks_smooth * (metric.startx1 - x1));
    Real z_alpha = Real(0);
    Real z_alpha_m1 = Real(0);
    Real z_alpha_m2 = Real(0);
    if (abs_val(metric.poly_alpha - Real(14)) < Real(1e-12)) {
        const Real z2 = z * z;
        const Real z4 = z2 * z2;
        const Real z8 = z4 * z4;
        z_alpha_m2 = z8 * z4;
        z_alpha_m1 = z_alpha_m2 * z;
        z_alpha = z_alpha_m2 * z2;
    } else {
        z_alpha = metric.fmks_power(z, metric.poly_alpha);
        z_alpha_m1 = metric.fmks_power(z, metric.poly_alpha - Real(1));
        z_alpha_m2 = metric.fmks_power(z, metric.poly_alpha - Real(2));
    }

    const Real thG = pi * x2 + ((Real(1) - metric.hslope) / Real(2)) *
        Kokkos::sin(Real(2) * pi * x2);
    const Real thJ = metric.poly_norm * y *
        (Real(1) + z_alpha / (metric.poly_alpha + Real(1))) + Real(0.5) * pi;
    const Real dthG = pi + (Real(1) - metric.hslope) * pi *
        Kokkos::cos(Real(2) * pi * x2);
    const Real dthJ = Real(2) * metric.poly_norm *
        (Real(1) + z_alpha / (metric.poly_alpha + Real(1))) +
        (Real(2) * metric.poly_alpha * metric.poly_norm * y * z_alpha_m1) /
            ((metric.poly_alpha + Real(1)) * metric.poly_xt);
    const Real d2thG = -Real(2) * (Real(1) - metric.hslope) * pi * pi *
        Kokkos::sin(Real(2) * pi * x2);
    const Real d2thJ = Real(4) * metric.poly_norm * metric.poly_alpha /
        (metric.poly_alpha + Real(1)) *
        (Real(2) * z_alpha_m1 / metric.poly_xt +
         y * (metric.poly_alpha - Real(1)) * z_alpha_m2 /
             (metric.poly_xt * metric.poly_xt));

    FMKSConnectionWork<Real> c;
    c.r = Kokkos::exp(x1);
    c.th = thG + expfac * (thJ - thG);
    c.dth1 = -metric.mks_smooth * expfac * (thJ - thG);
    c.dth2 = dthG + expfac * (dthJ - dthG);
    c.d2th11 = -metric.mks_smooth * c.dth1;
    c.d2th12 = -metric.mks_smooth * expfac * (dthJ - dthG);
    c.d2th22 = d2thG + expfac * (d2thJ - d2thG);
    const KerrSchildSphericalMetric<Real> spherical_metric(metric.mass, metric.spin);
    const Vec4<Real> xs(X[0], c.r, c.th, X[3]);
    c.spherical = build_spherical_ks_connection_work(spherical_metric, xs);
    return c;
}

template<class Real>
KPOLARIS_INLINE Vec4<Real> fmks_to_spherical_vector(
    const FMKSConnectionWork<Real>& c,
    const Vec4<Real>& v) {
    Vec4<Real> out;
    out[0] = v[0];
    out[1] = c.r * v[1];
    out[2] = c.dth1 * v[1] + c.dth2 * v[2];
    out[3] = v[3];
    return out;
}

template<class Real>
KPOLARIS_INLINE Vec4<Real> fmks_from_spherical_connection_contract(
    const FMKSConnectionWork<Real>& c,
    Vec4<Real> hs,
    const Vec4<Real>& p,
    const Vec4<Real>& q) {
    hs[1] += c.r * p[1] * q[1];
    hs[2] += c.d2th11 * p[1] * q[1] +
             c.d2th12 * (p[1] * q[2] + p[2] * q[1]) +
             c.d2th22 * p[2] * q[2];

    Vec4<Real> out;
    out[0] = hs[0];
    out[1] = hs[1] / c.r;
    out[2] = (hs[2] - c.dth1 * hs[1] / c.r) / c.dth2;
    out[3] = hs[3];
    return out;
}

template<class Real>
KPOLARIS_INLINE Vec4<Real> fmks_connection_contract(
    const FMKSConnectionWork<Real>& c,
    const Vec4<Real>& p,
    const Vec4<Real>& q) {
    const Vec4<Real> ps = fmks_to_spherical_vector(c, p);
    const Vec4<Real> qs = fmks_to_spherical_vector(c, q);
    Vec4<Real> hs = spherical_ks_connection_contract(c.spherical, ps, qs);
    return fmks_from_spherical_connection_contract(c, hs, p, q);
}

template<class Real>
KPOLARIS_INLINE Vec4<Real> fmks_connection_contract(
    const KerrFMKSMetric<Real>& metric,
    const Vec4<Real>& X,
    const Vec4<Real>& p,
    const Vec4<Real>& q) {
    return fmks_connection_contract(build_fmks_connection_work(metric, X), p, q);
}

template<class Real>
KPOLARIS_INLINE TransportState<Real> transport_rhs(
    const KerrFMKSMetric<Real>& metric,
    const TransportState<Real>& s) {
    TransportState<Real> d;
    d.x = s.k;
    const FMKSConnectionWork<Real> c = build_fmks_connection_work(metric, s.x);
    const Vec4<Real> ks = fmks_to_spherical_vector(c, s.k);
    const Vec4<Real> e1s = fmks_to_spherical_vector(c, s.e1);
    const Vec4<Real> e2s = fmks_to_spherical_vector(c, s.e2);
    const SphericalKSFixedPWork<Real> fixed_k =
        build_spherical_ks_fixed_p_work(c.spherical, ks);
    const Vec4<Real> ck = fmks_from_spherical_connection_contract(
        c, spherical_ks_connection_contract_fixed_p(c.spherical, fixed_k, ks, ks), s.k, s.k);
    const Vec4<Real> ce1 = fmks_from_spherical_connection_contract(
        c, spherical_ks_connection_contract_fixed_p(c.spherical, fixed_k, ks, e1s), s.k, s.e1);
    const Vec4<Real> ce2 = fmks_from_spherical_connection_contract(
        c, spherical_ks_connection_contract_fixed_p(c.spherical, fixed_k, ks, e2s), s.k, s.e2);
    for (int mu = 0; mu < ndim; ++mu) {
        d.k[mu] = -ck[mu];
        d.e1[mu] = -ce1[mu];
        d.e2[mu] = -ce2[mu];
    }
    return d;
}

} // namespace kpolaris
