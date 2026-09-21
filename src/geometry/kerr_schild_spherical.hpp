#pragma once

#include "common/vec.hpp"
#include "geometry/connection.hpp"
#include "geodesic/state.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct KerrSchildSphericalMetric {
    Real mass = Real(1);
    Real spin = Real(0);
    static constexpr CoordinateSystem coordinate_system = CoordinateSystem::SphericalKS;

    KPOLARIS_INLINE KerrSchildSphericalMetric() {}
    KPOLARIS_INLINE KerrSchildSphericalMetric(Real mass_in, Real spin_in)
        : mass(mass_in), spin(spin_in) {}

    struct Work {
        Real r = Real(0), th = Real(0), sth = Real(0), cth = Real(0), rho2 = Real(0);
    };

    KPOLARIS_INLINE Work build_work(const Vec4<Real>& X) const {
        Work w;
        w.r = max_val(X[1], tiny_positive<Real>());
        w.th = X[2];
        w.sth = Kokkos::sin(w.th);
        w.cth = Kokkos::cos(w.th);
        w.rho2 = w.r * w.r + spin * spin * w.cth * w.cth;
        return w;
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

    KPOLARIS_INLINE void gcov_matrix(const Vec4<Real>& X, Real g[ndim][ndim]) const {
        const Work w = build_work(X);
        const Real s2 = w.sth * w.sth;
        const Real a2 = spin * spin;
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                g[mu][nu] = Real(0);
            }
        }
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
        for (int i = 0; i < ndim; ++i) {
            for (int j = 0; j < ndim; ++j) inv[i][j] = m[i][j + ndim];
        }
    }

    KPOLARIS_INLINE void gcon_matrix(const Vec4<Real>& X, Real g[ndim][ndim]) const {
        const Work w = build_work(X);
        const Real a = spin;
        const Real a2 = a * a;
        const Real s2 = max_val(w.sth * w.sth, tiny_positive<Real>());
        const Real f = Real(2) * mass * w.r / w.rho2;
        const Real delta = w.r * w.r - Real(2) * mass * w.r + a2;
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                g[mu][nu] = Real(0);
            }
        }
        g[0][0] = -(Real(1) + f);
        g[0][1] = g[1][0] = f;
        g[1][1] = delta / w.rho2;
        g[1][3] = g[3][1] = a / w.rho2;
        g[2][2] = Real(1) / w.rho2;
        g[3][3] = Real(1) / (w.rho2 * s2);
    }

    KPOLARIS_INLINE Real dot(const Vec4<Real>& X,
                          const Vec4<Real>& u,
                          const Vec4<Real>& v) const {
        Real g[ndim][ndim];
        gcov_matrix(X, g);
        Real out = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) out += g[mu][nu] * u[mu] * v[nu];
        }
        return out;
    }

    KPOLARIS_INLINE void dgcov(const Vec4<Real>& X,
                            Real dg[ndim][ndim][ndim]) const {
        const Work w = build_work(X);
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                for (int a = 0; a < ndim; ++a) dg[mu][nu][a] = Real(0);
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

        auto set_sym = [&](int mu, int nu, int alpha, Real val) {
            dg[mu][nu][alpha] = val;
            dg[nu][mu][alpha] = val;
        };

        set_sym(0, 0, 1, df_dr);
        set_sym(0, 0, 2, df_dth);
        set_sym(0, 1, 1, df_dr);
        set_sym(0, 1, 2, df_dth);
        set_sym(0, 3, 1, -a * s2 * df_dr);
        set_sym(0, 3, 2, -a * (s2 * df_dth + ds2_dth * f));

        set_sym(1, 1, 1, df_dr);
        set_sym(1, 1, 2, df_dth);
        set_sym(1, 3, 1, -a * s2 * df_dr);
        set_sym(1, 3, 2, -a * (ds2_dth * (Real(1) + f) + s2 * df_dth));

        set_sym(2, 2, 1, drho_dr);
        set_sym(2, 2, 2, drho_dth);

        const Real bracket = rho2 + a2 * s2 * (Real(1) + f);
        const Real dbracket_dr = drho_dr + a2 * s2 * df_dr;
        const Real dbracket_dth = drho_dth + a2 * (ds2_dth * (Real(1) + f) + s2 * df_dth);
        set_sym(3, 3, 1, s2 * dbracket_dr);
        set_sym(3, 3, 2, ds2_dth * bracket + s2 * dbracket_dth);
    }

    KPOLARIS_INLINE void connection(const Vec4<Real>& X,
                                 Real gamma[ndim][ndim][ndim]) const {
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
struct SphericalKSConnectionWork {
    Real gcon[ndim][ndim];
    Real dgr[ndim][ndim];
    Real dgth[ndim][ndim];
};

template<class Real>
KPOLARIS_INLINE SphericalKSConnectionWork<Real> build_spherical_ks_connection_work(
    const KerrSchildSphericalMetric<Real>& metric,
    const Vec4<Real>& X) {
    const typename KerrSchildSphericalMetric<Real>::Work w = metric.build_work(X);
    SphericalKSConnectionWork<Real> c;
    for (int mu = 0; mu < ndim; ++mu) {
        for (int nu = 0; nu < ndim; ++nu) {
            c.gcon[mu][nu] = Real(0);
            c.dgr[mu][nu] = Real(0);
            c.dgth[mu][nu] = Real(0);
        }
    }

    const Real r = w.r;
    const Real a = metric.spin;
    const Real a2 = a * a;
    const Real s = w.sth;
    const Real ccosth = w.cth;
    const Real s2 = s * s;
    const Real safe_s2 = max_val(s2, tiny_positive<Real>());
    const Real rho2 = w.rho2;
    const Real rho4 = rho2 * rho2;
    const Real f = Real(2) * metric.mass * r / rho2;
    const Real delta = r * r - Real(2) * metric.mass * r + a2;

    c.gcon[0][0] = -(Real(1) + f);
    c.gcon[0][1] = c.gcon[1][0] = f;
    c.gcon[1][1] = delta / rho2;
    c.gcon[1][3] = c.gcon[3][1] = a / rho2;
    c.gcon[2][2] = Real(1) / rho2;
    c.gcon[3][3] = Real(1) / (rho2 * safe_s2);

    const Real drho_dr = Real(2) * r;
    const Real drho_dth = -Real(2) * a2 * s * ccosth;
    const Real ds2_dth = Real(2) * s * ccosth;
    const Real df_dr = Real(2) * metric.mass * (rho2 - r * drho_dr) / rho4;
    const Real df_dth = -Real(2) * metric.mass * r * drho_dth / rho4;

    auto set_sym_r = [&](int mu, int nu, Real val) {
        c.dgr[mu][nu] = val;
        c.dgr[nu][mu] = val;
    };
    auto set_sym_t = [&](int mu, int nu, Real val) {
        c.dgth[mu][nu] = val;
        c.dgth[nu][mu] = val;
    };

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
    return c;
}

template<class Real>
KPOLARIS_INLINE Real derivative_row_dot(const Real d[ndim][ndim], int row,
                                     const Vec4<Real>& v) {
    Real out = Real(0);
    for (int b = 0; b < ndim; ++b) {
        out += d[row][b] * v[b];
    }
    return out;
}

template<class Real>
KPOLARIS_INLINE Real derivative_bilinear(const Real d[ndim][ndim],
                                      const Vec4<Real>& p,
                                      const Vec4<Real>& q) {
    Real out = Real(0);
    for (int a = 0; a < ndim; ++a) {
        for (int b = 0; b < ndim; ++b) {
            out += d[a][b] * p[a] * q[b];
        }
    }
    return out;
}

template<class Real>
struct SphericalKSFixedPWork {
    Real dgr_row_p[ndim];
    Real dgth_row_p[ndim];
};

template<class Real>
KPOLARIS_INLINE SphericalKSFixedPWork<Real> build_spherical_ks_fixed_p_work(
    const SphericalKSConnectionWork<Real>& c,
    const Vec4<Real>& p) {
    SphericalKSFixedPWork<Real> out;
    for (int nu = 0; nu < ndim; ++nu) {
        out.dgr_row_p[nu] = derivative_row_dot(c.dgr, nu, p);
        out.dgth_row_p[nu] = derivative_row_dot(c.dgth, nu, p);
    }
    return out;
}

template<class Real>
KPOLARIS_INLINE Vec4<Real> spherical_ks_connection_contract_fixed_p(
    const SphericalKSConnectionWork<Real>& c,
    const SphericalKSFixedPWork<Real>& fixed_p,
    const Vec4<Real>& p,
    const Vec4<Real>& q) {
    Real pD1q = Real(0);
    Real pD2q = Real(0);
    Real row_q_r[ndim];
    Real row_q_th[ndim];
    for (int nu = 0; nu < ndim; ++nu) {
        row_q_r[nu] = derivative_row_dot(c.dgr, nu, q);
        row_q_th[nu] = derivative_row_dot(c.dgth, nu, q);
        pD1q += p[nu] * row_q_r[nu];
        pD2q += p[nu] * row_q_th[nu];
    }

    Real t[ndim];
    for (int nu = 0; nu < ndim; ++nu) {
        Real term = p[1] * row_q_r[nu] + q[1] * fixed_p.dgr_row_p[nu] +
                    p[2] * row_q_th[nu] + q[2] * fixed_p.dgth_row_p[nu];
        if (nu == 1) {
            term -= pD1q;
        } else if (nu == 2) {
            term -= pD2q;
        }
        t[nu] = term;
    }

    Vec4<Real> out;
    for (int mu = 0; mu < ndim; ++mu) {
        Real comp = Real(0);
        for (int nu = 0; nu < ndim; ++nu) {
            comp += c.gcon[mu][nu] * t[nu];
        }
        out[mu] = Real(0.5) * comp;
    }
    return out;
}

template<class Real>
KPOLARIS_INLINE Vec4<Real> spherical_ks_connection_contract(
    const SphericalKSConnectionWork<Real>& c,
    const Vec4<Real>& p,
    const Vec4<Real>& q) {
    const Real pD1q = derivative_bilinear(c.dgr, p, q);
    const Real pD2q = derivative_bilinear(c.dgth, p, q);
    Real t[ndim];
    for (int nu = 0; nu < ndim; ++nu) {
        Real term = p[1] * derivative_row_dot(c.dgr, nu, q) +
                    q[1] * derivative_row_dot(c.dgr, nu, p) +
                    p[2] * derivative_row_dot(c.dgth, nu, q) +
                    q[2] * derivative_row_dot(c.dgth, nu, p);
        if (nu == 1) {
            term -= pD1q;
        } else if (nu == 2) {
            term -= pD2q;
        }
        t[nu] = term;
    }

    Vec4<Real> out;
    for (int mu = 0; mu < ndim; ++mu) {
        Real comp = Real(0);
        for (int nu = 0; nu < ndim; ++nu) {
            comp += c.gcon[mu][nu] * t[nu];
        }
        out[mu] = Real(0.5) * comp;
    }
    return out;
}

template<class Real>
KPOLARIS_INLINE void spherical_ks_connection_contract_k_triplet(
    const SphericalKSConnectionWork<Real>& c,
    const Vec4<Real>& k,
    const Vec4<Real>& e1,
    const Vec4<Real>& e2,
    Vec4<Real>& ck,
    Vec4<Real>& ce1,
    Vec4<Real>& ce2) {
    Real row_k_r[ndim], row_k_th[ndim];
    Real row_e1_r[ndim], row_e1_th[ndim];
    Real row_e2_r[ndim], row_e2_th[ndim];
    Real kD1k = Real(0), kD2k = Real(0);
    Real kD1e1 = Real(0), kD2e1 = Real(0);
    Real kD1e2 = Real(0), kD2e2 = Real(0);
    for (int nu = 0; nu < ndim; ++nu) {
        const Real dgr0 = c.dgr[nu][0];
        const Real dgr1 = c.dgr[nu][1];
        const Real dgr2 = c.dgr[nu][2];
        const Real dgr3 = c.dgr[nu][3];
        const Real dgt0 = c.dgth[nu][0];
        const Real dgt1 = c.dgth[nu][1];
        const Real dgt2 = c.dgth[nu][2];
        const Real dgt3 = c.dgth[nu][3];
        row_k_r[nu] = dgr0 * k[0] + dgr1 * k[1] + dgr2 * k[2] + dgr3 * k[3];
        row_k_th[nu] = dgt0 * k[0] + dgt1 * k[1] + dgt2 * k[2] + dgt3 * k[3];
        row_e1_r[nu] = dgr0 * e1[0] + dgr1 * e1[1] + dgr2 * e1[2] + dgr3 * e1[3];
        row_e1_th[nu] = dgt0 * e1[0] + dgt1 * e1[1] + dgt2 * e1[2] + dgt3 * e1[3];
        row_e2_r[nu] = dgr0 * e2[0] + dgr1 * e2[1] + dgr2 * e2[2] + dgr3 * e2[3];
        row_e2_th[nu] = dgt0 * e2[0] + dgt1 * e2[1] + dgt2 * e2[2] + dgt3 * e2[3];
        kD1k += k[nu] * row_k_r[nu];
        kD2k += k[nu] * row_k_th[nu];
        kD1e1 += k[nu] * row_e1_r[nu];
        kD2e1 += k[nu] * row_e1_th[nu];
        kD1e2 += k[nu] * row_e2_r[nu];
        kD2e2 += k[nu] * row_e2_th[nu];
    }

    Real tk[ndim], te1[ndim], te2[ndim];
    for (int nu = 0; nu < ndim; ++nu) {
        Real term_k = Real(2) * (k[1] * row_k_r[nu] + k[2] * row_k_th[nu]);
        Real term_e1 = k[1] * row_e1_r[nu] + e1[1] * row_k_r[nu] +
                       k[2] * row_e1_th[nu] + e1[2] * row_k_th[nu];
        Real term_e2 = k[1] * row_e2_r[nu] + e2[1] * row_k_r[nu] +
                       k[2] * row_e2_th[nu] + e2[2] * row_k_th[nu];
        if (nu == 1) {
            term_k -= kD1k;
            term_e1 -= kD1e1;
            term_e2 -= kD1e2;
        } else if (nu == 2) {
            term_k -= kD2k;
            term_e1 -= kD2e1;
            term_e2 -= kD2e2;
        }
        tk[nu] = term_k;
        te1[nu] = term_e1;
        te2[nu] = term_e2;
    }

    for (int mu = 0; mu < ndim; ++mu) {
        Real comp_k = Real(0), comp_e1 = Real(0), comp_e2 = Real(0);
        for (int nu = 0; nu < ndim; ++nu) {
            const Real g = c.gcon[mu][nu];
            comp_k += g * tk[nu];
            comp_e1 += g * te1[nu];
            comp_e2 += g * te2[nu];
        }
        ck[mu] = Real(0.5) * comp_k;
        ce1[mu] = Real(0.5) * comp_e1;
        ce2[mu] = Real(0.5) * comp_e2;
    }
}

template<class Real>
KPOLARIS_INLINE Vec4<Real> spherical_ks_connection_contract(
    const KerrSchildSphericalMetric<Real>& metric,
    const Vec4<Real>& X,
    const Vec4<Real>& p,
    const Vec4<Real>& q) {
    return spherical_ks_connection_contract(build_spherical_ks_connection_work(metric, X), p, q);
}

template<class Real>
KPOLARIS_INLINE TransportState<Real> transport_rhs(
    const KerrSchildSphericalMetric<Real>& metric,
    const TransportState<Real>& s) {
    TransportState<Real> d;
    d.x = s.k;
    const SphericalKSConnectionWork<Real> c = build_spherical_ks_connection_work(metric, s.x);
    Vec4<Real> ck, ce1, ce2;
    spherical_ks_connection_contract_k_triplet(c, s.k, s.e1, s.e2, ck, ce1, ce2);
    for (int mu = 0; mu < ndim; ++mu) {
        d.k[mu] = -ck[mu];
        d.e1[mu] = -ce1[mu];
        d.e2[mu] = -ce2[mu];
    }
    return d;
}

} // namespace kpolaris
