#pragma once

#include "common/vec.hpp"
#include "geometry/connection.hpp"
#include "geodesic/state.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct KerrSchildInMetric {
    Real mass = Real(1);
    Real spin = Real(0);
    static constexpr CoordinateSystem coordinate_system = CoordinateSystem::CartesianKS;

    KPOLARIS_INLINE KerrSchildInMetric() {}
    KPOLARIS_INLINE KerrSchildInMetric(Real mass_in, Real spin_in)
        : mass(mass_in), spin(spin_in) {}

    struct Work {
        Real x = Real(0), y = Real(0), z = Real(0);
        Real a = Real(0), a2 = Real(0);
        Real r = Real(0), r2 = Real(0);
        Real sqsum = Real(0), discr = Real(0), denom = Real(0);
        Real amp = Real(0), amp_r = Real(0);
        Real l[ndim];
    };

    KPOLARIS_INLINE Work build_work(const Vec4<Real>& X) const {
        Work w;
        w.x = X[1];
        w.y = X[2];
        w.z = X[3];
        w.a = spin;
        w.a2 = spin * spin;
        const Real radius2 = w.x * w.x + w.y * w.y + w.z * w.z;
        w.sqsum = radius2 - w.a2;
        w.discr = Kokkos::sqrt(w.sqsum * w.sqsum + Real(4) * w.a2 * w.z * w.z);
        w.r2 = Real(0.5) * (w.sqsum + w.discr);
        w.r = Kokkos::sqrt(max_val(w.r2, tiny_positive<Real>()));
        w.denom = w.r2 * w.r2 + w.a2 * w.z * w.z;
        w.amp = Real(2) * mass * w.r2 * w.r / w.denom;
        w.amp_r = -Real(2) * mass * w.r2 *
                  (w.r2 * w.r2 - Real(3) * w.a2 * w.z * w.z) /
                  (w.denom * w.denom);

        const Real r2pa2 = w.r2 + w.a2;
        w.l[0] = Real(1);
        w.l[1] = (w.r * w.x + w.a * w.y) / r2pa2;
        w.l[2] = (w.r * w.y - w.a * w.x) / r2pa2;
        w.l[3] = w.z / w.r;
        return w;
    }

    KPOLARIS_INLINE Real eta(int mu, int nu) const {
        if (mu != nu) {
            return Real(0);
        }
        return mu == 0 ? Real(-1) : Real(1);
    }

    KPOLARIS_INLINE Real gcov(int mu, int nu, const Vec4<Real>& X) const {
        const Work w = build_work(X);
        return eta(mu, nu) + w.amp * w.l[mu] * w.l[nu];
    }

    KPOLARIS_INLINE Real gcon(int mu, int nu, const Vec4<Real>& X) const {
        const Work w = build_work(X);
        const Real lcon_mu = (mu == 0 ? -w.l[0] : w.l[mu]);
        const Real lcon_nu = (nu == 0 ? -w.l[0] : w.l[nu]);
        return eta(mu, nu) - w.amp * lcon_mu * lcon_nu;
    }

    KPOLARIS_INLINE void gcov_matrix(const Vec4<Real>& X, Real g[ndim][ndim]) const {
        const Work w = build_work(X);
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                g[mu][nu] = eta(mu, nu) + w.amp * w.l[mu] * w.l[nu];
            }
        }
    }

    KPOLARIS_INLINE void gcon_matrix(const Vec4<Real>& X, Real g[ndim][ndim]) const {
        const Work w = build_work(X);
        for (int mu = 0; mu < ndim; ++mu) {
            const Real lcon_mu = (mu == 0 ? -w.l[0] : w.l[mu]);
            for (int nu = 0; nu < ndim; ++nu) {
                const Real lcon_nu = (nu == 0 ? -w.l[0] : w.l[nu]);
                g[mu][nu] = eta(mu, nu) - w.amp * lcon_mu * lcon_nu;
            }
        }
    }

    KPOLARIS_INLINE Real dot(const Vec4<Real>& X,
                          const Vec4<Real>& u,
                          const Vec4<Real>& v) const {
        const Work w = build_work(X);
        Real eta_dot = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            eta_dot += eta(mu, mu) * u[mu] * v[mu];
        }
        Real lu = Real(0);
        Real lv = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            lu += w.l[mu] * u[mu];
            lv += w.l[mu] * v[mu];
        }
        return eta_dot + w.amp * lu * lv;
    }

    KPOLARIS_INLINE void dgcov(const Vec4<Real>& X,
                            Real dg[ndim][ndim][ndim]) const {
        const Work w = build_work(X);
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                for (int a = 0; a < ndim; ++a) {
                    dg[mu][nu][a] = Real(0);
                }
            }
        }

        const Real r2pa2 = w.r2 + w.a2;
        const Real r2pa2_2 = r2pa2 * r2pa2;
        const Real inv_2r = Real(0.5) / w.r;

        Real dr[ndim] = {Real(0), Real(0), Real(0), Real(0)};
        dr[1] = (w.x + w.sqsum * w.x / w.discr) * inv_2r;
        dr[2] = (w.y + w.sqsum * w.y / w.discr) * inv_2r;
        dr[3] = (w.z + (w.sqsum * w.z + Real(2) * w.a2 * w.z) / w.discr) * inv_2r;

        Real damp[ndim] = {Real(0), Real(0), Real(0), Real(0)};
        damp[1] = w.amp_r * dr[1];
        damp[2] = w.amp_r * dr[2];
        damp[3] = Real(6) * mass * w.r2 * dr[3] / w.denom -
                  w.amp / w.denom * (Real(4) * w.r * w.r2 * dr[3] +
                                      Real(2) * w.a2 * w.z);

        Real dl[ndim][ndim];
        for (int mu = 0; mu < ndim; ++mu) {
            for (int a = 0; a < ndim; ++a) {
                dl[mu][a] = Real(0);
            }
        }

        dl[1][1] = ((w.r + w.x * dr[1]) * r2pa2 -
                    Real(2) * w.r * (w.r * w.x + w.a * w.y) * dr[1]) /
                   r2pa2_2;
        dl[2][1] = ((-w.a + w.y * dr[1]) * r2pa2 -
                    Real(2) * w.r * (w.r * w.y - w.a * w.x) * dr[1]) /
                   r2pa2_2;
        dl[3][1] = -w.z / w.r2 * dr[1];

        dl[1][2] = ((w.a + w.x * dr[2]) * r2pa2 -
                    Real(2) * w.r * (w.r * w.x + w.a * w.y) * dr[2]) /
                   r2pa2_2;
        dl[2][2] = ((w.r + w.y * dr[2]) * r2pa2 -
                    Real(2) * w.r * (w.r * w.y - w.a * w.x) * dr[2]) /
                   r2pa2_2;
        dl[3][2] = -w.z / w.r2 * dr[2];

        dl[1][3] = (w.x * dr[3] * r2pa2 -
                    Real(2) * w.r * (w.r * w.x + w.a * w.y) * dr[3]) /
                   r2pa2_2;
        dl[2][3] = (w.y * dr[3] * r2pa2 -
                    Real(2) * w.r * (w.r * w.y - w.a * w.x) * dr[3]) /
                   r2pa2_2;
        dl[3][3] = (w.r - w.z * dr[3]) / w.r2;

        for (int alpha = 1; alpha < ndim; ++alpha) {
            for (int mu = 0; mu < ndim; ++mu) {
                for (int nu = 0; nu < ndim; ++nu) {
                    dg[mu][nu][alpha] = damp[alpha] * w.l[mu] * w.l[nu] +
                                        w.amp * (dl[mu][alpha] * w.l[nu] +
                                                 w.l[mu] * dl[nu][alpha]);
                }
            }
        }
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
struct KSConnectionWork {
    Real amp = Real(0);
    Real damp[ndim];
    Real l[ndim];
    Real lcon[ndim];
    Real dl[ndim][ndim];
};

template<class Real>
KPOLARIS_INLINE KSConnectionWork<Real> build_ks_connection_work(
    const KerrSchildInMetric<Real>& metric,
    const Vec4<Real>& X) {
    const typename KerrSchildInMetric<Real>::Work w = metric.build_work(X);
    KSConnectionWork<Real> c;
    c.amp = w.amp;
    for (int mu = 0; mu < ndim; ++mu) {
        c.l[mu] = w.l[mu];
        c.lcon[mu] = (mu == 0 ? -w.l[0] : w.l[mu]);
        c.damp[mu] = Real(0);
        for (int alpha = 0; alpha < ndim; ++alpha) {
            c.dl[mu][alpha] = Real(0);
        }
    }

    const Real r2pa2 = w.r2 + w.a2;
    const Real r2pa2_2 = r2pa2 * r2pa2;
    const Real inv_2r = Real(0.5) / w.r;

    Real dr[ndim] = {Real(0), Real(0), Real(0), Real(0)};
    dr[1] = (w.x + w.sqsum * w.x / w.discr) * inv_2r;
    dr[2] = (w.y + w.sqsum * w.y / w.discr) * inv_2r;
    dr[3] = (w.z + (w.sqsum * w.z + Real(2) * w.a2 * w.z) / w.discr) * inv_2r;

    c.damp[1] = w.amp_r * dr[1];
    c.damp[2] = w.amp_r * dr[2];
    c.damp[3] = Real(6) * metric.mass * w.r2 * dr[3] / w.denom -
                w.amp / w.denom * (Real(4) * w.r * w.r2 * dr[3] +
                                    Real(2) * w.a2 * w.z);

    c.dl[1][1] = ((w.r + w.x * dr[1]) * r2pa2 -
                  Real(2) * w.r * (w.r * w.x + w.a * w.y) * dr[1]) /
                 r2pa2_2;
    c.dl[2][1] = ((-w.a + w.y * dr[1]) * r2pa2 -
                  Real(2) * w.r * (w.r * w.y - w.a * w.x) * dr[1]) /
                 r2pa2_2;
    c.dl[3][1] = -w.z / w.r2 * dr[1];

    c.dl[1][2] = ((w.a + w.x * dr[2]) * r2pa2 -
                  Real(2) * w.r * (w.r * w.x + w.a * w.y) * dr[2]) /
                 r2pa2_2;
    c.dl[2][2] = ((w.r + w.y * dr[2]) * r2pa2 -
                  Real(2) * w.r * (w.r * w.y - w.a * w.x) * dr[2]) /
                 r2pa2_2;
    c.dl[3][2] = -w.z / w.r2 * dr[2];

    c.dl[1][3] = (w.x * dr[3] * r2pa2 -
                  Real(2) * w.r * (w.r * w.x + w.a * w.y) * dr[3]) /
                 r2pa2_2;
    c.dl[2][3] = (w.y * dr[3] * r2pa2 -
                  Real(2) * w.r * (w.r * w.y - w.a * w.x) * dr[3]) /
                 r2pa2_2;
    c.dl[3][3] = (w.r - w.z * dr[3]) / w.r2;
    return c;
}

template<class Real>
KPOLARIS_INLINE Real ks_l_dot(const KSConnectionWork<Real>& c,
                           const Vec4<Real>& v) {
    Real out = Real(0);
    for (int mu = 0; mu < ndim; ++mu) {
        out += c.l[mu] * v[mu];
    }
    return out;
}

template<class Real>
KPOLARIS_INLINE Real ks_dl_dot(const KSConnectionWork<Real>& c,
                            int alpha,
                            const Vec4<Real>& v) {
    Real out = Real(0);
    for (int mu = 0; mu < ndim; ++mu) {
        out += c.dl[mu][alpha] * v[mu];
    }
    return out;
}

template<class Real>
KPOLARIS_INLINE Real ks_derivative_row_dot(const KSConnectionWork<Real>& c,
                                        int alpha,
                                        int row,
                                        const Vec4<Real>& v) {
    if (alpha == 0) {
        return Real(0);
    }
    const Real lv = ks_l_dot(c, v);
    const Real dlv = ks_dl_dot(c, alpha, v);
    return c.damp[alpha] * c.l[row] * lv +
           c.amp * (c.dl[row][alpha] * lv + c.l[row] * dlv);
}

template<class Real>
KPOLARIS_INLINE Real ks_derivative_bilinear(const KSConnectionWork<Real>& c,
                                         int alpha,
                                         const Vec4<Real>& p,
                                         const Vec4<Real>& q) {
    if (alpha == 0) {
        return Real(0);
    }
    const Real lp = ks_l_dot(c, p);
    const Real lq = ks_l_dot(c, q);
    const Real dlp = ks_dl_dot(c, alpha, p);
    const Real dlq = ks_dl_dot(c, alpha, q);
    return c.damp[alpha] * lp * lq + c.amp * (dlp * lq + lp * dlq);
}

template<class Real>
KPOLARIS_INLINE Vec4<Real> ks_connection_contract(const KSConnectionWork<Real>& c,
                                               const Vec4<Real>& p,
                                               const Vec4<Real>& q) {
    Real t[ndim];
    for (int nu = 0; nu < ndim; ++nu) {
        Real term = Real(0);
        for (int alpha = 1; alpha < ndim; ++alpha) {
            term += p[alpha] * ks_derivative_row_dot(c, alpha, nu, q) +
                    q[alpha] * ks_derivative_row_dot(c, alpha, nu, p);
        }
        term -= ks_derivative_bilinear(c, nu, p, q);
        t[nu] = term;
    }

    Real lt = Real(0);
    for (int nu = 0; nu < ndim; ++nu) {
        lt += c.lcon[nu] * t[nu];
    }

    Vec4<Real> out;
    for (int mu = 0; mu < ndim; ++mu) {
        const Real eta_term = (mu == 0 ? -t[mu] : t[mu]);
        out[mu] = Real(0.5) * (eta_term - c.amp * c.lcon[mu] * lt);
    }
    return out;
}

template<class Real>
KPOLARIS_INLINE Vec4<Real> ks_connection_contract(
    const KerrSchildInMetric<Real>& metric,
    const Vec4<Real>& X,
    const Vec4<Real>& p,
    const Vec4<Real>& q) {
    return ks_connection_contract(build_ks_connection_work(metric, X), p, q);
}

template<class Real>
KPOLARIS_INLINE TransportState<Real> transport_rhs(
    const KerrSchildInMetric<Real>& metric,
    const TransportState<Real>& s) {
    TransportState<Real> d;
    d.x = s.k;
    const KSConnectionWork<Real> c = build_ks_connection_work(metric, s.x);
    const Vec4<Real> ck = ks_connection_contract(c, s.k, s.k);
    const Vec4<Real> ce1 = ks_connection_contract(c, s.k, s.e1);
    const Vec4<Real> ce2 = ks_connection_contract(c, s.k, s.e2);
    for (int mu = 0; mu < ndim; ++mu) {
        d.k[mu] = -ck[mu];
        d.e1[mu] = -ce1[mu];
        d.e2[mu] = -ce2[mu];
    }
    return d;
}

} // namespace kpolaris
