#pragma once

#include "common/vec.hpp"
#include "geometry/connection.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct KerrBoyerLindquistMetric {
    Real mass = Real(1);
    Real spin = Real(0);
    static constexpr CoordinateSystem coordinate_system = CoordinateSystem::BoyerLindquist;

    KPOLARIS_INLINE KerrBoyerLindquistMetric() {}
    KPOLARIS_INLINE KerrBoyerLindquistMetric(Real mass_in, Real spin_in)
        : mass(mass_in), spin(spin_in) {}

    struct Work {
        Real r = Real(0), th = Real(0);
        Real a = Real(0), a2 = Real(0), m = Real(1);
        Real sinth = Real(0), costh = Real(0), sin2 = Real(0), cos2 = Real(0);
        Real sigma = Real(0), delta = Real(0);
        Real dsigma_dr = Real(0), dsigma_dth = Real(0), ddelta_dr = Real(0);
        Real dsin2_dth = Real(0);
    };

    KPOLARIS_INLINE Work build_work(const Vec4<Real>& X) const {
        Work w;
        w.r = max_val(X[1], tiny_positive<Real>());
        w.th = X[2];
        w.a = spin;
        w.a2 = spin * spin;
        w.m = mass;
        w.sinth = Kokkos::sin(w.th);
        w.costh = Kokkos::cos(w.th);
        w.sin2 = max_val(w.sinth * w.sinth, tiny_positive<Real>());
        w.cos2 = w.costh * w.costh;
        w.sigma = w.r * w.r + w.a2 * w.cos2;
        w.delta = w.r * w.r - Real(2) * w.m * w.r + w.a2;
        w.dsigma_dr = Real(2) * w.r;
        w.dsigma_dth = -Real(2) * w.a2 * w.sinth * w.costh;
        w.ddelta_dr = Real(2) * (w.r - w.m);
        w.dsin2_dth = Real(2) * w.sinth * w.costh;
        return w;
    }

    KPOLARIS_INLINE Real gcov(int mu, int nu, const Vec4<Real>& X) const {
        const Work w = build_work(X);
        if (mu > nu) {
            const int tmp = mu;
            mu = nu;
            nu = tmp;
        }
        if (mu == 0 && nu == 0) {
            return -(Real(1) - Real(2) * w.m * w.r / w.sigma);
        }
        if (mu == 0 && nu == 3) {
            return -Real(2) * w.m * w.a * w.r * w.sin2 / w.sigma;
        }
        if (mu == 1 && nu == 1) {
            return w.sigma / w.delta;
        }
        if (mu == 2 && nu == 2) {
            return w.sigma;
        }
        if (mu == 3 && nu == 3) {
            return w.sin2 * (w.r * w.r + w.a2 +
                             Real(2) * w.m * w.a2 * w.r * w.sin2 / w.sigma);
        }
        return Real(0);
    }

    KPOLARIS_INLINE Real gcon(int mu, int nu, const Vec4<Real>& X) const {
        const Work w = build_work(X);
        if (mu > nu) {
            const int tmp = mu;
            mu = nu;
            nu = tmp;
        }
        const Real inv_sigma_delta = Real(1) / (w.sigma * w.delta);
        if (mu == 0 && nu == 0) {
            const Real r2pa2 = w.r * w.r + w.a2;
            return -(r2pa2 * r2pa2 - w.a2 * w.delta * w.sin2) * inv_sigma_delta;
        }
        if (mu == 0 && nu == 3) {
            return -Real(2) * w.m * w.a * w.r * inv_sigma_delta;
        }
        if (mu == 1 && nu == 1) {
            return w.delta / w.sigma;
        }
        if (mu == 2 && nu == 2) {
            return Real(1) / w.sigma;
        }
        if (mu == 3 && nu == 3) {
            return (w.delta - w.a2 * w.sin2) /
                   (w.sigma * w.delta * w.sin2);
        }
        return Real(0);
    }

    KPOLARIS_INLINE void gcov_matrix(const Vec4<Real>& X, Real g[ndim][ndim]) const {
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                g[mu][nu] = gcov(mu, nu, X);
            }
        }
    }

    KPOLARIS_INLINE void gcon_matrix(const Vec4<Real>& X, Real g[ndim][ndim]) const {
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                g[mu][nu] = gcon(mu, nu, X);
            }
        }
    }

    KPOLARIS_INLINE Real dot(const Vec4<Real>& X,
                          const Vec4<Real>& u,
                          const Vec4<Real>& v) const {
        Real out = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                out += gcov(mu, nu, X) * u[mu] * v[nu];
            }
        }
        return out;
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

        const Real sig = w.sigma;
        const Real sig2 = sig * sig;
        const Real del = w.delta;
        const Real del2 = del * del;
        const Real r = w.r;
        const Real m = w.m;
        const Real a = w.a;
        const Real a2 = w.a2;
        const Real s2 = w.sin2;
        const Real ds2 = w.dsin2_dth;
        const Real sig_r = w.dsigma_dr;
        const Real sig_t = w.dsigma_dth;
        const Real del_r = w.ddelta_dr;

        auto set_sym = [&](int mu, int nu, int alpha, Real val) {
            dg[mu][nu][alpha] = val;
            dg[nu][mu][alpha] = val;
        };

        set_sym(0, 0, 1, Real(2) * m * (sig - r * sig_r) / sig2);
        set_sym(0, 0, 2, -Real(2) * m * r * sig_t / sig2);

        set_sym(0, 3, 1, -Real(2) * m * a * s2 * (sig - r * sig_r) / sig2);
        set_sym(0, 3, 2, -Real(2) * m * a * r * (ds2 * sig - s2 * sig_t) / sig2);

        set_sym(1, 1, 1, (sig_r * del - sig * del_r) / del2);
        set_sym(1, 1, 2, sig_t / del);

        set_sym(2, 2, 1, sig_r);
        set_sym(2, 2, 2, sig_t);

        const Real extra = Real(2) * m * a2 * r * s2 / sig;
        const Real aterm = r * r + a2 + extra;
        const Real extra_r = Real(2) * m * a2 * s2 * (sig - r * sig_r) / sig2;
        const Real extra_t = Real(2) * m * a2 * r * (ds2 * sig - s2 * sig_t) / sig2;
        set_sym(3, 3, 1, s2 * (Real(2) * r + extra_r));
        set_sym(3, 3, 2, ds2 * aterm + s2 * extra_t);
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
KPOLARIS_INLINE Real bl_derivative_row_dot(Real d00, Real d03, Real d11,
                                        Real d22, Real d33,
                                        int row,
                                        const Vec4<Real>& v) {
    if (row == 0) {
        return d00 * v[0] + d03 * v[3];
    }
    if (row == 1) {
        return d11 * v[1];
    }
    if (row == 2) {
        return d22 * v[2];
    }
    return d03 * v[0] + d33 * v[3];
}

template<class Real>
KPOLARIS_INLINE Real bl_derivative_bilinear(Real d00, Real d03, Real d11,
                                         Real d22, Real d33,
                                         const Vec4<Real>& p,
                                         const Vec4<Real>& q) {
    return d00 * p[0] * q[0] +
           d03 * (p[0] * q[3] + p[3] * q[0]) +
           d11 * p[1] * q[1] +
           d22 * p[2] * q[2] +
           d33 * p[3] * q[3];
}

template<class Real>
struct BLConnectionWork {
    Real dr00, dr03, dr11, dr22, dr33;
    Real dt00, dt03, dt11, dt22, dt33;
    Real g00, g03, g11, g22, g33;
};

template<class Real>
KPOLARIS_INLINE BLConnectionWork<Real> build_bl_connection_work(
    const KerrBoyerLindquistMetric<Real>& metric,
    const Vec4<Real>& X) {
    const typename KerrBoyerLindquistMetric<Real>::Work w = metric.build_work(X);
    BLConnectionWork<Real> c;
    const Real sig = w.sigma;
    const Real sig2 = sig * sig;
    const Real del = w.delta;
    const Real del2 = del * del;
    const Real r = w.r;
    const Real m = w.m;
    const Real a = w.a;
    const Real a2 = w.a2;
    const Real s2 = w.sin2;
    const Real ds2 = w.dsin2_dth;
    const Real sig_r = w.dsigma_dr;
    const Real sig_t = w.dsigma_dth;
    const Real del_r = w.ddelta_dr;

    c.dr00 = Real(2) * m * (sig - r * sig_r) / sig2;
    c.dt00 = -Real(2) * m * r * sig_t / sig2;
    c.dr03 = -Real(2) * m * a * s2 * (sig - r * sig_r) / sig2;
    c.dt03 = -Real(2) * m * a * r * (ds2 * sig - s2 * sig_t) / sig2;
    c.dr11 = (sig_r * del - sig * del_r) / del2;
    c.dt11 = sig_t / del;
    c.dr22 = sig_r;
    c.dt22 = sig_t;
    const Real extra = Real(2) * m * a2 * r * s2 / sig;
    const Real aterm = r * r + a2 + extra;
    const Real extra_r = Real(2) * m * a2 * s2 * (sig - r * sig_r) / sig2;
    const Real extra_t = Real(2) * m * a2 * r * (ds2 * sig - s2 * sig_t) / sig2;
    c.dr33 = s2 * (Real(2) * r + extra_r);
    c.dt33 = ds2 * aterm + s2 * extra_t;

    const Real inv_sigma_delta = Real(1) / (sig * del);
    const Real r2pa2 = r * r + a2;
    c.g00 = -(r2pa2 * r2pa2 - a2 * del * s2) * inv_sigma_delta;
    c.g03 = -Real(2) * m * a * r * inv_sigma_delta;
    c.g11 = del / sig;
    c.g22 = Real(1) / sig;
    c.g33 = (del - a2 * s2) / (sig * del * s2);
    return c;
}

template<class Real>
KPOLARIS_INLINE Vec4<Real> bl_connection_contract(
    const BLConnectionWork<Real>& c,
    const Vec4<Real>& p,
    const Vec4<Real>& q) {
    Real t[ndim];
    for (int nu = 0; nu < ndim; ++nu) {
        t[nu] = p[1] * bl_derivative_row_dot(c.dr00, c.dr03, c.dr11, c.dr22, c.dr33, nu, q) +
                p[2] * bl_derivative_row_dot(c.dt00, c.dt03, c.dt11, c.dt22, c.dt33, nu, q) +
                q[1] * bl_derivative_row_dot(c.dr00, c.dr03, c.dr11, c.dr22, c.dr33, nu, p) +
                q[2] * bl_derivative_row_dot(c.dt00, c.dt03, c.dt11, c.dt22, c.dt33, nu, p);
    }
    t[1] -= bl_derivative_bilinear(c.dr00, c.dr03, c.dr11, c.dr22, c.dr33, p, q);
    t[2] -= bl_derivative_bilinear(c.dt00, c.dt03, c.dt11, c.dt22, c.dt33, p, q);

    Vec4<Real> out;
    out[0] = Real(0.5) * (c.g00 * t[0] + c.g03 * t[3]);
    out[1] = Real(0.5) * c.g11 * t[1];
    out[2] = Real(0.5) * c.g22 * t[2];
    out[3] = Real(0.5) * (c.g03 * t[0] + c.g33 * t[3]);
    return out;
}

template<class Real>
KPOLARIS_INLINE Vec4<Real> bl_connection_contract(
    const KerrBoyerLindquistMetric<Real>& metric,
    const Vec4<Real>& X,
    const Vec4<Real>& p,
    const Vec4<Real>& q) {
    return bl_connection_contract(build_bl_connection_work(metric, X), p, q);
}

template<class Real>
KPOLARIS_INLINE TransportState<Real> transport_rhs(
    const KerrBoyerLindquistMetric<Real>& metric,
    const TransportState<Real>& s) {
    TransportState<Real> d;
    d.x = s.k;
    const BLConnectionWork<Real> c = build_bl_connection_work(metric, s.x);
    const Vec4<Real> ck = bl_connection_contract(c, s.k, s.k);
    const Vec4<Real> ce1 = bl_connection_contract(c, s.k, s.e1);
    const Vec4<Real> ce2 = bl_connection_contract(c, s.k, s.e2);
    for (int mu = 0; mu < ndim; ++mu) {
        d.k[mu] = -ck[mu];
        d.e1[mu] = -ce1[mu];
        d.e2[mu] = -ce2[mu];
    }
    return d;
}


} // namespace kpolaris
