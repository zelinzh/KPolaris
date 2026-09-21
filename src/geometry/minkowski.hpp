#pragma once

#include "common/vec.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct MinkowskiMetric {
    KPOLARIS_INLINE Real gcov(int mu, int nu, const Vec4<Real>&) const {
        if (mu != nu) {
            return Real(0);
        }
        return mu == 0 ? Real(-1) : Real(1);
    }

    KPOLARIS_INLINE Real gcon(int mu, int nu, const Vec4<Real>& x) const {
        return gcov(mu, nu, x);
    }

    KPOLARIS_INLINE Real dot(const Vec4<Real>& x,
                          const Vec4<Real>& a,
                          const Vec4<Real>& b) const {
        Real out = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                out += gcov(mu, nu, x) * a[mu] * b[nu];
            }
        }
        return out;
    }

    KPOLARIS_INLINE void connection(const Vec4<Real>&,
                                 Real gamma[ndim][ndim][ndim]) const {
        for (int mu = 0; mu < ndim; ++mu) {
            for (int a = 0; a < ndim; ++a) {
                for (int b = 0; b < ndim; ++b) {
                    gamma[mu][a][b] = Real(0);
                }
            }
        }
    }
};

} // namespace kpolaris
