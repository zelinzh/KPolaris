#pragma once

#include "common/types.hpp"

namespace kpolaris {

template<class Real>
KPOLARIS_INLINE void fill_connection_from_metric_derivs(
    const Real gcon[ndim][ndim],
    const Real dg[ndim][ndim][ndim],
    Real gamma[ndim][ndim][ndim]) {
    for (int mu = 0; mu < ndim; ++mu) {
        for (int a = 0; a < ndim; ++a) {
            for (int b = 0; b < ndim; ++b) {
                Real comp = Real(0);
                for (int nu = 0; nu < ndim; ++nu) {
                    comp += Real(0.5) * gcon[mu][nu] *
                            (dg[nu][b][a] + dg[nu][a][b] - dg[a][b][nu]);
                }
                gamma[mu][a][b] = comp;
            }
        }
    }
}

} // namespace kpolaris
