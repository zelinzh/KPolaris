#pragma once

#include "common/types.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct Stokes {
    Real I = Real(0);
    Real Q = Real(0);
    Real U = Real(0);
    Real V = Real(0);

    KPOLARIS_INLINE Stokes() {}
    KPOLARIS_INLINE Stokes(Real i, Real q, Real u, Real v)
        : I(i), Q(q), U(u), V(v) {}
};

template<class Real = DefaultReal>
struct TransferCoeffs {
    Real jI = Real(0);
    Real jQ = Real(0);
    Real jU = Real(0);
    Real jV = Real(0);
    Real aI = Real(0);
    Real aQ = Real(0);
    Real aU = Real(0);
    Real aV = Real(0);
    Real rQ = Real(0);
    Real rU = Real(0);
    Real rV = Real(0);
};

} // namespace kpolaris
