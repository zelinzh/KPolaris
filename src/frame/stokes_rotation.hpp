#pragma once

#include "common/math.hpp"
#include "radiation/stokes.hpp"

namespace kpolaris {

template<class Real>
struct BasisOverlap2 {
    Real r11 = Real(1);
    Real r12 = Real(0);
    Real r21 = Real(0);
    Real r22 = Real(1);

    KPOLARIS_INLINE Real det() const { return r11 * r22 - r12 * r21; }

    KPOLARIS_INLINE Real orthogonality_error() const {
        const Real c11 = r11 * r11 + r21 * r21;
        const Real c12 = r11 * r12 + r21 * r22;
        const Real c22 = r12 * r12 + r22 * r22;
        return abs_val(c11 - Real(1)) + Real(2) * abs_val(c12) +
               abs_val(c22 - Real(1));
    }
};

template<class Real>
KPOLARIS_INLINE Stokes<Real> rotate_qu(Stokes<Real> in, Real psi) {
    // Active EVPA rotation by +psi in a fixed (e1,e2) basis. A passive
    // rotation of the axes by +psi instead uses -psi. No image registration.
    const Real c2 = Kokkos::cos(Real(2) * psi);
    const Real s2 = Kokkos::sin(Real(2) * psi);

    Stokes<Real> out = in;
    out.Q = in.Q * c2 - in.U * s2;
    out.U = in.Q * s2 + in.U * c2;
    return out;
}

template<class Real>
KPOLARIS_INLINE Stokes<Real> transform_to_observer_basis(
    Stokes<Real> in,
    const BasisOverlap2<Real>& overlap) {
    // R_ij = observer_e_i . transported_e_j. Transform the transverse
    // coherency tensor as C' = R C R^T; a reflection reverses V through det R.
    const Real r11 = overlap.r11;
    const Real r12 = overlap.r12;
    const Real r21 = overlap.r21;
    const Real r22 = overlap.r22;

    Stokes<Real> out = in;
    out.Q = Real(0.5) * in.Q * (r11 * r11 - r12 * r12 - r21 * r21 + r22 * r22) +
            in.U * (r11 * r12 - r21 * r22);
    out.U = in.Q * (r11 * r21 - r12 * r22) +
            in.U * (r11 * r22 + r12 * r21);
    out.V = overlap.det() * in.V;
    return out;
}

} // namespace kpolaris
