#pragma once

#include "common/math.hpp"
#include "common/vec.hpp"
#include "radiation/stokes.hpp"

namespace kpolaris {

template<class Real>
KPOLARIS_INLINE void faraday_rotate(Stokes<Real>& stokes,
                                 const TransferCoeffs<Real>& coeff,
                                 Real dlambda) {
    Vec3<Real> p(stokes.Q, stokes.U, stokes.V);
    const Vec3<Real> rho(coeff.rQ, coeff.rU, coeff.rV);
    const Real rho_norm = norm(rho);

    if (rho_norm == Real(0) || dlambda == Real(0)) {
        return;
    }

    const Real theta = rho_norm * dlambda;
    Vec3<Real> p_out;

    if (abs_val(theta) < Real(1e-8)) {
        p_out = p + cross(rho, p) * dlambda;
    } else {
        const Vec3<Real> n = rho * (Real(1) / rho_norm);
        const Real ct = Kokkos::cos(theta);
        const Real st = Kokkos::sin(theta);
        p_out = p * ct + n * (dot(n, p) * (Real(1) - ct)) +
                cross(n, p) * st;
    }

    stokes.Q = p_out.x;
    stokes.U = p_out.y;
    stokes.V = p_out.z;
}

template<class Real>
KPOLARIS_INLINE void absorb_emit(Stokes<Real>& stokes,
                              const TransferCoeffs<Real>& coeff,
                              Real dlambda) {
    Vec3<Real> p(stokes.Q, stokes.U, stokes.V);
    const Vec3<Real> j_p(coeff.jQ, coeff.jU, coeff.jV);
    const Vec3<Real> alpha(coeff.aQ, coeff.aU, coeff.aV);
    const Real alpha_norm = norm(alpha);

    if (alpha_norm < tiny_positive<Real>()) {
        const Real e = stable_attenuation(coeff.aI, dlambda);
        const Real f = stable_source_factor(coeff.aI, dlambda);

        stokes.I = e * stokes.I + f * coeff.jI;
        stokes.Q = e * stokes.Q + f * coeff.jQ;
        stokes.U = e * stokes.U + f * coeff.jU;
        stokes.V = e * stokes.V + f * coeff.jV;
        return;
    }

    const Vec3<Real> n = alpha * (Real(1) / alpha_norm);
    const Real p_parallel = dot(n, p);
    const Vec3<Real> p_perp = p - n * p_parallel;
    const Real j_parallel = dot(n, j_p);
    const Vec3<Real> j_perp = j_p - n * j_parallel;

    const Real s_plus = stokes.I + p_parallel;
    const Real s_minus = stokes.I - p_parallel;
    const Real j_plus = coeff.jI + j_parallel;
    const Real j_minus = coeff.jI - j_parallel;
    const Real k_plus = coeff.aI + alpha_norm;
    const Real k_minus = coeff.aI - alpha_norm;

    const Real s_plus_out =
        stable_attenuation(k_plus, dlambda) * s_plus +
        stable_source_factor(k_plus, dlambda) * j_plus;
    const Real s_minus_out =
        stable_attenuation(k_minus, dlambda) * s_minus +
        stable_source_factor(k_minus, dlambda) * j_minus;

    const Real e_i = stable_attenuation(coeff.aI, dlambda);
    const Real f_i = stable_source_factor(coeff.aI, dlambda);
    const Vec3<Real> p_perp_out = p_perp * e_i + j_perp * f_i;

    const Real i_out = Real(0.5) * (s_plus_out + s_minus_out);
    const Real p_parallel_out = Real(0.5) * (s_plus_out - s_minus_out);
    const Vec3<Real> p_out = n * p_parallel_out + p_perp_out;

    stokes.I = i_out;
    stokes.Q = p_out.x;
    stokes.U = p_out.y;
    stokes.V = p_out.z;
}

template<class Real>
KPOLARIS_INLINE void semi_analytic_stokes_step(Stokes<Real>& stokes,
                                            const TransferCoeffs<Real>& coeff,
                                            Real dlambda) {
    faraday_rotate(stokes, coeff, Real(0.5) * dlambda);
    absorb_emit(stokes, coeff, dlambda);
    faraday_rotate(stokes, coeff, Real(0.5) * dlambda);
}

} // namespace kpolaris
