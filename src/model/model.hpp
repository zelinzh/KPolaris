#pragma once

#include "geodesic/state.hpp"
#include "radiation/stokes.hpp"

namespace kpolaris {

template<class Real = DefaultReal>
struct ConstantRadiationModel {
    TransferCoeffs<Real> coeffs;

    template<class Metric>
    KPOLARIS_INLINE TransferCoeffs<Real> coefficients(const Metric&,
                                                   const TransportState<Real>&,
                                                   Real) const {
        return coeffs;
    }

    KPOLARIS_INLINE Real dlambda_scale() const { return Real(1); }
};

template<class Real = DefaultReal>
struct RadialPowerLawRadiationModel {
    Real jI0 = Real(1);
    Real pol_frac = Real(0.1);
    Real aI0 = Real(0.05);
    Real rV0 = Real(0.1);
    Real r_peak = Real(6);
    Real width = Real(5);
    Real floor = Real(1e-30);

    template<class Metric>
    KPOLARIS_INLINE TransferCoeffs<Real> coefficients(const Metric&,
                                                   const TransportState<Real>& state,
                                                   Real) const {
        const Real r = spatial_radius(state.x);
        const Real x = (r - r_peak) / width;
        const Real profile = Kokkos::exp(-x * x) + floor;
        const Real angle = Kokkos::atan2(state.x[2], state.x[1]);
        const Real c2 = Kokkos::cos(Real(2) * angle);
        const Real s2 = Kokkos::sin(Real(2) * angle);

        TransferCoeffs<Real> coeffs;
        coeffs.jI = jI0 * profile;
        coeffs.jQ = pol_frac * coeffs.jI * c2;
        coeffs.jU = pol_frac * coeffs.jI * s2;
        coeffs.jV = Real(0);
        coeffs.aI = aI0 * profile;
        coeffs.aQ = Real(0);
        coeffs.aU = Real(0);
        coeffs.aV = Real(0);
        coeffs.rQ = Real(0);
        coeffs.rU = Real(0);
        coeffs.rV = rV0 * profile;
        return coeffs;
    }

    KPOLARIS_INLINE Real dlambda_scale() const { return Real(1); }
};

} // namespace kpolaris
