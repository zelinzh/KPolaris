#pragma once

#include "common/math.hpp"

namespace kpolaris {

// Post-processing parameter families at fixed velocity, field direction and
// radiating-domain mask. q is a logarithmic scale, with the baseline at q=0.
enum class PlasmaParameter { none = 0, density_scale = 1, temperature_scale = 2,
                             magnetic_scale = 3, coefficients = 4 };

template<class Real = DefaultReal>
struct PlasmaPerturbation {
    PlasmaParameter parameter = PlasmaParameter::none;
    Real log_scale = Real(0);

    KPOLARIS_INLINE void apply(Real& ne, Real& thetae, Real& b,
                              Real& sigma, Real& beta) const {
        if (log_scale == Real(0)) return;
        const Real f = Kokkos::exp(log_scale);
        if (parameter == PlasmaParameter::density_scale) {
            // The standard GRMHD mass-unit family: rho,p ~ M_unit, B ~ sqrt(M_unit).
            ne *= f;
            b *= Kokkos::exp(log_scale * Real(0.5));
        } else if (parameter == PlasmaParameter::temperature_scale) {
            thetae *= f;
        } else if (parameter == PlasmaParameter::magnetic_scale) {
            b *= f;
            sigma *= f * f;
            beta /= f * f;
        }
    }
};

} // namespace kpolaris
