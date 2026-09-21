#include <array>
#include <iomanip>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "KPolaris.hpp"

namespace {
using Real = kpolaris::DefaultReal;

kpolaris::TransportState<Real> make_state(const kpolaris::KerrSchildInMetric<Real>& metric,
                                       const kpolaris::Vec4<Real>& x,
                                       const kpolaris::Vec3<Real>& spatial_k) {
    kpolaris::TransportState<Real> state;
    state.x = x;
    const Real k0 = kpolaris::solve_future_null_k0(metric, x, spatial_k);
    state.k = kpolaris::Vec4<Real>(k0, spatial_k.x, spatial_k.y, spatial_k.z);
    kpolaris::initialize_screen_frame_from_reference(metric, state.x, state.k,
                                                  state.e1, state.e2);
    return state;
}
} // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        const kpolaris::KerrSchildInMetric<Real> metric(Real(1), Real(0.9));
        kpolaris::MagnetizedTorusRadiationModel<Real> model;
        model.spin = Real(0.9);
        model.initialize_default_torus();

        const std::array<Real, 5> radii = {
            model.rcusp + (model.rc - model.rcusp) * Real(0.35),
            model.rcusp + (model.rc - model.rcusp) * Real(0.70),
            model.rc,
            model.rc * Real(1.25),
            model.rc * Real(1.75)
        };
        const std::array<Real, 5> zfrac = {Real(0), Real(0.05), Real(0), Real(0.08), Real(0)};
        const std::array<kpolaris::Vec3<Real>, 5> ks = {
            kpolaris::Vec3<Real>(Real(-1), Real(0.05), Real(0.02)),
            kpolaris::Vec3<Real>(Real(-0.9), Real(0.12), Real(-0.03)),
            kpolaris::Vec3<Real>(Real(-1), Real(0.02), Real(0.01)),
            kpolaris::Vec3<Real>(Real(-0.85), Real(-0.08), Real(0.03)),
            kpolaris::Vec3<Real>(Real(-0.8), Real(0.05), Real(0))
        };

        std::cout << std::setprecision(17);
        std::cout << "# l0," << model.l0 << "\n";
        std::cout << "# rcusp," << model.rcusp << "\n";
        std::cout << "# rc," << model.rc << "\n";
        std::cout << "# r_outer," << model.r_outer << "\n";
        std::cout << "# Wc," << model.Wc << "\n";
        std::cout << "# Win," << model.Win << "\n";
        std::cout << "sample,x1,x2,x3,k0,k1,k2,k3,r,theta,Wpot,dlambda_scale,jI,jQ,jU,jV,aI,aQ,aU,aV,rQ,rU,rV\n";
        for (std::size_t i = 0; i < radii.size(); ++i) {
            const Real r = radii[i];
            const Real z = zfrac[i] * r;
            const Real cylindrical = std::sqrt(std::max<Real>(Real(0), r * r - z * z));
            const kpolaris::Vec4<Real> x(Real(0), cylindrical, -model.spin, z);
            const auto state = make_state(metric, x, ks[i]);
            Real r_bl = Real(0), th = Real(0), cp = Real(1), sp = Real(0);
            model.bl_coordinates(state.x, r_bl, th, cp, sp);
            const Real Wpot = model.potential(r_bl, th);
            const auto coeffs = model.coefficients(metric, state, Real(0));
            std::cout << i << ','
                      << state.x[1] << ',' << state.x[2] << ',' << state.x[3] << ','
                      << state.k[0] << ',' << state.k[1] << ',' << state.k[2] << ',' << state.k[3] << ','
                      << r_bl << ',' << th << ',' << Wpot << ',' << model.dlambda_scale() << ','
                      << coeffs.jI << ',' << coeffs.jQ << ',' << coeffs.jU << ',' << coeffs.jV << ','
                      << coeffs.aI << ',' << coeffs.aQ << ',' << coeffs.aU << ',' << coeffs.aV << ','
                      << coeffs.rQ << ',' << coeffs.rU << ',' << coeffs.rV << '\n';
        }
    }
    Kokkos::finalize();
    return 0;
}
