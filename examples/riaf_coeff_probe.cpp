#include <array>
#include <iomanip>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "KPolaris.hpp"

namespace {

using Real = kpolaris::DefaultReal;

kpolaris::TransportState<Real> make_probe_state(const kpolaris::KerrSchildInMetric<Real>& metric,
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
        const kpolaris::KerrSchildInMetric<Real> metric(Real(1), Real(0.9375));
        kpolaris::RIAFAnalyticRadiationModel<Real> model;
        model.r_min = Real(2.1);
        model.r_max = Real(40);
        model.emission_scale = Real(1);
        model.absorption_scale = Real(1);
        model.faraday_scale = Real(1);
        model.keplerian_factor = Real(1);
        model.infall_factor = Real(0.1);

        const std::array<kpolaris::Vec4<Real>, 5> xs = {
            kpolaris::Vec4<Real>(Real(0), Real(6), Real(0), Real(0)),
            kpolaris::Vec4<Real>(Real(0), Real(8), Real(2), Real(0)),
            kpolaris::Vec4<Real>(Real(0), Real(10), Real(0), Real(1)),
            kpolaris::Vec4<Real>(Real(0), Real(14), Real(4), Real(2)),
            kpolaris::Vec4<Real>(Real(0), Real(25), Real(0), Real(0))
        };
        const std::array<kpolaris::Vec3<Real>, 5> ks = {
            kpolaris::Vec3<Real>(Real(1), Real(0.15), Real(0.05)),
            kpolaris::Vec3<Real>(Real(0.8), Real(-0.25), Real(0.08)),
            kpolaris::Vec3<Real>(Real(0.9), Real(0.05), Real(-0.15)),
            kpolaris::Vec3<Real>(Real(0.7), Real(-0.35), Real(0.1)),
            kpolaris::Vec3<Real>(Real(1), Real(0.05), Real(0))
        };

        std::cout << std::setprecision(17);
        std::cout << "sample,x1,x2,x3,k0,k1,k2,k3,r,ne_norm,thetae,B_cgs,nu_fluid_scale,dlambda_scale,cos_theta,sin_theta,u_dot_u,b_dot_u,b_dot_b,jI,jQ,jU,jV,aI,aQ,aU,aV,rQ,rU,rV\n";
        for (std::size_t i = 0; i < xs.size(); ++i) {
            const auto state = make_probe_state(metric, xs[i], ks[i]);
            const auto work = metric.build_work(state.x);
            const Real r = work.r;
            const Real ne_norm = model.density_profile(state.x, r);
            const Real thetae = model.thetae_profile(r);
            const Real b_cgs = model.magnetic_field_cgs(ne_norm * model.ne_unit, r);
            const auto ucon = model.fluid_four_velocity(metric, state.x, r);
            const auto bcon = model.magnetic_unit_four_vector(metric, state.x, ucon);
            const auto bcov = model.lower_vector(metric, state.x, bcon);
            const Real nu_scale = model.fluid_frequency_scale(metric, state, r);
            Real kdotb = Real(0);
            for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                kdotb += state.k[mu] * bcov[mu];
            }
            const Real cos_theta = kpolaris::clamp(kdotb / nu_scale, Real(-1), Real(1));
            const Real sin_theta = std::sqrt(std::max<Real>(Real(0), Real(1) - cos_theta * cos_theta));
            const auto coeffs = model.coefficients(metric, state, Real(0));

            std::cout << i << ','
                      << state.x[1] << ',' << state.x[2] << ',' << state.x[3] << ','
                      << state.k[0] << ',' << state.k[1] << ',' << state.k[2] << ',' << state.k[3] << ','
                      << r << ',' << ne_norm << ',' << thetae << ',' << b_cgs << ','
                      << nu_scale << ',' << model.dlambda_scale() << ','
                      << cos_theta << ',' << sin_theta << ','
                      << metric.dot(state.x, ucon, ucon) << ','
                      << metric.dot(state.x, bcon, ucon) << ','
                      << metric.dot(state.x, bcon, bcon) << ','
                      << coeffs.jI << ',' << coeffs.jQ << ',' << coeffs.jU << ',' << coeffs.jV << ','
                      << coeffs.aI << ',' << coeffs.aQ << ',' << coeffs.aU << ',' << coeffs.aV << ','
                      << coeffs.rQ << ',' << coeffs.rU << ',' << coeffs.rV << '\n';
        }
    }
    Kokkos::finalize();
    return 0;
}
