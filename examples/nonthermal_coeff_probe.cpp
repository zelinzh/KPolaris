#include <cmath>
#include <iomanip>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "KPolaris.hpp"

namespace {
using Real = kpolaris::DefaultReal;

void print_coeffs(const char* label, int emission_type, Real kappa) {
    kpolaris::LocalNonthermalSynchrotronState<Real> state;
    state.nu = Real(230.0e9);
    state.ne = Real(1.0e6);
    state.thetae = Real(20);
    state.b_cgs = Real(30);
    state.sin_theta = Real(0.7);
    state.cos_theta = Kokkos::sqrt(Real(1) - state.sin_theta * state.sin_theta);

    kpolaris::NonthermalSynchrotronParams<Real> params;
    params.distribution = emission_type;
    params.kappa = kappa;
    params.kappa_width = (kappa - Real(3)) / kappa * state.thetae;
    params.kappa_interp_begin = Real(7);
    params.kappa_interp_end = Real(7);
    params.power_law_p = Real(3.25);
    params.power_law_eta = Real(0.02);
    params.power_law_gamma_min = Real(1e2);
    params.power_law_gamma_max = Real(1e5);
    params.power_law_gamma_cutoff = Real(1e10);
    params.max_pol_frac_emission = Real(0.99);
    params.max_pol_frac_absorption = Real(0.99);

    const auto c = kpolaris::nonthermal_synchrotron_magnetic_basis_coefficients(state, params);
    std::cout << label << ',' << emission_type << ',' << kappa << ','
              << c.jI << ',' << c.jQ << ',' << c.jU << ',' << c.jV << ','
              << c.aI << ',' << c.aQ << ',' << c.aU << ',' << c.aV << ','
              << c.rQ << ',' << c.rU << ',' << c.rV << '\n';
}

void print_thermal_rho() {
    const Real nu = Real(230.0e9);
    const Real ne = Real(1.0e6);
    const Real thetae = Real(20);
    const Real b_cgs = Real(30);
    const Real sin_theta = Real(0.7);
    const Real cos_theta = Kokkos::sqrt(Real(1) - sin_theta * sin_theta);
    Real rq = Real(0), rv = Real(0);
    kpolaris::symphony_rho_QV(nu, ne, thetae, b_cgs, sin_theta, cos_theta, rq, rv);
    std::cout << "thermal_rho,1,0,0,0,0,0,0,0,0,0,"
              << rq * nu << ",0," << rv * nu << '\n';
}

void print_symphony_reference_row(const char* label,
                                  const kpolaris::TransferCoeffs<Real>& c) {
    std::cout << label << ",0,0,"
              << c.jI << ',' << c.jQ << ',' << c.jU << ',' << c.jV << ','
              << c.aI << ',' << c.aQ << ',' << c.aU << ',' << c.aV << ','
              << c.rQ << ',' << c.rU << ',' << c.rV << '\n';
}

void print_symphony_reference_points() {
    constexpr Real pi = Real(3.141592653589793238462643383279502884);
    const Real nu = Real(230.e9);
    const Real magnetic_field = Real(30);
    const Real electron_density = Real(1);
    const Real angle = pi / Real(3);

    kpolaris::LocalThermalSynchrotronState<Real> thermal_state;
    thermal_state.nu = nu;
    thermal_state.ne = electron_density;
    thermal_state.thetae = Real(10);
    thermal_state.b_cgs = magnetic_field;
    thermal_state.theta = angle;
    thermal_state.sin_theta = Kokkos::sin(angle);
    thermal_state.cos_theta = Kokkos::cos(angle);
    kpolaris::ThermalSynchrotronParams<Real> thermal_params;
    thermal_params.fit = kpolaris::ThermalSynchrotronPandya;
    print_symphony_reference_row(
        "symphony_thermal",
        kpolaris::thermal_synchrotron_magnetic_basis_coefficients(
            thermal_state, thermal_params));

    kpolaris::LocalNonthermalSynchrotronState<Real> state;
    state.nu = nu;
    state.ne = electron_density;
    state.thetae = Real(10);
    state.b_cgs = magnetic_field;
    state.sin_theta = Kokkos::sin(angle);
    state.cos_theta = Kokkos::cos(angle);

    kpolaris::NonthermalSynchrotronParams<Real> params;
    params.distribution = kpolaris::SynchrotronKappa;
    params.kappa = Real(3.5);
    params.kappa_width = Real(10);
    params.kappa_interp_begin = Real(1e20);
    params.kappa_interp_end = Real(1e20);
    print_symphony_reference_row(
        "symphony_kappa",
        kpolaris::nonthermal_synchrotron_magnetic_basis_coefficients(
            state, params));

    params.distribution = kpolaris::SynchrotronPowerLaw;
    params.power_law_p = Real(2.5);
    params.power_law_gamma_min = Real(1);
    params.power_law_gamma_max = Real(1000);
    params.power_law_gamma_cutoff = Real(1e10);
    const Real me = Real(9.1093826e-28);
    const Real cl = Real(2.99792458e10);
    params.power_law_eta =
        Real(2) * (params.power_law_p - Real(1)) * me * cl * cl *
        params.power_law_gamma_min /
        (magnetic_field * magnetic_field * (params.power_law_p - Real(2)));
    print_symphony_reference_row(
        "symphony_powerlaw",
        kpolaris::nonthermal_synchrotron_magnetic_basis_coefficients(
            state, params));
}

} // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        std::cout << std::setprecision(17);
        std::cout << "label,emission_type,kappa,jI,jQ,jU,jV,aI,aQ,aU,aV,rQ,rU,rV\n";
        print_coeffs("kappa35", kpolaris::SynchrotronKappa, Real(3.5));
        print_coeffs("kappa60", kpolaris::SynchrotronKappa, Real(6.0));
        print_coeffs("powerlaw", kpolaris::SynchrotronPowerLaw, Real(3.5));
        print_thermal_rho();
        print_symphony_reference_points();
    }
    Kokkos::finalize();
    return 0;
}
