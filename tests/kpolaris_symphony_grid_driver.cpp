#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

#include <Kokkos_Core.hpp>

#include "KPolaris.hpp"

namespace {

using Real = kpolaris::DefaultReal;

void print_row(int case_id, const char* distribution, Real nu, Real b_cgs,
               Real angle, Real theta_e, Real p, Real kappa, Real kappa_width,
               const char* stokes, Real j, Real alpha, Real rho) {
    std::cout << case_id << ',' << distribution << ',' << nu << ',' << b_cgs
              << ",1," << angle << ',' << theta_e << ',' << p << ",1,1000,"
              << kappa << ',' << stokes << ',' << kappa_width << ','
              << j << ',' << alpha << ',' << rho << ','
              << j << ',' << alpha << ',' << rho << '\n';
}

void emit_coefficients(int case_id, const char* distribution, Real nu,
                       Real b_cgs, Real angle, Real theta_e, Real p, Real kappa,
                       Real kappa_width,
                       const kpolaris::TransferCoeffs<Real>& coeffs) {
    const Real nu2 = nu * nu;
    print_row(case_id, distribution, nu, b_cgs, angle, theta_e, p, kappa,
              kappa_width, "I", coeffs.jI * nu2, coeffs.aI / nu, Real(0));
    print_row(case_id, distribution, nu, b_cgs, angle, theta_e, p, kappa,
              kappa_width, "Q", -coeffs.jQ * nu2, -coeffs.aQ / nu,
              coeffs.rQ / nu);
    print_row(case_id, distribution, nu, b_cgs, angle, theta_e, p, kappa,
              kappa_width, "U", coeffs.jU * nu2, coeffs.aU / nu,
              coeffs.rU / nu);
    print_row(case_id, distribution, nu, b_cgs, angle, theta_e, p, kappa,
              kappa_width, "V", coeffs.jV * nu2, coeffs.aV / nu,
              coeffs.rV / nu);
}

void emit_thermal(int case_id, Real nu, Real b_cgs, Real angle, Real theta_e) {
    kpolaris::LocalThermalSynchrotronState<Real> state;
    state.nu = nu;
    state.ne = Real(1);
    state.thetae = theta_e;
    state.b_cgs = b_cgs;
    state.theta = angle;
    state.sin_theta = std::sin(angle);
    state.cos_theta = std::cos(angle);
    kpolaris::ThermalSynchrotronParams<Real> params;
    params.fit = kpolaris::ThermalSynchrotronPandya;
    emit_coefficients(
        case_id, "thermal", nu, b_cgs, angle, theta_e, Real(2.5), Real(3.5),
        Real(10), kpolaris::thermal_synchrotron_magnetic_basis_coefficients(
                      state, params));
}

void emit_nonthermal(int case_id, const char* distribution, int type, Real nu,
                     Real b_cgs, Real angle, Real p, Real kappa,
                     Real kappa_width) {
    kpolaris::LocalNonthermalSynchrotronState<Real> state;
    state.nu = nu;
    state.ne = Real(1);
    state.thetae = Real(10);
    state.b_cgs = b_cgs;
    state.sin_theta = std::sin(angle);
    state.cos_theta = std::cos(angle);

    kpolaris::NonthermalSynchrotronParams<Real> params;
    params.distribution = type;
    params.kappa = kappa;
    params.kappa_width = kappa_width;
    params.kappa_interp_begin = Real(1e20);
    params.kappa_interp_end = Real(1e20);
    params.power_law_p = p;
    params.power_law_gamma_min = Real(1);
    params.power_law_gamma_max = Real(1000);
    params.power_law_gamma_cutoff = Real(1e10);
    const Real me = Real(9.1093826e-28);
    const Real cl = Real(2.99792458e10);
    params.power_law_eta =
        Real(2) * (p - Real(1)) * me * cl * cl /
        (b_cgs * b_cgs * (p - Real(2)));
    emit_coefficients(
        case_id, distribution, nu, b_cgs, angle, Real(10), p, kappa,
        kappa_width,
        kpolaris::nonthermal_synchrotron_magnetic_basis_coefficients(
            state, params));
}

} // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    {
        constexpr Real pi = Real(3.141592653589793238462643383279502884);
        const std::array<Real, 3> frequencies = {Real(86.e9), Real(230.e9),
                                                 Real(690.e9)};
        const std::array<Real, 3> magnetic_fields = {Real(3), Real(30), Real(100)};
        const std::array<Real, 3> angles = {Real(0.3), pi / Real(3), Real(1.4)};
        const std::array<Real, 4> temperatures = {Real(3), Real(10), Real(30),
                                                  Real(100)};
        const std::array<Real, 3> slopes = {Real(2.1), Real(2.5), Real(3.5)};
        const std::array<Real, 4> kappas = {Real(3.5), Real(3.5), Real(4.25),
                                            Real(5)};
        const std::array<Real, 4> widths = {Real(3), Real(10), Real(10),
                                            Real(10)};
        int case_id = 0;
        std::cout << std::setprecision(17);
        std::cout << "case_id,distribution,nu_hz,b_cgs,ne,angle,theta_e,p,"
                     "gamma_min,gamma_max,kappa,stokes,kappa_width,j,alpha,rho,"
                     "intended_j,intended_alpha,intended_rho\n";
        for (const Real nu : frequencies) {
            for (const Real b_cgs : magnetic_fields) {
                for (const Real theta_e : temperatures) {
                    for (const Real angle : angles) {
                        emit_thermal(case_id++, nu, b_cgs, angle, theta_e);
                    }
                }
            }
        }
        for (const Real nu : frequencies) {
            for (const Real b_cgs : magnetic_fields) {
                for (const Real p : slopes) {
                    for (const Real angle : angles) {
                        emit_nonthermal(case_id++, "powerlaw",
                                        kpolaris::SynchrotronPowerLaw, nu,
                                        b_cgs, angle, p, Real(3.5), Real(10));
                    }
                }
            }
        }
        for (const Real nu : frequencies) {
            for (const Real b_cgs : magnetic_fields) {
                for (std::size_t ik = 0; ik < kappas.size(); ++ik) {
                    for (const Real angle : angles) {
                        emit_nonthermal(case_id++, "kappa",
                                        kpolaris::SynchrotronKappa, nu,
                                        b_cgs, angle, Real(2.5), kappas[ik],
                                        widths[ik]);
                    }
                }
            }
        }
    }
    Kokkos::finalize();
    return 0;
}
