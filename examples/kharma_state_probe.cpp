#include <iostream>

#include <Kokkos_Core.hpp>

#include "camera/camera.hpp"
#include "geometry/kerr_schild_spherical.hpp"
#include "grmhd/kharma_loader.hpp"

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    try {
        if (argc < 2) {
            std::cerr << "usage: kpolaris_kharma_state_probe dump.phdf [r theta phi M_unit sigma_cut]\n";
            Kokkos::finalize();
            return 2;
        }
        using Real = kpolaris::DefaultReal;
        kpolaris::KHARMALoadOptions opt;
        opt.dump_path = argv[1];
        opt.M_unit = argc > 5 ? Real(std::stod(argv[5])) : Real(3e26);
        opt.sigma_cut = argc > 6 ? Real(std::stod(argv[6])) : Real(1);
        opt.timing = 1;
        auto model = kpolaris::load_kharma_model_from_phdf(opt);
        Real r = argc > 2 ? Real(std::stod(argv[2])) : Real(20);
        Real th = argc > 3 ? Real(std::stod(argv[3])) : Real(1.5707963267948966);
        Real ph = argc > 4 ? Real(std::stod(argv[4])) : Real(0);
        kpolaris::KerrSchildSphericalMetric<Real> metric(Real(1), model.spin);
        kpolaris::TransportState<Real> state;
        state.x = kpolaris::Vec4<Real>(Real(0), r, th, ph);
        Real rho = 0, uu = 0, ne = 0, thetae = 0, b_cgs = 0, sigma = 0, beta = 0, bnorm = 0;
        kpolaris::Vec4<Real> ucon, bcon;
        int ok = model.fluid_state(metric, state, rho, uu, ucon, bcon, ne, thetae, b_cgs, sigma, beta, nullptr, &bnorm);
        std::cout.precision(17);
        std::cout << "ok " << ok << "\n";
        std::cout << "n " << model.n1 << ' ' << model.n2 << ' ' << model.n3 << "\n";
        std::cout << "spin " << model.spin << " r_in " << model.r_in << " r_out " << model.r_out << "\n";
        std::cout << "x " << r << ' ' << th << ' ' << ph << "\n";
        std::cout << "rho " << rho << " uu " << uu << " ne " << ne << " thetae " << thetae
                  << " b_cgs " << b_cgs << " sigma " << sigma << " beta " << beta
                  << " bnorm " << bnorm << "\n";
        std::cout << "ucon " << ucon[0] << ' ' << ucon[1] << ' ' << ucon[2] << ' ' << ucon[3] << "\n";
        std::cout << "bcon " << bcon[0] << ' ' << bcon[1] << ' ' << bcon[2] << ' ' << bcon[3] << "\n";
        state.k = kpolaris::Vec4<Real>(Real(0), Real(-1), Real(0), Real(0));
        state.k[0] = kpolaris::solve_future_null_k0(metric, state.x, kpolaris::Vec3<Real>(Real(-1), Real(0), Real(0)));
        state.e1 = kpolaris::Vec4<Real>(Real(0), Real(0), Real(1) / r, Real(0));
        state.e2 = kpolaris::Vec4<Real>(Real(0), Real(0), Real(0), Real(1) / (r * Kokkos::sin(th)));
        auto coeffs = model.coefficients(metric, state, Real(0));
        std::cout << "coeffs j " << coeffs.jI << ' ' << coeffs.jQ << ' ' << coeffs.jU << ' ' << coeffs.jV
                  << " a " << coeffs.aI << ' ' << coeffs.aQ << ' ' << coeffs.aU << ' ' << coeffs.aV
                  << " rho " << coeffs.rQ << ' ' << coeffs.rU << ' ' << coeffs.rV << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        Kokkos::finalize();
        return 1;
    }
    Kokkos::finalize();
    return 0;
}
