#include <iostream>

#include <Kokkos_Core.hpp>

#include "camera/camera.hpp"
#include "geometry/kerr_schild_cartesian.hpp"
#include "grmhd/athenak_loader.hpp"

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    try {
        if (argc < 2) {
            std::cerr << "usage: kpolaris_athenak_state_probe dump.bin [x y z M_unit sigma_cut r_out gamma]\n";
            Kokkos::finalize();
            return 2;
        }
        using Real = kpolaris::DefaultReal;
        kpolaris::AthenaKLoadOptions opt;
        opt.dump_path = argv[1];
        opt.M_unit = argc > 5 ? Real(std::stod(argv[5])) : Real(1e26);
        opt.sigma_cut = argc > 6 ? Real(std::stod(argv[6])) : Real(1);
        opt.resample_r_out = argc > 7 ? Real(std::stod(argv[7])) : Real(100);
        opt.gamma = argc > 8 ? Real(std::stod(argv[8])) : Real(-1);
        opt.timing = 0;
        auto model = kpolaris::load_athenak_direct_model_from_binary(opt);
        Real x = argc > 2 ? Real(std::stod(argv[2])) : Real(20);
        Real y = argc > 3 ? Real(std::stod(argv[3])) : Real(0);
        Real z = argc > 4 ? Real(std::stod(argv[4])) : Real(0);
        kpolaris::KerrSchildInMetric<Real> metric(Real(1), model.spin);
        kpolaris::TransportState<Real> state;
        state.x = kpolaris::Vec4<Real>(Real(0), x, y, z);
        const auto work = metric.build_work(state.x);
        const int mb = model.find_meshblock(x, y, z);
        std::cout.precision(17);
        std::cout << "nblocks " << model.nblocks << " mb_n " << model.nx1 << ' ' << model.nx2 << ' ' << model.nx3 << "\n";
        std::cout << "spin " << model.spin << " gam " << model.gam << " r_in " << model.r_in << " r_out " << model.r_out << "\n";
        std::cout << "x " << x << ' ' << y << ' ' << z << " r " << work.r << " meshblock " << mb << "\n";
        if (mb >= 0) {
            std::cout << "raw_prim";
            for (int v = 0; v < 8; ++v) std::cout << ' ' << model.interp_prim(mb, v, x, y, z);
            std::cout << "\n";
        }
        Real ne = 0, thetae = 0, b_cgs = 0, sigma = 0, beta = 0, bnorm = 0;
        kpolaris::Vec4<Real> ucon, bcon;
        Real gcov[kpolaris::ndim][kpolaris::ndim];
        int ok = model.fluid_state(metric, state, ucon, bcon, ne, thetae, b_cgs, sigma, beta, gcov, bnorm);
        std::cout << "ok " << ok << "\n";
        std::cout << "derived ne " << ne << " thetae " << thetae << " b_cgs " << b_cgs
                  << " sigma " << sigma << " beta " << beta << " bnorm " << bnorm << "\n";
        std::cout << "ucon " << ucon[0] << ' ' << ucon[1] << ' ' << ucon[2] << ' ' << ucon[3] << "\n";
        std::cout << "bcon " << bcon[0] << ' ' << bcon[1] << ' ' << bcon[2] << ' ' << bcon[3] << "\n";
        state.k = kpolaris::Vec4<Real>(Real(0), Real(-1), Real(0), Real(0));
        state.k[0] = kpolaris::solve_future_null_k0(metric, state.x, kpolaris::Vec3<Real>(Real(-1), Real(0), Real(0)));
        state.e1 = kpolaris::Vec4<Real>(Real(0), Real(0), Real(1), Real(0));
        state.e2 = kpolaris::Vec4<Real>(Real(0), Real(0), Real(0), Real(1));
        auto coeffs = model.coefficients(metric, state, Real(0));
        std::cout << "coeffs j " << coeffs.jI << ' ' << coeffs.jQ << ' ' << coeffs.jU << ' ' << coeffs.jV
                  << " a " << coeffs.aI << ' ' << coeffs.aQ << ' ' << coeffs.aU << ' ' << coeffs.aV
                  << " r " << coeffs.rQ << ' ' << coeffs.rU << ' ' << coeffs.rV << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        Kokkos::finalize();
        return 1;
    }
    Kokkos::finalize();
    return 0;
}
