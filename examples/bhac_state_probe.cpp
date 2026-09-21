#include <iostream>

#include <Kokkos_Core.hpp>

#include "camera/camera.hpp"
#include "geometry/kerr_schild_cartesian.hpp"
#include "geometry/kerr_schild_spherical.hpp"
#include "grmhd/bhac_loader.hpp"

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    try {
        if (argc < 2) {
            std::cerr << "usage: kpolaris_bhac_state_probe dump.dat [r theta phi M_unit sigma_cut r_out]\n";
            Kokkos::finalize();
            return 2;
        }
        using Real = kpolaris::DefaultReal;
        kpolaris::BHACLoadOptions opt;
        opt.dump_path = argv[1];
        opt.M_unit = argc > 5 ? Real(std::stod(argv[5])) : opt.M_unit;
        opt.sigma_cut = argc > 6 ? Real(std::stod(argv[6])) : opt.sigma_cut;
        opt.r_out = argc > 7 ? Real(std::stod(argv[7])) : opt.r_out;
        opt.timing = 1;
        auto model = kpolaris::load_bhac_model_from_dat(opt);
        const Real r = argc > 2 ? Real(std::stod(argv[2])) : Real(10);
        const Real th = argc > 3 ? Real(std::stod(argv[3])) : Real(1.57079632679489661923);
        const Real ph = argc > 4 ? Real(std::stod(argv[4])) : Real(0);
        kpolaris::KerrSchildSphericalMetric<Real> metric(Real(1), model.spin);
        enum IntField { InBounds = 0, Meshblock = 1, FluidOk = 2, NumIntFields = 3 };
        enum RealField {
            NativeX1 = 0, NativeX2, NativeX3, Radius, Theta,
            RawPrim0, RawPrimEnd = RawPrim0 + 8,
            Rho = RawPrimEnd, Uu, Ne, Thetae, BCgs, Sigma, Beta,
            Ucon0, UconEnd = Ucon0 + 4,
            Bcon0, BconEnd = Bcon0 + 4,
            NumRealFields = BconEnd
        };
        Kokkos::View<int*> probe_ints("bhac_probe_ints", NumIntFields);
        Kokkos::View<Real*> probe_reals("bhac_probe_reals", NumRealFields);
        Kokkos::deep_copy(probe_ints, 0);
        Kokkos::deep_copy(probe_reals, Real(0));
        Kokkos::parallel_for(
            "bhac_state_probe",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
            KOKKOS_LAMBDA(const int) {
                kpolaris::TransportState<Real> state;
                state.x = kpolaris::Vec4<Real>(Real(0), r, th, ph);
                Real x1 = 0, x2 = 0, x3 = 0, rr = 0, tt = 0, cp = 1, sp = 0;
                const int in_bounds = model.bhac_data_coords_for_state(metric, state, x1, x2, x3, rr, tt, cp, sp);
                const int mb = in_bounds ? model.find_meshblock(x1, x2, x3) : -1;
                probe_ints(InBounds) = in_bounds;
                probe_ints(Meshblock) = mb;
                probe_reals(NativeX1) = x1;
                probe_reals(NativeX2) = x2;
                probe_reals(NativeX3) = x3;
                probe_reals(Radius) = rr;
                probe_reals(Theta) = tt;
                if (mb >= 0) {
                    for (int v = 0; v < 8; ++v) {
                        probe_reals(RawPrim0 + v) = model.interp_prim(mb, v, x1, x2, x3);
                    }
                }
                Real rho = 0, uu = 0, ne = 0, thetae = 0, b_cgs = 0, sigma = 0, beta = 0;
                kpolaris::Vec4<Real> ucon, bcon;
                const int ok = model.fluid_state(metric, state, rho, uu, ucon, bcon, ne, thetae, b_cgs, sigma, beta);
                probe_ints(FluidOk) = ok;
                probe_reals(Rho) = rho;
                probe_reals(Uu) = uu;
                probe_reals(Ne) = ne;
                probe_reals(Thetae) = thetae;
                probe_reals(BCgs) = b_cgs;
                probe_reals(Sigma) = sigma;
                probe_reals(Beta) = beta;
                for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                    probe_reals(Ucon0 + mu) = ucon[mu];
                    probe_reals(Bcon0 + mu) = bcon[mu];
                }
            });
        Kokkos::fence();
        auto h_ints = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), probe_ints);
        auto h_reals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), probe_reals);
        std::cout.precision(17);
        std::cout << "nblocks " << model.nblocks << " mb_n " << model.nx1 << ' ' << model.nx2 << ' ' << model.nx3 << "\n";
        std::cout << "spin " << model.spin << " gam " << model.gam << " r_in " << model.r_in << " r_out " << model.r_out << "\n";
        std::cout << "native " << h_reals(NativeX1) << ' ' << h_reals(NativeX2) << ' ' << h_reals(NativeX3)
                  << " in_bounds " << h_ints(InBounds) << " meshblock " << h_ints(Meshblock) << "\n";
        if (h_ints(Meshblock) >= 0) {
            std::cout << "raw_prim";
            for (int v = 0; v < 8; ++v) std::cout << ' ' << h_reals(RawPrim0 + v);
            std::cout << "\n";
        }
        std::cout << "ok " << h_ints(FluidOk) << " rho " << h_reals(Rho) << " uu " << h_reals(Uu) << "\n";
        std::cout << "derived ne " << h_reals(Ne) << " thetae " << h_reals(Thetae) << " b_cgs " << h_reals(BCgs)
                  << " sigma " << h_reals(Sigma) << " beta " << h_reals(Beta) << "\n";
        std::cout << "ucon " << h_reals(Ucon0) << ' ' << h_reals(Ucon0 + 1) << ' '
                  << h_reals(Ucon0 + 2) << ' ' << h_reals(Ucon0 + 3) << "\n";
        std::cout << "bcon " << h_reals(Bcon0) << ' ' << h_reals(Bcon0 + 1) << ' '
                  << h_reals(Bcon0 + 2) << ' ' << h_reals(Bcon0 + 3) << "\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        Kokkos::finalize();
        return 1;
    }
    Kokkos::finalize();
    return 0;
}
