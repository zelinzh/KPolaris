#include <cmath>
#include <iostream>
#include <stdexcept>
#include "grmhd/athenak_loader.hpp"
#include "geometry/kerr_schild_cartesian.hpp"

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    try {
        using Real = kpolaris::DefaultReal;
        using Model = kpolaris::AthenaKDirectRadiationModel<Real>;
        Model model;
        model.nblocks = 9;
        model.nx1 = model.nx2 = model.nx3 = 4;
        model.extents = Model::RealView("extents", 54);
        model.cell_geom = Model::RealView("geometry", 54);
        model.levels = Model::IntView("levels", 9);
        model.prims = Model::FloatView("primitives", 9 * 8 * 64);
        model.derived = Model::FloatView("derived", 9 * Model::NumDerived * 64);
        model.index_offsets = Model::IntView("offsets", 2);
        model.index_candidates = Model::IntView("candidates", 9);
        model.index_nx = model.index_ny = model.index_nz = 1;
        model.index_xmin = model.index_ymin = model.index_zmin = -2;
        model.index_xmax = model.index_ymax = model.index_zmax = 2;
        model.index_inv_dx = model.index_inv_dy = model.index_inv_dz = Real(0.25);
        model.startx3 = -2;
        model.stopx3 = 2;
        const Model m = model;
        Kokkos::parallel_for("sampling_fixture", 9, KOKKOS_LAMBDA(int b) {
            for (int a = 0; a < 3; ++a) {
                const Real lo = b == 8 ? Real(-1) : Real(-2 + 2 * ((b >> a) & 1));
                m.extents(b * 6 + 2 * a) = lo;
                m.extents(b * 6 + 2 * a + 1) = lo + 2;
                m.cell_geom(b * 6 + a) = lo;
                m.cell_geom(b * 6 + a + 3) = 2;
            }
            m.levels(b) = b == 8 ? 2 : 0;
            m.index_candidates(b) = b;
            if (b == 0) { m.index_offsets(0) = 0; m.index_offsets(1) = 9; }
            for (int v = 0; v < 8; ++v)
                for (int k = 0; k < 4; ++k)
                    for (int j = 0; j < 4; ++j)
                        for (int i = 0; i < 4; ++i)
                            m.prims(m.prim_index(b, v, i, j, k)) =
                                float(1 + b + v * 3 + i * i + 2 * j * j + 3 * k);
            for (int v = 0; v < Model::NumDerived; ++v)
                for (int k = 0; k < 4; ++k)
                    for (int j = 0; j < 4; ++j)
                        for (int i = 0; i < 4; ++i)
                            m.derived(m.derived_index(b, v, i, j, k)) =
                                float(2 + 2 * b + v * v + i + j * k);
        });
        Kokkos::fence();
        for (int periodic = 0; periodic < 2; ++periodic) {
            model.native_coordinate_transform = periodic;
            const Model device_model = model;
            int errors = 0;
            // Dense points touch interiors, faces, edges, corners, periodic seams,
            // missing neighbors, and overlapping AMR levels. Legacy scalar calls
            // remain an independent oracle without the cached neighbor IDs.
            Kokkos::parallel_reduce("sampling_equivalence", 31 * 31 * 31,
                KOKKOS_LAMBDA(int p, int& bad) {
                    const Real x = Real(-1.999) + Real(p % 31) * Real(3.998 / 30);
                    const Real y = Real(-1.999) + Real((p / 31) % 31) * Real(3.998 / 30);
                    const Real z = Real(-1.999) + Real(p / (31 * 31)) * Real(3.998 / 30);
                    const int mb = device_model.find_meshblock(x, y, z);
                    if (mb < 0) { ++bad; return; }
                    if (x > -1 && x < 1 && y > -1 && y < 1 && z > -1 && z < 1 && mb != 8) ++bad;
                    Real prim[8], derived[Model::NumDerived];
                    device_model.sample_fluid_values(mb, x, y, z, prim, derived);
                    const Real tol = sizeof(Real) == 4 ? Real(2e-6) : Real(2e-14);
                    for (int v = 0; v < 8; ++v) {
                        const Real ref = device_model.interp_prim(mb, v, x, y, z);
                        if (!(kpolaris::abs_val(prim[v] - ref) <= tol * (Real(1) + kpolaris::abs_val(ref)))) ++bad;
                    }
                    for (int v = 0; v < Model::NumDerived; ++v) {
                        const Real ref = device_model.interp_derived(mb, v, x, y, z);
                        if (!(kpolaris::abs_val(derived[v] - ref) <= tol * (Real(1) + kpolaris::abs_val(ref)))) ++bad;
                    }
                }, errors);
            if (errors) throw std::runtime_error("cached AMR/periodic sampling differs from scalar oracle");
        }
        // A uniform, static plasma in flat spacetime gives an independent
        // screen-orientation check.  The field projects along e2.  Rotating
        // the screen by 0, 45, 90 degrees gives (Q,U)=(q,0),(0,-q),(-q,0).
        // Exercise emissivity, dichroism and Faraday conversion together.
        model.native_coordinate_transform = 0;
        model.r_in = Real(0.1);
        model.spin = Real(0);
        const Model uniform = model;
        Kokkos::parallel_for("uniform_polarization_fixture", 9, KOKKOS_LAMBDA(int b) {
            for (int k = 0; k < 4; ++k)
                for (int j = 0; j < 4; ++j)
                    for (int i = 0; i < 4; ++i) {
                        const Real prim[8] = {1, 0, 0, 0, 1, 0, 1, Real(0.5)};
                        const Real scalar[5] = {Real(1e6), 10, 30, Real(0.1), 1};
                        for (int v = 0; v < 8; ++v)
                            uniform.prims(uniform.prim_index(b, v, i, j, k)) = float(prim[v]);
                        for (int v = 0; v < 5; ++v)
                            uniform.derived(uniform.derived_index(b, v, i, j, k)) = float(scalar[v]);
                    }
        });
        int polarization_errors = 0;
        Kokkos::parallel_reduce("magnetic_screen_orientation", 3,
            KOKKOS_LAMBDA(int angle, int& bad) {
                const kpolaris::KerrSchildInMetric<Real> flat(Real(0), Real(0));
                kpolaris::TransportState<Real> state;
                state.x = kpolaris::Vec4<Real>(0, Real(1.5), Real(0.25), Real(0.25));
                state.k = kpolaris::Vec4<Real>(1, 0, 0, 1);
                const Real h = Kokkos::sqrt(Real(0.5));
                const Real cs[3] = {1, h, 0}, sn[3] = {0, h, 1};
                state.e1 = kpolaris::Vec4<Real>(0, cs[angle], sn[angle], 0);
                state.e2 = kpolaris::Vec4<Real>(0, -sn[angle], cs[angle], 0);
                Model magnetic = uniform;
                magnetic.profile_mode = 3;  // Unrotated magnetic-basis coefficients.
                const auto ref = magnetic.coefficients(flat, state, Real(0));
                const auto got = uniform.coefficients(flat, state, Real(0));
                if (!(ref.jI > 0 && kpolaris::abs_val(ref.jQ) > 0 &&
                      kpolaris::abs_val(ref.aQ) > 0 && kpolaris::abs_val(ref.rQ) > 0)) ++bad;
                const Real q[3] = {1, 0, -1}, u[3] = {0, -1, 0};
                const Real expected[11] = {ref.jI, ref.jQ*q[angle], ref.jQ*u[angle], ref.jV,
                    ref.aI, ref.aQ*q[angle], ref.aQ*u[angle], ref.aV,
                    ref.rQ*q[angle], ref.rQ*u[angle], ref.rV};
                const Real actual[11] = {got.jI, got.jQ, got.jU, got.jV,
                    got.aI, got.aQ, got.aU, got.aV, got.rQ, got.rU, got.rV};
                const Real scale[11] = {ref.jI, ref.jQ, ref.jQ, ref.jV,
                    ref.aI, ref.aQ, ref.aQ, ref.aV, ref.rQ, ref.rQ, ref.rV};
                const Real tol = sizeof(Real) == 4 ? Real(2e-5) : Real(2e-12);
                for (int v = 0; v < 11; ++v)
                    if (!(kpolaris::abs_val(actual[v] - expected[v]) <=
                          tol * kpolaris::max_val(kpolaris::abs_val(scale[v]), Real(1e-300)))) ++bad;
            }, polarization_errors);
        if (polarization_errors)
            throw std::runtime_error("meshblock synchrotron coefficients use the wrong screen orientation");
        std::cout << "Meshblock magnetic screen orientation: 0/45/90 degrees passed\n";
        std::cout << "AthenaK/BHAC/H-AMR shared sampling: 59,582 positions x 13 fields passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
