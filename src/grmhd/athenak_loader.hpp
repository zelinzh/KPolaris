#pragma once

#include "model/time_interpolation.hpp"

#include "radiation/plasma_perturbation.hpp"

#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "common/math.hpp"
#include "common/types.hpp"
#include "common/vec.hpp"
#include "geodesic/state.hpp"
#include "model/grmhd.hpp"
#include "radiation/nonthermal_synchrotron.hpp"
#include "radiation/stokes.hpp"
#include "radiation/thermal_synchrotron.hpp"

namespace kpolaris {


template<class Real = DefaultReal>
struct AthenaKDirectRadiationModel {
    using RealView = Kokkos::View<Real*>;
    using FloatView = Kokkos::View<float*>;
    using IntView = Kokkos::View<int*>;
#if KPOLARIS_GRMHD_RANDOM_ACCESS_VIEWS
    using FloatReadView = Kokkos::View<const float*, typename FloatView::array_layout,
                                       typename FloatView::device_type,
                                       Kokkos::MemoryTraits<Kokkos::RandomAccess> >;
#else
    using FloatReadView = FloatView;
#endif

    int nblocks = 0;
    int nvar = 8;
    int nx1 = 0, nx2 = 0, nx3 = 0;
    Real spin = Real(0.9375);
    Real gam = Real(13) / Real(9);
    Real r_in = Real(1);
    Real r_out = Real(1000);
    Real freq_cgs = Real(230.0e9);
    Real mbh_solar = Real(6.2e9);
    Real M_unit = Real(1.0e26);
    Real trat_small = Real(1);
    Real trat_large = Real(40);
    Real beta_crit = Real(1);
    Real sigma_cut = Real(1);
    Real sigma_cut_high = Real(-1);
    int emission_type = 4;
    int profile_mode = 0;
    Real emission_scale = Real(1);
    Real absorption_scale = Real(1);
    Real faraday_scale = Real(1);
    Real max_pol_frac = Real(0.99);
    Real nonthermal_kappa = Real(3.5);
    int variable_kappa = 0;
    Real variable_kappa_min = Real(3.1);
    Real variable_kappa_interp_start = Real(1e20);
    Real variable_kappa_max = Real(7.0);
    Real powerlaw_p = Real(3.25);
    Real powerlaw_eta = Real(0.02);
    Real powerlaw_gamma_min = Real(1e2);
    Real powerlaw_gamma_max = Real(1e5);
    Real powerlaw_gamma_cutoff = Real(1e10);

    RealView extents;      // block-major: xmin,xmax,ymin,ymax,zmin,zmax
    RealView cell_geom;    // block-major: xmin,ymin,zmin,1/dx,1/dy,1/dz
    // Raw AthenaK primitive variables in CKS meshblock coordinates.
    // This direct path does not apply ipole's CKS->eKS/log-r primitive conversion.
    FloatView prims;       // block,var,k,j,i, matching AthenaK binary order
    FloatView derived;     // block,derived,k,j,i: ne,thetae,B_cgs,sigma,beta
    IntView levels;        // AthenaK AMR level per meshblock
    enum DerivedScalar : int { DerivedNe = 0, DerivedThetae = 1, DerivedB = 2, DerivedSigma = 3, DerivedBeta = 4, NumDerived = 5 };
    IntView index_offsets;
    IntView index_candidates;
    int index_nx = 0, index_ny = 0, index_nz = 0;
    Real index_xmin = Real(0), index_xmax = Real(0), index_dx = Real(1), index_inv_dx = Real(1);
    Real index_ymin = Real(0), index_ymax = Real(0), index_dy = Real(1), index_inv_dy = Real(1);
    Real index_zmin = Real(0), index_zmax = Real(0), index_dz = Real(1), index_inv_dz = Real(1);
    int data_coordinate_system = static_cast<int>(CoordinateSystem::CartesianKS);
    int radial_coordinate_log = 0;
    // 0: direct Cartesian KS meshblocks, 1: BHAC MKS with x2 in [0,pi].
    int native_coordinate_transform = 0;
    Real startx1 = Real(0), startx2 = Real(0), startx3 = Real(0);
    Real stopx1 = Real(0), stopx2 = Real(0), stopx3 = Real(0);
    Real hslope = Real(0.25);

    KPOLARIS_INLINE Real length_unit_cgs() const {
        const Real gnewt = Real(6.67430e-8);
        const Real msun = Real(1.98847e33);
        const Real cl = Real(2.99792458e10);
        return gnewt * mbh_solar * msun / (cl * cl);
    }
    KPOLARIS_INLINE Real rho_unit_cgs() const {
        const Real L = length_unit_cgs();
        return M_unit / max_val(L * L * L, Real(1e-300));
    }
    KPOLARIS_INLINE Real b_unit_cgs() const {
        const Real pi = Real(3.141592653589793238462643383279502884);
        const Real cl = Real(2.99792458e10);
        return cl * Kokkos::sqrt(Real(4) * pi * rho_unit_cgs());
    }
    KPOLARIS_INLINE Real dlambda_scale() const {
        return length_unit_cgs() / max_val(freq_cgs, Real(1));
    }
    KPOLARIS_INLINE Real sigma_smooth_factor(Real sigma) const {
        Real sigma_above = sigma_cut;
        if (sigma_cut_high > Real(0)) sigma_above = sigma_cut_high;
        if (sigma < sigma_cut) return Real(1);
        if (sigma >= sigma_above) return Real(0);
        const Real pi = Real(3.141592653589793238462643383279502884);
        const Real dsig = max_val(sigma_above - sigma_cut, Real(1e-300));
        return Kokkos::cos(pi / Real(2) / dsig * (sigma - sigma_cut));
    }
    KPOLARIS_INLINE size_t prim_index(int mb, int v, int i, int j, int k) const {
        return (((static_cast<size_t>(mb) * static_cast<size_t>(nvar) + static_cast<size_t>(v)) *
                 static_cast<size_t>(nx3) + static_cast<size_t>(k)) *
                static_cast<size_t>(nx2) + static_cast<size_t>(j)) *
               static_cast<size_t>(nx1) + static_cast<size_t>(i);
    }
    KPOLARIS_INLINE size_t derived_index(int mb, int v, int i, int j, int k) const {
        return (((static_cast<size_t>(mb) * static_cast<size_t>(NumDerived) + static_cast<size_t>(v)) *
                 static_cast<size_t>(nx3) + static_cast<size_t>(k)) *
                static_cast<size_t>(nx2) + static_cast<size_t>(j)) *
               static_cast<size_t>(nx1) + static_cast<size_t>(i);
    }
    KPOLARIS_INLINE Real extent(int mb, int c) const {
        return extents(static_cast<size_t>(mb) * 6u + static_cast<size_t>(c));
    }
    KPOLARIS_INLINE Real geom(int mb, int c) const {
        return cell_geom(static_cast<size_t>(mb) * 6u + static_cast<size_t>(c));
    }
    KPOLARIS_INLINE int block_contains(int mb, Real x, Real y, Real z) const {
        return x >= extent(mb, 0) && x < extent(mb, 1) &&
               y >= extent(mb, 2) && y < extent(mb, 3) &&
               z >= extent(mb, 4) && z < extent(mb, 5);
    }
    KPOLARIS_INLINE int clamp_bin(Real x, Real xmin, Real inv_dx, int n) const {
        int i = static_cast<int>(Kokkos::floor((x - xmin) * inv_dx));
        if (i < 0) i = 0;
        if (i >= n) i = n - 1;
        return i;
    }
    KPOLARIS_INLINE int find_meshblock(Real x, Real y, Real z) const {
        if (!(index_nx > 0 && x >= index_xmin && x < index_xmax &&
              y >= index_ymin && y < index_ymax && z >= index_zmin && z < index_zmax)) {
            return -1;
        }
        const int ix = clamp_bin(x, index_xmin, index_inv_dx, index_nx);
        const int iy = clamp_bin(y, index_ymin, index_inv_dy, index_ny);
        const int iz = clamp_bin(z, index_zmin, index_inv_dz, index_nz);
        const size_t bin = (static_cast<size_t>(iz) * static_cast<size_t>(index_ny) + static_cast<size_t>(iy)) *
                           static_cast<size_t>(index_nx) + static_cast<size_t>(ix);
        const int begin = index_offsets(bin);
        const int end = index_offsets(bin + 1);
        if (end == begin + 1) {
            const int mb = index_candidates(begin);
            const size_t ebase = static_cast<size_t>(mb) * 6u;
            return (x >= extents(ebase + 0u) && x < extents(ebase + 1u) &&
                    y >= extents(ebase + 2u) && y < extents(ebase + 3u) &&
                    z >= extents(ebase + 4u) && z < extents(ebase + 5u)) ? mb : -1;
        }
        int best = -1;
        int best_level = -2147483647;
        Real best_volume = Real(1e300);
        for (int n = begin; n < end; ++n) {
            const int mb = index_candidates(n);
            const size_t ebase = static_cast<size_t>(mb) * 6u;
            const Real xmin = extents(ebase + 0u);
            const Real xmax = extents(ebase + 1u);
            const Real ymin = extents(ebase + 2u);
            const Real ymax = extents(ebase + 3u);
            const Real zmin = extents(ebase + 4u);
            const Real zmax = extents(ebase + 5u);
            if (!(x >= xmin && x < xmax && y >= ymin && y < ymax && z >= zmin && z < zmax)) {
                continue;
            }
            const int lvl = levels(mb);
            const Real volume = (xmax - xmin) * (ymax - ymin) * (zmax - zmin);
            if (lvl > best_level || (lvl == best_level && volume < best_volume)) {
                best = mb;
                best_level = lvl;
                best_volume = volume;
            }
        }
        return best;
    }

    KPOLARIS_INLINE Real wrap_periodic_native_x3(Real z) const {
        if (native_coordinate_transform != 1) return z;
        const Real period = stopx3 - startx3;
        if (!(period > Real(0))) return z;
        z = startx3 + (z - startx3) - Kokkos::floor((z - startx3) / period) * period;
        if (z >= stopx3) z = startx3;
        return z;
    }

    KPOLARIS_INLINE Real interp_derived_local_clamped(int mb, int v, Real x, Real y, Real z) const {
        const Real xmin = geom(mb, 0);
        const Real ymin = geom(mb, 1);
        const Real zmin = geom(mb, 2);
        const Real inv_dx = geom(mb, 3);
        const Real inv_dy = geom(mb, 4);
        const Real inv_dz = geom(mb, 5);
        Real ii = (x - xmin) * inv_dx - Real(0.5);
        Real jj = (y - ymin) * inv_dy - Real(0.5);
        Real kk = (z - zmin) * inv_dz - Real(0.5);
        int i0 = static_cast<int>(Kokkos::floor(ii));
        int j0 = static_cast<int>(Kokkos::floor(jj));
        int k0 = static_cast<int>(Kokkos::floor(kk));
        if (i0 < 0) i0 = 0;
        if (i0 > nx1 - 2) i0 = nx1 - 2;
        if (j0 < 0) j0 = 0;
        if (j0 > nx2 - 2) j0 = nx2 - 2;
        if (k0 < 0) k0 = 0;
        if (k0 > nx3 - 2) k0 = nx3 - 2;
        Real di = ii - Real(i0);
        if (di < Real(0)) di = Real(0);
        if (di > Real(1)) di = Real(1);
        Real dj = jj - Real(j0);
        if (dj < Real(0)) dj = Real(0);
        if (dj > Real(1)) dj = Real(1);
        Real dk = kk - Real(k0);
        if (dk < Real(0)) dk = Real(0);
        if (dk > Real(1)) dk = Real(1);
        auto val = [&](int i, int j, int k) -> Real {
            return static_cast<Real>(derived(derived_index(mb, v, i, j, k)));
        };
        const Real c00 = val(i0, j0, k0) * (Real(1) - di) + val(i0 + 1, j0, k0) * di;
        const Real c10 = val(i0, j0 + 1, k0) * (Real(1) - di) + val(i0 + 1, j0 + 1, k0) * di;
        const Real c01 = val(i0, j0, k0 + 1) * (Real(1) - di) + val(i0 + 1, j0, k0 + 1) * di;
        const Real c11 = val(i0, j0 + 1, k0 + 1) * (Real(1) - di) + val(i0 + 1, j0 + 1, k0 + 1) * di;
        const Real c0 = c00 * (Real(1) - dj) + c10 * dj;
        const Real c1 = c01 * (Real(1) - dj) + c11 * dj;
        return c0 * (Real(1) - dk) + c1 * dk;
    }


    KPOLARIS_INLINE Real interp_prim_local_clamped(int mb, int v, Real x, Real y, Real z) const {
        const Real xmin = geom(mb, 0);
        const Real ymin = geom(mb, 1);
        const Real zmin = geom(mb, 2);
        const Real inv_dx = geom(mb, 3);
        const Real inv_dy = geom(mb, 4);
        const Real inv_dz = geom(mb, 5);
        Real ii = (x - xmin) * inv_dx - Real(0.5);
        Real jj = (y - ymin) * inv_dy - Real(0.5);
        Real kk = (z - zmin) * inv_dz - Real(0.5);
        int i0 = static_cast<int>(Kokkos::floor(ii));
        int j0 = static_cast<int>(Kokkos::floor(jj));
        int k0 = static_cast<int>(Kokkos::floor(kk));
        if (i0 < 0) i0 = 0;
        if (i0 > nx1 - 2) i0 = nx1 - 2;
        if (j0 < 0) j0 = 0;
        if (j0 > nx2 - 2) j0 = nx2 - 2;
        if (k0 < 0) k0 = 0;
        if (k0 > nx3 - 2) k0 = nx3 - 2;
        Real di = ii - Real(i0);
        if (di < Real(0)) di = Real(0);
        if (di > Real(1)) di = Real(1);
        Real dj = jj - Real(j0);
        if (dj < Real(0)) dj = Real(0);
        if (dj > Real(1)) dj = Real(1);
        Real dk = kk - Real(k0);
        if (dk < Real(0)) dk = Real(0);
        if (dk > Real(1)) dk = Real(1);
        auto val = [&](int i, int j, int k) -> Real {
            return static_cast<Real>(prims(prim_index(mb, v, i, j, k)));
        };
        const Real c00 = val(i0, j0, k0) * (Real(1) - di) + val(i0 + 1, j0, k0) * di;
        const Real c10 = val(i0, j0 + 1, k0) * (Real(1) - di) + val(i0 + 1, j0 + 1, k0) * di;
        const Real c01 = val(i0, j0, k0 + 1) * (Real(1) - di) + val(i0 + 1, j0, k0 + 1) * di;
        const Real c11 = val(i0, j0 + 1, k0 + 1) * (Real(1) - di) + val(i0 + 1, j0 + 1, k0 + 1) * di;
        const Real c0 = c00 * (Real(1) - dj) + c10 * dj;
        const Real c1 = c01 * (Real(1) - dj) + c11 * dj;
        return c0 * (Real(1) - dk) + c1 * dk;
    }

    KPOLARIS_INLINE Real prim_cell_or_ghost(int mb, int v, int i, int j, int k,
                                                 int neighbor_mb = -2) const {
        if (i >= 0 && i < nx1 && j >= 0 && j < nx2 && k >= 0 && k < nx3) {
            return static_cast<Real>(prims(prim_index(mb, v, i, j, k)));
        }
        const Real xmin = geom(mb, 0);
        const Real ymin = geom(mb, 1);
        const Real zmin = geom(mb, 2);
        const Real dx = Real(1) / geom(mb, 3);
        const Real dy = Real(1) / geom(mb, 4);
        const Real dz = Real(1) / geom(mb, 5);
        const Real cx = xmin + (Real(i) + Real(0.5)) * dx;
        const Real cy = ymin + (Real(j) + Real(0.5)) * dy;
        const Real cz = wrap_periodic_native_x3(zmin + (Real(k) + Real(0.5)) * dz);
        const int nmb = neighbor_mb == -2 ? find_meshblock(cx, cy, cz) : neighbor_mb;
        if (nmb >= 0 && nmb != mb) return interp_prim_local_clamped(nmb, v, cx, cy, cz);
        if (i < 0) i = 0;
        if (i >= nx1) i = nx1 - 1;
        if (j < 0) j = 0;
        if (j >= nx2) j = nx2 - 1;
        if (k < 0) k = 0;
        if (k >= nx3) k = nx3 - 1;
        return static_cast<Real>(prims(prim_index(mb, v, i, j, k)));
    }

    KPOLARIS_INLINE Real derived_cell_or_ghost(int mb, int v, int i, int j, int k,
                                                 int neighbor_mb = -2) const {
        if (i >= 0 && i < nx1 && j >= 0 && j < nx2 && k >= 0 && k < nx3) {
            return static_cast<Real>(derived(derived_index(mb, v, i, j, k)));
        }
        const Real xmin = geom(mb, 0);
        const Real ymin = geom(mb, 1);
        const Real zmin = geom(mb, 2);
        const Real dx = Real(1) / geom(mb, 3);
        const Real dy = Real(1) / geom(mb, 4);
        const Real dz = Real(1) / geom(mb, 5);
        const Real cx = xmin + (Real(i) + Real(0.5)) * dx;
        const Real cy = ymin + (Real(j) + Real(0.5)) * dy;
        const Real cz = wrap_periodic_native_x3(zmin + (Real(k) + Real(0.5)) * dz);
        const int nmb = neighbor_mb == -2 ? find_meshblock(cx, cy, cz) : neighbor_mb;
        if (nmb >= 0 && nmb != mb) return interp_derived_local_clamped(nmb, v, cx, cy, cz);
        if (i < 0) i = 0;
        if (i >= nx1) i = nx1 - 1;
        if (j < 0) j = 0;
        if (j >= nx2) j = nx2 - 1;
        if (k < 0) k = 0;
        if (k >= nx3) k = nx3 - 1;
        return static_cast<Real>(derived(derived_index(mb, v, i, j, k)));
    }


    KPOLARIS_INLINE Real interp_derived(int mb, int v, Real x, Real y, Real z,
                                        const int* neighbors = nullptr) const {
        const Real xmin = geom(mb, 0);
        const Real ymin = geom(mb, 1);
        const Real zmin = geom(mb, 2);
        const Real inv_dx = geom(mb, 3);
        const Real inv_dy = geom(mb, 4);
        const Real inv_dz = geom(mb, 5);
        const Real ii = (x - xmin) * inv_dx - Real(0.5);
        const Real jj = (y - ymin) * inv_dy - Real(0.5);
        const Real kk = (z - zmin) * inv_dz - Real(0.5);
        const int i0 = static_cast<int>(Kokkos::floor(ii));
        const int j0 = static_cast<int>(Kokkos::floor(jj));
        const int k0 = static_cast<int>(Kokkos::floor(kk));
        const Real di = ii - Real(i0);
        const Real dj = jj - Real(j0);
        const Real dk = kk - Real(k0);
        auto val = [&](int i, int j, int k) -> Real {
            const int corner = (i - i0) + 2 * (j - j0) + 4 * (k - k0);
            return derived_cell_or_ghost(mb, v, i, j, k, neighbors ? neighbors[corner] : -2);
        };
        const Real c00 = val(i0, j0, k0) * (Real(1) - di) + val(i0 + 1, j0, k0) * di;
        const Real c10 = val(i0, j0 + 1, k0) * (Real(1) - di) + val(i0 + 1, j0 + 1, k0) * di;
        const Real c01 = val(i0, j0, k0 + 1) * (Real(1) - di) + val(i0 + 1, j0, k0 + 1) * di;
        const Real c11 = val(i0, j0 + 1, k0 + 1) * (Real(1) - di) + val(i0 + 1, j0 + 1, k0 + 1) * di;
        const Real c0 = c00 * (Real(1) - dj) + c10 * dj;
        const Real c1 = c01 * (Real(1) - dj) + c11 * dj;
        return c0 * (Real(1) - dk) + c1 * dk;
    }

    KPOLARIS_INLINE Real interp_prim(int mb, int v, Real x, Real y, Real z,
                                        const int* neighbors = nullptr) const {
        const Real xmin = geom(mb, 0);
        const Real ymin = geom(mb, 1);
        const Real zmin = geom(mb, 2);
        const Real inv_dx = geom(mb, 3);
        const Real inv_dy = geom(mb, 4);
        const Real inv_dz = geom(mb, 5);
        const Real ii = (x - xmin) * inv_dx - Real(0.5);
        const Real jj = (y - ymin) * inv_dy - Real(0.5);
        const Real kk = (z - zmin) * inv_dz - Real(0.5);
        const int i0 = static_cast<int>(Kokkos::floor(ii));
        const int j0 = static_cast<int>(Kokkos::floor(jj));
        const int k0 = static_cast<int>(Kokkos::floor(kk));
        const Real di = ii - Real(i0);
        const Real dj = jj - Real(j0);
        const Real dk = kk - Real(k0);
        auto val = [&](int i, int j, int k) -> Real {
            const int corner = (i - i0) + 2 * (j - j0) + 4 * (k - k0);
            return prim_cell_or_ghost(mb, v, i, j, k, neighbors ? neighbors[corner] : -2);
        };
        const Real c00 = val(i0, j0, k0) * (Real(1) - di) + val(i0 + 1, j0, k0) * di;
        const Real c10 = val(i0, j0 + 1, k0) * (Real(1) - di) + val(i0 + 1, j0 + 1, k0) * di;
        const Real c01 = val(i0, j0, k0 + 1) * (Real(1) - di) + val(i0 + 1, j0, k0 + 1) * di;
        const Real c11 = val(i0, j0 + 1, k0 + 1) * (Real(1) - di) + val(i0 + 1, j0 + 1, k0 + 1) * di;
        const Real c0 = c00 * (Real(1) - dj) + c10 * dj;
        const Real c1 = c01 * (Real(1) - dj) + c11 * dj;
        return c0 * (Real(1) - dk) + c1 * dk;
    }


    KPOLARIS_INLINE Real trilerp_prim_interior(int mb, int v,
                                            int i0, int j0, int k0,
                                            Real di, Real dj, Real dk) const {
        auto val = [&](int i, int j, int k) -> Real {
            return static_cast<Real>(prims(prim_index(mb, v, i, j, k)));
        };
        const Real c00 = val(i0, j0, k0) * (Real(1) - di) + val(i0 + 1, j0, k0) * di;
        const Real c10 = val(i0, j0 + 1, k0) * (Real(1) - di) + val(i0 + 1, j0 + 1, k0) * di;
        const Real c01 = val(i0, j0, k0 + 1) * (Real(1) - di) + val(i0 + 1, j0, k0 + 1) * di;
        const Real c11 = val(i0, j0 + 1, k0 + 1) * (Real(1) - di) + val(i0 + 1, j0 + 1, k0 + 1) * di;
        const Real c0 = c00 * (Real(1) - dj) + c10 * dj;
        const Real c1 = c01 * (Real(1) - dj) + c11 * dj;
        return c0 * (Real(1) - dk) + c1 * dk;
    }

    KPOLARIS_INLINE Real trilerp_derived_interior(int mb, int v,
                                               int i0, int j0, int k0,
                                               Real di, Real dj, Real dk) const {
        auto val = [&](int i, int j, int k) -> Real {
            return static_cast<Real>(derived(derived_index(mb, v, i, j, k)));
        };
        const Real c00 = val(i0, j0, k0) * (Real(1) - di) + val(i0 + 1, j0, k0) * di;
        const Real c10 = val(i0, j0 + 1, k0) * (Real(1) - di) + val(i0 + 1, j0 + 1, k0) * di;
        const Real c01 = val(i0, j0, k0 + 1) * (Real(1) - di) + val(i0 + 1, j0, k0 + 1) * di;
        const Real c11 = val(i0, j0 + 1, k0 + 1) * (Real(1) - di) + val(i0 + 1, j0 + 1, k0 + 1) * di;
        const Real c0 = c00 * (Real(1) - dj) + c10 * dj;
        const Real c1 = c01 * (Real(1) - dj) + c11 * dj;
        return c0 * (Real(1) - dk) + c1 * dk;
    }

    KPOLARIS_INLINE void sample_fluid_values(int mb, Real x, Real y, Real z,
                                          Real prim_out[8],
                                          Real derived_out[NumDerived]) const {
        const Real xmin = geom(mb, 0);
        const Real ymin = geom(mb, 1);
        const Real zmin = geom(mb, 2);
        const Real inv_dx = geom(mb, 3);
        const Real inv_dy = geom(mb, 4);
        const Real inv_dz = geom(mb, 5);
        const Real ii = (x - xmin) * inv_dx - Real(0.5);
        const Real jj = (y - ymin) * inv_dy - Real(0.5);
        const Real kk = (z - zmin) * inv_dz - Real(0.5);
        const int i0 = static_cast<int>(Kokkos::floor(ii));
        const int j0 = static_cast<int>(Kokkos::floor(jj));
        const int k0 = static_cast<int>(Kokkos::floor(kk));
        const Real di = ii - Real(i0);
        const Real dj = jj - Real(j0);
        const Real dk = kk - Real(k0);

        if (i0 >= 0 && i0 + 1 < nx1 &&
            j0 >= 0 && j0 + 1 < nx2 &&
            k0 >= 0 && k0 + 1 < nx3) {
            const Real wi0 = Real(1) - di;
            const Real wj0 = Real(1) - dj;
            const Real wk0 = Real(1) - dk;
            const Real w000 = wi0 * wj0 * wk0;
            const Real w100 = di  * wj0 * wk0;
            const Real w010 = wi0 * dj  * wk0;
            const Real w110 = di  * dj  * wk0;
            const Real w001 = wi0 * wj0 * dk;
            const Real w101 = di  * wj0 * dk;
            const Real w011 = wi0 * dj  * dk;
            const Real w111 = di  * dj  * dk;
            const size_t sx = 1u;
            const size_t sy = static_cast<size_t>(nx1);
            const size_t sz = static_cast<size_t>(nx1) * static_cast<size_t>(nx2);
            const size_t var_stride = sz * static_cast<size_t>(nx3);
            const size_t cell = (static_cast<size_t>(k0) * static_cast<size_t>(nx2) +
                                 static_cast<size_t>(j0)) * static_cast<size_t>(nx1) +
                                static_cast<size_t>(i0);
            const size_t prim_block = static_cast<size_t>(mb) * static_cast<size_t>(nvar) * var_stride;
            const FloatReadView prim_values(prims);
            for (int v = 0; v < 8; ++v) {
                const size_t base = prim_block + static_cast<size_t>(v) * var_stride + cell;
                prim_out[v] =
                    w000 * static_cast<Real>(prim_values(base)) +
                    w100 * static_cast<Real>(prim_values(base + sx)) +
                    w010 * static_cast<Real>(prim_values(base + sy)) +
                    w110 * static_cast<Real>(prim_values(base + sy + sx)) +
                    w001 * static_cast<Real>(prim_values(base + sz)) +
                    w101 * static_cast<Real>(prim_values(base + sz + sx)) +
                    w011 * static_cast<Real>(prim_values(base + sz + sy)) +
                    w111 * static_cast<Real>(prim_values(base + sz + sy + sx));
            }
            const size_t derived_block = static_cast<size_t>(mb) * static_cast<size_t>(NumDerived) * var_stride;
            const FloatReadView derived_values(derived);
            for (int v = 0; v < NumDerived; ++v) {
                const size_t base = derived_block + static_cast<size_t>(v) * var_stride + cell;
                derived_out[v] =
                    w000 * static_cast<Real>(derived_values(base)) +
                    w100 * static_cast<Real>(derived_values(base + sx)) +
                    w010 * static_cast<Real>(derived_values(base + sy)) +
                    w110 * static_cast<Real>(derived_values(base + sy + sx)) +
                    w001 * static_cast<Real>(derived_values(base + sz)) +
                    w101 * static_cast<Real>(derived_values(base + sz + sx)) +
                    w011 * static_cast<Real>(derived_values(base + sz + sy)) +
                    w111 * static_cast<Real>(derived_values(base + sz + sy + sx));
            }
            return;
        }

        // The same ghost-cell centers are used by all 13 fields. Cache only
        // their block IDs (32 bytes), retaining the original interpolation order
        // and avoiding a large per-thread stencil or a rounded ghost-value cache.
        int neighbors[8];
        const Real dx = Real(1) / inv_dx;
        const Real dy = Real(1) / inv_dy;
        const Real dz = Real(1) / inv_dz;
        for (int c = 0; c < 8; ++c) {
            const int i = i0 + (c & 1);
            const int j = j0 + ((c >> 1) & 1);
            const int k = k0 + ((c >> 2) & 1);
            neighbors[c] = -1;
            if (i < 0 || i >= nx1 || j < 0 || j >= nx2 || k < 0 || k >= nx3) {
                const Real cx = xmin + (Real(i) + Real(0.5)) * dx;
                const Real cy = ymin + (Real(j) + Real(0.5)) * dy;
                const Real cz = wrap_periodic_native_x3(zmin + (Real(k) + Real(0.5)) * dz);
                neighbors[c] = find_meshblock(cx, cy, cz);
            }
        }
        for (int v = 0; v < 8; ++v) prim_out[v] = interp_prim(mb, v, x, y, z, neighbors);
        for (int v = 0; v < NumDerived; ++v) derived_out[v] = interp_derived(mb, v, x, y, z, neighbors);
    }


    KPOLARIS_INLINE Vec4<Real> lower_with_matrix(const Real g[ndim][ndim], const Vec4<Real>& v) const {
        Vec4<Real> out;
        for (int mu = 0; mu < ndim; ++mu) {
            Real sum = Real(0);
            for (int nu = 0; nu < ndim; ++nu) sum += g[mu][nu] * v[nu];
            out[mu] = sum;
        }
        return out;
    }

    template<class Metric>
    KPOLARIS_INLINE void bl_coordinates_for_metric(const Metric& metric,
                                                const Vec4<Real>& x,
                                                Real& r,
                                                Real& th,
                                                Real& cosphi,
                                                Real& sinphi) const {
        if constexpr (Metric::coordinate_system == CoordinateSystem::CartesianKS) {
            const Real xx = x[1];
            const Real yy = x[2];
            const Real zz = x[3];
            const Real a2 = metric.spin * metric.spin;
            const Real radius2 = xx * xx + yy * yy + zz * zz;
            const Real s = radius2 - a2;
            const Real discr = Kokkos::sqrt(max_val(s * s + Real(4) * a2 * zz * zz, Real(1e-300)));
            r = Kokkos::sqrt(max_val(Real(0.5) * (s + discr), Real(1e-300)));
            th = Kokkos::acos(clamp(zz / max_val(r, Real(1e-300)), Real(-1), Real(1)));
            const Real sinth = max_val(Kokkos::sin(th), Real(1e-30));
            const Real denom = max_val(sinth * (r * r + a2), Real(1e-300));
            cosphi = (r * xx + metric.spin * yy) / denom;
            sinphi = (r * yy - metric.spin * xx) / denom;
            const Real normp = Kokkos::sqrt(max_val(cosphi * cosphi + sinphi * sinphi, Real(1e-300)));
            cosphi /= normp;
            sinphi /= normp;
        } else if constexpr (Metric::coordinate_system == CoordinateSystem::SphericalKS ||
                             Metric::coordinate_system == CoordinateSystem::BoyerLindquist) {
            r = x[1];
            th = x[2];
            cosphi = Kokkos::cos(x[3]);
            sinphi = Kokkos::sin(x[3]);
        } else {
            r = Real(0);
            th = Real(0);
            cosphi = Real(1);
            sinphi = Real(0);
        }
    }

    KPOLARIS_INLINE Real bhac_theta_from_x2(Real x2) const {
        return x2 + Real(0.5) * hslope * Kokkos::sin(Real(2) * x2);
    }

    KPOLARIS_INLINE Real bhac_dtheta_dx2(Real x2) const {
        return Real(1) + hslope * Kokkos::cos(Real(2) * x2);
    }

    KPOLARIS_INLINE Real bhac_native_x2_from_theta(Real theta) const {
        const Real pi = Real(3.141592653589793238462643383279502884);
        theta = clamp(theta, Real(0), pi);
        Real x2 = theta;
        for (int it = 0; it < 8; ++it) {
            const Real f = bhac_theta_from_x2(x2) - theta;
            const Real df = bhac_dtheta_dx2(x2);
            x2 -= f / max_val(df, Real(1e-30));
            x2 = clamp(x2, Real(0), pi);
        }
        return x2;
    }

    KPOLARIS_INLINE Real normalize_phi(Real phi) const {
        const Real two_pi = Real(6.283185307179586476925286766559005768);
        phi = phi - Kokkos::floor(phi / two_pi) * two_pi;
        if (phi < Real(0)) phi += two_pi;
        return phi;
    }

    KPOLARIS_INLINE void bhac_native_metric(Real x1, Real x2, Real r, Real th,
                                         Real gcov_nat[ndim][ndim],
                                         Real gcon0_nat[ndim]) const {
        (void)x1;
        const Real cth = Kokkos::cos(th);
        const Real sth = Kokkos::sin(th);
        const Real s2 = sth * sth;
        const Real a2 = spin * spin;
        const Real rho2 = r * r + a2 * cth * cth;
        const Real f = Real(2) * r / rho2;
        const Real g_tt = Real(-1) + f;
        const Real g_tr = f;
        const Real g_tphi = -spin * r * Real(2) * s2 / rho2;
        const Real g_rr = Real(1) + f;
        const Real g_rphi = -spin * s2 * (Real(1) + f);
        const Real g_thth = rho2;
        const Real g_phiphi = s2 * (rho2 + a2 * s2 * (Real(1) + f));
        const Real hfac = bhac_dtheta_dx2(x2);
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                gcov_nat[mu][nu] = Real(0);
            }
        }
        gcov_nat[0][0] = g_tt;
        gcov_nat[0][1] = g_tr * r;
        gcov_nat[1][0] = gcov_nat[0][1];
        gcov_nat[0][3] = g_tphi;
        gcov_nat[3][0] = g_tphi;
        gcov_nat[1][1] = g_rr * r * r;
        gcov_nat[1][3] = g_rphi * r;
        gcov_nat[3][1] = gcov_nat[1][3];
        gcov_nat[2][2] = g_thth * hfac * hfac;
        gcov_nat[3][3] = g_phiphi;

        gcon0_nat[0] = -(Real(1) + f);
        gcon0_nat[1] = f / max_val(r, Real(1e-300));
        gcon0_nat[2] = Real(0);
        gcon0_nat[3] = Real(0);
    }

    KPOLARIS_INLINE Vec4<Real> bhac_native_to_spherical_contravariant(Real x2, Real r,
                                                                    const Vec4<Real>& vnat) const {
        return Vec4<Real>(vnat[0],
                          r * vnat[1],
                          bhac_dtheta_dx2(x2) * vnat[2],
                          vnat[3]);
    }

    KPOLARIS_INLINE Vec4<Real> bhac_native_to_cartesian_contravariant(Real x2, Real r, Real th,
                                                                    Real cp, Real sp,
                                                                    const Vec4<Real>& vnat) const {
        const Vec4<Real> vsph = bhac_native_to_spherical_contravariant(x2, r, vnat);
        const Real sinth = Kokkos::sin(th);
        const Real costh = Kokkos::cos(th);
        const Real dx_dr = cp * sinth;
        const Real dx_dth = r * cp * costh - spin * sp * costh;
        const Real dx_dphi = -r * sp * sinth - spin * cp * sinth;
        const Real dy_dr = sp * sinth;
        const Real dy_dth = r * sp * costh + spin * cp * costh;
        const Real dy_dphi = r * cp * sinth - spin * sp * sinth;
        const Real dz_dr = costh;
        const Real dz_dth = -r * sinth;
        return Vec4<Real>(vsph[0],
                          dx_dr * vsph[1] + dx_dth * vsph[2] + dx_dphi * vsph[3],
                          dy_dr * vsph[1] + dy_dth * vsph[2] + dy_dphi * vsph[3],
                          dz_dr * vsph[1] + dz_dth * vsph[2]);
    }

    template<class Metric>
    KPOLARIS_INLINE Vec4<Real> lower_vector(const Metric& metric,
                                         const Vec4<Real>& x,
                                         const Vec4<Real>& v) const {
        Vec4<Real> out;
        for (int mu = 0; mu < ndim; ++mu) {
            Real sum = Real(0);
            for (int nu = 0; nu < ndim; ++nu) {
                sum += metric.gcov(mu, nu, x) * v[nu];
            }
            out[mu] = sum;
        }
        return out;
    }

    template<class Metric>
    KPOLARIS_INLINE Real screen_inner_product(const Metric& metric,
                                           const Vec4<Real>& x,
                                           const Vec4<Real>& u,
                                           const Vec4<Real>& k,
                                           const Vec4<Real>& a,
                                           const Vec4<Real>& b) const {
        const Real uk = metric.dot(x, u, k);
        if (abs_val(uk) <= Real(1e-300)) {
            return metric.dot(x, a, b);
        }
        const Vec4<Real> eK = k * (Real(-1) / uk) - u;
        return metric.dot(x, a, b) +
               metric.dot(x, u, a) * metric.dot(x, u, b) -
               metric.dot(x, eK, a) * metric.dot(x, eK, b);
    }

    template<class Metric>
    KPOLARIS_INLINE int bhac_data_coords_for_state(const Metric& metric,
                                                const TransportState<Real>& state,
                                                Real& x1,
                                                Real& x2,
                                                Real& x3,
                                                Real& r,
                                                Real& th,
                                                Real& cp,
                                                Real& sp) const {
        if constexpr (Metric::coordinate_system == CoordinateSystem::SphericalKS ||
                      Metric::coordinate_system == CoordinateSystem::BoyerLindquist) {
            r = state.x[1];
            th = state.x[2];
            x3 = normalize_phi(state.x[3]);
            cp = Real(1);
            sp = Real(0);
        } else {
            bl_coordinates_for_metric(metric, state.x, r, th, cp, sp);
            x3 = normalize_phi(Kokkos::atan2(sp, cp));
        }
        if (!(r > Real(0) && th >= Real(0))) {
            return 0;
        }
        if (!(r >= r_in && r < r_out)) {
            return 0;
        }
        x1 = Kokkos::log(max_val(r, Real(1e-300)));
        x2 = bhac_native_x2_from_theta(clamp(th, Real(0), Real(3.141592653589793238462643383279502884)));
        const Real eps = Real(1e-10);
        if (x2 <= startx2) x2 = startx2 + eps;
        if (x2 >= stopx2) x2 = stopx2 - eps;
        return x1 >= startx1 && x1 < stopx1 &&
               x2 >= startx2 && x2 < stopx2 &&
               x3 >= startx3 && x3 < stopx3;
    }

    static constexpr bool supports_fluid_time_interpolation = true;

    bool can_interpolate_fluid_with(const AthenaKDirectRadiationModel& other) const {
        return data_coordinate_system == other.data_coordinate_system &&
            native_coordinate_transform == other.native_coordinate_transform &&
            spin == other.spin && r_in == other.r_in && r_out == other.r_out &&
            startx1 == other.startx1 && startx2 == other.startx2 && startx3 == other.startx3 &&
            stopx1 == other.stopx1 && stopx2 == other.stopx2 && stopx3 == other.stopx3 &&
            index_xmin == other.index_xmin && index_xmax == other.index_xmax &&
            index_ymin == other.index_ymin && index_ymax == other.index_ymax &&
            index_zmin == other.index_zmin && index_zmax == other.index_zmax &&
            hslope == other.hslope;
    }

    KPOLARIS_INLINE bool sample_temporal_fluid_values(int mb, Real x, Real y, Real z,
        Real* prim, Real* derived, const AthenaKDirectRadiationModel* upper,
        Real fraction) const {
        sample_fluid_values(mb, x, y, z, prim, derived);
        if (upper) {
            const int mb_upper = upper->find_meshblock(x, y, z);
            if (mb_upper < 0) return false;
            Real other_prim[8], other_derived[NumDerived];
            upper->sample_fluid_values(mb_upper, x, y, z, other_prim, other_derived);
            for (int n = 0; n < 8; ++n)
                prim[n] = interpolate_time_value(prim[n], other_prim[n], fraction);
            for (int n = 0; n < NumDerived; ++n)
                derived[n] = interpolate_time_value(derived[n], other_derived[n], fraction);
        }
        return true;
    }

    template<class Metric>
    KPOLARIS_INLINE int bhac_fluid_state_with_covectors(const Metric& metric,
                                                     const TransportState<Real>& state,
                                                     Vec4<Real>& ucon,
                                                     Vec4<Real>& bcon,
                                                     Vec4<Real>& ucov,
                                                     Vec4<Real>& bcov,
                                                     Real& ne_cgs,
                                                     Real& thetae,
                                                     Real& b_cgs,
                                                     Real& sigma,
                                                     Real& beta,
                                                     Real g[ndim][ndim],
                                                     Real& bnorm_geom,
                                                     const AthenaKDirectRadiationModel* time_upper = nullptr,
                                                     Real time_fraction = Real(0)) const {
        if constexpr (Metric::coordinate_system != CoordinateSystem::CartesianKS &&
                      Metric::coordinate_system != CoordinateSystem::SphericalKS) {
            return 0;
        } else {
            Real x1 = Real(0), x2 = Real(0), x3 = Real(0);
            Real r = Real(0), th = Real(0), cp = Real(1), sp = Real(0);
            if (!bhac_data_coords_for_state(metric, state, x1, x2, x3, r, th, cp, sp)) {
                return 0;
            }
            if (!(r >= r_in && r < r_out)) return 0;
            const int mb = find_meshblock(x1, x2, x3);
            if (mb < 0) return 0;

            Real prim_vals[8];
            Real derived_vals[NumDerived];
            if (!sample_temporal_fluid_values(mb, x1, x2, x3, prim_vals, derived_vals,
                                               time_upper, time_fraction)) return 0;
            const Real rho = prim_vals[0];
            const Real uu = prim_vals[4];
            if (!(rho > Real(0) && uu > Real(0))) return 0;

            Real gcov_nat[ndim][ndim];
            Real gcon0_nat[ndim];
            bhac_native_metric(x1, x2, r, th, gcov_nat, gcon0_nat);

            const Vec4<Real> vnat(Real(0), prim_vals[1], prim_vals[2], prim_vals[3]);
            Real spatial_norm = Real(0);
            for (int i = 1; i < ndim; ++i) {
                for (int j = 1; j < ndim; ++j) {
                    spatial_norm += gcov_nat[i][j] * vnat[i] * vnat[j];
                }
            }
            const Real vfac = Kokkos::sqrt(max_val(-Real(1) / gcon0_nat[0] *
                                                   (Real(1) + abs_val(spatial_norm)), Real(0)));
            Vec4<Real> ucon_nat;
            ucon_nat[0] = -vfac * gcon0_nat[0];
            for (int i = 1; i < ndim; ++i) {
                ucon_nat[i] = vnat[i] - vfac * gcon0_nat[i];
            }
            const Vec4<Real> ucov_nat = lower_with_matrix(gcov_nat, ucon_nat);

            const Vec4<Real> Bnat(Real(0), prim_vals[5], prim_vals[6], prim_vals[7]);
            Real udotB = Real(0);
            for (int i = 1; i < ndim; ++i) {
                udotB += ucov_nat[i] * Bnat[i];
            }
            Vec4<Real> bcon_nat;
            bcon_nat[0] = udotB;
            const Real inv_u0 = Real(1) / max_val(ucon_nat[0], Real(1e-300));
            for (int i = 1; i < ndim; ++i) {
                bcon_nat[i] = (Bnat[i] + ucon_nat[i] * udotB) * inv_u0;
            }
            const Vec4<Real> bcov_nat = lower_with_matrix(gcov_nat, bcon_nat);
            Real bsq = Real(0);
            for (int mu = 0; mu < ndim; ++mu) bsq += bcon_nat[mu] * bcov_nat[mu];
            bsq = max_val(abs_val(bsq), Real(1e-40));
            bnorm_geom = Kokkos::sqrt(bsq);

            if constexpr (Metric::coordinate_system == CoordinateSystem::CartesianKS) {
                ucon = bhac_native_to_cartesian_contravariant(x2, r, th, cp, sp, ucon_nat);
                bcon = bhac_native_to_cartesian_contravariant(x2, r, th, cp, sp, bcon_nat);
            } else {
                ucon = bhac_native_to_spherical_contravariant(x2, r, ucon_nat);
                bcon = bhac_native_to_spherical_contravariant(x2, r, bcon_nat);
            }
            metric.gcov_matrix(state.x, g);
            ucov = lower_with_matrix(g, ucon);
            bcov = lower_with_matrix(g, bcon);

            ne_cgs = derived_vals[DerivedNe];
            thetae = derived_vals[DerivedThetae];
            b_cgs = derived_vals[DerivedB];
            sigma = max_val(derived_vals[DerivedSigma], Real(1e-300));
            beta = max_val(derived_vals[DerivedBeta], Real(1e-300));
            ne_cgs *= sigma_smooth_factor(sigma);
            return ne_cgs > Real(0) && thetae > Real(0) && b_cgs > Real(0);
        }
    }

    template<class Metric>
    KPOLARIS_INLINE int fluid_state_with_covectors(const Metric& metric,
                                                const TransportState<Real>& state,
                                                Vec4<Real>& ucon,
                                                Vec4<Real>& bcon,
                                                Vec4<Real>& ucov,
                                                Vec4<Real>& bcov,
                                                Real& ne_cgs,
                                                Real& thetae,
                                                Real& b_cgs,
                                                Real& sigma,
                                                Real& beta,
                                                Real g[ndim][ndim],
                                                Real& bnorm_geom,
                                                     const AthenaKDirectRadiationModel* time_upper = nullptr,
                                                     Real time_fraction = Real(0)) const {
        if (data_coordinate_system == static_cast<int>(CoordinateSystem::MKS) &&
            native_coordinate_transform == 1) {
            return bhac_fluid_state_with_covectors(metric, state, ucon, bcon, ucov, bcov,
                                                   ne_cgs, thetae, b_cgs, sigma, beta,
                                                   g, bnorm_geom, time_upper, time_fraction);
        }
        if constexpr (Metric::coordinate_system != CoordinateSystem::CartesianKS) {
            return 0;
        } else {
            const Real x = state.x[1], y = state.x[2], z = state.x[3];
            const typename Metric::Work w = metric.build_work(state.x);
            if (!(w.r >= r_in && w.r < r_out)) return 0;
            const int mb = find_meshblock(x, y, z);
            if (mb < 0) return 0;
            Real prim_vals[8];
            Real derived_vals[NumDerived];
            if (!sample_temporal_fluid_values(mb, x, y, z, prim_vals, derived_vals,
                                               time_upper, time_fraction)) return 0;
            const Real rho = prim_vals[0];
            const Real uu = prim_vals[4];
            if (!(rho > Real(0) && uu > Real(0))) return 0;
            const Real U1 = prim_vals[1];
            const Real U2 = prim_vals[2];
            const Real U3 = prim_vals[3];
            const Real B1 = prim_vals[5];
            const Real B2 = prim_vals[6];
            const Real B3 = prim_vals[7];
            for (int mu = 0; mu < ndim; ++mu) {
                for (int nu = 0; nu < ndim; ++nu) {
                    const Real eta_mu_nu = (mu == nu ? (mu == 0 ? Real(-1) : Real(1)) : Real(0));
                    g[mu][nu] = eta_mu_nu + w.amp * w.l[mu] * w.l[nu];
                }
            }
            const Real gcon00 = Real(-1) - w.amp;
            const Real alpha = Real(1) / Kokkos::sqrt(-gcon00);
            Real udotu = Real(0);
            const Real U[3] = {U1, U2, U3};
            for (int a = 1; a < ndim; ++a) {
                for (int b = 1; b < ndim; ++b) udotu += g[a][b] * U[a - 1] * U[b - 1];
            }
            const Real gamma = Kokkos::sqrt(Real(1) + abs_val(udotu));
            ucon[0] = gamma / alpha;
            ucon[1] = U1 - gamma * alpha * g[0][1];
            ucon[2] = U2 - gamma * alpha * g[0][2];
            ucon[3] = U3 - gamma * alpha * g[0][3];
            ucov = lower_with_matrix(g, ucon);
            bcon[0] = B1 * ucov[1] + B2 * ucov[2] + B3 * ucov[3];
            bcon[1] = (B1 + ucon[1] * bcon[0]) / max_val(ucon[0], Real(1e-300));
            bcon[2] = (B2 + ucon[2] * bcon[0]) / max_val(ucon[0], Real(1e-300));
            bcon[3] = (B3 + ucon[3] * bcon[0]) / max_val(ucon[0], Real(1e-300));
            bcov = lower_with_matrix(g, bcon);
            Real bsq = Real(0);
            for (int mu = 0; mu < ndim; ++mu) bsq += bcon[mu] * bcov[mu];
            bsq = max_val(abs_val(bsq), Real(1e-40));
            bnorm_geom = Kokkos::sqrt(bsq);
            ne_cgs = derived_vals[DerivedNe];
            thetae = derived_vals[DerivedThetae];
            b_cgs = derived_vals[DerivedB];
            sigma = max_val(derived_vals[DerivedSigma], Real(1e-300));
            beta = max_val(derived_vals[DerivedBeta], Real(1e-300));
            ne_cgs *= sigma_smooth_factor(sigma);
            return ne_cgs > Real(0) && thetae > Real(0) && b_cgs > Real(0);
        }
    }

    template<class Metric>
    KPOLARIS_INLINE int fluid_state(const Metric& metric,
                                 const TransportState<Real>& state,
                                 Vec4<Real>& ucon,
                                 Vec4<Real>& bcon,
                                 Real& ne_cgs,
                                 Real& thetae,
                                 Real& b_cgs,
                                 Real& sigma,
                                 Real& beta,
                                 Real g[ndim][ndim],
                                 Real& bnorm_geom,
                                                     const AthenaKDirectRadiationModel* time_upper = nullptr,
                                                     Real time_fraction = Real(0)) const {
        Vec4<Real> ucov, bcov;
        return fluid_state_with_covectors(metric, state, ucon, bcon, ucov, bcov,
                                          ne_cgs, thetae, b_cgs, sigma, beta,
                                          g, bnorm_geom, time_upper, time_fraction);
    }

    template<class Metric>
    KPOLARIS_INLINE int fluid_state(const Metric& metric,
                                 const TransportState<Real>& state,
                                 Real& rho,
                                 Real& uu,
                                 Vec4<Real>& ucon,
                                 Vec4<Real>& bcon,
                                 Real& ne_cgs,
                                 Real& thetae,
                                 Real& b_cgs,
                                 Real& sigma,
                                 Real& beta,
                                 const AthenaKDirectRadiationModel* time_upper = nullptr,
                                 Real time_fraction = Real(0)) const {
        if (data_coordinate_system == static_cast<int>(CoordinateSystem::MKS) &&
            native_coordinate_transform == 1) {
            rho = uu = ne_cgs = thetae = b_cgs = sigma = beta = Real(0);
            Real x1 = Real(0), x2 = Real(0), x3 = Real(0);
            Real r = Real(0), th = Real(0), cp = Real(1), sp = Real(0);
            if (!bhac_data_coords_for_state(metric, state, x1, x2, x3, r, th, cp, sp)) {
                return 0;
            }
            const int mb = find_meshblock(x1, x2, x3);
            if (mb < 0) return 0;
            rho = interp_prim(mb, 0, x1, x2, x3);
            uu = interp_prim(mb, 4, x1, x2, x3);
            if (time_upper) {
                const int mb_upper = time_upper->find_meshblock(x1, x2, x3);
                if (mb_upper < 0) return 0;
                rho = interpolate_time_value(rho, time_upper->interp_prim(mb_upper, 0, x1, x2, x3), time_fraction);
                uu = interpolate_time_value(uu, time_upper->interp_prim(mb_upper, 4, x1, x2, x3), time_fraction);
            }
            Real gcov[ndim][ndim];
            Real bnorm_geom = Real(0);
            return fluid_state(metric, state, ucon, bcon, ne_cgs, thetae,
                               b_cgs, sigma, beta, gcov, bnorm_geom, time_upper, time_fraction);
        }
        if constexpr (Metric::coordinate_system != CoordinateSystem::CartesianKS) {
            rho = uu = ne_cgs = thetae = b_cgs = sigma = beta = Real(0);
            return 0;
        } else {
            const Real x = state.x[1], y = state.x[2], z = state.x[3];
            const int mb = find_meshblock(x, y, z);
            if (mb < 0) {
                rho = uu = ne_cgs = thetae = b_cgs = sigma = beta = Real(0);
                return 0;
            }
            rho = interp_prim(mb, 0, x, y, z);
            uu = interp_prim(mb, 4, x, y, z);
            if (time_upper) {
                const int mb_upper = time_upper->find_meshblock(x, y, z);
                if (mb_upper < 0) return 0;
                rho = interpolate_time_value(rho, time_upper->interp_prim(mb_upper, 0, x, y, z), time_fraction);
                uu = interpolate_time_value(uu, time_upper->interp_prim(mb_upper, 4, x, y, z), time_fraction);
            }
            Real gcov[ndim][ndim];
            Real bnorm_geom = Real(0);
            return fluid_state(metric, state, ucon, bcon, ne_cgs, thetae,
                               b_cgs, sigma, beta, gcov, bnorm_geom, time_upper, time_fraction);
        }
    }

    template<class Metric>
    KPOLARIS_INLINE int temporal_fluid_state(const AthenaKDirectRadiationModel& upper, Real fraction,
        const Metric& metric, const TransportState<Real>& state,
        Real& rho, Real& uu, Vec4<Real>& u, Vec4<Real>& b,
        Real& ne, Real& theta, Real& B, Real& sigma, Real& beta) const {
        return fluid_state(metric, state, rho, uu, u, b, ne, theta, B, sigma, beta,
                           &upper, fraction);
    }

    template<class Metric>
    KPOLARIS_INLINE TransferCoeffs<Real> coefficients(const Metric& metric,
                                                   const TransportState<Real>& state,
                                                   Real,
                                                   const PlasmaPerturbation<Real>& perturbation = {},
                                                   const AthenaKDirectRadiationModel* time_upper = nullptr,
                                                   Real time_fraction = Real(0)) const {
        TransferCoeffs<Real> coeffs;
        if (profile_mode == 1) return coeffs;
        Vec4<Real> ucon, bcon, ucov, bcov;
        Real ne_cgs = Real(0), thetae = Real(0), b_cgs = Real(0), sigma = Real(0), beta = Real(0);
        Real bnorm_geom = Real(0);
        Real gcov[ndim][ndim];
        if (!fluid_state_with_covectors(metric, state, ucon, bcon, ucov, bcov,
                                        ne_cgs, thetae, b_cgs, sigma, beta,
                                        gcov, bnorm_geom, time_upper, time_fraction)) return coeffs;
        if (profile_mode == 2) return coeffs;
        perturbation.apply(ne_cgs, thetae, b_cgs, sigma, beta);
        const Real inv_bnorm = Real(1) / max_val(bnorm_geom, Real(1e-150));
        Real uk = Real(0), kdotb = Real(0), ub = Real(0), ue1 = Real(0), ue2 = Real(0), ke1 = Real(0), ke2 = Real(0);
        Real e1b_cov = Real(0), e2b_cov = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            Real kcov_mu = Real(0);
            for (int nu_i = 0; nu_i < ndim; ++nu_i) {
                kcov_mu += gcov[mu][nu_i] * state.k[nu_i];
            }
            uk += ucov[mu] * state.k[mu];
            kdotb += state.k[mu] * bcov[mu];
            ub += ucov[mu] * bcon[mu];
            ue1 += ucov[mu] * state.e1[mu];
            ue2 += ucov[mu] * state.e2[mu];
            ke1 += kcov_mu * state.e1[mu];
            ke2 += kcov_mu * state.e2[mu];
            e1b_cov += bcov[mu] * state.e1[mu];
            e2b_cov += bcov[mu] * state.e2[mu];
        }
        const Real nu_scale = max_val(abs_val(-uk), Real(1e-8));
        const Real nu = max_val(freq_cgs * nu_scale, Real(1));
        const Real cos_theta = clamp(kdotb / max_val(nu_scale * bnorm_geom, Real(1e-30)), Real(-1), Real(1));
        const Real sin_theta = max_val(Kokkos::sqrt(max_val(Real(0), Real(1) - cos_theta * cos_theta)), Real(1e-4));
        ThermalSynchrotronParams<Real> thermal_params;
#if KPOLARIS_IHARM_COMPILED_EMISSION_TYPE > 0
        thermal_params.fit = KPOLARIS_IHARM_COMPILED_EMISSION_TYPE;
#else
        thermal_params.fit = emission_type;
#endif
        thermal_params.max_pol_frac = max_pol_frac;
        thermal_params.min_sin_theta = Real(1e-4);
        thermal_params.emission_scale = emission_scale;
        thermal_params.absorption_scale = absorption_scale;
        thermal_params.faraday_scale = faraday_scale;
        LocalThermalSynchrotronState<Real> thermal_state;
        thermal_state.nu = nu; thermal_state.ne = ne_cgs; thermal_state.thetae = thetae;
        thermal_state.b_cgs = b_cgs; thermal_state.theta = Real(0);
        thermal_state.sin_theta = sin_theta; thermal_state.cos_theta = cos_theta;
        TransferCoeffs<Real> magnetic_coeffs;
#if KPOLARIS_IHARM_COMPILED_EMISSION_TYPE == KPOLARIS_THERMAL_SYNCHROTRON_DEXTER
        magnetic_coeffs = thermal_synchrotron_magnetic_basis_coefficients_fit<ThermalSynchrotronDexter>(thermal_state, thermal_params);
#elif KPOLARIS_IHARM_COMPILED_EMISSION_TYPE == KPOLARIS_THERMAL_SYNCHROTRON_PANDYA
        magnetic_coeffs = thermal_synchrotron_magnetic_basis_coefficients_fit<ThermalSynchrotronPandya>(thermal_state, thermal_params);
#else
        if (emission_type == SynchrotronKappa || emission_type == SynchrotronPowerLaw) {
            NonthermalSynchrotronParams<Real> nonthermal_params;
            nonthermal_params.distribution = emission_type;
            nonthermal_params.max_pol_frac_emission = max_pol_frac;
            nonthermal_params.max_pol_frac_absorption = max_pol_frac;
            nonthermal_params.min_sin_theta = Real(1e-4);
            nonthermal_params.emission_scale = emission_scale;
            nonthermal_params.absorption_scale = absorption_scale;
            nonthermal_params.faraday_scale = faraday_scale;
            nonthermal_params.kappa = variable_kappa ? variable_kappa_from_sigma_beta(sigma, beta, nonthermal_params) : nonthermal_kappa;
            if (nonthermal_params.kappa < variable_kappa_min) nonthermal_params.kappa = variable_kappa_min;
            nonthermal_params.kappa_interp_begin = min_val(variable_kappa_interp_start, variable_kappa_max);
            nonthermal_params.kappa_interp_end = variable_kappa_max;
            nonthermal_params.power_law_p = powerlaw_p;
            nonthermal_params.power_law_eta = powerlaw_eta;
            nonthermal_params.power_law_gamma_min = powerlaw_gamma_min;
            nonthermal_params.power_law_gamma_max = powerlaw_gamma_max;
            nonthermal_params.power_law_gamma_cutoff = powerlaw_gamma_cutoff;
            LocalNonthermalSynchrotronState<Real> nonthermal_state;
            nonthermal_state.nu = nu; nonthermal_state.ne = ne_cgs; nonthermal_state.thetae = thetae;
            nonthermal_state.b_cgs = b_cgs; nonthermal_state.sin_theta = sin_theta; nonthermal_state.cos_theta = cos_theta;
            magnetic_coeffs = nonthermal_synchrotron_magnetic_basis_coefficients(nonthermal_state, nonthermal_params);
        } else {
            magnetic_coeffs = thermal_synchrotron_magnetic_basis_coefficients(thermal_state, thermal_params);
        }
#endif
        if (profile_mode == 4) return coeffs;
        if (profile_mode == 5) {
            const Real tiny = Real(1e-300);
            coeffs.jI = (magnetic_coeffs.jI + magnetic_coeffs.jQ + magnetic_coeffs.jV +
                         magnetic_coeffs.aI + magnetic_coeffs.aQ + magnetic_coeffs.aV +
                         magnetic_coeffs.rQ + magnetic_coeffs.rV) * tiny;
            return coeffs;
        }
        if (profile_mode == 3) {
            coeffs.jI = magnetic_coeffs.jI;
            coeffs.jQ = magnetic_coeffs.jQ;
            coeffs.jV = magnetic_coeffs.jV;
            coeffs.aI = magnetic_coeffs.aI;
            coeffs.aQ = magnetic_coeffs.aQ;
            coeffs.aV = magnetic_coeffs.aV;
            coeffs.rQ = magnetic_coeffs.rQ;
            coeffs.rV = magnetic_coeffs.rV;
            return coeffs;
        }
        const Real ubunit = ub * inv_bnorm;
        const Real kbunit = kdotb * inv_bnorm;
        Real b1 = e1b_cov * inv_bnorm;
        Real b2 = e2b_cov * inv_bnorm;
        if (abs_val(uk) > Real(1e-300)) {
            const Real minus_inv_uk = -Real(1) / uk;
            const Real eKb = minus_inv_uk * kbunit - ubunit;
            b1 += ue1 * ubunit - (minus_inv_uk * ke1 - ue1) * eKb;
            b2 += ue2 * ubunit - (minus_inv_uk * ke2 - ue2) * eKb;
        }
        const Real bproj2 = b1 * b1 + b2 * b2;
        coeffs.jI = magnetic_coeffs.jI;
        coeffs.jV = magnetic_coeffs.jV;
        coeffs.aI = magnetic_coeffs.aI;
        coeffs.aV = magnetic_coeffs.aV;
        coeffs.rV = magnetic_coeffs.rV;
        if (bproj2 > Real(1e-60)) {
            const Real inv_bproj2 = Real(1) / bproj2;
            // Synchrotron coefficients put the magnetic field along e2,
            // as in GRMHDGridRadiationModel and the ipole plasma tetrad.
            // An e1-aligned rotation reverses Q/U (90 degrees in EVPA),
            // including absorption and Faraday conversion.
            const Real cos2 = (b2 * b2 - b1 * b1) * inv_bproj2;
            const Real sin2 = -Real(2) * b1 * b2 * inv_bproj2;
            coeffs.jQ = magnetic_coeffs.jQ * cos2;
            coeffs.jU = magnetic_coeffs.jQ * sin2;
            coeffs.aQ = magnetic_coeffs.aQ * cos2;
            coeffs.aU = magnetic_coeffs.aQ * sin2;
            coeffs.rQ = magnetic_coeffs.rQ * cos2;
            coeffs.rU = magnetic_coeffs.rQ * sin2;
        }
        return coeffs;
    }

};

struct AthenaKLoadOptions {
    std::string dump_path;
    DefaultReal freq = DefaultReal(230.0e9);
    DefaultReal M_unit = DefaultReal(1.0e26);
    DefaultReal mbh_solar = DefaultReal(6.2e9);
    DefaultReal trat_small = DefaultReal(1);
    DefaultReal trat_large = DefaultReal(40);
    DefaultReal beta_crit = DefaultReal(1);
    DefaultReal gamma = DefaultReal(-1);
    DefaultReal sigma_cut = DefaultReal(1);
    DefaultReal sigma_cut_high = DefaultReal(-1);
    int emission_type = 4;
    int profile_mode = 0;
    DefaultReal nonthermal_kappa = DefaultReal(3.5);
    int variable_kappa = 0;
    DefaultReal variable_kappa_min = DefaultReal(3.1);
    DefaultReal variable_kappa_interp_start = DefaultReal(1e20);
    DefaultReal variable_kappa_max = DefaultReal(7.0);
    DefaultReal powerlaw_p = DefaultReal(3.25);
    DefaultReal powerlaw_eta = DefaultReal(0.02);
    DefaultReal powerlaw_gamma_min = DefaultReal(1e2);
    DefaultReal powerlaw_gamma_max = DefaultReal(1e5);
    DefaultReal powerlaw_gamma_cutoff = DefaultReal(1e10);
    int resample_n1 = 192;
    int resample_n2 = 96;
    int resample_n3 = 96;
    DefaultReal resample_r_in = DefaultReal(-1);
    DefaultReal resample_r_out = DefaultReal(1000);
    int timing = 0;
};

struct AthenaKStagedDump {
    int nblocks = 0;
    int nvar = 0;
    int nx1 = 0;
    int nx2 = 0;
    int nx3 = 0;
    DefaultReal spin = DefaultReal(0.9375);
    DefaultReal gamma = DefaultReal(13.0 / 9.0);
    std::vector<DefaultReal> extents;
    std::vector<DefaultReal> cell_geom;
    std::vector<float> prims;
    std::vector<int> levels;
    std::vector<int> index_offsets;
    std::vector<int> index_candidates;
    int index_nx = 0;
    int index_ny = 0;
    int index_nz = 0;
    DefaultReal index_xmin = DefaultReal(0);
    DefaultReal index_xmax = DefaultReal(0);
    DefaultReal index_ymin = DefaultReal(0);
    DefaultReal index_ymax = DefaultReal(0);
    DefaultReal index_zmin = DefaultReal(0);
    DefaultReal index_zmax = DefaultReal(0);
    DefaultReal index_dx = DefaultReal(1);
    DefaultReal index_dy = DefaultReal(1);
    DefaultReal index_dz = DefaultReal(1);
};

AthenaKDirectRadiationModel<DefaultReal> load_athenak_direct_model_from_binary(const AthenaKLoadOptions& opt);
AthenaKStagedDump read_athenak_staged_dump(const AthenaKLoadOptions& opt);
AthenaKDirectRadiationModel<DefaultReal> materialize_athenak_direct_model_from_staged(const AthenaKStagedDump& staged,
                                                                                      const AthenaKLoadOptions& opt);
DefaultReal read_athenak_dump_time(const std::string& dump_path);

} // namespace kpolaris
