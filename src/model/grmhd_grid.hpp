#pragma once

#include "radiation/plasma_perturbation.hpp"
#include "model/time_interpolation.hpp"

#include "common/math.hpp"
#include "geodesic/state.hpp"
#include "grmhd/grid_mapping.hpp"
#include "radiation/stokes.hpp"
#include "radiation/thermal_synchrotron.hpp"
#include "radiation/nonthermal_synchrotron.hpp"

#ifndef KPOLARIS_IHARM_COMPILED_EMISSION_TYPE
#define KPOLARIS_IHARM_COMPILED_EMISSION_TYPE 0
#endif

#ifndef KPOLARIS_GRMHD_REQUIRE_DERIVED_SCALARS
#define KPOLARIS_GRMHD_REQUIRE_DERIVED_SCALARS 0
#endif

#ifndef KPOLARIS_GRMHD_FLUID_STATE_FLOAT
#define KPOLARIS_GRMHD_FLUID_STATE_FLOAT 0
#endif

#ifndef KPOLARIS_GRMHD_RANDOM_ACCESS_VIEWS
#define KPOLARIS_GRMHD_RANDOM_ACCESS_VIEWS 0
#endif

namespace kpolaris {

template<class Real = DefaultReal>
struct GRMHDGridRadiationModel {
#if KPOLARIS_GRMHD_FLUID_STATE_FLOAT
    using FluidStateStorageReal = float;
#else
    using FluidStateStorageReal = Real;
#endif
    using RealView = Kokkos::View<Real*>;
    using FluidStateView = Kokkos::View<FluidStateStorageReal*>;
#if KPOLARIS_GRMHD_RANDOM_ACCESS_VIEWS
    using RealReadView = Kokkos::View<const Real*, typename RealView::array_layout,
                                      typename RealView::device_type,
                                      Kokkos::MemoryTraits<Kokkos::RandomAccess> >;
    using FluidStateReadView = Kokkos::View<const FluidStateStorageReal*, typename FluidStateView::array_layout,
                                            typename FluidStateView::device_type,
                                            Kokkos::MemoryTraits<Kokkos::RandomAccess> >;
#else
    using RealReadView = RealView;
    using FluidStateReadView = FluidStateView;
#endif

    int n1 = 0, n2 = 0, n3 = 0, nprim = 8;
    Real spin = Real(0.5);
    Real gam = Real(4) / Real(3);
    Real startx1 = Real(0), startx2 = Real(0), startx3 = Real(0);
    Real dx1 = Real(1), dx2 = Real(1), dx3 = Real(1);
    Real r_in = Real(1), r_out = Real(1000);
    Real hslope = Real(0.3);
    Real mks_smooth = Real(0.5);
    Real poly_alpha = Real(14);
    Real poly_xt = Real(0.82);
    Real poly_norm = Real(1);

    Real freq_cgs = Real(230.0e9);
    Real mbh_solar = Real(6.2e9);
    Real M_unit = Real(3.0e25);
    Real trat_small = Real(1);
    Real trat_large = Real(20);
    Real beta_crit = Real(1);
    Real sigma_cut = Real(1);
    Real sigma_cut_high = Real(-1);
    Real cooling_dynamical_times = Real(1e-20);
    Real ne_factor = Real(1);
    int emission_type = 4;
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

    RealView prims;
    RealView derived_scalars;
    FluidStateView fluid_states;
    int has_derived_scalars = 1;
    int precomputed_fluid_state = 0;
    int data_coordinate_system = static_cast<int>(CoordinateSystem::FMKS);
    int radial_coordinate_log = 1;
    int native_coordinate_transform = 0;  // 0=FMKS/MMKS, 1=MKS

    enum DerivedScalar : int {
        DerivedNe = 0,
        DerivedThetae = 1,
        DerivedB = 2,
        DerivedSigma = 3,
        DerivedBeta = 4,
        NumDerivedScalars = 5
    };

    enum FluidStateScalar : int {
        FluidU0 = 0,
        FluidU1 = 1,
        FluidU2 = 2,
        FluidU3 = 3,
        FluidB0 = 4,
        FluidB1 = 5,
        FluidB2 = 6,
        FluidB3 = 7,
        FluidNe = 8,
        FluidThetae = 9,
        FluidBCgs = 10,
        FluidSigma = 11,
        FluidBeta = 12,
        NumFluidStateScalars = 13
    };

    static constexpr bool supports_fluid_time_interpolation = true;

    // Different resolutions are allowed; the native coordinate map and domain
    // must describe the same spacetime. Cached four-vectors lack primitives.
    bool can_interpolate_fluid_with(const GRMHDGridRadiationModel& other) const {
        return !precomputed_fluid_state && !other.precomputed_fluid_state &&
            data_coordinate_system == other.data_coordinate_system &&
            native_coordinate_transform == other.native_coordinate_transform &&
            radial_coordinate_log == other.radial_coordinate_log &&
            spin == other.spin && r_in == other.r_in && r_out == other.r_out &&
            hslope == other.hslope && mks_smooth == other.mks_smooth &&
            poly_alpha == other.poly_alpha && poly_xt == other.poly_xt &&
            poly_norm == other.poly_norm && startx1 == other.startx1 &&
            has_derived_scalars == other.has_derived_scalars;
    }

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

    KPOLARIS_INLINE FMKSGridMapping<Real> grid_mapping() const {
        return make_fmks_grid_mapping(startx1, hslope, mks_smooth,
                                      poly_alpha, poly_xt, poly_norm);
    }

    KPOLARIS_INLINE MKSGridMapping<Real> mks_grid_mapping() const {
        MKSGridMapping<Real> mapping;
        mapping.hslope = hslope;
        return mapping;
    }

    KPOLARIS_INLINE size_t prim_index(int v, int i, int j, int k) const {
        return (((static_cast<size_t>(v) * static_cast<size_t>(n1) + static_cast<size_t>(i)) *
                 static_cast<size_t>(n2) + static_cast<size_t>(j)) *
                static_cast<size_t>(n3) + static_cast<size_t>(k));
    }

    KPOLARIS_INLINE Real prim(int v, int i, int j, int k) const {
        return prims(prim_index(v, i, j, k));
    }

    KPOLARIS_INLINE size_t derived_index(int v, int i, int j, int k) const {
        return (((static_cast<size_t>(v) * static_cast<size_t>(n1) + static_cast<size_t>(i)) *
                 static_cast<size_t>(n2) + static_cast<size_t>(j)) *
                static_cast<size_t>(n3) + static_cast<size_t>(k));
    }

    KPOLARIS_INLINE Real derived_scalar(int v, int i, int j, int k) const {
        return derived_scalars(derived_index(v, i, j, k));
    }

    KPOLARIS_INLINE size_t fluid_index(int v, int i, int j, int k) const {
        return (((static_cast<size_t>(v) * static_cast<size_t>(n1) + static_cast<size_t>(i)) *
                 static_cast<size_t>(n2) + static_cast<size_t>(j)) *
                static_cast<size_t>(n3) + static_cast<size_t>(k));
    }

    KPOLARIS_INLINE size_t cell_count() const {
        return static_cast<size_t>(n1) * static_cast<size_t>(n2) * static_cast<size_t>(n3);
    }

    KPOLARIS_INLINE Real fluid_scalar(int v, int i, int j, int k) const {
        return static_cast<Real>(fluid_states(fluid_index(v, i, j, k)));
    }

    KPOLARIS_INLINE Real fmks_theta_from_x2(Real x1, Real x2) const {
        if (native_coordinate_transform == 1) {
            return mks_grid_mapping().theta_from_x2(x1, x2);
        }
        return grid_mapping().theta_from_x2(x1, x2);
    }

    KPOLARIS_INLINE Real dfmks_theta_dx1(Real x1, Real x2) const {
        if (native_coordinate_transform == 1) {
            return mks_grid_mapping().dtheta_dx1(x1, x2);
        }
        return grid_mapping().dtheta_dx1(x1, x2);
    }

    KPOLARIS_INLINE Real dfmks_theta_dx2(Real x1, Real x2) const {
        if (native_coordinate_transform == 1) {
            return mks_grid_mapping().dtheta_dx2(x1, x2);
        }
        return grid_mapping().dtheta_dx2(x1, x2);
    }

    KPOLARIS_INLINE Real native_x2_from_theta(Real x1, Real theta) const {
        if (native_coordinate_transform == 1) {
            return mks_grid_mapping().native_x2_from_theta(x1, theta);
        }
        return grid_mapping().native_x2_from_theta(x1, theta);
    }

    template<class Metric>
    KPOLARIS_INLINE void bl_coordinates_for_metric(const Metric& metric,
                                                const Vec4<Real>& x,
                                                Real& r, Real& th,
                                                Real& cosphi, Real& sinphi) const {
        if constexpr (Metric::coordinate_system == CoordinateSystem::FMKS) {
            r = Kokkos::exp(x[1]);
            th = fmks_theta_from_x2(x[1], x[2]);
            cosphi = Kokkos::cos(x[3]);
            sinphi = Kokkos::sin(x[3]);
        } else if constexpr (Metric::coordinate_system != CoordinateSystem::CartesianKS) {
            r = x[1];
            th = x[2];
            cosphi = Kokkos::cos(x[3]);
            sinphi = Kokkos::sin(x[3]);
        } else {
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
        }
    }

    KPOLARIS_INLINE int native_indices_from_native_coords(Real x1, Real x2, Real phi,
                                                       int& i0, int& j0, int& k0,
                                                       Real& di, Real& dj, Real& dk) const {
        if (!(x1 >= startx1 && x1 < startx1 + Real(n1) * dx1 &&
              x2 >= startx2 && x2 < startx2 + Real(n2) * dx2)) {
            return 0;
        }
        const Real two_pi = Real(6.283185307179586476925286766559005768);
        phi = phi - Kokkos::floor(phi / two_pi) * two_pi;
        Real ii = (x1 - startx1) / dx1 - Real(0.5);
        Real jj = (x2 - startx2) / dx2 - Real(0.5);
        Real kk = (phi - startx3) / dx3 - Real(0.5);
        i0 = static_cast<int>(Kokkos::floor(ii));
        j0 = static_cast<int>(Kokkos::floor(jj));
        k0 = static_cast<int>(Kokkos::floor(kk));
        if (i0 < -1) i0 = -1;
        if (j0 < -1) j0 = -1;
        if (k0 < -1) k0 = -1;
        if (i0 >= n1) i0 = n1 - 1;
        if (j0 >= n2) j0 = n2 - 1;
        if (k0 >= n3) k0 = n3 - 1;
        di = clamp(ii - Real(i0), Real(0), Real(1));
        dj = clamp(jj - Real(j0), Real(0), Real(1));
        dk = clamp(kk - Real(k0), Real(0), Real(1));
        return 1;
    }

    KPOLARIS_INLINE int native_indices(Real r, Real th, Real phi,
                                    int& i0, int& j0, int& k0,
                                    Real& di, Real& dj, Real& dk) const {
        if (!(r >= r_in && r < r_out)) {
            return 0;
        }
        const Real x1 = Kokkos::log(max_val(r, Real(1e-300)));
        const Real x2 = native_x2_from_theta(x1, th);
        return native_indices_from_native_coords(x1, x2, phi, i0, j0, k0, di, dj, dk);
    }

    KPOLARIS_INLINE void map_ghost_index(int& i, int& j, int& k) const {
        if (i < 0) i = 0;
        if (i >= n1) i = n1 - 1;
        if (j < 0) {
            j = 0;
            k += n3 / 2;
        } else if (j >= n2) {
            j = n2 - 1;
            k += n3 / 2;
        }
        while (k < 0) k += n3;
        while (k >= n3) k -= n3;
    }

    KPOLARIS_INLINE Real prim_ghost(int v, int i, int j, int k) const {
        map_ghost_index(i, j, k);
        return prim(v, i, j, k);
    }

    KPOLARIS_INLINE Real derived_ghost(int v, int i, int j, int k) const {
        map_ghost_index(i, j, k);
        return derived_scalar(v, i, j, k);
    }

    KPOLARIS_INLINE Real trilinear(Real c000, Real c100, Real c010, Real c110,
                                 Real c001, Real c101, Real c011, Real c111,
                                 Real di, Real dj, Real dk) const {
        const Real c00 = c000 * (Real(1) - di) + c100 * di;
        const Real c10 = c010 * (Real(1) - di) + c110 * di;
        const Real c01 = c001 * (Real(1) - di) + c101 * di;
        const Real c11 = c011 * (Real(1) - di) + c111 * di;
        const Real c0 = c00 * (Real(1) - dj) + c10 * dj;
        const Real c1 = c01 * (Real(1) - dj) + c11 * dj;
        return c0 * (Real(1) - dk) + c1 * dk;
    }

    KPOLARIS_INLINE Real interp_prim(int v, int i0, int j0, int k0, Real di, Real dj, Real dk) const {
        const int i1 = i0 + 1;
        const int j1 = j0 + 1;
        const int k1 = k0 + 1;
        return trilinear(prim_ghost(v, i0, j0, k0), prim_ghost(v, i1, j0, k0),
                         prim_ghost(v, i0, j1, k0), prim_ghost(v, i1, j1, k0),
                         prim_ghost(v, i0, j0, k1), prim_ghost(v, i1, j0, k1),
                         prim_ghost(v, i0, j1, k1), prim_ghost(v, i1, j1, k1),
                         di, dj, dk);
    }

    KPOLARIS_INLINE Real interp_derived(int v, int i0, int j0, int k0, Real di, Real dj, Real dk) const {
        const int i1 = i0 + 1;
        const int j1 = j0 + 1;
        const int k1 = k0 + 1;
        return trilinear(derived_ghost(v, i0, j0, k0), derived_ghost(v, i1, j0, k0),
                         derived_ghost(v, i0, j1, k0), derived_ghost(v, i1, j1, k0),
                         derived_ghost(v, i0, j0, k1), derived_ghost(v, i1, j0, k1),
                         derived_ghost(v, i0, j1, k1), derived_ghost(v, i1, j1, k1),
                         di, dj, dk);
    }

    struct InterpStencil {
        size_t cell[8];
        Real weight[8];
    };

    KPOLARIS_INLINE size_t mapped_cell_index(int i, int j, int k) const {
        map_ghost_index(i, j, k);
        return (static_cast<size_t>(i) * static_cast<size_t>(n2) +
                static_cast<size_t>(j)) * static_cast<size_t>(n3) +
               static_cast<size_t>(k);
    }

    KPOLARIS_INLINE InterpStencil make_interp_stencil(int i0, int j0, int k0,
                                                   Real di, Real dj, Real dk) const {
        InterpStencil st;
        const int i1 = i0 + 1;
        const int j1 = j0 + 1;
        const int k1 = k0 + 1;
        const Real wi0 = Real(1) - di;
        const Real wi1 = di;
        const Real wj0 = Real(1) - dj;
        const Real wj1 = dj;
        const Real wk0 = Real(1) - dk;
        const Real wk1 = dk;
        st.cell[0] = mapped_cell_index(i0, j0, k0);
        st.cell[1] = mapped_cell_index(i1, j0, k0);
        st.cell[2] = mapped_cell_index(i0, j1, k0);
        st.cell[3] = mapped_cell_index(i1, j1, k0);
        st.cell[4] = mapped_cell_index(i0, j0, k1);
        st.cell[5] = mapped_cell_index(i1, j0, k1);
        st.cell[6] = mapped_cell_index(i0, j1, k1);
        st.cell[7] = mapped_cell_index(i1, j1, k1);
        st.weight[0] = wi0 * wj0 * wk0;
        st.weight[1] = wi1 * wj0 * wk0;
        st.weight[2] = wi0 * wj1 * wk0;
        st.weight[3] = wi1 * wj1 * wk0;
        st.weight[4] = wi0 * wj0 * wk1;
        st.weight[5] = wi1 * wj0 * wk1;
        st.weight[6] = wi0 * wj1 * wk1;
        st.weight[7] = wi1 * wj1 * wk1;
        return st;
    }

    KPOLARIS_INLINE Real interp_prim_stencil(int v, const InterpStencil& st) const {
        const RealReadView values(prims);
        const size_t base = static_cast<size_t>(v) * static_cast<size_t>(n1) *
                            static_cast<size_t>(n2) * static_cast<size_t>(n3);
        Real out = Real(0);
        for (int n = 0; n < 8; ++n) {
            out += st.weight[n] * static_cast<Real>(values(base + st.cell[n]));
        }
        return out;
    }

    KPOLARIS_INLINE Real interp_derived_stencil(int v, const InterpStencil& st) const {
        const RealReadView values(derived_scalars);
        const size_t base = static_cast<size_t>(v) * static_cast<size_t>(n1) *
                            static_cast<size_t>(n2) * static_cast<size_t>(n3);
        Real out = Real(0);
        for (int n = 0; n < 8; ++n) {
            out += st.weight[n] * static_cast<Real>(values(base + st.cell[n]));
        }
        return out;
    }

    KPOLARIS_INLINE Real interp_fluid_stencil(int v, const InterpStencil& st) const {
        const FluidStateReadView values(fluid_states);
        const size_t base = static_cast<size_t>(v) * static_cast<size_t>(n1) *
                            static_cast<size_t>(n2) * static_cast<size_t>(n3);
        Real out = Real(0);
        for (int n = 0; n < 8; ++n) {
            out += st.weight[n] * static_cast<Real>(values(base + st.cell[n]));
        }
        return out;
    }

    KPOLARIS_INLINE void interp_precomputed_fluid_state_stencil(const InterpStencil& st,
                                                             Vec4<Real>& ucon,
                                                             Vec4<Real>& bcon,
                                                             Real& ne_cgs,
                                                             Real& thetae,
                                                             Real& b_cgs,
                                                             Real& sigma,
                                                             Real& beta) const {
        const FluidStateReadView values(fluid_states);
        const size_t ngrid = cell_count();
        const size_t c0 = st.cell[0];
        const size_t c1 = st.cell[1];
        const size_t c2 = st.cell[2];
        const size_t c3 = st.cell[3];
        const size_t c4 = st.cell[4];
        const size_t c5 = st.cell[5];
        const size_t c6 = st.cell[6];
        const size_t c7 = st.cell[7];
        const Real w0 = st.weight[0];
        const Real w1 = st.weight[1];
        const Real w2 = st.weight[2];
        const Real w3 = st.weight[3];
        const Real w4 = st.weight[4];
        const Real w5 = st.weight[5];
        const Real w6 = st.weight[6];
        const Real w7 = st.weight[7];
        auto interp_scalar = [&](int v) -> Real {
            const size_t base = static_cast<size_t>(v) * ngrid;
            return w0 * static_cast<Real>(values(base + c0)) +
                   w1 * static_cast<Real>(values(base + c1)) +
                   w2 * static_cast<Real>(values(base + c2)) +
                   w3 * static_cast<Real>(values(base + c3)) +
                   w4 * static_cast<Real>(values(base + c4)) +
                   w5 * static_cast<Real>(values(base + c5)) +
                   w6 * static_cast<Real>(values(base + c6)) +
                   w7 * static_cast<Real>(values(base + c7));
        };
        ucon[0] = interp_scalar(FluidU0);
        ucon[1] = interp_scalar(FluidU1);
        ucon[2] = interp_scalar(FluidU2);
        ucon[3] = interp_scalar(FluidU3);
        bcon[0] = interp_scalar(FluidB0);
        bcon[1] = interp_scalar(FluidB1);
        bcon[2] = interp_scalar(FluidB2);
        bcon[3] = interp_scalar(FluidB3);
        ne_cgs = interp_scalar(FluidNe);
        thetae = interp_scalar(FluidThetae);
        b_cgs = interp_scalar(FluidBCgs);
        sigma = interp_scalar(FluidSigma);
        beta = interp_scalar(FluidBeta);
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

    KPOLARIS_INLINE Real dot_with_gcov(const Real gcov[ndim][ndim],
                                    const Vec4<Real>& u,
                                    const Vec4<Real>& v) const {
        Real out = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                out += gcov[mu][nu] * u[mu] * v[nu];
            }
        }
        return out;
    }

    KPOLARIS_INLINE Vec4<Real> lower_with_gcov(const Real gcov[ndim][ndim],
                                            const Vec4<Real>& v) const {
        Vec4<Real> out;
        for (int mu = 0; mu < ndim; ++mu) {
            Real sum = Real(0);
            for (int nu = 0; nu < ndim; ++nu) {
                sum += gcov[mu][nu] * v[nu];
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

    KPOLARIS_INLINE Vec3<Real> native_spatial_to_cartesian(Real r, Real th, Real cp, Real sp,
                                                        Real v1, Real v2, Real v3) const {
        const Real x1 = Kokkos::log(max_val(r, Real(1e-300)));
        const Real x2 = native_x2_from_theta(x1, th);
        const Real dr = r * v1;
        const Real dth = dfmks_theta_dx1(x1, x2) * v1 + dfmks_theta_dx2(x1, x2) * v2;
        const Real dphi = v3;
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
        return Vec3<Real>(dx_dr * dr + dx_dth * dth + dx_dphi * dphi,
                          dy_dr * dr + dy_dth * dth + dy_dphi * dphi,
                          dz_dr * dr + dz_dth * dth);
    }

    KPOLARIS_INLINE void ks_spherical_gcov(Real r, Real th, Real g[ndim][ndim]) const {
        const Real cth = Kokkos::cos(th);
        const Real sth = Kokkos::sin(th);
        const Real s2 = sth * sth;
        const Real rho2 = r * r + spin * spin * cth * cth;
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                g[mu][nu] = Real(0);
            }
        }
        g[0][0] = Real(-1) + Real(2) * r / rho2;
        g[0][1] = Real(2) * r / rho2;
        g[0][3] = -Real(2) * spin * r * s2 / rho2;
        g[1][0] = g[0][1];
        g[1][1] = Real(1) + Real(2) * r / rho2;
        g[1][3] = -spin * s2 * (Real(1) + Real(2) * r / rho2);
        g[2][2] = rho2;
        g[3][0] = g[0][3];
        g[3][1] = g[1][3];
        g[3][3] = s2 * (rho2 + spin * spin * s2 * (Real(1) + Real(2) * r / rho2));
    }

    KPOLARIS_INLINE void ks_spherical_gcon(Real r, Real th, Real g[ndim][ndim]) const {
        const Real sth = Kokkos::sin(th);
        const Real cth = Kokkos::cos(th);
        const Real s2 = max_val(sth * sth, Real(1e-300));
        const Real a2 = spin * spin;
        const Real rho2 = r * r + a2 * cth * cth;
        const Real delta = r * r - Real(2) * r + a2;
        const Real f = Real(2) * r / rho2;
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                g[mu][nu] = Real(0);
            }
        }
        g[0][0] = -(Real(1) + f);
        g[0][1] = f;
        g[1][0] = g[0][1];
        g[1][1] = delta / rho2;
        g[1][3] = spin / rho2;
        g[2][2] = Real(1) / rho2;
        g[3][1] = g[1][3];
        g[3][3] = Real(1) / (rho2 * s2);
    }

    KPOLARIS_INLINE void invert_4x4(const Real a[ndim][ndim], Real inv[ndim][ndim]) const {
        Real m[ndim][2 * ndim];
        for (int i = 0; i < ndim; ++i) {
            for (int j = 0; j < ndim; ++j) {
                m[i][j] = a[i][j];
                m[i][j + ndim] = (i == j) ? Real(1) : Real(0);
            }
        }
        for (int col = 0; col < ndim; ++col) {
            Real pivot = m[col][col];
            if (abs_val(pivot) < Real(1e-300)) {
                pivot = pivot < Real(0) ? Real(-1e-300) : Real(1e-300);
            }
            const Real inv_pivot = Real(1) / pivot;
            for (int j = 0; j < 2 * ndim; ++j) {
                m[col][j] *= inv_pivot;
            }
            for (int row = 0; row < ndim; ++row) {
                if (row == col) continue;
                const Real factor = m[row][col];
                for (int j = 0; j < 2 * ndim; ++j) {
                    m[row][j] -= factor * m[col][j];
                }
            }
        }
        for (int i = 0; i < ndim; ++i) {
            for (int j = 0; j < ndim; ++j) {
                inv[i][j] = m[i][j + ndim];
            }
        }
    }

    KPOLARIS_INLINE void native_to_ks_spherical_jacobian(Real x1, Real x2, Real r,
                                                       Real jac[ndim][ndim]) const {
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                jac[mu][nu] = (mu == nu) ? Real(1) : Real(0);
            }
        }
        jac[1][1] = r;
        jac[2][1] = dfmks_theta_dx1(x1, x2);
        jac[2][2] = dfmks_theta_dx2(x1, x2);
    }

    KPOLARIS_INLINE void native_metric(Real x1, Real x2, Real r, Real th,
                                    Real gcov_nat[ndim][ndim],
                                    Real gcon_nat[ndim][ndim]) const {
        Real gks[ndim][ndim];
        Real jac[ndim][ndim];
        ks_spherical_gcov(r, th, gks);
        native_to_ks_spherical_jacobian(x1, x2, r, jac);
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                Real sum = Real(0);
                for (int a = 0; a < ndim; ++a) {
                    for (int b = 0; b < ndim; ++b) {
                        sum += gks[a][b] * jac[a][mu] * jac[b][nu];
                    }
                }
                gcov_nat[mu][nu] = sum;
            }
        }
        Real gks_con[ndim][ndim];
        ks_spherical_gcon(r, th, gks_con);
        const Real dth1 = dfmks_theta_dx1(x1, x2);
        const Real dth2 = dfmks_theta_dx2(x1, x2);
        const Real inv_r = Real(1) / max_val(r, Real(1e-300));
        const Real inv_dth2 = Real(1) / max_val(dth2, Real(1e-300));
        Real inv_jac[ndim][ndim];
        for (int mu = 0; mu < ndim; ++mu) {
            for (int a = 0; a < ndim; ++a) {
                inv_jac[mu][a] = Real(0);
            }
        }
        inv_jac[0][0] = Real(1);
        inv_jac[1][1] = inv_r;
        inv_jac[2][1] = -dth1 * inv_r * inv_dth2;
        inv_jac[2][2] = inv_dth2;
        inv_jac[3][3] = Real(1);
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                Real sum = Real(0);
                for (int a = 0; a < ndim; ++a) {
                    for (int b = 0; b < ndim; ++b) {
                        sum += inv_jac[mu][a] * inv_jac[nu][b] * gks_con[a][b];
                    }
                }
                gcon_nat[mu][nu] = sum;
            }
        }
    }

    KPOLARIS_INLINE void native_fluid_metric(Real x1, Real x2, Real r, Real th,
                                          Real gcov_nat[ndim][ndim],
                                          Real gcon0_nat[ndim]) const {
        const Real cth = Kokkos::cos(th);
        const Real sth = Kokkos::sin(th);
        const Real s2 = sth * sth;
        const Real a2 = spin * spin;
        const Real rho2 = r * r + a2 * cth * cth;
        const Real f = Real(2) * r / rho2;
        const Real dth1 = dfmks_theta_dx1(x1, x2);
        const Real dth2 = dfmks_theta_dx2(x1, x2);

        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                gcov_nat[mu][nu] = Real(0);
            }
        }

        const Real g00 = Real(-1) + f;
        const Real g01 = f;
        const Real g03 = -spin * f * s2;
        const Real g11 = Real(1) + f;
        const Real g13 = -spin * s2 * (Real(1) + f);
        const Real g22 = rho2;
        const Real g33 = s2 * (rho2 + a2 * s2 * (Real(1) + f));

        gcov_nat[0][0] = g00;
        gcov_nat[0][1] = g01 * r;
        gcov_nat[1][0] = gcov_nat[0][1];
        gcov_nat[0][3] = g03;
        gcov_nat[3][0] = g03;
        gcov_nat[1][1] = g11 * r * r + g22 * dth1 * dth1;
        gcov_nat[1][2] = g22 * dth1 * dth2;
        gcov_nat[2][1] = gcov_nat[1][2];
        gcov_nat[1][3] = g13 * r;
        gcov_nat[3][1] = gcov_nat[1][3];
        gcov_nat[2][2] = g22 * dth2 * dth2;
        gcov_nat[3][3] = g33;

        const Real inv_r = Real(1) / max_val(r, Real(1e-300));
        const Real inv_dth2 = Real(1) / max_val(dth2, Real(1e-300));
        gcon0_nat[0] = -(Real(1) + f);
        gcon0_nat[1] = f * inv_r;
        gcon0_nat[2] = -f * dth1 * inv_r * inv_dth2;
        gcon0_nat[3] = Real(0);
    }

    KPOLARIS_INLINE Vec4<Real> lower_native(const Real gcov_nat[ndim][ndim],
                                         const Vec4<Real>& v) const {
        Vec4<Real> out;
        for (int mu = 0; mu < ndim; ++mu) {
            Real sum = Real(0);
            for (int nu = 0; nu < ndim; ++nu) {
                sum += gcov_nat[mu][nu] * v[nu];
            }
            out[mu] = sum;
        }
        return out;
    }

    KPOLARIS_INLINE Vec4<Real> native_to_cartesian_contravariant(Real x1, Real x2,
                                                              Real r, Real th,
                                                              Real cp, Real sp,
                                                              const Vec4<Real>& vnat) const {
        Real jac[ndim][ndim];
        native_to_ks_spherical_jacobian(x1, x2, r, jac);
        Vec4<Real> vsph;
        for (int mu = 0; mu < ndim; ++mu) {
            Real sum = Real(0);
            for (int nu = 0; nu < ndim; ++nu) {
                sum += jac[mu][nu] * vnat[nu];
            }
            vsph[mu] = sum;
        }
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

    KPOLARIS_INLINE Vec4<Real> native_to_ks_spherical_contravariant(Real x1, Real x2, Real r,
                                                                 const Vec4<Real>& vnat) const {
        Real jac[ndim][ndim];
        native_to_ks_spherical_jacobian(x1, x2, r, jac);
        Vec4<Real> vsph;
        for (int mu = 0; mu < ndim; ++mu) {
            Real sum = Real(0);
            for (int nu = 0; nu < ndim; ++nu) {
                sum += jac[mu][nu] * vnat[nu];
            }
            vsph[mu] = sum;
        }
        return vsph;
    }

    template<class Metric>
    KPOLARIS_INLINE int precomputed_fluid_state_at(const Metric& metric,
                                                const TransportState<Real>& state,
                                                Real& rho, Real& uu,
                                                Vec4<Real>& ucon,
                                                Vec4<Real>& bcon,
                                                Real& ne_cgs,
                                                Real& thetae,
                                                Real& b_cgs,
                                                Real& sigma,
                                                Real& beta,
                                                Real (*gcov_out)[ndim] = nullptr,
                                                Real* bnorm_geom_out = nullptr) const {
        if (data_coordinate_system != static_cast<int>(CoordinateSystem::SphericalKS)) {
            return 0;
        }
        Real r = Real(0), th = Real(0), cp = Real(1), sp = Real(0);
        Real phi_data = Real(0);
        if constexpr (Metric::coordinate_system == CoordinateSystem::SphericalKS) {
            r = state.x[1];
            th = state.x[2];
            phi_data = state.x[3];
        } else {
            bl_coordinates_for_metric(metric, state.x, r, th, cp, sp);
            phi_data = Kokkos::atan2(sp, cp);
        }
        if (!(r >= r_in && r < r_out)) {
            return 0;
        }
        const Real x1_data = radial_coordinate_log ? Kokkos::log(max_val(r, Real(1e-300))) : r;
        const Real x2_data = th;
        int i0, j0, k0;
        Real di, dj, dk;
        if (!native_indices_from_native_coords(x1_data, x2_data, phi_data, i0, j0, k0, di, dj, dk)) {
            return 0;
        }
        const InterpStencil stencil = make_interp_stencil(i0, j0, k0, di, dj, dk);
        interp_precomputed_fluid_state_stencil(stencil, ucon, bcon, ne_cgs, thetae, b_cgs, sigma, beta);
        if (!(ne_cgs > Real(0) && thetae > Real(0) && b_cgs > Real(0))) {
            return 0;
        }
        ne_cgs *= sigma_smooth_factor(sigma);
        rho = ne_cgs;
        uu = thetae;
        if (gcov_out) {
            if constexpr (Metric::coordinate_system == CoordinateSystem::SphericalKS) {
                ks_spherical_gcov(r, th, gcov_out);
            } else {
                metric.gcov_matrix(state.x, gcov_out);
            }
        }
        if (bnorm_geom_out) {
            *bnorm_geom_out = b_cgs / max_val(b_unit_cgs(), Real(1e-300));
        }
        return 1;
    }

    template<class Metric>
    KPOLARIS_INLINE int fluid_state(const Metric& metric,
                                 const TransportState<Real>& state,
                                 Real& rho, Real& uu,
                                 Vec4<Real>& ucon,
                                 Vec4<Real>& bcon,
                                 Real& ne_cgs,
                                 Real& thetae,
                                 Real& b_cgs,
                                 Real& sigma,
                                 Real& beta,
                                 Real (*gcov_out)[ndim] = nullptr,
                                 Real* bnorm_geom_out = nullptr,
                                 const GRMHDGridRadiationModel* time_upper = nullptr,
                                 Real time_fraction = Real(0)) const {
        if (precomputed_fluid_state) {
            return precomputed_fluid_state_at(metric, state, rho, uu, ucon, bcon, ne_cgs, thetae,
                                              b_cgs, sigma, beta, gcov_out, bnorm_geom_out);
        }
        Real r = Real(0), th = Real(0), cp = Real(1), sp = Real(0);
        Real x1_native = Real(0), x2_native = Real(0), phi_native = Real(0);
        const bool data_is_spherical_ks =
            data_coordinate_system == static_cast<int>(CoordinateSystem::SphericalKS);
        if (data_is_spherical_ks) {
            if constexpr (Metric::coordinate_system == CoordinateSystem::SphericalKS) {
                r = state.x[1];
                th = state.x[2];
                phi_native = state.x[3];
            } else {
                bl_coordinates_for_metric(metric, state.x, r, th, cp, sp);
                phi_native = Kokkos::atan2(sp, cp);
            }
            x1_native = radial_coordinate_log ? Kokkos::log(max_val(r, Real(1e-300))) : r;
            x2_native = th;
        } else if constexpr (Metric::coordinate_system == CoordinateSystem::FMKS) {
            x1_native = state.x[1];
            x2_native = state.x[2];
            phi_native = state.x[3];
            r = Kokkos::exp(x1_native);
            th = fmks_theta_from_x2(x1_native, x2_native);
        } else if constexpr (Metric::coordinate_system != CoordinateSystem::CartesianKS) {
            r = state.x[1];
            th = state.x[2];
            phi_native = state.x[3];
            x1_native = Kokkos::log(max_val(r, Real(1e-300)));
            x2_native = native_x2_from_theta(x1_native, th);
        } else {
            bl_coordinates_for_metric(metric, state.x, r, th, cp, sp);
            phi_native = Kokkos::atan2(sp, cp);
            x1_native = Kokkos::log(max_val(r, Real(1e-300)));
            x2_native = native_x2_from_theta(x1_native, th);
        }
        if (!(r >= r_in && r < r_out)) {
            return 0;
        }
        int i0, j0, k0;
        Real di, dj, dk;
        if (!native_indices_from_native_coords(x1_native, x2_native, phi_native,
                                               i0, j0, k0, di, dj, dk)) {
            return 0;
        }
        const InterpStencil stencil = make_interp_stencil(i0, j0, k0, di, dj, dk);
        InterpStencil upper_stencil = stencil;
        if (time_upper) {
            int iu, ju, ku;
            Real fu, gu, hu;
            if (!time_upper->native_indices_from_native_coords(x1_native, x2_native, phi_native,
                                                               iu, ju, ku, fu, gu, hu)) return 0;
            upper_stencil = time_upper->make_interp_stencil(iu, ju, ku, fu, gu, hu);
        }
        const auto sample_primitive = [&](int component) {
            const Real a = interp_prim_stencil(component, stencil);
            return time_upper ? interpolate_time_value(a,
                time_upper->interp_prim_stencil(component, upper_stencil), time_fraction) : a;
        };
        const auto sample_derived = [&](int component) {
            const Real a = interp_derived_stencil(component, stencil);
            return time_upper ? interpolate_time_value(a,
                time_upper->interp_derived_stencil(component, upper_stencil), time_fraction) : a;
        };
        rho = sample_primitive(0);
        uu = sample_primitive(1);
        if (!(rho > Real(0) && uu > Real(0))) {
            return 0;
        }
        const Real U1 = sample_primitive(2);
        const Real U2 = sample_primitive(3);
        const Real U3 = sample_primitive(4);
        const Real B1 = sample_primitive(5);
        const Real B2 = sample_primitive(6);
        const Real B3 = sample_primitive(7);
        const Real x1 = x1_native;
        const Real x2 = x2_native;
        Real gcov_nat[ndim][ndim];
        Real gcon0_nat[ndim];
        if (data_is_spherical_ks) {
            ks_spherical_gcov(r, th, gcov_nat);
            Real gcon_sks[ndim][ndim];
            ks_spherical_gcon(r, th, gcon_sks);
            for (int mu = 0; mu < ndim; ++mu) {
                gcon0_nat[mu] = gcon_sks[0][mu];
            }
        } else {
            native_fluid_metric(x1, x2, r, th, gcov_nat, gcon0_nat);
        }

        Vec4<Real> vnat(Real(0), U1, U2, U3);
        Real spatial_norm = Real(0);
        for (int i = 1; i < ndim; ++i) {
            for (int j = 1; j < ndim; ++j) {
                spatial_norm += gcov_nat[i][j] * vnat[i] * vnat[j];
            }
        }
        const Real vfac = Kokkos::sqrt(max_val(-Real(1) / gcon0_nat[0] * (Real(1) + abs_val(spatial_norm)), Real(0)));
        Vec4<Real> ucon_nat;
        ucon_nat[0] = -vfac * gcon0_nat[0];
        for (int i = 1; i < ndim; ++i) {
            ucon_nat[i] = vnat[i] - vfac * gcon0_nat[i];
        }
        const Vec4<Real> ucov_nat = lower_native(gcov_nat, ucon_nat);

        const Vec4<Real> Bnat(Real(0), B1, B2, B3);
        Real udotB = Real(0);
        for (int i = 1; i < ndim; ++i) {
            udotB += ucov_nat[i] * Bnat[i];
        }
        Vec4<Real> bcon_nat;
        bcon_nat[0] = udotB;
        for (int i = 1; i < ndim; ++i) {
            bcon_nat[i] = (Bnat[i] + ucon_nat[i] * udotB) / max_val(ucon_nat[0], Real(1e-300));
        }
        const Vec4<Real> bcov_nat = lower_native(gcov_nat, bcon_nat);
        Real bsq = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            bsq += bcon_nat[mu] * bcov_nat[mu];
        }
        bsq = max_val(abs_val(bsq), Real(1e-40));
        if (bnorm_geom_out) {
            *bnorm_geom_out = Kokkos::sqrt(bsq);
        }

        if (data_is_spherical_ks) {
            ucon = ucon_nat;
            bcon = bcon_nat;
        } else if constexpr (Metric::coordinate_system == CoordinateSystem::CartesianKS) {
            ucon = native_to_cartesian_contravariant(x1, x2, r, th, cp, sp, ucon_nat);
            bcon = native_to_cartesian_contravariant(x1, x2, r, th, cp, sp, bcon_nat);
        } else if constexpr (Metric::coordinate_system == CoordinateSystem::FMKS) {
            ucon = ucon_nat;
            bcon = bcon_nat;
        } else {
            ucon = native_to_ks_spherical_contravariant(x1, x2, r, ucon_nat);
            bcon = native_to_ks_spherical_contravariant(x1, x2, r, bcon_nat);
        }
#if KPOLARIS_GRMHD_REQUIRE_DERIVED_SCALARS
        ne_cgs = sample_derived(DerivedNe);
        thetae = sample_derived(DerivedThetae);
        b_cgs = sample_derived(DerivedB);
        sigma = sample_derived(DerivedSigma);
        beta = sample_derived(DerivedBeta);
#else
        if (has_derived_scalars) {
            ne_cgs = sample_derived(DerivedNe);
            thetae = sample_derived(DerivedThetae);
            b_cgs = sample_derived(DerivedB);
            sigma = sample_derived(DerivedSigma);
            beta = sample_derived(DerivedBeta);
        } else {
            sigma = bsq / max_val(rho, Real(1e-300));
            beta = (gam - Real(1)) * uu / (Real(0.5) * bsq);
            ne_cgs = rho * rho_unit_cgs() / Real(1.67353286206015e-24) * ne_factor;
            b_cgs = Kokkos::sqrt(bsq) * b_unit_cgs();
            const Real betasq = beta * beta / max_val(beta_crit * beta_crit, Real(1e-40));
            const Real trat = (trat_large * betasq + trat_small) / (Real(1) + betasq);
            const Real mp_me = Real(1836.1526734400013);
            const Real game = Real(4) / Real(3);
            const Real gamp = Real(5) / Real(3);
            const Real thetae_unit = mp_me * (game - Real(1)) * (gamp - Real(1)) /
                ((gamp - Real(1)) + (game - Real(1)) * trat);
            thetae = max_val(thetae_unit * uu / max_val(rho, Real(1e-300)), Real(1e-3));
        }
#endif
        ne_cgs *= sigma_smooth_factor(sigma);
        if (gcov_out) {
            if (data_is_spherical_ks) {
                metric.gcov_matrix(state.x, gcov_out);
            } else if constexpr (Metric::coordinate_system == CoordinateSystem::FMKS) {
                for (int mu = 0; mu < ndim; ++mu) {
                    for (int nu = 0; nu < ndim; ++nu) {
                        gcov_out[mu][nu] = gcov_nat[mu][nu];
                    }
                }
            } else {
                metric.gcov_matrix(state.x, gcov_out);
            }
        }
        return 1;
    }

    template<class Metric>
    KPOLARIS_INLINE int temporal_fluid_state(const GRMHDGridRadiationModel& upper, Real fraction,
        const Metric& metric, const TransportState<Real>& state,
        Real& rho, Real& uu, Vec4<Real>& u, Vec4<Real>& b,
        Real& ne, Real& theta, Real& B, Real& sigma, Real& beta) const {
        return fluid_state(metric, state, rho, uu, u, b, ne, theta, B, sigma, beta,
                           nullptr, nullptr, &upper, fraction);
    }

    template<class Metric>
    KPOLARIS_INLINE TransferCoeffs<Real> coefficients(const Metric& metric,
                                                   const TransportState<Real>& state,
                                                   Real,
                                                   const PlasmaPerturbation<Real>& perturbation = {},
                                                   const GRMHDGridRadiationModel* time_upper = nullptr,
                                                   Real time_fraction = Real(0)) const {
        TransferCoeffs<Real> coeffs;
        Real rho, uu, ne_cgs, thetae, b_cgs, sigma, beta;
        Real bnorm_geom = Real(0);
        Vec4<Real> ucon, bcon;
        Real gcov[ndim][ndim];
        if (!fluid_state(metric, state, rho, uu, ucon, bcon, ne_cgs, thetae,
                         b_cgs, sigma, beta, gcov, &bnorm_geom, time_upper, time_fraction)) {
            return coeffs;
        }
        perturbation.apply(ne_cgs, thetae, b_cgs, sigma, beta);
        const Vec4<Real> ucov = lower_with_gcov(gcov, ucon);
        Real uk = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            uk += ucov[mu] * state.k[mu];
        }
        const Real nu_scale = max_val(abs_val(-uk), Real(1e-8));
        const Real nu = max_val(freq_cgs * nu_scale, Real(1));
        const Real bnorm = max_val(bnorm_geom, Real(1e-150));
        const Real inv_bnorm = Real(1) / bnorm;
        Real kdotb = Real(0);
        Real ub = Real(0);
        Real ue1 = Real(0);
        Real ue2 = Real(0);
        Real ke1 = Real(0);
        Real ke2 = Real(0);
        Real e1b_cov = Real(0);
        Real e2b_cov = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            Real bcov_mu = Real(0);
            Real kcov_mu = Real(0);
            for (int nu_i = 0; nu_i < ndim; ++nu_i) {
                bcov_mu += gcov[mu][nu_i] * bcon[nu_i];
                kcov_mu += gcov[mu][nu_i] * state.k[nu_i];
            }
            kdotb += state.k[mu] * bcov_mu;
            ub += ucov[mu] * bcon[mu];
            ue1 += ucov[mu] * state.e1[mu];
            ue2 += ucov[mu] * state.e2[mu];
            ke1 += kcov_mu * state.e1[mu];
            ke2 += kcov_mu * state.e2[mu];
            e1b_cov += bcov_mu * state.e1[mu];
            e2b_cov += bcov_mu * state.e2[mu];
        }
        const Real cos_theta = clamp(kdotb / max_val(nu_scale * bnorm, Real(1e-30)), Real(-1), Real(1));
        const Real sin_theta = max_val(Kokkos::sqrt(max_val(Real(0), Real(1) - cos_theta * cos_theta)), Real(1e-4));
        const Real theta = Real(0);

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
        thermal_state.nu = nu;
        thermal_state.ne = ne_cgs;
        thermal_state.thetae = thetae;
        thermal_state.b_cgs = b_cgs;
        thermal_state.theta = theta;
        thermal_state.sin_theta = sin_theta;
        thermal_state.cos_theta = cos_theta;

        TransferCoeffs<Real> magnetic_coeffs;
#if KPOLARIS_IHARM_COMPILED_EMISSION_TYPE == KPOLARIS_THERMAL_SYNCHROTRON_DEXTER
        magnetic_coeffs =
            thermal_synchrotron_magnetic_basis_coefficients_fit<ThermalSynchrotronDexter>(thermal_state,
                                                                                         thermal_params);
#elif KPOLARIS_IHARM_COMPILED_EMISSION_TYPE == KPOLARIS_THERMAL_SYNCHROTRON_PANDYA
        magnetic_coeffs =
            thermal_synchrotron_magnetic_basis_coefficients_fit<ThermalSynchrotronPandya>(thermal_state,
                                                                                         thermal_params);
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
            nonthermal_params.kappa = variable_kappa ?
                variable_kappa_from_sigma_beta(sigma, beta, nonthermal_params) : nonthermal_kappa;
            if (nonthermal_params.kappa < variable_kappa_min) {
                nonthermal_params.kappa = variable_kappa_min;
            }
            nonthermal_params.kappa_interp_begin = min_val(variable_kappa_interp_start,
                                                           variable_kappa_max);
            nonthermal_params.kappa_interp_end = variable_kappa_max;
            nonthermal_params.power_law_p = powerlaw_p;
            nonthermal_params.power_law_eta = powerlaw_eta;
            nonthermal_params.power_law_gamma_min = powerlaw_gamma_min;
            nonthermal_params.power_law_gamma_max = powerlaw_gamma_max;
            nonthermal_params.power_law_gamma_cutoff = powerlaw_gamma_cutoff;
            LocalNonthermalSynchrotronState<Real> nonthermal_state;
            nonthermal_state.nu = nu;
            nonthermal_state.ne = ne_cgs;
            nonthermal_state.thetae = thetae;
            nonthermal_state.b_cgs = b_cgs;
            nonthermal_state.sin_theta = sin_theta;
            nonthermal_state.cos_theta = cos_theta;
            magnetic_coeffs = nonthermal_synchrotron_magnetic_basis_coefficients(nonthermal_state,
                                                                                 nonthermal_params);
        } else {
            magnetic_coeffs = thermal_synchrotron_magnetic_basis_coefficients(thermal_state,
                                                                              thermal_params);
        }
#endif

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
        if (bproj2 > Real(1e-30)) {
            const Real cos2chi = (b2 * b2 - b1 * b1) / bproj2;
            const Real sin2chi = -Real(2) * b1 * b2 / bproj2;
            coeffs.jQ = magnetic_coeffs.jQ * cos2chi;
            coeffs.jU = magnetic_coeffs.jQ * sin2chi;
            coeffs.aQ = magnetic_coeffs.aQ * cos2chi;
            coeffs.aU = magnetic_coeffs.aQ * sin2chi;
            coeffs.rQ = magnetic_coeffs.rQ * cos2chi;
            coeffs.rU = magnetic_coeffs.rQ * sin2chi;
        }
        return coeffs;
    }
};

} // namespace kpolaris
