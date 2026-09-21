#include "grmhd/iharm_loader.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <H5Cpp.h>
#include <Kokkos_Core.hpp>

#include "common/vec.hpp"
#include "geometry/kerr_schild_spherical.hpp"

namespace kpolaris {
namespace {

bool h5_path_exists(H5::H5File& file, const std::string& path) {
    return H5Lexists(file.getId(), path.c_str(), H5P_DEFAULT) > 0;
}

template<class T>
T read_h5_scalar_value(H5::H5File& file, const std::string& path) {
    H5::DataSet ds = file.openDataSet(path);
    T value{};
    if constexpr (std::is_integral_v<T>) {
        ds.read(&value, H5::PredType::NATIVE_INT);
    } else {
        double tmp = 0.0;
        ds.read(&tmp, H5::PredType::NATIVE_DOUBLE);
        value = static_cast<T>(tmp);
    }
    return value;
}

void report_timing(int enabled, const char* name, double seconds) {
    if (enabled) {
        std::cout << "timing " << name << ' ' << seconds << " s\n";
    }
}

template<class RealT>
GRMHDGridRadiationModel<RealT> resample_to_spherical_ks_primitives(
    const GRMHDGridRadiationModel<RealT>& source,
    const IHARMLoadOptions& opt) {
    Kokkos::Timer timer;
    GRMHDGridRadiationModel<RealT> target = source;
    const RealT pi = RealT(3.141592653589793238462643383279502884);
    const int n1 = opt.resample_n1 > 0 ? opt.resample_n1 : source.n1;
    const int n2 = opt.resample_n2 > 0 ? opt.resample_n2 : source.n2;
    const int n3 = opt.resample_n3 > 0 ? opt.resample_n3 : source.n3;
    const RealT rin = opt.resample_r_in > RealT(0) ? opt.resample_r_in : source.r_in;
    const RealT rout = opt.resample_r_out > RealT(0) ? opt.resample_r_out : source.r_out;
    if (!(n1 > 1 && n2 > 1 && n3 > 0 && rin > RealT(0) && rout > rin)) {
        throw std::runtime_error("invalid Spherical KS primitive resample grid for iharm");
    }

    target.n1 = n1;
    target.n2 = n2;
    target.n3 = n3;
    target.startx1 = Kokkos::log(rin);
    target.dx1 = (Kokkos::log(rout) - target.startx1) / RealT(n1);
    target.startx2 = RealT(0);
    target.dx2 = pi / RealT(n2);
    target.startx3 = RealT(0);
    target.dx3 = RealT(2) * pi / RealT(n3);
    target.r_in = rin;
    target.r_out = rout;
    target.precomputed_fluid_state = 0;
    target.data_coordinate_system = static_cast<int>(CoordinateSystem::SphericalKS);
    target.radial_coordinate_log = 1;
    target.has_derived_scalars = source.has_derived_scalars;
    target.fluid_states = typename GRMHDGridRadiationModel<RealT>::FluidStateView();
    target.prims = Kokkos::View<RealT*>("iharm_sks_resampled_prims",
        static_cast<size_t>(8) * target.n1 * target.n2 * target.n3);
    if (target.has_derived_scalars) {
        target.derived_scalars = Kokkos::View<RealT*>("iharm_sks_resampled_derived_scalars",
            static_cast<size_t>(GRMHDGridRadiationModel<RealT>::NumDerivedScalars) *
            target.n1 * target.n2 * target.n3);
    } else {
        target.derived_scalars = Kokkos::View<RealT*>();
    }

    using ExecSpace = Kokkos::DefaultExecutionSpace;
    const int ncell = target.n1 * target.n2 * target.n3;
    GRMHDGridRadiationModel<RealT> source_device = source;
    GRMHDGridRadiationModel<RealT> target_device = target;
    Kokkos::parallel_for(
        "KPOLARISIharmResampleSphericalKSPrimitives",
        Kokkos::RangePolicy<ExecSpace>(0, ncell),
        KOKKOS_LAMBDA(const int p) {
            const int k = p % target_device.n3;
            const int j = (p / target_device.n3) % target_device.n2;
            const int i = p / (target_device.n2 * target_device.n3);
            const RealT x1 = target_device.startx1 + (RealT(i) + RealT(0.5)) * target_device.dx1;
            const RealT r = Kokkos::exp(x1);
            const RealT th = target_device.startx2 + (RealT(j) + RealT(0.5)) * target_device.dx2;
            const RealT phi = target_device.startx3 + (RealT(k) + RealT(0.5)) * target_device.dx3;
            const RealT x2_fmks = source_device.native_x2_from_theta(x1, th);

            for (int v = 0; v < 8; ++v) {
                target_device.prims(target_device.prim_index(v, i, j, k)) = RealT(0);
            }
            if (target_device.has_derived_scalars) {
                for (int v = 0; v < GRMHDGridRadiationModel<RealT>::NumDerivedScalars; ++v) {
                    target_device.derived_scalars(target_device.derived_index(v, i, j, k)) = RealT(0);
                }
            }

            int i0, j0, k0;
            RealT di, dj, dk;
            if (!source_device.native_indices_from_native_coords(x1, x2_fmks, phi, i0, j0, k0, di, dj, dk)) {
                return;
            }
            const auto stencil = source_device.make_interp_stencil(i0, j0, k0, di, dj, dk);
            const RealT rho = source_device.interp_prim_stencil(0, stencil);
            const RealT uu = source_device.interp_prim_stencil(1, stencil);
            const RealT U1 = source_device.interp_prim_stencil(2, stencil);
            const RealT U2 = source_device.interp_prim_stencil(3, stencil);
            const RealT U3 = source_device.interp_prim_stencil(4, stencil);
            const RealT B1 = source_device.interp_prim_stencil(5, stencil);
            const RealT B2 = source_device.interp_prim_stencil(6, stencil);
            const RealT B3 = source_device.interp_prim_stencil(7, stencil);
            const RealT dth1 = source_device.dfmks_theta_dx1(x1, x2_fmks);
            const RealT dth2 = source_device.dfmks_theta_dx2(x1, x2_fmks);
            target_device.prims(target_device.prim_index(0, i, j, k)) = rho;
            target_device.prims(target_device.prim_index(1, i, j, k)) = uu;
            target_device.prims(target_device.prim_index(2, i, j, k)) = r * U1;
            target_device.prims(target_device.prim_index(3, i, j, k)) = dth1 * U1 + dth2 * U2;
            target_device.prims(target_device.prim_index(4, i, j, k)) = U3;
            target_device.prims(target_device.prim_index(5, i, j, k)) = r * B1;
            target_device.prims(target_device.prim_index(6, i, j, k)) = dth1 * B1 + dth2 * B2;
            target_device.prims(target_device.prim_index(7, i, j, k)) = B3;
            if (target_device.has_derived_scalars) {
                for (int v = 0; v < GRMHDGridRadiationModel<RealT>::NumDerivedScalars; ++v) {
                    target_device.derived_scalars(target_device.derived_index(v, i, j, k)) =
                        source_device.interp_derived_stencil(v, stencil);
                }
            }
        });
    Kokkos::fence();
    report_timing(opt.timing, "iharm_resample_spherical_ks_primitives", timer.seconds());
    return target;
}

template<class RealT>
GRMHDGridRadiationModel<RealT> resample_to_spherical_ks_precomputed(
    const GRMHDGridRadiationModel<RealT>& source,
    const IHARMLoadOptions& opt) {
    Kokkos::Timer timer;
    GRMHDGridRadiationModel<RealT> target = source;
    const RealT pi = RealT(3.141592653589793238462643383279502884);
    const int n1 = opt.resample_n1 > 0 ? opt.resample_n1 : source.n1;
    const int n2 = opt.resample_n2 > 0 ? opt.resample_n2 : source.n2;
    const int n3 = opt.resample_n3 > 0 ? opt.resample_n3 : source.n3;
    const RealT rin = opt.resample_r_in > RealT(0) ? opt.resample_r_in : source.r_in;
    const RealT rout = opt.resample_r_out > RealT(0) ? opt.resample_r_out : source.r_out;
    if (!(n1 > 1 && n2 > 1 && n3 > 0 && rin > RealT(0) && rout > rin)) {
        throw std::runtime_error("invalid Spherical KS resample grid for iharm");
    }

    target.n1 = n1;
    target.n2 = n2;
    target.n3 = n3;
    target.startx1 = Kokkos::log(rin);
    target.dx1 = (Kokkos::log(rout) - target.startx1) / RealT(n1);
    target.startx2 = RealT(0);
    target.dx2 = pi / RealT(n2);
    target.startx3 = RealT(0);
    target.dx3 = RealT(2) * pi / RealT(n3);
    target.r_in = rin;
    target.r_out = rout;
    target.precomputed_fluid_state = 1;
    target.data_coordinate_system = static_cast<int>(CoordinateSystem::SphericalKS);
    target.radial_coordinate_log = 1;
    target.has_derived_scalars = 0;
    target.prims = Kokkos::View<RealT*>();
    target.derived_scalars = Kokkos::View<RealT*>();
    target.fluid_states = typename GRMHDGridRadiationModel<RealT>::FluidStateView("iharm_sks_precomputed_fluid_state",
        static_cast<size_t>(GRMHDGridRadiationModel<RealT>::NumFluidStateScalars) *
        target.n1 * target.n2 * target.n3);

    using ExecSpace = Kokkos::DefaultExecutionSpace;
    const int ncell = target.n1 * target.n2 * target.n3;
    GRMHDGridRadiationModel<RealT> source_device = source;
    GRMHDGridRadiationModel<RealT> target_device = target;
    KerrSchildSphericalMetric<RealT> metric(RealT(1), source.spin);
    Kokkos::parallel_for(
        "KPOLARISIharmResampleSphericalKSPrecomputed",
        Kokkos::RangePolicy<ExecSpace>(0, ncell),
        KOKKOS_LAMBDA(const int p) {
            const int k = p % target_device.n3;
            const int j = (p / target_device.n3) % target_device.n2;
            const int i = p / (target_device.n2 * target_device.n3);
            const RealT x1 = target_device.startx1 + (RealT(i) + RealT(0.5)) * target_device.dx1;
            const RealT r = Kokkos::exp(x1);
            const RealT th = target_device.startx2 + (RealT(j) + RealT(0.5)) * target_device.dx2;
            const RealT phi = target_device.startx3 + (RealT(k) + RealT(0.5)) * target_device.dx3;

            TransportState<RealT> state;
            state.x = Vec4<RealT>(RealT(0), r, th, phi);
            RealT rho = RealT(0), uu = RealT(0), ne = RealT(0), thetae = RealT(0);
            RealT b_cgs = RealT(0), sigma = RealT(0), beta = RealT(0);
            RealT bnorm = RealT(0);
            Vec4<RealT> ucon, bcon;
            const int ok = source_device.fluid_state(metric, state, rho, uu, ucon, bcon,
                                                     ne, thetae, b_cgs, sigma, beta,
                                                     nullptr, &bnorm);
            const size_t cell = (static_cast<size_t>(i) * static_cast<size_t>(target_device.n2) +
                                 static_cast<size_t>(j)) * static_cast<size_t>(target_device.n3) +
                                static_cast<size_t>(k);
            const size_t ngrid = static_cast<size_t>(target_device.n1) *
                                 static_cast<size_t>(target_device.n2) *
                                 static_cast<size_t>(target_device.n3);
            for (int v = 0; v < GRMHDGridRadiationModel<RealT>::NumFluidStateScalars; ++v) {
                target_device.fluid_states(static_cast<size_t>(v) * ngrid + cell) = RealT(0);
            }
            if (ok) {
                target_device.fluid_states(static_cast<size_t>(GRMHDGridRadiationModel<RealT>::FluidU0) * ngrid + cell) = ucon[0];
                target_device.fluid_states(static_cast<size_t>(GRMHDGridRadiationModel<RealT>::FluidU1) * ngrid + cell) = ucon[1];
                target_device.fluid_states(static_cast<size_t>(GRMHDGridRadiationModel<RealT>::FluidU2) * ngrid + cell) = ucon[2];
                target_device.fluid_states(static_cast<size_t>(GRMHDGridRadiationModel<RealT>::FluidU3) * ngrid + cell) = ucon[3];
                target_device.fluid_states(static_cast<size_t>(GRMHDGridRadiationModel<RealT>::FluidB0) * ngrid + cell) = bcon[0];
                target_device.fluid_states(static_cast<size_t>(GRMHDGridRadiationModel<RealT>::FluidB1) * ngrid + cell) = bcon[1];
                target_device.fluid_states(static_cast<size_t>(GRMHDGridRadiationModel<RealT>::FluidB2) * ngrid + cell) = bcon[2];
                target_device.fluid_states(static_cast<size_t>(GRMHDGridRadiationModel<RealT>::FluidB3) * ngrid + cell) = bcon[3];
                target_device.fluid_states(static_cast<size_t>(GRMHDGridRadiationModel<RealT>::FluidNe) * ngrid + cell) = ne;
                target_device.fluid_states(static_cast<size_t>(GRMHDGridRadiationModel<RealT>::FluidThetae) * ngrid + cell) = thetae;
                target_device.fluid_states(static_cast<size_t>(GRMHDGridRadiationModel<RealT>::FluidBCgs) * ngrid + cell) = b_cgs;
                target_device.fluid_states(static_cast<size_t>(GRMHDGridRadiationModel<RealT>::FluidSigma) * ngrid + cell) = sigma;
                target_device.fluid_states(static_cast<size_t>(GRMHDGridRadiationModel<RealT>::FluidBeta) * ngrid + cell) = beta;
            }
        });
    Kokkos::fence();
    report_timing(opt.timing, "iharm_resample_spherical_ks_precomputed", timer.seconds());
    return target;
}

} // namespace

DefaultReal read_iharm_dump_time(const std::string& dump_path) {
    if (dump_path.empty()) {
        throw std::runtime_error("empty iharm dump path");
    }
    H5::H5File file(dump_path, H5F_ACC_RDONLY);
    if (h5_path_exists(file, "/t")) {
        return read_h5_scalar_value<DefaultReal>(file, "/t");
    }
    if (h5_path_exists(file, "/header/t")) {
        return read_h5_scalar_value<DefaultReal>(file, "/header/t");
    }
    throw std::runtime_error("iharm dump has no /t or /header/t time dataset: " + dump_path);
}

GRMHDRadiationModel<DefaultReal> load_iharm_model_from_hdf5(const IHARMLoadOptions& opt) {
    using RealT = DefaultReal;
    Kokkos::Timer total_timer;
    Kokkos::Timer phase_timer;
#if KPOLARIS_GRMHD_REQUIRE_DERIVED_SCALARS
    if (!opt.interpolate_derived_scalars) {
        throw std::runtime_error(
            "this iHARM build requires --iharm_interpolate_derived_scalars=1");
    }
#endif
    if (opt.dump_path.empty()) {
        throw std::runtime_error("--iharm_dump is required for --model=iharm");
    }
    H5::H5File file(opt.dump_path, H5F_ACC_RDONLY);
    GRMHDGridRadiationModel<RealT> model;
    model.n1 = read_h5_scalar_value<int>(file, "/header/n1");
    model.n2 = read_h5_scalar_value<int>(file, "/header/n2");
    model.n3 = read_h5_scalar_value<int>(file, "/header/n3");
    model.spin = read_h5_scalar_value<RealT>(file, "/header/a");
    model.gam = h5_path_exists(file, "/header/gam") ?
                read_h5_scalar_value<RealT>(file, "/header/gam") :
                read_h5_scalar_value<RealT>(file, "/header/gamma");
    model.startx1 = read_h5_scalar_value<RealT>(file, "/header/geom/startx1");
    model.startx2 = read_h5_scalar_value<RealT>(file, "/header/geom/startx2");
    model.startx3 = read_h5_scalar_value<RealT>(file, "/header/geom/startx3");
    model.dx1 = read_h5_scalar_value<RealT>(file, "/header/geom/dx1");
    model.dx2 = read_h5_scalar_value<RealT>(file, "/header/geom/dx2");
    model.dx3 = read_h5_scalar_value<RealT>(file, "/header/geom/dx3");
    const auto finite = [](RealT value) {
        return std::isfinite(static_cast<double>(value));
    };
    if (model.n1 <= 0 || model.n2 <= 0 || model.n3 <= 0) {
        throw std::runtime_error("iharm header dimensions must be positive");
    }
    if (!finite(model.spin) || std::abs(model.spin) > RealT(1)) {
        throw std::runtime_error("iharm header spin must be finite and lie in [-1, 1]");
    }
    if (!finite(model.gam) || model.gam <= RealT(1)) {
        throw std::runtime_error("iharm header gamma must be finite and greater than 1");
    }
    if (!finite(model.startx1) || !finite(model.startx2) || !finite(model.startx3)) {
        throw std::runtime_error("iharm grid starts must be finite");
    }
    if (!finite(model.dx1) || !finite(model.dx2) || !finite(model.dx3) ||
        model.dx1 <= RealT(0) || model.dx2 <= RealT(0) || model.dx3 <= RealT(0)) {
        throw std::runtime_error("iharm grid spacings must be finite and positive");
    }
    std::string geom_group = "/header/geom/fmks";
    if (!h5_path_exists(file, geom_group + "/hslope")) {
        geom_group = "/header/geom/mmks";
    }
    model.hslope = read_h5_scalar_value<RealT>(file, geom_group + "/hslope");
    model.mks_smooth = read_h5_scalar_value<RealT>(file, geom_group + "/mks_smooth");
    model.poly_alpha = read_h5_scalar_value<RealT>(file, geom_group + "/poly_alpha");
    model.poly_xt = read_h5_scalar_value<RealT>(file, geom_group + "/poly_xt");
    model.r_in = read_h5_scalar_value<RealT>(file, geom_group + "/r_in");
    model.r_out = read_h5_scalar_value<RealT>(file, geom_group + "/r_out");
    if (!finite(model.hslope) || !finite(model.mks_smooth) ||
        !finite(model.poly_alpha) || !finite(model.poly_xt) ||
        model.poly_alpha <= RealT(-1) || model.poly_xt <= RealT(0)) {
        throw std::runtime_error(
            "iharm FMKS/MMKS mapping parameters must be finite with poly_alpha > -1 and poly_xt > 0");
    }
    if (!finite(model.r_in) || !finite(model.r_out) || model.r_in <= RealT(0) ||
        model.r_out <= model.r_in) {
        throw std::runtime_error(
            "iharm radial bounds must be finite, positive, and increasing");
    }
    const RealT pi = RealT(3.141592653589793238462643383279502884);
    model.poly_norm = RealT(0.5) * pi /
        (RealT(1) + RealT(1) / (model.poly_alpha + RealT(1)) /
         Kokkos::pow(model.poly_xt, model.poly_alpha));
    if (!finite(model.poly_norm) || model.poly_norm <= RealT(0)) {
        throw std::runtime_error(
            "iharm FMKS/MMKS mapping normalization must be finite and positive");
    }
    report_timing(opt.timing, "iharm_header", phase_timer.seconds());
    phase_timer.reset();

    model.freq_cgs = opt.freq;
    model.M_unit = opt.M_unit;
    model.mbh_solar = opt.mbh_solar;
    model.trat_small = opt.trat_small;
    model.trat_large = opt.trat_large;
    model.beta_crit = opt.beta_crit;
    model.sigma_cut = opt.sigma_cut;
    model.sigma_cut_high = opt.sigma_cut_high;
    model.emission_type = opt.emission_type;
    model.nonthermal_kappa = opt.nonthermal_kappa;
    model.variable_kappa = opt.variable_kappa;
    model.variable_kappa_min = opt.variable_kappa_min;
    model.variable_kappa_interp_start = opt.variable_kappa_interp_start;
    model.variable_kappa_max = opt.variable_kappa_max;
    model.powerlaw_p = opt.powerlaw_p;
    model.powerlaw_eta = opt.powerlaw_eta;
    model.powerlaw_gamma_min = opt.powerlaw_gamma_min;
    model.powerlaw_gamma_max = opt.powerlaw_gamma_max;
    model.powerlaw_gamma_cutoff = opt.powerlaw_gamma_cutoff;
    model.has_derived_scalars = opt.interpolate_derived_scalars;

    H5::DataSet prims_ds = file.openDataSet("/prims");
    H5::DataSpace space = prims_ds.getSpace();
    hsize_t dims[4] = {0, 0, 0, 0};
    if (space.getSimpleExtentNdims() != 4) {
        throw std::runtime_error("iharm /prims dataset must be rank 4");
    }
    space.getSimpleExtentDims(dims);
    if (static_cast<int>(dims[0]) != model.n1 || static_cast<int>(dims[1]) != model.n2 ||
        static_cast<int>(dims[2]) != model.n3 || dims[3] < 8) {
        throw std::runtime_error("iharm /prims dimensions do not match header or have fewer than 8 primitives");
    }
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    size_t cell_count = static_cast<size_t>(dims[0]);
    if (dims[1] > std::numeric_limits<size_t>::max() / cell_count) {
        throw std::runtime_error("iharm grid cell count overflows size_t");
    }
    cell_count *= static_cast<size_t>(dims[1]);
    if (dims[2] > std::numeric_limits<size_t>::max() / cell_count) {
        throw std::runtime_error("iharm grid cell count overflows size_t");
    }
    cell_count *= static_cast<size_t>(dims[2]);
    if (cell_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("iharm grid cell count exceeds the supported int index range");
    }
    const size_t nprim_file = static_cast<size_t>(dims[3]);
    if (nprim_file > std::numeric_limits<size_t>::max() / cell_count) {
        throw std::runtime_error("iharm primitive array size overflows size_t");
    }
    const size_t raw_size = cell_count * nprim_file;
    Kokkos::View<float*, Kokkos::HostSpace> raw_host("iharm_raw_host", raw_size);
    prims_ds.read(raw_host.data(), H5::PredType::NATIVE_FLOAT);
    for (size_t cell = 0; cell < cell_count; ++cell) {
        const size_t base = cell * nprim_file;
        for (size_t primitive = 0; primitive < 8; ++primitive) {
            if (!std::isfinite(static_cast<double>(raw_host(base + primitive)))) {
                throw std::runtime_error(
                    "iharm /prims first eight primitive fields must be finite");
            }
        }
    }
    report_timing(opt.timing, "iharm_prims_hdf5_read", phase_timer.seconds());
    phase_timer.reset();
    Kokkos::View<float*, ExecSpace> raw_device("iharm_raw_device", raw_size);
    Kokkos::deep_copy(raw_device, raw_host);

    model.prims = Kokkos::View<RealT*>("iharm_prims", static_cast<size_t>(8) * model.n1 * model.n2 * model.n3);
    const int ncell = model.n1 * model.n2 * model.n3;
    GRMHDGridRadiationModel<RealT> reorder_model = model;
    Kokkos::parallel_for(
        "KPOLARISIharmReorderPrims",
        Kokkos::RangePolicy<ExecSpace>(0, ncell),
        KOKKOS_LAMBDA(const int p) {
            const int k = p % reorder_model.n3;
            const int j = (p / reorder_model.n3) % reorder_model.n2;
            const int i = p / (reorder_model.n2 * reorder_model.n3);
            const size_t base = (((static_cast<size_t>(i) * static_cast<size_t>(reorder_model.n2) +
                                  static_cast<size_t>(j)) * static_cast<size_t>(reorder_model.n3) +
                                  static_cast<size_t>(k)) * nprim_file);
            for (int v = 0; v < 8; ++v) {
                reorder_model.prims(reorder_model.prim_index(v, i, j, k)) =
                    static_cast<RealT>(raw_device(base + static_cast<size_t>(v)));
            }
        });
    Kokkos::fence();
    report_timing(opt.timing, "iharm_prims_upload_reorder", phase_timer.seconds());
    phase_timer.reset();

    if (model.has_derived_scalars) {
        model.derived_scalars = Kokkos::View<RealT*>("iharm_derived_scalars",
            static_cast<size_t>(GRMHDGridRadiationModel<RealT>::NumDerivedScalars) *
            model.n1 * model.n2 * model.n3);

        const RealT mp = RealT(1.67262171e-24);
        const RealT me = RealT(9.1093826e-28);
        const RealT mp_me = mp / me;
        const RealT game = RealT(4) / RealT(3);
        const RealT gamp = RealT(5) / RealT(3);
        GRMHDGridRadiationModel<RealT> device_model = model;
        Kokkos::parallel_for(
            "KPOLARISIharmDerivedScalars",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, ncell),
            KOKKOS_LAMBDA(const int p) {
                const int k = p % device_model.n3;
                const int j = (p / device_model.n3) % device_model.n2;
                const int i = p / (device_model.n2 * device_model.n3);

                const RealT x1 = device_model.startx1 + (RealT(i) + RealT(0.5)) * device_model.dx1;
                const RealT x2 = device_model.startx2 + (RealT(j) + RealT(0.5)) * device_model.dx2;
                const RealT r = Kokkos::exp(x1);
                const RealT th = device_model.fmks_theta_from_x2(x1, x2);
                RealT gcov_nat[ndim][ndim];
                RealT gcon_nat[ndim][ndim];
                device_model.native_metric(x1, x2, r, th, gcov_nat, gcon_nat);

                const RealT rho = device_model.prim(0, i, j, k);
                const RealT uu = device_model.prim(1, i, j, k);
                const RealT U1 = device_model.prim(2, i, j, k);
                const RealT U2 = device_model.prim(3, i, j, k);
                const RealT U3 = device_model.prim(4, i, j, k);
                const RealT B1 = device_model.prim(5, i, j, k);
                const RealT B2 = device_model.prim(6, i, j, k);
                const RealT B3 = device_model.prim(7, i, j, k);

                Vec4<RealT> vnat(RealT(0), U1, U2, U3);
                RealT spatial_norm = RealT(0);
                for (int a = 1; a < ndim; ++a) {
                    for (int b = 1; b < ndim; ++b) {
                        spatial_norm += gcov_nat[a][b] * vnat[a] * vnat[b];
                    }
                }
                const RealT vfac = Kokkos::sqrt(max_val(-RealT(1) / gcon_nat[0][0] *
                                                                 (RealT(1) + abs_val(spatial_norm)), RealT(0)));
                Vec4<RealT> ucon;
                ucon[0] = -vfac * gcon_nat[0][0];
                for (int a = 1; a < ndim; ++a) {
                    ucon[a] = vnat[a] - vfac * gcon_nat[0][a];
                }
                Vec4<RealT> ucov;
                for (int a = 0; a < ndim; ++a) {
                    RealT sum = RealT(0);
                    for (int b = 0; b < ndim; ++b) sum += gcov_nat[a][b] * ucon[b];
                    ucov[a] = sum;
                }
                const Vec4<RealT> Bnat(RealT(0), B1, B2, B3);
                RealT udotB = RealT(0);
                for (int a = 1; a < ndim; ++a) udotB += ucov[a] * Bnat[a];
                Vec4<RealT> bcon;
                bcon[0] = udotB;
                for (int a = 1; a < ndim; ++a) {
                    bcon[a] = (Bnat[a] + ucon[a] * udotB) / max_val(ucon[0], RealT(1e-300));
                }
                Vec4<RealT> bcov;
                for (int a = 0; a < ndim; ++a) {
                    RealT sum = RealT(0);
                    for (int b = 0; b < ndim; ++b) sum += gcov_nat[a][b] * bcon[b];
                    bcov[a] = sum;
                }
                RealT bsq = RealT(0);
                for (int a = 0; a < ndim; ++a) bsq += bcon[a] * bcov[a];
                bsq = max_val(abs_val(bsq), RealT(1e-40));
                const RealT beta = (device_model.gam - RealT(1)) * uu / (RealT(0.5) * bsq);
                const RealT sigma = bsq / max_val(rho, RealT(1e-300));
                const RealT betasq = beta * beta / max_val(device_model.beta_crit * device_model.beta_crit, RealT(1e-40));
                const RealT trat = (device_model.trat_large * betasq + device_model.trat_small) / (RealT(1) + betasq);
                const RealT thetae_unit = mp_me * (game - RealT(1)) * (gamp - RealT(1)) /
                    ((gamp - RealT(1)) + (game - RealT(1)) * trat);

                device_model.derived_scalars(device_model.derived_index(GRMHDGridRadiationModel<RealT>::DerivedNe, i, j, k)) =
                    rho * device_model.rho_unit_cgs() / (mp + me) * device_model.ne_factor;
                device_model.derived_scalars(device_model.derived_index(GRMHDGridRadiationModel<RealT>::DerivedThetae, i, j, k)) =
                    thetae_unit * uu / max_val(rho, RealT(1e-300));
                device_model.derived_scalars(device_model.derived_index(GRMHDGridRadiationModel<RealT>::DerivedB, i, j, k)) =
                    Kokkos::sqrt(bsq) * device_model.b_unit_cgs();
                device_model.derived_scalars(device_model.derived_index(GRMHDGridRadiationModel<RealT>::DerivedSigma, i, j, k)) = sigma;
                device_model.derived_scalars(device_model.derived_index(GRMHDGridRadiationModel<RealT>::DerivedBeta, i, j, k)) = beta;
            });
        Kokkos::fence();
        report_timing(opt.timing, "iharm_derived_precompute", phase_timer.seconds());
    } else {
        report_timing(opt.timing, "iharm_derived_precompute", 0.0);
    }
    if (opt.resample_spherical_ks_precomputed && opt.resample_spherical_ks_primitives) {
        throw std::runtime_error("choose only one iharm Spherical KS resample mode");
    }
    if (opt.resample_spherical_ks_precomputed) {
        model = resample_to_spherical_ks_precomputed(model, opt);
    } else if (opt.resample_spherical_ks_primitives) {
        model = resample_to_spherical_ks_primitives(model, opt);
    }
    report_timing(opt.timing, "iharm_load_total", total_timer.seconds());
    return model;
}

} // namespace kpolaris
