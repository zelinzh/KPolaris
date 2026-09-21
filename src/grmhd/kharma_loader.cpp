#include "grmhd/kharma_loader.hpp"
#include "grmhd/ddc_frame_resolver.hpp"

#include <algorithm>
#include <array>
#ifdef KPOLARIS_DDC_COMPACT_CUDA
#include "ddc_gpu_reconstruct.cuh"
#endif
#include <tuple>
#include <cctype>
#include <cmath>
#include <iostream>
#include <syncstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <H5Cpp.h>
#include <Kokkos_Core.hpp>

#include "geometry/kerr_schild_spherical.hpp"

namespace kpolaris {
namespace {

using RealT = DefaultReal;

bool h5_link_exists(H5::H5File& file, const std::string& path) {
    return H5Lexists(file.getId(), path.c_str(), H5P_DEFAULT) > 0;
}

bool h5_attr_exists(H5::H5Object& object, const std::string& name) {
    return H5Aexists(object.getId(), name.c_str()) > 0;
}

void report_timing(int enabled, const char* name, double seconds) {
    if (enabled) {
        std::osyncstream(std::cout) << "timing " << name << ' ' << seconds << " s\n";
    }
}

std::string trim_copy(const std::string& s) {
    size_t first = 0;
    while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first]))) ++first;
    size_t last = s.size();
    while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1]))) --last;
    return s.substr(first, last - first);
}

std::string lowercase_copy(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

bool parse_block_key_value(const std::string& text,
                           const std::string& block,
                           const std::string& key,
                           std::string& value) {
    const std::string tag = "<" + block + ">";
    const size_t block_pos = text.find(tag);
    if (block_pos == std::string::npos) return false;
    const size_t begin = block_pos + tag.size();
    size_t end = text.find('<', begin);
    if (end == std::string::npos) end = text.size();

    std::istringstream in(text.substr(begin, end - begin));
    std::string line;
    while (std::getline(in, line)) {
        const size_t comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        line = trim_copy(line);
        if (line.empty()) continue;
        if (line.rfind(key, 0) != 0) continue;
        if (line.size() > key.size()) {
            const char after = line[key.size()];
            if (!(std::isspace(static_cast<unsigned char>(after)) || after == '=')) {
                continue;
            }
        }
        const size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        value = trim_copy(line.substr(eq + 1));
        return !value.empty();
    }
    return false;
}

int par_int(const std::string& par, const std::string& block, const std::string& key, int def) {
    std::string value;
    if (!parse_block_key_value(par, block, key, value)) return def;
    return std::stoi(value);
}

double par_double(const std::string& par, const std::string& block, const std::string& key, double def) {
    std::string value;
    if (!parse_block_key_value(par, block, key, value)) return def;
    return std::stod(value);
}

std::string par_string(const std::string& par,
                       const std::string& block,
                       const std::string& key,
                       const std::string& def) {
    std::string value;
    if (!parse_block_key_value(par, block, key, value)) return def;
    return value;
}

std::string read_attr_string(H5::H5Object& object, const std::string& name) {
    H5::Attribute attr = object.openAttribute(name);
    H5::DataType type = attr.getDataType();
    if (H5Tis_variable_str(type.getId()) > 0) {
        std::string out;
        attr.read(type, out);
        return out;
    }
    const size_t n = type.getSize();
    std::vector<char> buffer(n + 1, '\0');
    attr.read(type, buffer.data());
    buffer[n] = '\0';
    return std::string(buffer.data());
}

template<class T>
T read_attr_numeric(H5::H5Object& object, const std::string& name, const H5::PredType& type) {
    H5::Attribute attr = object.openAttribute(name);
    T value{};
    attr.read(type, &value);
    return value;
}

template<class T>
std::vector<T> read_attr_numeric_array(H5::H5Object& object,
                                       const std::string& name,
                                       const H5::PredType& type) {
    H5::Attribute attr = object.openAttribute(name);
    H5::DataSpace space = attr.getSpace();
    const hssize_t npoints = space.getSimpleExtentNpoints();
    if (npoints <= 0) {
        throw std::runtime_error("empty KHARMA attribute: " + name);
    }
    std::vector<T> values(static_cast<size_t>(npoints));
    attr.read(type, values.data());
    return values;
}

template<class T>
std::vector<T> read_dataset_1d(H5::H5File& file,
                               const std::string& path,
                               const H5::PredType& type,
                               std::vector<hsize_t>* dims_out = nullptr) {
    H5::DataSet ds = file.openDataSet(path);
    H5::DataSpace space = ds.getSpace();
    const int rank = space.getSimpleExtentNdims();
    if (rank <= 0) {
        throw std::runtime_error("invalid KHARMA dataset rank: " + path);
    }
    std::vector<hsize_t> dims(static_cast<size_t>(rank), 0);
    space.getSimpleExtentDims(dims.data());
    size_t count = 1;
    for (hsize_t d : dims) {
        if (d > std::numeric_limits<size_t>::max() / count) {
            throw std::runtime_error(
                "KHARMA dataset element count overflows size_t: " + path);
        }
        count *= static_cast<size_t>(d);
    }
    std::vector<T> values(count);
    ds.read(values.data(), type);
    if (dims_out) *dims_out = std::move(dims);
    return values;
}

size_t checked_product(size_t left, size_t right, const char* description) {
    if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
        throw std::runtime_error(std::string("KHARMA ") + description + " overflows size_t");
    }
    return left * right;
}

template<class Buffer>
void require_finite_values(const Buffer& values, const std::string& path) {
    for (float value : values) {
        if (!std::isfinite(static_cast<double>(value))) {
            throw std::runtime_error("KHARMA dataset contains nonfinite primitive values: " + path);
        }
    }
}

bool dimensions_equal(const std::vector<hsize_t>& actual,
                      std::initializer_list<hsize_t> expected) {
    return actual.size() == expected.size() &&
           std::equal(actual.begin(), actual.end(), expected.begin());
}

void apply_common_options(GRMHDGridRadiationModel<RealT>& model,
                          const KHARMALoadOptions& opt) {
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
}

template<class BlockView, class RawView>
void assemble_kharma_primitives(GRMHDGridRadiationModel<RealT>& model,
    const Kokkos::DefaultExecutionSpace& exec, BlockView block_order_device,
    RawView rho_device, RawView uu_device, RawView uvec_device, RawView bvec_device,
    size_t nmb, size_t cells_per_mb, int nx1_mb, int nx2_mb, int nx3_mb, int reverse_field) {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    GRMHDGridRadiationModel<RealT> device_model = model;
    const RealT field_sign = reverse_field ? RealT(-1) : RealT(1);
    Kokkos::parallel_for(
        "KPOLARISKharmaAssemblePrims",
        Kokkos::RangePolicy<ExecSpace, Kokkos::IndexType<size_t>>(exec, 0, nmb * cells_per_mb),
        KOKKOS_LAMBDA(const size_t p) {
            const size_t mb = p / cells_per_mb;
            const size_t local = p - mb * cells_per_mb;
            const int ib = static_cast<int>(local % static_cast<size_t>(nx1_mb));
            const int jb = static_cast<int>((local / static_cast<size_t>(nx1_mb)) % static_cast<size_t>(nx2_mb));
            const int kb = static_cast<int>(local / (static_cast<size_t>(nx1_mb) * static_cast<size_t>(nx2_mb)));
            const int lx1 = static_cast<int>(block_order_device(mb * 3u + 0u));
            const int lx2 = static_cast<int>(block_order_device(mb * 3u + 1u));
            const int lx3 = static_cast<int>(block_order_device(mb * 3u + 2u));
            const int i = lx1 * nx1_mb + ib;
            const int j = lx2 * nx2_mb + jb;
            const int k = lx3 * nx3_mb + kb;
            const size_t scalar_index = mb * cells_per_mb + local;
            device_model.prims(device_model.prim_index(0, i, j, k)) = RealT(rho_device(scalar_index));
            device_model.prims(device_model.prim_index(1, i, j, k)) = RealT(uu_device(scalar_index));
            const size_t vec_block = mb * 3u * cells_per_mb + local;
            device_model.prims(device_model.prim_index(2, i, j, k)) = RealT(uvec_device(vec_block));
            device_model.prims(device_model.prim_index(3, i, j, k)) = RealT(uvec_device(vec_block + cells_per_mb));
            device_model.prims(device_model.prim_index(4, i, j, k)) = RealT(uvec_device(vec_block + 2u * cells_per_mb));
            device_model.prims(device_model.prim_index(5, i, j, k)) = field_sign * RealT(bvec_device(vec_block));
            device_model.prims(device_model.prim_index(6, i, j, k)) = field_sign * RealT(bvec_device(vec_block + cells_per_mb));
            device_model.prims(device_model.prim_index(7, i, j, k)) = field_sign * RealT(bvec_device(vec_block + 2u * cells_per_mb));
        });
}

void precompute_derived_scalars(GRMHDGridRadiationModel<RealT>& model,
                                int timing,
                                Kokkos::Timer& timer,
                                const Kokkos::DefaultExecutionSpace& exec = {},
                                bool synchronize = true,
                                Kokkos::View<RealT*> geometry = {}) {
    if (!model.has_derived_scalars) {
        report_timing(timing, "kharma_derived_precompute", 0.0);
        return;
    }
    if (model.derived_scalars.extent(0) != static_cast<size_t>(GRMHDGridRadiationModel<RealT>::NumDerivedScalars) * model.n1 * model.n2 * model.n3)
    model.derived_scalars = Kokkos::View<RealT*>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "kharma_derived_scalars"),
        static_cast<size_t>(GRMHDGridRadiationModel<RealT>::NumDerivedScalars) *
        model.n1 * model.n2 * model.n3);

    const RealT mp = RealT(1.67262171e-24);
    const RealT me = RealT(9.1093826e-28);
    const RealT mp_me = mp / me;
    const RealT game = RealT(4) / RealT(3);
    const RealT gamp = RealT(5) / RealT(3);
    GRMHDGridRadiationModel<RealT> device_model = model;
    const int ncell = model.n1 * model.n2 * model.n3;
    Kokkos::parallel_for(
        "KPOLARISKharmaDerivedScalars",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(exec, 0, ncell),
        KOKKOS_LAMBDA(const int p) {
            const int k = p % device_model.n3;
            const int j = (p / device_model.n3) % device_model.n2;
            const int i = p / (device_model.n2 * device_model.n3);

            const RealT x1 = device_model.startx1 + (RealT(i) + RealT(0.5)) * device_model.dx1;
            const RealT x2 = device_model.startx2 + (RealT(j) + RealT(0.5)) * device_model.dx2;
            const RealT r = Kokkos::exp(x1);
            const RealT th = device_model.fmks_theta_from_x2(x1, x2);
            RealT gcov_nat[ndim][ndim];
            RealT gcon0_nat[ndim];
            if (geometry.extent(0)) {
                const size_t offset = (static_cast<size_t>(i) * device_model.n2 + j) * 20;
                for (int a = 0; a < ndim; ++a) {
                    gcon0_nat[a] = geometry(offset + 16 + a);
                    for (int b = 0; b < ndim; ++b) gcov_nat[a][b] = geometry(offset + 4*a + b);
                }
            } else {
                device_model.native_fluid_metric(x1, x2, r, th, gcov_nat, gcon0_nat);
            }

            const RealT rho = device_model.prim(0, i, j, k);
            const RealT uu = device_model.prim(1, i, j, k);
            const Vec4<RealT> vnat(RealT(0),
                                   device_model.prim(2, i, j, k),
                                   device_model.prim(3, i, j, k),
                                   device_model.prim(4, i, j, k));
            RealT spatial_norm = RealT(0);
            for (int a = 1; a < ndim; ++a) {
                for (int b = 1; b < ndim; ++b) {
                    spatial_norm += gcov_nat[a][b] * vnat[a] * vnat[b];
                }
            }
            const RealT vfac = Kokkos::sqrt(max_val(-RealT(1) / gcon0_nat[0] *
                                                    (RealT(1) + abs_val(spatial_norm)), RealT(0)));
            Vec4<RealT> ucon;
            ucon[0] = -vfac * gcon0_nat[0];
            for (int a = 1; a < ndim; ++a) {
                ucon[a] = vnat[a] - vfac * gcon0_nat[a];
            }
            const Vec4<RealT> ucov = device_model.lower_native(gcov_nat, ucon);
            const Vec4<RealT> Bnat(RealT(0),
                                   device_model.prim(5, i, j, k),
                                   device_model.prim(6, i, j, k),
                                   device_model.prim(7, i, j, k));
            RealT udotB = RealT(0);
            for (int a = 1; a < ndim; ++a) udotB += ucov[a] * Bnat[a];
            Vec4<RealT> bcon;
            bcon[0] = udotB;
            for (int a = 1; a < ndim; ++a) {
                bcon[a] = (Bnat[a] + ucon[a] * udotB) / max_val(ucon[0], RealT(1e-300));
            }
            const Vec4<RealT> bcov = device_model.lower_native(gcov_nat, bcon);
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
                max_val(thetae_unit * uu / max_val(rho, RealT(1e-300)), RealT(1e-3));
            device_model.derived_scalars(device_model.derived_index(GRMHDGridRadiationModel<RealT>::DerivedB, i, j, k)) =
                Kokkos::sqrt(bsq) * device_model.b_unit_cgs();
            device_model.derived_scalars(device_model.derived_index(GRMHDGridRadiationModel<RealT>::DerivedSigma, i, j, k)) = sigma;
            device_model.derived_scalars(device_model.derived_index(GRMHDGridRadiationModel<RealT>::DerivedBeta, i, j, k)) = beta;
        });
    if (synchronize) exec.fence();
    report_timing(timing, "kharma_derived_precompute", timer.seconds());
    timer.reset();
}

template<class RealT>
GRMHDGridRadiationModel<RealT> resample_to_spherical_ks_primitives(
    const GRMHDGridRadiationModel<RealT>& source,
    const KHARMALoadOptions& opt) {
    Kokkos::Timer timer;
    GRMHDGridRadiationModel<RealT> target = source;
    const RealT pi = RealT(3.141592653589793238462643383279502884);
    const int n1 = opt.resample_n1 > 0 ? opt.resample_n1 : source.n1;
    const int n2 = opt.resample_n2 > 0 ? opt.resample_n2 : source.n2;
    const int n3 = opt.resample_n3 > 0 ? opt.resample_n3 : source.n3;
    const RealT rin = opt.resample_r_in > RealT(0) ? opt.resample_r_in : source.r_in;
    const RealT rout = opt.resample_r_out > RealT(0) ? opt.resample_r_out : source.r_out;
    if (!(n1 > 1 && n2 > 1 && n3 > 0 && rin > RealT(0) && rout > rin)) {
        throw std::runtime_error("invalid Spherical KS primitive resample grid for kharma");
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
    target.prims = Kokkos::View<RealT*>("kharma_sks_resampled_prims",
        static_cast<size_t>(8) * target.n1 * target.n2 * target.n3);
    if (target.has_derived_scalars) {
        target.derived_scalars = Kokkos::View<RealT*>("kharma_sks_resampled_derived_scalars",
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
        "KPOLARISKharmaResampleSphericalKSPrimitives",
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
    report_timing(opt.timing, "kharma_resample_spherical_ks_primitives", timer.seconds());
    return target;
}

template<class RealT>
GRMHDGridRadiationModel<RealT> resample_to_spherical_ks_precomputed(
    const GRMHDGridRadiationModel<RealT>& source,
    const KHARMALoadOptions& opt) {
    Kokkos::Timer timer;
    GRMHDGridRadiationModel<RealT> target = source;
    const RealT pi = RealT(3.141592653589793238462643383279502884);
    const int n1 = opt.resample_n1 > 0 ? opt.resample_n1 : source.n1;
    const int n2 = opt.resample_n2 > 0 ? opt.resample_n2 : source.n2;
    const int n3 = opt.resample_n3 > 0 ? opt.resample_n3 : source.n3;
    const RealT rin = opt.resample_r_in > RealT(0) ? opt.resample_r_in : source.r_in;
    const RealT rout = opt.resample_r_out > RealT(0) ? opt.resample_r_out : source.r_out;
    if (!(n1 > 1 && n2 > 1 && n3 > 0 && rin > RealT(0) && rout > rin)) {
        throw std::runtime_error("invalid Spherical KS resample grid for kharma");
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
    target.fluid_states = typename GRMHDGridRadiationModel<RealT>::FluidStateView("kharma_sks_precomputed_fluid_state",
        static_cast<size_t>(GRMHDGridRadiationModel<RealT>::NumFluidStateScalars) *
        target.n1 * target.n2 * target.n3);

    using ExecSpace = Kokkos::DefaultExecutionSpace;
    const int ncell = target.n1 * target.n2 * target.n3;
    GRMHDGridRadiationModel<RealT> source_device = source;
    GRMHDGridRadiationModel<RealT> target_device = target;
    KerrSchildSphericalMetric<RealT> metric(RealT(1), source.spin);
    Kokkos::parallel_for(
        "KPOLARISKharmaResampleSphericalKSPrecomputed",
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
    report_timing(opt.timing, "kharma_resample_spherical_ks_precomputed", timer.seconds());
    return target;
}


KHARMAStagedDump initialize_kharma_staged_metadata(
    size_t num_meshblocks,
    int nx1_mb,
    int nx2_mb,
    int nx3_mb,
    const std::string& par_text,
    const KHARMALoadOptions& opt) {
    if (num_meshblocks == 0 || nx1_mb <= 0 || nx2_mb <= 0 || nx3_mb <= 0) {
        throw std::runtime_error("invalid KHARMA meshblock metadata");
    }
    if (par_text.empty()) {
        throw std::runtime_error("KHARMA parameter text is empty");
    }
    KHARMAStagedDump staged;
    staged.num_meshblocks = num_meshblocks;
    staged.nx1_mb = nx1_mb;
    staged.nx2_mb = nx2_mb;
    staged.nx3_mb = nx3_mb;
    GRMHDGridRadiationModel<RealT>& model = staged.model;
    model.n1 = par_int(par_text, "parthenon/mesh", "nx1", 0);
    model.n2 = par_int(par_text, "parthenon/mesh", "nx2", 0);
    model.n3 = par_int(par_text, "parthenon/mesh", "nx3", 0);
    if (model.n1 <= 0 || model.n2 <= 0 || model.n3 <= 0) {
        throw std::runtime_error("failed to parse KHARMA mesh dimensions from /Input/File parameter text");
    }
    model.startx1 = RealT(par_double(par_text, "parthenon/mesh", "x1min", 0.0));
    model.startx2 = RealT(par_double(par_text, "parthenon/mesh", "x2min", 0.0));
    model.startx3 = RealT(par_double(par_text, "parthenon/mesh", "x3min", 0.0));
    const RealT x1max = RealT(par_double(par_text, "parthenon/mesh", "x1max", 0.0));
    const RealT x2max = RealT(par_double(par_text, "parthenon/mesh", "x2max", 1.0));
    const RealT x3max = RealT(par_double(par_text, "parthenon/mesh", "x3max", 6.2831853071795864769));
    model.dx1 = (x1max - model.startx1) / RealT(model.n1);
    model.dx2 = (x2max - model.startx2) / RealT(model.n2);
    model.dx3 = (x3max - model.startx3) / RealT(model.n3);
    model.spin = RealT(par_double(par_text, "coordinates", "a", 0.0));
    model.gam = RealT(par_double(par_text, "GRMHD", "gamma", 1.666666666666667));
    model.r_in = RealT(par_double(par_text, "coordinates", "r_in", Kokkos::exp(model.startx1)));
    model.r_out = RealT(par_double(par_text, "coordinates", "r_out", Kokkos::exp(x1max)));
    model.hslope = RealT(par_double(par_text, "coordinates", "hslope", 1.0));
    const auto finite = [](RealT value) {
        return std::isfinite(static_cast<double>(value));
    };
    if (!finite(model.startx1) || !finite(model.startx2) || !finite(model.startx3)) {
        throw std::runtime_error("KHARMA grid starts must be finite");
    }
    if (!finite(model.dx1) || !finite(model.dx2) || !finite(model.dx3) ||
        model.dx1 <= RealT(0) || model.dx2 <= RealT(0) || model.dx3 <= RealT(0)) {
        throw std::runtime_error("KHARMA grid spacings must be finite and positive");
    }
    if (!finite(model.spin) || std::abs(model.spin) > RealT(1)) {
        throw std::runtime_error("KHARMA spin must be finite and lie in [-1, 1]");
    }
    if (!finite(model.gam) || model.gam <= RealT(1)) {
        throw std::runtime_error("KHARMA gamma must be finite and greater than 1");
    }
    if (!finite(model.r_in) || !finite(model.r_out) || model.r_in <= RealT(0) ||
        model.r_out <= model.r_in) {
        throw std::runtime_error(
            "KHARMA radial bounds must be finite, positive, and increasing");
    }
    if (!finite(model.hslope)) {
        throw std::runtime_error("KHARMA coordinate hslope must be finite");
    }
    const std::string transform = lowercase_copy(
        par_string(par_text, "coordinates", "transform", "mks"));
    if (transform.find("fmks") != std::string::npos ||
        transform.find("mmks") != std::string::npos) {
        model.mks_smooth = RealT(par_double(par_text, "coordinates", "mks_smooth", 0.5));
        model.poly_xt = RealT(par_double(par_text, "coordinates", "poly_xt", 0.82));
        model.poly_alpha = RealT(par_double(par_text, "coordinates", "poly_alpha", 14.0));
        if (!finite(model.mks_smooth) || !finite(model.poly_xt) ||
            !finite(model.poly_alpha) ||
            model.poly_xt <= RealT(0) || model.poly_alpha <= RealT(-1)) {
            throw std::runtime_error(
                "KHARMA FMKS/MMKS mapping parameters must be finite with poly_alpha > -1 and poly_xt > 0");
        }
        const RealT pi = RealT(3.141592653589793238462643383279502884);
        model.poly_norm = RealT(0.5) * pi /
            (RealT(1) + RealT(1) / (model.poly_alpha + RealT(1)) /
             Kokkos::pow(model.poly_xt, model.poly_alpha));
        if (!finite(model.poly_norm) || model.poly_norm <= RealT(0)) {
            throw std::runtime_error(
                "KHARMA FMKS/MMKS mapping normalization must be finite and positive");
        }
        model.native_coordinate_transform = 0;
        model.data_coordinate_system = static_cast<int>(CoordinateSystem::FMKS);
    } else if (transform.find("mks") != std::string::npos) {
        model.mks_smooth = RealT(0);
        model.poly_xt = RealT(1);
        model.poly_alpha = RealT(0);
        model.poly_norm = RealT(0);
        model.native_coordinate_transform = 1;
        model.data_coordinate_system = static_cast<int>(CoordinateSystem::MKS);
    } else {
        throw std::runtime_error("unsupported KHARMA coordinate transform: " + transform);
    }
    model.radial_coordinate_log = 1;
    apply_common_options(model, opt);
    return staged;
}

void validate_kharma_staged_layout(const KHARMAStagedDump& staged) {
    const auto& model = staged.model;
    const size_t nmb = staged.num_meshblocks;
    size_t cells_per_mb = checked_product(static_cast<size_t>(staged.nx1_mb),
                                          static_cast<size_t>(staged.nx2_mb),
                                          "meshblock cell count");
    cells_per_mb = checked_product(cells_per_mb, static_cast<size_t>(staged.nx3_mb),
                                   "meshblock cell count");
    size_t global_cells = checked_product(static_cast<size_t>(model.n1),
                                          static_cast<size_t>(model.n2),
                                          "global cell count");
    global_cells = checked_product(global_cells, static_cast<size_t>(model.n3),
                                   "global cell count");
    if (global_cells > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("KHARMA global cell count exceeds the supported int index range");
    }
    const size_t staged_cells = checked_product(nmb, cells_per_mb,
                                                "staged cell count");
    const size_t location_values = checked_product(nmb, 3, "logical location count");
    if (staged.block_order.size() != location_values ||
        staged.rho.size() != staged_cells ||
        staged.uu.size() != staged_cells ||
        staged.uvec.size() != checked_product(staged_cells, 3, "vector primitive count") ||
        staged.bvec.size() != checked_product(staged_cells, 3, "vector primitive count")) {
        throw std::runtime_error("KHARMA dataset sizes do not match meshblock layout");
    }
    if (model.n1 % staged.nx1_mb != 0 || model.n2 % staged.nx2_mb != 0 ||
        model.n3 % staged.nx3_mb != 0 || staged_cells != global_cells) {
        throw std::runtime_error(
            "KHARMA meshblocks must tile the declared uniform global grid exactly");
    }
    const int blocks1 = model.n1 / staged.nx1_mb;
    const int blocks2 = model.n2 / staged.nx2_mb;
    const int blocks3 = model.n3 / staged.nx3_mb;
    std::vector<unsigned char> occupied(nmb, 0);
    for (size_t mb = 0; mb < nmb; ++mb) {
        const long long lx1 = staged.block_order[mb * 3u + 0u];
        const long long lx2 = staged.block_order[mb * 3u + 1u];
        const long long lx3 = staged.block_order[mb * 3u + 2u];
        if (lx1 < 0 || lx1 >= blocks1 || lx2 < 0 || lx2 >= blocks2 ||
            lx3 < 0 || lx3 >= blocks3) {
            throw std::runtime_error("KHARMA logical meshblock location lies outside the global grid");
        }
        const size_t slot =
            (static_cast<size_t>(lx1) * static_cast<size_t>(blocks2) +
             static_cast<size_t>(lx2)) * static_cast<size_t>(blocks3) +
            static_cast<size_t>(lx3);
        if (slot >= occupied.size() || occupied[slot] != 0) {
            throw std::runtime_error("KHARMA logical meshblock locations contain a duplicate");
        }
        occupied[slot] = 1;
    }
#ifdef KPOLARIS_DDC_COMPACT_CUDA
    if (staged.compact) return;
#endif
    require_finite_values(staged.rho, "/prims.rho");
    require_finite_values(staged.uu, "/prims.u");
    require_finite_values(staged.uvec, "/prims.uvec");
    require_finite_values(staged.bvec, "/prims.B");
}

} // namespace

DefaultReal read_kharma_dump_time(const std::string& dump_path, const DDCInputOptions& ddc) {
    ddc.validate();
    const int sequence = ddc.native ? ddc_detail::native_sequence_from_path(dump_path) : -1;
    if (sequence >= 0) {
        // STAGE1 has no metadata-only request. For large libraries supply an
        // explicit time list from the codec manifest to avoid this extra decode.
        return DefaultReal(ddc_detail::request_socket_staged_frame(
            ddc.socket_path, sequence, ddc.timeout_seconds).time);
    }
    if (dump_path.empty()) {
        throw std::runtime_error("KHARMA dump path is empty");
    }
    H5::H5File file(dump_path, H5F_ACC_RDONLY);
    if (h5_link_exists(file, "/Info")) {
        H5::Group info = file.openGroup("/Info");
        if (h5_attr_exists(info, "Time")) {
            return DefaultReal(read_attr_numeric<double>(info, "Time", H5::PredType::NATIVE_DOUBLE));
        }
        if (h5_attr_exists(info, "time")) {
            return DefaultReal(read_attr_numeric<double>(info, "time", H5::PredType::NATIVE_DOUBLE));
        }
    }
    throw std::runtime_error("KHARMA dump does not contain /Info Time attribute: " + dump_path);
}

KHARMAStagedDump read_kharma_staged_dump(const KHARMALoadOptions& opt) try {
    Kokkos::Timer phase_timer;
    if (opt.dump_path.empty()) {
        throw std::runtime_error("--kharma_dump is required for --model=kharma");
    }

    H5::H5File file(opt.dump_path, H5F_ACC_RDONLY);
    H5::Group info = file.openGroup("/Info");
    H5::Group input = file.openGroup("/Input");
    const long long num_meshblocks =
        read_attr_numeric<long long>(info, "NumMeshBlocks", H5::PredType::NATIVE_LLONG);
    const std::vector<long long> meshblock_size =
        read_attr_numeric_array<long long>(info, "MeshBlockSize", H5::PredType::NATIVE_LLONG);
    if (meshblock_size.size() < 3 || num_meshblocks <= 0 ||
        meshblock_size[0] <= 0 || meshblock_size[1] <= 0 || meshblock_size[2] <= 0 ||
        meshblock_size[0] > std::numeric_limits<int>::max() ||
        meshblock_size[1] > std::numeric_limits<int>::max() ||
        meshblock_size[2] > std::numeric_limits<int>::max()) {
        throw std::runtime_error("invalid KHARMA Info NumMeshBlocks/MeshBlockSize");
    }
    const int nx1_mb = static_cast<int>(meshblock_size[0]);
    const int nx2_mb = static_cast<int>(meshblock_size[1]);
    const int nx3_mb = static_cast<int>(meshblock_size[2]);
    const std::string par_text = read_attr_string(input, "File");
    if (par_text.empty()) {
        throw std::runtime_error("KHARMA /Input File attribute is empty");
    }

    KHARMAStagedDump staged = initialize_kharma_staged_metadata(
        static_cast<size_t>(num_meshblocks), nx1_mb, nx2_mb, nx3_mb, par_text, opt);
    report_timing(opt.timing, "kharma_header", phase_timer.seconds());
    phase_timer.reset();

    if (!h5_link_exists(file, "/Blocks/loc.lx123") ||
        !h5_link_exists(file, "/prims.rho") ||
        !h5_link_exists(file, "/prims.u") ||
        !h5_link_exists(file, "/prims.uvec") ||
        !h5_link_exists(file, "/prims.B")) {
        throw std::runtime_error("KHARMA new-format phdf requires Blocks/loc.lx123 and prims.rho/u/uvec/B");
    }

    std::vector<hsize_t> loc_dims;
    staged.block_order = read_dataset_1d<long long>(file, "/Blocks/loc.lx123", H5::PredType::NATIVE_LLONG, &loc_dims);
    std::vector<hsize_t> rho_dims;
    staged.rho = read_dataset_1d<float>(file, "/prims.rho", H5::PredType::NATIVE_FLOAT, &rho_dims);
    std::vector<hsize_t> uu_dims;
    staged.uu = read_dataset_1d<float>(file, "/prims.u", H5::PredType::NATIVE_FLOAT, &uu_dims);
    std::vector<hsize_t> uvec_dims;
    staged.uvec = read_dataset_1d<float>(file, "/prims.uvec", H5::PredType::NATIVE_FLOAT, &uvec_dims);
    std::vector<hsize_t> bvec_dims;
    staged.bvec = read_dataset_1d<float>(file, "/prims.B", H5::PredType::NATIVE_FLOAT, &bvec_dims);
    report_timing(opt.timing, "kharma_prims_hdf5_read", phase_timer.seconds());

    const hsize_t h_nmb = static_cast<hsize_t>(staged.num_meshblocks);
    const hsize_t h_nx1 = static_cast<hsize_t>(nx1_mb);
    const hsize_t h_nx2 = static_cast<hsize_t>(nx2_mb);
    const hsize_t h_nx3 = static_cast<hsize_t>(nx3_mb);
    if (!dimensions_equal(loc_dims, {h_nmb, 3}) ||
        !dimensions_equal(rho_dims, {h_nmb, h_nx3, h_nx2, h_nx1}) ||
        !dimensions_equal(uu_dims, {h_nmb, h_nx3, h_nx2, h_nx1}) ||
        !dimensions_equal(uvec_dims, {h_nmb, 3, h_nx3, h_nx2, h_nx1}) ||
        !dimensions_equal(bvec_dims, {h_nmb, 3, h_nx3, h_nx2, h_nx1})) {
        throw std::runtime_error(
            "KHARMA dataset dimensions do not match the documented meshblock layout");
    }
    validate_kharma_staged_layout(staged);
    return staged;
} catch (const H5::Exception& error) {
    throw std::runtime_error("Failed to read KHARMA dump '" + opt.dump_path +
                             "': " + error.getDetailMsg());
}

KHARMAStagedDump read_kharma_staged_dump_resolved(const KHARMALoadOptions& opt, KHARMAStagedDump reuse) {
    opt.ddc.validate();
    const int sequence = opt.ddc.native ? ddc_detail::native_sequence_from_path(opt.dump_path) : -1;
    if (sequence < 0) return read_kharma_staged_dump(opt);
    Kokkos::Timer timer;
    DDCNativeFrame buffers;
    buffers.block_order=std::move(reuse.block_order); buffers.rho=std::move(reuse.rho);
    buffers.uu=std::move(reuse.uu); buffers.uvec=std::move(reuse.uvec); buffers.bvec=std::move(reuse.bvec);
    DDCNativeFrame frame = ddc_detail::request_socket_staged_frame(
        opt.ddc.socket_path, sequence, opt.ddc.timeout_seconds, std::move(buffers));
    if (std::isfinite(opt.ddc_expected_time)) {
        const double expected = static_cast<double>(opt.ddc_expected_time);
        const double tolerance = 16 * std::numeric_limits<DefaultReal>::epsilon() *
            std::max(1.0, std::max(std::abs(expected), std::abs(frame.time)));
        if (std::abs(frame.time - expected) > tolerance) {
            throw std::runtime_error("DDC frame time disagrees with slow_light_time_list: " + opt.dump_path);
        }
    }
    KHARMAStagedDump staged = initialize_kharma_staged_metadata(
        frame.num_meshblocks, frame.nx1_mb, frame.nx2_mb, frame.nx3_mb,
        frame.par_text, opt);
    staged.block_order = std::move(frame.block_order);
    staged.rho = std::move(frame.rho);
    staged.uu = std::move(frame.uu);
    staged.uvec = std::move(frame.uvec);
    staged.bvec = std::move(frame.bvec);
#ifdef KPOLARIS_DDC_COMPACT_CUDA
    staged.compact = std::move(frame.compact);
#endif
    validate_kharma_staged_layout(staged);
    report_timing(opt.timing, "kharma_prims_ddc_native_socket_read", timer.seconds());
    return staged;
}

Kokkos::View<RealT*> cache_kharma_geometry(const GRMHDRadiationModel<DefaultReal>& first,
                                         const Kokkos::DefaultExecutionSpace& exec) {
    using Exec=Kokkos::DefaultExecutionSpace;
        auto geometry = Kokkos::View<RealT*>(Kokkos::view_alloc(Kokkos::WithoutInitializing,"slow_grid_geometry"),static_cast<size_t>(first.n1)*first.n2*20);
        const auto g=geometry; const auto m=first;
        Kokkos::parallel_for("KPOLARISKharmaCacheGeometry",Kokkos::RangePolicy<Exec>(exec,0,first.n1*first.n2),KOKKOS_LAMBDA(int p) {
            const int i=p/m.n2,j=p%m.n2;
            const RealT x1=m.startx1+(RealT(i)+RealT(0.5))*m.dx1;
            const RealT x2=m.startx2+(RealT(j)+RealT(0.5))*m.dx2;
            const RealT r=Kokkos::exp(x1),th=m.fmks_theta_from_x2(x1,x2);
            RealT cov[ndim][ndim],con0[ndim]; m.native_fluid_metric(x1,x2,r,th,cov,con0);
            for(int a=0;a<ndim;++a) { g(size_t(p)*20+16+a)=con0[a];
                for(int b=0;b<ndim;++b) g(size_t(p)*20+4*a+b)=cov[a][b]; }
        });
        exec.fence();
    return geometry;
}

struct KHARMAUploadWorkspace::Impl {
    using Exec = Kokkos::DefaultExecutionSpace;
    using Model = GRMHDRadiationModel<DefaultReal>;
#ifdef KOKKOS_ENABLE_CUDA
    using Pinned = Kokkos::CudaHostPinnedSpace;
    struct Stream {
        cudaStream_t value{};
        Stream() { KOKKOS_IMPL_CUDA_SAFE_CALL(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
        ~Stream() { cudaStreamDestroy(value); }
    } stream;
    Exec exec{stream.value};
    struct CopyEvents {
        cudaEvent_t values[2]{};
        CopyEvents() {
            for(auto& value:values) KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventCreateWithFlags(&value,cudaEventDisableTiming));
        }
        ~CopyEvents() { for(auto value:values) cudaEventDestroy(value); }
    } copies_done;
#else
    using Pinned = Kokkos::HostSpace;
    Exec exec;
#endif
    Model grid;
    std::vector<Model> slots;
    Kokkos::View<float*, Pinned> host[2];
    std::array<ddc_transport::HostBuffer<float>, 4> direct_host[2];
    Kokkos::View<float*> raw;
    Kokkos::View<long long*, Pinned> host_order;
    Kokkos::View<long long*> device_order;
    std::vector<long long> order;
    Kokkos::View<RealT*> geometry;
#ifdef KPOLARIS_DDC_COMPACT_CUDA
    std::shared_ptr<ddc_transport::CompactFrame> compact_host[2];
    std::map<uint64_t, Kokkos::View<float*>> compact_anchors;
    std::vector<Kokkos::View<float*>> compact_anchor_spares;
    Kokkos::View<unsigned char*> compact_packet;
    Kokkos::View<int> compact_invalid{"ddc_compact_invalid"};
#endif
    size_t next_host = 0;
    size_t staged_cells = 0;
    int mb1 = 0, mb2 = 0, mb3 = 0;
    static auto geometry_key(const Model& m) {
        return std::make_tuple(m.n1,m.n2,m.n3,m.spin,m.startx1,m.startx2,m.startx3,
            m.dx1,m.dx2,m.dx3,m.hslope,m.mks_smooth,m.poly_alpha,m.poly_xt,m.poly_norm,
            m.native_coordinate_transform,m.data_coordinate_system,m.radial_coordinate_log);
    }
    Impl(const Model& first, size_t count) : grid(first) {
        if (!count || first.precomputed_fluid_state ||
            (first.data_coordinate_system != static_cast<int>(CoordinateSystem::FMKS) &&
             first.data_coordinate_system != static_cast<int>(CoordinateSystem::MKS)))
            throw std::invalid_argument("KHARMA upload pipeline requires a native MKS/FMKS grid");
        const size_t cells = static_cast<size_t>(first.n1)*first.n2*first.n3;
        slots.reserve(count);
        for (size_t i=0;i<count;++i) {
            Model model = first;
            model.prims = Kokkos::View<RealT*>(Kokkos::view_alloc(Kokkos::WithoutInitializing,"slow_prims_slot"),8*cells);
            if (model.has_derived_scalars)
                model.derived_scalars = Kokkos::View<RealT*>(Kokkos::view_alloc(Kokkos::WithoutInitializing,"slow_derived_slot"),Model::NumDerivedScalars*cells);
            slots.push_back(model);
        }
        geometry=cache_kharma_geometry(first,exec);
    }
    ~Impl() { exec.fence(); }
};

KHARMAUploadWorkspace::KHARMAUploadWorkspace(const GRMHDRadiationModel<DefaultReal>& first, size_t slots)
    : impl_(std::make_unique<Impl>(first, slots)) {}
KHARMAUploadWorkspace::~KHARMAUploadWorkspace() = default;
void KHARMAUploadWorkspace::fence() {
    impl_->exec.fence();
#ifdef KPOLARIS_DDC_COMPACT_CUDA
    if (!impl_->compact_anchors.empty()) {
        int invalid = 0;
        Kokkos::deep_copy(invalid,impl_->compact_invalid);
        if (invalid) throw std::runtime_error("DDC GPU reconstruction produced nonfinite primitives");
    }
#endif
}
GRMHDRadiationModel<DefaultReal> KHARMAUploadWorkspace::prepare(
    const KHARMAStagedDump& staged, const KHARMALoadOptions& opt, size_t slot) {
    auto& w=*impl_;
    if (slot>=w.slots.size()) throw std::out_of_range("KHARMA upload slot outside pool");
    if (Impl::geometry_key(staged.model)!=Impl::geometry_key(w.grid) ||
        opt.resample_spherical_ks_precomputed || opt.resample_spherical_ks_primitives)
        throw std::invalid_argument("KHARMA grid changed or resampling requested; disable slow_light_pipeline");
    const size_t cells=staged.rho.size();
    if (!w.staged_cells) {
        w.staged_cells=cells; w.mb1=staged.nx1_mb; w.mb2=staged.nx2_mb; w.mb3=staged.nx3_mb;
        w.raw=Kokkos::View<float*>(Kokkos::view_alloc(Kokkos::WithoutInitializing,"slow_raw_device"),8*cells);
        for(auto& h:w.host) h=Kokkos::View<float*,Impl::Pinned>(Kokkos::view_alloc(Kokkos::WithoutInitializing,"slow_raw_pinned"),8*cells);
        w.host_order=Kokkos::View<long long*,Impl::Pinned>(Kokkos::view_alloc(Kokkos::WithoutInitializing,"slow_order_pinned"),staged.block_order.size());
        w.device_order=Kokkos::View<long long*>(Kokkos::view_alloc(Kokkos::WithoutInitializing,"slow_order_device"),staged.block_order.size());
    }
    if (cells!=w.staged_cells || staged.nx1_mb!=w.mb1 || staged.nx2_mb!=w.mb2 || staged.nx3_mb!=w.mb3 || staged.block_order.size()!=w.host_order.extent(0))
        throw std::invalid_argument("KHARMA meshblock layout changed; disable slow_light_pipeline");
    // Wait only for the previous DMA from this pinned buffer. Waiting for
    // the entire upload stream would serialize CPU filling with both queued
    // snapshots' assembly and derived-scalar kernels.
    const size_t host_slot=w.next_host%2;
    if (w.next_host>=2) {
#ifdef KOKKOS_ENABLE_CUDA
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventSynchronize(w.copies_done.values[host_slot]));
#else
        w.exec.fence();
#endif
    }
    ++w.next_host;
    auto h=w.host[host_slot];
    const bool direct = staged.rho.pinned() && staged.uu.pinned() && staged.uvec.pinned() && staged.bvec.pinned();
    bool compact = false;
#ifdef KPOLARIS_DDC_COMPACT_CUDA
    compact = bool(staged.compact);
#endif
    if (!direct && !compact) {
        std::copy(staged.rho.begin(),staged.rho.end(),h.data());
        std::copy(staged.uu.begin(),staged.uu.end(),h.data()+cells);
        std::copy(staged.uvec.begin(),staged.uvec.end(),h.data()+2*cells);
        std::copy(staged.bvec.begin(),staged.bvec.end(),h.data()+5*cells);
    }
    if (w.order!=staged.block_order) {
        w.exec.fence();
        w.order=staged.block_order;
        std::copy(w.order.begin(),w.order.end(),w.host_order.data());
        Kokkos::deep_copy(w.exec,w.device_order,w.host_order);
    }
#ifdef KPOLARIS_DDC_COMPACT_CUDA
    if (compact) {
        w.compact_host[host_slot] = staged.compact;
        const auto& frame = *staged.compact;
        const auto& meta = frame.metadata;
        for (auto iterator = w.compact_anchors.begin(); iterator != w.compact_anchors.end();) {
            if (iterator->first != meta.start && iterator->first != meta.end) {
                w.compact_anchor_spares.push_back(iterator->second);
                iterator = w.compact_anchors.erase(iterator);
            } else ++iterator;
        }
        auto upload_anchor = [&](uint64_t sequence, const auto& source) {
            if (w.compact_anchors.count(sequence)) return;
            Kokkos::View<float*> target;
            if (!w.compact_anchor_spares.empty()) {
                target = w.compact_anchor_spares.back(); w.compact_anchor_spares.pop_back();
                if (target.extent(0) != source.size()) throw std::runtime_error("Changed GPU anchor layout");
            } else target = Kokkos::View<float*>(Kokkos::view_alloc(Kokkos::WithoutInitializing,"ddc_compact_anchor"),source.size());
            const Kokkos::View<const float*, Impl::Pinned, Kokkos::MemoryTraits<Kokkos::Unmanaged>> input(source.data(),source.size());
            Kokkos::deep_copy(w.exec,target,input); w.compact_anchors.emplace(sequence,target);
        };
        upload_anchor(meta.start,frame.first); upload_anchor(meta.end,frame.last);
        if (w.compact_packet.extent(0) < frame.packet.size()) {
            w.exec.fence();
            w.compact_packet = Kokkos::View<unsigned char*>(Kokkos::view_alloc(Kokkos::WithoutInitializing,"ddc_compact_packet"),((frame.packet.size()+2097151)/2097152)*2097152);
        }
        if (frame.packet.size()) {
            const Kokkos::View<const unsigned char*, Impl::Pinned, Kokkos::MemoryTraits<Kokkos::Unmanaged>> input(frame.packet.data(),frame.packet.size());
            auto target = Kokkos::subview(w.compact_packet,std::make_pair(size_t(0),frame.packet.size()));
            Kokkos::deep_copy(w.exec,target,input);
        }
        ddc_transport::reconstruct(w.raw.data(),w.compact_packet.data(),w.compact_anchors.at(meta.start).data(),
            w.compact_anchors.at(meta.end).data(),meta,cells,w.mb3,w.mb2,w.mb1,w.exec.cuda_stream(),w.compact_invalid.data());
    } else
#endif
    if (direct) {
        w.direct_host[host_slot] = {staged.rho, staged.uu, staged.uvec, staged.bvec};
        const bool contiguous = staged.rho.data()+cells == staged.uu.data() &&
            staged.uu.data()+cells == staged.uvec.data() && staged.uvec.data()+3*cells == staged.bvec.data();
        if (contiguous) {
            const Kokkos::View<const float*, Impl::Pinned, Kokkos::MemoryTraits<Kokkos::Unmanaged>> input(staged.rho.data(),8*cells);
            Kokkos::deep_copy(w.exec,w.raw,input);
        } else {
            size_t offset = 0;
            for (const auto& source : w.direct_host[host_slot]) {
                const Kokkos::View<const float*, Impl::Pinned, Kokkos::MemoryTraits<Kokkos::Unmanaged>> input(source.data(), source.size());
                auto target = Kokkos::subview(w.raw, std::make_pair(offset, offset + source.size()));
                Kokkos::deep_copy(w.exec, target, input);
                offset += source.size();
            }
        }
    } else Kokkos::deep_copy(w.exec,w.raw,h);
#ifdef KOKKOS_ENABLE_CUDA
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventRecord(w.copies_done.values[host_slot],w.exec.cuda_stream()));
#endif
    auto model=staged.model; apply_common_options(model,opt);
    if (bool(model.has_derived_scalars)!=bool(w.grid.has_derived_scalars))
        throw std::invalid_argument("KHARMA derived-scalar layout changed in upload pipeline");
    model.prims=w.slots[slot].prims; model.derived_scalars=w.slots[slot].derived_scalars;
    auto range=[&](size_t a,size_t b){return Kokkos::subview(w.raw,std::make_pair(a*cells,b*cells));};
    assemble_kharma_primitives(model,w.exec,w.device_order,range(0,1),range(1,2),range(2,5),range(5,8),
        staged.num_meshblocks,size_t(w.mb1)*w.mb2*w.mb3,w.mb1,w.mb2,w.mb3,opt.reverse_field);
    Kokkos::Timer timer; precompute_derived_scalars(model,0,timer,w.exec,false,w.geometry);
    return model;
}

GRMHDRadiationModel<DefaultReal> materialize_kharma_model_from_staged(const KHARMAStagedDump& staged,
                                                                       const KHARMALoadOptions& opt) {
    Kokkos::Timer total_timer;
    Kokkos::Timer phase_timer;
#if KPOLARIS_GRMHD_REQUIRE_DERIVED_SCALARS
    if (!opt.interpolate_derived_scalars) {
        throw std::runtime_error(
            "this KHARMA build requires --kharma_interpolate_derived_scalars=1");
    }
#endif
    GRMHDGridRadiationModel<RealT> model = staged.model;
    apply_common_options(model, opt);
#ifdef KPOLARIS_DDC_COMPACT_CUDA
    if (staged.compact) {
        KHARMAUploadWorkspace workspace(model,1);
        auto result = workspace.prepare(staged,opt,0);
        workspace.fence();
        return result;
    }
#endif

    const size_t nmb = staged.num_meshblocks;
    const size_t cells_per_mb = static_cast<size_t>(staged.nx1_mb) *
                                static_cast<size_t>(staged.nx2_mb) *
                                static_cast<size_t>(staged.nx3_mb);
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    // Staged arrays already own contiguous host buffers. Copy directly from
    // unmanaged views instead of allocating, clearing and filling host mirrors.
    // Every device element is overwritten before use, so initialization would
    // add another full memory pass for each snapshot in a slow-light stream.
    auto upload = [](const auto& source, const char* label) {
        using Value = typename std::decay_t<decltype(source)>::value_type;
        Kokkos::View<Value*, ExecSpace> device(
            Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), source.size());
        const Kokkos::View<const Value*, Kokkos::HostSpace,
            Kokkos::MemoryTraits<Kokkos::Unmanaged>> host(source.data(), source.size());
        Kokkos::deep_copy(device, host);
        return device;
    };
    auto block_order_device = upload(staged.block_order, "kharma_block_order");
    auto rho_device = upload(staged.rho, "kharma_rho_raw");
    auto uu_device = upload(staged.uu, "kharma_uu_raw");
    auto uvec_device = upload(staged.uvec, "kharma_uvec_raw");
    auto bvec_device = upload(staged.bvec, "kharma_bvec_raw");

    model.prims = Kokkos::View<RealT*>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing, "kharma_prims"),
        static_cast<size_t>(8) * model.n1 * model.n2 * model.n3);
    assemble_kharma_primitives(model, ExecSpace{}, block_order_device, rho_device, uu_device,
        uvec_device, bvec_device, nmb, cells_per_mb, staged.nx1_mb, staged.nx2_mb, staged.nx3_mb, opt.reverse_field);
    Kokkos::fence();
    report_timing(opt.timing, "kharma_prims_assemble_upload", phase_timer.seconds());
    phase_timer.reset();

    precompute_derived_scalars(model, opt.timing, phase_timer);
    if (opt.resample_spherical_ks_precomputed && opt.resample_spherical_ks_primitives) {
        throw std::runtime_error("choose only one kharma Spherical KS resample mode");
    }
    if (opt.resample_spherical_ks_precomputed) {
        model = resample_to_spherical_ks_precomputed(model, opt);
    } else if (opt.resample_spherical_ks_primitives) {
        model = resample_to_spherical_ks_primitives(model, opt);
    }
    report_timing(opt.timing, "kharma_materialize_total", total_timer.seconds());
    return model;
}

GRMHDRadiationModel<DefaultReal> load_kharma_model_from_phdf(const KHARMALoadOptions& opt) {
    Kokkos::Timer total_timer;
    KHARMAStagedDump staged = read_kharma_staged_dump_resolved(opt);
    GRMHDRadiationModel<DefaultReal> model = materialize_kharma_model_from_staged(staged, opt);
    report_timing(opt.timing, "kharma_load_total", total_timer.seconds());
    return model;
}

} // namespace kpolaris
