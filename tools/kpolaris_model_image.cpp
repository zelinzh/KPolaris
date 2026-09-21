#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <set>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <H5Cpp.h>

#include "common/types.hpp"
#include "common/vec.hpp"
#include "common/version.hpp"
#include "geodesic/pass_a.hpp"
#include "kpolaris_model_image_options.hpp"
#include "kpolaris_response_options.hpp"
#include "kpolaris_response_output.hpp"
#include "kpolaris_model_image_output.hpp"
#include "kpolaris_conventions_output.hpp"
#include "geometry/kerr_boyer_lindquist.hpp"
#include "geometry/kerr_fmks.hpp"
#include "geometry/kerr_schild_cartesian.hpp"
#include "geometry/kerr_schild_spherical.hpp"
#include "image/driver.hpp"
#include "image/split_driver.hpp"
#include "image/multifrequency.hpp"

#ifndef KPOLARIS_DEFAULT_MODEL
#define KPOLARIS_DEFAULT_MODEL "riaf"
#endif
#ifndef KPOLARIS_ENABLE_MODEL_RIAF
#define KPOLARIS_ENABLE_MODEL_RIAF 1
#endif
#ifndef KPOLARIS_ENABLE_MODEL_BINARY_RIAF
#define KPOLARIS_ENABLE_MODEL_BINARY_RIAF 0
#endif
#ifndef KPOLARIS_ENABLE_MODEL_TORUS
#define KPOLARIS_ENABLE_MODEL_TORUS 1
#endif
#ifndef KPOLARIS_ENABLE_MODEL_IHARM
#define KPOLARIS_ENABLE_MODEL_IHARM 1
#endif
#ifndef KPOLARIS_ENABLE_MODEL_KHARMA
#define KPOLARIS_ENABLE_MODEL_KHARMA 1
#endif
#ifndef KPOLARIS_ENABLE_MODEL_ATHENAK
#define KPOLARIS_ENABLE_MODEL_ATHENAK 1
#endif
#ifndef KPOLARIS_ENABLE_MODEL_BHAC
#define KPOLARIS_ENABLE_MODEL_BHAC 1
#endif
#ifndef KPOLARIS_ENABLE_MODEL_HAMR
#define KPOLARIS_ENABLE_MODEL_HAMR 1
#endif
#ifndef KPOLARIS_ENABLE_CSV_OUTPUT
#define KPOLARIS_ENABLE_CSV_OUTPUT 0
#endif
#ifndef KPOLARIS_ENABLE_ANALYSIS_MODE
#define KPOLARIS_ENABLE_ANALYSIS_MODE 0
#endif
#ifndef KPOLARIS_ENABLE_SLOW_LIGHT
#define KPOLARIS_ENABLE_SLOW_LIGHT 1
#endif
#if KPOLARIS_ENABLE_ANALYSIS_MODE
#include "image/analysis_driver.hpp"
#endif
#if KPOLARIS_ENABLE_SLOW_LIGHT
#include "image/slow_light_driver.hpp"
#include "image/slow_light_batch_driver.hpp"
#endif

#if KPOLARIS_ENABLE_MODEL_RIAF
#include "model/riaf.hpp"
#include "image/riaf.hpp"
#endif
#if KPOLARIS_ENABLE_MODEL_BINARY_RIAF
#include "geometry/binary_trajectory_hdf5.hpp"
#include "model/binary_riaf.hpp"
#include "image/binary_riaf.hpp"
#endif
#if KPOLARIS_ENABLE_MODEL_TORUS
#include "image/torus.hpp"
#endif

#if KPOLARIS_ENABLE_MODEL_IHARM
#include "grmhd/iharm_loader.hpp"
#endif
#if KPOLARIS_ENABLE_MODEL_KHARMA
#include "grmhd/kharma_loader.hpp"
#endif
#if KPOLARIS_ENABLE_MODEL_ATHENAK
#include "grmhd/athenak_loader.hpp"
#endif
#if KPOLARIS_ENABLE_MODEL_BHAC
#include "grmhd/bhac_loader.hpp"
#endif
#if KPOLARIS_ENABLE_MODEL_HAMR
#include "grmhd/hamr_loader.hpp"
#endif
#if KPOLARIS_ENABLE_MODEL_IHARM || KPOLARIS_ENABLE_MODEL_KHARMA
#include "image/grmhd.hpp"
#include "model/grmhd.hpp"
#endif

namespace {

using Real = kpolaris::DefaultReal;

#if KPOLARIS_ENABLE_MODEL_BINARY_RIAF
class Sha256 {
  public:
    Sha256()
        : state_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u} {}

    void update(const unsigned char* data, size_t size) {
        total_bytes_ += static_cast<std::uint64_t>(size);
        while (size > 0) {
            const size_t take = std::min(size, block_.size() - used_);
            std::copy_n(data, take, block_.begin() +
                        static_cast<std::ptrdiff_t>(used_));
            used_ += take;
            data += take;
            size -= take;
            if (used_ == block_.size()) {
                transform(block_.data());
                used_ = 0;
            }
        }
    }

    std::string finish() {
        const std::uint64_t message_bits = total_bytes_ * 8u;
        block_[used_++] = 0x80u;
        if (used_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(used_),
                      block_.end(), 0u);
            transform(block_.data());
            used_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(used_),
                  block_.begin() + 56, 0u);
        for (int byte = 0; byte < 8; ++byte) {
            block_[63 - byte] = static_cast<unsigned char>(
                message_bits >> (8 * byte));
        }
        transform(block_.data());
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (std::uint32_t word : state_) {
            output << std::setw(8) << word;
        }
        return output.str();
    }

  private:
    static std::uint32_t rotate_right(std::uint32_t value, int shift) {
        return (value >> shift) | (value << (32 - shift));
    }

    void transform(const unsigned char* input) {
        static constexpr std::array<std::uint32_t, 64> constants = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
            0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
            0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
            0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
            0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};
        std::array<std::uint32_t, 64> words{};
        for (int i = 0; i < 16; ++i) {
            const int offset = 4 * i;
            words[static_cast<size_t>(i)] =
                (static_cast<std::uint32_t>(input[offset]) << 24) |
                (static_cast<std::uint32_t>(input[offset + 1]) << 16) |
                (static_cast<std::uint32_t>(input[offset + 2]) << 8) |
                static_cast<std::uint32_t>(input[offset + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t x = words[static_cast<size_t>(i - 15)];
            const std::uint32_t y = words[static_cast<size_t>(i - 2)];
            const std::uint32_t s0 = rotate_right(x, 7) ^
                rotate_right(x, 18) ^ (x >> 3);
            const std::uint32_t s1 = rotate_right(y, 17) ^
                rotate_right(y, 19) ^ (y >> 10);
            words[static_cast<size_t>(i)] =
                words[static_cast<size_t>(i - 16)] + s0 +
                words[static_cast<size_t>(i - 7)] + s1;
        }
        std::uint32_t a = state_[0], b = state_[1], c = state_[2],
                      d = state_[3], e = state_[4], f = state_[5],
                      g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t sum1 = rotate_right(e, 6) ^
                rotate_right(e, 11) ^ rotate_right(e, 25);
            const std::uint32_t choose = (e & f) ^ ((~e) & g);
            const std::uint32_t temporary1 = h + sum1 + choose +
                constants[static_cast<size_t>(i)] +
                words[static_cast<size_t>(i)];
            const std::uint32_t sum0 = rotate_right(a, 2) ^
                rotate_right(a, 13) ^ rotate_right(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temporary2 = sum0 + majority;
            h = g; g = f; f = e; e = d + temporary1;
            d = c; c = b; b = a; a = temporary1 + temporary2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_;
    std::array<unsigned char, 64> block_{};
    size_t used_ = 0;
    std::uint64_t total_bytes_ = 0;
};

std::string sha256_file(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "failed to open binary trajectory for SHA-256: " + path);
    }
    Sha256 digest;
    std::array<char, 65536> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            digest.update(reinterpret_cast<const unsigned char*>(buffer.data()),
                          static_cast<size_t>(count));
        }
    }
    if (!input.eof()) {
        throw std::runtime_error(
            "failed while hashing binary trajectory: " + path);
    }
    return digest.finish();
}

bool hdf5_link_exists(H5::H5File& file, const std::string& path) {
    return H5Lexists(file.getId(), path.c_str(), H5P_DEFAULT) > 0;
}

Real read_first_hdf5_number(H5::H5File& file,
                            const std::string& path) {
    if (!hdf5_link_exists(file, path)) {
        throw std::runtime_error(
            "cannot infer trajectory mass scale: missing " + path);
    }
    H5::DataSet dataset = file.openDataSet(path);
    H5::DataSpace file_space = dataset.getSpace();
    const int rank = file_space.getSimpleExtentNdims();
    if (rank < 0) {
        throw std::runtime_error("invalid HDF5 dataspace for " + path);
    }
    double value = 0.0;
    if (rank == 0) {
        dataset.read(&value, H5::PredType::NATIVE_DOUBLE);
    } else {
        std::vector<hsize_t> dimensions(static_cast<size_t>(rank));
        file_space.getSimpleExtentDims(dimensions.data());
        for (hsize_t dimension : dimensions) {
            if (dimension == 0) {
                throw std::runtime_error("empty HDF5 mass dataset " + path);
            }
        }
        std::vector<hsize_t> start(static_cast<size_t>(rank), 0);
        std::vector<hsize_t> count(static_cast<size_t>(rank), 1);
        file_space.selectHyperslab(H5S_SELECT_SET, count.data(), start.data());
        H5::DataSpace memory_space(rank, count.data());
        dataset.read(&value, H5::PredType::NATIVE_DOUBLE,
                     memory_space, file_space);
    }
    if (!std::isfinite(value)) {
        throw std::runtime_error("non-finite initial mass in " + path);
    }
    return static_cast<Real>(value);
}

Real infer_legacy_trajectory_mass_scale(const std::string& path) {
    H5::H5File file(path, H5F_ACC_RDONLY);
    const bool full = hdf5_link_exists(file, "m1_full") &&
                      hdf5_link_exists(file, "m2_full");
    const std::string first = full ? "m1_full" : "m1";
    const std::string second = full ? "m2_full" : "m2";
    const Real reference_mass = read_first_hdf5_number(file, first) +
                                read_first_hdf5_number(file, second);
    if (!(reference_mass > Real(0))) {
        throw std::runtime_error(
            "legacy trajectory initial total mass must be positive");
    }
    return Real(1) / reference_mass;
}

kpolaris::BinaryTrajectoryHdf5Format trajectory_format_from_options(
    const Options& opt, bool& native_file) {
    H5::H5File file(opt.binary_trajectory_file, H5F_ACC_RDONLY);
    native_file = hdf5_link_exists(file, "/trajectory");
    if (opt.binary_trajectory_format == "native") {
        return kpolaris::BinaryTrajectoryHdf5Format::native;
    }
    if (opt.binary_trajectory_format == "combi_ressler") {
        return kpolaris::BinaryTrajectoryHdf5Format::combi_ressler;
    }
    return kpolaris::BinaryTrajectoryHdf5Format::automatic;
}

void configure_tabulated_binary_trajectory(
    Options& opt,
    kpolaris::BinaryRIAFRadiationModel<Real>& model) {
    const bool tabulated =
        opt.binary_trajectory_model == "paper_cbwaves_4pn_local" ||
        opt.binary_trajectory_model == "table";
    if (!tabulated) return;

    bool native_file = false;
    kpolaris::BinaryTrajectoryLoadOptions load_options;
    load_options.format = trajectory_format_from_options(opt, native_file);
    const bool paper_model =
        opt.binary_trajectory_model == "paper_cbwaves_4pn_local";
    if (paper_model) {
        if (!native_file ||
            load_options.format ==
                kpolaris::BinaryTrajectoryHdf5Format::combi_ressler) {
            throw std::runtime_error(
                "paper_cbwaves_4pn_local requires a native KPolaris trajectory; use binary_trajectory_model=table for legacy or generic files");
        }
        load_options.format =
            kpolaris::BinaryTrajectoryHdf5Format::native;
        load_options.require_source_verified = true;
        load_options.require_merger_separation_reached = true;
        load_options.required_generator =
            "scripts/generate_paper_sks_trajectory.py";
        load_options.required_generator_version =
            kpolaris::paper_trajectory_generator_version;
        load_options.required_merger_reach_contract =
            kpolaris::paper_merger_reach_contract;
        load_options.required_trajectory_model =
            "paper_cbwaves_4pn_local";
        load_options.required_pn_terms =
            "PN,2PN,SO,SS,RR,PNSO,3PN,1RR,2PNSO,RRSO,RRSS,4PN";
        load_options.required_source_doi =
            "10.5281/zenodo.10841021";
        load_options.required_source_verification =
            "verified-zenodo-record-10841021";
        load_options.required_upstream_cbwaves_sha256 =
            "4a094a9bbac3b2bc142015b70afd5dc939898b7b38a34b05917dd94e94829a9a";
        load_options.required_patched_cbwaves_sha256 =
            "0a28a193f0964164876968b159b9df0c261190f4a605de138592878e967c6067";
    }
    if (native_file && opt.binary_trajectory_mass_scale > Real(0)) {
        throw std::runtime_error(
            "native KPolaris trajectories already use M_ref=1; binary_trajectory_mass_scale is only for legacy Combi--Ressler files");
    }
    if (!native_file ||
        load_options.format ==
            kpolaris::BinaryTrajectoryHdf5Format::combi_ressler) {
        load_options.geometric_unit_scale =
            opt.binary_trajectory_mass_scale > Real(0) ?
            Real(1) / opt.binary_trajectory_mass_scale :
            infer_legacy_trajectory_mass_scale(opt.binary_trajectory_file);
    }
    const kpolaris::LoadedBinaryTrajectory loaded =
        kpolaris::load_binary_trajectory_hdf5(
            opt.binary_trajectory_file, load_options);
    model.trajectory = loaded.provider;
    opt.binary_trajectory_format =
        loaded.metadata.format ==
                kpolaris::BinaryTrajectoryHdf5Format::native ?
            "native" : "combi_ressler";
    opt.binary_trajectory_sha256 = sha256_file(opt.binary_trajectory_file);
    // Camera x^0=0 maps to this absolute table time.  Along a past-directed
    // Pass A ray x^0 decreases, so the metric is evaluated at retarded events.
    model.observation_time = opt.binary_observation_time;
    model.trajectory_time_offset = opt.binary_trajectory_time_offset;

    opt.binary_trajectory_schema = loaded.metadata.schema;
    opt.binary_trajectory_gauge = loaded.metadata.position_gauge;
    opt.binary_trajectory_generator = loaded.metadata.generator;
    opt.binary_trajectory_generator_version =
        loaded.metadata.generator_version;
    opt.binary_trajectory_merger_reach_contract =
        loaded.metadata.merger_reach_contract;
    opt.binary_trajectory_samples =
        static_cast<int>(loaded.metadata.sample_count);
    opt.binary_trajectory_t_min = loaded.metadata.time_min;
    opt.binary_trajectory_t_max = loaded.metadata.time_max;
    opt.binary_transition_start = loaded.metadata.transition_start_time;
    opt.binary_transition_end = loaded.metadata.transition_end_time;
    opt.binary_trajectory_exact_remnant =
        loaded.metadata.has_exact_postmerger_tail ? 1 : 0;
    opt.binary_trajectory_declared_model =
        loaded.metadata.trajectory_model;
    opt.binary_trajectory_pn_terms = loaded.metadata.pn_terms;
    opt.binary_trajectory_source_verified =
        loaded.metadata.source_verified;
    opt.binary_trajectory_source_doi = loaded.metadata.source_doi;
    opt.binary_trajectory_pn_4pn_scope = loaded.metadata.pn_4pn_scope;
    opt.binary_trajectory_source_verification =
        loaded.metadata.source_verification;
    opt.binary_trajectory_upstream_cbwaves_sha256 =
        loaded.metadata.upstream_cbwaves_sha256;
    opt.binary_trajectory_patched_cbwaves_sha256 =
        loaded.metadata.patched_cbwaves_sha256;
    opt.binary_trajectory_merger_separation_reached =
        loaded.metadata.merger_separation_reached;
    opt.binary_trajectory_status = loaded.metadata.trajectory_status;
    opt.binary_trajectory_boost_velocity_model =
        loaded.metadata.boost_velocity_model;
    opt.binary_trajectory_worldline_velocity_consistent =
        loaded.metadata.worldline_velocity_consistent;
    opt.binary_trajectory_future_extension =
        loaded.metadata.future_postmerger_extension_enabled ? 1 : 0;
    if (opt.binary_trajectory_interpolation != "auto" &&
        opt.binary_trajectory_interpolation != loaded.metadata.interpolation) {
        throw std::runtime_error(
            "binary_trajectory_interpolation does not match the validated file contract (requested '" +
            opt.binary_trajectory_interpolation + "', file declares '" +
            loaded.metadata.interpolation + "')");
    }
    opt.binary_trajectory_interpolation = loaded.metadata.interpolation;
    for (const std::string& warning : loaded.metadata.warnings) {
        std::cerr << "Binary trajectory warning: " << warning << "\n";
    }
}
#endif

#if KPOLARIS_ENABLE_ANALYSIS_MODE || KPOLARIS_ENABLE_SLOW_LIGHT
kpolaris::AnalysisConfig<Real> make_analysis_config(
    const Options& opt, const kpolaris::PassAParams<Real>& pass_a) {
    kpolaris::AnalysisConfig<Real> config;
    config.radial_bins = opt.analysis_radial_bins;
    config.radial_min = opt.analysis_radial_min > Real(0) ?
        opt.analysis_radial_min : pass_a.inner_radius;
    config.radial_max = opt.analysis_radial_max > Real(0) ?
        opt.analysis_radial_max : pass_a.outer_radius;
    config.formation_fraction = opt.analysis_formation_fraction;
    config.response = make_response_config(opt, config.radial_bins, config.radial_min, config.radial_max);
    if (!(config.radial_min > Real(0)) ||
        !(config.radial_max > config.radial_min)) {
        throw std::runtime_error(
            "resolved analysis radial bounds must be positive and increasing");
    }
    return config;
}
#endif

[[maybe_unused]] Real safe_inner_radius(Real requested, bool explicit_request, Real model_r_in, Real spin) {
    const Real horizon = Real(1) + std::sqrt(std::max<Real>(Real(0), Real(1) - spin * spin));
    const Real floor_radius = std::max(model_r_in, horizon * Real(1.05));
    if (explicit_request && requested > Real(0)) {
        return std::max(requested, horizon * Real(1.05));
    }
    return floor_radius;
}

std::string metric_name_for_options(const Options& opt) {
    return opt.model == "binary_riaf" ?
        "Dynamic Superposed-Kerr-Schild (Combi-Ressler prescribed binary ansatz)" :
        metric_name_from_coordinate(opt.coordinate);
}

bool model_defines_inner_radius(const std::string& model) {
    return model == "binary_riaf" || model == "iharm" || model == "kharma" || model == "athenak" ||
           model == "bhac" || model == "hamr";
}

void report_timing(const Options& opt, const char* name, double seconds) {
    if (opt.timing) {
        std::cout << "timing " << name << ' ' << seconds << " s\n";
    }
}

[[maybe_unused]] std::string emission_fit_name(int emission_type) {
    switch (emission_type) {
    case 1: return "symphony_pandya_thermal";
    case 2: return "symphony_kappa";
    case 3: return "symphony_power_law";
    case 4: return "dexter_thermal";
    default: return "unspecified";
    }
}

[[maybe_unused]] int effective_emission_type(int requested, int compiled, int model_default) {
    if (compiled > 0) {
        if (requested > 0 && requested != compiled) {
            throw std::runtime_error(
                "requested emission_type=" + std::to_string(requested) +
                " conflicts with compile-time emission type " + std::to_string(compiled) +
                "; reconfigure the matching KPOLARIS_*_EMISSION_TYPE=0 option "
                "to enable runtime thermal/nonthermal selection");
        }
        return compiled;
    }
    return requested > 0 ? requested : model_default;
}

std::string bhac_cache_mode_name(int mode) {
    switch (mode) {
    case 1: return "read";
    case 2: return "write";
    case 3: return "read_write";
    default: return "off";
    }
}

const char* hamr_hslope_source(const Options& opt) {
    return opt.hamr_hslope_explicit ? "explicit" : "default_pending_dump_metadata";
}

[[maybe_unused]] void warn_pending_hamr_hslope(const Options& opt) {
    if (!opt.hamr_hslope_explicit) {
        std::cerr << "warning: hamr_hslope=" << opt.hamr_hslope
                  << " is a temporary default; no confirmed H-AMR hslope metadata field "
                     "has been identified in the dump yet. Use --hamr_hslope to override.\n";
    }
}

Real output_mbh_solar(const Options& opt) {
    if (opt.model == "iharm") return opt.iharm_mbh_solar;
    if (opt.model == "kharma") return opt.kharma_mbh_solar;
    if (opt.model == "athenak") return opt.athenak_mbh_solar;
    if (opt.model == "bhac") return opt.bhac_mbh_solar;
#if KPOLARIS_ENABLE_MODEL_HAMR
    if (opt.model == "hamr") return opt.hamr_mbh_solar;
#endif
    if (opt.model == "torus") return opt.torus_bh_mass_solar;
    return opt.riaf_mbh_solar;
}

Real output_length_unit_cgs(Real mbh_solar) {
    const Real gnewt = Real(6.6742e-8);
    const Real msun = Real(1.989e33);
    const Real cl = Real(2.99792458e10);
    return gnewt * mbh_solar * msun / (cl * cl);
}

struct FluxScale {
    int valid = 0;
    Real l_unit_cm = std::numeric_limits<Real>::quiet_NaN();
    Real dsource_cm = std::numeric_limits<Real>::quiet_NaN();
    Real pixel_solid_angle_sr = std::numeric_limits<Real>::quiet_NaN();
    Real intensity_to_flux_jy_per_pixel = std::numeric_limits<Real>::quiet_NaN();
};

FluxScale compute_flux_scale(const Options& opt) {
    FluxScale scale;
    const Real pc_cgs = Real(3.085678e18);
    const Real jy_cgs = Real(1.0e-23);
    scale.l_unit_cm = output_length_unit_cgs(output_mbh_solar(opt));
    scale.dsource_cm = opt.dsource_pc * pc_cgs;
    if (opt.dsource_pc > Real(0) && opt.image_width_x > Real(0) &&
        opt.image_width_y > Real(0) && opt.nx > 0 && opt.ny > 0) {
        const Real pixel_x_cm = opt.image_width_x * scale.l_unit_cm / Real(opt.nx);
        const Real pixel_y_cm = opt.image_width_y * scale.l_unit_cm / Real(opt.ny);
        scale.pixel_solid_angle_sr = pixel_x_cm * pixel_y_cm /
                                     (scale.dsource_cm * scale.dsource_cm);
        scale.intensity_to_flux_jy_per_pixel = scale.pixel_solid_angle_sr / jy_cgs;
        scale.valid = std::isfinite(static_cast<double>(scale.intensity_to_flux_jy_per_pixel)) &&
                      scale.intensity_to_flux_jy_per_pixel > Real(0);
    }
    return scale;
}

Real nan_if_invalid(const FluxScale& scale, Real value) {
    return scale.valid ? value : std::numeric_limits<Real>::quiet_NaN();
}

Real flux_density_jy(const FluxScale& scale, Real invariant_sum, Real frequency) {
    if (!scale.valid) {
        return std::numeric_limits<Real>::quiet_NaN();
    }
    return invariant_sum * frequency * frequency * frequency *
           scale.intensity_to_flux_jy_per_pixel;
}


#if KPOLARIS_ENABLE_MODEL_RIAF
void write_model_metadata(std::ofstream& out,
                          const kpolaris::RIAFAnalyticRadiationModel<Real>& model) {
    out << "# riaf_r_min," << model.r_min << "\n";
    out << "# riaf_r_max," << model.r_max << "\n";
    out << "# riaf_nth0," << model.nth0 << "\n";
    out << "# riaf_Te0," << model.Te0 << "\n";
    out << "# riaf_disk_h," << model.disk_h << "\n";
    out << "# riaf_pow_nth," << model.pow_nth << "\n";
    out << "# riaf_pow_T," << model.pow_T << "\n";
    out << "# riaf_ne_unit," << model.ne_unit << "\n";
    out << "# riaf_te_unit," << model.te_unit << "\n";
    out << "# riaf_mbh_solar," << model.mbh_solar << "\n";
    out << "# riaf_keplerian_factor," << model.keplerian_factor << "\n";
    out << "# riaf_infall_factor," << model.infall_factor << "\n";
    out << "# emission_type," << model.emission_type << "\n";
    out << "# emission_fit," << emission_fit_name(model.emission_type) << "\n";
    out << "# nonthermal_kappa," << model.nonthermal_kappa << "\n";
    out << "# variable_kappa," << model.variable_kappa << "\n";
    out << "# variable_kappa_min," << model.variable_kappa_min << "\n";
    out << "# variable_kappa_interp_start," << model.variable_kappa_interp_start << "\n";
    out << "# variable_kappa_max," << model.variable_kappa_max << "\n";
    out << "# powerlaw_p," << model.powerlaw_p << "\n";
    out << "# powerlaw_eta," << model.powerlaw_eta << "\n";
    out << "# powerlaw_gamma_min," << model.powerlaw_gamma_min << "\n";
    out << "# powerlaw_gamma_max," << model.powerlaw_gamma_max << "\n";
    out << "# powerlaw_gamma_cutoff," << model.powerlaw_gamma_cutoff << "\n";
    out << "# dlambda_scale," << model.dlambda_scale() << "\n";
}
#endif

#if KPOLARIS_ENABLE_MODEL_BINARY_RIAF
void write_model_metadata(std::ofstream& out,
                          const kpolaris::BinaryRIAFRadiationModel<Real>& model) {
    write_model_metadata(out, model.disk);
    out << "# spacetime,SuperposedKerrSchild\n";
    const bool table = model.trajectory.is_tabulated();
    out << "# dynamic_spacetime," << (table || model.orbit_enabled) << "\n";
    if (!table) {
        out << "# binary_mass_ratio," << model.mass_ratio << "\n";
        out << "# binary_chi1," << model.chi1 << "\n";
        out << "# binary_chi2," << model.chi2 << "\n";
        out << "# binary_reference_separation," << model.reference_separation << "\n";
        out << "# binary_reference_phase," << model.reference_phase << "\n";
        out << "# binary_reference_time," << model.reference_time << "\n";
    }
    out << "# binary_observation_time," << model.observation_time << "\n";
    out << "# binary_trajectory_time_offset,"
        << model.trajectory_time_offset << "\n";
    out << "# binary_resolved_observation_time,"
        << model.observation_time + model.trajectory_time_offset << "\n";
    out << "# binary_minimum_separation," << model.minimum_separation << "\n";
    out << "# binary_inspiral," << model.inspiral_enabled << "\n";
    out << "# binary_orbit," << model.orbit_enabled << "\n";
    out << "# binary_trajectory_mode,"
        << (model.trajectory.is_tabulated() ? "tabulated" :
            "leading_quadrupole") << "\n";
    out << "# binary_trajectory_samples," << model.trajectory.sample_count
        << "\n";
    out << "# binary_trajectory_time_min,"
        << model.trajectory.table_time_min << "\n";
    out << "# binary_trajectory_time_max,"
        << model.trajectory.table_time_max << "\n";
    out << "# binary_trajectory_exact_remnant_tail,"
        << model.trajectory.has_exact_postmerger_tail << "\n";
    if (table) {
        const auto state = model.trajectory.state(
            model.observation_time + model.trajectory_time_offset);
        out << "# binary_observation_mass1," << state.mass1 << "\n";
        out << "# binary_observation_mass2," << state.mass2 << "\n";
        out << "# binary_observation_mass_ratio_m2_over_m1,"
            << state.mass2 / state.mass1 << "\n";
        out << "# binary_observation_position1_xyz," << state.position1.x
            << ',' << state.position1.y << ',' << state.position1.z << "\n";
        out << "# binary_observation_position2_xyz," << state.position2.x
            << ',' << state.position2.y << ',' << state.position2.z << "\n";
        out << "# binary_observation_velocity1_xyz," << state.velocity1.x
            << ',' << state.velocity1.y << ',' << state.velocity1.z << "\n";
        out << "# binary_observation_velocity2_xyz," << state.velocity2.x
            << ',' << state.velocity2.y << ',' << state.velocity2.z << "\n";
        out << "# binary_observation_kerr_a1_xyz," << state.kerr_a1.x
            << ',' << state.kerr_a1.y << ',' << state.kerr_a1.z << "\n";
        out << "# binary_observation_kerr_a2_xyz," << state.kerr_a2.x
            << ',' << state.kerr_a2.y << ',' << state.kerr_a2.z << "\n";
        out << "# binary_observation_separation," << state.separation << "\n";
        out << "# binary_observation_phase," << state.phase << "\n";
        out << "# binary_observation_merger_weight,"
            << state.merger_weight << "\n";
    }
    out << "# binary_metric_derivative_step," << model.metric_derivative_step << "\n";
    out << "# binary_capture_factor," << model.capture_factor << "\n";
    out << "# binary_sampled_min_inverse_denominator,"
        << model.sampled_min_inverse_denominator << "\n";
    out << "# binary_sampled_min_fluid_slice_timelike_margin,"
        << model.sampled_min_fluid_slice_timelike_margin << "\n";
    out << "# binary_tidal_fraction," << model.tidal_fraction << "\n";
    out << "# binary_taper_start_fraction," << model.taper_start_fraction << "\n";
    out << "# binary_density_scale1," << model.density_scale1 << "\n";
    out << "# binary_density_scale2," << model.density_scale2 << "\n";
    out << "# binary_temperature_scale1," << model.temperature_scale1 << "\n";
    out << "# binary_temperature_scale2," << model.temperature_scale2 << "\n";
    out << "# binary_field_polarity1," << model.field_polarity1 << "\n";
    out << "# binary_field_polarity2," << model.field_polarity2 << "\n";
}
#endif

#if KPOLARIS_ENABLE_MODEL_TORUS
void write_model_metadata(std::ofstream& out,
                          const kpolaris::MagnetizedTorusRadiationModel<Real>& model) {
    out << "# torus_l_lambda," << model.l_lambda << "\n";
    out << "# torus_wwin," << model.wwin << "\n";
    out << "# torus_kappa," << model.kappa << "\n";
    out << "# torus_omegac," << model.omegac << "\n";
    out << "# torus_betac," << model.betac << "\n";
    out << "# torus_beta," << model.beta << "\n";
    out << "# torus_Rhigh," << model.Rhigh << "\n";
    out << "# torus_bh_mass_solar," << model.bh_mass_solar << "\n";
    out << "# torus_mdot_cgs," << model.accretion_rate_cgs << "\n";
    out << "# torus_mdot_code," << model.accretion_rate_code << "\n";
    out << "# torus_thetae_min," << model.thetae_min << "\n";
    out << "# torus_l0," << model.l0 << "\n";
    out << "# torus_rcusp," << model.rcusp << "\n";
    out << "# torus_rc," << model.rc << "\n";
    out << "# torus_r_outer," << model.r_outer << "\n";
    out << "# torus_Wc," << model.Wc << "\n";
    out << "# torus_Win," << model.Win << "\n";
    out << "# dlambda_scale," << model.dlambda_scale() << "\n";
}
#endif



#if KPOLARIS_ENABLE_MODEL_IHARM || KPOLARIS_ENABLE_MODEL_KHARMA
std::string grmhd_metadata_prefix(const std::string& model_name) {
    if (model_name == "iharm" || model_name == "kharma") {
        return model_name;
    }
    throw std::runtime_error(
        "shared GRMHD image metadata requires model=iharm or model=kharma, got: " +
        model_name);
}

void write_model_metadata(std::ofstream& out,
                          const kpolaris::GRMHDRadiationModel<Real>& model,
                          const std::string& model_name) {
    const std::string prefix = grmhd_metadata_prefix(model_name);
    out << "# grmhd_n1," << model.n1 << "\n";
    out << "# grmhd_n2," << model.n2 << "\n";
    out << "# grmhd_n3," << model.n3 << "\n";
    out << "# grmhd_M_unit," << model.M_unit << "\n";
    out << "# grmhd_mbh_solar," << model.mbh_solar << "\n";
    out << "# grmhd_trat_small," << model.trat_small << "\n";
    out << "# grmhd_trat_large," << model.trat_large << "\n";
    out << "# grmhd_beta_crit," << model.beta_crit << "\n";
    out << "# grmhd_sigma_cut," << model.sigma_cut << "\n";
    out << "# grmhd_sigma_cut_high," << model.sigma_cut_high << "\n";
    out << "# " << prefix << "_n1," << model.n1 << "\n";
    out << "# " << prefix << "_n2," << model.n2 << "\n";
    out << "# " << prefix << "_n3," << model.n3 << "\n";
    out << "# " << prefix << "_M_unit," << model.M_unit << "\n";
    out << "# " << prefix << "_mbh_solar," << model.mbh_solar << "\n";
    out << "# " << prefix << "_trat_small," << model.trat_small << "\n";
    out << "# " << prefix << "_trat_large," << model.trat_large << "\n";
    out << "# " << prefix << "_beta_crit," << model.beta_crit << "\n";
    out << "# " << prefix << "_sigma_cut," << model.sigma_cut << "\n";
    out << "# " << prefix << "_sigma_cut_high," << model.sigma_cut_high << "\n";
    out << "# emission_type," << model.emission_type << "\n";
    out << "# emission_fit," << emission_fit_name(model.emission_type) << "\n";
    out << "# nonthermal_kappa," << model.nonthermal_kappa << "\n";
    out << "# variable_kappa," << model.variable_kappa << "\n";
    out << "# variable_kappa_min," << model.variable_kappa_min << "\n";
    out << "# variable_kappa_interp_start," << model.variable_kappa_interp_start << "\n";
    out << "# variable_kappa_max," << model.variable_kappa_max << "\n";
    out << "# powerlaw_p," << model.powerlaw_p << "\n";
    out << "# powerlaw_eta," << model.powerlaw_eta << "\n";
    out << "# powerlaw_gamma_min," << model.powerlaw_gamma_min << "\n";
    out << "# powerlaw_gamma_max," << model.powerlaw_gamma_max << "\n";
    out << "# powerlaw_gamma_cutoff," << model.powerlaw_gamma_cutoff << "\n";
    out << "# grmhd_precomputed_fluid_state," << model.precomputed_fluid_state << "\n";
    out << "# grmhd_data_coordinate_system," << model.data_coordinate_system << "\n";
    out << "# grmhd_radial_coordinate_log," << model.radial_coordinate_log << "\n";
    out << "# grmhd_dlambda_scale," << model.dlambda_scale() << "\n";
    out << "# grmhd_interpolate_derived_scalars," << model.has_derived_scalars << "\n";
    out << "# " << prefix << "_interpolate_derived_scalars," << model.has_derived_scalars << "\n";
    out << "# " << prefix << "_precomputed_fluid_state," << model.precomputed_fluid_state << "\n";
    out << "# " << prefix << "_data_coordinate_system," << model.data_coordinate_system << "\n";
    out << "# " << prefix << "_radial_coordinate_log," << model.radial_coordinate_log << "\n";
    out << "# " << prefix << "_dlambda_scale," << model.dlambda_scale() << "\n";
}
#endif


#if KPOLARIS_ENABLE_MODEL_ATHENAK
void write_model_metadata(std::ofstream& out,
                          const kpolaris::AthenaKDirectRadiationModel<Real>& model) {
    out << "# athenak_backend,direct_cks_meshblocks\n";
    out << "# athenak_nblocks," << model.nblocks << "\n";
    out << "# athenak_meshblock_nx1," << model.nx1 << "\n";
    out << "# athenak_meshblock_nx2," << model.nx2 << "\n";
    out << "# athenak_meshblock_nx3," << model.nx3 << "\n";
    out << "# athenak_M_unit," << model.M_unit << "\n";
    out << "# athenak_mbh_solar," << model.mbh_solar << "\n";
    out << "# athenak_trat_small," << model.trat_small << "\n";
    out << "# athenak_trat_large," << model.trat_large << "\n";
    out << "# athenak_beta_crit," << model.beta_crit << "\n";
    out << "# athenak_sigma_cut," << model.sigma_cut << "\n";
    out << "# athenak_sigma_cut_high," << model.sigma_cut_high << "\n";
    out << "# emission_type," << model.emission_type << "\n";
    out << "# emission_fit," << emission_fit_name(model.emission_type) << "\n";
    out << "# dlambda_scale," << model.dlambda_scale() << "\n";
}
#endif


#if KPOLARIS_ENABLE_MODEL_BHAC || KPOLARIS_ENABLE_MODEL_HAMR
std::string amr_metadata_prefix(const std::string& model_name) {
    if (model_name == "bhac" || model_name == "hamr") {
        return model_name;
    }
    throw std::runtime_error(
        "shared AMR image metadata requires model=bhac or model=hamr, got: " +
        model_name);
}

std::string amr_metadata_backend(const std::string& model_name) {
    return model_name == "bhac" ? "direct_bhac_mks_amr" : "direct_hamr_mks_amr";
}

void write_model_metadata(std::ofstream& out,
                          const kpolaris::BHACAMRRadiationModel<Real>& model,
                          const std::string& model_name) {
    const std::string prefix = amr_metadata_prefix(model_name);
    out << "# " << prefix << "_backend," << amr_metadata_backend(model_name) << "\n";
    out << "# " << prefix << "_nblocks," << model.nblocks << "\n";
    out << "# " << prefix << "_meshblock_nx1," << model.nx1 << "\n";
    out << "# " << prefix << "_meshblock_nx2," << model.nx2 << "\n";
    out << "# " << prefix << "_meshblock_nx3," << model.nx3 << "\n";
    out << "# " << prefix << "_M_unit," << model.M_unit << "\n";
    out << "# " << prefix << "_mbh_solar," << model.mbh_solar << "\n";
    out << "# " << prefix << "_trat_small," << model.trat_small << "\n";
    out << "# " << prefix << "_trat_large," << model.trat_large << "\n";
    out << "# " << prefix << "_beta_crit," << model.beta_crit << "\n";
    out << "# " << prefix << "_gamma," << model.gam << "\n";
    out << "# " << prefix << "_sigma_cut," << model.sigma_cut << "\n";
    out << "# " << prefix << "_sigma_cut_high," << model.sigma_cut_high << "\n";
    if (model_name == "hamr") {
        out << "# hamr_hslope," << Real(1) - model.hslope << "\n";
        out << "# hamr_internal_hslope," << model.hslope << "\n";
    } else {
        out << "# bhac_hslope," << model.hslope << "\n";
    }
    out << "# " << prefix << "_x1_min," << model.startx1 << "\n";
    out << "# " << prefix << "_x1_max," << model.stopx1 << "\n";
    out << "# " << prefix << "_x2_min," << model.startx2 << "\n";
    out << "# " << prefix << "_x2_max," << model.stopx2 << "\n";
    out << "# " << prefix << "_x3_min," << model.startx3 << "\n";
    out << "# " << prefix << "_x3_max," << model.stopx3 << "\n";
    out << "# emission_type," << model.emission_type << "\n";
    out << "# emission_fit," << emission_fit_name(model.emission_type) << "\n";
    out << "# dlambda_scale," << model.dlambda_scale() << "\n";
}
#endif

template<class Model>
void write_model_metadata(std::ofstream& out,
                          const Model& model,
                          const std::string&) {
    write_model_metadata(out, model);
}



#if KPOLARIS_ENABLE_MODEL_RIAF
void write_effective_model_parameters(std::ofstream& out,
                                      const kpolaris::RIAFAnalyticRadiationModel<Real>& model) {
    out << "riaf_r_min=" << model.r_min << "\n";
    out << "riaf_r_max=" << model.r_max << "\n";
    out << "riaf_nth0=" << model.nth0 << "\n";
    out << "riaf_Te0=" << model.Te0 << "\n";
    out << "riaf_disk_h=" << model.disk_h << "\n";
    out << "riaf_pow_nth=" << model.pow_nth << "\n";
    out << "riaf_pow_T=" << model.pow_T << "\n";
    out << "riaf_ne_unit=" << model.ne_unit << "\n";
    out << "riaf_te_unit=" << model.te_unit << "\n";
    out << "riaf_mbh_solar=" << model.mbh_solar << "\n";
    out << "riaf_keplerian_factor=" << model.keplerian_factor << "\n";
    out << "riaf_infall_factor=" << model.infall_factor << "\n";
    out << "emission_type=" << model.emission_type << "\n";
    out << "emission_fit=" << emission_fit_name(model.emission_type) << "\n";
    out << "nonthermal_kappa=" << model.nonthermal_kappa << "\n";
    out << "variable_kappa=" << model.variable_kappa << "\n";
    out << "variable_kappa_min=" << model.variable_kappa_min << "\n";
    out << "variable_kappa_interp_start=" << model.variable_kappa_interp_start << "\n";
    out << "variable_kappa_max=" << model.variable_kappa_max << "\n";
    out << "powerlaw_p=" << model.powerlaw_p << "\n";
    out << "powerlaw_eta=" << model.powerlaw_eta << "\n";
    out << "powerlaw_gamma_min=" << model.powerlaw_gamma_min << "\n";
    out << "powerlaw_gamma_max=" << model.powerlaw_gamma_max << "\n";
    out << "powerlaw_gamma_cutoff=" << model.powerlaw_gamma_cutoff << "\n";
    out << "# derived_dlambda_scale=" << model.dlambda_scale() << "\n";
}
#endif

#if KPOLARIS_ENABLE_MODEL_BINARY_RIAF
void write_effective_model_parameters(std::ofstream& out,
                                      const kpolaris::BinaryRIAFRadiationModel<Real>& model) {
    write_effective_model_parameters(out, model.disk);
    out << "# derived_spacetime=superposed_kerr_schild\n";
    const bool table = model.trajectory.is_tabulated();
    out << "# derived_dynamic_spacetime="
        << (table || model.orbit_enabled) << "\n";
    if (!table) {
        out << "binary_mass_ratio=" << model.mass_ratio << "\n";
        out << "binary_chi1=" << model.chi1 << "\n";
        out << "binary_chi2=" << model.chi2 << "\n";
        out << "binary_reference_separation=" << model.reference_separation << "\n";
        out << "binary_reference_phase=" << model.reference_phase << "\n";
        out << "binary_reference_time=" << model.reference_time << "\n";
    }
    out << "binary_observation_time=" << model.observation_time << "\n";
    out << "binary_trajectory_time_offset="
        << model.trajectory_time_offset << "\n";
    out << "# derived_binary_resolved_observation_time="
        << model.observation_time + model.trajectory_time_offset << "\n";
    if (!table) {
        out << "binary_minimum_separation=" << model.minimum_separation << "\n";
        out << "binary_inspiral=" << model.inspiral_enabled << "\n";
    }
    out << "binary_orbit=" << model.orbit_enabled << "\n";
    out << "# derived_binary_trajectory_mode="
        << (model.trajectory.is_tabulated() ? "tabulated" :
            "leading_quadrupole") << "\n";
    out << "# derived_binary_trajectory_samples="
        << model.trajectory.sample_count << "\n";
    out << "# derived_binary_trajectory_time_min="
        << model.trajectory.table_time_min << "\n";
    out << "# derived_binary_trajectory_time_max="
        << model.trajectory.table_time_max << "\n";
    out << "# derived_binary_trajectory_exact_remnant_tail="
        << model.trajectory.has_exact_postmerger_tail << "\n";
    if (table) {
        const auto state = model.trajectory.state(
            model.observation_time + model.trajectory_time_offset);
        out << "# derived_binary_observation_mass1=" << state.mass1 << "\n";
        out << "# derived_binary_observation_mass2=" << state.mass2 << "\n";
        out << "# derived_binary_observation_separation="
            << state.separation << "\n";
        out << "# derived_binary_observation_phase=" << state.phase << "\n";
        out << "# derived_binary_observation_merger_weight="
            << state.merger_weight << "\n";
    }
    out << "binary_metric_derivative_step=" << model.metric_derivative_step << "\n";
    out << "binary_capture_factor=" << model.capture_factor << "\n";
    out << "# derived_binary_sampled_min_inverse_denominator="
        << model.sampled_min_inverse_denominator << "\n";
    out << "# derived_binary_sampled_min_fluid_slice_timelike_margin="
        << model.sampled_min_fluid_slice_timelike_margin << "\n";
    out << "binary_tidal_fraction=" << model.tidal_fraction << "\n";
    out << "binary_taper_start_fraction=" << model.taper_start_fraction << "\n";
    out << "binary_density_scale1=" << model.density_scale1 << "\n";
    out << "binary_density_scale2=" << model.density_scale2 << "\n";
    out << "binary_temperature_scale1=" << model.temperature_scale1 << "\n";
    out << "binary_temperature_scale2=" << model.temperature_scale2 << "\n";
    out << "binary_field_polarity1=" << model.field_polarity1 << "\n";
    out << "binary_field_polarity2=" << model.field_polarity2 << "\n";
}
#endif

#if KPOLARIS_ENABLE_MODEL_TORUS
void write_effective_model_parameters(std::ofstream& out,
                                      const kpolaris::MagnetizedTorusRadiationModel<Real>& model) {
    out << "torus_l_lambda=" << model.l_lambda << "\n";
    out << "torus_wwin=" << model.wwin << "\n";
    out << "torus_kappa=" << model.kappa << "\n";
    out << "torus_omegac=" << model.omegac << "\n";
    out << "torus_betac=" << model.betac << "\n";
    out << "torus_beta=" << model.beta << "\n";
    out << "torus_Rhigh=" << model.Rhigh << "\n";
    out << "torus_bh_mass_solar=" << model.bh_mass_solar << "\n";
    out << "torus_mdot_cgs=" << model.accretion_rate_cgs << "\n";
    out << "torus_mdot_code=" << model.accretion_rate_code << "\n";
    out << "torus_thetae_min=" << model.thetae_min << "\n";
    out << "scalar_transport=" << model.scalar_transport << "\n";
    out << "# derived_torus_l0=" << model.l0 << "\n";
    out << "# derived_torus_rcusp=" << model.rcusp << "\n";
    out << "# derived_torus_rc=" << model.rc << "\n";
    out << "# derived_torus_r_outer=" << model.r_outer << "\n";
    out << "# derived_torus_Wc=" << model.Wc << "\n";
    out << "# derived_torus_Win=" << model.Win << "\n";
    out << "# derived_dlambda_scale=" << model.dlambda_scale() << "\n";
}
#endif


#if KPOLARIS_ENABLE_MODEL_IHARM || KPOLARIS_ENABLE_MODEL_KHARMA
void write_effective_model_parameters(std::ofstream& out,
                                      const kpolaris::GRMHDRadiationModel<Real>& model,
                                      const std::string& model_name) {
    const std::string prefix = grmhd_metadata_prefix(model_name);
    out << prefix << "_M_unit=" << model.M_unit << "\n";
    out << prefix << "_mbh_solar=" << model.mbh_solar << "\n";
    out << prefix << "_trat_small=" << model.trat_small << "\n";
    out << prefix << "_trat_large=" << model.trat_large << "\n";
    out << prefix << "_beta_crit=" << model.beta_crit << "\n";
    out << prefix << "_sigma_cut=" << model.sigma_cut << "\n";
    out << prefix << "_sigma_cut_high=" << model.sigma_cut_high << "\n";
    out << "emission_type=" << model.emission_type << "\n";
    out << "emission_fit=" << emission_fit_name(model.emission_type) << "\n";
    out << "nonthermal_kappa=" << model.nonthermal_kappa << "\n";
    out << "variable_kappa=" << model.variable_kappa << "\n";
    out << "variable_kappa_min=" << model.variable_kappa_min << "\n";
    out << "variable_kappa_interp_start=" << model.variable_kappa_interp_start << "\n";
    out << "variable_kappa_max=" << model.variable_kappa_max << "\n";
    out << "powerlaw_p=" << model.powerlaw_p << "\n";
    out << "powerlaw_eta=" << model.powerlaw_eta << "\n";
    out << "powerlaw_gamma_min=" << model.powerlaw_gamma_min << "\n";
    out << "powerlaw_gamma_max=" << model.powerlaw_gamma_max << "\n";
    out << "powerlaw_gamma_cutoff=" << model.powerlaw_gamma_cutoff << "\n";
    out << "# derived_grmhd_interpolate_derived_scalars=" << model.has_derived_scalars << "\n";
    out << "# derived_grmhd_precomputed_fluid_state=" << model.precomputed_fluid_state << "\n";
    out << "# derived_grmhd_data_coordinate_system=" << model.data_coordinate_system << "\n";
    out << "# derived_grmhd_radial_coordinate_log=" << model.radial_coordinate_log << "\n";
    out << prefix << "_interpolate_derived_scalars=" << model.has_derived_scalars << "\n";
    out << "# derived_" << prefix << "_precomputed_fluid_state=" << model.precomputed_fluid_state << "\n";
    out << "# derived_" << prefix << "_data_coordinate_system=" << model.data_coordinate_system << "\n";
    out << "# derived_" << prefix << "_radial_coordinate_log=" << model.radial_coordinate_log << "\n";
    out << "# derived_grmhd_n1=" << model.n1 << "\n";
    out << "# derived_grmhd_n2=" << model.n2 << "\n";
    out << "# derived_grmhd_n3=" << model.n3 << "\n";
    out << "# derived_" << prefix << "_n1=" << model.n1 << "\n";
    out << "# derived_" << prefix << "_n2=" << model.n2 << "\n";
    out << "# derived_" << prefix << "_n3=" << model.n3 << "\n";
    out << "# derived_dlambda_scale=" << model.dlambda_scale() << "\n";
}


#endif

#if KPOLARIS_ENABLE_MODEL_BHAC || KPOLARIS_ENABLE_MODEL_HAMR
void write_effective_model_parameters(std::ofstream& out,
                                      const kpolaris::BHACAMRRadiationModel<Real>& model,
                                      const std::string& model_name) {
    const std::string prefix = amr_metadata_prefix(model_name);
    out << "# derived_" << prefix << "_backend=" << amr_metadata_backend(model_name) << "\n";
    out << "# derived_" << prefix << "_M_unit=" << model.M_unit << "\n";
    out << "# derived_" << prefix << "_mbh_solar=" << model.mbh_solar << "\n";
    out << "# derived_" << prefix << "_trat_small=" << model.trat_small << "\n";
    out << "# derived_" << prefix << "_trat_large=" << model.trat_large << "\n";
    out << "# derived_" << prefix << "_beta_crit=" << model.beta_crit << "\n";
    out << "# derived_" << prefix << "_gamma=" << model.gam << "\n";
    out << "# derived_" << prefix << "_sigma_cut=" << model.sigma_cut << "\n";
    out << "# derived_" << prefix << "_sigma_cut_high=" << model.sigma_cut_high << "\n";
    out << "# derived_" << prefix
        << (model_name == "hamr" ? "_internal_hslope=" : "_hslope=")
        << model.hslope << "\n";
    out << "# derived_" << prefix << "_x1_min=" << model.startx1 << "\n";
    out << "# derived_" << prefix << "_x1_max=" << model.stopx1 << "\n";
    out << "# derived_" << prefix << "_x2_min=" << model.startx2 << "\n";
    out << "# derived_" << prefix << "_x2_max=" << model.stopx2 << "\n";
    out << "# derived_" << prefix << "_x3_min=" << model.startx3 << "\n";
    out << "# derived_" << prefix << "_x3_max=" << model.stopx3 << "\n";
    out << "emission_type=" << model.emission_type << "\n";
    out << "emission_fit=" << emission_fit_name(model.emission_type) << "\n";
    out << "nonthermal_kappa=" << model.nonthermal_kappa << "\n";
    out << "variable_kappa=" << model.variable_kappa << "\n";
    out << "variable_kappa_min=" << model.variable_kappa_min << "\n";
    out << "variable_kappa_interp_start=" << model.variable_kappa_interp_start << "\n";
    out << "variable_kappa_max=" << model.variable_kappa_max << "\n";
    out << "powerlaw_p=" << model.powerlaw_p << "\n";
    out << "powerlaw_eta=" << model.powerlaw_eta << "\n";
    out << "powerlaw_gamma_min=" << model.powerlaw_gamma_min << "\n";
    out << "powerlaw_gamma_max=" << model.powerlaw_gamma_max << "\n";
    out << "powerlaw_gamma_cutoff=" << model.powerlaw_gamma_cutoff << "\n";
    out << "# derived_grmhd_data_coordinate_system=" << model.data_coordinate_system << "\n";
    out << "# derived_grmhd_radial_coordinate_log=" << model.radial_coordinate_log << "\n";
    out << "# derived_" << prefix << "_nblocks=" << model.nblocks << "\n";
    out << "# derived_" << prefix << "_meshblock_nx1=" << model.nx1 << "\n";
    out << "# derived_" << prefix << "_meshblock_nx2=" << model.nx2 << "\n";
    out << "# derived_" << prefix << "_meshblock_nx3=" << model.nx3 << "\n";
    out << "# derived_dlambda_scale=" << model.dlambda_scale() << "\n";
}
#endif

#if KPOLARIS_ENABLE_MODEL_ATHENAK
void write_effective_model_parameters(std::ofstream& out,
                                      const kpolaris::AthenaKDirectRadiationModel<Real>& model) {
    out << "# derived_athenak_backend=direct_cks_meshblocks\n";
    out << "athenak_M_unit=" << model.M_unit << "\n";
    out << "athenak_mbh_solar=" << model.mbh_solar << "\n";
    out << "athenak_trat_small=" << model.trat_small << "\n";
    out << "athenak_trat_large=" << model.trat_large << "\n";
    out << "athenak_beta_crit=" << model.beta_crit << "\n";
    out << "athenak_gamma=" << model.gam << "\n";
    out << "athenak_sigma_cut=" << model.sigma_cut << "\n";
    out << "athenak_sigma_cut_high=" << model.sigma_cut_high << "\n";
    out << "emission_type=" << model.emission_type << "\n";
    out << "emission_fit=" << emission_fit_name(model.emission_type) << "\n";
    out << "nonthermal_kappa=" << model.nonthermal_kappa << "\n";
    out << "variable_kappa=" << model.variable_kappa << "\n";
    out << "variable_kappa_min=" << model.variable_kappa_min << "\n";
    out << "variable_kappa_interp_start=" << model.variable_kappa_interp_start << "\n";
    out << "variable_kappa_max=" << model.variable_kappa_max << "\n";
    out << "powerlaw_p=" << model.powerlaw_p << "\n";
    out << "powerlaw_eta=" << model.powerlaw_eta << "\n";
    out << "powerlaw_gamma_min=" << model.powerlaw_gamma_min << "\n";
    out << "powerlaw_gamma_max=" << model.powerlaw_gamma_max << "\n";
    out << "powerlaw_gamma_cutoff=" << model.powerlaw_gamma_cutoff << "\n";
    out << "# derived_athenak_data_coordinate_system=cartesian_ks\n";
    out << "# derived_athenak_nblocks=" << model.nblocks << "\n";
    out << "# derived_athenak_meshblock_nx1=" << model.nx1 << "\n";
    out << "# derived_athenak_meshblock_nx2=" << model.nx2 << "\n";
    out << "# derived_athenak_meshblock_nx3=" << model.nx3 << "\n";
    out << "# derived_dlambda_scale=" << model.dlambda_scale() << "\n";
}

void write_hdf5_model_metadata(H5::H5Object& obj,
                               const kpolaris::AthenaKDirectRadiationModel<Real>& model) {
    write_h5_string(obj, "athenak_backend", "direct_cks_meshblocks");
    write_h5_int(obj, "athenak_nblocks", model.nblocks);
    write_h5_int(obj, "athenak_meshblock_nx1", model.nx1);
    write_h5_int(obj, "athenak_meshblock_nx2", model.nx2);
    write_h5_int(obj, "athenak_meshblock_nx3", model.nx3);
    write_h5_scalar(obj, "athenak_M_unit", model.M_unit);
    write_h5_scalar(obj, "athenak_mbh_solar", model.mbh_solar);
    write_h5_scalar(obj, "athenak_trat_small", model.trat_small);
    write_h5_scalar(obj, "athenak_trat_large", model.trat_large);
    write_h5_scalar(obj, "athenak_beta_crit", model.beta_crit);
    write_h5_scalar(obj, "athenak_gamma", model.gam);
    write_h5_scalar(obj, "athenak_sigma_cut", model.sigma_cut);
    write_h5_scalar(obj, "athenak_sigma_cut_high", model.sigma_cut_high);
    write_h5_int(obj, "emission_type", model.emission_type);
    write_h5_string(obj, "emission_fit", emission_fit_name(model.emission_type));
    write_h5_scalar(obj, "dlambda_scale", model.dlambda_scale());
}
#endif

#if KPOLARIS_ENABLE_MODEL_BHAC || KPOLARIS_ENABLE_MODEL_HAMR
void write_hdf5_model_metadata(H5::H5Object& obj,
                               const kpolaris::BHACAMRRadiationModel<Real>& model,
                               const std::string& model_name) {
    const std::string prefix = amr_metadata_prefix(model_name);
    write_h5_string(obj, prefix + "_backend", amr_metadata_backend(model_name));
    write_h5_int(obj, prefix + "_nblocks", model.nblocks);
    write_h5_int(obj, prefix + "_meshblock_nx1", model.nx1);
    write_h5_int(obj, prefix + "_meshblock_nx2", model.nx2);
    write_h5_int(obj, prefix + "_meshblock_nx3", model.nx3);
    write_h5_scalar(obj, prefix + "_M_unit", model.M_unit);
    write_h5_scalar(obj, prefix + "_mbh_solar", model.mbh_solar);
    write_h5_scalar(obj, prefix + "_trat_small", model.trat_small);
    write_h5_scalar(obj, prefix + "_trat_large", model.trat_large);
    write_h5_scalar(obj, prefix + "_beta_crit", model.beta_crit);
    write_h5_scalar(obj, prefix + "_gamma", model.gam);
    write_h5_scalar(obj, prefix + "_sigma_cut", model.sigma_cut);
    write_h5_scalar(obj, prefix + "_sigma_cut_high", model.sigma_cut_high);
    if (model_name == "hamr") {
        write_h5_scalar(obj, "hamr_hslope", Real(1) - model.hslope);
        write_h5_scalar(obj, "hamr_internal_hslope", model.hslope);
    } else {
        write_h5_scalar(obj, "bhac_hslope", model.hslope);
    }
    write_h5_scalar(obj, prefix + "_x1_min", model.startx1);
    write_h5_scalar(obj, prefix + "_x1_max", model.stopx1);
    write_h5_scalar(obj, prefix + "_x2_min", model.startx2);
    write_h5_scalar(obj, prefix + "_x2_max", model.stopx2);
    write_h5_scalar(obj, prefix + "_x3_min", model.startx3);
    write_h5_scalar(obj, prefix + "_x3_max", model.stopx3);
    write_h5_int(obj, "grmhd_data_coordinate_system", model.data_coordinate_system);
    write_h5_int(obj, "grmhd_radial_coordinate_log", model.radial_coordinate_log);
    write_h5_int(obj, "emission_type", model.emission_type);
    write_h5_string(obj, "emission_fit", emission_fit_name(model.emission_type));
    write_h5_scalar(obj, "dlambda_scale", model.dlambda_scale());
}
#endif

template<class Model>
void write_effective_model_parameters(std::ofstream& out,
                                      const Model& model,
                                      const std::string&) {
    write_effective_model_parameters(out, model);
}

std::string lowercase_copy(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

std::vector<Real> parse_frequency_list(const std::string& text) {
    std::string normalized = text;
    for (char& c : normalized) {
        if (c == ';') {
            c = ',';
        }
    }
    std::vector<Real> frequencies;
    std::stringstream ss(normalized);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim_copy(item);
        if (item.empty()) {
            continue;
        }
        frequencies.push_back(Real(std::stod(item)));
    }
    if (frequencies.empty()) {
        throw std::runtime_error("freq_list did not contain any valid frequencies");
    }
    for (Real freq : frequencies) {
        if (!(freq > Real(0))) {
            throw std::runtime_error("all frequencies must be positive");
        }
    }
    return frequencies;
}

std::vector<Real> build_frequency_grid(const Options& opt) {
    if (!opt.freq_list.empty()) {
        return parse_frequency_list(opt.freq_list);
    }
    if (opt.nfreq > 0 || opt.freq_min > Real(0) || opt.freq_max > Real(0)) {
        if (opt.nfreq <= 0) {
            throw std::runtime_error("freq_min/freq_max range requires nfreq > 0");
        }
        if (!(opt.freq_min > Real(0)) || !(opt.freq_max > Real(0))) {
            throw std::runtime_error("freq_min and freq_max must be positive");
        }
        if (opt.nfreq == 1) {
            return {opt.freq_min};
        }
        if (!(opt.freq_max > opt.freq_min)) {
            throw std::runtime_error("freq_max must be greater than freq_min when nfreq > 1");
        }
        std::vector<Real> frequencies(static_cast<size_t>(opt.nfreq));
        const std::string spacing = lowercase_copy(opt.freq_spacing);
        if (spacing == "linear" || spacing == "lin") {
            for (int i = 0; i < opt.nfreq; ++i) {
                const Real t = Real(i) / Real(opt.nfreq - 1);
                frequencies[static_cast<size_t>(i)] = opt.freq_min + t * (opt.freq_max - opt.freq_min);
            }
        } else if (spacing == "log" || spacing == "logarithmic") {
            const Real log_min = Kokkos::log(opt.freq_min);
            const Real log_max = Kokkos::log(opt.freq_max);
            for (int i = 0; i < opt.nfreq; ++i) {
                const Real t = Real(i) / Real(opt.nfreq - 1);
                frequencies[static_cast<size_t>(i)] = Kokkos::exp(log_min + t * (log_max - log_min));
            }
        } else {
            throw std::runtime_error("unknown freq_spacing: " + opt.freq_spacing);
        }
        return frequencies;
    }
    if (!(opt.freq > Real(0))) {
        throw std::runtime_error("freq must be positive");
    }
    return {opt.freq};
}

std::string frequency_list_string(const std::vector<Real>& frequencies) {
    std::ostringstream out;
    out.precision(17);
    for (size_t i = 0; i < frequencies.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        out << frequencies[i];
    }
    return out.str();
}

std::vector<std::string> parse_string_list(const std::string& text) {
    std::string normalized = text;
    for (char& c : normalized) {
        if (c == ';') {
            c = ',';
        }
    }
    std::vector<std::string> items;
    std::stringstream ss(normalized);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim_copy(item);
        if (!item.empty()) {
            items.push_back(item);
        }
    }
    return items;
}

[[maybe_unused]] std::vector<Real> parse_real_list(const std::string& text,
                                                  const char* name) {
    std::vector<Real> values;
    for (const std::string& item : parse_string_list(text)) {
        values.push_back(Real(std::stod(item)));
    }
    if (values.empty()) {
        throw std::runtime_error(std::string(name) + " did not contain any values");
    }
    return values;
}

std::string format_indexed_dump_path(const std::string& pattern, int index) {
    const size_t brace = pattern.find("{}");
    if (brace != std::string::npos) {
        std::string out = pattern;
        out.replace(brace, 2, std::to_string(index));
        return out;
    }
    if (pattern.find('%') != std::string::npos) {
        char buffer[4096];
        const int n = std::snprintf(buffer, sizeof(buffer), pattern.c_str(), index);
        if (n < 0 || static_cast<size_t>(n) >= sizeof(buffer)) {
            throw std::runtime_error("slow_light_dump_pattern formatted path is too long");
        }
        return std::string(buffer, static_cast<size_t>(n));
    }
    throw std::runtime_error("slow_light_dump_pattern must contain printf-style integer format or {}");
}

std::vector<std::string> build_slow_light_dump_paths(const Options& opt) {
    std::vector<std::string> paths;
    if (!opt.slow_light_dump_list.empty()) {
        paths = parse_string_list(opt.slow_light_dump_list);
    } else if (!opt.slow_light_dump_pattern.empty()) {
        if (opt.slow_light_dump_end < opt.slow_light_dump_start) {
            throw std::runtime_error("slow_light_dump_pattern requires slow_light_dump_end >= slow_light_dump_start");
        }
        if (opt.slow_light_dump_stride <= 0) {
            throw std::runtime_error("slow_light_dump_stride must be positive");
        }
        for (int i = opt.slow_light_dump_start; i <= opt.slow_light_dump_end; i += opt.slow_light_dump_stride) {
            paths.push_back(format_indexed_dump_path(opt.slow_light_dump_pattern, i));
        }
    }
    if (paths.size() < 2) {
        throw std::runtime_error("slow_light requires at least two dumps via slow_light_dump_list or slow_light_dump_pattern");
    }
    return paths;
}

#if KPOLARIS_ENABLE_SLOW_LIGHT

template<class Reader>
std::vector<Real> build_model_slow_light_dump_times(const Options& opt,
                                                    const std::vector<std::string>& dump_paths,
                                                    Reader read_time) {
    std::vector<Real> times;
    if (!opt.slow_light_time_list.empty()) {
        times = parse_real_list(opt.slow_light_time_list, "slow_light_time_list");
    } else {
        times.reserve(dump_paths.size());
        for (const std::string& path : dump_paths) {
            times.push_back(Real(read_time(path)));
        }
    }
    if (times.size() != dump_paths.size()) {
        throw std::runtime_error("slow_light_time_list length must match slow_light dumps");
    }
    if (opt.model == "kharma") opt.kharma_ddc.validate_schedule(dump_paths);
    for (size_t i = 1; i < times.size(); ++i) {
        if (!(times[i] > times[i - 1])) {
            throw std::runtime_error("slow-light dump times must be strictly increasing");
        }
    }
    return times;
}

std::vector<Real> build_slow_light_dump_times(const Options& opt) {
    if (opt.slow_light_dump_list.empty() && opt.slow_light_dump_pattern.empty()) {
        return {};
    }
    const std::vector<std::string> dump_paths = build_slow_light_dump_paths(opt);
#if KPOLARIS_ENABLE_MODEL_IHARM
    if (opt.model == "iharm") {
        return build_model_slow_light_dump_times(
            opt, dump_paths, [](const std::string& path) { return kpolaris::read_iharm_dump_time(path); });
    }
#endif
#if KPOLARIS_ENABLE_MODEL_KHARMA
    if (opt.model == "kharma") {
        return build_model_slow_light_dump_times(
            opt, dump_paths, [&opt](const std::string& path) { return kpolaris::read_kharma_dump_time(path, opt.kharma_ddc); });
    }
#endif
#if KPOLARIS_ENABLE_MODEL_ATHENAK
    if (opt.model == "athenak") {
        return build_model_slow_light_dump_times(
            opt, dump_paths, [](const std::string& path) { return kpolaris::read_athenak_dump_time(path); });
    }
#endif
#if KPOLARIS_ENABLE_MODEL_BHAC
    if (opt.model == "bhac") {
        return build_model_slow_light_dump_times(
            opt, dump_paths, [](const std::string& path) { return kpolaris::read_bhac_dump_time(path); });
    }
#endif
#if KPOLARIS_ENABLE_MODEL_HAMR
    if (opt.model == "hamr") {
        return build_model_slow_light_dump_times(
            opt, dump_paths, [](const std::string& path) { return kpolaris::read_hamr_dump_time(path); });
    }
#endif
    throw std::runtime_error("slow_light_time_probe dump-time matching is supported for iharm, kharma, athenak, bhac, and hamr only");
}

[[maybe_unused]] void print_slow_light_probe_dump_match(
    const Options& opt,
    const kpolaris_image_detail::SlowLightTimeProbeHostResult& probe) {
    const std::vector<Real> times = build_slow_light_dump_times(opt);
    if (times.empty()) {
        std::cout << "dump_time_matching none\n";
        return;
    }
    auto bracket = [&](Real tmin, Real tmax) {
        auto first_it = std::upper_bound(times.begin(), times.end(), tmin);
        const int first = static_cast<int>(first_it - times.begin()) - 1;
        auto last_it = std::lower_bound(times.begin(), times.end(), tmax);
        const int last = static_cast<int>(last_it - times.begin());
        const bool covered = first >= 0 && last < static_cast<int>(times.size()) && first < last &&
                             times[static_cast<size_t>(first)] <= tmin &&
                             times[static_cast<size_t>(last)] >= tmax;
        return std::tuple<int, int, bool>(
            std::max(0, first), std::min(static_cast<int>(times.size()) - 1, last), covered);
    };
    const auto [required_first, required_last, required_covered] =
        bracket(probe.min_fluid_time, probe.max_fluid_time);
    const Real runtime_min = probe.min_fluid_time;
    // Outgoing vacuum is completed without requesting further fluid dumps.
    const Real runtime_max = probe.max_fluid_time;
    const auto [runtime_first, runtime_last, runtime_covered] = bracket(runtime_min, runtime_max);
    std::cout << "dump_time_matching checked\n";
    std::cout << "dump_count " << times.size() << "\n";
    std::cout << "dump_time_min " << times.front() << "\n";
    std::cout << "dump_time_max " << times.back() << "\n";
    std::cout << "dump_time_covers_required " << (required_covered ? 1 : 0) << "\n";
    std::cout << "required_first_dump_index " << required_first << "\n";
    std::cout << "required_last_dump_index " << required_last << "\n";
    std::cout << "required_window_count " << std::max(0, required_last - required_first) << "\n";
    std::cout << "dump_time_covers_runtime " << (runtime_covered ? 1 : 0) << "\n";
    std::cout << "runtime_first_dump_index " << runtime_first << "\n";
    std::cout << "runtime_last_dump_index " << runtime_last << "\n";
    std::cout << "runtime_window_count " << std::max(0, runtime_last - runtime_first) << "\n";
    if (!opt.slow_light_dump_pattern.empty()) {
        std::cout << "required_first_dump_number "
                  << opt.slow_light_dump_start + required_first * opt.slow_light_dump_stride << "\n";
        std::cout << "required_last_dump_number "
                  << opt.slow_light_dump_start + required_last * opt.slow_light_dump_stride << "\n";
        std::cout << "runtime_first_dump_number "
                  << opt.slow_light_dump_start + runtime_first * opt.slow_light_dump_stride << "\n";
        std::cout << "runtime_last_dump_number "
                  << opt.slow_light_dump_start + runtime_last * opt.slow_light_dump_stride << "\n";
    }
    if (!required_covered) {
        std::cout << "warning_required_time_outside_dump_range 1\n";
    }
    if (!runtime_covered) {
        std::cout << "warning_runtime_time_outside_dump_range 1\n";
    }
}

template<class Metric>
void run_slow_light_time_probe_with_metric(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    const Metric& metric,
    const char* label) {
    const auto probe = kpolaris_image_detail::run_slow_light_time_probe_metric<Kokkos::DefaultExecutionSpace>(
        pass_a, opt.slow_light_observation_time, metric, opt.timing);
    const double mean_pass_a = probe.pixels > 0 ?
        static_cast<double>(probe.pass_a_steps) / static_cast<double>(probe.pixels) : 0.0;
    const double mean_pass_b = probe.pixels > 0 ?
        static_cast<double>(probe.pass_b_steps) / static_cast<double>(probe.pixels) : 0.0;
    std::cout << "KPolaris slow-light time probe\n";
    std::cout << "model " << opt.model << "\n";
    std::cout << "coordinate " << opt.coordinate << "\n";
    std::cout << "metric_label " << label << "\n";
    std::cout << "pixels " << probe.pixels << "\n";
    std::cout << "observation_time " << opt.slow_light_observation_time << "\n";
    std::cout << "radiating_pixels " << probe.radiating_pixels << "\n";
    std::cout << "radiating_samples " << probe.radiating_samples << "\n";
    std::cout << "valid_endpoint_pixels " << probe.valid_endpoint_pixels << "\n";
    std::cout << "returned_pixels " << probe.returned_pixels << "\n";
    std::cout << "x0_min " << probe.min_x0 << "\n";
    std::cout << "x0_max " << probe.max_x0 << "\n";
    std::cout << "required_time_min " << probe.min_fluid_time << "\n";
    std::cout << "required_time_max " << probe.max_fluid_time << "\n";
    std::cout << "required_time_span " << probe.max_fluid_time - probe.min_fluid_time << "\n";
    std::cout << "camera_time_pixels " << probe.camera_time_pixels << "\n";
    std::cout << "camera_time_min " << probe.min_camera_time << "\n";
    std::cout << "camera_time_max " << probe.max_camera_time << "\n";
    const Real runtime_min = probe.min_fluid_time;
    // Outgoing vacuum is completed without requesting further fluid dumps.
    const Real runtime_max = probe.max_fluid_time;
    std::cout << "runtime_window_time_min " << runtime_min << "\n";
    std::cout << "runtime_window_time_max " << runtime_max << "\n";
    std::cout << "runtime_window_time_span " << runtime_max - runtime_min << "\n";
    std::cout << "mean_pass_a_steps " << mean_pass_a << "\n";
    std::cout << "mean_pass_b_steps " << mean_pass_b << "\n";
    std::cout << "max_pass_a_steps " << probe.max_pass_a_steps << "\n";
    std::cout << "max_pass_b_steps " << probe.max_pass_b_steps << "\n";
    print_slow_light_probe_dump_match(opt, probe);
}

void run_slow_light_time_probe_for_coordinate(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a) {
    (void)opt;
    switch (pass_a.coordinate_system) {
    case kpolaris::CoordinateSystem::CartesianKS: {
#if KPOLARIS_ENABLE_COORDINATE_CARTESIAN_KS
        kpolaris::KerrSchildInMetric<Real> metric(pass_a.mass, pass_a.spin);
        run_slow_light_time_probe_with_metric(opt, pass_a, metric, "cartesian_ks");
        return;
#else
        throw std::runtime_error("slow-light time probe cartesian_ks coordinate backend is not built");
#endif
    }
    case kpolaris::CoordinateSystem::SphericalKS: {
#if KPOLARIS_ENABLE_COORDINATE_SPHERICAL_KS
        kpolaris::KerrSchildSphericalMetric<Real> metric(pass_a.mass, pass_a.spin);
        run_slow_light_time_probe_with_metric(opt, pass_a, metric, "spherical_ks");
        return;
#else
        throw std::runtime_error("slow-light time probe spherical_ks coordinate backend is not built");
#endif
    }
    case kpolaris::CoordinateSystem::BoyerLindquist: {
#if KPOLARIS_ENABLE_COORDINATE_BOYER_LINDQUIST
        kpolaris::KerrBoyerLindquistMetric<Real> metric(pass_a.mass, pass_a.spin);
        run_slow_light_time_probe_with_metric(opt, pass_a, metric, "boyer_lindquist");
        return;
#else
        throw std::runtime_error("slow-light time probe boyer_lindquist coordinate backend is not built");
#endif
    }
    case kpolaris::CoordinateSystem::FMKS: {
#if KPOLARIS_ENABLE_COORDINATE_FMKS
        kpolaris::KerrFMKSMetric<Real> metric(pass_a.mass, pass_a.spin, pass_a.fmks_startx1,
                                             pass_a.fmks_hslope, pass_a.fmks_mks_smooth,
                                             pass_a.fmks_poly_alpha, pass_a.fmks_poly_xt,
                                             pass_a.fmks_poly_norm);
        run_slow_light_time_probe_with_metric(opt, pass_a, metric, "fmks");
        return;
#else
        throw std::runtime_error("slow-light time probe fmks coordinate backend is not built");
#endif
    }
    case kpolaris::CoordinateSystem::MKS: {
#if KPOLARIS_ENABLE_COORDINATE_MKS
        kpolaris::KerrSchildSphericalMetric<Real> metric(pass_a.mass, pass_a.spin);
        run_slow_light_time_probe_with_metric(opt, pass_a, metric, "mks");
        return;
#else
        throw std::runtime_error("slow-light time probe mks coordinate backend is not built");
#endif
    }
    }
    throw std::runtime_error("unknown slow-light time probe coordinate system");
}

template<class RadiationModel, class LoadModel>
std::vector<ImageHostData> run_grmhd_slow_light_coordinate_batch(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    const std::vector<std::string>& dump_paths,
    const std::vector<Real>& dump_times,
    RadiationModel first_model,
    LoadModel load_model,
    const std::vector<kpolaris_image_detail::SlowLightBatchJob>& jobs,
    const std::vector<Real>& frequencies = {}) {
    kpolaris_image_detail::SlowLightBatchOptions options;
    options.windows_per_block = opt.slow_light_windows_per_block;
    options.frequency_chunk_size = opt.multifrequency_chunk_size;
    options.prefetch = opt.slow_light_prefetch;
    options.prefetch_snapshots = opt.slow_light_prefetch_snapshots;
    options.pipeline = opt.slow_light_pipeline && !opt.kharma_resample_spherical_ks_precomputed && !opt.kharma_resample_spherical_ks_primitives;
    options.timing = opt.timing;
    options.analysis_mode = opt.analysis_mode;
    options.interpolation = opt.slow_light_interpolation;
    options.step_mode = opt.slow_light_step_mode;
    options.analysis_config = make_analysis_config(opt, pass_a);
    switch (pass_a.coordinate_system) {
    case kpolaris::CoordinateSystem::CartesianKS: {
#if KPOLARIS_ENABLE_COORDINATE_CARTESIAN_KS
        kpolaris::KerrSchildInMetric<Real> metric(pass_a.mass, pass_a.spin);
        return kpolaris_image_detail::run_slow_light_batch_metric<decltype(metric), RadiationModel>(
            pass_a, dump_paths, dump_times, jobs, load_model, metric, options, &first_model, frequencies);
#else
        throw std::runtime_error("slow-light cartesian_ks coordinate backend is not built");
#endif
    }
    case kpolaris::CoordinateSystem::SphericalKS: {
#if KPOLARIS_ENABLE_COORDINATE_SPHERICAL_KS
        kpolaris::KerrSchildSphericalMetric<Real> metric(pass_a.mass, pass_a.spin);
        return kpolaris_image_detail::run_slow_light_batch_metric<decltype(metric), RadiationModel>(
            pass_a, dump_paths, dump_times, jobs, load_model, metric, options, &first_model, frequencies);
#else
        throw std::runtime_error("slow-light spherical_ks coordinate backend is not built");
#endif
    }
    case kpolaris::CoordinateSystem::BoyerLindquist: {
#if KPOLARIS_ENABLE_COORDINATE_BOYER_LINDQUIST
        kpolaris::KerrBoyerLindquistMetric<Real> metric(pass_a.mass, pass_a.spin);
        return kpolaris_image_detail::run_slow_light_batch_metric<decltype(metric), RadiationModel>(
            pass_a, dump_paths, dump_times, jobs, load_model, metric, options, &first_model, frequencies);
#else
        throw std::runtime_error("slow-light boyer_lindquist coordinate backend is not built");
#endif
    }
    case kpolaris::CoordinateSystem::FMKS: {
#if KPOLARIS_ENABLE_COORDINATE_FMKS
        kpolaris::KerrFMKSMetric<Real> metric(pass_a.mass, pass_a.spin, pass_a.fmks_startx1,
                                             pass_a.fmks_hslope, pass_a.fmks_mks_smooth,
                                             pass_a.fmks_poly_alpha, pass_a.fmks_poly_xt,
                                             pass_a.fmks_poly_norm);
        return kpolaris_image_detail::run_slow_light_batch_metric<decltype(metric), RadiationModel>(
            pass_a, dump_paths, dump_times, jobs, load_model, metric, options, &first_model, frequencies);
#else
        throw std::runtime_error("slow-light fmks coordinate backend is not built");
#endif
    }
    case kpolaris::CoordinateSystem::MKS: {
#if KPOLARIS_ENABLE_COORDINATE_MKS
        kpolaris::KerrSchildSphericalMetric<Real> metric(pass_a.mass, pass_a.spin);
        return kpolaris_image_detail::run_slow_light_batch_metric<decltype(metric), RadiationModel>(
            pass_a, dump_paths, dump_times, jobs, load_model, metric, options, &first_model, frequencies);
#else
        throw std::runtime_error("slow-light mks coordinate backend is not built");
#endif
    }
    }
    throw std::runtime_error("unknown slow-light coordinate system");
}

template<class RadiationModel, class LoadModel>
ImageHostData run_grmhd_slow_light_coordinate_image(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    const std::vector<std::string>& dump_paths,
    const std::vector<Real>& dump_times,
    RadiationModel first_model,
    LoadModel load_model,
    const char* label_prefix) {
    if (opt.slow_light_step_mode == kpolaris::SlowLightStepMode::decoupled ||
        opt.slow_light_windows_per_block > 1 || opt.slow_light_prefetch_snapshots > 0 ||
        (opt.slow_light_pipeline && opt.model == "kharma" && opt.slow_light_prefetch)) {
        const std::vector<kpolaris_image_detail::SlowLightBatchJob> jobs = {
            {opt.slow_light_observation_time, 0, dump_paths.size() - 1}};
        return run_grmhd_slow_light_coordinate_batch(opt, pass_a, dump_paths, dump_times,
                                                    first_model, load_model, jobs).front();
    }
    const std::string prefix(label_prefix);
    switch (pass_a.coordinate_system) {
    case kpolaris::CoordinateSystem::CartesianKS: {
#if KPOLARIS_ENABLE_COORDINATE_CARTESIAN_KS
        kpolaris::KerrSchildInMetric<Real> metric(pass_a.mass, pass_a.spin);
        const std::string label = prefix + "_slow_light_cartesian_ks";
        return kpolaris_image_detail::run_slow_light_image_metric<decltype(metric), RadiationModel>(
            pass_a, dump_paths, dump_times, opt.slow_light_observation_time, first_model, load_model,
            opt.timing, metric, label.c_str(), opt.analysis_mode,
            opt.slow_light_prefetch, make_analysis_config(opt, pass_a), opt.slow_light_interpolation);
#else
        throw std::runtime_error("slow-light cartesian_ks coordinate backend is not built");
#endif
    }
    case kpolaris::CoordinateSystem::SphericalKS: {
#if KPOLARIS_ENABLE_COORDINATE_SPHERICAL_KS
        kpolaris::KerrSchildSphericalMetric<Real> metric(pass_a.mass, pass_a.spin);
        const std::string label = prefix + "_slow_light_spherical_ks";
        return kpolaris_image_detail::run_slow_light_image_metric<decltype(metric), RadiationModel>(
            pass_a, dump_paths, dump_times, opt.slow_light_observation_time, first_model, load_model,
            opt.timing, metric, label.c_str(), opt.analysis_mode,
            opt.slow_light_prefetch, make_analysis_config(opt, pass_a), opt.slow_light_interpolation);
#else
        throw std::runtime_error("slow-light spherical_ks coordinate backend is not built");
#endif
    }
    case kpolaris::CoordinateSystem::BoyerLindquist: {
#if KPOLARIS_ENABLE_COORDINATE_BOYER_LINDQUIST
        kpolaris::KerrBoyerLindquistMetric<Real> metric(pass_a.mass, pass_a.spin);
        const std::string label = prefix + "_slow_light_boyer_lindquist";
        return kpolaris_image_detail::run_slow_light_image_metric<decltype(metric), RadiationModel>(
            pass_a, dump_paths, dump_times, opt.slow_light_observation_time, first_model, load_model,
            opt.timing, metric, label.c_str(), opt.analysis_mode,
            opt.slow_light_prefetch, make_analysis_config(opt, pass_a), opt.slow_light_interpolation);
#else
        throw std::runtime_error("slow-light boyer_lindquist coordinate backend is not built");
#endif
    }
    case kpolaris::CoordinateSystem::FMKS: {
#if KPOLARIS_ENABLE_COORDINATE_FMKS
        kpolaris::KerrFMKSMetric<Real> metric(pass_a.mass, pass_a.spin, pass_a.fmks_startx1,
                                             pass_a.fmks_hslope, pass_a.fmks_mks_smooth,
                                             pass_a.fmks_poly_alpha, pass_a.fmks_poly_xt,
                                             pass_a.fmks_poly_norm);
        const std::string label = prefix + "_slow_light_fmks";
        return kpolaris_image_detail::run_slow_light_image_metric<decltype(metric), RadiationModel>(
            pass_a, dump_paths, dump_times, opt.slow_light_observation_time, first_model, load_model,
            opt.timing, metric, label.c_str(), opt.analysis_mode,
            opt.slow_light_prefetch, make_analysis_config(opt, pass_a), opt.slow_light_interpolation);
#else
        throw std::runtime_error("slow-light fmks coordinate backend is not built");
#endif
    }
    case kpolaris::CoordinateSystem::MKS: {
#if KPOLARIS_ENABLE_COORDINATE_MKS
        kpolaris::KerrSchildSphericalMetric<Real> metric(pass_a.mass, pass_a.spin);
        const std::string label = prefix + "_slow_light_mks";
        return kpolaris_image_detail::run_slow_light_image_metric<decltype(metric), RadiationModel>(
            pass_a, dump_paths, dump_times, opt.slow_light_observation_time, first_model, load_model,
            opt.timing, metric, label.c_str(), opt.analysis_mode,
            opt.slow_light_prefetch, make_analysis_config(opt, pass_a), opt.slow_light_interpolation);
#else
        throw std::runtime_error("slow-light mks coordinate backend is not built");
#endif
    }
    }
    throw std::runtime_error("unknown slow-light coordinate system");
}

template<class RadiationModel, class LoadModel>
std::vector<ImageHostData> run_grmhd_slow_light_coordinate_multifrequency_images(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    const std::vector<std::string>& dump_paths,
    const std::vector<Real>& dump_times,
    RadiationModel first_model,
    LoadModel load_model,
    const std::vector<Real>& frequencies) {
    const std::vector<kpolaris_image_detail::SlowLightBatchJob> jobs = {
        {opt.slow_light_observation_time, 0, dump_paths.size() - 1}};
    return run_grmhd_slow_light_coordinate_batch(opt, pass_a, dump_paths, dump_times,
                                                first_model, load_model, jobs, frequencies);
}

#if KPOLARIS_ENABLE_MODEL_IHARM
kpolaris::IHARMLoadOptions make_iharm_load_options(const Options& opt, Real frequency) {
    kpolaris::IHARMLoadOptions load_opt;
    load_opt.freq = frequency;
    load_opt.M_unit = opt.iharm_M_unit;
    load_opt.mbh_solar = opt.iharm_mbh_solar;
    load_opt.trat_small = opt.iharm_trat_small;
    load_opt.trat_large = opt.iharm_trat_large;
    load_opt.beta_crit = opt.iharm_beta_crit;
    load_opt.sigma_cut = opt.iharm_sigma_cut;
    load_opt.sigma_cut_high = opt.iharm_sigma_cut_high;
    load_opt.emission_type = effective_emission_type(opt.emission_type,
                                                      KPOLARIS_IHARM_COMPILED_EMISSION_TYPE,
                                                      4);
    load_opt.nonthermal_kappa = opt.nonthermal_kappa;
    load_opt.variable_kappa = opt.variable_kappa;
    load_opt.variable_kappa_min = opt.variable_kappa_min;
    load_opt.variable_kappa_interp_start = opt.variable_kappa_interp_start;
    load_opt.variable_kappa_max = opt.variable_kappa_max;
    load_opt.powerlaw_p = opt.powerlaw_p;
    load_opt.powerlaw_eta = opt.powerlaw_eta;
    load_opt.powerlaw_gamma_min = opt.powerlaw_gamma_min;
    load_opt.powerlaw_gamma_max = opt.powerlaw_gamma_max;
    load_opt.powerlaw_gamma_cutoff = opt.powerlaw_gamma_cutoff;
    load_opt.interpolate_derived_scalars = opt.iharm_interpolate_derived_scalars;
    load_opt.resample_spherical_ks_precomputed = opt.iharm_resample_spherical_ks_precomputed;
    load_opt.resample_spherical_ks_primitives = opt.iharm_resample_spherical_ks_primitives;
    load_opt.resample_n1 = opt.iharm_resample_n1;
    load_opt.resample_n2 = opt.iharm_resample_n2;
    load_opt.resample_n3 = opt.iharm_resample_n3;
    load_opt.resample_r_in = opt.iharm_resample_r_in;
    load_opt.resample_r_out = opt.iharm_resample_r_out;
    load_opt.timing = opt.timing;
    return load_opt;
}

ImageHostData run_iharm_slow_light_image(const Options& opt,
                                         const kpolaris::PassAParams<Real>& pass_a,
                                         kpolaris::GRMHDRadiationModel<Real> first_model,
                                         Real frequency) {
    const std::vector<std::string> dump_paths = build_slow_light_dump_paths(opt);
    const std::vector<Real> dump_times = build_model_slow_light_dump_times(
        opt, dump_paths, [](const std::string& path) { return kpolaris::read_iharm_dump_time(path); });
    auto load_model = [&](const std::string& path) {
        kpolaris::IHARMLoadOptions load_opt = make_iharm_load_options(opt, frequency);
        load_opt.dump_path = path;
        return kpolaris::load_iharm_model_from_hdf5(load_opt);
    };
    first_model.freq_cgs = frequency;
    if (opt.iharm_dump.empty() || opt.iharm_dump != dump_paths.front()) {
        first_model = load_model(dump_paths.front());
    }
    return run_grmhd_slow_light_coordinate_image(
        opt, pass_a, dump_paths, dump_times, first_model, load_model, "iharm");
}

std::vector<ImageHostData> run_iharm_slow_light_multifrequency_image_data(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::GRMHDRadiationModel<Real> first_model,
    const std::vector<Real>& frequencies) {
    const std::vector<std::string> dump_paths = build_slow_light_dump_paths(opt);
    const std::vector<Real> dump_times = build_model_slow_light_dump_times(
        opt, dump_paths, [](const std::string& path) { return kpolaris::read_iharm_dump_time(path); });
    auto load_base_model = [&](const std::string& path, Real frequency) {
        kpolaris::IHARMLoadOptions load_opt = make_iharm_load_options(opt, frequency);
        load_opt.dump_path = path;
        return kpolaris::load_iharm_model_from_hdf5(load_opt);
    };
    auto load_models = [&](const std::string& path) {
        return load_base_model(path, frequencies.front());
    };
    first_model.freq_cgs = frequencies.front();
    if (opt.iharm_dump.empty() || opt.iharm_dump != dump_paths.front()) {
        first_model = load_models(dump_paths.front());
    }
    return run_grmhd_slow_light_coordinate_multifrequency_images(
        opt, pass_a, dump_paths, dump_times, first_model, load_models, frequencies);
}

#endif

#if KPOLARIS_ENABLE_MODEL_KHARMA
kpolaris::KHARMALoadOptions make_kharma_load_options(const Options& opt, Real frequency) {
    kpolaris::KHARMALoadOptions load_opt;
    load_opt.ddc = opt.kharma_ddc;
    load_opt.freq = frequency;
    load_opt.M_unit = opt.kharma_M_unit;
    load_opt.mbh_solar = opt.kharma_mbh_solar;
    load_opt.trat_small = opt.kharma_trat_small;
    load_opt.trat_large = opt.kharma_trat_large;
    load_opt.beta_crit = opt.kharma_beta_crit;
    load_opt.sigma_cut = opt.kharma_sigma_cut;
    load_opt.sigma_cut_high = opt.kharma_sigma_cut_high;
    load_opt.emission_type = effective_emission_type(opt.emission_type,
                                                      KPOLARIS_IHARM_COMPILED_EMISSION_TYPE,
                                                      4);
    load_opt.nonthermal_kappa = opt.nonthermal_kappa;
    load_opt.variable_kappa = opt.variable_kappa;
    load_opt.variable_kappa_min = opt.variable_kappa_min;
    load_opt.variable_kappa_interp_start = opt.variable_kappa_interp_start;
    load_opt.variable_kappa_max = opt.variable_kappa_max;
    load_opt.powerlaw_p = opt.powerlaw_p;
    load_opt.powerlaw_eta = opt.powerlaw_eta;
    load_opt.powerlaw_gamma_min = opt.powerlaw_gamma_min;
    load_opt.powerlaw_gamma_max = opt.powerlaw_gamma_max;
    load_opt.powerlaw_gamma_cutoff = opt.powerlaw_gamma_cutoff;
    load_opt.interpolate_derived_scalars = opt.kharma_interpolate_derived_scalars;
    load_opt.resample_spherical_ks_precomputed = opt.kharma_resample_spherical_ks_precomputed;
    load_opt.resample_spherical_ks_primitives = opt.kharma_resample_spherical_ks_primitives;
    load_opt.resample_n1 = opt.kharma_resample_n1;
    load_opt.resample_n2 = opt.kharma_resample_n2;
    load_opt.resample_n3 = opt.kharma_resample_n3;
    load_opt.resample_r_in = opt.kharma_resample_r_in;
    load_opt.resample_r_out = opt.kharma_resample_r_out;
    load_opt.reverse_field = opt.kharma_reverse_field;
    load_opt.timing = opt.timing;
    return load_opt;
}

struct KHARMASlowLightStagedLoader {
    using model_type = kpolaris::GRMHDRadiationModel<Real>;
    using slow_light_prefetch_result_type = kpolaris::KHARMAStagedDump;
    using slow_light_upload_workspace_type = kpolaris::KHARMAUploadWorkspace;

    const Options* opt = nullptr;
    Real frequency = Real(230.0e9);
    const std::vector<std::string>* paths = nullptr;
    const std::vector<Real>* times = nullptr;
    std::shared_ptr<kpolaris::KHARMAHostBufferPool> host_pool = std::make_shared<kpolaris::KHARMAHostBufferPool>();

    kpolaris::KHARMALoadOptions load_options(const std::string& path) const {
        kpolaris::KHARMALoadOptions load_opt = make_kharma_load_options(*opt, frequency);
        load_opt.dump_path = path;
        if (paths && times && !path.empty()) {
            const auto found = std::find(paths->begin(), paths->end(), path);
            if (found != paths->end()) load_opt.ddc_expected_time = times->at(static_cast<size_t>(found - paths->begin()));
        }
        return load_opt;
    }

    slow_light_prefetch_result_type prefetch(const std::string& path) const {
        return kpolaris::read_kharma_staged_dump_resolved(load_options(path), host_pool->take());
    }

    model_type materialize(slow_light_prefetch_result_type&& staged) const {
        auto model=kpolaris::materialize_kharma_model_from_staged(staged, load_options(std::string()));
        host_pool->recycle(std::move(staged));
        return model;
    }

    model_type pipeline_prepare(slow_light_upload_workspace_type& workspace,
                                slow_light_prefetch_result_type&& staged, size_t slot) const {
        auto model=workspace.prepare(staged, load_options(std::string()), slot);
        host_pool->recycle(std::move(staged));
        return model;
    }

    model_type load(const std::string& path) const {
        return materialize(prefetch(path));
    }
};

ImageHostData run_kharma_slow_light_image(const Options& opt,
                                          const kpolaris::PassAParams<Real>& pass_a,
                                          kpolaris::GRMHDRadiationModel<Real> first_model,
                                          Real frequency) {
    const std::vector<std::string> dump_paths = build_slow_light_dump_paths(opt);
    const std::vector<Real> dump_times = build_model_slow_light_dump_times(
        opt, dump_paths, [&opt](const std::string& path) { return kpolaris::read_kharma_dump_time(path, opt.kharma_ddc); });
    KHARMASlowLightStagedLoader load_model{&opt, frequency, &dump_paths, &dump_times};
    first_model.freq_cgs = frequency;
    if (opt.kharma_ddc.native || opt.kharma_dump.empty() || opt.kharma_dump != dump_paths.front()) {
        first_model = load_model.load(dump_paths.front());
    }
    return run_grmhd_slow_light_coordinate_image(
        opt, pass_a, dump_paths, dump_times, first_model, load_model, "kharma");
}
std::vector<ImageHostData> run_kharma_slow_light_multifrequency_image_data(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::GRMHDRadiationModel<Real> first_model,
    const std::vector<Real>& frequencies) {
    const std::vector<std::string> dump_paths = build_slow_light_dump_paths(opt);
    const std::vector<Real> dump_times = build_model_slow_light_dump_times(
        opt, dump_paths, [&opt](const std::string& path) { return kpolaris::read_kharma_dump_time(path, opt.kharma_ddc); });
    KHARMASlowLightStagedLoader first_frequency_loader{&opt, frequencies.front(), &dump_paths, &dump_times};
    first_model.freq_cgs = frequencies.front();
    if (opt.kharma_ddc.native || opt.kharma_dump.empty() || opt.kharma_dump != dump_paths.front()) {
        first_model = first_frequency_loader.load(dump_paths.front());
    }
    return run_grmhd_slow_light_coordinate_multifrequency_images(
        opt, pass_a, dump_paths, dump_times, first_model, first_frequency_loader, frequencies);
}

#endif

#if KPOLARIS_ENABLE_MODEL_ATHENAK
kpolaris::AthenaKLoadOptions make_athenak_load_options(const Options& opt, Real frequency) {
    kpolaris::AthenaKLoadOptions load_opt;
    load_opt.freq = frequency;
    load_opt.M_unit = opt.athenak_M_unit;
    load_opt.mbh_solar = opt.athenak_mbh_solar;
    load_opt.trat_small = opt.athenak_trat_small;
    load_opt.trat_large = opt.athenak_trat_large;
    load_opt.beta_crit = opt.athenak_beta_crit;
    load_opt.gamma = opt.athenak_gamma;
    load_opt.sigma_cut = opt.athenak_sigma_cut;
    load_opt.sigma_cut_high = opt.athenak_sigma_cut_high;
    load_opt.emission_type = effective_emission_type(opt.emission_type,
                                                      KPOLARIS_IHARM_COMPILED_EMISSION_TYPE,
                                                      4);
    load_opt.profile_mode = opt.athenak_profile_mode;
    load_opt.nonthermal_kappa = opt.nonthermal_kappa;
    load_opt.variable_kappa = opt.variable_kappa;
    load_opt.variable_kappa_min = opt.variable_kappa_min;
    load_opt.variable_kappa_interp_start = opt.variable_kappa_interp_start;
    load_opt.variable_kappa_max = opt.variable_kappa_max;
    load_opt.powerlaw_p = opt.powerlaw_p;
    load_opt.powerlaw_eta = opt.powerlaw_eta;
    load_opt.powerlaw_gamma_min = opt.powerlaw_gamma_min;
    load_opt.powerlaw_gamma_max = opt.powerlaw_gamma_max;
    load_opt.powerlaw_gamma_cutoff = opt.powerlaw_gamma_cutoff;
    load_opt.resample_r_in = opt.athenak_r_in;
    load_opt.resample_r_out = opt.athenak_r_out;
    load_opt.timing = opt.timing;
    return load_opt;
}

struct AthenaKSlowLightStagedLoader {
    using model_type = kpolaris::AthenaKDirectRadiationModel<Real>;
    using slow_light_prefetch_result_type = kpolaris::AthenaKStagedDump;

    const Options* opt = nullptr;
    Real frequency = Real(230.0e9);

    kpolaris::AthenaKLoadOptions load_options(const std::string& path) const {
        kpolaris::AthenaKLoadOptions load_opt = make_athenak_load_options(*opt, frequency);
        load_opt.dump_path = path;
        return load_opt;
    }

    slow_light_prefetch_result_type prefetch(const std::string& path) const {
        return kpolaris::read_athenak_staged_dump(load_options(path));
    }

    model_type materialize(slow_light_prefetch_result_type&& staged) const {
        auto model = kpolaris::materialize_athenak_direct_model_from_staged(staged, load_options(std::string()));
        if (opt->outer_radius > Real(0)) {
            model.r_out = std::min(model.r_out, opt->outer_radius);
        }
        return model;
    }

    model_type load(const std::string& path) const {
        return materialize(prefetch(path));
    }
};

ImageHostData run_athenak_slow_light_image(const Options& opt,
                                           const kpolaris::PassAParams<Real>& pass_a,
                                           kpolaris::AthenaKDirectRadiationModel<Real> first_model,
                                           Real frequency) {
    const std::vector<std::string> dump_paths = build_slow_light_dump_paths(opt);
    const std::vector<Real> dump_times = build_model_slow_light_dump_times(
        opt, dump_paths, [](const std::string& path) { return kpolaris::read_athenak_dump_time(path); });
    AthenaKSlowLightStagedLoader load_model{&opt, frequency};
    first_model.freq_cgs = frequency;
    if (opt.outer_radius > Real(0)) {
        first_model.r_out = std::min(first_model.r_out, opt.outer_radius);
    }
    if (opt.athenak_dump.empty() || opt.athenak_dump != dump_paths.front()) {
        first_model = load_model.load(dump_paths.front());
    }
    return run_grmhd_slow_light_coordinate_image(
        opt, pass_a, dump_paths, dump_times, first_model, load_model, "athenak");
}

std::vector<ImageHostData> run_athenak_slow_light_multifrequency_image_data(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::AthenaKDirectRadiationModel<Real> first_model,
    const std::vector<Real>& frequencies) {
    const std::vector<std::string> dump_paths = build_slow_light_dump_paths(opt);
    const std::vector<Real> dump_times = build_model_slow_light_dump_times(
        opt, dump_paths, [](const std::string& path) { return kpolaris::read_athenak_dump_time(path); });
    auto apply_runtime_limits = [&](kpolaris::AthenaKDirectRadiationModel<Real>& model) {
        if (opt.outer_radius > Real(0)) {
            model.r_out = std::min(model.r_out, opt.outer_radius);
        }
    };
    first_model.freq_cgs = frequencies.front();
    AthenaKSlowLightStagedLoader first_frequency_loader{&opt, frequencies.front()};
    if (opt.athenak_dump.empty() || opt.athenak_dump != dump_paths.front()) {
        first_model = first_frequency_loader.load(dump_paths.front());
    } else {
        apply_runtime_limits(first_model);
    }
    return run_grmhd_slow_light_coordinate_multifrequency_images(
        opt, pass_a, dump_paths, dump_times, first_model, first_frequency_loader, frequencies);
}

#endif

#if KPOLARIS_ENABLE_MODEL_BHAC
kpolaris::BHACLoadOptions make_bhac_load_options(const Options& opt, Real frequency) {
    kpolaris::BHACLoadOptions load_opt;
    load_opt.freq = frequency;
    load_opt.M_unit = opt.bhac_M_unit;
    load_opt.mbh_solar = opt.bhac_mbh_solar;
    load_opt.trat_small = opt.bhac_trat_small;
    load_opt.trat_large = opt.bhac_trat_large;
    load_opt.beta_crit = opt.bhac_beta_crit;
    load_opt.gamma = opt.bhac_gamma;
    load_opt.sigma_cut = opt.bhac_sigma_cut;
    load_opt.sigma_cut_high = opt.bhac_sigma_cut_high;
    load_opt.emission_type = effective_emission_type(opt.emission_type,
                                                      KPOLARIS_IHARM_COMPILED_EMISSION_TYPE,
                                                      4);
    load_opt.profile_mode = opt.bhac_profile_mode;
    load_opt.nonthermal_kappa = opt.nonthermal_kappa;
    load_opt.variable_kappa = opt.variable_kappa;
    load_opt.variable_kappa_min = opt.variable_kappa_min;
    load_opt.variable_kappa_interp_start = opt.variable_kappa_interp_start;
    load_opt.variable_kappa_max = opt.variable_kappa_max;
    load_opt.powerlaw_p = opt.powerlaw_p;
    load_opt.powerlaw_eta = opt.powerlaw_eta;
    load_opt.powerlaw_gamma_min = opt.powerlaw_gamma_min;
    load_opt.powerlaw_gamma_max = opt.powerlaw_gamma_max;
    load_opt.powerlaw_gamma_cutoff = opt.powerlaw_gamma_cutoff;
    load_opt.nxlone1 = opt.bhac_nxlone1;
    load_opt.nxlone2 = opt.bhac_nxlone2;
    load_opt.nxlone3 = opt.bhac_nxlone3;
    load_opt.spin_index = opt.bhac_spin_index;
    load_opt.x1_min = opt.bhac_x1_min;
    load_opt.x1_max = opt.bhac_x1_max;
    load_opt.x2_min = opt.bhac_x2_min;
    load_opt.x2_max = opt.bhac_x2_max;
    load_opt.x3_min = opt.bhac_x3_min;
    load_opt.x3_max = opt.bhac_x3_max;
    load_opt.hslope = opt.bhac_hslope;
    load_opt.r_in = opt.bhac_r_in;
    load_opt.r_out = opt.bhac_r_out;
    load_opt.sfc = opt.bhac_sfc;
    load_opt.reverse_field = opt.bhac_reverse_field;
    load_opt.timing = opt.timing;
    return load_opt;
}

struct BHACSlowLightRawLoader {
    using model_type = kpolaris::BHACAMRRadiationModel<Real>;
    using slow_light_prefetch_result_type = kpolaris::BHACRawStagedDump;

    const Options* opt = nullptr;
    Real frequency = Real(230.0e9);

    kpolaris::BHACLoadOptions load_options(const std::string& path) const {
        kpolaris::BHACLoadOptions load_opt = make_bhac_load_options(*opt, frequency);
        load_opt.dump_path = path;
        load_opt.cache_path.clear();
        load_opt.cache_mode = kpolaris::BHACCacheOff;
        return load_opt;
    }

    slow_light_prefetch_result_type prefetch(const std::string& path) const {
        return kpolaris::read_bhac_raw_staged_dump(load_options(path));
    }

    model_type materialize(slow_light_prefetch_result_type&& raw) const {
        model_type model = kpolaris::materialize_bhac_model_from_raw(raw, load_options(std::string()));
        if (opt->outer_radius > Real(0)) {
            model.r_out = std::min(model.r_out, opt->outer_radius);
        }
        return model;
    }

    model_type load(const std::string& path) const {
        return materialize(prefetch(path));
    }
};

ImageHostData run_bhac_slow_light_image(const Options& opt,
                                        const kpolaris::PassAParams<Real>& pass_a,
                                        kpolaris::BHACAMRRadiationModel<Real> first_model,
                                        Real frequency) {
    const std::vector<std::string> dump_paths = build_slow_light_dump_paths(opt);
    const std::vector<Real> dump_times = build_model_slow_light_dump_times(
        opt, dump_paths, [](const std::string& path) { return kpolaris::read_bhac_dump_time(path); });
    BHACSlowLightRawLoader load_model{&opt, frequency};
    first_model.freq_cgs = frequency;
    if (opt.outer_radius > Real(0)) {
        first_model.r_out = std::min(first_model.r_out, opt.outer_radius);
    }
    if (opt.bhac_dump.empty() || opt.bhac_dump != dump_paths.front()) {
        first_model = load_model.load(dump_paths.front());
    }
    return run_grmhd_slow_light_coordinate_image(
        opt, pass_a, dump_paths, dump_times, first_model, load_model, "bhac");
}

std::vector<ImageHostData> run_bhac_slow_light_multifrequency_image_data(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::BHACAMRRadiationModel<Real> first_model,
    const std::vector<Real>& frequencies) {
    const std::vector<std::string> dump_paths = build_slow_light_dump_paths(opt);
    const std::vector<Real> dump_times = build_model_slow_light_dump_times(
        opt, dump_paths, [](const std::string& path) { return kpolaris::read_bhac_dump_time(path); });
    auto apply_runtime_limits = [&](kpolaris::BHACAMRRadiationModel<Real>& model) {
        if (opt.outer_radius > Real(0)) {
            model.r_out = std::min(model.r_out, opt.outer_radius);
        }
    };
    first_model.freq_cgs = frequencies.front();
    BHACSlowLightRawLoader first_frequency_loader{&opt, frequencies.front()};
    if (opt.bhac_dump.empty() || opt.bhac_dump != dump_paths.front()) {
        first_model = first_frequency_loader.load(dump_paths.front());
    } else {
        apply_runtime_limits(first_model);
    }
    return run_grmhd_slow_light_coordinate_multifrequency_images(
        opt, pass_a, dump_paths, dump_times, first_model, first_frequency_loader, frequencies);
}

#endif

#if KPOLARIS_ENABLE_MODEL_HAMR
kpolaris::HAMRLoadOptions make_hamr_load_options(const Options& opt, Real frequency) {
    kpolaris::HAMRLoadOptions load_opt;
    load_opt.freq = frequency;
    load_opt.M_unit = opt.hamr_M_unit;
    load_opt.mbh_solar = opt.hamr_mbh_solar;
    load_opt.trat_small = opt.hamr_trat_small;
    load_opt.trat_large = opt.hamr_trat_large;
    load_opt.beta_crit = opt.hamr_beta_crit;
    load_opt.gamma = opt.hamr_gamma;
    load_opt.sigma_cut = opt.hamr_sigma_cut;
    load_opt.sigma_cut_high = opt.hamr_sigma_cut_high;
    load_opt.emission_type = effective_emission_type(opt.emission_type,
                                                      KPOLARIS_IHARM_COMPILED_EMISSION_TYPE,
                                                      4);
    load_opt.profile_mode = opt.hamr_profile_mode;
    load_opt.nonthermal_kappa = opt.nonthermal_kappa;
    load_opt.variable_kappa = opt.variable_kappa;
    load_opt.variable_kappa_min = opt.variable_kappa_min;
    load_opt.variable_kappa_interp_start = opt.variable_kappa_interp_start;
    load_opt.variable_kappa_max = opt.variable_kappa_max;
    load_opt.powerlaw_p = opt.powerlaw_p;
    load_opt.powerlaw_eta = opt.powerlaw_eta;
    load_opt.powerlaw_gamma_min = opt.powerlaw_gamma_min;
    load_opt.powerlaw_gamma_max = opt.powerlaw_gamma_max;
    load_opt.powerlaw_gamma_cutoff = opt.powerlaw_gamma_cutoff;
    load_opt.r_in = opt.hamr_r_in;
    load_opt.r_out = opt.hamr_r_out;
    load_opt.hslope = opt.hamr_hslope;
    load_opt.reverse_field = opt.hamr_reverse_field;
    load_opt.id_order = opt.hamr_id_order;
    load_opt.root_order = opt.hamr_root_order;
    load_opt.timing = opt.timing;
    return load_opt;
}

struct HAMRSlowLightStagedLoader {
    using model_type = kpolaris::HAMRRadiationModel<Real>;
    using slow_light_prefetch_result_type = kpolaris::BHACStagedDump;

    const Options* opt = nullptr;
    Real frequency = Real(230.0e9);

    kpolaris::HAMRLoadOptions load_options(const std::string& path) const {
        kpolaris::HAMRLoadOptions load_opt = make_hamr_load_options(*opt, frequency);
        load_opt.dump_path = path;
        return load_opt;
    }

    slow_light_prefetch_result_type prefetch(const std::string& path) const {
        return kpolaris::read_hamr_staged_dump(load_options(path));
    }

    model_type materialize(slow_light_prefetch_result_type&& staged) const {
        model_type model = kpolaris::materialize_hamr_model_from_staged(staged, load_options(std::string()));
        if (opt->outer_radius > Real(0)) {
            model.r_out = std::min(model.r_out, opt->outer_radius);
        }
        return model;
    }

    model_type load(const std::string& path) const {
        return materialize(prefetch(path));
    }
};

ImageHostData run_hamr_slow_light_image(const Options& opt,
                                        const kpolaris::PassAParams<Real>& pass_a,
                                        kpolaris::HAMRRadiationModel<Real> first_model,
                                        Real frequency) {
    const std::vector<std::string> dump_paths = build_slow_light_dump_paths(opt);
    const std::vector<Real> dump_times = build_model_slow_light_dump_times(
        opt, dump_paths, [](const std::string& path) { return kpolaris::read_hamr_dump_time(path); });
    HAMRSlowLightStagedLoader load_model{&opt, frequency};
    first_model.freq_cgs = frequency;
    if (opt.outer_radius > Real(0)) {
        first_model.r_out = std::min(first_model.r_out, opt.outer_radius);
    }
    if (opt.hamr_dump.empty() || opt.hamr_dump != dump_paths.front()) {
        first_model = load_model.load(dump_paths.front());
    }
    return run_grmhd_slow_light_coordinate_image(
        opt, pass_a, dump_paths, dump_times, first_model, load_model, "hamr");
}

std::vector<ImageHostData> run_hamr_slow_light_multifrequency_image_data(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::HAMRRadiationModel<Real> first_model,
    const std::vector<Real>& frequencies) {
    const std::vector<std::string> dump_paths = build_slow_light_dump_paths(opt);
    const std::vector<Real> dump_times = build_model_slow_light_dump_times(
        opt, dump_paths, [](const std::string& path) { return kpolaris::read_hamr_dump_time(path); });
    auto apply_runtime_limits = [&](kpolaris::HAMRRadiationModel<Real>& model) {
        if (opt.outer_radius > Real(0)) {
            model.r_out = std::min(model.r_out, opt.outer_radius);
        }
    };
    first_model.freq_cgs = frequencies.front();
    HAMRSlowLightStagedLoader first_frequency_loader{&opt, frequencies.front()};
    if (opt.hamr_dump.empty() || opt.hamr_dump != dump_paths.front()) {
        first_model = first_frequency_loader.load(dump_paths.front());
    } else {
        apply_runtime_limits(first_model);
    }
    return run_grmhd_slow_light_coordinate_multifrequency_images(
        opt, pass_a, dump_paths, dump_times, first_model, first_frequency_loader, frequencies);
}
#endif

// Model-specific input decoding ends here; scheduling and output ordering are
// common to individual images, time batches, spectra, and diagnostics.
template<class Model>
std::vector<ImageHostData> run_model_slow_light_batch(
    const Options& opt, const kpolaris::PassAParams<Real>& pass_a, Model model,
    const std::vector<std::string>& paths, const std::vector<Real>& times,
    const std::vector<kpolaris_image_detail::SlowLightBatchJob>& jobs,
    const std::vector<Real>& frequencies) {
    [[maybe_unused]] auto run = [&](auto loader, const std::string& initial_path, bool reload = false) {
        using Traits = kpolaris_image_detail::SlowLightLoaderTraits<decltype(loader), Model>;
        Model first = initial_path == paths.front() && !reload ? model : Traits::load(loader, paths.front());
        return run_grmhd_slow_light_coordinate_batch(opt, pass_a, paths, times, first, loader, jobs, frequencies);
    };
#if KPOLARIS_ENABLE_MODEL_IHARM || KPOLARIS_ENABLE_MODEL_KHARMA
    if constexpr (std::is_same_v<Model, kpolaris::GRMHDRadiationModel<Real>>) {
#if KPOLARIS_ENABLE_MODEL_IHARM
        if (opt.model == "iharm") {
            auto loader = [&](const std::string& path) {
                auto load = make_iharm_load_options(opt, frequencies.front());
                load.dump_path = path;
                return kpolaris::load_iharm_model_from_hdf5(load);
            };
            return run(loader, opt.iharm_dump);
        }
#endif
#if KPOLARIS_ENABLE_MODEL_KHARMA
        if (opt.model == "kharma")
            return run(KHARMASlowLightStagedLoader{&opt, frequencies.front(), &paths, &times},
                       opt.kharma_dump, opt.kharma_ddc.native);
#endif
    }
#endif
#if KPOLARIS_ENABLE_MODEL_ATHENAK
    if constexpr (std::is_same_v<Model, kpolaris::AthenaKDirectRadiationModel<Real>>)
        return run(AthenaKSlowLightStagedLoader{&opt, frequencies.front()}, opt.athenak_dump);
#endif
#if KPOLARIS_ENABLE_MODEL_BHAC || KPOLARIS_ENABLE_MODEL_HAMR
    if constexpr (std::is_same_v<Model, kpolaris::BHACAMRRadiationModel<Real>>) {
#if KPOLARIS_ENABLE_MODEL_BHAC
        if (opt.model == "bhac") return run(BHACSlowLightRawLoader{&opt, frequencies.front()}, opt.bhac_dump);
#endif
#if KPOLARIS_ENABLE_MODEL_HAMR
        if (opt.model == "hamr") return run(HAMRSlowLightStagedLoader{&opt, frequencies.front()}, opt.hamr_dump);
#endif
    }
#endif
    throw std::invalid_argument("slow-light batches require a supported GRMHD input model");
}

#endif

template<class Model>
void set_model_frequency(Model& model, Real frequency) {
    model.freq_cgs = frequency;
}

#if KPOLARIS_ENABLE_MODEL_BINARY_RIAF
void set_model_frequency(kpolaris::BinaryRIAFRadiationModel<Real>& model,
                         Real frequency) {
    model.freq_cgs = frequency;
    model.disk.freq_cgs = frequency;
}
#endif

struct FrequencyImage {
    Real frequency = Real(0);
    std::vector<double> I_inv;
    std::vector<double> Q_inv;
    std::vector<double> U_inv;
    std::vector<double> V_inv;
    std::vector<double> I_nu;
    std::vector<double> Q_nu;
    std::vector<double> U_nu;
    std::vector<double> V_nu;
    std::vector<double> closure_x;
    std::vector<double> closure_k;
    std::vector<double> final_null;
    std::vector<double> frame_error;
    std::vector<double> det_r;
    std::vector<double> overlap_r11;
    std::vector<double> overlap_r12;
    std::vector<double> overlap_r21;
    std::vector<double> overlap_r22;
    std::vector<double> basis_identity_error;
    std::vector<double> basis_rotation_angle;
    std::vector<int> reason;
    std::vector<int> pass_a_steps;
    std::vector<int> steps;
    std::vector<int> total_steps;
    int has_analysis = 0;
    int has_observer_weighted_analysis = 0;
    kpolaris::ResponseConfig<Real> response_config;
    std::vector<double> response_data;
    std::vector<double> radiating_path_length;
    std::vector<double> emission_weight;
    std::vector<double> emission_weighted_radius;
    std::vector<double> emission_weighted_optical_depth_to_camera;
    std::vector<double> absorption_depth;
    std::vector<double> absorption_operator_depth;
    std::vector<double> faraday_rotation_depth;
    std::vector<double> faraday_conversion_depth;
    std::vector<double> faraday_operator_depth;
    std::vector<double> dominant_emission_radius;
    std::vector<int> dominant_emission_region;
    std::vector<double> dominant_ne_cgs;
    std::vector<double> dominant_thetae;
    std::vector<double> dominant_b_cgs;
    std::vector<double> dominant_beta;
    std::vector<double> dominant_sigma;
    std::vector<double> emission_weighted_ne_cgs;
    std::vector<double> emission_weighted_thetae;
    std::vector<double> emission_weighted_b_cgs;
    std::vector<double> emission_weighted_beta;
    std::vector<double> emission_weighted_sigma;
    std::vector<double> photon_ring_winding_estimate;
    std::vector<int> radiation_substeps;
    int analysis_radial_bins = 0;
    double analysis_radial_min = 0.0;
    double analysis_radial_max = 0.0;
    double analysis_formation_fraction = 0.0;
    std::vector<double> analysis_radial_bin_edges;
    std::vector<double> analysis_radial_bin_centers;
    std::vector<double> radial_stokes_i_contribution;
    std::vector<double> radial_stokes_q_contribution;
    std::vector<double> radial_stokes_u_contribution;
    std::vector<double> radial_stokes_v_contribution;
    std::vector<double> radial_absorption_depth;
    std::vector<double> radial_faraday_rotation_depth;
    std::vector<double> radial_faraday_conversion_depth;
    std::vector<double> radial_faraday_operator_depth;
    std::vector<double> observer_weighted_radius_i;
    std::vector<double> observer_weighted_radius_linear;
    std::vector<double> observer_weighted_radius_circular;
    std::vector<double> intensity_formation_radius_low;
    std::vector<double> intensity_formation_radius_median;
    std::vector<double> intensity_formation_radius_high;
    std::vector<double> linear_formation_radius_low;
    std::vector<double> linear_formation_radius_median;
    std::vector<double> linear_formation_radius_high;
    std::vector<double> circular_formation_radius_low;
    std::vector<double> circular_formation_radius_median;
    std::vector<double> circular_formation_radius_high;
    std::vector<double> los_linear_coherence;
    std::vector<double> los_circular_coherence;
    std::vector<double> contribution_closure_max_abs;
    std::vector<double> contribution_closure_relative_l1;
    std::vector<double> foreground_absorption_depth;
    std::vector<double> foreground_faraday_rotation_depth;
    std::vector<double> foreground_faraday_conversion_depth;
    std::vector<double> foreground_faraday_operator_depth;
    std::vector<double> foreground_faraday_operator_fraction;
    int returned = 0;
    double mean_pass_a_steps = 0.0;
    double mean_pass_b_steps = 0.0;
    double mean_total_steps = 0.0;
    int max_pass_a_steps = 0;
    int max_pass_b_steps = 0;
    int max_total_steps = 0;
    Real sum_i = Real(0);
    Real sum_q = Real(0);
    Real sum_u = Real(0);
    Real sum_v = Real(0);
    DiagnosticSummary diagnostics;
};

void write_physical_analysis_group(H5::Group& analysis,
                                   const FrequencyImage& image,
                                   const Options& opt) {
    write_h5_string(analysis, "evpa_0", opt.evpa_0);
    write_h5_string(analysis, "stokes_basis", "observer; same convention as parent image");
    write_response_analysis(analysis, image.response_config, image.response_data, opt);
    write_h5_string(analysis, "schema", "kpolaris_physical_analysis");
    write_h5_int(analysis, "schema_version", 2);
    write_h5_string(analysis, "emission_weight_definition",
                    "integral max(j_I,0) dlambda over radiating samples");
    write_h5_string(analysis, "emission_weighted_definition",
                    "line-of-sight emissivity moment: integral quantity max(j_I,0) dlambda divided by emission_weight");
    write_h5_string(analysis, "dominant_emission_definition",
                    "sample with maximum max(j_I,0) dlambda along the ray");
    write_h5_string(analysis, "absorption_depth_definition",
                    "integral max(alpha_I,0) dlambda");
    write_h5_string(analysis, "emission_weighted_optical_depth_to_camera_definition",
                    "emission-weighted remaining alpha_I optical depth from emission sample to camera");
    write_h5_string(analysis, "faraday_rotation_depth_definition",
                    "signed integral rho_V dlambda in the propagated screen basis");
    write_h5_string(analysis, "faraday_conversion_depth_definition",
                    "integral sqrt(rho_Q^2+rho_U^2) dlambda");
    write_h5_string(analysis, "operator_depth_definition",
                    "operator-norm depth used as a numerical stiffness diagnostic");
    write_h5_int(analysis, "observer_weighted_available",
                 image.has_observer_weighted_analysis);
    if (image.has_observer_weighted_analysis) {
        write_h5_string(analysis, "radial_contribution_definition",
                        "observer-basis Stokes contribution from source terms emitted in each logarithmic radial bin and propagated with the production transfer operator");
        write_h5_string(analysis, "radial_contribution_units", "invariant_stokes");
        write_h5_string(analysis, "radial_array_layout", "radial_bin,y,x");
        write_h5_string(analysis, "formation_weight_I", "max(radial_stokes_I_inv_contribution,0)");
        write_h5_string(analysis, "formation_weight_linear",
                        "sqrt(radial_stokes_Q_inv_contribution^2+radial_stokes_U_inv_contribution^2)");
        write_h5_string(analysis, "formation_weight_circular",
                        "abs(radial_stokes_V_inv_contribution)");
        write_h5_string(analysis, "foreground_definition",
                        "outer-radius proxy: raw depth in radial bins whose centers exceed intensity_formation_radius_high");
        write_h5_int(analysis, "radial_bins", image.analysis_radial_bins);
        write_h5_scalar(analysis, "radial_min", image.analysis_radial_min);
        write_h5_scalar(analysis, "radial_max", image.analysis_radial_max);
        write_h5_scalar(analysis, "formation_fraction", image.analysis_formation_fraction);
        write_h5_dataset_1d(analysis, "radial_bin_edges", image.analysis_radial_bin_edges);
        write_h5_dataset_1d(analysis, "radial_bin_centers", image.analysis_radial_bin_centers);
    } else {
        write_h5_string(analysis, "observer_weighted_unavailable_reason",
                        "observer-weighted radial decomposition is currently defined for fast-light transport; legacy slow-light analysis contains line-of-sight moments only");
    }

    write_h5_dataset_2d(analysis, "radiating_path_length", image.radiating_path_length, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "emission_weight", image.emission_weight, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "emission_weighted_radius", image.emission_weighted_radius, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "emission_weighted_optical_depth_to_camera", image.emission_weighted_optical_depth_to_camera, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "absorption_depth", image.absorption_depth, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "absorption_operator_depth", image.absorption_operator_depth, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "faraday_rotation_depth", image.faraday_rotation_depth, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "faraday_conversion_depth", image.faraday_conversion_depth, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "faraday_operator_depth", image.faraday_operator_depth, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "dominant_emission_radius", image.dominant_emission_radius, opt.nx, opt.ny);
    write_h5_dataset_2d_int(analysis, "dominant_emission_region", image.dominant_emission_region, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "dominant_ne_cgs", image.dominant_ne_cgs, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "dominant_thetae", image.dominant_thetae, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "dominant_b_cgs", image.dominant_b_cgs, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "dominant_beta", image.dominant_beta, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "dominant_sigma", image.dominant_sigma, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "emission_weighted_ne_cgs", image.emission_weighted_ne_cgs, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "emission_weighted_thetae", image.emission_weighted_thetae, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "emission_weighted_b_cgs", image.emission_weighted_b_cgs, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "emission_weighted_beta", image.emission_weighted_beta, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "emission_weighted_sigma", image.emission_weighted_sigma, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "photon_ring_winding_estimate", image.photon_ring_winding_estimate, opt.nx, opt.ny);
    write_h5_dataset_2d_int(analysis, "radiation_substeps", image.radiation_substeps, opt.nx, opt.ny);

    if (image.has_observer_weighted_analysis) {
    write_h5_dataset_3d(analysis, "radial_stokes_I_inv_contribution", image.radial_stokes_i_contribution,
                        image.analysis_radial_bins, opt.nx, opt.ny);
    write_h5_dataset_3d(analysis, "radial_stokes_Q_inv_contribution", image.radial_stokes_q_contribution,
                        image.analysis_radial_bins, opt.nx, opt.ny);
    write_h5_dataset_3d(analysis, "radial_stokes_U_inv_contribution", image.radial_stokes_u_contribution,
                        image.analysis_radial_bins, opt.nx, opt.ny);
    write_h5_dataset_3d(analysis, "radial_stokes_V_inv_contribution", image.radial_stokes_v_contribution,
                        image.analysis_radial_bins, opt.nx, opt.ny);
    write_h5_dataset_3d(analysis, "radial_absorption_depth", image.radial_absorption_depth,
                        image.analysis_radial_bins, opt.nx, opt.ny);
    write_h5_dataset_3d(analysis, "radial_faraday_rotation_depth", image.radial_faraday_rotation_depth,
                        image.analysis_radial_bins, opt.nx, opt.ny);
    write_h5_dataset_3d(analysis, "radial_faraday_conversion_depth", image.radial_faraday_conversion_depth,
                        image.analysis_radial_bins, opt.nx, opt.ny);
    write_h5_dataset_3d(analysis, "radial_faraday_operator_depth", image.radial_faraday_operator_depth,
                        image.analysis_radial_bins, opt.nx, opt.ny);

    write_h5_dataset_2d(analysis, "observer_weighted_radius_I", image.observer_weighted_radius_i, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "observer_weighted_radius_linear", image.observer_weighted_radius_linear, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "observer_weighted_radius_circular", image.observer_weighted_radius_circular, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "intensity_formation_radius_low", image.intensity_formation_radius_low, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "intensity_formation_radius_median", image.intensity_formation_radius_median, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "intensity_formation_radius_high", image.intensity_formation_radius_high, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "linear_formation_radius_low", image.linear_formation_radius_low, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "linear_formation_radius_median", image.linear_formation_radius_median, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "linear_formation_radius_high", image.linear_formation_radius_high, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "circular_formation_radius_low", image.circular_formation_radius_low, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "circular_formation_radius_median", image.circular_formation_radius_median, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "circular_formation_radius_high", image.circular_formation_radius_high, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "los_linear_coherence", image.los_linear_coherence, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "los_circular_coherence", image.los_circular_coherence, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "contribution_closure_max_abs", image.contribution_closure_max_abs, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "contribution_closure_relative_l1", image.contribution_closure_relative_l1, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "foreground_absorption_depth", image.foreground_absorption_depth, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "foreground_faraday_rotation_depth", image.foreground_faraday_rotation_depth, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "foreground_faraday_conversion_depth", image.foreground_faraday_conversion_depth, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "foreground_faraday_operator_depth", image.foreground_faraday_operator_depth, opt.nx, opt.ny);
    write_h5_dataset_2d(analysis, "foreground_faraday_operator_fraction", image.foreground_faraday_operator_fraction, opt.nx, opt.ny);
    }
}
#if KPOLARIS_ENABLE_MODEL_RIAF || KPOLARIS_ENABLE_MODEL_TORUS || KPOLARIS_ENABLE_MODEL_IHARM || KPOLARIS_ENABLE_MODEL_KHARMA || KPOLARIS_ENABLE_MODEL_ATHENAK || KPOLARIS_ENABLE_MODEL_BHAC || KPOLARIS_ENABLE_MODEL_HAMR
int effective_multifrequency_chunk_size(const Options& opt, size_t nfreq) {
    if (opt.analysis_mode && !opt.slow_light) return 1;
    int chunk = opt.multifrequency_chunk_size;
    if (chunk <= 0) {
        chunk = KPOLARIS_MAX_FREQUENCIES;
    }
    chunk = std::max(chunk, 1);
    chunk = std::min(chunk, KPOLARIS_MAX_FREQUENCIES);
    chunk = std::min(chunk, static_cast<int>(nfreq));
    return std::max(chunk, 1);
}

FrequencyImage pack_transport_frequency_image(const Options& opt,
                                              const ImageHostData& data,
                                                    Real frequency,
                                                    int freq_index,
                                                    int source_freq_index,
                                                    bool hdf5_output,
                                                    bool multi_frequency,
                                                    std::ofstream& out) {
    Kokkos::Timer pack_timer;
    const int npix = opt.nx * opt.ny;
    const size_t image_offset = static_cast<size_t>(source_freq_index) * static_cast<size_t>(npix);

    FrequencyImage image;
    image.frequency = frequency;
    image.I_inv.resize(npix);
    image.Q_inv.resize(npix);
    image.U_inv.resize(npix);
    image.V_inv.resize(npix);
    image.I_nu.resize(npix);
    image.Q_nu.resize(npix);
    image.U_nu.resize(npix);
    image.V_nu.resize(npix);
    image.closure_x.resize(npix);
    image.closure_k.resize(npix);
    image.final_null.resize(npix);
    image.frame_error.resize(npix);
    image.det_r.resize(npix);
    image.overlap_r11.resize(npix);
    image.overlap_r12.resize(npix);
    image.overlap_r21.resize(npix);
    image.overlap_r22.resize(npix);
    image.basis_identity_error.resize(npix);
    image.basis_rotation_angle.resize(npix);
    image.reason.resize(npix);
    image.pass_a_steps.resize(npix);
    image.steps.resize(npix);
    image.total_steps.resize(npix);
    image.has_analysis = data.has_analysis;
    image.response_config = data.response_config;
    image.response_data.assign(data.response_data.begin(), data.response_data.end());
    // Every response channel is an observed I,Q,U,V image, including reruns.
    if (opt.evpa_0 == "N") {
        for (size_t i = 0; i < image.response_data.size(); ++i) {
            const size_t component = (i / static_cast<size_t>(npix)) % 4;
            if (component == 1 || component == 2) image.response_data[i] = -image.response_data[i];
        }
    }
    image.has_observer_weighted_analysis =
        image.has_analysis && data.analysis_radial_bins > 0;
    if (image.has_analysis) {
        image.radiating_path_length.resize(npix);
        image.emission_weight.resize(npix);
        image.emission_weighted_radius.resize(npix);
        image.emission_weighted_optical_depth_to_camera.resize(npix);
        image.absorption_depth.resize(npix);
        image.absorption_operator_depth.resize(npix);
        image.faraday_rotation_depth.resize(npix);
        image.faraday_conversion_depth.resize(npix);
        image.faraday_operator_depth.resize(npix);
        image.dominant_emission_radius.resize(npix);
        image.dominant_emission_region.resize(npix);
        image.dominant_ne_cgs.resize(npix);
        image.dominant_thetae.resize(npix);
        image.dominant_b_cgs.resize(npix);
        image.dominant_beta.resize(npix);
        image.dominant_sigma.resize(npix);
        image.emission_weighted_ne_cgs.resize(npix);
        image.emission_weighted_thetae.resize(npix);
        image.emission_weighted_b_cgs.resize(npix);
        image.emission_weighted_beta.resize(npix);
        image.emission_weighted_sigma.resize(npix);
        image.photon_ring_winding_estimate.resize(npix);
        image.radiation_substeps.resize(npix);
    }
    if (image.has_observer_weighted_analysis) {
        image.analysis_radial_bins = data.analysis_radial_bins;
        image.analysis_radial_min = static_cast<double>(data.analysis_radial_min);
        image.analysis_radial_max = static_cast<double>(data.analysis_radial_max);
        image.analysis_formation_fraction =
            static_cast<double>(data.analysis_formation_fraction);
        image.analysis_radial_bin_edges.assign(
            data.analysis_radial_bin_edges.begin(), data.analysis_radial_bin_edges.end());
        image.analysis_radial_bin_centers.assign(
            data.analysis_radial_bin_centers.begin(), data.analysis_radial_bin_centers.end());
        image.observer_weighted_radius_i.resize(npix);
        image.observer_weighted_radius_linear.resize(npix);
        image.observer_weighted_radius_circular.resize(npix);
        image.intensity_formation_radius_low.resize(npix);
        image.intensity_formation_radius_median.resize(npix);
        image.intensity_formation_radius_high.resize(npix);
        image.linear_formation_radius_low.resize(npix);
        image.linear_formation_radius_median.resize(npix);
        image.linear_formation_radius_high.resize(npix);
        image.circular_formation_radius_low.resize(npix);
        image.circular_formation_radius_median.resize(npix);
        image.circular_formation_radius_high.resize(npix);
        image.los_linear_coherence.resize(npix);
        image.los_circular_coherence.resize(npix);
        image.contribution_closure_max_abs.resize(npix);
        image.contribution_closure_relative_l1.resize(npix);
        image.foreground_absorption_depth.resize(npix);
        image.foreground_faraday_rotation_depth.resize(npix);
        image.foreground_faraday_conversion_depth.resize(npix);
        image.foreground_faraday_operator_depth.resize(npix);
        image.foreground_faraday_operator_fraction.resize(npix);
    }

    long long sum_pass_a_steps = 0;
    long long sum_pass_b_steps = 0;
    long long sum_total_steps = 0;
    const Real nu3 = frequency * frequency * frequency;
    for (int p = 0; p < npix; ++p) {
#if KPOLARIS_ENABLE_CSV_OUTPUT
        const int ix = p % opt.nx;
        const int iy = p / opt.nx;
#endif
        const size_t image_index = image_offset + static_cast<size_t>(p);
        image.reason[p] = data.reason[p];
        const bool reached_camera =
            image.reason[p] == static_cast<int>(kpolaris::TerminationReason::reached_camera);
        const Real i_inv = reached_camera ? data.image_i[image_index] : Real(0);
        const Real q_inv = reached_camera ? Real(output_qu_sign(opt.evpa_0)) * data.image_q[image_index] : Real(0);
        const Real u_inv = reached_camera ? Real(output_qu_sign(opt.evpa_0)) * data.image_u[image_index] : Real(0);
        const Real v_inv = reached_camera ? data.image_v[image_index] : Real(0);
        image.I_inv[p] = static_cast<double>(i_inv);
        image.Q_inv[p] = static_cast<double>(q_inv);
        image.U_inv[p] = static_cast<double>(u_inv);
        image.V_inv[p] = static_cast<double>(v_inv);
        image.I_nu[p] = static_cast<double>(i_inv * nu3);
        image.Q_nu[p] = static_cast<double>(q_inv * nu3);
        image.U_nu[p] = static_cast<double>(u_inv * nu3);
        image.V_nu[p] = static_cast<double>(v_inv * nu3);
        image.closure_x[p] = static_cast<double>(data.closure_x[p]);
        image.closure_k[p] = static_cast<double>(data.closure_k[p]);
        image.final_null[p] = static_cast<double>(data.final_null[p]);
        image.frame_error[p] = static_cast<double>(data.frame_error[p]);
        image.det_r[p] = static_cast<double>(data.det_r[p]);
        image.overlap_r11[p] = static_cast<double>(data.overlap_r11[p]);
        image.overlap_r12[p] = static_cast<double>(data.overlap_r12[p]);
        image.overlap_r21[p] = static_cast<double>(data.overlap_r21[p]);
        image.overlap_r22[p] = static_cast<double>(data.overlap_r22[p]);
        image.basis_identity_error[p] = static_cast<double>(data.basis_identity_error[p]);
        image.basis_rotation_angle[p] = static_cast<double>(data.basis_rotation_angle[p]);
        image.pass_a_steps[p] = data.pass_a_steps[p];
        image.steps[p] = data.steps[p];
        image.total_steps[p] = image.pass_a_steps[p] + image.steps[p];
        if (image.has_analysis) {
            image.radiating_path_length[p] = static_cast<double>(data.radiating_path_length[p]);
            image.emission_weight[p] = static_cast<double>(data.emission_weight[p]);
            image.emission_weighted_radius[p] = static_cast<double>(data.emission_weighted_radius[p]);
            image.emission_weighted_optical_depth_to_camera[p] = static_cast<double>(data.emission_weighted_optical_depth_to_camera[p]);
            image.absorption_depth[p] = static_cast<double>(data.absorption_depth[p]);
            image.absorption_operator_depth[p] = static_cast<double>(data.absorption_operator_depth[p]);
            image.faraday_rotation_depth[p] = static_cast<double>(data.faraday_rotation_depth[p]);
            image.faraday_conversion_depth[p] = static_cast<double>(data.faraday_conversion_depth[p]);
            image.faraday_operator_depth[p] = static_cast<double>(data.faraday_operator_depth[p]);
            image.dominant_emission_radius[p] = static_cast<double>(data.dominant_emission_radius[p]);
            image.dominant_emission_region[p] = data.dominant_emission_region[p];
            image.dominant_ne_cgs[p] = static_cast<double>(data.dominant_ne_cgs[p]);
            image.dominant_thetae[p] = static_cast<double>(data.dominant_thetae[p]);
            image.dominant_b_cgs[p] = static_cast<double>(data.dominant_b_cgs[p]);
            image.dominant_beta[p] = static_cast<double>(data.dominant_beta[p]);
            image.dominant_sigma[p] = static_cast<double>(data.dominant_sigma[p]);
            image.emission_weighted_ne_cgs[p] = static_cast<double>(data.emission_weighted_ne_cgs[p]);
            image.emission_weighted_thetae[p] = static_cast<double>(data.emission_weighted_thetae[p]);
            image.emission_weighted_b_cgs[p] = static_cast<double>(data.emission_weighted_b_cgs[p]);
            image.emission_weighted_beta[p] = static_cast<double>(data.emission_weighted_beta[p]);
            image.emission_weighted_sigma[p] = static_cast<double>(data.emission_weighted_sigma[p]);
            image.photon_ring_winding_estimate[p] = static_cast<double>(data.photon_ring_winding_estimate[p]);
            image.radiation_substeps[p] = data.radiation_substeps[p];
        }
        if (image.has_observer_weighted_analysis) {
            image.observer_weighted_radius_i[p] = static_cast<double>(data.observer_weighted_radius_i[p]);
            image.observer_weighted_radius_linear[p] = static_cast<double>(data.observer_weighted_radius_linear[p]);
            image.observer_weighted_radius_circular[p] = static_cast<double>(data.observer_weighted_radius_circular[p]);
            image.intensity_formation_radius_low[p] = static_cast<double>(data.intensity_formation_radius_low[p]);
            image.intensity_formation_radius_median[p] = static_cast<double>(data.intensity_formation_radius_median[p]);
            image.intensity_formation_radius_high[p] = static_cast<double>(data.intensity_formation_radius_high[p]);
            image.linear_formation_radius_low[p] = static_cast<double>(data.linear_formation_radius_low[p]);
            image.linear_formation_radius_median[p] = static_cast<double>(data.linear_formation_radius_median[p]);
            image.linear_formation_radius_high[p] = static_cast<double>(data.linear_formation_radius_high[p]);
            image.circular_formation_radius_low[p] = static_cast<double>(data.circular_formation_radius_low[p]);
            image.circular_formation_radius_median[p] = static_cast<double>(data.circular_formation_radius_median[p]);
            image.circular_formation_radius_high[p] = static_cast<double>(data.circular_formation_radius_high[p]);
            image.los_linear_coherence[p] = static_cast<double>(data.los_linear_coherence[p]);
            image.los_circular_coherence[p] = static_cast<double>(data.los_circular_coherence[p]);
            image.contribution_closure_max_abs[p] = static_cast<double>(data.contribution_closure_max_abs[p]);
            image.contribution_closure_relative_l1[p] = static_cast<double>(data.contribution_closure_relative_l1[p]);
            image.foreground_absorption_depth[p] = static_cast<double>(data.foreground_absorption_depth[p]);
            image.foreground_faraday_rotation_depth[p] = static_cast<double>(data.foreground_faraday_rotation_depth[p]);
            image.foreground_faraday_conversion_depth[p] = static_cast<double>(data.foreground_faraday_conversion_depth[p]);
            image.foreground_faraday_operator_depth[p] = static_cast<double>(data.foreground_faraday_operator_depth[p]);
            image.foreground_faraday_operator_fraction[p] = static_cast<double>(data.foreground_faraday_operator_fraction[p]);
        }
        sum_pass_a_steps += image.pass_a_steps[p];
        sum_pass_b_steps += image.steps[p];
        sum_total_steps += image.total_steps[p];
        image.max_pass_a_steps = std::max(image.max_pass_a_steps, image.pass_a_steps[p]);
        image.max_pass_b_steps = std::max(image.max_pass_b_steps, image.steps[p]);
        image.max_total_steps = std::max(image.max_total_steps, image.total_steps[p]);

#if KPOLARIS_ENABLE_CSV_OUTPUT
        if (!hdf5_output) {
            if (multi_frequency) {
                out << freq_index << ',' << static_cast<double>(frequency) << ',';
            }
            out << ix << ',' << iy << ','
                << image.I_inv[p] << ',' << image.Q_inv[p] << ','
                << image.U_inv[p] << ',' << image.V_inv[p] << ','
                << image.I_nu[p] << ',' << image.Q_nu[p] << ','
                << image.U_nu[p] << ',' << image.V_nu[p] << ','
                << image.reason[p] << ',' << image.pass_a_steps[p] << ','
                << image.steps[p] << ',' << image.total_steps[p] << ','
                << image.closure_x[p] << ',' << image.closure_k[p] << ','
                << image.final_null[p] << ',' << image.frame_error[p] << ','
                << image.det_r[p] << ',' << image.overlap_r11[p] << ','
                << image.overlap_r12[p] << ',' << image.overlap_r21[p] << ','
                << image.overlap_r22[p] << ',' << image.basis_identity_error[p]
                << ',' << image.basis_rotation_angle[p] << '\n';
        }
#else
        (void)hdf5_output;
        (void)freq_index;
        (void)multi_frequency;
        (void)out;
#endif
        if (reached_camera) {
            image.returned += 1;
            image.sum_i += i_inv;
            image.sum_q += q_inv;
            image.sum_u += u_inv;
            image.sum_v += v_inv;
        }
    }

    if (image.has_observer_weighted_analysis) {
        image.radial_stokes_i_contribution.assign(
            data.radial_stokes_i_contribution.begin(), data.radial_stokes_i_contribution.end());
        image.radial_stokes_q_contribution.assign(
            data.radial_stokes_q_contribution.begin(), data.radial_stokes_q_contribution.end());
        image.radial_stokes_u_contribution.assign(
            data.radial_stokes_u_contribution.begin(), data.radial_stokes_u_contribution.end());
        image.radial_stokes_v_contribution.assign(
            data.radial_stokes_v_contribution.begin(), data.radial_stokes_v_contribution.end());
        image.radial_absorption_depth.assign(
            data.radial_absorption_depth.begin(), data.radial_absorption_depth.end());
        image.radial_faraday_rotation_depth.assign(
            data.radial_faraday_rotation_depth.begin(), data.radial_faraday_rotation_depth.end());
        image.radial_faraday_conversion_depth.assign(
            data.radial_faraday_conversion_depth.begin(), data.radial_faraday_conversion_depth.end());
        image.radial_faraday_operator_depth.assign(
            data.radial_faraday_operator_depth.begin(), data.radial_faraday_operator_depth.end());
    }

    if (opt.evpa_0 == "N") {
        for (auto& q : image.radial_stokes_q_contribution) q = -q;
        for (auto& u : image.radial_stokes_u_contribution) u = -u;
    }
    if (npix > 0) {
        image.mean_pass_a_steps = static_cast<double>(sum_pass_a_steps) / static_cast<double>(npix);
        image.mean_pass_b_steps = static_cast<double>(sum_pass_b_steps) / static_cast<double>(npix);
        image.mean_total_steps = static_cast<double>(sum_total_steps) / static_cast<double>(npix);
    }
    image.diagnostics = summarize_diagnostics(opt, image.closure_x, image.closure_k,
                                              image.frame_error,
                                              image.basis_identity_error,
                                              image.reason);
    report_timing(opt, "image_host_pack_single", pack_timer.seconds());
    return image;
}

#if KPOLARIS_ENABLE_MODEL_IHARM || KPOLARIS_ENABLE_MODEL_KHARMA
FrequencyImage compute_frequency_image(const Options& opt,
                                       const kpolaris::PassAParams<Real>& pass_a,
                                       kpolaris::GRMHDRadiationModel<Real> model,
                                       Real frequency,
                                       int freq_index,
                                       bool hdf5_output,
                                       bool multi_frequency,
                                       std::ofstream& out) {
    set_model_frequency(model, frequency);
    ImageHostData data;
    if (opt.slow_light) {
#if KPOLARIS_ENABLE_SLOW_LIGHT
        if (opt.analysis_mode) {
#if !KPOLARIS_ENABLE_ANALYSIS_MODE
            throw std::runtime_error("analysis_mode requested but KPOLARIS_ENABLE_ANALYSIS_MODE is OFF");
#endif
        }
        if (opt.model == "iharm") {
#if KPOLARIS_ENABLE_MODEL_IHARM
            data = run_iharm_slow_light_image(opt, pass_a, model, frequency);
#else
            throw std::runtime_error("slow_light requested for iharm but iharm support is not built");
#endif
        } else if (opt.model == "kharma") {
#if KPOLARIS_ENABLE_MODEL_KHARMA
            data = run_kharma_slow_light_image(opt, pass_a, model, frequency);
#else
            throw std::runtime_error("slow_light requested for kharma but kharma support is not built");
#endif
        } else {
            throw std::runtime_error("slow_light is supported for iharm, kharma, and athenak only");
        }
#else
        throw std::runtime_error("slow_light requested but KPOLARIS_ENABLE_SLOW_LIGHT is OFF");
#endif
    } else if (opt.analysis_mode) {
#if KPOLARIS_ENABLE_ANALYSIS_MODE
        data = run_grmhd_analysis_image(
            pass_a, make_analysis_config(opt, pass_a), model,
            opt.radiation_substeps, opt.timing, opt.split_transport);
#else
        throw std::runtime_error("analysis_mode requested but KPOLARIS_ENABLE_ANALYSIS_MODE is OFF");
#endif
    } else {
        data = run_grmhd_image(
            pass_a, model, opt.radiation_substeps, opt.timing, opt.split_transport);
    }
    return pack_transport_frequency_image(opt, data, frequency, freq_index, 0,
                                      hdf5_output, multi_frequency, out);
}
#endif
#endif


#if KPOLARIS_ENABLE_MODEL_RIAF
void write_hdf5_model_metadata(H5::H5Object& obj,
                               const kpolaris::RIAFAnalyticRadiationModel<Real>& model) {
    write_h5_scalar(obj, "riaf_r_min", model.r_min);
    write_h5_scalar(obj, "riaf_r_max", model.r_max);
    write_h5_scalar(obj, "riaf_nth0", model.nth0);
    write_h5_scalar(obj, "riaf_Te0", model.Te0);
    write_h5_scalar(obj, "riaf_disk_h", model.disk_h);
    write_h5_scalar(obj, "riaf_pow_nth", model.pow_nth);
    write_h5_scalar(obj, "riaf_pow_T", model.pow_T);
    write_h5_scalar(obj, "riaf_ne_unit", model.ne_unit);
    write_h5_scalar(obj, "riaf_te_unit", model.te_unit);
    write_h5_scalar(obj, "riaf_mbh_solar", model.mbh_solar);
    write_h5_scalar(obj, "riaf_keplerian_factor", model.keplerian_factor);
    write_h5_scalar(obj, "riaf_infall_factor", model.infall_factor);
    write_h5_int(obj, "emission_type", model.emission_type);
    write_h5_string(obj, "emission_fit", emission_fit_name(model.emission_type));
    write_h5_scalar(obj, "nonthermal_kappa", model.nonthermal_kappa);
    write_h5_int(obj, "variable_kappa", model.variable_kappa);
    write_h5_scalar(obj, "variable_kappa_min", model.variable_kappa_min);
    write_h5_scalar(obj, "variable_kappa_interp_start", model.variable_kappa_interp_start);
    write_h5_scalar(obj, "variable_kappa_max", model.variable_kappa_max);
    write_h5_scalar(obj, "powerlaw_p", model.powerlaw_p);
    write_h5_scalar(obj, "powerlaw_eta", model.powerlaw_eta);
    write_h5_scalar(obj, "powerlaw_gamma_min", model.powerlaw_gamma_min);
    write_h5_scalar(obj, "powerlaw_gamma_max", model.powerlaw_gamma_max);
    write_h5_scalar(obj, "powerlaw_gamma_cutoff", model.powerlaw_gamma_cutoff);
    write_h5_scalar(obj, "dlambda_scale", model.dlambda_scale());
}
#endif

#if KPOLARIS_ENABLE_MODEL_BINARY_RIAF
void write_hdf5_model_metadata(H5::H5Object& obj,
                               const kpolaris::BinaryRIAFRadiationModel<Real>& model) {
    write_hdf5_model_metadata(obj, model.disk);
    write_h5_string(obj, "spacetime", "SuperposedKerrSchild");
    const bool table = model.trajectory.is_tabulated();
    write_h5_int(obj, "dynamic_spacetime", table || model.orbit_enabled);
    if (!table) {
        write_h5_scalar(obj, "binary_mass_ratio", model.mass_ratio);
        write_h5_scalar(obj, "binary_chi1", model.chi1);
        write_h5_scalar(obj, "binary_chi2", model.chi2);
        write_h5_scalar(obj, "binary_reference_separation", model.reference_separation);
        write_h5_scalar(obj, "binary_reference_phase", model.reference_phase);
        write_h5_scalar(obj, "binary_reference_time", model.reference_time);
    }
    write_h5_scalar(obj, "binary_observation_time", model.observation_time);
    write_h5_scalar(obj, "binary_trajectory_time_offset",
                    model.trajectory_time_offset);
    write_h5_scalar(obj, "binary_resolved_observation_time",
                    model.observation_time + model.trajectory_time_offset);
    write_h5_scalar(obj, "binary_minimum_separation", model.minimum_separation);
    write_h5_int(obj, "binary_inspiral", model.inspiral_enabled);
    write_h5_int(obj, "binary_orbit", model.orbit_enabled);
    write_h5_string(obj, "binary_trajectory_mode",
                    model.trajectory.is_tabulated() ? "tabulated" :
                    "leading_quadrupole");
    write_h5_int(obj, "binary_trajectory_samples",
                 model.trajectory.sample_count);
    write_h5_scalar(obj, "binary_trajectory_time_min",
                    model.trajectory.table_time_min);
    write_h5_scalar(obj, "binary_trajectory_time_max",
                    model.trajectory.table_time_max);
    write_h5_int(obj, "binary_trajectory_exact_remnant_tail",
                 model.trajectory.has_exact_postmerger_tail);
    if (table) {
        const auto state = model.trajectory.state(
            model.observation_time + model.trajectory_time_offset);
        write_h5_scalar(obj, "binary_observation_mass1", state.mass1);
        write_h5_scalar(obj, "binary_observation_mass2", state.mass2);
        write_h5_scalar(obj, "binary_observation_mass_ratio_m2_over_m1",
                        state.mass2 / state.mass1);
        write_h5_scalar(obj, "binary_observation_position1_x", state.position1.x);
        write_h5_scalar(obj, "binary_observation_position1_y", state.position1.y);
        write_h5_scalar(obj, "binary_observation_position1_z", state.position1.z);
        write_h5_scalar(obj, "binary_observation_position2_x", state.position2.x);
        write_h5_scalar(obj, "binary_observation_position2_y", state.position2.y);
        write_h5_scalar(obj, "binary_observation_position2_z", state.position2.z);
        write_h5_scalar(obj, "binary_observation_velocity1_x", state.velocity1.x);
        write_h5_scalar(obj, "binary_observation_velocity1_y", state.velocity1.y);
        write_h5_scalar(obj, "binary_observation_velocity1_z", state.velocity1.z);
        write_h5_scalar(obj, "binary_observation_velocity2_x", state.velocity2.x);
        write_h5_scalar(obj, "binary_observation_velocity2_y", state.velocity2.y);
        write_h5_scalar(obj, "binary_observation_velocity2_z", state.velocity2.z);
        write_h5_scalar(obj, "binary_observation_kerr_a1_x", state.kerr_a1.x);
        write_h5_scalar(obj, "binary_observation_kerr_a1_y", state.kerr_a1.y);
        write_h5_scalar(obj, "binary_observation_kerr_a1_z", state.kerr_a1.z);
        write_h5_scalar(obj, "binary_observation_kerr_a2_x", state.kerr_a2.x);
        write_h5_scalar(obj, "binary_observation_kerr_a2_y", state.kerr_a2.y);
        write_h5_scalar(obj, "binary_observation_kerr_a2_z", state.kerr_a2.z);
        write_h5_scalar(obj, "binary_observation_separation", state.separation);
        write_h5_scalar(obj, "binary_observation_phase", state.phase);
        write_h5_scalar(obj, "binary_observation_merger_weight",
                        state.merger_weight);
    }
    write_h5_scalar(obj, "binary_metric_derivative_step", model.metric_derivative_step);
    write_h5_scalar(obj, "binary_capture_factor", model.capture_factor);
    write_h5_scalar(obj, "binary_sampled_min_inverse_denominator",
                    model.sampled_min_inverse_denominator);
    write_h5_scalar(obj, "binary_sampled_min_fluid_slice_timelike_margin",
                    model.sampled_min_fluid_slice_timelike_margin);
    write_h5_scalar(obj, "binary_tidal_fraction", model.tidal_fraction);
    write_h5_scalar(obj, "binary_taper_start_fraction", model.taper_start_fraction);
    write_h5_scalar(obj, "binary_density_scale1", model.density_scale1);
    write_h5_scalar(obj, "binary_density_scale2", model.density_scale2);
    write_h5_scalar(obj, "binary_temperature_scale1", model.temperature_scale1);
    write_h5_scalar(obj, "binary_temperature_scale2", model.temperature_scale2);
    write_h5_int(obj, "binary_field_polarity1", model.field_polarity1);
    write_h5_int(obj, "binary_field_polarity2", model.field_polarity2);
}
#endif

#if KPOLARIS_ENABLE_MODEL_TORUS
void write_hdf5_model_metadata(H5::H5Object& obj,
                               const kpolaris::MagnetizedTorusRadiationModel<Real>& model) {
    write_h5_scalar(obj, "torus_l_lambda", model.l_lambda);
    write_h5_scalar(obj, "torus_wwin", model.wwin);
    write_h5_scalar(obj, "torus_kappa", model.kappa);
    write_h5_scalar(obj, "torus_omegac", model.omegac);
    write_h5_scalar(obj, "torus_betac", model.betac);
    write_h5_scalar(obj, "torus_beta", model.beta);
    write_h5_scalar(obj, "torus_Rhigh", model.Rhigh);
    write_h5_scalar(obj, "torus_bh_mass_solar", model.bh_mass_solar);
    write_h5_scalar(obj, "torus_mdot_cgs", model.accretion_rate_cgs);
    write_h5_scalar(obj, "torus_mdot_code", model.accretion_rate_code);
    write_h5_scalar(obj, "torus_thetae_min", model.thetae_min);
    write_h5_scalar(obj, "torus_l0", model.l0);
    write_h5_scalar(obj, "torus_rcusp", model.rcusp);
    write_h5_scalar(obj, "torus_rc", model.rc);
    write_h5_scalar(obj, "torus_r_outer", model.r_outer);
    write_h5_scalar(obj, "torus_Wc", model.Wc);
    write_h5_scalar(obj, "torus_Win", model.Win);
    write_h5_scalar(obj, "dlambda_scale", model.dlambda_scale());
}
#endif


std::string effective_parameter_output_path(const Options& opt) {
    if (opt.parameter_output == "none" || opt.parameter_output == "off" || opt.parameter_output == "0") {
        return {};
    }
    if (opt.parameter_output.empty() || opt.parameter_output == "auto") {
        return opt.output + ".params";
    }
    return opt.parameter_output;
}

template<class Model>
void write_effective_parameter_file(const std::string& path,
                                    const Options& opt,
                                    const Model& model,
                                    const kpolaris::PassAParams<Real>& pass_a,
                                    const std::string& grid_units,
                                    Real x_half_extent,
                                    Real y_half_extent,
                                    Real dx,
                                    Real dy,
                                    const FluxScale& flux_scale,
                                    const std::vector<Real>& frequencies) {
    if (path.empty()) {
        return;
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open effective parameter output: " + path);
    }
    out.precision(17);
    out << "# KPolaris effective parameter file\n";
    out << "# Generated by kpolaris_model_image after applying parameter files, command-line overrides, and derived defaults.\n";
    out << "# code_version=" << kpolaris::build_info::version << "\n";
    out << "# code_revision=" << kpolaris::build_info::source_revision << "\n";
    out << "# code_source_dirty=" << kpolaris::build_info::source_dirty << "\n";
    out << "# code_source_fingerprint=" << kpolaris::build_info::source_fingerprint << "\n";
    out << "# compiler_id=" << kpolaris::build_info::compiler_id << "\n";
    out << "# compiler_version=" << kpolaris::build_info::compiler_version << "\n";
    out << "# build_type=" << kpolaris::build_info::build_type << "\n";
    if (!opt.parameter_file.empty()) {
        out << "# source_parameter_file=" << opt.parameter_file << "\n";
    }
    out << "parameter_output=" << path << "\n";
    out << "model=" << opt.model << "\n";
    if (opt.model == "binary_riaf") {
        out << "binary_trajectory_model="
            << opt.binary_trajectory_model << "\n";
        if (!opt.binary_trajectory_file.empty()) {
            out << "binary_trajectory_file="
                << opt.binary_trajectory_file << "\n";
        }
        out << "binary_trajectory_format="
            << opt.binary_trajectory_format << "\n";
        out << "binary_trajectory_interpolation="
            << opt.binary_trajectory_interpolation << "\n";
        out << "binary_trajectory_mass_scale="
            << opt.binary_trajectory_mass_scale << "\n";
        out << "# derived_binary_trajectory_schema="
            << opt.binary_trajectory_schema << "\n";
        out << "# derived_binary_trajectory_sha256="
            << opt.binary_trajectory_sha256 << "\n";
        out << "# derived_binary_trajectory_gauge="
            << opt.binary_trajectory_gauge << "\n";
        out << "# derived_binary_trajectory_generator="
            << opt.binary_trajectory_generator << "\n";
        out << "# derived_binary_trajectory_generator_version="
            << opt.binary_trajectory_generator_version << "\n";
        out << "# derived_binary_trajectory_merger_reach_contract="
            << opt.binary_trajectory_merger_reach_contract << "\n";
        out << "# derived_binary_trajectory_declared_model="
            << opt.binary_trajectory_declared_model << "\n";
        out << "# derived_binary_trajectory_pn_terms="
            << opt.binary_trajectory_pn_terms << "\n";
        out << "# derived_binary_trajectory_source_verified="
            << opt.binary_trajectory_source_verified << "\n";
        out << "# derived_binary_trajectory_source_doi="
            << opt.binary_trajectory_source_doi << "\n";
        out << "# derived_binary_trajectory_pn_4pn_scope="
            << opt.binary_trajectory_pn_4pn_scope << "\n";
        out << "# derived_binary_trajectory_source_verification="
            << opt.binary_trajectory_source_verification << "\n";
        out << "# derived_binary_trajectory_upstream_cbwaves_sha256="
            << opt.binary_trajectory_upstream_cbwaves_sha256 << "\n";
        out << "# derived_binary_trajectory_patched_cbwaves_sha256="
            << opt.binary_trajectory_patched_cbwaves_sha256 << "\n";
        out << "# derived_binary_trajectory_merger_separation_reached="
            << opt.binary_trajectory_merger_separation_reached << "\n";
        out << "# derived_binary_trajectory_status="
            << opt.binary_trajectory_status << "\n";
        out << "# derived_binary_trajectory_boost_velocity_model="
            << opt.binary_trajectory_boost_velocity_model << "\n";
        out << "# derived_binary_trajectory_worldline_velocity_consistent="
            << opt.binary_trajectory_worldline_velocity_consistent << "\n";
        out << "# derived_binary_trajectory_future_extension="
            << opt.binary_trajectory_future_extension << "\n";
        out << "# derived_binary_trajectory_samples="
            << opt.binary_trajectory_samples << "\n";
        out << "# derived_binary_trajectory_t_min="
            << opt.binary_trajectory_t_min << "\n";
        out << "# derived_binary_trajectory_t_max="
            << opt.binary_trajectory_t_max << "\n";
        out << "# derived_binary_transition_start="
            << opt.binary_transition_start << "\n";
        out << "# derived_binary_transition_end="
            << opt.binary_transition_end << "\n";
        out << "# derived_binary_trajectory_exact_remnant="
            << opt.binary_trajectory_exact_remnant << "\n";
    }
    if (!opt.iharm_dump.empty()) out << "iharm_dump=" << opt.iharm_dump << "\n";
    if (!opt.kharma_dump.empty()) out << "kharma_dump=" << opt.kharma_dump << "\n";
    if (opt.model == "kharma") opt.kharma_ddc.write_parameters(out);
    if (!opt.athenak_dump.empty()) out << "athenak_dump=" << opt.athenak_dump << "\n";
    if (!opt.bhac_dump.empty()) out << "bhac_dump=" << opt.bhac_dump << "\n";
    if (!opt.hamr_dump.empty()) out << "hamr_dump=" << opt.hamr_dump << "\n";
    out << "output=" << opt.output << "\n";
    out << "format=" << opt.output_format << "\n";
    out << "camera=" << opt.camera << "\n";
    out << "coordinate=" << opt.coordinate << "\n";
    out << "# derived_metric=" << metric_name_for_options(opt) << "\n";
    out << "nx=" << opt.nx << "\n";
    out << "ny=" << opt.ny << "\n";
    out << "radius=" << opt.radius << "\n";
    out << "inclination_rad=" << opt.inclination << "\n";
    out << "fov=" << pass_a.camera.fov << "\n";
    out << "fovy=" << pass_a.camera.fov_y << "\n";
    if (pass_a.camera.model == kpolaris::CameraModel::ParallelPlane) {
        out << "xspan=" << pass_a.camera.xspan << "\n";
        out << "yspan=" << pass_a.camera.yspan << "\n";
    }
    if (opt.dsource_pc > Real(0)) {
        out << "dsource=" << opt.dsource_pc << "\n";
    }
    if (opt.effective_fovx_dsource > Real(0)) {
        out << "fovx_dsource=" << opt.effective_fovx_dsource << "\n";
    }
    if (opt.effective_fovy_dsource > Real(0)) {
        out << "fovy_dsource=" << opt.effective_fovy_dsource << "\n";
    }
    out << "# derived_image_width_x_M=" << opt.image_width_x << "\n";
    out << "# derived_image_width_y_M=" << opt.image_width_y << "\n";
    out << "x_offset=" << opt.x_offset << "\n";
    out << "y_offset=" << opt.y_offset << "\n";
    out << "use_pinhole_pixel_bias=" << opt.use_pinhole_pixel_bias << "\n";
    out << "pinhole_pixel_bias=" << opt.pinhole_pixel_bias << "\n";
    out << "# effective_x_offset=" << pass_a.camera.x_offset << "\n";
    out << "# effective_y_offset=" << pass_a.camera.y_offset << "\n";
    out << "# derived_grid_units=" << grid_units << "\n";
    out << "# derived_x_half_extent=" << x_half_extent << "\n";
    out << "# derived_y_half_extent=" << y_half_extent << "\n";
    out << "# derived_dx=" << dx << "\n";
    out << "# derived_dy=" << dy << "\n";
    out << "# derived_l_unit_cm=" << flux_scale.l_unit_cm << "\n";
    out << "# derived_dsource_cm=" << flux_scale.dsource_cm << "\n";
    out << "# derived_pixel_solid_angle_sr=" << flux_scale.pixel_solid_angle_sr << "\n";
    out << "# derived_intensity_to_flux_jy_per_pixel="
        << flux_scale.intensity_to_flux_jy_per_pixel << "\n";
    if (opt.model != "binary_riaf") {
        out << "spin=" << opt.spin << "\n";
    }
    if (opt.model != "binary_riaf") {
        out << "inner_radius=" << opt.inner_radius << "\n";
    }
    out << "outer_radius=" << opt.outer_radius << "\n";
    out << "step=" << opt.step << "\n";
    out << "adaptive=" << opt.adaptive << "\n";
    out << "adaptive_tolerance=" << opt.adaptive_tolerance << "\n";
    out << "min_step=" << opt.min_step << "\n";
    out << "max_step=" << opt.max_step << "\n";
    out << "max_radiation_step=" << opt.max_radiation_step << "\n";
    out << "max_radiation_depth=" << opt.max_radiation_depth << "\n";
    out << "max_absorption_depth=" << opt.max_absorption_depth << "\n";
    out << "max_faraday_depth=" << opt.max_faraday_depth << "\n";
    out << "closure_x_warning=" << opt.closure_x_warning << "\n";
    out << "closure_k_warning=" << opt.closure_k_warning << "\n";
    out << "frame_error_warning=" << opt.frame_error_warning << "\n";
    out << "basis_identity_warning=" << opt.basis_identity_warning << "\n";
    out << "split_transport=" << opt.split_transport << "\n";
    out << "analysis_mode=" << opt.analysis_mode << "\n";
    out << "direct_only=" << opt.direct_only << "\n";
    out << "equatorial_h_over_r=" << opt.equatorial_h_over_r << "\n";
    out << "equatorial_samples=" << opt.equatorial_samples << "\n";
    out << "faraday_rotation=" << opt.faraday_rotation << "\n";
    out << "analysis_response=" << opt.analysis_response << "\n";
    out << "analysis_response_step=" << opt.analysis_response_step << "\n";
    out << "analysis_partition=" << opt.analysis_partition << "\n";
    out << "analysis_partition_edges=" << opt.analysis_partition_edges << "\n";
    out << "analysis_funnel_angle_deg=" << opt.analysis_funnel_angle_deg << "\n";
    out << "analysis_disk_angle_deg=" << opt.analysis_disk_angle_deg << "\n";
    out << "analysis_sigma_boundary=" << opt.analysis_sigma_boundary << "\n";
    out << "analysis_beta_boundary=" << opt.analysis_beta_boundary << "\n";

    out << "analysis_radial_bins=" << opt.analysis_radial_bins << "\n";
    out << "analysis_radial_min=" << opt.analysis_radial_min << "\n";
    out << "analysis_radial_max=" << opt.analysis_radial_max << "\n";
    out << "analysis_formation_fraction=" << opt.analysis_formation_fraction << "\n";
    out << "slow_light=" << opt.slow_light << "\n";
    out << "slow_light_prefetch=" << opt.slow_light_prefetch << "\n";
    out << "slow_light_step_mode=" << kpolaris::slow_light_step_mode_name(opt.slow_light_step_mode) << "\n";
    out << "slow_light_interpolation=" << kpolaris::slow_light_interpolation_name(opt.slow_light_interpolation) << "\n";
    out << "slow_light_pipeline=" << opt.slow_light_pipeline << "\n";
    out << "slow_light_windows_per_block=" << opt.slow_light_windows_per_block << "\n";
    out << "slow_light_prefetch_snapshots=" << opt.slow_light_prefetch_snapshots << "\n";
    out << "slow_light_snapshot_cache_gib=" << opt.slow_light_snapshot_cache_gib << "\n";
    out << "slow_light_time_probe=" << opt.slow_light_time_probe << "\n";
    out << "slow_light_observation_time=" << opt.slow_light_observation_time << "\n";
    if (!opt.slow_light_dump_list.empty()) out << "slow_light_dump_list=" << opt.slow_light_dump_list << "\n";
    if (!opt.slow_light_time_list.empty()) out << "slow_light_time_list=" << opt.slow_light_time_list << "\n";
    if (!opt.slow_light_dump_pattern.empty()) {
        out << "slow_light_dump_pattern=" << opt.slow_light_dump_pattern << "\n";
        out << "slow_light_dump_start=" << opt.slow_light_dump_start << "\n";
        out << "slow_light_dump_end=" << opt.slow_light_dump_end << "\n";
        out << "slow_light_dump_stride=" << opt.slow_light_dump_stride << "\n";
    }
    out << "max_steps=" << opt.max_steps << "\n";
    out << "substeps=" << opt.radiation_substeps << "\n";
    out << "freq=" << opt.freq << "\n";
    if (!opt.freq_list.empty()) {
        out << "freq_list=" << opt.freq_list << "\n";
    }
    if (opt.nfreq > 0 || opt.freq_min > Real(0) || opt.freq_max > Real(0)) {
        out << "freq_min=" << opt.freq_min << "\n";
        out << "freq_max=" << opt.freq_max << "\n";
        out << "nfreq=" << opt.nfreq << "\n";
        out << "freq_spacing=" << opt.freq_spacing << "\n";
    }
    out << "multifrequency_chunk_size=" << opt.multifrequency_chunk_size << "\n";
    out << "# derived_nfreq=" << frequencies.size() << "\n";
    out << "# effective_multifrequency_chunk_size=" << effective_multifrequency_chunk_size(opt, frequencies.size()) << "\n";
    out << "# derived_freq_list=" << frequency_list_string(frequencies) << "\n";
    out << "# compile_KPOLARIS_MAX_FREQUENCIES=" << KPOLARIS_MAX_FREQUENCIES << "\n";
    out << "iharm_resample_spherical_ks_precomputed=" << opt.iharm_resample_spherical_ks_precomputed << "\n";
    out << "iharm_resample_spherical_ks_primitives=" << opt.iharm_resample_spherical_ks_primitives << "\n";
    out << "iharm_resample_n1=" << opt.iharm_resample_n1 << "\n";
    out << "iharm_resample_n2=" << opt.iharm_resample_n2 << "\n";
    out << "iharm_resample_n3=" << opt.iharm_resample_n3 << "\n";
    out << "iharm_resample_r_in=" << opt.iharm_resample_r_in << "\n";
    out << "iharm_resample_r_out=" << opt.iharm_resample_r_out << "\n";
    out << "kharma_M_unit=" << opt.kharma_M_unit << "\n";
    out << "kharma_mbh_solar=" << opt.kharma_mbh_solar << "\n";
    out << "kharma_trat_small=" << opt.kharma_trat_small << "\n";
    out << "kharma_trat_large=" << opt.kharma_trat_large << "\n";
    out << "kharma_beta_crit=" << opt.kharma_beta_crit << "\n";
    out << "kharma_sigma_cut=" << opt.kharma_sigma_cut << "\n";
    out << "kharma_sigma_cut_high=" << opt.kharma_sigma_cut_high << "\n";
    out << "kharma_interpolate_derived_scalars=" << opt.kharma_interpolate_derived_scalars << "\n";
    out << "kharma_resample_spherical_ks_precomputed=" << opt.kharma_resample_spherical_ks_precomputed << "\n";
    out << "kharma_resample_spherical_ks_primitives=" << opt.kharma_resample_spherical_ks_primitives << "\n";
    out << "kharma_resample_n1=" << opt.kharma_resample_n1 << "\n";
    out << "kharma_resample_n2=" << opt.kharma_resample_n2 << "\n";
    out << "kharma_resample_n3=" << opt.kharma_resample_n3 << "\n";
    out << "kharma_resample_r_in=" << opt.kharma_resample_r_in << "\n";
    out << "kharma_resample_r_out=" << opt.kharma_resample_r_out << "\n";
    out << "kharma_reverse_field=" << opt.kharma_reverse_field << "\n";
    out << "athenak_M_unit=" << opt.athenak_M_unit << "\n";
    out << "athenak_mbh_solar=" << opt.athenak_mbh_solar << "\n";
    out << "athenak_trat_small=" << opt.athenak_trat_small << "\n";
    out << "athenak_trat_large=" << opt.athenak_trat_large << "\n";
    out << "athenak_beta_crit=" << opt.athenak_beta_crit << "\n";
    out << "athenak_gamma=" << opt.athenak_gamma << "\n";
    out << "athenak_sigma_cut=" << opt.athenak_sigma_cut << "\n";
    out << "athenak_sigma_cut_high=" << opt.athenak_sigma_cut_high << "\n";
    out << "athenak_r_in=" << opt.athenak_r_in << "\n";
    out << "athenak_r_out=" << opt.athenak_r_out << "\n";
    out << "athenak_profile_mode=" << opt.athenak_profile_mode << "\n";
    out << "bhac_M_unit=" << opt.bhac_M_unit << "\n";
    out << "bhac_mbh_solar=" << opt.bhac_mbh_solar << "\n";
    out << "bhac_trat_small=" << opt.bhac_trat_small << "\n";
    out << "bhac_trat_large=" << opt.bhac_trat_large << "\n";
    out << "bhac_beta_crit=" << opt.bhac_beta_crit << "\n";
    out << "bhac_gamma=" << opt.bhac_gamma << "\n";
    out << "bhac_sigma_cut=" << opt.bhac_sigma_cut << "\n";
    out << "bhac_sigma_cut_high=" << opt.bhac_sigma_cut_high << "\n";
    out << "bhac_r_in=" << opt.bhac_r_in << "\n";
    out << "bhac_r_out=" << opt.bhac_r_out << "\n";
    out << "bhac_hslope=" << opt.bhac_hslope << "\n";
    out << "bhac_nxlone1=" << opt.bhac_nxlone1 << "\n";
    out << "bhac_nxlone2=" << opt.bhac_nxlone2 << "\n";
    out << "bhac_nxlone3=" << opt.bhac_nxlone3 << "\n";
    out << "bhac_spin_index=" << opt.bhac_spin_index << "\n";
    out << "bhac_x1_min=" << opt.bhac_x1_min << "\n";
    out << "bhac_x1_max=" << opt.bhac_x1_max << "\n";
    out << "bhac_x2_min=" << opt.bhac_x2_min << "\n";
    out << "bhac_x2_max=" << opt.bhac_x2_max << "\n";
    out << "bhac_x3_min=" << opt.bhac_x3_min << "\n";
    out << "bhac_x3_max=" << opt.bhac_x3_max << "\n";
    out << "bhac_sfc=" << opt.bhac_sfc << "\n";
    out << "bhac_reverse_field=" << opt.bhac_reverse_field << "\n";
    out << "bhac_profile_mode=" << opt.bhac_profile_mode << "\n";
    if (!opt.bhac_cache.empty()) out << "bhac_cache=" << opt.bhac_cache << "\n";
    out << "bhac_cache_mode=" << bhac_cache_mode_name(opt.bhac_cache_mode) << "\n";
    out << "hamr_M_unit=" << opt.hamr_M_unit << "\n";
    out << "hamr_mbh_solar=" << opt.hamr_mbh_solar << "\n";
    out << "hamr_trat_small=" << opt.hamr_trat_small << "\n";
    out << "hamr_trat_large=" << opt.hamr_trat_large << "\n";
    out << "hamr_beta_crit=" << opt.hamr_beta_crit << "\n";
    out << "hamr_gamma=" << opt.hamr_gamma << "\n";
    out << "hamr_sigma_cut=" << opt.hamr_sigma_cut << "\n";
    out << "hamr_sigma_cut_high=" << opt.hamr_sigma_cut_high << "\n";
    out << "hamr_r_in=" << opt.hamr_r_in << "\n";
    out << "hamr_r_out=" << opt.hamr_r_out << "\n";
    out << "hamr_hslope=" << opt.hamr_hslope << "\n";
    out << "# hamr_hslope_source=" << hamr_hslope_source(opt) << "\n";
    out << "hamr_reverse_field=" << opt.hamr_reverse_field << "\n";
    out << "hamr_profile_mode=" << opt.hamr_profile_mode << "\n";
    out << "hamr_id_order=" << opt.hamr_id_order << "\n";
    out << "hamr_root_order=" << opt.hamr_root_order << "\n";
    out << "timing=" << opt.timing << "\n";
    out << "# stokes_convention=camera_frame\n";
    out << "# polarization_basis=camera_screen\n";
    out << "evpa_0=" << opt.evpa_0 << "\n";
    write_effective_model_parameters(out, model, opt.model);
}


#if KPOLARIS_ENABLE_MODEL_IHARM || KPOLARIS_ENABLE_MODEL_KHARMA
void write_hdf5_model_metadata(H5::H5Object& obj,
                               const kpolaris::GRMHDRadiationModel<Real>& model,
                               const std::string& model_name) {
    const std::string prefix = grmhd_metadata_prefix(model_name);
    write_h5_int(obj, "grmhd_n1", model.n1);
    write_h5_int(obj, "grmhd_n2", model.n2);
    write_h5_int(obj, "grmhd_n3", model.n3);
    write_h5_scalar(obj, "grmhd_M_unit", model.M_unit);
    write_h5_scalar(obj, "grmhd_mbh_solar", model.mbh_solar);
    write_h5_scalar(obj, "grmhd_trat_small", model.trat_small);
    write_h5_scalar(obj, "grmhd_trat_large", model.trat_large);
    write_h5_scalar(obj, "grmhd_beta_crit", model.beta_crit);
    write_h5_scalar(obj, "grmhd_sigma_cut", model.sigma_cut);
    write_h5_scalar(obj, "grmhd_sigma_cut_high", model.sigma_cut_high);
    write_h5_int(obj, prefix + "_n1", model.n1);
    write_h5_int(obj, prefix + "_n2", model.n2);
    write_h5_int(obj, prefix + "_n3", model.n3);
    write_h5_scalar(obj, prefix + "_M_unit", model.M_unit);
    write_h5_scalar(obj, prefix + "_mbh_solar", model.mbh_solar);
    write_h5_scalar(obj, prefix + "_trat_small", model.trat_small);
    write_h5_scalar(obj, prefix + "_trat_large", model.trat_large);
    write_h5_scalar(obj, prefix + "_beta_crit", model.beta_crit);
    write_h5_scalar(obj, prefix + "_sigma_cut", model.sigma_cut);
    write_h5_scalar(obj, prefix + "_sigma_cut_high", model.sigma_cut_high);
    write_h5_int(obj, "emission_type", model.emission_type);
    write_h5_string(obj, "emission_fit", emission_fit_name(model.emission_type));
    write_h5_scalar(obj, "nonthermal_kappa", model.nonthermal_kappa);
    write_h5_int(obj, "variable_kappa", model.variable_kappa);
    write_h5_scalar(obj, "variable_kappa_min", model.variable_kappa_min);
    write_h5_scalar(obj, "variable_kappa_interp_start", model.variable_kappa_interp_start);
    write_h5_scalar(obj, "variable_kappa_max", model.variable_kappa_max);
    write_h5_scalar(obj, "powerlaw_p", model.powerlaw_p);
    write_h5_scalar(obj, "powerlaw_eta", model.powerlaw_eta);
    write_h5_scalar(obj, "powerlaw_gamma_min", model.powerlaw_gamma_min);
    write_h5_scalar(obj, "powerlaw_gamma_max", model.powerlaw_gamma_max);
    write_h5_scalar(obj, "powerlaw_gamma_cutoff", model.powerlaw_gamma_cutoff);
    write_h5_int(obj, "grmhd_interpolate_derived_scalars", model.has_derived_scalars);
    write_h5_int(obj, "grmhd_precomputed_fluid_state", model.precomputed_fluid_state);
    write_h5_int(obj, "grmhd_data_coordinate_system", model.data_coordinate_system);
    write_h5_int(obj, "grmhd_radial_coordinate_log", model.radial_coordinate_log);
    write_h5_int(obj, prefix + "_interpolate_derived_scalars", model.has_derived_scalars);
    write_h5_int(obj, prefix + "_precomputed_fluid_state", model.precomputed_fluid_state);
    write_h5_int(obj, prefix + "_data_coordinate_system", model.data_coordinate_system);
    write_h5_int(obj, prefix + "_radial_coordinate_log", model.radial_coordinate_log);
    write_h5_scalar(obj, "dlambda_scale", model.dlambda_scale());
    write_h5_scalar(obj, prefix + "_dlambda_scale", model.dlambda_scale());
}
#endif

template<class Model>
void write_hdf5_model_metadata(H5::H5Object& obj,
                               const Model& model,
                               const std::string&) {
    write_hdf5_model_metadata(obj, model);
}


#if KPOLARIS_ENABLE_MODEL_RIAF
FrequencyImage compute_frequency_image(const Options& opt,
                                       const kpolaris::PassAParams<Real>& pass_a,
                                       kpolaris::RIAFAnalyticRadiationModel<Real> model,
                                       Real frequency,
                                       int freq_index,
                                       bool hdf5_output,
                                       bool multi_frequency,
                                       std::ofstream& out) {
    set_model_frequency(model, frequency);
    ImageHostData data;
    if (opt.analysis_mode) {
#if KPOLARIS_ENABLE_ANALYSIS_MODE
        data = run_riaf_analysis_image(
            pass_a, make_analysis_config(opt, pass_a), model,
            opt.radiation_substeps, opt.timing, opt.split_transport);
#else
        throw std::runtime_error("analysis_mode requested but KPOLARIS_ENABLE_ANALYSIS_MODE is OFF");
#endif
    } else {
        data = run_riaf_image(
            pass_a, model, opt.radiation_substeps, opt.timing, opt.split_transport);
    }
    return pack_transport_frequency_image(opt, data, frequency, freq_index, 0,
                                          hdf5_output, multi_frequency, out);
}
#endif

#if KPOLARIS_ENABLE_MODEL_RIAF && KPOLARIS_MAX_FREQUENCIES > 1
std::vector<FrequencyImage> compute_multifrequency_images(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::RIAFAnalyticRadiationModel<Real> model,
    const std::vector<Real>& frequencies,
    bool hdf5_output,
    std::ofstream& out) {
    const int chunk_size = effective_multifrequency_chunk_size(opt, frequencies.size());
    std::vector<FrequencyImage> images(static_cast<size_t>(frequencies.size()));
    for (int start = 0; start < static_cast<int>(frequencies.size()); start += chunk_size) {
        const int count = std::min(chunk_size, static_cast<int>(frequencies.size()) - start);
        std::vector<Real> chunk(frequencies.begin() + start, frequencies.begin() + start + count);
        const ImageHostData data = run_riaf_multifrequency_image(
            pass_a, model, opt.radiation_substeps, chunk, frequencies, opt.timing, opt.split_transport);
        for (int local = 0; local < count; ++local) {
            const int global = start + local;
            images[static_cast<size_t>(global)] = pack_transport_frequency_image(
                opt, data, frequencies[static_cast<size_t>(global)], global, local,
                hdf5_output, true, out);
        }
    }
    return images;
}
#endif

#if KPOLARIS_ENABLE_MODEL_BINARY_RIAF
FrequencyImage compute_frequency_image(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::BinaryRIAFRadiationModel<Real> model,
    Real frequency, int freq_index, bool hdf5_output,
    bool multi_frequency, std::ofstream& out) {
    set_model_frequency(model, frequency);
    const ImageHostData data = run_binary_riaf_image(
        pass_a, model, opt.radiation_substeps, opt.timing,
        opt.split_transport);
    return pack_transport_frequency_image(
        opt, data, frequency, freq_index, 0, hdf5_output,
        multi_frequency, out);
}
#endif

#if KPOLARIS_ENABLE_MODEL_BINARY_RIAF && KPOLARIS_MAX_FREQUENCIES > 1
std::vector<FrequencyImage> compute_multifrequency_images(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::BinaryRIAFRadiationModel<Real> model,
    const std::vector<Real>& frequencies,
    bool hdf5_output, std::ofstream& out) {
    const int chunk_size = effective_multifrequency_chunk_size(
        opt, frequencies.size());
    std::vector<FrequencyImage> images(frequencies.size());
    for (int start = 0; start < static_cast<int>(frequencies.size());
         start += chunk_size) {
        const int count = std::min(
            chunk_size, static_cast<int>(frequencies.size()) - start);
        std::vector<Real> chunk(frequencies.begin() + start,
                                frequencies.begin() + start + count);
        const ImageHostData data = run_binary_riaf_multifrequency_image(
            pass_a, model, opt.radiation_substeps, chunk, frequencies,
            opt.timing, opt.split_transport);
        for (int local = 0; local < count; ++local) {
            const int global = start + local;
            images[static_cast<size_t>(global)] =
                pack_transport_frequency_image(
                    opt, data, frequencies[static_cast<size_t>(global)],
                    global, local, hdf5_output, true, out);
        }
    }
    return images;
}
#endif

#if KPOLARIS_ENABLE_MODEL_TORUS
FrequencyImage compute_frequency_image(const Options& opt,
                                       const kpolaris::PassAParams<Real>& pass_a,
                                       kpolaris::MagnetizedTorusRadiationModel<Real> model,
                                       Real frequency,
                                       int freq_index,
                                       bool hdf5_output,
                                       bool multi_frequency,
                                       std::ofstream& out) {
    set_model_frequency(model, frequency);
    ImageHostData data;
    if (opt.analysis_mode) {
#if KPOLARIS_ENABLE_ANALYSIS_MODE
        data = run_torus_analysis_image(
            pass_a, make_analysis_config(opt, pass_a), model,
            opt.radiation_substeps, opt.timing, opt.split_transport);
#else
        throw std::runtime_error("analysis_mode requested but KPOLARIS_ENABLE_ANALYSIS_MODE is OFF");
#endif
    } else {
        data = run_torus_image(
            pass_a, model, opt.radiation_substeps, opt.timing, opt.split_transport);
    }
    return pack_transport_frequency_image(opt, data, frequency, freq_index, 0,
                                          hdf5_output, multi_frequency, out);
}
#endif


#if KPOLARIS_ENABLE_MODEL_TORUS && KPOLARIS_MAX_FREQUENCIES > 1
std::vector<FrequencyImage> compute_multifrequency_images(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::MagnetizedTorusRadiationModel<Real> model,
    const std::vector<Real>& frequencies,
    bool hdf5_output,
    std::ofstream& out) {
    const int chunk_size = effective_multifrequency_chunk_size(opt, frequencies.size());
    std::vector<FrequencyImage> images(static_cast<size_t>(frequencies.size()));
    for (int start = 0; start < static_cast<int>(frequencies.size()); start += chunk_size) {
        const int count = std::min(chunk_size, static_cast<int>(frequencies.size()) - start);
        std::vector<Real> chunk(frequencies.begin() + start, frequencies.begin() + start + count);
        const ImageHostData data = run_torus_multifrequency_image(
            pass_a, model, opt.radiation_substeps, chunk, frequencies, opt.timing, opt.split_transport);
        for (int local = 0; local < count; ++local) {
            const int global = start + local;
            images[static_cast<size_t>(global)] = pack_transport_frequency_image(
                opt, data, frequencies[static_cast<size_t>(global)], global, local,
                hdf5_output, true, out);
        }
    }
    return images;
}
#endif



#if KPOLARIS_ENABLE_MODEL_ATHENAK
FrequencyImage compute_frequency_image(const Options& opt,
                                       const kpolaris::PassAParams<Real>& pass_a,
                                       kpolaris::AthenaKDirectRadiationModel<Real> model,
                                       Real frequency,
                                       int freq_index,
                                       bool hdf5_output,
                                       bool multi_frequency,
                                       std::ofstream& out) {
    set_model_frequency(model, frequency);
    kpolaris::KerrSchildInMetric<Real> metric(pass_a.mass, pass_a.spin);
    ImageHostData data;
    if (opt.slow_light) {
#if KPOLARIS_ENABLE_SLOW_LIGHT
        if (opt.analysis_mode) {
#if !KPOLARIS_ENABLE_ANALYSIS_MODE
            throw std::runtime_error("analysis_mode requested but KPOLARIS_ENABLE_ANALYSIS_MODE is OFF");
#endif
        }
        data = run_athenak_slow_light_image(opt, pass_a, model, frequency);
#else
        throw std::runtime_error("slow_light requested but KPOLARIS_ENABLE_SLOW_LIGHT is OFF");
#endif
    } else if (opt.analysis_mode) {
#if KPOLARIS_ENABLE_ANALYSIS_MODE
        data = kpolaris_image_detail::run_analysis_image_metric_with_optional_split(
            pass_a, make_analysis_config(opt, pass_a), model,
            opt.radiation_substeps, opt.timing, opt.split_transport,
            metric, "athenak_direct_cartesian_ks_analysis");
#else
        throw std::runtime_error("analysis_mode requested but KPOLARIS_ENABLE_ANALYSIS_MODE is OFF");
#endif
    } else {
        data = kpolaris_image_detail::run_image_metric_with_optional_split(
            pass_a, model, opt.radiation_substeps, opt.timing, opt.split_transport,
            metric, "athenak_direct_cartesian_ks");
    }
    return pack_transport_frequency_image(opt, data, frequency, freq_index, 0,
                                          hdf5_output, multi_frequency, out);
}
#endif


#if KPOLARIS_ENABLE_MODEL_ATHENAK && KPOLARIS_MAX_FREQUENCIES > 1
std::vector<FrequencyImage> compute_multifrequency_images(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::AthenaKDirectRadiationModel<Real> model,
    const std::vector<Real>& frequencies,
    bool hdf5_output,
    std::ofstream& out) {
    const int chunk_size = effective_multifrequency_chunk_size(opt, frequencies.size());
    kpolaris::KerrSchildInMetric<Real> metric(pass_a.mass, pass_a.spin);
    std::vector<FrequencyImage> images(static_cast<size_t>(frequencies.size()));
    for (int start = 0; start < static_cast<int>(frequencies.size()); start += chunk_size) {
        const int count = std::min(chunk_size, static_cast<int>(frequencies.size()) - start);
        std::vector<Real> chunk(frequencies.begin() + start, frequencies.begin() + start + count);
        const ImageHostData data = kpolaris_image_detail::run_multifrequency_image_metric_with_step_control(
            pass_a, model, opt.radiation_substeps, chunk, frequencies, opt.timing,
            metric, "athenak_direct_cartesian_ks_multifrequency");
        for (int local = 0; local < count; ++local) {
            const int global = start + local;
            images[static_cast<size_t>(global)] = pack_transport_frequency_image(
                opt, data, frequencies[static_cast<size_t>(global)], global, local,
                hdf5_output, true, out);
        }
    }
    return images;
}
#endif


#if KPOLARIS_ENABLE_MODEL_BHAC || KPOLARIS_ENABLE_MODEL_HAMR
template<class Metric>
ImageHostData run_native_amr_fast_light_with_metric(const Options& opt,
                                                    const kpolaris::PassAParams<Real>& pass_a,
                                                    kpolaris::BHACAMRRadiationModel<Real> model,
                                                    const Metric& metric,
                                                    const char* label) {
    if (opt.analysis_mode) {
#if KPOLARIS_ENABLE_ANALYSIS_MODE
        return kpolaris_image_detail::run_analysis_image_metric_with_optional_split(
            pass_a, make_analysis_config(opt, pass_a), model,
            opt.radiation_substeps, opt.timing, opt.split_transport,
            metric, label);
#else
        throw std::runtime_error("analysis_mode requested but KPOLARIS_ENABLE_ANALYSIS_MODE is OFF");
#endif
    }
    return kpolaris_image_detail::run_image_metric_with_optional_split(
        pass_a, model, opt.radiation_substeps, opt.timing, opt.split_transport,
        metric, label);
}

FrequencyImage compute_frequency_image(const Options& opt,
                                       const kpolaris::PassAParams<Real>& pass_a,
                                       kpolaris::BHACAMRRadiationModel<Real> model,
                                       Real frequency,
                                       int freq_index,
                                       bool hdf5_output,
                                       bool multi_frequency,
                                       std::ofstream& out) {
    set_model_frequency(model, frequency);
    if (opt.slow_light) {
#if KPOLARIS_ENABLE_SLOW_LIGHT
        if (opt.analysis_mode) {
#if !KPOLARIS_ENABLE_ANALYSIS_MODE
            throw std::runtime_error("analysis_mode requested but KPOLARIS_ENABLE_ANALYSIS_MODE is OFF");
#endif
        }
        ImageHostData data;
        if (opt.model == "hamr") {
#if KPOLARIS_ENABLE_MODEL_HAMR
            data = run_hamr_slow_light_image(opt, pass_a, model, frequency);
#else
            throw std::runtime_error("slow_light requested for hamr but hamr support is not built");
#endif
        } else {
#if KPOLARIS_ENABLE_MODEL_BHAC
            data = run_bhac_slow_light_image(opt, pass_a, model, frequency);
#else
            throw std::runtime_error("slow_light requested for bhac but bhac support is not built");
#endif
        }
        return pack_transport_frequency_image(opt, data, frequency, freq_index, 0,
                                              hdf5_output, multi_frequency, out);
#else
        throw std::runtime_error("slow_light requested but KPOLARIS_ENABLE_SLOW_LIGHT is OFF");
#endif
    }
    ImageHostData data;
    if (opt.coordinate == "spherical_ks") {
        kpolaris::KerrSchildSphericalMetric<Real> metric(pass_a.mass, pass_a.spin);
        data = run_native_amr_fast_light_with_metric(opt, pass_a, model, metric,
            opt.model == "hamr" ? "hamr_direct_spherical_ks" : "bhac_direct_spherical_ks");
    } else if (opt.coordinate == "cartesian_ks") {
        kpolaris::KerrSchildInMetric<Real> metric(pass_a.mass, pass_a.spin);
        data = run_native_amr_fast_light_with_metric(opt, pass_a, model, metric,
            opt.model == "hamr" ? "hamr_direct_cartesian_ks" : "bhac_direct_cartesian_ks");
    } else {
        throw std::runtime_error("bhac/hamr currently support --coordinate=cartesian_ks or spherical_ks");
    }
    return pack_transport_frequency_image(opt, data, frequency, freq_index, 0,
                                          hdf5_output, multi_frequency, out);
}

#if KPOLARIS_MAX_FREQUENCIES > 1
std::vector<FrequencyImage> compute_multifrequency_images(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::BHACAMRRadiationModel<Real> model,
    const std::vector<Real>& frequencies,
    bool hdf5_output,
    std::ofstream& out) {
    if (opt.slow_light) {
#if KPOLARIS_ENABLE_SLOW_LIGHT
        std::vector<ImageHostData> data;
        if (opt.model == "hamr") {
#if KPOLARIS_ENABLE_MODEL_HAMR
            data = run_hamr_slow_light_multifrequency_image_data(opt, pass_a, model, frequencies);
#else
            throw std::runtime_error("slow_light requested for hamr but hamr support is not built");
#endif
        } else {
#if KPOLARIS_ENABLE_MODEL_BHAC
            data = run_bhac_slow_light_multifrequency_image_data(opt, pass_a, model, frequencies);
#else
            throw std::runtime_error("slow_light requested for bhac but bhac support is not built");
#endif
        }
        if (data.size() != frequencies.size()) {
            throw std::runtime_error("slow-light multi-frequency returned wrong image count");
        }
        std::vector<FrequencyImage> images(static_cast<size_t>(frequencies.size()));
        for (int f = 0; f < static_cast<int>(frequencies.size()); ++f) {
            images[static_cast<size_t>(f)] = pack_transport_frequency_image(
                opt, data[static_cast<size_t>(f)], frequencies[static_cast<size_t>(f)],
                f, 0, hdf5_output, true, out);
        }
        return images;
#else
        throw std::runtime_error("slow_light requested but KPOLARIS_ENABLE_SLOW_LIGHT is OFF");
#endif
    }
    const int chunk_size = effective_multifrequency_chunk_size(opt, frequencies.size());
    std::vector<FrequencyImage> images(static_cast<size_t>(frequencies.size()));
    for (int start = 0; start < static_cast<int>(frequencies.size()); start += chunk_size) {
        const int count = std::min(chunk_size, static_cast<int>(frequencies.size()) - start);
        std::vector<Real> chunk(frequencies.begin() + start, frequencies.begin() + start + count);
        ImageHostData data;
        if (opt.coordinate == "spherical_ks") {
            kpolaris::KerrSchildSphericalMetric<Real> metric(pass_a.mass, pass_a.spin);
            data = kpolaris_image_detail::run_multifrequency_image_metric_with_step_control(
                pass_a, model, opt.radiation_substeps, chunk, frequencies, opt.timing,
                metric, opt.model == "hamr" ? "hamr_direct_spherical_ks_multifrequency" :
                                              "bhac_direct_spherical_ks_multifrequency");
        } else if (opt.coordinate == "cartesian_ks") {
            kpolaris::KerrSchildInMetric<Real> metric(pass_a.mass, pass_a.spin);
            data = kpolaris_image_detail::run_multifrequency_image_metric_with_step_control(
                pass_a, model, opt.radiation_substeps, chunk, frequencies, opt.timing,
                metric, opt.model == "hamr" ? "hamr_direct_cartesian_ks_multifrequency" :
                                              "bhac_direct_cartesian_ks_multifrequency");
        } else {
            throw std::runtime_error("bhac/hamr currently support --coordinate=cartesian_ks or spherical_ks");
        }
        for (int local = 0; local < count; ++local) {
            const int global = start + local;
            images[static_cast<size_t>(global)] = pack_transport_frequency_image(
                opt, data, frequencies[static_cast<size_t>(global)], global, local,
                hdf5_output, true, out);
        }
    }
    return images;
}
#endif
#endif


#if (KPOLARIS_ENABLE_MODEL_IHARM || KPOLARIS_ENABLE_MODEL_KHARMA) && KPOLARIS_MAX_FREQUENCIES > 1
std::vector<FrequencyImage> compute_multifrequency_images(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::GRMHDRadiationModel<Real> model,
    const std::vector<Real>& frequencies,
    bool hdf5_output,
    std::ofstream& out) {
    const int chunk_size = effective_multifrequency_chunk_size(opt, frequencies.size());
    std::vector<FrequencyImage> images(static_cast<size_t>(frequencies.size()));
    for (int start = 0; start < static_cast<int>(frequencies.size()); start += chunk_size) {
        const int count = std::min(chunk_size, static_cast<int>(frequencies.size()) - start);
        std::vector<Real> chunk(frequencies.begin() + start, frequencies.begin() + start + count);
        const ImageHostData data = run_grmhd_multifrequency_image(
            pass_a, model, opt.radiation_substeps, chunk, frequencies, opt.timing);
        for (int local = 0; local < count; ++local) {
            const int global = start + local;
            images[static_cast<size_t>(global)] = pack_transport_frequency_image(
                opt, data, frequencies[static_cast<size_t>(global)], global, local,
                hdf5_output, true, out);
        }
    }
    return images;
}
#endif



#if KPOLARIS_ENABLE_SLOW_LIGHT && (KPOLARIS_ENABLE_MODEL_BHAC || KPOLARIS_ENABLE_MODEL_HAMR)
std::vector<FrequencyImage> compute_slow_light_multifrequency_images(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::BHACAMRRadiationModel<Real> model,
    const std::vector<Real>& frequencies,
    bool hdf5_output,
    std::ofstream& out) {
    std::vector<ImageHostData> data;
    if (opt.model == "hamr") {
#if KPOLARIS_ENABLE_MODEL_HAMR
        data = run_hamr_slow_light_multifrequency_image_data(opt, pass_a, model, frequencies);
#else
        throw std::runtime_error("slow_light requested for hamr but hamr support is not built");
#endif
    } else {
#if KPOLARIS_ENABLE_MODEL_BHAC
        data = run_bhac_slow_light_multifrequency_image_data(opt, pass_a, model, frequencies);
#else
        throw std::runtime_error("slow_light requested for bhac but bhac support is not built");
#endif
    }
    if (data.size() != frequencies.size()) {
        throw std::runtime_error("slow-light multi-frequency returned wrong image count");
    }
    std::vector<FrequencyImage> images(static_cast<size_t>(frequencies.size()));
    for (int f = 0; f < static_cast<int>(frequencies.size()); ++f) {
        images[static_cast<size_t>(f)] = pack_transport_frequency_image(
            opt, data[static_cast<size_t>(f)], frequencies[static_cast<size_t>(f)],
            f, 0, hdf5_output, true, out);
    }
    return images;
}
#endif

template<class Model>
std::vector<FrequencyImage> compute_slow_light_multifrequency_images(
    const Options&,
    const kpolaris::PassAParams<Real>&,
    Model,
    const std::vector<Real>&,
    bool,
    std::ofstream&) {
    throw std::runtime_error("slow_light multi-frequency is supported for iharm, kharma, athenak, bhac, and hamr only");
}

#if KPOLARIS_ENABLE_SLOW_LIGHT && (KPOLARIS_ENABLE_MODEL_IHARM || KPOLARIS_ENABLE_MODEL_KHARMA)
std::vector<FrequencyImage> compute_slow_light_multifrequency_images(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::GRMHDRadiationModel<Real> model,
    const std::vector<Real>& frequencies,
    bool hdf5_output,
    std::ofstream& out) {
    std::vector<ImageHostData> data;
    if (opt.model == "iharm") {
#if KPOLARIS_ENABLE_MODEL_IHARM
        data = run_iharm_slow_light_multifrequency_image_data(opt, pass_a, model, frequencies);
#else
        throw std::runtime_error("slow_light requested for iharm but iharm support is not built");
#endif
    } else if (opt.model == "kharma") {
#if KPOLARIS_ENABLE_MODEL_KHARMA
        data = run_kharma_slow_light_multifrequency_image_data(opt, pass_a, model, frequencies);
#else
        throw std::runtime_error("slow_light requested for kharma but kharma support is not built");
#endif
    } else {
        throw std::runtime_error("slow_light is supported for iharm, kharma, athenak, bhac, and hamr only");
    }
    if (data.size() != frequencies.size()) {
        throw std::runtime_error("slow-light multi-frequency returned wrong image count");
    }
    std::vector<FrequencyImage> images(static_cast<size_t>(frequencies.size()));
    for (int f = 0; f < static_cast<int>(frequencies.size()); ++f) {
        images[static_cast<size_t>(f)] = pack_transport_frequency_image(
            opt, data[static_cast<size_t>(f)], frequencies[static_cast<size_t>(f)],
            f, 0, hdf5_output, true, out);
    }
    return images;
}
#endif

#if KPOLARIS_ENABLE_SLOW_LIGHT && KPOLARIS_ENABLE_MODEL_ATHENAK
std::vector<FrequencyImage> compute_slow_light_multifrequency_images(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::AthenaKDirectRadiationModel<Real> model,
    const std::vector<Real>& frequencies,
    bool hdf5_output,
    std::ofstream& out) {
    std::vector<ImageHostData> data =
        run_athenak_slow_light_multifrequency_image_data(opt, pass_a, model, frequencies);
    if (data.size() != frequencies.size()) {
        throw std::runtime_error("slow-light multi-frequency returned wrong image count");
    }
    std::vector<FrequencyImage> images(static_cast<size_t>(frequencies.size()));
    for (int f = 0; f < static_cast<int>(frequencies.size()); ++f) {
        images[static_cast<size_t>(f)] = pack_transport_frequency_image(
            opt, data[static_cast<size_t>(f)], frequencies[static_cast<size_t>(f)],
            f, 0, hdf5_output, true, out);
    }
    return images;
}
#endif


template<class Model>
void run_model(const Options& opt, Model model, const std::vector<ImageHostData>* precomputed = nullptr) {
    if (opt.repeat_images > 1) {
        if (precomputed) throw std::invalid_argument("repeat_images cannot reuse a precomputed image");
        // The model's device Views remain resident. Each call below performs
        // complete geodesic integration, transfer, host copy and file output.
        const std::filesystem::path base(opt.output);
        std::vector<std::filesystem::path> outputs;
        for (int i = 0; i < opt.repeat_images; ++i) {
            std::ostringstream suffix;
            suffix << "_repeat" << std::setfill('0') << std::setw(4) << i;
            const auto path = base.parent_path() /
                (base.stem().string() + suffix.str() + base.extension().string());
            if (std::filesystem::exists(path) || std::filesystem::exists(path.string() + ".params"))
                throw std::invalid_argument("repeat_images output already exists: " + path.string());
            outputs.push_back(path);
        }
        Kokkos::Timer repeat_timer;
        for (int i = 0; i < opt.repeat_images; ++i) {
            Options frame = opt;
            frame.repeat_images = 1;
            frame.output = outputs[i].string();
            std::cout << "repeat_image_begin " << i << " of " << opt.repeat_images << "\n";
            Kokkos::Timer frame_timer;
            run_model(frame, model);
            report_timing(frame, "repeat_image_elapsed", frame_timer.seconds());
            std::cout << "repeat_image_end " << i << "\n";
        }
        report_timing(opt, "repeat_images_total", repeat_timer.seconds());
        return;
    }
    Kokkos::Timer run_timer;
    if (opt.slow_light_snapshot_cache_gib > Real(0) && opt.model != "kharma")
        throw std::invalid_argument("slow_light_snapshot_cache_gib currently requires model=kharma; other models can select windows_per_block directly");
    kpolaris::PassAParams<Real> pass_a;
    pass_a.camera.nx = opt.nx;
    pass_a.camera.ny = opt.ny;
    pass_a.camera.radius = opt.radius;
    pass_a.camera.inclination = opt.inclination;
    pass_a.camera.fov = opt.fov;
    pass_a.camera.fov_y = opt.fovy > Real(0) ? opt.fovy : opt.fov;
    pass_a.camera.x_offset = opt.x_offset;
    pass_a.camera.y_offset = opt.y_offset;
    if (opt.camera == "pinhole" && opt.use_pinhole_pixel_bias) {
        pass_a.camera.x_offset += opt.pinhole_pixel_bias / Real(opt.nx) * opt.fov;
    }
    pass_a.camera.xspan = opt.camera == "parallel_plane" ? opt.xspan : Real(-1);
    pass_a.camera.yspan = (opt.camera == "parallel_plane" && opt.xspan > Real(0)) ?
                          (opt.yspan > Real(0) ? opt.yspan : opt.xspan * Real(opt.ny) / Real(opt.nx)) : Real(-1);
    pass_a.camera.model = opt.camera == "parallel_plane" ?
                          kpolaris::CameraModel::ParallelPlane : kpolaris::CameraModel::Pinhole;
    pass_a.coordinate_system = coordinate_system_from_name(opt.coordinate);
    if (pass_a.coordinate_system != kpolaris::CoordinateSystem::CartesianKS &&
        pass_a.camera.model == kpolaris::CameraModel::ParallelPlane) {
        throw std::runtime_error("parallel_plane camera is currently defined only with cartesian_ks coordinates");
    }
    pass_a.mass = Real(1);
    pass_a.spin = opt.spin;
    pass_a.inner_radius = opt.inner_radius;
    pass_a.outer_radius = opt.outer_radius;
    pass_a.step = opt.step;
    pass_a.max_steps = opt.max_steps;
    pass_a.adaptive = opt.adaptive;
    pass_a.direct_only = opt.direct_only;
    pass_a.emission_selection.equatorial_h_over_r = opt.equatorial_h_over_r;
    pass_a.emission_selection.equatorial_samples = opt.equatorial_samples;
    pass_a.emission_selection.faraday_rotation = opt.faraday_rotation;
    pass_a.adaptive_tolerance = opt.adaptive_tolerance;
    pass_a.min_step = opt.min_step;
    pass_a.max_step = opt.max_step;
    pass_a.max_radiation_step = opt.max_radiation_step;
    pass_a.max_radiation_depth = opt.max_radiation_depth;
    pass_a.max_absorption_depth = opt.max_absorption_depth;
    pass_a.max_faraday_depth = opt.max_faraday_depth;
    pass_a.fmks_startx1 = opt.fmks_startx1;
    pass_a.fmks_hslope = opt.fmks_hslope;
    pass_a.fmks_mks_smooth = opt.fmks_mks_smooth;
    pass_a.fmks_poly_alpha = opt.fmks_poly_alpha;
    pass_a.fmks_poly_xt = opt.fmks_poly_xt;
    pass_a.fmks_poly_norm = opt.fmks_poly_norm;

    const Real x_extent = pass_a.camera.xspan > Real(0) ? Real(2) * pass_a.camera.xspan : pass_a.camera.fov;
    const Real y_extent = pass_a.camera.yspan > Real(0) ? Real(2) * pass_a.camera.yspan :
                          (pass_a.camera.xspan > Real(0) ?
                           Real(2) * pass_a.camera.xspan * Real(opt.ny) / Real(opt.nx) :
                           (pass_a.camera.fov_y > Real(0) ? pass_a.camera.fov_y : pass_a.camera.fov));
    const Real dx = x_extent / Real(opt.nx);
    const Real dy = y_extent / Real(opt.ny);
    const Real x_half_extent = Real(0.5) * x_extent;
    const Real y_half_extent = Real(0.5) * y_extent;
    const std::string grid_units = opt.camera == "pinhole" ? "dimensionless_slope" : "M";
    const Real output_mbh = output_mbh_solar(opt);
    const FluxScale flux_scale = compute_flux_scale(opt);
    const std::vector<Real> frequencies = build_frequency_grid(opt);
    const bool multi_frequency = frequencies.size() > 1;
#if !KPOLARIS_ENABLE_SLOW_LIGHT
    if (opt.slow_light) {
        throw std::runtime_error("slow_light requested but KPOLARIS_ENABLE_SLOW_LIGHT is OFF");
    }
#endif
    if (opt.slow_light && opt.model != "iharm" && opt.model != "kharma" &&
        opt.model != "athenak" && opt.model != "bhac" && opt.model != "hamr") {
        throw std::runtime_error("slow_light is supported for iharm, kharma, athenak, bhac, and hamr only");
    }
    if (opt.slow_light_time_probe && opt.model != "iharm" && opt.model != "kharma" &&
        opt.model != "athenak" && opt.model != "bhac" && opt.model != "hamr") {
        throw std::runtime_error("slow_light_time_probe is supported for iharm, kharma, athenak, bhac, and hamr only");
    }
#if KPOLARIS_MAX_FREQUENCIES <= 1
    if (!opt.slow_light && multi_frequency && !opt.analysis_mode) {
        throw std::runtime_error(
            "multi-frequency shared-geometry transport is disabled in this build; "
            "reconfigure with -DKPOLARIS_MAX_FREQUENCIES=N where N > 1");
    }
#endif

    if ((!opt.slow_light_batch_jobs.empty() || opt.slow_light_windows_per_block > 1 || opt.slow_light_prefetch_snapshots > 0) &&
        !opt.slow_light && !opt.slow_light_time_probe)
        throw std::invalid_argument("resident slow-light blocks/batches require slow_light=1");
    if (!opt.slow_light_batch_jobs.empty() && opt.slow_light_time_probe)
        throw std::invalid_argument("slow_light_batch_jobs cannot be combined with a geometric time probe");
    if (!opt.slow_light_batch_jobs.empty() && !precomputed) {
#if KPOLARIS_ENABLE_SLOW_LIGHT
            const auto paths = build_slow_light_dump_paths(opt);
            const auto times = build_slow_light_dump_times(opt);
            std::ifstream input(opt.slow_light_batch_jobs);
            if (!input) throw std::invalid_argument("cannot open slow_light_batch_jobs");
            std::vector<kpolaris_image_detail::SlowLightBatchJob> jobs;
            std::vector<std::string> outputs;
            std::set<std::filesystem::path> destinations;
            std::string line;
            while (std::getline(input, line)) {
                const auto first = line.find_first_not_of(" \t\r");
                if (first == std::string::npos || line[first] == '#') continue;
                std::istringstream row(line);
                kpolaris_image_detail::SlowLightBatchJob job;
                std::string output, extra;
                if (!(row >> job.observation_time >> job.first >> job.last >> std::quoted(output)) || (row >> extra) || output.empty())
                    throw std::invalid_argument("batch job must contain: observer_time first_index last_index output_path");
                const auto destination = std::filesystem::weakly_canonical(output);
                if (!destinations.insert(destination).second || std::filesystem::exists(destination))
                    throw std::invalid_argument("slow-light batch output already exists or is duplicated: " + output);
                if (job.first >= job.last || job.last >= paths.size() || !std::isfinite(job.observation_time))
                    throw std::invalid_argument("invalid slow-light batch snapshot support or observer time");
                jobs.push_back(job);
                outputs.push_back(output);
            }
            if (jobs.empty()) throw std::invalid_argument("slow-light batch job list is empty");
            if (std::none_of(jobs.begin(), jobs.end(), [](const auto& job) { return job.first == 0; }))
                throw std::invalid_argument("slow-light batch union must start at snapshot zero; trim the input lists to its support");
            auto data = run_model_slow_light_batch(opt, pass_a, model, paths, times, jobs, frequencies);
            for (size_t j = 0; j < jobs.size(); ++j) {
                Options frame = opt;
                frame.output = outputs[j];
                if (!effective_parameter_output_path(opt).empty()) frame.parameter_output = "auto";
                frame.slow_light_observation_time = jobs[j].observation_time;
                frame.slow_light_batch_size = static_cast<int>(jobs.size());
                frame.slow_light_batch_index = static_cast<int>(j);
                // Block boundaries affect the numerical partition. Retain the
                // shared prefix in a block-mode replay so its resident blocks
                // have exactly the same alignment as the original batch.
                const size_t replay_first = opt.slow_light_step_mode == kpolaris::SlowLightStepMode::block
                    ? 0 : jobs[j].first;
                if (opt.model == "iharm") frame.iharm_dump = paths[replay_first];
                else if (opt.model == "kharma") frame.kharma_dump = paths[replay_first];
                else if (opt.model == "athenak") frame.athenak_dump = paths[replay_first];
                else if (opt.model == "bhac") frame.bhac_dump = paths[replay_first];
                else if (opt.model == "hamr") frame.hamr_dump = paths[replay_first];
                std::ostringstream path_list, time_list;
                time_list << std::setprecision(17);
                for (size_t k = replay_first; k <= jobs[j].last; ++k) {
                    if (k != replay_first) { path_list << ','; time_list << ','; }
                    path_list << paths[k]; time_list << times[k];
                }
                frame.slow_light_dump_list = path_list.str();
                frame.slow_light_time_list = time_list.str();
                // Each effective file is independently rerunnable as a single
                // image; batch provenance is stored separately as HDF5 metadata.
                frame.slow_light_batch_jobs.clear();
                std::vector<ImageHostData> frame_data;
                for (size_t f = 0; f < frequencies.size(); ++f)
                    frame_data.push_back(std::move(data[j * frequencies.size() + f]));
                run_model(frame, model, &frame_data);
            }
            return;
#else
        throw std::invalid_argument("slow-light support is disabled in this build");
#endif
    }

    const std::string effective_params_path = effective_parameter_output_path(opt);
    write_effective_parameter_file(effective_params_path, opt, model, pass_a, grid_units,
                                   x_half_extent, y_half_extent, dx, dy, flux_scale, frequencies);

    if (opt.slow_light_time_probe) {
#if KPOLARIS_ENABLE_SLOW_LIGHT
        run_slow_light_time_probe_for_coordinate(opt, pass_a);
        return;
#else
        throw std::runtime_error("slow_light_time_probe requested but KPOLARIS_ENABLE_SLOW_LIGHT is OFF");
#endif
    }

    const int npix = pass_a.camera.nx * pass_a.camera.ny;
    const bool hdf5_output = opt.output_format == "hdf5" ||
                             (opt.output_format == "auto" && opt.output.size() >= 3 &&
                              opt.output.substr(opt.output.size() - 3) == ".h5");
    if (opt.output_format != "auto" && opt.output_format != "hdf5" && opt.output_format != "csv") {
        throw std::runtime_error("unknown output format: " + opt.output_format);
    }
#if !KPOLARIS_ENABLE_CSV_OUTPUT
    if (!hdf5_output) {
        throw std::runtime_error("CSV image output is disabled in this build; reconfigure with -DKPOLARIS_ENABLE_CSV_OUTPUT=ON or use --format=hdf5/--output=*.h5");
    }
#endif

    std::vector<double> grid_x(opt.nx), grid_y(opt.ny);
    for (int i = 0; i < opt.nx; ++i) {
        grid_x[i] = static_cast<double>((Real(i) + Real(0.5)) * dx - x_half_extent + pass_a.camera.x_offset);
    }
    for (int j = 0; j < opt.ny; ++j) {
        grid_y[j] = static_cast<double>((Real(j) + Real(0.5)) * dy - y_half_extent + pass_a.camera.y_offset);
    }
    std::vector<double> frequency_values(frequencies.size());
    for (size_t i = 0; i < frequencies.size(); ++i) {
        frequency_values[i] = static_cast<double>(frequencies[i]);
    }

    std::ofstream out;
    if (!hdf5_output) {
        out.open(opt.output);
        if (!out) {
            throw std::runtime_error("failed to open output: " + opt.output);
        }
        out.precision(17);
        out << "# code,KPolaris\n";
        out << "# code_version," << kpolaris::build_info::version << "\n";
        out << "# code_revision," << kpolaris::build_info::source_revision << "\n";
        out << "# code_source_dirty," << kpolaris::build_info::source_dirty << "\n";
        out << "# code_source_fingerprint," << kpolaris::build_info::source_fingerprint << "\n";
        out << "# compiler_id," << kpolaris::build_info::compiler_id << "\n";
        out << "# compiler_version," << kpolaris::build_info::compiler_version << "\n";
        out << "# build_type," << kpolaris::build_info::build_type << "\n";
        out << "# model," << opt.model << "\n";
        out << "# parameter_file," << opt.parameter_file << "\n";
        out << "# effective_parameter_file," << effective_params_path << "\n";
        out << "# nx," << opt.nx << "\n";
        out << "# ny," << opt.ny << "\n";
        out << "# camera," << opt.camera << "\n";
        out << "# coordinate," << opt.coordinate << "\n";
        out << "# metric," << metric_name_for_options(opt) << "\n";
        out << "# stokes_convention,camera_frame\n";
        out << "# polarization_basis,camera_screen\n";
        out << "# evpa_0," << opt.evpa_0 << "\n";
        out << "# camera_radius," << opt.radius << "\n";
        out << "# inclination_rad," << opt.inclination << "\n";
        out << "# fov," << opt.fov << "\n";
        out << "# x_offset," << pass_a.camera.x_offset << "\n";
        out << "# y_offset," << pass_a.camera.y_offset << "\n";
        out << "# use_pinhole_pixel_bias," << opt.use_pinhole_pixel_bias << "\n";
        out << "# pinhole_pixel_bias," << opt.pinhole_pixel_bias << "\n";
        out << "# grid_units," << grid_units << "\n";
        out << "# x_half_extent," << x_half_extent << "\n";
        out << "# y_half_extent," << y_half_extent << "\n";
        out << "# dx," << dx << "\n";
        out << "# dy," << dy << "\n";
        if (opt.xspan > Real(0)) {
            const Real yspan_out = opt.yspan > Real(0) ? opt.yspan : opt.xspan * Real(opt.ny) / Real(opt.nx);
            out << "# xspan," << opt.xspan << "\n";
            out << "# yspan," << yspan_out << "\n";
        }
        if (opt.model != "binary_riaf") {
            out << "# spin," << opt.spin << "\n";
        }
        out << "# step," << opt.step << "\n";
        out << "# adaptive," << opt.adaptive << "\n";
        out << "# adaptive_tolerance," << opt.adaptive_tolerance << "\n";
        out << "# min_step," << opt.min_step << "\n";
        out << "# max_step," << opt.max_step << "\n";
        out << "# max_radiation_step," << opt.max_radiation_step << "\n";
        out << "# max_radiation_depth," << opt.max_radiation_depth << "\n";
        out << "# max_absorption_depth," << opt.max_absorption_depth << "\n";
        out << "# max_faraday_depth," << opt.max_faraday_depth << "\n";
        out << "# closure_x_warning," << opt.closure_x_warning << "\n";
        out << "# closure_k_warning," << opt.closure_k_warning << "\n";
        out << "# frame_error_warning," << opt.frame_error_warning << "\n";
        out << "# basis_identity_warning," << opt.basis_identity_warning << "\n";
        out << "# split_transport," << opt.split_transport << "\n";
        out << "# analysis_mode," << opt.analysis_mode << "\n";
        if (opt.model == "kharma") {
            out << "# kharma_ddc_native," << opt.kharma_ddc.native << "\n";
            out << "# kharma_ddc_socket," << opt.kharma_ddc.socket_path << "\n";
            out << "# kharma_ddc_manifest," << opt.kharma_ddc.manifest << "\n";
            out << "# kharma_ddc_timeout_seconds," << opt.kharma_ddc.timeout_seconds << "\n";
            out << "# kharma_ddc_protocol," << (opt.kharma_ddc.native ? "STAGE1" : "disabled") << "\n";
        }
        out << "# equatorial_h_over_r," << opt.equatorial_h_over_r << "\n";
        out << "# equatorial_samples," << opt.equatorial_samples << "\n";
        out << "# faraday_rotation," << opt.faraday_rotation << "\n";
        out << "# analysis_radial_bins," << opt.analysis_radial_bins << "\n";
        out << "# analysis_radial_min," <<
            (opt.analysis_radial_min > Real(0) ? opt.analysis_radial_min : pass_a.inner_radius) << "\n";
        out << "# analysis_radial_max," <<
            (opt.analysis_radial_max > Real(0) ? opt.analysis_radial_max : pass_a.outer_radius) << "\n";
        out << "# analysis_formation_fraction," <<
            opt.analysis_formation_fraction << "\n";
        out << "# slow_light," << opt.slow_light << "\n";
        out << "# slow_light_prefetch," << opt.slow_light_prefetch << "\n";
        out << "# slow_light_step_mode," << kpolaris::slow_light_step_mode_name(opt.slow_light_step_mode) << "\n";
        out << "# slow_light_windows_per_block," << opt.slow_light_windows_per_block << "\n";
        out << "# slow_light_batch_size," << opt.slow_light_batch_size << "\n";
        out << "# slow_light_batch_index," << opt.slow_light_batch_index << "\n";
        out << "# slow_light_time_probe," << opt.slow_light_time_probe << "\n";
        out << "# slow_light_observation_time," << opt.slow_light_observation_time << "\n";
        if (!opt.slow_light_dump_list.empty()) out << "# slow_light_dump_list," << opt.slow_light_dump_list << "\n";
        if (!opt.slow_light_time_list.empty()) out << "# slow_light_time_list," << opt.slow_light_time_list << "\n";
        if (!opt.slow_light_dump_pattern.empty()) out << "# slow_light_dump_pattern," << opt.slow_light_dump_pattern << "\n";
        out << "# radiation_substeps," << opt.radiation_substeps << "\n";
        out << "# max_steps," << opt.max_steps << "\n";
        out << "# nfreq," << frequencies.size() << "\n";
        out << "# frequency_hz," << frequencies.front() << "\n";
        out << "# frequency_list_hz," << frequency_list_string(frequencies) << "\n";
        if (opt.model == "binary_riaf") {
            out << "# inner_boundary,full_metric_radial_characteristic_capture_surrogate\n";
        } else {
            out << "# inner_radius," << opt.inner_radius << "\n";
        }
        out << "# outer_radius," << opt.outer_radius << "\n";
        Model metadata_model = model;
        set_model_frequency(metadata_model, frequencies.front());
        write_model_metadata(out, metadata_model, opt.model);
        if (multi_frequency) {
            out << "freq_index,frequency_hz,";
        }
        out << "ix,iy,I_inv,Q_inv,U_inv,V_inv,I_nu,Q_nu,U_nu,V_nu,reason,pass_a_steps,pass_b_steps,total_steps,closure_x,closure_k,final_null,frame_error,det_r,overlap_r11,overlap_r12,overlap_r21,overlap_r22,basis_identity_error,basis_rotation_angle\n";
    }

    Kokkos::Timer transport_timer;
    std::vector<FrequencyImage> images;
    if (precomputed) {
        if (precomputed->size() != frequencies.size()) throw std::logic_error("precomputed frequency count mismatch");
        for (size_t f = 0; f < frequencies.size(); ++f)
            images.push_back(pack_transport_frequency_image(opt, (*precomputed)[f], frequencies[f],
                static_cast<int>(f), 0, hdf5_output, multi_frequency, out));
    } else if (multi_frequency) {
        if (opt.slow_light) {
#if KPOLARIS_ENABLE_SLOW_LIGHT
            images = compute_slow_light_multifrequency_images(opt, pass_a, model, frequencies,
                                                              hdf5_output, out);
#else
            throw std::runtime_error("slow_light requested but KPOLARIS_ENABLE_SLOW_LIGHT is OFF");
#endif
        } else if (opt.analysis_mode) {
            for (size_t fi = 0; fi < frequencies.size(); ++fi) {
                images.push_back(compute_frequency_image(
                    opt, pass_a, model, frequencies[fi], static_cast<int>(fi),
                    hdf5_output, true, out));
            }
        } else {
#if KPOLARIS_MAX_FREQUENCIES > 1
            images = compute_multifrequency_images(opt, pass_a, model, frequencies,
                                                   hdf5_output, out);
#else
            throw std::runtime_error("multi-frequency shared-geometry transport is disabled in this build");
#endif
        }
    } else {
        images.push_back(compute_frequency_image(opt, pass_a, model, frequencies.front(),
                                                 0, hdf5_output, false, out));
    }
    const double transport_seconds = transport_timer.seconds();
    report_timing(opt, "image_hostcopy", transport_seconds);
    const FrequencyImage& first = images.front();

    Kokkos::Timer output_timer;
    if (hdf5_output) {
        H5::H5File file(opt.output, H5F_ACC_TRUNC);
        write_h5_string(file, "schema", "kpolaris_image");
        write_h5_int(file, "schema_version", 1);
        write_h5_string(file, "code", "KPolaris");
        write_h5_string(file, "code_version", kpolaris::build_info::version);
        write_h5_string(file, "code_revision", kpolaris::build_info::source_revision);
        write_h5_int(file, "code_source_dirty", kpolaris::build_info::source_dirty);
        write_h5_string(file, "code_source_fingerprint", kpolaris::build_info::source_fingerprint);
        write_h5_string(file, "compiler_id", kpolaris::build_info::compiler_id);
        write_h5_string(file, "compiler_version", kpolaris::build_info::compiler_version);
        write_h5_string(file, "build_type", kpolaris::build_info::build_type);
        write_h5_string(file, "model", opt.model);
        if (!opt.iharm_dump.empty()) write_h5_string(file, "iharm_dump", opt.iharm_dump);
        if (!opt.kharma_dump.empty()) write_h5_string(file, "kharma_dump", opt.kharma_dump);
        if (opt.model == "kharma") {
            write_h5_int(file, "kharma_ddc_native", opt.kharma_ddc.native);
            write_h5_string(file, "kharma_ddc_socket", opt.kharma_ddc.socket_path);
            write_h5_string(file, "kharma_ddc_manifest", opt.kharma_ddc.manifest);
            write_h5_string(file, "kharma_ddc_protocol", opt.kharma_ddc.native ? "STAGE1" : "disabled");
            write_h5_int(file, "kharma_ddc_timeout_seconds", opt.kharma_ddc.timeout_seconds);
        }
        if (!opt.athenak_dump.empty()) write_h5_string(file, "athenak_dump", opt.athenak_dump);
        if (!opt.bhac_dump.empty()) write_h5_string(file, "bhac_dump", opt.bhac_dump);
        if (!opt.hamr_dump.empty()) write_h5_string(file, "hamr_dump", opt.hamr_dump);
        write_h5_string(file, "parameter_file", opt.parameter_file);
        write_h5_string(file, "effective_parameter_file", effective_params_path);
        write_h5_string(file, "camera", opt.camera);
        write_h5_string(file, "coordinate", opt.coordinate);
        write_h5_string(file, "metric", metric_name_for_options(opt));
        write_h5_string(file, "stokes_convention", "camera_frame");
        write_h5_string(file, "polarization_basis", "camera_screen");
        write_h5_string(file, "evpa_0", opt.evpa_0);
        write_kpolaris_camera_conventions(file, opt.evpa_0);
        write_h5_int(file, "nx", opt.nx);
        write_h5_int(file, "ny", opt.ny);
        write_h5_int(file, "nfreq", static_cast<int>(frequencies.size()));
        write_h5_string(file, "frequency_list_hz", frequency_list_string(frequencies));
        write_h5_scalar(file, "frequency_hz", frequencies.front());
        write_h5_scalar(file, "camera_radius", opt.radius);
        write_h5_scalar(file, "inclination_rad", opt.inclination);
        write_h5_scalar(file, "fov", opt.fov);
        write_h5_scalar(file, "fovy", opt.fovy > Real(0) ? opt.fovy : opt.fov);
        write_h5_scalar(file, "dsource", opt.dsource_pc);
        write_h5_scalar(file, "fovx_dsource", opt.effective_fovx_dsource);
        write_h5_scalar(file, "fovy_dsource", opt.effective_fovy_dsource);
        write_h5_scalar(file, "mbh_solar", output_mbh);
        if (opt.model == "hamr") {
            write_h5_string(file, "hamr_hslope_source", hamr_hslope_source(opt));
        }
        write_h5_scalar(file, "image_width_x_M", opt.image_width_x);
        write_h5_scalar(file, "image_width_y_M", opt.image_width_y);
        write_h5_scalar(file, "x_offset", pass_a.camera.x_offset);
        write_h5_scalar(file, "y_offset", pass_a.camera.y_offset);
        write_h5_int(file, "use_pinhole_pixel_bias", opt.use_pinhole_pixel_bias);
        write_h5_scalar(file, "pinhole_pixel_bias", opt.pinhole_pixel_bias);
        write_h5_string(file, "grid_units", grid_units);
        write_h5_scalar(file, "x_half_extent", x_half_extent);
        write_h5_scalar(file, "y_half_extent", y_half_extent);
        write_h5_scalar(file, "dx", dx);
        write_h5_scalar(file, "dy", dy);
        write_h5_int(file, "flux_scale_valid", flux_scale.valid);
        write_h5_scalar(file, "l_unit_cm", flux_scale.l_unit_cm);
        write_h5_scalar(file, "dsource_cm", flux_scale.dsource_cm);
        write_h5_scalar(file, "pixel_solid_angle_sr",
                        nan_if_invalid(flux_scale, flux_scale.pixel_solid_angle_sr));
        write_h5_scalar(file, "intensity_to_flux_jy_per_pixel",
                        nan_if_invalid(flux_scale, flux_scale.intensity_to_flux_jy_per_pixel));
        write_h5_string(file, "image_stokes_units", "cgs_specific_intensity");
        write_h5_string(file, "flux_density_unit", "Jy");
        if (opt.xspan > Real(0)) {
            const Real yspan_out = opt.yspan > Real(0) ? opt.yspan : opt.xspan * Real(opt.ny) / Real(opt.nx);
            write_h5_scalar(file, "xspan", opt.xspan);
            write_h5_scalar(file, "yspan", yspan_out);
        }
        if (opt.model != "binary_riaf") {
            write_h5_scalar(file, "spin", opt.spin);
        }
        write_h5_scalar(file, "step", opt.step);
        write_h5_int(file, "adaptive", opt.adaptive);
        write_h5_scalar(file, "adaptive_tolerance", opt.adaptive_tolerance);
        write_h5_scalar(file, "min_step", opt.min_step);
        write_h5_scalar(file, "max_step", opt.max_step);
        write_h5_scalar(file, "max_radiation_step", opt.max_radiation_step);
        write_h5_scalar(file, "max_radiation_depth", opt.max_radiation_depth);
        write_h5_scalar(file, "max_absorption_depth", opt.max_absorption_depth);
        write_h5_scalar(file, "max_faraday_depth", opt.max_faraday_depth);
        write_h5_scalar(file, "closure_x_warning", opt.closure_x_warning);
        write_h5_scalar(file, "closure_k_warning", opt.closure_k_warning);
        write_h5_scalar(file, "frame_error_warning", opt.frame_error_warning);
        write_h5_scalar(file, "basis_identity_warning", opt.basis_identity_warning);
        write_h5_int(file, "split_transport", opt.split_transport);
        write_h5_int(file, "multifrequency_chunk_size", opt.multifrequency_chunk_size);
        write_h5_int(file, "effective_multifrequency_chunk_size",
                     effective_multifrequency_chunk_size(opt, frequencies.size()));
        write_h5_int(file, "analysis_mode", opt.analysis_mode);
        write_h5_int(file, "analysis_radial_bins", opt.analysis_radial_bins);
        if (opt.analysis_mode) {
#if KPOLARIS_ENABLE_ANALYSIS_MODE
            const kpolaris::AnalysisConfig<Real> analysis_config =
                make_analysis_config(opt, pass_a);
            write_h5_scalar(file, "analysis_radial_min", analysis_config.radial_min);
            write_h5_scalar(file, "analysis_radial_max", analysis_config.radial_max);
            write_h5_scalar(file, "analysis_formation_fraction",
                            analysis_config.formation_fraction);
#endif
        }
        write_h5_scalar(file, "equatorial_h_over_r", opt.equatorial_h_over_r);
        write_h5_int(file, "equatorial_samples", opt.equatorial_samples);
        write_h5_int(file, "faraday_rotation", opt.faraday_rotation);
        write_h5_string(file, "equatorial_emission_definition",
            "0 disabled; otherwise seed emission only where abs(cos(theta_BL))=abs(z_KS)/r_BL <= h_over_r; h is half-height; 1 retains all emission");
        write_h5_string(file, "equatorial_transfer_definition",
            "only jI,jQ,jU,jV are masked; absorption and Faraday conversion remain everywhere in the configured plasma domain; rhoV is independently controlled by faraday_rotation");
        write_h5_string(file, "equatorial_sampling",
            "limit accepted-step height-ratio variation to 2*h_over_r/equatorial_samples and refine first crossed wedge surface with the same RK composition");
        if (opt.equatorial_h_over_r > Real(0))
            write_h5_string(file, "equatorial_reference", "arXiv:2606.12518v2, Figure 5 and Section V.1; thin equatorial emission experiment");
        write_h5_int(file, "direct_only", opt.direct_only);
        write_h5_string(file, "emission_segment", opt.direct_only ? "n0_vertical_turn" : "all");
        if (opt.direct_only) {
            write_h5_string(file, "direct_only_reference", "arXiv:2512.09641, Section II.1");
            write_h5_string(file, "direct_only_boundary",
                "first observer-to-source away-to-midplane reversal of dz/ds; z=r*cos(theta)=Cartesian_KS_z");
            write_h5_string(file, "direct_only_transfer",
                "zero incident Stokes at refined first vertical turn; foreground transfer follows declared emission/Faraday selections; equivalent to j=0 on more distant segments");
            write_h5_string(file, "direct_only_no_turn", "retain all emission to ordinary model endpoint");
        }
        write_h5_int(file, "slow_light", opt.slow_light);
        write_h5_int(file, "slow_light_compiled", KPOLARIS_ENABLE_SLOW_LIGHT);
        write_h5_int(file, "slow_light_prefetch", opt.slow_light_prefetch);
        write_h5_string(file, "slow_light_step_mode", kpolaris::slow_light_step_mode_name(opt.slow_light_step_mode));
        write_h5_string(file, "slow_light_interpolation", kpolaris::slow_light_interpolation_name(opt.slow_light_interpolation));
        write_h5_int(file, "slow_light_pipeline", opt.slow_light_pipeline);
        write_h5_int(file, "slow_light_windows_per_block", opt.slow_light_windows_per_block);
        write_h5_int(file, "slow_light_prefetch_snapshots", opt.slow_light_prefetch_snapshots);
        write_h5_scalar(file, "slow_light_snapshot_cache_gib", opt.slow_light_snapshot_cache_gib);
        write_h5_int(file, "slow_light_batch_size", opt.slow_light_batch_size);
        write_h5_int(file, "slow_light_batch_index", opt.slow_light_batch_index);
        write_h5_int(file, "slow_light_time_probe", opt.slow_light_time_probe);
        write_h5_scalar(file, "slow_light_observation_time", opt.slow_light_observation_time);
        if (!opt.slow_light_dump_list.empty()) write_h5_string(file, "slow_light_dump_list", opt.slow_light_dump_list);
        if (!opt.slow_light_time_list.empty()) write_h5_string(file, "slow_light_time_list", opt.slow_light_time_list);
        if (!opt.slow_light_dump_pattern.empty()) write_h5_string(file, "slow_light_dump_pattern", opt.slow_light_dump_pattern);
        write_h5_int(file, "analysis_compiled", KPOLARIS_ENABLE_ANALYSIS_MODE);
        write_h5_int(file, "radiation_substeps", opt.radiation_substeps);
        write_h5_int(file, "max_steps", opt.max_steps);
        if (opt.model == "binary_riaf") {
            write_h5_string(file, "inner_boundary", "full_metric_radial_characteristic_capture_surrogate");
        } else {
            write_h5_scalar(file, "inner_radius", opt.inner_radius);
        }
        write_h5_scalar(file, "outer_radius", opt.outer_radius);
        Model first_model = model;
        set_model_frequency(first_model, frequencies.front());
        write_hdf5_model_metadata(file, first_model, opt.model);

        H5::Group header = file.createGroup("/header");
        write_h5_string(header, "schema", "kpolaris_image");
        write_h5_int(header, "schema_version", 1);
        write_h5_string(header, "code", "KPolaris");
        write_h5_string(header, "code_version", kpolaris::build_info::version);
        write_h5_string(header, "code_revision", kpolaris::build_info::source_revision);
        write_h5_int(header, "code_source_dirty", kpolaris::build_info::source_dirty);
        write_h5_string(header, "code_source_fingerprint", kpolaris::build_info::source_fingerprint);
        write_h5_string(header, "compiler_id", kpolaris::build_info::compiler_id);
        write_h5_string(header, "compiler_version", kpolaris::build_info::compiler_version);
        write_h5_string(header, "build_type", kpolaris::build_info::build_type);
        write_h5_string(header, "image_layout", "/frame_0/freq_i");
        write_h5_string(header, "diagnostics_layout", "/frame_0/diagnostics");
        write_h5_string(header, "analysis_layout", multi_frequency ?
                        "/frame_0/freq_N/analysis" : "/frame_0/analysis");
        write_h5_string(header, "parameter_file", opt.parameter_file);
        write_h5_string(header, "effective_parameter_file", effective_params_path);
        write_h5_string(header, "stokes_convention", "camera_frame");
        write_h5_string(header, "polarization_basis", "camera_screen");
        write_h5_string(header, "evpa_0", opt.evpa_0);

        H5::Group parameters = file.createGroup("/parameters");
        write_h5_string(parameters, "parameter_file", opt.parameter_file);
        write_h5_string(parameters, "effective_parameter_file", effective_params_path);
        H5::Group camera = parameters.createGroup("camera");
        write_h5_string(camera, "model", opt.camera);
        write_h5_string(camera, "coordinate", opt.coordinate);
        write_h5_int(camera, "nx", opt.nx);
        write_h5_int(camera, "ny", opt.ny);
        write_h5_scalar(camera, "radius", opt.radius);
        write_h5_scalar(camera, "inclination_rad", opt.inclination);
        write_h5_scalar(camera, "fov", opt.fov);
        write_h5_scalar(camera, "fovy", opt.fovy > Real(0) ? opt.fovy : opt.fov);
        write_h5_scalar(camera, "dsource", opt.dsource_pc);
        write_h5_scalar(camera, "fovx_dsource", opt.effective_fovx_dsource);
        write_h5_scalar(camera, "fovy_dsource", opt.effective_fovy_dsource);
        write_h5_scalar(camera, "mbh_solar", output_mbh);
        write_h5_scalar(camera, "image_width_x_M", opt.image_width_x);
        write_h5_scalar(camera, "image_width_y_M", opt.image_width_y);
        write_h5_scalar(camera, "x_offset", pass_a.camera.x_offset);
        write_h5_scalar(camera, "y_offset", pass_a.camera.y_offset);
        write_h5_string(camera, "grid_units", grid_units);
        write_h5_scalar(camera, "x_half_extent", x_half_extent);
        write_h5_scalar(camera, "y_half_extent", y_half_extent);
        write_h5_scalar(camera, "dx", dx);
        write_h5_scalar(camera, "dy", dy);
        write_h5_int(camera, "flux_scale_valid", flux_scale.valid);
        write_h5_scalar(camera, "l_unit_cm", flux_scale.l_unit_cm);
        write_h5_scalar(camera, "dsource_cm", flux_scale.dsource_cm);
        write_h5_scalar(camera, "pixel_solid_angle_sr",
                        nan_if_invalid(flux_scale, flux_scale.pixel_solid_angle_sr));
        write_h5_scalar(camera, "intensity_to_flux_jy_per_pixel",
                        nan_if_invalid(flux_scale, flux_scale.intensity_to_flux_jy_per_pixel));
        write_h5_int(camera, "use_pinhole_pixel_bias", opt.use_pinhole_pixel_bias);
        write_h5_scalar(camera, "pinhole_pixel_bias", opt.pinhole_pixel_bias);
        if (opt.xspan > Real(0)) {
            const Real yspan_out = opt.yspan > Real(0) ? opt.yspan : opt.xspan * Real(opt.ny) / Real(opt.nx);
            write_h5_scalar(camera, "xspan", opt.xspan);
            write_h5_scalar(camera, "yspan", yspan_out);
        }

        H5::Group spacetime = parameters.createGroup("spacetime");
        write_h5_string(spacetime, "metric", metric_name_for_options(opt));
        write_h5_string(spacetime, "coordinate", opt.coordinate);
        write_h5_scalar(spacetime, "mass", Real(1));
        write_h5_scalar(spacetime, "mbh_solar", output_mbh);
        write_h5_scalar(spacetime, "l_unit_cm", flux_scale.l_unit_cm);
        if (opt.model != "binary_riaf") {
            write_h5_scalar(spacetime, "spin", opt.spin);
        }
        if (opt.model == "binary_riaf") {
            const bool tabulated_trajectory =
                opt.binary_trajectory_model != "leading_quadrupole" &&
                opt.binary_trajectory_model != "leading_order" &&
                opt.binary_trajectory_model != "peters";
            write_h5_int(spacetime, "dynamic",
                         tabulated_trajectory || opt.binary_orbit);
            write_h5_string(spacetime, "metric_regime",
                            tabulated_trajectory &&
                                    opt.binary_trajectory_exact_remnant ?
                            "inspiral_merger_exact_kerr_remnant" :
                            "inspiral_only");
            write_h5_string(spacetime, "radiation_source_regime",
                            "two_analytic_mini_riafs_pretransition_only");
            write_h5_string(
                spacetime, "radiation_source_kinematic_scope",
                "instantaneous_spin_aligned_prograde_axisymmetric_prescription; no_precessing_frame_Omega_cross_X_term");
            write_h5_string(
                spacetime, "metric_admissibility_audit_scope",
                "sampled_observation_slice_proxy_not_full_past_light_cone");
            write_h5_string(
                spacetime, "capture_surface_interpretation",
                "smooth_full_metric_radial_characteristic_numerical_excision_proxy_not_an_apparent_or_event_horizon");
            write_h5_string(spacetime, "orbit_model",
                            tabulated_trajectory ?
                            opt.binary_trajectory_model :
                            (!opt.binary_orbit ? "static" :
                             (opt.binary_inspiral ?
                                  "leading_quadrupole_quasicircular" :
                                  "constant_separation_circular")));
            write_h5_string(spacetime, "trajectory_file",
                            opt.binary_trajectory_file);
            write_h5_string(spacetime, "trajectory_format",
                            opt.binary_trajectory_format);
            write_h5_string(spacetime, "trajectory_schema",
                            opt.binary_trajectory_schema);
            write_h5_string(spacetime, "trajectory_sha256",
                            opt.binary_trajectory_sha256);
            write_h5_string(spacetime, "trajectory_position_gauge",
                            opt.binary_trajectory_gauge);
            write_h5_string(spacetime, "trajectory_generator",
                            opt.binary_trajectory_generator);
            write_h5_string(spacetime, "trajectory_generator_version",
                            opt.binary_trajectory_generator_version);
            write_h5_string(
                spacetime, "trajectory_merger_reach_contract",
                opt.binary_trajectory_merger_reach_contract);
            write_h5_string(spacetime, "trajectory_declared_model",
                            opt.binary_trajectory_declared_model);
            write_h5_string(spacetime, "trajectory_pn_terms",
                            opt.binary_trajectory_pn_terms);
            write_h5_int(spacetime, "trajectory_source_verified",
                         opt.binary_trajectory_source_verified);
            write_h5_string(spacetime, "trajectory_source_doi",
                            opt.binary_trajectory_source_doi);
            write_h5_string(spacetime, "trajectory_pn_4pn_scope",
                            opt.binary_trajectory_pn_4pn_scope);
            write_h5_string(spacetime, "trajectory_source_verification",
                            opt.binary_trajectory_source_verification);
            write_h5_string(spacetime,
                            "trajectory_upstream_cbwaves_sha256",
                            opt.binary_trajectory_upstream_cbwaves_sha256);
            write_h5_string(spacetime,
                            "trajectory_patched_cbwaves_sha256",
                            opt.binary_trajectory_patched_cbwaves_sha256);
            write_h5_int(
                spacetime, "trajectory_merger_separation_reached",
                opt.binary_trajectory_merger_separation_reached);
            write_h5_string(spacetime, "trajectory_status",
                            opt.binary_trajectory_status);
            write_h5_string(spacetime, "trajectory_boost_velocity_model",
                            opt.binary_trajectory_boost_velocity_model);
            write_h5_int(
                spacetime, "trajectory_worldline_velocity_consistent",
                opt.binary_trajectory_worldline_velocity_consistent);
            write_h5_int(spacetime, "trajectory_future_extension_enabled",
                         opt.binary_trajectory_future_extension);
            write_h5_string(spacetime, "trajectory_interpolation",
                            opt.binary_trajectory_interpolation);
            write_h5_int(spacetime, "trajectory_samples",
                         opt.binary_trajectory_samples);
            write_h5_scalar(spacetime, "trajectory_time_min_M",
                            opt.binary_trajectory_t_min);
            write_h5_scalar(spacetime, "trajectory_time_max_M",
                            opt.binary_trajectory_t_max);
            write_h5_scalar(spacetime, "trajectory_transition_start_M",
                            opt.binary_transition_start);
            write_h5_scalar(spacetime, "trajectory_transition_end_M",
                            opt.binary_transition_end);
            write_h5_int(spacetime, "trajectory_exact_remnant_tail",
                         opt.binary_trajectory_exact_remnant);
            write_h5_scalar(spacetime, "trajectory_time_offset_M",
                            opt.binary_trajectory_time_offset);
            write_h5_scalar(spacetime, "trajectory_observation_time_M",
                            opt.binary_observation_time +
                            opt.binary_trajectory_time_offset);
            if (!tabulated_trajectory) {
                write_h5_scalar(spacetime, "mass_ratio_m2_over_m1",
                                opt.binary_mass_ratio);
                write_h5_scalar(spacetime, "chi1_z", opt.binary_chi1);
                write_h5_scalar(spacetime, "chi2_z", opt.binary_chi2);
                write_h5_scalar(spacetime, "reference_separation_M",
                                opt.binary_reference_separation);
                write_h5_scalar(spacetime, "reference_phase_rad",
                                opt.binary_reference_phase);
                write_h5_scalar(spacetime, "reference_time_M",
                                opt.binary_reference_time);
                write_h5_scalar(spacetime, "minimum_separation_M",
                                opt.binary_minimum_separation);
            }
            write_h5_scalar(spacetime, "observation_time_M", opt.binary_observation_time);
            write_h5_scalar(spacetime, "metric_derivative_step", opt.binary_metric_derivative_step);
            write_h5_scalar(spacetime, "capture_factor", opt.binary_capture_factor);
        }

        H5::Group integration = parameters.createGroup("integration");
        write_h5_scalar(integration, "step", opt.step);
        write_h5_int(integration, "adaptive", opt.adaptive);
        write_h5_scalar(integration, "adaptive_tolerance", opt.adaptive_tolerance);
        write_h5_scalar(integration, "min_step", opt.min_step);
        write_h5_scalar(integration, "max_step", opt.max_step);
        write_h5_scalar(integration, "max_radiation_step", opt.max_radiation_step);
        write_h5_scalar(integration, "max_radiation_depth", opt.max_radiation_depth);
        write_h5_scalar(integration, "max_absorption_depth", opt.max_absorption_depth);
        write_h5_scalar(integration, "max_faraday_depth", opt.max_faraday_depth);
        write_h5_scalar(integration, "closure_x_warning", opt.closure_x_warning);
        write_h5_scalar(integration, "closure_k_warning", opt.closure_k_warning);
        write_h5_scalar(integration, "frame_error_warning", opt.frame_error_warning);
        write_h5_scalar(integration, "basis_identity_warning", opt.basis_identity_warning);
        write_h5_int(integration, "split_transport", opt.split_transport);
        write_h5_int(integration, "analysis_mode", opt.analysis_mode);
        write_h5_int(integration, "analysis_radial_bins", opt.analysis_radial_bins);
        if (opt.analysis_mode) {
#if KPOLARIS_ENABLE_ANALYSIS_MODE
            const kpolaris::AnalysisConfig<Real> analysis_config =
                make_analysis_config(opt, pass_a);
            write_h5_scalar(integration, "analysis_radial_min", analysis_config.radial_min);
            write_h5_scalar(integration, "analysis_radial_max", analysis_config.radial_max);
            write_h5_scalar(integration, "analysis_formation_fraction",
                            analysis_config.formation_fraction);
#endif
        }
        write_h5_int(integration, "slow_light", opt.slow_light);
        write_h5_int(integration, "slow_light_compiled", KPOLARIS_ENABLE_SLOW_LIGHT);
        write_h5_int(integration, "slow_light_prefetch", opt.slow_light_prefetch);
        write_h5_int(integration, "slow_light_time_probe", opt.slow_light_time_probe);
        write_h5_scalar(integration, "slow_light_observation_time", opt.slow_light_observation_time);
        write_h5_int(integration, "max_steps", opt.max_steps);
        if (opt.model == "binary_riaf") {
            write_h5_string(integration, "inner_boundary",
                            "full_metric_radial_characteristic_capture_surrogate");
            write_h5_scalar(integration, "capture_factor",
                            opt.binary_capture_factor);
        } else {
            write_h5_scalar(integration, "inner_radius", opt.inner_radius);
        }
        write_h5_scalar(integration, "outer_radius", opt.outer_radius);
        write_h5_int(integration, "radiation_substeps", opt.radiation_substeps);

        H5::Group radiation = parameters.createGroup("radiation");
        write_h5_string(radiation, "model", opt.model);
        write_h5_scalar(radiation, "mbh_solar", output_mbh);
        if (opt.model == "hamr") {
            write_h5_string(radiation, "hamr_hslope_source", hamr_hslope_source(opt));
        }
        if (opt.model == "kharma") {
            write_h5_int(radiation, "kharma_reverse_field", opt.kharma_reverse_field);
        }
        if (!opt.iharm_dump.empty()) write_h5_string(radiation, "iharm_dump", opt.iharm_dump);
        if (!opt.kharma_dump.empty()) write_h5_string(radiation, "kharma_dump", opt.kharma_dump);
        if (opt.model == "kharma") {
            write_h5_int(radiation, "kharma_ddc_native", opt.kharma_ddc.native);
            write_h5_string(radiation, "kharma_ddc_socket", opt.kharma_ddc.socket_path);
            write_h5_string(radiation, "kharma_ddc_manifest", opt.kharma_ddc.manifest);
            write_h5_string(radiation, "kharma_ddc_protocol", opt.kharma_ddc.native ? "STAGE1" : "disabled");
            write_h5_int(radiation, "kharma_ddc_timeout_seconds", opt.kharma_ddc.timeout_seconds);
        }
        if (!opt.athenak_dump.empty()) write_h5_string(radiation, "athenak_dump", opt.athenak_dump);
        if (!opt.bhac_dump.empty()) write_h5_string(radiation, "bhac_dump", opt.bhac_dump);
        if (!opt.hamr_dump.empty()) write_h5_string(radiation, "hamr_dump", opt.hamr_dump);
        write_h5_int(radiation, "slow_light", opt.slow_light);
        write_h5_int(radiation, "slow_light_compiled", KPOLARIS_ENABLE_SLOW_LIGHT);
        write_h5_int(radiation, "slow_light_prefetch", opt.slow_light_prefetch);
        write_h5_int(radiation, "slow_light_time_probe", opt.slow_light_time_probe);
        write_h5_scalar(radiation, "slow_light_observation_time", opt.slow_light_observation_time);
        if (!opt.slow_light_dump_list.empty()) write_h5_string(radiation, "slow_light_dump_list", opt.slow_light_dump_list);
        if (!opt.slow_light_time_list.empty()) write_h5_string(radiation, "slow_light_time_list", opt.slow_light_time_list);
        if (!opt.slow_light_dump_pattern.empty()) write_h5_string(radiation, "slow_light_dump_pattern", opt.slow_light_dump_pattern);
        write_h5_scalar(radiation, "equatorial_h_over_r", opt.equatorial_h_over_r);
        write_h5_int(radiation, "equatorial_samples", opt.equatorial_samples);
        write_h5_int(radiation, "faraday_rotation", opt.faraday_rotation);
        write_h5_int(radiation, "nfreq", static_cast<int>(frequencies.size()));
        write_h5_int(radiation, "multifrequency_chunk_size", opt.multifrequency_chunk_size);
        write_h5_int(radiation, "effective_multifrequency_chunk_size",
                     effective_multifrequency_chunk_size(opt, frequencies.size()));
        write_h5_scalar(radiation, "frequency_hz", frequencies.front());
        write_h5_string(radiation, "frequency_list_hz", frequency_list_string(frequencies));
        write_h5_string(radiation, "freq_spacing", opt.freq_spacing);
        write_h5_string(radiation, "stokes_convention", "camera_frame");
        write_h5_dataset_1d(radiation, "frequencies", frequency_values);
        write_hdf5_model_metadata(radiation, first_model, opt.model);

        H5::Group grid = file.createGroup("/grid");
        write_h5_string(grid, "units", grid_units);
        write_h5_dataset_1d(grid, "x", grid_x);
        write_h5_dataset_1d(grid, "y", grid_y);
        write_h5_dataset_1d(grid, "frequency_hz", frequency_values);

        H5::Group results = file.createGroup("/results");
        write_h5_int(results, "pixels", npix);
        write_h5_int(results, "returned", first.returned);
        write_h5_int(results, "nfreq", static_cast<int>(frequencies.size()));
        write_h5_scalar(results, "max_closure_x", first.diagnostics.max_closure_x);
        write_h5_scalar(results, "max_closure_k", first.diagnostics.max_closure_k);
        write_h5_scalar(results, "max_frame_error", first.diagnostics.max_frame_error);
        write_h5_scalar(results, "max_basis_identity_error", first.diagnostics.max_basis_identity_error);
        write_h5_int(results, "max_closure_x_ix", pixel_ix(first.diagnostics.max_closure_x_pixel, opt.nx));
        write_h5_int(results, "max_closure_x_iy", pixel_iy(first.diagnostics.max_closure_x_pixel, opt.nx));
        write_h5_int(results, "closure_x_warning_count", first.diagnostics.closure_x_warning_count);
        write_h5_int(results, "closure_k_warning_count", first.diagnostics.closure_k_warning_count);
        write_h5_int(results, "frame_error_warning_count", first.diagnostics.frame_error_warning_count);
        write_h5_int(results, "basis_identity_warning_count", first.diagnostics.basis_identity_warning_count);
        write_h5_int(results, "diagnostic_warning_count", first.diagnostics.any_warning_count);
        write_h5_scalar(results, "mean_pass_a_steps", first.mean_pass_a_steps);
        write_h5_scalar(results, "mean_pass_b_steps", first.mean_pass_b_steps);
        write_h5_scalar(results, "mean_total_steps", first.mean_total_steps);
        write_h5_int(results, "max_pass_a_steps", first.max_pass_a_steps);
        write_h5_int(results, "max_pass_b_steps", first.max_pass_b_steps);
        write_h5_int(results, "max_total_steps", first.max_total_steps);
        write_h5_scalar(results, "sum_I_inv", first.sum_i);
        write_h5_scalar(results, "sum_Q_inv", first.sum_q);
        write_h5_scalar(results, "sum_U_inv", first.sum_u);
        write_h5_scalar(results, "sum_V_inv", first.sum_v);
        write_h5_scalar(results, "intensity_to_flux_jy_per_pixel",
                        nan_if_invalid(flux_scale, flux_scale.intensity_to_flux_jy_per_pixel));
        std::vector<double> sum_i_by_freq(images.size()), sum_q_by_freq(images.size());
        std::vector<double> sum_u_by_freq(images.size()), sum_v_by_freq(images.size());
        std::vector<double> flux_i_by_freq(images.size()), flux_q_by_freq(images.size());
        std::vector<double> flux_u_by_freq(images.size()), flux_v_by_freq(images.size());
        for (size_t fi = 0; fi < images.size(); ++fi) {
            const FrequencyImage& image = images[fi];
            sum_i_by_freq[fi] = static_cast<double>(image.sum_i);
            sum_q_by_freq[fi] = static_cast<double>(image.sum_q);
            sum_u_by_freq[fi] = static_cast<double>(image.sum_u);
            sum_v_by_freq[fi] = static_cast<double>(image.sum_v);
            flux_i_by_freq[fi] =
                static_cast<double>(flux_density_jy(flux_scale, image.sum_i, image.frequency));
            flux_q_by_freq[fi] =
                static_cast<double>(flux_density_jy(flux_scale, image.sum_q, image.frequency));
            flux_u_by_freq[fi] =
                static_cast<double>(flux_density_jy(flux_scale, image.sum_u, image.frequency));
            flux_v_by_freq[fi] =
                static_cast<double>(flux_density_jy(flux_scale, image.sum_v, image.frequency));
        }
        if (!images.empty()) {
            write_h5_scalar(results, "flux_I_Jy", flux_i_by_freq.front());
            write_h5_scalar(results, "flux_Q_Jy", flux_q_by_freq.front());
            write_h5_scalar(results, "flux_U_Jy", flux_u_by_freq.front());
            write_h5_scalar(results, "flux_V_Jy", flux_v_by_freq.front());
        }
        write_h5_dataset_1d(results, "frequency_hz", frequency_values);
        write_h5_dataset_1d(results, "sum_I_inv_by_freq", sum_i_by_freq);
        write_h5_dataset_1d(results, "sum_Q_inv_by_freq", sum_q_by_freq);
        write_h5_dataset_1d(results, "sum_U_inv_by_freq", sum_u_by_freq);
        write_h5_dataset_1d(results, "sum_V_inv_by_freq", sum_v_by_freq);
        write_h5_dataset_1d(results, "flux_I_Jy_by_freq", flux_i_by_freq);
        write_h5_dataset_1d(results, "flux_Q_Jy_by_freq", flux_q_by_freq);
        write_h5_dataset_1d(results, "flux_U_Jy_by_freq", flux_u_by_freq);
        write_h5_dataset_1d(results, "flux_V_Jy_by_freq", flux_v_by_freq);

        H5::Group frame = file.createGroup("/frame_0");
        write_h5_int(frame, "index", 0);
        write_h5_int(frame, "nfreq", static_cast<int>(frequencies.size()));
        for (size_t fi = 0; fi < images.size(); ++fi) {
            const FrequencyImage& image = images[fi];
            H5::Group freq_group = frame.createGroup("freq_" + std::to_string(fi));
            write_h5_int(freq_group, "index", static_cast<int>(fi));
            write_h5_scalar(freq_group, "frequency", image.frequency);
            write_h5_scalar(freq_group, "frequency_hz", image.frequency);
            write_h5_string(freq_group, "stokes_convention", "camera_frame");
            write_h5_string(freq_group, "polarization_basis", "camera_screen");
            write_h5_string(freq_group, "evpa_0", opt.evpa_0);
            write_h5_string(freq_group, "stokes_units", "cgs_specific_intensity");
            write_h5_scalar(freq_group, "intensity_to_flux_jy_per_pixel",
                            nan_if_invalid(flux_scale, flux_scale.intensity_to_flux_jy_per_pixel));
            write_h5_scalar(freq_group, "flux_I_Jy",
                            flux_density_jy(flux_scale, image.sum_i, image.frequency));
            write_h5_scalar(freq_group, "flux_Q_Jy",
                            flux_density_jy(flux_scale, image.sum_q, image.frequency));
            write_h5_scalar(freq_group, "flux_U_Jy",
                            flux_density_jy(flux_scale, image.sum_u, image.frequency));
            write_h5_scalar(freq_group, "flux_V_Jy",
                            flux_density_jy(flux_scale, image.sum_v, image.frequency));
            Model freq_model = model;
            set_model_frequency(freq_model, image.frequency);
            write_hdf5_model_metadata(freq_group, freq_model, opt.model);
            write_h5_dataset_2d(freq_group, "I", image.I_nu, opt.nx, opt.ny);
            write_h5_dataset_2d(freq_group, "Q", image.Q_nu, opt.nx, opt.ny);
            write_h5_dataset_2d(freq_group, "U", image.U_nu, opt.nx, opt.ny);
            write_h5_dataset_2d(freq_group, "V", image.V_nu, opt.nx, opt.ny);
            write_h5_dataset_2d(freq_group, "I_inv", image.I_inv, opt.nx, opt.ny);
            write_h5_dataset_2d(freq_group, "Q_inv", image.Q_inv, opt.nx, opt.ny);
            write_h5_dataset_2d(freq_group, "U_inv", image.U_inv, opt.nx, opt.ny);
            write_h5_dataset_2d(freq_group, "V_inv", image.V_inv, opt.nx, opt.ny);
            if (multi_frequency && image.has_analysis) {
                H5::Group analysis = freq_group.createGroup("analysis");
                write_physical_analysis_group(analysis, image, opt);
            }
        }
        H5::Group diag = frame.createGroup("diagnostics");
        write_h5_dataset_2d_int(diag, "reason", first.reason, opt.nx, opt.ny);
        write_h5_dataset_2d_int(diag, "pass_a_steps", first.pass_a_steps, opt.nx, opt.ny);
        write_h5_dataset_2d_int(diag, "steps", first.steps, opt.nx, opt.ny);
        write_h5_dataset_2d_int(diag, "total_steps", first.total_steps, opt.nx, opt.ny);
        write_h5_dataset_2d(diag, "closure_x", first.closure_x, opt.nx, opt.ny);
        write_h5_dataset_2d(diag, "closure_k", first.closure_k, opt.nx, opt.ny);
        write_h5_dataset_2d(diag, "final_null", first.final_null, opt.nx, opt.ny);
        write_h5_dataset_2d(diag, "frame_error", first.frame_error, opt.nx, opt.ny);
        write_h5_dataset_2d(diag, "det_r", first.det_r, opt.nx, opt.ny);
        write_h5_dataset_2d(diag, "overlap_r11", first.overlap_r11, opt.nx, opt.ny);
        write_h5_dataset_2d(diag, "overlap_r12", first.overlap_r12, opt.nx, opt.ny);
        write_h5_dataset_2d(diag, "overlap_r21", first.overlap_r21, opt.nx, opt.ny);
        write_h5_dataset_2d(diag, "overlap_r22", first.overlap_r22, opt.nx, opt.ny);
        write_h5_dataset_2d(diag, "basis_identity_error", first.basis_identity_error, opt.nx, opt.ny);
        write_h5_dataset_2d(diag, "basis_rotation_angle", first.basis_rotation_angle, opt.nx, opt.ny);
        if (first.has_analysis && !multi_frequency) {
            H5::Group analysis = frame.createGroup("analysis");
            write_physical_analysis_group(analysis, first, opt);
        }
    }
    report_timing(opt, "image_output", output_timer.seconds());
    report_timing(opt, "image_run_model_total", run_timer.seconds());

    std::cout << "KPolaris model image output\n";
    std::cout << "version " << kpolaris::build_info::version << "\n";
    std::cout << "revision " << kpolaris::build_info::source_revision << "\n";
    std::cout << "source_dirty " << kpolaris::build_info::source_dirty_label() << "\n";
    std::cout << "model " << opt.model << "\n";
    std::cout << "format " << (hdf5_output ? "hdf5" : "csv") << "\n";
    std::cout << "analysis_mode " << opt.analysis_mode << "\n";
    std::cout << "slow_light " << opt.slow_light << "\n";
    std::cout << "output " << opt.output << "\n";
    if (!effective_params_path.empty()) {
        std::cout << "parameters " << effective_params_path << "\n";
    }
    std::cout << "pixels " << npix << "\n";
    std::cout << "frequencies " << frequencies.size() << "\n";
    std::cout << "intensity_to_flux_jy_per_pixel "
              << nan_if_invalid(flux_scale, flux_scale.intensity_to_flux_jy_per_pixel) << "\n";
    for (size_t fi = 0; fi < images.size(); ++fi) {
        const FrequencyImage& image = images[fi];
        const Real flux_i = flux_density_jy(flux_scale, image.sum_i, image.frequency);
        const Real flux_q = flux_density_jy(flux_scale, image.sum_q, image.frequency);
        const Real flux_u = flux_density_jy(flux_scale, image.sum_u, image.frequency);
        const Real flux_v = flux_density_jy(flux_scale, image.sum_v, image.frequency);
        std::cout << "freq_index " << fi << "\n";
        std::cout << "frequency_hz " << image.frequency << "\n";
        std::cout << "returned " << image.returned << "\n";
        std::cout << "mean_pass_a_steps " << image.mean_pass_a_steps << "\n";
        std::cout << "mean_pass_b_steps " << image.mean_pass_b_steps << "\n";
        std::cout << "mean_total_steps " << image.mean_total_steps << "\n";
        std::cout << "max_pass_a_steps " << image.max_pass_a_steps << "\n";
        std::cout << "max_pass_b_steps " << image.max_pass_b_steps << "\n";
        std::cout << "max_total_steps " << image.max_total_steps << "\n";
        std::cout << "I_inv " << image.sum_i << "\n";
        std::cout << "Q_inv " << image.sum_q << "\n";
        std::cout << "U_inv " << image.sum_u << "\n";
        std::cout << "V_inv " << image.sum_v << "\n";
        std::cout << "flux_I_Jy " << flux_i << "\n";
        std::cout << "flux_Q_Jy " << flux_q << "\n";
        std::cout << "flux_U_Jy " << flux_u << "\n";
        std::cout << "flux_V_Jy " << flux_v << "\n";
    }
    std::cout << "max_closure_x " << first.diagnostics.max_closure_x
              << " pixel " << pixel_ix(first.diagnostics.max_closure_x_pixel, opt.nx)
              << ',' << pixel_iy(first.diagnostics.max_closure_x_pixel, opt.nx) << "\n";
    std::cout << "max_closure_k " << first.diagnostics.max_closure_k << "\n";
    std::cout << "max_frame_error " << first.diagnostics.max_frame_error << "\n";
    std::cout << "max_basis_identity_error "
              << first.diagnostics.max_basis_identity_error << "\n";
    if (first.diagnostics.any_warning_count > 0) {
        std::cerr << "warning: " << first.diagnostics.any_warning_count
                  << " returned ray(s) exceeded closure/frame diagnostic thresholds. "
                  << "Worst closure_x pixel is (ix,iy)=("
                  << pixel_ix(first.diagnostics.max_closure_x_pixel, opt.nx) << ','
                  << pixel_iy(first.diagnostics.max_closure_x_pixel, opt.nx)
                  << ") with closure_x=" << first.diagnostics.max_closure_x << ".\n";
        std::cerr << "warning: first try lowering --adaptive_tolerance and --min_step, "
                  << "and increasing --max_steps if rejected or small-step rays stop early. "
                  << "If high-order photon-ring rays remain problematic, also inspect that pixel "
                  << "with a single-ray trace.\n";
    }
}



} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--version" || std::string(argv[i]) == "-V") {
            kpolaris::build_info::print_version(std::cout);
            return 0;
        }
    }
    Kokkos::initialize(argc, argv);
    try {
        Options opt = parse_options(argc, argv);
        opt.model = lowercase_copy(opt.model);
        opt.camera = canonical_camera_name(opt.camera);
        opt.coordinate = canonical_coordinate_name(opt.coordinate);
        validate_options(opt);
            resolve_camera_extents(opt);
        validate_resolved_camera(opt);
        const Real horizon = Real(1) + std::sqrt(std::max<Real>(Real(0), Real(1) - opt.spin * opt.spin));
        if (opt.inner_radius <= Real(0) && !model_defines_inner_radius(opt.model)) {
            opt.inner_radius = horizon * Real(1.05);
        }
        bool handled_model = false;
#if KPOLARIS_ENABLE_MODEL_RIAF
        if (!handled_model && opt.model == "riaf") {
            handled_model = true;
            kpolaris::RIAFAnalyticRadiationModel<Real> model;
            model.freq_cgs = opt.freq;
            model.r_min = opt.riaf_r_min;
            model.r_max = opt.riaf_r_max;
            model.nth0 = opt.riaf_nth0;
            model.Te0 = opt.riaf_Te0;
            model.disk_h = opt.riaf_disk_h;
            model.pow_nth = opt.riaf_pow_nth;
            model.pow_T = opt.riaf_pow_T;
            model.ne_unit = opt.riaf_ne_unit;
            model.te_unit = opt.riaf_te_unit;
            model.mbh_solar = opt.riaf_mbh_solar;
            model.keplerian_factor = opt.riaf_keplerian_factor;
            model.infall_factor = opt.riaf_infall_factor;
            model.emission_type = effective_emission_type(opt.emission_type,
                                                           KPOLARIS_RIAF_COMPILED_EMISSION_TYPE,
                                                           1);
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
            if (opt.outer_radius <= Real(0)) {
                opt.outer_radius = model.r_max * Real(1.15);
            }
            run_model(opt, model);
        }
#endif
#if KPOLARIS_ENABLE_MODEL_BINARY_RIAF
        if (!handled_model && opt.model == "binary_riaf") {
            handled_model = true;
            kpolaris::BinaryRIAFRadiationModel<Real> model;
            model.disk.freq_cgs = opt.freq;
            model.disk.r_min = opt.riaf_r_min;
            model.disk.r_max = opt.riaf_r_max;
            model.disk.nth0 = opt.riaf_nth0;
            model.disk.Te0 = opt.riaf_Te0;
            model.disk.disk_h = opt.riaf_disk_h;
            model.disk.pow_nth = opt.riaf_pow_nth;
            model.disk.pow_T = opt.riaf_pow_T;
            model.disk.ne_unit = opt.riaf_ne_unit;
            model.disk.te_unit = opt.riaf_te_unit;
            model.disk.mbh_solar = opt.riaf_mbh_solar;
            model.disk.keplerian_factor = opt.riaf_keplerian_factor;
            model.disk.infall_factor = opt.riaf_infall_factor;
            model.disk.emission_type = effective_emission_type(
                opt.emission_type, KPOLARIS_RIAF_COMPILED_EMISSION_TYPE, 1);
            model.disk.nonthermal_kappa = opt.nonthermal_kappa;
            model.disk.variable_kappa = opt.variable_kappa;
            model.disk.variable_kappa_min = opt.variable_kappa_min;
            model.disk.variable_kappa_interp_start = opt.variable_kappa_interp_start;
            model.disk.variable_kappa_max = opt.variable_kappa_max;
            model.disk.powerlaw_p = opt.powerlaw_p;
            model.disk.powerlaw_eta = opt.powerlaw_eta;
            model.disk.powerlaw_gamma_min = opt.powerlaw_gamma_min;
            model.disk.powerlaw_gamma_max = opt.powerlaw_gamma_max;
            model.disk.powerlaw_gamma_cutoff = opt.powerlaw_gamma_cutoff;
            model.mass_ratio = opt.binary_mass_ratio;
            model.chi1 = opt.binary_chi1;
            model.chi2 = opt.binary_chi2;
            model.reference_separation = opt.binary_reference_separation;
            model.reference_phase = opt.binary_reference_phase;
            model.reference_time = opt.binary_reference_time;
            model.observation_time = opt.binary_observation_time;
            model.minimum_separation = opt.binary_minimum_separation;
            model.inspiral_enabled = opt.binary_inspiral;
            model.orbit_enabled = opt.binary_orbit;
            model.metric_derivative_step = opt.binary_metric_derivative_step;
            model.capture_factor = opt.binary_capture_factor;
            model.tidal_fraction = opt.binary_tidal_fraction;
            model.taper_start_fraction = opt.binary_taper_start_fraction;
            model.density_scale1 = opt.binary_density_scale1;
            model.density_scale2 = opt.binary_density_scale2;
            model.temperature_scale1 = opt.binary_temperature_scale1;
            model.temperature_scale2 = opt.binary_temperature_scale2;
            model.field_polarity1 = opt.binary_field_polarity1;
            model.field_polarity2 = opt.binary_field_polarity2;
            configure_tabulated_binary_trajectory(opt, model);

            const auto metric = model.make_metric();
            const auto orbit_state = metric.orbit.state(metric.time_origin);
            if (!orbit_state.valid || !orbit_state.inspiral_valid) {
                throw std::runtime_error(
                    "binary observation event is outside the available inspiral trajectory domain");
            }
            if (orbit_state.merger_weight > Real(0)) {
                throw std::runtime_error(
                    "the double analytic mini-RIAF source is restricted to W=0; select a pre-transition observation time (the SKS metric and trajectory do support merger/remnant states)");
            }
            opt.inner_radius = model.capture_factor;
            if (opt.outer_radius <= Real(0)) {
                const Real r1 = orbit_state.mass1 * model.outer_radius_hat(
                    orbit_state.mass1, orbit_state.mass2,
                    orbit_state.separation);
                const Real r2 = orbit_state.mass2 * model.outer_radius_hat(
                    orbit_state.mass2, orbit_state.mass1,
                    orbit_state.separation);
                const Real extent1 = kpolaris::norm(orbit_state.position1) + r1;
                const Real extent2 = kpolaris::norm(orbit_state.position2) + r2;
                opt.outer_radius = Real(1.25) * std::max(extent1, extent2);
            }
            model.validate_metric_domain(opt.outer_radius);
            run_model(opt, model);
        }
#endif
#if KPOLARIS_ENABLE_MODEL_TORUS
        if (!handled_model && opt.model == "torus") {
            handled_model = true;
            kpolaris::MagnetizedTorusRadiationModel<Real> model;
            model.spin = opt.spin;
            model.freq_cgs = opt.freq;
            model.l_lambda = opt.torus_l_lambda;
            model.wwin = opt.torus_wwin;
            model.kappa = opt.torus_kappa;
            model.omegac = opt.torus_omegac;
            model.betac = opt.torus_betac;
            model.beta = opt.torus_beta;
            model.Rhigh = opt.torus_Rhigh;
            model.bh_mass_solar = opt.torus_bh_mass_solar;
            model.accretion_rate_cgs = opt.torus_mdot_cgs;
            model.accretion_rate_code = opt.torus_mdot_code;
            model.thetae_min = opt.torus_thetae_min;
            model.scalar_transport = opt.scalar_transport;
            model.initialize_default_torus();
            if (opt.outer_radius <= Real(0)) {
                opt.outer_radius = model.r_outer * Real(1.15);
            }
            run_model(opt, model);
        }
#endif
#if KPOLARIS_ENABLE_MODEL_IHARM
        if (!handled_model && opt.model == "iharm") {
            handled_model = true;
            if (opt.coordinate != "fmks" && opt.coordinate != "mks" && opt.coordinate != "spherical_ks" && opt.coordinate != "cartesian_ks") {
                throw std::runtime_error("iharm currently supports --coordinate=mks, fmks, spherical_ks, or cartesian_ks");
            }
            if (opt.iharm_dump.empty() && (opt.slow_light || opt.slow_light_time_probe)) {
                opt.iharm_dump = build_slow_light_dump_paths(opt).front();
            }
            kpolaris::IHARMLoadOptions load_opt;
            load_opt.dump_path = opt.iharm_dump;
            load_opt.freq = opt.freq;
            load_opt.M_unit = opt.iharm_M_unit;
            load_opt.mbh_solar = opt.iharm_mbh_solar;
            load_opt.trat_small = opt.iharm_trat_small;
            load_opt.trat_large = opt.iharm_trat_large;
            load_opt.beta_crit = opt.iharm_beta_crit;
            load_opt.sigma_cut = opt.iharm_sigma_cut;
            load_opt.sigma_cut_high = opt.iharm_sigma_cut_high;
            load_opt.emission_type = effective_emission_type(opt.emission_type,
                                                              KPOLARIS_IHARM_COMPILED_EMISSION_TYPE,
                                                              4);
            load_opt.nonthermal_kappa = opt.nonthermal_kappa;
            load_opt.variable_kappa = opt.variable_kappa;
            load_opt.variable_kappa_min = opt.variable_kappa_min;
            load_opt.variable_kappa_interp_start = opt.variable_kappa_interp_start;
            load_opt.variable_kappa_max = opt.variable_kappa_max;
            load_opt.powerlaw_p = opt.powerlaw_p;
            load_opt.powerlaw_eta = opt.powerlaw_eta;
            load_opt.powerlaw_gamma_min = opt.powerlaw_gamma_min;
            load_opt.powerlaw_gamma_max = opt.powerlaw_gamma_max;
            load_opt.powerlaw_gamma_cutoff = opt.powerlaw_gamma_cutoff;
            load_opt.interpolate_derived_scalars = opt.iharm_interpolate_derived_scalars;
            load_opt.resample_spherical_ks_precomputed = opt.iharm_resample_spherical_ks_precomputed;
            load_opt.resample_spherical_ks_primitives = opt.iharm_resample_spherical_ks_primitives;
            load_opt.resample_n1 = opt.iharm_resample_n1;
            load_opt.resample_n2 = opt.iharm_resample_n2;
            load_opt.resample_n3 = opt.iharm_resample_n3;
            load_opt.resample_r_in = opt.iharm_resample_r_in;
            load_opt.resample_r_out = opt.iharm_resample_r_out;
            load_opt.timing = opt.timing;
            if (load_opt.resample_spherical_ks_precomputed ||
                load_opt.resample_spherical_ks_primitives) {
                opt.coordinate = "spherical_ks";
            }
            kpolaris::GRMHDRadiationModel<Real> model = kpolaris::load_iharm_model_from_hdf5(load_opt);
            opt.spin = model.spin;
            opt.fmks_startx1 = model.startx1;
            opt.fmks_hslope = model.hslope;
            opt.fmks_mks_smooth = model.mks_smooth;
            opt.fmks_poly_alpha = model.poly_alpha;
            opt.fmks_poly_xt = model.poly_xt;
            opt.fmks_poly_norm = model.poly_norm;
            if (opt.outer_radius <= Real(0)) {
                opt.outer_radius = opt.camera == "pinhole" ?
                    std::min(model.r_out, opt.radius * Real(0.9)) : model.r_out;
            }
            opt.inner_radius = safe_inner_radius(opt.inner_radius, opt.inner_radius_explicit,
                                                model.r_in, model.spin);
            run_model(opt, model);
        }
#endif
#if KPOLARIS_ENABLE_MODEL_KHARMA
        if (!handled_model && opt.model == "kharma") {
            handled_model = true;
            if (opt.coordinate != "fmks" && opt.coordinate != "mks" && opt.coordinate != "spherical_ks" && opt.coordinate != "cartesian_ks") {
                throw std::runtime_error("kharma currently supports --coordinate=mks, fmks, spherical_ks, or cartesian_ks");
            }
            if (opt.kharma_dump.empty() && (opt.slow_light || opt.slow_light_time_probe)) {
                opt.kharma_dump = build_slow_light_dump_paths(opt).front();
            }
            kpolaris::KHARMALoadOptions load_opt;
            load_opt.ddc = opt.kharma_ddc;
            load_opt.dump_path = opt.kharma_dump;
            load_opt.freq = opt.freq;
            load_opt.M_unit = opt.kharma_M_unit;
            load_opt.mbh_solar = opt.kharma_mbh_solar;
            load_opt.trat_small = opt.kharma_trat_small;
            load_opt.trat_large = opt.kharma_trat_large;
            load_opt.beta_crit = opt.kharma_beta_crit;
            load_opt.sigma_cut = opt.kharma_sigma_cut;
            load_opt.sigma_cut_high = opt.kharma_sigma_cut_high;
            load_opt.emission_type = effective_emission_type(opt.emission_type,
                                                              KPOLARIS_IHARM_COMPILED_EMISSION_TYPE,
                                                              4);
            load_opt.nonthermal_kappa = opt.nonthermal_kappa;
            load_opt.variable_kappa = opt.variable_kappa;
            load_opt.variable_kappa_min = opt.variable_kappa_min;
            load_opt.variable_kappa_interp_start = opt.variable_kappa_interp_start;
            load_opt.variable_kappa_max = opt.variable_kappa_max;
            load_opt.powerlaw_p = opt.powerlaw_p;
            load_opt.powerlaw_eta = opt.powerlaw_eta;
            load_opt.powerlaw_gamma_min = opt.powerlaw_gamma_min;
            load_opt.powerlaw_gamma_max = opt.powerlaw_gamma_max;
            load_opt.powerlaw_gamma_cutoff = opt.powerlaw_gamma_cutoff;
            load_opt.interpolate_derived_scalars = opt.kharma_interpolate_derived_scalars;
            load_opt.resample_spherical_ks_precomputed = opt.kharma_resample_spherical_ks_precomputed;
            load_opt.resample_spherical_ks_primitives = opt.kharma_resample_spherical_ks_primitives;
            load_opt.resample_n1 = opt.kharma_resample_n1;
            load_opt.resample_n2 = opt.kharma_resample_n2;
            load_opt.resample_n3 = opt.kharma_resample_n3;
            load_opt.resample_r_in = opt.kharma_resample_r_in;
            load_opt.resample_r_out = opt.kharma_resample_r_out;
            load_opt.reverse_field = opt.kharma_reverse_field;
            load_opt.timing = opt.timing;
            if (load_opt.resample_spherical_ks_precomputed ||
                load_opt.resample_spherical_ks_primitives) {
                opt.coordinate = "spherical_ks";
            }
            kpolaris::GRMHDRadiationModel<Real> model = kpolaris::load_kharma_model_from_phdf(load_opt);
            opt.spin = model.spin;
            if (model.data_coordinate_system == static_cast<int>(kpolaris::CoordinateSystem::MKS) && opt.coordinate == "fmks") {
                throw std::runtime_error("KHARMA dump declares transform=mks; use --coordinate=mks, spherical_ks, or cartesian_ks, not fmks");
            }
            opt.fmks_startx1 = model.startx1;
            opt.fmks_hslope = model.hslope;
            opt.fmks_mks_smooth = model.mks_smooth;
            opt.fmks_poly_alpha = model.poly_alpha;
            opt.fmks_poly_xt = model.poly_xt;
            opt.fmks_poly_norm = model.poly_norm;
            if (opt.outer_radius <= Real(0)) {
                opt.outer_radius = opt.camera == "pinhole" ?
                    std::min(model.r_out, opt.radius * Real(0.9)) : model.r_out;
            }
            opt.inner_radius = safe_inner_radius(opt.inner_radius, opt.inner_radius_explicit,
                                                model.r_in, model.spin);
            if (opt.slow_light_snapshot_cache_gib > Real(0)) {
                // Budget the live snapshot payload, reserving one additional
                // slot for the initial metadata model retained by run_model.
                const size_t snapshot_bytes = model.prims.span() * sizeof(Real) +
                    model.derived_scalars.span() * sizeof(Real) +
                    model.fluid_states.span() * sizeof(typename decltype(model)::FluidStateStorageReal);
                const long double budget = static_cast<long double>(opt.slow_light_snapshot_cache_gib) * 1073741824.L;
                const int banks = opt.slow_light_pipeline && opt.slow_light_prefetch &&
                    !opt.kharma_resample_spherical_ks_precomputed && !opt.kharma_resample_spherical_ks_primitives ? 2 : 1;
                const int retained = (banks == 2 ? 3 : 2) +
                    (opt.slow_light_step_mode == kpolaris::SlowLightStepMode::decoupled ?
                        kpolaris::decoupled_cache_halo_snapshots : 0);
                if (!snapshot_bytes || budget / snapshot_bytes < banks + retained)
                    throw std::invalid_argument("slow_light_snapshot_cache_gib cannot fit the snapshot banks and retained initial models");
                const int windows = static_cast<int>(std::min<long double>(
                    std::floor((std::floor(budget / snapshot_bytes) - retained) / banks), std::numeric_limits<int>::max()));
                opt.slow_light_windows_per_block = opt.slow_light_windows_per_block_explicit ?
                    std::min(opt.slow_light_windows_per_block, windows) : windows;
                if (opt.timing) std::cout << "slow_light_cache snapshot_bytes " << snapshot_bytes
                    << " budget_gib " << opt.slow_light_snapshot_cache_gib
                    << " windows_per_block " << opt.slow_light_windows_per_block << "\n";
            }
            run_model(opt, model);
        }
#endif
#if KPOLARIS_ENABLE_MODEL_ATHENAK
        if (!handled_model && opt.model == "athenak") {
            handled_model = true;
            if (opt.athenak_dump.empty() && (opt.slow_light || opt.slow_light_time_probe)) {
                opt.athenak_dump = build_slow_light_dump_paths(opt).front();
            }
            kpolaris::AthenaKLoadOptions load_opt;
            load_opt.dump_path = opt.athenak_dump;
            load_opt.freq = opt.freq;
            load_opt.M_unit = opt.athenak_M_unit;
            load_opt.mbh_solar = opt.athenak_mbh_solar;
            load_opt.trat_small = opt.athenak_trat_small;
            load_opt.trat_large = opt.athenak_trat_large;
            load_opt.beta_crit = opt.athenak_beta_crit;
            load_opt.gamma = opt.athenak_gamma;
            load_opt.sigma_cut = opt.athenak_sigma_cut;
            load_opt.sigma_cut_high = opt.athenak_sigma_cut_high;
            load_opt.emission_type = effective_emission_type(opt.emission_type,
                                                              KPOLARIS_IHARM_COMPILED_EMISSION_TYPE,
                                                              4);
            load_opt.profile_mode = opt.athenak_profile_mode;
            load_opt.nonthermal_kappa = opt.nonthermal_kappa;
            load_opt.variable_kappa = opt.variable_kappa;
            load_opt.variable_kappa_min = opt.variable_kappa_min;
            load_opt.variable_kappa_interp_start = opt.variable_kappa_interp_start;
            load_opt.variable_kappa_max = opt.variable_kappa_max;
            load_opt.powerlaw_p = opt.powerlaw_p;
            load_opt.powerlaw_eta = opt.powerlaw_eta;
            load_opt.powerlaw_gamma_min = opt.powerlaw_gamma_min;
            load_opt.powerlaw_gamma_max = opt.powerlaw_gamma_max;
            load_opt.powerlaw_gamma_cutoff = opt.powerlaw_gamma_cutoff;
            load_opt.resample_r_in = opt.athenak_r_in;
            load_opt.resample_r_out = opt.athenak_r_out;
            load_opt.timing = opt.timing;
            opt.coordinate = "cartesian_ks";
            kpolaris::AthenaKDirectRadiationModel<Real> model = kpolaris::load_athenak_direct_model_from_binary(load_opt);
            opt.spin = model.spin;
            if (opt.outer_radius <= Real(0)) {
                opt.outer_radius = opt.camera == "pinhole" ?
                    std::min(model.r_out, opt.radius * Real(0.9)) : model.r_out;
            }
            // The model's fluid validity radius can be larger than the intended
            // radiating domain. Keep the dump bound, but honor the run-level
            // outer_radius so floor/background material outside rmax is not emitted.
            model.r_out = std::min(model.r_out, opt.outer_radius);
            opt.inner_radius = safe_inner_radius(opt.inner_radius, opt.inner_radius_explicit,
                                                model.r_in, model.spin);
            run_model(opt, model);
        }
#endif
#if KPOLARIS_ENABLE_MODEL_BHAC
        if (!handled_model && opt.model == "bhac") {
            handled_model = true;
            if (opt.coordinate != "cartesian_ks" && opt.coordinate != "spherical_ks") {
                throw std::runtime_error("bhac currently supports --coordinate=cartesian_ks or spherical_ks");
            }
            if (opt.bhac_dump.empty() && (opt.slow_light || opt.slow_light_time_probe)) {
                opt.bhac_dump = build_slow_light_dump_paths(opt).front();
            }
            kpolaris::BHACLoadOptions load_opt;
            load_opt.dump_path = opt.bhac_dump;
            load_opt.freq = opt.freq;
            load_opt.M_unit = opt.bhac_M_unit;
            load_opt.mbh_solar = opt.bhac_mbh_solar;
            load_opt.trat_small = opt.bhac_trat_small;
            load_opt.trat_large = opt.bhac_trat_large;
            load_opt.beta_crit = opt.bhac_beta_crit;
            load_opt.gamma = opt.bhac_gamma;
            load_opt.sigma_cut = opt.bhac_sigma_cut;
            load_opt.sigma_cut_high = opt.bhac_sigma_cut_high;
            load_opt.emission_type = effective_emission_type(opt.emission_type,
                                                              KPOLARIS_IHARM_COMPILED_EMISSION_TYPE,
                                                              4);
            load_opt.profile_mode = opt.bhac_profile_mode;
            load_opt.nonthermal_kappa = opt.nonthermal_kappa;
            load_opt.variable_kappa = opt.variable_kappa;
            load_opt.variable_kappa_min = opt.variable_kappa_min;
            load_opt.variable_kappa_interp_start = opt.variable_kappa_interp_start;
            load_opt.variable_kappa_max = opt.variable_kappa_max;
            load_opt.powerlaw_p = opt.powerlaw_p;
            load_opt.powerlaw_eta = opt.powerlaw_eta;
            load_opt.powerlaw_gamma_min = opt.powerlaw_gamma_min;
            load_opt.powerlaw_gamma_max = opt.powerlaw_gamma_max;
            load_opt.powerlaw_gamma_cutoff = opt.powerlaw_gamma_cutoff;
            load_opt.nxlone1 = opt.bhac_nxlone1;
            load_opt.nxlone2 = opt.bhac_nxlone2;
            load_opt.nxlone3 = opt.bhac_nxlone3;
            load_opt.spin_index = opt.bhac_spin_index;
            load_opt.x1_min = opt.bhac_x1_min;
            load_opt.x1_max = opt.bhac_x1_max;
            load_opt.x2_min = opt.bhac_x2_min;
            load_opt.x2_max = opt.bhac_x2_max;
            load_opt.x3_min = opt.bhac_x3_min;
            load_opt.x3_max = opt.bhac_x3_max;
            load_opt.hslope = opt.bhac_hslope;
            load_opt.r_in = opt.bhac_r_in;
            load_opt.r_out = opt.bhac_r_out;
            load_opt.sfc = opt.bhac_sfc;
            load_opt.reverse_field = opt.bhac_reverse_field;
            load_opt.cache_path = opt.bhac_cache;
            load_opt.cache_mode = opt.bhac_cache.empty() ? kpolaris::BHACCacheOff : opt.bhac_cache_mode;
            load_opt.timing = opt.timing;
            kpolaris::BHACAMRRadiationModel<Real> model = kpolaris::load_bhac_model_from_dat(load_opt);
            opt.spin = model.spin;
            if (opt.outer_radius <= Real(0)) {
                opt.outer_radius = opt.camera == "pinhole" ?
                    std::min(model.r_out, opt.radius * Real(0.9)) : model.r_out;
            }
            model.r_out = std::min(model.r_out, opt.outer_radius);
            opt.inner_radius = safe_inner_radius(opt.inner_radius, opt.inner_radius_explicit,
                                                model.r_in, model.spin);
            run_model(opt, model);
        }
#endif
#if KPOLARIS_ENABLE_MODEL_HAMR
        if (!handled_model && opt.model == "hamr") {
            handled_model = true;
            if (opt.coordinate != "cartesian_ks" && opt.coordinate != "spherical_ks") {
                throw std::runtime_error("hamr currently supports --coordinate=cartesian_ks or spherical_ks");
            }
#if KPOLARIS_ENABLE_SLOW_LIGHT
            if ((opt.slow_light || opt.slow_light_time_probe) && opt.hamr_dump.empty()) {
                opt.hamr_dump = build_slow_light_dump_paths(opt).front();
            }
#endif
            warn_pending_hamr_hslope(opt);
            kpolaris::HAMRLoadOptions load_opt;
            load_opt.dump_path = opt.hamr_dump;
            load_opt.freq = opt.freq;
            load_opt.M_unit = opt.hamr_M_unit;
            load_opt.mbh_solar = opt.hamr_mbh_solar;
            load_opt.trat_small = opt.hamr_trat_small;
            load_opt.trat_large = opt.hamr_trat_large;
            load_opt.beta_crit = opt.hamr_beta_crit;
            load_opt.gamma = opt.hamr_gamma;
            load_opt.sigma_cut = opt.hamr_sigma_cut;
            load_opt.sigma_cut_high = opt.hamr_sigma_cut_high;
            load_opt.emission_type = effective_emission_type(opt.emission_type,
                                                             KPOLARIS_IHARM_COMPILED_EMISSION_TYPE,
                                                             4);
            load_opt.profile_mode = opt.hamr_profile_mode;
            load_opt.nonthermal_kappa = opt.nonthermal_kappa;
            load_opt.variable_kappa = opt.variable_kappa;
            load_opt.variable_kappa_min = opt.variable_kappa_min;
            load_opt.variable_kappa_interp_start = opt.variable_kappa_interp_start;
            load_opt.variable_kappa_max = opt.variable_kappa_max;
            load_opt.powerlaw_p = opt.powerlaw_p;
            load_opt.powerlaw_eta = opt.powerlaw_eta;
            load_opt.powerlaw_gamma_min = opt.powerlaw_gamma_min;
            load_opt.powerlaw_gamma_max = opt.powerlaw_gamma_max;
            load_opt.powerlaw_gamma_cutoff = opt.powerlaw_gamma_cutoff;
            load_opt.r_in = opt.hamr_r_in;
            load_opt.r_out = opt.hamr_r_out;
            load_opt.hslope = opt.hamr_hslope;
            load_opt.reverse_field = opt.hamr_reverse_field;
            load_opt.id_order = opt.hamr_id_order;
            load_opt.root_order = opt.hamr_root_order;
            load_opt.timing = opt.timing;
            kpolaris::HAMRRadiationModel<Real> model = kpolaris::load_hamr_model_from_dump(load_opt);
            opt.spin = model.spin;
            if (opt.outer_radius <= Real(0)) {
                opt.outer_radius = opt.camera == "pinhole" ?
                    std::min(model.r_out, opt.radius * Real(0.9)) : model.r_out;
            }
            model.r_out = std::min(model.r_out, opt.outer_radius);
            opt.inner_radius = safe_inner_radius(opt.inner_radius, opt.inner_radius_explicit,
                                                model.r_in, model.spin);
            run_model(opt, model);
        }
#endif
        if (!handled_model) {
            throw std::runtime_error("model not enabled in this executable: " + opt.model);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        Kokkos::finalize();
        return 1;
    } catch (...) {
        std::cerr << "error: unknown exception\n";
        Kokkos::finalize();
        return 1;
    }
    Kokkos::finalize();
    return 0;
}
