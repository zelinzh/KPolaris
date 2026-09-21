#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <H5Cpp.h>
#include <Kokkos_Core.hpp>

#include "diagnostics/trace_moments.hpp"
#include "KPolaris.hpp"
#include "common/version.hpp"
#include "kpolaris_conventions_output.hpp"
#include "grmhd/athenak_loader.hpp"
#include "grmhd/bhac_loader.hpp"
#include "grmhd/hamr_loader.hpp"
#include "grmhd/kharma_loader.hpp"
#include "kernels/trace_kernel.hpp"
#include "model/time_interpolation.hpp"

#ifndef KPOLARIS_ENABLE_SLOW_LIGHT
#define KPOLARIS_ENABLE_SLOW_LIGHT 1
#endif
#if KPOLARIS_ENABLE_SLOW_LIGHT
#include "image/slow_light_driver.hpp"
#endif

#ifndef KPOLARIS_TRACE_ONLY_ATHENAK
#define KPOLARIS_TRACE_ONLY_ATHENAK 0
#endif
#ifndef KPOLARIS_TRACE_ONLY_RIAF
#define KPOLARIS_TRACE_ONLY_RIAF 0
#endif
#ifndef KPOLARIS_TRACE_ONLY_TORUS
#define KPOLARIS_TRACE_ONLY_TORUS 0
#endif
#ifndef KPOLARIS_TRACE_ONLY_IHARM
#define KPOLARIS_TRACE_ONLY_IHARM 0
#endif
#ifndef KPOLARIS_TRACE_ONLY_KHARMA
#define KPOLARIS_TRACE_ONLY_KHARMA 0
#endif
#ifndef KPOLARIS_TRACE_ONLY_BHAC
#define KPOLARIS_TRACE_ONLY_BHAC 0
#endif
#ifndef KPOLARIS_TRACE_ONLY_HAMR
#define KPOLARIS_TRACE_ONLY_HAMR 0
#endif

namespace {
using Real = kpolaris::DefaultReal;

[[maybe_unused]] Real safe_inner_radius(
    Real requested, bool explicit_request, Real model_r_in, Real spin) {
    const Real horizon = Real(1) + std::sqrt(std::max<Real>(Real(0), Real(1) - spin * spin));
    const Real floor_radius = std::max(model_r_in, horizon * Real(1.05));
    if (explicit_request && requested > Real(0)) {
        return std::max(requested, horizon * Real(1.05));
    }
    return floor_radius;
}

bool model_defines_inner_radius(const std::string& model) {
    return model == "iharm" || model == "kharma" || model == "athenak" ||
           model == "bhac" || model == "hamr";
}

struct Options {
    std::string model = "riaf";
    std::string evpa_0 = "N";
    std::string output = "kpolaris_trace.h5";
    std::string parameter_file;
    std::string parameter_output = "auto";
    std::string camera = "pinhole";
    std::string coordinate = "cartesian_ks";
    std::string trace_mode = "single";
    std::string trace_fields = "coords,coeffs,stokes";
    std::string trace_precision = "float";
    std::string trace_layout = "ragged";
    int trace_compression = 4;
    int ix = 0;
    int iy = 0;
    int nx = 96;
    int ny = 96;
    int max_trace_samples = 2048;
    int trace_stride = 1;
    int timing = 0;
    Real radius = Real(35);
    Real inclination = Real(1.04719755119659774615);
    Real fov = Real(0.08);
    Real fovy = Real(-1);
    Real xspan = Real(-1);
    Real yspan = Real(-1);
    Real dsource_pc = Real(-1);
    Real fovx_dsource = Real(-1);
    Real fovy_dsource = Real(-1);
    Real image_width_x = Real(-1);
    Real image_width_y = Real(-1);
    Real effective_fovx_dsource = Real(-1);
    Real effective_fovy_dsource = Real(-1);
    Real x_offset = Real(0);
    Real y_offset = Real(0);
    int use_pinhole_pixel_bias = 0;
    Real pinhole_pixel_bias = Real(-0.01);
    Real spin = Real(0.9375);
    Real inner_radius = Real(-1);
    bool inner_radius_explicit = false;
    Real outer_radius = Real(-1);
    Real step = Real(0.025);
    int max_steps = 4096;
    int adaptive = 1;
    Real adaptive_tolerance = Real(1e-7);
    Real min_step = Real(1e-5);
    Real max_step = Real(1);
    Real max_radiation_step = Real(1);
    Real max_radiation_depth = Real(1);
    Real max_absorption_depth = Real(1);
    Real max_faraday_depth = Real(4);
    Real fmks_startx1 = Real(0);
    Real fmks_hslope = Real(0.3);
    Real fmks_mks_smooth = Real(0.5);
    Real fmks_poly_alpha = Real(14);
    Real fmks_poly_xt = Real(0.82);
    Real fmks_poly_norm = Real(1);
    Real freq = Real(230.0e9);
    std::string freq_list;
    Real freq_min = Real(-1);
    Real freq_max = Real(-1);
    int nfreq = 0;
    std::string freq_spacing = "log";

    int slow_light = 0;
    kpolaris::SlowLightInterpolation slow_light_interpolation = kpolaris::default_slow_light_interpolation;
    Real slow_light_observation_time = Real(0);
    std::string slow_light_dump_list;
    std::string slow_light_time_list;
    std::string slow_light_dump_pattern;
    int slow_light_dump_start = 0;
    int slow_light_dump_end = -1;
    int slow_light_dump_stride = 1;

    Real riaf_r_min = Real(1);
    Real riaf_r_max = Real(100);
    Real riaf_nth0 = Real(1);
    Real riaf_Te0 = Real(1);
    Real riaf_disk_h = Real(0.35);
    Real riaf_pow_nth = Real(-1.1);
    Real riaf_pow_T = Real(-0.84);
    Real riaf_ne_unit = Real(5.0e6);
    Real riaf_te_unit = Real(1.0e11);
    Real riaf_mbh_solar = Real(4.3e6);
    Real riaf_keplerian_factor = Real(1);
    Real riaf_infall_factor = Real(0);
    int emission_type = 0;
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

    Real torus_l_lambda = Real(0.78);
    Real torus_wwin = Real(1);
    Real torus_kappa = Real(4) / Real(3);
    Real torus_omegac = Real(1);
    Real torus_betac = Real(10);
    Real torus_beta = Real(10);
    Real torus_Rhigh = Real(1);
    Real torus_bh_mass_solar = Real(4.0e6);
    Real torus_mdot_cgs = Real(1.57e15);
    Real torus_mdot_code = Real(3.0e-3);
    Real torus_thetae_min = Real(5.0e-2);

    std::string iharm_dump;
    Real iharm_M_unit = Real(3.0e25);
    Real iharm_mbh_solar = Real(6.2e9);
    Real iharm_trat_small = Real(1);
    Real iharm_trat_large = Real(20);
    Real iharm_beta_crit = Real(1);
    Real iharm_sigma_cut = Real(1);
    Real iharm_sigma_cut_high = Real(-1);
    int iharm_interpolate_derived_scalars = 1;

    std::string kharma_dump;
    kpolaris::DDCInputOptions kharma_ddc = kpolaris::DDCInputOptions::from_environment();
    Real kharma_M_unit = Real(3.0e25);
    Real kharma_mbh_solar = Real(6.2e9);
    Real kharma_trat_small = Real(1);
    Real kharma_trat_large = Real(20);
    Real kharma_beta_crit = Real(1);
    Real kharma_sigma_cut = Real(1);
    Real kharma_sigma_cut_high = Real(-1);
    int kharma_interpolate_derived_scalars = 1;
    int kharma_reverse_field = 0;

    std::string athenak_dump;
    Real athenak_M_unit = Real(1.0e26);
    Real athenak_mbh_solar = Real(6.2e9);
    Real athenak_trat_small = Real(1);
    Real athenak_trat_large = Real(40);
    Real athenak_beta_crit = Real(1);
    Real athenak_gamma = Real(-1);
    Real athenak_sigma_cut = Real(1);
    Real athenak_sigma_cut_high = Real(-1);
    Real athenak_r_in = Real(-1);
    Real athenak_r_out = Real(1000);

    std::string bhac_dump;
    Real bhac_M_unit = Real(1.0e18);
    Real bhac_mbh_solar = Real(4.14e6);
    Real bhac_trat_small = Real(1);
    Real bhac_trat_large = Real(40);
    Real bhac_beta_crit = Real(1);
    Real bhac_gamma = Real(-1);
    Real bhac_sigma_cut = Real(1);
    Real bhac_sigma_cut_high = Real(-1);
    Real bhac_r_in = Real(-1);
    Real bhac_r_out = Real(-1);
    Real bhac_hslope = Real(0.25);
    int bhac_nxlone1 = 0;
    int bhac_nxlone2 = 0;
    int bhac_nxlone3 = 0;
    int bhac_spin_index = -1;
    Real bhac_x1_min = Real(0.17);
    Real bhac_x1_max = Real(8.1117280833);
    Real bhac_x2_min = Real(0);
    Real bhac_x2_max = Real(3.141592653589793238462643383279502884);
    Real bhac_x3_min = Real(0);
    Real bhac_x3_max = Real(6.283185307179586476925286766559005768);
    int bhac_sfc = 1;
    int bhac_reverse_field = 0;
    int bhac_profile_mode = 0;

    std::string hamr_dump;
    Real hamr_M_unit = Real(1.0e26);
    Real hamr_mbh_solar = Real(6.2e9);
    Real hamr_trat_small = Real(1);
    Real hamr_trat_large = Real(40);
    Real hamr_beta_crit = Real(1);
    Real hamr_gamma = Real(-1);
    Real hamr_sigma_cut = Real(1);
    Real hamr_sigma_cut_high = Real(-1);
    Real hamr_r_in = Real(-1);
    Real hamr_r_out = Real(-1);
    Real hamr_hslope = Real(0.3);
    bool hamr_hslope_explicit = false;
    int hamr_reverse_field = 0;
    int hamr_profile_mode = 0;
    std::string hamr_id_order = "root_slot";
    std::string hamr_root_order = "morton";

    int scalar_transport = 0;
};

const char* hamr_hslope_source(const Options& opt) {
    return opt.hamr_hslope_explicit ? "explicit" : "default_pending_dump_metadata";
}

const char* emission_fit_name(int emission_type) {
    switch (emission_type) {
    case 1: return "symphony_pandya_thermal";
    case 2: return "symphony_kappa";
    case 3: return "symphony_power_law";
    case 4: return "dexter_thermal";
    default: return "unspecified";
    }
}

int effective_trace_emission_type(const Options& opt) {
    if (opt.emission_type > 0) return opt.emission_type;
    return opt.model == "riaf" ? 1 : 4;
}

[[maybe_unused]] void warn_pending_hamr_hslope(const Options& opt) {
    if (!opt.hamr_hslope_explicit) {
        std::cerr << "warning: hamr_hslope=" << opt.hamr_hslope
                  << " is a temporary default; no confirmed H-AMR hslope metadata field "
                     "has been identified in the dump yet. Use --hamr_hslope to override.\n";
    }
}

std::string trim_copy(const std::string& input) {
    size_t first = 0;
    while (first < input.size() && std::isspace(static_cast<unsigned char>(input[first]))) ++first;
    size_t last = input.size();
    while (last > first && std::isspace(static_cast<unsigned char>(input[last - 1]))) --last;
    return input.substr(first, last - first);
}

std::string lowercase_copy(std::string value) {
    for (char& c : value) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return value;
}

std::string normalize_key(std::string key) {
    key = trim_copy(key);
    while (key.rfind("--", 0) == 0) key.erase(0, 2);
    for (char& c : key) if (c == '-') c = '_';
    return key;
}

Real length_unit_cgs_from_mbh_solar(Real mbh_solar) {
    const Real gnewt = Real(6.6742e-8);
    const Real msun = Real(1.989e33);
    const Real cl = Real(2.99792458e10);
    return gnewt * mbh_solar * msun / (cl * cl);
}

Real camera_mbh_solar(const Options& opt) {
    if (opt.model == "iharm") return opt.iharm_mbh_solar;
    if (opt.model == "kharma") return opt.kharma_mbh_solar;
    if (opt.model == "athenak") return opt.athenak_mbh_solar;
    if (opt.model == "bhac") return opt.bhac_mbh_solar;
    if (opt.model == "hamr") return opt.hamr_mbh_solar;
    if (opt.model == "torus") return opt.torus_bh_mass_solar;
    return opt.riaf_mbh_solar;
}

void resolve_camera_extents(Options& opt) {
    const Real pc_cgs = Real(3.085678e18);
    const Real muas_per_rad = Real(2.06265e11);
    const bool source_fov = opt.fovx_dsource > Real(0) || opt.fovy_dsource > Real(0);

    if (source_fov) {
        if (opt.dsource_pc <= Real(0)) {
            throw std::runtime_error("fovx_dsource/fovy_dsource require dsource in parsec");
        }
        if (opt.fovx_dsource <= Real(0) && opt.fovy_dsource > Real(0)) opt.fovx_dsource = opt.fovy_dsource;
        if (opt.fovy_dsource <= Real(0) && opt.fovx_dsource > Real(0)) opt.fovy_dsource = opt.fovx_dsource;
        const Real l_unit = length_unit_cgs_from_mbh_solar(camera_mbh_solar(opt));
        const Real fov_to_m = opt.dsource_pc * pc_cgs / l_unit / muas_per_rad;
        opt.image_width_x = opt.fovx_dsource * fov_to_m;
        opt.image_width_y = opt.fovy_dsource * fov_to_m;
        opt.effective_fovx_dsource = opt.fovx_dsource;
        opt.effective_fovy_dsource = opt.fovy_dsource;
        opt.fov = opt.image_width_x / opt.radius;
        opt.fovy = opt.image_width_y / opt.radius;
        if (opt.camera == "parallel_plane") {
            opt.xspan = opt.image_width_x / Real(2);
            opt.yspan = opt.image_width_y / Real(2);
        } else {
            opt.xspan = Real(-1);
            opt.yspan = Real(-1);
        }
        return;
    }

    if (opt.camera != "parallel_plane" && opt.xspan > Real(0)) {
        opt.fov = Real(2) * opt.xspan / opt.radius;
        if (opt.yspan > Real(0)) opt.fovy = Real(2) * opt.yspan / opt.radius;
    }
    if (opt.fovy <= Real(0)) opt.fovy = opt.fov;

    if (opt.camera == "parallel_plane") {
        if (opt.xspan > Real(0) && opt.yspan <= Real(0)) opt.yspan = opt.xspan * Real(opt.ny) / Real(opt.nx);
        opt.image_width_x = opt.xspan > Real(0) ? Real(2) * opt.xspan : opt.fov;
        opt.image_width_y = opt.yspan > Real(0) ? Real(2) * opt.yspan : opt.fovy;
    } else {
        opt.image_width_x = opt.fov * opt.radius;
        opt.image_width_y = opt.fovy * opt.radius;
    }

    if (opt.dsource_pc > Real(0) && opt.image_width_x > Real(0) && opt.image_width_y > Real(0)) {
        const Real l_unit = length_unit_cgs_from_mbh_solar(camera_mbh_solar(opt));
        const Real fov_to_m = opt.dsource_pc * pc_cgs / l_unit / muas_per_rad;
        opt.effective_fovx_dsource = opt.image_width_x / fov_to_m;
        opt.effective_fovy_dsource = opt.image_width_y / fov_to_m;
    }
}

void validate_options(const Options& opt) {
    if (opt.model == "kharma") opt.kharma_ddc.validate();
    const auto finite = [](Real value) {
        return std::isfinite(static_cast<double>(value));
    };
    const auto require_positive = [&](Real value, const char* name) {
        if (!finite(value) || !(value > Real(0))) {
            throw std::runtime_error(std::string(name) + " must be finite and positive");
        }
    };

    if (opt.nx <= 0 || opt.ny <= 0) {
        throw std::runtime_error("nx and ny must be positive");
    }
    if (opt.nx > std::numeric_limits<int>::max() / opt.ny) {
        throw std::runtime_error("nx*ny exceeds the supported integer pixel count");
    }
    require_positive(opt.radius, "radius");
    if (!finite(opt.inclination)) {
        throw std::runtime_error("inclination must be finite");
    }
    if (!finite(opt.spin) || std::abs(opt.spin) > Real(1)) {
        throw std::runtime_error("spin must be finite and lie in [-1, 1]");
    }
    require_positive(opt.step, "step");
    if (opt.max_steps <= 0) {
        throw std::runtime_error("max_steps must be positive");
    }
    if (opt.adaptive != 0 && opt.adaptive != 1) {
        throw std::runtime_error("adaptive must be 0 or 1");
    }
    require_positive(opt.adaptive_tolerance, "adaptive_tolerance");
    require_positive(opt.min_step, "min_step");
    require_positive(opt.max_step, "max_step");
    if (opt.max_step < opt.min_step) {
        throw std::runtime_error("max_step must be greater than or equal to min_step");
    }
    if (!finite(opt.max_radiation_step) ||
        !finite(opt.max_radiation_depth) ||
        !finite(opt.max_absorption_depth) ||
        !finite(opt.max_faraday_depth)) {
        throw std::runtime_error("radiation step/depth controls must be finite");
    }
    if (opt.max_trace_samples <= 0) {
        throw std::runtime_error("max_trace_samples must be positive");
    }
    if (opt.trace_stride <= 0) {
        throw std::runtime_error("trace_stride must be positive");
    }
    if (opt.trace_compression < 0 || opt.trace_compression > 9) {
        throw std::runtime_error("trace_compression must lie in [0, 9]");
    }
    if (opt.output.empty()) {
        throw std::runtime_error("output path must not be empty");
    }
}

void validate_resolved_camera(const Options& opt) {
    const auto valid_extent = [](Real value) {
        return std::isfinite(static_cast<double>(value)) && value > Real(0);
    };
    if (!valid_extent(opt.fov) || !valid_extent(opt.fovy) ||
        !valid_extent(opt.image_width_x) || !valid_extent(opt.image_width_y)) {
        throw std::runtime_error("resolved camera field of view must be finite and positive");
    }
}

int emission_type_from_name(std::string name) {
    name = lowercase_copy(trim_copy(name));
    if (name == "symphony" || name == "pandya" || name == "thermal" || name == "symphony_pandya_thermal") return 1;
    if (name == "kappa" || name == "symphony_kappa") return 2;
    if (name == "powerlaw" || name == "power_law" || name == "symphony_powerlaw" || name == "symphony_power_law") return 3;
    if (name == "dexter" || name == "dexter_thermal") return 4;
    return std::stoi(name);
}

template<class Model>
void apply_nonthermal_options(Model& model, const Options& opt) {
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
}

std::vector<Real> parse_frequency_list(const std::string& text) {
    std::string normalized = text;
    for (char& c : normalized) {
        if (c == ';') c = ',';
    }
    std::vector<Real> frequencies;
    std::stringstream ss(normalized);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = trim_copy(item);
        if (item.empty()) continue;
        frequencies.push_back(Real(std::stod(item)));
    }
    if (frequencies.empty()) {
        throw std::runtime_error("freq_list did not contain any valid frequencies");
    }
    for (Real frequency : frequencies) {
        if (!(frequency > Real(0))) {
            throw std::runtime_error("all trace frequencies must be positive");
        }
    }
    return frequencies;
}

std::vector<Real> build_trace_frequency_grid(const Options& opt) {
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
        if (i > 0) out << ',';
        out << frequencies[i];
    }
    return out.str();
}

void resolve_trace_frequency(Options& opt) {
    const std::vector<Real> frequencies = build_trace_frequency_grid(opt);
    opt.freq = frequencies.front();
    opt.freq_list = frequency_list_string(frequencies);
    opt.freq_min = Real(-1);
    opt.freq_max = Real(-1);
    opt.nfreq = static_cast<int>(frequencies.size());
}

void print_usage_and_exit() {
    std::cout << "Usage: kpolaris_model_trace [--version] [--parameter_file=file.par] [--model=riaf|torus|iharm|kharma|athenak|bhac|hamr] "
                 "[--trace_mode=single|image] [--ix=i --iy=j] [--output=trace.h5] "
                 "[--evpa_0=N|W] [--trace_fields=coords,coeffs,stokes|state|all|...] "
                 "[--trace_precision=float|double] [--trace_layout=ragged|dense] [--trace_compression=0..9] "
                 "[--max_trace_samples=N] [--trace_stride=N] [--freq=Hz|--freq_list=Hz,...]\n"
                 "Slow-light trace: --slow_light=1 --slow_light_interpolation=fluid|coefficients (default: fluid).\n"
                 "Fast-light trace supports multi-frequency output with shared geometry under /trace/shared.\n"
                 "Trace fields: lambda, coords, plasma, x, k, e1, e2, state, coeffs, stokes, all.\n";
    std::cout << "KHARMA DDC: --kharma_ddc_native=0|1 --kharma_ddc_socket=/tmp/ddc.sock "
                 "--kharma_ddc_manifest=sequence_manifest.json --kharma_ddc_timeout_seconds=7200\n"
                 "Native frame names: ddc_frame_<sequence>.phdf; service protocol STAGE1.\n";
    std::exit(0);
}

void apply_option(Options& opt, std::string key, const std::string& value, const std::string& source) {
    key = normalize_key(std::move(key));
    try {
        if (opt.kharma_ddc.parse(key, value)) return;
        if (key == "help") print_usage_and_exit();
        else if (key == "parameter_file" || key == "input" || key == "input_file" || key == "params" || key == "config") opt.parameter_file = value;
        else if (key == "parameter_output" || key == "params_output" || key == "effective_parameters" || key == "effective_parameter_file") opt.parameter_output = value;
        else if (key == "model") opt.model = lowercase_copy(value);
        else if (key == "output") opt.output = value;
        else if (key == "evpa_0") opt.evpa_0 = parse_evpa_zero(value);
        else if (key == "format") { /* trace output is HDF5 */ }
        else if (key == "trace_mode") opt.trace_mode = lowercase_copy(value);
        else if (key == "trace_fields" || key == "fields") opt.trace_fields = value;
        else if (key == "trace_precision" || key == "precision") opt.trace_precision = lowercase_copy(value);
        else if (key == "trace_layout" || key == "layout") opt.trace_layout = lowercase_copy(value);
        else if (key == "trace_compression" || key == "compression" || key == "gzip") opt.trace_compression = std::stoi(value);
        else if (key == "max_trace_samples" || key == "trace_max_samples") opt.max_trace_samples = std::stoi(value);
        else if (key == "trace_stride" || key == "sample_stride") opt.trace_stride = std::stoi(value);
        else if (key == "ix") opt.ix = std::stoi(value);
        else if (key == "iy") opt.iy = std::stoi(value);
        else if (key == "camera") opt.camera = lowercase_copy(value);
        else if (key == "coordinate") opt.coordinate = lowercase_copy(value);
        else if (key == "nx") opt.nx = std::stoi(value);
        else if (key == "ny") opt.ny = std::stoi(value);
        else if (key == "radius") opt.radius = Real(std::stod(value));
        else if (key == "inclination_rad") opt.inclination = Real(std::stod(value));
        else if (key == "inclination_deg") opt.inclination = Real(std::stod(value)) * Real(3.141592653589793238462643383279502884) / Real(180);
        else if (key == "fov") { opt.fov = Real(std::stod(value)); opt.fovy = Real(-1); opt.xspan = Real(-1); opt.yspan = Real(-1); opt.fovx_dsource = Real(-1); opt.fovy_dsource = Real(-1); }
        else if (key == "fovy" || key == "fov_y") { opt.fovy = Real(std::stod(value)); opt.fovy_dsource = Real(-1); }
        else if (key == "xspan") { opt.xspan = Real(std::stod(value)); opt.fovx_dsource = Real(-1); }
        else if (key == "yspan") { opt.yspan = Real(std::stod(value)); opt.fovy_dsource = Real(-1); }
        else if (key == "dsource" || key == "dsource_pc") opt.dsource_pc = Real(std::stod(value));
        else if (key == "fovx_dsource" || key == "fovx_muas" || key == "fovx_uas") { opt.fovx_dsource = Real(std::stod(value)); opt.xspan = Real(-1); }
        else if (key == "fovy_dsource" || key == "fovy_muas" || key == "fovy_uas") { opt.fovy_dsource = Real(std::stod(value)); opt.yspan = Real(-1); }
        else if (key == "x_offset") opt.x_offset = Real(std::stod(value));
        else if (key == "y_offset") opt.y_offset = Real(std::stod(value));
        else if (key == "use_pinhole_pixel_bias") opt.use_pinhole_pixel_bias = std::stoi(value);
        else if (key == "pinhole_pixel_bias") opt.pinhole_pixel_bias = Real(std::stod(value));
        else if (key == "spin") opt.spin = Real(std::stod(value));
        else if (key == "inner_radius") {
            opt.inner_radius = Real(std::stod(value));
            opt.inner_radius_explicit = true;
        }
        else if (key == "outer_radius") opt.outer_radius = Real(std::stod(value));
        else if (key == "step") opt.step = Real(std::stod(value));
        else if (key == "max_steps") opt.max_steps = std::stoi(value);
        else if (key == "adaptive") opt.adaptive = std::stoi(value);
        else if (key == "adaptive_tolerance") opt.adaptive_tolerance = Real(std::stod(value));
        else if (key == "min_step") opt.min_step = Real(std::stod(value));
        else if (key == "max_step") opt.max_step = Real(std::stod(value));
        else if (key == "max_radiation_step" || key == "radiation_sampling_interval") opt.max_radiation_step = Real(std::stod(value));
        else if (key == "max_radiation_depth" || key == "radiation_sampling_depth") {
            opt.max_radiation_depth = Real(std::stod(value));
            opt.max_absorption_depth = opt.max_radiation_depth;
            opt.max_faraday_depth = opt.max_radiation_depth;
        }
        else if (key == "max_absorption_depth" || key == "absorption_sampling_depth") opt.max_absorption_depth = Real(std::stod(value));
        else if (key == "max_faraday_depth" || key == "faraday_sampling_depth") opt.max_faraday_depth = Real(std::stod(value));
        else if (key == "substeps" || key == "radiation_substeps") { /* retained for image compatibility */ }
        else if (key == "slow_light") opt.slow_light = std::stoi(value);
        else if (key == "slow_light_interpolation") {
            if (value == "coefficients") opt.slow_light_interpolation = kpolaris::SlowLightInterpolation::coefficients;
            else if (value == "fluid") opt.slow_light_interpolation = kpolaris::SlowLightInterpolation::fluid;
            else throw std::invalid_argument("slow_light_interpolation must be coefficients or fluid");
        }
        else if (key == "slow_light_observation_time" || key == "observation_time") opt.slow_light_observation_time = Real(std::stod(value));
        else if (key == "slow_light_dump_list" || key == "dump_list") opt.slow_light_dump_list = value;
        else if (key == "slow_light_time_list" || key == "dump_time_list") opt.slow_light_time_list = value;
        else if (key == "slow_light_dump_pattern" || key == "dump_pattern") opt.slow_light_dump_pattern = value;
        else if (key == "slow_light_dump_start" || key == "dump_start") opt.slow_light_dump_start = std::stoi(value);
        else if (key == "slow_light_dump_end" || key == "dump_end") opt.slow_light_dump_end = std::stoi(value);
        else if (key == "slow_light_dump_stride" || key == "dump_stride") opt.slow_light_dump_stride = std::stoi(value);
        else if (key == "freq" || key == "frequency_hz") {
            opt.freq = Real(std::stod(value));
            opt.freq_list.clear();
            opt.freq_min = Real(-1);
            opt.freq_max = Real(-1);
            opt.nfreq = 0;
        }
        else if (key == "freq_list" || key == "frequency_list" || key == "frequencies") opt.freq_list = value;
        else if (key == "freq_min" || key == "frequency_min") opt.freq_min = Real(std::stod(value));
        else if (key == "freq_max" || key == "frequency_max") opt.freq_max = Real(std::stod(value));
        else if (key == "nfreq" || key == "frequency_count") opt.nfreq = std::stoi(value);
        else if (key == "freq_spacing" || key == "frequency_spacing") opt.freq_spacing = lowercase_copy(value);
        else if (key == "timing" || key == "profile") opt.timing = std::stoi(value);
        else if (key == "closure_x_warning" || key == "closure_k_warning" || key == "frame_error_warning" || key == "basis_identity_warning") { /* image diagnostics */ }
        else if (key == "riaf_r_min") opt.riaf_r_min = Real(std::stod(value));
        else if (key == "riaf_r_max") opt.riaf_r_max = Real(std::stod(value));
        else if (key == "riaf_nth0") opt.riaf_nth0 = Real(std::stod(value));
        else if (key == "riaf_Te0" || key == "riaf_te0") opt.riaf_Te0 = Real(std::stod(value));
        else if (key == "riaf_disk_h") opt.riaf_disk_h = Real(std::stod(value));
        else if (key == "riaf_pow_nth") opt.riaf_pow_nth = Real(std::stod(value));
        else if (key == "riaf_pow_T" || key == "riaf_pow_t") opt.riaf_pow_T = Real(std::stod(value));
        else if (key == "riaf_ne_unit") opt.riaf_ne_unit = Real(std::stod(value));
        else if (key == "riaf_te_unit") opt.riaf_te_unit = Real(std::stod(value));
        else if (key == "riaf_mbh_solar") opt.riaf_mbh_solar = Real(std::stod(value));
        else if (key == "riaf_keplerian_factor") opt.riaf_keplerian_factor = Real(std::stod(value));
        else if (key == "riaf_infall_factor") opt.riaf_infall_factor = Real(std::stod(value));
        else if (key == "emission_type") opt.emission_type = std::stoi(value);
        else if (key == "emission_fit") opt.emission_type = emission_type_from_name(value);
        else if (key == "kappa" || key == "nonthermal_kappa") opt.nonthermal_kappa = Real(std::stod(value));
        else if (key == "variable_kappa") opt.variable_kappa = std::stoi(value);
        else if (key == "variable_kappa_min") opt.variable_kappa_min = Real(std::stod(value));
        else if (key == "variable_kappa_interp_start") opt.variable_kappa_interp_start = Real(std::stod(value));
        else if (key == "variable_kappa_max") opt.variable_kappa_max = Real(std::stod(value));
        else if (key == "powerlaw_p" || key == "power_law_p") opt.powerlaw_p = Real(std::stod(value));
        else if (key == "powerlaw_eta" || key == "power_law_eta") opt.powerlaw_eta = Real(std::stod(value));
        else if (key == "powerlaw_gamma_min" || key == "power_law_gamma_min") opt.powerlaw_gamma_min = Real(std::stod(value));
        else if (key == "powerlaw_gamma_max" || key == "power_law_gamma_max") opt.powerlaw_gamma_max = Real(std::stod(value));
        else if (key == "powerlaw_gamma_cutoff" || key == "power_law_gamma_cutoff") opt.powerlaw_gamma_cutoff = Real(std::stod(value));
        else if (key == "torus_l_lambda") opt.torus_l_lambda = Real(std::stod(value));
        else if (key == "torus_wwin") opt.torus_wwin = Real(std::stod(value));
        else if (key == "torus_kappa") opt.torus_kappa = Real(std::stod(value));
        else if (key == "torus_omegac") opt.torus_omegac = Real(std::stod(value));
        else if (key == "torus_betac") opt.torus_betac = Real(std::stod(value));
        else if (key == "torus_beta") opt.torus_beta = Real(std::stod(value));
        else if (key == "torus_Rhigh" || key == "torus_rhigh") opt.torus_Rhigh = Real(std::stod(value));
        else if (key == "torus_bh_mass_solar") opt.torus_bh_mass_solar = Real(std::stod(value));
        else if (key == "torus_mdot_cgs") opt.torus_mdot_cgs = Real(std::stod(value));
        else if (key == "torus_mdot_code") opt.torus_mdot_code = Real(std::stod(value));
        else if (key == "torus_thetae_min") opt.torus_thetae_min = Real(std::stod(value));
        else if (key == "iharm_dump") opt.iharm_dump = value;
        else if (key == "kharma_dump" || key == "phdf") opt.kharma_dump = value;
        else if (key == "athenak_dump" || key == "athenak_bin" || key == "bin") opt.athenak_dump = value;
        else if (key == "bhac_dump" || key == "bhac_dat" || key == "dat") opt.bhac_dump = value;
        else if (key == "hamr_dump" || key == "hamr_dir" || key == "hamr_path") opt.hamr_dump = value;
        else if (key == "dump") { opt.iharm_dump = value; opt.kharma_dump = value; opt.athenak_dump = value; opt.bhac_dump = value; opt.hamr_dump = value; }
        else if (key == "iharm_M_unit" || key == "iharm_m_unit") opt.iharm_M_unit = Real(std::stod(value));
        else if (key == "kharma_M_unit" || key == "kharma_m_unit") opt.kharma_M_unit = Real(std::stod(value));
        else if (key == "athenak_M_unit" || key == "athenak_m_unit") opt.athenak_M_unit = Real(std::stod(value));
        else if (key == "bhac_M_unit" || key == "bhac_m_unit") opt.bhac_M_unit = Real(std::stod(value));
        else if (key == "hamr_M_unit" || key == "hamr_m_unit") opt.hamr_M_unit = Real(std::stod(value));
        else if (key == "M_unit") { opt.iharm_M_unit = Real(std::stod(value)); opt.kharma_M_unit = Real(std::stod(value)); opt.athenak_M_unit = Real(std::stod(value)); opt.bhac_M_unit = Real(std::stod(value)); opt.hamr_M_unit = Real(std::stod(value)); }
        else if (key == "iharm_mbh_solar") opt.iharm_mbh_solar = Real(std::stod(value));
        else if (key == "kharma_mbh_solar") opt.kharma_mbh_solar = Real(std::stod(value));
        else if (key == "athenak_mbh_solar") opt.athenak_mbh_solar = Real(std::stod(value));
        else if (key == "bhac_mbh_solar") opt.bhac_mbh_solar = Real(std::stod(value));
        else if (key == "hamr_mbh_solar") opt.hamr_mbh_solar = Real(std::stod(value));
        else if (key == "MBH") { opt.iharm_mbh_solar = Real(std::stod(value)); opt.kharma_mbh_solar = Real(std::stod(value)); opt.athenak_mbh_solar = Real(std::stod(value)); opt.bhac_mbh_solar = Real(std::stod(value)); opt.hamr_mbh_solar = Real(std::stod(value)); }
        else if (key == "iharm_trat_small") opt.iharm_trat_small = Real(std::stod(value));
        else if (key == "kharma_trat_small") opt.kharma_trat_small = Real(std::stod(value));
        else if (key == "athenak_trat_small") opt.athenak_trat_small = Real(std::stod(value));
        else if (key == "bhac_trat_small") opt.bhac_trat_small = Real(std::stod(value));
        else if (key == "hamr_trat_small") opt.hamr_trat_small = Real(std::stod(value));
        else if (key == "trat_small") { opt.iharm_trat_small = Real(std::stod(value)); opt.kharma_trat_small = Real(std::stod(value)); opt.athenak_trat_small = Real(std::stod(value)); opt.bhac_trat_small = Real(std::stod(value)); opt.hamr_trat_small = Real(std::stod(value)); }
        else if (key == "iharm_trat_large") opt.iharm_trat_large = Real(std::stod(value));
        else if (key == "kharma_trat_large") opt.kharma_trat_large = Real(std::stod(value));
        else if (key == "athenak_trat_large") opt.athenak_trat_large = Real(std::stod(value));
        else if (key == "bhac_trat_large") opt.bhac_trat_large = Real(std::stod(value));
        else if (key == "hamr_trat_large") opt.hamr_trat_large = Real(std::stod(value));
        else if (key == "trat_large") { opt.iharm_trat_large = Real(std::stod(value)); opt.kharma_trat_large = Real(std::stod(value)); opt.athenak_trat_large = Real(std::stod(value)); opt.bhac_trat_large = Real(std::stod(value)); opt.hamr_trat_large = Real(std::stod(value)); }
        else if (key == "iharm_beta_crit") opt.iharm_beta_crit = Real(std::stod(value));
        else if (key == "kharma_beta_crit") opt.kharma_beta_crit = Real(std::stod(value));
        else if (key == "athenak_beta_crit") opt.athenak_beta_crit = Real(std::stod(value));
        else if (key == "bhac_beta_crit") opt.bhac_beta_crit = Real(std::stod(value));
        else if (key == "athenak_gamma") opt.athenak_gamma = Real(std::stod(value));
        else if (key == "bhac_gamma") opt.bhac_gamma = Real(std::stod(value));
        else if (key == "hamr_beta_crit") opt.hamr_beta_crit = Real(std::stod(value));
        else if (key == "hamr_gamma") opt.hamr_gamma = Real(std::stod(value));
        else if (key == "beta_crit") { opt.iharm_beta_crit = Real(std::stod(value)); opt.kharma_beta_crit = Real(std::stod(value)); opt.athenak_beta_crit = Real(std::stod(value)); opt.bhac_beta_crit = Real(std::stod(value)); opt.hamr_beta_crit = Real(std::stod(value)); }
        else if (key == "iharm_sigma_cut") opt.iharm_sigma_cut = Real(std::stod(value));
        else if (key == "kharma_sigma_cut") opt.kharma_sigma_cut = Real(std::stod(value));
        else if (key == "athenak_sigma_cut") opt.athenak_sigma_cut = Real(std::stod(value));
        else if (key == "bhac_sigma_cut") opt.bhac_sigma_cut = Real(std::stod(value));
        else if (key == "hamr_sigma_cut") opt.hamr_sigma_cut = Real(std::stod(value));
        else if (key == "sigma_cut") { opt.iharm_sigma_cut = Real(std::stod(value)); opt.kharma_sigma_cut = Real(std::stod(value)); opt.athenak_sigma_cut = Real(std::stod(value)); opt.bhac_sigma_cut = Real(std::stod(value)); opt.hamr_sigma_cut = Real(std::stod(value)); }
        else if (key == "iharm_sigma_cut_high") opt.iharm_sigma_cut_high = Real(std::stod(value));
        else if (key == "kharma_sigma_cut_high") opt.kharma_sigma_cut_high = Real(std::stod(value));
        else if (key == "athenak_sigma_cut_high") opt.athenak_sigma_cut_high = Real(std::stod(value));
        else if (key == "bhac_sigma_cut_high") opt.bhac_sigma_cut_high = Real(std::stod(value));
        else if (key == "hamr_sigma_cut_high") opt.hamr_sigma_cut_high = Real(std::stod(value));
        else if (key == "sigma_cut_high") { opt.iharm_sigma_cut_high = Real(std::stod(value)); opt.kharma_sigma_cut_high = Real(std::stod(value)); opt.athenak_sigma_cut_high = Real(std::stod(value)); opt.bhac_sigma_cut_high = Real(std::stod(value)); opt.hamr_sigma_cut_high = Real(std::stod(value)); }
        else if (key == "athenak_r_in" || key == "athenak_resample_r_in") opt.athenak_r_in = Real(std::stod(value));
        else if (key == "athenak_r_out" || key == "athenak_resample_r_out") opt.athenak_r_out = Real(std::stod(value));
        else if (key == "bhac_r_in") opt.bhac_r_in = Real(std::stod(value));
        else if (key == "bhac_r_out") opt.bhac_r_out = Real(std::stod(value));
        else if (key == "bhac_hslope") opt.bhac_hslope = Real(std::stod(value));
        else if (key == "hamr_r_in" || key == "hamr_resample_r_in") opt.hamr_r_in = Real(std::stod(value));
        else if (key == "hamr_r_out" || key == "hamr_resample_r_out") opt.hamr_r_out = Real(std::stod(value));
        else if (key == "hamr_hslope") {
            opt.hamr_hslope = Real(std::stod(value));
            opt.hamr_hslope_explicit = true;
        }
        else if (key == "bhac_nxlone1") opt.bhac_nxlone1 = std::stoi(value);
        else if (key == "bhac_nxlone2") opt.bhac_nxlone2 = std::stoi(value);
        else if (key == "bhac_nxlone3") opt.bhac_nxlone3 = std::stoi(value);
        else if (key == "bhac_spin_index") opt.bhac_spin_index = std::stoi(value);
        else if (key == "bhac_x1_min") opt.bhac_x1_min = Real(std::stod(value));
        else if (key == "bhac_x1_max") opt.bhac_x1_max = Real(std::stod(value));
        else if (key == "bhac_x2_min") opt.bhac_x2_min = Real(std::stod(value));
        else if (key == "bhac_x2_max") opt.bhac_x2_max = Real(std::stod(value));
        else if (key == "bhac_x3_min") opt.bhac_x3_min = Real(std::stod(value));
        else if (key == "bhac_x3_max") opt.bhac_x3_max = Real(std::stod(value));
        else if (key == "bhac_sfc") opt.bhac_sfc = std::stoi(value);
        else if (key == "bhac_profile_mode") opt.bhac_profile_mode = std::stoi(value);
        else if (key == "athenak_resample_n1" || key == "athenak_resample_n2" || key == "athenak_resample_n3") { /* direct CKS trace does not resample */ }
        else if (key == "iharm_resample" || key == "kharma_resample" ||
                 key == "iharm_resample_spherical_ks_precomputed" ||
                 key == "iharm_resample_spherical_ks_primitives" ||
                 key == "kharma_resample_spherical_ks_precomputed" ||
                 key == "kharma_resample_spherical_ks_primitives" ||
                 key == "iharm_resample_n1" || key == "iharm_resample_n2" || key == "iharm_resample_n3" ||
                 key == "kharma_resample_n1" || key == "kharma_resample_n2" || key == "kharma_resample_n3" ||
                 key == "iharm_resample_r_in" || key == "iharm_resample_r_out" ||
                 key == "kharma_resample_r_in" || key == "kharma_resample_r_out") { /* image-only resampling controls */ }
        else if (key == "iharm_interpolate_derived_scalars") opt.iharm_interpolate_derived_scalars = std::stoi(value);
        else if (key == "kharma_interpolate_derived_scalars") opt.kharma_interpolate_derived_scalars = std::stoi(value);
        else if (key == "interpolate_derived_scalars") { opt.iharm_interpolate_derived_scalars = std::stoi(value); opt.kharma_interpolate_derived_scalars = std::stoi(value); }
        else if (key == "kharma_reverse_field") opt.kharma_reverse_field = std::stoi(value);
        else if (key == "bhac_reverse_field") opt.bhac_reverse_field = std::stoi(value);
        else if (key == "hamr_reverse_field") opt.hamr_reverse_field = std::stoi(value);
        else if (key == "hamr_profile_mode") opt.hamr_profile_mode = std::stoi(value);
        else if (key == "hamr_id_order") opt.hamr_id_order = trim_copy(value);
        else if (key == "hamr_root_order") opt.hamr_root_order = trim_copy(value);
        else if (key == "reverse_field") { opt.kharma_reverse_field = std::stoi(value); opt.bhac_reverse_field = std::stoi(value); opt.hamr_reverse_field = std::stoi(value); }
        else if (key == "scalar_transport") opt.scalar_transport = std::stoi(value);
        else throw std::runtime_error("unknown option '" + key + "' in " + source);
    } catch (const std::exception& e) {
        throw std::runtime_error("invalid value for option '" + key + "' in " + source + ": " + value + " (" + e.what() + ")");
    }
}

void apply_parameter_file(Options& opt, const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("failed to open parameter file: " + path);
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const size_t comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        line = trim_copy(line);
        if (line.empty()) continue;
        std::string key, value;
        const size_t eq = line.find('=');
        if (eq != std::string::npos) {
            key = trim_copy(line.substr(0, eq));
            value = trim_copy(line.substr(eq + 1));
        } else {
            std::istringstream iss(line);
            iss >> key >> value;
        }
        if (!key.empty()) apply_option(opt, key, value, path + ":" + std::to_string(line_no));
    }
}

Options parse_options(int argc, char** argv) {
    Options opt;
    std::vector<std::string> parameter_files;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") print_usage_and_exit();
        if (arg.rfind("--", 0) != 0) throw std::runtime_error("expected --key=value option: " + arg);
        const size_t eq = arg.find('=');
        if (eq == std::string::npos) throw std::runtime_error("expected --key=value option: " + arg);
        const std::string key = normalize_key(arg.substr(0, eq));
        const std::string value = arg.substr(eq + 1);
        if (key == "parameter_file" || key == "input" || key == "input_file" || key == "params" || key == "config") {
            parameter_files.push_back(value);
        }
    }
    for (const auto& path : parameter_files) apply_parameter_file(opt, path);
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help" || arg == "-h") continue;
        const size_t eq = arg.find('=');
        apply_option(opt, arg.substr(0, eq), arg.substr(eq + 1), "command line");
    }
    resolve_trace_frequency(opt);
    return opt;
}

std::string canonical_camera_name(std::string name) {
    name = lowercase_copy(name);
    if (name == "parallel" || name == "parallel_plane" || name == "plane") return "parallel_plane";
    if (name == "pinhole" || name == "point") return "pinhole";
    throw std::runtime_error("unknown camera: " + name);
}

std::string canonical_coordinate_name(std::string name) {
    name = lowercase_copy(name);
    if (name == "cartesian_ks" || name == "ks" || name == "cartesian-kerr-schild") return "cartesian_ks";
    if (name == "spherical_ks" || name == "spherical-kerr-schild" || name == "ks_spherical" ||
        name == "kerr_schild_spherical" || name == "sks") return "spherical_ks";
    if (name == "mks" || name == "modified_ks" || name == "modified-kerr-schild" || name == "native_mks") return "mks";
    if (name == "fmks" || name == "funky_modified_ks" || name == "funky-modified-kerr-schild" ||
        name == "native_fmks" || name == "mmks") return "fmks";
    if (name == "boyer_lindquist" || name == "boyer-lindquist" || name == "bl") return "boyer_lindquist";
    throw std::runtime_error("unknown coordinate: " + name);
}

kpolaris::CoordinateSystem coordinate_system_from_name(const std::string& name) {
    const std::string canonical = canonical_coordinate_name(name);
    if (canonical == "boyer_lindquist") return kpolaris::CoordinateSystem::BoyerLindquist;
    if (canonical == "spherical_ks") return kpolaris::CoordinateSystem::SphericalKS;
    if (canonical == "fmks") return kpolaris::CoordinateSystem::FMKS;
    if (canonical == "mks") return kpolaris::CoordinateSystem::MKS;
    return kpolaris::CoordinateSystem::CartesianKS;
}

struct FieldSelection {
    bool lambda = false, coords = false, plasma = false, x = false, k = false, e1 = false, e2 = false;
    bool coeffs = false, stokes = false;
};

FieldSelection parse_fields(std::string text) {
    for (char& c : text) if (c == ';' || c == '+') c = ',';
    FieldSelection f;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        item = lowercase_copy(trim_copy(item));
        if (item.empty()) continue;
        if (item == "minimal" || item == "default") { f.lambda = f.coords = f.coeffs = f.stokes = true; }
        else if (item == "all") { f.lambda = f.coords = f.plasma = f.x = f.k = f.e1 = f.e2 = f.coeffs = f.stokes = true; }
        else if (item == "state") { f.x = f.k = f.e1 = f.e2 = true; }
        else if (item == "screen" || item == "basis") { f.e1 = f.e2 = true; }
        else if (item == "lambda" || item == "dlambda") f.lambda = true;
        else if (item == "coords" || item == "coordinates" || item == "r" || item == "theta" || item == "phi") f.coords = true;
        else if (item == "plasma" || item == "fluid") f.plasma = true;
        else if (item == "x" || item == "position") f.x = true;
        else if (item == "k" || item == "wavevector") f.k = true;
        else if (item == "e1") f.e1 = true;
        else if (item == "e2") f.e2 = true;
        else if (item == "coeffs" || item == "coefficients") f.coeffs = true;
        else if (item == "stokes") f.stokes = true;
        else throw std::runtime_error("unknown trace field: " + item);
    }
    if (!(f.lambda || f.coords || f.plasma || f.x || f.k || f.e1 || f.e2 || f.coeffs || f.stokes)) {
        f.lambda = f.coords = f.coeffs = f.stokes = true;
    }
    return f;
}

template<class StoreReal, class ExecSpace>
Kokkos::View<StoreReal**, ExecSpace> maybe_view(bool enabled, const std::string& name, int nray, int nsample) {
    if (!enabled) return {};
    return Kokkos::View<StoreReal**, ExecSpace>(name, nray, nsample);
}

template<class StoreReal, class ExecSpace>
kpolaris::TraceViews<StoreReal, ExecSpace> allocate_trace_views(const FieldSelection& fields, int nray, int nsample) {
    kpolaris::TraceViews<StoreReal, ExecSpace> v;
    v.pixel = Kokkos::View<int*, ExecSpace>("trace_pixel", nray);
    v.sample_count = Kokkos::View<int*, ExecSpace>("trace_sample_count", nray);
    v.pass_a_steps = Kokkos::View<int*, ExecSpace>("trace_pass_a_steps", nray);
    v.pass_b_steps = Kokkos::View<int*, ExecSpace>("trace_pass_b_steps", nray);
    v.reason = Kokkos::View<int*, ExecSpace>("trace_reason", nray);
    v.closure_x = Kokkos::View<StoreReal*, ExecSpace>("trace_closure_x", nray);
    v.closure_k = Kokkos::View<StoreReal*, ExecSpace>("trace_closure_k", nray);
    v.final_null = Kokkos::View<StoreReal*, ExecSpace>("trace_final_null", nray);
    v.frame_error = Kokkos::View<StoreReal*, ExecSpace>("trace_frame_error", nray);
    v.final_propagated_i = Kokkos::View<StoreReal*, ExecSpace>("trace_final_propagated_i", nray);
    v.final_propagated_q = Kokkos::View<StoreReal*, ExecSpace>("trace_final_propagated_q", nray);
    v.final_propagated_u = Kokkos::View<StoreReal*, ExecSpace>("trace_final_propagated_u", nray);
    v.final_propagated_v = Kokkos::View<StoreReal*, ExecSpace>("trace_final_propagated_v", nray);
    v.final_observed_i = Kokkos::View<StoreReal*, ExecSpace>("trace_final_observed_i", nray);
    v.final_observed_q = Kokkos::View<StoreReal*, ExecSpace>("trace_final_observed_q", nray);
    v.final_observed_u = Kokkos::View<StoreReal*, ExecSpace>("trace_final_observed_u", nray);
    v.final_observed_v = Kokkos::View<StoreReal*, ExecSpace>("trace_final_observed_v", nray);
    v.lambda = maybe_view<StoreReal, ExecSpace>(fields.lambda, "lambda", nray, nsample);
    v.dlambda = maybe_view<StoreReal, ExecSpace>(fields.lambda, "dlambda", nray, nsample);
    v.r = maybe_view<StoreReal, ExecSpace>(fields.coords, "r", nray, nsample);
    v.theta = maybe_view<StoreReal, ExecSpace>(fields.coords, "theta", nray, nsample);
    v.phi = maybe_view<StoreReal, ExecSpace>(fields.coords, "phi", nray, nsample);
    v.ne_cgs = maybe_view<StoreReal, ExecSpace>(fields.plasma, "ne_cgs", nray, nsample);
    v.thetae = maybe_view<StoreReal, ExecSpace>(fields.plasma, "thetae", nray, nsample);
    v.b_cgs = maybe_view<StoreReal, ExecSpace>(fields.plasma, "b_cgs", nray, nsample);
    v.beta = maybe_view<StoreReal, ExecSpace>(fields.plasma, "beta", nray, nsample);
    v.sigma = maybe_view<StoreReal, ExecSpace>(fields.plasma, "sigma", nray, nsample);
    v.nu_fluid_hz = maybe_view<StoreReal, ExecSpace>(fields.plasma, "nu_fluid_hz", nray, nsample);
    v.theta_bk = maybe_view<StoreReal, ExecSpace>(fields.plasma, "theta_bk", nray, nsample);
    v.b1 = maybe_view<StoreReal, ExecSpace>(fields.plasma, "b1", nray, nsample);
    v.b2 = maybe_view<StoreReal, ExecSpace>(fields.plasma, "b2", nray, nsample);
    v.cos2chi = maybe_view<StoreReal, ExecSpace>(fields.plasma, "cos2chi", nray, nsample);
    v.sin2chi = maybe_view<StoreReal, ExecSpace>(fields.plasma, "sin2chi", nray, nsample);
    v.x0 = maybe_view<StoreReal, ExecSpace>(fields.x, "x0", nray, nsample);
    v.x1 = maybe_view<StoreReal, ExecSpace>(fields.x, "x1", nray, nsample);
    v.x2 = maybe_view<StoreReal, ExecSpace>(fields.x, "x2", nray, nsample);
    v.x3 = maybe_view<StoreReal, ExecSpace>(fields.x, "x3", nray, nsample);
    v.k0 = maybe_view<StoreReal, ExecSpace>(fields.k, "k0", nray, nsample);
    v.k1 = maybe_view<StoreReal, ExecSpace>(fields.k, "k1", nray, nsample);
    v.k2 = maybe_view<StoreReal, ExecSpace>(fields.k, "k2", nray, nsample);
    v.k3 = maybe_view<StoreReal, ExecSpace>(fields.k, "k3", nray, nsample);
    v.e10 = maybe_view<StoreReal, ExecSpace>(fields.e1, "e10", nray, nsample);
    v.e11 = maybe_view<StoreReal, ExecSpace>(fields.e1, "e11", nray, nsample);
    v.e12 = maybe_view<StoreReal, ExecSpace>(fields.e1, "e12", nray, nsample);
    v.e13 = maybe_view<StoreReal, ExecSpace>(fields.e1, "e13", nray, nsample);
    v.e20 = maybe_view<StoreReal, ExecSpace>(fields.e2, "e20", nray, nsample);
    v.e21 = maybe_view<StoreReal, ExecSpace>(fields.e2, "e21", nray, nsample);
    v.e22 = maybe_view<StoreReal, ExecSpace>(fields.e2, "e22", nray, nsample);
    v.e23 = maybe_view<StoreReal, ExecSpace>(fields.e2, "e23", nray, nsample);
    v.jI = maybe_view<StoreReal, ExecSpace>(fields.coeffs, "jI", nray, nsample);
    v.jQ = maybe_view<StoreReal, ExecSpace>(fields.coeffs, "jQ", nray, nsample);
    v.jU = maybe_view<StoreReal, ExecSpace>(fields.coeffs, "jU", nray, nsample);
    v.jV = maybe_view<StoreReal, ExecSpace>(fields.coeffs, "jV", nray, nsample);
    v.aI = maybe_view<StoreReal, ExecSpace>(fields.coeffs, "aI", nray, nsample);
    v.aQ = maybe_view<StoreReal, ExecSpace>(fields.coeffs, "aQ", nray, nsample);
    v.aU = maybe_view<StoreReal, ExecSpace>(fields.coeffs, "aU", nray, nsample);
    v.aV = maybe_view<StoreReal, ExecSpace>(fields.coeffs, "aV", nray, nsample);
    v.rQ = maybe_view<StoreReal, ExecSpace>(fields.coeffs, "rhoQ", nray, nsample);
    v.rU = maybe_view<StoreReal, ExecSpace>(fields.coeffs, "rhoU", nray, nsample);
    v.rV = maybe_view<StoreReal, ExecSpace>(fields.coeffs, "rhoV", nray, nsample);
    v.SI = maybe_view<StoreReal, ExecSpace>(fields.stokes, "SI", nray, nsample);
    v.SQ = maybe_view<StoreReal, ExecSpace>(fields.stokes, "SQ", nray, nsample);
    v.SU = maybe_view<StoreReal, ExecSpace>(fields.stokes, "SU", nray, nsample);
    v.SV = maybe_view<StoreReal, ExecSpace>(fields.stokes, "SV", nray, nsample);
    return v;
}

void write_h5_string(H5::H5Object& obj, const std::string& name, const std::string& value) {
    H5::DataSpace space(H5S_SCALAR);
    H5::StrType type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute attr = obj.createAttribute(name, type, space);
    const char* cstr = value.c_str();
    attr.write(type, &cstr);
}

template<class T>
void write_h5_scalar(H5::H5Object& obj, const std::string& name, T value) {
    H5::DataSpace space(H5S_SCALAR);
    H5::Attribute attr = obj.createAttribute(name, H5::PredType::NATIVE_DOUBLE, space);
    const double out = static_cast<double>(value);
    attr.write(H5::PredType::NATIVE_DOUBLE, &out);
}

void write_h5_int_attr(H5::H5Object& obj, const std::string& name, int value) {
    H5::DataSpace space(H5S_SCALAR);
    H5::Attribute attr = obj.createAttribute(name, H5::PredType::NATIVE_INT, space);
    attr.write(H5::PredType::NATIVE_INT, &value);
}

int trace_compression_level(int requested) {
    if (requested < 0) return 0;
    if (requested > 9) return 9;
    return requested;
}

H5::DataSet create_1d_dataset(H5::Group& group,
                              const std::string& name,
                              const H5::PredType& type,
                              hsize_t n,
                              int compression) {
    hsize_t dims[1] = {n};
    H5::DataSpace space(1, dims);
    const int level = trace_compression_level(compression);
    if (level <= 0 || n == 0) {
        return group.createDataSet(name, type, space);
    }
    H5::DSetCreatPropList props;
    hsize_t chunk[1] = {std::min<hsize_t>(n, hsize_t(1024 * 1024))};
    props.setChunk(1, chunk);
    props.setDeflate(level);
    return group.createDataSet(name, type, space, props);
}

H5::DataSet create_2d_dataset(H5::Group& group,
                              const std::string& name,
                              const H5::PredType& type,
                              hsize_t n0,
                              hsize_t n1,
                              int compression) {
    hsize_t dims[2] = {n0, n1};
    H5::DataSpace space(2, dims);
    const int level = trace_compression_level(compression);
    if (level <= 0 || n0 == 0 || n1 == 0) {
        return group.createDataSet(name, type, space);
    }
    H5::DSetCreatPropList props;
    hsize_t chunk[2] = {std::min<hsize_t>(n0, hsize_t(256)), std::min<hsize_t>(n1, hsize_t(256))};
    props.setChunk(2, chunk);
    props.setDeflate(level);
    return group.createDataSet(name, type, space, props);
}

void write_h5_dataset_1d_int(H5::Group& group, const std::string& name, const std::vector<int>& values, int compression = 0) {
    H5::DataSet ds = create_1d_dataset(group, name, H5::PredType::NATIVE_INT,
                                       static_cast<hsize_t>(values.size()), compression);
    ds.write(values.data(), H5::PredType::NATIVE_INT);
}

void write_h5_dataset_1d_ll(H5::Group& group, const std::string& name, const std::vector<long long>& values, int compression = 0) {
    H5::DataSet ds = create_1d_dataset(group, name, H5::PredType::NATIVE_LLONG,
                                       static_cast<hsize_t>(values.size()), compression);
    ds.write(values.data(), H5::PredType::NATIVE_LLONG);
}

template<class StoreReal>
void write_h5_dataset_1d_real(H5::Group& group, const std::string& name, const std::vector<StoreReal>& values, int compression = 0) {
    const H5::PredType type = std::is_same_v<StoreReal, float> ? H5::PredType::NATIVE_FLOAT : H5::PredType::NATIVE_DOUBLE;
    H5::DataSet ds = create_1d_dataset(group, name, type, static_cast<hsize_t>(values.size()), compression);
    ds.write(values.data(), type);
}

template<class StoreReal>
void write_h5_dataset_1d_real_ptr(H5::Group& group,
                                  const std::string& name,
                                  const StoreReal* values,
                                  size_t count,
                                  int compression = 0) {
    const H5::PredType type = std::is_same_v<StoreReal, float> ? H5::PredType::NATIVE_FLOAT : H5::PredType::NATIVE_DOUBLE;
    H5::DataSet ds = create_1d_dataset(group, name, type, static_cast<hsize_t>(count), compression);
    if (count > 0) {
        ds.write(values, type);
    }
}

template<class StoreReal>
void write_h5_dataset_2d_real(H5::Group& group, const std::string& name,
                              const std::vector<StoreReal>& values,
                              int nray,
                              int nsample,
                              int compression = 0) {
    const H5::PredType type = std::is_same_v<StoreReal, float> ? H5::PredType::NATIVE_FLOAT : H5::PredType::NATIVE_DOUBLE;
    H5::DataSet ds = create_2d_dataset(group, name, type,
                                       static_cast<hsize_t>(nray),
                                       static_cast<hsize_t>(nsample),
                                       compression);
    ds.write(values.data(), type);
}

template<class StoreReal, class ExecSpace>
std::vector<StoreReal> copy_view_1d(Kokkos::View<StoreReal*, ExecSpace> view) {
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    std::vector<StoreReal> out(static_cast<size_t>(view.extent(0)));
    for (size_t i = 0; i < out.size(); ++i) out[i] = host(i);
    return out;
}

template<class ExecSpace>
std::vector<int> copy_view_1d_int(Kokkos::View<int*, ExecSpace> view) {
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    std::vector<int> out(static_cast<size_t>(view.extent(0)));
    for (size_t i = 0; i < out.size(); ++i) out[i] = host(i);
    return out;
}

template<class StoreReal, class ExecSpace>
void write_optional_trace_real(H5::Group& group,
                               const Options& opt,
                               const std::string& name,
                               Kokkos::View<StoreReal**, ExecSpace> view,
                               Kokkos::View<int*, ExecSpace> sample_count_view,
                               Kokkos::View<long long*, ExecSpace> sample_offset_view,
                               Kokkos::View<StoreReal*, ExecSpace> compact_scratch,
                               int nray,
                               int nsample,
                               const std::vector<int>& sample_count,
                               const std::vector<long long>& sample_offset) {
    if (view.extent(0) == 0) return;
    (void)sample_count;
    if (opt.trace_layout == "dense") {
        auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
        std::vector<StoreReal> values(static_cast<size_t>(nray) * static_cast<size_t>(nsample));
        for (int i = 0; i < nray; ++i) {
            for (int j = 0; j < nsample; ++j) {
                values[static_cast<size_t>(i) * static_cast<size_t>(nsample) + static_cast<size_t>(j)] = host(i, j);
            }
        }
        write_h5_dataset_2d_real(group, name, values, nray, nsample, opt.trace_compression);
        return;
    }
    if (opt.trace_layout != "ragged") {
        throw std::runtime_error("unknown trace_layout: " + opt.trace_layout);
    }
    const size_t total = static_cast<size_t>(sample_offset.back());
    if (compact_scratch.extent(0) < total) {
        throw std::runtime_error("trace compact scratch is smaller than ragged output");
    }
    Kokkos::parallel_for(
        "KPOLARISTraceCompact" + name,
        Kokkos::RangePolicy<ExecSpace>(0, nray),
        KOKKOS_LAMBDA(const int i) {
            const int nvalid = sample_count_view(i) < nsample ? sample_count_view(i) : nsample;
            const long long base = sample_offset_view(i);
            for (int j = 0; j < nvalid; ++j) {
                compact_scratch(static_cast<size_t>(base) + static_cast<size_t>(j)) = view(i, j);
            }
        });
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), compact_scratch);
    write_h5_dataset_1d_real_ptr(group, name, host.data(), total, opt.trace_compression);
}

struct TraceLayoutHost {
    std::vector<int> pixel;
    std::vector<int> ix;
    std::vector<int> iy;
    std::vector<int> sample_count;
    std::vector<long long> sample_offset;
};

struct TraceDerivedHost {
    int available = 0;
    std::vector<int> complete;
    std::vector<int> intensity_freeze_valid;
    std::vector<double> net_linear_fraction;
    std::vector<double> net_circular_fraction;
    std::vector<double> final_evpa_wrapped_rad;
    std::vector<double> evpa_unwrapped_change_rad;
    std::vector<double> absorption_depth;
    std::vector<double> faraday_rotation_depth;
    std::vector<double> faraday_conversion_depth;
    std::vector<double> faraday_operator_depth;
    std::vector<double> emissivity_formation_radius_low;
    std::vector<double> emissivity_formation_radius_median;
    std::vector<double> emissivity_formation_radius_high;
    std::vector<double> intensity_freeze_lambda;
    std::vector<double> intensity_freeze_radius;
    std::vector<double> post_freeze_absorption_depth;
    std::vector<double> post_freeze_faraday_rotation_depth;
    std::vector<double> post_freeze_faraday_conversion_depth;
    std::vector<double> post_freeze_faraday_operator_depth;
    std::vector<double> linear_fraction;
    std::vector<double> circular_fraction;
    std::vector<double> evpa_wrapped_rad;
    std::vector<double> evpa_unwrapped_rad;
    std::vector<double> cumulative_absorption_depth;
    std::vector<double> cumulative_faraday_rotation_depth;
    std::vector<double> cumulative_faraday_conversion_depth;
    std::vector<double> cumulative_faraday_operator_depth;
    std::vector<double> emissivity_weight;
    std::vector<double> cumulative_emissivity_fraction;
    std::vector<double> intensity_fraction_of_final;
};

template<class StoreReal, class ExecSpace>
TraceDerivedHost derive_trace_physics(
    const Options& opt,
    const kpolaris::TraceConfig<Real>& config,
    const kpolaris::TraceViews<StoreReal, ExecSpace>& views,
    const TraceLayoutHost& layout) {
    TraceDerivedHost out;
    const bool inputs_available =
        views.lambda.extent(0) > 0 && views.dlambda.extent(0) > 0 &&
        views.r.extent(0) > 0 && views.jI.extent(0) > 0 &&
        views.aI.extent(0) > 0 && views.aQ.extent(0) > 0 &&
        views.aU.extent(0) > 0 && views.aV.extent(0) > 0 &&
        views.rQ.extent(0) > 0 && views.rU.extent(0) > 0 &&
        views.rV.extent(0) > 0 && views.SI.extent(0) > 0 &&
        views.SQ.extent(0) > 0 && views.SU.extent(0) > 0 &&
        views.SV.extent(0) > 0;
    if (!inputs_available) {
        return out;
    }
    out.available = 1;
    const int nray = config.ray_count;
    const size_t sample_storage = opt.trace_layout == "dense" ?
        static_cast<size_t>(nray) * static_cast<size_t>(config.max_samples) :
        static_cast<size_t>(layout.sample_offset.back());
    out.complete.assign(nray, 0);
    out.intensity_freeze_valid.assign(nray, 0);
    out.net_linear_fraction.assign(nray, 0.0);
    out.net_circular_fraction.assign(nray, 0.0);
    out.final_evpa_wrapped_rad.assign(nray, 0.0);
    out.evpa_unwrapped_change_rad.assign(nray, 0.0);
    out.absorption_depth.assign(nray, 0.0);
    out.faraday_rotation_depth.assign(nray, 0.0);
    out.faraday_conversion_depth.assign(nray, 0.0);
    out.faraday_operator_depth.assign(nray, 0.0);
    out.emissivity_formation_radius_low.assign(nray, 0.0);
    out.emissivity_formation_radius_median.assign(nray, 0.0);
    out.emissivity_formation_radius_high.assign(nray, 0.0);
    out.intensity_freeze_lambda.assign(nray, 0.0);
    out.intensity_freeze_radius.assign(nray, 0.0);
    out.post_freeze_absorption_depth.assign(nray, 0.0);
    out.post_freeze_faraday_rotation_depth.assign(nray, 0.0);
    out.post_freeze_faraday_conversion_depth.assign(nray, 0.0);
    out.post_freeze_faraday_operator_depth.assign(nray, 0.0);
    out.linear_fraction.assign(sample_storage, 0.0);
    out.circular_fraction.assign(sample_storage, 0.0);
    out.evpa_wrapped_rad.assign(sample_storage, 0.0);
    out.evpa_unwrapped_rad.assign(sample_storage, 0.0);
    out.cumulative_absorption_depth.assign(sample_storage, 0.0);
    out.cumulative_faraday_rotation_depth.assign(sample_storage, 0.0);
    out.cumulative_faraday_conversion_depth.assign(sample_storage, 0.0);
    out.cumulative_faraday_operator_depth.assign(sample_storage, 0.0);
    out.emissivity_weight.assign(sample_storage, 0.0);
    out.cumulative_emissivity_fraction.assign(sample_storage, 0.0);
    out.intensity_fraction_of_final.assign(sample_storage, 0.0);

    const auto lambda = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), views.lambda);
    const auto dlambda = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), views.dlambda);
    const auto radius = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), views.r);
    const auto j_i = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), views.jI);
    const auto a_i = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), views.aI);
    const auto a_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), views.aQ);
    const auto a_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), views.aU);
    const auto a_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), views.aV);
    const auto rho_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), views.rQ);
    const auto rho_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), views.rU);
    const auto rho_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), views.rV);
    const auto s_i = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), views.SI);
    const auto s_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), views.SQ);
    const auto s_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), views.SU);
    const auto s_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), views.SV);
    const std::vector<int> pass_b_steps = copy_view_1d_int(views.pass_b_steps);
    const std::vector<StoreReal> final_prop_i = copy_view_1d(views.final_propagated_i);
    const std::vector<StoreReal> final_prop_q = copy_view_1d(views.final_propagated_q);
    const std::vector<StoreReal> final_prop_u = copy_view_1d(views.final_propagated_u);
    const std::vector<StoreReal> final_obs_i = copy_view_1d(views.final_observed_i);
    const std::vector<StoreReal> final_obs_q = copy_view_1d(views.final_observed_q);
    const std::vector<StoreReal> final_obs_u = copy_view_1d(views.final_observed_u);
    const std::vector<StoreReal> final_obs_v = copy_view_1d(views.final_observed_v);
    const double pi = 3.141592653589793238462643383279502884;

    auto storage_index = [&](int ray, int sample) -> size_t {
        return opt.trace_layout == "dense" ?
            static_cast<size_t>(ray) * static_cast<size_t>(config.max_samples) +
                static_cast<size_t>(sample) :
            static_cast<size_t>(layout.sample_offset[static_cast<size_t>(ray)]) +
                static_cast<size_t>(sample);
    };

    for (int ray = 0; ray < nray; ++ray) {
        const int n = std::min(layout.sample_count[static_cast<size_t>(ray)], config.max_samples);
        out.complete[static_cast<size_t>(ray)] =
            config.sample_stride == 1 && n == pass_b_steps[static_cast<size_t>(ray)] ? 1 : 0;
        const double obs_i = static_cast<double>(final_obs_i[static_cast<size_t>(ray)]);
        const double obs_q = output_qu_sign(opt.evpa_0) * static_cast<double>(final_obs_q[static_cast<size_t>(ray)]);
        const double obs_u = output_qu_sign(opt.evpa_0) * static_cast<double>(final_obs_u[static_cast<size_t>(ray)]);
        const double obs_v = static_cast<double>(final_obs_v[static_cast<size_t>(ray)]);
        if (std::abs(obs_i) > std::numeric_limits<double>::min()) {
            out.net_linear_fraction[static_cast<size_t>(ray)] =
                std::hypot(obs_q, obs_u) / std::abs(obs_i);
            out.net_circular_fraction[static_cast<size_t>(ray)] = obs_v / obs_i;
        }
        if (std::hypot(obs_q, obs_u) > 0.0) {
            out.final_evpa_wrapped_rad[static_cast<size_t>(ray)] =
                0.5 * std::atan2(obs_u, obs_q);
        }

        double total_emissivity = 0.0;
        for (int sample = 0; sample < n; ++sample) {
            const double dl = std::abs(static_cast<double>(dlambda(ray, sample)));
            total_emissivity += std::max(static_cast<double>(j_i(ray, sample)), 0.0) * dl;
        }

        double cumulative_absorption = 0.0;
        double cumulative_rotation = 0.0;
        double cumulative_conversion = 0.0;
        double cumulative_operator = 0.0;
        double cumulative_emissivity = 0.0;
        double previous_evpa = 0.0;
        double unwrapped_evpa = 0.0;
        bool have_evpa = false;
        std::vector<double> prefix_absorption(static_cast<size_t>(n) + 1, 0.0);
        std::vector<double> prefix_rotation(static_cast<size_t>(n) + 1, 0.0);
        std::vector<double> prefix_conversion(static_cast<size_t>(n) + 1, 0.0);
        std::vector<double> prefix_operator(static_cast<size_t>(n) + 1, 0.0);
        for (int sample = 0; sample < n; ++sample) {
            const size_t index = storage_index(ray, sample);
            const double dl = std::abs(static_cast<double>(dlambda(ray, sample)));
            kpolaris::TransferCoeffs<double> coeffs;
            coeffs.aI = static_cast<double>(a_i(ray, sample));
            coeffs.aQ = static_cast<double>(a_q(ray, sample));
            coeffs.aU = static_cast<double>(a_u(ray, sample));
            coeffs.aV = static_cast<double>(a_v(ray, sample));
            coeffs.rQ = static_cast<double>(rho_q(ray, sample));
            coeffs.rU = static_cast<double>(rho_u(ray, sample));
            coeffs.rV = static_cast<double>(rho_v(ray, sample));
            cumulative_absorption += std::max(coeffs.aI, 0.0) * dl;
            cumulative_rotation += coeffs.rV * dl;
            cumulative_conversion += std::hypot(coeffs.rQ, coeffs.rU) * dl;
            cumulative_operator += kpolaris::faraday_operator_rate(coeffs) * dl;
            const double emission = std::max(static_cast<double>(j_i(ray, sample)), 0.0) * dl;
            cumulative_emissivity += emission;
            const double si = static_cast<double>(s_i(ray, sample));
            const double sq = static_cast<double>(s_q(ray, sample));
            const double su = static_cast<double>(s_u(ray, sample));
            const double sv = static_cast<double>(s_v(ray, sample));
            if (std::abs(si) > std::numeric_limits<double>::min()) {
                out.linear_fraction[index] = std::hypot(sq, su) / std::abs(si);
                out.circular_fraction[index] = sv / si;
            }
            const double final_i =
                static_cast<double>(final_prop_i[static_cast<size_t>(ray)]);
            if (std::abs(final_i) > std::numeric_limits<double>::min()) {
                out.intensity_fraction_of_final[index] =
                    si / final_i;
            }
            if (std::hypot(sq, su) > 0.0) {
                const double wrapped = 0.5 * std::atan2(su, sq);
                out.evpa_wrapped_rad[index] = wrapped;
                if (!have_evpa) {
                    unwrapped_evpa = wrapped;
                    have_evpa = true;
                } else {
                    double delta = wrapped - previous_evpa;
                    while (delta > 0.5 * pi) delta -= pi;
                    while (delta < -0.5 * pi) delta += pi;
                    unwrapped_evpa += delta;
                }
                previous_evpa = wrapped;
                out.evpa_unwrapped_rad[index] = unwrapped_evpa;
            } else if (have_evpa) {
                out.evpa_unwrapped_rad[index] = unwrapped_evpa;
            }
            out.cumulative_absorption_depth[index] = cumulative_absorption;
            out.cumulative_faraday_rotation_depth[index] = cumulative_rotation;
            out.cumulative_faraday_conversion_depth[index] = cumulative_conversion;
            out.cumulative_faraday_operator_depth[index] = cumulative_operator;
            out.emissivity_weight[index] = emission;
            out.cumulative_emissivity_fraction[index] =
                total_emissivity > 0.0 ? cumulative_emissivity / total_emissivity : 0.0;
            prefix_absorption[static_cast<size_t>(sample) + 1] = cumulative_absorption;
            prefix_rotation[static_cast<size_t>(sample) + 1] = cumulative_rotation;
            prefix_conversion[static_cast<size_t>(sample) + 1] = cumulative_conversion;
            prefix_operator[static_cast<size_t>(sample) + 1] = cumulative_operator;
        }
        out.absorption_depth[static_cast<size_t>(ray)] = cumulative_absorption;
        out.faraday_rotation_depth[static_cast<size_t>(ray)] = cumulative_rotation;
        out.faraday_conversion_depth[static_cast<size_t>(ray)] = cumulative_conversion;
        out.faraday_operator_depth[static_cast<size_t>(ray)] = cumulative_operator;

        if (have_evpa) {
            const double final_q = static_cast<double>(final_prop_q[static_cast<size_t>(ray)]);
            const double final_u = static_cast<double>(final_prop_u[static_cast<size_t>(ray)]);
            if (std::hypot(final_q, final_u) > 0.0) {
                const double wrapped_final = 0.5 * std::atan2(final_u, final_q);
                double delta = wrapped_final - previous_evpa;
                while (delta > 0.5 * pi) delta -= pi;
                while (delta < -0.5 * pi) delta += pi;
                out.evpa_unwrapped_change_rad[static_cast<size_t>(ray)] =
                    unwrapped_evpa + delta - out.evpa_unwrapped_rad[storage_index(ray, 0)];
            }
        }

        if (total_emissivity > 0.0) {
            std::vector<std::pair<double, double>> radius_weight;
            radius_weight.reserve(static_cast<size_t>(n));
            for (int sample = 0; sample < n; ++sample) {
                radius_weight.emplace_back(
                    static_cast<double>(radius(ray, sample)),
                    std::max(static_cast<double>(j_i(ray, sample)), 0.0) *
                        std::abs(static_cast<double>(dlambda(ray, sample))));
            }
            const auto quantiles = kpolaris::emissivity_radius_quantiles(
                std::move(radius_weight));
            out.emissivity_formation_radius_low[static_cast<size_t>(ray)] = quantiles[0];
            out.emissivity_formation_radius_median[static_cast<size_t>(ray)] = quantiles[1];
            out.emissivity_formation_radius_high[static_cast<size_t>(ray)] = quantiles[2];
        }

        if (out.complete[static_cast<size_t>(ray)] && n > 0) {
            const double final_i = static_cast<double>(final_prop_i[static_cast<size_t>(ray)]);
            const int freeze = kpolaris::intensity_freeze_sample(n, final_i,
                [&](int sample) { return static_cast<double>(s_i(ray, sample)); });
            if (freeze >= 0) {
                out.intensity_freeze_valid[static_cast<size_t>(ray)] = 1;
                out.intensity_freeze_lambda[static_cast<size_t>(ray)] =
                    static_cast<double>(lambda(ray, freeze));
                out.intensity_freeze_radius[static_cast<size_t>(ray)] =
                    static_cast<double>(radius(ray, freeze));
                out.post_freeze_absorption_depth[static_cast<size_t>(ray)] =
                    cumulative_absorption - prefix_absorption[static_cast<size_t>(freeze)];
                out.post_freeze_faraday_rotation_depth[static_cast<size_t>(ray)] =
                    cumulative_rotation - prefix_rotation[static_cast<size_t>(freeze)];
                out.post_freeze_faraday_conversion_depth[static_cast<size_t>(ray)] =
                    cumulative_conversion - prefix_conversion[static_cast<size_t>(freeze)];
                out.post_freeze_faraday_operator_depth[static_cast<size_t>(ray)] =
                    cumulative_operator - prefix_operator[static_cast<size_t>(freeze)];
            }
        }
    }
    return out;
}

template<class StoreReal, class ExecSpace>
TraceLayoutHost make_trace_layout_host(const Options& opt,
                                       const kpolaris::TraceConfig<Real>& config,
                                       const kpolaris::TraceViews<StoreReal, ExecSpace>& views) {
    TraceLayoutHost layout;
    layout.pixel = copy_view_1d_int(views.pixel);
    layout.sample_count = copy_view_1d_int(views.sample_count);
    layout.sample_offset.assign(layout.sample_count.size() + 1, 0);
    for (size_t i = 0; i < layout.sample_count.size(); ++i) {
        layout.sample_offset[i + 1] =
            layout.sample_offset[i] + static_cast<long long>(std::min(layout.sample_count[i], config.max_samples));
    }
    layout.ix.resize(layout.pixel.size());
    layout.iy.resize(layout.pixel.size());
    for (size_t i = 0; i < layout.pixel.size(); ++i) {
        layout.ix[i] = layout.pixel[i] % opt.nx;
        layout.iy[i] = layout.pixel[i] / opt.nx;
    }
    return layout;
}

bool same_trace_sample_layout(const TraceLayoutHost& a, const TraceLayoutHost& b) {
    return a.pixel == b.pixel && a.sample_count == b.sample_count && a.sample_offset == b.sample_offset;
}

template<class StoreReal, class ExecSpace, class Writer>
void with_trace_write_scratch(const Options& opt,
                              const TraceLayoutHost& layout,
                              Writer writer) {
    Kokkos::View<long long*, ExecSpace> sample_offset_device("trace_sample_offset", layout.sample_offset.size());
    auto sample_offset_host = Kokkos::create_mirror_view(sample_offset_device);
    for (size_t i = 0; i < layout.sample_offset.size(); ++i) {
        sample_offset_host(i) = layout.sample_offset[i];
    }
    Kokkos::deep_copy(sample_offset_device, sample_offset_host);
    Kokkos::View<StoreReal*, ExecSpace> trace_compact_scratch;
    if (opt.trace_layout == "ragged") {
        trace_compact_scratch =
            Kokkos::View<StoreReal*, ExecSpace>("trace_compact_scratch", static_cast<size_t>(layout.sample_offset.back()));
    }
    writer(sample_offset_device, trace_compact_scratch);
}

template<class StoreReal, class ExecSpace>
void write_trace_rays_group(H5::Group& rays,
                            const Options& opt,
                            const kpolaris::TraceViews<StoreReal, ExecSpace>& views,
                            const TraceLayoutHost& layout) {
    write_h5_string(rays, "evpa_0", opt.evpa_0);
    write_h5_string(rays, "stokes_basis", "final_observed: output N/W basis; final_propagated: transported internal screen");
    write_h5_dataset_1d_int(rays, "pixel", layout.pixel);
    write_h5_dataset_1d_int(rays, "ix", layout.ix);
    write_h5_dataset_1d_int(rays, "iy", layout.iy);
    write_h5_dataset_1d_int(rays, "sample_count", layout.sample_count, opt.trace_compression);
    if (opt.trace_layout == "ragged") {
        write_h5_dataset_1d_ll(rays, "sample_offset", layout.sample_offset, opt.trace_compression);
    }
    write_h5_dataset_1d_int(rays, "pass_a_steps", copy_view_1d_int(views.pass_a_steps));
    write_h5_dataset_1d_int(rays, "pass_b_steps", copy_view_1d_int(views.pass_b_steps));
    write_h5_dataset_1d_int(rays, "reason", copy_view_1d_int(views.reason));
    write_h5_dataset_1d_real(rays, "closure_x", copy_view_1d(views.closure_x));
    write_h5_dataset_1d_real(rays, "closure_k", copy_view_1d(views.closure_k));
    write_h5_dataset_1d_real(rays, "final_null", copy_view_1d(views.final_null));
    write_h5_dataset_1d_real(rays, "frame_error", copy_view_1d(views.frame_error));
    write_h5_dataset_1d_real(rays, "final_propagated_I_inv", copy_view_1d(views.final_propagated_i));
    write_h5_dataset_1d_real(rays, "final_propagated_Q_inv", copy_view_1d(views.final_propagated_q));
    write_h5_dataset_1d_real(rays, "final_propagated_U_inv", copy_view_1d(views.final_propagated_u));
    write_h5_dataset_1d_real(rays, "final_propagated_V_inv", copy_view_1d(views.final_propagated_v));
    write_h5_dataset_1d_real(rays, "final_observed_I_inv", copy_view_1d(views.final_observed_i));
    write_h5_dataset_1d_real(rays, "final_observed_Q_inv", output_linear_component(copy_view_1d(views.final_observed_q), opt.evpa_0));
    write_h5_dataset_1d_real(rays, "final_observed_U_inv", output_linear_component(copy_view_1d(views.final_observed_u), opt.evpa_0));
    write_h5_dataset_1d_real(rays, "final_observed_V_inv", copy_view_1d(views.final_observed_v));
}

template<class StoreReal, class ExecSpace>
void write_trace_geometry_group(H5::Group& trace,
                                const Options& opt,
                                const kpolaris::TraceConfig<Real>& config,
                                const kpolaris::TraceViews<StoreReal, ExecSpace>& views,
                                const TraceLayoutHost& layout) {
    with_trace_write_scratch<StoreReal, ExecSpace>(
        opt, layout,
        [&](Kokkos::View<long long*, ExecSpace> sample_offset_device,
            Kokkos::View<StoreReal*, ExecSpace> trace_compact_scratch) {
            write_optional_trace_real(trace, opt, "lambda", views.lambda, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "dlambda", views.dlambda, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "r", views.r, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "theta", views.theta, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "phi", views.phi, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "ne_cgs", views.ne_cgs, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "thetae", views.thetae, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "b_cgs", views.b_cgs, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "beta", views.beta, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "sigma", views.sigma, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "nu_fluid_hz", views.nu_fluid_hz, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "theta_bk", views.theta_bk, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "b1", views.b1, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "b2", views.b2, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "cos2chi", views.cos2chi, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "sin2chi", views.sin2chi, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "x0", views.x0, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "x1", views.x1, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "x2", views.x2, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "x3", views.x3, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "k0", views.k0, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "k1", views.k1, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "k2", views.k2, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "k3", views.k3, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "e10", views.e10, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "e11", views.e11, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "e12", views.e12, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "e13", views.e13, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "e20", views.e20, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "e21", views.e21, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "e22", views.e22, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "e23", views.e23, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
        });
}

template<class StoreReal, class ExecSpace>
void write_trace_frequency_group(H5::Group& trace,
                                 const Options& opt,
                                 const kpolaris::TraceConfig<Real>& config,
                                 const kpolaris::TraceViews<StoreReal, ExecSpace>& views,
                                 const TraceLayoutHost& layout) {
    with_trace_write_scratch<StoreReal, ExecSpace>(
        opt, layout,
        [&](Kokkos::View<long long*, ExecSpace> sample_offset_device,
            Kokkos::View<StoreReal*, ExecSpace> trace_compact_scratch) {
            write_optional_trace_real(trace, opt, "jI", views.jI, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "jQ", views.jQ, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "jU", views.jU, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "jV", views.jV, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "aI", views.aI, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "aQ", views.aQ, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "aU", views.aU, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "aV", views.aV, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "rhoQ", views.rQ, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "rhoU", views.rU, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "rhoV", views.rV, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "SI", views.SI, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "SQ", views.SQ, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "SU", views.SU, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
            write_optional_trace_real(trace, opt, "SV", views.SV, views.sample_count,
                                      sample_offset_device, trace_compact_scratch, config.ray_count,
                                      config.max_samples, layout.sample_count, layout.sample_offset);
        });
}

template<class StoreReal, class ExecSpace>
void write_trace_derived_group(
    H5::Group& derived,
    const Options& opt,
    const kpolaris::TraceConfig<Real>& config,
    const kpolaris::TraceViews<StoreReal, ExecSpace>& views,
    const TraceLayoutHost& layout) {
    const TraceDerivedHost values = derive_trace_physics(opt, config, views, layout);
    write_h5_int_attr(derived, "available", values.available);
    write_h5_string(derived, "final_evpa_0", opt.evpa_0);
    write_h5_string(derived, "sample_evpa_basis", "transported internal screen; independent of output evpa_0");
    write_h5_scalar(derived, "intensity_freeze_relative_tolerance", 0.01);
    write_h5_string(derived, "evpa_definition", "0.5*atan2(U,Q); wrapped modulo pi and unwrapped by nearest-period continuation");
    write_h5_string(derived, "emissivity_formation_definition", "5/50/95 radius-sorted empirical quantiles weighted by max(j_I,0)*abs(dlambda); not observer weighted");
    write_h5_string(derived, "intensity_freeze_definition", "earliest complete recorded sample after which propagated I remains within 1 percent of positive final propagated I (no peak-history normalization)");
    write_h5_string(derived, "post_freeze_definition", "raw absorption/Faraday depth accumulated from the intensity-freeze sample to the camera");
    if (!values.available) {
        write_h5_string(derived, "unavailable_reason",
                        "derived trace physics requires lambda, coords, coeffs, and stokes trace fields");
        return;
    }
    write_h5_dataset_1d_int(derived, "complete", values.complete);
    write_h5_dataset_1d_int(derived, "intensity_freeze_valid", values.intensity_freeze_valid);
    write_h5_dataset_1d_real(derived, "net_linear_fraction", values.net_linear_fraction);
    write_h5_dataset_1d_real(derived, "net_circular_fraction", values.net_circular_fraction);
    write_h5_dataset_1d_real(derived, "final_evpa_wrapped_rad", values.final_evpa_wrapped_rad);
    write_h5_dataset_1d_real(derived, "evpa_unwrapped_change_rad", values.evpa_unwrapped_change_rad);
    write_h5_dataset_1d_real(derived, "absorption_depth", values.absorption_depth);
    write_h5_dataset_1d_real(derived, "faraday_rotation_depth", values.faraday_rotation_depth);
    write_h5_dataset_1d_real(derived, "faraday_conversion_depth", values.faraday_conversion_depth);
    write_h5_dataset_1d_real(derived, "faraday_operator_depth", values.faraday_operator_depth);
    write_h5_dataset_1d_real(derived, "emissivity_formation_radius_low", values.emissivity_formation_radius_low);
    write_h5_dataset_1d_real(derived, "emissivity_formation_radius_median", values.emissivity_formation_radius_median);
    write_h5_dataset_1d_real(derived, "emissivity_formation_radius_high", values.emissivity_formation_radius_high);
    write_h5_dataset_1d_real(derived, "intensity_freeze_lambda", values.intensity_freeze_lambda);
    write_h5_dataset_1d_real(derived, "intensity_freeze_radius", values.intensity_freeze_radius);
    write_h5_dataset_1d_real(derived, "post_freeze_absorption_depth", values.post_freeze_absorption_depth);
    write_h5_dataset_1d_real(derived, "post_freeze_faraday_rotation_depth", values.post_freeze_faraday_rotation_depth);
    write_h5_dataset_1d_real(derived, "post_freeze_faraday_conversion_depth", values.post_freeze_faraday_conversion_depth);
    write_h5_dataset_1d_real(derived, "post_freeze_faraday_operator_depth", values.post_freeze_faraday_operator_depth);

    auto write_sample = [&](const std::string& name, const std::vector<double>& data) {
        if (opt.trace_layout == "dense") {
            write_h5_dataset_2d_real(
                derived, name, data, config.ray_count, config.max_samples,
                opt.trace_compression);
        } else {
            write_h5_dataset_1d_real(derived, name, data, opt.trace_compression);
        }
    };
    write_sample("linear_fraction", values.linear_fraction);
    write_sample("circular_fraction", values.circular_fraction);
    write_sample("evpa_wrapped_rad", values.evpa_wrapped_rad);
    write_sample("evpa_unwrapped_rad", values.evpa_unwrapped_rad);
    write_sample("cumulative_absorption_depth", values.cumulative_absorption_depth);
    write_sample("cumulative_faraday_rotation_depth", values.cumulative_faraday_rotation_depth);
    write_sample("cumulative_faraday_conversion_depth", values.cumulative_faraday_conversion_depth);
    write_sample("cumulative_faraday_operator_depth", values.cumulative_faraday_operator_depth);
    write_sample("emissivity_weight", values.emissivity_weight);
    write_sample("cumulative_emissivity_fraction", values.cumulative_emissivity_fraction);
    write_sample("intensity_fraction_of_final", values.intensity_fraction_of_final);
}

std::string effective_parameter_output_path(const Options& opt) {
    if (opt.parameter_output == "none" || opt.parameter_output == "off" ||
        opt.parameter_output == "0") {
        return {};
    }
    if (opt.parameter_output.empty() || opt.parameter_output == "auto") {
        return opt.output + ".params";
    }
    return opt.parameter_output;
}

void write_effective_parameter_file(const Options& opt,
                                    const kpolaris::PassAParams<Real>& pass_a,
                                    const std::vector<Real>& frequencies) {
    const std::string path = effective_parameter_output_path(opt);
    if (path.empty()) return;
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open effective parameter output: " + path);
    }
    out.precision(17);
    out << "# KPolaris effective trace parameter file\n"
        << "# Generated after parameter-file parsing, command-line overrides, model loading, and derived defaults.\n"
        << "# code_version=" << kpolaris::build_info::version << "\n"
        << "# code_revision=" << kpolaris::build_info::source_revision << "\n"
        << "# code_source_dirty=" << kpolaris::build_info::source_dirty << "\n"
        << "# code_source_fingerprint=" << kpolaris::build_info::source_fingerprint << "\n"
        << "# compiler_id=" << kpolaris::build_info::compiler_id << "\n"
        << "# compiler_version=" << kpolaris::build_info::compiler_version << "\n"
        << "# build_type=" << kpolaris::build_info::build_type << "\n";
    if (!opt.parameter_file.empty()) out << "# source_parameter_file=" << opt.parameter_file << "\n";
    out << "parameter_output=" << path << "\n"
        << "model=" << opt.model << "\n";
    if (!opt.iharm_dump.empty()) out << "iharm_dump=" << opt.iharm_dump << "\n";
    if (!opt.kharma_dump.empty()) out << "kharma_dump=" << opt.kharma_dump << "\n";
    if (opt.model == "kharma") opt.kharma_ddc.write_parameters(out);
    if (!opt.athenak_dump.empty()) out << "athenak_dump=" << opt.athenak_dump << "\n";
    if (!opt.bhac_dump.empty()) out << "bhac_dump=" << opt.bhac_dump << "\n";
    if (!opt.hamr_dump.empty()) out << "hamr_dump=" << opt.hamr_dump << "\n";
    out << "output=" << opt.output << "\n"
        << "camera=" << opt.camera << "\n"
        << "coordinate=" << opt.coordinate << "\n"
        << "trace_mode=" << opt.trace_mode << "\n"
        << "trace_fields=" << opt.trace_fields << "\n"
        << "trace_precision=" << opt.trace_precision << "\n"
        << "trace_layout=" << opt.trace_layout << "\n"
        << "trace_compression=" << opt.trace_compression << "\n"
        << "ix=" << opt.ix << "\n"
        << "iy=" << opt.iy << "\n"
        << "nx=" << opt.nx << "\n"
        << "ny=" << opt.ny << "\n"
        << "max_trace_samples=" << opt.max_trace_samples << "\n"
        << "trace_stride=" << opt.trace_stride << "\n"
        << "radius=" << opt.radius << "\n"
        << "inclination_rad=" << opt.inclination << "\n"
        << "fov=" << pass_a.camera.fov << "\n"
        << "fovy=" << (pass_a.camera.fov_y > Real(0) ? pass_a.camera.fov_y : pass_a.camera.fov) << "\n"
        << "# requested_xspan_M=" << opt.xspan << "\n"
        << "# requested_yspan_M=" << opt.yspan << "\n"
        << "dsource=" << opt.dsource_pc << "\n"
        << "# effective_fovx_dsource_uas=" << opt.effective_fovx_dsource << "\n"
        << "# effective_fovy_dsource_uas=" << opt.effective_fovy_dsource << "\n"
        << "# derived_image_width_x_M=" << opt.image_width_x << "\n"
        << "# derived_image_width_y_M=" << opt.image_width_y << "\n"
        << "x_offset=" << opt.x_offset << "\n"
        << "y_offset=" << opt.y_offset << "\n"
        << "use_pinhole_pixel_bias=" << opt.use_pinhole_pixel_bias << "\n"
        << "pinhole_pixel_bias=" << opt.pinhole_pixel_bias << "\n"
        << "# effective_x_offset=" << pass_a.camera.x_offset << "\n"
        << "# effective_y_offset=" << pass_a.camera.y_offset << "\n"
        << "spin=" << opt.spin << "\n"
        << "inner_radius=" << pass_a.inner_radius << "\n"
        << "outer_radius=" << pass_a.outer_radius << "\n"
        << "step=" << opt.step << "\n"
        << "max_steps=" << opt.max_steps << "\n"
        << "adaptive=" << opt.adaptive << "\n"
        << "adaptive_tolerance=" << opt.adaptive_tolerance << "\n"
        << "min_step=" << opt.min_step << "\n"
        << "max_step=" << opt.max_step << "\n"
        << "max_radiation_step=" << opt.max_radiation_step << "\n"
        << "max_radiation_depth=" << opt.max_radiation_depth << "\n"
        << "max_absorption_depth=" << opt.max_absorption_depth << "\n"
        << "max_faraday_depth=" << opt.max_faraday_depth << "\n"
        << "# derived_fmks_startx1=" << opt.fmks_startx1 << "\n"
        << "# derived_fmks_hslope=" << opt.fmks_hslope << "\n"
        << "# derived_fmks_mks_smooth=" << opt.fmks_mks_smooth << "\n"
        << "# derived_fmks_poly_alpha=" << opt.fmks_poly_alpha << "\n"
        << "# derived_fmks_poly_xt=" << opt.fmks_poly_xt << "\n"
        << "# derived_fmks_poly_norm=" << opt.fmks_poly_norm << "\n"
        << "freq=" << frequencies.front() << "\n"
        << "freq_list=" << frequency_list_string(frequencies) << "\n"
        << "nfreq=" << frequencies.size() << "\n"
        << "freq_spacing=" << opt.freq_spacing << "\n"
        << "# derived_nfreq=" << frequencies.size() << "\n"
        << "# derived_freq_list=" << frequency_list_string(frequencies) << "\n"
        << "slow_light=" << opt.slow_light << "\n"
        << "slow_light_interpolation=" << kpolaris::slow_light_interpolation_name(opt.slow_light_interpolation) << "\n"
        << "slow_light_observation_time=" << opt.slow_light_observation_time << "\n";
    if (!opt.slow_light_dump_list.empty()) out << "slow_light_dump_list=" << opt.slow_light_dump_list << "\n";
    if (!opt.slow_light_time_list.empty()) out << "slow_light_time_list=" << opt.slow_light_time_list << "\n";
    if (!opt.slow_light_dump_pattern.empty()) {
        out << "slow_light_dump_pattern=" << opt.slow_light_dump_pattern << "\n"
            << "slow_light_dump_start=" << opt.slow_light_dump_start << "\n"
            << "slow_light_dump_end=" << opt.slow_light_dump_end << "\n"
            << "slow_light_dump_stride=" << opt.slow_light_dump_stride << "\n";
    }

    out << "emission_type=" << effective_trace_emission_type(opt) << "\n"
        << "# emission_fit=" << emission_fit_name(effective_trace_emission_type(opt)) << "\n"
        << "nonthermal_kappa=" << opt.nonthermal_kappa << "\n"
        << "variable_kappa=" << opt.variable_kappa << "\n"
        << "variable_kappa_min=" << opt.variable_kappa_min << "\n"
        << "variable_kappa_interp_start=" << opt.variable_kappa_interp_start << "\n"
        << "variable_kappa_max=" << opt.variable_kappa_max << "\n"
        << "powerlaw_p=" << opt.powerlaw_p << "\n"
        << "powerlaw_eta=" << opt.powerlaw_eta << "\n"
        << "powerlaw_gamma_min=" << opt.powerlaw_gamma_min << "\n"
        << "powerlaw_gamma_max=" << opt.powerlaw_gamma_max << "\n"
        << "powerlaw_gamma_cutoff=" << opt.powerlaw_gamma_cutoff << "\n";

    if (opt.model == "riaf") {
        out << "riaf_r_min=" << opt.riaf_r_min << "\n"
            << "riaf_r_max=" << opt.riaf_r_max << "\n"
            << "riaf_nth0=" << opt.riaf_nth0 << "\n"
            << "riaf_Te0=" << opt.riaf_Te0 << "\n"
            << "riaf_disk_h=" << opt.riaf_disk_h << "\n"
            << "riaf_pow_nth=" << opt.riaf_pow_nth << "\n"
            << "riaf_pow_T=" << opt.riaf_pow_T << "\n"
            << "riaf_ne_unit=" << opt.riaf_ne_unit << "\n"
            << "riaf_te_unit=" << opt.riaf_te_unit << "\n"
            << "riaf_mbh_solar=" << opt.riaf_mbh_solar << "\n"
            << "riaf_keplerian_factor=" << opt.riaf_keplerian_factor << "\n"
            << "riaf_infall_factor=" << opt.riaf_infall_factor << "\n";
    } else if (opt.model == "torus") {
        out << "torus_l_lambda=" << opt.torus_l_lambda << "\n"
            << "torus_wwin=" << opt.torus_wwin << "\n"
            << "torus_kappa=" << opt.torus_kappa << "\n"
            << "torus_omegac=" << opt.torus_omegac << "\n"
            << "torus_betac=" << opt.torus_betac << "\n"
            << "torus_beta=" << opt.torus_beta << "\n"
            << "torus_Rhigh=" << opt.torus_Rhigh << "\n"
            << "torus_bh_mass_solar=" << opt.torus_bh_mass_solar << "\n"
            << "torus_mdot_cgs=" << opt.torus_mdot_cgs << "\n"
            << "torus_mdot_code=" << opt.torus_mdot_code << "\n"
            << "torus_thetae_min=" << opt.torus_thetae_min << "\n";
    } else if (opt.model == "iharm") {
        out << "iharm_M_unit=" << opt.iharm_M_unit << "\n"
            << "iharm_mbh_solar=" << opt.iharm_mbh_solar << "\n"
            << "iharm_trat_small=" << opt.iharm_trat_small << "\n"
            << "iharm_trat_large=" << opt.iharm_trat_large << "\n"
            << "iharm_beta_crit=" << opt.iharm_beta_crit << "\n"
            << "iharm_sigma_cut=" << opt.iharm_sigma_cut << "\n"
            << "iharm_sigma_cut_high=" << opt.iharm_sigma_cut_high << "\n"
            << "iharm_interpolate_derived_scalars=" << opt.iharm_interpolate_derived_scalars << "\n";
    } else if (opt.model == "kharma") {
        out << "kharma_M_unit=" << opt.kharma_M_unit << "\n"
            << "kharma_mbh_solar=" << opt.kharma_mbh_solar << "\n"
            << "kharma_trat_small=" << opt.kharma_trat_small << "\n"
            << "kharma_trat_large=" << opt.kharma_trat_large << "\n"
            << "kharma_beta_crit=" << opt.kharma_beta_crit << "\n"
            << "kharma_sigma_cut=" << opt.kharma_sigma_cut << "\n"
            << "kharma_sigma_cut_high=" << opt.kharma_sigma_cut_high << "\n"
            << "kharma_interpolate_derived_scalars=" << opt.kharma_interpolate_derived_scalars << "\n"
            << "kharma_reverse_field=" << opt.kharma_reverse_field << "\n";
    } else if (opt.model == "athenak") {
        out << "athenak_M_unit=" << opt.athenak_M_unit << "\n"
            << "athenak_mbh_solar=" << opt.athenak_mbh_solar << "\n"
            << "athenak_trat_small=" << opt.athenak_trat_small << "\n"
            << "athenak_trat_large=" << opt.athenak_trat_large << "\n"
            << "athenak_beta_crit=" << opt.athenak_beta_crit << "\n"
            << "athenak_gamma=" << opt.athenak_gamma << "\n"
            << "athenak_sigma_cut=" << opt.athenak_sigma_cut << "\n"
            << "athenak_sigma_cut_high=" << opt.athenak_sigma_cut_high << "\n"
            << "athenak_r_in=" << opt.athenak_r_in << "\n"
            << "athenak_r_out=" << opt.athenak_r_out << "\n";
    } else if (opt.model == "bhac") {
        out << "bhac_M_unit=" << opt.bhac_M_unit << "\n"
            << "bhac_mbh_solar=" << opt.bhac_mbh_solar << "\n"
            << "bhac_trat_small=" << opt.bhac_trat_small << "\n"
            << "bhac_trat_large=" << opt.bhac_trat_large << "\n"
            << "bhac_beta_crit=" << opt.bhac_beta_crit << "\n"
            << "bhac_gamma=" << opt.bhac_gamma << "\n"
            << "bhac_sigma_cut=" << opt.bhac_sigma_cut << "\n"
            << "bhac_sigma_cut_high=" << opt.bhac_sigma_cut_high << "\n"
            << "bhac_r_in=" << opt.bhac_r_in << "\n"
            << "bhac_r_out=" << opt.bhac_r_out << "\n"
            << "bhac_hslope=" << opt.bhac_hslope << "\n"
            << "bhac_nxlone1=" << opt.bhac_nxlone1 << "\n"
            << "bhac_nxlone2=" << opt.bhac_nxlone2 << "\n"
            << "bhac_nxlone3=" << opt.bhac_nxlone3 << "\n"
            << "bhac_spin_index=" << opt.bhac_spin_index << "\n"
            << "bhac_x1_min=" << opt.bhac_x1_min << "\n"
            << "bhac_x1_max=" << opt.bhac_x1_max << "\n"
            << "bhac_x2_min=" << opt.bhac_x2_min << "\n"
            << "bhac_x2_max=" << opt.bhac_x2_max << "\n"
            << "bhac_x3_min=" << opt.bhac_x3_min << "\n"
            << "bhac_x3_max=" << opt.bhac_x3_max << "\n"
            << "bhac_sfc=" << opt.bhac_sfc << "\n"
            << "bhac_reverse_field=" << opt.bhac_reverse_field << "\n"
            << "bhac_profile_mode=" << opt.bhac_profile_mode << "\n";
    } else if (opt.model == "hamr") {
        out << "hamr_M_unit=" << opt.hamr_M_unit << "\n"
            << "hamr_mbh_solar=" << opt.hamr_mbh_solar << "\n"
            << "hamr_trat_small=" << opt.hamr_trat_small << "\n"
            << "hamr_trat_large=" << opt.hamr_trat_large << "\n"
            << "hamr_beta_crit=" << opt.hamr_beta_crit << "\n"
            << "hamr_gamma=" << opt.hamr_gamma << "\n"
            << "hamr_sigma_cut=" << opt.hamr_sigma_cut << "\n"
            << "hamr_sigma_cut_high=" << opt.hamr_sigma_cut_high << "\n"
            << "hamr_r_in=" << opt.hamr_r_in << "\n"
            << "hamr_r_out=" << opt.hamr_r_out << "\n"
            << "hamr_hslope=" << opt.hamr_hslope << "\n"
            << "# hamr_hslope_source=" << hamr_hslope_source(opt) << "\n"
            << "hamr_reverse_field=" << opt.hamr_reverse_field << "\n"
            << "hamr_profile_mode=" << opt.hamr_profile_mode << "\n"
            << "hamr_id_order=" << opt.hamr_id_order << "\n"
            << "hamr_root_order=" << opt.hamr_root_order << "\n";
    }
    out << "scalar_transport=" << opt.scalar_transport << "\n"
        << "timing=" << opt.timing << "\n"
        << "# stokes_convention=camera_frame\n"
        << "# polarization_basis=camera_screen\n"
        << "evpa_0=" << opt.evpa_0 << "\n";
}

void write_trace_root_metadata(H5::H5File& file,
                               const Options& opt,
                               const kpolaris::TraceConfig<Real>& config,
                               int schema_version,
                               const std::vector<Real>& frequencies) {
    write_h5_string(file, "schema", "kpolaris_trace");
    write_kpolaris_camera_conventions(file, opt.evpa_0, true);
    write_h5_int_attr(file, "schema_version", schema_version);
    write_h5_string(file, "code", "KPolaris");
    write_h5_string(file, "code_version", kpolaris::build_info::version);
    write_h5_string(file, "code_revision", kpolaris::build_info::source_revision);
    write_h5_int_attr(file, "code_source_dirty", kpolaris::build_info::source_dirty);
    write_h5_string(file, "code_source_fingerprint", kpolaris::build_info::source_fingerprint);
    write_h5_string(file, "compiler_id", kpolaris::build_info::compiler_id);
    write_h5_string(file, "compiler_version", kpolaris::build_info::compiler_version);
    write_h5_string(file, "build_type", kpolaris::build_info::build_type);
    write_h5_string(file, "trace_mode", opt.trace_mode);
    write_h5_string(file, "trace_precision", opt.trace_precision);
    write_h5_string(file, "trace_fields", opt.trace_fields);
    write_h5_string(file, "trace_layout", opt.trace_layout);
    write_h5_int_attr(file, "trace_compression", trace_compression_level(opt.trace_compression));
    write_h5_string(file, "model", opt.model);
    if (!opt.parameter_file.empty()) write_h5_string(file, "parameter_file", opt.parameter_file);
    const std::string parameter_output = effective_parameter_output_path(opt);
    if (!parameter_output.empty()) write_h5_string(file, "effective_parameter_file", parameter_output);
    if (opt.model == "iharm" && !opt.iharm_dump.empty()) write_h5_string(file, "iharm_dump", opt.iharm_dump);
    if (opt.model == "kharma" && !opt.kharma_dump.empty()) write_h5_string(file, "kharma_dump", opt.kharma_dump);
    if (opt.model == "kharma") {
        write_h5_int_attr(file, "kharma_ddc_native", opt.kharma_ddc.native);
        write_h5_string(file, "kharma_ddc_socket", opt.kharma_ddc.socket_path);
        write_h5_string(file, "kharma_ddc_manifest", opt.kharma_ddc.manifest);
        write_h5_string(file, "kharma_ddc_protocol", opt.kharma_ddc.native ? "STAGE1" : "disabled");
        write_h5_int_attr(file, "kharma_ddc_timeout_seconds", opt.kharma_ddc.timeout_seconds);
    }
    if (opt.model == "athenak" && !opt.athenak_dump.empty()) write_h5_string(file, "athenak_dump", opt.athenak_dump);
    if (opt.model == "bhac" && !opt.bhac_dump.empty()) write_h5_string(file, "bhac_dump", opt.bhac_dump);
    if (opt.model == "hamr" && !opt.hamr_dump.empty()) write_h5_string(file, "hamr_dump", opt.hamr_dump);
    write_h5_int_attr(file, "slow_light", opt.slow_light);
    write_h5_string(file, "slow_light_interpolation", kpolaris::slow_light_interpolation_name(opt.slow_light_interpolation));
    write_h5_scalar(file, "slow_light_observation_time", opt.slow_light_observation_time);
    if (!opt.slow_light_dump_list.empty()) write_h5_string(file, "slow_light_dump_list", opt.slow_light_dump_list);
    if (!opt.slow_light_time_list.empty()) write_h5_string(file, "slow_light_time_list", opt.slow_light_time_list);
    if (!opt.slow_light_dump_pattern.empty()) write_h5_string(file, "slow_light_dump_pattern", opt.slow_light_dump_pattern);
    write_h5_string(file, "camera", opt.camera);
    write_h5_string(file, "coordinate", opt.coordinate);
    write_h5_int_attr(file, "nx", opt.nx);
    write_h5_int_attr(file, "ny", opt.ny);
    write_h5_int_attr(file, "ray_count", config.ray_count);
    write_h5_int_attr(file, "max_trace_samples", config.max_samples);
    write_h5_int_attr(file, "trace_stride", config.sample_stride);
    write_h5_int_attr(file, "nfreq", static_cast<int>(frequencies.size()));
    write_h5_scalar(file, "frequency_hz", frequencies.front());
    write_h5_string(file, "frequency_list_hz", frequency_list_string(frequencies));
}

void write_trace_parameter_groups(H5::H5File& file,
                                  const Options& opt,
                                  const kpolaris::PassAParams<Real>& pass_a,
                                  const std::vector<Real>& frequencies) {
    H5::Group params = file.createGroup("/parameters");
    H5::Group radiation = params.createGroup("radiation");
    write_h5_int_attr(radiation, "nfreq", static_cast<int>(frequencies.size()));
    write_h5_scalar(radiation, "frequency_hz", frequencies.front());
    write_h5_string(radiation, "frequency_list_hz", frequency_list_string(frequencies));
    if (frequencies.size() > 1) {
        write_h5_dataset_1d_real(radiation, "frequency_hz_by_freq", frequencies);
    }
    write_h5_string(radiation, "model", opt.model);
    if (!opt.parameter_file.empty()) write_h5_string(params, "parameter_file", opt.parameter_file);
    const std::string parameter_output = effective_parameter_output_path(opt);
    if (!parameter_output.empty()) write_h5_string(params, "effective_parameter_file", parameter_output);

    if (opt.model != "torus") {
        const int emission_type = effective_trace_emission_type(opt);
        write_h5_int_attr(radiation, "emission_type", emission_type);
        write_h5_string(radiation, "emission_fit", emission_fit_name(emission_type));
        write_h5_scalar(radiation, "nonthermal_kappa", opt.nonthermal_kappa);
        write_h5_int_attr(radiation, "variable_kappa", opt.variable_kappa);
        write_h5_scalar(radiation, "variable_kappa_min", opt.variable_kappa_min);
        write_h5_scalar(radiation, "variable_kappa_interp_start", opt.variable_kappa_interp_start);
        write_h5_scalar(radiation, "variable_kappa_max", opt.variable_kappa_max);
        write_h5_scalar(radiation, "powerlaw_p", opt.powerlaw_p);
        write_h5_scalar(radiation, "powerlaw_eta", opt.powerlaw_eta);
        write_h5_scalar(radiation, "powerlaw_gamma_min", opt.powerlaw_gamma_min);
        write_h5_scalar(radiation, "powerlaw_gamma_max", opt.powerlaw_gamma_max);
        write_h5_scalar(radiation, "powerlaw_gamma_cutoff", opt.powerlaw_gamma_cutoff);
    }

    if (opt.model == "riaf") {
        write_h5_scalar(radiation, "riaf_r_min", opt.riaf_r_min);
        write_h5_scalar(radiation, "riaf_r_max", opt.riaf_r_max);
        write_h5_scalar(radiation, "riaf_nth0", opt.riaf_nth0);
        write_h5_scalar(radiation, "riaf_Te0", opt.riaf_Te0);
        write_h5_scalar(radiation, "riaf_disk_h", opt.riaf_disk_h);
        write_h5_scalar(radiation, "riaf_pow_nth", opt.riaf_pow_nth);
        write_h5_scalar(radiation, "riaf_pow_T", opt.riaf_pow_T);
        write_h5_scalar(radiation, "riaf_ne_unit", opt.riaf_ne_unit);
        write_h5_scalar(radiation, "riaf_te_unit", opt.riaf_te_unit);
        write_h5_scalar(radiation, "riaf_mbh_solar", opt.riaf_mbh_solar);
        write_h5_scalar(radiation, "riaf_keplerian_factor", opt.riaf_keplerian_factor);
        write_h5_scalar(radiation, "riaf_infall_factor", opt.riaf_infall_factor);
    } else if (opt.model == "torus") {
        write_h5_scalar(radiation, "torus_l_lambda", opt.torus_l_lambda);
        write_h5_scalar(radiation, "torus_wwin", opt.torus_wwin);
        write_h5_scalar(radiation, "torus_kappa", opt.torus_kappa);
        write_h5_scalar(radiation, "torus_omegac", opt.torus_omegac);
        write_h5_scalar(radiation, "torus_betac", opt.torus_betac);
        write_h5_scalar(radiation, "torus_beta", opt.torus_beta);
        write_h5_scalar(radiation, "torus_Rhigh", opt.torus_Rhigh);
        write_h5_scalar(radiation, "torus_bh_mass_solar", opt.torus_bh_mass_solar);
        write_h5_scalar(radiation, "torus_mdot_cgs", opt.torus_mdot_cgs);
        write_h5_scalar(radiation, "torus_mdot_code", opt.torus_mdot_code);
        write_h5_scalar(radiation, "torus_thetae_min", opt.torus_thetae_min);
        write_h5_int_attr(radiation, "scalar_transport", opt.scalar_transport);
    } else if (opt.model == "iharm") {
        if (!opt.iharm_dump.empty()) write_h5_string(radiation, "iharm_dump", opt.iharm_dump);
        write_h5_scalar(radiation, "iharm_M_unit", opt.iharm_M_unit);
        write_h5_scalar(radiation, "iharm_mbh_solar", opt.iharm_mbh_solar);
        write_h5_scalar(radiation, "iharm_trat_small", opt.iharm_trat_small);
        write_h5_scalar(radiation, "iharm_trat_large", opt.iharm_trat_large);
        write_h5_scalar(radiation, "iharm_beta_crit", opt.iharm_beta_crit);
        write_h5_scalar(radiation, "iharm_sigma_cut", opt.iharm_sigma_cut);
        write_h5_scalar(radiation, "iharm_sigma_cut_high", opt.iharm_sigma_cut_high);
        write_h5_int_attr(radiation, "iharm_interpolate_derived_scalars", opt.iharm_interpolate_derived_scalars);
    } else if (opt.model == "kharma") {
        if (!opt.kharma_dump.empty()) write_h5_string(radiation, "kharma_dump", opt.kharma_dump);
        if (opt.model == "kharma") {
            write_h5_int_attr(radiation, "kharma_ddc_native", opt.kharma_ddc.native);
            write_h5_string(radiation, "kharma_ddc_socket", opt.kharma_ddc.socket_path);
            write_h5_string(radiation, "kharma_ddc_manifest", opt.kharma_ddc.manifest);
            write_h5_string(radiation, "kharma_ddc_protocol", opt.kharma_ddc.native ? "STAGE1" : "disabled");
            write_h5_int_attr(radiation, "kharma_ddc_timeout_seconds", opt.kharma_ddc.timeout_seconds);
        }
        write_h5_scalar(radiation, "kharma_M_unit", opt.kharma_M_unit);
        write_h5_scalar(radiation, "kharma_mbh_solar", opt.kharma_mbh_solar);
        write_h5_scalar(radiation, "kharma_trat_small", opt.kharma_trat_small);
        write_h5_scalar(radiation, "kharma_trat_large", opt.kharma_trat_large);
        write_h5_scalar(radiation, "kharma_beta_crit", opt.kharma_beta_crit);
        write_h5_scalar(radiation, "kharma_sigma_cut", opt.kharma_sigma_cut);
        write_h5_scalar(radiation, "kharma_sigma_cut_high", opt.kharma_sigma_cut_high);
        write_h5_int_attr(radiation, "kharma_interpolate_derived_scalars", opt.kharma_interpolate_derived_scalars);
        write_h5_int_attr(radiation, "kharma_reverse_field", opt.kharma_reverse_field);
    } else if (opt.model == "athenak") {
        if (!opt.athenak_dump.empty()) write_h5_string(radiation, "athenak_dump", opt.athenak_dump);
        write_h5_scalar(radiation, "athenak_M_unit", opt.athenak_M_unit);
        write_h5_scalar(radiation, "athenak_mbh_solar", opt.athenak_mbh_solar);
        write_h5_scalar(radiation, "athenak_trat_small", opt.athenak_trat_small);
        write_h5_scalar(radiation, "athenak_trat_large", opt.athenak_trat_large);
        write_h5_scalar(radiation, "athenak_beta_crit", opt.athenak_beta_crit);
        write_h5_scalar(radiation, "athenak_gamma", opt.athenak_gamma);
        write_h5_scalar(radiation, "athenak_sigma_cut", opt.athenak_sigma_cut);
        write_h5_scalar(radiation, "athenak_sigma_cut_high", opt.athenak_sigma_cut_high);
        write_h5_scalar(radiation, "athenak_r_in", opt.athenak_r_in);
        write_h5_scalar(radiation, "athenak_r_out", opt.athenak_r_out);
    } else if (opt.model == "bhac") {
        if (!opt.bhac_dump.empty()) write_h5_string(radiation, "bhac_dump", opt.bhac_dump);
        write_h5_scalar(radiation, "bhac_M_unit", opt.bhac_M_unit);
        write_h5_scalar(radiation, "bhac_mbh_solar", opt.bhac_mbh_solar);
        write_h5_scalar(radiation, "bhac_trat_small", opt.bhac_trat_small);
        write_h5_scalar(radiation, "bhac_trat_large", opt.bhac_trat_large);
        write_h5_scalar(radiation, "bhac_beta_crit", opt.bhac_beta_crit);
        write_h5_scalar(radiation, "bhac_gamma", opt.bhac_gamma);
        write_h5_scalar(radiation, "bhac_sigma_cut", opt.bhac_sigma_cut);
        write_h5_scalar(radiation, "bhac_sigma_cut_high", opt.bhac_sigma_cut_high);
        write_h5_scalar(radiation, "bhac_r_in", opt.bhac_r_in);
        write_h5_scalar(radiation, "bhac_r_out", opt.bhac_r_out);
        write_h5_scalar(radiation, "bhac_hslope", opt.bhac_hslope);
        write_h5_int_attr(radiation, "bhac_reverse_field", opt.bhac_reverse_field);
        write_h5_int_attr(radiation, "bhac_profile_mode", opt.bhac_profile_mode);
    } else if (opt.model == "hamr") {
        if (!opt.hamr_dump.empty()) write_h5_string(radiation, "hamr_dump", opt.hamr_dump);
        write_h5_scalar(radiation, "hamr_M_unit", opt.hamr_M_unit);
        write_h5_scalar(radiation, "hamr_mbh_solar", opt.hamr_mbh_solar);
        write_h5_scalar(radiation, "hamr_trat_small", opt.hamr_trat_small);
        write_h5_scalar(radiation, "hamr_trat_large", opt.hamr_trat_large);
        write_h5_scalar(radiation, "hamr_beta_crit", opt.hamr_beta_crit);
        write_h5_scalar(radiation, "hamr_gamma", opt.hamr_gamma);
        write_h5_scalar(radiation, "hamr_sigma_cut", opt.hamr_sigma_cut);
        write_h5_scalar(radiation, "hamr_sigma_cut_high", opt.hamr_sigma_cut_high);
        write_h5_scalar(radiation, "hamr_r_in", opt.hamr_r_in);
        write_h5_scalar(radiation, "hamr_r_out", opt.hamr_r_out);
        write_h5_scalar(radiation, "hamr_hslope", opt.hamr_hslope);
        write_h5_string(radiation, "hamr_hslope_source", hamr_hslope_source(opt));
        write_h5_int_attr(radiation, "hamr_reverse_field", opt.hamr_reverse_field);
        write_h5_int_attr(radiation, "hamr_profile_mode", opt.hamr_profile_mode);
        write_h5_string(radiation, "hamr_id_order", opt.hamr_id_order);
        write_h5_string(radiation, "hamr_root_order", opt.hamr_root_order);
    }

    H5::Group spacetime = params.createGroup("spacetime");
    write_h5_string(spacetime, "coordinate", opt.coordinate);
    write_h5_string(spacetime, "metric", "Kerr");
    write_h5_scalar(spacetime, "mass", Real(1));
    write_h5_scalar(spacetime, "spin", opt.spin);

    H5::Group camera = params.createGroup("camera");
    write_h5_string(camera, "model", opt.camera);
    write_h5_int_attr(camera, "nx", opt.nx);
    write_h5_int_attr(camera, "ny", opt.ny);
    write_h5_scalar(camera, "radius", opt.radius);
    write_h5_scalar(camera, "inclination_rad", opt.inclination);
    write_h5_scalar(camera, "fov", pass_a.camera.fov);
    write_h5_scalar(camera, "fovy", pass_a.camera.fov_y > Real(0) ? pass_a.camera.fov_y : pass_a.camera.fov);
    write_h5_scalar(camera, "xspan", opt.xspan);
    write_h5_scalar(camera, "yspan", opt.yspan);
    write_h5_scalar(camera, "dsource", opt.dsource_pc);
    write_h5_scalar(camera, "fovx_dsource", opt.effective_fovx_dsource);
    write_h5_scalar(camera, "fovy_dsource", opt.effective_fovy_dsource);
    write_h5_scalar(camera, "image_width_x_M", opt.image_width_x);
    write_h5_scalar(camera, "image_width_y_M", opt.image_width_y);
    write_h5_scalar(camera, "x_offset", pass_a.camera.x_offset);
    write_h5_scalar(camera, "y_offset", pass_a.camera.y_offset);
    write_h5_int_attr(camera, "use_pinhole_pixel_bias", opt.use_pinhole_pixel_bias);
    write_h5_scalar(camera, "pinhole_pixel_bias", opt.pinhole_pixel_bias);

    H5::Group integration = params.createGroup("integration");
    write_h5_scalar(integration, "step", opt.step);
    write_h5_int_attr(integration, "adaptive", opt.adaptive);
    write_h5_scalar(integration, "adaptive_tolerance", opt.adaptive_tolerance);
    write_h5_scalar(integration, "min_step", opt.min_step);
    write_h5_scalar(integration, "max_step", opt.max_step);
    write_h5_scalar(integration, "max_radiation_step", opt.max_radiation_step);
    write_h5_scalar(integration, "max_radiation_depth", opt.max_radiation_depth);
    write_h5_scalar(integration, "max_absorption_depth", opt.max_absorption_depth);
    write_h5_scalar(integration, "max_faraday_depth", opt.max_faraday_depth);
    write_h5_int_attr(integration, "max_steps", opt.max_steps);
    write_h5_int_attr(integration, "slow_light", opt.slow_light);
    write_h5_string(integration, "slow_light_interpolation", kpolaris::slow_light_interpolation_name(opt.slow_light_interpolation));
    write_h5_scalar(integration, "slow_light_observation_time", opt.slow_light_observation_time);
    write_h5_scalar(integration, "inner_radius", pass_a.inner_radius);
    write_h5_scalar(integration, "outer_radius", pass_a.outer_radius);
}

template<class StoreReal, class ExecSpace>
double write_trace_file(const Options& opt,
                        const kpolaris::PassAParams<Real>& pass_a,
                        const kpolaris::TraceConfig<Real>& config,
                        const kpolaris::TraceViews<StoreReal, ExecSpace>& views) {
    Kokkos::Timer write_timer;
    H5::H5File file(opt.output, H5F_ACC_TRUNC);
    const std::vector<Real> frequencies = {opt.freq};
    write_trace_root_metadata(file, opt, config, 2, frequencies);
    write_trace_parameter_groups(file, opt, pass_a, frequencies);

    const TraceLayoutHost layout = make_trace_layout_host(opt, config, views);
    H5::Group rays = file.createGroup("/rays");
    write_trace_rays_group(rays, opt, views, layout);

    H5::Group trace = file.createGroup("/trace");
    write_trace_geometry_group(trace, opt, config, views, layout);
    write_trace_frequency_group(trace, opt, config, views, layout);
    H5::Group derived = file.createGroup("/derived");
    write_trace_derived_group(derived, opt, config, views, layout);

    return write_timer.seconds();
}

template<class StoreReal, class ExecSpace>
double write_multifrequency_trace_file(
    const Options& opt,
    const kpolaris::PassAParams<Real>& pass_a,
    const kpolaris::TraceConfig<Real>& config,
    const std::vector<Real>& frequencies,
    const std::vector<kpolaris::TraceViews<StoreReal, ExecSpace>>& frequency_views) {
    if (frequencies.empty() || frequencies.size() != frequency_views.size()) {
        throw std::runtime_error("multi-frequency trace write received inconsistent frequency/view counts");
    }
    Kokkos::Timer write_timer;
    H5::H5File file(opt.output, H5F_ACC_TRUNC);
    write_trace_root_metadata(file, opt, config, 3, frequencies);

    H5::Group grid = file.createGroup("/grid");
    write_h5_dataset_1d_real(grid, "frequency_hz", frequencies);
    write_trace_parameter_groups(file, opt, pass_a, frequencies);

    const TraceLayoutHost shared_layout = make_trace_layout_host(opt, config, frequency_views.front());
    H5::Group rays = file.createGroup("/rays");
    write_trace_rays_group(rays, opt, frequency_views.front(), shared_layout);

    H5::Group trace = file.createGroup("/trace");
    write_h5_string(trace, "layout", "shared_geometry_multifrequency");
    H5::Group shared = trace.createGroup("shared");
    write_trace_geometry_group(shared, opt, config, frequency_views.front(), shared_layout);

    for (size_t i = 0; i < frequencies.size(); ++i) {
        const TraceLayoutHost layout = make_trace_layout_host(opt, config, frequency_views[i]);
        if (!same_trace_sample_layout(shared_layout, layout)) {
            throw std::runtime_error("multi-frequency trace sample layout mismatch despite shared step control");
        }
        H5::Group freq_group = trace.createGroup("freq_" + std::to_string(i));
        write_h5_int_attr(freq_group, "freq_index", static_cast<int>(i));
        write_h5_scalar(freq_group, "frequency_hz", frequencies[i]);
        H5::Group frequency_rays = freq_group.createGroup("rays");
        write_trace_rays_group(frequency_rays, opt, frequency_views[i], layout);
        write_trace_frequency_group(freq_group, opt, config, frequency_views[i], layout);
        H5::Group derived = freq_group.createGroup("derived");
        write_trace_derived_group(
            derived, opt, config, frequency_views[i], layout);
    }

    return write_timer.seconds();
}


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

template<class RealT>
kpolaris::GRMHDRadiationModel<RealT> load_iharm_model_from_hdf5(const Options& opt) {
    if (opt.iharm_dump.empty()) throw std::runtime_error("--iharm_dump is required for --model=iharm");
    H5::H5File file(opt.iharm_dump, H5F_ACC_RDONLY);
    kpolaris::GRMHDRadiationModel<RealT> model;
    model.n1 = read_h5_scalar_value<int>(file, "/header/n1");
    model.n2 = read_h5_scalar_value<int>(file, "/header/n2");
    model.n3 = read_h5_scalar_value<int>(file, "/header/n3");
    model.spin = read_h5_scalar_value<RealT>(file, "/header/a");
    model.gam = h5_path_exists(file, "/header/gam") ? read_h5_scalar_value<RealT>(file, "/header/gam") : read_h5_scalar_value<RealT>(file, "/header/gamma");
    model.startx1 = read_h5_scalar_value<RealT>(file, "/header/geom/startx1");
    model.startx2 = read_h5_scalar_value<RealT>(file, "/header/geom/startx2");
    model.startx3 = read_h5_scalar_value<RealT>(file, "/header/geom/startx3");
    model.dx1 = read_h5_scalar_value<RealT>(file, "/header/geom/dx1");
    model.dx2 = read_h5_scalar_value<RealT>(file, "/header/geom/dx2");
    model.dx3 = read_h5_scalar_value<RealT>(file, "/header/geom/dx3");
    std::string geom_group = "/header/geom/fmks";
    if (!h5_path_exists(file, geom_group + "/hslope")) geom_group = "/header/geom/mmks";
    model.hslope = read_h5_scalar_value<RealT>(file, geom_group + "/hslope");
    model.mks_smooth = read_h5_scalar_value<RealT>(file, geom_group + "/mks_smooth");
    model.poly_alpha = read_h5_scalar_value<RealT>(file, geom_group + "/poly_alpha");
    model.poly_xt = read_h5_scalar_value<RealT>(file, geom_group + "/poly_xt");
    model.r_in = read_h5_scalar_value<RealT>(file, geom_group + "/r_in");
    model.r_out = read_h5_scalar_value<RealT>(file, geom_group + "/r_out");
    const RealT pi = RealT(3.141592653589793238462643383279502884);
    model.poly_norm = RealT(0.5) * pi / (RealT(1) + RealT(1) / (model.poly_alpha + RealT(1)) / Kokkos::pow(model.poly_xt, model.poly_alpha));
    model.freq_cgs = opt.freq;
    model.M_unit = opt.iharm_M_unit;
    model.mbh_solar = opt.iharm_mbh_solar;
    model.trat_small = opt.iharm_trat_small;
    model.trat_large = opt.iharm_trat_large;
    model.beta_crit = opt.iharm_beta_crit;
    model.sigma_cut = opt.iharm_sigma_cut;
    model.sigma_cut_high = opt.iharm_sigma_cut_high;
    model.emission_type = opt.emission_type > 0 ? opt.emission_type : 4;
    apply_nonthermal_options(model, opt);
    model.has_derived_scalars = opt.iharm_interpolate_derived_scalars;

    H5::DataSet prims_ds = file.openDataSet("/prims");
    H5::DataSpace space = prims_ds.getSpace();
    hsize_t dims[4] = {0, 0, 0, 0};
    if (space.getSimpleExtentNdims() != 4) throw std::runtime_error("iharm /prims dataset must be rank 4");
    space.getSimpleExtentDims(dims);
    if (static_cast<int>(dims[0]) != model.n1 || static_cast<int>(dims[1]) != model.n2 ||
        static_cast<int>(dims[2]) != model.n3 || dims[3] < 8) {
        throw std::runtime_error("iharm /prims dimensions do not match header or have fewer than 8 primitives");
    }
    std::vector<float> raw(static_cast<size_t>(dims[0]) * dims[1] * dims[2] * dims[3]);
    prims_ds.read(raw.data(), H5::PredType::NATIVE_FLOAT);
    model.prims = Kokkos::View<RealT*>("iharm_prims", static_cast<size_t>(8) * model.n1 * model.n2 * model.n3);
    auto host = Kokkos::create_mirror_view(model.prims);
    const size_t nprim_file = static_cast<size_t>(dims[3]);
    for (int i = 0; i < model.n1; ++i) {
        for (int j = 0; j < model.n2; ++j) {
            for (int k = 0; k < model.n3; ++k) {
                const size_t base = (((static_cast<size_t>(i) * model.n2 + j) * model.n3 + k) * nprim_file);
                for (int v = 0; v < 8; ++v) host(model.prim_index(v, i, j, k)) = static_cast<RealT>(raw[base + static_cast<size_t>(v)]);
            }
        }
    }
    Kokkos::deep_copy(model.prims, host);

    model.derived_scalars = Kokkos::View<RealT*>("iharm_derived_scalars", static_cast<size_t>(kpolaris::GRMHDRadiationModel<RealT>::NumDerivedScalars) * model.n1 * model.n2 * model.n3);
    auto derived_host = Kokkos::create_mirror_view(model.derived_scalars);
    const RealT mp = RealT(1.67262171e-24);
    const RealT me = RealT(9.1093826e-28);
    const RealT mp_me = mp / me;
    const RealT game = RealT(4) / RealT(3);
    const RealT gamp = RealT(5) / RealT(3);
    for (int i = 0; i < model.n1; ++i) {
        for (int j = 0; j < model.n2; ++j) {
            const RealT x1 = model.startx1 + (RealT(i) + RealT(0.5)) * model.dx1;
            const RealT x2 = model.startx2 + (RealT(j) + RealT(0.5)) * model.dx2;
            const RealT r = Kokkos::exp(x1);
            const RealT th = model.fmks_theta_from_x2(x1, x2);
            RealT gcov_nat[kpolaris::ndim][kpolaris::ndim];
            RealT gcon_nat[kpolaris::ndim][kpolaris::ndim];
            model.native_metric(x1, x2, r, th, gcov_nat, gcon_nat);
            for (int k = 0; k < model.n3; ++k) {
                const RealT rho = host(model.prim_index(0, i, j, k));
                const RealT uu = host(model.prim_index(1, i, j, k));
                const RealT U1 = host(model.prim_index(2, i, j, k));
                const RealT U2 = host(model.prim_index(3, i, j, k));
                const RealT U3 = host(model.prim_index(4, i, j, k));
                const RealT B1 = host(model.prim_index(5, i, j, k));
                const RealT B2 = host(model.prim_index(6, i, j, k));
                const RealT B3 = host(model.prim_index(7, i, j, k));
                kpolaris::Vec4<RealT> vnat(RealT(0), U1, U2, U3);
                RealT spatial_norm = RealT(0);
                for (int a = 1; a < kpolaris::ndim; ++a) for (int b = 1; b < kpolaris::ndim; ++b) spatial_norm += gcov_nat[a][b] * vnat[a] * vnat[b];
                const RealT vfac = Kokkos::sqrt(std::max<RealT>(-RealT(1) / gcon_nat[0][0] * (RealT(1) + std::abs(spatial_norm)), RealT(0)));
                kpolaris::Vec4<RealT> ucon;
                ucon[0] = -vfac * gcon_nat[0][0];
                for (int a = 1; a < kpolaris::ndim; ++a) ucon[a] = vnat[a] - vfac * gcon_nat[0][a];
                kpolaris::Vec4<RealT> ucov;
                for (int a = 0; a < kpolaris::ndim; ++a) { RealT sum = RealT(0); for (int b = 0; b < kpolaris::ndim; ++b) sum += gcov_nat[a][b] * ucon[b]; ucov[a] = sum; }
                const kpolaris::Vec4<RealT> Bnat(RealT(0), B1, B2, B3);
                RealT udotB = RealT(0);
                for (int a = 1; a < kpolaris::ndim; ++a) udotB += ucov[a] * Bnat[a];
                kpolaris::Vec4<RealT> bcon;
                bcon[0] = udotB;
                for (int a = 1; a < kpolaris::ndim; ++a) bcon[a] = (Bnat[a] + ucon[a] * udotB) / std::max<RealT>(ucon[0], RealT(1e-300));
                kpolaris::Vec4<RealT> bcov;
                for (int a = 0; a < kpolaris::ndim; ++a) { RealT sum = RealT(0); for (int b = 0; b < kpolaris::ndim; ++b) sum += gcov_nat[a][b] * bcon[b]; bcov[a] = sum; }
                RealT bsq = RealT(0);
                for (int a = 0; a < kpolaris::ndim; ++a) bsq += bcon[a] * bcov[a];
                bsq = std::max<RealT>(std::abs(bsq), RealT(1e-40));
                const RealT beta = (model.gam - RealT(1)) * uu / (RealT(0.5) * bsq);
                const RealT sigma = bsq / std::max<RealT>(rho, RealT(1e-300));
                const RealT betasq = beta * beta / std::max<RealT>(model.beta_crit * model.beta_crit, RealT(1e-40));
                const RealT trat = (model.trat_large * betasq + model.trat_small) / (RealT(1) + betasq);
                const RealT thetae_unit = mp_me * (game - RealT(1)) * (gamp - RealT(1)) / ((gamp - RealT(1)) + (game - RealT(1)) * trat);
                derived_host(model.derived_index(kpolaris::GRMHDRadiationModel<RealT>::DerivedNe, i, j, k)) = rho * model.rho_unit_cgs() / (mp + me) * model.ne_factor;
                derived_host(model.derived_index(kpolaris::GRMHDRadiationModel<RealT>::DerivedThetae, i, j, k)) = thetae_unit * uu / std::max<RealT>(rho, RealT(1e-300));
                derived_host(model.derived_index(kpolaris::GRMHDRadiationModel<RealT>::DerivedB, i, j, k)) = Kokkos::sqrt(bsq) * model.b_unit_cgs();
                derived_host(model.derived_index(kpolaris::GRMHDRadiationModel<RealT>::DerivedSigma, i, j, k)) = sigma;
                derived_host(model.derived_index(kpolaris::GRMHDRadiationModel<RealT>::DerivedBeta, i, j, k)) = beta;
            }
        }
    }
    Kokkos::deep_copy(model.derived_scalars, derived_host);
    return model;
}

kpolaris::PassAParams<Real> build_pass_a(const Options& opt) {
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
    pass_a.camera.model = opt.camera == "parallel_plane" ? kpolaris::CameraModel::ParallelPlane : kpolaris::CameraModel::Pinhole;
    pass_a.coordinate_system = coordinate_system_from_name(opt.coordinate);
    if (pass_a.coordinate_system == kpolaris::CoordinateSystem::BoyerLindquist && pass_a.camera.model == kpolaris::CameraModel::ParallelPlane) {
        throw std::runtime_error("parallel_plane camera is currently defined only with cartesian_ks coordinates");
    }
    pass_a.mass = Real(1);
    pass_a.spin = opt.spin;
    pass_a.inner_radius = opt.inner_radius;
    pass_a.outer_radius = opt.outer_radius;
    pass_a.step = opt.step;
    pass_a.max_steps = opt.max_steps;
    pass_a.adaptive = opt.adaptive;
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
    return pass_a;
}


#if KPOLARIS_ENABLE_SLOW_LIGHT

std::vector<std::string> parse_string_list(const std::string& text) {
    std::vector<std::string> out;
    std::string item;
    std::stringstream ss(text);
    while (std::getline(ss, item, ',')) {
        item = trim_copy(item);
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

[[maybe_unused]] std::vector<Real> parse_real_list(
    const std::string& text, const std::string& label) {
    std::vector<Real> out;
    std::string item;
    std::stringstream ss(text);
    while (std::getline(ss, item, ',')) {
        item = trim_copy(item);
        if (!item.empty()) out.push_back(Real(std::stod(item)));
    }
    if (out.empty()) throw std::runtime_error(label + " must not be empty");
    return out;
}

std::string format_indexed_dump_path(const std::string& pattern, int index) {
    const size_t placeholder = pattern.find("{}");
    if (placeholder != std::string::npos) {
        return pattern.substr(0, placeholder) + std::to_string(index) + pattern.substr(placeholder + 2);
    }
    const size_t percent = pattern.find('%');
    if (percent != std::string::npos) {
        int needed = std::snprintf(nullptr, 0, pattern.c_str(), index);
        if (needed <= 0) throw std::runtime_error("failed to format slow_light_dump_pattern");
        std::vector<char> buf(static_cast<size_t>(needed) + 1);
        std::snprintf(buf.data(), buf.size(), pattern.c_str(), index);
        return std::string(buf.data());
    }
    throw std::runtime_error("slow_light_dump_pattern must contain '{}' or printf-style integer format");
}

[[maybe_unused]] std::vector<std::string> build_slow_light_dump_paths(const Options& opt) {
    std::vector<std::string> paths;
    if (!opt.slow_light_dump_list.empty()) {
        paths = parse_string_list(opt.slow_light_dump_list);
    } else if (!opt.slow_light_dump_pattern.empty()) {
        if (opt.slow_light_dump_end < opt.slow_light_dump_start) {
            throw std::runtime_error("slow_light_dump_end must be >= slow_light_dump_start");
        }
        if (opt.slow_light_dump_stride <= 0) {
            throw std::runtime_error("slow_light_dump_stride must be positive");
        }
        for (int i = opt.slow_light_dump_start; i <= opt.slow_light_dump_end; i += opt.slow_light_dump_stride) {
            paths.push_back(format_indexed_dump_path(opt.slow_light_dump_pattern, i));
        }
    }
    if (paths.size() < 2) {
        throw std::runtime_error("slow_light trace requires at least two dumps via slow_light_dump_list or slow_light_dump_pattern");
    }
    return paths;
}

[[maybe_unused]] Real read_iharm_dump_time_for_trace(const std::string& path) {
    H5::H5File file(path, H5F_ACC_RDONLY);
    if (h5_path_exists(file, "/t")) return read_h5_scalar_value<Real>(file, "/t");
    if (h5_path_exists(file, "/header/t")) return read_h5_scalar_value<Real>(file, "/header/t");
    throw std::runtime_error("iHARM dump has no /t or /header/t time dataset: " + path);
}

template<class Reader>
std::vector<Real> build_model_slow_light_dump_times(const Options& opt,
                                                    const std::vector<std::string>& dump_paths,
                                                    Reader read_time) {
    std::vector<Real> times;
    if (!opt.slow_light_time_list.empty()) {
        times = parse_real_list(opt.slow_light_time_list, "slow_light_time_list");
    } else {
        times.reserve(dump_paths.size());
        for (const std::string& path : dump_paths) times.push_back(Real(read_time(path)));
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

kpolaris::TraceConfig<Real> build_trace_config(const Options& opt) {
    kpolaris::TraceConfig<Real> config;
    config.max_samples = opt.max_trace_samples;
    config.sample_stride = opt.trace_stride;
    if (opt.trace_mode == "single") {
        if (opt.ix < 0 || opt.ix >= opt.nx || opt.iy < 0 || opt.iy >= opt.ny) {
            throw std::runtime_error("single trace pixel is outside image grid");
        }
        config.first_pixel = opt.iy * opt.nx + opt.ix;
        config.ray_count = 1;
    } else if (opt.trace_mode == "image" || opt.trace_mode == "all") {
        config.first_pixel = 0;
        config.ray_count = opt.nx * opt.ny;
    } else {
        throw std::runtime_error("unknown trace_mode: " + opt.trace_mode);
    }
    if (config.max_samples <= 0) throw std::runtime_error("max_trace_samples must be positive");
    if (config.sample_stride <= 0) throw std::runtime_error("trace_stride must be positive");
    return config;
}

void apply_trace_field_flags(kpolaris::TraceConfig<Real>& config, const FieldSelection& fields) {
    config.record_lambda = fields.lambda ? 1 : 0;
    config.record_coords = fields.coords ? 1 : 0;
    config.record_plasma = fields.plasma ? 1 : 0;
    config.record_x = fields.x ? 1 : 0;
    config.record_k = fields.k ? 1 : 0;
    config.record_e1 = fields.e1 ? 1 : 0;
    config.record_e2 = fields.e2 ? 1 : 0;
    config.record_coeffs = fields.coeffs ? 1 : 0;
    config.record_stokes = fields.stokes ? 1 : 0;
}

template<class StoreReal, class ExecSpace>
void print_trace_summary_typed(const Options& opt,
                               const kpolaris::TraceConfig<Real>& config,
                               const kpolaris::TraceViews<StoreReal, ExecSpace>& views) {
    const auto counts = copy_view_1d_int(views.sample_count);
    const auto reasons = copy_view_1d_int(views.reason);
    int returned = 0;
    int truncated = 0;
    int max_count = 0;
    for (size_t i = 0; i < counts.size(); ++i) {
        returned += reasons[i] == static_cast<int>(kpolaris::TerminationReason::reached_camera) ? 1 : 0;
        truncated += counts[i] >= config.max_samples ? 1 : 0;
        max_count = std::max(max_count, counts[i]);
    }
    std::cout << "KPolaris trace output\n"
              << "version " << kpolaris::build_info::version << "\n"
              << "revision " << kpolaris::build_info::source_revision << "\n"
              << "source_dirty " << kpolaris::build_info::source_dirty_label() << "\n"
              << "output " << opt.output << "\n"
              << "parameters " << (effective_parameter_output_path(opt).empty() ? "none" : effective_parameter_output_path(opt)) << "\n"
              << "mode " << opt.trace_mode << "\n"
              << "slow_light " << opt.slow_light << "\n"
              << "rays " << config.ray_count << "\n"
              << "returned " << returned << "\n"
              << "max_samples_used " << max_count << "\n"
              << "truncated_rays " << truncated << "\n"
              << "precision " << opt.trace_precision << "\n"
              << "layout " << opt.trace_layout << "\n"
              << "compression " << trace_compression_level(opt.trace_compression) << "\n";
}

template<class RealT, class ExecSpace>
struct SlowLightTraceStateViews {
    Kokkos::View<RealT*, ExecSpace> state_x;
    Kokkos::View<RealT*, ExecSpace> state_k;
    Kokkos::View<RealT*, ExecSpace> state_e1;
    Kokkos::View<RealT*, ExecSpace> state_e2;
    Kokkos::View<RealT*, ExecSpace> stokes_i;
    Kokkos::View<RealT*, ExecSpace> stokes_q;
    Kokkos::View<RealT*, ExecSpace> stokes_u;
    Kokkos::View<RealT*, ExecSpace> stokes_v;
    Kokkos::View<RealT*, ExecSpace> h_current;
    Kokkos::View<RealT*, ExecSpace> radiation_step_cap;
    Kokkos::View<RealT*, ExecSpace> lambda;
    Kokkos::View<int*, ExecSpace> active;
};

template<class ExecSpace, class RealT>
SlowLightTraceStateViews<RealT, ExecSpace> allocate_slow_light_trace_state_views(int nray) {
    SlowLightTraceStateViews<RealT, ExecSpace> v;
    v.state_x = Kokkos::View<RealT*, ExecSpace>("slow_trace_state_x", nray * kpolaris::ndim);
    v.state_k = Kokkos::View<RealT*, ExecSpace>("slow_trace_state_k", nray * kpolaris::ndim);
    v.state_e1 = Kokkos::View<RealT*, ExecSpace>("slow_trace_state_e1", nray * kpolaris::ndim);
    v.state_e2 = Kokkos::View<RealT*, ExecSpace>("slow_trace_state_e2", nray * kpolaris::ndim);
    v.stokes_i = Kokkos::View<RealT*, ExecSpace>("slow_trace_stokes_i", nray);
    v.stokes_q = Kokkos::View<RealT*, ExecSpace>("slow_trace_stokes_q", nray);
    v.stokes_u = Kokkos::View<RealT*, ExecSpace>("slow_trace_stokes_u", nray);
    v.stokes_v = Kokkos::View<RealT*, ExecSpace>("slow_trace_stokes_v", nray);
    v.h_current = Kokkos::View<RealT*, ExecSpace>("slow_trace_h_current", nray);
    v.radiation_step_cap = Kokkos::View<RealT*, ExecSpace>("slow_trace_radiation_step_cap", nray);
    v.lambda = Kokkos::View<RealT*, ExecSpace>("slow_trace_lambda", nray);
    v.active = Kokkos::View<int*, ExecSpace>("slow_trace_active", nray);
    return v;
}

template<class ExecSpace, class StoreReal, class Metric>
void initialize_slow_light_trace_states_metric(
    const kpolaris::PassAParams<Real>& params,
    const kpolaris::TraceConfig<Real>& config,
    const Metric& metric,
    const kpolaris::TraceViews<StoreReal, ExecSpace>& trace_views,
    SlowLightTraceStateViews<Real, ExecSpace>& state_views) {
    Kokkos::parallel_for(
        "KPOLARISSlowTraceInit",
        Kokkos::RangePolicy<ExecSpace>(0, config.ray_count),
        KOKKOS_LAMBDA(const int ray) {
            const int pixel = config.first_pixel + ray;
            trace_views.pixel(ray) = pixel;
            trace_views.sample_count(ray) = 0;
            trace_views.pass_a_steps(ray) = 0;
            trace_views.pass_b_steps(ray) = 0;
            trace_views.reason(ray) = static_cast<int>(kpolaris::TerminationReason::max_steps);
            trace_views.closure_x(ray) = StoreReal(0);
            trace_views.closure_k(ray) = StoreReal(0);
            trace_views.final_null(ray) = StoreReal(0);
            trace_views.frame_error(ray) = StoreReal(0);
            trace_views.final_propagated_i(ray) = StoreReal(0);
            trace_views.final_propagated_q(ray) = StoreReal(0);
            trace_views.final_propagated_u(ray) = StoreReal(0);
            trace_views.final_propagated_v(ray) = StoreReal(0);
            trace_views.final_observed_i(ray) = StoreReal(0);
            trace_views.final_observed_q(ray) = StoreReal(0);
            trace_views.final_observed_u(ray) = StoreReal(0);
            trace_views.final_observed_v(ray) = StoreReal(0);

            const auto endpoint = kpolaris::trace_pass_a_segment_endpoint_pixel_metric(pixel, params, metric);
            const auto camera_state = kpolaris::initialize_camera_ray(metric, pixel, params.camera);
            kpolaris::TransportState<Real> state = endpoint.state;
            trace_views.pass_a_steps(ray) = endpoint.steps;
            trace_views.reason(ray) = static_cast<int>(endpoint.reason);
            if (!endpoint.valid) {
                trace_views.closure_x(ray) = static_cast<StoreReal>(kpolaris::spatial_distance(state.x, camera_state.x));
                trace_views.closure_k(ray) = static_cast<StoreReal>(kpolaris::vector_max_abs_difference(state.k, camera_state.k));
                trace_views.final_null(ray) = static_cast<StoreReal>(metric.dot(state.x, state.k, state.k));
                trace_views.frame_error(ray) = static_cast<StoreReal>(kpolaris::max_frame_error(kpolaris::frame_errors(metric, state)));
                state_views.active(ray) = 0;
            } else {
                state_views.active(ray) = 1;
            }
            const int base = ray * kpolaris::ndim;
            for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                state_views.state_x(base + mu) = state.x[mu];
                state_views.state_k(base + mu) = state.k[mu];
                state_views.state_e1(base + mu) = state.e1[mu];
                state_views.state_e2(base + mu) = state.e2[mu];
            }
            state_views.stokes_i(ray) = Real(0);
            state_views.stokes_q(ray) = Real(0);
            state_views.stokes_u(ray) = Real(0);
            state_views.stokes_v(ray) = Real(0);
            state_views.h_current(ray) = params.step;
            state_views.radiation_step_cap(ray) = params.max_radiation_step > Real(0) ? params.max_radiation_step : params.max_step;
            state_views.lambda(ray) = Real(0);
        });
    Kokkos::fence();
}

template<class ExecSpace, class StoreReal, class Metric, class TemporalModel>
void run_slow_light_trace_window_metric(
    const kpolaris::PassAParams<Real>& params,
    const kpolaris::TraceConfig<Real>& config,
    const Metric& metric,
    const TemporalModel& radiation_model,
    Real window_upper_time,
    const kpolaris::TraceViews<StoreReal, ExecSpace>& trace_views,
    SlowLightTraceStateViews<Real, ExecSpace>& state_views) {
    Kokkos::parallel_for(
        "KPOLARISSlowTraceWindow",
        Kokkos::RangePolicy<ExecSpace>(0, config.ray_count),
        KOKKOS_LAMBDA(const int ray) {
            if (!state_views.active(ray)) return;
            const int pixel = trace_views.pixel(ray);
            const int base = ray * kpolaris::ndim;
            kpolaris::TransportState<Real> state;
            for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                state.x[mu] = state_views.state_x(base + mu);
                state.k[mu] = state_views.state_k(base + mu);
                state.e1[mu] = state_views.state_e1(base + mu);
                state.e2[mu] = state_views.state_e2(base + mu);
            }
            kpolaris::Stokes<Real> stokes(state_views.stokes_i(ray), state_views.stokes_q(ray),
                                          state_views.stokes_u(ray), state_views.stokes_v(ray));
            const auto camera_state = kpolaris::initialize_camera_ray(metric, pixel, params.camera);
            const int screen_orientation = kpolaris::backward_screen_orientation_sign(params.camera);
            kpolaris::AdaptiveRK4Control<Real> transfer_control;
            transfer_control.tolerance = params.adaptive_tolerance;
            transfer_control.min_step = params.min_step;
            transfer_control.max_step = params.max_step;
            Real h_current = state_views.h_current(ray);
            Real radiation_step_cap = state_views.radiation_step_cap(ray);
            Real lambda = state_views.lambda(ray);
            int pass_b_steps = trace_views.pass_b_steps(ray);
            int sample_count = trace_views.sample_count(ray);
            int reached_camera = 0;
            const int sample_stride = config.sample_stride > 0 ? config.sample_stride : 1;
            const int max_samples = config.max_samples > 0 ? config.max_samples : 0;
            const Real dlambda_scale = radiation_model.dlambda_scale();

            while (!reached_camera && pass_b_steps < params.max_steps) {
                Real h = h_current;
                const kpolaris::TransportState<Real> old_state = state;
                const Real old_sample_time = radiation_model.observation_time + old_state.x[0];
                if (old_sample_time >= window_upper_time) break;
                const Real start_r_bl = kpolaris::radial_coordinate(metric, old_state.x);
                if (start_r_bl <= params.outer_radius && radiation_step_cap > Real(0)) {
                    h = kpolaris::min_val(h, radiation_step_cap);
                }
                kpolaris::TransportState<Real> next_state;
                kpolaris::TransportState<Real> sample_state;
                if (params.adaptive) {
                    const auto proposed = kpolaris::adaptive_rk4_step(metric, old_state, h, transfer_control);
                    if (!proposed.accepted) {
                        h_current = kpolaris::abs_val(proposed.next_h);
                        if (h_current <= params.min_step * Real(1.0001)) {
                            trace_views.reason(ray) = static_cast<int>(kpolaris::TerminationReason::max_steps);
                            state_views.active(ray) = 0;
                            break;
                        }
                        continue;
                    }
                    h = kpolaris::abs_val(proposed.used_h);
                    h_current = kpolaris::abs_val(proposed.next_h);
                    next_state = proposed.state;
                    sample_state = proposed.mid_state;
                } else {
                    next_state = kpolaris::rk4_step(metric, old_state, h);
                    sample_state = next_state;
                }

                const Real s0 = kpolaris::camera_surface_value(metric, params.camera, old_state);
                const Real s1 = kpolaris::camera_surface_value(metric, params.camera, next_state);
                int crosses_camera = kpolaris::camera_surface_crossed(s0, s1);
                Real camera_frac = Real(2);
                if (crosses_camera) camera_frac = kpolaris::camera_surface_crossing_fraction(s0, s1);
                const Real trial_next_sample_time = radiation_model.observation_time + next_state.x[0];
                const int crosses_time = (trial_next_sample_time > window_upper_time &&
                                          trial_next_sample_time > old_sample_time);
                Real time_frac = Real(2);
                if (crosses_time) {
                    time_frac = kpolaris::min_val(Real(1), kpolaris::max_val(Real(0),
                        (window_upper_time - old_sample_time) / (trial_next_sample_time - old_sample_time)));
                }

                if (crosses_time && (!crosses_camera || time_frac < camera_frac)) {
                    crosses_camera = 0;
                    const Real h_boundary = h * time_frac;
                    if (kpolaris::abs_val(h_boundary) <= Real(1e-14)) break;
                    h = h_boundary;
                    if (params.adaptive) {
                        kpolaris::AdaptiveRK4Control<Real> boundary_control = transfer_control;
                        boundary_control.min_step = kpolaris::min_val(boundary_control.min_step, kpolaris::abs_val(h));
                        boundary_control.max_step = kpolaris::max_val(boundary_control.min_step, kpolaris::abs_val(h));
                        const auto boundary_step = kpolaris::adaptive_rk4_step(metric, old_state, h, boundary_control);
                        if (!boundary_step.accepted) {
                            h_current = kpolaris::abs_val(boundary_step.next_h);
                            continue;
                        }
                        h = kpolaris::abs_val(boundary_step.used_h);
                        next_state = boundary_step.state;
                        sample_state = boundary_step.mid_state;
                    } else {
                        next_state = kpolaris::rk4_step(metric, old_state, h);
                        sample_state = next_state;
                    }
                    // Match image transport: a boundary step can round back to
                    // the same coordinate time. Never add repeated emission at
                    // that event; pass a rounding-sized remainder to the next window.
                    if (next_state.x[0] == old_state.x[0]) {
                        const Real time_scale = kpolaris::max_val(
                            kpolaris::abs_val(old_state.x[0]),
                            kpolaris::max_val(kpolaris::abs_val(window_upper_time),
                                              kpolaris::abs_val(radiation_model.observation_time)));
                        const Real rounding = Real(0.5) * kpolaris::adaptive_tolerance_floor<Real>() * time_scale;
                        if (kpolaris::abs_val(window_upper_time - old_sample_time) > rounding) {
                            trace_views.reason(ray) = static_cast<int>(kpolaris::TerminationReason::adaptive_step_underflow);
                            state_views.active(ray) = 0;
                        }
                        break;
                    }
                } else if (crosses_camera) {
                    const Real h_cross = h * camera_frac;
                    if (kpolaris::abs_val(h_cross) <= Real(1e-14)) {
                        h = Real(0);
                        next_state = old_state;
                        sample_state = old_state;
                    } else if (camera_frac < Real(0.999999999999)) {
                        h = h_cross;
                        if (params.adaptive) {
                            kpolaris::AdaptiveRK4Control<Real> crossing_control = transfer_control;
                            crossing_control.min_step = kpolaris::min_val(crossing_control.min_step, kpolaris::abs_val(h));
                            crossing_control.max_step = kpolaris::max_val(crossing_control.min_step, kpolaris::abs_val(h));
                            const auto crossing_step = kpolaris::adaptive_rk4_step(metric, old_state, h, crossing_control);
                            if (!crossing_step.accepted) {
                                h_current = kpolaris::abs_val(crossing_step.next_h);
                                continue;
                            }
                            h = kpolaris::abs_val(crossing_step.used_h);
                            next_state = crossing_step.state;
                            sample_state = crossing_step.mid_state;
                        } else {
                            next_state = kpolaris::rk4_step(metric, old_state, h);
                            sample_state = next_state;
                        }
                    }
                }

                const Real old_r_bl = kpolaris::radial_coordinate(metric, old_state.x);
                const Real sample_r_bl = kpolaris::radial_coordinate(metric, sample_state.x);
                const Real next_r_bl = kpolaris::radial_coordinate(metric, next_state.x);
                kpolaris::TransferCoeffs<Real> coeffs;
                if (sample_r_bl >= params.inner_radius && sample_r_bl < params.outer_radius) {
                    coeffs = radiation_model.coefficients(metric, sample_state, Real(0.5));
                    kpolaris::transform_axial_coefficients_to_screen_orientation(coeffs, screen_orientation);
                }
                const Real local_max_radiation_step =
                    (params.max_radiation_step > Real(0) &&
                     (old_r_bl <= params.outer_radius || sample_r_bl <= params.outer_radius ||
                      next_r_bl <= params.outer_radius)) ? params.max_radiation_step : Real(0);
                const int required_steps = kpolaris::radiation_substep_count(
                    coeffs, h, dlambda_scale, 1,
                    local_max_radiation_step, params.max_radiation_depth,
                    params.max_absorption_depth, params.max_faraday_depth);
                if (required_steps > 1 && h > params.min_step * Real(1.0001)) {
                    h_current = kpolaris::max_val(params.min_step, h / Real(required_steps));
                    radiation_step_cap = h_current;
                    continue;
                }
                if (local_max_radiation_step > Real(0) || params.max_radiation_depth > Real(0) ||
                    params.max_absorption_depth > Real(0) || params.max_faraday_depth > Real(0)) {
                    const int predictive_steps = kpolaris::radiation_substep_count(
                        coeffs, h_current, dlambda_scale, 1,
                        local_max_radiation_step, params.max_radiation_depth,
                        params.max_absorption_depth, params.max_faraday_depth);
                    radiation_step_cap = predictive_steps > 1 ?
                        kpolaris::max_val(params.min_step, h_current / Real(predictive_steps)) : params.max_step;
                } else {
                    radiation_step_cap = params.max_step;
                }

                if ((pass_b_steps % sample_stride) == 0 && sample_count < max_samples) {
                    kpolaris::record_trace_sample(metric, radiation_model, sample_state, coeffs, stokes,
                                                  lambda + Real(0.5) * h,
                                                  kpolaris::abs_val(h) * dlambda_scale,
                                                  ray, sample_count, config, trace_views);
                    ++sample_count;
                }
                if (h > Real(0)) {
                    kpolaris::semi_analytic_stokes_step(stokes, coeffs,
                                                        kpolaris::abs_val(h) * dlambda_scale);
                }
                state = next_state;
                lambda += h;
                ++pass_b_steps;
                if (crosses_camera) reached_camera = 1;
            }

            for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                state_views.state_x(base + mu) = state.x[mu];
                state_views.state_k(base + mu) = state.k[mu];
                state_views.state_e1(base + mu) = state.e1[mu];
                state_views.state_e2(base + mu) = state.e2[mu];
            }
            state_views.stokes_i(ray) = stokes.I;
            state_views.stokes_q(ray) = stokes.Q;
            state_views.stokes_u(ray) = stokes.U;
            state_views.stokes_v(ray) = stokes.V;
            state_views.h_current(ray) = h_current;
            state_views.radiation_step_cap(ray) = radiation_step_cap;
            state_views.lambda(ray) = lambda;
            trace_views.pass_b_steps(ray) = pass_b_steps;
            trace_views.sample_count(ray) = sample_count;

            if (reached_camera || pass_b_steps >= params.max_steps || !state_views.active(ray)) {
                if (reached_camera) {
                    trace_views.reason(ray) = static_cast<int>(kpolaris::TerminationReason::reached_camera);
                } else if (state_views.active(ray)) {
                    trace_views.reason(ray) = static_cast<int>(kpolaris::TerminationReason::max_steps);
                }
                trace_views.closure_x(ray) = static_cast<StoreReal>(kpolaris::spatial_distance(state.x, camera_state.x));
                kpolaris::Vec4<Real> target_k;
                for (int mu = 0; mu < kpolaris::ndim; ++mu) target_k[mu] = camera_state.k[mu];
                trace_views.closure_k(ray) = static_cast<StoreReal>(kpolaris::vector_max_abs_difference(state.k, target_k));
                trace_views.final_null(ray) = static_cast<StoreReal>(metric.dot(state.x, state.k, state.k));
                trace_views.frame_error(ray) = static_cast<StoreReal>(kpolaris::max_frame_error(kpolaris::frame_errors(metric, state)));
                trace_views.final_propagated_i(ray) = static_cast<StoreReal>(stokes.I);
                trace_views.final_propagated_q(ray) = static_cast<StoreReal>(stokes.Q);
                trace_views.final_propagated_u(ray) = static_cast<StoreReal>(stokes.U);
                trace_views.final_propagated_v(ray) = static_cast<StoreReal>(stokes.V);
                const auto overlap = kpolaris::screen_overlap(metric, state.x, camera_state, state);
                const auto observed = kpolaris::transform_to_observer_basis(stokes, overlap);
                trace_views.final_observed_i(ray) = static_cast<StoreReal>(observed.I);
                trace_views.final_observed_q(ray) = static_cast<StoreReal>(observed.Q);
                trace_views.final_observed_u(ray) = static_cast<StoreReal>(observed.U);
                trace_views.final_observed_v(ray) = static_cast<StoreReal>(observed.V);
                state_views.active(ray) = 0;
            }
        });
    Kokkos::fence();
}

template<class ExecSpace, class StoreReal, class Metric>
void finalize_slow_light_trace_unfinished_metric(
    const kpolaris::PassAParams<Real>& params,
    const Metric& metric,
    const kpolaris::TraceViews<StoreReal, ExecSpace>& trace_views,
    SlowLightTraceStateViews<Real, ExecSpace>& state_views) {
    Kokkos::parallel_for(
        "KPOLARISSlowTraceFinalizeUnfinished",
        Kokkos::RangePolicy<ExecSpace>(0, static_cast<int>(state_views.active.extent(0))),
        KOKKOS_LAMBDA(const int ray) {
            if (!state_views.active(ray)) return;
            const int pixel = trace_views.pixel(ray);
            const int base = ray * kpolaris::ndim;
            kpolaris::TransportState<Real> state;
            for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                state.x[mu] = state_views.state_x(base + mu);
                state.k[mu] = state_views.state_k(base + mu);
                state.e1[mu] = state_views.state_e1(base + mu);
                state.e2[mu] = state_views.state_e2(base + mu);
            }
            const auto camera_state = kpolaris::initialize_camera_ray(metric, pixel, params.camera);
            trace_views.reason(ray) = static_cast<int>(kpolaris::TerminationReason::slow_light_time_exhausted);
            trace_views.closure_x(ray) = static_cast<StoreReal>(kpolaris::spatial_distance(state.x, camera_state.x));
            kpolaris::Vec4<Real> target_k;
            for (int mu = 0; mu < kpolaris::ndim; ++mu) target_k[mu] = camera_state.k[mu];
            trace_views.closure_k(ray) = static_cast<StoreReal>(kpolaris::vector_max_abs_difference(state.k, target_k));
            trace_views.final_null(ray) = static_cast<StoreReal>(metric.dot(state.x, state.k, state.k));
            trace_views.frame_error(ray) = static_cast<StoreReal>(kpolaris::max_frame_error(kpolaris::frame_errors(metric, state)));
            const kpolaris::Stokes<Real> stokes(
                state_views.stokes_i(ray), state_views.stokes_q(ray),
                state_views.stokes_u(ray), state_views.stokes_v(ray));
            trace_views.final_propagated_i(ray) = static_cast<StoreReal>(stokes.I);
            trace_views.final_propagated_q(ray) = static_cast<StoreReal>(stokes.Q);
            trace_views.final_propagated_u(ray) = static_cast<StoreReal>(stokes.U);
            trace_views.final_propagated_v(ray) = static_cast<StoreReal>(stokes.V);
            const auto overlap = kpolaris::screen_overlap(metric, state.x, camera_state, state);
            const auto observed = kpolaris::transform_to_observer_basis(stokes, overlap);
            trace_views.final_observed_i(ray) = static_cast<StoreReal>(observed.I);
            trace_views.final_observed_q(ray) = static_cast<StoreReal>(observed.Q);
            trace_views.final_observed_u(ray) = static_cast<StoreReal>(observed.U);
            trace_views.final_observed_v(ray) = static_cast<StoreReal>(observed.V);
            state_views.active(ray) = 0;
        });
    Kokkos::fence();
}

template<class StoreReal, class Metric, class RadiationModel, class LoadModel>
void run_slow_light_trace_metric(const Options& opt,
                                 const kpolaris::PassAParams<Real>& pass_a,
                                 const std::vector<std::string>& dump_paths,
                                 const std::vector<Real>& dump_times,
                                 RadiationModel lower,
                                 LoadModel load_model,
                                 const Metric& metric) {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    const FieldSelection fields = parse_fields(opt.trace_fields);
    kpolaris::TraceConfig<Real> config = build_trace_config(opt);
    apply_trace_field_flags(config, fields);
    auto trace_views = allocate_trace_views<StoreReal, ExecSpace>(fields, config.ray_count, config.max_samples);
    auto state_views = allocate_slow_light_trace_state_views<ExecSpace, Real>(config.ray_count);
    initialize_slow_light_trace_states_metric<ExecSpace>(pass_a, config, metric, trace_views, state_views);

    for (size_t i = 0; i + 1 < dump_paths.size(); ++i) {
        RadiationModel upper = load_model(dump_paths[i + 1]);
        kpolaris_image_detail::SlowLightTemporalModel<Real, RadiationModel> temporal;
        temporal.lower = lower;
        temporal.upper = upper;
        temporal.lower_time = dump_times[i];
        temporal.upper_time = dump_times[i + 1];
        temporal.observation_time = opt.slow_light_observation_time;
        temporal.freq_cgs = lower.freq_cgs;
        temporal.interpolation = opt.slow_light_interpolation;
        kpolaris_image_detail::validate_slow_light_interpolation(lower, upper, temporal.interpolation);
        run_slow_light_trace_window_metric<ExecSpace>(pass_a, config, metric, temporal,
                                                      dump_times[i + 1], trace_views, state_views);
        lower = upper;
        const int active = kpolaris_image_detail::count_active_rays<ExecSpace, Real>(state_views.active);
        if (active == 0) break;
    }
    finalize_slow_light_trace_unfinished_metric<ExecSpace>(pass_a, metric, trace_views, state_views);
    const double write_seconds = write_trace_file(opt, pass_a, config, trace_views);
    write_effective_parameter_file(opt, pass_a, {opt.freq});
    print_trace_summary_typed(opt, config, trace_views);
    if (opt.timing) {
        std::cout << "trace_write_seconds " << write_seconds << "\n";
    }
}

template<class StoreReal, class RadiationModel, class LoadModel>
void run_grmhd_slow_light_trace_coordinate(const Options& opt,
                                           const kpolaris::PassAParams<Real>& pass_a,
                                           const std::vector<std::string>& dump_paths,
                                           const std::vector<Real>& dump_times,
                                           RadiationModel first_model,
                                           LoadModel load_model) {
    switch (pass_a.coordinate_system) {
    case kpolaris::CoordinateSystem::CartesianKS: {
        kpolaris::KerrSchildInMetric<Real> metric(pass_a.mass, pass_a.spin);
        run_slow_light_trace_metric<StoreReal>(opt, pass_a, dump_paths, dump_times, first_model, load_model, metric);
        return;
    }
    case kpolaris::CoordinateSystem::SphericalKS: {
        kpolaris::KerrSchildSphericalMetric<Real> metric(pass_a.mass, pass_a.spin);
        run_slow_light_trace_metric<StoreReal>(opt, pass_a, dump_paths, dump_times, first_model, load_model, metric);
        return;
    }
    case kpolaris::CoordinateSystem::BoyerLindquist: {
        kpolaris::KerrBoyerLindquistMetric<Real> metric(pass_a.mass, pass_a.spin);
        run_slow_light_trace_metric<StoreReal>(opt, pass_a, dump_paths, dump_times, first_model, load_model, metric);
        return;
    }
    case kpolaris::CoordinateSystem::FMKS: {
        kpolaris::KerrFMKSMetric<Real> metric(pass_a.mass, pass_a.spin, pass_a.fmks_startx1,
                                             pass_a.fmks_hslope, pass_a.fmks_mks_smooth,
                                             pass_a.fmks_poly_alpha, pass_a.fmks_poly_xt,
                                             pass_a.fmks_poly_norm);
        run_slow_light_trace_metric<StoreReal>(opt, pass_a, dump_paths, dump_times, first_model, load_model, metric);
        return;
    }
    case kpolaris::CoordinateSystem::MKS: {
        kpolaris::KerrSchildSphericalMetric<Real> metric(pass_a.mass, pass_a.spin);
        run_slow_light_trace_metric<StoreReal>(opt, pass_a, dump_paths, dump_times, first_model, load_model, metric);
        return;
    }
    }
    throw std::runtime_error("unknown slow-light coordinate system");
}

template<class StoreReal>
void run_iharm_slow_light_trace(const Options& opt,
                                const kpolaris::GRMHDRadiationModel<Real>& initial_model,
                                const kpolaris::PassAParams<Real>& pass_a) {
    const std::vector<std::string> dump_paths = build_slow_light_dump_paths(opt);
    const std::vector<Real> dump_times = build_model_slow_light_dump_times(
        opt, dump_paths, [](const std::string& path) { return read_iharm_dump_time_for_trace(path); });
    auto load_model = [&](const std::string& path) {
        Options load_opt = opt;
        load_opt.iharm_dump = path;
        return load_iharm_model_from_hdf5<Real>(load_opt);
    };
    kpolaris::GRMHDRadiationModel<Real> first_model = initial_model;
    first_model.freq_cgs = opt.freq;
    if (opt.iharm_dump.empty() || opt.iharm_dump != dump_paths.front()) first_model = load_model(dump_paths.front());
    run_grmhd_slow_light_trace_coordinate<StoreReal>(opt, pass_a, dump_paths, dump_times, first_model, load_model);
}

[[maybe_unused]] kpolaris::KHARMALoadOptions make_kharma_trace_load_options(const Options& opt, const std::string& path) {
    kpolaris::KHARMALoadOptions load_opt;
    load_opt.ddc = opt.kharma_ddc;
    load_opt.dump_path = path;
    load_opt.freq = opt.freq;
    load_opt.M_unit = opt.kharma_M_unit;
    load_opt.mbh_solar = opt.kharma_mbh_solar;
    load_opt.trat_small = opt.kharma_trat_small;
    load_opt.trat_large = opt.kharma_trat_large;
    load_opt.beta_crit = opt.kharma_beta_crit;
    load_opt.sigma_cut = opt.kharma_sigma_cut;
    load_opt.sigma_cut_high = opt.kharma_sigma_cut_high;
    load_opt.emission_type = opt.emission_type > 0 ? opt.emission_type : 4;
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
    load_opt.reverse_field = opt.kharma_reverse_field;
    return load_opt;
}

template<class StoreReal>
void run_kharma_slow_light_trace(const Options& opt,
                                 const kpolaris::GRMHDRadiationModel<Real>& initial_model,
                                 const kpolaris::PassAParams<Real>& pass_a) {
    const std::vector<std::string> dump_paths = build_slow_light_dump_paths(opt);
    const std::vector<Real> dump_times = build_model_slow_light_dump_times(
        opt, dump_paths, [&opt](const std::string& path) { return kpolaris::read_kharma_dump_time(path, opt.kharma_ddc); });
    auto load_model = [&](const std::string& path) {
        auto load_opt = make_kharma_trace_load_options(opt, path);
        const auto found = std::find(dump_paths.begin(), dump_paths.end(), path);
        if (found != dump_paths.end()) load_opt.ddc_expected_time = dump_times.at(static_cast<size_t>(found - dump_paths.begin()));
        return kpolaris::load_kharma_model_from_phdf(load_opt);
    };
    kpolaris::GRMHDRadiationModel<Real> first_model = initial_model;
    first_model.freq_cgs = opt.freq;
    if (opt.kharma_ddc.native || opt.kharma_dump.empty() || opt.kharma_dump != dump_paths.front()) first_model = load_model(dump_paths.front());
    run_grmhd_slow_light_trace_coordinate<StoreReal>(opt, pass_a, dump_paths, dump_times, first_model, load_model);
}

[[maybe_unused]] kpolaris::AthenaKLoadOptions make_athenak_trace_load_options(const Options& opt, const std::string& path) {
    kpolaris::AthenaKLoadOptions load_opt;
    load_opt.dump_path = path;
    load_opt.freq = opt.freq;
    load_opt.M_unit = opt.athenak_M_unit;
    load_opt.mbh_solar = opt.athenak_mbh_solar;
    load_opt.trat_small = opt.athenak_trat_small;
    load_opt.trat_large = opt.athenak_trat_large;
    load_opt.beta_crit = opt.athenak_beta_crit;
    load_opt.gamma = opt.athenak_gamma;
    load_opt.sigma_cut = opt.athenak_sigma_cut;
    load_opt.sigma_cut_high = opt.athenak_sigma_cut_high;
    load_opt.emission_type = opt.emission_type > 0 ? opt.emission_type : 4;
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
    return load_opt;
}

template<class StoreReal>
void run_athenak_slow_light_trace(const Options& opt,
                                  kpolaris::AthenaKDirectRadiationModel<Real> initial_model,
                                  const kpolaris::PassAParams<Real>& pass_a) {
    const std::vector<std::string> dump_paths = build_slow_light_dump_paths(opt);
    const std::vector<Real> dump_times = build_model_slow_light_dump_times(
        opt, dump_paths, [](const std::string& path) { return kpolaris::read_athenak_dump_time(path); });
    auto load_model = [&](const std::string& path) {
        auto model = kpolaris::load_athenak_direct_model_from_binary(make_athenak_trace_load_options(opt, path));
        if (opt.outer_radius > Real(0)) model.r_out = std::min(model.r_out, opt.outer_radius);
        return model;
    };
    initial_model.freq_cgs = opt.freq;
    if (opt.outer_radius > Real(0)) initial_model.r_out = std::min(initial_model.r_out, opt.outer_radius);
    if (opt.athenak_dump.empty() || opt.athenak_dump != dump_paths.front()) initial_model = load_model(dump_paths.front());
    kpolaris::KerrSchildInMetric<Real> metric(pass_a.mass, pass_a.spin);
    run_slow_light_trace_metric<StoreReal>(opt, pass_a, dump_paths, dump_times, initial_model, load_model, metric);
}

#endif

[[maybe_unused]] kpolaris::BHACLoadOptions make_bhac_trace_load_options(const Options& opt, const std::string& path) {
    kpolaris::BHACLoadOptions load_opt;
    load_opt.dump_path = path;
    load_opt.freq = opt.freq;
    load_opt.M_unit = opt.bhac_M_unit;
    load_opt.mbh_solar = opt.bhac_mbh_solar;
    load_opt.trat_small = opt.bhac_trat_small;
    load_opt.trat_large = opt.bhac_trat_large;
    load_opt.beta_crit = opt.bhac_beta_crit;
    load_opt.gamma = opt.bhac_gamma;
    load_opt.sigma_cut = opt.bhac_sigma_cut;
    load_opt.sigma_cut_high = opt.bhac_sigma_cut_high;
    load_opt.emission_type = opt.emission_type > 0 ? opt.emission_type : 4;
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

#if KPOLARIS_ENABLE_SLOW_LIGHT
template<class StoreReal>
void run_bhac_slow_light_trace(const Options& opt,
                               kpolaris::BHACAMRRadiationModel<Real> initial_model,
                               const kpolaris::PassAParams<Real>& pass_a) {
    const std::vector<std::string> dump_paths = build_slow_light_dump_paths(opt);
    const std::vector<Real> dump_times = build_model_slow_light_dump_times(
        opt, dump_paths, [](const std::string& path) { return kpolaris::read_bhac_dump_time(path); });
    auto load_model = [&](const std::string& path) {
        auto model = kpolaris::load_bhac_model_from_dat(make_bhac_trace_load_options(opt, path));
        if (opt.outer_radius > Real(0)) model.r_out = std::min(model.r_out, opt.outer_radius);
        return model;
    };
    initial_model.freq_cgs = opt.freq;
    if (opt.outer_radius > Real(0)) initial_model.r_out = std::min(initial_model.r_out, opt.outer_radius);
    if (opt.bhac_dump.empty() || opt.bhac_dump != dump_paths.front()) initial_model = load_model(dump_paths.front());
    run_grmhd_slow_light_trace_coordinate<StoreReal>(opt, pass_a, dump_paths, dump_times, initial_model, load_model);
}
#endif

[[maybe_unused]] kpolaris::HAMRLoadOptions make_hamr_trace_load_options(
    const Options& opt, const std::string& path) {
    kpolaris::HAMRLoadOptions load_opt;
    load_opt.dump_path = path;
    load_opt.freq = opt.freq;
    load_opt.M_unit = opt.hamr_M_unit;
    load_opt.mbh_solar = opt.hamr_mbh_solar;
    load_opt.trat_small = opt.hamr_trat_small;
    load_opt.trat_large = opt.hamr_trat_large;
    load_opt.beta_crit = opt.hamr_beta_crit;
    load_opt.gamma = opt.hamr_gamma;
    load_opt.sigma_cut = opt.hamr_sigma_cut;
    load_opt.sigma_cut_high = opt.hamr_sigma_cut_high;
    load_opt.emission_type = opt.emission_type > 0 ? opt.emission_type : 4;
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

#if KPOLARIS_ENABLE_SLOW_LIGHT
template<class StoreReal>
void run_hamr_slow_light_trace(const Options& opt,
                               kpolaris::HAMRRadiationModel<Real> initial_model,
                               const kpolaris::PassAParams<Real>& pass_a) {
    const std::vector<std::string> dump_paths = build_slow_light_dump_paths(opt);
    const std::vector<Real> dump_times = build_model_slow_light_dump_times(
        opt, dump_paths, [](const std::string& path) { return kpolaris::read_hamr_dump_time(path); });
    auto load_model = [&](const std::string& path) {
        auto model = kpolaris::load_hamr_model_from_dump(make_hamr_trace_load_options(opt, path));
        if (opt.outer_radius > Real(0)) model.r_out = std::min(model.r_out, opt.outer_radius);
        return model;
    };
    initial_model.freq_cgs = opt.freq;
    if (opt.outer_radius > Real(0)) initial_model.r_out = std::min(initial_model.r_out, opt.outer_radius);
    if (opt.hamr_dump.empty() || opt.hamr_dump != dump_paths.front()) initial_model = load_model(dump_paths.front());
    run_grmhd_slow_light_trace_coordinate<StoreReal>(opt, pass_a, dump_paths, dump_times, initial_model, load_model);
}
#endif

template<class StoreReal, class Model>
void run_trace(const Options& opt, const Model& model, const kpolaris::PassAParams<Real>& pass_a) {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    const FieldSelection fields = parse_fields(opt.trace_fields);
    kpolaris::TraceConfig<Real> config = build_trace_config(opt);
    apply_trace_field_flags(config, fields);
    const std::vector<Real> frequencies = build_trace_frequency_grid(opt);

    Kokkos::View<Real*, ExecSpace> step_control_freqs("trace_step_control_frequencies", frequencies.size());
    auto h_freqs = Kokkos::create_mirror_view(step_control_freqs);
    for (size_t i = 0; i < frequencies.size(); ++i) h_freqs(i) = frequencies[i];
    Kokkos::deep_copy(step_control_freqs, h_freqs);

    std::vector<kpolaris::TraceViews<StoreReal, ExecSpace>> frequency_views;
    frequency_views.reserve(frequencies.size());
    double kernel_seconds = 0.0;
    for (size_t i = 0; i < frequencies.size(); ++i) {
        Model freq_model = model;
        freq_model.freq_cgs = frequencies[i];
        auto views = allocate_trace_views<StoreReal, ExecSpace>(fields, config.ray_count, config.max_samples);
        Kokkos::Timer kernel_timer;
        if (frequencies.size() > 1) {
            kpolaris::run_trace_pass_b_segment_model_control<ExecSpace, StoreReal>(
                pass_a, freq_model, config, step_control_freqs.data(),
                static_cast<int>(frequencies.size()), views);
        } else {
            kpolaris::run_trace_pass_b_segment_model<ExecSpace, StoreReal>(pass_a, freq_model, config, views);
        }
        Kokkos::fence();
        kernel_seconds += kernel_timer.seconds();
        frequency_views.push_back(views);
    }

    const double write_seconds = frequencies.size() > 1 ?
        write_multifrequency_trace_file(opt, pass_a, config, frequencies, frequency_views) :
        write_trace_file(opt, pass_a, config, frequency_views.front());
    write_effective_parameter_file(opt, pass_a, frequencies);

    print_trace_summary_typed(opt, config, frequency_views.front());
    std::cout << "frequencies " << frequencies.size() << "\n";
    if (opt.timing) {
        std::cout << "trace_kernel_seconds " << kernel_seconds << "\n"
                  << "trace_write_seconds " << write_seconds << "\n";
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
        opt.camera = canonical_camera_name(opt.camera);
        opt.coordinate = canonical_coordinate_name(opt.coordinate);
        validate_options(opt);
        resolve_camera_extents(opt);
        validate_resolved_camera(opt);
        const Real horizon = Real(1) + std::sqrt(std::max<Real>(Real(0), Real(1) - opt.spin * opt.spin));
        if (opt.inner_radius <= Real(0) && !model_defines_inner_radius(opt.model)) {
            opt.inner_radius = horizon * Real(1.05);
        }
#if !KPOLARIS_ENABLE_SLOW_LIGHT
        if (opt.slow_light) throw std::runtime_error("slow_light requested but KPOLARIS_ENABLE_SLOW_LIGHT is OFF");
#endif
        if (opt.slow_light && opt.nfreq > 1) {
            throw std::runtime_error("multi-frequency slow-light trace is not implemented yet; use fast-light trace or a single frequency");
        }

#if KPOLARIS_TRACE_ONLY_RIAF
        if (opt.model != "riaf") {
            throw std::runtime_error("this trace executable was built for --model=riaf only");
        }
        if (opt.slow_light) {
            throw std::runtime_error("slow_light trace is supported for iharm, kharma, athenak, bhac, and hamr only");
        }
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
        model.emission_type = opt.emission_type > 0 ? opt.emission_type : 1;
        apply_nonthermal_options(model, opt);
        if (opt.outer_radius <= Real(0)) opt.outer_radius = model.r_max * Real(1.15);
        {
            auto pass_a = build_pass_a(opt);
            if (opt.trace_precision == "double" || opt.trace_precision == "float64") {
                run_trace<double>(opt, model, pass_a);
            } else if (opt.trace_precision == "float" || opt.trace_precision == "float32") {
                run_trace<float>(opt, model, pass_a);
            } else {
                throw std::runtime_error("unknown trace_precision: " + opt.trace_precision);
            }
        }
#elif KPOLARIS_TRACE_ONLY_TORUS
        if (opt.model != "torus") {
            throw std::runtime_error("this trace executable was built for --model=torus only");
        }
        if (opt.slow_light) {
            throw std::runtime_error("slow_light trace is supported for iharm, kharma, athenak, bhac, and hamr only");
        }
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
        if (opt.outer_radius <= Real(0)) opt.outer_radius = model.r_outer * Real(1.15);
        {
            auto pass_a = build_pass_a(opt);
            if (opt.trace_precision == "double" || opt.trace_precision == "float64") {
                run_trace<double>(opt, model, pass_a);
            } else if (opt.trace_precision == "float" || opt.trace_precision == "float32") {
                run_trace<float>(opt, model, pass_a);
            } else {
                throw std::runtime_error("unknown trace_precision: " + opt.trace_precision);
            }
        }
#elif KPOLARIS_TRACE_ONLY_IHARM
        if (opt.model != "iharm") {
            throw std::runtime_error("this trace executable was built for --model=iharm only");
        }
        if (opt.coordinate != "fmks" && opt.coordinate != "mks" && opt.coordinate != "spherical_ks" && opt.coordinate != "cartesian_ks") {
            throw std::runtime_error("iharm currently supports --coordinate=mks, fmks, spherical_ks, or cartesian_ks");
        }
#if KPOLARIS_ENABLE_SLOW_LIGHT
        if (opt.slow_light && opt.iharm_dump.empty()) opt.iharm_dump = build_slow_light_dump_paths(opt).front();
#endif
        kpolaris::GRMHDRadiationModel<Real> model = load_iharm_model_from_hdf5<Real>(opt);
        opt.spin = model.spin;
        opt.fmks_startx1 = model.startx1;
        opt.fmks_hslope = model.hslope;
        opt.fmks_mks_smooth = model.mks_smooth;
        opt.fmks_poly_alpha = model.poly_alpha;
        opt.fmks_poly_xt = model.poly_xt;
        opt.fmks_poly_norm = model.poly_norm;
        opt.inner_radius = safe_inner_radius(opt.inner_radius, opt.inner_radius_explicit,
                                            model.r_in, model.spin);
        if (opt.outer_radius <= Real(0)) opt.outer_radius = Real(100);
        auto pass_a = build_pass_a(opt);
        if (opt.trace_precision == "double" || opt.trace_precision == "float64") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
            if (opt.slow_light) run_iharm_slow_light_trace<double>(opt, model, pass_a);
            else
#endif
            run_trace<double>(opt, model, pass_a);
        } else if (opt.trace_precision == "float" || opt.trace_precision == "float32") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
            if (opt.slow_light) run_iharm_slow_light_trace<float>(opt, model, pass_a);
            else
#endif
            run_trace<float>(opt, model, pass_a);
        } else throw std::runtime_error("unknown trace_precision: " + opt.trace_precision);
#elif KPOLARIS_TRACE_ONLY_KHARMA
        if (opt.model != "kharma") {
            throw std::runtime_error("this trace executable was built for --model=kharma only");
        }
        if (opt.coordinate != "fmks" && opt.coordinate != "mks" && opt.coordinate != "spherical_ks" && opt.coordinate != "cartesian_ks") {
            throw std::runtime_error("kharma currently supports --coordinate=mks, fmks, spherical_ks, or cartesian_ks");
        }
#if KPOLARIS_ENABLE_SLOW_LIGHT
        if (opt.slow_light && opt.kharma_dump.empty()) opt.kharma_dump = build_slow_light_dump_paths(opt).front();
#endif
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
        load_opt.emission_type = opt.emission_type > 0 ? opt.emission_type : 4;
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
        load_opt.reverse_field = opt.kharma_reverse_field;
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
        opt.inner_radius = safe_inner_radius(opt.inner_radius, opt.inner_radius_explicit,
                                            model.r_in, model.spin);
        if (opt.outer_radius <= Real(0)) {
            opt.outer_radius = opt.camera == "pinhole" ? std::min(model.r_out, opt.radius * Real(0.9)) : model.r_out;
        }
        auto pass_a = build_pass_a(opt);
        if (opt.trace_precision == "double" || opt.trace_precision == "float64") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
            if (opt.slow_light) run_kharma_slow_light_trace<double>(opt, model, pass_a);
            else
#endif
            run_trace<double>(opt, model, pass_a);
        } else if (opt.trace_precision == "float" || opt.trace_precision == "float32") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
            if (opt.slow_light) run_kharma_slow_light_trace<float>(opt, model, pass_a);
            else
#endif
            run_trace<float>(opt, model, pass_a);
        } else throw std::runtime_error("unknown trace_precision: " + opt.trace_precision);
#elif KPOLARIS_TRACE_ONLY_BHAC
        if (opt.model != "bhac") {
            throw std::runtime_error("this trace executable was built for --model=bhac only");
        }
        if (opt.coordinate != "spherical_ks" && opt.coordinate != "cartesian_ks") {
            throw std::runtime_error("bhac currently supports --coordinate=spherical_ks or cartesian_ks");
        }
#if KPOLARIS_ENABLE_SLOW_LIGHT
        if (opt.slow_light && opt.bhac_dump.empty()) opt.bhac_dump = build_slow_light_dump_paths(opt).front();
#endif
        kpolaris::BHACAMRRadiationModel<Real> model =
            kpolaris::load_bhac_model_from_dat(make_bhac_trace_load_options(opt, opt.bhac_dump));
        opt.spin = model.spin;
        if (opt.outer_radius <= Real(0)) {
            opt.outer_radius = opt.camera == "pinhole" ? std::min(model.r_out, opt.radius * Real(0.9)) : model.r_out;
        }
        model.r_out = std::min(model.r_out, opt.outer_radius);
        opt.inner_radius = safe_inner_radius(opt.inner_radius, opt.inner_radius_explicit,
                                            model.r_in, model.spin);
        auto pass_a = build_pass_a(opt);
        if (opt.trace_precision == "double" || opt.trace_precision == "float64") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
            if (opt.slow_light) run_bhac_slow_light_trace<double>(opt, model, pass_a);
            else
#endif
            run_trace<double>(opt, model, pass_a);
        } else if (opt.trace_precision == "float" || opt.trace_precision == "float32") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
            if (opt.slow_light) run_bhac_slow_light_trace<float>(opt, model, pass_a);
            else
#endif
            run_trace<float>(opt, model, pass_a);
        } else throw std::runtime_error("unknown trace_precision: " + opt.trace_precision);
#elif KPOLARIS_TRACE_ONLY_HAMR
        if (opt.model != "hamr") {
            throw std::runtime_error("this trace executable was built for --model=hamr only");
        }
        if (opt.coordinate != "spherical_ks" && opt.coordinate != "cartesian_ks") {
            throw std::runtime_error("hamr currently supports --coordinate=spherical_ks or cartesian_ks");
        }
#if KPOLARIS_ENABLE_SLOW_LIGHT
        if (opt.slow_light && opt.hamr_dump.empty()) opt.hamr_dump = build_slow_light_dump_paths(opt).front();
#endif
        warn_pending_hamr_hslope(opt);
        kpolaris::HAMRRadiationModel<Real> model =
            kpolaris::load_hamr_model_from_dump(make_hamr_trace_load_options(opt, opt.hamr_dump));
        opt.spin = model.spin;
        if (opt.outer_radius <= Real(0)) {
            opt.outer_radius = opt.camera == "pinhole" ? std::min(model.r_out, opt.radius * Real(0.9)) : model.r_out;
        }
        model.r_out = std::min(model.r_out, opt.outer_radius);
        opt.inner_radius = safe_inner_radius(opt.inner_radius, opt.inner_radius_explicit,
                                            model.r_in, model.spin);
        auto pass_a = build_pass_a(opt);
        if (opt.trace_precision == "double" || opt.trace_precision == "float64") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
            if (opt.slow_light) run_hamr_slow_light_trace<double>(opt, model, pass_a);
            else
#endif
            run_trace<double>(opt, model, pass_a);
        } else if (opt.trace_precision == "float" || opt.trace_precision == "float32") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
            if (opt.slow_light) run_hamr_slow_light_trace<float>(opt, model, pass_a);
            else
#endif
            run_trace<float>(opt, model, pass_a);
        } else throw std::runtime_error("unknown trace_precision: " + opt.trace_precision);
#else
#if !KPOLARIS_TRACE_ONLY_ATHENAK
        if (opt.model == "riaf") {
            if (opt.slow_light) throw std::runtime_error("slow_light trace is supported for iharm, kharma, athenak, bhac, and hamr only");
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
            model.emission_type = opt.emission_type > 0 ? opt.emission_type : 1;
            apply_nonthermal_options(model, opt);
            if (opt.outer_radius <= Real(0)) opt.outer_radius = model.r_max * Real(1.15);
            auto pass_a = build_pass_a(opt);
            if (opt.trace_precision == "double" || opt.trace_precision == "float64") run_trace<double>(opt, model, pass_a);
            else if (opt.trace_precision == "float" || opt.trace_precision == "float32") run_trace<float>(opt, model, pass_a);
            else throw std::runtime_error("unknown trace_precision: " + opt.trace_precision);
        } else if (opt.model == "torus") {
            if (opt.slow_light) throw std::runtime_error("slow_light trace is supported for iharm, kharma, athenak, bhac, and hamr only");
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
            if (opt.outer_radius <= Real(0)) opt.outer_radius = model.r_outer * Real(1.15);
            auto pass_a = build_pass_a(opt);
            if (opt.trace_precision == "double" || opt.trace_precision == "float64") run_trace<double>(opt, model, pass_a);
            else if (opt.trace_precision == "float" || opt.trace_precision == "float32") run_trace<float>(opt, model, pass_a);
            else throw std::runtime_error("unknown trace_precision: " + opt.trace_precision);
        } else if (opt.model == "iharm") {
            if (opt.coordinate != "fmks" && opt.coordinate != "mks" && opt.coordinate != "spherical_ks" && opt.coordinate != "cartesian_ks") {
                throw std::runtime_error("iharm currently supports --coordinate=mks, fmks, spherical_ks, or cartesian_ks");
            }
#if KPOLARIS_ENABLE_SLOW_LIGHT
            if (opt.slow_light && opt.iharm_dump.empty()) opt.iharm_dump = build_slow_light_dump_paths(opt).front();
#endif
            kpolaris::GRMHDRadiationModel<Real> model = load_iharm_model_from_hdf5<Real>(opt);
            opt.spin = model.spin;
            opt.fmks_startx1 = model.startx1;
            opt.fmks_hslope = model.hslope;
            opt.fmks_mks_smooth = model.mks_smooth;
            opt.fmks_poly_alpha = model.poly_alpha;
            opt.fmks_poly_xt = model.poly_xt;
            opt.fmks_poly_norm = model.poly_norm;
            opt.inner_radius = safe_inner_radius(opt.inner_radius, opt.inner_radius_explicit,
                                                model.r_in, model.spin);
            if (opt.outer_radius <= Real(0)) opt.outer_radius = Real(100);
            auto pass_a = build_pass_a(opt);
            if (opt.trace_precision == "double" || opt.trace_precision == "float64") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
                if (opt.slow_light) run_iharm_slow_light_trace<double>(opt, model, pass_a);
                else
#endif
                run_trace<double>(opt, model, pass_a);
            } else if (opt.trace_precision == "float" || opt.trace_precision == "float32") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
                if (opt.slow_light) run_iharm_slow_light_trace<float>(opt, model, pass_a);
                else
#endif
                run_trace<float>(opt, model, pass_a);
            } else throw std::runtime_error("unknown trace_precision: " + opt.trace_precision);
        } else if (opt.model == "kharma") {
            if (opt.coordinate != "fmks" && opt.coordinate != "mks" && opt.coordinate != "spherical_ks" && opt.coordinate != "cartesian_ks") {
                throw std::runtime_error("kharma currently supports --coordinate=mks, fmks, spherical_ks, or cartesian_ks");
            }
#if KPOLARIS_ENABLE_SLOW_LIGHT
            if (opt.slow_light && opt.kharma_dump.empty()) opt.kharma_dump = build_slow_light_dump_paths(opt).front();
#endif
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
            load_opt.emission_type = opt.emission_type > 0 ? opt.emission_type : 4;
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
            load_opt.reverse_field = opt.kharma_reverse_field;
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
            opt.inner_radius = safe_inner_radius(opt.inner_radius, opt.inner_radius_explicit,
                                                model.r_in, model.spin);
            if (opt.outer_radius <= Real(0)) {
                opt.outer_radius = opt.camera == "pinhole" ? std::min(model.r_out, opt.radius * Real(0.9)) : model.r_out;
            }
            auto pass_a = build_pass_a(opt);
            if (opt.trace_precision == "double" || opt.trace_precision == "float64") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
                if (opt.slow_light) run_kharma_slow_light_trace<double>(opt, model, pass_a);
                else
#endif
                run_trace<double>(opt, model, pass_a);
            } else if (opt.trace_precision == "float" || opt.trace_precision == "float32") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
                if (opt.slow_light) run_kharma_slow_light_trace<float>(opt, model, pass_a);
                else
#endif
                run_trace<float>(opt, model, pass_a);
            } else throw std::runtime_error("unknown trace_precision: " + opt.trace_precision);
        } else if (opt.model == "bhac") {
            if (opt.coordinate != "spherical_ks" && opt.coordinate != "cartesian_ks") {
                throw std::runtime_error("bhac currently supports --coordinate=spherical_ks or cartesian_ks");
            }
#if KPOLARIS_ENABLE_SLOW_LIGHT
            if (opt.slow_light && opt.bhac_dump.empty()) opt.bhac_dump = build_slow_light_dump_paths(opt).front();
#endif
            kpolaris::BHACAMRRadiationModel<Real> model =
                kpolaris::load_bhac_model_from_dat(make_bhac_trace_load_options(opt, opt.bhac_dump));
            opt.spin = model.spin;
            if (opt.outer_radius <= Real(0)) {
                opt.outer_radius = opt.camera == "pinhole" ? std::min(model.r_out, opt.radius * Real(0.9)) : model.r_out;
            }
            model.r_out = std::min(model.r_out, opt.outer_radius);
            opt.inner_radius = safe_inner_radius(opt.inner_radius, opt.inner_radius_explicit,
                                                model.r_in, model.spin);
            auto pass_a = build_pass_a(opt);
            if (opt.trace_precision == "double" || opt.trace_precision == "float64") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
                if (opt.slow_light) run_bhac_slow_light_trace<double>(opt, model, pass_a);
                else
#endif
                run_trace<double>(opt, model, pass_a);
            } else if (opt.trace_precision == "float" || opt.trace_precision == "float32") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
                if (opt.slow_light) run_bhac_slow_light_trace<float>(opt, model, pass_a);
                else
#endif
                run_trace<float>(opt, model, pass_a);
            } else throw std::runtime_error("unknown trace_precision: " + opt.trace_precision);
        } else if (opt.model == "hamr") {
            if (opt.coordinate != "spherical_ks" && opt.coordinate != "cartesian_ks") {
                throw std::runtime_error("hamr currently supports --coordinate=spherical_ks or cartesian_ks");
            }
#if KPOLARIS_ENABLE_SLOW_LIGHT
            if (opt.slow_light && opt.hamr_dump.empty()) opt.hamr_dump = build_slow_light_dump_paths(opt).front();
#endif
            warn_pending_hamr_hslope(opt);
            kpolaris::HAMRRadiationModel<Real> model =
                kpolaris::load_hamr_model_from_dump(make_hamr_trace_load_options(opt, opt.hamr_dump));
            opt.spin = model.spin;
            if (opt.outer_radius <= Real(0)) {
                opt.outer_radius = opt.camera == "pinhole" ? std::min(model.r_out, opt.radius * Real(0.9)) : model.r_out;
            }
            model.r_out = std::min(model.r_out, opt.outer_radius);
            opt.inner_radius = safe_inner_radius(opt.inner_radius, opt.inner_radius_explicit,
                                                model.r_in, model.spin);
            auto pass_a = build_pass_a(opt);
            if (opt.trace_precision == "double" || opt.trace_precision == "float64") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
                if (opt.slow_light) run_hamr_slow_light_trace<double>(opt, model, pass_a);
                else
#endif
                run_trace<double>(opt, model, pass_a);
            } else if (opt.trace_precision == "float" || opt.trace_precision == "float32") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
                if (opt.slow_light) run_hamr_slow_light_trace<float>(opt, model, pass_a);
                else
#endif
                run_trace<float>(opt, model, pass_a);
            } else throw std::runtime_error("unknown trace_precision: " + opt.trace_precision);
        } else
#endif
        if (opt.model == "athenak") {
            opt.coordinate = "cartesian_ks";
#if KPOLARIS_ENABLE_SLOW_LIGHT
            if (opt.slow_light && opt.athenak_dump.empty()) opt.athenak_dump = build_slow_light_dump_paths(opt).front();
#endif
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
            load_opt.emission_type = opt.emission_type > 0 ? opt.emission_type : 4;
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
            kpolaris::AthenaKDirectRadiationModel<Real> model = kpolaris::load_athenak_direct_model_from_binary(load_opt);
            opt.spin = model.spin;
            if (opt.outer_radius <= Real(0)) {
                opt.outer_radius = opt.camera == "pinhole" ? std::min(model.r_out, opt.radius * Real(0.9)) : model.r_out;
            }
            model.r_out = std::min(model.r_out, opt.outer_radius);
            opt.inner_radius = safe_inner_radius(opt.inner_radius, opt.inner_radius_explicit,
                                                model.r_in, model.spin);
            auto pass_a = build_pass_a(opt);
            if (opt.trace_precision == "double" || opt.trace_precision == "float64") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
                if (opt.slow_light) run_athenak_slow_light_trace<double>(opt, model, pass_a);
                else
#endif
                run_trace<double>(opt, model, pass_a);
            } else if (opt.trace_precision == "float" || opt.trace_precision == "float32") {
#if KPOLARIS_ENABLE_SLOW_LIGHT
                if (opt.slow_light) run_athenak_slow_light_trace<float>(opt, model, pass_a);
                else
#endif
                run_trace<float>(opt, model, pass_a);
            } else throw std::runtime_error("unknown trace_precision: " + opt.trace_precision);
        } else {
            throw std::runtime_error("unknown model: " + opt.model);
        }
#endif
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
