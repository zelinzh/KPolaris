#include "kpolaris_response_options.hpp"
#include "kpolaris_model_image_options.hpp"
#include "kpolaris_evpa.hpp"

#include <Kokkos_Core.hpp>

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using Real = kpolaris::DefaultReal;
}

std::string canonical_camera_name(const std::string& name) {
    if (name == "pinhole" || name == "parallel_plane") {
        return name;
    }
    throw std::runtime_error("unknown camera: " + name);
}

std::string canonical_coordinate_name(const std::string& name) {
    if (name == "cartesian_ks" || name == "cartesian-kerr-schild" || name == "ks") {
        return "cartesian_ks";
    }
    if (name == "spherical_ks" || name == "spherical-kerr-schild" || name == "ks_spherical" ||
        name == "kerr_schild_spherical" || name == "sks") {
        return "spherical_ks";
    }
    if (name == "mks" || name == "modified_ks" || name == "modified-kerr-schild" ||
        name == "native_mks") {
        return "mks";
    }
    if (name == "fmks" || name == "funky_modified_ks" || name == "funky-modified-kerr-schild" ||
        name == "native_fmks" || name == "mmks") {
        return "fmks";
    }
    if (name == "boyer_lindquist" || name == "boyer-lindquist" || name == "bl") {
        return "boyer_lindquist";
    }
    throw std::runtime_error("unknown coordinate: " + name);
}

kpolaris::CoordinateSystem coordinate_system_from_name(const std::string& name) {
    if (name == "boyer_lindquist") return kpolaris::CoordinateSystem::BoyerLindquist;
    if (name == "spherical_ks") return kpolaris::CoordinateSystem::SphericalKS;
    if (name == "fmks") return kpolaris::CoordinateSystem::FMKS;
    if (name == "mks") return kpolaris::CoordinateSystem::MKS;
    return kpolaris::CoordinateSystem::CartesianKS;
}

std::string metric_name_from_coordinate(const std::string& name) {
    if (name == "boyer_lindquist") return "Kerr-Boyer-Lindquist";
    if (name == "spherical_ks") return "Kerr-Spherical-Kerr-Schild";
    if (name == "fmks") return "Kerr-Funky-Modified-Kerr-Schild";
    if (name == "mks") return "Kerr-Spherical-Kerr-Schild with native MKS fluid grid";
    return "Kerr-Cartesian-Kerr-Schild";
}

int emission_type_from_name(const std::string& name) {
    if (name == "symphony" || name == "pandya" || name == "thermal" ||
        name == "symphony_pandya_thermal") {
        return 1;
    }
    if (name == "kappa" || name == "symphony_kappa") {
        return 2;
    }
    if (name == "powerlaw" || name == "power_law" || name == "symphony_powerlaw" || name == "symphony_power_law") {
        return 3;
    }
    if (name == "dexter" || name == "dexter_thermal") {
        return 4;
    }
    throw std::runtime_error("unknown emission fit: " + name);
}

std::string trim_copy(const std::string& input) {
    size_t first = 0;
    while (first < input.size() && std::isspace(static_cast<unsigned char>(input[first]))) {
        ++first;
    }
    size_t last = input.size();
    while (last > first && std::isspace(static_cast<unsigned char>(input[last - 1]))) {
        --last;
    }
    return input.substr(first, last - first);
}

std::string normalize_key(std::string key) {
    key = trim_copy(key);
    while (key.rfind("--", 0) == 0) {
        key.erase(0, 2);
    }
    for (char& c : key) {
        if (c == '-') {
            c = '_';
        }
    }
    return key;
}

std::string normalize_token(std::string value) {
    value = trim_copy(value);
    for (char& c : value) {
        if (c == '-') c = '_';
        else c = static_cast<char>(
            std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
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
    if (opt.repeat_images < 1)
        throw std::runtime_error("repeat_images must be positive");
    if (opt.repeat_images > 1 && (opt.slow_light || opt.slow_light_time_probe || !opt.slow_light_batch_jobs.empty()))
        throw std::runtime_error("repeat_images requires ordinary fast-light imaging");
    if (opt.repeat_images > 1 && opt.parameter_output != "auto" && opt.parameter_output != "none")
        throw std::runtime_error("repeat_images requires parameter_output=auto or none");
    if (opt.model == "kharma") opt.kharma_ddc.validate();
    if (!std::isfinite(opt.equatorial_h_over_r) || opt.equatorial_h_over_r < Real(0) ||
        opt.equatorial_h_over_r > Real(1) ||
        (opt.equatorial_h_over_r > Real(0) && opt.equatorial_h_over_r < Real(1e-6)))
        throw std::runtime_error("equatorial_h_over_r must be 0 (disabled) or in [1e-6,1]");
    if (opt.equatorial_h_over_r > Real(0) &&
        opt.equatorial_h_over_r < Real(1024)*std::numeric_limits<Real>::epsilon())
        throw std::runtime_error("equatorial_h_over_r is too small for this floating-point precision; use a double build");
    if (opt.equatorial_samples < 2 || opt.equatorial_samples > 1024)
        throw std::runtime_error("equatorial_samples must be an integer in [2,1024]");
    if (opt.faraday_rotation != 0 && opt.faraday_rotation != 1)
        throw std::runtime_error("faraday_rotation must be 0 or 1");
    if (opt.equatorial_h_over_r > Real(0) && opt.model == "binary_riaf")
        throw std::runtime_error("equatorial_h_over_r uses a single Kerr spin-axis midplane and is not defined for binary_riaf");
    if (opt.direct_only != 0 && opt.direct_only != 1)
        throw std::runtime_error("direct_only must be 0 or 1");
    if (opt.direct_only && opt.model == "binary_riaf")
        throw std::runtime_error("direct_only uses the single Kerr spin-axis midplane and is not defined for binary_riaf");
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
    if (!finite(opt.outer_radius)) {
        throw std::runtime_error("outer_radius must be finite (use a non-positive value for automatic resolution)");
    }
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
    if (opt.radiation_substeps <= 0) {
        throw std::runtime_error("radiation_substeps must be positive");
    }
    if (opt.multifrequency_chunk_size < 0) {
        throw std::runtime_error("multifrequency_chunk_size must be non-negative");
    }
    if (opt.analysis_radial_bins < 1 ||
        opt.analysis_radial_bins > KPOLARIS_MAX_ANALYSIS_RADIAL_BINS) {
        throw std::runtime_error(
            "analysis_radial_bins must lie in [1, KPOLARIS_MAX_ANALYSIS_RADIAL_BINS]");
    }
    if ((opt.analysis_radial_min > Real(0) && !finite(opt.analysis_radial_min)) ||
        (opt.analysis_radial_max > Real(0) && !finite(opt.analysis_radial_max))) {
        throw std::runtime_error("analysis radial bounds must be finite when specified");
    }
    if (opt.analysis_radial_min > Real(0) && opt.analysis_radial_max > Real(0) &&
        !(opt.analysis_radial_max > opt.analysis_radial_min)) {
        throw std::runtime_error("analysis_radial_max must exceed analysis_radial_min");
    }
    if (!finite(opt.analysis_formation_fraction) ||
        !(opt.analysis_formation_fraction > Real(0)) ||
        !(opt.analysis_formation_fraction < Real(1))) {
        throw std::runtime_error("analysis_formation_fraction must lie strictly between 0 and 1");
    }
    (void)make_response_config(opt, opt.analysis_radial_bins, Real(2), Real(100));
    if (opt.model == "binary_riaf") {
        if (opt.camera != "parallel_plane") {
            throw std::runtime_error("binary_riaf currently requires --camera=parallel_plane; the pinhole camera still assumes a single Kerr center");
        }
        if (opt.coordinate != "cartesian_ks") {
            throw std::runtime_error("binary_riaf requires --coordinate=cartesian_ks");
        }
        if (opt.analysis_mode) {
            throw std::runtime_error("binary_riaf does not yet support analysis_mode because two overlapping fluid components are not representable by the single-fluid diagnostics schema");
        }
        if (!opt.split_transport) {
            throw std::runtime_error("binary_riaf currently requires --split_transport=1");
        }
        if (opt.inner_radius_explicit) {
            throw std::runtime_error(
                "binary_riaf does not use --inner_radius; set --binary_capture_factor for the full-metric radial-characteristic capture surrogate");
        }
        require_positive(opt.riaf_r_min, "riaf_r_min");
        require_positive(opt.riaf_r_max, "riaf_r_max");
        if (!(opt.riaf_r_min < opt.riaf_r_max)) {
            throw std::runtime_error("riaf_r_min must be smaller than riaf_r_max");
        }
        require_positive(opt.riaf_nth0, "riaf_nth0");
        require_positive(opt.riaf_Te0, "riaf_Te0");
        require_positive(opt.riaf_disk_h, "riaf_disk_h");
        require_positive(opt.riaf_ne_unit, "riaf_ne_unit");
        require_positive(opt.riaf_te_unit, "riaf_te_unit");
        require_positive(opt.riaf_mbh_solar, "riaf_mbh_solar");
        if (!finite(opt.riaf_pow_nth) || !finite(opt.riaf_pow_T) ||
            !finite(opt.riaf_keplerian_factor) ||
            !finite(opt.riaf_infall_factor)) {
            throw std::runtime_error(
                "binary RIAF power-law and velocity factors must be finite");
        }
        const bool tabulated_trajectory =
            opt.binary_trajectory_model == "paper_cbwaves_4pn_local" ||
            opt.binary_trajectory_model == "table";
        const bool leading_trajectory =
            opt.binary_trajectory_model == "leading_quadrupole" ||
            opt.binary_trajectory_model == "leading_order" ||
            opt.binary_trajectory_model == "peters";
        if (!tabulated_trajectory && !leading_trajectory) {
            throw std::runtime_error(
                "binary_trajectory_model must be leading_quadrupole, table, or paper_cbwaves_4pn_local");
        }
        if (tabulated_trajectory) {
            if (opt.binary_trajectory_file.empty()) {
                throw std::runtime_error(
                    "tabulated binary trajectories require binary_trajectory_file");
            }
            if (opt.binary_trajectory_format != "auto" &&
                opt.binary_trajectory_format != "native" &&
                opt.binary_trajectory_format != "combi_ressler") {
                throw std::runtime_error(
                    "binary_trajectory_format must be auto, native, or combi_ressler");
            }
            if (opt.binary_trajectory_interpolation != "auto" &&
                opt.binary_trajectory_interpolation != "linear" &&
                opt.binary_trajectory_interpolation !=
                    "cubic_hermite_position_velocity") {
                throw std::runtime_error(
                    "binary_trajectory_interpolation must be auto, linear, or cubic_hermite_position_velocity");
            }
            if (!finite(opt.binary_trajectory_time_offset) ||
                !finite(opt.binary_trajectory_mass_scale) ||
                opt.binary_trajectory_mass_scale == Real(0)) {
                throw std::runtime_error(
                    "binary trajectory time offset and mass scale must be finite; mass scale cannot be zero");
            }
            if (!opt.binary_orbit) {
                throw std::runtime_error(
                    "a tabulated 4PN trajectory requires binary_orbit=1");
            }
        } else {
            require_positive(opt.binary_mass_ratio, "binary_mass_ratio");
            if (!finite(opt.binary_chi1) || !finite(opt.binary_chi2) ||
                std::abs(opt.binary_chi1) > Real(1) ||
                std::abs(opt.binary_chi2) > Real(1)) {
                throw std::runtime_error(
                    "binary_chi1 and binary_chi2 must lie in [-1,1]");
            }
            require_positive(opt.binary_reference_separation,
                             "binary_reference_separation");
            require_positive(opt.binary_minimum_separation,
                             "binary_minimum_separation");
        }
        if (!finite(opt.binary_reference_phase) ||
            !finite(opt.binary_reference_time) ||
            !finite(opt.binary_observation_time)) {
            throw std::runtime_error(
                "binary_reference_phase, binary_reference_time, and binary_observation_time must be finite");
        }
        if (leading_trajectory && opt.binary_minimum_separation < Real(6)) {
            throw std::runtime_error(
                "binary_minimum_separation must be at least 6 M for the built-in leading-quadrupole inspiral trajectory");
        }
        if (leading_trajectory &&
            opt.binary_reference_separation < opt.binary_minimum_separation) {
            throw std::runtime_error("binary_reference_separation must not be below binary_minimum_separation");
        }
        if ((opt.binary_inspiral != 0 && opt.binary_inspiral != 1) ||
            (opt.binary_orbit != 0 && opt.binary_orbit != 1)) {
            throw std::runtime_error("binary_inspiral and binary_orbit must be 0 or 1");
        }
        require_positive(opt.binary_metric_derivative_step, "binary_metric_derivative_step");
        if (!finite(opt.binary_capture_factor) || !(opt.binary_capture_factor > Real(1)) ||
            !(opt.binary_capture_factor <= Real(1.05))) {
            throw std::runtime_error("binary_capture_factor must lie in (1,1.05] for the full-metric radial-characteristic capture surrogate");
        }
        if (!finite(opt.binary_tidal_fraction) || !(opt.binary_tidal_fraction > Real(0)) ||
            !(opt.binary_tidal_fraction <= Real(1))) {
            throw std::runtime_error("binary_tidal_fraction must lie in (0,1]");
        }
        if (!finite(opt.binary_taper_start_fraction) ||
            opt.binary_taper_start_fraction < Real(0) ||
            !(opt.binary_taper_start_fraction < Real(1))) {
            throw std::runtime_error("binary_taper_start_fraction must lie in [0,1)");
        }
        require_positive(opt.binary_density_scale1, "binary_density_scale1");
        require_positive(opt.binary_density_scale2, "binary_density_scale2");
        require_positive(opt.binary_temperature_scale1, "binary_temperature_scale1");
        require_positive(opt.binary_temperature_scale2, "binary_temperature_scale2");
        if (std::abs(opt.binary_field_polarity1) != 1 ||
            std::abs(opt.binary_field_polarity2) != 1) {
            throw std::runtime_error("binary_field_polarity1/2 must be +1 or -1");
        }
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

void print_usage_and_exit() {
    std::cout << "Usage: kpolaris_model_image [--version] [--parameter_file=file.par] [--parameter_output=file.par|auto|none] "
                 "[--model=riaf|binary_riaf|torus|iharm|kharma|athenak|bhac|hamr] [--output=file.h5] [--format=auto|hdf5|csv] [--evpa_0=N|W] "
                 "[--repeat_images=N] [--nx=N] [--ny=N] [--camera=pinhole|parallel_plane] [--coordinate=cartesian_ks|spherical_ks|mks|fmks|boyer_lindquist] "
                 "[--radius=R] [--inclination_rad=i|--inclination_deg=i] "
                 "[--fov=v|--fovy=v|--xspan=M|--yspan=M|--dsource=pc --fovx_dsource=muas --fovy_dsource=muas] [--x_offset=v] [--y_offset=v] "
                 "[--pinhole_pixel_bias=pixels] [--use_pinhole_pixel_bias=0|1] "
                 "[--spin=a] [--inner_radius=r] [--outer_radius=r] "
                 "[--max_steps=N] [--step=h] [--adaptive=0|1] [--adaptive_tolerance=t] "
                 "[--min_step=h] [--max_step=h] [--max_radiation_step=h] "
                 "[--max_radiation_depth=tau] [--max_absorption_depth=tau] [--max_faraday_depth=psi] [--substeps=N] "
                 "[--closure_x_warning=M] [--closure_k_warning=v] "
                 "[--frame_error_warning=v] [--basis_identity_warning=v] [--split_transport=0|1] [--direct_only=0|1] [--equatorial_h_over_r=h --equatorial_samples=N] [--faraday_rotation=0|1] [--analysis_mode=0|1] "
                 "[--analysis_response=none|density_scale|temperature_scale|magnetic_scale|coefficients --analysis_response_step=h] "
                 "[--analysis_partition=none|radial|region|plasma_region|near_far|thetae|sigma|beta|ne_cgs|b_cgs --analysis_partition_edges=a,b,...] "
                 "[--analysis_funnel_angle_deg=20 --analysis_disk_angle_deg=60 --analysis_sigma_boundary=1 --analysis_beta_boundary=1] "
                 "[--analysis_radial_bins=N --analysis_radial_min=M --analysis_radial_max=M --analysis_formation_fraction=f] "
                 "[--slow_light=0|1 --slow_light_interpolation=fluid|coefficients --slow_light_step_mode=decoupled|block|snapshot --slow_light_prefetch=0|1 --slow_light_pipeline=0|1 --slow_light_windows_per_block=N --slow_light_snapshot_cache_gib=GiB --slow_light_prefetch_snapshots=N --slow_light_batch_jobs=jobs.txt --slow_light_time_probe=0|1 --slow_light_dump_list=a.h5,b.h5 --slow_light_time_list=t0,t1 --slow_light_observation_time=tobs] "
                 "[--freq=Hz|--freq_list=a,b,c|--freq_min=a --freq_max=b --nfreq=N] "
                 "[--multifrequency_chunk_size=N]\n"
                 "Slow-light images default to decoupled stepping and fluid interpolation; block/snapshot and coefficients remain optional.\n"
                 "Use resident window counts or a KHARMA snapshot-cache budget to control memory and scheduling.\n"
                 "Parameter files use key=value lines with the same names as long options, comments starting with #, "
                 "and command-line options override file values.\n"
                 "Parameter meanings, units, aliases and scope: docs/parameters.md.\n"
                 "RIAF options: --riaf_r_min --riaf_r_max --riaf_nth0 --riaf_Te0 "
                 "--riaf_disk_h --riaf_pow_nth --riaf_pow_T --riaf_ne_unit "
                 "--riaf_te_unit --riaf_mbh_solar\n"
                 "Binary RIAF options: --binary_mass_ratio --binary_chi1 --binary_chi2 "
                 "--binary_reference_separation --binary_reference_phase --binary_reference_time "
                 "--binary_observation_time --binary_minimum_separation --binary_inspiral=0|1 "
                 "--binary_orbit=0|1 --binary_metric_derivative_step "
                 "--binary_trajectory_model=leading_quadrupole|paper_cbwaves_4pn_local "
                 "--binary_trajectory_file=trajectory.h5 --binary_trajectory_format=auto|native|combi_ressler "
                 "--binary_trajectory_interpolation=auto|linear|cubic_hermite_position_velocity --binary_trajectory_time_offset=M "
                 "--binary_trajectory_mass_scale=M_ref "
                 "--binary_capture_factor --binary_tidal_fraction --binary_taper_start_fraction "
                 "--binary_density_scale1/2 --binary_temperature_scale1/2 "
                 "--binary_field_polarity1/2=+1|-1\n"
                 "Emission options: --emission_type=1|2|3|4 --emission_fit=thermal|kappa|powerlaw|dexter "
                 "--nonthermal_kappa --variable_kappa --powerlaw_p --powerlaw_eta\n"
                 "Torus options: --torus_l_lambda --torus_wwin --torus_kappa "
                 "--torus_omegac --torus_betac --torus_beta --torus_Rhigh "
                 "--torus_bh_mass_solar --torus_mdot_cgs --torus_mdot_code "
                 "--torus_thetae_min --scalar_transport=0|1\n"
                 "GRMHD dump options: --iharm_dump or --kharma_dump or --athenak_dump or --bhac_dump or --hamr_dump "
                 "--iharm_M_unit/--kharma_M_unit/--athenak_M_unit/--bhac_M_unit/--hamr_M_unit --iharm_mbh_solar/--kharma_mbh_solar/--athenak_mbh_solar/--bhac_mbh_solar/--hamr_mbh_solar "
                 "--iharm_trat_small/--kharma_trat_small/--athenak_trat_small/--bhac_trat_small/--hamr_trat_small --iharm_trat_large/--kharma_trat_large/--athenak_trat_large/--bhac_trat_large/--hamr_trat_large "
                 "--iharm_beta_crit/--kharma_beta_crit/--athenak_beta_crit/--bhac_beta_crit/--hamr_beta_crit "
                 "--iharm_sigma_cut/--kharma_sigma_cut/--athenak_sigma_cut/--bhac_sigma_cut/--hamr_sigma_cut --iharm_sigma_cut_high/--kharma_sigma_cut_high/--athenak_sigma_cut_high/--bhac_sigma_cut_high/--hamr_sigma_cut_high "
                 "--bhac_nxlone1 --bhac_nxlone2 --bhac_nxlone3 (0=auto) --bhac_spin_index=-1(auto) --bhac_hslope --bhac_sfc "
                 "--hamr_hslope --hamr_id_order=root_slot|child_major|root_major --hamr_root_order=morton|x3x2x1|x1x2x3|x1x3x2 "
                 "--bhac_cache=staged.bin --bhac_cache_mode=read|write|read_write|off "
                 "--iharm_resample=spherical_ks_primitives|spherical_ks_precomputed "
                 "--kharma_resample=spherical_ks_primitives|spherical_ks_precomputed "
                 "--resample_n1=N --resample_n2=N --resample_n3=N\n";
    if (Kokkos::is_initialized()) {
        Kokkos::finalize();
    }
    std::cout << "KHARMA DDC: --kharma_ddc_native=0|1 --kharma_ddc_socket=/tmp/ddc.sock "
                 "--kharma_ddc_manifest=sequence_manifest.json --kharma_ddc_timeout_seconds=7200\n"
                 "Native frame names: ddc_frame_<sequence>.phdf; service protocol STAGE1.\n";
    std::exit(0);
}

void apply_option(Options& opt, std::string key, const std::string& value,
                  const std::string& source) {
    key = normalize_key(std::move(key));
    try {
        if (opt.kharma_ddc.parse(key, value)) return;
        if (key == "help") {
            print_usage_and_exit();
        } else if (key == "parameter_file" || key == "input" || key == "input_file" ||
                   key == "params" || key == "config") {
            opt.parameter_file = value;
        } else if (key == "parameter_output" || key == "params_output" ||
                   key == "effective_parameters" || key == "effective_parameter_file") {
            opt.parameter_output = value;
        } else if (key == "model") {
            opt.model = value;
        } else if (key == "output") {
            opt.output = value;
        } else if (key == "evpa_0") {
            opt.evpa_0 = parse_evpa_zero(value);
        } else if (key == "format") {
            opt.output_format = value;
        } else if (key == "camera") {
            opt.camera = value;
        } else if (key == "coordinate") {
            opt.coordinate = value;
        } else if (key == "nx") {
            opt.nx = std::stoi(value);
        } else if (key == "ny") {
            opt.ny = std::stoi(value);
        } else if (key == "radius") {
            opt.radius = Real(std::stod(value));
        } else if (key == "inclination_rad") {
            opt.inclination = Real(std::stod(value));
        } else if (key == "inclination_deg") {
            opt.inclination = Real(std::stod(value)) * Real(3.141592653589793238462643383279502884) / Real(180);
        } else if (key == "fov") {
            opt.fov = Real(std::stod(value));
            opt.fovy = Real(-1);
            opt.xspan = Real(-1);
            opt.yspan = Real(-1);
            opt.fovx_dsource = Real(-1);
            opt.fovy_dsource = Real(-1);
        } else if (key == "fovy" || key == "fov_y") {
            opt.fovy = Real(std::stod(value));
            opt.fovy_dsource = Real(-1);
        } else if (key == "xspan") {
            opt.xspan = Real(std::stod(value));
            opt.fovx_dsource = Real(-1);
        } else if (key == "yspan") {
            opt.yspan = Real(std::stod(value));
            opt.fovy_dsource = Real(-1);
        } else if (key == "dsource" || key == "dsource_pc") {
            opt.dsource_pc = Real(std::stod(value));
        } else if (key == "fovx_dsource" || key == "fovx_muas" || key == "fovx_uas") {
            opt.fovx_dsource = Real(std::stod(value));
            opt.xspan = Real(-1);
        } else if (key == "fovy_dsource" || key == "fovy_muas" || key == "fovy_uas") {
            opt.fovy_dsource = Real(std::stod(value));
            opt.yspan = Real(-1);
        } else if (key == "x_offset") {
            opt.x_offset = Real(std::stod(value));
        } else if (key == "y_offset") {
            opt.y_offset = Real(std::stod(value));
        } else if (key == "use_pinhole_pixel_bias") {
            opt.use_pinhole_pixel_bias = std::stoi(value);
        } else if (key == "pinhole_pixel_bias") {
            opt.pinhole_pixel_bias = Real(std::stod(value));
        } else if (key == "spin") {
            opt.spin = Real(std::stod(value));
        } else if (key == "inner_radius") {
            opt.inner_radius = Real(std::stod(value));
            opt.inner_radius_explicit = true;
        } else if (key == "outer_radius") {
            opt.outer_radius = Real(std::stod(value));
        } else if (key == "max_steps") {
            opt.max_steps = std::stoi(value);
        } else if (key == "equatorial_h_over_r") {
            size_t used=0;
            opt.equatorial_h_over_r = Real(std::stod(value, &used));
            if (used != value.size()) throw std::runtime_error("invalid equatorial_h_over_r");
        } else if (key == "equatorial_samples") {
            size_t used=0;
            opt.equatorial_samples = std::stoi(value, &used);
            if (used != value.size()) throw std::runtime_error("equatorial_samples must be an integer");
        } else if (key == "faraday_rotation") {
            if (value != "0" && value != "1") throw std::runtime_error("faraday_rotation must be 0 or 1");
            opt.faraday_rotation = std::stoi(value);
        } else if (key == "direct_only") {
            if (value != "0" && value != "1")
                throw std::runtime_error("direct_only must be 0 or 1");
            opt.direct_only = std::stoi(value);
        } else if (key == "adaptive") {
            opt.adaptive = std::stoi(value);
        } else if (key == "adaptive_tolerance") {
            opt.adaptive_tolerance = Real(std::stod(value));
        } else if (key == "min_step") {
            opt.min_step = Real(std::stod(value));
        } else if (key == "max_step") {
            opt.max_step = Real(std::stod(value));
        } else if (key == "max_radiation_step" || key == "radiation_sampling_interval") {
            opt.max_radiation_step = Real(std::stod(value));
        } else if (key == "max_radiation_depth" || key == "radiation_sampling_depth") {
            opt.max_radiation_depth = Real(std::stod(value));
            opt.max_absorption_depth = opt.max_radiation_depth;
            opt.max_faraday_depth = opt.max_radiation_depth;
        } else if (key == "max_absorption_depth" || key == "absorption_sampling_depth") {
            opt.max_absorption_depth = Real(std::stod(value));
        } else if (key == "max_faraday_depth" || key == "faraday_sampling_depth") {
            opt.max_faraday_depth = Real(std::stod(value));
        } else if (key == "step") {
            opt.step = Real(std::stod(value));
        } else if (key == "substeps" || key == "radiation_substeps") {
            opt.radiation_substeps = std::stoi(value);
        } else if (key == "closure_x_warning" || key == "warn_closure_x") {
            opt.closure_x_warning = Real(std::stod(value));
        } else if (key == "closure_k_warning" || key == "warn_closure_k") {
            opt.closure_k_warning = Real(std::stod(value));
        } else if (key == "frame_error_warning" || key == "warn_frame_error") {
            opt.frame_error_warning = Real(std::stod(value));
        } else if (key == "basis_identity_warning" || key == "warn_basis_identity") {
            opt.basis_identity_warning = Real(std::stod(value));
        } else if (key == "split_transport" || key == "split_passes" || key == "separate_passes") {
            opt.split_transport = std::stoi(value);
        } else if (key == "analysis_mode" || key == "analysis" || key == "physical_diagnostics") {
            opt.analysis_mode = std::stoi(value);
        } else if (key == "analysis_response") {
            opt.analysis_response = value;
        } else if (key == "analysis_response_step") {
            opt.analysis_response_step = Real(std::stod(value));
        } else if (key == "analysis_partition") {
            opt.analysis_partition = value;
        } else if (key == "analysis_partition_edges") {
            opt.analysis_partition_edges = value;
        } else if (key == "analysis_funnel_angle_deg") {
            opt.analysis_funnel_angle_deg = Real(std::stod(value));
        } else if (key == "analysis_disk_angle_deg") {
            opt.analysis_disk_angle_deg = Real(std::stod(value));
        } else if (key == "analysis_sigma_boundary") {
            opt.analysis_sigma_boundary = Real(std::stod(value));
        } else if (key == "analysis_beta_boundary") {
            opt.analysis_beta_boundary = Real(std::stod(value));
        } else if (key == "analysis_radial_bins" || key == "formation_radial_bins") {
            opt.analysis_radial_bins = std::stoi(value);
        } else if (key == "analysis_radial_min" || key == "formation_radial_min") {
            opt.analysis_radial_min = Real(std::stod(value));
        } else if (key == "analysis_radial_max" || key == "formation_radial_max") {
            opt.analysis_radial_max = Real(std::stod(value));
        } else if (key == "analysis_formation_fraction" || key == "formation_fraction") {
            opt.analysis_formation_fraction = Real(std::stod(value));
        } else if (key == "slow_light" || key == "slowlight") {
            opt.slow_light = std::stoi(value);
        } else if (key == "slow_light_step_mode") {
            if (value == "snapshot") opt.slow_light_step_mode = kpolaris::SlowLightStepMode::snapshot;
            else if (value == "block") opt.slow_light_step_mode = kpolaris::SlowLightStepMode::block;
            else if (value == "decoupled" || value == "continuous") opt.slow_light_step_mode = kpolaris::SlowLightStepMode::decoupled;
            else throw std::invalid_argument("slow_light_step_mode must be decoupled, block or snapshot (continuous is an alias for decoupled)");
            opt.slow_light_step_mode_explicit = 1;
        } else if (key == "slow_light_interpolation") {
            if (value == "coefficients") opt.slow_light_interpolation = kpolaris::SlowLightInterpolation::coefficients;
            else if (value == "fluid") opt.slow_light_interpolation = kpolaris::SlowLightInterpolation::fluid;
            else throw std::invalid_argument("slow_light_interpolation must be coefficients or fluid");
        } else if (key == "slow_light_prefetch" || key == "slowlight_prefetch" || key == "prefetch") {
            opt.slow_light_prefetch = std::stoi(value);
        } else if (key == "slow_light_pipeline") {
            opt.slow_light_pipeline = std::stoi(value);
            if (opt.slow_light_pipeline != 0 && opt.slow_light_pipeline != 1)
                throw std::invalid_argument("slow_light_pipeline must be 0 or 1");
        } else if (key == "slow_light_windows_per_block") {
            opt.slow_light_windows_per_block = std::stoi(value);
            opt.slow_light_windows_per_block_explicit = 1;
            if (opt.slow_light_windows_per_block < 1)
                throw std::invalid_argument("slow_light_windows_per_block must be positive");
        } else if (key == "slow_light_snapshot_cache_gib") {
            opt.slow_light_snapshot_cache_gib = Real(std::stod(value));
            if (!std::isfinite(opt.slow_light_snapshot_cache_gib) || opt.slow_light_snapshot_cache_gib < Real(0))
                throw std::invalid_argument("slow_light_snapshot_cache_gib must be finite and nonnegative");
        } else if (key == "slow_light_prefetch_snapshots") {
            opt.slow_light_prefetch_snapshots = std::stoi(value);
            if (opt.slow_light_prefetch_snapshots < 0)
                throw std::invalid_argument("slow_light_prefetch_snapshots must be nonnegative");
        } else if (key == "slow_light_batch_jobs") {
            opt.slow_light_batch_jobs = value;
        } else if (key == "slow_light_time_probe" || key == "slowlight_time_probe" || key == "time_probe") {
            opt.slow_light_time_probe = std::stoi(value);
        } else if (key == "slow_light_observation_time" || key == "slow_light_t_obs" || key == "t_obs") {
            opt.slow_light_observation_time = Real(std::stod(value));
        } else if (key == "slow_light_dump_list" || key == "slow_light_dumps" || key == "slowlight_dumps") {
            opt.slow_light_dump_list = value;
        } else if (key == "slow_light_time_list" || key == "slow_light_times" || key == "slowlight_times") {
            opt.slow_light_time_list = value;
        } else if (key == "slow_light_dump_pattern" || key == "slow_light_pattern" || key == "slowlight_pattern") {
            opt.slow_light_dump_pattern = value;
        } else if (key == "slow_light_dump_start" || key == "slow_light_start" || key == "dump_start") {
            opt.slow_light_dump_start = std::stoi(value);
        } else if (key == "slow_light_dump_end" || key == "slow_light_end" || key == "dump_end") {
            opt.slow_light_dump_end = std::stoi(value);
        } else if (key == "slow_light_dump_stride" || key == "slow_light_stride" || key == "dump_stride") {
            opt.slow_light_dump_stride = std::stoi(value);
        } else if (key == "freq" || key == "frequency_hz") {
            opt.freq = Real(std::stod(value));
        } else if (key == "freq_list" || key == "frequency_list" || key == "frequencies") {
            opt.freq_list = value;
        } else if (key == "freq_min" || key == "frequency_min") {
            opt.freq_min = Real(std::stod(value));
        } else if (key == "freq_max" || key == "frequency_max") {
            opt.freq_max = Real(std::stod(value));
        } else if (key == "nfreq" || key == "frequency_count") {
            opt.nfreq = std::stoi(value);
        } else if (key == "freq_spacing" || key == "frequency_spacing") {
            opt.freq_spacing = value;
        } else if (key == "multifrequency_chunk_size" || key == "frequency_chunk_size" || key == "freq_chunk_size") {
            opt.multifrequency_chunk_size = std::stoi(value);
        } else if (key == "riaf_r_min") {
            opt.riaf_r_min = Real(std::stod(value));
        } else if (key == "riaf_r_max") {
            opt.riaf_r_max = Real(std::stod(value));
        } else if (key == "riaf_nth0") {
            opt.riaf_nth0 = Real(std::stod(value));
        } else if (key == "riaf_Te0" || key == "riaf_te0") {
            opt.riaf_Te0 = Real(std::stod(value));
        } else if (key == "riaf_disk_h") {
            opt.riaf_disk_h = Real(std::stod(value));
        } else if (key == "riaf_pow_nth") {
            opt.riaf_pow_nth = Real(std::stod(value));
        } else if (key == "riaf_pow_T" || key == "riaf_pow_t") {
            opt.riaf_pow_T = Real(std::stod(value));
        } else if (key == "riaf_ne_unit") {
            opt.riaf_ne_unit = Real(std::stod(value));
        } else if (key == "riaf_te_unit") {
            opt.riaf_te_unit = Real(std::stod(value));
        } else if (key == "riaf_mbh_solar") {
            opt.riaf_mbh_solar = Real(std::stod(value));
        } else if (key == "riaf_keplerian_factor") {
            opt.riaf_keplerian_factor = Real(std::stod(value));
        } else if (key == "riaf_infall_factor") {
            opt.riaf_infall_factor = Real(std::stod(value));
        } else if (key == "binary_mass_ratio" || key == "binary_q") {
            opt.binary_mass_ratio = Real(std::stod(value));
        } else if (key == "binary_chi1") {
            opt.binary_chi1 = Real(std::stod(value));
        } else if (key == "binary_chi2") {
            opt.binary_chi2 = Real(std::stod(value));
        } else if (key == "binary_reference_separation" || key == "binary_separation") {
            opt.binary_reference_separation = Real(std::stod(value));
        } else if (key == "binary_reference_phase" || key == "binary_phase") {
            opt.binary_reference_phase = Real(std::stod(value));
        } else if (key == "binary_reference_time") {
            opt.binary_reference_time = Real(std::stod(value));
        } else if (key == "binary_observation_time" || key == "binary_t_obs") {
            opt.binary_observation_time = Real(std::stod(value));
        } else if (key == "binary_minimum_separation") {
            opt.binary_minimum_separation = Real(std::stod(value));
        } else if (key == "binary_inspiral") {
            opt.binary_inspiral = std::stoi(value);
        } else if (key == "binary_orbit") {
            opt.binary_orbit = std::stoi(value);
        } else if (key == "binary_trajectory_model" ||
                   key == "binary_orbit_model") {
            opt.binary_trajectory_model = normalize_token(value);
        } else if (key == "binary_trajectory_file" ||
                   key == "binary_trajectory") {
            opt.binary_trajectory_file = value;
        } else if (key == "binary_trajectory_format") {
            opt.binary_trajectory_format = normalize_token(value);
        } else if (key == "binary_trajectory_interpolation") {
            opt.binary_trajectory_interpolation = normalize_token(value);
        } else if (key == "binary_trajectory_time_offset") {
            opt.binary_trajectory_time_offset = Real(std::stod(value));
        } else if (key == "binary_trajectory_mass_scale") {
            opt.binary_trajectory_mass_scale = Real(std::stod(value));
        } else if (key == "binary_metric_derivative_step") {
            opt.binary_metric_derivative_step = Real(std::stod(value));
        } else if (key == "binary_capture_factor") {
            opt.binary_capture_factor = Real(std::stod(value));
        } else if (key == "binary_tidal_fraction") {
            opt.binary_tidal_fraction = Real(std::stod(value));
        } else if (key == "binary_taper_start_fraction") {
            opt.binary_taper_start_fraction = Real(std::stod(value));
        } else if (key == "binary_density_scale1") {
            opt.binary_density_scale1 = Real(std::stod(value));
        } else if (key == "binary_density_scale2") {
            opt.binary_density_scale2 = Real(std::stod(value));
        } else if (key == "binary_temperature_scale1") {
            opt.binary_temperature_scale1 = Real(std::stod(value));
        } else if (key == "binary_temperature_scale2") {
            opt.binary_temperature_scale2 = Real(std::stod(value));
        } else if (key == "binary_field_polarity1") {
            opt.binary_field_polarity1 = std::stoi(value);
        } else if (key == "binary_field_polarity2") {
            opt.binary_field_polarity2 = std::stoi(value);
        } else if (key == "emission_type") {
            opt.emission_type = std::stoi(value);
        } else if (key == "emission_fit") {
            opt.emission_type = emission_type_from_name(value);
        } else if (key == "kappa" || key == "nonthermal_kappa") {
            opt.nonthermal_kappa = Real(std::stod(value));
        } else if (key == "variable_kappa") {
            opt.variable_kappa = std::stoi(value);
        } else if (key == "variable_kappa_min") {
            opt.variable_kappa_min = Real(std::stod(value));
        } else if (key == "variable_kappa_interp_start") {
            opt.variable_kappa_interp_start = Real(std::stod(value));
        } else if (key == "variable_kappa_max") {
            opt.variable_kappa_max = Real(std::stod(value));
        } else if (key == "powerlaw_p" || key == "power_law_p") {
            opt.powerlaw_p = Real(std::stod(value));
        } else if (key == "powerlaw_eta" || key == "power_law_eta") {
            opt.powerlaw_eta = Real(std::stod(value));
        } else if (key == "powerlaw_gamma_min" || key == "power_law_gamma_min") {
            opt.powerlaw_gamma_min = Real(std::stod(value));
        } else if (key == "powerlaw_gamma_max" || key == "power_law_gamma_max") {
            opt.powerlaw_gamma_max = Real(std::stod(value));
        } else if (key == "powerlaw_gamma_cutoff" || key == "power_law_gamma_cutoff") {
            opt.powerlaw_gamma_cutoff = Real(std::stod(value));
        } else if (key == "torus_l_lambda") {
            opt.torus_l_lambda = Real(std::stod(value));
        } else if (key == "torus_wwin") {
            opt.torus_wwin = Real(std::stod(value));
        } else if (key == "torus_kappa") {
            opt.torus_kappa = Real(std::stod(value));
        } else if (key == "torus_omegac") {
            opt.torus_omegac = Real(std::stod(value));
        } else if (key == "torus_betac") {
            opt.torus_betac = Real(std::stod(value));
        } else if (key == "torus_beta") {
            opt.torus_beta = Real(std::stod(value));
        } else if (key == "torus_Rhigh" || key == "torus_rhigh") {
            opt.torus_Rhigh = Real(std::stod(value));
        } else if (key == "torus_bh_mass_solar") {
            opt.torus_bh_mass_solar = Real(std::stod(value));
        } else if (key == "torus_mdot_cgs") {
            opt.torus_mdot_cgs = Real(std::stod(value));
        } else if (key == "torus_mdot_code") {
            opt.torus_mdot_code = Real(std::stod(value));
        } else if (key == "torus_thetae_min") {
            opt.torus_thetae_min = Real(std::stod(value));
        } else if (key == "scalar_transport") {
            opt.scalar_transport = std::stoi(value);
        } else if (key == "iharm_dump") {
            opt.iharm_dump = value;
        } else if (key == "kharma_dump" || key == "phdf") {
            opt.kharma_dump = value;
        } else if (key == "athenak_dump" || key == "athenak_bin" || key == "bin") {
            opt.athenak_dump = value;
        } else if (key == "bhac_dump" || key == "bhac_dat" || key == "dat") {
            opt.bhac_dump = value;
        } else if (key == "hamr_dump" || key == "hamr_dir" || key == "hamr_path") {
            opt.hamr_dump = value;
        } else if (key == "dump") {
            opt.iharm_dump = value;
            opt.kharma_dump = value;
            opt.athenak_dump = value;
            opt.bhac_dump = value;
            opt.hamr_dump = value;
        } else if (key == "iharm_M_unit" || key == "iharm_m_unit") {
            opt.iharm_M_unit = Real(std::stod(value));
        } else if (key == "kharma_M_unit" || key == "kharma_m_unit") {
            opt.kharma_M_unit = Real(std::stod(value));
        } else if (key == "athenak_M_unit" || key == "athenak_m_unit") {
            opt.athenak_M_unit = Real(std::stod(value));
        } else if (key == "bhac_M_unit" || key == "bhac_m_unit") {
            opt.bhac_M_unit = Real(std::stod(value));
        } else if (key == "hamr_M_unit" || key == "hamr_m_unit") {
            opt.hamr_M_unit = Real(std::stod(value));
        } else if (key == "M_unit") {
            opt.iharm_M_unit = Real(std::stod(value));
            opt.kharma_M_unit = Real(std::stod(value));
            opt.athenak_M_unit = Real(std::stod(value));
            opt.bhac_M_unit = Real(std::stod(value));
            opt.hamr_M_unit = Real(std::stod(value));
        } else if (key == "iharm_mbh_solar") {
            opt.iharm_mbh_solar = Real(std::stod(value));
        } else if (key == "kharma_mbh_solar") {
            opt.kharma_mbh_solar = Real(std::stod(value));
        } else if (key == "athenak_mbh_solar") {
            opt.athenak_mbh_solar = Real(std::stod(value));
        } else if (key == "bhac_mbh_solar") {
            opt.bhac_mbh_solar = Real(std::stod(value));
        } else if (key == "hamr_mbh_solar") {
            opt.hamr_mbh_solar = Real(std::stod(value));
        } else if (key == "MBH") {
            opt.iharm_mbh_solar = Real(std::stod(value));
            opt.kharma_mbh_solar = Real(std::stod(value));
            opt.athenak_mbh_solar = Real(std::stod(value));
            opt.bhac_mbh_solar = Real(std::stod(value));
            opt.hamr_mbh_solar = Real(std::stod(value));
        } else if (key == "iharm_trat_small") {
            opt.iharm_trat_small = Real(std::stod(value));
        } else if (key == "kharma_trat_small") {
            opt.kharma_trat_small = Real(std::stod(value));
        } else if (key == "athenak_trat_small") {
            opt.athenak_trat_small = Real(std::stod(value));
        } else if (key == "bhac_trat_small") {
            opt.bhac_trat_small = Real(std::stod(value));
        } else if (key == "hamr_trat_small") {
            opt.hamr_trat_small = Real(std::stod(value));
        } else if (key == "trat_small") {
            opt.iharm_trat_small = Real(std::stod(value));
            opt.kharma_trat_small = Real(std::stod(value));
            opt.athenak_trat_small = Real(std::stod(value));
            opt.bhac_trat_small = Real(std::stod(value));
            opt.hamr_trat_small = Real(std::stod(value));
        } else if (key == "iharm_trat_large") {
            opt.iharm_trat_large = Real(std::stod(value));
        } else if (key == "kharma_trat_large") {
            opt.kharma_trat_large = Real(std::stod(value));
        } else if (key == "athenak_trat_large") {
            opt.athenak_trat_large = Real(std::stod(value));
        } else if (key == "bhac_trat_large") {
            opt.bhac_trat_large = Real(std::stod(value));
        } else if (key == "hamr_trat_large") {
            opt.hamr_trat_large = Real(std::stod(value));
        } else if (key == "trat_large") {
            opt.iharm_trat_large = Real(std::stod(value));
            opt.kharma_trat_large = Real(std::stod(value));
            opt.athenak_trat_large = Real(std::stod(value));
            opt.bhac_trat_large = Real(std::stod(value));
            opt.hamr_trat_large = Real(std::stod(value));
        } else if (key == "iharm_beta_crit") {
            opt.iharm_beta_crit = Real(std::stod(value));
        } else if (key == "kharma_beta_crit") {
            opt.kharma_beta_crit = Real(std::stod(value));
        } else if (key == "athenak_beta_crit") {
            opt.athenak_beta_crit = Real(std::stod(value));
        } else if (key == "bhac_beta_crit") {
            opt.bhac_beta_crit = Real(std::stod(value));
        } else if (key == "hamr_beta_crit") {
            opt.hamr_beta_crit = Real(std::stod(value));
        } else if (key == "athenak_gamma") {
            opt.athenak_gamma = Real(std::stod(value));
        } else if (key == "bhac_gamma") {
            opt.bhac_gamma = Real(std::stod(value));
        } else if (key == "hamr_gamma") {
            opt.hamr_gamma = Real(std::stod(value));
        } else if (key == "beta_crit") {
            opt.iharm_beta_crit = Real(std::stod(value));
            opt.kharma_beta_crit = Real(std::stod(value));
            opt.athenak_beta_crit = Real(std::stod(value));
            opt.bhac_beta_crit = Real(std::stod(value));
            opt.hamr_beta_crit = Real(std::stod(value));
        } else if (key == "iharm_sigma_cut") {
            opt.iharm_sigma_cut = Real(std::stod(value));
        } else if (key == "kharma_sigma_cut") {
            opt.kharma_sigma_cut = Real(std::stod(value));
        } else if (key == "athenak_sigma_cut") {
            opt.athenak_sigma_cut = Real(std::stod(value));
        } else if (key == "bhac_sigma_cut") {
            opt.bhac_sigma_cut = Real(std::stod(value));
        } else if (key == "hamr_sigma_cut") {
            opt.hamr_sigma_cut = Real(std::stod(value));
        } else if (key == "sigma_cut") {
            opt.iharm_sigma_cut = Real(std::stod(value));
            opt.kharma_sigma_cut = Real(std::stod(value));
            opt.athenak_sigma_cut = Real(std::stod(value));
            opt.bhac_sigma_cut = Real(std::stod(value));
            opt.hamr_sigma_cut = Real(std::stod(value));
        } else if (key == "iharm_sigma_cut_high") {
            opt.iharm_sigma_cut_high = Real(std::stod(value));
        } else if (key == "kharma_sigma_cut_high") {
            opt.kharma_sigma_cut_high = Real(std::stod(value));
        } else if (key == "athenak_sigma_cut_high") {
            opt.athenak_sigma_cut_high = Real(std::stod(value));
        } else if (key == "bhac_sigma_cut_high") {
            opt.bhac_sigma_cut_high = Real(std::stod(value));
        } else if (key == "hamr_sigma_cut_high") {
            opt.hamr_sigma_cut_high = Real(std::stod(value));
        } else if (key == "sigma_cut_high") {
            opt.iharm_sigma_cut_high = Real(std::stod(value));
            opt.kharma_sigma_cut_high = Real(std::stod(value));
            opt.athenak_sigma_cut_high = Real(std::stod(value));
            opt.bhac_sigma_cut_high = Real(std::stod(value));
            opt.hamr_sigma_cut_high = Real(std::stod(value));
        } else if (key == "iharm_interpolate_derived_scalars") {
            opt.iharm_interpolate_derived_scalars = std::stoi(value);
        } else if (key == "kharma_interpolate_derived_scalars") {
            opt.kharma_interpolate_derived_scalars = std::stoi(value);
        } else if (key == "interpolate_derived_scalars") {
            opt.iharm_interpolate_derived_scalars = std::stoi(value);
            opt.kharma_interpolate_derived_scalars = std::stoi(value);
        } else if (key == "kharma_reverse_field" || key == "reverse_field") {
            opt.kharma_reverse_field = std::stoi(value);
            opt.bhac_reverse_field = std::stoi(value);
            opt.hamr_reverse_field = std::stoi(value);
        } else if (key == "iharm_resample" || key == "kharma_resample" || key == "resample") {
            const std::string mode = trim_copy(value);
            int precomputed = 0;
            int primitives = 0;
            if (mode == "none" || mode == "0" || mode == "false") {
                // leave both modes disabled
            } else if (mode == "spherical_ks_precomputed" || mode == "sks_precomputed") {
                precomputed = 1;
            } else if (mode == "spherical_ks_primitives" || mode == "sks_primitives") {
                primitives = 1;
            } else {
                throw std::runtime_error("unknown resample mode: " + value);
            }
            if (key == "iharm_resample" || key == "resample") {
                opt.iharm_resample_spherical_ks_precomputed = precomputed;
                opt.iharm_resample_spherical_ks_primitives = primitives;
            }
            if (key == "kharma_resample" || key == "resample") {
                opt.kharma_resample_spherical_ks_precomputed = precomputed;
                opt.kharma_resample_spherical_ks_primitives = primitives;
            }
        } else if (key == "iharm_resample_spherical_ks_precomputed") {
            opt.iharm_resample_spherical_ks_precomputed = std::stoi(value);
        } else if (key == "kharma_resample_spherical_ks_precomputed") {
            opt.kharma_resample_spherical_ks_precomputed = std::stoi(value);
        } else if (key == "resample_spherical_ks_precomputed") {
            opt.iharm_resample_spherical_ks_precomputed = std::stoi(value);
            opt.kharma_resample_spherical_ks_precomputed = std::stoi(value);
        } else if (key == "iharm_resample_spherical_ks_primitives") {
            opt.iharm_resample_spherical_ks_primitives = std::stoi(value);
        } else if (key == "kharma_resample_spherical_ks_primitives") {
            opt.kharma_resample_spherical_ks_primitives = std::stoi(value);
        } else if (key == "resample_spherical_ks_primitives") {
            opt.iharm_resample_spherical_ks_primitives = std::stoi(value);
            opt.kharma_resample_spherical_ks_primitives = std::stoi(value);
        } else if (key == "iharm_resample_n1") {
            opt.iharm_resample_n1 = std::stoi(value);
        } else if (key == "kharma_resample_n1") {
            opt.kharma_resample_n1 = std::stoi(value);
        } else if (key == "resample_n1") {
            opt.iharm_resample_n1 = std::stoi(value);
            opt.kharma_resample_n1 = std::stoi(value);
        } else if (key == "iharm_resample_n2") {
            opt.iharm_resample_n2 = std::stoi(value);
        } else if (key == "kharma_resample_n2") {
            opt.kharma_resample_n2 = std::stoi(value);
        } else if (key == "resample_n2") {
            opt.iharm_resample_n2 = std::stoi(value);
            opt.kharma_resample_n2 = std::stoi(value);
        } else if (key == "iharm_resample_n3") {
            opt.iharm_resample_n3 = std::stoi(value);
        } else if (key == "kharma_resample_n3") {
            opt.kharma_resample_n3 = std::stoi(value);
        } else if (key == "resample_n3") {
            opt.iharm_resample_n3 = std::stoi(value);
            opt.kharma_resample_n3 = std::stoi(value);
        } else if (key == "iharm_resample_r_in") {
            opt.iharm_resample_r_in = Real(std::stod(value));
        } else if (key == "kharma_resample_r_in") {
            opt.kharma_resample_r_in = Real(std::stod(value));
        } else if (key == "resample_r_in") {
            opt.iharm_resample_r_in = Real(std::stod(value));
            opt.kharma_resample_r_in = Real(std::stod(value));
            opt.athenak_r_in = Real(std::stod(value));
            opt.bhac_r_in = Real(std::stod(value));
            opt.hamr_r_in = Real(std::stod(value));
        } else if (key == "iharm_resample_r_out") {
            opt.iharm_resample_r_out = Real(std::stod(value));
        } else if (key == "kharma_resample_r_out") {
            opt.kharma_resample_r_out = Real(std::stod(value));
        } else if (key == "resample_r_out") {
            opt.iharm_resample_r_out = Real(std::stod(value));
            opt.kharma_resample_r_out = Real(std::stod(value));
            opt.athenak_r_out = Real(std::stod(value));
            opt.bhac_r_out = Real(std::stod(value));
            opt.hamr_r_out = Real(std::stod(value));
        } else if (key == "athenak_r_in" || key == "athenak_resample_r_in") {
            opt.athenak_r_in = Real(std::stod(value));
        } else if (key == "athenak_r_out" || key == "athenak_resample_r_out") {
            opt.athenak_r_out = Real(std::stod(value));
        } else if (key == "athenak_profile_mode") {
            opt.athenak_profile_mode = std::stoi(value);
        } else if (key == "bhac_r_in" || key == "bhac_resample_r_in") {
            opt.bhac_r_in = Real(std::stod(value));
        } else if (key == "bhac_r_out" || key == "bhac_resample_r_out") {
            opt.bhac_r_out = Real(std::stod(value));
        } else if (key == "bhac_hslope") {
            opt.bhac_hslope = Real(std::stod(value));
        } else if (key == "bhac_nxlone1" || key == "bhac_nxlone_1") {
            opt.bhac_nxlone1 = std::stoi(value);
        } else if (key == "bhac_nxlone2" || key == "bhac_nxlone_2") {
            opt.bhac_nxlone2 = std::stoi(value);
        } else if (key == "bhac_nxlone3" || key == "bhac_nxlone_3") {
            opt.bhac_nxlone3 = std::stoi(value);
        } else if (key == "bhac_spin_index" || key == "bhac_nspin") {
            opt.bhac_spin_index = std::stoi(value);
        } else if (key == "bhac_x1_min") {
            opt.bhac_x1_min = Real(std::stod(value));
        } else if (key == "bhac_x1_max") {
            opt.bhac_x1_max = Real(std::stod(value));
        } else if (key == "bhac_x2_min") {
            opt.bhac_x2_min = Real(std::stod(value));
        } else if (key == "bhac_x2_max") {
            opt.bhac_x2_max = Real(std::stod(value));
        } else if (key == "bhac_x3_min") {
            opt.bhac_x3_min = Real(std::stod(value));
        } else if (key == "bhac_x3_max") {
            opt.bhac_x3_max = Real(std::stod(value));
        } else if (key == "bhac_sfc") {
            opt.bhac_sfc = std::stoi(value);
        } else if (key == "bhac_reverse_field") {
            opt.bhac_reverse_field = std::stoi(value);
        } else if (key == "bhac_profile_mode") {
            opt.bhac_profile_mode = std::stoi(value);
        } else if (key == "bhac_cache" || key == "bhac_staged_cache") {
            opt.bhac_cache = value;
        } else if (key == "bhac_cache_mode" || key == "bhac_staged_cache_mode") {
            const std::string mode = trim_copy(value);
            if (mode == "off" || mode == "none" || mode == "0" || mode == "false") {
                opt.bhac_cache_mode = 0;
            } else if (mode == "read" || mode == "read_only") {
                opt.bhac_cache_mode = 1;
            } else if (mode == "write" || mode == "write_only" || mode == "2") {
                opt.bhac_cache_mode = 2;
            } else if (mode == "read_write" || mode == "auto" || mode == "update" ||
                       mode == "1" || mode == "3" || mode == "true") {
                opt.bhac_cache_mode = 3;
            } else {
                throw std::runtime_error("unknown BHAC cache mode: " + value);
            }
        } else if (key == "hamr_r_in" || key == "hamr_resample_r_in") {
            opt.hamr_r_in = Real(std::stod(value));
        } else if (key == "hamr_r_out" || key == "hamr_resample_r_out") {
            opt.hamr_r_out = Real(std::stod(value));
        } else if (key == "hamr_hslope") {
            opt.hamr_hslope = Real(std::stod(value));
            opt.hamr_hslope_explicit = true;
        } else if (key == "hamr_reverse_field") {
            opt.hamr_reverse_field = std::stoi(value);
        } else if (key == "hamr_profile_mode") {
            opt.hamr_profile_mode = std::stoi(value);
        } else if (key == "hamr_id_order") {
            opt.hamr_id_order = trim_copy(value);
        } else if (key == "hamr_root_order") {
            opt.hamr_root_order = trim_copy(value);
        } else if (key == "repeat_images") {
            opt.repeat_images = std::stoi(value);
        } else if (key == "timing" || key == "profile") {
            opt.timing = std::stoi(value);
        } else {
            throw std::runtime_error("unknown option '" + key + "' in " + source);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("invalid value for option '" + key + "' in " + source + ": " + value + " (" + e.what() + ")");
    }
}

void apply_parameter_file(Options& opt, const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open parameter file: " + path);
    }
    std::string line;
    int line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        line = trim_copy(line);
        if (line.empty()) {
            continue;
        }
        std::string key;
        std::string value;
        const size_t eq = line.find('=');
        if (eq != std::string::npos) {
            key = trim_copy(line.substr(0, eq));
            value = trim_copy(line.substr(eq + 1));
        } else {
            std::istringstream iss(line);
            iss >> key >> value;
            std::string extra;
            if (iss >> extra) {
                throw std::runtime_error(path + ":" + std::to_string(line_no) + ": expected key=value or key value");
            }
        }
        if (key.empty()) {
            throw std::runtime_error(path + ":" + std::to_string(line_no) + ": empty key");
        }
        if (value.empty()) {
            const std::string normalized_key = normalize_key(key);
            // Empty edges select the built-in scalar partition boundaries.
            // Preserve this optional field when replaying an effective file.
            if (normalized_key == "analysis_partition_edges") {
                opt.analysis_partition_edges.clear();
                continue;
            }
            if (normalized_key == "parameter_file" || normalized_key == "input" ||
                normalized_key == "input_file" || normalized_key == "params" ||
                normalized_key == "config") {
                continue;
            }
            throw std::runtime_error(path + ":" + std::to_string(line_no) + ": empty value");
        }
        apply_option(opt, key, value, path + ":" + std::to_string(line_no));
    }
}

Options parse_options(int argc, char** argv) {
    Options opt;
    std::vector<std::string> parameter_files;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help") {
            print_usage_and_exit();
        }
        const size_t eq = arg.find('=');
        if (eq == std::string::npos || arg.rfind("--", 0) != 0) {
            throw std::runtime_error("expected --key=value option: " + arg);
        }
        const std::string key = normalize_key(arg.substr(0, eq));
        const std::string value = arg.substr(eq + 1);
        if (key == "parameter_file" || key == "input" || key == "input_file" ||
            key == "params" || key == "config") {
            parameter_files.push_back(value);
        }
    }

    for (const std::string& path : parameter_files) {
        apply_parameter_file(opt, path);
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help") {
            print_usage_and_exit();
        }
        const size_t eq = arg.find('=');
        const std::string key = arg.substr(0, eq);
        const std::string value = arg.substr(eq + 1);
        apply_option(opt, key, value, "command line");
    }
    if (opt.slow_light_snapshot_cache_gib > Real(0) && !opt.slow_light)
        throw std::invalid_argument("slow_light_snapshot_cache_gib requires slow_light=1");
    return opt;
}
