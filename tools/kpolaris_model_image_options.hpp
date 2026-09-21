#pragma once

#include <string>

#include "common/types.hpp"
#include "model/time_interpolation.hpp"
#include "grmhd/ddc_input_options.hpp"

#ifndef KPOLARIS_DEFAULT_MODEL
#define KPOLARIS_DEFAULT_MODEL "riaf"
#endif
#ifndef KPOLARIS_DEFAULT_BINARY_RIAF
#define KPOLARIS_DEFAULT_BINARY_RIAF 0
#endif

using ImageReal = kpolaris::DefaultReal;

struct Options {
    std::string model = KPOLARIS_DEFAULT_MODEL;
    std::string output = "kpolaris_image.h5";
    std::string output_format = "auto";
    std::string evpa_0 = "N";
    std::string parameter_file;
    std::string parameter_output = "auto";
#if KPOLARIS_DEFAULT_BINARY_RIAF
    std::string camera = "parallel_plane";
#else
    std::string camera = "pinhole";
#endif
    std::string coordinate = "cartesian_ks";
    int nx = 96;
    int ny = 96;
    ImageReal radius = ImageReal(35);
    ImageReal inclination = ImageReal(1.04719755119659774615);
    ImageReal fov = ImageReal(0.08);
    ImageReal fovy = ImageReal(-1);
#if KPOLARIS_DEFAULT_BINARY_RIAF
    ImageReal xspan = ImageReal(18);
    ImageReal yspan = ImageReal(18);
#else
    ImageReal xspan = ImageReal(-1);
    ImageReal yspan = ImageReal(-1);
#endif
    ImageReal dsource_pc = ImageReal(-1);
    ImageReal fovx_dsource = ImageReal(-1);
    ImageReal fovy_dsource = ImageReal(-1);
    ImageReal image_width_x = ImageReal(-1);
    ImageReal image_width_y = ImageReal(-1);
    ImageReal effective_fovx_dsource = ImageReal(-1);
    ImageReal effective_fovy_dsource = ImageReal(-1);
    ImageReal spin = ImageReal(0.9375);
    ImageReal inner_radius = ImageReal(-1);
    bool inner_radius_explicit = false;
    ImageReal outer_radius = ImageReal(-1);
    ImageReal step = ImageReal(0.025);
    int max_steps = 4096;
    int adaptive = 1;
    int direct_only = 0;
    ImageReal equatorial_h_over_r = ImageReal(0);
    int equatorial_samples = 8;
    int faraday_rotation = 1;
    ImageReal adaptive_tolerance = ImageReal(1e-10);
    ImageReal min_step = ImageReal(1e-5);
    ImageReal max_step = ImageReal(1);
    ImageReal max_radiation_step = ImageReal(1);
    ImageReal max_radiation_depth = ImageReal(1);
    ImageReal max_absorption_depth = ImageReal(1);
    ImageReal max_faraday_depth = ImageReal(4);
    ImageReal fmks_startx1 = ImageReal(0);
    ImageReal fmks_hslope = ImageReal(0.3);
    ImageReal fmks_mks_smooth = ImageReal(0.5);
    ImageReal fmks_poly_alpha = ImageReal(14);
    ImageReal fmks_poly_xt = ImageReal(0.82);
    ImageReal fmks_poly_norm = ImageReal(1);
    int radiation_substeps = 2;
    ImageReal closure_x_warning = ImageReal(1);
    ImageReal closure_k_warning = ImageReal(1e-2);
    ImageReal frame_error_warning = ImageReal(1e-2);
    ImageReal basis_identity_warning = ImageReal(1e-2);
    ImageReal freq = ImageReal(230.0e9);
    std::string freq_list;
    ImageReal freq_min = ImageReal(-1);
    ImageReal freq_max = ImageReal(-1);
    int nfreq = 0;
    std::string freq_spacing = "log";
    int multifrequency_chunk_size = 0;

    ImageReal riaf_r_min = ImageReal(1);
    ImageReal riaf_r_max = ImageReal(100);
    ImageReal riaf_nth0 = ImageReal(1);
    ImageReal riaf_Te0 = ImageReal(1);
    ImageReal riaf_disk_h = ImageReal(0.35);
    ImageReal riaf_pow_nth = ImageReal(-1.1);
    ImageReal riaf_pow_T = ImageReal(-0.84);
    ImageReal riaf_ne_unit = ImageReal(5.0e6);
    ImageReal riaf_te_unit = ImageReal(1.0e11);
    ImageReal riaf_mbh_solar = ImageReal(4.3e6);
    ImageReal riaf_keplerian_factor = ImageReal(1);
    ImageReal riaf_infall_factor = ImageReal(0);

    // Superposed-Kerr-Schild binary and mini-disk parameters.
    ImageReal binary_mass_ratio = ImageReal(1);
    ImageReal binary_chi1 = ImageReal(0);
    ImageReal binary_chi2 = ImageReal(0);
    ImageReal binary_reference_separation = ImageReal(20);
    ImageReal binary_reference_phase = ImageReal(0);
    ImageReal binary_reference_time = ImageReal(0);
    ImageReal binary_observation_time = ImageReal(0);
    ImageReal binary_minimum_separation = ImageReal(6);
    int binary_inspiral = 1;
    int binary_orbit = 1;
    // ``leading_quadrupole`` keeps the inexpensive Peters-law demonstration
    // orbit. ``table`` accepts a generic validated native/legacy trajectory.
    // ``paper_cbwaves_4pn_local`` additionally requires the verified native
    // provenance contract emitted from the pinned Combi--Ressler source.
    std::string binary_trajectory_model = "leading_quadrupole";
    std::string binary_trajectory_file;
    std::string binary_trajectory_format = "auto";
    // auto trusts the validated native schema.  Explicit values are useful
    // for reproducibility checks and must match the file exactly.
    std::string binary_trajectory_interpolation = "auto";
    ImageReal binary_trajectory_time_offset = ImageReal(0);
    // A positive value overrides the dimensional reference mass inferred by
    // the legacy Combi--Ressler table importer.  Native KPolaris tables are
    // already expressed in G=c=M_ref=1 and must leave this non-positive.
    ImageReal binary_trajectory_mass_scale = ImageReal(-1);
    // Host-only provenance populated after a trajectory has been loaded.
    std::string binary_trajectory_sha256;
    std::string binary_trajectory_schema;
    std::string binary_trajectory_gauge;
    std::string binary_trajectory_generator;
    std::string binary_trajectory_generator_version;
    std::string binary_trajectory_merger_reach_contract;
    std::string binary_trajectory_declared_model;
    std::string binary_trajectory_pn_terms;
    std::string binary_trajectory_source_doi;
    std::string binary_trajectory_pn_4pn_scope;
    int binary_trajectory_source_verified = -1;
    std::string binary_trajectory_source_verification;
    std::string binary_trajectory_upstream_cbwaves_sha256;
    std::string binary_trajectory_patched_cbwaves_sha256;
    int binary_trajectory_merger_separation_reached = -1;
    std::string binary_trajectory_status;
    std::string binary_trajectory_boost_velocity_model;
    int binary_trajectory_worldline_velocity_consistent = -1;
    int binary_trajectory_future_extension = 0;
    int binary_trajectory_samples = 0;
    ImageReal binary_trajectory_t_min = ImageReal(0);
    ImageReal binary_trajectory_t_max = ImageReal(0);
    ImageReal binary_transition_start = ImageReal(0);
    ImageReal binary_transition_end = ImageReal(0);
    int binary_trajectory_exact_remnant = 0;
    ImageReal binary_metric_derivative_step = ImageReal(2e-5);
    ImageReal binary_capture_factor = ImageReal(1.02);
    ImageReal binary_tidal_fraction = ImageReal(0.8);
    ImageReal binary_taper_start_fraction = ImageReal(0.85);
    ImageReal binary_density_scale1 = ImageReal(1);
    ImageReal binary_density_scale2 = ImageReal(1);
    ImageReal binary_temperature_scale1 = ImageReal(1);
    ImageReal binary_temperature_scale2 = ImageReal(1);
    int binary_field_polarity1 = 1;
    int binary_field_polarity2 = 1;
    int emission_type = 0;
    ImageReal nonthermal_kappa = ImageReal(3.5);
    int variable_kappa = 0;
    ImageReal variable_kappa_min = ImageReal(3.1);
    ImageReal variable_kappa_interp_start = ImageReal(1e20);
    ImageReal variable_kappa_max = ImageReal(7.0);
    ImageReal powerlaw_p = ImageReal(3.25);
    ImageReal powerlaw_eta = ImageReal(0.02);
    ImageReal powerlaw_gamma_min = ImageReal(1e2);
    ImageReal powerlaw_gamma_max = ImageReal(1e5);
    ImageReal powerlaw_gamma_cutoff = ImageReal(1e10);
    ImageReal x_offset = ImageReal(0);
    ImageReal y_offset = ImageReal(0);
    // Pixel centers by default; the ipole sampling offset is opt-in for replay.
    int use_pinhole_pixel_bias = 0;
    ImageReal pinhole_pixel_bias = ImageReal(-0.01);

    ImageReal torus_l_lambda = ImageReal(0.78);
    ImageReal torus_wwin = ImageReal(1);
    ImageReal torus_kappa = ImageReal(4) / ImageReal(3);
    ImageReal torus_omegac = ImageReal(1);
    ImageReal torus_betac = ImageReal(10);
    ImageReal torus_beta = ImageReal(10);
    ImageReal torus_Rhigh = ImageReal(1);
    ImageReal torus_bh_mass_solar = ImageReal(4.0e6);
    ImageReal torus_mdot_cgs = ImageReal(1.57e15);
    ImageReal torus_mdot_code = ImageReal(3.0e-3);
    ImageReal torus_thetae_min = ImageReal(5.0e-2);
    int scalar_transport = 0;

    std::string iharm_dump;
    ImageReal iharm_M_unit = ImageReal(3.0e25);
    ImageReal iharm_mbh_solar = ImageReal(6.2e9);
    ImageReal iharm_trat_small = ImageReal(1);
    ImageReal iharm_trat_large = ImageReal(20);
    ImageReal iharm_beta_crit = ImageReal(1);
    ImageReal iharm_sigma_cut = ImageReal(1);
    ImageReal iharm_sigma_cut_high = ImageReal(-1);
    int iharm_interpolate_derived_scalars = 1;
    int iharm_resample_spherical_ks_precomputed = 0;
    int iharm_resample_spherical_ks_primitives = 0;
    int iharm_resample_n1 = 0;
    int iharm_resample_n2 = 0;
    int iharm_resample_n3 = 0;
    ImageReal iharm_resample_r_in = ImageReal(-1);
    ImageReal iharm_resample_r_out = ImageReal(-1);

    std::string kharma_dump;
    kpolaris::DDCInputOptions kharma_ddc = kpolaris::DDCInputOptions::from_environment();
    ImageReal kharma_M_unit = ImageReal(3.0e25);
    ImageReal kharma_mbh_solar = ImageReal(6.2e9);
    ImageReal kharma_trat_small = ImageReal(1);
    ImageReal kharma_trat_large = ImageReal(20);
    ImageReal kharma_beta_crit = ImageReal(1);
    ImageReal kharma_sigma_cut = ImageReal(1);
    ImageReal kharma_sigma_cut_high = ImageReal(-1);
    int kharma_interpolate_derived_scalars = 1;
    int kharma_resample_spherical_ks_precomputed = 0;
    int kharma_resample_spherical_ks_primitives = 0;
    int kharma_resample_n1 = 0;
    int kharma_resample_n2 = 0;
    int kharma_resample_n3 = 0;
    ImageReal kharma_resample_r_in = ImageReal(-1);
    ImageReal kharma_resample_r_out = ImageReal(-1);
    int kharma_reverse_field = 0;

    std::string athenak_dump;
    ImageReal athenak_M_unit = ImageReal(1.0e26);
    ImageReal athenak_mbh_solar = ImageReal(6.2e9);
    ImageReal athenak_trat_small = ImageReal(1);
    ImageReal athenak_trat_large = ImageReal(40);
    ImageReal athenak_beta_crit = ImageReal(1);
    ImageReal athenak_gamma = ImageReal(-1);
    ImageReal athenak_sigma_cut = ImageReal(1);
    ImageReal athenak_sigma_cut_high = ImageReal(-1);
    ImageReal athenak_r_in = ImageReal(-1);
    ImageReal athenak_r_out = ImageReal(1000);
    int athenak_profile_mode = 0;

    std::string bhac_dump;
    ImageReal bhac_M_unit = ImageReal(1.0e18);
    ImageReal bhac_mbh_solar = ImageReal(4.14e6);
    ImageReal bhac_trat_small = ImageReal(1);
    ImageReal bhac_trat_large = ImageReal(40);
    ImageReal bhac_beta_crit = ImageReal(1);
    ImageReal bhac_gamma = ImageReal(-1);
    ImageReal bhac_sigma_cut = ImageReal(1);
    ImageReal bhac_sigma_cut_high = ImageReal(-1);
    ImageReal bhac_r_in = ImageReal(-1);
    ImageReal bhac_r_out = ImageReal(-1);
    ImageReal bhac_hslope = ImageReal(0.25);
    int bhac_nxlone1 = 0;
    int bhac_nxlone2 = 0;
    int bhac_nxlone3 = 0;
    int bhac_spin_index = -1;
    ImageReal bhac_x1_min = ImageReal(0.17);
    ImageReal bhac_x1_max = ImageReal(8.1117280833);
    ImageReal bhac_x2_min = ImageReal(0);
    ImageReal bhac_x2_max = ImageReal(3.141592653589793238462643383279502884);
    ImageReal bhac_x3_min = ImageReal(0);
    ImageReal bhac_x3_max = ImageReal(6.283185307179586476925286766559005768);
    int bhac_sfc = 1;
    int bhac_reverse_field = 0;
    int bhac_profile_mode = 0;
    std::string bhac_cache;
    int bhac_cache_mode = 3;

    std::string hamr_dump;
    ImageReal hamr_M_unit = ImageReal(1.0e26);
    ImageReal hamr_mbh_solar = ImageReal(6.2e9);
    ImageReal hamr_trat_small = ImageReal(1);
    ImageReal hamr_trat_large = ImageReal(40);
    ImageReal hamr_beta_crit = ImageReal(1);
    ImageReal hamr_gamma = ImageReal(-1);
    ImageReal hamr_sigma_cut = ImageReal(1);
    ImageReal hamr_sigma_cut_high = ImageReal(-1);
    ImageReal hamr_r_in = ImageReal(-1);
    ImageReal hamr_r_out = ImageReal(-1);
    ImageReal hamr_hslope = ImageReal(0.3);
    bool hamr_hslope_explicit = false;
    int hamr_reverse_field = 0;
    int hamr_profile_mode = 0;
    std::string hamr_id_order = "root_slot";
    std::string hamr_root_order = "morton";

    int timing = 0;
    int repeat_images = 1; // Recompute fast-light images with one loaded model.
    int split_transport = 1;
    int analysis_mode = 0;
    std::string analysis_response = "none";
    ImageReal analysis_response_step = ImageReal(1e-3);
    std::string analysis_partition = "none";
    std::string analysis_partition_edges;
    ImageReal analysis_funnel_angle_deg = ImageReal(20);
    ImageReal analysis_disk_angle_deg = ImageReal(60);
    ImageReal analysis_sigma_boundary = ImageReal(1);
    ImageReal analysis_beta_boundary = ImageReal(1);
    int analysis_radial_bins = 16;
    ImageReal analysis_radial_min = ImageReal(-1);
    ImageReal analysis_radial_max = ImageReal(-1);
    ImageReal analysis_formation_fraction = ImageReal(0.9);
    int slow_light = 0;
    kpolaris::SlowLightStepMode slow_light_step_mode = kpolaris::default_slow_light_step_mode;
    int slow_light_step_mode_explicit = 0;
    kpolaris::SlowLightInterpolation slow_light_interpolation = kpolaris::default_slow_light_interpolation;
    int slow_light_prefetch = 1;
    int slow_light_pipeline = 1;
    int slow_light_windows_per_block = 1;
    int slow_light_windows_per_block_explicit = 0;
    ImageReal slow_light_snapshot_cache_gib = ImageReal(0);
    int slow_light_prefetch_snapshots = 0;
    std::string slow_light_batch_jobs;
    int slow_light_batch_size = 1;
    int slow_light_batch_index = 0;
    int slow_light_time_probe = 0;
    ImageReal slow_light_observation_time = ImageReal(0);
    std::string slow_light_dump_list;
    std::string slow_light_time_list;
    std::string slow_light_dump_pattern;
    int slow_light_dump_start = 0;
    int slow_light_dump_end = -1;
    int slow_light_dump_stride = 1;
};



std::string canonical_camera_name(const std::string& name);
std::string canonical_coordinate_name(const std::string& name);
kpolaris::CoordinateSystem coordinate_system_from_name(const std::string& name);
std::string metric_name_from_coordinate(const std::string& name);
std::string trim_copy(const std::string& input);
void resolve_camera_extents(Options& opt);
void validate_options(const Options& opt);
void validate_resolved_camera(const Options& opt);
Options parse_options(int argc, char** argv);
