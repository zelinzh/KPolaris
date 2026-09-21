#pragma once

#include <vector>

#include "common/types.hpp"
#include "diagnostics/response_config.hpp"
#include "geodesic/pass_a.hpp"

struct ImageHostData {
    using Real = kpolaris::DefaultReal;
    int npix = 0;
    int nfreq = 1;
    std::vector<Real> image_i;
    std::vector<Real> image_q;
    std::vector<Real> image_u;
    std::vector<Real> image_v;
    std::vector<Real> closure_x;
    std::vector<Real> closure_k;
    std::vector<Real> final_null;
    std::vector<Real> frame_error;
    std::vector<Real> det_r;
    std::vector<Real> overlap_r11;
    std::vector<Real> overlap_r12;
    std::vector<Real> overlap_r21;
    std::vector<Real> overlap_r22;
    std::vector<Real> basis_identity_error;
    std::vector<Real> basis_rotation_angle;
    std::vector<int> pass_a_steps;
    std::vector<int> steps;
    std::vector<int> reason;
    int has_analysis = 0;
    kpolaris::ResponseConfig<Real> response_config;
    std::vector<Real> response_data;
    std::vector<Real> radiating_path_length;
    std::vector<Real> emission_weight;
    std::vector<Real> emission_weighted_radius;
    std::vector<Real> emission_weighted_optical_depth_to_camera;
    std::vector<Real> absorption_depth;
    std::vector<Real> absorption_operator_depth;
    std::vector<Real> faraday_rotation_depth;
    std::vector<Real> faraday_conversion_depth;
    std::vector<Real> faraday_operator_depth;
    std::vector<Real> dominant_emission_radius;
    std::vector<int> dominant_emission_region;
    std::vector<Real> dominant_ne_cgs;
    std::vector<Real> dominant_thetae;
    std::vector<Real> dominant_b_cgs;
    std::vector<Real> dominant_beta;
    std::vector<Real> dominant_sigma;
    std::vector<Real> emission_weighted_ne_cgs;
    std::vector<Real> emission_weighted_thetae;
    std::vector<Real> emission_weighted_b_cgs;
    std::vector<Real> emission_weighted_beta;
    std::vector<Real> emission_weighted_sigma;
    std::vector<Real> photon_ring_winding_estimate;
    std::vector<int> radiation_substeps;
    int analysis_radial_bins = 0;
    Real analysis_radial_min = Real(0);
    Real analysis_radial_max = Real(0);
    Real analysis_formation_fraction = Real(0);
    std::vector<Real> analysis_radial_bin_edges;
    std::vector<Real> analysis_radial_bin_centers;
    std::vector<Real> radial_stokes_i_contribution;
    std::vector<Real> radial_stokes_q_contribution;
    std::vector<Real> radial_stokes_u_contribution;
    std::vector<Real> radial_stokes_v_contribution;
    std::vector<Real> radial_absorption_depth;
    std::vector<Real> radial_faraday_rotation_depth;
    std::vector<Real> radial_faraday_conversion_depth;
    std::vector<Real> radial_faraday_operator_depth;
    std::vector<Real> observer_weighted_radius_i;
    std::vector<Real> observer_weighted_radius_linear;
    std::vector<Real> observer_weighted_radius_circular;
    std::vector<Real> intensity_formation_radius_low;
    std::vector<Real> intensity_formation_radius_median;
    std::vector<Real> intensity_formation_radius_high;
    std::vector<Real> linear_formation_radius_low;
    std::vector<Real> linear_formation_radius_median;
    std::vector<Real> linear_formation_radius_high;
    std::vector<Real> circular_formation_radius_low;
    std::vector<Real> circular_formation_radius_median;
    std::vector<Real> circular_formation_radius_high;
    std::vector<Real> los_linear_coherence;
    std::vector<Real> los_circular_coherence;
    std::vector<Real> contribution_closure_max_abs;
    std::vector<Real> contribution_closure_relative_l1;
    std::vector<Real> foreground_absorption_depth;
    std::vector<Real> foreground_faraday_rotation_depth;
    std::vector<Real> foreground_faraday_conversion_depth;
    std::vector<Real> foreground_faraday_operator_depth;
    std::vector<Real> foreground_faraday_operator_fraction;
};
