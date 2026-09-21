#pragma once

#include <string>
#include <vector>

#include <H5Cpp.h>

#include "common/types.hpp"
#include "kpolaris_model_image_options.hpp"

struct DiagnosticSummary {
    double max_closure_x = 0.0;
    double max_closure_k = 0.0;
    double max_frame_error = 0.0;
    double max_basis_identity_error = 0.0;
    int max_closure_x_pixel = -1;
    int max_closure_k_pixel = -1;
    int max_frame_error_pixel = -1;
    int max_basis_identity_error_pixel = -1;
    int closure_x_warning_count = 0;
    int closure_k_warning_count = 0;
    int frame_error_warning_count = 0;
    int basis_identity_warning_count = 0;
    int any_warning_count = 0;
};

struct FrequencyImage {
    kpolaris::DefaultReal frequency = kpolaris::DefaultReal(0);
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
    int returned = 0;
    double mean_pass_a_steps = 0.0;
    double mean_pass_b_steps = 0.0;
    double mean_total_steps = 0.0;
    int max_pass_a_steps = 0;
    int max_pass_b_steps = 0;
    int max_total_steps = 0;
    kpolaris::DefaultReal sum_i = kpolaris::DefaultReal(0);
    kpolaris::DefaultReal sum_q = kpolaris::DefaultReal(0);
    kpolaris::DefaultReal sum_u = kpolaris::DefaultReal(0);
    kpolaris::DefaultReal sum_v = kpolaris::DefaultReal(0);
    DiagnosticSummary diagnostics;
};

bool exceeds_positive_threshold(double value, kpolaris::DefaultReal threshold);
void update_max_abs(double value, int pixel, double& max_value, int& max_pixel);
DiagnosticSummary summarize_diagnostics(const Options& opt,
                                        const std::vector<double>& closure_x,
                                        const std::vector<double>& closure_k,
                                        const std::vector<double>& frame_error,
                                        const std::vector<double>& basis_identity_error,
                                        const std::vector<int>& reason);
int pixel_ix(int pixel, int nx);
int pixel_iy(int pixel, int nx);

void write_h5_scalar(H5::H5Object& obj, const std::string& name, double value);
void write_h5_int(H5::H5Object& obj, const std::string& name, int value);
void write_h5_string(H5::H5Object& obj, const std::string& name, const std::string& value);
void write_h5_dataset_2d(H5::Group& group, const std::string& name,
                         const std::vector<double>& values, int nx, int ny);
void write_h5_dataset_2d_int(H5::Group& group, const std::string& name,
                             const std::vector<int>& values, int nx, int ny);
void write_h5_dataset_3d(H5::Group& group, const std::string& name,
                         const std::vector<double>& values,
                         int nplane, int nx, int ny);
void write_h5_dataset_1d(H5::Group& group, const std::string& name,
                         const std::vector<double>& values);
