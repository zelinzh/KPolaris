#pragma once

#include <vector>

#include "image/grmhd.hpp"

ImageHostData run_grmhd_image_cartesian_ks(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::GRMHDRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    int timing,
    int split_transport);
ImageHostData run_grmhd_image_spherical_ks(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::GRMHDRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    int timing,
    int split_transport);
ImageHostData run_grmhd_image_boyer_lindquist(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::GRMHDRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    int timing,
    int split_transport);
ImageHostData run_grmhd_image_fmks(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::GRMHDRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    int timing,
    int split_transport);
ImageHostData run_grmhd_image_mks(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::GRMHDRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    int timing,
    int split_transport);

#if KPOLARIS_MAX_FREQUENCIES > 1
ImageHostData run_grmhd_multifrequency_image_cartesian_ks(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::GRMHDRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    const std::vector<kpolaris::DefaultReal>& frequencies,
    const std::vector<kpolaris::DefaultReal>& step_control_frequencies,
    int timing);
ImageHostData run_grmhd_multifrequency_image_spherical_ks(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::GRMHDRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    const std::vector<kpolaris::DefaultReal>& frequencies,
    const std::vector<kpolaris::DefaultReal>& step_control_frequencies,
    int timing);
ImageHostData run_grmhd_multifrequency_image_boyer_lindquist(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::GRMHDRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    const std::vector<kpolaris::DefaultReal>& frequencies,
    const std::vector<kpolaris::DefaultReal>& step_control_frequencies,
    int timing);
ImageHostData run_grmhd_multifrequency_image_fmks(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::GRMHDRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    const std::vector<kpolaris::DefaultReal>& frequencies,
    const std::vector<kpolaris::DefaultReal>& step_control_frequencies,
    int timing);
ImageHostData run_grmhd_multifrequency_image_mks(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::GRMHDRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    const std::vector<kpolaris::DefaultReal>& frequencies,
    const std::vector<kpolaris::DefaultReal>& step_control_frequencies,
    int timing);
#endif
