#pragma once

#ifndef KPOLARIS_ENABLE_ANALYSIS_MODE
#define KPOLARIS_ENABLE_ANALYSIS_MODE 0
#endif

#include <vector>

#include "image/result.hpp"
#include "model/magnetized_torus.hpp"

ImageHostData run_torus_image(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::MagnetizedTorusRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    int timing,
    int split_transport);


#if KPOLARIS_ENABLE_ANALYSIS_MODE
#include "image/analysis_driver.hpp"
ImageHostData run_torus_analysis_image(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    const kpolaris::AnalysisConfig<kpolaris::DefaultReal>& analysis_config,
    kpolaris::MagnetizedTorusRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    int timing,
    int split_transport);
#endif

#if KPOLARIS_MAX_FREQUENCIES > 1
ImageHostData run_torus_multifrequency_image(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::MagnetizedTorusRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    const std::vector<kpolaris::DefaultReal>& frequencies,
    const std::vector<kpolaris::DefaultReal>& step_control_frequencies,
    int timing,
    int split_transport);
#endif
