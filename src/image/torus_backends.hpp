#pragma once

#include <vector>

#include "image/torus.hpp"

ImageHostData run_torus_image_cartesian_ks(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::MagnetizedTorusRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    int timing,
    int split_transport);

#if KPOLARIS_MAX_FREQUENCIES > 1
ImageHostData run_torus_multifrequency_image_cartesian_ks(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::MagnetizedTorusRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    const std::vector<kpolaris::DefaultReal>& frequencies,
    const std::vector<kpolaris::DefaultReal>& step_control_frequencies,
    int timing,
    int split_transport);
#endif
