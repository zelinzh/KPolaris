#pragma once

#include <vector>

#include "image/riaf.hpp"

ImageHostData run_riaf_image_boyer_lindquist(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::RIAFAnalyticRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    int timing,
    int split_transport);

#if KPOLARIS_MAX_FREQUENCIES > 1
ImageHostData run_riaf_multifrequency_image_boyer_lindquist(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::RIAFAnalyticRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    const std::vector<kpolaris::DefaultReal>& frequencies,
    const std::vector<kpolaris::DefaultReal>& step_control_frequencies,
    int timing,
    int split_transport);
#endif
