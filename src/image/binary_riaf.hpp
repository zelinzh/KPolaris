#pragma once

#include <vector>

#include "geodesic/pass_a_core.hpp"
#include "image/result.hpp"
#include "model/binary_riaf.hpp"

ImageHostData run_binary_riaf_image(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::BinaryRIAFRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    int timing,
    int split_transport);

#if KPOLARIS_MAX_FREQUENCIES > 1
ImageHostData run_binary_riaf_multifrequency_image(
    const kpolaris::PassAParams<kpolaris::DefaultReal>& pass_a,
    kpolaris::BinaryRIAFRadiationModel<kpolaris::DefaultReal> model,
    int radiation_substeps,
    const std::vector<kpolaris::DefaultReal>& frequencies,
    const std::vector<kpolaris::DefaultReal>& step_control_frequencies,
    int timing,
    int split_transport);
#endif
