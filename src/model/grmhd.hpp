#pragma once

#include "model/grmhd_grid.hpp"

namespace kpolaris {

// Shared grid-based GRMHD radiation model used by iharm/kharma/athena-style
// loaders. The current implementation keeps the validated IHARM layout and
// coefficient path; future loaders should populate this common model rather
// than introducing source-specific transport kernels.
template<class Real = DefaultReal>
using GRMHDRadiationModel = GRMHDGridRadiationModel<Real>;

} // namespace kpolaris
