#pragma once

#include "model/grmhd_grid.hpp"

namespace kpolaris {

// Compatibility alias for the current iharm loader.  The implementation is a
// generic structured-GRMHD radiation model and is reused by future dump readers
// that populate the same primitive/derived-scalar layout.
template<class Real = DefaultReal>
using IHARMRadiationModel = GRMHDGridRadiationModel<Real>;

} // namespace kpolaris
