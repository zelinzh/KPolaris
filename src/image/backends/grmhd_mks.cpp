#include "image/driver.hpp"
#include "image/split_driver.hpp"
#if KPOLARIS_MAX_FREQUENCIES > 1
#include "image/multifrequency.hpp"
#endif
#include "image/grmhd.hpp"

#include "geometry/kerr_schild_spherical.hpp"

// MKS is a native fluid-grid coordinate.  Ray tracing remains in spherical KS;
// GRMHDRadiationModel maps samples back to the MKS dump grid.

namespace {
using Real = kpolaris::DefaultReal;
}

ImageHostData run_grmhd_image_mks(
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::GRMHDRadiationModel<Real> model,
    int radiation_substeps,
    int timing,
    int split_transport) {
    kpolaris::KerrSchildSphericalMetric<Real> metric(pass_a.mass, pass_a.spin);
    return kpolaris_image_detail::run_image_metric_with_optional_split(
        pass_a, model, radiation_substeps, timing, split_transport, metric, "mks_spherical_ks");
}

#if KPOLARIS_MAX_FREQUENCIES > 1
ImageHostData run_grmhd_multifrequency_image_mks(
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::GRMHDRadiationModel<Real> model,
    int radiation_substeps,
    const std::vector<Real>& frequencies,
    const std::vector<Real>& step_control_frequencies,
    int timing) {
    kpolaris::KerrSchildSphericalMetric<Real> metric(pass_a.mass, pass_a.spin);
    return kpolaris_image_detail::run_multifrequency_image_metric_with_step_control(
        pass_a, model, radiation_substeps, frequencies, step_control_frequencies, timing, metric, "mks_spherical_ks");
}
#endif
