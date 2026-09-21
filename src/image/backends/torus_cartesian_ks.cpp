#include "image/split_driver.hpp"
#if KPOLARIS_MAX_FREQUENCIES > 1
#include "image/multifrequency.hpp"
#endif
#include "image/torus.hpp"

#include "geometry/kerr_schild_cartesian.hpp"

namespace {
using Real = kpolaris::DefaultReal;
}

ImageHostData run_torus_image_cartesian_ks(
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::MagnetizedTorusRadiationModel<Real> model,
    int radiation_substeps,
    int timing,
    int split_transport) {
    kpolaris::KerrSchildInMetric<Real> metric(pass_a.mass, pass_a.spin);
    return kpolaris_image_detail::run_image_metric_with_optional_split(
        pass_a, model, radiation_substeps, timing, split_transport,
        metric, "torus_cartesian_ks");
}

#if KPOLARIS_MAX_FREQUENCIES > 1
ImageHostData run_torus_multifrequency_image_cartesian_ks(
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::MagnetizedTorusRadiationModel<Real> model,
    int radiation_substeps,
    const std::vector<Real>& frequencies,
    const std::vector<Real>& step_control_frequencies,
    int timing,
    int) {
    kpolaris::KerrSchildInMetric<Real> metric(pass_a.mass, pass_a.spin);
    return kpolaris_image_detail::run_multifrequency_image_metric_with_step_control(
        pass_a, model, radiation_substeps, frequencies, step_control_frequencies, timing, metric,
        "torus_cartesian_ks");
}
#endif
