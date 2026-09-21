#include "image/binary_riaf.hpp"

#include <stdexcept>

#include "image/split_driver.hpp"
#if KPOLARIS_MAX_FREQUENCIES > 1
#include "image/multifrequency.hpp"
#endif

namespace {
using Real = kpolaris::DefaultReal;
}

ImageHostData run_binary_riaf_image(
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::BinaryRIAFRadiationModel<Real> model,
    int radiation_substeps,
    int timing,
    int split_transport) {
    if (pass_a.coordinate_system != kpolaris::CoordinateSystem::CartesianKS) {
        throw std::runtime_error(
            "binary_riaf requires --coordinate=cartesian_ks");
    }
    if (!(model.sampled_min_inverse_denominator >= Real(0.1)) ||
        !(model.sampled_min_fluid_slice_timelike_margin > Real(0))) {
        model.validate_metric_domain(pass_a.outer_radius);
    }
    const kpolaris::SuperposedKerrSchildMetric<Real> metric = model.make_metric();
    return kpolaris_image_detail::run_image_metric_with_optional_split(
        pass_a, model, radiation_substeps, timing, split_transport,
        metric, "binary_riaf_superposed_ks");
}

#if KPOLARIS_MAX_FREQUENCIES > 1
ImageHostData run_binary_riaf_multifrequency_image(
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::BinaryRIAFRadiationModel<Real> model,
    int radiation_substeps,
    const std::vector<Real>& frequencies,
    const std::vector<Real>& step_control_frequencies,
    int timing,
    int) {
    if (pass_a.coordinate_system != kpolaris::CoordinateSystem::CartesianKS) {
        throw std::runtime_error(
            "binary_riaf requires --coordinate=cartesian_ks");
    }
    if (!(model.sampled_min_inverse_denominator >= Real(0.1)) ||
        !(model.sampled_min_fluid_slice_timelike_margin > Real(0))) {
        model.validate_metric_domain(pass_a.outer_radius);
    }
    const kpolaris::SuperposedKerrSchildMetric<Real> metric = model.make_metric();
    return kpolaris_image_detail::run_multifrequency_image_metric_with_step_control(
        pass_a, model, radiation_substeps, frequencies,
        step_control_frequencies, timing, metric,
        "binary_riaf_superposed_ks");
}
#endif
