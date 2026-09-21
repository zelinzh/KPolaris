#include "image/split_driver.hpp"
#if KPOLARIS_MAX_FREQUENCIES > 1
#include "image/multifrequency.hpp"
#endif

#include <stdexcept>

#include "geometry/kerr_boyer_lindquist.hpp"
#include "model/riaf.hpp"
#include "image/riaf.hpp"

namespace {
using Real = kpolaris::DefaultReal;
}

ImageHostData run_riaf_image_boyer_lindquist(
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::RIAFAnalyticRadiationModel<Real> model,
    int radiation_substeps,
    int timing,
    int split_transport) {
    kpolaris::KerrBoyerLindquistMetric<Real> metric(pass_a.mass, pass_a.spin);
    return kpolaris_image_detail::run_image_metric_with_optional_split(
        pass_a, model, radiation_substeps, timing, split_transport,
        metric, "riaf_boyer_lindquist");
}

#if KPOLARIS_MAX_FREQUENCIES > 1
ImageHostData run_riaf_multifrequency_image_boyer_lindquist(
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::RIAFAnalyticRadiationModel<Real> model,
    int radiation_substeps,
    const std::vector<Real>& frequencies,
    const std::vector<Real>& step_control_frequencies,
    int timing,
    int) {
    kpolaris::KerrBoyerLindquistMetric<Real> metric(pass_a.mass, pass_a.spin);
    return kpolaris_image_detail::run_multifrequency_image_metric_with_step_control(
        pass_a, model, radiation_substeps, frequencies, step_control_frequencies, timing, metric, "riaf_boyer_lindquist");
}
#endif
