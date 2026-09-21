#include "image/driver.hpp"
#include "image/split_driver.hpp"
#if KPOLARIS_MAX_FREQUENCIES > 1
#include "image/multifrequency.hpp"
#endif
#include "image/grmhd.hpp"

#include "geometry/kerr_fmks.hpp"

namespace {
using Real = kpolaris::DefaultReal;
}

ImageHostData run_grmhd_image_fmks(
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::GRMHDRadiationModel<Real> model,
    int radiation_substeps,
    int timing,
    int split_transport) {
    kpolaris::KerrFMKSMetric<Real> metric(pass_a.mass, pass_a.spin, pass_a.fmks_startx1, pass_a.fmks_hslope, pass_a.fmks_mks_smooth, pass_a.fmks_poly_alpha, pass_a.fmks_poly_xt, pass_a.fmks_poly_norm);
    return kpolaris_image_detail::run_image_metric_with_optional_split(
        pass_a, model, radiation_substeps, timing, split_transport, metric, "fmks");
}

#if KPOLARIS_MAX_FREQUENCIES > 1
ImageHostData run_grmhd_multifrequency_image_fmks(
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::GRMHDRadiationModel<Real> model,
    int radiation_substeps,
    const std::vector<Real>& frequencies,
    const std::vector<Real>& step_control_frequencies,
    int timing) {
    kpolaris::KerrFMKSMetric<Real> metric(pass_a.mass, pass_a.spin, pass_a.fmks_startx1, pass_a.fmks_hslope, pass_a.fmks_mks_smooth, pass_a.fmks_poly_alpha, pass_a.fmks_poly_xt, pass_a.fmks_poly_norm);
    return kpolaris_image_detail::run_multifrequency_image_metric_with_step_control(
        pass_a, model, radiation_substeps, frequencies, step_control_frequencies, timing, metric, "fmks");
}
#endif
