#include "image/grmhd.hpp"

#include <stdexcept>

#include "image/grmhd_backends.hpp"
#if KPOLARIS_ENABLE_ANALYSIS_MODE
#include "image/analysis_driver.hpp"
#include "geometry/kerr_boyer_lindquist.hpp"
#include "geometry/kerr_fmks.hpp"
#include "geometry/kerr_schild_cartesian.hpp"
#include "geometry/kerr_schild_spherical.hpp"
#endif

namespace {
using Real = kpolaris::DefaultReal;

std::runtime_error coordinate_not_built(const char* coordinate) {
    return std::runtime_error(std::string("GRMHD image coordinate backend not built: ") +
                              coordinate + ". Reconfigure KPOLARIS_GRMHD_COORDINATES to include it.");
}
} // namespace

ImageHostData run_grmhd_image(
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::GRMHDRadiationModel<Real> model,
    int radiation_substeps,
    int timing,
    int split_transport) {
    switch (pass_a.coordinate_system) {
    case kpolaris::CoordinateSystem::CartesianKS:
#if KPOLARIS_ENABLE_COORDINATE_CARTESIAN_KS
        return run_grmhd_image_cartesian_ks(pass_a, model, radiation_substeps, timing, split_transport);
#else
        throw coordinate_not_built("cartesian_ks");
#endif
    case kpolaris::CoordinateSystem::SphericalKS:
#if KPOLARIS_ENABLE_COORDINATE_SPHERICAL_KS
        return run_grmhd_image_spherical_ks(pass_a, model, radiation_substeps, timing, split_transport);
#else
        throw coordinate_not_built("spherical_ks");
#endif
    case kpolaris::CoordinateSystem::BoyerLindquist:
#if KPOLARIS_ENABLE_COORDINATE_BOYER_LINDQUIST
        return run_grmhd_image_boyer_lindquist(pass_a, model, radiation_substeps, timing, split_transport);
#else
        throw coordinate_not_built("boyer_lindquist");
#endif
    case kpolaris::CoordinateSystem::FMKS:
#if KPOLARIS_ENABLE_COORDINATE_FMKS
        return run_grmhd_image_fmks(pass_a, model, radiation_substeps, timing, split_transport);
#else
        throw coordinate_not_built("fmks");
#endif
    case kpolaris::CoordinateSystem::MKS:
#if KPOLARIS_ENABLE_COORDINATE_MKS
        return run_grmhd_image_mks(pass_a, model, radiation_substeps, timing, split_transport);
#else
        throw coordinate_not_built("mks");
#endif
    }
    throw coordinate_not_built("unknown");
}

#if KPOLARIS_ENABLE_ANALYSIS_MODE
ImageHostData run_grmhd_analysis_image(
    const kpolaris::PassAParams<Real>& pass_a,
    const kpolaris::AnalysisConfig<Real>& analysis_config,
    kpolaris::GRMHDRadiationModel<Real> model,
    int radiation_substeps,
    int timing,
    int split_transport) {
    switch (pass_a.coordinate_system) {
    case kpolaris::CoordinateSystem::CartesianKS: {
#if KPOLARIS_ENABLE_COORDINATE_CARTESIAN_KS
        kpolaris::KerrSchildInMetric<Real> metric(pass_a.mass, pass_a.spin);
        return kpolaris_image_detail::run_analysis_image_metric_with_optional_split(
            pass_a, analysis_config, model, radiation_substeps, timing, split_transport,
            metric, "cartesian_ks_analysis");
#else
        throw coordinate_not_built("cartesian_ks");
#endif
    }
    case kpolaris::CoordinateSystem::SphericalKS: {
#if KPOLARIS_ENABLE_COORDINATE_SPHERICAL_KS
        kpolaris::KerrSchildSphericalMetric<Real> metric(pass_a.mass, pass_a.spin);
        return kpolaris_image_detail::run_analysis_image_metric_with_optional_split(
            pass_a, analysis_config, model, radiation_substeps, timing, split_transport,
            metric, "spherical_ks_analysis");
#else
        throw coordinate_not_built("spherical_ks");
#endif
    }
    case kpolaris::CoordinateSystem::BoyerLindquist: {
#if KPOLARIS_ENABLE_COORDINATE_BOYER_LINDQUIST
        kpolaris::KerrBoyerLindquistMetric<Real> metric(pass_a.mass, pass_a.spin);
        return kpolaris_image_detail::run_analysis_image_metric_with_optional_split(
            pass_a, analysis_config, model, radiation_substeps, timing, split_transport,
            metric, "boyer_lindquist_analysis");
#else
        throw coordinate_not_built("boyer_lindquist");
#endif
    }
    case kpolaris::CoordinateSystem::FMKS: {
#if KPOLARIS_ENABLE_COORDINATE_FMKS
        kpolaris::KerrFMKSMetric<Real> metric(pass_a.mass, pass_a.spin, pass_a.fmks_startx1,
                                             pass_a.fmks_hslope, pass_a.fmks_mks_smooth,
                                             pass_a.fmks_poly_alpha, pass_a.fmks_poly_xt,
                                             pass_a.fmks_poly_norm);
        return kpolaris_image_detail::run_analysis_image_metric_with_optional_split(
            pass_a, analysis_config, model, radiation_substeps, timing, split_transport,
            metric, "fmks_analysis");
#else
        throw coordinate_not_built("fmks");
#endif
    }
    case kpolaris::CoordinateSystem::MKS: {
#if KPOLARIS_ENABLE_COORDINATE_MKS
        kpolaris::KerrSchildSphericalMetric<Real> metric(pass_a.mass, pass_a.spin);
        return kpolaris_image_detail::run_analysis_image_metric_with_optional_split(
            pass_a, analysis_config, model, radiation_substeps, timing, split_transport,
            metric, "mks_analysis");
#else
        throw coordinate_not_built("mks");
#endif
    }
    }
    throw coordinate_not_built("unknown");
}
#endif

#if KPOLARIS_MAX_FREQUENCIES > 1
ImageHostData run_grmhd_multifrequency_image(
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::GRMHDRadiationModel<Real> model,
    int radiation_substeps,
    const std::vector<Real>& frequencies,
    const std::vector<Real>& step_control_frequencies,
    int timing) {
    switch (pass_a.coordinate_system) {
    case kpolaris::CoordinateSystem::CartesianKS:
#if KPOLARIS_ENABLE_COORDINATE_CARTESIAN_KS
        return run_grmhd_multifrequency_image_cartesian_ks(pass_a, model, radiation_substeps, frequencies, step_control_frequencies, timing);
#else
        throw coordinate_not_built("cartesian_ks");
#endif
    case kpolaris::CoordinateSystem::SphericalKS:
#if KPOLARIS_ENABLE_COORDINATE_SPHERICAL_KS
        return run_grmhd_multifrequency_image_spherical_ks(pass_a, model, radiation_substeps, frequencies, step_control_frequencies, timing);
#else
        throw coordinate_not_built("spherical_ks");
#endif
    case kpolaris::CoordinateSystem::BoyerLindquist:
#if KPOLARIS_ENABLE_COORDINATE_BOYER_LINDQUIST
        return run_grmhd_multifrequency_image_boyer_lindquist(pass_a, model, radiation_substeps, frequencies, step_control_frequencies, timing);
#else
        throw coordinate_not_built("boyer_lindquist");
#endif
    case kpolaris::CoordinateSystem::FMKS:
#if KPOLARIS_ENABLE_COORDINATE_FMKS
        return run_grmhd_multifrequency_image_fmks(pass_a, model, radiation_substeps, frequencies, step_control_frequencies, timing);
#else
        throw coordinate_not_built("fmks");
#endif
    case kpolaris::CoordinateSystem::MKS:
#if KPOLARIS_ENABLE_COORDINATE_MKS
        return run_grmhd_multifrequency_image_mks(pass_a, model, radiation_substeps, frequencies, step_control_frequencies, timing);
#else
        throw coordinate_not_built("mks");
#endif
    }
    throw coordinate_not_built("unknown");
}
#endif
