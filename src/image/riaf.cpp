#include "image/riaf.hpp"

#include <stdexcept>
#include <string>

#include "image/riaf_backends.hpp"
#if KPOLARIS_ENABLE_ANALYSIS_MODE
#include "image/analysis_driver.hpp"
#include "geometry/kerr_boyer_lindquist.hpp"
#endif

namespace {
using Real = kpolaris::DefaultReal;

std::runtime_error riaf_coordinate_not_built(const char* coordinate) {
    return std::runtime_error(std::string("RIAF image coordinate backend not built: ") +
                              coordinate + ". Reconfigure KPOLARIS_RIAF_COORDINATES to include it.");
}
} // namespace

ImageHostData run_riaf_image(
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::RIAFAnalyticRadiationModel<Real> model,
    int radiation_substeps,
    int timing,
    int split_transport) {
    switch (pass_a.coordinate_system) {
    case kpolaris::CoordinateSystem::BoyerLindquist:
#if KPOLARIS_ENABLE_RIAF_COORDINATE_BOYER_LINDQUIST
        return run_riaf_image_boyer_lindquist(pass_a, model, radiation_substeps,
                                                        timing, split_transport);
#else
        throw riaf_coordinate_not_built("boyer_lindquist");
#endif
    case kpolaris::CoordinateSystem::CartesianKS:
        throw riaf_coordinate_not_built("cartesian_ks");
    case kpolaris::CoordinateSystem::SphericalKS:
        throw riaf_coordinate_not_built("spherical_ks");
    case kpolaris::CoordinateSystem::FMKS:
        throw riaf_coordinate_not_built("fmks");
    case kpolaris::CoordinateSystem::MKS:
        throw riaf_coordinate_not_built("mks");
    }
    throw riaf_coordinate_not_built("unknown");
}

#if KPOLARIS_ENABLE_ANALYSIS_MODE
ImageHostData run_riaf_analysis_image(
    const kpolaris::PassAParams<Real>& pass_a,
    const kpolaris::AnalysisConfig<Real>& analysis_config,
    kpolaris::RIAFAnalyticRadiationModel<Real> model,
    int radiation_substeps,
    int timing,
    int split_transport) {
    switch (pass_a.coordinate_system) {
    case kpolaris::CoordinateSystem::BoyerLindquist: {
#if KPOLARIS_ENABLE_RIAF_COORDINATE_BOYER_LINDQUIST
        kpolaris::KerrBoyerLindquistMetric<Real> metric(pass_a.mass, pass_a.spin);
        return kpolaris_image_detail::run_analysis_image_metric_with_optional_split(
            pass_a, analysis_config, model, radiation_substeps, timing, split_transport,
            metric, "riaf_boyer_lindquist_analysis");
#else
        throw riaf_coordinate_not_built("boyer_lindquist");
#endif
    }
    case kpolaris::CoordinateSystem::CartesianKS:
        throw riaf_coordinate_not_built("cartesian_ks");
    case kpolaris::CoordinateSystem::SphericalKS:
        throw riaf_coordinate_not_built("spherical_ks");
    case kpolaris::CoordinateSystem::FMKS:
        throw riaf_coordinate_not_built("fmks");
    case kpolaris::CoordinateSystem::MKS:
        throw riaf_coordinate_not_built("mks");
    }
    throw riaf_coordinate_not_built("unknown");
}
#endif

#if KPOLARIS_MAX_FREQUENCIES > 1
ImageHostData run_riaf_multifrequency_image(
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::RIAFAnalyticRadiationModel<Real> model,
    int radiation_substeps,
    const std::vector<Real>& frequencies,
    const std::vector<Real>& step_control_frequencies,
    int timing,
    int split_transport) {
    switch (pass_a.coordinate_system) {
    case kpolaris::CoordinateSystem::BoyerLindquist:
#if KPOLARIS_ENABLE_RIAF_COORDINATE_BOYER_LINDQUIST
        return run_riaf_multifrequency_image_boyer_lindquist(
            pass_a, model, radiation_substeps, frequencies, step_control_frequencies, timing, split_transport);
#else
        throw riaf_coordinate_not_built("boyer_lindquist");
#endif
    case kpolaris::CoordinateSystem::CartesianKS:
        throw riaf_coordinate_not_built("cartesian_ks");
    case kpolaris::CoordinateSystem::SphericalKS:
        throw riaf_coordinate_not_built("spherical_ks");
    case kpolaris::CoordinateSystem::FMKS:
        throw riaf_coordinate_not_built("fmks");
    case kpolaris::CoordinateSystem::MKS:
        throw riaf_coordinate_not_built("mks");
    }
    throw riaf_coordinate_not_built("unknown");
}
#endif
