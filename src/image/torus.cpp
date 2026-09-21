#include "image/torus.hpp"

#include <stdexcept>
#include <string>

#include "image/torus_backends.hpp"
#if KPOLARIS_ENABLE_ANALYSIS_MODE
#include "image/analysis_driver.hpp"
#include "geometry/kerr_schild_cartesian.hpp"
#endif

namespace {
using Real = kpolaris::DefaultReal;

std::runtime_error torus_coordinate_not_built(const char* coordinate) {
    return std::runtime_error(std::string("Torus image coordinate backend not built: ") +
                              coordinate + ". Reconfigure KPOLARIS_TORUS_COORDINATES to include it.");
}
} // namespace

ImageHostData run_torus_image(
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::MagnetizedTorusRadiationModel<Real> model,
    int radiation_substeps,
    int timing,
    int split_transport) {
    switch (pass_a.coordinate_system) {
    case kpolaris::CoordinateSystem::CartesianKS:
#if KPOLARIS_ENABLE_TORUS_COORDINATE_CARTESIAN_KS
        return run_torus_image_cartesian_ks(pass_a, model, radiation_substeps,
                                            timing, split_transport);
#else
        throw torus_coordinate_not_built("cartesian_ks");
#endif
    case kpolaris::CoordinateSystem::SphericalKS:
        throw torus_coordinate_not_built("spherical_ks");
    case kpolaris::CoordinateSystem::BoyerLindquist:
        throw torus_coordinate_not_built("boyer_lindquist");
    case kpolaris::CoordinateSystem::FMKS:
        throw torus_coordinate_not_built("fmks");
    case kpolaris::CoordinateSystem::MKS:
        throw torus_coordinate_not_built("mks");
    }
    throw torus_coordinate_not_built("unknown");
}

#if KPOLARIS_ENABLE_ANALYSIS_MODE
ImageHostData run_torus_analysis_image(
    const kpolaris::PassAParams<Real>& pass_a,
    const kpolaris::AnalysisConfig<Real>& analysis_config,
    kpolaris::MagnetizedTorusRadiationModel<Real> model,
    int radiation_substeps,
    int timing,
    int split_transport) {
    switch (pass_a.coordinate_system) {
    case kpolaris::CoordinateSystem::CartesianKS: {
#if KPOLARIS_ENABLE_TORUS_COORDINATE_CARTESIAN_KS
        kpolaris::KerrSchildInMetric<Real> metric(pass_a.mass, pass_a.spin);
        return kpolaris_image_detail::run_analysis_image_metric_with_optional_split(
            pass_a, analysis_config, model, radiation_substeps, timing, split_transport,
            metric, "torus_cartesian_ks_analysis");
#else
        throw torus_coordinate_not_built("cartesian_ks");
#endif
    }
    case kpolaris::CoordinateSystem::SphericalKS:
        throw torus_coordinate_not_built("spherical_ks");
    case kpolaris::CoordinateSystem::BoyerLindquist:
        throw torus_coordinate_not_built("boyer_lindquist");
    case kpolaris::CoordinateSystem::FMKS:
        throw torus_coordinate_not_built("fmks");
    case kpolaris::CoordinateSystem::MKS:
        throw torus_coordinate_not_built("mks");
    }
    throw torus_coordinate_not_built("unknown");
}
#endif

#if KPOLARIS_MAX_FREQUENCIES > 1
ImageHostData run_torus_multifrequency_image(
    const kpolaris::PassAParams<Real>& pass_a,
    kpolaris::MagnetizedTorusRadiationModel<Real> model,
    int radiation_substeps,
    const std::vector<Real>& frequencies,
    const std::vector<Real>& step_control_frequencies,
    int timing,
    int split_transport) {
    switch (pass_a.coordinate_system) {
    case kpolaris::CoordinateSystem::CartesianKS:
#if KPOLARIS_ENABLE_TORUS_COORDINATE_CARTESIAN_KS
        return run_torus_multifrequency_image_cartesian_ks(
            pass_a, model, radiation_substeps, frequencies, step_control_frequencies, timing, split_transport);
#else
        throw torus_coordinate_not_built("cartesian_ks");
#endif
    case kpolaris::CoordinateSystem::SphericalKS:
        throw torus_coordinate_not_built("spherical_ks");
    case kpolaris::CoordinateSystem::BoyerLindquist:
        throw torus_coordinate_not_built("boyer_lindquist");
    case kpolaris::CoordinateSystem::FMKS:
        throw torus_coordinate_not_built("fmks");
    case kpolaris::CoordinateSystem::MKS:
        throw torus_coordinate_not_built("mks");
    }
    throw torus_coordinate_not_built("unknown");
}
#endif
