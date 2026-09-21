#pragma once

#include "geodesic/pass_a_core.hpp"
#include "geometry/kerr_schild_cartesian.hpp"
#include "geometry/kerr_schild_spherical.hpp"
#include "geometry/kerr_fmks.hpp"
#include "geometry/kerr_boyer_lindquist.hpp"

namespace kpolaris {

template<class Real>
KPOLARIS_INLINE PassAResult<Real> trace_pass_a_pixel(int pixel,
                                                  const PassAParams<Real>& params) {
    if (params.coordinate_system == CoordinateSystem::BoyerLindquist) {
        const KerrBoyerLindquistMetric<Real> metric(params.mass, params.spin);
        return trace_pass_a_pixel_metric(pixel, params, metric);
    }
    if (params.coordinate_system == CoordinateSystem::SphericalKS) {
        const KerrSchildSphericalMetric<Real> metric(params.mass, params.spin);
        return trace_pass_a_pixel_metric(pixel, params, metric);
    }
    if (params.coordinate_system == CoordinateSystem::MKS) {
        const KerrSchildSphericalMetric<Real> metric(params.mass, params.spin);
        return trace_pass_a_pixel_metric(pixel, params, metric);
    }
    if (params.coordinate_system == CoordinateSystem::FMKS) {
        const KerrFMKSMetric<Real> metric(params.mass, params.spin, params.fmks_startx1,
                                         params.fmks_hslope, params.fmks_mks_smooth,
                                         params.fmks_poly_alpha, params.fmks_poly_xt,
                                         params.fmks_poly_norm);
        return trace_pass_a_pixel_metric(pixel, params, metric);
    }
    const KerrSchildInMetric<Real> metric(params.mass, params.spin);
    return trace_pass_a_pixel_metric(pixel, params, metric);
}

} // namespace kpolaris
