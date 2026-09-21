#pragma once

#include "radiation/plasma_perturbation.hpp"

namespace kpolaris {

enum class SourcePartition { none = 0, radial, region, plasma_region, near_far,
                             thetae, sigma, beta, ne_cgs, b_cgs };
constexpr int max_partition_edges = 61;

template<class Real = DefaultReal>
struct ResponseConfig {
    PlasmaParameter parameter = PlasmaParameter::none;
    SourcePartition partition = SourcePartition::none;
    Real step = Real(1e-3);
    int bins = 0;
    int edge_count = 0;
    Real edges[max_partition_edges]{};
    Real funnel_angle = Real(0.3490658503988659); // 20 degrees from either spin axis
    Real disk_angle = Real(1.0471975511965976);   // 60 degrees from either spin axis
    Real sigma_boundary = Real(1);
    Real beta_boundary = Real(1);
    Real observer_theta = Real(0);
    Real observer_phi = Real(0);

    KPOLARIS_INLINE bool enabled() const { return partition != SourcePartition::none; }
    KPOLARIS_INLINE bool responses() const { return parameter != PlasmaParameter::none; }
    // Source tags; four mechanism derivatives per bin; four full transfer
    // reruns at q = +h,-h,+h/2,-h/2. Each channel contains I,Q,U,V.
    KPOLARIS_INLINE int channels() const { return enabled() ? bins * (responses() ? 5 : 1) + (responses() ? 4 : 0) : 0; }
    KPOLARIS_INLINE int response_channel(int mechanism, int bin) const { return bins + mechanism * bins + bin; }
    KPOLARIS_INLINE int rerun_channel(int variant) const { return 5 * bins + variant; }
};

} // namespace kpolaris
