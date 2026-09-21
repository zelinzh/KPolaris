#pragma once

#include <Kokkos_Core.hpp>
#include "diagnostics/plasma.hpp"
#include "diagnostics/response_config.hpp"
#include "radiation/emission_selection.hpp"
#include "radiation/semi_analytic.hpp"
#include "geodesic/pass_b.hpp"

namespace kpolaris {

template<class Real, class View>
KPOLARIS_INLINE Stokes<Real> response_read(const View& data, int pixel, int npix, int channel) {
    const size_t base = size_t(4) * channel * npix + pixel;
    return {data(base), data(base + npix), data(base + 2 * npix), data(base + 3 * npix)};
}

template<class Real, class View>
KPOLARIS_INLINE void response_write(const View& data, int pixel, int npix, int channel,
                                    const Stokes<Real>& s) {
    const size_t base = size_t(4) * channel * npix + pixel;
    data(base) = s.I; data(base + npix) = s.Q;
    data(base + 2 * npix) = s.U; data(base + 3 * npix) = s.V;
}

template<class Real>
KPOLARIS_INLINE Stokes<Real> response_add(const Stokes<Real>& a, const Stokes<Real>& b) {
    return {a.I + b.I, a.Q + b.Q, a.U + b.U, a.V + b.V};
}

template<class Real>
KPOLARIS_INLINE Stokes<Real> response_difference(const Stokes<Real>& a, const Stokes<Real>& b, Real denom) {
    return {(a.I - b.I)/denom, (a.Q - b.Q)/denom, (a.U - b.U)/denom, (a.V - b.V)/denom};
}

// A single homogeneous step matrix is shared by every tagged vector. Form its
// columns without emission, avoiding subtraction of a potentially bright source.
template<class Real>
struct ResponseStepMatrix {
    Stokes<Real> columns[4];
    KPOLARIS_INLINE ResponseStepMatrix(TransferCoeffs<Real> c, Real dl) {
        c.jI = c.jQ = c.jU = c.jV = Real(0);
        columns[0].I = Real(1); columns[1].Q = Real(1);
        columns[2].U = Real(1); columns[3].V = Real(1);
        for (int k=0; k<4; ++k) semi_analytic_stokes_step(columns[k], c, dl);
    }
    KPOLARIS_INLINE Stokes<Real> apply(const Stokes<Real>& s) const {
        return {columns[0].I*s.I + columns[1].I*s.Q + columns[2].I*s.U + columns[3].I*s.V,
                columns[0].Q*s.I + columns[1].Q*s.Q + columns[2].Q*s.U + columns[3].Q*s.V,
                columns[0].U*s.I + columns[1].U*s.Q + columns[2].U*s.U + columns[3].U*s.V,
                columns[0].V*s.I + columns[1].V*s.Q + columns[2].V*s.U + columns[3].V*s.V};
    }
};

template<class Real>
KPOLARIS_INLINE TransferCoeffs<Real> response_replace_block(
    TransferCoeffs<Real> base, const TransferCoeffs<Real>& variant, int mechanism) {
    if (mechanism == 0) {
        base.jI=variant.jI; base.jQ=variant.jQ; base.jU=variant.jU; base.jV=variant.jV;
    } else if (mechanism == 1) {
        base.aI=variant.aI; base.aQ=variant.aQ; base.aU=variant.aU; base.aV=variant.aV;
    } else if (mechanism == 2) {
        base.rV=variant.rV;
    } else {
        base.rQ=variant.rQ; base.rU=variant.rU;
    }
    return base;
}

template<class Real>
KPOLARIS_INLINE TransferCoeffs<Real> response_scale_coefficients(TransferCoeffs<Real> c, Real q) {
    const Real f = Kokkos::exp(q);
    c.jI*=f; c.jQ*=f; c.jU*=f; c.jV*=f;
    c.aI*=f; c.aQ*=f; c.aU*=f; c.aV*=f;
    c.rQ*=f; c.rU*=f; c.rV*=f;
    return c;
}

template<class Real>
KPOLARIS_INLINE int response_partition_bin(const ResponseConfig<Real>& config,
    Real r, Real theta, Real cp, Real sp, const AnalysisPlasmaDiagnostics<Real>& plasma) {
    if (config.partition == SourcePartition::radial) {
        int bin = 0;
        while (bin < config.bins-1 && r >= config.edges[bin+1]) ++bin;
        return bin;
    }
    const Real sight_dot = Kokkos::sin(theta)*Kokkos::sin(config.observer_theta)*
        (cp*Kokkos::cos(config.observer_phi) + sp*Kokkos::sin(config.observer_phi)) +
        Kokkos::cos(theta)*Kokkos::cos(config.observer_theta);
    const int side = sight_dot >= Real(0) ? 0 : 1;
    if (config.partition == SourcePartition::near_far) return side;
    if (config.partition == SourcePartition::region) {
        const Real polar = min_val(theta, Real(3.14159265358979323846) - theta);
        const int region = polar < config.funnel_angle ? 0 : (polar < config.disk_angle ? 1 : 2);
        return 2 * region + side;
    }
    if (config.partition == SourcePartition::plasma_region) {
        if (!plasma.valid || !plasma.magnetization_valid) return 3;
        return plasma.sigma >= config.sigma_boundary ? 0 : (plasma.beta < config.beta_boundary ? 1 : 2);
    }
    if (!plasma.valid) return config.bins-1;
    Real value = plasma.thetae;
    if (config.partition == SourcePartition::ne_cgs) value=plasma.ne_cgs;
    if (config.partition == SourcePartition::b_cgs) value=plasma.b_cgs;
    if (config.partition == SourcePartition::sigma || config.partition == SourcePartition::beta) {
        if (!plasma.magnetization_valid) return config.bins-1;
        value = config.partition == SourcePartition::sigma ? plasma.sigma : plasma.beta;
    }
    int bin=0;
    while (bin < config.edge_count && value >= config.edges[bin]) ++bin;
    return bin;
}

// Differentiate the same discrete Strang step used for the baseline. Local
// block derivatives are propagated by the full baseline operator. Their sum
// converges to dS/dq, including both dj/dq and -(dK/dq) S. Finite coefficient
// differences introduce O(h^2) error; independent full +/- reruns measure it.
template<class Real, class View>
KPOLARIS_INLINE void response_transfer_step(const ResponseConfig<Real>& config,
    const View& data, int pixel, int npix, int bin, const Stokes<Real>& incoming,
    const TransferCoeffs<Real>& base, const TransferCoeffs<Real>* variants, Real dl) {
    const ResponseStepMatrix<Real> matrix(base, dl);
    Stokes<Real> emitted;
    semi_analytic_stokes_step(emitted, base, dl);
    for (int k=0; k<config.bins; ++k) {
        auto s = matrix.apply(response_read<Real>(data, pixel, npix, k));
        if (k == bin) s = response_add(s, emitted);
        response_write(data, pixel, npix, k, s);
    }
    if (!config.responses()) return;
    for (int m=0; m<4; ++m) {
        Stokes<Real> plus=incoming, minus=incoming;
        semi_analytic_stokes_step(plus, response_replace_block(base, variants[0], m), dl);
        semi_analytic_stokes_step(minus, response_replace_block(base, variants[1], m), dl);
        const auto forcing = response_difference(plus, minus, Real(2)*config.step);
        for (int k=0; k<config.bins; ++k) {
            const int ch = config.response_channel(m,k);
            auto s = matrix.apply(response_read<Real>(data,pixel,npix,ch));
            if (k == bin) s=response_add(s,forcing);
            response_write(data,pixel,npix,ch,s);
        }
    }
    for (int v=0; v<4; ++v) {
        const int ch=config.rerun_channel(v);
        auto s=response_read<Real>(data,pixel,npix,ch);
        semi_analytic_stokes_step(s,variants[v],dl);
        response_write(data,pixel,npix,ch,s);
    }
}

template<class Metric, class Real, class Model, class View>
KPOLARIS_INLINE void accumulate_response_sample(const Metric& metric, const Model& model,
    const TransportState<Real>& state, const Stokes<Real>& incoming,
    const TransferCoeffs<Real>& base, Real dl, int screen_orientation,
    const ResponseConfig<Real>& config, const View& data, int pixel, int npix,
    const EmissionSelection<Real>& selection = {}) {
    if (!config.enabled()) return;
    Real r=Real(0), theta=Real(0), cp=Real(1), sp=Real(0);
    model.bl_coordinates_for_metric(metric,state.x,r,theta,cp,sp);
    AnalysisPlasmaDiagnostics<Real> plasma;
    if (config.partition >= SourcePartition::plasma_region && config.partition != SourcePartition::near_far)
        plasma=analysis_plasma_diagnostics(metric,model,state);
    const int bin=response_partition_bin(config,r,theta,cp,sp,plasma);
    TransferCoeffs<Real> variants[4];
    if (config.responses()) {
        for (int v=0; v<4; ++v) {
            const Real q=config.step*(v%2 == 0 ? Real(1) : Real(-1))*(v<2 ? Real(1) : Real(0.5));
            if (config.parameter == PlasmaParameter::coefficients) {
                variants[v]=response_scale_coefficients(base,q);
            } else {
                if constexpr (requires { model.coefficients(metric,state,Real(0.5),PlasmaPerturbation<Real>{}); }) {
                    variants[v]=model.coefficients(metric,state,Real(0.5),PlasmaPerturbation<Real>{config.parameter,q});
                    apply_emission_selection(selection, metric, state, variants[v]);
                    transform_axial_coefficients_to_screen_orientation(variants[v],screen_orientation);
                } else {
                    Kokkos::abort("model does not implement physical parameter perturbations");
                }
            }
        }
    }
    response_transfer_step(config,data,pixel,npix,bin,incoming,base,variants,dl);
}

template<class Real, class View>
KPOLARIS_INLINE void finalize_response_basis(const ResponseConfig<Real>& config,
    const View& data, int pixel, int npix, const BasisOverlap2<Real>& overlap) {
    for (int ch=0; ch<config.channels(); ++ch) {
        const auto s=response_read<Real>(data,pixel,npix,ch);
        response_write(data,pixel,npix,ch,transform_to_observer_basis(s,overlap));
    }
}

} // namespace kpolaris
