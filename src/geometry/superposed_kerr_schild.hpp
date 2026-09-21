#pragma once

#include "common/vec.hpp"
#include "geometry/binary_trajectory.hpp"
#include "geometry/connection.hpp"
#include "geodesic/state.hpp"

namespace kpolaris {

// Backward-compatible name for callers that explicitly instantiate the old
// Peters-law provider.  The unified provider retains that mode and additionally
// carries the paper's device-resident 4PN trajectory tables.
template<class Real = DefaultReal>
using LeadingOrderBinaryOrbit = BinaryTrajectoryProvider<Real>;


// Superposition of two instantaneously boosted Cartesian Kerr-Schild metrics,
// following the metric ansatz of Combi & Ressler (arXiv:2403.13308):
//   g_mn = eta_mn + 2 H_1 L^(1)_m L^(1)_n
//                   + 2 H_2 L^(2)_m L^(2)_n.
// The worldlines and event-dependent masses/spins are supplied by a unified
// trajectory provider.  In the paper mode this is an HDF5 table produced from
// the modified 4PN-local CBwaves evolution; the table declares either linear
// interpolation or coupled position/velocity cubic Hermite interpolation.
template<class Real = DefaultReal>
struct SuperposedKerrSchildMetric {
    static constexpr CoordinateSystem coordinate_system = CoordinateSystem::CartesianKS;
    static constexpr Real capture_safe_step_fraction = Real(0.9);
    static constexpr Real capture_min_step_margin = Real(1.0001);
    // A half-distance limit leaves every ordinary RK abscissa comfortably on
    // the supported side of a finite trajectory-table endpoint.  The final
    // minimum-step-sized gap is classified by the terminal-event hook below.
    static constexpr Real time_domain_safe_step_fraction = Real(0.5);
    static constexpr Real time_domain_min_step_margin = Real(1.0001);

    BinaryTrajectoryProvider<Real> orbit;
    Vec3<Real> spin1 = Vec3<Real>(Real(0), Real(0), Real(0));
    Vec3<Real> spin2 = Vec3<Real>(Real(0), Real(0), Real(0));
    // Legacy Cartesian pinhole camera helpers expect a single spin member.
    // Binary images deliberately use the parallel-plane camera; this value is
    // present only so the generic camera template remains instantiable.
    Real spin = Real(0);
    // Camera rays use x^0=0 at reception.  time_origin maps that event to the
    // absolute time used by the orbit prescription.
    Real time_origin = Real(0);
    // Relative finite-difference scale used for all four metric derivatives.
    Real derivative_step = sizeof(Real) <= sizeof(float) ? Real(2e-3) : Real(2e-5);


    // The affine normalization of a null tangent is arbitrary.  Interpret the
    // user-facing minimum step as a local coordinate-displacement floor so a
    // large |k^mu| near a critical orbit does not cause a spurious underflow.
    KPOLARIS_INLINE Real effective_min_step(
        const Vec4<Real>& k,
        Real configured_min_step) const {
        Real tangent_scale = Real(1);
        for (int mu = 0; mu < ndim; ++mu) {
            tangent_scale = max_val(tangent_scale, abs_val(k[mu]));
        }
        return abs_val(configured_min_step) / tangent_scale;
    }

    struct HoleKinematics {
        // mass is the coefficient of this individual Kerr-Schild term.
        // capture_mass grows to the total remnant mass as W->1 because the
        // coincident half-mass terms then describe one physical Kerr hole.
        Real mass = Real(0);
        Real capture_mass = Real(0);
        Real merger_weight = Real(0);
        Vec3<Real> spin;
        Vec3<Real> position;
        Vec3<Real> velocity;
        Real gamma = Real(1);
    };

    struct HoleWork {
        HoleKinematics hole;
        Vec4<Real> rest_x;
        Real r = Real(0);
        Real amp = Real(0);
        Real l_rest[ndim];
        Real l_global[ndim];
    };

    struct Work {
        // Barycentric radius is used only for the outer integration domain.
        Real r = Real(0);
        typename BinaryTrajectoryProvider<Real>::State trajectory;
        HoleWork holes[2];
    };

    KPOLARIS_INLINE SuperposedKerrSchildMetric() {}

    KPOLARIS_INLINE Real eta(int mu, int nu) const {
        if (mu != nu) return Real(0);
        return mu == 0 ? Real(-1) : Real(1);
    }

    KPOLARIS_INLINE Real absolute_time(const Vec4<Real>& x) const {
        return time_origin + x[0];
    }

    KPOLARIS_INLINE int time_domain_valid(const Vec4<Real>& x) const {
        return orbit.has_state(absolute_time(x)) ? 1 : 0;
    }

    // Positive affine distance, in the direction of the signed proposed
    // step, to the next finite table-time boundary.  A negative value means
    // that this metric/direction has no finite temporal endpoint.  The exact
    // native Kerr-remnant continuation removes only the upper boundary; the
    // lower table boundary remains strict.
    KPOLARIS_INLINE Real time_domain_boundary_distance(
        const Vec4<Real>& x,
        const Vec4<Real>& k,
        Real proposed_h) const {
        if (!orbit.is_tabulated()) return Real(-1);
        const Real direction = proposed_h < Real(0) ? Real(-1) : Real(1);
        const Real time_rate = direction * k[0];
        if (!(time_rate == time_rate) ||
            abs_val(time_rate) >= large_positive<Real>()) {
            return Real(-1);
        }
        const Real time = absolute_time(x);
        if (!orbit.has_state(time)) return Real(0);
        if (time_rate < -tiny_positive<Real>()) {
            return max_val(Real(0),
                           (time - orbit.table_time_min) / (-time_rate));
        }
        const int has_unbounded_future =
            orbit.has_exact_postmerger_tail &&
            orbit.enable_future_postmerger_extension;
        if (time_rate > tiny_positive<Real>() && !has_unbounded_future) {
            return max_val(Real(0),
                           (orbit.table_time_max - time) / time_rate);
        }
        return Real(-1);
    }

    KPOLARIS_INLINE Real time_domain_step_limit(
        const Vec4<Real>& x,
        const Vec4<Real>& k,
        Real proposed_h) const {
        const Real proposed = abs_val(proposed_h);
        const Real distance = time_domain_boundary_distance(
            x, k, proposed_h);
        if (!(distance >= Real(0))) return proposed;
        return min_val(proposed,
                       time_domain_safe_step_fraction * distance);
    }

    KPOLARIS_INLINE int time_domain_step_is_terminal(
        const Vec4<Real>& x,
        const Vec4<Real>& k,
        Real proposed_h,
        Real event_tolerance) const {
        const Real distance = time_domain_boundary_distance(
            x, k, proposed_h);
        if (!(distance >= Real(0))) return 0;
        const Real window =
            time_domain_min_step_margin / time_domain_safe_step_fraction *
            abs_val(event_tolerance);
        return distance <= window;
    }

    KPOLARIS_INLINE HoleKinematics hole_kinematics_from_state(
        int index,
        const typename BinaryTrajectoryProvider<Real>::State& os) const {
        HoleKinematics out;
        out.mass = os.mass(index);
        out.position = os.position(index);
        out.velocity = os.velocity(index);
        out.spin = orbit.is_tabulated() ? os.kerr_a(index) :
            (index == 0 ? spin1 : spin2);
        const Real W = clamp(os.merger_weight, Real(0), Real(1));
        out.merger_weight = W;
        const Real total_mass = max_val(os.mass1 + os.mass2, Real(0));
        // No true individual horizons exist during the algebraic merger
        // transition.  This is a numerical excision/capture proxy, not an
        // apparent-horizon finder.  It reaches the component mass at W=0 and
        // the common remnant mass at W=1, while remaining subextremal when
        // a(t) grows faster than an unequal-mass SKS term.  Endpoints retain
        // their exact Kerr meaning.
        const Real blended_capture_mass =
            (Real(1) - W) * out.mass + W * total_mass;
        const Real spin_magnitude = kpolaris::norm(out.spin);
        if (W > Real(0) && W < Real(1)) {
            // A differentiable maximum avoids a kink in the moving capture
            // event function (and therefore in adaptive event localization).
            // The smoothing scale vanishes at both exact physical endpoints.
            const Real smoothing_fraction = sizeof(Real) <= sizeof(float) ?
                Real(2e-4) : Real(2e-8);
            const Real epsilon = smoothing_fraction *
                max_val(total_mass, Real(1)) * W * (Real(1) - W);
            const Real difference = blended_capture_mass - spin_magnitude;
            out.capture_mass = Real(0.5) *
                (blended_capture_mass + spin_magnitude +
                 Kokkos::sqrt(difference * difference +
                              epsilon * epsilon));
        } else {
            out.capture_mass = blended_capture_mass;
        }
        const Real v2 = kpolaris::dot(out.velocity, out.velocity);
        // Host validation reserves a precision-dependent margin below c.  Do
        // not clamp gamma independently of v: that would cease to be the
        // Lorentz transformation associated with the supplied worldline.
        out.gamma = Real(1) / Kokkos::sqrt(Real(1) - v2);
        return out;
    }

    KPOLARIS_INLINE HoleKinematics hole_kinematics(int index,
                                                  const Vec4<Real>& x) const {
        const typename BinaryTrajectoryProvider<Real>::State os =
            orbit.state(absolute_time(x));
        return hole_kinematics_from_state(index, os);
    }

    KPOLARIS_INLINE void lab_to_rest_spatial(const HoleKinematics& hole,
                                            const Vec4<Real>& x,
                                            Vec4<Real>& rest_x) const {
        const Vec3<Real> dx(x[1] - hole.position.x,
                            x[2] - hole.position.y,
                            x[3] - hole.position.z);
        const Real v2 = kpolaris::dot(hole.velocity, hole.velocity);
        const Real vdotx = kpolaris::dot(hole.velocity, dx);
        const Real boost_factor = v2 > tiny_positive<Real>() ?
            (hole.gamma - Real(1)) * vdotx / v2 : Real(0);
        rest_x[0] = Real(0);
        rest_x[1] = dx.x + boost_factor * hole.velocity.x;
        rest_x[2] = dx.y + boost_factor * hole.velocity.y;
        rest_x[3] = dx.z + boost_factor * hole.velocity.z;
    }

    KPOLARIS_INLINE Vec4<Real> rest_to_lab_vector(const HoleKinematics& hole,
                                                 const Vec4<Real>& vrest) const {
        const Real v2 = kpolaris::dot(hole.velocity, hole.velocity);
        const Real vdotu = hole.velocity.x * vrest[1] +
                           hole.velocity.y * vrest[2] +
                           hole.velocity.z * vrest[3];
        const Real factor = v2 > tiny_positive<Real>() ?
            (hole.gamma - Real(1)) * vdotu / v2 : Real(0);
        Vec4<Real> out;
        out[0] = hole.gamma * (vrest[0] + vdotu);
        out[1] = vrest[1] + factor * hole.velocity.x +
                 hole.gamma * hole.velocity.x * vrest[0];
        out[2] = vrest[2] + factor * hole.velocity.y +
                 hole.gamma * hole.velocity.y * vrest[0];
        out[3] = vrest[3] + factor * hole.velocity.z +
                 hole.gamma * hole.velocity.z * vrest[0];
        return out;
    }

    KPOLARIS_INLINE Vec4<Real> lab_to_rest_vector(const HoleKinematics& hole,
                                                 const Vec4<Real>& vlab) const {
        const Real v2 = kpolaris::dot(hole.velocity, hole.velocity);
        const Real vdotu = hole.velocity.x * vlab[1] +
                           hole.velocity.y * vlab[2] +
                           hole.velocity.z * vlab[3];
        const Real factor = v2 > tiny_positive<Real>() ?
            (hole.gamma - Real(1)) * vdotu / v2 : Real(0);
        Vec4<Real> out;
        out[0] = hole.gamma * (vlab[0] - vdotu);
        out[1] = vlab[1] + factor * hole.velocity.x -
                 hole.gamma * hole.velocity.x * vlab[0];
        out[2] = vlab[2] + factor * hole.velocity.y -
                 hole.gamma * hole.velocity.y * vlab[0];
        out[3] = vlab[3] + factor * hole.velocity.z -
                 hole.gamma * hole.velocity.z * vlab[0];
        return out;
    }

    KPOLARIS_INLINE HoleWork build_hole_work_from_state(
        int index,
        const Vec4<Real>& x,
        const typename BinaryTrajectoryProvider<Real>::State& os) const {
        HoleWork w;
        w.hole = hole_kinematics_from_state(index, os);
        lab_to_rest_spatial(w.hole, x, w.rest_x);

        const Vec3<Real> X(w.rest_x[1], w.rest_x[2], w.rest_x[3]);
        const Vec3<Real> a = w.hole.spin;
        const Real a2 = kpolaris::dot(a, a);
        const Real x2 = kpolaris::dot(X, X);
        const Real adx = kpolaris::dot(a, X);
        const Real q = x2 - a2;
        const Real discr = Kokkos::sqrt(max_val(q * q + Real(4) * adx * adx,
                                                Real(0)));
        const Real r2 = max_val(Real(0.5) * (q + discr), tiny_positive<Real>());
        const Real r = Kokkos::sqrt(r2);
        const Real denom = max_val(r2 * r2 + adx * adx, tiny_positive<Real>());
        w.r = r;
        w.amp = Real(2) * w.hole.mass * r2 * r / denom;

        const Vec3<Real> axx = cross(a, X);
        const Real spatial_denom = max_val(r2 + a2, tiny_positive<Real>());
        w.l_rest[0] = Real(1);
        w.l_rest[1] = (r * X.x - axx.x + adx * a.x / r) / spatial_denom;
        w.l_rest[2] = (r * X.y - axx.y + adx * a.y / r) / spatial_denom;
        w.l_rest[3] = (r * X.z - axx.z + adx * a.z / r) / spatial_denom;

        // Covector transformation L_mu = (d X^a / d x^mu) l_a for the
        // instantaneous Lorentz boost from the barycentric frame to the hole.
        const Vec3<Real> v = w.hole.velocity;
        const Real v2 = kpolaris::dot(v, v);
        const Real vdotl = v.x * w.l_rest[1] + v.y * w.l_rest[2] +
                           v.z * w.l_rest[3];
        w.l_global[0] = w.hole.gamma * (w.l_rest[0] - vdotl);
        const Real factor = v2 > tiny_positive<Real>() ?
            (w.hole.gamma - Real(1)) * vdotl / v2 : Real(0);
        w.l_global[1] = w.l_rest[1] + factor * v.x -
                        w.hole.gamma * v.x * w.l_rest[0];
        w.l_global[2] = w.l_rest[2] + factor * v.y -
                        w.hole.gamma * v.y * w.l_rest[0];
        w.l_global[3] = w.l_rest[3] + factor * v.z -
                        w.hole.gamma * v.z * w.l_rest[0];
        return w;
    }

    KPOLARIS_INLINE HoleWork build_hole_work(int index,
                                            const Vec4<Real>& x) const {
        const typename BinaryTrajectoryProvider<Real>::State os =
            orbit.state(absolute_time(x));
        return build_hole_work_from_state(index, x, os);
    }

    KPOLARIS_INLINE Work build_work(const Vec4<Real>& x) const {
        Work w;
        w.r = Kokkos::sqrt(x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
        w.trajectory = orbit.state(absolute_time(x));
        w.holes[0] = build_hole_work_from_state(0, x, w.trajectory);
        w.holes[1] = build_hole_work_from_state(1, x, w.trajectory);
        return w;
    }

    KPOLARIS_INLINE Real radial_coordinate(const Vec4<Real>& x) const {
        return Kokkos::sqrt(x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
    }

    KPOLARIS_INLINE Real horizon_radius_from_kinematics(
        const HoleKinematics& hole) const {
        const Real m = max_val(hole.capture_mass, Real(0));
        return m + Kokkos::sqrt(max_val(
            m * m - kpolaris::dot(hole.spin, hole.spin), Real(0)));
    }

    KPOLARIS_INLINE Real hole_horizon_radius(int index,
                                             const Vec4<Real>& x) const {
        return horizon_radius_from_kinematics(hole_kinematics(index, x));
    }

    // Compatibility overload: evaluate at the camera reception event.
    KPOLARIS_INLINE Real hole_horizon_radius(int index) const {
        const Vec4<Real> x(Real(0), Real(0), Real(0), Real(0));
        return hole_horizon_radius(index, x);
    }

    // The spheroidal Kerr radius without constructing H or the KS null form.
    // This is used by the time component of dr_mu, so keeping it separate from
    // build_hole_work avoids two unnecessary metric evaluations per hole.
    KPOLARIS_INLINE Real hole_radius_only(int index,
                                         const Vec4<Real>& x) const {
        const HoleKinematics hole = hole_kinematics(index, x);
        Vec4<Real> rest_x;
        lab_to_rest_spatial(hole, x, rest_x);
        const Vec3<Real> X(rest_x[1], rest_x[2], rest_x[3]);
        const Real a2 = kpolaris::dot(hole.spin, hole.spin);
        const Real x2 = kpolaris::dot(X, X);
        const Real adx = kpolaris::dot(hole.spin, X);
        const Real spheroidal_q = x2 - a2;
        const Real discr = Kokkos::sqrt(max_val(
            spheroidal_q * spheroidal_q + Real(4) * adx * adx, Real(0)));
        return Kokkos::sqrt(max_val(
            Real(0.5) * (spheroidal_q + discr), tiny_positive<Real>()));
    }

    // Full coordinate four-gradient of the moving hole's local spheroidal
    // Kerr radius.  Spatial derivatives are analytic.  The time derivative is
    // centered in coordinate time so that translation, instantaneous boost,
    // acceleration and inspiral are all included in the same radius function
    // used by build_hole_work.  A local orbital time scale, rather than the
    // absolute coordinate time, keeps the subtraction well conditioned at
    // late simulation times.
    KPOLARIS_INLINE void hole_radius_gradient(int index,
                                              const Vec4<Real>& x,
                                              const HoleWork& w,
                                              Vec4<Real>& dr) const {
        const Vec3<Real> X(w.rest_x[1], w.rest_x[2], w.rest_x[3]);
        const Vec3<Real> a = w.hole.spin;
        const Real a2 = kpolaris::dot(a, a);
        const Real x2 = kpolaris::dot(X, X);
        const Real adx = kpolaris::dot(a, X);
        const Real r = max_val(w.r, tiny_positive<Real>());
        const Real r2 = r * r;
        // 2 r^2 - (X^2-a^2) is the positive square-root discriminant
        // in the implicit Kerr-radius equation.
        const Real discr = max_val(Real(2) * r2 - (x2 - a2),
                                   tiny_positive<Real>());
        const Real inv_r_discr = Real(1) / (r * discr);
        const Vec3<Real> grad_rest(
            (X.x * r2 + adx * a.x) * inv_r_discr,
            (X.y * r2 + adx * a.y) * inv_r_discr,
            (X.z * r2 + adx * a.z) * inv_r_discr);

        // X = [I + (gamma-1) vv^T/v^2] (x-z(t)) at fixed t.  The
        // spatial Jacobian is symmetric, so its transpose has this form too.
        const Vec3<Real> v = w.hole.velocity;
        const Real v2 = kpolaris::dot(v, v);
        const Real vdot_grad = kpolaris::dot(v, grad_rest);
        const Real boost_factor = v2 > tiny_positive<Real>() ?
            (w.hole.gamma - Real(1)) * vdot_grad / v2 : Real(0);
        dr[1] = grad_rest.x + boost_factor * v.x;
        dr[2] = grad_rest.y + boost_factor * v.y;
        dr[3] = grad_rest.z + boost_factor * v.z;

        // A tabulated provider is dynamic even though its legacy
        // orbit_enabled flag is intentionally unused.  Only the explicitly
        // static leading-order mode has a vanishing time derivative.
        if (!orbit.is_tabulated() && !orbit.orbit_enabled) {
            dr[0] = Real(0);
            return;
        }
        const typename LeadingOrderBinaryOrbit<Real>::State os =
            orbit.state(absolute_time(x));
        const Real orbital_speed = abs_val(os.separation_rate) +
                                   abs_val(os.separation * os.angular_frequency);
        Real time_scale = os.separation /
            max_val(orbital_speed, Real(1e-3));
        time_scale = min_val(Real(1e3), max_val(Real(1), time_scale));
        const Real precision_floor = sizeof(Real) <= sizeof(float) ?
            Real(2e-4) : Real(2e-8);
        const Real ht = max_val(derivative_step * time_scale, precision_floor);
        Vec4<Real> xp = x;
        Vec4<Real> xm = x;
        xp[0] += ht;
        xm[0] -= ht;
        const int plus_valid = time_domain_valid(xp);
        const int minus_valid = time_domain_valid(xm);
        if (plus_valid && minus_valid) {
            dr[0] = (hole_radius_only(index, xp) -
                     hole_radius_only(index, xm)) / (Real(2) * ht);
        } else if (plus_valid) {
            dr[0] = (hole_radius_only(index, xp) - w.r) / ht;
        } else if (minus_valid) {
            dr[0] = (w.r - hole_radius_only(index, xm)) / ht;
        } else {
            dr[0] = Real(0);
        }
    }

    // Exact inverse from an already-built Work object.  The capture surrogate
    // needs the inverse and both local radii at the same event; sharing Work
    // avoids rebuilding all Kerr-Schild terms on every near-hole query.
    KPOLARIS_INLINE void gcon_matrix_from_work(const Work& w,
                                              Real ginv[ndim][ndim]) const {
        Real u[2][ndim];
        Real v[2][ndim];
        for (int h = 0; h < 2; ++h) {
            const Real root_amp = Kokkos::sqrt(max_val(w.holes[h].amp, Real(0)));
            for (int mu = 0; mu < ndim; ++mu) {
                u[h][mu] = root_amp * w.holes[h].l_global[mu];
                v[h][mu] = (mu == 0 ? -u[h][mu] : u[h][mu]);
            }
        }
        Real s00 = Real(1), s01 = Real(0), s11 = Real(1);
        for (int mu = 0; mu < ndim; ++mu) {
            s00 += u[0][mu] * v[0][mu];
            s01 += u[0][mu] * v[1][mu];
            s11 += u[1][mu] * v[1][mu];
        }
        const Real det = s00 * s11 - s01 * s01;
        const Real inv00 = s11 / det;
        const Real inv01 = -s01 / det;
        const Real inv11 = s00 / det;
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                ginv[mu][nu] = eta(mu, nu) -
                    v[0][mu] * (inv00 * v[0][nu] + inv01 * v[1][nu]) -
                    v[1][mu] * (inv01 * v[0][nu] + inv11 * v[1][nu]);
            }
        }
    }

    // Apply the exact Woodbury inverse to one covector without materializing
    // the 4x4 inverse.  The Binary RIAF frame needs only g^{-1}q; keeping this
    // contraction explicit materially reduces CUDA per-thread stack pressure.
    KPOLARIS_INLINE void raise_covector_from_work(
        const Work& w,
        const Vec4<Real>& q,
        Vec4<Real>& raised) const {
        Real u[2][ndim];
        Real v[2][ndim];
        for (int h = 0; h < 2; ++h) {
            const Real root_amp = Kokkos::sqrt(max_val(w.holes[h].amp, Real(0)));
            for (int mu = 0; mu < ndim; ++mu) {
                u[h][mu] = root_amp * w.holes[h].l_global[mu];
                v[h][mu] = (mu == 0 ? -u[h][mu] : u[h][mu]);
            }
        }
        Real s00 = Real(1), s01 = Real(0), s11 = Real(1);
        Real vq[2] = {Real(0), Real(0)};
        for (int mu = 0; mu < ndim; ++mu) {
            s00 += u[0][mu] * v[0][mu];
            s01 += u[0][mu] * v[1][mu];
            s11 += u[1][mu] * v[1][mu];
            vq[0] += v[0][mu] * q[mu];
            vq[1] += v[1][mu] * q[mu];
        }
        const Real det = s00 * s11 - s01 * s01;
        const Real weight0 = (s11 * vq[0] - s01 * vq[1]) / det;
        const Real weight1 = (-s01 * vq[0] + s00 * vq[1]) / det;
        for (int mu = 0; mu < ndim; ++mu) {
            raised[mu] = (mu == 0 ? -q[mu] : q[mu]) -
                v[0][mu] * weight0 - v[1][mu] * weight1;
        }
    }

    // Per-hole full-metric radial-characteristic capture surrogate.  It is
    // deliberately not called an apparent/event horizon: q=g^{mu nu}dr_mu
    // dr_nu diagnoses the characteristic direction of the chosen moving
    // local-r foliation, not the nullness of the q=constant hypersurface.
    //
    // The target is q for an isolated Kerr hole at r=horizon_factor*r_+
    // and at the event's spheroidal polar angle.  Consequently the zero set
    // reduces exactly to the former r=horizon_factor*r_+ surface in the
    // isolated-hole limit, while the companion's full metric automatically
    // introduces angular and time dependence in a binary.
    KPOLARIS_INLINE Real hole_capture_surrogate_value(
        int index,
        const Vec4<Real>& x,
        const HoleWork& w,
        const Real ginv[ndim][ndim],
        Real horizon_factor) const {
        Vec4<Real> dr;
        hole_radius_gradient(index, x, w, dr);
        Real radial_characteristic = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                radial_characteristic += ginv[mu][nu] * dr[mu] * dr[nu];
            }
        }

        // Use the kinematics already evaluated at this event.  In the exact
        // post-merger representation each SKS term carries half of M_f, but
        // the coincident pair is one Kerr hole; capture_mass therefore equals
        // M_f rather than the coefficient of either individual rank-one term.
        const Real horizon = max_val(
            horizon_radius_from_kinematics(w.hole), tiny_positive<Real>());
        const Real target_r = horizon_factor * horizon;
        const Real target_r2 = target_r * target_r;
        const Real spin2 = kpolaris::dot(w.hole.spin, w.hole.spin);
        const Real adx = w.hole.spin.x * w.rest_x[1] +
                         w.hole.spin.y * w.rest_x[2] +
                         w.hole.spin.z * w.rest_x[3];
        const Real local_r = max_val(w.r, tiny_positive<Real>());
        const Real angular_term = (adx / local_r) * (adx / local_r);
        const Real target_sigma = max_val(target_r2 + angular_term,
                                          tiny_positive<Real>());
        const Real target_delta = target_r2 -
            Real(2) * w.hole.capture_mass * target_r + spin2;
        const Real target_characteristic = target_delta / target_sigma;

        // Normalize by the isolated q slope with respect to r/R0.  This is
        // immaterial to the zero set, but keeps F O(1) and event interpolation
        // well scaled for unequal masses and high spins.
        const Real dq_dr = ((Real(2) * target_r -
                             Real(2) * w.hole.capture_mass) *
                            target_sigma -
                            Real(2) * target_r * target_delta) /
                           (target_sigma * target_sigma);
        const Real slope_floor = sizeof(Real) <= sizeof(float) ?
            Real(1e-4) : Real(1e-10);
        const Real radial_scale = max_val(abs_val(target_r * dq_dr), slope_floor);
        return (radial_characteristic - target_characteristic) / radial_scale;
    }

    KPOLARIS_INLINE Real capture_surrogate_guard_ratio(
        Real horizon_factor) const {
        return max_val(Real(1.3), horizon_factor + Real(0.1));
    }

    // Positive outside the union of the two capture surrogates and non-positive
    // inside.  Most ray steps are far from either hole; beyond 2.5 local
    // horizon radii the old radius value is already safely positive and avoids
    // the inverse metric plus four-gradient work entirely.
    KPOLARIS_INLINE Real inner_boundary_value(const Vec4<Real>& x,
                                              Real horizon_factor) const {
        const Work w = build_work(x);
        Real radial_value[2];
        int near_hole[2];
        int any_near = 0;
        for (int h = 0; h < 2; ++h) {
            if (!(w.holes[h].hole.mass > tiny_positive<Real>())) {
                radial_value[h] = large_positive<Real>();
                near_hole[h] = 0;
                continue;
            }
            const Real horizon = max_val(
                horizon_radius_from_kinematics(w.holes[h].hole),
                tiny_positive<Real>());
            const Real ratio = w.holes[h].r / horizon;
            const Real capture_ratio =
                max_val(horizon_factor, tiny_positive<Real>());
            radial_value[h] = ratio / capture_ratio - Real(1);
            // q-target is an exterior positive-buffer construction.  Retain
            // the old radial boundary for unsupported factors at/below r_+;
            // host-side binary parameter validation requires factor > 1.
            near_hole[h] = horizon_factor > Real(1) && ratio < Real(2.5) ? 1 : 0;
            any_near |= near_hole[h];
        }
        if (!any_near) {
            return min_val(radial_value[0], radial_value[1]);
        }

        Real ginv[ndim][ndim];
        gcon_matrix_from_work(w, ginv);
        Real value[2] = {radial_value[0], radial_value[1]};
        for (int h = 0; h < 2; ++h) {
            if (near_hole[h]) {
                const Real horizon = max_val(
                    horizon_radius_from_kinematics(w.holes[h].hole),
                    tiny_positive<Real>());
                const Real characteristic_value = hole_capture_surrogate_value(
                    h, x, w.holes[h], ginv, horizon_factor);
                // q can turn upward below the Kerr inner root and a generic
                // superposition can also create a disconnected exterior
                // negative shell.  A=r/R0-1 keeps r<=R0 captured, while
                // G=r/Rguard-1 forces the exterior branch positive beyond a
                // conservative local guard radius.  For the supported
                // factor<=1.05 range, Rguard=1.3 r_+ leaves ample room for the
                // full-metric outward deformation.
                const Real guard_ratio =
                    capture_surrogate_guard_ratio(horizon_factor);
                const Real exterior_guard =
                    w.holes[h].r / (guard_ratio * horizon) - Real(1);
                value[h] = min_val(
                    radial_value[h],
                    max_val(characteristic_value, exterior_guard));
            }
        }
        return min_val(value[0], value[1]);
    }

    // Positive affine distance to an inward crossing of F=0 from the local
    // directional derivative, or a negative sentinel when none is predicted.
    KPOLARIS_INLINE Real inner_boundary_inward_crossing_distance(
        const Vec4<Real>& x,
        const Vec4<Real>& k,
        Real horizon_factor) const {
        const Real f0 = inner_boundary_value(x, horizon_factor);
        if (!(f0 > Real(0)) || f0 > Real(0.5)) {
            return Real(-1);
        }
        const Real kscale = max_abs_component(k);
        if (!(kscale > tiny_positive<Real>())) {
            return Real(-1);
        }
        const Real h1 = hole_horizon_radius(0, x);
        const Real h2 = hole_horizon_radius(1, x);
        const Real local_length = max_val(Real(1), max_val(h1, h2));
        const Real coordinate_probe = max_val(
            derivative_step * local_length,
            sizeof(Real) <= sizeof(float) ? Real(2e-4) : Real(2e-8));
        const Real dl = coordinate_probe / kscale;
        Vec4<Real> xp = x;
        Vec4<Real> xm = x;
        for (int mu = 0; mu < ndim; ++mu) {
            xp[mu] += dl * k[mu];
            xm[mu] -= dl * k[mu];
        }
        const int plus_valid = time_domain_valid(xp);
        const int minus_valid = time_domain_valid(xm);
        Real dfdlambda = Real(0);
        if (plus_valid && minus_valid) {
            dfdlambda =
                (inner_boundary_value(xp, horizon_factor) -
                 inner_boundary_value(xm, horizon_factor)) / (Real(2) * dl);
        } else if (plus_valid) {
            dfdlambda =
                (inner_boundary_value(xp, horizon_factor) - f0) / dl;
        } else if (minus_valid) {
            dfdlambda =
                (f0 - inner_boundary_value(xm, horizon_factor)) / dl;
        } else {
            return Real(-1);
        }
        if (!(dfdlambda < Real(0))) {
            return Real(-1);
        }
        return f0 / (-dfdlambda);
    }

    // Near the capture surrogate, stop at 90% of the predicted crossing.
    // The accepted RK trial therefore remains on the physical exterior side;
    // the explicit terminal-event hook handles the final minimum-step gap.
    KPOLARIS_INLINE Real inner_boundary_step_limit(
        const Vec4<Real>& x,
        const Vec4<Real>& k,
        Real horizon_factor,
        Real proposed_abs_h) const {
        const Real proposed = abs_val(proposed_abs_h);
        const Real crossing_distance =
            inner_boundary_inward_crossing_distance(x, k, horizon_factor);
        if (!(crossing_distance > Real(0))) {
            return proposed;
        }
        const Real numerical_floor = sizeof(Real) <= sizeof(float) ?
            Real(1e-7) : Real(1e-14);
        const Real limited = max_val(
            capture_safe_step_fraction * crossing_distance, numerical_floor);
        return min_val(proposed, limited);
    }

    // Terminate before constructing an RK trial once a proven transverse
    // crossing is close enough that the 90%-safe step would reach the adaptive
    // minimum.  The 1.0001 factor matches the underflow comparison; the forward
    // sign probe rejects a tangent contact, which is positive again after touch.
    KPOLARIS_INLINE int inner_boundary_step_is_terminal(
        const Vec4<Real>& x,
        const Vec4<Real>& k,
        Real horizon_factor,
        Real event_tolerance) const {
        const Real crossing_distance =
            inner_boundary_inward_crossing_distance(x, k, horizon_factor);
        const Real window =
            capture_min_step_margin / capture_safe_step_fraction *
            abs_val(event_tolerance);
        if (!(crossing_distance > Real(0)) || crossing_distance > window) {
            return 0;
        }
        Vec4<Real> probe = x;
        for (int mu = 0; mu < ndim; ++mu) {
            probe[mu] += Real(2) * crossing_distance * k[mu];
        }
        if (!time_domain_valid(probe)) return 0;
        const Real probe_value = inner_boundary_value(probe, horizon_factor);
        return probe_value == probe_value &&
               abs_val(probe_value) < large_positive<Real>() &&
               probe_value <= Real(0);
    }

    KPOLARIS_INLINE void gcov_matrix_from_work(const Work& w,
                                              Real g[ndim][ndim]) const {
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) {
                g[mu][nu] = eta(mu, nu);
                for (int h = 0; h < 2; ++h) {
                    g[mu][nu] += w.holes[h].amp *
                        w.holes[h].l_global[mu] * w.holes[h].l_global[nu];
                }
            }
        }
    }

    KPOLARIS_INLINE void gcov_matrix(const Vec4<Real>& x,
                                    Real g[ndim][ndim]) const {
        const Work w = build_work(x);
        gcov_matrix_from_work(w, g);
    }

    KPOLARIS_INLINE Real gcov(int mu, int nu, const Vec4<Real>& x) const {
        const Work w = build_work(x);
        Real out = eta(mu, nu);
        for (int h = 0; h < 2; ++h) {
            out += w.holes[h].amp * w.holes[h].l_global[mu] *
                   w.holes[h].l_global[nu];
        }
        return out;
    }

    KPOLARIS_INLINE Real inverse_denominator(const Vec4<Real>& x) const {
        const Work w = build_work(x);
        Real cross_null = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            cross_null += (mu == 0 ? Real(-1) : Real(1)) *
                w.holes[0].l_global[mu] * w.holes[1].l_global[mu];
        }
        return Real(1) - w.holes[0].amp * w.holes[1].amp *
            cross_null * cross_null;
    }

    KPOLARIS_INLINE void gcon_matrix(const Vec4<Real>& x,
                                    Real ginv[ndim][ndim]) const {
        // Exact rank-two Woodbury inverse of eta + U U^T, with
        // U_mu,A=sqrt(2H_A)L_mu,A.  This avoids a device-side generic 4x4
        // elimination and reduces exactly to the analytic single-hole inverse.
        // A non-positive Woodbury determinant marks an invalid superposition;
        // host-side parameter validation excludes that domain.
        const Work w = build_work(x);
        gcon_matrix_from_work(w, ginv);
    }

    KPOLARIS_INLINE Real gcon(int mu, int nu, const Vec4<Real>& x) const {
        Real g[ndim][ndim];
        gcon_matrix(x, g);
        return g[mu][nu];
    }

    KPOLARIS_INLINE Real dot(const Vec4<Real>& x,
                           const Vec4<Real>& a,
                           const Vec4<Real>& b) const {
        const Work w = build_work(x);
        Real out = -a[0] * b[0] + a[1] * b[1] +
                   a[2] * b[2] + a[3] * b[3];
        for (int h = 0; h < 2; ++h) {
            Real la = Real(0), lb = Real(0);
            for (int mu = 0; mu < ndim; ++mu) {
                la += w.holes[h].l_global[mu] * a[mu];
                lb += w.holes[h].l_global[mu] * b[mu];
            }
            out += w.holes[h].amp * la * lb;
        }
        return out;
    }

    KPOLARIS_INLINE void dgcov(const Vec4<Real>& x,
                              Real dg[ndim][ndim][ndim]) const {
        for (int alpha = 0; alpha < ndim; ++alpha) {
            const Real scale = max_val(Real(1), abs_val(x[alpha]));
            const Real h = max_val(derivative_step * scale,
                                   sizeof(Real) <= sizeof(float) ?
                                       Real(1e-5) : Real(1e-8));
            Vec4<Real> xp = x;
            Vec4<Real> xm = x;
            xp[alpha] += h;
            xm[alpha] -= h;
            Real gp[ndim][ndim], gm[ndim][ndim], g0[ndim][ndim];
            const int plus_valid = alpha != 0 || time_domain_valid(xp);
            const int minus_valid = alpha != 0 || time_domain_valid(xm);
            if (plus_valid) gcov_matrix(xp, gp);
            if (minus_valid) gcov_matrix(xm, gm);
            if (!plus_valid || !minus_valid) gcov_matrix(x, g0);
            for (int mu = 0; mu < ndim; ++mu) {
                for (int nu = 0; nu < ndim; ++nu) {
                    if (plus_valid && minus_valid) {
                        dg[mu][nu][alpha] =
                            (gp[mu][nu] - gm[mu][nu]) * (Real(0.5) / h);
                    } else if (plus_valid) {
                        dg[mu][nu][alpha] =
                            (gp[mu][nu] - g0[mu][nu]) / h;
                    } else if (minus_valid) {
                        dg[mu][nu][alpha] =
                            (g0[mu][nu] - gm[mu][nu]) / h;
                    } else {
                        dg[mu][nu][alpha] = Real(0);
                    }
                }
            }
        }
    }

    KPOLARIS_INLINE void connection(const Vec4<Real>& x,
                                   Real gamma[ndim][ndim][ndim]) const {
        Real ginv[ndim][ndim];
        Real dg[ndim][ndim][ndim];
        gcon_matrix(x, ginv);
        dgcov(x, dg);
        fill_connection_from_metric_derivs(ginv, dg, gamma);
    }
};

} // namespace kpolaris
