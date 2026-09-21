#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "geometry/kerr_schild_cartesian.hpp"
#include "geometry/superposed_kerr_schild.hpp"
#include "model/riaf.hpp"

namespace kpolaris {

// Two tidally truncated analytic mini-RIAFs, each constructed in the
// instantaneous rest frame of its own hole.  The plasma components remain
// independent: their transfer coefficients are projected into the common
// transported screen basis and only then added.
template<class Real = DefaultReal>
struct BinaryRIAFRadiationModel {
    RIAFAnalyticRadiationModel<Real> disk;
    // Device-resident paper trajectory.  The default-constructed provider is
    // the legacy leading-quadrupole orbit; the HDF5 loader switches it to the
    // tabulated 4PN/merger mode without introducing host pointers into kernels.
    BinaryTrajectoryProvider<Real> trajectory;
    // Kept at the model level so generic fused multi-frequency transport can
    // specialize a copied model without mutating the shared disk parameters.
    Real freq_cgs = Real(230e9);
    // Host-side admissibility audit results.  Negative means not yet sampled.
    Real sampled_min_inverse_denominator = Real(-1);
    Real sampled_min_fluid_slice_timelike_margin = Real(-1);

    // q = m2/m1.  The total code mass is one.
    Real mass_ratio = Real(1);
    Real chi1 = Real(0);
    Real chi2 = Real(0);
    Real reference_separation = Real(20);
    Real reference_phase = Real(0);
    Real reference_time = Real(0);
    Real observation_time = Real(0);
    Real trajectory_time_offset = Real(0);
    Real minimum_separation = Real(6);
    int inspiral_enabled = 1;
    int orbit_enabled = 1;
    Real metric_derivative_step = sizeof(Real) <= sizeof(float) ?
        Real(2e-3) : Real(2e-5);
    Real capture_factor = Real(1.02);

    // Eggleton Roche-lobe radius multiplied by this factor.  0.8 gives
    // r_out approximately 0.30 d for an equal-mass binary.
    Real tidal_fraction = Real(0.8);
    Real taper_start_fraction = Real(0.85);
    Real density_scale1 = Real(1);
    Real density_scale2 = Real(1);
    Real temperature_scale1 = Real(1);
    Real temperature_scale2 = Real(1);
    int field_polarity1 = 1;
    int field_polarity2 = 1;

    KPOLARIS_INLINE Real mass1() const {
        return Real(1) / (Real(1) + max_val(mass_ratio, tiny_positive<Real>()));
    }

    KPOLARIS_INLINE Real mass2() const {
        return Real(1) - mass1();
    }

    KPOLARIS_INLINE SuperposedKerrSchildMetric<Real> make_metric() const {
        SuperposedKerrSchildMetric<Real> metric;
        if (trajectory.is_tabulated()) {
            metric.orbit = trajectory;
        } else {
            metric.orbit.total_mass = Real(1);
            metric.orbit.mass1 = mass1();
            metric.orbit.mass2 = mass2();
            metric.orbit.reference_separation = reference_separation;
            metric.orbit.reference_phase = reference_phase;
            metric.orbit.reference_time = reference_time;
            metric.orbit.minimum_separation = minimum_separation;
            metric.orbit.inspiral_enabled = inspiral_enabled;
            metric.orbit.orbit_enabled = orbit_enabled;
            metric.orbit.kerr_a1 =
                Vec3<Real>(Real(0), Real(0), chi1 * metric.orbit.mass1);
            metric.orbit.kerr_a2 =
                Vec3<Real>(Real(0), Real(0), chi2 * metric.orbit.mass2);
        }
        // Retained for the legacy provider.  Tabulated trajectories supply
        // all three components of a=S/M at every event.
        metric.spin1 = Vec3<Real>(Real(0), Real(0), chi1 * metric.orbit.mass1);
        metric.spin2 = Vec3<Real>(Real(0), Real(0), chi2 * metric.orbit.mass2);
        metric.time_origin = observation_time + trajectory_time_offset;
        metric.derivative_step = metric_derivative_step;
        return metric;
    }

    // A proper rotation from a local Kerr frame, whose +z axis is the spin
    // direction, to the instantaneous Cartesian rest frame of a hole.  The
    // deterministic x-axis convention makes +z reproduce the historical
    // implementation exactly and keeps a precessing trajectory continuous
    // except at the unavoidable chart pole of this auxiliary disk basis.
    KPOLARIS_INLINE void spin_frame_basis(const Vec3<Real>& kerr_a,
                                          Vec3<Real>& ex,
                                          Vec3<Real>& ey,
                                          Vec3<Real>& ez) const {
        const Real amag = kpolaris::norm(kerr_a);
        if (!(amag > tiny_positive<Real>())) {
            ex = Vec3<Real>(Real(1), Real(0), Real(0));
            ey = Vec3<Real>(Real(0), Real(1), Real(0));
            ez = Vec3<Real>(Real(0), Real(0), Real(1));
            return;
        }
        ez = kerr_a * (Real(1) / amag);
        Vec3<Real> reference(Real(1), Real(0), Real(0));
        if (abs_val(ez.x) > Real(0.9)) {
            reference = Vec3<Real>(Real(0), Real(1), Real(0));
        }
        ex = reference - ez * kpolaris::dot(reference, ez);
        ex = ex * (Real(1) / max_val(kpolaris::norm(ex),
                                     tiny_positive<Real>()));
        ey = cross(ez, ex);
    }

    KPOLARIS_INLINE Vec4<Real> rest_point_to_spin_frame(
        const Vec4<Real>& rest,
        const Vec3<Real>& kerr_a) const {
        Vec3<Real> ex, ey, ez;
        spin_frame_basis(kerr_a, ex, ey, ez);
        const Vec3<Real> spatial(rest[1], rest[2], rest[3]);
        return Vec4<Real>(rest[0], kpolaris::dot(spatial, ex),
                          kpolaris::dot(spatial, ey),
                          kpolaris::dot(spatial, ez));
    }

    KPOLARIS_INLINE Vec4<Real> spin_vector_to_rest_frame(
        const Vec4<Real>& local,
        const Vec3<Real>& kerr_a) const {
        Vec3<Real> ex, ey, ez;
        spin_frame_basis(kerr_a, ex, ey, ez);
        const Vec3<Real> spatial = ex * local[1] + ey * local[2] +
                                   ez * local[3];
        return Vec4<Real>(local[0], spatial.x, spatial.y, spatial.z);
    }

    KPOLARIS_INLINE int finite_scalar(Real value) const {
        return value == value && abs_val(value) < large_positive<Real>();
    }

    // The covector q=dT selects the instantaneous rest-time slice of a hole.
    // Its future unit normal is -g^{-1}q/sqrt(-q.g^{-1}.q).  For the supported
    // Lorentzian SKS domain the q=constant hypersurface is spacelike: every
    // tangent X has eta(X,X)>0 and each positive Kerr-Schild rank-one term can
    // only increase g(X,X).
    KPOLARIS_INLINE int slice_unit_normal_from_raised(
        const Vec4<Real>& q,
        const Vec4<Real>& raised,
        Vec4<Real>& normal,
        Real* timelike_margin = nullptr) const {
        Real q2 = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            q2 += q[mu] * raised[mu];
        }
        if (timelike_margin) *timelike_margin = -q2;
        if (!(q2 < Real(0)) || !finite_scalar(q2)) return 0;
        const Real inv_norm = Real(1) / Kokkos::sqrt(-q2);
        for (int mu = 0; mu < ndim; ++mu) {
            normal[mu] = -raised[mu] * inv_norm;
            if (!finite_scalar(normal[mu])) return 0;
        }
        return normal[0] > Real(0);
    }

    KPOLARIS_INLINE int slice_unit_normal(
        const Real gcon[ndim][ndim],
        const Vec4<Real>& q,
        Vec4<Real>& normal,
        Real* timelike_margin = nullptr) const {
        Vec4<Real> raised;
        for (int mu = 0; mu < ndim; ++mu) {
            raised[mu] = Real(0);
            for (int nu = 0; nu < ndim; ++nu) {
                raised[mu] += gcon[mu][nu] * q[nu];
            }
        }
        return slice_unit_normal_from_raised(
            q, raised, normal, timelike_margin);
    }

    // If {S_i} is orthonormal for h0 and h=h0+A L.L, the induced spatial
    // Gram matrix is I+w.w^T with w_i=sqrt(A)L(S_i).  Its symmetric inverse
    // square root is analytic, avoiding an axis-ordered Gram-Schmidt and any
    // velocity cap or pivot floor.
    KPOLARIS_INLINE int rank_one_coordinate_spatial_frame(
        Real amp,
        const Real lcov[ndim],
        Vec4<Real> axes[3]) const {
        if (!(amp >= Real(0)) || !finite_scalar(amp)) return 0;
        const Real root_amp = Kokkos::sqrt(amp);
        Real w[3] = {Real(0), Real(0), Real(0)};
        Real w2 = Real(0);
        for (int i = 0; i < 3; ++i) {
            w[i] = root_amp * lcov[i + 1];
            w2 += w[i] * w[i];
        }
        if (!(w2 >= Real(0)) || !finite_scalar(w2)) return 0;
        const Real radial = Kokkos::sqrt(Real(1) + w2);
        const Real coefficient = -Real(1) /
            (radial * (Real(1) + radial));
        for (int i = 0; i < 3; ++i) {
            axes[i][0] = Real(0);
            for (int j = 0; j < 3; ++j) {
                axes[i][j + 1] = (i == j ? Real(1) : Real(0)) +
                    coefficient * w[i] * w[j];
                if (!finite_scalar(axes[i][j + 1])) return 0;
            }
        }
        return 1;
    }

    // Map the isolated-hole plasma state into the complete SKS metric through
    // paired orthonormal frames on the same hole-rest-time slice.  Proper
    // velocity and magnetic tetrad components are preserved.  The construction
    // is an exact identity in the isolated-hole limit and is future timelike by
    // construction in the admitted binary domain.
    template<class Metric>
    KPOLARIS_INLINE int map_plasma_state_to_full_metric(
        int hole,
        const Metric& metric,
        const typename Metric::Work& work,
        const typename KerrSchildInMetric<Real>::Work& local_work,
        const Vec4<Real>& urest,
        const Vec4<Real>& brest,
        const Real local_gcov[ndim][ndim],
        const Real full_gcov[ndim][ndim],
        Vec4<Real>& ucon,
        Vec4<Real>& bcon,
        Real* full_slice_margin = nullptr) const {
        Vec4<Real> local_axes[3];
        if (!rank_one_coordinate_spatial_frame(
                local_work.amp, local_work.l, local_axes)) return 0;

        const Real local_normal_scale = Kokkos::sqrt(Real(1) + local_work.amp);
        if (!(local_normal_scale > Real(0)) ||
            !finite_scalar(local_normal_scale)) return 0;
        Vec4<Real> local_normal;
        local_normal[0] = local_normal_scale;
        for (int i = 0; i < 3; ++i) {
            local_normal[i + 1] =
                -local_work.amp * local_work.l[i + 1] / local_normal_scale;
        }

        const typename Metric::HoleKinematics& hk = work.holes[hole].hole;
        const Vec4<Real> qlab(
            hk.gamma,
            -hk.gamma * hk.velocity.x,
            -hk.gamma * hk.velocity.y,
            -hk.gamma * hk.velocity.z);
        Vec4<Real> raised_qlab;
        metric.raise_covector_from_work(work, qlab, raised_qlab);
        Vec4<Real> full_normal;
        if (!slice_unit_normal_from_raised(
                qlab, raised_qlab, full_normal,
                full_slice_margin)) return 0;

        Vec4<Real> boosted_local_axes[3];
        for (int i = 0; i < 3; ++i) {
            // local_axes live in the z-aligned isolated-Kerr chart.  Rotate
            // them into the physical spin direction before applying the
            // instantaneous Lorentz boost to the barycentric frame.
            const Vec4<Real> rest_axis = spin_vector_to_rest_frame(
                local_axes[i], hk.spin);
            boosted_local_axes[i] =
                metric.rest_to_lab_vector(hk, rest_axis);
        }
        const int companion = 1 - hole;
        const Real companion_amp = work.holes[companion].amp;
        if (!(companion_amp >= Real(0)) ||
            !finite_scalar(companion_amp)) return 0;
        const Real companion_root_amp = Kokkos::sqrt(companion_amp);
        Real companion_w[3] = {Real(0), Real(0), Real(0)};
        Real companion_w2 = Real(0);
        for (int i = 0; i < 3; ++i) {
            Real contraction = Real(0);
            for (int mu = 0; mu < ndim; ++mu) {
                contraction += work.holes[companion].l_global[mu] *
                               boosted_local_axes[i][mu];
            }
            companion_w[i] = companion_root_amp * contraction;
            companion_w2 += companion_w[i] * companion_w[i];
        }
        if (!(companion_w2 >= Real(0)) ||
            !finite_scalar(companion_w2)) return 0;
        const Real companion_radial =
            Kokkos::sqrt(Real(1) + companion_w2);
        const Real companion_coefficient = -Real(1) /
            (companion_radial * (Real(1) + companion_radial));

        Real proper_velocity[3] = {Real(0), Real(0), Real(0)};
        Real proper_velocity2 = Real(0);
        Real magnetic_spatial[3] = {Real(0), Real(0), Real(0)};
        Real magnetic_spatial2 = Real(0);
        Real velocity_dot_magnetic = Real(0);
        for (int i = 0; i < 3; ++i) {
            proper_velocity[i] =
                dot_with_metric(local_gcov, urest, local_axes[i]);
            magnetic_spatial[i] =
                dot_with_metric(local_gcov, brest, local_axes[i]);
            proper_velocity2 += proper_velocity[i] * proper_velocity[i];
            magnetic_spatial2 += magnetic_spatial[i] * magnetic_spatial[i];
            velocity_dot_magnetic +=
                proper_velocity[i] * magnetic_spatial[i];
        }
        if (!(proper_velocity2 >= Real(0)) ||
            !finite_scalar(proper_velocity2)) return 0;
        const Real magnetic_time =
            -dot_with_metric(local_gcov, local_normal, brest);
        const Real source_gamma =
            -dot_with_metric(local_gcov, local_normal, urest);
        const Real norm_tolerance = sizeof(Real) <= sizeof(float) ?
            Real(5e-4) : Real(1e-9);
        if (!(source_gamma > Real(0)) || !finite_scalar(source_gamma) ||
            !finite_scalar(magnetic_time) ||
            abs_val(-source_gamma * source_gamma + proper_velocity2 + Real(1)) >
                norm_tolerance ||
            abs_val(-magnetic_time * magnetic_time + magnetic_spatial2 - Real(1)) >
                norm_tolerance ||
            abs_val(-source_gamma * magnetic_time + velocity_dot_magnetic) >
                norm_tolerance) return 0;

        // Apply the companion rank-one inverse square root directly to the
        // tetrad coefficients.  This is algebraically identical to forming
        // three full axes, while shortening GPU live ranges and stack use.
        Real wdot_velocity = Real(0), wdot_magnetic = Real(0);
        for (int i = 0; i < 3; ++i) {
            wdot_velocity += companion_w[i] * proper_velocity[i];
            wdot_magnetic += companion_w[i] * magnetic_spatial[i];
        }
        for (int i = 0; i < 3; ++i) {
            proper_velocity[i] += companion_coefficient * companion_w[i] *
                                  wdot_velocity;
            magnetic_spatial[i] += companion_coefficient * companion_w[i] *
                                   wdot_magnetic;
        }
        const Real fluid_gamma = Kokkos::sqrt(Real(1) + proper_velocity2);
        ucon = full_normal * fluid_gamma;
        Vec4<Real> btrial = full_normal * magnetic_time;
        for (int i = 0; i < 3; ++i) {
            ucon = ucon + boosted_local_axes[i] * proper_velocity[i];
            btrial = btrial + boosted_local_axes[i] * magnetic_spatial[i];
        }

        const Real unorm = dot_with_metric(full_gcov, ucon, ucon);
        if (!(unorm < Real(0)) || !finite_scalar(unorm) ||
            abs_val(unorm + Real(1)) > norm_tolerance ||
            !(ucon[0] > Real(0))) return 0;
        ucon = ucon * (Real(1) / Kokkos::sqrt(-unorm));

        // This projection changes only roundoff in a valid paired-frame map.
        const Real udotb = dot_with_metric(full_gcov, ucon, btrial);
        bcon = btrial + ucon * udotb;
        const Real bsq = dot_with_metric(full_gcov, bcon, bcon);
        if (!(bsq > Real(0)) || !finite_scalar(bsq)) return 0;
        bcon = bcon * (Real(1) / Kokkos::sqrt(bsq));
        return 1;
    }

    Real validate_metric_domain(Real outer_domain_radius) {
        const bool tabulated = trajectory.is_tabulated();
        const bool invalid_legacy = !tabulated &&
            (!std::isfinite(double(mass_ratio)) || !(mass_ratio > Real(0)) ||
             !std::isfinite(double(chi1)) || !std::isfinite(double(chi2)) ||
             std::abs(double(chi1)) > 1.0 ||
             std::abs(double(chi2)) > 1.0 ||
             !std::isfinite(double(reference_separation)) ||
             !(reference_separation > Real(0)));
        if (invalid_legacy ||
            !std::isfinite(double(metric_derivative_step)) ||
            !(metric_derivative_step > Real(0)) ||
            !std::isfinite(double(capture_factor)) ||
            !(capture_factor > Real(1)) ||
            !(capture_factor <= Real(1.05)) ||
            !std::isfinite(double(outer_domain_radius))) {
            throw std::runtime_error(
                "invalid binary metric parameters for the host admissibility audit");
        }
        const SuperposedKerrSchildMetric<Real> metric = make_metric();
        const Vec4<Real> camera_event(Real(0), Real(0), Real(0), Real(0));
        const typename LeadingOrderBinaryOrbit<Real>::State orbit_state =
            metric.orbit.state(metric.absolute_time(camera_event));
        if (!orbit_state.valid || !orbit_state.inspiral_valid) {
            throw std::runtime_error(
                "binary observation event is outside the available inspiral trajectory domain");
        }
        if (orbit_state.merger_weight > Real(0)) {
            throw std::runtime_error(
                "the analytic double-mini-RIAF model is defined only before the merger transition (W=0); the SKS metric itself remains available through merger and remnant");
        }

        // Include an explicit mini-disk inner-edge radius in addition to the
        // global logarithmic metric scan.  The strongest lensing and the old
        // direct-boost failures live in this thin r_+ + horizon_buffer layer.
        const int radial_samples = 66;
        const int theta_samples = 33;
        const int phi_samples = 64;
        const Real pi = Real(3.141592653589793238462643383279502884);
        Real minimum = Real(1);
        Real minimum_fluid_margin = large_positive<Real>();
        int fluid_frame_samples = 0;
        int fluid_frame_samples_by_hole[2] = {0, 0};
        for (int hole = 0; hole < 2; ++hole) {
            const typename SuperposedKerrSchildMetric<Real>::HoleKinematics hk =
                metric.hole_kinematics(hole, camera_event);
            const Real ma = hk.mass;
            const Real mb = orbit_state.mass(1 - hole);
            const Real chi = kpolaris::norm(hk.spin) /
                max_val(ma, tiny_positive<Real>());
            if (!(ma > Real(0)) || !(mb > Real(0)) || !(chi < Real(1))) {
                throw std::runtime_error(
                    "each emitting mini-RIAF requires positive component masses and a subextremal instantaneous Kerr spin");
            }
            const Real horizon = max_val(
                metric.hole_horizon_radius(hole, camera_event),
                                         tiny_positive<Real>());
            const Real horizon_hat = Real(1) +
                Real(std::sqrt(std::max(Real(0), Real(1) - chi * chi)));
            const Real rout_hat = outer_radius_hat(
                ma, mb, orbit_state.separation);
            const Real coordinate_scale = max_val(
                Real(1), kpolaris::norm(hk.position) + horizon);
            const Real fd_step = metric.derivative_step * coordinate_scale;
            if (fd_step > Real(0.02) * horizon) {
                throw std::runtime_error(
                    "binary_metric_derivative_step under-resolves an individual horizon; reduce it or use a less extreme mass ratio");
            }

            const Real spin_length = kpolaris::norm(hk.spin);
            const Real r_min = capture_factor * horizon * Real(1.0001);
            const Real requested_extent = outer_domain_radius > Real(0) ?
                outer_domain_radius : Real(2) * orbit_state.separation;
            const Real r_max = max_val(
                r_min * Real(1.001),
                max_val(Real(2) * orbit_state.separation,
                        requested_extent + kpolaris::norm(hk.position) + horizon));
            const Real log_span = Real(std::log(double(r_max / r_min)));
            const Real v2 = kpolaris::dot(hk.velocity, hk.velocity);

            // Audit the ungated radial-characteristic value B itself between
            // R0 and the signed exterior guard.  G guarantees a usable signed
            // level set, but must never manufacture a guard-radius boundary
            // when B has no outward-connected root.  Along every sampled angle
            // B may cross from negative to positive at most once, may not turn
            // negative again, and must be safely positive at Rguard.
            const int capture_radial_samples = 33;
            const Real capture_guard_ratio =
                metric.capture_surrogate_guard_ratio(capture_factor);
            const Real capture_r0 = capture_factor * horizon;
            const Real capture_rguard = capture_guard_ratio * horizon;
            const Real capture_sign_tolerance =
                sizeof(Real) <= sizeof(float) ? Real(5e-4) : Real(1e-7);
            const Real capture_guard_margin =
                sizeof(Real) <= sizeof(float) ? Real(5e-3) : Real(1e-4);
            for (int it = 0; it < theta_samples; ++it) {
                const Real theta = pi * Real(it) / Real(theta_samples - 1);
                const Real sint = Real(std::sin(double(theta)));
                const Real cost = Real(std::cos(double(theta)));
                for (int ip = 0; ip < phi_samples; ++ip) {
                    const Real phi = Real(2) * pi * Real(ip) /
                                     Real(phi_samples);
                    const Real cosphi = Real(std::cos(double(phi)));
                    const Real sinphi = Real(std::sin(double(phi)));
                    bool seen_negative = false;
                    bool seen_positive = false;
                    int outward_roots = 0;
                    Real guard_value = -large_positive<Real>();
                    for (int ic = 0; ic < capture_radial_samples; ++ic) {
                        const Real fc = Real(ic) /
                            Real(capture_radial_samples - 1);
                        const Real r = capture_r0 +
                            fc * (capture_rguard - capture_r0);
                        const Real rho = Real(std::sqrt(double(
                            r * r + spin_length * spin_length)));
                        const Vec4<Real> Xlocal(Real(0),
                            rho * sint * cosphi,
                            rho * sint * sinphi,
                            r * cost);
                        const Vec4<Real> Xrest4 =
                            spin_vector_to_rest_frame(Xlocal, hk.spin);
                        const Vec3<Real> X(
                            Xrest4[1], Xrest4[2], Xrest4[3]);
                        const Real vdotx = kpolaris::dot(hk.velocity, X);
                        const Real inverse_factor =
                            v2 > tiny_positive<Real>() ?
                            (Real(1) / hk.gamma - Real(1)) * vdotx / v2 :
                            Real(0);
                        const Vec3<Real> dx =
                            X + hk.velocity * inverse_factor;
                        const Vec4<Real> x(
                            Real(0), hk.position.x + dx.x,
                            hk.position.y + dx.y, hk.position.z + dx.z);
                        const typename SuperposedKerrSchildMetric<Real>::Work
                            work = metric.build_work(x);
                        Real ginv[ndim][ndim];
                        metric.gcon_matrix_from_work(work, ginv);
                        const Real characteristic_value =
                            metric.hole_capture_surrogate_value(
                                hole, x, work.holes[hole], ginv,
                                capture_factor);
                        if (!std::isfinite(double(characteristic_value))) {
                            throw std::runtime_error(
                                "non-finite full-metric radial-characteristic capture surrogate in the R0-to-Rguard audit");
                        }
                        guard_value = characteristic_value;
                        if (characteristic_value <
                            -capture_sign_tolerance) {
                            if (seen_positive) {
                                throw std::runtime_error(
                                    "full-metric radial-characteristic capture surrogate has a positive-to-negative return before Rguard");
                            }
                            seen_negative = true;
                        } else if (characteristic_value >
                                   capture_sign_tolerance) {
                            if (!seen_positive && seen_negative) {
                                ++outward_roots;
                            }
                            seen_positive = true;
                        }
                    }
                    if (outward_roots > 1 ||
                        !(guard_value > capture_guard_margin)) {
                        throw std::runtime_error(
                            "full-metric radial-characteristic capture surrogate lacks a unique outward-connected root with positive margin before Rguard");
                    }
                }
            }

            for (int ir = 0; ir < radial_samples; ++ir) {
                Real r = r_min;
                if (ir == 1) {
                    const Real emission_inner = ma *
                        (horizon_hat + disk.horizon_buffer) * Real(1.0001);
                    r = min_val(r_max, max_val(r_min, emission_inner));
                } else if (ir >= 2) {
                    const Real fr = Real(ir - 2) /
                        Real(radial_samples - 3);
                    r = r_min * Real(std::exp(double(fr * log_span)));
                }
                const Real rho = Real(std::sqrt(double(r * r +
                                                        spin_length * spin_length)));
                for (int it = 0; it < theta_samples; ++it) {
                    const Real theta = pi * Real(it) / Real(theta_samples - 1);
                    const Real sint = Real(std::sin(double(theta)));
                    const Real cost = Real(std::cos(double(theta)));
                    for (int ip = 0; ip < phi_samples; ++ip) {
                        const Real phi = Real(2) * pi * Real(ip) /
                                         Real(phi_samples);
                        const Vec4<Real> Xlocal(Real(0),
                            rho * sint * Real(std::cos(double(phi))),
                            rho * sint * Real(std::sin(double(phi))),
                            r * cost);
                        const Vec4<Real> Xrest4 =
                            spin_vector_to_rest_frame(Xlocal, hk.spin);
                        const Vec3<Real> X(
                            Xrest4[1], Xrest4[2], Xrest4[3]);
                        const Real vdotx = kpolaris::dot(hk.velocity, X);
                        const Real inverse_factor = v2 > tiny_positive<Real>() ?
                            (Real(1) / hk.gamma - Real(1)) * vdotx / v2 :
                            Real(0);
                        const Vec3<Real> dx = X + hk.velocity * inverse_factor;
                        const Vec4<Real> x(Real(0),
                            hk.position.x + dx.x, hk.position.y + dx.y,
                            hk.position.z + dx.z);
                        if (metric.inner_boundary_value(x, capture_factor) <=
                            Real(0)) {
                            continue;
                        }
                        const Real denominator = metric.inverse_denominator(x);
                        if (!std::isfinite(double(denominator))) {
                            throw std::runtime_error(
                                "non-finite Superposed Kerr-Schild inverse denominator outside the capture surfaces");
                        }
                        minimum = min_val(minimum, denominator);

                        const Real rhat = r / max_val(ma, tiny_positive<Real>());
                        if (rhat < disk.r_min ||
                            rhat <= horizon_hat + disk.horizon_buffer ||
                            !(outer_taper(rhat, rout_hat) > Real(0))) {
                            continue;
                        }

                        // Audit exactly the paired-frame construction used by
                        // the device radiation model throughout the sampled
                        // emitting domain.  A failure is fatal here rather
                        // than becoming a dark arc inside a render kernel.
                        const typename SuperposedKerrSchildMetric<Real>::Work
                            work = metric.build_work(x);
                        Vec4<Real> xhat = rest_point_to_spin_frame(
                            work.holes[hole].rest_x, hk.spin);
                        xhat[1] /= ma; xhat[2] /= ma; xhat[3] /= ma;
                        const KerrSchildInMetric<Real> local_metric(
                            Real(1), chi);
                        const typename KerrSchildInMetric<Real>::Work local_work =
                            local_metric.build_work(xhat);
                        Real local_r = Real(0), local_th = Real(0);
                        Real local_cp = Real(1), local_sp = Real(0);
                        disk.bl_coordinates(xhat, chi, local_r, local_th,
                                            local_cp, local_sp);
                        Real local_gcov[ndim][ndim];
                        local_metric.gcov_matrix(xhat, local_gcov);
                        const Vec4<Real> urest =
                            disk.fluid_four_velocity_from_bl_coords(
                                local_metric, xhat, local_r, local_th,
                                local_cp, local_sp, local_gcov);
                        const Vec4<Real> brest =
                            disk.magnetic_unit_four_vector_with_gcov(
                                local_metric, xhat, urest, local_gcov);
                        Real full_gcov[ndim][ndim];
                        metric.gcov_matrix_from_work(work, full_gcov);
                        Vec4<Real> ucon, bcon;
                        Real slice_margin = Real(-1);
                        if (!map_plasma_state_to_full_metric(
                                hole, metric, work, local_work, urest, brest,
                                local_gcov, full_gcov,
                                ucon, bcon, &slice_margin)) {
                            throw std::runtime_error(
                                "Binary RIAF paired plasma frame is invalid in the sampled emitting domain");
                        }
                        const Real unorm = dot_with_metric(
                            full_gcov, ucon, ucon);
                        const Real bnorm = dot_with_metric(
                            full_gcov, bcon, bcon);
                        const Real udotb = dot_with_metric(
                            full_gcov, ucon, bcon);
                        const Real audit_tolerance =
                            sizeof(Real) <= sizeof(float) ?
                            Real(2e-3) : Real(2e-9);
                        if (!std::isfinite(double(slice_margin)) ||
                            !(slice_margin > Real(0)) ||
                            !std::isfinite(double(unorm)) ||
                            !std::isfinite(double(bnorm)) ||
                            !std::isfinite(double(udotb)) ||
                            std::abs(double(unorm + Real(1))) >
                                double(audit_tolerance) ||
                            std::abs(double(bnorm - Real(1))) >
                                double(audit_tolerance) ||
                            std::abs(double(udotb)) >
                                double(audit_tolerance) ||
                            !(ucon[0] > Real(0))) {
                            throw std::runtime_error(
                                "Binary RIAF paired plasma frame fails its orthonormal invariant audit");
                        }
                        minimum_fluid_margin = min_val(
                            minimum_fluid_margin, slice_margin);
                        ++fluid_frame_samples;
                        ++fluid_frame_samples_by_hole[hole];
                    }
                }
            }
        }

        sampled_min_inverse_denominator = minimum;
        if (fluid_frame_samples == 0 ||
            fluid_frame_samples_by_hole[0] == 0 ||
            fluid_frame_samples_by_hole[1] == 0) {
            throw std::runtime_error(
                "Binary RIAF admissibility audit found no emitting-domain samples for one or both mini-disks");
        }
        sampled_min_fluid_slice_timelike_margin = minimum_fluid_margin;
        if (minimum < Real(0.1)) {
            throw std::runtime_error(
                "Superposed Kerr-Schild metric fails the D_min >= 0.1 Lorentz-signature margin outside the capture surfaces (sampled D_min=" +
                std::to_string(double(minimum)) +
                "); increase separation, reduce spin/extreme mass ratio, or use a different spacetime model");
        }
        const Real slice_margin_tolerance =
            sizeof(Real) <= sizeof(float) ? Real(1e-5) : Real(1e-10);
        if (!(sampled_min_fluid_slice_timelike_margin >
              slice_margin_tolerance)) {
            throw std::runtime_error(
                "Binary RIAF hole-rest-time slice approaches null in the sampled emitting domain; use a less extreme binary configuration");
        }
        return minimum;
    }

    KPOLARIS_INLINE Real dlambda_scale() const {
        // disk.mbh_solar denotes the total binary mass, never an individual
        // component mass.
        return disk.length_unit_cgs() / max_val(freq_cgs, Real(1));
    }

    KPOLARIS_INLINE Real roche_lobe_fraction(Real ma, Real mb) const {
        const Real q = max_val(ma / max_val(mb, tiny_positive<Real>()),
                               tiny_positive<Real>());
        const Real q13 = Kokkos::cbrt(q);
        const Real q23 = q13 * q13;
        return Real(0.49) * q23 /
            (Real(0.6) * q23 + Kokkos::log(Real(1) + q13));
    }

    KPOLARIS_INLINE Real outer_radius_hat(Real ma, Real mb,
                                         Real separation) const {
        if (!(ma > Real(0)) || !(mb > Real(0)) ||
            !(separation > Real(0))) return Real(0);
        const Real roche = separation * roche_lobe_fraction(ma, mb);
        const Real tidal = tidal_fraction * roche / ma;
        return min_val(disk.r_max, tidal);
    }

    // Compatibility helper for the leading-order, constant-mass model.
    KPOLARIS_INLINE Real outer_radius_hat(int hole,
                                         Real separation) const {
        return outer_radius_hat(hole == 0 ? mass1() : mass2(),
                                hole == 0 ? mass2() : mass1(),
                                separation);
    }

    KPOLARIS_INLINE Real outer_taper(Real rhat, Real rout_hat) const {
        if (!(rout_hat > Real(0)) || rhat >= rout_hat) return Real(0);
        const Real start = clamp(taper_start_fraction, Real(0), Real(0.999)) *
                           rout_hat;
        if (rhat <= start) return Real(1);
        const Real s = clamp((rhat - start) /
                             max_val(rout_hat - start, tiny_positive<Real>()),
                             Real(0), Real(1));
        return Real(1) - Real(3) * s * s + Real(2) * s * s * s;
    }

    KPOLARIS_INLINE void add_coefficients(TransferCoeffs<Real>& sum,
                                         const TransferCoeffs<Real>& a) const {
        sum.jI += a.jI; sum.jQ += a.jQ; sum.jU += a.jU; sum.jV += a.jV;
        sum.aI += a.aI; sum.aQ += a.aQ; sum.aU += a.aU; sum.aV += a.aV;
        sum.rQ += a.rQ; sum.rU += a.rU; sum.rV += a.rV;
    }

    KPOLARIS_INLINE Vec4<Real> lower_with_metric(
        const Real gcov[ndim][ndim], const Vec4<Real>& v) const {
        Vec4<Real> out;
        for (int mu = 0; mu < ndim; ++mu) {
            out[mu] = Real(0);
            for (int nu = 0; nu < ndim; ++nu) out[mu] += gcov[mu][nu] * v[nu];
        }
        return out;
    }

    KPOLARIS_INLINE Real dot_with_metric(const Real gcov[ndim][ndim],
                                        const Vec4<Real>& a,
                                        const Vec4<Real>& b) const {
        Real out = Real(0);
        for (int mu = 0; mu < ndim; ++mu) {
            for (int nu = 0; nu < ndim; ++nu) out += gcov[mu][nu] * a[mu] * b[nu];
        }
        return out;
    }

    KPOLARIS_INLINE TransferCoeffs<Real> plasma_coefficients(
        const Real gcov[ndim][ndim],
        const TransportState<Real>& state,
        const Vec4<Real>& ucon,
        const Vec4<Real>& bcon_unit,
        Real ne_cgs,
        Real thetae,
        Real b_cgs) const {
        TransferCoeffs<Real> coeffs;
        const Vec4<Real> ucov = lower_with_metric(gcov, ucon);
        Real uk = Real(0);
        for (int mu = 0; mu < ndim; ++mu) uk += ucov[mu] * state.k[mu];
        const Real nu_fluid_scale = -uk;
        // Pass B is future directed.  A non-positive value indicates an
        // invalid local fluid state or photon orientation and must never be
        // painted as a dark arc.  A positive value below the configured
        // frequency floor remains an explicit numerical cutoff.
        if (!finite_scalar(nu_fluid_scale) ||
            !(nu_fluid_scale > Real(0))) {
            Kokkos::abort("invalid BinaryRIAF fluid-frame photon frequency");
            return coeffs;
        }
        if (!(nu_fluid_scale > disk.min_frequency_scale)) return coeffs;

        const Vec4<Real> bcov = lower_with_metric(gcov, bcon_unit);
        Real kdotb = Real(0);
        for (int mu = 0; mu < ndim; ++mu) kdotb += state.k[mu] * bcov[mu];
        const Real cos_theta = clamp(kdotb / nu_fluid_scale, Real(-1), Real(1));
        const Real sin_theta = max_val(
            Kokkos::sqrt(max_val(Real(0), Real(1) - cos_theta * cos_theta)),
            disk.min_sin_theta);
        const Real theta = Kokkos::acos(cos_theta);
        const Real nu = max_val(freq_cgs * nu_fluid_scale, Real(1));

        ThermalSynchrotronParams<Real> thermal_params;
#if KPOLARIS_RIAF_COMPILED_EMISSION_TYPE > 0
        thermal_params.fit = KPOLARIS_RIAF_COMPILED_EMISSION_TYPE;
#else
        thermal_params.fit = disk.emission_type;
#endif
        thermal_params.max_pol_frac = disk.max_pol_frac;
        thermal_params.min_sin_theta = disk.min_sin_theta;
        thermal_params.emission_scale = disk.emission_scale;
        thermal_params.absorption_scale = disk.absorption_scale;
        thermal_params.faraday_scale = disk.faraday_scale;

        const Real pi = Real(3.141592653589793238462643383279502884);
        if (theta <= Real(0) || theta >= pi) {
            coeffs.rV = disk.faraday_scale *
                symphony_rho_V(nu, ne_cgs, thetae, b_cgs, theta,
                               disk.min_sin_theta) * nu;
            return coeffs;
        }

        LocalThermalSynchrotronState<Real> thermal_state;
        thermal_state.nu = nu;
        thermal_state.ne = ne_cgs;
        thermal_state.thetae = thetae;
        thermal_state.b_cgs = b_cgs;
        thermal_state.theta = theta;
        thermal_state.sin_theta = sin_theta;
        thermal_state.cos_theta = cos_theta;

        TransferCoeffs<Real> magnetic;
#if KPOLARIS_RIAF_COMPILED_EMISSION_TYPE == KPOLARIS_THERMAL_SYNCHROTRON_DEXTER
        magnetic = thermal_synchrotron_magnetic_basis_coefficients_fit<
            ThermalSynchrotronDexter>(thermal_state, thermal_params);
#elif KPOLARIS_RIAF_COMPILED_EMISSION_TYPE == KPOLARIS_THERMAL_SYNCHROTRON_PANDYA
        magnetic = thermal_synchrotron_magnetic_basis_coefficients_fit<
            ThermalSynchrotronPandya>(thermal_state, thermal_params);
#else
        if (disk.emission_type == SynchrotronKappa ||
            disk.emission_type == SynchrotronPowerLaw) {
            NonthermalSynchrotronParams<Real> p;
            p.distribution = disk.emission_type;
            p.max_pol_frac_emission = disk.max_pol_frac;
            p.max_pol_frac_absorption = disk.max_pol_frac;
            p.min_sin_theta = disk.min_sin_theta;
            p.emission_scale = disk.emission_scale;
            p.absorption_scale = disk.absorption_scale;
            p.faraday_scale = disk.faraday_scale;
            p.kappa = disk.nonthermal_kappa;
            p.kappa_interp_begin = disk.variable_kappa_interp_start;
            p.kappa_interp_end = disk.variable_kappa_max;
            p.power_law_p = disk.powerlaw_p;
            p.power_law_eta = disk.powerlaw_eta;
            p.power_law_gamma_min = disk.powerlaw_gamma_min;
            p.power_law_gamma_max = disk.powerlaw_gamma_max;
            p.power_law_gamma_cutoff = disk.powerlaw_gamma_cutoff;
            LocalNonthermalSynchrotronState<Real> s;
            s.nu = nu; s.ne = ne_cgs; s.thetae = thetae; s.b_cgs = b_cgs;
            s.sin_theta = sin_theta; s.cos_theta = cos_theta;
            magnetic = nonthermal_synchrotron_magnetic_basis_coefficients(s, p);
        } else {
            magnetic = thermal_synchrotron_magnetic_basis_coefficients(
                thermal_state, thermal_params);
        }
#endif

        const Real b1 = disk.screen_inner_product_with_gcov(
            gcov, ucon, state.k, state.e1, bcon_unit);
        const Real b2 = disk.screen_inner_product_with_gcov(
            gcov, ucon, state.k, state.e2, bcon_unit);
        const Real bproj2 = b1 * b1 + b2 * b2;
        coeffs.jI = magnetic.jI; coeffs.jV = magnetic.jV;
        coeffs.aI = magnetic.aI; coeffs.aV = magnetic.aV;
        coeffs.rV = magnetic.rV;
        if (bproj2 > Real(1e-30)) {
            const Real cos2chi = (b2 * b2 - b1 * b1) / bproj2;
            const Real sin2chi = -Real(2) * b1 * b2 / bproj2;
            coeffs.jQ = magnetic.jQ * cos2chi;
            coeffs.jU = magnetic.jQ * sin2chi;
            coeffs.aQ = magnetic.aQ * cos2chi;
            coeffs.aU = magnetic.aQ * sin2chi;
            coeffs.rQ = magnetic.rQ * cos2chi;
            coeffs.rU = magnetic.rQ * sin2chi;
        }
        return coeffs;
    }

    template<class Metric>
    KPOLARIS_INLINE TransferCoeffs<Real> component_coefficients_from_work(
        int hole,
        const Metric& metric,
        const TransportState<Real>& state,
        const typename Metric::Work& work) const {
        TransferCoeffs<Real> coeffs;
        const typename Metric::HoleWork& hw = work.holes[hole];
        const Real ma = hw.hole.mass;
        if (!(ma > Real(0))) return coeffs;
        const Real mb = work.trajectory.mass(1 - hole);
        const Real chi = kpolaris::norm(hw.hole.spin) / ma;
        // The analytic source model represents two distinct subextremal
        // mini-disks.  It is intentionally not extrapolated into the
        // coincident half-mass remnant representation, where |a| may exceed
        // either term's mass even though the combined Kerr hole is physical.
        if (!(mb > Real(0)) || !(chi < Real(1)) ||
            !(work.trajectory.merger_weight == Real(0))) return coeffs;
        const Real rhat = hw.r / ma;
        const Real rout_hat = outer_radius_hat(
            ma, mb, work.trajectory.separation);
        const Real taper = outer_taper(rhat, rout_hat);
        const Real horizon_hat = Real(1) +
            Kokkos::sqrt(max_val(Real(1) - chi * chi, Real(0)));
        if (!(taper > Real(0)) || rhat < disk.r_min ||
            rhat <= horizon_hat + disk.horizon_buffer) return coeffs;

        Vec4<Real> xhat = rest_point_to_spin_frame(
            hw.rest_x, hw.hole.spin);
        xhat[1] /= ma; xhat[2] /= ma; xhat[3] /= ma;
        const KerrSchildInMetric<Real> local_metric(Real(1), chi);
        const typename KerrSchildInMetric<Real>::Work local_work =
            local_metric.build_work(xhat);
        Real r = Real(0), th = Real(0), cosphi = Real(1), sinphi = Real(0);
        disk.bl_coordinates(xhat, chi, r, th, cosphi, sinphi);

        const Real density_scale = hole == 0 ? density_scale1 : density_scale2;
        const Real temperature_scale = hole == 0 ? temperature_scale1 : temperature_scale2;
        const Real ne_norm = disk.density_profile_bl(r, th) * taper * density_scale;
        if (!(ne_norm > Real(0))) return coeffs;
        const Real ne_cgs = ne_norm * disk.ne_unit;
        const Real thetae = max_val(disk.thetae_profile(r) * temperature_scale,
                                    disk.min_thetae);
        const Real b_cgs = disk.magnetic_field_cgs(ne_cgs, r);

        Real local_gcov[ndim][ndim];
        local_metric.gcov_matrix(xhat, local_gcov);
        const Vec4<Real> urest = disk.fluid_four_velocity_from_bl_coords(
            local_metric, xhat, r, th, cosphi, sinphi, local_gcov);
        Vec4<Real> brest = disk.magnetic_unit_four_vector_with_gcov(
            local_metric, xhat, urest, local_gcov);
        const int polarity = hole == 0 ? field_polarity1 : field_polarity2;
        if (polarity < 0) brest = brest * Real(-1);

        Real full_gcov[ndim][ndim];
        metric.gcov_matrix_from_work(work, full_gcov);
        Vec4<Real> ucon, bcon;
        if (!map_plasma_state_to_full_metric(
                hole, metric, work, local_work, urest, brest,
                local_gcov, full_gcov, ucon, bcon)) {
            // validate_metric_domain() audits the emitting domain before any
            // kernel launch.  Reaching this branch means that contract was
            // violated; abort instead of painting a false zero-emissivity arc.
            Kokkos::abort("invalid BinaryRIAF full-metric plasma frame");
            return coeffs;
        }
        return plasma_coefficients(full_gcov, state, ucon, bcon,
                                   ne_cgs, thetae, b_cgs);
    }

    template<class Metric>
    KPOLARIS_INLINE TransferCoeffs<Real> component_coefficients(
        int hole,
        const Metric& metric,
        const TransportState<Real>& state) const {
        const typename Metric::Work work = metric.build_work(state.x);
        return component_coefficients_from_work(hole, metric, state, work);
    }

    template<class Metric>
    KPOLARIS_INLINE TransferCoeffs<Real> coefficients(
        const Metric& metric,
        const TransportState<Real>& state,
        Real) const {
        TransferCoeffs<Real> out;
        const typename Metric::Work work = metric.build_work(state.x);
        add_coefficients(out, component_coefficients_from_work(
            0, metric, state, work));
        add_coefficients(out, component_coefficients_from_work(
            1, metric, state, work));
        return out;
    }
};

} // namespace kpolaris
