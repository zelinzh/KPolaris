#include <algorithm>
#include <cstddef>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include <Kokkos_Core.hpp>

#include "camera/camera.hpp"
#include "geodesic/metric_domain.hpp"
#include "geometry/kerr_schild_cartesian.hpp"
#include "geometry/superposed_kerr_schild.hpp"
#include "model/binary_riaf.hpp"

namespace {
using Real = kpolaris::DefaultReal;

using BinaryMetric = kpolaris::SuperposedKerrSchildMetric<Real>;
using SharedRealView = Kokkos::View<Real*, Kokkos::SharedSpace>;

struct OwnedTabulatedMetric {
    SharedRealView times;
    SharedRealView values;
    BinaryMetric metric;
};

void require_close(const char* name, Real got, Real expected, Real tol) {
    const Real scale = std::max<Real>(Real(1), std::abs(expected));
    if (!std::isfinite(got) || std::abs(got - expected) > tol * scale) {
        throw std::runtime_error(std::string(name) + " mismatch");
    }
}

kpolaris::Vec3<Real> normalized(const kpolaris::Vec3<Real>& v) {
    const Real length = kpolaris::norm(v);
    if (!(length > Real(0))) {
        throw std::runtime_error("cannot normalize a zero test vector");
    }
    return v * (Real(1) / length);
}

void spin_aligned_basis(const kpolaris::Vec3<Real>& spin,
                        kpolaris::Vec3<Real> basis[3]) {
    basis[2] = normalized(spin);
    const kpolaris::Vec3<Real> reference =
        std::abs(basis[2].z) < Real(0.9) ?
        kpolaris::Vec3<Real>(Real(0), Real(0), Real(1)) :
        kpolaris::Vec3<Real>(Real(1), Real(0), Real(0));
    basis[0] = normalized(kpolaris::cross(basis[2], reference));
    basis[1] = kpolaris::cross(basis[2], basis[0]);
}

Real component(const kpolaris::Vec3<Real>& v, int i) {
    return i == 0 ? v.x : (i == 1 ? v.y : v.z);
}

// Independent reference construction: rotate a standard +z-spin Cartesian
// Kerr-Schild metric into an arbitrary spin direction, then pull it back with
// the full inverse Lorentz Jacobian of a uniformly moving center.
void explicit_boosted_rotated_kerr_gcov(
    Real mass,
    const kpolaris::Vec3<Real>& spin,
    const kpolaris::Vec3<Real>& center_at_zero,
    const kpolaris::Vec3<Real>& velocity,
    const kpolaris::Vec4<Real>& x,
    Real g[4][4]) {
    kpolaris::Vec3<Real> basis[3];
    spin_aligned_basis(spin, basis);

    const Real v2 = kpolaris::dot(velocity, velocity);
    const Real gamma = Real(1) / std::sqrt(Real(1) - v2);
    const kpolaris::Vec3<Real> center = center_at_zero + velocity * x[0];
    const kpolaris::Vec3<Real> dx(
        x[1] - center.x, x[2] - center.y, x[3] - center.z);
    const Real vdotx = kpolaris::dot(velocity, dx);
    const Real boost_factor = v2 > Real(0) ?
        (gamma - Real(1)) * vdotx / v2 : Real(0);
    const kpolaris::Vec3<Real> rest = dx + velocity * boost_factor;
    const kpolaris::Vec4<Real> local_x(
        Real(0), kpolaris::dot(basis[0], rest),
        kpolaris::dot(basis[1], rest),
        kpolaris::dot(basis[2], rest));

    const kpolaris::KerrSchildInMetric<Real> local_metric(
        mass, kpolaris::norm(spin));
    Real local_g[4][4];
    local_metric.gcov_matrix(local_x, local_g);

    // jacobian[a][mu] = d X_local^a / d x_global^mu.
    Real jacobian[4][4] = {};
    jacobian[0][0] = gamma;
    for (int j = 0; j < 3; ++j) {
        jacobian[0][j + 1] = -gamma * component(velocity, j);
    }
    for (int a = 0; a < 3; ++a) {
        jacobian[a + 1][0] = -gamma * kpolaris::dot(basis[a], velocity);
        for (int j = 0; j < 3; ++j) {
            Real value = component(basis[a], j);
            if (v2 > Real(0)) {
                value += (gamma - Real(1)) *
                    kpolaris::dot(basis[a], velocity) *
                    component(velocity, j) / v2;
            }
            jacobian[a + 1][j + 1] = value;
        }
    }

    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            Real value = Real(0);
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    value += jacobian[a][mu] * local_g[a][b] *
                             jacobian[b][nu];
                }
            }
            g[mu][nu] = value;
        }
    }
}

void invert_reference_metric(const Real matrix[4][4], Real inverse[4][4]) {
    Real augmented[4][8];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            augmented[i][j] = matrix[i][j];
            augmented[i][j + 4] = i == j ? Real(1) : Real(0);
        }
    }
    for (int column = 0; column < 4; ++column) {
        int pivot = column;
        for (int row = column + 1; row < 4; ++row) {
            if (std::abs(augmented[row][column]) >
                std::abs(augmented[pivot][column])) {
                pivot = row;
            }
        }
        if (!(std::abs(augmented[pivot][column]) > Real(1e-14))) {
            throw std::runtime_error("singular reference metric in binary SKS test");
        }
        if (pivot != column) {
            for (int j = 0; j < 8; ++j) {
                std::swap(augmented[pivot][j], augmented[column][j]);
            }
        }
        const Real scale = Real(1) / augmented[column][column];
        for (int j = 0; j < 8; ++j) augmented[column][j] *= scale;
        for (int row = 0; row < 4; ++row) {
            if (row == column) continue;
            const Real factor = augmented[row][column];
            for (int j = 0; j < 8; ++j) {
                augmented[row][j] -= factor * augmented[column][j];
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) inverse[i][j] = augmented[i][j + 4];
    }
}

OwnedTabulatedMetric make_uniform_tabulated_metric(
    Real mass1, Real mass2,
    const kpolaris::Vec3<Real>& spin,
    const kpolaris::Vec3<Real>& center_at_zero,
    const kpolaris::Vec3<Real>& velocity,
    Real merger_weight) {
    constexpr int samples = 2;
    OwnedTabulatedMetric owned{
        SharedRealView("binary_sks_reference_times", samples),
        SharedRealView("binary_sks_reference_values",
                       kpolaris::binary_trajectory_field_count * samples),
        BinaryMetric()};
    auto host_times = Kokkos::create_mirror_view(owned.times);
    auto host_values = Kokkos::create_mirror_view(owned.values);
    host_times(0) = Real(-8);
    host_times(1) = Real(8);
    for (std::size_t i = 0; i < host_values.extent(0); ++i) {
        host_values(i) = Real(0);
    }
    const auto set = [&](kpolaris::BinaryTrajectoryField field,
                         int sample, Real value) {
        host_values(static_cast<std::size_t>(static_cast<int>(field)) * samples +
                    static_cast<std::size_t>(sample)) = value;
    };
    for (int sample = 0; sample < samples; ++sample) {
        const Real time = host_times(sample);
        const kpolaris::Vec3<Real> position =
            center_at_zero + velocity * time;
        set(kpolaris::BinaryTrajectoryField::mass1, sample, mass1);
        set(kpolaris::BinaryTrajectoryField::mass2, sample, mass2);
        set(kpolaris::BinaryTrajectoryField::position1_x, sample, position.x);
        set(kpolaris::BinaryTrajectoryField::position1_y, sample, position.y);
        set(kpolaris::BinaryTrajectoryField::position1_z, sample, position.z);
        set(kpolaris::BinaryTrajectoryField::position2_x, sample, position.x);
        set(kpolaris::BinaryTrajectoryField::position2_y, sample, position.y);
        set(kpolaris::BinaryTrajectoryField::position2_z, sample, position.z);
        set(kpolaris::BinaryTrajectoryField::velocity1_x, sample, velocity.x);
        set(kpolaris::BinaryTrajectoryField::velocity1_y, sample, velocity.y);
        set(kpolaris::BinaryTrajectoryField::velocity1_z, sample, velocity.z);
        set(kpolaris::BinaryTrajectoryField::velocity2_x, sample, velocity.x);
        set(kpolaris::BinaryTrajectoryField::velocity2_y, sample, velocity.y);
        set(kpolaris::BinaryTrajectoryField::velocity2_z, sample, velocity.z);
        set(kpolaris::BinaryTrajectoryField::kerr_a1_x, sample, spin.x);
        set(kpolaris::BinaryTrajectoryField::kerr_a1_y, sample, spin.y);
        set(kpolaris::BinaryTrajectoryField::kerr_a1_z, sample, spin.z);
        set(kpolaris::BinaryTrajectoryField::kerr_a2_x, sample,
            mass2 > Real(0) ? spin.x : Real(0));
        set(kpolaris::BinaryTrajectoryField::kerr_a2_y, sample,
            mass2 > Real(0) ? spin.y : Real(0));
        set(kpolaris::BinaryTrajectoryField::kerr_a2_z, sample,
            mass2 > Real(0) ? spin.z : Real(0));
        set(kpolaris::BinaryTrajectoryField::merger_weight, sample,
            merger_weight);
    }
    Kokkos::deep_copy(owned.times, host_times);
    Kokkos::deep_copy(owned.values, host_values);

    auto& orbit = owned.metric.orbit;
    orbit.mode = static_cast<int>(kpolaris::BinaryTrajectoryMode::tabulated);
    orbit.interpolation =
        static_cast<int>(kpolaris::BinaryTrajectoryInterpolation::linear);
    orbit.table_times = owned.times;
    orbit.table_values = owned.values;
    orbit.sample_count = samples;
    orbit.table_time_min = host_times(0);
    orbit.table_time_max = host_times(1);
    orbit.has_exact_postmerger_tail = merger_weight == Real(1) ? 1 : 0;
    orbit.enable_future_postmerger_extension = 0;
    return owned;
}

kpolaris::SuperposedKerrSchildMetric<Real> static_single_hole_metric() {
    kpolaris::SuperposedKerrSchildMetric<Real> metric;
    metric.orbit.total_mass = Real(1);
    metric.orbit.mass1 = Real(1);
    metric.orbit.mass2 = Real(0);
    metric.orbit.reference_separation = Real(6);
    metric.orbit.minimum_separation = Real(6);
    metric.orbit.orbit_enabled = 0;
    metric.orbit.inspiral_enabled = 0;
    metric.spin1 = kpolaris::Vec3<Real>(Real(0), Real(0), Real(0.3));
    metric.spin2 = kpolaris::Vec3<Real>();
    return metric;
}

void test_single_hole_limit() {
    const auto sks = static_single_hole_metric();
    const kpolaris::KerrSchildInMetric<Real> kerr(Real(1), Real(0.3));
    const kpolaris::Vec4<Real> x(Real(0.2), Real(4.1), Real(-1.7), Real(0.8));
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        for (int nu = 0; nu < kpolaris::ndim; ++nu) {
            require_close("single-hole gcov", sks.gcov(mu, nu, x),
                          kerr.gcov(mu, nu, x), Real(2e-12));
        }
    }
}

void test_tilted_spin_single_hole_rotation() {
    const Real mass = Real(0.93);
    const kpolaris::Vec3<Real> spin(
        Real(0.21), Real(-0.17), Real(0.29));
    const kpolaris::Vec3<Real> center(
        Real(-0.4), Real(0.25), Real(-0.1));
    const kpolaris::Vec3<Real> zero_velocity;
    const auto owned = make_uniform_tabulated_metric(
        mass, Real(0), spin, center, zero_velocity, Real(0));
    const kpolaris::Vec4<Real> points[] = {
        {Real(-0.3), Real(3.1), Real(-1.4), Real(0.8)},
        {Real(0.7), Real(-2.2), Real(2.7), Real(1.3)},
        {Real(1.1), Real(1.6), Real(0.9), Real(-2.5)}
    };
    for (const auto& x : points) {
        Real got[4][4], expected[4][4];
        owned.metric.gcov_matrix(x, got);
        explicit_boosted_rotated_kerr_gcov(
            mass, spin, center, zero_velocity, x, expected);
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
                require_close("tilted-spin rotated Kerr gcov",
                              got[mu][nu], expected[mu][nu], Real(8e-12));
            }
        }
    }
}

void test_boosted_tilted_single_hole_pullback() {
    const Real mass = Real(0.91);
    const kpolaris::Vec3<Real> spin(
        Real(0.18), Real(0.24), Real(-0.27));
    const kpolaris::Vec3<Real> center(
        Real(0.35), Real(-0.28), Real(0.16));
    const kpolaris::Vec3<Real> velocity(
        Real(0.17), Real(-0.11), Real(0.07));
    const auto owned = make_uniform_tabulated_metric(
        mass, Real(0), spin, center, velocity, Real(0));
    const kpolaris::Vec4<Real> points[] = {
        {Real(-0.6), Real(3.2), Real(-1.1), Real(0.7)},
        {Real(0.4), Real(-1.8), Real(2.4), Real(1.6)},
        {Real(1.2), Real(1.3), Real(0.8), Real(-2.1)}
    };
    for (const auto& x : points) {
        Real got[4][4], expected[4][4];
        owned.metric.gcov_matrix(x, got);
        explicit_boosted_rotated_kerr_gcov(
            mass, spin, center, velocity, x, expected);
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
                require_close("boosted tilted Kerr pullback gcov",
                              got[mu][nu], expected[mu][nu], Real(1e-11));
            }
        }
    }
}

void test_exact_remnant_matches_single_boosted_kerr() {
    const Real final_mass = Real(0.947);
    const kpolaris::Vec3<Real> final_spin(
        Real(-0.16), Real(0.22), Real(0.31));
    const kpolaris::Vec3<Real> center(
        Real(-0.2), Real(0.45), Real(-0.12));
    const kpolaris::Vec3<Real> kick(
        Real(0.08), Real(-0.05), Real(0.035));
    const auto owned = make_uniform_tabulated_metric(
        Real(0.5) * final_mass, Real(0.5) * final_mass,
        final_spin, center, kick, Real(1));
    const kpolaris::Vec4<Real> points[] = {
        {Real(-0.7), Real(3.4), Real(-1.5), Real(0.9)},
        {Real(0.35), Real(-2.1), Real(2.8), Real(1.2)},
        {Real(1.0), Real(1.5), Real(0.7), Real(-2.4)}
    };
    for (const auto& x : points) {
        Real got[4][4], got_inverse[4][4];
        Real expected[4][4], expected_inverse[4][4];
        owned.metric.gcov_matrix(x, got);
        owned.metric.gcon_matrix(x, got_inverse);
        explicit_boosted_rotated_kerr_gcov(
            final_mass, final_spin, center, kick, x, expected);
        invert_reference_metric(expected, expected_inverse);
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
                require_close("exact-remnant single Kerr gcov",
                              got[mu][nu], expected[mu][nu], Real(1e-11));
                require_close("exact-remnant single Kerr gcon",
                              got_inverse[mu][nu], expected_inverse[mu][nu],
                              Real(2e-10));
            }
        }

        Real got_connection[4][4][4];
        Real expected_dg[4][4][4];
        owned.metric.connection(x, got_connection);
        for (int alpha = 0; alpha < 4; ++alpha) {
            const Real scale = std::max(Real(1), std::abs(x[alpha]));
            const Real h = std::max(
                owned.metric.derivative_step * scale,
                sizeof(Real) <= sizeof(float) ? Real(1e-5) : Real(1e-8));
            auto xp = x;
            auto xm = x;
            xp[alpha] += h;
            xm[alpha] -= h;
            Real gp[4][4], gm[4][4];
            explicit_boosted_rotated_kerr_gcov(
                final_mass, final_spin, center, kick, xp, gp);
            explicit_boosted_rotated_kerr_gcov(
                final_mass, final_spin, center, kick, xm, gm);
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    expected_dg[mu][nu][alpha] =
                        (gp[mu][nu] - gm[mu][nu]) / (Real(2) * h);
                }
            }
        }
        for (int rho = 0; rho < 4; ++rho) {
            for (int mu = 0; mu < 4; ++mu) {
                for (int nu = 0; nu < 4; ++nu) {
                    Real value = Real(0);
                    for (int sigma = 0; sigma < 4; ++sigma) {
                        value += Real(0.5) * expected_inverse[rho][sigma] *
                            (expected_dg[sigma][nu][mu] +
                             expected_dg[sigma][mu][nu] -
                             expected_dg[mu][nu][sigma]);
                    }
                    require_close("exact-remnant single Kerr connection",
                                  got_connection[rho][mu][nu], value,
                                  Real(3e-8));
                }
            }
        }
    }
}

kpolaris::Vec4<Real> isolated_kerr_point(Real r, Real theta,
                                                Real phi, Real spin) {
    const Real cylindrical =
        std::sqrt(r * r + spin * spin) * std::sin(theta);
    return kpolaris::Vec4<Real>(
        Real(0),
        cylindrical * std::cos(phi),
        cylindrical * std::sin(phi),
        r * std::cos(theta));
}

kpolaris::Vec4<Real> isolated_kerr_point_for_spin(
    Real r, Real theta, Real phi, const kpolaris::Vec3<Real>& spin) {
    const Real spin_length = kpolaris::norm(spin);
    const kpolaris::Vec3<Real> axis(
        spin.x / spin_length, spin.y / spin_length, spin.z / spin_length);
    const kpolaris::Vec3<Real> reference =
        std::abs(axis.z) < Real(0.9) ?
        kpolaris::Vec3<Real>(Real(0), Real(0), Real(1)) :
        kpolaris::Vec3<Real>(Real(1), Real(0), Real(0));
    const auto e1_unscaled = kpolaris::cross(axis, reference);
    const Real e1_norm = kpolaris::norm(e1_unscaled);
    const kpolaris::Vec3<Real> e1(
        e1_unscaled.x / e1_norm,
        e1_unscaled.y / e1_norm,
        e1_unscaled.z / e1_norm);
    const auto e2 = kpolaris::cross(axis, e1);
    const Real cylindrical =
        std::sqrt(r * r + spin_length * spin_length) * std::sin(theta);
    const Real azimuth_c = std::cos(phi);
    const Real azimuth_s = std::sin(phi);
    const Real axial = r * std::cos(theta);
    const kpolaris::Vec3<Real> x =
        e1 * (cylindrical * azimuth_c) +
        e2 * (cylindrical * azimuth_s) +
        axis * axial;
    return kpolaris::Vec4<Real>(Real(0), x.x, x.y, x.z);
}

void test_single_hole_capture_surrogate() {
    const auto metric = static_single_hole_metric();
    const Real factor = Real(1.02);
    const Real horizon = metric.hole_horizon_radius(0);
    const Real target_r = factor * horizon;
    const Real angles[][2] = {
        {Real(0.2), Real(0.1)},
        {Real(0.7), Real(1.3)},
        {Real(1.2), Real(-0.8)}
    };
    for (const auto& angle : angles) {
        const auto on_surface =
            isolated_kerr_point(target_r, angle[0], angle[1], metric.spin1.z);
        require_close("isolated capture surface",
                      metric.inner_boundary_value(on_surface, factor),
                      Real(0), Real(3e-11));

        const auto inside = isolated_kerr_point(
            target_r * Real(0.99), angle[0], angle[1], metric.spin1.z);
        const auto outside = isolated_kerr_point(
            target_r * Real(1.01), angle[0], angle[1], metric.spin1.z);
        if (!(metric.inner_boundary_value(inside, factor) < Real(0)) ||
            !(metric.inner_boundary_value(outside, factor) > Real(0))) {
            throw std::runtime_error(
                "isolated capture surrogate has incorrect inner/outer sign");
        }
    }

    // The angle-dependent isolated target must also hold when the spin is
    // not aligned with the global z axis.
    auto tilted_metric = metric;
    tilted_metric.spin1 = kpolaris::Vec3<Real>(
        Real(0.18), Real(-0.12), Real(0.20784609690826528));
    const Real tilted_horizon = tilted_metric.hole_horizon_radius(0);
    for (const auto& angle : angles) {
        const auto on_surface = isolated_kerr_point_for_spin(
            factor * tilted_horizon, angle[0], angle[1],
            tilted_metric.spin1);
        require_close("tilted-spin isolated capture surface",
                      tilted_metric.inner_boundary_value(on_surface, factor),
                      Real(0), Real(5e-11));
    }

    // q rises again below the Kerr inner root.  The radial guard must keep a
    // genuinely deep interior point negative rather than opening a false hole.
    const auto deep_inside = isolated_kerr_point(
        target_r * Real(0.01), Real(0.4), Real(0.2), metric.spin1.z);
    if (!(metric.inner_boundary_value(deep_inside, factor) < Real(0))) {
        throw std::runtime_error(
            "capture surrogate became positive in the deep Kerr interior");
    }

    // Factors at/below one cannot use a positive exterior q target.  The
    // metric-level fallback retains the legacy radial surface; binary option
    // validation separately requires a factor above one.
    const Real fallback_factor = Real(0.9);
    const auto fallback_surface = isolated_kerr_point(
        fallback_factor * horizon, Real(0.8), Real(-0.2), metric.spin1.z);
    require_close("sub-horizon-factor radial fallback",
                  metric.inner_boundary_value(fallback_surface, fallback_factor),
                  Real(0), Real(3e-11));
}

void test_moving_radius_four_gradient() {
    kpolaris::BinaryRIAFRadiationModel<Real> model;
    model.reference_separation = Real(20);
    model.reference_phase = Real(0.35);
    model.chi1 = Real(0.3);
    model.chi2 = Real(-0.2);
    const auto metric = model.make_metric();
    const Real coordinate_time = Real(-3.5);
    const auto os = metric.orbit.state(metric.time_origin + coordinate_time);
    const kpolaris::Vec4<Real> x(
        coordinate_time,
        os.position1.x + Real(0.7),
        os.position1.y - Real(0.35),
        Real(0.4));
    const auto work = metric.build_hole_work(0, x);
    kpolaris::Vec4<Real> gradient;
    metric.hole_radius_gradient(0, x, work, gradient);

    for (int mu = 0; mu < 4; ++mu) {
        const Real h = mu == 0 ? Real(2e-4) : Real(2e-6);
        auto xp = x;
        auto xm = x;
        xp[mu] += h;
        xm[mu] -= h;
        const Real finite_difference =
            (metric.hole_radius_only(0, xp) - metric.hole_radius_only(0, xm)) /
            (Real(2) * h);
        require_close("moving local-r four-gradient", gradient[mu],
                      finite_difference, Real(3e-6));
    }
    if (!(std::abs(gradient[0]) > Real(1e-4))) {
        throw std::runtime_error(
            "moving local-r gradient omitted its time component");
    }
}

kpolaris::Vec4<Real> binary_ray_point(
    const kpolaris::SuperposedKerrSchildMetric<Real>& metric,
    int hole,
    const kpolaris::Vec3<Real>& direction,
    Real offset) {
    const auto os = metric.orbit.state(metric.time_origin);
    const auto center = hole == 0 ? os.position1 : os.position2;
    return kpolaris::Vec4<Real>(
        Real(0),
        center.x + offset * direction.x,
        center.y + offset * direction.y,
        center.z + offset * direction.z);
}

Real binary_capture_root(
    const kpolaris::SuperposedKerrSchildMetric<Real>& metric,
    int hole,
    const kpolaris::Vec3<Real>& direction,
    Real factor) {
    const Real horizon = metric.hole_horizon_radius(hole);
    Real lo = Real(0.45) * horizon;
    Real hi = Real(2.2) * horizon;
    Real flo = metric.inner_boundary_value(
        binary_ray_point(metric, hole, direction, lo), factor);
    Real fhi = metric.inner_boundary_value(
        binary_ray_point(metric, hole, direction, hi), factor);
    if (!(flo < Real(0)) || !(fhi > Real(0))) {
        throw std::runtime_error(
            "binary capture surrogate root is not bracketed");
    }
    for (int iteration = 0; iteration < 80; ++iteration) {
        const Real mid = Real(0.5) * (lo + hi);
        const Real fmid = metric.inner_boundary_value(
            binary_ray_point(metric, hole, direction, mid), factor);
        if (fmid > Real(0)) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    return Real(0.5) * (lo + hi);
}

void test_effective_min_step_affine_scaling() {
    kpolaris::BinaryRIAFRadiationModel<Real> model;
    model.reference_separation = Real(6.6);
    model.reference_phase = Real(0.37);
    model.chi1 = Real(0.3);
    model.chi2 = Real(-0.2);
    const auto metric = model.make_metric();
    const Real factor = Real(1.02);
    const kpolaris::Vec3<Real> direction(
        Real(0), Real(0), Real(1));
    const Real root = binary_capture_root(metric, 0, direction, factor);
    const Real horizon = metric.hole_horizon_radius(0);

    // Reproduce the scale of the late-inspiral critical rays that motivated
    // this hook without baking in a particular image pixel.  Solve on the
    // connected exterior branch for F=0.005.
    const Real target_surface_value = Real(0.005);
    Real lo = root;
    Real hi = root + Real(0.08) * horizon;
    if (!(metric.inner_boundary_value(
              binary_ray_point(metric, 0, direction, hi), factor) >
          target_surface_value)) {
        throw std::runtime_error(
            "late-inspiral effective-step test did not bracket F=0.005");
    }
    for (int iteration = 0; iteration < 80; ++iteration) {
        const Real mid = Real(0.5) * (lo + hi);
        const Real value = metric.inner_boundary_value(
            binary_ray_point(metric, 0, direction, mid), factor);
        if (value < target_surface_value) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const auto x = binary_ray_point(
        metric, 0, direction, Real(0.5) * (lo + hi));
    require_close("late-inspiral capture value",
                  metric.inner_boundary_value(x, factor),
                  target_surface_value, Real(2e-10));

    // Start from a physical future-directed null tangent aimed toward the
    // capture surface, then use the arbitrary affine normalization to reach
    // the |k| scales observed in the problematic rays.
    const kpolaris::Vec3<Real> spatial_k(
        Real(0), Real(0), Real(-1));
    kpolaris::Vec4<Real> unit_k(
        kpolaris::solve_future_null_k0(metric, x, spatial_k),
        spatial_k.x, spatial_k.y, spatial_k.z);
    const Real unit_scale = kpolaris::max_abs_component(unit_k);
    unit_k = unit_k * (Real(1) / unit_scale);

    const Real low_scale = Real(1000);
    const Real high_scale = Real(5000);
    const auto low_k = unit_k * low_scale;
    const auto high_k = unit_k * high_scale;
    require_close("late-inspiral low tangent scale",
                  kpolaris::max_abs_component(low_k), low_scale, Real(2e-12));
    require_close("late-inspiral high tangent scale",
                  kpolaris::max_abs_component(high_k), high_scale, Real(2e-12));

    const Real configured_min_step = Real(1e-5);
    const Real low_min = kpolaris::metric_effective_min_step(
        metric, low_k, configured_min_step);
    const Real high_min = kpolaris::metric_effective_min_step(
        metric, high_k, configured_min_step);
    require_close("late-inspiral low effective minimum", low_min,
                  configured_min_step / low_scale, Real(2e-14));
    require_close("late-inspiral high effective minimum", high_min,
                  configured_min_step / high_scale, Real(2e-14));
    require_close("affine-rescaled effective minimum", high_min,
                  low_min * low_scale / high_scale, Real(2e-14));

    // h*k is the physical coordinate displacement.  It must be unchanged by
    // k -> alpha*k and h_min -> h_min/alpha.
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        require_close("affine-invariant minimum displacement",
                      low_min * low_k[mu], high_min * high_k[mu],
                      Real(2e-13));
    }

    // The SKS directional capture predictor uses affine distance too, so its
    // rescaling provides an independent metric-level invariance check.
    const Real low_crossing =
        metric.inner_boundary_inward_crossing_distance(
            x, low_k, factor);
    const Real high_crossing =
        metric.inner_boundary_inward_crossing_distance(
            x, high_k, factor);
    if (!(low_crossing > Real(0)) || !(high_crossing > Real(0))) {
        throw std::runtime_error(
            "late-inspiral null tangent did not predict an inward crossing");
    }
    require_close("affine-rescaled capture distance", high_crossing,
                  low_crossing * low_scale / high_scale, Real(2e-10));
    require_close("affine-invariant capture/minimum ratio",
                  low_crossing / low_min,
                  high_crossing / high_min, Real(2e-8));

    const kpolaris::Vec4<Real> subunit_k(
        Real(0.5), Real(-0.25), Real(0.125), Real(-0.5));
    require_close("subunit SKS tangent keeps configured minimum",
                  kpolaris::metric_effective_min_step(
                      metric, subunit_k, configured_min_step),
                  configured_min_step, Real(2e-14));
}

void test_binary_capture_surrogate() {
    kpolaris::BinaryRIAFRadiationModel<Real> model;
    model.reference_separation = Real(20);
    model.reference_phase = Real(0);
    model.chi1 = Real(0.3);
    model.chi2 = Real(-0.2);
    const auto metric = model.make_metric();
    const Real factor = Real(1.02);
    const Real inv_sqrt_two = Real(0.7071067811865475244);
    const kpolaris::Vec3<Real> directions[] = {
        {Real(1), Real(0), Real(0)},
        {Real(-1), Real(0), Real(0)},
        {Real(0), Real(1), Real(0)},
        {Real(0), Real(-1), Real(0)},
        {Real(0), Real(0), Real(1)},
        {Real(0), Real(0), Real(-1)},
        {inv_sqrt_two, inv_sqrt_two, Real(0)},
        {-inv_sqrt_two, inv_sqrt_two, Real(0)},
        {inv_sqrt_two, Real(0), inv_sqrt_two},
        {-inv_sqrt_two, Real(0), inv_sqrt_two},
        {Real(0), inv_sqrt_two, inv_sqrt_two},
        {Real(0), -inv_sqrt_two, inv_sqrt_two}
    };

    Real min_ratio = Real(1e30);
    Real max_ratio = Real(-1e30);
    for (int hole = 0; hole < 2; ++hole) {
        const Real horizon = metric.hole_horizon_radius(hole);
        for (const auto& direction : directions) {
            const Real root =
                binary_capture_root(metric, hole, direction, factor);
            const auto root_point =
                binary_ray_point(metric, hole, direction, root);
            const auto work = metric.build_hole_work(hole, root_point);
            const Real ratio = work.r / horizon;
            min_ratio = std::min(min_ratio, ratio);
            max_ratio = std::max(max_ratio, ratio);
            if (!(ratio >= factor - Real(2e-10)) ||
                !(ratio < Real(1.5))) {
                throw std::runtime_error(
                    "binary capture surrogate has an unphysical candidate radius");
            }

            // Audit the complete signed interior segment from R0 to the
            // candidate root; it must not contain a positive pocket.
            for (int sample = 0; sample <= 64; ++sample) {
                const Real offset = root - Real(0.12) * horizon *
                    Real(64 - sample) / Real(64);
                const auto interior_point =
                    binary_ray_point(metric, hole, direction, offset);
                const auto interior_work =
                    metric.build_hole_work(hole, interior_point);
                if (interior_work.r / horizon >=
                    factor - Real(2e-10)) {
                    const Real value = metric.inner_boundary_value(
                        interior_point, factor);
                    if (!(value <= Real(2e-9))) {
                        throw std::runtime_error(
                            "binary capture surrogate has an interior positive pocket");
                    }
                }
            }

            // Audit the exterior connected branch for spurious negative shells.
            for (int sample = 1; sample <= 12; ++sample) {
                const Real offset = root +
                    Real(sample) / Real(12) * (Real(2.35) * horizon - root);
                const Real value = metric.inner_boundary_value(
                    binary_ray_point(metric, hole, direction, offset), factor);
                if (!(value > Real(-2e-10))) {
                    throw std::runtime_error(
                        "binary capture surrogate has an exterior negative shell");
                }
            }
            const Real inner_value = metric.inner_boundary_value(
                binary_ray_point(metric, hole, direction,
                                 Real(0.7) * horizon),
                factor);
            if (!(inner_value < Real(0))) {
                throw std::runtime_error(
                    "binary capture surrogate lost its deep-interior sign");
            }
        }
    }
    if (!(max_ratio - min_ratio > Real(1e-3)) ||
        !(max_ratio > factor + Real(1e-3))) {
        throw std::runtime_error(
            "full binary metric did not deform the capture surrogate");
    }

    // The limiter must leave an inward trial safely outside F=0; the
    // event-first hook owns the final gap once it is within min_step.
    const auto direction = directions[4];
    const Real root = binary_capture_root(metric, 0, direction, factor);
    const Real horizon = metric.hole_horizon_radius(0);
    const auto x = binary_ray_point(
        metric, 0, direction, root + Real(0.03) * horizon);
    const kpolaris::Vec4<Real> inward(
        Real(0), -direction.x, -direction.y, -direction.z);
    const Real crossing = metric.inner_boundary_inward_crossing_distance(
        x, inward, factor);
    const Real proposed = Real(0.5);
    const Real limited = metric.inner_boundary_step_limit(
        x, inward, factor, proposed);
    if (!(crossing > Real(0)) || !(limited > Real(0)) ||
        !(limited < crossing) || !(limited < proposed)) {
        throw std::runtime_error(
            "capture-surrogate limiter did not take a safe exterior step");
    }
    auto safe_x = x;
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        safe_x[mu] += limited * inward[mu];
    }
    if (!(metric.inner_boundary_value(safe_x, factor) > Real(0))) {
        throw std::runtime_error(
            "capture-surrogate safe step crossed the inner surface");
    }
    if (!metric.inner_boundary_step_is_terminal(
            x, inward, factor, crossing)) {
        throw std::runtime_error(
            "near transverse capture crossing was not terminal");
    }
    if (metric.inner_boundary_step_is_terminal(
            x, inward, factor, Real(0.5) * crossing)) {
        throw std::runtime_error(
            "capture crossing beyond min_step was terminated early");
    }
    const kpolaris::Vec4<Real> moving_inward(
        Real(1), -direction.x, -direction.y, -direction.z);
    const Real moving_crossing =
        metric.inner_boundary_inward_crossing_distance(
            x, moving_inward, factor);
    if (!(moving_crossing > Real(0)) ||
        !metric.inner_boundary_step_is_terminal(
            x, moving_inward, factor, moving_crossing)) {
        throw std::runtime_error(
            "moving transverse capture crossing was not terminal");
    }
    const kpolaris::Vec4<Real> tangent(
        Real(0), Real(0), Real(1), Real(0));
    if (metric.inner_boundary_step_is_terminal(
            x, tangent, factor, crossing)) {
        throw std::runtime_error(
            "tangent capture trajectory was falsely terminated");
    }

}

void test_inverse_and_symmetry() {
    kpolaris::BinaryRIAFRadiationModel<Real> model;
    model.reference_separation = Real(14);
    model.reference_phase = Real(0.3);
    model.chi1 = Real(0.4);
    model.chi2 = Real(-0.2);
    const auto metric = model.make_metric();
    const kpolaris::Vec4<Real> samples[] = {
        {Real(0), Real(0), Real(0), Real(2)},
        {Real(-3), Real(5), Real(1), Real(-0.4)},
        {Real(-12), Real(25), Real(-8), Real(4)}
    };
    for (const auto& x : samples) {
        Real g[4][4], gi[4][4];
        metric.gcov_matrix(x, g);
        metric.gcon_matrix(x, gi);
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
                require_close("metric symmetry", g[mu][nu], g[nu][mu], Real(2e-12));
                Real product = Real(0);
                for (int a = 0; a < 4; ++a) product += g[mu][a] * gi[a][nu];
                require_close("metric inverse", product,
                              mu == nu ? Real(1) : Real(0), Real(2e-10));
            }
        }
    }
}

void test_orbit_and_boost() {
    kpolaris::LeadingOrderBinaryOrbit<Real> orbit;
    orbit.mass1 = Real(0.7);
    orbit.mass2 = Real(0.3);
    orbit.reference_separation = Real(18);
    const auto os = orbit.state(Real(-4));
    const auto com_x = os.position1 * orbit.mass1 + os.position2 * orbit.mass2;
    const auto com_v = os.velocity1 * orbit.mass1 + os.velocity2 * orbit.mass2;
    require_close("COM x", com_x.x, Real(0), Real(2e-13));
    require_close("COM y", com_x.y, Real(0), Real(2e-13));
    require_close("COM vx", com_v.x, Real(0), Real(2e-13));
    require_close("COM vy", com_v.y, Real(0), Real(2e-13));

    kpolaris::SuperposedKerrSchildMetric<Real> metric;
    metric.orbit = orbit;
    const kpolaris::Vec4<Real> x(Real(-4), Real(2), Real(-1), Real(0.4));
    const auto hole = metric.hole_kinematics(0, x);
    const kpolaris::Vec4<Real> u(Real(1.3), Real(0.2), Real(-0.1), Real(0.4));
    const auto rest = metric.lab_to_rest_vector(hole, u);
    const auto roundtrip = metric.rest_to_lab_vector(hole, rest);
    for (int mu = 0; mu < 4; ++mu) {
        require_close("boost roundtrip", roundtrip[mu], u[mu], Real(3e-12));
    }
}

void test_dynamic_time_derivative() {
    kpolaris::BinaryRIAFRadiationModel<Real> model;
    model.reference_separation = Real(16);
    model.reference_phase = Real(0.4);
    const auto metric = model.make_metric();
    const kpolaris::Vec4<Real> x(Real(-2), Real(3), Real(4), Real(1));
    Real dg[4][4][4];
    metric.dgcov(x, dg);
    Real max_dt = Real(0);
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            max_dt = std::max(max_dt, std::abs(dg[mu][nu][0]));
        }
    }
    if (!(max_dt > Real(1e-8))) {
        throw std::runtime_error("dynamic metric has zero time derivative");
    }
}

void test_tabulated_time_domain_boundary_hooks() {
    const kpolaris::Vec3<Real> spin(
        Real(0), Real(0), Real(0.2));
    const kpolaris::Vec3<Real> center;
    const kpolaris::Vec3<Real> velocity;
    auto owned = make_uniform_tabulated_metric(
        Real(1), Real(0), spin, center, velocity, Real(0));
    auto& metric = owned.metric;

    metric.time_origin = Real(-7.95);
    const kpolaris::Vec4<Real> event(
        Real(0), Real(10), Real(0), Real(0));
    const kpolaris::Vec4<Real> past(
        Real(-2), Real(-1), Real(0), Real(0));
    require_close(
        "SKS lower time-boundary affine distance",
        metric.time_domain_boundary_distance(event, past, Real(1)),
        Real(0.025), Real(1e-13));
    require_close(
        "SKS lower time-boundary safe step",
        kpolaris::metric_limit_time_domain_step(
            metric, event, past, Real(1)),
        Real(0.0125), Real(1e-13));
    if (kpolaris::metric_time_domain_step_is_terminal(
            metric, event, past, Real(1), Real(0.01))) {
        throw std::runtime_error(
            "SKS time boundary was terminal outside its minimum-step window");
    }
    if (!kpolaris::metric_time_domain_step_is_terminal(
            metric, event, past, Real(1), Real(0.02))) {
        throw std::runtime_error(
            "SKS time boundary was not terminal inside its minimum-step window");
    }

    metric.time_origin = Real(7.95);
    const kpolaris::Vec4<Real> future(
        Real(2), Real(1), Real(0), Real(0));
    require_close(
        "SKS upper time-boundary affine distance",
        metric.time_domain_boundary_distance(event, future, Real(1)),
        Real(0.025), Real(1e-13));
    metric.orbit.has_exact_postmerger_tail = 1;
    metric.orbit.enable_future_postmerger_extension = 1;
    if (!(metric.time_domain_boundary_distance(
              event, future, Real(1)) < Real(0))) {
        throw std::runtime_error(
            "exact Kerr future extension retained a false upper time boundary");
    }
    metric.orbit.has_exact_postmerger_tail = 0;
    metric.orbit.enable_future_postmerger_extension = 0;

    // At t_min an inward capture predictor has only the +time-valid side of
    // its directional stencil.  It must use that one-sided derivative, while
    // the terminal sign probe must refuse to sample beyond the table.
    metric.time_origin = metric.orbit.table_time_min;
    const Real factor = Real(1.02);
    const kpolaris::Vec3<Real> direction(
        Real(0), Real(0), Real(1));
    const Real root =
        binary_capture_root(metric, 0, direction, factor);
    const Real horizon = metric.hole_horizon_radius(0);
    const auto near_surface = binary_ray_point(
        metric, 0, direction, root + Real(0.03) * horizon);
    const kpolaris::Vec4<Real> spatial_inward(
        Real(0), -direction.x, -direction.y, -direction.z);
    const kpolaris::Vec4<Real> past_inward(
        Real(-1), -direction.x, -direction.y, -direction.z);
    const Real central_crossing =
        metric.inner_boundary_inward_crossing_distance(
            near_surface, spatial_inward, factor);
    const Real one_sided_crossing =
        metric.inner_boundary_inward_crossing_distance(
            near_surface, past_inward, factor);
    if (!(central_crossing > Real(0)) ||
        !(one_sided_crossing > Real(0))) {
        throw std::runtime_error(
            "SKS capture predictor failed at the lower time boundary");
    }
    require_close(
        "SKS one-sided capture derivative",
        one_sided_crossing, central_crossing, Real(2e-4));
    if (!metric.time_domain_step_is_terminal(
            near_surface, past_inward, Real(1), one_sided_crossing)) {
        throw std::runtime_error(
            "SKS lower table endpoint was not a terminal time event");
    }
    if (metric.inner_boundary_step_is_terminal(
            near_surface, past_inward, factor, one_sided_crossing)) {
        throw std::runtime_error(
            "SKS capture terminal probe sampled beyond the table time domain");
    }
}

void test_binary_riaf_tidal_taper() {
    kpolaris::BinaryRIAFRadiationModel<Real> model;
    const Real rout = model.outer_radius_hat(0, Real(20));
    if (!(rout > Real(1) && rout < model.disk.r_max)) {
        throw std::runtime_error("tidal truncation did not limit mini-disk");
    }
    require_close("taper interior", model.outer_taper(Real(0.5) * rout, rout),
                  Real(1), Real(1e-14));
    require_close("taper edge", model.outer_taper(rout, rout), Real(0), Real(0));
}

void test_metric_domain_admissibility() {
    kpolaris::BinaryRIAFRadiationModel<Real> valid;
    valid.reference_separation = Real(20);
    valid.chi1 = Real(0.3);
    valid.chi2 = Real(-0.2);
    const Real safe_minimum = valid.validate_metric_domain(Real(20));
    if (!(safe_minimum >= Real(0.1))) {
        throw std::runtime_error("default binary metric failed its signature margin");
    }
    if (!(valid.sampled_min_fluid_slice_timelike_margin > Real(1e-8))) {
        throw std::runtime_error(
            "default binary RIAF failed its fluid-slice timelike margin");
    }

    kpolaris::BinaryRIAFRadiationModel<Real> late_valid = valid;
    late_valid.reference_separation = Real(6.6);
    late_valid.minimum_separation = Real(6);
    late_valid.validate_metric_domain(Real(8));
    if (!(late_valid.sampled_min_fluid_slice_timelike_margin > Real(1e-8))) {
        throw std::runtime_error(
            "late-inspiral binary RIAF failed its fluid-slice timelike margin");
    }

    const Real invalid_capture_factors[] = {Real(1), Real(1.051)};
    for (const Real factor : invalid_capture_factors) {
        kpolaris::BinaryRIAFRadiationModel<Real> bad_capture;
        bad_capture.capture_factor = factor;
        bool capture_rejected = false;
        try {
            bad_capture.validate_metric_domain(Real(20));
        } catch (const std::runtime_error&) {
            capture_rejected = true;
        }
        if (!capture_rejected) {
            throw std::runtime_error(
                "unsupported binary capture factor was not rejected");
        }
    }

    kpolaris::BinaryRIAFRadiationModel<Real> invalid;
    invalid.reference_separation = Real(6.0001);
    invalid.minimum_separation = Real(6);
    invalid.inspiral_enabled = 0;
    invalid.orbit_enabled = 1;
    invalid.chi1 = Real(-1);
    invalid.chi2 = Real(-1);
    bool rejected = false;
    try {
        invalid.validate_metric_domain(Real(8));
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected) {
        throw std::runtime_error(
            "wrong-signature Superposed Kerr-Schild parameters were not rejected");
    }
}

void test_device_metric_smoke() {
    kpolaris::BinaryRIAFRadiationModel<Real> model;
    model.reference_separation = Real(15);
    const auto metric = model.make_metric();
    Kokkos::View<Real*> errors("binary_sks_errors", 2);
    Kokkos::View<Real*> capture("binary_sks_capture", 4);
    Kokkos::parallel_for("BinarySKSDeviceSmoke", 2, KOKKOS_LAMBDA(int i) {
        const kpolaris::Vec4<Real> x(Real(-i), Real(2 + i), Real(1), Real(0.5));
        Real g[4][4], gi[4][4];
        metric.gcov_matrix(x, g);
        metric.gcon_matrix(x, gi);
        Real err = Real(0);
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
                Real p = Real(0);
                for (int a = 0; a < 4; ++a) p += g[mu][a] * gi[a][nu];
                const Real target = mu == nu ? Real(1) : Real(0);
                err = kpolaris::max_val(err, kpolaris::abs_val(p - target));
            }
        }
        const auto work = metric.build_work(x);
        const kpolaris::Vec4<Real> q(
            Real(1.1), Real(-0.2), Real(0.3), Real(0.4));
        kpolaris::Vec4<Real> raised;
        metric.raise_covector_from_work(work, q, raised);
        for (int mu = 0; mu < 4; ++mu) {
            Real expected = Real(0);
            for (int nu = 0; nu < 4; ++nu) {
                expected += gi[mu][nu] * q[nu];
            }
            err = kpolaris::max_val(
                err, kpolaris::abs_val(raised[mu] - expected));
        }
        errors(i) = err;

        const auto os = metric.orbit.state(metric.time_origin);
        const auto center = i == 0 ? os.position1 : os.position2;
        const Real horizon = metric.hole_horizon_radius(i);
        const kpolaris::Vec4<Real> near_hole(
            Real(0), center.x, center.y, center.z + Real(1.2) * horizon);
        const kpolaris::Vec4<Real> inward(
            Real(0), Real(0), Real(0), Real(-1));
        capture(2 * i) =
            metric.inner_boundary_value(near_hole, Real(1.02));
        capture(2 * i + 1) = metric.inner_boundary_step_limit(
            near_hole, inward, Real(1.02), Real(0.3));
    });
    const auto host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), errors);
    const auto capture_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), capture);
    for (int i = 0; i < 2; ++i) {
        if (!(host(i) < Real(2e-9))) {
            throw std::runtime_error("device SKS inverse residual too large");
        }
        if (!std::isfinite(capture_host(2 * i)) ||
            !std::isfinite(capture_host(2 * i + 1)) ||
            !(capture_host(2 * i + 1) > Real(0)) ||
            !(capture_host(2 * i + 1) <= Real(0.3))) {
            throw std::runtime_error(
                "device capture surrogate or step limiter is invalid");
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    try {
        test_single_hole_limit();
        test_tilted_spin_single_hole_rotation();
        test_boosted_tilted_single_hole_pullback();
        test_exact_remnant_matches_single_boosted_kerr();
        test_single_hole_capture_surrogate();
        test_moving_radius_four_gradient();
        test_effective_min_step_affine_scaling();
        test_binary_capture_surrogate();
        test_inverse_and_symmetry();
        test_orbit_and_boost();
        test_dynamic_time_derivative();
        test_tabulated_time_domain_boundary_hooks();
        test_binary_riaf_tidal_taper();
        test_metric_domain_admissibility();
        test_device_metric_smoke();
    } catch (...) {
        Kokkos::finalize();
        throw;
    }
    Kokkos::finalize();
    std::cout << "KPolaris binary SKS tests passed\n";
    return 0;
}
