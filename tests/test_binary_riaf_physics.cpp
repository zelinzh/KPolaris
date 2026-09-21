#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "camera/camera.hpp"
#include "model/binary_riaf.hpp"

namespace {
using Real = kpolaris::DefaultReal;

void require_close(const std::string& name, Real got, Real expected,
                   Real tol = Real(2e-10)) {
    const Real scale = std::max<Real>(Real(1), std::abs(expected));
    if (!std::isfinite(got) || std::abs(got - expected) > tol * scale) {
        throw std::runtime_error(name + " mismatch");
    }
}

void require_finite_coeffs(const kpolaris::TransferCoeffs<Real>& c) {
    const Real values[] = {c.jI,c.jQ,c.jU,c.jV,c.aI,c.aQ,c.aU,c.aV,
                           c.rQ,c.rU,c.rV};
    for (Real v : values) {
        if (!std::isfinite(v)) throw std::runtime_error("non-finite transfer coefficient");
    }
    if (c.jI < Real(0) || c.aI < Real(0)) {
        throw std::runtime_error("negative Stokes-I emission/absorption coefficient");
    }
}

void test_bl_cartesian_ks_coordinate_and_vector_convention() {
    kpolaris::RIAFAnalyticRadiationModel<Real> disk;
    const Real spins[] = {Real(0.7), Real(-0.7)};
    const Real r0 = Real(5.3);
    const Real th0 = Real(1.1);
    const Real phi0 = Real(-0.7);
    const Real cp0 = std::cos(phi0);
    const Real sp0 = std::sin(phi0);
    const Real st0 = std::sin(th0);
    const Real ct0 = std::cos(th0);

    for (Real spin : spins) {
        // KerrSchildInMetric uses
        // x=(r cos(phi)-a sin(phi)) sin(theta),
        // y=(r sin(phi)+a cos(phi)) sin(theta).
        const kpolaris::Vec4<Real> x(
            Real(0),
            (r0 * cp0 - spin * sp0) * st0,
            (r0 * sp0 + spin * cp0) * st0,
            r0 * ct0);
        Real r, th, cp, sp;
        disk.bl_coordinates(x, spin, r, th, cp, sp);
        require_close("Cartesian-KS -> BL r", r, r0, Real(2e-13));
        require_close("Cartesian-KS -> BL theta", th, th0, Real(2e-13));
        require_close("Cartesian-KS -> BL cos(phi)", cp, cp0, Real(2e-13));
        require_close("Cartesian-KS -> BL sin(phi)", sp, sp0, Real(2e-13));

        // A nonzero radial component is essential: a purely circular axial
        // vector is insensitive to the erroneous spin-sign convention that
        // this regression test is meant to catch.
        const Real vt_bl = Real(1.4);
        const Real vr_bl = Real(-0.08);
        const Real vth_bl = Real(0.03);
        const Real vphi_bl = Real(0.12);
        const Real delta = r0 * r0 - Real(2) * r0 + spin * spin;
        const Real vt_ks = vt_bl + Real(2) * r0 / delta * vr_bl;
        const Real vphi_ks = vphi_bl + spin / delta * vr_bl;
        const kpolaris::Vec4<Real> got = disk.bl_to_cartesian_ks_vector(
            vt_bl, vr_bl, vth_bl, vphi_bl, r0, th0, cp0, sp0, spin);
        const kpolaris::Vec4<Real> expected(
            vt_ks,
            vr_bl * cp0 * st0 +
                vth_bl * (r0 * cp0 - spin * sp0) * ct0 +
                vphi_ks * (-r0 * sp0 - spin * cp0) * st0,
            vr_bl * sp0 * st0 +
                vth_bl * (r0 * sp0 + spin * cp0) * ct0 +
                vphi_ks * (r0 * cp0 - spin * sp0) * st0,
            vr_bl * ct0 - vth_bl * r0 * st0);
        for (int mu = 0; mu < kpolaris::ndim; ++mu) {
            require_close("BL -> Cartesian-KS vector", got[mu], expected[mu],
                          Real(2e-13));
        }

        Real gcov_bl[4][4], gcon_bl[4][4];
        disk.bl_metric(r0, th0, spin, gcov_bl, gcon_bl);
        const Real vbl[4] = {vt_bl, vr_bl, vth_bl, vphi_bl};
        Real norm_bl = Real(0);
        for (int mu = 0; mu < 4; ++mu) {
            for (int nu = 0; nu < 4; ++nu) {
                norm_bl += gcov_bl[mu][nu] * vbl[mu] * vbl[nu];
            }
        }
        const kpolaris::KerrSchildInMetric<Real> metric(Real(1), spin);
        require_close("BL/Cartesian-KS vector norm", metric.dot(x, got, got),
                      norm_bl, Real(2e-12));
    }
}

kpolaris::TransportState<Real> state_in_disk(
    const kpolaris::BinaryRIAFRadiationModel<Real>& model,
    const kpolaris::SuperposedKerrSchildMetric<Real>& metric,
    int hole) {
    kpolaris::TransportState<Real> state;
    const kpolaris::Vec4<Real> origin(Real(0), Real(0), Real(0), Real(0));
    const auto hk = metric.hole_kinematics(hole, origin);
    const Real chi = hole == 0 ? model.chi1 : model.chi2;
    const Real rhat = Real(6);
    const Real xhat = std::sqrt(rhat * rhat + chi * chi);
    state.x = kpolaris::Vec4<Real>(Real(0),
        hk.position.x + hk.mass * xhat,
        hk.position.y,
        hk.position.z);
    const kpolaris::Vec3<Real> ksp(Real(-0.35), Real(0.2), Real(1));
    state.k = kpolaris::Vec4<Real>(
        kpolaris::solve_future_null_k0(metric, state.x, ksp),
        ksp.x, ksp.y, ksp.z);
    kpolaris::initialize_screen_frame_from_reference(
        metric, state.x, state.k, state.e1, state.e2);
    return state;
}

void test_local_plasma_invariants() {
    kpolaris::BinaryRIAFRadiationModel<Real> model;
    model.reference_separation = Real(20);
    model.chi1 = Real(0.3);
    model.chi2 = Real(-0.2);
    const auto metric = model.make_metric();
    const auto state = state_in_disk(model, metric, 0);
    const auto hw = metric.build_hole_work(0, state.x);
    const Real ma = hw.hole.mass;
    kpolaris::Vec4<Real> xhat = hw.rest_x;
    xhat[1] /= ma; xhat[2] /= ma; xhat[3] /= ma;
    const kpolaris::KerrSchildInMetric<Real> local_metric(Real(1), model.chi1);
    Real r, th, cp, sp;
    model.disk.bl_coordinates(xhat, model.chi1, r, th, cp, sp);
    Real glocal[4][4];
    local_metric.gcov_matrix(xhat, glocal);
    const auto urest = model.disk.fluid_four_velocity_from_bl_coords(
        local_metric, xhat, r, th, cp, sp, glocal);
    const auto brest = model.disk.magnetic_unit_four_vector_with_gcov(
        local_metric, xhat, urest, glocal);

    auto u = metric.rest_to_lab_vector(hw.hole, urest);
    auto b = metric.rest_to_lab_vector(hw.hole, brest);
    Real g[4][4];
    metric.gcov_matrix(state.x, g);
    const Real unorm0 = model.dot_with_metric(g, u, u);
    if (!(unorm0 < Real(0)) || !(u[0] > Real(0))) {
        throw std::runtime_error("boosted mini-disk velocity is not future timelike");
    }
    u = u * (Real(1) / std::sqrt(-unorm0));
    b = b + u * model.dot_with_metric(g, u, b);
    const Real bsq0 = model.dot_with_metric(g, b, b);
    if (!(bsq0 > Real(0))) throw std::runtime_error("mini-disk magnetic axis is not spacelike");
    b = b * (Real(1) / std::sqrt(bsq0));

    require_close("u.u", model.dot_with_metric(g, u, u), Real(-1));
    require_close("u.b", model.dot_with_metric(g, u, b), Real(0));
    require_close("b.b", model.dot_with_metric(g, b, b), Real(1));
    const Real fluid_frequency = -model.dot_with_metric(g, u, state.k);
    if (!(fluid_frequency > Real(0))) {
        throw std::runtime_error("future photon has non-positive fluid frequency");
    }
}

kpolaris::Vec4<Real> lab_event_from_rest_equatorial(
    const kpolaris::BinaryRIAFRadiationModel<Real>& model,
    const kpolaris::SuperposedKerrSchildMetric<Real>& metric,
    int hole, Real rhat, Real phi) {
    const kpolaris::Vec4<Real> origin(Real(0), Real(0), Real(0), Real(0));
    const auto hk = metric.hole_kinematics(hole, origin);
    const Real chi = kpolaris::norm(hk.spin) / hk.mass;
    const Real rhohat = std::sqrt(rhat * rhat + chi * chi);
    const kpolaris::Vec4<Real> Xlocal(Real(0),
        hk.mass * rhohat * std::cos(phi),
        hk.mass * rhohat * std::sin(phi), Real(0));
    const kpolaris::Vec4<Real> Xrest4 =
        model.spin_vector_to_rest_frame(Xlocal, hk.spin);
    const kpolaris::Vec3<Real> X(
        Xrest4[1], Xrest4[2], Xrest4[3]);
    const Real v2 = kpolaris::dot(hk.velocity, hk.velocity);
    const Real vdotx = kpolaris::dot(hk.velocity, X);
    const Real inverse_factor = v2 > Real(0) ?
        (Real(1) / hk.gamma - Real(1)) * vdotx / v2 : Real(0);
    const auto dx = X + hk.velocity * inverse_factor;
    return kpolaris::Vec4<Real>(
        Real(0), hk.position.x + dx.x, hk.position.y + dx.y,
        hk.position.z + dx.z);
}

Real check_paired_frame_point(
    const kpolaris::BinaryRIAFRadiationModel<Real>& model,
    const kpolaris::SuperposedKerrSchildMetric<Real>& metric,
    int hole, Real rhat, Real phi,
    bool require_positive_emission = false) {
    const auto x = lab_event_from_rest_equatorial(
        model, metric, hole, rhat, phi);
    const auto work = metric.build_work(x);
    const auto& hw = work.holes[hole];
    kpolaris::Vec4<Real> xhat = model.rest_point_to_spin_frame(
        hw.rest_x, hw.hole.spin);
    xhat[1] /= hw.hole.mass;
    xhat[2] /= hw.hole.mass;
    xhat[3] /= hw.hole.mass;
    const Real chi = kpolaris::norm(hw.hole.spin) / hw.hole.mass;
    const kpolaris::KerrSchildInMetric<Real> local_metric(Real(1), chi);
    const auto local_work = local_metric.build_work(xhat);
    Real r, th, cp, sp;
    model.disk.bl_coordinates(xhat, chi, r, th, cp, sp);
    require_close("requested local radius", r, rhat, Real(3e-12));

    Real local_gcov[4][4], full_gcov[4][4], full_gcon[4][4];
    local_metric.gcov_matrix(xhat, local_gcov);
    metric.gcov_matrix_from_work(work, full_gcov);
    metric.gcon_matrix_from_work(work, full_gcon);
    const auto urest = model.disk.fluid_four_velocity_from_bl_coords(
        local_metric, xhat, r, th, cp, sp, local_gcov);
    const auto brest = model.disk.magnetic_unit_four_vector_with_gcov(
        local_metric, xhat, urest, local_gcov);
    require_close("local paired-frame u.u",
                  model.dot_with_metric(local_gcov, urest, urest), Real(-1),
                  Real(3e-10));

    const auto old_trial = metric.rest_to_lab_vector(
        hw.hole, model.spin_vector_to_rest_frame(urest, hw.hole.spin));
    const Real old_trial_norm = model.dot_with_metric(
        full_gcov, old_trial, old_trial);
    kpolaris::Vec4<Real> ucon, bcon;
    Real slice_margin = Real(-1);
    if (!model.map_plasma_state_to_full_metric(
            hole, metric, work, local_work, urest, brest,
            local_gcov, full_gcov, ucon, bcon,
            &slice_margin)) {
        throw std::runtime_error("paired-frame plasma map failed");
    }
    if (!(slice_margin > Real(1e-8)) || !(ucon[0] > Real(0))) {
        throw std::runtime_error("paired-frame time orientation/margin failed");
    }
    require_close("paired-frame u.u",
                  model.dot_with_metric(full_gcov, ucon, ucon), Real(-1),
                  Real(3e-10));
    require_close("paired-frame b.b",
                  model.dot_with_metric(full_gcov, bcon, bcon), Real(1),
                  Real(3e-10));
    require_close("paired-frame u.b",
                  model.dot_with_metric(full_gcov, ucon, bcon), Real(0),
                  Real(3e-10));

    // Build a manifestly future null vector from the full-metric coordinate-
    // time normal.  A fixed spatial-k quadratic root can select the wrong
    // branch inside an ergoregion and is unsuitable for this invariant test.
    const kpolaris::Vec4<Real> qtime(
        Real(1), Real(0), Real(0), Real(0));
    kpolaris::Vec4<Real> observer;
    if (!model.slice_unit_normal(full_gcon, qtime, observer)) {
        throw std::runtime_error("future-null test observer failed");
    }
    kpolaris::Vec4<Real> direction(
        Real(0), Real(0), Real(0), Real(1));
    direction = direction + observer *
        model.dot_with_metric(full_gcov, observer, direction);
    const Real direction2 = model.dot_with_metric(
        full_gcov, direction, direction);
    if (!(direction2 > Real(0))) {
        throw std::runtime_error("future-null test direction failed");
    }
    direction = direction * (Real(1) / std::sqrt(direction2));
    const kpolaris::Vec4<Real> k = observer + direction;
    require_close("paired-frame photon null",
                  model.dot_with_metric(full_gcov, k, k), Real(0),
                  Real(3e-10));
    if (!(-model.dot_with_metric(full_gcov, ucon, k) > Real(0))) {
        throw std::runtime_error("paired-frame future photon frequency failed");
    }
    if (require_positive_emission) {
        kpolaris::TransportState<Real> state;
        state.x = x;
        state.k = k;
        kpolaris::initialize_screen_frame_from_reference(
            metric, state.x, state.k, state.e1, state.e2);
        const auto coeffs = model.component_coefficients(
            hole, metric, state);
        require_finite_coeffs(coeffs);
        if (!(coeffs.jI > Real(0))) {
            throw std::runtime_error(
                "paired-frame worst point has no positive Stokes-I emission");
        }
    }
    return old_trial_norm;
}

void test_paired_frame_single_hole_identity() {
    kpolaris::BinaryRIAFRadiationModel<Real> model;
    model.chi1 = Real(0.4);
    const kpolaris::KerrSchildInMetric<Real> local_metric(
        Real(1), model.chi1);
    const Real rhat = Real(4.2);
    const Real phi = Real(0.8);
    const Real rhohat = std::sqrt(rhat * rhat + model.chi1 * model.chi1);
    const kpolaris::Vec4<Real> xhat(
        Real(0), rhohat * std::cos(phi), rhohat * std::sin(phi), Real(0));
    const auto local_work = local_metric.build_work(xhat);
    Real r, th, cp, sp;
    model.disk.bl_coordinates(xhat, model.chi1, r, th, cp, sp);
    Real gcov[4][4];
    local_metric.gcov_matrix(xhat, gcov);
    const auto urest = model.disk.fluid_four_velocity_from_bl_coords(
        local_metric, xhat, r, th, cp, sp, gcov);
    const auto brest = model.disk.magnetic_unit_four_vector_with_gcov(
        local_metric, xhat, urest, gcov);

    kpolaris::SuperposedKerrSchildMetric<Real> metric;
    kpolaris::SuperposedKerrSchildMetric<Real>::Work work{};
    work.holes[0].hole.mass = Real(1);
    work.holes[0].hole.gamma = Real(1);
    work.holes[0].hole.velocity =
        kpolaris::Vec3<Real>(Real(0), Real(0), Real(0));
    work.holes[0].amp = local_work.amp;
    for (int mu = 0; mu < 4; ++mu) {
        work.holes[0].l_global[mu] = local_work.l[mu];
    }
    work.holes[1].amp = Real(0);
    for (int mu = 0; mu < 4; ++mu) work.holes[1].l_global[mu] = Real(0);
    kpolaris::Vec4<Real> ucon, bcon;
    if (!model.map_plasma_state_to_full_metric(
            0, metric, work, local_work, urest, brest,
            gcov, gcov, ucon, bcon)) {
        throw std::runtime_error("single-hole paired-frame identity failed");
    }
    for (int mu = 0; mu < 4; ++mu) {
        require_close("single-hole u identity", ucon[mu], urest[mu],
                      Real(3e-10));
        require_close("single-hole b identity", bcon[mu], brest[mu],
                      Real(3e-10));
    }

    // Also retain an actual orbital boost while removing only the companion
    // rank-one metric term.  This locks the covector q=(gamma,-gamma v) and
    // contravariant rest_to_lab_vector sign conventions independently of the
    // full-binary invariant tests.
    model.chi1 = Real(0.3);
    model.chi2 = Real(-0.2);
    model.reference_separation = Real(20);
    const auto moving_metric = model.make_metric();
    const auto x = lab_event_from_rest_equatorial(
        model, moving_metric, 0, Real(4.2), Real(0.8));
    auto moving_work = moving_metric.build_work(x);
    const auto& moving_hw = moving_work.holes[0];
    kpolaris::Vec4<Real> moving_xhat = moving_hw.rest_x;
    moving_xhat[1] /= moving_hw.hole.mass;
    moving_xhat[2] /= moving_hw.hole.mass;
    moving_xhat[3] /= moving_hw.hole.mass;
    const kpolaris::KerrSchildInMetric<Real> moving_local_metric(
        Real(1), model.chi1);
    const auto moving_local_work =
        moving_local_metric.build_work(moving_xhat);
    Real moving_r, moving_th, moving_cp, moving_sp;
    model.disk.bl_coordinates(moving_xhat, model.chi1, moving_r,
                              moving_th, moving_cp, moving_sp);
    Real moving_local_gcov[4][4];
    moving_local_metric.gcov_matrix(moving_xhat, moving_local_gcov);
    const auto moving_urest =
        model.disk.fluid_four_velocity_from_bl_coords(
            moving_local_metric, moving_xhat, moving_r, moving_th,
            moving_cp, moving_sp, moving_local_gcov);
    const auto moving_brest =
        model.disk.magnetic_unit_four_vector_with_gcov(
            moving_local_metric, moving_xhat, moving_urest,
            moving_local_gcov);
    moving_work.holes[1].amp = Real(0);
    for (int mu = 0; mu < 4; ++mu) {
        moving_work.holes[1].l_global[mu] = Real(0);
    }
    Real moving_gcov[4][4];
    moving_metric.gcov_matrix_from_work(moving_work, moving_gcov);
    kpolaris::Vec4<Real> moving_ucon, moving_bcon;
    if (!model.map_plasma_state_to_full_metric(
            0, moving_metric, moving_work, moving_local_work,
            moving_urest, moving_brest, moving_local_gcov,
            moving_gcov, moving_ucon, moving_bcon)) {
        throw std::runtime_error("moving single-hole paired-frame identity failed");
    }
    const auto expected_u = moving_metric.rest_to_lab_vector(
        moving_hw.hole, moving_urest);
    const auto expected_b = moving_metric.rest_to_lab_vector(
        moving_hw.hole, moving_brest);
    for (int mu = 0; mu < 4; ++mu) {
        require_close("moving single-hole u identity",
                      moving_ucon[mu], expected_u[mu], Real(5e-10));
        require_close("moving single-hole b identity",
                      moving_bcon[mu], expected_b[mu], Real(5e-10));
    }
}

void test_paired_frame_binary_domain() {
    kpolaris::BinaryRIAFRadiationModel<Real> model;
    model.chi1 = Real(0.3);
    model.chi2 = Real(-0.2);
    model.minimum_separation = Real(6);
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real separations[] = {Real(20), Real(10), Real(6.6)};
    for (Real separation : separations) {
        model.reference_separation = separation;
        const auto metric = model.make_metric();
        const auto orbit_state = metric.orbit.state(Real(0));
        for (int hole = 0; hole < 2; ++hole) {
            const Real chi = hole == 0 ? model.chi1 : model.chi2;
            const Real inner = Real(1) + std::sqrt(Real(1) - chi * chi) +
                               model.disk.horizon_buffer + Real(2e-3);
            const Real outer = model.outer_radius_hat(
                hole, orbit_state.separation) * Real(0.999);
            if (!(outer > inner)) {
                throw std::runtime_error("paired-frame scan has empty disk");
            }
            for (int ir = 0; ir < 12; ++ir) {
                const Real f = Real(ir) / Real(11);
                const Real rhat = inner + f * (outer - inner);
                for (int ip = 0; ip < 48; ++ip) {
                    const Real phi = Real(2) * pi * Real(ip) / Real(48);
                    check_paired_frame_point(model, metric, hole, rhat, phi);
                }
            }
        }
    }

    // These points were the strongest failures of the old direct-boost
    // prescription after the Cartesian-KS/BL sign correction.  Keeping them
    // explicit prevents a regression back to false zero-emissivity arcs.
    model.reference_separation = Real(6.6);
    const auto metric = model.make_metric();
    const Real old0 = check_paired_frame_point(
        model, metric, 0, Real(2.054), Real(244.5) * pi / Real(180), true);
    const Real old1 = check_paired_frame_point(
        model, metric, 1, Real(2.080), Real(60.5) * pi / Real(180), true);
    if (!(old0 > Real(1)) || !(old1 > Real(1))) {
        throw std::runtime_error(
            "fixed worst-case points no longer exercise spacelike direct boost");
    }
}

void test_component_sum_and_field_parity() {
    kpolaris::BinaryRIAFRadiationModel<Real> model;
    model.reference_separation = Real(20);
    model.chi1 = Real(0.3);
    model.chi2 = Real(-0.2);
    const auto metric = model.make_metric();
    const auto state = state_in_disk(model, metric, 0);
    const auto c0 = model.component_coefficients(0, metric, state);
    const auto c1 = model.component_coefficients(1, metric, state);
    const auto sum = model.coefficients(metric, state, Real(0.5));
    require_finite_coeffs(c0);
    require_finite_coeffs(c1);
    require_finite_coeffs(sum);
    if (!(c0.jI > Real(0))) throw std::runtime_error("mini-disk test point has no emission");

#define CHECK_SUM(field) require_close("sum " #field, sum.field, c0.field + c1.field)
    CHECK_SUM(jI); CHECK_SUM(jQ); CHECK_SUM(jU); CHECK_SUM(jV);
    CHECK_SUM(aI); CHECK_SUM(aQ); CHECK_SUM(aU); CHECK_SUM(aV);
    CHECK_SUM(rQ); CHECK_SUM(rU); CHECK_SUM(rV);
#undef CHECK_SUM

    auto flipped = model;
    flipped.field_polarity1 = -model.field_polarity1;
    const auto cf = flipped.component_coefficients(0, metric, state);
    require_finite_coeffs(cf);
    const Real even_tol = Real(2e-8);
    require_close("field parity jI", cf.jI, c0.jI, even_tol);
    require_close("field parity jQ", cf.jQ, c0.jQ, even_tol);
    require_close("field parity jU", cf.jU, c0.jU, even_tol);
    require_close("field parity aI", cf.aI, c0.aI, even_tol);
    require_close("field parity aQ", cf.aQ, c0.aQ, even_tol);
    require_close("field parity aU", cf.aU, c0.aU, even_tol);
    require_close("field parity rQ", cf.rQ, c0.rQ, even_tol);
    require_close("field parity rU", cf.rU, c0.rU, even_tol);
    require_close("field parity jV", cf.jV, -c0.jV, even_tol);
    require_close("field parity aV", cf.aV, -c0.aV, even_tol);
    require_close("field parity rV", cf.rV, -c0.rV, even_tol);
}

} // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    try {
        test_bl_cartesian_ks_coordinate_and_vector_convention();
        test_local_plasma_invariants();
        test_paired_frame_single_hole_identity();
        test_paired_frame_binary_domain();
        test_component_sum_and_field_parity();
    } catch (...) {
        Kokkos::finalize();
        throw;
    }
    Kokkos::finalize();
    std::cout << "KPolaris binary RIAF physics tests passed\n";
    return 0;
}
