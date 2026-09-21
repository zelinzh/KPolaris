#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "KPolaris.hpp"
#include "diagnostics/plasma.hpp"
#include "diagnostics/trace_moments.hpp"
#include "kernels/trace_kernel.hpp"

namespace {

using Real = kpolaris::DefaultReal;

struct FlatDomainMetric : kpolaris::MinkowskiMetric<Real> {
    static constexpr kpolaris::CoordinateSystem coordinate_system =
        kpolaris::CoordinateSystem::CartesianKS;
    Real spin = Real(0);

    KPOLARIS_INLINE Real radial_coordinate(
        const kpolaris::Vec4<Real>& x) const {
        return Kokkos::sqrt(x[1] * x[1] + x[2] * x[2] + x[3] * x[3]);
    }

    KPOLARIS_INLINE Real inner_boundary_value(
        const kpolaris::Vec4<Real>& x,
        Real inner_radius) const {
        return radial_coordinate(x) - inner_radius;
    }
};

// Test-only opt-in metric used to force integration below the configured
// affine floor.  Its fixed factor isolates whether every image/trace path
// consults the metric hook; the physical SKS scaling law is tested separately.
struct LocalFloorKerrBoyerLindquistMetric :
    kpolaris::KerrBoyerLindquistMetric<Real> {
    KPOLARIS_INLINE LocalFloorKerrBoyerLindquistMetric(Real mass, Real spin)
        : kpolaris::KerrBoyerLindquistMetric<Real>(mass, spin) {}

    KPOLARIS_INLINE Real effective_min_step(
        const kpolaris::Vec4<Real>&,
        Real configured_min_step) const {
        return kpolaris::abs_val(configured_min_step) / Real(1000);
    }
};

struct EventLimitedFlatMetric : FlatDomainMetric {
    KPOLARIS_INLINE Real inner_boundary_step_limit(
        const kpolaris::Vec4<Real>& x,
        const kpolaris::Vec4<Real>& k,
        Real inner_radius,
        Real proposed_abs_h) const {
        const Real radius = radial_coordinate(x);
        const Real radial_rate = radius > Real(0) ?
            (x[1] * k[1] + x[2] * k[2] + x[3] * k[3]) / radius : Real(0);
        if (radius > inner_radius && radial_rate < Real(0)) {
            // A small overshoot is intentional: it produces a shallow, finite
            // sign bracket rather than asymptoting outside the event surface.
            return kpolaris::min_val(
                proposed_abs_h,
                Real(1.05) * (radius - inner_radius) / (-radial_rate));
        }
        return proposed_abs_h;
    }
};

struct MovingSurfaceFlatMetric : FlatDomainMetric {
    Real surface_velocity = Real(0.25);

    KPOLARIS_INLINE Real inner_boundary_value(
        const kpolaris::Vec4<Real>& x,
        Real inner_radius) const {
        return x[1] - surface_velocity * x[0] - inner_radius;
    }
};

struct TangentSurfaceFlatMetric : FlatDomainMetric {
    KPOLARIS_INLINE Real inner_boundary_value(
        const kpolaris::Vec4<Real>& x,
        Real inner_radius) const {
        return (x[1] - inner_radius) * (x[1] - inner_radius);
    }
};

struct NonlinearConnectionTestMetric : FlatDomainMetric {
    KPOLARIS_INLINE void connection(
        const kpolaris::Vec4<Real>&,
        Real gamma[kpolaris::ndim][kpolaris::ndim][kpolaris::ndim]) const {
        for (int mu = 0; mu < kpolaris::ndim; ++mu) {
            for (int a = 0; a < kpolaris::ndim; ++a) {
                for (int b = 0; b < kpolaris::ndim; ++b) {
                    gamma[mu][a][b] = Real(0);
                }
            }
        }
        gamma[1][1][1] = Real(0.5);
    }
};

struct TerminalInnerEventTestMetric : NonlinearConnectionTestMetric {
    Real surface_velocity = Real(0);

    KPOLARIS_INLINE Real inner_boundary_value(
        const kpolaris::Vec4<Real>& x, Real inner_radius) const {
        return x[1] - surface_velocity * x[0] - inner_radius;
    }

    KPOLARIS_INLINE int inner_boundary_step_is_terminal(
        const kpolaris::Vec4<Real>& x, const kpolaris::Vec4<Real>& k,
        Real inner_radius, Real event_tolerance) const {
        const Real f0 = inner_boundary_value(x, inner_radius);
        const Real rate = k[1] - surface_velocity * k[0];
        if (!(f0 > Real(0)) || !(rate < Real(0))) return 0;
        const Real crossing = f0 / (-rate);
        if (crossing > Real(1.0001) / Real(0.9) * event_tolerance) {
            return 0;
        }
        auto probe = x;
        for (int mu = 0; mu < kpolaris::ndim; ++mu) {
            probe[mu] += Real(2) * crossing * k[mu];
        }
        return inner_boundary_value(probe, inner_radius) <= Real(0);
    }
};

struct FiniteTimeFlatMetric : FlatDomainMetric {
    Real time_min = Real(-0.05);
    Real time_max = Real(0.05);

    KPOLARIS_INLINE int time_domain_valid(
        const kpolaris::Vec4<Real>& x) const {
        return x[0] >= time_min && x[0] <= time_max;
    }

    KPOLARIS_INLINE Real time_domain_boundary_distance(
        const kpolaris::Vec4<Real>& x,
        const kpolaris::Vec4<Real>& k,
        Real proposed_h) const {
        const Real direction = proposed_h < Real(0) ? Real(-1) : Real(1);
        const Real rate = direction * k[0];
        if (rate < -kpolaris::tiny_positive<Real>()) {
            return kpolaris::max_val(
                Real(0), (x[0] - time_min) / (-rate));
        }
        if (rate > kpolaris::tiny_positive<Real>()) {
            return kpolaris::max_val(
                Real(0), (time_max - x[0]) / rate);
        }
        return Real(-1);
    }

    KPOLARIS_INLINE Real time_domain_step_limit(
        const kpolaris::Vec4<Real>& x,
        const kpolaris::Vec4<Real>& k,
        Real proposed_h) const {
        const Real distance =
            time_domain_boundary_distance(x, k, proposed_h);
        return distance >= Real(0) ?
            kpolaris::min_val(kpolaris::abs_val(proposed_h),
                              Real(0.5) * distance) :
            kpolaris::abs_val(proposed_h);
    }

    KPOLARIS_INLINE int time_domain_step_is_terminal(
        const kpolaris::Vec4<Real>& x,
        const kpolaris::Vec4<Real>& k,
        Real proposed_h,
        Real event_tolerance) const {
        const Real distance =
            time_domain_boundary_distance(x, k, proposed_h);
        return distance >= Real(0) &&
               distance <= Real(2.0002) *
                   kpolaris::abs_val(event_tolerance);
    }
};

Real numerical_tolerance(Real requested) {
    return std::max(requested,
                    Real(64) * std::numeric_limits<Real>::epsilon());
}

void require_close(const std::string& name,
                   Real got,
                   Real expected,
                   Real tol = Real(1e-10)) {
    const Real diff = std::abs(got - expected);
    const Real scale = std::max<Real>(Real(1), std::abs(expected));
    const Real effective_tol = tol > Real(0) ? numerical_tolerance(tol) : Real(0);
    if (diff > effective_tol * scale) {
        throw std::runtime_error(name + " got " + std::to_string(got) +
                                 " expected " + std::to_string(expected));
    }
}

#if !KPOLARIS_USE_FLOAT
void require_relative(const std::string& name,
                      Real got,
                      Real expected,
                      Real relative_tolerance) {
    if (!std::isfinite(got) || !std::isfinite(expected)) {
        throw std::runtime_error(name + " is not finite");
    }
    if (expected == Real(0)) {
        if (got != Real(0)) {
            throw std::runtime_error(name + " got a non-zero value");
        }
        return;
    }
    const Real relative_error = std::abs(got - expected) / std::abs(expected);
    if (relative_error > relative_tolerance) {
        throw std::runtime_error(
            name + " relative error " + std::to_string(relative_error) +
            " exceeds " + std::to_string(relative_tolerance));
    }
}
#endif

template<class Metric>
void require_camera_frequency(const Metric& metric,
                              const kpolaris::TransportState<Real>& state,
                              const std::string& name) {
    const auto ucam = kpolaris::normal_observer_velocity(metric, state.x);
    const auto ucov = kpolaris::lower_vector_camera(metric, state.x, ucam);
    Real omega = Real(0);
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        omega += ucov[mu] * state.k[mu];
    }
    require_close(name, -omega, Real(1), Real(1e-10));
}

void require_spatial_direction(const std::string& name,
                               const kpolaris::Vec4<Real>& k,
                               Real x,
                               Real y,
                               Real z,
                               Real tol = Real(1e-12)) {
    const Real n = std::sqrt(k[1] * k[1] + k[2] * k[2] + k[3] * k[3]);
    require_close(name + " x", k[1] / n, x, tol);
    require_close(name + " y", k[2] / n, y, tol);
    require_close(name + " z", k[3] / n, z, tol);
}

void test_pure_emission() {
    kpolaris::Stokes<Real> s;
    kpolaris::TransferCoeffs<Real> c;
    c.jI = Real(2);
    kpolaris::semi_analytic_stokes_step(s, c, Real(0.25));
    require_close("pure emission I", s.I, Real(0.5));
}

void test_absorption_emission() {
    kpolaris::Stokes<Real> s(Real(3), Real(0), Real(0), Real(0));
    kpolaris::TransferCoeffs<Real> c;
    c.jI = Real(4);
    c.aI = Real(2);
    const Real dl = Real(0.5);
    kpolaris::semi_analytic_stokes_step(s, c, dl);

    const Real expected = Real(3) * std::exp(Real(-1)) +
                          Real(2) * (Real(1) - std::exp(Real(-1)));
    require_close("absorption/emission I", s.I, expected);
}

void test_faraday_rotation() {
    kpolaris::Stokes<Real> s(Real(0), Real(1), Real(0), Real(0));
    kpolaris::TransferCoeffs<Real> c;
    c.rV = Real(1.57079632679489661923);
    kpolaris::semi_analytic_stokes_step(s, c, Real(1));

    require_close("faraday Q", s.Q, Real(0), Real(1e-9));
    require_close("faraday U", s.U, Real(1), Real(1e-9));
    require_close("faraday V", s.V, Real(0), Real(1e-9));
}

void test_qu_rotation() {
    kpolaris::Stokes<Real> s(Real(1), Real(1), Real(0), Real(0.25));
    const Real psi = Real(0.52359877559829887308);
    const kpolaris::Stokes<Real> out = kpolaris::rotate_qu(s, psi);

    require_close("rotate I", out.I, Real(1));
    require_close("rotate Q", out.Q, Real(0.5), Real(1e-9));
    require_close("rotate U", out.U, Real(0.86602540378443864676), Real(1e-9));
    require_close("rotate V", out.V, Real(0.25));
}

void test_basis_overlap() {
    const Real psi = Real(0.25);
    kpolaris::BasisOverlap2<Real> overlap;
    overlap.r11 = std::cos(psi);
    overlap.r12 = -std::sin(psi);
    overlap.r21 = std::sin(psi);
    overlap.r22 = std::cos(psi);

    require_close("basis det", overlap.det(), Real(1), Real(1e-9));
    require_close("basis orthogonality", overlap.orthogonality_error(), Real(0),
                  Real(1e-9));

    const kpolaris::Stokes<Real> in(Real(1), Real(1), Real(0), Real(0));
    const kpolaris::Stokes<Real> out =
        kpolaris::transform_to_observer_basis(in, overlap);
    require_close("basis Q", out.Q, std::cos(Real(2) * psi), Real(1e-9));
    require_close("basis U", out.U, std::sin(Real(2) * psi), Real(1e-9));
}

void test_stokes_jones_basis_contract() {
    // Independent electric-field oracle: rotate/reflect complex amplitudes,
    // then derive Stokes, rather than compare two Stokes rotation formulae.
    using Complex = std::complex<Real>;
    auto stokes = [](Complex x, Complex y) {
        return kpolaris::Stokes<Real>(std::norm(x) + std::norm(y),
            std::norm(x) - std::norm(y), Real(2) * std::real(x * std::conj(y)),
            -Real(2) * std::imag(x * std::conj(y)));
    };
    const Complex x(Real(.7), Real(.2)), y(Real(-.3), Real(.4));
    const auto initial = stokes(x, y);
    for (Real angle : {Real(0), Real(.37), Real(1.5707963267948966)}) {
        for (Real parity : {Real(1), Real(-1)}) {
            kpolaris::BasisOverlap2<Real> r;
            r.r11 = std::cos(angle); r.r12 = std::sin(angle);
            r.r21 = -parity * std::sin(angle); r.r22 = parity * std::cos(angle);
            const auto expected = stokes(r.r11*x + r.r12*y, r.r21*x + r.r22*y);
            const auto actual = kpolaris::transform_to_observer_basis(initial, r);
            require_close("Jones rotation I", actual.I, expected.I, Real(1e-12));
            require_close("Jones rotation Q", actual.Q, expected.Q, Real(1e-12));
            require_close("Jones rotation U", actual.U, expected.U, Real(1e-12));
            require_close("Jones reflection V", actual.V, expected.V, Real(1e-12));
        }
    }
    const auto right = stokes(Complex(1,0), Complex(0,1));
    require_close("positive V electric-field rotation", right.V, Real(2));
}

void test_adaptive_error_includes_screen_basis() {
    kpolaris::TransportState<Real> a;
    kpolaris::TransportState<Real> b;
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        a.x[mu] = b.x[mu] = Real(mu + 1);
        a.k[mu] = b.k[mu] = Real(0.1) * Real(mu + 1);
        a.e1[mu] = Real(mu + 2);
        a.e2[mu] = Real(mu + 3);
        b.e1[mu] = -Real(10) * a.e1[mu];
        b.e2[mu] = Real(7) * a.e2[mu];
    }
    if (!(kpolaris::state_error_norm(a, b) > Real(0))) {
        throw std::runtime_error("adaptive error did not include screen basis");
    }
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        b.e1[mu] = a.e1[mu];
        b.e2[mu] = a.e2[mu];
    }
    require_close("adaptive error identical state",
                  kpolaris::state_error_norm(a, b), Real(0));
    b.k[2] += Real(0.25);
    if (!(kpolaris::state_error_norm(a, b) > Real(0))) {
        throw std::runtime_error("adaptive error did not include k");
    }
}

void test_toy_kernel() {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    constexpr int nx = 4;
    constexpr int ny = 3;
    const int npix = nx * ny;

    kpolaris::ToyRaytraceParams<Real> params;
    params.nx = nx;
    params.ny = ny;
    params.nsteps = 20;
    params.length = Real(1);
    params.coeffs.jI = Real(1);
    params.coeffs.jQ = Real(0.1);
    params.coeffs.aI = Real(0);

    Kokkos::View<Real*, ExecSpace> image_i("I", npix);
    Kokkos::View<Real*, ExecSpace> image_q("Q", npix);
    Kokkos::View<Real*, ExecSpace> image_u("U", npix);
    Kokkos::View<Real*, ExecSpace> image_v("V", npix);

    kpolaris::run_toy_raytrace<ExecSpace>(params, image_i, image_q, image_u,
                                       image_v);
    Kokkos::fence();

    auto h_i = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_i);
    auto h_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_q);
    auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_u);
    auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_v);

    for (int p = 0; p < npix; ++p) {
        if (!(h_i(p) > Real(0))) {
            throw std::runtime_error("toy kernel produced non-positive I");
        }
        if (!(h_q(p) > Real(0))) {
            throw std::runtime_error("toy kernel produced non-positive Q");
        }
        require_close("toy U", h_u(p), Real(0), Real(1e-10));
        require_close("toy V", h_v(p), Real(0), Real(1e-10));
    }
}


void test_minkowski_metric_and_rk4() {
    const kpolaris::MinkowskiMetric<Real> metric;
    kpolaris::TransportState<Real> state;
    state.x = kpolaris::Vec4<Real>(Real(0), Real(0), Real(0), Real(1));
    state.k = kpolaris::Vec4<Real>(Real(1), Real(0), Real(0), Real(-1));
    state.e1 = kpolaris::Vec4<Real>(Real(0), Real(1), Real(0), Real(0));
    state.e2 = kpolaris::Vec4<Real>(Real(0), Real(0), Real(1), Real(0));

    const auto initial_errors = kpolaris::frame_errors(metric, state);
    require_close("initial null", initial_errors.k_null, Real(0), Real(1e-12));
    require_close("initial frame error", kpolaris::max_frame_error(initial_errors), Real(0), Real(1e-12));

    const auto out = kpolaris::integrate_fixed_rk4(metric, state, Real(0.01), 100);
    require_close("flat t", out.x[0], Real(1), Real(1e-12));
    require_close("flat z", out.x[3], Real(0), Real(1e-12));
    require_close("flat k0", out.k[0], Real(1), Real(1e-12));
    require_close("flat k3", out.k[3], Real(-1), Real(1e-12));
    require_close("flat frame error", kpolaris::max_frame_error(kpolaris::frame_errors(metric, out)), Real(0), Real(1e-12));
}

void test_rk4_midpoint_state() {
    const kpolaris::MinkowskiMetric<Real> metric;
    kpolaris::TransportState<Real> state;
    state.x = kpolaris::Vec4<Real>(Real(0), Real(0), Real(0), Real(1));
    state.k = kpolaris::Vec4<Real>(Real(1), Real(0), Real(0), Real(-1));
    state.e1 = kpolaris::Vec4<Real>(Real(0), Real(1), Real(0), Real(0));
    state.e2 = kpolaris::Vec4<Real>(Real(0), Real(0), Real(1), Real(0));

    const auto mid = kpolaris::rk4_midpoint_state(metric, state, Real(2));
    require_close("rk4 midpoint t", mid.x[0], Real(1), Real(1e-12));
    require_close("rk4 midpoint z", mid.x[3], Real(0), Real(1e-12));
    require_close("rk4 midpoint k0", mid.k[0], Real(1), Real(1e-12));
    require_close("rk4 midpoint k3", mid.k[3], Real(-1), Real(1e-12));
}

void test_double_pass_toy_kernel() {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    constexpr int nx = 5;
    constexpr int ny = 4;
    const int npix = nx * ny;

    kpolaris::DoublePassToyParams<Real> params;
    params.nx = nx;
    params.ny = ny;
    params.nsteps = 32;
    params.camera_z = Real(2);
    params.inner_z = Real(0.25);
    params.coeffs.jI = Real(0.8);
    params.coeffs.jQ = Real(0.05);
    params.coeffs.aI = Real(0.1);
    params.coeffs.rV = Real(0.2);

    Kokkos::View<Real*, ExecSpace> image_i("I2", npix);
    Kokkos::View<Real*, ExecSpace> image_q("Q2", npix);
    Kokkos::View<Real*, ExecSpace> image_u("U2", npix);
    Kokkos::View<Real*, ExecSpace> image_v("V2", npix);
    Kokkos::View<Real*, ExecSpace> closure("closure", npix);
    Kokkos::View<Real*, ExecSpace> frame_error("frame_error", npix);

    kpolaris::run_double_pass_toy<ExecSpace>(params, image_i, image_q, image_u,
                                          image_v, closure, frame_error);
    Kokkos::fence();

    auto h_i = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_i);
    auto h_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_q);
    auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_u);
    auto h_v = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_v);
    auto h_closure = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), closure);
    auto h_frame_error = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), frame_error);

    for (int p = 0; p < npix; ++p) {
        if (!(h_i(p) > Real(0))) {
            throw std::runtime_error("double-pass toy produced non-positive I");
        }
        if (!(std::abs(h_q(p)) > Real(0))) {
            throw std::runtime_error("double-pass toy produced zero Q");
        }
        if (!(std::abs(h_u(p)) > Real(0))) {
            throw std::runtime_error("double-pass toy produced zero U");
        }
        require_close("double-pass V", h_v(p), Real(0), Real(1e-10));
        require_close("double-pass closure", h_closure(p), Real(0), Real(1e-10));
        require_close("double-pass frame", h_frame_error(p), Real(0), Real(1e-10));
    }
}



void test_kerr_schild_metric_inverse() {
    const kpolaris::KerrSchildInMetric<Real> metric(Real(1), Real(0.7));
    const kpolaris::Vec4<Real> x(Real(0), Real(5.0), Real(1.2), Real(0.8));

    Real gcov[kpolaris::ndim][kpolaris::ndim];
    Real gcon[kpolaris::ndim][kpolaris::ndim];
    metric.gcov_matrix(x, gcov);
    metric.gcon_matrix(x, gcon);

    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        for (int nu = 0; nu < kpolaris::ndim; ++nu) {
            Real sum = Real(0);
            for (int a = 0; a < kpolaris::ndim; ++a) {
                sum += gcov[mu][a] * gcon[a][nu];
            }
            require_close("ks inverse", sum, mu == nu ? Real(1) : Real(0), Real(1e-10));
        }
    }
}

void test_kerr_boyer_lindquist_metric_inverse() {
    const kpolaris::KerrBoyerLindquistMetric<Real> metric(Real(1), Real(0.7));
    const kpolaris::Vec4<Real> x(Real(0), Real(5.0), Real(1.2), Real(0.8));

    Real gcov[kpolaris::ndim][kpolaris::ndim];
    Real gcon[kpolaris::ndim][kpolaris::ndim];
    metric.gcov_matrix(x, gcov);
    metric.gcon_matrix(x, gcon);

    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        for (int nu = 0; nu < kpolaris::ndim; ++nu) {
            Real sum = Real(0);
            for (int a = 0; a < kpolaris::ndim; ++a) {
                sum += gcov[mu][a] * gcon[a][nu];
            }
            require_close("bl inverse", sum, mu == nu ? Real(1) : Real(0), Real(1e-10));
        }
    }

    Real gamma[kpolaris::ndim][kpolaris::ndim][kpolaris::ndim];
    metric.connection(x, gamma);
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        for (int a = 0; a < kpolaris::ndim; ++a) {
            for (int b = 0; b < kpolaris::ndim; ++b) {
                if (!std::isfinite(gamma[mu][a][b])) {
                    throw std::runtime_error("bl connection produced non-finite value");
                }
                require_close("bl connection symmetry", gamma[mu][a][b], gamma[mu][b][a], Real(1e-12));
            }
        }
    }
}

void test_kerr_schild_derivatives() {
    const kpolaris::KerrSchildInMetric<Real> metric(Real(1), Real(0.5));
    const kpolaris::Vec4<Real> x(Real(0), Real(4.0), Real(-1.3), Real(0.9));
    Real dg[kpolaris::ndim][kpolaris::ndim][kpolaris::ndim];
    metric.dgcov(x, dg);

    const Real h = sizeof(Real) <= sizeof(float) ? Real(1e-3) : Real(1e-5);
    const Real derivative_tol =
        sizeof(Real) <= sizeof(float) ? Real(5e-3) : Real(2e-6);
    for (int alpha = 1; alpha < kpolaris::ndim; ++alpha) {
        kpolaris::Vec4<Real> xp = x;
        kpolaris::Vec4<Real> xm = x;
        xp[alpha] += h;
        xm[alpha] -= h;
        for (int mu = 0; mu < kpolaris::ndim; ++mu) {
            for (int nu = 0; nu < kpolaris::ndim; ++nu) {
                const Real fd = (metric.gcov(mu, nu, xp) - metric.gcov(mu, nu, xm)) / (Real(2) * h);
                require_close("ks dgcov", dg[mu][nu][alpha], fd, derivative_tol);
            }
        }
    }
}

void test_fmks_metric_inverse_and_derivatives() {
    const kpolaris::KerrFMKSMetric<Real> metric(Real(1), Real(0.5), std::log(Real(1.21756)),
                                            Real(0.3), Real(0.5), Real(14), Real(0.82), Real(1));
    const kpolaris::Vec4<Real> x(Real(0), std::log(Real(6.0)), Real(0.42), Real(1.1));

    Real gcov[kpolaris::ndim][kpolaris::ndim];
    Real gcon[kpolaris::ndim][kpolaris::ndim];
    metric.gcov_matrix(x, gcov);
    metric.gcon_matrix(x, gcon);
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        for (int nu = 0; nu < kpolaris::ndim; ++nu) {
            Real sum = Real(0);
            for (int a = 0; a < kpolaris::ndim; ++a) sum += gcov[mu][a] * gcon[a][nu];
            require_close("fmks inverse", sum, mu == nu ? Real(1) : Real(0), Real(1e-9));
        }
    }

    Real dg[kpolaris::ndim][kpolaris::ndim][kpolaris::ndim];
    metric.dgcov(x, dg);
    for (int alpha = 1; alpha <= 2; ++alpha) {
        const Real base_h =
            sizeof(Real) <= sizeof(float) ? Real(1e-3) : Real(1e-6);
        const Real derivative_tol =
            sizeof(Real) <= sizeof(float) ? Real(1e-2) : Real(5e-6);
        const Real h = base_h * std::max<Real>(Real(1), std::abs(x[alpha]));
        kpolaris::Vec4<Real> xp = x;
        kpolaris::Vec4<Real> xm = x;
        xp[alpha] += h;
        xm[alpha] -= h;
        for (int mu = 0; mu < kpolaris::ndim; ++mu) {
            for (int nu = 0; nu < kpolaris::ndim; ++nu) {
                const Real fd = (metric.gcov(mu, nu, xp) - metric.gcov(mu, nu, xm)) / (Real(2) * h);
                require_close("fmks dgcov", dg[mu][nu][alpha], fd, derivative_tol);
            }
        }
    }

    Real gamma[kpolaris::ndim][kpolaris::ndim][kpolaris::ndim];
    metric.connection(x, gamma);
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        for (int a = 0; a < kpolaris::ndim; ++a) {
            for (int b = 0; b < kpolaris::ndim; ++b) {
                if (!std::isfinite(gamma[mu][a][b])) {
                    throw std::runtime_error("fmks connection produced non-finite value");
                }
                require_close("fmks gamma symmetry", gamma[mu][a][b], gamma[mu][b][a], Real(1e-11));
            }
        }
    }
}


void test_fmks_specialized_connection_contract() {
    const kpolaris::KerrFMKSMetric<Real> metric(Real(1), Real(0.5), std::log(Real(1.21756)),
                                            Real(0.3), Real(0.5), Real(14), Real(0.82), Real(1));
    const kpolaris::Vec4<Real> x(Real(0.1), std::log(Real(6.0)), Real(0.42), Real(1.1));
    const kpolaris::Vec4<Real> p(Real(0.8), Real(-0.2), Real(0.15), Real(0.35));
    const kpolaris::Vec4<Real> q(Real(1.1), Real(0.05), Real(-0.08), Real(0.2));

    Real gamma[kpolaris::ndim][kpolaris::ndim][kpolaris::ndim];
    metric.connection(x, gamma);
    const auto c = kpolaris::fmks_connection_contract(metric, x, p, q);
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        Real ref = Real(0);
        for (int a = 0; a < kpolaris::ndim; ++a) {
            for (int b = 0; b < kpolaris::ndim; ++b) {
                ref += gamma[mu][a][b] * p[a] * q[b];
            }
        }
        require_close("fmks specialized connection contract", c[mu], ref, Real(2e-11));
    }
}

void test_kerr_schild_connection_symmetry() {
    const kpolaris::KerrSchildInMetric<Real> metric(Real(1), Real(0.8));
    const kpolaris::Vec4<Real> x(Real(0), Real(7.0), Real(0.4), Real(2.0));
    Real gamma[kpolaris::ndim][kpolaris::ndim][kpolaris::ndim];
    metric.connection(x, gamma);
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        for (int a = 0; a < kpolaris::ndim; ++a) {
            for (int b = 0; b < kpolaris::ndim; ++b) {
                require_close("ks gamma symmetry", gamma[mu][a][b], gamma[mu][b][a], Real(1e-12));
            }
        }
    }
}


void test_kerr_schild_specialized_connection_contract() {
    const kpolaris::KerrSchildInMetric<Real> metric(Real(1), Real(0.8));
    const kpolaris::Vec4<Real> x(Real(0.2), Real(7.0), Real(0.4), Real(2.0));
    const kpolaris::Vec4<Real> p(Real(1.1), Real(-0.3), Real(0.2), Real(0.7));
    const kpolaris::Vec4<Real> q(Real(0.9), Real(0.4), Real(-0.5), Real(0.1));

    Real gamma[kpolaris::ndim][kpolaris::ndim][kpolaris::ndim];
    metric.connection(x, gamma);
    const auto c = kpolaris::ks_connection_contract(metric, x, p, q);
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        Real ref = Real(0);
        for (int a = 0; a < kpolaris::ndim; ++a) {
            for (int b = 0; b < kpolaris::ndim; ++b) {
                ref += gamma[mu][a][b] * p[a] * q[b];
            }
        }
        require_close("ks specialized connection contract", c[mu], ref, Real(2e-12));
    }
}

void test_kerr_schild_short_geodesic_null_control() {
    const kpolaris::KerrSchildInMetric<Real> metric(Real(1), Real(0));
    kpolaris::TransportState<Real> state;
    state.x = kpolaris::Vec4<Real>(Real(0), Real(20), Real(0), Real(0));
    state.k = kpolaris::Vec4<Real>(Real(1), Real(-1), Real(0), Real(0));
    state.k = kpolaris::Vec4<Real>(Real(1), Real(-1), Real(0), Real(0));
    state.e1 = kpolaris::Vec4<Real>(Real(0), Real(0), Real(1), Real(0));
    state.e2 = kpolaris::Vec4<Real>(Real(0), Real(0), Real(0), Real(1));

    const Real initial_null = metric.dot(state.x, state.k, state.k);
    require_close("ks initial null", initial_null, Real(0), Real(1e-10));

    const auto out = kpolaris::integrate_fixed_rk4(metric, state, Real(1e-3), 100);
    const Real final_null = metric.dot(out.x, out.k, out.k);
    if (std::abs(final_null) > Real(1e-5)) {
        throw std::runtime_error("ks short geodesic null drift too large: " + std::to_string(final_null));
    }
}



void test_kerr_geodesic_smoke_kernel() {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    constexpr int nrays = 16;
    kpolaris::KerrGeodesicSmokeParams<Real> params;
    params.nrays = nrays;
    params.nsteps = 64;
    params.mass = Real(1);
    params.spin = Real(0);
    params.r0 = Real(20);
    params.impact_span = Real(0.25);
    params.step = Real(0.005);

    Kokkos::View<Real*, ExecSpace> initial_null("ks_initial_null", nrays);
    Kokkos::View<Real*, ExecSpace> final_null("ks_final_null", nrays);
    Kokkos::View<Real*, ExecSpace> initial_r("ks_initial_r", nrays);
    Kokkos::View<Real*, ExecSpace> final_r("ks_final_r", nrays);
    Kokkos::View<Real*, ExecSpace> frame_error("ks_frame_error", nrays);

    kpolaris::run_kerr_geodesic_smoke<ExecSpace>(params, initial_null, final_null,
                                              initial_r, final_r, frame_error);
    Kokkos::fence();

    auto h_initial_null = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), initial_null);
    auto h_final_null = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), final_null);
    auto h_initial_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), initial_r);
    auto h_final_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), final_r);
    auto h_frame_error = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), frame_error);

    for (int ray = 0; ray < nrays; ++ray) {
        require_close("ks smoke initial null", h_initial_null(ray), Real(0), Real(1e-10));
        if (std::abs(h_final_null(ray)) > numerical_tolerance(Real(1e-8))) {
            throw std::runtime_error("ks smoke final null drift too large: " + std::to_string(h_final_null(ray)));
        }
        if (!(h_final_r(ray) < h_initial_r(ray))) {
            throw std::runtime_error("ks smoke ray did not move inward");
        }
        if (std::abs(h_frame_error(ray)) > Real(5e-4)) {
            throw std::runtime_error("ks smoke frame error too large: " + std::to_string(h_frame_error(ray)));
        }
    }
}



void test_camera_initialization() {
    const kpolaris::KerrSchildInMetric<Real> metric(Real(1), Real(0.5));
    kpolaris::CameraParams<Real> cam;
    cam.nx = 3;
    cam.ny = 3;
    cam.radius = Real(30);
    cam.inclination = Real(1.0);
    cam.fov = Real(0.05);

    const int center = 4;
    const auto state = kpolaris::initialize_camera_ray(metric, center, cam);
    require_close("camera initial null", metric.dot(state.x, state.k, state.k), Real(0), Real(1e-10));
    require_camera_frequency(metric, state, "camera frequency");
    const auto err = kpolaris::frame_errors(metric, state);
    require_close("camera e1.k", err.e1_dot_k, Real(0), Real(1e-10));
    require_close("camera e2.k", err.e2_dot_k, Real(0), Real(1e-10));
    require_close("camera e1 norm", err.e1_norm, Real(0), Real(1e-10));
    require_close("camera e2 norm", err.e2_norm, Real(0), Real(1e-10));
    require_close("camera e1.e2", err.e1_dot_e2, Real(0), Real(1e-10));
}

template<class Metric>
void check_affine_direction_equivalence(const Metric& metric,
                                       kpolaris::CameraModel camera_model,
                                       Real& max_difference) {
    kpolaris::CameraParams<Real> cam;
    cam.model = camera_model;
    cam.nx = cam.ny = 3;
    cam.radius = Real(30);
    cam.inclination = Real(1);
    cam.fov = Real(0.1);
    // Compare two parameterizations of the same past light ray, including its
    // coordinate time and transported screen basis. This does not assume a
    // stationary metric or exact reversal of a finite RK step.
    for (int pixel : {0, 4, 8}) {
        const auto camera = kpolaris::initialize_camera_ray(metric, pixel, cam);
        if (!(camera.k[0] > Real(0)))
            throw std::runtime_error("camera photon wavevector must be future directed");
        for (int adaptive : {0, 1}) {
            auto past = camera;
            auto physical = camera;
            past.k = past.k * Real(-1);
            kpolaris::AdaptiveRK4Control<Real> control;
            control.tolerance = Real(1e-9);
            control.min_step = Real(1e-6);
            control.max_step = Real(0.3);
            Real h = Real(0.2);
            for (int n = 0; n < 32; ++n) {
                if (adaptive) {
                    const auto a = kpolaris::adaptive_rk4_step(metric, past, h, control);
                    const auto b = kpolaris::adaptive_rk4_step(metric, physical, -h, control);
                    if (!a.accepted || !b.accepted)
                        throw std::runtime_error("affine-direction test step was not accepted");
                    require_close("opposite accepted affine steps", a.used_h, -b.used_h, Real(1e-12));
                    require_close("opposite next affine steps", a.next_h, -b.next_h, Real(1e-12));
                    require_close("equal affine step errors", a.error, b.error, Real(1e-12));
                    past = a.state;
                    physical = b.state;
                    h = a.next_h;
                } else {
                    past = kpolaris::rk4_step(metric, past, h);
                    physical = kpolaris::rk4_step(metric, physical, -h);
                }
                for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                    const Real deltas[] = {past.x[mu] - physical.x[mu],
                        past.k[mu] + physical.k[mu], past.e1[mu] - physical.e1[mu],
                        past.e2[mu] - physical.e2[mu]};
                    for (Real delta : deltas) {
                        if (!std::isfinite(delta))
                            throw std::runtime_error("nonfinite affine-direction comparison");
                        max_difference = std::max(max_difference, std::abs(delta));
                        require_close("signed affine parameterization", delta, Real(0), Real(1e-12));
                    }
                }
                const auto u = kpolaris::normal_observer_velocity(metric, physical.x);
                if (!(physical.x[0] < camera.x[0] && physical.k[0] > Real(0) &&
                      -metric.dot(physical.x, u, physical.k) > Real(0)))
                    throw std::runtime_error("negative steps must trace a physical photon into the past");
            }
            // The former convention restored k at the endpoint. The new
            // convention retains physical k; both give the same Pass-B state.
            past.k = past.k * Real(-1);
            const auto a = kpolaris::rk4_step(metric, past, Real(0.1));
            const auto b = kpolaris::rk4_step(metric, physical, Real(0.1));
            if (!(a.x[0] > past.x[0]))
                throw std::runtime_error("Pass B must advance toward the camera time");
            require_close("same forward state", kpolaris::state_error_norm(a, b), Real(0), Real(1e-12));
        }
    }
}

void test_affine_direction_equivalence() {
    Real max_difference = Real(0);
    for (auto camera : {kpolaris::CameraModel::Pinhole, kpolaris::CameraModel::ParallelPlane}) {
        check_affine_direction_equivalence(FlatDomainMetric{}, camera, max_difference);
        check_affine_direction_equivalence(kpolaris::KerrSchildInMetric<Real>(1, Real(0.5)),
                                           camera, max_difference);
    }
    check_affine_direction_equivalence(kpolaris::KerrBoyerLindquistMetric<Real>(1, Real(0.5)),
                                       kpolaris::CameraModel::Pinhole, max_difference);
    kpolaris::KerrFMKSMetric<Real> fmks;
    fmks.spin = Real(0.5);
    fmks.poly_norm = Real(0.5) * std::acos(Real(-1)) /
        (Real(1) + Real(1) / (fmks.poly_alpha + Real(1)) / std::pow(fmks.poly_xt, fmks.poly_alpha));
    check_affine_direction_equivalence(fmks, kpolaris::CameraModel::Pinhole, max_difference);
    kpolaris::SuperposedKerrSchildMetric<Real> binary;
    binary.orbit.reference_separation = binary.orbit.minimum_separation = Real(12);
    binary.orbit.inspiral_enabled = 0;
    binary.orbit.orbit_enabled = 1;
    binary.time_origin = Real(3);
    const kpolaris::Vec4<Real> x0(0, 25, 2, 16), x1(1, 25, 2, 16);
    if (!(std::abs(binary.gcov(0, 0, x1) - binary.gcov(0, 0, x0)) > Real(1e-8)))
        throw std::runtime_error("affine-direction fixture must include time-dependent geometry");
    check_affine_direction_equivalence(binary, kpolaris::CameraModel::ParallelPlane, max_difference);
    std::cout << "Affine direction: fixed/adaptive RK4, flat/Kerr/FMKS/time-dependent binary, "
              << "max component difference=" << max_difference << "\n";
}

void test_camera_parallel_plane_alignment() {
    const kpolaris::KerrSchildInMetric<Real> metric(Real(1), Real(0.5));
    kpolaris::CameraParams<Real> cam;
    cam.nx = 3;
    cam.ny = 3;
    cam.radius = Real(30);
    cam.inclination = Real(1.0);
    cam.xspan = Real(6);
    cam.yspan = Real(3);
    cam.model = kpolaris::CameraModel::ParallelPlane;

    const auto center = kpolaris::initialize_camera_ray(metric, 4, cam);
    const auto corner = kpolaris::initialize_camera_ray(metric, 0, cam);
    const Real si = std::sin(cam.inclination);
    const Real ci = std::cos(cam.inclination);

    require_close("parallel center x", center.x[1], cam.radius * si, Real(1e-12));
    require_close("parallel center y", center.x[2], Real(0), Real(1e-12));
    require_close("parallel center z", center.x[3], cam.radius * ci, Real(1e-12));
    require_spatial_direction("parallel center k", center.k, si, Real(0), ci);

    require_spatial_direction("parallel corner k", corner.k, si, Real(0), ci);
    require_close("parallel corner x", corner.x[1], cam.radius * si + Real(2) * ci, Real(1e-12));
    require_close("parallel corner y", corner.x[2], Real(-4), Real(1e-12));
    require_close("parallel corner z", corner.x[3], cam.radius * ci - Real(2) * si, Real(1e-12));

    require_close("parallel initial null", metric.dot(center.x, center.k, center.k), Real(0), Real(1e-10));
    require_close("parallel corner null", metric.dot(corner.x, corner.k, corner.k), Real(0), Real(1e-10));
    require_camera_frequency(metric, center, "parallel center frequency");
    require_camera_frequency(metric, corner, "parallel corner frequency");
}

void test_camera_pinhole_alignment() {
    kpolaris::CameraParams<Real> cam;
    cam.nx = 3;
    cam.ny = 3;
    cam.radius = Real(30);
    cam.inclination = Real(1.0);
    cam.fov = Real(0.6);
    cam.model = kpolaris::CameraModel::Pinhole;

    const auto basis = kpolaris::make_camera_basis(cam);
    const Real si = std::sin(cam.inclination);
    const Real ci = std::cos(cam.inclination);
    require_close("pinhole basis right x", basis.right.x, Real(0), Real(1e-12));
    require_close("pinhole basis right y", basis.right.y, Real(1), Real(1e-12));
    require_close("pinhole basis right z", basis.right.z, Real(0), Real(1e-12));
    require_close("pinhole basis up x", basis.up.x, -ci, Real(1e-12));
    require_close("pinhole basis up z", basis.up.z, si, Real(1e-12));

    const Real spin = Real(0.5);
    const auto spin_basis = kpolaris::make_pinhole_basis(cam, spin);
    require_close("pinhole spin phi0 y", spin_basis.position.y, spin * si, Real(1e-12));
    require_close("pinhole spin up y", spin_basis.up.y, -spin * ci / std::sqrt(cam.radius * cam.radius + spin * spin * ci * ci), Real(1e-12));

    const kpolaris::KerrSchildInMetric<Real> metric(Real(1), spin);
    const auto camera_x = kpolaris::camera_position_phi0(metric, cam.radius, cam.inclination);
    require_close("pinhole Cartesian phi0 x", camera_x[1], spin_basis.position.x, Real(1e-12));
    require_close("pinhole Cartesian phi0 y", camera_x[2], spin_basis.position.y, Real(1e-12));
    require_close("pinhole Cartesian phi0 z", camera_x[3], spin_basis.position.z, Real(1e-12));

    const auto center = kpolaris::pixel_direction(4, cam);
    require_close("pinhole center kx", center.x, -si, Real(1e-12));
    require_close("pinhole center ky", center.y, Real(0), Real(1e-12));
    require_close("pinhole center kz", center.z, -ci, Real(1e-12));

    const auto corner = kpolaris::pixel_direction(0, cam);
    const kpolaris::Vec3<Real> e_phi(Real(0), Real(1), Real(0));
    const kpolaris::Vec3<Real> e_theta(ci, Real(0), -si);
    if (!(kpolaris::dot(corner, e_phi) < Real(0))) {
        throw std::runtime_error("pinhole lower-left ray has wrong horizontal sign");
    }
    if (!(kpolaris::dot(corner, e_theta) > Real(0))) {
        throw std::runtime_error("pinhole lower-left ray has wrong vertical sign");
    }

    // The direction helper must agree with the actual photon in flat space,
    // including off-axis pixels; a reflected test expectation cannot hide it.
    const FlatDomainMetric flat;
    for (int pixel = 0; pixel < 9; ++pixel) {
        const auto ray = kpolaris::initialize_camera_ray(flat, pixel, cam);
        const auto direction = kpolaris::pixel_direction(pixel, cam);
        require_close("flat pinhole direction x", direction.x, -ray.k[1], Real(1e-12));
        require_close("flat pinhole direction y", direction.y, -ray.k[2], Real(1e-12));
        require_close("flat pinhole direction z", direction.z, -ray.k[3], Real(1e-12));
        const auto observer = kpolaris::normal_observer_velocity(flat, ray.x);
        const auto n = ray.k - observer;
        if (!(kpolaris::tetrad_orientation(observer, ray.e1, ray.e2, n) > 0))
            throw std::runtime_error("camera screen must be right handed about physical k");
    }
}

kpolaris::Vec4<Real> bl_vector_to_spherical_ks(
    const kpolaris::KerrBoyerLindquistMetric<Real>& metric,
    const kpolaris::Vec4<Real>& X,
    const kpolaris::Vec4<Real>& v) {
    const Real r = X[1];
    const Real delta = r * r - Real(2) * metric.mass * r +
                       metric.spin * metric.spin;
    return kpolaris::Vec4<Real>(
        v[0] + Real(2) * metric.mass * r * v[1] / delta,
        v[1],
        v[2],
        v[3] + metric.spin * v[1] / delta);
}

void test_pinhole_camera_bl_spherical_ks_equivalence() {
    const Real spin = Real(0.9375);
    const kpolaris::KerrBoyerLindquistMetric<Real> bl(Real(1), spin);
    const kpolaris::KerrSchildSphericalMetric<Real> ks(Real(1), spin);
    kpolaris::CameraParams<Real> cam;
    cam.nx = 3;
    cam.ny = 3;
    cam.radius = Real(1000);
    cam.inclination = Real(85) *
        Real(3.141592653589793238462643383279502884) / Real(180);
    cam.fov = Real(0.0391);
    cam.model = kpolaris::CameraModel::Pinhole;

    const auto xbl = kpolaris::camera_position_phi0(
        bl, cam.radius, cam.inclination);
    const auto center_cov = kpolaris::ipole_centering_covector(bl, xbl);
    const Real delta = cam.radius * cam.radius -
        Real(2) * bl.mass * cam.radius + spin * spin;
    require_close("BL camera native centering covector", center_cov[1],
                  Real(1) / cam.radius +
                      Real(2) * bl.mass * cam.radius / delta,
                  Real(1e-14));

    for (int pixel = 0; pixel < cam.nx * cam.ny; ++pixel) {
        const auto sb = kpolaris::initialize_camera_ray(bl, pixel, cam);
        const auto sk = kpolaris::initialize_camera_ray(ks, pixel, cam);
        const auto kb = bl_vector_to_spherical_ks(bl, sb.x, sb.k);
        const auto e1b = bl_vector_to_spherical_ks(bl, sb.x, sb.e1);
        const auto e2b = bl_vector_to_spherical_ks(bl, sb.x, sb.e2);
        for (int mu = 0; mu < kpolaris::ndim; ++mu) {
            require_close("BL/KS camera k", kb[mu], sk.k[mu], Real(1e-10));
            require_close("BL/KS camera e1", e1b[mu], sk.e1[mu], Real(1e-10));
            require_close("BL/KS camera e2", e2b[mu], sk.e2[mu], Real(1e-10));
        }
    }
}


kpolaris::Vec4<Real> fmks_vector_to_cartesian_phi(const kpolaris::KerrFMKSMetric<Real>& metric,
                                                const kpolaris::Vec4<Real>& X,
                                                const kpolaris::Vec4<Real>& v) {
    const Real x1 = X[1];
    const Real x2 = X[2];
    const Real phi = X[3];
    const Real r = std::exp(x1);
    const Real th = metric.theta_from_x2(x1, x2);
    const Real cp = std::cos(phi);
    const Real sp = std::sin(phi);
    const Real sinth = std::sin(th);
    const Real costh = std::cos(th);
    const Real vr = r * v[1];
    const Real vth = metric.dtheta_dx1(x1, x2) * v[1] +
                     metric.dtheta_dx2(x1, x2) * v[2];
    const Real vphi = v[3];
    const Real dx_dr = cp * sinth;
    const Real dx_dth = r * cp * costh - metric.spin * sp * costh;
    const Real dx_dphi = -r * sp * sinth - metric.spin * cp * sinth;
    const Real dy_dr = sp * sinth;
    const Real dy_dth = r * sp * costh + metric.spin * cp * costh;
    const Real dy_dphi = r * cp * sinth - metric.spin * sp * sinth;
    const Real dz_dr = costh;
    const Real dz_dth = -r * sinth;
    return kpolaris::Vec4<Real>(v[0],
                             dx_dr * vr + dx_dth * vth + dx_dphi * vphi,
                             dy_dr * vr + dy_dth * vth + dy_dphi * vphi,
                             dz_dr * vr + dz_dth * vth);
}

kpolaris::Vec4<Real> fmks_position_to_cartesian(const kpolaris::KerrFMKSMetric<Real>& metric,
                                             const kpolaris::Vec4<Real>& X) {
    const Real x1 = X[1];
    const Real x2 = X[2];
    const Real phi = X[3];
    const Real r = std::exp(x1);
    const Real th = metric.theta_from_x2(x1, x2);
    const Real cp = std::cos(phi);
    const Real sp = std::sin(phi);
    const Real sinth = std::sin(th);
    return kpolaris::Vec4<Real>(X[0],
                             (r * cp - metric.spin * sp) * sinth,
                             (r * sp + metric.spin * cp) * sinth,
                             r * std::cos(th));
}

void test_pinhole_camera_fmks_cartesian_equivalence() {
    const Real spin = Real(0.5);
    const Real startx1 = std::log(Real(1.217564));
    const Real hslope = Real(0.3);
    const Real mks_smooth = Real(0.5);
    const Real poly_alpha = Real(14);
    const Real poly_xt = Real(0.82);
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real poly_norm = Real(0.5) * pi /
        (Real(1) + Real(1) / (poly_alpha + Real(1)) /
         std::pow(poly_xt, poly_alpha));
    const kpolaris::KerrFMKSMetric<Real> fmks(Real(1), spin, startx1, hslope,
                                           mks_smooth, poly_alpha, poly_xt,
                                           poly_norm);
    const kpolaris::KerrSchildInMetric<Real> cart(Real(1), spin);
    kpolaris::CameraParams<Real> cam;
    cam.nx = 3;
    cam.ny = 3;
    cam.radius = Real(1000);
    cam.inclination = Real(0.3);
    cam.fov = Real(0.04);
    cam.model = kpolaris::CameraModel::Pinhole;

    for (int pixel = 0; pixel < cam.nx * cam.ny; ++pixel) {
        const auto sf = kpolaris::initialize_camera_ray(fmks, pixel, cam);
        const auto sc = kpolaris::initialize_camera_ray(cart, pixel, cam);
        const auto xf = fmks_position_to_cartesian(fmks, sf.x);
        const auto kf = fmks_vector_to_cartesian_phi(fmks, sf.x, sf.k);
        const auto e1f = fmks_vector_to_cartesian_phi(fmks, sf.x, sf.e1);
        const auto e2f = fmks_vector_to_cartesian_phi(fmks, sf.x, sf.e2);
        for (int mu = 0; mu < kpolaris::ndim; ++mu) {
            require_close("fmks/cart camera x", xf[mu], sc.x[mu], Real(1e-10));
            require_close("fmks/cart camera k", kf[mu], sc.k[mu], Real(1e-8));
            require_close("fmks/cart camera e1", e1f[mu], sc.e1[mu], Real(1e-8));
            require_close("fmks/cart camera e2", e2f[mu], sc.e2[mu], Real(1e-8));
        }
    }
}



void test_fmks_cartesian_metric_equivalence() {
    const Real spin = Real(0.5);
    const Real startx1 = std::log(Real(1.217564));
    const Real hslope = Real(0.3);
    const Real mks_smooth = Real(0.5);
    const Real poly_alpha = Real(14);
    const Real poly_xt = Real(0.82);
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real poly_norm = Real(0.5) * pi /
        (Real(1) + Real(1) / (poly_alpha + Real(1)) /
         std::pow(poly_xt, poly_alpha));
    const kpolaris::KerrFMKSMetric<Real> fmks(Real(1), spin, startx1, hslope,
                                           mks_smooth, poly_alpha, poly_xt,
                                           poly_norm);
    const kpolaris::KerrSchildInMetric<Real> cart(Real(1), spin);
    const kpolaris::Vec4<Real> X(Real(0), Real(2.1), Real(0.37), Real(1.2));
    const kpolaris::Vec4<Real> a(Real(0.7), Real(0.11), Real(-0.23), Real(0.031));
    const kpolaris::Vec4<Real> b(Real(-0.2), Real(0.04), Real(0.19), Real(-0.017));
    const auto Xc = fmks_position_to_cartesian(fmks, X);
    const auto ac = fmks_vector_to_cartesian_phi(fmks, X, a);
    const auto bc = fmks_vector_to_cartesian_phi(fmks, X, b);
    require_close("fmks/cart metric dot", fmks.dot(X, a, b), cart.dot(Xc, ac, bc), Real(1e-10));
}



kpolaris::Vec4<Real> fmks_component_derivative_to_cartesian(
    const kpolaris::KerrFMKSMetric<Real>& metric,
    const kpolaris::Vec4<Real>& X,
    const kpolaris::Vec4<Real>& k,
    const kpolaris::Vec4<Real>& v,
    const kpolaris::Vec4<Real>& dv) {
    kpolaris::Vec4<Real> out = fmks_vector_to_cartesian_phi(metric, X, dv);
    const Real eps =
        sizeof(Real) <= sizeof(float) ? Real(1e-3) : Real(1e-6);
    kpolaris::Vec4<Real> Xp = X;
    kpolaris::Vec4<Real> Xm = X;
    for (int mu = 1; mu < kpolaris::ndim; ++mu) {
        Xp[mu] += eps * k[mu];
        Xm[mu] -= eps * k[mu];
    }
    const auto vp = fmks_vector_to_cartesian_phi(metric, Xp, v);
    const auto vm = fmks_vector_to_cartesian_phi(metric, Xm, v);
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        out[mu] += (vp[mu] - vm[mu]) / (Real(2) * eps);
    }
    return out;
}

void test_fmks_cartesian_transport_rhs_equivalence() {
    const Real spin = Real(0.5);
    const Real startx1 = std::log(Real(1.217564));
    const Real hslope = Real(0.3);
    const Real mks_smooth = Real(0.5);
    const Real poly_alpha = Real(14);
    const Real poly_xt = Real(0.82);
    const Real pi = Real(3.141592653589793238462643383279502884);
    const Real poly_norm = Real(0.5) * pi /
        (Real(1) + Real(1) / (poly_alpha + Real(1)) /
         std::pow(poly_xt, poly_alpha));
    const kpolaris::KerrFMKSMetric<Real> fmks(Real(1), spin, startx1, hslope,
                                           mks_smooth, poly_alpha, poly_xt,
                                           poly_norm);
    const kpolaris::KerrSchildInMetric<Real> cart(Real(1), spin);
    kpolaris::TransportState<Real> sf;
    sf.x = kpolaris::Vec4<Real>(Real(0), Real(2.1), Real(0.37), Real(1.2));
    sf.k = kpolaris::Vec4<Real>(Real(0.9), Real(-0.04), Real(0.02), Real(0.003));
    sf.e1 = kpolaris::Vec4<Real>(Real(0.1), Real(0.01), Real(0.2), Real(-0.004));
    sf.e2 = kpolaris::Vec4<Real>(Real(-0.03), Real(0.02), Real(-0.01), Real(0.03));
    kpolaris::TransportState<Real> sc;
    sc.x = fmks_position_to_cartesian(fmks, sf.x);
    sc.k = fmks_vector_to_cartesian_phi(fmks, sf.x, sf.k);
    sc.e1 = fmks_vector_to_cartesian_phi(fmks, sf.x, sf.e1);
    sc.e2 = fmks_vector_to_cartesian_phi(fmks, sf.x, sf.e2);
    const auto rf = kpolaris::transport_rhs(fmks, sf);
    const auto rc = kpolaris::transport_rhs(cart, sc);
    const auto dxf = fmks_vector_to_cartesian_phi(fmks, sf.x, rf.x);
    const auto dkf = fmks_component_derivative_to_cartesian(fmks, sf.x, sf.k, sf.k, rf.k);
    const auto de1f = fmks_component_derivative_to_cartesian(fmks, sf.x, sf.k, sf.e1, rf.e1);
    const auto de2f = fmks_component_derivative_to_cartesian(fmks, sf.x, sf.k, sf.e2, rf.e2);
    const Real derivative_tol =
        sizeof(Real) <= sizeof(float) ? Real(5e-3) : Real(1e-6);
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        require_close("fmks/cart rhs x", dxf[mu], rc.x[mu], Real(1e-10));
        require_close("fmks/cart rhs k", dkf[mu], rc.k[mu], derivative_tol);
        require_close("fmks/cart rhs e1", de1f[mu], rc.e1[mu], derivative_tol);
        require_close("fmks/cart rhs e2", de2f[mu], rc.e2[mu], derivative_tol);
    }
}

void test_grmhd_native_cartesian_vector_transform_metric_invariance() {
    kpolaris::GRMHDGridRadiationModel<Real> model;
    model.spin = Real(0.5);
    model.startx1 = std::log(Real(1.217564));
    model.hslope = Real(0.3);
    model.mks_smooth = Real(0.5);
    model.poly_alpha = Real(14);
    model.poly_xt = Real(0.82);
    const Real pi = Real(3.141592653589793238462643383279502884);
    model.poly_norm = Real(0.5) * pi /
        (Real(1) + Real(1) / (model.poly_alpha + Real(1)) /
         std::pow(model.poly_xt, model.poly_alpha));
    const kpolaris::KerrSchildInMetric<Real> cart(Real(1), model.spin);

    const Real x1 = Real(2.1);
    const Real x2 = Real(0.37);
    const Real phi = Real(1.2);
    const Real r = std::exp(x1);
    const Real th = model.fmks_theta_from_x2(x1, x2);
    const Real cp = std::cos(phi);
    const Real sp = std::sin(phi);
    const Real sinth = std::sin(th);
    const kpolaris::Vec4<Real> Xcart(Real(0),
                                  (r * cp - model.spin * sp) * sinth,
                                  (r * sp + model.spin * cp) * sinth,
                                  r * std::cos(th));
    Real gcov_nat[kpolaris::ndim][kpolaris::ndim];
    Real gcon_nat[kpolaris::ndim][kpolaris::ndim];
    model.native_metric(x1, x2, r, th, gcov_nat, gcon_nat);
    const kpolaris::Vec4<Real> a_nat(Real(0.7), Real(0.11), Real(-0.23), Real(0.031));
    const kpolaris::Vec4<Real> b_nat(Real(-0.2), Real(0.04), Real(0.19), Real(-0.017));
    const auto a_cart = model.native_to_cartesian_contravariant(x1, x2, r, th, cp, sp, a_nat);
    const auto b_cart = model.native_to_cartesian_contravariant(x1, x2, r, th, cp, sp, b_nat);
    Real native_dot = Real(0);
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        for (int nu = 0; nu < kpolaris::ndim; ++nu) {
            native_dot += gcov_nat[mu][nu] * a_nat[mu] * b_nat[nu];
        }
    }
    const Real cart_dot = cart.dot(Xcart, a_cart, b_cart);
    require_close("native/cart vector dot", cart_dot, native_dot, Real(1e-10));
}


void test_grmhd_fluid_state_fmks_cartesian_equivalence() {
    kpolaris::GRMHDGridRadiationModel<Real> model;
    model.n1 = 5;
    model.n2 = 6;
    model.n3 = 8;
    model.spin = Real(0.5);
    model.gam = Real(4) / Real(3);
    model.startx1 = Real(0);
    model.startx2 = Real(0);
    model.startx3 = Real(0);
    model.dx1 = Real(1);
    model.dx2 = Real(1) / Real(model.n2);
    model.dx3 = Real(6.283185307179586476925286766559005768) / Real(model.n3);
    model.r_in = Real(1);
    model.r_out = Real(1000);
    model.hslope = Real(0.3);
    model.mks_smooth = Real(0.5);
    model.poly_alpha = Real(14);
    model.poly_xt = Real(0.82);
    const Real pi = Real(3.141592653589793238462643383279502884);
    model.poly_norm = Real(0.5) * pi /
        (Real(1) + Real(1) / (model.poly_alpha + Real(1)) /
         std::pow(model.poly_xt, model.poly_alpha));
    model.has_derived_scalars = 0;
    const int ncell = model.n1 * model.n2 * model.n3;
    model.prims = Kokkos::View<Real*>("test_grmhd_prims", static_cast<size_t>(8) * ncell);
    auto h = Kokkos::create_mirror_view(model.prims);
    for (int i = 0; i < model.n1; ++i) {
        for (int j = 0; j < model.n2; ++j) {
            for (int k = 0; k < model.n3; ++k) {
                h(model.prim_index(0, i, j, k)) = Real(1.2);
                h(model.prim_index(1, i, j, k)) = Real(0.08);
                h(model.prim_index(2, i, j, k)) = Real(0.03);
                h(model.prim_index(3, i, j, k)) = Real(-0.015);
                h(model.prim_index(4, i, j, k)) = Real(0.02);
                h(model.prim_index(5, i, j, k)) = Real(0.04);
                h(model.prim_index(6, i, j, k)) = Real(-0.025);
                h(model.prim_index(7, i, j, k)) = Real(0.018);
            }
        }
    }
    Kokkos::deep_copy(model.prims, h);

    const kpolaris::KerrFMKSMetric<Real> fmks(Real(1), model.spin, model.startx1,
                                           model.hslope, model.mks_smooth,
                                           model.poly_alpha, model.poly_xt,
                                           model.poly_norm);
    const kpolaris::KerrSchildInMetric<Real> cart(Real(1), model.spin);
    const Real x1 = Real(2.1);
    const Real x2 = Real(0.37);
    const Real phi = Real(1.2);
    const Real r = std::exp(x1);
    const Real th = model.fmks_theta_from_x2(x1, x2);
    const Real cp = std::cos(phi);
    const Real sp = std::sin(phi);
    const Real sinth = std::sin(th);

    kpolaris::TransportState<Real> sf;
    sf.x = kpolaris::Vec4<Real>(Real(0), x1, x2, phi);
    sf.k = kpolaris::Vec4<Real>(Real(1), Real(-0.2), Real(0.03), Real(0.01));
    sf.e1 = kpolaris::Vec4<Real>(Real(0), Real(0.01), Real(0.7), Real(0.02));
    sf.e2 = kpolaris::Vec4<Real>(Real(0), Real(0.03), Real(0.02), Real(0.5));
    kpolaris::TransportState<Real> sc;
    sc.x = kpolaris::Vec4<Real>(Real(0),
                             (r * cp - model.spin * sp) * sinth,
                             (r * sp + model.spin * cp) * sinth,
                             r * std::cos(th));
    sc.k = model.native_to_cartesian_contravariant(x1, x2, r, th, cp, sp, sf.k);
    sc.e1 = model.native_to_cartesian_contravariant(x1, x2, r, th, cp, sp, sf.e1);
    sc.e2 = model.native_to_cartesian_contravariant(x1, x2, r, th, cp, sp, sf.e2);

    // fluid_state dereferences the primitive View. Evaluate it in the
    // configured execution space so CUDA tests never read a device View from
    // host while the Serial build continues to exercise the same code path.
    Kokkos::View<int*> fluid_ok("test_grmhd_fluid_ok", 2);
    Kokkos::View<Real*> candidates("test_grmhd_fluid_candidates", 26);
    Kokkos::View<Real*> references("test_grmhd_fluid_references", 26);
    const auto device_model = model;
    Kokkos::parallel_for(
        "test_grmhd_fluid_state_fmks_cartesian_equivalence", 1,
        KOKKOS_LAMBDA(const int) {
            Real rho_f, uu_f, ne_f, thetae_f, b_f, sigma_f, beta_f;
            Real rho_c, uu_c, ne_c, thetae_c, b_c, sigma_c, beta_c;
            kpolaris::Vec4<Real> u_f, bcon_f, u_c, bcon_c;
            fluid_ok(0) = device_model.fluid_state(
                fmks, sf, rho_f, uu_f, u_f, bcon_f, ne_f, thetae_f,
                b_f, sigma_f, beta_f) ? 1 : 0;
            fluid_ok(1) = device_model.fluid_state(
                cart, sc, rho_c, uu_c, u_c, bcon_c, ne_c, thetae_c,
                b_c, sigma_c, beta_c) ? 1 : 0;
            if (fluid_ok(0) == 0 || fluid_ok(1) == 0) {
                return;
            }

            int index = 0;
            candidates(index) = rho_c; references(index++) = rho_f;
            candidates(index) = uu_c; references(index++) = uu_f;
            candidates(index) = ne_c; references(index++) = ne_f;
            candidates(index) = thetae_c; references(index++) = thetae_f;
            candidates(index) = b_c; references(index++) = b_f;
            candidates(index) = sigma_c; references(index++) = sigma_f;
            candidates(index) = beta_c; references(index++) = beta_f;

            const auto u_f_cart =
                device_model.native_to_cartesian_contravariant(
                    x1, x2, r, th, cp, sp, u_f);
            const auto b_f_cart =
                device_model.native_to_cartesian_contravariant(
                    x1, x2, r, th, cp, sp, bcon_f);
            for (int mu = 0; mu < kpolaris::ndim; ++mu) {
                candidates(index) = u_c[mu]; references(index++) = u_f_cart[mu];
                candidates(index) = bcon_c[mu]; references(index++) = b_f_cart[mu];
            }

            const auto cf = device_model.coefficients(fmks, sf, Real(0.5));
            const auto cc = device_model.coefficients(cart, sc, Real(0.5));
            candidates(index) = cc.jI; references(index++) = cf.jI;
            candidates(index) = cc.jQ; references(index++) = cf.jQ;
            candidates(index) = cc.jU; references(index++) = cf.jU;
            candidates(index) = cc.jV; references(index++) = cf.jV;
            candidates(index) = cc.aI; references(index++) = cf.aI;
            candidates(index) = cc.aQ; references(index++) = cf.aQ;
            candidates(index) = cc.aU; references(index++) = cf.aU;
            candidates(index) = cc.aV; references(index++) = cf.aV;
            candidates(index) = cc.rQ; references(index++) = cf.rQ;
            candidates(index) = cc.rU; references(index++) = cf.rU;
            candidates(index) = cc.rV; references(index++) = cf.rV;
        });
    Kokkos::fence();

    const auto h_ok = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), fluid_ok);
    const auto h_candidates = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), candidates);
    const auto h_references = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), references);
    if (h_ok(0) == 0) {
        throw std::runtime_error("FMKS fluid_state failed");
    }
    if (h_ok(1) == 0) {
        throw std::runtime_error("Cartesian fluid_state failed");
    }
    const char* scalar_labels[] = {
        "fluid rho", "fluid uu", "fluid ne", "fluid thetae",
        "fluid B", "fluid sigma", "fluid beta"};
    for (int index = 0; index < 7; ++index) {
        require_close(scalar_labels[index], h_candidates(index),
                      h_references(index), Real(1e-12));
    }
    const char* vector_and_coefficient_labels[] = {
        "fluid u cart 0", "fluid b cart 0", "fluid u cart 1",
        "fluid b cart 1", "fluid u cart 2", "fluid b cart 2",
        "fluid u cart 3", "fluid b cart 3", "coeff jI", "coeff jQ",
        "coeff jU", "coeff jV", "coeff aI", "coeff aQ", "coeff aU",
        "coeff aV", "coeff rQ", "coeff rU", "coeff rV"};
    for (int index = 7; index < 26; ++index) {
        require_close(vector_and_coefficient_labels[index - 7],
                      h_candidates(index), h_references(index), Real(1e-10));
    }
}

void test_adaptive_rk4_flat_step() {
    const kpolaris::MinkowskiMetric<Real> metric;
    kpolaris::TransportState<Real> state;
    state.x = kpolaris::Vec4<Real>(Real(0), Real(0), Real(0), Real(1));
    state.k = kpolaris::Vec4<Real>(Real(1), Real(0), Real(0), Real(-1));
    state.e1 = kpolaris::Vec4<Real>(Real(0), Real(1), Real(0), Real(0));
    state.e2 = kpolaris::Vec4<Real>(Real(0), Real(0), Real(1), Real(0));

    kpolaris::AdaptiveRK4Control<Real> control;
    control.tolerance = Real(1e-12);
    control.min_step = Real(1e-4);
    control.max_step = Real(2);
    const auto step = kpolaris::adaptive_rk4_step(metric, state, Real(1), control);
    require_close("adaptive flat t", step.state.x[0], Real(1), Real(1e-12));
    require_close("adaptive flat z", step.state.x[3], Real(0), Real(1e-12));
    if (!(step.accepted && step.used_h > Real(0) && step.next_h > Real(0))) {
        throw std::runtime_error("adaptive RK4 flat step did not accept positive step");
    }
}

void test_metric_effective_min_step_default_semantics() {
    const kpolaris::MinkowskiMetric<Real> metric;
    const Real configured_min_step = Real(1e-5);
    const kpolaris::Vec4<Real> large_tangent(
        Real(5000), Real(-1200), Real(800), Real(-3500));

    const Real effective = kpolaris::metric_effective_min_step(
        metric, large_tangent, configured_min_step);
    if (effective != configured_min_step) {
        throw std::runtime_error(
            "ordinary metric changed the configured adaptive minimum step");
    }

    const auto rescaled_tangent = large_tangent * Real(7);
    const Real rescaled_effective = kpolaris::metric_effective_min_step(
        metric, rescaled_tangent, configured_min_step);
    if (rescaled_effective != configured_min_step) {
        throw std::runtime_error(
            "ordinary metric acquired affine-dependent minimum-step semantics");
    }
}

void test_metric_inner_surface_event_handling() {
    EventLimitedFlatMetric metric;
    kpolaris::TransportState<Real> state;
    state.x = kpolaris::Vec4<Real>(Real(0), Real(3), Real(0), Real(0));
    state.k = kpolaris::Vec4<Real>(Real(1), Real(-1), Real(0), Real(0));
    state.e1 = kpolaris::Vec4<Real>(Real(0), Real(0), Real(1), Real(0));
    state.e2 = kpolaris::Vec4<Real>(Real(0), Real(0), Real(0), Real(1));

    const Real limited_h = kpolaris::metric_limit_inner_surface_step(
        metric, state.x, state.k, Real(2), Real(4));
    require_close("metric event step limiter", limited_h, Real(1.05), Real(1e-12));

    const auto crossed = kpolaris::metric_surface_path_state(
        metric, state, limited_h, 1);
    if (!(metric.inner_boundary_value(crossed.x, Real(2)) < Real(0))) {
        throw std::runtime_error("event-limited trial did not make a shallow bracket");
    }
    const auto event = kpolaris::locate_metric_surface_crossing(
        metric, state, crossed, limited_h, Real(2), 1, 1);
    if (!(event.surface_value >= Real(0))) {
        throw std::runtime_error("inner event locator returned an excised state");
    }
    require_close("inner event radius", metric.radial_coordinate(event.state.x),
                  Real(2), Real(2e-4));
    require_close("inner event affine step", event.used_h, Real(1), Real(2e-4));

    // A future-directed outgoing photon reaches the same surface when traced
    // with a negative affine step. Check the limiter and refined endpoint.
    auto physical = state;
    physical.k = kpolaris::Vec4<Real>(Real(1), Real(1), Real(0), Real(0));
    const Real backward_h = kpolaris::metric_limit_inner_surface_step(
        metric, physical.x, physical.k, Real(2), Real(-4));
    require_close("negative affine event step limiter", backward_h, -limited_h);
    const auto backward_end = kpolaris::metric_surface_path_state(
        metric, physical, backward_h, 1);
    const auto backward_event = kpolaris::locate_metric_surface_crossing(
        metric, physical, backward_end, backward_h, Real(2), 1, 1);
    require_close("negative affine inner event radius", backward_event.state.x[1],
                  event.state.x[1]);
    require_close("negative affine inner event time", backward_event.state.x[0],
                  -event.state.x[0]);
    require_close("negative affine inner event step", backward_event.used_h,
                  -event.used_h);
    require_close("outgoing photon is not capture limited",
        kpolaris::metric_limit_inner_surface_step(
            metric, physical.x, physical.k, Real(2), Real(4)), Real(4));

    // A moving surface tests the time component as well as the spatial sign.
    TerminalInnerEventTestMetric terminal;
    terminal.surface_velocity = Real(0.25);
    if (!kpolaris::metric_inner_surface_step_is_terminal(
            terminal, physical.x, physical.k, Real(2.5), Real(-1), Real(0.7)) ||
        kpolaris::metric_inner_surface_step_is_terminal(
            terminal, physical.x, physical.k, Real(2.5), Real(1), Real(0.7)) ||
        kpolaris::metric_inner_surface_step_is_terminal(
            terminal, physical.x, physical.k, Real(2.5), Real(-1), Real(0.4))) {
        throw std::runtime_error("signed affine moving capture event is inconsistent");
    }

    const kpolaris::MinkowskiMetric<Real> no_hook_metric;
    const Real unchanged_h = kpolaris::metric_limit_inner_surface_step(
        no_hook_metric, state.x, state.k, Real(2), Real(0.75));
    require_close("metric event no-hook legacy step", unchanged_h, Real(0.75));

    MovingSurfaceFlatMetric moving_metric;
    const auto moving_end = kpolaris::metric_surface_path_state(
        moving_metric, state, Real(1), 1);
    const auto moving_event = kpolaris::locate_metric_surface_crossing(
        moving_metric, state, moving_end, Real(1), Real(2), 1, 1);
    if (!(moving_event.surface_value >= Real(0))) {
        throw std::runtime_error("moving event locator returned an excised state");
    }
    require_close("moving event affine step", moving_event.used_h, Real(0.8),
                  Real(2e-4));

    TangentSurfaceFlatMetric tangent_metric;
    const auto tangent_end = kpolaris::metric_surface_path_state(
        tangent_metric, state, Real(2), 1);
    const auto tangent_event = kpolaris::locate_metric_surface_crossing(
        tangent_metric, state, tangent_end, Real(2), Real(2), 1, 1);
    require_close("tangent event no false crossing", tangent_event.used_h, Real(0));
    require_close("tangent event keeps accepted exterior state",
                  tangent_event.state.x[1], state.x[1]);
}

void test_adaptive_step_underflow_reason() {
    NonlinearConnectionTestMetric metric;
    kpolaris::PassAParams<Real> params;
    params.camera.model = kpolaris::CameraModel::ParallelPlane;
    params.camera.nx = 1;
    params.camera.ny = 1;
    params.camera.radius = Real(10);
    params.camera.inclination =
        Real(0.5) * Real(3.141592653589793238462643383279502884);
    params.inner_radius = Real(0.1);
    params.outer_radius = Real(20);
    params.step = Real(1);
    params.min_step = Real(1);
    params.max_step = Real(1);
    params.adaptive_tolerance = Real(0);
    params.adaptive = 1;
    params.max_steps = 4;

    const auto result = kpolaris::trace_pass_a_pixel_metric(0, params, metric);
    if (result.reason != kpolaris::TerminationReason::adaptive_step_underflow) {
        throw std::runtime_error("adaptive rejection was not classified as step underflow");
    }
    if (result.steps != 0 || result.path_length != Real(0)) {
        throw std::runtime_error("underflow advanced an unaccepted geodesic state");
    }

    // Only a metric that explicitly opts in may declare an event before an
    // RK trial.  Exercise the safe-step threshold plus static/moving surfaces.
    TerminalInnerEventTestMetric event_metric;
    params.inner_radius = Real(8.8);
    const auto beyond_tolerance =
        kpolaris::trace_pass_a_pixel_metric(0, params, event_metric);
    if (beyond_tolerance.reason !=
        kpolaris::TerminationReason::adaptive_step_underflow) {
        throw std::runtime_error(
            "crossing beyond event tolerance was captured early");
    }
    // d_cross=1.08*min_step lies above the former 1.05 window but its
    // 0.9-safe step is sub-minimum; event-first must close that gap.
    params.inner_radius = Real(8.92);
    const auto safe_step_gap_event =
        kpolaris::trace_pass_a_pixel_metric(0, params, event_metric);
    if (safe_step_gap_event.reason !=
            kpolaris::TerminationReason::reached_inner_boundary ||
        safe_step_gap_event.steps != 0) {
        throw std::runtime_error(
            "1.08-min-step crossing fell through the event-first window");
    }
    params.inner_radius = Real(9.5);
    const auto static_event =
        kpolaris::trace_pass_a_pixel_metric(0, params, event_metric);
    if (static_event.reason !=
        kpolaris::TerminationReason::reached_inner_boundary ||
        static_event.steps != 0) {
        throw std::runtime_error(
            "opt-in rejected static crossing was not a terminal event");
    }
    const auto segment_event =
        kpolaris::trace_pass_a_segment_endpoint_pixel_metric(
            0, params, event_metric);
    if (segment_event.reason !=
            kpolaris::TerminationReason::reached_inner_boundary ||
        segment_event.valid != 1 || segment_event.steps != 0) {
        throw std::runtime_error(
            "production Pass-A imminent crossing was not a valid endpoint");
    }
    event_metric.surface_velocity = Real(0.25);
    const auto moving_event =
        kpolaris::trace_pass_a_pixel_metric(0, params, event_metric);
    if (moving_event.reason !=
        kpolaris::TerminationReason::reached_inner_boundary ||
        moving_event.steps != 0) {
        throw std::runtime_error(
            "opt-in rejected moving crossing was not a terminal event");
    }
    params.inner_radius = Real(0.1);

    kpolaris::TransportState<Real> endpoint =
        kpolaris::initialize_camera_ray(metric, 0, params.camera);
    endpoint.x[1] = Real(5);
    endpoint.x[2] = Real(0);
    endpoint.x[3] = Real(0);
    const kpolaris::ConstantRadiationModel<Real> radiation_model;
    const auto pass_b =
        kpolaris::trace_pass_b_segment_model_endpoint_pixel_metric(
            0, params, radiation_model, 1, endpoint, 1,
            kpolaris::TerminationReason::reached_inner_boundary, 1, metric);
    if (pass_b.reason !=
        kpolaris::TerminationReason::adaptive_step_underflow) {
        throw std::runtime_error("Pass B finalizer overwrote adaptive step underflow");
    }
    if (pass_b.steps != 0) {
        throw std::runtime_error("Pass B underflow advanced an unaccepted state");
    }
}

void test_metric_time_domain_event_handling() {
    FiniteTimeFlatMetric metric;
    const kpolaris::Vec4<Real> x(
        Real(0), Real(10), Real(0), Real(0));
    const kpolaris::Vec4<Real> past_tangent(
        Real(-1), Real(-1), Real(0), Real(0));
    require_close(
        "finite-time generic step limiter",
        kpolaris::metric_limit_time_domain_step(
            metric, x, past_tangent, Real(1)),
        Real(0.025), Real(1e-13));
    if (kpolaris::metric_time_domain_step_is_terminal(
            metric, x, past_tangent, Real(1), Real(0.01))) {
        throw std::runtime_error(
            "finite-time event was classified before the minimum-step window");
    }
    if (!kpolaris::metric_time_domain_step_is_terminal(
            metric, x, past_tangent, Real(1), Real(0.03))) {
        throw std::runtime_error(
            "finite-time event was not classified inside the minimum-step window");
    }

    const kpolaris::MinkowskiMetric<Real> unlimited_metric;
    require_close(
        "ordinary metric keeps time-unlimited step",
        kpolaris::metric_limit_time_domain_step(
            unlimited_metric, x, past_tangent, Real(0.7)),
        Real(0.7), Real(0));

    kpolaris::PassAParams<Real> params;
    params.camera.model = kpolaris::CameraModel::ParallelPlane;
    params.camera.nx = 1;
    params.camera.ny = 1;
    params.camera.radius = Real(10);
    params.camera.inclination =
        Real(0.5) * Real(3.141592653589793238462643383279502884);
    params.inner_radius = Real(0.1);
    params.outer_radius = Real(20);
    params.step = Real(1);
    params.min_step = Real(0.01);
    params.max_step = Real(1);
    params.adaptive_tolerance = Real(1e-12);
    params.adaptive = 1;
    params.max_steps = 32;

    const auto pass_a =
        kpolaris::trace_pass_a_pixel_metric(0, params, metric);
    if (pass_a.reason !=
            kpolaris::TerminationReason::metric_time_exhausted ||
        pass_a.steps <= 0 ||
        !metric.time_domain_valid(pass_a.state.x)) {
        throw std::runtime_error(
            "Pass A did not stop deterministically on the finite time boundary");
    }

    const auto endpoint =
        kpolaris::trace_pass_a_segment_endpoint_pixel_metric(
            0, params, metric);
    if (endpoint.reason !=
            kpolaris::TerminationReason::metric_time_exhausted ||
        endpoint.valid != 0 || endpoint.steps <= 0 ||
        !metric.time_domain_valid(endpoint.state.x)) {
        throw std::runtime_error(
            "segment Pass A did not preserve the finite time-boundary reason");
    }

    kpolaris::TransportState<Real> injected =
        kpolaris::initialize_camera_ray(metric, 0, params.camera);
    injected.x[1] = Real(5);
    injected.x[2] = Real(0);
    injected.x[3] = Real(0);
    const kpolaris::ConstantRadiationModel<Real> radiation_model;
    const auto pass_b =
        kpolaris::trace_pass_b_segment_model_endpoint_pixel_metric(
            0, params, radiation_model, 1, injected, 1,
            kpolaris::TerminationReason::reached_inner_boundary,
            1, metric);
    if (pass_b.reason !=
            kpolaris::TerminationReason::metric_time_exhausted ||
        pass_b.steps <= 0 ||
        !metric.time_domain_valid(pass_b.state.x)) {
        throw std::runtime_error(
            "Pass B did not stop deterministically on the finite time boundary");
    }
}

void test_never_entered_outer_sphere_terminates() {
    FlatDomainMetric metric;
    kpolaris::PassAParams<Real> params;
    params.camera.model = kpolaris::CameraModel::ParallelPlane;
    params.camera.nx = 1;
    params.camera.ny = 1;
    params.camera.radius = Real(10);
    params.camera.inclination =
        Real(0.5) * Real(3.141592653589793238462643383279502884);
    params.camera.x_offset = Real(5);
    params.inner_radius = Real(1);
    params.outer_radius = Real(3);
    params.step = Real(0.25);
    params.max_steps = 200;

    const auto result = kpolaris::trace_pass_a_segment_endpoint_pixel_metric(
        0, params, metric);
    if (result.reason != kpolaris::TerminationReason::escaped_domain) {
        throw std::runtime_error("outer-sphere miss did not terminate as escaped_domain");
    }
    if (result.valid != 0) {
        throw std::runtime_error("never-entered outer-sphere miss became a valid endpoint");
    }
    if (!(result.steps > 0 && result.steps < params.max_steps)) {
        throw std::runtime_error("outer-sphere miss exhausted the geodesic step budget");
    }
}

void test_pass_a_single_center_ray() {
    kpolaris::PassAParams<Real> params;
    params.camera.nx = 3;
    params.camera.ny = 3;
    params.camera.radius = Real(30);
    params.camera.inclination = Real(1.0);
    params.camera.fov = Real(0.02);
    params.mass = Real(1);
    params.spin = Real(0.5);
    params.inner_radius = Real(3.0);
    params.outer_radius = Real(60);
    params.step = Real(0.02);
    params.max_steps = 3000;

    const auto result = kpolaris::trace_pass_a_pixel(4, params);
    require_close("passA initial null", result.initial_null, Real(0), Real(1e-10));
    if (result.reason != kpolaris::TerminationReason::reached_inner_boundary) {
        throw std::runtime_error("passA center ray did not reach inner boundary");
    }
    if (!(result.final_radius <= params.inner_radius + params.step * Real(2))) {
        throw std::runtime_error("passA final radius too large");
    }
    if (std::abs(result.final_null) > Real(2e-5)) {
        throw std::runtime_error("passA final null drift too large: " + std::to_string(result.final_null));
    }

    kpolaris::PassAParams<Real> dispatch_params = params;
    dispatch_params.camera.nx = 1;
    dispatch_params.camera.ny = 1;
    dispatch_params.camera.radius = Real(30);
    dispatch_params.max_steps = 0;
    dispatch_params.fmks_poly_norm =
        Real(0.5) * Real(3.141592653589793238462643383279502884) /
        (Real(1) + Real(1) / (dispatch_params.fmks_poly_alpha + Real(1)) /
         std::pow(dispatch_params.fmks_poly_xt, dispatch_params.fmks_poly_alpha));
    const kpolaris::CoordinateSystem coordinate_systems[] = {
        kpolaris::CoordinateSystem::BoyerLindquist,
        kpolaris::CoordinateSystem::CartesianKS,
        kpolaris::CoordinateSystem::SphericalKS,
        kpolaris::CoordinateSystem::FMKS,
        kpolaris::CoordinateSystem::MKS,
    };
    for (const auto coordinate_system : coordinate_systems) {
        dispatch_params.coordinate_system = coordinate_system;
        const auto dispatched = kpolaris::trace_pass_a_pixel(0, dispatch_params);
        if (dispatched.reason != kpolaris::TerminationReason::max_steps ||
            dispatched.steps != 0 || dispatched.path_length != Real(0)) {
            throw std::runtime_error("Pass A coordinate dispatch corrupted zero-step metadata");
        }
        require_close("Pass A coordinate dispatch radius", dispatched.final_radius,
                      dispatch_params.camera.radius, Real(1e-10));
        require_close("Pass A coordinate dispatch initial null", dispatched.initial_null,
                      Real(0), Real(1e-10));
        require_close("Pass A coordinate dispatch final null", dispatched.final_null,
                      dispatched.initial_null, Real(1e-12));
    }
}

void test_pass_a_kernel() {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    kpolaris::PassAParams<Real> params;
    params.camera.nx = 7;
    params.camera.ny = 5;
    params.camera.radius = Real(35);
    params.camera.inclination = Real(1.0);
    params.camera.fov = Real(0.08);
    params.mass = Real(1);
    params.spin = Real(0.5);
    params.inner_radius = Real(3.0);
    params.outer_radius = Real(80);
    params.step = Real(0.025);
    params.max_steps = 3000;

    const int npix = params.camera.nx * params.camera.ny;
    Kokkos::View<Real*, ExecSpace> final_r("passA_final_r", npix);
    Kokkos::View<Real*, ExecSpace> min_r("passA_min_r", npix);
    Kokkos::View<Real*, ExecSpace> initial_null("passA_initial_null", npix);
    Kokkos::View<Real*, ExecSpace> final_null("passA_final_null", npix);
    Kokkos::View<Real*, ExecSpace> frame_error("passA_frame_error", npix);
    Kokkos::View<int*, ExecSpace> steps("passA_steps", npix);
    Kokkos::View<int*, ExecSpace> reason("passA_reason", npix);

    kpolaris::run_pass_a<ExecSpace>(params, final_r, min_r, initial_null, final_null,
                                 frame_error, steps, reason);
    Kokkos::fence();

    auto h_final_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), final_r);
    auto h_min_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), min_r);
    auto h_initial_null = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), initial_null);
    auto h_final_null = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), final_null);
    auto h_frame_error = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), frame_error);
    auto h_steps = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), steps);
    auto h_reason = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reason);

    int inner_hits = 0;
    for (int p = 0; p < npix; ++p) {
        require_close("passA kernel initial null", h_initial_null(p), Real(0), Real(1e-9));
        if (!(h_steps(p) > 0)) {
            throw std::runtime_error("passA kernel ray took zero steps");
        }
        if (!(h_min_r(p) <= params.camera.radius)) {
            throw std::runtime_error("passA kernel min radius invalid");
        }
        if (std::abs(h_final_null(p)) > Real(5e-5)) {
            throw std::runtime_error("passA kernel final null drift too large: " + std::to_string(h_final_null(p)));
        }
        if (std::abs(h_frame_error(p)) > Real(5e-3)) {
            throw std::runtime_error("passA kernel frame error too large: " + std::to_string(h_frame_error(p)));
        }
        if (h_reason(p) == static_cast<int>(kpolaris::TerminationReason::reached_inner_boundary)) {
            inner_hits += 1;
            if (!(h_final_r(p) <= params.inner_radius + params.step * Real(2))) {
                throw std::runtime_error("passA kernel inner hit final radius invalid");
            }
        }
    }
    if (inner_hits == 0) {
        throw std::runtime_error("passA kernel had no inner-boundary hits");
    }
}



void test_pass_b_single_center_ray() {
    kpolaris::PassBParams<Real> params;
    params.pass_a.camera.nx = 3;
    params.pass_a.camera.ny = 3;
    params.pass_a.camera.radius = Real(30);
    params.pass_a.camera.inclination = Real(1.0);
    params.pass_a.camera.fov = Real(0.02);
    params.pass_a.mass = Real(1);
    params.pass_a.spin = Real(0.5);
    params.pass_a.inner_radius = Real(3.0);
    params.pass_a.outer_radius = Real(60);
    params.pass_a.step = Real(0.01);
    params.pass_a.max_steps = 5000;
    params.coeffs.jI = Real(0.8);
    params.coeffs.jQ = Real(0.05);
    params.coeffs.aI = Real(0.1);
    params.coeffs.rV = Real(0.2);
    params.radiation_substeps = 1;

    const auto result = kpolaris::trace_pass_b_pixel(4, params);
    if (result.reason != kpolaris::TerminationReason::reached_camera) {
        throw std::runtime_error("passB center ray did not reach camera");
    }
    if (!(result.pass_a.state.k[0] > Real(0) && result.state.k[0] > Real(0))) {
        throw std::runtime_error("both passes must retain a future-directed photon wavevector");
    }
    require_close("Pass B returns to camera time", result.state.x[0], Real(0), Real(2e-4));
    if (!(result.observed_stokes.I > Real(0))) {
        throw std::runtime_error("passB center ray produced non-positive I");
    }
    if (!(std::abs(result.observed_stokes.Q) > Real(0))) {
        throw std::runtime_error("passB center ray produced zero Q");
    }
    if (!(std::abs(result.observed_stokes.U) > Real(0))) {
        throw std::runtime_error("passB center ray produced zero U");
    }
    if (result.closure_x > Real(2e-4)) {
        throw std::runtime_error("passB center closure_x too large: " + std::to_string(result.closure_x));
    }
    if (result.closure_k > Real(2e-4)) {
        throw std::runtime_error("passB center closure_k too large: " + std::to_string(result.closure_k));
    }
    if (std::abs(result.final_null) > Real(5e-5)) {
        throw std::runtime_error("passB center final null too large: " + std::to_string(result.final_null));
    }
    if (std::abs(result.frame_error) > Real(5e-3)) {
        throw std::runtime_error("passB center frame error too large: " + std::to_string(result.frame_error));
    }
    if (result.overlap.det() < Real(0.5)) {
        throw std::runtime_error("passB center overlap determinant is not right-handed");
    }
}

void test_pass_b_parallel_plane_screen_orientation() {
    kpolaris::PassBParams<Real> params;
    params.pass_a.camera.model = kpolaris::CameraModel::ParallelPlane;
    params.pass_a.camera.nx = 3;
    params.pass_a.camera.ny = 3;
    params.pass_a.camera.radius = Real(30);
    params.pass_a.camera.inclination = Real(1.0);
    params.pass_a.camera.xspan = Real(0.02);
    params.pass_a.mass = Real(1);
    params.pass_a.spin = Real(0.5);
    params.pass_a.inner_radius = Real(3.0);
    params.pass_a.outer_radius = Real(60);
    params.pass_a.step = Real(0.01);
    params.pass_a.max_steps = 5000;
    params.coeffs.jI = Real(0.8);
    params.coeffs.jQ = Real(0.05);
    params.coeffs.jV = Real(0.02);
    params.coeffs.aI = Real(0.1);
    params.coeffs.rV = Real(0.2);

    const auto result = kpolaris::trace_pass_b_pixel(4, params);
    if (result.reason != kpolaris::TerminationReason::reached_camera) {
        throw std::runtime_error("parallel-plane PassB center ray did not reach camera");
    }
    if (result.overlap.det() < Real(0.5)) {
        throw std::runtime_error("parallel-plane PassB propagation basis is not right-handed");
    }
    if (!(result.observed_stokes.I > Real(0))) {
        throw std::runtime_error("parallel-plane PassB produced non-positive I");
    }
}

void test_pass_b_kernel() {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    kpolaris::PassBParams<Real> params;
    params.pass_a.camera.nx = 5;
    params.pass_a.camera.ny = 5;
    params.pass_a.camera.radius = Real(30);
    params.pass_a.camera.inclination = Real(1.0);
    params.pass_a.camera.fov = Real(0.04);
    params.pass_a.mass = Real(1);
    params.pass_a.spin = Real(0.5);
    params.pass_a.inner_radius = Real(3.0);
    params.pass_a.outer_radius = Real(70);
    params.pass_a.step = Real(0.0125);
    params.pass_a.max_steps = 5000;
    params.coeffs.jI = Real(0.8);
    params.coeffs.jQ = Real(0.05);
    params.coeffs.aI = Real(0.1);
    params.coeffs.rV = Real(0.2);
    params.radiation_substeps = 1;

    const int npix = params.pass_a.camera.nx * params.pass_a.camera.ny;
    Kokkos::View<Real*, ExecSpace> image_i("passB_I", npix);
    Kokkos::View<Real*, ExecSpace> image_q("passB_Q", npix);
    Kokkos::View<Real*, ExecSpace> image_u("passB_U", npix);
    Kokkos::View<Real*, ExecSpace> image_v("passB_V", npix);
    Kokkos::View<Real*, ExecSpace> closure_x("passB_closure_x", npix);
    Kokkos::View<Real*, ExecSpace> closure_k("passB_closure_k", npix);
    Kokkos::View<Real*, ExecSpace> final_null("passB_final_null", npix);
    Kokkos::View<Real*, ExecSpace> frame_error("passB_frame_error", npix);
    Kokkos::View<Real*, ExecSpace> det_r("passB_det_r", npix);
    Kokkos::View<int*, ExecSpace> steps("passB_steps", npix);
    Kokkos::View<int*, ExecSpace> reason("passB_reason", npix);

    kpolaris::run_pass_b<ExecSpace>(params, image_i, image_q, image_u, image_v,
                                 closure_x, closure_k, final_null, frame_error,
                                 det_r, steps, reason);
    Kokkos::fence();

    auto h_i = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_i);
    auto h_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_q);
    auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_u);
    auto h_closure_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), closure_x);
    auto h_closure_k = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), closure_k);
    auto h_final_null = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), final_null);
    auto h_frame_error = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), frame_error);
    auto h_det_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), det_r);
    auto h_steps = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), steps);
    auto h_reason = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reason);

    int returned = 0;
    for (int p = 0; p < npix; ++p) {
        if (h_reason(p) == static_cast<int>(kpolaris::TerminationReason::reached_camera)) {
            returned += 1;
            if (!(h_i(p) > Real(0))) {
                throw std::runtime_error("passB kernel produced non-positive I");
            }
            if (!(std::abs(h_q(p)) > Real(0) && std::abs(h_u(p)) > Real(0))) {
                throw std::runtime_error("passB kernel produced zero linear polarization component");
            }
            if (h_steps(p) <= 0) {
                throw std::runtime_error("passB kernel returned ray has zero steps");
            }
            if (h_closure_x(p) > Real(5e-3) || h_closure_k(p) > Real(5e-3)) {
                throw std::runtime_error("passB kernel closure too large");
            }
            if (std::abs(h_final_null(p)) > Real(5e-4)) {
                throw std::runtime_error("passB kernel final null too large");
            }
            if (std::abs(h_frame_error(p)) > Real(1e-2)) {
                throw std::runtime_error("passB kernel frame error too large");
            }
            if (h_det_r(p) < Real(0.5)) {
                throw std::runtime_error("passB kernel overlap determinant is not right-handed");
            }
        }
    }
    if (returned == 0) {
        throw std::runtime_error("passB kernel had no returned rays");
    }
}



void test_pass_b_constant_model_matches_constant_path() {
    kpolaris::PassBParams<Real> params;
    params.pass_a.camera.nx = 3;
    params.pass_a.camera.ny = 3;
    params.pass_a.camera.radius = Real(30);
    params.pass_a.camera.inclination = Real(1.0);
    params.pass_a.camera.fov = Real(0.02);
    params.pass_a.mass = Real(1);
    params.pass_a.spin = Real(0.5);
    params.pass_a.inner_radius = Real(3.0);
    params.pass_a.outer_radius = Real(60);
    params.pass_a.step = Real(0.0125);
    params.pass_a.max_steps = 5000;
    params.coeffs.jI = Real(0.8);
    params.coeffs.jQ = Real(0.05);
    params.coeffs.aI = Real(0.1);
    params.coeffs.rV = Real(0.2);
    params.radiation_substeps = 1;

    kpolaris::ConstantRadiationModel<Real> model;
    model.coeffs = params.coeffs;

    const auto direct = kpolaris::trace_pass_b_pixel(4, params);
    const auto modeled = kpolaris::trace_pass_b_model_pixel(4, params.pass_a, model,
                                                         params.radiation_substeps);
    require_close("passB model I", modeled.observed_stokes.I, direct.observed_stokes.I, Real(1e-10));
    require_close("passB model Q", modeled.observed_stokes.Q, direct.observed_stokes.Q, Real(1e-10));
    require_close("passB model U", modeled.observed_stokes.U, direct.observed_stokes.U, Real(1e-10));
    require_close("passB model V", modeled.observed_stokes.V, direct.observed_stokes.V, Real(1e-10));
}

void test_radiation_model_dlambda_scales() {
    kpolaris::ConstantRadiationModel<Real> constant_model;
    require_close("constant dlambda scale", constant_model.dlambda_scale(), Real(1), Real(1e-15));

    kpolaris::RadialPowerLawRadiationModel<Real> radial_model;
    require_close("radial dlambda scale", radial_model.dlambda_scale(), Real(1), Real(1e-15));

    kpolaris::RIAFAnalyticRadiationModel<Real> riaf_model;
    riaf_model.mbh_solar = Real(4.3e6);
    riaf_model.freq_cgs = Real(230.0e9);
    const Real gnewt = Real(6.67430e-8);
    const Real msun = Real(1.98847e33);
    const Real cl = Real(2.99792458e10);
    const Real expected = gnewt * riaf_model.mbh_solar * msun /
                          (cl * cl * riaf_model.freq_cgs);
    require_close("RIAF dlambda scale", riaf_model.dlambda_scale(), expected, Real(1e-14));
}

void test_riaf_fluid_four_velocity() {
    const kpolaris::KerrSchildInMetric<Real> metric(Real(1), Real(0.5));
    kpolaris::RIAFAnalyticRadiationModel<Real> model;
    model.keplerian_factor = Real(1);
    model.infall_factor = Real(0.2);

    kpolaris::TransportState<Real> state;
    state.x = kpolaris::Vec4<Real>(Real(0), Real(8), Real(0), Real(0));
    state.k = kpolaris::Vec4<Real>(Real(-1), Real(1), Real(0.1), Real(0.05));
    state.e1 = kpolaris::Vec4<Real>(Real(0), Real(0), Real(1), Real(0));
    state.e2 = kpolaris::Vec4<Real>(Real(0), Real(0), Real(0), Real(1));

    const auto work = metric.build_work(state.x);
    const auto ucon = model.fluid_four_velocity(metric, state.x, work.r);
    require_close("RIAF u.u", metric.dot(state.x, ucon, ucon), Real(-1), Real(1e-10));

    const auto bcon = model.magnetic_unit_four_vector(metric, state.x, ucon);
    require_close("RIAF b.u", metric.dot(state.x, bcon, ucon), Real(0), Real(1e-10));
    require_close("RIAF b.b", metric.dot(state.x, bcon, bcon), Real(1), Real(1e-10));

    const Real nu_scale = model.fluid_frequency_scale(metric, state, work.r);
    if (!(nu_scale > Real(0))) {
        throw std::runtime_error("RIAF fluid frequency scale is non-positive");
    }
    const auto bcov = model.lower_vector(metric, state.x, bcon);
    Real kdotb = Real(0);
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        kdotb += state.k[mu] * bcov[mu];
    }
    const Real mu = kdotb / nu_scale;
    if (std::abs(mu) > Real(1) + Real(1e-10)) {
        throw std::runtime_error("RIAF magnetic angle cosine outside [-1,1]");
    }
}

void test_riaf_mixed_velocity_is_timelike() {
    const kpolaris::KerrBoyerLindquistMetric<Real> metric(Real(1), Real(-0.4127));
    kpolaris::RIAFAnalyticRadiationModel<Real> model;
    model.keplerian_factor = Real(0.1161);
    model.infall_factor = Real(0.8839);

    const kpolaris::Vec4<Real> x(Real(0), Real(21.952706789353797),
                                 Real(1.582666), Real(3.139894));
    const auto ucon = model.fluid_four_velocity(metric, x, x[1]);
    const Real norm = metric.dot(x, ucon, ucon);
    if (!std::isfinite(norm)) {
        throw std::runtime_error("RIAF mixed velocity norm is non-finite");
    }
    require_close("RIAF mixed velocity u.u", norm, Real(-1), Real(1e-10));
}

void test_thermal_synchrotron_coefficients() {
    kpolaris::LocalThermalSynchrotronState<Real> state;
    state.nu = Real(230.0e9);
    state.ne = Real(5.0e6);
    state.thetae = Real(20);
    state.b_cgs = Real(30);
    state.theta = Real(1.0);
    state.sin_theta = std::sin(Real(1.0));

    kpolaris::ThermalSynchrotronParams<Real> params;
    params.fit = kpolaris::ThermalSynchrotronPandya;
    const auto pandya = kpolaris::thermal_synchrotron_magnetic_basis_coefficients(state, params);
    if (!(pandya.jI > Real(0) && pandya.aI > Real(0))) {
        throw std::runtime_error("Pandya thermal synchrotron coefficients are non-positive");
    }
    if (!(std::abs(pandya.jQ) > Real(0) || std::abs(pandya.jV) > Real(0))) {
        throw std::runtime_error("Pandya thermal synchrotron polarization coefficients are zero");
    }
    if (!std::isfinite(pandya.rQ) || !std::isfinite(pandya.rV)) {
        throw std::runtime_error("Pandya Faraday coefficients are not finite");
    }

    params.fit = kpolaris::ThermalSynchrotronDexter;
    const auto dexter = kpolaris::thermal_synchrotron_magnetic_basis_coefficients(state, params);
    if (!(dexter.jI > Real(0) && dexter.aI > Real(0))) {
        throw std::runtime_error("Dexter thermal synchrotron coefficients are non-positive");
    }
    if (!std::isfinite(dexter.rQ) || !std::isfinite(dexter.rV)) {
        throw std::runtime_error("Dexter Faraday coefficients are not finite");
    }
}


void test_nonthermal_synchrotron_coefficients() {
    kpolaris::LocalNonthermalSynchrotronState<Real> state;
    state.nu = Real(230.0e9);
    state.ne = Real(1.0e6);
    state.thetae = Real(20);
    state.b_cgs = Real(30);
    state.sin_theta = Real(0.7);
    state.cos_theta = std::sqrt(Real(1) - state.sin_theta * state.sin_theta);

    kpolaris::NonthermalSynchrotronParams<Real> params;
    params.distribution = kpolaris::SynchrotronKappa;
    params.kappa = Real(3.5);
    const auto kappa = kpolaris::nonthermal_synchrotron_magnetic_basis_coefficients(state, params);
    if (!(kappa.jI > Real(0) && kappa.aI >= Real(0))) {
        throw std::runtime_error("kappa synchrotron coefficients are non-positive");
    }
    if (!std::isfinite(kappa.jQ) || !std::isfinite(kappa.jV) ||
        !std::isfinite(kappa.aQ) || !std::isfinite(kappa.aV) ||
        !std::isfinite(kappa.rQ) || !std::isfinite(kappa.rV)) {
        throw std::runtime_error("kappa synchrotron coefficients are not finite");
    }

    params.kappa = Real(6.0);
    const auto kappa_outside_rho_fit = kpolaris::nonthermal_synchrotron_magnetic_basis_coefficients(state, params);
    require_close("kappa rhoQ outside Symphony fit range", kappa_outside_rho_fit.rQ, Real(0), Real(0));
    require_close("kappa rhoV outside Symphony fit range", kappa_outside_rho_fit.rV, Real(0), Real(0));

    params.distribution = kpolaris::SynchrotronPowerLaw;
    const auto powerlaw = kpolaris::nonthermal_synchrotron_magnetic_basis_coefficients(state, params);
    if (!(powerlaw.jI > Real(0) && powerlaw.aI >= Real(0))) {
        throw std::runtime_error("power-law synchrotron coefficients are non-positive");
    }
    if (!std::isfinite(powerlaw.jQ) || !std::isfinite(powerlaw.jV) ||
        !std::isfinite(powerlaw.aQ) || !std::isfinite(powerlaw.aV)) {
        throw std::runtime_error("power-law synchrotron coefficients are not finite");
    }
    require_close("power-law rhoQ", powerlaw.rQ, Real(0), Real(0));
    require_close("power-law rhoV", powerlaw.rV, Real(0), Real(0));
}

#if !KPOLARIS_USE_FLOAT
void test_symphony_fast_fit_checkpoints() {
    // Independent fast-fit values from the archived upstream Symphony tree at
    // a869c6bc2ee31ce5e211f8d201523c6c8f0c87c9.  KPolaris stores invariant
    // coefficients; convert them back to the physical j_nu/alpha_nu/rho_nu
    // values emitted by the reference probe.  Its magnetic-screen Q axis is
    // opposite for emission and absorption, but not for rho_Q.
    const Real nu = Real(230.e9);
    const Real nu2 = nu * nu;
    const Real tolerance = sizeof(Real) < sizeof(double) ? Real(5e-4) : Real(1e-5);

    kpolaris::LocalThermalSynchrotronState<Real> thermal_state;
    thermal_state.nu = nu;
    thermal_state.ne = Real(1);
    thermal_state.thetae = Real(10);
    thermal_state.b_cgs = Real(30);
    thermal_state.theta = Real(3.141592653589793238462643383279502884) / Real(3);
    thermal_state.sin_theta = std::sin(thermal_state.theta);
    thermal_state.cos_theta = std::cos(thermal_state.theta);
    kpolaris::ThermalSynchrotronParams<Real> thermal_params;
    thermal_params.fit = kpolaris::ThermalSynchrotronPandya;
    const auto thermal =
        kpolaris::thermal_synchrotron_magnetic_basis_coefficients(
            thermal_state, thermal_params);
    require_relative("Symphony thermal jI", thermal.jI * nu2,
                     Real(1.31261797658467561e-22), tolerance);
    require_relative("Symphony thermal jQ", -thermal.jQ * nu2,
                     Real(-1.04527441105734533e-22), tolerance);
    require_relative("Symphony thermal alphaI", thermal.aI / nu,
                     Real(1.36195875354652547e-19), tolerance);
    require_relative("Symphony thermal alphaQ", -thermal.aQ / nu,
                     Real(-1.08456585190299210e-19), tolerance);
    require_relative("Symphony thermal rhoQ", thermal.rQ / nu,
                     Real(1.00838208010819689e-19), tolerance);
    // These magnitudes are the retired Pandya V branch retained in the same
    // upstream repository's regression file; the active Symphony V function
    // at this revision uses a later formula.
    require_relative("Symphony archived thermal jV", std::abs(thermal.jV * nu2),
                     Real(2.81598011687e-24), tolerance);
    require_relative("Symphony archived thermal alphaV", std::abs(thermal.aV / nu),
                     Real(2.92183166648e-21), tolerance);

    kpolaris::LocalNonthermalSynchrotronState<Real> state;
    state.nu = nu;
    state.ne = Real(1);
    state.thetae = Real(10);
    state.b_cgs = Real(30);
    state.sin_theta = thermal_state.sin_theta;
    state.cos_theta = thermal_state.cos_theta;

    kpolaris::NonthermalSynchrotronParams<Real> params;
    params.distribution = kpolaris::SynchrotronKappa;
    params.kappa = Real(3.5);
    params.kappa_width = Real(10);
    params.kappa_interp_begin = Real(1e20);
    params.kappa_interp_end = Real(1e20);
    const auto kappa =
        kpolaris::nonthermal_synchrotron_magnetic_basis_coefficients(state, params);
    require_relative("Symphony kappa jI", kappa.jI * nu2,
                     Real(2.77087098467486997e-22), tolerance);
    require_relative("Symphony kappa jQ", -kappa.jQ * nu2,
                     Real(-1.79193573530907640e-22), tolerance);
    require_relative("Symphony kappa jV", kappa.jV * nu2,
                     Real(4.31126410186166409e-24), tolerance);
    require_relative("Symphony kappa alphaI", kappa.aI / nu,
                     Real(1.01739581552720008e-19), tolerance);
    require_relative("Symphony kappa alphaQ", -kappa.aQ / nu,
                     Real(-7.08859966781963241e-20), tolerance);
    require_relative("Symphony kappa alphaV", kappa.aV / nu,
                     Real(1.65908484701845353e-21), tolerance);
    require_relative("Symphony kappa rhoQ", kappa.rQ / nu,
                     Real(-2.75845020624880583e-20), tolerance);
    require_relative("Symphony kappa rhoV", kappa.rV / nu,
                     Real(5.35150622743995775e-20), tolerance);

    params.distribution = kpolaris::SynchrotronPowerLaw;
    params.power_law_p = Real(2.5);
    params.power_law_gamma_min = Real(1);
    params.power_law_gamma_max = Real(1000);
    params.power_law_gamma_cutoff = Real(1e10);
    const Real me = Real(9.1093826e-28);
    const Real cl = Real(2.99792458e10);
    params.power_law_eta =
        Real(2) * (params.power_law_p - Real(1)) * me * cl * cl *
        params.power_law_gamma_min /
        (state.b_cgs * state.b_cgs * (params.power_law_p - Real(2)));
    const auto powerlaw =
        kpolaris::nonthermal_synchrotron_magnetic_basis_coefficients(state, params);
    require_relative("Symphony power-law jI", powerlaw.jI * nu2,
                     Real(2.03808334020804221e-24), tolerance);
    require_relative("Symphony power-law jQ", -powerlaw.jQ * nu2,
                     Real(-1.47585345325409924e-24), tolerance);
    require_relative("Symphony power-law jV", powerlaw.jV * nu2,
                     Real(3.88375258782347956e-26), tolerance);
    require_relative("Symphony power-law alphaI", powerlaw.aI / nu,
                     Real(1.90379277421107568e-21), tolerance);
    require_relative("Symphony power-law alphaQ", -powerlaw.aQ / nu,
                     Real(-1.47291138324468217e-21), tolerance);
    require_relative("Symphony power-law alphaV", powerlaw.aV / nu,
                     Real(4.24640504782263757e-23), tolerance);
}
#endif

void test_riaf_model_coefficients() {
    const kpolaris::KerrSchildInMetric<Real> metric(Real(1), Real(0.5));
    kpolaris::RIAFAnalyticRadiationModel<Real> model;
    model.r_min = Real(2.1);
    model.r_max = Real(30);
    model.emission_scale = Real(1);
    model.absorption_scale = Real(1);
    model.faraday_scale = Real(1);

    kpolaris::TransportState<Real> state;
    state.x = kpolaris::Vec4<Real>(Real(0), Real(8), Real(0), Real(0));
    state.k = kpolaris::Vec4<Real>(Real(1), Real(-1), Real(0.1), Real(0.05));
    state.e1 = kpolaris::Vec4<Real>(Real(0), Real(0), Real(1), Real(0));
    state.e2 = kpolaris::Vec4<Real>(Real(0), Real(0), Real(0), Real(1));

    const auto coeffs = model.coefficients(metric, state, Real(0.5));
    if (!(coeffs.jI > Real(0))) {
        throw std::runtime_error("RIAF model produced non-positive jI in disk body");
    }
    if (!(std::abs(coeffs.jQ) > Real(0) || std::abs(coeffs.jU) > Real(0))) {
        throw std::runtime_error("RIAF model produced zero linear polarized emissivity");
    }
    if (!(coeffs.aI > Real(0))) {
        throw std::runtime_error("RIAF model produced non-positive aI in disk body");
    }

    const auto analysis_plasma =
        kpolaris::analysis_plasma_diagnostics(metric, model, state);
    const auto trace_plasma =
        kpolaris::trace_plasma_diagnostics(metric, model, state);
    if (!analysis_plasma.valid || !(analysis_plasma.ne_cgs > Real(0)) ||
        !(analysis_plasma.thetae > Real(0)) ||
        !(analysis_plasma.b_cgs > Real(0))) {
        throw std::runtime_error("RIAF analysis plasma diagnostics are invalid in disk body");
    }
    require_close("RIAF analysis/trace ne", analysis_plasma.ne_cgs,
                  trace_plasma.ne_cgs, Real(1e-12));
    require_close("RIAF analysis/trace thetae", analysis_plasma.thetae,
                  trace_plasma.thetae, Real(1e-12));
    require_close("RIAF analysis/trace B", analysis_plasma.b_cgs,
                  trace_plasma.b_cgs, Real(1e-12));

    const auto pandya_coeffs = coeffs;

    model.emission_type = 2;
    const auto kappa_coeffs = model.coefficients(metric, state, Real(0.5));
    if (!(kappa_coeffs.jI > Real(0)) ||
        std::abs(kappa_coeffs.jI - pandya_coeffs.jI) <=
            Real(1e-8) * std::max(std::abs(pandya_coeffs.jI), std::abs(kappa_coeffs.jI))) {
        throw std::runtime_error("RIAF runtime kappa dispatch did not change jI");
    }

    model.emission_type = 3;
    const auto powerlaw_coeffs = model.coefficients(metric, state, Real(0.5));
    if (!(powerlaw_coeffs.jI > Real(0)) ||
        std::abs(powerlaw_coeffs.jI - pandya_coeffs.jI) <=
            Real(1e-8) * std::max(std::abs(pandya_coeffs.jI), std::abs(powerlaw_coeffs.jI))) {
        throw std::runtime_error("RIAF runtime power-law dispatch did not change jI");
    }

    model.emission_type = 4;
    const auto dexter_coeffs = model.coefficients(metric, state, Real(0.5));
    if (!(dexter_coeffs.jI > Real(0))) {
        throw std::runtime_error("RIAF Dexter thermal branch produced non-positive jI");
    }
#if !KPOLARIS_USE_FLOAT
    if (std::abs(dexter_coeffs.jI - pandya_coeffs.jI) <=
        Real(1e-8) * std::max(std::abs(pandya_coeffs.jI), std::abs(dexter_coeffs.jI))) {
        throw std::runtime_error("RIAF runtime Dexter dispatch did not change jI");
    }
#else
    // The production image/trace targets reject a float transport state.  At
    // this checkpoint the two thermal fits round to the same float jI, so
    // positivity is the meaningful contract; double and mixed-special builds
    // retain the fit-discrimination check above.
#endif

    state.x = kpolaris::Vec4<Real>(Real(0), Real(80), Real(0), Real(0));
    const auto outside = model.coefficients(metric, state, Real(0.5));
    require_close("RIAF outside jI", outside.jI, Real(0), Real(1e-12));
    require_close("RIAF outside aI", outside.aI, Real(0), Real(1e-12));
    const auto outside_plasma =
        kpolaris::analysis_plasma_diagnostics(metric, model, state);
    if (outside_plasma.valid || outside_plasma.ne_cgs != Real(0) ||
        outside_plasma.thetae != Real(0) || outside_plasma.b_cgs != Real(0)) {
        throw std::runtime_error("RIAF analysis plasma diagnostics accepted an out-of-domain state");
    }
}

void test_magnetized_torus_model_coefficients() {
    const kpolaris::KerrSchildInMetric<Real> metric(Real(1), Real(0.9));
    kpolaris::MagnetizedTorusRadiationModel<Real> model;
    model.spin = Real(0.9);
    model.initialize_default_torus();

    if (!(model.rc > model.rcusp && model.KK > Real(0))) {
        throw std::runtime_error("magnetized torus initialization failed");
    }

    kpolaris::TransportState<Real> state;
    state.x = kpolaris::Vec4<Real>(Real(0), model.rc, -model.spin, Real(0));
    state.k = kpolaris::Vec4<Real>(Real(1), Real(-1), Real(0.02), Real(0.01));
    kpolaris::initialize_screen_frame_from_reference(metric, state.x, state.k, state.e1, state.e2);

    const auto coeffs = model.coefficients(metric, state, Real(0));
    if (!(coeffs.jI > Real(0))) {
        throw std::runtime_error("magnetized torus model produced non-positive jI at pressure maximum");
    }
    if (!(coeffs.aI > Real(0))) {
        throw std::runtime_error("magnetized torus model produced non-positive aI at pressure maximum");
    }
    if (!(std::abs(coeffs.jQ) > Real(0) || std::abs(coeffs.jU) > Real(0))) {
        throw std::runtime_error("magnetized torus model produced zero linear polarization");
    }

    const auto analysis_plasma =
        kpolaris::analysis_plasma_diagnostics(metric, model, state);
    const auto trace_plasma =
        kpolaris::trace_plasma_diagnostics(metric, model, state);
    if (!analysis_plasma.valid || !(analysis_plasma.ne_cgs > Real(0)) ||
        !(analysis_plasma.thetae > Real(0)) ||
        !(analysis_plasma.b_cgs > Real(0)) || !(analysis_plasma.beta > Real(0)) ||
        !(analysis_plasma.sigma > Real(0))) {
        throw std::runtime_error("torus analysis plasma diagnostics are invalid at pressure maximum");
    }
    require_close("torus analysis beta", analysis_plasma.beta, model.beta,
                  Real(1e-12));
    require_close("torus analysis/trace ne", analysis_plasma.ne_cgs,
                  trace_plasma.ne_cgs, Real(1e-12));
    require_close("torus analysis/trace thetae", analysis_plasma.thetae,
                  trace_plasma.thetae, Real(1e-12));
    require_close("torus analysis/trace B", analysis_plasma.b_cgs,
                  trace_plasma.b_cgs, Real(1e-10));
    require_close("torus analysis/trace beta", analysis_plasma.beta,
                  trace_plasma.beta, Real(1e-12));
    require_close("torus analysis/trace sigma", analysis_plasma.sigma,
                  trace_plasma.sigma, Real(1e-12));

    state.x = kpolaris::Vec4<Real>(Real(0), model.r_outer * Real(2), Real(0), Real(0));
    const auto outside = model.coefficients(metric, state, Real(0));
    require_close("torus outside jI", outside.jI, Real(0), Real(1e-12));
    require_close("torus outside aI", outside.aI, Real(0), Real(1e-12));
    const auto outside_plasma =
        kpolaris::analysis_plasma_diagnostics(metric, model, state);
    if (outside_plasma.valid || outside_plasma.ne_cgs != Real(0) ||
        outside_plasma.thetae != Real(0) || outside_plasma.b_cgs != Real(0) ||
        outside_plasma.beta != Real(0) || outside_plasma.sigma != Real(0)) {
        throw std::runtime_error("torus analysis plasma diagnostics accepted an out-of-domain state");
    }
}

void test_pass_b_segment_model_single_ray() {
    kpolaris::PassAParams<Real> pass_a;
    pass_a.camera.nx = 5;
    pass_a.camera.ny = 5;
    pass_a.camera.radius = Real(30);
    pass_a.camera.inclination = Real(1.0);
    pass_a.camera.fov = Real(0.04);
    pass_a.mass = Real(1);
    pass_a.spin = Real(0.5);
    pass_a.inner_radius = Real(2.1);
    pass_a.outer_radius = Real(12);
    pass_a.step = Real(0.025);
    pass_a.max_steps = 5000;

    kpolaris::RIAFAnalyticRadiationModel<Real> model;
    model.r_min = Real(2.2);
    model.r_max = Real(10);
    model.emission_scale = Real(1);
    model.absorption_scale = Real(1);
    model.faraday_scale = Real(1);

    const auto out = kpolaris::trace_pass_b_segment_model_pixel(12, pass_a, model, 2);
    if (out.reason != kpolaris::TerminationReason::reached_camera) {
        throw std::runtime_error("segment Pass B did not return to camera");
    }
#if !KPOLARIS_USE_FLOAT
    // CGS synchrotron invariants can underflow in the experimental main-float
    // build. Its image/trace science tools are rejected at configuration time;
    // retain the geometric endpoint, closure and null checks below in float.
    if (!(out.observed_stokes.I > Real(0))) {
        throw std::runtime_error("segment Pass B produced non-positive I");
    }
#endif
    if (out.closure_x > Real(1e-3) || out.closure_k > Real(1e-3)) {
        throw std::runtime_error("segment Pass B closure too large");
    }
    if (std::abs(out.final_null) > Real(1e-4)) {
        throw std::runtime_error("segment Pass B final null too large");
    }
}

void test_trace_image_geometry_schedule_consistency() {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    kpolaris::PassAParams<Real> pass_a;
    pass_a.camera.nx = 5;
    pass_a.camera.ny = 5;
    pass_a.camera.radius = Real(30);
    pass_a.camera.inclination = Real(1.0);
    pass_a.camera.fov = Real(0.04);
    pass_a.coordinate_system = kpolaris::CoordinateSystem::BoyerLindquist;
    pass_a.mass = Real(1);
    pass_a.spin = Real(0.5);
    pass_a.inner_radius = Real(2.1);
    pass_a.outer_radius = Real(12);
    pass_a.step = Real(0.1);
    // Deliberately larger than the nominal step.  Only the metric opt-in
    // floor (2e-4) makes this high-accuracy trajectory integrable; this catches
    // a trace path that accidentally reuses the configured value (0.2).
    pass_a.min_step = Real(0.2);
    pass_a.max_step = Real(0.5);
    pass_a.max_radiation_step = Real(0.15);
    pass_a.max_radiation_depth = Real(0.5);
    pass_a.max_absorption_depth = Real(0.5);
    pass_a.max_faraday_depth = Real(0.5);
    pass_a.adaptive = 1;
    pass_a.adaptive_tolerance = Real(1e-10);
    pass_a.max_steps = 5000;

    kpolaris::RIAFAnalyticRadiationModel<Real> model;
    model.r_min = Real(2.2);
    model.r_max = Real(10);
    model.emission_scale = Real(1);
    model.absorption_scale = Real(1);
    model.faraday_scale = Real(1);

    constexpr int pixel = 12;
    const LocalFloorKerrBoyerLindquistMetric metric(
        pass_a.mass, pass_a.spin);
    const auto image_result = kpolaris::trace_pass_b_segment_model_pixel_metric(
        pixel, pass_a, model, 1, metric);
    const auto endpoint_result =
        kpolaris::trace_pass_a_segment_endpoint_pixel_metric(
            pixel, pass_a, metric);

    constexpr int tested_frequency_count =
        kpolaris::max_frequencies > 1 ? 2 : 1;
    const Real frequencies[2] = {Real(230e9), Real(100e9)};
    kpolaris::Stokes<Real> multifrequency_stokes[2];
    kpolaris::Stokes<Real> control_stokes[2];
    const auto multifrequency_result =
        kpolaris::trace_pass_b_segment_model_multifrequency_pixel_metric(
            pixel, pass_a, model, 1, frequencies, tested_frequency_count,
            multifrequency_stokes, metric);
    const auto control_result =
        kpolaris::trace_pass_b_segment_model_multifrequency_control_pixel_metric(
            pixel, pass_a, model, 1, frequencies, tested_frequency_count,
            frequencies, tested_frequency_count, control_stokes, metric);

    Real invalid_frequencies[KPOLARIS_MAX_FREQUENCIES + 1] = {};
    kpolaris::Stokes<Real> invalid_stokes[KPOLARIS_MAX_FREQUENCIES + 1];
    const auto invalid_multifrequency_result =
        kpolaris::trace_pass_b_segment_model_multifrequency_pixel_metric(
            pixel, pass_a, model, 1, invalid_frequencies,
            KPOLARIS_MAX_FREQUENCIES + 1, invalid_stokes, metric);
    if (invalid_multifrequency_result.reason !=
        kpolaris::TerminationReason::invalid_state) {
        throw std::runtime_error(
            "oversized multifrequency request was not rejected");
    }

    if (image_result.pass_a.reason != kpolaris::TerminationReason::reached_inner_boundary &&
        image_result.pass_a.reason != kpolaris::TerminationReason::escaped_domain) {
        throw std::runtime_error("segment Pass A did not terminate on a radiation boundary");
    }
    const Real expected_boundary =
        image_result.pass_a.reason == kpolaris::TerminationReason::reached_inner_boundary ?
            pass_a.inner_radius : pass_a.outer_radius;
    require_close("segment Pass A boundary radius",
                  image_result.pass_a.final_radius, expected_boundary, Real(2e-2));

    const auto require_matching_pass_a = [&](const auto& candidate,
                                             const char* label) {
        if (candidate.pass_a.steps != image_result.pass_a.steps ||
            candidate.pass_a.reason != image_result.pass_a.reason) {
            throw std::runtime_error(std::string(label) +
                                     " Pass A schedule diverged");
        }
        require_close(label, candidate.pass_a.final_radius,
                      image_result.pass_a.final_radius, Real(1e-12));
    };
    if (endpoint_result.steps != image_result.pass_a.steps ||
        endpoint_result.reason != image_result.pass_a.reason) {
        throw std::runtime_error("endpoint/image Pass A schedule diverged");
    }
    require_close("endpoint/image Pass A radius",
                  kpolaris::radial_coordinate(metric, endpoint_result.state.x),
                  image_result.pass_a.final_radius, Real(1e-12));
    require_matching_pass_a(multifrequency_result, "multifrequency/image Pass A radius");
    require_matching_pass_a(control_result, "control/image Pass A radius");

    kpolaris::TraceConfig<Real> config;
    config.first_pixel = pixel;
    config.ray_count = 1;
    config.max_samples = 0;
    config.record_lambda = 0;
    config.record_coords = 0;
    config.record_plasma = 0;
    config.record_coeffs = 0;
    config.record_stokes = 0;

    kpolaris::TraceViews<Real, ExecSpace> views;
    views.pixel = Kokkos::View<int*, ExecSpace>("consistency_pixel", 1);
    views.sample_count = Kokkos::View<int*, ExecSpace>("consistency_samples", 1);
    views.pass_a_steps = Kokkos::View<int*, ExecSpace>("consistency_pass_a", 1);
    views.pass_b_steps = Kokkos::View<int*, ExecSpace>("consistency_pass_b", 1);
    views.reason = Kokkos::View<int*, ExecSpace>("consistency_reason", 1);
    views.closure_x = Kokkos::View<Real*, ExecSpace>("consistency_closure_x", 1);
    views.closure_k = Kokkos::View<Real*, ExecSpace>("consistency_closure_k", 1);
    views.final_null = Kokkos::View<Real*, ExecSpace>("consistency_final_null", 1);
    views.frame_error = Kokkos::View<Real*, ExecSpace>("consistency_frame_error", 1);
    views.final_propagated_i = Kokkos::View<Real*, ExecSpace>("consistency_final_propagated_i", 1);
    views.final_propagated_q = Kokkos::View<Real*, ExecSpace>("consistency_final_propagated_q", 1);
    views.final_propagated_u = Kokkos::View<Real*, ExecSpace>("consistency_final_propagated_u", 1);
    views.final_propagated_v = Kokkos::View<Real*, ExecSpace>("consistency_final_propagated_v", 1);
    views.final_observed_i = Kokkos::View<Real*, ExecSpace>("consistency_final_observed_i", 1);
    views.final_observed_q = Kokkos::View<Real*, ExecSpace>("consistency_final_observed_q", 1);
    views.final_observed_u = Kokkos::View<Real*, ExecSpace>("consistency_final_observed_u", 1);
    views.final_observed_v = Kokkos::View<Real*, ExecSpace>("consistency_final_observed_v", 1);

    Kokkos::parallel_for(
        "KPOLARISTraceImageScheduleConsistency",
        Kokkos::RangePolicy<ExecSpace>(0, 1),
        KOKKOS_LAMBDA(const int) {
            kpolaris::trace_pass_b_segment_samples_pixel_metric(
                0, pixel, pass_a, model, config, metric,
                static_cast<const Real*>(nullptr), 0, views);
        });
    Kokkos::fence();

    const auto pass_a_steps = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), views.pass_a_steps);
    const auto pass_b_steps = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), views.pass_b_steps);
    const auto reason = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), views.reason);
    const auto closure_x = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), views.closure_x);
    const auto closure_k = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), views.closure_k);
    const auto final_null = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), views.final_null);
    const auto frame_error = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), views.frame_error);

    if (pass_a_steps(0) != image_result.pass_a.steps ||
        pass_b_steps(0) != image_result.steps ||
        reason(0) != static_cast<int>(image_result.reason)) {
        throw std::runtime_error("trace/image integration schedules diverged");
    }
    // The host reference and CUDA kernel follow an identical discrete
    // schedule, but device math can accumulate diagnostics a few ulps
    // differently over thousands of RK4 steps. Keep the exact schedule gate
    // above and use an absolute-scale tolerance below the production warning
    // thresholds for these near-zero diagnostics.
    constexpr Real diagnostic_tolerance = Real(1e-8);
    require_close("trace/image closure_x", closure_x(0),
                  image_result.closure_x, diagnostic_tolerance);
    require_close("trace/image closure_k", closure_k(0),
                  image_result.closure_k, diagnostic_tolerance);
    require_close("trace/image final_null", final_null(0),
                  image_result.final_null, diagnostic_tolerance);
    require_close("trace/image frame_error", frame_error(0),
                  image_result.frame_error, diagnostic_tolerance);
}

#if !KPOLARIS_USE_FLOAT
void test_pass_b_riaf_model_kernel() {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    kpolaris::PassAParams<Real> pass_a;
    pass_a.camera.nx = 5;
    pass_a.camera.ny = 5;
    pass_a.camera.radius = Real(30);
    pass_a.camera.inclination = Real(1.0);
    pass_a.camera.fov = Real(0.04);
    pass_a.mass = Real(1);
    pass_a.spin = Real(0.5);
    pass_a.inner_radius = Real(2.1);
    pass_a.outer_radius = Real(70);
    pass_a.step = Real(0.0125);
    pass_a.max_steps = 5000;

    kpolaris::RIAFAnalyticRadiationModel<Real> model;
    model.r_min = Real(2.2);
    model.r_max = Real(30);
    model.emission_scale = Real(1);
    model.absorption_scale = Real(1);
    model.faraday_scale = Real(1);

    const int npix = pass_a.camera.nx * pass_a.camera.ny;
    Kokkos::View<Real*, ExecSpace> image_i("passB_riaf_I", npix);
    Kokkos::View<Real*, ExecSpace> image_q("passB_riaf_Q", npix);
    Kokkos::View<Real*, ExecSpace> image_u("passB_riaf_U", npix);
    Kokkos::View<Real*, ExecSpace> image_v("passB_riaf_V", npix);
    Kokkos::View<Real*, ExecSpace> closure_x("passB_riaf_closure_x", npix);
    Kokkos::View<Real*, ExecSpace> closure_k("passB_riaf_closure_k", npix);
    Kokkos::View<Real*, ExecSpace> final_null("passB_riaf_final_null", npix);
    Kokkos::View<Real*, ExecSpace> frame_error("passB_riaf_frame_error", npix);
    Kokkos::View<Real*, ExecSpace> det_r("passB_riaf_det_r", npix);
    Kokkos::View<int*, ExecSpace> steps("passB_riaf_steps", npix);
    Kokkos::View<int*, ExecSpace> reason("passB_riaf_reason", npix);

    kpolaris::run_pass_b_model<ExecSpace>(pass_a, model, 2, image_i, image_q,
                                       image_u, image_v, closure_x, closure_k,
                                       final_null, frame_error, det_r, steps,
                                       reason);
    Kokkos::fence();

    auto h_i = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_i);
    auto h_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_q);
    auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_u);
    auto h_closure_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), closure_x);
    auto h_closure_k = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), closure_k);
    auto h_final_null = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), final_null);
    auto h_frame_error = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), frame_error);
    auto h_det_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), det_r);
    auto h_reason = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reason);

    int returned = 0;
    Real sum_i = Real(0);
    Real min_i = kpolaris::large_positive<Real>();
    Real max_i = Real(0);
    for (int p = 0; p < npix; ++p) {
        if (h_reason(p) == static_cast<int>(kpolaris::TerminationReason::reached_camera)) {
            returned += 1;
            if (!(h_i(p) > Real(0))) {
                throw std::runtime_error("passB RIAF kernel produced non-positive I");
            }
            if (!(std::abs(h_q(p)) > Real(0) || std::abs(h_u(p)) > Real(0))) {
                throw std::runtime_error("passB RIAF kernel produced zero linear polarization");
            }
            if (h_closure_x(p) > Real(5e-3) || h_closure_k(p) > Real(5e-3)) {
                throw std::runtime_error("passB RIAF kernel closure too large");
            }
            if (std::abs(h_final_null(p)) > Real(5e-4)) {
                throw std::runtime_error("passB RIAF kernel final null too large");
            }
            if (std::abs(h_frame_error(p)) > Real(1e-2)) {
                throw std::runtime_error("passB RIAF kernel frame error too large");
            }
            if (h_det_r(p) < Real(0.5)) {
                throw std::runtime_error("passB RIAF overlap determinant is not right-handed");
            }
            sum_i += h_i(p);
            min_i = std::min(min_i, h_i(p));
            max_i = std::max(max_i, h_i(p));
        }
    }
    if (returned == 0 || !(sum_i > Real(0))) {
        throw std::runtime_error("passB RIAF kernel had no positive returned intensity");
    }
    if (!(max_i > min_i)) {
        throw std::runtime_error("passB RIAF model did not vary intensity");
    }
}
#endif

void test_pass_b_radial_model_kernel() {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    kpolaris::PassAParams<Real> pass_a;
    pass_a.camera.nx = 5;
    pass_a.camera.ny = 5;
    pass_a.camera.radius = Real(30);
    pass_a.camera.inclination = Real(1.0);
    pass_a.camera.fov = Real(0.04);
    pass_a.mass = Real(1);
    pass_a.spin = Real(0.5);
    pass_a.inner_radius = Real(3.0);
    pass_a.outer_radius = Real(70);
    pass_a.step = Real(0.0125);
    pass_a.max_steps = 5000;

    kpolaris::RadialPowerLawRadiationModel<Real> model;
    model.jI0 = Real(0.8);
    model.pol_frac = Real(0.12);
    model.aI0 = Real(0.05);
    model.rV0 = Real(0.15);
    model.r_peak = Real(8);
    model.width = Real(4);

    const int npix = pass_a.camera.nx * pass_a.camera.ny;
    Kokkos::View<Real*, ExecSpace> image_i("passB_model_I", npix);
    Kokkos::View<Real*, ExecSpace> image_q("passB_model_Q", npix);
    Kokkos::View<Real*, ExecSpace> image_u("passB_model_U", npix);
    Kokkos::View<Real*, ExecSpace> image_v("passB_model_V", npix);
    Kokkos::View<Real*, ExecSpace> closure_x("passB_model_closure_x", npix);
    Kokkos::View<Real*, ExecSpace> closure_k("passB_model_closure_k", npix);
    Kokkos::View<Real*, ExecSpace> final_null("passB_model_final_null", npix);
    Kokkos::View<Real*, ExecSpace> frame_error("passB_model_frame_error", npix);
    Kokkos::View<Real*, ExecSpace> det_r("passB_model_det_r", npix);
    Kokkos::View<int*, ExecSpace> steps("passB_model_steps", npix);
    Kokkos::View<int*, ExecSpace> reason("passB_model_reason", npix);

    kpolaris::run_pass_b_model<ExecSpace>(pass_a, model, 2, image_i, image_q,
                                       image_u, image_v, closure_x, closure_k,
                                       final_null, frame_error, det_r, steps,
                                       reason);
    Kokkos::fence();

    auto h_i = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_i);
    auto h_q = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_q);
    auto h_u = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), image_u);
    auto h_closure_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), closure_x);
    auto h_closure_k = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), closure_k);
    auto h_final_null = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), final_null);
    auto h_frame_error = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), frame_error);
    auto h_det_r = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), det_r);
    auto h_reason = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), reason);

    int returned = 0;
    Real min_i = kpolaris::large_positive<Real>();
    Real max_i = Real(0);
    for (int p = 0; p < npix; ++p) {
        if (h_reason(p) == static_cast<int>(kpolaris::TerminationReason::reached_camera)) {
            returned += 1;
            if (!(h_i(p) > Real(0))) {
                throw std::runtime_error("passB model kernel produced non-positive I");
            }
            if (!(std::abs(h_q(p)) > Real(0) || std::abs(h_u(p)) > Real(0))) {
                throw std::runtime_error("passB model kernel produced zero linear polarization");
            }
            if (h_closure_x(p) > Real(5e-3) || h_closure_k(p) > Real(5e-3)) {
                throw std::runtime_error("passB model kernel closure too large");
            }
            if (std::abs(h_final_null(p)) > Real(5e-4)) {
                throw std::runtime_error("passB model kernel final null too large");
            }
            if (std::abs(h_frame_error(p)) > Real(1e-2)) {
                throw std::runtime_error("passB model kernel frame error too large");
            }
            if (h_det_r(p) < Real(0.5)) {
                throw std::runtime_error("passB model overlap determinant is not right-handed");
            }
            min_i = std::min(min_i, h_i(p));
            max_i = std::max(max_i, h_i(p));
        }
    }
    if (returned == 0) {
        throw std::runtime_error("passB model kernel had no returned rays");
    }
    if (!(max_i > min_i)) {
        throw std::runtime_error("passB radial model did not vary intensity");
    }
}


void test_emissivity_radius_quantiles() {
    const std::array<double, 4> intensity{100.0, 1.1, 1.005, 1.0};
    const auto value_at = [&](int i) { return intensity[static_cast<size_t>(i)]; };
    if (kpolaris::intensity_freeze_sample(4, 1.0, value_at) != 2 ||
        kpolaris::intensity_freeze_sample(4, 0.0, value_at) != -1) {
        throw std::runtime_error("intensity freeze used peak history instead of final I");
    }

    // A far-side emitter, a pericenter, and a near-side emitter in path order.
    const std::vector<std::pair<double, double>> samples{
        {20.0, 1.0}, {2.0, 8.0}, {10.0, 1.0}, {1000.0, 0.0}};
    const auto q = kpolaris::emissivity_radius_quantiles(samples);
    if (q != std::array<double, 3>{2.0, 2.0, 20.0}) {
        throw std::runtime_error("emissivity radius quantiles used path order");
    }
    auto reversed = samples;
    std::reverse(reversed.begin(), reversed.end());
    for (auto& sample : reversed) sample.second *= 1e-250;
    if (kpolaris::emissivity_radius_quantiles(reversed) != q ||
        kpolaris::emissivity_radius_quantiles({{3.0, 0.0}}) !=
            std::array<double, 3>{}) {
        throw std::runtime_error("radius quantiles failed scaling/empty-weight check");
    }
}

} // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    try {
        test_emissivity_radius_quantiles();
        test_pure_emission();
        test_absorption_emission();
        test_faraday_rotation();
        test_qu_rotation();
        test_basis_overlap();
        test_stokes_jones_basis_contract();
        test_adaptive_error_includes_screen_basis();
        test_toy_kernel();
        test_minkowski_metric_and_rk4();
        test_rk4_midpoint_state();
        test_double_pass_toy_kernel();
        test_kerr_schild_metric_inverse();
        test_kerr_boyer_lindquist_metric_inverse();
        test_kerr_schild_derivatives();
        test_fmks_metric_inverse_and_derivatives();
        test_fmks_specialized_connection_contract();
        test_kerr_schild_connection_symmetry();
        test_kerr_schild_specialized_connection_contract();
        test_kerr_schild_short_geodesic_null_control();
        test_kerr_geodesic_smoke_kernel();
        test_camera_initialization();
        test_affine_direction_equivalence();
        test_camera_parallel_plane_alignment();
        test_camera_pinhole_alignment();
        test_pinhole_camera_bl_spherical_ks_equivalence();
        test_pinhole_camera_fmks_cartesian_equivalence();
        test_fmks_cartesian_metric_equivalence();
        test_fmks_cartesian_transport_rhs_equivalence();
        test_grmhd_native_cartesian_vector_transform_metric_invariance();
        test_grmhd_fluid_state_fmks_cartesian_equivalence();
        test_adaptive_rk4_flat_step();
        test_metric_effective_min_step_default_semantics();
        test_metric_inner_surface_event_handling();
        test_adaptive_step_underflow_reason();
        test_metric_time_domain_event_handling();
        test_never_entered_outer_sphere_terminates();
        test_pass_a_single_center_ray();
        test_pass_a_kernel();
        test_pass_b_single_center_ray();
        test_pass_b_parallel_plane_screen_orientation();
        test_pass_b_kernel();
        test_pass_b_constant_model_matches_constant_path();
        test_radiation_model_dlambda_scales();
        test_riaf_fluid_four_velocity();
        test_riaf_mixed_velocity_is_timelike();
        test_thermal_synchrotron_coefficients();
        test_nonthermal_synchrotron_coefficients();
#if !KPOLARIS_USE_FLOAT
        test_symphony_fast_fit_checkpoints();
#endif
        test_riaf_model_coefficients();
        test_magnetized_torus_model_coefficients();
        test_pass_b_segment_model_single_ray();
        test_trace_image_geometry_schedule_consistency();
#if !KPOLARIS_USE_FLOAT
        test_pass_b_riaf_model_kernel();
#else
        std::cout << "Skipping the astrophysical RIAF image-kernel test in "
                     "unsupported main-float precision\n";
#endif
        test_pass_b_radial_model_kernel();
    } catch (...) {
        Kokkos::finalize();
        throw;
    }
    Kokkos::finalize();
    std::cout << "KPolaris core tests passed\n";
    return 0;
}
