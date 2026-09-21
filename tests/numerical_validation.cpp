#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include "KPolaris.hpp"

namespace {

using Real = kpolaris::DefaultReal;

static_assert(sizeof(Real) >= sizeof(double),
              "numerical validation requires a double main state");

struct GeodesicRow {
    Real step = Real(0);
    int steps = 0;
    Real state_error = Real(0);
    Real position_error = Real(0);
    Real wavevector_error = Real(0);
    Real null_error = Real(0);
    Real frame_error = Real(0);
    Real observed_order = Real(0);
};

struct RadiationRow {
    int steps = 0;
    Real error = Real(0);
    Real observed_order = Real(0);
};

struct AdaptiveGeodesicRow {
    Real tolerance = Real(0);
    int accepted_steps = 0;
    int calls = 0;
    Real state_error = Real(0);
    Real null_error = Real(0);
    Real frame_error = Real(0);
};

Real stokes_error(const kpolaris::Stokes<Real>& value,
                  const kpolaris::Stokes<Real>& reference) {
    Real error = Real(0);
    error = std::max(error, std::abs(value.I - reference.I));
    error = std::max(error, std::abs(value.Q - reference.Q));
    error = std::max(error, std::abs(value.U - reference.U));
    error = std::max(error, std::abs(value.V - reference.V));
    return error;
}

Real vec4_max_error(const kpolaris::Vec4<Real>& value,
                    const kpolaris::Vec4<Real>& reference) {
    Real error = Real(0);
    for (int mu = 0; mu < kpolaris::ndim; ++mu) {
        error = std::max(error, std::abs(value[mu] - reference[mu]));
    }
    return error;
}

kpolaris::Stokes<Real> integrate_transfer(
    kpolaris::Stokes<Real> stokes,
    const kpolaris::TransferCoeffs<Real>& coeffs,
    Real path_length,
    int steps) {
    const Real dlambda = path_length / Real(steps);
    for (int n = 0; n < steps; ++n) {
        kpolaris::semi_analytic_stokes_step(stokes, coeffs, dlambda);
    }
    return stokes;
}

kpolaris::Stokes<Real> transfer_rhs(
    const kpolaris::Stokes<Real>& stokes,
    const kpolaris::TransferCoeffs<Real>& coeffs) {
    kpolaris::Stokes<Real> derivative;
    derivative.I = coeffs.jI - coeffs.aI * stokes.I -
                   coeffs.aQ * stokes.Q - coeffs.aU * stokes.U -
                   coeffs.aV * stokes.V;
    derivative.Q = coeffs.jQ - coeffs.aI * stokes.Q -
                   coeffs.aQ * stokes.I +
                   coeffs.rU * stokes.V - coeffs.rV * stokes.U;
    derivative.U = coeffs.jU - coeffs.aI * stokes.U -
                   coeffs.aU * stokes.I +
                   coeffs.rV * stokes.Q - coeffs.rQ * stokes.V;
    derivative.V = coeffs.jV - coeffs.aI * stokes.V -
                   coeffs.aV * stokes.I +
                   coeffs.rQ * stokes.U - coeffs.rU * stokes.Q;
    return derivative;
}

kpolaris::Stokes<Real> stokes_axpy(
    const kpolaris::Stokes<Real>& base,
    Real scale,
    const kpolaris::Stokes<Real>& increment) {
    return {base.I + scale * increment.I,
            base.Q + scale * increment.Q,
            base.U + scale * increment.U,
            base.V + scale * increment.V};
}

kpolaris::Stokes<Real> integrate_transfer_rk4_reference(
    kpolaris::Stokes<Real> stokes,
    const kpolaris::TransferCoeffs<Real>& coeffs,
    Real path_length,
    int steps) {
    const Real h = path_length / Real(steps);
    for (int n = 0; n < steps; ++n) {
        const auto k1 = transfer_rhs(stokes, coeffs);
        const auto k2 = transfer_rhs(stokes_axpy(stokes, Real(0.5) * h, k1), coeffs);
        const auto k3 = transfer_rhs(stokes_axpy(stokes, Real(0.5) * h, k2), coeffs);
        const auto k4 = transfer_rhs(stokes_axpy(stokes, h, k3), coeffs);
        stokes.I += h / Real(6) * (k1.I + Real(2) * k2.I + Real(2) * k3.I + k4.I);
        stokes.Q += h / Real(6) * (k1.Q + Real(2) * k2.Q + Real(2) * k3.Q + k4.Q);
        stokes.U += h / Real(6) * (k1.U + Real(2) * k2.U + Real(2) * k3.U + k4.U);
        stokes.V += h / Real(6) * (k1.V + Real(2) * k2.V + Real(2) * k3.V + k4.V);
    }
    return stokes;
}

kpolaris::TransferCoeffs<Real> variable_transfer_coefficients(Real lambda) {
    kpolaris::TransferCoeffs<Real> coeffs;
    coeffs.jI = Real(0.7) + Real(0.2) * std::sin(Real(1.3) * lambda);
    coeffs.jQ = Real(-0.11) + Real(0.04) * std::cos(Real(0.8) * lambda);
    coeffs.jU = Real(0.06) + Real(0.03) * std::sin(Real(1.7) * lambda);
    coeffs.jV = Real(0.025) - Real(0.012) * std::cos(Real(1.1) * lambda);
    coeffs.aI = Real(0.55) + Real(0.08) * std::cos(Real(0.9) * lambda);
    coeffs.aQ = Real(0.10) + Real(0.025) * std::sin(Real(1.2) * lambda);
    coeffs.aU = Real(-0.07) + Real(0.018) * std::cos(Real(1.5) * lambda);
    coeffs.aV = Real(0.045) - Real(0.015) * std::sin(Real(0.7) * lambda);
    coeffs.rQ = Real(0.65) + Real(0.2) * std::sin(Real(0.6) * lambda);
    coeffs.rU = Real(-0.35) + Real(0.12) * std::cos(Real(1.4) * lambda);
    coeffs.rV = Real(0.95) - Real(0.17) * std::sin(Real(1.0) * lambda);
    return coeffs;
}

kpolaris::Stokes<Real> integrate_variable_transfer_rk4_reference(
    kpolaris::Stokes<Real> stokes,
    Real path_length,
    int steps) {
    const Real h = path_length / Real(steps);
    for (int n = 0; n < steps; ++n) {
        const Real lambda = Real(n) * h;
        const auto k1 = transfer_rhs(
            stokes, variable_transfer_coefficients(lambda));
        const auto k2 = transfer_rhs(
            stokes_axpy(stokes, Real(0.5) * h, k1),
            variable_transfer_coefficients(lambda + Real(0.5) * h));
        const auto k3 = transfer_rhs(
            stokes_axpy(stokes, Real(0.5) * h, k2),
            variable_transfer_coefficients(lambda + Real(0.5) * h));
        const auto k4 = transfer_rhs(
            stokes_axpy(stokes, h, k3),
            variable_transfer_coefficients(lambda + h));
        stokes.I += h / Real(6) * (k1.I + Real(2) * k2.I + Real(2) * k3.I + k4.I);
        stokes.Q += h / Real(6) * (k1.Q + Real(2) * k2.Q + Real(2) * k3.Q + k4.Q);
        stokes.U += h / Real(6) * (k1.U + Real(2) * k2.U + Real(2) * k3.U + k4.U);
        stokes.V += h / Real(6) * (k1.V + Real(2) * k2.V + Real(2) * k3.V + k4.V);
    }
    return stokes;
}

kpolaris::Stokes<Real> exact_faraday_rotation(
    const kpolaris::Stokes<Real>& initial,
    const kpolaris::TransferCoeffs<Real>& coeffs,
    Real path_length) {
    const kpolaris::Vec3<Real> p(initial.Q, initial.U, initial.V);
    const kpolaris::Vec3<Real> rho(coeffs.rQ, coeffs.rU, coeffs.rV);
    const Real rho_norm = kpolaris::norm(rho);
    if (rho_norm == Real(0)) {
        return initial;
    }
    const kpolaris::Vec3<Real> axis = rho * (Real(1) / rho_norm);
    const Real angle = rho_norm * path_length;
    const kpolaris::Vec3<Real> rotated =
        p * std::cos(angle) +
        axis * (kpolaris::dot(axis, p) * (Real(1) - std::cos(angle))) +
        kpolaris::cross(axis, p) * std::sin(angle);
    return {initial.I, rotated.x, rotated.y, rotated.z};
}

std::vector<RadiationRow> split_transfer_convergence() {
    kpolaris::TransferCoeffs<Real> coeffs;
    coeffs.jI = Real(0.8);
    coeffs.jQ = Real(-0.13);
    coeffs.jU = Real(0.09);
    coeffs.jV = Real(0.04);
    coeffs.aI = Real(0.6);
    coeffs.aQ = Real(0.17);
    coeffs.aU = Real(-0.11);
    coeffs.aV = Real(0.08);
    coeffs.rQ = Real(0.9);
    coeffs.rU = Real(-0.4);
    coeffs.rV = Real(1.3);
    const kpolaris::Stokes<Real> initial(
        Real(1.2), Real(0.18), Real(-0.07), Real(0.03));
    const Real path_length = Real(1.4);
    const auto reference = integrate_transfer_rk4_reference(
        initial, coeffs, path_length, 1 << 18);

    std::vector<RadiationRow> rows;
    Real previous_error = Real(0);
    for (const int steps : {1, 2, 4, 8, 16, 32, 64}) {
        const auto value = integrate_transfer(initial, coeffs, path_length, steps);
        const Real error = stokes_error(value, reference);
        RadiationRow row;
        row.steps = steps;
        row.error = error;
        if (previous_error > Real(0) && error > Real(0)) {
            row.observed_order = std::log(previous_error / error) / std::log(Real(2));
        }
        rows.push_back(row);
        previous_error = error;
    }
    return rows;
}

std::vector<RadiationRow> variable_transfer_convergence() {
    const kpolaris::Stokes<Real> initial(
        Real(1.05), Real(0.16), Real(-0.09), Real(0.035));
    const Real path_length = Real(1.6);
    const auto reference = integrate_variable_transfer_rk4_reference(
        initial, path_length, 1 << 18);

    std::vector<RadiationRow> rows;
    Real previous_error = Real(0);
    for (const int steps : {1, 2, 4, 8, 16, 32, 64, 128}) {
        auto value = initial;
        const Real h = path_length / Real(steps);
        for (int n = 0; n < steps; ++n) {
            const Real midpoint = (Real(n) + Real(0.5)) * h;
            const auto coeffs = variable_transfer_coefficients(midpoint);
            kpolaris::semi_analytic_stokes_step(value, coeffs, h);
        }
        const Real error = stokes_error(value, reference);
        RadiationRow row;
        row.steps = steps;
        row.error = error;
        if (previous_error > Real(0) && error > Real(0)) {
            row.observed_order = std::log(previous_error / error) / std::log(Real(2));
        }
        rows.push_back(row);
        previous_error = error;
    }
    return rows;
}

template<class Metric>
std::vector<GeodesicRow> geodesic_convergence(const Metric& metric) {
    kpolaris::CameraParams<Real> camera;
    camera.nx = 5;
    camera.ny = 5;
    camera.radius = Real(20);
    camera.inclination = Real(1.1);
    camera.fov = Real(0.18);
    camera.model = kpolaris::CameraModel::Pinhole;
    const auto initial = kpolaris::initialize_camera_ray(metric, 6, camera);

    const Real affine_length = Real(8);
    constexpr int reference_steps = 1 << 16;
    const auto reference = kpolaris::integrate_fixed_rk4(
        metric, initial, -affine_length / Real(reference_steps), reference_steps);

    std::vector<GeodesicRow> rows;
    Real previous_error = Real(0);
    for (const int steps : {16, 32, 64, 128, 256}) {
        const Real step = affine_length / Real(steps);
        const auto value = kpolaris::integrate_fixed_rk4(metric, initial, -step, steps);
        GeodesicRow row;
        row.step = step;
        row.steps = steps;
        row.state_error = kpolaris::state_error_norm(value, reference);
        row.position_error = vec4_max_error(value.x, reference.x);
        row.wavevector_error = vec4_max_error(value.k, reference.k);
        row.null_error = std::abs(metric.dot(value.x, value.k, value.k));
        row.frame_error = kpolaris::max_frame_error(
            kpolaris::frame_errors(metric, value));
        if (previous_error > Real(0) && row.state_error > Real(0)) {
            row.observed_order =
                std::log(previous_error / row.state_error) / std::log(Real(2));
        }
        rows.push_back(row);
        previous_error = row.state_error;
    }
    return rows;
}

template<class Metric>
std::vector<AdaptiveGeodesicRow> adaptive_geodesic_convergence(
    const Metric& metric) {
    kpolaris::CameraParams<Real> camera;
    camera.nx = 5;
    camera.ny = 5;
    camera.radius = Real(20);
    camera.inclination = Real(1.1);
    camera.fov = Real(0.18);
    camera.model = kpolaris::CameraModel::Pinhole;
    const auto initial = kpolaris::initialize_camera_ray(metric, 6, camera);

    const Real affine_length = Real(8);
    constexpr int reference_steps = 1 << 16;
    const auto reference = kpolaris::integrate_fixed_rk4(
        metric, initial, -affine_length / Real(reference_steps), reference_steps);

    std::vector<AdaptiveGeodesicRow> rows;
    for (const Real tolerance :
         {Real(1e-4), Real(1e-6), Real(1e-8), Real(1e-10), Real(1e-12)}) {
        kpolaris::AdaptiveRK4Control<Real> control;
        control.tolerance = tolerance;
        control.min_step = Real(1e-12);
        control.max_step = Real(1);
        control.max_attempts = 20;
        auto state = initial;
        Real lambda = Real(0);
        Real next_h = control.max_step;
        AdaptiveGeodesicRow row;
        row.tolerance = tolerance;
        while (lambda < affine_length && row.calls < 1000000) {
            const Real remaining = affine_length - lambda;
            const Real requested_h = std::min(next_h, remaining);
            const auto step = kpolaris::adaptive_rk4_step(
                metric, state, -requested_h, control);
            row.calls += 1;
            if (!step.accepted || !(step.used_h < Real(0))) {
                next_h = std::abs(step.next_h);
                continue;
            }
            state = step.state;
            lambda += std::abs(step.used_h);
            next_h = std::abs(step.next_h);
            row.accepted_steps += 1;
        }
        if (lambda < affine_length) {
            throw std::runtime_error("adaptive geodesic validation did not finish");
        }
        row.state_error = kpolaris::state_error_norm(state, reference);
        row.null_error = std::abs(metric.dot(state.x, state.k, state.k));
        row.frame_error = kpolaris::max_frame_error(
            kpolaris::frame_errors(metric, state));
        rows.push_back(row);
    }
    return rows;
}

void write_radiation_rows(std::ostream& out,
                          const std::vector<RadiationRow>& rows) {
    out << "[\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        out << "        {\"steps\": " << row.steps
            << ", \"error\": " << row.error
            << ", \"observed_order\": " << row.observed_order << "}";
        out << (i + 1 == rows.size() ? "\n" : ",\n");
    }
    out << "      ]";
}

void write_geodesic_rows(std::ostream& out,
                         const std::vector<GeodesicRow>& rows) {
    out << "[\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        out << "        {\"step\": " << row.step
            << ", \"steps\": " << row.steps
            << ", \"state_error\": " << row.state_error
            << ", \"position_error\": " << row.position_error
            << ", \"wavevector_error\": " << row.wavevector_error
            << ", \"null_error\": " << row.null_error
            << ", \"frame_error\": " << row.frame_error
            << ", \"observed_order\": " << row.observed_order << "}";
        out << (i + 1 == rows.size() ? "\n" : ",\n");
    }
    out << "      ]";
}

void write_adaptive_geodesic_rows(
    std::ostream& out,
    const std::vector<AdaptiveGeodesicRow>& rows) {
    out << "[\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto& row = rows[i];
        out << "        {\"tolerance\": " << row.tolerance
            << ", \"accepted_steps\": " << row.accepted_steps
            << ", \"calls\": " << row.calls
            << ", \"state_error\": " << row.state_error
            << ", \"null_error\": " << row.null_error
            << ", \"frame_error\": " << row.frame_error << "}";
        out << (i + 1 == rows.size() ? "\n" : ",\n");
    }
    out << "      ]";
}

void require_validation_thresholds(
    Real pure_emission_error,
    Real absorption_error,
    Real faraday_error,
    const std::vector<RadiationRow>& split_rows,
    const std::vector<RadiationRow>& variable_rows,
    const std::vector<GeodesicRow>& schwarzschild_rows,
    const std::vector<GeodesicRow>& kerr_rows,
    const std::vector<AdaptiveGeodesicRow>& schwarzschild_adaptive,
    const std::vector<AdaptiveGeodesicRow>& kerr_adaptive) {
    const Real analytic_limit = Real(2e-12);
    if (pure_emission_error > analytic_limit ||
        absorption_error > analytic_limit || faraday_error > analytic_limit) {
        throw std::runtime_error("constant-coefficient analytic transfer validation failed");
    }
    for (std::size_t i = 2; i < 6; ++i) {
        if (split_rows[i].observed_order < Real(1.95)) {
            throw std::runtime_error("Strang transfer convergence fell below second order");
        }
    }
    for (std::size_t i = 2; i + 1 < variable_rows.size(); ++i) {
        if (variable_rows[i].error >= variable_rows[i - 1].error ||
            variable_rows[i].observed_order < Real(1.8)) {
            throw std::runtime_error(
                "variable-coefficient transfer convergence fell below second order");
        }
    }
    const auto require_rk4_order = [](const std::vector<GeodesicRow>& rows,
                                      const char* label) {
        for (std::size_t i = 1; i < 4; ++i) {
            if (rows[i].observed_order < Real(3.7)) {
                throw std::runtime_error(std::string(label) +
                                         " geodesic convergence fell below fourth order");
            }
        }
    };
    require_rk4_order(schwarzschild_rows, "Schwarzschild");
    require_rk4_order(kerr_rows, "Kerr");
    const auto require_adaptive_monotonic = [](
        const std::vector<AdaptiveGeodesicRow>& rows,
        const char* label) {
        for (std::size_t i = 1; i < rows.size(); ++i) {
            if (rows[i].state_error > rows[i - 1].state_error * Real(1.05)) {
                throw std::runtime_error(std::string(label) +
                                         " adaptive error was not monotonic");
            }
        }
    };
    require_adaptive_monotonic(schwarzschild_adaptive, "Schwarzschild");
    require_adaptive_monotonic(kerr_adaptive, "Kerr");
}

} // namespace

int main(int argc, char** argv) {
    std::string output_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        const std::string prefix = "--output=";
        if (arg.rfind(prefix, 0) == 0) {
            output_path = arg.substr(prefix.size());
        } else {
            std::cerr << "unknown argument: " << arg << "\n";
            return 2;
        }
    }

    Kokkos::initialize(argc, argv);
    try {
        kpolaris::TransferCoeffs<Real> pure_coeffs;
        pure_coeffs.jI = Real(2.0);
        pure_coeffs.jQ = Real(-0.4);
        pure_coeffs.jU = Real(0.3);
        pure_coeffs.jV = Real(0.1);
        const kpolaris::Stokes<Real> pure_initial(
            Real(0.7), Real(0.1), Real(-0.2), Real(0.05));
        const Real pure_length = Real(1.25);
        const auto pure_value = integrate_transfer(
            pure_initial, pure_coeffs, pure_length, 7);
        const kpolaris::Stokes<Real> pure_exact(
            pure_initial.I + pure_coeffs.jI * pure_length,
            pure_initial.Q + pure_coeffs.jQ * pure_length,
            pure_initial.U + pure_coeffs.jU * pure_length,
            pure_initial.V + pure_coeffs.jV * pure_length);
        const Real pure_emission_error = stokes_error(pure_value, pure_exact);

        kpolaris::TransferCoeffs<Real> absorption_coeffs;
        absorption_coeffs.jI = Real(1.1);
        absorption_coeffs.jQ = Real(-0.2);
        absorption_coeffs.jU = Real(0.07);
        absorption_coeffs.jV = Real(0.03);
        absorption_coeffs.aI = Real(0.7);
        const kpolaris::Stokes<Real> absorption_initial(
            Real(1.3), Real(0.2), Real(-0.1), Real(0.04));
        const Real absorption_length = Real(1.4);
        const auto absorption_value = integrate_transfer(
            absorption_initial, absorption_coeffs, absorption_length, 11);
        const Real attenuation = std::exp(-absorption_coeffs.aI * absorption_length);
        const Real source_factor =
            (Real(1) - attenuation) / absorption_coeffs.aI;
        const kpolaris::Stokes<Real> absorption_exact(
            attenuation * absorption_initial.I + source_factor * absorption_coeffs.jI,
            attenuation * absorption_initial.Q + source_factor * absorption_coeffs.jQ,
            attenuation * absorption_initial.U + source_factor * absorption_coeffs.jU,
            attenuation * absorption_initial.V + source_factor * absorption_coeffs.jV);
        const Real absorption_error = stokes_error(absorption_value, absorption_exact);

        kpolaris::TransferCoeffs<Real> faraday_coeffs;
        faraday_coeffs.rQ = Real(0.3);
        faraday_coeffs.rU = Real(-0.5);
        faraday_coeffs.rV = Real(1.1);
        const kpolaris::Stokes<Real> faraday_initial(
            Real(0.9), Real(0.2), Real(-0.3), Real(0.15));
        const Real faraday_length = Real(1.7);
        const auto faraday_value = integrate_transfer(
            faraday_initial, faraday_coeffs, faraday_length, 13);
        const auto faraday_exact = exact_faraday_rotation(
            faraday_initial, faraday_coeffs, faraday_length);
        const Real faraday_error = stokes_error(faraday_value, faraday_exact);

        const auto split_rows = split_transfer_convergence();
        const auto variable_rows = variable_transfer_convergence();
        const auto schwarzschild_rows = geodesic_convergence(
            kpolaris::KerrSchildInMetric<Real>(Real(1), Real(0)));
        const auto kerr_rows = geodesic_convergence(
            kpolaris::KerrSchildInMetric<Real>(Real(1), Real(0.7)));
        const auto schwarzschild_adaptive = adaptive_geodesic_convergence(
            kpolaris::KerrSchildInMetric<Real>(Real(1), Real(0)));
        const auto kerr_adaptive = adaptive_geodesic_convergence(
            kpolaris::KerrSchildInMetric<Real>(Real(1), Real(0.7)));

        require_validation_thresholds(
            pure_emission_error, absorption_error, faraday_error,
            split_rows, variable_rows, schwarzschild_rows, kerr_rows,
            schwarzschild_adaptive, kerr_adaptive);

        std::unique_ptr<std::ofstream> file;
        std::ostream* output = &std::cout;
        if (!output_path.empty()) {
            file = std::make_unique<std::ofstream>(output_path);
            if (!*file) {
                throw std::runtime_error("failed to open numerical validation output");
            }
            output = file.get();
        }
        auto& out = *output;
        out << std::setprecision(17);
        out << "{\n"
            << "  \"schema\": \"kpolaris_numerical_validation\",\n"
            << "  \"schema_version\": 1,\n"
            << "  \"precision\": \"double\",\n"
            << "  \"radiation\": {\n"
            << "    \"pure_emission_max_abs_error\": " << pure_emission_error << ",\n"
            << "    \"scalar_absorption_emission_max_abs_error\": " << absorption_error << ",\n"
            << "    \"faraday_rotation_max_abs_error\": " << faraday_error << ",\n"
            << "    \"noncommuting_split_convergence\": ";
        write_radiation_rows(out, split_rows);
        out << ",\n    \"variable_coefficient_midpoint_convergence\": ";
        write_radiation_rows(out, variable_rows);
        out << "\n  },\n"
            << "  \"geodesic\": {\n"
            << "    \"affine_length\": 8,\n"
            << "    \"reference_steps\": 65536,\n"
            << "    \"schwarzschild_cartesian_ks\": ";
        write_geodesic_rows(out, schwarzschild_rows);
        out << ",\n    \"kerr_a0p7_cartesian_ks\": ";
        write_geodesic_rows(out, kerr_rows);
        out << ",\n    \"schwarzschild_adaptive\": ";
        write_adaptive_geodesic_rows(out, schwarzschild_adaptive);
        out << ",\n    \"kerr_a0p7_adaptive\": ";
        write_adaptive_geodesic_rows(out, kerr_adaptive);
        out << "\n  }\n}\n";
    } catch (const std::exception& error) {
        Kokkos::finalize();
        std::cerr << error.what() << "\n";
        return 1;
    }
    Kokkos::finalize();
    return 0;
}
