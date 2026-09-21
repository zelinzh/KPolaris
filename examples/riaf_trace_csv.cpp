#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <Kokkos_Core.hpp>

#include "KPolaris.hpp"

namespace {
using Real = kpolaris::DefaultReal;

std::string value_after_equals(const std::string& arg) {
    const auto pos = arg.find('=');
    return pos == std::string::npos ? std::string() : arg.substr(pos + 1);
}

struct Options {
    int ix = 13;
    int iy = 16;
    int nx = 32;
    int ny = 32;
    Real radius = Real(1000);
    Real inclination = Real(85) * Real(3.141592653589793238462643383279502884) / Real(180);
    Real xspan = Real(19.554708160431073);
    Real spin = Real(0.9375);
    Real freq = Real(230.0e9);
    Real step = Real(0.025);
    Real min_step = Real(1e-5);
    Real max_step = Real(1);
    Real adaptive_tolerance = Real(1e-7);
    int adaptive = 1;
    int max_steps = 200000;
    int substeps = 8;
    Real riaf_r_min = Real(1);
    Real riaf_r_max = Real(100);
    int emission_type = 1;
    std::string output = "kpolaris_riaf_trace.csv";
    std::string coordinate = "cartesian_ks";
};

Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--help") {
            std::cout << "Usage: kpolaris_riaf_trace_csv [--ix=N] [--iy=N] [--nx=N] [--ny=N] [--coordinate=cartesian_ks|boyer_lindquist] [--output=file.csv] ...\n";
            std::exit(0);
        } else if (arg.rfind("--ix=", 0) == 0) opt.ix = std::stoi(value_after_equals(arg));
        else if (arg.rfind("--iy=", 0) == 0) opt.iy = std::stoi(value_after_equals(arg));
        else if (arg.rfind("--nx=", 0) == 0) opt.nx = std::stoi(value_after_equals(arg));
        else if (arg.rfind("--ny=", 0) == 0) opt.ny = std::stoi(value_after_equals(arg));
        else if (arg.rfind("--radius=", 0) == 0) opt.radius = Real(std::stod(value_after_equals(arg)));
        else if (arg.rfind("--inclination_deg=", 0) == 0) opt.inclination = Real(std::stod(value_after_equals(arg))) * Real(3.141592653589793238462643383279502884) / Real(180);
        else if (arg.rfind("--xspan=", 0) == 0) opt.xspan = Real(std::stod(value_after_equals(arg)));
        else if (arg.rfind("--spin=", 0) == 0) opt.spin = Real(std::stod(value_after_equals(arg)));
        else if (arg.rfind("--freq=", 0) == 0) opt.freq = Real(std::stod(value_after_equals(arg)));
        else if (arg.rfind("--step=", 0) == 0) opt.step = Real(std::stod(value_after_equals(arg)));
        else if (arg.rfind("--min_step=", 0) == 0) opt.min_step = Real(std::stod(value_after_equals(arg)));
        else if (arg.rfind("--max_step=", 0) == 0) opt.max_step = Real(std::stod(value_after_equals(arg)));
        else if (arg.rfind("--adaptive_tolerance=", 0) == 0) opt.adaptive_tolerance = Real(std::stod(value_after_equals(arg)));
        else if (arg.rfind("--adaptive=", 0) == 0) opt.adaptive = std::stoi(value_after_equals(arg));
        else if (arg.rfind("--max_steps=", 0) == 0) opt.max_steps = std::stoi(value_after_equals(arg));
        else if (arg.rfind("--substeps=", 0) == 0) opt.substeps = std::stoi(value_after_equals(arg));
        else if (arg.rfind("--riaf_r_min=", 0) == 0) opt.riaf_r_min = Real(std::stod(value_after_equals(arg)));
        else if (arg.rfind("--riaf_r_max=", 0) == 0) opt.riaf_r_max = Real(std::stod(value_after_equals(arg)));
        else if (arg.rfind("--emission_type=", 0) == 0) opt.emission_type = std::stoi(value_after_equals(arg));
        else if (arg.rfind("--coordinate=", 0) == 0) opt.coordinate = value_after_equals(arg);
        else if (arg.rfind("--output=", 0) == 0) opt.output = value_after_equals(arg);
        else throw std::runtime_error("unknown option: " + arg);
    }
    return opt;
}

template<class Metric, class Model>
void write_sample(std::ofstream& out, const Metric& metric, const Model& model,
                  const kpolaris::TransportState<Real>& state, const kpolaris::Stokes<Real>& stokes,
                  Real lambda, Real dlambda, int step, int substep) {
    Real r = Real(0), th = Real(0), cp = Real(1), sp = Real(0);
    model.bl_coordinates_for_metric(metric, state.x, r, th, cp, sp);
    Real phi = std::atan2(static_cast<double>(sp), static_cast<double>(cp));
    if (phi < Real(0)) phi += Real(2) * Real(3.141592653589793238462643383279502884);
    const Real ne_norm = model.density_profile_bl(r, th);
    const Real ne_cgs = ne_norm * model.ne_unit;
    const Real thetae = model.thetae_profile(r);
    const Real b_cgs = model.magnetic_field_cgs(ne_cgs, r);
    const auto ucon = model.fluid_four_velocity(metric, state.x, r);
    const auto ucov = model.lower_vector(metric, state.x, ucon);
    Real uk = Real(0);
    for (int mu = 0; mu < kpolaris::ndim; ++mu) uk += ucov[mu] * state.k[mu];
    const Real nu_scale = kpolaris::max_val(kpolaris::abs_val(-uk), model.min_frequency_scale);
    const auto bcon = model.magnetic_unit_four_vector(metric, state.x, ucon);
    const auto bcov = model.lower_vector(metric, state.x, bcon);
    Real kdotb = Real(0);
    for (int mu = 0; mu < kpolaris::ndim; ++mu) kdotb += state.k[mu] * bcov[mu];
    const Real cos_theta = kpolaris::clamp(kdotb / kpolaris::max_val(nu_scale, Real(1e-30)), Real(-1), Real(1));
    const Real theta_bk = std::acos(static_cast<double>(cos_theta));
    const auto coeffs = model.coefficients(metric, state, Real(0));
    const Real b1 = model.screen_inner_product(metric, state.x, ucon, state.k, state.e1, bcon);
    const Real b2 = model.screen_inner_product(metric, state.x, ucon, state.k, state.e2, bcon);
    const Real bproj2 = b1 * b1 + b2 * b2;
    const Real cos2chi = bproj2 > Real(0) ? (b2 * b2 - b1 * b1) / bproj2 : Real(0);
    const Real sin2chi = bproj2 > Real(0) ? -Real(2) * b1 * b2 / bproj2 : Real(0);

    out << step << ',' << substep << ',' << lambda << ',' << dlambda << ','
        << state.x[0] << ',' << state.x[1] << ',' << state.x[2] << ',' << state.x[3] << ','
        << state.k[0] << ',' << state.k[1] << ',' << state.k[2] << ',' << state.k[3] << ','
        << state.e1[0] << ',' << state.e1[1] << ',' << state.e1[2] << ',' << state.e1[3] << ','
        << state.e2[0] << ',' << state.e2[1] << ',' << state.e2[2] << ',' << state.e2[3] << ','
        << r << ',' << th << ',' << phi << ',' << ne_cgs << ',' << thetae << ',' << b_cgs << ','
        << nu_scale * model.freq_cgs << ',' << theta_bk << ','
        << b1 << ',' << b2 << ',' << cos2chi << ',' << sin2chi << ','
        << coeffs.jI << ',' << coeffs.jQ << ',' << coeffs.jU << ',' << coeffs.jV << ','
        << coeffs.aI << ',' << coeffs.aQ << ',' << coeffs.aU << ',' << coeffs.aV << ','
        << coeffs.rQ << ',' << coeffs.rU << ',' << coeffs.rV << ','
        << stokes.I << ',' << stokes.Q << ',' << stokes.U << ',' << stokes.V << '\n';
}

} // namespace

int main(int argc, char** argv) {
    Kokkos::initialize(argc, argv);
    try {
        const Options opt = parse_options(argc, argv);
        kpolaris::PassAParams<Real> params;
        params.camera.nx = opt.nx;
        params.camera.ny = opt.ny;
        params.camera.radius = opt.radius;
        params.camera.inclination = opt.inclination;
        params.camera.fov = Real(2) * opt.xspan / opt.radius;
        params.camera.x_offset = Real(-0.01) / Real(opt.nx) * params.camera.fov;
        params.camera.model = kpolaris::CameraModel::Pinhole;
        params.mass = Real(1);
        params.spin = opt.spin;
        params.step = opt.step;
        params.min_step = opt.min_step;
        params.max_step = opt.max_step;
        params.adaptive_tolerance = opt.adaptive_tolerance;
        params.adaptive = opt.adaptive;
        params.max_steps = opt.max_steps;
        const bool use_bl_coordinate = opt.coordinate == "boyer_lindquist" || opt.coordinate == "boyer-lindquist" || opt.coordinate == "bl";
        const bool use_ks_coordinate = opt.coordinate == "cartesian_ks" || opt.coordinate == "cartesian-kerr-schild" || opt.coordinate == "ks";
        if (!use_bl_coordinate && !use_ks_coordinate) {
            throw std::runtime_error("unknown coordinate: " + opt.coordinate);
        }
        params.coordinate_system = use_bl_coordinate ? kpolaris::CoordinateSystem::BoyerLindquist
                                                     : kpolaris::CoordinateSystem::CartesianKS;
        const Real horizon = Real(1) + std::sqrt(std::max<Real>(Real(0), Real(1) - opt.spin * opt.spin));
        params.inner_radius = horizon * Real(1.05);
        params.outer_radius = opt.riaf_r_max * Real(1.15);

        kpolaris::RIAFAnalyticRadiationModel<Real> model;
        model.freq_cgs = opt.freq;
        model.r_min = opt.riaf_r_min;
        model.r_max = opt.riaf_r_max;
        model.nth0 = Real(1);
        model.Te0 = Real(1);
        model.disk_h = Real(0.1);
        model.pow_nth = Real(-1.1);
        model.pow_T = Real(-0.84);
        model.ne_unit = Real(3.0e7);
        model.te_unit = Real(3.0e11);
        model.mbh_solar = Real(4.3e6);
        model.keplerian_factor = Real(1);
        model.infall_factor = Real(0);
        model.emission_type = opt.emission_type;

        auto run_trace = [&](const auto& metric) {
        const int pixel = opt.iy * opt.nx + opt.ix;
        const auto camera_state = kpolaris::initialize_camera_ray(metric, pixel, params.camera);
        auto state = camera_state;

        kpolaris::AdaptiveRK4Control<Real> control;
        control.tolerance = params.adaptive_tolerance;
        control.min_step = params.min_step;
        control.max_step = params.max_step;
        Real h_current = params.step;
        bool entered = false;
        int entered_steps = 0;
        kpolaris::PassAResult<Real> pass;
        pass.state = state;
        pass.min_radius = metric.build_work(state.x).r;
        pass.reason = kpolaris::TerminationReason::max_steps;
        for (int n = 0; n < params.max_steps; ++n) {
            const Real r = metric.build_work(state.x).r;
            pass.min_radius = kpolaris::min_val(pass.min_radius, r);
            if (!entered && r <= params.outer_radius) {
                entered = true;
                entered_steps = 0;
            } else if (entered) {
                if (r <= params.inner_radius) { pass.reason = kpolaris::TerminationReason::reached_inner_boundary; break; }
                if (entered_steps > 0 && r >= params.outer_radius) { pass.reason = kpolaris::TerminationReason::escaped_domain; break; }
            }
            Real used_h = h_current;
            if (params.adaptive) {
                const auto step = kpolaris::adaptive_rk4_step(metric, state, used_h, control);
                state = step.state;
                used_h = step.used_h;
                h_current = step.next_h;
            } else {
                state = kpolaris::rk4_step(metric, state, used_h);
            }
            pass.path_length += kpolaris::abs_val(used_h);
            pass.steps += 1;
            if (entered) entered_steps += 1;
        }

        for (int mu = 0; mu < kpolaris::ndim; ++mu) state.k[mu] = -state.k[mu];
        // Match the image-production Pass B path: Pass A has already
        // parallel-transported the camera screen basis to this endpoint, so
        // reversing k changes only the propagation direction. Reinitializing
        // e1/e2 here would trace a different polarization convention.

        std::ofstream out(opt.output);
        out << std::setprecision(17);
        out << "# ix," << opt.ix << "\n# iy," << opt.iy << "\n# pass_steps," << pass.steps << "\n# path_length," << pass.path_length << "\n";
        out << "step,substep,lambda,dlambda,x0,x1,x2,x3,k0,k1,k2,k3,e10,e11,e12,e13,e20,e21,e22,e23,r,theta,phi,ne_cgs,thetae,b_cgs,nu_hz,b_k_angle,b1,b2,cos2chi,sin2chi,jI,jQ,jU,jV,aI,aQ,aU,aV,rQ,rU,rV,SI,SQ,SU,SV\n";
        kpolaris::Stokes<Real> stokes;
        Real lambda = Real(0);
        Real remaining = pass.path_length;
        h_current = params.step;
        int out_step = 0;
        const int safe_substeps = opt.substeps > 0 ? opt.substeps : 1;
        while (remaining > kpolaris::max_val(params.min_step, Real(1e-12)) && out_step < params.max_steps) {
            Real h = kpolaris::min_val(h_current, remaining);
            if (params.adaptive) {
                const auto proposed = kpolaris::adaptive_rk4_step(metric, state, h, control);
                h = kpolaris::abs_val(proposed.used_h);
                h_current = kpolaris::abs_val(proposed.next_h);
                h = kpolaris::min_val(h, remaining);
            }
            const Real sub_h = h / Real(safe_substeps);
            for (int s = 0; s < safe_substeps; ++s) {
                const auto sample = kpolaris::rk4_midpoint_state(metric, state, sub_h);
                const auto coeffs = model.coefficients(metric, sample, (Real(s) + Real(0.5)) / Real(safe_substeps));
                write_sample(out, metric, model, sample, stokes, lambda + (Real(s) + Real(0.5)) * sub_h,
                             kpolaris::abs_val(sub_h) * model.dlambda_scale(), out_step, s);
                kpolaris::semi_analytic_stokes_step(stokes, coeffs, kpolaris::abs_val(sub_h) * model.dlambda_scale());
                state = kpolaris::rk4_step(metric, state, sub_h);
            }
            lambda += h;
            remaining -= h;
            out_step += 1;
        }
        std::cerr << "wrote " << opt.output << " samples=" << out_step * safe_substeps << "\n";
        };
        if (use_bl_coordinate) {
            run_trace(kpolaris::KerrBoyerLindquistMetric<Real>(params.mass, params.spin));
        } else {
            run_trace(kpolaris::KerrSchildInMetric<Real>(params.mass, params.spin));
        }
    } catch (...) {
        Kokkos::finalize();
        throw;
    }
    Kokkos::finalize();
    return 0;
}
