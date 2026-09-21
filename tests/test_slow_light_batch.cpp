#include <iostream>
#include <map>
#include <atomic>
#include <thread>
#include <memory>
#include "image/slow_light_batch_driver.hpp"
#include "geometry/kerr_boyer_lindquist.hpp"
#include "geometry/minkowski.hpp"
#include "image_batch_checks.hpp"

struct Model {
    using Real = kpolaris::DefaultReal;
    Real freq_cgs = 230e9;
    Real time = 0;
    template<class Metric>
    KPOLARIS_INLINE kpolaris::TransferCoeffs<Real> coefficients(
        const Metric&, const kpolaris::TransportState<Real>&, Real,
        const kpolaris::PlasmaPerturbation<Real>& perturbation = {}) const {
        kpolaris::TransferCoeffs<Real> c;
        c.jI = (Real(.1) + Real(.0001)*time*time) * freq_cgs / Real(230e9);
        c.jQ = Real(.01); c.jU = Real(.005); c.jV = Real(.001);
        c.aI = Real(.02); c.rV = Real(.03); c.rQ = Real(.01);
        // A synthetic, explicitly parameter-dependent transfer problem checks
        // that responses/rerun channels survive every block boundary.
        if (perturbation.parameter != kpolaris::PlasmaParameter::none) {
            const Real factor = Kokkos::exp(perturbation.log_scale);
            c.jI *= factor; c.jQ *= factor; c.aI *= factor; c.rV *= factor;
        }
        return c;
    }
    template<class Metric>
    KPOLARIS_INLINE int fluid_state(const Metric&, const kpolaris::TransportState<Real>&,
        Real& rho, Real& uu, kpolaris::Vec4<Real>& u, kpolaris::Vec4<Real>& b,
        Real& ne, Real& theta, Real& B, Real& beta, Real& sigma) const {
        rho=uu=ne=theta=B=beta=sigma=1; u[0]=1; b[1]=1; return 1;
    }
    KPOLARIS_INLINE Real dlambda_scale() const { return Real(230e9) / freq_cgs; }
    template<class Metric>
    KPOLARIS_INLINE void bl_coordinates_for_metric(
        const Metric&, const kpolaris::Vec4<Real>& x, Real& r, Real& th, Real& cp, Real& sp) const {
        r=x[1]; th=x[2]; cp=cos(x[3]); sp=sin(x[3]);
    }
};

struct StagedLoader {
    using slow_light_prefetch_result_type = double;
    std::thread::id caller = std::this_thread::get_id();
    double fail_time = 1e10;
    slow_light_prefetch_result_type prefetch(const std::string& path) const {
        const double t = std::stod(path);
        if (t == fail_time) throw std::runtime_error("injected input failure");
        return t;
    }
    Model materialize(double&& t) const {
        if (std::this_thread::get_id() != caller)
            throw std::runtime_error("device materialization escaped the calling thread");
        return Model{230e9, t};
    }
    Model load(const std::string& path) const { return materialize(prefetch(path)); }
};

void require_same_analysis(const ImageHostData& a, const ImageHostData& b) {
    if (a.has_analysis != b.has_analysis) throw std::runtime_error("analysis flag mismatch");
    if (a.response_data != b.response_data) throw std::runtime_error("analysis response_data mismatch");
    if (a.radiating_path_length != b.radiating_path_length) throw std::runtime_error("analysis radiating_path_length mismatch");
    if (a.emission_weight != b.emission_weight) throw std::runtime_error("analysis emission_weight mismatch");
    if (a.emission_weighted_radius != b.emission_weighted_radius) throw std::runtime_error("analysis emission_weighted_radius mismatch");
    if (a.emission_weighted_optical_depth_to_camera != b.emission_weighted_optical_depth_to_camera) throw std::runtime_error("analysis emission_weighted_optical_depth_to_camera mismatch");
    if (a.absorption_depth != b.absorption_depth) throw std::runtime_error("analysis absorption_depth mismatch");
    if (a.absorption_operator_depth != b.absorption_operator_depth) throw std::runtime_error("analysis absorption_operator_depth mismatch");
    if (a.faraday_rotation_depth != b.faraday_rotation_depth) throw std::runtime_error("analysis faraday_rotation_depth mismatch");
    if (a.faraday_conversion_depth != b.faraday_conversion_depth) throw std::runtime_error("analysis faraday_conversion_depth mismatch");
    if (a.faraday_operator_depth != b.faraday_operator_depth) throw std::runtime_error("analysis faraday_operator_depth mismatch");
    if (a.dominant_emission_radius != b.dominant_emission_radius) throw std::runtime_error("analysis dominant_emission_radius mismatch");
    if (a.dominant_emission_region != b.dominant_emission_region) throw std::runtime_error("analysis dominant_emission_region mismatch");
    if (a.dominant_ne_cgs != b.dominant_ne_cgs) throw std::runtime_error("analysis dominant_ne_cgs mismatch");
    if (a.dominant_thetae != b.dominant_thetae) throw std::runtime_error("analysis dominant_thetae mismatch");
    if (a.dominant_b_cgs != b.dominant_b_cgs) throw std::runtime_error("analysis dominant_b_cgs mismatch");
    if (a.dominant_beta != b.dominant_beta) throw std::runtime_error("analysis dominant_beta mismatch");
    if (a.dominant_sigma != b.dominant_sigma) throw std::runtime_error("analysis dominant_sigma mismatch");
    if (a.emission_weighted_ne_cgs != b.emission_weighted_ne_cgs) throw std::runtime_error("analysis emission_weighted_ne_cgs mismatch");
    if (a.emission_weighted_thetae != b.emission_weighted_thetae) throw std::runtime_error("analysis emission_weighted_thetae mismatch");
    if (a.emission_weighted_b_cgs != b.emission_weighted_b_cgs) throw std::runtime_error("analysis emission_weighted_b_cgs mismatch");
    if (a.emission_weighted_beta != b.emission_weighted_beta) throw std::runtime_error("analysis emission_weighted_beta mismatch");
    if (a.emission_weighted_sigma != b.emission_weighted_sigma) throw std::runtime_error("analysis emission_weighted_sigma mismatch");
    if (a.photon_ring_winding_estimate != b.photon_ring_winding_estimate) throw std::runtime_error("analysis photon_ring_winding_estimate mismatch");
    if (a.radiation_substeps != b.radiation_substeps) throw std::runtime_error("analysis radiation_substeps mismatch");
    if (a.analysis_radial_bin_edges != b.analysis_radial_bin_edges) throw std::runtime_error("analysis analysis_radial_bin_edges mismatch");
    if (a.analysis_radial_bin_centers != b.analysis_radial_bin_centers) throw std::runtime_error("analysis analysis_radial_bin_centers mismatch");
    if (a.radial_stokes_i_contribution != b.radial_stokes_i_contribution) throw std::runtime_error("analysis radial_stokes_i_contribution mismatch");
    if (a.radial_stokes_q_contribution != b.radial_stokes_q_contribution) throw std::runtime_error("analysis radial_stokes_q_contribution mismatch");
    if (a.radial_stokes_u_contribution != b.radial_stokes_u_contribution) throw std::runtime_error("analysis radial_stokes_u_contribution mismatch");
    if (a.radial_stokes_v_contribution != b.radial_stokes_v_contribution) throw std::runtime_error("analysis radial_stokes_v_contribution mismatch");
    if (a.radial_absorption_depth != b.radial_absorption_depth) throw std::runtime_error("analysis radial_absorption_depth mismatch");
    if (a.radial_faraday_rotation_depth != b.radial_faraday_rotation_depth) throw std::runtime_error("analysis radial_faraday_rotation_depth mismatch");
    if (a.radial_faraday_conversion_depth != b.radial_faraday_conversion_depth) throw std::runtime_error("analysis radial_faraday_conversion_depth mismatch");
    if (a.radial_faraday_operator_depth != b.radial_faraday_operator_depth) throw std::runtime_error("analysis radial_faraday_operator_depth mismatch");
    if (a.observer_weighted_radius_i != b.observer_weighted_radius_i) throw std::runtime_error("analysis observer_weighted_radius_i mismatch");
    if (a.observer_weighted_radius_linear != b.observer_weighted_radius_linear) throw std::runtime_error("analysis observer_weighted_radius_linear mismatch");
    if (a.observer_weighted_radius_circular != b.observer_weighted_radius_circular) throw std::runtime_error("analysis observer_weighted_radius_circular mismatch");
    if (a.intensity_formation_radius_low != b.intensity_formation_radius_low) throw std::runtime_error("analysis intensity_formation_radius_low mismatch");
    if (a.intensity_formation_radius_median != b.intensity_formation_radius_median) throw std::runtime_error("analysis intensity_formation_radius_median mismatch");
    if (a.intensity_formation_radius_high != b.intensity_formation_radius_high) throw std::runtime_error("analysis intensity_formation_radius_high mismatch");
    if (a.linear_formation_radius_low != b.linear_formation_radius_low) throw std::runtime_error("analysis linear_formation_radius_low mismatch");
    if (a.linear_formation_radius_median != b.linear_formation_radius_median) throw std::runtime_error("analysis linear_formation_radius_median mismatch");
    if (a.linear_formation_radius_high != b.linear_formation_radius_high) throw std::runtime_error("analysis linear_formation_radius_high mismatch");
    if (a.circular_formation_radius_low != b.circular_formation_radius_low) throw std::runtime_error("analysis circular_formation_radius_low mismatch");
    if (a.circular_formation_radius_median != b.circular_formation_radius_median) throw std::runtime_error("analysis circular_formation_radius_median mismatch");
    if (a.circular_formation_radius_high != b.circular_formation_radius_high) throw std::runtime_error("analysis circular_formation_radius_high mismatch");
    if (a.los_linear_coherence != b.los_linear_coherence) throw std::runtime_error("analysis los_linear_coherence mismatch");
    if (a.los_circular_coherence != b.los_circular_coherence) throw std::runtime_error("analysis los_circular_coherence mismatch");
    if (a.contribution_closure_max_abs != b.contribution_closure_max_abs) throw std::runtime_error("analysis contribution_closure_max_abs mismatch");
    if (a.contribution_closure_relative_l1 != b.contribution_closure_relative_l1) throw std::runtime_error("analysis contribution_closure_relative_l1 mismatch");
    if (a.foreground_absorption_depth != b.foreground_absorption_depth) throw std::runtime_error("analysis foreground_absorption_depth mismatch");
    if (a.foreground_faraday_rotation_depth != b.foreground_faraday_rotation_depth) throw std::runtime_error("analysis foreground_faraday_rotation_depth mismatch");
    if (a.foreground_faraday_conversion_depth != b.foreground_faraday_conversion_depth) throw std::runtime_error("analysis foreground_faraday_conversion_depth mismatch");
    if (a.foreground_faraday_operator_depth != b.foreground_faraday_operator_depth) throw std::runtime_error("analysis foreground_faraday_operator_depth mismatch");
    if (a.foreground_faraday_operator_fraction != b.foreground_faraday_operator_fraction) throw std::runtime_error("analysis foreground_faraday_operator_fraction mismatch");
}


// A flat-space ray with a known time-dependent slab solution. This checks
// sampling across nonuniform snapshot knots independently of either solver.
struct SlabModel : Model {
    template<class Metric>
    KPOLARIS_INLINE kpolaris::TransferCoeffs<Real> coefficients(
        const Metric&, const kpolaris::TransportState<Real>&, Real,
        const kpolaris::PlasmaPerturbation<Real>& = {}) const {
        kpolaris::TransferCoeffs<Real> c;
        c.jI = Real(.3) + Real(.2) * Kokkos::sin(Real(12) * time);
        c.aI = Real(2);
        return c;
    }
};

struct SlabMetric : kpolaris::MinkowskiMetric<kpolaris::DefaultReal> {
    using Real = kpolaris::DefaultReal;
    static constexpr kpolaris::CoordinateSystem coordinate_system =
        kpolaris::CoordinateSystem::CartesianKS;
    Real spin = 0;
    KPOLARIS_INLINE Real radial_coordinate(const kpolaris::Vec4<Real>& x) const {
        return Kokkos::sqrt(x[1]*x[1] + x[2]*x[2] + x[3]*x[3]);
    }
};

void test_time_boundary_convergence() {
    using namespace kpolaris_image_detail;
    using Exec = Kokkos::DefaultExecutionSpace;
    using Mode = kpolaris::SlowLightStepMode;
    const std::vector<Real> times{0, .03, .12, .28, .39, .65, .71, 1, 1.1};
    std::vector<SlabModel> models(times.size());
    for (size_t i = 0; i < times.size(); ++i) models[i].time = times[i];
    auto device_models = copy_slow_light_models_to_device<Exec>(models, "slab_models");
    Kokkos::View<Real*, Exec> device_times("slab_times", times.size());
    auto ht = Kokkos::create_mirror_view(device_times);
    for (size_t i = 0; i < times.size(); ++i) ht(i) = times[i];
    Kokkos::deep_copy(device_times, ht);
    double exact = 0;
    for (size_t i = 0; i + 1 < times.size(); ++i) {
        if (times[i] >= 1) break; // the ray reaches the camera at fluid time 1
        const double dt = times[i + 1] - times[i];
        const double j0 = .3 + .2 * std::sin(12 * times[i]);
        const double j1 = .3 + .2 * std::sin(12 * times[i + 1]);
        const double one_minus_decay = -std::expm1(-2 * dt);
        exact = exact * std::exp(-2 * dt) + j0 * one_minus_decay / 2
            + (j1 - j0) / dt * (dt / 2 - one_minus_decay / 4);
    }
    const SlabMetric metric;
    auto run = [&](Mode mode, Real step, int block) {
        kpolaris::PassAParams<Real> params;
        params.camera.nx = params.camera.ny = 1;
        params.camera.radius = 30;
        params.inner_radius = 0; params.outer_radius = 100;
        params.adaptive = 1; params.adaptive_tolerance = 1e-12;
        params.min_step = 1e-14; params.max_step = step;
        params.max_radiation_step = step; params.max_radiation_depth = 0;
        params.max_absorption_depth = params.max_faraday_depth = 0;
        params.max_steps = 10000;
        auto v = allocate_slow_light_views<Exec, Real>(1, 0, {});
        Kokkos::parallel_for("initialize_slab_ray", Kokkos::RangePolicy<Exec>(0, 1),
            KOKKOS_LAMBDA(int) {
                auto state = kpolaris::initialize_camera_ray(metric, 0, params.camera);
                for (int mu = 0; mu < 4; ++mu) {
                    v.state_x(mu) = state.x[mu] - state.k[mu];
                    v.state_k(mu) = state.k[mu];
                    v.state_e1(mu) = state.e1[mu]; v.state_e2(mu) = state.e2[mu];
                }
                v.active(0) = 1; v.h_current(0) = step; v.radiation_step_cap(0) = step;
            });
        int rounds = 0;
        for (int first = 0; first < int(times.size()) - 1;) {
            if (++rounds > 10000) throw std::runtime_error("slab scheduler stalled");
            SlowLightResidentWindows<Exec, Real, SlabModel> windows{
                device_models, device_times, 1, first,
                std::min(first + block, int(times.size()) - 1),
                kpolaris::SlowLightInterpolation::coefficients, mode};
            run_slow_light_windows_metric(params, windows, metric, v, 0);
            if (mode == Mode::decoupled) {
                auto active = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v.active);
                if (!active(0)) break;
                auto requested = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, v.requested_time);
                if (requested(0) < times.front() || requested(0) > times.back())
                    throw std::runtime_error("slab missing input support");
                const int bracket = std::min(int(times.size()) - 2, int(
                    std::upper_bound(times.begin(), times.end(), requested(0)) - times.begin() - 1));
                first = (bracket / block) * block;
            } else first += block;
        }
        auto result = copy_slow_light_views_to_image_host_data(v, 1, 0);
        if (result.reason[0] != 1 || !std::isfinite(result.image_i[0]))
            throw std::runtime_error("slab ray failed: reason=" + std::to_string(result.reason[0])
                + " I=" + std::to_string(result.image_i[0])
                + " steps=" + std::to_string(result.steps[0]));
        return std::make_pair(double(result.image_i[0]), result.steps[0]);
    };
    for (auto mode : {Mode::snapshot, Mode::block, Mode::decoupled}) {
        auto coarse = run(mode, .2, 8);
        auto fine = run(mode, .025, 8);
        const double e0 = std::abs(coarse.first - exact), e1 = std::abs(fine.first - exact);
        std::cout << "slab " << kpolaris::slow_light_step_mode_name(mode)
                  << " coarse_error " << e0 << " fine_error " << e1
                  << " coarse_steps " << coarse.second << " fine_steps " << fine.second << "\n";
        if (!(e1 < e0 * .25 && e1 < 2e-4))
            throw std::runtime_error("time-boundary solver failed analytic slab refinement");
    }
    for (int block : {1, 2, 4, 8}) {
        if (run(Mode::decoupled, .2, block) != run(Mode::decoupled, .2, 8))
            throw std::runtime_error("decoupled slab depends on cache block");
        const auto refined = run(Mode::block, 1. / 1024, block);
        if (std::abs(refined.first - exact) > 3e-7)
            throw std::runtime_error("resident block selected the wrong temporal bracket");
    }
    // A cache with one pair must preserve the legacy integration exactly.
    if (run(Mode::snapshot, .2, 1) != run(Mode::block, .2, 1))
        throw std::runtime_error("one-pair block no longer reproduces snapshot stepping");
    if (run(Mode::block, .2, 8).second >= run(Mode::snapshot, .2, 8).second)
        throw std::runtime_error("decoupled stepping did not remove internal boundary steps");
    std::cout << "PASS: analytic time-dependent slab; nonuniform knots; both modes converge; blocks 1/2/4/8\n";
}

void test_decoupled_batch() {
    using namespace kpolaris_image_detail;
    kpolaris::PassAParams<Real> params;
    params.camera.nx = 3; params.camera.ny = 2; params.camera.radius = 30;
    params.camera.fov = .3; params.camera.inclination = .7;
    params.coordinate_system = kpolaris::CoordinateSystem::BoyerLindquist;
    params.inner_radius = 2.2; params.outer_radius = 35;
    params.step = .2; params.max_steps = 20000;
    params.adaptive = 1; params.adaptive_tolerance = 1e-10;
    params.min_step = 1e-12; params.max_step = 1;
    params.max_absorption_depth = .002; params.max_faraday_depth = .003;
    params.max_radiation_step = 0; params.max_radiation_depth = 0;
    kpolaris::KerrBoyerLindquistMetric<Real> metric(1, 0);
    std::vector<Real> times;
    std::vector<std::string> paths;
    for (int index = 0; index < 161; ++index) {
        const Real time = -100 + index + (index % 2 ? .17 : 0);
        times.push_back(time); paths.push_back(std::to_string(time));
    }
    const std::vector<SlowLightBatchJob> jobs{{0, 0, 150}, {1, 2, 152}};
    SlowLightBatchOptions options;
    options.step_mode = kpolaris::SlowLightStepMode::decoupled;
    options.interpolation = kpolaris::SlowLightInterpolation::coefficients;
    for (int analysis : {0, 1}) {
#if !KPOLARIS_ENABLE_ANALYSIS_MODE
        if (analysis) continue;
#endif
        options.analysis_mode = analysis;
        options.analysis_config.radial_bins = 3;
        options.analysis_config.radial_min = params.inner_radius;
        options.analysis_config.radial_max = params.outer_radius;
        auto& response = options.analysis_config.response;
        response.partition = kpolaris::SourcePartition::radial;
        response.bins = 2; response.edge_count = 3;
        response.edges[0] = params.inner_radius; response.edges[1] = 10; response.edges[2] = params.outer_radius;
        response.parameter = kpolaris::PlasmaParameter::density_scale;
        const std::vector<Real> frequencies{230e9, 345e9, 86e9};
        options.windows_per_block = 160;
        options.frequency_chunk_size = 0;
        auto reference = run_slow_light_batch_metric<decltype(metric), Model>(
            params, paths, times, jobs, StagedLoader{}, metric, options, nullptr, frequencies);
        for (int block : {1, 3, 24}) {
            for (int prefetch : {0, 1}) {
                options.windows_per_block = block; options.prefetch = prefetch;
                auto actual = run_slow_light_batch_metric<decltype(metric), Model>(
                    params, paths, times, jobs, StagedLoader{}, metric, options, nullptr, frequencies);
                for (size_t index = 0; index < reference.size(); ++index) {
                    require_same_image(reference[index], actual[index]);
                    require_same_analysis(reference[index], actual[index]);
                }
            }
        }
        auto single = run_slow_light_batch_metric<decltype(metric), Model>(
            params, paths, times, {jobs[1]}, StagedLoader{}, metric, options, nullptr, frequencies);
        for (size_t f = 0; f < frequencies.size(); ++f) {
            require_same_image(reference[frequencies.size() + f], single[f]);
            require_same_analysis(reference[frequencies.size() + f], single[f]);
        }
        // With radiation caps disabled, each frequency must reproduce an
        // independent solve on exactly the same geometric steps, including diagnostics.
        auto geometric = params;
        geometric.max_absorption_depth = geometric.max_faraday_depth = 0;
        auto fused = run_slow_light_batch_metric<decltype(metric), Model>(
            geometric, paths, times, jobs, StagedLoader{}, metric, options, nullptr, frequencies);
        options.frequency_chunk_size = 1;
        auto separate = run_slow_light_batch_metric<decltype(metric), Model>(
            geometric, paths, times, jobs, StagedLoader{}, metric, options, nullptr, frequencies);
        for (size_t index = 0; index < fused.size(); ++index) {
            require_same_image(fused[index], separate[index]);
            require_same_analysis(fused[index], separate[index]);
        }
    }
    bool rejected = false;
    try {
        (void)run_slow_light_batch_metric<decltype(metric), Model>(
            params, paths, times, {{0, 100, 150}}, StagedLoader{}, metric, options);
    } catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("decoupled accepted missing radiation support");
    std::cout << "PASS: decoupled multi-frequency batches; cache/prefetch invariance; fused/separate diagnostics and responses; depth retries; replay; missing support\n";
}

void test_decoupled_backward_retry() {
    using namespace kpolaris_image_detail;
    using Exec = Kokkos::DefaultExecutionSpace;
    const std::vector<Real> times{0, .03, .12, .28, .39, .65, .71, 1, 1.1};
    std::vector<SlabModel> models(times.size());
    for (size_t index = 0; index < times.size(); ++index) models[index].time = times[index];
    auto device_models = copy_slow_light_models_to_device<Exec>(models, "retry_models");
    Kokkos::View<Real*, Exec> device_times("retry_times", times.size());
    auto host_times = Kokkos::create_mirror_view(device_times);
    for (size_t index = 0; index < times.size(); ++index) host_times(index) = times[index];
    Kokkos::deep_copy(device_times, host_times);
    const SlabMetric metric;
    int backwards = 0;
    auto run = [&](int block) {
        kpolaris::PassAParams<Real> params;
        params.camera.nx = params.camera.ny = 1; params.camera.radius = 30;
        params.inner_radius = 0; params.outer_radius = 100;
        params.adaptive = 1; params.adaptive_tolerance = 1e-12;
        params.min_step = 1e-14; params.max_step = .8;
        params.max_radiation_step = params.max_radiation_depth = params.max_faraday_depth = 0;
        params.max_absorption_depth = .02; params.max_steps = 10000;
        auto views = allocate_slow_light_views<Exec, Real>(1, 0, {});
        Kokkos::parallel_for("initialize_retry_ray", Kokkos::RangePolicy<Exec>(0, 1), KOKKOS_LAMBDA(int) {
            auto state = kpolaris::initialize_camera_ray(metric, 0, params.camera);
            for (int component = 0; component < 4; ++component) {
                views.state_x(component) = state.x[component] - state.k[component];
                views.state_k(component) = state.k[component];
                views.state_e1(component) = state.e1[component]; views.state_e2(component) = state.e2[component];
            }
            views.active(0) = 1; views.h_current(0) = .8; views.radiation_step_cap(0) = .8;
        });
        int first = 0;
        for (int iteration = 0; iteration < 1000; ++iteration) {
            SlowLightResidentWindows<Exec, Real, SlabModel> windows{
                device_models, device_times, 1, first, std::min(first + block, int(times.size()) - 1),
                kpolaris::SlowLightInterpolation::coefficients, kpolaris::SlowLightStepMode::decoupled};
            run_slow_light_windows_metric(params, windows, metric, views, 0);
            auto active = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, views.active);
            if (!active(0)) return copy_slow_light_views_to_image_host_data(views, 1, 0);
            auto requested = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, views.requested_time);
            if (requested(0) < times.front() || requested(0) > times.back()) throw std::runtime_error("missing support");
            const int bracket = std::min(int(times.size()) - 2, int(
                std::upper_bound(times.begin(), times.end(), requested(0)) - times.begin() - 1));
            const int next = (bracket / block) * block;
            if (next < first) ++backwards;
            first = next;
        }
        throw std::runtime_error("retry scheduler stalled");
    };
    auto reference = run(8);
    for (int block : {1, 2, 3}) require_same_image(reference, run(block));
    if (!backwards) throw std::runtime_error("did not exercise backwards request after rejection");
    std::cout << "PASS: forced backwards cache requests " << backwards << "; exact image and step equality\n";
}

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    try {
        using namespace kpolaris_image_detail;
        test_time_boundary_convergence();
        test_decoupled_batch();
        test_decoupled_backward_retry();
        // This fixture defines coefficients directly and has no interpolable plasma.
        constexpr auto method = kpolaris::SlowLightInterpolation::coefficients;
        SlowLightBatchOptions coefficient_options;
        coefficient_options.interpolation = method;
        coefficient_options.step_mode = kpolaris::SlowLightStepMode::snapshot;
        kpolaris::PassAParams<Real> p;
        p.camera.nx=3; p.camera.ny=2; p.camera.radius=30;
        p.camera.fov=.3; p.camera.inclination=.7;
        p.coordinate_system=kpolaris::CoordinateSystem::BoyerLindquist;
        p.inner_radius=2.2; p.outer_radius=35; p.step=.2; p.max_steps=2000;
        kpolaris::KerrBoyerLindquistMetric<Real> metric(1,0);
        std::vector<Real> times;
        std::vector<std::string> paths;
        for (int i=0;i<161;++i) { times.push_back(-100+i); paths.push_back(std::to_string(-100+i)); }
        const std::vector<SlowLightBatchJob> jobs={{0,0,150},{.5,1,151},{1,2,152},{2,4,154}};
        std::map<std::string,int> counts;
        auto load=[&](const std::string& path) { ++counts[path]; return Model{230e9, std::stod(path)}; };
        std::vector<ImageHostData> expected;
        for (const auto& job:jobs) {
            std::vector<std::string> subpaths(paths.begin()+job.first,paths.begin()+job.last+1);
            std::vector<Real> subtimes(times.begin()+job.first,times.begin()+job.last+1);
            expected.push_back(run_slow_light_image_metric(p,subpaths,subtimes,job.observation_time,
                               load(subpaths[0]),load,0,metric,"reference",0,0,{},method));
        }
        int serial_loads=0; for (auto [key,count]:counts) serial_loads+=count;
        counts.clear();
        auto actual=run_slow_light_batch_metric<decltype(metric),Model>(p,paths,times,jobs,load,metric,coefficient_options);
        for (size_t j=0;j<jobs.size();++j) require_same_image(expected[j],actual[j]);
        for (auto [key,count]:counts) if(count!=1) throw std::runtime_error("snapshot loaded twice");
        if (counts.size()>=static_cast<size_t>(serial_loads)) throw std::runtime_error("no reuse");
        bool nonzero=false; for(auto x:actual[0].image_i) nonzero=nonzero || x>0;
        if(!nonzero) throw std::runtime_error("vacuous zero-intensity fixture");
        std::cout<<"PASS: exact IQUV and diagnostic equality; serial loads="<<serial_loads
                 <<" shared loads="<<counts.size()<<"\n";
        const std::vector<std::vector<SlowLightBatchJob>> cases={
            {{0,0,150}}, {{2,4,154},{0,0,150}},
            {{0,0,10},{.5,1,12}}, {{0,0,10},{1,15,30}}};
        for (const auto& batch:cases) {
            auto got=run_slow_light_batch_metric<decltype(metric),Model>(p,paths,times,batch,load,metric,coefficient_options);
            for(size_t j=0;j<batch.size();++j) {
                const auto& job=batch[j];
                std::vector<std::string> sp(paths.begin()+job.first,paths.begin()+job.last+1);
                std::vector<Real> st(times.begin()+job.first,times.begin()+job.last+1);
                auto ref=run_slow_light_image_metric(p,sp,st,job.observation_time,
                         load(sp[0]),load,0,metric,"reference",0,0,{},method);
                require_same_image(ref,got[j]);
            }
        }
        counts.clear();
        if(!run_slow_light_batch_metric<decltype(metric),Model>(p,paths,times,{},load,metric).empty() || !counts.empty())
            throw std::runtime_error("empty batch performed work");
        for(const auto& bad:std::vector<SlowLightBatchJob>{{0,0,0},{0,0,999}}) {
            bool rejected=false;
            try { (void)run_slow_light_batch_metric<decltype(metric),Model>(p,paths,times,{bad},load,metric); }
            catch(const std::invalid_argument&) { rejected=true; }
            if(!rejected) throw std::runtime_error("invalid job accepted");
        }
        std::cout<<"PASS: single, reordered, truncated, disjoint, empty and invalid cases\n";
        for (int adaptive : {0, 1}) {
            auto params = p;
            params.adaptive = adaptive;
            params.adaptive_tolerance = 1e-10;
            params.min_step = 1e-12;
            params.max_step = 1;
            for (int analysis_mode : {0, 1}) {
#if !KPOLARIS_ENABLE_ANALYSIS_MODE
                if (analysis_mode) continue;
#endif
                SlowLightBatchOptions options;
                options.interpolation = method;
                // This regression intentionally checks the legacy per-snapshot
                // partition against the original single-window driver.
                options.step_mode = kpolaris::SlowLightStepMode::snapshot;
                options.analysis_mode = analysis_mode;
                options.analysis_config.radial_bins = 3;
                options.analysis_config.radial_min = params.inner_radius;
                options.analysis_config.radial_max = params.outer_radius;
                options.analysis_config.response.partition = kpolaris::SourcePartition::radial;
                options.analysis_config.response.bins = 2;
                options.analysis_config.response.parameter = kpolaris::PlasmaParameter::density_scale;
                options.analysis_config.response.edge_count = 3;
                options.analysis_config.response.edges[0] = params.inner_radius;
                options.analysis_config.response.edges[1] = 10;
                options.analysis_config.response.edges[2] = params.outer_radius;
                const std::vector<SlowLightBatchJob> batch = {{0, 0, 150}, {1, 2, 152}};
                std::vector<ImageHostData> reference;
                for (const auto& job : batch) {
                    std::vector<std::string> sp(paths.begin()+job.first, paths.begin()+job.last+1);
                    std::vector<Real> st(times.begin()+job.first, times.begin()+job.last+1);
                    reference.push_back(run_slow_light_image_metric(params, sp, st,
                        job.observation_time, Model{230e9, st.front()}, StagedLoader{}, 0, metric,
                        "reference", analysis_mode, 0, options.analysis_config, method));
                }
                for (int block : {1, 2, 8, 32, 200}) {
                    options.windows_per_block = block;
                    for (int prefetch : {0, 1}) {
                        options.prefetch = prefetch;
                        options.prefetch_snapshots = 3; // smaller than a large GPU block
                        auto got = run_slow_light_batch_metric<decltype(metric), Model>(
                            params, paths, times, batch, StagedLoader{}, metric, options);
                        for (size_t j=0; j<batch.size(); ++j) {
                            require_same_image(reference[j], got[j]);
                            require_same_analysis(reference[j], got[j]);
                        }
                    }
                }
            }
        }
        {
            SlowLightBatchOptions options;
            options.interpolation = method;
            options.prefetch_snapshots = 2;
            StagedLoader broken;
            broken.fail_time = -98;
            bool rejected = false;
            try { (void)run_slow_light_batch_metric<decltype(metric),Model>(
                p, paths, times, {{0,0,150}}, broken, metric, options); }
            catch (const std::runtime_error& e) { rejected = std::string(e.what()) == "injected input failure"; }
            if (!rejected) throw std::runtime_error("streaming input error was lost");
            options.prefetch_snapshots = -1;
            rejected = false;
            try { (void)run_slow_light_batch_metric<decltype(metric),Model>(
                p, paths, times, {{0,0,150}}, StagedLoader{}, metric, options); }
            catch (const std::invalid_argument&) { rejected=true; }
            if (!rejected) throw std::runtime_error("negative host queue capacity accepted");
        }
        std::cout << "PASS: blocks 1/2/8/32/200; fixed/adaptive; all analysis arrays and source tags; bounded streaming; input errors\n";

    } catch(const std::exception& e) { std::cerr<<e.what()<<"\n"; return 1; }
}
