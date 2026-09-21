#include <iostream>
#include <limits>
#include "image/batch_driver.hpp"
#include "geometry/kerr_boyer_lindquist.hpp"
#include "model/model.hpp"
#include "image_batch_checks.hpp"

// Distinct allocations emulate independently resident snapshots. This catches
// accidental reuse of the first model when scheduling jobs on different streams.
struct ViewBackedConstantModel {
    using Real = kpolaris::DefaultReal;
    Kokkos::View<kpolaris::TransferCoeffs<Real>*> values;
    template<class Metric>
    KPOLARIS_INLINE kpolaris::TransferCoeffs<Real> coefficients(
        const Metric&, const kpolaris::TransportState<Real>&, Real) const {
        return values(0);
    }
    KPOLARIS_INLINE Real dlambda_scale() const { return Real(1); }
};

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    try {
        using namespace kpolaris_image_detail;
        using Metric = kpolaris::KerrBoyerLindquistMetric<Real>;
        using Model = ViewBackedConstantModel;
        using Job = ImageBatchJob<Metric, Model>;
        std::vector<Job> jobs;
        for (int j = 0; j < 5; ++j) {
            Job job;
            job.params.camera.nx = 2 + j % 2;
            job.params.camera.ny = 2;
            job.params.camera.radius = 30;
            job.params.camera.fov = Real(0.3);
            job.params.camera.inclination = Real(0.7) + Real(j) * Real(0.1);
            job.params.coordinate_system = kpolaris::CoordinateSystem::BoyerLindquist;
            job.params.inner_radius = Real(2.2);
            job.params.outer_radius = 35;
            job.params.step = Real(0.2);
            job.params.max_steps = 2000;
            job.model.values = Kokkos::View<kpolaris::TransferCoeffs<Real>*>("job_coefficients", 1);
            auto host = Kokkos::create_mirror_view(job.model.values);
            host(0).jI = Real(0.1) * (j + 1);
            host(0).jQ = Real(0.01);
            host(0).aI = Real(0.02);
            host(0).rV = Real(0.03);
            Kokkos::deep_copy(job.model.values, host);
            job.metric = Metric(1, 0);
            jobs.push_back(job);
        }
        std::vector<ImageHostData> expected;
        for (const auto& job : jobs)
            expected.push_back(run_image_metric(job.params, job.model, 1, 0, job.metric, "test"));
        ImageBatchOptions opt;
        // Force uneven windows; final partial window and mixed image shapes.
        opt.max_in_flight = 3;
        opt.max_device_output_bytes = 10 * BatchImageDeviceData::bytes_per_pixel;
        ImageBatchTiming timing;
        const auto actual = run_image_batch_metric(jobs, opt, &timing);
        if (actual.size() != expected.size() || timing.windows != 3 ||
            timing.peak_device_output_bytes > opt.max_device_output_bytes)
            throw std::runtime_error("batch window/budget contract failed");
        for (size_t i = 0; i < jobs.size(); ++i) require_same_image(expected[i], actual[i]);
        if (!run_image_batch_metric(std::vector<Job>{}).empty())
            throw std::runtime_error("empty batch failed");
        auto rejects = [&](const std::vector<Job>& input, const ImageBatchOptions& options) {
            try { (void)run_image_batch_metric(input, options); }
            catch (const std::invalid_argument&) { return; }
            throw std::runtime_error("invalid batch accepted");
        };
        opt.max_in_flight = 0; rejects(jobs, opt);
        opt.max_in_flight = 2; opt.max_device_output_bytes = 1; rejects(jobs, opt);
        opt = {};
        jobs.back().params.camera.nx = std::numeric_limits<int>::max(); rejects(jobs, opt);
        jobs.back().params.camera.nx = 0; rejects(jobs, opt);
        jobs.back().params.camera.nx = 2; jobs.back().radiation_substeps = 0; rejects(jobs, opt);
        std::cout << "batch images, diagnostics, ordering, and memory bounds passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
