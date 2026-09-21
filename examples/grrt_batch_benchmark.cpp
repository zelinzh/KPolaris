// In-memory experiment; output excludes GRMHD loading, CUDA initialization and HDF5.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include "image/batch_driver.hpp"
#include "tests/image_batch_checks.hpp"
#ifdef KPOLARIS_BENCHMARK_ATHENAK
#include "grmhd/athenak_loader.hpp"
#include "geometry/kerr_schild_cartesian.hpp"
#else
#include "geometry/kerr_boyer_lindquist.hpp"
#include "model/riaf.hpp"
#endif

int main(int argc, char** argv) {
    Kokkos::ScopeGuard guard(argc, argv);
    try {
        using namespace kpolaris_image_detail;
        int n = 64, count = 4, repeats = 3, slots = 4;
        std::string dump;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            const auto eq = arg.find('=');
            const auto key = arg.substr(0, eq);
            const auto value = eq == std::string::npos ? std::string() : arg.substr(eq + 1);
            if (key == "--nx") n = std::stoi(value);
            else if (key == "--images") count = std::stoi(value);
            else if (key == "--repeats") repeats = std::stoi(value);
            else if (key == "--slots") slots = std::stoi(value);
            else if (key == "--dump") dump = value;
            else throw std::invalid_argument("expected --nx=N --images=N --repeats=N --slots=N [--dump=PATH]");
        }
        if (n < 1 || count < 1 || repeats < 1 || slots < 1)
            throw std::invalid_argument("benchmark counts must be positive");
#ifdef KPOLARIS_BENCHMARK_ATHENAK
        using Model = kpolaris::AthenaKDirectRadiationModel<Real>;
        using Metric = kpolaris::KerrSchildInMetric<Real>;
        kpolaris::AthenaKLoadOptions load;
        load.dump_path = dump;
        load.resample_r_out = 100;
        load.M_unit = Real(1e26);
        load.trat_large = 40;
        load.timing = 1;
        Model model = kpolaris::load_athenak_direct_model_from_binary(load);
        const Real spin = model.spin;
#else
        using Model = kpolaris::RIAFAnalyticRadiationModel<Real>;
        using Metric = kpolaris::KerrBoyerLindquistMetric<Real>;
        Model model;
        model.r_max = 100;
        const Real spin = Real(0.9375);
#endif
        using Job = ImageBatchJob<Metric, Model>;
        std::vector<Job> jobs;
        for (int i = 0; i < count; ++i) {
            Job job;
            job.params.camera.nx = n;
            job.params.camera.ny = n;
            job.params.camera.radius = 1000;
            job.params.camera.fov = Real(0.08); // explicit dimensionless screen width
            job.params.camera.inclination = (Real(17) + Real(i % 4) * Real(23)) * Real(0.017453292519943295);
            job.params.coordinate_system = Metric::coordinate_system;
            job.params.spin = spin;
            job.params.inner_radius = Real(1.05) * (Real(1) + Kokkos::sqrt(Real(1) - spin * spin));
            job.params.outer_radius = 100;
            job.params.step = Real(0.025);
            job.params.adaptive = 1;
            job.params.adaptive_tolerance = Real(1e-10);
            job.params.min_step = Real(1e-20);
            job.params.max_step = 2;
            job.params.max_steps = 500000;
            job.params.max_radiation_step = 2;
            job.params.max_radiation_depth = 4;
            job.params.max_absorption_depth = 4;
            job.params.max_faraday_depth = 16;
            job.model = model; // Kokkos Views share one dump, no duplicate upload.
            job.metric = Metric(1, spin);
            jobs.push_back(job);
        }
        ImageBatchOptions options;
        options.max_in_flight = 1;
        auto reference = run_image_batch_metric(jobs, options);
        options.max_in_flight = static_cast<size_t>(slots);
        auto warm = run_image_batch_metric(jobs, options);
        for (size_t j = 0; j < jobs.size(); ++j) require_same_image(reference[j], warm[j]);
        std::cout << std::setprecision(12);
        std::cout << "configuration backend=" << Kokkos::DefaultExecutionSpace::name()
                  << " nx=" << n << " images=" << count << " slots=" << slots
                  << " tolerance=1e-10 radius=1000 fov=0.08 source="
#ifdef KPOLARIS_BENCHMARK_ATHENAK
                  << "athenak"
#else
                  << "riaf"
#endif
                  << "\n";
        for (int rep = 0; rep < repeats; ++rep) {
            // Alternate order after warming both paths to reduce ordering bias.
            for (int which = 0; which < 2; ++which) {
                const bool concurrent = ((rep + which) % 2) != 0;
                options.max_in_flight = concurrent ? static_cast<size_t>(slots) : 1;
                ImageBatchTiming timing;
                Kokkos::Timer wall;
                auto images = run_image_batch_metric(jobs, options, &timing);
                const double seconds = wall.seconds();
                for (size_t j = 0; j < jobs.size(); ++j) require_same_image(reference[j], images[j]);
                std::cout << "measurement {\"rep\":" << rep << ",\"slots\":" << options.max_in_flight
                          << ",\"wall_seconds\":" << seconds << ",\"kernel_seconds\":" << timing.kernel_seconds
                          << ",\"allocate_seconds\":" << timing.allocate_seconds
                          << ",\"copy_seconds\":" << timing.copy_seconds
                          << ",\"images_per_second\":" << count / seconds
                          << ",\"peak_output_bytes\":" << timing.peak_device_output_bytes
                          << ",\"bitwise_equal\":true}\n";
            }
        }
        for (size_t j = 0; j < reference.size(); ++j) {
            auto steps = reference[j].steps;
            std::sort(steps.begin(), steps.end());
            const double mean = std::accumulate(steps.begin(), steps.end(), 0.0) / steps.size();
            size_t returned = 0;
            for (const auto reason : reference[j].reason)
                returned += reason == static_cast<int>(kpolaris::TerminationReason::reached_camera);
            std::cout << "steps image=" << j << " mean=" << mean << " p50=" << steps[steps.size()/2]
                      << " p95=" << steps[(steps.size()-1)*95/100] << " max=" << steps.back()
                      << " returned=" << returned << "/" << steps.size() << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
