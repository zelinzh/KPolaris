#pragma once

#include <cmath>
#include <memory>
#include "image/slow_light_driver.hpp"
#include "image/slow_light_prefetch.hpp"

namespace kpolaris_image_detail {

// All jobs share the camera, metric and plasma prescription.
// Only observer time and the inclusive snapshot support may vary. This reuses
// stationary-metric geometry, not Stokes solutions from another observer time.
struct SlowLightBatchJob {
    Real observation_time = 0;
    size_t first = 0;
    size_t last = 0;
};

struct SlowLightBatchOptions {
    int windows_per_block = 1;
    int prefetch = 1;
    // Independent host queue capacity; zero selects windows_per_block.
    int prefetch_snapshots = 0;
    int pipeline = 0;
    int timing = 0;
    int frequency_chunk_size = 0; // zero selects the compiled capacity
    int analysis_mode = 0;
    kpolaris::SlowLightInterpolation interpolation = kpolaris::default_slow_light_interpolation;
    kpolaris::AnalysisConfig<Real> analysis_config;
    kpolaris::SlowLightStepMode step_mode = kpolaris::default_slow_light_step_mode;
};

template<class LoadModel, class = void>
struct SlowLightHasHostStaging : std::false_type {};
template<class LoadModel>
struct SlowLightHasHostStaging<LoadModel,
    std::void_t<typename LoadModel::slow_light_prefetch_result_type>> : std::true_type {};

template<class Loader, class Model, class = void>
struct SlowLightUploadPipeline {
    static constexpr bool available = false;
    SlowLightUploadPipeline(Loader&, const Model&, size_t) {}
    Model prepare(Loader& loader, typename SlowLightLoaderTraits<Loader,Model>::PrefetchResult&& staged, size_t) {
        return SlowLightLoaderTraits<Loader,Model>::materialize(loader,std::move(staged));
    }
    void fence() {}
};
template<class Loader, class Model>
struct SlowLightUploadPipeline<Loader,Model,std::void_t<typename Loader::slow_light_upload_workspace_type>> {
    static constexpr bool available = true;
    typename Loader::slow_light_upload_workspace_type workspace;
    SlowLightUploadPipeline(Loader&, const Model& first, size_t slots) : workspace(first,slots) {}
    Model prepare(Loader& loader, typename Loader::slow_light_prefetch_result_type&& staged, size_t slot) {
        return loader.pipeline_prepare(workspace,std::move(staged),slot);
    }
    void fence() { workspace.fence(); }
};

// CUDA events exclude concurrent host input waits from ray-kernel timing.
struct SlowLightPipelineKernelTimer {
#ifdef KOKKOS_ENABLE_CUDA
    cudaEvent_t begin{}, end{};
    SlowLightPipelineKernelTimer() {
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventCreate(&begin));
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventCreate(&end));
    }
    ~SlowLightPipelineKernelTimer() { cudaEventDestroy(end); cudaEventDestroy(begin); }
    void start() { KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventRecord(begin,Kokkos::DefaultExecutionSpace{}.cuda_stream())); }
    void stop() { KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventRecord(end,Kokkos::DefaultExecutionSpace{}.cuda_stream())); }
    double seconds() { float ms=0; KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventSynchronize(end));
        KOKKOS_IMPL_CUDA_SAFE_CALL(cudaEventElapsedTime(&ms,begin,end)); return double(ms)*0.001; }
#else
    Kokkos::Timer timer;
    double elapsed = 0;
    void start() { timer.reset(); }
    void stop() { Kokkos::DefaultExecutionSpace{}.fence(); elapsed=timer.seconds(); }
    double seconds() const { return elapsed; }
#endif
};

template<class Exec, class RealT>
std::pair<RealT, RealT> slow_light_pending_range(
    const SlowLightViews<Exec, RealT>& pending, RealT resident_first, RealT resident_last) {
    const int npix = pending.active.extent_int(0);
    RealT first_request, last_request;
    int invalid_requests;
    Kokkos::parallel_reduce("slow_request_min", Kokkos::RangePolicy<Exec>(0, npix),
        KOKKOS_LAMBDA(int pixel, RealT& minimum) {
            if (pending.active(pixel)) minimum = kpolaris::min_val(minimum, pending.requested_time(pixel));
        }, Kokkos::Min<RealT>(first_request));
    Kokkos::parallel_reduce("slow_request_max", Kokkos::RangePolicy<Exec>(0, npix),
        KOKKOS_LAMBDA(int pixel, RealT& maximum) {
            if (pending.active(pixel)) maximum = kpolaris::max_val(maximum, pending.requested_time(pixel));
        }, Kokkos::Max<RealT>(last_request));
    Kokkos::parallel_reduce("slow_request_valid", Kokkos::RangePolicy<Exec>(0, npix),
        KOKKOS_LAMBDA(int pixel, int& invalid) {
            if (pending.active(pixel) && (!pending.waiting_for_data(pixel) ||
                !Kokkos::isfinite(pending.requested_time(pixel)) ||
                (pending.requested_time(pixel) >= resident_first && pending.requested_time(pixel) <= resident_last))) ++invalid;
        }, invalid_requests);
    if (invalid_requests) throw std::runtime_error("decoupled slow-light scheduler stalled with invalid cache requests");
    return {first_request, last_request};
}

template<int FrequencyCapacity, class Metric, class RadiationModel, class LoadModel>
std::vector<ImageHostData> run_slow_light_batch_metric_impl(
    const kpolaris::PassAParams<Real>& params,
    const std::vector<std::string>& paths,
    const std::vector<Real>& times,
    const std::vector<SlowLightBatchJob>& observer_jobs,
    LoadModel loader,
    const Metric& metric,
    const SlowLightBatchOptions& options,
    const RadiationModel* first_model,
    const std::vector<Real>& frequencies) {
    if (observer_jobs.empty()) return {};
    const size_t nfreq = frequencies.empty() ? 1 : frequencies.size();
    if (nfreq > size_t(std::numeric_limits<int>::max()) / observer_jobs.size())
        throw std::invalid_argument("slow-light frequency/image count exceeds indexing limits");
    for (Real frequency : frequencies)
        if (!std::isfinite(frequency) || frequency <= 0)
            throw std::invalid_argument("slow-light frequencies must be positive and finite");
    if (options.frequency_chunk_size < 0)
        throw std::invalid_argument("slow-light frequency chunk size must be nonnegative");
    const size_t chunk = options.frequency_chunk_size > 0 ?
        std::min(size_t(FrequencyCapacity), size_t(options.frequency_chunk_size)) : size_t(FrequencyCapacity);
    std::vector<SlowLightBatchJob> jobs;
    jobs.reserve(observer_jobs.size() * nfreq);
    for (const auto& job : observer_jobs)
        for (size_t f = 0; f < nfreq; ++f) jobs.push_back(job);
    // Host results use observation-major, frequency-minor ordering.

    if (paths.size() != times.size() || paths.size() < 2)
        throw std::invalid_argument("slow-light batch requires matching path/time lists with at least two snapshots");
    for (size_t i = 0; i < times.size(); ++i)
        if (!std::isfinite(times[i]) || (i && times[i] <= times[i - 1]))
            throw std::invalid_argument("slow-light batch times must be finite and strictly increasing");
    if (options.windows_per_block < 1 || options.prefetch_snapshots < 0)
        throw std::invalid_argument("slow-light windows_per_block must be positive and prefetch_snapshots nonnegative");
    if (params.camera.nx <= 0 || params.camera.ny <= 0 ||
        params.camera.nx > std::numeric_limits<int>::max() / params.camera.ny / kpolaris::ndim)
        throw std::invalid_argument("slow-light batch image dimensions exceed state indexing limits");
    size_t begin = paths.size(), end = 0;
    for (const auto& job : jobs) {
        if (!std::isfinite(job.observation_time) || job.first >= job.last || job.last >= paths.size())
            throw std::invalid_argument("invalid slow-light batch observer time or snapshot support");
        begin = std::min(begin, job.first);
        end = std::max(end, job.last);
    }
    if (first_model && begin != 0)
        throw std::invalid_argument("supplied first_model must correspond to snapshot zero and the batch must start there");
#if !KPOLARIS_ENABLE_ANALYSIS_MODE
    if (options.analysis_mode) throw std::invalid_argument("slow-light analysis is disabled in this build");
#endif
    using Exec = Kokkos::DefaultExecutionSpace;
    using Traits = SlowLightLoaderTraits<LoadModel, RadiationModel>;
    const bool decoupled = options.step_mode == kpolaris::SlowLightStepMode::decoupled;
    const bool prefetch = options.prefetch && SlowLightHasHostStaging<LoadModel>::value;
    const int npix = params.camera.nx * params.camera.ny;
    if (jobs.size() > size_t(std::numeric_limits<int>::max() / npix))
        throw std::invalid_argument("slow-light batch image count exceeds ray indexing limits");
    const auto& analysis = options.analysis_config;
    const int analysis_mode = options.analysis_mode;
    const int timing = options.timing;
    std::vector<SlowLightViews<Exec, Real>> views(jobs.size());
    std::vector<ImageHostData> result(jobs.size());
    std::vector<bool> done(jobs.size(), false);
    for (size_t j = 0; j < jobs.size(); ++j) {
        views[j] = allocate_slow_light_views<Exec, Real>(npix, analysis_mode, analysis);
        if (!j) initialize_slow_light_states_metric(params, metric, views[j], timing, analysis_mode, analysis);
        else copy_slow_light_views(views[0], views[j], timing, analysis_mode);
    }
    RadiationModel lower = first_model ? *first_model : Traits::load(loader, paths[begin]);
    using Stream = SlowLightHostStream<Traits, LoadModel>;
    std::unique_ptr<Stream> stream;
    const size_t capacity = options.prefetch_snapshots > 0 ?
        size_t(options.prefetch_snapshots) : size_t(options.windows_per_block);
    if (prefetch) stream = std::make_unique<Stream>(loader, paths, begin + 1, end, capacity);
    using Pipeline = SlowLightUploadPipeline<LoadModel,RadiationModel>;
    std::unique_ptr<Pipeline> pipeline;
    const size_t block_capacity = std::min(size_t(options.windows_per_block),end-begin);
    const size_t halo = decoupled ? size_t(kpolaris::decoupled_cache_halo_snapshots) : 0;
    const size_t ring_size = 2 * block_capacity + halo + 1;
    if (options.pipeline && prefetch && Pipeline::available) {
        Kokkos::Timer timer;
        pipeline = std::make_unique<Pipeline>(loader,lower,ring_size);
        report_image_timing(timing,"slow_light_pipeline_allocate",timer.seconds());
    }
    auto prepare_block = [&](size_t start, size_t stop, const std::vector<RadiationModel>& retained) {
        auto models = retained;
        models.reserve(stop-start+retained.size());
        double input_seconds=0, materialize_seconds=0;
        for (size_t k=start+1;k<=stop;++k) {
            Kokkos::Timer timer;
            if (stream) {
                auto staged=stream->get(k);
                input_seconds+=timer.seconds(); timer.reset();
                if (pipeline) models.push_back(pipeline->prepare(loader,std::move(staged),(k-begin-1)%ring_size));
                else models.push_back(Traits::materialize(loader,std::move(staged)));
                materialize_seconds+=timer.seconds();
            } else {
                models.push_back(Traits::load(loader,paths[k])); input_seconds+=timer.seconds();
            }
        }
        report_image_timing(timing,stream ? "slow_light_block_input_wait" : "slow_light_block_load",input_seconds);
        report_image_timing(timing,"slow_light_block_materialize",materialize_seconds);
        if (timing) std::osyncstream(std::cout) << "timing slow_light_cache_input first " << start + 1
            << " last " << stop << " states " << stop - start << "\n";
        return models;
    };
    size_t remaining = jobs.size();
    std::vector<RadiationModel> prepared;
    SlowLightPipelineKernelTimer device_timer;
    for (size_t start = begin; start < end && remaining;) {
        const size_t stop = start + std::min(block_capacity, end - start);
        auto models = prepared.empty() ? prepare_block(start,stop,{lower}) : std::move(prepared);
        prepared.clear();
        const size_t resident_start = stop + 1 - models.size();
        if (pipeline) {
            Kokkos::Timer timer; pipeline->fence();
            report_image_timing(timing,"slow_light_upload_ready_wait",timer.seconds());
        }
        // Host models retain ownership of every nested device view until after
        // the block fence; the device descriptor array does not own snapshots.
        for (size_t k = 1; k < models.size(); ++k)
            validate_slow_light_interpolation(models[k - 1], models[k], options.interpolation);
        auto device_models = copy_slow_light_models_to_device<Exec>(models, "slow_resident_models");
        Kokkos::View<Real*, Exec> device_times("slow_resident_times", models.size());
        auto host_times = Kokkos::create_mirror_view(device_times);
        for (size_t k = resident_start; k <= stop; ++k) host_times(k - resident_start) = times[k];
        Kokkos::deep_copy(device_times, host_times);
        using Windows = SlowLightResidentWindows<Exec, Real, RadiationModel>;
        std::vector<SlowLightViews<Exec, Real>> active_images;
        std::vector<Windows> active_windows;
        std::vector<int> offsets, counts;
        std::vector<Real> active_frequencies;
        for (size_t observer = 0; observer < observer_jobs.size(); ++observer) {
            for (size_t f0 = 0; f0 < nfreq; f0 += chunk) {
                const size_t j = observer * nfreq + f0;
                if (done[j] || stop <= jobs[j].first || resident_start >= jobs[j].last) continue;
                const int first = static_cast<int>(std::max(resident_start, jobs[j].first) - resident_start);
                const int last = static_cast<int>(std::min(stop, jobs[j].last) - resident_start);
                const size_t count = std::min(chunk, nfreq - f0);
                offsets.push_back(static_cast<int>(active_images.size()));
                counts.push_back(static_cast<int>(count));
                for (size_t f = 0; f < count; ++f) {
                    active_images.push_back(views[j + f]);
                    active_frequencies.push_back(frequencies.empty() ? lower.freq_cgs : frequencies[f0 + f]);
                }
                active_windows.push_back({device_models, device_times, jobs[j].observation_time, first, last,
                                          options.interpolation, options.step_mode, active_frequencies.back()});
            }
        }
        // One launch spans all participating observation/frequency groups.
        // Snapshot descriptors and their host owners live through the fence.
        using Rays = std::conditional_t<FrequencyCapacity == 1,
            SlowLightRayBatch<Exec, Real, Windows>,
            SlowLightFrequencyRayBatch<Exec, Real, Windows, FrequencyCapacity>>;
        Rays rays{};
        rays.images = copy_slow_light_models_to_device<Exec>(active_images, "slow_batch_images");
        rays.windows = copy_slow_light_models_to_device<Exec>(active_windows, "slow_batch_windows");
        if constexpr (FrequencyCapacity > 1) {
            rays.offsets = copy_slow_light_models_to_device<Exec>(offsets, "slow_frequency_offsets");
            rays.counts = copy_slow_light_models_to_device<Exec>(counts, "slow_frequency_counts");
            rays.frequencies = copy_slow_light_models_to_device<Exec>(active_frequencies, "slow_frequencies");
        }
        struct ComputeFenceOnExit {
            bool pending = true;
            ~ComputeFenceOnExit() { if (pending) Exec{}.fence(); }
        } compute_guard;
        Kokkos::Timer kernel_timer;
        if (pipeline) device_timer.start();
        if (!active_images.empty())
            run_slow_light_rays_metric<Exec>(params, rays, metric, 0, analysis_mode, analysis, false);
        if (pipeline) {
            device_timer.stop();
            if (stop<end) prepared=prepare_block(stop,stop+std::min(block_capacity,end-stop),
                std::vector<RadiationModel>(models.end() - std::min(halo + 1, models.size()), models.end()));
            report_image_timing(timing,"slow_light_pass_b_block",device_timer.seconds());
        } else {
            Kokkos::fence();
            report_image_timing(timing,"slow_light_pass_b_block",kernel_timer.seconds());
        }
        compute_guard.pending = false;
        Kokkos::Timer completion_timer;
        Real earliest_request = std::numeric_limits<Real>::max();
        for (size_t j = 0; j < jobs.size(); ++j) {
            if (done[j] || stop <= jobs[j].first || resident_start >= jobs[j].last) continue;
            const int active = count_active_rays<Exec, Real>(views[j].active);
            if (decoupled && active) {
                const Real resident_first = times[std::max(resident_start, jobs[j].first)];
                const Real resident_last = times[std::min(stop, jobs[j].last)];
                const auto [first_request, last_request] = slow_light_pending_range(views[j], resident_first, resident_last);
                if (first_request < times[jobs[j].first] || last_request > times[jobs[j].last])
                    throw std::runtime_error("decoupled slow-light radiation sample outside supplied physical time support: requested " +
                        std::to_string(first_request) + ".." + std::to_string(last_request) + ", available " +
                        std::to_string(times[jobs[j].first]) + ".." + std::to_string(times[jobs[j].last]));
                earliest_request = std::min(earliest_request, first_request);
            }
            if (!active || (!decoupled && stop >= jobs[j].last)) {
                auto final_model = models[std::min(stop, jobs[j].last) - resident_start];
                if (!frequencies.empty()) final_model.freq_cgs = frequencies[j % nfreq];
                finalize_slow_light_unfinished_metric(params, metric, final_model, views[j], timing, analysis_mode, analysis);
                result[j] = copy_slow_light_views_to_image_host_data(views[j], npix, timing, analysis_mode, analysis);
                views[j] = {};
                done[j] = true;
                --remaining;
            }
        }
        report_image_timing(timing, "slow_light_block_completion", completion_timer.seconds());
        if (timing) std::osyncstream(std::cout) << "timing slow_light_block first " << start << " last " << stop
                              << " remaining_images " << remaining << "\n";
        rays = {};
        active_windows.clear();
        device_models = {};
        lower = models.back();
        if (decoupled && remaining && earliest_request < times[resident_start]) {
            const size_t rewind = std::min(end - 1, static_cast<size_t>(
                std::upper_bound(times.begin(), times.end(), earliest_request) - times.begin() - 1));
            if (pipeline) pipeline->fence();
            prepared.clear();
            models.clear();
            lower = {};
            if (stream) { stream->finish(); stream.reset(); }
            lower = Traits::load(loader, paths[rewind]);
            if (prefetch) stream = std::make_unique<Stream>(loader, paths, rewind + 1, end, capacity);
            if (timing) std::osyncstream(std::cout) << "timing slow_light_cache_rewind from " << start << " to " << rewind << "\n";
            start = rewind;
        } else {
            if (!pipeline && decoupled && remaining && stop < end) {
                const std::vector<RadiationModel> retained_models(
                    models.end() - std::min(halo + 1, models.size()), models.end());
                models.clear();
                prepared = prepare_block(stop, stop + std::min(block_capacity, end - stop),
                    retained_models);
            }
            start = stop;
        }
    }
    if (pipeline) pipeline->fence();
    if (stream) {
        Kokkos::Timer timer;
        stream->finish();
        report_image_timing(timing, "slow_light_prefetch_discard_wait", timer.seconds());
    }
    return result;
}

// All snapshot-backed models and all frequency groups use this scheduler.
// Scalar calls instantiate capacity one even in a large multi-frequency build.
template<class Metric, class RadiationModel, class LoadModel>
std::vector<ImageHostData> run_slow_light_batch_metric(
    const kpolaris::PassAParams<Real>& params,
    const std::vector<std::string>& paths,
    const std::vector<Real>& times,
    const std::vector<SlowLightBatchJob>& jobs,
    LoadModel loader,
    const Metric& metric,
    const SlowLightBatchOptions& options = {},
    const RadiationModel* first_model = nullptr,
    const std::vector<Real>& frequencies = {}) {
#if KPOLARIS_MAX_FREQUENCIES > 1
    if (frequencies.size() > 1 && options.frequency_chunk_size != 1)
        return run_slow_light_batch_metric_impl<KPOLARIS_MAX_FREQUENCIES>(
            params, paths, times, jobs, loader, metric, options, first_model, frequencies);
#endif
    return run_slow_light_batch_metric_impl<1>(
        params, paths, times, jobs, loader, metric, options, first_model, frequencies);
}

} // namespace kpolaris_image_detail
