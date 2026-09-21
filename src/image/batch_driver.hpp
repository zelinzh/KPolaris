#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include "image/driver.hpp"

// Experimental host batch interface: each job uses the ordinary single-frequency
// unsplit kernel. Models/metrics may differ between jobs, but have the same C++
// types. Their device views must already be materialized by the caller.
namespace kpolaris_image_detail {

template<class Metric, class RadiationModel>
struct ImageBatchJob {
    kpolaris::PassAParams<Real> params;
    RadiationModel model;
    Metric metric;
    int radiation_substeps = 1;
};

struct ImageBatchOptions {
    size_t max_in_flight = 4;
    // Output scratch only; caller-owned models and returned host images excluded.
    size_t max_device_output_bytes = size_t(256) * 1024 * 1024;
};

struct ImageBatchTiming {
    double allocate_seconds = 0;
    double kernel_seconds = 0;
    double copy_seconds = 0;
    size_t windows = 0;
    size_t peak_device_output_bytes = 0;
};

inline size_t batch_image_pixels(const kpolaris::PassAParams<Real>& params) {
    const int nx = params.camera.nx;
    const int ny = params.camera.ny;
    if (nx <= 0 || ny <= 0 || nx > std::numeric_limits<int>::max() / ny)
        throw std::invalid_argument("batch image dimensions must have a positive int-sized product");
    return static_cast<size_t>(nx) * static_cast<size_t>(ny);
}

struct BatchImageDeviceData {
    using Exec = Kokkos::DefaultExecutionSpace;
    using RealView = Kokkos::View<Real*, Exec>;
    using IntView = Kokkos::View<int*, Exec>;
    static constexpr size_t bytes_per_pixel = 15 * sizeof(Real) + 3 * sizeof(int);
    int npix;
    RealView image_i;
    RealView image_q;
    RealView image_u;
    RealView image_v;
    RealView closure_x;
    RealView closure_k;
    RealView final_null;
    RealView frame_error;
    RealView det_r;
    RealView overlap_r11;
    RealView overlap_r12;
    RealView overlap_r21;
    RealView overlap_r22;
    RealView basis_identity_error;
    RealView basis_rotation_angle;
    IntView pass_a_steps;
    IntView steps;
    IntView reason;

    explicit BatchImageDeviceData(size_t count) : npix(static_cast<int>(count)),
        image_i("batch_image_i", count),
        image_q("batch_image_q", count),
        image_u("batch_image_u", count),
        image_v("batch_image_v", count),
        closure_x("batch_closure_x", count),
        closure_k("batch_closure_k", count),
        final_null("batch_final_null", count),
        frame_error("batch_frame_error", count),
        det_r("batch_det_r", count),
        overlap_r11("batch_overlap_r11", count),
        overlap_r12("batch_overlap_r12", count),
        overlap_r21("batch_overlap_r21", count),
        overlap_r22("batch_overlap_r22", count),
        basis_identity_error("batch_basis_identity_error", count),
        basis_rotation_angle("batch_basis_rotation_angle", count),
        pass_a_steps("batch_pass_a_steps", count),
        steps("batch_steps", count),
        reason("batch_reason", count) {}

    template<class Metric, class RadiationModel>
    void launch(const ImageBatchJob<Metric, RadiationModel>& job, const Exec& exec) const {
        run_pass_b_segment_model_with_overlap_metric<Exec>(
            job.params, job.model, job.radiation_substeps, job.metric,
            image_i, image_q, image_u, image_v,
            closure_x, closure_k, final_null, frame_error, det_r,
            overlap_r11, overlap_r12, overlap_r21, overlap_r22,
            basis_identity_error, basis_rotation_angle, pass_a_steps, steps, reason, exec);
    }

    ImageHostData copy() const {
        ImageHostData out;
        out.npix = npix;
        out.nfreq = 1;
        out.image_i = copy_real_view_1d(image_i);
        out.image_q = copy_real_view_1d(image_q);
        out.image_u = copy_real_view_1d(image_u);
        out.image_v = copy_real_view_1d(image_v);
        out.closure_x = copy_real_view_1d(closure_x);
        out.closure_k = copy_real_view_1d(closure_k);
        out.final_null = copy_real_view_1d(final_null);
        out.frame_error = copy_real_view_1d(frame_error);
        out.det_r = copy_real_view_1d(det_r);
        out.overlap_r11 = copy_real_view_1d(overlap_r11);
        out.overlap_r12 = copy_real_view_1d(overlap_r12);
        out.overlap_r21 = copy_real_view_1d(overlap_r21);
        out.overlap_r22 = copy_real_view_1d(overlap_r22);
        out.basis_identity_error = copy_real_view_1d(basis_identity_error);
        out.basis_rotation_angle = copy_real_view_1d(basis_rotation_angle);
        out.pass_a_steps = copy_int_view_1d(pass_a_steps);
        out.steps = copy_int_view_1d(steps);
        out.reason = copy_int_view_1d(reason);
        return out;
    }
};

template<class Metric, class RadiationModel>
std::vector<ImageHostData> run_image_batch_metric(
    const std::vector<ImageBatchJob<Metric, RadiationModel>>& jobs,
    const ImageBatchOptions& options = {}, ImageBatchTiming* timing = nullptr) {
    using Exec = Kokkos::DefaultExecutionSpace;
    if (options.max_in_flight == 0 || options.max_device_output_bytes == 0)
        throw std::invalid_argument("batch limits must be positive");
    if (timing) *timing = {};
    std::vector<size_t> pixels;
    pixels.reserve(jobs.size());
    // Validate every job before launching any work.
    for (const auto& job : jobs) {
        const size_t n = batch_image_pixels(job.params);
        if (n > options.max_device_output_bytes / BatchImageDeviceData::bytes_per_pixel)
            throw std::invalid_argument("one batch image exceeds the device output budget");
        if (job.radiation_substeps < 1)
            throw std::invalid_argument("batch radiation_substeps must be positive");
        pixels.push_back(n);
    }
    if (jobs.empty()) return {};
    const size_t slots = std::min(options.max_in_flight, jobs.size());
    const auto instances = Kokkos::Experimental::partition_space(Exec(), std::vector<int>(slots, 1));
    std::vector<ImageHostData> outputs;
    outputs.reserve(jobs.size());
    for (size_t begin = 0; begin < jobs.size();) {
        Kokkos::Timer timer;
        std::vector<BatchImageDeviceData> device;
        device.reserve(slots);
        size_t bytes = 0;
        while (device.size() < slots && begin + device.size() < jobs.size()) {
            const size_t n = pixels[begin + device.size()];
            const size_t next_bytes = n * BatchImageDeviceData::bytes_per_pixel;
            if (next_bytes > options.max_device_output_bytes - bytes) break;
            // Allocate the entire window before submission: View initialization
            // and allocation may synchronize otherwise independent CUDA streams.
            device.emplace_back(n);
            bytes += next_bytes;
        }
        if (timing) {
            timing->allocate_seconds += timer.seconds();
            ++timing->windows;
            timing->peak_device_output_bytes = std::max(timing->peak_device_output_bytes, bytes);
        }
        timer.reset();
        try {
            for (size_t i = 0; i < device.size(); ++i)
                device[i].launch(jobs[begin + i], instances[i]);
            for (size_t i = 0; i < device.size(); ++i)
                instances[i].fence("KPolaris batch image complete");
        } catch (...) {
            // Keep captured buffers alive until already submitted work completes.
            for (size_t i = 0; i < device.size(); ++i) instances[i].fence();
            throw;
        }
        if (timing) timing->kernel_seconds += timer.seconds();
        timer.reset();
        for (const auto& data : device) outputs.push_back(data.copy());
        if (timing) timing->copy_seconds += timer.seconds();
        begin += device.size();
    }
    return outputs;
}

} // namespace kpolaris_image_detail
