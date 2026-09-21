#pragma once

#include <condition_variable>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include "image/slow_light_driver.hpp"

namespace kpolaris_image_detail {

// A bounded stream of host-staged dumps. The worker never materializes a model
// or touches Kokkos; the caller uploads each item after consuming it in order.
// Capacity counts queued dumps; one additional read may be in flight.
template<class Traits, class LoadModel>
class SlowLightHostStream {
public:
    using Staged = typename Traits::PrefetchResult;
    SlowLightHostStream(LoadModel loader, const std::vector<std::string>& paths,
                        size_t first, size_t last, size_t capacity)
        : capacity_(std::max(size_t(1), capacity)), next_(first) {
        worker_ = std::thread([this, loader, &paths, first, last]() mutable {
            try {
                for (size_t k = first; k <= last; ++k) {
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        changed_.wait(lock, [&] { return stop_ || queue_.size() < capacity_; });
                        if (stop_) return;
                    }
                    auto staged = Traits::prefetch(loader, paths[k]);
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        if (stop_) return;
                        queue_.push_back(std::move(staged));
                    }
                    changed_.notify_all();
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(mutex_);
                error_ = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                finished_ = true;
            }
            changed_.notify_all();
        });
    }
    SlowLightHostStream(const SlowLightHostStream&) = delete;
    SlowLightHostStream& operator=(const SlowLightHostStream&) = delete;
    ~SlowLightHostStream() { stop(); }

    Staged get(size_t index) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (index != next_) throw std::logic_error("slow-light host stream consumed out of order");
        changed_.wait(lock, [&] { return !queue_.empty() || finished_ || stop_; });
        if (queue_.empty()) {
            if (error_) std::rethrow_exception(error_);
            throw std::runtime_error("slow-light host stream ended before the requested snapshot");
        }
        auto staged = std::move(queue_.front());
        queue_.pop_front();
        ++next_;
        lock.unlock();
        changed_.notify_all();
        return staged;
    }
    void finish() {
        stop();
        if (error_) std::rethrow_exception(error_);
    }
private:
    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        changed_.notify_all();
        if (worker_.joinable()) worker_.join();
    }
    size_t capacity_, next_;
    std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<Staged> queue_;
    std::exception_ptr error_;
    bool stop_ = false, finished_ = false;
    std::thread worker_;
};

} // namespace kpolaris_image_detail
