#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <string>

namespace zero_overhead {

class CudaStreamScheduler {
public:
    // Initializes the engine scheduler with a pool of non-blocking hardware streams
    explicit CudaStreamScheduler(size_t stream_count = 2);
    ~CudaStreamScheduler();

    // Disable copy semantics to prevent hardware handle corruption
    CudaStreamScheduler(const CudaStreamScheduler&) = delete;
    CudaStreamScheduler& operator=(const CudaStreamScheduler&) = delete;

    // Allow moving resources safely
    CudaStreamScheduler(CudaStreamScheduler&& other) noexcept;
    CudaStreamScheduler& operator=(CudaStreamScheduler&& other) noexcept;

    // Stream and Queue Management
    cudaStream_t get_stream(size_t index) const;
    void synchronize_stream(size_t index);
    void synchronize_all();

    // Low-Latency Event-Based Hardware Profiling
    void record_start(size_t stream_index);
    void record_stop(size_t stream_index);
    
    // Returns the exact execution time in milliseconds elapsed on the GPU hardware
    float get_elapsed_time_ms();

private:
    std::vector<cudaStream_t> streams_;
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_  = nullptr;
};

} // namespace zero_overhead
