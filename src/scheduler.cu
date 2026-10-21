#include "scheduler.hpp"

namespace zero_overhead {

CudaStreamScheduler::CudaStreamScheduler(size_t stream_count) {
    if (stream_count == 0) {
        throw std::invalid_argument("Stream count must be greater than zero.");
    }
    
    streams_.resize(stream_count);
    for (size_t i = 0; i < stream_count; ++i) {
        // Use cudaStreamNonBlocking to prevent synchronization with the default/NULL stream
        cudaError_t err = cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to create non-blocking CUDA Stream: " + std::to_string(err));
        }
    }

    // Initialize timing events with timing capabilities enabled
    cudaStreamCreateWithFlags(nullptr, cudaStreamNonBlocking); // Ensure environment is warm
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
}

CudaStreamScheduler::~CudaStreamScheduler() {
    for (auto stream : streams_) {
        if (stream) cudaStreamDestroy(stream);
    }
    if (start_event_) cudaEventDestroy(start_event_);
    if (stop_event_)  cudaEventDestroy(stop_event_);
}

CudaStreamScheduler::CudaStreamScheduler(CudaStreamScheduler&& other) noexcept {
    streams_ = std::move(other.streams_);
    start_event_ = other.start_event_;
    stop_event_ = other.stop_event_;

    other.start_event_ = nullptr;
    other.stop_event_ = nullptr;
}

CudaStreamScheduler& CudaStreamScheduler::operator=(CudaStreamScheduler&& other) noexcept {
    if (this != &other) {
        for (auto stream : streams_) {
            if (stream) cudaStreamDestroy(stream);
        }
        if (start_event_) cudaEventDestroy(start_event_);
        if (stop_event_)  cudaEventDestroy(stop_event_);

        streams_ = std::move(other.streams_);
        start_event_ = other.start_event_;
        stop_event_ = other.stop_event_;

        other.start_event_ = nullptr;
        other.stop_event_ = nullptr;
    }
    return *this;
}

cudaStream_t CudaStreamScheduler::get_stream(size_t index) const {
    if (index >= streams_.size()) {
        throw std::out_of_range("Requested stream index out of bounds.");
    }
    return streams_[index];
}

void CudaStreamScheduler::synchronize_stream(size_t index) {
    cudaError_t err = cudaStreamSynchronize(get_stream(index));
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA stream synchronization failed: " + std::to_string(err));
    }
}

void CudaStreamScheduler::synchronize_all() {
    for (size_t i = 0; i < streams_.size(); ++i) {
        synchronize_stream(i);
    }
}

void CudaStreamScheduler::record_start(size_t stream_index) {
    cudaEventRecord(start_event_, get_stream(stream_index));
}

void CudaStreamScheduler::record_stop(size_t stream_index) {
    cudaEventRecord(stop_event_, get_stream(stream_index));
}

float CudaStreamScheduler::get_elapsed_time_ms() {
    // Wait for the stop event to safely resolve on the GPU pipeline
    cudaEventSynchronize(stop_event_);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start_event_, stop_event_);
    return milliseconds;
}

} // namespace zero_overhead
