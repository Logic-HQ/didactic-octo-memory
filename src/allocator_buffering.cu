#include "allocator.hpp"
#include <iostream>

namespace zero_overhead {

CudaScratchpadAllocator::CudaScratchpadAllocator(size_t max_layer_bytes)
    : buffer_capacity_(max_layer_bytes), is_swapped_(false) {
    
    // Allocate two separate monolithic blocks upfront matching our max expected layer sizing
    cudaError_t err_a = cudaMalloc(&device_buffer_a_, buffer_capacity_);
    cudaError_t err_b = cudaMalloc(&device_buffer_b_, buffer_capacity_);
    
    if (err_a != cudaSuccess || err_b != cudaSuccess) {
        throw std::runtime_error("Failed to allocate static ping-pong buffers of size " + 
                                 std::to_string(buffer_capacity_) + " bytes.");
    }
}

CudaScratchpadAllocator::~CudaScratchpadAllocator() {
    if (device_buffer_a_) cudaFree(device_buffer_a_);
    if (device_buffer_b_) cudaFree(device_buffer_b_);
}

CudaScratchpadAllocator::CudaScratchpadAllocator(CudaScratchpadAllocator&& other) noexcept {
    device_buffer_a_ = other.device_buffer_a_;
    device_buffer_b_ = other.device_buffer_b_;
    buffer_capacity_ = other.buffer_capacity_;
    is_swapped_      = other.is_swapped_;

    other.device_buffer_a_ = nullptr;
    other.device_buffer_b_ = nullptr;
}

CudaScratchpadAllocator& CudaScratchpadAllocator::operator=(CudaScratchpadAllocator&& other) noexcept {
    if (this != &other) {
        if (device_buffer_a_) cudaFree(device_buffer_a_);
        if (device_buffer_b_) cudaFree(device_buffer_b_);

        device_buffer_a_ = other.device_buffer_a_;
        device_buffer_b_ = other.device_buffer_b_;
        buffer_capacity_ = other.buffer_capacity_;
        is_swapped_      = other.is_swapped_;

        other.device_buffer_a_ = nullptr;
        other.device_buffer_b_ = nullptr;
    }
    return *this;
}

void* CudaScratchpadAllocator::get_current_ping_buffer() const {
    return is_swapped_ ? device_buffer_b_ : device_buffer_a_;
}

void* CudaScratchpadAllocator::get_current_pong_buffer() const {
    return is_swapped_ ? device_buffer_a_ : device_buffer_b_;
}

void CudaScratchpadAllocator::swap_buffers() {
    // Pure bitwise/boolean inversion. Completely free execution.
    is_swapped_ = !is_swapped_;
}

void CudaScratchpadAllocator::reset() {
    is_swapped_ = false;
}

} // namespace zero_overhead
