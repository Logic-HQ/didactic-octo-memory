#include "allocator.hpp"
#include <iostream>

namespace zero_overhead {

CudaScratchpadAllocator::CudaScratchpadAllocator(size_t total_capacity)
    : capacity_(total_capacity), offset_(0) {
    
    // Allocate the entire pool monolithically upfront
    cudaError_t err = cudaMalloc(&device_memory_ptr_, capacity_);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate CUDA scratchpad pool of size " + 
                                 std::to_string(capacity_) + " bytes: " + cudaGetErrorString(err));
    }
}

CudaScratchpadAllocator::~CudaScratchpadAllocator() {
    if (device_memory_ptr_) {
        cudaFree(device_memory_ptr_);
        device_memory_ptr_ = nullptr;
    }
}

CudaScratchpadAllocator::CudaScratchpadAllocator(CudaScratchpadAllocator&& other) noexcept {
    device_memory_ptr_ = other.device_memory_ptr_;
    capacity_          = other.capacity_;
    offset_            = other.offset_;

    other.device_memory_ptr_ = nullptr;
    other.capacity_          = 0;
    other.offset_            = 0;
}

CudaScratchpadAllocator& CudaScratchpadAllocator::operator=(CudaScratchpadAllocator&& other) noexcept {
    if (this != &other) {
        if (device_memory_ptr_) {
            cudaFree(device_memory_ptr_);
        }
        device_memory_ptr_ = other.device_memory_ptr_;
        capacity_          = other.capacity_;
        offset_            = other.offset_;

        other.device_memory_ptr_ = nullptr;
        other.capacity_          = 0;
        other.offset_            = 0;
    }
    return *this;
}

void* CudaScratchpadAllocator::allocate(size_t bytes, size_t alignment) {
    // Calculate alignment adjustment padding
    uintptr_t current_address = reinterpret_cast<uintptr_t>(device_memory_ptr_) + offset_;
    size_t padding = (alignment - (current_address % alignment)) % alignment;
    
    size_t total_needed = bytes + padding;

    if (offset_ + total_needed > capacity_) {
        throw std::runtime_error("CudaScratchpadAllocator Out of Memory! Requested: " + 
                                 std::to_string(total_needed) + " bytes, Free: " + 
                                 std::to_string(get_free_bytes()) + " bytes.");
    }

    void* allocated_ptr = reinterpret_cast<void*>(current_address + padding);
    offset_ += total_needed;
    
    return allocated_ptr;
}

void CudaScratchpadAllocator::reset() {
    // O(1) adjustment: We simply move the pointer index back to zero.
    // No actual hardware deallocation happens, keeping latency at absolute zero.
    offset_ = 0;
}

} // namespace zero_overhead
