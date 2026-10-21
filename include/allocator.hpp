#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace zero_overhead {

class CudaScratchpadAllocator {
public:
    // Initialize the pool with a fixed capacity in bytes
    explicit CudaScratchpadAllocator(size_t total_capacity);
    ~CudaScratchpadAllocator();

    // Prevent copying to avoid accidental double-frees of GPU memory
    CudaScratchpadAllocator(const CudaScratchpadAllocator&) = delete;
    CudaScratchpadAllocator& operator=(const CudaScratchpadAllocator&) = delete;

    // Move semantics allowed
    CudaScratchpadAllocator(CudaScratchpadAllocator&& other) noexcept;
    CudaScratchpadAllocator& operator=(CudaScratchpadAllocator&& other) noexcept;

    // Allocates a chunk of memory from the pool with explicit alignment
    void* allocate(size_t bytes, size_t alignment = 256);

    // Resets the offset back to 0. Call this at the start of every inference pass!
    void reset();

    // Diagnostic queries
    size_t get_total_capacity() const { return capacity_; }
    size_t get_allocated_bytes() const { return offset_; }
    size_t get_free_bytes() const { return capacity_ - offset_; }

private:
    void*  device_memory_ptr_ = nullptr;
    size_t capacity_          = 0;
    size_t offset_            = 0;
};

} // namespace zero_overhead
