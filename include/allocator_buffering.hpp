#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace zero_overhead {

class CudaScratchpadAllocator {
public:
    explicit CudaScratchpadAllocator(size_t max_layer_bytes);
    ~CudaScratchpadAllocator();

    // Prevent copying
    CudaScratchpadAllocator(const CudaScratchpadAllocator&) = delete;
    CudaScratchpadAllocator& operator=(const CudaScratchpadAllocator&) = delete;

    // Move semantics
    CudaScratchpadAllocator(CudaScratchpadAllocator&& other) noexcept;
    CudaScratchpadAllocator& operator=(CudaScratchpadAllocator&& other) noexcept;

    /**
     * @brief Alternates the execution frames between Buffer A and Buffer B.
     * @return A raw device pointer to the designated workspace tensor layer.
     */
    void* get_current_ping_buffer() const;
    void* get_current_pong_buffer() const;
    
    /**
     * @brief Swaps the active roles of Buffer A and Buffer B.
     * O(1) latency overhead—simply toggles an internal boolean state flag.
     */
    void swap_buffers();

    // Reset the toggle and parameters back to initial states
    void reset();

    size_t get_buffer_capacity() const { return buffer_capacity_; }

private:
    void*  device_buffer_a_  = nullptr;
    void*  device_buffer_b_  = nullptr;
    size_t buffer_capacity_  = 0;
    bool   is_swapped_       = false; // Tracks whether Buffer A or B is current source
};

} // namespace zero_overhead
