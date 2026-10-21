#pragma once

#include "allocator.hpp"
#include "scheduler.hpp"
#include "kernels.cuh"
#include <memory>
#include <vector>

namespace zero_overhead {

struct LinearLayerParams {
    int M;
    int N;
    int K;
    float alpha_scale;
    const int8_t* d_weights; // Pre-loaded device weights
    const float* d_bias;     // Pre-loaded device bias
};

class Engine {
public:
    /**
     * @brief Constructor for the Engine
     * @param scratchpad_capacity Total memory pool pre-allocated for mid-layer tensors (bytes)
     */
    explicit Engine(size_t scratchpad_capacity);
    ~Engine() = default;

    // Disable copy semantics to prevent hardware descriptor collisions
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /**
     * @brief Registers a linear layer structure to execute in sequence
     */
    void add_linear_layer(int M, int N, int K, float alpha_scale, 
                          const int8_t* d_weights, const float* d_bias);

    /**
     * @brief Runs an inference pass completely asynchronously on a CUDA Stream
     * @param d_input Device pointer to initial network INT8 input
     * @param d_output Device pointer to final network FP32 output destination
     * @param target_stream_idx Index of the hardware execution stream queue to use
     * @return Execution latency in milliseconds tracked directly via GPU hardware events
     */
    float forward(const int8_t* d_input, float* d_output, size_t target_stream_idx = 0);

private:
    CudaScratchpadAllocator allocator_;
    CudaStreamScheduler scheduler_;
    std::vector<LinearLayerParams> layers_;
};

} // namespace zero_overhead
