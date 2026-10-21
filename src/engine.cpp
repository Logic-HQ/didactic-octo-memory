#include "engine.hpp"
#include <stdexcept>

namespace zero_overhead {

Engine::Engine(size_t scratchpad_capacity)
    : allocator_(scratchpad_capacity), scheduler_(2) {
    // Initializes pool allocator and configures two independent hardware command queues
}

void Engine::add_linear_layer(int M, int N, int K, float alpha_scale, 
                              const int8_t* d_weights, const float* d_bias) {
    if (d_weights == nullptr) {
        throw std::invalid_argument("Layer weights device pointer cannot be null.");
    }
    layers_.push_back({M, N, K, alpha_scale, d_weights, d_bias});
}

float Engine::forward(const int8_t* d_input, float* d_output, size_t target_stream_idx) {
    if (layers_.empty()) {
        throw std::runtime_error("Cannot execute forward pass; no layers registered.");
    }

    cudaStream_t stream = scheduler_.get_stream(target_stream_idx);

    // 1. Start hardware-level profiling clock without blocking the CPU thread
    scheduler_.record_start(target_stream_idx);

    // 2. Reset the scratchpad allocator instantly. O(1) latency overhead.
    allocator_.reset();

    const int8_t* current_input = d_input;

    // 3. Process registered layers sequentially
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto& layer = layers_[i];
        bool is_last_layer = (i == layers_.size() - 1);

        float* layer_output = nullptr;

        if (is_last_layer) {
            // Write directly to the provided final application target memory buffer
            layer_output = d_output;
        } else {
            // Allocate internal layer activation buffers from the pre-allocated pool.
            // Requires allocating M * N floats. For deep networks, we'd quantize this 
            // back to INT8, but for a hybrid runtime, we allocate temporary FP32 space.
            size_t bytes_needed = layer.M * layer.N * sizeof(float);
            layer_output = static_cast<float*>(allocator_.allocate(bytes_needed));
        }

        // 4. Dispatch the execution token completely asynchronously to the GPU hardware lane
        launch_int8_gemm(
            current_input, 
            layer.d_weights, 
            layer.d_bias, 
            layer.layer_output, 
            layer.M, layer.N, layer.K, 
            layer.alpha_scale, 
            stream
        );

        // If this weren't the final layer and we had an activation function (e.g. ReLU), 
        // it would be fused here or called next. For now, we update the input tracker.
        // NOTE: In a pure INT8 multi-layer network, you would feed a quantized INT8 pointer.
        // For demonstration, each layer evaluates to an optimized FP32 matrix block.
        current_input = reinterpret_cast<const int8_t*>(layer_output); 
    }

    // 5. Enqueue the stop hardware counter timestamp
    scheduler_.record_stop(target_stream_idx);

    // 6. Block the host thread explicitly here *only* to extract the execution metrics
    return scheduler_.get_elapsed_time_ms();
}

} // namespace zero_overhead
