#include "engine.hpp"

namespace zero_overhead {

Engine::Engine(size_t max_layer_bytes)
    : allocator_(max_layer_bytes), scheduler_(2) {}

void Engine::add_linear_layer(int M, int N, int K, float alpha_scale, bool fuse_relu,
                              const int8_t* d_weights, const float* d_bias) {
    // Make sure our allocated ping-pong space is large enough to host this layer's output size
    size_t output_bytes = M * N * sizeof(float);
    if (output_bytes > allocator_.get_buffer_capacity()) {
        throw std::runtime_error("Layer output exceeds static Ping-Pong arena buffer capacity!");
    }
    
    layers_.push_back({M, N, K, alpha_scale, fuse_relu, d_weights, d_bias});
}

float Engine::forward(const int8_t* d_input, float* d_output, size_t target_stream_idx) {
    cudaStream_t stream = scheduler_.get_stream(target_stream_idx);
    scheduler_.record_start(target_stream_idx);
    allocator_.reset();

    // Loop through our layers utilizing the ping-pong design
    for (size_t i = 0; i < layers_.size(); ++i) {
        const auto& layer = layers_[i];
        bool is_first_layer = (i == 0);
        bool is_last_layer  = (i == layers_.size() - 1);

        // Deterministically resolve where to read from
        const int8_t* layer_input = is_first_layer 
            ? d_input 
            : reinterpret_cast<const int8_t*>(allocator_.get_current_ping_buffer());

        // Deterministically resolve where to write to
        float* layer_output = is_last_layer 
            ? d_output 
            : reinterpret_cast<float*>(allocator_.get_current_pong_buffer());

        // Execute kernel
        launch_int8_gemm(
            layer_input, 
            layer.d_weights, 
            layer.d_bias, 
            layer_output, 
            layer.M, layer.N, layer.K, 
            layer.alpha_scale, 
            layer.fuse_relu, 
            stream
        );

        // If we have more layers to process, flip the active pointers for the next iteration step
        if (!is_last_layer) {
            allocator_.swap_buffers();
        }
    }

    scheduler_.record_stop(target_stream_idx);
    return scheduler_.get_elapsed_time_ms();
}

} // namespace zero_overhead
