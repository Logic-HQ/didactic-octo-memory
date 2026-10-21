void Engine::add_linear_layer(int M, int N, int K, float alpha_scale, bool fuse_relu,
                              const int8_t* d_weights, const float* d_bias) {
    if (d_weights == nullptr) {
        throw std::invalid_argument("Layer weights device pointer cannot be null.");
    }
    layers_.push_back({M, N, K, alpha_scale, fuse_relu, d_weights, d_bias});
}

// Inside Engine::forward loop, update the launch function call:
launch_int8_gemm(
    current_input, 
    layer.d_weights, 
    layer.d_bias, 
    layer_output, 
    layer.M, layer.N, layer.K, 
    layer.alpha_scale, 
    layer.fuse_relu, // <-- Pass the fused activation directive to the kernel
    stream
);
