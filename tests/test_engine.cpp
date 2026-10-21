#define CATCH_CONFIG_MAIN
#include "catch.hpp" // Header-only testing framework
#include "engine.hpp"
#include <cuda_runtime.h>
#include <vector>

// Suite 1: Safety & Architecture Constraints
TEST_CASE("Custom Memory Pool Error Boundaries", "[allocator]") {
    // Initialize a tiny 64-byte arena capacity pool
    size_t tiny_pool_size = 64;
    zero_overhead::Engine engine(tiny_pool_size);

    // Attempting to register a layer whose requirements exceed 64 bytes 
    // should trigger a predictable standard runtime exception instead of an OS crash.
    int BATCH = 1;
    int HIDDEN = 512; // 512 * 1 * sizeof(float) = 2048 bytes -> Massive overflow
    int INPUT = 256;
    
    // Allocate placeholder mock device memory pointers for initialization arguments
    int8_t* d_mock_w = nullptr;
    float* d_mock_b = nullptr;
    cudaMalloc(&d_mock_w, 10);
    cudaMalloc(&d_mock_b, 10);

    REQUIRE_THROWS_AS(
        engine.add_linear_layer(BATCH, HIDDEN, INPUT, 0.001f, true, d_mock_w, d_mock_b),
        std::runtime_error
    );

    cudaFree(d_mock_w);
    cudaFree(d_mock_b);
}

// Suite 2: Core Algorithmic Mathematical Correctness
TEST_CASE("Fused ReLU Logic Clamping Verification", "[kernels]") {
    // Setup minimal dimensions
    const int M = 1, N = 2, K = 2;
    
    // Prepare host data: We configure inputs and weights such that 
    // Element 0 evaluates to a POSITIVE value, and Element 1 evaluates to a NEGATIVE value.
    std::vector<int8_t> h_input = {2, 4}; 
    std::vector<int8_t> h_weight = {
         10, -20, // Row 0
          5, -10  // Row 1
    };
    std::vector<float> h_bias = {0.0f, 0.0f};

    // Allocate GPU Space
    int8_t *d_in, *d_w;
    float *d_b, *d_out;
    cudaMalloc(&d_in,  M * K * sizeof(int8_t));
    cudaMalloc(&d_w,   K * N * sizeof(int8_t));
    cudaMalloc(&d_b,   N * sizeof(float));
    cudaMalloc(&d_out, M * N * sizeof(float));

    cudaMemcpy(d_in, h_input.data(), h_input.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w,  h_weight.data(), h_weight.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,  h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Instantiate engine with large capacity, enabling Fused ReLU activation flag
    zero_overhead::Engine engine(1024);
    engine.add_linear_layer(M, N, K, 1.0f, true, d_w, d_b); // Scale = 1.0f, Fuse ReLU = true

    // Compute execution
    engine.forward(d_in, d_out, 0);
    cudaDeviceSynchronize();

    // Pull results back to CPU for verification assertions
    std::vector<float> h_out(N);
    cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Math check:
    // Accum 0 = (2 * 10) + (4 * 5)  = 40.  ReLU(40)  -> Should remain 40.0f
    // Accum 1 = (2 * -20) + (4 * -10) = -80. ReLU(-80) -> Should clamp to 0.0f
    REQUIRE(h_out[0] == Approx(40.0f));
    REQUIRE(h_out[1] == Approx(0.0f));

    // Cleanup
    cudaFree(d_in); cudaFree(d_w); cudaFree(d_b); cudaFree(d_out);
}
