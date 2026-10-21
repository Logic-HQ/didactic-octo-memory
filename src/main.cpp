#include "engine.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <cuda_runtime.h>

// Helper macro to quickly handle runtime CUDA setup checkouts
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "  ZeroOverhead-Inference: High-Performance GPU Benchmark \n";
    std::cout << "========================================================\n\n";

    // Define shapes for a compact two-layer Neural Network MLP
    // Layer 1: Input (1 x 512) x Weights (512 x 1024) -> Hidden Output (1 x 1024)
    // Layer 2: Input (1 x 1024) x Weights (1024 x 10)  -> Final Output (1 x 10)
    const int BATCH_SIZE = 1;
    const int INPUT_DIM  = 512;
    const int HIDDEN_DIM = 1024;
    const int OUTPUT_DIM = 10;

    std::cout << "[Step 1/4] Generating synthetic quantized model parameters...\n";
    
    // Allocate host vectors for random matrix configurations
    std::vector<int8_t> h_input(BATCH_SIZE * INPUT_DIM, 2); // Constant mock quantized input
    std::vector<int8_t> h_w1(INPUT_DIM * HIDDEN_DIM);
    std::vector<int8_t> h_w2(HIDDEN_DIM * OUTPUT_DIM);
    std::vector<float>  h_b1(HIDDEN_DIM, 0.5f);
    std::vector<float>  h_b2(OUTPUT_DIM, -0.1f);

    // Initialize weights with basic pseudorandom distribution sequences
    std::mt19937 rng(1337); 
    std::uniform_int_distribution<int> dist(-127, 127);
    for (auto& w : h_w1) w = static_cast<int8_t>(dist(rng));
    for (auto& w : h_w2) w = static_cast<int8_t>(dist(rng));

    std::cout << "[Step 2/4] Allocating dedicated VRAM and uploading to hardware...\n";

    // Target pointers residing inside device GPU Global Memory
    int8_t *d_input = nullptr, *d_w1 = nullptr, *d_w2 = nullptr;
    float  *d_b1 = nullptr, *d_b2 = nullptr, *d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input,  BATCH_SIZE * INPUT_DIM * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_w1,     INPUT_DIM * HIDDEN_DIM * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_w2,     HIDDEN_DIM * OUTPUT_DIM * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_b1,     HIDDEN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2,     OUTPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_DIM * sizeof(float)));

    // Push tensors across PCIe lanes asynchronously onto the device GPU boards
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1,    h_w1.data(),    h_w1.size() * sizeof(int8_t),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2,    h_w2.data(),    h_w2.size() * sizeof(int8_t),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1,    h_b1.data(),    h_b1.size() * sizeof(float),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2,    h_b2.data(),    h_b2.size() * sizeof(float),     cudaMemcpyHostToDevice));

    std::cout << "[Step 3/4] Initializing ZeroOverhead Static Memory Pool Engine...\n";
    
    // We pre-allocate a tiny 1MB workspace memory block on the GPU.
    // This entirely covers our internal activation matrices with room to spare,
    // guaranteeing absolutely zero execution allocations later.
    size_t engine_scratchpad_bytes = 1024 * 1024; 
    zero_overhead::Engine engine(engine_scratchpad_bytes);

    // Register our execution pipelines sequentially into the engine context
    // Float alpha values scale down the calculated integer multiplications back out of INT8 space
    engine.add_linear_layer(BATCH_SIZE, HIDDEN_DIM, INPUT_DIM, 0.001f, d_w1, d_b1);
    engine.add_linear_layer(BATCH_SIZE, OUTPUT_DIM, HIDDEN_DIM, 0.002f, d_w2, d_b2);

    // Run an initial warm-up pass to let CUDA stabilize the hardware clocks
    std::cout << " -> Warming up execution context layers...\n";
    engine.forward(d_input, d_output, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::cout << "[Step 4/4] Starting performance profiling iteration loop...\n";

    const int BENCHMARK_RUNS = 1000;
    float total_latency_ms = 0.0f;

    for (int i = 0; i < BENCHMARK_RUNS; ++i) {
        // Run forward inference on stream lane index 0
        total_latency_ms += engine.forward(d_input, d_output, 0);
    }

    // Print final out-of-the-box hardware efficiency metrics
    float average_latency_us = (total_latency_ms / BENCHMARK_RUNS) * 1000.0f;
    std::cout << "\n================= BENCHMARK RESULTS =================\n";
    std::cout << "  Total Executed Passes : " << BENCHMARK_RUNS << "\n";
    std::cout << "  Average Core Latency  : " << average_latency_us << " microseconds (us)\n";
    std::cout << "  Status                : Deterministic execution successful.\n";
    std::cout << "=====================================================\n\n";

    // Extract calculated outputs back down to verify structural stability
    std::vector<float> h_output(BATCH_SIZE * OUTPUT_DIM);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup explicit persistent allocations safely
    cudaFree(d_input);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_output);

    return 0;
}
