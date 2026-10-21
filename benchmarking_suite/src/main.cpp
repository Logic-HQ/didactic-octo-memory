#include "engine.hpp"
#include <iostream>
#include <vector>
#include <random>
#include <fstream>   // Added for file I/O operations
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "  ZeroOverhead-Inference: High-Precision CSV Telemetry  \n";
    std::cout << "========================================================\n\n";

    const int BATCH_SIZE = 1;
    const int INPUT_DIM  = 512;
    const int HIDDEN_DIM = 1024;
    const int OUTPUT_DIM = 10;

    // Allocate host memory
    std::vector<int8_t> h_input(BATCH_SIZE * INPUT_DIM, 2);
    std::vector<int8_t> h_w1(INPUT_DIM * HIDDEN_DIM);
    std::vector<int8_t> h_w2(HIDDEN_DIM * OUTPUT_DIM);
    std::vector<float>  h_b1(HIDDEN_DIM, 0.5f);
    std::vector<float>  h_b2(OUTPUT_DIM, -0.1f);

    std::mt19937 rng(1337); 
    std::uniform_int_distribution<int> dist(-127, 127);
    for (auto& w : h_w1) w = static_cast<int8_t>(dist(rng));
    for (auto& w : h_w2) w = static_cast<int8_t>(dist(rng));

    // Allocate Device VRAM
    int8_t *d_input = nullptr, *d_w1 = nullptr, *d_w2 = nullptr;
    float  *d_b1 = nullptr, *d_b2 = nullptr, *d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input,  BATCH_SIZE * INPUT_DIM * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_w1,     INPUT_DIM * HIDDEN_DIM * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_w2,     HIDDEN_DIM * OUTPUT_DIM * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_b1,     HIDDEN_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2,     OUTPUT_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_DIM * sizeof(float)));

    // Copy to Hardware Device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1,    h_w1.data(),    h_w1.size() * sizeof(int8_t),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2,    h_w2.data(),    h_w2.size() * sizeof(int8_t),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1,    h_b1.data(),    h_b1.size() * sizeof(float),     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2,    h_b2.data(),    h_b2.size() * sizeof(float),     cudaMemcpyHostToDevice));

    // Initialize Engine Runtime
    size_t max_layer_bytes = HIDDEN_DIM * BATCH_SIZE * sizeof(float); 
    zero_overhead::Engine engine(max_layer_bytes);

    engine.add_linear_layer(BATCH_SIZE, HIDDEN_DIM, INPUT_DIM, 0.001f, true, d_w1, d_b1);
    engine.add_linear_layer(BATCH_SIZE, OUTPUT_DIM, HIDDEN_DIM, 0.002f, false, d_w2, d_b2);

    // Warm-up pass
    engine.forward(d_input, d_output, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Benchmark Execution Loop ---
    const int BENCHMARK_RUNS = 1000;
    
    // Pre-allocating memory for telemetry storage avoids dynamic allocations during timing!
    std::vector<float> run_latencies_ms(BENCHMARK_RUNS, 0.0f);

    std::cout << "[Processing] Running " << BENCHMARK_RUNS << " inference iterations...\n";
    for (int i = 0; i < BENCHMARK_RUNS; ++i) {
        // Collect raw hardware events elapsed time directly
        run_latencies_ms[i] = engine.forward(d_input, d_output, 0);
    }
    std::cout << "[Success] Telemetry capture complete.\n\n";

    // --- CSV Generation Layer ---
    std::string csv_filename = "inference_benchmarks.csv";
    std::ofstream csv_file(csv_filename);

    if (csv_file.is_open()) {
        auto meta = engine.get_metadata();
        
        // Write CSV Metadata Header Line
        csv_file << "iteration,latency_ms,latency_us,total_layers,scratchpad_capacity_bytes\n";
        
        // Flush structural runs
        for (int i = 0; i < BENCHMARK_RUNS; ++i) {
            float ms = run_latencies_ms[i];
            float us = ms * 1000.0f;
            csv_file << i << "," 
                     << ms << "," 
                     << us << "," 
                     << meta.total_layers << "," 
                     << meta.scratchpad_capacity_bytes << "\n";
        }
        csv_file.close();
        std::cout << "💾 Benchmark metrics exported successfully to: " << csv_filename << "\n";
    } else {
        std::cerr << "❌ Error: Could not generate benchmark metrics CSV file.\n";
    }

    // Cleanup Resources
    cudaFree(d_input); cudaFree(d_w1); cudaFree(d_w2);
    cudaFree(d_b1);    cudaFree(d_b2);    cudaFree(d_output);

    return 0;
}
