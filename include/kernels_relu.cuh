#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace zero_overhead {

/**
 * @brief Computes Custom Quantized Linear Layer with optional Fused ReLU Activation
 * Formula: Output = fuse_relu ? max(0.0f, (alpha * (A * B)) + bias) : ((alpha * (A * B)) + bias)
 */
void launch_int8_gemm(const int8_t* A, const int8_t* B, const float* bias, float* C,
                      int M, int N, int K, float alpha_scale, bool fuse_relu, cudaStream_t stream);

} // namespace zero_overhead
