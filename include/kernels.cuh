#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace zero_overhead {

/**
 * @brief Computes Custom Quantized Linear Layer (INT8 Matrix Multiplication)
 * Formula: Output_fp32[i] = (Alpha_Scale * (A_int8 * B_int8)[i]) + Bias_fp32[i]
 * 
 * @param A             Device pointer to Input Matrix (M x K), row-major, INT8
 * @param B             Device pointer to Weight Matrix (K x N), column-major (or tiled), INT8
 * @param bias          Device pointer to Bias Vector (N), FP32
 * @param C             Device pointer to Output Matrix (M x N), row-major, FP32
 * @param M             Rows of Matrix A
 * @param N             Columns of Matrix B / Outputs
 * @param K             Columns of Matrix A / Rows of Matrix B
 * @param alpha_scale   De-quantization multiplier (Scale_A * Scale_B)
 * @param stream        CUDA Stream handle for non-blocking execution
 */
void launch_int8_gemm(const int8_t* A, const int8_t* B, const float* bias, float* C,
                      int M, int N, int K, float alpha_scale, cudaStream_t stream);

} // namespace zero_overhead
